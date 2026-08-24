from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill import exit_maker_report as _exit_report
from maker.src.quote_fill import filled_entry_report as _filled_entry_report
from maker.src.quote_fill.filled_entry_report import (
    FilledEntryPrimaryReport,
    _report_implementation_sources,
)
from maker.src.quote_fill.post_cross_position_evaluator import (
    COST_COMPONENT_COLUMNS,
    COST_PROFILE_SCHEMA_VERSION,
    CostSensitivity,
    FormalPostCrossSources,
    PositionLimit,
    PostCrossEvaluationConfig,
    PHYSICAL_PATH_KEY,
    POLICY_KEY,
    _ARTIFACTS,
    _EXIT_ARTIFACT_ROW_COLUMNS,
    _EXIT_MAKER_MANIFEST_SCHEMA,
    _ENTRY_ZERO_ACTION_SCHEMA,
    _ENTRY_ZERO_EXIT_RULE_SCHEMA,
    _ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA,
    _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA,
    _ENTRY_NONZERO_ACTION_SCHEMAS,
    _EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256,
    _EXPECTED_ENTRY_NONZERO_ACTION_SCHEMA_SHA256,
    _EXPECTED_ENTRY_RAW_ORDER_PARTITION_INVENTORY_SHA256,
    _EXPECTED_ENTRY_RAW_ORDER_PARTITION_UNIQUE_SUM,
    _NARROW_ACTION_SCHEMA,
    _NARROW_EXIT_RULE_SCHEMA,
    _NARROW_POSITION_SCHEMA,
    _canonical_frame_records_sha256,
    _canonical_sha256,
    _current_formal_source_binding_record,
    _discard_clean_file_cache,
    _discard_cross_session_artifact_cache,
    _emit_memory_phase,
    _formal_exit_artifact_schemas,
    _file_sha256,
    _load_narrow_formal_partition_inputs,
    _load_narrow_entry_sources,
    _merge_partition_execution_diagnostics,
    _ordered_schema_sha256,
    _policy_path_universe_sha256,
    _prepare_policy_paths,
    _record_partition_raw_order_inventory,
    _publish_evaluation,
    _validate_formal_coverage,
    _validate_cross_embedded_source_bindings,
    _validate_prerequisite_source_root_binding,
    _verify_post_cross_position_evaluation_with_sources,
    build_post_cross_position_evaluation,
    run_post_cross_position_evaluation,
    verify_post_cross_position_evaluation,
)
from maker.src.tests.test_quote_fill_exit_maker_report import (
    VALUE_CODE,
    _publish_synthetic_partition,
)
from maker.src.tests.test_quote_fill_filled_entry_report import (
    _inputs as _filled_entry_inputs,
)


FUTURE_ENTRY = "future_ask_spot_taker"
SPOT_ENTRY = "spot_bid_future_taker"
FUTURE_EXIT = "future_bid_spot_taker"
SPOT_EXIT = "spot_ask_future_taker"
SESSIONS = tuple(f"202601{day:02d}" for day in range(1, 11))
ENTRY_SESSIONS = SESSIONS[1:]


def _empty_report(metadata: dict[str, object]) -> FilledEntryPrimaryReport:
    empty = pl.DataFrame()
    return FilledEntryPrimaryReport(
        coverage=empty,
        filled_entry_policy_paths=empty,
        policy_summary=empty,
        pooled_12_cell_summary=empty,
        product_policy_summary=empty,
        execution_diagnostics=empty,
        metadata=metadata,
    )


def _sources(*, censored: bool) -> FormalPostCrossSources:
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(ENTRY_SESSIONS):
        for q in (50, 80, 95):
            for entry_route in (FUTURE_ENTRY, SPOT_ENTRY):
                raw = f"{date}/{entry_route}/q{q}"
                dependency = f"dep/{raw}"
                for rule in ("frozen_center", "frozen_lower"):
                    for exit_route in (FUTURE_EXIT, SPOT_EXIT):
                        is_censored = bool(
                            censored
                            and date == ENTRY_SESSIONS[-1]
                            and q == 50
                            and entry_route == FUTURE_ENTRY
                            and rule == "frozen_center"
                            and exit_route == FUTURE_EXIT
                        )
                        # Keep one q50/future/center/future position open across
                        # two entry dates so the count=1 sweep must reject one.
                        overnight = bool(
                            q == 50
                            and entry_route == FUTURE_ENTRY
                            and rule == "frozen_center"
                            and exit_route == FUTURE_EXIT
                            and date_index < len(ENTRY_SESSIONS) - 2
                        )
                        terminal_date = (
                            None
                            if is_censored
                            else SESSIONS[min(date_index + 3, len(SESSIONS) - 1)]
                            if overnight
                            else date
                        )
                        category = "censored" if is_censored else "completed"
                        notional = 200_000.0
                        gross_bp = float(q - 40) + (
                            2.0 if rule == "frozen_lower" else 0.0
                        )
                        gross = None if is_censored else gross_bp / 10_000 * notional
                        observed = terminal_date or SESSIONS[-1]
                        trial = f"trial/{date}/{entry_route}/{q}/{rule}/{exit_route}"
                        rows.append(
                            {
                                "Date": date,
                                "ValueCode": "2330",
                                "QuoteCode": "CDF1",
                                "entry_route": entry_route,
                                "boundary_quantile": q,
                                "entry_raw_order_fact_id": raw,
                                "position_established_ns": 1_000_000_000,
                                "exit_rule_id": rule,
                                "exit_route": exit_route,
                                "exit_policy_trial_id": trial,
                                "physical_entry_dependency_id": dependency,
                                "filled_entry_policy_aliases": (
                                    2
                                    if q == 50
                                    and entry_route == FUTURE_ENTRY
                                    else 1
                                ),
                                "normalization_notional_twd": notional,
                                "filled_entry_outcome_category": category,
                                "terminal_date": terminal_date,
                                "terminal_reason": (
                                    "right_censored_expiry_settlement_unpriced"
                                    if is_censored
                                    else "synthetic_terminal"
                                ),
                                "gross_cycle_pnl_twd": gross,
                                "gross_cycle_bp": None if is_censored else gross_bp,
                                "last_observed_session_date": observed,
                                "exit_decision_time_ns": (
                                    None if is_censored else 2_000_000_000
                                ),
                                "outcome_type": (
                                    "censored" if is_censored else "terminal"
                                ),
                                "outcome_status": (
                                    "right_censored_expiry_settlement_unpriced"
                                    if is_censored
                                    else "synthetic_terminal"
                                ),
                                "terminal_cashflow_priced": not is_censored,
                                "cancel_semantics": "nominal_instant_cancel_v0",
                                "nominal_cancel_model_assumption": True,
                                "pathwise_ev_ready": False,
                                "joint_volume_allocated": False,
                            }
                        )
    metadata = {
        "cross_session_manifest_sha256": "a" * 64,
        "cross_session_runner_config_sha256": "b" * 64,
        "cross_session_prerequisite_marker_sha256": "c" * 64,
        "cross_session_prerequisite_marker_payload_sha256": "d" * 64,
        "cross_session_prerequisite_config_sha256": "e" * 64,
        "cross_session_prerequisite_source_identity_sha256": "f" * 64,
        "cross_session_partition_hashes_validated": True,
        "cross_session_policy_lineage_crossvalidated": True,
        "cross_embedded_formal_sources_reconciled": True,
        "cross_session_formal_prerequisite_bound": True,
        "formal_entry_root_prerequisite_bound": True,
        "cross_session_d_minus_one_lineage_validated": True,
        "position_policy_to_bound_entry_exit_facts_crossvalidated": True,
        "entry_exit_hedge_delay_match": True,
        "entry_action_hedge_delay_ns": 50_000_000,
        "exit_maker_hedge_delay_ns": 50_000_000,
        "complete_center_lower_x_two_exit_routes_per_established_entry": True,
        "gross_zero_imputation": False,
        "unknown_or_censored_cashflow_imputed": False,
        "strict_cross_session_outcomes_in_primary": False,
        "terminal_overlay_cancel_semantics": "nominal_instant_cancel_v0",
        "report_implementation_sources": _report_implementation_sources(),
        "report_implementation_sources_sha256": _canonical_sha256(
            _report_implementation_sources()
        ),
        "formal_hash_validation_skippable": False,
        "formal_narrow_loader": True,
        "formal_exit_artifact_schemas_exact": True,
        "entry_action_nonzero_full_schemas_exact": True,
        "entry_action_schema_inventory_sha256": (
            _EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256
        ),
        "entry_action_full_schema_inventory_bound": True,
        "entry_raw_order_partition_inventory_sha256": (
            _EXPECTED_ENTRY_RAW_ORDER_PARTITION_INVENTORY_SHA256
        ),
        "entry_raw_order_partition_unique_sum": (
            _EXPECTED_ENTRY_RAW_ORDER_PARTITION_UNIQUE_SUM
        ),
        "entry_raw_order_partition_disjoint_bound": True,
        "entry_raw_order_full_inventory_bound": True,
        "formal_source_cache_discard_required": True,
        "formal_source_cache_discard_complete": True,
        "formal_source_cache_discard_files": 10,
        "cross_session_source_cache_discard_files": 6,
        "post_cross_full_actions_preaggregated": True,
        "post_cross_full_action_diagnostics_preaggregated": True,
        "post_cross_retained_established_action_rows": 1,
        "post_cross_execution_diagnostic_rows": 1,
        "unused_exit_artifacts_materialized": False,
        "bound_entry_exit_rules_only": True,
        "same_day_root_manifest_exact_inventory": True,
        "same_day_root_manifest_exact_schema": True,
        "formal_entry_product_days": 2_687,
        "formal_entry_grid_product_days": 2_700,
        "formal_entry_missing_product_days": 13,
        "formal_entry_session_count": 60,
        "exit_maker_root": "/synthetic/exit",
        "entry_execution_root": "/synthetic/entry",
        "cross_session_root": "/synthetic/cross",
        "cross_session_prerequisite_root": "/synthetic/prerequisite",
    }
    return FormalPostCrossSources(
        report=_empty_report(metadata),
        policy_paths=pl.from_dicts(rows, infer_schema_length=None),
        session_calendar=SESSIONS,
        entry_sessions=ENTRY_SESSIONS,
        metadata=metadata,
    )


def _config() -> PostCrossEvaluationConfig:
    return PostCrossEvaluationConfig(
        lookback_sessions=9,
        min_training_dates=2,
        min_policy_origins=2,
        min_completed_cycles_for_diagnostic_rank=2,
        confidence_level=0.95,
        minimum_lcb_bp=0.0,
        expected_product_days=1,
    )


def _write_cost_profile(root: Path, sources: FormalPostCrossSources) -> None:
    prepared = _prepare_policy_paths(sources.policy_paths, sources.session_calendar)
    universe = _policy_path_universe_sha256(prepared)
    costs = prepared.select("policy_path_id").with_columns(
        *(pl.lit(1.0).alias(name) for name in COST_COMPONENT_COLUMNS)
    )
    root.mkdir()
    artifact = root / "path_costs.parquet"
    costs.write_parquet(artifact)
    artifact_sha = _file_sha256(artifact)
    profile_hash = _canonical_sha256(
        {
            "profile_id": "synthetic_full_cost",
            "profile_version": "v1",
            "source_asof_date": "20251231",
            "contains_target_day_outcome": False,
            "component_columns": list(COST_COMPONENT_COLUMNS),
            "artifact_sha256": artifact_sha,
            "bound_policy_path_universe_sha256": universe,
            "bound_cross_session_manifest_sha256": "a" * 64,
        }
    )
    marker = {
        "complete": True,
        "schema_version": COST_PROFILE_SCHEMA_VERSION,
        "profile_id": "synthetic_full_cost",
        "profile_version": "v1",
        "profile_hash": profile_hash,
        "source_asof_date": "20251231",
        "contains_target_day_outcome": False,
        "component_definitions_complete": True,
        "bound_policy_path_universe_sha256": universe,
        "bound_cross_session_manifest_sha256": "a" * 64,
        "artifact": {
            "name": "path_costs.parquet",
            "rows": costs.height,
            "sha256": artifact_sha,
        },
    }
    (root / "complete.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )


def _conform_frame(
    frame: pl.DataFrame, schema: dict[str, pl.DataType] | object
) -> pl.DataFrame:
    expected = dict(schema)  # type: ignore[arg-type]
    return frame.with_columns(
        *(
            pl.col(name).cast(dtype)
            if name in frame.columns
            else pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in expected.items()
        )
    ).select(list(expected))


def _make_exact_narrow_loader_fixture(
    exit_root: Path,
    entry_root: Path,
    *,
    zero_rows: bool = False,
) -> None:
    date = "20260102"
    _publish_synthetic_partition(exit_root, entry_root, date)
    partition = exit_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    entry_partition = entry_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    action_path = entry_partition / "execution_action_facts.parquet"
    action_frame = pl.read_parquet(action_path).with_columns(
        pl.lit(False).alias("cancel_required")
    )
    policy_id_replacements: dict[str, str] = {}
    if zero_rows:
        action_frame = pl.DataFrame(schema=_ENTRY_ZERO_ACTION_SCHEMA)
    else:
        policy_id_replacements = {
            str(value): f"{date}/{VALUE_CODE}/{str(value)}"
            for value in action_frame["policy_generation_id"].to_list()
        }
        action_frame = action_frame.with_columns(
            pl.col("policy_generation_id").replace(policy_id_replacements)
        )
        action_frame = _conform_frame(
            action_frame, _ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA
        )
    action_frame.write_parquet(action_path)
    exit_rule_path = entry_partition / "exit_facts.parquet"
    if zero_rows:
        pl.DataFrame(schema=_ENTRY_ZERO_EXIT_RULE_SCHEMA).write_parquet(
            exit_rule_path
        )
    else:
        pl.read_parquet(exit_rule_path).with_columns(
            pl.col("policy_generation_id").replace(policy_id_replacements)
        ).write_parquet(exit_rule_path)
    target_audit_path = entry_partition / "target_audit.parquet"
    if not target_audit_path.exists():
        pl.DataFrame(schema={"Date": pl.String}).write_parquet(target_audit_path)
    entry_marker_path = entry_partition / "complete.json"
    entry_marker = json.loads(entry_marker_path.read_text(encoding="utf-8"))
    for artifact_path in (action_path, exit_rule_path, target_audit_path):
        artifact_frame = pl.read_parquet(artifact_path)
        entry_marker["artifacts"][artifact_path.name] = {
            "rows": artifact_frame.height,
            "columns": artifact_frame.width,
            "bytes": artifact_path.stat().st_size,
            "sha256": _file_sha256(artifact_path),
        }
    entry_marker_path.write_text(
        json.dumps(entry_marker, sort_keys=True), encoding="utf-8"
    )

    exit_marker_path = partition / "complete.json"
    exit_marker = json.loads(exit_marker_path.read_text(encoding="utf-8"))
    exit_marker["config"]["source"]["action_source"]["sha256"] = (
        entry_marker["artifacts"]["execution_action_facts.parquet"]["sha256"]
    )
    exit_marker["config"]["source"]["exit_rule_source"]["sha256"] = (
        entry_marker["artifacts"]["exit_facts.parquet"]["sha256"]
    )
    exit_marker["config_sha256"] = _exit_report._canonical_sha256(
        exit_marker["config"]
    )
    exit_marker_path.write_text(
        json.dumps(exit_marker, sort_keys=True), encoding="utf-8"
    )

    schemas = _formal_exit_artifact_schemas()
    for name, schema in schemas.items():
        path = partition / name
        current = pl.read_parquet(path)
        if (
            name == "exit_maker_position_policy_facts.parquet"
            and policy_id_replacements
        ):
            current = current.with_columns(
                pl.col("entry_policy_generation_id").replace(
                    policy_id_replacements
                )
            )
        if zero_rows:
            current = current.head(0)
        _conform_frame(current, schema).write_parquet(path)

    marker_path = exit_marker_path
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, object]] = {}
    for name in schemas:
        path = partition / name
        frame_schema = pl.read_parquet_schema(path)
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        artifacts[name] = {
            "rows": rows,
            "columns": len(frame_schema),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    marker["artifacts"] = artifacts
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")

    source = marker["config"]["source"]
    manifest_row: dict[str, object] = {
        "Date": date,
        "ValueCode": VALUE_CODE,
        "partition": str(partition),
        "runner_config_sha256": marker["runner_config_sha256"],
        "config_sha256": marker["config_sha256"],
        "action_source_sha256": source["action_source"]["sha256"],
        "exit_rule_source_kind": source["exit_rule_source"]["kind"],
        "exit_rule_source_sha256": source["exit_rule_source"]["sha256"],
        "complete": True,
    }
    for name, column in _EXIT_ARTIFACT_ROW_COLUMNS.items():
        manifest_row[column] = artifacts[name]["rows"]
    pl.from_dicts(
        [manifest_row], schema=_EXIT_MAKER_MANIFEST_SCHEMA, strict=True
    ).write_parquet(exit_root / "exit_maker_partition_manifest.parquet")


def _resign_narrow_entry_sources(exit_root: Path, entry_root: Path) -> None:
    """Coordinate marker/root declarations after a synthetic entry rewrite."""

    date = "20260102"
    entry_partition = entry_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    entry_marker_path = entry_partition / "complete.json"
    entry_marker = json.loads(entry_marker_path.read_text(encoding="utf-8"))
    for name in ("execution_action_facts.parquet", "exit_facts.parquet"):
        path = entry_partition / name
        schema = pl.read_parquet_schema(path)
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        entry_marker["artifacts"][name] = {
            "rows": rows,
            "columns": len(schema),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    entry_marker_path.write_text(
        json.dumps(entry_marker, sort_keys=True), encoding="utf-8"
    )

    partition = exit_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    exit_marker_path = partition / "complete.json"
    exit_marker = json.loads(exit_marker_path.read_text(encoding="utf-8"))
    source = exit_marker["config"]["source"]
    source["action_source"]["sha256"] = entry_marker["artifacts"][
        "execution_action_facts.parquet"
    ]["sha256"]
    source["exit_rule_source"]["sha256"] = entry_marker["artifacts"][
        "exit_facts.parquet"
    ]["sha256"]
    exit_marker["config_sha256"] = _exit_report._canonical_sha256(
        exit_marker["config"]
    )
    exit_marker_path.write_text(
        json.dumps(exit_marker, sort_keys=True), encoding="utf-8"
    )

    manifest_path = exit_root / "exit_maker_partition_manifest.parquet"
    pl.read_parquet(manifest_path).with_columns(
        pl.lit(exit_marker["config_sha256"]).alias("config_sha256"),
        pl.lit(source["action_source"]["sha256"]).alias(
            "action_source_sha256"
        ),
        pl.lit(source["exit_rule_source"]["sha256"]).alias(
            "exit_rule_source_sha256"
        ),
    ).write_parquet(manifest_path)


def _legacy_prepare_policy_paths_reference(
    frame: pl.DataFrame, calendar: tuple[str, ...]
) -> pl.DataFrame:
    index = {value: position for position, value in enumerate(calendar)}
    records: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        item = dict(row)
        entry_date = str(item["Date"])
        category = str(item["filled_entry_outcome_category"])
        terminal_date = item.get("terminal_date")
        observed_date = item.get("last_observed_session_date")
        if category == "completed":
            label_date = str(terminal_date)
            interval_end = str(terminal_date)
        else:
            label_date = str(observed_date)
            observed_index = index[label_date]
            interval_end = (
                calendar[observed_index + 1]
                if observed_index + 1 < len(calendar)
                else None
            )
        path_identity = {name: item[name] for name in PHYSICAL_PATH_KEY}
        item.update(
            policy_path_id=_canonical_sha256(path_identity),
            label_availability_date=label_date,
            outstanding_interval_end_exclusive=interval_end,
            holding_session_boundaries=index[label_date] - index[entry_date],
            completed_same_day=(
                category == "completed" and str(terminal_date) == entry_date
            ),
            completed_overnight=(
                category == "completed" and str(terminal_date) > entry_date
            ),
            outstanding_interval_semantics=(
                "entry_eod_through_before_terminal_date;unresolved_through_last_observed_eod"
            ),
        )
        records.append(item)
    return pl.from_dicts(records, infer_schema_length=None).sort(
        [*POLICY_KEY, "Date", "position_established_ns", "policy_path_id"]
    )


class PostCrossPositionEvaluatorTest(unittest.TestCase):
    def test_vectorized_path_prepare_and_streaming_digest_match_legacy(self) -> None:
        sources = _sources(censored=True)
        actual = _prepare_policy_paths(
            sources.policy_paths, sources.session_calendar
        )
        legacy = _legacy_prepare_policy_paths_reference(
            sources.policy_paths, tuple(sources.session_calendar)
        )
        self.assertEqual(actual.schema, legacy.schema)
        self.assertTrue(actual.equals(legacy, null_equal=True))
        columns = [
            "policy_path_id",
            "exit_policy_trial_id",
            "filled_entry_outcome_category",
            "terminal_date",
            "label_availability_date",
            "gross_cycle_pnl_twd",
            "normalization_notional_twd",
        ]
        legacy_digest = _canonical_sha256(
            legacy.select(columns).sort("policy_path_id").to_dicts()
        )
        self.assertEqual(_policy_path_universe_sha256(actual), legacy_digest)

    def test_incremental_canonical_digest_matches_legacy_edge_values(self) -> None:
        frame = pl.DataFrame(
            {
                "identity": ["z", "a", "unicode/台灣", "quote/\"\\\n"],
                "nullable": [None, "", "null", "\\t"],
                "value": [-0.0, 1.7976931348623157e308, 5e-324, 1.25],
            },
            schema={
                "identity": pl.String,
                "nullable": pl.String,
                "value": pl.Float64,
            },
        )
        expected = _canonical_sha256(
            frame.select("identity", "nullable", "value")
            .sort("identity")
            .to_dicts()
        )
        self.assertEqual(
            _canonical_frame_records_sha256(
                frame,
                columns=("identity", "nullable", "value"),
                sort_by=("identity",),
            ),
            expected,
        )

    def test_narrow_formal_loader_never_materializes_unused_exit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            original = pl.read_parquet
            unused = set(_formal_exit_artifact_schemas()) - {
                "exit_maker_position_policy_facts.parquet"
            }

            def guarded(path: object, *args: object, **kwargs: object) -> pl.DataFrame:
                if Path(path).name in unused:
                    raise AssertionError(f"unused artifact materialized: {path}")
                return original(path, *args, **kwargs)

            with patch.object(pl, "read_parquet", side_effect=guarded):
                inputs = _load_narrow_formal_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                    value_codes=None,
                    session_calendar=("20260101", "20260102"),
                )
            self.assertEqual(inputs.action_facts.height, 2)
            self.assertEqual(inputs.position_policy_facts.height, 4)
            self.assertEqual(inputs.taker_exit_facts.height, 2)
            self.assertEqual(
                inputs.taker_exit_facts.schema, _NARROW_EXIT_RULE_SCHEMA
            )
            self.assertEqual(
                inputs.position_policy_facts.schema, _NARROW_POSITION_SCHEMA
            )
            self.assertFalse(inputs.metadata["unused_exit_artifacts_materialized"])
            self.assertEqual(
                inputs.metadata["formal_exit_artifacts_hash_validated"], 7
            )
            expected = _exit_report._validate_position_rule_lineage(
                inputs.position_policy_facts,
                inputs.taker_exit_facts,
                expected_session_predecessors={"20260102": "20260101"},
            )
            self.assertEqual(inputs.metadata["position_rule_lineage"], expected)

    def test_frozen_nonzero_action_schema_whitelist_is_exact(self) -> None:
        self.assertEqual(
            {
                _ordered_schema_sha256(schema)
                for schema in _ENTRY_NONZERO_ACTION_SCHEMAS
            },
            set(_EXPECTED_ENTRY_NONZERO_ACTION_SCHEMA_SHA256),
        )
        self.assertEqual(len(_ENTRY_NONZERO_ACTION_SCHEMAS), 5)

    def test_formal_cache_discard_preserves_digest_and_invokes_fadvise(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.parquet"
            path.write_bytes(b"immutable-formal-bytes")
            before = _file_sha256(path)
            with patch.object(os, "posix_fadvise") as advise:
                _discard_clean_file_cache(path, source="unit artifact")
            self.assertEqual(_file_sha256(path), before)
            advise.assert_called_once()
            self.assertEqual(advise.call_args.args[1:], (0, 0, os.POSIX_FADV_DONTNEED))

    def test_formal_cache_discard_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.parquet"
            path.write_bytes(b"immutable-formal-bytes")
            with patch.object(
                os, "posix_fadvise", side_effect=OSError("advice failed")
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "failed to discard validated"
                ):
                    _discard_clean_file_cache(path, source="unit artifact")

    def test_cross_cache_discard_enumerates_exact_v8_artifacts(self) -> None:
        from maker.src.quote_fill.exit_maker_cross_session_runner import (
            CROSS_SESSION_MANIFEST_NAME,
            _OUTPUT_ARTIFACTS,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition = root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
            partition.mkdir(parents=True)
            (root / CROSS_SESSION_MANIFEST_NAME).touch()
            coverage = pl.DataFrame(
                {
                    "Date": ["20260102", "20260102"],
                    "ValueCode": [VALUE_CODE, "missing"],
                    "partition_complete": [True, False],
                }
            )
            with patch(
                "maker.src.quote_fill.post_cross_position_evaluator."
                "_discard_clean_file_cache"
            ) as discard:
                count = _discard_cross_session_artifact_cache(root, coverage)
            self.assertEqual(count, len(_OUTPUT_ARTIFACTS) + 1)
            actual = {call.args[0] for call in discard.call_args_list}
            self.assertEqual(
                actual,
                {
                    *(partition / name for name in _OUTPUT_ARTIFACTS),
                    root / CROSS_SESSION_MANIFEST_NAME,
                },
            )

    def test_memory_phase_emits_structured_nonsemantic_diagnostic(self) -> None:
        stream = io.StringIO()
        with redirect_stderr(stream):
            _emit_memory_phase("unit_phase", rows=7)
        payload = json.loads(stream.getvalue())[
            "post_cross_memory_phase"
        ]
        self.assertEqual(payload["phase"], "unit_phase")
        self.assertEqual(payload["rows"], 7)
        self.assertIsInstance(payload["pid"], int)

    def test_partition_execution_diagnostic_merge_matches_full_actions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            base = pl.read_parquet(
                entry_root
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / "execution_action_facts.parquet"
            ).select(list(_NARROW_ACTION_SCHEMA))
            no_fill = base.with_columns(
                pl.lit("20260103").alias("Date"),
                pl.concat_str(
                    pl.lit("20260103/"),
                    pl.col("raw_order_fact_id"),
                ).alias("raw_order_fact_id"),
                pl.concat_str(
                    pl.lit(f"20260103/{VALUE_CODE}/"),
                    pl.col("policy_generation_id"),
                ).alias("policy_generation_id"),
                pl.lit(False).alias("any_fill"),
                pl.lit(False).alias("full_fill"),
                pl.lit(False).alias("partial_fill"),
                pl.lit(True).alias("cancel_required"),
                pl.lit(None, dtype=pl.Int64).alias(
                    "full_fill_recv_time_ns"
                ),
                pl.lit(None, dtype=pl.String).alias("entry_hedge_status"),
                pl.lit(None, dtype=pl.Int64).alias(
                    "entry_hedge_decision_time_ns"
                ),
                pl.lit(False).alias("entry_hedge_label_observed"),
                pl.lit(False).alias("entry_hedge_executable"),
            )
            combined = pl.concat((base, no_fill), how="vertical")
            expected = _filled_entry_report._build_execution_diagnostics(
                combined
            ).sort("ValueCode", "entry_route", "boundary_quantile")
            actual = _merge_partition_execution_diagnostics(
                [
                    _filled_entry_report._build_execution_diagnostics(base),
                    _filled_entry_report._build_execution_diagnostics(no_fill),
                ],
                expected_action_rows=combined.height,
                expected_established_rows=base.height,
                raw_order_ids_partition_disjoint=True,
            )
            self.assertEqual(actual.schema, expected.schema)
            self.assertTrue(actual.equals(expected, null_equal=True))

    def test_partition_diagnostics_reject_cross_partition_raw_id_collision(
        self,
    ) -> None:
        base = _filled_entry_inputs().action_facts.head(1)
        duplicate = base.with_columns(
            pl.lit("20260103").alias("Date"),
            pl.concat_str(
                pl.lit("20260103/"), pl.col("policy_generation_id")
            ).alias("policy_generation_id"),
        )
        combined = pl.concat((base, duplicate), how="vertical")
        legacy = _filled_entry_report._build_execution_diagnostics(combined)
        partitioned = [
            _filled_entry_report._build_execution_diagnostics(base),
            _filled_entry_report._build_execution_diagnostics(duplicate),
        ]
        self.assertEqual(
            int(legacy["submitted_physical_raw_orders"].sum()), 1
        )
        self.assertEqual(
            sum(
                int(frame["submitted_physical_raw_orders"].sum())
                for frame in partitioned
            ),
            2,
        )
        with self.assertRaisesRegex(ValueError, "partition-disjoint"):
            _merge_partition_execution_diagnostics(
                partitioned,
                expected_action_rows=combined.height,
                expected_established_rows=combined.height,
                raw_order_ids_partition_disjoint=False,
            )

        seen: set[str] = set()
        _record_partition_raw_order_inventory(
            base,
            partition="Date=20260102/ValueCode=2330",
            seen_raw_order_fact_ids=seen,
        )
        with self.assertRaisesRegex(ValueError, "crosses a product-day"):
            _record_partition_raw_order_inventory(
                duplicate,
                partition="Date=20260103/ValueCode=2330",
                seen_raw_order_fact_ids=seen,
            )

    def test_established_only_report_path_is_exactly_legacy_equivalent(
        self,
    ) -> None:
        inputs = _filled_entry_inputs()
        legacy = _filled_entry_report.build_filled_entry_primary_report(inputs)
        retained = inputs.action_facts.filter(
            _filled_entry_report._established_entry_expr()
        )
        optimized_inputs = _exit_report.ExitMakerPartitionInputs(
            policy_support=inputs.policy_support,
            candidate_aliases=inputs.candidate_aliases,
            raw_candidate_facts=inputs.raw_candidate_facts,
            position_policy_facts=inputs.position_policy_facts,
            action_facts=retained,
            taker_exit_facts=inputs.taker_exit_facts,
            coverage=inputs.coverage,
            metadata=inputs.metadata,
        )
        optimized = _filled_entry_report.build_filled_entry_primary_report(
            optimized_inputs
        )
        for name in (
            "coverage",
            "filled_entry_policy_paths",
            "policy_summary",
            "pooled_12_cell_summary",
            "product_policy_summary",
        ):
            expected = getattr(legacy, name)
            actual = getattr(optimized, name)
            self.assertEqual(actual.schema, expected.schema, name)
            self.assertTrue(actual.equals(expected, null_equal=True), name)
        diagnostics = _merge_partition_execution_diagnostics(
            [legacy.execution_diagnostics],
            expected_action_rows=inputs.action_facts.height,
            expected_established_rows=retained.height,
            raw_order_ids_partition_disjoint=True,
        )
        expected_diagnostics = legacy.execution_diagnostics.sort(
            "ValueCode", "entry_route", "boundary_quantile"
        )
        self.assertEqual(diagnostics.schema, expected_diagnostics.schema)
        self.assertTrue(
            diagnostics.equals(expected_diagnostics, null_equal=True)
        )
        optimized_metadata = dict(optimized.metadata)
        optimized_metadata["entry_policy_aliases_all"] = (
            inputs.action_facts.height
        )
        self.assertEqual(optimized_metadata, dict(legacy.metadata))

    def test_narrow_entry_loader_normalizes_exact_no_fill_null_cursor_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            entry_partition = (
                entry_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
            )
            action_path = entry_partition / "execution_action_facts.parquet"
            null_columns = [
                name
                for name, dtype in _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA.items()
                if dtype == pl.Null
            ]
            action = pl.read_parquet(action_path).with_columns(
                *(pl.lit(None).alias(name) for name in null_columns),
                pl.lit(False).alias("any_fill"),
                pl.lit(False).alias("full_fill"),
                pl.lit(False).alias("partial_fill"),
                pl.lit(True).alias("cancel_required"),
                pl.lit(0, dtype=pl.Int64).alias("known_filled_quantity"),
                pl.lit(None, dtype=pl.String).alias("entry_hedge_status"),
                pl.lit(None, dtype=pl.Int64).alias(
                    "entry_hedge_decision_time_ns"
                ),
                pl.lit(False).alias("entry_hedge_label_observed"),
                pl.lit(False).alias("entry_hedge_executable"),
            )
            _conform_frame(
                action, _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA
            ).write_parquet(action_path)
            _resign_narrow_entry_sources(exit_root, entry_root)
            exit_marker = json.loads(
                (
                    exit_root
                    / "Date=20260102"
                    / f"ValueCode={VALUE_CODE}"
                    / "complete.json"
                ).read_text(encoding="utf-8")
            )
            actions, _, _, _, schema_record = _load_narrow_entry_sources(
                entry_partition,
                exit_marker["config"]["source"],
                date="20260102",
                value_code=VALUE_CODE,
            )
            self.assertEqual(actions.schema, _NARROW_ACTION_SCHEMA)
            self.assertEqual(actions["full_fill_recv_time_ns"].dtype, pl.Int64)
            self.assertEqual(actions["full_fill_recv_time_ns"].null_count(), 2)
            self.assertEqual(int(actions["full_fill"].sum()), 0)
            self.assertEqual(
                schema_record["schema_sha256"],
                _ordered_schema_sha256(
                    _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA
                ),
            )

    def test_narrow_entry_loader_rejects_nonzero_schema_near_misses(self) -> None:
        for attack in ("other_null", "extra", "reordered"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                exit_root = root / "exit"
                entry_root = root / "entry"
                _make_exact_narrow_loader_fixture(exit_root, entry_root)
                entry_partition = (
                    entry_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
                )
                action_path = entry_partition / "execution_action_facts.parquet"
                action = pl.read_parquet(action_path)
                if attack == "other_null":
                    action = action.with_columns(
                        pl.lit(None).alias("entry_hedge_decision_time_ns")
                    )
                elif attack == "extra":
                    action = action.with_columns(
                        pl.lit(b"attacker", dtype=pl.Binary).alias(
                            "attacker_extra"
                        )
                    )
                else:
                    action = action.select(list(reversed(action.columns)))
                action.write_parquet(action_path)
                _resign_narrow_entry_sources(exit_root, entry_root)
                exit_marker = json.loads(
                    (
                        exit_root
                        / "Date=20260102"
                        / f"ValueCode={VALUE_CODE}"
                        / "complete.json"
                    ).read_text(encoding="utf-8")
                )
                with self.assertRaisesRegex(
                    ValueError, "nonzero producer schema mismatch"
                ):
                    _load_narrow_entry_sources(
                        entry_partition,
                        exit_marker["config"]["source"],
                        date="20260102",
                        value_code=VALUE_CODE,
                    )

    def test_narrow_entry_loader_rejects_full_true_with_null_full_cursor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            entry_partition = (
                entry_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
            )
            action_path = entry_partition / "execution_action_facts.parquet"
            null_columns = [
                name
                for name, dtype in _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA.items()
                if dtype == pl.Null
            ]
            action = pl.read_parquet(action_path).with_columns(
                *(pl.lit(None).alias(name) for name in null_columns),
                pl.lit(True).alias("any_fill"),
                pl.lit(True).alias("full_fill"),
                pl.lit(False).alias("partial_fill"),
                pl.lit(False).alias("entry_hedge_label_observed"),
                pl.lit(False).alias("entry_hedge_executable"),
            )
            _conform_frame(
                action, _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA
            ).write_parquet(action_path)
            _resign_narrow_entry_sources(exit_root, entry_root)
            exit_marker = json.loads(
                (
                    exit_root
                    / "Date=20260102"
                    / f"ValueCode={VALUE_CODE}"
                    / "complete.json"
                ).read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "incoherent 50 ms hedge"):
                _load_narrow_entry_sources(
                    entry_partition,
                    exit_marker["config"]["source"],
                    date="20260102",
                    value_code=VALUE_CODE,
                )

    def test_real_no_fill_partition_uses_whitelisted_null_cursor_schema(
        self,
    ) -> None:
        entry_partition = Path(
            "maker/data/walkforward/execution_narrow_60d/"
            "Date=20260520/ValueCode=2412"
        )
        same_day_marker = Path(
            "maker/data/walkforward/exit_maker_narrow_60d/"
            "Date=20260520/ValueCode=2412/complete.json"
        )
        if not entry_partition.is_dir() or not same_day_marker.is_file():
            self.skipTest("frozen 60-day formal roots are unavailable")
        exit_marker = json.loads(same_day_marker.read_text(encoding="utf-8"))
        actions, _, _, _, schema_record = _load_narrow_entry_sources(
            entry_partition.resolve(),
            exit_marker["config"]["source"],
            date="20260520",
            value_code="2412",
        )
        self.assertEqual(actions.height, 133)
        self.assertEqual(actions.schema, _NARROW_ACTION_SCHEMA)
        self.assertEqual(actions["full_fill_recv_time_ns"].dtype, pl.Int64)
        self.assertEqual(actions["full_fill_recv_time_ns"].null_count(), 133)
        self.assertEqual(int(actions["full_fill"].sum()), 0)
        self.assertEqual(
            schema_record["schema_sha256"],
            "d4462644570e6095fc171d71412f43a3b2dec461146bab0eb38f38bd1fa8dbcc",
        )

    def test_narrow_loader_normalizes_a_typed_zero_row_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(
                exit_root, entry_root, zero_rows=True
            )
            inputs = _load_narrow_formal_partition_inputs(
                exit_root,
                entry_root,
                sessions=1,
                value_codes=None,
                session_calendar=("20260101", "20260102"),
            )
            self.assertTrue(inputs.action_facts.is_empty())
            self.assertTrue(inputs.position_policy_facts.is_empty())
            self.assertTrue(inputs.taker_exit_facts.is_empty())
            self.assertEqual(
                inputs.position_policy_facts.schema, _NARROW_POSITION_SCHEMA
            )
            self.assertEqual(
                inputs.taker_exit_facts.schema, _NARROW_EXIT_RULE_SCHEMA
            )
            self.assertEqual(
                inputs.metadata["position_rule_lineage"][
                    "bound_rule_lineage_sha256"
                ],
                _canonical_sha256([]),
            )

    def test_narrow_loader_rejects_resigned_zero_row_entry_schema_tamper(self) -> None:
        attacks = (
            (
                "execution_action_facts.parquet",
                _ENTRY_ZERO_ACTION_SCHEMA,
                _NARROW_ACTION_SCHEMA,
            ),
            (
                "exit_facts.parquet",
                _ENTRY_ZERO_EXIT_RULE_SCHEMA,
                _NARROW_EXIT_RULE_SCHEMA,
            ),
        )
        for artifact_name, valid_schema, projected_schema in attacks:
            for attack in ("wrong_dtype", "projected_plus_extra"):
                with self.subTest(
                    artifact=artifact_name, attack=attack
                ), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    exit_root = root / "exit"
                    entry_root = root / "entry"
                    _make_exact_narrow_loader_fixture(
                        exit_root, entry_root, zero_rows=True
                    )
                    if attack == "wrong_dtype":
                        wrong_schema = dict(valid_schema)
                        first = next(iter(wrong_schema))
                        wrong_schema[first] = pl.Int64
                    else:
                        wrong_schema = dict(projected_schema)
                        wrong_schema["attacker_extra"] = pl.Binary
                    path = (
                        entry_root
                        / "Date=20260102"
                        / f"ValueCode={VALUE_CODE}"
                        / artifact_name
                    )
                    pl.DataFrame(schema=wrong_schema).write_parquet(path)
                    _resign_narrow_entry_sources(exit_root, entry_root)
                    with self.assertRaisesRegex(
                        ValueError, "zero-row producer schema mismatch"
                    ):
                        _load_narrow_formal_partition_inputs(
                            exit_root,
                            entry_root,
                            sessions=1,
                            value_codes=None,
                            session_calendar=("20260101", "20260102"),
                        )

    def test_narrow_loader_rejects_noncanonical_manifest_partition_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            manifest_path = exit_root / "exit_maker_partition_manifest.parquet"
            pl.read_parquet(manifest_path).with_columns(
                pl.lit(
                    "attacker-prefix/"
                    + str(
                        exit_root
                        / "Date=20260102"
                        / f"ValueCode={VALUE_CODE}"
                    )
                ).alias("partition")
            ).write_parquet(manifest_path)
            with self.assertRaisesRegex(ValueError, "disagrees on partition"):
                _load_narrow_formal_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                    value_codes=None,
                    session_calendar=("20260101", "20260102"),
                )

    def test_narrow_loader_rejects_partition_marker_and_artifact_symlinks(self) -> None:
        for attack in (
            "partition",
            "marker",
            "artifact",
            "entry_partition",
            "entry_artifact",
        ):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                exit_root = root / "exit"
                entry_root = root / "entry"
                _make_exact_narrow_loader_fixture(exit_root, entry_root)
                partition = (
                    exit_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
                )
                if attack == "partition":
                    outside = root / "escaped-partition"
                    partition.rename(outside)
                    partition.symlink_to(outside, target_is_directory=True)
                    manifest_path = (
                        exit_root / "exit_maker_partition_manifest.parquet"
                    )
                    pl.read_parquet(manifest_path).with_columns(
                        pl.lit(str(outside)).alias("partition")
                    ).write_parquet(manifest_path)
                elif attack == "marker":
                    marker = partition / "complete.json"
                    outside = root / "escaped-complete.json"
                    marker.rename(outside)
                    marker.symlink_to(outside)
                else:
                    if attack == "artifact":
                        artifact = partition / "exit_maker_observations.parquet"
                        outside = root / "escaped-observations.parquet"
                    elif attack == "entry_partition":
                        artifact = (
                            entry_root
                            / "Date=20260102"
                            / f"ValueCode={VALUE_CODE}"
                        )
                        outside = root / "escaped-entry-partition"
                        artifact.rename(outside)
                        artifact.symlink_to(outside, target_is_directory=True)
                        artifact = None
                    else:
                        artifact = (
                            entry_root
                            / "Date=20260102"
                            / f"ValueCode={VALUE_CODE}"
                            / "execution_action_facts.parquet"
                        )
                        outside = root / "escaped-entry-actions.parquet"
                    if artifact is None:
                        pass
                    else:
                        artifact.rename(outside)
                        artifact.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "symlink|canonical"):
                    _load_narrow_formal_partition_inputs(
                        exit_root,
                        entry_root,
                        sessions=1,
                        value_codes=None,
                        session_calendar=("20260101", "20260102"),
                    )

    def test_prerequisite_rejects_byte_identical_entry_root_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prerequisite = root / "prerequisite"
            entry = root / "entry"
            clone = root / "entry-clone"
            prerequisite.mkdir()
            entry.mkdir()
            clone.mkdir()
            manifest = entry / "execution_partition_manifest.parquet"
            manifest.write_bytes(b"same immutable manifest bytes")
            (clone / manifest.name).write_bytes(manifest.read_bytes())
            identity = {
                "root": str(prerequisite),
                "source_identity": {
                    "entry_execution_root": str(entry),
                    "product_days": {"path": str(manifest)},
                },
            }
            _validate_prerequisite_source_root_binding(
                identity,
                entry_root=entry.resolve(),
                prerequisite_root=prerequisite.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "differs from prerequisite"):
                _validate_prerequisite_source_root_binding(
                    identity,
                    entry_root=clone.resolve(),
                    prerequisite_root=prerequisite.resolve(),
                )

    def test_cross_embedded_sources_reject_clone_repoint_and_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            cross_root = root / "cross"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            inputs = _load_narrow_formal_partition_inputs(
                exit_root,
                entry_root,
                sessions=1,
                value_codes=None,
                session_calendar=("20260101", "20260102"),
            )
            date = "20260102"
            entry_partition = (
                entry_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
            )
            same_day_partition = (
                exit_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
            )
            same_day_payload = json.loads(
                (same_day_partition / "complete.json").read_text(encoding="utf-8")
            )
            record = _current_formal_source_binding_record(
                date=date,
                value_code=VALUE_CODE,
                entry_partition=entry_partition,
                same_day_partition=same_day_partition,
                same_day_payload=same_day_payload,
            )
            original_entry_identity = json.loads(
                json.dumps(record["entry_execution"])
            )
            cross_partition = (
                cross_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
            )
            cross_partition.mkdir(parents=True)
            config = {
                "source": {
                    "Date": date,
                    "ValueCode": VALUE_CODE,
                    "entry_execution": json.loads(
                        json.dumps(original_entry_identity)
                    ),
                    "same_day_exit_maker": record["same_day_exit_maker"],
                }
            }
            marker_path = cross_partition / "complete.json"

            def write_cross_marker() -> None:
                marker_path.write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "Date": date,
                            "ValueCode": VALUE_CODE,
                            "config": config,
                            "config_sha256": _canonical_sha256(config),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            write_cross_marker()
            expected_rows = inputs.metadata[
                "formal_partition_source_binding_rows"
            ]
            expected_sha = inputs.metadata[
                "formal_partition_source_binding_sha256"
            ]
            result = _validate_cross_embedded_source_bindings(
                cross_root.resolve(),
                entry_root.resolve(),
                exit_root.resolve(),
                inputs.coverage,
                expected_rows=expected_rows,
                expected_sha256=expected_sha,
            )
            self.assertTrue(result["cross_embedded_formal_sources_reconciled"])

            clone_root = root / "entry-clone"
            shutil.copytree(entry_root, clone_root)
            config["source"]["entry_execution"]["partition"] = str(
                clone_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
            )
            write_cross_marker()
            with self.assertRaisesRegex(ValueError, "path differs"):
                _validate_cross_embedded_source_bindings(
                    cross_root.resolve(),
                    entry_root.resolve(),
                    exit_root.resolve(),
                    inputs.coverage,
                    expected_rows=expected_rows,
                    expected_sha256=expected_sha,
                )

            config["source"]["entry_execution"] = json.loads(
                json.dumps(original_entry_identity)
            )
            config["source"]["entry_execution"]["marker_sha256"] = "0" * 64
            write_cross_marker()
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                _validate_cross_embedded_source_bindings(
                    cross_root.resolve(),
                    entry_root.resolve(),
                    exit_root.resolve(),
                    inputs.coverage,
                    expected_rows=expected_rows,
                    expected_sha256=expected_sha,
                )

            config["source"]["entry_execution"] = json.loads(
                json.dumps(original_entry_identity)
            )
            config["source"]["entry_execution"]["artifacts"][
                "execution_action_facts.parquet"
            ] = "0" * 64
            write_cross_marker()
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                _validate_cross_embedded_source_bindings(
                    cross_root.resolve(),
                    entry_root.resolve(),
                    exit_root.resolve(),
                    inputs.coverage,
                    expected_rows=expected_rows,
                    expected_sha256=expected_sha,
                )

            config["source"]["entry_execution"] = json.loads(
                json.dumps(original_entry_identity)
            )
            config["source"]["same_day_exit_maker"] = json.loads(
                json.dumps(record["same_day_exit_maker"])
            )
            config["source"]["same_day_exit_maker"]["marker_sha256"] = "0" * 64
            write_cross_marker()
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                _validate_cross_embedded_source_bindings(
                    cross_root.resolve(),
                    entry_root.resolve(),
                    exit_root.resolve(),
                    inputs.coverage,
                    expected_rows=expected_rows,
                    expected_sha256=expected_sha,
                )

    def test_narrow_loader_rejects_unused_hash_and_coordinated_schema_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            observation = (
                exit_root
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / "exit_maker_observations.parquet"
            )
            observation.write_bytes(observation.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                _load_narrow_formal_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                    value_codes=None,
                    session_calendar=("20260101", "20260102"),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _make_exact_narrow_loader_fixture(exit_root, entry_root)
            partition = exit_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
            observation = partition / "exit_maker_observations.parquet"
            pl.DataFrame({"wrong": [1]}).write_parquet(observation)
            marker_path = partition / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"]["exit_maker_observations.parquet"] = {
                "rows": 1,
                "columns": 1,
                "bytes": observation.stat().st_size,
                "sha256": _file_sha256(observation),
            }
            marker_path.write_text(
                json.dumps(marker, sort_keys=True), encoding="utf-8"
            )
            manifest_path = exit_root / "exit_maker_partition_manifest.parquet"
            pl.read_parquet(manifest_path).with_columns(
                pl.lit(1).alias("exit_maker_observations_rows")
            ).write_parquet(manifest_path)
            with self.assertRaisesRegex(ValueError, "artifact schema mismatch"):
                _load_narrow_formal_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                    value_codes=None,
                    session_calendar=("20260101", "20260102"),
                )

    def test_formal_coverage_accepts_2687_complete_inside_2700_grid(self) -> None:
        rows = [
            {
                "Date": f"D{date:02d}",
                "ValueCode": f"V{product:02d}",
                "partition_complete": index < 2_687,
            }
            for index, (date, product) in enumerate(
                (date, product)
                for date in range(60)
                for product in range(45)
            )
        ]
        coverage = pl.from_dicts(rows)
        metadata = {
            "selected_session_count": 60,
            "selected_product_count": 45,
            "selected_product_day_count": 2_687,
            "expected_product_day_count": 2_700,
            "missing_product_day_count": 13,
        }
        complete, missing = _validate_formal_coverage(
            coverage,
            metadata,
            expected_sessions=60,
            expected_product_count=45,
            expected_complete_product_days=2_687,
            expected_grid_product_days=2_700,
            expected_missing_product_days=13,
        )
        self.assertEqual(complete.height, 2_687)
        self.assertEqual(missing.height, 13)
        with self.assertRaisesRegex(ValueError, "coverage grid"):
            _validate_formal_coverage(
                complete,
                metadata,
                expected_sessions=60,
                expected_product_count=45,
                expected_complete_product_days=2_687,
                expected_grid_product_days=2_700,
                expected_missing_product_days=13,
            )

    def test_missing_cost_and_censor_never_fabricate_ev_or_selection(self) -> None:
        evaluation = build_post_cross_position_evaluation(
            _sources(censored=True),
            config=_config(),
            cost_sensitivities=(CostSensitivity("sensitivity", 19.0, 34.0),),
            position_limits=(
                PositionLimit("one", max_concurrent_positions=1),
            ),
        )
        self.assertEqual(evaluation.policy_summary.height, 24)
        self.assertFalse(
            evaluation.policy_summary["alternative_policy_rows_additive"].any()
        )
        self.assertFalse(
            evaluation.prequential_rankings[
                "nominal_post_fill_ev_ready"
            ].any()
        )
        self.assertFalse(
            evaluation.prequential_decisions["selected_action_ev_ready"].any()
        )
        selected_columns = [
            "selected_boundary_quantile",
            "selected_entry_route",
            "selected_exit_rule_id",
            "selected_exit_route",
        ]
        for column in selected_columns:
            self.assertEqual(
                evaluation.prequential_decisions[column].drop_nulls().len(), 0
            )
        readiness = {
            row["gate"]: row["go"]
            for row in evaluation.readiness.iter_rows(named=True)
        }
        self.assertTrue(readiness["descriptive_terminal_report_go"])
        self.assertFalse(readiness["terminal_cashflow_point_identified"])
        self.assertFalse(readiness["full_cost_profile_go"])
        self.assertFalse(readiness["best_q_selection_go"])
        self.assertFalse(readiness["production_strategy_go"])

    def test_terminal_date_cashflow_outstanding_and_limit_sweep(self) -> None:
        evaluation = build_post_cross_position_evaluation(
            _sources(censored=False),
            config=_config(),
            cost_sensitivities=(CostSensitivity("sensitivity", 19.0, 34.0),),
            position_limits=(PositionLimit("one", max_concurrent_positions=1),),
        )
        terminal = evaluation.daily_terminal_cashflows.filter(
            (pl.col("boundary_quantile") == 95)
            & (pl.col("entry_route") == SPOT_ENTRY)
            & (pl.col("exit_rule_id") == "frozen_lower")
            & (pl.col("exit_route") == SPOT_EXIT)
        )
        self.assertEqual(
            terminal["terminal_completed_cycles"].sum(), len(ENTRY_SESSIONS)
        )
        self.assertFalse(terminal["unresolved_cashflow_imputed"].any())

        outstanding = evaluation.daily_outstanding.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("entry_route") == FUTURE_ENTRY)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_EXIT)
        )
        self.assertGreater(outstanding["outstanding_eod_positions"].max(), 0)
        self.assertFalse(outstanding["notional_is_capital_requirement"].any())

        sweep = evaluation.position_limit_sweep.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("entry_route") == FUTURE_ENTRY)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).row(0, named=True)
        self.assertGreater(sweep["rejected_positions"], 0)
        self.assertLessEqual(sweep["peak_concurrent_positions"], 1)
        self.assertFalse(sweep["position_limit_sweep_production_ready"])

    def test_portfolio_and_per_product_limit_scopes_are_distinct(self) -> None:
        sources = _sources(censored=False)
        second = sources.policy_paths.with_columns(
            pl.lit("2317").alias("ValueCode"),
            pl.lit("DHG1").alias("QuoteCode"),
            pl.concat_str(pl.lit("2317/"), pl.col("entry_raw_order_fact_id")).alias(
                "entry_raw_order_fact_id"
            ),
            pl.concat_str(pl.lit("2317/"), pl.col("exit_policy_trial_id")).alias(
                "exit_policy_trial_id"
            ),
            pl.concat_str(
                pl.lit("2317/"), pl.col("physical_entry_dependency_id")
            ).alias("physical_entry_dependency_id"),
        )
        doubled = FormalPostCrossSources(
            report=sources.report,
            policy_paths=pl.concat([sources.policy_paths, second]),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        evaluation = build_post_cross_position_evaluation(
            doubled,
            config=_config(),
            position_limits=(
                PositionLimit("portfolio_one", 1, None, "portfolio"),
                PositionLimit("product_one", 1, None, "per_value_code"),
            ),
        )
        cell = evaluation.position_limit_sweep.filter(
            (pl.col("boundary_quantile") == 95)
            & (pl.col("entry_route") == SPOT_ENTRY)
            & (pl.col("exit_rule_id") == "frozen_lower")
            & (pl.col("exit_route") == SPOT_EXIT)
        )
        portfolio = cell.filter(pl.col("limit_id") == "portfolio_one").row(
            0, named=True
        )
        product = cell.filter(pl.col("limit_id") == "product_one").row(
            0, named=True
        )
        self.assertGreater(
            product["accepted_positions"], portfolio["accepted_positions"]
        )
        self.assertEqual(product["peak_single_product_concurrent_positions"], 1)
        self.assertEqual(product["peak_concurrent_positions"], 2)

    def test_caller_built_cost_profile_can_never_unlock_ev(self) -> None:
        sources = _sources(censored=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cost_root = root / "cost"
            _write_cost_profile(cost_root, sources)
            with self.assertRaisesRegex(
                ValueError, "not a formal source-bound component producer"
            ):
                build_post_cross_position_evaluation(
                    sources,
                    config=_config(),
                    cost_sensitivities=(
                        CostSensitivity("sensitivity", 19.0, 34.0),
                    ),
                    position_limits=(
                        PositionLimit(
                            "one_hundred", max_concurrent_positions=100
                        ),
                    ),
                    full_cost_profile_root=cost_root,
                )
            output = root / "must_not_exist"
            with self.assertRaisesRegex(
                ValueError, "not a formal source-bound component producer"
            ):
                run_post_cross_position_evaluation(
                    exit_maker_root=root / "missing-exit",
                    entry_execution_root=root / "missing-entry",
                    cross_session_root=root / "missing-cross",
                    prerequisite_root=root / "missing-prerequisite",
                    output=output,
                    full_cost_profile_root=cost_root,
                    config=_config(),
                )
            self.assertFalse(output.exists())
        evaluation = build_post_cross_position_evaluation(
            sources,
            config=_config(),
            position_limits=(PositionLimit("one_hundred", 100),),
        )
        self.assertFalse(
            evaluation.prequential_rankings["nominal_post_fill_ev_ready"].any()
        )
        self.assertFalse(
            evaluation.prequential_decisions["selected_action_ev_ready"].any()
        )
        self.assertFalse(
            evaluation.metadata["formal_component_cost_source_bound"]
        )

    def test_rehashed_caller_cost_profile_still_fails_closed(self) -> None:
        sources = _sources(censored=False)
        with tempfile.TemporaryDirectory() as temporary:
            cost_root = Path(temporary) / "cost"
            _write_cost_profile(cost_root, sources)
            marker_path = cost_root / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["bound_cross_session_manifest_sha256"] = "c" * 64
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "not a formal source-bound component producer"
            ):
                build_post_cross_position_evaluation(
                    sources,
                    config=_config(),
                    position_limits=(
                        PositionLimit("one", max_concurrent_positions=1),
                    ),
                    full_cost_profile_root=cost_root,
                )

    def test_strict_or_mixed_terminal_semantics_fail_closed(self) -> None:
        sources = _sources(censored=False)
        bad = FormalPostCrossSources(
            report=sources.report,
            policy_paths=sources.policy_paths.with_columns(
                pl.when(pl.col("boundary_quantile") == 50)
                .then(pl.lit("strict"))
                .otherwise(pl.col("cancel_semantics"))
                .alias("cancel_semantics")
            ),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        with self.assertRaisesRegex(ValueError, "nominal terminal paths only"):
            build_post_cross_position_evaluation(
                bad,
                config=_config(),
                position_limits=(
                    PositionLimit("one", max_concurrent_positions=1),
                ),
            )

        allocated = FormalPostCrossSources(
            report=sources.report,
            policy_paths=sources.policy_paths.with_columns(
                pl.lit(True).alias("joint_volume_allocated")
            ),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        with self.assertRaisesRegex(ValueError, "readiness declarations"):
            build_post_cross_position_evaluation(
                allocated,
                config=_config(),
                position_limits=(
                    PositionLimit("one", max_concurrent_positions=1),
                ),
            )

    def test_same_day_chronology_and_physical_dependency_fail_closed(self) -> None:
        sources = _sources(censored=False)
        same_day_trial = sources.policy_paths.filter(
            pl.col("terminal_date") == pl.col("Date")
        ).item(0, "exit_policy_trial_id")
        chronology = FormalPostCrossSources(
            report=sources.report,
            policy_paths=sources.policy_paths.with_columns(
                pl.when(pl.col("exit_policy_trial_id") == same_day_trial)
                .then(pl.lit(999_999_999))
                .otherwise(pl.col("exit_decision_time_ns"))
                .alias("exit_decision_time_ns")
            ),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        with self.assertRaisesRegex(ValueError, "precedes position establishment"):
            build_post_cross_position_evaluation(
                chronology,
                config=_config(),
                position_limits=(PositionLimit("one", 1),),
            )

        first_date, second_date = ENTRY_SESSIONS[:2]
        first_dependency = sources.policy_paths.filter(
            (pl.col("Date") == first_date)
            & (pl.col("boundary_quantile") == 50)
            & (pl.col("entry_route") == FUTURE_ENTRY)
        ).item(0, "physical_entry_dependency_id")
        dependency_reuse = FormalPostCrossSources(
            report=sources.report,
            policy_paths=sources.policy_paths.with_columns(
                pl.when(
                    (pl.col("Date") == second_date)
                    & (pl.col("boundary_quantile") == 50)
                    & (pl.col("entry_route") == FUTURE_ENTRY)
                )
                .then(pl.lit(first_dependency))
                .otherwise(pl.col("physical_entry_dependency_id"))
                .alias("physical_entry_dependency_id")
            ),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        with self.assertRaisesRegex(ValueError, "unique within every policy cell"):
            build_post_cross_position_evaluation(
                dependency_reuse,
                config=_config(),
                position_limits=(PositionLimit("one", 1),),
            )

        incomplete_grid = FormalPostCrossSources(
            report=sources.report,
            policy_paths=sources.policy_paths.filter(
                pl.col("exit_policy_trial_id")
                != sources.policy_paths.item(0, "exit_policy_trial_id")
            ),
            session_calendar=sources.session_calendar,
            entry_sessions=sources.entry_sessions,
            metadata=sources.metadata,
        )
        with self.assertRaisesRegex(ValueError, "four-action exit grid"):
            build_post_cross_position_evaluation(
                incomplete_grid,
                config=_config(),
                position_limits=(PositionLimit("one", 1),),
            )

    def test_every_source_integrity_component_is_explicit_and_fail_closed(self) -> None:
        sources = _sources(censored=False)
        baseline = build_post_cross_position_evaluation(
            sources,
            config=_config(),
            position_limits=(PositionLimit("one", 1),),
        )
        baseline_gates = {
            row["gate"]: row["go"]
            for row in baseline.readiness.iter_rows(named=True)
        }
        source_gates = {
            name for name in baseline_gates if name.startswith("source_")
        }
        self.assertEqual(
            source_gates,
            {
                "source_coverage_contract_go",
                "source_cross_partition_hashes_go",
                "source_cross_policy_lineage_go",
                "source_prerequisite_binding_go",
                "source_d_minus_one_lineage_go",
                "source_entry_exit_foreign_key_go",
                "source_hedge_50ms_go",
                "source_complete_four_action_grid_go",
                "source_no_gross_imputation_go",
                "source_nominal_only_go",
                "source_implementation_identity_go",
                "source_integrity_go",
            },
        )
        self.assertTrue(all(baseline_gates[name] for name in source_gates))

        attacks = (
            ("formal_entry_product_days", 2_686, "source_coverage_contract_go"),
            (
                "cross_session_runner_config_sha256",
                "not-a-hash",
                "source_cross_partition_hashes_go",
            ),
            (
                "cross_session_policy_lineage_crossvalidated",
                False,
                "source_cross_policy_lineage_go",
            ),
            (
                "cross_session_formal_prerequisite_bound",
                False,
                "source_prerequisite_binding_go",
            ),
            (
                "cross_session_d_minus_one_lineage_validated",
                False,
                "source_d_minus_one_lineage_go",
            ),
            (
                "position_policy_to_bound_entry_exit_facts_crossvalidated",
                False,
                "source_entry_exit_foreign_key_go",
            ),
            ("exit_maker_hedge_delay_ns", 49_999_999, "source_hedge_50ms_go"),
            (
                "complete_center_lower_x_two_exit_routes_per_established_entry",
                False,
                "source_complete_four_action_grid_go",
            ),
            ("gross_zero_imputation", True, "source_no_gross_imputation_go"),
            (
                "strict_cross_session_outcomes_in_primary",
                True,
                "source_nominal_only_go",
            ),
            (
                "report_implementation_sources_sha256",
                "0" * 64,
                "source_implementation_identity_go",
            ),
        )
        for field, bad_value, expected_gate in attacks:
            with self.subTest(field=field):
                metadata = {**sources.metadata, field: bad_value}
                attacked = FormalPostCrossSources(
                    report=sources.report,
                    policy_paths=sources.policy_paths,
                    session_calendar=sources.session_calendar,
                    entry_sessions=sources.entry_sessions,
                    metadata=metadata,
                )
                evaluation = build_post_cross_position_evaluation(
                    attacked,
                    config=_config(),
                    position_limits=(PositionLimit("one", 1),),
                )
                gates = {
                    row["gate"]: row["go"]
                    for row in evaluation.readiness.iter_rows(named=True)
                }
                self.assertFalse(gates[expected_gate])
                self.assertFalse(gates["source_integrity_go"])
                self.assertFalse(gates["descriptive_terminal_report_go"])
                self.assertFalse(gates["position_limit_sweep_analysis_go"])

    def test_source_rebuild_rejects_rehashed_plausible_artifact_tamper(self) -> None:
        sources = _sources(censored=False)
        evaluation = build_post_cross_position_evaluation(
            sources,
            config=_config(),
            position_limits=(PositionLimit("one", 1),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            _publish_evaluation(evaluation, bundle)
            _verify_post_cross_position_evaluation_with_sources(bundle, sources)

            name = "policy_summary.parquet"
            path = bundle / name
            frame = pl.read_parquet(path).with_row_index("_row").with_columns(
                pl.when(pl.col("_row") == 0)
                .then(pl.col("completed_gross_twd") + 1.0)
                .otherwise(pl.col("completed_gross_twd"))
                .alias("completed_gross_twd")
            ).drop("_row")
            frame.write_parquet(path)
            marker_path = bundle / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"][name] = {
                "rows": frame.height,
                "columns": frame.width,
                "schema": {
                    column: str(dtype) for column, dtype in frame.schema.items()
                },
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            marker_path.write_text(
                json.dumps(marker, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source rebuild"):
                _verify_post_cross_position_evaluation_with_sources(bundle, sources)

    def test_exact_bundle_envelope_rejects_coordinated_audit_attack(self) -> None:
        sources = _sources(censored=False)
        evaluation = build_post_cross_position_evaluation(
            sources,
            config=_config(),
            position_limits=(PositionLimit("one", 1),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hidden_bundle = root / "hidden"
            _publish_evaluation(evaluation, hidden_bundle)
            (hidden_bundle / "hidden_stage.parquet").write_bytes(b"hidden")
            with self.assertRaisesRegex(ValueError, "root inventory"):
                _verify_post_cross_position_evaluation_with_sources(
                    hidden_bundle, sources
                )

            extra_key_bundle = root / "extra-key"
            _publish_evaluation(evaluation, extra_key_bundle)
            marker_path = extra_key_bundle / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["undeclared"] = True
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker envelope"):
                _verify_post_cross_position_evaluation_with_sources(
                    extra_key_bundle, sources
                )

            coordinated = root / "coordinated"
            _publish_evaluation(evaluation, coordinated)
            attacks = {
                "daily_outstanding.parquet": (
                    "outstanding_eod_positions",
                    -999,
                ),
                "prequential_policy_rankings.parquet": (
                    "contains_target_day_outcome",
                    True,
                ),
            }
            marker_path = coordinated / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for name, (column, value) in attacks.items():
                path = coordinated / name
                frame = pl.read_parquet(path).with_columns(
                    pl.lit(value).alias(column)
                )
                frame.write_parquet(path)
                marker["artifacts"][name] = {
                    "rows": frame.height,
                    "columns": frame.width,
                    "schema": {
                        key: str(dtype) for key, dtype in frame.schema.items()
                    },
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            marker["metadata"]["pathwise_ev_ready"] = True
            marker["metadata"]["production_strategy_go"] = True
            marker["metadata"]["injected_key"] = True
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fail-closed flags"):
                _verify_post_cross_position_evaluation_with_sources(
                    coordinated, sources
                )

    def test_atomic_bundle_verifier_and_missing_formal_prerequisite_no_output(self) -> None:
        sources = _sources(censored=True)
        evaluation = build_post_cross_position_evaluation(
            sources,
            config=_config(),
            position_limits=(PositionLimit("one", max_concurrent_positions=1),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _publish_evaluation(evaluation, bundle)
            marker = _verify_post_cross_position_evaluation_with_sources(
                bundle, sources
            )
            self.assertTrue(marker["complete"])
            self.assertEqual(len(marker["artifacts"]), 9)
            with self.assertRaises(FileNotFoundError):
                # The public verifier has no source override and must rebuild
                # from the immutable formal roots recorded in the bundle.
                verify_post_cross_position_evaluation(bundle)

            missing_output = root / "must_not_exist"
            with self.assertRaises(FileNotFoundError):
                run_post_cross_position_evaluation(
                    exit_maker_root=root / "missing-exit",
                    entry_execution_root=root / "missing-entry",
                    cross_session_root=root / "missing-cross",
                    prerequisite_root=root / "missing-prerequisite",
                    output=missing_output,
                    config=_config(),
                )
            self.assertFalse(missing_output.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
