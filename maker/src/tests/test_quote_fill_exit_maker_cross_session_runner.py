from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import polars as pl

from maker.src.quote_fill import exit_maker_cross_session_runner as runner_module
from maker.src.quote_fill.exit_maker_cross_session import (
    CrossSessionExitMakerResult,
    CrossSessionExitMakerSession,
    _attempt_schema,
    _outcome_schema,
)
from maker.src.quote_fill.exit_maker_cross_session_runner import (
    CANDIDATE_CACHE_SCHEMA_VERSION,
    CROSS_SESSION_MANIFEST_NAME,
    CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
    _OUTPUT_SCHEMAS,
    _RUNNER_AUDIT_SCHEMA,
    CandidateSessionRequirement,
    CrossSessionRunnerConfig,
    _artifact_metadata as _runner_artifact_metadata,
    _canonical_sha256,
    _file_sha256,
    _load_candidate_date_batch,
    _load_same_day_result,
    _prepare_candidate_session_cache,
    _runner_audit,
    _schema_metadata,
    run_cross_session_exit_replay,
    verify_cross_session_sources,
    verify_cross_session_output_partition,
)
from maker.src.quote_fill.raw_tape import RawTapeDay


DATE = "20260601"
PRIOR = "20260529"
NEXT = "20260602"
LATER = "20260603"
VALUE = "2303"
QUOTE = "CCFF6"
EXIT_ROUTES = ("future_bid_spot_taker", "spot_ask_future_taker")
OUTPUT_ARTIFACTS = {
    "cross_session_strict_policy_outcomes.parquet",
    "cross_session_strict_session_attempts.parquet",
    "cross_session_nominal_policy_outcomes.parquet",
    "cross_session_nominal_session_attempts.parquet",
    "cross_session_runner_audit.parquet",
}


def _prerequisite_identity(root: Path) -> dict[str, object]:
    source_identity = {
        "candidate_sessions": {"sha256": "1" * 64},
        "product_days": {"sha256": "2" * 64},
    }
    return {
        "binding_version": CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
        "root": str((root / "prerequisite").resolve()),
        "schema_version": "cross_session_prerequisites_v1",
        "marker_sha256": "3" * 64,
        "marker_payload_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "source_identity": source_identity,
        "source_identity_sha256": _canonical_sha256(source_identity),
    }


def _actions(*, established: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE],
            "QuoteCode": [QUOTE],
            "route": ["future_ask_spot_taker"],
            "raw_order_fact_id": ["entry-raw-1"],
            "policy_generation_id": ["entry-policy-1"],
            "full_fill": [established],
            "any_fill": [established],
            "partial_fill": [False],
            "full_fill_recv_time_ns": [1_000_000_000 if established else None],
            "entry_hedge_status": ["executable" if established else None],
            "entry_hedge_decision_time_ns": [
                1_050_000_000 if established else None
            ],
            "entry_hedge_label_observed": [established],
            "entry_hedge_executable": [established],
        }
    )


def _real_typed_empty_actions() -> pl.DataFrame:
    """Match the narrow schema emitted by a real zero-action partition."""

    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "boundary_quantile": pl.Int64,
            "lookup_action_id": pl.String,
            "raw_order_fact_id": pl.String,
            "policy_generation_id": pl.String,
            "entry_execution_outcome": pl.String,
            "pathwise_ev_ready": pl.Boolean,
        }
    )


def _real_typed_empty_exit_rules() -> pl.DataFrame:
    """Match the narrow schema emitted by a real zero-exit partition."""

    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "raw_order_fact_id": pl.String,
            "policy_generation_id": pl.String,
            "exit_rule_id": pl.String,
            "branch_status": pl.String,
            "same_day_exit": pl.Boolean,
            "overnight_carry": pl.Boolean,
            "terminal_outcome": pl.Boolean,
            "needs_next_session_label": pl.Boolean,
            "gross_cycle_pnl_twd": pl.Float64,
            "eod_liquidation_gross_pnl_twd": pl.Float64,
            "pathwise_ev_ready": pl.Boolean,
        }
    )


def _exit_rules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE, DATE],
            "ValueCode": [VALUE, VALUE],
            "QuoteCode": [QUOTE, QUOTE],
            "route": ["future_ask_spot_taker"] * 2,
            "raw_order_fact_id": ["entry-raw-1"] * 2,
            "policy_generation_id": ["entry-policy-1"] * 2,
            "exit_rule_id": ["frozen_center", "frozen_lower"],
            "exit_threshold_basis_bp": [0.0, -10.0],
            "exit_rule_source_asof_date": ["20260529", "20260529"],
        }
    )


def _policy_support() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for rule, threshold in (("frozen_center", 0.0), ("frozen_lower", -10.0)):
        for exit_route in EXIT_ROUTES:
            trial = f"entry-policy-1/exit/{rule}/{exit_route}"
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE,
                    "QuoteCode": QUOTE,
                    "entry_route": "future_ask_spot_taker",
                    "entry_policy_generation_id": "entry-policy-1",
                    "entry_raw_order_fact_id": "entry-raw-1",
                    "exit_rule_id": rule,
                    "exit_route": exit_route,
                    "exit_policy_trial_id": trial,
                    "exit_threshold_basis_bp": threshold,
                    "exit_rule_source_asof_date": "20260529",
                    "position_status": "position_established",
                    "position_established_ns": 1_050_000_000,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _position_policy_facts(support: pl.DataFrame) -> pl.DataFrame:
    return support.with_columns(
        pl.lit("carry_at_eod_no_admission").alias(
            "nominal_instant_cancel_v0_branch"
        ),
        pl.lit("carry_at_eod_no_admission").alias("branch_status"),
        pl.lit(False).alias("terminal_outcome"),
        pl.lit(True).alias("needs_next_session_label"),
        pl.lit(None, dtype=pl.String).alias("exit_hedge_status"),
        pl.lit(None, dtype=pl.Float64).alias("exit_spot_price"),
        pl.lit(None, dtype=pl.Float64).alias("exit_future_price"),
        pl.lit(None, dtype=pl.Float64).alias("gross_cycle_pnl_twd"),
    )


def _artifact_metadata(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _write_sources(
    root: Path,
    *,
    established: bool = True,
    action_facts: pl.DataFrame | None = None,
    exit_rules: pl.DataFrame | None = None,
    policy_support: pl.DataFrame | None = None,
    position_policy_facts: pl.DataFrame | None = None,
) -> tuple[Path, Path]:
    entry_root = root / "entry"
    exit_root = root / "exit-maker"
    entry_partition = entry_root / f"Date={DATE}" / f"ValueCode={VALUE}"
    exit_partition = exit_root / f"Date={DATE}" / f"ValueCode={VALUE}"
    entry_partition.mkdir(parents=True)
    exit_partition.mkdir(parents=True)

    actions = (
        action_facts
        if action_facts is not None
        else _actions(established=established)
    )
    exits = exit_rules if exit_rules is not None else _exit_rules()
    action_path = entry_partition / "execution_action_facts.parquet"
    rule_path = entry_partition / "exit_facts.parquet"
    target_audit_path = entry_partition / "target_audit.parquet"
    actions.write_parquet(action_path)
    exits.write_parquet(rule_path)
    target_audit = pl.DataFrame(
        {"Date": [DATE], "ValueCode": [VALUE], "QuoteCode": [QUOTE]}
    )
    target_audit.write_parquet(target_audit_path)
    entry_config = {
        "runner": "synthetic-entry-v1",
        "hedge_delay_ns": 50_000_000,
    }
    entry_marker = {
        "complete": True,
        "Date": DATE,
        "ValueCode": VALUE,
        "runner_version": "synthetic-entry-v1",
        "config": entry_config,
        "config_sha256": _canonical_sha256(entry_config),
        "artifacts": {
            action_path.name: _artifact_metadata(action_path, actions),
            rule_path.name: _artifact_metadata(rule_path, exits),
            target_audit_path.name: _artifact_metadata(
                target_audit_path, target_audit
            ),
        },
    }
    (entry_partition / "complete.json").write_text(
        json.dumps(entry_marker, sort_keys=True) + "\n", encoding="utf-8"
    )

    support = policy_support if policy_support is not None else _policy_support()
    if (not established or actions.is_empty()) and policy_support is None:
        support = support.head(0)
    positions = (
        position_policy_facts
        if position_policy_facts is not None
        else _position_policy_facts(support)
    )
    exit_frames = {
        "exit_maker_policy_support.parquet": support,
        "exit_maker_observations.parquet": pl.DataFrame(
            {"Date": [DATE], "ValueCode": [VALUE]}
        ),
        "exit_maker_transitions.parquet": pl.DataFrame(
            {"Date": [DATE], "ValueCode": [VALUE]}
        ),
        "exit_maker_candidate_aliases.parquet": pl.DataFrame(
            {"Date": [DATE], "ValueCode": [VALUE]}
        ),
        "exit_maker_raw_candidate_facts.parquet": pl.DataFrame(
            {"Date": [DATE], "ValueCode": [VALUE]}
        ),
        "exit_maker_position_policy_facts.parquet": (
            positions
        ),
        "exit_maker_audit.parquet": pl.DataFrame(
            {"Date": [DATE], "ValueCode": [VALUE]}
        ),
    }
    exit_artifacts: dict[str, dict[str, object]] = {}
    for filename, frame in exit_frames.items():
        path = exit_partition / filename
        frame.write_parquet(path)
        exit_artifacts[filename] = _artifact_metadata(path, frame)
    exit_config = {
        "runner": {
            "runner_version": "synthetic-exit-maker-v1",
            "hedge_delay_ns": 50_000_000,
        },
        "source": {
            "Date": DATE,
            "ValueCode": VALUE,
            "upstream_config_sha256": entry_marker["config_sha256"],
            "action_source": {
                "artifact": action_path.name,
                "sha256": _file_sha256(action_path),
            },
            "exit_rule_source": {
                "artifact": rule_path.name,
                "kind": "entry_execution_exit_facts_v1",
                "sha256": _file_sha256(rule_path),
            },
        },
    }
    exit_marker = {
        "complete": True,
        "Date": DATE,
        "ValueCode": VALUE,
        "runner_version": "synthetic-exit-maker-v1",
        "config": exit_config,
        "config_sha256": _canonical_sha256(exit_config),
        "artifacts": exit_artifacts,
    }
    (exit_partition / "complete.json").write_text(
        json.dumps(exit_marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return entry_root, exit_root


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": [QUOTE],
            "expiry_session": [NEXT],
            "calendar_version": ["official-synthetic-v1"],
        }
    )


def _candidate_session(
    *,
    date: str = NEXT,
    value_code: str = VALUE,
    quote_code: str = QUOTE,
) -> CrossSessionExitMakerSession:
    mapping = pl.DataFrame(
        {
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
            "spot_ref_price": [100.0],
            "fut_ref_price": [101.0],
            "contract_size": [2_000.0],
        }
    )
    states = pl.DataFrame(
        {
            "Date": [date],
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
        }
    )
    trades = states.head(0)
    audit = pl.DataFrame(
        {
            "Date": [date],
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
            "market": ["spot"],
        }
    )
    return CrossSessionExitMakerSession(
        date=date,
        raw_tape=RawTapeDay(
            date,
            mapping,
            states,
            states,
            trades,
            trades,
            audit,
        ),
        spread_pair_clock=pl.DataFrame(
            {"Date": [date], "ValueCode": [value_code]}
        ),
        spot_ref_price=100.0,
        future_ref_price=101.0,
        ref_price_source_date=date,
        ref_price_source_version="synthetic-candidate-v1",
    )


def _result(cancel_semantics: str, *, empty: bool = False) -> CrossSessionExitMakerResult:
    if empty:
        outcomes = pl.DataFrame(schema=_outcome_schema())
        attempts = pl.DataFrame(schema=_attempt_schema())
    else:
        nominal = cancel_semantics == "nominal_instant_cancel_v0"
        outcomes = pl.from_dicts(
            [
                {
                    "Date": DATE,
                    "ValueCode": VALUE,
                    "QuoteCode": QUOTE,
                    "cancel_semantics": cancel_semantics,
                    "filled_entry_outcome_category": (
                        "completed" if nominal else "unknown"
                    ),
                    "terminal_branch": (
                        "cross_session_maker_exit" if nominal else None
                    ),
                }
            ],
            schema=_outcome_schema(),
            strict=True,
        )
        attempts = pl.from_dicts(
            [
                {
                    "exit_policy_trial_id": "synthetic-trial",
                    "entry_policy_generation_id": "entry-policy-1",
                    "entry_raw_order_fact_id": "entry-raw-1",
                    "exit_rule_id": "frozen_center",
                    "exit_route": "future_bid_spot_taker",
                    "session_date": DATE,
                    "session_ordinal": 0,
                }
            ],
            schema=_attempt_schema(),
            strict=True,
        )
    blank = pl.DataFrame()
    return CrossSessionExitMakerResult(
        policy_outcomes=outcomes,
        session_attempts=attempts,
        candidate_aliases=blank,
        observations=blank,
        transitions=blank,
        audit=blank,
    )


def _replay_side_effect(*args: object, **_kwargs: object) -> CrossSessionExitMakerResult:
    config = args[5]
    return _result(config.cancel_semantics)  # type: ignore[attr-defined]


class CrossSessionExitMakerRunnerTests(unittest.TestCase):
    def test_runner_audit_schema_is_stable_for_empty_loaded_and_failure(self) -> None:
        results = {
            semantics: _result(semantics)
            for semantics in ("strict", "nominal_instant_cancel_v0")
        }
        cases = (
            ((), (), None, None),
            ((NEXT,), (NEXT,), None, None),
            (
                (NEXT, LATER),
                (NEXT,),
                "candidate_raw_missing",
                "synthetic missing candidate source",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (candidates, loaded, status, detail) in enumerate(cases):
                audit = _runner_audit(
                    DATE,
                    VALUE,
                    QUOTE,
                    results,
                    candidate_dates=candidates,
                    loaded_sessions=loaded,
                    load_failure_status=status,
                    load_failure_detail=detail,
                )
                self.assertEqual(
                    list(audit.schema.items()),
                    list(_RUNNER_AUDIT_SCHEMA.items()),
                )
                self.assertEqual(
                    audit["loaded_candidate_sessions"].dtype,
                    pl.List(pl.String),
                )
                self.assertEqual(audit["load_failure_status"].dtype, pl.String)
                self.assertEqual(audit["load_failure_detail"].dtype, pl.String)
                path = Path(directory) / f"audit-{index}.parquet"
                audit.write_parquet(path)
                paths.append(path)
            combined = pl.scan_parquet(paths).collect()
            self.assertEqual(
                list(combined.schema.items()),
                list(_RUNNER_AUDIT_SCHEMA.items()),
            )
            self.assertEqual(combined.height, 6)

    def test_zero_and_nonzero_partitions_have_concat_stable_output_schemas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def publish(label: str, *, established: bool) -> Path:
                source_root = root / f"{label}-source"
                entry_root, exit_root = _write_sources(
                    source_root, established=established
                )
                output_root = root / f"{label}-output"
                with patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    side_effect=_replay_side_effect,
                ):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=output_root,
                        candidate_session_loader=(
                            (lambda *_: _candidate_session())
                            if established
                            else Mock(
                                side_effect=AssertionError(
                                    "zero-entry loader must not run"
                                )
                            )
                        ),
                    )
                partition = (
                    output_root
                    / f"Date={DATE}"
                    / f"ValueCode={VALUE}"
                )
                verify_cross_session_output_partition(partition)
                return partition

            zero = publish("zero", established=False)
            nonzero = publish("nonzero", established=True)
            for filename, schema in _OUTPUT_SCHEMAS.items():
                combined = pl.scan_parquet(
                    [zero / filename, nonzero / filename]
                ).collect()
                self.assertEqual(
                    list(combined.schema.items()), list(schema.items())
                )

            audit_path = zero / "cross_session_runner_audit.parquet"
            audit = pl.read_parquet(audit_path).with_columns(
                pl.Series(
                    "loaded_candidate_sessions",
                    [[], []],
                    dtype=pl.List(pl.Null),
                )
            )
            audit.write_parquet(audit_path)
            marker_path = zero / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"][audit_path.name] = {
                **_runner_artifact_metadata(audit_path, audit),
                "schema": _schema_metadata(audit.schema),
            }
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "declared schema mismatch"):
                verify_cross_session_output_partition(zero)

    def test_cross_runner_rejects_coordinated_entry_delay_marker_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            entry_marker = (
                entry_root
                / f"Date={DATE}"
                / f"ValueCode={VALUE}"
                / "complete.json"
            )
            entry_payload = json.loads(entry_marker.read_text(encoding="utf-8"))
            entry_payload["config"]["hedge_delay_ns"] = 100_000_000
            entry_payload["config_sha256"] = _canonical_sha256(
                entry_payload["config"]
            )
            entry_marker.write_text(
                json.dumps(entry_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            exit_marker = (
                exit_root
                / f"Date={DATE}"
                / f"ValueCode={VALUE}"
                / "complete.json"
            )
            exit_payload = json.loads(exit_marker.read_text(encoding="utf-8"))
            exit_payload["config"]["source"]["upstream_config_sha256"] = (
                entry_payload["config_sha256"]
            )
            exit_payload["config_sha256"] = _canonical_sha256(
                exit_payload["config"]
            )
            exit_marker.write_text(
                json.dumps(exit_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "entry source hedge delay"):
                verify_cross_session_sources(
                    DATE,
                    VALUE,
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    config=CrossSessionRunnerConfig(),
                )

    def test_cross_runner_rejects_contradictory_entry_fill_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_actions = _actions().with_columns(
                pl.lit(False).alias("any_fill")
            )
            entry_root, exit_root = _write_sources(
                root,
                action_facts=bad_actions,
            )
            replayer = Mock()
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                with self.assertRaisesRegex(
                    ValueError, "fill flags are contradictory"
                ):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=root / "cross-session",
                        candidate_session_loader=Mock(),
                    )
            replayer.assert_not_called()
            self.assertFalse(
                (
                    root
                    / "cross-session"
                    / f"Date={DATE}"
                    / f"ValueCode={VALUE}"
                ).exists()
            )

    def test_shared_raw_no_fill_alias_is_not_an_established_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _actions()
            no_fill = base.with_columns(
                pl.lit("entry-policy-no-fill").alias("policy_generation_id"),
                pl.lit(False).alias("any_fill"),
                pl.lit(False).alias("full_fill"),
                pl.lit(False).alias("partial_fill"),
                pl.lit(None, dtype=pl.Int64).alias("full_fill_recv_time_ns"),
                # The canonical physical status/cursor remain populated.
                pl.lit(False).alias("entry_hedge_label_observed"),
                pl.lit(False).alias("entry_hedge_executable"),
            )
            actions = pl.concat([base, no_fill], how="vertical")
            no_fill_rules = _exit_rules().with_columns(
                pl.lit("entry-policy-no-fill").alias(
                    "policy_generation_id"
                )
            )
            exits = pl.concat([_exit_rules(), no_fill_rules], how="vertical")
            entry_root, exit_root = _write_sources(
                root,
                action_facts=actions,
                exit_rules=exits,
            )
            replayer = Mock(side_effect=_replay_side_effect)
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=root / "cross-session",
                    candidate_session_loader=lambda *_: _candidate_session(),
                )
            self.assertEqual(replayer.call_count, 2)
            self.assertEqual(replayer.call_args_list[0].args[0].height, 2)
            marker = json.loads(
                (
                    root
                    / "cross-session"
                    / f"Date={DATE}"
                    / f"ValueCode={VALUE}"
                    / "complete.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["config"]["established_entry_rows"], 1
            )

    def test_position_policy_lineage_tamper_fails_before_candidate_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = _policy_support()
            positions = _position_policy_facts(support).with_columns(
                (pl.col("exit_threshold_basis_bp") + 1.0).alias(
                    "exit_threshold_basis_bp"
                )
            )
            entry_root, exit_root = _write_sources(
                root,
                position_policy_facts=positions,
            )
            loader = Mock(return_value=_candidate_session())
            replayer = Mock()
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "position facts differ|threshold/source differs",
                ):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=root / "cross-session",
                        candidate_session_loader=loader,
                    )
            loader.assert_not_called()
            replayer.assert_not_called()

    def test_invalid_prerequisite_identity_fails_before_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _prerequisite_identity(root)
            identity["source_identity_sha256"] = "f" * 64
            output = root / "cross-session"
            with self.assertRaisesRegex(
                ValueError, "source identity hash mismatch"
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    output_root=output,
                    config=CrossSessionRunnerConfig(
                        prerequisite_identity=identity
                    ),
                    candidate_session_loader=Mock(),
                )
            self.assertFalse(output.exists())

            valid_identity = _prerequisite_identity(root)
            protected = Path(str(valid_identity["root"]))
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    output_root=protected,
                    config=CrossSessionRunnerConfig(
                        prerequisite_identity=valid_identity
                    ),
                    candidate_session_loader=Mock(),
                )
            self.assertFalse(protected.exists())

    def test_output_and_cache_roots_cannot_alias_canonical_raw_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            data_root = root / "raw-data"
            futures_root = root / "raw-futures"
            data_root.mkdir()
            futures_root.mkdir()
            data_alias = root / "raw-data-alias"
            futures_alias = root / "raw-futures-alias"
            data_alias.symlink_to(data_root, target_is_directory=True)
            futures_alias.symlink_to(futures_root, target_is_directory=True)

            cases = (
                (
                    "output equals canonical data root",
                    data_alias,
                    root / "safe-cache-1",
                    True,
                ),
                (
                    "output equals canonical futures root",
                    futures_alias,
                    root / "safe-cache-2",
                    True,
                ),
                (
                    "cache equals canonical data root",
                    root / "safe-output-1",
                    data_alias,
                    True,
                ),
                (
                    "disabled cache still cannot equal futures root",
                    root / "safe-output-2",
                    futures_alias,
                    False,
                ),
            )
            for label, output, cache, cache_enabled in cases:
                with self.subTest(label=label):
                    output_existed = output.exists()
                    cache_existed = cache.exists()
                    loader = Mock()
                    replayer = Mock()
                    with patch(
                        "maker.src.quote_fill.exit_maker_cross_session_runner."
                        "replay_cross_session_exit_maker",
                        replayer,
                    ):
                        with self.assertRaisesRegex(ValueError, "must be disjoint"):
                            run_cross_session_exit_replay(
                                ((DATE, VALUE),),
                                sessions=(PRIOR, DATE, NEXT),
                                contract_calendar=_calendar(),
                                entry_execution_root=entry_root,
                                exit_maker_root=exit_root,
                                output_root=output,
                                data_root=data_root,
                                futures_raw_root=futures_root,
                                candidate_cache_root=cache,
                                candidate_cache_enabled=cache_enabled,
                                candidate_session_loader=loader,
                            )
                    loader.assert_not_called()
                    replayer.assert_not_called()
                    self.assertEqual(output.exists(), output_existed)
                    self.assertEqual(cache.exists(), cache_existed)

    def test_policy_only_same_day_loader_does_not_materialize_support_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            source = verify_cross_session_sources(
                DATE,
                VALUE,
                entry_execution_root=entry_root,
                exit_maker_root=exit_root,
                config=CrossSessionRunnerConfig(),
            )
            result = _load_same_day_result(source, policy_only=True)
            self.assertFalse(result.position_policy_facts.is_empty())
            self.assertTrue(result.policy_support.is_empty())
            self.assertTrue(result.observations.is_empty())
            self.assertTrue(result.transitions.is_empty())
            self.assertTrue(result.candidate_aliases.is_empty())
            self.assertTrue(result.raw_candidate_facts.is_empty())
            self.assertTrue(result.audit.is_empty())

    def test_runner_shares_one_spooled_day_policy_cache_between_classifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            replayer = Mock(side_effect=_replay_side_effect)
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=root / "cross-session",
                    candidate_session_loader=lambda *_: _candidate_session(),
                )
            self.assertEqual(replayer.call_count, 2)
            first, second = replayer.call_args_list
            self.assertIs(
                first.kwargs["shared_day_policy_facts"],
                second.kwargs["shared_day_policy_facts"],
            )
            self.assertFalse(first.kwargs["retain_session_artifacts"])
            self.assertFalse(second.kwargs["retain_session_artifacts"])
            self.assertEqual(
                first.kwargs["day_policy_spool_root"],
                (root / "cross-session").resolve(),
            )
            self.assertEqual(
                first.args[5].lifecycle_policy_version,
                second.args[5].lifecycle_policy_version,
            )
            self.assertNotEqual(
                first.args[5].cancel_semantics,
                second.args[5].cancel_semantics,
            )

    def test_cache_on_off_publishes_identical_result_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            outputs: list[Path] = []
            for enabled in (False, True):
                output = root / f"cross-session-{enabled}"
                outputs.append(output)
                with patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    side_effect=_replay_side_effect,
                ):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=output,
                        candidate_session_loader=lambda *_: _candidate_session(),
                        candidate_cache_enabled=enabled,
                    )
            left = outputs[0] / f"Date={DATE}" / f"ValueCode={VALUE}"
            right = outputs[1] / f"Date={DATE}" / f"ValueCode={VALUE}"
            for filename in OUTPUT_ARTIFACTS:
                self.assertTrue(
                    pl.read_parquet(left / filename).equals(
                        pl.read_parquet(right / filename)
                    ),
                    filename,
                )

    def test_candidate_date_batch_opens_large_raw_once(self) -> None:
        requirements = (
            CandidateSessionRequirement(
                NEXT, "2303", "CCFF6", {"Date": NEXT}
            ),
            CandidateSessionRequirement(
                NEXT, "2317", "DXFF6", {"Date": NEXT}
            ),
        )
        mappings = {
            ("2303", "CCFF6"): pl.DataFrame(
                {
                    "ValueCode": ["2303"],
                    "QuoteCode": ["CCFF6"],
                    "spot_ref_price": [100.0],
                    "fut_ref_price": [101.0],
                    "contract_size": [2_000.0],
                }
            ),
            ("2317", "DXFF6"): pl.DataFrame(
                {
                    "ValueCode": ["2317"],
                    "QuoteCode": ["DXFF6"],
                    "spot_ref_price": [50.0],
                    "fut_ref_price": [51.0],
                    "contract_size": [2_000.0],
                }
            ),
        }
        combined_mapping = pl.concat(list(mappings.values()))
        states = pl.DataFrame(
            {
                "Date": [NEXT, NEXT],
                "ValueCode": ["2303", "2317"],
                "QuoteCode": ["CCFF6", "DXFF6"],
            }
        )
        tape = RawTapeDay(
            NEXT,
            combined_mapping,
            states,
            states,
            states.head(0),
            states.head(0),
            states.with_columns(pl.lit("spot").alias("market")),
        )
        raw_loader = Mock(return_value=tape)
        clock_loader = Mock(
            return_value=pl.DataFrame(
                {"Date": [NEXT, NEXT], "ValueCode": ["2303", "2317"]}
            )
        )

        def mapping_loader(
            _date: str, value_code: str, quote_code: str, **_kwargs: object
        ) -> tuple[pl.DataFrame, str]:
            return mappings[(value_code, quote_code)], "X"

        with (
            patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "load_candidate_session_mapping",
                side_effect=mapping_loader,
            ),
            patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "load_raw_tape_day",
                raw_loader,
            ),
            patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "load_spread_pair_clock_for_raw_day",
                clock_loader,
            ),
        ):
            result = _load_candidate_date_batch(
                requirements,
                data_root=Path("/synthetic/data"),
                futures_raw_root=Path("/synthetic/futures"),
                contract_metadata_root=Path("/synthetic/contracts"),
            )
        self.assertEqual(set(result), {item.identity for item in requirements})
        self.assertTrue(
            all(isinstance(value, CrossSessionExitMakerSession) for value in result.values())
        )
        raw_loader.assert_called_once()
        clock_loader.assert_called_once()
        self.assertEqual(
            set(clock_loader.call_args.args[1]), {"2303", "2317"}
        )

    def test_candidate_cache_source_change_uses_a_new_bound_key(self) -> None:
        first = CandidateSessionRequirement(
            NEXT,
            VALUE,
            QUOTE,
            {"Date": NEXT, "future_raw": {"mtime_ns": 1}},
        )
        changed = CandidateSessionRequirement(
            NEXT,
            VALUE,
            QUOTE,
            {"Date": NEXT, "future_raw": {"mtime_ns": 2}},
        )
        config = CrossSessionRunnerConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_loader = Mock()

            def resolve(
                rows: tuple[CandidateSessionRequirement, ...], **_kwargs: object
            ) -> dict[tuple[str, str, str], CrossSessionExitMakerSession]:
                return {
                    item.identity: _candidate_session()
                    for item in rows
                }

            batch_loader.side_effect = resolve
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "_load_candidate_date_batch",
                batch_loader,
            ):
                one = _prepare_candidate_session_cache(
                    (first,),
                    cache_root=root,
                    config=config,
                    data_root=root,
                    futures_raw_root=root,
                    contract_metadata_root=root,
                    custom_loader=None,
                )
                again = _prepare_candidate_session_cache(
                    (first,),
                    cache_root=root,
                    config=config,
                    data_root=root,
                    futures_raw_root=root,
                    contract_metadata_root=root,
                    custom_loader=None,
                )
                two = _prepare_candidate_session_cache(
                    (changed,),
                    cache_root=root,
                    config=config,
                    data_root=root,
                    futures_raw_root=root,
                    contract_metadata_root=root,
                    custom_loader=None,
                )
            self.assertEqual(batch_loader.call_count, 2)
            first_path = one[first.identity].cache_partition
            self.assertEqual(first_path, again[first.identity].cache_partition)
            self.assertNotEqual(first_path, two[changed.identity].cache_partition)
            assert first_path is not None
            marker = json.loads(
                (first_path / "complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["schema_version"], CANDIDATE_CACHE_SCHEMA_VERSION
            )

    def test_warm_cache_restats_source_mutated_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            data_root = root / "data"
            futures_root = root / "futures"
            metadata_root = root / "metadata"
            source_paths = (
                data_root / "tickData" / f"{NEXT}_StockTick.parquet",
                futures_root
                / NEXT[:4]
                / NEXT[4:6]
                / NEXT[6:8]
                / "stock_futures.parquet",
                data_root / "tickFeature" / f"{NEXT}_tickFeature.parquet",
                data_root / "marketData" / f"{NEXT}_marketData.parquet",
                metadata_root / f"{NEXT}_contracts.parquet",
            )
            for path in source_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stable-source")

            cache_root = root / "candidate-cache"
            arguments = {
                "product_days": ((DATE, VALUE),),
                "sessions": (PRIOR, DATE, NEXT),
                "contract_calendar": _calendar(),
                "entry_execution_root": entry_root,
                "exit_maker_root": exit_root,
                "data_root": data_root,
                "futures_raw_root": futures_root,
                "contract_metadata_root": metadata_root,
                "candidate_cache_root": cache_root,
            }
            with (
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "_load_candidate_date_batch",
                    return_value={
                        (NEXT, VALUE, QUOTE): _candidate_session()
                    },
                ),
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    side_effect=_replay_side_effect,
                ),
            ):
                run_cross_session_exit_replay(
                    output_root=root / "cold-output", **arguments
                )

            original_get = runner_module._CandidateSessionRepository.get
            mutated = False

            def mutate_before_hit(
                repository: object, identity: tuple[str, str, str]
            ) -> object:
                nonlocal mutated
                if not mutated:
                    source_paths[0].write_bytes(b"mutated-source-is-larger")
                    mutated = True
                return original_get(repository, identity)  # type: ignore[arg-type]

            batch_loader = Mock()
            replayer = Mock()
            with (
                patch.object(
                    runner_module._CandidateSessionRepository,
                    "get",
                    new=mutate_before_hit,
                ),
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "_load_candidate_date_batch",
                    batch_loader,
                ),
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    replayer,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "candidate source changed after preflight"
                ):
                    run_cross_session_exit_replay(
                        output_root=root / "warm-output", **arguments
                    )
            self.assertTrue(mutated)
            batch_loader.assert_not_called()
            replayer.assert_not_called()

    def test_candidate_batch_refuses_two_exact_contracts_for_one_product(self) -> None:
        rows = (
            CandidateSessionRequirement(
                NEXT, VALUE, QUOTE, {"Date": NEXT}
            ),
            CandidateSessionRequirement(
                NEXT, VALUE, "CCFG6", {"Date": NEXT}
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact-contract substitution"):
            _load_candidate_date_batch(
                rows,
                data_root=Path("/synthetic/data"),
                futures_raw_root=Path("/synthetic/futures"),
                contract_metadata_root=Path("/synthetic/contracts"),
            )

    def test_cached_loader_stops_at_the_first_missing_candidate_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            loader = Mock(side_effect=FileNotFoundError("first-gap"))
            replacement = Mock(side_effect=lambda result, **_kwargs: result)
            with (
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    side_effect=_replay_side_effect,
                ),
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "_replace_open_horizon_censor",
                    replacement,
                ),
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT, LATER),
                    contract_calendar=pl.DataFrame(
                        {
                            "QuoteCode": [QUOTE],
                            "expiry_session": [LATER],
                            "calendar_version": ["official-synthetic-v1"],
                        }
                    ),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=root / "cross-session",
                    candidate_session_loader=loader,
                )
            loader.assert_called_once_with(NEXT, VALUE, QUOTE)
            self.assertEqual(replacement.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["status"]
                    == "right_censored_missing_raw_session"
                    for call in replacement.call_args_list
                )
            )

    def test_atomic_publish_resume_and_strict_nominal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            output_root = root / "cross-session"
            prerequisite_identity = _prerequisite_identity(root)
            loader = Mock(
                return_value=SimpleNamespace(date=NEXT, validate=lambda: None)
            )
            replayer = Mock(side_effect=_replay_side_effect)
            arguments = {
                "sessions": (PRIOR, DATE, NEXT),
                "contract_calendar": _calendar(),
                "entry_execution_root": entry_root,
                "exit_maker_root": exit_root,
                "output_root": output_root,
                "candidate_session_loader": loader,
                "config": CrossSessionRunnerConfig(
                    prerequisite_identity=prerequisite_identity
                ),
            }
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                first = run_cross_session_exit_replay(((DATE, VALUE),), **arguments)
                resumed = run_cross_session_exit_replay(((DATE, VALUE),), **arguments)

            self.assertEqual(first.height, 1)
            self.assertTrue(first.equals(resumed))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(replayer.call_count, 2)
            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            self.assertEqual(
                {path.name for path in partition.iterdir()},
                {"complete.json", *OUTPUT_ARTIFACTS},
            )
            self.assertTrue((output_root / CROSS_SESSION_MANIFEST_NAME).is_file())
            strict = pl.read_parquet(
                partition / "cross_session_strict_policy_outcomes.parquet"
            )
            nominal = pl.read_parquet(
                partition / "cross_session_nominal_policy_outcomes.parquet"
            )
            self.assertEqual(strict.item(0, "cancel_semantics"), "strict")
            self.assertEqual(
                strict.item(0, "filled_entry_outcome_category"), "unknown"
            )
            self.assertEqual(
                nominal.item(0, "cancel_semantics"),
                "nominal_instant_cancel_v0",
            )
            self.assertEqual(
                nominal.item(0, "filled_entry_outcome_category"), "completed"
            )

            marker = json.loads(
                (partition / "complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["prerequisite"], prerequisite_identity)
            self.assertEqual(
                marker["config"]["prerequisite"], prerequisite_identity
            )
            self.assertEqual(
                marker["config"]["runner"]["prerequisite_identity"],
                prerequisite_identity,
            )
            self.assertEqual(
                marker["config"]["runner"]["global_inputs"][
                    "prerequisite"
                ],
                prerequisite_identity,
            )
            self.assertEqual(
                marker["runner_config_sha256"],
                _canonical_sha256(marker["config"]["runner"]),
            )
            self.assertTrue(
                marker["fact_semantics"]["formal_prerequisite_bound"]
            )
            self.assertTrue(
                marker["fact_semantics"][
                    "embedded_runner_config_sha256_recomputed"
                ]
            )
            self.assertTrue(
                marker["fact_semantics"][
                    "all_prerequisite_identity_copies_consistent"
                ]
            )
            self.assertTrue(
                marker["fact_semantics"][
                    "output_and_cache_disjoint_from_raw_source_roots"
                ]
            )
            self.assertTrue(marker["config"]["frozen_rule_lineage_crossvalidated"])
            self.assertEqual(
                marker["config"]["source"]["entry_execution"]["artifacts"]
                ["execution_action_facts.parquet"],
                _file_sha256(
                    entry_root
                    / f"Date={DATE}"
                    / f"ValueCode={VALUE}"
                    / "execution_action_facts.parquet"
                ),
            )
            runner_sources = marker["config"]["runner"]["implementation_sources"]
            self.assertIn("targets.py", runner_sources)
            self.assertEqual(
                runner_sources["exit_maker_report.py"],
                _file_sha256(
                    Path(
                        __import__(
                            "maker.src.quote_fill.exit_maker_report",
                            fromlist=["__file__"],
                        ).__file__
                    )
                ),
            )
            self.assertEqual(
                runner_sources["exit_maker_cross_session_runner.py"],
                _file_sha256(
                    Path(
                        __import__(
                            "maker.src.quote_fill.exit_maker_cross_session_runner",
                            fromlist=["__file__"],
                        ).__file__
                    )
                ),
            )
            verified = verify_cross_session_output_partition(partition)
            self.assertEqual(verified["config_sha256"], marker["config_sha256"])
            self.assertEqual(
                verified["prerequisite_marker_sha256"],
                prerequisite_identity["marker_sha256"],
            )
            manifest = pl.read_parquet(
                output_root / CROSS_SESSION_MANIFEST_NAME
            )
            self.assertTrue(manifest.equals(first))
            self.assertEqual(
                manifest.item(0, "prerequisite_source_identity_sha256"),
                prerequisite_identity["source_identity_sha256"],
            )

    def test_verifier_and_resume_recompute_embedded_runner_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            output_root = root / "cross-session"
            prerequisite_identity = _prerequisite_identity(root)
            arguments = {
                "sessions": (PRIOR, DATE, NEXT),
                "contract_calendar": _calendar(),
                "entry_execution_root": entry_root,
                "exit_maker_root": exit_root,
                "output_root": output_root,
                "config": CrossSessionRunnerConfig(
                    prerequisite_identity=prerequisite_identity
                ),
            }
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                side_effect=_replay_side_effect,
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    candidate_session_loader=Mock(
                        return_value=SimpleNamespace(
                            date=NEXT, validate=lambda: None
                        )
                    ),
                    **arguments,
                )

            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            marker_path = partition / "complete.json"
            original_marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )
            stale_runner_sha = original_marker["runner_config_sha256"]
            tampered_identity = json.loads(
                json.dumps(prerequisite_identity)
            )
            tampered_identity["marker_sha256"] = "6" * 64

            # Exact independent-audit reproduction: coordinate the three
            # previously checked copies, recompute only the partition config
            # hash, and leave both the runner-global copy and runner SHA stale.
            three_copy_tamper = json.loads(json.dumps(original_marker))
            three_copy_tamper["prerequisite"] = tampered_identity
            three_copy_tamper["config"]["prerequisite"] = tampered_identity
            three_copy_tamper["config"]["runner"][
                "prerequisite_identity"
            ] = tampered_identity
            three_copy_tamper["config_sha256"] = _canonical_sha256(
                three_copy_tamper["config"]
            )
            marker_path.write_text(
                json.dumps(three_copy_tamper, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "prerequisite lineage mismatch"
            ):
                verify_cross_session_output_partition(partition)

            # Also coordinate the fourth copy.  Copy reconciliation now passes,
            # so canonical runner-hash recomputation must independently reject
            # the stale declared SHA used by resume/preflight.
            marker = json.loads(json.dumps(original_marker))
            marker["prerequisite"] = tampered_identity
            marker["config"]["prerequisite"] = tampered_identity
            marker["config"]["runner"][
                "prerequisite_identity"
            ] = tampered_identity
            marker["config"]["runner"]["global_inputs"][
                "prerequisite"
            ] = tampered_identity
            marker["config_sha256"] = _canonical_sha256(marker["config"])
            self.assertEqual(marker["runner_config_sha256"], stale_runner_sha)
            self.assertNotEqual(
                _canonical_sha256(marker["config"]["runner"]),
                stale_runner_sha,
            )
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "embedded runner config hash mismatch"
            ):
                verify_cross_session_output_partition(partition)

            resume_loader = Mock()
            resume_replayer = Mock()
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                resume_replayer,
            ):
                with self.assertRaisesRegex(
                    ValueError, "embedded runner config hash mismatch"
                ):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        candidate_session_loader=resume_loader,
                        **arguments,
                    )
            resume_loader.assert_not_called()
            resume_replayer.assert_not_called()

    def test_verifier_checks_runner_global_prerequisite_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            output_root = root / "cross-session"
            prerequisite_identity = _prerequisite_identity(root)
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                side_effect=_replay_side_effect,
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=output_root,
                    config=CrossSessionRunnerConfig(
                        prerequisite_identity=prerequisite_identity
                    ),
                    candidate_session_loader=Mock(
                        return_value=SimpleNamespace(
                            date=NEXT, validate=lambda: None
                        )
                    ),
                )

            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            marker_path = partition / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            divergent_identity = json.loads(
                json.dumps(prerequisite_identity)
            )
            divergent_identity["marker_sha256"] = "7" * 64
            marker["config"]["runner"]["global_inputs"][
                "prerequisite"
            ] = divergent_identity
            marker["runner_config_sha256"] = _canonical_sha256(
                marker["config"]["runner"]
            )
            marker["config_sha256"] = _canonical_sha256(marker["config"])
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "prerequisite lineage mismatch"
            ):
                verify_cross_session_output_partition(partition)

    def test_mid_publish_failure_leaves_no_partition_or_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            output_root = root / "cross-session"
            calls = 0

            def fail_second_metadata(
                path: Path, frame: pl.DataFrame
            ) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("synthetic publish failure")
                return _runner_artifact_metadata(path, frame)

            with (
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    side_effect=_replay_side_effect,
                ),
                patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "_artifact_metadata",
                    side_effect=fail_second_metadata,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "publish failure"):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=output_root,
                        candidate_session_loader=Mock(
                            return_value=SimpleNamespace(
                                date=NEXT, validate=lambda: None
                            )
                        ),
                    )
            date_root = output_root / f"Date={DATE}"
            self.assertFalse((date_root / f"ValueCode={VALUE}").exists())
            self.assertEqual(list(date_root.glob(".*.tmp-*")), [])

    def test_tampered_upstream_source_hash_fails_before_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root)
            action_path = (
                entry_root
                / f"Date={DATE}"
                / f"ValueCode={VALUE}"
                / "execution_action_facts.parquet"
            )
            _actions().with_columns(pl.lit("tampered").alias("extra")).write_parquet(
                action_path
            )
            loader = Mock()
            replayer = Mock()
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    run_cross_session_exit_replay(
                        ((DATE, VALUE),),
                        sessions=(PRIOR, DATE, NEXT),
                        contract_calendar=_calendar(),
                        entry_execution_root=entry_root,
                        exit_maker_root=exit_root,
                        output_root=root / "cross-session",
                        candidate_session_loader=loader,
                    )
            loader.assert_not_called()
            replayer.assert_not_called()

    def test_zero_established_skips_candidate_raw_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(root, established=False)
            loader = Mock(side_effect=AssertionError("raw loader must not run"))
            replayer = Mock(
                side_effect=lambda *args, **_kwargs: _result(
                    args[5].cancel_semantics, empty=True
                )
            )
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=root / "cross-session",
                    candidate_session_loader=loader,
                )
            loader.assert_not_called()
            replayer.assert_not_called()
            marker = json.loads(
                (
                    root
                    / "cross-session"
                    / f"Date={DATE}"
                    / f"ValueCode={VALUE}"
                    / "complete.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                marker["config"]["candidate_raw_io_skipped_zero_established"]
            )
            self.assertEqual(marker["config"]["candidate_sessions"], [])
            self.assertEqual(marker["config"]["candidate_source_fingerprints"], [])

    def test_real_typed_empty_actions_publish_without_outcome_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry_root, exit_root = _write_sources(
                root,
                action_facts=_real_typed_empty_actions(),
                exit_rules=_real_typed_empty_exit_rules(),
            )
            loader = Mock(side_effect=AssertionError("raw loader must not run"))
            replayer = Mock(side_effect=AssertionError("replay must not run"))
            output_root = root / "cross-session"
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_runner."
                "replay_cross_session_exit_maker",
                replayer,
            ):
                manifest = run_cross_session_exit_replay(
                    ((DATE, VALUE),),
                    sessions=(PRIOR, DATE, NEXT),
                    contract_calendar=_calendar(),
                    entry_execution_root=entry_root,
                    exit_maker_root=exit_root,
                    output_root=output_root,
                    candidate_session_loader=loader,
                )
            loader.assert_not_called()
            replayer.assert_not_called()
            self.assertEqual(manifest.height, 1)
            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            marker = json.loads(
                (partition / "complete.json").read_text(encoding="utf-8")
            )
            self.assertTrue(marker["complete"])
            self.assertEqual(marker["config"]["established_entry_rows"], 0)
            self.assertTrue(
                marker["config"]["candidate_raw_io_skipped_zero_established"]
            )
            for filename in (
                "cross_session_strict_policy_outcomes.parquet",
                "cross_session_strict_session_attempts.parquet",
                "cross_session_nominal_policy_outcomes.parquet",
                "cross_session_nominal_session_attempts.parquet",
            ):
                self.assertTrue(pl.read_parquet(partition / filename).is_empty())

    def test_frozen_rule_lineage_mismatches_fail_closed(self) -> None:
        threshold_mismatch = _policy_support().with_columns(
            pl.when(pl.col("exit_rule_id") == "frozen_lower")
            .then(pl.lit(-9.0))
            .otherwise(pl.col("exit_threshold_basis_bp"))
            .alias("exit_threshold_basis_bp")
        )
        null_support = _policy_support().with_columns(
            pl.when(pl.col("exit_rule_id") == "frozen_lower")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("exit_rule_source_asof_date"))
            .alias("exit_rule_source_asof_date")
        )
        future_sourced_rules = _exit_rules().with_columns(
            pl.when(pl.col("exit_rule_id") == "frozen_lower")
            .then(pl.lit(DATE))
            .otherwise(pl.col("exit_rule_source_asof_date"))
            .alias("exit_rule_source_asof_date")
        )
        missing_lower_support = _policy_support().filter(
            pl.col("exit_rule_id") == "frozen_center"
        )
        cases = (
            (
                "threshold mismatch",
                None,
                threshold_mismatch,
                "support threshold/source differs",
            ),
            (
                "null source",
                None,
                null_support,
                "contains null lineage",
            ),
            (
                "non point-in-time source",
                future_sourced_rules,
                None,
                "source must equal the exact session predecessor",
            ),
            (
                "missing lower support",
                None,
                missing_lower_support,
                "support grid differs from established entries",
            ),
        )
        for label, exits, support, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry_root, exit_root = _write_sources(
                    root, exit_rules=exits, policy_support=support
                )
                loader = Mock()
                replayer = Mock()
                with patch(
                    "maker.src.quote_fill.exit_maker_cross_session_runner."
                    "replay_cross_session_exit_maker",
                    replayer,
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        run_cross_session_exit_replay(
                            ((DATE, VALUE),),
                            sessions=(PRIOR, DATE, NEXT),
                            contract_calendar=_calendar(),
                            entry_execution_root=entry_root,
                            exit_maker_root=exit_root,
                            output_root=root / "cross-session",
                            candidate_session_loader=loader,
                        )
                loader.assert_not_called()
                replayer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
