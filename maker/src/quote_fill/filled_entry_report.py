"""Filled-entry-only primary reports for exit-maker policy research.

The profit denominator in this module starts *after* an entry maker order has
fully filled and its delayed taker hedge is executable.  Submitted orders that
did not fill, partial entries, entry-fill unknowns, and failed entry hedges are
kept in a separate execution-diagnostics table and never receive a zero PnL in
the primary report.

Same-day exit-maker partitions stop at the entry session.  Their nominal
instant-cancel V0 branch can therefore classify a path as completed same-day,
still open at EOD, or unknown.  A hash-verified cross-session nominal replay
may attach one normalized terminal row per ``exit_policy_trial_id`` and replace
the still-open/unknown classification without changing the report contract.
Strict cancel-race outcomes remain a separate diagnostic artifact.

Rows for different entry quantiles, exit rules, and exit routes are
counterfactual policies.  A physical entry dependency may appear in several
such rows; every output explicitly marks those rows as non-additive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from .exit_maker_report import (
    DEFAULT_SESSION_CALENDAR_PATH,
    EXPECTED_HEDGE_DELAY_NS,
    ExitMakerPartitionInputs,
    _expected_session_predecessors,
    _load_session_calendar,
    _validate_entry_action_execution_contract,
    _validate_position_rule_lineage,
    load_exit_maker_partition_inputs,
)


REPORT_VERSION = "filled_entry_primary_report_v2_alias_local_provenance_bound"
CROSS_SESSION_NOMINAL_ARTIFACT = (
    "cross_session_nominal_policy_outcomes.parquet"
)
CROSS_SESSION_STRICT_ARTIFACT = "cross_session_strict_policy_outcomes.parquet"
OUTCOME_COMPLETED = "completed"
OUTCOME_STILL_OPEN = "still_open"
OUTCOME_CENSORED = "censored"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_CATEGORIES = frozenset(
    {
        OUTCOME_COMPLETED,
        OUTCOME_STILL_OPEN,
        OUTCOME_CENSORED,
        OUTCOME_UNKNOWN,
    }
)
FROZEN_EXIT_RULE_IDS = ("frozen_center", "frozen_lower")
FROZEN_EXIT_ROUTES = (
    "future_bid_spot_taker",
    "spot_ask_future_taker",
)
FROZEN_ENTRY_ROUTES = (
    "future_ask_spot_taker",
    "spot_bid_future_taker",
)
FROZEN_BOUNDARY_QUANTILES = (50, 80, 95)


def _established_entry_expr() -> pl.Expr:
    """Select actual positions using policy-alias-local hedge facts."""

    return (
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
        & (pl.col("entry_hedge_executable") == True)  # noqa: E712
    )


def _report_implementation_sources() -> dict[str, str]:
    module_root = Path(__file__).parent
    return {
        name: _file_sha256(module_root / name)
        for name in (
            "filled_entry_report.py",
            "exit_maker_report.py",
            "exit_maker_cross_session_runner.py",
            "cross_session_prerequisite.py",
            "exit_maker_cross_session_cli.py",
        )
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

PRIMARY_POLICY_KEY = [
    "entry_route",
    "boundary_quantile",
    "exit_rule_id",
    "exit_route",
]
POOLED_12_CELL_KEY = [
    "boundary_quantile",
    "exit_rule_id",
    "exit_route",
]
PRODUCT_POLICY_KEY = ["ValueCode", *PRIMARY_POLICY_KEY]
EXECUTION_DIAGNOSTIC_KEY = ["ValueCode", "entry_route", "boundary_quantile"]

_POSITION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "position_status",
    "position_established_ns",
    "nominal_instant_cancel_v0_branch",
    "gross_cycle_pnl_twd",
}
_ACTION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "full_fill",
    "partial_fill",
    "any_fill",
    "cancel_required",
    "entry_hedge_status",
    "full_fill_recv_time_ns",
    "entry_hedge_decision_time_ns",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_spot_price",
    "entry_hedge_contract_size_shares",
}
_TERMINAL_REQUIRED = {
    "exit_policy_trial_id",
    "filled_entry_outcome_category",
    "gross_cycle_pnl_twd",
}
_FORMAL_NOMINAL_TERMINAL_REQUIRED = {
    "Date",
    "cancel_semantics",
    "nominal_cancel_model_assumption",
    "entry_no_fill_in_primary",
    "pathwise_ev_ready",
    "outcome_type",
    "terminal_cashflow_priced",
    "terminal_date",
    "terminal_reason",
}

_NOMINAL_SAME_DAY = "flat_same_day"
_NOMINAL_STILL_OPEN = frozenset(
    {
        "carry_at_eod_cancel_unconfirmed",
        "carry_at_eod_no_admission",
        "no_fill_before_cancel_request",
    }
)


@dataclass(frozen=True)
class FilledEntryPrimaryReport:
    """Auditable filled-entry paths, primary summaries, and diagnostics."""

    coverage: pl.DataFrame
    filled_entry_policy_paths: pl.DataFrame
    policy_summary: pl.DataFrame
    pooled_12_cell_summary: pl.DataFrame
    product_policy_summary: pl.DataFrame
    execution_diagnostics: pl.DataFrame
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class CrossSessionNominalTerminalFacts:
    """Validated nominal-V0 overlay loaded from a partitioned cross root."""

    terminal_policy_facts: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


def build_filled_entry_primary_report(
    inputs: ExitMakerPartitionInputs,
    *,
    terminal_policy_facts: pl.DataFrame | None = None,
) -> FilledEntryPrimaryReport:
    """Build a no-zero-imputation report conditional on established entries.

    ``terminal_policy_facts`` is an optional cross-session overlay.  It must be
    unique by ``exit_policy_trial_id`` and use the normalized categories in
    :data:`OUTCOME_CATEGORIES`.  It may be partial: supplied rows override
    same-day still-open/unknown rows, while an already-completed same-day row
    can only be repeated with the identical completed cashflow.
    """

    declared_overlay_semantics: list[str] = []
    if terminal_policy_facts is not None:
        _validate_primary_overlay_semantics(terminal_policy_facts)
        if "cancel_semantics" in terminal_policy_facts.columns:
            declared_overlay_semantics = sorted(
                str(value)
                for value in terminal_policy_facts["cancel_semantics"]
                .drop_nulls()
                .unique()
                .to_list()
            )
    actions = _normalise_actions(inputs.action_facts)
    positions = _normalise_positions(inputs.position_policy_facts)
    predecessor_map = inputs.metadata.get("session_predecessors")
    if not isinstance(predecessor_map, Mapping) or not predecessor_map:
        raise ValueError(
            "filled-entry primary requires exact session_predecessors metadata"
        )
    position_lineage = _validate_position_rule_lineage(
        positions,
        inputs.taker_exit_facts,
        expected_session_predecessors=predecessor_map,
    )
    population = _validate_complete_policy_population(actions, positions)
    paths = _build_policy_paths(
        actions,
        positions,
        terminal_policy_facts=terminal_policy_facts,
    )
    policy = _aggregate_primary(paths, PRIMARY_POLICY_KEY)
    pooled = _aggregate_primary(paths, POOLED_12_CELL_KEY)
    product = _aggregate_primary(paths, PRODUCT_POLICY_KEY)
    diagnostics = _build_execution_diagnostics(actions)

    dependency_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_raw_order_fact_id",
        "position_established_ns",
    ]
    dependency_quantiles = paths.select(
        *dependency_key, "boundary_quantile"
    ).unique().group_by(dependency_key).agg(
        pl.col("boundary_quantile").n_unique().alias("quantiles")
    )
    cross_q = dependency_quantiles.filter(pl.col("quantiles") > 1).height

    established = actions.filter(_established_entry_expr())
    implementation_sources = _report_implementation_sources()
    metadata = dict(inputs.metadata)
    metadata.update(
        {
            "report_version": REPORT_VERSION,
            "report_implementation_sources": implementation_sources,
            "report_implementation_sources_sha256": _canonical_sha256(
                implementation_sources
            ),
            "report_semantics": "profit_conditional_on_entry_full_fill_and_executable_50ms_hedge",
            "outcome_source": (
                "same_day_nominal_instant_cancel_v0"
                if terminal_policy_facts is None
                else "same_day_v0_with_normalized_cross_session_overlay"
            ),
            "entry_policy_aliases_all": actions.height,
            "entry_policy_aliases_full_hedged": established.height,
            "entry_physical_dependencies_full_hedged": established[
                "raw_order_fact_id"
            ].n_unique(),
            "primary_physical_policy_paths": paths.height,
            "cross_q_shared_physical_entry_dependencies": cross_q,
            "entry_no_fill_rows_in_primary": 0,
            "entry_partial_or_fill_unknown_rows_in_primary": 0,
            "profit_denominator": "physical_filled_entry_position_within_each_policy_cell",
            "gross_statistics_conditioning": "completed_cycles_only",
            "gross_zero_imputation": False,
            "unknown_or_censored_cashflow_imputed": False,
            "net_or_after_cost_statistics_present": False,
            "non_price_cost_profile_applied": False,
            "fees_tax_financing_overnight_cancel_emergency_cost_complete": False,
            "cancel_fields_in_profit_summary": False,
            "cancel_diagnostics_separate": True,
            "freshness_gate_applied_by_report": False,
            "book_age_is_diagnostic_only": True,
            "boundary_quantile_rows_safe_to_sum": False,
            "exit_rule_rows_safe_to_sum": False,
            "exit_route_rows_safe_to_sum": False,
            "entry_route_pooled_rows_are_descriptive_only": True,
            "alternative_policy_rows_additive": False,
            "terminal_overlay_cancel_semantics": (
                declared_overlay_semantics[0]
                if len(declared_overlay_semantics) == 1
                else None
            ),
            "strict_cross_session_outcomes_in_primary": (
                declared_overlay_semantics == ["strict"]
            ),
            "pathwise_ev_ready": False,
            "entry_action_hedge_delay_ns": EXPECTED_HEDGE_DELAY_NS,
            "exit_maker_hedge_delay_ns": _validated_exit_hedge_delay(
                inputs.metadata
            ),
            "entry_exit_hedge_delay_match": True,
            "position_rule_lineage": position_lineage,
            "position_policy_to_bound_entry_exit_facts_crossvalidated": True,
            "complete_policy_population": population,
            "complete_center_lower_x_two_exit_routes_per_established_entry": True,
        }
    )
    return FilledEntryPrimaryReport(
        coverage=inputs.coverage.sort(["Date", "ValueCode"]),
        filled_entry_policy_paths=paths.sort(
            [
                "Date",
                "ValueCode",
                "entry_route",
                "boundary_quantile",
                "exit_rule_id",
                "exit_route",
                "entry_raw_order_fact_id",
            ]
        ),
        policy_summary=policy.sort(PRIMARY_POLICY_KEY),
        pooled_12_cell_summary=pooled.sort(POOLED_12_CELL_KEY),
        product_policy_summary=product.sort(PRODUCT_POLICY_KEY),
        execution_diagnostics=diagnostics.sort(EXECUTION_DIAGNOSTIC_KEY),
        metadata=metadata,
    )


def run_filled_entry_primary_report(
    exit_maker_root: Path,
    entry_execution_root: Path,
    *,
    output_dir: Path | None = None,
    terminal_policy_facts_path: Path | None = None,
    cross_session_root: Path | None = None,
    prerequisite_root: Path | None = None,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
) -> FilledEntryPrimaryReport:
    """Validate existing partitions, build the report, and publish atomically."""

    if terminal_policy_facts_path is not None and cross_session_root is not None:
        raise ValueError(
            "terminal_policy_facts_path and cross_session_root are mutually exclusive"
        )
    if cross_session_root is None and prerequisite_root is not None:
        raise ValueError("prerequisite_root requires cross_session_root")
    expected_prerequisite: Mapping[str, object] | None = None
    session_calendar = _load_session_calendar(DEFAULT_SESSION_CALENDAR_PATH)
    if cross_session_root is not None:
        if not validate_hashes:
            raise ValueError(
                "cross-session primary overlay requires hash validation"
            )
        if prerequisite_root is None:
            raise ValueError(
                "formal cross-session primary overlay requires --prerequisite-root"
            )
        expected_prerequisite = _expected_formal_prerequisite_identity(
            prerequisite_root
        )
        session_calendar = _prerequisite_session_calendar(
            expected_prerequisite
        )

    inputs = load_exit_maker_partition_inputs(
        Path(exit_maker_root),
        Path(entry_execution_root),
        sessions=sessions,
        value_codes=value_codes,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=require_balanced_product_days,
        validate_hashes=validate_hashes,
        session_calendar=session_calendar,
        require_root_manifest=True,
    )
    overlay_metadata: Mapping[str, object] = {}
    if cross_session_root is not None:
        assert expected_prerequisite is not None
        overlay = load_cross_session_nominal_terminal_facts(
            Path(cross_session_root),
            expected_coverage=inputs.coverage,
            expected_position_policy_facts=inputs.position_policy_facts,
            expected_prerequisite_identity=expected_prerequisite,
        )
        terminal = overlay.terminal_policy_facts
        overlay_metadata = overlay.metadata
    elif terminal_policy_facts_path is not None:
        terminal = pl.read_parquet(Path(terminal_policy_facts_path))
        _validate_primary_overlay_semantics(
            terminal,
            require_nominal_declaration=True,
        )
    else:
        terminal = None
    report = build_filled_entry_primary_report(
        inputs,
        terminal_policy_facts=terminal,
    )
    if terminal_policy_facts_path is not None:
        terminal_path = Path(terminal_policy_facts_path)
        metadata = dict(report.metadata)
        metadata.update(
            {
                "terminal_policy_facts_path": str(terminal_path),
                "terminal_policy_facts_rows": terminal.height,
                "terminal_policy_facts_sha256": _file_sha256(terminal_path),
            }
        )
        report = replace(report, metadata=metadata)
    if cross_session_root is not None:
        metadata = dict(report.metadata)
        metadata.update(overlay_metadata)
        report = replace(report, metadata=metadata)
    count = int(inputs.metadata["selected_session_count"])
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(exit_maker_root) / f"filled_entry_report_{count}_sessions"
    )
    _publish_report(report, destination)
    return report


def _expected_formal_prerequisite_identity(
    prerequisite_root: Path,
) -> dict[str, object]:
    """Verify the exact formal prerequisite marker and derive its bound identity."""

    from .cross_session_prerequisite import verify_cross_session_prerequisites
    from .exit_maker_cross_session_cli import (
        _verify_formal_prerequisite_binding,
    )

    root = Path(prerequisite_root).resolve()
    payload = verify_cross_session_prerequisites(root)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("formal prerequisite marker lacks config")

    def configured_path(name: str) -> Path:
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"formal prerequisite config lacks {name}")
        return Path(value)

    return _verify_formal_prerequisite_binding(
        root,
        sessions_path=root / "candidate_sessions.txt",
        contract_calendar_path=root / "exact_contract_calendar_v1.parquet",
        product_days_path=configured_path("product_days_path"),
        entry_execution_root=configured_path("entry_execution_root"),
        data_root=configured_path("data_root"),
        futures_raw_root=configured_path("futures_raw_root"),
        contract_metadata_root=root / "metadata",
    )


def _prerequisite_session_calendar(
    prerequisite_identity: Mapping[str, object],
) -> tuple[str, ...]:
    source_identity = prerequisite_identity.get("source_identity")
    candidate = (
        source_identity.get("candidate_sessions")
        if isinstance(source_identity, Mapping)
        else None
    )
    if not isinstance(candidate, Mapping):
        raise ValueError("prerequisite identity lacks candidate session lineage")
    path_value = candidate.get("path")
    digest = candidate.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError("prerequisite candidate session identity is invalid")
    path = Path(path_value)
    if not path.is_file() or _file_sha256(path) != digest:
        raise ValueError("prerequisite candidate session artifact changed")
    return _load_session_calendar(path)


def _validate_exact_rule_source_predecessors(
    frame: pl.DataFrame,
    predecessors: Mapping[str, str],
    *,
    source: str,
) -> None:
    _require(
        frame,
        {"Date", "exit_rule_source_asof_date"},
        source,
    )
    expected = pl.DataFrame(
        {
            "Date": list(predecessors),
            "_expected_rule_source_asof_date": list(predecessors.values()),
        }
    )
    checked = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("exit_rule_source_asof_date").cast(pl.String),
    ).join(expected, on="Date", how="left", validate="m:1")
    if checked.filter(
        pl.col("_expected_rule_source_asof_date").is_null()
        | (
            pl.col("exit_rule_source_asof_date")
            != pl.col("_expected_rule_source_asof_date")
        ).fill_null(True)
    ).height:
        raise ValueError(
            f"{source} exit_rule_source_asof_date is not the exact session predecessor"
        )


def load_cross_session_nominal_terminal_facts(
    cross_session_root: Path,
    *,
    expected_coverage: pl.DataFrame,
    expected_position_policy_facts: pl.DataFrame,
    expected_prerequisite_identity: Mapping[str, object] | None,
) -> CrossSessionNominalTerminalFacts:
    """Load one hash-verified nominal overlay for every selected product-day.

    The strict cross-session outcome is deliberately never returned here.  It
    remains a separate identification diagnostic and must not silently replace
    the nominal-V0 terminal population used by the primary filled-entry report.
    """

    from .exit_maker_cross_session_runner import (
        CROSS_SESSION_MANIFEST_NAME,
        _normalise_prerequisite_identity,
        verify_cross_session_output_partition,
    )

    expected_prerequisite = _normalise_prerequisite_identity(
        expected_prerequisite_identity
    )
    if expected_prerequisite is None:
        raise ValueError(
            "formal cross-session primary requires an explicit prerequisite identity"
        )

    root = Path(cross_session_root)
    if not root.is_dir():
        raise FileNotFoundError(f"cross-session root does not exist: {root}")
    _require(
        expected_coverage,
        {"Date", "ValueCode", "partition_complete"},
        "expected same-day coverage",
    )
    _require(
        expected_position_policy_facts,
        _POSITION_REQUIRED,
        "expected same-day position policy facts",
    )
    selected_coverage = expected_coverage.filter(
        pl.col("partition_complete") == True  # noqa: E712
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    if selected_coverage.select("Date", "ValueCode").n_unique() != (
        selected_coverage.height
    ):
        raise ValueError("expected same-day coverage contains duplicate product-days")
    expected_keys = {
        (str(row["Date"]), str(row["ValueCode"]))
        for row in selected_coverage.iter_rows(named=True)
    }
    if not expected_keys:
        raise ValueError("expected same-day coverage has no complete product-days")
    predecessor_map = _expected_session_predecessors(
        _prerequisite_session_calendar(expected_prerequisite),
        sorted({date for date, _ in expected_keys}),
    )
    assert predecessor_map is not None
    _validate_exact_rule_source_predecessors(
        expected_position_policy_facts,
        predecessor_map,
        source="expected same-day position policies",
    )

    manifest_path = root / CROSS_SESSION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"cross-session root manifest does not exist: {manifest_path}"
        )
    manifest = pl.read_parquet(manifest_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    manifest_required = {
        "Date",
        "ValueCode",
        "runner_config_sha256",
        "config_sha256",
        "nominal_policy_outcome_rows",
        "prerequisite_root",
        "prerequisite_marker_sha256",
        "prerequisite_marker_payload_sha256",
        "prerequisite_config_sha256",
        "prerequisite_source_identity_sha256",
        "complete",
    }
    _require(manifest, manifest_required, "cross-session root manifest")
    if manifest.select("Date", "ValueCode").n_unique() != manifest.height:
        raise ValueError("cross-session root manifest contains duplicate product-days")
    manifest_rows = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in manifest.iter_rows(named=True)
    }
    missing = sorted(expected_keys - set(manifest_rows))
    if missing:
        rendered = ", ".join(f"{date}/{value}" for date, value in missing[:5])
        raise ValueError(
            "cross-session root does not cover every selected same-day partition: "
            + rendered
        )

    frames: list[pl.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    runner_hashes: set[str] = set()
    prerequisite_fields = {
        "prerequisite_root": str(expected_prerequisite["root"]),
        "prerequisite_marker_sha256": str(
            expected_prerequisite["marker_sha256"]
        ),
        "prerequisite_marker_payload_sha256": str(
            expected_prerequisite["marker_payload_sha256"]
        ),
        "prerequisite_config_sha256": str(
            expected_prerequisite["config_sha256"]
        ),
        "prerequisite_source_identity_sha256": str(
            expected_prerequisite["source_identity_sha256"]
        ),
    }
    for date, value_code in sorted(expected_keys):
        partition = root / f"Date={date}" / f"ValueCode={value_code}"
        verified = verify_cross_session_output_partition(partition)
        if (
            str(verified["Date"]) != date
            or str(verified["ValueCode"]) != value_code
        ):
            raise ValueError(
                f"cross-session partition identity mismatch: {partition}"
            )
        declared = manifest_rows[(date, value_code)]
        for column in (
            "runner_config_sha256",
            "config_sha256",
            "nominal_policy_outcome_rows",
            "complete",
        ):
            if declared[column] != verified[column]:
                raise ValueError(
                    f"cross-session root manifest disagrees on {column}: "
                    f"{date}/{value_code}"
                )
        for column, expected_value in prerequisite_fields.items():
            if (
                declared[column] != expected_value
                or verified[column] != expected_value
            ):
                raise ValueError(
                    "cross-session prerequisite identity mismatch on "
                    f"{column}: {date}/{value_code}"
                )
        runner_hashes.add(str(verified["runner_config_sha256"]))
        path = partition / CROSS_SESSION_NOMINAL_ARTIFACT
        frame = pl.read_parquet(path)
        _validate_primary_overlay_semantics(
            frame,
            require_nominal_declaration=True,
        )
        _validate_cross_partition_identity(frame, date, value_code, path)
        if frame.height != int(verified["nominal_policy_outcome_rows"]):
            raise ValueError(
                f"cross-session nominal row count disagrees with marker: {path}"
            )
        frames.append(frame)
        coverage_rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": str(partition),
                "nominal_policy_outcome_rows": frame.height,
                "partition_complete": True,
                "partition_hashes_validated": True,
            }
        )
    if len(runner_hashes) != 1:
        raise ValueError(
            "selected cross-session partitions do not share one runner config"
        )

    terminal = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
    if terminal.select("exit_policy_trial_id").n_unique() != terminal.height:
        raise ValueError(
            "cross-session nominal overlay duplicates exit_policy_trial_id"
        )
    _crossvalidate_terminal_policy_lineage(
        terminal,
        expected_position_policy_facts,
    )
    _validate_exact_rule_source_predecessors(
        terminal,
        predecessor_map,
        source="cross-session nominal outcomes",
    )
    coverage = pl.from_dicts(coverage_rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )
    metadata = {
        "cross_session_root": str(root),
        "cross_session_manifest_path": str(manifest_path),
        "cross_session_manifest_sha256": _file_sha256(manifest_path),
        "cross_session_runner_config_sha256": next(iter(runner_hashes)),
        "cross_session_selected_product_day_count": len(expected_keys),
        "cross_session_nominal_policy_rows": terminal.height,
        "terminal_overlay_cancel_semantics": "nominal_instant_cancel_v0",
        "nominal_instant_cancel_model_assumption": True,
        "cross_session_partition_hashes_validated": True,
        "cross_session_policy_lineage_crossvalidated": True,
        "cross_session_formal_prerequisite_bound": True,
        "cross_session_prerequisite_root": prerequisite_fields[
            "prerequisite_root"
        ],
        "cross_session_prerequisite_marker_sha256": prerequisite_fields[
            "prerequisite_marker_sha256"
        ],
        "cross_session_prerequisite_marker_payload_sha256": prerequisite_fields[
            "prerequisite_marker_payload_sha256"
        ],
        "cross_session_prerequisite_config_sha256": prerequisite_fields[
            "prerequisite_config_sha256"
        ],
        "cross_session_prerequisite_source_identity_sha256": prerequisite_fields[
            "prerequisite_source_identity_sha256"
        ],
        "cross_session_exact_session_predecessors": predecessor_map,
        "cross_session_d_minus_one_lineage_validated": True,
        "cross_session_terminal_and_censored_outcomes_separate": True,
        "cross_session_strict_outcomes_artifact": CROSS_SESSION_STRICT_ARTIFACT,
        "strict_cross_session_outcomes_in_primary": False,
        "pathwise_ev_ready": False,
    }
    return CrossSessionNominalTerminalFacts(
        terminal_policy_facts=terminal,
        coverage=coverage,
        metadata=metadata,
    )


def _validate_primary_overlay_semantics(
    frame: pl.DataFrame,
    *,
    require_nominal_declaration: bool = False,
) -> None:
    """Reject strict or mixed cross-session facts from the nominal primary."""

    _require(frame, _TERMINAL_REQUIRED, "terminal policy facts")
    if require_nominal_declaration:
        missing = sorted(
            _FORMAL_NOMINAL_TERMINAL_REQUIRED - set(frame.columns)
        )
        if missing:
            raise ValueError(
                "formal nominal terminal overlay is missing semantic fields: "
                f"{missing}"
            )
    if frame.is_empty():
        return
    if "cancel_semantics" in frame.columns:
        semantics = {
            str(value)
            for value in frame["cancel_semantics"].drop_nulls().unique().to_list()
        }
        if semantics != {"nominal_instant_cancel_v0"}:
            raise ValueError(
                "filled-entry primary terminal overlay must be nominal_instant_cancel_v0; "
                f"found {sorted(semantics)}"
            )
        if frame["cancel_semantics"].null_count():
            raise ValueError("cross-session nominal overlay has null cancel semantics")
    if "nominal_cancel_model_assumption" in frame.columns and frame.filter(
        (pl.col("nominal_cancel_model_assumption") != True).fill_null(  # noqa: E712
            True
        )
    ).height:
        raise ValueError(
            "cross-session nominal overlay must declare its cancel-model assumption"
        )
    if "entry_no_fill_in_primary" in frame.columns and frame.filter(
        (pl.col("entry_no_fill_in_primary") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError(
            "cross-session primary overlay includes a non-established entry"
        )
    if "pathwise_ev_ready" in frame.columns and frame.filter(
        (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError(
            "gross cross-session overlay must not advertise pathwise EV readiness"
        )

    category = pl.col("filled_entry_outcome_category")
    invalid_category = frame.filter(
        category.is_null() | ~category.is_in(sorted(OUTCOME_CATEGORIES))
    )
    if invalid_category.height:
        raise ValueError("terminal policy facts contain an invalid outcome category")
    completed = category == OUTCOME_COMPLETED
    gross = pl.col("gross_cycle_pnl_twd")
    if frame.filter(completed & (gross.is_null() | ~gross.is_finite())).height:
        raise ValueError(
            "completed terminal overlay paths require finite gross cashflow"
        )
    if frame.filter(~completed & gross.is_not_null()).height:
        raise ValueError(
            "unresolved terminal overlay paths must not carry imputed gross cashflow"
        )
    if "terminal_cashflow_priced" in frame.columns and frame.filter(
        (pl.col("terminal_cashflow_priced") != completed).fill_null(True)
    ).height:
        raise ValueError(
            "terminal_cashflow_priced disagrees with completed outcome category"
        )
    if "outcome_type" in frame.columns:
        invalid_outcome_type = frame.filter(
            pl.when(completed)
            .then(pl.col("outcome_type") != "terminal")
            .otherwise(pl.col("outcome_type") != "censored")
            .fill_null(True)
        )
        if invalid_outcome_type.height:
            raise ValueError(
                "outcome_type must keep priced completed cycles terminal and "
                "unresolved paths censored"
            )
    if "terminal_date" in frame.columns:
        terminal_date = pl.col("terminal_date").cast(pl.String)
        invalid_date = frame.filter(
            completed
            & (
                terminal_date.is_null()
                | ~terminal_date.str.contains(r"^\d{8}$")
                | (
                    terminal_date < pl.col("Date").cast(pl.String)
                    if "Date" in frame.columns
                    else pl.lit(False)
                )
            )
        )
        if invalid_date.height:
            raise ValueError(
                "completed terminal overlay requires terminal_date >= entry Date"
            )
        if frame.filter(~completed & terminal_date.is_not_null()).height:
            raise ValueError(
                "unresolved terminal overlay paths must have null terminal_date"
            )
    if "terminal_reason" in frame.columns and frame.filter(
        completed & pl.col("terminal_reason").is_null()
    ).height:
        raise ValueError("completed terminal overlay requires terminal_reason")


def _validate_cross_partition_identity(
    frame: pl.DataFrame,
    date: str,
    value_code: str,
    path: Path,
) -> None:
    _require(frame, {"Date", "ValueCode"}, f"cross-session outcomes {path}")
    if frame.is_empty():
        return
    if set(map(str, frame["Date"].unique().to_list())) != {date}:
        raise ValueError(f"cross-session outcome Date mismatch: {path}")
    if set(map(str, frame["ValueCode"].unique().to_list())) != {value_code}:
        raise ValueError(f"cross-session outcome ValueCode mismatch: {path}")


def _crossvalidate_terminal_policy_lineage(
    terminal: pl.DataFrame,
    positions: pl.DataFrame,
) -> None:
    identity = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
        "exit_route",
        "exit_policy_trial_id",
        "exit_rule_source_asof_date",
    ]
    _require(
        terminal,
        {*identity, "frozen_exit_threshold_basis_bp"},
        "cross-session nominal outcomes",
    )
    _require(
        positions,
        {*identity, "exit_threshold_basis_bp"},
        "same-day position policies",
    )
    if terminal.filter(
        pl.col("frozen_exit_threshold_basis_bp").is_null()
        | ~pl.col("frozen_exit_threshold_basis_bp").is_finite()
    ).height:
        raise ValueError("cross-session frozen exit threshold must be finite")
    if positions.filter(
        pl.col("exit_threshold_basis_bp").is_null()
        | ~pl.col("exit_threshold_basis_bp").is_finite()
    ).height:
        raise ValueError("same-day frozen exit threshold must be finite")
    expected = positions.select(
        *identity,
        pl.col("exit_threshold_basis_bp").alias(
            "_expected_frozen_exit_threshold_basis_bp"
        ),
    )
    if expected.select("exit_policy_trial_id").n_unique() != expected.height:
        raise ValueError("same-day position policies duplicate exit_policy_trial_id")
    terminal_ids = set(map(str, terminal["exit_policy_trial_id"].to_list()))
    expected_ids = set(map(str, expected["exit_policy_trial_id"].to_list()))
    if terminal_ids != expected_ids:
        missing = len(expected_ids - terminal_ids)
        extra = len(terminal_ids - expected_ids)
        raise ValueError(
            "cross-session nominal overlay policy coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    joined = terminal.select(
        *identity,
        "frozen_exit_threshold_basis_bp",
    ).join(
        expected,
        on="exit_policy_trial_id",
        how="left",
        suffix="_expected",
        validate="1:1",
    )
    mismatch: pl.Expr | None = None
    for name in identity:
        if name == "exit_policy_trial_id":
            continue
        term = (
            pl.col(name).cast(pl.String)
            != pl.col(f"{name}_expected").cast(pl.String)
        ).fill_null(True)
        mismatch = term if mismatch is None else mismatch | term
    threshold_mismatch = (
        pl.col("frozen_exit_threshold_basis_bp").cast(pl.Float64)
        - pl.col("_expected_frozen_exit_threshold_basis_bp").cast(pl.Float64)
    ).abs() > 1e-12
    assert mismatch is not None
    if joined.filter(mismatch | threshold_mismatch.fill_null(True)).height:
        raise ValueError(
            "cross-session nominal overlay disagrees with frozen same-day policy lineage"
        )


def _normalise_actions(frame: pl.DataFrame) -> pl.DataFrame:
    result = _validate_entry_action_execution_contract(
        frame,
        expected_hedge_delay_ns=EXPECTED_HEDGE_DELAY_NS,
    )
    if result.is_empty():
        empty_primary_schema = {
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "boundary_quantile": pl.Int64,
            "raw_order_fact_id": pl.String,
            "policy_generation_id": pl.String,
            "full_fill": pl.Boolean,
            "partial_fill": pl.Boolean,
            "any_fill": pl.Boolean,
            "cancel_required": pl.Boolean,
            "entry_hedge_status": pl.String,
            "full_fill_recv_time_ns": pl.Int64,
            "entry_hedge_decision_time_ns": pl.Int64,
            "entry_hedge_label_observed": pl.Boolean,
            "entry_hedge_executable": pl.Boolean,
            "entry_spot_price": pl.Float64,
            "entry_hedge_contract_size_shares": pl.Int64,
        }
        result = result.with_columns(
            *(
                pl.col(name).cast(dtype)
                if name in result.columns
                else pl.lit(None, dtype=dtype).alias(name)
                for name, dtype in empty_primary_schema.items()
            )
        )
    _require(result, _ACTION_REQUIRED, "entry execution actions")
    result = result.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    )
    if result.select("policy_generation_id").n_unique() != result.height:
        raise ValueError("entry actions must be unique by policy_generation_id")
    return result


def _validated_exit_hedge_delay(metadata: Mapping[str, object]) -> int:
    """Reconcile every declared entry/exit delay with the observed 50 ms facts."""

    for name in (
        "hedge_delay_ns",
        "entry_action_hedge_delay_ns",
        "exit_maker_hedge_delay_ns",
    ):
        value = metadata.get(name)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != EXPECTED_HEDGE_DELAY_NS
        ):
            raise ValueError(
                f"{name} disagrees with the formal {EXPECTED_HEDGE_DELAY_NS} ns hedge delay"
            )
    return EXPECTED_HEDGE_DELAY_NS


def _validate_complete_policy_population(
    actions: pl.DataFrame,
    positions: pl.DataFrame,
) -> dict[str, object]:
    """Require the complete Center/Lower x two-route grid per entry alias."""

    invalid_actions = actions.filter(
        pl.col("route").is_null()
        | pl.col("boundary_quantile").is_null()
        | ~pl.col("route").is_in(FROZEN_ENTRY_ROUTES)
        | ~pl.col("boundary_quantile").is_in(FROZEN_BOUNDARY_QUANTILES)
    )
    if invalid_actions.height:
        raise ValueError("entry action contains an unexpected route or q quantile")
    established = actions.filter(_established_entry_expr())
    expected_pairs = {
        (rule, route)
        for rule in FROZEN_EXIT_RULE_IDS
        for route in FROZEN_EXIT_ROUTES
    }
    rows_by_entry: dict[str, list[tuple[str, str]]] = {}
    for row in positions.select(
        "entry_policy_generation_id", "exit_rule_id", "exit_route"
    ).iter_rows(named=True):
        rows_by_entry.setdefault(
            str(row["entry_policy_generation_id"]), []
        ).append((str(row["exit_rule_id"]), str(row["exit_route"])))
    established_ids = {
        str(value) for value in established["policy_generation_id"].to_list()
    }
    if set(rows_by_entry) != established_ids:
        missing = len(established_ids - set(rows_by_entry))
        extra = len(set(rows_by_entry) - established_ids)
        raise ValueError(
            "filled-entry exit-policy coverage mismatch: "
            f"missing established entries={missing}, non-established entries={extra}"
        )
    for entry_id, pairs in rows_by_entry.items():
        if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
            missing = sorted(expected_pairs - set(pairs))
            extra = sorted(set(pairs) - expected_pairs)
            duplicates = len(pairs) - len(set(pairs))
            raise ValueError(
                "established entry lacks the exact frozen Center/Lower x two-route "
                f"population: {entry_id}; missing={missing}, extra={extra}, "
                f"duplicates={duplicates}"
            )
    return {
        "established_entry_aliases": established.height,
        "expected_rules": list(FROZEN_EXIT_RULE_IDS),
        "expected_exit_routes": list(FROZEN_EXIT_ROUTES),
        "expected_entry_routes": list(FROZEN_ENTRY_ROUTES),
        "expected_boundary_quantiles": list(FROZEN_BOUNDARY_QUANTILES),
        "policy_rows_per_established_entry": len(expected_pairs),
        "validated_position_rows": positions.height,
    }


def _normalise_positions(frame: pl.DataFrame) -> pl.DataFrame:
    _require(frame, _POSITION_REQUIRED, "exit-maker position policy facts")
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if result.select("exit_policy_trial_id").n_unique() != result.height:
        raise ValueError("position policy facts must be unique by exit_policy_trial_id")
    return result


def _build_policy_paths(
    actions: pl.DataFrame,
    positions: pl.DataFrame,
    *,
    terminal_policy_facts: pl.DataFrame | None,
) -> pl.DataFrame:
    established = actions.filter(_established_entry_expr())
    established_ids = set(established["policy_generation_id"].to_list())
    position_entry_ids = set(
        positions["entry_policy_generation_id"].to_list()
    )
    if established_ids != position_entry_ids:
        missing = len(established_ids - position_entry_ids)
        extra = len(position_entry_ids - established_ids)
        raise ValueError(
            "filled-entry exit-policy coverage mismatch: "
            f"missing established entries={missing}, non-established entries={extra}"
        )
    action_meta = established.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        pl.col("Date").alias("_action_Date"),
        pl.col("ValueCode").alias("_action_ValueCode"),
        pl.col("QuoteCode").alias("_action_QuoteCode"),
        pl.col("route").alias("_action_entry_route"),
        pl.col("raw_order_fact_id").alias("_action_entry_raw_order_fact_id"),
        "boundary_quantile",
        "entry_spot_price",
        "entry_hedge_contract_size_shares",
    )
    joined = positions.join(
        action_meta,
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )
    missing = joined.filter(pl.col("boundary_quantile").is_null())
    if missing.height:
        raise ValueError(
            "position policy facts contain an entry that was not full-filled with an executable hedge"
        )
    identity_mismatch = joined.filter(
        (pl.col("Date") != pl.col("_action_Date")).fill_null(True)
        | (pl.col("ValueCode") != pl.col("_action_ValueCode")).fill_null(True)
        | (pl.col("QuoteCode") != pl.col("_action_QuoteCode")).fill_null(True)
        | (pl.col("entry_route") != pl.col("_action_entry_route")).fill_null(True)
        | (
            pl.col("entry_raw_order_fact_id")
            != pl.col("_action_entry_raw_order_fact_id")
        ).fill_null(True)
    )
    if identity_mismatch.height:
        raise ValueError("entry action and exit policy physical identities disagree")
    if joined.filter(pl.col("position_status") != "position_established").height:
        raise ValueError("filled-entry primary facts must be established positions")

    notional = (
        pl.col("entry_spot_price")
        * pl.col("entry_hedge_contract_size_shares")
    )
    invalid_notional = joined.filter(
        notional.is_null() | ~notional.is_finite() | (notional <= 0)
    )
    if invalid_notional.height:
        raise ValueError("filled-entry positions require a finite positive notional")

    joined = joined.with_columns(
        pl.when(pl.col("nominal_instant_cancel_v0_branch") == _NOMINAL_SAME_DAY)
        .then(pl.lit(OUTCOME_COMPLETED))
        .when(
            pl.col("nominal_instant_cancel_v0_branch").is_in(
                sorted(_NOMINAL_STILL_OPEN)
            )
        )
        .then(pl.lit(OUTCOME_STILL_OPEN))
        .otherwise(pl.lit(OUTCOME_UNKNOWN))
        .alias("filled_entry_outcome_category"),
        pl.col("nominal_instant_cancel_v0_branch").alias("terminal_reason"),
        pl.when(pl.col("nominal_instant_cancel_v0_branch") == _NOMINAL_SAME_DAY)
        .then(pl.col("Date"))
        .otherwise(None)
        .alias("terminal_date"),
    )
    joined = _attach_terminal_overlay(joined, terminal_policy_facts)
    joined = joined.with_columns(
        notional.alias("normalization_notional_twd"),
        pl.when(pl.col("filled_entry_outcome_category") == OUTCOME_COMPLETED)
        .then(pl.col("gross_cycle_pnl_twd") / notional * 10_000.0)
        .otherwise(None)
        .alias("gross_cycle_bp"),
    )
    _validate_terminal_cashflows(joined)

    physical_cell_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "boundary_quantile",
        "entry_raw_order_fact_id",
        "position_established_ns",
        "exit_rule_id",
        "exit_route",
    ]
    consistency_columns = [
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "filled_entry_outcome_category",
        "terminal_reason",
        "terminal_date",
        "gross_cycle_pnl_twd",
        "gross_cycle_bp",
        "normalization_notional_twd",
    ]
    conflicts = joined.group_by(physical_cell_key).agg(
        *(pl.col(column).n_unique().alias(column) for column in consistency_columns)
    ).filter(
        pl.any_horizontal(*(pl.col(column) > 1 for column in consistency_columns))
    )
    if conflicts.height:
        raise ValueError(
            "policy aliases disagree within one physical filled-entry policy cell"
        )

    passthrough = [
        "entry_policy_generation_id",
        "exit_policy_trial_id",
        *consistency_columns,
    ]
    collapsed = joined.group_by(physical_cell_key, maintain_order=True).agg(
        pl.len().alias("filled_entry_policy_aliases"),
        *(pl.col(column).first().alias(column) for column in passthrough),
    )

    dependency_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_raw_order_fact_id",
        "position_established_ns",
    ]
    dependency = established.select(
        pl.col("Date"),
        pl.col("ValueCode"),
        pl.col("QuoteCode"),
        pl.col("route").alias("entry_route"),
        pl.col("raw_order_fact_id").alias("entry_raw_order_fact_id"),
        pl.col("entry_hedge_decision_time_ns").alias("position_established_ns")
        if "entry_hedge_decision_time_ns" in established.columns
        else pl.lit(None, dtype=pl.Int64).alias("position_established_ns"),
        "boundary_quantile",
    )
    # Existing action facts always carry the exact establishment cursor.  For
    # deliberately minimal synthetic callers, recover it from policy facts.
    if dependency["position_established_ns"].null_count() == dependency.height:
        dependency = joined.select(*dependency_key, "boundary_quantile")
    dependency = dependency.unique().group_by(dependency_key).agg(
        pl.col("boundary_quantile")
        .n_unique()
        .alias("physical_entry_boundary_quantile_alias_count")
    )
    collapsed = collapsed.join(
        dependency,
        on=dependency_key,
        how="left",
        validate="m:1",
    )
    if collapsed.filter(
        pl.col("physical_entry_boundary_quantile_alias_count").is_null()
    ).height:
        raise ValueError("failed to resolve cross-q physical entry dependency")
    collapsed = collapsed.with_columns(
        pl.concat_str(
            *(pl.col(column).cast(pl.String) for column in dependency_key),
            separator="|",
        ).alias("physical_entry_dependency_id"),
        (
            pl.col("physical_entry_boundary_quantile_alias_count") > 1
        ).alias("physical_entry_shared_across_q"),
        pl.lit(False).alias("boundary_quantile_rows_safe_to_sum"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("alternative_policy_rows_additive"),
        pl.lit(False).alias("entry_no_fill_included"),
        pl.lit(False).alias("gross_zero_imputation"),
        pl.lit("completed_cycles_only").alias(
            "gross_statistics_conditioning"
        ),
    )
    return collapsed


def _attach_terminal_overlay(
    frame: pl.DataFrame,
    terminal_policy_facts: pl.DataFrame | None,
) -> pl.DataFrame:
    if terminal_policy_facts is None:
        return frame
    _require(terminal_policy_facts, _TERMINAL_REQUIRED, "terminal policy facts")
    terminal = terminal_policy_facts
    if terminal.select("exit_policy_trial_id").n_unique() != terminal.height:
        raise ValueError("terminal policy facts must be unique by exit_policy_trial_id")
    categories = set(
        str(value)
        for value in terminal["filled_entry_outcome_category"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    invalid = sorted(categories - OUTCOME_CATEGORIES)
    if invalid or terminal["filled_entry_outcome_category"].null_count():
        raise ValueError(f"terminal policy facts contain invalid categories: {invalid}")
    unknown_ids = set(terminal["exit_policy_trial_id"].to_list()) - set(
        frame["exit_policy_trial_id"].to_list()
    )
    if unknown_ids:
        raise ValueError("terminal policy facts contain unknown exit policy trials")

    for column, dtype in (
        ("terminal_date", pl.String),
        ("terminal_reason", pl.String),
    ):
        if column not in terminal.columns:
            terminal = terminal.with_columns(pl.lit(None, dtype=dtype).alias(column))
    overlay = terminal.select(
        "exit_policy_trial_id",
        pl.col("filled_entry_outcome_category").alias("_overlay_category"),
        pl.col("gross_cycle_pnl_twd").alias("_overlay_gross"),
        pl.col("terminal_date").cast(pl.String).alias("_overlay_terminal_date"),
        pl.col("terminal_reason").cast(pl.String).alias("_overlay_terminal_reason"),
    )
    result = frame.join(
        overlay,
        on="exit_policy_trial_id",
        how="left",
        validate="1:1",
    )
    supplied = pl.col("_overlay_category").is_not_null()
    contradicts_same_day = result.filter(
        supplied
        & (pl.col("filled_entry_outcome_category") == OUTCOME_COMPLETED)
        & (
            (pl.col("_overlay_category") != OUTCOME_COMPLETED)
            | (
                pl.col("_overlay_gross") != pl.col("gross_cycle_pnl_twd")
            ).fill_null(True)
            | (
                pl.col("_overlay_terminal_date") != pl.col("terminal_date")
            ).fill_null(True)
        )
    )
    if contradicts_same_day.height:
        raise ValueError(
            "terminal overlay contradicts an existing same-day terminal identity"
        )
    baseline_completed = (
        pl.col("filled_entry_outcome_category") == OUTCOME_COMPLETED
    )
    apply_overlay = supplied & ~baseline_completed
    return result.with_columns(
        pl.when(apply_overlay)
        .then(pl.col("_overlay_category"))
        .otherwise(pl.col("filled_entry_outcome_category"))
        .alias("filled_entry_outcome_category"),
        pl.when(apply_overlay)
        .then(pl.col("_overlay_gross"))
        .otherwise(pl.col("gross_cycle_pnl_twd"))
        .alias("gross_cycle_pnl_twd"),
        pl.when(apply_overlay)
        .then(pl.col("_overlay_terminal_date"))
        .otherwise(pl.col("terminal_date"))
        .alias("terminal_date"),
        pl.when(apply_overlay)
        .then(pl.col("_overlay_terminal_reason"))
        .otherwise(pl.col("terminal_reason"))
        .alias("terminal_reason"),
    ).drop(
        "_overlay_category",
        "_overlay_gross",
        "_overlay_terminal_date",
        "_overlay_terminal_reason",
    )


def _validate_terminal_cashflows(frame: pl.DataFrame) -> None:
    category = pl.col("filled_entry_outcome_category")
    completed = category == OUTCOME_COMPLETED
    gross = pl.col("gross_cycle_pnl_twd")
    gross_bp = pl.col("gross_cycle_bp")
    invalid_completed = frame.filter(
        completed
        & (
            gross.is_null()
            | ~gross.is_finite()
            | gross_bp.is_null()
            | ~gross_bp.is_finite()
        )
    )
    if invalid_completed.height:
        raise ValueError("completed filled-entry cycles require finite gross cashflow")
    invalid_open = frame.filter(
        ~completed & (gross.is_not_null() | gross_bp.is_not_null())
    )
    if invalid_open.height:
        raise ValueError("non-completed filled-entry paths must not carry imputed gross")
    terminal_date = pl.col("terminal_date").cast(pl.String)
    invalid_terminal_date = frame.filter(
        completed
        & (
            terminal_date.is_null()
            | ~terminal_date.str.contains(r"^\d{8}$")
            | (terminal_date < pl.col("Date").cast(pl.String))
        )
    )
    if invalid_terminal_date.height:
        raise ValueError(
            "completed filled-entry cycles require terminal_date >= entry Date"
        )
    if frame.filter(~completed & terminal_date.is_not_null()).height:
        raise ValueError("unresolved filled-entry paths must have null terminal_date")


def _aggregate_primary(frame: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    category = pl.col("filled_entry_outcome_category")
    completed = category == OUTCOME_COMPLETED
    result = frame.group_by(list(keys)).agg(
        pl.len().alias("filled_entry_positions"),
        pl.col("filled_entry_policy_aliases").sum().alias(
            "filled_entry_policy_aliases"
        ),
        pl.col("Date").n_unique().alias("filled_entry_sessions"),
        completed.sum().alias("completed_cycles"),
        (category == OUTCOME_STILL_OPEN).sum().alias("still_open_positions"),
        (category == OUTCOME_CENSORED).sum().alias("censored_positions"),
        (category == OUTCOME_UNKNOWN).sum().alias("unknown_positions"),
        pl.col("gross_cycle_bp").count().alias("gross_completed_cycles"),
        pl.col("gross_cycle_bp").quantile(0.05).alias("gross_cycle_bp_p05"),
        pl.col("gross_cycle_bp").median().alias("gross_cycle_bp_p50"),
        pl.col("gross_cycle_bp").mean().alias("gross_cycle_bp_mean"),
        pl.col("gross_cycle_bp").quantile(0.95).alias("gross_cycle_bp_p95"),
        pl.col("gross_cycle_pnl_twd").quantile(0.05).alias(
            "gross_cycle_pnl_twd_p05"
        ),
        pl.col("gross_cycle_pnl_twd").median().alias("gross_cycle_pnl_twd_p50"),
        pl.col("gross_cycle_pnl_twd").mean().alias("gross_cycle_pnl_twd_mean"),
        pl.col("gross_cycle_pnl_twd").quantile(0.95).alias(
            "gross_cycle_pnl_twd_p95"
        ),
        pl.col("physical_entry_dependency_id").n_unique().alias(
            "unique_physical_entry_dependencies"
        ),
        pl.col("physical_entry_shared_across_q").sum().alias(
            "positions_shared_across_q"
        ),
    )
    result = result.with_columns(
        (pl.col("completed_cycles") / pl.col("filled_entry_positions")).alias(
            "completion_rate_given_filled_entry"
        ),
        (
            (
                pl.col("completed_cycles")
                + pl.col("still_open_positions")
                + pl.col("censored_positions")
                + pl.col("unknown_positions")
            )
            / pl.col("filled_entry_positions")
        ).alias("status_accounting_rate"),
        pl.lit("completed_cycles_only").alias(
            "gross_statistics_conditioning"
        ),
        pl.lit(False).alias("gross_zero_imputation"),
        pl.lit(False).alias("entry_no_fill_included"),
        pl.lit(False).alias("boundary_quantile_rows_safe_to_sum"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("alternative_policy_rows_additive"),
    )
    invalid = result.filter(
        (pl.col("gross_completed_cycles") != pl.col("completed_cycles"))
        | (pl.col("status_accounting_rate") != 1.0)
    )
    if invalid.height:
        raise AssertionError("filled-entry primary denominator accounting failed")
    return result


def _build_execution_diagnostics(actions: pl.DataFrame) -> pl.DataFrame:
    full = pl.col("full_fill") == True  # noqa: E712
    any_known = pl.col("any_fill").is_not_null()
    no_fill = any_known & (pl.col("any_fill") == False)  # noqa: E712
    hedge_executable = _established_entry_expr()
    diagnostics = (
        actions.rename({"route": "entry_route"})
        .group_by(EXECUTION_DIAGNOSTIC_KEY)
        .agg(
            pl.len().alias("submitted_entry_policy_aliases"),
            pl.col("raw_order_fact_id").n_unique().alias(
                "submitted_physical_raw_orders"
            ),
            no_fill.sum().alias("known_no_fill_policy_aliases"),
            (~any_known).sum().alias("entry_fill_unknown_policy_aliases"),
            pl.col("partial_fill").fill_null(False).sum().alias(
                "entry_partial_fill_policy_aliases"
            ),
            full.sum().alias("entry_full_fill_policy_aliases"),
            hedge_executable.sum().alias(
                "entry_full_fill_hedge_executable_policy_aliases"
            ),
            (full & ~hedge_executable).sum().alias(
                "entry_full_fill_hedge_not_executable_policy_aliases"
            ),
            pl.col("cancel_required").fill_null(False).sum().alias(
                "entry_cancel_request_policy_aliases"
            ),
        )
        .with_columns(
            (
                pl.col("entry_cancel_request_policy_aliases")
                / pl.col("submitted_entry_policy_aliases")
            ).alias("entry_cancel_request_rate"),
            (
                pl.col("entry_full_fill_hedge_executable_policy_aliases")
                / pl.col("submitted_entry_policy_aliases")
            ).alias("full_hedged_rate_per_submitted_alias"),
            pl.lit("execution_diagnostic_only_not_profit_denominator").alias(
                "table_semantics"
            ),
            pl.lit(True).alias("cancel_rate_is_request_only"),
            pl.lit(False).alias("boundary_quantile_rows_safe_to_sum"),
        )
    )
    forbidden = [
        column
        for column in diagnostics.columns
        if any(token in column for token in ("gross", "net_pnl", "profit"))
    ]
    if forbidden:
        raise AssertionError(f"execution diagnostics leaked profit columns: {forbidden}")
    return diagnostics


def _publish_report(report: FilledEntryPrimaryReport, destination: Path) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    tables: Mapping[str, tuple[pl.DataFrame, str]] = {
        "coverage.csv": (report.coverage, "csv"),
        "filled_entry_policy_paths.parquet": (
            report.filled_entry_policy_paths,
            "parquet",
        ),
        "filled_entry_primary_by_policy.csv": (report.policy_summary, "csv"),
        "filled_entry_primary_pooled_12_cells.csv": (
            report.pooled_12_cell_summary,
            "csv",
        ),
        "filled_entry_primary_by_product_policy.csv": (
            report.product_policy_summary,
            "csv",
        ),
        "entry_execution_diagnostics.csv": (
            report.execution_diagnostics,
            "csv",
        ),
    }
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name, (frame, kind) in tables.items():
            path = stage / name
            if kind == "parquet":
                frame.write_parquet(path)
            else:
                frame.write_csv(path)
            artifacts[name] = {
                "rows": frame.height,
                "columns": frame.width,
                "sha256": _file_sha256(path),
            }
        marker = dict(report.metadata)
        marker.update({"complete": True, "artifacts": artifacts})
        (stage / "report_complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    overlay = parser.add_mutually_exclusive_group()
    overlay.add_argument("--terminal-policy-facts", type=Path)
    overlay.add_argument(
        "--cross-session-root",
        type=Path,
        help=(
            "Hash-verified partitioned cross-session root; the primary report "
            "loads only its nominal-instant-cancel-V0 outcomes"
        ),
    )
    parser.add_argument(
        "--prerequisite-root",
        type=Path,
        help=(
            "Verified formal prerequisite root whose exact marker identity "
            "must bind every cross-session partition"
        ),
    )
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--require-balanced-product-days", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_filled_entry_primary_report(
        args.exit_root,
        args.entry_root,
        output_dir=args.output_dir,
        terminal_policy_facts_path=args.terminal_policy_facts,
        cross_session_root=args.cross_session_root,
        prerequisite_root=args.prerequisite_root,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        require_balanced_product_days=args.require_balanced_product_days,
        validate_hashes=not args.skip_hash_validation,
    )
    print(json.dumps(dict(report.metadata), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
