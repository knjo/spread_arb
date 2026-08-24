"""Re-runnable strict terminal-path and D-safe EV audit.

This command is intentionally an *audit bridge*, not a strategy optimizer. It
loads completed execution partitions with the validated execution-report
loader, maps action/exit observations through the strict terminal-path
adapter, and builds exactly one lookup snapshot as of the last selected date.

The last date's target outcomes remain visible in the terminal-path artifact
but are explicitly excluded from lookup training.  Unknown cancel races,
right-censored EOD carry, null costs, and missing exit-slippage diagnostics
are retained.  Consequently this v1 command asserts that every lookup cell is
not ready and that no net EV has been priced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import polars as pl

from .ev_surface import EVLookupConfig, build_daily_ev_lookup
from .execution_report import (
    PartitionedExecutionInputs,
    load_partitioned_execution_inputs,
)
from .terminal_paths import (
    EXECUTION_PATH_LOOKUP_KEYS_V1,
    TerminalPathAdapterConfig,
    build_execution_terminal_paths_v1,
)


AUDIT_VERSION = "execution_ev_audit_v1"
STATE_COLLAPSE_VALUES = frozenset({"observed", "all"})
UNASSIGNED_EXIT_RULE_ID = "__unassigned_exit_rule__"


@dataclass(frozen=True)
class ExecutionEVAudit:
    """Strict path facts, last-date lookup, and compact audit summaries."""

    terminal_paths: pl.DataFrame
    ev_lookup: pl.DataFrame
    terminal_path_summary: pl.DataFrame
    ev_lookup_summary: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


def build_execution_ev_audit(
    inputs: PartitionedExecutionInputs,
    *,
    state_collapse: str = "observed",
    adapter_config: TerminalPathAdapterConfig = TerminalPathAdapterConfig(
        lifecycle_policy_version="layered_execution_observation_v1",
        queue_scenario="displayed_queue_independent_candidate_v1",
    ),
    lookup_config: EVLookupConfig = EVLookupConfig(
        lookup_keys=EXECUTION_PATH_LOOKUP_KEYS_V1
    ),
) -> ExecutionEVAudit:
    """Build one last-selected-date, target-outcome-safe lookup snapshot."""

    if state_collapse not in STATE_COLLAPSE_VALUES:
        raise ValueError(
            f"state_collapse must be one of {sorted(STATE_COLLAPSE_VALUES)}"
        )
    if adapter_config.nominal_cancel_model_version is not None:
        raise ValueError(
            "strict execution EV audit does not permit a nominal cancel model"
        )
    if tuple(lookup_config.lookup_keys) != EXECUTION_PATH_LOOKUP_KEYS_V1:
        raise ValueError(
            "lookup_config must use EXECUTION_PATH_LOOKUP_KEYS_V1"
        )
    selected_dates = _selected_dates(inputs)
    asof_date = selected_dates[-1]

    exit_facts, unassigned_exit_rows = _preserve_unassigned_exit_paths(
        inputs.action_facts,
        inputs.exit_facts,
    )
    terminal_paths = build_execution_terminal_paths_v1(
        inputs.action_facts,
        exit_facts,
        adapter_config,
    )
    if not terminal_paths.is_empty():
        if state_collapse == "all":
            terminal_paths = terminal_paths.with_columns(
                pl.lit("all").alias("state_family"),
                pl.lit("all").alias("state_bucket"),
            )
        terminal_paths = _annotate_asof_paths(terminal_paths, asof_date)
        ev_lookup = build_daily_ev_lookup(
            terminal_paths,
            selected_dates,
            lookup_config,
            asof_dates=[asof_date],
        )
    else:
        ev_lookup = pl.DataFrame()

    _validate_strict_audit(terminal_paths, ev_lookup, asof_date)
    path_summary = _terminal_path_summary(terminal_paths)
    lookup_summary = _ev_lookup_summary(ev_lookup, asof_date)
    target_rows = (
        terminal_paths.filter(pl.col("contains_asof_target_day_outcome")).height
        if not terminal_paths.is_empty()
        else 0
    )
    training_rows = (
        terminal_paths.filter(pl.col("included_in_asof_lookup")).height
        if not terminal_paths.is_empty()
        else 0
    )
    input_metadata = dict(inputs.metadata)
    metadata: dict[str, object] = {
        "audit_version": AUDIT_VERSION,
        "asof_date": asof_date,
        "selected_dates": selected_dates,
        "selected_session_count": len(selected_dates),
        "state_collapse": state_collapse,
        "state_family_in_lookup": (
            "all" if state_collapse == "all" else adapter_config.state_family
        ),
        "terminal_path_rows": terminal_paths.height,
        "asof_target_day_path_rows": target_rows,
        "prior_mature_lookup_input_rows": training_rows,
        "unassigned_exit_rule_path_rows": unassigned_exit_rows,
        "lookup_cells": ev_lookup.height,
        "ev_ready_cells": (
            ev_lookup.filter(pl.col("ev_ready")).height
            if not ev_lookup.is_empty()
            else 0
        ),
        "priced_expected_ev_cells": (
            ev_lookup.filter(pl.col("expected_net_cashflow_bp").is_not_null()).height
            if not ev_lookup.is_empty()
            else 0
        ),
        "terminal_paths_are_realized_outcome_labels": True,
        "terminal_paths_contain_asof_target_day_outcomes": target_rows > 0,
        "asof_target_day_outcomes_in_lookup": False,
        "lookup_execution_safe_snapshot": True,
        "unknown_imputed_as_zero": False,
        "censored_imputed_as_zero": False,
        "null_costs_imputed_as_zero": False,
        "eod_mark_used_as_terminal_cashflow": False,
        "nominal_cancel_assumed_complete": False,
        "missing_exit_rule_fact_preserved_as_unknown": True,
        "unassigned_exit_rule_id": UNASSIGNED_EXIT_RULE_ID,
        "pathwise_ev_ready": False,
        "adapter_config": asdict(adapter_config),
        "lookup_config": asdict(lookup_config),
        "input_metadata": input_metadata,
    }
    return ExecutionEVAudit(
        terminal_paths=terminal_paths,
        ev_lookup=ev_lookup,
        terminal_path_summary=path_summary,
        ev_lookup_summary=lookup_summary,
        coverage=inputs.coverage,
        metadata=metadata,
    )


def run_execution_ev_audit(
    execution_root: Path,
    *,
    output_dir: Path | None = None,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    allow_incomplete_product_grid: bool = False,
    validate_hashes: bool = True,
    state_collapse: str = "observed",
    adapter_config: TerminalPathAdapterConfig = TerminalPathAdapterConfig(
        lifecycle_policy_version="layered_execution_observation_v1",
        queue_scenario="displayed_queue_independent_candidate_v1",
    ),
    lookup_config: EVLookupConfig = EVLookupConfig(
        lookup_keys=EXECUTION_PATH_LOOKUP_KEYS_V1
    ),
) -> ExecutionEVAudit:
    """Load execution partitions, build the strict audit, and write artifacts."""

    requested_value_codes = (
        tuple(str(value) for value in value_codes)
        if value_codes is not None
        else None
    )
    inputs = load_partitioned_execution_inputs(
        Path(execution_root),
        sessions=sessions,
        value_codes=requested_value_codes,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=not allow_incomplete_product_grid,
        validate_hashes=validate_hashes,
    )
    audit = build_execution_ev_audit(
        inputs,
        state_collapse=state_collapse,
        adapter_config=adapter_config,
        lookup_config=lookup_config,
    )
    audit = ExecutionEVAudit(
        terminal_paths=audit.terminal_paths,
        ev_lookup=audit.ev_lookup,
        terminal_path_summary=audit.terminal_path_summary,
        ev_lookup_summary=audit.ev_lookup_summary,
        coverage=audit.coverage,
        metadata={
            **dict(audit.metadata),
            "loader_options": {
                "execution_root": str(Path(execution_root)),
                "sessions": sessions,
                "value_codes": requested_value_codes,
                "require_exact_sessions": require_exact_sessions,
                "allow_incomplete_product_grid": allow_incomplete_product_grid,
                "validate_hashes": validate_hashes,
            },
        },
    )
    count = int(audit.metadata["selected_session_count"])
    asof_date = str(audit.metadata["asof_date"])
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(execution_root)
        / f"ev_audit_{count}_sessions_asof_{asof_date}"
    )
    _write_audit(audit, destination)
    return audit


def _selected_dates(inputs: PartitionedExecutionInputs) -> list[str]:
    metadata_dates = inputs.metadata.get("selected_dates")
    if isinstance(metadata_dates, (list, tuple)):
        dates = sorted({str(value) for value in metadata_dates})
    else:
        dates = sorted(
            {
                str(value)
                for frame in (
                    inputs.action_facts,
                    inputs.daily_support,
                    inputs.coverage,
                )
                if "Date" in frame.columns
                for value in frame["Date"].drop_nulls().to_list()
            }
        )
    if not dates:
        raise ValueError("execution inputs contain no selected dates")
    malformed = pl.DataFrame({"Date": dates}).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_d")
    ).filter(pl.col("_d").is_null() | (pl.col("Date").str.len_chars() != 8))
    if malformed.height:
        raise ValueError("selected dates must be valid YYYYMMDD values")
    if not inputs.action_facts.is_empty():
        fact_dates = set(str(value) for value in inputs.action_facts["Date"])
        unknown = sorted(fact_dates - set(dates))
        if unknown:
            raise ValueError(f"action facts contain unselected dates: {unknown[:5]}")
    return dates


def _preserve_unassigned_exit_paths(
    action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Retain admitted actions lacking an exit fact as one unknown path.

    The missing row is deliberately *not* duplicated into the observed
    ``frozen_center``/``frozen_lower`` policies: the partition contains no
    evidence that either rule was assigned.  A separate sentinel policy keeps
    the admitted-action denominator visible without manufacturing an exit
    outcome or a D-1 lineage date.
    """

    if action_facts.is_empty():
        return exit_facts, 0
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "raw_order_fact_id",
        "policy_generation_id",
    }
    missing_columns = sorted(required - set(action_facts.columns))
    if missing_columns:
        raise ValueError(
            f"execution action facts missing identity columns: {missing_columns}"
        )
    observed_ids = (
        exit_facts.select("policy_generation_id").unique()
        if "policy_generation_id" in exit_facts.columns
        else pl.DataFrame(schema={"policy_generation_id": pl.String})
    )
    missing = action_facts.join(
        observed_ids,
        on="policy_generation_id",
        how="anti",
    )
    if missing.is_empty():
        return exit_facts, 0
    placeholders = missing.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "raw_order_fact_id",
        "policy_generation_id",
    ).with_columns(
        pl.lit(UNASSIGNED_EXIT_RULE_ID).alias("exit_rule_id"),
        pl.lit(None, dtype=pl.String).alias("exit_rule_source_asof_date"),
        pl.lit("exit_policy_unassigned").alias("branch_status"),
        pl.lit(False).alias("terminal_outcome"),
        pl.lit(False).alias("needs_next_session_label"),
        pl.lit(None, dtype=pl.Int64).alias("exit_decision_time_ns"),
        pl.lit(None, dtype=pl.Float64).alias("gross_cycle_pnl_twd"),
    )
    combined = (
        pl.concat([exit_facts, placeholders], how="diagonal_relaxed")
        if not exit_facts.is_empty()
        else placeholders
    )
    return combined, placeholders.height


def _annotate_asof_paths(paths: pl.DataFrame, asof_date: str) -> pl.DataFrame:
    target = pl.col("Date") == asof_date
    mature_prior = (pl.col("Date") < asof_date) & (
        pl.col("label_end_date") < asof_date
    )
    return paths.with_columns(
        target.alias("contains_asof_target_day_outcome"),
        (pl.col("label_end_date") < asof_date).alias(
            "label_mature_before_asof"
        ),
        mature_prior.alias("included_in_asof_lookup"),
        pl.when(target)
        .then(pl.lit("excluded_asof_target_day_outcome"))
        .when(pl.col("Date") >= asof_date)
        .then(pl.lit("excluded_not_prior_date"))
        .when(pl.col("label_end_date") >= asof_date)
        .then(pl.lit("excluded_label_not_mature"))
        .otherwise(pl.lit("included_prior_mature_admitted_path"))
        .alias("asof_lookup_role"),
    )


def _validate_strict_audit(
    terminal_paths: pl.DataFrame,
    lookup: pl.DataFrame,
    asof_date: str,
) -> None:
    if not terminal_paths.is_empty():
        leaked = terminal_paths.filter(
            pl.col("contains_asof_target_day_outcome")
            & pl.col("included_in_asof_lookup")
        )
        if leaked.height:
            raise ValueError("as-of target-day path was admitted to lookup input")
        unknown_priced = terminal_paths.filter(
            (pl.col("outcome_status") != "known")
            & pl.col("filled_cashflow_before_cost_bp").is_not_null()
        )
        if unknown_priced.height:
            raise ValueError("unknown/censored terminal path was assigned cashflow")
        nonnull_cost = terminal_paths.filter(
            pl.any_horizontal(
                pl.col("fee_cost_bp").is_not_null(),
                pl.col("tax_cost_bp").is_not_null(),
                pl.col("financing_cost_bp").is_not_null(),
                pl.col("overnight_cost_bp").is_not_null(),
                pl.col("cancel_cost_bp").is_not_null(),
                pl.col("emergency_cost_bp").is_not_null(),
                pl.col("cost_profile_version").is_not_null(),
            )
        )
        if nonnull_cost.height:
            raise ValueError("strict v1 audit unexpectedly contains priced costs")
        invalid_unassigned = terminal_paths.filter(
            (pl.col("exit_rule_id") == UNASSIGNED_EXIT_RULE_ID)
            & (
                (pl.col("outcome_status") != "unknown")
                | pl.col("terminal_branch").is_not_null()
                | pl.col("exit_rule_fact_observed")
            )
        )
        if invalid_unassigned.height:
            raise ValueError("unassigned exit rule was treated as a known outcome")
    if lookup.is_empty():
        return
    unsafe = lookup.filter(
        ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
        | (pl.col("asof_date") != asof_date)
        | (pl.col("train_end_date") >= asof_date)
    )
    if unsafe.height:
        raise ValueError("lookup contains target-day leakage or unsafe lineage")
    if lookup.filter(pl.col("ev_ready")).height:
        raise ValueError("strict unpriced audit unexpectedly produced ev_ready cell")
    if lookup.filter(
        pl.col("expected_net_cashflow_bp").is_not_null()
        | pl.col("action_score_bp").is_not_null()
    ).height:
        raise ValueError("strict unpriced audit unexpectedly produced a priced EV")


def _terminal_path_summary(paths: pl.DataFrame) -> pl.DataFrame:
    if paths.is_empty():
        return pl.DataFrame(
            schema={
                "Date": pl.String,
                "contains_asof_target_day_outcome": pl.Boolean,
                "included_in_asof_lookup": pl.Boolean,
                "outcome_status": pl.String,
                "terminal_branch": pl.String,
                "terminal_mapping_reason": pl.String,
                "joint_policy_paths": pl.UInt32,
                "physical_raw_paths": pl.UInt32,
            }
        )
    keys = [
        "Date",
        "contains_asof_target_day_outcome",
        "included_in_asof_lookup",
        "outcome_status",
        "terminal_branch",
        "terminal_mapping_reason",
    ]
    return paths.group_by(keys).agg(
        pl.len().alias("joint_policy_paths"),
        pl.col("physical_path_id").n_unique().alias("physical_raw_paths"),
        pl.col("filled_cashflow_before_cost_bp")
        .is_not_null()
        .sum()
        .alias("gross_cashflow_observed_paths"),
        pl.col("cost_profile_version")
        .is_not_null()
        .sum()
        .alias("cost_profile_observed_paths"),
    ).sort(keys, nulls_last=True)


def _ev_lookup_summary(lookup: pl.DataFrame, asof_date: str) -> pl.DataFrame:
    if lookup.is_empty():
        return pl.DataFrame(
            {
                "asof_date": [asof_date],
                "ev_ready": [False],
                "ev_status": ["no_lookup_cells"],
                "lookup_cells": [0],
                "n_paths": [0],
                "n_outcome_known": [0],
                "n_outcome_censored": [0],
                "n_outcome_unknown": [0],
                "priced_expected_ev_cells": [0],
                "execution_safe_snapshot": [True],
                "contains_target_day_outcome": [False],
            }
        )
    return lookup.group_by(
        "asof_date",
        "ev_ready",
        "ev_status",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    ).agg(
        pl.len().alias("lookup_cells"),
        pl.col("n_paths").sum().alias("n_paths"),
        pl.col("n_outcome_known").sum().alias("n_outcome_known"),
        pl.col("n_outcome_censored").sum().alias("n_outcome_censored"),
        pl.col("n_outcome_unknown").sum().alias("n_outcome_unknown"),
        pl.col("expected_net_cashflow_bp")
        .is_not_null()
        .sum()
        .alias("priced_expected_ev_cells"),
        pl.col("train_end_date").max().alias("max_train_end_date"),
        pl.col("label_cutoff_date").max().alias("max_label_cutoff_date"),
    ).sort(["asof_date", "ev_status"])


def _write_audit(audit: ExecutionEVAudit, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    parquet_tables = {
        "strict_terminal_paths.parquet": audit.terminal_paths,
        "asof_ev_lookup.parquet": audit.ev_lookup,
    }
    csv_tables = {
        "terminal_path_summary.csv": audit.terminal_path_summary,
        "ev_lookup_status_summary.csv": audit.ev_lookup_summary,
        "coverage.csv": audit.coverage,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, frame in parquet_tables.items():
        path = destination / name
        _atomic_write_parquet(frame, path)
        artifacts[name] = _artifact_metadata(frame, path)
    for name, frame in csv_tables.items():
        path = destination / name
        _atomic_write_csv(frame, path)
        artifacts[name] = _artifact_metadata(frame, path)
    payload = {
        "complete": True,
        **_jsonable(dict(audit.metadata)),
        "artifacts": artifacts,
    }
    _atomic_write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        destination / "audit_config.json",
    )


def _artifact_metadata(frame: pl.DataFrame, path: Path) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "sha256": _sha256(path),
    }


def _atomic_write_parquet(frame: pl.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(frame: pl.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_csv(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(text: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict terminal paths and a last-date D-safe, not-ready "
            "execution EV lookup"
        )
    )
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--allow-incomplete-product-grid", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument(
        "--state-collapse",
        choices=sorted(STATE_COLLAPSE_VALUES),
        default="observed",
    )
    parser.add_argument(
        "--lifecycle-policy-version",
        default="layered_execution_observation_v1",
    )
    parser.add_argument(
        "--queue-scenario",
        default="displayed_queue_independent_candidate_v1",
    )
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=40)
    parser.add_argument("--min-group-sessions", type=int, default=20)
    parser.add_argument("--min-known-paths", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = run_execution_ev_audit(
        args.execution_root,
        output_dir=args.output_dir,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        allow_incomplete_product_grid=args.allow_incomplete_product_grid,
        validate_hashes=not args.skip_hash_validation,
        state_collapse=args.state_collapse,
        adapter_config=TerminalPathAdapterConfig(
            lifecycle_policy_version=args.lifecycle_policy_version,
            queue_scenario=args.queue_scenario,
        ),
        lookup_config=EVLookupConfig(
            lookback_sessions=args.lookback_sessions,
            min_history_sessions=args.min_history_sessions,
            min_group_sessions=args.min_group_sessions,
            min_known_paths=args.min_known_paths,
            lookup_keys=EXECUTION_PATH_LOOKUP_KEYS_V1,
        ),
    )
    print(audit.ev_lookup_summary)


if __name__ == "__main__":
    main()
