"""Source-bound AB1/2 walk-forward ledger for the diagnostic challenger.

The post-cross evaluator deliberately refuses to select a production action
because terminal cashflows and formal costs are incomplete.  It nevertheless
publishes a D-safe *diagnostic challenger*: for each target session D, the
challenger is ranked only from labels visible before D.  This module applies
that frozen challenger to D's already-frozen entry action facts and keeps only
the route-valid first two maker levels (ASK1/ASK2 or BID1/BID2).

This is an analysis-only pseudo-out-of-sample ledger.  Entry actions are
independent counterfactual labels, cancel ACK is unavailable, overlapping
orders do not jointly consume volume, unresolved position cashflows stay
null, and the challenger is explicitly not the selected production action.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from .compact_frozen_facts import (
    EXPECTED_EXECUTION_MANIFEST_SHA256,
    EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256,
    FROZEN_COMPACT_VERSION,
    _scan_actions,
    _selected_manifest_paths,
)
from .cross_session_prerequisite import verify_cross_session_prerequisites
from .post_cross_position_evaluator import DEFAULT_POSITION_LIMITS, PositionLimit


LEDGER_VERSION = "prequential_diagnostic_challenger_ab12_v1"
BUNDLE_SCHEMA_VERSION = "prequential_diagnostic_challenger_bundle_v1"

DEFAULT_EXECUTION_ROOT = Path("maker/data/walkforward/execution_narrow_60d")
DEFAULT_POST_CROSS_ROOT = Path(
    "maker/data/walkforward/post_cross_position_evaluation_60d"
)
DEFAULT_PREREQUISITE_ROOT = Path(
    "maker/data/walkforward/cross_session_prerequisites_v1_20260819"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/prequential_challenger_ab12_60d_20260821"
)

ARTIFACTS: Mapping[str, str] = {
    "selected_actions.parquet": "selected_actions",
    "selected_policy_paths.parquet": "selected_paths",
    "daily_cohort_ledger.parquet": "daily_cohort_ledger",
    "daily_realized_cashflows.parquet": "daily_realized_cashflows",
    "daily_outstanding.parquet": "daily_outstanding",
    "challenger_selection_counts.parquet": "selection_counts",
    "entry_rank_summary.parquet": "entry_rank_summary",
    "position_limit_sweep.parquet": "position_limit_sweep",
    "overall_summary.parquet": "overall_summary",
}

_DECISION_REQUIRED = {
    "asof_date",
    "selected_action_ev_ready",
    "decision_status",
    "diagnostic_challenger_boundary_quantile",
    "diagnostic_challenger_entry_route",
    "diagnostic_challenger_exit_rule_id",
    "diagnostic_challenger_exit_route",
    "diagnostic_challenger_completed_only_after_cost_mean_bp",
    "diagnostic_challenger_is_selected_action",
    "ranking_uses_only_Date_less_than_asof",
    "label_visible_only_if_label_availability_date_less_than_asof",
    "nominal_cancel_model_assumption",
    "production_strategy_go",
}

_ACTION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_rank_at_submit",
    "intended_quantity",
    "submit_recv_time_ns",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "full_fill_recv_time_ns",
    "terminal_recv_time_ns",
    "terminal_reason",
    "cancel_required",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_hedge_signed_total_slippage_bp",
}

_PATH_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "boundary_quantile",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "position_established_ns",
    "exit_rule_id",
    "exit_route",
    "filled_entry_outcome_category",
    "terminal_date",
    "gross_cycle_pnl_twd",
    "gross_cycle_bp",
    "normalization_notional_twd",
    "physical_entry_dependency_id",
    "policy_path_id",
    "label_availability_date",
    "outstanding_interval_end_exclusive",
    "completed_same_day",
    "completed_overnight",
    "terminal_cashflow_priced",
    "exit_decision_time_ns",
    "nominal_cancel_model_assumption",
    "pathwise_ev_ready",
    "joint_volume_allocated",
}

_ACTION_OUTPUT_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_rank_at_submit",
    "intended_quantity",
    "submit_recv_time_ns",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "full_fill_recv_time_ns",
    "terminal_recv_time_ns",
    "terminal_reason",
    "cancel_required",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_hedge_signed_total_slippage_bp",
)


@dataclass(frozen=True)
class ChallengerLedgerConfig:
    same_day_cost_bp: float = 19.0
    overnight_cost_bp: float = 34.0
    position_limits: tuple[PositionLimit, ...] = DEFAULT_POSITION_LIMITS

    def validate(self) -> None:
        for name in ("same_day_cost_bp", "overnight_cost_bp"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.position_limits:
            raise ValueError("position_limits must be nonempty")
        identifiers: set[str] = set()
        for item in self.position_limits:
            item.validate()
            if item.limit_id in identifiers:
                raise ValueError("position limit identifiers must be unique")
            identifiers.add(item.limit_id)


@dataclass(frozen=True)
class ChallengerLedgerResult:
    selected_actions: pl.DataFrame
    selected_paths: pl.DataFrame
    daily_cohort_ledger: pl.DataFrame
    daily_realized_cashflows: pl.DataFrame
    daily_outstanding: pl.DataFrame
    selection_counts: pl.DataFrame
    entry_rank_summary: pl.DataFrame
    position_limit_sweep: pl.DataFrame
    overall_summary: pl.DataFrame


@dataclass(frozen=True)
class FormalChallengerSources:
    decisions: pl.DataFrame
    actions: pl.DataFrame
    paths: pl.DataFrame
    sessions: tuple[str, ...]
    metadata: Mapping[str, object]


def build_prequential_challenger_ledger(
    decisions: pl.DataFrame,
    actions: pl.DataFrame,
    paths: pl.DataFrame,
    sessions: Sequence[str] | Iterable[str],
    *,
    config: ChallengerLedgerConfig = ChallengerLedgerConfig(),
) -> ChallengerLedgerResult:
    """Apply each D-safe diagnostic challenger to target-day AB1/2 actions."""

    config.validate()
    decisions = _normalise_decisions(decisions)
    actions = _normalise_actions(actions)
    paths = _normalise_paths(paths)
    calendar = _normalise_sessions(sessions)
    if not set(decisions["asof_date"].to_list()).issubset(calendar):
        raise ValueError("decision dates are absent from the session calendar")

    choices = _decision_choices(decisions)
    selected_actions = _select_actions(actions, choices)
    selected_paths = _select_paths(paths, selected_actions, choices, config)
    daily = _daily_cohort(decisions, selected_actions, selected_paths, config)
    realized = _daily_realized(selected_paths, calendar, decisions, config)
    outstanding = _daily_outstanding(selected_paths, calendar, decisions)
    selections = _selection_counts(decisions)
    rank_summary = _entry_rank_summary(selected_actions)
    sweep = _position_limit_sweep(selected_paths, config)
    overall = _overall_summary(
        decisions,
        selected_actions,
        selected_paths,
        daily,
        outstanding,
        config,
    )
    result = ChallengerLedgerResult(
        selected_actions=selected_actions,
        selected_paths=selected_paths,
        daily_cohort_ledger=daily,
        daily_realized_cashflows=realized,
        daily_outstanding=outstanding,
        selection_counts=selections,
        entry_rank_summary=rank_summary,
        position_limit_sweep=sweep,
        overall_summary=overall,
    )
    _validate_result(result, decisions, config)
    return result


def load_formal_challenger_sources(
    *,
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    post_cross_root: Path = DEFAULT_POST_CROSS_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
) -> FormalChallengerSources:
    """Verify the frozen roots and retain only target-day challenger actions."""

    execution_root = Path(execution_root).resolve()
    post_cross_root = Path(post_cross_root).resolve()
    prerequisite_root = Path(prerequisite_root).resolve()
    post_marker = _verify_post_cross_inputs(post_cross_root)
    prerequisite_marker = verify_cross_session_prerequisites(prerequisite_root)
    decisions = pl.read_parquet(post_cross_root / "prequential_decisions.parquet")
    choices = _decision_choices(_normalise_decisions(decisions))

    manifest, action_paths = _selected_manifest_paths(
        execution_root,
        dates=None,
        value_codes=None,
    )
    choice_scan = choices.select(
        "Date", "boundary_quantile", "route"
    ).lazy()
    actions = (
        _scan_actions(action_paths)
        .join(choice_scan, on=["Date", "boundary_quantile", "route"], how="inner")
        .filter(_supported_rank_expression())
        .select(*_ACTION_OUTPUT_COLUMNS)
        .collect(engine="streaming")
    )
    paths = pl.read_parquet(post_cross_root / "physical_policy_paths.parquet")
    sessions_path = prerequisite_root / "candidate_sessions.txt"
    sessions = _normalise_sessions(sessions_path.read_text(encoding="utf-8").splitlines())
    metadata = {
        "execution_root": str(execution_root),
        "execution_manifest_sha256": _file_sha256(
            execution_root / "execution_partition_manifest.parquet"
        ),
        "execution_partition_source_inventory_sha256": (
            EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256
        ),
        "execution_manifest_rows": manifest.height,
        "execution_action_rows": int(
            manifest["execution_action_facts_rows"].sum()
        ),
        "post_cross_root": str(post_cross_root),
        "post_cross_complete_sha256": _file_sha256(post_cross_root / "complete.json"),
        "post_cross_paths_sha256": post_marker["artifacts"]
        ["physical_policy_paths.parquet"]["sha256"],
        "post_cross_decisions_sha256": post_marker["artifacts"]
        ["prequential_decisions.parquet"]["sha256"],
        "prerequisite_root": str(prerequisite_root),
        "prerequisite_complete_sha256": _file_sha256(
            prerequisite_root / "complete.json"
        ),
        "prerequisite_marker_payload_sha256": prerequisite_marker.get(
            "marker_payload_sha256"
        ),
        "candidate_sessions_sha256": _file_sha256(sessions_path),
        "candidate_session_count": len(sessions),
        "compact_source_version": FROZEN_COMPACT_VERSION,
    }
    if metadata["execution_manifest_sha256"] != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise ValueError("formal execution manifest changed")
    return FormalChallengerSources(decisions, actions, paths, sessions, metadata)


def run_prequential_challenger_ledger(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    post_cross_root: Path = DEFAULT_POST_CROSS_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
    config: ChallengerLedgerConfig = ChallengerLedgerConfig(),
) -> ChallengerLedgerResult:
    sources = load_formal_challenger_sources(
        execution_root=execution_root,
        post_cross_root=post_cross_root,
        prerequisite_root=prerequisite_root,
    )
    result = build_prequential_challenger_ledger(
        sources.decisions,
        sources.actions,
        sources.paths,
        sources.sessions,
        config=config,
    )
    publish_challenger_ledger(output_root, result, sources.metadata, config=config)
    return result


def publish_challenger_ledger(
    output_root: Path,
    result: ChallengerLedgerResult,
    source_metadata: Mapping[str, object],
    *,
    config: ChallengerLedgerConfig = ChallengerLedgerConfig(),
) -> None:
    """Atomically publish a self-hashed analysis-only bundle."""

    config.validate()
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        declarations: dict[str, dict[str, object]] = {}
        for filename, attribute in ARTIFACTS.items():
            frame = getattr(result, attribute)
            path = stage / filename
            frame.write_parquet(path)
            declarations[filename] = _frame_declaration(path, frame)
        marker: dict[str, object] = {
            "complete": True,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "ledger_version": LEDGER_VERSION,
            "config": _config_payload(config),
            "sources": json.loads(json.dumps(dict(source_metadata), sort_keys=True)),
            "implementation_sources": _implementation_sources(),
            "fact_semantics": _fact_semantics(),
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _verify_published_files(stage, verify_sources=True)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_challenger_ledger(
    output_root: Path,
    *,
    rebuild: bool = True,
) -> dict[str, object]:
    """Verify the exact bundle and optionally rebuild every value from sources."""

    root = Path(output_root)
    marker, frames = _verify_published_files(root, verify_sources=True)
    config = _config_from_payload(marker["config"])
    decisions = pl.read_parquet(
        Path(str(marker["sources"]["post_cross_root"]))
        / "prequential_decisions.parquet"
    )
    result = ChallengerLedgerResult(
        **{attribute: frames[filename] for filename, attribute in ARTIFACTS.items()}
    )
    _validate_result(result, _normalise_decisions(decisions), config)
    if rebuild:
        sources = load_formal_challenger_sources(
            execution_root=Path(str(marker["sources"]["execution_root"])),
            post_cross_root=Path(str(marker["sources"]["post_cross_root"])),
            prerequisite_root=Path(str(marker["sources"]["prerequisite_root"])),
        )
        if dict(sources.metadata) != marker["sources"]:
            raise ValueError("challenger source metadata changed")
        rebuilt = build_prequential_challenger_ledger(
            sources.decisions,
            sources.actions,
            sources.paths,
            sources.sessions,
            config=config,
        )
        for filename, attribute in ARTIFACTS.items():
            actual = frames[filename]
            expected = getattr(rebuilt, attribute)
            if actual.schema != expected.schema or not actual.equals(
                expected, null_equal=True
            ):
                raise ValueError(f"source rebuild differs: {filename}")
    return marker


def _normalise_decisions(frame: pl.DataFrame) -> pl.DataFrame:
    _require(frame, _DECISION_REQUIRED, "prequential decisions")
    result = frame.with_columns(pl.col("asof_date").cast(pl.String)).sort("asof_date")
    if result.is_empty() or result["asof_date"].n_unique() != result.height:
        raise ValueError("prequential decisions must be nonempty and unique by date")
    if result["asof_date"].to_list() != sorted(result["asof_date"].to_list()):
        raise ValueError("prequential decision dates must be ascending")
    invalid = result.filter(
        (pl.col("selected_action_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("diagnostic_challenger_is_selected_action") != False).fill_null(True)  # noqa: E712
        | (pl.col("ranking_uses_only_Date_less_than_asof") != True).fill_null(True)  # noqa: E712
        | (
            pl.col("label_visible_only_if_label_availability_date_less_than_asof")
            != True  # noqa: E712
        ).fill_null(True)
        | (pl.col("nominal_cancel_model_assumption") != True).fill_null(True)  # noqa: E712
        | (pl.col("production_strategy_go") != False).fill_null(True)  # noqa: E712
    )
    if invalid.height:
        raise ValueError("decision table is not a D-safe diagnostic-only NO-GO table")
    nonnull = result.filter(
        pl.col("diagnostic_challenger_boundary_quantile").is_not_null()
    )
    joint_null_count = result.select(
        pl.concat_list(
            pl.col("diagnostic_challenger_boundary_quantile").is_null(),
            pl.col("diagnostic_challenger_entry_route").is_null(),
            pl.col("diagnostic_challenger_exit_rule_id").is_null(),
            pl.col("diagnostic_challenger_exit_route").is_null(),
        )
        .list.n_unique()
        .alias("joint_null_values")
    )
    if joint_null_count.filter(pl.col("joint_null_values") != 1).height:
        raise ValueError("diagnostic challenger keys must be jointly null or non-null")
    if nonnull.filter(
        ~pl.col("diagnostic_challenger_boundary_quantile").is_in([50, 80, 95])
        | ~pl.col("diagnostic_challenger_entry_route").is_in(
            ["future_ask_spot_taker", "spot_bid_future_taker"]
        )
        | ~pl.col("diagnostic_challenger_exit_rule_id").is_in(
            ["frozen_center", "frozen_lower"]
        )
        | ~pl.col("diagnostic_challenger_exit_route").is_in(
            ["future_bid_spot_taker", "spot_ask_future_taker"]
        )
    ).height:
        raise ValueError("diagnostic challenger contains an unsupported cell")
    return result


def _decision_choices(decisions: pl.DataFrame) -> pl.DataFrame:
    return (
        decisions.filter(
            pl.col("diagnostic_challenger_boundary_quantile").is_not_null()
        )
        .select(
            pl.col("asof_date").alias("Date"),
            pl.col("diagnostic_challenger_boundary_quantile")
            .cast(pl.Int64)
            .alias("boundary_quantile"),
            pl.col("diagnostic_challenger_entry_route").alias("route"),
            pl.col("diagnostic_challenger_exit_rule_id").alias("exit_rule_id"),
            pl.col("diagnostic_challenger_exit_route").alias("exit_route"),
            pl.col("diagnostic_challenger_completed_only_after_cost_mean_bp").alias(
                "prior_completed_only_after_cost_mean_bp"
            ),
        )
        .sort("Date")
    )


def _normalise_actions(frame: pl.DataFrame) -> pl.DataFrame:
    _require(frame, _ACTION_REQUIRED, "execution actions")
    result = frame.select(*_ACTION_OUTPUT_COLUMNS).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    )
    if result.select("policy_generation_id").n_unique() != result.height:
        raise ValueError("execution action IDs must be unique")
    return result


def _normalise_paths(frame: pl.DataFrame) -> pl.DataFrame:
    _require(frame, _PATH_REQUIRED, "post-cross physical paths")
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    )
    if result.select("policy_path_id").n_unique() != result.height:
        raise ValueError("post-cross policy_path_id must be unique")
    if result.filter(
        (pl.col("nominal_cancel_model_assumption") != True).fill_null(True)  # noqa: E712
        | (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("post-cross paths violate nominal analysis-only semantics")
    return result


def _supported_rank_expression() -> pl.Expr:
    return (
        (
            (pl.col("route") == "future_ask_spot_taker")
            & (pl.col("maker_market") == "future")
            & (pl.col("maker_side") == "ask")
            & pl.col("target_rank_at_submit").is_in(["ASK1", "ASK2"])
        )
        | (
            (pl.col("route") == "spot_bid_future_taker")
            & (pl.col("maker_market") == "spot")
            & (pl.col("maker_side") == "bid")
            & pl.col("target_rank_at_submit").is_in(["BID1", "BID2"])
        )
    )


def _select_actions(actions: pl.DataFrame, choices: pl.DataFrame) -> pl.DataFrame:
    selected = (
        actions.join(
            choices,
            on=["Date", "boundary_quantile", "route"],
            how="inner",
            validate="m:1",
        )
        .filter(_supported_rank_expression())
        .sort(["Date", "ValueCode", "submit_recv_time_ns", "policy_generation_id"])
    )
    if selected.is_empty():
        raise ValueError("diagnostic challenger has no supported AB1/2 actions")
    invalid = selected.filter(
        pl.col("raw_order_fact_id").is_null()
        | pl.col("policy_generation_id").is_null()
        | pl.col("target_rank_at_submit").is_null()
        | (pl.col("any_fill") != (pl.col("full_fill") | pl.col("partial_fill")))
        | (pl.col("full_fill") & pl.col("partial_fill"))
        | (pl.col("entry_hedge_executable") & ~pl.col("full_fill"))
    )
    if invalid.height:
        raise ValueError("selected AB1/2 execution outcomes are incoherent")
    alias_counts = selected.group_by("raw_order_fact_id").agg(
        pl.len().cast(pl.UInt32).alias("selected_raw_order_alias_count"),
        pl.struct(
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "boundary_quantile",
            "target_rank_at_submit",
            "submit_recv_time_ns",
        ).n_unique().alias("_physical_values"),
    )
    if alias_counts.filter(pl.col("_physical_values") != 1).height:
        raise ValueError("raw order ID crosses physical quote identities")
    return (
        selected.join(
            alias_counts.drop("_physical_values"),
            on="raw_order_fact_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (1.0 / pl.col("selected_raw_order_alias_count")).alias(
                "physical_quote_weight"
            ),
            pl.when(pl.col("full_fill"))
            .then(pl.lit("full_fill"))
            .when(pl.col("partial_fill"))
            .then(pl.lit("partial_fill"))
            .otherwise(pl.lit("no_fill"))
            .alias("entry_outcome_category"),
            pl.lit("route_valid_AB1_AB2_only").alias("rank_scope"),
            pl.lit("independent_policy_alias").alias("quote_sampling_unit"),
            pl.lit(False).alias("joint_volume_allocated"),
            pl.lit(False).alias("diagnostic_challenger_is_selected_action"),
            pl.lit(False).alias("production_strategy_go"),
            pl.when(pl.col("full_fill"))
            .then(
                (pl.col("full_fill_recv_time_ns") - pl.col("submit_recv_time_ns"))
                / 1_000_000_000.0
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("full_fill_wait_seconds"),
        )
    )


def _select_paths(
    paths: pl.DataFrame,
    selected_actions: pl.DataFrame,
    choices: pl.DataFrame,
    config: ChallengerLedgerConfig,
) -> pl.DataFrame:
    path_choices = choices.rename({"route": "entry_route"})
    candidates = paths.join(
        path_choices,
        on=[
            "Date",
            "boundary_quantile",
            "entry_route",
            "exit_rule_id",
            "exit_route",
        ],
        how="inner",
        validate="m:1",
    )
    established = selected_actions.filter(
        pl.col("full_fill") & pl.col("entry_hedge_executable")
    ).select(
        "Date",
        "ValueCode",
        pl.col("route").alias("entry_route"),
        "boundary_quantile",
        "policy_generation_id",
        "raw_order_fact_id",
        "target_rank_at_submit",
    )
    selected = candidates.join(
        established,
        left_on=["Date", "ValueCode", "entry_policy_generation_id"],
        right_on=["Date", "ValueCode", "policy_generation_id"],
        how="inner",
        validate="1:1",
        suffix="_selected_action",
    )
    if selected.height != established.height:
        matched = selected.select(
            "Date", "ValueCode", pl.col("entry_policy_generation_id").alias("policy_generation_id")
        )
        missing = established.join(
            matched,
            on=["Date", "ValueCode", "policy_generation_id"],
            how="anti",
        )
        raise ValueError(
            "selected hedged AB1/2 fills do not map one-to-one to challenger paths: "
            f"missing={missing.height}, paths={selected.height}, actions={established.height}"
        )
    if selected.filter(
        (pl.col("entry_raw_order_fact_id") != pl.col("raw_order_fact_id"))
        | (pl.col("entry_route") != pl.col("entry_route_selected_action"))
        | (
            pl.col("boundary_quantile")
            != pl.col("boundary_quantile_selected_action")
        )
    ).height:
        raise ValueError("selected action/path identity mismatch")
    completed = pl.col("terminal_cashflow_priced") == True  # noqa: E712
    cost = (
        pl.when(pl.col("completed_same_day"))
        .then(pl.lit(float(config.same_day_cost_bp)))
        .when(pl.col("completed_overnight"))
        .then(pl.lit(float(config.overnight_cost_bp)))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return (
        selected.with_columns(cost.alias("diagnostic_cost_bp"))
        .with_columns(
            pl.when(completed)
            .then(pl.col("gross_cycle_bp") - pl.col("diagnostic_cost_bp"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("diagnostic_after_cost_bp"),
            pl.when(completed)
            .then(
                pl.col("gross_cycle_pnl_twd")
                - pl.col("normalization_notional_twd")
                * pl.col("diagnostic_cost_bp")
                / 10_000.0
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("diagnostic_after_cost_twd"),
            pl.lit(False).alias("unresolved_cashflow_imputed"),
            pl.lit(False).alias("diagnostic_challenger_is_selected_action"),
            pl.lit(False).alias("production_strategy_go"),
            pl.lit("target_day_pseudo_oos_diagnostic").alias(
                "evaluation_role"
            ),
        )
        .sort(["Date", "position_established_ns", "ValueCode", "policy_path_id"])
    )


def _daily_cohort(
    decisions: pl.DataFrame,
    actions: pl.DataFrame,
    paths: pl.DataFrame,
    config: ChallengerLedgerConfig,
) -> pl.DataFrame:
    action_groups = _partition_by_date(actions)
    path_groups = _partition_by_date(paths)
    rows: list[dict[str, object]] = []
    for decision in decisions.iter_rows(named=True):
        date = str(decision["asof_date"])
        day_actions = action_groups.get(date, actions.head(0))
        day_paths = path_groups.get(date, paths.head(0))
        full = day_actions.filter(pl.col("full_fill"))
        partial = day_actions.filter(pl.col("partial_fill"))
        no_fill = day_actions.filter(~pl.col("any_fill"))
        completed = day_paths.filter(pl.col("terminal_cashflow_priced"))
        censored = day_paths.filter(
            pl.col("filled_entry_outcome_category") == "censored"
        )
        unknown = day_paths.filter(
            pl.col("filled_entry_outcome_category").is_in(["unknown", "still_open"])
        )
        physical_quotes = day_actions["raw_order_fact_id"].n_unique()
        after_bp_sum = _sum_float(completed, "diagnostic_after_cost_bp")
        rows.append(
            {
                "Date": date,
                "challenger_available": decision[
                    "diagnostic_challenger_boundary_quantile"
                ]
                is not None,
                "boundary_quantile": decision[
                    "diagnostic_challenger_boundary_quantile"
                ],
                "entry_route": decision["diagnostic_challenger_entry_route"],
                "exit_rule_id": decision["diagnostic_challenger_exit_rule_id"],
                "exit_route": decision["diagnostic_challenger_exit_route"],
                "prior_completed_only_after_cost_mean_bp": decision[
                    "diagnostic_challenger_completed_only_after_cost_mean_bp"
                ],
                "supported_ab12_quote_aliases": day_actions.height,
                "supported_ab12_unique_physical_quotes": physical_quotes,
                "quote_alias_excess": day_actions.height - physical_quotes,
                "rank1_quotes": day_actions.filter(
                    pl.col("target_rank_at_submit").is_in(["ASK1", "BID1"])
                ).height,
                "rank2_quotes": day_actions.filter(
                    pl.col("target_rank_at_submit").is_in(["ASK2", "BID2"])
                ).height,
                "full_fills": full.height,
                "partial_fills": partial.height,
                "no_fills": no_fill.height,
                "cancel_required_quotes": day_actions.filter(
                    pl.col("cancel_required")
                ).height,
                "full_fill_hedge_observed": full.filter(
                    pl.col("entry_hedge_label_observed")
                ).height,
                "full_fill_hedge_executable": full.filter(
                    pl.col("entry_hedge_executable")
                ).height,
                "entry_full_fill_rate": (
                    full.height / physical_quotes if physical_quotes else None
                ),
                "entry_partial_fill_rate": (
                    partial.height / physical_quotes if physical_quotes else None
                ),
                "cancel_required_rate": (
                    day_actions.filter(pl.col("cancel_required")).height
                    / physical_quotes
                    if physical_quotes
                    else None
                ),
                "established_positions": day_paths.height,
                "new_entry_one_way_notional_twd": _sum_float(
                    day_paths, "normalization_notional_twd"
                ),
                "completed_cycles": completed.height,
                "completed_same_day": completed.filter(
                    pl.col("completed_same_day")
                ).height,
                "completed_overnight": completed.filter(
                    pl.col("completed_overnight")
                ).height,
                "censored_positions": censored.height,
                "unknown_or_open_positions": unknown.height,
                "completion_rate_given_established": (
                    completed.height / day_paths.height if day_paths.height else None
                ),
                "completed_gross_twd": _sum_float(
                    completed, "gross_cycle_pnl_twd"
                ),
                "completed_gross_mean_bp": _mean_float(completed, "gross_cycle_bp"),
                "completed_gross_p50_bp": _quantile(
                    completed, "gross_cycle_bp", 0.5
                ),
                "completed_after_19_34_twd": _sum_float(
                    completed, "diagnostic_after_cost_twd"
                ),
                "completed_after_19_34_mean_bp": _mean_float(
                    completed, "diagnostic_after_cost_bp"
                ),
                "zero_unresolved_per_submitted_quote_after_19_34_bp": (
                    after_bp_sum / physical_quotes if physical_quotes else None
                ),
                "zero_unresolved_per_established_after_19_34_bp": (
                    after_bp_sum / day_paths.height if day_paths.height else None
                ),
                "terminal_cashflow_point_identified": bool(
                    day_paths.height and completed.height == day_paths.height
                ),
                "unresolved_cashflow_imputed_in_completed_metrics": False,
                "zero_unresolved_columns_are_diagnostic_only": True,
                "diagnostic_challenger_is_selected_action": False,
                "pathwise_ev_ready": False,
                "production_strategy_go": False,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("Date")


def _daily_realized(
    paths: pl.DataFrame,
    sessions: tuple[str, ...],
    decisions: pl.DataFrame,
    config: ChallengerLedgerConfig,
) -> pl.DataFrame:
    first = str(decisions["asof_date"].min())
    last = max(
        [str(decisions["asof_date"].max())]
        + [str(value) for value in paths["label_availability_date"].drop_nulls()]
    )
    dates = [date for date in sessions if first <= date <= last]
    completed = paths.filter(pl.col("terminal_cashflow_priced"))
    groups = _partition_by(completed, "terminal_date")
    rows: list[dict[str, object]] = []
    cumulative_gross = cumulative_after = 0.0
    running_peak_after = 0.0
    for date in dates:
        day = groups.get(date, completed.head(0))
        gross = _sum_float(day, "gross_cycle_pnl_twd")
        after = _sum_float(day, "diagnostic_after_cost_twd")
        cumulative_gross += gross
        cumulative_after += after
        running_peak_after = max(running_peak_after, cumulative_after)
        rows.append(
            {
                "terminal_date": date,
                "completed_cycles": day.height,
                "gross_realized_twd": gross,
                "completed_after_19_34_realized_twd": after,
                "cumulative_gross_realized_twd": cumulative_gross,
                "cumulative_completed_after_19_34_realized_twd": cumulative_after,
                "completed_only_after_19_34_running_peak_twd": running_peak_after,
                "completed_only_after_19_34_drawdown_twd": (
                    cumulative_after - running_peak_after
                ),
                "same_day_cost_bp": float(config.same_day_cost_bp),
                "overnight_cost_bp": float(config.overnight_cost_bp),
                "unresolved_cashflow_imputed": False,
                "date_semantics": "terminal_date_realized_cashflow",
                "pathwise_ev_ready": False,
                "completed_only_equity_curve_not_strategy_equity": True,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("terminal_date")


def _daily_outstanding(
    paths: pl.DataFrame,
    sessions: tuple[str, ...],
    decisions: pl.DataFrame,
) -> pl.DataFrame:
    first = str(decisions["asof_date"].min())
    last = max(
        [str(decisions["asof_date"].max())]
        + [str(value) for value in paths["label_availability_date"].drop_nulls()]
    )
    dates = [date for date in sessions if first <= date <= last]
    index = {date: position for position, date in enumerate(dates)}
    count_delta = [0] * (len(dates) + 1)
    notional_delta = [0.0] * (len(dates) + 1)
    completed_delta = [0] * (len(dates) + 1)
    censored_delta = [0] * (len(dates) + 1)
    unknown_delta = [0] * (len(dates) + 1)
    new_count = [0] * len(dates)
    new_notional = [0.0] * len(dates)
    for item in paths.iter_rows(named=True):
        start = index[str(item["Date"])]
        end_value = item["outstanding_interval_end_exclusive"]
        end = index.get(str(end_value), len(dates)) if end_value else len(dates)
        notional = float(item["normalization_notional_twd"])
        new_count[start] += 1
        new_notional[start] += notional
        if end <= start:
            continue
        count_delta[start] += 1
        count_delta[end] -= 1
        notional_delta[start] += notional
        notional_delta[end] -= notional
        category = str(item["filled_entry_outcome_category"])
        target = (
            completed_delta
            if category == "completed"
            else censored_delta
            if category == "censored"
            else unknown_delta
        )
        target[start] += 1
        target[end] -= 1
    rows: list[dict[str, object]] = []
    live_count = live_completed = live_censored = live_unknown = 0
    live_notional = 0.0
    for position, date in enumerate(dates):
        live_count += count_delta[position]
        live_notional += notional_delta[position]
        live_completed += completed_delta[position]
        live_censored += censored_delta[position]
        live_unknown += unknown_delta[position]
        rows.append(
            {
                "Date": date,
                "new_positions": new_count[position],
                "new_entry_one_way_notional_twd": new_notional[position],
                "outstanding_eod_positions": live_count,
                "outstanding_eod_one_way_entry_notional_twd": live_notional,
                "eventually_completed_outstanding_positions": live_completed,
                "eventually_censored_outstanding_positions": live_censored,
                "eventually_unknown_or_open_outstanding_positions": live_unknown,
                "outstanding_category_breakdown_uses_final_outcome": True,
                "notional_is_eod_mark": False,
                "notional_is_two_leg_gross_exposure": False,
                "notional_is_futures_margin": False,
                "notional_is_capital_requirement": False,
                "partial_entry_inventory_included": False,
                "partial_entry_inventory_unmodeled": True,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("Date")


def _selection_counts(decisions: pl.DataFrame) -> pl.DataFrame:
    selected = decisions.filter(
        pl.col("diagnostic_challenger_boundary_quantile").is_not_null()
    )
    return (
        selected.group_by(
            "diagnostic_challenger_boundary_quantile",
            "diagnostic_challenger_entry_route",
            "diagnostic_challenger_exit_rule_id",
            "diagnostic_challenger_exit_route",
        )
        .agg(
            pl.len().alias("target_sessions"),
            pl.col("asof_date").min().alias("first_target_session"),
            pl.col("asof_date").max().alias("last_target_session"),
            pl.col("diagnostic_challenger_completed_only_after_cost_mean_bp")
            .mean()
            .alias("prior_completed_only_after_cost_mean_bp_mean"),
        )
        .with_columns(
            pl.lit(False).alias("diagnostic_challenger_is_selected_action"),
            pl.lit(False).alias("production_strategy_go"),
        )
        .sort(
            "diagnostic_challenger_boundary_quantile",
            "diagnostic_challenger_entry_route",
            "diagnostic_challenger_exit_rule_id",
            "diagnostic_challenger_exit_route",
        )
    )


def _entry_rank_summary(actions: pl.DataFrame) -> pl.DataFrame:
    return (
        actions.group_by("route", "boundary_quantile", "target_rank_at_submit")
        .agg(
            pl.len().alias("quote_aliases"),
            pl.col("raw_order_fact_id").n_unique().alias("unique_physical_quotes"),
            pl.col("full_fill").sum().alias("full_fills"),
            pl.col("partial_fill").sum().alias("partial_fills"),
            (~pl.col("any_fill")).sum().alias("no_fills"),
            pl.col("cancel_required").sum().alias("cancel_required_quotes"),
            pl.col("entry_hedge_executable").sum().alias(
                "full_fill_hedge_executable"
            ),
            pl.col("full_fill_wait_seconds")
            .drop_nulls()
            .quantile(0.5, interpolation="nearest")
            .alias("full_fill_wait_seconds_p50"),
            pl.col("full_fill_wait_seconds")
            .drop_nulls()
            .quantile(0.9, interpolation="nearest")
            .alias("full_fill_wait_seconds_p90"),
            pl.col("entry_hedge_signed_total_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .mean()
            .alias("entry_50ms_hedge_slippage_bp_mean"),
            pl.col("entry_hedge_signed_total_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .quantile(0.9, interpolation="nearest")
            .alias("entry_50ms_hedge_slippage_bp_p90"),
        )
        .with_columns(
            (pl.col("full_fills") / pl.col("unique_physical_quotes")).alias(
                "full_fill_rate"
            ),
            (pl.col("partial_fills") / pl.col("unique_physical_quotes")).alias(
                "partial_fill_rate"
            ),
            (
                pl.col("cancel_required_quotes")
                / pl.col("unique_physical_quotes")
            ).alias("cancel_required_rate"),
            pl.lit(False).alias("joint_volume_allocated"),
            pl.lit(False).alias("pathwise_ev_ready"),
        )
        .sort("route", "boundary_quantile", "target_rank_at_submit")
    )


def _position_limit_sweep(
    paths: pl.DataFrame, config: ChallengerLedgerConfig
) -> pl.DataFrame:
    ordered = paths.sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )
    rows: list[dict[str, object]] = []
    for limit in config.position_limits:
        active: list[tuple[str, int, str, float, str]] = []
        active_count = 0
        active_notional = 0.0
        active_count_by_product: dict[str, int] = {}
        active_notional_by_product: dict[str, float] = {}
        accepted = rejected_count = rejected_notional = 0
        completed = censored = unknown = 0
        gross = after = accepted_notional = 0.0
        peak_count = 0
        peak_notional = 0.0
        peak_product_count = 0
        peak_product_notional = 0.0
        for item in ordered.iter_rows(named=True):
            entry_key = (str(item["Date"]), int(item["position_established_ns"]))
            while active and (active[0][0], active[0][1]) < entry_key:
                _, _, _, released, product = heapq.heappop(active)
                active_count -= 1
                active_notional -= released
                active_count_by_product[product] -= 1
                active_notional_by_product[product] -= released
            product = str(item["ValueCode"])
            notional = float(item["normalization_notional_twd"])
            scoped_count = (
                active_count
                if limit.scope == "portfolio"
                else active_count_by_product.get(product, 0)
            )
            scoped_notional = (
                active_notional
                if limit.scope == "portfolio"
                else active_notional_by_product.get(product, 0.0)
            )
            count_block = bool(
                limit.max_concurrent_positions is not None
                and scoped_count + 1 > limit.max_concurrent_positions
            )
            notional_block = bool(
                limit.max_outstanding_notional_twd is not None
                and scoped_notional + notional
                > limit.max_outstanding_notional_twd + 1e-9
            )
            if count_block or notional_block:
                rejected_count += int(count_block)
                rejected_notional += int(notional_block)
                continue
            accepted += 1
            accepted_notional += notional
            active_count += 1
            active_notional += notional
            active_count_by_product[product] = active_count_by_product.get(product, 0) + 1
            active_notional_by_product[product] = (
                active_notional_by_product.get(product, 0.0) + notional
            )
            peak_count = max(peak_count, active_count)
            peak_notional = max(peak_notional, active_notional)
            peak_product_count = max(peak_product_count, active_count_by_product[product])
            peak_product_notional = max(
                peak_product_notional, active_notional_by_product[product]
            )
            category = str(item["filled_entry_outcome_category"])
            if category == "completed":
                completed += 1
                gross += float(item["gross_cycle_pnl_twd"])
                after += float(item["diagnostic_after_cost_twd"])
                exit_time = item["exit_decision_time_ns"]
                heapq.heappush(
                    active,
                    (
                        str(item["terminal_date"]),
                        int(exit_time),
                        str(item["policy_path_id"]),
                        notional,
                        product,
                    ),
                )
            elif category == "censored":
                censored += 1
            else:
                unknown += 1
        point_identified = bool(accepted and completed == accepted)
        rows.append(
            {
                **asdict(limit),
                "candidate_positions": ordered.height,
                "accepted_positions": accepted,
                "rejected_positions": ordered.height - accepted,
                "rejected_by_count_cap": rejected_count,
                "rejected_by_notional_cap": rejected_notional,
                "acceptance_rate": accepted / ordered.height if ordered.height else None,
                "accepted_completed_cycles": completed,
                "accepted_censored_positions": censored,
                "accepted_unknown_or_open_positions": unknown,
                "accepted_completed_gross_twd": gross,
                "accepted_completed_after_19_34_twd": after,
                "accepted_new_entry_one_way_notional_twd": accepted_notional,
                "peak_concurrent_positions": peak_count,
                "peak_outstanding_one_way_entry_notional_twd": peak_notional,
                "peak_single_product_concurrent_positions": peak_product_count,
                "peak_single_product_outstanding_one_way_entry_notional_twd": (
                    peak_product_notional
                ),
                "accepted_terminal_cashflow_point_identified": point_identified,
                "position_limit_sweep_analysis_only": True,
                "position_limit_sweep_production_ready": False,
                "joint_volume_allocated": False,
                "unresolved_cashflow_imputed": False,
                "tie_handling": "entry_before_exit_at_equal_recv_time_ns",
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("limit_id")


def _overall_summary(
    decisions: pl.DataFrame,
    actions: pl.DataFrame,
    paths: pl.DataFrame,
    daily: pl.DataFrame,
    outstanding: pl.DataFrame,
    config: ChallengerLedgerConfig,
) -> pl.DataFrame:
    completed = paths.filter(pl.col("terminal_cashflow_priced"))
    physical_quotes = actions["raw_order_fact_id"].n_unique()
    after_bp_sum = _sum_float(completed, "diagnostic_after_cost_bp")
    out_count = outstanding["outstanding_eod_positions"]
    out_notional = outstanding["outstanding_eod_one_way_entry_notional_twd"]
    quote_count = daily.filter(pl.col("challenger_available"))[
        "supported_ab12_unique_physical_quotes"
    ]
    daily_challenger = daily.filter(pl.col("challenger_available"))
    daily_full = daily_challenger["full_fills"]
    daily_established = daily_challenger["established_positions"]
    daily_entry_notional = daily_challenger["new_entry_one_way_notional_twd"]
    return pl.from_dicts(
        [
            {
                "decision_sessions": decisions.height,
                "challenger_sessions": decisions.filter(
                    pl.col("diagnostic_challenger_boundary_quantile").is_not_null()
                ).height,
                "q50_challenger_sessions": decisions.filter(
                    pl.col("diagnostic_challenger_boundary_quantile") == 50
                ).height,
                "q80_challenger_sessions": decisions.filter(
                    pl.col("diagnostic_challenger_boundary_quantile") == 80
                ).height,
                "q95_challenger_sessions": decisions.filter(
                    pl.col("diagnostic_challenger_boundary_quantile") == 95
                ).height,
                "supported_ab12_quote_aliases": actions.height,
                "supported_ab12_unique_physical_quotes": physical_quotes,
                "quote_alias_excess": actions.height - physical_quotes,
                "daily_physical_quotes_p50": _series_quantile(quote_count, 0.5),
                "daily_physical_quotes_p90": _series_quantile(quote_count, 0.9),
                "daily_physical_quotes_max": int(quote_count.max()),
                "daily_full_fills_p50": _series_quantile(daily_full, 0.5),
                "daily_full_fills_p90": _series_quantile(daily_full, 0.9),
                "daily_full_fills_max": int(daily_full.max()),
                "daily_established_positions_p50": _series_quantile(
                    daily_established, 0.5
                ),
                "daily_established_positions_p90": _series_quantile(
                    daily_established, 0.9
                ),
                "daily_established_positions_max": int(daily_established.max()),
                "daily_new_entry_notional_p50_twd": _series_quantile(
                    daily_entry_notional, 0.5
                ),
                "daily_new_entry_notional_p90_twd": _series_quantile(
                    daily_entry_notional, 0.9
                ),
                "daily_new_entry_notional_max_twd": float(
                    daily_entry_notional.max()
                ),
                "full_fills": actions.filter(pl.col("full_fill")).height,
                "partial_fills": actions.filter(pl.col("partial_fill")).height,
                "no_fills": actions.filter(~pl.col("any_fill")).height,
                "cancel_required_quotes": actions.filter(
                    pl.col("cancel_required")
                ).height,
                "full_fill_rate": (
                    actions.filter(pl.col("full_fill")).height / physical_quotes
                ),
                "partial_fill_rate": (
                    actions.filter(pl.col("partial_fill")).height / physical_quotes
                ),
                "cancel_required_rate": (
                    actions.filter(pl.col("cancel_required")).height / physical_quotes
                ),
                "full_fill_hedge_executable": actions.filter(
                    pl.col("full_fill") & pl.col("entry_hedge_executable")
                ).height,
                "full_fill_wait_seconds_p50": _quantile(
                    actions.filter(pl.col("full_fill")),
                    "full_fill_wait_seconds",
                    0.5,
                ),
                "full_fill_wait_seconds_p90": _quantile(
                    actions.filter(pl.col("full_fill")),
                    "full_fill_wait_seconds",
                    0.9,
                ),
                "entry_50ms_hedge_slippage_bp_mean": _mean_float(
                    actions.filter(pl.col("entry_hedge_executable")),
                    "entry_hedge_signed_total_slippage_bp",
                ),
                "entry_50ms_hedge_slippage_bp_p90": _quantile(
                    actions.filter(pl.col("entry_hedge_executable")),
                    "entry_hedge_signed_total_slippage_bp",
                    0.9,
                ),
                "established_positions": paths.height,
                "completed_cycles": completed.height,
                "completed_same_day": completed.filter(
                    pl.col("completed_same_day")
                ).height,
                "completed_overnight": completed.filter(
                    pl.col("completed_overnight")
                ).height,
                "censored_positions": paths.filter(
                    pl.col("filled_entry_outcome_category") == "censored"
                ).height,
                "unknown_or_open_positions": paths.filter(
                    pl.col("filled_entry_outcome_category").is_in(
                        ["unknown", "still_open"]
                    )
                ).height,
                "completion_rate_given_established": completed.height / paths.height,
                "new_entry_one_way_notional_twd": _sum_float(
                    paths, "normalization_notional_twd"
                ),
                "completed_gross_twd": _sum_float(completed, "gross_cycle_pnl_twd"),
                "completed_gross_mean_bp": _mean_float(completed, "gross_cycle_bp"),
                "completed_gross_p50_bp": _quantile(completed, "gross_cycle_bp", 0.5),
                "completed_after_19_34_twd": _sum_float(
                    completed, "diagnostic_after_cost_twd"
                ),
                "completed_after_19_34_mean_bp": _mean_float(
                    completed, "diagnostic_after_cost_bp"
                ),
                "zero_unresolved_per_submitted_quote_after_19_34_bp": (
                    after_bp_sum / physical_quotes
                ),
                "zero_unresolved_per_established_after_19_34_bp": (
                    after_bp_sum / paths.height
                ),
                "daily_outstanding_positions_p50": _series_quantile(out_count, 0.5),
                "daily_outstanding_positions_p90": _series_quantile(out_count, 0.9),
                "daily_outstanding_positions_max": int(out_count.max()),
                "daily_outstanding_notional_p50_twd": _series_quantile(
                    out_notional, 0.5
                ),
                "daily_outstanding_notional_p90_twd": _series_quantile(
                    out_notional, 0.9
                ),
                "daily_outstanding_notional_max_twd": float(out_notional.max()),
                "same_day_cost_bp": float(config.same_day_cost_bp),
                "overnight_cost_bp": float(config.overnight_cost_bp),
                "terminal_cashflow_point_identified": completed.height == paths.height,
                "full_cost_profile_complete": False,
                "strict_cancel_ack_identified": False,
                "partial_entry_inventory_included": False,
                "joint_volume_allocated": False,
                "diagnostic_challenger_is_selected_action": False,
                "best_q_selection_go": False,
                "pathwise_ev_ready": False,
                "production_strategy_go": False,
            }
        ],
        infer_schema_length=None,
    )


def _validate_result(
    result: ChallengerLedgerResult,
    decisions: pl.DataFrame,
    config: ChallengerLedgerConfig,
) -> None:
    del config
    for attribute in ARTIFACTS.values():
        if not isinstance(getattr(result, attribute), pl.DataFrame):
            raise TypeError(f"challenger artifact is not a DataFrame: {attribute}")
    if result.daily_cohort_ledger.height != decisions.height:
        raise ValueError("daily cohort ledger does not cover every decision date")
    expected_challenger_days = decisions.filter(
        pl.col("diagnostic_challenger_boundary_quantile").is_not_null()
    ).height
    if int(result.selection_counts["target_sessions"].sum()) != expected_challenger_days:
        raise ValueError("challenger selection counts do not cover target sessions")
    actions = result.selected_actions
    paths = result.selected_paths
    if actions.filter(~_supported_rank_expression()).height:
        raise ValueError("published selected actions escape AB1/2 route scope")
    if actions.select("policy_generation_id").n_unique() != actions.height:
        raise ValueError("published selected action IDs are duplicated")
    if paths.select("policy_path_id").n_unique() != paths.height:
        raise ValueError("published challenger paths are duplicated")
    established = actions.filter(
        pl.col("full_fill") & pl.col("entry_hedge_executable")
    ).height
    if paths.height != established:
        raise ValueError("established selected action/path count differs")
    if paths.filter(
        (pl.col("diagnostic_challenger_is_selected_action") != False).fill_null(True)  # noqa: E712
        | (pl.col("production_strategy_go") != False).fill_null(True)  # noqa: E712
        | (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("challenger path safety flags changed")
    unresolved = paths.filter(~pl.col("terminal_cashflow_priced"))
    if unresolved.filter(
        pl.col("diagnostic_after_cost_bp").is_not_null()
        | pl.col("diagnostic_after_cost_twd").is_not_null()
    ).height:
        raise ValueError("unresolved challenger paths were assigned cashflow")
    summary = result.overall_summary.row(0, named=True)
    if result.overall_summary.height != 1:
        raise ValueError("overall summary must have one row")
    if int(summary["supported_ab12_quote_aliases"]) != actions.height:
        raise ValueError("overall action count mismatch")
    if int(summary["established_positions"]) != paths.height:
        raise ValueError("overall path count mismatch")
    if summary["production_strategy_go"] is not False:
        raise ValueError("challenger ledger cannot claim production GO")


def _verify_post_cross_inputs(root: Path) -> dict[str, object]:
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = _read_json(marker_path)
    if marker.get("complete") is not True:
        raise ValueError("post-cross source marker is incomplete")
    artifacts = marker.get("artifacts")
    required = {"physical_policy_paths.parquet", "prequential_decisions.parquet"}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("post-cross source lacks required artifacts")
    for filename in required:
        path = root / filename
        declared = artifacts[filename]
        if (
            not isinstance(declared, dict)
            or _file_sha256(path) != declared.get("sha256")
            or path.stat().st_size != int(declared.get("bytes", -1))
        ):
            raise ValueError(f"post-cross source artifact changed: {filename}")
        schema = {name: str(dtype) for name, dtype in pl.read_parquet_schema(path).items()}
        if schema != declared.get("schema"):
            raise ValueError(f"post-cross source schema changed: {filename}")
    metadata = marker.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("production_strategy_go") is not False
        or metadata.get("best_q_selection_go") is not False
        or metadata.get("pathwise_ev_ready") is not False
    ):
        raise ValueError("post-cross source safety contract changed")
    return marker


def _verify_published_files(
    root: Path, *, verify_sources: bool
) -> tuple[dict[str, object], dict[str, pl.DataFrame]]:
    root = Path(root)
    expected = {"complete.json", *ARTIFACTS}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError("challenger bundle inventory is not exact")
    marker = _read_json(root / "complete.json")
    declared_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or marker.get("ledger_version") != LEDGER_VERSION
        or declared_sha != _canonical_sha256(unhashed)
        or marker.get("fact_semantics") != _fact_semantics()
        or marker.get("implementation_sources") != _implementation_sources()
    ):
        raise ValueError("challenger bundle marker is invalid")
    _config_from_payload(marker.get("config"))
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACTS):
        raise ValueError("challenger artifact inventory mismatch")
    frames: dict[str, pl.DataFrame] = {}
    for filename in ARTIFACTS:
        path = root / filename
        frame = pl.read_parquet(path)
        if artifacts[filename] != _frame_declaration(path, frame):
            raise ValueError(f"challenger artifact declaration differs: {filename}")
        frames[filename] = frame
    if verify_sources:
        sources = marker.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("challenger source metadata is invalid")
        execution_root = Path(str(sources["execution_root"]))
        post_root = Path(str(sources["post_cross_root"]))
        prerequisite_root = Path(str(sources["prerequisite_root"]))
        if (
            _file_sha256(execution_root / "execution_partition_manifest.parquet")
            != sources["execution_manifest_sha256"]
            or _file_sha256(post_root / "complete.json")
            != sources["post_cross_complete_sha256"]
            or _file_sha256(prerequisite_root / "complete.json")
            != sources["prerequisite_complete_sha256"]
            or _file_sha256(prerequisite_root / "candidate_sessions.txt")
            != sources["candidate_sessions_sha256"]
        ):
            raise ValueError("challenger formal source hash changed")
        _verify_post_cross_inputs(post_root)
        verify_cross_session_prerequisites(prerequisite_root)
    return marker, frames


def _fact_semantics() -> dict[str, object]:
    return {
        "analysis_only": True,
        "target_day_policy_ranked_only_from_prior_visible_labels": True,
        "route_valid_AB1_AB2_only": True,
        "unsupported_rank_outcomes_imputed": False,
        "entry_hedge_delay_ns": 50_000_000,
        "nominal_cancel_model_assumption": True,
        "strict_cancel_ack_identified": False,
        "independent_entry_event_labels": True,
        "joint_volume_allocated": False,
        "unresolved_cashflow_imputed": False,
        "zero_unresolved_columns_are_diagnostic_only": True,
        "completed_cashflows_include_observed_price_and_latency_slippage": True,
        "same_day_and_overnight_costs_are_sensitivities_not_formal_costs": True,
        "partial_entry_inventory_in_outstanding": False,
        "diagnostic_challenger_is_selected_action": False,
        "best_q_selection_go": False,
        "pathwise_ev_ready": False,
        "production_strategy_go": False,
    }


def _config_payload(config: ChallengerLedgerConfig) -> dict[str, object]:
    return {
        "same_day_cost_bp": float(config.same_day_cost_bp),
        "overnight_cost_bp": float(config.overnight_cost_bp),
        "position_limits": [asdict(item) for item in config.position_limits],
    }


def _config_from_payload(payload: object) -> ChallengerLedgerConfig:
    if not isinstance(payload, dict):
        raise ValueError("challenger config must be an object")
    if set(payload) != {"same_day_cost_bp", "overnight_cost_bp", "position_limits"}:
        raise ValueError("challenger config keys differ")
    limits = payload["position_limits"]
    if not isinstance(limits, list):
        raise ValueError("challenger position limits must be a list")
    config = ChallengerLedgerConfig(
        same_day_cost_bp=float(payload["same_day_cost_bp"]),
        overnight_cost_bp=float(payload["overnight_cost_bp"]),
        position_limits=tuple(PositionLimit(**item) for item in limits),
    )
    config.validate()
    return config


def _implementation_sources() -> dict[str, str]:
    current = Path(__file__)
    dependencies = (
        current,
        current.with_name("compact_frozen_facts.py"),
        current.with_name("post_cross_position_evaluator.py"),
    )
    return {path.name: _file_sha256(path) for path in dependencies}


def _frame_declaration(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _partition_by_date(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return _partition_by(frame, "Date")


def _partition_by(frame: pl.DataFrame, column: str) -> dict[str, pl.DataFrame]:
    result: dict[str, pl.DataFrame] = {}
    if frame.is_empty():
        return result
    for value in frame[column].unique(maintain_order=True).to_list():
        result[str(value)] = frame.filter(pl.col(column) == value)
    return result


def _normalise_sessions(values: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    sessions = tuple(str(value).strip() for value in values if str(value).strip())
    if not sessions or sessions != tuple(sorted(sessions)) or len(sessions) != len(
        set(sessions)
    ):
        raise ValueError("sessions must be nonempty, ascending and unique")
    return sessions


def _sum_float(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty():
        return 0.0
    value = frame[column].sum()
    return 0.0 if value is None else float(value)


def _mean_float(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty():
        return None
    value = frame[column].mean()
    return None if value is None else float(value)


def _quantile(frame: pl.DataFrame, column: str, quantile: float) -> float | None:
    if frame.is_empty():
        return None
    return _series_quantile(frame[column].drop_nulls(), quantile)


def _series_quantile(series: pl.Series, quantile: float) -> float | None:
    if series.is_empty():
        return None
    value = series.quantile(quantile, interpolation="nearest")
    return None if value is None else float(value)


def _require(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--post-cross-root", type=Path, default=DEFAULT_POST_CROSS_ROOT)
    parser.add_argument("--prerequisite-root", type=Path, default=DEFAULT_PREREQUISITE_ROOT)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only is not None:
        marker = verify_challenger_ledger(
            args.verify_only, rebuild=not args.no_rebuild
        )
        print(json.dumps(marker, indent=2, sort_keys=True))
        return 0
    if args.no_rebuild:
        raise ValueError("--no-rebuild is valid only with --verify-only")
    result = run_prequential_challenger_ledger(
        args.output,
        execution_root=args.execution_root,
        post_cross_root=args.post_cross_root,
        prerequisite_root=args.prerequisite_root,
    )
    print(result.overall_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
