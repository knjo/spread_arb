"""Fast compact views over already-frozen execution action facts.

This is the practical fixed-q v1 path.  It does not reopen raw spot/futures
tapes and it does not recompute fills.  Instead it projects the immutable
q50/q80/q95 action facts that already contain exact first-passage stops,
indexed fill cursors, and full-fill+50ms hedge facts.  Spot/future maker ranks
A1/A2/B1/B2 are the supported primary universe; deeper/inside ranks remain in
the candidate audit with null outcomes and are never imputed as no-fill.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import polars as pl

from .compact_evaluator import ONE_SECOND_NS


FROZEN_COMPACT_VERSION = "compact_frozen_execution_facts_v1_l1_l2"
DEFAULT_EXECUTION_ROOT = Path("maker/data/walkforward/execution_narrow_60d")
SUPPORTED_RANKS = ("ASK1", "ASK2", "BID1", "BID2")
SUPPORTED_QUANTILES = (50, 80, 95)
EXPECTED_EXECUTION_CONFIG_SHA256 = (
    "b0a26ee2f6cb42decb8272742e1bd248cb6a550d64896f4a5fb9d3974874ab29"
)
EXPECTED_EXECUTION_RUNNER_VERSION = (
    "product_day_execution_facts_v3_price_ladder"
)
EXPECTED_EXECUTION_MANIFEST_SHA256 = (
    "ba8436a6a085fe8b8716d67294e9202e9d7fe2ba4a874245dd13d1b9a64efe04"
)
EXPECTED_PRODUCT_DAY_UNIVERSE_SHA256 = (
    "95fdb7f88483b79035b1875c279ac536007a6306494f902e47bb43652d0f8df5"
)
EXPECTED_PRODUCT_DAY_KEY_SHA256 = (
    "c65b3b0a5782ed10f60fd8483c34971b08801d712078e18623e6380c692f84dc"
)
EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256 = (
    "476ddb144118ae3696798dc5b9383c32c1ae85d5e9f5a85bc4f4d7d28a6030eb"
)
EXPECTED_EXECUTION_PARTITIONS = 2_687
EXPECTED_EXECUTION_ACTION_ROWS = 4_032_586
EXPECTED_REQUESTED_PRODUCT_DAYS = 2_700
EXPECTED_SESSIONS = 60
_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "boundary_quantile": pl.Int64,
    "spread_pair_epoch": pl.Int64,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "target_price_tick": pl.Int64,
    "target_price": pl.Float64,
    "target_rank_at_submit": pl.String,
    "initial_queue_ahead": pl.Int64,
    "queue_known": pl.Boolean,
    "intended_quantity": pl.Int64,
    "submit_recv_time_ns": pl.Int64,
    "submit_event_sequence": pl.Int64,
    "submit_row_index": pl.Int64,
    "nominal_stop_recv_time_ns": pl.Int64,
    "nominal_stop_reason": pl.String,
    "first_fill_recv_time_ns": pl.Int64,
    "first_fill_event_sequence": pl.Int64,
    "first_fill_row_index": pl.Int64,
    "full_fill_recv_time_ns": pl.Int64,
    "full_fill_event_sequence": pl.Int64,
    "full_fill_row_index": pl.Int64,
    "known_filled_quantity": pl.Int64,
    "any_fill": pl.Boolean,
    "full_fill": pl.Boolean,
    "partial_fill": pl.Boolean,
    "trade_through_fill": pl.Boolean,
    "terminal_recv_time_ns": pl.Int64,
    "terminal_reason": pl.String,
    "cancel_required": pl.Boolean,
    "entry_hedge_status": pl.String,
    "entry_hedge_decision_time_ns": pl.Int64,
    "entry_hedge_executable_vwap_price": pl.Float64,
    "entry_hedge_executed_quantity": pl.Int64,
    "entry_hedge_depth_shortfall": pl.Int64,
    "entry_hedge_decision_book_age_ms": pl.Float64,
    "entry_hedge_signed_latency_slippage_bp": pl.Float64,
    "entry_hedge_signed_depth_slippage_bp": pl.Float64,
    "entry_hedge_signed_total_slippage_bp": pl.Float64,
    "entry_hedge_label_observed": pl.Boolean,
    "entry_hedge_executable": pl.Boolean,
}
_REQUIRED_ACTION_COLUMNS = set(_ACTION_SCHEMA)
_CANONICAL_ROUTES = {
    "future_ask_spot_taker": ("future", "ask", ("ASK1", "ASK2")),
    "spot_bid_future_taker": ("spot", "bid", ("BID1", "BID2")),
}
_SUPPORTED_ACTION = (
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
_IDENTITY_NONNULL_COLUMNS = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "spread_pair_epoch",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_price_tick",
    "target_price",
    "target_rank_at_submit",
    "intended_quantity",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "partial_fill",
    "trade_through_fill",
    "terminal_recv_time_ns",
    "terminal_reason",
    "cancel_required",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
}
_SUPPORTED_NONNULL_COLUMNS = {
    "initial_queue_ahead",
    "queue_known",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "trade_through_fill",
    "cancel_required",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
}


@dataclass(frozen=True)
class FrozenCompactResult:
    order_outcomes: pl.DataFrame
    state_changes: pl.DataFrame
    audit: pl.DataFrame
    q_summary: pl.DataFrame


@dataclass(frozen=True)
class FrozenUniverseSummary:
    q_summary: pl.DataFrame
    product_day_audit: pl.DataFrame
    source_partitions: int
    source_action_rows: int
    source_verification_seconds: float
    projection_validation_aggregate_seconds: float
    elapsed_seconds: float


def load_frozen_compact_product_days(
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    *,
    dates: Iterable[str] | None = None,
    value_codes: Iterable[str] | None = None,
) -> FrozenCompactResult:
    """Load selected persisted partitions and build small reusable rows."""

    manifest, paths = _selected_manifest_paths(
        Path(execution_root), dates=dates, value_codes=value_codes
    )
    actions = _scan_actions(paths).collect(engine="streaming")
    expected = int(manifest.get_column("execution_action_facts_rows").sum())
    if actions.height != expected:
        raise ValueError(
            f"selected action row count mismatch: {actions.height} != {expected}"
        )
    outcomes = build_compact_from_execution_actions(actions)
    state_changes = build_compact_state_changes(outcomes)
    audit = summarize_frozen_product_days(outcomes)
    q_summary = _summarize_lazy(outcomes.lazy()).collect(engine="streaming")
    _validate_summary_partition(q_summary)
    return FrozenCompactResult(outcomes, state_changes, audit, q_summary)


def summarize_frozen_compact_universe(
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    *,
    dates: Iterable[str] | None = None,
    value_codes: Iterable[str] | None = None,
) -> FrozenUniverseSummary:
    """Stream a 60-session summary without retaining four million rows."""

    started = time.perf_counter()
    manifest, paths = _selected_manifest_paths(
        Path(execution_root), dates=dates, value_codes=value_codes
    )
    verified_at = time.perf_counter()
    actions = _scan_actions(paths)
    _validate_action_contract_lazy(actions)
    compact = _compact_lazy(actions)
    summary = _summarize_lazy(compact).collect(engine="streaming")
    _validate_summary_partition(summary)
    audit = _audit_lazy(compact).collect(engine="streaming")
    expected = int(manifest.get_column("execution_action_facts_rows").sum())
    actual = int(audit.get_column("candidate_policy_aliases").sum())
    if actual != expected:
        raise ValueError(f"streamed action row count mismatch: {actual} != {expected}")
    return FrozenUniverseSummary(
        q_summary=summary,
        product_day_audit=audit,
        source_partitions=manifest.height,
        source_action_rows=actual,
        source_verification_seconds=verified_at - started,
        projection_validation_aggregate_seconds=(
            time.perf_counter() - verified_at
        ),
        elapsed_seconds=time.perf_counter() - started,
    )


def build_compact_from_execution_actions(actions: pl.DataFrame) -> pl.DataFrame:
    """Project validated formal actions into the supported compact schema."""

    if actions.is_empty():
        normalized = _normalize_empty_actions(actions)
        return _compact_lazy(normalized.lazy()).collect()
    _require_columns(actions, _REQUIRED_ACTION_COLUMNS, "execution actions")
    normalized = actions.select(
        *[
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in _ACTION_SCHEMA.items()
        ]
    )
    _validate_action_contract(normalized)
    invalid_q = normalized.filter(
        ~pl.col("boundary_quantile").is_in(list(SUPPORTED_QUANTILES))
    )
    if invalid_q.height:
        raise ValueError("execution actions contain unsupported fixed quantiles")
    result = _compact_lazy(normalized.lazy()).collect()
    _validate_compact_projection(result, source_rows=normalized.height)
    return result


def build_compact_state_changes(outcomes: pl.DataFrame) -> pl.DataFrame:
    """Emit two material rows per supported order: submit and terminal."""

    supported = outcomes.filter(pl.col("outcome_supported"))
    submit = supported.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "boundary_quantile",
        "policy_generation_id",
        "raw_order_fact_id",
        "spread_pair_epoch",
        "target_price_tick",
        pl.lit("submit").alias("change_kind"),
        pl.lit("frozen_candidate_submit").alias("change_reason"),
        pl.col("submit_recv_time_ns").alias("change_recv_time_ns"),
        pl.col("submit_event_sequence").alias("change_event_sequence"),
        pl.col("submit_row_index").alias("change_row_index"),
        pl.lit(True).alias("fill_cursor_exact"),
        pl.lit(True).alias("change_cursor_exact"),
        pl.lit("event_cursor").alias("change_cursor_resolution"),
        pl.lit("frozen_execution_action_facts").alias("source"),
    )
    terminal = supported.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "boundary_quantile",
        "policy_generation_id",
        "raw_order_fact_id",
        "spread_pair_epoch",
        "target_price_tick",
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.lit("full_fill"))
        .otherwise(pl.lit("cancel"))
        .alias("change_kind"),
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.lit("full_fill"))
        .otherwise(pl.col("nominal_stop_reason"))
        .alias("change_reason"),
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.col("full_fill_recv_time_ns"))
        .otherwise(pl.col("nominal_stop_recv_time_ns"))
        .alias("change_recv_time_ns"),
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.col("full_fill_event_sequence"))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("change_event_sequence"),
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.col("full_fill_row_index"))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("change_row_index"),
        pl.lit(True).alias("fill_cursor_exact"),
        (pl.col("full_fill") == True).alias("change_cursor_exact"),  # noqa: E712
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.lit("event_cursor"))
        .otherwise(pl.lit("recv_time_ns_only"))
        .alias("change_cursor_resolution"),
        pl.lit("frozen_execution_action_facts").alias("source"),
    )
    return pl.concat([submit, terminal], how="vertical").sort(
        [
            "Date",
            "ValueCode",
            "route",
            "boundary_quantile",
            "change_recv_time_ns",
            "policy_generation_id",
            "change_kind",
        ]
    )


def summarize_frozen_product_days(outcomes: pl.DataFrame) -> pl.DataFrame:
    return _audit_lazy(outcomes.lazy()).collect().sort(
        ["Date", "ValueCode", "route", "boundary_quantile"]
    )


def _compact_lazy(actions: pl.LazyFrame) -> pl.LazyFrame:
    schema = actions.collect_schema()
    missing = sorted(_REQUIRED_ACTION_COLUMNS - set(schema.names()))
    if missing:
        raise ValueError(f"execution actions missing columns: {missing}")
    supported = _SUPPORTED_ACTION
    exact = lambda name: pl.when(supported).then(pl.col(name)).otherwise(None)
    return actions.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "maker_market",
        "maker_side",
        "boundary_quantile",
        "spread_pair_epoch",
        "raw_order_fact_id",
        "policy_generation_id",
        "target_price_tick",
        "target_price",
        "target_rank_at_submit",
        "initial_queue_ahead",
        exact("queue_known").cast(pl.Boolean).alias("queue_known"),
        "intended_quantity",
        "submit_recv_time_ns",
        "submit_event_sequence",
        "submit_row_index",
        "nominal_stop_recv_time_ns",
        "nominal_stop_reason",
        exact("first_fill_recv_time_ns")
        .cast(pl.Int64)
        .alias("first_fill_recv_time_ns"),
        exact("first_fill_event_sequence")
        .cast(pl.Int64)
        .alias("first_fill_event_sequence"),
        exact("first_fill_row_index").cast(pl.Int64).alias("first_fill_row_index"),
        exact("full_fill_recv_time_ns")
        .cast(pl.Int64)
        .alias("full_fill_recv_time_ns"),
        exact("full_fill_event_sequence")
        .cast(pl.Int64)
        .alias("full_fill_event_sequence"),
        exact("full_fill_row_index").cast(pl.Int64).alias("full_fill_row_index"),
        exact("known_filled_quantity")
        .cast(pl.Int64)
        .alias("known_filled_quantity"),
        exact("any_fill").cast(pl.Boolean).alias("any_fill"),
        exact("full_fill").cast(pl.Boolean).alias("full_fill"),
        exact("partial_fill").cast(pl.Boolean).alias("partial_fill"),
        exact("trade_through_fill")
        .cast(pl.Boolean)
        .alias("trade_through_fill"),
        exact("terminal_recv_time_ns").cast(pl.Int64).alias("terminal_recv_time_ns"),
        pl.when(supported)
        .then(pl.col("terminal_reason"))
        .otherwise(pl.lit("unsupported_rank_unknown"))
        .alias("terminal_reason"),
        exact("cancel_required").cast(pl.Boolean).alias("cancel_required"),
        supported.alias("outcome_supported"),
        pl.when(supported)
        .then(pl.lit("frozen_exact_execution_fact"))
        .otherwise(pl.lit("unsupported_unknown"))
        .alias("outcome_status"),
        pl.when(supported)
        .then(pl.lit("frozen_execution_action_facts"))
        .otherwise(pl.lit("excluded_fail_closed"))
        .alias("fill_backend"),
        pl.lit("persisted_formal_exact_stop_fill_hedge").alias(
            "fill_backend_reason"
        ),
        pl.lit(False).alias("makerfill_mapping_exact"),
        supported.alias("fill_outcome_exact_within_model"),
        supported.alias("fill_cursor_exact"),
        pl.lit(0, dtype=pl.Int64).alias("indexed_quantity_query_count"),
        pl.lit(None, dtype=pl.Float64).alias("makerfill_fill_seconds"),
        pl.lit(None, dtype=pl.Int64).alias("makerfill_candidate_fill_ns"),
        pl.lit(None, dtype=pl.Int64).alias("maker_snapshot_sequence"),
        pl.lit(None, dtype=pl.Int64).alias("maker_snapshot_recv_time_ns"),
        pl.lit(None, dtype=pl.Boolean).alias("maker_snapshot_exact_at_decision"),
        (
            supported
            & pl.col("entry_hedge_label_observed")
        ).alias("hedge_estimate_available"),
        exact("entry_hedge_status").alias("hedge_estimate_status"),
        exact("entry_hedge_decision_time_ns")
        .cast(pl.Int64)
        .alias("hedge_estimate_decision_time_ns"),
        exact("entry_hedge_executable_vwap_price")
        .cast(pl.Float64)
        .alias("hedge_estimate_executable_vwap_price"),
        exact("entry_hedge_executed_quantity")
        .cast(pl.Int64)
        .alias("hedge_estimate_executed_quantity"),
        exact("entry_hedge_depth_shortfall")
        .cast(pl.Int64)
        .alias("hedge_estimate_depth_shortfall"),
        exact("entry_hedge_decision_book_age_ms")
        .cast(pl.Float64)
        .alias("hedge_estimate_decision_book_age_ms"),
        exact("entry_hedge_signed_latency_slippage_bp")
        .cast(pl.Float64)
        .alias("hedge_estimate_signed_latency_slippage_bp"),
        exact("entry_hedge_signed_depth_slippage_bp")
        .cast(pl.Float64)
        .alias("hedge_estimate_signed_depth_slippage_bp"),
        exact("entry_hedge_signed_total_slippage_bp")
        .cast(pl.Float64)
        .alias("hedge_estimate_signed_total_slippage_bp"),
        exact("entry_hedge_status").alias("hedge_status"),
        exact("entry_hedge_decision_time_ns")
        .cast(pl.Int64)
        .alias("hedge_decision_time_ns"),
        exact("entry_hedge_executable_vwap_price")
        .cast(pl.Float64)
        .alias("hedge_executable_vwap_price"),
        exact("entry_hedge_executed_quantity")
        .cast(pl.Int64)
        .alias("hedge_executed_quantity"),
        exact("entry_hedge_depth_shortfall")
        .cast(pl.Int64)
        .alias("hedge_depth_shortfall"),
        exact("entry_hedge_decision_book_age_ms")
        .cast(pl.Float64)
        .alias("hedge_decision_book_age_ms"),
        (supported & pl.col("entry_hedge_label_observed")).alias(
            "hedge_label_observed"
        ),
        (supported & pl.col("entry_hedge_executable")).alias(
            "hedge_executable"
        ),
        pl.when(supported & pl.col("full_fill").fill_null(False))
        .then(pl.lit(True))
        .otherwise(None)
        .alias("hedge_input_fill_cursor_exact"),
        pl.lit(True).alias("independent_event_label"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("pathwise_ev_ready"),
        pl.lit(FROZEN_COMPACT_VERSION).alias("compact_source_version"),
    )


def _summarize_lazy(compact: pl.LazyFrame) -> pl.LazyFrame:
    supported = pl.col("outcome_supported")
    full = supported & pl.col("full_fill").fill_null(False)
    partial = supported & pl.col("partial_fill").fill_null(False)
    any_fill = supported & pl.col("any_fill").fill_null(False)
    no_fill = supported & (~pl.col("any_fill").fill_null(False))
    cancel = supported & pl.col("cancel_required").fill_null(False)
    wait = (
        pl.col("full_fill_recv_time_ns") - pl.col("submit_recv_time_ns")
    ) / ONE_SECOND_NS
    groups = ["route", "boundary_quantile", "target_rank_at_submit"]
    return (
        compact.group_by(groups)
        .agg(
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.len().alias("candidate_policy_aliases"),
            pl.col("raw_order_fact_id")
            .n_unique()
            .alias("candidate_unique_physical_orders"),
            supported.sum().alias("supported_outcome_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported)
            .n_unique()
            .alias("supported_unique_physical_orders"),
            (~supported).sum().alias("unsupported_unknown_aliases"),
            pl.col("raw_order_fact_id")
            .filter(~supported)
            .n_unique()
            .alias("unsupported_unique_physical_orders"),
            pl.lit(0, dtype=pl.UInt32).alias("makerfill_approx_aliases"),
            pl.lit(0, dtype=pl.UInt32).alias(
                "makerfill_approx_full_aliases"
            ),
            pl.lit(0, dtype=pl.UInt32).alias(
                "makerfill_approx_full_unique_physical"
            ),
            pl.lit(0, dtype=pl.UInt32).alias(
                "makerfill_approx_cancel_aliases"
            ),
            pl.lit(None, dtype=pl.Float64).alias(
                "makerfill_approx_wait_seconds_p50"
            ),
            pl.lit(None, dtype=pl.Float64).alias(
                "makerfill_approx_wait_seconds_p90"
            ),
            full.sum().alias("supported_full_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(full)
            .n_unique()
            .alias("supported_full_fill_unique_physical"),
            partial.sum().alias("supported_partial_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(partial)
            .n_unique()
            .alias("supported_partial_fill_unique_physical"),
            any_fill.sum().alias("supported_any_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(any_fill)
            .n_unique()
            .alias("supported_any_fill_unique_physical"),
            no_fill.sum().alias("supported_no_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(no_fill)
            .n_unique()
            .alias("supported_no_fill_unique_physical"),
            cancel.sum().alias("supported_cancel_required_aliases"),
            wait.filter(full).quantile(0.5).alias("supported_fill_wait_seconds_p50"),
            wait.filter(full).quantile(0.9).alias("supported_fill_wait_seconds_p90"),
            pl.col("hedge_estimate_available")
            .sum()
            .alias("hedge_estimate_available_aliases"),
            pl.col("hedge_estimate_executable_vwap_price")
            .filter(pl.col("hedge_estimate_available"))
            .median()
            .alias("hedge_estimate_vwap_p50"),
            pl.col("hedge_estimate_signed_total_slippage_bp")
            .filter(pl.col("hedge_estimate_available"))
            .quantile(0.5)
            .alias("hedge_estimate_total_slippage_bp_p50"),
            pl.col("hedge_estimate_signed_total_slippage_bp")
            .filter(pl.col("hedge_estimate_available"))
            .quantile(0.9)
            .alias("hedge_estimate_total_slippage_bp_p90"),
            pl.col("hedge_label_observed")
            .sum()
            .alias("exact_hedge_observed_aliases"),
            pl.col("hedge_executable")
            .sum()
            .alias("exact_hedge_executable_aliases"),
        )
        .with_columns(
            (
                pl.col("candidate_policy_aliases")
                - pl.col("candidate_unique_physical_orders")
            ).alias("policy_alias_rows_above_unique_physical"),
            pl.lit(None, dtype=pl.Float64).alias(
                "makerfill_approx_full_rate"
            ),
            (
                pl.col("hedge_estimate_available_aliases")
                / pl.col("supported_full_fill_aliases")
            ).fill_nan(None).alias("hedge_estimate_coverage_given_full"),
            (
                pl.col("supported_full_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_full_fill_rate"),
            (
                pl.col("supported_partial_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_partial_fill_rate"),
            (
                pl.col("supported_any_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_any_fill_rate"),
            (
                pl.col("supported_no_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_no_fill_rate"),
            (
                pl.col("supported_cancel_required_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_cancel_required_rate"),
            pl.lit(True).alias("full_partial_no_fill_partition_exact"),
            pl.lit(True).alias("partial_is_cancel_required_subset"),
            pl.col("target_rank_at_submit")
            .is_in(list(SUPPORTED_RANKS))
            .alias("makerfill_primary_l1_l2_rank"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("joint_volume_allocated"),
        )
        .sort(groups)
    )


def _audit_lazy(compact: pl.LazyFrame) -> pl.LazyFrame:
    supported = pl.col("outcome_supported")
    groups = ["Date", "ValueCode", "QuoteCode", "route", "boundary_quantile"]
    return (
        compact.group_by(groups)
        .agg(
            pl.len().alias("candidate_policy_aliases"),
            pl.col("raw_order_fact_id")
            .n_unique()
            .alias("candidate_unique_physical_orders"),
            supported.sum().alias("supported_outcome_aliases"),
            (~supported).sum().alias("unsupported_unknown_aliases"),
            (supported & pl.col("full_fill").fill_null(False))
            .sum()
            .alias("supported_full_fill_aliases"),
            pl.col("hedge_label_observed")
            .sum()
            .alias("exact_hedge_observed_aliases"),
            pl.col("hedge_executable")
            .sum()
            .alias("exact_hedge_executable_aliases"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("joint_volume_allocated"),
        )
        .sort(groups)
    )


def _validate_summary_partition(summary: pl.DataFrame) -> None:
    if summary.is_empty():
        return
    invalid = summary.filter(
        (
            pl.col("supported_full_fill_aliases")
            + pl.col("supported_partial_fill_aliases")
            + pl.col("supported_no_fill_aliases")
            != pl.col("supported_outcome_aliases")
        )
        | (
            pl.col("supported_any_fill_aliases")
            != pl.col("supported_full_fill_aliases")
            + pl.col("supported_partial_fill_aliases")
        )
        | (
            pl.col("supported_cancel_required_aliases")
            != pl.col("supported_partial_fill_aliases")
            + pl.col("supported_no_fill_aliases")
        )
        | (pl.col("exact_hedge_observed_aliases") != pl.col("supported_full_fill_aliases"))
        | (
            pl.col("exact_hedge_executable_aliases")
            > pl.col("exact_hedge_observed_aliases")
        )
        | ~pl.col("full_partial_no_fill_partition_exact")
        | ~pl.col("partial_is_cancel_required_subset")
    )
    if invalid.height:
        raise ValueError("compact q summary fill/cancel/hedge partition is incoherent")


def _scan_actions(paths: tuple[Path, ...]) -> pl.LazyFrame:
    """Scan heterogeneous historical schemas into one typed action contract."""

    if not paths:
        return pl.DataFrame(schema=_ACTION_SCHEMA).lazy()
    schema_groups: dict[
        tuple[tuple[str, str], ...], tuple[dict[str, pl.DataType], list[Path]]
    ] = {}
    for path in paths:
        schema = dict(pl.read_parquet_schema(path))
        signature = tuple(
            (column, str(schema.get(column)))
            for column in _ACTION_SCHEMA
        )
        if signature not in schema_groups:
            schema_groups[signature] = (schema, [])
        schema_groups[signature][1].append(path)
    normalized: list[pl.LazyFrame] = []
    for schema, group_paths in schema_groups.values():
        scan = pl.scan_parquet(
            [str(path) for path in group_paths],
            missing_columns="insert",
            extra_columns="ignore",
        )
        normalized.append(
            scan.select(
                *[
                    (
                        pl.col(column).cast(dtype, strict=True)
                        if column in schema
                        else pl.lit(None, dtype=dtype).alias(column)
                    )
                    for column, dtype in _ACTION_SCHEMA.items()
                ]
            )
        )
    return (
        normalized[0]
        if len(normalized) == 1
        else pl.concat(normalized, how="vertical")
    )


def _selected_manifest_paths(
    root: Path,
    *,
    dates: Iterable[str] | None,
    value_codes: Iterable[str] | None,
) -> tuple[pl.DataFrame, tuple[Path, ...]]:
    manifest_path = root / "execution_partition_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pl.read_parquet(manifest_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    required = {
        "Date",
        "ValueCode",
        "partition",
        "complete",
        "execution_action_facts_rows",
    }
    _require_columns(manifest, required, "execution partition manifest")
    _validate_frozen_root_inventory(root, manifest, manifest_path)
    if manifest.select("Date", "ValueCode").n_unique() != manifest.height:
        raise ValueError("execution partition manifest keys are duplicated")
    if manifest.filter(~pl.col("complete")).height:
        raise ValueError("execution partition manifest contains incomplete rows")
    if dates is not None:
        requested = sorted({str(value) for value in dates})
        manifest = manifest.filter(pl.col("Date").is_in(requested))
        missing = sorted(set(requested) - set(manifest["Date"].to_list()))
        if missing:
            raise ValueError(f"requested dates missing from manifest: {missing}")
    if value_codes is not None:
        requested_codes = sorted({str(value) for value in value_codes})
        manifest = manifest.filter(pl.col("ValueCode").is_in(requested_codes))
        missing_codes = sorted(
            set(requested_codes) - set(manifest["ValueCode"].to_list())
        )
        if missing_codes:
            raise ValueError(
                f"requested products missing from selected manifest: {missing_codes}"
            )
    if manifest.is_empty():
        raise ValueError("no execution partitions selected")
    paths: list[Path] = []
    for row in manifest.sort("Date", "ValueCode").iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        expected_partition = root / f"Date={date}" / f"ValueCode={value_code}"
        partition = Path(str(row["partition"]))
        if not partition.is_absolute():
            partition = Path.cwd() / partition
        if partition.resolve() != expected_partition.resolve():
            raise ValueError(
                "execution manifest partition path/identity mismatch: "
                f"{partition}"
            )
        path = partition / "execution_action_facts.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        _verify_action_partition(partition, row, path)
        if int(row["execution_action_facts_rows"]) > 0:
            paths.append(path)
    return manifest, tuple(paths)


def _validate_frozen_root_inventory(
    root: Path, manifest: pl.DataFrame, manifest_path: Path
) -> None:
    """Reject a truncated/repointed manifest before applying subset filters."""

    if _file_sha256(manifest_path) != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise ValueError("execution root manifest is not the frozen full inventory")
    if (
        manifest.height != EXPECTED_EXECUTION_PARTITIONS
        or int(manifest["execution_action_facts_rows"].sum())
        != EXPECTED_EXECUTION_ACTION_ROWS
        or manifest["Date"].n_unique() != EXPECTED_SESSIONS
        or manifest.select("Date", "ValueCode").n_unique() != manifest.height
        or manifest.filter(~pl.col("complete")).height
        or manifest.filter(
            pl.col("config_sha256") != EXPECTED_EXECUTION_CONFIG_SHA256
        ).height
    ):
        raise ValueError("execution root manifest inventory/count/config mismatch")
    keys = manifest.select("Date", "ValueCode").sort(
        "Date", "ValueCode"
    ).to_dicts()
    if _canonical_sha256(keys) != EXPECTED_PRODUCT_DAY_KEY_SHA256:
        raise ValueError("execution root product-day key inventory mismatch")
    universe_path = root / "product_day_universe.csv"
    if (
        not universe_path.is_file()
        or _file_sha256(universe_path) != EXPECTED_PRODUCT_DAY_UNIVERSE_SHA256
    ):
        raise ValueError("execution product-day universe is missing or changed")
    universe = pl.read_csv(universe_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    _require_columns(
        universe,
        {
            "Date",
            "ValueCode",
            "requested",
            "available_daily_mapping",
            "availability_status",
        },
        "execution product-day universe",
    )
    available = universe.filter(pl.col("available_daily_mapping"))
    if (
        universe.height != EXPECTED_REQUESTED_PRODUCT_DAYS
        or universe.filter(pl.col("requested") != True).height  # noqa: E712
        or available.height != EXPECTED_EXECUTION_PARTITIONS
        or available.select("Date", "ValueCode").n_unique() != available.height
        or available.join(
            manifest.select("Date", "ValueCode"),
            on=["Date", "ValueCode"],
            how="anti",
        ).height
        or manifest.select("Date", "ValueCode").join(
            available,
            on=["Date", "ValueCode"],
            how="anti",
        ).height
    ):
        raise ValueError("execution manifest differs from requested/available universe")
    inventory: list[dict[str, object]] = []
    for row in manifest.sort("Date", "ValueCode").iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        marker_path = (
            root / f"Date={date}" / f"ValueCode={value_code}" / "complete.json"
        )
        if not marker_path.is_file():
            raise FileNotFoundError(marker_path)
        marker_bytes = marker_path.read_bytes()
        try:
            payload = json.loads(marker_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid execution marker JSON: {marker_path}") from exc
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        action = (
            artifacts.get("execution_action_facts.parquet")
            if isinstance(artifacts, dict)
            else None
        )
        if not isinstance(action, dict):
            raise ValueError(f"execution marker lacks action lineage: {marker_path}")
        inventory.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
                "action_sha256": action.get("sha256"),
                "action_rows": int(action.get("rows", -1)),
                "action_columns": int(action.get("columns", -1)),
                "action_bytes": int(action.get("bytes", -1)),
                "config_sha256": payload.get("config_sha256"),
                "runner_version": payload.get("runner_version"),
            }
        )
    if _canonical_sha256(inventory) != EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256:
        raise ValueError("execution marker/action source inventory digest mismatch")


def _verify_action_partition(
    partition: Path,
    manifest_row: dict[str, object],
    action_path: Path,
) -> None:
    """Bind each projected file to its immutable completion marker/config."""

    marker_path = partition / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid execution completion marker: {marker_path}") from exc
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"execution partition is not complete: {marker_path}")
    date = str(manifest_row["Date"])
    value_code = str(manifest_row["ValueCode"])
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"execution marker identity mismatch: {marker_path}")
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"execution marker config is invalid: {marker_path}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"execution marker config hash mismatch: {marker_path}")
    if (
        config_sha != str(manifest_row.get("config_sha256"))
        or config_sha != EXPECTED_EXECUTION_CONFIG_SHA256
        or payload.get("runner_version") != EXPECTED_EXECUTION_RUNNER_VERSION
        or config.get("runner_version") != EXPECTED_EXECUTION_RUNNER_VERSION
        or config.get("boundary_quantiles") != list(SUPPORTED_QUANTILES)
        or config.get("hedge_delay_ns") != 50_000_000
        or config.get("routes") != list(_CANONICAL_ROUTES)
    ):
        raise ValueError(f"execution marker is outside the frozen v1 config: {marker_path}")
    semantics = payload.get("fact_semantics")
    if (
        not isinstance(semantics, dict)
        or semantics.get("hedge_delay_ns") != 50_000_000
        or semantics.get("independent_event_label") is not True
        or semantics.get("joint_volume_allocated") is not False
        or semantics.get("same_absolute_price_alias_key") != "raw_order_fact_id"
    ):
        raise ValueError(f"execution marker semantics mismatch: {marker_path}")
    artifacts = payload.get("artifacts")
    metadata = (
        artifacts.get("execution_action_facts.parquet")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"execution action artifact is undeclared: {marker_path}")
    declared_rows = int(metadata.get("rows", -1))
    manifest_rows = int(manifest_row["execution_action_facts_rows"])
    if declared_rows != manifest_rows or declared_rows < 0:
        raise ValueError(f"execution action row count lineage mismatch: {action_path}")
    if int(metadata.get("bytes", -1)) != action_path.stat().st_size:
        raise ValueError(f"execution action byte count mismatch: {action_path}")
    if _file_sha256(action_path) != metadata.get("sha256"):
        raise ValueError(f"execution action hash mismatch: {action_path}")
    schema = pl.read_parquet_schema(action_path)
    if int(metadata.get("columns", -1)) != len(schema):
        raise ValueError(f"execution action column count mismatch: {action_path}")
    if declared_rows:
        actions = pl.read_parquet(action_path)
        _require_columns(actions, _REQUIRED_ACTION_COLUMNS, "execution actions")
        normalized = actions.select(
            *[
                pl.col(column).cast(dtype, strict=True).alias(column)
                for column, dtype in _ACTION_SCHEMA.items()
            ]
        )
        if normalized.filter(
            (pl.col("Date") != date)
            | (pl.col("ValueCode") != value_code)
            | ~pl.col("policy_generation_id").str.starts_with(
                f"{date}/{value_code}/"
            )
        ).height:
            raise ValueError(f"execution action partition identity mismatch: {action_path}")
        _validate_action_contract(normalized)


def _normalize_empty_actions(actions: pl.DataFrame) -> pl.DataFrame:
    if not actions.is_empty():
        raise ValueError("empty action normalization received nonempty input")
    return actions.with_columns(
        *[
            (
                pl.col(column).cast(dtype, strict=True)
                if column in actions.columns
                else pl.lit(None, dtype=dtype).alias(column)
            )
            for column, dtype in _ACTION_SCHEMA.items()
        ]
    ).select(*_ACTION_SCHEMA)


def _validate_action_contract(actions: pl.DataFrame) -> None:
    invalid = actions.filter(_invalid_action_row())
    if invalid.height:
        raise ValueError(
            "supported frozen execution action has incoherent identity/fill/hedge facts"
        )
    if actions.select("policy_generation_id").n_unique() != actions.height:
        raise ValueError("execution action policy_generation_id is not unique")
    full_true = pl.col("full_fill") == True  # noqa: E712
    groups = actions.group_by("raw_order_fact_id").agg(
        pl.struct(
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "maker_market",
            "maker_side",
            "spread_pair_epoch",
            "target_price_tick",
            "target_price",
            "target_rank_at_submit",
            "initial_queue_ahead",
            "queue_known",
            "intended_quantity",
            "submit_recv_time_ns",
            "submit_event_sequence",
            "submit_row_index",
        ).n_unique().alias("physical_identity_values"),
        pl.col("entry_hedge_status").drop_nulls().n_unique().alias("status_values"),
        pl.col("entry_hedge_decision_time_ns")
        .drop_nulls()
        .n_unique()
        .alias("decision_values"),
        pl.col("entry_hedge_status").is_not_null().sum().alias("status_rows"),
        pl.col("entry_hedge_decision_time_ns")
        .is_not_null()
        .sum()
        .alias("decision_rows"),
        full_true.sum().alias("full_aliases"),
        pl.col("full_fill_recv_time_ns")
        .filter(full_true)
        .drop_nulls()
        .n_unique()
        .alias("full_cursor_values"),
    )
    bad_groups = groups.filter(
        pl.col("raw_order_fact_id").is_null()
        | (pl.col("physical_identity_values") != 1)
        | (pl.col("status_values") > 1)
        | (pl.col("decision_values") > 1)
        | (pl.col("full_cursor_values") > 1)
        | ((pl.col("status_rows") > 0) != (pl.col("decision_rows") > 0))
        | (
            ((pl.col("status_rows") > 0) | (pl.col("decision_rows") > 0))
            & (pl.col("full_aliases") == 0)
        )
    )
    if bad_groups.height:
        raise ValueError("execution action aliases disagree on physical hedge facts")


def _validate_action_contract_lazy(actions: pl.LazyFrame) -> None:
    invalid_rows = int(
        actions.select(_invalid_action_row().sum().alias("invalid_rows"))
        .collect(engine="streaming")
        .item()
    )
    if invalid_rows:
        raise ValueError(
            "frozen execution universe contains incoherent supported action facts"
        )


def _invalid_action_row() -> pl.Expr:
    identity_null = pl.any_horizontal(
        [pl.col(column).is_null() for column in _IDENTITY_NONNULL_COLUMNS]
    )
    supported_null = _SUPPORTED_ACTION & pl.any_horizontal(
        [pl.col(column).is_null() for column in _SUPPORTED_NONNULL_COLUMNS]
    )
    canonical_route = (
        (
            (pl.col("route") == "future_ask_spot_taker")
            & (pl.col("maker_market") == "future")
            & (pl.col("maker_side") == "ask")
            & (
                pl.col("target_rank_at_submit").str.starts_with("ASK")
                | pl.col("target_rank_at_submit").is_in(
                    ["inside", "behind_visible"]
                )
            )
        )
        | (
            (pl.col("route") == "spot_bid_future_taker")
            & (pl.col("maker_market") == "spot")
            & (pl.col("maker_side") == "bid")
            & (
                pl.col("target_rank_at_submit").str.starts_with("BID")
                | pl.col("target_rank_at_submit").is_in(
                    ["inside", "behind_visible"]
                )
            )
        )
    )
    supported = _SUPPORTED_ACTION
    any_true = pl.col("any_fill") == True  # noqa: E712
    full_true = pl.col("full_fill") == True  # noqa: E712
    partial_true = pl.col("partial_fill") == True  # noqa: E712
    first_present = pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in (
                "first_fill_recv_time_ns",
                "first_fill_event_sequence",
                "first_fill_row_index",
            )
        ]
    )
    first_any_present = pl.any_horizontal(
        [
            pl.col(column).is_not_null()
            for column in (
                "first_fill_recv_time_ns",
                "first_fill_event_sequence",
                "first_fill_row_index",
            )
        ]
    )
    full_present = pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in (
                "full_fill_recv_time_ns",
                "full_fill_event_sequence",
                "full_fill_row_index",
            )
        ]
    )
    full_any_present = pl.any_horizontal(
        [
            pl.col(column).is_not_null()
            for column in (
                "full_fill_recv_time_ns",
                "full_fill_event_sequence",
                "full_fill_row_index",
            )
        ]
    )
    executable = pl.col("entry_hedge_executable") == True  # noqa: E712
    submit_before_first = _cursor_strictly_before("submit", "first_fill")
    submit_before_full = _cursor_strictly_before("submit", "full_fill")
    bad_supported = supported & (
        supported_null
        | (pl.col("queue_known") != True).fill_null(True)  # noqa: E712
        | (pl.col("initial_queue_ahead") < 0).fill_null(True)
        | (pl.col("known_filled_quantity") < 0).fill_null(True)
        | (
            pl.col("known_filled_quantity") > pl.col("intended_quantity")
        ).fill_null(True)
        | (
            any_true != (pl.col("known_filled_quantity") > 0)
        ).fill_null(True)
        | (
            full_true
            != (pl.col("known_filled_quantity") == pl.col("intended_quantity"))
        ).fill_null(True)
        | (
            partial_true
            != (
                (pl.col("known_filled_quantity") > 0)
                & (
                    pl.col("known_filled_quantity")
                    < pl.col("intended_quantity")
                )
            )
        ).fill_null(True)
        | (first_present != any_true).fill_null(True)
        | (first_any_present != any_true).fill_null(True)
        | (full_present != full_true).fill_null(True)
        | (full_any_present != full_true).fill_null(True)
        | (any_true & ~submit_before_first.fill_null(False))
        | (full_true & ~submit_before_full.fill_null(False))
        | (
            any_true
            & (
                pl.col("first_fill_recv_time_ns")
                > pl.col("nominal_stop_recv_time_ns")
            ).fill_null(True)
        )
        | (
            full_true
            & (
                (pl.col("first_fill_recv_time_ns") > pl.col("full_fill_recv_time_ns"))
                | (
                    (pl.col("first_fill_recv_time_ns") == pl.col("full_fill_recv_time_ns"))
                    & (
                        pl.col("first_fill_event_sequence")
                        > pl.col("full_fill_event_sequence")
                    )
                )
                | (
                    (pl.col("first_fill_recv_time_ns") == pl.col("full_fill_recv_time_ns"))
                    & (
                        pl.col("first_fill_event_sequence")
                        == pl.col("full_fill_event_sequence")
                    )
                    & (pl.col("first_fill_row_index") > pl.col("full_fill_row_index"))
                )
            ).fill_null(True)
        )
        | (pl.col("cancel_required") != ~full_true).fill_null(True)
        | (
            partial_true
            & (
                pl.col("terminal_reason")
                != pl.concat_str(
                    [pl.lit("partial_then_"), pl.col("nominal_stop_reason")]
                )
            ).fill_null(True)
        )
        | (
            (~any_true)
            & (
                pl.col("terminal_reason")
                != pl.col("nominal_stop_reason")
            ).fill_null(True)
        )
        | (
            full_true
            & (
                (pl.col("terminal_recv_time_ns") != pl.col("full_fill_recv_time_ns"))
                .fill_null(True)
                | (pl.col("terminal_reason") != "full_fill").fill_null(True)
                | (pl.col("full_fill_recv_time_ns") > pl.col("nominal_stop_recv_time_ns"))
                .fill_null(True)
                | (pl.col("entry_hedge_label_observed") != True).fill_null(True)  # noqa: E712
                | pl.col("entry_hedge_status").is_null()
                | pl.col("entry_hedge_decision_time_ns").is_null()
                | (
                    pl.col("entry_hedge_decision_time_ns")
                    != pl.col("full_fill_recv_time_ns") + 50_000_000
                ).fill_null(True)
                | (
                    pl.col("entry_hedge_executable")
                    != (pl.col("entry_hedge_status") == "executable")
                ).fill_null(True)
            )
        )
        | (
            (~full_true)
            & (
                (pl.col("terminal_recv_time_ns") != pl.col("nominal_stop_recv_time_ns"))
                .fill_null(True)
                | (pl.col("entry_hedge_label_observed") != False).fill_null(True)  # noqa: E712
                | (pl.col("entry_hedge_executable") != False).fill_null(True)  # noqa: E712
            )
        )
        | (
            executable
            & (
                pl.col("entry_hedge_executable_vwap_price").is_null()
                | ~pl.col("entry_hedge_executable_vwap_price").is_finite()
                | (pl.col("entry_hedge_executed_quantity") <= 0).fill_null(True)
                | (pl.col("entry_hedge_depth_shortfall") != 0).fill_null(True)
            )
        )
    )
    return (
        identity_null
        | ~canonical_route.fill_null(False)
        | ~pl.col("target_price").is_finite().fill_null(False)
        | (pl.col("target_price") <= 0).fill_null(True)
        | (pl.col("target_price_tick") <= 0).fill_null(True)
        | ~pl.col("boundary_quantile")
        .is_in(list(SUPPORTED_QUANTILES))
        .fill_null(False)
        | (pl.col("intended_quantity") <= 0).fill_null(True)
        | (
            pl.col("nominal_stop_recv_time_ns")
            < pl.col("submit_recv_time_ns")
        ).fill_null(True)
        | (
            (pl.col("terminal_recv_time_ns") < pl.col("submit_recv_time_ns"))
            | (
                pl.col("terminal_recv_time_ns")
                > pl.col("nominal_stop_recv_time_ns")
            )
        ).fill_null(True)
        | bad_supported
    )


def _cursor_strictly_before(left: str, right: str) -> pl.Expr:
    return (
        (pl.col(f"{left}_recv_time_ns") < pl.col(f"{right}_recv_time_ns"))
        | (
            (pl.col(f"{left}_recv_time_ns") == pl.col(f"{right}_recv_time_ns"))
            & (
                pl.col(f"{left}_event_sequence")
                < pl.col(f"{right}_event_sequence")
            )
        )
        | (
            (pl.col(f"{left}_recv_time_ns") == pl.col(f"{right}_recv_time_ns"))
            & (
                pl.col(f"{left}_event_sequence")
                == pl.col(f"{right}_event_sequence")
            )
            & (pl.col(f"{left}_row_index") < pl.col(f"{right}_row_index"))
        )
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_compact_projection(result: pl.DataFrame, *, source_rows: int) -> None:
    if result.height != source_rows:
        raise ValueError("compact projection changed the candidate universe")
    invalid = result.filter(
        (~pl.col("outcome_supported"))
        & (
            pl.col("full_fill").is_not_null()
            | pl.col("cancel_required").is_not_null()
            | pl.col("hedge_label_observed")
            | pl.col("hedge_executable")
        )
    )
    if invalid.height:
        raise ValueError("unsupported ranks leaked exact outcomes")
    bad_delay = result.filter(
        pl.col("hedge_label_observed")
        & (
            pl.col("hedge_decision_time_ns")
            != pl.col("full_fill_recv_time_ns") + 50_000_000
        )
    )
    if bad_delay.height:
        raise ValueError("frozen hedge is not exactly full_fill+50ms")


def _require_columns(
    frame: pl.DataFrame, required: set[str], source: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
