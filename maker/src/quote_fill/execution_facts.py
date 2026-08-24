"""Execution facts for lookup-table maker decisions and conservative exits.

This module is deliberately split at the boundary between *observed paths*
and *estimated value*:

* :func:`build_execution_action_facts` joins one independent maker-policy
  alias to its de-duplicated 50 ms hedge result and retains exact cursors;
* :func:`summarize_execution_daily` produces count/sum denominators suitable
  for a D-1 rolling lookup table; and
* :func:`build_executable_taker_exit_path` plus
  :func:`build_executable_exit_facts` add a conservative same-day
  taker/taker close or an explicit carry-at-EOD branch.

No function in this file estimates probabilities.  In particular, aliases
which round to the same absolute maker price retain separate policy rows but
share ``raw_order_fact_id`` and one physical hedge label.  This prevents a
q50/q80 (or later tail-knot) alias collision from fabricating maker fills.

The exit branch is not a maker-exit fill model.  It crosses two full raw books
on a causal wall-clock grid: sell two spot board lots at bid depth and buy one
stock-future contract at executable ask depth.  Fees, tax, financing,
overnight marks and joint visible-volume allocation remain explicit missing
layers, so ``pathwise_ev_ready`` is always false here.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import polars as pl

from .hedge_study import (
    CONTRACT_SHARES,
    FUTURE_ASK_ROUTE,
    SPOT_BID_ROUTE,
    SPOT_HEDGE_LOTS,
    FUTURE_HEDGE_CONTRACTS,
    executable_levels_from_state,
)
from .raw_tape import RawTapeDay


NS_PER_SECOND = 1_000_000_000
DEFAULT_EXIT_GRID_NS = NS_PER_SECOND
DEFAULT_MAX_EXIT_BOOK_AGE_NS = NS_PER_SECOND
SUPPORTED_ENTRY_ROUTES = (FUTURE_ASK_ROUTE, SPOT_BID_ROUTE)

_ALIAS_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "spread_pair_epoch",
    "target_price_tick",
    "target_price",
    "target_rank_at_submit",
    "initial_queue_ahead",
    "intended_quantity",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "first_fill_recv_time_ns",
    "first_fill_event_sequence",
    "first_fill_row_index",
    "full_fill_recv_time_ns",
    "full_fill_event_sequence",
    "full_fill_row_index",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "cancel_required",
    "spot_book_age_ms_at_submit",
    "future_book_age_ms_at_submit",
}

_HEDGE_OPTIONAL_SCHEMA: Mapping[str, pl.DataType] = {
    "status": pl.String,
    "decision_time_ns": pl.Int64,
    "decision_snapshot_recv_time_ns": pl.Int64,
    "decision_snapshot_event_sequence": pl.Int64,
    "decision_snapshot_row_index": pl.Int64,
    "arrival_reference_price": pl.Float64,
    "decision_best_price": pl.Float64,
    "executable_vwap_price": pl.Float64,
    "partial_vwap_price": pl.Float64,
    "available_quantity": pl.Int64,
    "executed_quantity": pl.Int64,
    "depth_shortfall": pl.Int64,
    "levels_swept": pl.Int64,
    "signed_latency_slippage_bp": pl.Float64,
    "signed_depth_slippage_bp": pl.Float64,
    "signed_total_slippage_bp": pl.Float64,
    "decision_book_age_ms": pl.Float64,
    "contract_size_shares": pl.Int64,
}

_ACTION_KEY = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "lookup_action_id",
    "boundary_quantile",
]


def build_execution_action_facts(
    order_aliases: pl.DataFrame,
    hedge_facts: pl.DataFrame,
) -> pl.DataFrame:
    """Join exact entry/hedge outcomes without estimating an EV.

    The result remains at policy-alias level.  ``same_price_policy_aliases``
    reports how many aliases share the same raw maker order; callers must use
    ``raw_order_fact_id`` whenever estimating physical fill or hedge support.
    Action-level comparisons may still use every alias because their nominal
    cancellation horizons can differ after submission.
    """

    if order_aliases.is_empty():
        return pl.DataFrame(schema=_empty_action_schema())
    _require(order_aliases, _ALIAS_REQUIRED, "order aliases")
    _require(hedge_facts, {"raw_order_fact_id"}, "hedge facts")
    if (
        order_aliases.select("policy_generation_id").n_unique()
        != order_aliases.height
    ):
        raise ValueError("policy_generation_id must be unique in order aliases")

    _validate_raw_alias_identity(order_aliases)
    _validate_fill_cursors(order_aliases)
    if hedge_facts.select("raw_order_fact_id").n_unique() != hedge_facts.height:
        raise ValueError("hedge facts must be unique by raw_order_fact_id")

    hedge = _with_optional_columns(hedge_facts, _HEDGE_OPTIONAL_SCHEMA).select(
        "raw_order_fact_id",
        *[
            pl.col(column).alias(f"entry_hedge_{column}")
            for column in _HEDGE_OPTIONAL_SCHEMA
        ],
    )
    alias_counts = order_aliases.group_by("raw_order_fact_id").agg(
        pl.len().alias("same_price_policy_aliases")
    )
    facts = order_aliases.join(
        alias_counts,
        on="raw_order_fact_id",
        how="left",
        validate="m:1",
    ).join(
        hedge,
        on="raw_order_fact_id",
        how="left",
        validate="m:1",
    )

    action_id = (
        pl.col("lookup_action_id").cast(pl.String)
        if "lookup_action_id" in facts.columns
        else pl.concat_str(
            pl.lit("q"),
            pl.col("boundary_quantile").cast(pl.String),
        )
    )
    local_second = (
        (pl.col("submit_recv_time_ns") // NS_PER_SECOND + 8 * 3600)
        % (24 * 3600)
    )
    max_age = pl.max_horizontal(
        pl.col("spot_book_age_ms_at_submit"),
        pl.col("future_book_age_ms_at_submit"),
    )
    facts = facts.with_columns(
        action_id.alias("lookup_action_id"),
        (pl.col("same_price_policy_aliases") > 1).alias(
            "same_absolute_price_alias"
        ),
        pl.when(
            pl.col("target_rank_at_submit").is_in(
                ["A1", "B1", "ASK1", "BID1"]
            )
        )
        .then(pl.lit("at_bbo"))
        .when(pl.col("target_rank_at_submit") == "inside")
        .then(pl.lit("inside"))
        .otherwise(pl.lit("behind"))
        .alias("rank_bucket"),
        pl.when(pl.col("initial_queue_ahead").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("initial_queue_ahead") <= 1)
        .then(pl.lit("00_0to1"))
        .when(pl.col("initial_queue_ahead") <= 5)
        .then(pl.lit("01_2to5"))
        .when(pl.col("initial_queue_ahead") <= 20)
        .then(pl.lit("02_6to20"))
        .otherwise(pl.lit("03_21plus"))
        .alias("queue_bucket"),
        pl.when(local_second < 9 * 3600 + 30 * 60)
        .then(pl.lit("open_0905_0930"))
        .when(local_second < 12 * 3600)
        .then(pl.lit("mid_0930_1200"))
        .otherwise(pl.lit("late_1200_1320"))
        .alias("tod_bucket"),
        pl.when(max_age.is_null())
        .then(pl.lit("unknown"))
        .when(max_age <= 100)
        .then(pl.lit("fresh_le100ms"))
        .when(max_age <= 1_000)
        .then(pl.lit("fresh_le1000ms"))
        .otherwise(pl.lit("stale_gt1000ms"))
        .alias("freshness_bucket"),
    )

    full = pl.col("full_fill") == True  # noqa: E712
    hedge_status = pl.col("entry_hedge_status")
    facts = facts.with_columns(
        (full & hedge_status.is_not_null()).fill_null(False).alias(
            "entry_hedge_label_observed"
        ),
        (full & (hedge_status == "executable")).fill_null(False).alias(
            "entry_hedge_executable"
        ),
        pl.when(full & (hedge_status == "executable"))
        .then(pl.lit("full_fill_hedge_executable"))
        .when(full & (hedge_status == "insufficient_depth"))
        .then(pl.lit("full_fill_hedge_insufficient_depth"))
        .when(full & hedge_status.is_not_null())
        .then(pl.lit("full_fill_hedge_invalid_state"))
        .when(full)
        .then(pl.lit("full_fill_hedge_unlabeled"))
        .when(pl.col("partial_fill"))
        .then(pl.lit("partial_fill_then_cancel"))
        .when(pl.col("any_fill").is_null())
        .then(pl.lit("unknown_queue_then_cancel"))
        .otherwise(pl.lit("no_fill_then_cancel"))
        .alias("entry_execution_outcome"),
    )

    future_price = (
        pl.when(pl.col("route") == FUTURE_ASK_ROUTE)
        .then(pl.col("target_price"))
        .when(pl.col("route") == SPOT_BID_ROUTE)
        .then(pl.col("entry_hedge_executable_vwap_price"))
        .otherwise(None)
    )
    spot_price = (
        pl.when(pl.col("route") == FUTURE_ASK_ROUTE)
        .then(pl.col("entry_hedge_executable_vwap_price"))
        .when(pl.col("route") == SPOT_BID_ROUTE)
        .then(pl.col("target_price"))
        .otherwise(None)
    )
    facts = facts.with_columns(
        future_price.alias("entry_future_price"),
        spot_price.alias("entry_spot_price"),
    ).with_columns(
        pl.when(pl.col("entry_hedge_executable"))
        .then(
            (
                pl.col("entry_future_price") / pl.col("entry_spot_price")
                - 1.0
            )
            * 10_000.0
        )
        .otherwise(None)
        .alias("entry_locked_basis_bp"),
        pl.when(pl.col("entry_hedge_executable"))
        .then(
            (
                pl.col("entry_future_price") - pl.col("entry_spot_price")
            )
            * pl.col("entry_hedge_contract_size_shares")
        )
        .otherwise(None)
        .alias("entry_gross_cash_edge_twd"),
        pl.when(pl.col("entry_hedge_executable"))
        .then(pl.lit("pending_executable_exit_rule"))
        .otherwise(pl.lit("not_opened_or_unhedged"))
        .alias("same_day_exit_status"),
        pl.when(pl.col("entry_hedge_executable"))
        .then(pl.lit("pending_if_no_same_day_exit"))
        .otherwise(pl.lit("not_opened_or_unhedged"))
        .alias("overnight_branch_status"),
        pl.lit("not_applied").alias("fees_tax_status"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("executable_maker_exit_included"),
        pl.lit(False).alias("fees_tax_included"),
        pl.lit(False).alias("overnight_cost_included"),
        pl.lit(False).alias("pathwise_ev_ready"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )
    return facts.sort(
        ["Date", "ValueCode", "route", "lookup_action_id", "submit_recv_time_ns"]
    )


def summarize_execution_daily(
    action_facts: pl.DataFrame,
    policy_support: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aggregate daily numerators and denominators for future D-1 tables.

    ``policy_support`` may contain zero-order action rows.  When supplied, its
    key columns are used as a left-hand grid so an unavailable action-day is
    not silently omitted from later probability estimation.
    """

    if action_facts.is_empty():
        aggregated = pl.DataFrame(schema=_empty_daily_schema())
    else:
        _require(
            action_facts,
            set(_ACTION_KEY)
            | {
                "raw_order_fact_id",
                "same_absolute_price_alias",
                "queue_known",
                "any_fill",
                "full_fill",
                "partial_fill",
                "cancel_required",
                "nominal_stop_reason",
                "entry_hedge_label_observed",
                "entry_hedge_executable",
                "entry_hedge_status",
                "entry_hedge_signed_latency_slippage_bp",
                "entry_hedge_signed_depth_slippage_bp",
                "entry_hedge_signed_total_slippage_bp",
                "entry_locked_basis_bp",
                "full_fill_recv_time_ns",
            },
            "execution action facts",
        )
        aggregated = action_facts.group_by(_ACTION_KEY).agg(
            pl.len().alias("policy_alias_orders"),
            pl.col("raw_order_fact_id").n_unique().alias("unique_raw_order_facts"),
            pl.col("same_absolute_price_alias").sum().alias(
                "same_price_alias_rows"
            ),
            pl.col("queue_known").sum().alias("queue_known_orders"),
            pl.col("any_fill").is_not_null().sum().alias("any_fill_observed_n"),
            pl.col("any_fill").fill_null(False).sum().alias("any_fills"),
            pl.col("full_fill").is_not_null().sum().alias(
                "full_fill_observed_n"
            ),
            pl.col("full_fill").fill_null(False).sum().alias("full_fills"),
            pl.col("partial_fill").sum().alias("partial_fills"),
            pl.col("cancel_required").sum().alias("cancel_required_orders"),
            (
                pl.col("cancel_required")
                & (pl.col("nominal_stop_reason") == "target_retreat")
            )
            .sum()
            .alias("target_retreat_cancels"),
            (
                pl.col("cancel_required")
                & (pl.col("nominal_stop_reason") == "session_cutoff")
            )
            .sum()
            .alias("session_cutoff_cancels"),
            pl.col("entry_hedge_label_observed").sum().alias(
                "hedge_label_observed"
            ),
            pl.col("entry_hedge_executable").sum().alias("hedge_executable"),
            (pl.col("entry_hedge_status") == "insufficient_depth")
            .sum()
            .alias("hedge_insufficient_depth"),
            pl.col("full_fill_recv_time_ns")
            .is_not_null()
            .sum()
            .alias("exact_full_fill_cursor_rows"),
            pl.col("entry_hedge_signed_latency_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .median()
            .alias("hedge_latency_slippage_bp_p50"),
            pl.col("entry_hedge_signed_depth_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .median()
            .alias("hedge_depth_slippage_bp_p50"),
            pl.col("entry_hedge_signed_total_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .median()
            .alias("hedge_total_slippage_bp_p50"),
            pl.col("entry_hedge_signed_total_slippage_bp")
            .filter(pl.col("entry_hedge_executable"))
            .quantile(0.95, interpolation="nearest")
            .alias("hedge_total_slippage_bp_p95"),
            pl.col("entry_locked_basis_bp")
            .filter(pl.col("entry_hedge_executable"))
            .median()
            .alias("entry_locked_basis_bp_p50"),
        ).with_columns(
            _safe_rate("any_fills", "policy_alias_orders", "p_any_fill_lower_bound"),
            _safe_rate("full_fills", "policy_alias_orders", "p_full_fill_lower_bound"),
            _safe_rate(
                "cancel_required_orders",
                "policy_alias_orders",
                "cancel_required_rate",
            ),
            _safe_rate(
                "hedge_executable",
                "full_fills",
                "p_hedge_executable_given_full_fill",
            ),
            _safe_rate(
                "hedge_insufficient_depth",
                "full_fills",
                "p_hedge_depth_shortfall_given_full_fill",
            ),
            pl.lit(False).alias("pathwise_ev_ready"),
        )

    if policy_support is None:
        return aggregated.sort(_ACTION_KEY) if not aggregated.is_empty() else aggregated
    _require(policy_support, set(_ACTION_KEY), "policy support")
    support = policy_support.select(
        *dict.fromkeys(
            [*_ACTION_KEY, *[column for column in policy_support.columns if column not in _ACTION_KEY]]
        )
    )
    if support.select(_ACTION_KEY).n_unique() != support.height:
        raise ValueError("policy support must be unique by daily action key")
    result = support.join(
        aggregated,
        on=_ACTION_KEY,
        how="left",
        validate="1:1",
    )
    count_columns = [
        column
        for column in (
            "policy_alias_orders",
            "unique_raw_order_facts",
            "same_price_alias_rows",
            "queue_known_orders",
            "any_fill_observed_n",
            "any_fills",
            "full_fill_observed_n",
            "full_fills",
            "partial_fills",
            "cancel_required_orders",
            "target_retreat_cancels",
            "session_cutoff_cancels",
            "hedge_label_observed",
            "hedge_executable",
            "hedge_insufficient_depth",
            "exact_full_fill_cursor_rows",
        )
        if column in result.columns
    ]
    result = result.with_columns(
        *[pl.col(column).fill_null(0) for column in count_columns],
        pl.lit(False).alias("pathwise_ev_ready"),
    )
    return result.sort(_ACTION_KEY)


@dataclass(frozen=True)
class ExecutableExitPathConfig:
    """Causal wall-clock grid for an immediate taker/taker close."""

    grid_ns: int = DEFAULT_EXIT_GRID_NS
    max_book_age_ns: int | None = DEFAULT_MAX_EXIT_BOOK_AGE_NS

    def validate(self) -> None:
        if (
            isinstance(self.grid_ns, bool)
            or not isinstance(self.grid_ns, int)
            or self.grid_ns <= 0
        ):
            raise ValueError("grid_ns must be a positive integer")
        if self.max_book_age_ns is not None and (
            isinstance(self.max_book_age_ns, bool)
            or not isinstance(self.max_book_age_ns, int)
            or self.max_book_age_ns < 0
        ):
            raise ValueError("max_book_age_ns must be non-negative or None")


class _StateFrameIndex:
    def __init__(self, frame: pl.DataFrame, market: str) -> None:
        _require(
            frame,
            {
                "recv_time_ns",
                "sequence",
                "trial_match",
                "ref_price",
                "book_state_available",
                "book_recv_time_ns",
                *(f"{side}_price_{level}" for side in ("bid", "ask") for level in range(1, 6)),
                *(f"{side}_lots_{level}" for side in ("bid", "ask") for level in range(1, 6)),
            },
            f"raw {market} exit states",
        )
        sort_columns = ["recv_time_ns", "sequence"]
        if "packet_sequence" in frame.columns:
            sort_columns.append("packet_sequence")
        self.market = market
        ordered = frame.sort(sort_columns)
        trials = [bool(value) for value in ordered["trial_match"].to_list()]
        raw_book_column = (
            "raw_has_book"
            if "raw_has_book" in ordered.columns
            else "book_update"
            if "book_update" in ordered.columns
            else None
        )
        raw_books = (
            [bool(value) for value in ordered[raw_book_column].to_list()]
            if raw_book_column is not None
            else [bool(value) for value in ordered["book_state_available"].to_list()]
        )
        formal_flags: list[bool] = []
        formal_after_trial = False
        for trial, raw_book in zip(trials, raw_books, strict=True):
            if trial:
                formal_after_trial = False
            elif raw_book:
                formal_after_trial = True
            formal_flags.append((not trial) and formal_after_trial)
        self.frame = ordered.with_columns(
            pl.Series("_formal_after_trial", formal_flags, dtype=pl.Boolean)
        )
        self.recv_times = tuple(
            int(value) for value in self.frame["recv_time_ns"].to_list()
        )

    def latest(self, timestamp_ns: int) -> dict[str, object] | None:
        index = bisect_right(self.recv_times, timestamp_ns) - 1
        return None if index < 0 else self.frame.row(index, named=True)


def build_executable_taker_exit_path(
    raw_tape: RawTapeDay,
    value_code: str,
    *,
    start_time_ns: int,
    cutoff_time_ns: int,
    config: ExecutableExitPathConfig = ExecutableExitPathConfig(),
) -> pl.DataFrame:
    """Build one product-day causal spot-bid/future-ask close path.

    Every row is one decision grid.  A row is ``executable`` only when both
    books are formal, inside the reference-price gate, fresh enough, and have
    complete L1--L5 depth for two spot lots and one futures contract.
    ``exit_basis_bp`` uses the actual two-leg VWAPs, not B1/A1 alone.
    """

    config.validate()
    if (
        isinstance(start_time_ns, bool)
        or isinstance(cutoff_time_ns, bool)
        or not isinstance(start_time_ns, int)
        or not isinstance(cutoff_time_ns, int)
        or start_time_ns < 0
        or cutoff_time_ns <= start_time_ns
    ):
        raise ValueError("exit path requires 0 <= start_time_ns < cutoff_time_ns")
    selected_mapping = raw_tape.mapping.filter(
        pl.col("ValueCode").cast(pl.String) == str(value_code)
    )
    if selected_mapping.height != 1:
        raise ValueError("exit path needs exactly one contract mapping")
    quote_code = str(selected_mapping.item(0, "QuoteCode"))
    contract_size = float(selected_mapping.item(0, "contract_size"))
    if not math.isclose(contract_size, CONTRACT_SHARES, abs_tol=1e-9):
        raise ValueError(
            f"exit quantity contract requires {CONTRACT_SHARES} shares, got {contract_size}"
        )

    spot = raw_tape.spot_states.filter(
        pl.col("ValueCode").cast(pl.String) == str(value_code)
    )
    future = raw_tape.future_states.filter(
        pl.col("ValueCode").cast(pl.String) == str(value_code)
    )
    if spot.is_empty() or future.is_empty():
        raise ValueError("exit path is missing spot or future raw states")
    spot_index = _StateFrameIndex(spot, "spot")
    future_index = _StateFrameIndex(future, "future")

    first_grid = _ceil_multiple(start_time_ns, config.grid_ns)
    records: list[dict[str, object]] = []
    for timestamp_ns in range(first_grid, cutoff_time_ns, config.grid_ns):
        spot_row = spot_index.latest(timestamp_ns)
        future_row = future_index.latest(timestamp_ns)
        spot_status = _exit_book_status(
            spot_row,
            "spot",
            timestamp_ns,
            config.max_book_age_ns,
        )
        future_status = _exit_book_status(
            future_row,
            "future",
            timestamp_ns,
            config.max_book_age_ns,
        )
        spot_price = future_price = None
        spot_available = future_available = 0
        status = "executable"
        if spot_status != "ok":
            status = f"spot_{spot_status}"
        elif future_status != "ok":
            status = f"future_{future_status}"
        else:
            assert spot_row is not None and future_row is not None
            spot_price, spot_available = _sweep_vwap(
                executable_levels_from_state(spot_row, "spot", "bid"),
                SPOT_HEDGE_LOTS,
            )
            future_price, future_available = _sweep_vwap(
                executable_levels_from_state(future_row, "future", "ask"),
                FUTURE_HEDGE_CONTRACTS,
            )
            if spot_price is None:
                status = "spot_insufficient_bid_depth"
            elif future_price is None:
                status = "future_insufficient_ask_depth"

        executable = status == "executable"
        basis = (
            (future_price / spot_price - 1.0) * 10_000.0
            if executable and spot_price is not None and future_price is not None
            else None
        )
        records.append(
            {
                "Date": str(raw_tape.date),
                "ValueCode": str(value_code),
                "QuoteCode": quote_code,
                "exit_decision_time_ns": timestamp_ns,
                "status": status,
                "executable": executable,
                "spot_snapshot_recv_time_ns": _row_int(spot_row, "recv_time_ns"),
                "spot_snapshot_sequence": _row_int(spot_row, "sequence"),
                "spot_book_recv_time_ns": _row_int(spot_row, "book_recv_time_ns"),
                "future_snapshot_recv_time_ns": _row_int(future_row, "recv_time_ns"),
                "future_snapshot_sequence": _row_int(future_row, "sequence"),
                "future_book_recv_time_ns": _row_int(future_row, "book_recv_time_ns"),
                "spot_book_age_ms": _book_age_ms(spot_row, timestamp_ns),
                "future_book_age_ms": _book_age_ms(future_row, timestamp_ns),
                "spot_bid_available_lots": spot_available,
                "future_ask_available_contracts": future_available,
                "spot_sell_vwap_price": spot_price,
                "future_buy_vwap_price": future_price,
                "exit_basis_bp": basis,
                "contract_size_shares": int(contract_size),
                "spot_exit_lots": SPOT_HEDGE_LOTS,
                "future_exit_contracts": FUTURE_HEDGE_CONTRACTS,
                "causal_raw_asof": True,
                "joint_volume_allocated": False,
            }
        )
    return pl.from_dicts(records, infer_schema_length=None)


def build_executable_exit_facts(
    action_facts: pl.DataFrame,
    exit_rules: pl.DataFrame,
    exit_path: pl.DataFrame,
) -> pl.DataFrame:
    """Apply pre-target-day exit rules to a causal taker/taker close path.

    Rules are keyed by ``policy_generation_id`` and ``exit_rule_id``.  They
    must carry a strictly prior ``source_asof_date`` and explicitly state
    ``contains_target_day_outcome = false``.  The first executable grid whose
    VWAP basis is at or below the threshold is a same-day close.  Otherwise an
    opened/hedged position is labelled ``carry_at_eod``; an EOD liquidation
    mark is diagnostic only and is not treated as an executed close.
    """

    _require(
        action_facts,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "raw_order_fact_id",
            "policy_generation_id",
            "full_fill",
            "partial_fill",
            "any_fill",
            "full_fill_recv_time_ns",
            "entry_hedge_status",
            "entry_hedge_decision_time_ns",
            "entry_future_price",
            "entry_spot_price",
            "entry_hedge_contract_size_shares",
        },
        "execution action facts",
    )
    _require(
        exit_rules,
        {
            "policy_generation_id",
            "exit_rule_id",
            "exit_threshold_basis_bp",
            "source_asof_date",
            "contains_target_day_outcome",
        },
        "exit rules",
    )
    _require(
        exit_path,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "exit_decision_time_ns",
            "status",
            "executable",
            "spot_snapshot_recv_time_ns",
            "spot_snapshot_sequence",
            "future_snapshot_recv_time_ns",
            "future_snapshot_sequence",
            "spot_sell_vwap_price",
            "future_buy_vwap_price",
            "exit_basis_bp",
        },
        "executable exit path",
    )
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("execution facts must be unique by policy_generation_id")
    rule_key = ["policy_generation_id", "exit_rule_id"]
    if exit_rules.select(rule_key).n_unique() != exit_rules.height:
        raise ValueError("exit rules contain duplicate policy/rule keys")
    if exit_rules.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("exit rules contain target-day outcomes")

    rules_for_join = exit_rules.rename(
        {
            "source_asof_date": "_exit_rule_source_asof_date",
            "contains_target_day_outcome": "_exit_rule_contains_target_day_outcome",
        }
    )
    joined = action_facts.join(
        rules_for_join,
        on="policy_generation_id",
        how="inner",
        validate="1:m",
    )
    malformed = joined.select(
        pl.col("_exit_rule_source_asof_date")
        .cast(pl.String)
        .alias("_source"),
        pl.col("Date").cast(pl.String).alias("_date"),
    ).filter(
        (pl.col("_source").str.len_chars() != 8)
        | (pl.col("_date").str.len_chars() != 8)
        | (pl.col("_source") >= pl.col("_date"))
    )
    if malformed.height:
        raise ValueError("exit rule source_asof_date must be strictly before Date")
    invalid_threshold = joined.filter(
        ~pl.col("exit_threshold_basis_bp").is_finite()
    )
    if invalid_threshold.height:
        raise ValueError("exit thresholds must be finite")

    path_identity = exit_path.select(
        "Date", "ValueCode", "QuoteCode"
    ).unique()
    action_identity = joined.select(
        "Date", "ValueCode", "QuoteCode"
    ).unique()
    if path_identity.height != 1 or action_identity.height != 1:
        raise ValueError(
            "executable exit builder requires one exact product-day slice"
        )
    if path_identity.row(0) != action_identity.row(0):
        raise ValueError(
            "exit path Date/ValueCode/QuoteCode does not match action slice"
        )

    path_rows = list(
        exit_path.sort("exit_decision_time_ns").iter_rows(named=True)
    )
    eligible = [row for row in path_rows if row["executable"] is True]
    eligible_times = [int(row["exit_decision_time_ns"]) for row in eligible]
    threshold_cache: dict[float, tuple[list[int], list[dict[str, object]]]] = {}

    def candidates(threshold: float) -> tuple[list[int], list[dict[str, object]]]:
        cached = threshold_cache.get(threshold)
        if cached is not None:
            return cached
        selected = [
            row
            for row in eligible
            if float(row["exit_basis_bp"]) <= threshold
        ]
        result = (
            [int(row["exit_decision_time_ns"]) for row in selected],
            selected,
        )
        threshold_cache[threshold] = result
        return result

    records: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        full = row["full_fill"] is True
        hedge_ok = row["entry_hedge_status"] == "executable"
        threshold = float(row["exit_threshold_basis_bp"])
        start_ns = _optional_max(
            row.get("full_fill_recv_time_ns"),
            row.get("entry_hedge_decision_time_ns"),
        )
        hit: dict[str, object] | None = None
        eod: dict[str, object] | None = None
        if full and hedge_ok and start_ns is not None:
            eod_index = bisect_left(eligible_times, start_ns)
            if eod_index < len(eligible):
                eod = eligible[-1]
            hit_times, hit_rows = candidates(threshold)
            hit_index = bisect_left(hit_times, start_ns)
            if hit_index < len(hit_rows):
                hit = hit_rows[hit_index]

        if not full and row.get("partial_fill") is True:
            branch = "partial_entry_unhedged"
        elif not full and row.get("any_fill") is None:
            branch = "entry_fill_unknown"
        elif not full:
            branch = "no_entry_fill"
        elif not hedge_ok:
            branch = "entry_hedge_unpriceable"
        elif hit is not None:
            branch = "same_day_taker_exit"
        else:
            branch = "carry_at_eod"

        entry_future = _finite_or_none(row.get("entry_future_price"))
        entry_spot = _finite_or_none(row.get("entry_spot_price"))
        contract_size = _finite_or_none(
            row.get("entry_hedge_contract_size_shares")
        )
        gross = _gross_cycle_pnl(entry_future, entry_spot, hit, contract_size)
        eod_gross = _gross_cycle_pnl(entry_future, entry_spot, eod, contract_size)
        chosen = hit if hit is not None else eod
        records.append(
            {
                "Date": str(row["Date"]),
                "ValueCode": str(row["ValueCode"]),
                "QuoteCode": str(row["QuoteCode"]),
                "route": str(row["route"]),
                "raw_order_fact_id": str(row["raw_order_fact_id"]),
                "policy_generation_id": str(row["policy_generation_id"]),
                "exit_rule_id": str(row["exit_rule_id"]),
                "exit_threshold_basis_bp": threshold,
                "exit_rule_source_asof_date": str(
                    row["_exit_rule_source_asof_date"]
                ),
                "branch_status": branch,
                "same_day_exit": branch == "same_day_taker_exit",
                "overnight_carry": branch == "carry_at_eod",
                "terminal_outcome": branch
                in {"same_day_taker_exit", "no_entry_fill"},
                "needs_next_session_label": branch == "carry_at_eod",
                "exit_decision_time_ns": _row_int(hit, "exit_decision_time_ns"),
                "holding_time_ms": (
                    (int(hit["exit_decision_time_ns"]) - start_ns) / 1_000_000.0
                    if hit is not None and start_ns is not None
                    else None
                ),
                "exit_basis_bp": _row_float(hit, "exit_basis_bp"),
                "spot_exit_vwap_price": _row_float(hit, "spot_sell_vwap_price"),
                "future_exit_vwap_price": _row_float(hit, "future_buy_vwap_price"),
                "spot_exit_snapshot_recv_time_ns": _row_int(
                    hit, "spot_snapshot_recv_time_ns"
                ),
                "spot_exit_snapshot_sequence": _row_int(
                    hit, "spot_snapshot_sequence"
                ),
                "future_exit_snapshot_recv_time_ns": _row_int(
                    hit, "future_snapshot_recv_time_ns"
                ),
                "future_exit_snapshot_sequence": _row_int(
                    hit, "future_snapshot_sequence"
                ),
                "gross_cycle_pnl_twd": gross,
                "eod_mark_available": eod is not None,
                "eod_mark_time_ns": _row_int(eod, "exit_decision_time_ns"),
                "eod_liquidation_basis_bp": _row_float(eod, "exit_basis_bp"),
                "eod_liquidation_gross_pnl_twd": eod_gross,
                "selected_or_eod_spot_book_time_ns": _row_int(
                    chosen, "spot_snapshot_recv_time_ns"
                ),
                "selected_or_eod_future_book_time_ns": _row_int(
                    chosen, "future_snapshot_recv_time_ns"
                ),
                "exit_style": "immediate_taker_taker_on_causal_grid",
                "maker_exit_fill_included": False,
                "exit_added_latency_ns": 0,
                "joint_volume_allocated": False,
                "fees_tax_included": False,
                "overnight_cost_included": False,
                "overnight_realized_pnl_included": False,
                "net_cycle_pnl_twd": None,
                "pathwise_ev_ready": False,
                "contains_target_day_outcome": True,
            }
        )
    return (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame(schema=_empty_exit_fact_schema())
    )


def summarize_executable_exit_daily(
    action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
) -> pl.DataFrame:
    """Count mutually exclusive exit branches by action and exit rule."""

    if exit_facts.is_empty():
        return pl.DataFrame(schema=_empty_exit_daily_schema())
    _require(
        action_facts,
        {"policy_generation_id", *_ACTION_KEY},
        "execution action facts",
    )
    _require(
        exit_facts,
        {
            "policy_generation_id",
            "exit_rule_id",
            "branch_status",
            "same_day_exit",
            "overnight_carry",
            "gross_cycle_pnl_twd",
            "eod_liquidation_gross_pnl_twd",
        },
        "executable exit facts",
    )
    metadata = action_facts.select("policy_generation_id", *_ACTION_KEY)
    if metadata.select("policy_generation_id").n_unique() != metadata.height:
        raise ValueError("action facts contain duplicate policy generations")
    joined = exit_facts.join(
        metadata,
        on="policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined.filter(pl.col("Date").is_null()).height:
        raise ValueError("exit fact does not resolve to an execution action")
    group = [*_ACTION_KEY, "exit_rule_id"]
    return joined.group_by(group).agg(
        pl.len().alias("exit_rule_aliases"),
        pl.col("same_day_exit").sum().alias("same_day_taker_exits"),
        pl.col("overnight_carry").sum().alias("overnight_carry_branches"),
        (pl.col("branch_status") == "no_entry_fill")
        .sum()
        .alias("no_entry_fill_branches"),
        (pl.col("branch_status") == "partial_entry_unhedged")
        .sum()
        .alias("partial_entry_unhedged_branches"),
        (pl.col("branch_status") == "entry_fill_unknown")
        .sum()
        .alias("entry_fill_unknown_branches"),
        (pl.col("branch_status") == "entry_hedge_unpriceable")
        .sum()
        .alias("entry_hedge_unpriceable_branches"),
        pl.col("gross_cycle_pnl_twd")
        .drop_nulls()
        .median()
        .alias("same_day_gross_cycle_pnl_twd_p50"),
        pl.col("gross_cycle_pnl_twd")
        .drop_nulls()
        .sum()
        .alias("same_day_gross_cycle_pnl_twd_sum"),
        pl.col("eod_liquidation_gross_pnl_twd")
        .filter(pl.col("overnight_carry"))
        .median()
        .alias("carry_eod_liquidation_gross_pnl_twd_p50"),
    ).with_columns(
        _safe_rate(
            "same_day_taker_exits",
            "exit_rule_aliases",
            "p_same_day_taker_exit_per_alias",
        ),
        _safe_rate(
            "overnight_carry_branches",
            "exit_rule_aliases",
            "p_overnight_carry_per_alias",
        ),
        pl.lit(False).alias("fees_tax_included"),
        pl.lit(False).alias("overnight_cost_included"),
        pl.lit(False).alias("pathwise_ev_ready"),
    ).sort(group)


def _validate_raw_alias_identity(aliases: pl.DataFrame) -> None:
    identity_columns = (
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "spread_pair_epoch",
        "target_price_tick",
        "submit_recv_time_ns",
        "submit_event_sequence",
        "submit_row_index",
    )
    inconsistent = aliases.group_by("raw_order_fact_id").agg(
        *[
            pl.col(column).n_unique().alias(column)
            for column in identity_columns
        ]
    ).filter(
        pl.any_horizontal(*[pl.col(column) != 1 for column in identity_columns])
    )
    if inconsistent.height:
        raise ValueError("raw_order_fact_id aliases disagree on physical identity")


def _validate_fill_cursors(aliases: pl.DataFrame) -> None:
    full = aliases.filter(pl.col("full_fill") == True)  # noqa: E712
    if full.filter(
        pl.any_horizontal(
            pl.col("full_fill_recv_time_ns").is_null(),
            pl.col("full_fill_event_sequence").is_null(),
            pl.col("full_fill_row_index").is_null(),
        )
    ).height:
        raise ValueError("every full fill requires an exact three-part cursor")
    partial = aliases.filter(pl.col("partial_fill"))
    if partial.filter(
        pl.any_horizontal(
            pl.col("first_fill_recv_time_ns").is_null(),
            pl.col("first_fill_event_sequence").is_null(),
            pl.col("first_fill_row_index").is_null(),
        )
    ).height:
        raise ValueError("every partial fill requires an exact first-fill cursor")


def _exit_book_status(
    row: dict[str, object] | None,
    market: str,
    query_ns: int,
    max_book_age_ns: int | None,
) -> str:
    if row is None:
        return "no_state"
    if bool(row.get("trial_match", False)):
        return "trial_match"
    if row.get("_formal_after_trial") is not True:
        return "awaiting_formal_book"
    if not bool(row.get("book_state_available", False)):
        return "no_book"
    book_ns = row.get("book_recv_time_ns")
    if book_ns is None:
        return "no_book_time"
    if max_book_age_ns is not None and query_ns - int(book_ns) > max_book_age_ns:
        return "stale_book"
    try:
        bids = executable_levels_from_state(row, market, "bid")  # type: ignore[arg-type]
        asks = executable_levels_from_state(row, market, "ask")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "invalid_book"
    if not bids or not asks or bids[0].price > asks[0].price:
        return "invalid_book"
    reference = _finite_or_none(row.get("ref_price"))
    if reference is None or reference <= 0:
        return "missing_ref"
    lower, upper = reference * 0.91, reference * 1.08
    if not (
        lower < bids[0].price < upper and lower < asks[0].price < upper
    ):
        return "ref_gate"
    return "ok"


def _sweep_vwap(levels: Iterable[object], quantity: int) -> tuple[float | None, int]:
    levels = tuple(levels)
    available = sum(int(getattr(level, "quantity")) for level in levels)
    remaining = quantity
    notional = 0.0
    executed = 0
    for level in levels:
        price = float(getattr(level, "price"))
        level_quantity = int(getattr(level, "quantity"))
        take = min(remaining, level_quantity)
        notional += price * take
        executed += take
        remaining -= take
        if remaining == 0:
            break
    return ((notional / executed) if remaining == 0 and executed else None, available)


def _gross_cycle_pnl(
    entry_future: float | None,
    entry_spot: float | None,
    exit_row: dict[str, object] | None,
    contract_size: float | None,
) -> float | None:
    if (
        entry_future is None
        or entry_spot is None
        or exit_row is None
        or contract_size is None
    ):
        return None
    spot_exit = _finite_or_none(exit_row.get("spot_sell_vwap_price"))
    future_exit = _finite_or_none(exit_row.get("future_buy_vwap_price"))
    if spot_exit is None or future_exit is None:
        return None
    return contract_size * (
        entry_future - entry_spot + spot_exit - future_exit
    )


def _safe_rate(numerator: str, denominator: str, output: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(output)
    )


def _with_optional_columns(
    frame: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    expressions = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    return frame.with_columns(*expressions) if expressions else frame


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _ceil_multiple(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def _book_age_ms(
    row: dict[str, object] | None, query_ns: int
) -> float | None:
    if row is None or row.get("book_recv_time_ns") is None:
        return None
    return (query_ns - int(row["book_recv_time_ns"])) / 1_000_000.0


def _row_int(row: dict[str, object] | None, column: str) -> int | None:
    if row is None or row.get(column) is None:
        return None
    return int(row[column])


def _row_float(row: dict[str, object] | None, column: str) -> float | None:
    return _finite_or_none(None if row is None else row.get(column))


def _finite_or_none(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else None
    )


def _optional_max(*values: object) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _empty_action_schema() -> Mapping[str, pl.DataType]:
    return {
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


def _empty_daily_schema() -> Mapping[str, pl.DataType]:
    return {
        **{column: pl.String for column in _ACTION_KEY if column != "boundary_quantile"},
        "boundary_quantile": pl.Int64,
        "policy_alias_orders": pl.Int64,
        "unique_raw_order_facts": pl.Int64,
        "cancel_required_orders": pl.Int64,
        "cancel_required_rate": pl.Float64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_exit_fact_schema() -> Mapping[str, pl.DataType]:
    return {
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
        "net_cycle_pnl_twd": pl.Float64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_exit_daily_schema() -> Mapping[str, pl.DataType]:
    return {
        **_empty_daily_schema(),
        "exit_rule_id": pl.String,
        "exit_rule_aliases": pl.Int64,
        "same_day_taker_exits": pl.Int64,
        "overnight_carry_branches": pl.Int64,
    }
