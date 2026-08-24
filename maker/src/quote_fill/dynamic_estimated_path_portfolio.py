"""Analysis-only dynamic-universe estimated paths and capacity replay.

This module intentionally does not read the historical fixed-symbol execution
root.  Its entry population is supplied by a causal daily manifest makerFill
study and is kept intact even when a product leaves the following day's entry
universe.  A filled long-spot/short-futures position is searched against the
one-second ``causal_fair`` grid until the first executable frozen-lower close.

The resulting paths are estimates, not formal execution facts:

* entry fill time/outcome may come from the end-of-day-looking makerFill cache;
* the +50 ms futures hedge must be supplied separately and is never replaced
  with a one-second price;
* exits have one-second decision resolution and zero extra latency; and
* an expiry close can be an official paired-close accounting mark, which is
  explicitly non-executable.

Resolved, fully priced rows are sent to ``portfolio_cap_backtester`` only as a
clearly labelled completed-only diagnostic.  The primary inventory-aware cap
replay admits the full fill population: unresolved rows continue consuming
capacity through the observation horizon and are never imputed to zero PnL.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import shutil
import tempfile
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import (
    DEFAULT_DAILY_ROOT,
    completed_artifact_paths,
    discover_common_sessions,
)
from .combined_cost_cap_sweep import TransactionCostProfile
from .portfolio_cap_backtester import (
    PortfolioCapBacktestConfig,
    PortfolioCapBacktestResult,
    backtest_priced_paths,
    default_cap_scenarios,
    local_session_timestamp_ns,
)

ANALYSIS_VERSION = "dynamic_causal_q95_frozen_lower_estimated_paths_v1"
DEFAULT_ENTRY_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "one_second_makerfill_causal_v2_20260822_v1"
)
DEFAULT_MANIFEST_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "monthly_product_selector_causal_v2_20260822"
    / "daily_entry_manifest.csv"
)
DEFAULT_BOUNDARY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "rolling_boundaries"
    / "rolling_boundary_snapshots.parquet"
)
DEFAULT_EXPIRY_CLOSE_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "dynamic_expiry_paired_close_facts_20260822_v1"
    / "paired_close_facts.parquet"
)
DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
ONE_SECOND_NS = 1_000_000_000
SPOT_LOT_SHARES = 1_000
SUPPORTED_ENTRY_ROUTE = "spot_bid_future_taker"


@dataclass(frozen=True)
class DynamicEstimatedPathConfig:
    """Frozen analysis contract for the dynamic q95 path estimate."""

    boundary_quantile: int = 95
    entry_route: str = SUPPORTED_ENTRY_ROUTE
    eligibility_column: str = "analysis_eligible"
    hedge_delay_ns: int = 50_000_000
    future_contract_quantity: int = 1
    first_exit_on_next_full_second: bool = True
    expiry_grid_fallback: bool = True
    expiry_paired_accounting_mark_fallback: bool = True
    timezone_name: str = "Asia/Taipei"
    cost_profile: TransactionCostProfile = field(
        default_factory=TransactionCostProfile
    )
    analysis_version: str = ANALYSIS_VERSION

    def validate(self) -> None:
        if self.boundary_quantile != 95:
            raise ValueError("dynamic estimated-path v1 is frozen to q95")
        if self.entry_route != SUPPORTED_ENTRY_ROUTE:
            raise ValueError("v1 supports the sell-spread spot-maker route only")
        if not self.eligibility_column:
            raise ValueError("eligibility_column must be nonempty")
        if self.hedge_delay_ns != 50_000_000:
            raise ValueError("v1 requires the user-specified +50 ms entry hedge")
        if self.future_contract_quantity <= 0:
            raise ValueError("future_contract_quantity must be positive")
        try:
            ZoneInfo(self.timezone_name)
        except (KeyError, ValueError) as error:
            raise ValueError("timezone_name is invalid") from error
        self.cost_profile.validate()
        if self.analysis_version != ANALYSIS_VERSION:
            raise ValueError(f"analysis_version must be {ANALYSIS_VERSION!r}")


DEFAULT_PATH_CONFIG = DynamicEstimatedPathConfig()
DEFAULT_COST_PROFILE = TransactionCostProfile()


@dataclass(frozen=True)
class DynamicEstimatedPathResult:
    entry_positions: pl.DataFrame
    terminal_paths: pl.DataFrame
    priced_paths: pl.DataFrame
    unresolved_paths: pl.DataFrame
    coverage: pl.DataFrame
    full_population_cap: FullPopulationCapResult | None
    completed_only_cap: PortfolioCapBacktestResult | None


@dataclass(frozen=True)
class FullPopulationCapResult:
    """Inventory-aware replay retaining unresolved positions through horizon."""

    events: pl.DataFrame
    daily: pl.DataFrame
    summary: pl.DataFrame


_ENTRY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "physical_order_id": ("physical_order_id", "raw_order_fact_id"),
    "q": ("q", "boundary_quantile"),
    "target_price": ("target_price", "entry_spot_price"),
    "submit_ns": (
        "submit_ns",
        "submit_decision_time_ns",
        "submit_recv_time_ns",
    ),
    "fill_ns": (
        "fill_ns",
        "makerfill_implied_fill_time_ns",
        "full_fill_recv_time_ns",
    ),
    "full_fill": ("full_fill", "makerfill_full_fill"),
    "anchor_ewma_120s_bp": (
        "anchor_ewma_120s_bp",
        "entry_anchor_ewma_120s_bp",
    ),
    "lower_distance_bp": ("lower_distance_bp",),
    "upper_distance_bp": ("upper_distance_bp",),
    "contract_size": ("contract_size", "entry_contract_size_shares"),
    "end_date": ("end_date", "expiry_date", "expiry_session"),
}
_ENTRY_REQUIRED_DIRECT = {"Date", "ValueCode", "QuoteCode"}
_HEDGE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "physical_order_id": ("physical_order_id", "raw_order_fact_id"),
    "entry_hedge_decision_time_ns": (
        "entry_hedge_decision_time_ns",
        "hedge_decision_time_ns",
        "decision_time_ns",
    ),
    "entry_future_price": (
        "entry_future_price",
        "entry_hedge_executable_vwap_price",
        "executable_vwap_price",
    ),
    "entry_hedge_executable": (
        "entry_hedge_executable",
        "hedge_executable",
    ),
    "entry_hedge_depth_shortfall": (
        "entry_hedge_depth_shortfall",
        "hedge_depth_shortfall",
        "depth_shortfall",
    ),
}
_HEDGE_OPTIONAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "entry_hedge_status": ("entry_hedge_status", "status"),
    "entry_hedge_price_approximate": ("entry_hedge_price_approximate",),
    "entry_hedge_source": ("entry_hedge_source", "hedge_version"),
    "entry_hedge_delay_ns": ("entry_hedge_delay_ns",),
    "entry_hedge_signed_latency_slippage_bp": (
        "entry_hedge_signed_latency_slippage_bp",
        "signed_latency_slippage_bp",
    ),
    "entry_hedge_signed_depth_slippage_bp": (
        "entry_hedge_signed_depth_slippage_bp",
        "signed_depth_slippage_bp",
    ),
    "entry_hedge_signed_total_slippage_bp": (
        "entry_hedge_signed_total_slippage_bp",
        "signed_total_slippage_bp",
    ),
    "entry_hedge_decision_book_age_ms": (
        "entry_hedge_decision_book_age_ms",
        "decision_book_age_ms",
    ),
}


def build_dynamic_entry_positions(
    entry_outcomes: pl.DataFrame,
    *,
    hedge_facts: pl.DataFrame | None = None,
    manifest: pl.DataFrame | None = None,
    config: DynamicEstimatedPathConfig = DEFAULT_PATH_CONFIG,
) -> pl.DataFrame:
    """Adapt causal makerFill outcomes into one row per physical position.

    ``fut_exec_bid`` from the candidate snapshot is deliberately not accepted
    as an entry hedge price.  A +50 ms hedge fact must either already be in the
    entry frame or be supplied through ``hedge_facts``.
    """

    config.validate()
    if not isinstance(entry_outcomes, pl.DataFrame):
        raise TypeError("entry_outcomes must be a polars DataFrame")
    if entry_outcomes.is_empty():
        return _empty_entry_positions()
    missing_direct = sorted(_ENTRY_REQUIRED_DIRECT - set(entry_outcomes.columns))
    if missing_direct:
        raise ValueError(f"entry outcomes missing columns: {missing_direct}")
    source = _canonicalize_aliases(entry_outcomes, _ENTRY_ALIASES, "entry outcomes")

    if hedge_facts is not None:
        hedge = hedge_facts
        if "entry_hedge_executable" not in hedge.columns and "status" in hedge.columns:
            hedge = hedge.with_columns(
                (pl.col("status") == "executable").alias("entry_hedge_executable")
            )
        if (
            "entry_hedge_price_approximate" not in hedge.columns
            and "entry_fill_time_exact" in hedge.columns
        ):
            hedge = hedge.with_columns(
                ~pl.col("entry_fill_time_exact").fill_null(False).alias(
                    "entry_hedge_price_approximate"
                )
            )
        hedge = _canonicalize_aliases(hedge, _HEDGE_ALIASES, "hedge facts")
        hedge = _canonicalize_optional_aliases(hedge, _HEDGE_OPTIONAL_ALIASES)
        optional_hedge = [
            column for column in _HEDGE_OPTIONAL_ALIASES if column in hedge.columns
        ]
        hedge = hedge.select(*_HEDGE_ALIASES, *optional_hedge)
        if hedge["physical_order_id"].n_unique() != hedge.height:
            raise ValueError("hedge facts must be unique by physical_order_id")
        collisions = sorted(
            (set(_HEDGE_ALIASES) - {"physical_order_id"}) & set(source.columns)
        )
        if collisions:
            source = source.drop(collisions)
        source = source.join(hedge, on="physical_order_id", how="left", validate="m:1")
    else:
        source = _canonicalize_aliases(source, _HEDGE_ALIASES, "entry hedge fields")

    source = source.filter(
        (pl.col("q").cast(pl.Int64) == config.boundary_quantile)
        & pl.col("full_fill").fill_null(False)
    )
    if "outcome_supported" in source.columns:
        source = source.filter(pl.col("outcome_supported").fill_null(False))
    if source.is_empty():
        return _empty_entry_positions()
    if source["physical_order_id"].n_unique() != source.height:
        raise ValueError("filled entry outcomes duplicate physical_order_id")

    normalized = source.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("physical_order_id").cast(pl.String),
        pl.col("q").cast(pl.Int64),
        pl.col("target_price").cast(pl.Float64),
        pl.col("submit_ns").cast(pl.Int64),
        pl.col("fill_ns").cast(pl.Int64),
        pl.col("anchor_ewma_120s_bp").cast(pl.Float64),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("lower_distance_bp").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Int64),
        pl.col("entry_hedge_decision_time_ns").cast(pl.Int64),
        pl.col("entry_future_price").cast(pl.Float64),
        pl.col("entry_hedge_executable").cast(pl.Boolean),
        pl.col("entry_hedge_depth_shortfall").cast(pl.Int64),
        _expiry_string_expr("end_date").alias("expiry_session"),
    )
    if manifest is not None:
        _validate_entry_manifest_membership(normalized, manifest)

    if "entry_outcome_approximate" not in normalized.columns:
        normalized = normalized.with_columns(
            (
                ~pl.col("fill_outcome_exact_within_model").fill_null(False)
                if "fill_outcome_exact_within_model" in normalized.columns
                else pl.lit(True)
            ).alias("entry_outcome_approximate")
        )
    if "entry_fill_time_approximate" not in normalized.columns:
        normalized = normalized.with_columns(
            (
                ~pl.col("fill_cursor_exact").fill_null(False)
                if "fill_cursor_exact" in normalized.columns
                else pl.lit(True)
            ).alias("entry_fill_time_approximate")
        )
    if "entry_hedge_price_approximate" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit(True).alias("entry_hedge_price_approximate")
        )
    if "entry_fill_source" not in normalized.columns:
        normalized = normalized.with_columns(
            (
                pl.col("fill_backend").cast(pl.String)
                if "fill_backend" in normalized.columns
                else pl.lit("makerfill_cached_displayed_queue_estimate")
            ).alias("entry_fill_source")
        )
    if "entry_hedge_source" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit("supplied_plus_50ms_hedge_fact").alias("entry_hedge_source")
        )
    if "entry_hedge_delay_ns" not in normalized.columns:
        normalized = normalized.with_columns(
            (pl.col("entry_hedge_decision_time_ns") - pl.col("fill_ns")).alias(
                "entry_hedge_delay_ns"
            )
        )
    if "nominal_stop_is_cutoff" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit(False).alias("nominal_stop_is_cutoff")
        )
    if "nominal_stop_time_ns" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("nominal_stop_time_ns")
        )

    finite_required = normalized.filter(
        ~pl.all_horizontal(
            pl.col(
                "target_price",
                "anchor_ewma_120s_bp",
                "upper_distance_bp",
                "lower_distance_bp",
            ).is_finite()
        )
        | (pl.col("target_price") <= 0)
        | (pl.col("contract_size") <= 0)
        | (pl.col("fill_ns") < pl.col("submit_ns"))
        | (pl.col("lower_distance_bp") < 0)
        | (pl.col("upper_distance_bp") < 0)
    )
    if finite_required.height:
        raise ValueError("filled entry outcomes contain invalid prices/times/boundaries")

    hedge_supported = (
        pl.col("entry_hedge_executable").fill_null(False)
        & (pl.col("entry_hedge_depth_shortfall").fill_null(1) == 0)
        & pl.col("entry_future_price").is_finite()
        & (pl.col("entry_future_price") > 0)
        & pl.col("entry_hedge_decision_time_ns").is_not_null()
        & (
            pl.col("entry_hedge_decision_time_ns")
            == pl.col("fill_ns") + config.hedge_delay_ns
        )
    )
    result = normalized.with_columns(
        pl.lit(config.entry_route).alias("entry_route"),
        pl.col("target_price").alias("entry_spot_price"),
        pl.col("contract_size").alias("entry_contract_size_shares"),
        pl.lit(config.future_contract_quantity).alias(
            "entry_future_contract_quantity"
        ),
        (
            pl.col("anchor_ewma_120s_bp") - pl.col("lower_distance_bp")
        ).alias("exit_threshold_basis_bp"),
        hedge_supported.alias("entry_pricing_supported"),
        pl.when(hedge_supported)
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col("entry_future_price").is_null())
        .then(pl.lit("missing_plus_50ms_hedge_price"))
        .when(~pl.col("entry_hedge_executable").fill_null(False))
        .then(pl.lit("plus_50ms_hedge_not_executable"))
        .when(pl.col("entry_hedge_depth_shortfall").fill_null(1) != 0)
        .then(pl.lit("plus_50ms_hedge_depth_shortfall"))
        .otherwise(pl.lit("plus_50ms_hedge_timing_mismatch"))
        .alias("entry_pricing_unsupported_reason"),
        pl.lit(True).alias("entry_selected_by_causal_dynamic_manifest"),
        pl.lit(False).alias("fixed45_universe_used"),
        (
            pl.col("nominal_stop_is_cutoff").fill_null(False)
            & (pl.col("fill_ns") == pl.col("nominal_stop_time_ns"))
        ).alias("entry_fill_cancel_race_at_1300_cutoff"),
        pl.lit(config.analysis_version).alias("analysis_version"),
    ).with_columns(
        pl.when(pl.col("entry_pricing_supported"))
        .then(pl.col("entry_hedge_decision_time_ns"))
        .otherwise(pl.col("fill_ns") + config.hedge_delay_ns)
        .alias("position_established_ns"),
        (
            pl.col("entry_spot_price") * pl.col("entry_contract_size_shares")
        ).alias("normalization_notional_twd"),
    )
    columns = [
        "physical_order_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "q",
        "submit_ns",
        "fill_ns",
        "position_established_ns",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        "entry_future_contract_quantity",
        "normalization_notional_twd",
        "anchor_ewma_120s_bp",
        "upper_distance_bp",
        "lower_distance_bp",
        "exit_threshold_basis_bp",
        "expiry_session",
        "entry_pricing_supported",
        "entry_pricing_unsupported_reason",
        "entry_outcome_approximate",
        "entry_fill_time_approximate",
        "entry_hedge_price_approximate",
        "entry_fill_source",
        "entry_hedge_source",
        "entry_hedge_delay_ns",
        "entry_hedge_status",
        "entry_hedge_signed_latency_slippage_bp",
        "entry_hedge_signed_depth_slippage_bp",
        "entry_hedge_signed_total_slippage_bp",
        "entry_hedge_decision_book_age_ms",
        "entry_selected_by_causal_dynamic_manifest",
        "fixed45_universe_used",
        "entry_fill_cancel_race_at_1300_cutoff",
        "analysis_version",
    ]
    for column in (
        "entry_hedge_status",
        "entry_hedge_signed_latency_slippage_bp",
        "entry_hedge_signed_depth_slippage_bp",
        "entry_hedge_signed_total_slippage_bp",
        "entry_hedge_decision_book_age_ms",
    ):
        if column not in result.columns:
            dtype = pl.String if column == "entry_hedge_status" else pl.Float64
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return result.select(*columns).sort(
        ["Date", "position_established_ns", "ValueCode", "physical_order_id"]
    )


class _MinTree:
    """Small segment tree supporting first value <= threshold after index."""

    def __init__(self, values: Sequence[float]) -> None:
        size = 1
        while size < len(values):
            size *= 2
        self.size = size
        self.length = len(values)
        self.tree = [math.inf] * (2 * size)
        self.tree[size : size + len(values)] = values
        for index in range(size - 1, 0, -1):
            self.tree[index] = min(self.tree[2 * index], self.tree[2 * index + 1])

    def first_le(self, start: int, threshold: float) -> int | None:
        if start >= self.length or self.tree[1] > threshold:
            return None
        return self._first_le(1, 0, self.size, max(0, start), threshold)

    def _first_le(
        self,
        node: int,
        left: int,
        right: int,
        start: int,
        threshold: float,
    ) -> int | None:
        if right <= start or self.tree[node] > threshold:
            return None
        if right - left == 1:
            return left if left < self.length else None
        middle = (left + right) // 2
        first = self._first_le(node * 2, left, middle, start, threshold)
        if first is not None:
            return first
        return self._first_le(node * 2 + 1, middle, right, start, threshold)


@dataclass(frozen=True)
class _FairSeries:
    timestamps_ns: tuple[int, ...]
    basis_bp: tuple[float, ...]
    spot_prices: tuple[float, ...]
    future_prices: tuple[float, ...]
    minimums: _MinTree

    def first_hit(self, earliest_ns: int, threshold: float) -> int | None:
        start = bisect_left(self.timestamps_ns, earliest_ns)
        return self.minimums.first_le(start, threshold)

    def last_index(self) -> int | None:
        return len(self.timestamps_ns) - 1 if self.timestamps_ns else None


def label_dynamic_frozen_lower_paths(
    entry_positions: pl.DataFrame,
    causal_fair: pl.DataFrame,
    sessions: Sequence[str] | Iterable[str],
    *,
    expiry_close_facts: pl.DataFrame | None = None,
    config: DynamicEstimatedPathConfig = DEFAULT_PATH_CONFIG,
) -> pl.DataFrame:
    """Find the first q95 frozen-lower taker/taker close across sessions."""

    config.validate()
    if entry_positions.is_empty():
        return _empty_terminal_paths()
    required_positions = {
        "physical_order_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "position_established_ns",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        "entry_future_contract_quantity",
        "normalization_notional_twd",
        "exit_threshold_basis_bp",
        "expiry_session",
        "entry_pricing_supported",
    }
    _require_columns(entry_positions, required_positions, "entry positions")
    required_fair = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "spot_bid",
        "spot_bid_lots",
        "fut_exec_ask",
        "fut_exec_ask_lots",
        "basis_buy_taker_bp",
        "contract_size",
        config.eligibility_column,
    }
    _require_columns(causal_fair, required_fair, "causal fair")
    calendar = _normalize_sessions(sessions)
    calendar_index = {value: index for index, value in enumerate(calendar)}
    unknown_entry_dates = sorted(set(entry_positions["Date"].to_list()) - set(calendar))
    if unknown_entry_dates:
        raise ValueError(
            f"entry dates absent from session calendar: {unknown_entry_dates[:5]}"
        )

    series = _build_fair_series(causal_fair, config)
    close_lookup = _expiry_close_lookup(expiry_close_facts)
    last_observed_date = max((key[0] for key in series), default=calendar[-1])
    rows: list[dict[str, object]] = []
    for position in entry_positions.iter_rows(named=True):
        if not bool(position["entry_pricing_supported"]):
            rows.append(
                _unresolved_record(
                    position,
                    "entry_hedge_unpriced",
                    str(position.get("entry_pricing_unsupported_reason") or "unknown"),
                    last_observed_date,
                    config,
                )
            )
            continue
        rows.append(
            _find_terminal(
                position,
                series,
                calendar,
                calendar_index,
                close_lookup,
                last_observed_date,
                config,
            )
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )


def _build_fair_series(
    causal_fair: pl.DataFrame,
    config: DynamicEstimatedPathConfig,
) -> dict[tuple[str, str, str, int, int], _FairSeries]:
    frame = causal_fair.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("timestamp").dt.epoch("ns").alias("_timestamp_ns"),
    )
    records: dict[tuple[str, str, str, int, int], _FairSeries] = {}
    for (day, value_code, quote_code), group in frame.group_by(
        "Date", "ValueCode", "QuoteCode"
    ):
        contract_sizes = group.get_column("contract_size").drop_nulls().unique()
        if contract_sizes.len() != 1:
            continue
        shares = round(float(contract_sizes.item()))
        executable = group.filter(
            pl.col(config.eligibility_column).fill_null(False)
            & pl.col("spot_bid").is_finite()
            & (pl.col("spot_bid") > 0)
            & pl.col("fut_exec_ask").is_finite()
            & (pl.col("fut_exec_ask") > 0)
            & pl.col("basis_buy_taker_bp").is_finite()
            & (pl.col("spot_bid_lots") * SPOT_LOT_SHARES >= shares)
            & (pl.col("fut_exec_ask_lots") >= config.future_contract_quantity)
        ).sort("_timestamp_ns")
        if executable.is_empty():
            continue
        timestamps = tuple(int(value) for value in executable["_timestamp_ns"])
        basis = tuple(float(value) for value in executable["basis_buy_taker_bp"])
        key = (
            str(day),
            str(value_code),
            str(quote_code),
            shares,
            config.future_contract_quantity,
        )
        records[key] = _FairSeries(
            timestamps_ns=timestamps,
            basis_bp=basis,
            spot_prices=tuple(float(value) for value in executable["spot_bid"]),
            future_prices=tuple(
                float(value) for value in executable["fut_exec_ask"]
            ),
            minimums=_MinTree(basis),
        )
    return records


def _find_terminal(
    position: Mapping[str, object],
    series: Mapping[tuple[str, str, str, int, int], _FairSeries],
    calendar: tuple[str, ...],
    calendar_index: Mapping[str, int],
    close_lookup: Mapping[tuple[str, str, str], Mapping[str, object]],
    last_observed_date: str,
    config: DynamicEstimatedPathConfig,
) -> dict[str, object]:
    entry_date = str(position["Date"])
    expiry = str(position["expiry_session"])
    value_code = str(position["ValueCode"])
    quote_code = str(position["QuoteCode"])
    shares = int(position["entry_contract_size_shares"])
    contracts = int(position["entry_future_contract_quantity"])
    threshold = float(position["exit_threshold_basis_bp"])
    established = int(position["position_established_ns"])
    start_index = calendar_index[entry_date]
    search_dates = [value for value in calendar[start_index:] if value <= expiry]
    expiry_grid: tuple[_FairSeries, int] | None = None
    for day in search_dates:
        key = (day, value_code, quote_code, shares, contracts)
        path = series.get(key)
        if path is None:
            continue
        earliest = (
            _next_full_second_ns(established)
            if day == entry_date and config.first_exit_on_next_full_second
            else established if day == entry_date else 0
        )
        hit_index = path.first_hit(earliest, threshold)
        if hit_index is not None:
            return _terminal_record(
                position,
                terminal_date=day,
                exit_ns=path.timestamps_ns[hit_index],
                exit_spot=path.spot_prices[hit_index],
                exit_future=path.future_prices[hit_index],
                exit_basis=path.basis_bp[hit_index],
                status="same_day_frozen_lower_hit"
                if day == entry_date
                else "cross_session_frozen_lower_hit",
                executable_grid=True,
                expiry_grid_fallback=False,
                paired_expiry_mark_fallback=False,
                config=config,
            )
        if day == expiry:
            last_index = path.last_index()
            if last_index is not None:
                expiry_grid = (path, last_index)

    close = close_lookup.get((expiry, value_code, quote_code))
    if config.expiry_paired_accounting_mark_fallback and close is not None:
        return _paired_expiry_mark_terminal(position, close, config)

    if config.expiry_grid_fallback and expiry_grid is not None:
        path, index = expiry_grid
        return _terminal_record(
            position,
            terminal_date=expiry,
            exit_ns=path.timestamps_ns[index],
            exit_spot=path.spot_prices[index],
            exit_future=path.future_prices[index],
            exit_basis=path.basis_bp[index],
            status="expiry_last_joint_executable_grid_estimated_nonofficial",
            executable_grid=True,
            expiry_grid_fallback=True,
            paired_expiry_mark_fallback=False,
            config=config,
        )

    reason = (
        "expiry_not_yet_observed_and_no_paired_accounting_mark"
        if expiry > last_observed_date
        else "expiry_reached_without_joint_grid_or_paired_close_fact"
    )
    return _unresolved_record(
        position,
        "observation_horizon_open" if expiry > last_observed_date else "expiry_unpriced",
        reason,
        last_observed_date,
        config,
    )


def _terminal_record(
    position: Mapping[str, object],
    *,
    terminal_date: str,
    exit_ns: int,
    exit_spot: float,
    exit_future: float,
    exit_basis: float,
    status: str,
    executable_grid: bool,
    expiry_grid_fallback: bool,
    paired_expiry_mark_fallback: bool,
    config: DynamicEstimatedPathConfig,
    expiry_mark_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    shares = int(position["entry_contract_size_shares"])
    gross = shares * (
        (exit_spot - float(position["entry_spot_price"]))
        + (float(position["entry_future_price"]) - exit_future)
    )
    same_day = terminal_date == str(position["Date"])
    metadata = expiry_mark_metadata or {}
    spot_official = bool(
        metadata.get("spot_close_is_official_daily_close_field", False)
    )
    future_official_close = bool(
        metadata.get("future_close_is_official_daily_close", False)
    )
    future_official_settlement = bool(
        metadata.get("future_close_is_official_settlement", False)
    )
    future_last_trade_proxy = bool(
        metadata.get("future_close_is_last_trade_proxy", False)
    )
    paired_fully_official = bool(
        metadata.get(
            "paired_mark_is_fully_official_close",
            paired_expiry_mark_fallback and not future_last_trade_proxy,
        )
    )
    return {
        **dict(position),
        "policy_path_id": f"{position['physical_order_id']}/q95/frozen_lower/taker_taker",
        "exit_rule_id": "frozen_lower",
        "exit_route": "spot_sell_taker_future_buy_taker",
        "path_status": status,
        "terminal_date": terminal_date,
        "exit_decision_time_ns": int(exit_ns),
        "exit_spot_price": float(exit_spot),
        "exit_future_price": float(exit_future),
        "exit_basis_buy_taker_bp": float(exit_basis),
        "gross_cycle_pnl_twd": float(gross),
        "gross_cycle_pnl_bp": float(gross)
        / float(position["normalization_notional_twd"])
        * 10_000.0,
        "completed_same_day": same_day,
        "completed_overnight": not same_day,
        "terminal_cashflow_priced": True,
        "terminal_cashflow_point_identified_on_estimated_path": True,
        "unresolved_cashflow_imputed": False,
        "exit_zero_added_latency_assumption": executable_grid,
        "exit_one_second_grid_estimate": executable_grid,
        "exit_joint_book_executable_at_grid": executable_grid,
        "expiry_last_joint_grid_fallback": expiry_grid_fallback,
        "expiry_paired_accounting_mark_fallback": paired_expiry_mark_fallback,
        "expiry_spot_mark_is_official_close": spot_official,
        "expiry_future_mark_is_official_close": future_official_close,
        "expiry_future_mark_is_official_settlement": future_official_settlement,
        "expiry_future_mark_is_last_trade_proxy": future_last_trade_proxy,
        "expiry_paired_mark_is_fully_official_close": paired_fully_official,
        "expiry_mark_is_executable": (
            False if paired_expiry_mark_fallback else executable_grid
        ),
        "path_execution_exact": False,
        "joint_volume_allocated": False,
        "pathwise_ev_ready": False,
        "production_strategy_go": False,
        "coverage_horizon_last_causal_fair_date": None,
        "unresolved_reason": None,
        "analysis_version": config.analysis_version,
    }


def _unresolved_record(
    position: Mapping[str, object],
    status: str,
    reason: str,
    horizon_date: str,
    config: DynamicEstimatedPathConfig,
) -> dict[str, object]:
    return {
        **dict(position),
        "policy_path_id": f"{position['physical_order_id']}/q95/frozen_lower/taker_taker",
        "exit_rule_id": "frozen_lower",
        "exit_route": "spot_sell_taker_future_buy_taker",
        "path_status": status,
        "terminal_date": None,
        "exit_decision_time_ns": None,
        "exit_spot_price": None,
        "exit_future_price": None,
        "exit_basis_buy_taker_bp": None,
        "gross_cycle_pnl_twd": None,
        "gross_cycle_pnl_bp": None,
        "completed_same_day": False,
        "completed_overnight": False,
        "terminal_cashflow_priced": False,
        "terminal_cashflow_point_identified_on_estimated_path": False,
        "unresolved_cashflow_imputed": False,
        "exit_zero_added_latency_assumption": False,
        "exit_one_second_grid_estimate": False,
        "exit_joint_book_executable_at_grid": False,
        "expiry_last_joint_grid_fallback": False,
        "expiry_paired_accounting_mark_fallback": False,
        "expiry_spot_mark_is_official_close": False,
        "expiry_future_mark_is_official_close": False,
        "expiry_future_mark_is_official_settlement": False,
        "expiry_future_mark_is_last_trade_proxy": False,
        "expiry_paired_mark_is_fully_official_close": False,
        "expiry_mark_is_executable": False,
        "path_execution_exact": False,
        "joint_volume_allocated": False,
        "pathwise_ev_ready": False,
        "production_strategy_go": False,
        "coverage_horizon_last_causal_fair_date": horizon_date,
        "unresolved_reason": reason,
        "analysis_version": config.analysis_version,
    }


def price_dynamic_terminal_paths(
    terminal_paths: pl.DataFrame,
    profile: TransactionCostProfile = DEFAULT_COST_PROFILE,
) -> pl.DataFrame:
    """Apply the user's per-leg Taiwan spot/futures transaction costs."""

    profile.validate()
    if terminal_paths.is_empty():
        return terminal_paths
    _require_columns(
        terminal_paths,
        {
            "terminal_cashflow_priced",
            "completed_same_day",
            "entry_contract_size_shares",
            "normalization_notional_twd",
            "entry_spot_price",
            "entry_future_price",
            "exit_spot_price",
            "exit_future_price",
            "gross_cycle_pnl_twd",
        },
        "terminal paths",
    )
    completed = pl.col("terminal_cashflow_priced").fill_null(False)
    shares = pl.col("entry_contract_size_shares").cast(pl.Float64)
    spot_commission_rate = profile.spot_commission_bp_per_side / 10_000.0
    future_tax_rate = profile.futures_tax_bp_per_side / 10_000.0
    spot_tax_rate = pl.when(pl.col("completed_same_day")).then(
        profile.spot_sell_tax_bp
        * profile.same_day_spot_sell_tax_multiplier
        / 10_000.0
    ).otherwise(profile.spot_sell_tax_bp / 10_000.0)
    result = terminal_paths.with_columns(
        pl.when(completed)
        .then(shares * pl.col("entry_spot_price") * spot_commission_rate)
        .otherwise(None)
        .alias("spot_buy_commission_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_spot_price") * spot_commission_rate)
        .otherwise(None)
        .alias("spot_sell_commission_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_spot_price") * spot_tax_rate)
        .otherwise(None)
        .alias("spot_sell_tax_twd"),
        pl.when(completed)
        .then(shares * pl.col("entry_future_price") * future_tax_rate)
        .otherwise(None)
        .alias("futures_entry_tax_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_future_price") * future_tax_rate)
        .otherwise(None)
        .alias("futures_exit_tax_twd"),
        pl.when(completed)
        .then(profile.futures_commission_twd_per_side)
        .otherwise(None)
        .alias("futures_entry_commission_twd"),
        pl.when(completed)
        .then(profile.futures_commission_twd_per_side)
        .otherwise(None)
        .alias("futures_exit_commission_twd"),
        pl.lit(profile.profile_id).alias("transaction_cost_profile_id"),
        pl.lit(True).alias("user_fee_tax_profile_applied_per_leg"),
    ).with_columns(
        pl.when(completed)
        .then(
            pl.sum_horizontal(
                "spot_buy_commission_twd",
                "spot_sell_commission_twd",
                "spot_sell_tax_twd",
                "futures_entry_tax_twd",
                "futures_exit_tax_twd",
                "futures_entry_commission_twd",
                "futures_exit_commission_twd",
            )
        )
        .otherwise(None)
        .alias("total_transaction_cost_twd")
    ).with_columns(
        pl.when(completed)
        .then(pl.col("gross_cycle_pnl_twd") - pl.col("total_transaction_cost_twd"))
        .otherwise(None)
        .alias("net_cycle_pnl_twd"),
        pl.when(completed)
        .then(
            pl.col("total_transaction_cost_twd")
            / pl.col("normalization_notional_twd")
            * 10_000.0
        )
        .otherwise(None)
        .alias("effective_transaction_cost_bp"),
    )
    return result.with_columns(
        pl.when(completed)
        .then(
            pl.col("net_cycle_pnl_twd")
            / pl.col("normalization_notional_twd")
            * 10_000.0
        )
        .otherwise(None)
        .alias("net_cycle_pnl_bp")
    )


def summarize_dynamic_path_coverage(priced_or_terminal: pl.DataFrame) -> pl.DataFrame:
    """One-row population/terminal/cost coverage summary."""

    if priced_or_terminal.is_empty():
        return pl.DataFrame(
            {
                "entry_positions": [0],
                "priced_terminal_paths": [0],
                "unresolved_paths": [0],
                "priced_terminal_coverage": [None],
            }
        )
    frame = priced_or_terminal
    priced = frame.filter(pl.col("terminal_cashflow_priced"))
    status = {
        str(key): int(value)
        for key, value in frame.group_by("path_status").len().iter_rows()
    }
    total = frame.height
    same_day_all = priced.filter(pl.col("Date") == pl.col("terminal_date")).height
    same_day_frozen_lower = status.get("same_day_frozen_lower_hit", 0)
    result: dict[str, list[object]] = {
        "entry_positions": [total],
        "entry_pricing_supported": [
            int(frame.get_column("entry_pricing_supported").sum() or 0)
        ],
        "priced_terminal_paths": [priced.height],
        "unresolved_paths": [total - priced.height],
        "priced_terminal_coverage": [priced.height / total],
        "same_day_frozen_lower_hits": [
            status.get("same_day_frozen_lower_hit", 0)
        ],
        "cross_session_frozen_lower_hits": [
            status.get("cross_session_frozen_lower_hit", 0)
        ],
        "expiry_last_joint_grid_fallbacks": [
            status.get(
                "expiry_last_joint_executable_grid_estimated_nonofficial", 0
            )
        ],
        "expiry_paired_accounting_mark_fallbacks": [
            int(priced["expiry_paired_accounting_mark_fallback"].sum() or 0)
            if priced.height
            else 0
        ],
        "observation_horizon_open": [status.get("observation_horizon_open", 0)],
        "expiry_unpriced": [status.get("expiry_unpriced", 0)],
        "entry_hedge_unpriced": [status.get("entry_hedge_unpriced", 0)],
        "entry_fill_cancel_race_at_1300_cutoff": [
            int(frame["entry_fill_cancel_race_at_1300_cutoff"].sum() or 0)
            if "entry_fill_cancel_race_at_1300_cutoff" in frame.columns
            else 0
        ],
        "same_day_close_rate_all_filled_entries": [
            same_day_all / total
        ],
        "same_day_close_rate_priced_paths": [
            same_day_all / priced.height
            if priced.height
            else None
        ],
        "same_day_frozen_lower_rate_all_filled_entries": [
            same_day_frozen_lower / total
        ],
        "same_day_frozen_lower_rate_priced_paths": [
            same_day_frozen_lower / priced.height if priced.height else None
        ],
        "fixed45_universe_used": [False],
        "makerfill_entry_is_approximate": [True],
        "exit_grid_is_one_second": [True],
        "pathwise_ev_ready": [False],
        "analysis_version": [ANALYSIS_VERSION],
    }
    if "total_transaction_cost_twd" in frame.columns and priced.height:
        result.update(
            {
                "gross_pnl_twd_priced_paths": [
                    float(priced["gross_cycle_pnl_twd"].sum())
                ],
                "transaction_cost_twd_priced_paths": [
                    float(priced["total_transaction_cost_twd"].sum())
                ],
                "net_pnl_twd_priced_paths": [
                    float(priced["net_cycle_pnl_twd"].sum())
                ],
                "net_pnl_on_entry_notional_bp_priced_paths": [
                    float(priced["net_cycle_pnl_twd"].sum())
                    / float(priced["normalization_notional_twd"].sum())
                    * 10_000.0
                ],
            }
        )
    if "entry_hedge_signed_total_slippage_bp" in frame.columns:
        executable_hedges = frame.filter(pl.col("entry_pricing_supported"))
        slippage = executable_hedges["entry_hedge_signed_total_slippage_bp"].drop_nulls()
        result.update(
            {
                "executable_entry_hedges": [executable_hedges.height],
                "entry_hedge_executable_rate": [executable_hedges.height / total],
                "entry_hedge_total_slippage_bp_mean": [
                    float(slippage.mean()) if slippage.len() else None
                ],
                "entry_hedge_total_slippage_bp_p50": [
                    float(slippage.median()) if slippage.len() else None
                ],
                "entry_hedge_total_slippage_bp_p95": [
                    float(slippage.quantile(0.95, "linear"))
                    if slippage.len()
                    else None
                ],
            }
        )
    return pl.DataFrame(result)


def run_completed_only_cap_backtest(
    priced_paths: pl.DataFrame,
    sessions: Sequence[str],
) -> PortfolioCapBacktestResult | None:
    """Run the existing cap engine on resolved paths, explicitly conditional."""

    resolved = priced_paths.filter(pl.col("terminal_cashflow_priced"))
    if resolved.is_empty():
        return None
    config = PortfolioCapBacktestConfig(
        scenarios=default_cap_scenarios(),
        per_product_fraction=0.30,
        entry_cutoff_local_time=time(13, 0),
        block_new_entries_for_products_held_at_session_open=True,
    )
    session_dates = sorted(
        set(_normalize_sessions(sessions))
        | set(resolved["Date"].to_list())
        | set(resolved["terminal_date"].to_list())
    )
    result = backtest_priced_paths(
        resolved.with_columns(pl.col("fill_ns").alias("entry_admission_time_ns")),
        config=config,
        session_dates=session_dates,
    )
    return PortfolioCapBacktestResult(
        events=result.events.with_columns(
            pl.lit(True).alias("completed_only_conditional_replay"),
            pl.lit(False).alias("unresolved_entries_admitted_or_reserved"),
        ),
        daily=result.daily.with_columns(
            pl.lit(True).alias("completed_only_conditional_replay"),
            pl.lit(False).alias("unresolved_entries_admitted_or_reserved"),
        ),
        summary=result.summary.with_columns(
            pl.lit(True).alias("completed_only_conditional_replay"),
            pl.lit(False).alias("unresolved_entries_admitted_or_reserved"),
            pl.lit(False).alias("full_population_portfolio_ev"),
        ),
    )


def backtest_full_population_inventory(
    paths: pl.DataFrame,
    sessions: Sequence[str],
    *,
    config: PortfolioCapBacktestConfig | None = None,
) -> FullPopulationCapResult | None:
    """Replay all fills while unresolved positions keep consuming capacity.

    Unlike :func:`portfolio_cap_backtester.backtest_priced_paths`, this
    analysis ledger accepts nullable terminal cashflows.  It never invents an
    exit for such rows: an admitted unresolved position stays active through
    the final reporting session.  Realized PnL therefore covers resolved exits
    only, while entry turnover and inventory cover the full fill population.
    """

    if paths.is_empty():
        return None
    required = {
        "Date",
        "ValueCode",
        "policy_path_id",
        "position_established_ns",
        "fill_ns",
        "normalization_notional_twd",
        "path_status",
        "terminal_cashflow_priced",
        "terminal_date",
        "exit_decision_time_ns",
        "gross_cycle_pnl_twd",
        "total_transaction_cost_twd",
        "net_cycle_pnl_twd",
    }
    _require_columns(paths, required, "full-population paths")
    cap_config = config or PortfolioCapBacktestConfig(
        scenarios=default_cap_scenarios(),
        per_product_fraction=0.30,
        entry_cutoff_local_time=time(13, 0),
        block_new_entries_for_products_held_at_session_open=True,
    )
    cap_config.validate()
    source = paths.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("policy_path_id").cast(pl.String),
        pl.col("position_established_ns").cast(pl.Int64),
        pl.col("normalization_notional_twd").cast(pl.Float64),
        pl.col("terminal_cashflow_priced").fill_null(False).cast(pl.Boolean),
        pl.col("terminal_date").cast(pl.String),
        pl.col("exit_decision_time_ns").cast(pl.Int64),
    ).sort(["Date", "position_established_ns", "ValueCode", "policy_path_id"])
    if source["policy_path_id"].n_unique() != source.height:
        raise ValueError("full-population policy_path_id values must be unique")
    invalid_notional = source.filter(
        ~pl.col("normalization_notional_twd").is_finite()
        | (pl.col("normalization_notional_twd") <= 0)
    )
    if invalid_notional.height:
        raise ValueError("full-population paths contain invalid entry notional")
    malformed_resolved = source.filter(
        pl.col("terminal_cashflow_priced")
        & (
            pl.col("terminal_date").is_null()
            | pl.col("exit_decision_time_ns").is_null()
            | ~pl.all_horizontal(
                pl.col(
                    "gross_cycle_pnl_twd",
                    "total_transaction_cost_twd",
                    "net_cycle_pnl_twd",
                ).is_finite()
            )
        )
    )
    malformed_unresolved = source.filter(
        ~pl.col("terminal_cashflow_priced")
        & (
            pl.col("terminal_date").is_not_null()
            | pl.col("exit_decision_time_ns").is_not_null()
            | pl.any_horizontal(
                pl.col(
                    "gross_cycle_pnl_twd",
                    "total_transaction_cost_twd",
                    "net_cycle_pnl_twd",
                ).is_not_null()
            )
        )
    )
    if malformed_resolved.height or malformed_unresolved.height:
        raise ValueError("resolved/unresolved terminal fields are incoherent")

    calendar_values = set(_normalize_sessions(sessions))
    calendar_values.update(str(value) for value in source["Date"].to_list())
    calendar_values.update(
        str(value)
        for value in source["terminal_date"].drop_nulls().to_list()
    )
    calendar = tuple(sorted(calendar_values))
    all_events: list[dict[str, object]] = []
    all_daily: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for scenario in cap_config.scenarios:
        events, accepted = _replay_full_population_scenario(
            source, scenario.hard_intraday_cap_twd, cap_config
        )
        daily = _full_population_daily(
            accepted,
            events,
            calendar,
            scenario.scenario_id,
            scenario.hard_intraday_cap_twd,
            cap_config,
        )
        summary = _full_population_summary(
            events,
            accepted,
            daily,
            scenario.scenario_id,
            scenario.hard_intraday_cap_twd,
            cap_config,
        )
        all_events.extend(events)
        all_daily.extend(daily)
        all_summary.append(summary)
    result = FullPopulationCapResult(
        events=pl.from_dicts(all_events, infer_schema_length=None).sort(
            ["scenario_id", "event_sequence"]
        ),
        daily=pl.from_dicts(all_daily, infer_schema_length=None).sort(
            ["scenario_id", "Date"]
        ),
        summary=pl.from_dicts(all_summary, infer_schema_length=None).sort(
            "hard_intraday_cap_twd"
        ),
    )
    if result.events.filter(
        (pl.col("active_portfolio_notional_after_event_twd")
         > pl.col("hard_intraday_cap_twd") + 1e-6)
        | (pl.col("active_same_product_notional_after_event_twd")
           > pl.col("per_product_hard_intraday_cap_twd") + 1e-6)
    ).height:
        raise AssertionError("full-population inventory replay exceeded a hard cap")
    return result


def _replay_full_population_scenario(
    paths: pl.DataFrame,
    hard_cap_twd: float,
    config: PortfolioCapBacktestConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scenario = next(
        item for item in config.scenarios if item.hard_intraday_cap_twd == hard_cap_twd
    )
    product_cap_twd = hard_cap_twd * config.per_product_fraction
    active: dict[str, dict[str, object]] = {}
    active_product: dict[str, float] = {}
    active_notional = 0.0
    exit_heap: list[tuple[str, int, str, dict[str, object]]] = []
    events: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    current_date: str | None = None
    opening_carry_products: set[str] = set()

    def emit_exit(row: dict[str, object]) -> None:
        nonlocal active_notional
        identifier = str(row["policy_path_id"])
        if identifier not in active:
            raise AssertionError("resolved accepted position is not active")
        product = str(row["ValueCode"])
        notional = float(row["normalization_notional_twd"])
        active_before = active_notional
        product_before = active_product[product]
        del active[identifier]
        active_notional -= notional
        active_product[product] -= notional
        if abs(active_notional) < 1e-8:
            active_notional = 0.0
        if abs(active_product[product]) < 1e-8:
            active_product[product] = 0.0
        events.append(
            {
                "scenario_id": scenario.scenario_id,
                "hard_intraday_cap_twd": hard_cap_twd,
                "per_product_hard_intraday_cap_twd": product_cap_twd,
                "event_sequence": len(events) + 1,
                "event_type": "position_exit",
                "event_date": str(row["terminal_date"]),
                "event_timestamp_ns": int(row["exit_decision_time_ns"]),
                "Date": str(row["Date"]),
                "ValueCode": product,
                "policy_path_id": identifier,
                "path_status": str(row["path_status"]),
                "entry_admission_status": None,
                "entry_admitted": None,
                "entry_cutoff_blocked": None,
                "opening_carry_exit_only_blocked": None,
                "portfolio_cap_blocked": None,
                "per_product_cap_blocked": None,
                "terminal_cashflow_priced": True,
                "capacity_notional_twd": notional,
                "accepted_new_spot_notional_twd": None,
                "released_capacity_notional_twd": notional,
                "gross_cycle_pnl_twd": float(row["gross_cycle_pnl_twd"]),
                "total_transaction_cost_twd": float(
                    row["total_transaction_cost_twd"]
                ),
                "net_cycle_pnl_twd": float(row["net_cycle_pnl_twd"]),
                "active_portfolio_notional_before_event_twd": active_before,
                "active_portfolio_notional_after_event_twd": active_notional,
                "active_same_product_notional_before_event_twd": product_before,
                "active_same_product_notional_after_event_twd": active_product[product],
                "unresolved_active_positions_after_event": sum(
                    not bool(item["terminal_cashflow_priced"])
                    for item in active.values()
                ),
                "full_population_inventory_replay": True,
                "unresolved_cashflow_imputed": False,
            }
        )

    for row in paths.iter_rows(named=True):
        entry_date = str(row["Date"])
        entry_ns = int(row["position_established_ns"])
        if entry_date != current_date:
            open_ns = local_session_timestamp_ns(
                entry_date, config.session_open_local_time, config.timezone_name
            )
            while exit_heap and (exit_heap[0][0], exit_heap[0][1]) < (
                entry_date,
                open_ns,
            ):
                _, _, _, exiting = heapq.heappop(exit_heap)
                emit_exit(exiting)
            opening_carry_products = {
                str(item["ValueCode"])
                for item in active.values()
                if str(item["Date"]) < entry_date
            }
            current_date = entry_date
        while exit_heap and (exit_heap[0][0], exit_heap[0][1]) < (
            entry_date,
            entry_ns,
        ):
            _, _, _, exiting = heapq.heappop(exit_heap)
            emit_exit(exiting)

        product = str(row["ValueCode"])
        identifier = str(row["policy_path_id"])
        notional = float(row["normalization_notional_twd"])
        before = active_notional
        product_before = active_product.get(product, 0.0)
        prospective = before + notional
        prospective_product = product_before + notional
        cutoff_ns = local_session_timestamp_ns(
            entry_date, config.entry_cutoff_local_time, config.timezone_name
        )
        admission_ns = int(row["fill_ns"])
        cutoff_blocked = admission_ns >= cutoff_ns
        carry_blocked = (
            config.block_new_entries_for_products_held_at_session_open
            and product in opening_carry_products
        )
        portfolio_blocked = prospective > hard_cap_twd + 1e-9
        product_blocked = prospective_product > product_cap_twd + 1e-9
        admitted = not any(
            (cutoff_blocked, carry_blocked, portfolio_blocked, product_blocked)
        )
        if cutoff_blocked:
            status = "rejected_entry_cutoff"
        elif carry_blocked:
            status = "rejected_opening_carry_exit_only"
        elif portfolio_blocked and product_blocked:
            status = "rejected_both_caps"
        elif portfolio_blocked:
            status = "rejected_portfolio_cap"
        elif product_blocked:
            status = "rejected_product_cap"
        else:
            status = "accepted"
        if admitted:
            active[identifier] = row
            active_notional = prospective
            active_product[product] = prospective_product
            accepted.append(row)
            if bool(row["terminal_cashflow_priced"]):
                heapq.heappush(
                    exit_heap,
                    (
                        str(row["terminal_date"]),
                        int(row["exit_decision_time_ns"]),
                        identifier,
                        row,
                    ),
                )
        events.append(
            {
                "scenario_id": scenario.scenario_id,
                "hard_intraday_cap_twd": hard_cap_twd,
                "per_product_hard_intraday_cap_twd": product_cap_twd,
                "event_sequence": len(events) + 1,
                "event_type": "entry_candidate",
                "event_date": entry_date,
                "event_timestamp_ns": entry_ns,
                "entry_admission_time_ns": admission_ns,
                "Date": entry_date,
                "ValueCode": product,
                "policy_path_id": identifier,
                "path_status": str(row["path_status"]),
                "entry_admission_status": status,
                "entry_admitted": admitted,
                "entry_cutoff_blocked": cutoff_blocked,
                "entry_cutoff_uses_fill_not_hedge_time": True,
                "opening_carry_exit_only_blocked": carry_blocked,
                "portfolio_cap_blocked": portfolio_blocked,
                "per_product_cap_blocked": product_blocked,
                "terminal_cashflow_priced": bool(row["terminal_cashflow_priced"]),
                "capacity_notional_twd": notional,
                "accepted_new_spot_notional_twd": notional if admitted else None,
                "released_capacity_notional_twd": None,
                "gross_cycle_pnl_twd": None,
                "total_transaction_cost_twd": None,
                "net_cycle_pnl_twd": None,
                "active_portfolio_notional_before_event_twd": before,
                "active_portfolio_notional_after_event_twd": active_notional,
                "active_same_product_notional_before_event_twd": product_before,
                "active_same_product_notional_after_event_twd": active_product.get(
                    product, 0.0
                ),
                "unresolved_active_positions_after_event": sum(
                    not bool(item["terminal_cashflow_priced"])
                    for item in active.values()
                ),
                "full_population_inventory_replay": True,
                "unresolved_cashflow_imputed": False,
            }
        )
    while exit_heap:
        _, _, _, exiting = heapq.heappop(exit_heap)
        emit_exit(exiting)
    # Deliberately do not emit exits for unresolved active rows.
    return events, accepted


def _full_population_daily(
    accepted: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    calendar: Sequence[str],
    scenario_id: str,
    hard_cap_twd: float,
    config: PortfolioCapBacktestConfig,
) -> list[dict[str, object]]:
    events_by_date: dict[str, list[Mapping[str, object]]] = {
        day: [] for day in calendar
    }
    for event in events:
        events_by_date.setdefault(str(event["event_date"]), []).append(event)
    rows: list[dict[str, object]] = []
    cumulative_net = 0.0
    peak_cumulative = 0.0
    for day in calendar:
        open_ns = local_session_timestamp_ns(
            day, config.session_open_local_time, config.timezone_name
        )
        close_ns = local_session_timestamp_ns(
            day, config.session_close_local_time, config.timezone_name
        )
        day_events = events_by_date.get(day, [])
        entries = [
            event for event in day_events if event["event_type"] == "entry_candidate"
        ]
        admitted_entries = [event for event in entries if event["entry_admitted"]]
        exits = [event for event in day_events if event["event_type"] == "position_exit"]
        opening = [
            path
            for path in accepted
            if int(path["position_established_ns"]) < open_ns
            and (
                not bool(path["terminal_cashflow_priced"])
                or int(path["exit_decision_time_ns"]) >= open_ns
            )
        ]
        eod = [
            path
            for path in accepted
            if int(path["position_established_ns"]) <= close_ns
            and (
                not bool(path["terminal_cashflow_priced"])
                or int(path["exit_decision_time_ns"]) > close_ns
            )
        ]
        unresolved_eod = [
            path for path in eod if not bool(path["terminal_cashflow_priced"])
        ]
        known_carry_eod = [
            path
            for path in eod
            if bool(path["terminal_cashflow_priced"])
            and str(path["terminal_date"]) > day
        ]
        day_gross = sum(float(event["gross_cycle_pnl_twd"]) for event in exits)
        day_cost = sum(
            float(event["total_transaction_cost_twd"]) for event in exits
        )
        day_net = sum(float(event["net_cycle_pnl_twd"]) for event in exits)
        cumulative_net += day_net
        peak_cumulative = max(peak_cumulative, cumulative_net)
        eod_notional = sum(float(path["normalization_notional_twd"]) for path in eod)
        opening_notional = sum(
            float(path["normalization_notional_twd"]) for path in opening
        )
        intraday_peak = max(
            [opening_notional]
            + [
                float(event["active_portfolio_notional_after_event_twd"])
                for event in day_events
            ]
        )
        intraday_product_peak = max(
            [0.0]
            + [
                float(event["active_same_product_notional_after_event_twd"])
                for event in day_events
            ]
            + [
                sum(
                    float(path["normalization_notional_twd"])
                    for path in opening
                    if str(path["ValueCode"]) == product
                )
                for product in {str(path["ValueCode"]) for path in opening}
            ]
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "Date": day,
                "hard_intraday_cap_twd": hard_cap_twd,
                "per_product_hard_intraday_cap_twd": (
                    hard_cap_twd * config.per_product_fraction
                ),
                "entry_candidates": len(entries),
                "accepted_entries": len(admitted_entries),
                "rejected_entries": len(entries) - len(admitted_entries),
                "rejected_entry_cutoff": sum(
                    event["entry_admission_status"] == "rejected_entry_cutoff"
                    for event in entries
                ),
                "rejected_opening_carry_exit_only": sum(
                    event["entry_admission_status"]
                    == "rejected_opening_carry_exit_only"
                    for event in entries
                ),
                "rejected_portfolio_cap": sum(
                    event["entry_admission_status"] == "rejected_portfolio_cap"
                    for event in entries
                ),
                "rejected_product_cap": sum(
                    event["entry_admission_status"] == "rejected_product_cap"
                    for event in entries
                ),
                "rejected_both_caps": sum(
                    event["entry_admission_status"] == "rejected_both_caps"
                    for event in entries
                ),
                "new_spot_notional_twd": sum(
                    float(event["accepted_new_spot_notional_twd"])
                    for event in admitted_entries
                ),
                "resolved_exits": len(exits),
                "same_day_frozen_lower_exits": sum(
                    event["path_status"] == "same_day_frozen_lower_hit"
                    for event in exits
                ),
                "same_day_all_resolved_exits": sum(
                    str(event["Date"]) == str(event["event_date"])
                    for event in exits
                ),
                "cross_session_frozen_lower_exits": sum(
                    event["path_status"] == "cross_session_frozen_lower_hit"
                    for event in exits
                ),
                "expiry_grid_exits": sum(
                    event["path_status"]
                    == "expiry_last_joint_executable_grid_estimated_nonofficial"
                    for event in exits
                ),
                "expiry_paired_accounting_mark_exits": sum(
                    str(event["path_status"]).startswith("expiry_")
                    and "accounting_mark_non_executable" in str(event["path_status"])
                    for event in exits
                ),
                "opening_positions": len(opening),
                "opening_spot_notional_twd": opening_notional,
                "opening_carry_products": len(
                    {str(path["ValueCode"]) for path in opening}
                ),
                "intraday_peak_spot_notional_twd": intraday_peak,
                "intraday_peak_cap_utilization": intraday_peak / hard_cap_twd,
                "intraday_peak_single_product_spot_notional_twd": (
                    intraday_product_peak
                ),
                "eod_positions": len(eod),
                "eod_spot_notional_twd": eod_notional,
                "eod_cap_utilization": eod_notional / hard_cap_twd,
                "known_future_terminal_eod_positions": len(known_carry_eod),
                "unresolved_eod_positions": len(unresolved_eod),
                "unresolved_eod_spot_notional_twd": sum(
                    float(path["normalization_notional_twd"])
                    for path in unresolved_eod
                ),
                "realized_gross_pnl_twd": day_gross,
                "realized_transaction_cost_twd": day_cost,
                "realized_net_pnl_twd": day_net,
                "cumulative_realized_net_pnl_twd": cumulative_net,
                "realized_drawdown_twd": peak_cumulative - cumulative_net,
                "realized_pnl_excludes_open_unresolved": True,
                "unresolved_cashflow_imputed": False,
                "full_population_inventory_replay": True,
                "opening_carry_product_exit_only_policy": (
                    config.block_new_entries_for_products_held_at_session_open
                ),
                "entry_cutoff_is_exclusive": True,
                "notional_basis": "one_way_spot_entry_notional_twd",
            }
        )
    return rows


def _full_population_summary(
    events: Sequence[Mapping[str, object]],
    accepted: Sequence[Mapping[str, object]],
    daily: Sequence[Mapping[str, object]],
    scenario_id: str,
    hard_cap_twd: float,
    config: PortfolioCapBacktestConfig,
) -> dict[str, object]:
    entries = [event for event in events if event["event_type"] == "entry_candidate"]
    admitted = [event for event in entries if event["entry_admitted"]]
    exits = [event for event in events if event["event_type"] == "position_exit"]
    accepted_resolved = [
        path for path in accepted if bool(path["terminal_cashflow_priced"])
    ]
    accepted_unresolved = [
        path for path in accepted if not bool(path["terminal_cashflow_priced"])
    ]
    total_new_spot = sum(float(row["new_spot_notional_twd"]) for row in daily)
    total_gross = sum(float(event["gross_cycle_pnl_twd"]) for event in exits)
    total_cost = sum(float(event["total_transaction_cost_twd"]) for event in exits)
    total_net = sum(float(event["net_cycle_pnl_twd"]) for event in exits)
    same_day_frozen_lower = sum(
        str(path["path_status"]) == "same_day_frozen_lower_hit"
        for path in accepted_resolved
    )
    same_day_all_resolved = sum(
        str(path["Date"]) == str(path["terminal_date"])
        for path in accepted_resolved
    )
    return {
        "scenario_id": scenario_id,
        "hard_intraday_cap_twd": hard_cap_twd,
        "per_product_fraction": config.per_product_fraction,
        "per_product_hard_intraday_cap_twd": (
            hard_cap_twd * config.per_product_fraction
        ),
        "session_count": len(daily),
        "candidate_filled_entries": len(entries),
        "accepted_filled_entries": len(admitted),
        "rejected_filled_entries": len(entries) - len(admitted),
        "acceptance_rate": len(admitted) / len(entries) if entries else None,
        "rejected_entry_cutoff": sum(
            event["entry_admission_status"] == "rejected_entry_cutoff"
            for event in entries
        ),
        "rejected_opening_carry_exit_only": sum(
            event["entry_admission_status"] == "rejected_opening_carry_exit_only"
            for event in entries
        ),
        "rejected_portfolio_cap": sum(
            event["entry_admission_status"] == "rejected_portfolio_cap"
            for event in entries
        ),
        "rejected_product_cap": sum(
            event["entry_admission_status"] == "rejected_product_cap"
            for event in entries
        ),
        "rejected_both_caps": sum(
            event["entry_admission_status"] == "rejected_both_caps"
            for event in entries
        ),
        "accepted_resolved_paths": len(accepted_resolved),
        "accepted_unresolved_open_paths": len(accepted_unresolved),
        "accepted_terminal_pricing_coverage": (
            len(accepted_resolved) / len(accepted) if accepted else None
        ),
        "same_day_close_rate_all_accepted": (
            same_day_all_resolved / len(accepted) if accepted else None
        ),
        "same_day_close_rate_accepted_resolved": (
            same_day_all_resolved / len(accepted_resolved)
            if accepted_resolved
            else None
        ),
        "same_day_frozen_lower_rate_all_accepted": (
            same_day_frozen_lower / len(accepted) if accepted else None
        ),
        "total_new_spot_notional_twd": total_new_spot,
        "mean_daily_new_spot_notional_twd": total_new_spot / len(daily),
        "new_spot_notional_cap_turns": total_new_spot / hard_cap_twd,
        "mean_daily_new_spot_cap_turns": (
            total_new_spot / hard_cap_twd / len(daily)
        ),
        "mean_eod_spot_notional_twd": sum(
            float(row["eod_spot_notional_twd"]) for row in daily
        )
        / len(daily),
        "peak_eod_spot_notional_twd": max(
            float(row["eod_spot_notional_twd"]) for row in daily
        ),
        "peak_intraday_spot_notional_twd": max(
            float(row["intraday_peak_spot_notional_twd"]) for row in daily
        ),
        "mean_daily_intraday_peak_spot_notional_twd": sum(
            float(row["intraday_peak_spot_notional_twd"]) for row in daily
        )
        / len(daily),
        "final_eod_spot_notional_twd": float(daily[-1]["eod_spot_notional_twd"]),
        "final_unresolved_eod_positions": int(daily[-1]["unresolved_eod_positions"]),
        "final_unresolved_eod_spot_notional_twd": float(
            daily[-1]["unresolved_eod_spot_notional_twd"]
        ),
        "realized_gross_pnl_twd_resolved_only": total_gross,
        "realized_transaction_cost_twd_resolved_only": total_cost,
        "realized_net_pnl_twd_resolved_only": total_net,
        "net_pnl_on_new_spot_notional_bp_resolved_only": (
            total_net / total_new_spot * 10_000.0 if total_new_spot else None
        ),
        "worst_daily_realized_net_pnl_twd": min(
            float(row["realized_net_pnl_twd"]) for row in daily
        ),
        "max_realized_drawdown_twd": max(
            float(row["realized_drawdown_twd"]) for row in daily
        ),
        "full_population_inventory_replay": True,
        "unresolved_positions_keep_capacity_through_horizon": True,
        "unresolved_cashflow_imputed": False,
        "realized_pnl_excludes_open_unresolved": True,
        "all_accepted_terminal_cashflows_priced": len(accepted_unresolved) == 0,
        "full_population_portfolio_ev": False,
        "opening_carry_product_exit_only_policy": (
            config.block_new_entries_for_products_held_at_session_open
        ),
        "entry_cutoff_is_exclusive": True,
        "notional_basis": "one_way_spot_entry_notional_twd",
    }


def build_dynamic_estimated_paths(
    entry_outcomes: pl.DataFrame,
    causal_fair: pl.DataFrame,
    sessions: Sequence[str],
    *,
    hedge_facts: pl.DataFrame | None = None,
    manifest: pl.DataFrame | None = None,
    expiry_close_facts: pl.DataFrame | None = None,
    config: DynamicEstimatedPathConfig = DEFAULT_PATH_CONFIG,
) -> DynamicEstimatedPathResult:
    """Pure in-memory end-to-end analysis entry point."""

    positions = build_dynamic_entry_positions(
        entry_outcomes,
        hedge_facts=hedge_facts,
        manifest=manifest,
        config=config,
    )
    terminals = label_dynamic_frozen_lower_paths(
        positions,
        causal_fair,
        sessions,
        expiry_close_facts=expiry_close_facts,
        config=config,
    )
    all_costed = price_dynamic_terminal_paths(terminals, config.cost_profile)
    priced = all_costed.filter(pl.col("terminal_cashflow_priced"))
    unresolved = all_costed.filter(~pl.col("terminal_cashflow_priced"))
    coverage = summarize_dynamic_path_coverage(all_costed)
    full_cap = backtest_full_population_inventory(all_costed, sessions)
    cap = run_completed_only_cap_backtest(priced, sessions)
    return DynamicEstimatedPathResult(
        entry_positions=positions,
        terminal_paths=all_costed,
        priced_paths=priced,
        unresolved_paths=unresolved,
        coverage=coverage,
        full_population_cap=full_cap,
        completed_only_cap=cap,
    )


def load_candidate_outcomes(root: Path = DEFAULT_ENTRY_ROOT) -> pl.DataFrame:
    paths = sorted(Path(root).glob("candidate_outcomes/Date=*/candidate_outcomes.parquet"))
    if not paths:
        raise FileNotFoundError(f"no candidate outcome partitions under {root}")
    return pl.scan_parquet(paths).collect(engine="streaming")


def load_causal_fair_for_positions(
    positions: pl.DataFrame,
    *,
    daily_root: Path = DEFAULT_DAILY_ROOT,
) -> pl.DataFrame:
    """Read all available carry dates for position products/contracts.

    Selection is based only on established positions.  It never intersects a
    later session with that session's entry universe.
    """

    if positions.is_empty():
        return pl.DataFrame()
    value_codes = positions["ValueCode"].unique().to_list()
    quote_codes = positions["QuoteCode"].unique().to_list()
    minimum_date = str(positions["Date"].min())
    paths = {
        path.parent.name.removeprefix("Date="): path
        for path in completed_artifact_paths(Path(daily_root), "causal_fair.parquet")
    }
    selected = [path for day, path in sorted(paths.items()) if day >= minimum_date]
    if not selected:
        raise FileNotFoundError("no causal_fair partitions cover entry positions")
    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "contract_size",
        "spot_bid",
        "spot_bid_lots",
        "fut_exec_ask",
        "fut_exec_ask_lots",
        "basis_buy_taker_bp",
        "analysis_eligible",
    ]
    frames = [
        pl.scan_parquet(path)
        .filter(
            pl.col("ValueCode").is_in(value_codes)
            & pl.col("QuoteCode").is_in(quote_codes)
        )
        .select(columns)
        .collect(engine="streaming")
        for path in selected
    ]
    return pl.concat(frames, how="vertical_relaxed")


def label_dynamic_frozen_lower_paths_partitioned(
    entry_positions: pl.DataFrame,
    sessions: Sequence[str],
    *,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    additional_causal_fair_paths: Sequence[Path] = (),
    expiry_close_facts: pl.DataFrame | None = None,
    config: DynamicEstimatedPathConfig = DEFAULT_PATH_CONFIG,
) -> pl.DataFrame:
    """Low-memory day-batched equivalent of the in-memory path labeler.

    Each causal-fair partition is projected and scanned at most once.  The
    filter is derived from positions still alive on that date, never from the
    date's entry allowlist, so a removed product remains exit-eligible.
    """

    config.validate()
    if entry_positions.is_empty():
        return _empty_terminal_paths()
    calendar = _normalize_sessions(sessions)
    calendar_set = set(calendar)
    unknown = sorted(set(entry_positions["Date"].to_list()) - calendar_set)
    if unknown:
        raise ValueError(f"entry dates absent from session calendar: {unknown[:5]}")
    sources_by_date: dict[str, list[Path]] = {
        path.parent.name.removeprefix("Date="): [path]
        for path in completed_artifact_paths(Path(daily_root), "causal_fair.parquet")
    }
    for additional in additional_causal_fair_paths:
        path = Path(additional)
        if not path.is_file():
            raise FileNotFoundError(path)
        schema = pl.read_parquet_schema(path)
        if "Date" not in schema:
            raise ValueError(f"additional causal-fair path lacks Date: {path}")
        dates = (
            pl.scan_parquet(path)
            .select(pl.col("Date").cast(pl.String).unique())
            .collect(engine="streaming")
            .get_column("Date")
            .to_list()
        )
        for day in dates:
            sources_by_date.setdefault(str(day), []).append(path)
    observed_dates = tuple(sorted(set(sources_by_date) & calendar_set))
    if not observed_dates:
        raise FileNotFoundError("no completed causal_fair partitions in session calendar")
    minimum_entry = str(entry_positions["Date"].min())
    observed_dates = tuple(day for day in observed_dates if day >= minimum_entry)
    last_observed = observed_dates[-1]
    close_lookup = _expiry_close_lookup(expiry_close_facts)
    pending: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for row in entry_positions.iter_rows(named=True):
        if not bool(row["entry_pricing_supported"]):
            records.append(
                _unresolved_record(
                    row,
                    "entry_hedge_unpriced",
                    str(row.get("entry_pricing_unsupported_reason") or "unknown"),
                    last_observed,
                    config,
                )
            )
        else:
            pending[str(row["physical_order_id"])] = row

    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "contract_size",
        "spot_bid",
        "spot_bid_lots",
        "fut_exec_ask",
        "fut_exec_ask_lots",
        "basis_buy_taker_bp",
        config.eligibility_column,
    ]
    for day in observed_dates:
        active = [
            row
            for row in pending.values()
            if str(row["Date"]) <= day <= str(row["expiry_session"])
        ]
        if not active:
            continue
        value_codes = sorted({str(row["ValueCode"]) for row in active})
        quote_codes = sorted({str(row["QuoteCode"]) for row in active})
        fair_parts = [
            pl.scan_parquet(path)
            .filter(
                (pl.col("Date").cast(pl.String) == day)
                & pl.col("ValueCode").is_in(value_codes)
                & pl.col("QuoteCode").is_in(quote_codes)
            )
            .select(columns)
            .collect(engine="streaming")
            for path in sources_by_date[day]
        ]
        fair = pl.concat(fair_parts, how="vertical_relaxed")
        duplicate_grid = fair.group_by(
            "Date", "ValueCode", "QuoteCode", "timestamp"
        ).len().filter(pl.col("len") != 1)
        if duplicate_grid.height:
            raise ValueError(
                f"{day}: base/additional causal-fair sources overlap exact grid keys"
            )
        day_series = _build_fair_series(fair, config)
        completed_ids: list[str] = []
        for position in active:
            identifier = str(position["physical_order_id"])
            key = (
                day,
                str(position["ValueCode"]),
                str(position["QuoteCode"]),
                int(position["entry_contract_size_shares"]),
                int(position["entry_future_contract_quantity"]),
            )
            path = day_series.get(key)
            if path is not None:
                earliest = (
                    _next_full_second_ns(int(position["position_established_ns"]))
                    if day == str(position["Date"])
                    and config.first_exit_on_next_full_second
                    else int(position["position_established_ns"])
                    if day == str(position["Date"])
                    else 0
                )
                hit = path.first_hit(
                    earliest, float(position["exit_threshold_basis_bp"])
                )
                if hit is not None:
                    records.append(
                        _terminal_record(
                            position,
                            terminal_date=day,
                            exit_ns=path.timestamps_ns[hit],
                            exit_spot=path.spot_prices[hit],
                            exit_future=path.future_prices[hit],
                            exit_basis=path.basis_bp[hit],
                            status="same_day_frozen_lower_hit"
                            if day == str(position["Date"])
                            else "cross_session_frozen_lower_hit",
                            executable_grid=True,
                            expiry_grid_fallback=False,
                            paired_expiry_mark_fallback=False,
                            config=config,
                        )
                    )
                    completed_ids.append(identifier)
                    continue
            if day != str(position["expiry_session"]):
                continue
            close = close_lookup.get(
                (
                    day,
                    str(position["ValueCode"]),
                    str(position["QuoteCode"]),
                )
            )
            if config.expiry_paired_accounting_mark_fallback and close is not None:
                records.append(_paired_expiry_mark_terminal(position, close, config))
                completed_ids.append(identifier)
                continue
            if config.expiry_grid_fallback and path is not None:
                last_index = path.last_index()
                if last_index is not None:
                    records.append(
                        _terminal_record(
                            position,
                            terminal_date=day,
                            exit_ns=path.timestamps_ns[last_index],
                            exit_spot=path.spot_prices[last_index],
                            exit_future=path.future_prices[last_index],
                            exit_basis=path.basis_bp[last_index],
                            status=(
                                "expiry_last_joint_executable_grid_"
                                "estimated_nonofficial"
                            ),
                            executable_grid=True,
                            expiry_grid_fallback=True,
                            paired_expiry_mark_fallback=False,
                            config=config,
                        )
                    )
                    completed_ids.append(identifier)
                    continue
            records.append(
                _unresolved_record(
                    position,
                    "expiry_unpriced",
                    "expiry_reached_without_joint_grid_or_paired_close_fact",
                    last_observed,
                    config,
                )
            )
            completed_ids.append(identifier)
        for identifier in completed_ids:
            pending.pop(identifier, None)

    for position in pending.values():
        close = close_lookup.get(
            (
                str(position["expiry_session"]),
                str(position["ValueCode"]),
                str(position["QuoteCode"]),
            )
        )
        if config.expiry_paired_accounting_mark_fallback and close is not None:
            records.append(_paired_expiry_mark_terminal(position, close, config))
            continue
        expiry = str(position["expiry_session"])
        records.append(
            _unresolved_record(
                position,
                "observation_horizon_open" if expiry > last_observed else "expiry_unpriced",
                "expiry_not_yet_observed_and_no_paired_accounting_mark"
                if expiry > last_observed
                else "expiry_reached_without_joint_grid_or_paired_close_fact",
                last_observed,
                config,
            )
        )
    return pl.from_dicts(records, infer_schema_length=None).sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )


def _paired_expiry_mark_terminal(
    position: Mapping[str, object],
    close: Mapping[str, object],
    config: DynamicEstimatedPathConfig,
) -> dict[str, object]:
    expiry = str(position["expiry_session"])
    spot = float(close["spot_close_price"])
    future = float(close["future_close_price"])
    future_proxy = bool(close.get("future_close_is_last_trade_proxy", False))
    fully_official = bool(
        close.get("paired_mark_is_fully_official_close", not future_proxy)
    )
    status = (
        "expiry_spot_close_future_last_trade_proxy_"
        "accounting_mark_non_executable"
        if future_proxy
        else "expiry_paired_official_close_accounting_mark_non_executable"
        if fully_official
        else "expiry_paired_accounting_mark_non_executable"
    )
    return _terminal_record(
        position,
        terminal_date=expiry,
        exit_ns=local_session_timestamp_ns(
            expiry, time(13, 30), config.timezone_name
        ),
        exit_spot=spot,
        exit_future=future,
        exit_basis=(future / spot - 1.0) * 10_000.0,
        status=status,
        executable_grid=False,
        expiry_grid_fallback=False,
        paired_expiry_mark_fallback=True,
        config=config,
        expiry_mark_metadata=close,
    )


def run_dynamic_estimated_path_portfolio(
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    hedge_facts_path: Path,
    output_dir: Path,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expiry_close_path: Path | None = DEFAULT_EXPIRY_CLOSE_PATH,
    additional_causal_fair_paths: Sequence[Path] = (),
    sessions_path: Path = DEFAULT_SESSIONS_PATH,
    config: DynamicEstimatedPathConfig = DEFAULT_PATH_CONFIG,
) -> DynamicEstimatedPathResult:
    """Load, estimate, cost, replay, and atomically publish a new analysis root."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    entries = load_candidate_outcomes(entry_root)
    hedges = pl.read_parquet(hedge_facts_path)
    manifest = pl.read_csv(manifest_path, schema_overrides={
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
    })
    positions = build_dynamic_entry_positions(
        entries, hedge_facts=hedges, manifest=manifest, config=config
    )
    minimum_entry_date = str(positions["Date"].min())
    sessions = tuple(
        day
        for day in augment_sessions_with_raw_inventory_calendar(
            _read_sessions(sessions_path)
        )
        if day >= minimum_entry_date
    )
    expiry = (
        pl.read_parquet(expiry_close_path)
        if expiry_close_path is not None and Path(expiry_close_path).is_file()
        else None
    )
    terminals = label_dynamic_frozen_lower_paths_partitioned(
        positions,
        sessions,
        daily_root=daily_root,
        additional_causal_fair_paths=additional_causal_fair_paths,
        expiry_close_facts=expiry,
        config=config,
    )
    all_costed = price_dynamic_terminal_paths(terminals, config.cost_profile)
    priced = all_costed.filter(pl.col("terminal_cashflow_priced"))
    unresolved = all_costed.filter(~pl.col("terminal_cashflow_priced"))
    result = DynamicEstimatedPathResult(
        entry_positions=positions,
        terminal_paths=all_costed,
        priced_paths=priced,
        unresolved_paths=unresolved,
        coverage=summarize_dynamic_path_coverage(all_costed),
        full_population_cap=backtest_full_population_inventory(
            all_costed, sessions
        ),
        completed_only_cap=run_completed_only_cap_backtest(priced, sessions),
    )
    _publish_result(
        result,
        destination,
        config,
        sources={
            "entry_root": str(Path(entry_root).resolve()),
            "hedge_facts_path": str(Path(hedge_facts_path).resolve()),
            "manifest_path": str(Path(manifest_path).resolve()),
            "daily_root": str(Path(daily_root).resolve()),
            "expiry_close_path": (
                str(Path(expiry_close_path).resolve())
                if expiry_close_path is not None
                else None
            ),
            "additional_causal_fair_paths": [
                str(Path(path).resolve()) for path in additional_causal_fair_paths
            ],
        },
    )
    return result


def _publish_result(
    result: DynamicEstimatedPathResult,
    destination: Path,
    config: DynamicEstimatedPathConfig,
    *,
    sources: Mapping[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        result.entry_positions.write_parquet(stage / "entry_positions.parquet")
        result.terminal_paths.write_parquet(stage / "estimated_terminal_paths.parquet")
        result.priced_paths.write_parquet(stage / "estimated_priced_paths.parquet")
        result.unresolved_paths.write_parquet(stage / "unresolved_paths.parquet")
        result.coverage.write_csv(stage / "coverage.csv")
        if result.full_population_cap is not None:
            result.full_population_cap.events.write_parquet(
                stage / "full_population_cap_events.parquet"
            )
            result.full_population_cap.daily.write_parquet(
                stage / "full_population_cap_daily.parquet"
            )
            result.full_population_cap.summary.write_csv(
                stage / "full_population_cap_summary.csv"
            )
        if result.completed_only_cap is not None:
            result.completed_only_cap.events.write_parquet(
                stage / "completed_only_cap_events.parquet"
            )
            result.completed_only_cap.daily.write_parquet(
                stage / "completed_only_cap_daily.parquet"
            )
            result.completed_only_cap.summary.write_csv(
                stage / "completed_only_cap_summary.csv"
            )
        artifacts = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        marker = {
            "complete": True,
            "analysis_version": config.analysis_version,
            "config": asdict(config),
            "sources": dict(sources),
            "entry_positions": result.entry_positions.height,
            "priced_paths": result.priced_paths.height,
            "unresolved_paths": result.unresolved_paths.height,
            "fixed45_universe_used": False,
            "completed_only_cap_is_conditional": True,
            "full_population_cap_retains_unresolved_inventory": True,
            "unresolved_cashflow_imputed": False,
            "pathwise_ev_ready": False,
            "artifacts": artifacts,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _canonicalize_aliases(
    frame: pl.DataFrame,
    aliases: Mapping[str, Sequence[str]],
    label: str,
) -> pl.DataFrame:
    result = frame
    missing: list[str] = []
    for canonical, choices in aliases.items():
        if canonical in result.columns:
            continue
        found = next((name for name in choices if name in result.columns), None)
        if found is None:
            missing.append(canonical)
        else:
            result = result.rename({found: canonical})
    if missing:
        raise ValueError(f"{label} missing canonical columns: {sorted(missing)}")
    return result


def _canonicalize_optional_aliases(
    frame: pl.DataFrame,
    aliases: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    result = frame
    for canonical, choices in aliases.items():
        if canonical in result.columns:
            continue
        found = next((name for name in choices if name in result.columns), None)
        if found is not None:
            result = result.rename({found: canonical})
    return result


def _validate_entry_manifest_membership(
    entries: pl.DataFrame, manifest: pl.DataFrame
) -> None:
    _require_columns(manifest, {"Date", "ValueCode", "QuoteCode"}, "entry manifest")
    keys = ["Date", "ValueCode", "QuoteCode"]
    allowed = manifest.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    ).unique()
    if allowed.select(keys).n_unique() != allowed.height:
        raise ValueError("entry manifest has duplicate exact product-contract keys")
    missing = entries.select(*keys).unique().join(
        allowed, on=keys, how="anti"
    )
    if missing.height:
        raise ValueError(
            "filled entry is outside its same-day causal manifest: "
            f"{missing.head(5).to_dicts()}"
        )


def _expiry_close_lookup(
    facts: pl.DataFrame | None,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    if facts is None or facts.is_empty():
        return {}
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "spot_close_price",
        "future_close_price",
    }
    _require_columns(facts, required, "expiry close facts")
    normalized = facts.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("spot_close_price").cast(pl.Float64),
        pl.col("future_close_price").cast(pl.Float64),
    )
    if normalized.select("Date", "ValueCode", "QuoteCode").n_unique() != normalized.height:
        raise ValueError("expiry close facts duplicate exact keys")
    invalid = normalized.filter(
        ~pl.all_horizontal(
            pl.col("spot_close_price", "future_close_price").is_finite()
        )
        | (pl.col("spot_close_price") <= 0)
        | (pl.col("future_close_price") <= 0)
    )
    if invalid.height:
        raise ValueError("expiry close facts contain invalid paired prices")
    return {
        (str(row["Date"]), str(row["ValueCode"]), str(row["QuoteCode"])): row
        for row in normalized.iter_rows(named=True)
    }


def _expiry_string_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.replace_all("-", "").str.slice(0, 8)


def _next_full_second_ns(value: int) -> int:
    return (value // ONE_SECOND_NS + 1) * ONE_SECOND_NS


def _normalize_sessions(values: Iterable[str]) -> tuple[str, ...]:
    sessions = tuple(str(value).strip() for value in values if str(value).strip())
    if not sessions or tuple(sorted(set(sessions))) != sessions:
        raise ValueError("sessions must be nonempty, ascending, and unique")
    if any(len(value) != 8 or not value.isdigit() for value in sessions):
        raise ValueError("sessions must use YYYYMMDD")
    return sessions


def _read_sessions(path: Path) -> tuple[str, ...]:
    return _normalize_sessions(Path(path).read_text(encoding="utf-8").splitlines())


def augment_sessions_with_raw_inventory_calendar(
    sessions: Sequence[str],
) -> tuple[str, ...]:
    """Extend the fair-grid calendar with locally observed raw trading days.

    Added dates are inventory-reporting dates only when no completed
    causal-fair partition exists.  They do not create exit opportunities.
    """

    base = _normalize_sessions(sessions)
    discovered: set[str] = set()
    for year in sorted({value[:4] for value in base}):
        discovered.update(discover_common_sessions(year=year))
    return _normalize_sessions(sorted(set(base) | discovered))


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _empty_entry_positions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "physical_order_id": pl.String,
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "position_established_ns": pl.Int64,
            "entry_spot_price": pl.Float64,
            "entry_future_price": pl.Float64,
            "entry_contract_size_shares": pl.Int64,
            "normalization_notional_twd": pl.Float64,
            "exit_threshold_basis_bp": pl.Float64,
            "expiry_session": pl.String,
            "entry_pricing_supported": pl.Boolean,
        }
    )


def _empty_terminal_paths() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            **_empty_entry_positions().schema,
            "policy_path_id": pl.String,
            "path_status": pl.String,
            "terminal_date": pl.String,
            "exit_decision_time_ns": pl.Int64,
            "terminal_cashflow_priced": pl.Boolean,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


__all__ = [
    "ANALYSIS_VERSION",
    "DynamicEstimatedPathConfig",
    "DynamicEstimatedPathResult",
    "augment_sessions_with_raw_inventory_calendar",
    "backtest_full_population_inventory",
    "build_dynamic_entry_positions",
    "build_dynamic_estimated_paths",
    "label_dynamic_frozen_lower_paths",
    "label_dynamic_frozen_lower_paths_partitioned",
    "load_candidate_outcomes",
    "load_causal_fair_for_positions",
    "price_dynamic_terminal_paths",
    "run_completed_only_cap_backtest",
    "run_dynamic_estimated_path_portfolio",
    "summarize_dynamic_path_coverage",
]
