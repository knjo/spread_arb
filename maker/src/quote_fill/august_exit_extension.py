"""Causal one-second carry-exit grid beyond the canonical daily-fact horizon.

The regular daily facts currently stop at 2026-08-13.  Positions established
by the dynamic (not fixed-45) entry replay can nevertheless remain open until
the August expiry session.  This module builds the missing 2026-08-14 and
2026-08-17--19 spot-bid/future-ask grid directly from raw tape.

The implementation deliberately keeps this bundle independent from the entry
hedge and portfolio runners.  Population comes only from canonical filled
positions, exact ``(ValueCode, QuoteCode, contract_size)`` pairs are retained,
and a product is never intersected with a later day's entry allow-list.

Futures L1 and Best quote snapshots persist independently.  A zero-book trade
row therefore does not erase either component, while a genuine L1 or Best
snapshot updates only its own component.  If L1 and Best expose the same
executable price, available lots are the maximum of the two fields, matching
the raw-tape execution convention used elsewhere in this research tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date as date_type
from datetime import datetime, time, timedelta
from pathlib import Path
from time import perf_counter

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT, futures_raw_path

DEFAULT_ENTRY_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "one_second_makerfill_causal_v2_20260822_v1"
)
DEFAULT_CONTRACT_METADATA_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "cross_session_prerequisites_v1_20260819"
    / "metadata"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "august_exit_extension_causal_v1_20260822"
)
DEFAULT_DATES = (
    "20260814",
    "20260817",
    "20260818",
    "20260819",
    "20260820",
    "20260821",
)

SESSION_STATE_START = time(9, 0)
SESSION_GRID_START = time(9, 5)
SESSION_GRID_END = time(13, 20)
NS_PER_SECOND = 1_000_000_000
REF_LOWER_RETURN = -0.09
REF_UPPER_RETURN = 0.08
REF_COMPARISON_EPS_RATIO = 1e-12
SCHEMA_VERSION = "dynamic_carry_exit_extension_1hz_v1"

REQUIRED_GRID_COLUMNS = (
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
)


@dataclass(frozen=True)
class ExtensionConfig:
    state_start: time = SESSION_STATE_START
    grid_start: time = SESSION_GRID_START
    grid_end: time = SESSION_GRID_END
    interval: str = "1s"
    ref_lower_return: float = REF_LOWER_RETURN
    ref_upper_return: float = REF_UPPER_RETURN
    # Mirrors daily causal_fair: freshness is recorded, not used by the base
    # analysis_eligible gate when primary_age_ms is None.
    freshness_gate_ms: int | None = None
    trial_transition_resolution: str = "one_second_compacted"

    def validate(self) -> None:
        if self.interval != "1s":
            raise ValueError("this extension supports only a 1s grid")
        if not (self.state_start <= self.grid_start < self.grid_end):
            raise ValueError("invalid state/grid session bounds")


DEFAULT_CONFIG = ExtensionConfig()


@dataclass(frozen=True)
class ExtensionDayResult:
    date: str
    products: int
    rows: int
    eligible_rows: int
    elapsed_seconds: float
    status: str


def load_canonical_filled_population(
    session_date: str,
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
) -> pl.DataFrame:
    """Return exact filled pairs still unexpired on ``session_date``.

    This is intentionally a set of physical filled-position keys, not a
    stability-selected symbol list.  The source date must not be later than
    the requested session, which keeps the helper safe if reused before the
    current post-horizon dates.
    """

    parsed = _parse_date(session_date)
    paths = sorted(
        Path(entry_root).glob(
            "candidate_outcomes/Date=*/candidate_outcomes.parquet"
        )
    )
    if not paths:
        raise FileNotFoundError(f"no candidate outcomes below {entry_root}")
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "contract_size",
        "end_date",
        "approximate_fill_before_nominal_stop",
        "full_fill",
        "outcome_supported",
    }
    scan = pl.scan_parquet(paths)
    _require_schema(scan.collect_schema(), required, "candidate outcomes")
    population = (
        scan.filter(
            pl.col("approximate_fill_before_nominal_stop")
            & pl.col("full_fill")
            & pl.col("outcome_supported")
            & (pl.col("Date").cast(pl.String) <= pl.lit(session_date))
            & (pl.col("end_date").cast(pl.Date) >= pl.lit(parsed))
        )
        .select(
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("contract_size").cast(pl.Float64),
            pl.col("end_date").cast(pl.Date),
        )
        .unique()
        .sort(["ValueCode", "QuoteCode"])
        .collect(engine="streaming")
    )
    if population.is_empty():
        raise RuntimeError(f"{session_date}: no unexpired canonical fills")
    _validate_exact_pairs(population, "canonical filled population")
    return population


def load_exact_session_mapping(
    session_date: str,
    population: pl.DataFrame,
    *,
    metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
    market_data_root: Path | None = None,
) -> pl.DataFrame:
    """Attach same-session references to every exact carry pair.

    Day-trade eligibility is retained as an audit field but is not a filter:
    an already-open position must remain exit-eligible even if a product is no
    longer admitted for new entries.
    """

    _validate_exact_pairs(population, "canonical filled population")
    metadata_path, metadata_date = _resolve_contract_metadata(
        Path(metadata_root), session_date
    )
    market_root = (
        Path(market_data_root)
        if market_data_root is not None
        else HFT_DATA_ROOT / "marketData"
    )
    market_path = market_root / f"{session_date}_marketData.parquet"
    if not market_path.exists():
        raise FileNotFoundError(market_path)

    contract = pl.read_parquet(metadata_path).select(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("contract_size").cast(pl.Float64).alias("daily_contract_size"),
        pl.col("decimal_locator").cast(pl.Int16),
        pl.col("end_date").cast(pl.Date).alias("daily_end_date"),
        pl.col("fut_ref_price").cast(pl.Float64),
    )
    exact = population.join(
        contract,
        on=["ValueCode", "QuoteCode"],
        how="left",
        validate="1:1",
    )
    missing_contract = exact.filter(pl.col("fut_ref_price").is_null())
    if missing_contract.height:
        raise ValueError(
            f"{session_date}: exact contracts absent from metadata: "
            f"{missing_contract.select('ValueCode', 'QuoteCode').to_dicts()}"
        )
    mismatch = exact.filter(
        ((pl.col("contract_size") - pl.col("daily_contract_size")).abs() > 1e-6)
        | (pl.col("end_date") != pl.col("daily_end_date"))
    )
    if mismatch.height:
        raise ValueError(
            f"{session_date}: canonical/daily contract mismatch: "
            f"{mismatch.select('ValueCode', 'QuoteCode').to_dicts()}"
        )

    market_scan = pl.scan_parquet(market_path)
    _require_schema(
        market_scan.collect_schema(),
        {"quote_code", "opening_ref_price", "allow_day_trade_mark"},
        "spot market data",
    )
    spot = (
        market_scan.select(
            pl.col("quote_code").cast(pl.String).alias("ValueCode"),
            pl.col("opening_ref_price").cast(pl.Float64).alias("spot_ref_price"),
            pl.col("allow_day_trade_mark")
            .cast(pl.String)
            .alias("day_trade_mark"),
        )
        .filter(pl.col("ValueCode").is_in(population["ValueCode"].to_list()))
        .collect(engine="streaming")
        .unique(subset=["ValueCode"], keep="first")
    )
    mapping = (
        exact.join(spot, on="ValueCode", how="left", validate="1:1")
        .filter(pl.col("end_date") >= pl.lit(_parse_date(session_date)))
        .with_columns(
            pl.lit(metadata_date).alias("contract_metadata_date"),
            pl.lit(metadata_date == session_date).alias("fut_ref_same_day_metadata"),
            pl.lit(
                "same_day_exact_contract_metadata"
                if metadata_date == session_date
                else "pending_first_causal_two_sided_l1_proxy"
            ).alias("fut_ref_source"),
            pl.lit(_first_grid_ns(session_date)).alias("fut_ref_available_ns"),
        )
        .select(
            "ValueCode",
            "QuoteCode",
            "contract_size",
            "decimal_locator",
            "end_date",
            "spot_ref_price",
            "fut_ref_price",
            "day_trade_mark",
            "contract_metadata_date",
            "fut_ref_same_day_metadata",
            "fut_ref_source",
            "fut_ref_available_ns",
        )
        .sort(["ValueCode", "QuoteCode"])
    )
    missing_spot = mapping.filter(
        pl.col("spot_ref_price").is_null() | (pl.col("spot_ref_price") <= 0)
    )
    if missing_spot.height:
        raise ValueError(
            f"{session_date}: exact carry spots lack opening reference: "
            f"{missing_spot.select('ValueCode').to_series().to_list()}"
        )
    if mapping.height != population.height:
        raise ValueError(f"{session_date}: exact mapping lost population rows")
    _validate_exact_pairs(mapping, "exact session mapping")
    return mapping


def refresh_stale_future_references(
    session_date: str,
    mapping: pl.DataFrame,
    future_compact: pl.DataFrame,
) -> pl.DataFrame:
    """Replace stale metadata refs with the first causal two-sided L1 midpoint.

    Structural exact-contract fields may safely come from a prior session, but
    its daily futures reference must not silently masquerade as current.  The
    fallback reference becomes available only at the effective second of the
    first valid same-session L1 snapshot; earlier grid rows fail closed.
    """

    required = {
        "QuoteCode",
        "fut_ref_same_day_metadata",
        "fut_ref_price",
        "fut_ref_source",
        "fut_ref_available_ns",
    }
    _require_schema(mapping.schema, required, "exact session mapping")
    stale = mapping.filter(~pl.col("fut_ref_same_day_metadata"))
    if stale.is_empty():
        return mapping
    stale_codes = stale["QuoteCode"].to_list()
    proxy = (
        future_compact.filter(
            pl.col("QuoteCode").is_in(stale_codes)
            & (pl.col("fut_trial_match") == 0)
            & (pl.col("fut_bid") > 0)
            & (pl.col("fut_ask") > 0)
            & (pl.col("fut_bid_lots") > 0)
            & (pl.col("fut_ask_lots") > 0)
            & (pl.col("fut_bid") <= pl.col("fut_ask"))
        )
        .sort(["QuoteCode", "effective_second_ns"])
        .group_by("QuoteCode", maintain_order=True)
        .agg(
            ((pl.col("fut_bid").first() + pl.col("fut_ask").first()) / 2)
            .cast(pl.Float64)
            .alias("_proxy_fut_ref_price"),
            pl.col("effective_second_ns")
            .first()
            .cast(pl.Int64)
            .alias("_proxy_available_ns"),
        )
    )
    missing = stale.select("QuoteCode").join(proxy, on="QuoteCode", how="anti")
    if missing.height:
        raise ValueError(
            f"{session_date}: no causal two-sided futures L1 for stale metadata "
            f"contracts: {missing['QuoteCode'].to_list()}"
        )
    return (
        mapping.join(proxy, on="QuoteCode", how="left", validate="1:1")
        .with_columns(
            pl.when(~pl.col("fut_ref_same_day_metadata"))
            .then(pl.col("_proxy_fut_ref_price"))
            .otherwise(pl.col("fut_ref_price"))
            .alias("fut_ref_price"),
            pl.when(~pl.col("fut_ref_same_day_metadata"))
            .then(pl.col("_proxy_available_ns"))
            .otherwise(pl.col("fut_ref_available_ns"))
            .cast(pl.Int64)
            .alias("fut_ref_available_ns"),
            pl.when(~pl.col("fut_ref_same_day_metadata"))
            .then(pl.lit("first_causal_formal_two_sided_l1_midpoint_proxy"))
            .otherwise(pl.col("fut_ref_source"))
            .alias("fut_ref_source"),
        )
        .drop("_proxy_fut_ref_price", "_proxy_available_ns")
    )


def compact_spot_seconds(
    source: pl.LazyFrame | pl.DataFrame,
    value_codes: Sequence[str],
    session_date: str,
    *,
    config: ExtensionConfig = DEFAULT_CONFIG,
) -> pl.DataFrame:
    """Compress raw spot rows to causal per-second L1/trial updates."""

    config.validate()
    lazy = source.lazy() if isinstance(source, pl.DataFrame) else source
    required = {
        "RecvTime",
        "TransTime",
        "ValueCode",
        "ChannelSeq",
        "PacketSeq",
        "TrialMatch",
        "BidPrice1",
        "AskPrice1",
        "BidLots1",
        "AskLots1",
        *(f"BidPrice{level}" for level in range(2, 6)),
        *(f"AskPrice{level}" for level in range(2, 6)),
    }
    schema = lazy.collect_schema()
    _require_schema(schema, required, "raw spot tape")
    recv = _utc_naive_recv_expr(schema)
    update = pl.any_horizontal(
        *[
            pl.col(f"{side}Price{level}").cast(pl.Float64) > 0
            for side in ("Bid", "Ask")
            for level in range(1, 6)
        ]
    )
    prepared = (
        lazy.filter(
            pl.col("ValueCode").cast(pl.String).is_in(list(value_codes))
            & _local_session_filter(session_date, config)
        )
        .select(
            pl.col("ValueCode").cast(pl.String),
            recv.alias("_recv_time"),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("_sequence"),
            pl.col("PacketSeq").cast(pl.UInt64).alias("_packet_sequence"),
            pl.col("TrialMatch").fill_null(0).cast(pl.Int16).alias("_trial"),
            update.alias("_l1_update"),
            _positive_price("BidPrice1").alias("_spot_bid"),
            _positive_price("AskPrice1").alias("_spot_ask"),
            _positive_lots("BidPrice1", "BidLots1").alias("_spot_bid_lots"),
            _positive_lots("AskPrice1", "AskLots1").alias("_spot_ask_lots"),
        )
        .with_columns(
            pl.col("_recv_time").cast(pl.Int64).alias("_recv_time_ns")
        )
        .with_columns(_ceil_second_expr("_recv_time_ns"))
    )
    trial = _last_struct(
        {
            "spot_trial_match": "_trial",
            "spot_trial_recv_time_ns": "_recv_time_ns",
            "spot_trial_sequence": "_sequence",
            "spot_trial_packet_sequence": "_packet_sequence",
        }
    ).alias("_trial_state")
    l1 = _last_struct(
        {
            "spot_bid": "_spot_bid",
            "spot_ask": "_spot_ask",
            "spot_bid_lots": "_spot_bid_lots",
            "spot_ask_lots": "_spot_ask_lots",
            "spot_l1_recv_time_ns": "_recv_time_ns",
            "spot_l1_sequence": "_sequence",
            "spot_l1_packet_sequence": "_packet_sequence",
        },
        predicate=pl.col("_l1_update"),
    ).alias("_l1_state")
    return (
        prepared.group_by(["ValueCode", "effective_second_ns"])
        .agg(trial, l1)
        .collect(engine="streaming")
        .unnest(["_trial_state", "_l1_state"])
        .sort(["ValueCode", "effective_second_ns"])
    )


def compact_future_seconds(
    source: pl.LazyFrame | pl.DataFrame,
    quote_codes: Sequence[str],
    session_date: str,
    *,
    config: ExtensionConfig = DEFAULT_CONFIG,
) -> pl.DataFrame:
    """Compress raw futures rows while keeping L1 and Best state separate."""

    config.validate()
    lazy = source.lazy() if isinstance(source, pl.DataFrame) else source
    required = {
        "RecvTime",
        "TransTime",
        "QuoteCode",
        "ChannelSeq",
        "PacketSeq",
        "TrialMatch",
        "DecimalLocator",
        "BidPrice1",
        "AskPrice1",
        "BidLots1",
        "AskLots1",
        "BestBidPrice",
        "BestAskPrice",
        "BestBidLots",
        "BestAskLots",
        *(f"BidPrice{level}" for level in range(2, 6)),
        *(f"AskPrice{level}" for level in range(2, 6)),
    }
    schema = lazy.collect_schema()
    _require_schema(schema, required, "raw futures tape")
    recv = _utc_naive_recv_expr(schema)
    l1_update = pl.any_horizontal(
        *[
            pl.col(f"{side}Price{level}").cast(pl.Float64) > 0
            for side in ("Bid", "Ask")
            for level in range(1, 6)
        ]
    )
    best_update = pl.any_horizontal(
        pl.col("BestBidPrice").cast(pl.Float64) > 0,
        pl.col("BestAskPrice").cast(pl.Float64) > 0,
    )
    divisor = pl.lit(10.0).pow(pl.col("DecimalLocator").cast(pl.Float64))

    def price(column: str) -> pl.Expr:
        decoded = pl.col(column).cast(pl.Float64) / divisor
        return pl.when(decoded > 0).then(decoded).otherwise(None)

    def lots(price_column: str, lots_column: str) -> pl.Expr:
        return (
            pl.when(price(price_column).is_not_null())
            .then(pl.col(lots_column).fill_null(0).cast(pl.Int64))
            .otherwise(None)
        )

    prepared = (
        lazy.filter(
            pl.col("QuoteCode").cast(pl.String).is_in(list(quote_codes))
            & _local_session_filter(session_date, config)
        )
        .select(
            pl.col("QuoteCode").cast(pl.String),
            recv.alias("_recv_time"),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("_sequence"),
            pl.col("PacketSeq").cast(pl.UInt64).alias("_packet_sequence"),
            pl.col("TrialMatch").fill_null(0).cast(pl.Int16).alias("_trial"),
            pl.col("DecimalLocator").cast(pl.Int16).alias("_decimal_locator"),
            l1_update.alias("_l1_update"),
            best_update.alias("_best_update"),
            price("BidPrice1").alias("_fut_bid"),
            price("AskPrice1").alias("_fut_ask"),
            lots("BidPrice1", "BidLots1").alias("_fut_bid_lots"),
            lots("AskPrice1", "AskLots1").alias("_fut_ask_lots"),
            price("BestBidPrice").alias("_best_bid"),
            price("BestAskPrice").alias("_best_ask"),
            lots("BestBidPrice", "BestBidLots").alias("_best_bid_lots"),
            lots("BestAskPrice", "BestAskLots").alias("_best_ask_lots"),
        )
        .with_columns(
            pl.col("_recv_time").cast(pl.Int64).alias("_recv_time_ns")
        )
        .with_columns(_ceil_second_expr("_recv_time_ns"))
    )
    trial = _last_struct(
        {
            "fut_trial_match": "_trial",
            "fut_trial_recv_time_ns": "_recv_time_ns",
            "fut_trial_sequence": "_sequence",
            "fut_trial_packet_sequence": "_packet_sequence",
        }
    ).alias("_trial_state")
    l1 = _last_struct(
        {
            "fut_bid": "_fut_bid",
            "fut_ask": "_fut_ask",
            "fut_bid_lots": "_fut_bid_lots",
            "fut_ask_lots": "_fut_ask_lots",
            "fut_l1_recv_time_ns": "_recv_time_ns",
            "fut_l1_sequence": "_sequence",
            "fut_l1_packet_sequence": "_packet_sequence",
            "fut_l1_decimal_locator": "_decimal_locator",
        },
        predicate=pl.col("_l1_update"),
    ).alias("_l1_state")
    best = _last_struct(
        {
            "fut_best_bid": "_best_bid",
            "fut_best_ask": "_best_ask",
            "fut_best_bid_lots": "_best_bid_lots",
            "fut_best_ask_lots": "_best_ask_lots",
            "fut_best_recv_time_ns": "_recv_time_ns",
            "fut_best_sequence": "_sequence",
            "fut_best_packet_sequence": "_packet_sequence",
            "fut_best_decimal_locator": "_decimal_locator",
        },
        predicate=pl.col("_best_update"),
    ).alias("_best_state")
    return (
        prepared.group_by(["QuoteCode", "effective_second_ns"])
        .agg(trial, l1, best)
        .collect(engine="streaming")
        .unnest(["_trial_state", "_l1_state", "_best_state"])
        .sort(["QuoteCode", "effective_second_ns"])
    )


def assemble_one_second_grid(
    session_date: str,
    mapping: pl.DataFrame,
    spot_compact: pl.DataFrame,
    future_compact: pl.DataFrame,
    *,
    config: ExtensionConfig = DEFAULT_CONFIG,
    timestamps: pl.Series | None = None,
) -> pl.DataFrame:
    """As-of compact component updates onto the exact one-second grid."""

    config.validate()
    _validate_exact_pairs(mapping, "exact session mapping")
    if timestamps is None:
        timestamps = _grid_timestamps(session_date, config)
    timestamps = timestamps.cast(pl.Datetime("ns"))
    grid = (
        mapping.select(
            "ValueCode",
            "QuoteCode",
            pl.col("contract_size").cast(pl.Float64),
            pl.col("decimal_locator").cast(pl.Int16),
            pl.col("end_date").cast(pl.Date),
            pl.col("spot_ref_price").cast(pl.Float64),
            pl.col("fut_ref_price").cast(pl.Float64),
            pl.col("day_trade_mark").cast(pl.String),
            pl.col("contract_metadata_date").cast(pl.String),
            pl.col("fut_ref_same_day_metadata").cast(pl.Boolean),
            pl.col("fut_ref_source").cast(pl.String),
            pl.col("fut_ref_available_ns").cast(pl.Int64),
        )
        .with_columns(pl.lit(session_date).alias("Date"))
        .join(pl.DataFrame({"timestamp": timestamps}), how="cross")
        .with_columns(pl.col("timestamp").cast(pl.Int64).alias("grid_time_ns"))
    )

    spot_l1 = _component_frame(
        spot_compact,
        "ValueCode",
        "spot_l1_recv_time_ns",
        "spot_l1_effective_ns",
        (
            "spot_bid",
            "spot_ask",
            "spot_bid_lots",
            "spot_ask_lots",
            "spot_l1_recv_time_ns",
            "spot_l1_sequence",
            "spot_l1_packet_sequence",
        ),
    )
    spot_trial = _trial_state_frame(spot_compact, "ValueCode", "spot")
    spot_transition = _trial_transition_frame(spot_compact, "ValueCode", "spot")
    future_l1 = _component_frame(
        future_compact,
        "QuoteCode",
        "fut_l1_recv_time_ns",
        "fut_l1_effective_ns",
        (
            "fut_bid",
            "fut_ask",
            "fut_bid_lots",
            "fut_ask_lots",
            "fut_l1_recv_time_ns",
            "fut_l1_sequence",
            "fut_l1_packet_sequence",
            "fut_l1_decimal_locator",
        ),
    )
    future_best = _component_frame(
        future_compact,
        "QuoteCode",
        "fut_best_recv_time_ns",
        "fut_best_effective_ns",
        (
            "fut_best_bid",
            "fut_best_ask",
            "fut_best_bid_lots",
            "fut_best_ask_lots",
            "fut_best_recv_time_ns",
            "fut_best_sequence",
            "fut_best_packet_sequence",
            "fut_best_decimal_locator",
        ),
    )
    future_trial = _trial_state_frame(future_compact, "QuoteCode", "fut")
    future_transition = _trial_transition_frame(
        future_compact, "QuoteCode", "fut"
    )
    for right, by, right_on in (
        (spot_l1, "ValueCode", "spot_l1_effective_ns"),
        (spot_trial, "ValueCode", "spot_trial_effective_ns"),
        (spot_transition, "ValueCode", "spot_transition_effective_ns"),
        (future_l1, "QuoteCode", "fut_l1_effective_ns"),
        (future_best, "QuoteCode", "fut_best_effective_ns"),
        (future_trial, "QuoteCode", "fut_trial_effective_ns"),
        (future_transition, "QuoteCode", "fut_transition_effective_ns"),
    ):
        grid = _join_asof_component(
            grid,
            right,
            by=by,
            left_on="grid_time_ns",
            right_on=right_on,
        )

    grid = grid.with_columns(
        _exec_price("bid"),
        _exec_price("ask"),
    ).with_columns(
        _exec_lots("bid"),
        _exec_lots("ask"),
        pl.max_horizontal(
            "fut_l1_recv_time_ns", "fut_best_recv_time_ns"
        ).alias("fut_book_recv_time_ns"),
    )
    grid = grid.with_columns(
        _latest_book_sequence().alias("fut_book_sequence"),
        ((pl.col("grid_time_ns") - pl.col("spot_l1_recv_time_ns")) / 1_000_000)
        .alias("spot_age_ms"),
        ((pl.col("grid_time_ns") - pl.col("fut_l1_recv_time_ns")) / 1_000_000)
        .alias("fut_l1_age_ms"),
        ((pl.col("grid_time_ns") - pl.col("fut_best_recv_time_ns")) / 1_000_000)
        .alias("fut_best_age_ms"),
        ((pl.col("grid_time_ns") - pl.col("fut_book_recv_time_ns")) / 1_000_000)
        .alias("fut_age_ms"),
    )
    grid = grid.with_columns(
        _formal_after_transition("spot", "spot_l1").alias("spot_formal"),
        _formal_after_transition("fut", "fut_book").alias("fut_formal"),
        (
            (pl.col("spot_bid") > 0)
            & (pl.col("spot_ask") > 0)
            & (pl.col("spot_bid_lots") > 0)
            & (pl.col("spot_ask_lots") > 0)
            & (pl.col("spot_bid") <= pl.col("spot_ask"))
        )
        .fill_null(False)
        .alias("spot_book_ok"),
        (
            (pl.col("fut_bid") > 0)
            & (pl.col("fut_ask") > 0)
            & (pl.col("fut_bid_lots") > 0)
            & (pl.col("fut_ask_lots") > 0)
            & (pl.col("fut_bid") <= pl.col("fut_ask"))
        )
        .fill_null(False)
        .alias("fut_book_ok"),
        (
            (pl.col("fut_exec_bid") > 0)
            & (pl.col("fut_exec_ask") > 0)
            & (pl.col("fut_exec_bid_lots") > 0)
            & (pl.col("fut_exec_ask_lots") > 0)
            & (pl.col("fut_exec_bid") <= pl.col("fut_exec_ask"))
        )
        .fill_null(False)
        .alias("fut_exec_book_ok"),
        _ref_ok("spot", ("bid", "ask"), config).alias("spot_ref_ok"),
        _ref_ok(
            "fut", ("bid", "ask", "exec_bid", "exec_ask"), config
        ).alias("fut_ref_ok"),
        (pl.col("fut_l1_decimal_locator") == pl.col("decimal_locator"))
        .fill_null(False)
        .alias("fut_decimal_locator_ok"),
    )
    eligible = (
        pl.col("spot_formal")
        & pl.col("fut_formal")
        & pl.col("spot_book_ok")
        & pl.col("fut_book_ok")
        & pl.col("fut_exec_book_ok")
        & pl.col("spot_ref_ok")
        & pl.col("fut_ref_ok")
        & (pl.col("grid_time_ns") >= pl.col("fut_ref_available_ns"))
        & pl.col("fut_decimal_locator_ok")
    )
    if config.freshness_gate_ms is not None:
        eligible = (
            eligible
            & (pl.col("spot_age_ms") <= config.freshness_gate_ms)
            & (pl.col("fut_age_ms") <= config.freshness_gate_ms)
        )
    grid = grid.with_columns(eligible.fill_null(False).alias("analysis_eligible"))
    grid = grid.with_columns(
        pl.when(pl.col("analysis_eligible"))
        .then((pl.col("fut_exec_ask") / pl.col("spot_bid") - 1) * 10_000)
        .otherwise(None)
        .alias("basis_buy_taker_bp")
    )
    preferred = [
        *REQUIRED_GRID_COLUMNS,
        "spot_ref_price",
        "fut_ref_price",
        "end_date",
        "day_trade_mark",
        "contract_metadata_date",
        "fut_ref_same_day_metadata",
        "fut_ref_source",
        "fut_ref_available_ns",
        "spot_ask",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "fut_bid_lots",
        "fut_ask_lots",
        "fut_exec_bid_lots",
        "spot_age_ms",
        "fut_l1_age_ms",
        "fut_best_age_ms",
        "fut_age_ms",
        "spot_trial_match",
        "fut_trial_match",
        "spot_formal",
        "fut_formal",
        "spot_book_ok",
        "fut_book_ok",
        "fut_exec_book_ok",
        "spot_ref_ok",
        "fut_ref_ok",
        "fut_decimal_locator_ok",
        "spot_l1_recv_time_ns",
        "fut_l1_recv_time_ns",
        "fut_best_recv_time_ns",
        "fut_book_recv_time_ns",
    ]
    return grid.select(preferred).sort(["ValueCode", "timestamp"])


def build_extension_day(
    session_date: str,
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    spot_path: Path | None = None,
    future_path: Path | None = None,
    market_data_root: Path | None = None,
    config: ExtensionConfig = DEFAULT_CONFIG,
    resume: bool = True,
) -> ExtensionDayResult:
    """Build and atomically publish one extension partition."""

    config.validate()
    partition = Path(output_root) / f"Date={session_date}"
    marker = partition / "complete.json"
    if resume and marker.exists():
        payload = validate_extension_partition(marker)
        return ExtensionDayResult(
            date=session_date,
            products=int(payload["products"]),
            rows=int(payload["rows"]),
            eligible_rows=int(payload["eligible_rows"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            status="skipped_complete",
        )

    population = load_canonical_filled_population(
        session_date, entry_root=entry_root
    )
    mapping = load_exact_session_mapping(
        session_date,
        population,
        metadata_root=metadata_root,
        market_data_root=market_data_root,
    )
    spot_path = (
        Path(spot_path)
        if spot_path is not None
        else HFT_DATA_ROOT / "tickData" / f"{session_date}_StockTick.parquet"
    )
    future_path = (
        Path(future_path)
        if future_path is not None
        else futures_raw_path(session_date)
    )
    for path in (spot_path, future_path):
        if not path.exists():
            raise FileNotFoundError(path)

    start = perf_counter()
    spot_compact = compact_spot_seconds(
        pl.scan_parquet(spot_path),
        mapping["ValueCode"].to_list(),
        session_date,
        config=config,
    )
    future_compact = compact_future_seconds(
        pl.scan_parquet(future_path),
        mapping["QuoteCode"].to_list(),
        session_date,
        config=config,
    )
    mapping = refresh_stale_future_references(
        session_date, mapping, future_compact
    )
    grid = assemble_one_second_grid(
        session_date,
        mapping,
        spot_compact,
        future_compact,
        config=config,
    )
    elapsed = perf_counter() - start
    audit = summarize_grid_audit(
        session_date,
        mapping,
        spot_compact,
        future_compact,
        grid,
        elapsed_seconds=elapsed,
    )
    return publish_extension_partition(
        session_date,
        mapping,
        grid,
        audit,
        output_root=output_root,
        config=config,
        source_paths=(spot_path, future_path),
        elapsed_seconds=elapsed,
    )


def summarize_grid_audit(
    session_date: str,
    mapping: pl.DataFrame,
    spot_compact: pl.DataFrame,
    future_compact: pl.DataFrame,
    grid: pl.DataFrame,
    *,
    elapsed_seconds: float,
) -> pl.DataFrame:
    """Create one-row coverage and exclusion audit for a built day."""

    row: dict[str, object] = {
        "Date": session_date,
        "products": mapping.height,
        "grid_rows": grid.height,
        "spot_compact_seconds": spot_compact.height,
        "future_compact_seconds": future_compact.height,
        "analysis_eligible_rows": int(grid["analysis_eligible"].sum()),
        "rows_missing_spot_book": int((~grid["spot_book_ok"]).sum()),
        "rows_missing_future_l1_book": int((~grid["fut_book_ok"]).sum()),
        "rows_missing_future_exec_book": int((~grid["fut_exec_book_ok"]).sum()),
        "rows_blocked_spot_trial": int((~grid["spot_formal"]).sum()),
        "rows_blocked_future_trial": int((~grid["fut_formal"]).sum()),
        "rows_blocked_spot_ref": int((~grid["spot_ref_ok"]).sum()),
        "rows_blocked_future_ref": int((~grid["fut_ref_ok"]).sum()),
        "rows_decimal_locator_mismatch": int(
            (~grid["fut_decimal_locator_ok"]).sum()
        ),
        "spot_age_p50_ms": _quantile(grid, "spot_age_ms", 0.50),
        "spot_age_p95_ms": _quantile(grid, "spot_age_ms", 0.95),
        "future_age_p50_ms": _quantile(grid, "fut_age_ms", 0.50),
        "future_age_p95_ms": _quantile(grid, "fut_age_ms", 0.95),
        "day_trade_mark_not_xy_products": mapping.filter(
            ~pl.col("day_trade_mark").str.to_uppercase().is_in(["X", "Y"])
        ).height,
        "future_ref_proxy_products": mapping.filter(
            ~pl.col("fut_ref_same_day_metadata")
        ).height,
        "elapsed_seconds": elapsed_seconds,
    }
    return pl.DataFrame([row])


def publish_extension_partition(
    session_date: str,
    mapping: pl.DataFrame,
    grid: pl.DataFrame,
    audit: pl.DataFrame,
    *,
    output_root: Path,
    config: ExtensionConfig,
    source_paths: Sequence[Path],
    elapsed_seconds: float,
) -> ExtensionDayResult:
    """Write artifacts via temporary files and publish the marker last."""

    partition = Path(output_root) / f"Date={session_date}"
    partition.mkdir(parents=True, exist_ok=True)
    marker = partition / "complete.json"
    if marker.exists():
        marker.unlink()
    frames = {
        "causal_fair.parquet": grid,
        "mapping.parquet": mapping.with_columns(pl.lit(session_date).alias("Date")),
        "audit.parquet": audit,
    }
    for name, frame in frames.items():
        temporary = partition / f".{name}.{os.getpid()}.tmp"
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(partition / name)

    artifacts = {
        name: _artifact_identity(partition / name) for name in frames
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "date": session_date,
        "products": mapping.height,
        "rows": grid.height,
        "eligible_rows": int(grid["analysis_eligible"].sum()),
        "elapsed_seconds": elapsed_seconds,
        "required_vertical_concat_columns": list(REQUIRED_GRID_COLUMNS),
        "population_source": "canonical_full_fills_exact_pairs_unexpired_on_session",
        "fixed45_universe_used": False,
        "future_l1_best_persist_independently": True,
        "same_price_lots_rule": "max_not_sum",
        "zero_book_trade_rows_erase_state": False,
        "analysis_eligible_semantics": (
            "formal_after_compacted_trial_transition_and_valid_two_sided_books_"
            "and_strict_reference_gates; no_age_gate"
        ),
        "trial_transition_limitation": (
            "trial state is compacted to its last raw state per effective 1s; "
            "a sub-second round trip ending in the prior state is not observable"
        ),
        "quantity_limitation": (
            "displayed L1/Best lots only; no queue position, partial-fill, impact, "
            "or cancel/ACK simulation"
        ),
        "config": _config_payload(config),
        "sources": [_source_identity(path) for path in source_paths],
        "artifacts": artifacts,
    }
    temporary_marker = partition / f".complete.json.{os.getpid()}.tmp"
    temporary_marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(marker)
    validate_extension_partition(marker)
    return ExtensionDayResult(
        date=session_date,
        products=mapping.height,
        rows=grid.height,
        eligible_rows=int(grid["analysis_eligible"].sum()),
        elapsed_seconds=elapsed_seconds,
        status="built",
    )


def validate_extension_partition(marker: Path) -> dict[str, object]:
    payload = json.loads(Path(marker).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported extension schema: {marker}")
    expected_date = marker.parent.name.removeprefix("Date=")
    if payload.get("date") != expected_date:
        raise ValueError(f"extension date mismatch: {marker}")
    for name in ("causal_fair.parquet", "mapping.parquet", "audit.parquet"):
        path = marker.parent / name
        if not path.exists():
            raise FileNotFoundError(path)
        if payload.get("artifacts", {}).get(name) != _artifact_identity(path):
            raise ValueError(f"extension artifact identity mismatch: {path}")
    schema = pl.read_parquet_schema(marker.parent / "causal_fair.parquet")
    _require_schema(schema, set(REQUIRED_GRID_COLUMNS), "extension causal grid")
    if _parquet_rows(marker.parent / "causal_fair.parquet") != int(payload["rows"]):
        raise ValueError(f"extension grid row mismatch: {marker}")
    return payload


def build_extension_bundle(
    dates: Iterable[str] = DEFAULT_DATES,
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    market_data_root: Path | None = None,
    config: ExtensionConfig = DEFAULT_CONFIG,
    resume: bool = True,
) -> list[ExtensionDayResult]:
    """Build requested days and publish a root marker only after all validate."""

    selected = tuple(str(value) for value in dates)
    if not selected:
        raise ValueError("at least one extension date is required")
    results = [
        build_extension_day(
            day,
            entry_root=entry_root,
            metadata_root=metadata_root,
            output_root=output_root,
            market_data_root=market_data_root,
            config=config,
            resume=resume,
        )
        for day in selected
    ]
    root = Path(output_root)
    summary = pl.DataFrame([asdict(result) for result in results])
    temporary_summary = root / f".daily_summary.csv.{os.getpid()}.tmp"
    summary.write_csv(temporary_summary)
    temporary_summary.replace(root / "daily_summary.csv")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dates": list(selected),
        "partitions": [
            _source_identity(root / f"Date={day}" / "complete.json")
            for day in selected
        ],
        "rows": sum(result.rows for result in results),
        "eligible_rows": sum(result.eligible_rows for result in results),
        "fixed45_universe_used": False,
        "complete": True,
    }
    temporary = root / f".complete.json.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(root / "complete.json")
    return results


def _component_frame(
    compact: pl.DataFrame,
    key: str,
    cursor: str,
    effective_alias: str,
    columns: Sequence[str],
) -> pl.DataFrame:
    return (
        compact.filter(pl.col(cursor).is_not_null())
        .select(
            key,
            pl.col("effective_second_ns").alias(effective_alias),
            *columns,
        )
        .sort([key, effective_alias])
    )


def _trial_state_frame(
    compact: pl.DataFrame, key: str, prefix: str
) -> pl.DataFrame:
    effective = f"{prefix}_trial_effective_ns"
    return compact.select(
        key,
        pl.col("effective_second_ns").alias(effective),
        f"{prefix}_trial_match",
    ).sort([key, effective])


def _trial_transition_frame(
    compact: pl.DataFrame, key: str, prefix: str
) -> pl.DataFrame:
    state = f"{prefix}_trial_match"
    recv = f"{prefix}_trial_recv_time_ns"
    sequence = f"{prefix}_trial_sequence"
    packet = f"{prefix}_trial_packet_sequence"
    effective = f"{prefix}_transition_effective_ns"
    ordered = compact.select(
        key, "effective_second_ns", state, recv, sequence, packet
    ).sort([key, "effective_second_ns"])
    return (
        ordered.with_columns(pl.col(state).shift(1).over(key).alias("_previous"))
        .filter(pl.col("_previous").is_null() | (pl.col(state) != pl.col("_previous")))
        .select(
            key,
            pl.col("effective_second_ns").alias(effective),
            pl.col(recv).alias(f"{prefix}_transition_recv_time_ns"),
            pl.col(sequence).alias(f"{prefix}_transition_sequence"),
            pl.col(packet).alias(f"{prefix}_transition_packet_sequence"),
        )
        .sort([key, effective])
    )


def _join_asof_component(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    by: str,
    left_on: str,
    right_on: str,
) -> pl.DataFrame:
    return left.sort([by, left_on]).join_asof(
        right.sort([by, right_on]),
        left_on=left_on,
        right_on=right_on,
        by=by,
        strategy="backward",
        check_sortedness=False,
    )


def _last_struct(
    aliases: dict[str, str], predicate: pl.Expr | None = None
) -> pl.Expr:
    value = pl.struct(
        [pl.col(source).alias(target) for target, source in aliases.items()]
    ).sort_by(["_recv_time_ns", "_sequence", "_packet_sequence"])
    if predicate is not None:
        value = value.filter(predicate)
    return value.last()


def _exec_price(side: str) -> pl.Expr:
    l1 = pl.col(f"fut_{side}")
    best = pl.col(f"fut_best_{side}")
    if side == "bid":
        return pl.max_horizontal(l1, best).alias("fut_exec_bid")
    return pl.min_horizontal(l1, best).alias("fut_exec_ask")


def _exec_lots(side: str) -> pl.Expr:
    price = pl.col(f"fut_exec_{side}")
    l1_price = pl.col(f"fut_{side}")
    best_price = pl.col(f"fut_best_{side}")
    l1_lots = pl.col(f"fut_{side}_lots")
    best_lots = pl.col(f"fut_best_{side}_lots")
    return (
        pl.when(price.is_null())
        .then(None)
        .when((l1_price == price) & (best_price == price))
        .then(pl.max_horizontal(l1_lots, best_lots))
        .when(l1_price == price)
        .then(l1_lots)
        .otherwise(best_lots)
        .cast(pl.Int64)
        .alias(f"fut_exec_{side}_lots")
    )


def _latest_book_sequence() -> pl.Expr:
    l1_time = pl.col("fut_l1_recv_time_ns")
    best_time = pl.col("fut_best_recv_time_ns")
    return (
        pl.when(best_time.is_null() | (l1_time > best_time))
        .then(pl.col("fut_l1_sequence"))
        .when(l1_time.is_null() | (best_time > l1_time))
        .then(pl.col("fut_best_sequence"))
        .otherwise(
            pl.max_horizontal("fut_l1_sequence", "fut_best_sequence")
        )
        .cast(pl.UInt64)
    )


def _formal_after_transition(prefix: str, book_prefix: str) -> pl.Expr:
    book_time = pl.col(f"{book_prefix}_recv_time_ns")
    book_sequence = pl.col(f"{book_prefix}_sequence")
    transition_time = pl.col(f"{prefix}_transition_recv_time_ns")
    transition_sequence = pl.col(f"{prefix}_transition_sequence")
    book_after = (book_time > transition_time) | (
        (book_time == transition_time) & (book_sequence >= transition_sequence)
    )
    return ((pl.col(f"{prefix}_trial_match") == 0) & book_after).fill_null(False)


def _ref_ok(
    prefix: str,
    suffixes: Sequence[str],
    config: ExtensionConfig,
) -> pl.Expr:
    reference = pl.col(f"{prefix}_ref_price")
    epsilon = reference.abs() * REF_COMPARISON_EPS_RATIO
    lower = reference * (1 + config.ref_lower_return) + epsilon
    upper = reference * (1 + config.ref_upper_return) - epsilon
    result = reference.is_not_null() & (reference > 0)
    for suffix in suffixes:
        result = result & (pl.col(f"{prefix}_{suffix}") > lower) & (
            pl.col(f"{prefix}_{suffix}") < upper
        )
    return result.fill_null(False)


def _positive_price(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64)
    return pl.when(value > 0).then(value).otherwise(None)


def _positive_lots(price_column: str, lots_column: str) -> pl.Expr:
    return (
        pl.when(pl.col(price_column).cast(pl.Float64) > 0)
        .then(pl.col(lots_column).fill_null(0).cast(pl.Int64))
        .otherwise(None)
    )


def _ceil_second_expr(column: str) -> pl.Expr:
    return (
        ((pl.col(column) + NS_PER_SECOND - 1) // NS_PER_SECOND) * NS_PER_SECOND
    ).cast(pl.Int64).alias("effective_second_ns")


def _utc_naive_recv_expr(schema: pl.Schema) -> pl.Expr:
    dtype = schema.get("RecvTime")
    if not isinstance(dtype, pl.Datetime):
        raise TypeError("RecvTime must be Datetime")
    value = pl.col("RecvTime")
    if dtype.time_zone is not None:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return value.cast(pl.Datetime("ns"))


def _local_session_filter(
    session_date: str, config: ExtensionConfig
) -> pl.Expr:
    parsed = _parse_date(session_date)
    local_time = pl.col("TransTime").dt.time()
    return (
        (pl.col("TransTime").dt.date() == pl.lit(parsed))
        & (local_time >= pl.lit(config.state_start))
        & (local_time < pl.lit(config.grid_end))
    )


def _grid_timestamps(session_date: str, config: ExtensionConfig) -> pl.Series:
    parsed = _parse_date(session_date)
    local_start = datetime.combine(parsed, config.grid_start)
    local_end = datetime.combine(parsed, config.grid_end)
    utc_start = local_start - timedelta(hours=8)
    utc_end = local_end - timedelta(hours=8)
    return pl.datetime_range(
        utc_start,
        utc_end,
        interval=config.interval,
        closed="left",
        time_unit="ns",
        eager=True,
    )


def _validate_exact_pairs(frame: pl.DataFrame, source: str) -> None:
    _require_schema(
        frame.schema,
        {"ValueCode", "QuoteCode", "contract_size"},
        source,
    )
    for key in ("ValueCode", "QuoteCode"):
        duplicate = frame.group_by(key).len().filter(pl.col("len") != 1)
        if duplicate.height:
            raise ValueError(f"{source} is not one-to-one on {key}")
    invalid_size = frame.filter(
        pl.col("contract_size").is_null() | (pl.col("contract_size") <= 0)
    )
    if invalid_size.height:
        raise ValueError(f"{source} contains invalid contract size")


def _require_schema(
    schema: pl.Schema, required: set[str], source: str
) -> None:
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_date(value: str) -> date_type:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid YYYYMMDD date: {value!r}")
    return date_type.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")


def _resolve_contract_metadata(
    metadata_root: Path, session_date: str
) -> tuple[Path, str]:
    exact = Path(metadata_root) / f"{session_date}_contracts.parquet"
    if exact.exists():
        return exact, session_date
    prior: list[tuple[str, Path]] = []
    for path in Path(metadata_root).glob("*_contracts.parquet"):
        candidate = path.name.removesuffix("_contracts.parquet")
        if len(candidate) == 8 and candidate.isdigit() and candidate < session_date:
            prior.append((candidate, path))
    if not prior:
        raise FileNotFoundError(
            f"{exact} and no prior exact-contract metadata fallback exists"
        )
    selected_date, selected_path = max(prior, key=lambda item: item[0])
    return selected_path, selected_date


def _first_grid_ns(session_date: str) -> int:
    return int(_grid_timestamps(session_date, DEFAULT_CONFIG).cast(pl.Int64)[0])


def _quantile(frame: pl.DataFrame, column: str, value: float) -> float | None:
    result = frame.select(pl.col(column).drop_nulls().quantile(value)).item()
    return float(result) if result is not None else None


def _config_payload(config: ExtensionConfig) -> dict[str, object]:
    return {
        "state_start": config.state_start.isoformat(),
        "grid_start": config.grid_start.isoformat(),
        "grid_end": config.grid_end.isoformat(),
        "interval": config.interval,
        "ref_lower_return": config.ref_lower_return,
        "ref_upper_return": config.ref_upper_return,
        "freshness_gate_ms": config.freshness_gate_ms,
        "trial_transition_resolution": config.trial_transition_resolution,
    }


def _source_identity(path: Path) -> dict[str, object]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _artifact_identity(path: Path) -> dict[str, object]:
    schema = pl.read_parquet_schema(path)
    schema_json = json.dumps(
        [(name, str(dtype)) for name, dtype in schema.items()],
        separators=(",", ":"),
    )
    return {
        "rows": _parquet_rows(path),
        "bytes": path.stat().st_size,
        "schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
    }


def _parquet_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument(
        "--metadata-root", type=Path, default=DEFAULT_CONTRACT_METADATA_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = build_extension_bundle(
        args.dates,
        entry_root=args.entry_root,
        metadata_root=args.metadata_root,
        output_root=args.output_root,
        resume=not args.no_resume,
    )
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
