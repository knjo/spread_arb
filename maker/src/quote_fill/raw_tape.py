"""Raw spot/futures tape normalization for maker-fill replay.

The two feeds use different receive-time and price encodings.  This module
puts them on one causal schema while retaining every raw event.  In
particular, futures trade messages commonly carry a zero L1--L5 payload; such
rows remain in the state frame and inherit the last *complete* book snapshot.
Zeros inside a genuine book snapshot still clear that level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Literal

import polars as pl

from ..common.paths import futures_raw_path, spot_tick_path


SESSION_START = time(9, 5)
SESSION_END = time(13, 20)
Market = Literal["spot", "future"]

_RAW_COMMON = {
    "RecvTime",
    "TransTime",
    "QuoteCode",
    "PacketSeq",
    "ChannelSeq",
    "TrialMatch",
    "TotalFillLots",
    "FillPrice",
    "FillLots",
    *(f"BidPrice{level}" for level in range(1, 6)),
    *(f"BidLots{level}" for level in range(1, 6)),
    *(f"AskPrice{level}" for level in range(1, 6)),
    *(f"AskLots{level}" for level in range(1, 6)),
    "BestBidPrice",
    "BestBidLots",
    "BestAskPrice",
    "BestAskLots",
}


@dataclass(frozen=True)
class RawTapeDay:
    """One selected date of exact mapped spot/futures raw tape."""

    date: str
    mapping: pl.DataFrame
    spot_states: pl.DataFrame
    future_states: pl.DataFrame
    spot_trades: pl.DataFrame
    future_trades: pl.DataFrame
    audit: pl.DataFrame


def normalize_spot_tape(
    raw: pl.DataFrame,
    mapping: pl.DataFrame,
    date: str,
    *,
    session_start: time = SESSION_START,
    session_end: time = SESSION_END,
) -> pl.DataFrame:
    """Normalize selected spot rows; spot executable prices are L1."""

    return _normalize_tape(
        raw,
        mapping,
        date,
        market="spot",
        session_start=session_start,
        session_end=session_end,
    )


def normalize_future_tape(
    raw: pl.DataFrame,
    mapping: pl.DataFrame,
    date: str,
    *,
    session_start: time = SESSION_START,
    session_end: time = SESSION_END,
) -> pl.DataFrame:
    """Normalize integer futures prices and derive executable L1/Best state."""

    return _normalize_tape(
        raw,
        mapping,
        date,
        market="future",
        session_start=session_start,
        session_end=session_end,
    )


def extract_trade_tape(states: pl.DataFrame) -> pl.DataFrame:
    """Project positive incremental fills without discarding their state."""

    required = {
        "Date",
        "market",
        "ValueCode",
        "QuoteCode",
        "instrument_code",
        "recv_time",
        "recv_time_ns",
        "trans_time",
        "sequence",
        "packet_sequence",
        "trial_match",
        "ref_price",
        "contract_size",
        "fill_price",
        "fill_lots",
        "raw_has_book",
        "book_recv_time",
        "book_recv_time_ns",
        "book_sequence",
    }
    _require_columns(states, required, "normalized state")
    return (
        states.filter(
            (pl.col("fill_lots") > 0) & pl.col("fill_price").is_not_null()
        )
        .select(
            "Date",
            "market",
            "ValueCode",
            "QuoteCode",
            "instrument_code",
            "recv_time",
            "recv_time_ns",
            "trans_time",
            "sequence",
            "packet_sequence",
            "trial_match",
            "ref_price",
            "contract_size",
            pl.col("fill_price").alias("trade_price"),
            pl.col("fill_lots").alias("trade_lots"),
            "raw_has_book",
            "book_recv_time",
            "book_recv_time_ns",
            "book_sequence",
        )
        .sort(["ValueCode", "recv_time", "sequence", "packet_sequence"])
    )


def load_raw_tape_day(
    date: str,
    mapping: pl.DataFrame,
    *,
    spot_path: Path | None = None,
    future_path: Path | None = None,
    session_start: time = SESSION_START,
    session_end: time = SESSION_END,
) -> RawTapeDay:
    """Read and normalize one day for the exact pairs in ``mapping``."""

    selected = _validate_mapping(mapping)
    spot_path = Path(spot_path) if spot_path is not None else spot_tick_path(date)
    future_path = (
        Path(future_path) if future_path is not None else futures_raw_path(date)
    )
    for path in (spot_path, future_path):
        if not path.exists():
            raise FileNotFoundError(path)

    spot_codes = selected["ValueCode"].to_list()
    future_codes = selected["QuoteCode"].to_list()
    spot_raw = _scan_selected(spot_path, spot_codes, future=False)
    future_raw = _scan_selected(future_path, future_codes, future=True)
    spot_states = normalize_spot_tape(
        spot_raw,
        selected,
        date,
        session_start=session_start,
        session_end=session_end,
    )
    future_states = normalize_future_tape(
        future_raw,
        selected,
        date,
        session_start=session_start,
        session_end=session_end,
    )
    spot_trades = extract_trade_tape(spot_states)
    future_trades = extract_trade_tape(future_states)
    audit = pl.concat(
        [
            _audit_market(spot_states, spot_trades),
            _audit_market(future_states, future_trades),
        ],
        how="vertical_relaxed",
    ).sort(["market", "ValueCode"])
    return RawTapeDay(
        date=str(date),
        mapping=selected,
        spot_states=spot_states,
        future_states=future_states,
        spot_trades=spot_trades,
        future_trades=future_trades,
        audit=audit,
    )


def _normalize_tape(
    raw: pl.DataFrame,
    mapping: pl.DataFrame,
    date: str,
    *,
    market: Market,
    session_start: time,
    session_end: time,
) -> pl.DataFrame:
    selected = _validate_mapping(mapping)
    required = set(_RAW_COMMON)
    if market == "future":
        required.add("DecimalLocator")
    _require_columns(raw, required, f"raw {market} tape")
    trade_date = datetime.strptime(str(date), "%Y%m%d").date()
    recv_expr = _utc_naive_recv_expr(raw)
    local_time = pl.col("TransTime").dt.time()
    filtered = (
        raw.with_row_index("_source_order")
        .filter(
            (pl.col("TransTime").dt.date() == pl.lit(trade_date))
            & (local_time >= pl.lit(session_start))
            & (local_time < pl.lit(session_end))
        )
    )

    if market == "spot":
        filtered = (
            filtered.rename({"QuoteCode": "instrument_code"})
            .filter(
                pl.col("instrument_code").is_in(selected["ValueCode"].to_list())
            )
            .join(
                selected,
                left_on="instrument_code",
                right_on="ValueCode",
                how="inner",
                validate="m:1",
            )
            .with_columns(pl.col("instrument_code").alias("ValueCode"))
        )
        price = lambda column: pl.col(column).cast(pl.Float64)  # noqa: E731
        ref_column = "spot_ref_price"
    else:
        filtered = (
            filtered.rename({"QuoteCode": "instrument_code"})
            .filter(
                pl.col("instrument_code").is_in(selected["QuoteCode"].to_list())
            )
            .join(
                selected,
                left_on="instrument_code",
                right_on="QuoteCode",
                how="inner",
                validate="m:1",
            )
            .with_columns(pl.col("instrument_code").alias("QuoteCode"))
        )
        divisor = pl.lit(10.0).pow(pl.col("DecimalLocator").cast(pl.Float64))
        price = lambda column: pl.col(column).cast(pl.Float64) / divisor  # noqa: E731
        ref_column = "fut_ref_price"

    # Recreate the receive-time expression after the join/rename so Polars does
    # not need to resolve it against a different lazy schema.
    recv_expr = _utc_naive_recv_expr(filtered)
    base_exprs: list[pl.Expr] = [
        pl.lit(str(date)).alias("Date"),
        pl.lit(market).alias("market"),
        recv_expr.alias("recv_time"),
        pl.col("TransTime").cast(pl.Datetime("us")).alias("trans_time"),
        pl.col("ChannelSeq").cast(pl.UInt64).alias("sequence"),
        pl.col("PacketSeq").cast(pl.UInt64).alias("packet_sequence"),
        (pl.col("TrialMatch").fill_null(0) != 0).alias("trial_match"),
        pl.col(ref_column).cast(pl.Float64).alias("ref_price"),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("TotalFillLots").cast(pl.Int64).alias("total_fill_lots"),
        pl.when(price("FillPrice") > 0)
        .then(price("FillPrice"))
        .otherwise(None)
        .alias("fill_price"),
        pl.col("FillLots").fill_null(0).cast(pl.Int64).alias("fill_lots"),
        pl.col("_source_order"),
    ]
    if market == "future":
        base_exprs.append(pl.col("DecimalLocator").cast(pl.Int16).alias("decimal_locator"))
    else:
        base_exprs.append(pl.lit(None, dtype=pl.Int16).alias("decimal_locator"))

    raw_prices: list[pl.Expr] = []
    raw_lots: list[pl.Expr] = []
    for side in ("Bid", "Ask"):
        lower = side.lower()
        for level in range(1, 6):
            raw_prices.append(
                pl.when(price(f"{side}Price{level}") > 0)
                .then(price(f"{side}Price{level}"))
                .otherwise(None)
                .alias(f"_{lower}_price_{level}_raw")
            )
            raw_lots.append(
                pl.col(f"{side}Lots{level}")
                .fill_null(0)
                .cast(pl.Int64)
                .alias(f"_{lower}_lots_{level}_raw")
            )
    for side in ("Bid", "Ask"):
        lower = side.lower()
        raw_prices.append(
            pl.when(price(f"Best{side}Price") > 0)
            .then(price(f"Best{side}Price"))
            .otherwise(None)
            .alias(f"_best_{lower}_price_raw")
        )
        raw_lots.append(
            pl.col(f"Best{side}Lots")
            .fill_null(0)
            .cast(pl.Int64)
            .alias(f"_best_{lower}_lots_raw")
        )

    states = filtered.select(
        "ValueCode", "QuoteCode", "instrument_code", *base_exprs, *raw_prices, *raw_lots
    ).with_columns(
        pl.col("recv_time").cast(pl.Int64).alias("recv_time_ns"),
        pl.any_horizontal(
            *[pl.col(f"_{side}_price_{level}_raw").is_not_null()
              for side in ("bid", "ask") for level in range(1, 6)]
        ).alias("raw_has_l1_book"),
        pl.any_horizontal(
            pl.col("_best_bid_price_raw").is_not_null(),
            pl.col("_best_ask_price_raw").is_not_null(),
        ).alias("raw_has_best_book"),
    ).sort(
        ["ValueCode", "recv_time", "sequence", "packet_sequence", "_source_order"]
    )
    states = states.with_columns(
        (pl.col("raw_has_l1_book") | pl.col("raw_has_best_book")).alias(
            "raw_has_book"
        )
    )

    l1_fields: list[pl.Expr] = []
    for side in ("bid", "ask"):
        for level in range(1, 6):
            l1_fields.extend(
                [
                    pl.col(f"_{side}_price_{level}_raw").alias(
                        f"{side}_price_{level}"
                    ),
                    pl.when(pl.col(f"_{side}_price_{level}_raw").is_not_null())
                    .then(pl.col(f"_{side}_lots_{level}_raw"))
                    .otherwise(None)
                    .alias(f"{side}_lots_{level}"),
                ]
            )
    best_fields = [
        pl.col(f"_best_{side}_price_raw").alias(f"best_{side}_price")
        for side in ("bid", "ask")
    ] + [
        pl.when(pl.col(f"_best_{side}_price_raw").is_not_null())
        .then(pl.col(f"_best_{side}_lots_raw"))
        .otherwise(None)
        .alias(f"best_{side}_lots")
        for side in ("bid", "ask")
    ]
    cursor_fields = [
        pl.col("recv_time").alias("book_recv_time"),
        pl.col("recv_time_ns").alias("book_recv_time_ns"),
        pl.col("sequence").alias("book_sequence"),
        pl.col("packet_sequence").alias("book_packet_sequence"),
    ]
    states = states.with_columns(
        pl.when("raw_has_l1_book").then(pl.struct(l1_fields)).otherwise(None).alias("_l1_snapshot"),
        pl.when("raw_has_best_book").then(pl.struct(best_fields)).otherwise(None).alias("_best_snapshot"),
        pl.when("raw_has_book").then(pl.struct(cursor_fields)).otherwise(None).alias("_book_cursor"),
    ).with_columns(
        pl.col("_l1_snapshot").forward_fill().over("ValueCode"),
        pl.col("_best_snapshot").forward_fill().over("ValueCode"),
        pl.col("_book_cursor").forward_fill().over("ValueCode"),
    ).unnest(["_l1_snapshot", "_best_snapshot", "_book_cursor"])

    if market == "spot":
        states = states.with_columns(
            pl.col("bid_price_1").alias("exec_bid_price"),
            pl.col("bid_lots_1").alias("exec_bid_lots"),
            pl.col("ask_price_1").alias("exec_ask_price"),
            pl.col("ask_lots_1").alias("exec_ask_lots"),
        )
    else:
        states = states.with_columns(
            _exec_price("bid"),
            _exec_price("ask"),
        ).with_columns(
            _exec_lots("bid"),
            _exec_lots("ask"),
        )

    raw_helpers = [
        column
        for column in states.columns
        if column.startswith("_") and column != "_source_order"
    ]
    return (
        states.with_columns(
            pl.col("book_recv_time").is_not_null().alias("book_state_available"),
            pl.col("raw_has_book").alias("book_update"),
        )
        .drop(raw_helpers + ["_source_order"])
        .sort(["ValueCode", "recv_time", "sequence", "packet_sequence"])
    )


def _exec_price(side: Literal["bid", "ask"]) -> pl.Expr:
    l1 = pl.col(f"{side}_price_1")
    best = pl.col(f"best_{side}_price")
    if side == "bid":
        return pl.max_horizontal(l1, best).alias("exec_bid_price")
    return pl.min_horizontal(l1, best).alias("exec_ask_price")


def _exec_lots(side: Literal["bid", "ask"]) -> pl.Expr:
    price = pl.col(f"exec_{side}_price")
    l1_price = pl.col(f"{side}_price_1")
    best_price = pl.col(f"best_{side}_price")
    l1_lots = pl.col(f"{side}_lots_1")
    best_lots = pl.col(f"best_{side}_lots")
    return (
        pl.when(price.is_null())
        .then(None)
        .when((l1_price == price) & (best_price == price))
        .then(pl.max_horizontal(l1_lots, best_lots))
        .when(l1_price == price)
        .then(l1_lots)
        .otherwise(best_lots)
        .alias(f"exec_{side}_lots")
    )


def _utc_naive_recv_expr(frame: pl.DataFrame) -> pl.Expr:
    dtype = frame.schema.get("RecvTime")
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("RecvTime must be a Datetime column")
    value = pl.col("RecvTime")
    if dtype.time_zone is not None:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return value.cast(pl.Datetime("ns"))


def _validate_mapping(mapping: pl.DataFrame) -> pl.DataFrame:
    required = {
        "ValueCode",
        "QuoteCode",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
    }
    _require_columns(mapping, required, "contract mapping")
    selected = mapping.with_columns(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    for key in ("ValueCode", "QuoteCode"):
        duplicate = selected.group_by(key).len().filter(pl.col("len") != 1)
        if duplicate.height:
            raise ValueError(f"contract mapping is not one-to-one on {key}")
    return selected.sort(["ValueCode", "QuoteCode"])


def _scan_selected(path: Path, codes: list[str], *, future: bool) -> pl.DataFrame:
    required = set(_RAW_COMMON)
    if future:
        required.add("DecimalLocator")
    scan = pl.scan_parquet(path)
    available = set(scan.collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return (
        scan.filter(pl.col("QuoteCode").cast(pl.String).is_in(codes))
        .select(sorted(required))
        .collect(engine="streaming")
    )


def _audit_market(states: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    if states.is_empty():
        return pl.DataFrame(
            schema={
                "Date": pl.String,
                "market": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
                "event_rows": pl.UInt32,
                "book_update_rows": pl.UInt32,
                "zero_book_trade_rows": pl.UInt32,
                "trade_rows": pl.UInt32,
                "rows_without_prior_book": pl.UInt32,
            }
        )
    return states.group_by(["Date", "market", "ValueCode", "QuoteCode"]).agg(
        pl.len().alias("event_rows"),
        pl.col("raw_has_book").sum().cast(pl.UInt32).alias("book_update_rows"),
        ((pl.col("fill_lots") > 0) & ~pl.col("raw_has_book"))
        .sum()
        .cast(pl.UInt32)
        .alias("zero_book_trade_rows"),
        (pl.col("fill_lots") > 0).sum().cast(pl.UInt32).alias("trade_rows"),
        (~pl.col("book_state_available"))
        .sum()
        .cast(pl.UInt32)
        .alias("rows_without_prior_book"),
    )


def _require_columns(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
