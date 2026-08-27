"""Exact official spot-close facts for S1 futures-expiry accounting.

The raw stock feed carries an exchange ``Close`` field.  This adapter only
observes rows at or after an explicit close-publication boundary and retains
the final such row for each exact daily ValueCode/QuoteCode mapping.  The fact
is accounting provenance; it is never an executable trade or book event.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final

import polars as pl

from .layered import EventCursor

SPOT_CLOSE_EVENT_SEQUENCE: Final = 2

_MAPPING_COLUMNS: Final = {"ValueCode", "QuoteCode"}
_RAW_COLUMNS: Final = {
    "RecvTime",
    "QuoteCode",
    "ChannelSeq",
    "PacketSeq",
    "Close",
}


@dataclass(frozen=True, slots=True)
class OfficialSpotClose:
    """One non-executable close-price fact with exact raw provenance."""

    date: str
    product_id: str
    quote_code: str
    close_price: float
    source_cursor: EventCursor
    source_id: str
    channel_sequence: int
    packet_sequence: int
    source_row: int

    def __post_init__(self) -> None:
        _yyyymmdd(self.date)
        _text(self.product_id, "product_id")
        _text(self.quote_code, "quote_code")
        _positive_float(self.close_price, "close_price")
        if not isinstance(self.source_cursor, EventCursor):
            raise TypeError("source_cursor must be an EventCursor")
        if self.source_cursor.event_sequence != SPOT_CLOSE_EVENT_SEQUENCE:
            raise ValueError("spot close must use the close-provenance phase")
        for name in ("channel_sequence", "packet_sequence", "source_row"):
            _nonnegative_integer(getattr(self, name), name)
        expected = official_spot_close_source_id(
            self.date,
            self.product_id,
            self.quote_code,
            self.source_cursor.recv_time_ns,
            self.channel_sequence,
            self.packet_sequence,
            self.source_row,
        )
        if self.source_id != expected:
            raise ValueError("source_id does not match spot-close provenance")


@dataclass(frozen=True, slots=True)
class SpotCloseDayIndex:
    """Small immutable product lookup for one session's official closes."""

    date: str
    close_not_before_ns: int
    _product_ids: tuple[str, ...] = field(repr=False)
    _facts: Mapping[str, OfficialSpotClose] = field(repr=False)

    def __post_init__(self) -> None:
        _yyyymmdd(self.date)
        _nonnegative_integer(self.close_not_before_ns, "close_not_before_ns")
        if tuple(sorted(set(self._product_ids))) != self._product_ids:
            raise ValueError("product_ids must be sorted and unique")
        copied = dict(self._facts)
        if any(
            key not in self._product_ids
            or not isinstance(value, OfficialSpotClose)
            or value.date != self.date
            or value.product_id != key
            or value.source_cursor.recv_time_ns < self.close_not_before_ns
            for key, value in copied.items()
        ):
            raise ValueError("spot-close facts disagree with index metadata")
        object.__setattr__(self, "_facts", MappingProxyType(copied))

    @classmethod
    def from_selected_rows(
        cls,
        spot_rows: pl.DataFrame,
        mapping: pl.DataFrame,
        *,
        date: str,
        close_not_before_ns: int,
    ) -> SpotCloseDayIndex:
        if not isinstance(spot_rows, pl.DataFrame):
            raise TypeError("spot_rows must be a Polars DataFrame")
        return cls.from_selected_scan(
            spot_rows.lazy(),
            mapping,
            date=date,
            close_not_before_ns=close_not_before_ns,
        )

    @classmethod
    def from_selected_scan(
        cls,
        spot_rows: pl.DataFrame | pl.LazyFrame,
        mapping: pl.DataFrame,
        *,
        date: str,
        close_not_before_ns: int,
    ) -> SpotCloseDayIndex:
        selected_date = _yyyymmdd(date)
        boundary = _nonnegative_integer(
            close_not_before_ns,
            "close_not_before_ns",
        )
        raw = spot_rows.lazy() if isinstance(spot_rows, pl.DataFrame) else spot_rows
        if not isinstance(raw, pl.LazyFrame):
            raise TypeError("spot_rows must be a Polars DataFrame or LazyFrame")
        schema = raw.collect_schema()
        missing_raw = sorted(_RAW_COLUMNS - set(schema.names()))
        if missing_raw:
            raise ValueError(f"spot close raw input missing columns: {missing_raw}")
        products = _product_frame(mapping)
        lookup = products.select(
            pl.col("product_id").alias("instrument_code"),
            "product_id",
            "quote_code",
        )
        recv_time_ns = _recv_time_ns_expr(schema)
        value_matches = (
            (
                pl.col("ValueCode").is_not_null()
                & (pl.col("ValueCode").cast(pl.String) == pl.col("product_id"))
            )
            .fill_null(False)
            .alias("raw_value_code_matches")
            if "ValueCode" in schema
            else pl.lit(True).alias("raw_value_code_matches")
        )
        candidates = (
            raw.with_row_index("source_row")
            .join(
                lookup.lazy(),
                left_on="QuoteCode",
                right_on="instrument_code",
                how="inner",
                validate="m:1",
                maintain_order="left",
            )
            .select(
                "product_id",
                "quote_code",
                recv_time_ns.alias("recv_time_ns"),
                pl.col("ChannelSeq")
                .cast(pl.Int64, strict=True)
                .alias("channel_sequence"),
                pl.col("PacketSeq")
                .cast(pl.Int64, strict=True)
                .alias("packet_sequence"),
                pl.col("source_row").cast(pl.UInt64),
                pl.col("Close").cast(pl.Float64, strict=True).alias("close_price"),
                value_matches,
            )
            .filter(
                (pl.col("recv_time_ns") >= boundary)
                & pl.col("close_price").is_finite()
                & (pl.col("close_price") > 0)
            )
            .collect(engine="streaming")
        )
        if candidates.filter(~pl.col("raw_value_code_matches")).height:
            raise ValueError("raw ValueCode disagrees with exact close mapping")
        latest = (
            candidates.sort(
                "product_id",
                "recv_time_ns",
                "channel_sequence",
                "packet_sequence",
                "source_row",
            )
            .group_by("product_id", maintain_order=True)
            .tail(1)
            .sort(
                "recv_time_ns",
                "product_id",
                "channel_sequence",
                "packet_sequence",
                "source_row",
            )
            .with_row_index("loop_row_index")
        )
        facts: dict[str, OfficialSpotClose] = {}
        for row in latest.iter_rows(named=True):
            cursor = EventCursor(
                int(row["recv_time_ns"]),
                SPOT_CLOSE_EVENT_SEQUENCE,
                int(row["loop_row_index"]),
            )
            product_id = str(row["product_id"])
            quote_code = str(row["quote_code"])
            source_id = official_spot_close_source_id(
                selected_date,
                product_id,
                quote_code,
                cursor.recv_time_ns,
                int(row["channel_sequence"]),
                int(row["packet_sequence"]),
                int(row["source_row"]),
            )
            facts[product_id] = OfficialSpotClose(
                date=selected_date,
                product_id=product_id,
                quote_code=quote_code,
                close_price=float(row["close_price"]),
                source_cursor=cursor,
                source_id=source_id,
                channel_sequence=int(row["channel_sequence"]),
                packet_sequence=int(row["packet_sequence"]),
                source_row=int(row["source_row"]),
            )
        return cls(
            selected_date,
            boundary,
            tuple(products["product_id"].to_list()),
            facts,
        )

    @property
    def product_ids(self) -> tuple[str, ...]:
        return self._product_ids

    @property
    def available_count(self) -> int:
        return len(self._facts)

    @property
    def facts(self) -> tuple[OfficialSpotClose, ...]:
        return tuple(sorted(self._facts.values(), key=lambda fact: fact.source_cursor))

    def close_fact(self, product_id: str) -> OfficialSpotClose | None:
        _text(product_id, "product_id")
        if product_id not in self._product_ids:
            raise KeyError(f"unknown product_id: {product_id}")
        return self._facts.get(product_id)


def official_spot_close_source_id(
    date: str,
    product_id: str,
    quote_code: str,
    recv_time_ns: int,
    channel_sequence: int,
    packet_sequence: int,
    source_row: int,
) -> str:
    return (
        "official_spot_close/"
        f"{_yyyymmdd(date)}/{_text(product_id, 'product_id')}/"
        f"{_text(quote_code, 'quote_code')}/{_nonnegative_integer(recv_time_ns, 'recv_time_ns')}/"
        f"{_nonnegative_integer(channel_sequence, 'channel_sequence')}/"
        f"{_nonnegative_integer(packet_sequence, 'packet_sequence')}/"
        f"{_nonnegative_integer(source_row, 'source_row')}"
    )


def _product_frame(mapping: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(mapping, pl.DataFrame):
        raise TypeError("mapping must be a Polars DataFrame")
    missing = sorted(_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise ValueError(f"spot close mapping missing columns: {missing}")
    products = (
        mapping.select(
            pl.col("ValueCode").cast(pl.String).alias("product_id"),
            pl.col("QuoteCode").cast(pl.String).alias("quote_code"),
        )
        .unique()
        .sort("product_id")
    )
    if products.is_empty() or products["product_id"].null_count():
        raise ValueError("spot close mapping cannot be empty or null")
    if products["quote_code"].null_count():
        raise ValueError("spot close mapping quote_code cannot be null")
    if products["product_id"].n_unique() != products.height:
        raise ValueError("spot close mapping product_id is not one-to-one")
    if products["quote_code"].n_unique() != products.height:
        raise ValueError("spot close mapping quote_code is not one-to-one")
    return products


def _recv_time_ns_expr(schema: pl.Schema) -> pl.Expr:
    dtype = schema["RecvTime"]
    if dtype == pl.Int64 or dtype == pl.UInt64:
        return pl.col("RecvTime").cast(pl.Int64, strict=True)
    if isinstance(dtype, pl.Datetime):
        return pl.col("RecvTime").dt.timestamp("ns")
    raise ValueError("RecvTime must be integer nanoseconds or datetime")


def _yyyymmdd(value: object) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError("date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("date must be a valid YYYYMMDD") from error
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


__all__ = [
    "SPOT_CLOSE_EVENT_SEQUENCE",
    "OfficialSpotClose",
    "SpotCloseDayIndex",
    "official_spot_close_source_id",
]
