"""Compact physical spot-trade index for the S1 joint event clock.

The index consumes an already-selected raw spot scan and the exact daily
``ValueCode``/``QuoteCode`` mapping.  It retains every row with a finite,
positive ``FillPrice`` and positive ``FillLots``.  ``TrialMatch`` is audit
provenance only and never suppresses a physical trade.

Raw-book effects use event sequence zero.  Spot trades deliberately use event
sequence one, so every same-receive-time book effect precedes every trade.
Within that trade phase, ``row_index`` is the ordinal of positive trade rows
ordered by ValueCode, ChannelSeq, PacketSeq, and global source row.  It is not
the raw-book index's row ordinal.

The retained day is columnar.  Only one small product-to-range map is built in
Python; immutable trade facts are reconstructed for queried rows.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import polars as pl

from .layered import EventCursor

SelectedSpotFrame = pl.DataFrame | pl.LazyFrame

SPOT_TRADE_EVENT_SEQUENCE: Final = 1
SPOT_BOARD_LOT_SHARES: Final = 1_000

_MAPPING_COLUMNS: Final = {"ValueCode", "QuoteCode"}
_RAW_COLUMNS: Final = {
    "RecvTime",
    "QuoteCode",
    "ChannelSeq",
    "PacketSeq",
    "TrialMatch",
    "FillPrice",
    "FillLots",
}
_TRADE_COLUMNS: Final = (
    "series_id",
    "recv_time_ns",
    "loop_row_index",
    "channel_sequence",
    "packet_sequence",
    "source_row",
    "trade_price",
    "quantity_shares",
    "trial_match",
)
_PRODUCT_COLUMNS: Final = ("series_id", "product_id", "quote_code")


@dataclass(frozen=True, slots=True)
class PhysicalSpotTrade:
    """One immutable exchange trade and its exact source identity."""

    product_id: str
    quote_code: str
    cursor: EventCursor
    trade_price: float
    quantity_shares: int
    source_id: str
    channel_sequence: int
    packet_sequence: int
    source_row: int
    trial_match: bool

    def __post_init__(self) -> None:
        _text(self.product_id, "product_id")
        _text(self.quote_code, "quote_code")
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if self.cursor.event_sequence != SPOT_TRADE_EVENT_SEQUENCE:
            raise ValueError("physical spot trade must use the spot-trade phase")
        _positive_float(self.trade_price, "trade_price")
        _positive_integer(self.quantity_shares, "quantity_shares")
        if self.quantity_shares % SPOT_BOARD_LOT_SHARES:
            raise ValueError("spot trade quantity must contain whole board lots")
        for name in ("channel_sequence", "packet_sequence", "source_row"):
            _nonnegative_integer(getattr(self, name), name)
        if not isinstance(self.trial_match, bool):
            raise TypeError("trial_match must be boolean")
        expected = physical_spot_trade_source_id(
            self.product_id,
            self.quote_code,
            self.cursor.recv_time_ns,
            self.channel_sequence,
            self.packet_sequence,
            self.source_row,
        )
        if self.source_id != expected:
            raise ValueError("source_id does not match physical trade provenance")

    @property
    def fill_lots(self) -> int:
        """Return the exchange-native board-lot quantity."""

        return self.quantity_shares // SPOT_BOARD_LOT_SHARES


@dataclass(frozen=True, slots=True)
class _TradeSeries:
    series_id: int
    position_start: int
    length: int
    quote_code: str


@dataclass(frozen=True, slots=True)
class _RangeMaxTree:
    """Immutable compact global range-max tree over retained trade prices."""

    leaf_count: int
    value_count: int
    packed_values: bytes = field(repr=False)

    @classmethod
    def from_prices(cls, prices: pl.Series) -> _RangeMaxTree:
        value_count = len(prices)
        leaf_count = 1 << (max(value_count, 1) - 1).bit_length()
        values = array("d", [float("-inf")]) * (2 * leaf_count)
        for position, raw_price in enumerate(prices):
            values[leaf_count + position] = _positive_float(
                raw_price, "indexed trade_price"
            )
        for node in range(leaf_count - 1, 0, -1):
            values[node] = max(values[2 * node], values[2 * node + 1])
        return cls(leaf_count, value_count, values.tobytes())

    def first_at_or_above(
        self,
        position_start: int,
        position_stop: int,
        threshold: float,
    ) -> int | None:
        """Return the first qualifying global position in ``[start, stop)``."""

        if (
            position_start < 0
            or position_start > position_stop
            or position_stop > self.value_count
        ):
            raise ValueError("range-max query is outside retained trades")
        if position_start == position_stop:
            return None
        values = memoryview(self.packed_values).cast("d")

        def search(node: int, node_start: int, node_stop: int) -> int | None:
            if (
                node_stop <= position_start
                or position_stop <= node_start
                or values[node] < threshold
            ):
                return None
            if node >= self.leaf_count:
                return node_start if node_start < self.value_count else None
            middle = (node_start + node_stop) // 2
            left = search(2 * node, node_start, middle)
            if left is not None:
                return left
            return search(2 * node + 1, middle, node_stop)

        return search(1, 0, self.leaf_count)


@dataclass(frozen=True, slots=True)
class SpotTradeDayIndex:
    """Immutable compact day index with logarithmic per-product lookup."""

    _trades: pl.DataFrame = field(repr=False)
    _products: pl.DataFrame = field(repr=False)
    _series: Mapping[str, _TradeSeries] = field(repr=False)
    _range_max_tree: _RangeMaxTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_columns(self._trades, set(_TRADE_COLUMNS), "spot trade clock")
        _require_columns(self._products, set(_PRODUCT_COLUMNS), "spot products")
        copied = dict(self._series)
        if any(
            not isinstance(product_id, str)
            or not product_id
            or not isinstance(series, _TradeSeries)
            for product_id, series in copied.items()
        ):
            raise TypeError("spot trade index has invalid series metadata")
        if tuple(sorted(copied)) != tuple(self._products["product_id"].to_list()):
            raise ValueError("spot trade product metadata is inconsistent")
        object.__setattr__(self, "_series", MappingProxyType(copied))
        object.__setattr__(
            self,
            "_range_max_tree",
            _RangeMaxTree.from_prices(self._trades["trade_price"]),
        )

    @classmethod
    def from_selected_rows(
        cls,
        spot_rows: pl.DataFrame,
        mapping: pl.DataFrame,
    ) -> SpotTradeDayIndex:
        """Compatibility constructor for an eager selected raw frame."""

        if not isinstance(spot_rows, pl.DataFrame):
            raise TypeError("spot_rows must be a Polars DataFrame")
        return cls.from_selected_scan(spot_rows.lazy(), mapping)

    @classmethod
    def from_selected_scan(
        cls,
        spot_rows: SelectedSpotFrame,
        mapping: pl.DataFrame,
    ) -> SpotTradeDayIndex:
        """Build directly from an eager frame or lazy scan without row maps."""

        raw = _as_lazy(spot_rows)
        schema = raw.collect_schema()
        _validate_raw_schema(schema)
        products = _product_frame(mapping)
        lookup = products.select(
            pl.col("product_id").alias("instrument_code"),
            "product_id",
            "quote_code",
            "series_id",
        )

        selected = raw.with_row_index("source_row").join(
            lookup.lazy(),
            left_on="QuoteCode",
            right_on="instrument_code",
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
        raw_mapping_check = (
            (
                pl.col("ValueCode").is_not_null()
                & (pl.col("ValueCode").cast(pl.String) == pl.col("product_id"))
            )
            .fill_null(False)
            .alias("raw_value_code_matches")
            if "ValueCode" in schema
            else pl.lit(True).alias("raw_value_code_matches")
        )

        candidate = (
            selected.filter(_positive_trade_expr())
            .select(
                "series_id",
                "product_id",
                "quote_code",
                _recv_time_ns_expr(schema).alias("recv_time_ns"),
                pl.col("ChannelSeq")
                .cast(pl.Int64, strict=True)
                .alias("channel_sequence"),
                pl.col("PacketSeq")
                .cast(pl.Int64, strict=True)
                .alias("packet_sequence"),
                pl.col("source_row").cast(pl.UInt64),
                raw_mapping_check,
                pl.col("FillPrice").cast(pl.Float64, strict=True).alias("trade_price"),
                (
                    pl.col("FillLots").cast(pl.Int64, strict=True)
                    * SPOT_BOARD_LOT_SHARES
                ).alias("quantity_shares"),
                (pl.col("TrialMatch").fill_null(0) != 0).alias("trial_match"),
            )
            .collect(engine="streaming")
        )
        _validate_candidate(candidate)
        trades = _order_and_compact(candidate)
        return cls(trades, products, _series_metadata(products, trades))

    @property
    def product_ids(self) -> tuple[str, ...]:
        """All exact-mapping ValueCodes, including products with no trades."""

        return tuple(self._products["product_id"].to_list())

    @property
    def trade_count(self) -> int:
        return self._trades.height

    @property
    def retained_trade_count(self) -> int:
        return self.trade_count

    @property
    def estimated_size_bytes(self) -> int:
        """Estimated bytes for canonical frames and the derived query tree."""

        return (
            self._trades.estimated_size()
            + self._products.estimated_size()
            + len(self._range_max_tree.packed_values)
        )

    def storage_frames(self) -> Mapping[str, pl.DataFrame]:
        """Return canonical frames; the derived range-max tree is rebuilt."""

        return MappingProxyType(
            {"trades": self._trades.clone(), "products": self._products.clone()}
        )

    def trade_count_for(self, product_id: str) -> int:
        return self._require_series(product_id).length

    def next_trade_cursor(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        """Return the first trade strictly after ``after_cursor`` by deadline."""

        series = self._require_series(product_id)
        after = _event_cursor(after_cursor, "after_cursor")
        deadline = _nonnegative_integer(deadline_ns, "deadline_ns")
        if deadline < after.recv_time_ns:
            raise ValueError("deadline_ns cannot precede after_cursor")
        relative = self._right_cursor_index(series, after)
        if relative >= series.length:
            return None
        cursor = self._cursor(series.position_start + relative)
        return None if cursor.recv_time_ns > deadline else cursor

    def next_trade_cursor_at_or_above(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
        minimum_trade_price: float,
    ) -> EventCursor | None:
        """Return the first later trade meeting a price floor by deadline.

        Cursor ordering is strict, while ``deadline_ns`` and the price floor
        are inclusive.  The global range-max tree finds the first qualifying
        row without waking or scanning through intervening lower-price trades.
        """

        series = self._require_series(product_id)
        after = _event_cursor(after_cursor, "after_cursor")
        deadline = _nonnegative_integer(deadline_ns, "deadline_ns")
        threshold = _positive_float(minimum_trade_price, "minimum_trade_price")
        if deadline < after.recv_time_ns:
            raise ValueError("deadline_ns cannot precede after_cursor")
        relative_start = self._right_cursor_index(series, after)
        relative_stop = self._right_time_index(series, deadline)
        if relative_start >= relative_stop:
            return None
        position = self._range_max_tree.first_at_or_above(
            series.position_start + relative_start,
            series.position_start + relative_stop,
            threshold,
        )
        return None if position is None else self._cursor(position)

    def trades_at(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> tuple[PhysicalSpotTrade, ...]:
        """Return all physical trades at one receive timestamp in loop order."""

        series = self._require_series(product_id)
        timestamp = _nonnegative_integer(timestamp_ns, "timestamp_ns")
        left = self._left_time_index(series, timestamp)
        right = self._right_time_index(series, timestamp)
        return tuple(
            self._trade(series.position_start + relative, product_id)
            for relative in range(left, right)
        )

    def _left_time_index(self, series: _TradeSeries, timestamp_ns: int) -> int:
        low = 0
        high = series.length
        while low < high:
            middle = (low + high) // 2
            position = series.position_start + middle
            if int(self._trades["recv_time_ns"][position]) < timestamp_ns:
                low = middle + 1
            else:
                high = middle
        return low

    def _right_time_index(self, series: _TradeSeries, timestamp_ns: int) -> int:
        low = 0
        high = series.length
        while low < high:
            middle = (low + high) // 2
            position = series.position_start + middle
            if int(self._trades["recv_time_ns"][position]) <= timestamp_ns:
                low = middle + 1
            else:
                high = middle
        return low

    def _right_cursor_index(
        self,
        series: _TradeSeries,
        after: EventCursor,
    ) -> int:
        low = 0
        high = series.length
        while low < high:
            middle = (low + high) // 2
            position = series.position_start + middle
            if self._cursor(position) <= after:
                low = middle + 1
            else:
                high = middle
        return low

    def _cursor(self, position: int) -> EventCursor:
        return EventCursor(
            int(self._trades["recv_time_ns"][position]),
            SPOT_TRADE_EVENT_SEQUENCE,
            int(self._trades["loop_row_index"][position]),
        )

    def _trade(self, position: int, product_id: str) -> PhysicalSpotTrade:
        row = self._trades.row(position, named=True)
        series = self._require_series(product_id)
        cursor = EventCursor(
            int(row["recv_time_ns"]),
            SPOT_TRADE_EVENT_SEQUENCE,
            int(row["loop_row_index"]),
        )
        channel_sequence = int(row["channel_sequence"])
        packet_sequence = int(row["packet_sequence"])
        source_row = int(row["source_row"])
        source_id = physical_spot_trade_source_id(
            product_id,
            series.quote_code,
            cursor.recv_time_ns,
            channel_sequence,
            packet_sequence,
            source_row,
        )
        return PhysicalSpotTrade(
            product_id=product_id,
            quote_code=series.quote_code,
            cursor=cursor,
            trade_price=float(row["trade_price"]),
            quantity_shares=int(row["quantity_shares"]),
            source_id=source_id,
            channel_sequence=channel_sequence,
            packet_sequence=packet_sequence,
            source_row=source_row,
            trial_match=bool(row["trial_match"]),
        )

    def _require_series(self, product_id: str) -> _TradeSeries:
        _text(product_id, "product_id")
        try:
            return self._series[product_id]
        except KeyError as error:
            raise ValueError(
                "product_id is not in the exact spot-trade mapping"
            ) from error


def physical_spot_trade_source_id(
    product_id: str,
    quote_code: str,
    recv_time_ns: int,
    channel_sequence: int,
    packet_sequence: int,
    source_row: int,
) -> str:
    """Build the stable, audit-readable identity of one physical raw trade."""

    _text(product_id, "product_id")
    _text(quote_code, "quote_code")
    values = (
        _nonnegative_integer(recv_time_ns, "recv_time_ns"),
        _nonnegative_integer(channel_sequence, "channel_sequence"),
        _nonnegative_integer(packet_sequence, "packet_sequence"),
        _nonnegative_integer(source_row, "source_row"),
    )
    return "spot_trade:" + ":".join(
        (product_id, quote_code, *(str(value) for value in values))
    )


def build_spot_trade_day_index(
    spot_rows: pl.DataFrame,
    mapping: pl.DataFrame,
) -> SpotTradeDayIndex:
    """Compatibility entry point for eager selected raw rows."""

    return SpotTradeDayIndex.from_selected_rows(spot_rows, mapping)


def build_spot_trade_day_index_from_scan(
    spot_rows: SelectedSpotFrame,
    mapping: pl.DataFrame,
) -> SpotTradeDayIndex:
    """Scale-oriented entry point accepting a lazy parquet scan."""

    return SpotTradeDayIndex.from_selected_scan(spot_rows, mapping)


def build_spot_trade_day_index_from_scans(
    spot_rows: SelectedSpotFrame,
    mapping: pl.DataFrame,
) -> SpotTradeDayIndex:
    """Plural compatibility alias matching other daily scan builders."""

    return build_spot_trade_day_index_from_scan(spot_rows, mapping)


def _product_frame(mapping: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(mapping, pl.DataFrame):
        raise TypeError("mapping must be a Polars DataFrame")
    _require_columns(mapping, _MAPPING_COLUMNS, "contract mapping")
    products = (
        mapping.select(
            pl.col("ValueCode").cast(pl.String).alias("product_id"),
            pl.col("QuoteCode").cast(pl.String).alias("quote_code"),
        )
        .sort("product_id", "quote_code")
        .with_row_index("series_id")
        .select("series_id", "product_id", "quote_code")
    )
    invalid = products.filter(
        pl.col("product_id").is_null()
        | (pl.col("product_id").str.len_chars() == 0)
        | pl.col("quote_code").is_null()
        | (pl.col("quote_code").str.len_chars() == 0)
    )
    if not invalid.is_empty():
        raise ValueError("contract mapping codes must be non-empty")
    for column in ("product_id", "quote_code"):
        duplicate = products.group_by(column).len().filter(pl.col("len") != 1)
        if not duplicate.is_empty():
            raise ValueError(f"contract mapping is not one-to-one on {column}")
    return products.with_columns(pl.col("series_id").cast(pl.UInt32))


def _validate_raw_schema(schema: pl.Schema) -> None:
    _require_schema_columns(schema, _RAW_COLUMNS, "raw spot rows")
    if not isinstance(schema["RecvTime"], pl.Datetime):
        raise TypeError("RecvTime must be a Datetime column")
    for column in ("ChannelSeq", "PacketSeq", "FillLots"):
        if not schema[column].is_integer():
            raise TypeError(f"{column} must be an integer column")
    if not schema["FillPrice"].is_numeric():
        raise TypeError("FillPrice must be a numeric column")


def _positive_trade_expr() -> pl.Expr:
    price = pl.col("FillPrice").cast(pl.Float64, strict=True)
    return price.is_finite() & (price > 0) & (pl.col("FillLots") > 0)


def _validate_candidate(candidate: pl.DataFrame) -> None:
    if not candidate.filter(~pl.col("raw_value_code_matches")).is_empty():
        raise ValueError("raw spot ValueCode does not match the exact daily mapping")
    invalid = candidate.filter(
        pl.col("recv_time_ns").is_null()
        | (pl.col("recv_time_ns") < 0)
        | pl.col("channel_sequence").is_null()
        | (pl.col("channel_sequence") < 0)
        | pl.col("packet_sequence").is_null()
        | (pl.col("packet_sequence") < 0)
        | pl.col("quantity_shares").is_null()
        | (pl.col("quantity_shares") <= 0)
        | pl.col("trade_price").is_null()
        | ~pl.col("trade_price").is_finite()
        | (pl.col("trade_price") <= 0)
    )
    if not invalid.is_empty():
        raise ValueError("positive spot trade rows have invalid cursor or payload")
    duplicate = (
        candidate.group_by(
            "series_id",
            "recv_time_ns",
            "channel_sequence",
            "packet_sequence",
        )
        .len()
        .filter(pl.col("len") != 1)
    )
    if not duplicate.is_empty():
        raise ValueError("raw spot trades have an undecidable duplicate cursor")


def _order_and_compact(candidate: pl.DataFrame) -> pl.DataFrame:
    ordered = candidate.sort(
        "recv_time_ns",
        "product_id",
        "channel_sequence",
        "packet_sequence",
        "source_row",
    ).with_columns(
        (
            pl.struct(
                "product_id",
                "channel_sequence",
                "packet_sequence",
                "source_row",
            )
            .rank("ordinal")
            .over("recv_time_ns")
            - 1
        )
        .cast(pl.UInt32)
        .alias("loop_row_index")
    )
    return (
        ordered.select(*_TRADE_COLUMNS)
        .with_columns(
            pl.col("series_id").cast(pl.UInt32),
            pl.col("channel_sequence").cast(pl.UInt64),
            pl.col("packet_sequence").cast(pl.UInt64),
            pl.col("source_row").cast(pl.UInt64),
            pl.col("quantity_shares").cast(pl.UInt64),
        )
        .sort("series_id", "recv_time_ns", "loop_row_index")
    )


def _series_metadata(
    products: pl.DataFrame,
    trades: pl.DataFrame,
) -> Mapping[str, _TradeSeries]:
    ranges = {
        int(row[0]): (int(row[1]), int(row[2]))
        for row in (
            trades.select("series_id")
            .with_row_index("position_offset")
            .group_by("series_id", maintain_order=True)
            .agg(
                pl.col("position_offset").min().alias("position_start"),
                pl.len().alias("length"),
            )
            .iter_rows()
        )
    }
    result: dict[str, _TradeSeries] = {}
    for series_id, product_id, quote_code in products.iter_rows():
        start, length = ranges.get(int(series_id), (0, 0))
        result[str(product_id)] = _TradeSeries(
            int(series_id), start, length, str(quote_code)
        )
    return result


def _recv_time_ns_expr(schema: pl.Schema) -> pl.Expr:
    dtype = schema["RecvTime"]
    assert isinstance(dtype, pl.Datetime)
    value = pl.col("RecvTime")
    if dtype.time_zone is not None:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return value.cast(pl.Datetime("ns")).cast(pl.Int64)


def _as_lazy(frame: SelectedSpotFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    if isinstance(frame, pl.LazyFrame):
        return frame
    raise TypeError("spot_rows must be a Polars DataFrame or LazyFrame")


def _event_cursor(value: object, name: str) -> EventCursor:
    if not isinstance(value, EventCursor):
        raise TypeError(f"{name} must be an EventCursor")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _require_columns(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _require_schema_columns(
    schema: pl.Schema,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


__all__ = [
    "SPOT_BOARD_LOT_SHARES",
    "SPOT_TRADE_EVENT_SEQUENCE",
    "PhysicalSpotTrade",
    "SpotTradeDayIndex",
    "build_spot_trade_day_index",
    "build_spot_trade_day_index_from_scan",
    "build_spot_trade_day_index_from_scans",
    "physical_spot_trade_source_id",
]
