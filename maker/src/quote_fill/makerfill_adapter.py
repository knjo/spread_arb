"""Legacy makerFill adapter for actual-send S1 order lifecycles.

The historical makerFill dataset is an approximate, EOD-looking label.  It can
identify an implied full-fill cursor for a target displayed at spot BID1 or
BID2, but it must not create an order window.  S1 first creates a physical raw
order at its actual new-send cursor; this module then derives a potential fill
event from that cursor's exact raw-tick snapshot.  A separate step resolves the
event against the order's actual terminal cursor.

The active interval is ``(actual_new_send_time_ns, actual_terminal_time_ns]``.
The inclusive upper bound matches the common event-loop phase rule: an
existing working-order fill is processed before same-cursor cancel effects.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import polars as pl

from ..fair_mid.quote_churn import tick_index_to_price as tick_index_to_price_expr
from .targets import tick_index_to_price as scalar_tick_index_to_price

ADAPTER_VERSION = "s1_actual_send_legacy_makerfill_v1"
ENTRY_FILL_TRUTH = "approximate"
ONE_SECOND_NS = 1_000_000_000
PRICE_EPSILON = 1e-8
PACKED_CHANNEL_SEQUENCE_STRIDE = 1 << 48

TargetRank = Literal["BID1", "BID2"]

_INDEX_TICK_REQUIRED_COLUMNS = {
    "ValueCode",
    "ChannelSeq",
    "RecvTime",
    "BidPrice1",
    "BidPrice2",
    "BidLots1",
    "BidLots2",
}


@dataclass(frozen=True, slots=True)
class MakerFillScalarOrder:
    """Minimal immutable actual-new fact consumed by the indexed adapter."""

    raw_order_fact_id: str
    date: str
    value_code: str
    quote_code: str
    absolute_price_tick: int
    maker_snapshot_channel_seq: int
    maker_snapshot_recv_time_ns: int
    actual_new_send_time_ns: int

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_order_fact_id", self.raw_order_fact_id),
            ("date", self.date),
            ("value_code", self.value_code),
            ("quote_code", self.quote_code),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _require_integer(
            self.absolute_price_tick,
            "absolute_price_tick",
            positive=True,
        )
        _require_integer(
            self.maker_snapshot_channel_seq,
            "maker_snapshot_channel_seq",
        )
        _require_integer(
            self.maker_snapshot_recv_time_ns,
            "maker_snapshot_recv_time_ns",
        )
        _require_integer(
            self.actual_new_send_time_ns,
            "actual_new_send_time_ns",
        )

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> MakerFillScalarOrder:
        """Build from one row using the canonical batch-order column names."""

        _require_mapping_fields(record, ORDER_REQUIRED_COLUMNS, "scalar order")
        return cls(
            raw_order_fact_id=record["raw_order_fact_id"],  # type: ignore[arg-type]
            date=record["Date"],  # type: ignore[arg-type]
            value_code=record["ValueCode"],  # type: ignore[arg-type]
            quote_code=record["QuoteCode"],  # type: ignore[arg-type]
            absolute_price_tick=record["absolute_price_tick"],  # type: ignore[arg-type]
            maker_snapshot_channel_seq=record[  # type: ignore[arg-type]
                "maker_snapshot_channel_seq"
            ],
            maker_snapshot_recv_time_ns=record[  # type: ignore[arg-type]
                "maker_snapshot_recv_time_ns"
            ],
            actual_new_send_time_ns=record[  # type: ignore[arg-type]
                "actual_new_send_time_ns"
            ],
        )


@dataclass(frozen=True, slots=True)
class MakerFillScalarEvent:
    """Scalar equivalent of one row from :func:`derive_potential_fill_events`."""

    raw_order_fact_id: str
    date: str
    value_code: str
    quote_code: str
    absolute_price_tick: int
    maker_snapshot_channel_seq: int
    maker_snapshot_recv_time_ns: int
    actual_new_send_time_ns: int
    raw_tick_recv_time_ns: int | None
    snapshot_bid_price1: float | None
    snapshot_bid_price2: float | None
    snapshot_bid_lots1: int | None
    snapshot_bid_lots2: int | None
    target_price: float
    snapshot_recv_time_exact_match: bool
    maker_snapshot_age_ms: float
    exact_target_rank: TargetRank | None
    makerfill_fill_seconds: float | None
    initial_displayed_lots: int | None
    makerfill_column: str | None
    makerfill_potential_fill_time_ns: int | None
    makerfill_mapping_exact: bool
    outcome_supported: bool
    legacy_no_fill_through_eod: bool | None
    makerfill_potential_fill: bool | None
    potential_outcome_status: str
    makerfill_adapter_version: str = ADAPTER_VERSION
    entry_fill_truth: str = ENTRY_FILL_TRUTH
    fill_cursor_exact: bool = False
    own_quantity_included: bool = False
    partial_fill_included: bool = False


@dataclass(frozen=True, slots=True)
class MakerFillObservedSnapshot:
    """Exact raw spot snapshot already frozen by the actual-send resolver."""

    value_code: str
    channel_seq: int
    recv_time_ns: int
    bid_price1: float | None
    bid_price2: float | None
    bid_lots1: int | None
    bid_lots2: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.value_code, str) or not self.value_code:
            raise ValueError("value_code must be a non-empty string")
        _require_integer(self.channel_seq, "channel_seq")
        _require_integer(self.recv_time_ns, "recv_time_ns")
        for name, value in (
            ("bid_price1", self.bid_price1),
            ("bid_price2", self.bid_price2),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite or None")
        for name, value in (
            ("bid_lots1", self.bid_lots1),
            ("bid_lots2", self.bid_lots2),
        ):
            if value is not None:
                _require_integer(value, name)


@dataclass(frozen=True, slots=True)
class _TickSnapshot:
    recv_time_ns: int | None
    bid_price1: float | None
    bid_price2: float | None
    bid_lots1: int | None
    bid_lots2: int | None


@dataclass(frozen=True, slots=True)
class _MakerFillSnapshot:
    bid1_fill_seconds: float | None
    bid2_fill_seconds: float | None


@dataclass(frozen=True, slots=True)
class _ColumnarSnapshotTable:
    """Sorted Polars storage with only one Python span per product.

    A trading day contains millions of source snapshots.  A Python dictionary
    entry and dataclass per row multiplies the resident set by several times,
    even though the underlying columns are already compact.  This table keeps
    the rows columnar and uses binary search inside a product's contiguous
    ``ChannelSeq`` range.
    """

    frame: pl.DataFrame = field(repr=False)
    spans: Mapping[str, tuple[int, int]] = field(repr=False)

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        source: str,
    ) -> _ColumnarSnapshotTable:
        if frame.filter(
            pl.col("ValueCode").is_null()
            | (pl.col("ValueCode").str.len_chars() == 0)
            | pl.col("ChannelSeq").is_null()
        ).height:
            raise ValueError(f"{source} has an invalid snapshot key")
        ordered = frame.sort("ValueCode", "ChannelSeq").rechunk()
        _assert_unique(ordered, ["ValueCode", "ChannelSeq"], source)
        spans: dict[str, tuple[int, int]] = {}
        offset = 0
        for value_code, length in ordered.group_by(
            "ValueCode",
            maintain_order=True,
        ).len().iter_rows():
            spans[value_code] = (offset, length)
            offset += length
        if offset != ordered.height:
            raise RuntimeError(f"{source} product spans do not cover all rows")
        return cls(ordered, MappingProxyType(spans))

    def find_row(self, value_code: str, channel_seq: int) -> int | None:
        span = self.spans.get(value_code)
        if span is None:
            return None
        offset, length = span
        sequences = self.frame["ChannelSeq"].slice(offset, length)
        position = sequences.search_sorted(channel_seq)
        if position >= length or sequences[position] != channel_seq:
            return None
        return offset + position

    @property
    def estimated_size_bytes(self) -> int:
        return self.frame.estimated_size()


@dataclass(frozen=True, slots=True)
class _PackedMakerFillTable:
    """Numeric makerFill lookup used by the full event-loop runner."""

    frame: pl.DataFrame = field(repr=False)
    product_ids: Mapping[str, int] = field(repr=False)

    @classmethod
    def from_frame(cls, makerfill: pl.DataFrame) -> _PackedMakerFillTable:
        _require_columns(makerfill, MAKERFILL_REQUIRED_COLUMNS, "makerFill")
        selected = makerfill.select(
            pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
            pl.col("ChannelSeq").cast(pl.UInt64),
            pl.col("Bid1_FillSeconds").cast(pl.Float32),
            pl.col("Bid2_FillSeconds").cast(pl.Float32),
        )
        if selected.filter(
            pl.col("ValueCode").is_null()
            | (pl.col("ValueCode").str.len_chars() == 0)
            | pl.col("ChannelSeq").is_null()
            | (pl.col("ChannelSeq") >= PACKED_CHANNEL_SEQUENCE_STRIDE)
        ).height:
            raise ValueError("makerFill has an invalid packed snapshot key")
        value_codes = selected["ValueCode"].unique().sort().to_list()
        product_ids = {
            value_code: index for index, value_code in enumerate(value_codes)
        }
        if len(product_ids) >= (1 << 16):
            raise ValueError("makerFill has too many products for packed snapshot keys")
        product_frame = pl.DataFrame(
            {
                "ValueCode": value_codes,
                "_product_id": range(len(value_codes)),
            },
            schema_overrides={"_product_id": pl.UInt64},
        )
        packed = (
            selected.join(
                product_frame,
                on="ValueCode",
                how="left",
                validate="m:1",
            )
            .select(
                (
                    pl.col("_product_id") * PACKED_CHANNEL_SEQUENCE_STRIDE
                    + pl.col("ChannelSeq")
                )
                .cast(pl.UInt64)
                .alias("_packed_key"),
                "Bid1_FillSeconds",
                "Bid2_FillSeconds",
            )
            .sort("_packed_key")
            .rechunk()
        )
        if packed["_packed_key"].is_duplicated().any():
            raise ValueError(
                "makerFill keys are duplicated: ['ValueCode', 'ChannelSeq']"
            )
        return cls(packed, MappingProxyType(product_ids))

    def snapshot(
        self,
        value_code: str,
        channel_seq: int,
    ) -> _MakerFillSnapshot | None:
        product_id = self.product_ids.get(value_code)
        if product_id is None or channel_seq >= PACKED_CHANNEL_SEQUENCE_STRIDE:
            return None
        packed_key = product_id * PACKED_CHANNEL_SEQUENCE_STRIDE + channel_seq
        keys = self.frame["_packed_key"]
        position = keys.search_sorted(packed_key)
        if position >= self.frame.height or keys[position] != packed_key:
            return None
        return _MakerFillSnapshot(
            bid1_fill_seconds=self.frame.item(position, "Bid1_FillSeconds"),
            bid2_fill_seconds=self.frame.item(position, "Bid2_FillSeconds"),
        )

    @property
    def estimated_size_bytes(self) -> int:
        return self.frame.estimated_size()


@dataclass(frozen=True, slots=True)
class MakerFillLabelIndex:
    """Low-memory makerFill index for raw snapshots already held by replay.

    The event loop necessarily has the exact raw spot state at actual send.
    Re-indexing all raw tick snapshots a second time would duplicate hundreds
    of megabytes, so production replay passes that snapshot explicitly and
    this index stores only the two historical makerFill labels.
    """

    _table: _PackedMakerFillTable = field(repr=False)

    @classmethod
    def from_frame(cls, makerfill: pl.DataFrame) -> MakerFillLabelIndex:
        return cls(_PackedMakerFillTable.from_frame(makerfill))

    @property
    def snapshot_count(self) -> int:
        return self._table.frame.height

    @property
    def product_count(self) -> int:
        return len(self._table.product_ids)

    @property
    def estimated_size_bytes(self) -> int:
        return self._table.estimated_size_bytes

    def derive_potential_fill_event(
        self,
        order: MakerFillScalarOrder,
        snapshot: MakerFillObservedSnapshot,
    ) -> MakerFillScalarEvent:
        if not isinstance(order, MakerFillScalarOrder):
            raise TypeError("order must be a MakerFillScalarOrder")
        if not isinstance(snapshot, MakerFillObservedSnapshot):
            raise TypeError("snapshot must be a MakerFillObservedSnapshot")
        if (
            snapshot.value_code != order.value_code
            or snapshot.channel_seq != order.maker_snapshot_channel_seq
            or snapshot.recv_time_ns != order.maker_snapshot_recv_time_ns
        ):
            raise ValueError("observed snapshot does not match actual-new provenance")
        tick = _TickSnapshot(
            recv_time_ns=snapshot.recv_time_ns,
            bid_price1=snapshot.bid_price1,
            bid_price2=snapshot.bid_price2,
            bid_lots1=snapshot.bid_lots1,
            bid_lots2=snapshot.bid_lots2,
        )
        maker = self._table.snapshot(
            order.value_code,
            order.maker_snapshot_channel_seq,
        )
        return _derive_event(order, tick=tick, maker=maker)


@dataclass(frozen=True, slots=True)
class MakerFillSnapshotIndex:
    """Read-only daily snapshot lookup built once for scalar actual-new events."""

    _ticks: _ColumnarSnapshotTable = field(repr=False)
    _makerfill: _ColumnarSnapshotTable = field(repr=False)

    @classmethod
    def from_frames(
        cls,
        tick_snapshots: pl.DataFrame,
        makerfill: pl.DataFrame,
    ) -> MakerFillSnapshotIndex:
        """Build compact columnar indexes from already filtered daily frames."""

        _require_columns(
            tick_snapshots,
            _INDEX_TICK_REQUIRED_COLUMNS,
            "tick snapshots",
        )
        _require_columns(makerfill, MAKERFILL_REQUIRED_COLUMNS, "makerFill")
        tick = tick_snapshots.select(
            pl.col("ValueCode").cast(pl.String),
            pl.col("ChannelSeq").cast(pl.UInt64),
            pl.col("RecvTime").dt.timestamp("ns").alias("raw_tick_recv_time_ns"),
            pl.col("BidPrice1").cast(pl.Float64),
            pl.col("BidPrice2").cast(pl.Float64),
            pl.col("BidLots1").cast(pl.Int64),
            pl.col("BidLots2").cast(pl.Int64),
        )
        fill = makerfill.select(
            pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
            pl.col("ChannelSeq").cast(pl.UInt64),
            pl.col("Bid1_FillSeconds").cast(pl.Float64),
            pl.col("Bid2_FillSeconds").cast(pl.Float64),
        )
        return cls(
            _ColumnarSnapshotTable.from_frame(
                tick,
                source="tick snapshots",
            ),
            _ColumnarSnapshotTable.from_frame(
                fill,
                source="makerFill",
            ),
        )

    @property
    def tick_snapshot_count(self) -> int:
        return self._ticks.frame.height

    @property
    def makerfill_snapshot_count(self) -> int:
        return self._makerfill.frame.height

    @property
    def product_span_count(self) -> int:
        """Number of Python index records, not source snapshot rows."""

        return len(set(self._ticks.spans) | set(self._makerfill.spans))

    @property
    def estimated_size_bytes(self) -> int:
        """Estimated bytes held by the two compact Polars tables."""

        return (
            self._ticks.estimated_size_bytes
            + self._makerfill.estimated_size_bytes
        )

    def _tick_snapshot(
        self,
        value_code: str,
        channel_seq: int,
    ) -> _TickSnapshot | None:
        row_index = self._ticks.find_row(value_code, channel_seq)
        if row_index is None:
            return None
        row = self._ticks.frame.row(row_index)
        return _TickSnapshot(
            recv_time_ns=row[2],
            bid_price1=row[3],
            bid_price2=row[4],
            bid_lots1=row[5],
            bid_lots2=row[6],
        )

    def _makerfill_snapshot(
        self,
        value_code: str,
        channel_seq: int,
    ) -> _MakerFillSnapshot | None:
        row_index = self._makerfill.find_row(value_code, channel_seq)
        if row_index is None:
            return None
        row = self._makerfill.frame.row(row_index)
        return _MakerFillSnapshot(
            bid1_fill_seconds=row[2],
            bid2_fill_seconds=row[3],
        )

    def derive_potential_fill_event(
        self,
        order: MakerFillScalarOrder,
    ) -> MakerFillScalarEvent:
        """Derive one approximate event without another frame join or scan."""

        if not isinstance(order, MakerFillScalarOrder):
            raise TypeError("order must be a MakerFillScalarOrder")
        return _derive_indexed_event(order, self)


ORDER_REQUIRED_COLUMNS = {
    "raw_order_fact_id",
    "Date",
    "ValueCode",
    "QuoteCode",
    "absolute_price_tick",
    "maker_snapshot_channel_seq",
    "maker_snapshot_recv_time_ns",
    "actual_new_send_time_ns",
}
TICK_REQUIRED_COLUMNS = {
    "ValueCode",
    "ChannelSeq",
    "RecvTime",
    "TransTime",
    "BidPrice1",
    "BidPrice2",
    "BidLots1",
    "BidLots2",
}
MAKERFILL_REQUIRED_COLUMNS = {
    "QuoteCode",
    "ChannelSeq",
    "Bid1_FillSeconds",
    "Bid2_FillSeconds",
}


def _require_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _assert_unique(
    frame: pl.DataFrame,
    keys: Sequence[str],
    source: str,
) -> None:
    if frame.select(*keys).n_unique() != frame.height:
        raise ValueError(f"{source} keys are duplicated: {list(keys)}")


def _require_mapping_fields(
    record: Mapping[str, object],
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(record))
    if missing:
        raise ValueError(f"{source} missing required fields: {missing}")


def _require_integer(value: object, name: str, *, positive: bool = False) -> int:
    lower_bound = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _index_key(
    value_code: object,
    channel_seq: object,
    source: str,
) -> tuple[str, int]:
    if not isinstance(value_code, str) or not value_code:
        raise ValueError(f"{source} ValueCode key must be a non-empty string")
    try:
        sequence = _require_integer(channel_seq, "ChannelSeq")
    except ValueError as error:
        raise ValueError(f"{source} has an invalid ChannelSeq key") from error
    return value_code, sequence


def _price_matches(target_price: float, displayed_price: float | None) -> bool:
    return (
        displayed_price is not None
        and math.isfinite(displayed_price)
        and abs(target_price - displayed_price) < PRICE_EPSILON
    )


def _derive_indexed_event(
    order: MakerFillScalarOrder,
    index: MakerFillSnapshotIndex,
) -> MakerFillScalarEvent:
    tick = index._tick_snapshot(
        order.value_code,
        order.maker_snapshot_channel_seq,
    )
    maker = index._makerfill_snapshot(
        order.value_code,
        order.maker_snapshot_channel_seq,
    )
    return _derive_event(order, tick=tick, maker=maker)


def _derive_event(
    order: MakerFillScalarOrder,
    *,
    tick: _TickSnapshot | None,
    maker: _MakerFillSnapshot | None,
) -> MakerFillScalarEvent:
    target_price = scalar_tick_index_to_price(
        order.absolute_price_tick,
        market="spot",
    )
    tick_missing = tick is None
    raw_tick_recv_time_ns = None if tick is None else tick.recv_time_ns
    recv_exact = (
        raw_tick_recv_time_ns is not None
        and raw_tick_recv_time_ns == order.maker_snapshot_recv_time_ns
    )
    snapshot_age_ms = (
        order.actual_new_send_time_ns - order.maker_snapshot_recv_time_ns
    ) / 1_000_000.0
    snapshot_after_send = snapshot_age_ms < 0

    target_rank: TargetRank | None = None
    if tick is not None:
        if _price_matches(target_price, tick.bid_price1):
            target_rank = "BID1"
        elif _price_matches(target_price, tick.bid_price2):
            target_rank = "BID2"

    fill_seconds: float | None = None
    displayed_lots: int | None = None
    makerfill_column: str | None = None
    if target_rank == "BID1":
        displayed_lots = None if tick is None else tick.bid_lots1
        makerfill_column = "Bid1_FillSeconds"
        fill_seconds = None if maker is None else maker.bid1_fill_seconds
    elif target_rank == "BID2":
        displayed_lots = None if tick is None else tick.bid_lots2
        makerfill_column = "Bid2_FillSeconds"
        fill_seconds = None if maker is None else maker.bid2_fill_seconds

    fill_nan = fill_seconds is not None and math.isnan(fill_seconds)
    fill_finite = fill_seconds is not None and math.isfinite(fill_seconds)
    fill_finite_nonnegative = fill_finite and fill_seconds >= 0
    potential_fill_time_ns: int | None = None
    if fill_finite:
        potential_fill_time_ns = order.maker_snapshot_recv_time_ns + round(
            fill_seconds * ONE_SECOND_NS
        )
    implied_after_actual_new = (
        potential_fill_time_ns is not None
        and potential_fill_time_ns > order.actual_new_send_time_ns
    )
    rank_missing = target_rank is None
    makerfill_missing = maker is None
    outcome_supported = (
        not tick_missing
        and not makerfill_missing
        and recv_exact
        and not snapshot_after_send
        and not rank_missing
        and (fill_nan or (fill_finite_nonnegative and implied_after_actual_new))
    )
    potential_fill = outcome_supported and fill_finite_nonnegative

    if tick_missing:
        status = "unsupported_missing_raw_tick_snapshot"
    elif not recv_exact:
        status = "unsupported_snapshot_recv_time_mismatch"
    elif snapshot_after_send:
        status = "unsupported_snapshot_after_actual_new"
    elif rank_missing:
        status = "unsupported_target_not_displayed_l1_l2"
    elif makerfill_missing:
        status = "unsupported_missing_makerfill_key"
    elif fill_nan:
        status = "approx_no_fill_through_eod"
    elif not fill_finite_nonnegative:
        status = "unsupported_invalid_fill_seconds"
    elif not implied_after_actual_new:
        status = "unsupported_fill_not_after_actual_new"
    else:
        status = "approx_potential_fill"

    return MakerFillScalarEvent(
        raw_order_fact_id=order.raw_order_fact_id,
        date=order.date,
        value_code=order.value_code,
        quote_code=order.quote_code,
        absolute_price_tick=order.absolute_price_tick,
        maker_snapshot_channel_seq=order.maker_snapshot_channel_seq,
        maker_snapshot_recv_time_ns=order.maker_snapshot_recv_time_ns,
        actual_new_send_time_ns=order.actual_new_send_time_ns,
        raw_tick_recv_time_ns=raw_tick_recv_time_ns,
        snapshot_bid_price1=None if tick is None else tick.bid_price1,
        snapshot_bid_price2=None if tick is None else tick.bid_price2,
        snapshot_bid_lots1=None if tick is None else tick.bid_lots1,
        snapshot_bid_lots2=None if tick is None else tick.bid_lots2,
        target_price=target_price,
        snapshot_recv_time_exact_match=recv_exact,
        maker_snapshot_age_ms=snapshot_age_ms,
        exact_target_rank=target_rank,
        makerfill_fill_seconds=fill_seconds,
        initial_displayed_lots=displayed_lots,
        makerfill_column=makerfill_column,
        makerfill_potential_fill_time_ns=potential_fill_time_ns,
        makerfill_mapping_exact=not tick_missing and not rank_missing,
        outcome_supported=outcome_supported,
        legacy_no_fill_through_eod=fill_nan if outcome_supported else None,
        makerfill_potential_fill=potential_fill if outcome_supported else None,
        potential_outcome_status=status,
    )


def derive_potential_fill_events(
    actual_orders: pl.DataFrame,
    tick_snapshots: pl.DataFrame,
    makerfill: pl.DataFrame,
) -> pl.DataFrame:
    """Derive one potential approximate fill event per actual-sent order.

    ``actual_orders`` must already contain the physical identity and the exact
    as-of spot snapshot frozen at the actual new-send cursor.  Unsupported
    joins, ranks, clocks, or labels are retained with a fail-closed status.
    No cancel/expiry cursor is consulted here, so the adapter cannot silently
    recreate a nominal order window.
    """

    _require_columns(actual_orders, ORDER_REQUIRED_COLUMNS, "actual orders")
    _require_columns(tick_snapshots, TICK_REQUIRED_COLUMNS, "tick snapshots")
    _require_columns(makerfill, MAKERFILL_REQUIRED_COLUMNS, "makerFill")
    _assert_unique(actual_orders, ["raw_order_fact_id"], "actual orders")

    snapshot_keys = ["ValueCode", "maker_snapshot_channel_seq"]
    tick = tick_snapshots.select(
        pl.col("ValueCode").cast(pl.String),
        pl.col("ChannelSeq").cast(pl.UInt64).alias("maker_snapshot_channel_seq"),
        pl.col("RecvTime").dt.timestamp("ns").alias("raw_tick_recv_time_ns"),
        pl.col("TransTime").dt.timestamp("us").alias("raw_tick_trans_time_us"),
        pl.col("BidPrice1").cast(pl.Float64).alias("snapshot_bid_price1"),
        pl.col("BidPrice2").cast(pl.Float64).alias("snapshot_bid_price2"),
        pl.col("BidLots1").cast(pl.Int64).alias("snapshot_bid_lots1"),
        pl.col("BidLots2").cast(pl.Int64).alias("snapshot_bid_lots2"),
        pl.lit(True).alias("_raw_tick_snapshot_found"),
    )
    maker = makerfill.select(
        pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
        pl.col("ChannelSeq").cast(pl.UInt64).alias("maker_snapshot_channel_seq"),
        pl.col("Bid1_FillSeconds").cast(pl.Float64),
        pl.col("Bid2_FillSeconds").cast(pl.Float64),
        pl.lit(True).alias("_makerfill_snapshot_found"),
    )
    _assert_unique(tick, snapshot_keys, "tick snapshots")
    _assert_unique(maker, snapshot_keys, "makerFill snapshots")

    joined = (
        actual_orders.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("maker_snapshot_channel_seq").cast(pl.UInt64),
            pl.col("maker_snapshot_recv_time_ns").cast(pl.Int64),
            pl.col("actual_new_send_time_ns").cast(pl.Int64),
            pl.col("absolute_price_tick").cast(pl.Int64),
        )
        .join(tick, on=snapshot_keys, how="left", validate="m:1")
        .join(maker, on=snapshot_keys, how="left", validate="m:1")
        .with_columns(
            tick_index_to_price_expr(
                pl.col("absolute_price_tick"), market="spot"
            ).alias("target_price")
        )
        .with_columns(
            (pl.col("raw_tick_recv_time_ns") == pl.col("maker_snapshot_recv_time_ns"))
            .fill_null(False)
            .alias("snapshot_recv_time_exact_match"),
            (pl.col("actual_new_send_time_ns") - pl.col("maker_snapshot_recv_time_ns"))
            .truediv(1_000_000.0)
            .alias("maker_snapshot_age_ms"),
            pl.when(
                (pl.col("target_price") - pl.col("snapshot_bid_price1")).abs()
                < PRICE_EPSILON
            )
            .then(pl.lit("BID1"))
            .when(
                (pl.col("target_price") - pl.col("snapshot_bid_price2")).abs()
                < PRICE_EPSILON
            )
            .then(pl.lit("BID2"))
            .otherwise(None)
            .alias("exact_target_rank"),
        )
        .with_columns(
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.col("Bid1_FillSeconds"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.col("Bid2_FillSeconds"))
            .otherwise(None)
            .alias("makerfill_fill_seconds"),
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.col("snapshot_bid_lots1"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.col("snapshot_bid_lots2"))
            .otherwise(None)
            .alias("initial_displayed_lots"),
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.lit("Bid1_FillSeconds"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.lit("Bid2_FillSeconds"))
            .otherwise(None)
            .alias("makerfill_column"),
        )
        .with_columns(
            (
                pl.col("maker_snapshot_recv_time_ns")
                + (
                    pl.when(pl.col("makerfill_fill_seconds").is_finite())
                    .then(pl.col("makerfill_fill_seconds"))
                    .otherwise(None)
                    .cast(pl.Float64)
                    * ONE_SECOND_NS
                )
                .round(0)
                .cast(pl.Int64)
            ).alias("makerfill_potential_fill_time_ns")
        )
    )

    tick_missing = ~pl.col("_raw_tick_snapshot_found").fill_null(False)
    makerfill_missing = ~pl.col("_makerfill_snapshot_found").fill_null(False)
    recv_mismatch = ~pl.col("snapshot_recv_time_exact_match")
    snapshot_after_send = (pl.col("maker_snapshot_age_ms") < 0).fill_null(True)
    rank_missing = pl.col("exact_target_rank").is_null()
    fill_nan = pl.col("makerfill_fill_seconds").is_nan().fill_null(False)
    fill_finite_nonnegative = (
        pl.col("makerfill_fill_seconds").is_finite()
        & (pl.col("makerfill_fill_seconds") >= 0)
    ).fill_null(False)
    implied_after_actual_new = (
        pl.col("makerfill_potential_fill_time_ns") > pl.col("actual_new_send_time_ns")
    ).fill_null(False)
    outcome_supported = (
        ~tick_missing
        & ~makerfill_missing
        & ~recv_mismatch
        & ~snapshot_after_send
        & ~rank_missing
        & (fill_nan | (fill_finite_nonnegative & implied_after_actual_new))
    ).fill_null(False)
    potential_fill = (outcome_supported & fill_finite_nonnegative).fill_null(False)

    return (
        joined.with_columns(
            (~tick_missing & ~rank_missing)
            .fill_null(False)
            .alias("makerfill_mapping_exact"),
            outcome_supported.alias("outcome_supported"),
            pl.when(outcome_supported)
            .then(fill_nan)
            .otherwise(None)
            .alias("legacy_no_fill_through_eod"),
            pl.when(outcome_supported)
            .then(potential_fill)
            .otherwise(None)
            .alias("makerfill_potential_fill"),
            pl.when(tick_missing)
            .then(pl.lit("unsupported_missing_raw_tick_snapshot"))
            .when(recv_mismatch)
            .then(pl.lit("unsupported_snapshot_recv_time_mismatch"))
            .when(snapshot_after_send)
            .then(pl.lit("unsupported_snapshot_after_actual_new"))
            .when(rank_missing)
            .then(pl.lit("unsupported_target_not_displayed_l1_l2"))
            .when(makerfill_missing)
            .then(pl.lit("unsupported_missing_makerfill_key"))
            .when(fill_nan)
            .then(pl.lit("approx_no_fill_through_eod"))
            .when(~fill_finite_nonnegative)
            .then(pl.lit("unsupported_invalid_fill_seconds"))
            .when(~implied_after_actual_new)
            .then(pl.lit("unsupported_fill_not_after_actual_new"))
            .otherwise(pl.lit("approx_potential_fill"))
            .alias("potential_outcome_status"),
            pl.lit(ADAPTER_VERSION).alias("makerfill_adapter_version"),
            pl.lit(ENTRY_FILL_TRUTH).alias("entry_fill_truth"),
            pl.lit(False).alias("fill_cursor_exact"),
            pl.lit(False).alias("own_quantity_included"),
            pl.lit(False).alias("partial_fill_included"),
        )
        .drop(
            "Bid1_FillSeconds",
            "Bid2_FillSeconds",
            "_raw_tick_snapshot_found",
            "_makerfill_snapshot_found",
        )
        .sort(["Date", "ValueCode", "raw_order_fact_id"])
    )


def classify_actual_active_intervals(
    potential_events: pl.DataFrame,
    *,
    terminal_time_column: str = "actual_terminal_time_ns",
) -> pl.DataFrame:
    """Resolve potential events against actual cancel/expiry lifecycles."""

    required = {
        "raw_order_fact_id",
        "actual_new_send_time_ns",
        "makerfill_potential_fill_time_ns",
        "makerfill_potential_fill",
        "outcome_supported",
        "potential_outcome_status",
        terminal_time_column,
    }
    _require_columns(potential_events, required, "potential fill events")
    _assert_unique(potential_events, ["raw_order_fact_id"], "potential fill events")
    invalid_terminal = potential_events.filter(
        pl.col(terminal_time_column).is_null()
        | (
            pl.col(terminal_time_column).cast(pl.Int64)
            <= pl.col("actual_new_send_time_ns").cast(pl.Int64)
        )
    )
    if not invalid_terminal.is_empty():
        raise ValueError("actual terminal cursor must be after actual new send")

    supported = pl.col("outcome_supported").fill_null(False)
    potential_fill = pl.col("makerfill_potential_fill").fill_null(False)
    fill_inside = (
        supported
        & potential_fill
        & (pl.col("makerfill_potential_fill_time_ns") <= pl.col(terminal_time_column))
    ).fill_null(False)

    return potential_events.with_columns(
        pl.when(supported)
        .then(fill_inside)
        .otherwise(None)
        .alias("approximate_full_fill_within_actual_interval"),
        pl.when(fill_inside)
        .then(pl.col("makerfill_potential_fill_time_ns"))
        .otherwise(None)
        .alias("accepted_approximate_fill_time_ns"),
        pl.when(~supported)
        .then(pl.col("potential_outcome_status"))
        .when(~potential_fill)
        .then(pl.lit("approx_no_fill_through_eod"))
        .when(fill_inside)
        .then(pl.lit("approx_fill_within_actual_interval"))
        .otherwise(pl.lit("actual_terminal_before_later_approx_fill"))
        .alias("actual_interval_outcome_status"),
        pl.lit("(actual_new_send_time_ns, actual_terminal_time_ns]").alias(
            "active_interval_contract"
        ),
    )


__all__ = [
    "ADAPTER_VERSION",
    "ENTRY_FILL_TRUTH",
    "MakerFillLabelIndex",
    "MakerFillObservedSnapshot",
    "MakerFillScalarEvent",
    "MakerFillScalarOrder",
    "MakerFillSnapshotIndex",
    "classify_actual_active_intervals",
    "derive_potential_fill_events",
]
