"""Compact causal raw-book index for S1 B6 hedge and rollback replay.

The input may contain millions of already-selected spot/future raw rows. The
index keeps only rows that can affect causal book state: TrialMatch barriers
and formal L1/Best snapshots. Non-book trade/status rows are intentionally
absent from ``event_as_of`` and ``iter_raw_changes`` in this compact-v2 API.

No input row becomes a Python ``dict`` or dataclass during construction.
Metadata lives in one compact Polars clock, while market-native L1 and Best
payloads live in separate columnar frames. Immutable ``RawBookEvent`` and
``CausalBookState`` objects are created only for rows a caller asks for.

Exchange ``ChannelSeq`` values are venue-local provenance, not a cross-market
clock. Retained events are merged by receive time, fixed venue rank, product,
channel, packet, and source row. Loop cursors use raw phase zero and put the
same-time merged ordinal in ``EventCursor.row_index``.

TrialMatch wins when a row also has non-zero book payload. L1 and Best are
independent snapshot bundles: an update replaces its entire bundle, including
clearing the absent side, and cannot forward-fill through a TrialMatch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import polars as pl

from .layered import EventCursor
from .s1_hedge import (
    CausalBookState,
    RawBookCursor,
    RawBookEvent,
    RawBookLevel,
    RawBookStateMachine,
)

Market = Literal["spot", "future"]
QueryBoundary = int | EventCursor | RawBookCursor
SelectedFrame = pl.DataFrame | pl.LazyFrame

_MAPPING_COLUMNS: Final = {
    "ValueCode",
    "QuoteCode",
    "spot_ref_price",
    "fut_ref_price",
}
_RAW_COLUMNS: Final = {
    "RecvTime",
    "QuoteCode",
    "PacketSeq",
    "ChannelSeq",
    "TrialMatch",
    *(f"BidPrice{level}" for level in range(1, 6)),
    *(f"BidLots{level}" for level in range(1, 6)),
    *(f"AskPrice{level}" for level in range(1, 6)),
    *(f"AskLots{level}" for level in range(1, 6)),
    "BestBidPrice",
    "BestBidLots",
    "BestAskPrice",
    "BestAskLots",
}
_MARKET_ORDER: Final[Mapping[Market, int]] = MappingProxyType({"spot": 0, "future": 1})
_CLOCK_COLUMNS: Final = (
    "series_id",
    "recv_time_ns",
    "loop_row_index",
    "channel_sequence",
    "packet_sequence",
    "source_row",
    "trial_match",
    "own_l1_payload_id",
    "current_l1_payload_id",
    "current_best_payload_id",
)
_EFFECTIVE_CLOCK_COLUMNS: Final = (
    "series_id",
    "clock_position",
)
_EXIT_QUOTE_CLOCK_COLUMNS: Final = (
    "series_id",
    "clock_position",
)
_L1_PAYLOAD_COLUMNS: Final = tuple(
    f"{side}_{kind}_{level}"
    for side in ("bid", "ask")
    for kind in ("price", "lots")
    for level in range(1, 6)
)
_BEST_PAYLOAD_COLUMNS: Final = (
    "best_bid_price",
    "best_bid_lots",
    "best_ask_price",
    "best_ask_lots",
)


@dataclass(frozen=True, order=True, slots=True)
class RawBookKey:
    market: Market
    value_code: str
    quote_code: str

    def __post_init__(self) -> None:
        if self.market not in ("spot", "future"):
            raise ValueError("market must be spot or future")
        for name, value in (
            ("value_code", self.value_code),
            ("quote_code", self.quote_code),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RawBookSourceCursor:
    """Immutable venue-local provenance; deliberately not orderable."""

    recv_time_ns: int
    channel_sequence: int
    packet_sequence: int
    source_row: int

    def __post_init__(self) -> None:
        for name, value in (
            ("recv_time_ns", self.recv_time_ns),
            ("channel_sequence", self.channel_sequence),
            ("packet_sequence", self.packet_sequence),
            ("source_row", self.source_row),
        ):
            _nonnegative_integer(value, name)


@dataclass(frozen=True, slots=True)
class IndexedRawBookEvent:
    """One lazily reconstructed event and its source provenance."""

    event: RawBookEvent
    source_cursor: RawBookSourceCursor

    def __post_init__(self) -> None:
        if not isinstance(self.event, RawBookEvent):
            raise TypeError("event must be a RawBookEvent")
        if not isinstance(self.source_cursor, RawBookSourceCursor):
            raise TypeError("source_cursor must be a RawBookSourceCursor")
        book_cursor = self.event.book_cursor
        if book_cursor.cursor.recv_time_ns != self.source_cursor.recv_time_ns:
            raise ValueError("loop and source receive times must match")
        if book_cursor.packet_sequence != self.source_cursor.packet_sequence:
            raise ValueError("book and source packet sequences must match")


@dataclass(frozen=True, slots=True)
class _IndexedIntervalCache:
    relative: int
    indexed: IndexedRawBookEvent | None


@dataclass(frozen=True, slots=True)
class _StateIntervalCache:
    relative: int
    state: CausalBookState | None


@dataclass(frozen=True, slots=True)
class SpotSourceSnapshot:
    """Exact compact spot snapshot for scalar actual-send/makerFill lookup."""

    key: RawBookKey
    source_cursor: RawBookSourceCursor
    bid_price1: float
    bid_price2: float
    bid_lots1: int
    bid_lots2: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, RawBookKey) or self.key.market != "spot":
            raise ValueError("spot source snapshot requires a spot RawBookKey")
        if not isinstance(self.source_cursor, RawBookSourceCursor):
            raise TypeError("source_cursor must be a RawBookSourceCursor")
        for name, value in (
            ("bid_price1", self.bid_price1),
            ("bid_price2", self.bid_price2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        _nonnegative_integer(self.bid_lots1, "bid_lots1")
        _nonnegative_integer(self.bid_lots2, "bid_lots2")


@dataclass(frozen=True, slots=True)
class _MappingPair:
    pair_id: int
    value_code: str
    quote_code: str
    spot_reference: float | None
    future_reference: float | None

    def key(self, market: Market) -> RawBookKey:
        return RawBookKey(market, self.value_code, self.quote_code)

    def series_id(self, market: Market) -> int:
        return self.pair_id * 2 + _MARKET_ORDER[market]

    def reference(self, market: Market) -> float | None:
        return self.spot_reference if market == "spot" else self.future_reference


@dataclass(frozen=True, slots=True)
class _BookSeries:
    series_id: int
    position_start: int
    length: int
    effective_position_start: int
    effective_length: int
    exit_quote_position_start: int
    exit_quote_length: int
    spot_lookup_start: int
    spot_lookup_length: int
    reference_price: float | None


@dataclass(frozen=True, slots=True)
class _MarketPayloads:
    l1: pl.DataFrame = field(repr=False)
    best: pl.DataFrame = field(repr=False)


@dataclass(frozen=True, slots=True)
class _MarketBuild:
    clock: pl.DataFrame = field(repr=False)
    l1: pl.DataFrame = field(repr=False)
    best: pl.DataFrame = field(repr=False)


@dataclass(frozen=True, slots=True)
class RawBookDayIndex:
    """Immutable compact day index with O(log n) causal boundary lookup."""

    _clock: pl.DataFrame = field(repr=False)
    _effective_clock: pl.DataFrame = field(repr=False)
    _exit_quote_clock: pl.DataFrame = field(repr=False)
    _spot_lookup: pl.DataFrame = field(repr=False)
    _series: Mapping[RawBookKey, _BookSeries] = field(repr=False)
    _payloads: Mapping[Market, _MarketPayloads] = field(repr=False)
    _key_frame: pl.DataFrame = field(repr=False)
    _clock_recv_time_ns: object = field(init=False, repr=False, compare=False)
    _clock_loop_row_index: object = field(init=False, repr=False, compare=False)
    _clock_packet_sequence: object = field(init=False, repr=False, compare=False)
    _effective_clock_positions: object = field(init=False, repr=False, compare=False)
    _exit_quote_clock_positions: object = field(init=False, repr=False, compare=False)
    _indexed_interval_cache: dict[RawBookKey, _IndexedIntervalCache] = field(
        init=False, repr=False, compare=False
    )
    _state_interval_cache: dict[RawBookKey, _StateIntervalCache] = field(
        init=False, repr=False, compare=False
    )
    _query_cache_counts: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_columns(self._clock, set(_CLOCK_COLUMNS), "compact event clock")
        _require_columns(
            self._effective_clock,
            set(_EFFECTIVE_CLOCK_COLUMNS),
            "effective-state change clock",
        )
        _require_columns(
            self._exit_quote_clock,
            set(_EXIT_QUOTE_CLOCK_COLUMNS),
            "normal-exit quote change clock",
        )
        _require_columns(
            self._spot_lookup,
            {"channel_sequence", "clock_position"},
            "compact spot source lookup",
        )
        copied_series = dict(self._series)
        if any(
            not isinstance(key, RawBookKey) or not isinstance(value, _BookSeries)
            for key, value in copied_series.items()
        ):
            raise TypeError("raw-book index has invalid series metadata")
        copied_payloads = dict(self._payloads)
        if set(copied_payloads) != {"spot", "future"}:
            raise ValueError("raw-book index requires both market payload stores")
        if any(
            not isinstance(payloads, _MarketPayloads)
            for payloads in copied_payloads.values()
        ):
            raise TypeError("raw-book payload store is invalid")
        object.__setattr__(self, "_series", MappingProxyType(copied_series))
        object.__setattr__(self, "_payloads", MappingProxyType(copied_payloads))
        object.__setattr__(
            self,
            "_clock_recv_time_ns",
            _packed_integer_column(self._clock, "recv_time_ns"),
        )
        object.__setattr__(
            self,
            "_clock_loop_row_index",
            _packed_integer_column(self._clock, "loop_row_index"),
        )
        object.__setattr__(
            self,
            "_clock_packet_sequence",
            _packed_integer_column(self._clock, "packet_sequence"),
        )
        object.__setattr__(
            self,
            "_effective_clock_positions",
            _packed_integer_column(self._effective_clock, "clock_position"),
        )
        object.__setattr__(
            self,
            "_exit_quote_clock_positions",
            _packed_integer_column(self._exit_quote_clock, "clock_position"),
        )
        object.__setattr__(self, "_indexed_interval_cache", {})
        object.__setattr__(self, "_state_interval_cache", {})
        object.__setattr__(
            self,
            "_query_cache_counts",
            {
                "indexed_hits": 0,
                "indexed_misses": 0,
                "state_hits": 0,
                "state_misses": 0,
            },
        )

    @classmethod
    def from_selected_rows(
        cls,
        spot_rows: pl.DataFrame,
        future_rows: pl.DataFrame,
        mapping: pl.DataFrame,
    ) -> RawBookDayIndex:
        """Compatibility constructor for materialized selected rows."""

        if not isinstance(spot_rows, pl.DataFrame) or not isinstance(
            future_rows, pl.DataFrame
        ):
            raise TypeError("from_selected_rows requires Polars DataFrames")
        return cls.from_selected_scans(spot_rows.lazy(), future_rows.lazy(), mapping)

    @classmethod
    def from_selected_scans(
        cls,
        spot_rows: SelectedFrame,
        future_rows: SelectedFrame,
        mapping: pl.DataFrame,
    ) -> RawBookDayIndex:
        """Build directly from eager frames or lazy scans without row objects."""

        pairs, key_frame = _mapping_pairs(mapping)
        spot = _build_market(
            _as_lazy(spot_rows, "spot_rows"), pairs, key_frame, market="spot"
        )
        future = _build_market(
            _as_lazy(future_rows, "future_rows"),
            pairs,
            key_frame,
            market="future",
        )
        clock = _merge_clocks(spot.clock, future.clock)
        payloads = {
            "spot": _MarketPayloads(spot.l1, spot.best),
            "future": _MarketPayloads(future.l1, future.best),
        }
        effective_clock = _effective_change_clock(clock, payloads)
        exit_quote_clock = _exit_quote_change_clock(clock, payloads, pairs)
        spot_lookup = _spot_source_lookup(clock)
        return cls(
            clock,
            effective_clock,
            exit_quote_clock,
            spot_lookup.select("channel_sequence", "clock_position"),
            _series_metadata(
                pairs,
                clock,
                effective_clock,
                exit_quote_clock,
                spot_lookup,
            ),
            payloads,
            key_frame,
        )

    @property
    def keys(self) -> tuple[RawBookKey, ...]:
        return tuple(sorted(self._series))

    @property
    def retained_event_count(self) -> int:
        return self._clock.height

    @property
    def effective_change_count(self) -> int:
        """Number of retained rows that alter the forward-filled book state."""

        return self._effective_clock.height

    @property
    def exit_quote_change_count(self) -> int:
        """Number of normal Spot-Ask route decision-state transitions."""

        return self._exit_quote_clock.height

    @property
    def query_cache_counts(self) -> Mapping[str, int]:
        """Diagnostic exact-query cache counters for profiling and tests."""

        return MappingProxyType(dict(self._query_cache_counts))

    @property
    def spot_source_snapshot_count(self) -> int:
        return self._clock.filter(
            (pl.col("series_id") % 2 == 0) & ~pl.col("trial_match")
        ).height

    @property
    def estimated_size_bytes(self) -> int:
        """Estimated in-memory bytes for all canonical compact frames."""

        return sum(frame.estimated_size() for frame in self.storage_frames().values())

    def storage_frames(self) -> Mapping[str, pl.DataFrame]:
        """Return shallow frame clones for deterministic serialization."""

        return MappingProxyType(
            {
                "clock": self._clock.clone(),
                "effective_clock": self._effective_clock.clone(),
                "exit_quote_clock": self._exit_quote_clock.clone(),
                "spot_lookup": self._spot_lookup.clone(),
                "spot_l1": self._payloads["spot"].l1.clone(),
                "spot_best": self._payloads["spot"].best.clone(),
                "future_l1": self._payloads["future"].l1.clone(),
                "future_best": self._payloads["future"].best.clone(),
                "keys": self._key_frame.clone(),
            }
        )

    def as_risk_book_adapter(self) -> RawBookRiskAdapter:
        """Expose the event-loop RiskBookAdapter contract using ValueCode IDs."""

        return RawBookRiskAdapter(self)

    def spot_source_snapshots(self) -> pl.DataFrame:
        """Materialize formal-spot provenance for actual-send/makerFill.

        The schema is directly consumable by ``MakerFillSnapshotIndex``;
        PacketSeq and source_row remain as additional audit provenance. This
        large projection is created only when the runner needs it.
        """

        source = self._clock.filter(
            (pl.col("series_id") % 2 == 0) & ~pl.col("trial_match")
        ).select(
            "series_id",
            "recv_time_ns",
            "channel_sequence",
            "packet_sequence",
            "source_row",
            "own_l1_payload_id",
        )
        payload = (
            self._payloads["spot"]
            .l1.with_row_index("own_l1_payload_id")
            .select(
                "own_l1_payload_id",
                pl.col("bid_price_1").alias("BidPrice1"),
                pl.col("bid_price_2").alias("BidPrice2"),
                pl.col("bid_lots_1").alias("BidLots1"),
                pl.col("bid_lots_2").alias("BidLots2"),
            )
        )
        return (
            source.join(
                payload,
                on="own_l1_payload_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .join(
                self._key_frame.select("series_id", "ValueCode", "QuoteCode"),
                on="series_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .select(
                "ValueCode",
                "QuoteCode",
                pl.from_epoch("recv_time_ns", time_unit="ns").alias("RecvTime"),
                pl.col("channel_sequence").alias("ChannelSeq"),
                pl.col("packet_sequence").alias("PacketSeq"),
                "source_row",
                pl.col("BidPrice1").fill_null(0.0),
                pl.col("BidPrice2").fill_null(0.0),
                pl.col("BidLots1").fill_null(0),
                pl.col("BidLots2").fill_null(0),
            )
        )

    def spot_source_snapshot(
        self,
        key: RawBookKey,
        channel_sequence: int,
        *,
        recv_time_ns: int | None = None,
    ) -> SpotSourceSnapshot | None:
        """Lookup one exact spot source in O(log n) without a Python day map."""

        if not isinstance(key, RawBookKey) or key.market != "spot":
            raise ValueError("spot source lookup requires a spot RawBookKey")
        target = _nonnegative_integer(channel_sequence, "channel_sequence")
        if recv_time_ns is not None:
            _nonnegative_integer(recv_time_ns, "recv_time_ns")
        series = self._require_series(key)
        low = series.spot_lookup_start
        high = series.spot_lookup_start + series.spot_lookup_length
        while low < high:
            middle = (low + high) // 2
            value = int(self._spot_lookup["channel_sequence"][middle])
            if value < target:
                low = middle + 1
            else:
                high = middle
        matches: list[int] = []
        end = series.spot_lookup_start + series.spot_lookup_length
        while low < end and int(self._spot_lookup["channel_sequence"][low]) == target:
            position = int(self._spot_lookup["clock_position"][low])
            actual_time = int(self._clock["recv_time_ns"][position])
            if recv_time_ns is None or actual_time == recv_time_ns:
                matches.append(position)
                if len(matches) > 1:
                    return None
            low += 1
        if len(matches) != 1:
            return None
        position = matches[0]
        actual_recv_time_ns = int(self._clock["recv_time_ns"][position])
        payload_id = self._clock["own_l1_payload_id"][position]
        bid_price1 = 0.0
        bid_price2 = 0.0
        bid_lots1 = 0
        bid_lots2 = 0
        if payload_id is not None:
            payload = self._payloads["spot"].l1.row(int(payload_id), named=True)
            bid_price1 = float(payload["bid_price_1"])
            bid_price2 = float(payload["bid_price_2"])
            bid_lots1 = int(payload["bid_lots_1"])
            bid_lots2 = int(payload["bid_lots_2"])
        return SpotSourceSnapshot(
            key,
            RawBookSourceCursor(
                actual_recv_time_ns,
                target,
                int(self._clock["packet_sequence"][position]),
                int(self._clock["source_row"][position]),
            ),
            bid_price1,
            bid_price2,
            bid_lots1,
            bid_lots2,
        )

    def event_count(self, key: RawBookKey) -> int:
        return self._require_series(key).length

    def effective_event_count(self, key: RawBookKey) -> int:
        """Return the derived state-changing row count for one exact series."""

        return self._require_series(key).effective_length

    def exit_quote_event_count(self, key: RawBookKey) -> int:
        """Return route-specific normal-exit wake count for one series."""

        return self._require_series(key).exit_quote_length

    def event_as_of(
        self,
        key: RawBookKey,
        at: QueryBoundary,
    ) -> RawBookEvent | None:
        indexed = self.indexed_event_as_of(key, at)
        return None if indexed is None else indexed.event

    def indexed_event_as_of(
        self,
        key: RawBookKey,
        at: QueryBoundary,
    ) -> IndexedRawBookEvent | None:
        series = self._require_series(key)
        boundary_kind, target = _query_boundary_parts(at)
        cached = self._indexed_interval_cache.get(key)
        if cached is not None and self._relative_contains(
            series,
            cached.relative,
            boundary_kind,
            target,
        ):
            self._query_cache_counts["indexed_hits"] += 1
            return cached.indexed
        self._query_cache_counts["indexed_misses"] += 1
        relative = self._right_index(series, at) - 1
        if relative < 0:
            indexed = None
        else:
            indexed = self._indexed_event(self._clock_position(series, relative), key)
        self._indexed_interval_cache[key] = _IndexedIntervalCache(relative, indexed)
        return indexed

    def source_cursor_as_of(
        self,
        key: RawBookKey,
        at: QueryBoundary,
    ) -> RawBookSourceCursor | None:
        indexed = self.indexed_event_as_of(key, at)
        return None if indexed is None else indexed.source_cursor

    def next_indexed_event(
        self,
        key: RawBookKey,
        after: QueryBoundary,
    ) -> IndexedRawBookEvent | None:
        """Return the first retained state-changing event strictly after ``after``."""

        series = self._require_series(key)
        relative = self._right_index(series, after)
        if relative >= series.length:
            return None
        return self._indexed_event(self._clock_position(series, relative), key)

    def next_change_cursor(
        self,
        key: RawBookKey,
        after: QueryBoundary,
    ) -> RawBookCursor | None:
        """Return the next Trial/formal-book cursor without materializing a range."""

        indexed = self.next_indexed_event(key, after)
        return None if indexed is None else indexed.event.book_cursor

    def next_effective_change_cursor(
        self,
        key: RawBookKey,
        after: QueryBoundary,
    ) -> RawBookCursor | None:
        """Return the first effective book-state change strictly after ``after``.

        Unlike :meth:`next_change_cursor`, this derived clock suppresses a
        formal snapshot when its forward-filled L1 and Best bundles are
        exactly equal to the preceding formal state.  Trial/formal mode
        transitions and actual payload changes are always retained.  The
        returned cursor still belongs to the original raw clock, so
        ``state_as_of`` and source provenance remain exact.
        """

        series = self._require_series(key)
        relative = self._effective_right_index(series, after)
        if relative >= series.effective_length:
            return None
        effective_position = series.effective_position_start + relative
        clock_position = int(self._effective_clock_positions[effective_position])
        return self._indexed_event(clock_position, key).event.book_cursor

    def next_exit_quote_change_cursor(
        self,
        key: RawBookKey,
        after: QueryBoundary,
    ) -> RawBookCursor | None:
        """Return the next normal Spot-Ask exit decision-state change.

        Spot rows wake on Trial/formal transitions or normalized BBO
        price/presence changes.  This is sufficient for every frozen lower
        threshold because the Spot contribution to scheduler-visible desired
        tick/gate is only the BBO gate and target passivity; actual-send queue
        and source provenance still come from the full ``state_as_of`` book.

        Future rows implement the frozen one-contract normal-exit route and
        wake on Trial/formal transitions or a change in buy-one executable
        status/VWAP.  Positive quantity changes at the same executable price
        and irrelevant opposite-side depth therefore do not wake the loop.
        """

        series = self._require_series(key)
        relative = self._exit_quote_right_index(series, after)
        if relative >= series.exit_quote_length:
            return None
        derived_position = series.exit_quote_position_start + relative
        clock_position = int(self._exit_quote_clock_positions[derived_position])
        return self._indexed_event(clock_position, key).event.book_cursor

    def state_as_of(
        self,
        key: RawBookKey,
        at: QueryBoundary,
    ) -> CausalBookState | None:
        series = self._require_series(key)
        boundary_kind, target = _query_boundary_parts(at)
        cached = self._state_interval_cache.get(key)
        if cached is not None and self._relative_contains(
            series,
            cached.relative,
            boundary_kind,
            target,
        ):
            self._query_cache_counts["state_hits"] += 1
            return cached.state
        self._query_cache_counts["state_misses"] += 1
        indexed = self.indexed_event_as_of(key, at)
        state = None if indexed is None else RawBookStateMachine().ingest(indexed.event)
        indexed_cache = self._indexed_interval_cache[key]
        self._state_interval_cache[key] = _StateIntervalCache(
            indexed_cache.relative,
            state,
        )
        return state

    def iter_raw_changes(
        self,
        key: RawBookKey,
        start_exclusive: QueryBoundary,
        end_inclusive: QueryBoundary,
    ) -> Iterator[RawBookEvent]:
        return (
            indexed.event
            for indexed in self.iter_indexed_changes(
                key, start_exclusive, end_inclusive
            )
        )

    def iter_indexed_changes(
        self,
        key: RawBookKey,
        start_exclusive: QueryBoundary,
        end_inclusive: QueryBoundary,
    ) -> Iterator[IndexedRawBookEvent]:
        _validate_interval(start_exclusive, end_inclusive)
        series = self._require_series(key)
        left = self._right_index(series, start_exclusive)
        right = self._right_index(series, end_inclusive)
        return (
            self._indexed_event(series.position_start + relative, key)
            for relative in range(left, right)
        )

    def _right_index(self, series: _BookSeries, at: QueryBoundary) -> int:
        boundary_kind, target = _query_boundary_parts(at)
        low = 0
        high = series.length
        while low < high:
            middle = (low + high) // 2
            position = self._clock_position(series, middle)
            if self._packed_boundary(position, boundary_kind) <= target:
                low = middle + 1
            else:
                high = middle
        return low

    def _effective_right_index(self, series: _BookSeries, at: QueryBoundary) -> int:
        boundary_kind, target = _query_boundary_parts(at)
        low = 0
        high = series.effective_length
        while low < high:
            middle = (low + high) // 2
            effective_position = series.effective_position_start + middle
            clock_position = int(self._effective_clock_positions[effective_position])
            if self._packed_boundary(clock_position, boundary_kind) <= target:
                low = middle + 1
            else:
                high = middle
        return low

    def _exit_quote_right_index(self, series: _BookSeries, at: QueryBoundary) -> int:
        boundary_kind, target = _query_boundary_parts(at)
        low = 0
        high = series.exit_quote_length
        while low < high:
            middle = (low + high) // 2
            derived_position = series.exit_quote_position_start + middle
            clock_position = int(self._exit_quote_clock_positions[derived_position])
            if self._packed_boundary(clock_position, boundary_kind) <= target:
                low = middle + 1
            else:
                high = middle
        return low

    def _packed_boundary(
        self,
        position: int,
        boundary_kind: Literal["int", "event", "raw"],
    ) -> tuple[int, ...]:
        recv_time_ns = int(self._clock_recv_time_ns[position])
        if boundary_kind == "int":
            return (recv_time_ns,)
        event = (
            recv_time_ns,
            0,
            int(self._clock_loop_row_index[position]),
        )
        if boundary_kind == "event":
            return event
        return (*event, int(self._clock_packet_sequence[position]))

    def _relative_contains(
        self,
        series: _BookSeries,
        relative: int,
        boundary_kind: Literal["int", "event", "raw"],
        target: tuple[int, ...],
    ) -> bool:
        if relative < 0:
            return (
                series.length == 0
                or self._packed_boundary(
                    series.position_start,
                    boundary_kind,
                )
                > target
            )
        position = self._clock_position(series, relative)
        if self._packed_boundary(position, boundary_kind) > target:
            return False
        next_relative = relative + 1
        return (
            next_relative >= series.length
            or self._packed_boundary(
                self._clock_position(series, next_relative),
                boundary_kind,
            )
            > target
        )

    def _clock_position(self, series: _BookSeries, relative: int) -> int:
        return series.position_start + relative

    def _indexed_event(
        self,
        clock_position: int,
        key: RawBookKey,
    ) -> IndexedRawBookEvent:
        row = self._clock.row(clock_position, named=True)
        loop_cursor = EventCursor(
            int(row["recv_time_ns"]), 0, int(row["loop_row_index"])
        )
        raw_cursor = RawBookCursor(loop_cursor, int(row["packet_sequence"]))
        source = RawBookSourceCursor(
            loop_cursor.recv_time_ns,
            int(row["channel_sequence"]),
            int(row["packet_sequence"]),
            int(row["source_row"]),
        )
        if bool(row["trial_match"]):
            return IndexedRawBookEvent(
                RawBookEvent(raw_cursor, trial_match=True), source
            )
        series = self._require_series(key)
        payloads = self._payloads[key.market]
        l1_bids, l1_asks = _l1_levels(
            payloads.l1, row["current_l1_payload_id"], market=key.market
        )
        best_bid, best_ask = _best_levels(
            payloads.best, row["current_best_payload_id"], market=key.market
        )
        return IndexedRawBookEvent(
            RawBookEvent(
                raw_cursor,
                formal_book=True,
                reference_price=series.reference_price,
                l1_bids=l1_bids,
                l1_asks=l1_asks,
                best_bid=best_bid,
                best_ask=best_ask,
            ),
            source,
        )

    def _require_series(self, key: RawBookKey) -> _BookSeries:
        if not isinstance(key, RawBookKey):
            raise TypeError("key must be a RawBookKey")
        try:
            return self._series[key]
        except KeyError as error:
            raise ValueError("raw-book key is not in the exact mapping") from error


@dataclass(frozen=True, slots=True)
class RawBookRiskAdapter:
    """Thin bridge from ``RawBookDayIndex`` to the S1 event-loop protocol."""

    index: RawBookDayIndex
    _product_keys: Mapping[tuple[Market, str], RawBookKey] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.index, RawBookDayIndex):
            raise TypeError("index must be a RawBookDayIndex")
        keys = {(key.market, key.value_code): key for key in self.index.keys}
        if len(keys) != len(self.index.keys):
            raise ValueError("ValueCode is not unique within raw-book venue")
        object.__setattr__(self, "_product_keys", MappingProxyType(keys))

    def state_as_of(
        self,
        venue: Market,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        return self.index.state_as_of(
            self._key(venue, product_id),
            _event_cursor(cursor, "cursor"),
        )

    def next_change_cursor(
        self,
        venue: Market,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        after = _event_cursor(after_cursor, "after_cursor")
        deadline = _nonnegative_integer(deadline_ns, "deadline_ns")
        if deadline < after.recv_time_ns:
            raise ValueError("deadline_ns cannot precede after_cursor")
        candidate = self.index.next_effective_change_cursor(
            self._key(venue, product_id), after
        )
        if candidate is None or candidate.cursor.recv_time_ns > deadline:
            return None
        return candidate.cursor

    def next_exit_quote_change_cursor(
        self,
        venue: Market,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        """Return the route-specific normal-exit wake through the deadline."""

        after = _event_cursor(after_cursor, "after_cursor")
        deadline = _nonnegative_integer(deadline_ns, "deadline_ns")
        if deadline < after.recv_time_ns:
            raise ValueError("deadline_ns cannot precede after_cursor")
        candidate = self.index.next_exit_quote_change_cursor(
            self._key(venue, product_id), after
        )
        if candidate is None or candidate.cursor.recv_time_ns > deadline:
            return None
        return candidate.cursor

    def _key(self, venue: Market, product_id: str) -> RawBookKey:
        if venue not in ("spot", "future"):
            raise ValueError("venue must be spot or future")
        if not isinstance(product_id, str) or not product_id:
            raise ValueError("product_id must be a non-empty string")
        try:
            return self._product_keys[(venue, product_id)]
        except KeyError as error:
            raise ValueError(
                "product_id is not in the exact raw-book mapping"
            ) from error


def build_raw_book_day_index(
    spot_rows: pl.DataFrame,
    future_rows: pl.DataFrame,
    mapping: pl.DataFrame,
) -> RawBookDayIndex:
    """Compatibility entry point for materialized input frames."""

    return RawBookDayIndex.from_selected_rows(spot_rows, future_rows, mapping)


def build_raw_book_day_index_from_scans(
    spot_rows: SelectedFrame,
    future_rows: SelectedFrame,
    mapping: pl.DataFrame,
) -> RawBookDayIndex:
    """Scale-oriented entry point accepting lazy parquet scans."""

    return RawBookDayIndex.from_selected_scans(spot_rows, future_rows, mapping)


def _mapping_pairs(
    mapping: pl.DataFrame,
) -> tuple[tuple[_MappingPair, ...], pl.DataFrame]:
    if not isinstance(mapping, pl.DataFrame):
        raise TypeError("mapping must be a Polars DataFrame")
    _require_columns(mapping, _MAPPING_COLUMNS, "contract mapping")
    selected = mapping.select(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("spot_ref_price").cast(pl.Float64),
        pl.col("fut_ref_price").cast(pl.Float64),
    ).sort("ValueCode", "QuoteCode")
    invalid = selected.filter(
        pl.col("ValueCode").is_null()
        | (pl.col("ValueCode").str.len_chars() == 0)
        | pl.col("QuoteCode").is_null()
        | (pl.col("QuoteCode").str.len_chars() == 0)
    )
    if not invalid.is_empty():
        raise ValueError("contract mapping codes must be non-empty")
    for column in ("ValueCode", "QuoteCode"):
        duplicate = selected.group_by(column).len().filter(pl.col("len") != 1)
        if not duplicate.is_empty():
            raise ValueError(f"contract mapping is not one-to-one on {column}")
    pairs = tuple(
        _MappingPair(index, row[0], row[1], row[2], row[3])
        for index, row in enumerate(selected.iter_rows())
    )
    key_rows = [
        {
            "series_id": pair.series_id(market),
            "pair_id": pair.pair_id,
            "market_order": _MARKET_ORDER[market],
            "market": market,
            "ValueCode": pair.value_code,
            "QuoteCode": pair.quote_code,
        }
        for pair in pairs
        for market in ("spot", "future")
    ]
    key_frame = pl.from_dicts(key_rows, infer_schema_length=None).with_columns(
        pl.col("series_id").cast(pl.UInt32),
        pl.col("pair_id").cast(pl.UInt32),
        pl.col("market_order").cast(pl.UInt8),
    )
    return pairs, key_frame


def _build_market(
    raw: pl.LazyFrame,
    pairs: tuple[_MappingPair, ...],
    key_frame: pl.DataFrame,
    *,
    market: Market,
) -> _MarketBuild:
    del pairs
    schema = raw.collect_schema()
    required = set(_RAW_COLUMNS)
    if market == "future":
        required.add("DecimalLocator")
    _require_schema_columns(schema, required, f"raw {market} rows")
    lookup = key_frame.filter(pl.col("market") == market).select(
        pl.col("ValueCode" if market == "spot" else "QuoteCode").alias(
            "instrument_code"
        ),
        "series_id",
        "pair_id",
        "market_order",
    )
    selected = (
        raw.with_row_index("source_row")
        .filter(pl.col("QuoteCode").is_in(lookup["instrument_code"].to_list()))
        .join(
            lookup.lazy(),
            left_on="QuoteCode",
            right_on="instrument_code",
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
    )
    recv_time_ns = _recv_time_ns_expr(schema).alias("recv_time_ns")
    l1_update = pl.any_horizontal(
        *(
            pl.col(f"{side}Price{level}").fill_null(0) > 0
            for side in ("Bid", "Ask")
            for level in range(1, 6)
        )
    )
    best_update = pl.any_horizontal(
        pl.col("BestBidPrice").fill_null(0) > 0,
        pl.col("BestAskPrice").fill_null(0) > 0,
    )
    trial = pl.col("TrialMatch").fill_null(0) != 0
    formal = ~trial & (l1_update | best_update)
    candidate = (
        selected.select(
            "series_id",
            "pair_id",
            "market_order",
            recv_time_ns,
            pl.col("ChannelSeq").cast(pl.UInt32).alias("channel_sequence"),
            pl.col("PacketSeq").cast(pl.UInt32).alias("packet_sequence"),
            pl.col("source_row").cast(pl.UInt32),
            trial.alias("trial_match"),
            formal.alias("formal_book"),
            (formal & l1_update).alias("l1_update"),
            (formal & best_update).alias("best_update"),
            *_payload_expressions(market),
        )
        .filter(pl.col("trial_match") | pl.col("formal_book"))
        .with_columns(
            _payload_id("l1_update").alias("own_l1_payload_id"),
            _payload_id("best_update").alias("own_best_payload_id"),
        )
    )
    _validate_source_cursors(selected, schema, market)
    clock_lf = candidate.select(
        "series_id",
        "pair_id",
        "market_order",
        "recv_time_ns",
        "channel_sequence",
        "packet_sequence",
        "source_row",
        "trial_match",
        "own_l1_payload_id",
        "own_best_payload_id",
    )
    l1_lf = candidate.filter(pl.col("l1_update")).select(
        *_L1_PAYLOAD_COLUMNS,
        *(["decimal_locator"] if market == "future" else []),
    )
    best_lf = candidate.filter(pl.col("best_update")).select(
        *_BEST_PAYLOAD_COLUMNS,
        *(["decimal_locator"] if market == "future" else []),
    )
    # Collect sequentially.  Parallel collect_all evaluates the shared raw
    # projection three times at once and raises peak RSS by several GiB on a
    # 9.5M-row day without reducing the retained index size.
    clock = clock_lf.collect(engine="streaming")
    l1 = l1_lf.collect(engine="streaming")
    best = best_lf.collect(engine="streaming")
    _validate_payload_ids(clock, l1, best, market)
    return _MarketBuild(clock, l1, best)


def _validate_source_cursors(
    selected: pl.LazyFrame,
    schema: pl.Schema,
    market: Market,
) -> None:
    cursor = selected.select(
        "series_id",
        _recv_time_ns_expr(schema).alias("recv_time_ns"),
        pl.col("ChannelSeq").cast(pl.UInt32).alias("channel_sequence"),
        pl.col("PacketSeq").cast(pl.UInt32).alias("packet_sequence"),
    )
    invalid = cursor.filter(
        pl.col("recv_time_ns").is_null()
        | (pl.col("recv_time_ns") < 0)
        | pl.col("channel_sequence").is_null()
        | pl.col("packet_sequence").is_null()
    ).limit(1)
    duplicate = (
        cursor.group_by(
            "series_id", "recv_time_ns", "channel_sequence", "packet_sequence"
        )
        .len()
        .filter(pl.col("len") != 1)
        .limit(1)
    )
    invalid_result, duplicate_result = pl.collect_all(
        [invalid, duplicate], engine="streaming"
    )
    if not invalid_result.is_empty():
        raise ValueError(
            f"raw {market} cursor fields must be non-negative and non-null"
        )
    if not duplicate_result.is_empty():
        raise ValueError(f"raw {market} rows have an undecidable duplicate cursor")


def _payload_expressions(market: Market) -> tuple[pl.Expr, ...]:
    values: list[pl.Expr] = []
    price_dtype = pl.Float64 if market == "spot" else pl.Int32
    for side in ("Bid", "Ask"):
        lower = side.lower()
        for kind in ("Price", "Lots"):
            for level in range(1, 6):
                dtype = price_dtype if kind == "Price" else pl.Int32
                values.append(
                    pl.col(f"{side}{kind}{level}")
                    .fill_null(0)
                    .cast(dtype)
                    .alias(f"{lower}_{kind.lower()}_{level}")
                )
    for side in ("Bid", "Ask"):
        lower = side.lower()
        values.extend(
            [
                pl.col(f"Best{side}Price")
                .fill_null(0)
                .cast(price_dtype)
                .alias(f"best_{lower}_price"),
                pl.col(f"Best{side}Lots")
                .fill_null(0)
                .cast(pl.Int32)
                .alias(f"best_{lower}_lots"),
            ]
        )
    if market == "future":
        values.append(pl.col("DecimalLocator").cast(pl.Int16).alias("decimal_locator"))
    return tuple(values)


def _payload_id(flag: str) -> pl.Expr:
    running = pl.col(flag).cast(pl.Int64).cum_sum() - 1
    return pl.when(pl.col(flag)).then(running).otherwise(None).cast(pl.UInt32)


def _validate_payload_ids(
    clock: pl.DataFrame,
    l1: pl.DataFrame,
    best: pl.DataFrame,
    market: Market,
) -> None:
    for name, frame in (("l1", l1), ("best", best)):
        column = f"own_{name}_payload_id"
        values = clock[column].drop_nulls()
        if len(values) != frame.height:
            raise AssertionError(f"{market} {name} payload IDs lost alignment")
        if frame.height and int(values.max()) != frame.height - 1:
            raise AssertionError(f"{market} {name} payload IDs are not dense")


def _merge_clocks(
    spot: pl.DataFrame,
    future: pl.DataFrame,
) -> pl.DataFrame:
    clock = (
        pl.concat([spot, future], how="vertical")
        .sort(
            "series_id",
            "recv_time_ns",
            "channel_sequence",
            "packet_sequence",
            "source_row",
        )
        .with_columns(
            (
                pl.struct(
                    "market_order",
                    "pair_id",
                    "channel_sequence",
                    "packet_sequence",
                    "source_row",
                )
                .rank("ordinal")
                .over("recv_time_ns")
                - 1
            )
            .cast(pl.UInt32)
            .alias("loop_row_index"),
            pl.col("trial_match")
            .cast(pl.UInt32)
            .cum_sum()
            .over("series_id")
            .alias("trial_segment"),
        )
        .with_columns(
            pl.when(~pl.col("trial_match"))
            .then(pl.col("own_l1_payload_id"))
            .otherwise(None)
            .forward_fill()
            .over("series_id", "trial_segment")
            .alias("current_l1_payload_id"),
            pl.when(~pl.col("trial_match"))
            .then(pl.col("own_best_payload_id"))
            .otherwise(None)
            .forward_fill()
            .over("series_id", "trial_segment")
            .alias("current_best_payload_id"),
        )
        .select(*_CLOCK_COLUMNS)
    )
    return clock


def _effective_change_clock(
    clock: pl.DataFrame,
    payloads: Mapping[Market, _MarketPayloads],
) -> pl.DataFrame:
    """Derive a narrow clock of actual forward-filled state transitions.

    Payload structs provide collision-free equality without materializing
    Python rows or treating the dense, source-specific payload IDs as state
    identity.  The original clock position is retained as the sole lookup
    pointer, keeping exact raw cursor and ChannelSeq provenance out of this
    derived optimization layer.
    """

    positioned = clock.with_row_index("clock_position")
    market_clocks: list[pl.DataFrame] = []
    for market, market_order in _MARKET_ORDER.items():
        market_payloads = payloads[market]
        l1_states = market_payloads.l1.with_row_index("current_l1_payload_id").select(
            "current_l1_payload_id",
            pl.struct(*market_payloads.l1.columns).alias("l1_state"),
        )
        best_states = market_payloads.best.with_row_index(
            "current_best_payload_id"
        ).select(
            "current_best_payload_id",
            pl.struct(*market_payloads.best.columns).alias("best_state"),
        )
        annotated = (
            positioned.filter(pl.col("series_id") % 2 == market_order)
            .join(
                l1_states,
                on="current_l1_payload_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .join(
                best_states,
                on="current_best_payload_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .with_columns(
                pl.col("clock_position")
                .shift(1)
                .over("series_id")
                .alias("previous_clock_position"),
                pl.col("trial_match")
                .shift(1)
                .over("series_id")
                .alias("previous_trial_match"),
                pl.col("l1_state")
                .eq_missing(pl.col("l1_state").shift(1).over("series_id"))
                .alias("same_l1_state"),
                pl.col("best_state")
                .eq_missing(pl.col("best_state").shift(1).over("series_id"))
                .alias("same_best_state"),
            )
        )
        market_clocks.append(
            annotated.filter(
                pl.col("previous_clock_position").is_null()
                | pl.col("trial_match").ne_missing(pl.col("previous_trial_match"))
                | (
                    ~pl.col("trial_match")
                    & (~pl.col("same_l1_state") | ~pl.col("same_best_state"))
                )
            ).select(*_EFFECTIVE_CLOCK_COLUMNS)
        )
    return pl.concat(market_clocks, how="vertical").sort("series_id", "clock_position")


def _exit_quote_change_clock(
    clock: pl.DataFrame,
    payloads: Mapping[Market, _MarketPayloads],
    pairs: tuple[_MappingPair, ...],
) -> pl.DataFrame:
    """Derive normal-exit wakes from normalized route-visible book state."""

    positioned = clock.with_row_index("clock_position")
    spot = _normalized_bbo_clock(positioned, payloads["spot"], market="spot")
    spot_changes = (
        spot.with_columns(
            pl.col("clock_position")
            .shift(1)
            .over("series_id")
            .alias("previous_clock_position"),
            pl.col("trial_match")
            .shift(1)
            .over("series_id")
            .alias("previous_trial_match"),
            pl.col("bbo_bid_price")
            .shift(1)
            .over("series_id")
            .alias("previous_bbo_bid_price"),
            pl.col("bbo_ask_price")
            .shift(1)
            .over("series_id")
            .alias("previous_bbo_ask_price"),
        )
        .filter(
            pl.col("previous_clock_position").is_null()
            | pl.col("trial_match").ne_missing(pl.col("previous_trial_match"))
            | (
                ~pl.col("trial_match")
                & (
                    pl.col("bbo_bid_price").ne_missing(pl.col("previous_bbo_bid_price"))
                    | pl.col("bbo_ask_price").ne_missing(
                        pl.col("previous_bbo_ask_price")
                    )
                )
            )
        )
        .select(*_EXIT_QUOTE_CLOCK_COLUMNS)
    )

    future_references = pl.DataFrame(
        {
            "series_id": pl.Series(
                [pair.series_id("future") for pair in pairs],
                dtype=pl.UInt32,
            ),
            "reference_price": pl.Series(
                [pair.future_reference for pair in pairs],
                dtype=pl.Float64,
            ),
        }
    )
    future = _normalized_bbo_clock(
        positioned, payloads["future"], market="future"
    ).join(
        future_references,
        on="series_id",
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    reference = pl.col("reference_price")
    bid = pl.col("bbo_bid_price")
    ask = pl.col("bbo_ask_price")
    future_executable = (
        ~pl.col("trial_match")
        & reference.is_not_null()
        & reference.is_finite()
        & (reference > 0)
        & bid.is_not_null()
        & ask.is_not_null()
        & bid.is_finite()
        & ask.is_finite()
        & (bid <= ask)
        & (bid > reference * 0.91)
        & (bid < reference * 1.08)
        & (ask > reference * 0.91)
        & (ask < reference * 1.08)
    )
    future_changes = (
        future.with_columns(
            pl.when(future_executable).then(ask).otherwise(None).alias("buy_one_vwap")
        )
        .with_columns(
            pl.col("clock_position")
            .shift(1)
            .over("series_id")
            .alias("previous_clock_position"),
            pl.col("trial_match")
            .shift(1)
            .over("series_id")
            .alias("previous_trial_match"),
            pl.col("buy_one_vwap")
            .shift(1)
            .over("series_id")
            .alias("previous_buy_one_vwap"),
        )
        .filter(
            pl.col("previous_clock_position").is_null()
            | pl.col("trial_match").ne_missing(pl.col("previous_trial_match"))
            | (
                ~pl.col("trial_match")
                & pl.col("buy_one_vwap").ne_missing(pl.col("previous_buy_one_vwap"))
            )
        )
        .select(*_EXIT_QUOTE_CLOCK_COLUMNS)
    )
    return pl.concat([spot_changes, future_changes], how="vertical").sort(
        "series_id", "clock_position"
    )


def _normalized_bbo_clock(
    positioned_clock: pl.DataFrame,
    payloads: _MarketPayloads,
    *,
    market: Market,
) -> pl.DataFrame:
    l1 = payloads.l1.with_row_index("current_l1_payload_id").select(
        "current_l1_payload_id",
        pl.max_horizontal(
            *(
                _normalized_payload_price(
                    f"bid_price_{level}",
                    f"bid_lots_{level}",
                    market=market,
                )
                for level in range(1, 6)
            )
        ).alias("l1_bid_price"),
        pl.min_horizontal(
            *(
                _normalized_payload_price(
                    f"ask_price_{level}",
                    f"ask_lots_{level}",
                    market=market,
                )
                for level in range(1, 6)
            )
        ).alias("l1_ask_price"),
    )
    best = payloads.best.with_row_index("current_best_payload_id").select(
        "current_best_payload_id",
        _normalized_payload_price(
            "best_bid_price",
            "best_bid_lots",
            market=market,
        ).alias("best_bid_price"),
        _normalized_payload_price(
            "best_ask_price",
            "best_ask_lots",
            market=market,
        ).alias("best_ask_price"),
    )
    market_order = _MARKET_ORDER[market]
    return (
        positioned_clock.filter(pl.col("series_id") % 2 == market_order)
        .join(
            l1,
            on="current_l1_payload_id",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(
            best,
            on="current_best_payload_id",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .select(
            "series_id",
            "clock_position",
            "trial_match",
            pl.max_horizontal("l1_bid_price", "best_bid_price").alias("bbo_bid_price"),
            pl.min_horizontal("l1_ask_price", "best_ask_price").alias("bbo_ask_price"),
        )
    )


def _normalized_payload_price(
    price_column: str,
    lots_column: str,
    *,
    market: Market,
) -> pl.Expr:
    price = pl.col(price_column)
    valid = (price > 0) & (pl.col(lots_column) > 0)
    normalized = price.cast(pl.Float64)
    if market == "future":
        scale = pl.lit(10.0).pow(pl.col("decimal_locator").cast(pl.Float64))
        normalized = normalized / scale
    return pl.when(valid).then(normalized).otherwise(None)


def _series_metadata(
    pairs: tuple[_MappingPair, ...],
    clock: pl.DataFrame,
    effective_clock: pl.DataFrame,
    exit_quote_clock: pl.DataFrame,
    spot_lookup: pl.DataFrame,
) -> Mapping[RawBookKey, _BookSeries]:
    ranges = _frame_ranges(clock.select("series_id"))
    effective_ranges = _frame_ranges(effective_clock.select("series_id"))
    exit_quote_ranges = _frame_ranges(exit_quote_clock.select("series_id"))
    spot_ranges = _frame_ranges(spot_lookup.select("series_id"))
    result: dict[RawBookKey, _BookSeries] = {}
    for pair in pairs:
        for market in ("spot", "future"):
            series_id = pair.series_id(market)
            start, length = ranges.get(series_id, (0, 0))
            effective_start, effective_length = effective_ranges.get(series_id, (0, 0))
            exit_quote_start, exit_quote_length = exit_quote_ranges.get(
                series_id, (0, 0)
            )
            spot_start, spot_length = spot_ranges.get(series_id, (0, 0))
            result[pair.key(market)] = _BookSeries(
                series_id,
                start,
                length,
                effective_start,
                effective_length,
                exit_quote_start,
                exit_quote_length,
                spot_start,
                spot_length,
                pair.reference(market),
            )
    return result


def _frame_ranges(frame: pl.DataFrame) -> dict[int, tuple[int, int]]:
    return {
        int(row[0]): (int(row[1]), int(row[2]))
        for row in (
            frame.with_row_index("position_offset")
            .group_by("series_id", maintain_order=True)
            .agg(
                pl.col("position_offset").min().alias("position_start"),
                pl.len().alias("length"),
            )
            .iter_rows()
        )
    }


def _spot_source_lookup(clock: pl.DataFrame) -> pl.DataFrame:
    """Build a compact exact-key index without assuming ChannelSeq monotonicity."""

    return (
        clock.with_row_index("clock_position")
        .filter((pl.col("series_id") % 2 == 0) & ~pl.col("trial_match"))
        .select("series_id", "channel_sequence", "clock_position")
        .sort("series_id", "channel_sequence", "clock_position")
    )


def _packed_integer_column(frame: pl.DataFrame, name: str) -> object:
    series = frame[name]
    if series.null_count():
        raise ValueError(f"packed clock column contains nulls: {name}")
    values = series.to_numpy()
    values.setflags(write=False)
    return values


def _query_boundary_parts(
    at: QueryBoundary,
) -> tuple[Literal["int", "event", "raw"], tuple[int, ...]]:
    if isinstance(at, bool):
        raise TypeError("query boundary must be an integer or book cursor")
    if isinstance(at, int):
        return "int", (at,)
    if isinstance(at, EventCursor):
        return "event", (at.recv_time_ns, at.event_sequence, at.row_index)
    if isinstance(at, RawBookCursor):
        cursor = at.cursor
        return "raw", (
            cursor.recv_time_ns,
            cursor.event_sequence,
            cursor.row_index,
            at.packet_sequence,
        )
    raise TypeError("query boundary must be an integer or book cursor")


def _clock_boundary(
    clock: pl.DataFrame,
    position: int,
    template: QueryBoundary,
) -> QueryBoundary:
    recv_time_ns = int(clock["recv_time_ns"][position])
    if isinstance(template, bool):
        raise TypeError("query boundary must be an integer or book cursor")
    if isinstance(template, int):
        return recv_time_ns
    cursor = EventCursor(recv_time_ns, 0, int(clock["loop_row_index"][position]))
    if isinstance(template, EventCursor):
        return cursor
    if isinstance(template, RawBookCursor):
        return RawBookCursor(cursor, int(clock["packet_sequence"][position]))
    raise TypeError("query boundary must be an integer or book cursor")


def _l1_levels(
    payload: pl.DataFrame,
    payload_id: object,
    *,
    market: Market,
) -> tuple[tuple[RawBookLevel, ...], tuple[RawBookLevel, ...]]:
    if payload_id is None:
        return (), ()
    row = payload.row(_nonnegative_integer(payload_id, "L1 payload ID"), named=True)
    decimal_locator = _payload_decimal_locator(row, market)
    bids = _levels_from_row(row, "bid", market=market, decimal_locator=decimal_locator)
    asks = _levels_from_row(row, "ask", market=market, decimal_locator=decimal_locator)
    return bids, asks


def _best_levels(
    payload: pl.DataFrame,
    payload_id: object,
    *,
    market: Market,
) -> tuple[RawBookLevel | None, RawBookLevel | None]:
    if payload_id is None:
        return None, None
    row = payload.row(_nonnegative_integer(payload_id, "Best payload ID"), named=True)
    decimal_locator = _payload_decimal_locator(row, market)
    bid = _best_from_row(row, "bid", market=market, decimal_locator=decimal_locator)
    ask = _best_from_row(row, "ask", market=market, decimal_locator=decimal_locator)
    return bid, ask


def _levels_from_row(
    row: Mapping[str, object],
    side: Literal["bid", "ask"],
    *,
    market: Market,
    decimal_locator: int | None,
) -> tuple[RawBookLevel, ...]:
    levels: list[RawBookLevel] = []
    for level in range(1, 6):
        price = _stored_price(
            row[f"{side}_price_{level}"],
            market=market,
            decimal_locator=decimal_locator,
        )
        lots = _nonnegative_integer(row[f"{side}_lots_{level}"], "raw lots")
        if price is not None and lots > 0:
            levels.append(
                RawBookLevel(price, lots * (1_000 if market == "spot" else 1))
            )
    return tuple(levels)


def _best_from_row(
    row: Mapping[str, object],
    side: Literal["bid", "ask"],
    *,
    market: Market,
    decimal_locator: int | None,
) -> RawBookLevel | None:
    price = _stored_price(
        row[f"best_{side}_price"],
        market=market,
        decimal_locator=decimal_locator,
    )
    lots = _nonnegative_integer(row[f"best_{side}_lots"], "raw lots")
    if price is None or lots == 0:
        return None
    return RawBookLevel(price, lots * (1_000 if market == "spot" else 1))


def _payload_decimal_locator(row: Mapping[str, object], market: Market) -> int | None:
    if market == "spot":
        return None
    return _nonnegative_integer(row["decimal_locator"], "DecimalLocator")


def _stored_price(
    value: object,
    *,
    market: Market,
    decimal_locator: int | None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("stored price must be numeric")
    if market == "future":
        raw_integer = _nonnegative_integer(value, "future raw price")
        if raw_integer == 0:
            return None
        assert decimal_locator is not None
        return raw_integer / (10**decimal_locator)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("spot raw price must be finite and non-negative")
    return None if result == 0 else result


def _validate_interval(start: QueryBoundary, end: QueryBoundary) -> None:
    if type(start) is not type(end):
        raise TypeError("interval boundaries must use the same cursor type")
    if isinstance(start, bool):
        raise TypeError("interval boundaries must be integers or book cursors")
    if isinstance(start, int):
        if start < 0 or end < start:  # type: ignore[operator]
            raise ValueError("interval end cannot precede its non-negative start")
        return
    if isinstance(start, (EventCursor, RawBookCursor)):
        if end < start:  # type: ignore[operator]
            raise ValueError("interval end cannot precede its start")
        return
    raise TypeError("interval boundaries must be integers or book cursors")


def _recv_time_ns_expr(schema: pl.Schema) -> pl.Expr:
    dtype = schema.get("RecvTime")
    if not isinstance(dtype, pl.Datetime):
        raise TypeError("RecvTime must be a Datetime column")
    value = pl.col("RecvTime")
    if dtype.time_zone is not None:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return value.cast(pl.Datetime("ns")).cast(pl.Int64)


def _as_lazy(frame: SelectedFrame, name: str) -> pl.LazyFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    if isinstance(frame, pl.LazyFrame):
        return frame
    raise TypeError(f"{name} must be a Polars DataFrame or LazyFrame")


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _event_cursor(value: object, name: str) -> EventCursor:
    if not isinstance(value, EventCursor):
        raise TypeError(f"{name} must be an EventCursor")
    return value


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
    "IndexedRawBookEvent",
    "RawBookDayIndex",
    "RawBookKey",
    "RawBookRiskAdapter",
    "RawBookSourceCursor",
    "SpotSourceSnapshot",
    "build_raw_book_day_index",
    "build_raw_book_day_index_from_scans",
]
