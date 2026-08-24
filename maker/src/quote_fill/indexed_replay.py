"""Indexed independent MBP maker-fill replay for many sparse windows.

The scalar replay in :mod:`maker.src.quote_fill.replay` is intentionally
simple, but scanning the complete trade tape once per candidate order is too
expensive for a pilot containing thousands of overlapping generations.  This
module builds two immutable indexes over one strictly cursor-sorted tape:

* a segment tree finds the first bid/ask trade-through in ``O(log N)``;
* per-price cursor/prefix-quantity arrays answer queue queries in
  ``O(log N_p)``, where ``N_p`` is the number of trades at that price.

Building the index costs ``O(N)`` time and memory.  Labeling ``W`` windows
with ``H`` post-cancel shadow horizons costs
``O(W * (H + 1) * log N)`` and does not allocate tape-sized data per window.
The active order interval remains exactly ``(start_cursor, stop_cursor]``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

from .layered import EventCursor
from .replay import IndependentFillLabel, IndependentOrderWindow, TradeEvent


DEFAULT_SHADOW_HORIZONS_MS: tuple[int, ...] = (10, 50, 100, 500)
_NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class ShadowTradeLabel:
    """Post-stop tape diagnostics for one cancellation-latency horizon.

    Shadow windows are ``(stop_cursor, stop_time + horizon_ms]``.  A touch is
    either a same-price print or a print through the maker price.  Quantity is
    the total same-price printed quantity across the complete shadow window;
    it is diagnostic only and is never treated as an executable fill.
    """

    horizon_ms: int
    touched_after_stop: bool
    trade_through_after_stop: bool
    first_touch_cursor: EventCursor | None
    first_trade_through_cursor: EventCursor | None
    same_price_quantity_after_stop: int


@dataclass(frozen=True)
class IndexedIndependentFillLabel(IndependentFillLabel):
    """Scalar-compatible fill label plus post-stop shadow diagnostics."""

    shadow_labels: tuple[ShadowTradeLabel, ...] = ()

    def shadow(self, horizon_ms: int) -> ShadowTradeLabel:
        """Return one shadow result, raising ``KeyError`` when not requested."""

        for label in self.shadow_labels:
            if label.horizon_ms == horizon_ms:
                return label
        raise KeyError(horizon_ms)


@dataclass(frozen=True)
class QuantityFillLabel:
    """Known own quantity after the visible queue, capped at policy size."""

    generation_id: str
    intended_quantity: int
    queue_known: bool
    first_fill_cursor: EventCursor | None
    full_fill_cursor: EventCursor | None
    known_filled_quantity_before_stop: int | None
    any_fill: bool | None
    full_fill: bool | None
    trade_through_fill: bool

    @property
    def partial_fill(self) -> bool:
        return (
            self.known_filled_quantity_before_stop is not None
            and 0 < self.known_filled_quantity_before_stop < self.intended_quantity
        )


@dataclass(frozen=True)
class _SamePriceSeries:
    cursors: tuple[EventCursor, ...]
    recv_times_ns: tuple[int, ...]
    cumulative_quantity: tuple[int, ...]

    @classmethod
    def from_events(cls, events: list[TradeEvent]) -> _SamePriceSeries:
        cursors: list[EventCursor] = []
        cumulative = [0]
        total = 0
        for event in events:
            cursors.append(event.cursor)
            total += event.quantity
            cumulative.append(total)
        return cls(
            tuple(cursors),
            tuple(cursor.recv_time_ns for cursor in cursors),
            tuple(cumulative),
        )

    def cursor_bounds(
        self,
        start_exclusive: EventCursor,
        end_inclusive: EventCursor | None = None,
    ) -> tuple[int, int]:
        left = bisect_right(self.cursors, start_exclusive)
        if end_inclusive is None:
            right = len(self.cursors)
        else:
            right = bisect_right(self.cursors, end_inclusive)
        return left, right

    def quantity(self, left: int, right: int) -> int:
        return self.cumulative_quantity[right] - self.cumulative_quantity[left]

    def first_cursor(self, left: int, right: int) -> EventCursor | None:
        if left >= right:
            return None
        return self.cursors[left]

    def queue_depletion_cursor(
        self,
        left: int,
        right: int,
        queue_ahead: int,
    ) -> EventCursor | None:
        if left >= right:
            return None

        # Exactly consuming the visible queue only moves us to the front.
        # First fill needs one additional lot; queue_ahead == 0 therefore
        # still requires the first same-price print.
        threshold = self.cumulative_quantity[left] + queue_ahead + 1
        prefix_index = bisect_left(
            self.cumulative_quantity,
            threshold,
            left + 1,
            right + 1,
        )
        if prefix_index > right:
            return None
        return self.cursors[prefix_index - 1]


class _PriceSegmentTree:
    """Range min/max tree supporting leftmost threshold searches."""

    def __init__(self, prices: Sequence[int]) -> None:
        size = 1
        while size < len(prices):
            size *= 2
        self.length = len(prices)
        self.size = size
        self.minimum = [float("inf")] * (2 * size)
        self.maximum = [float("-inf")] * (2 * size)
        for index, price in enumerate(prices):
            leaf = size + index
            self.minimum[leaf] = price
            self.maximum[leaf] = price
        for node in range(size - 1, 0, -1):
            self.minimum[node] = min(
                self.minimum[node * 2], self.minimum[node * 2 + 1]
            )
            self.maximum[node] = max(
                self.maximum[node * 2], self.maximum[node * 2 + 1]
            )

    def first_less(self, left: int, right: int, threshold: int) -> int | None:
        return self._first(
            left, right, threshold, less=True, node=1, lo=0, hi=self.size
        )

    def first_greater(
        self, left: int, right: int, threshold: int
    ) -> int | None:
        return self._first(
            left, right, threshold, less=False, node=1, lo=0, hi=self.size
        )

    def _first(
        self,
        query_left: int,
        query_right: int,
        threshold: int,
        *,
        less: bool,
        node: int,
        lo: int,
        hi: int,
    ) -> int | None:
        if hi <= query_left or query_right <= lo:
            return None
        if less:
            if self.minimum[node] >= threshold:
                return None
        elif self.maximum[node] <= threshold:
            return None
        if hi - lo == 1:
            return lo if lo < self.length else None

        middle = (lo + hi) // 2
        left_result = self._first(
            query_left,
            query_right,
            threshold,
            less=less,
            node=node * 2,
            lo=lo,
            hi=middle,
        )
        if left_result is not None:
            return left_result
        return self._first(
            query_left,
            query_right,
            threshold,
            less=less,
            node=node * 2 + 1,
            lo=middle,
            hi=hi,
        )


class IndexedTradeReplay:
    """Immutable query index over one strictly cursor-sorted trade tape."""

    def __init__(self, trades: Sequence[TradeEvent] | Iterable[TradeEvent]) -> None:
        self._trades = tuple(trades)
        for previous, current in zip(self._trades, self._trades[1:]):
            if current.cursor <= previous.cursor:
                raise ValueError("trades must be strictly cursor-sorted")

        self._cursors = tuple(trade.cursor for trade in self._trades)
        self._recv_times_ns = tuple(cursor.recv_time_ns for cursor in self._cursors)
        self._price_tree = _PriceSegmentTree(
            tuple(trade.price_tick for trade in self._trades)
        )

        events_by_price: dict[int, list[TradeEvent]] = {}
        for trade in self._trades:
            events_by_price.setdefault(trade.price_tick, []).append(trade)
        self._price_series = {
            price: _SamePriceSeries.from_events(events)
            for price, events in events_by_price.items()
        }

    def __len__(self) -> int:
        return len(self._trades)

    def label_window(
        self,
        window: IndependentOrderWindow,
        *,
        shadow_horizons_ms: Sequence[int] = DEFAULT_SHADOW_HORIZONS_MS,
    ) -> IndexedIndependentFillLabel:
        """Label one active window and its requested post-stop horizons."""

        horizons = _normalize_horizons(shadow_horizons_ms)
        return self._label_window(window, horizons)

    def _label_window(
        self,
        window: IndependentOrderWindow,
        horizons: tuple[int, ...],
    ) -> IndexedIndependentFillLabel:
        active_left = bisect_right(self._cursors, window.start_cursor)
        active_right = bisect_right(self._cursors, window.stop_cursor)
        series = self._price_series.get(window.target_price_tick)

        same_left = same_right = 0
        first_same: EventCursor | None = None
        queue_fill: EventCursor | None = None
        if series is not None:
            same_left, same_right = series.cursor_bounds(
                window.start_cursor, window.stop_cursor
            )
            first_same = series.first_cursor(same_left, same_right)
            if window.initial_queue_ahead is not None:
                queue_fill = series.queue_depletion_cursor(
                    same_left, same_right, window.initial_queue_ahead
                )

        through_index = self._first_through_index(
            window.maker_side,
            window.target_price_tick,
            active_left,
            active_right,
        )
        through_cursor = (
            None if through_index is None else self._cursors[through_index]
        )

        if through_cursor is not None and (
            queue_fill is None or through_cursor < queue_fill
        ):
            fill_cursor = through_cursor
            fill_reason = "trade_through"
        elif queue_fill is not None:
            fill_cursor = queue_fill
            fill_reason = "queue_depletion"
        else:
            fill_cursor = None
            fill_reason = None

        # Match scalar replay: once the independent order fills, later prints
        # in its nominal cancellation window no longer belong to this order.
        quantity_end = fill_cursor or window.stop_cursor
        if series is None:
            same_price_quantity = 0
        else:
            quantity_right = bisect_right(series.cursors, quantity_end)
            same_price_quantity = series.quantity(same_left, quantity_right)

        touched = (
            through_cursor is not None
            and (fill_cursor is None or through_cursor <= fill_cursor)
        ) or (
            first_same is not None
            and (fill_cursor is None or first_same <= fill_cursor)
        )

        shadow_labels = tuple(
            self._label_shadow(window, horizon_ms, series)
            for horizon_ms in horizons
        )
        return IndexedIndependentFillLabel(
            generation_id=window.generation_id,
            executable_fill=fill_cursor is not None,
            fill_cursor=fill_cursor,
            fill_reason=fill_reason,
            same_price_quantity_before_stop=same_price_quantity,
            initial_queue_ahead=window.initial_queue_ahead,
            queue_known=window.initial_queue_ahead is not None,
            touched_before_stop=touched,
            trade_through_before_stop=fill_reason == "trade_through",
            stop_reason=window.stop_reason,
            shadow_labels=shadow_labels,
        )

    def label_windows(
        self,
        windows: Iterable[IndependentOrderWindow],
        *,
        shadow_horizons_ms: Sequence[int] = DEFAULT_SHADOW_HORIZONS_MS,
    ) -> tuple[IndexedIndependentFillLabel, ...]:
        """Label windows in input order while reusing this tape index."""

        horizons = _normalize_horizons(shadow_horizons_ms)
        return tuple(
            self._label_window(window, horizons)
            for window in windows
        )

    def label_quantity(
        self,
        window: IndependentOrderWindow,
        intended_quantity: int,
    ) -> QuantityFillLabel:
        """Label partial/full own quantity within ``(start, stop]``.

        Same-price prints first consume ``initial_queue_ahead``.  A print
        through the maker price fills every remaining intended unit.  When
        queue ahead is unknown and no trade-through occurs, own quantity and
        fill booleans stay unknown instead of being silently treated as zero.
        """

        if (
            isinstance(intended_quantity, bool)
            or not isinstance(intended_quantity, int)
            or intended_quantity <= 0
        ):
            raise ValueError("intended_quantity must be a positive integer")
        left = bisect_right(self._cursors, window.start_cursor)
        right = bisect_right(self._cursors, window.stop_cursor)
        through_index = self._first_through_index(
            window.maker_side,
            window.target_price_tick,
            left,
            right,
        )
        through = None if through_index is None else self._cursors[through_index]
        series = self._price_series.get(window.target_price_tick)

        first_queue: EventCursor | None = None
        full_queue: EventCursor | None = None
        own_quantity: int | None
        if window.initial_queue_ahead is None:
            own_quantity = intended_quantity if through is not None else None
        else:
            same_left = same_right = 0
            same_quantity = 0
            if series is not None:
                same_left, same_right = series.cursor_bounds(
                    window.start_cursor, window.stop_cursor
                )
                same_quantity = series.quantity(same_left, same_right)
                first_queue = series.queue_depletion_cursor(
                    same_left, same_right, window.initial_queue_ahead
                )
                full_queue = series.queue_depletion_cursor(
                    same_left,
                    same_right,
                    window.initial_queue_ahead + intended_quantity - 1,
                )
            own_quantity = min(
                intended_quantity,
                max(0, same_quantity - window.initial_queue_ahead),
            )
            if through is not None:
                own_quantity = intended_quantity

        first_fill = _earlier_cursor(first_queue, through)
        full_fill = _earlier_cursor(full_queue, through)
        return QuantityFillLabel(
            generation_id=window.generation_id,
            intended_quantity=intended_quantity,
            queue_known=window.initial_queue_ahead is not None,
            first_fill_cursor=first_fill,
            full_fill_cursor=full_fill,
            known_filled_quantity_before_stop=own_quantity,
            any_fill=(first_fill is not None if own_quantity is not None else None),
            full_fill=(full_fill is not None if own_quantity is not None else None),
            trade_through_fill=through is not None,
        )

    def _label_shadow(
        self,
        window: IndependentOrderWindow,
        horizon_ms: int,
        series: _SamePriceSeries | None,
    ) -> ShadowTradeLabel:
        left = bisect_right(self._cursors, window.stop_cursor)
        deadline_ns = (
            window.stop_cursor.recv_time_ns
            + horizon_ms * _NANOSECONDS_PER_MILLISECOND
        )
        right = bisect_right(self._recv_times_ns, deadline_ns)

        through_index = self._first_through_index(
            window.maker_side,
            window.target_price_tick,
            left,
            right,
        )
        first_through = (
            None if through_index is None else self._cursors[through_index]
        )

        first_same: EventCursor | None = None
        same_price_quantity = 0
        if series is not None:
            same_left = bisect_right(series.cursors, window.stop_cursor)
            # Cursor ordering is lexicographic, but a millisecond deadline is
            # time-based and includes every tie-break event at that recv time.
            same_right = bisect_right(series.recv_times_ns, deadline_ns)
            first_same = series.first_cursor(same_left, same_right)
            same_price_quantity = series.quantity(same_left, same_right)

        first_touch = _earlier_cursor(first_same, first_through)
        return ShadowTradeLabel(
            horizon_ms=horizon_ms,
            touched_after_stop=first_touch is not None,
            trade_through_after_stop=first_through is not None,
            first_touch_cursor=first_touch,
            first_trade_through_cursor=first_through,
            same_price_quantity_after_stop=same_price_quantity,
        )

    def _first_through_index(
        self,
        maker_side: str,
        target_price_tick: int,
        left: int,
        right: int,
    ) -> int | None:
        if maker_side == "bid":
            return self._price_tree.first_less(left, right, target_price_tick)
        if maker_side == "ask":
            return self._price_tree.first_greater(left, right, target_price_tick)
        # IndependentOrderWindow's annotation is not runtime validation.
        raise ValueError(f"unknown maker_side: {maker_side}")


def label_independent_windows_indexed(
    windows: Iterable[IndependentOrderWindow],
    trades: Sequence[TradeEvent] | Iterable[TradeEvent],
    *,
    shadow_horizons_ms: Sequence[int] = DEFAULT_SHADOW_HORIZONS_MS,
) -> tuple[IndexedIndependentFillLabel, ...]:
    """Convenience wrapper for one batch over one trade tape."""

    return IndexedTradeReplay(trades).label_windows(
        windows, shadow_horizons_ms=shadow_horizons_ms
    )


def _normalize_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for value in horizons:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("shadow horizons must be positive integer milliseconds")
        normalized.add(value)
    return tuple(sorted(normalized))


def _earlier_cursor(
    first: EventCursor | None, second: EventCursor | None
) -> EventCursor | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)
