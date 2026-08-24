"""Pure rolling-window venue request scheduler.

The scheduler owns only request ordering and venue tokens.  It deliberately
does not mutate books, orders, positions, or capacity reservations.  A caller
enqueues causally available intents, freezes send assignments with
``dispatch``, and applies the returned assignments in its own event-loop
phase.

The rolling limit is evaluated at the actual send timestamp over the interval
``(t - window, t]``.  A request sent exactly one window after the oldest send
can therefore reuse that token.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .layered import EventCursor

ONE_SECOND_NS = 1_000_000_000

RequestClass = Literal["exposed_risk", "cancel", "new"]
MakerSide = Literal["bid", "ask"]

_REQUEST_CLASS_ORDER: tuple[RequestClass, ...] = (
    "exposed_risk",
    "cancel",
    "new",
)
_REQUEST_CLASS_RANK = {
    request_class: rank
    for rank, request_class in enumerate(_REQUEST_CLASS_ORDER)
}


@dataclass(frozen=True)
class VenueRequestIntent:
    """One causally available request waiting for a venue token.

    ``stable_id`` is the final deterministic FIFO tie breaker.  Cutoff-drain
    cancels additionally carry maker side and absolute price so equally timed
    Spot Bid cancels can drain the most aggressive price first.
    """

    request_id: str
    venue: str
    request_class: RequestClass
    original_cursor: EventCursor
    stable_id: str
    cutoff_drain: bool = False
    maker_side: MakerSide | None = None
    absolute_price_tick: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("venue", self.venue),
            ("stable_id", self.stable_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.request_class not in _REQUEST_CLASS_RANK:
            raise ValueError(f"unsupported request_class: {self.request_class}")
        if not isinstance(self.original_cursor, EventCursor):
            raise TypeError("original_cursor must be an EventCursor")
        if not isinstance(self.cutoff_drain, bool):
            raise TypeError("cutoff_drain must be boolean")
        if self.maker_side not in (None, "bid", "ask"):
            raise ValueError("maker_side must be bid, ask, or None")
        if self.absolute_price_tick is not None and (
            isinstance(self.absolute_price_tick, bool)
            or not isinstance(self.absolute_price_tick, int)
            or self.absolute_price_tick <= 0
        ):
            raise ValueError("absolute_price_tick must be a positive integer")
        if self.cutoff_drain:
            if self.request_class != "cancel":
                raise ValueError("only cancel requests can be cutoff drains")
            if self.maker_side is None or self.absolute_price_tick is None:
                raise ValueError(
                    "cutoff drain requires maker_side and absolute_price_tick"
                )

    @property
    def intent_cursor(self) -> EventCursor:
        """Compatibility alias for the original request cursor."""

        return self.original_cursor


@dataclass(frozen=True)
class VenueSendAssignment:
    """One request assigned an actual send cursor and venue token."""

    request: VenueRequestIntent
    actual_send_cursor: EventCursor
    send_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, VenueRequestIntent):
            raise TypeError("request must be a VenueRequestIntent")
        if not isinstance(self.actual_send_cursor, EventCursor):
            raise TypeError("actual_send_cursor must be an EventCursor")
        if self.actual_send_cursor < self.request.original_cursor:
            raise ValueError("actual send cannot precede the original cursor")
        if (
            isinstance(self.send_sequence, bool)
            or not isinstance(self.send_sequence, int)
            or self.send_sequence <= 0
        ):
            raise ValueError("send_sequence must be a positive integer")

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def queue_delay_ns(self) -> int:
        return (
            self.actual_send_cursor.recv_time_ns
            - self.request.original_cursor.recv_time_ns
        )

    @property
    def was_delayed(self) -> bool:
        return self.queue_delay_ns > 0


class RollingVenueScheduler:
    """Priority scheduler for one venue's rolling request limit.

    Dispatch cursors must be non-decreasing.  The scheduler records a token as
    consumed as soon as it returns an assignment; the caller may then apply
    cancels and new orders in separate event-loop phases without changing the
    frozen send decision.
    """

    def __init__(
        self,
        venue: str,
        request_cap: int,
        *,
        window_ns: int = ONE_SECOND_NS,
    ) -> None:
        if not isinstance(venue, str) or not venue:
            raise ValueError("venue must be a non-empty string")
        if (
            isinstance(request_cap, bool)
            or not isinstance(request_cap, int)
            or request_cap <= 0
        ):
            raise ValueError("request_cap must be a positive integer")
        if (
            isinstance(window_ns, bool)
            or not isinstance(window_ns, int)
            or window_ns <= 0
        ):
            raise ValueError("window_ns must be a positive integer")

        self.venue = venue
        self.request_cap = request_cap
        self.window_ns = window_ns
        self._pending: dict[str, VenueRequestIntent] = {}
        self._pending_heaps: dict[
            RequestClass,
            list[tuple[tuple[object, ...], str]],
        ] = {request_class: [] for request_class in _REQUEST_CLASS_ORDER}
        self._known_request_ids: set[str] = set()
        self._sent_times_ns: deque[int] = deque()
        self._last_dispatch_cursor: EventCursor | None = None
        self._next_send_sequence = 1
        self._total_sent = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def total_sent(self) -> int:
        return self._total_sent

    @property
    def last_dispatch_cursor(self) -> EventCursor | None:
        return self._last_dispatch_cursor

    @property
    def pending_requests(self) -> tuple[VenueRequestIntent, ...]:
        """Pending requests in the order a sufficiently late dispatch uses."""

        return tuple(sorted(self._pending.values(), key=_dispatch_order_key))

    def enqueue(self, request: VenueRequestIntent) -> None:
        """Add one unique request to the send-eligible queue."""

        if not isinstance(request, VenueRequestIntent):
            raise TypeError("request must be a VenueRequestIntent")
        if request.venue != self.venue:
            raise ValueError(
                f"request venue {request.venue!r} does not match "
                f"scheduler venue {self.venue!r}"
            )
        if request.request_id in self._known_request_ids:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        self._known_request_ids.add(request.request_id)
        self._pending[request.request_id] = request
        heapq.heappush(
            self._pending_heaps[request.request_class],
            (_within_class_order_key(request), request.request_id),
        )

    def cancel_pending(self, request_id: str) -> VenueRequestIntent | None:
        """Remove an unsent request and return it, or return ``None``."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        return self._pending.pop(request_id, None)

    def dispatch(
        self,
        cursor: EventCursor,
        *,
        max_requests: int | None = None,
    ) -> tuple[VenueSendAssignment, ...]:
        """Freeze all currently possible send assignments at ``cursor``.

        ``max_requests`` can impose a caller-side bound below the available
        venue tokens.  It does not reserve unused tokens.
        """

        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if (
            self._last_dispatch_cursor is not None
            and cursor < self._last_dispatch_cursor
        ):
            raise ValueError("dispatch cursors must be non-decreasing")
        if max_requests is not None and (
            isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests <= 0
        ):
            raise ValueError("max_requests must be a positive integer")

        self._last_dispatch_cursor = cursor
        self._expire_tokens(cursor.recv_time_ns)
        available = self.request_cap - len(self._sent_times_ns)
        if max_requests is not None:
            available = min(available, max_requests)

        assignments: list[VenueSendAssignment] = []
        for _ in range(available):
            request = self._pop_next_eligible(cursor)
            if request is None:
                break
            assignment = VenueSendAssignment(
                request=request,
                actual_send_cursor=cursor,
                send_sequence=self._next_send_sequence,
            )
            self._next_send_sequence += 1
            self._total_sent += 1
            self._sent_times_ns.append(cursor.recv_time_ns)
            assignments.append(assignment)
        return tuple(assignments)

    def next_token_time_ns(self) -> int | None:
        """Return the earliest timestamp at which pending work can send.

        The return value is a timestamp, not a complete event cursor.  If an
        intent shares that timestamp with the most recent dispatch but is
        later in causal phase order, the caller must construct a cursor not
        earlier than that intent's ``original_cursor``.
        """

        if not self._pending:
            return None
        earliest_intent_ns = min(
            request.original_cursor.recv_time_ns
            for request in self._pending.values()
        )
        if self._last_dispatch_cursor is None:
            return earliest_intent_ns

        current_ns = self._last_dispatch_cursor.recv_time_ns
        self._expire_tokens(current_ns)
        if len(self._sent_times_ns) < self.request_cap:
            return max(current_ns, earliest_intent_ns)
        return max(
            earliest_intent_ns,
            self._sent_times_ns[0] + self.window_ns,
        )

    def _expire_tokens(self, at_time_ns: int) -> None:
        lower_bound = at_time_ns - self.window_ns
        while self._sent_times_ns and self._sent_times_ns[0] <= lower_bound:
            self._sent_times_ns.popleft()

    def _pop_next_eligible(
        self,
        cursor: EventCursor,
    ) -> VenueRequestIntent | None:
        for request_class in _REQUEST_CLASS_ORDER:
            heap = self._pending_heaps[request_class]
            while heap and heap[0][1] not in self._pending:
                heapq.heappop(heap)
            if not heap:
                continue
            request = self._pending[heap[0][1]]
            if request.original_cursor > cursor:
                continue
            heapq.heappop(heap)
            return self._pending.pop(request.request_id)
        return None


def _dispatch_order_key(request: VenueRequestIntent) -> tuple[object, ...]:
    return (
        _REQUEST_CLASS_RANK[request.request_class],
        *_within_class_order_key(request),
        request.request_id,
    )


def _within_class_order_key(
    request: VenueRequestIntent,
) -> tuple[object, ...]:
    if request.cutoff_drain:
        assert request.absolute_price_tick is not None
        assert request.maker_side is not None
        price_priority = (
            -request.absolute_price_tick
            if request.maker_side == "bid"
            else request.absolute_price_tick
        )
        return (
            request.original_cursor.recv_time_ns,
            request.original_cursor.event_sequence,
            1,
            price_priority,
            request.stable_id,
            request.original_cursor.row_index,
        )
    return (
        request.original_cursor.recv_time_ns,
        request.original_cursor.event_sequence,
        0,
        request.original_cursor.row_index,
        request.stable_id,
        0,
    )


__all__ = [
    "ONE_SECOND_NS",
    "EventCursor",
    "MakerSide",
    "RequestClass",
    "RollingVenueScheduler",
    "VenueRequestIntent",
    "VenueSendAssignment",
]
