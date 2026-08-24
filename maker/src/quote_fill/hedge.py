"""Pure WP03 delayed-taker and spot-partial-fill labels.

The functions in this module do not load tape.  Callers pass a causally
ordered sequence of full L1--L5 snapshots for the opposite market.  The
official hedge decision uses the last snapshot received by
``maker_fill_recv_time + 50 ms``; it never looks at the first update after the
deadline.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

from .layered import EventCursor


HedgeSide = Literal["buy", "sell"]
BookStatus = Literal["ok", "trial_match", "gate_closed"]
HedgeStatus = Literal[
    "executable",
    "insufficient_depth",
    "no_arrival_book",
    "stale_arrival_book",
    "arrival_trial_match",
    "arrival_gate_closed",
    "no_decision_book",
    "stale_decision_book",
    "decision_trial_match",
    "decision_gate_closed",
    "empty_arrival_side",
    "empty_decision_side",
]
PartialStatus = Literal[
    "complete",
    "partial",
    "no_fill",
    "trial_match_fill",
    "gate_closed_fill",
    "trial_match_and_gate_closed_fill",
]


DEFAULT_HEDGE_DELAY_NS = 50_000_000


@dataclass(frozen=True)
class BookLevel:
    """One executable price level in the native quantity unit."""

    price: float
    quantity: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("book price must be finite and positive")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("book quantity must be a positive integer")


@dataclass(frozen=True)
class OppositeBookSnapshot:
    """A complete causal L1--L5 snapshot for the market being crossed.

    Invalid/zero raw levels must be omitted by the loader.  ``bids`` are
    ordered best-to-worst (strictly decreasing price) and ``asks`` are ordered
    best-to-worst (strictly increasing price).
    """

    cursor: EventCursor
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    trial_match: bool = False
    gate_open: bool = True
    gate_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if not isinstance(self.bids, tuple) or not isinstance(self.asks, tuple):
            raise TypeError("bids and asks must be tuples")
        if len(self.bids) > 5 or len(self.asks) > 5:
            raise ValueError("book snapshots are limited to L1--L5")
        if not isinstance(self.trial_match, bool) or not isinstance(
            self.gate_open, bool
        ):
            raise ValueError("trial_match and gate_open must be boolean")
        if not self.gate_open and not self.gate_reason:
            raise ValueError("a closed gate requires gate_reason")
        self._validate_side(self.bids, descending=True, name="bids")
        self._validate_side(self.asks, descending=False, name="asks")

    @property
    def status(self) -> BookStatus:
        # TrialMatch takes priority because it is a direct exchange state.
        if self.trial_match:
            return "trial_match"
        if not self.gate_open:
            return "gate_closed"
        return "ok"

    @staticmethod
    def _validate_side(
        levels: tuple[BookLevel, ...], *, descending: bool, name: str
    ) -> None:
        if any(not isinstance(level, BookLevel) for level in levels):
            raise TypeError(f"{name} must contain BookLevel values")
        prices = [level.price for level in levels]
        pairs = zip(prices, prices[1:])
        valid = all(left > right for left, right in pairs) if descending else all(
            left < right for left, right in pairs
        )
        if not valid:
            order = "decreasing" if descending else "increasing"
            raise ValueError(f"{name} prices must be strictly {order}")


@dataclass(frozen=True)
class _IndexedOppositeBookSnapshots:
    """Immutable as-of index over an already validated snapshot tape."""

    snapshots: tuple[OppositeBookSnapshot, ...]
    cursors: tuple[EventCursor, ...]
    recv_times_ns: tuple[int, ...]


def _index_validated_opposite_snapshots(
    snapshots: tuple[OppositeBookSnapshot, ...],
) -> _IndexedOppositeBookSnapshots:
    return _IndexedOppositeBookSnapshots(
        snapshots,
        tuple(snapshot.cursor for snapshot in snapshots),
        tuple(snapshot.cursor.recv_time_ns for snapshot in snapshots),
    )


@dataclass(frozen=True)
class MakerFillHedgeRequest:
    """One incremental maker fill and its converted hedge quantity.

    ``maker_fill_quantity`` remains in maker-market units for auditing.
    ``hedge_quantity`` must already be converted to the opposite market's book
    unit (for example contracts to shares using that day's contract size).
    """

    generation_id: str
    fill_cursor: EventCursor
    maker_fill_quantity: int
    hedge_side: HedgeSide
    hedge_quantity: int
    delay_ns: int = DEFAULT_HEDGE_DELAY_NS

    def __post_init__(self) -> None:
        if not self.generation_id:
            raise ValueError("generation_id cannot be empty")
        if not isinstance(self.fill_cursor, EventCursor):
            raise TypeError("fill_cursor must be an EventCursor")
        for name, value in (
            ("maker_fill_quantity", self.maker_fill_quantity),
            ("hedge_quantity", self.hedge_quantity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hedge_side not in ("buy", "sell"):
            raise ValueError("hedge_side must be 'buy' or 'sell'")
        if (
            isinstance(self.delay_ns, bool)
            or not isinstance(self.delay_ns, int)
            or self.delay_ns < 0
        ):
            raise ValueError("delay_ns must be a non-negative integer")


@dataclass(frozen=True)
class HedgeExecutionLabel:
    """Delayed taker result; positive signed slippage is adverse."""

    generation_id: str
    status: HedgeStatus
    hedge_side: HedgeSide
    maker_fill_cursor: EventCursor
    maker_fill_quantity: int
    requested_hedge_quantity: int
    decision_time_ns: int
    arrival_snapshot_cursor: EventCursor | None = None
    decision_snapshot_cursor: EventCursor | None = None
    arrival_book_status: BookStatus | None = None
    decision_book_status: BookStatus | None = None
    gate_reason: str | None = None
    arrival_reference_price: float | None = None
    decision_best_price: float | None = None
    available_quantity: int = 0
    executed_quantity: int = 0
    depth_shortfall: int = 0
    levels_swept: int = 0
    partial_vwap_price: float | None = None
    executable_vwap_price: float | None = None
    signed_latency_slippage_price: float | None = None
    signed_latency_slippage_bp: float | None = None
    signed_depth_slippage_price: float | None = None
    signed_depth_slippage_bp: float | None = None
    signed_total_slippage_price: float | None = None
    signed_total_slippage_bp: float | None = None

    @property
    def hedge_complete(self) -> bool:
        return self.status == "executable"


def label_delayed_taker_hedge(
    request: MakerFillHedgeRequest,
    snapshots: Sequence[OppositeBookSnapshot]
    | Iterable[OppositeBookSnapshot],
    *,
    max_book_age_ns: int | None = None,
) -> HedgeExecutionLabel:
    """Label one 50 ms hedge using causal as-of full-book snapshots.

    The arrival reference is the opposite-side B1/A1 observable at the maker
    fill cursor.  Total signed slippage therefore measures latency plus depth;
    separate latency and depth components are also returned.  For a buy,
    higher prices are adverse.  For a sell, lower prices are adverse.
    """
    if not isinstance(request, MakerFillHedgeRequest):
        raise TypeError("request must be a MakerFillHedgeRequest")
    if max_book_age_ns is not None and (
        isinstance(max_book_age_ns, bool)
        or not isinstance(max_book_age_ns, int)
        or max_book_age_ns < 0
    ):
        raise ValueError("max_book_age_ns must be a non-negative integer or None")

    ordered = tuple(snapshots)
    _validate_snapshot_order(ordered)
    return _label_delayed_taker_hedge_indexed(
        request,
        _index_validated_opposite_snapshots(ordered),
        max_book_age_ns=max_book_age_ns,
    )


def _label_delayed_taker_hedge_validated_snapshots(
    request: MakerFillHedgeRequest,
    snapshots: tuple[OppositeBookSnapshot, ...],
    *,
    max_book_age_ns: int | None = None,
) -> HedgeExecutionLabel:
    """Label one hedge against an already validated immutable snapshot tape."""

    return _label_delayed_taker_hedge_indexed(
        request,
        _index_validated_opposite_snapshots(snapshots),
        max_book_age_ns=max_book_age_ns,
    )


def _label_delayed_taker_hedge_indexed(
    request: MakerFillHedgeRequest,
    snapshot_index: _IndexedOppositeBookSnapshots,
    *,
    max_book_age_ns: int | None = None,
) -> HedgeExecutionLabel:
    """Label one hedge using a validated, indexed immutable snapshot tape."""

    decision_time_ns = request.fill_cursor.recv_time_ns + request.delay_ns
    common = dict(
        generation_id=request.generation_id,
        hedge_side=request.hedge_side,
        maker_fill_cursor=request.fill_cursor,
        maker_fill_quantity=request.maker_fill_quantity,
        requested_hedge_quantity=request.hedge_quantity,
        decision_time_ns=decision_time_ns,
    )

    arrival = _latest_at_or_before_cursor(snapshot_index, request.fill_cursor)
    if arrival is None:
        return HedgeExecutionLabel(status="no_arrival_book", **common)
    if _is_stale(arrival, request.fill_cursor.recv_time_ns, max_book_age_ns):
        return HedgeExecutionLabel(
            status="stale_arrival_book",
            arrival_snapshot_cursor=arrival.cursor,
            arrival_book_status=arrival.status,
            gate_reason=arrival.gate_reason,
            **common,
        )
    if arrival.status != "ok":
        return HedgeExecutionLabel(
            status=f"arrival_{arrival.status}",  # type: ignore[arg-type]
            arrival_snapshot_cursor=arrival.cursor,
            arrival_book_status=arrival.status,
            gate_reason=arrival.gate_reason,
            **common,
        )

    arrival_levels = _levels_for_side(arrival, request.hedge_side)
    if not arrival_levels:
        return HedgeExecutionLabel(
            status="empty_arrival_side",
            arrival_snapshot_cursor=arrival.cursor,
            arrival_book_status=arrival.status,
            **common,
        )
    arrival_reference = arrival_levels[0].price

    decision = _latest_at_or_before_time(snapshot_index, decision_time_ns)
    if decision is None:
        # Kept as an explicit outcome even though an arrival snapshot normally
        # also qualifies for a later decision timestamp.
        return HedgeExecutionLabel(
            status="no_decision_book",
            arrival_snapshot_cursor=arrival.cursor,
            arrival_book_status=arrival.status,
            arrival_reference_price=arrival_reference,
            **common,
        )
    base = dict(
        arrival_snapshot_cursor=arrival.cursor,
        decision_snapshot_cursor=decision.cursor,
        arrival_book_status=arrival.status,
        decision_book_status=decision.status,
        arrival_reference_price=arrival_reference,
    )
    if _is_stale(decision, decision_time_ns, max_book_age_ns):
        return HedgeExecutionLabel(
            status="stale_decision_book",
            gate_reason=decision.gate_reason,
            **base,
            **common,
        )
    if decision.status != "ok":
        return HedgeExecutionLabel(
            status=f"decision_{decision.status}",  # type: ignore[arg-type]
            gate_reason=decision.gate_reason,
            **base,
            **common,
        )

    levels = _levels_for_side(decision, request.hedge_side)
    if not levels:
        return HedgeExecutionLabel(
            status="empty_decision_side", **base, **common
        )
    decision_best = levels[0].price
    latency_price = _adverse_difference(
        request.hedge_side, decision_best, arrival_reference
    )
    latency_bp = latency_price / arrival_reference * 10_000.0

    remaining = request.hedge_quantity
    notional = 0.0
    executed = 0
    levels_swept = 0
    for level in levels:
        take = min(remaining, level.quantity)
        if take <= 0:
            continue
        notional += level.price * take
        executed += take
        remaining -= take
        levels_swept += 1
        if remaining == 0:
            break

    partial_vwap = notional / executed if executed else None
    available = sum(level.quantity for level in levels)
    if remaining:
        return HedgeExecutionLabel(
            status="insufficient_depth",
            decision_best_price=decision_best,
            available_quantity=available,
            executed_quantity=executed,
            depth_shortfall=remaining,
            levels_swept=levels_swept,
            partial_vwap_price=partial_vwap,
            signed_latency_slippage_price=latency_price,
            signed_latency_slippage_bp=latency_bp,
            **base,
            **common,
        )

    assert partial_vwap is not None
    depth_price = _adverse_difference(
        request.hedge_side, partial_vwap, decision_best
    )
    total_price = _adverse_difference(
        request.hedge_side, partial_vwap, arrival_reference
    )
    return HedgeExecutionLabel(
        status="executable",
        decision_best_price=decision_best,
        available_quantity=available,
        executed_quantity=executed,
        depth_shortfall=0,
        levels_swept=levels_swept,
        partial_vwap_price=partial_vwap,
        executable_vwap_price=partial_vwap,
        signed_latency_slippage_price=latency_price,
        signed_latency_slippage_bp=latency_bp,
        signed_depth_slippage_price=depth_price,
        signed_depth_slippage_bp=depth_price / decision_best * 10_000.0,
        signed_total_slippage_price=total_price,
        signed_total_slippage_bp=total_price / arrival_reference * 10_000.0,
        **base,
        **common,
    )


@dataclass(frozen=True)
class SpotMakerFillEvent:
    """One incremental fill belonging to one spot-maker generation."""

    cursor: EventCursor
    quantity: int
    trial_match: bool = False
    gate_open: bool = True
    gate_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("fill quantity must be a positive integer")
        if not isinstance(self.trial_match, bool) or not isinstance(
            self.gate_open, bool
        ):
            raise ValueError("trial_match and gate_open must be boolean")
        if not self.gate_open and not self.gate_reason:
            raise ValueError("a gate-closed fill requires gate_reason")


@dataclass(frozen=True)
class PartialFillHorizonLabel:
    generation_id: str
    horizon_ns: int
    cutoff_time_ns: int
    status: PartialStatus
    futures_equivalent_quantity: int
    cumulative_fill_quantity: int
    residual_quantity: int
    completion_cursor: EventCursor | None
    incremental_fill_count: int
    trial_match_fill_quantity: int
    gate_closed_fill_quantity: int
    gate_reasons: tuple[str, ...]

    @property
    def hedge_unit_complete(self) -> bool:
        return self.cumulative_fill_quantity >= self.futures_equivalent_quantity


def label_spot_partial_fill_horizons(
    generation_id: str,
    first_fill_cursor: EventCursor,
    fills: Sequence[SpotMakerFillEvent] | Iterable[SpotMakerFillEvent],
    *,
    futures_equivalent_quantity: int,
    wait_horizons_ns: Sequence[int],
) -> tuple[PartialFillHorizonLabel, ...]:
    """Accumulate one spot maker generation from its first fill.

    A fill at a horizon's exact receive timestamp is included.  Quantities from
    TrialMatch/gate-closed rows remain physical fills and are accumulated, but
    the horizon status makes the invalid tape/gate state explicit.
    """
    if not generation_id:
        raise ValueError("generation_id cannot be empty")
    if not isinstance(first_fill_cursor, EventCursor):
        raise TypeError("first_fill_cursor must be an EventCursor")
    if (
        isinstance(futures_equivalent_quantity, bool)
        or not isinstance(futures_equivalent_quantity, int)
        or futures_equivalent_quantity <= 0
    ):
        raise ValueError("futures_equivalent_quantity must be a positive integer")
    horizons = tuple(wait_horizons_ns)
    if not horizons:
        raise ValueError("wait_horizons_ns cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in horizons
    ):
        raise ValueError("wait horizons must be non-negative integers")
    if any(right <= left for left, right in zip(horizons, horizons[1:])):
        raise ValueError("wait horizons must be strictly increasing and unique")

    ordered = tuple(fills)
    for fill in ordered:
        if not isinstance(fill, SpotMakerFillEvent):
            raise TypeError("fills must contain SpotMakerFillEvent values")
    for previous, current in zip(ordered, ordered[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("fills must be strictly cursor-sorted")

    relevant = tuple(fill for fill in ordered if fill.cursor >= first_fill_cursor)
    results: list[PartialFillHorizonLabel] = []
    for horizon in horizons:
        cutoff = first_fill_cursor.recv_time_ns + horizon
        observed = tuple(
            fill for fill in relevant if fill.cursor.recv_time_ns <= cutoff
        )
        cumulative = sum(fill.quantity for fill in observed)
        completion = _completion_cursor(observed, futures_equivalent_quantity)
        trial_qty = sum(fill.quantity for fill in observed if fill.trial_match)
        gate_qty = sum(fill.quantity for fill in observed if not fill.gate_open)
        reasons = tuple(
            dict.fromkeys(
                fill.gate_reason
                for fill in observed
                if not fill.gate_open and fill.gate_reason is not None
            )
        )
        if trial_qty and gate_qty:
            status: PartialStatus = "trial_match_and_gate_closed_fill"
        elif trial_qty:
            status = "trial_match_fill"
        elif gate_qty:
            status = "gate_closed_fill"
        elif cumulative >= futures_equivalent_quantity:
            status = "complete"
        elif cumulative:
            status = "partial"
        else:
            status = "no_fill"
        results.append(
            PartialFillHorizonLabel(
                generation_id=generation_id,
                horizon_ns=horizon,
                cutoff_time_ns=cutoff,
                status=status,
                futures_equivalent_quantity=futures_equivalent_quantity,
                cumulative_fill_quantity=cumulative,
                residual_quantity=max(futures_equivalent_quantity - cumulative, 0),
                completion_cursor=completion,
                incremental_fill_count=len(observed),
                trial_match_fill_quantity=trial_qty,
                gate_closed_fill_quantity=gate_qty,
                gate_reasons=reasons,
            )
        )
    return tuple(results)


def _validate_snapshot_order(snapshots: tuple[OppositeBookSnapshot, ...]) -> None:
    for snapshot in snapshots:
        if not isinstance(snapshot, OppositeBookSnapshot):
            raise TypeError("snapshots must contain OppositeBookSnapshot values")
    for previous, current in zip(snapshots, snapshots[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("snapshots must be strictly cursor-sorted")


def _latest_at_or_before_cursor(
    snapshot_index: _IndexedOppositeBookSnapshots, cursor: EventCursor
) -> OppositeBookSnapshot | None:
    index = bisect_right(snapshot_index.cursors, cursor) - 1
    if index < 0:
        return None
    return snapshot_index.snapshots[index]


def _latest_at_or_before_time(
    snapshot_index: _IndexedOppositeBookSnapshots, recv_time_ns: int
) -> OppositeBookSnapshot | None:
    index = bisect_right(snapshot_index.recv_times_ns, recv_time_ns) - 1
    if index < 0:
        return None
    return snapshot_index.snapshots[index]


def _is_stale(
    snapshot: OppositeBookSnapshot,
    query_time_ns: int,
    max_book_age_ns: int | None,
) -> bool:
    return (
        max_book_age_ns is not None
        and query_time_ns - snapshot.cursor.recv_time_ns > max_book_age_ns
    )


def _levels_for_side(
    snapshot: OppositeBookSnapshot, side: HedgeSide
) -> tuple[BookLevel, ...]:
    return snapshot.asks if side == "buy" else snapshot.bids


def _adverse_difference(side: HedgeSide, price: float, reference: float) -> float:
    return price - reference if side == "buy" else reference - price


def _completion_cursor(
    fills: tuple[SpotMakerFillEvent, ...], target: int
) -> EventCursor | None:
    cumulative = 0
    for fill in fills:
        cumulative += fill.quantity
        if cumulative >= target:
            return fill.cursor
    return None
