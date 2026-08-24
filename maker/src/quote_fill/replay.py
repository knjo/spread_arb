"""Independent MBP maker-fill replay for sparse WP02 order windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .layered import EventCursor
from .targets import MakerSide


@dataclass(frozen=True)
class TradeEvent:
    cursor: EventCursor
    price_tick: int
    quantity: int

    def __post_init__(self) -> None:
        if self.price_tick <= 0 or self.quantity <= 0:
            raise ValueError("trade price_tick and quantity must be positive")


@dataclass(frozen=True)
class IndependentOrderWindow:
    generation_id: str
    maker_side: MakerSide
    target_price_tick: int
    start_cursor: EventCursor
    stop_cursor: EventCursor
    initial_queue_ahead: int | None
    stop_reason: str

    def __post_init__(self) -> None:
        if self.target_price_tick <= 0:
            raise ValueError("target_price_tick must be positive")
        if self.stop_cursor <= self.start_cursor:
            raise ValueError("stop_cursor must be after start_cursor")
        if self.initial_queue_ahead is not None and self.initial_queue_ahead < 0:
            raise ValueError("initial_queue_ahead cannot be negative")


@dataclass(frozen=True)
class IndependentFillLabel:
    generation_id: str
    executable_fill: bool
    fill_cursor: EventCursor | None
    fill_reason: Literal["trade_through", "queue_depletion"] | None
    same_price_quantity_before_stop: int
    initial_queue_ahead: int | None
    queue_known: bool
    touched_before_stop: bool
    trade_through_before_stop: bool
    stop_reason: str
    independent_event_label: bool = True
    joint_volume_allocated: bool = False


def label_independent_window(
    window: IndependentOrderWindow,
    trades: Sequence[TradeEvent] | Iterable[TradeEvent],
) -> IndependentFillLabel:
    """Label one order using future trades strictly inside its V0 window.

    Trade-through is definite even when the target is outside the visible MBP
    ladder.  Same-price queue depletion is only labelable when the starting
    queue ahead is known.  This deliberately mirrors the existing HFT sanity
    label's visible-queue convention while making cancellation explicit.
    """
    same_price_quantity = 0
    touched = False
    trade_through = False
    fill_cursor: EventCursor | None = None
    fill_reason: Literal["trade_through", "queue_depletion"] | None = None

    ordered_trades = tuple(trades)
    for previous, current in zip(ordered_trades, ordered_trades[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("trades must be strictly cursor-sorted")

    for trade in ordered_trades:
        if trade.cursor <= window.start_cursor:
            continue
        if trade.cursor > window.stop_cursor:
            break

        if window.maker_side == "bid":
            is_through = trade.price_tick < window.target_price_tick
            is_same = trade.price_tick == window.target_price_tick
        else:
            is_through = trade.price_tick > window.target_price_tick
            is_same = trade.price_tick == window.target_price_tick

        if is_through:
            touched = True
            trade_through = True
            fill_cursor = trade.cursor
            fill_reason = "trade_through"
            break
        if not is_same:
            continue

        touched = True
        same_price_quantity += trade.quantity
        # Consuming exactly the visible quantity only reaches the front of the
        # queue.  At least one additional integer lot is required for our
        # hypothetical order's first fill.  The old HFT makerFill label uses
        # >= and is retained separately as a boundary-optimistic sanity label.
        if (
            window.initial_queue_ahead is not None
            and same_price_quantity > window.initial_queue_ahead
        ):
            fill_cursor = trade.cursor
            fill_reason = "queue_depletion"
            break

    return IndependentFillLabel(
        generation_id=window.generation_id,
        executable_fill=fill_cursor is not None,
        fill_cursor=fill_cursor,
        fill_reason=fill_reason,
        same_price_quantity_before_stop=same_price_quantity,
        initial_queue_ahead=window.initial_queue_ahead,
        queue_known=window.initial_queue_ahead is not None,
        touched_before_stop=touched,
        trade_through_before_stop=trade_through,
        stop_reason=window.stop_reason,
    )
