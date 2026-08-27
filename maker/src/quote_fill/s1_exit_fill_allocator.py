"""Joint printed-volume allocator for S1 Spot Ask aggregate exit orders.

Same-price spot prints are consumed once: first by the displayed queue frozen
at actual new, then by the aggregate order's own leaves.  A later print above
an ask limit is a definite trade-through and fills all remaining leaves even
when the original queue was not observable.  The allocator is intentionally
separate from inventory allocation; its physical fills are fed into
``S1ExitInventoryController.on_fill`` in the same event-loop phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .layered import EventCursor
from .replay import TradeEvent
from .s1_exit_inventory import AggregateOrderTerminal, AggregateWorkingOrder

FillReason = Literal["same_price_queue_depletion", "trade_through"]


class ExitFillAllocationError(ValueError):
    """Physical exit maker replay violated its deterministic contract."""


@dataclass(frozen=True, slots=True)
class ExitPhysicalFill:
    fill_id: str
    raw_order_fact_id: str
    fill_cursor: EventCursor
    target_price_tick: int
    trade_price_tick: int
    source_trade_quantity_shares: int
    fill_shares: int
    fill_reason: FillReason
    queue_ahead_before_shares: int | None
    queue_ahead_after_shares: int | None
    leaves_before_shares: int
    leaves_after_shares: int
    joint_same_price_volume_allocated: bool
    trade_through_inference: bool


@dataclass(slots=True)
class _LiveOrder:
    raw_order_fact_id: str
    target_price_tick: int
    actual_start_cursor: EventCursor
    leaves_shares: int
    queue_ahead_shares: int | None


class S1SpotAskFillAllocator:
    """One product/scenario exact Spot Ask maker volume clock."""

    def __init__(self, *, session_date: str, product_id: str) -> None:
        self.session_date = _valid_date(session_date, "session_date")
        self.product_id = _identifier(product_id, "product_id")
        self._orders: dict[str, _LiveOrder] = {}
        self._raw_id_by_tick: dict[int, str] = {}
        self._fills: list[ExitPhysicalFill] = []
        self._last_trade_cursor: EventCursor | None = None
        self._next_fill_sequence = 1
        self._used_raw_ids: set[str] = set()

    @property
    def fills(self) -> tuple[ExitPhysicalFill, ...]:
        return tuple(self._fills)

    @property
    def active_raw_order_ids(self) -> tuple[str, ...]:
        return tuple(
            order.raw_order_fact_id
            for order in sorted(
                self._orders.values(),
                key=lambda value: (
                    value.target_price_tick,
                    value.actual_start_cursor,
                    value.raw_order_fact_id,
                ),
            )
        )

    @property
    def minimum_active_target_price_tick(self) -> int | None:
        """Lowest working ask target, used to skip irrelevant lower trades."""

        if not self._orders:
            return None
        return min(order.target_price_tick for order in self._orders.values())

    def register_order(
        self,
        order: AggregateWorkingOrder,
        *,
        initial_queue_ahead_shares: int | None,
    ) -> None:
        """Register a physical order only after its actual new-send effect."""

        if not isinstance(order, AggregateWorkingOrder):
            raise TypeError("order must be an AggregateWorkingOrder")
        if order.raw_order_fact_id in self._used_raw_ids:
            raise ExitFillAllocationError("raw order identity was already registered")
        if order.absolute_price_tick in self._raw_id_by_tick:
            raise ExitFillAllocationError(
                "only one aggregate physical order may work at an absolute price"
            )
        if order.leaves_shares <= 0:
            raise ExitFillAllocationError("registered order must have positive leaves")
        if initial_queue_ahead_shares is not None:
            _nonnegative_integer(
                initial_queue_ahead_shares,
                "initial_queue_ahead_shares",
            )
        self._orders[order.raw_order_fact_id] = _LiveOrder(
            order.raw_order_fact_id,
            order.absolute_price_tick,
            order.actual_start_cursor,
            order.leaves_shares,
            initial_queue_ahead_shares,
        )
        self._raw_id_by_tick[order.absolute_price_tick] = order.raw_order_fact_id
        self._used_raw_ids.add(order.raw_order_fact_id)

    def on_trade(self, trade: TradeEvent) -> tuple[ExitPhysicalFill, ...]:
        """Allocate one raw spot print across currently working ask orders."""

        if not isinstance(trade, TradeEvent):
            raise TypeError("trade must be a TradeEvent")
        if (
            self._last_trade_cursor is not None
            and trade.cursor <= self._last_trade_cursor
        ):
            raise ExitFillAllocationError("spot trades must be strictly cursor ordered")
        self._last_trade_cursor = trade.cursor
        emitted: list[ExitPhysicalFill] = []
        for order in sorted(
            self._orders.values(),
            key=lambda value: (
                value.target_price_tick,
                value.actual_start_cursor,
                value.raw_order_fact_id,
            ),
        ):
            if order.leaves_shares == 0 or trade.cursor <= order.actual_start_cursor:
                continue
            if trade.price_tick > order.target_price_tick:
                emitted.append(
                    self._fill(
                        order,
                        trade,
                        order.leaves_shares,
                        reason="trade_through",
                        queue_before=order.queue_ahead_shares,
                        queue_after=order.queue_ahead_shares,
                    )
                )
                continue
            if trade.price_tick != order.target_price_tick:
                continue
            queue_before = order.queue_ahead_shares
            if queue_before is None:
                continue
            queue_consumed = min(queue_before, trade.quantity)
            queue_after = queue_before - queue_consumed
            order.queue_ahead_shares = queue_after
            own_printed = trade.quantity - queue_consumed
            fill_shares = min(order.leaves_shares, own_printed)
            if fill_shares:
                emitted.append(
                    self._fill(
                        order,
                        trade,
                        fill_shares,
                        reason="same_price_queue_depletion",
                        queue_before=queue_before,
                        queue_after=queue_after,
                    )
                )
        return tuple(emitted)

    def on_order_terminal(self, terminal: AggregateOrderTerminal) -> None:
        """Remove an order after controller fill/cancel/expiry terminalization."""

        if not isinstance(terminal, AggregateOrderTerminal):
            raise TypeError("terminal must be an AggregateOrderTerminal")
        try:
            order = self._orders[terminal.raw_order_fact_id]
        except KeyError as error:
            raise ExitFillAllocationError(
                "terminal raw order is not registered in the fill allocator"
            ) from error
        if terminal.terminal_cursor <= order.actual_start_cursor:
            raise ExitFillAllocationError("terminal cursor must follow actual new")
        if terminal.leaves_shares != order.leaves_shares:
            raise ExitFillAllocationError(
                "controller and physical allocator leaves diverged at terminal"
            )
        del self._orders[terminal.raw_order_fact_id]
        del self._raw_id_by_tick[order.target_price_tick]

    def assert_working_order(self, order: AggregateWorkingOrder) -> None:
        """Cross-check controller leaves after every accepted physical fill."""

        if not isinstance(order, AggregateWorkingOrder):
            raise TypeError("order must be an AggregateWorkingOrder")
        try:
            physical = self._orders[order.raw_order_fact_id]
        except KeyError as error:
            raise ExitFillAllocationError("working order is not registered") from error
        if (
            physical.target_price_tick != order.absolute_price_tick
            or physical.leaves_shares != order.leaves_shares
        ):
            raise ExitFillAllocationError(
                "controller and physical allocator working state diverged"
            )

    def _fill(
        self,
        order: _LiveOrder,
        trade: TradeEvent,
        fill_shares: int,
        *,
        reason: FillReason,
        queue_before: int | None,
        queue_after: int | None,
    ) -> ExitPhysicalFill:
        leaves_before = order.leaves_shares
        if fill_shares <= 0 or fill_shares > leaves_before:
            raise RuntimeError("physical fill is outside aggregate leaves")
        order.leaves_shares -= fill_shares
        fact = ExitPhysicalFill(
            fill_id=(
                f"exit_physical_fill/{self.session_date}/{self.product_id}/"
                f"{order.raw_order_fact_id}/{self._next_fill_sequence:012d}"
            ),
            raw_order_fact_id=order.raw_order_fact_id,
            fill_cursor=trade.cursor,
            target_price_tick=order.target_price_tick,
            trade_price_tick=trade.price_tick,
            source_trade_quantity_shares=trade.quantity,
            fill_shares=fill_shares,
            fill_reason=reason,
            queue_ahead_before_shares=queue_before,
            queue_ahead_after_shares=queue_after,
            leaves_before_shares=leaves_before,
            leaves_after_shares=order.leaves_shares,
            joint_same_price_volume_allocated=(reason == "same_price_queue_depletion"),
            trade_through_inference=reason == "trade_through",
        )
        self._next_fill_sequence += 1
        self._fills.append(fact)
        return fact


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExitFillAllocationError(f"{name} must be a nonnegative integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExitFillAllocationError(f"{name} must be a non-empty string")
    return value


def _valid_date(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ExitFillAllocationError(f"{name} must be valid YYYYMMDD")
    try:
        parsed = date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as error:
        raise ExitFillAllocationError(f"{name} must be valid YYYYMMDD") from error
    if parsed.strftime("%Y%m%d") != value:
        raise ExitFillAllocationError(f"{name} must be valid YYYYMMDD")
    return value


__all__ = [
    "ExitFillAllocationError",
    "ExitPhysicalFill",
    "S1SpotAskFillAllocator",
]
