"""Pure state machine for WP02 layered maker-order admission.

The sampler is deliberately independent of raw-tape loading and fill replay.
One instance must be scoped to one product-day and one route.  It converts a
causally ordered stream of ``SpreadPairTotalCount`` epochs and rounded target
ticks into sparse submit/cancel/suppression actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .targets import MakerSide, ROUTE_SPECS, Stage


ActionKind = Literal["submit", "cancel", "suppress", "terminal"]


@dataclass(frozen=True, order=True)
class EventCursor:
    """Total causal order assigned by the merged-event builder.

    ``event_sequence`` is a merged-stream tie breaker.  It must not be filled
    by directly comparing ``ChannelSeq`` values from different markets.
    ``row_index`` provides a final deterministic order when one merged event
    produces multiple state observations.
    """

    recv_time_ns: int
    event_sequence: int = 0
    row_index: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("recv_time_ns", self.recv_time_ns),
            ("event_sequence", self.event_sequence),
            ("row_index", self.row_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class LayeredOrder:
    """One independent WP02 research generation."""

    generation: int
    spread_pair_epoch: int
    route: str
    stage: Stage
    maker_side: MakerSide
    absolute_price_tick: int
    submit_cursor: EventCursor

    @property
    def sampling_key(self) -> tuple[int, str, Stage, int]:
        """The within-product part of the raw candidate de-duplication key."""
        return (
            self.spread_pair_epoch,
            self.route,
            self.stage,
            self.absolute_price_tick,
        )


@dataclass(frozen=True)
class LayeredAction:
    """A sparse state transition emitted by :class:`LayeredSampler`."""

    kind: ActionKind
    reason: str
    cursor: EventCursor
    spread_pair_epoch: int
    target_price_tick: int | None
    route: str
    stage: Stage
    maker_side: MakerSide
    order: LayeredOrder | None = None


class LayeredSampler:
    """Admit layered maker-order generations for one product-day route.

    The caller supplies legal *absolute* price-tick indices.  B1/B2 rank is
    intentionally absent: a live order keeps its absolute price when the book
    moves.  ``SpreadPairTotalCount`` is supplied as ``spread_pair_epoch``.
    """

    def __init__(self, route: str) -> None:
        try:
            spec = ROUTE_SPECS[route]
        except KeyError as error:
            raise ValueError(f"unknown route: {route}") from error
        self.route = spec.route
        self.stage = spec.stage
        self.maker_side = spec.maker_side

        self._last_cursor: EventCursor | None = None
        self._current_epoch: int | None = None
        self._current_target_tick: int | None = None
        self._seen_ticks: set[int] = set()
        self._next_generation = 1
        self._active: dict[int, LayeredOrder] = {}
        self._orders: dict[int, LayeredOrder] = {}

    @property
    def current_epoch(self) -> int | None:
        return self._current_epoch

    @property
    def current_target_tick(self) -> int | None:
        return self._current_target_tick

    @property
    def seen_price_ticks(self) -> frozenset[int]:
        """Absolute prices already admitted in the current epoch."""
        return frozenset(self._seen_ticks)

    @property
    def active_orders(self) -> tuple[LayeredOrder, ...]:
        return tuple(self._active[generation] for generation in sorted(self._active))

    @property
    def all_orders(self) -> tuple[LayeredOrder, ...]:
        return tuple(self._orders[generation] for generation in sorted(self._orders))

    def reconcile(
        self,
        cursor: EventCursor,
        spread_pair_epoch: int,
        absolute_target_tick: int | None,
        *,
        gate_open: bool = True,
        gate_reason: str = "gate_closed",
    ) -> tuple[LayeredAction, ...]:
        """Reconcile working layers with one causally ordered target state.

        A new epoch admits one base generation whenever the gate is open, even
        if an older epoch already has a live order at the same price.  Inside
        one epoch only a new, more-aggressive, unseen price can add a layer.
        Retreat never adds: it only cancels live prices more aggressive than
        the new target.  A closed gate immediately cancels every live layer.
        """
        self._validate_next_cursor(cursor)
        epoch = self._validate_epoch(spread_pair_epoch)
        target_tick = self._validate_target(absolute_target_tick, gate_open)

        if self._current_epoch is not None and epoch < self._current_epoch:
            raise ValueError(
                "spread_pair_epoch must be monotonic (SpreadPairTotalCount)"
            )
        self._last_cursor = cursor
        new_epoch = self._current_epoch is None or epoch > self._current_epoch
        previous_target = self._current_target_tick
        if new_epoch:
            self._current_epoch = epoch
            self._seen_ticks = set()

        actions: list[LayeredAction] = []
        if not gate_open:
            actions.extend(
                self._cancel_orders(
                    cursor,
                    epoch,
                    target_tick,
                    tuple(self._active.values()),
                    gate_reason,
                )
            )
            self._current_target_tick = target_tick
            return tuple(actions)

        assert target_tick is not None
        actions.extend(
            self._cancel_orders(
                cursor,
                epoch,
                target_tick,
                tuple(
                    order
                    for order in self._active.values()
                    if self._aggressiveness(order.absolute_price_tick)
                    > self._aggressiveness(target_tick)
                ),
                "target_retreat",
            )
        )

        if new_epoch:
            # Epoch admission is independent of any live older generation.
            actions.append(self._submit(cursor, epoch, target_tick, "new_epoch"))
        elif previous_target is not None:
            movement = self._aggressiveness(target_tick) - self._aggressiveness(
                previous_target
            )
            if movement > 0:
                if target_tick not in self._seen_ticks:
                    actions.append(
                        self._submit(cursor, epoch, target_tick, "forward_new_price")
                    )
                else:
                    actions.append(
                        self._action(
                            "suppress",
                            "seen_price_in_epoch",
                            cursor,
                            epoch,
                            target_tick,
                        )
                    )
            # movement <= 0 deliberately never creates a generation.

        self._current_target_tick = target_tick
        return tuple(actions)

    def mark_terminal(
        self,
        cursor: EventCursor,
        generation: int,
        *,
        reason: str = "filled",
    ) -> LayeredAction:
        """Remove a filled/externally-ended order before later reconciliation."""
        self._validate_next_cursor(cursor)
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("generation must be an integer")
        try:
            order = self._active[generation]
        except KeyError as error:
            raise ValueError(f"generation is not active: {generation}") from error
        self._last_cursor = cursor
        del self._active[generation]
        assert self._current_epoch is not None
        return self._action(
            "terminal",
            reason,
            cursor,
            self._current_epoch,
            self._current_target_tick,
            order,
        )

    def _submit(
        self,
        cursor: EventCursor,
        epoch: int,
        target_tick: int,
        reason: str,
    ) -> LayeredAction:
        order = LayeredOrder(
            generation=self._next_generation,
            spread_pair_epoch=epoch,
            route=self.route,
            stage=self.stage,
            maker_side=self.maker_side,
            absolute_price_tick=target_tick,
            submit_cursor=cursor,
        )
        self._next_generation += 1
        self._active[order.generation] = order
        self._orders[order.generation] = order
        self._seen_ticks.add(target_tick)
        return self._action(
            "submit", reason, cursor, epoch, target_tick, order
        )

    def _cancel_orders(
        self,
        cursor: EventCursor,
        epoch: int,
        target_tick: int | None,
        orders: tuple[LayeredOrder, ...],
        reason: str,
    ) -> list[LayeredAction]:
        actions: list[LayeredAction] = []
        for order in sorted(orders, key=lambda value: value.generation):
            if self._active.pop(order.generation, None) is None:
                continue
            actions.append(
                self._action(
                    "cancel", reason, cursor, epoch, target_tick, order
                )
            )
        return actions

    def _action(
        self,
        kind: ActionKind,
        reason: str,
        cursor: EventCursor,
        epoch: int,
        target_tick: int | None,
        order: LayeredOrder | None = None,
    ) -> LayeredAction:
        return LayeredAction(
            kind=kind,
            reason=reason,
            cursor=cursor,
            spread_pair_epoch=epoch,
            target_price_tick=target_tick,
            route=self.route,
            stage=self.stage,
            maker_side=self.maker_side,
            order=order,
        )

    def _validate_next_cursor(self, cursor: EventCursor) -> None:
        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if self._last_cursor is not None and cursor <= self._last_cursor:
            raise ValueError("EventCursor must be strictly increasing")

    @staticmethod
    def _validate_epoch(epoch: int) -> int:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("spread_pair_epoch must be a non-negative integer")
        return epoch

    @staticmethod
    def _validate_target(target_tick: int | None, gate_open: bool) -> int | None:
        if not isinstance(gate_open, bool):
            raise ValueError("gate_open must be boolean")
        if target_tick is None:
            if gate_open:
                raise ValueError("an open gate requires an absolute target tick")
            return None
        if (
            isinstance(target_tick, bool)
            or not isinstance(target_tick, int)
            or target_tick <= 0
        ):
            raise ValueError("absolute_target_tick must be a positive integer")
        return target_tick

    def _aggressiveness(self, price_tick: int) -> int:
        return price_tick if self.maker_side == "bid" else -price_tick
