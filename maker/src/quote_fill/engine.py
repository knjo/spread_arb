"""Pure layered-order window engine for WP02 raw replay.

The data loader is responsible for turning spot/futures/fair updates into a
sparse, causally ordered stream of :class:`TargetObservation` values.  This
module applies the agreed SpreadPair/forward-layer/retreat rules and emits one
independent replay window per admitted generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .layered import EventCursor, LayeredSampler
from .replay import IndependentOrderWindow


@dataclass(frozen=True)
class TargetObservation:
    """One state-changing target observation for a single route/policy."""

    cursor: EventCursor
    spread_pair_epoch: int
    absolute_target_tick: int | None
    gate_open: bool
    gate_reason: str
    initial_queue_ahead: int | None = None
    target_rank: str | None = None
    source: str = "market_state"

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if (
            isinstance(self.spread_pair_epoch, bool)
            or not isinstance(self.spread_pair_epoch, int)
            or self.spread_pair_epoch < 0
        ):
            raise ValueError("spread_pair_epoch must be a non-negative integer")
        if not isinstance(self.gate_open, bool):
            raise ValueError("gate_open must be boolean")
        if not self.gate_open and not self.gate_reason:
            raise ValueError("a closed gate requires gate_reason")
        if self.initial_queue_ahead is not None and (
            isinstance(self.initial_queue_ahead, bool)
            or not isinstance(self.initial_queue_ahead, int)
            or self.initial_queue_ahead < 0
        ):
            raise ValueError("initial_queue_ahead must be non-negative or None")


@dataclass(frozen=True)
class IntentTransition:
    """Submit/cancel/suppression fact emitted by the sparse sampler."""

    policy_id: str
    generation_id: str | None
    kind: str
    reason: str
    cursor: EventCursor
    spread_pair_epoch: int
    target_price_tick: int | None
    target_rank: str | None
    active_layers_after: int


@dataclass(frozen=True)
class LayeredWindowBuildResult:
    windows: tuple[IndependentOrderWindow, ...]
    transitions: tuple[IntentTransition, ...]
    peak_active_layers: int


@dataclass
class _OpenGeneration:
    generation_id: str
    target_tick: int
    start_cursor: EventCursor
    initial_queue_ahead: int | None


def build_layered_order_windows(
    observations: Iterable[TargetObservation],
    *,
    route: str,
    policy_id: str,
    cutoff_cursor: EventCursor,
) -> LayeredWindowBuildResult:
    """Apply layered admission and return cancellation-bounded raw windows.

    Fill is intentionally not evaluated here.  A later indexed tape pass lets
    fill win whenever its cursor precedes the nominal cancellation/cutoff.
    This keeps the state machine sparse while preserving the correct V0 risk
    interval ``(submit, stop]``.
    """

    if not policy_id:
        raise ValueError("policy_id cannot be empty")
    if not isinstance(cutoff_cursor, EventCursor):
        raise TypeError("cutoff_cursor must be an EventCursor")

    ordered = tuple(observations)
    for previous, current in zip(ordered, ordered[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("observations must be strictly cursor-sorted")
    if ordered and cutoff_cursor <= ordered[-1].cursor:
        raise ValueError("cutoff_cursor must follow every observation")

    sampler = LayeredSampler(route)
    open_generations: dict[int, _OpenGeneration] = {}
    windows: list[IndependentOrderWindow] = []
    transitions: list[IntentTransition] = []
    peak_active = 0

    for observation in ordered:
        active_count = len(sampler.active_orders)
        actions = sampler.reconcile(
            observation.cursor,
            observation.spread_pair_epoch,
            observation.absolute_target_tick,
            gate_open=observation.gate_open,
            gate_reason=observation.gate_reason,
        )
        for action in actions:
            generation_id: str | None = None
            if action.order is not None:
                generation_id = f"{policy_id}/{action.order.generation}"
            if action.kind == "submit":
                assert action.order is not None
                open_generations[action.order.generation] = _OpenGeneration(
                    generation_id=generation_id,
                    target_tick=action.order.absolute_price_tick,
                    start_cursor=action.order.submit_cursor,
                    initial_queue_ahead=observation.initial_queue_ahead,
                )
                active_count += 1
            elif action.kind == "cancel":
                assert action.order is not None
                opened = open_generations.pop(action.order.generation)
                windows.append(
                    IndependentOrderWindow(
                        generation_id=opened.generation_id,
                        maker_side=action.order.maker_side,
                        target_price_tick=opened.target_tick,
                        start_cursor=opened.start_cursor,
                        stop_cursor=observation.cursor,
                        initial_queue_ahead=opened.initial_queue_ahead,
                        stop_reason=action.reason,
                    )
                )
                active_count -= 1

            transitions.append(
                IntentTransition(
                    policy_id=policy_id,
                    generation_id=generation_id,
                    kind=action.kind,
                    reason=action.reason,
                    cursor=observation.cursor,
                    spread_pair_epoch=observation.spread_pair_epoch,
                    target_price_tick=action.target_price_tick,
                    target_rank=observation.target_rank,
                    active_layers_after=active_count,
                )
            )
        peak_active = max(peak_active, len(sampler.active_orders))

    cutoff_active = len(open_generations)
    for generation, opened in sorted(open_generations.items()):
        order = next(
            item for item in sampler.active_orders if item.generation == generation
        )
        windows.append(
            IndependentOrderWindow(
                generation_id=opened.generation_id,
                maker_side=order.maker_side,
                target_price_tick=opened.target_tick,
                start_cursor=opened.start_cursor,
                stop_cursor=cutoff_cursor,
                initial_queue_ahead=opened.initial_queue_ahead,
                stop_reason="session_cutoff",
            )
        )
        transitions.append(
            IntentTransition(
                policy_id=policy_id,
                generation_id=opened.generation_id,
                kind="cancel",
                reason="session_cutoff",
                cursor=cutoff_cursor,
                spread_pair_epoch=order.spread_pair_epoch,
                target_price_tick=order.absolute_price_tick,
                target_rank=None,
                active_layers_after=cutoff_active - 1,
            )
        )
        cutoff_active -= 1

    return LayeredWindowBuildResult(
        windows=tuple(
            sorted(
                windows,
                key=lambda item: (item.start_cursor, item.generation_id),
            )
        ),
        transitions=tuple(transitions),
        peak_active_layers=peak_active,
    )
