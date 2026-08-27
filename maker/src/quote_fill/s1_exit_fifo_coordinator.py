"""Deterministic oldest-unresolved FIFO gate for the normal S1 exit route.

The inventory controller deliberately owns only physical aggregate-order
state.  This module supplies its outer product/scenario gate: at one causal
observation it turns a complete set of position views and exit targets into a
complete atomic ``ExitDesiredUpdate`` batch.

Resolved positions are transparent to FIFO selection.  Starting at the oldest
unresolved position, the only quoteable cohort is the maximal consecutive
prefix whose targets are open at one identical absolute tick and whose
available inventory consists of integral, per-position hedge units.  A
different target or any unresolved execution state stops the prefix; no newer
position may jump it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .layered import EventCursor
from .s1_exit_inventory import ExitDesiredUpdate, ExitPositionView
from .s1_exit_target import S1SpotAskTarget


class ExitFifoCoordinatorError(ValueError):
    """Invalid or causally inconsistent outer-coordinator input."""


def build_exit_fifo_desired_updates(
    positions: Sequence[ExitPositionView],
    targets: Sequence[S1SpotAskTarget],
    *,
    observation_cursor: EventCursor,
) -> tuple[ExitDesiredUpdate, ...]:
    """Build one canonical full batch for ``set_positions_desired``.

    Inputs may be presented in any order.  The returned rows are always sorted
    by ``(position_established_ns, position_id)`` and contain exactly one row
    for every input position.  Every target must belong to the same product,
    scenario, session, quote contract, and exact observation cursor.

    Hedge units are never pooled across positions.  A position contributes its
    full currently available inventory only when that inventory is independently
    divisible by its own ``contract_size_shares``.
    """

    if not isinstance(observation_cursor, EventCursor):
        raise TypeError("observation_cursor must be an EventCursor")
    position_rows = _as_nonempty_sequence(
        positions,
        expected_type=ExitPositionView,
        name="positions",
    )
    target_rows = _as_nonempty_sequence(
        targets,
        expected_type=S1SpotAskTarget,
        name="targets",
    )

    positions_by_id: dict[str, ExitPositionView] = {}
    for position in position_rows:
        _validate_position(position)
        if position.position_id in positions_by_id:
            raise ExitFifoCoordinatorError("positions repeat position_id")
        positions_by_id[position.position_id] = position

    canonical_positions = tuple(
        sorted(positions_by_id.values(), key=lambda item: item.fifo_key)
    )
    value_code = canonical_positions[0].value_code
    scenario_id = canonical_positions[0].scenario_id
    contract_size = canonical_positions[0].contract_size_shares
    for position in canonical_positions[1:]:
        if position.value_code != value_code:
            raise ExitFifoCoordinatorError("positions must share one value_code")
        if position.scenario_id != scenario_id:
            raise ExitFifoCoordinatorError("positions must share one scenario_id")
        if position.contract_size_shares != contract_size:
            raise ExitFifoCoordinatorError(
                "one product must share one contract_size_shares"
            )

    targets_by_id: dict[str, S1SpotAskTarget] = {}
    common_date: str | None = None
    common_quote_code: str | None = None
    for target in target_rows:
        _validate_target(target, observation_cursor)
        if target.position_id in targets_by_id:
            raise ExitFifoCoordinatorError("targets repeat position_id")
        targets_by_id[target.position_id] = target
        if common_date is None:
            common_date = target.date
            common_quote_code = target.quote_code
        elif target.date != common_date:
            raise ExitFifoCoordinatorError("targets must share one date")
        elif target.quote_code != common_quote_code:
            raise ExitFifoCoordinatorError("targets must share one quote_code")

    position_ids = set(positions_by_id)
    target_ids = set(targets_by_id)
    if target_ids != position_ids:
        missing = sorted(position_ids - target_ids)
        extra = sorted(target_ids - position_ids)
        raise ExitFifoCoordinatorError(
            f"targets must cover positions exactly; missing={missing}, extra={extra}"
        )

    for position in canonical_positions:
        target = targets_by_id[position.position_id]
        if target.value_code != position.value_code:
            raise ExitFifoCoordinatorError(
                "target value_code differs from its position"
            )
        if target.scenario_id != position.scenario_id:
            raise ExitFifoCoordinatorError(
                "target scenario_id differs from its position"
            )
        if position.position_established_ns > observation_cursor.recv_time_ns:
            raise ExitFifoCoordinatorError(
                "target observation precedes position establishment"
            )

    updates: list[ExitDesiredUpdate] = []
    cohort_tick: int | None = None
    cohort_stopped = False
    for position in canonical_positions:
        target = targets_by_id[position.position_id]
        if position.resolution_state == "flat":
            updates.append(ExitDesiredUpdate(position.position_id, None, 0))
            continue

        quoteable = _position_is_quoteable(position, target)
        target_tick = target.absolute_price_tick
        if cohort_stopped or not quoteable:
            cohort_stopped = True
            updates.append(ExitDesiredUpdate(position.position_id, None, 0))
            continue

        assert target_tick is not None
        if cohort_tick is None:
            cohort_tick = target_tick
        elif target_tick != cohort_tick:
            cohort_stopped = True
            updates.append(ExitDesiredUpdate(position.position_id, None, 0))
            continue

        updates.append(
            ExitDesiredUpdate(
                position.position_id,
                target_tick,
                position.available_spot_shares,
            )
        )
    return tuple(updates)


def _as_nonempty_sequence[InputRow: (ExitPositionView, S1SpotAskTarget)](
    values: object,
    *,
    expected_type: type[InputRow],
    name: str,
) -> tuple[InputRow, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if not values:
        raise ExitFifoCoordinatorError(f"{name} cannot be empty")
    rows = tuple(values)
    if any(not isinstance(value, expected_type) for value in rows):
        raise TypeError(f"{name} must contain {expected_type.__name__} values")
    return rows


def _validate_position(position: ExitPositionView) -> None:
    for name in ("position_id", "value_code", "scenario_id", "capacity_id"):
        _identifier(getattr(position, name), name)
    _positive_integer(position.position_established_ns, "position_established_ns")
    _positive_integer(position.contract_size_shares, "contract_size_shares")
    for name in (
        "available_spot_shares",
        "desired_shares",
        "unhedged_fill_shares",
        "hedge_pending_shares",
        "hedged_exit_shares",
        "rollback_pending_shares",
        "rollback_failed_shares",
        "unresolved_shares",
    ):
        _nonnegative_integer(getattr(position, name), name)

    if position.desired_shares > position.available_spot_shares:
        raise ExitFifoCoordinatorError("desired shares exceed available spot inventory")
    if position.desired_shares == 0:
        if position.desired_absolute_target_tick is not None:
            raise ExitFifoCoordinatorError(
                "zero desired shares require a null desired target"
            )
    else:
        _positive_integer(
            position.desired_absolute_target_tick,
            "desired_absolute_target_tick",
        )

    if (position.allocation_id is None) != (position.allocation_kind is None):
        raise ExitFifoCoordinatorError(
            "allocation identity and kind must be present together"
        )
    if position.allocation_id is not None:
        _identifier(position.allocation_id, "allocation_id")
    if position.allocation_kind not in (None, "candidate", "working"):
        raise ExitFifoCoordinatorError("invalid allocation_kind")

    expected_unresolved = (
        position.available_spot_shares
        + position.unhedged_fill_shares
        + position.hedge_pending_shares
        + position.rollback_pending_shares
        + position.rollback_failed_shares
    )
    if position.unresolved_shares != expected_unresolved:
        raise ExitFifoCoordinatorError(
            "position unresolved quantity buckets do not reconcile"
        )
    expected_state = "flat" if expected_unresolved == 0 else "unresolved"
    if position.resolution_state != expected_state:
        raise ExitFifoCoordinatorError(
            "position resolution_state disagrees with unresolved inventory"
        )
    if not isinstance(position.capacity_releasable, bool):
        raise TypeError("capacity_releasable must be bool")
    if position.capacity_releasable != (expected_state == "flat"):
        raise ExitFifoCoordinatorError(
            "capacity_releasable disagrees with position resolution"
        )
    expected_reasons = tuple(
        reason
        for quantity, reason in (
            (position.available_spot_shares, "available_inventory"),
            (position.unhedged_fill_shares, "unhedged_fill"),
            (position.hedge_pending_shares, "hedge_pending"),
            (position.rollback_pending_shares, "rollback_pending"),
            (position.rollback_failed_shares, "rollback_failed"),
        )
        if quantity
    )
    if position.unresolved_reasons != expected_reasons:
        raise ExitFifoCoordinatorError(
            "position unresolved_reasons disagree with quantity buckets"
        )


def _validate_target(
    target: S1SpotAskTarget,
    observation_cursor: EventCursor,
) -> None:
    for name in (
        "date",
        "value_code",
        "quote_code",
        "position_id",
        "scenario_id",
        "gate_reason",
    ):
        _identifier(getattr(target, name), name)
    if len(target.date) != 8 or not target.date.isdigit():
        raise ExitFifoCoordinatorError("target date must be YYYYMMDD")
    if target.observation_cursor != observation_cursor:
        raise ExitFifoCoordinatorError(
            "target cursor differs from coordinator observation cursor"
        )
    for name in ("spot_book_cursor", "future_book_cursor"):
        cursor = getattr(target, name)
        if cursor is not None:
            if not isinstance(cursor, EventCursor):
                raise TypeError(f"{name} must be an EventCursor or None")
            if cursor > observation_cursor:
                raise ExitFifoCoordinatorError(
                    f"{name} cannot follow the observation cursor"
                )

    if not isinstance(target.gate_open, bool):
        raise TypeError("gate_open must be bool")
    if target.absolute_price_tick is not None:
        _positive_integer(target.absolute_price_tick, "absolute_price_tick")
    if target.gate_open:
        if target.gate_reason != "eligible":
            raise ExitFifoCoordinatorError(
                "an open target gate must have eligible reason"
            )
        if target.absolute_price_tick is None:
            raise ExitFifoCoordinatorError(
                "an open target gate requires an absolute tick"
            )
        if target.spot_book_cursor is None or target.future_book_cursor is None:
            raise ExitFifoCoordinatorError(
                "an open target gate requires both causal book cursors"
            )
    elif target.gate_reason == "eligible":
        raise ExitFifoCoordinatorError(
            "a closed target gate cannot have eligible reason"
        )


def _position_is_quoteable(
    position: ExitPositionView,
    target: S1SpotAskTarget,
) -> bool:
    execution_blocked = any(
        (
            position.unhedged_fill_shares,
            position.hedge_pending_shares,
            position.rollback_pending_shares,
            position.rollback_failed_shares,
        )
    )
    inventory_is_integral = (
        position.available_spot_shares > 0
        and position.available_spot_shares % position.contract_size_shares == 0
    )
    return (
        not execution_blocked
        and inventory_is_integral
        and target.gate_open
        and target.absolute_price_tick is not None
    )


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExitFifoCoordinatorError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExitFifoCoordinatorError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExitFifoCoordinatorError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "ExitFifoCoordinatorError",
    "build_exit_fifo_desired_updates",
]
