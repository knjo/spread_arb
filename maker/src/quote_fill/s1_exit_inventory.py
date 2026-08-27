"""Replayable pooled-FIFO inventory controller for the S1 Spot Ask exit route.

The controller owns no scheduler, book, or market-data state.  It emits
scheduler-neutral commands and accepts assignment/send/fill/expiry callbacks.
Only an immutable :class:`PositionEstablishedFact` may introduce inventory.

An aggregate order freezes its member quantities when its new request is
created.  Any later membership, target, or quantity change therefore cancels
and replaces the whole physical order.  The old leaves remain fillable until
the actual cancel effect.  A replacement receives a new physical identity and
queue start only when its new request is actually sent.

This primitive orders members within each aggregate and records every future
hedge and spot rollback transition.  The integrated loop must add a stricter
product/scenario gate: only the oldest unresolved
``(position_established_ns, position_id)`` cohort may receive desired state or
an actual new send.  It must not release the next cohort until the current
cohort is flat.  A successful rollback restores the same oldest cohort to
available inventory; a failed rollback remains explicitly unresolved and may
not be quoted again or release capacity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Final, Literal

from .layered import EventCursor
from .order_identity import candidate_intent_id, raw_order_fact_id
from .s1_accounting import PositionEstablishedFact

CommandKind = Literal[
    "enqueue_new",
    "withdraw_pending_new",
    "enqueue_cancel",
    "withdraw_pending_cancel",
]
NewSchedulerState = Literal["pending", "assigned"]
CancelState = Literal["none", "pending", "assigned"]
TerminalReason = Literal["filled", "actual_cancelled", "session_expired"]
RollbackReason = Literal[
    "actual_cancelled",
    "session_expired",
    "hedge_retry_timeout",
]
Operation = Literal[
    "add_position",
    "set_desired",
    "set_desired_batch",
    "new_assigned",
    "new_sent",
    "cancel_assigned",
    "cancel_sent",
    "fill",
    "hedge_sent",
    "hedge_timeout",
    "rollback_sent",
    "rollback_failed",
    "session_expiry",
]
PositionResolutionState = Literal["unresolved", "flat"]
PositionUnresolvedReason = Literal[
    "available_inventory",
    "unhedged_fill",
    "hedge_pending",
    "rollback_pending",
    "rollback_failed",
]
HedgeTerminalReason = Literal["sent", "retry_timeout"]
RollbackTerminalReason = Literal["sent", "failed"]

_ROUTE: Final = "spot_ask_future_taker"
_STAGE: Final = "exit"
_MAKER_SIDE: Final = "ask"


class ExitInventoryError(ValueError):
    """Invalid pooled exit inventory transition."""


class ExitInventoryReplayError(ExitInventoryError):
    """The append-only controller facts cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class AggregateOrderMember:
    """One frozen position allocation inside a physical aggregate order."""

    position_id: str
    position_established_ns: int
    assigned_shares: int
    filled_shares: int

    @property
    def leaves_shares(self) -> int:
        return self.assigned_shares - self.filled_shares

    @property
    def fifo_key(self) -> tuple[int, str]:
        return self.position_established_ns, self.position_id


@dataclass(frozen=True, slots=True)
class ExitControllerCommand:
    """A request mutation for the shared venue scheduler."""

    kind: CommandKind
    request_id: str
    cursor: EventCursor
    reason: str
    absolute_price_tick: int
    total_shares: int
    members: tuple[AggregateOrderMember, ...]
    candidate_intent_id: str | None
    raw_order_fact_id: str | None


@dataclass(frozen=True, slots=True)
class ExitDesiredUpdate:
    """One member of an atomic outer-coordinator desired-state update."""

    position_id: str
    absolute_target_tick: int | None
    desired_shares: int


@dataclass(frozen=True, slots=True)
class PendingAggregateOrder:
    candidate_intent_id: str
    absolute_price_tick: int
    intent_cursor: EventCursor
    scheduler_state: NewSchedulerState
    stale: bool
    members: tuple[AggregateOrderMember, ...]

    @property
    def total_shares(self) -> int:
        return sum(member.assigned_shares for member in self.members)


@dataclass(frozen=True, slots=True)
class AggregateWorkingOrder:
    candidate_intent_id: str
    raw_order_fact_id: str
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    cancel_state: CancelState
    members: tuple[AggregateOrderMember, ...]

    @property
    def total_shares(self) -> int:
        return sum(member.assigned_shares for member in self.members)

    @property
    def filled_shares(self) -> int:
        return sum(member.filled_shares for member in self.members)

    @property
    def leaves_shares(self) -> int:
        return self.total_shares - self.filled_shares


@dataclass(frozen=True, slots=True)
class AggregateOrderTerminal:
    raw_order_fact_id: str
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    terminal_cursor: EventCursor
    terminal_reason: TerminalReason
    members: tuple[AggregateOrderMember, ...]
    frozen_cancel_will_be_noop: bool

    @property
    def filled_shares(self) -> int:
        return sum(member.filled_shares for member in self.members)

    @property
    def leaves_shares(self) -> int:
        return sum(member.leaves_shares for member in self.members)


@dataclass(frozen=True, slots=True)
class ExitFillAllocation:
    """Exact own-share allocation from one aggregate physical fill."""

    allocation_id: str
    raw_order_fact_id: str
    fill_cursor: EventCursor
    position_id: str
    position_established_ns: int
    member_fifo_index: int
    allocated_shares: int
    member_leaves_before: int
    member_leaves_after: int
    partial_member_fill: bool
    execution_truth: Literal["exact"]


@dataclass(frozen=True, slots=True)
class HedgeSourceAllocation:
    """Shares from a spot-fill allocation consumed by a follow-up request."""

    allocation_id: str
    shares: int


@dataclass(frozen=True, slots=True)
class ExitHedgeUnitRequest:
    """One integral futures contract requested after enough spot exit fills."""

    request_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    trigger_cursor: EventCursor
    trigger_raw_order_fact_id: str
    contracts: int
    spot_exit_shares: int
    future_share_equivalent: int
    sources: tuple[HedgeSourceAllocation, ...]


@dataclass(frozen=True, slots=True)
class ExitRollbackRequest:
    """Reverse exact initiating spot fills after cancel/expiry/hedge timeout."""

    request_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    trigger_cursor: EventCursor
    trigger_raw_order_fact_id: str
    reason: RollbackReason
    rollback_spot_shares: int
    sources: tuple[HedgeSourceAllocation, ...]


@dataclass(frozen=True, slots=True)
class ExitHedgeUnitTerminal:
    """Terminal outcome of one integral future-buy hedge request."""

    request_id: str
    position_id: str
    terminal_cursor: EventCursor
    terminal_reason: HedgeTerminalReason
    contracts: int
    spot_exit_shares: int
    future_share_equivalent: int
    sources: tuple[HedgeSourceAllocation, ...]
    resulting_rollback_request_id: str | None


@dataclass(frozen=True, slots=True)
class ExitRollbackTerminal:
    """Terminal outcome of one causally linked spot-buy rollback request."""

    request_id: str
    position_id: str
    terminal_cursor: EventCursor
    terminal_reason: RollbackTerminalReason
    rollback_spot_shares: int
    sources: tuple[HedgeSourceAllocation, ...]
    restored_available_shares: int
    failed_unresolved_shares: int


@dataclass(frozen=True, slots=True)
class ExitPositionView:
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    position_established_ns: int
    contract_size_shares: int
    available_spot_shares: int
    desired_absolute_target_tick: int | None
    desired_shares: int
    allocation_id: str | None
    allocation_kind: Literal["candidate", "working"] | None
    unhedged_fill_shares: int
    rollback_pending_shares: int
    hedge_pending_shares: int = 0
    hedged_exit_shares: int = 0
    rollback_failed_shares: int = 0
    unresolved_shares: int = 0
    resolution_state: PositionResolutionState = "unresolved"
    unresolved_reasons: tuple[PositionUnresolvedReason, ...] = ()
    capacity_releasable: bool = False

    @property
    def fifo_key(self) -> tuple[int, str]:
        return self.position_established_ns, self.position_id


@dataclass(frozen=True, slots=True)
class ExitInventoryFact:
    """One source callback plus all deterministic derived output facts."""

    sequence: int
    operation: Operation
    cursor: EventCursor
    position_fact: PositionEstablishedFact | None
    position_id: str | None
    absolute_target_tick: int | None
    desired_shares: int | None
    candidate_intent_id: str | None
    raw_order_fact_id: str | None
    actual_price_tick: int | None
    fill_shares: int | None
    commands: tuple[ExitControllerCommand, ...]
    order_updates: tuple[AggregateWorkingOrder, ...]
    terminals: tuple[AggregateOrderTerminal, ...]
    fill_allocations: tuple[ExitFillAllocation, ...]
    hedge_unit_requests: tuple[ExitHedgeUnitRequest, ...]
    rollback_requests: tuple[ExitRollbackRequest, ...]
    state_digest: str
    desired_updates: tuple[ExitDesiredUpdate, ...] = ()
    followup_request_id: str | None = None
    hedge_terminals: tuple[ExitHedgeUnitTerminal, ...] = ()
    rollback_terminals: tuple[ExitRollbackTerminal, ...] = ()
    digest_version: int = 2


@dataclass(slots=True)
class _Position:
    fact: PositionEstablishedFact
    contract_size_shares: int
    available_spot_shares: int
    desired_absolute_target_tick: int | None
    desired_shares: int
    allocation_id: str | None = None
    allocation_kind: Literal["candidate", "working"] | None = None
    hedge_pending_shares: int = 0
    hedged_exit_shares: int = 0
    rollback_pending_shares: int = 0
    rollback_failed_shares: int = 0
    next_hedge_unit: int = 1
    next_rollback: int = 1


@dataclass(slots=True)
class _Member:
    position_id: str
    position_established_ns: int
    assigned_shares: int
    filled_shares: int = 0

    @property
    def leaves_shares(self) -> int:
        return self.assigned_shares - self.filled_shares

    @property
    def fifo_key(self) -> tuple[int, str]:
        return self.position_established_ns, self.position_id


@dataclass(slots=True)
class _PendingNew:
    candidate_intent_id: str
    absolute_price_tick: int
    intent_cursor: EventCursor
    scheduler_state: NewSchedulerState
    members: list[_Member]
    stale: bool = False


@dataclass(slots=True)
class _Working:
    candidate_intent_id: str
    raw_order_fact_id: str
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    members: list[_Member]
    cancel_state: CancelState = "none"

    @property
    def leaves_shares(self) -> int:
        return sum(member.leaves_shares for member in self.members)


@dataclass(slots=True)
class _UnhedgedSource:
    allocation_id: str
    shares: int


class S1ExitInventoryController:
    """One product/session/scenario pooled Spot Ask order state machine.

    The owner remains responsible for the global oldest-unresolved cohort gate
    described in the module contract.
    """

    def __init__(
        self,
        *,
        Date: str,
        ValueCode: str,
        QuoteCode: str,
        scenario_id: str,
        route: str = _ROUTE,
        stage: str = _STAGE,
        maker_side: Literal["ask"] = _MAKER_SIDE,
    ) -> None:
        self.Date = _valid_date(Date, "Date")
        self.ValueCode = _identifier(ValueCode, "ValueCode")
        self.QuoteCode = _identifier(QuoteCode, "QuoteCode")
        self.scenario_id = _identifier(scenario_id, "scenario_id")
        self.route = _identifier(route, "route")
        self.stage = _identifier(stage, "stage")
        if self.route != _ROUTE or self.stage != _STAGE or maker_side != _MAKER_SIDE:
            raise ExitInventoryError(
                "S1 exit inventory requires spot_ask_future_taker/exit/ask semantics"
            )
        self.maker_side = maker_side
        self._last_cursor: EventCursor | None = None
        self._session_expired = False
        self._positions: dict[str, _Position] = {}
        self._pending_by_tick: dict[int, _PendingNew] = {}
        self._pending_by_id: dict[str, _PendingNew] = {}
        self._working_by_tick: dict[int, _Working] = {}
        self._working_by_id: dict[str, _Working] = {}
        self._terminals_by_id: dict[str, AggregateOrderTerminal] = {}
        self._consumed_cancel_callbacks: set[str] = set()
        self._unhedged_by_position: dict[str, list[_UnhedgedSource]] = {}
        self._facts: list[ExitInventoryFact] = []
        self._fill_allocations: list[ExitFillAllocation] = []
        self._hedge_requests: list[ExitHedgeUnitRequest] = []
        self._pending_hedges: dict[str, ExitHedgeUnitRequest] = {}
        self._hedge_terminals_by_id: dict[str, ExitHedgeUnitTerminal] = {}
        self._sent_hedge_shares_by_position: dict[str, int] = {}
        self._rollback_requests: list[ExitRollbackRequest] = []
        self._pending_rollbacks: dict[str, ExitRollbackRequest] = {}
        self._rollback_terminals_by_id: dict[str, ExitRollbackTerminal] = {}
        self._failed_rollback_shares_by_position: dict[str, int] = {}
        self._next_fill_allocation = 1
        self._history_digest = "0" * 64

    @property
    def facts(self) -> tuple[ExitInventoryFact, ...]:
        return tuple(self._facts)

    @property
    def positions(self) -> tuple[ExitPositionView, ...]:
        return tuple(
            self._position_view(position)
            for position in sorted(
                self._positions.values(), key=lambda item: item.fact.fifo_key
            )
        )

    @property
    def pending_orders(self) -> tuple[PendingAggregateOrder, ...]:
        return tuple(
            self._pending_snapshot(pending)
            for pending in sorted(
                self._pending_by_tick.values(),
                key=lambda item: (
                    item.absolute_price_tick,
                    item.candidate_intent_id,
                ),
            )
        )

    @property
    def working_orders(self) -> tuple[AggregateWorkingOrder, ...]:
        return tuple(
            self._working_snapshot(order)
            for order in sorted(
                self._working_by_id.values(),
                key=lambda item: (
                    item.actual_start_cursor,
                    item.raw_order_fact_id,
                ),
            )
        )

    @property
    def terminals(self) -> tuple[AggregateOrderTerminal, ...]:
        return tuple(self._terminals_by_id.values())

    @property
    def fill_allocations(self) -> tuple[ExitFillAllocation, ...]:
        return tuple(self._fill_allocations)

    @property
    def hedge_unit_requests(self) -> tuple[ExitHedgeUnitRequest, ...]:
        return tuple(self._hedge_requests)

    @property
    def pending_hedge_unit_requests(self) -> tuple[ExitHedgeUnitRequest, ...]:
        return tuple(
            self._pending_hedges[request_id]
            for request_id in sorted(self._pending_hedges)
        )

    @property
    def hedge_unit_terminals(self) -> tuple[ExitHedgeUnitTerminal, ...]:
        return tuple(
            self._hedge_terminals_by_id[request_id]
            for request_id in sorted(self._hedge_terminals_by_id)
        )

    @property
    def rollback_requests(self) -> tuple[ExitRollbackRequest, ...]:
        return tuple(self._rollback_requests)

    @property
    def pending_rollback_requests(self) -> tuple[ExitRollbackRequest, ...]:
        return tuple(
            self._pending_rollbacks[request_id]
            for request_id in sorted(self._pending_rollbacks)
        )

    @property
    def rollback_terminals(self) -> tuple[ExitRollbackTerminal, ...]:
        return tuple(
            self._rollback_terminals_by_id[request_id]
            for request_id in sorted(self._rollback_terminals_by_id)
        )

    @property
    def unresolved_positions(self) -> tuple[ExitPositionView, ...]:
        return tuple(
            position
            for position in self.positions
            if position.resolution_state == "unresolved"
        )

    @property
    def flat_positions(self) -> tuple[ExitPositionView, ...]:
        return tuple(
            position
            for position in self.positions
            if position.resolution_state == "flat"
        )

    @property
    def session_expired(self) -> bool:
        return self._session_expired

    def add_paired_position(
        self,
        position_fact: PositionEstablishedFact,
        *,
        absolute_target_tick: int,
        cursor: EventCursor,
        desired_shares: int | None = None,
    ) -> ExitInventoryFact:
        """Add exactly paired inventory and reconcile its aggregate new intent."""

        self._validate_next_cursor(cursor)
        if self._session_expired:
            raise ExitInventoryError("cannot add paired inventory after expiry")
        contract_size = self._validate_position_fact(position_fact, cursor)
        target = _positive_integer(absolute_target_tick, "absolute_target_tick")
        shares = (
            position_fact.spot_shares
            if desired_shares is None
            else _positive_integer(desired_shares, "desired_shares")
        )
        if shares > position_fact.spot_shares:
            raise ExitInventoryError("desired shares exceed paired spot inventory")
        if shares % contract_size != 0:
            raise ExitInventoryError(
                "initial desired shares must contain integral hedge units"
            )
        position = _Position(
            fact=position_fact,
            contract_size_shares=contract_size,
            available_spot_shares=position_fact.spot_shares,
            desired_absolute_target_tick=target,
            desired_shares=shares,
        )
        self._positions[position_fact.position_id] = position
        self._unhedged_by_position[position_fact.position_id] = []
        commands = self._reconcile(cursor, "position_added")
        return self._finish(
            "add_position",
            cursor,
            position_fact=position_fact,
            position_id=position_fact.position_id,
            absolute_target_tick=target,
            desired_shares=shares,
            commands=commands,
        )

    def set_position_desired(
        self,
        position_id: str,
        *,
        absolute_target_tick: int | None,
        desired_shares: int,
        cursor: EventCursor,
    ) -> ExitInventoryFact:
        """Replace one position's desired target/quantity without double allocation."""

        self._validate_next_cursor(cursor)
        if self._session_expired:
            raise ExitInventoryError("cannot change desired inventory after expiry")
        position = self._require_position(position_id)
        target, shares = self._validated_desired(
            position,
            absolute_target_tick=absolute_target_tick,
            desired_shares=desired_shares,
        )
        position.desired_absolute_target_tick = target
        position.desired_shares = shares
        commands = self._reconcile(cursor, "desired_changed")
        return self._finish(
            "set_desired",
            cursor,
            position_id=position_id,
            absolute_target_tick=target,
            desired_shares=shares,
            commands=commands,
        )

    def set_positions_desired(
        self,
        updates: Sequence[ExitDesiredUpdate],
        *,
        cursor: EventCursor,
    ) -> ExitInventoryFact:
        """Apply a complete desired-state batch before one reconciliation.

        The outer FIFO coordinator uses this method when releasing one oldest
        cohort and withdrawing another.  Every row is validated before any
        desired state is mutated, so an invalid row cannot expose a transient
        set of scheduler commands or a partially applied batch.
        """

        self._validate_next_cursor(cursor)
        if self._session_expired:
            raise ExitInventoryError("cannot change desired inventory after expiry")
        if isinstance(updates, (str, bytes)) or not isinstance(updates, Sequence):
            raise TypeError("updates must be a sequence of ExitDesiredUpdate")
        if not updates:
            raise ExitInventoryError("desired update batch cannot be empty")

        validated: list[tuple[_Position, ExitDesiredUpdate]] = []
        seen: set[str] = set()
        for update in updates:
            if not isinstance(update, ExitDesiredUpdate):
                raise TypeError("updates must contain ExitDesiredUpdate values")
            position_id = _identifier(update.position_id, "position_id")
            if position_id in seen:
                raise ExitInventoryError("desired update batch repeats position_id")
            seen.add(position_id)
            position = self._require_position(position_id)
            target, shares = self._validated_desired(
                position,
                absolute_target_tick=update.absolute_target_tick,
                desired_shares=update.desired_shares,
            )
            validated.append(
                (
                    position,
                    ExitDesiredUpdate(position_id, target, shares),
                )
            )

        validated.sort(key=lambda item: item[0].fact.fifo_key)
        canonical_updates = tuple(update for _, update in validated)
        for position, update in validated:
            position.desired_absolute_target_tick = update.absolute_target_tick
            position.desired_shares = update.desired_shares
        commands = self._reconcile(cursor, "desired_batch_changed")
        return self._finish(
            "set_desired_batch",
            cursor,
            desired_updates=canonical_updates,
            commands=commands,
        )

    def on_new_assigned(
        self, candidate_id: str, cursor: EventCursor
    ) -> ExitInventoryFact:
        """Freeze a scheduler assignment before its actual new-request send."""

        self._validate_next_cursor(cursor)
        candidate_id = _identifier(candidate_id, "candidate_id")
        pending = self._require_pending(candidate_id)
        if pending.scheduler_state != "pending":
            raise ExitInventoryError("new request was already assigned")
        pending.scheduler_state = "assigned"
        return self._finish(
            "new_assigned",
            cursor,
            candidate_intent_id=candidate_id,
        )

    def on_new_sent(
        self,
        candidate_id: str,
        actual_send_cursor: EventCursor,
        actual_price_tick: int,
    ) -> ExitInventoryFact:
        """Create a physical order only at the actual new-send cursor."""

        self._validate_next_cursor(actual_send_cursor)
        candidate_id = _identifier(candidate_id, "candidate_id")
        pending = self._require_pending(candidate_id)
        if pending.scheduler_state != "assigned":
            raise ExitInventoryError("new request must be assigned before send")
        tick = _positive_integer(actual_price_tick, "actual_price_tick")
        if tick != pending.absolute_price_tick:
            raise ExitInventoryError(
                "actual new price must equal the frozen absolute target"
            )
        if actual_send_cursor <= pending.intent_cursor:
            raise ExitInventoryError("actual new send must follow its intent cursor")
        if tick in self._working_by_tick:
            raise ExitInventoryError("an aggregate order already works at this price")
        raw_id = raw_order_fact_id(
            **self._identity_fields(tick),
            actual_start_cursor=actual_send_cursor,
        )
        if raw_id in self._working_by_id or raw_id in self._terminals_by_id:
            raise ExitInventoryError("physical raw order identity was already used")
        members = [
            _Member(
                member.position_id,
                member.position_established_ns,
                member.assigned_shares,
                member.filled_shares,
            )
            for member in pending.members
        ]
        order = _Working(
            candidate_intent_id=candidate_id,
            raw_order_fact_id=raw_id,
            absolute_price_tick=tick,
            actual_start_cursor=actual_send_cursor,
            members=members,
        )
        del self._pending_by_tick[pending.absolute_price_tick]
        del self._pending_by_id[candidate_id]
        for member in members:
            position = self._positions[member.position_id]
            if (
                position.allocation_id != candidate_id
                or position.allocation_kind != "candidate"
            ):
                raise RuntimeError("pending membership allocation diverged")
            position.allocation_id = raw_id
            position.allocation_kind = "working"
        self._working_by_tick[tick] = order
        self._working_by_id[raw_id] = order
        commands = self._reconcile(actual_send_cursor, "sent_membership_stale")
        return self._finish(
            "new_sent",
            actual_send_cursor,
            candidate_intent_id=candidate_id,
            raw_order_fact_id=raw_id,
            actual_price_tick=tick,
            commands=commands,
            order_updates=(self._working_snapshot(order),),
        )

    def on_cancel_assigned(self, raw_id: str, cursor: EventCursor) -> ExitInventoryFact:
        """Freeze a cancel assignment before same-cursor fill allocation."""

        self._validate_next_cursor(cursor)
        raw_id = _identifier(raw_id, "raw_id")
        order = self._require_working(raw_id)
        if order.cancel_state != "pending":
            raise ExitInventoryError("only a pending cancel can be assigned")
        order.cancel_state = "assigned"
        return self._finish(
            "cancel_assigned",
            cursor,
            raw_order_fact_id=raw_id,
            order_updates=(self._working_snapshot(order),),
        )

    def on_cancel_sent(
        self, raw_id: str, actual_send_cursor: EventCursor
    ) -> ExitInventoryFact:
        """Apply an assigned cancel or consume its frozen post-fill noop."""

        self._validate_next_cursor(actual_send_cursor)
        raw_id = _identifier(raw_id, "raw_id")
        if raw_id in self._consumed_cancel_callbacks:
            raise ExitInventoryError("cancel send callback was already consumed")
        prior_terminal = self._terminals_by_id.get(raw_id)
        if prior_terminal is not None:
            if not prior_terminal.frozen_cancel_will_be_noop:
                raise ExitInventoryError(
                    "terminal order has no frozen cancel assignment"
                )
            self._consumed_cancel_callbacks.add(raw_id)
            return self._finish(
                "cancel_sent",
                actual_send_cursor,
                raw_order_fact_id=raw_id,
            )
        order = self._require_working(raw_id)
        if order.cancel_state != "assigned":
            raise ExitInventoryError("cancel must be assigned before it is sent")
        terminal = self._terminalize(order, actual_send_cursor, "actual_cancelled")
        self._consumed_cancel_callbacks.add(raw_id)
        rollbacks = self._emit_subunit_rollbacks(
            terminal.members,
            actual_send_cursor,
            raw_id,
            "actual_cancelled",
        )
        commands = self._reconcile(actual_send_cursor, "after_actual_cancel")
        return self._finish(
            "cancel_sent",
            actual_send_cursor,
            raw_order_fact_id=raw_id,
            commands=commands,
            terminals=(terminal,),
            rollback_requests=rollbacks,
        )

    def on_fill(
        self,
        raw_id: str,
        fill_cursor: EventCursor,
        fill_shares: int,
    ) -> ExitInventoryFact:
        """Allocate one exact physical fill across frozen members in FIFO order."""

        self._validate_next_cursor(fill_cursor)
        raw_id = _identifier(raw_id, "raw_id")
        order = self._require_working(raw_id)
        shares = _positive_integer(fill_shares, "fill_shares")
        if fill_cursor <= order.actual_start_cursor:
            raise ExitInventoryError("fill must follow the actual new-send cursor")
        if shares > order.leaves_shares:
            raise ExitInventoryError("fill exceeds physical aggregate leaves")
        allocations: list[ExitFillAllocation] = []
        hedge_requests: list[ExitHedgeUnitRequest] = []
        remaining = shares
        for member_index, member in enumerate(order.members):
            if remaining == 0:
                break
            before = member.leaves_shares
            if before == 0:
                continue
            allocated = min(remaining, before)
            position = self._positions[member.position_id]
            if allocated > position.available_spot_shares:
                raise RuntimeError("order allocation exceeds position inventory")
            member.filled_shares += allocated
            position.available_spot_shares -= allocated
            position.desired_shares = max(0, position.desired_shares - allocated)
            after = member.leaves_shares
            allocation = ExitFillAllocation(
                allocation_id=(
                    f"exit_fill_allocation/{self.Date}/{self.ValueCode}/"
                    f"{raw_id}/{self._next_fill_allocation:012d}"
                ),
                raw_order_fact_id=raw_id,
                fill_cursor=fill_cursor,
                position_id=member.position_id,
                position_established_ns=member.position_established_ns,
                member_fifo_index=member_index,
                allocated_shares=allocated,
                member_leaves_before=before,
                member_leaves_after=after,
                partial_member_fill=after > 0,
                execution_truth="exact",
            )
            self._next_fill_allocation += 1
            allocations.append(allocation)
            self._fill_allocations.append(allocation)
            self._unhedged_by_position[member.position_id].append(
                _UnhedgedSource(allocation.allocation_id, allocated)
            )
            hedge_requests.extend(
                self._emit_complete_hedge_units(position, fill_cursor, raw_id)
            )
            remaining -= allocated
        if remaining != 0:
            raise RuntimeError("FIFO allocation failed to consume exact fill")

        commands: list[ExitControllerCommand] = []
        terminals: tuple[AggregateOrderTerminal, ...] = ()
        order_updates: tuple[AggregateWorkingOrder, ...] = ()
        if order.leaves_shares == 0:
            if order.cancel_state == "pending":
                commands.append(
                    self._cancel_command(
                        order, fill_cursor, "fill_terminal", withdraw=True
                    )
                )
                order.cancel_state = "none"
            terminal = self._terminalize(order, fill_cursor, "filled")
            terminals = (terminal,)
            commands.extend(self._reconcile(fill_cursor, "after_full_fill"))
        else:
            commands.extend(self._reconcile(fill_cursor, "after_partial_fill"))
            order_updates = (self._working_snapshot(order),)
        return self._finish(
            "fill",
            fill_cursor,
            raw_order_fact_id=raw_id,
            fill_shares=shares,
            commands=commands,
            order_updates=order_updates,
            terminals=terminals,
            fill_allocations=tuple(allocations),
            hedge_unit_requests=tuple(hedge_requests),
        )

    def on_hedge_sent(
        self, request_id: str, actual_send_cursor: EventCursor
    ) -> ExitInventoryFact:
        """Complete one exact integral future-buy hedge request."""

        self._validate_next_cursor(actual_send_cursor)
        request_id = _identifier(request_id, "request_id")
        request = self._require_pending_hedge(request_id)
        if actual_send_cursor <= request.trigger_cursor:
            raise ExitInventoryError("hedge send must follow its trigger cursor")
        position = self._positions[request.position_id]
        shares = request.future_share_equivalent
        if position.hedge_pending_shares < shares:
            raise ExitInventoryError("hedge completion exceeds pending quantity")
        terminal = ExitHedgeUnitTerminal(
            request_id=request_id,
            position_id=request.position_id,
            terminal_cursor=actual_send_cursor,
            terminal_reason="sent",
            contracts=request.contracts,
            spot_exit_shares=request.spot_exit_shares,
            future_share_equivalent=shares,
            sources=request.sources,
            resulting_rollback_request_id=None,
        )
        del self._pending_hedges[request_id]
        self._hedge_terminals_by_id[request_id] = terminal
        position.hedge_pending_shares -= shares
        position.hedged_exit_shares += shares
        self._sent_hedge_shares_by_position[request.position_id] = (
            self._sent_hedge_shares_by_position.get(request.position_id, 0) + shares
        )
        return self._finish(
            "hedge_sent",
            actual_send_cursor,
            followup_request_id=request_id,
            hedge_terminals=(terminal,),
        )

    def on_hedge_timeout(
        self, request_id: str, timeout_cursor: EventCursor
    ) -> ExitInventoryFact:
        """Terminalize a timed-out hedge and request an exact spot-buy rollback."""

        self._validate_next_cursor(timeout_cursor)
        request_id = _identifier(request_id, "request_id")
        request = self._require_pending_hedge(request_id)
        if timeout_cursor <= request.trigger_cursor:
            raise ExitInventoryError("hedge timeout must follow its trigger cursor")
        position = self._positions[request.position_id]
        shares = request.future_share_equivalent
        if position.hedge_pending_shares < shares:
            raise ExitInventoryError("hedge timeout exceeds pending quantity")

        del self._pending_hedges[request_id]
        position.hedge_pending_shares -= shares
        position.desired_absolute_target_tick = None
        position.desired_shares = 0
        rollback = self._emit_rollback_request(
            position,
            timeout_cursor,
            request.trigger_raw_order_fact_id,
            "hedge_retry_timeout",
            shares,
            request.sources,
        )
        if rollback.sources != request.sources:
            raise RuntimeError("hedge-timeout rollback changed source allocations")
        terminal = ExitHedgeUnitTerminal(
            request_id=request_id,
            position_id=request.position_id,
            terminal_cursor=timeout_cursor,
            terminal_reason="retry_timeout",
            contracts=request.contracts,
            spot_exit_shares=request.spot_exit_shares,
            future_share_equivalent=shares,
            sources=request.sources,
            resulting_rollback_request_id=rollback.request_id,
        )
        self._hedge_terminals_by_id[request_id] = terminal
        commands = self._reconcile(timeout_cursor, "after_hedge_timeout")
        return self._finish(
            "hedge_timeout",
            timeout_cursor,
            followup_request_id=request_id,
            commands=commands,
            rollback_requests=(rollback,),
            hedge_terminals=(terminal,),
        )

    def on_rollback_sent(
        self, request_id: str, actual_send_cursor: EventCursor
    ) -> ExitInventoryFact:
        """Complete a spot-buy rollback and restore its exact paired inventory."""

        self._validate_next_cursor(actual_send_cursor)
        request_id = _identifier(request_id, "request_id")
        request = self._require_pending_rollback(request_id)
        if actual_send_cursor <= request.trigger_cursor:
            raise ExitInventoryError("rollback send must follow its trigger cursor")
        position = self._positions[request.position_id]
        shares = request.rollback_spot_shares
        if position.rollback_pending_shares < shares:
            raise ExitInventoryError("rollback completion exceeds pending quantity")
        if position.available_spot_shares + shares > position.fact.spot_shares:
            raise ExitInventoryError("rollback completion over-restores inventory")

        terminal = ExitRollbackTerminal(
            request_id=request_id,
            position_id=request.position_id,
            terminal_cursor=actual_send_cursor,
            terminal_reason="sent",
            rollback_spot_shares=shares,
            sources=request.sources,
            restored_available_shares=shares,
            failed_unresolved_shares=0,
        )
        del self._pending_rollbacks[request_id]
        self._rollback_terminals_by_id[request_id] = terminal
        position.rollback_pending_shares -= shares
        position.available_spot_shares += shares
        return self._finish(
            "rollback_sent",
            actual_send_cursor,
            followup_request_id=request_id,
            rollback_terminals=(terminal,),
        )

    def on_rollback_failed(
        self, request_id: str, failure_cursor: EventCursor
    ) -> ExitInventoryFact:
        """Make a failed rollback terminal while preserving naked unresolved risk."""

        self._validate_next_cursor(failure_cursor)
        request_id = _identifier(request_id, "request_id")
        request = self._require_pending_rollback(request_id)
        if failure_cursor <= request.trigger_cursor:
            raise ExitInventoryError("rollback failure must follow its trigger cursor")
        position = self._positions[request.position_id]
        shares = request.rollback_spot_shares
        if position.rollback_pending_shares < shares:
            raise ExitInventoryError("rollback failure exceeds pending quantity")

        terminal = ExitRollbackTerminal(
            request_id=request_id,
            position_id=request.position_id,
            terminal_cursor=failure_cursor,
            terminal_reason="failed",
            rollback_spot_shares=shares,
            sources=request.sources,
            restored_available_shares=0,
            failed_unresolved_shares=shares,
        )
        del self._pending_rollbacks[request_id]
        self._rollback_terminals_by_id[request_id] = terminal
        position.rollback_pending_shares -= shares
        position.rollback_failed_shares += shares
        self._failed_rollback_shares_by_position[request.position_id] = (
            self._failed_rollback_shares_by_position.get(request.position_id, 0)
            + shares
        )
        position.desired_absolute_target_tick = None
        position.desired_shares = 0
        commands = self._reconcile(failure_cursor, "after_rollback_failed")
        return self._finish(
            "rollback_failed",
            failure_cursor,
            followup_request_id=request_id,
            commands=commands,
            rollback_terminals=(terminal,),
        )

    def session_expiry(self, cursor: EventCursor) -> ExitInventoryFact:
        """Expire day orders and roll back every still-sub-unit spot exit."""

        self._validate_next_cursor(cursor)
        if self._session_expired:
            raise ExitInventoryError("session expiry was already applied")
        if any(
            pending.scheduler_state == "assigned"
            for pending in self._pending_by_id.values()
        ):
            raise ExitInventoryError(
                "cannot expire with an assigned new callback outstanding"
            )
        if any(
            order.cancel_state == "assigned" for order in self._working_by_id.values()
        ):
            raise ExitInventoryError(
                "cannot expire with an assigned cancel callback outstanding"
            )
        commands: list[ExitControllerCommand] = []
        for pending in tuple(
            sorted(
                self._pending_by_id.values(),
                key=lambda item: item.candidate_intent_id,
            )
        ):
            commands.append(self._withdraw_pending(pending, cursor, "session_expiry"))
        terminals: list[AggregateOrderTerminal] = []
        rollback_requests: list[ExitRollbackRequest] = []
        for order in tuple(
            sorted(
                self._working_by_id.values(),
                key=lambda item: (
                    item.actual_start_cursor,
                    item.raw_order_fact_id,
                ),
            )
        ):
            if order.cancel_state == "pending":
                commands.append(
                    self._cancel_command(order, cursor, "session_expiry", withdraw=True)
                )
                order.cancel_state = "none"
            terminal = self._terminalize(order, cursor, "session_expired")
            terminals.append(terminal)
            rollback_requests.extend(
                self._emit_subunit_rollbacks(
                    terminal.members,
                    cursor,
                    terminal.raw_order_fact_id,
                    "session_expired",
                )
            )
        self._session_expired = True
        return self._finish(
            "session_expiry",
            cursor,
            commands=commands,
            terminals=tuple(terminals),
            rollback_requests=tuple(rollback_requests),
        )

    def verify(self) -> None:
        """Replay every source fact and compare all derived facts and final state."""

        replayed = replay_exit_inventory_facts(
            self._facts,
            Date=self.Date,
            ValueCode=self.ValueCode,
            QuoteCode=self.QuoteCode,
            scenario_id=self.scenario_id,
            route=self.route,
            stage=self.stage,
            maker_side=self.maker_side,
        )
        if replayed._state_digest() != self._state_digest():
            raise ExitInventoryReplayError(
                "replayed exit inventory final state does not match"
            )

    def _reconcile(
        self, cursor: EventCursor, reason: str
    ) -> list[ExitControllerCommand]:
        commands: list[ExitControllerCommand] = []
        while True:
            withdrew_pending = False
            for pending in tuple(
                sorted(
                    self._pending_by_id.values(),
                    key=lambda item: item.candidate_intent_id,
                )
            ):
                if self._pending_signature(pending) == self._desired_signature(
                    pending.absolute_price_tick,
                    pending.candidate_intent_id,
                ):
                    continue
                pending.stale = True
                if pending.scheduler_state == "pending":
                    commands.append(self._withdraw_pending(pending, cursor, reason))
                    withdrew_pending = True
            if not withdrew_pending:
                break

        for order in tuple(
            sorted(
                self._working_by_id.values(),
                key=lambda item: item.raw_order_fact_id,
            )
        ):
            if order.cancel_state == "none" and self._working_signature(
                order
            ) != self._desired_signature(
                order.absolute_price_tick, order.raw_order_fact_id
            ):
                order.cancel_state = "pending"
                commands.append(
                    self._cancel_command(order, cursor, reason, withdraw=False)
                )

        if self._session_expired:
            return commands
        grouped: dict[int, list[_Position]] = {}
        for position in self._positions.values():
            tick = position.desired_absolute_target_tick
            if (
                tick is None
                or position.desired_shares == 0
                or position.allocation_id is not None
                or position.rollback_pending_shares
            ):
                continue
            grouped.setdefault(tick, []).append(position)
        for tick in sorted(grouped):
            if tick in self._pending_by_tick or tick in self._working_by_tick:
                continue
            members = [
                _Member(
                    position.fact.position_id,
                    position.fact.position_established_ns,
                    position.desired_shares,
                )
                for position in sorted(
                    grouped[tick], key=lambda item: item.fact.fifo_key
                )
            ]
            commands.append(self._enqueue_new(tick, members, cursor, reason))
        return commands

    def _enqueue_new(
        self,
        tick: int,
        members: list[_Member],
        cursor: EventCursor,
        reason: str,
    ) -> ExitControllerCommand:
        if not members or any(member.assigned_shares <= 0 for member in members):
            raise RuntimeError("aggregate new requires positive frozen membership")
        if tick in self._pending_by_tick or tick in self._working_by_tick:
            raise RuntimeError("aggregate price is already reserved")
        candidate_id = candidate_intent_id(
            **self._identity_fields(tick), intent_cursor=cursor
        )
        if candidate_id in self._pending_by_id:
            raise RuntimeError("candidate identity collision")
        pending = _PendingNew(
            candidate_intent_id=candidate_id,
            absolute_price_tick=tick,
            intent_cursor=cursor,
            scheduler_state="pending",
            members=members,
        )
        for member in members:
            position = self._positions[member.position_id]
            if position.allocation_id is not None:
                raise RuntimeError("position inventory was allocated twice")
            position.allocation_id = candidate_id
            position.allocation_kind = "candidate"
        self._pending_by_tick[tick] = pending
        self._pending_by_id[candidate_id] = pending
        snapshot = self._pending_snapshot(pending)
        return ExitControllerCommand(
            kind="enqueue_new",
            request_id=f"{candidate_id}/new",
            cursor=cursor,
            reason=reason,
            absolute_price_tick=tick,
            total_shares=snapshot.total_shares,
            members=snapshot.members,
            candidate_intent_id=candidate_id,
            raw_order_fact_id=None,
        )

    def _withdraw_pending(
        self, pending: _PendingNew, cursor: EventCursor, reason: str
    ) -> ExitControllerCommand:
        if pending.scheduler_state != "pending":
            raise RuntimeError("an assigned new request cannot be withdrawn")
        snapshot = self._pending_snapshot(pending)
        del self._pending_by_tick[pending.absolute_price_tick]
        del self._pending_by_id[pending.candidate_intent_id]
        for member in pending.members:
            position = self._positions[member.position_id]
            if position.allocation_id != pending.candidate_intent_id:
                raise RuntimeError("pending allocation ownership diverged")
            position.allocation_id = None
            position.allocation_kind = None
        return ExitControllerCommand(
            kind="withdraw_pending_new",
            request_id=f"{pending.candidate_intent_id}/new",
            cursor=cursor,
            reason=reason,
            absolute_price_tick=pending.absolute_price_tick,
            total_shares=snapshot.total_shares,
            members=snapshot.members,
            candidate_intent_id=pending.candidate_intent_id,
            raw_order_fact_id=None,
        )

    def _cancel_command(
        self,
        order: _Working,
        cursor: EventCursor,
        reason: str,
        *,
        withdraw: bool,
    ) -> ExitControllerCommand:
        snapshot = self._working_snapshot(order)
        return ExitControllerCommand(
            kind=("withdraw_pending_cancel" if withdraw else "enqueue_cancel"),
            request_id=f"{order.raw_order_fact_id}/cancel",
            cursor=cursor,
            reason=reason,
            absolute_price_tick=order.absolute_price_tick,
            total_shares=snapshot.leaves_shares,
            members=snapshot.members,
            candidate_intent_id=order.candidate_intent_id,
            raw_order_fact_id=order.raw_order_fact_id,
        )

    def _terminalize(
        self,
        order: _Working,
        cursor: EventCursor,
        reason: TerminalReason,
    ) -> AggregateOrderTerminal:
        members = tuple(self._member_snapshot(member) for member in order.members)
        terminal = AggregateOrderTerminal(
            raw_order_fact_id=order.raw_order_fact_id,
            absolute_price_tick=order.absolute_price_tick,
            actual_start_cursor=order.actual_start_cursor,
            terminal_cursor=cursor,
            terminal_reason=reason,
            members=members,
            frozen_cancel_will_be_noop=(
                order.cancel_state == "assigned" and reason != "actual_cancelled"
            ),
        )
        del self._working_by_tick[order.absolute_price_tick]
        del self._working_by_id[order.raw_order_fact_id]
        for member in order.members:
            position = self._positions[member.position_id]
            if position.allocation_id != order.raw_order_fact_id:
                raise RuntimeError("working allocation ownership diverged")
            position.allocation_id = None
            position.allocation_kind = None
        self._terminals_by_id[order.raw_order_fact_id] = terminal
        return terminal

    def _emit_complete_hedge_units(
        self,
        position: _Position,
        cursor: EventCursor,
        raw_id: str,
    ) -> list[ExitHedgeUnitRequest]:
        emitted: list[ExitHedgeUnitRequest] = []
        sources = self._unhedged_by_position[position.fact.position_id]
        while sum(source.shares for source in sources) >= position.contract_size_shares:
            consumed = self._consume_sources(sources, position.contract_size_shares)
            request = ExitHedgeUnitRequest(
                request_id=(
                    f"exit_hedge_unit/{self.Date}/{self.ValueCode}/"
                    f"{position.fact.position_id}/{raw_id}/"
                    f"{position.next_hedge_unit:08d}"
                ),
                position_id=position.fact.position_id,
                value_code=position.fact.value_code,
                scenario_id=position.fact.scenario_id,
                capacity_id=position.fact.capacity_id,
                trigger_cursor=cursor,
                trigger_raw_order_fact_id=raw_id,
                contracts=1,
                spot_exit_shares=position.contract_size_shares,
                future_share_equivalent=position.contract_size_shares,
                sources=consumed,
            )
            if (
                request.request_id in self._pending_hedges
                or request.request_id in self._hedge_terminals_by_id
            ):
                raise RuntimeError("exit hedge request identity was already used")
            position.next_hedge_unit += 1
            position.hedge_pending_shares += request.future_share_equivalent
            self._hedge_requests.append(request)
            self._pending_hedges[request.request_id] = request
            emitted.append(request)
        return emitted

    def _emit_subunit_rollbacks(
        self,
        members: tuple[AggregateOrderMember, ...],
        cursor: EventCursor,
        raw_id: str,
        reason: RollbackReason,
    ) -> tuple[ExitRollbackRequest, ...]:
        requests: list[ExitRollbackRequest] = []
        for member in members:
            position = self._positions[member.position_id]
            sources = self._unhedged_by_position[member.position_id]
            shares = sum(source.shares for source in sources)
            if shares == 0:
                continue
            if shares >= position.contract_size_shares:
                raise RuntimeError("complete hedge unit remained unrequested")
            consumed = self._consume_sources(sources, shares)
            request = self._emit_rollback_request(
                position,
                cursor,
                raw_id,
                reason,
                shares,
                consumed,
            )
            if sum(source.shares for source in consumed) != shares:
                raise RuntimeError("rollback lost initiating spot-fill linkage")
            requests.append(request)
        return tuple(requests)

    def _emit_rollback_request(
        self,
        position: _Position,
        cursor: EventCursor,
        raw_id: str,
        reason: RollbackReason,
        shares: int,
        sources: tuple[HedgeSourceAllocation, ...],
    ) -> ExitRollbackRequest:
        if shares <= 0 or sum(source.shares for source in sources) != shares:
            raise RuntimeError("rollback source quantity does not match request")
        request_prefix = (
            "exit_hedge_timeout_rollback"
            if reason == "hedge_retry_timeout"
            else "exit_partial_rollback"
        )
        request = ExitRollbackRequest(
            request_id=(
                f"{request_prefix}/{self.Date}/{self.ValueCode}/"
                f"{position.fact.position_id}/{raw_id}/"
                f"{position.next_rollback:08d}"
            ),
            position_id=position.fact.position_id,
            value_code=position.fact.value_code,
            scenario_id=position.fact.scenario_id,
            capacity_id=position.fact.capacity_id,
            trigger_cursor=cursor,
            trigger_raw_order_fact_id=raw_id,
            reason=reason,
            rollback_spot_shares=shares,
            sources=sources,
        )
        if (
            request.request_id in self._pending_rollbacks
            or request.request_id in self._rollback_terminals_by_id
        ):
            raise RuntimeError("exit rollback request identity was already used")
        position.rollback_pending_shares += shares
        position.desired_absolute_target_tick = None
        position.desired_shares = 0
        position.next_rollback += 1
        self._rollback_requests.append(request)
        self._pending_rollbacks[request.request_id] = request
        return request

    @staticmethod
    def _consume_sources(
        sources: list[_UnhedgedSource], shares: int
    ) -> tuple[HedgeSourceAllocation, ...]:
        remaining = shares
        consumed: list[HedgeSourceAllocation] = []
        while remaining:
            if not sources:
                raise RuntimeError("unhedged fill sources were exhausted")
            source = sources[0]
            take = min(remaining, source.shares)
            consumed.append(HedgeSourceAllocation(source.allocation_id, take))
            source.shares -= take
            remaining -= take
            if source.shares == 0:
                sources.pop(0)
        return tuple(consumed)

    def _desired_signature(
        self, tick: int, allocation_id: str
    ) -> tuple[tuple[int, str, int], ...]:
        rows: list[tuple[int, str, int]] = []
        for position in self._positions.values():
            if (
                position.desired_absolute_target_tick != tick
                or position.desired_shares == 0
                or position.rollback_pending_shares
                or position.allocation_id not in (None, allocation_id)
            ):
                continue
            rows.append(
                (
                    position.fact.position_established_ns,
                    position.fact.position_id,
                    position.desired_shares,
                )
            )
        return tuple(sorted(rows))

    @staticmethod
    def _pending_signature(
        pending: _PendingNew,
    ) -> tuple[tuple[int, str, int], ...]:
        return tuple(
            (
                member.position_established_ns,
                member.position_id,
                member.assigned_shares,
            )
            for member in pending.members
        )

    @staticmethod
    def _working_signature(
        order: _Working,
    ) -> tuple[tuple[int, str, int], ...]:
        return tuple(
            (
                member.position_established_ns,
                member.position_id,
                member.leaves_shares,
            )
            for member in order.members
            if member.leaves_shares
        )

    def _validate_position_fact(
        self, fact: PositionEstablishedFact, cursor: EventCursor
    ) -> int:
        if not isinstance(fact, PositionEstablishedFact):
            raise TypeError("position_fact must be a PositionEstablishedFact")
        if fact.position_id in self._positions:
            raise ExitInventoryError("position was already added")
        for name in (
            "establishment_id",
            "position_id",
            "value_code",
            "scenario_id",
            "capacity_id",
            "capacity_transition_id",
        ):
            _identifier(getattr(fact, name), name)
        _positive_integer(fact.sequence, "position fact sequence")
        _valid_date(fact.establishment_date, "establishment_date")
        if fact.establishment_date > self.Date:
            raise ExitInventoryError(
                "position cannot be established after session Date"
            )
        if fact.value_code != self.ValueCode:
            raise ExitInventoryError("position value_code differs from controller")
        if fact.scenario_id != self.scenario_id:
            raise ExitInventoryError("position scenario_id differs from controller")
        if fact.execution_truth not in ("approximate", "exact"):
            raise ExitInventoryError("invalid position execution_truth")
        if not isinstance(fact.cursor, EventCursor):
            raise TypeError("position fact cursor must be an EventCursor")
        established_ns = _positive_integer(
            fact.position_established_ns, "position_established_ns"
        )
        if established_ns != fact.cursor.recv_time_ns:
            raise ExitInventoryError("position establishment cannot be backfilled")
        if fact.cursor >= cursor:
            raise ExitInventoryError("position must be established before allocation")
        spot = _positive_integer(fact.spot_shares, "spot_shares")
        contracts = _positive_integer(
            fact.short_future_contracts, "short_future_contracts"
        )
        equivalent = _positive_integer(
            fact.short_future_share_equivalent,
            "short_future_share_equivalent",
        )
        if spot != equivalent:
            raise ExitInventoryError(
                "exit FIFO accepts only paired long spot and short future"
            )
        if equivalent % contracts != 0:
            raise ExitInventoryError("future contract share equivalent is not integral")
        return equivalent // contracts

    def _require_position(self, position_id: str) -> _Position:
        position_id = _identifier(position_id, "position_id")
        try:
            return self._positions[position_id]
        except KeyError as error:
            raise ExitInventoryError("unknown position_id") from error

    def _validated_desired(
        self,
        position: _Position,
        *,
        absolute_target_tick: int | None,
        desired_shares: int,
    ) -> tuple[int | None, int]:
        shares = _nonnegative_integer(desired_shares, "desired_shares")
        if shares == 0:
            if absolute_target_tick is not None:
                raise ExitInventoryError("zero desired shares require a null target")
            return None, 0
        if absolute_target_tick is None:
            raise ExitInventoryError("positive desired shares require a target")
        target = _positive_integer(absolute_target_tick, "absolute_target_tick")
        if position.rollback_pending_shares:
            raise ExitInventoryError(
                "rollback-pending inventory cannot receive a new target"
            )
        if position.rollback_failed_shares:
            raise ExitInventoryError(
                "rollback-failed inventory cannot receive a new target"
            )
        if shares > position.available_spot_shares:
            raise ExitInventoryError(
                "desired shares exceed available paired spot inventory"
            )
        if shares % position.contract_size_shares != 0:
            raise ExitInventoryError(
                "explicit desired shares must contain integral hedge units"
            )
        return target, shares

    def _require_pending(self, candidate_id: str) -> _PendingNew:
        try:
            return self._pending_by_id[candidate_id]
        except KeyError as error:
            raise ExitInventoryError("candidate new request is not pending") from error

    def _require_working(self, raw_id: str) -> _Working:
        try:
            return self._working_by_id[raw_id]
        except KeyError as error:
            raise ExitInventoryError("raw aggregate order is not working") from error

    def _require_pending_hedge(self, request_id: str) -> ExitHedgeUnitRequest:
        if request_id in self._hedge_terminals_by_id:
            raise ExitInventoryError("hedge request callback was already completed")
        try:
            return self._pending_hedges[request_id]
        except KeyError as error:
            raise ExitInventoryError("unknown pending hedge request_id") from error

    def _require_pending_rollback(self, request_id: str) -> ExitRollbackRequest:
        if request_id in self._rollback_terminals_by_id:
            raise ExitInventoryError("rollback request callback was already completed")
        try:
            return self._pending_rollbacks[request_id]
        except KeyError as error:
            raise ExitInventoryError("unknown pending rollback request_id") from error

    def _validate_next_cursor(self, cursor: EventCursor) -> None:
        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if self._last_cursor is not None and cursor <= self._last_cursor:
            raise ExitInventoryError("controller cursors must be strictly increasing")

    def _finish(
        self,
        operation: Operation,
        cursor: EventCursor,
        *,
        position_fact: PositionEstablishedFact | None = None,
        position_id: str | None = None,
        absolute_target_tick: int | None = None,
        desired_shares: int | None = None,
        candidate_intent_id: str | None = None,
        raw_order_fact_id: str | None = None,
        actual_price_tick: int | None = None,
        fill_shares: int | None = None,
        commands: Sequence[ExitControllerCommand] = (),
        order_updates: Sequence[AggregateWorkingOrder] = (),
        terminals: Sequence[AggregateOrderTerminal] = (),
        fill_allocations: Sequence[ExitFillAllocation] = (),
        hedge_unit_requests: Sequence[ExitHedgeUnitRequest] = (),
        rollback_requests: Sequence[ExitRollbackRequest] = (),
        desired_updates: Sequence[ExitDesiredUpdate] = (),
        followup_request_id: str | None = None,
        hedge_terminals: Sequence[ExitHedgeUnitTerminal] = (),
        rollback_terminals: Sequence[ExitRollbackTerminal] = (),
    ) -> ExitInventoryFact:
        self._last_cursor = cursor
        self._assert_quantity_invariants()
        fact = ExitInventoryFact(
            sequence=len(self._facts) + 1,
            operation=operation,
            cursor=cursor,
            position_fact=position_fact,
            position_id=position_id,
            absolute_target_tick=absolute_target_tick,
            desired_shares=desired_shares,
            candidate_intent_id=candidate_intent_id,
            raw_order_fact_id=raw_order_fact_id,
            actual_price_tick=actual_price_tick,
            fill_shares=fill_shares,
            commands=tuple(commands),
            order_updates=tuple(order_updates),
            terminals=tuple(terminals),
            fill_allocations=tuple(fill_allocations),
            hedge_unit_requests=tuple(hedge_unit_requests),
            rollback_requests=tuple(rollback_requests),
            state_digest="",
            desired_updates=tuple(desired_updates),
            followup_request_id=followup_request_id,
            hedge_terminals=tuple(hedge_terminals),
            rollback_terminals=tuple(rollback_terminals),
        )
        self._history_digest = self._advance_history_digest(fact)
        fact = replace(fact, state_digest=self._state_digest())
        self._facts.append(fact)
        return fact

    def _identity_fields(self, tick: int) -> dict[str, object]:
        return {
            "Date": self.Date,
            "ValueCode": self.ValueCode,
            "QuoteCode": self.QuoteCode,
            "route": self.route,
            "stage": self.stage,
            "maker_side": self.maker_side,
            "absolute_price_tick": tick,
        }

    @staticmethod
    def _member_snapshot(member: _Member) -> AggregateOrderMember:
        return AggregateOrderMember(
            member.position_id,
            member.position_established_ns,
            member.assigned_shares,
            member.filled_shares,
        )

    def _pending_snapshot(self, pending: _PendingNew) -> PendingAggregateOrder:
        return PendingAggregateOrder(
            candidate_intent_id=pending.candidate_intent_id,
            absolute_price_tick=pending.absolute_price_tick,
            intent_cursor=pending.intent_cursor,
            scheduler_state=pending.scheduler_state,
            stale=pending.stale,
            members=tuple(self._member_snapshot(member) for member in pending.members),
        )

    def _working_snapshot(self, order: _Working) -> AggregateWorkingOrder:
        return AggregateWorkingOrder(
            candidate_intent_id=order.candidate_intent_id,
            raw_order_fact_id=order.raw_order_fact_id,
            absolute_price_tick=order.absolute_price_tick,
            actual_start_cursor=order.actual_start_cursor,
            cancel_state=order.cancel_state,
            members=tuple(self._member_snapshot(member) for member in order.members),
        )

    def _position_view(self, position: _Position) -> ExitPositionView:
        unhedged_fill_shares = sum(
            source.shares
            for source in self._unhedged_by_position[position.fact.position_id]
        )
        unresolved_reasons: list[PositionUnresolvedReason] = []
        if position.available_spot_shares:
            unresolved_reasons.append("available_inventory")
        if unhedged_fill_shares:
            unresolved_reasons.append("unhedged_fill")
        if position.hedge_pending_shares:
            unresolved_reasons.append("hedge_pending")
        if position.rollback_pending_shares:
            unresolved_reasons.append("rollback_pending")
        if position.rollback_failed_shares:
            unresolved_reasons.append("rollback_failed")
        unresolved_shares = position.fact.spot_shares - position.hedged_exit_shares
        resolution_state: PositionResolutionState = (
            "flat" if unresolved_shares == 0 else "unresolved"
        )
        return ExitPositionView(
            position_id=position.fact.position_id,
            value_code=position.fact.value_code,
            scenario_id=position.fact.scenario_id,
            capacity_id=position.fact.capacity_id,
            position_established_ns=position.fact.position_established_ns,
            contract_size_shares=position.contract_size_shares,
            available_spot_shares=position.available_spot_shares,
            desired_absolute_target_tick=position.desired_absolute_target_tick,
            desired_shares=position.desired_shares,
            allocation_id=position.allocation_id,
            allocation_kind=position.allocation_kind,
            unhedged_fill_shares=unhedged_fill_shares,
            hedge_pending_shares=position.hedge_pending_shares,
            hedged_exit_shares=position.hedged_exit_shares,
            rollback_pending_shares=position.rollback_pending_shares,
            rollback_failed_shares=position.rollback_failed_shares,
            unresolved_shares=unresolved_shares,
            resolution_state=resolution_state,
            unresolved_reasons=tuple(unresolved_reasons),
            capacity_releasable=resolution_state == "flat",
        )

    def _assert_quantity_invariants(self) -> None:
        if len(self._hedge_requests) != (
            len(self._pending_hedges) + len(self._hedge_terminals_by_id)
        ):
            raise RuntimeError("hedge request terminal/pending partition diverged")
        if any(
            request_id in self._hedge_terminals_by_id
            for request_id in self._pending_hedges
        ):
            raise RuntimeError("hedge request is both pending and terminal")

        if len(self._rollback_requests) != (
            len(self._pending_rollbacks) + len(self._rollback_terminals_by_id)
        ):
            raise RuntimeError("rollback request terminal/pending partition diverged")
        if any(
            request_id in self._rollback_terminals_by_id
            for request_id in self._pending_rollbacks
        ):
            raise RuntimeError("rollback request is both pending and terminal")

        for position in self._positions.values():
            position_id = position.fact.position_id
            unhedged = sum(
                source.shares for source in self._unhedged_by_position[position_id]
            )
            pending_hedge = sum(
                request.future_share_equivalent
                for request in self._pending_hedges.values()
                if request.position_id == position_id
            )
            pending_rollback = sum(
                request.rollback_spot_shares
                for request in self._pending_rollbacks.values()
                if request.position_id == position_id
            )
            sent_hedge = self._sent_hedge_shares_by_position.get(position_id, 0)
            failed_rollback = self._failed_rollback_shares_by_position.get(
                position_id, 0
            )
            if pending_hedge != position.hedge_pending_shares:
                raise RuntimeError("position hedge-pending quantity diverged")
            if pending_rollback != position.rollback_pending_shares:
                raise RuntimeError("position rollback-pending quantity diverged")
            if sent_hedge != position.hedged_exit_shares:
                raise RuntimeError("position hedged-exit quantity diverged")
            if failed_rollback != position.rollback_failed_shares:
                raise RuntimeError("position rollback-failed quantity diverged")
            buckets = (
                position.available_spot_shares,
                unhedged,
                position.hedge_pending_shares,
                position.hedged_exit_shares,
                position.rollback_pending_shares,
                position.rollback_failed_shares,
            )
            if any(quantity < 0 for quantity in buckets):
                raise RuntimeError("position exit quantity bucket became negative")
            if sum(buckets) != position.fact.spot_shares:
                raise RuntimeError("position exit quantity conservation failed")
            if position.desired_shares > position.available_spot_shares:
                raise RuntimeError("desired position quantity exceeds available")
            if position.rollback_failed_shares and (
                position.desired_shares
                or position.desired_absolute_target_tick is not None
            ):
                raise RuntimeError("rollback-failed position was made quoteable")

    def _state_digest(self) -> str:
        """Commit the incremental history chain and canonical live state.

        Version 2 deliberately excludes append-only audit collections from the
        live-state payload.  Every newly appended row is already committed once
        by ``_history_digest``; re-encoding all prior rows on every callback made
        a long session quadratic.  Replay still reconstructs and compares every
        source and derived fact before accepting this digest.
        """

        payload = {
            "digest_version": 2,
            "history_digest": self._history_digest,
            "controller_scope": {
                "Date": self.Date,
                "ValueCode": self.ValueCode,
                "QuoteCode": self.QuoteCode,
                "scenario_id": self.scenario_id,
                "route": self.route,
                "stage": self.stage,
                "maker_side": self.maker_side,
            },
            "last_cursor": asdict(self._last_cursor)
            if self._last_cursor is not None
            else None,
            "session_expired": self._session_expired,
            "positions": [asdict(item) for item in self.positions],
            "pending_orders": [asdict(item) for item in self.pending_orders],
            "working_orders": [asdict(item) for item in self.working_orders],
            "pending_hedges": [
                asdict(item) for item in self.pending_hedge_unit_requests
            ],
            "pending_rollbacks": [
                asdict(item) for item in self.pending_rollback_requests
            ],
            "unhedged_sources": {
                position_id: [asdict(source) for source in sources]
                for position_id, sources in sorted(self._unhedged_by_position.items())
            },
            "position_counters": {
                position.fact.position_id: {
                    "next_hedge_unit": position.next_hedge_unit,
                    "next_rollback": position.next_rollback,
                }
                for position in sorted(
                    self._positions.values(), key=lambda item: item.fact.fifo_key
                )
            },
            "next_fill_allocation": self._next_fill_allocation,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _advance_history_digest(self, fact: ExitInventoryFact) -> str:
        payload = asdict(fact)
        payload.pop("state_digest")
        encoded = json.dumps(
            {
                "previous_history_digest": self._history_digest,
                "fact": payload,
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_exit_inventory_facts(
    facts: Sequence[ExitInventoryFact],
    *,
    Date: str,
    ValueCode: str,
    QuoteCode: str,
    scenario_id: str,
    route: str = _ROUTE,
    stage: str = _STAGE,
    maker_side: Literal["ask"] = _MAKER_SIDE,
) -> S1ExitInventoryController:
    """Rebuild a controller and reject any source or derived-fact tampering."""

    controller = S1ExitInventoryController(
        Date=Date,
        ValueCode=ValueCode,
        QuoteCode=QuoteCode,
        scenario_id=scenario_id,
        route=route,
        stage=stage,
        maker_side=maker_side,
    )
    for expected_sequence, expected in enumerate(facts, start=1):
        if not isinstance(expected, ExitInventoryFact):
            raise ExitInventoryReplayError("unexpected exit inventory fact type")
        if expected.sequence != expected_sequence:
            raise ExitInventoryReplayError("exit inventory fact sequence is not dense")
        try:
            if expected.operation == "add_position":
                if expected.position_fact is None:
                    raise ExitInventoryReplayError("add fact lacks position source")
                actual = controller.add_paired_position(
                    expected.position_fact,
                    absolute_target_tick=_required_source(
                        expected.absolute_target_tick, "absolute_target_tick"
                    ),
                    desired_shares=_required_source(
                        expected.desired_shares, "desired_shares"
                    ),
                    cursor=expected.cursor,
                )
            elif expected.operation == "set_desired":
                actual = controller.set_position_desired(
                    _required_source(expected.position_id, "position_id"),
                    absolute_target_tick=expected.absolute_target_tick,
                    desired_shares=_required_source(
                        expected.desired_shares, "desired_shares"
                    ),
                    cursor=expected.cursor,
                )
            elif expected.operation == "set_desired_batch":
                if not expected.desired_updates:
                    raise ExitInventoryReplayError(
                        "batch desired fact lacks update sources"
                    )
                actual = controller.set_positions_desired(
                    expected.desired_updates,
                    cursor=expected.cursor,
                )
            elif expected.operation == "new_assigned":
                actual = controller.on_new_assigned(
                    _required_source(
                        expected.candidate_intent_id, "candidate_intent_id"
                    ),
                    expected.cursor,
                )
            elif expected.operation == "new_sent":
                actual = controller.on_new_sent(
                    _required_source(
                        expected.candidate_intent_id, "candidate_intent_id"
                    ),
                    expected.cursor,
                    _required_source(expected.actual_price_tick, "actual_price_tick"),
                )
            elif expected.operation == "cancel_assigned":
                actual = controller.on_cancel_assigned(
                    _required_source(expected.raw_order_fact_id, "raw_order_fact_id"),
                    expected.cursor,
                )
            elif expected.operation == "cancel_sent":
                actual = controller.on_cancel_sent(
                    _required_source(expected.raw_order_fact_id, "raw_order_fact_id"),
                    expected.cursor,
                )
            elif expected.operation == "fill":
                actual = controller.on_fill(
                    _required_source(expected.raw_order_fact_id, "raw_order_fact_id"),
                    expected.cursor,
                    _required_source(expected.fill_shares, "fill_shares"),
                )
            elif expected.operation == "hedge_sent":
                actual = controller.on_hedge_sent(
                    _required_source(
                        expected.followup_request_id, "followup_request_id"
                    ),
                    expected.cursor,
                )
            elif expected.operation == "hedge_timeout":
                actual = controller.on_hedge_timeout(
                    _required_source(
                        expected.followup_request_id, "followup_request_id"
                    ),
                    expected.cursor,
                )
            elif expected.operation == "rollback_sent":
                actual = controller.on_rollback_sent(
                    _required_source(
                        expected.followup_request_id, "followup_request_id"
                    ),
                    expected.cursor,
                )
            elif expected.operation == "rollback_failed":
                actual = controller.on_rollback_failed(
                    _required_source(
                        expected.followup_request_id, "followup_request_id"
                    ),
                    expected.cursor,
                )
            elif expected.operation == "session_expiry":
                actual = controller.session_expiry(expected.cursor)
            else:
                raise ExitInventoryReplayError(
                    f"unknown exit inventory operation: {expected.operation}"
                )
        except ExitInventoryReplayError:
            raise
        except (ExitInventoryError, TypeError, RuntimeError) as error:
            raise ExitInventoryReplayError(
                f"exit inventory replay failed at sequence {expected_sequence}"
            ) from error
        if actual != expected:
            raise ExitInventoryReplayError(
                f"exit inventory fact mismatch at sequence {expected_sequence}"
            )
    return controller


def _required_source(value: object | None, name: str):  # type: ignore[no-untyped-def]
    if value is None:
        raise ExitInventoryReplayError(f"fact lacks source field: {name}")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExitInventoryError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExitInventoryError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExitInventoryError(f"{name} must be a non-negative integer")
    return value


def _valid_date(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ExitInventoryError(f"{name} must be valid YYYYMMDD")
    try:
        parsed = date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as error:
        raise ExitInventoryError(f"{name} must be valid YYYYMMDD") from error
    if parsed.strftime("%Y%m%d") != value:
        raise ExitInventoryError(f"{name} must be valid YYYYMMDD")
    return value
