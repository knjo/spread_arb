"""Pure pre-send and working-order controller for S1 entry maker orders.

The controller deliberately does not own venue tokens, capacity, books, or
fill discovery.  It emits scheduler commands and accepts actual-send and
terminal callbacks from the chronological outer loop.  A physical raw order
therefore cannot exist before the new request is actually sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from .layered import EventCursor
from .order_identity import (
    candidate_intent_id,
    policy_alias_id,
    raw_order_fact_id,
)

CommandKind = Literal[
    "enqueue_new",
    "withdraw_pending_new",
    "enqueue_cancel",
    "withdraw_pending_cancel",
]
TerminalReason = Literal["filled", "actual_cancelled", "session_expired"]
UnsentReason = Literal["coalesced", "expired", "cap_blocked"]

_UNSENT_REASONS: Final = frozenset({"coalesced", "expired", "cap_blocked"})
_S1_ROUTE: Final = "spot_bid_future_taker"
_S1_STAGE: Final = "entry"
_S1_MAKER_SIDE: Final = "bid"


@dataclass(frozen=True, slots=True)
class EntryControllerCommand:
    kind: CommandKind
    request_id: str
    cursor: EventCursor
    reason: str
    absolute_price_tick: int
    candidate_intent_id: str | None
    raw_order_fact_id: str | None


@dataclass(frozen=True, slots=True)
class CandidateIntentAudit:
    candidate_intent_id: str
    absolute_price_tick: int
    intent_cursor: EventCursor
    terminal_cursor: EventCursor
    status: UnsentReason
    raw_order_fact_id: None = None


@dataclass(frozen=True, slots=True)
class WorkingOrder:
    candidate_intent_id: str
    raw_order_fact_id: str
    policy_alias_id: str
    intent_absolute_price_tick: int
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    cancel_state: Literal["none", "pending", "assigned"]


@dataclass(frozen=True, slots=True)
class OrderTerminal:
    raw_order_fact_id: str
    policy_alias_id: str
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    terminal_cursor: EventCursor
    terminal_reason: TerminalReason
    frozen_cancel_will_be_noop: bool


@dataclass(slots=True)
class _PendingNew:
    candidate_intent_id: str
    absolute_price_tick: int
    intent_cursor: EventCursor


@dataclass(slots=True)
class _Working:
    candidate_intent_id: str
    raw_order_fact_id: str
    policy_alias_id: str
    intent_absolute_price_tick: int
    absolute_price_tick: int
    actual_start_cursor: EventCursor
    cancel_state: Literal["none", "pending", "assigned"] = "none"


class S1EntryController:
    """One-policy, one-product-day desired/working lifecycle controller."""

    def __init__(
        self,
        *,
        Date: str,
        ValueCode: str,
        QuoteCode: str,
        route: str,
        stage: str,
        maker_side: Literal["bid"],
        policy_id: str,
    ) -> None:
        for name, value in (
            ("Date", Date),
            ("ValueCode", ValueCode),
            ("QuoteCode", QuoteCode),
            ("route", route),
            ("stage", stage),
            ("policy_id", policy_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if route != _S1_ROUTE or stage != _S1_STAGE or maker_side != _S1_MAKER_SIDE:
            raise ValueError(
                "S1 entry controller requires spot_bid_future_taker/entry/bid semantics"
            )
        self.Date = Date
        self.ValueCode = ValueCode
        self.QuoteCode = QuoteCode
        self.route = route
        self.stage = stage
        self.maker_side = maker_side
        self.policy_id = policy_id
        self._last_cursor: EventCursor | None = None
        self._current_target_tick: int | None = None
        self._gate_open = False
        self._last_admission_open = False
        self._has_observation = False
        self._cutoff = False
        self._session_expired = False
        self._pending_new: _PendingNew | None = None
        self._working_by_tick: dict[int, _Working] = {}
        self._working_by_raw_id: dict[str, _Working] = {}
        self._terminal_by_raw_id: dict[str, OrderTerminal] = {}
        self._consumed_cancel_callbacks: set[str] = set()
        self._intent_audit: list[CandidateIntentAudit] = []

    @property
    def pending_candidate_intent_id(self) -> str | None:
        return (
            None if self._pending_new is None else self._pending_new.candidate_intent_id
        )

    @property
    def working_orders(self) -> tuple[WorkingOrder, ...]:
        return tuple(
            self._snapshot(order)
            for order in sorted(
                self._working_by_tick.values(),
                key=lambda value: (
                    value.actual_start_cursor,
                    value.raw_order_fact_id,
                ),
            )
        )

    @property
    def intent_audit(self) -> tuple[CandidateIntentAudit, ...]:
        return tuple(self._intent_audit)

    @property
    def terminals(self) -> tuple[OrderTerminal, ...]:
        return tuple(self._terminal_by_raw_id.values())

    @property
    def session_expired(self) -> bool:
        return self._session_expired

    def observe(
        self,
        cursor: EventCursor,
        absolute_target_tick: int | None,
        *,
        base_gate_open: bool,
        admission_open: bool,
        gate_reason: str = "base_gate_closed",
        candidate_intent_cursor: EventCursor | None = None,
    ) -> tuple[EntryControllerCommand, ...]:
        """Reconcile the latest desired absolute price into request commands.

        ``cursor`` remains the causal controller/scheduler effect cursor.  The
        optional ``candidate_intent_cursor`` is a product-local logical cursor
        used only when this observation creates a candidate identity.  Keeping
        those clocks separate prevents an unrelated product's same-phase
        effect row from changing this product's pre-send physical intent ID.
        """

        self._validate_next_cursor(cursor)
        identity_cursor = self._candidate_intent_cursor(
            cursor,
            candidate_intent_cursor,
        )
        if self._cutoff:
            raise RuntimeError("cannot observe after cutoff")
        if not isinstance(base_gate_open, bool) or not isinstance(admission_open, bool):
            raise TypeError("gate flags must be boolean")
        if admission_open and not base_gate_open:
            raise ValueError("admission cannot be open while the base gate is closed")
        if not isinstance(gate_reason, str) or not gate_reason:
            raise ValueError("gate_reason must be a non-empty string")
        target = self._target(absolute_target_tick, base_gate_open)
        previous_target = self._current_target_tick
        was_gate_open = self._gate_open
        previous_admission = self._last_admission_open
        commands: list[EntryControllerCommand] = []

        if not base_gate_open:
            commands.extend(self._withdraw_pending_new(cursor, "coalesced"))
            commands.extend(self._cancel_all(cursor, gate_reason))
            if target is not None:
                self._current_target_tick = target
            self._gate_open = False
            self._last_admission_open = False
            self._has_observation = True
            self._commit_cursor(cursor)
            return tuple(commands)

        assert target is not None
        if self._pending_new is not None and (
            self._pending_new.absolute_price_tick != target or not admission_open
        ):
            commands.extend(self._withdraw_pending_new(cursor, "coalesced"))

        commands.extend(self._cancel_above(target, cursor, "target_retreat"))

        live = target in self._working_by_tick
        pending = (
            self._pending_new is not None
            and self._pending_new.absolute_price_tick == target
        )
        reason: str | None = None
        if admission_open and not live and not pending:
            if not self._has_observation:
                reason = "initial_eligible"
            elif not was_gate_open:
                reason = "gate_reopen"
            elif previous_target is not None and target > previous_target:
                reason = "forward_new_price"
            elif previous_target == target and not previous_admission:
                reason = "became_admission_eligible"
        if reason is not None:
            commands.append(
                self._enqueue_new(
                    cursor,
                    target,
                    reason,
                    candidate_intent_cursor=identity_cursor,
                )
            )

        self._current_target_tick = target
        self._gate_open = True
        self._last_admission_open = admission_open
        self._has_observation = True
        self._commit_cursor(cursor)
        return tuple(commands)

    def on_new_sent(
        self,
        candidate_id: str,
        actual_send_cursor: EventCursor,
        actual_price_tick: int,
    ) -> WorkingOrder:
        """Create the physical lifecycle only after an actual new send."""

        self._validate_next_cursor(actual_send_cursor)
        self._validate_candidate_id(candidate_id)
        tick = self._positive_tick(actual_price_tick, "actual_price_tick")
        ineligible_reason = self._new_send_ineligible_reason(candidate_id, tick)
        if ineligible_reason is not None:
            raise ValueError(
                f"actual new send failed atomic pre-send validation: "
                f"{ineligible_reason}"
            )
        pending = self._pending_new
        assert pending is not None
        raw_id = raw_order_fact_id(
            **self._identity_fields(tick),
            actual_start_cursor=actual_send_cursor,
        )
        order = _Working(
            candidate_intent_id=candidate_id,
            raw_order_fact_id=raw_id,
            policy_alias_id=policy_alias_id(
                raw_order_fact_id=raw_id,
                policy_id=self.policy_id,
            ),
            intent_absolute_price_tick=pending.absolute_price_tick,
            absolute_price_tick=tick,
            actual_start_cursor=actual_send_cursor,
        )
        self._pending_new = None
        self._working_by_tick[tick] = order
        self._working_by_raw_id[raw_id] = order
        self._commit_cursor(actual_send_cursor)
        return self._snapshot(order)

    def can_send_pending_at_tick(
        self,
        candidate_id: str,
        actual_price_tick: int,
    ) -> bool:
        """Pre-send check after actual-cursor target recomputation."""

        self._validate_candidate_id(candidate_id)
        tick = self._positive_tick(actual_price_tick, "actual_price_tick")
        return self._new_send_ineligible_reason(candidate_id, tick) is None

    def on_cancel_assigned(self, raw_id: str, cursor: EventCursor) -> None:
        """Freeze a cancel assignment before same-cursor fill allocation."""

        self._validate_next_cursor(cursor)
        order = self._require_working(raw_id)
        if order.cancel_state != "pending":
            raise ValueError("only a pending cancel can be assigned")
        order.cancel_state = "assigned"
        self._commit_cursor(cursor)

    def on_cancel_sent(
        self,
        raw_id: str,
        actual_send_cursor: EventCursor,
    ) -> tuple[OrderTerminal, tuple[EntryControllerCommand, ...]]:
        """Apply an assigned cancel; a prior same-cursor fill makes it a noop."""

        self._validate_next_cursor(actual_send_cursor)
        if raw_id in self._consumed_cancel_callbacks:
            raise ValueError("cancel send callback was already consumed")
        terminal = self._terminal_by_raw_id.get(raw_id)
        if terminal is not None:
            if not terminal.frozen_cancel_will_be_noop:
                raise ValueError("terminal order has no frozen cancel assignment")
            self._consumed_cancel_callbacks.add(raw_id)
            self._commit_cursor(actual_send_cursor)
            return terminal, ()
        order = self._require_working(raw_id)
        if order.cancel_state != "assigned":
            raise ValueError("cancel must be assigned before it is sent")
        cancelled_tick = order.absolute_price_tick
        terminal = self._terminalize(order, actual_send_cursor, "actual_cancelled")
        commands: tuple[EntryControllerCommand, ...] = ()
        if self._should_replace_actual_cancel(cancelled_tick):
            commands = (
                self._enqueue_new(
                    actual_send_cursor,
                    cancelled_tick,
                    "desired_after_actual_cancel",
                ),
            )
        self._consumed_cancel_callbacks.add(raw_id)
        self._commit_cursor(actual_send_cursor)
        return terminal, commands

    def on_fill(
        self,
        raw_id: str,
        fill_cursor: EventCursor,
    ) -> tuple[OrderTerminal, tuple[EntryControllerCommand, ...]]:
        """Terminalize a working order and withdraw only an unfrozen cancel."""

        self._validate_next_cursor(fill_cursor)
        order = self._require_working(raw_id)
        if fill_cursor <= order.actual_start_cursor:
            raise ValueError("fill must follow the actual new-send cursor")
        commands: tuple[EntryControllerCommand, ...] = ()
        if order.cancel_state == "pending":
            commands = (
                self._command(
                    "withdraw_pending_cancel",
                    f"{raw_id}/cancel",
                    fill_cursor,
                    "fill_terminal",
                    order.absolute_price_tick,
                    order.candidate_intent_id,
                    raw_id,
                ),
            )
            order.cancel_state = "none"
        terminal = self._terminalize(order, fill_cursor, "filled")
        self._commit_cursor(fill_cursor)
        return terminal, commands

    def cutoff(
        self,
        cursor: EventCursor,
        *,
        unsent_reason: UnsentReason = "expired",
    ) -> tuple[EntryControllerCommand, ...]:
        """Stop entry intents and request cancellation of every working order."""

        self._validate_next_cursor(cursor)
        if self._cutoff:
            raise RuntimeError("cutoff already applied")
        if unsent_reason not in _UNSENT_REASONS:
            raise ValueError("invalid unsent cutoff reason")
        commands = [*self._withdraw_pending_new(cursor, unsent_reason)]
        commands.extend(self._cancel_all(cursor, "entry_cutoff"))
        self._cutoff = True
        self._gate_open = False
        self._last_admission_open = False
        self._commit_cursor(cursor)
        return tuple(commands)

    def session_expiry(
        self,
        cursor: EventCursor,
    ) -> tuple[tuple[OrderTerminal, ...], tuple[EntryControllerCommand, ...]]:
        """Expire live orders and remove all still-unassigned requests."""

        self._validate_next_cursor(cursor)
        if self._session_expired:
            raise RuntimeError("session expiry already applied")
        commands = list(self._withdraw_pending_new(cursor, "expired"))
        terminals: list[OrderTerminal] = []
        for order in tuple(self._working_by_tick.values()):
            if order.cancel_state == "pending":
                commands.append(
                    self._command(
                        "withdraw_pending_cancel",
                        f"{order.raw_order_fact_id}/cancel",
                        cursor,
                        "session_expiry",
                        order.absolute_price_tick,
                        order.candidate_intent_id,
                        order.raw_order_fact_id,
                    )
                )
                order.cancel_state = "none"
            terminals.append(self._terminalize(order, cursor, "session_expired"))
        self._cutoff = True
        self._session_expired = True
        self._gate_open = False
        self._last_admission_open = False
        self._commit_cursor(cursor)
        return tuple(terminals), tuple(commands)

    def _enqueue_new(
        self,
        cursor: EventCursor,
        tick: int,
        reason: str,
        *,
        candidate_intent_cursor: EventCursor | None = None,
    ) -> EntryControllerCommand:
        if self._pending_new is not None:
            raise RuntimeError("only one unsent desired new may exist")
        identity_cursor = self._candidate_intent_cursor(
            cursor,
            candidate_intent_cursor,
        )
        intent_id = candidate_intent_id(
            **self._identity_fields(tick), intent_cursor=identity_cursor
        )
        self._pending_new = _PendingNew(intent_id, tick, identity_cursor)
        return self._command(
            "enqueue_new",
            f"{intent_id}/new",
            cursor,
            reason,
            tick,
            intent_id,
            None,
        )

    @staticmethod
    def _candidate_intent_cursor(
        effect_cursor: EventCursor,
        candidate_intent_cursor: EventCursor | None,
    ) -> EventCursor:
        if candidate_intent_cursor is None:
            return effect_cursor
        if not isinstance(candidate_intent_cursor, EventCursor):
            raise TypeError("candidate_intent_cursor must be an EventCursor or None")
        if (
            candidate_intent_cursor.recv_time_ns != effect_cursor.recv_time_ns
            or candidate_intent_cursor.event_sequence != effect_cursor.event_sequence
        ):
            raise ValueError(
                "candidate_intent_cursor must share the effect timestamp and phase"
            )
        if candidate_intent_cursor > effect_cursor:
            raise ValueError("candidate_intent_cursor cannot follow the effect cursor")
        return candidate_intent_cursor

    def _withdraw_pending_new(
        self, cursor: EventCursor, status: UnsentReason
    ) -> tuple[EntryControllerCommand, ...]:
        pending = self._pending_new
        if pending is None:
            return ()
        self._pending_new = None
        self._intent_audit.append(
            CandidateIntentAudit(
                pending.candidate_intent_id,
                pending.absolute_price_tick,
                pending.intent_cursor,
                cursor,
                status,
            )
        )
        return (
            self._command(
                "withdraw_pending_new",
                f"{pending.candidate_intent_id}/new",
                cursor,
                status,
                pending.absolute_price_tick,
                pending.candidate_intent_id,
                None,
            ),
        )

    def _cancel_above(
        self, tick: int, cursor: EventCursor, reason: str
    ) -> list[EntryControllerCommand]:
        return self._request_cancels(
            tuple(
                order for price, order in self._working_by_tick.items() if price > tick
            ),
            cursor,
            reason,
        )

    def _cancel_all(
        self, cursor: EventCursor, reason: str
    ) -> list[EntryControllerCommand]:
        return self._request_cancels(
            tuple(self._working_by_tick.values()), cursor, reason
        )

    def _request_cancels(
        self,
        orders: tuple[_Working, ...],
        cursor: EventCursor,
        reason: str,
    ) -> list[EntryControllerCommand]:
        commands: list[EntryControllerCommand] = []
        for order in sorted(orders, key=lambda value: value.raw_order_fact_id):
            if order.cancel_state != "none":
                continue
            order.cancel_state = "pending"
            commands.append(
                self._command(
                    "enqueue_cancel",
                    f"{order.raw_order_fact_id}/cancel",
                    cursor,
                    reason,
                    order.absolute_price_tick,
                    order.candidate_intent_id,
                    order.raw_order_fact_id,
                )
            )
        return commands

    def _terminalize(
        self,
        order: _Working,
        cursor: EventCursor,
        reason: TerminalReason,
    ) -> OrderTerminal:
        terminal = OrderTerminal(
            order.raw_order_fact_id,
            order.policy_alias_id,
            order.absolute_price_tick,
            order.actual_start_cursor,
            cursor,
            reason,
            order.cancel_state == "assigned" and reason != "actual_cancelled",
        )
        del self._working_by_tick[order.absolute_price_tick]
        del self._working_by_raw_id[order.raw_order_fact_id]
        self._terminal_by_raw_id[order.raw_order_fact_id] = terminal
        return terminal

    def _require_working(self, raw_id: str) -> _Working:
        try:
            return self._working_by_raw_id[raw_id]
        except KeyError as error:
            raise ValueError("raw order is not working") from error

    def _new_send_ineligible_reason(
        self,
        candidate_id: str,
        actual_price_tick: int,
    ) -> str | None:
        if self._session_expired:
            return "session_expired"
        if self._cutoff:
            return "entry_cutoff"
        if not self._has_observation:
            return "missing_desired_observation"
        if not self._gate_open:
            return "base_gate_closed"
        if not self._last_admission_open:
            return "admission_closed"
        pending = self._pending_new
        if pending is None:
            return "missing_pending_intent"
        if pending.candidate_intent_id != candidate_id:
            return "pending_identity_mismatch"
        if self._current_target_tick is None:
            return "missing_current_target"
        if pending.absolute_price_tick != self._current_target_tick:
            return "pending_target_is_stale"
        if actual_price_tick != self._current_target_tick:
            return "actual_tick_differs_from_current_desired_target"
        if actual_price_tick in self._working_by_tick:
            return "absolute_price_already_working"
        return None

    def _should_replace_actual_cancel(self, cancelled_tick: int) -> bool:
        return (
            not self._cutoff
            and not self._session_expired
            and self._has_observation
            and self._gate_open
            and self._last_admission_open
            and self._current_target_tick == cancelled_tick
            and self._pending_new is None
            and cancelled_tick not in self._working_by_tick
        )

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

    def _validate_next_cursor(self, cursor: EventCursor) -> None:
        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if self._last_cursor is not None and cursor <= self._last_cursor:
            raise ValueError("controller cursors must be strictly increasing")

    def _commit_cursor(self, cursor: EventCursor) -> None:
        self._last_cursor = cursor

    @staticmethod
    def _validate_candidate_id(value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("candidate_id must be a non-empty string")

    @staticmethod
    def _target(value: int | None, gate_open: bool) -> int | None:
        if value is None:
            if gate_open:
                raise ValueError("open base gate requires an absolute target")
            return None
        return S1EntryController._positive_tick(value, "absolute target tick")

    @staticmethod
    def _positive_tick(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _snapshot(order: _Working) -> WorkingOrder:
        return WorkingOrder(
            order.candidate_intent_id,
            order.raw_order_fact_id,
            order.policy_alias_id,
            order.intent_absolute_price_tick,
            order.absolute_price_tick,
            order.actual_start_cursor,
            order.cancel_state,
        )

    @staticmethod
    def _command(
        kind: CommandKind,
        request_id: str,
        cursor: EventCursor,
        reason: str,
        tick: int,
        candidate_id: str | None,
        raw_id: str | None,
    ) -> EntryControllerCommand:
        return EntryControllerCommand(
            kind,
            request_id,
            cursor,
            reason,
            tick,
            candidate_id,
            raw_id,
        )


__all__ = [
    "CandidateIntentAudit",
    "EntryControllerCommand",
    "OrderTerminal",
    "S1EntryController",
    "WorkingOrder",
]
