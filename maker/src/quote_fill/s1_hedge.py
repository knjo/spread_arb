"""Pure B6 hedge retry domain core.

This module consumes normalized raw book events but performs no file I/O and
does not assign venue request quota.  A successful observation is therefore a
``send_eligible`` candidate, never an actual request send or execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .layered import EventCursor

HEDGE_DELAY_NS = 50_000_000
HEDGE_RETRY_NS = 5_000_000_000

Side = Literal["buy", "sell"]
QuantityUnit = Literal["spot_shares", "future_contracts"]
AttemptStatus = Literal[
    "waiting",
    "send_eligible",
    "hedge_sent",
    "hedge_retry_timeout",
]
RollbackAttemptStatus = Literal[
    "waiting",
    "send_eligible",
    "rollback_sent",
    "rollback_retry_timeout",
]


@dataclass(frozen=True, order=True)
class RawBookCursor:
    """Causal cursor plus exchange packet provenance."""

    cursor: EventCursor
    packet_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if (
            isinstance(self.packet_sequence, bool)
            or not isinstance(self.packet_sequence, int)
            or self.packet_sequence < 0
        ):
            raise ValueError("packet_sequence must be a non-negative integer")


@dataclass(frozen=True)
class RawBookLevel:
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if not isinstance(self.price, (int, float)) or isinstance(self.price, bool):
            raise TypeError("price must be a number")
        if not math.isfinite(float(self.price)) or self.price <= 0:
            raise ValueError("price must be a finite positive number")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("quantity must be a positive integer")


@dataclass(frozen=True)
class RawBookEvent:
    """One normalized raw-state event.

    TrialMatch always closes the gate and clears persisted depth.  Only an
    event explicitly marked ``formal_book`` may replace depth and reopen it.
    Best and L1 are supplied independently because their feed clocks differ.
    """

    book_cursor: RawBookCursor
    trial_match: bool = False
    formal_book: bool = False
    reference_price: float | None = None
    l1_bids: tuple[RawBookLevel, ...] = ()
    l1_asks: tuple[RawBookLevel, ...] = ()
    best_bid: RawBookLevel | None = None
    best_ask: RawBookLevel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.book_cursor, RawBookCursor):
            raise TypeError("book_cursor must be a RawBookCursor")
        if not isinstance(self.trial_match, bool) or not isinstance(
            self.formal_book, bool
        ):
            raise TypeError("trial_match and formal_book must be boolean")
        if self.trial_match and self.formal_book:
            raise ValueError("TrialMatch cannot simultaneously be a formal book")
        if len(self.l1_bids) > 5 or len(self.l1_asks) > 5:
            raise ValueError("raw L1 depth is limited to five levels per side")
        for side in (self.l1_bids, self.l1_asks):
            if any(not isinstance(level, RawBookLevel) for level in side):
                raise TypeError("L1 sides must contain RawBookLevel values")
        if self.best_bid is not None and not isinstance(self.best_bid, RawBookLevel):
            raise TypeError("best_bid must be a RawBookLevel or None")
        if self.best_ask is not None and not isinstance(self.best_ask, RawBookLevel):
            raise TypeError("best_ask must be a RawBookLevel or None")


@dataclass(frozen=True)
class CausalBookState:
    book_cursor: RawBookCursor
    gate_open: bool
    gate_reason: str | None
    reference_price: float | None
    bids: tuple[RawBookLevel, ...]
    asks: tuple[RawBookLevel, ...]


class RawBookStateMachine:
    """Persist the latest complete formal book with TrialMatch invalidation."""

    def __init__(self) -> None:
        self._state: CausalBookState | None = None
        self._last_cursor: RawBookCursor | None = None

    @property
    def state(self) -> CausalBookState | None:
        return self._state

    def ingest(self, event: RawBookEvent) -> CausalBookState | None:
        cursor = event.book_cursor
        if self._last_cursor is not None and cursor <= self._last_cursor:
            raise ValueError("raw book events must be strictly cursor ordered")
        self._last_cursor = cursor
        if event.trial_match:
            self._state = CausalBookState(
                event.book_cursor, False, "trial_match", None, (), ()
            )
        elif event.formal_book:
            self._state = CausalBookState(
                event.book_cursor,
                True,
                None,
                event.reference_price,
                _merge_best_l1(event.l1_bids, event.best_bid, descending=True),
                _merge_best_l1(event.l1_asks, event.best_ask, descending=False),
            )
        return self._state


@dataclass(frozen=True)
class HedgeIntent:
    hedge_intent_id: str
    trigger_cursor: EventCursor
    side: Side
    hedge_quantity: int
    hedge_quantity_unit: QuantityUnit
    session_end_time_ns: int
    initiating_first_leg_venue: str
    initiating_execution_id: str
    initiating_first_leg_side: Side
    initiating_first_leg_quantity: int
    initiating_first_leg_quantity_unit: QuantityUnit
    rollback_session_end_time_ns: int | None = None
    delay_ns: int = HEDGE_DELAY_NS
    retry_ns: int = HEDGE_RETRY_NS

    def __post_init__(self) -> None:
        for name in (
            "hedge_intent_id",
            "initiating_first_leg_venue",
            "initiating_execution_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.trigger_cursor, EventCursor):
            raise TypeError("trigger_cursor must be an EventCursor")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.initiating_first_leg_side not in ("buy", "sell"):
            raise ValueError("initiating_first_leg_side must be 'buy' or 'sell'")
        for name in (
            "hedge_quantity",
            "initiating_first_leg_quantity",
            "session_end_time_ns",
            "delay_ns",
            "retry_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _validate_quantity_unit(self.hedge_quantity_unit, "hedge_quantity_unit")
        _validate_quantity_unit(
            self.initiating_first_leg_quantity_unit,
            "initiating_first_leg_quantity_unit",
        )
        if self.rollback_session_end_time_ns is not None and (
            isinstance(self.rollback_session_end_time_ns, bool)
            or not isinstance(self.rollback_session_end_time_ns, int)
            or self.rollback_session_end_time_ns <= 0
        ):
            raise ValueError("rollback_session_end_time_ns must be positive or None")
        if self.session_end_time_ns < self.target_time_ns:
            raise ValueError("hedge session must remain open through t0")
        if (
            self.rollback_session_end_time_ns is not None
            and self.rollback_session_end_time_ns < self.deadline_time_ns
        ):
            raise ValueError("rollback venue closes before the hedge timeout cursor")

    @property
    def target_time_ns(self) -> int:
        return self.trigger_cursor.recv_time_ns + self.delay_ns

    @property
    def deadline_time_ns(self) -> int:
        return min(self.target_time_ns + self.retry_ns, self.session_end_time_ns)


@dataclass(frozen=True)
class ExecutableBook:
    send_eligible_cursor: EventCursor
    book_cursor: RawBookCursor
    side: Side
    requested_quantity: int
    requested_quantity_unit: QuantityUnit
    executable_vwap: float
    best_price: float
    levels_swept: int


@dataclass(frozen=True)
class RollbackSpec:
    source_hedge_intent_id: str
    trigger_cursor: EventCursor
    deadline_time_ns: int
    initiating_first_leg_venue: str
    initiating_execution_id: str
    side: Side
    initiating_first_leg_quantity: int
    initiating_first_leg_quantity_unit: QuantityUnit

    def __post_init__(self) -> None:
        for name in (
            "source_hedge_intent_id",
            "initiating_first_leg_venue",
            "initiating_execution_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.trigger_cursor, EventCursor):
            raise TypeError("trigger_cursor must be an EventCursor")
        if (
            isinstance(self.deadline_time_ns, bool)
            or not isinstance(self.deadline_time_ns, int)
            or self.deadline_time_ns < self.trigger_cursor.recv_time_ns
        ):
            raise ValueError("rollback deadline cannot precede its trigger")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        _positive_integer(
            self.initiating_first_leg_quantity,
            "initiating_first_leg_quantity",
        )
        _validate_quantity_unit(
            self.initiating_first_leg_quantity_unit,
            "initiating_first_leg_quantity_unit",
        )

    @property
    def quantity(self) -> int:
        """Book-native rollback quantity from the initiating first leg."""

        return self.initiating_first_leg_quantity

    @property
    def quantity_unit(self) -> QuantityUnit:
        return self.initiating_first_leg_quantity_unit


@dataclass(frozen=True)
class ArrivalReference:
    """Independent trigger-time slippage reference.

    A missing or illegal arrival book produces an unavailable reference with
    a null VWAP.  It never gates the later t0 execution attempt.  Keeping the
    full raw-book cursor retains PacketSeq provenance for later latency and
    adverse-slippage calculation.
    """

    reference_cursor: EventCursor
    book_cursor: RawBookCursor | None
    side: Side
    requested_quantity: int
    requested_quantity_unit: QuantityUnit
    available: bool
    executable_vwap: float | None
    best_price: float | None
    levels_swept: int | None
    gate_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_cursor, EventCursor):
            raise TypeError("reference_cursor must be an EventCursor")
        if self.book_cursor is not None:
            if not isinstance(self.book_cursor, RawBookCursor):
                raise TypeError("book_cursor must be a RawBookCursor or None")
            if self.book_cursor.cursor > self.reference_cursor:
                raise ValueError("arrival book cannot follow its reference cursor")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        _positive_integer(self.requested_quantity, "requested_quantity")
        _validate_quantity_unit(
            self.requested_quantity_unit,
            "requested_quantity_unit",
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be boolean")
        if self.available:
            if self.book_cursor is None:
                raise ValueError("available arrival reference requires a book cursor")
            _positive_finite(self.executable_vwap, "executable_vwap")
            _positive_finite(self.best_price, "best_price")
            _positive_integer(self.levels_swept, "levels_swept")
            if self.gate_reason is not None:
                raise ValueError(
                    "available arrival reference cannot have a gate reason"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.executable_vwap,
                    self.best_price,
                    self.levels_swept,
                )
            ):
                raise ValueError(
                    "unavailable arrival reference must keep price fields null"
                )
            if not isinstance(self.gate_reason, str) or not self.gate_reason:
                raise ValueError("unavailable arrival reference requires a gate reason")

    def adverse_slippage_bp(self, execution_vwap: float) -> float | None:
        """Return positive-is-adverse slip, or null without an arrival VWAP."""

        execution = _positive_finite(execution_vwap, "execution_vwap")
        if not self.available:
            return None
        assert self.executable_vwap is not None
        if self.side == "buy":
            return (execution / self.executable_vwap - 1.0) * 10_000.0
        return (self.executable_vwap - execution) / self.executable_vwap * 10_000.0


@dataclass(frozen=True)
class HedgeAttemptResult:
    status: AttemptStatus
    target_time_ns: int
    deadline_time_ns: int
    target_evaluated: bool
    arrival_reference: ArrivalReference
    send_eligible: ExecutableBook | None = None
    actual_send: ExecutableBook | None = None
    initial_gate_reason: str | None = None
    last_gate_reason: str | None = None
    rollback: RollbackSpec | None = None


@dataclass(frozen=True)
class RollbackAttemptResult:
    status: RollbackAttemptStatus
    target_time_ns: int
    deadline_time_ns: int
    target_evaluated: bool
    arrival_reference: ArrivalReference
    send_eligible: ExecutableBook | None = None
    actual_send: ExecutableBook | None = None
    initial_gate_reason: str | None = None
    last_gate_reason: str | None = None


def capture_arrival_reference(
    state: CausalBookState | None,
    *,
    reference_cursor: EventCursor,
    side: Side,
    quantity: int,
    quantity_unit: QuantityUnit,
) -> ArrivalReference:
    """Capture a valid or explicitly-null trigger-time execution reference."""

    if not isinstance(reference_cursor, EventCursor):
        raise TypeError("reference_cursor must be an EventCursor")
    _positive_integer(quantity, "quantity")
    _validate_quantity_unit(quantity_unit, "quantity_unit")
    if state is None:
        return ArrivalReference(
            reference_cursor=reference_cursor,
            book_cursor=None,
            side=side,
            requested_quantity=quantity,
            requested_quantity_unit=quantity_unit,
            available=False,
            executable_vwap=None,
            best_price=None,
            levels_swept=None,
            gate_reason="missing_book",
        )
    executable, reason = executable_book(
        state,
        side=side,
        quantity=quantity,
        quantity_unit=quantity_unit,
        send_eligible_cursor=reference_cursor,
    )
    if executable is None:
        return ArrivalReference(
            reference_cursor=reference_cursor,
            book_cursor=state.book_cursor,
            side=side,
            requested_quantity=quantity,
            requested_quantity_unit=quantity_unit,
            available=False,
            executable_vwap=None,
            best_price=None,
            levels_swept=None,
            gate_reason=reason or "book_not_executable",
        )
    return ArrivalReference(
        reference_cursor=reference_cursor,
        book_cursor=executable.book_cursor,
        side=side,
        requested_quantity=quantity,
        requested_quantity_unit=quantity_unit,
        available=True,
        executable_vwap=executable.executable_vwap,
        best_price=executable.best_price,
        levels_swept=executable.levels_swept,
        gate_reason=None,
    )


class HedgeAttempt:
    """One retrying hedge lifecycle before venue quota assignment."""

    def __init__(
        self,
        intent: HedgeIntent,
        *,
        arrival_reference: ArrivalReference | None = None,
    ) -> None:
        if not isinstance(intent, HedgeIntent):
            raise TypeError("intent must be a HedgeIntent")
        self.intent = intent
        self._arrival_reference = arrival_reference or capture_arrival_reference(
            None,
            reference_cursor=intent.trigger_cursor,
            side=intent.side,
            quantity=intent.hedge_quantity,
            quantity_unit=intent.hedge_quantity_unit,
        )
        _validate_arrival_reference(
            self._arrival_reference,
            reference_cursor=intent.trigger_cursor,
            side=intent.side,
            quantity=intent.hedge_quantity,
            quantity_unit=intent.hedge_quantity_unit,
        )
        self._status: AttemptStatus = "waiting"
        self._eligible: ExecutableBook | None = None
        self._actual_send: ExecutableBook | None = None
        self._initial_reason: str | None = None
        self._last_reason: str | None = None
        self._rollback: RollbackSpec | None = None
        self._target_evaluated = False
        self._last_evaluation_cursor: EventCursor | None = None

    @property
    def terminal(self) -> bool:
        return self._status in ("hedge_sent", "hedge_retry_timeout")

    def observe(
        self,
        state: CausalBookState | None,
        *,
        evaluation_cursor: EventCursor | None = None,
    ) -> HedgeAttemptResult:
        """Consider a raw cursor; quota assignment remains a caller concern."""

        if state is not None and not isinstance(state, CausalBookState):
            raise TypeError("state must be a CausalBookState or None")
        if self.terminal:
            return self.result
        if state is None and evaluation_cursor is None:
            raise ValueError("missing book evaluation requires an explicit cursor")
        if evaluation_cursor is not None:
            cursor = evaluation_cursor
        else:
            assert state is not None
            cursor = state.book_cursor.cursor
        if not isinstance(cursor, EventCursor):
            raise TypeError("evaluation_cursor must be an EventCursor or None")
        if state is not None and cursor < state.book_cursor.cursor:
            raise ValueError("evaluation cursor cannot precede its causal book")
        now = cursor.recv_time_ns
        first_evaluation = not self._target_evaluated
        if first_evaluation:
            if now != self.intent.target_time_ns:
                raise ValueError("first hedge evaluation must occur exactly at t0")
            self._target_evaluated = True
        elif (
            self._last_evaluation_cursor is not None
            and cursor <= self._last_evaluation_cursor
        ):
            raise ValueError("retry evaluation cursors must be strictly increasing")
        if now > self.intent.deadline_time_ns:
            raise ValueError("cannot evaluate hedge after its inclusive deadline")
        self._last_evaluation_cursor = cursor
        if state is None:
            executable, reason = None, "missing_book"
        else:
            executable, reason = executable_book(
                state,
                side=self.intent.side,
                quantity=self.intent.hedge_quantity,
                quantity_unit=self.intent.hedge_quantity_unit,
                send_eligible_cursor=cursor,
            )
        if first_evaluation:
            self._initial_reason = reason
        self._last_reason = reason
        if executable is not None:
            self._eligible = executable
            self._status = "send_eligible"
        else:
            self._eligible = None
            self._status = "waiting"
        return self.result

    def confirm_actual_send(self, send_cursor: EventCursor) -> HedgeAttemptResult:
        """Confirm quota assignment at the just-revalidated candidate cursor."""

        if not isinstance(send_cursor, EventCursor):
            raise TypeError("send_cursor must be an EventCursor")
        if self.terminal:
            return self.result
        if self._status != "send_eligible" or self._eligible is None:
            raise ValueError("actual send requires a currently legal candidate")
        if send_cursor != self._eligible.send_eligible_cursor:
            raise ValueError("actual send cursor must equal the revalidated cursor")
        self._actual_send = self._eligible
        self._eligible = None
        self._status = "hedge_sent"
        return self.result

    def expire(self, timeout_cursor: EventCursor) -> HedgeAttemptResult:
        """Atomically time out the hedge and create its rollback specification."""

        if not isinstance(timeout_cursor, EventCursor):
            raise TypeError("timeout_cursor must be an EventCursor")
        if self.terminal:
            return self.result
        if not self._target_evaluated:
            raise ValueError("cannot expire before the explicit t0 evaluation")
        if (
            self._last_evaluation_cursor is None
            or self._last_evaluation_cursor.recv_time_ns != self.intent.deadline_time_ns
            or timeout_cursor.recv_time_ns != self.intent.deadline_time_ns
            or timeout_cursor <= self._last_evaluation_cursor
        ):
            raise ValueError(
                "timeout requires deadline-inclusive evaluation then a later phase cursor"
            )
        self._status = "hedge_retry_timeout"
        self._eligible = None
        rollback_end = (
            self.intent.rollback_session_end_time_ns
            if self.intent.rollback_session_end_time_ns is not None
            else self.intent.session_end_time_ns
        )
        self._rollback = RollbackSpec(
            source_hedge_intent_id=self.intent.hedge_intent_id,
            trigger_cursor=timeout_cursor,
            deadline_time_ns=min(
                self.intent.deadline_time_ns + self.intent.retry_ns, rollback_end
            ),
            initiating_first_leg_venue=self.intent.initiating_first_leg_venue,
            initiating_execution_id=self.intent.initiating_execution_id,
            side=("sell" if self.intent.initiating_first_leg_side == "buy" else "buy"),
            initiating_first_leg_quantity=(self.intent.initiating_first_leg_quantity),
            initiating_first_leg_quantity_unit=(
                self.intent.initiating_first_leg_quantity_unit
            ),
        )
        return self.result

    @property
    def result(self) -> HedgeAttemptResult:
        return HedgeAttemptResult(
            status=self._status,
            target_time_ns=self.intent.target_time_ns,
            deadline_time_ns=self.intent.deadline_time_ns,
            target_evaluated=self._target_evaluated,
            send_eligible=self._eligible,
            actual_send=self._actual_send,
            initial_gate_reason=self._initial_reason,
            last_gate_reason=self._last_reason,
            rollback=self._rollback,
            arrival_reference=self._arrival_reference,
        )


class RollbackAttempt:
    """One non-recursive, zero-delay emergency rollback lifecycle.

    The rollback reverses the initiating first leg in that venue's native
    quantity unit.  Book legality can appear and disappear before quota is
    assigned.  A timeout is terminal and deliberately cannot create another
    rollback.
    """

    def __init__(
        self,
        spec: RollbackSpec,
        *,
        arrival_reference: ArrivalReference | None = None,
    ) -> None:
        if not isinstance(spec, RollbackSpec):
            raise TypeError("spec must be a RollbackSpec")
        self.spec = spec
        self._arrival_reference = arrival_reference or capture_arrival_reference(
            None,
            reference_cursor=spec.trigger_cursor,
            side=spec.side,
            quantity=spec.quantity,
            quantity_unit=spec.quantity_unit,
        )
        _validate_arrival_reference(
            self._arrival_reference,
            reference_cursor=spec.trigger_cursor,
            side=spec.side,
            quantity=spec.quantity,
            quantity_unit=spec.quantity_unit,
        )
        self._status: RollbackAttemptStatus = "waiting"
        self._eligible: ExecutableBook | None = None
        self._actual_send: ExecutableBook | None = None
        self._initial_reason: str | None = None
        self._last_reason: str | None = None
        self._target_evaluated = False
        self._last_evaluation_cursor: EventCursor | None = None

    @property
    def terminal(self) -> bool:
        return self._status in ("rollback_sent", "rollback_retry_timeout")

    def observe(
        self,
        state: CausalBookState | None,
        *,
        evaluation_cursor: EventCursor | None = None,
    ) -> RollbackAttemptResult:
        if state is not None and not isinstance(state, CausalBookState):
            raise TypeError("state must be a CausalBookState or None")
        if self.terminal:
            return self.result
        if state is None and evaluation_cursor is None:
            raise ValueError("missing book evaluation requires an explicit cursor")
        if evaluation_cursor is not None:
            cursor = evaluation_cursor
        else:
            assert state is not None
            cursor = state.book_cursor.cursor
        if not isinstance(cursor, EventCursor):
            raise TypeError("evaluation_cursor must be an EventCursor or None")
        if state is not None and cursor < state.book_cursor.cursor:
            raise ValueError("evaluation cursor cannot precede its causal book")
        now = cursor.recv_time_ns
        first_evaluation = not self._target_evaluated
        if first_evaluation:
            if now != self.spec.trigger_cursor.recv_time_ns:
                raise ValueError(
                    "first rollback evaluation must occur exactly at its trigger"
                )
            if cursor < self.spec.trigger_cursor:
                raise ValueError(
                    "rollback evaluation cannot precede its trigger cursor"
                )
            self._target_evaluated = True
        elif (
            self._last_evaluation_cursor is not None
            and cursor <= self._last_evaluation_cursor
        ):
            raise ValueError("retry evaluation cursors must be strictly increasing")
        if now > self.spec.deadline_time_ns:
            raise ValueError("cannot evaluate rollback after its inclusive deadline")
        self._last_evaluation_cursor = cursor
        if state is None:
            executable, reason = None, "missing_book"
        else:
            executable, reason = executable_book(
                state,
                side=self.spec.side,
                quantity=self.spec.quantity,
                quantity_unit=self.spec.quantity_unit,
                send_eligible_cursor=cursor,
            )
        if first_evaluation:
            self._initial_reason = reason
        self._last_reason = reason
        if executable is not None:
            self._eligible = executable
            self._status = "send_eligible"
        else:
            self._eligible = None
            self._status = "waiting"
        return self.result

    def confirm_actual_send(
        self,
        send_cursor: EventCursor,
    ) -> RollbackAttemptResult:
        if not isinstance(send_cursor, EventCursor):
            raise TypeError("send_cursor must be an EventCursor")
        if self.terminal:
            return self.result
        if self._status != "send_eligible" or self._eligible is None:
            raise ValueError("actual send requires a currently legal candidate")
        if send_cursor != self._eligible.send_eligible_cursor:
            raise ValueError("actual send cursor must equal the revalidated cursor")
        self._actual_send = self._eligible
        self._eligible = None
        self._status = "rollback_sent"
        return self.result

    def expire(self, timeout_cursor: EventCursor) -> RollbackAttemptResult:
        if not isinstance(timeout_cursor, EventCursor):
            raise TypeError("timeout_cursor must be an EventCursor")
        if self.terminal:
            return self.result
        if not self._target_evaluated:
            raise ValueError("cannot expire before the explicit trigger evaluation")
        if (
            self._last_evaluation_cursor is None
            or self._last_evaluation_cursor.recv_time_ns != self.spec.deadline_time_ns
            or timeout_cursor.recv_time_ns != self.spec.deadline_time_ns
            or timeout_cursor <= self._last_evaluation_cursor
        ):
            raise ValueError(
                "timeout requires deadline-inclusive evaluation then a later phase cursor"
            )
        self._status = "rollback_retry_timeout"
        self._eligible = None
        return self.result

    @property
    def result(self) -> RollbackAttemptResult:
        return RollbackAttemptResult(
            status=self._status,
            target_time_ns=self.spec.trigger_cursor.recv_time_ns,
            deadline_time_ns=self.spec.deadline_time_ns,
            target_evaluated=self._target_evaluated,
            send_eligible=self._eligible,
            actual_send=self._actual_send,
            initial_gate_reason=self._initial_reason,
            last_gate_reason=self._last_reason,
            arrival_reference=self._arrival_reference,
        )


def executable_book(
    state: CausalBookState,
    *,
    side: Side,
    quantity: int,
    quantity_unit: QuantityUnit,
    send_eligible_cursor: EventCursor | None = None,
) -> tuple[ExecutableBook | None, str | None]:
    """Validate a legal full-depth book and compute requested-quantity VWAP."""

    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    _positive_integer(quantity, "quantity")
    _validate_quantity_unit(quantity_unit, "quantity_unit")
    eligible_cursor = send_eligible_cursor or state.book_cursor.cursor
    if not isinstance(eligible_cursor, EventCursor):
        raise TypeError("send_eligible_cursor must be an EventCursor or None")
    if eligible_cursor < state.book_cursor.cursor:
        raise ValueError("send-eligible cursor cannot precede its causal book")
    if not state.gate_open:
        return None, state.gate_reason or "gate_closed"
    reference = state.reference_price
    if (
        not isinstance(reference, (int, float))
        or isinstance(reference, bool)
        or not math.isfinite(float(reference))
        or reference <= 0
    ):
        return None, "invalid_reference_price"
    if not state.bids or not state.asks:
        return None, "empty_book_side"
    if state.bids[0].price > state.asks[0].price:
        return None, "crossed_book"
    levels = state.asks if side == "buy" else state.bids
    remaining = quantity
    notional = 0.0
    swept = 0
    lower = float(reference) * 0.91
    upper = float(reference) * 1.08
    if not (
        lower < state.bids[0].price < upper and lower < state.asks[0].price < upper
    ):
        return None, "bbo_outside_reference_band"
    for level in levels:
        take = min(remaining, level.quantity)
        if take <= 0:
            continue
        if not lower < level.price < upper:
            return None, "swept_level_outside_reference_band"
        notional += level.price * take
        remaining -= take
        swept += 1
        if remaining == 0:
            return (
                ExecutableBook(
                    eligible_cursor,
                    state.book_cursor,
                    side,
                    quantity,
                    quantity_unit,
                    notional / quantity,
                    levels[0].price,
                    swept,
                ),
                None,
            )
    return None, "insufficient_depth"


def _validate_arrival_reference(
    reference: ArrivalReference,
    *,
    reference_cursor: EventCursor,
    side: Side,
    quantity: int,
    quantity_unit: QuantityUnit,
) -> None:
    if not isinstance(reference, ArrivalReference):
        raise TypeError("arrival_reference must be an ArrivalReference")
    if reference.reference_cursor != reference_cursor:
        raise ValueError("arrival reference cursor does not match attempt trigger")
    if reference.side != side:
        raise ValueError("arrival reference side does not match attempt side")
    if reference.requested_quantity != quantity:
        raise ValueError("arrival reference quantity does not match attempt quantity")
    if reference.requested_quantity_unit != quantity_unit:
        raise ValueError("arrival reference unit does not match attempt unit")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _validate_quantity_unit(value: object, name: str) -> QuantityUnit:
    if value == "spot_shares":
        return "spot_shares"
    if value == "future_contracts":
        return "future_contracts"
    raise ValueError(f"{name} must be spot_shares or future_contracts")


def _merge_best_l1(
    l1: tuple[RawBookLevel, ...],
    best: RawBookLevel | None,
    *,
    descending: bool,
) -> tuple[RawBookLevel, ...]:
    candidates = list(l1)
    if best is not None:
        candidates.append(best)
    by_price: dict[str, RawBookLevel] = {}
    for level in candidates:
        key = float(level.price).hex()
        previous = by_price.get(key)
        if previous is None or level.quantity > previous.quantity:
            by_price[key] = level
    return tuple(
        sorted(by_price.values(), key=lambda level: level.price, reverse=descending)[:5]
    )


__all__ = [
    "HEDGE_DELAY_NS",
    "HEDGE_RETRY_NS",
    "ArrivalReference",
    "CausalBookState",
    "ExecutableBook",
    "HedgeAttempt",
    "HedgeAttemptResult",
    "HedgeIntent",
    "QuantityUnit",
    "RawBookCursor",
    "RawBookEvent",
    "RawBookLevel",
    "RawBookStateMachine",
    "RollbackAttempt",
    "RollbackAttemptResult",
    "RollbackSpec",
    "capture_arrival_reference",
    "executable_book",
]
