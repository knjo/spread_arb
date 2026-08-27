"""Pure deterministic S1 chronological replay core.

One instance owns one ``Date x policy`` scenario.  All products share the
same spot/future rolling request schedulers and the same 20M/10M capacity
ledger.  File loading, legacy makerFill lookup, and accounting persistence are
deliberately adapter boundaries so this module can be tested with synthetic
causal events.

Within one receive timestamp the canonical effect order is:

1. raw books and desired entry observations;
2. frozen venue assignments and pre-send capacity reservations;
3. already-working maker fills;
4. assigned marketable hedge/rollback executions;
5. cancel effects;
6. session expiry;
7. assigned new orders become working;
8. deadline timeout and rollback creation;
9. same-timestamp emergency rollback assignment/execution.

Consequently a working order can fill before its assigned cancel, while a new
order cannot consume a same-timestamp fill.  New admission uses the committed
balance at assignment phase and never a later same-timestamp release.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, Protocol

from .capacity_ledger import (
    DEFAULT_GLOBAL_CAP_TWD,
    DEFAULT_PRODUCT_CAP_TWD,
    CapacityLedger,
    CapacityReplayResult,
    CapacityTransition,
)
from .layered import EventCursor
from .replay import TradeEvent
from .s1_accounting import (
    AccountingError,
    InitiatingExecutionAllocation,
    PositionEstablishedFact,
    decode_accounting_fact,
    encode_accounting_fact,
)
from .s1_admission import CapacityAdmissionPlanner
from .s1_entry_controller import (
    CandidateIntentAudit,
    EntryControllerCommand,
    S1EntryController,
)
from .s1_exit_fifo_coordinator import build_exit_fifo_desired_updates
from .s1_exit_fill_allocator import ExitPhysicalFill, S1SpotAskFillAllocator
from .s1_exit_inventory import (
    ExitControllerCommand,
    ExitFillAllocation,
    ExitHedgeUnitRequest,
    ExitInventoryFact,
    ExitPositionView,
    ExitRollbackRequest,
    S1ExitInventoryController,
)
from .s1_exit_target import S1SpotAskTarget, build_s1_spot_ask_target
from .s1_hedge import (
    HEDGE_DELAY_NS,
    HEDGE_RETRY_NS,
    ArrivalReference,
    CausalBookState,
    HedgeAttempt,
    HedgeIntent,
    RawBookCursor,
    RawBookEvent,
    RawBookStateMachine,
    RollbackAttempt,
    RollbackSpec,
    capture_arrival_reference,
)
from .s1_spot_close_adapter import OfficialSpotClose
from .s1_spot_trade_adapter import PhysicalSpotTrade
from .targets import absolute_price_tick, tick_index_to_price
from .venue_scheduler import (
    ONE_SECOND_NS,
    RequestClass,
    RiskSubtype,
    RollingVenueScheduler,
    VenueRequestIntent,
    VenueSendAssignment,
)

PHASE_OBSERVE = 100
PHASE_PRE_SEND_REFRESH = 150
PHASE_ASSIGN = 200
PHASE_MAKER_FILL = 300
PHASE_MARKETABLE = 400
PHASE_SETTLEMENT = 450
PHASE_CANCEL = 500
PHASE_SESSION_EXPIRY = 600
PHASE_NEW_WORKING = 700
PHASE_RISK_TIMEOUT = 800
PHASE_ROLLBACK_ASSIGN = 900
PHASE_ROLLBACK_EXECUTION = 1_000
PHASE_POST_ROLLBACK_TIMEOUT = 1_100

ROUTE = "spot_bid_future_taker"
STAGE = "entry"
EXIT_ROUTE = "spot_ask_future_taker"
EXIT_STAGE = "exit"
SPOT = "spot"
FUTURE = "future"
CAP_RETRY_DELAY_NS = 1
S1_CARRY_POSITION_RECORD_SCHEMA_VERSION = 1
S1_CARRY_CONTRACT_BINDING_RECORD_SCHEMA_VERSION = 1
_S1_CARRY_POSITION_RECORD_TYPE = "s1_carry_position"
_S1_CARRY_CONTRACT_BINDING_RECORD_TYPE = "s1_carry_contract_binding"

Venue = Literal["spot", "future"]
RequestBindingKind = Literal["new", "cancel", "hedge", "rollback"]
RequestStage = Literal["entry", "exit"]
PositionState = Literal[
    "hedge_pending",
    "paired_open",
    "entry_rollback_pending",
    "entry_emergency_rollback_flat",
    "entry_hedge_timeout_unresolved",
    "exit_in_progress",
    "exit_rollback_pending",
    "exit_maker_flat",
    "exit_rollback_failed_unresolved",
    "expiry_basis_zero_accounting",
]
ExecutionRole = Literal[
    "entry_maker",
    "entry_hedge",
    "entry_rollback",
    "exit_maker",
    "exit_hedge",
    "exit_rollback",
]
ExecutionTruth = Literal["approximate", "exact"]


@dataclass(frozen=True, slots=True)
class S1LoopConfig:
    date: str
    policy_id: str
    global_cap_twd: int = DEFAULT_GLOBAL_CAP_TWD
    product_cap_twd: int = DEFAULT_PRODUCT_CAP_TWD
    spot_request_cap: int = 100
    future_request_cap: int = 5
    request_window_ns: int = ONE_SECOND_NS

    def __post_init__(self) -> None:
        _text(self.date, "date")
        if len(self.date) != 8 or not self.date.isdigit():
            raise ValueError("date must be YYYYMMDD")
        _text(self.policy_id, "policy_id")
        for name in (
            "global_cap_twd",
            "product_cap_twd",
            "spot_request_cap",
            "future_request_cap",
            "request_window_ns",
        ):
            _positive_int(getattr(self, name), name)
        if self.product_cap_twd > self.global_cap_twd:
            raise ValueError("product cap cannot exceed global cap")


@dataclass(frozen=True, slots=True)
class S1Product:
    product_id: str
    value_code: str
    quote_code: str
    contract_size_shares: int
    future_contracts: int
    future_session_end_time_ns: int
    spot_session_end_time_ns: int
    end_date: str | None = None

    def __post_init__(self) -> None:
        for name in ("product_id", "value_code", "quote_code"):
            _text(getattr(self, name), name)
        for name in (
            "contract_size_shares",
            "future_contracts",
            "future_session_end_time_ns",
            "spot_session_end_time_ns",
        ):
            _positive_int(getattr(self, name), name)
        if self.end_date is not None:
            _valid_date(self.end_date, "end_date")


@dataclass(frozen=True, slots=True)
class S1CarryContractBinding:
    """Frozen contract and quote identity persisted beside an open carry."""

    product_id: str
    value_code: str
    quote_code: str
    contract_size_shares: int
    future_contracts: int
    end_date: str | None

    def __post_init__(self) -> None:
        for name in ("product_id", "value_code", "quote_code"):
            _text(getattr(self, name), name)
        _positive_int(self.contract_size_shares, "contract_size_shares")
        _positive_int(self.future_contracts, "future_contracts")
        if self.end_date is not None:
            _valid_date(self.end_date, "end_date")

    @classmethod
    def from_product(cls, product: S1Product) -> S1CarryContractBinding:
        if not isinstance(product, S1Product):
            raise TypeError("product must be an S1Product")
        return cls(
            product_id=product.product_id,
            value_code=product.value_code,
            quote_code=product.quote_code,
            contract_size_shares=product.contract_size_shares,
            future_contracts=product.future_contracts,
            end_date=product.end_date,
        )


@dataclass(frozen=True, slots=True)
class S1CarryPosition:
    """Exactly paired position state permitted to cross one day boundary.

    Venue orders, queue age, scheduler tokens, and pending risk requests are
    deliberately absent.  They are session-local and must be rebuilt from the
    next day's first causal raw books.
    """

    position_established_fact: PositionEstablishedFact
    product_id: str
    quote_code: str
    capacity_id: str
    initiating_raw_order_fact_id: str
    hedge_intent_id: str
    frozen_exit_threshold_basis_bp: float
    execution_truth: ExecutionTruth

    def __post_init__(self) -> None:
        fact = self.position_established_fact
        if not isinstance(fact, PositionEstablishedFact):
            raise TypeError(
                "position_established_fact must be a PositionEstablishedFact"
            )
        for name in (
            "product_id",
            "quote_code",
            "capacity_id",
            "initiating_raw_order_fact_id",
            "hedge_intent_id",
        ):
            _text(getattr(self, name), name)
        if self.capacity_id != fact.capacity_id:
            raise ValueError("carry capacity_id differs from its establishment fact")
        if self.execution_truth not in ("approximate", "exact"):
            raise ValueError("unsupported carry execution truth")
        if self.execution_truth != fact.execution_truth:
            raise ValueError("carry execution truth differs from establishment fact")
        _finite_float(
            self.frozen_exit_threshold_basis_bp,
            "frozen_exit_threshold_basis_bp",
        )

    @property
    def position_id(self) -> str:
        return self.position_established_fact.position_id


class S1CarryCodecError(ValueError):
    """A canonical JSON carry or contract-binding record is malformed."""


def encode_s1_carry_position(carry: S1CarryPosition) -> dict[str, object]:
    """Encode one cross-day carry as a canonical JSON-safe object."""

    if not isinstance(carry, S1CarryPosition):
        raise TypeError("carry must be an S1CarryPosition")
    _validate_carry_position_for_codec(carry)
    return {
        "record_type": _S1_CARRY_POSITION_RECORD_TYPE,
        "schema_version": S1_CARRY_POSITION_RECORD_SCHEMA_VERSION,
        "position_established_fact": encode_accounting_fact(
            carry.position_established_fact
        ),
        "product_id": carry.product_id,
        "quote_code": carry.quote_code,
        "capacity_id": carry.capacity_id,
        "initiating_raw_order_fact_id": carry.initiating_raw_order_fact_id,
        "hedge_intent_id": carry.hedge_intent_id,
        "frozen_exit_threshold_basis_bp": carry.frozen_exit_threshold_basis_bp,
        "execution_truth": carry.execution_truth,
    }


def decode_s1_carry_position(record: Mapping[str, object]) -> S1CarryPosition:
    """Strictly reconstruct one carry, reusing the accounting fact codec."""

    if not isinstance(record, Mapping):
        raise TypeError("S1 carry position record must be a mapping")
    expected_keys = {
        "record_type",
        "schema_version",
        "position_established_fact",
        "product_id",
        "quote_code",
        "capacity_id",
        "initiating_raw_order_fact_id",
        "hedge_intent_id",
        "frozen_exit_threshold_basis_bp",
        "execution_truth",
    }
    _require_carry_codec_keys(record, expected_keys, "S1 carry position")
    _require_carry_codec_tag(
        record,
        expected_type=_S1_CARRY_POSITION_RECORD_TYPE,
        expected_schema=S1_CARRY_POSITION_RECORD_SCHEMA_VERSION,
        path="S1 carry position",
    )
    fact_record = record["position_established_fact"]
    if not isinstance(fact_record, Mapping):
        raise S1CarryCodecError(
            "S1 carry position position_established_fact must be a JSON object"
        )
    try:
        fact = decode_accounting_fact(fact_record)
    except (AccountingError, TypeError) as error:
        raise S1CarryCodecError(
            f"S1 carry position establishment fact is invalid: {error}"
        ) from error
    if not isinstance(fact, PositionEstablishedFact):
        raise S1CarryCodecError(
            "S1 carry position requires a position_established accounting fact"
        )
    threshold = record["frozen_exit_threshold_basis_bp"]
    if type(threshold) is not float or not math.isfinite(threshold):
        raise S1CarryCodecError(
            "S1 carry position frozen_exit_threshold_basis_bp must be a finite float"
        )
    execution_truth = record["execution_truth"]
    if type(execution_truth) is not str or execution_truth not in (
        "approximate",
        "exact",
    ):
        raise S1CarryCodecError("S1 carry position execution_truth is invalid")
    try:
        carry = S1CarryPosition(
            position_established_fact=fact,
            product_id=_decode_carry_codec_text(
                record["product_id"], "S1 carry position product_id"
            ),
            quote_code=_decode_carry_codec_text(
                record["quote_code"], "S1 carry position quote_code"
            ),
            capacity_id=_decode_carry_codec_text(
                record["capacity_id"], "S1 carry position capacity_id"
            ),
            initiating_raw_order_fact_id=_decode_carry_codec_text(
                record["initiating_raw_order_fact_id"],
                "S1 carry position initiating_raw_order_fact_id",
            ),
            hedge_intent_id=_decode_carry_codec_text(
                record["hedge_intent_id"], "S1 carry position hedge_intent_id"
            ),
            frozen_exit_threshold_basis_bp=threshold,
            execution_truth=execution_truth,
        )
        _validate_carry_position_for_codec(carry)
    except (TypeError, ValueError) as error:
        if isinstance(error, S1CarryCodecError):
            raise
        raise S1CarryCodecError(str(error)) from error
    return carry


def encode_s1_carry_contract_binding(
    binding: S1CarryContractBinding,
) -> dict[str, object]:
    """Encode one frozen carry contract binding as canonical JSON data."""

    if not isinstance(binding, S1CarryContractBinding):
        raise TypeError("binding must be an S1CarryContractBinding")
    _validate_carry_contract_binding_for_codec(binding)
    return {
        "record_type": _S1_CARRY_CONTRACT_BINDING_RECORD_TYPE,
        "schema_version": S1_CARRY_CONTRACT_BINDING_RECORD_SCHEMA_VERSION,
        "product_id": binding.product_id,
        "value_code": binding.value_code,
        "quote_code": binding.quote_code,
        "contract_size_shares": binding.contract_size_shares,
        "future_contracts": binding.future_contracts,
        "end_date": binding.end_date,
    }


def decode_s1_carry_contract_binding(
    record: Mapping[str, object],
) -> S1CarryContractBinding:
    """Strictly reconstruct one frozen carry contract binding."""

    if not isinstance(record, Mapping):
        raise TypeError("S1 carry contract binding record must be a mapping")
    expected_keys = {
        "record_type",
        "schema_version",
        "product_id",
        "value_code",
        "quote_code",
        "contract_size_shares",
        "future_contracts",
        "end_date",
    }
    _require_carry_codec_keys(record, expected_keys, "S1 carry contract binding")
    _require_carry_codec_tag(
        record,
        expected_type=_S1_CARRY_CONTRACT_BINDING_RECORD_TYPE,
        expected_schema=S1_CARRY_CONTRACT_BINDING_RECORD_SCHEMA_VERSION,
        path="S1 carry contract binding",
    )
    end_date = record["end_date"]
    if end_date is not None and type(end_date) is not str:
        raise S1CarryCodecError(
            "S1 carry contract binding end_date must be a string or null"
        )
    try:
        binding = S1CarryContractBinding(
            product_id=_decode_carry_codec_text(
                record["product_id"], "S1 carry contract binding product_id"
            ),
            value_code=_decode_carry_codec_text(
                record["value_code"], "S1 carry contract binding value_code"
            ),
            quote_code=_decode_carry_codec_text(
                record["quote_code"], "S1 carry contract binding quote_code"
            ),
            contract_size_shares=_decode_carry_codec_positive_int(
                record["contract_size_shares"],
                "S1 carry contract binding contract_size_shares",
            ),
            future_contracts=_decode_carry_codec_positive_int(
                record["future_contracts"],
                "S1 carry contract binding future_contracts",
            ),
            end_date=end_date,
        )
        _validate_carry_contract_binding_for_codec(binding)
    except (TypeError, ValueError) as error:
        if isinstance(error, S1CarryCodecError):
            raise
        raise S1CarryCodecError(str(error)) from error
    return binding


@dataclass(frozen=True, slots=True)
class ActualSendMakerSnapshot:
    """Exact raw spot snapshot frozen by the actual-send state resolver."""

    value_code: str
    channel_seq: int
    recv_time_ns: int
    bid_price1: float | None
    bid_price2: float | None
    bid_lots1: int | None
    bid_lots2: int | None

    def __post_init__(self) -> None:
        _text(self.value_code, "value_code")
        _nonnegative_int(self.channel_seq, "channel_seq")
        _nonnegative_int(self.recv_time_ns, "recv_time_ns")
        for name in ("bid_price1", "bid_price2"):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)
        for name in ("bid_lots1", "bid_lots2"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int(value, name)


@dataclass(frozen=True, slots=True)
class EntryObservation:
    source_cursor: EventCursor
    product_id: str
    absolute_price_tick: int | None
    target_price: float | None
    reservation_notional_twd: int | None
    base_gate_open: bool
    admission_open: bool
    gate_reason: str = "base_gate_closed"
    maker_snapshot: ActualSendMakerSnapshot | None = None
    frozen_exit_threshold_basis_bp: float | None = None

    def __post_init__(self) -> None:
        _cursor(self.source_cursor, "source_cursor")
        _text(self.product_id, "product_id")
        if not isinstance(self.base_gate_open, bool) or not isinstance(
            self.admission_open, bool
        ):
            raise TypeError("gate flags must be boolean")
        if self.admission_open and not self.base_gate_open:
            raise ValueError("admission cannot be open while base gate is closed")
        _text(self.gate_reason, "gate_reason")
        if self.absolute_price_tick is not None:
            _positive_int(self.absolute_price_tick, "absolute_price_tick")
        if self.target_price is not None:
            _positive_float(self.target_price, "target_price")
        if self.reservation_notional_twd is not None:
            _positive_int(
                self.reservation_notional_twd,
                "reservation_notional_twd",
            )
        if self.base_gate_open and (
            self.absolute_price_tick is None or self.target_price is None
        ):
            raise ValueError("open base gate requires a target tick and price")
        if self.admission_open and self.reservation_notional_twd is None:
            raise ValueError("open admission requires reservation notional")
        if self.maker_snapshot is not None:
            if not isinstance(self.maker_snapshot, ActualSendMakerSnapshot):
                raise TypeError(
                    "maker_snapshot must be an ActualSendMakerSnapshot or None"
                )
            if self.maker_snapshot.recv_time_ns > self.source_cursor.recv_time_ns:
                raise ValueError("maker snapshot cannot follow its state observation")
        if self.admission_open and self.maker_snapshot is None:
            raise ValueError("open admission requires an exact maker snapshot")
        if self.frozen_exit_threshold_basis_bp is not None:
            _finite_float(
                self.frozen_exit_threshold_basis_bp,
                "frozen_exit_threshold_basis_bp",
            )


@dataclass(frozen=True, slots=True)
class VenueBookUpdate:
    venue: Venue
    product_id: str
    event: RawBookEvent

    def __post_init__(self) -> None:
        if self.venue not in (SPOT, FUTURE):
            raise ValueError("venue must be spot or future")
        _text(self.product_id, "product_id")
        if not isinstance(self.event, RawBookEvent):
            raise TypeError("event must be a RawBookEvent")
        if self.event.book_cursor.cursor.event_sequence >= PHASE_OBSERVE:
            raise ValueError("raw-book cursor must precede loop effect phases")


@dataclass(frozen=True, slots=True)
class EntryCutoff:
    source_cursor: EventCursor

    def __post_init__(self) -> None:
        _cursor(self.source_cursor, "source_cursor")


@dataclass(frozen=True, slots=True)
class SessionExpiry:
    source_cursor: EventCursor

    def __post_init__(self) -> None:
        _cursor(self.source_cursor, "source_cursor")


@dataclass(frozen=True, slots=True)
class ContractExpiry:
    """Official non-executable basis-zero settlement trigger."""

    close: OfficialSpotClose

    def __post_init__(self) -> None:
        if not isinstance(self.close, OfficialSpotClose):
            raise TypeError("close must be an OfficialSpotClose")
        if self.close.source_cursor.event_sequence >= PHASE_OBSERVE:
            raise ValueError("official close source must precede loop effect phases")


type S1ExternalEvent = (
    EntryObservation | VenueBookUpdate | EntryCutoff | SessionExpiry | ContractExpiry
)


@dataclass(frozen=True, slots=True)
class SentEntryOrder:
    date: str
    policy_id: str
    product_id: str
    value_code: str
    quote_code: str
    request_id: str
    candidate_intent_id: str
    raw_order_fact_id: str
    policy_alias_id: str
    capacity_id: str
    actual_start_cursor: EventCursor
    absolute_price_tick: int
    target_price: float
    reservation_notional_twd: int
    contract_size_shares: int
    maker_snapshot: ActualSendMakerSnapshot
    frozen_exit_threshold_basis_bp: float | None

    @property
    def maker_snapshot_channel_seq(self) -> int:
        return self.maker_snapshot.channel_seq

    @property
    def maker_snapshot_recv_time_ns(self) -> int:
        return self.maker_snapshot.recv_time_ns


@dataclass(frozen=True, slots=True)
class PotentialEntryFill:
    fill_time_ns: int
    source_id: str
    execution_truth: ExecutionTruth = "approximate"
    fill_cursor_exact: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.fill_time_ns, "fill_time_ns")
        _text(self.source_id, "source_id")
        if self.execution_truth not in ("approximate", "exact"):
            raise ValueError("unsupported execution truth")
        if not isinstance(self.fill_cursor_exact, bool):
            raise TypeError("fill_cursor_exact must be boolean")


class EntryFillAdapter(Protocol):
    def potential_fill(
        self,
        order: SentEntryOrder,
        snapshot: ActualSendMakerSnapshot,
    ) -> PotentialEntryFill | None: ...


class EntryStateAdapter(Protocol):
    """Return the complete state frozen at an actual assignment cursor.

    Sparse intent observations are insufficient here: after a venue-token
    delay the target, AB1/2 flag, maker snapshot, and reservation notional all
    have to be looked up again at the candidate send timestamp.
    """

    def current_state(
        self,
        product_id: str,
        assignment_cursor: EventCursor,
    ) -> EntryObservation | None: ...


class RiskBookAdapter(Protocol):
    """Causal random access to raw risk books without replaying a full day.

    ``next_change_cursor`` must return the earliest state transition after the
    supplied cursor, including TrialMatch invalidation and the later formal
    reopen as separate changes.  It returns ``None`` only when no transition
    exists through the inclusive deadline.
    """

    def state_as_of(
        self,
        venue: Venue,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None: ...

    def next_change_cursor(
        self,
        venue: Venue,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None: ...


class SpotTradeAdapter(Protocol):
    """Query exact physical spot prints only while an exit order is live."""

    def next_trade_cursor(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None: ...

    def trades_at(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> tuple[PhysicalSpotTrade, ...]: ...


@dataclass(frozen=True, slots=True)
class S1ExecutionFact:
    execution_id: str
    position_id: str
    product_id: str
    capacity_id: str
    request_id: str | None
    role: ExecutionRole
    market: Venue
    side: Literal["buy", "sell"]
    cursor: EventCursor
    price: float
    quantity: int
    quantity_unit: Literal["spot_shares", "future_contracts"]
    execution_truth: ExecutionTruth
    execution_source_id: str
    book_recv_time_ns: int | None = None
    book_packet_sequence: int | None = None
    initiating_execution_allocations: tuple[InitiatingExecutionAllocation, ...] = ()
    domain_allocation_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "execution_id",
            "position_id",
            "product_id",
            "capacity_id",
            "execution_source_id",
        ):
            _text(getattr(self, name), name)
        if self.request_id is not None:
            _text(self.request_id, "request_id")
        if self.role not in (
            "entry_maker",
            "entry_hedge",
            "entry_rollback",
            "exit_maker",
            "exit_hedge",
            "exit_rollback",
        ):
            raise ValueError("unsupported execution role")
        if self.market not in (SPOT, FUTURE):
            raise ValueError("execution market must be spot or future")
        if self.side not in ("buy", "sell"):
            raise ValueError("execution side must be buy or sell")
        _cursor(self.cursor, "cursor")
        _positive_float(self.price, "price")
        _positive_int(self.quantity, "quantity")
        if self.quantity_unit not in ("spot_shares", "future_contracts"):
            raise ValueError("unsupported execution quantity unit")
        if self.execution_truth not in ("approximate", "exact"):
            raise ValueError("unsupported execution truth")
        expected = {
            "entry_maker": (SPOT, "buy", "spot_shares"),
            "entry_hedge": (FUTURE, "sell", "future_contracts"),
            "entry_rollback": (SPOT, "sell", "spot_shares"),
            "exit_maker": (SPOT, "sell", "spot_shares"),
            "exit_hedge": (FUTURE, "buy", "future_contracts"),
            "exit_rollback": (SPOT, "buy", "spot_shares"),
        }[self.role]
        if (self.market, self.side, self.quantity_unit) != expected:
            raise ValueError("execution role/market/side/unit semantics drifted")
        if (self.book_recv_time_ns is None) != (self.book_packet_sequence is None):
            raise ValueError("book time and packet sequence must be jointly null")
        if self.book_recv_time_ns is not None:
            _nonnegative_int(self.book_recv_time_ns, "book_recv_time_ns")
            _nonnegative_int(self.book_packet_sequence, "book_packet_sequence")
            if self.book_recv_time_ns > self.cursor.recv_time_ns:
                raise ValueError("execution book cannot follow its execution cursor")
        if not isinstance(self.initiating_execution_allocations, tuple) or any(
            not isinstance(value, InitiatingExecutionAllocation)
            for value in self.initiating_execution_allocations
        ):
            raise TypeError(
                "initiating_execution_allocations must contain "
                "InitiatingExecutionAllocation values"
            )
        if (
            self.role
            in (
                "entry_maker",
                "exit_maker",
            )
            and self.initiating_execution_allocations
        ):
            raise ValueError("maker executions cannot cite initiating executions")
        if self.domain_allocation_id is not None:
            _text(self.domain_allocation_id, "domain_allocation_id")
        if self.role == "exit_maker" and self.domain_allocation_id is None:
            raise ValueError("exit maker execution requires domain_allocation_id")


class ExecutionAdapter(Protocol):
    def record_execution(self, fact: S1ExecutionFact) -> None: ...


class AccountingSettlementAdapter(ExecutionAdapter, Protocol):
    def establish_position(
        self,
        *,
        establishment_id: str,
        position_id: str,
        product_id: str,
        capacity_id: str,
        cursor: EventCursor,
        capacity_transition_id: str,
        execution_truth: str,
    ) -> PositionEstablishedFact: ...

    def seal_terminal(
        self,
        *,
        terminal_id: str,
        position_id: str,
        cursor: EventCursor,
        terminal_outcome: str,
        capacity_release_transition_id: str,
    ) -> object: ...

    def record_expiry_mark(
        self,
        *,
        mark_id: str,
        position_id: str,
        cursor: EventCursor,
        capacity_release_transition_id: str,
        spot_close_source_id: str,
        spot_close_source_cursor: EventCursor,
        spot_close_price: float,
    ) -> object: ...

    def verify(self) -> object: ...


class NoFillAdapter:
    def potential_fill(
        self,
        order: SentEntryOrder,
        snapshot: ActualSendMakerSnapshot,
    ) -> None:
        del order, snapshot


class NoOpExecutionAdapter:
    def record_execution(self, fact: S1ExecutionFact) -> None:
        del fact


@dataclass(frozen=True, slots=True)
class RequestEvent:
    request_id: str
    product_id: str
    venue: Venue
    request_class: RequestClass
    risk_subtype: RiskSubtype | None
    event_type: str
    event_cursor: EventCursor
    original_cursor: EventCursor
    send_sequence: int | None = None
    effect_status: str | None = None
    stage: RequestStage = STAGE


@dataclass(frozen=True, slots=True)
class AdmissionEvent:
    request_id: str
    product_id: str
    capacity_id: str
    cursor: EventCursor
    requested_notional_twd: int
    status: str
    admitted: bool


@dataclass(frozen=True, slots=True)
class OrderEvent:
    raw_order_fact_id: str
    candidate_intent_id: str
    capacity_id: str
    product_id: str
    event_type: str
    cursor: EventCursor
    request_id: str | None
    stage: RequestStage = STAGE


@dataclass(frozen=True, slots=True)
class FillEvent:
    raw_order_fact_id: str
    product_id: str
    source_id: str
    cursor: EventCursor
    status: str
    execution_truth: ExecutionTruth
    fill_cursor_exact: bool
    stage: RequestStage = STAGE


@dataclass(frozen=True, slots=True)
class PositionEvent:
    position_id: str
    product_id: str
    capacity_id: str
    from_state: PositionState | None
    to_state: PositionState
    cursor: EventCursor
    reason: str
    stage: RequestStage = STAGE


@dataclass(frozen=True, slots=True)
class RiskEvent:
    risk_id: str
    request_id: str
    position_id: str
    product_id: str
    risk_kind: Literal["hedge", "rollback"]
    event_type: str
    cursor: EventCursor
    status: str
    gate_reason: str | None
    arrival_reference: ArrivalReference
    stage: RequestStage = STAGE


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    position_id: str
    product_id: str
    capacity_id: str
    state: PositionState
    initiating_raw_order_fact_id: str
    hedge_intent_id: str
    rollback_request_id: str | None
    exit_resolution_state: Literal["unresolved", "flat"] | None = None
    exit_unresolved_shares: int | None = None


@dataclass(frozen=True, slots=True)
class S1ExpiryMarkSummary:
    """Minimal loop-owned provenance for one accounting expiry mark."""

    mark_id: str
    position_id: str
    product_id: str
    capacity_id: str
    capacity_release_transition_id: str
    cursor: EventCursor
    spot_close_source_id: str
    spot_close_source_cursor: EventCursor
    spot_close_price: float


@dataclass(frozen=True, slots=True)
class S1ReplayResult:
    request_events: tuple[RequestEvent, ...]
    admission_events: tuple[AdmissionEvent, ...]
    orders: tuple[SentEntryOrder, ...]
    order_events: tuple[OrderEvent, ...]
    fill_events: tuple[FillEvent, ...]
    position_events: tuple[PositionEvent, ...]
    risk_events: tuple[RiskEvent, ...]
    positions: tuple[PositionSnapshot, ...]
    executions: tuple[S1ExecutionFact, ...]
    capacity_transitions: tuple[CapacityTransition, ...]
    candidate_intent_audit: tuple[CandidateIntentAudit, ...]
    spot_requests_sent: int
    future_requests_sent: int
    approx_screen_completion_rate_20m: None = None
    mean_daily_net_twd: None = None
    performance_unavailable_reason: str = "normal_exit_route_not_integrated"
    normal_exit_enabled: bool = False
    exit_inventory_facts: tuple[ExitInventoryFact, ...] = ()
    exit_physical_fills: tuple[ExitPhysicalFill, ...] = ()
    carry_in: tuple[S1CarryPosition, ...] = ()
    carry_out: tuple[S1CarryPosition, ...] = ()
    entry_disabled_product_ids: frozenset[str] = frozenset()
    expiry_marks: tuple[S1ExpiryMarkSummary, ...] = ()
    expiry_mark_count: int = 0
    suppressed_redundant_blocked_admission_probes: int = 0


@dataclass(slots=True)
class _RequestBinding:
    kind: RequestBindingKind
    product_id: str
    request: VenueRequestIntent
    candidate_intent_id: str | None = None
    raw_order_fact_id: str | None = None
    risk_id: str | None = None
    stage: RequestStage = STAGE


@dataclass(slots=True)
class _OrderState:
    order: SentEntryOrder
    status: Literal["working", "filled", "actual_cancelled", "session_expired"]


@dataclass(slots=True)
class _Position:
    position_id: str
    product_id: str
    capacity_id: str
    initiating_raw_order_fact_id: str
    hedge_intent_id: str
    state: PositionState
    rollback_request_id: str | None = None
    frozen_exit_threshold_basis_bp: float | None = None
    entry_execution_truth: ExecutionTruth = "approximate"
    position_established_fact: PositionEstablishedFact | None = None


@dataclass(slots=True)
class _HedgeState:
    position_id: str
    request_id: str
    attempt: HedgeAttempt


@dataclass(slots=True)
class _RollbackState:
    position_id: str
    request_id: str
    attempt: RollbackAttempt


@dataclass(slots=True)
class _ExitHedgeState:
    request: ExitHedgeUnitRequest
    attempt: HedgeAttempt
    unit_notional_twd: int


@dataclass(slots=True)
class _ExitRollbackState:
    request: ExitRollbackRequest
    attempt: RollbackAttempt
    unit_notional_twd: int


@dataclass(frozen=True, slots=True)
class _ExitActivation:
    position_id: str
    activation_time_ns: int


@dataclass(frozen=True, slots=True)
class _TerminalSettlement:
    position_id: str
    terminal_outcome: Literal[
        "entry_emergency_rollback_flat",
        "exit_maker_flat",
    ]
    capacity_release_transition_id: str


@dataclass(frozen=True, slots=True)
class _ExpirySettlement:
    position_id: str
    product_id: str
    capacity_release_transition_id: str
    close: OfficialSpotClose


@dataclass(frozen=True, slots=True)
class _ScheduledFill:
    raw_order_fact_id: str
    potential: PotentialEntryFill


@dataclass(frozen=True, slots=True)
class _AssignedNew:
    binding: _RequestBinding
    observation: EntryObservation
    assignment: VenueSendAssignment


@dataclass(frozen=True, slots=True)
class _AssignedExitNew:
    binding: _RequestBinding
    target: S1SpotAskTarget
    assignment: VenueSendAssignment


@dataclass(frozen=True, slots=True)
class _AssignedRequest:
    binding: _RequestBinding
    assignment: VenueSendAssignment


@dataclass(frozen=True, slots=True)
class _BlockedAdmissionSleep:
    product_id: str
    candidate_id: str
    admission_inputs: tuple[object, ...]
    global_committed_twd: int
    product_committed_twd: int


class S1EventLoop:
    """Replay one date/policy over a shared deterministic venue/capacity clock."""

    def __init__(
        self,
        config: S1LoopConfig,
        products: Sequence[S1Product],
        *,
        entry_state_adapter: EntryStateAdapter,
        risk_book_adapter: RiskBookAdapter | None = None,
        fill_adapter: EntryFillAdapter | None = None,
        execution_adapter: ExecutionAdapter | None = None,
        accounting_adapter: AccountingSettlementAdapter | None = None,
        normal_exit_enabled: bool = False,
        spot_trade_adapter: SpotTradeAdapter | None = None,
        capacity_ledger: CapacityLedger | None = None,
        carry_in: Sequence[S1CarryPosition] = (),
        entry_enabled_product_ids: frozenset[str] | None = None,
        day_open_time_ns: int | None = None,
        suppress_redundant_blocked_admission_probes: bool = True,
    ) -> None:
        if not isinstance(config, S1LoopConfig):
            raise TypeError("config must be an S1LoopConfig")
        product_values = tuple(products)
        if not product_values:
            raise ValueError("products cannot be empty")
        if any(not isinstance(product, S1Product) for product in product_values):
            raise TypeError("products must contain only S1Product values")
        product_ids = [product.product_id for product in product_values]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("product_id values must be unique")
        carry_values = tuple(carry_in)
        if any(not isinstance(value, S1CarryPosition) for value in carry_values):
            raise TypeError("carry_in must contain only S1CarryPosition values")
        if len({value.position_id for value in carry_values}) != len(carry_values):
            raise ValueError("carry position_id values must be unique")
        if len({value.capacity_id for value in carry_values}) != len(carry_values):
            raise ValueError("carry capacity_id values must be unique")
        if carry_values and not normal_exit_enabled:
            raise ValueError("carry_in requires normal_exit_enabled")
        if carry_values and day_open_time_ns is None:
            raise ValueError("carry_in requires day_open_time_ns")
        if day_open_time_ns is not None:
            _nonnegative_int(day_open_time_ns, "day_open_time_ns")
        if not isinstance(suppress_redundant_blocked_admission_probes, bool):
            raise TypeError(
                "suppress_redundant_blocked_admission_probes must be boolean"
            )
        if entry_enabled_product_ids is not None:
            if not isinstance(entry_enabled_product_ids, frozenset) or any(
                not isinstance(value, str) or not value
                for value in entry_enabled_product_ids
            ):
                raise TypeError(
                    "entry_enabled_product_ids must be a frozenset of strings"
                )
            unknown_entry_ids = entry_enabled_product_ids.difference(product_ids)
            if unknown_entry_ids:
                raise ValueError(
                    "entry_enabled_product_ids contains unknown products: "
                    f"{sorted(unknown_entry_ids)}"
                )
        if not hasattr(entry_state_adapter, "current_state"):
            raise TypeError("entry_state_adapter must implement current_state")
        if risk_book_adapter is not None and (
            not hasattr(risk_book_adapter, "state_as_of")
            or not hasattr(risk_book_adapter, "next_change_cursor")
        ):
            raise TypeError(
                "risk_book_adapter must implement state_as_of and next_change_cursor"
            )
        if not isinstance(normal_exit_enabled, bool):
            raise TypeError("normal_exit_enabled must be boolean")
        if spot_trade_adapter is not None and (
            not hasattr(spot_trade_adapter, "next_trade_cursor")
            or not hasattr(spot_trade_adapter, "trades_at")
        ):
            raise TypeError(
                "spot_trade_adapter must implement next_trade_cursor and trades_at"
            )
        if normal_exit_enabled and spot_trade_adapter is None:
            raise ValueError("normal exit requires an exact spot_trade_adapter")
        if normal_exit_enabled and any(
            product.future_contracts != 1 for product in product_values
        ):
            raise ValueError(
                "normal exit currently requires one futures contract per position"
            )
        if accounting_adapter is not None and (
            not hasattr(accounting_adapter, "record_execution")
            or not hasattr(accounting_adapter, "establish_position")
            or not hasattr(accounting_adapter, "seal_terminal")
            or not hasattr(accounting_adapter, "record_expiry_mark")
            or not hasattr(accounting_adapter, "verify")
        ):
            raise TypeError(
                "accounting_adapter must record executions, establish positions, "
                "seal terminals, record expiry marks, and verify"
            )
        if (
            accounting_adapter is not None
            and execution_adapter is not None
            and execution_adapter is not accounting_adapter
        ):
            raise ValueError(
                "accounting_adapter and execution_adapter must be the same object"
            )

        self.config = config
        self.products = {product.product_id: product for product in product_values}
        carry_product_ids = frozenset(value.product_id for value in carry_values)
        unknown_carry_products = carry_product_ids.difference(self.products)
        if unknown_carry_products:
            raise ValueError(
                f"carry_in contains unknown products: {sorted(unknown_carry_products)}"
            )
        requested_entry_ids = (
            frozenset(product_ids)
            if entry_enabled_product_ids is None
            else entry_enabled_product_ids
        )
        self.entry_enabled_product_ids = requested_entry_ids.difference(
            carry_product_ids
        )
        self.entry_disabled_product_ids = frozenset(product_ids).difference(
            self.entry_enabled_product_ids
        )
        self.carry_in = tuple(
            sorted(
                carry_values,
                key=lambda value: (
                    value.position_established_fact.establishment_date,
                    value.position_established_fact.fifo_key,
                ),
            )
        )
        self.day_open_time_ns = day_open_time_ns
        self.entry_state_adapter = entry_state_adapter
        self.risk_book_adapter = risk_book_adapter
        self.fill_adapter = fill_adapter or NoFillAdapter()
        self.accounting_adapter = accounting_adapter
        self.execution_adapter = (
            accounting_adapter or execution_adapter or NoOpExecutionAdapter()
        )
        self.normal_exit_enabled = normal_exit_enabled
        self.spot_trade_adapter = spot_trade_adapter
        self.controllers = {
            product.product_id: S1EntryController(
                Date=config.date,
                ValueCode=product.value_code,
                QuoteCode=product.quote_code,
                route=ROUTE,
                stage=STAGE,
                maker_side="bid",
                policy_id=config.policy_id,
            )
            for product in product_values
        }
        self._books = {
            SPOT: {
                product.product_id: RawBookStateMachine() for product in product_values
            },
            FUTURE: {
                product.product_id: RawBookStateMachine() for product in product_values
            },
        }
        self._schedulers = {
            SPOT: RollingVenueScheduler(
                SPOT,
                config.spot_request_cap,
                window_ns=config.request_window_ns,
            ),
            FUTURE: RollingVenueScheduler(
                FUTURE,
                config.future_request_cap,
                window_ns=config.request_window_ns,
            ),
        }
        if capacity_ledger is not None and not isinstance(
            capacity_ledger, CapacityLedger
        ):
            raise TypeError("capacity_ledger must be a CapacityLedger or None")
        self._ledger = capacity_ledger or CapacityLedger(
            global_cap_twd=config.global_cap_twd,
            product_cap_twd=config.product_cap_twd,
        )
        if (
            self._ledger.global_cap_twd != config.global_cap_twd
            or self._ledger.product_cap_twd != config.product_cap_twd
        ):
            raise ValueError("capacity_ledger caps differ from S1LoopConfig")
        if capacity_ledger is not None and self._ledger.transitions:
            raise ValueError(
                "injected capacity_ledger must expose an empty daily transition delta"
            )
        self._validate_carry_capacity()

        self._request_bindings: dict[str, _RequestBinding] = {}
        self._orders_by_raw_id: dict[str, _OrderState] = {}
        self._sent_orders: list[SentEntryOrder] = []
        self._positions: dict[str, _Position] = {}
        self._hedges: dict[str, _HedgeState] = {}
        self._rollbacks: dict[str, _RollbackState] = {}
        self.exit_controllers = (
            {
                product.product_id: S1ExitInventoryController(
                    Date=config.date,
                    ValueCode=product.value_code,
                    QuoteCode=product.quote_code,
                    scenario_id=config.policy_id,
                )
                for product in product_values
            }
            if normal_exit_enabled
            else {}
        )
        self._exit_fill_allocators = (
            {
                product.product_id: S1SpotAskFillAllocator(
                    session_date=config.date,
                    product_id=product.product_id,
                )
                for product in product_values
            }
            if normal_exit_enabled
            else {}
        )
        self._exit_hedges: dict[str, _ExitHedgeState] = {}
        self._exit_rollbacks: dict[str, _ExitRollbackState] = {}
        self._exit_activations: dict[str, _ExitActivation] = {
            value.position_id: _ExitActivation(
                value.position_id,
                _nonnegative_int(day_open_time_ns, "day_open_time_ns"),
            )
            for value in self.carry_in
        }
        self._exit_registered_positions: set[str] = set()
        self._exit_probe_products: dict[int, set[str]] = defaultdict(set)
        self._exit_trade_probe_products: dict[int, set[str]] = defaultdict(set)
        self._exit_targets: dict[str, S1SpotAskTarget] = {}
        self._exit_order_request_ids: dict[str, str] = {}
        self._exit_allocation_execution_ids: dict[str, str] = {}
        self._exit_allocation_execution_cursors: dict[str, EventCursor] = {}
        self._exit_capacity_started: set[str] = set()
        self._exit_capacity_notional: dict[str, int] = {}
        self._exit_hedge_unit_notional: dict[str, int] = {}
        self._exit_rollback_unit_notional: dict[str, int] = {}
        self._exit_inventory_facts: list[ExitInventoryFact] = []
        self._exit_physical_fills: list[ExitPhysicalFill] = []
        self._next_position_establishment_sequence = 1 + max(
            (value.position_established_fact.sequence for value in self.carry_in),
            default=0,
        )
        self._positions_needing_settlement: list[str] = []
        self._terminal_settlements: list[_TerminalSettlement] = []
        self._expiry_settlements: list[_ExpirySettlement] = []
        self._expiry_marks: list[S1ExpiryMarkSummary] = []
        self._contract_expired_product_ids: set[str] = set()
        self._new_exit_rollbacks_at_timestamp: set[str] = set()
        self._potential_fills: dict[int, list[_ScheduledFill]] = defaultdict(list)
        self._actual_state: dict[str, EntryObservation | None] = {
            product_id: None for product_id in self.products
        }
        self._last_risk_query_cursor: dict[tuple[Venue, str], EventCursor] = {}
        self._last_risk_book_cursor: dict[tuple[Venue, str], RawBookCursor] = {}

        self._request_events: list[RequestEvent] = []
        self._admission_events: list[AdmissionEvent] = []
        self._order_events: list[OrderEvent] = []
        self._fill_events: list[FillEvent] = []
        self._position_events: list[PositionEvent] = []
        self._risk_events: list[RiskEvent] = []
        self._executions: list[S1ExecutionFact] = []

        self._timeline: list[int] = []
        self._scheduled_times: set[int] = set()
        self._new_probe_products: dict[int, set[str]] = defaultdict(set)
        self._blocked_admission_sleep: dict[str, _BlockedAdmissionSleep] = {}
        self._suppress_redundant_blocked_admission_probes = (
            suppress_redundant_blocked_admission_probes
        )
        self._suppressed_redundant_blocked_admission_probes = 0
        self._phase_rows: dict[int, int] = defaultdict(int)
        self._current_time_ns: int | None = None
        self._last_processed_time_ns: int | None = None
        self._cutoff_applied = False
        self._expiry_applied = False
        self._expiry_due = False
        self._ran = False
        for value in self.carry_in:
            fact = value.position_established_fact
            self._positions[value.position_id] = _Position(
                position_id=value.position_id,
                product_id=value.product_id,
                capacity_id=value.capacity_id,
                initiating_raw_order_fact_id=value.initiating_raw_order_fact_id,
                hedge_intent_id=value.hedge_intent_id,
                state="paired_open",
                frozen_exit_threshold_basis_bp=(value.frozen_exit_threshold_basis_bp),
                entry_execution_truth=value.execution_truth,
                position_established_fact=fact,
            )
            assert day_open_time_ns is not None
            self._exit_probe_products[day_open_time_ns].add(value.product_id)
        if self.carry_in:
            assert day_open_time_ns is not None
            self._schedule_time(day_open_time_ns)

    def run(self, events: Iterable[S1ExternalEvent]) -> S1ReplayResult:
        """Streaming-merge sorted external events with internal wake timers.

        The iterable is consumed exactly once.  It must be non-decreasing by
        :class:`EventCursor`; only the current receive-timestamp group and one
        look-ahead event are buffered, so a full raw day is never materialized
        as Python objects by this loop.
        """

        if self._ran:
            raise RuntimeError("an S1EventLoop instance can only run once")
        self._ran = True
        iterator = iter(events)
        last_external_cursor: EventCursor | None = None

        def pull() -> S1ExternalEvent | None:
            nonlocal last_external_cursor
            try:
                event = next(iterator)
            except StopIteration:
                return None
            if not isinstance(
                event,
                (
                    EntryObservation,
                    VenueBookUpdate,
                    EntryCutoff,
                    SessionExpiry,
                    ContractExpiry,
                ),
            ):
                raise TypeError("events contain an unsupported S1 event")
            self._validate_external_product(event)
            cursor = _external_cursor(event)
            if last_external_cursor is not None and cursor < last_external_cursor:
                raise ValueError("external event cursors must be non-decreasing")
            last_external_cursor = cursor
            return event

        pending_external = pull()
        while pending_external is not None or self._timeline:
            internal_time = self._peek_internal_time()
            external_time = (
                None
                if pending_external is None
                else _external_cursor(pending_external).recv_time_ns
            )
            if external_time is not None and (
                internal_time is None or external_time <= internal_time
            ):
                timestamp_ns = external_time
                group: list[S1ExternalEvent] = []
                while (
                    pending_external is not None
                    and _external_cursor(pending_external).recv_time_ns == timestamp_ns
                ):
                    group.append(pending_external)
                    pending_external = pull()
                self._consume_internal_time(timestamp_ns)
                self._process_timestamp(timestamp_ns, group)
            else:
                assert internal_time is not None
                timestamp_ns = internal_time
                self._consume_internal_time(timestamp_ns)
                self._process_timestamp(timestamp_ns, ())

        self._verify()
        if self.accounting_adapter is not None:
            self.accounting_adapter.verify()
        exit_views = {
            view.position_id: view
            for controller in self.exit_controllers.values()
            for view in controller.positions
        }
        positions = tuple(
            PositionSnapshot(
                position.position_id,
                position.product_id,
                position.capacity_id,
                position.state,
                position.initiating_raw_order_fact_id,
                position.hedge_intent_id,
                position.rollback_request_id,
                (
                    None
                    if position.position_id not in exit_views
                    else exit_views[position.position_id].resolution_state
                ),
                (
                    None
                    if position.position_id not in exit_views
                    else exit_views[position.position_id].unresolved_shares
                ),
            )
            for position in sorted(
                self._positions.values(), key=lambda value: value.position_id
            )
        )
        audit = tuple(
            audit
            for product_id in sorted(self.controllers)
            for audit in self.controllers[product_id].intent_audit
        )
        carry_out = tuple(
            carry
            for position in sorted(
                self._positions.values(),
                key=lambda value: (
                    (
                        value.position_established_fact.establishment_date
                        if value.position_established_fact is not None
                        else self.config.date
                    ),
                    (
                        value.position_established_fact.fifo_key
                        if value.position_established_fact is not None
                        else (0, value.position_id)
                    ),
                ),
            )
            if (
                carry := self._paired_carry_out(
                    position,
                    exit_views.get(position.position_id),
                )
            )
            is not None
        )
        return S1ReplayResult(
            request_events=tuple(self._request_events),
            admission_events=tuple(self._admission_events),
            orders=tuple(self._sent_orders),
            order_events=tuple(self._order_events),
            fill_events=tuple(self._fill_events),
            position_events=tuple(self._position_events),
            risk_events=tuple(self._risk_events),
            positions=positions,
            executions=tuple(self._executions),
            capacity_transitions=self._ledger.transitions,
            candidate_intent_audit=audit,
            spot_requests_sent=self._schedulers[SPOT].total_sent,
            future_requests_sent=self._schedulers[FUTURE].total_sent,
            performance_unavailable_reason=(
                "portfolio_accounting_aggregation_required"
                if self.normal_exit_enabled
                else "normal_exit_route_not_integrated"
            ),
            normal_exit_enabled=self.normal_exit_enabled,
            exit_inventory_facts=tuple(self._exit_inventory_facts),
            exit_physical_fills=tuple(self._exit_physical_fills),
            carry_in=self.carry_in,
            carry_out=carry_out,
            entry_disabled_product_ids=self.entry_disabled_product_ids,
            expiry_marks=tuple(self._expiry_marks),
            expiry_mark_count=len(self._expiry_marks),
            suppressed_redundant_blocked_admission_probes=(
                self._suppressed_redundant_blocked_admission_probes
            ),
        )

    def _process_timestamp(
        self,
        timestamp_ns: int,
        external: Sequence[S1ExternalEvent],
    ) -> None:
        if (
            self._last_processed_time_ns is not None
            and timestamp_ns <= self._last_processed_time_ns
        ):
            raise RuntimeError("loop timestamp failed to advance")
        self._current_time_ns = timestamp_ns
        self._phase_rows.clear()
        expiry_at_timestamp = self._process_external(timestamp_ns, external)
        self._expiry_due = self._expiry_due or expiry_at_timestamp
        self._refresh_actual_send_state(timestamp_ns)
        self._refresh_exit_desired(timestamp_ns)
        assigned_new, assigned_cancel, assigned_risk = self._freeze_assignments(
            timestamp_ns
        )
        self._process_fills(timestamp_ns)
        released = self._process_marketables(timestamp_ns, assigned_risk)
        self._process_entry_settlements(timestamp_ns)
        self._process_contract_expiry_settlements(timestamp_ns)
        released = self._process_cancels(timestamp_ns, assigned_cancel) or released
        if expiry_at_timestamp:
            released = self._process_expiry(timestamp_ns) or released
        self._process_new_working(timestamp_ns, assigned_new)
        new_rollbacks = self._process_deadlines(timestamp_ns)
        new_rollbacks.update(self._new_exit_rollbacks_at_timestamp)
        self._new_exit_rollbacks_at_timestamp.clear()
        if new_rollbacks:
            self._dispatch_new_rollbacks(timestamp_ns, new_rollbacks)
            self._process_terminal_settlements(
                timestamp_ns,
                PHASE_POST_ROLLBACK_TIMEOUT,
            )
            self._expire_new_rollbacks_at_deadline(timestamp_ns, new_rollbacks)
        if released and self._has_pending_entry_new():
            retry_time_ns = timestamp_ns + CAP_RETRY_DELAY_NS
            self._new_probe_products[retry_time_ns].update(
                self._pending_entry_product_ids()
            )
            self._schedule_time(retry_time_ns)
        self._schedule_token_wakes(timestamp_ns)
        self._new_probe_products.pop(timestamp_ns, None)
        self._last_processed_time_ns = timestamp_ns
        self._current_time_ns = None

    def _process_external(
        self,
        timestamp_ns: int,
        external: Sequence[S1ExternalEvent],
    ) -> bool:
        events = sorted(
            external,
            key=_external_sort_key,
        )
        expiry = False
        # ContractExpiry is the only external handler that mutates capacity,
        # and each capacity identity belongs to exactly one product.  A release
        # for an earlier product therefore cannot change the pre-expiry account
        # slice inspected by a later product.  Reuse one immutable replay per
        # timestamp; each release still validates against the live ledger.
        contract_expiry_replay: CapacityReplayResult | None = None
        for event in events:
            if isinstance(event, VenueBookUpdate):
                if self.risk_book_adapter is not None:
                    raise ValueError(
                        "VenueBookUpdate cannot be mixed with RiskBookAdapter"
                    )
                self._books[event.venue][event.product_id].ingest(event.event)
                if self.normal_exit_enabled:
                    self._exit_probe_products[timestamp_ns].add(event.product_id)
                continue
            if isinstance(event, ContractExpiry):
                contract_expiry_replay = self._process_contract_expiry(
                    event,
                    timestamp_ns,
                    replay=contract_expiry_replay,
                )
                continue
            if isinstance(event, EntryObservation):
                if (
                    event.product_id not in self.entry_enabled_product_ids
                    or event.product_id in self._contract_expired_product_ids
                ):
                    continue
                cursor = self._effect_cursor(timestamp_ns, PHASE_OBSERVE)
                commands = self.controllers[event.product_id].observe(
                    cursor,
                    event.absolute_price_tick,
                    base_gate_open=event.base_gate_open,
                    admission_open=event.admission_open,
                    gate_reason=event.gate_reason,
                )
                self._apply_commands(event.product_id, commands)
                if (
                    self.controllers[event.product_id].pending_candidate_intent_id
                    is not None
                ):
                    self._new_probe_products[timestamp_ns].add(event.product_id)
                continue
            if isinstance(event, EntryCutoff):
                if self._cutoff_applied:
                    raise ValueError("entry cutoff appears more than once")
                self._cutoff_applied = True
                for product_id in sorted(self.controllers):
                    if product_id in self._contract_expired_product_ids:
                        continue
                    cursor = self._effect_cursor(timestamp_ns, PHASE_OBSERVE)
                    commands = self.controllers[product_id].cutoff(cursor)
                    self._apply_commands(product_id, commands)
                continue
            if not isinstance(event, SessionExpiry):
                raise TypeError("unsupported external event")
            if self._expiry_applied or expiry:
                raise ValueError("session expiry appears more than once")
            expiry = True
        if contract_expiry_replay is not None:
            # Validate every release in the batch before accounting marks are
            # emitted in the settlement phase.  This keeps the verification
            # cost constant in the number of expiring products.
            self._ledger.verify()
        return expiry

    def _refresh_actual_send_state(self, timestamp_ns: int) -> None:
        if (
            self._cutoff_applied
            or self._expiry_due
            or not self._new_probe_products.get(timestamp_ns)
        ):
            return
        assignment_cursor = EventCursor(timestamp_ns, PHASE_ASSIGN, 0)
        probe_products = self._entry_probe_products_at(timestamp_ns)
        for product_id in sorted(probe_products):
            if self.controllers[product_id].pending_candidate_intent_id is None:
                continue
            state = self.entry_state_adapter.current_state(
                product_id,
                assignment_cursor,
            )
            if state is not None:
                if not isinstance(state, EntryObservation):
                    raise TypeError(
                        "current_state must return EntryObservation or None"
                    )
                if state.product_id != product_id:
                    raise ValueError("current-state product_id mismatch")
                if state.source_cursor > assignment_cursor:
                    raise ValueError("current-state observation is not causal")
                if (
                    state.maker_snapshot is not None
                    and state.maker_snapshot.value_code
                    != self.products[product_id].value_code
                ):
                    raise ValueError("current-state maker snapshot product mismatch")
                cursor = self._effect_cursor(timestamp_ns, PHASE_PRE_SEND_REFRESH)
                commands = self.controllers[product_id].observe(
                    cursor,
                    state.absolute_price_tick,
                    base_gate_open=state.base_gate_open,
                    admission_open=state.admission_open,
                    gate_reason=state.gate_reason,
                )
                self._apply_commands(product_id, commands)
            self._actual_state[product_id] = state

    def _process_contract_expiry(
        self,
        event: ContractExpiry,
        timestamp_ns: int,
        *,
        replay: CapacityReplayResult | None,
    ) -> CapacityReplayResult:
        close = event.close
        product_id = close.product_id
        if product_id in self._contract_expired_product_ids:
            raise ValueError("contract expiry appears more than once for a product")
        product = self.products[product_id]
        if close.date != self.config.date:
            raise ValueError("official close date differs from loop date")
        if product.end_date is None:
            raise ValueError("ContractExpiry requires S1Product.end_date")
        if product.end_date != self.config.date:
            raise ValueError("contract expiry does not match product end_date")
        if product.value_code != product_id:
            raise ValueError("contract expiry product differs from value_code")
        if close.quote_code != product.quote_code:
            raise ValueError("official close quote differs from S1Product")
        if close.source_cursor.recv_time_ns != timestamp_ns:
            raise ValueError("official close cursor differs from expiry timestamp")
        if self.accounting_adapter is None:
            raise ValueError("ContractExpiry requires an accounting_adapter")

        if replay is None:
            replay = self._ledger.verify()
        positions_by_capacity = {
            position.capacity_id: position
            for position in self._positions.values()
            if position.product_id == product_id
        }
        active_positions: list[_Position] = []
        for capacity_id, balances in replay.account_balances.items():
            if (
                replay.account_products[capacity_id] != product_id
                or balances.total_committed_notional_twd == 0
            ):
                continue
            position = positions_by_capacity.get(capacity_id)
            if position is None:
                raise ValueError(
                    "contract expiry found committed capacity without a position"
                )
            if (
                position.state != "paired_open"
                or balances.paired_open <= 0
                or balances.total_committed_notional_twd != balances.paired_open
            ):
                raise ValueError(
                    "contract expiry requires entirely paired-open capacity"
                )
            fact = self._required_position_fact(position.position_id)
            expected_shares = product.contract_size_shares * product.future_contracts
            if (
                fact.spot_shares != expected_shares
                or fact.short_future_contracts != product.future_contracts
                or fact.short_future_share_equivalent != expected_shares
            ):
                raise ValueError("contract expiry position is not exactly paired")
            active_positions.append(position)

        exit_controller = self.exit_controllers.get(product_id)
        if exit_controller is not None:
            views = {view.position_id: view for view in exit_controller.positions}
            for position in active_positions:
                view = views.get(position.position_id)
                fact = self._required_position_fact(position.position_id)
                if view is not None and (
                    view.available_spot_shares != fact.spot_shares
                    or view.unhedged_fill_shares
                    or view.hedge_pending_shares
                    or view.hedged_exit_shares
                    or view.rollback_pending_shares
                    or view.rollback_failed_shares
                ):
                    raise ValueError(
                        "contract expiry requires complete paired inventory"
                    )

        if not self._expiry_applied:
            cursor = self._effect_cursor(timestamp_ns, PHASE_OBSERVE)
            terminals, commands = self.controllers[product_id].session_expiry(cursor)
            if terminals:
                raise RuntimeError(
                    "paired contract expiry unexpectedly found a working entry order"
                )
            self._apply_commands(product_id, commands)
        if exit_controller is not None and not exit_controller.session_expired:
            cursor = self._effect_cursor(timestamp_ns, PHASE_OBSERVE)
            callback = exit_controller.session_expiry(cursor)
            if callback.rollback_requests:
                raise RuntimeError(
                    "paired contract expiry unexpectedly created a rollback"
                )
            self._apply_exit_inventory_fact(product_id, callback)

        self._contract_expired_product_ids.add(product_id)
        self._exit_activations = {
            position_id: activation
            for position_id, activation in self._exit_activations.items()
            if self._positions[position_id].product_id != product_id
        }
        for position in sorted(
            active_positions,
            key=lambda value: self._required_position_fact(value.position_id).fifo_key,
        ):
            cursor = self._effect_cursor(timestamp_ns, PHASE_OBSERVE)
            transition_id = (
                f"{position.capacity_id}/expiry-basis-zero/{self.config.date}"
            )
            self._ledger.complete_expiry_basis_zero(
                transition_id=transition_id,
                timestamp_ns=cursor.recv_time_ns,
                event_sequence=cursor.event_sequence,
                row_index=cursor.row_index,
                capacity_id=position.capacity_id,
            )
            previous = position.state
            position.state = "expiry_basis_zero_accounting"
            self._append_position_transition(
                position,
                previous,
                "expiry_basis_zero_accounting",
                cursor,
                "official_spot_close_basis_zero",
                stage=EXIT_STAGE,
            )
            self._expiry_settlements.append(
                _ExpirySettlement(
                    position_id=position.position_id,
                    product_id=product_id,
                    capacity_release_transition_id=transition_id,
                    close=close,
                )
            )
        return replay

    def _process_contract_expiry_settlements(self, timestamp_ns: int) -> None:
        pending = tuple(self._expiry_settlements)
        self._expiry_settlements.clear()
        adapter = self.accounting_adapter
        if pending and adapter is None:
            raise RuntimeError("contract expiry settlement lost accounting adapter")
        for settlement in pending:
            assert adapter is not None
            close = settlement.close
            cursor = self._effect_cursor(timestamp_ns, PHASE_SETTLEMENT)
            mark_id = (
                f"{settlement.position_id}/expiry-basis-zero/{self.config.date}/mark"
            )
            mark = adapter.record_expiry_mark(
                mark_id=mark_id,
                position_id=settlement.position_id,
                cursor=cursor,
                capacity_release_transition_id=(
                    settlement.capacity_release_transition_id
                ),
                spot_close_source_id=close.source_id,
                spot_close_source_cursor=close.source_cursor,
                spot_close_price=close.close_price,
            )
            if (
                getattr(mark, "mark_id", None) != mark_id
                or getattr(mark, "position_id", None) != settlement.position_id
                or getattr(mark, "capacity_release_transition_id", None)
                != settlement.capacity_release_transition_id
                or getattr(mark, "cursor", None) != cursor
                or getattr(mark, "capacity_id", None)
                != self._positions[settlement.position_id].capacity_id
                or getattr(mark, "value_code", None) != settlement.product_id
                or getattr(mark, "terminal_outcome", None)
                != "expiry_basis_zero_accounting"
                or getattr(mark, "spot_close_source_id", None) != close.source_id
                or getattr(mark, "spot_close_source_cursor", None)
                != close.source_cursor
                or not math.isclose(
                    getattr(mark, "spot_close_price", math.nan),
                    close.close_price,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or getattr(mark, "non_executable", None) is not True
                or getattr(mark, "request_id", object()) is not None
                or getattr(mark, "execution_id", object()) is not None
            ):
                raise TypeError(
                    "accounting expiry adapter returned a mismatched mark fact"
                )
            self._expiry_marks.append(
                S1ExpiryMarkSummary(
                    mark_id=mark_id,
                    position_id=settlement.position_id,
                    product_id=settlement.product_id,
                    capacity_id=self._positions[settlement.position_id].capacity_id,
                    capacity_release_transition_id=(
                        settlement.capacity_release_transition_id
                    ),
                    cursor=cursor,
                    spot_close_source_id=close.source_id,
                    spot_close_source_cursor=close.source_cursor,
                    spot_close_price=close.close_price,
                )
            )

    def _refresh_exit_desired(self, timestamp_ns: int) -> None:
        if not self.normal_exit_enabled:
            return
        product_ids = self._exit_probe_products.pop(timestamp_ns, set())
        if self._expiry_due:
            return
        for product_id in sorted(product_ids):
            controller = self.exit_controllers[product_id]
            if controller.session_expired:
                continue
            observation_cursor = self._effect_cursor(
                timestamp_ns,
                PHASE_PRE_SEND_REFRESH,
            )
            spot_book = self._risk_state(SPOT, product_id, observation_cursor)
            future_book = self._risk_state(FUTURE, product_id, observation_cursor)
            pending = sorted(
                (
                    activation
                    for activation in self._exit_activations.values()
                    if self._positions[activation.position_id].product_id == product_id
                    and activation.activation_time_ns <= timestamp_ns
                ),
                key=lambda value: (
                    self._required_position_fact(value.position_id).fifo_key
                ),
            )
            positions = controller.positions
            targets: dict[str, S1SpotAskTarget] = {}
            target_by_threshold: dict[float, S1SpotAskTarget] = {}

            for position in positions:
                targets[position.position_id] = self._cached_exit_target(
                    position.position_id,
                    observation_cursor,
                    spot_book,
                    future_book,
                    target_by_threshold,
                )
            positions_changed = False
            for activation in pending:
                target = self._cached_exit_target(
                    activation.position_id,
                    observation_cursor,
                    spot_book,
                    future_book,
                    target_by_threshold,
                )
                targets[activation.position_id] = target
                if not target.gate_open or target.absolute_price_tick is None:
                    break
                position_fact = self._required_position_fact(activation.position_id)
                add_fact = controller.add_paired_position(
                    position_fact,
                    absolute_target_tick=target.absolute_price_tick,
                    cursor=self._effect_cursor(
                        timestamp_ns,
                        PHASE_PRE_SEND_REFRESH,
                    ),
                )
                self._exit_registered_positions.add(activation.position_id)
                self._exit_activations.pop(activation.position_id, None)
                self._apply_exit_inventory_fact(product_id, add_fact)
                positions_changed = True

            if positions_changed:
                positions = controller.positions
            desired_normalized = False
            for position in positions:
                if (
                    position.desired_shares == 0
                    and position.desired_absolute_target_tick is not None
                ):
                    normalized = controller.set_position_desired(
                        position.position_id,
                        absolute_target_tick=None,
                        desired_shares=0,
                        cursor=self._effect_cursor(
                            timestamp_ns,
                            PHASE_PRE_SEND_REFRESH,
                        ),
                    )
                    self._apply_exit_inventory_fact(product_id, normalized)
                    desired_normalized = True
            if desired_normalized:
                positions = controller.positions
            if positions:
                complete_targets = tuple(
                    targets.get(position.position_id)
                    or self._cached_exit_target(
                        position.position_id,
                        observation_cursor,
                        spot_book,
                        future_book,
                        target_by_threshold,
                    )
                    for position in positions
                )
                updates = build_exit_fifo_desired_updates(
                    positions,
                    complete_targets,
                    observation_cursor=observation_cursor,
                )
                current = tuple(
                    (
                        position.position_id,
                        position.desired_absolute_target_tick,
                        position.desired_shares,
                    )
                    for position in positions
                )
                desired = tuple(
                    (
                        update.position_id,
                        update.absolute_target_tick,
                        update.desired_shares,
                    )
                    for update in updates
                )
                if current != desired:
                    desired_fact = controller.set_positions_desired(
                        updates,
                        cursor=self._effect_cursor(
                            timestamp_ns,
                            PHASE_PRE_SEND_REFRESH,
                        ),
                    )
                    self._apply_exit_inventory_fact(product_id, desired_fact)
                for target in complete_targets:
                    self._exit_targets[target.position_id] = target

            if (
                any(position.resolution_state == "unresolved" for position in positions)
                or pending
            ):
                self._schedule_next_exit_book_changes(
                    product_id,
                    observation_cursor,
                )

    def _cached_exit_target(
        self,
        position_id: str,
        cursor: EventCursor,
        spot_book: CausalBookState | None,
        future_book: CausalBookState | None,
        target_by_threshold: dict[float, S1SpotAskTarget],
    ) -> S1SpotAskTarget:
        threshold = self._positions[position_id].frozen_exit_threshold_basis_bp
        if threshold is None:
            raise RuntimeError("normal exit position lacks a frozen threshold")
        template = target_by_threshold.get(threshold)
        if template is None:
            template = self._build_exit_target(
                position_id,
                cursor,
                spot_book,
                future_book,
            )
            target_by_threshold[threshold] = template
            return template
        return replace(template, position_id=position_id)

    def _build_exit_target(
        self,
        position_id: str,
        cursor: EventCursor,
        spot_book: CausalBookState | None,
        future_book: CausalBookState | None,
    ) -> S1SpotAskTarget:
        position = self._positions[position_id]
        threshold = position.frozen_exit_threshold_basis_bp
        if threshold is None:
            raise RuntimeError("normal exit position lacks a frozen threshold")
        product = self.products[position.product_id]
        return build_s1_spot_ask_target(
            date=self.config.date,
            value_code=product.value_code,
            quote_code=product.quote_code,
            position_id=position_id,
            scenario_id=self.config.policy_id,
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=threshold,
            spot_book=spot_book,
            future_book=future_book,
            future_contracts=product.future_contracts,
        )

    def _required_position_fact(self, position_id: str) -> PositionEstablishedFact:
        fact = self._positions[position_id].position_established_fact
        if fact is None:
            raise RuntimeError("normal exit position was not established")
        return fact

    def _validate_carry_capacity(self) -> None:
        replay = self._ledger.verify()
        committed_accounts = {
            capacity_id: balances
            for capacity_id, balances in replay.account_balances.items()
            if balances.total_committed_notional_twd
        }
        carry_by_capacity = {value.capacity_id: value for value in self.carry_in}
        if set(committed_accounts) != set(carry_by_capacity):
            missing = sorted(set(committed_accounts).difference(carry_by_capacity))
            extra = sorted(set(carry_by_capacity).difference(committed_accounts))
            raise ValueError(
                "committed capacity accounts must match carry_in exactly; "
                f"missing={missing}, extra={extra}"
            )
        if self.carry_in and replay.last_timestamp_ns is not None:
            assert self.day_open_time_ns is not None
            if self.day_open_time_ns <= replay.last_timestamp_ns:
                raise ValueError(
                    "day_open_time_ns must follow the carried ledger cursor"
                )
        seen_transition_ids = set(replay.seen_transition_ids)
        historical_establishments: dict[str, PositionEstablishedFact] = {}
        if self.carry_in and self._ledger.historical_transition_identities_omitted:
            adapter_facts = getattr(self.accounting_adapter, "facts", None)
            if not isinstance(adapter_facts, tuple):
                raise ValueError(
                    "compact carry provenance requires replayed accounting facts"
                )
            assert self.accounting_adapter is not None
            self.accounting_adapter.verify()
            for value in adapter_facts:
                if not isinstance(value, PositionEstablishedFact):
                    continue
                if value.position_id in historical_establishments:
                    raise ValueError(
                        "compact carry accounting history contains duplicate "
                        "establishments"
                    )
                historical_establishments[value.position_id] = value
        initiating_raw_ids: set[str] = set()
        hedge_ids: set[str] = set()
        for carry in self.carry_in:
            fact = carry.position_established_fact
            product = self.products[carry.product_id]
            balances = committed_accounts[carry.capacity_id]
            if (
                balances.paired_open <= 0
                or balances.total_committed_notional_twd != balances.paired_open
            ):
                raise ValueError(
                    "carry capacity account must contain exactly paired_open"
                )
            if replay.account_products[carry.capacity_id] != carry.product_id:
                raise ValueError("carry product_id differs from capacity ledger")
            if fact.capacity_id != carry.capacity_id:
                raise ValueError("carry capacity_id differs from establishment fact")
            if fact.value_code != product.value_code:
                raise ValueError("carry value_code differs from S1Product")
            if carry.quote_code != product.quote_code:
                raise ValueError("carry quote_code differs from S1Product")
            if fact.scenario_id != self.config.policy_id:
                raise ValueError("carry scenario differs from loop policy")
            if fact.establishment_date >= self.config.date:
                raise ValueError(
                    "carry position must be established before the loop date"
                )
            assert self.day_open_time_ns is not None
            if fact.cursor.recv_time_ns >= self.day_open_time_ns:
                raise ValueError(
                    "carry establishment cursor must precede day_open_time_ns"
                )
            expected_shares = product.contract_size_shares * product.future_contracts
            if (
                fact.spot_shares != expected_shares
                or fact.short_future_contracts != product.future_contracts
                or fact.short_future_share_equivalent != expected_shares
            ):
                raise ValueError("carry quantities differ from S1Product contract")
            if fact.position_id != f"{carry.capacity_id}/position":
                raise ValueError("carry position_id differs from S1 identity contract")
            if fact.establishment_id != f"{fact.position_id}/established":
                raise ValueError(
                    "carry establishment_id differs from S1 identity contract"
                )
            expected_transition_id = f"{carry.capacity_id}/entry-hedge"
            if fact.capacity_transition_id != expected_transition_id:
                raise ValueError(
                    "carry capacity transition differs from S1 identity contract"
                )
            if fact.capacity_transition_id not in seen_transition_ids:
                if not self._ledger.historical_transition_identities_omitted:
                    raise ValueError(
                        "carry establishment transition is absent from capacity ledger"
                    )
                if historical_establishments.get(fact.position_id) != fact:
                    raise ValueError(
                        "compact carry establishment differs from accounting history"
                    )
            if carry.hedge_intent_id != f"{fact.position_id}/hedge":
                raise ValueError("carry hedge_intent_id differs from S1 contract")
            if carry.initiating_raw_order_fact_id in initiating_raw_ids:
                raise ValueError("carry initiating raw order identities must be unique")
            if carry.hedge_intent_id in hedge_ids:
                raise ValueError("carry hedge intent identities must be unique")
            initiating_raw_ids.add(carry.initiating_raw_order_fact_id)
            hedge_ids.add(carry.hedge_intent_id)

    def _paired_carry_out(
        self,
        position: _Position,
        exit_view: ExitPositionView | None,
    ) -> S1CarryPosition | None:
        if position.state != "paired_open":
            return None
        fact = position.position_established_fact
        threshold = position.frozen_exit_threshold_basis_bp
        if fact is None or threshold is None:
            return None
        balances = self._ledger.account_balances(position.capacity_id)
        if (
            balances.paired_open <= 0
            or balances.total_committed_notional_twd != balances.paired_open
        ):
            return None
        if exit_view is not None and (
            exit_view.available_spot_shares != fact.spot_shares
            or exit_view.unhedged_fill_shares
            or exit_view.hedge_pending_shares
            or exit_view.hedged_exit_shares
            or exit_view.rollback_pending_shares
            or exit_view.rollback_failed_shares
        ):
            return None
        product = self.products[position.product_id]
        return S1CarryPosition(
            position_established_fact=fact,
            product_id=position.product_id,
            quote_code=product.quote_code,
            capacity_id=position.capacity_id,
            initiating_raw_order_fact_id=(position.initiating_raw_order_fact_id),
            hedge_intent_id=position.hedge_intent_id,
            frozen_exit_threshold_basis_bp=threshold,
            execution_truth=position.entry_execution_truth,
        )

    def _schedule_next_exit_book_changes(
        self,
        product_id: str,
        after_cursor: EventCursor,
    ) -> None:
        adapter = self.risk_book_adapter
        if adapter is None:
            return
        product = self.products[product_id]
        for venue, deadline_ns in (
            (SPOT, product.spot_session_end_time_ns),
            (FUTURE, product.future_session_end_time_ns),
        ):
            exit_quote_change = getattr(
                adapter,
                "next_exit_quote_change_cursor",
                None,
            )
            if callable(exit_quote_change):
                candidate = exit_quote_change(
                    venue,
                    product_id,
                    after_cursor,
                    deadline_ns,
                )
                query_name = "next_exit_quote_change_cursor"
            else:
                candidate = adapter.next_change_cursor(
                    venue,
                    product_id,
                    after_cursor,
                    deadline_ns,
                )
                query_name = "next_change_cursor"
            if candidate is None:
                continue
            if not isinstance(candidate, EventCursor):
                raise TypeError(
                    f"RiskBookAdapter.{query_name} must return EventCursor or None"
                )
            if candidate <= after_cursor:
                raise ValueError("next exit-book change must follow the query cursor")
            if candidate.event_sequence >= PHASE_OBSERVE:
                raise ValueError("next exit-book change must be a raw-event cursor")
            if candidate.recv_time_ns > deadline_ns:
                raise ValueError("next exit-book change exceeds the session deadline")
            self._exit_probe_products[candidate.recv_time_ns].add(product_id)
            self._schedule_time(candidate.recv_time_ns)

    def _apply_commands(
        self,
        product_id: str,
        commands: Sequence[EntryControllerCommand],
    ) -> None:
        for command in commands:
            self._apply_command(product_id, command)

    def _apply_command(
        self,
        product_id: str,
        command: EntryControllerCommand,
    ) -> None:
        if command.kind in ("enqueue_new", "enqueue_cancel"):
            request_class: RequestClass = (
                "new" if command.kind == "enqueue_new" else "cancel"
            )
            request = VenueRequestIntent(
                request_id=command.request_id,
                venue=SPOT,
                request_class=request_class,
                original_cursor=command.cursor,
                stable_id=command.request_id,
                cutoff_drain=(
                    command.kind == "enqueue_cancel"
                    and command.reason == "entry_cutoff"
                ),
                maker_side="bid",
                absolute_price_tick=command.absolute_price_tick,
            )
            binding = _RequestBinding(
                kind="new" if command.kind == "enqueue_new" else "cancel",
                product_id=product_id,
                request=request,
                candidate_intent_id=command.candidate_intent_id,
                raw_order_fact_id=command.raw_order_fact_id,
            )
            self._schedulers[SPOT].enqueue(request)
            self._request_bindings[request.request_id] = binding
            if command.kind == "enqueue_new":
                self._new_probe_products[command.cursor.recv_time_ns].add(product_id)
            self._request_events.append(
                RequestEvent(
                    request.request_id,
                    product_id,
                    SPOT,
                    request.request_class,
                    request.risk_subtype,
                    "enqueued",
                    command.cursor,
                    request.original_cursor,
                    effect_status=command.reason,
                )
            )
            return

        binding = self._request_bindings.pop(command.request_id, None)
        self._blocked_admission_sleep.pop(command.request_id, None)
        removed = self._schedulers[SPOT].cancel_pending(command.request_id)
        if binding is not None and removed is None:
            raise RuntimeError("controller withdrew a request already assigned")
        original = (
            command.cursor if binding is None else binding.request.original_cursor
        )
        self._request_events.append(
            RequestEvent(
                command.request_id,
                product_id,
                SPOT,
                "new" if command.kind == "withdraw_pending_new" else "cancel",
                None,
                "withdrawn",
                command.cursor,
                original,
                effect_status=command.reason,
            )
        )

    def _apply_exit_inventory_fact(
        self,
        product_id: str,
        fact: ExitInventoryFact,
        *,
        rollback_specs: dict[str, RollbackSpec] | None = None,
        rollback_notionals: dict[str, int] | None = None,
    ) -> None:
        if not isinstance(fact, ExitInventoryFact):
            raise TypeError("exit controller callback must return ExitInventoryFact")
        self._exit_inventory_facts.append(fact)
        for command in fact.commands:
            self._apply_exit_command(product_id, command)
        allocator = self._exit_fill_allocators[product_id]
        active_raw_ids = set(allocator.active_raw_order_ids)
        for terminal in fact.terminals:
            if terminal.raw_order_fact_id in active_raw_ids:
                allocator.on_order_terminal(terminal)
                active_raw_ids.remove(terminal.raw_order_fact_id)
            self._exit_order_request_ids.pop(terminal.raw_order_fact_id, None)
        for request in fact.hedge_unit_requests:
            self._enqueue_exit_hedge(product_id, request)
        for request in fact.rollback_requests:
            self._enqueue_exit_rollback(
                product_id,
                request,
                spec=(
                    None
                    if rollback_specs is None
                    else rollback_specs.get(request.request_id)
                ),
                unit_notional_twd=(
                    None
                    if rollback_notionals is None
                    else rollback_notionals.get(request.request_id)
                ),
            )

    def _apply_exit_command(
        self,
        product_id: str,
        command: ExitControllerCommand,
    ) -> None:
        if command.kind in ("enqueue_new", "enqueue_cancel"):
            request_class: RequestClass = (
                "new" if command.kind == "enqueue_new" else "cancel"
            )
            request = VenueRequestIntent(
                request_id=command.request_id,
                venue=SPOT,
                request_class=request_class,
                original_cursor=command.cursor,
                stable_id=command.request_id,
                maker_side="ask",
                absolute_price_tick=command.absolute_price_tick,
            )
            binding = _RequestBinding(
                kind="new" if command.kind == "enqueue_new" else "cancel",
                product_id=product_id,
                request=request,
                candidate_intent_id=command.candidate_intent_id,
                raw_order_fact_id=command.raw_order_fact_id,
                stage=EXIT_STAGE,
            )
            self._schedulers[SPOT].enqueue(request)
            self._request_bindings[request.request_id] = binding
            self._request_events.append(
                RequestEvent(
                    request.request_id,
                    product_id,
                    SPOT,
                    request.request_class,
                    request.risk_subtype,
                    "enqueued",
                    command.cursor,
                    request.original_cursor,
                    effect_status=command.reason,
                    stage=EXIT_STAGE,
                )
            )
            probe_time_ns = command.cursor.recv_time_ns
            if command.cursor.event_sequence >= PHASE_ASSIGN:
                probe_time_ns += 1
            self._exit_probe_products[probe_time_ns].add(product_id)
            if probe_time_ns > command.cursor.recv_time_ns:
                self._schedule_time(probe_time_ns)
            return

        binding = self._request_bindings.pop(command.request_id, None)
        removed = self._schedulers[SPOT].cancel_pending(command.request_id)
        if binding is not None and removed is None:
            raise RuntimeError("exit controller withdrew an assigned request")
        original = (
            command.cursor if binding is None else binding.request.original_cursor
        )
        request_class: RequestClass = (
            "new" if command.kind == "withdraw_pending_new" else "cancel"
        )
        self._request_events.append(
            RequestEvent(
                command.request_id,
                product_id,
                SPOT,
                request_class,
                None,
                "withdrawn",
                command.cursor,
                original,
                effect_status=command.reason,
                stage=EXIT_STAGE,
            )
        )

    def _enqueue_exit_hedge(
        self,
        product_id: str,
        request: ExitHedgeUnitRequest,
    ) -> None:
        if request.request_id in self._exit_hedges:
            raise RuntimeError("duplicate exit hedge request")
        source_execution_id = self._source_execution_id(
            request.sources[0].allocation_id
        )
        trigger_cursor = max(
            self._source_execution_cursor(source.allocation_id)
            for source in request.sources
        )
        product = self.products[product_id]
        effective_hedge_end_ns = max(
            product.future_session_end_time_ns,
            trigger_cursor.recv_time_ns + HEDGE_DELAY_NS,
        )
        intent = HedgeIntent(
            hedge_intent_id=request.request_id,
            trigger_cursor=trigger_cursor,
            side="buy",
            hedge_quantity=request.contracts,
            hedge_quantity_unit="future_contracts",
            session_end_time_ns=effective_hedge_end_ns,
            initiating_first_leg_venue=SPOT,
            initiating_execution_id=source_execution_id,
            initiating_first_leg_side="sell",
            initiating_first_leg_quantity=request.spot_exit_shares,
            initiating_first_leg_quantity_unit="spot_shares",
            rollback_session_end_time_ns=max(
                product.spot_session_end_time_ns,
                effective_hedge_end_ns,
            ),
        )
        arrival = capture_arrival_reference(
            self._risk_attempt_state(FUTURE, product_id, trigger_cursor),
            reference_cursor=trigger_cursor,
            side="buy",
            quantity=request.contracts,
            quantity_unit="future_contracts",
        )
        notional = self._exit_capacity_notional[request.position_id]
        attempt = HedgeAttempt(intent, arrival_reference=arrival)
        self._exit_hedges[request.request_id] = _ExitHedgeState(
            request,
            attempt,
            notional,
        )
        venue_request = VenueRequestIntent(
            request_id=request.request_id,
            venue=FUTURE,
            request_class="exposed_risk",
            risk_subtype="hedge",
            original_cursor=EventCursor(intent.target_time_ns, PHASE_ASSIGN, 0),
            stable_id=request.request_id,
        )
        self._schedulers[FUTURE].enqueue(venue_request)
        self._request_bindings[request.request_id] = _RequestBinding(
            "hedge",
            product_id,
            venue_request,
            risk_id=request.request_id,
            stage=EXIT_STAGE,
        )
        self._request_events.append(
            RequestEvent(
                request.request_id,
                product_id,
                FUTURE,
                "exposed_risk",
                "hedge",
                "enqueued",
                trigger_cursor,
                venue_request.original_cursor,
                effect_status="exit_maker_fill_triggered",
                stage=EXIT_STAGE,
            )
        )
        self._risk_events.append(
            RiskEvent(
                request.request_id,
                request.request_id,
                request.position_id,
                product_id,
                "hedge",
                "created",
                trigger_cursor,
                attempt.result.status,
                arrival.gate_reason,
                arrival,
                EXIT_STAGE,
            )
        )
        self._schedule_time(intent.target_time_ns)
        self._schedule_time(intent.deadline_time_ns)

    def _enqueue_exit_rollback(
        self,
        product_id: str,
        request: ExitRollbackRequest,
        *,
        spec: RollbackSpec | None = None,
        unit_notional_twd: int | None = None,
    ) -> None:
        if request.request_id in self._exit_rollbacks:
            raise RuntimeError("duplicate exit rollback request")
        source_execution_id = self._source_execution_id(
            request.sources[0].allocation_id
        )
        product = self.products[product_id]
        if spec is None:
            spec = RollbackSpec(
                source_hedge_intent_id=request.request_id,
                trigger_cursor=request.trigger_cursor,
                deadline_time_ns=min(
                    request.trigger_cursor.recv_time_ns + HEDGE_RETRY_NS,
                    max(
                        product.spot_session_end_time_ns,
                        request.trigger_cursor.recv_time_ns,
                    ),
                ),
                initiating_first_leg_venue=SPOT,
                initiating_execution_id=source_execution_id,
                side="buy",
                initiating_first_leg_quantity=request.rollback_spot_shares,
                initiating_first_leg_quantity_unit="spot_shares",
            )
        if (
            spec.side != "buy"
            or spec.quantity != request.rollback_spot_shares
            or spec.quantity_unit != "spot_shares"
        ):
            raise RuntimeError("exit rollback spec changed its exact spot source")
        arrival = capture_arrival_reference(
            self._risk_attempt_state(SPOT, product_id, request.trigger_cursor),
            reference_cursor=request.trigger_cursor,
            side="buy",
            quantity=request.rollback_spot_shares,
            quantity_unit="spot_shares",
        )
        attempt = RollbackAttempt(spec, arrival_reference=arrival)
        notional = (
            self._exit_capacity_notional[request.position_id]
            if unit_notional_twd is None
            else unit_notional_twd
        )
        self._exit_rollbacks[request.request_id] = _ExitRollbackState(
            request,
            attempt,
            notional,
        )
        venue_request = VenueRequestIntent(
            request_id=request.request_id,
            venue=SPOT,
            request_class="exposed_risk",
            risk_subtype="emergency_rollback",
            original_cursor=request.trigger_cursor,
            stable_id=request.request_id,
        )
        self._schedulers[SPOT].enqueue(venue_request)
        self._request_bindings[request.request_id] = _RequestBinding(
            "rollback",
            product_id,
            venue_request,
            risk_id=request.request_id,
            stage=EXIT_STAGE,
        )
        self._request_events.append(
            RequestEvent(
                request.request_id,
                product_id,
                SPOT,
                "exposed_risk",
                "emergency_rollback",
                "enqueued",
                request.trigger_cursor,
                venue_request.original_cursor,
                effect_status=request.reason,
                stage=EXIT_STAGE,
            )
        )
        self._risk_events.append(
            RiskEvent(
                request.request_id,
                request.request_id,
                request.position_id,
                product_id,
                "rollback",
                "created",
                request.trigger_cursor,
                attempt.result.status,
                arrival.gate_reason,
                arrival,
                EXIT_STAGE,
            )
        )
        self._new_exit_rollbacks_at_timestamp.add(request.request_id)
        self._schedule_time(spec.deadline_time_ns)

    def _source_execution_id(self, allocation_id: str) -> str:
        try:
            return self._exit_allocation_execution_ids[allocation_id]
        except KeyError as error:
            raise RuntimeError(
                "exit risk request cites an unknown maker allocation"
            ) from error

    def _source_execution_cursor(self, allocation_id: str) -> EventCursor:
        try:
            return self._exit_allocation_execution_cursors[allocation_id]
        except KeyError as error:
            raise RuntimeError(
                "exit risk request cites an unknown maker execution cursor"
            ) from error

    def _freeze_assignments(
        self,
        timestamp_ns: int,
    ) -> tuple[
        tuple[_AssignedNew | _AssignedExitNew, ...],
        tuple[_AssignedRequest, ...],
        tuple[_AssignedRequest, ...],
    ]:
        spot_scheduler = self._schedulers[SPOT]
        future_scheduler = self._schedulers[FUTURE]
        new_probe_products = self._entry_probe_products_at(timestamp_ns)
        spot_risk_or_cancel = any(
            binding.request.venue == SPOT and binding.kind != "new"
            for binding in self._request_bindings.values()
        )
        dispatch_spot = spot_scheduler.pending_count > 0 and (
            spot_risk_or_cancel
            or bool(new_probe_products)
            or any(
                binding.stage == EXIT_STAGE and binding.kind == "new"
                for binding in self._request_bindings.values()
            )
        )
        dispatch_future = future_scheduler.pending_count > 0
        if not dispatch_spot and not dispatch_future:
            return (), (), ()

        assignment_cursor = EventCursor(timestamp_ns, PHASE_ASSIGN, 0)
        self._evaluate_existing_risks(assignment_cursor)
        planner = (
            CapacityAdmissionPlanner(self._ledger)
            if dispatch_spot
            and new_probe_products
            and self._has_pending_entry_new(new_probe_products)
            else None
        )
        frozen_new_state: dict[str, EntryObservation] = {}
        frozen_exit_target: dict[str, S1SpotAskTarget] = {}

        def spot_eligible(request: VenueRequestIntent) -> bool:
            binding = self._binding(request)
            if binding.kind == "cancel":
                return True
            if binding.kind == "rollback":
                rollback = (
                    self._required_rollback(binding)
                    if binding.stage == STAGE
                    else self._required_exit_rollback(binding)
                )
                return rollback.attempt.result.status == "send_eligible"
            if binding.kind != "new":
                raise RuntimeError("spot scheduler contains an invalid request kind")
            if binding.stage == EXIT_STAGE:
                if self._expiry_due:
                    return False
                candidate_id = _required_text(
                    binding.candidate_intent_id,
                    "exit new candidate_intent_id",
                )
                controller = self.exit_controllers[binding.product_id]
                pending = next(
                    (
                        value
                        for value in controller.pending_orders
                        if value.candidate_intent_id == candidate_id
                    ),
                    None,
                )
                if pending is None or pending.scheduler_state != "pending":
                    return False
                member_targets = tuple(
                    self._exit_targets.get(member.position_id)
                    for member in pending.members
                )
                if any(target is None for target in member_targets):
                    return False
                targets = tuple(
                    target for target in member_targets if target is not None
                )
                if any(
                    target.observation_cursor.recv_time_ns != timestamp_ns
                    or not target.gate_open
                    or target.absolute_price_tick != pending.absolute_price_tick
                    for target in targets
                ):
                    return False
                frozen_exit_target[request.request_id] = targets[0]
                return True
            if (
                self._cutoff_applied
                or self._expiry_due
                or binding.product_id not in new_probe_products
                or planner is None
            ):
                return False
            state = self._actual_state[binding.product_id]
            candidate_id = binding.candidate_intent_id
            if (
                state is None
                or candidate_id is None
                or state.absolute_price_tick is None
                or state.reservation_notional_twd is None
                or state.maker_snapshot is None
                or not state.base_gate_open
                or not state.admission_open
                or not self.controllers[binding.product_id].can_send_pending_at_tick(
                    candidate_id,
                    state.absolute_price_tick,
                )
            ):
                return False
            if self._blocked_admission_probe_is_redundant(binding, state):
                self._suppressed_redundant_blocked_admission_probes += 1
                return False
            plan = planner.evaluate(
                request_id=request.request_id,
                capacity_id=candidate_id,
                product_id=binding.product_id,
                requested_notional_twd=state.reservation_notional_twd,
            )
            frozen_new_state[request.request_id] = state
            return plan.admitted

        def future_eligible(request: VenueRequestIntent) -> bool:
            binding = self._binding(request)
            if binding.kind != "hedge":
                raise RuntimeError("future scheduler contains a non-hedge request")
            hedge = (
                self._required_hedge(binding)
                if binding.stage == STAGE
                else self._required_exit_hedge(binding)
            )
            return hedge.attempt.result.status == "send_eligible"

        spot_assignments = (
            spot_scheduler.dispatch(
                assignment_cursor,
                send_eligible=spot_eligible,
            )
            if dispatch_spot
            else ()
        )
        future_assignments = (
            future_scheduler.dispatch(
                assignment_cursor,
                send_eligible=future_eligible,
            )
            if dispatch_future
            else ()
        )
        for assignment in spot_assignments:
            binding = self._binding(assignment.request)
            if binding.kind == "new" and binding.stage == STAGE:
                assert planner is not None
                planner.bind_assignment(
                    assignment.request_id,
                    send_sequence=assignment.send_sequence,
                )
        committed = (
            planner.commit(
                timestamp_ns=timestamp_ns,
                event_sequence=PHASE_ASSIGN,
                first_row_index=1,
                transition_prefix=f"s1/{self.config.date}/{timestamp_ns}/admission",
            )
            if planner is not None
            else ()
        )
        for value in committed:
            transition = value.transition
            self._admission_events.append(
                AdmissionEvent(
                    value.plan.request_id,
                    value.plan.product_id,
                    value.plan.capacity_id,
                    EventCursor(
                        transition.timestamp_ns,
                        transition.event_sequence,
                        transition.row_index,
                    ),
                    value.plan.requested_notional_twd,
                    transition.status,
                    value.plan.admitted,
                )
            )
            state = frozen_new_state[value.plan.request_id]
            if value.plan.admitted:
                self._blocked_admission_sleep.pop(value.plan.request_id, None)
            elif self._suppress_redundant_blocked_admission_probes:
                self._blocked_admission_sleep[value.plan.request_id] = (
                    self._blocked_admission_sleep_value(
                        value.plan.request_id,
                        value.plan.product_id,
                        value.plan.capacity_id,
                        state,
                    )
                )

        assigned_new: list[_AssignedNew | _AssignedExitNew] = []
        assigned_cancel: list[_AssignedRequest] = []
        assigned_risk: list[_AssignedRequest] = []
        for assignment in (*spot_assignments, *future_assignments):
            binding = self._binding(assignment.request)
            effect_phase = {
                "new": PHASE_NEW_WORKING,
                "cancel": PHASE_CANCEL,
                "hedge": PHASE_MARKETABLE,
                "rollback": PHASE_MARKETABLE,
            }[binding.kind]
            effect_cursor = EventCursor(
                timestamp_ns,
                effect_phase,
                assignment.send_sequence,
            )
            self._request_events.append(
                RequestEvent(
                    assignment.request_id,
                    binding.product_id,
                    assignment.request.venue,  # type: ignore[arg-type]
                    assignment.request.request_class,
                    assignment.request.risk_subtype,
                    "assigned",
                    EventCursor(
                        timestamp_ns,
                        PHASE_ASSIGN,
                        assignment.send_sequence,
                    ),
                    assignment.request.original_cursor,
                    assignment.send_sequence,
                    "effect_frozen",
                    binding.stage,
                )
            )
            if effect_cursor < assignment.request.original_cursor:
                raise RuntimeError("request effect precedes its causal intent")
            if binding.kind == "new":
                if binding.stage == STAGE:
                    state = frozen_new_state[assignment.request_id]
                    assigned_new.append(_AssignedNew(binding, state, assignment))
                else:
                    candidate_id = _required_text(
                        binding.candidate_intent_id,
                        "exit new candidate_intent_id",
                    )
                    callback = self.exit_controllers[
                        binding.product_id
                    ].on_new_assigned(
                        candidate_id,
                        EventCursor(
                            timestamp_ns,
                            PHASE_ASSIGN,
                            assignment.send_sequence,
                        ),
                    )
                    self._apply_exit_inventory_fact(binding.product_id, callback)
                    assigned_new.append(
                        _AssignedExitNew(
                            binding,
                            frozen_exit_target[assignment.request_id],
                            assignment,
                        )
                    )
            elif binding.kind == "cancel":
                raw_id = _required_text(
                    binding.raw_order_fact_id,
                    "cancel raw_order_fact_id",
                )
                cursor = EventCursor(
                    timestamp_ns,
                    PHASE_ASSIGN,
                    assignment.send_sequence,
                )
                if binding.stage == STAGE:
                    self.controllers[binding.product_id].on_cancel_assigned(
                        raw_id,
                        cursor,
                    )
                else:
                    callback = self.exit_controllers[
                        binding.product_id
                    ].on_cancel_assigned(raw_id, cursor)
                    self._apply_exit_inventory_fact(binding.product_id, callback)
                assigned_cancel.append(_AssignedRequest(binding, assignment))
            else:
                assigned_risk.append(_AssignedRequest(binding, assignment))
        self._schedule_pending_risk_changes(
            assignment_cursor,
            exclude_request_ids={
                value.assignment.request_id for value in assigned_risk
            },
        )
        return tuple(assigned_new), tuple(assigned_cancel), tuple(assigned_risk)

    def _evaluate_existing_risks(self, cursor: EventCursor) -> None:
        for hedge_id in sorted(self._hedges):
            hedge = self._hedges[hedge_id]
            attempt = hedge.attempt
            if attempt.terminal or cursor.recv_time_ns < attempt.intent.target_time_ns:
                continue
            if cursor.recv_time_ns > attempt.intent.deadline_time_ns:
                raise RuntimeError("hedge timeline skipped its inclusive deadline")
            product_id = self._positions[hedge.position_id].product_id
            result = attempt.observe(
                self._risk_attempt_state(FUTURE, product_id, cursor),
                evaluation_cursor=cursor,
            )
            self._append_risk_evaluation(
                hedge_id,
                hedge.request_id,
                hedge.position_id,
                "hedge",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
            )
        for request_id in sorted(self._rollbacks):
            rollback = self._rollbacks[request_id]
            attempt = rollback.attempt
            if attempt.terminal or cursor < attempt.spec.trigger_cursor:
                continue
            if cursor.recv_time_ns > attempt.spec.deadline_time_ns:
                raise RuntimeError("rollback timeline skipped its inclusive deadline")
            product_id = self._positions[rollback.position_id].product_id
            result = attempt.observe(
                self._risk_attempt_state(SPOT, product_id, cursor),
                evaluation_cursor=cursor,
            )
            self._append_risk_evaluation(
                request_id,
                request_id,
                rollback.position_id,
                "rollback",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
            )
        for request_id in sorted(self._exit_hedges):
            hedge = self._exit_hedges[request_id]
            attempt = hedge.attempt
            if attempt.terminal or cursor.recv_time_ns < attempt.intent.target_time_ns:
                continue
            if cursor.recv_time_ns > attempt.intent.deadline_time_ns:
                raise RuntimeError("exit hedge timeline skipped its inclusive deadline")
            product_id = self._positions[hedge.request.position_id].product_id
            result = attempt.observe(
                self._risk_attempt_state(FUTURE, product_id, cursor),
                evaluation_cursor=cursor,
            )
            self._append_risk_evaluation(
                request_id,
                request_id,
                hedge.request.position_id,
                "hedge",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
                stage=EXIT_STAGE,
            )
        for request_id in sorted(self._exit_rollbacks):
            rollback = self._exit_rollbacks[request_id]
            attempt = rollback.attempt
            if attempt.terminal or cursor < attempt.spec.trigger_cursor:
                continue
            if cursor.recv_time_ns > attempt.spec.deadline_time_ns:
                raise RuntimeError(
                    "exit rollback timeline skipped its inclusive deadline"
                )
            product_id = self._positions[rollback.request.position_id].product_id
            result = attempt.observe(
                self._risk_attempt_state(SPOT, product_id, cursor),
                evaluation_cursor=cursor,
            )
            self._append_risk_evaluation(
                request_id,
                request_id,
                rollback.request.position_id,
                "rollback",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
                stage=EXIT_STAGE,
            )

    def _append_risk_evaluation(
        self,
        risk_id: str,
        request_id: str,
        position_id: str,
        kind: Literal["hedge", "rollback"],
        cursor: EventCursor,
        status: str,
        gate_reason: str | None,
        arrival: ArrivalReference,
        *,
        stage: RequestStage = STAGE,
    ) -> None:
        position = self._positions[position_id]
        self._risk_events.append(
            RiskEvent(
                risk_id,
                request_id,
                position_id,
                position.product_id,
                kind,
                "evaluated",
                cursor,
                status,
                gate_reason,
                arrival,
                stage,
            )
        )

    def _process_fills(self, timestamp_ns: int) -> None:
        scheduled = sorted(
            self._potential_fills.pop(timestamp_ns, ()),
            key=lambda value: value.raw_order_fact_id,
        )
        for value in scheduled:
            order_state = self._orders_by_raw_id[value.raw_order_fact_id]
            potential = value.potential
            if order_state.status != "working":
                self._fill_events.append(
                    FillEvent(
                        value.raw_order_fact_id,
                        order_state.order.product_id,
                        potential.source_id,
                        self._effect_cursor(timestamp_ns, PHASE_MAKER_FILL),
                        "suppressed_after_terminal",
                        potential.execution_truth,
                        potential.fill_cursor_exact,
                    )
                )
                continue
            cursor = self._effect_cursor(timestamp_ns, PHASE_MAKER_FILL)
            controller = self.controllers[order_state.order.product_id]
            _, commands = controller.on_fill(value.raw_order_fact_id, cursor)
            self._apply_commands(order_state.order.product_id, commands)
            order_state.status = "filled"
            order = order_state.order
            self._ledger.record_entry_full_fill(
                transition_id=f"{order.capacity_id}/entry-full-fill",
                timestamp_ns=cursor.recv_time_ns,
                event_sequence=cursor.event_sequence,
                row_index=cursor.row_index,
                capacity_id=order.capacity_id,
            )
            self._order_events.append(
                OrderEvent(
                    order.raw_order_fact_id,
                    order.candidate_intent_id,
                    order.capacity_id,
                    order.product_id,
                    "filled",
                    cursor,
                    None,
                )
            )
            self._fill_events.append(
                FillEvent(
                    order.raw_order_fact_id,
                    order.product_id,
                    potential.source_id,
                    cursor,
                    "filled",
                    potential.execution_truth,
                    potential.fill_cursor_exact,
                )
            )

            product = self.products[order.product_id]
            initiating_quantity = (
                product.contract_size_shares * product.future_contracts
            )
            entry_execution_id = f"{order.raw_order_fact_id}/entry-maker-fill"
            self._record_execution(
                S1ExecutionFact(
                    entry_execution_id,
                    f"{order.capacity_id}/position",
                    order.product_id,
                    order.capacity_id,
                    None,
                    "entry_maker",
                    SPOT,
                    "buy",
                    cursor,
                    order.target_price,
                    initiating_quantity,
                    "spot_shares",
                    potential.execution_truth,
                    potential.source_id,
                    order.maker_snapshot_recv_time_ns,
                    order.maker_snapshot_channel_seq,
                )
            )
            position_id = f"{order.capacity_id}/position"
            hedge_id = f"{position_id}/hedge"
            hedge_request_id = f"{hedge_id}/request"
            hedge_intent = HedgeIntent(
                hedge_intent_id=hedge_id,
                trigger_cursor=cursor,
                side="sell",
                hedge_quantity=product.future_contracts,
                hedge_quantity_unit="future_contracts",
                session_end_time_ns=max(
                    product.future_session_end_time_ns,
                    cursor.recv_time_ns + HEDGE_DELAY_NS,
                ),
                initiating_first_leg_venue=SPOT,
                initiating_execution_id=entry_execution_id,
                initiating_first_leg_side="buy",
                initiating_first_leg_quantity=initiating_quantity,
                initiating_first_leg_quantity_unit="spot_shares",
                rollback_session_end_time_ns=max(
                    product.spot_session_end_time_ns,
                    cursor.recv_time_ns + HEDGE_DELAY_NS,
                ),
            )
            arrival = capture_arrival_reference(
                self._risk_attempt_state(FUTURE, order.product_id, cursor),
                reference_cursor=cursor,
                side=hedge_intent.side,
                quantity=hedge_intent.hedge_quantity,
                quantity_unit=hedge_intent.hedge_quantity_unit,
            )
            attempt = HedgeAttempt(hedge_intent, arrival_reference=arrival)
            position = _Position(
                position_id,
                order.product_id,
                order.capacity_id,
                order.raw_order_fact_id,
                hedge_id,
                "hedge_pending",
                frozen_exit_threshold_basis_bp=(order.frozen_exit_threshold_basis_bp),
                entry_execution_truth=potential.execution_truth,
            )
            self._positions[position_id] = position
            self._hedges[hedge_id] = _HedgeState(
                position_id,
                hedge_request_id,
                attempt,
            )
            self._append_position_transition(
                position,
                None,
                "hedge_pending",
                cursor,
                "entry_maker_fill",
            )
            request = VenueRequestIntent(
                request_id=hedge_request_id,
                venue=FUTURE,
                request_class="exposed_risk",
                risk_subtype="hedge",
                original_cursor=EventCursor(
                    hedge_intent.target_time_ns,
                    PHASE_ASSIGN,
                    0,
                ),
                stable_id=hedge_id,
            )
            self._schedulers[FUTURE].enqueue(request)
            self._request_bindings[hedge_request_id] = _RequestBinding(
                "hedge",
                order.product_id,
                request,
                risk_id=hedge_id,
            )
            self._request_events.append(
                RequestEvent(
                    hedge_request_id,
                    order.product_id,
                    FUTURE,
                    "exposed_risk",
                    "hedge",
                    "enqueued",
                    cursor,
                    request.original_cursor,
                    effect_status="entry_fill_triggered",
                )
            )
            self._risk_events.append(
                RiskEvent(
                    hedge_id,
                    hedge_request_id,
                    position_id,
                    order.product_id,
                    "hedge",
                    "created",
                    cursor,
                    attempt.result.status,
                    arrival.gate_reason,
                    arrival,
                )
            )
            self._schedule_time(hedge_intent.target_time_ns)
            self._schedule_time(hedge_intent.deadline_time_ns)
        self._process_exit_trades(timestamp_ns)

    def _process_exit_trades(self, timestamp_ns: int) -> None:
        if not self.normal_exit_enabled:
            return
        adapter = self.spot_trade_adapter
        assert adapter is not None
        trades: list[PhysicalSpotTrade] = []
        active_products = tuple(
            product_id
            for product_id in sorted(
                self._exit_trade_probe_products.pop(timestamp_ns, set())
            )
            if self._exit_fill_allocators[product_id].active_raw_order_ids
        )
        for product_id in active_products:
            values = adapter.trades_at(product_id, timestamp_ns)
            if not isinstance(values, tuple) or any(
                not isinstance(value, PhysicalSpotTrade) for value in values
            ):
                raise TypeError(
                    "SpotTradeAdapter.trades_at must return PhysicalSpotTrade tuple"
                )
            if any(
                value.product_id != product_id
                or value.cursor.recv_time_ns != timestamp_ns
                for value in values
            ):
                raise ValueError("spot trade adapter returned the wrong product/time")
            trades.extend(values)

        for trade in sorted(trades, key=lambda value: value.cursor):
            allocator = self._exit_fill_allocators[trade.product_id]
            physical_fills = allocator.on_trade(
                TradeEvent(
                    trade.cursor,
                    absolute_price_tick(
                        trade.trade_price,
                        market="spot",
                        session_date=self.config.date,
                    ),
                    trade.quantity_shares,
                )
            )
            for physical in physical_fills:
                self._exit_physical_fills.append(physical)
                cursor = self._effect_cursor(timestamp_ns, PHASE_MAKER_FILL)
                controller = self.exit_controllers[trade.product_id]
                fact = controller.on_fill(
                    physical.raw_order_fact_id,
                    cursor,
                    physical.fill_shares,
                )
                for allocation in fact.fill_allocations:
                    self._begin_exit_capacity(allocation, cursor)
                    execution_cursor = self._effect_cursor(
                        timestamp_ns,
                        PHASE_MAKER_FILL,
                    )
                    execution_id = f"{allocation.allocation_id}/execution"
                    self._exit_allocation_execution_ids[allocation.allocation_id] = (
                        execution_id
                    )
                    self._exit_allocation_execution_cursors[
                        allocation.allocation_id
                    ] = execution_cursor
                    position = self._positions[allocation.position_id]
                    self._record_execution(
                        S1ExecutionFact(
                            execution_id=execution_id,
                            position_id=allocation.position_id,
                            product_id=trade.product_id,
                            capacity_id=position.capacity_id,
                            request_id=self._exit_order_request_ids.get(
                                physical.raw_order_fact_id
                            ),
                            role="exit_maker",
                            market=SPOT,
                            side="sell",
                            cursor=execution_cursor,
                            price=tick_index_to_price(
                                physical.target_price_tick,
                                market="spot",
                                session_date=self.config.date,
                            ),
                            quantity=allocation.allocated_shares,
                            quantity_unit="spot_shares",
                            execution_truth="exact",
                            execution_source_id=trade.source_id,
                            domain_allocation_id=allocation.allocation_id,
                        )
                    )
                self._apply_exit_inventory_fact(trade.product_id, fact)
                for order in fact.order_updates:
                    allocator.assert_working_order(order)
                self._schedule_exit_refresh_after_phase(
                    trade.product_id,
                    timestamp_ns,
                )

        for product_id in active_products:
            self._schedule_next_exit_trade(product_id, timestamp_ns)

    def _begin_exit_capacity(
        self,
        allocation: ExitFillAllocation,
        cursor: EventCursor,
    ) -> None:
        position = self._positions[allocation.position_id]
        if position.capacity_id in self._exit_capacity_started:
            return
        ledger_cursor = self._effect_cursor(
            cursor.recv_time_ns,
            PHASE_MAKER_FILL,
        )
        balances = self._ledger.account_balances(position.capacity_id)
        notional = balances.paired_open
        if notional <= 0:
            raise RuntimeError("first exit fill lacks paired-open capacity")
        self._ledger.begin_exit(
            transition_id=(
                f"{position.capacity_id}/exit-begin/{allocation.allocation_id}"
            ),
            timestamp_ns=ledger_cursor.recv_time_ns,
            event_sequence=ledger_cursor.event_sequence,
            row_index=ledger_cursor.row_index,
            capacity_id=position.capacity_id,
            notional_twd=notional,
        )
        self._exit_capacity_started.add(position.capacity_id)
        self._exit_capacity_notional[position.position_id] = notional
        previous = position.state
        position.state = "exit_in_progress"
        self._append_position_transition(
            position,
            previous,
            "exit_in_progress",
            ledger_cursor,
            "first_exit_maker_fill",
            stage=EXIT_STAGE,
        )

    def _schedule_next_exit_trade(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> None:
        allocator = self._exit_fill_allocators[product_id]
        if not allocator.active_raw_order_ids:
            return
        minimum_tick = allocator.minimum_active_target_price_tick
        if minimum_tick is None:
            raise RuntimeError("active exit orders lack a minimum target tick")
        adapter = self.spot_trade_adapter
        assert adapter is not None
        after_cursor = EventCursor(timestamp_ns, PHASE_MAKER_FILL, 2**31 - 1)
        deadline_ns = self.products[product_id].spot_session_end_time_ns
        price_aware = getattr(adapter, "next_trade_cursor_at_or_above", None)
        if callable(price_aware):
            candidate = price_aware(
                product_id,
                after_cursor,
                deadline_ns,
                tick_index_to_price(
                    minimum_tick,
                    market="spot",
                    session_date=self.config.date,
                ),
            )
        else:
            candidate = adapter.next_trade_cursor(
                product_id,
                after_cursor,
                deadline_ns,
            )
        if candidate is None:
            return
        if not isinstance(candidate, EventCursor):
            raise TypeError(
                "SpotTradeAdapter.next_trade_cursor must return EventCursor or None"
            )
        if candidate <= after_cursor:
            raise ValueError("next spot trade must follow the query cursor")
        if candidate.recv_time_ns > deadline_ns:
            raise ValueError("next spot trade exceeds the session deadline")
        self._exit_trade_probe_products[candidate.recv_time_ns].add(product_id)
        self._schedule_time(candidate.recv_time_ns)

    def _schedule_exit_refresh_after_phase(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> None:
        wake = timestamp_ns + 1
        self._exit_probe_products[wake].add(product_id)
        self._schedule_time(wake)

    def _process_marketables(
        self,
        timestamp_ns: int,
        assigned: Sequence[_AssignedRequest],
    ) -> bool:
        released = False
        for value in sorted(
            assigned,
            key=lambda item: (
                item.assignment.request.venue,
                item.assignment.send_sequence,
            ),
        ):
            binding = value.binding
            cursor = self._effect_cursor(timestamp_ns, PHASE_MARKETABLE)
            if binding.kind == "hedge":
                if binding.stage == STAGE:
                    self._complete_hedge(binding, value.assignment, cursor)
                else:
                    self._complete_exit_hedge(binding, value.assignment, cursor)
                    released = True
            elif binding.kind == "rollback":
                if binding.stage == STAGE:
                    self._complete_rollback(binding, value.assignment, cursor)
                    released = True
                else:
                    self._complete_exit_rollback(
                        binding,
                        value.assignment,
                        cursor,
                    )
            else:
                raise RuntimeError("non-risk assignment reached marketable phase")
        return released

    def _complete_hedge(
        self,
        binding: _RequestBinding,
        assignment: VenueSendAssignment,
        cursor: EventCursor,
    ) -> None:
        hedge = self._required_hedge(binding)
        position = self._positions[hedge.position_id]
        result = hedge.attempt.observe(
            self._risk_attempt_state(FUTURE, position.product_id, cursor),
            evaluation_cursor=cursor,
        )
        if result.status != "send_eligible":
            raise RuntimeError(
                "frozen hedge assignment became illegal without a raw event"
            )
        result = hedge.attempt.confirm_actual_send(cursor)
        executable = result.actual_send
        assert executable is not None
        notional = self._ledger.account_balances(position.capacity_id).hedge_pending
        self._ledger.complete_entry_hedge(
            transition_id=f"{position.capacity_id}/entry-hedge",
            timestamp_ns=cursor.recv_time_ns,
            event_sequence=cursor.event_sequence,
            row_index=cursor.row_index,
            capacity_id=position.capacity_id,
            notional_twd=notional,
        )
        previous = position.state
        position.state = "paired_open"
        self._append_position_transition(
            position,
            previous,
            "paired_open",
            cursor,
            "entry_hedge_actual_send",
        )
        self._record_execution(
            S1ExecutionFact(
                f"{hedge.attempt.intent.hedge_intent_id}/execution",
                position.position_id,
                position.product_id,
                position.capacity_id,
                assignment.request_id,
                "entry_hedge",
                FUTURE,
                hedge.attempt.intent.side,
                cursor,
                executable.executable_vwap,
                hedge.attempt.intent.hedge_quantity,
                hedge.attempt.intent.hedge_quantity_unit,
                "exact",
                _book_source_id(executable.book_cursor),
                executable.book_cursor.cursor.recv_time_ns,
                executable.book_cursor.packet_sequence,
                (
                    InitiatingExecutionAllocation(
                        hedge.attempt.intent.initiating_execution_id,
                        hedge.attempt.intent.initiating_first_leg_quantity,
                    ),
                ),
            )
        )
        self._request_actual_send(binding, assignment, cursor, "hedge_sent")
        self._risk_events.append(
            RiskEvent(
                hedge.attempt.intent.hedge_intent_id,
                assignment.request_id,
                position.position_id,
                position.product_id,
                "hedge",
                "actual_send",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
            )
        )
        self._request_bindings.pop(assignment.request_id, None)
        self._hedges.pop(hedge.attempt.intent.hedge_intent_id, None)
        if self.normal_exit_enabled:
            if position.frozen_exit_threshold_basis_bp is None:
                raise RuntimeError(
                    "normal exit requires a frozen exit threshold on entry"
                )
            self._positions_needing_settlement.append(position.position_id)

    def _process_entry_settlements(self, timestamp_ns: int) -> None:
        pending = (
            tuple(sorted(self._positions_needing_settlement))
            if self.normal_exit_enabled
            else ()
        )
        self._positions_needing_settlement.clear()
        for position_id in pending:
            position = self._positions[position_id]
            product = self.products[position.product_id]
            cursor = self._effect_cursor(timestamp_ns, PHASE_SETTLEMENT)
            share_equivalent = product.contract_size_shares * product.future_contracts
            if self.accounting_adapter is not None:
                fact = self.accounting_adapter.establish_position(
                    establishment_id=f"{position.position_id}/established",
                    position_id=position.position_id,
                    product_id=position.product_id,
                    capacity_id=position.capacity_id,
                    cursor=cursor,
                    capacity_transition_id=f"{position.capacity_id}/entry-hedge",
                    execution_truth=position.entry_execution_truth,
                )
                if not isinstance(fact, PositionEstablishedFact):
                    raise TypeError(
                        "accounting establish_position must return "
                        "PositionEstablishedFact"
                    )
            else:
                fact = PositionEstablishedFact(
                    sequence=self._next_position_establishment_sequence,
                    establishment_id=f"{position.position_id}/established",
                    position_id=position.position_id,
                    value_code=product.value_code,
                    scenario_id=self.config.policy_id,
                    capacity_id=position.capacity_id,
                    establishment_date=self.config.date,
                    cursor=cursor,
                    position_established_ns=cursor.recv_time_ns,
                    capacity_transition_id=f"{position.capacity_id}/entry-hedge",
                    spot_shares=share_equivalent,
                    short_future_contracts=product.future_contracts,
                    short_future_share_equivalent=share_equivalent,
                    execution_truth=position.entry_execution_truth,
                )
                self._next_position_establishment_sequence += 1
            position.position_established_fact = fact
            activation_time_ns = timestamp_ns + 1
            self._exit_activations[position_id] = _ExitActivation(
                position_id,
                activation_time_ns,
            )
            self._exit_probe_products[activation_time_ns].add(position.product_id)
            self._schedule_time(activation_time_ns)
        self._process_terminal_settlements(timestamp_ns, PHASE_SETTLEMENT)

    def _process_terminal_settlements(
        self,
        timestamp_ns: int,
        phase: int,
    ) -> None:
        terminals = tuple(self._terminal_settlements)
        self._terminal_settlements.clear()
        for terminal in terminals:
            adapter = self.accounting_adapter
            if adapter is None:
                raise RuntimeError("terminal settlement lost its accounting adapter")
            cursor = self._effect_cursor(timestamp_ns, phase)
            adapter.seal_terminal(
                terminal_id=(
                    f"{terminal.position_id}/{terminal.terminal_outcome}/terminal"
                ),
                position_id=terminal.position_id,
                cursor=cursor,
                terminal_outcome=terminal.terminal_outcome,
                capacity_release_transition_id=(
                    terminal.capacity_release_transition_id
                ),
            )

    def _complete_rollback(
        self,
        binding: _RequestBinding,
        assignment: VenueSendAssignment,
        cursor: EventCursor,
    ) -> None:
        rollback = self._required_rollback(binding)
        position = self._positions[rollback.position_id]
        result = rollback.attempt.observe(
            self._risk_attempt_state(SPOT, position.product_id, cursor),
            evaluation_cursor=cursor,
        )
        if result.status != "send_eligible":
            raise RuntimeError(
                "frozen rollback assignment became illegal without a raw event"
            )
        result = rollback.attempt.confirm_actual_send(cursor)
        executable = result.actual_send
        assert executable is not None
        notional = self._ledger.account_balances(position.capacity_id).hedge_pending
        self._ledger.complete_entry_rollback(
            transition_id=f"{position.capacity_id}/entry-rollback",
            timestamp_ns=cursor.recv_time_ns,
            event_sequence=cursor.event_sequence,
            row_index=cursor.row_index,
            capacity_id=position.capacity_id,
            notional_twd=notional,
        )
        previous = position.state
        position.state = "entry_emergency_rollback_flat"
        self._append_position_transition(
            position,
            previous,
            "entry_emergency_rollback_flat",
            cursor,
            "entry_rollback_actual_send",
        )
        spec = rollback.attempt.spec
        self._record_execution(
            S1ExecutionFact(
                f"{assignment.request_id}/execution",
                position.position_id,
                position.product_id,
                position.capacity_id,
                assignment.request_id,
                "entry_rollback",
                SPOT,
                spec.side,
                cursor,
                executable.executable_vwap,
                spec.quantity,
                spec.quantity_unit,
                "exact",
                _book_source_id(executable.book_cursor),
                executable.book_cursor.cursor.recv_time_ns,
                executable.book_cursor.packet_sequence,
                (
                    InitiatingExecutionAllocation(
                        spec.initiating_execution_id,
                        spec.initiating_first_leg_quantity,
                    ),
                ),
            )
        )
        self._request_actual_send(binding, assignment, cursor, "rollback_sent")
        self._risk_events.append(
            RiskEvent(
                assignment.request_id,
                assignment.request_id,
                position.position_id,
                position.product_id,
                "rollback",
                "actual_send",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
            )
        )
        self._request_bindings.pop(assignment.request_id, None)
        self._rollbacks.pop(assignment.request_id, None)
        if self.accounting_adapter is not None:
            self._terminal_settlements.append(
                _TerminalSettlement(
                    position.position_id,
                    "entry_emergency_rollback_flat",
                    f"{position.capacity_id}/entry-rollback",
                )
            )

    def _complete_exit_hedge(
        self,
        binding: _RequestBinding,
        assignment: VenueSendAssignment,
        cursor: EventCursor,
    ) -> None:
        hedge = self._required_exit_hedge(binding)
        request = hedge.request
        position = self._positions[request.position_id]
        result = hedge.attempt.observe(
            self._risk_attempt_state(FUTURE, position.product_id, cursor),
            evaluation_cursor=cursor,
        )
        if result.status != "send_eligible":
            raise RuntimeError(
                "frozen exit hedge assignment became illegal without a raw event"
            )
        result = hedge.attempt.confirm_actual_send(cursor)
        executable = result.actual_send
        assert executable is not None
        callback = self.exit_controllers[position.product_id].on_hedge_sent(
            request.request_id,
            cursor,
        )
        self._ledger.complete_exit_hedge(
            transition_id=(f"{position.capacity_id}/exit-hedge/{request.request_id}"),
            timestamp_ns=cursor.recv_time_ns,
            event_sequence=cursor.event_sequence,
            row_index=cursor.row_index,
            capacity_id=position.capacity_id,
            notional_twd=hedge.unit_notional_twd,
        )
        self._exit_capacity_started.discard(position.capacity_id)
        self._exit_capacity_notional.pop(position.position_id, None)
        previous = position.state
        position.state = "exit_maker_flat"
        self._append_position_transition(
            position,
            previous,
            "exit_maker_flat",
            cursor,
            "exit_hedge_actual_send",
            stage=EXIT_STAGE,
        )
        allocations = tuple(
            InitiatingExecutionAllocation(
                self._source_execution_id(source.allocation_id),
                source.shares,
            )
            for source in request.sources
        )
        self._record_execution(
            S1ExecutionFact(
                execution_id=f"{request.request_id}/execution",
                position_id=request.position_id,
                product_id=position.product_id,
                capacity_id=position.capacity_id,
                request_id=assignment.request_id,
                role="exit_hedge",
                market=FUTURE,
                side="buy",
                cursor=cursor,
                price=executable.executable_vwap,
                quantity=request.contracts,
                quantity_unit="future_contracts",
                execution_truth="exact",
                execution_source_id=_book_source_id(executable.book_cursor),
                book_recv_time_ns=executable.book_cursor.cursor.recv_time_ns,
                book_packet_sequence=executable.book_cursor.packet_sequence,
                initiating_execution_allocations=allocations,
            )
        )
        self._request_actual_send(
            binding,
            assignment,
            cursor,
            "exit_hedge_sent",
        )
        self._risk_events.append(
            RiskEvent(
                request.request_id,
                assignment.request_id,
                request.position_id,
                position.product_id,
                "hedge",
                "actual_send",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
                EXIT_STAGE,
            )
        )
        self._request_bindings.pop(assignment.request_id, None)
        self._exit_hedges.pop(request.request_id, None)
        if self.accounting_adapter is not None:
            self._terminal_settlements.append(
                _TerminalSettlement(
                    position.position_id,
                    "exit_maker_flat",
                    f"{position.capacity_id}/exit-hedge/{request.request_id}",
                )
            )
        self._apply_exit_inventory_fact(position.product_id, callback)
        self._schedule_exit_refresh_after_phase(
            position.product_id, cursor.recv_time_ns
        )

    def _complete_exit_rollback(
        self,
        binding: _RequestBinding,
        assignment: VenueSendAssignment,
        cursor: EventCursor,
    ) -> None:
        rollback = self._required_exit_rollback(binding)
        request = rollback.request
        position = self._positions[request.position_id]
        result = rollback.attempt.observe(
            self._risk_attempt_state(SPOT, position.product_id, cursor),
            evaluation_cursor=cursor,
        )
        if result.status != "send_eligible":
            raise RuntimeError(
                "frozen exit rollback assignment became illegal without a raw event"
            )
        result = rollback.attempt.confirm_actual_send(cursor)
        executable = result.actual_send
        assert executable is not None
        callback = self.exit_controllers[position.product_id].on_rollback_sent(
            request.request_id,
            cursor,
        )
        self._ledger.complete_exit_rollback(
            transition_id=(
                f"{position.capacity_id}/exit-rollback/{request.request_id}"
            ),
            timestamp_ns=cursor.recv_time_ns,
            event_sequence=cursor.event_sequence,
            row_index=cursor.row_index,
            capacity_id=position.capacity_id,
            notional_twd=rollback.unit_notional_twd,
        )
        self._exit_capacity_started.discard(position.capacity_id)
        self._exit_capacity_notional.pop(position.position_id, None)
        previous = position.state
        position.state = "paired_open"
        self._append_position_transition(
            position,
            previous,
            "paired_open",
            cursor,
            "exit_rollback_actual_send",
            stage=EXIT_STAGE,
        )
        allocations = tuple(
            InitiatingExecutionAllocation(
                self._source_execution_id(source.allocation_id),
                source.shares,
            )
            for source in request.sources
        )
        self._record_execution(
            S1ExecutionFact(
                execution_id=f"{request.request_id}/execution",
                position_id=request.position_id,
                product_id=position.product_id,
                capacity_id=position.capacity_id,
                request_id=assignment.request_id,
                role="exit_rollback",
                market=SPOT,
                side="buy",
                cursor=cursor,
                price=executable.executable_vwap,
                quantity=request.rollback_spot_shares,
                quantity_unit="spot_shares",
                execution_truth="exact",
                execution_source_id=_book_source_id(executable.book_cursor),
                book_recv_time_ns=executable.book_cursor.cursor.recv_time_ns,
                book_packet_sequence=executable.book_cursor.packet_sequence,
                initiating_execution_allocations=allocations,
            )
        )
        self._request_actual_send(
            binding,
            assignment,
            cursor,
            "exit_rollback_sent",
        )
        self._risk_events.append(
            RiskEvent(
                request.request_id,
                assignment.request_id,
                request.position_id,
                position.product_id,
                "rollback",
                "actual_send",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
                EXIT_STAGE,
            )
        )
        self._request_bindings.pop(assignment.request_id, None)
        self._exit_rollbacks.pop(request.request_id, None)
        self._apply_exit_inventory_fact(position.product_id, callback)
        self._schedule_exit_refresh_after_phase(
            position.product_id, cursor.recv_time_ns
        )

    def _process_cancels(
        self,
        timestamp_ns: int,
        assigned: Sequence[_AssignedRequest],
    ) -> bool:
        released = False
        for value in sorted(
            assigned,
            key=lambda item: item.assignment.send_sequence,
        ):
            binding = value.binding
            raw_id = _required_text(
                binding.raw_order_fact_id,
                "cancel raw_order_fact_id",
            )
            cursor = EventCursor(
                timestamp_ns,
                PHASE_CANCEL,
                value.assignment.send_sequence,
            )
            if binding.stage == EXIT_STAGE:
                callback = self.exit_controllers[binding.product_id].on_cancel_sent(
                    raw_id, cursor
                )
                self._request_actual_send(
                    binding,
                    value.assignment,
                    cursor,
                    (
                        "exit_cancel_noop_after_fill"
                        if not callback.terminals
                        else "exit_actual_cancelled"
                    ),
                )
                self._request_bindings.pop(value.assignment.request_id, None)
                self._apply_exit_inventory_fact(binding.product_id, callback)
                self._schedule_exit_refresh_after_phase(
                    binding.product_id,
                    timestamp_ns,
                )
                continue
            terminal, commands = self.controllers[binding.product_id].on_cancel_sent(
                raw_id,
                cursor,
            )
            state = self._orders_by_raw_id[raw_id]
            self._request_actual_send(
                binding,
                value.assignment,
                cursor,
                (
                    "cancel_noop_after_fill"
                    if terminal.terminal_reason == "filled"
                    else "actual_cancelled"
                ),
            )
            if terminal.terminal_reason == "actual_cancelled":
                state.status = "actual_cancelled"
                balances = self._ledger.account_balances(state.order.capacity_id)
                if balances.working_unfilled > 0:
                    self._ledger.release_working_leaves(
                        transition_id=f"{state.order.capacity_id}/actual-cancel",
                        timestamp_ns=cursor.recv_time_ns,
                        event_sequence=cursor.event_sequence,
                        row_index=cursor.row_index,
                        capacity_id=state.order.capacity_id,
                        reason="actual_cancel",
                    )
                    released = True
                event_type = "actual_cancelled"
            else:
                event_type = "cancel_noop_after_fill"
            self._order_events.append(
                OrderEvent(
                    raw_id,
                    state.order.candidate_intent_id,
                    state.order.capacity_id,
                    state.order.product_id,
                    event_type,
                    cursor,
                    value.assignment.request_id,
                )
            )
            self._request_bindings.pop(value.assignment.request_id, None)
            self._apply_commands(binding.product_id, commands)
        return released

    def _process_expiry(self, timestamp_ns: int) -> bool:
        if self._expiry_applied:
            raise ValueError("session expiry appears more than once")
        self._expiry_applied = True
        released = False
        for product_id in sorted(self.controllers):
            if product_id in self._contract_expired_product_ids:
                continue
            cursor = self._effect_cursor(timestamp_ns, PHASE_SESSION_EXPIRY)
            terminals, commands = self.controllers[product_id].session_expiry(cursor)
            self._apply_commands(product_id, commands)
            for terminal in terminals:
                state = self._orders_by_raw_id[terminal.raw_order_fact_id]
                state.status = "session_expired"
                balances = self._ledger.account_balances(state.order.capacity_id)
                if balances.working_unfilled > 0:
                    self._ledger.release_working_leaves(
                        transition_id=f"{state.order.capacity_id}/session-expiry",
                        timestamp_ns=cursor.recv_time_ns,
                        event_sequence=cursor.event_sequence,
                        row_index=cursor.row_index,
                        capacity_id=state.order.capacity_id,
                        reason="session_expiry",
                    )
                    released = True
                self._order_events.append(
                    OrderEvent(
                        state.order.raw_order_fact_id,
                        state.order.candidate_intent_id,
                        state.order.capacity_id,
                        product_id,
                        "session_expired",
                        cursor,
                        None,
                    )
                )
        if self.normal_exit_enabled:
            for product_id in sorted(self.exit_controllers):
                controller = self.exit_controllers[product_id]
                if controller.session_expired:
                    continue
                cursor = self._effect_cursor(timestamp_ns, PHASE_SESSION_EXPIRY)
                callback = controller.session_expiry(cursor)
                self._apply_exit_inventory_fact(product_id, callback)
        return released

    def _process_new_working(
        self,
        timestamp_ns: int,
        assigned: Sequence[_AssignedNew | _AssignedExitNew],
    ) -> None:
        if self._expiry_due and assigned:
            raise RuntimeError("new request was assigned at a session-expiry cursor")
        for value in sorted(
            assigned,
            key=lambda item: item.assignment.send_sequence,
        ):
            if isinstance(value, _AssignedExitNew):
                self._process_exit_new_working(timestamp_ns, value)
                continue
            binding = value.binding
            state = value.observation
            candidate_id = _required_text(
                binding.candidate_intent_id,
                "new candidate_intent_id",
            )
            tick = _required_positive_int(
                state.absolute_price_tick,
                "actual-send absolute_price_tick",
            )
            target_price = _required_positive_float(
                state.target_price,
                "actual-send target_price",
            )
            reservation = _required_positive_int(
                state.reservation_notional_twd,
                "actual-send reservation_notional_twd",
            )
            snapshot = state.maker_snapshot
            if snapshot is None:
                raise RuntimeError("assigned new lacks its frozen maker snapshot")
            cursor = EventCursor(
                timestamp_ns,
                PHASE_NEW_WORKING,
                value.assignment.send_sequence,
            )
            working = self.controllers[binding.product_id].on_new_sent(
                candidate_id,
                cursor,
                tick,
            )
            product = self.products[binding.product_id]
            order = SentEntryOrder(
                self.config.date,
                self.config.policy_id,
                binding.product_id,
                product.value_code,
                product.quote_code,
                value.assignment.request_id,
                candidate_id,
                working.raw_order_fact_id,
                working.policy_alias_id,
                candidate_id,
                cursor,
                tick,
                target_price,
                reservation,
                product.contract_size_shares,
                snapshot,
                state.frozen_exit_threshold_basis_bp,
            )
            order_state = _OrderState(order, "working")
            self._orders_by_raw_id[order.raw_order_fact_id] = order_state
            self._sent_orders.append(order)
            self._order_events.append(
                OrderEvent(
                    order.raw_order_fact_id,
                    candidate_id,
                    order.capacity_id,
                    order.product_id,
                    "actual_new_working",
                    cursor,
                    value.assignment.request_id,
                )
            )
            self._request_actual_send(
                binding,
                value.assignment,
                cursor,
                "new_working",
            )
            self._request_bindings.pop(value.assignment.request_id, None)
            potential = self.fill_adapter.potential_fill(order, snapshot)
            if potential is None:
                continue
            if not isinstance(potential, PotentialEntryFill):
                raise TypeError("potential_fill must return PotentialEntryFill or None")
            if potential.fill_time_ns <= cursor.recv_time_ns:
                self._fill_events.append(
                    FillEvent(
                        order.raw_order_fact_id,
                        order.product_id,
                        potential.source_id,
                        cursor,
                        "suppressed_not_after_actual_start",
                        potential.execution_truth,
                        potential.fill_cursor_exact,
                    )
                )
                continue
            self._potential_fills[potential.fill_time_ns].append(
                _ScheduledFill(order.raw_order_fact_id, potential)
            )
            self._schedule_time(potential.fill_time_ns)

    def _process_exit_new_working(
        self,
        timestamp_ns: int,
        value: _AssignedExitNew,
    ) -> None:
        binding = value.binding
        candidate_id = _required_text(
            binding.candidate_intent_id,
            "exit new candidate_intent_id",
        )
        target_tick = _required_positive_int(
            value.target.absolute_price_tick,
            "exit actual-send target tick",
        )
        cursor = EventCursor(
            timestamp_ns,
            PHASE_NEW_WORKING,
            value.assignment.send_sequence,
        )
        callback = self.exit_controllers[binding.product_id].on_new_sent(
            candidate_id,
            cursor,
            target_tick,
        )
        if len(callback.order_updates) != 1:
            raise RuntimeError("exit actual new must create one aggregate order")
        order = callback.order_updates[0]
        self._exit_fill_allocators[binding.product_id].register_order(
            order,
            initial_queue_ahead_shares=value.target.initial_queue_ahead_shares,
        )
        self._exit_order_request_ids[order.raw_order_fact_id] = (
            value.assignment.request_id
        )
        self._order_events.append(
            OrderEvent(
                order.raw_order_fact_id,
                order.candidate_intent_id,
                self._positions[order.members[0].position_id].capacity_id,
                binding.product_id,
                "actual_new_working",
                cursor,
                value.assignment.request_id,
                EXIT_STAGE,
            )
        )
        self._request_actual_send(
            binding,
            value.assignment,
            cursor,
            "exit_new_working",
        )
        self._request_bindings.pop(value.assignment.request_id, None)
        self._apply_exit_inventory_fact(binding.product_id, callback)
        self._schedule_next_exit_trade(binding.product_id, timestamp_ns)

    def _process_deadlines(self, timestamp_ns: int) -> set[str]:
        self._expire_existing_rollbacks(timestamp_ns, PHASE_RISK_TIMEOUT)
        self._expire_existing_exit_rollbacks(timestamp_ns, PHASE_RISK_TIMEOUT)
        created: set[str] = set()
        expired_hedges: list[str] = []
        for hedge_id in sorted(self._hedges):
            hedge = self._hedges[hedge_id]
            attempt = hedge.attempt
            if attempt.terminal or attempt.intent.deadline_time_ns != timestamp_ns:
                continue
            timeout_cursor = self._effect_cursor(
                timestamp_ns,
                PHASE_RISK_TIMEOUT,
            )
            result = attempt.expire(timeout_cursor)
            spec = result.rollback
            assert spec is not None
            self._cancel_risk_pending(
                hedge.request_id,
                timeout_cursor,
                "hedge_retry_timeout",
            )
            position = self._positions[hedge.position_id]
            previous = position.state
            position.state = "entry_rollback_pending"
            rollback_request_id = f"{hedge_id}/rollback/request"
            position.rollback_request_id = rollback_request_id
            self._append_position_transition(
                position,
                previous,
                "entry_rollback_pending",
                timeout_cursor,
                "hedge_retry_timeout",
            )
            rollback_arrival = capture_arrival_reference(
                self._risk_attempt_state(SPOT, position.product_id, timeout_cursor),
                reference_cursor=spec.trigger_cursor,
                side=spec.side,
                quantity=spec.quantity,
                quantity_unit=spec.quantity_unit,
            )
            rollback = RollbackAttempt(spec, arrival_reference=rollback_arrival)
            self._rollbacks[rollback_request_id] = _RollbackState(
                position.position_id,
                rollback_request_id,
                rollback,
            )
            request = VenueRequestIntent(
                request_id=rollback_request_id,
                venue=SPOT,
                request_class="exposed_risk",
                risk_subtype="emergency_rollback",
                original_cursor=spec.trigger_cursor,
                stable_id=rollback_request_id,
            )
            self._schedulers[SPOT].enqueue(request)
            self._request_bindings[rollback_request_id] = _RequestBinding(
                "rollback",
                position.product_id,
                request,
                risk_id=rollback_request_id,
            )
            self._request_events.append(
                RequestEvent(
                    rollback_request_id,
                    position.product_id,
                    SPOT,
                    "exposed_risk",
                    "emergency_rollback",
                    "enqueued",
                    timeout_cursor,
                    request.original_cursor,
                    effect_status="hedge_retry_timeout",
                )
            )
            self._risk_events.append(
                RiskEvent(
                    hedge_id,
                    hedge.request_id,
                    position.position_id,
                    position.product_id,
                    "hedge",
                    "timeout",
                    timeout_cursor,
                    result.status,
                    result.last_gate_reason,
                    result.arrival_reference,
                )
            )
            self._risk_events.append(
                RiskEvent(
                    rollback_request_id,
                    rollback_request_id,
                    position.position_id,
                    position.product_id,
                    "rollback",
                    "created",
                    timeout_cursor,
                    rollback.result.status,
                    rollback_arrival.gate_reason,
                    rollback_arrival,
                )
            )
            created.add(rollback_request_id)
            expired_hedges.append(hedge_id)
            self._schedule_time(spec.deadline_time_ns)
        for hedge_id in expired_hedges:
            self._hedges.pop(hedge_id, None)
        expired_exit_hedges: list[str] = []
        for request_id in sorted(self._exit_hedges):
            hedge = self._exit_hedges[request_id]
            attempt = hedge.attempt
            if attempt.terminal or attempt.intent.deadline_time_ns != timestamp_ns:
                continue
            timeout_cursor = self._effect_cursor(
                timestamp_ns,
                PHASE_RISK_TIMEOUT,
            )
            result = attempt.expire(timeout_cursor)
            spec = result.rollback
            assert spec is not None
            self._cancel_risk_pending(
                request_id,
                timeout_cursor,
                "exit_hedge_retry_timeout",
            )
            position = self._positions[hedge.request.position_id]
            callback = self.exit_controllers[position.product_id].on_hedge_timeout(
                request_id,
                timeout_cursor,
            )
            if len(callback.rollback_requests) != 1:
                raise RuntimeError("exit hedge timeout must create one exact rollback")
            rollback_request = callback.rollback_requests[0]
            previous = position.state
            position.state = "exit_rollback_pending"
            self._append_position_transition(
                position,
                previous,
                "exit_rollback_pending",
                timeout_cursor,
                "exit_hedge_retry_timeout",
                stage=EXIT_STAGE,
            )
            self._risk_events.append(
                RiskEvent(
                    request_id,
                    request_id,
                    position.position_id,
                    position.product_id,
                    "hedge",
                    "timeout",
                    timeout_cursor,
                    result.status,
                    result.last_gate_reason,
                    result.arrival_reference,
                    EXIT_STAGE,
                )
            )
            self._apply_exit_inventory_fact(
                position.product_id,
                callback,
                rollback_specs={rollback_request.request_id: spec},
                rollback_notionals={
                    rollback_request.request_id: hedge.unit_notional_twd
                },
            )
            created.add(rollback_request.request_id)
            expired_exit_hedges.append(request_id)
        for request_id in expired_exit_hedges:
            self._exit_hedges.pop(request_id, None)
        return created

    def _dispatch_new_rollbacks(
        self,
        timestamp_ns: int,
        request_ids: set[str],
    ) -> None:
        cursor = EventCursor(timestamp_ns, PHASE_ROLLBACK_ASSIGN, 0)
        for request_id in sorted(request_ids):
            binding = self._request_bindings[request_id]
            if binding.stage == STAGE:
                state = self._rollbacks[request_id]
                position_id = state.position_id
            else:
                state = self._exit_rollbacks[request_id]
                position_id = state.request.position_id
            product_id = self._positions[position_id].product_id
            result = state.attempt.observe(
                self._risk_attempt_state(SPOT, product_id, cursor),
                evaluation_cursor=cursor,
            )
            self._append_risk_evaluation(
                request_id,
                request_id,
                position_id,
                "rollback",
                cursor,
                result.status,
                result.last_gate_reason,
                result.arrival_reference,
                stage=binding.stage,
            )

        assignments = self._schedulers[SPOT].dispatch(
            cursor,
            send_eligible=lambda request: (
                request.request_id in request_ids
                and (
                    self._required_rollback(self._binding(request)).attempt
                    if self._binding(request).stage == STAGE
                    else self._required_exit_rollback(self._binding(request)).attempt
                ).result.status
                == "send_eligible"
            ),
        )
        assigned: list[_AssignedRequest] = []
        for assignment in assignments:
            binding = self._binding(assignment.request)
            self._request_events.append(
                RequestEvent(
                    assignment.request_id,
                    binding.product_id,
                    SPOT,
                    "exposed_risk",
                    "emergency_rollback",
                    "assigned",
                    EventCursor(
                        timestamp_ns,
                        PHASE_ROLLBACK_ASSIGN,
                        assignment.send_sequence,
                    ),
                    assignment.request.original_cursor,
                    assignment.send_sequence,
                    "same_timestamp_emergency",
                    binding.stage,
                )
            )
            assigned.append(_AssignedRequest(binding, assignment))
        self._schedule_pending_risk_changes(
            cursor,
            exclude_request_ids={value.assignment.request_id for value in assigned},
            only_request_ids=request_ids,
        )
        for value in assigned:
            cursor = self._effect_cursor(
                timestamp_ns,
                PHASE_ROLLBACK_EXECUTION,
            )
            if value.binding.stage == STAGE:
                self._complete_rollback(value.binding, value.assignment, cursor)
            else:
                self._complete_exit_rollback(
                    value.binding,
                    value.assignment,
                    cursor,
                )

    def _expire_existing_rollbacks(
        self,
        timestamp_ns: int,
        phase: int,
        *,
        only: set[str] | None = None,
    ) -> None:
        expired: list[str] = []
        for request_id in sorted(self._rollbacks):
            if only is not None and request_id not in only:
                continue
            rollback = self._rollbacks[request_id]
            attempt = rollback.attempt
            if attempt.terminal or attempt.spec.deadline_time_ns != timestamp_ns:
                continue
            cursor = self._effect_cursor(timestamp_ns, phase)
            result = attempt.expire(cursor)
            self._cancel_risk_pending(
                request_id,
                cursor,
                "rollback_retry_timeout",
            )
            position = self._positions[rollback.position_id]
            notional = self._ledger.account_balances(position.capacity_id).hedge_pending
            self._ledger.record_entry_rollback_failure(
                transition_id=f"{position.capacity_id}/entry-rollback-failed",
                timestamp_ns=cursor.recv_time_ns,
                event_sequence=cursor.event_sequence,
                row_index=cursor.row_index,
                capacity_id=position.capacity_id,
                notional_twd=notional,
            )
            previous = position.state
            position.state = "entry_hedge_timeout_unresolved"
            self._append_position_transition(
                position,
                previous,
                "entry_hedge_timeout_unresolved",
                cursor,
                "rollback_retry_timeout",
            )
            self._risk_events.append(
                RiskEvent(
                    request_id,
                    request_id,
                    position.position_id,
                    position.product_id,
                    "rollback",
                    "timeout",
                    cursor,
                    result.status,
                    result.last_gate_reason,
                    result.arrival_reference,
                )
            )
            expired.append(request_id)
        for request_id in expired:
            self._rollbacks.pop(request_id, None)

    def _expire_existing_exit_rollbacks(
        self,
        timestamp_ns: int,
        phase: int,
        *,
        only: set[str] | None = None,
    ) -> None:
        expired: list[str] = []
        for request_id in sorted(self._exit_rollbacks):
            if only is not None and request_id not in only:
                continue
            rollback = self._exit_rollbacks[request_id]
            attempt = rollback.attempt
            if attempt.terminal or attempt.spec.deadline_time_ns != timestamp_ns:
                continue
            cursor = self._effect_cursor(timestamp_ns, phase)
            result = attempt.expire(cursor)
            self._cancel_risk_pending(
                request_id,
                cursor,
                "exit_rollback_retry_timeout",
            )
            request = rollback.request
            position = self._positions[request.position_id]
            callback = self.exit_controllers[position.product_id].on_rollback_failed(
                request_id,
                cursor,
            )
            self._ledger.record_exit_rollback_failure(
                transition_id=(
                    f"{position.capacity_id}/exit-rollback-failed/{request_id}"
                ),
                timestamp_ns=cursor.recv_time_ns,
                event_sequence=cursor.event_sequence,
                row_index=cursor.row_index,
                capacity_id=position.capacity_id,
                notional_twd=rollback.unit_notional_twd,
            )
            previous = position.state
            position.state = "exit_rollback_failed_unresolved"
            self._append_position_transition(
                position,
                previous,
                "exit_rollback_failed_unresolved",
                cursor,
                "exit_rollback_retry_timeout",
                stage=EXIT_STAGE,
            )
            self._risk_events.append(
                RiskEvent(
                    request_id,
                    request_id,
                    request.position_id,
                    position.product_id,
                    "rollback",
                    "timeout",
                    cursor,
                    result.status,
                    result.last_gate_reason,
                    result.arrival_reference,
                    EXIT_STAGE,
                )
            )
            self._apply_exit_inventory_fact(position.product_id, callback)
            expired.append(request_id)
        for request_id in expired:
            self._exit_rollbacks.pop(request_id, None)

    def _expire_new_rollbacks_at_deadline(
        self,
        timestamp_ns: int,
        request_ids: set[str],
    ) -> None:
        at_deadline = {
            request_id
            for request_id in request_ids
            if request_id in self._rollbacks
            and self._rollbacks[request_id].attempt.spec.deadline_time_ns
            == timestamp_ns
        }
        if at_deadline:
            self._expire_existing_rollbacks(
                timestamp_ns,
                PHASE_POST_ROLLBACK_TIMEOUT,
                only=at_deadline,
            )
        exit_at_deadline = {
            request_id
            for request_id in request_ids
            if request_id in self._exit_rollbacks
            and self._exit_rollbacks[request_id].attempt.spec.deadline_time_ns
            == timestamp_ns
        }
        if exit_at_deadline:
            self._expire_existing_exit_rollbacks(
                timestamp_ns,
                PHASE_POST_ROLLBACK_TIMEOUT,
                only=exit_at_deadline,
            )

    def _cancel_risk_pending(
        self,
        request_id: str,
        cursor: EventCursor,
        reason: str,
    ) -> None:
        binding = self._request_bindings.pop(request_id, None)
        if binding is None:
            raise RuntimeError("terminal risk request has no pending binding")
        removed = self._schedulers[binding.request.venue].cancel_pending(request_id)
        if removed is None:
            raise RuntimeError("terminal risk request was already assigned")
        self._request_events.append(
            RequestEvent(
                request_id,
                binding.product_id,
                binding.request.venue,  # type: ignore[arg-type]
                binding.request.request_class,
                binding.request.risk_subtype,
                "withdrawn",
                cursor,
                binding.request.original_cursor,
                effect_status=reason,
                stage=binding.stage,
            )
        )

    def _request_actual_send(
        self,
        binding: _RequestBinding,
        assignment: VenueSendAssignment,
        cursor: EventCursor,
        status: str,
    ) -> None:
        self._request_events.append(
            RequestEvent(
                assignment.request_id,
                binding.product_id,
                assignment.request.venue,  # type: ignore[arg-type]
                assignment.request.request_class,
                assignment.request.risk_subtype,
                "actual_send",
                cursor,
                assignment.request.original_cursor,
                assignment.send_sequence,
                status,
                binding.stage,
            )
        )

    def _append_position_transition(
        self,
        position: _Position,
        previous: PositionState | None,
        current: PositionState,
        cursor: EventCursor,
        reason: str,
        *,
        stage: RequestStage = STAGE,
    ) -> None:
        self._position_events.append(
            PositionEvent(
                position.position_id,
                position.product_id,
                position.capacity_id,
                previous,
                current,
                cursor,
                reason,
                stage,
            )
        )

    def _record_execution(self, fact: S1ExecutionFact) -> None:
        self.execution_adapter.record_execution(fact)
        self._executions.append(fact)

    def _binding(self, request: VenueRequestIntent) -> _RequestBinding:
        try:
            return self._request_bindings[request.request_id]
        except KeyError as error:
            raise RuntimeError("scheduler request has no domain binding") from error

    def _required_hedge(self, binding: _RequestBinding) -> _HedgeState:
        risk_id = _required_text(binding.risk_id, "hedge risk_id")
        try:
            return self._hedges[risk_id]
        except KeyError as error:
            raise RuntimeError("hedge binding has no attempt") from error

    def _required_rollback(self, binding: _RequestBinding) -> _RollbackState:
        risk_id = _required_text(binding.risk_id, "rollback risk_id")
        try:
            return self._rollbacks[risk_id]
        except KeyError as error:
            raise RuntimeError("rollback binding has no attempt") from error

    def _required_exit_hedge(self, binding: _RequestBinding) -> _ExitHedgeState:
        risk_id = _required_text(binding.risk_id, "exit hedge risk_id")
        try:
            return self._exit_hedges[risk_id]
        except KeyError as error:
            raise RuntimeError("exit hedge binding has no attempt") from error

    def _required_exit_rollback(
        self,
        binding: _RequestBinding,
    ) -> _ExitRollbackState:
        risk_id = _required_text(binding.risk_id, "exit rollback risk_id")
        try:
            return self._exit_rollbacks[risk_id]
        except KeyError as error:
            raise RuntimeError("exit rollback binding has no attempt") from error

    def _risk_state(
        self,
        venue: Venue,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        adapter = self.risk_book_adapter
        if adapter is None:
            return self._books[venue][product_id].state
        state = adapter.state_as_of(venue, product_id, cursor)
        if state is not None and not isinstance(state, CausalBookState):
            raise TypeError(
                "RiskBookAdapter.state_as_of must return CausalBookState or None"
            )
        if state is not None and state.book_cursor.cursor > cursor:
            raise ValueError("risk book state is not causal at its query cursor")
        key = (venue, product_id)
        previous_query = self._last_risk_query_cursor.get(key)
        previous_book = self._last_risk_book_cursor.get(key)
        if previous_query is not None and cursor < previous_query:
            raise ValueError("risk book query cursor regressed")
        if previous_query == cursor and previous_book is None and state is not None:
            raise ValueError("RiskBookAdapter changed an identical as-of query")
        if previous_book is not None:
            if state is None or state.book_cursor < previous_book:
                raise ValueError("RiskBookAdapter state regressed")
            if cursor == previous_query and state.book_cursor != previous_book:
                raise ValueError("RiskBookAdapter changed an identical as-of query")
        self._last_risk_query_cursor[key] = cursor
        if state is not None:
            self._last_risk_book_cursor[key] = state.book_cursor
        return state

    def _risk_attempt_state(
        self,
        venue: Venue,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        product = self.products[product_id]
        session_end_ns = (
            product.spot_session_end_time_ns
            if venue == SPOT
            else product.future_session_end_time_ns
        )
        if cursor.recv_time_ns > session_end_ns:
            return None
        return self._risk_state(venue, product_id, cursor)

    def _schedule_pending_risk_changes(
        self,
        after_cursor: EventCursor,
        *,
        exclude_request_ids: set[str],
        only_request_ids: set[str] | None = None,
    ) -> None:
        if self.risk_book_adapter is None:
            return
        for hedge in self._hedges.values():
            if hedge.request_id in exclude_request_ids or (
                only_request_ids is not None
                and hedge.request_id not in only_request_ids
            ):
                continue
            result = hedge.attempt.result
            if hedge.attempt.terminal or not result.target_evaluated:
                continue
            product_id = self._positions[hedge.position_id].product_id
            self._schedule_next_risk_change(
                FUTURE,
                product_id,
                after_cursor,
                hedge.attempt.intent.deadline_time_ns,
            )
        for rollback in self._rollbacks.values():
            if rollback.request_id in exclude_request_ids or (
                only_request_ids is not None
                and rollback.request_id not in only_request_ids
            ):
                continue
            result = rollback.attempt.result
            if rollback.attempt.terminal or not result.target_evaluated:
                continue
            product_id = self._positions[rollback.position_id].product_id
            self._schedule_next_risk_change(
                SPOT,
                product_id,
                after_cursor,
                rollback.attempt.spec.deadline_time_ns,
            )
        for hedge in self._exit_hedges.values():
            if hedge.request.request_id in exclude_request_ids or (
                only_request_ids is not None
                and hedge.request.request_id not in only_request_ids
            ):
                continue
            result = hedge.attempt.result
            if hedge.attempt.terminal or not result.target_evaluated:
                continue
            product_id = self._positions[hedge.request.position_id].product_id
            self._schedule_next_risk_change(
                FUTURE,
                product_id,
                after_cursor,
                hedge.attempt.intent.deadline_time_ns,
            )
        for rollback in self._exit_rollbacks.values():
            if rollback.request.request_id in exclude_request_ids or (
                only_request_ids is not None
                and rollback.request.request_id not in only_request_ids
            ):
                continue
            result = rollback.attempt.result
            if rollback.attempt.terminal or not result.target_evaluated:
                continue
            product_id = self._positions[rollback.request.position_id].product_id
            self._schedule_next_risk_change(
                SPOT,
                product_id,
                after_cursor,
                rollback.attempt.spec.deadline_time_ns,
            )

    def _schedule_next_risk_change(
        self,
        venue: Venue,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> None:
        adapter = self.risk_book_adapter
        assert adapter is not None
        candidate = adapter.next_change_cursor(
            venue,
            product_id,
            after_cursor,
            deadline_ns,
        )
        if candidate is None:
            return
        if not isinstance(candidate, EventCursor):
            raise TypeError(
                "RiskBookAdapter.next_change_cursor must return EventCursor or None"
            )
        if candidate <= after_cursor:
            raise ValueError("next risk-book change must follow the query cursor")
        if candidate.event_sequence >= PHASE_OBSERVE:
            raise ValueError("next risk-book change must be a raw-event cursor")
        if candidate.recv_time_ns > deadline_ns:
            raise ValueError("next risk-book change exceeds the attempt deadline")
        if self.normal_exit_enabled:
            self._exit_probe_products[candidate.recv_time_ns].add(product_id)
        self._schedule_time(candidate.recv_time_ns)

    def _schedule_token_wakes(self, timestamp_ns: int) -> None:
        for venue, scheduler in self._schedulers.items():
            wake = scheduler.next_token_time_ns()
            if wake is not None and wake > timestamp_ns:
                pending_entry_products = (
                    self._pending_entry_product_ids() if venue == SPOT else frozenset()
                )
                has_non_entry_spot = venue == SPOT and any(
                    not (binding.kind == "new" and binding.stage == STAGE)
                    for binding in self._request_bindings.values()
                    if binding.request.venue == SPOT
                )
                if (
                    venue == SPOT
                    and not pending_entry_products
                    and not has_non_entry_spot
                ):
                    continue
                self._schedule_time(wake)
                if pending_entry_products:
                    self._new_probe_products[wake].update(pending_entry_products)
                if venue == SPOT and self.normal_exit_enabled:
                    for binding in self._request_bindings.values():
                        if binding.stage == EXIT_STAGE and binding.kind == "new":
                            self._exit_probe_products[wake].add(binding.product_id)

    def _has_pending_entry_new(
        self,
        product_ids: set[str] | frozenset[str] | None = None,
    ) -> bool:
        return any(
            binding.kind == "new"
            and binding.stage == STAGE
            and (product_ids is None or binding.product_id in product_ids)
            for binding in self._request_bindings.values()
        )

    def _entry_probe_products_at(self, timestamp_ns: int) -> set[str]:
        triggered = self._new_probe_products.get(timestamp_ns)
        if not triggered:
            return set()
        # Any entry probe is a venue-wide scheduler event.  Even a request whose
        # last capacity decision was blocked must participate in current-state
        # refresh and priority traversal; only spot_eligible may suppress the
        # duplicate planner audit after proving its complete signature unchanged.
        return set(self._pending_entry_product_ids())

    def _pending_entry_product_ids(self) -> frozenset[str]:
        return frozenset(
            binding.product_id
            for binding in self._request_bindings.values()
            if binding.kind == "new" and binding.stage == STAGE
        )

    @staticmethod
    def _admission_inputs_signature(state: EntryObservation) -> tuple[object, ...]:
        return (
            state.absolute_price_tick,
            state.target_price,
            state.reservation_notional_twd,
            state.base_gate_open,
            state.admission_open,
            state.gate_reason,
            state.maker_snapshot,
            state.frozen_exit_threshold_basis_bp,
        )

    def _blocked_admission_sleep_value(
        self,
        request_id: str,
        product_id: str,
        candidate_id: str,
        state: EntryObservation,
    ) -> _BlockedAdmissionSleep:
        del request_id
        return _BlockedAdmissionSleep(
            product_id=product_id,
            candidate_id=candidate_id,
            admission_inputs=self._admission_inputs_signature(state),
            global_committed_twd=(
                self._ledger.global_balances.total_committed_notional_twd
            ),
            product_committed_twd=(
                self._ledger.product_balances(product_id).total_committed_notional_twd
            ),
        )

    def _blocked_admission_probe_is_redundant(
        self,
        binding: _RequestBinding,
        state: EntryObservation,
    ) -> bool:
        if not self._suppress_redundant_blocked_admission_probes:
            return False
        sleep = self._blocked_admission_sleep.get(binding.request.request_id)
        candidate_id = binding.candidate_intent_id
        return (
            sleep is not None
            and candidate_id is not None
            and sleep.product_id == binding.product_id
            and sleep.candidate_id == candidate_id
            and sleep.admission_inputs == self._admission_inputs_signature(state)
            and sleep.global_committed_twd
            == self._ledger.global_balances.total_committed_notional_twd
            and sleep.product_committed_twd
            == self._ledger.product_balances(
                binding.product_id
            ).total_committed_notional_twd
        )

    def _schedule_time(self, timestamp_ns: int) -> None:
        _nonnegative_int(timestamp_ns, "timeline timestamp")
        if self._current_time_ns is not None and timestamp_ns <= self._current_time_ns:
            return
        if (
            self._last_processed_time_ns is not None
            and timestamp_ns <= self._last_processed_time_ns
        ):
            return
        if timestamp_ns not in self._scheduled_times:
            heapq.heappush(self._timeline, timestamp_ns)
            self._scheduled_times.add(timestamp_ns)

    def _peek_internal_time(self) -> int | None:
        while self._timeline and self._timeline[0] not in self._scheduled_times:
            heapq.heappop(self._timeline)
        return None if not self._timeline else self._timeline[0]

    def _consume_internal_time(self, timestamp_ns: int) -> None:
        if timestamp_ns not in self._scheduled_times:
            return
        if not self._timeline or self._timeline[0] != timestamp_ns:
            raise RuntimeError("internal wake heap/set drifted")
        heapq.heappop(self._timeline)
        self._scheduled_times.remove(timestamp_ns)

    def _effect_cursor(self, timestamp_ns: int, phase: int) -> EventCursor:
        if timestamp_ns != self._current_time_ns:
            raise RuntimeError("effect cursor must belong to the active timestamp")
        self._phase_rows[phase] += 1
        return EventCursor(timestamp_ns, phase, self._phase_rows[phase])

    def _validate_external_product(self, event: S1ExternalEvent) -> None:
        if isinstance(event, (EntryObservation, VenueBookUpdate)):
            product_id = event.product_id
        elif isinstance(event, ContractExpiry):
            product_id = event.close.product_id
        else:
            return
        if product_id not in self.products:
            raise ValueError(f"unknown product_id: {product_id}")

    def _verify(self) -> None:
        self._ledger.verify()
        for controller in self.exit_controllers.values():
            controller.verify()
        if any(
            position.state == "entry_emergency_rollback_flat"
            and self._ledger.account_balances(
                position.capacity_id
            ).total_committed_notional_twd
            != 0
            for position in self._positions.values()
        ):
            raise RuntimeError("successful rollback retained committed capacity")
        if any(
            position.state == "paired_open"
            and self._ledger.account_balances(position.capacity_id).paired_open <= 0
            for position in self._positions.values()
        ):
            raise RuntimeError("paired position lacks paired-open capacity")
        if any(
            position.state == "exit_maker_flat"
            and self._ledger.account_balances(
                position.capacity_id
            ).total_committed_notional_twd
            != 0
            for position in self._positions.values()
        ):
            raise RuntimeError("flat normal exit retained committed capacity")
        if any(
            position.state == "expiry_basis_zero_accounting"
            and self._ledger.account_balances(
                position.capacity_id
            ).total_committed_notional_twd
            != 0
            for position in self._positions.values()
        ):
            raise RuntimeError("expiry accounting retained committed capacity")
        marked_position_ids = {mark.position_id for mark in self._expiry_marks}
        expected_marked_ids = {
            position.position_id
            for position in self._positions.values()
            if position.state == "expiry_basis_zero_accounting"
        }
        if marked_position_ids != expected_marked_ids:
            raise RuntimeError("expiry accounting marks differ from expired positions")
        for previous, current in zip(self._executions, self._executions[1:]):
            if current.cursor <= previous.cursor:
                raise RuntimeError("execution facts must be strictly cursor ordered")
        for venue, cap in (
            (SPOT, self.config.spot_request_cap),
            (FUTURE, self.config.future_request_cap),
        ):
            sent = sorted(
                event.event_cursor.recv_time_ns
                for event in self._request_events
                if event.venue == venue and event.event_type == "actual_send"
            )
            active: deque[int] = deque()
            for timestamp_ns in sent:
                lower = timestamp_ns - self.config.request_window_ns
                while active and active[0] <= lower:
                    active.popleft()
                active.append(timestamp_ns)
                if len(active) > cap:
                    raise RuntimeError(f"{venue} rolling request cap was exceeded")
        admitted_ids = {
            event.capacity_id for event in self._admission_events if event.admitted
        }
        if any(order.capacity_id not in admitted_ids for order in self._sent_orders):
            raise RuntimeError("physical new order lacks committed admission")


def _external_cursor(event: S1ExternalEvent) -> EventCursor:
    if isinstance(event, VenueBookUpdate):
        return event.event.book_cursor.cursor
    if isinstance(event, ContractExpiry):
        return event.close.source_cursor
    return event.source_cursor


def _external_sort_key(event: S1ExternalEvent) -> tuple[object, ...]:
    kind = {
        VenueBookUpdate: 0,
        EntryObservation: 1,
        EntryCutoff: 2,
        SessionExpiry: 3,
        ContractExpiry: 4,
    }[type(event)]
    if isinstance(event, VenueBookUpdate):
        packet = event.event.book_cursor.packet_sequence
        product_id = event.product_id
        venue = event.venue
    elif isinstance(event, EntryObservation):
        packet = -1
        product_id = event.product_id
        venue = ""
    elif isinstance(event, ContractExpiry):
        packet = event.close.packet_sequence
        product_id = event.close.product_id
        venue = SPOT
    else:
        packet = -1
        product_id = ""
        venue = ""
    return (_external_cursor(event), kind, venue, product_id, packet)


def _book_source_id(cursor: object) -> str:
    book_cursor = cursor
    return (
        f"raw-book/{book_cursor.cursor.recv_time_ns}/"
        f"{book_cursor.cursor.event_sequence}/{book_cursor.cursor.row_index}/"
        f"{book_cursor.packet_sequence}"
    )


def _validate_carry_position_for_codec(carry: S1CarryPosition) -> None:
    fact = carry.position_established_fact
    if not isinstance(fact, PositionEstablishedFact):
        raise S1CarryCodecError("S1 carry position requires a PositionEstablishedFact")
    for name in (
        "product_id",
        "quote_code",
        "capacity_id",
        "initiating_raw_order_fact_id",
        "hedge_intent_id",
    ):
        value = getattr(carry, name)
        if type(value) is not str or not value:
            raise S1CarryCodecError(f"S1 carry position {name} must be text")
    if carry.capacity_id != fact.capacity_id:
        raise S1CarryCodecError(
            "S1 carry position capacity_id differs from establishment fact"
        )
    if type(carry.frozen_exit_threshold_basis_bp) is not float or not math.isfinite(
        carry.frozen_exit_threshold_basis_bp
    ):
        raise S1CarryCodecError(
            "S1 carry position frozen_exit_threshold_basis_bp must be a finite float"
        )
    if type(carry.execution_truth) is not str or carry.execution_truth not in (
        "approximate",
        "exact",
    ):
        raise S1CarryCodecError("S1 carry position execution_truth is invalid")
    if carry.execution_truth != fact.execution_truth:
        raise S1CarryCodecError(
            "S1 carry position execution truth differs from establishment fact"
        )
    for name in (
        "establishment_id",
        "position_id",
        "value_code",
        "scenario_id",
        "capacity_id",
        "capacity_transition_id",
    ):
        value = getattr(fact, name)
        if type(value) is not str or not value:
            raise S1CarryCodecError(
                f"S1 carry position establishment fact {name} must be text"
            )
    if type(fact.sequence) is not int or fact.sequence <= 0:
        raise S1CarryCodecError(
            "S1 carry position establishment fact sequence must be positive"
        )
    try:
        _valid_date(fact.establishment_date, "establishment_date")
        _cursor(fact.cursor, "establishment cursor")
        _positive_int(fact.position_established_ns, "position_established_ns")
        _positive_int(fact.spot_shares, "spot_shares")
        _positive_int(fact.short_future_contracts, "short_future_contracts")
        _positive_int(
            fact.short_future_share_equivalent,
            "short_future_share_equivalent",
        )
    except (TypeError, ValueError) as error:
        raise S1CarryCodecError(str(error)) from error
    if fact.position_established_ns != fact.cursor.recv_time_ns:
        raise S1CarryCodecError(
            "S1 carry position establishment time differs from its cursor"
        )
    if fact.spot_shares != fact.short_future_share_equivalent:
        raise S1CarryCodecError(
            "S1 carry position establishment fact is not exactly paired"
        )
    if type(fact.execution_truth) is not str or fact.execution_truth not in (
        "approximate",
        "exact",
    ):
        raise S1CarryCodecError(
            "S1 carry position establishment execution_truth is invalid"
        )


def _validate_carry_contract_binding_for_codec(
    binding: S1CarryContractBinding,
) -> None:
    for name in ("product_id", "value_code", "quote_code"):
        value = getattr(binding, name)
        if type(value) is not str or not value:
            raise S1CarryCodecError(f"S1 carry contract binding {name} must be text")
    for name in ("contract_size_shares", "future_contracts"):
        value = getattr(binding, name)
        if type(value) is not int or value <= 0:
            raise S1CarryCodecError(
                f"S1 carry contract binding {name} must be a positive integer"
            )
    if binding.end_date is not None:
        if type(binding.end_date) is not str:
            raise S1CarryCodecError(
                "S1 carry contract binding end_date must be a string or null"
            )
        try:
            _valid_date(binding.end_date, "end_date")
        except ValueError as error:
            raise S1CarryCodecError(str(error)) from error


def _require_carry_codec_keys(
    record: Mapping[object, object],
    expected_keys: set[str],
    path: str,
) -> None:
    actual_keys = set(record)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys.difference(actual_keys))
    unknown = sorted(repr(key) for key in actual_keys.difference(expected_keys))
    raise S1CarryCodecError(
        f"{path} schema mismatch: missing={missing}, unknown={unknown}"
    )


def _require_carry_codec_tag(
    record: Mapping[str, object],
    *,
    expected_type: str,
    expected_schema: int,
    path: str,
) -> None:
    record_type = record["record_type"]
    if type(record_type) is not str or record_type != expected_type:
        raise S1CarryCodecError(f"{path} record_type is invalid")
    schema_version = record["schema_version"]
    if type(schema_version) is not int or schema_version != expected_schema:
        raise S1CarryCodecError(f"{path} schema_version is unsupported")


def _decode_carry_codec_text(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise S1CarryCodecError(f"{path} must be a non-empty string")
    return value


def _decode_carry_codec_positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise S1CarryCodecError(f"{path} must be a positive integer")
    return value


def _cursor(value: object, name: str) -> EventCursor:
    if not isinstance(value, EventCursor):
        raise TypeError(f"{name} must be an EventCursor")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_text(value: object, name: str) -> str:
    return _text(value, name)


def _valid_date(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} must be valid YYYYMMDD")
    try:
        parsed = date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as error:
        raise ValueError(f"{name} must be valid YYYYMMDD") from error
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"{name} must be valid YYYYMMDD")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_positive_int(value: object, name: str) -> int:
    return _positive_int(value, name)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def _required_positive_float(value: object, name: str) -> float:
    return _positive_float(value, name)


__all__ = [
    "S1_CARRY_CONTRACT_BINDING_RECORD_SCHEMA_VERSION",
    "S1_CARRY_POSITION_RECORD_SCHEMA_VERSION",
    "ActualSendMakerSnapshot",
    "AdmissionEvent",
    "ContractExpiry",
    "EntryCutoff",
    "EntryFillAdapter",
    "EntryObservation",
    "EntryStateAdapter",
    "ExecutionAdapter",
    "FillEvent",
    "NoFillAdapter",
    "NoOpExecutionAdapter",
    "OrderEvent",
    "PositionEvent",
    "PositionSnapshot",
    "PotentialEntryFill",
    "RequestEvent",
    "RiskBookAdapter",
    "RiskEvent",
    "S1CarryCodecError",
    "S1CarryContractBinding",
    "S1CarryPosition",
    "S1EventLoop",
    "S1ExecutionFact",
    "S1ExpiryMarkSummary",
    "S1ExternalEvent",
    "S1LoopConfig",
    "S1Product",
    "S1ReplayResult",
    "SentEntryOrder",
    "SessionExpiry",
    "VenueBookUpdate",
    "decode_s1_carry_contract_binding",
    "decode_s1_carry_position",
    "encode_s1_carry_contract_binding",
    "encode_s1_carry_position",
]
