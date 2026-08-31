"""Replayable executed-leg and expiry accounting for the S1 portfolio.

Executed legs and non-executable expiry marks are deliberately different fact
types. An expiry mark never creates a request or execution and never charges
synthetic exit costs. Every position is sealed by exactly one terminal fact;
open or unresolved inventory cannot contribute to terminal realized net.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date
from numbers import Integral
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .layered import EventCursor
from .transaction_costs import TransactionCostProfile

ExecutionRole = Literal["normal", "hedge", "rollback", "forced"]
ExecutionTruth = Literal["approximate", "exact"]
Side = Literal["buy", "sell"]
Market = Literal["spot", "future"]
ExecutableTerminalOutcome = Literal[
    "exit_maker_flat",
    "aggressive_hard_flat",
    "entry_emergency_rollback_flat",
    "entry_partial_rollback_flat",
    "control_taker_taker_terminal",
]

EXECUTABLE_TERMINAL_OUTCOMES: tuple[ExecutableTerminalOutcome, ...] = (
    "exit_maker_flat",
    "aggressive_hard_flat",
    "entry_emergency_rollback_flat",
    "entry_partial_rollback_flat",
    "control_taker_taker_terminal",
)


class AccountingError(ValueError):
    """Invalid accounting fact or inventory transition."""


class AccountingReplayError(AccountingError):
    """The append-only accounting facts cannot be reproduced exactly."""


class AccountingCodecError(AccountingError):
    """A canonical JSON accounting fact record is malformed."""


class InventoryNotFlatError(AccountingError):
    """Executable terminal realized net was requested for open inventory."""


@dataclass(frozen=True)
class LotAllocation:
    lot_id: str
    position_id: str
    acquisition_date: str
    quantity: int
    contracts: int
    share_equivalent: int
    opening_price: float
    realized_pnl_twd: float
    tax_twd: float
    same_day: bool | None


@dataclass(frozen=True)
class InitiatingExecutionAllocation:
    """Exact share-equivalent slice consumed from one initiating execution."""

    execution_id: str
    share_equivalent: int


@dataclass(frozen=True)
class ExecutedLeg:
    sequence: int
    execution_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    market: Market
    side: Side
    role: ExecutionRole
    execution_truth: ExecutionTruth
    execution_source_id: str
    request_id: str | None
    hedge_intent_id: str | None
    initiating_execution_id: str | None
    initiating_execution_allocations: tuple[InitiatingExecutionAllocation, ...]
    execution_date: str
    cursor: EventCursor
    price: float
    quantity: int
    contracts: int
    shares: int
    share_equivalent: int
    signed_cashflow_twd: float
    realized_pnl_twd: float
    commission_twd: float
    tax_twd: float
    total_cost_twd: float
    opened_lot_id: str | None
    opened_acquisition_date: str | None
    allocations: tuple[LotAllocation, ...]
    spot_shares_after: int
    future_contracts_after: int
    future_share_equivalent_after: int
    cost_profile_id: str
    route_role: str | None = None

    def as_dict(self) -> dict[str, object]:
        return _fact_as_dict(self)


@dataclass(frozen=True)
class PositionEstablishedFact:
    """Causal creation of the canonical S3 FIFO allocation key."""

    sequence: int
    establishment_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    establishment_date: str
    cursor: EventCursor
    position_established_ns: int
    capacity_transition_id: str
    spot_shares: int
    short_future_contracts: int
    short_future_share_equivalent: int
    execution_truth: ExecutionTruth

    @property
    def fifo_key(self) -> tuple[int, str]:
        return self.position_established_ns, self.position_id

    def as_dict(self) -> dict[str, object]:
        row = _fact_as_dict(self)
        row["fifo_position_id"] = self.position_id
        return row


@dataclass(frozen=True)
class TerminalRealizedAccounting:
    """Immutable terminal fact for an executable, actually flat lifecycle."""

    sequence: int
    terminal_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    terminal_date: str
    cursor: EventCursor
    terminal_outcome: ExecutableTerminalOutcome
    capacity_release_transition_id: str
    position_established_ns: int | None
    execution_truth: ExecutionTruth
    executed_leg_count: int
    spot_cashflow_twd: float
    futures_realized_pnl_twd: float
    commission_twd: float
    tax_twd: float
    realized_net_twd: float
    non_executable: bool

    def as_dict(self) -> dict[str, object]:
        return _fact_as_dict(self)


@dataclass(frozen=True)
class ExpiryAccountingMark:
    """Non-executable basis-zero terminal mark for an exactly paired residual."""

    sequence: int
    mark_id: str
    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    expiry_date: str
    cursor: EventCursor
    terminal_outcome: Literal["expiry_basis_zero_accounting"]
    capacity_release_transition_id: str
    position_established_ns: int
    spot_close_source_id: str
    spot_close_source_cursor: EventCursor
    spot_close_price: float
    marked_spot_shares: int
    marked_future_contracts: int
    marked_future_share_equivalent: int
    spot_mark_cashflow_twd: float
    futures_mark_pnl_twd: float
    spot_cashflow_twd: float
    futures_realized_pnl_twd: float
    actual_commission_twd: float
    actual_tax_twd: float
    synthetic_commission_twd: float
    synthetic_tax_twd: float
    realized_net_twd: float
    execution_truth: ExecutionTruth
    spot_allocations: tuple[LotAllocation, ...]
    future_allocations: tuple[LotAllocation, ...]
    request_id: None
    execution_id: None
    non_executable: bool

    def as_dict(self) -> dict[str, object]:
        return _fact_as_dict(self)


type AccountingFact = (
    ExecutedLeg
    | PositionEstablishedFact
    | TerminalRealizedAccounting
    | ExpiryAccountingMark
)
type TerminalFact = TerminalRealizedAccounting | ExpiryAccountingMark

_FACT_TYPE_BY_CLASS = {
    ExecutedLeg: "executed_leg",
    PositionEstablishedFact: "position_established",
    TerminalRealizedAccounting: "terminal_realized_accounting",
    ExpiryAccountingMark: "expiry_accounting_mark",
}
_FACT_CLASS_BY_TYPE = {value: key for key, value in _FACT_TYPE_BY_CLASS.items()}
_CURSOR_RECORD_FIELDS = ("recv_time_ns", "event_sequence", "row_index")
_SPOT_CLOSE_CURSOR_RECORD_FIELDS = (
    "spot_close_source_recv_time_ns",
    "spot_close_source_event_sequence",
    "spot_close_source_row_index",
)
_ALLOCATION_FIELDS_BY_CLASS = {
    ExecutedLeg: {
        "initiating_execution_allocations": InitiatingExecutionAllocation,
        "allocations": LotAllocation,
    },
    ExpiryAccountingMark: {
        "spot_allocations": LotAllocation,
        "future_allocations": LotAllocation,
    },
}


@dataclass
class _SpotLot:
    lot_id: str
    position_id: str
    acquisition_date: str
    shares: int
    price: float


@dataclass
class _FutureLot:
    lot_id: str
    position_id: str
    acquisition_date: str
    side: Side
    contracts: int
    share_equivalent: int
    price: float


@dataclass(frozen=True)
class _PositionMeta:
    value_code: str
    scenario_id: str
    capacity_id: str


@dataclass(frozen=True)
class _ValidatedLinkage:
    initiating_execution_id: str | None
    allocations: tuple[InitiatingExecutionAllocation, ...]


@dataclass(frozen=True, slots=True)
class S1SpotInventoryLotSnapshot:
    """Immutable public view of one remaining Spot FIFO lot."""

    lot_id: str
    acquisition_date: str
    shares: int
    opening_price: float


@dataclass(frozen=True, slots=True)
class S1FutureInventoryLotSnapshot:
    """Immutable public view of one remaining stock-futures FIFO lot."""

    lot_id: str
    acquisition_date: str
    side: Side
    contracts: int
    share_equivalent: int
    opening_price: float


@dataclass(frozen=True, slots=True)
class S1OpenInventorySnapshot:
    """Verified current lots for one established, unsealed position."""

    position_id: str
    value_code: str
    scenario_id: str
    capacity_id: str
    establishment_date: str
    spot_lots: tuple[S1SpotInventoryLotSnapshot, ...]
    future_lots: tuple[S1FutureInventoryLotSnapshot, ...]


type BookKey = tuple[str, str]
type CursorDateResolver = Callable[[EventCursor], str]


class S1AccountingLedger:
    """Product/scenario FIFO inventory with replayable terminal accounting."""

    def __init__(
        self,
        profile: TransactionCostProfile | None = None,
        *,
        cursor_date_resolver: CursorDateResolver | None = None,
        require_route_roles: bool = False,
    ) -> None:
        self.profile = profile or TransactionCostProfile()
        self.profile.validate()
        if cursor_date_resolver is not None and not callable(cursor_date_resolver):
            raise TypeError("cursor_date_resolver must be callable or None")
        if not isinstance(require_route_roles, bool):
            raise TypeError("require_route_roles must be boolean")
        self.cursor_date_resolver = cursor_date_resolver
        self.require_route_roles = require_route_roles
        self._facts: list[AccountingFact] = []
        self._rows: list[ExecutedLeg] = []
        self._establishment_rows: list[PositionEstablishedFact] = []
        self._marks: list[ExpiryAccountingMark] = []
        self._terminal_rows: list[TerminalRealizedAccounting] = []
        self._positions: dict[str, _PositionMeta] = {}
        self._position_last_date: dict[str, str] = {}
        self._spot_books: dict[BookKey, list[_SpotLot]] = {}
        self._future_books: dict[BookKey, list[_FutureLot]] = {}
        self._executions_by_id: dict[str, ExecutedLeg] = {}
        self._initiating_consumed_share_equivalent: dict[str, int] = {}
        self._establishment_by_position: dict[str, PositionEstablishedFact] = {}
        self._terminal_by_position: dict[str, TerminalFact] = {}
        self._fact_ids: set[str] = set()
        self._last_cursor: EventCursor | None = None
        self._last_date: str | None = None

    @property
    def facts(self) -> tuple[AccountingFact, ...]:
        return tuple(self._facts)

    @property
    def fact_count(self) -> int:
        """Number of append-only facts without materializing the facts tuple."""

        return len(self._facts)

    def facts_since(self, index: int) -> tuple[AccountingFact, ...]:
        """Return the suffix beginning at a previously captured fact count."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("fact index must be an integer")
        if index < 0 or index > len(self._facts):
            raise IndexError("fact index is outside the append-only ledger")
        return tuple(self._facts[index:])

    @property
    def rows(self) -> tuple[ExecutedLeg, ...]:
        return tuple(self._rows)

    @property
    def establishment_rows(self) -> tuple[PositionEstablishedFact, ...]:
        return tuple(self._establishment_rows)

    @property
    def expiry_marks(self) -> tuple[ExpiryAccountingMark, ...]:
        return tuple(self._marks)

    @property
    def terminal_rows(self) -> tuple[TerminalRealizedAccounting, ...]:
        return tuple(self._terminal_rows)

    def record_spot_execution(
        self,
        *,
        execution_id: str,
        position_id: str,
        value_code: str,
        scenario_id: str,
        capacity_id: str,
        side: Side,
        role: ExecutionRole,
        execution_truth: ExecutionTruth,
        execution_source_id: str,
        request_id: str | None,
        hedge_intent_id: str | None,
        initiating_execution_id: str | None,
        initiating_execution_allocations: Sequence[InitiatingExecutionAllocation] = (),
        execution_date: str,
        cursor: EventCursor,
        price: float,
        shares: int,
        route_role: str | None = None,
    ) -> ExecutedLeg:
        price = _positive_price(price, "price")
        shares = _positive_integer(shares, "shares")
        if route_role is not None:
            route_role = _identifier(route_role, "route_role")
        elif self.require_route_roles:
            raise AccountingError("strict accounting requires route_role")
        meta, linkage = self._validate_execution_common(
            execution_id=execution_id,
            position_id=position_id,
            value_code=value_code,
            scenario_id=scenario_id,
            capacity_id=capacity_id,
            market="spot",
            side=side,
            role=role,
            execution_truth=execution_truth,
            execution_source_id=execution_source_id,
            request_id=request_id,
            hedge_intent_id=hedge_intent_id,
            initiating_execution_id=initiating_execution_id,
            initiating_execution_allocations=initiating_execution_allocations,
            execution_date=execution_date,
            cursor=cursor,
            share_equivalent=shares,
            contracts=0,
        )
        key = (meta.value_code, meta.scenario_id)
        inventory = [
            _SpotLot(
                lot.lot_id,
                lot.position_id,
                lot.acquisition_date,
                lot.shares,
                lot.price,
            )
            for lot in self._spot_books.get(key, ())
        ]
        commission = self.profile.spot_commission_twd(price, shares)
        allocations: list[LotAllocation] = []
        tax = 0.0
        realized = 0.0
        opened_lot_id: str | None = None
        if side == "buy":
            opened_lot_id = f"{execution_id}:spot"
            inventory.append(
                _SpotLot(opened_lot_id, position_id, execution_date, shares, price)
            )
        else:
            if role == "normal":
                self._validate_normal_exit_leg_fifo(
                    position_id,
                    cursor,
                    market="spot",
                    closing_side=side,
                )
            elif role == "forced":
                self._validate_exit_fifo_allocation(position_id, cursor)
            own_shares = sum(
                lot.shares for lot in inventory if lot.position_id == position_id
            )
            if own_shares < shares:
                raise AccountingError("spot sell exceeds position long inventory")
            remaining = shares
            while remaining > 0:
                own_index = next(
                    (
                        index
                        for index, candidate in enumerate(inventory)
                        if candidate.position_id == position_id
                    ),
                    None,
                )
                if own_index is None:
                    raise AccountingError("spot FIFO ownership disappeared")
                lot = inventory[own_index]
                matched = min(remaining, lot.shares)
                same_day = lot.acquisition_date == execution_date
                allocation_tax = self.profile.spot_sell_tax_twd(
                    price, matched, same_day=same_day
                )
                allocation_pnl = (price - lot.price) * matched
                allocations.append(
                    LotAllocation(
                        lot_id=lot.lot_id,
                        position_id=lot.position_id,
                        acquisition_date=lot.acquisition_date,
                        quantity=matched,
                        contracts=0,
                        share_equivalent=matched,
                        opening_price=lot.price,
                        realized_pnl_twd=allocation_pnl,
                        tax_twd=allocation_tax,
                        same_day=same_day,
                    )
                )
                tax += allocation_tax
                realized += allocation_pnl
                lot.shares -= matched
                remaining -= matched
                if lot.shares == 0:
                    inventory.pop(own_index)
        signed_cashflow = price * shares * (1.0 if side == "sell" else -1.0)
        _require_finite_money(
            signed_cashflow=signed_cashflow,
            realized=realized,
            commission=commission,
            tax=tax,
        )
        self._validate_rollback_lot_linkage(
            role=role,
            opened_lot_id=opened_lot_id,
            linkage=linkage,
            allocations=allocations,
        )
        self._spot_books[key] = inventory
        self._positions.setdefault(position_id, meta)
        leg = self._build_execution(
            execution_id=execution_id,
            position_id=position_id,
            meta=meta,
            market="spot",
            side=side,
            role=role,
            execution_truth=execution_truth,
            execution_source_id=execution_source_id,
            request_id=request_id,
            hedge_intent_id=hedge_intent_id,
            initiating_execution_id=linkage.initiating_execution_id,
            initiating_execution_allocations=linkage.allocations,
            execution_date=execution_date,
            cursor=cursor,
            price=price,
            quantity=shares,
            contracts=0,
            shares=shares,
            share_equivalent=shares,
            signed_cashflow=signed_cashflow,
            realized=realized,
            commission=commission,
            tax=tax,
            opened_lot_id=opened_lot_id,
            allocations=allocations,
            route_role=route_role,
        )
        self._commit_execution(leg)
        return leg

    def record_future_execution(
        self,
        *,
        execution_id: str,
        position_id: str,
        value_code: str,
        scenario_id: str,
        capacity_id: str,
        side: Side,
        role: ExecutionRole,
        execution_truth: ExecutionTruth,
        execution_source_id: str,
        request_id: str | None,
        hedge_intent_id: str | None,
        initiating_execution_id: str | None,
        initiating_execution_allocations: Sequence[InitiatingExecutionAllocation] = (),
        execution_date: str,
        cursor: EventCursor,
        price: float,
        contracts: int,
        share_equivalent: int,
        route_role: str | None = None,
    ) -> ExecutedLeg:
        price = _positive_price(price, "price")
        contracts = _positive_integer(contracts, "contracts")
        share_equivalent = _positive_integer(share_equivalent, "share_equivalent")
        if route_role is not None:
            route_role = _identifier(route_role, "route_role")
        elif self.require_route_roles:
            raise AccountingError("strict accounting requires route_role")
        if share_equivalent % contracts != 0:
            raise AccountingError(
                "future share_equivalent must be integral per contract"
            )
        meta, linkage = self._validate_execution_common(
            execution_id=execution_id,
            position_id=position_id,
            value_code=value_code,
            scenario_id=scenario_id,
            capacity_id=capacity_id,
            market="future",
            side=side,
            role=role,
            execution_truth=execution_truth,
            execution_source_id=execution_source_id,
            request_id=request_id,
            hedge_intent_id=hedge_intent_id,
            initiating_execution_id=initiating_execution_id,
            initiating_execution_allocations=initiating_execution_allocations,
            execution_date=execution_date,
            cursor=cursor,
            share_equivalent=share_equivalent,
            contracts=contracts,
        )
        key = (meta.value_code, meta.scenario_id)
        inventory = [
            _FutureLot(
                lot.lot_id,
                lot.position_id,
                lot.acquisition_date,
                lot.side,
                lot.contracts,
                lot.share_equivalent,
                lot.price,
            )
            for lot in self._future_books.get(key, ())
        ]
        own = [lot for lot in inventory if lot.position_id == position_id]
        if own and any(lot.side != own[0].side for lot in own):
            raise AccountingError("position contains opposing gross future lots")
        closing = bool(own and own[0].side != side)
        allocations: list[LotAllocation] = []
        realized = 0.0
        opened_lot_id: str | None = None
        if closing:
            if role == "normal":
                self._validate_normal_exit_leg_fifo(
                    position_id,
                    cursor,
                    market="future",
                    closing_side=side,
                )
            elif role == "forced":
                self._validate_exit_fifo_allocation(position_id, cursor)
            if sum(lot.contracts for lot in own) < contracts:
                raise AccountingError("future close exceeds position inventory")
            if sum(lot.share_equivalent for lot in own) < share_equivalent:
                raise AccountingError(
                    "future close exceeds position share-equivalent inventory"
                )
            remaining_contracts = contracts
            remaining_equivalent = share_equivalent
            while remaining_contracts > 0:
                own_index = next(
                    (
                        index
                        for index, candidate in enumerate(inventory)
                        if candidate.position_id == position_id
                    ),
                    None,
                )
                if own_index is None:
                    raise AccountingError("future FIFO ownership disappeared")
                lot = inventory[own_index]
                if lot.side == side:
                    raise AccountingError("future close side does not offset FIFO lot")
                if lot.share_equivalent * contracts != share_equivalent * lot.contracts:
                    raise AccountingError("FIFO future share-equivalent ratio mismatch")
                matched_contracts = min(remaining_contracts, lot.contracts)
                numerator = matched_contracts * lot.share_equivalent
                if numerator % lot.contracts != 0:
                    raise AccountingError(
                        "matched future share-equivalent is not integral"
                    )
                matched_equivalent = numerator // lot.contracts
                pnl = (
                    (price - lot.price) * matched_equivalent
                    if lot.side == "buy"
                    else (lot.price - price) * matched_equivalent
                )
                allocations.append(
                    LotAllocation(
                        lot_id=lot.lot_id,
                        position_id=lot.position_id,
                        acquisition_date=lot.acquisition_date,
                        quantity=matched_contracts,
                        contracts=matched_contracts,
                        share_equivalent=matched_equivalent,
                        opening_price=lot.price,
                        realized_pnl_twd=pnl,
                        tax_twd=0.0,
                        same_day=None,
                    )
                )
                realized += pnl
                lot.contracts -= matched_contracts
                lot.share_equivalent -= matched_equivalent
                remaining_contracts -= matched_contracts
                remaining_equivalent -= matched_equivalent
                if lot.contracts == 0:
                    if lot.share_equivalent != 0:
                        raise AccountingError(
                            "future lot contracts and share equivalent diverged"
                        )
                    inventory.pop(own_index)
            if remaining_equivalent != 0:
                raise AccountingError("future close share equivalent did not reconcile")
        else:
            if inventory and any(lot.side != side for lot in inventory):
                raise AccountingError(
                    "cannot open opposing gross future inventory in one scenario"
                )
            opened_lot_id = f"{execution_id}:future"
            inventory.append(
                _FutureLot(
                    opened_lot_id,
                    position_id,
                    execution_date,
                    side,
                    contracts,
                    share_equivalent,
                    price,
                )
            )
        commission = self.profile.futures_commission_twd(contracts)
        tax = self.profile.futures_tax_twd(price, share_equivalent)
        _require_finite_money(
            realized=realized,
            commission=commission,
            tax=tax,
        )
        self._validate_rollback_lot_linkage(
            role=role,
            opened_lot_id=opened_lot_id,
            linkage=linkage,
            allocations=allocations,
        )
        self._future_books[key] = inventory
        self._positions.setdefault(position_id, meta)
        leg = self._build_execution(
            execution_id=execution_id,
            position_id=position_id,
            meta=meta,
            market="future",
            side=side,
            role=role,
            execution_truth=execution_truth,
            execution_source_id=execution_source_id,
            request_id=request_id,
            hedge_intent_id=hedge_intent_id,
            initiating_execution_id=linkage.initiating_execution_id,
            initiating_execution_allocations=linkage.allocations,
            execution_date=execution_date,
            cursor=cursor,
            price=price,
            quantity=contracts,
            contracts=contracts,
            shares=0,
            share_equivalent=share_equivalent,
            signed_cashflow=0.0,
            realized=realized,
            commission=commission,
            tax=tax,
            opened_lot_id=opened_lot_id,
            allocations=allocations,
            route_role=route_role,
        )
        self._commit_execution(leg)
        return leg

    def establish_position(
        self,
        *,
        establishment_id: str,
        position_id: str,
        establishment_date: str,
        cursor: EventCursor,
        position_established_ns: int,
        capacity_transition_id: str,
    ) -> PositionEstablishedFact:
        """Create the immutable ``(established_ns, position_id)`` FIFO key."""

        position_id = _identifier(position_id, "position_id")
        meta = self._required_open_position(position_id)
        self._validate_fact_common(establishment_id, establishment_date, cursor)
        if position_id in self._establishment_by_position:
            raise AccountingError("position establishment cannot be repeated")
        position_established_ns = _positive_integer(
            position_established_ns, "position_established_ns"
        )
        if position_established_ns != cursor.recv_time_ns:
            raise AccountingError(
                "position establishment cannot be backfilled after its causal cursor"
            )
        if establishment_date != self._position_last_date[position_id]:
            raise AccountingError(
                "position establishment date must equal the pairing execution date"
            )
        capacity_transition_id = _identifier(
            capacity_transition_id, "capacity_transition_id"
        )
        spot_shares, future_contracts, future_equivalent = self._required_exact_pair(
            position_id
        )
        fact = PositionEstablishedFact(
            sequence=len(self._facts) + 1,
            establishment_id=establishment_id,
            position_id=position_id,
            value_code=meta.value_code,
            scenario_id=meta.scenario_id,
            capacity_id=meta.capacity_id,
            establishment_date=establishment_date,
            cursor=cursor,
            position_established_ns=position_established_ns,
            capacity_transition_id=capacity_transition_id,
            spot_shares=spot_shares,
            short_future_contracts=future_contracts,
            short_future_share_equivalent=future_equivalent,
            execution_truth=self._position_execution_truth(position_id),
        )
        self._establishment_rows.append(fact)
        self._establishment_by_position[position_id] = fact
        self._commit_fact(fact, establishment_id, establishment_date, cursor)
        return fact

    def seal_terminal(
        self,
        *,
        terminal_id: str,
        position_id: str,
        terminal_date: str,
        cursor: EventCursor,
        terminal_outcome: ExecutableTerminalOutcome,
        capacity_release_transition_id: str,
    ) -> TerminalRealizedAccounting:
        position_id = _identifier(position_id, "position_id")
        meta = self._required_open_position(position_id)
        self._validate_fact_common(terminal_id, terminal_date, cursor)
        if terminal_outcome not in EXECUTABLE_TERMINAL_OUTCOMES:
            raise AccountingError(
                f"invalid executable terminal outcome: {terminal_outcome}"
            )
        capacity_release_transition_id = _identifier(
            capacity_release_transition_id, "capacity_release_transition_id"
        )
        if terminal_date != self._position_last_date[position_id]:
            raise AccountingError(
                "executable terminal date must equal the final execution date"
            )
        if not self._position_is_exactly_flat(position_id):
            raise InventoryNotFlatError(
                "spot and future lot inventory must both be exactly flat"
            )
        establishment = self._establishment_by_position.get(position_id)
        route_roles = tuple(
            row.route_role
            for row in self._rows
            if row.position_id == position_id and row.route_role is not None
        )
        if (
            terminal_outcome
            in (
                "exit_maker_flat",
                "aggressive_hard_flat",
                "control_taker_taker_terminal",
            )
            and establishment is None
        ):
            raise AccountingError(
                "executable paired exit requires a causal position establishment"
            )
        if (
            terminal_outcome
            in (
                "entry_emergency_rollback_flat",
                "entry_partial_rollback_flat",
            )
            and establishment is not None
        ):
            raise AccountingError(
                "entry rollback terminal cannot follow position establishment"
            )
        if route_roles:
            if terminal_outcome == "exit_maker_flat" and not {
                "exit_maker",
                "exit_hedge",
            }.issubset(route_roles):
                raise AccountingError(
                    "exit_maker_flat route roles lack maker/hedge executions"
                )
            if terminal_outcome in (
                "entry_emergency_rollback_flat",
                "entry_partial_rollback_flat",
            ) and not {"entry_maker", "entry_rollback"}.issubset(route_roles):
                raise AccountingError(
                    "entry rollback route roles lack maker/rollback executions"
                )
        totals = self._position_actual_totals(position_id)
        terminal = TerminalRealizedAccounting(
            sequence=len(self._facts) + 1,
            terminal_id=terminal_id,
            position_id=position_id,
            value_code=meta.value_code,
            scenario_id=meta.scenario_id,
            capacity_id=meta.capacity_id,
            terminal_date=terminal_date,
            cursor=cursor,
            terminal_outcome=terminal_outcome,
            capacity_release_transition_id=capacity_release_transition_id,
            position_established_ns=(
                establishment.position_established_ns if establishment else None
            ),
            execution_truth=self._position_execution_truth(position_id),
            executed_leg_count=totals[0],
            spot_cashflow_twd=totals[1],
            futures_realized_pnl_twd=totals[2],
            commission_twd=totals[3],
            tax_twd=totals[4],
            realized_net_twd=totals[1] + totals[2] - totals[3] - totals[4],
            non_executable=False,
        )
        _require_finite_money(
            spot_cashflow=terminal.spot_cashflow_twd,
            futures_pnl=terminal.futures_realized_pnl_twd,
            commission=terminal.commission_twd,
            tax=terminal.tax_twd,
            realized_net=terminal.realized_net_twd,
        )
        self._terminal_rows.append(terminal)
        self._terminal_by_position[position_id] = terminal
        self._commit_fact(terminal, terminal_id, terminal_date, cursor)
        return terminal

    def record_expiry_accounting_mark(
        self,
        *,
        mark_id: str,
        position_id: str,
        expiry_date: str,
        cursor: EventCursor,
        spot_close_source_id: str,
        spot_close_source_cursor: EventCursor,
        spot_close_price: float,
        capacity_release_transition_id: str,
    ) -> ExpiryAccountingMark:
        """Seal an exactly paired residual without manufacturing exit legs."""

        position_id = _identifier(position_id, "position_id")
        meta = self._required_open_position(position_id)
        self._validate_fact_common(mark_id, expiry_date, cursor)
        establishment = self._establishment_by_position.get(position_id)
        capacity_release_transition_id = _identifier(
            capacity_release_transition_id, "capacity_release_transition_id"
        )
        spot_close_source_id = _identifier(spot_close_source_id, "spot_close_source_id")
        if not isinstance(spot_close_source_cursor, EventCursor):
            raise AccountingError("spot_close_source_cursor must be EventCursor")
        if spot_close_source_cursor > cursor:
            raise AccountingError(
                "spot close source cursor cannot follow the accounting mark cursor"
            )
        if self.cursor_date_resolver is not None:
            source_date = self.cursor_date_resolver(spot_close_source_cursor)
            _valid_yyyymmdd(source_date, "resolved spot close source date")
            if source_date != expiry_date:
                raise AccountingError(
                    "spot close source date disagrees with accounting mark date"
                )
        spot_close_price = _positive_price(spot_close_price, "spot_close_price")
        key = (meta.value_code, meta.scenario_id)
        spot_lots = [
            lot
            for lot in self._spot_books.get(key, ())
            if lot.position_id == position_id
        ]
        future_lots = [
            lot
            for lot in self._future_books.get(key, ())
            if lot.position_id == position_id
        ]
        spot_shares = sum(lot.shares for lot in spot_lots)
        future_contracts = sum(lot.contracts for lot in future_lots)
        future_equivalent = sum(lot.share_equivalent for lot in future_lots)
        if (
            spot_shares <= 0
            or future_contracts <= 0
            or future_equivalent <= 0
            or any(lot.side != "sell" for lot in future_lots)
            or spot_shares != future_equivalent
        ):
            raise InventoryNotFlatError(
                "expiry mark requires exactly paired long spot and short future"
            )
        if establishment is None or establishment.cursor >= cursor:
            raise AccountingError(
                "expiry mark requires an earlier causal position establishment"
            )
        spot_allocations = tuple(
            LotAllocation(
                lot_id=lot.lot_id,
                position_id=lot.position_id,
                acquisition_date=lot.acquisition_date,
                quantity=lot.shares,
                contracts=0,
                share_equivalent=lot.shares,
                opening_price=lot.price,
                realized_pnl_twd=(spot_close_price - lot.price) * lot.shares,
                tax_twd=0.0,
                same_day=None,
            )
            for lot in spot_lots
        )
        future_allocations = tuple(
            LotAllocation(
                lot_id=lot.lot_id,
                position_id=lot.position_id,
                acquisition_date=lot.acquisition_date,
                quantity=lot.contracts,
                contracts=lot.contracts,
                share_equivalent=lot.share_equivalent,
                opening_price=lot.price,
                realized_pnl_twd=(lot.price - spot_close_price) * lot.share_equivalent,
                tax_twd=0.0,
                same_day=None,
            )
            for lot in future_lots
        )
        spot_mark_cashflow = spot_close_price * spot_shares
        futures_mark_pnl = math.fsum(
            allocation.realized_pnl_twd for allocation in future_allocations
        )
        totals = self._position_actual_totals(position_id)
        spot_cashflow = totals[1] + spot_mark_cashflow
        futures_pnl = totals[2] + futures_mark_pnl
        realized_net = spot_cashflow + futures_pnl - totals[3] - totals[4]
        _require_finite_money(
            spot_mark_cashflow=spot_mark_cashflow,
            futures_mark_pnl=futures_mark_pnl,
            spot_cashflow=spot_cashflow,
            futures_pnl=futures_pnl,
            commission=totals[3],
            tax=totals[4],
            realized_net=realized_net,
        )
        mark = ExpiryAccountingMark(
            sequence=len(self._facts) + 1,
            mark_id=mark_id,
            position_id=position_id,
            value_code=meta.value_code,
            scenario_id=meta.scenario_id,
            capacity_id=meta.capacity_id,
            expiry_date=expiry_date,
            cursor=cursor,
            terminal_outcome="expiry_basis_zero_accounting",
            capacity_release_transition_id=capacity_release_transition_id,
            position_established_ns=establishment.position_established_ns,
            spot_close_source_id=spot_close_source_id,
            spot_close_source_cursor=spot_close_source_cursor,
            spot_close_price=spot_close_price,
            marked_spot_shares=spot_shares,
            marked_future_contracts=future_contracts,
            marked_future_share_equivalent=future_equivalent,
            spot_mark_cashflow_twd=spot_mark_cashflow,
            futures_mark_pnl_twd=futures_mark_pnl,
            spot_cashflow_twd=spot_cashflow,
            futures_realized_pnl_twd=futures_pnl,
            actual_commission_twd=totals[3],
            actual_tax_twd=totals[4],
            synthetic_commission_twd=0.0,
            synthetic_tax_twd=0.0,
            realized_net_twd=realized_net,
            execution_truth=self._position_execution_truth(position_id),
            spot_allocations=spot_allocations,
            future_allocations=future_allocations,
            request_id=None,
            execution_id=None,
            non_executable=True,
        )
        self._spot_books[key] = [
            lot
            for lot in self._spot_books.get(key, ())
            if lot.position_id != position_id
        ]
        self._future_books[key] = [
            lot
            for lot in self._future_books.get(key, ())
            if lot.position_id != position_id
        ]
        self._marks.append(mark)
        self._terminal_by_position[position_id] = mark
        self._commit_fact(mark, mark_id, expiry_date, cursor)
        return mark

    def inventory(self, position_id: str) -> tuple[int, int, int]:
        position_id = _identifier(position_id, "position_id")
        meta = self._positions.get(position_id)
        if meta is None:
            return (0, 0, 0)
        key = (meta.value_code, meta.scenario_id)
        spot = sum(
            lot.shares
            for lot in self._spot_books.get(key, ())
            if lot.position_id == position_id
        )
        future_contracts = sum(
            lot.contracts * (1 if lot.side == "buy" else -1)
            for lot in self._future_books.get(key, ())
            if lot.position_id == position_id
        )
        future_equivalent = sum(
            lot.share_equivalent * (1 if lot.side == "buy" else -1)
            for lot in self._future_books.get(key, ())
            if lot.position_id == position_id
        )
        return spot, future_contracts, future_equivalent

    def open_inventory_snapshot(self, position_id: str) -> S1OpenInventorySnapshot:
        """Return copied FIFO lots without exposing mutable ledger internals."""

        position_id = _identifier(position_id, "position_id")
        meta = self._positions.get(position_id)
        if meta is None:
            raise AccountingError(f"unknown position_id: {position_id}")
        establishment = self._establishment_by_position.get(position_id)
        if establishment is None:
            raise AccountingError("open inventory snapshot requires establishment")
        if position_id in self._terminal_by_position:
            raise InventoryNotFlatError("sealed position has no open inventory snapshot")
        key = (meta.value_code, meta.scenario_id)
        return S1OpenInventorySnapshot(
            position_id=position_id,
            value_code=meta.value_code,
            scenario_id=meta.scenario_id,
            capacity_id=meta.capacity_id,
            establishment_date=establishment.establishment_date,
            spot_lots=tuple(
                S1SpotInventoryLotSnapshot(
                    lot_id=lot.lot_id,
                    acquisition_date=lot.acquisition_date,
                    shares=lot.shares,
                    opening_price=lot.price,
                )
                for lot in self._spot_books.get(key, ())
                if lot.position_id == position_id
            ),
            future_lots=tuple(
                S1FutureInventoryLotSnapshot(
                    lot_id=lot.lot_id,
                    acquisition_date=lot.acquisition_date,
                    side=lot.side,
                    contracts=lot.contracts,
                    share_equivalent=lot.share_equivalent,
                    opening_price=lot.price,
                )
                for lot in self._future_books.get(key, ())
                if lot.position_id == position_id
            ),
        )

    def terminal_realized(self, position_id: str) -> TerminalFact:
        position_id = _identifier(position_id, "position_id")
        if position_id not in self._positions:
            raise AccountingError(f"unknown position_id: {position_id}")
        terminal = self._terminal_by_position.get(position_id)
        if terminal is None:
            raise InventoryNotFlatError(
                "position has not been sealed by a terminal fact"
            )
        return terminal

    def verify(self) -> S1AccountingLedger:
        return replay_accounting_facts(
            self._facts,
            profile=self.profile,
            cursor_date_resolver=self.cursor_date_resolver,
            require_route_roles=self.require_route_roles,
        )

    def _validate_execution_common(
        self,
        *,
        execution_id: str,
        position_id: str,
        value_code: str,
        scenario_id: str,
        capacity_id: str,
        market: Market,
        side: Side,
        role: ExecutionRole,
        execution_truth: ExecutionTruth,
        execution_source_id: str,
        request_id: str | None,
        hedge_intent_id: str | None,
        initiating_execution_id: str | None,
        initiating_execution_allocations: Sequence[InitiatingExecutionAllocation],
        execution_date: str,
        cursor: EventCursor,
        share_equivalent: int,
        contracts: int,
    ) -> tuple[_PositionMeta, _ValidatedLinkage]:
        execution_id = _identifier(execution_id, "execution_id")
        position_id = _identifier(position_id, "position_id")
        value_code = _identifier(value_code, "value_code")
        scenario_id = _identifier(scenario_id, "scenario_id")
        capacity_id = _identifier(capacity_id, "capacity_id")
        _identifier(execution_source_id, "execution_source_id")
        if request_id is not None:
            _identifier(request_id, "request_id")
        self._validate_fact_common(execution_id, execution_date, cursor)
        if position_id in self._terminal_by_position:
            raise AccountingError("sealed position cannot be reopened")
        if market not in ("spot", "future"):
            raise AccountingError(f"invalid market: {market}")
        if side not in ("buy", "sell"):
            raise AccountingError(f"invalid side: {side}")
        if role not in ("normal", "hedge", "rollback", "forced"):
            raise AccountingError(f"invalid role: {role}")
        if execution_truth not in ("approximate", "exact"):
            raise AccountingError(f"invalid execution_truth: {execution_truth}")
        meta = _PositionMeta(value_code, scenario_id, capacity_id)
        existing_meta = self._positions.get(position_id)
        if existing_meta is not None and existing_meta != meta:
            raise AccountingError(
                "position value_code/scenario_id/capacity_id cannot change"
            )
        linkage = self._validate_execution_linkage(
            position_id=position_id,
            market=market,
            side=side,
            role=role,
            request_id=request_id,
            hedge_intent_id=hedge_intent_id,
            initiating_execution_id=initiating_execution_id,
            initiating_execution_allocations=initiating_execution_allocations,
            cursor=cursor,
            share_equivalent=share_equivalent,
            contracts=contracts,
        )
        if role == "forced":
            _identifier(request_id, "request_id")
        return meta, linkage

    def _validate_execution_linkage(
        self,
        *,
        position_id: str,
        market: Market,
        side: Side,
        role: ExecutionRole,
        request_id: str | None,
        hedge_intent_id: str | None,
        initiating_execution_id: str | None,
        initiating_execution_allocations: Sequence[InitiatingExecutionAllocation],
        cursor: EventCursor,
        share_equivalent: int,
        contracts: int,
    ) -> _ValidatedLinkage:
        if isinstance(initiating_execution_allocations, (str, bytes)):
            raise AccountingError("initiating execution allocations must be a sequence")
        try:
            supplied = tuple(initiating_execution_allocations)
        except TypeError as error:
            raise AccountingError(
                "initiating execution allocations must be a sequence"
            ) from error
        if role not in ("hedge", "rollback"):
            if (
                hedge_intent_id is not None
                or initiating_execution_id is not None
                or supplied
            ):
                raise AccountingError(
                    "normal/forced execution cannot carry hedge/rollback linkage"
                )
            return _ValidatedLinkage(None, ())

        _identifier(request_id, "request_id")
        _identifier(hedge_intent_id, "hedge_intent_id")
        legacy_link = not supplied
        if legacy_link:
            source_id = _identifier(initiating_execution_id, "initiating_execution_id")
            initiating = self._executions_by_id.get(source_id)
            if initiating is None:
                raise AccountingError(
                    "hedge/rollback must reference a prior initiating execution"
                )
            if initiating.share_equivalent != share_equivalent:
                raise AccountingError(
                    "hedge/rollback share quantity must match initiating execution"
                )
            supplied = (InitiatingExecutionAllocation(source_id, share_equivalent),)
        else:
            if any(
                not isinstance(item, InitiatingExecutionAllocation) for item in supplied
            ):
                raise AccountingError("invalid initiating execution allocation type")
            if initiating_execution_id is not None and (
                len(supplied) != 1
                or supplied[0].execution_id != initiating_execution_id
            ):
                raise AccountingError(
                    "single initiating_execution_id disagrees with allocations"
                )
        seen: set[str] = set()
        allocated_total = 0
        allocated_future_contracts = 0
        normalized: list[InitiatingExecutionAllocation] = []
        for item in supplied:
            source_id = _identifier(item.execution_id, "initiating execution_id")
            allocated = _positive_integer(
                item.share_equivalent,
                "initiating share_equivalent",
            )
            if source_id in seen:
                raise AccountingError(
                    "initiating execution allocations cannot repeat a source"
                )
            seen.add(source_id)
            initiating = self._executions_by_id.get(source_id)
            if initiating is None:
                raise AccountingError(
                    "hedge/rollback must reference prior initiating executions"
                )
            if initiating.cursor >= cursor:
                raise AccountingError(
                    "initiating execution must causally precede hedge/rollback"
                )
            if initiating.position_id != position_id:
                raise AccountingError(
                    "initiating execution belongs to another position"
                )
            if initiating.role not in ("normal", "forced"):
                raise AccountingError(
                    "initiating execution must be a normal/forced first leg"
                )
            if initiating.side == side:
                raise AccountingError("hedge/rollback must reverse initiating side")
            if role == "hedge" and initiating.market == market:
                raise AccountingError("hedge must execute on the opposite market")
            if role == "rollback" and initiating.market != market:
                raise AccountingError("rollback must execute on the initiating market")
            consumed = self._initiating_consumed_share_equivalent.get(source_id, 0)
            if allocated > initiating.share_equivalent - consumed:
                raise AccountingError(
                    "initiating execution allocation exceeds unconsumed quantity"
                )
            if initiating.market == "future":
                numerator = allocated * initiating.contracts
                if numerator % initiating.share_equivalent != 0:
                    raise AccountingError(
                        "initiating future allocation is not whole-contract integral"
                    )
                allocated_future_contracts += numerator // initiating.share_equivalent
            allocated_total += allocated
            normalized.append(InitiatingExecutionAllocation(source_id, allocated))
        if allocated_total != share_equivalent:
            raise AccountingError(
                "initiating execution allocations must sum to leg share_equivalent"
            )
        canonical = self._canonical_initiating_prefix(
            position_id=position_id,
            market=market,
            side=side,
            role=role,
            cursor=cursor,
            share_equivalent=share_equivalent,
        )
        if tuple(normalized) != canonical:
            raise AccountingError(
                "initiating allocations must equal the canonical earliest "
                "unconsumed eligible source prefix"
            )
        if (
            role == "rollback"
            and market == "future"
            and allocated_future_contracts != contracts
        ):
            raise AccountingError(
                "future rollback contracts must match initiating allocations"
            )
        primary_id = normalized[0].execution_id if len(normalized) == 1 else None
        return _ValidatedLinkage(primary_id, tuple(normalized))

    def _canonical_initiating_prefix(
        self,
        *,
        position_id: str,
        market: Market,
        side: Side,
        role: Literal["hedge", "rollback"],
        cursor: EventCursor,
        share_equivalent: int,
    ) -> tuple[InitiatingExecutionAllocation, ...]:
        """Return the unique FIFO source prefix for a linked execution."""

        candidates = sorted(
            (
                execution
                for execution in self._rows
                if execution.position_id == position_id
                and execution.cursor < cursor
                and execution.role in ("normal", "forced")
                and execution.side != side
                and (
                    execution.market != market
                    if role == "hedge"
                    else execution.market == market
                )
                and self._initiating_consumed_share_equivalent.get(
                    execution.execution_id, 0
                )
                < execution.share_equivalent
            ),
            key=lambda execution: (execution.cursor, execution.execution_id),
        )
        remaining = share_equivalent
        prefix: list[InitiatingExecutionAllocation] = []
        for execution in candidates:
            consumed = self._initiating_consumed_share_equivalent.get(
                execution.execution_id, 0
            )
            available = execution.share_equivalent - consumed
            allocated = min(remaining, available)
            if execution.market == "future":
                shares_per_contract = execution.share_equivalent // execution.contracts
                if allocated % shares_per_contract != 0:
                    raise AccountingError(
                        "canonical initiating future allocation would split a contract"
                    )
            prefix.append(
                InitiatingExecutionAllocation(execution.execution_id, allocated)
            )
            remaining -= allocated
            if remaining == 0:
                return tuple(prefix)
        raise AccountingError(
            "insufficient canonical unconsumed initiating execution quantity"
        )

    def _validate_rollback_lot_linkage(
        self,
        *,
        role: ExecutionRole,
        opened_lot_id: str | None,
        linkage: _ValidatedLinkage,
        allocations: Sequence[LotAllocation],
    ) -> None:
        """Keep rollback source slices and the inventory transition identical."""

        if role != "rollback":
            return
        sources = tuple(
            self._executions_by_id[allocation.execution_id]
            for allocation in linkage.allocations
        )
        if opened_lot_id is not None:
            if allocations:
                raise AccountingError("rollback cannot both open and close inventory")
            if any(source.opened_lot_id is not None for source in sources):
                raise AccountingError(
                    "rollback opening must reverse initiating close executions"
                )
            return
        expected = tuple(
            (source.opened_lot_id, allocation.share_equivalent)
            for source, allocation in zip(sources, linkage.allocations, strict=True)
        )
        if any(lot_id is None for lot_id, _ in expected):
            raise AccountingError(
                "rollback close must reverse initiating opened inventory lots"
            )
        actual = tuple(
            (allocation.lot_id, allocation.share_equivalent)
            for allocation in allocations
        )
        if actual != expected:
            raise AccountingError(
                "rollback initiating allocations disagree with FIFO lot closure"
            )

    def _validate_fact_common(
        self,
        fact_id: str,
        event_date: str,
        cursor: EventCursor,
    ) -> None:
        fact_id = _identifier(fact_id, "fact_id")
        parsed = _valid_yyyymmdd(event_date, "event_date")
        if fact_id in self._fact_ids:
            raise AccountingError(f"duplicate accounting fact id: {fact_id}")
        if not isinstance(cursor, EventCursor):
            raise AccountingError("cursor must be EventCursor")
        if self._last_cursor is not None and cursor <= self._last_cursor:
            raise AccountingError("accounting fact cursor must be strictly increasing")
        if self._last_date is not None:
            last = _valid_yyyymmdd(self._last_date, "last_date")
            if parsed < last:
                raise AccountingError("accounting event date is out of order")
        if self.cursor_date_resolver is not None:
            resolved = self.cursor_date_resolver(cursor)
            _valid_yyyymmdd(resolved, "resolved cursor date")
            if resolved != event_date:
                raise AccountingError(
                    "event date disagrees with cursor calendar resolver"
                )

    def _build_execution(self, **values: object) -> ExecutedLeg:
        position_id = str(values["position_id"])
        meta = values.pop("meta")
        if not isinstance(meta, _PositionMeta):
            raise AccountingError("internal position metadata is invalid")
        commission = float(values.pop("commission"))
        tax = float(values.pop("tax"))
        opened_lot_id = values.pop("opened_lot_id")
        allocations = tuple(values.pop("allocations"))
        signed_cashflow = float(values.pop("signed_cashflow"))
        realized = float(values.pop("realized"))
        spot, future_contracts, future_equivalent = self.inventory(position_id)
        return ExecutedLeg(
            sequence=len(self._facts) + 1,
            **values,
            value_code=meta.value_code,
            scenario_id=meta.scenario_id,
            capacity_id=meta.capacity_id,
            signed_cashflow_twd=signed_cashflow,
            realized_pnl_twd=realized,
            commission_twd=commission,
            tax_twd=tax,
            total_cost_twd=commission + tax,
            opened_lot_id=opened_lot_id,
            opened_acquisition_date=(
                str(values["execution_date"]) if opened_lot_id else None
            ),
            allocations=allocations,
            spot_shares_after=spot,
            future_contracts_after=future_contracts,
            future_share_equivalent_after=future_equivalent,
            cost_profile_id=self.profile.profile_id,
        )

    def _commit_execution(self, leg: ExecutedLeg) -> None:
        for allocation in leg.initiating_execution_allocations:
            self._initiating_consumed_share_equivalent[allocation.execution_id] = (
                self._initiating_consumed_share_equivalent.get(
                    allocation.execution_id, 0
                )
                + allocation.share_equivalent
            )
        self._rows.append(leg)
        self._executions_by_id[leg.execution_id] = leg
        self._position_last_date[leg.position_id] = leg.execution_date
        self._commit_fact(leg, leg.execution_id, leg.execution_date, leg.cursor)

    def _commit_fact(
        self,
        fact: AccountingFact,
        fact_id: str,
        event_date: str,
        cursor: EventCursor,
    ) -> None:
        self._facts.append(fact)
        self._fact_ids.add(fact_id)
        self._last_cursor = cursor
        self._last_date = event_date

    def _required_open_position(self, position_id: str) -> _PositionMeta:
        meta = self._positions.get(position_id)
        if meta is None:
            raise AccountingError(f"unknown position_id: {position_id}")
        if position_id in self._terminal_by_position:
            raise AccountingError("position already has a terminal fact")
        return meta

    def _required_exact_pair(self, position_id: str) -> tuple[int, int, int]:
        meta = self._positions[position_id]
        key = (meta.value_code, meta.scenario_id)
        future_lots = [
            lot
            for lot in self._future_books.get(key, ())
            if lot.position_id == position_id
        ]
        spot, future_contracts, future_equivalent = self.inventory(position_id)
        if (
            spot <= 0
            or future_contracts >= 0
            or future_equivalent >= 0
            or any(lot.side != "sell" for lot in future_lots)
            or spot != -future_equivalent
        ):
            raise AccountingError(
                "position establishment requires exactly paired long spot and short future"
            )
        return spot, -future_contracts, -future_equivalent

    def _validate_exit_fifo_allocation(
        self,
        position_id: str,
        exit_cursor: EventCursor,
    ) -> None:
        """Keep forced first legs in canonical whole-position FIFO order."""

        establishment = self._establishment_by_position.get(position_id)
        if establishment is None:
            raise AccountingError("exit cannot allocate an unestablished position")
        if (
            establishment.position_established_ns > exit_cursor.recv_time_ns
            or establishment.cursor >= exit_cursor
        ):
            raise AccountingError("exit cannot precede position establishment")
        meta = self._positions[position_id]
        eligible_keys = [
            fact.fifo_key
            for candidate_id, fact in self._establishment_by_position.items()
            if self._positions[candidate_id].value_code == meta.value_code
            and self._positions[candidate_id].scenario_id == meta.scenario_id
            and candidate_id not in self._terminal_by_position
            and not self._position_is_exactly_flat(candidate_id)
        ]
        if not eligible_keys or establishment.fifo_key != min(eligible_keys):
            raise AccountingError(
                "exit skips canonical (position_established_ns, position_id) FIFO"
            )

    def _validate_normal_exit_leg_fifo(
        self,
        position_id: str,
        exit_cursor: EventCursor,
        *,
        market: Market,
        closing_side: Side,
    ) -> None:
        """Allocate an initiating normal exit in FIFO order for its market leg.

        One aggregate maker fill can be split across multiple positions before
        its linked taker hedges execute.  Consequently, an already-consumed
        Spot leg must not remain ahead merely because its Future leg is still
        open, and vice versa.  Hedge linkage separately proves which initiating
        execution authorized a second leg; rollback and linked hedge legs never
        enter this gate.  Forced first legs deliberately continue to use the
        whole-position gate above.
        """

        establishment = self._establishment_by_position.get(position_id)
        if establishment is None:
            raise AccountingError("exit cannot allocate an unestablished position")
        if (
            establishment.position_established_ns > exit_cursor.recv_time_ns
            or establishment.cursor >= exit_cursor
        ):
            raise AccountingError("exit cannot precede position establishment")
        meta = self._positions[position_id]
        eligible_keys = [
            fact.fifo_key
            for candidate_id, fact in self._establishment_by_position.items()
            if self._positions[candidate_id].value_code == meta.value_code
            and self._positions[candidate_id].scenario_id == meta.scenario_id
            and candidate_id not in self._terminal_by_position
            and self._position_has_closable_market_inventory(
                candidate_id,
                market=market,
                closing_side=closing_side,
            )
        ]
        if not eligible_keys or establishment.fifo_key != min(eligible_keys):
            raise AccountingError(
                "exit skips canonical (position_established_ns, position_id) "
                f"{market}-leg FIFO"
            )

    def _position_has_closable_market_inventory(
        self,
        position_id: str,
        *,
        market: Market,
        closing_side: Side,
    ) -> bool:
        meta = self._positions[position_id]
        key = (meta.value_code, meta.scenario_id)
        if market == "spot":
            return closing_side == "sell" and any(
                lot.position_id == position_id for lot in self._spot_books.get(key, ())
            )
        return any(
            lot.position_id == position_id and lot.side != closing_side
            for lot in self._future_books.get(key, ())
        )

    def _position_is_exactly_flat(self, position_id: str) -> bool:
        meta = self._positions[position_id]
        key = (meta.value_code, meta.scenario_id)
        return not any(
            lot.position_id == position_id for lot in self._spot_books.get(key, ())
        ) and not any(
            lot.position_id == position_id for lot in self._future_books.get(key, ())
        )

    def _position_actual_totals(
        self, position_id: str
    ) -> tuple[int, float, float, float, float]:
        rows = [row for row in self._rows if row.position_id == position_id]
        if not rows:
            raise AccountingError("position has no executed legs")
        return (
            len(rows),
            math.fsum(row.signed_cashflow_twd for row in rows if row.market == "spot"),
            math.fsum(row.realized_pnl_twd for row in rows if row.market == "future"),
            math.fsum(row.commission_twd for row in rows),
            math.fsum(row.tax_twd for row in rows),
        )

    def _position_execution_truth(self, position_id: str) -> ExecutionTruth:
        return (
            "approximate"
            if any(
                row.position_id == position_id and row.execution_truth == "approximate"
                for row in self._rows
            )
            else "exact"
        )


def replay_accounting_facts(
    facts: Sequence[AccountingFact],
    *,
    profile: TransactionCostProfile | None = None,
    cursor_date_resolver: CursorDateResolver | None = None,
    require_route_roles: bool = False,
) -> S1AccountingLedger:
    ledger = S1AccountingLedger(
        profile,
        cursor_date_resolver=cursor_date_resolver,
        require_route_roles=require_route_roles,
    )
    for expected_sequence, row in enumerate(facts, start=1):
        if row.sequence != expected_sequence:
            raise AccountingReplayError("accounting sequence is not contiguous")
        try:
            if isinstance(row, ExecutedLeg):
                common = {
                    "execution_id": row.execution_id,
                    "position_id": row.position_id,
                    "value_code": row.value_code,
                    "scenario_id": row.scenario_id,
                    "capacity_id": row.capacity_id,
                    "side": row.side,
                    "role": row.role,
                    "execution_truth": row.execution_truth,
                    "execution_source_id": row.execution_source_id,
                    "request_id": row.request_id,
                    "hedge_intent_id": row.hedge_intent_id,
                    "initiating_execution_id": row.initiating_execution_id,
                    "initiating_execution_allocations": (
                        row.initiating_execution_allocations
                    ),
                    "execution_date": row.execution_date,
                    "cursor": row.cursor,
                    "price": row.price,
                    "route_role": row.route_role,
                }
                if row.market == "spot":
                    actual: AccountingFact = ledger.record_spot_execution(
                        **common, shares=row.shares
                    )
                elif row.market == "future":
                    actual = ledger.record_future_execution(
                        **common,
                        contracts=row.contracts,
                        share_equivalent=row.share_equivalent,
                    )
                else:
                    raise AccountingReplayError(f"invalid market: {row.market}")
            elif isinstance(row, PositionEstablishedFact):
                actual = ledger.establish_position(
                    establishment_id=row.establishment_id,
                    position_id=row.position_id,
                    establishment_date=row.establishment_date,
                    cursor=row.cursor,
                    position_established_ns=row.position_established_ns,
                    capacity_transition_id=row.capacity_transition_id,
                )
            elif isinstance(row, TerminalRealizedAccounting):
                actual = ledger.seal_terminal(
                    terminal_id=row.terminal_id,
                    position_id=row.position_id,
                    terminal_date=row.terminal_date,
                    cursor=row.cursor,
                    terminal_outcome=row.terminal_outcome,
                    capacity_release_transition_id=(row.capacity_release_transition_id),
                )
            elif isinstance(row, ExpiryAccountingMark):
                actual = ledger.record_expiry_accounting_mark(
                    mark_id=row.mark_id,
                    position_id=row.position_id,
                    expiry_date=row.expiry_date,
                    cursor=row.cursor,
                    spot_close_source_id=row.spot_close_source_id,
                    spot_close_source_cursor=row.spot_close_source_cursor,
                    spot_close_price=row.spot_close_price,
                    capacity_release_transition_id=(row.capacity_release_transition_id),
                )
            else:
                raise AccountingReplayError(
                    f"unsupported accounting fact: {type(row).__name__}"
                )
        except AccountingError as error:
            raise AccountingReplayError(str(error)) from error
        _assert_replay_equal(actual, row, path=f"fact[{expected_sequence}]")
    return ledger


def replay_accounting_legs(
    rows: Sequence[ExecutedLeg],
    *,
    profile: TransactionCostProfile | None = None,
    cursor_date_resolver: CursorDateResolver | None = None,
    require_route_roles: bool = False,
) -> S1AccountingLedger:
    """Backward-compatible execution-only replay; canonical replay uses facts."""

    return replay_accounting_facts(
        rows,
        profile=profile,
        cursor_date_resolver=cursor_date_resolver,
        require_route_roles=require_route_roles,
    )


def encode_accounting_fact(fact: AccountingFact) -> dict[str, object]:
    """Encode one accounting fact as a canonical JSON-safe object record."""

    if type(fact) not in _FACT_TYPE_BY_CLASS:
        raise TypeError("fact must be a supported accounting fact")
    return fact.as_dict()


def encode_accounting_facts(
    facts: Sequence[AccountingFact],
) -> list[dict[str, object]]:
    """Encode an ordered fact stream as a JSON-safe array of object records."""

    if isinstance(facts, (str, bytes)) or not isinstance(facts, Sequence):
        raise TypeError("facts must be a sequence")
    return [encode_accounting_fact(fact) for fact in facts]


def decode_accounting_fact(record: Mapping[str, object]) -> AccountingFact:
    """Strictly decode one canonical JSON object into its immutable fact type."""

    if not isinstance(record, Mapping):
        raise TypeError("accounting fact record must be a mapping")
    if "fact_type" not in record:
        raise AccountingCodecError("accounting fact record is missing fact_type")
    fact_type = record["fact_type"]
    if not isinstance(fact_type, str):
        raise AccountingCodecError("fact_type must be a string")
    try:
        fact_class = _FACT_CLASS_BY_TYPE[fact_type]
    except KeyError as error:
        raise AccountingCodecError(f"unknown fact_type: {fact_type}") from error

    expected_keys = _fact_record_keys(fact_class)
    actual_keys = set(record)
    missing = sorted(expected_keys.difference(actual_keys))
    unknown = sorted(actual_keys.difference(expected_keys))
    if missing or unknown:
        raise AccountingCodecError(
            f"{fact_type} record schema mismatch: missing={missing}, unknown={unknown}"
        )

    values = dict(record)
    values.pop("fact_type")
    values["cursor"] = _decode_record_cursor(values, prefix="")
    if fact_class is ExpiryAccountingMark:
        values["spot_close_source_cursor"] = _decode_record_cursor(
            values,
            prefix="spot_close_source_",
        )
    if fact_class is PositionEstablishedFact:
        fifo_position_id = values.pop("fifo_position_id")
        if not isinstance(fifo_position_id, str):
            raise AccountingCodecError("fifo_position_id must be a string")
        if fifo_position_id != values["position_id"]:
            raise AccountingCodecError("fifo_position_id disagrees with position_id")

    for field_name, item_class in _ALLOCATION_FIELDS_BY_CLASS.get(
        fact_class, {}
    ).items():
        values[field_name] = _decode_dataclass_array(
            values[field_name],
            item_class,
            path=field_name,
        )

    hints = get_type_hints(fact_class)
    for field in fields(fact_class):
        value = values[field.name]
        if not _matches_decoded_type(value, hints[field.name]):
            raise AccountingCodecError(
                f"{fact_type}.{field.name} has an invalid JSON value type"
            )
    return fact_class(**values)


def decode_accounting_facts(
    records: Sequence[Mapping[str, object]],
) -> tuple[AccountingFact, ...]:
    """Decode an ordered mixed-type JSON fact stream without reordering it."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("accounting fact records must be a sequence")
    return tuple(decode_accounting_fact(record) for record in records)


def _fact_as_dict(fact: AccountingFact) -> dict[str, object]:
    row = asdict(fact)
    row.pop("cursor")
    row["recv_time_ns"] = fact.cursor.recv_time_ns
    row["event_sequence"] = fact.cursor.event_sequence
    row["row_index"] = fact.cursor.row_index
    if isinstance(fact, ExpiryAccountingMark):
        row.pop("spot_close_source_cursor")
        row["spot_close_source_recv_time_ns"] = (
            fact.spot_close_source_cursor.recv_time_ns
        )
        row["spot_close_source_event_sequence"] = (
            fact.spot_close_source_cursor.event_sequence
        )
        row["spot_close_source_row_index"] = fact.spot_close_source_cursor.row_index
    for field_name in _ALLOCATION_FIELDS_BY_CLASS.get(type(fact), {}):
        row[field_name] = list(row[field_name])
    return {"fact_type": _FACT_TYPE_BY_CLASS[type(fact)], **row}


def _fact_record_keys(fact_class: type[AccountingFact]) -> set[str]:
    result = {field.name for field in fields(fact_class)}
    result.remove("cursor")
    result.update(_CURSOR_RECORD_FIELDS)
    if fact_class is ExpiryAccountingMark:
        result.remove("spot_close_source_cursor")
        result.update(_SPOT_CLOSE_CURSOR_RECORD_FIELDS)
    if fact_class is PositionEstablishedFact:
        result.add("fifo_position_id")
    result.add("fact_type")
    return result


def _decode_record_cursor(
    values: dict[str, object],
    *,
    prefix: str,
) -> EventCursor:
    names = (
        f"{prefix}recv_time_ns",
        f"{prefix}event_sequence",
        f"{prefix}row_index",
    )
    components = tuple(values.pop(name) for name in names)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in components
    ):
        raise AccountingCodecError(f"{prefix or 'fact_'}cursor fields must be integers")
    try:
        return EventCursor(*components)
    except ValueError as error:
        raise AccountingCodecError(
            f"{prefix or 'fact_'}cursor fields are invalid"
        ) from error


def _decode_dataclass_array[ItemT](
    value: object,
    item_class: type[ItemT],
    *,
    path: str,
) -> tuple[ItemT, ...]:
    if not isinstance(value, list):
        raise AccountingCodecError(f"{path} must be a JSON array")
    expected_keys = {field.name for field in fields(item_class)}
    hints = get_type_hints(item_class)
    decoded: list[ItemT] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AccountingCodecError(f"{path}[{index}] must be a JSON object")
        actual_keys = set(item)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys.difference(actual_keys))
            unknown = sorted(actual_keys.difference(expected_keys))
            raise AccountingCodecError(
                f"{path}[{index}] schema mismatch: missing={missing}, unknown={unknown}"
            )
        values = dict(item)
        for field_name, field_value in values.items():
            if not _matches_decoded_type(field_value, hints[field_name]):
                raise AccountingCodecError(
                    f"{path}[{index}].{field_name} has an invalid JSON value type"
                )
        decoded.append(item_class(**values))
    return tuple(decoded)


def _matches_decoded_type(value: object, expected: object) -> bool:
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is Literal:
        return any(
            type(value) is type(option) and value == option for option in arguments
        )
    if origin in (Union, UnionType):
        return any(_matches_decoded_type(value, option) for option in arguments)
    if origin is tuple:
        return (
            isinstance(value, tuple)
            and len(arguments) == 2
            and arguments[1] is Ellipsis
            and all(_matches_decoded_type(item, arguments[0]) for item in value)
        )
    if expected is float:
        return type(value) is float and math.isfinite(value)
    if expected in (str, int, bool, type(None)):
        return type(value) is expected
    return type(value) is expected


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AccountingError(f"{name} must be a nonempty canonical string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AccountingError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise AccountingError(f"{name} must be a positive integer")
    return result


def _positive_price(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise AccountingError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AccountingError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise AccountingError(f"{name} must be finite and positive")
    return result


def _valid_yyyymmdd(value: object, name: str) -> date:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise AccountingError(f"{name} must be a valid YYYYMMDD date")
    try:
        return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise AccountingError(f"{name} must be a valid YYYYMMDD date") from error


def _require_finite_money(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise AccountingError(f"derived {name} must be finite")


def _assert_replay_equal(actual: object, expected: object, *, path: str) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not isinstance(
            expected, (int, float)
        ):
            raise AccountingReplayError(f"{path} differs from deterministic replay")
        actual_float = float(actual)
        expected_float = float(expected)
        if (
            not math.isfinite(actual_float)
            or not math.isfinite(expected_float)
            or not math.isclose(
                actual_float,
                expected_float,
                rel_tol=1e-15,
                abs_tol=1e-12,
            )
        ):
            raise AccountingReplayError(f"{path} differs from deterministic replay")
        return
    if is_dataclass(actual) or is_dataclass(expected):
        if type(actual) is not type(expected) or not is_dataclass(actual):
            raise AccountingReplayError(f"{path} differs from deterministic replay")
        for field in fields(actual):
            _assert_replay_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or not isinstance(
            expected, (tuple, list)
        ):
            raise AccountingReplayError(f"{path} differs from deterministic replay")
        if len(actual) != len(expected):
            raise AccountingReplayError(f"{path} differs from deterministic replay")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_replay_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if actual != expected:
        raise AccountingReplayError(f"{path} differs from deterministic replay")
