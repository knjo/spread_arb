"""Typed bridge from S1 loop executions to replayable portfolio accounting.

The chronological loop owns request, execution, and capacity timing.  This
bridge owns the accounting ledger and performs the deliberately boring but
important semantic translation from route-specific execution roles to the
generic accounting roles.  Keeping the translation in one place prevents a
report builder from silently reconstructing PnL with different quantities,
dates, or linkage than the replay that generated the executions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from .layered import EventCursor
from .s1_accounting import (
    AccountingFact,
    ExecutableTerminalOutcome,
    ExecutedLeg,
    ExpiryAccountingMark,
    InitiatingExecutionAllocation,
    PositionEstablishedFact,
    S1AccountingLedger,
    TerminalRealizedAccounting,
    replay_accounting_facts,
)
from .transaction_costs import TransactionCostProfile

ExecutionDateResolver = Callable[[EventCursor], str]

_ROLE_MAP: Final = {
    "entry_maker": "normal",
    "entry_hedge": "hedge",
    "entry_rollback": "rollback",
    "exit_maker": "normal",
    "exit_hedge": "hedge",
    "exit_rollback": "rollback",
}


class S1ExecutionLike(Protocol):
    execution_id: str
    position_id: str
    product_id: str
    capacity_id: str
    request_id: str | None
    role: str
    market: str
    side: str
    cursor: EventCursor
    price: float
    quantity: int
    quantity_unit: str
    execution_truth: str
    execution_source_id: str


@dataclass(frozen=True, slots=True)
class S1AccountingProduct:
    product_id: str
    value_code: str
    contract_size_shares: int

    def __post_init__(self) -> None:
        _identifier(self.product_id, "product_id")
        _identifier(self.value_code, "value_code")
        _positive_integer(self.contract_size_shares, "contract_size_shares")


class S1AccountingBridge:
    """Record loop facts once and expose verifier-backed accounting rows."""

    def __init__(
        self,
        *,
        default_date: str,
        scenario_id: str,
        products: Sequence[S1AccountingProduct],
        profile: TransactionCostProfile | None = None,
        execution_date_resolver: ExecutionDateResolver | None = None,
    ) -> None:
        self.default_date = _yyyymmdd(default_date, "default_date")
        self.scenario_id = _identifier(scenario_id, "scenario_id")
        product_values = tuple(products)
        if not product_values or any(
            not isinstance(product, S1AccountingProduct) for product in product_values
        ):
            raise TypeError("products must contain S1AccountingProduct values")
        if len({product.product_id for product in product_values}) != len(
            product_values
        ):
            raise ValueError("accounting product_id values must be unique")
        if execution_date_resolver is not None and not callable(
            execution_date_resolver
        ):
            raise TypeError("execution_date_resolver must be callable or None")
        self.products: Mapping[str, S1AccountingProduct] = {
            product.product_id: product for product in product_values
        }
        self._date_resolver = execution_date_resolver or (
            lambda _cursor: self.default_date
        )
        self.ledger = S1AccountingLedger(
            profile,
            cursor_date_resolver=self._resolved_date,
            require_route_roles=True,
        )
        self._loop_execution_ids: set[str] = set()
        self._position_bindings: dict[str, tuple[str, str]] = {}
        self._position_truth: dict[str, str] = {}
        self._execution_roles_by_position: dict[str, list[str]] = {}
        self._established_positions: set[str] = set()

    @classmethod
    def from_facts(
        cls,
        *,
        default_date: str,
        scenario_id: str,
        products: Sequence[S1AccountingProduct],
        facts: Sequence[AccountingFact],
        profile: TransactionCostProfile | None = None,
        execution_date_resolver: ExecutionDateResolver | None = None,
    ) -> S1AccountingBridge:
        """Resume a bridge from its verified append-only accounting facts.

        Accounting facts intentionally store the stable ``value_code`` rather
        than the runner's ``product_id``.  Resumption therefore requires a
        one-to-one configured value mapping; accepting an ambiguous mapping
        would make the rebuilt capacity binding depend on input order.
        """

        if isinstance(facts, (str, bytes)) or not isinstance(facts, Sequence):
            raise TypeError("facts must be a sequence of accounting facts")
        serialized = tuple(facts)
        bridge = cls(
            default_date=default_date,
            scenario_id=scenario_id,
            products=products,
            profile=profile,
            execution_date_resolver=execution_date_resolver,
        )

        products_by_value: dict[str, S1AccountingProduct] = {}
        for product in bridge.products.values():
            if product.value_code in products_by_value:
                raise ValueError(
                    "resume products must define a one-to-one value_code mapping"
                )
            products_by_value[product.value_code] = product

        loop_execution_ids: set[str] = set()
        position_bindings: dict[str, tuple[str, str]] = {}
        position_truth: dict[str, str] = {}
        roles_by_position: dict[str, list[str]] = {}
        established_positions: set[str] = set()
        supported_types = (
            ExecutedLeg,
            PositionEstablishedFact,
            TerminalRealizedAccounting,
            ExpiryAccountingMark,
        )
        for fact in serialized:
            if not isinstance(fact, supported_types):
                raise TypeError("facts contains an unsupported accounting fact")
            if fact.scenario_id != bridge.scenario_id:
                raise ValueError("accounting fact scenario_id disagrees with bridge")
            try:
                product = products_by_value[fact.value_code]
            except KeyError as error:
                raise ValueError(
                    "accounting fact value_code has no configured product mapping"
                ) from error

            binding = (product.product_id, fact.capacity_id)
            prior_binding = position_bindings.get(fact.position_id)
            if prior_binding is not None and prior_binding != binding:
                raise ValueError(
                    "accounting facts change a position product/capacity binding"
                )
            position_bindings.setdefault(fact.position_id, binding)

            if isinstance(fact, ExecutedLeg):
                if fact.cost_profile_id != bridge.ledger.profile.profile_id:
                    raise ValueError(
                        "accounting fact cost profile disagrees with bridge"
                    )
                if fact.route_role not in _ROLE_MAP:
                    raise ValueError(
                        "accounting execution has an unsupported S1 route_role"
                    )
                if _ROLE_MAP[fact.route_role] != fact.role:
                    raise ValueError(
                        "accounting execution route_role disagrees with generic role"
                    )
                if fact.market == "future" and fact.share_equivalent != (
                    fact.contracts * product.contract_size_shares
                ):
                    raise ValueError(
                        "accounting future execution disagrees with product contract size"
                    )
                loop_execution_ids.add(fact.execution_id)
                prior_truth = position_truth.get(fact.position_id, "exact")
                position_truth[fact.position_id] = (
                    "approximate"
                    if "approximate" in (prior_truth, fact.execution_truth)
                    else "exact"
                )
                roles_by_position.setdefault(fact.position_id, []).append(
                    fact.route_role
                )
            elif isinstance(fact, PositionEstablishedFact):
                established_positions.add(fact.position_id)

        bridge.ledger = replay_accounting_facts(
            serialized,
            profile=bridge.ledger.profile,
            cursor_date_resolver=bridge._resolved_date,
            require_route_roles=True,
        )
        bridge._loop_execution_ids = loop_execution_ids
        bridge._position_bindings = position_bindings
        bridge._position_truth = position_truth
        bridge._execution_roles_by_position = roles_by_position
        bridge._established_positions = established_positions
        return bridge.verify()

    @property
    def facts(self) -> tuple[AccountingFact, ...]:
        return self.ledger.facts

    @property
    def fact_count(self) -> int:
        """Return the append-only checkpoint index without copying facts."""

        return self.ledger.fact_count

    def facts_since(self, index: int) -> tuple[AccountingFact, ...]:
        """Return only facts appended since a captured ``fact_count``."""

        return self.ledger.facts_since(index)

    def record_execution(self, fact: S1ExecutionLike) -> ExecutedLeg:
        """Translate and record one chronological S1 execution exactly once."""

        if not _looks_like_execution(fact):
            raise TypeError("fact must satisfy the S1 execution contract")
        if fact.execution_id in self._loop_execution_ids:
            raise ValueError("loop execution_id was already recorded")
        try:
            product = self.products[fact.product_id]
        except KeyError as error:
            raise ValueError("execution references an unknown product_id") from error
        binding = (fact.product_id, fact.capacity_id)
        prior_binding = self._position_bindings.get(fact.position_id)
        if prior_binding is not None and prior_binding != binding:
            raise ValueError("position product/capacity binding changed")
        try:
            accounting_role = _ROLE_MAP[fact.role]
        except KeyError as error:
            raise ValueError("unsupported S1 execution role") from error

        allocations = _initiating_allocations(fact)
        initiating_execution_id = (
            allocations[0].execution_id if len(allocations) == 1 else None
        )
        linked = accounting_role in ("hedge", "rollback")
        if linked and not allocations:
            raise ValueError(
                "hedge/rollback execution requires exact source allocations"
            )
        if not linked and allocations:
            raise ValueError(
                "normal maker execution cannot cite initiating allocations"
            )
        hedge_intent_id = (
            _optional_identifier(
                getattr(fact, "hedge_intent_id", None),
                "hedge_intent_id",
            )
            if linked
            else None
        )
        if linked and hedge_intent_id is None:
            hedge_intent_id = _optional_identifier(fact.request_id, "request_id")
        if linked and hedge_intent_id is None:
            raise ValueError("hedge/rollback execution requires a request identity")
        execution_date = self._resolved_date(fact.cursor)

        common = {
            "execution_id": fact.execution_id,
            "position_id": fact.position_id,
            "value_code": product.value_code,
            "scenario_id": self.scenario_id,
            "capacity_id": fact.capacity_id,
            "side": fact.side,
            "role": accounting_role,
            "execution_truth": fact.execution_truth,
            "execution_source_id": fact.execution_source_id,
            "request_id": fact.request_id,
            "hedge_intent_id": hedge_intent_id,
            "initiating_execution_id": initiating_execution_id,
            "initiating_execution_allocations": allocations,
            "execution_date": execution_date,
            "cursor": fact.cursor,
            "price": fact.price,
            "route_role": fact.role,
        }
        if fact.market == "spot":
            if fact.quantity_unit != "spot_shares":
                raise ValueError("spot execution must use spot_shares")
            row = self.ledger.record_spot_execution(
                **common,
                shares=fact.quantity,
            )
        elif fact.market == "future":
            if fact.quantity_unit != "future_contracts":
                raise ValueError("future execution must use future_contracts")
            row = self.ledger.record_future_execution(
                **common,
                contracts=fact.quantity,
                share_equivalent=(fact.quantity * product.contract_size_shares),
            )
        else:
            raise ValueError("execution market must be spot or future")
        self._loop_execution_ids.add(fact.execution_id)
        self._position_bindings.setdefault(fact.position_id, binding)
        prior_truth = self._position_truth.get(fact.position_id, "exact")
        self._position_truth[fact.position_id] = (
            "approximate"
            if "approximate" in (prior_truth, fact.execution_truth)
            else "exact"
        )
        self._execution_roles_by_position.setdefault(fact.position_id, []).append(
            fact.role
        )
        return row

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
    ) -> PositionEstablishedFact:
        self._product(product_id)
        if self._position_bindings.get(position_id) != (product_id, capacity_id):
            raise ValueError("position establishment disagrees with execution binding")
        if self._position_truth.get(position_id) != execution_truth:
            raise ValueError("position establishment truth disagrees with executions")
        fact = self.ledger.establish_position(
            establishment_id=establishment_id,
            position_id=position_id,
            establishment_date=self._resolved_date(cursor),
            cursor=cursor,
            position_established_ns=cursor.recv_time_ns,
            capacity_transition_id=capacity_transition_id,
        )
        if fact.execution_truth != execution_truth:
            raise RuntimeError("accounting establishment truth invariant failed")
        self._established_positions.add(position_id)
        return fact

    def seal_terminal(
        self,
        *,
        terminal_id: str,
        position_id: str,
        cursor: EventCursor,
        terminal_outcome: ExecutableTerminalOutcome,
        capacity_release_transition_id: str,
    ) -> TerminalRealizedAccounting:
        roles = tuple(self._execution_roles_by_position.get(position_id, ()))
        if terminal_outcome == "exit_maker_flat":
            if position_id not in self._established_positions:
                raise ValueError("exit_maker_flat requires an established position")
            if "exit_maker" not in roles or "exit_hedge" not in roles:
                raise ValueError(
                    "exit_maker_flat requires normal exit maker and hedge legs"
                )
        elif terminal_outcome in (
            "entry_emergency_rollback_flat",
            "entry_partial_rollback_flat",
        ):
            if position_id in self._established_positions:
                raise ValueError("entry rollback terminal cannot follow establishment")
            if "entry_maker" not in roles or "entry_rollback" not in roles:
                raise ValueError(
                    "entry rollback terminal requires maker and rollback legs"
                )
        return self.ledger.seal_terminal(
            terminal_id=terminal_id,
            position_id=position_id,
            terminal_date=self._resolved_date(cursor),
            cursor=cursor,
            terminal_outcome=terminal_outcome,
            capacity_release_transition_id=capacity_release_transition_id,
        )

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
    ) -> ExpiryAccountingMark:
        return self.ledger.record_expiry_accounting_mark(
            mark_id=mark_id,
            position_id=position_id,
            expiry_date=self._resolved_date(cursor),
            cursor=cursor,
            capacity_release_transition_id=capacity_release_transition_id,
            spot_close_source_id=spot_close_source_id,
            spot_close_source_cursor=spot_close_source_cursor,
            spot_close_price=spot_close_price,
        )

    def verify(self) -> S1AccountingBridge:
        self.ledger.verify()
        accounting_ids = {row.execution_id for row in self.ledger.rows}
        if self._loop_execution_ids != accounting_ids:
            raise RuntimeError("loop execution/accounting identities diverged")
        return self

    def _product(self, product_id: str) -> S1AccountingProduct:
        try:
            return self.products[product_id]
        except KeyError as error:
            raise ValueError("unknown accounting product_id") from error

    def _resolved_date(self, cursor: EventCursor) -> str:
        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        return _yyyymmdd(self._date_resolver(cursor), "resolved execution date")


def _initiating_allocations(
    fact: S1ExecutionLike,
) -> tuple[InitiatingExecutionAllocation, ...]:
    raw = getattr(fact, "initiating_execution_allocations", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("initiating_execution_allocations must be a sequence")
    values = tuple(raw)
    if any(not isinstance(value, InitiatingExecutionAllocation) for value in values):
        raise TypeError("initiating_execution_allocations contains an invalid value")
    return values


def _looks_like_execution(value: object) -> bool:
    required = (
        "execution_id",
        "position_id",
        "product_id",
        "capacity_id",
        "request_id",
        "role",
        "market",
        "side",
        "cursor",
        "price",
        "quantity",
        "quantity_unit",
        "execution_truth",
        "execution_source_id",
    )
    return all(hasattr(value, name) for name in required)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _yyyymmdd(value: object, name: str) -> str:
    text = _identifier(value, name)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{name} must be YYYYMMDD")
    return text


__all__ = [
    "S1AccountingBridge",
    "S1AccountingProduct",
    "S1ExecutionLike",
]
