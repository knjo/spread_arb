"""Date-major multi-day orchestrator for the frozen seven-policy S1 replay.

The one-day runner owns physical market-data preparation and one chronological
``Date x policy`` loop.  This module owns only portfolio continuity:

* days are replayed before policies (date-major order);
* every policy has its own accounting bridge, capacity checkpoint, and carry;
* a checkpoint is restored before each later day so a daily run exposes only
  that day's capacity-transition delta;
* opening carry products remain entry-disabled for the whole session in the
  underlying one-day loop; and
* the complete accounting and capacity ledgers are verified together before a
  result is returned; and
* the production API prepares and releases one physical day at a time.

No files are read or written here.  ``run_s1_portfolio`` preserves the original
in-memory API.  ``run_s1_portfolio_from_dates`` accepts a strict date source and
a pre-frozen accounting catalog so a 71-day production replay never needs 71
``PreparedS1EntryDay`` objects concurrently.  Heavy daily replay results may be
sent to a sink and discarded; immutable summaries, accounting facts, and the
complete capacity transition history remain in memory so final accounting,
capacity, and cross-ledger verification is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final, Protocol
from zoneinfo import ZoneInfo

from .capacity_ledger import (
    CapacityLedger,
    CapacityLedgerSeed,
    CapacityReplayResult,
    CapacityTransition,
    replay_capacity_transitions,
)
from .layered import EventCursor
from .policy_spec import POLICY_IDS
from .s1_accounting import AccountingFact
from .s1_accounting_bridge import S1AccountingBridge, S1AccountingProduct
from .s1_cross_ledger_verifier import (
    S1CrossLedgerReport,
    verify_s1_accounting_capacity_links,
)
from .s1_entry_day_runner import (
    PreparedS1EntryDay,
    S1EntryDayRun,
    S1EntryDaySummary,
    run_s1_entry_policy,
)
from .s1_event_loop import S1CarryContractBinding, S1CarryPosition, S1Product

PORTFOLIO_RUNNER_VERSION: Final = "s1_portfolio_date_major_v2_streaming"
TAIPEI: Final = ZoneInfo("Asia/Taipei")


class S1PolicyDayRunner(Protocol):
    """Injectable one-day boundary used by focused orchestration tests."""

    def __call__(
        self,
        prepared: PreparedS1EntryDay,
        policy_id: str,
        *,
        normal_exit_enabled: bool,
        accounting_adapter: S1AccountingBridge,
        capacity_ledger: CapacityLedger,
        carry_in: Sequence[S1CarryPosition],
        entry_enabled_product_ids: frozenset[str],
    ) -> S1EntryDayRun: ...


@dataclass(frozen=True, slots=True)
class S1PortfolioDayPartition:
    """One sink-ready policy/day partition emitted before a heavy run is dropped."""

    date: str
    policy_id: str
    run: S1EntryDayRun
    accounting_facts: tuple[AccountingFact, ...]
    capacity_transitions: tuple[CapacityTransition, ...]

    def __post_init__(self) -> None:
        if self.run.summary.date != self.date:
            raise ValueError("sink partition run date disagrees with partition")
        if self.run.summary.policy_id != self.policy_id:
            raise ValueError("sink partition policy disagrees with partition")


class S1PortfolioSink(Protocol):
    """Persist one policy/day run and its exact append-only ledger deltas."""

    def __call__(self, partition: S1PortfolioDayPartition) -> None: ...


@dataclass(frozen=True, slots=True)
class S1PortfolioPolicyResult:
    """Verified complete history for one isolated S1 policy."""

    policy_id: str
    daily_runs: tuple[S1EntryDayRun, ...]
    daily_summaries: tuple[S1EntryDaySummary, ...]
    daily_runs_retained: bool
    accounting_facts: tuple[AccountingFact, ...]
    capacity_transitions: tuple[CapacityTransition, ...]
    final_carry: tuple[S1CarryPosition, ...]
    capacity_seed: CapacityLedgerSeed
    cross_ledger_report: S1CrossLedgerReport


@dataclass(frozen=True, slots=True)
class S1PortfolioResult:
    """Immutable seven-policy result in canonical policy order."""

    dates: tuple[str, ...]
    policy_results: tuple[S1PortfolioPolicyResult, ...]
    runner_version: str = PORTFOLIO_RUNNER_VERSION

    def __post_init__(self) -> None:
        if tuple(result.policy_id for result in self.policy_results) != POLICY_IDS:
            raise ValueError("portfolio policy results are not in canonical order")
        for result in self.policy_results:
            summary_dates = tuple(summary.date for summary in result.daily_summaries)
            if summary_dates != self.dates:
                raise ValueError("portfolio daily summaries do not match result dates")
            if result.daily_runs_retained:
                run_dates = tuple(run.summary.date for run in result.daily_runs)
                if run_dates != self.dates:
                    raise ValueError("retained daily runs do not match result dates")
            elif result.daily_runs:
                raise ValueError("daily_runs must be empty when retention is disabled")

    @property
    def by_policy(self) -> Mapping[str, S1PortfolioPolicyResult]:
        return MappingProxyType(
            {result.policy_id: result for result in self.policy_results}
        )


@dataclass(slots=True)
class _PolicyState:
    accounting: S1AccountingBridge
    capacity_seed: CapacityLedgerSeed | None = None
    carry: tuple[S1CarryPosition, ...] = ()
    carry_bindings: dict[str, S1CarryContractBinding] | None = None
    daily_runs: list[S1EntryDayRun] | None = None
    daily_summaries: list[S1EntryDaySummary] | None = None
    capacity_transitions: list[CapacityTransition] | None = None

    def __post_init__(self) -> None:
        self.carry_bindings = {}
        self.daily_runs = []
        self.daily_summaries = []
        self.capacity_transitions = []


def run_s1_portfolio(
    prepared_days: Sequence[PreparedS1EntryDay],
    *,
    policy_runner: S1PolicyDayRunner | None = None,
    retain_daily_runs: bool = True,
    sink: S1PortfolioSink | None = None,
) -> S1PortfolioResult:
    """Replay prepared days in strict date-major, seven-policy order.

    All prepared days are validated before the first replay.  A later day's
    missing or changed carried contract therefore raises rather than silently
    dropping, remapping, or synthetically flattening the position.
    """

    days = _validated_days(prepared_days)
    products = _accounting_product_union(days)
    by_date = {day.date: day for day in days}
    return _run_s1_portfolio_dates(
        tuple(by_date),
        by_date.__getitem__,
        products,
        policy_runner=policy_runner,
        retain_daily_runs=retain_daily_runs,
        sink=sink,
    )


def run_s1_portfolio_from_dates(
    dates: Sequence[str],
    prepare_day: Callable[[str], PreparedS1EntryDay],
    accounting_products: Sequence[S1AccountingProduct],
    *,
    policy_runner: S1PolicyDayRunner | None = None,
    retain_daily_runs: bool = False,
    sink: S1PortfolioSink | None = None,
) -> S1PortfolioResult:
    """Prepare, replay, optionally sink, and release one date at a time.

    ``accounting_products`` is the complete product catalog frozen before the
    replay.  Every prepared day's products must be a subset with exactly the
    same product/value/contract-size identity.  Capacity transition rows are
    deliberately retained even when ``retain_daily_runs`` is false: the final
    result therefore still performs uninterrupted capacity replay, checkpoint
    equivalence, accounting verification, and cross-ledger verification.
    """

    selected_dates = _validated_dates(dates)
    if not callable(prepare_day):
        raise TypeError("prepare_day must be callable")
    products = _validated_accounting_catalog(accounting_products)
    return _run_s1_portfolio_dates(
        selected_dates,
        prepare_day,
        products,
        policy_runner=policy_runner,
        retain_daily_runs=retain_daily_runs,
        sink=sink,
    )


def _run_s1_portfolio_dates(
    dates: tuple[str, ...],
    prepare_day: Callable[[str], PreparedS1EntryDay],
    accounting_products: tuple[S1AccountingProduct, ...],
    *,
    policy_runner: S1PolicyDayRunner | None,
    retain_daily_runs: bool,
    sink: S1PortfolioSink | None,
) -> S1PortfolioResult:
    runner = run_s1_entry_policy if policy_runner is None else policy_runner
    if not callable(runner):
        raise TypeError("policy_runner must be callable or None")
    if not isinstance(retain_daily_runs, bool):
        raise TypeError("retain_daily_runs must be boolean")
    if sink is not None and not callable(sink):
        raise TypeError("sink must be callable or None")
    catalog = {product.product_id: product for product in accounting_products}
    states = {
        policy_id: _PolicyState(
            accounting=S1AccountingBridge(
                default_date=dates[0],
                scenario_id=policy_id,
                products=accounting_products,
                execution_date_resolver=_taipei_execution_date,
            )
        )
        for policy_id in POLICY_IDS
    }

    for date in dates:
        prepared = prepare_day(date)
        _validate_source_day(prepared, date, catalog)
        products_today = {product.product_id: product for product in prepared.products}
        for policy_id in POLICY_IDS:
            state = states[policy_id]
            assert state.carry_bindings is not None
            _validate_opening_carry(
                policy_id=policy_id,
                date=prepared.date,
                carry=state.carry,
                bindings=state.carry_bindings,
                products_today=products_today,
            )
        for policy_id in POLICY_IDS:
            state = states[policy_id]
            assert state.carry_bindings is not None
            assert state.daily_runs is not None
            assert state.daily_summaries is not None
            assert state.capacity_transitions is not None
            ledger = (
                CapacityLedger()
                if state.capacity_seed is None
                else CapacityLedger.from_seed(state.capacity_seed)
            )
            if ledger.transitions:
                raise RuntimeError(
                    "restored capacity ledger did not start a daily delta"
                )

            accounting_start = len(state.accounting.facts)
            run = runner(
                prepared,
                policy_id,
                normal_exit_enabled=True,
                accounting_adapter=state.accounting,
                capacity_ledger=ledger,
                carry_in=state.carry,
                entry_enabled_product_ids=frozenset(prepared.entry_product_ids),
            )
            _validate_daily_run(
                prepared=prepared,
                policy_id=policy_id,
                opening_carry=state.carry,
                ledger=ledger,
                run=run,
            )
            accounting_delta = state.accounting.facts[accounting_start:]
            capacity_delta = ledger.transitions
            partition = S1PortfolioDayPartition(
                date=prepared.date,
                policy_id=policy_id,
                run=run,
                accounting_facts=accounting_delta,
                capacity_transitions=capacity_delta,
            )
            if sink is not None:
                sink(partition)
            state.daily_summaries.append(run.summary)
            if retain_daily_runs:
                state.daily_runs.append(run)
            state.capacity_transitions.extend(capacity_delta)
            state.capacity_seed = ledger.to_seed(prepared.date)
            state.carry = tuple(run.result.carry_out)
            state.carry_bindings = _next_carry_bindings(
                state.carry,
                products_today,
                prior=state.carry_bindings,
            )
            del accounting_delta, capacity_delta, ledger, partition, run, state
        del prepared, products_today

    policy_results = tuple(
        _finalize_policy(
            policy_id,
            states[policy_id],
            daily_runs_retained=retain_daily_runs,
        )
        for policy_id in POLICY_IDS
    )
    return S1PortfolioResult(
        dates=dates,
        policy_results=policy_results,
    )


def _validated_days(
    prepared_days: Sequence[PreparedS1EntryDay],
) -> tuple[PreparedS1EntryDay, ...]:
    if isinstance(prepared_days, (str, bytes)) or not isinstance(
        prepared_days, Sequence
    ):
        raise TypeError("prepared_days must be a sequence")
    days = tuple(prepared_days)
    if not days or any(not isinstance(day, PreparedS1EntryDay) for day in days):
        raise TypeError("prepared_days must contain PreparedS1EntryDay values")
    _validated_dates(tuple(day.date for day in days), source="prepared days")
    for day in days:
        _validate_source_day_shape(day)
    return days


def _validated_dates(
    dates: Sequence[str],
    *,
    source: str = "dates",
) -> tuple[str, ...]:
    if isinstance(dates, (str, bytes)) or not isinstance(dates, Sequence):
        raise TypeError(f"{source} must be a sequence")
    values = tuple(dates)
    if not values or any(
        not isinstance(date, str) or len(date) != 8 or not date.isdigit()
        for date in values
    ):
        raise ValueError(f"{source} must contain YYYYMMDD values")
    try:
        for date in values:
            datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{source} must contain valid YYYYMMDD values") from error
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{source} must have unique, strictly increasing dates")
    return values


def _validated_accounting_catalog(
    accounting_products: Sequence[S1AccountingProduct],
) -> tuple[S1AccountingProduct, ...]:
    if isinstance(accounting_products, (str, bytes)) or not isinstance(
        accounting_products, Sequence
    ):
        raise TypeError("accounting_products must be a sequence")
    products = tuple(accounting_products)
    if not products or any(
        not isinstance(product, S1AccountingProduct) for product in products
    ):
        raise TypeError("accounting_products must contain S1AccountingProduct values")
    if len({product.product_id for product in products}) != len(products):
        raise ValueError("accounting product_id values must be unique")
    if any(product.product_id != product.value_code for product in products):
        raise ValueError(
            "portfolio product_id must equal accounting value_code for "
            "cross-ledger verification"
        )
    return tuple(sorted(products, key=lambda product: product.product_id))


def _validate_source_day_shape(prepared: object) -> PreparedS1EntryDay:
    if not isinstance(prepared, PreparedS1EntryDay):
        raise TypeError("prepare_day must return PreparedS1EntryDay")
    if prepared.policy_ids != POLICY_IDS:
        raise ValueError(
            "each prepared day must contain all seven policies in canonical order"
        )
    return prepared


def _validate_source_day(
    prepared: object,
    requested_date: str,
    catalog: Mapping[str, S1AccountingProduct],
) -> None:
    day = _validate_source_day_shape(prepared)
    if day.date != requested_date:
        raise ValueError("prepare_day returned a different date than requested")
    for product in day.products:
        if product.future_contracts != 1:
            raise ValueError(
                "S1 normal exit requires one futures contract per position"
            )
        try:
            expected = catalog[product.product_id]
        except KeyError as error:
            raise ValueError(
                "daily product is absent from frozen accounting catalog: "
                f"{product.product_id}"
            ) from error
        actual = S1AccountingProduct(
            product_id=product.product_id,
            value_code=product.value_code,
            contract_size_shares=product.contract_size_shares,
        )
        if actual != expected:
            raise ValueError(
                "daily product identity/contract differs from frozen accounting "
                f"catalog: {product.product_id}"
            )


def _accounting_product_union(
    days: Sequence[PreparedS1EntryDay],
) -> tuple[S1AccountingProduct, ...]:
    identities: dict[str, tuple[str, int]] = {}
    for day in days:
        for product in day.products:
            if product.future_contracts != 1:
                raise ValueError(
                    "S1 normal exit requires one futures contract per position"
                )
            identity = (product.value_code, product.contract_size_shares)
            prior = identities.setdefault(product.product_id, identity)
            if prior != identity:
                raise ValueError(
                    "product/value/contract size changed across prepared days: "
                    f"{product.product_id}"
                )
            if product.product_id != product.value_code:
                raise ValueError(
                    "portfolio product_id must equal accounting value_code for "
                    "cross-ledger verification"
                )
    return _validated_accounting_catalog(
        tuple(
            S1AccountingProduct(
                product_id=product_id,
                value_code=value_code,
                contract_size_shares=contract_size,
            )
            for product_id, (value_code, contract_size) in sorted(identities.items())
        )
    )


def _validate_opening_carry(
    *,
    policy_id: str,
    date: str,
    carry: Sequence[S1CarryPosition],
    bindings: Mapping[str, S1CarryContractBinding],
    products_today: Mapping[str, S1Product],
) -> None:
    if set(bindings) != {position.position_id for position in carry}:
        raise RuntimeError("internal carry contract bindings diverged")
    for position in carry:
        try:
            product = products_today[position.product_id]
        except KeyError as error:
            raise ValueError(
                f"{date} {policy_id} carry product is absent from daily mapping: "
                f"{position.product_id}"
            ) from error
        expected = bindings[position.position_id]
        actual = S1CarryContractBinding.from_product(product)
        if actual != expected or position.quote_code != product.quote_code:
            raise ValueError(
                f"{date} {policy_id} carry contract/quote mapping changed: "
                f"{position.product_id}"
            )


def _validate_daily_run(
    *,
    prepared: PreparedS1EntryDay,
    policy_id: str,
    opening_carry: Sequence[S1CarryPosition],
    ledger: CapacityLedger,
    run: S1EntryDayRun,
) -> None:
    if not isinstance(run, S1EntryDayRun):
        raise TypeError("policy_runner must return S1EntryDayRun")
    if run.summary.date != prepared.date or run.summary.policy_id != policy_id:
        raise ValueError("daily run summary identity disagrees with its call")
    if not run.result.normal_exit_enabled:
        raise ValueError("portfolio daily run must enable normal exit")
    if run.result.carry_in != tuple(opening_carry):
        raise ValueError("daily result carry_in differs from portfolio state")
    opening_product_ids = frozenset(position.product_id for position in opening_carry)
    if not opening_product_ids.issubset(run.result.entry_disabled_product_ids):
        raise ValueError("opening carry product was not disabled for the full day")
    if any(order.product_id in opening_product_ids for order in run.result.orders):
        raise ValueError("opening carry product reopened an entry during the day")
    if run.result.capacity_transitions != ledger.transitions:
        raise ValueError("daily result does not expose the ledger's exact daily delta")
    ledger.verify()


def _next_carry_bindings(
    carry: Sequence[S1CarryPosition],
    products_today: Mapping[str, S1Product],
    *,
    prior: Mapping[str, S1CarryContractBinding],
) -> dict[str, S1CarryContractBinding]:
    result: dict[str, S1CarryContractBinding] = {}
    for position in carry:
        try:
            product = products_today[position.product_id]
        except KeyError as error:
            raise ValueError("carry_out references an unknown daily product") from error
        binding = S1CarryContractBinding.from_product(product)
        prior_binding = prior.get(position.position_id)
        if prior_binding is not None and prior_binding != binding:
            raise ValueError("carry_out contract binding changed within its lifecycle")
        if position.quote_code != binding.quote_code:
            raise ValueError("carry_out quote_code differs from the daily product")
        result[position.position_id] = binding
    return result


def _finalize_policy(
    policy_id: str,
    state: _PolicyState,
    *,
    daily_runs_retained: bool,
) -> S1PortfolioPolicyResult:
    if state.capacity_seed is None:
        raise RuntimeError("policy completed without a capacity checkpoint")
    assert state.daily_runs is not None
    assert state.daily_summaries is not None
    assert state.capacity_transitions is not None
    facts = state.accounting.verify().facts
    transitions = tuple(state.capacity_transitions)
    full_replay = replay_capacity_transitions(transitions)
    resumed_replay = CapacityLedger.from_seed(state.capacity_seed).verify()
    _verify_seed_equivalence(full_replay, resumed_replay, state.capacity_seed)
    report = verify_s1_accounting_capacity_links(facts, transitions)
    _verify_final_carry_matches_seed(state.carry, resumed_replay)
    return S1PortfolioPolicyResult(
        policy_id=policy_id,
        daily_runs=tuple(state.daily_runs),
        daily_summaries=tuple(state.daily_summaries),
        daily_runs_retained=daily_runs_retained,
        accounting_facts=facts,
        capacity_transitions=transitions,
        final_carry=state.carry,
        capacity_seed=state.capacity_seed,
        cross_ledger_report=report,
    )


def _verify_seed_equivalence(
    full: CapacityReplayResult,
    resumed: CapacityReplayResult,
    seed: CapacityLedgerSeed,
) -> None:
    fields = (
        "account_balances",
        "account_products",
        "product_balances",
        "global_balances",
        "last_timestamp_ns",
        "last_event_sequence",
        "last_row_index",
        "last_sequence",
        "seen_transition_ids",
        "transition_chain_sha256",
    )
    if any(getattr(full, field) != getattr(resumed, field) for field in fields):
        raise RuntimeError("checkpoint restore differs from uninterrupted replay")
    if full.last_sequence != seed.transition_sequence_offset:
        raise RuntimeError("capacity checkpoint sequence offset is inconsistent")
    if full.transition_chain_sha256 != seed.transition_chain_sha256:
        raise RuntimeError("capacity checkpoint chain hash is inconsistent")


def _verify_final_carry_matches_seed(
    carry: Sequence[S1CarryPosition],
    replay: CapacityReplayResult,
) -> None:
    committed = {
        capacity_id
        for capacity_id, balances in replay.account_balances.items()
        if balances.total_committed_notional_twd
    }
    carried = {position.capacity_id for position in carry}
    if committed != carried:
        raise RuntimeError("final carry does not match committed capacity accounts")


def _taipei_execution_date(cursor: EventCursor) -> str:
    if not isinstance(cursor, EventCursor):
        raise TypeError("cursor must be an EventCursor")
    seconds, _nanoseconds = divmod(cursor.recv_time_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).astimezone(TAIPEI).strftime("%Y%m%d")


__all__ = [
    "PORTFOLIO_RUNNER_VERSION",
    "S1PolicyDayRunner",
    "S1PortfolioDayPartition",
    "S1PortfolioPolicyResult",
    "S1PortfolioResult",
    "S1PortfolioSink",
    "run_s1_portfolio",
    "run_s1_portfolio_from_dates",
]
