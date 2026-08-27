"""Verifier-backed scenario metrics for the S1 accounting fact stream.

The accounting terminal facts are the only realized-PnL source.  Executed-leg
cashflows belonging to an open or unresolved position are deliberately not
marked to market here; only their already-incurred commission and tax are
reported.  Expiry accounting is kept separate from executable completion even
though its realized net is included in the scenario total.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from typing import Literal, Protocol

from .layered import EventCursor
from .s1_accounting import (
    AccountingFact,
    AccountingReplayError,
    ExecutedLeg,
    ExpiryAccountingMark,
    PositionEstablishedFact,
    TerminalRealizedAccounting,
    replay_accounting_facts,
)
from .s1_ranking import ScenarioMetrics
from .transaction_costs import TransactionCostProfile

ExecutionTruth = Literal["exact", "approximate"]
CoverageSource = Literal[
    "daily_all_risk_pricing",
    "entry_hedge_success_proxy",
]

_ROUTE_ROLES = frozenset(
    {
        "entry_maker",
        "entry_hedge",
        "entry_rollback",
        "exit_maker",
        "exit_hedge",
        "exit_rollback",
    }
)
_ENTRY_ROLLBACK_OUTCOMES = frozenset(
    {
        "entry_emergency_rollback_flat",
        "entry_partial_rollback_flat",
    }
)
_RISK_EVENT_TYPES = frozenset({"created", "evaluated", "actual_send", "timeout"})
_RISK_TERMINAL_EVENT_TYPES = frozenset({"actual_send", "timeout"})
_RISK_TERMINAL_STATUSES = {
    ("hedge", "actual_send"): "hedge_sent",
    ("hedge", "timeout"): "hedge_retry_timeout",
    ("rollback", "actual_send"): "rollback_sent",
    ("rollback", "timeout"): "rollback_retry_timeout",
}


class S1PerformanceError(ValueError):
    """Accounting facts cannot produce one causally valid scenario report."""


class _RiskEventLike(Protocol):
    risk_id: str
    request_id: str
    position_id: str
    product_id: str
    risk_kind: str
    event_type: str
    cursor: EventCursor
    status: str
    stage: str


@dataclass(frozen=True, slots=True)
class S1DailyReplaySummary:
    """Pre-aggregated all-risk hedge-pricing coverage for one session.

    The denominator contains complete entry/exit hedge and rollback lifecycles;
    the numerator contains their actual sends.  This optional input is
    intentionally independent of entry hedge success, so it must not be
    substituted for the entry-maker denominator in the descriptive metrics.
    """

    date: str
    scenario_id: str
    all_risk_hedge_priced_numerator: int
    all_risk_hedge_priced_denominator: int

    def __post_init__(self) -> None:
        _yyyymmdd(self.date, "date")
        _identifier(self.scenario_id, "scenario_id")
        _rate_counts(
            self.all_risk_hedge_priced_numerator,
            self.all_risk_hedge_priced_denominator,
            "all-risk hedge pricing",
        )


def build_s1_daily_replay_summary_from_risk_events(
    risk_events: Sequence[_RiskEventLike],
    *,
    date: str,
    scenario_id: str,
) -> S1DailyReplaySummary:
    """Verify one complete daily risk stream and summarize actual-send coverage.

    Each unique risk lifecycle must be ``created``, may be ``evaluated`` zero or
    more times, and must end in exactly one ``actual_send`` or ``timeout``.  An
    ``actual_send`` is the event-loop contract that a legal executable price was
    obtained and the venue request was sent.  Trigger-arrival reference coverage
    is deliberately outside this metric.
    """

    selected_date = _yyyymmdd(date, "date")
    selected_scenario = _identifier(scenario_id, "scenario_id")
    events = _risk_event_sequence(risk_events)
    identities: dict[str, tuple[str, str, str, str, str]] = {}
    terminal_event_types: dict[str, str] = {}
    last_cursor: EventCursor | None = None

    for event in events:
        if last_cursor is not None and event.cursor < last_cursor:
            raise S1PerformanceError("risk event stream cursor moved backwards")
        last_cursor = event.cursor
        identity = (
            event.request_id,
            event.position_id,
            event.product_id,
            event.risk_kind,
            event.stage,
        )
        if event.event_type == "created":
            if event.risk_id in identities:
                raise S1PerformanceError("risk lifecycle has multiple created events")
            identities[event.risk_id] = identity
            continue
        if event.risk_id not in identities:
            raise S1PerformanceError("risk lifecycle event is orphaned from created")
        expected_identity = identities[event.risk_id]
        for name, expected, actual in zip(
            ("request_id", "position_id", "product_id", "risk_kind", "stage"),
            expected_identity,
            identity,
            strict=True,
        ):
            if actual != expected:
                raise S1PerformanceError(f"risk lifecycle {name} drifted")
        if event.risk_id in terminal_event_types:
            if event.event_type in _RISK_TERMINAL_EVENT_TYPES:
                raise S1PerformanceError("risk lifecycle has multiple terminal events")
            raise S1PerformanceError("risk lifecycle has an event after its terminal")
        if event.event_type in _RISK_TERMINAL_EVENT_TYPES:
            expected_status = _RISK_TERMINAL_STATUSES[
                (event.risk_kind, event.event_type)
            ]
            if event.status != expected_status:
                raise S1PerformanceError(
                    "risk terminal event disagrees with its lifecycle status"
                )
            terminal_event_types[event.risk_id] = event.event_type

    missing_terminal = sorted(set(identities).difference(terminal_event_types))
    if missing_terminal:
        raise S1PerformanceError("risk lifecycle is missing a terminal event")
    return S1DailyReplaySummary(
        date=selected_date,
        scenario_id=selected_scenario,
        all_risk_hedge_priced_numerator=sum(
            event_type == "actual_send" for event_type in terminal_event_types.values()
        ),
        all_risk_hedge_priced_denominator=len(identities),
    )


@dataclass(frozen=True, slots=True)
class S1CalendarNet:
    calendar_key: str
    executable_terminal_count: int
    expiry_mark_count: int
    executable_terminal_net_twd: Decimal
    expiry_mark_net_twd: Decimal
    total_net_twd: Decimal


@dataclass(frozen=True, slots=True)
class S1EntryFillTruthMetrics:
    """Lifecycle outcomes grouped by the initiating maker fill truth."""

    entry_fill_truth: ExecutionTruth
    entry_positions: int
    same_day_exit_maker_flat: int
    cross_day_exit_maker_flat: int
    expiry_marks: int
    entry_rollbacks: int
    other_executable_terminals: int
    open_or_unresolved: int
    terminal_realized_net_twd: Decimal
    open_execution_actual_cost_twd: Decimal

    @property
    def terminal_count(self) -> int:
        return (
            self.same_day_exit_maker_flat
            + self.cross_day_exit_maker_flat
            + self.expiry_marks
            + self.entry_rollbacks
            + self.other_executable_terminals
        )


@dataclass(frozen=True, slots=True)
class S1ScenarioPerformance:
    scenario_id: str
    reporting_dates: tuple[str, ...]
    entry_positions: int
    entry_fill_exact: int
    entry_fill_approximate: int
    entry_hedge_success_numerator: int
    entry_hedge_success_denominator: int
    same_day_exit_maker_flat: int
    cross_day_exit_maker_flat: int
    expiry_marks: int
    entry_rollbacks: int
    other_executable_terminals: int
    open_or_unresolved: int
    terminal_coverage_numerator: int
    terminal_coverage_denominator: int
    executable_terminal_net_twd: Decimal
    expiry_mark_net_twd: Decimal
    total_net_twd: Decimal
    open_execution_actual_commission_twd: Decimal
    open_execution_actual_tax_twd: Decimal
    open_execution_actual_cost_twd: Decimal
    hedge_priced_numerator: int
    hedge_priced_denominator: int
    hedge_pricing_coverage_source: CoverageSource
    by_entry_fill_truth: tuple[S1EntryFillTruthMetrics, ...]
    daily_net: tuple[S1CalendarNet, ...]
    monthly_net: tuple[S1CalendarNet, ...]

    @property
    def reporting_sessions(self) -> int:
        return len(self.reporting_dates)

    @property
    def completion_rate(self) -> Decimal | None:
        return _optional_ratio(self.same_day_exit_maker_flat, self.entry_positions)

    @property
    def approx_screen_completion_numerator(self) -> int:
        """Same-day normal exits initiated by approximate maker fills."""

        return self._entry_fill_truth_metrics("approximate").same_day_exit_maker_flat

    @property
    def approx_screen_completion_denominator(self) -> int:
        """Entry positions initiated by approximate maker fills."""

        return self._entry_fill_truth_metrics("approximate").entry_positions

    @property
    def approx_screen_completion_rate(self) -> Decimal | None:
        """Approximate-fill screen rate, without any capacity-cap attestation."""

        return _optional_ratio(
            self.approx_screen_completion_numerator,
            self.approx_screen_completion_denominator,
        )

    @property
    def terminal_coverage(self) -> Decimal | None:
        return _optional_ratio(
            self.terminal_coverage_numerator,
            self.terminal_coverage_denominator,
        )

    @property
    def entry_hedge_success_rate(self) -> Decimal | None:
        return _optional_ratio(
            self.entry_hedge_success_numerator,
            self.entry_hedge_success_denominator,
        )

    @property
    def mean_daily_net_twd(self) -> Decimal:
        return _ratio(self.total_net_twd, self.reporting_sessions)

    def to_scenario_metrics(self) -> ScenarioMetrics:
        """Return the sufficient statistics consumed by ``s1_ranking``."""

        return ScenarioMetrics(
            scenario_id=self.scenario_id,
            completion_numerator=self.same_day_exit_maker_flat,
            completion_denominator=self.entry_positions,
            total_net_twd=self.total_net_twd,
            reporting_sessions=self.reporting_sessions,
            hedge_priced_numerator=self.hedge_priced_numerator,
            hedge_priced_denominator=self.hedge_priced_denominator,
        )

    def to_approx_screen_scenario_metrics(self) -> ScenarioMetrics:
        """Return Spot-Bid ranking metrics using only approximate entry truth.

        This conversion selects the approximate-fill completion slice.  It does
        not attest that any particular capacity cap was configured or enforced.
        """

        return ScenarioMetrics(
            scenario_id=self.scenario_id,
            completion_numerator=self.approx_screen_completion_numerator,
            completion_denominator=self.approx_screen_completion_denominator,
            total_net_twd=self.total_net_twd,
            reporting_sessions=self.reporting_sessions,
            hedge_priced_numerator=self.hedge_priced_numerator,
            hedge_priced_denominator=self.hedge_priced_denominator,
        )

    def _entry_fill_truth_metrics(
        self,
        truth: ExecutionTruth,
    ) -> S1EntryFillTruthMetrics:
        matches = tuple(
            row for row in self.by_entry_fill_truth if row.entry_fill_truth == truth
        )
        if len(matches) != 1:
            raise S1PerformanceError(
                f"expected exactly one {truth} entry-fill truth metrics row"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class _PositionOutcome:
    entry: ExecutedLeg
    terminal: TerminalRealizedAccounting | ExpiryAccountingMark | None
    category: Literal[
        "same_day",
        "cross_day",
        "expiry",
        "entry_rollback",
        "other_executable",
        "open",
    ]


def aggregate_s1_scenario_metrics(
    accounting_facts: Sequence[AccountingFact],
    reporting_dates: Sequence[str],
    *,
    scenario_id: str | None = None,
    daily_replay_summaries: Sequence[S1DailyReplaySummary] | None = None,
    profile: TransactionCostProfile | None = None,
    cursor_date_resolver: Callable[[EventCursor], str] | None = None,
) -> S1ScenarioPerformance:
    """Verify a complete accounting stream and aggregate one S1 scenario.

    ``same_day_exit_maker_flat / entry_maker`` is the ranking completion rate.
    Cross-day normal exits, entry rollbacks, other executable terminals, and
    non-executable expiry marks remain visible but cannot enter that numerator.
    """

    facts = _accounting_fact_sequence(accounting_facts)
    dates = _reporting_dates(reporting_dates)
    try:
        replay_accounting_facts(
            facts,
            profile=profile,
            cursor_date_resolver=cursor_date_resolver,
            require_route_roles=True,
        ).verify()
    except (AccountingReplayError, TypeError, ValueError) as error:
        raise S1PerformanceError(
            f"accounting replay verification failed: {error}"
        ) from error

    summaries = _daily_summaries(daily_replay_summaries)
    selected_scenario = _scenario(facts, summaries, scenario_id)
    date_set = frozenset(dates)
    _validate_fact_dates(facts, date_set)

    executions = tuple(fact for fact in facts if isinstance(fact, ExecutedLeg))
    if any(row.route_role not in _ROUTE_ROLES for row in executions):
        raise S1PerformanceError("execution contains an unsupported S1 route_role")
    by_position: dict[str, list[ExecutedLeg]] = defaultdict(list)
    for row in executions:
        by_position[row.position_id].append(row)

    entries: dict[str, ExecutedLeg] = {}
    for position_id, rows in by_position.items():
        position_entries = [row for row in rows if row.route_role == "entry_maker"]
        if len(position_entries) != 1:
            raise S1PerformanceError(
                "every accounting position must have exactly one entry_maker execution"
            )
        entries[position_id] = position_entries[0]

    establishments = _unique_position_facts(
        facts,
        PositionEstablishedFact,
        "position establishment",
    )
    terminals = _terminal_facts(facts)
    known_positions = set(entries)
    if set(establishments) - known_positions or set(terminals) - known_positions:
        raise S1PerformanceError(
            "establishment or terminal fact has no unique entry_maker position"
        )

    outcomes: list[_PositionOutcome] = []
    for position_id, entry in entries.items():
        rows = by_position[position_id]
        if any(row.cursor < entry.cursor for row in rows):
            raise S1PerformanceError(
                "position execution causally precedes its entry_maker"
            )
        establishment = establishments.get(position_id)
        if establishment is not None:
            if establishment.cursor <= entry.cursor:
                raise S1PerformanceError(
                    "position establishment must follow its entry_maker"
                )
            if establishment.establishment_date != entry.execution_date:
                raise S1PerformanceError(
                    "position establishment date must equal entry execution date"
                )
            if not any(row.route_role == "entry_hedge" for row in rows):
                raise S1PerformanceError(
                    "position establishment has no entry_hedge execution"
                )

        terminal = terminals.get(position_id)
        category = _position_category(entry, establishment, terminal)
        outcomes.append(_PositionOutcome(entry, terminal, category))

    entry_count = len(entries)
    entry_exact = sum(outcome.entry.execution_truth == "exact" for outcome in outcomes)
    entry_approximate = sum(
        outcome.entry.execution_truth == "approximate" for outcome in outcomes
    )
    if entry_exact + entry_approximate != entry_count:
        raise S1PerformanceError("entry fill truth is not exact or approximate")

    category_counts = {
        category: sum(outcome.category == category for outcome in outcomes)
        for category in (
            "same_day",
            "cross_day",
            "expiry",
            "entry_rollback",
            "other_executable",
            "open",
        )
    }
    terminal_count = entry_count - category_counts["open"]
    executable_terminal_net = _decimal_sum(
        outcome.terminal.realized_net_twd
        for outcome in outcomes
        if isinstance(outcome.terminal, TerminalRealizedAccounting)
    )
    expiry_net = _decimal_sum(
        outcome.terminal.realized_net_twd
        for outcome in outcomes
        if isinstance(outcome.terminal, ExpiryAccountingMark)
    )

    open_position_ids = {
        position_id for position_id in entries if position_id not in terminals
    }
    open_rows = tuple(row for row in executions if row.position_id in open_position_ids)
    open_commission = _decimal_sum(row.commission_twd for row in open_rows)
    open_tax = _decimal_sum(row.tax_twd for row in open_rows)
    open_cost = _decimal_sum(row.total_cost_twd for row in open_rows)

    hedge_success = len(establishments)
    if summaries is None:
        hedge_priced_numerator = hedge_success
        hedge_priced_denominator = entry_count
        coverage_source: CoverageSource = "entry_hedge_success_proxy"
    else:
        _validate_daily_summaries(
            summaries,
            dates=dates,
            scenario_id=selected_scenario,
            entries=tuple(entries.values()),
            establishments=tuple(establishments.values()),
        )
        hedge_priced_numerator = sum(
            row.all_risk_hedge_priced_numerator for row in summaries
        )
        hedge_priced_denominator = sum(
            row.all_risk_hedge_priced_denominator for row in summaries
        )
        coverage_source = "daily_all_risk_pricing"

    daily = _calendar_net(outcomes, dates, monthly=False)
    monthly_keys = tuple(sorted({date[:6] for date in dates}))
    monthly = _calendar_net(outcomes, monthly_keys, monthly=True)
    by_truth = tuple(
        _truth_metrics(truth, outcomes, by_position)
        for truth in ("exact", "approximate")
    )
    return S1ScenarioPerformance(
        scenario_id=selected_scenario,
        reporting_dates=dates,
        entry_positions=entry_count,
        entry_fill_exact=entry_exact,
        entry_fill_approximate=entry_approximate,
        entry_hedge_success_numerator=hedge_success,
        entry_hedge_success_denominator=entry_count,
        same_day_exit_maker_flat=category_counts["same_day"],
        cross_day_exit_maker_flat=category_counts["cross_day"],
        expiry_marks=category_counts["expiry"],
        entry_rollbacks=category_counts["entry_rollback"],
        other_executable_terminals=category_counts["other_executable"],
        open_or_unresolved=category_counts["open"],
        terminal_coverage_numerator=terminal_count,
        terminal_coverage_denominator=entry_count,
        executable_terminal_net_twd=executable_terminal_net,
        expiry_mark_net_twd=expiry_net,
        total_net_twd=executable_terminal_net + expiry_net,
        open_execution_actual_commission_twd=open_commission,
        open_execution_actual_tax_twd=open_tax,
        open_execution_actual_cost_twd=open_cost,
        hedge_priced_numerator=hedge_priced_numerator,
        hedge_priced_denominator=hedge_priced_denominator,
        hedge_pricing_coverage_source=coverage_source,
        by_entry_fill_truth=by_truth,
        daily_net=daily,
        monthly_net=monthly,
    )


def _position_category(
    entry: ExecutedLeg,
    establishment: PositionEstablishedFact | None,
    terminal: TerminalRealizedAccounting | ExpiryAccountingMark | None,
) -> Literal[
    "same_day",
    "cross_day",
    "expiry",
    "entry_rollback",
    "other_executable",
    "open",
]:
    if terminal is None:
        return "open"
    terminal_date = (
        terminal.expiry_date
        if isinstance(terminal, ExpiryAccountingMark)
        else terminal.terminal_date
    )
    if terminal.cursor <= entry.cursor:
        raise S1PerformanceError("terminal fact must causally follow entry_maker")
    if terminal_date < entry.execution_date:
        raise S1PerformanceError("terminal date cannot precede entry execution date")
    if isinstance(terminal, ExpiryAccountingMark):
        if establishment is None:
            raise S1PerformanceError("expiry mark requires an established position")
        return "expiry"
    if terminal.terminal_outcome == "exit_maker_flat":
        if establishment is None:
            raise S1PerformanceError(
                "normal exit terminal requires an established position"
            )
        return "same_day" if terminal_date == entry.execution_date else "cross_day"
    if terminal.terminal_outcome in _ENTRY_ROLLBACK_OUTCOMES:
        if establishment is not None:
            raise S1PerformanceError(
                "entry rollback cannot follow position establishment"
            )
        return "entry_rollback"
    if establishment is None:
        raise S1PerformanceError(
            "non-rollback executable terminal requires an established position"
        )
    return "other_executable"


def _calendar_net(
    outcomes: Sequence[_PositionOutcome],
    keys: Sequence[str],
    *,
    monthly: bool,
) -> tuple[S1CalendarNet, ...]:
    executable_count: dict[str, int] = defaultdict(int)
    expiry_count: dict[str, int] = defaultdict(int)
    executable_net: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    expiry_net: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for outcome in outcomes:
        terminal = outcome.terminal
        if terminal is None:
            continue
        terminal_date = (
            terminal.expiry_date
            if isinstance(terminal, ExpiryAccountingMark)
            else terminal.terminal_date
        )
        key = terminal_date[:6] if monthly else terminal_date
        value = _money(terminal.realized_net_twd)
        if isinstance(terminal, ExpiryAccountingMark):
            expiry_count[key] += 1
            expiry_net[key] += value
        else:
            executable_count[key] += 1
            executable_net[key] += value
    return tuple(
        S1CalendarNet(
            calendar_key=key,
            executable_terminal_count=executable_count[key],
            expiry_mark_count=expiry_count[key],
            executable_terminal_net_twd=executable_net[key],
            expiry_mark_net_twd=expiry_net[key],
            total_net_twd=executable_net[key] + expiry_net[key],
        )
        for key in keys
    )


def _truth_metrics(
    truth: ExecutionTruth,
    outcomes: Sequence[_PositionOutcome],
    executions: dict[str, list[ExecutedLeg]],
) -> S1EntryFillTruthMetrics:
    selected = tuple(
        outcome for outcome in outcomes if outcome.entry.execution_truth == truth
    )
    open_ids = {
        outcome.entry.position_id for outcome in selected if outcome.terminal is None
    }
    return S1EntryFillTruthMetrics(
        entry_fill_truth=truth,
        entry_positions=len(selected),
        same_day_exit_maker_flat=sum(
            outcome.category == "same_day" for outcome in selected
        ),
        cross_day_exit_maker_flat=sum(
            outcome.category == "cross_day" for outcome in selected
        ),
        expiry_marks=sum(outcome.category == "expiry" for outcome in selected),
        entry_rollbacks=sum(
            outcome.category == "entry_rollback" for outcome in selected
        ),
        other_executable_terminals=sum(
            outcome.category == "other_executable" for outcome in selected
        ),
        open_or_unresolved=sum(outcome.category == "open" for outcome in selected),
        terminal_realized_net_twd=_decimal_sum(
            outcome.terminal.realized_net_twd
            for outcome in selected
            if outcome.terminal is not None
        ),
        open_execution_actual_cost_twd=_decimal_sum(
            row.total_cost_twd
            for position_id in open_ids
            for row in executions[position_id]
        ),
    )


def _terminal_facts(
    facts: Sequence[AccountingFact],
) -> dict[str, TerminalRealizedAccounting | ExpiryAccountingMark]:
    result: dict[str, TerminalRealizedAccounting | ExpiryAccountingMark] = {}
    for fact in facts:
        if not isinstance(fact, (TerminalRealizedAccounting, ExpiryAccountingMark)):
            continue
        if fact.position_id in result:
            raise S1PerformanceError("position has multiple terminal facts")
        result[fact.position_id] = fact
    return result


def _unique_position_facts[FactT](
    facts: Sequence[AccountingFact],
    fact_type: type[FactT],
    name: str,
) -> dict[str, FactT]:
    result: dict[str, FactT] = {}
    for fact in facts:
        if not isinstance(fact, fact_type):
            continue
        position_id = fact.position_id
        if position_id in result:
            raise S1PerformanceError(f"position has multiple {name} facts")
        result[position_id] = fact
    return result


def _scenario(
    facts: Sequence[AccountingFact],
    summaries: tuple[S1DailyReplaySummary, ...] | None,
    requested: str | None,
) -> str:
    values = {fact.scenario_id for fact in facts}
    if summaries is not None:
        values.update(row.scenario_id for row in summaries)
    if requested is not None:
        values.add(_identifier(requested, "scenario_id"))
    if not values:
        raise S1PerformanceError(
            "scenario_id is required when accounting facts and summaries are empty"
        )
    if len(values) != 1:
        raise S1PerformanceError("accounting inputs do not describe one scenario")
    return next(iter(values))


def _validate_fact_dates(
    facts: Sequence[AccountingFact], reporting_dates: frozenset[str]
) -> None:
    for fact in facts:
        if isinstance(fact, ExecutedLeg):
            value = fact.execution_date
        elif isinstance(fact, PositionEstablishedFact):
            value = fact.establishment_date
        elif isinstance(fact, TerminalRealizedAccounting):
            value = fact.terminal_date
        else:
            value = fact.expiry_date
        if value not in reporting_dates:
            raise S1PerformanceError("accounting fact date is outside reporting_dates")


def _validate_daily_summaries(
    summaries: tuple[S1DailyReplaySummary, ...],
    *,
    dates: tuple[str, ...],
    scenario_id: str,
    entries: tuple[ExecutedLeg, ...],
    establishments: tuple[PositionEstablishedFact, ...],
) -> None:
    summary_dates = [row.date for row in summaries]
    if len(set(summary_dates)) != len(summary_dates):
        raise S1PerformanceError("daily replay summary date is duplicated")
    if set(summary_dates) != set(dates):
        raise S1PerformanceError(
            "daily replay summaries must cover every reporting date exactly once"
        )
    if any(row.scenario_id != scenario_id for row in summaries):
        raise S1PerformanceError("daily replay summary scenario_id disagrees")
    entry_count_by_date: dict[str, int] = defaultdict(int)
    success_count_by_date: dict[str, int] = defaultdict(int)
    for entry in entries:
        entry_count_by_date[entry.execution_date] += 1
    for establishment in establishments:
        success_count_by_date[establishment.establishment_date] += 1
    for summary in summaries:
        if (
            summary.all_risk_hedge_priced_denominator
            < entry_count_by_date[summary.date]
        ):
            raise S1PerformanceError(
                "all-risk hedge denominator omits entry hedge risks"
            )
        if (
            summary.all_risk_hedge_priced_numerator
            < success_count_by_date[summary.date]
        ):
            raise S1PerformanceError(
                "all-risk hedge numerator omits successful entry hedges"
            )


def _accounting_fact_sequence(values: object) -> tuple[AccountingFact, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("accounting_facts must be a sequence")
    facts = tuple(values)
    expected = (
        ExecutedLeg,
        PositionEstablishedFact,
        TerminalRealizedAccounting,
        ExpiryAccountingMark,
    )
    if any(not isinstance(fact, expected) for fact in facts):
        raise TypeError("accounting_facts contains an invalid value")
    return facts


def _risk_event_sequence(values: object) -> tuple[_RiskEventLike, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("risk_events must be a sequence")
    events = tuple(values)
    for event in events:
        try:
            risk_id = event.risk_id
            request_id = event.request_id
            position_id = event.position_id
            product_id = event.product_id
            risk_kind = event.risk_kind
            event_type = event.event_type
            cursor = event.cursor
            status = event.status
            stage = event.stage
        except AttributeError as error:
            raise TypeError("risk_events contains an invalid value") from error
        _identifier(risk_id, "risk_id")
        _identifier(request_id, "request_id")
        _identifier(position_id, "position_id")
        _identifier(product_id, "product_id")
        if risk_kind not in ("hedge", "rollback"):
            raise S1PerformanceError("risk_kind must be hedge or rollback")
        if event_type not in _RISK_EVENT_TYPES:
            raise S1PerformanceError("risk event_type is unsupported")
        if not isinstance(cursor, EventCursor):
            raise TypeError("risk event cursor must be an EventCursor")
        _identifier(status, "risk event status")
        if stage not in ("entry", "exit"):
            raise S1PerformanceError("risk event stage must be entry or exit")
    return events


def _daily_summaries(
    values: object,
) -> tuple[S1DailyReplaySummary, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("daily_replay_summaries must be a sequence or None")
    summaries = tuple(values)
    if any(not isinstance(row, S1DailyReplaySummary) for row in summaries):
        raise TypeError("daily_replay_summaries contains an invalid value")
    return summaries


def _reporting_dates(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("reporting_dates must be a sequence")
    dates = tuple(_yyyymmdd(value, "reporting date") for value in values)
    if not dates:
        raise S1PerformanceError("reporting_dates cannot be empty")
    if len(set(dates)) != len(dates):
        raise S1PerformanceError("reporting_dates cannot contain duplicates")
    return tuple(sorted(dates))


def _decimal_sum(values: Iterable[object]) -> Decimal:
    return sum((_money(value) for value in values), Decimal(0))


def _money(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise S1PerformanceError("money value must be numeric")
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise S1PerformanceError("money value must be finite")
    return result


def _optional_ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return _ratio(Decimal(numerator), denominator)


def _ratio(numerator: Decimal, denominator: int) -> Decimal:
    if denominator <= 0:
        raise S1PerformanceError("ratio denominator must be positive")
    with localcontext() as context:
        context.prec = 50
        return numerator / Decimal(denominator)


def _rate_counts(numerator: object, denominator: object, name: str) -> None:
    numerator_value = _nonnegative_integer(numerator, f"{name} numerator")
    denominator_value = _nonnegative_integer(denominator, f"{name} denominator")
    if numerator_value > denominator_value:
        raise S1PerformanceError(f"{name} numerator cannot exceed denominator")


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S1PerformanceError(f"{name} must be a non-negative integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise S1PerformanceError(f"{name} must be a nonempty canonical string")
    return value


def _yyyymmdd(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise S1PerformanceError(f"{name} must be YYYYMMDD")
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as error:
        raise S1PerformanceError(f"{name} must be a valid YYYYMMDD") from error
    return value


__all__ = [
    "CoverageSource",
    "S1CalendarNet",
    "S1DailyReplaySummary",
    "S1EntryFillTruthMetrics",
    "S1PerformanceError",
    "S1ScenarioPerformance",
    "aggregate_s1_scenario_metrics",
    "build_s1_daily_replay_summary_from_risk_events",
]
