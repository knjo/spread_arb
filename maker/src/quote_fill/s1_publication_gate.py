"""Pure fail-closed publication checks for the rebuilt S1 screen.

The replay and accounting modules remain the sources of facts.  This module
does no I/O and does not rank scenarios.  It decides whether a caller may
publish the supplied descriptive and ranking claims after checking the common
lookup denominator, entry funnel, position accounting, modeled direct costs,
and open-position valuation contract.

Unsupported lookup cells are valid only when they remain explicit no-trade
rows in the common denominator.  Unmodeled costs are disclosures with no
numeric value; using zero would falsely claim that the cost was measured.
Terminal cashflow rankings are not comparable while any open position lacks a
common-horizon valuation including both incurred and remaining exit costs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Final

from .s1_ranking import ScenarioMetrics, rank_s1_shortlist

MONEY_TOLERANCE_TWD: Final = Decimal("0.000001")
REQUIRED_UNMODELED_COST_IDS: Final = (
    "futures_margin_opportunity_cost",
    "overnight_financing",
    "spot_borrow_cost",
)
_COST_HORIZONS: Final = frozenset({"ungated", "same_day", "overnight"})


class S1PublicationGateError(ValueError):
    """The gate input is malformed or a blocked publication was requested."""


@dataclass(frozen=True, slots=True)
class S1LookupDenominatorFacts:
    """One scenario's coverage of the common mother-by-TOD lookup grid."""

    common_mother_product_days: int
    tod_bucket_count: int
    expected_policy_cells: int
    policy_spec_rows: int
    supported_policy_cells: int
    unsupported_policy_cells: int
    reporting_denominator_cells: int
    unsupported_no_trade_cells: int
    common_population_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "common_population_sha256":
                continue
            _nonnegative_integer(getattr(self, name), name)
        if self.common_mother_product_days == 0:
            raise S1PublicationGateError(
                "common_mother_product_days must be positive"
            )
        if self.tod_bucket_count == 0:
            raise S1PublicationGateError("tod_bucket_count must be positive")
        _sha256(self.common_population_sha256, "common_population_sha256")


@dataclass(frozen=True, slots=True)
class S1EntryFunnelFacts:
    """Exact count funnel from candidate intent through order terminal state."""

    candidate_intents: int
    reservation_attempts: int
    admitted_orders: int
    cap_blocked_attempts: int
    blocked_candidate_intents: int
    sent_orders: int
    makerfill_supported_orders: int
    makerfill_unsupported_orders: int
    filled_orders: int
    actual_cancelled_orders: int
    session_expired_orders: int
    unknown_terminal_orders: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class S1PositionAccountingFacts:
    """Position outcomes and the terminal-coverage counts already reported."""

    entry_positions: int
    same_day_exit_maker_flat: int
    cross_day_exit_maker_flat: int
    expiry_marks: int
    entry_rollbacks: int
    other_executable_terminals: int
    paired_open_positions: int
    naked_unresolved_positions: int
    open_or_unresolved: int
    terminal_coverage_numerator: int
    terminal_coverage_denominator: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class S1ModeledCostAccountingFacts:
    """Aggregated direct-cost components reconstructed from executed legs.

    All fields cover terminal positions except the explicitly named open fields.
    ``terminal_gross_pnl_twd`` must be independently reconstructed from spot
    cashflow plus futures realized PnL, rather than inferred from net.
    """

    executable_terminal_gross_pnl_twd: Decimal
    expiry_mark_gross_pnl_twd: Decimal
    terminal_gross_pnl_twd: Decimal
    executable_terminal_commission_twd: Decimal
    executable_terminal_tax_twd: Decimal
    expiry_mark_commission_twd: Decimal
    expiry_mark_tax_twd: Decimal
    terminal_modeled_direct_cost_twd: Decimal
    executable_terminal_net_twd: Decimal
    expiry_mark_net_twd: Decimal
    total_net_twd: Decimal
    terminal_executed_turnover_twd: Decimal
    open_executed_turnover_twd: Decimal
    total_executed_turnover_twd: Decimal
    open_execution_actual_commission_twd: Decimal
    open_execution_actual_tax_twd: Decimal
    open_execution_actual_cost_twd: Decimal

    def __post_init__(self) -> None:
        for name in (
            "executable_terminal_gross_pnl_twd",
            "expiry_mark_gross_pnl_twd",
            "terminal_gross_pnl_twd",
            "executable_terminal_net_twd",
            "expiry_mark_net_twd",
            "total_net_twd",
        ):
            _money(getattr(self, name), name)
        for name in (
            "executable_terminal_commission_twd",
            "executable_terminal_tax_twd",
            "expiry_mark_commission_twd",
            "expiry_mark_tax_twd",
            "terminal_modeled_direct_cost_twd",
            "terminal_executed_turnover_twd",
            "open_executed_turnover_twd",
            "total_executed_turnover_twd",
            "open_execution_actual_commission_twd",
            "open_execution_actual_tax_twd",
            "open_execution_actual_cost_twd",
        ):
            _nonnegative_money(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class S1UnavailableCost:
    """One cost that is explicitly unavailable, never silently numeric zero."""

    cost_id: str
    amount_twd: Decimal | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _canonical_text(self.cost_id, "cost_id")
        if self.amount_twd is not None:
            _money(self.amount_twd, "amount_twd")
        if self.unavailable_reason is not None:
            _canonical_text(self.unavailable_reason, "unavailable_reason")


@dataclass(frozen=True, slots=True)
class S1OpenPositionValuation:
    """Comparable common-horizon mark for all open positions in one scenario.

    ``net_mark_pnl_twd`` is after both the already-incurred open execution cost
    and the remaining executable exit cost.  The publication gate verifies this
    conservation equation against ``S1ModeledCostAccountingFacts``.
    """

    position_count: int
    valuation_method_id: str
    valuation_asof_id: str
    comparable_across_scenarios: bool
    gross_mark_pnl_twd: Decimal
    remaining_exit_cost_twd: Decimal
    net_mark_pnl_twd: Decimal

    def __post_init__(self) -> None:
        _nonnegative_integer(self.position_count, "position_count")
        _canonical_text(self.valuation_method_id, "valuation_method_id")
        _canonical_text(self.valuation_asof_id, "valuation_asof_id")
        _boolean(
            self.comparable_across_scenarios,
            "comparable_across_scenarios",
        )
        _money(self.gross_mark_pnl_twd, "gross_mark_pnl_twd")
        _nonnegative_money(
            self.remaining_exit_cost_twd,
            "remaining_exit_cost_twd",
        )
        _money(self.net_mark_pnl_twd, "net_mark_pnl_twd")


@dataclass(frozen=True, slots=True)
class S1PublicationScenarioFacts:
    """All verifier-backed facts needed for one scenario publication row."""

    scenario_id: str
    reporting_sessions: int
    cost_horizon: str
    economic_gate_enabled: bool
    deployment_shortlist_eligible: bool
    lookup: S1LookupDenominatorFacts
    funnel: S1EntryFunnelFacts
    accounting: S1PositionAccountingFacts
    costs: S1ModeledCostAccountingFacts
    unmodeled_costs: tuple[S1UnavailableCost, ...]
    open_valuation: S1OpenPositionValuation | None
    economic_ranking_net_twd: Decimal | None
    hedge_priced_numerator: int
    hedge_priced_denominator: int

    def __post_init__(self) -> None:
        _canonical_text(self.scenario_id, "scenario_id")
        _positive_integer(self.reporting_sessions, "reporting_sessions")
        if self.cost_horizon not in _COST_HORIZONS:
            raise S1PublicationGateError("cost_horizon is unsupported")
        _boolean(self.economic_gate_enabled, "economic_gate_enabled")
        _boolean(
            self.deployment_shortlist_eligible,
            "deployment_shortlist_eligible",
        )
        for value, expected, name in (
            (self.lookup, S1LookupDenominatorFacts, "lookup"),
            (self.funnel, S1EntryFunnelFacts, "funnel"),
            (self.accounting, S1PositionAccountingFacts, "accounting"),
            (self.costs, S1ModeledCostAccountingFacts, "costs"),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} has an invalid type")
        values = _typed_tuple(
            self.unmodeled_costs,
            S1UnavailableCost,
            "unmodeled_costs",
        )
        object.__setattr__(self, "unmodeled_costs", values)
        if self.open_valuation is not None and not isinstance(
            self.open_valuation,
            S1OpenPositionValuation,
        ):
            raise TypeError(
                "open_valuation must be S1OpenPositionValuation or None"
            )
        if self.economic_ranking_net_twd is not None:
            _money(
                self.economic_ranking_net_twd,
                "economic_ranking_net_twd",
            )
        _rate_counts(
            self.hedge_priced_numerator,
            self.hedge_priced_denominator,
            "hedge_priced",
        )


@dataclass(frozen=True, slots=True)
class S1PublicationClaims:
    """Ranking identities a report proposes to publish."""

    completion_champion_id: str | None = None
    net_champion_id: str | None = None
    pareto_frontier_ids: tuple[str, ...] = ()
    s2_shortlist_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("completion_champion_id", "net_champion_id"):
            value = getattr(self, name)
            if value is not None:
                _canonical_text(value, name)
        for name in ("pareto_frontier_ids", "s2_shortlist_ids"):
            values = _string_tuple(getattr(self, name), name)
            object.__setattr__(self, name, values)

    @property
    def requests_completion_ranking(self) -> bool:
        return self.completion_champion_id is not None

    @property
    def requests_economic_ranking(self) -> bool:
        return self.net_champion_id is not None or bool(self.pareto_frontier_ids)

    @property
    def requests_s2_shortlist(self) -> bool:
        return bool(self.s2_shortlist_ids)


@dataclass(frozen=True, slots=True)
class S1PublicationDecision:
    """Deterministic gate result for the supplied publication claims."""

    publication_allowed: bool
    descriptive_publication_allowed: bool
    completion_ranking_allowed: bool
    economic_ranking_allowed: bool
    s2_shortlist_allowed: bool
    integrity_failures: tuple[str, ...]
    completion_ranking_blockers: tuple[str, ...]
    economic_ranking_blockers: tuple[str, ...]
    shortlist_blockers: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    unavailable_cost_disclosures: tuple[str, ...]

    def require_publishable(self) -> None:
        """Raise one stable error if any supplied claim is blocked."""

        if not self.publication_allowed:
            raise S1PublicationGateError(
                "S1 publication gate blocked: " + "; ".join(self.failure_reasons)
            )


def evaluate_s1_publication_gate(
    scenarios: Sequence[object],
    claims: object,
) -> S1PublicationDecision:
    """Validate all publication invariants and authorize only safe claims.

    An open, unvalued scenario does not prevent publication of honest
    descriptive tables.  It does prevent economic rankings, Pareto claims, and
    the S2 shortlist.  Structural accounting, funnel, lookup, or cost failures
    prevent even descriptive publication.
    """

    raw_values = _sequence_tuple(scenarios, "scenarios")
    values = tuple(
        coerce_s1_publication_scenario_facts(value) for value in raw_values
    )
    if not values:
        raise S1PublicationGateError("scenarios cannot be empty")
    selected_claims = coerce_s1_publication_claims(claims)
    scenario_ids = [value.scenario_id for value in values]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise S1PublicationGateError("scenario_id must be unique")
    ordered = tuple(sorted(values, key=lambda value: value.scenario_id))

    integrity: list[str] = []
    completion_blockers: list[str] = []
    economic_only_blockers: list[str] = []
    shortlist_only_blockers: list[str] = []
    disclosures: list[str] = []

    _check_common_population(ordered, integrity)
    for scenario in ordered:
        prefix = f"scenario[{scenario.scenario_id}]"
        _check_scenario_definition(scenario, prefix, integrity)
        _check_lookup(scenario.lookup, prefix, integrity)
        _check_funnel(
            scenario.funnel,
            prefix,
            integrity,
            completion_blockers,
        )
        _check_accounting(scenario, prefix, integrity)
        _check_costs(scenario.costs, prefix, integrity)
        _check_unmodeled_costs(
            scenario,
            prefix,
            integrity,
            disclosures,
        )
        _check_open_valuation(
            scenario,
            prefix,
            economic_only_blockers,
        )

    _check_common_open_valuation(ordered, economic_only_blockers)
    _check_claims(
        ordered,
        selected_claims,
        integrity,
        shortlist_only_blockers,
    )

    integrity_result = _stable_unique(integrity)
    completion_result = _stable_unique(completion_blockers)
    economic_result = _stable_unique(
        (*completion_result, *economic_only_blockers)
    )
    shortlist_result = _stable_unique(shortlist_only_blockers)
    descriptive_allowed = not integrity_result
    completion_allowed = descriptive_allowed and not completion_result
    economic_allowed = descriptive_allowed and not economic_result
    shortlist_allowed = economic_allowed and not shortlist_result

    requested_failures: list[str] = list(integrity_result)
    if selected_claims.requests_completion_ranking:
        requested_failures.extend(completion_result)
    if selected_claims.requests_economic_ranking:
        requested_failures.extend(economic_result)
    if selected_claims.requests_s2_shortlist:
        requested_failures.extend(economic_result)
        requested_failures.extend(shortlist_result)
    failure_reasons = _stable_unique(requested_failures)
    return S1PublicationDecision(
        publication_allowed=not failure_reasons,
        descriptive_publication_allowed=descriptive_allowed,
        completion_ranking_allowed=completion_allowed,
        economic_ranking_allowed=economic_allowed,
        s2_shortlist_allowed=shortlist_allowed,
        integrity_failures=integrity_result,
        completion_ranking_blockers=completion_result,
        economic_ranking_blockers=economic_result,
        shortlist_blockers=shortlist_result,
        failure_reasons=failure_reasons,
        unavailable_cost_disclosures=_stable_unique(disclosures),
    )


def coerce_s1_publication_scenario_facts(
    value: object,
) -> S1PublicationScenarioFacts:
    """Copy a same-shaped Mapping or dataclass into strict scenario facts."""

    if isinstance(value, S1PublicationScenarioFacts):
        return value
    record = _dataclass_record(
        value,
        S1PublicationScenarioFacts,
        "publication scenario",
    )
    record["lookup"] = _coerce_simple_dataclass(
        record["lookup"],
        S1LookupDenominatorFacts,
        "publication scenario lookup",
    )
    record["funnel"] = _coerce_simple_dataclass(
        record["funnel"],
        S1EntryFunnelFacts,
        "publication scenario funnel",
    )
    record["accounting"] = _coerce_simple_dataclass(
        record["accounting"],
        S1PositionAccountingFacts,
        "publication scenario accounting",
    )
    record["costs"] = _coerce_simple_dataclass(
        record["costs"],
        S1ModeledCostAccountingFacts,
        "publication scenario costs",
    )
    raw_unmodeled = _sequence_tuple(
        record["unmodeled_costs"],
        "publication scenario unmodeled_costs",
    )
    record["unmodeled_costs"] = tuple(
        _coerce_simple_dataclass(
            item,
            S1UnavailableCost,
            "publication scenario unmodeled cost",
        )
        for item in raw_unmodeled
    )
    if record["open_valuation"] is not None:
        record["open_valuation"] = _coerce_simple_dataclass(
            record["open_valuation"],
            S1OpenPositionValuation,
            "publication scenario open valuation",
        )
    return S1PublicationScenarioFacts(**record)  # type: ignore[arg-type]


def coerce_s1_publication_claims(value: object) -> S1PublicationClaims:
    """Copy a same-shaped Mapping or dataclass into strict publication claims."""

    if isinstance(value, S1PublicationClaims):
        return value
    record = _dataclass_record(value, S1PublicationClaims, "publication claims")
    return S1PublicationClaims(**record)  # type: ignore[arg-type]


def _check_common_population(
    scenarios: tuple[S1PublicationScenarioFacts, ...],
    failures: list[str],
) -> None:
    if len({value.reporting_sessions for value in scenarios}) != 1:
        failures.append("publication.reporting_sessions_not_common")
    lookup_shapes = {
        (
            value.lookup.common_mother_product_days,
            value.lookup.tod_bucket_count,
            value.lookup.expected_policy_cells,
            value.lookup.reporting_denominator_cells,
        )
        for value in scenarios
    }
    if len(lookup_shapes) != 1:
        failures.append("publication.lookup_denominator_not_common")
    population_hashes = {
        value.lookup.common_population_sha256 for value in scenarios
    }
    if len(population_hashes) != 1:
        failures.append("publication.lookup_population_identity_not_common")


def _check_scenario_definition(
    scenario: S1PublicationScenarioFacts,
    prefix: str,
    failures: list[str],
) -> None:
    if scenario.cost_horizon == "ungated":
        if scenario.economic_gate_enabled:
            failures.append(f"{prefix}.definition.ungated_gate_enabled")
        if scenario.deployment_shortlist_eligible:
            failures.append(f"{prefix}.definition.ungated_shortlist_eligible")
    elif not scenario.economic_gate_enabled:
        failures.append(f"{prefix}.definition.priced_gate_disabled")


def _check_lookup(
    facts: S1LookupDenominatorFacts,
    prefix: str,
    failures: list[str],
) -> None:
    expected = facts.common_mother_product_days * facts.tod_bucket_count
    if facts.expected_policy_cells != expected:
        failures.append(f"{prefix}.lookup.expected_cells_mismatch")
    if facts.policy_spec_rows != facts.expected_policy_cells:
        failures.append(f"{prefix}.lookup.policy_rows_not_complete")
    if (
        facts.supported_policy_cells + facts.unsupported_policy_cells
        != facts.policy_spec_rows
    ):
        failures.append(f"{prefix}.lookup.support_not_conserved")
    if facts.reporting_denominator_cells != facts.expected_policy_cells:
        failures.append(f"{prefix}.lookup.denominator_dropped_cells")
    if facts.unsupported_no_trade_cells != facts.unsupported_policy_cells:
        failures.append(f"{prefix}.lookup.unsupported_not_retained_as_no_trade")


def _check_funnel(
    facts: S1EntryFunnelFacts,
    prefix: str,
    failures: list[str],
    ranking_blockers: list[str],
) -> None:
    if facts.reservation_attempts != (
        facts.admitted_orders + facts.cap_blocked_attempts
    ):
        failures.append(f"{prefix}.funnel.admission_not_conserved")
    if facts.admitted_orders != facts.sent_orders:
        failures.append(f"{prefix}.funnel.admitted_sent_mismatch")
    if facts.sent_orders > facts.candidate_intents:
        failures.append(f"{prefix}.funnel.sent_exceeds_candidates")
    if facts.blocked_candidate_intents > facts.candidate_intents:
        failures.append(f"{prefix}.funnel.blocked_candidates_exceed_candidates")
    if facts.blocked_candidate_intents > facts.cap_blocked_attempts:
        failures.append(f"{prefix}.funnel.blocked_candidates_exceed_attempts")
    if (
        facts.makerfill_supported_orders + facts.makerfill_unsupported_orders
        != facts.sent_orders
    ):
        failures.append(f"{prefix}.funnel.makerfill_support_not_conserved")
    if facts.filled_orders > facts.makerfill_supported_orders:
        failures.append(f"{prefix}.funnel.fills_exceed_supported")
    terminal_count = (
        facts.filled_orders
        + facts.actual_cancelled_orders
        + facts.session_expired_orders
        + facts.unknown_terminal_orders
    )
    if terminal_count != facts.sent_orders:
        failures.append(f"{prefix}.funnel.order_terminal_not_conserved")
    if facts.unknown_terminal_orders != 0:
        ranking_blockers.append(f"{prefix}.funnel.unknown_terminal_nonzero")


def _check_accounting(
    scenario: S1PublicationScenarioFacts,
    prefix: str,
    failures: list[str],
) -> None:
    facts = scenario.accounting
    if facts.open_or_unresolved != (
        facts.paired_open_positions + facts.naked_unresolved_positions
    ):
        failures.append(f"{prefix}.accounting.open_state_not_conserved")
    terminal_count = (
        facts.same_day_exit_maker_flat
        + facts.cross_day_exit_maker_flat
        + facts.expiry_marks
        + facts.entry_rollbacks
        + facts.other_executable_terminals
    )
    if terminal_count + facts.open_or_unresolved != facts.entry_positions:
        failures.append(f"{prefix}.accounting.position_outcomes_not_conserved")
    if facts.terminal_coverage_numerator != terminal_count:
        failures.append(f"{prefix}.accounting.terminal_coverage_numerator_mismatch")
    if facts.terminal_coverage_denominator != facts.entry_positions:
        failures.append(f"{prefix}.accounting.terminal_coverage_denominator_mismatch")
    if scenario.funnel.filled_orders != facts.entry_positions:
        failures.append(f"{prefix}.accounting.fills_positions_mismatch")
    if facts.naked_unresolved_positions != 0:
        failures.append(f"{prefix}.accounting.naked_unresolved_nonzero")
    costs = scenario.costs
    if terminal_count > 0 and costs.terminal_executed_turnover_twd == 0:
        failures.append(f"{prefix}.accounting.terminal_turnover_missing")
    if facts.paired_open_positions > 0 and costs.open_executed_turnover_twd == 0:
        failures.append(f"{prefix}.accounting.open_turnover_missing")
    if facts.open_or_unresolved == 0 and not all(
        _money_close(value, Decimal(0))
        for value in (
            costs.open_executed_turnover_twd,
            costs.open_execution_actual_commission_twd,
            costs.open_execution_actual_tax_twd,
            costs.open_execution_actual_cost_twd,
        )
    ):
        failures.append(f"{prefix}.accounting.open_cost_without_open_position")


def _check_costs(
    facts: S1ModeledCostAccountingFacts,
    prefix: str,
    failures: list[str],
) -> None:
    if not _money_close(
        facts.executable_terminal_gross_pnl_twd
        + facts.expiry_mark_gross_pnl_twd,
        facts.terminal_gross_pnl_twd,
    ):
        failures.append(f"{prefix}.cost.terminal_gross_not_conserved")
    direct_cost = (
        facts.executable_terminal_commission_twd
        + facts.executable_terminal_tax_twd
        + facts.expiry_mark_commission_twd
        + facts.expiry_mark_tax_twd
    )
    if not _money_close(direct_cost, facts.terminal_modeled_direct_cost_twd):
        failures.append(f"{prefix}.cost.modeled_components_not_conserved")
    executable_net = (
        facts.executable_terminal_gross_pnl_twd
        - facts.executable_terminal_commission_twd
        - facts.executable_terminal_tax_twd
    )
    if not _money_close(executable_net, facts.executable_terminal_net_twd):
        failures.append(f"{prefix}.cost.executable_net_not_conserved")
    expiry_net = (
        facts.expiry_mark_gross_pnl_twd
        - facts.expiry_mark_commission_twd
        - facts.expiry_mark_tax_twd
    )
    if not _money_close(expiry_net, facts.expiry_mark_net_twd):
        failures.append(f"{prefix}.cost.expiry_net_not_conserved")
    expected_net = facts.terminal_gross_pnl_twd - direct_cost
    if not _money_close(expected_net, facts.total_net_twd):
        failures.append(f"{prefix}.cost.gross_cost_net_not_conserved")
    if not _money_close(
        facts.executable_terminal_net_twd + facts.expiry_mark_net_twd,
        facts.total_net_twd,
    ):
        failures.append(f"{prefix}.cost.terminal_net_not_conserved")
    if not _money_close(
        facts.open_execution_actual_commission_twd
        + facts.open_execution_actual_tax_twd,
        facts.open_execution_actual_cost_twd,
    ):
        failures.append(f"{prefix}.cost.open_execution_cost_not_conserved")
    if not _money_close(
        facts.terminal_executed_turnover_twd
        + facts.open_executed_turnover_twd,
        facts.total_executed_turnover_twd,
    ):
        failures.append(f"{prefix}.cost.turnover_not_conserved")


def _check_unmodeled_costs(
    scenario: S1PublicationScenarioFacts,
    prefix: str,
    failures: list[str],
    disclosures: list[str],
) -> None:
    by_id: dict[str, S1UnavailableCost] = {}
    duplicated: set[str] = set()
    for value in scenario.unmodeled_costs:
        if value.cost_id in by_id:
            duplicated.add(value.cost_id)
        else:
            by_id[value.cost_id] = value
    for cost_id in sorted(duplicated):
        failures.append(f"{prefix}.unmodeled_cost[{cost_id}].duplicated")
    for cost_id in REQUIRED_UNMODELED_COST_IDS:
        if cost_id not in by_id:
            failures.append(f"{prefix}.unmodeled_cost[{cost_id}].missing")
    for cost_id in sorted(by_id):
        value = by_id[cost_id]
        if value.amount_twd is not None:
            failures.append(
                f"{prefix}.unmodeled_cost[{cost_id}].amount_must_be_unavailable"
            )
        if value.unavailable_reason is None:
            failures.append(
                f"{prefix}.unmodeled_cost[{cost_id}].reason_missing"
            )
        elif value.amount_twd is None:
            disclosures.append(
                f"{prefix}.unmodeled_cost[{cost_id}]="
                f"{value.unavailable_reason}"
            )


def _check_open_valuation(
    scenario: S1PublicationScenarioFacts,
    prefix: str,
    blockers: list[str],
) -> None:
    open_count = scenario.accounting.paired_open_positions
    valuation = scenario.open_valuation
    expected_ranking_net: Decimal | None = None
    if open_count == 0:
        expected_ranking_net = scenario.costs.total_net_twd
        if valuation is not None:
            if valuation.position_count != 0:
                blockers.append(f"{prefix}.open_valuation.position_count_mismatch")
            if not all(
                _money_close(value, Decimal(0))
                for value in (
                    valuation.gross_mark_pnl_twd,
                    valuation.remaining_exit_cost_twd,
                    valuation.net_mark_pnl_twd,
                )
            ):
                blockers.append(f"{prefix}.open_valuation.nonzero_without_open")
    elif valuation is None:
        blockers.append(f"{prefix}.open_valuation.missing")
    else:
        if valuation.position_count != open_count:
            blockers.append(f"{prefix}.open_valuation.position_count_mismatch")
        if not valuation.comparable_across_scenarios:
            blockers.append(f"{prefix}.open_valuation.not_comparable")
        expected_open_net = (
            valuation.gross_mark_pnl_twd
            - scenario.costs.open_execution_actual_cost_twd
            - valuation.remaining_exit_cost_twd
        )
        if not _money_close(expected_open_net, valuation.net_mark_pnl_twd):
            blockers.append(f"{prefix}.open_valuation.costs_not_conserved")
        else:
            expected_ranking_net = (
                scenario.costs.total_net_twd
                + valuation.net_mark_pnl_twd
            )

    if scenario.economic_ranking_net_twd is None:
        blockers.append(f"{prefix}.economic_ranking.net_missing")
    elif expected_ranking_net is not None and not _money_close(
        scenario.economic_ranking_net_twd,
        expected_ranking_net,
    ):
        blockers.append(f"{prefix}.economic_ranking.net_not_conserved")


def _check_common_open_valuation(
    scenarios: tuple[S1PublicationScenarioFacts, ...],
    blockers: list[str],
) -> None:
    values = tuple(
        value.open_valuation
        for value in scenarios
        if value.accounting.paired_open_positions > 0
        and value.open_valuation is not None
    )
    methods = {value.valuation_method_id for value in values}
    asof_ids = {value.valuation_asof_id for value in values}
    if len(methods) > 1:
        blockers.append("publication.open_valuation_method_not_common")
    if len(asof_ids) > 1:
        blockers.append("publication.open_valuation_asof_not_common")


def _check_claims(
    scenarios: tuple[S1PublicationScenarioFacts, ...],
    claims: S1PublicationClaims,
    failures: list[str],
    shortlist_blockers: list[str],
) -> None:
    by_id = {value.scenario_id: value for value in scenarios}
    for name, identifiers in (
        (
            "completion_champion",
            ()
            if claims.completion_champion_id is None
            else (claims.completion_champion_id,),
        ),
        (
            "net_champion",
            () if claims.net_champion_id is None else (claims.net_champion_id,),
        ),
        ("pareto_frontier", claims.pareto_frontier_ids),
        ("s2_shortlist", claims.s2_shortlist_ids),
    ):
        if len(set(identifiers)) != len(identifiers):
            failures.append(f"claims.{name}.duplicates")
        for identifier in identifiers:
            if identifier not in by_id:
                failures.append(f"claims.{name}.unknown[{identifier}]")
    if len(claims.s2_shortlist_ids) > 2:
        shortlist_blockers.append("claims.s2_shortlist.more_than_two")
    for identifier in claims.s2_shortlist_ids:
        scenario = by_id.get(identifier)
        if scenario is None:
            continue
        if not scenario.deployment_shortlist_eligible:
            shortlist_blockers.append(
                f"claims.s2_shortlist.ineligible[{identifier}]"
            )
        if scenario.cost_horizon == "ungated":
            shortlist_blockers.append(
                f"claims.s2_shortlist.ungated_control[{identifier}]"
            )
    if not (
        claims.requests_completion_ranking
        or claims.requests_economic_ranking
        or claims.requests_s2_shortlist
    ):
        return
    eligible = tuple(
        value for value in scenarios if value.deployment_shortlist_eligible
    )
    if not eligible or any(
        value.economic_ranking_net_twd is None for value in eligible
    ):
        return
    expected = rank_s1_shortlist(
        tuple(
            ScenarioMetrics(
                scenario_id=value.scenario_id,
                completion_numerator=(
                    value.accounting.same_day_exit_maker_flat
                ),
                completion_denominator=value.accounting.entry_positions,
                total_net_twd=value.economic_ranking_net_twd,
                reporting_sessions=value.reporting_sessions,
                hedge_priced_numerator=value.hedge_priced_numerator,
                hedge_priced_denominator=value.hedge_priced_denominator,
            )
            for value in eligible
            if value.economic_ranking_net_twd is not None
        )
    )
    expected_completion = expected.completion_champion.scenario_id
    if (
        claims.completion_champion_id is not None
        and claims.completion_champion_id != expected_completion
    ):
        failures.append("claims.completion_champion.not_actual_champion")
    expected_net = expected.net_champion.scenario_id
    if claims.net_champion_id is not None and claims.net_champion_id != expected_net:
        failures.append("claims.net_champion.not_actual_champion")
    expected_frontier = tuple(value.scenario_id for value in expected.pareto_frontier)
    if claims.pareto_frontier_ids and claims.pareto_frontier_ids != expected_frontier:
        failures.append("claims.pareto_frontier.not_actual_frontier")
    expected_shortlist = tuple(value.scenario_id for value in expected.selected)
    if claims.s2_shortlist_ids and claims.s2_shortlist_ids != expected_shortlist:
        shortlist_blockers.append("claims.s2_shortlist.not_actual_shortlist")


def _stable_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _money_close(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= MONEY_TOLERANCE_TWD


def _money(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise S1PublicationGateError(f"{name} must be a finite Decimal")
    return value


def _nonnegative_money(value: object, name: str) -> Decimal:
    result = _money(value, name)
    if result < 0:
        raise S1PublicationGateError(f"{name} must be non-negative")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S1PublicationGateError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise S1PublicationGateError(f"{name} must be positive")
    return result


def _rate_counts(numerator: object, denominator: object, name: str) -> None:
    left = _nonnegative_integer(numerator, f"{name}_numerator")
    right = _nonnegative_integer(denominator, f"{name}_denominator")
    if left > right:
        raise S1PublicationGateError(f"{name} numerator exceeds denominator")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise S1PublicationGateError(f"{name} must be a lowercase SHA-256")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise S1PublicationGateError(f"{name} must be boolean")
    return value


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise S1PublicationGateError(
            f"{name} must be a nonempty canonical string"
        )
    return value


def _typed_tuple[ValueT](
    values: object,
    expected: type[ValueT],
    name: str,
) -> tuple[ValueT, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(values)
    if any(not isinstance(value, expected) for value in result):
        raise TypeError(f"{name} contains an invalid value")
    return result


def _string_tuple(values: object, name: str) -> tuple[str, ...]:
    return tuple(
        _canonical_text(value, name)
        for value in _sequence_tuple(values, name)
    )


def _sequence_tuple(values: object, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(values)


def _coerce_simple_dataclass[ValueT](
    value: object,
    expected: type[ValueT],
    name: str,
) -> ValueT:
    if isinstance(value, expected):
        return value
    record = _dataclass_record(value, expected, name)
    return expected(**record)  # type: ignore[arg-type]


def _dataclass_record[ValueT](
    value: object,
    expected: type[ValueT],
    name: str,
) -> dict[str, object]:
    if isinstance(value, Mapping):
        record = dict(value)
    elif is_dataclass(value) and not isinstance(value, type):
        record = {
            field.name: getattr(value, field.name)
            for field in fields(value)
        }
    else:
        raise TypeError(f"{name} must be a Mapping or dataclass")
    expected_names = {field.name for field in fields(expected)}
    missing = sorted(expected_names.difference(record))
    extra = sorted(set(record).difference(expected_names))
    if missing or extra:
        raise S1PublicationGateError(
            f"{name} schema mismatch; missing={missing}, extra={extra}"
        )
    return record


__all__ = [
    "MONEY_TOLERANCE_TWD",
    "REQUIRED_UNMODELED_COST_IDS",
    "S1EntryFunnelFacts",
    "S1LookupDenominatorFacts",
    "S1ModeledCostAccountingFacts",
    "S1OpenPositionValuation",
    "S1PositionAccountingFacts",
    "S1PublicationClaims",
    "S1PublicationDecision",
    "S1PublicationGateError",
    "S1PublicationScenarioFacts",
    "S1UnavailableCost",
    "coerce_s1_publication_claims",
    "coerce_s1_publication_scenario_facts",
    "evaluate_s1_publication_gate",
]
