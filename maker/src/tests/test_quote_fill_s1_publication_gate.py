"""Focused tests for the pure S1 publication gate."""

from __future__ import annotations

import unittest
from dataclasses import asdict, fields, make_dataclass, replace
from decimal import Decimal

from ..quote_fill.s1_publication_gate import (
    REQUIRED_UNMODELED_COST_IDS,
    S1EntryFunnelFacts,
    S1LookupDenominatorFacts,
    S1ModeledCostAccountingFacts,
    S1OpenPositionValuation,
    S1PositionAccountingFacts,
    S1PublicationClaims,
    S1PublicationGateError,
    S1PublicationScenarioFacts,
    S1UnavailableCost,
    evaluate_s1_publication_gate,
)


def _decimal(value: str | int) -> Decimal:
    return Decimal(str(value))


def _lookup() -> S1LookupDenominatorFacts:
    return S1LookupDenominatorFacts(
        common_mother_product_days=10,
        tod_bucket_count=4,
        expected_policy_cells=40,
        policy_spec_rows=40,
        supported_policy_cells=35,
        unsupported_policy_cells=5,
        reporting_denominator_cells=40,
        unsupported_no_trade_cells=5,
        common_population_sha256="a" * 64,
    )


def _funnel() -> S1EntryFunnelFacts:
    return S1EntryFunnelFacts(
        candidate_intents=20,
        reservation_attempts=12,
        admitted_orders=10,
        cap_blocked_attempts=2,
        blocked_candidate_intents=2,
        sent_orders=10,
        makerfill_supported_orders=8,
        makerfill_unsupported_orders=2,
        filled_orders=4,
        actual_cancelled_orders=3,
        session_expired_orders=3,
        unknown_terminal_orders=0,
    )


def _accounting(*, open_positions: int = 0) -> S1PositionAccountingFacts:
    if open_positions:
        same_day = 2
        cross_day = 1
        rollbacks = 0
        terminal_count = 3
    else:
        same_day = 2
        cross_day = 1
        rollbacks = 1
        terminal_count = 4
    return S1PositionAccountingFacts(
        entry_positions=4,
        same_day_exit_maker_flat=same_day,
        cross_day_exit_maker_flat=cross_day,
        expiry_marks=0,
        entry_rollbacks=rollbacks,
        other_executable_terminals=0,
        paired_open_positions=open_positions,
        naked_unresolved_positions=0,
        open_or_unresolved=open_positions,
        terminal_coverage_numerator=terminal_count,
        terminal_coverage_denominator=4,
    )


def _costs(*, open_positions: int = 0) -> S1ModeledCostAccountingFacts:
    return S1ModeledCostAccountingFacts(
        executable_terminal_gross_pnl_twd=_decimal(85),
        expiry_mark_gross_pnl_twd=_decimal(15),
        terminal_gross_pnl_twd=_decimal(100),
        executable_terminal_commission_twd=_decimal(10),
        executable_terminal_tax_twd=_decimal(5),
        expiry_mark_commission_twd=_decimal(4),
        expiry_mark_tax_twd=_decimal(1),
        terminal_modeled_direct_cost_twd=_decimal(20),
        executable_terminal_net_twd=_decimal(70),
        expiry_mark_net_twd=_decimal(10),
        total_net_twd=_decimal(80),
        terminal_executed_turnover_twd=_decimal(1000),
        open_executed_turnover_twd=(
            _decimal(200) if open_positions else _decimal(0)
        ),
        total_executed_turnover_twd=(
            _decimal(1200) if open_positions else _decimal(1000)
        ),
        open_execution_actual_commission_twd=(
            _decimal(4) if open_positions else _decimal(0)
        ),
        open_execution_actual_tax_twd=(
            _decimal(1) if open_positions else _decimal(0)
        ),
        open_execution_actual_cost_twd=(
            _decimal(5) if open_positions else _decimal(0)
        ),
    )


def _unmodeled() -> tuple[S1UnavailableCost, ...]:
    return tuple(
        S1UnavailableCost(
            cost_id=cost_id,
            amount_twd=None,
            unavailable_reason=f"not_modeled:{cost_id}",
        )
        for cost_id in REQUIRED_UNMODELED_COST_IDS
    )


def _valuation(
    *,
    method: str = "official_common_horizon_executable_mark_v1",
    asof: str = "20260831T133000+0800",
) -> S1OpenPositionValuation:
    return S1OpenPositionValuation(
        position_count=1,
        valuation_method_id=method,
        valuation_asof_id=asof,
        comparable_across_scenarios=True,
        gross_mark_pnl_twd=_decimal(30),
        remaining_exit_cost_twd=_decimal(4),
        net_mark_pnl_twd=_decimal(21),
    )


def _scenario(
    scenario_id: str = "q95_C0_sd_f5",
    *,
    open_positions: int = 0,
    valuation: S1OpenPositionValuation | None = None,
    economic_ranking_net_twd: Decimal | None = None,
    cost_horizon: str = "same_day",
    economic_gate_enabled: bool = True,
    deployment_shortlist_eligible: bool = True,
) -> S1PublicationScenarioFacts:
    if economic_ranking_net_twd is None and open_positions == 0:
        economic_ranking_net_twd = _decimal(80)
    return S1PublicationScenarioFacts(
        scenario_id=scenario_id,
        reporting_sessions=71,
        cost_horizon=cost_horizon,
        economic_gate_enabled=economic_gate_enabled,
        deployment_shortlist_eligible=deployment_shortlist_eligible,
        lookup=_lookup(),
        funnel=_funnel(),
        accounting=_accounting(open_positions=open_positions),
        costs=_costs(open_positions=open_positions),
        unmodeled_costs=_unmodeled(),
        open_valuation=valuation,
        economic_ranking_net_twd=economic_ranking_net_twd,
        hedge_priced_numerator=9,
        hedge_priced_denominator=10,
    )


def _claims(scenario_id: str = "q95_C0_sd_f5") -> S1PublicationClaims:
    return S1PublicationClaims(
        completion_champion_id=scenario_id,
        net_champion_id=scenario_id,
        pareto_frontier_ids=(scenario_id,),
        s2_shortlist_ids=(scenario_id,),
    )


class S1PublicationGateTest(unittest.TestCase):
    def test_valid_gate_retains_unsupported_and_discloses_missing_costs(
        self,
    ) -> None:
        decision = evaluate_s1_publication_gate((_scenario(),), _claims())

        self.assertTrue(decision.publication_allowed)
        self.assertTrue(decision.descriptive_publication_allowed)
        self.assertTrue(decision.completion_ranking_allowed)
        self.assertTrue(decision.economic_ranking_allowed)
        self.assertTrue(decision.s2_shortlist_allowed)
        self.assertEqual(decision.failure_reasons, ())
        self.assertEqual(
            len(decision.unavailable_cost_disclosures),
            len(REQUIRED_UNMODELED_COST_IDS),
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].unmodeled_cost[overnight_financing]="
            "not_modeled:overnight_financing",
            decision.unavailable_cost_disclosures,
        )
        decision.require_publishable()

    def test_lookup_unsupported_cells_cannot_be_dropped_or_hidden(self) -> None:
        lookup = replace(
            _lookup(),
            reporting_denominator_cells=35,
            unsupported_no_trade_cells=0,
        )
        scenario = replace(_scenario(), lookup=lookup)

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertFalse(decision.descriptive_publication_allowed)
        self.assertFalse(decision.publication_allowed)
        self.assertEqual(
            decision.integrity_failures,
            (
                "scenario[q95_C0_sd_f5].lookup.denominator_dropped_cells",
                (
                    "scenario[q95_C0_sd_f5].lookup."
                    "unsupported_not_retained_as_no_trade"
                ),
            ),
        )

    def test_funnel_conservation_fails_closed(self) -> None:
        funnel = replace(
            _funnel(),
            makerfill_unsupported_orders=1,
            actual_cancelled_orders=2,
        )
        scenario = replace(_scenario(), funnel=funnel)

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertFalse(decision.descriptive_publication_allowed)
        self.assertIn(
            "scenario[q95_C0_sd_f5].funnel.makerfill_support_not_conserved",
            decision.integrity_failures,
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].funnel.order_terminal_not_conserved",
            decision.integrity_failures,
        )

    def test_unknown_terminal_allows_description_but_blocks_rankings(self) -> None:
        funnel = replace(
            _funnel(),
            actual_cancelled_orders=2,
            unknown_terminal_orders=1,
        )
        scenario = replace(_scenario(), funnel=funnel)
        blocker = (
            "scenario[q95_C0_sd_f5].funnel.unknown_terminal_nonzero"
        )

        ranked = evaluate_s1_publication_gate((scenario,), _claims())
        descriptive = evaluate_s1_publication_gate(
            (scenario,),
            S1PublicationClaims(),
        )

        self.assertTrue(ranked.descriptive_publication_allowed)
        self.assertFalse(ranked.completion_ranking_allowed)
        self.assertFalse(ranked.economic_ranking_allowed)
        self.assertFalse(ranked.publication_allowed)
        self.assertEqual(ranked.completion_ranking_blockers, (blocker,))
        self.assertTrue(descriptive.publication_allowed)

    def test_position_and_cost_conservation_fail_closed(self) -> None:
        accounting = replace(
            _accounting(),
            terminal_coverage_numerator=3,
        )
        costs = replace(
            _costs(),
            terminal_modeled_direct_cost_twd=_decimal(19),
            total_net_twd=_decimal(79),
            total_executed_turnover_twd=_decimal(999),
        )
        scenario = replace(
            _scenario(),
            accounting=accounting,
            costs=costs,
        )

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertFalse(decision.descriptive_publication_allowed)
        self.assertIn(
            "scenario[q95_C0_sd_f5].accounting."
            "terminal_coverage_numerator_mismatch",
            decision.integrity_failures,
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].cost.modeled_components_not_conserved",
            decision.integrity_failures,
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].cost.terminal_net_not_conserved",
            decision.integrity_failures,
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].cost.turnover_not_conserved",
            decision.integrity_failures,
        )

    def test_unmodeled_cost_zero_is_not_an_unavailable_value(self) -> None:
        values = list(_unmodeled())
        values[0] = replace(values[0], amount_twd=_decimal(0))
        scenario = replace(_scenario(), unmodeled_costs=tuple(values))

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertFalse(decision.descriptive_publication_allowed)
        self.assertIn(
            "scenario[q95_C0_sd_f5].unmodeled_cost["
            "futures_margin_opportunity_cost].amount_must_be_unavailable",
            decision.integrity_failures,
        )

    def test_required_unmodeled_cost_needs_an_explicit_reason(self) -> None:
        missing = tuple(
            value
            for value in _unmodeled()
            if value.cost_id != "spot_borrow_cost"
        )
        no_reason = replace(
            next(
                value
                for value in missing
                if value.cost_id == "overnight_financing"
            ),
            unavailable_reason=None,
        )
        values = tuple(
            no_reason if value.cost_id == no_reason.cost_id else value
            for value in missing
        )
        scenario = replace(_scenario(), unmodeled_costs=values)

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertIn(
            "scenario[q95_C0_sd_f5].unmodeled_cost[spot_borrow_cost].missing",
            decision.integrity_failures,
        )
        self.assertIn(
            "scenario[q95_C0_sd_f5].unmodeled_cost["
            "overnight_financing].reason_missing",
            decision.integrity_failures,
        )

    def test_ungated_control_cannot_enter_s2_shortlist(self) -> None:
        control = _scenario(
            "ctrl_q95_C0_ungated",
            cost_horizon="ungated",
            economic_gate_enabled=False,
            deployment_shortlist_eligible=False,
        )
        claims = S1PublicationClaims(
            completion_champion_id="ctrl_q95_C0_ungated",
            net_champion_id="ctrl_q95_C0_ungated",
            pareto_frontier_ids=("ctrl_q95_C0_ungated",),
            s2_shortlist_ids=("ctrl_q95_C0_ungated",),
        )

        decision = evaluate_s1_publication_gate((control,), claims)

        self.assertTrue(decision.economic_ranking_allowed)
        self.assertFalse(decision.s2_shortlist_allowed)
        self.assertFalse(decision.publication_allowed)
        self.assertEqual(
            decision.shortlist_blockers,
            (
                "claims.s2_shortlist.ineligible[ctrl_q95_C0_ungated]",
                "claims.s2_shortlist.ungated_control[ctrl_q95_C0_ungated]",
            ),
        )

    def test_open_without_comparable_exit_costed_mark_blocks_economics(self) -> None:
        scenario = _scenario(
            open_positions=1,
            valuation=None,
            economic_ranking_net_twd=_decimal(80),
        )

        ranked = evaluate_s1_publication_gate((scenario,), _claims())
        descriptive = evaluate_s1_publication_gate(
            (scenario,),
            S1PublicationClaims(completion_champion_id=scenario.scenario_id),
        )

        self.assertTrue(ranked.descriptive_publication_allowed)
        self.assertTrue(ranked.completion_ranking_allowed)
        self.assertFalse(ranked.economic_ranking_allowed)
        self.assertFalse(ranked.s2_shortlist_allowed)
        self.assertFalse(ranked.publication_allowed)
        self.assertEqual(
            ranked.economic_ranking_blockers,
            ("scenario[q95_C0_sd_f5].open_valuation.missing",),
        )
        self.assertTrue(descriptive.publication_allowed)

    def test_open_mark_includes_incurred_and_remaining_exit_costs(self) -> None:
        scenario = _scenario(
            open_positions=1,
            valuation=_valuation(),
            economic_ranking_net_twd=_decimal(101),
        )

        decision = evaluate_s1_publication_gate((scenario,), _claims())

        self.assertTrue(decision.publication_allowed)
        self.assertTrue(decision.economic_ranking_allowed)
        self.assertTrue(decision.s2_shortlist_allowed)

        bad_valuation = replace(
            _valuation(),
            net_mark_pnl_twd=_decimal(26),
        )
        bad = replace(scenario, open_valuation=bad_valuation)
        blocked = evaluate_s1_publication_gate((bad,), _claims())
        self.assertIn(
            "scenario[q95_C0_sd_f5].open_valuation.costs_not_conserved",
            blocked.economic_ranking_blockers,
        )

    def test_open_valuation_method_and_horizon_must_be_common(self) -> None:
        first = _scenario(
            "q95_C0_sd_f5",
            open_positions=1,
            valuation=_valuation(method="method_a", asof="horizon_a"),
            economic_ranking_net_twd=_decimal(101),
        )
        second = _scenario(
            "q80_C0_sd_f5",
            open_positions=1,
            valuation=_valuation(method="method_b", asof="horizon_b"),
            economic_ranking_net_twd=_decimal(101),
        )
        claims = S1PublicationClaims(
            net_champion_id=first.scenario_id,
            pareto_frontier_ids=(first.scenario_id, second.scenario_id),
        )

        decision = evaluate_s1_publication_gate((second, first), claims)

        self.assertFalse(decision.economic_ranking_allowed)
        self.assertIn(
            "publication.open_valuation_method_not_common",
            decision.economic_ranking_blockers,
        )
        self.assertIn(
            "publication.open_valuation_asof_not_common",
            decision.economic_ranking_blockers,
        )

    def test_error_message_is_deterministic_across_input_order(self) -> None:
        first = replace(
            _scenario("z_policy"),
            lookup=replace(_lookup(), unsupported_no_trade_cells=0),
        )
        second = replace(
            _scenario("a_policy"),
            funnel=replace(
                _funnel(),
                actual_cancelled_orders=2,
                unknown_terminal_orders=1,
            ),
        )
        claims = S1PublicationClaims(
            completion_champion_id="a_policy",
            net_champion_id="z_policy",
        )
        forward = evaluate_s1_publication_gate((first, second), claims)
        reverse = evaluate_s1_publication_gate((second, first), claims)

        self.assertEqual(forward, reverse)
        with self.assertRaisesRegex(
            S1PublicationGateError,
            "S1 publication gate blocked: claims.net_champion",
        ):
            forward.require_publishable()

    def test_common_population_hash_mismatch_blocks_description(self) -> None:
        first = _scenario("policy_a")
        second = replace(
            _scenario("policy_b"),
            lookup=replace(_lookup(), common_population_sha256="b" * 64),
        )

        decision = evaluate_s1_publication_gate(
            (first, second),
            S1PublicationClaims(),
        )

        self.assertFalse(decision.descriptive_publication_allowed)
        self.assertIn(
            "publication.lookup_population_identity_not_common",
            decision.integrity_failures,
        )

    def test_claimed_champions_frontier_and_shortlist_are_recomputed(self) -> None:
        first = _scenario("policy_a")
        second = _scenario("policy_b")
        wrong = S1PublicationClaims(
            completion_champion_id="policy_b",
            net_champion_id="policy_b",
            pareto_frontier_ids=("policy_b",),
            s2_shortlist_ids=("policy_b",),
        )

        decision = evaluate_s1_publication_gate((first, second), wrong)

        self.assertIn(
            "claims.completion_champion.not_actual_champion",
            decision.integrity_failures,
        )
        self.assertIn(
            "claims.net_champion.not_actual_champion",
            decision.integrity_failures,
        )
        self.assertIn(
            "claims.pareto_frontier.not_actual_frontier",
            decision.integrity_failures,
        )
        self.assertIn(
            "claims.s2_shortlist.not_actual_shortlist",
            decision.shortlist_blockers,
        )

    def test_money_inputs_are_decimal_and_s2_has_at_most_two_ids(self) -> None:
        with self.assertRaisesRegex(
            S1PublicationGateError,
            "terminal_gross_pnl_twd must be a finite Decimal",
        ):
            replace(_costs(), terminal_gross_pnl_twd=100)  # type: ignore[arg-type]

        scenarios = tuple(_scenario(f"policy_{index}") for index in range(3))
        claims = S1PublicationClaims(
            s2_shortlist_ids=tuple(value.scenario_id for value in scenarios)
        )
        decision = evaluate_s1_publication_gate(scenarios, claims)
        self.assertIn(
            "claims.s2_shortlist.more_than_two",
            decision.shortlist_blockers,
        )

    def test_same_shaped_mapping_and_dataclass_inputs_are_supported(self) -> None:
        scenario = _scenario()
        claims = _claims()
        mapped = evaluate_s1_publication_gate(
            (asdict(scenario),),
            asdict(claims),
        )

        scenario_record_type = make_dataclass(
            "ScenarioRecord",
            [(field.name, object) for field in fields(scenario)],
            frozen=True,
            slots=True,
        )
        scenario_record = scenario_record_type(
            **{
                field.name: getattr(scenario, field.name)
                for field in fields(scenario)
            }
        )
        dataclass_decision = evaluate_s1_publication_gate(
            (scenario_record,),
            claims,
        )

        self.assertTrue(mapped.publication_allowed)
        self.assertEqual(mapped, dataclass_decision)

        malformed = asdict(scenario)
        malformed["unexpected"] = 1
        with self.assertRaisesRegex(
            S1PublicationGateError,
            "publication scenario schema mismatch",
        ):
            evaluate_s1_publication_gate((malformed,), claims)


if __name__ == "__main__":
    unittest.main()
