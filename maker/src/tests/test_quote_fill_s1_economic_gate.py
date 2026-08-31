"""Focused contracts for the causal S1 actual-send cost gate."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..quote_fill.layered import EventCursor
from ..quote_fill.policy_spec import PolicySpec
from ..quote_fill.s1_economic_gate import (
    UNGATED_CONTROL_RULE,
    S1EconomicGateEstimate,
    S1EconomicGateRule,
    evaluate_s1_entry_economics,
)
from ..quote_fill.s1_target import S1_ROUTE, build_s1_spot_bid_target
from ..quote_fill.targets import absolute_price_tick

DATE = "20260505"
TAIPEI = ZoneInfo("Asia/Taipei")


def _cursor() -> EventCursor:
    value = datetime(2026, 5, 5, 9, 5, tzinfo=TAIPEI) + timedelta(seconds=1)
    return EventCursor(int(value.timestamp()) * 1_000_000_000, 4, 1)


def _target(*, spot_bid: float = 100.5, future_ask: float = 101.1):
    cursor = _cursor()
    spec = PolicySpec(
        Date=DATE,
        ValueCode="2330",
        QuoteCode="CDFE6",
        entry_tod_bucket="0905_1000",
        policy_id="q95",
        kind="quantile",
        upper_distance_bp=30.0,
        lower_distance_bp=0.0,
        upper_source_id="Q2_trail20_date_equal",
        lower_source_id="C0_center",
        upper_source_asof_date="20260504",
        lower_source_asof_date="20260504",
        combined_source_asof_date="20260504",
        boundary_quantile=95,
    )
    return build_s1_spot_bid_target(
        spec,
        date=DATE,
        value_code="2330",
        quote_code="CDFE6",
        route=S1_ROUTE,
        actual_new_send_cursor=cursor,
        causal_anchor_basis_bp=10.0,
        fut_exec_bid=101.0,
        fut_exec_ask=future_ask,
        spot_bid=spot_bid,
        spot_ask=101.5,
        contract_size_shares=1_000.0,
    )


class S1EconomicGateTest(unittest.TestCase):
    def test_same_day_gate_uses_exact_cost_decomposition_and_strict_floor(self) -> None:
        target = _target()
        rule = S1EconomicGateRule("same_day", 5.0, True, True)
        estimate = evaluate_s1_entry_economics(
            target,
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=rule,
        )

        self.assertTrue(estimate.gate_open)
        self.assertEqual(estimate.status, "eligible")
        self.assertEqual(estimate.future_buy_vwap, target.fut_exec_ask_at_actual_new)
        self.assertEqual(
            estimate.hypothetical_exit_spot_target,
            target.frozen_exit_target_price,
        )
        self.assertEqual(
            estimate.frozen_exit_spot_target_tick,
            target.frozen_exit_absolute_price_tick,
        )
        self.assertEqual(
            estimate.frozen_exit_spot_target_tick,
            absolute_price_tick(estimate.hypothetical_exit_spot_target),
        )
        self.assertIsNotNone(estimate.same_day_expected_margin_bp)
        self.assertGreater(estimate.same_day_expected_margin_bp or 0.0, 5.0)
        self.assertAlmostEqual(
            (estimate.gross_expected_twd or 0.0)
            - (estimate.same_day_modeled_cost_twd or 0.0),
            estimate.same_day_expected_margin_twd or 0.0,
            places=9,
        )
        self.assertAlmostEqual(
            (estimate.spot_round_trip_commission_twd or 0.0)
            + (estimate.same_day_spot_sell_tax_twd or 0.0)
            + (estimate.futures_round_trip_tax_twd or 0.0)
            + (estimate.futures_round_trip_commission_twd or 0.0),
            estimate.same_day_modeled_cost_twd or 0.0,
            places=9,
        )

        blocked = evaluate_s1_entry_economics(
            target,
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=S1EconomicGateRule("same_day", 20.0, True, True),
        )
        self.assertFalse(blocked.gate_open)
        self.assertEqual(blocked.status, "below_floor")

    def test_same_day_and_overnight_tax_are_separate(self) -> None:
        estimate = evaluate_s1_entry_economics(
            _target(),
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=S1EconomicGateRule("overnight", 0.0, True, True),
        )

        self.assertGreater(
            estimate.overnight_spot_sell_tax_twd or 0.0,
            estimate.same_day_spot_sell_tax_twd or 0.0,
        )
        self.assertGreater(
            estimate.same_day_expected_margin_bp or 0.0,
            estimate.overnight_expected_margin_bp or 0.0,
        )
        self.assertAlmostEqual(
            estimate.selected_expected_margin_bp or 0.0,
            estimate.overnight_expected_margin_bp or 0.0,
            places=12,
        )

    def test_illegal_frozen_exit_fails_closed(self) -> None:
        rule = S1EconomicGateRule("same_day", 0.0, True, True)
        non_passive = evaluate_s1_entry_economics(
            _target(spot_bid=101.0),
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=rule,
        )
        self.assertFalse(non_passive.gate_open)
        self.assertEqual(non_passive.status, "route_ineligible")
        self.assertEqual(
            non_passive.reason,
            "frozen_exit_target_not_passive",
        )

    def test_ungated_control_remains_sendable_but_never_shortlist_eligible(
        self,
    ) -> None:
        estimate = evaluate_s1_entry_economics(
            _target(),
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=UNGATED_CONTROL_RULE,
        )
        self.assertTrue(estimate.gate_open)
        self.assertEqual(estimate.status, "ungated_priced")
        self.assertFalse(estimate.deployment_shortlist_eligible)

    def test_rule_contract_rejects_ungated_shortlist_and_missing_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "shortlist"):
            S1EconomicGateRule("ungated", None, False, True)
        with self.assertRaisesRegex(ValueError, "finite"):
            S1EconomicGateRule("same_day", None, True, True)

    def test_versioned_audit_record_roundtrip_preserves_evaluation_stage(self) -> None:
        estimate = evaluate_s1_entry_economics(
            _target(),
            observation_cursor=_cursor(),
            spot_reference_price=100.0,
            rule=S1EconomicGateRule("same_day", 5.0, True, True),
            evaluation_stage="decision_observation",
        )

        decoded = S1EconomicGateEstimate.from_record(estimate.to_record())

        self.assertEqual(decoded, estimate)
        self.assertEqual(decoded.evaluation_stage, "decision_observation")
        drifted = estimate.to_record()
        drifted["schema_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "version"):
            S1EconomicGateEstimate.from_record(drifted)
        mismatched = estimate.to_record()
        mismatched["frozen_exit_spot_target_tick"] = (
            estimate.frozen_exit_spot_target_tick + 1
        )
        with self.assertRaisesRegex(ValueError, "price/tick mismatch"):
            S1EconomicGateEstimate.from_record(mismatched)


if __name__ == "__main__":
    unittest.main()
