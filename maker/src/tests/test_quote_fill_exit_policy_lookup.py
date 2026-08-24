"""Tests for the D-safe conditional exit-policy lookup checkpoint."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.exit_maker_terminal import (
    FUTURE_BID_EXIT_ROUTE,
    ExitMakerTerminalConfig,
    canonical_exit_raw_candidate_fact_id,
)
from maker.src.quote_fill.exit_policy_lookup import (
    ExitPolicyLookupConfig,
    attach_overnight_labels,
    build_daily_prequential_lookup,
    build_exit_policy_label_tables,
)


DATE = "20260521"


def _action(
    policy_id: str,
    *,
    q: str = "q80",
    quantile: int = 80,
    rank: str = "at_bbo",
) -> dict[str, object]:
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "route": "future_ask_spot_taker",
        "parameter_version": "boundary-v1",
        "source_asof_date": "20260520",
        "lookup_action_id": q,
        "boundary_quantile": quantile,
        "raw_order_fact_id": f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "rank_bucket": rank,
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
        "intended_quantity": 1,
        "submit_recv_time_ns": 1_000_000_000,
        "terminal_recv_time_ns": 2_000_000_000,
        "queue_known": True,
        "any_fill": True,
        "full_fill": True,
        "partial_fill": False,
        "cancel_required": False,
        "entry_hedge_status": "executable",
        "entry_hedge_signed_total_slippage_bp": 1.25,
        "entry_hedge_contract_size_shares": 2_000,
        "entry_spot_price": 115.0,
    }


def _policy(
    entry_policy_id: str,
    *,
    branch: str,
    nominal_branch: str | None = None,
    gross_twd: float | None = None,
    trial_suffix: str = "",
) -> dict[str, object]:
    raw_id = f"raw-{entry_policy_id}"
    winner = branch in {"flat_same_day", "cancel_race_unknown"}
    winner_epoch = 42 if winner else None
    winner_tick = 2_340 if winner else None
    winner_recv = 3_000_000_000 if winner else None
    winner_sequence = 17 if winner else None
    winner_row = 23 if winner else None
    winner_raw_id = (
        canonical_exit_raw_candidate_fact_id(
            entry_raw_order_fact_id=raw_id,
            exit_route=FUTURE_BID_EXIT_ROUTE,
            spread_pair_epoch=winner_epoch,
            target_price_tick=winner_tick,
            submit_recv_time_ns=winner_recv,
            submit_event_sequence=winner_sequence,
            submit_row_index=winner_row,
        )
        if winner
        else None
    )
    terminal = branch == "flat_same_day"
    carry = branch in {
        "carry_at_eod_cancel_unconfirmed",
        "carry_at_eod_no_admission",
    }
    cancel_race = branch == "cancel_race_unknown"
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "entry_route": "future_ask_spot_taker",
        "entry_policy_generation_id": entry_policy_id,
        "entry_raw_order_fact_id": raw_id,
        "exit_rule_id": "frozen_center",
        "exit_route": FUTURE_BID_EXIT_ROUTE,
        "exit_policy_trial_id": f"trial-{entry_policy_id}{trial_suffix}",
        "exit_threshold_basis_bp": 129.95,
        "exit_rule_source_asof_date": "20260520",
        "exit_lifecycle_policy_version": "static-one-order-v1",
        "exit_queue_scenario": "displayed-independent-v1",
        "exit_style": "maker_taker",
        "exit_maker_quantity": 1,
        "exit_hedge_quantity": 2,
        "raw_candidate_count": 1 if winner else 0,
        "oco_winner_raw_candidate_fact_id": winner_raw_id,
        "oco_winner_spread_pair_epoch": winner_epoch,
        "oco_winner_target_price_tick": winner_tick,
        "oco_winner_submit_recv_time_ns": winner_recv,
        "oco_winner_submit_event_sequence": winner_sequence,
        "oco_winner_submit_row_index": winner_row,
        "oco_winner_raw_identity_excludes_rule": winner,
        "oco_position_projection_safe": terminal,
        "oco_active_sibling_cancel_count": 1 if cancel_race else 0,
        "prior_unacked_cancel_count_before_winner": 0,
        "nominal_instant_cancel_v0_branch": nominal_branch or branch,
        "branch_status": branch,
        "terminal_outcome": terminal,
        "needs_next_session_label": carry or cancel_race,
        "exit_decision_time_ns": (
            5_000_000_000 if terminal or cancel_race else None
        ),
        "gross_cycle_pnl_twd": gross_twd,
        "exit_hedge_status": "executable" if terminal or cancel_race else None,
        "exit_hedge_signed_total_slippage_bp": (
            2.5 if terminal or cancel_race else None
        ),
        "cancel_ack_observed": False,
        "cancel_race_modeled": False,
        "joint_volume_allocated": False,
        "instant_cancel_v0": True,
    }


def _overnight(trial_id: str) -> dict[str, object]:
    return {
        "exit_policy_trial_id": trial_id,
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "entry_route": "future_ask_spot_taker",
        "exit_rule_id": "frozen_center",
        "exit_route": FUTURE_BID_EXIT_ROUTE,
        "source_branch_status": "carry_at_eod_cancel_unconfirmed",
        "outcome_status": "known",
        "terminal_branch": "overnight_exit",
        "label_end_date": "20260523",
        "filled_cashflow_before_cost_bp": 25.0,
        "label_status": "overnight_exit",
        "unresolved_reason": None,
    }


class ExitPolicyLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.terminal_config = ExitMakerTerminalConfig(
            lifecycle_policy_version="entry-layered-v1",
            queue_scenario="entry-displayed-independent-v1",
        )
        self.lookup_config = ExitPolicyLookupConfig(
            lookback_sessions=1,
            min_history_sessions=1,
            min_group_sessions=1,
            min_known_paths=1,
        )

    def test_strict_cancel_race_stays_unknown_and_v0_is_separate(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy = pl.DataFrame(
            [
                _policy(
                    "entry-1",
                    branch="cancel_race_unknown",
                    nominal_branch="flat_same_day",
                    gross_twd=2_000.0,
                )
            ]
        )
        labels = build_exit_policy_label_tables(
            action,
            policy,
            terminal_config=self.terminal_config,
        )
        strict = labels.strict.row(0, named=True)
        nominal = labels.nominal_v0.row(0, named=True)
        self.assertEqual(strict["outcome_status"], "unknown")
        self.assertIsNone(strict["actual_four_leg_gross_bp"])
        self.assertFalse(strict["terminal_model_assumption"])
        self.assertEqual(nominal["outcome_status"], "known")
        self.assertEqual(nominal["terminal_branch"], "same_day_target_exit")
        self.assertTrue(nominal["terminal_model_assumption"])
        self.assertTrue(nominal["strict_cancel_ambiguity_overridden"])
        self.assertFalse(nominal["production_eligible"])
        self.assertFalse(nominal["ev_ready"])

        strict_lookup = build_daily_prequential_lookup(
            labels.strict,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=["20260522"],
        ).row(0, named=True)
        nominal_lookup = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=["20260522"],
        ).row(0, named=True)
        self.assertEqual(strict_lookup["n_outcome_unknown"], 1)
        self.assertIsNone(strict_lookup["conditional_known_gross_mean_bp"])
        self.assertIsNone(strict_lookup["gross_mean_identification_lower_bp"])
        self.assertEqual(
            strict_lookup["cashflow_identification_status"],
            "unbounded_without_unresolved_cashflow_limits",
        )
        self.assertEqual(nominal_lookup["n_outcome_known"], 1)
        self.assertAlmostEqual(
            nominal_lookup["conditional_known_gross_mean_bp"],
            2_000.0 / 230_000.0 * 10_000.0,
        )
        self.assertEqual(
            nominal_lookup["ev_status"],
            "nominal_instant_cancel_model_assumption",
        )
        self.assertFalse(nominal_lookup["ev_ready"])
        self.assertIsNone(nominal_lookup["expected_net_cashflow_bp"])

    def test_overnight_label_enters_only_after_strict_maturity(self) -> None:
        action = pl.DataFrame([_action("entry-carry")])
        policy = pl.DataFrame(
            [
                _policy(
                    "entry-carry",
                    branch="carry_at_eod_cancel_unconfirmed",
                )
            ]
        )
        labels = build_exit_policy_label_tables(
            action,
            policy,
            terminal_config=self.terminal_config,
        )
        strict = attach_overnight_labels(
            labels.strict,
            pl.DataFrame([_overnight("trial-entry-carry")]),
        )
        self.assertEqual(strict.item(0, "outcome_status"), "known")
        self.assertEqual(strict.item(0, "label_end_date"), "20260523")
        self.assertAlmostEqual(strict.item(0, "actual_four_leg_gross_bp"), 25.0)
        sessions = [DATE, "20260522", "20260523", "20260525"]
        before = build_daily_prequential_lookup(
            strict,
            sessions,
            ExitPolicyLookupConfig(
                lookback_sessions=3,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
            ),
            asof_dates=["20260523"],
        )
        after = build_daily_prequential_lookup(
            strict,
            sessions,
            ExitPolicyLookupConfig(
                lookback_sessions=3,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
            ),
            asof_dates=["20260525"],
        )
        self.assertEqual(before.height, 1)
        self.assertEqual(before.item(0, "n_labels_pending"), 1)
        self.assertEqual(before.item(0, "n_outcome_known"), 0)
        self.assertEqual(after.item(0, "n_overnight_terminal_known"), 1)
        self.assertEqual(after.item(0, "n_labels_pending"), 0)
        self.assertLess(after.item(0, "label_cutoff_date"), "20260525")

    def test_primary_key_collapses_state_and_optional_table_keeps_it(self) -> None:
        actions = pl.DataFrame(
            [
                _action("entry-a", rank="at_bbo"),
                _action("entry-b", rank="behind"),
            ]
        )
        policies = pl.DataFrame(
            [
                _policy("entry-a", branch="flat_same_day", gross_twd=1_000.0),
                _policy("entry-b", branch="flat_same_day", gross_twd=2_000.0),
            ]
        )
        labels = build_exit_policy_label_tables(
            actions,
            policies,
            terminal_config=self.terminal_config,
        )
        primary = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=["20260522"],
        )
        state = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            self.lookup_config,
            include_causal_state=True,
            asof_dates=["20260522"],
        )
        self.assertEqual(primary.height, 1)
        self.assertEqual(primary.item(0, "n_policy_paths"), 2)
        self.assertEqual(state.height, 2)

    def test_null_maturity_censor_is_retained_as_pending_not_dropped(self) -> None:
        action = pl.DataFrame([_action("entry-carry")])
        policy = pl.DataFrame(
            [
                _policy(
                    "entry-carry",
                    branch="carry_at_eod_cancel_unconfirmed",
                )
            ]
        )
        unresolved = {
            **_overnight("trial-entry-carry"),
            "outcome_status": "censored",
            "terminal_branch": None,
            "label_end_date": None,
            "filled_cashflow_before_cost_bp": None,
            "label_status": "expiry_settlement_unpriced",
            "unresolved_reason": "no_final_exact_contract_settlement_fact",
        }
        labels = build_exit_policy_label_tables(
            action,
            policy,
            terminal_config=self.terminal_config,
            overnight_labels=pl.DataFrame([unresolved]),
        )
        self.assertEqual(labels.strict.item(0, "outcome_status"), "censored")
        self.assertIsNone(labels.strict.item(0, "label_end_date"))
        lookup = build_daily_prequential_lookup(
            labels.strict,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=["20260522"],
        ).row(0, named=True)
        self.assertEqual(lookup["n_policy_paths"], 1)
        self.assertEqual(lookup["n_labels_matured"], 0)
        self.assertEqual(lookup["n_labels_pending"], 1)
        self.assertEqual(lookup["n_outcome_censored"], 0)
        self.assertEqual(lookup["n_unpriced_terminal_mass"], 1)
        self.assertEqual(
            lookup["known_gross_zero_for_unpriced_sensitivity_bp"], 0.0
        )
        self.assertIsNone(lookup["gross_mean_identification_lower_bp"])

    def test_non_19bp_cost_uses_generic_column_names(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy = pl.DataFrame(
            [_policy("entry-1", branch="flat_same_day", gross_twd=2_000.0)]
        )
        labels = build_exit_policy_label_tables(
            action,
            policy,
            terminal_config=self.terminal_config,
            assumed_non_price_cycle_cost_bp=7.0,
        )
        lookup = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            ExitPolicyLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
                assumed_non_price_cycle_cost_bp=7.0,
            ),
            asof_dates=["20260522"],
        ).row(0, named=True)
        gross = 2_000.0 / 230_000.0 * 10_000.0
        self.assertAlmostEqual(
            lookup[
                "conditional_known_net_after_assumed_cost_mean_bp"
            ],
            gross - 7.0,
        )
        self.assertAlmostEqual(
            lookup[
                "known_net_after_assumed_cost_zero_for_unpriced_sensitivity_bp"
            ],
            gross - 7.0,
        )
        self.assertNotIn("known_net19_zero_for_unpriced_sensitivity_bp", lookup)
        with self.assertRaisesRegex(ValueError, "assumed cost does not match"):
            build_daily_prequential_lookup(
                labels.nominal_v0,
                [DATE, "20260522"],
                ExitPolicyLookupConfig(
                    lookback_sessions=1,
                    min_history_sessions=1,
                    min_group_sessions=1,
                    min_known_paths=1,
                    assumed_non_price_cycle_cost_bp=8.0,
                ),
                asof_dates=["20260522"],
            )

    def test_target_day_and_immature_labels_never_enter(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy = pl.DataFrame(
            [_policy("entry-1", branch="flat_same_day", gross_twd=1_000.0)]
        )
        labels = build_exit_policy_label_tables(
            action,
            policy,
            terminal_config=self.terminal_config,
        )
        same_day = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=[DATE],
        )
        next_day = build_daily_prequential_lookup(
            labels.nominal_v0,
            [DATE, "20260522"],
            self.lookup_config,
            asof_dates=["20260522"],
        )
        self.assertTrue(same_day.is_empty())
        self.assertEqual(next_day.item(0, "label_cutoff_date"), DATE)
        self.assertFalse(next_day.item(0, "contains_target_day_outcome"))


if __name__ == "__main__":
    unittest.main()
