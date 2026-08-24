"""Tests for the per-submitted-entry-quote nominal-V0 lookup."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.submitted_entry_lookup import (
    EXIT_ROUTES,
    SubmittedEntryLookupConfig,
    build_daily_submitted_entry_lookup,
    build_submitted_entry_policy_labels,
)


DATE = "20260521"


def _action(policy: str, outcome: str, *, q: str) -> dict[str, object]:
    full = outcome == "full_fill_hedge_executable"
    no_fill = outcome == "no_fill_then_cancel"
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "route": "future_ask_spot_taker",
        "policy_generation_id": policy,
        "raw_order_fact_id": f"raw-{policy}",
        "lookup_action_id": q,
        "boundary_quantile": int(q[1:]),
        "source_asof_date": "20260520",
        "parameter_version": "entry-v1",
        "rank_bucket": "at_bbo",
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
        "entry_execution_outcome": outcome,
        "queue_known": not outcome.startswith("unknown"),
        "any_fill": True if full else (False if no_fill else None),
        "full_fill": True if full else (False if no_fill else None),
        "partial_fill": False,
        "entry_hedge_status": "executable" if full else None,
    }


def _rule(policy: str) -> dict[str, object]:
    return {
        "Date": DATE,
        "policy_generation_id": policy,
        "exit_rule_id": "frozen_center",
        "exit_rule_source_asof_date": "20260520",
        "exit_threshold_basis_bp": 100.0,
    }


def _conditional(route: str, *, branch: str, gross: float) -> dict[str, object]:
    label_end = "20260521" if branch == "same_day_target_exit" else "20260523"
    return {
        "entry_policy_generation_id": "full",
        "exit_rule_id": "frozen_center",
        "exit_route": route,
        "exit_policy_trial_id": f"conditional-{route}",
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "entry_route": "future_ask_spot_taker",
        "label_end_date": label_end,
        "outcome_status": "known",
        "terminal_branch": branch,
        "actual_four_leg_gross_bp": gross,
        "source_exit_branch_status": (
            "flat_same_day" if branch == "same_day_target_exit" else "carry_at_eod"
        ),
        "strict_cancel_ambiguity_overridden": False,
        "any_unacked_cancel_count": 0,
        "overnight_label_attached": branch == "overnight_exit",
        "terminal_label_source": branch,
        "unresolved_reason": None,
    }


class SubmittedEntryLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SubmittedEntryLookupConfig(
            lookback_sessions=3,
            min_history_sessions=1,
            min_group_sessions=1,
            min_known_paths=1,
            assumed_flat_completed_cycle_cost_bp=19.0,
            assumed_same_day_completed_cycle_cost_bp=19.0,
            assumed_overnight_completed_cycle_cost_bp=34.0,
        )
        self.actions = pl.DataFrame(
            [
                _action("full", "full_fill_hedge_executable", q="q50"),
                _action("no-fill", "no_fill_then_cancel", q="q80"),
                _action("unknown", "unknown_queue_then_cancel", q="q95"),
            ]
        )
        self.rules = pl.DataFrame([_rule("full"), _rule("unknown")])
        self.conditional = pl.DataFrame(
            [
                _conditional(
                    EXIT_ROUTES[0], branch="same_day_target_exit", gross=50.0
                ),
                _conditional(EXIT_ROUTES[1], branch="overnight_exit", gross=40.0),
            ]
        )

    def test_natural_denominator_and_unknowns_are_retained(self) -> None:
        labels = build_submitted_entry_policy_labels(
            self.actions, self.rules, self.conditional, self.config
        )
        self.assertEqual(labels.height, 5)
        counts = {
            row["entry_outcome_category"]: row["len"]
            for row in labels.group_by("entry_outcome_category").len().iter_rows(
                named=True
            )
        }
        self.assertEqual(
            counts,
            {"full_hedged": 2, "known_no_fill_v0": 1, "entry_fill_unknown": 2},
        )
        no_fill = labels.filter(
            pl.col("entry_outcome_category") == "known_no_fill_v0"
        ).row(0, named=True)
        self.assertFalse(no_fill["exit_policy_assigned"])
        self.assertEqual(no_fill["outcome_status"], "known")
        self.assertEqual(no_fill["actual_four_leg_gross_bp"], 0.0)
        self.assertEqual(no_fill["completed_cycle_cost_applied_count"], 0)
        self.assertEqual(
            no_fill["cashflow_after_flat_completed_cycle_cost_sensitivity_bp"],
            0.0,
        )
        self.assertTrue(no_fill["no_fill_cancel_cost_missing"])
        unknown = labels.filter(
            pl.col("entry_outcome_category") == "entry_fill_unknown"
        )
        self.assertTrue(unknown["actual_four_leg_gross_bp"].is_null().all())

    def test_branch_cost_charges_overnight_extra_once(self) -> None:
        labels = build_submitted_entry_policy_labels(
            self.actions, self.rules, self.conditional, self.config
        )
        same_day = labels.filter(
            pl.col("terminal_branch") == "same_day_target_exit"
        ).row(0, named=True)
        overnight = labels.filter(
            pl.col("terminal_branch") == "overnight_exit"
        ).row(0, named=True)
        self.assertEqual(
            same_day["cashflow_after_flat_completed_cycle_cost_sensitivity_bp"],
            31.0,
        )
        self.assertEqual(
            same_day["cashflow_after_branch_completed_cycle_cost_sensitivity_bp"],
            31.0,
        )
        self.assertEqual(
            overnight["cashflow_after_flat_completed_cycle_cost_sensitivity_bp"],
            21.0,
        )
        self.assertEqual(
            overnight[
                "cashflow_after_branch_completed_cycle_cost_sensitivity_bp"
            ],
            6.0,
        )
        self.assertEqual(overnight["completed_cycle_cost_applied_count"], 1)

    def test_lookup_is_d_safe_and_never_calls_sensitivity_ev(self) -> None:
        labels = build_submitted_entry_policy_labels(
            self.actions, self.rules, self.conditional, self.config
        )
        sessions = [DATE, "20260522", "20260523", "20260524"]
        before_maturity = build_daily_submitted_entry_lookup(
            labels,
            sessions,
            self.config,
            asof_dates=["20260522"],
        )
        overnight_before = before_maturity.filter(
            pl.col("exit_route") == EXIT_ROUTES[1]
        ).row(0, named=True)
        self.assertEqual(overnight_before["n_labels_pending"], 1)
        self.assertEqual(overnight_before["n_outcome_known"], 0)
        after_maturity = build_daily_submitted_entry_lookup(
            labels,
            sessions,
            self.config,
            asof_dates=["20260524"],
        )
        overnight_after = after_maturity.filter(
            (pl.col("entry_q") == "q50")
            & (pl.col("exit_route") == EXIT_ROUTES[1])
        ).row(0, named=True)
        self.assertEqual(overnight_after["n_labels_pending"], 0)
        self.assertEqual(overnight_after["n_outcome_known"], 1)
        self.assertEqual(
            overnight_after[
                "conditional_known_cashflow_after_branch_completed_cycle_cost_mean_bp"
            ],
            6.0,
        )
        self.assertFalse(after_maturity["ev_ready"].any())
        self.assertTrue(after_maturity["expected_net_cashflow_bp"].is_null().all())
        self.assertTrue(
            after_maturity["known_gross_zero_for_unpriced_sensitivity_bp"]
            .is_not_null()
            .any()
        )

    def test_cost_config_validation_rejects_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "overnight"):
            SubmittedEntryLookupConfig(
                assumed_overnight_completed_cycle_cost_bp=float("nan")
            ).validate()


if __name__ == "__main__":
    unittest.main()
