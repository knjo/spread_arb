"""Tests for the execution-observation to terminal-path adapter."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.ev_surface import EVLookupConfig, build_daily_ev_lookup
from maker.src.quote_fill.terminal_paths import (
    EXECUTION_PATH_LOOKUP_KEYS_V1,
    TerminalPathAdapterConfig,
    build_execution_terminal_paths_v1,
)


DATE = "20260101"


def _action(policy_id: str, kind: str) -> dict[str, object]:
    full = kind in {"same_day", "carry", "hedge_failed"}
    partial = kind == "partial"
    any_fill: bool | None = True if full or partial else False
    hedge_status: str | None = (
        "executable" if kind in {"same_day", "carry"} else None
    )
    if kind == "hedge_failed":
        hedge_status = "insufficient_depth"
    return {
        "Date": DATE,
        "ValueCode": "2317",
        "QuoteCode": "DHFA6",
        "route": "future_ask_spot_taker",
        "parameter_version": "boundary-v1",
        "lookup_action_id": "q80",
        "raw_order_fact_id": f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "rank_bucket": "at_bbo",
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
        "intended_quantity": 1,
        "submit_recv_time_ns": 1_000_000_000,
        "terminal_recv_time_ns": 2_000_000_000,
        "nominal_stop_reason": "target_retreat",
        "queue_known": True,
        "any_fill": any_fill,
        "full_fill": full,
        "partial_fill": partial,
        "cancel_required": not full,
        "entry_hedge_status": hedge_status,
        "entry_hedge_signed_total_slippage_bp": (
            1.25 if hedge_status == "executable" else None
        ),
        "entry_hedge_executed_quantity": 2 if hedge_status == "executable" else 0,
        "entry_hedge_contract_size_shares": 2_000,
        "entry_spot_price": 100.0,
        "joint_volume_allocated": False,
    }


def _exit(policy_id: str, kind: str) -> dict[str, object]:
    branches = {
        "no_fill": "no_entry_fill",
        "same_day": "same_day_taker_exit",
        "carry": "carry_at_eod",
        "partial": "partial_entry_unhedged",
        "hedge_failed": "entry_hedge_unpriceable",
    }
    branch = branches[kind]
    return {
        "Date": DATE,
        "ValueCode": "2317",
        "QuoteCode": "DHFA6",
        "route": "future_ask_spot_taker",
        "raw_order_fact_id": f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "exit_rule_id": "frozen_center",
        "exit_rule_source_asof_date": "20251231",
        "branch_status": branch,
        "terminal_outcome": branch in {"no_entry_fill", "same_day_taker_exit"},
        "needs_next_session_label": branch == "carry_at_eod",
        "exit_decision_time_ns": 5_000_000_000 if kind == "same_day" else None,
        "gross_cycle_pnl_twd": 200.0 if kind == "same_day" else None,
    }


class ExecutionTerminalPathAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TerminalPathAdapterConfig(
            lifecycle_policy_version="layered-v1",
            queue_scenario="observed_displayed_queue_v1",
        )

    def test_only_same_day_close_is_known_without_extra_models(self) -> None:
        kinds = ("no_fill", "same_day", "carry", "partial", "hedge_failed")
        actions = pl.from_dicts(
            [_action(f"policy-{kind}", kind) for kind in kinds],
            infer_schema_length=None,
        )
        exits = pl.from_dicts(
            [_exit(f"policy-{kind}", kind) for kind in kinds],
            infer_schema_length=None,
        )
        paths = build_execution_terminal_paths_v1(actions, exits, self.config)
        by_policy = {
            row["policy_generation_id"]: row
            for row in paths.iter_rows(named=True)
        }

        same_day = by_policy["policy-same_day"]
        self.assertEqual(same_day["outcome_status"], "known")
        self.assertEqual(
            same_day["terminal_branch"], "same_day_aggressive_exit"
        )
        self.assertAlmostEqual(same_day["filled_cashflow_before_cost_bp"], 10.0)
        self.assertEqual(same_day["capital_time_seconds"], 4.0)
        self.assertEqual(same_day["hedge_slippage_bp_50ms"], 1.25)
        self.assertIsNone(same_day["exit_slippage_bp"])
        self.assertIsNone(same_day["fee_cost_bp"])
        self.assertIsNone(same_day["cost_profile_version"])

        no_fill = by_policy["policy-no_fill"]
        self.assertEqual(no_fill["outcome_status"], "unknown")
        self.assertIsNone(no_fill["terminal_branch"])
        self.assertEqual(
            no_fill["terminal_mapping_reason"],
            "cancel_ack_and_cancel_race_unobserved",
        )
        carry = by_policy["policy-carry"]
        self.assertEqual(carry["outcome_status"], "censored")
        self.assertIsNone(carry["terminal_branch"])
        self.assertEqual(carry["overnight_status"], "carried_open")
        self.assertFalse(carry["overnight_terminal_label_complete"])
        self.assertEqual(
            by_policy["policy-partial"]["outcome_status"], "unknown"
        )
        self.assertEqual(
            by_policy["policy-hedge_failed"]["outcome_status"], "unknown"
        )

    def test_nominal_cancel_requires_an_explicit_version(self) -> None:
        actions = pl.DataFrame([_action("cancel", "no_fill")])
        exits = pl.DataFrame([_exit("cancel", "no_fill")])
        paths = build_execution_terminal_paths_v1(
            actions,
            exits,
            TerminalPathAdapterConfig(
                lifecycle_policy_version="layered-v1",
                queue_scenario="observed_displayed_queue_v1",
                nominal_cancel_model_version="instant_at_stop_v1",
            ),
        )
        row = paths.row(0, named=True)
        self.assertEqual(row["outcome_status"], "known")
        self.assertEqual(row["terminal_branch"], "no_fill_cancel")
        self.assertEqual(row["cancel_status"], "cancelled")
        self.assertEqual(row["filled_cashflow_before_cost_bp"], 0.0)
        self.assertEqual(row["capital_time_seconds"], 1.0)
        self.assertIn("instant_at_stop_v1", row["lifecycle_policy_version"])

    def test_adapter_lookup_is_diagnostic_and_never_ev_ready(self) -> None:
        actions = pl.DataFrame([_action("same-day", "same_day")])
        exits = pl.DataFrame([_exit("same-day", "same_day")])
        paths = build_execution_terminal_paths_v1(actions, exits, self.config)
        lookup = build_daily_ev_lookup(
            paths,
            [DATE, "20260102"],
            EVLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
                lookup_keys=EXECUTION_PATH_LOOKUP_KEYS_V1,
            ),
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["n_outcome_known"], 1)
        self.assertEqual(row["n_cost_complete_known"], 0)
        self.assertEqual(row["n_slippage_complete_known"], 0)
        self.assertIsNone(row["expected_net_cashflow_bp"])
        self.assertFalse(row["ev_ready"])
        self.assertEqual(row["ev_status"], "incomplete_cost_inputs")

    def test_identity_mismatch_and_missing_exit_policy_fail_closed(self) -> None:
        actions = pl.DataFrame([_action("same-day", "same_day")])
        exits = pl.DataFrame([_exit("same-day", "same_day")])
        mismatch = exits.with_columns(pl.lit("OTHER").alias("QuoteCode"))
        with self.assertRaisesRegex(ValueError, "disagree on QuoteCode"):
            build_execution_terminal_paths_v1(actions, mismatch, self.config)

        with self.assertRaisesRegex(ValueError, "both be present"):
            build_execution_terminal_paths_v1(actions, pl.DataFrame(), self.config)


if __name__ == "__main__":
    unittest.main()
