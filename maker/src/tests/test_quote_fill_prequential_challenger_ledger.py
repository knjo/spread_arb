from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.post_cross_position_evaluator import PositionLimit
from maker.src.quote_fill.prequential_challenger_ledger import (
    ChallengerLedgerConfig,
    build_prequential_challenger_ledger,
)


D1 = "20260102"
D2 = "20260105"
D3 = "20260106"
D4 = "20260107"


def _decisions() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "asof_date": D1,
                "selected_action_ev_ready": False,
                "decision_status": "NO_GO_NO_EV_READY_ACTION",
                "diagnostic_challenger_boundary_quantile": None,
                "diagnostic_challenger_entry_route": None,
                "diagnostic_challenger_exit_rule_id": None,
                "diagnostic_challenger_exit_route": None,
                "diagnostic_challenger_completed_only_after_cost_mean_bp": None,
                "diagnostic_challenger_is_selected_action": False,
                "ranking_uses_only_Date_less_than_asof": True,
                "label_visible_only_if_label_availability_date_less_than_asof": True,
                "nominal_cancel_model_assumption": True,
                "production_strategy_go": False,
            },
            {
                "asof_date": D2,
                "selected_action_ev_ready": False,
                "decision_status": "NO_GO_NO_EV_READY_ACTION",
                "diagnostic_challenger_boundary_quantile": 95,
                "diagnostic_challenger_entry_route": "spot_bid_future_taker",
                "diagnostic_challenger_exit_rule_id": "frozen_lower",
                "diagnostic_challenger_exit_route": "future_bid_spot_taker",
                "diagnostic_challenger_completed_only_after_cost_mean_bp": 12.5,
                "diagnostic_challenger_is_selected_action": False,
                "ranking_uses_only_Date_less_than_asof": True,
                "label_visible_only_if_label_availability_date_less_than_asof": True,
                "nominal_cancel_model_assumption": True,
                "production_strategy_go": False,
            },
        ],
        infer_schema_length=None,
    )


def _action(
    generation: str,
    rank: str,
    *,
    quantile: int = 95,
    full: bool = False,
    partial: bool = False,
    hedge: bool = False,
    submit: int = 10,
) -> dict[str, object]:
    any_fill = full or partial
    return {
        "Date": D2,
        "ValueCode": "2330",
        "QuoteCode": "CDF1",
        "route": "spot_bid_future_taker",
        "maker_market": "spot",
        "maker_side": "bid",
        "boundary_quantile": quantile,
        "raw_order_fact_id": f"raw-{generation}",
        "policy_generation_id": generation,
        "target_rank_at_submit": rank,
        "intended_quantity": 1_000,
        "submit_recv_time_ns": submit,
        "nominal_stop_recv_time_ns": 100,
        "nominal_stop_reason": "target_retreat",
        "known_filled_quantity": 1_000 if full else 500 if partial else 0,
        "any_fill": any_fill,
        "full_fill": full,
        "partial_fill": partial,
        "full_fill_recv_time_ns": 50 if full else None,
        "terminal_recv_time_ns": 50 if full else 100,
        "terminal_reason": "full_fill" if full else "target_retreat",
        "cancel_required": not full,
        "entry_hedge_label_observed": full,
        "entry_hedge_executable": hedge,
        "entry_hedge_signed_total_slippage_bp": 2.0 if hedge else None,
    }


def _actions() -> pl.DataFrame:
    return pl.from_dicts(
        [
            _action("p-complete", "BID1", full=True, hedge=True, submit=10),
            _action("p-unknown", "BID1", full=True, hedge=True, submit=20),
            _action("p-partial", "BID2", partial=True, submit=30),
            _action("p-no-fill", "BID2", submit=40),
            _action("p-deep", "BID3", full=True, hedge=True, submit=50),
            _action(
                "p-other-q", "BID1", quantile=50, full=True, hedge=True, submit=60
            ),
        ],
        infer_schema_length=None,
    )


def _path(
    generation: str,
    identifier: str,
    *,
    completed: bool,
    rule: str = "frozen_lower",
    quantile: int = 95,
) -> dict[str, object]:
    return {
        "Date": D2,
        "ValueCode": "2330",
        "QuoteCode": "CDF1",
        "entry_route": "spot_bid_future_taker",
        "boundary_quantile": quantile,
        "entry_policy_generation_id": generation,
        "entry_raw_order_fact_id": f"raw-{generation}",
        "position_established_ns": 60 if generation == "p-complete" else 70,
        "exit_rule_id": rule,
        "exit_route": "future_bid_spot_taker",
        "filled_entry_outcome_category": "completed" if completed else "unknown",
        "terminal_date": D3 if completed else None,
        "gross_cycle_pnl_twd": 100.0 if completed else None,
        "gross_cycle_bp": 50.0 if completed else None,
        "normalization_notional_twd": 20_000.0,
        "physical_entry_dependency_id": f"physical-{identifier}",
        "policy_path_id": identifier,
        "label_availability_date": D3,
        "outstanding_interval_end_exclusive": D3 if completed else D4,
        "completed_same_day": False,
        "completed_overnight": completed,
        "terminal_cashflow_priced": completed,
        "exit_decision_time_ns": 200 if completed else None,
        "nominal_cancel_model_assumption": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
    }


def _paths() -> pl.DataFrame:
    return pl.from_dicts(
        [
            _path("p-complete", "path-complete", completed=True),
            _path("p-unknown", "path-unknown", completed=False),
            _path("p-deep", "path-deep", completed=True),
            _path(
                "p-complete",
                "path-other-rule",
                completed=True,
                rule="frozen_center",
            ),
            _path(
                "p-other-q",
                "path-other-q",
                completed=True,
                quantile=50,
            ),
        ],
        infer_schema_length=None,
    )


class PrequentialChallengerLedgerTest(unittest.TestCase):
    def test_ab12_target_day_join_and_unresolved_are_separate(self) -> None:
        config = ChallengerLedgerConfig(
            same_day_cost_bp=19.0,
            overnight_cost_bp=34.0,
            position_limits=(
                PositionLimit("positions_1", max_concurrent_positions=1),
            ),
        )
        result = build_prequential_challenger_ledger(
            _decisions(),
            _actions(),
            _paths(),
            [D1, D2, D3, D4],
            config=config,
        )

        self.assertEqual(result.selected_actions.height, 4)
        self.assertEqual(
            set(result.selected_actions["target_rank_at_submit"].to_list()),
            {"BID1", "BID2"},
        )
        self.assertEqual(result.selected_paths.height, 2)
        self.assertEqual(
            set(result.selected_paths["policy_path_id"].to_list()),
            {"path-complete", "path-unknown"},
        )
        unknown = result.selected_paths.filter(
            pl.col("policy_path_id") == "path-unknown"
        )
        self.assertIsNone(unknown.item(0, "diagnostic_after_cost_bp"))
        self.assertIsNone(unknown.item(0, "diagnostic_after_cost_twd"))

        day = result.daily_cohort_ledger.filter(pl.col("Date") == D2).row(
            0, named=True
        )
        self.assertEqual(day["supported_ab12_unique_physical_quotes"], 4)
        self.assertEqual(day["full_fills"], 2)
        self.assertEqual(day["partial_fills"], 1)
        self.assertEqual(day["no_fills"], 1)
        self.assertEqual(day["established_positions"], 2)
        self.assertEqual(day["completed_cycles"], 1)
        self.assertEqual(day["unknown_or_open_positions"], 1)
        self.assertAlmostEqual(day["completed_after_19_34_mean_bp"], 16.0)
        self.assertAlmostEqual(
            day["zero_unresolved_per_submitted_quote_after_19_34_bp"], 4.0
        )
        self.assertFalse(day["terminal_cashflow_point_identified"])

        outstanding = {
            row["Date"]: row["outstanding_eod_positions"]
            for row in result.daily_outstanding.iter_rows(named=True)
        }
        self.assertEqual(outstanding[D2], 2)
        self.assertEqual(outstanding[D3], 1)
        self.assertNotIn(D4, outstanding)
        realized = result.daily_realized_cashflows.filter(
            pl.col("terminal_date") == D3
        ).row(0, named=True)
        self.assertEqual(realized["gross_realized_twd"], 100.0)
        self.assertEqual(realized["completed_after_19_34_realized_twd"], 32.0)

        limit = result.position_limit_sweep.row(0, named=True)
        self.assertEqual(limit["candidate_positions"], 2)
        self.assertEqual(limit["accepted_positions"], 1)
        self.assertEqual(limit["rejected_positions"], 1)
        self.assertFalse(limit["position_limit_sweep_production_ready"])

    def test_fail_closed_when_decision_claims_selection(self) -> None:
        decisions = _decisions().with_columns(
            pl.when(pl.col("asof_date") == D2)
            .then(pl.lit(True))
            .otherwise(pl.col("diagnostic_challenger_is_selected_action"))
            .alias("diagnostic_challenger_is_selected_action")
        )
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            build_prequential_challenger_ledger(
                decisions,
                _actions(),
                _paths(),
                [D1, D2, D3, D4],
                config=ChallengerLedgerConfig(
                    position_limits=(
                        PositionLimit("positions_1", max_concurrent_positions=1),
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
