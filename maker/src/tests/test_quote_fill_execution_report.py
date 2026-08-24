from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from maker.src.quote_fill.execution_report import (
    build_narrow_execution_report,
    load_boundary_snapshot_for_report,
)

ROUTE = "future_ask_spot_taker"


def _inputs() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    # raw-a implements q50 and q80 at the same legal price.  q50 also has two
    # policy aliases for raw-a, which must still be one physical observation.
    action = pl.DataFrame(
        {
            "Date": ["20260102", "20260102", "20260102", "20260102", "20260103"],
            "ValueCode": ["2317"] * 5,
            "QuoteCode": ["DHFA6"] * 5,
            "route": [ROUTE] * 5,
            "boundary_quantile": [50, 50, 50, 80, 50],
            "lookup_action_id": ["q50", "q50", "q50", "q80", "q50"],
            "raw_order_fact_id": ["raw-a", "raw-a", "raw-b", "raw-a", "raw-c"],
            "policy_generation_id": ["p-a1", "p-a2", "p-b", "p-a80", "p-c"],
            "any_fill": [True, True, False, True, False],
            "full_fill": [True, True, False, True, False],
            "partial_fill": [False] * 5,
            "cancel_required": [False, False, True, False, True],
        }
    )
    raw = pl.DataFrame(
        {
            "Date": ["20260102", "20260102", "20260103"],
            "ValueCode": ["2317"] * 3,
            "QuoteCode": ["DHFA6"] * 3,
            "route": [ROUTE] * 3,
            "raw_order_fact_id": ["raw-a", "raw-b", "raw-c"],
        }
    )
    hedge = pl.DataFrame(
        {
            "raw_order_fact_id": ["raw-a"],
            "status": ["executable"],
            "decision_book_age_ms": [50.0],
            "signed_total_slippage_bp": [5.0],
            "depth_shortfall": [0],
        }
    )
    support = pl.DataFrame(
        {
            "Date": [
                "20260102",
                "20260102",
                "20260102",
                "20260103",
                "20260103",
                "20260103",
            ],
            "ValueCode": ["2317"] * 6,
            "QuoteCode": ["DHFA6"] * 6,
            "route": [ROUTE] * 6,
            "boundary_quantile": [50, 80, 95, 50, 80, 95],
            "lookup_action_id": ["q50", "q80", "q95", "q50", "q80", "q95"],
            "target_observations": [10, 10, 10, 8, 8, 8],
            "submitted_generations": [3, 1, 0, 1, 0, 0],
            "unique_raw_order_facts": [2, 1, 0, 1, 0, 0],
        }
    )
    exit_facts = pl.DataFrame(
        {
            "Date": ["20260102", "20260102", "20260102", "20260102", "20260103"],
            "ValueCode": ["2317"] * 5,
            "QuoteCode": ["DHFA6"] * 5,
            "route": [ROUTE] * 5,
            "raw_order_fact_id": ["raw-a", "raw-a", "raw-b", "raw-a", "raw-c"],
            "policy_generation_id": ["p-a1", "p-a2", "p-b", "p-a80", "p-c"],
            "exit_rule_id": ["center"] * 5,
            "branch_status": [
                "same_day_taker_exit",
                "same_day_taker_exit",
                "no_entry_fill",
                "same_day_taker_exit",
                "no_entry_fill",
            ],
            "same_day_exit": [True, True, False, True, False],
            "overnight_carry": [False] * 5,
            "gross_cycle_pnl_twd": [10.0, 10.0, None, 10.0, None],
            "eod_liquidation_gross_pnl_twd": [None] * 5,
        }
    )
    return action, raw, hedge, support, exit_facts


def _boundary_snapshot() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260102"] * 3 + ["20260103"] * 3,
            "ValueCode": ["2317"] * 6,
            "QuoteCode": ["DHFA6"] * 6,
            "boundary_quantile": [50, 80, 95, 50, 80, 95],
            "boundary_role": [
                "rolling_latent_candidate",
                "rolling_latent_candidate",
                "tail_diagnostic",
            ]
            * 2,
            "upper_distance_bp": [10.0, 20.0, 30.0, 10.0, 24.0, 34.0],
            "lower_distance_bp": [8.0, 16.0, 24.0, 8.0, 18.0, 26.0],
            "adaptive_parameter_valid": [True, True, True, True, False, True],
            "fut_ref_price": [101.0] * 6,
            "spot_ref_price": [100.0] * 6,
            "contract_size": [2000.0] * 6,
            "target_ref_future_ask_tick_bp": [5.0] * 6,
            "target_ref_future_tick_bp": [5.0] * 6,
            "target_ref_spot_bid_tick_bp": [10.0] * 6,
            "source_asof_date": ["20260101"] * 3 + ["20260102"] * 3,
            "parameter_version": ["rolling-test-v1"] * 6,
            "price_ladder_version": [PRICE_LADDER_VERSION] * 6,
            "future_one_dollar_tick_effective_date": [
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            ] * 6,
            "execution_safe_snapshot": [True] * 6,
            "contains_target_day_outcome": [False] * 6,
        }
    )


class NarrowExecutionReportTest(unittest.TestCase):
    def test_report_loader_rejects_legacy_unmarked_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rolling_boundary_snapshots.parquet"
            _boundary_snapshot().write_parquet(path)
            daily_support = _inputs()[3]
            with self.assertRaisesRegex(
                FileNotFoundError,
                "publication is incomplete",
            ):
                load_boundary_snapshot_for_report(path, daily_support)

    def test_aliases_do_not_inflate_physical_entry_or_hedge_rates(self) -> None:
        report = build_narrow_execution_report(*_inputs())
        row = report.product_entry.filter(
            (pl.col("ValueCode") == "2317")
            & (pl.col("route") == ROUTE)
            & (pl.col("boundary_quantile") == 50)
        ).row(0, named=True)

        self.assertEqual(row["submitted_policy_aliases"], 4)
        self.assertEqual(row["physical_raw_order_facts"], 3)
        self.assertEqual(row["cross_q_shared_raw_order_facts"], 1)
        self.assertEqual(row["full_fill_raw_orders"], 1)
        self.assertEqual(row["cancel_required_raw_orders"], 2)
        self.assertAlmostEqual(row["pooled_full_fill_rate_physical"], 1 / 3)
        self.assertAlmostEqual(row["daily_balanced_full_fill_rate"], 0.25)
        self.assertEqual(row["daily_fill_rate_support_product_days"], 2)
        self.assertEqual(row["hedge_executable_full_fills"], 1)
        self.assertEqual(row["hedge_book_fresh_le100ms_full_fills"], 1)
        self.assertEqual(row["hedge_total_slippage_bp_p95"], 5.0)
        self.assertFalse(row["cross_q_rows_safe_to_sum"])

    def test_zero_submission_product_day_is_retained_with_null_rate(self) -> None:
        report = build_narrow_execution_report(*_inputs())
        row = report.daily_entry.filter(
            (pl.col("Date") == "20260103") & (pl.col("boundary_quantile") == 80)
        ).row(0, named=True)
        self.assertEqual(row["physical_raw_order_facts"], 0)
        self.assertEqual(row["submitted_policy_aliases"], 0)
        self.assertIsNone(row["full_fill_rate_physical"])

        product = report.product_entry.filter(pl.col("boundary_quantile") == 80).row(
            0, named=True
        )
        self.assertEqual(product["product_days_in_grid"], 2)
        self.assertEqual(product["product_days_with_submissions"], 1)
        self.assertEqual(product["daily_fill_rate_support_product_days"], 1)

    def test_exit_branches_are_deduplicated_by_raw_order_inside_q(self) -> None:
        report = build_narrow_execution_report(*_inputs())
        row = report.product_exit.filter(
            (pl.col("boundary_quantile") == 50) & (pl.col("exit_rule_id") == "center")
        ).row(0, named=True)
        self.assertEqual(row["exit_policy_alias_facts"], 4)
        self.assertEqual(row["exit_rule_physical_raw_facts"], 3)
        self.assertEqual(row["branch_same_day_taker_exit_raw_orders"], 1)
        self.assertEqual(row["branch_no_entry_fill_raw_orders"], 2)
        self.assertAlmostEqual(
            row["pooled_same_day_exit_rate_per_physical_entry"], 1 / 3
        )
        self.assertEqual(row["pooled_same_day_exit_rate_given_opened_and_hedged"], 1.0)
        self.assertFalse(row["pathwise_ev_ready"])

    def test_conflicting_same_q_alias_outcomes_fail_closed(self) -> None:
        action, raw, hedge, support, exit_facts = _inputs()
        action = action.with_columns(
            pl.when(pl.col("policy_generation_id") == "p-a2")
            .then(False)
            .otherwise(pl.col("full_fill"))
            .alias("full_fill")
        )
        with self.assertRaisesRegex(ValueError, "same-q aliases disagree"):
            build_narrow_execution_report(action, raw, hedge, support, exit_facts)

    def test_missing_zero_order_support_cell_fails_closed(self) -> None:
        action, raw, hedge, support, exit_facts = _inputs()
        support = support.filter(
            ~((pl.col("Date") == "20260103") & (pl.col("boundary_quantile") == 80))
        )
        with self.assertRaisesRegex(ValueError, "omits zero-order"):
            build_narrow_execution_report(action, raw, hedge, support, exit_facts)

    def test_d1_geometry_and_entry_join_preserve_q_role_and_tick_scale(self) -> None:
        report = build_narrow_execution_report(
            *_inputs(), boundary_snapshot=_boundary_snapshot()
        )
        future_q50 = report.product_geometry.filter(
            (pl.col("route") == ROUTE) & (pl.col("boundary_quantile") == 50)
        ).row(0, named=True)
        self.assertEqual(future_q50["boundary_role"], "rolling_latent_candidate")
        self.assertEqual(future_q50["geometry_product_days_in_grid"], 2)
        self.assertEqual(future_q50["geometry_valid_product_days"], 2)
        self.assertEqual(future_q50["upper_distance_bp_p50"], 10.0)
        self.assertEqual(future_q50["lower_distance_bp_p95"], 8.0)
        self.assertEqual(future_q50["maker_route_tick_bp_p50"], 5.0)
        self.assertEqual(future_q50["upper_distance_maker_ticks_p50"], 2.0)
        self.assertEqual(future_q50["future_to_spot_tick_bp_ratio_p50"], 0.5)

        q80 = report.product_geometry.filter(pl.col("boundary_quantile") == 80).row(
            0, named=True
        )
        self.assertEqual(q80["geometry_valid_product_days"], 1)
        self.assertEqual(q80["geometry_valid_product_day_rate"], 0.5)
        q95 = report.lookup_checkpoint.filter(pl.col("boundary_quantile") == 95).row(
            0, named=True
        )
        self.assertEqual(q95["boundary_role"], "tail_diagnostic")
        self.assertTrue(q95["tail_diagnostic_only"])
        self.assertEqual(q95["submitted_policy_aliases"], 0)
        self.assertFalse(q95["lookup_checkpoint_is_final_ev"])

    def test_geometry_fails_closed_on_safety_source_contract_and_q95_role(self) -> None:
        action, raw, hedge, support, exit_facts = _inputs()
        mutations = (
            (
                "execution-safe",
                _boundary_snapshot().with_columns(
                    pl.when(
                        (pl.col("Date") == "20260102")
                        & (pl.col("boundary_quantile") == 50)
                    )
                    .then(True)
                    .otherwise(pl.col("contains_target_day_outcome"))
                    .alias("contains_target_day_outcome")
                ),
            ),
            (
                "strictly before",
                _boundary_snapshot().with_columns(
                    pl.when(pl.col("Date") == "20260103")
                    .then(pl.lit("20260103"))
                    .otherwise(pl.col("source_asof_date"))
                    .alias("source_asof_date")
                ),
            ),
            (
                "exact execution contract",
                _boundary_snapshot().with_columns(pl.lit("WRONG").alias("QuoteCode")),
            ),
            (
                "price ladder lineage",
                _boundary_snapshot().with_columns(
                    pl.lit("legacy-v1").alias("price_ladder_version")
                ),
            ),
            (
                "q95 must retain role",
                _boundary_snapshot().with_columns(
                    pl.when(pl.col("boundary_quantile") == 95)
                    .then(pl.lit("rolling_latent_candidate"))
                    .otherwise(pl.col("boundary_role"))
                    .alias("boundary_role")
                ),
            ),
        )
        for message, boundary in mutations:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                build_narrow_execution_report(
                    action,
                    raw,
                    hedge,
                    support,
                    exit_facts,
                    boundary,
                )


if __name__ == "__main__":
    unittest.main()
