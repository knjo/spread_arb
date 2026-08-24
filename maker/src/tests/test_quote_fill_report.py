from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.report import build_product_action_research_table


class ProductActionReportTest(unittest.TestCase):
    def test_geometry_fill_hedge_and_latent_denominators(self) -> None:
        aliases = pl.DataFrame(
            {
                "Date": ["20260128", "20260128"],
                "ValueCode": ["2317", "2317"],
                "QuoteCode": ["DHFB6", "DHFB6"],
                "route": ["spot_bid_future_taker"] * 2,
                "boundary_quantile": [50, 50],
                "raw_order_fact_id": ["raw-a", "raw-b"],
                "policy_generation_id": ["a", "b"],
                "threshold_basis_bp": [60.0, 60.0],
                "effective_basis_bp": [62.0, 64.0],
                "full_fill": [True, False],
                "partial_fill": [False, True],
                "cancel_required": [False, True],
            }
        )
        snapshot = pl.DataFrame(
            {
                "Date": ["20260128"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFB6"],
                "boundary_quantile": [50],
                "upper_distance_bp": [10.0],
                "lower_distance_bp": [8.0],
                "adaptive_parameter_valid": [True],
                "contains_target_day_outcome": [False],
            }
        )
        hedge = pl.DataFrame(
            {
                "raw_order_fact_id": ["raw-a"],
                "status": ["executable"],
                "signed_total_slippage_bp": [3.0],
                "decision_book_age_ms": [50.0],
                "depth_shortfall": [0],
            }
        )
        labels = pl.DataFrame(
            {
                "ValueCode": ["2317", "2317"],
                "route": ["spot_bid_future_taker"] * 2,
                "boundary_quantile": [50, 50],
                "policy_generation_id": ["a", "a"],
                "target_id": ["frozen_center", "frozen_adaptive_lower"],
                "observation_delay_seconds": [30, 30],
                "status": ["hit", "session_no_hit"],
                "time_to_latent_hit_seconds": [40.0, None],
            }
        )
        row = build_product_action_research_table(
            aliases, snapshot, hedge, labels
        ).row(0, named=True)
        self.assertEqual(row["orders"], 2)
        self.assertEqual(row["full_fills"], 1)
        self.assertEqual(row["partial_fills"], 1)
        self.assertEqual(row["p_full_fill"], 0.5)
        # Anchor=threshold-upper=50; effective opens are 12 and 14 bp.
        self.assertEqual(row["rounded_effective_open_bp_p50"], 13.0)
        self.assertEqual(row["nominal_latent_band_bp_p50"], 21.0)
        self.assertEqual(row["entry_hedge_total_slippage_bp_p95"], 3.0)
        self.assertEqual(row["p_frozen_center_given_full_fill"], 1.0)
        self.assertEqual(row["p_frozen_lower_given_full_fill"], 0.0)
        self.assertFalse(row["ev_ready"])

    def test_rejects_target_day_snapshot(self) -> None:
        aliases = pl.DataFrame(
            {
                "Date": ["20260128"], "ValueCode": ["2317"],
                "QuoteCode": ["DHFB6"], "route": ["spot_bid_future_taker"],
                "boundary_quantile": [50], "raw_order_fact_id": ["raw-a"],
                "policy_generation_id": ["a"], "threshold_basis_bp": [60.0],
                "effective_basis_bp": [62.0], "full_fill": [False],
                "partial_fill": [False], "cancel_required": [True],
            }
        )
        snapshot = pl.DataFrame(
            {
                "Date": ["20260128"], "ValueCode": ["2317"],
                "QuoteCode": ["DHFB6"], "boundary_quantile": [50],
                "upper_distance_bp": [10.0], "lower_distance_bp": [8.0],
                "adaptive_parameter_valid": [True],
                "contains_target_day_outcome": [True],
            }
        )
        hedge = pl.DataFrame(
            schema={
                "raw_order_fact_id": pl.String, "status": pl.String,
                "signed_total_slippage_bp": pl.Float64,
                "decision_book_age_ms": pl.Float64, "depth_shortfall": pl.Int64,
            }
        )
        labels = pl.DataFrame(
            schema={
                "ValueCode": pl.String, "route": pl.String,
                "boundary_quantile": pl.Int64, "policy_generation_id": pl.String,
                "target_id": pl.String, "observation_delay_seconds": pl.Int64,
                "status": pl.String, "time_to_latent_hit_seconds": pl.Float64,
            }
        )
        with self.assertRaisesRegex(ValueError, "target-day outcomes"):
            build_product_action_research_table(aliases, snapshot, hedge, labels)


if __name__ == "__main__":
    unittest.main()
