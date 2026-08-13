"""Synthetic tests for product-adaptive latent basis boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

import polars as pl

from ..quote_width.adaptive import (
    add_boundary_supply,
    build_adaptive_boundaries,
    build_adaptive_cycle_universe,
)
from ..quote_width.cycle import build_latent_cycles


def _parameter_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": "20260128",
        "ValueCode": "2303",
        "QuoteCode": "CCFB6",
        "prior_date": "20260127",
        "calendar_gap_days": 1,
        "target_dte_days": 20,
        "target_tick_bp_bucket": "02_22to30bp",
        "prior_future_spread_bucket": "1tick",
        "target_ref_future_ask_tick_bp": 5.0,
        "target_ref_spot_bid_tick_bp": 10.0,
        "prior_spot_spread_ticks_p50": 1.0,
        "prior_fut_spread_ticks_p50": 2.0,
        "prior_tt_band_width_bp_p50": 40.0,
        "prior_eligible_rate": 0.90,
        "prior_completed_positive": 40,
        "prior_completed_negative": 45,
        "prior_censored_positive": 2,
        "prior_censored_negative": 3,
        "prior_p50_amplitude_bp_positive": 10.0,
        "prior_p50_amplitude_bp_negative": 20.0,
        "prior_p80_amplitude_bp_positive": 15.0,
        "prior_p80_amplitude_bp_negative": 30.0,
        "prior_p95_amplitude_bp_positive": 25.0,
        "prior_p95_amplitude_bp_negative": 45.0,
    }
    row.update(overrides)
    return row


def _cycle_panel(
    basis: list[float],
    anchor: list[float] | None = None,
) -> pl.DataFrame:
    rows = len(basis)
    anchor = anchor or [0.0] * rows
    start = datetime(2026, 1, 28, 1, 5)
    return pl.DataFrame(
        {
            "Date": ["20260128"] * rows,
            "ValueCode": ["2303"] * rows,
            "QuoteCode": ["CCFB6"] * rows,
            "timestamp": [start + timedelta(seconds=value) for value in range(rows)],
            "seconds_from_open": list(range(300, 300 + rows)),
            "basis_mid_bp": basis,
            "anchor_ewma_120s_bp": anchor,
            "analysis_eligible": [True] * rows,
            "eligible_base": [True] * rows,
            "eligible_1000ms": [True] * rows,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))


class AdaptiveBoundaryTest(unittest.TestCase):
    def test_asymmetric_prior_bounds_keep_separate_tick_scales(self) -> None:
        result = build_adaptive_boundaries(
            pl.DataFrame([_parameter_row()]),
            quantiles=(50, 95),
        )

        p50 = result.filter(pl.col("boundary_quantile") == 50).row(0, named=True)
        self.assertEqual(p50["upper_distance_bp"], 10.0)
        self.assertEqual(p50["lower_distance_bp"], 20.0)
        self.assertEqual(p50["upper_distance_future_ticks"], 2.0)
        self.assertEqual(p50["lower_distance_future_ticks"], 4.0)
        self.assertEqual(p50["upper_distance_spot_ticks"], 1.0)
        self.assertEqual(p50["lower_distance_spot_ticks"], 2.0)
        self.assertEqual(p50["boundary_role"], "latent_prior_candidate")
        self.assertTrue(p50["adaptive_parameter_valid"])
        self.assertFalse(p50["actionable_execution"])
        self.assertTrue(p50["execution_safe_snapshot"])
        self.assertFalse(p50["contains_target_day_outcome"])
        self.assertEqual(p50["source_asof_date"], "20260127")

        p95 = result.filter(pl.col("boundary_quantile") == 95).row(0, named=True)
        self.assertEqual(p95["upper_distance_bp"], 25.0)
        self.assertEqual(p95["lower_distance_bp"], 45.0)
        self.assertEqual(p95["boundary_role"], "tail_diagnostic")

    def test_invalid_prior_coverage_or_either_side_count_fails_gate(self) -> None:
        parameters = pl.DataFrame(
            [
                _parameter_row(ValueCode="valid"),
                _parameter_row(ValueCode="coverage", prior_eligible_rate=0.79),
                _parameter_row(
                    ValueCode="positive", prior_completed_positive=29
                ),
                _parameter_row(
                    ValueCode="negative", prior_completed_negative=29
                ),
            ]
        )
        result = build_adaptive_boundaries(parameters, quantiles=(50,)).sort(
            "ValueCode"
        )

        validity = dict(
            result.select("ValueCode", "adaptive_parameter_valid").iter_rows()
        )
        self.assertEqual(
            validity,
            {
                "coverage": False,
                "negative": False,
                "positive": False,
                "valid": True,
            },
        )
        universe = build_adaptive_cycle_universe(result)
        self.assertEqual(universe["ValueCode"].unique().to_list(), ["valid"])

    def test_prior_must_precede_target_and_target_keys_are_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "before target"):
            build_adaptive_boundaries(
                pl.DataFrame([_parameter_row(prior_date="20260128")]),
                quantiles=(50,),
            )
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            build_adaptive_boundaries(
                pl.DataFrame([_parameter_row(), _parameter_row()]),
                quantiles=(50,),
            )
        with self.assertRaisesRegex(ValueError, "valid YYYYMMDD"):
            build_adaptive_boundaries(
                pl.DataFrame([_parameter_row(prior_date=None)]),
                quantiles=(50,),
            )

    def test_invalid_boundary_has_no_supply_denominator(self) -> None:
        boundaries = build_adaptive_boundaries(
            pl.DataFrame([_parameter_row(prior_eligible_rate=0.79)]),
            quantiles=(50,),
        )
        excursions = pl.DataFrame(
            {
                "Date": ["20260128", "20260128"],
                "ValueCode": ["2303", "2303"],
                "QuoteCode": ["CCFB6", "CCFB6"],
                "side": ["positive", "negative"],
                "amplitude_bp": [20.0, 20.0],
                "completed": [True, True],
            }
        )
        row = add_boundary_supply(boundaries, excursions).row(0, named=True)
        self.assertEqual(row["upper_started"], 0)
        self.assertEqual(row["lower_started"], 0)
        self.assertIsNone(row["p_upper_reach_all_started_lower_bound"])
        self.assertIsNone(row["p_lower_reach_all_started_lower_bound"])

    def test_boundary_supply_separates_hit_known_miss_and_unknown(self) -> None:
        boundaries = build_adaptive_boundaries(
            pl.DataFrame([_parameter_row()]),
            quantiles=(50,),
        )
        excursions = pl.DataFrame(
            {
                "Date": ["20260128"] * 8,
                "ValueCode": ["2303"] * 8,
                "QuoteCode": ["CCFB6"] * 8,
                "side": ["positive"] * 4 + ["negative"] * 4,
                "amplitude_bp": [12.0, 8.0, 9.0, 11.0, 25.0, 10.0, 15.0, 22.0],
                "completed": [True, True, False, False] * 2,
            }
        )
        row = add_boundary_supply(boundaries, excursions).row(0, named=True)

        self.assertFalse(row["execution_safe_snapshot"])
        self.assertTrue(row["contains_target_day_outcome"])

        for prefix in ("upper", "lower"):
            self.assertEqual(row[f"{prefix}_started"], 4)
            self.assertEqual(row[f"{prefix}_hits"], 2)
            self.assertEqual(row[f"{prefix}_known_misses"], 1)
            self.assertEqual(row[f"{prefix}_unknown"], 1)
            self.assertAlmostEqual(
                row[f"p_{prefix}_reach_all_started_lower_bound"], 0.5
            )
            self.assertAlmostEqual(
                row[f"p_{prefix}_reach_known_case_conditional"], 2 / 3
            )

    def test_cycle_universe_uses_only_adaptive_product_bounds(self) -> None:
        boundaries = build_adaptive_boundaries(
            pl.DataFrame([_parameter_row()]),
            quantiles=(50,),
        )
        universe = build_adaptive_cycle_universe(boundaries)

        self.assertEqual(universe.height, 8)
        self.assertEqual(universe["width_family"].unique().to_list(), [
            "adaptive_product_prior"
        ])
        self.assertEqual(universe["width_policy"].unique().to_list(), [
            "adaptive_q50"
        ])
        self.assertFalse(
            universe["width_policy"].str.contains("fixed").any()
        )
        self.assertTrue((universe["candidate_width_bp"] == 10.0).all())

        center = universe.filter(pl.col("exit_target") == "center")
        adaptive = universe.filter(pl.col("exit_target") == "adaptive_lower")
        self.assertTrue((center["exit_width_bp"] == 0.0).all())
        self.assertTrue((center["exit_width_ratio"] == 0.0).all())
        self.assertTrue((adaptive["exit_width_bp"] == 20.0).all())
        self.assertTrue((adaptive["exit_width_ratio"] == 2.0).all())

    def test_adaptive_lower_exits_at_empirical_asymmetric_bound(self) -> None:
        boundaries = build_adaptive_boundaries(
            pl.DataFrame([_parameter_row()]),
            quantiles=(50,),
        )
        universe = build_adaptive_cycle_universe(boundaries).filter(
            (pl.col("anchor_mode") == "frozen_entry")
            & (pl.col("exit_delay_seconds") == 1)
        )
        cycles = build_latent_cycles(
            _cycle_panel([-1.0, 10.0, 0.0, -10.0, -20.0]),
            universe,
        )
        self.assertFalse(cycles["execution_safe_snapshot"].any())
        self.assertTrue(cycles["contains_target_day_outcome"].all())

        center = cycles.filter(pl.col("exit_target") == "center").row(
            0, named=True
        )
        adaptive = cycles.filter(
            pl.col("exit_target") == "adaptive_lower"
        ).row(0, named=True)
        self.assertEqual(center["entry_seconds_from_open"], 301)
        self.assertEqual(center["end_seconds_from_open"], 302)
        self.assertEqual(adaptive["entry_seconds_from_open"], 301)
        self.assertEqual(adaptive["end_seconds_from_open"], 304)
        self.assertEqual(adaptive["exit_width_bp"], 20.0)
        self.assertEqual(adaptive["exit_width_ratio"], 2.0)
        self.assertAlmostEqual(adaptive["basis_capture_bp"], 30.0)


if __name__ == "__main__":
    unittest.main()
