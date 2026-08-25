"""Tests for D-safe foundation geometry and exact transaction costs."""

from __future__ import annotations

import math
import unittest

import polars as pl

from maker.src.quote_fill.foundation_geometry import (
    FOUNDATION_GEOMETRY_VERSION,
    build_policy_geometry,
    summarize_policy_geometry,
)
from maker.src.quote_fill.targets import PRICE_LADDER_VERSION
from maker.src.quote_fill.transaction_costs import TransactionCostProfile


class FoundationGeometryTest(unittest.TestCase):
    def test_exact_per_leg_costs_and_same_day_tax_delta(self) -> None:
        profile = TransactionCostProfile()
        breakdown = profile.paired_cycle_cost_breakdown(
            entry_spot_price=100.0,
            exit_spot_price=101.0,
            entry_future_price=102.0,
            exit_future_price=99.0,
            shares=2_000,
            contracts=1,
            same_day=True,
        )
        self.assertAlmostEqual(breakdown.spot_entry_commission_twd, 34.2)
        self.assertAlmostEqual(breakdown.spot_exit_commission_twd, 34.542)
        self.assertAlmostEqual(breakdown.spot_exit_tax_twd, 303.0)
        self.assertAlmostEqual(breakdown.futures_entry_tax_twd, 4.08)
        self.assertAlmostEqual(breakdown.futures_exit_tax_twd, 3.96)
        self.assertEqual(breakdown.futures_entry_commission_twd, 20.0)
        self.assertEqual(breakdown.futures_exit_commission_twd, 20.0)
        self.assertAlmostEqual(
            profile.paired_cycle_cost_twd(
                entry_spot_price=100.0,
                exit_spot_price=101.0,
                entry_future_price=102.0,
                exit_future_price=99.0,
                shares=2_000,
                contracts=1,
                same_day=True,
            ),
            breakdown.total_twd,
        )
        overnight = profile.paired_cycle_cost_twd(
            entry_spot_price=100.0,
            exit_spot_price=101.0,
            entry_future_price=102.0,
            exit_future_price=99.0,
            shares=2_000,
            contracts=1,
            same_day=False,
        )
        self.assertAlmostEqual(
            overnight - breakdown.total_twd,
            2_000 * 101.0 * 15.0 / 10_000.0,
        )
        # Legacy aggregate API remains unchanged.
        self.assertAlmostEqual(profile.same_day_variable_cost_bp, 18.82)
        self.assertAlmostEqual(profile.overnight_variable_cost_bp, 33.82)
        self.assertEqual(profile.futures_round_trip_commission_twd, 40.0)

    def test_cost_primitives_reject_nonpositive_or_nonfinite_inputs(self) -> None:
        profile = TransactionCostProfile()
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    profile.spot_commission_twd(value, 2_000)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    profile.futures_commission_twd(value)
        with self.assertRaisesRegex(TypeError, "finite and positive"):
            profile.spot_commission_twd(True, 2_000)
        with self.assertRaisesRegex(TypeError, "finite and positive"):
            profile.futures_commission_twd(True)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            profile.spot_commission_twd(None, 2_000)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            profile.futures_commission_twd(None)
        with self.assertRaisesRegex(TypeError, "same_day must be boolean"):
            profile.spot_sell_tax_twd(
                100.0,
                2_000,
                same_day=1,  # type: ignore[arg-type]
            )

    def test_seven_policy_common_geometry_adds_positive_lower_distance(self) -> None:
        result = build_policy_geometry(_cohort(), _long_boundaries())
        self.assertEqual(result.height, 14)
        self.assertEqual(
            set(result["policy_id"].to_list()),
            {"q50", "q80", "q95", "fixed15", "fixed20", "fixed25", "fixed30"},
        )
        self.assertTrue(
            result.group_by("policy_id")
            .len()
            .select((pl.col("len") == 2).all())
            .item()
        )
        q50 = result.filter(
            (pl.col("Date") == "20260703") & (pl.col("policy_id") == "q50")
        ).row(0, named=True)
        self.assertEqual(q50["upper_distance_bp"], 10.0)
        self.assertEqual(q50["lower_distance_bp"], 7.0)
        self.assertEqual(q50["nominal_band_bp"], 17.0)
        self.assertGreaterEqual(q50["reference_entry_rounding_excess_bp"], -1e-7)
        self.assertGreaterEqual(q50["reference_exit_rounding_excess_bp"], -1e-7)
        self.assertGreaterEqual(q50["rounded_reference_band_bp"], 17.0 - 1e-7)
        fixed = result.filter(
            (pl.col("Date") == "20260703")
            & (pl.col("policy_id") == "fixed15")
        ).row(0, named=True)
        self.assertEqual(fixed["upper_distance_bp"], 15.0)
        self.assertEqual(fixed["lower_distance_bp"], 15.0)
        self.assertEqual(fixed["nominal_band_bp"], 30.0)
        self.assertEqual(fixed["geometry_version"], FOUNDATION_GEOMETRY_VERSION)
        self.assertTrue(fixed["reference_diagnostic"])
        self.assertFalse(fixed["intraday_bbo_available"])
        self.assertFalse(fixed["passive_clamp_applied"])
        self.assertFalse(fixed["actionable_execution"])
        self.assertFalse(fixed["ev_ready"])

    def test_future_reference_tick_uses_20260706_ladder_regime(self) -> None:
        result = build_policy_geometry(_cohort(), _long_boundaries()).filter(
            pl.col("policy_id") == "q50"
        )
        before = result.filter(pl.col("Date") == "20260703").row(0, named=True)
        after = result.filter(pl.col("Date") == "20260706").row(0, named=True)
        self.assertEqual(before["reference_future_next_tick_price"], 2135.0)
        self.assertEqual(after["reference_future_next_tick_price"], 2131.0)
        self.assertAlmostEqual(
            before["reference_future_tick_bp"],
            10_000 * 5 / 2130,
        )
        self.assertAlmostEqual(
            after["reference_future_tick_bp"],
            10_000 * 1 / 2130,
        )
        self.assertEqual(after["price_ladder_version"], PRICE_LADDER_VERSION)

    def test_fixed_twd_commission_has_larger_bp_effect_for_low_price(self) -> None:
        cohort = _cohort_for_prices((10.0, 100.0))
        boundary = _boundaries_for_prices((10.0, 100.0))
        fixed30 = build_policy_geometry(cohort, boundary).filter(
            pl.col("policy_id") == "fixed30"
        )
        low = fixed30.filter(pl.col("ValueCode") == "1001").row(0, named=True)
        high = fixed30.filter(pl.col("ValueCode") == "1002").row(0, named=True)
        self.assertGreater(
            low["same_day_reference_cost_bp"],
            high["same_day_reference_cost_bp"],
        )
        self.assertLess(
            low["same_day_known_cost_margin_bp"],
            high["same_day_known_cost_margin_bp"],
        )

    def test_ttband_is_not_deducted_again(self) -> None:
        low_ttband = _cohort().with_columns(
            pl.lit(1.0).alias("tt_band_reference_bp")
        )
        high_ttband = low_ttband.with_columns(
            pl.lit(999.0).alias("tt_band_reference_bp")
        )
        first = build_policy_geometry(low_ttband, _long_boundaries())
        second = build_policy_geometry(high_ttband, _long_boundaries())
        columns = [
            "reference_gross_cycle_pnl_twd",
            "same_day_reference_cost_twd",
            "same_day_known_cost_margin_bp",
            "overnight_known_cost_margin_bp",
        ]
        self.assertEqual(
            first.select(columns).to_dicts(),
            second.select(columns).to_dicts(),
        )
        self.assertFalse(first["tt_band_deducted_as_cost"].any())

    def test_adverse_sensitivity_is_a_single_explicit_haircut(self) -> None:
        row = build_policy_geometry(_cohort(), _long_boundaries()).row(
            0,
            named=True,
        )
        base = row["same_day_known_cost_margin_bp"]
        self.assertAlmostEqual(
            row["same_day_margin_after_10bp_adverse_bp"],
            base - 10.0,
        )
        self.assertAlmostEqual(
            row["same_day_margin_after_20bp_adverse_bp"],
            base - 20.0,
        )
        self.assertAlmostEqual(
            row["same_day_margin_after_30bp_adverse_bp"],
            base - 30.0,
        )

    def test_rejects_leakage_or_incomplete_common_quantile_coverage(self) -> None:
        unsafe = _cohort().with_columns(
            pl.col("Date").alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "strictly pre-open safe"):
            build_policy_geometry(unsafe, _long_boundaries())
        incomplete = _long_boundaries().filter(
            ~(
                (pl.col("Date") == "20260703")
                & (pl.col("boundary_quantile") == 95)
            )
        )
        with self.assertRaisesRegex(ValueError, "common q50/q80/q95 coverage"):
            build_policy_geometry(_cohort(), incomplete)

    def test_accepts_true_wide_boundary_input_and_summary_stays_not_ready(self) -> None:
        wide = _wide_boundaries()
        result = build_policy_geometry(_cohort().head(1), wide)
        self.assertEqual(result.height, 7)
        summary = summarize_policy_geometry(result)
        self.assertEqual(summary.height, 7)
        self.assertFalse(summary["actionable_execution"].any())
        self.assertFalse(summary["ev_ready"].any())

    def test_accepts_foundation_boundary_wide_names_with_enriched_cohort(self) -> None:
        wide = (
            _wide_boundaries()
            .rename(
                {
                    "q50_upper_distance_bp": "upper_distance_bp_50",
                    "q50_lower_distance_bp": "lower_distance_bp_50",
                    "q80_upper_distance_bp": "upper_distance_bp_80",
                    "q80_lower_distance_bp": "lower_distance_bp_80",
                    "q95_upper_distance_bp": "upper_distance_bp_95",
                    "q95_lower_distance_bp": "lower_distance_bp_95",
                }
            )
            .drop(
                "spot_ref_price",
                "fut_ref_price",
                "contract_size",
                "execution_safe_snapshot",
                "adaptive_parameter_valid",
            )
            .with_columns(pl.lit(True).alias("all_boundary_supported"))
        )
        cohort = _cohort().head(1).with_columns(
            pl.lit(2130.0).alias("spot_ref_price"),
            pl.lit(2130.0).alias("fut_ref_price"),
            pl.lit(2000.0).alias("contract_size"),
        )
        result = build_policy_geometry(cohort, wide)
        self.assertEqual(result.height, 7)
        self.assertEqual(result["nominal_band_bp"].min(), 17.0)


def _cohort() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260703", "20260706"],
            "ValueCode": ["2317", "2317"],
            "QuoteCode": ["DHFN6", "DHFN6"],
            "source_asof_date": ["20260702", "20260703"],
            "execution_safe_snapshot": [True, True],
            "contains_target_day_outcome": [False, False],
            "tt_band_reference_bp": [20.0, 20.0],
        }
    )


def _long_boundaries() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    distances = {50: (10.0, 7.0), 80: (20.0, 15.0), 95: (35.0, 30.0)}
    for date, source in (("20260703", "20260702"), ("20260706", "20260703")):
        for quantile, (upper, lower) in distances.items():
            rows.append(
                {
                    "Date": date,
                    "ValueCode": "2317",
                    "QuoteCode": "DHFN6",
                    "source_asof_date": source,
                    "execution_safe_snapshot": True,
                    "contains_target_day_outcome": False,
                    "spot_ref_price": 2130.0,
                    "fut_ref_price": 2130.0,
                    "contract_size": 2000.0,
                    "boundary_quantile": quantile,
                    "boundary_role": (
                        "tail_diagnostic"
                        if quantile == 95
                        else "rolling_latent_candidate"
                    ),
                    "upper_distance_bp": upper,
                    "lower_distance_bp": lower,
                    "adaptive_parameter_valid": True,
                    "price_ladder_version": PRICE_LADDER_VERSION,
                }
            )
    return pl.from_dicts(rows)


def _cohort_for_prices(prices: tuple[float, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260601"] * len(prices),
            "ValueCode": [f"{1001 + index}" for index in range(len(prices))],
            "QuoteCode": [f"AA{index}F6" for index in range(len(prices))],
            "source_asof_date": ["20260529"] * len(prices),
            "execution_safe_snapshot": [True] * len(prices),
            "contains_target_day_outcome": [False] * len(prices),
        }
    )


def _boundaries_for_prices(prices: tuple[float, ...]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, price in enumerate(prices):
        for quantile, distance in ((50, 10.0), (80, 20.0), (95, 35.0)):
            rows.append(
                {
                    "Date": "20260601",
                    "ValueCode": f"{1001 + index}",
                    "QuoteCode": f"AA{index}F6",
                    "source_asof_date": "20260529",
                    "execution_safe_snapshot": True,
                    "contains_target_day_outcome": False,
                    "spot_ref_price": price,
                    "fut_ref_price": price,
                    "contract_size": 2000.0,
                    "boundary_quantile": quantile,
                    "boundary_role": "rolling_latent_candidate",
                    "upper_distance_bp": distance,
                    "lower_distance_bp": distance,
                    "adaptive_parameter_valid": True,
                    "price_ladder_version": PRICE_LADDER_VERSION,
                }
            )
    return pl.from_dicts(rows)


def _wide_boundaries() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260703"],
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFN6"],
            "source_asof_date": ["20260702"],
            "execution_safe_snapshot": [True],
            "contains_target_day_outcome": [False],
            "spot_ref_price": [2130.0],
            "fut_ref_price": [2130.0],
            "contract_size": [2000.0],
            "price_ladder_version": [PRICE_LADDER_VERSION],
            "q50_upper_distance_bp": [10.0],
            "q50_lower_distance_bp": [7.0],
            "q80_upper_distance_bp": [20.0],
            "q80_lower_distance_bp": [15.0],
            "q95_upper_distance_bp": [35.0],
            "q95_lower_distance_bp": [30.0],
            "adaptive_parameter_valid": [True],
        }
    )


if __name__ == "__main__":
    unittest.main()
