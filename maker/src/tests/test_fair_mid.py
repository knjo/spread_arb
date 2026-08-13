"""Synthetic tests for causal fair-mid candidates and labels."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import unittest

import polars as pl

from ..common.contracts import select_near_standard_contracts
from ..common.landmarks import _formal_after_trial, _join_state, _ref_ok
from ..fair_mid.anchors import add_anchor_candidates, prepare_fair_panel
from ..fair_mid.metrics import summarize_fresh_endpoint_reversion
from ..fair_mid.level_stability import analyze_level_stability
from ..fair_mid.prior_day import add_prior_candidates, summarize_prior_landmarks
from ..fair_mid.quote_churn import round_down_to_tick, round_up_to_tick
from ..quote_width.table import (
    _group_excursions,
    _outcome_at_horizon,
    add_microstructure_columns,
    summarize_entry_route_geometry,
)


class ContractSelectionTest(unittest.TestCase):
    def test_nearest_unexpired_standard_contract_is_selected(self) -> None:
        basic = pl.DataFrame(
            {
                "quote_code": ["AAFF6", "AAFG6", "AAFG6A", "AAMG6"],
                "value_code": ["1101", "1101", "1101", "1101"],
                "ref_price": [50.0, 51.0, 51.0, 51.0],
                "contract_size": [2000.0, 2000.0, 2000.0, 100.0],
                "decimal_locator": [2, 2, 2, 2],
                "end_date": [
                    date(2026, 6, 17),
                    date(2026, 7, 15),
                    date(2026, 7, 15),
                    date(2026, 7, 15),
                ],
            }
        )
        selected = select_near_standard_contracts(basic, date(2026, 6, 17))
        self.assertEqual(selected.height, 1)
        self.assertEqual(selected.item(0, "QuoteCode"), "AAFF6")


class FairAnchorTest(unittest.TestCase):
    def test_opening_anchor_and_forward_center_use_expected_windows(self) -> None:
        start = datetime(2026, 1, 28, 1, 0)
        seconds = list(range(700))
        landmarks = pl.DataFrame(
            {
                "Date": ["20260128"] * len(seconds),
                "ValueCode": ["2303"] * len(seconds),
                "QuoteCode": ["CCFB6"] * len(seconds),
                "timestamp": [start + timedelta(seconds=value) for value in seconds],
                "seconds_from_open": seconds,
                "basis_mid_bp": [float(value) for value in seconds],
                "eligible_base": [True] * len(seconds),
                "eligible_1000ms": [True] * len(seconds),
            }
        )
        panel = prepare_fair_panel(landmarks)
        row = panel.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        self.assertAlmostEqual(row["anchor_open_5m_bp"], 149.5)
        self.assertAlmostEqual(row["future_center_bp"], 465.0)
        self.assertAlmostEqual(row["future_300s_bp"], 600.0)

    def test_invalid_rows_freeze_ewma_state_instead_of_repeating_stale_basis(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["20260128"] * 5,
                "ValueCode": ["2303"] * 5,
                "seconds_from_open": [300, 301, 302, 303, 304],
                "basis_eval_bp": [0.0, 10.0, None, None, 20.0],
            }
        )
        result = add_anchor_candidates(frame)
        compressed = pl.DataFrame({"value": [0.0, 10.0, 20.0]}).select(
            pl.col("value")
            .ewm_mean(half_life=30, adjust=False, ignore_nulls=True)
            .alias("expected")
        )
        self.assertIsNone(result.item(2, "anchor_ewma_30s_bp"))
        self.assertIsNone(result.item(3, "anchor_ewma_30s_bp"))
        self.assertAlmostEqual(
            result.item(4, "anchor_ewma_30s_bp"),
            compressed.item(2, "expected"),
        )

    def test_endpoint_horizon_uses_timestamp_not_row_count(self) -> None:
        start = datetime(2026, 1, 28, 1, 5)
        rows = 6
        panel = pl.DataFrame(
            {
                "Date": ["20260128"] * rows,
                "ValueCode": ["2303"] * rows,
                "timestamp": [
                    start + timedelta(milliseconds=250 * value)
                    for value in range(rows)
                ],
                "seconds_from_open": [300] * rows,
                "basis_mid_bp": [float(value) for value in range(rows)],
                "leg_skew_ms": [0.0] * rows,
                "eligible_base": [True] * rows,
                "eligible_100ms": [True] * rows,
                "eligible_1000ms": [True] * rows,
                "anchor_ewma_120s_bp": [-1.0] * rows,
            }
        )
        result = summarize_fresh_endpoint_reversion(
            panel,
            horizon_seconds=1,
        )
        base = result.summary.filter(
            pl.col("sample") == "both_endpoints_base"
        )
        self.assertEqual(base.item(0, "n"), 2)


class PriorDayTest(unittest.TestCase):
    def test_prior_summary_uses_only_legal_fixed_grid_rows(self) -> None:
        frame = pl.DataFrame(
            {
                "ValueCode": ["2303"] * 4,
                "QuoteCode": ["CCFB6"] * 4,
                "seconds_from_open": [300, 301, 302, 13_800],
                "eligible_base": [True, False, True, True],
                "eligible_1000ms": [True, False, False, True],
                "basis_mid_bp": [10.0, 500.0, 30.0, 50.0],
                "spot_age_ms": [1.0, 2.0, 3.0, 4.0],
                "fut_age_ms": [5.0, 6.0, 7.0, 8.0],
            }
        )
        result = summarize_prior_landmarks(
            frame,
            target_date="20260128",
            prior_date="20260127",
        )
        self.assertAlmostEqual(result.item(0, "prior_mean_bp"), 30.0)
        self.assertAlmostEqual(result.item(0, "prior_coverage"), 0.75)
        self.assertAlmostEqual(result.item(0, "prior_close30_median_bp"), 50.0)

    def test_seeded_prior_affects_opening_anchor_without_fixed_blend(self) -> None:
        start = datetime(2026, 1, 28, 1, 0)
        panel = pl.DataFrame(
            {
                "Date": ["20260128"] * 3,
                "ValueCode": ["2303"] * 3,
                "QuoteCode": ["CCFB6"] * 3,
                "timestamp": [start + timedelta(seconds=value) for value in range(3)],
                "basis_eval_bp": [0.0, 0.0, 0.0],
                "anchor_ewma_120s_bp": [0.0, 0.0, 0.0],
            }
        )
        priors = pl.DataFrame(
            {
                "Date": ["20260128"],
                "ValueCode": ["2303"],
                "QuoteCode": ["CCFB6"],
                "prior_coverage": [1.0],
                "prior_mean_bp": [100.0],
                "prior_median_bp": [100.0],
                "prior_close30_median_bp": [100.0],
            }
        )
        result = add_prior_candidates(panel, priors)
        first = result.item(0, "anchor_seeded_prior_mean_bp")
        last = result.item(2, "anchor_seeded_prior_mean_bp")
        self.assertGreater(first, 0.0)
        self.assertGreater(first, last)
        self.assertAlmostEqual(
            result.item(0, "anchor_blend_prior_mean_10pct_bp"),
            10.0,
        )


class LevelStabilityTest(unittest.TestCase):
    def test_level_study_builds_all_quintiles_on_fixed_grid(self) -> None:
        start = datetime(2026, 1, 28, 1, 0)
        rows = 700
        seconds = list(range(rows))
        basis = [float(value % 100) for value in seconds]
        panel = pl.DataFrame(
            {
                "Date": ["20260128"] * rows,
                "ValueCode": ["2303"] * rows,
                "timestamp": [start + timedelta(seconds=value) for value in seconds],
                "seconds_from_open": seconds,
                "analysis_eligible": [value >= 300 for value in seconds],
                "future_center_bp": basis,
                "future_center_coverage": [1.0] * rows,
                "anchor_ewma_120s_bp": [value * 0.9 for value in basis],
                "anchor_ewma_300s_bp": [value * 0.8 for value in basis],
                "basis_mid_bp": basis,
                "leg_skew_ms": [0.0] * rows,
                "eligible_base": [True] * rows,
                "eligible_100ms": [True] * rows,
                "eligible_1000ms": [True] * rows,
            }
        )
        result = analyze_level_stability(panel)
        self.assertEqual(result.global_abs_level.height, 5)
        self.assertEqual(result.local_level.height, 5)
        self.assertEqual(result.fast_slow_gap.height, 5)
        self.assertGreater(result.gap_direction.height, 0)
        self.assertTrue(
            result.gap_direction.filter(pl.col("gap_side") == "positive").height
            > 0
        )


class EligibilityTest(unittest.TestCase):
    def test_nanosecond_event_after_landmark_is_not_joined(self) -> None:
        base_ns = 1_769_563_200_000_000_000
        grid = pl.DataFrame(
            {"ValueCode": ["2303"], "timestamp": [base_ns]}
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        state = pl.DataFrame(
            {
                "ValueCode": ["2303", "2303"],
                "fut_recv_time": [base_ns - 500, base_ns + 500],
                "state_id": [1, 2],
            }
        ).with_columns(pl.col("fut_recv_time").cast(pl.Datetime("ns")))
        joined = _join_state(grid, state, "fut_recv_time")
        self.assertEqual(joined.item(0, "state_id"), 1)

    def test_ref_boundaries_are_strict(self) -> None:
        frame = pl.DataFrame(
            {
                "spot_ref_price": [100.0, 100.0, 100.0],
                "spot_bid": [91.0, 91.0001, 91.0001],
                "spot_ask": [100.0, 107.9999, 108.0],
            }
        ).with_columns(_ref_ok("spot", "spot_ref_price").alias("valid"))
        self.assertEqual(frame["valid"].to_list(), [False, True, False])

    def test_formal_transition_requires_book_sequence_not_before_transition(self) -> None:
        timestamp = datetime(2026, 1, 28, 1, 0)
        frame = pl.DataFrame(
            {
                "spot_recv_time": [timestamp, timestamp, timestamp],
                "spot_sequence": [1, 2, 3],
                "spot_trial_time": [timestamp, timestamp, timestamp],
                "spot_trial_sequence": [2, 2, 2],
                "spot_trial_match": [0, 0, 0],
            }
        ).with_columns(_formal_after_trial("spot").alias("formal"))
        self.assertEqual(frame["formal"].to_list(), [False, True, True])


class TickRoundingTest(unittest.TestCase):
    def test_rounding_across_price_ladder_boundaries(self) -> None:
        frame = pl.DataFrame(
            {"price": [9.999, 10.001, 49.999, 50.001, 99.999, 100.001]}
        ).select(
            round_down_to_tick(pl.col("price")).alias("down"),
            round_up_to_tick(pl.col("price")).alias("up"),
        )
        self.assertEqual(frame["down"].to_list(), [9.99, 10.0, 49.95, 50.0, 99.9, 100.0])
        self.assertEqual(frame["up"].to_list(), [10.0, 10.05, 50.0, 50.1, 100.0, 100.5])


class WidthTableTest(unittest.TestCase):
    def test_microstructure_scales_keep_tick_bp_and_spread_ticks_separate(self) -> None:
        frame = pl.DataFrame(
            {
                "spot_bid": [99.9],
                "spot_ask": [100.0],
                "fut_bid": [100.0],
                "fut_ask": [100.5],
                "fut_exec_bid": [100.0],
                "basis_sell_taker_bp": [0.0],
                "basis_buy_taker_bp": [60.0],
            }
        )
        result = add_microstructure_columns(frame).row(0, named=True)
        self.assertEqual(result["spot_spread_ticks"], 1.0)
        self.assertEqual(result["fut_spread_ticks"], 1.0)
        self.assertAlmostEqual(result["future_ask_route_tick_bp"], 50.0)
        self.assertGreater(result["spot_bid_route_tick_bp"], 0.0)
        self.assertAlmostEqual(result["tt_band_width_bp"], 60.0)

    def test_zero_crossing_excursions_are_non_overlapping_and_censor_gaps(self) -> None:
        residuals = [-1.0, 1.0, 3.0, 1.0, -1.0, 2.0, 4.0, None]
        valid = [True, True, True, True, True, True, True, False]
        result = _group_excursions(residuals, valid, list(range(len(residuals))))
        positive = [row for row in result if row["side"] == "positive"]
        self.assertEqual(len(positive), 2)
        self.assertEqual(positive[0]["amplitude_bp"], 3.0)
        self.assertTrue(positive[0]["completed"])
        self.assertEqual(positive[1]["amplitude_bp"], 4.0)
        self.assertFalse(positive[1]["completed"])
        self.assertEqual(positive[1]["end_reason"], "eligibility_gap")

    def test_censored_horizon_is_not_conditioned_on_an_early_hit(self) -> None:
        residuals = [10.0, 4.0, -1.0, None]
        valid = [True, True, True, False]
        seconds = [0, 1, 2, 3]
        center, symmetric = _outcome_at_horizon(
            residuals,
            valid,
            seconds,
            trigger_index=0,
            width_bp=10.0,
            horizon_seconds=3,
            observation_gap_seconds=1,
        )
        self.assertIsNone(center)
        self.assertIsNone(symmetric)

    def test_entry_routes_round_to_at_least_nominal_basis_width(self) -> None:
        start = datetime(2026, 1, 28, 1, 5)
        panel = pl.DataFrame(
            {
                "Date": ["20260128"],
                "ValueCode": ["2303"],
                "QuoteCode": ["CCFB6"],
                "timestamp": [start],
                "analysis_eligible": [True],
                "anchor_ewma_120s_bp": [50.0],
                "spot_bid": [99.9],
                "spot_ask": [100.0],
                "fut_bid": [100.5],
                "fut_ask": [101.0],
                "fut_exec_bid": [100.5],
                "basis_sell_taker_bp": [50.0],
                "spot_ref_price": [100.0],
                "fut_ref_price": [100.5],
            }
        )
        candidates = pl.DataFrame(
            {
                "Date": ["20260128"],
                "ValueCode": ["2303"],
                "QuoteCode": ["CCFB6"],
                "width_family": ["fixed_bp"],
                "width_policy": ["fixed_20bp"],
                "candidate_width_bp": [20.0],
            }
        )
        result = summarize_entry_route_geometry(panel, candidates)
        self.assertEqual(result.height, 2)
        self.assertEqual(result["rounding_inequality_violations"].sum(), 0)
        self.assertTrue((result["effective_width_bp_p50"] >= 20.0).all())


if __name__ == "__main__":
    unittest.main()
