"""Focused tests for the S0.5 canonical runner utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from maker.src.quote_fill.foundation_revalidation_runner import (
    DEFAULT_SESSIONS_PATH,
    _anchor_bootstrap,
    _assert_input_inventory_current,
    _published_anchor_equivalence,
    _sink_lazy_frame,
    _summarize_anchor_daily,
    build_input_inventory,
    load_frozen_sessions,
    rank_validation_by_date,
    stage_for_date,
    summarize_calibration,
    whole_date_bootstrap,
)


class SessionContractTest(unittest.TestCase):
    def test_current_frozen_calendar_and_stages(self) -> None:
        sessions = load_frozen_sessions(DEFAULT_SESSIONS_PATH)
        self.assertEqual(len(sessions), 131)
        self.assertEqual(sessions[0], "20260126")
        self.assertEqual(sessions[-1], "20260813")
        self.assertEqual(stage_for_date("20260504"), "history")
        self.assertEqual(stage_for_date("20260505"), "fine_tune")
        self.assertEqual(stage_for_date("20260601"), "confirmation")
        self.assertEqual(stage_for_date("20260701"), "pseudo_holdout")

    def test_protected_forward_session_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "sessions.txt"
            path.write_text(
                DEFAULT_SESSIONS_PATH.read_text(encoding="utf-8")
                + "20260814\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "protected forward"):
                load_frozen_sessions(path)


class BootstrapAndSummaryTest(unittest.TestCase):
    def test_whole_date_bootstrap_is_deterministic(self) -> None:
        daily = pl.DataFrame(
            {
                "Date": ["20260505", "20260505", "20260506", "20260506"],
                "model": ["m"] * 4,
                "metric": [0.0, 2.0, 10.0, 12.0],
            }
        )
        first = whole_date_bootstrap(
            daily,
            group_columns=["model"],
            metric_columns=["metric"],
            seed=7,
            replicates=50,
        )
        second = whole_date_bootstrap(
            daily,
            group_columns=["model"],
            metric_columns=["metric"],
            seed=7,
            replicates=50,
        )
        self.assertTrue(first.equals(second, null_equal=True))
        self.assertAlmostEqual(first.item(0, "estimate"), 6.0)
        self.assertEqual(first.item(0, "resampling_unit"), "whole_Date")

    def test_anchor_summary_uses_evaluable_not_pairwise_denominator(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["20260505"],
                "ValueCode": ["2330"],
                "model": ["open_5m"],
                "freshness_sample": ["base"],
                "stratum_family": ["overall"],
                "stratum_value": ["all"],
                "n_evaluable": [10],
                "n_pairwise_common": [5],
                "sum_error_bp": [20.0],
                "sum_abs_error_bp": [30.0],
                "pairwise_model_sum_abs_error_bp": [12.0],
                "pairwise_baseline_sum_abs_error_bp": [10.0],
                "anchor_tv_bp": [5.0],
                "basis_tv_bp": [10.0],
                "n_adjacent_legal_pairs": [9],
                "mae_bp": [3.0],
                "p80_abs_error_bp": [4.0],
                "p95_abs_error_bp": [5.0],
            }
        )
        row = _summarize_anchor_daily(frame).row(0, named=True)
        self.assertAlmostEqual(row["occupancy_weighted_bias_bp"], 2.0)
        self.assertAlmostEqual(row["occupancy_weighted_mae_bp"], 3.0)
        self.assertAlmostEqual(row["pairwise_delta_mae_vs_ewma120_bp"], 0.4)
        bootstrap = _anchor_bootstrap(frame)
        self.assertEqual(bootstrap.height, 3)

    def test_calibration_preserves_unknown_product_day(self) -> None:
        calibration = pl.DataFrame(
            {
                "Date": ["20260505", "20260505"],
                "ValueCode": ["2330", "2317"],
                "boundary_quantile": [95, 95],
                "side": ["positive", "positive"],
                "n_started": [10, 0],
                "n_hit": [1, 0],
                "n_known_miss": [8, 0],
                "n_unknown_censored": [1, 0],
                "reach_lower_bound": [0.1, None],
                "reach_upper_bound": [0.2, None],
                "complete_case_reach": [1 / 9, None],
                "predicted_distance_bp": [50.0, 60.0],
                "realized_completed_quantile_bp": [49.0, None],
            }
        )
        row = summarize_calibration(calibration).row(0, named=True)
        self.assertEqual(row["product_days"], 2)
        self.assertEqual(row["observable_product_days"], 1)
        self.assertEqual(row["no_observable_excursion_product_days"], 1)
        self.assertAlmostEqual(row["event_pooled_reach_lower_bound"], 0.1)

    def test_rank_validation_uses_polars_spearman_expression(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["20260505"] * 3,
                "boundary_quantile": [95] * 3,
                "side": ["positive"] * 3,
                "predicted_distance_bp": [1.0, 2.0, 3.0],
                "realized_completed_quantile_bp": [10.0, 20.0, 30.0],
            }
        )
        row = rank_validation_by_date(frame).row(0, named=True)
        self.assertAlmostEqual(row["spearman_predicted_vs_realized"], 1.0)


class ProvenanceAndStreamingTest(unittest.TestCase):
    def test_inventory_rehash_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "input.txt"
            path.write_text("before\n", encoding="utf-8")
            inventory = build_input_inventory([(path, "test")])
            _assert_input_inventory_current(inventory)
            path.write_text("after\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input content drift"):
                _assert_input_inventory_current(inventory)

    def test_streaming_sink_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "result.parquet"
            frame = pl.DataFrame({"x": [1, 2, 3]})
            _sink_lazy_frame(frame.lazy(), path)
            self.assertEqual(pl.read_parquet(path)["x"].to_list(), [1, 2, 3])

    def test_published_anchor_equivalence_flags_drift(self) -> None:
        day = pl.DataFrame(
            {
                "Date": ["20260505"] * 3,
                "ValueCode": ["2330"] * 3,
                "QuoteCode": ["CDF"] * 3,
                "timestamp": pl.datetime_range(
                    pl.datetime(2026, 5, 5, 9, 0),
                    pl.datetime(2026, 5, 5, 9, 0, 2),
                    interval="1s",
                    eager=True,
                ).cast(pl.Datetime("ns")),
                "seconds_from_open": [300, 301, 302],
                "basis_mid_bp": [1.0, 2.0, 3.0],
                "basis_eval_bp": [1.0, 2.0, 3.0],
                "eligible_base": [True, True, True],
                "analysis_eligible": [True, True, True],
                "anchor_ewma_120s_bp": [1.0, 99.0, 99.0],
            }
        )
        audit = _published_anchor_equivalence(day)
        self.assertGreater(audit["published_ewma120_max_abs_difference_bp"], 1.0)

    def test_published_anchor_bridge_flags_basis_and_gate_drift(self) -> None:
        expected_basis = pl.Series([1.0, 2.0, 3.0])
        day = pl.DataFrame(
            {
                "Date": ["20260505"] * 3,
                "ValueCode": ["2330"] * 3,
                "QuoteCode": ["CDF"] * 3,
                "timestamp": pl.datetime_range(
                    pl.datetime(2026, 5, 5, 9, 0),
                    pl.datetime(2026, 5, 5, 9, 0, 2),
                    interval="1s",
                    eager=True,
                ).cast(pl.Datetime("ns")),
                "seconds_from_open": [300, 301, 302],
                "basis_mid_bp": expected_basis,
                "basis_eval_bp": [1.0, 200.0, 3.0],
                "eligible_base": [True, True, True],
                "analysis_eligible": [True, False, True],
                "anchor_ewma_120s_bp": expected_basis.ewm_mean(
                    half_life=120,
                    adjust=False,
                    ignore_nulls=True,
                ),
            }
        )
        audit = _published_anchor_equivalence(day)
        self.assertGreater(
            audit["published_basis_eval_max_abs_difference_bp"],
            100.0,
        )
        self.assertEqual(
            audit["published_analysis_eligible_mismatch_rows"],
            1,
        )
        self.assertAlmostEqual(
            audit["published_ewma120_max_abs_difference_bp"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
