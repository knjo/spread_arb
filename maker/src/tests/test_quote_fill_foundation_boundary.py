"""Contracts for S0.5 censor-aware boundaries and broad cohorts."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from maker.src.quote_fill.foundation_boundary import (
    build_foundation_cohorts,
    build_product_day_censor_bounds,
    calibrate_boundary_product_days,
    extract_censor_aware_excursions,
)


def _panel(
    observations: list[tuple[int, bool, float | None]],
    *,
    date: str = "20260505",
    value_code: str = "1111",
    quote_code: str = "Q1",
) -> pl.DataFrame:
    start = datetime(2026, 5, 5, 9)  # noqa: DTZ001 - source timestamps are naive
    rows = []
    for second, eligible, residual in observations:
        anchor = 100.0 if residual is not None else None
        rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "timestamp": start + timedelta(seconds=second),
                "seconds_from_open": second,
                "basis_mid_bp": (
                    anchor + residual
                    if anchor is not None and residual is not None
                    else None
                ),
                "anchor_ewma_120s_bp": anchor,
                "analysis_eligible": eligible,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _boundaries(
    *,
    value_codes: tuple[str, ...] = ("1111",),
    source_asof_date: str = "20260504",
) -> pl.DataFrame:
    rows = []
    for value_code in value_codes:
        for quantile, upper, lower in (
            (50, 5.0, 4.0),
            (80, 8.0, 7.0),
            (95, 10.0, 9.0),
        ):
            rows.append(
                {
                    "Date": "20260505",
                    "ValueCode": value_code,
                    "QuoteCode": f"Q{value_code}",
                    "boundary_quantile": quantile,
                    "upper_distance_bp": upper,
                    "lower_distance_bp": lower,
                    "adaptive_parameter_valid": True,
                    "history_sessions_global": 60,
                    "source_asof_date": source_asof_date,
                    "train_start_date": "20260202",
                    "train_end_date": source_asof_date,
                    "parameter_version": "test-60-v1",
                    "execution_safe_snapshot": True,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _liquidity(
    *,
    value_codes: tuple[str, ...] = ("1111",),
    candidate_by_product: dict[str, bool] | None = None,
) -> pl.DataFrame:
    rows = []
    candidates = candidate_by_product or {
        value_code: True for value_code in value_codes
    }
    for value_code in value_codes:
        candidate = candidates[value_code]
        for quantile in (50, 80):
            rows.append(
                {
                    "Date": "20260505",
                    "ValueCode": value_code,
                    "QuoteCode": f"Q{value_code}",
                    "boundary_quantile": quantile,
                    "route": "spot_bid_future_taker",
                    "support_gate": True,
                    "hard_data_gate": candidate,
                    "pre_replay_candidate": candidate,
                    "liquidity_gate_status": "pass" if candidate else "known_fail",
                    "source_asof_date": "20260504",
                    "execution_safe_snapshot": True,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


class EpisodeOverlayTests(unittest.TestCase):
    def test_gap_creates_right_and_left_censored_episodes(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, 2.0),
                    (302, False, None),
                    (303, True, 3.0),
                    (304, True, 0.0),
                ]
            )
        ).sort("episode_sequence")

        self.assertEqual(episodes.height, 2)
        before = episodes.row(0, named=True)
        after = episodes.row(1, named=True)
        self.assertFalse(before["left_censored"])
        self.assertTrue(before["right_censored"])
        self.assertEqual(before["right_censor_reason"], "eligibility_gap")
        self.assertAlmostEqual(before["observed_amplitude_bp"], 2.0)
        self.assertTrue(after["left_censored"])
        self.assertEqual(after["left_censor_reason"], "eligibility_gap")
        self.assertTrue(after["completed_center_return"])
        self.assertFalse(after["fully_observed"])

    def test_timestamp_gap_and_session_cutoff_are_explicit(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, -2.0),
                    (304, True, -3.0),
                    (305, True, -4.0),
                ]
            )
        ).sort("episode_sequence")

        self.assertEqual(episodes.height, 2)
        first = episodes.row(0, named=True)
        second = episodes.row(1, named=True)
        self.assertEqual(first["side"], "negative")
        self.assertEqual(first["right_censor_reason"], "timestamp_gap")
        self.assertTrue(second["left_censored"])
        self.assertEqual(second["left_censor_reason"], "timestamp_gap")
        self.assertEqual(second["right_censor_reason"], "session_cutoff")

    def test_direct_sign_change_starts_the_other_side_on_same_row(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, 3.0),
                    (302, True, -4.0),
                    (303, True, 0.0),
                ]
            )
        ).sort("episode_sequence")

        self.assertEqual(episodes.height, 2)
        positive = episodes.row(0, named=True)
        negative = episodes.row(1, named=True)
        self.assertEqual(positive["end_seconds_from_open"], 302)
        self.assertEqual(negative["start_seconds_from_open"], 302)
        self.assertTrue(positive["fully_observed"])
        self.assertTrue(negative["fully_observed"])


class CensorBoundTests(unittest.TestCase):
    def test_threshold_equality_is_a_confirmed_hit(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, 5.0),
                    (302, True, 0.0),
                ],
                quote_code="Q1111",
            )
        )
        bounds = build_product_day_censor_bounds(_boundaries(), episodes)
        q50_positive = bounds.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("side") == "positive")
        ).row(0, named=True)

        self.assertEqual(q50_positive["observable_started"], 1)
        self.assertEqual(q50_positive["confirmed_hits"], 1)
        self.assertEqual(q50_positive["n_started"], 1)
        self.assertEqual(q50_positive["n_hit"], 1)
        self.assertEqual(q50_positive["n_known_miss"], 0)
        self.assertEqual(q50_positive["n_unknown_censored"], 0)
        self.assertAlmostEqual(q50_positive["predicted_distance_bp"], 5.0)
        self.assertAlmostEqual(
            q50_positive["realized_completed_quantile_bp"],
            5.0,
        )
        self.assertAlmostEqual(q50_positive["reach_lower_bound"], 1.0)
        self.assertAlmostEqual(q50_positive["reach_upper_bound"], 1.0)

    def test_right_censored_below_boundary_is_unknown(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, 2.0),
                ],
                quote_code="Q1111",
            )
        )
        bounds = build_product_day_censor_bounds(_boundaries(), episodes)
        q50_positive = bounds.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("side") == "positive")
        ).row(0, named=True)

        self.assertEqual(q50_positive["confirmed_hits"], 0)
        self.assertEqual(q50_positive["known_nonhits"], 0)
        self.assertEqual(q50_positive["unknown_nonhits"], 1)
        self.assertAlmostEqual(q50_positive["reach_lower_bound"], 0.0)
        self.assertAlmostEqual(q50_positive["reach_upper_bound"], 1.0)
        self.assertIsNone(q50_positive["complete_case_reach"])

    def test_no_observable_excursion_boundary_rows_are_preserved(self) -> None:
        empty = extract_censor_aware_excursions(
            _panel(
                [
                    (300, True, 0.0),
                    (301, True, 0.0),
                ],
                quote_code="Q1111",
            )
        )
        bounds = build_product_day_censor_bounds(_boundaries(), empty)

        self.assertEqual(bounds.height, 6)
        self.assertTrue(
            (
                bounds["outcome_status"]
                == "no_observable_excursion"
            ).all()
        )
        self.assertTrue((bounds["observable_started"] == 0).all())
        self.assertTrue(bounds["reach_lower_bound"].is_null().all())
        self.assertTrue(bounds["reach_upper_bound"].is_null().all())
        self.assertTrue(bounds["realized_completed_quantile_bp"].is_null().all())

    def test_runner_api_filters_to_explicit_supported_keys(self) -> None:
        episodes = extract_censor_aware_excursions(
            _panel(
                [(300, True, 0.0), (301, True, 5.0), (302, True, 0.0)],
                quote_code="Q1111",
            )
        )
        boundaries = _boundaries(value_codes=("1111", "2222"))
        supported = pl.DataFrame(
            {
                "Date": ["20260505"],
                "ValueCode": ["1111"],
                "QuoteCode": ["Q1111"],
            }
        )
        result = calibrate_boundary_product_days(
            boundaries,
            episodes,
            supported,
        )

        self.assertEqual(result.height, 6)
        self.assertEqual(result["ValueCode"].unique().to_list(), ["1111"])


class FoundationCohortTests(unittest.TestCase):
    def test_builds_full_supported_and_q_independent_spot_bid_cohort(self) -> None:
        boundaries = _boundaries(value_codes=("1111", "2222"))
        liquidity = _liquidity(
            value_codes=("1111", "2222"),
            candidate_by_product={"1111": True, "2222": False},
        )
        cohorts = build_foundation_cohorts(
            boundaries,
            liquidity,
            primary_start="20260505",
            source_end="20260505",
        )

        self.assertEqual(cohorts.all_boundary_supported.height, 2)
        self.assertEqual(cohorts.spot_bid_broad.height, 1)
        self.assertTrue(
            cohorts.spot_bid_broad.equals(
                cohorts.spot_bid_broad_candidate
            )
        )
        self.assertEqual(
            cohorts.spot_bid_broad.row(0, named=True)["ValueCode"],
            "1111",
        )

        changed_q95 = boundaries.with_columns(
            pl.when(pl.col("boundary_quantile") == 95)
            .then(pl.col("upper_distance_bp") + 100.0)
            .otherwise(pl.col("upper_distance_bp"))
            .alias("upper_distance_bp"),
            pl.when(pl.col("boundary_quantile") == 95)
            .then(pl.col("lower_distance_bp") + 100.0)
            .otherwise(pl.col("lower_distance_bp"))
            .alias("lower_distance_bp"),
        )
        changed = build_foundation_cohorts(changed_q95, liquidity)
        original_keys = cohorts.spot_bid_broad.select(
            "Date", "ValueCode", "QuoteCode"
        )
        changed_keys = changed.spot_bid_broad.select(
            "Date", "ValueCode", "QuoteCode"
        )
        self.assertTrue(original_keys.equals(changed_keys))

    def test_q50_q80_gate_mismatch_fails_closed(self) -> None:
        liquidity = _liquidity().with_columns(
            pl.when(pl.col("boundary_quantile") == 80)
            .then(pl.lit(False))
            .otherwise(pl.col("hard_data_gate"))
            .alias("hard_data_gate"),
            pl.when(pl.col("boundary_quantile") == 80)
            .then(pl.lit(False))
            .otherwise(pl.col("pre_replay_candidate"))
            .alias("pre_replay_candidate"),
        )
        with self.assertRaisesRegex(ValueError, "q50/q80 liquidity gate mismatch"):
            build_foundation_cohorts(_boundaries(), liquidity)

    def test_nonmonotone_q_and_same_day_source_fail_closed(self) -> None:
        nonmonotone = _boundaries().with_columns(
            pl.when(pl.col("boundary_quantile") == 95)
            .then(pl.lit(1.0))
            .otherwise(pl.col("upper_distance_bp"))
            .alias("upper_distance_bp")
        )
        with self.assertRaisesRegex(ValueError, "not monotone"):
            build_foundation_cohorts(nonmonotone, _liquidity())

        with self.assertRaisesRegex(ValueError, "strictly prior"):
            build_foundation_cohorts(
                _boundaries(source_asof_date="20260505"),
                _liquidity(),
            )

    def test_selector_columns_are_rejected(self) -> None:
        selector_contaminated = _liquidity().with_columns(
            pl.lit(True).alias("monthly_primary_selected")
        )
        with self.assertRaisesRegex(ValueError, "forbidden selector fields"):
            build_foundation_cohorts(_boundaries(), selector_contaminated)


if __name__ == "__main__":
    unittest.main()
