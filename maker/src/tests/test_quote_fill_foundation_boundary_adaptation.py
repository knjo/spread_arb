"""Focused contracts for causal boundary adaptation."""

from __future__ import annotations

import unittest
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta

import polars as pl

from maker.src.quote_fill.foundation_boundary_adaptation import (
    TOD_BUCKETS,
    apply_boundary_multipliers,
    fit_side_specific_multipliers,
    fit_tod_bucket_multipliers,
    score_boundary_predictions,
)

QUANTILE_BOUNDARIES = {50: 10.5, 80: 16.5, 95: 19.5}


def _previous_day(date_text: str) -> str:
    value = datetime.strptime(date_text, "%Y%m%d")  # noqa: DTZ007
    return (value - timedelta(days=1)).strftime("%Y%m%d")


def _prediction_rows(
    dates: Sequence[str],
    *,
    boundary: Callable[[str, int], float] | None = None,
    source_asof: str | None = None,
    anchor_model_id: str = "time_ewma_30s",
    candidate_id: str = "Q1_trail60_date_equal",
) -> pl.DataFrame:
    boundary_function = boundary or (
        lambda _side, quantile: QUANTILE_BOUNDARIES[quantile]
    )
    rows: list[dict[str, object]] = []
    for date_text in dates:
        for tod_bucket, _, _ in TOD_BUCKETS:
            for side in ("positive", "negative"):
                for quantile in (50, 80, 95):
                    row_source = source_asof or _previous_day(date_text)
                    rows.append(
                        {
                            "Date": date_text,
                            "ValueCode": "1111",
                            "QuoteCode": "QFUT",
                            "anchor_model_id": anchor_model_id,
                            "candidate_id": candidate_id,
                            "tod_bucket": tod_bucket,
                            "boundary_quantile": quantile,
                            "side": side,
                            "boundary_distance_bp": boundary_function(
                                side,
                                quantile,
                            ),
                            "source_asof_date": row_source,
                            "history_start_date": "20260401",
                            "history_end_date": row_source,
                            "candidate_kind": "date_equal",
                            "native_supported": True,
                            "contains_target_day_outcome": False,
                        }
                    )
    return pl.from_dicts(rows, infer_schema_length=None)


def _episode_row(
    date_text: str,
    *,
    number: int,
    side: str,
    amplitude: float,
    start_second: int,
    anchor_model_id: str = "time_ewma_30s",
    completed: bool = True,
    left_censored: bool = False,
) -> dict[str, object]:
    return {
        "episode_id": f"{date_text}/{side}/{start_second}/{number}",
        "Date": date_text,
        "ValueCode": "1111",
        "QuoteCode": "QFUT",
        "anchor_model_id": anchor_model_id,
        "side": side,
        "start_seconds_from_open": start_second,
        "observed_amplitude_bp": amplitude,
        "left_censored": left_censored,
        "right_censored": not completed,
        "completed_center_return": completed,
    }


def _complete_episode_panel(
    dates: Sequence[str],
    *,
    amplitude_factor: Callable[[str, str], float] | None = None,
) -> pl.DataFrame:
    factor_function = amplitude_factor or (lambda _tod, _side: 1.0)
    rows: list[dict[str, object]] = []
    for date_text in dates:
        for tod_bucket, start, _ in TOD_BUCKETS:
            for side in ("positive", "negative"):
                factor = factor_function(tod_bucket, side)
                for number in range(1, 21):
                    rows.append(
                        _episode_row(
                            date_text,
                            number=number,
                            side=side,
                            amplitude=number * factor,
                            start_second=start + number,
                        )
                    )
    return pl.from_dicts(rows, infer_schema_length=None)


class BoundaryScoreTests(unittest.TestCase):
    def test_score_keeps_right_censored_below_boundary_unknown(self) -> None:
        predictions = _prediction_rows(
            ["20260505"],
            boundary=lambda _side, quantile: {50: 5.0, 80: 8.0, 95: 12.0}[quantile],
        )
        episodes = pl.from_dicts(
            [
                _episode_row(
                    "20260505",
                    number=1,
                    side="positive",
                    amplitude=10.0,
                    start_second=301,
                ),
                _episode_row(
                    "20260505",
                    number=2,
                    side="positive",
                    amplitude=4.0,
                    start_second=302,
                    completed=False,
                ),
                _episode_row(
                    "20260505",
                    number=3,
                    side="positive",
                    amplitude=100.0,
                    start_second=303,
                    left_censored=True,
                ),
                _episode_row(
                    "20260505",
                    number=4,
                    side="negative",
                    amplitude=6.0,
                    start_second=304,
                ),
            ],
            infer_schema_length=None,
        )

        scored = score_boundary_predictions(predictions, episodes)
        q50 = scored.filter(
            (pl.col("tod_bucket") == "0905_1000")
            & (pl.col("side") == "positive")
            & (pl.col("boundary_quantile") == 50)
        ).row(0, named=True)
        self.assertEqual(q50["observable_started"], 2)
        self.assertEqual(q50["confirmed_hits"], 1)
        self.assertEqual(q50["known_nonhits"], 0)
        self.assertEqual(q50["unknown_nonhits"], 1)
        self.assertEqual(q50["left_censored_count"], 1)
        self.assertAlmostEqual(q50["reach_lower_bound"], 0.5)
        self.assertAlmostEqual(q50["reach_upper_bound"], 1.0)
        self.assertAlmostEqual(q50["realized_amplitude_lower_bp"], 4.0)
        self.assertAlmostEqual(q50["realized_amplitude_upper_bp"], 10.0)
        self.assertAlmostEqual(q50["amplitude_interval_distance_bp"], 0.0)
        self.assertTrue(q50["score_contains_same_day_outcome"])

        q80 = scored.filter(
            (pl.col("tod_bucket") == "0905_1000")
            & (pl.col("side") == "positive")
            & (pl.col("boundary_quantile") == 80)
        ).row(0, named=True)
        self.assertTrue(q80["realized_amplitude_upper_unbounded"])
        self.assertIsNone(q80["realized_amplitude_upper_bp"])
        self.assertAlmostEqual(q80["amplitude_interval_distance_bp"], 2.0)

    def test_episode_tod_is_derived_from_start_and_frozen(self) -> None:
        predictions = _prediction_rows(["20260505"])
        episodes = _complete_episode_panel(["20260505"]).with_columns(
            pl.when(pl.col("start_seconds_from_open") < 3_600)
            .then(pl.lit("1000_1100"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("tod_bucket")
        )
        with self.assertRaisesRegex(ValueError, "TOD disagrees"):
            score_boundary_predictions(predictions, episodes)

    def test_predictions_fail_on_incomplete_duplicate_or_unsafe_cells(self) -> None:
        predictions = _prediction_rows(["20260505"])
        episodes = _complete_episode_panel(["20260505"])
        incomplete = predictions.filter(
            ~(
                (pl.col("tod_bucket") == "0905_1000")
                & (pl.col("side") == "positive")
                & (pl.col("boundary_quantile") == 95)
            )
        )
        with self.assertRaisesRegex(ValueError, "incomplete q"):
            score_boundary_predictions(incomplete, episodes)

        duplicated = pl.concat([predictions, predictions.head(1)])
        with self.assertRaisesRegex(ValueError, "duplicate long"):
            score_boundary_predictions(duplicated, episodes)

        unsafe = predictions.with_columns(pl.col("Date").alias("source_asof_date"))
        with self.assertRaisesRegex(ValueError, "strictly before"):
            score_boundary_predictions(unsafe, episodes)

    def test_anchor_identity_prevents_cross_anchor_episode_collisions(self) -> None:
        first_predictions = _prediction_rows(["20260505"])
        second_predictions = _prediction_rows(
            ["20260505"],
            anchor_model_id="time_ewma_60s",
        )
        first_episodes = _complete_episode_panel(["20260505"])
        second_episodes = first_episodes.with_columns(
            pl.lit("time_ewma_60s").alias("anchor_model_id")
        )

        scored = score_boundary_predictions(
            pl.concat([first_predictions, second_predictions]),
            pl.concat([first_episodes, second_episodes]),
        )
        self.assertEqual(scored["anchor_model_id"].n_unique(), 2)
        self.assertEqual(scored.height, first_predictions.height * 2)

    def test_fallback_requires_explicit_effective_lineage(self) -> None:
        predictions = _prediction_rows(["20260505"]).with_columns(
            pl.lit(False).alias("native_supported"),
            pl.lit(True).alias("effective_supported"),
        )
        episodes = _complete_episode_panel(["20260505"])
        with self.assertRaisesRegex(ValueError, "effective fallback"):
            score_boundary_predictions(predictions, episodes)

        explicit = predictions.with_columns(
            pl.lit("Q1_trail60_date_equal").alias("effective_candidate_id"),
            pl.col("source_asof_date").alias("effective_source_asof_date"),
        )
        scored = score_boundary_predictions(explicit, episodes)
        self.assertTrue(scored["effective_supported"].all())
        self.assertTrue((~scored["native_supported"]).all())


class MultiplierFitTests(unittest.TestCase):
    def test_global_fit_is_date_equal_q_equal_and_side_specific(self) -> None:
        dates = ["20260505", "20260506"]
        predictions = _prediction_rows(dates)
        episodes = _complete_episode_panel(
            dates,
            amplitude_factor=lambda _tod, side: 0.5 if side == "negative" else 1.0,
        )

        multipliers = fit_side_specific_multipliers(
            predictions,
            episodes,
            target_date="20260507",
            calibration_dates=dates,
        )
        self.assertEqual(multipliers.height, 2)
        by_side = {
            row["side"]: row["multiplier"] for row in multipliers.iter_rows(named=True)
        }
        self.assertAlmostEqual(by_side["positive"], 1.0)
        self.assertAlmostEqual(by_side["negative"], 0.51)
        self.assertEqual(
            multipliers["source_asof_date"].unique().to_list(),
            ["20260506"],
        )
        self.assertTrue(
            (multipliers["objective_reach_interval_distance"].abs() < 1e-12).all()
        )
        self.assertTrue((~multipliers["contains_target_day_outcome"]).all())

    def test_global_fit_rejects_window_mismatch_and_target_date(self) -> None:
        dates = ["20260505", "20260506"]
        predictions = _prediction_rows(dates)
        episodes = _complete_episode_panel(dates)
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            fit_side_specific_multipliers(
                predictions,
                episodes,
                target_date="20260507",
                calibration_dates=["20260505"],
            )
        with self.assertRaisesRegex(ValueError, "strictly before"):
            fit_side_specific_multipliers(
                predictions,
                episodes,
                target_date="20260506",
                calibration_dates=dates,
            )

    def test_tod_fit_uses_exactly_prior_ten_and_separate_buckets(self) -> None:
        first = date(2026, 5, 1)
        dates = [
            (first + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(10)
        ]
        predictions = _prediction_rows(dates)
        episodes = _complete_episode_panel(
            dates,
            amplitude_factor=lambda tod, _side: 0.5 if tod == "1000_1100" else 1.0,
        )

        multipliers = fit_tod_bucket_multipliers(
            predictions,
            episodes,
            target_date="20260511",
            calibration_dates=dates,
        )
        self.assertEqual(multipliers.height, 8)
        morning = multipliers.filter(
            (pl.col("side") == "positive") & (pl.col("tod_bucket") == "0905_1000")
        )["multiplier"].item()
        second = multipliers.filter(
            (pl.col("side") == "positive") & (pl.col("tod_bucket") == "1000_1100")
        )["multiplier"].item()
        self.assertAlmostEqual(morning, 1.0)
        # Several grid points reproduce the same empirical hit counts.  The
        # frozen distance-to-one tie-break selects the largest such point.
        self.assertAlmostEqual(second, 0.51)

        with self.assertRaisesRegex(ValueError, "exactly ten"):
            fit_tod_bucket_multipliers(
                _prediction_rows(dates[:9]),
                _complete_episode_panel(dates[:9]),
                target_date="20260511",
                calibration_dates=dates[:9],
            )


class MultiplierApplicationTests(unittest.TestCase):
    def test_apply_scales_and_propagates_latest_asof(self) -> None:
        calibration_dates = ["20260505", "20260506"]
        fitted = fit_side_specific_multipliers(
            _prediction_rows(calibration_dates),
            _complete_episode_panel(calibration_dates),
            target_date="20260507",
            calibration_dates=calibration_dates,
        ).with_columns(pl.lit(1.20).alias("multiplier"))
        target = _prediction_rows(
            ["20260507"],
            boundary=lambda _side, quantile: {50: 10.0, 80: 20.0, 95: 30.0}[quantile],
            source_asof="20260505",
        )

        scaled = apply_boundary_multipliers(
            target,
            fitted,
            output_candidate_id="Q3_shape60_level5",
            output_candidate_kind="global_level_scale",
        )
        self.assertEqual(
            scaled["candidate_id"].unique().to_list(),
            ["Q3_shape60_level5"],
        )
        self.assertEqual(
            scaled["source_asof_date"].unique().to_list(),
            ["20260506"],
        )
        self.assertEqual(
            scaled["history_end_date"].unique().to_list(),
            ["20260506"],
        )
        q50 = (
            scaled.filter(pl.col("boundary_quantile") == 50)["boundary_distance_bp"]
            .head(1)
            .item()
        )
        self.assertAlmostEqual(q50, 12.0)
        self.assertTrue((~scaled["contains_target_day_outcome"]).all())

        with self.assertRaisesRegex(ValueError, "incomplete side/TOD"):
            apply_boundary_multipliers(
                target,
                fitted.filter(pl.col("side") == "positive"),
                output_candidate_id="Q3_shape60_level5",
            )


if __name__ == "__main__":
    unittest.main()
