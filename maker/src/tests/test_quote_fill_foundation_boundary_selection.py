"""Contracts for selected-anchor boundary candidate estimation."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from maker.src.quote_fill.foundation_boundary import (
    extract_censor_aware_excursions,
)
from maker.src.quote_fill.foundation_boundary_selection import (
    ANALYSIS_START_SECOND,
    ENTRY_STOP_SECOND,
    annotate_expiry_dte,
    assign_episode_start_tod_bucket,
    build_boundary_candidate_predictions,
    censor_identified_quantile_interval,
    clip_completed_quantile_to_interval,
    date_equal_completed_quantile,
    event_pooled_completed_quantile,
    extract_entry_window_excursions,
    select_prior_expiry_dte_history,
    select_trailing_session_history,
    validate_monotonic_quantiles,
    validate_strict_source_asof,
)


def _entry_panel(
    observations: list[tuple[int, bool, float | None]],
) -> pl.DataFrame:
    start = datetime(2026, 8, 13, 9)  # noqa: DTZ001
    rows: list[dict[str, object]] = []
    for second, eligible, residual in observations:
        anchor = 100.0
        rows.append(
            {
                "Date": "20260813",
                "ValueCode": "1111",
                "QuoteCode": "QAUG",
                "timestamp": start + timedelta(seconds=second),
                "seconds_from_open": second,
                "basis_mid_bp": (None if residual is None else anchor + residual),
                "anchor_test_bp": anchor,
                "analysis_eligible": eligible,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _oracle_entry_window_excursions(panel: pl.DataFrame) -> pl.DataFrame:
    entry_panel = panel.with_columns(
        (
            pl.col("analysis_eligible").fill_null(False).cast(pl.Boolean)
            & (pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
            & (pl.col("seconds_from_open") < ENTRY_STOP_SECOND)
        ).alias("analysis_eligible")
    )
    episodes = extract_censor_aware_excursions(
        entry_panel,
        anchor_column="anchor_test_bp",
    ).with_columns(
        pl.lit("test").alias("anchor_model_id"),
        pl.when(
            pl.col("right_censored")
            & (pl.col("end_seconds_from_open") == ENTRY_STOP_SECOND - 1)
        )
        .then(pl.lit("entry_stop"))
        .otherwise(pl.col("right_censor_reason"))
        .alias("right_censor_reason"),
    )
    return assign_episode_start_tod_bucket(episodes, strict=True)


def _episode(
    date_text: str,
    amplitude: float,
    *,
    value_code: str = "1111",
    quote_code: str = "Q1",
    side: str = "positive",
    completed: bool = True,
    left_censored: bool = False,
) -> dict[str, object]:
    return {
        "Date": date_text,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "side": side,
        "observed_amplitude_bp": amplitude,
        "left_censored": left_censored,
        "right_censored": not completed,
        "completed_center_return": completed,
    }


class CompletedQuantileTests(unittest.TestCase):
    def test_event_pooled_and_date_equal_are_materially_different(self) -> None:
        rows = [_episode("20260501", 1.0) for _ in range(100)]
        rows.append(_episode("20260502", 100.0))
        episodes = pl.from_dicts(rows, infer_schema_length=None)

        self.assertAlmostEqual(
            event_pooled_completed_quantile(episodes, 0.80),
            1.0,
        )
        self.assertAlmostEqual(
            date_equal_completed_quantile(episodes, 0.80),
            100.0,
        )

    def test_censor_interval_and_clipped_point_are_identified(self) -> None:
        episodes = pl.from_dicts(
            [
                _episode("20260501", 2.0, completed=True),
                _episode("20260501", 5.0, completed=False),
            ],
            infer_schema_length=None,
        )

        q50 = censor_identified_quantile_interval(episodes, 0.50)
        self.assertAlmostEqual(q50.lower_bp, 2.0)
        self.assertAlmostEqual(q50.upper_bp, 2.0)
        self.assertFalse(q50.upper_unbounded)

        q80 = censor_identified_quantile_interval(episodes, 0.80)
        self.assertAlmostEqual(q80.lower_bp, 5.0)
        self.assertIsNone(q80.upper_bp)
        self.assertTrue(q80.upper_unbounded)
        self.assertEqual(q80.observable_started, 2)
        self.assertEqual(q80.completed_count, 1)
        self.assertEqual(q80.right_censored_count, 1)
        self.assertAlmostEqual(
            clip_completed_quantile_to_interval(2.0, q80),
            5.0,
        )


class MappingAndHistorySelectorTests(unittest.TestCase):
    def test_prior_expiry_selector_excludes_target_cycle(self) -> None:
        history = pl.from_dicts(
            [
                _episode("20260804", 9.0, quote_code="QAUG"),
                _episode("20260701", 7.0, quote_code="QJUL"),
                _episode("20260603", 6.0, quote_code="QJUN"),
            ],
            infer_schema_length=None,
        )
        mapping = pl.DataFrame(
            {
                "Date": ["20260804", "20260701", "20260603"],
                "ValueCode": ["1111", "1111", "1111"],
                "QuoteCode": ["QAUG", "QJUL", "QJUN"],
                "end_date": [
                    date(2026, 8, 19),
                    date(2026, 7, 15),
                    date(2026, 6, 17),
                ],
            }
        )
        annotated = annotate_expiry_dte(history, mapping)

        previous_one = select_prior_expiry_dte_history(
            annotated,
            target_date="20260805",
            value_code="1111",
            target_expiry_date=date(2026, 8, 19),
            target_calendar_dte=14,
            prior_expiry_cycles=1,
            dte_radius_calendar_days=5,
        )
        self.assertEqual(previous_one.selected_expiry_cycles, ("20260715",))
        self.assertEqual(previous_one.frame["Date"].to_list(), ["20260701"])
        self.assertNotIn("20260804", previous_one.frame["Date"].to_list())

        previous_two = select_prior_expiry_dte_history(
            annotated,
            target_date="20260805",
            value_code="1111",
            target_expiry_date="20260819",
            target_calendar_dte=14,
            prior_expiry_cycles=2,
            dte_radius_calendar_days=5,
        )
        self.assertEqual(
            previous_two.selected_expiry_cycles,
            ("20260715", "20260617"),
        )
        self.assertEqual(
            set(previous_two.frame["Date"].to_list()),
            {"20260701", "20260603"},
        )

    def test_mapping_requires_the_exact_quote_code(self) -> None:
        history = pl.from_dicts(
            [_episode("20260701", 7.0, quote_code="QJUL")],
            infer_schema_length=None,
        )
        wrong_mapping = pl.DataFrame(
            {
                "Date": ["20260701"],
                "ValueCode": ["1111"],
                "QuoteCode": ["OTHER"],
                "end_date": [date(2026, 7, 15)],
            }
        )
        with self.assertRaisesRegex(ValueError, "exact.*mapping"):
            annotate_expiry_dte(history, wrong_mapping)

    def test_target_day_history_is_rejected_not_silently_filtered(self) -> None:
        history = pl.from_dicts(
            [
                _episode("20260504", 2.0),
                _episode("20260505", 999.0),
            ],
            infer_schema_length=None,
        )
        with self.assertRaisesRegex(ValueError, "target-day/future"):
            select_trailing_session_history(
                history,
                target_date="20260505",
                lookback_sessions=20,
            )


class TodAndPredictionContractTests(unittest.TestCase):
    def test_vectorized_entry_extractor_matches_generic_oracle(self) -> None:
        fixtures = {
            "complete": [
                (300, True, 0.0),
                (301, True, 2.0),
                (302, True, 4.0),
                (303, True, 0.0),
                (304, True, -3.0),
                (305, True, 0.0),
            ],
            "eligibility_gap": [
                (300, True, 0.0),
                (301, True, 2.0),
                (302, False, None),
                (303, True, 3.0),
                (304, True, 0.0),
            ],
            "zero_runs": [
                (300, True, 0.0),
                (301, True, 0.0),
                (302, True, 3.0),
                (303, True, 0.0),
                (304, True, 0.0),
                (305, True, -2.0),
                (306, True, 0.0),
            ],
            "all_zero": [
                (300, True, 0.0),
                (301, True, 0.0),
                (302, True, 0.0),
            ],
            "direct_sign_crossing": [
                (300, True, 0.0),
                (301, True, 3.0),
                (302, True, -4.0),
                (303, True, 5.0),
                (304, True, 0.0),
            ],
            "timestamp_gap": [
                (300, True, 0.0),
                (301, True, -2.0),
                (304, True, -3.0),
                (305, True, 0.0),
            ],
            "entry_stop": [
                (14_398, True, 2.0),
                (14_399, True, 3.0),
                (14_400, True, 100.0),
            ],
        }
        for name, observations in fixtures.items():
            with self.subTest(name=name):
                panel = _entry_panel(observations).reverse()
                expected = _oracle_entry_window_excursions(panel)
                actual = extract_entry_window_excursions(
                    panel,
                    anchor_column="anchor_test_bp",
                    anchor_model_id="test",
                )
                assert_frame_equal(
                    actual,
                    expected,
                    check_row_order=True,
                    check_column_order=True,
                    check_dtypes=True,
                )

    def test_entry_window_censors_before_post_1300_extreme(self) -> None:
        seconds = list(range(15_600))
        basis = [0.0] * len(seconds)
        basis[14_398] = 2.0
        basis[14_399] = 3.0
        basis[14_400] = 100.0
        basis[14_401] = 100.0
        start = datetime(2026, 8, 13, 9, 0)  # noqa: DTZ001
        day = pl.DataFrame(
            {
                "Date": ["20260813"] * len(seconds),
                "ValueCode": ["1111"] * len(seconds),
                "QuoteCode": ["QAUG"] * len(seconds),
                "timestamp": [start + timedelta(seconds=value) for value in seconds],
                "seconds_from_open": seconds,
                "basis_mid_bp": basis,
                "anchor_test_bp": [0.0] * len(seconds),
                "analysis_eligible": [value >= 300 for value in seconds],
            }
        )
        episodes = extract_entry_window_excursions(
            day,
            anchor_column="anchor_test_bp",
            anchor_model_id="test",
        )
        final = episodes.filter(pl.col("start_seconds_from_open") == 14_398)
        self.assertEqual(final.height, 1)
        self.assertAlmostEqual(final.item(0, "observed_amplitude_bp"), 3.0)
        self.assertTrue(final.item(0, "right_censored"))
        self.assertEqual(final.item(0, "right_censor_reason"), "entry_stop")
        self.assertEqual(final.item(0, "anchor_model_id"), "test")

    def test_episode_start_bucket_uses_half_open_boundaries(self) -> None:
        episodes = pl.DataFrame(
            {
                "start_seconds_from_open": [
                    300,
                    3_599,
                    3_600,
                    7_199,
                    7_200,
                    10_799,
                    10_800,
                    14_399,
                ]
            }
        )
        assigned = assign_episode_start_tod_bucket(episodes)
        self.assertEqual(
            assigned["tod_bucket"].to_list(),
            [
                "0905_1000",
                "0905_1000",
                "1000_1100",
                "1000_1100",
                "1100_1200",
                "1100_1200",
                "1200_1300",
                "1200_1300",
            ],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            assign_episode_start_tod_bucket(
                pl.DataFrame({"start_seconds_from_open": [14_400]})
            )

    def test_q2_builder_emits_safe_long_lineage(self) -> None:
        first = date(2026, 5, 5)
        rows: list[dict[str, object]] = []
        for offset in range(15):
            date_text = (first + timedelta(days=offset)).strftime("%Y%m%d")
            rows.append(
                _episode(
                    date_text,
                    float(offset + 1),
                    side="positive",
                )
            )
            rows.append(
                _episode(
                    date_text,
                    float(offset + 2),
                    side="negative",
                )
            )
        history = pl.from_dicts(rows, infer_schema_length=None)
        targets = pl.DataFrame(
            {
                "Date": ["20260520"],
                "ValueCode": ["1111"],
                "QuoteCode": ["QTARGET"],
            }
        )

        predictions = build_boundary_candidate_predictions(
            history,
            targets,
            candidate_id="Q2_trail20_date_equal",
            minimum_completed_per_side={50: 1, 80: 1, 95: 1},
        )

        self.assertEqual(predictions.height, 4 * 2 * 3)
        self.assertTrue(predictions["native_supported"].all())
        self.assertEqual(
            predictions["source_asof_date"].unique().to_list(),
            ["20260519"],
        )
        self.assertEqual(
            predictions["history_start_date"].unique().to_list(),
            ["20260505"],
        )
        self.assertEqual(
            predictions["completed_history_dates"].unique().to_list(),
            [15],
        )
        self.assertTrue((~predictions["contains_target_day_outcome"]).all())

    def test_q6_builder_uses_two_prior_cycles_and_target_dte(self) -> None:
        rows: list[dict[str, object]] = []
        mappings: list[dict[str, object]] = []
        for expiry, quote_code in (
            (date(2026, 7, 15), "QJUL"),
            (date(2026, 6, 17), "QJUN"),
        ):
            for dte in (11, 12, 13, 14):
                date_text = (expiry - timedelta(days=dte)).strftime("%Y%m%d")
                for side in ("positive", "negative"):
                    rows.append(
                        _episode(
                            date_text,
                            float(dte),
                            quote_code=quote_code,
                            side=side,
                        )
                    )
                mappings.append(
                    {
                        "Date": date_text,
                        "ValueCode": "1111",
                        "QuoteCode": quote_code,
                        "end_date": expiry,
                    }
                )
        history = annotate_expiry_dte(
            pl.from_dicts(rows, infer_schema_length=None),
            pl.from_dicts(mappings, infer_schema_length=None),
        )
        target = annotate_expiry_dte(
            pl.DataFrame(
                {
                    "Date": ["20260805"],
                    "ValueCode": ["1111"],
                    "QuoteCode": ["QAUG"],
                }
            ),
            pl.DataFrame(
                {
                    "Date": ["20260805"],
                    "ValueCode": ["1111"],
                    "QuoteCode": ["QAUG"],
                    "end_date": [date(2026, 8, 19)],
                }
            ),
        )

        predictions = build_boundary_candidate_predictions(
            history,
            target,
            candidate_id="Q6_prev2_expiry_dte5",
            minimum_completed_per_side={50: 1, 80: 1, 95: 1},
        )

        self.assertEqual(predictions.height, 4 * 2 * 3)
        self.assertTrue(predictions["native_supported"].all())
        self.assertEqual(
            predictions["selected_expiry_cycles"].unique().to_list(),
            ["20260715,20260617"],
        )
        self.assertEqual(
            predictions["target_calendar_dte"].unique().to_list(),
            [14],
        )

    def test_lineage_and_monotonicity_validators_fail_closed(self) -> None:
        leakage = pl.DataFrame(
            {
                "Date": ["20260505"],
                "source_asof_date": ["20260505"],
                "history_end_date": ["20260505"],
                "native_supported": [True],
                "contains_target_day_outcome": [False],
            }
        )
        with self.assertRaisesRegex(ValueError, "source_asof"):
            validate_strict_source_asof(leakage)

        inverted = pl.DataFrame(
            {
                "Date": ["20260505", "20260505", "20260505"],
                "ValueCode": ["1111"] * 3,
                "QuoteCode": ["Q1"] * 3,
                "tod_bucket": ["0905_1000"] * 3,
                "side": ["positive"] * 3,
                "candidate_id": ["test"] * 3,
                "boundary_quantile": [50, 80, 95],
                "boundary_distance_bp": [5.0, 4.0, 8.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "not monotone"):
            validate_monotonic_quantiles(inverted)


if __name__ == "__main__":
    unittest.main()
