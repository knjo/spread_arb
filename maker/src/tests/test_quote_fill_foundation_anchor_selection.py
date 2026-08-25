"""Tests for the pure frozen anchor-selection layer."""

from __future__ import annotations

import math
import unittest
from datetime import date, datetime, timedelta

import polars as pl

from maker.src.quote_fill.foundation_anchor_selection import (
    FUTURE_HORIZONS,
    LABEL_COLUMNS,
    MODEL_COLUMNS,
    build_anchor_selection_grid,
    evaluate_anchor_selection_day,
    materialize_anchor_column,
)


def _complete_day(
    *,
    basis: list[float] | None = None,
    eligible: list[bool] | None = None,
    session_date: str = "20260813",
) -> pl.DataFrame:
    seconds = list(range(15_600))
    values = basis if basis is not None else [float(value) for value in seconds]
    gate = eligible if eligible is not None else [True] * len(seconds)
    start = datetime(2026, 8, 13, 9, 0)  # noqa: DTZ001
    return pl.DataFrame(
        {
            "Date": [session_date] * len(seconds),
            "ValueCode": ["2303"] * len(seconds),
            "QuoteCode": ["CCFH6"] * len(seconds),
            "timestamp": [start + timedelta(seconds=value) for value in seconds],
            "seconds_from_open": seconds,
            "end_date": [date(2026, 8, 19)] * len(seconds),
            "basis_mid_bp": values,
            "eligible_base": gate,
            "analysis_eligible": [
                is_eligible and second >= 300
                for second, is_eligible in zip(seconds, gate, strict=True)
            ],
        }
    )


class FoundationAnchorSelectionTest(unittest.TestCase):
    def test_requires_the_exact_complete_product_second_grid(self) -> None:
        incomplete = _complete_day().filter(pl.col("seconds_from_open") != 777)
        with self.assertRaisesRegex(ValueError, "complete 0..15599"):
            build_anchor_selection_grid(incomplete)

    def test_eligibility_gap_separates_wall_clock_and_count_ewma(self) -> None:
        basis = [0.0] * 15_600
        gate = [False] * 15_600
        gate[0] = True
        gate[300] = True
        basis[300] = 10.0
        grid = build_anchor_selection_grid(_complete_day(basis=basis, eligible=gate))
        row = grid.filter(pl.col("seconds_from_open") == 300)

        wall = row.item(0, MODEL_COLUMNS["time_ewma_30s"])
        count = row.item(0, MODEL_COLUMNS["count_ewma_30obs"])
        self.assertAlmostEqual(wall, 10.0 * (1.0 - 0.5 ** (300 / 30)))
        self.assertAlmostEqual(count, 10.0 * (1.0 - 0.5 ** (1 / 30)))
        self.assertGreater(wall - count, 9.0)

    def test_future_mutation_changes_labels_but_not_any_anchor(self) -> None:
        original_basis = [0.0] * 15_600
        mutated_basis = list(original_basis)
        for second in range(501, 15_600):
            mutated_basis[second] = 100.0

        original = build_anchor_selection_grid(
            _complete_day(basis=original_basis)
        ).filter(pl.col("seconds_from_open") == 500)
        mutated = build_anchor_selection_grid(
            _complete_day(basis=mutated_basis)
        ).filter(pl.col("seconds_from_open") == 500)

        for column in MODEL_COLUMNS.values():
            self.assertAlmostEqual(
                original.item(0, column),
                mutated.item(0, column),
                msg=column,
            )
        primary_label = LABEL_COLUMNS["future_median_30_300s"]
        self.assertAlmostEqual(original.item(0, primary_label), 0.0)
        self.assertAlmostEqual(mutated.item(0, primary_label), 100.0)

    def test_all_three_inclusive_future_medians_are_frozen(self) -> None:
        grid = build_anchor_selection_grid(_complete_day()).filter(
            pl.col("seconds_from_open") == 300
        )
        expected = {
            "future_median_10_60s": 335.0,
            "future_median_30_300s": 465.0,
            "future_median_300_900s": 900.0,
        }
        for horizon in FUTURE_HORIZONS:
            self.assertAlmostEqual(
                grid.item(0, LABEL_COLUMNS[horizon.horizon_id]),
                expected[horizon.horizon_id],
            )
            self.assertAlmostEqual(
                grid.item(0, f"label_{horizon.horizon_id}_coverage"),
                1.0,
            )

    def test_future_label_fails_when_coverage_is_below_ninety_percent(
        self,
    ) -> None:
        gate = [True] * 15_600
        # The primary inclusive window has 271 rows and needs 244.  Removing
        # 28 leaves 243 legal samples and must not create a label.
        for second in range(330, 358):
            gate[second] = False
        row = build_anchor_selection_grid(_complete_day(eligible=gate)).filter(
            pl.col("seconds_from_open") == 300
        )
        self.assertEqual(row.item(0, "label_future_median_30_300s_sample_count"), 243)
        self.assertIsNone(row.item(0, LABEL_COLUMNS["future_median_30_300s"]))

    def test_product_day_stats_use_one_common_support_and_keep_strata(
        self,
    ) -> None:
        result = evaluate_anchor_selection_day(_complete_day())
        self.assertEqual(result.product_day_stats.height, 150)
        self.assertEqual(result.support_stats.height, 15)
        self.assertEqual(
            set(result.product_day_stats["tod_bucket"].unique()),
            {"all", "0905_1000", "1000_1100", "1100_1200", "1200_1300"},
        )
        self.assertEqual(
            result.product_day_stats["month"].unique().to_list(), ["202608"]
        )
        self.assertEqual(
            result.product_day_stats["calendar_dte"].unique().to_list(), [6]
        )
        self.assertEqual(
            result.product_day_stats["dte_bucket"].unique().to_list(),
            ["06-10"],
        )

        primary_overall = result.product_day_stats.filter(
            (pl.col("horizon_id") == "future_median_30_300s")
            & (pl.col("tod_bucket") == "all")
        )
        self.assertEqual(primary_overall["n_common_evaluable"].n_unique(), 1)
        self.assertEqual(primary_overall.item(0, "n_common_evaluable"), 14_100)
        persistence = primary_overall.filter(pl.col("model") == "persistence")
        self.assertTrue(
            math.isclose(
                persistence.item(0, "mae_bp"),
                165.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    def test_materialization_preserves_runner_columns(self) -> None:
        day = _complete_day()
        materialized = materialize_anchor_column(day, "time_ewma_30s")
        self.assertTrue(materialized["timestamp"].equals(day["timestamp"]))
        self.assertTrue(
            materialized["analysis_eligible"].equals(day["analysis_eligible"])
        )
        self.assertEqual(
            materialized["selected_anchor_model"].unique().to_list(),
            ["time_ewma_30s"],
        )
        difference = (
            materialized["selected_anchor_bp"]
            - materialized[MODEL_COLUMNS["time_ewma_30s"]]
        ).abs()
        self.assertLess(float(difference.max()), 1e-12)
        self.assertFalse(
            any(column.startswith("label_future") for column in materialized.columns)
        )

    def test_frozen_model_can_materialize_after_development_cutoff(self) -> None:
        forward = _complete_day(session_date="20260814")
        materialized = materialize_anchor_column(forward, "time_ewma_30s")
        self.assertEqual(materialized["Date"].unique().to_list(), ["20260814"])
        with self.assertRaisesRegex(ValueError, "locked development cutoff"):
            build_anchor_selection_grid(forward)


if __name__ == "__main__":
    unittest.main()
