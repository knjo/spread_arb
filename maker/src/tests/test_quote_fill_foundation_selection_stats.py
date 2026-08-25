"""Focused contracts for foundation selection statistics."""

from __future__ import annotations

import unittest
from itertools import pairwise

import polars as pl

from maker.src.quote_fill.foundation_selection_stats import (
    censor_interval_distance,
    censor_interval_signed_bias,
    common_support_units,
    daily_cross_sectional_spearman,
    filter_common_support,
    moving_whole_date_block_indices,
    paired_moving_whole_date_block_bootstrap,
    rank_q_candidates,
    select_anchor_top_two,
    temporal_product_spearman,
    with_censor_interval_loss,
)


class MovingDateBlockBootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self) -> None:
        daily = pl.DataFrame(
            {
                "Date": [f"202605{day:02d}" for day in range(1, 11)] * 2,
                "model": ["a"] * 10 + ["b"] * 10,
                "mae_bp": [float(day) for day in range(10)]
                + [float(day + (day % 3)) for day in range(10)],
            }
        )
        first = paired_moving_whole_date_block_bootstrap(
            daily,
            reference="a",
            candidates=("a", "b"),
            block_sessions=5,
            replicates=100,
            seed=11,
        )
        second = paired_moving_whole_date_block_bootstrap(
            daily,
            reference="a",
            candidates=("a", "b"),
            block_sessions=5,
            replicates=100,
            seed=11,
        )
        self.assertTrue(first.equals(second, null_equal=True))
        reference = first.filter(pl.col("model") == "a").row(0, named=True)
        self.assertAlmostEqual(reference["observed_delta_mae_bp"], 0.0)
        self.assertAlmostEqual(reference["paired_se_mae_bp"], 0.0)

    def test_each_sample_is_composed_of_serial_session_blocks(self) -> None:
        samples = moving_whole_date_block_indices(
            13,
            block_sessions=5,
            replicates=20,
            seed=29,
        )
        for sample in samples:
            self.assertEqual(len(sample), 13)
            for start in range(0, len(sample), 5):
                block = sample[start : start + 5]
                self.assertTrue(
                    all(right == left + 1 for left, right in pairwise(block))
                )


class AnchorSelectionTests(unittest.TestCase):
    @staticmethod
    def _daily(*, winner_is_smoothest: bool = False) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        dates = [
            "20260501",
            "20260502",
            "20260503",
            "20260601",
            "20260602",
            "20260603",
        ]
        paired_addition = [0.60, -0.50, 0.20, -0.50, 0.60, -0.10]
        for index, date in enumerate(dates):
            month = date[:6]
            rows.extend(
                [
                    {
                        "Date": date,
                        "month": month,
                        "model": "winner",
                        "mae_bp": 1.0,
                        "tv_ratio": 0.1 if winner_is_smoothest else 0.8,
                    },
                    {
                        "Date": date,
                        "month": month,
                        "model": "smooth",
                        "mae_bp": 1.0 + paired_addition[index],
                        "tv_ratio": 0.2,
                    },
                    {
                        "Date": date,
                        "month": month,
                        "model": "next",
                        "mae_bp": 1.2,
                        "tv_ratio": 0.4,
                    },
                    {
                        "Date": date,
                        "month": month,
                        "model": "control",
                        "mae_bp": 0.1,
                        "tv_ratio": 0.01,
                    },
                ]
            )
        return pl.from_dicts(rows, infer_schema_length=None)

    def test_one_se_rule_selects_smoother_actionable_and_excludes_control(self) -> None:
        selected = select_anchor_top_two(
            self._daily(),
            actionable_models=("winner", "smooth", "next", "control"),
            controls=("control",),
            block_sessions=2,
            replicates=500,
            seed=7,
        )
        ranked = selected.filter(pl.col("selection_rank").is_not_null()).sort(
            "selection_rank"
        )
        self.assertEqual(ranked["model"].to_list(), ["winner", "smooth"])
        self.assertEqual(
            ranked.row(1, named=True)["selection_role"],
            "smoothest_within_paired_one_se",
        )
        self.assertNotIn("control", selected["model"].to_list())

    def test_identical_smoothest_falls_back_to_next_lowest_mae(self) -> None:
        selected = select_anchor_top_two(
            self._daily(winner_is_smoothest=True),
            actionable_models=("winner", "smooth", "next"),
            block_sessions=2,
            replicates=200,
            seed=7,
        )
        ranked = selected.filter(pl.col("selection_rank").is_not_null()).sort(
            "selection_rank"
        )
        self.assertEqual(ranked["model"].to_list(), ["winner", "smooth"])
        self.assertEqual(
            ranked.row(1, named=True)["selection_role"],
            "next_lowest_mae_after_identical_smoothest",
        )

    def test_product_day_weights_are_carried_through_month_score(self) -> None:
        rows = []
        for model, losses in (("a", (0.0, 10.0)), ("b", (4.0, 4.0))):
            for date_text, loss, product_days in zip(
                ("20260501", "20260502"),
                losses,
                (9, 1),
                strict=True,
            ):
                rows.append(
                    {
                        "Date": date_text,
                        "month": "202605",
                        "model": model,
                        "mae_bp": loss,
                        "tv_ratio": 0.5,
                        "product_days": product_days,
                    }
                )
        selected = select_anchor_top_two(
            pl.from_dicts(rows),
            actionable_models=("a", "b"),
            weight_column="product_days",
            block_sessions=1,
            replicates=50,
            seed=5,
        )
        score_a = selected.filter(pl.col("model") == "a").row(0, named=True)
        score_b = selected.filter(pl.col("model") == "b").row(0, named=True)
        self.assertAlmostEqual(score_a["month_equal_mae_bp"], 1.0)
        self.assertAlmostEqual(score_b["month_equal_mae_bp"], 4.0)
        self.assertEqual(
            selected.filter(pl.col("selection_rank") == 1).item(0, "model"),
            "a",
        )


class SupportAndIntervalTests(unittest.TestCase):
    def test_common_support_is_exact_candidate_intersection(self) -> None:
        frame = pl.DataFrame(
            {
                "candidate": ["a", "a", "b", "b", "c"],
                "Date": ["d1", "d2", "d1", "d3", "d1"],
                "supported": [True, True, True, True, True],
                "loss": [1.0, 2.0, 3.0, 4.0, 9.0],
            }
        )
        units = common_support_units(
            frame,
            candidates=("a", "b"),
            unit_columns=("Date",),
            supported_column="supported",
            required_value_columns=("loss",),
        )
        self.assertEqual(units["Date"].to_list(), ["d1"])
        common = filter_common_support(
            frame,
            candidates=("a", "b"),
            unit_columns=("Date",),
            supported_column="supported",
            required_value_columns=("loss",),
        )
        self.assertEqual(common["candidate"].to_list(), ["a", "b"])
        self.assertEqual(common["Date"].to_list(), ["d1", "d1"])

    def test_interval_loss_has_direction_and_zero_inside_identified_set(self) -> None:
        self.assertAlmostEqual(censor_interval_signed_bias(0.05, 0.08, 0.10), 0.03)
        self.assertAlmostEqual(censor_interval_signed_bias(0.20, 0.10, 0.15), -0.05)
        self.assertAlmostEqual(censor_interval_distance(0.10, 0.08, 0.12), 0.0)

        scored = with_censor_interval_loss(
            pl.DataFrame(
                {
                    "nominal_probability": [0.05, 0.20, 0.10],
                    "reach_lower_bound": [0.08, 0.10, 0.08],
                    "reach_upper_bound": [0.10, 0.15, 0.12],
                }
            )
        )
        self.assertEqual(
            scored["censor_interval_signed_bias"].to_list(),
            [0.03, -0.05000000000000002, 0.0],
        )
        self.assertEqual(
            scored["censor_interval_distance"].to_list(),
            [0.03, 0.05000000000000002, 0.0],
        )

    def test_daily_and_product_spearman_keep_their_declared_units(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["d1"] * 3 + ["d2"] * 3 + ["d3"] * 3,
                "ValueCode": ["a", "b", "c"] * 3,
                "predicted": [1.0, 2.0, 3.0] * 3,
                "realized": [10.0, 20.0, 30.0] * 3,
            }
        )
        daily = daily_cross_sectional_spearman(
            frame,
            predicted_column="predicted",
            realized_column="realized",
        )
        self.assertEqual(daily.height, 3)
        self.assertTrue(
            all(
                abs(value - 1.0) < 1e-12
                for value in daily["daily_cross_product_spearman"]
            )
        )
        temporal = temporal_product_spearman(
            frame,
            predicted_column="predicted",
            realized_column="realized",
            product_columns=("ValueCode",),
        )
        self.assertEqual(temporal.height, 3)
        self.assertTrue(temporal["temporal_product_spearman"].is_null().all())

    def test_q_rank_is_lexicographic_and_uses_simplicity_last(self) -> None:
        scores = pl.DataFrame(
            {
                "candidate": ["complex", "simple", "worse_primary"],
                "primary_loss": [0.1, 0.1, 0.2],
                "worst_month_loss": [0.2, 0.2, 0.0],
                "amplitude_absolute_error_bp": [1.0, 1.0, 0.0],
                "native_all_q_coverage": [0.9, 0.9, 1.0],
                "daily_cross_product_spearman": [0.7, 0.7, 1.0],
                "boundary_turnover": [0.3, 0.3, 0.0],
                "allow_primary_selection": [True, True, True],
            }
        )
        ranked = rank_q_candidates(
            scores,
            simpler_order=("simple", "complex", "worse_primary"),
        )
        self.assertEqual(
            ranked["candidate"].to_list(),
            ["simple", "complex", "worse_primary"],
        )

    def test_q_rank_excludes_diagnostics_and_rejects_nonfinite_metrics(self) -> None:
        scores = pl.DataFrame(
            {
                "candidate": ["selectable", "diagnostic"],
                "primary_loss": [0.1, 0.0],
                "worst_month_loss": [0.1, 0.0],
                "amplitude_absolute_error_bp": [1.0, 0.0],
                "native_all_q_coverage": [0.9, 1.0],
                "daily_cross_product_spearman": [0.7, float("nan")],
                "boundary_turnover": [0.3, 0.0],
                "allow_primary_selection": [True, False],
            }
        )
        ranked = rank_q_candidates(scores)
        self.assertEqual(ranked["candidate"].to_list(), ["selectable"])
        with self.assertRaisesRegex(ValueError, "must be finite"):
            rank_q_candidates(
                scores.with_columns(pl.lit(True).alias("allow_primary_selection"))
            )


if __name__ == "__main__":
    unittest.main()
