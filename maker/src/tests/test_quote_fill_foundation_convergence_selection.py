"""Tests for conditional post-touch convergence facts."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from maker.src.quote_fill.foundation_convergence_selection import (
    build_post_touch_convergence_facts,
)


def _day(
    residuals: list[float | None], eligible: list[bool] | None = None
) -> pl.DataFrame:
    seconds = list(range(15_600))
    basis: list[float | None] = [0.0] * len(seconds)
    gate = [False] * len(seconds)
    supplied_gate = eligible or [True] * len(residuals)
    for offset, residual in enumerate(residuals):
        basis[300 + offset] = residual
        gate[300 + offset] = supplied_gate[offset]
    return pl.DataFrame(
        {
            "Date": ["20260505"] * len(seconds),
            "ValueCode": ["2330"] * len(seconds),
            "QuoteCode": ["CDF"] * len(seconds),
            "timestamp": [
                datetime(2026, 5, 5, 9, 0) + timedelta(seconds=value)  # noqa: DTZ001
                for value in seconds
            ],
            "seconds_from_open": seconds,
            "basis_mid_bp": basis,
            "analysis_eligible": gate,
            "anchor": [0.0] * len(seconds),
        }
    )


def _boundaries(source_asof: str = "20260504") -> pl.DataFrame:
    rows = []
    for quantile, upper, lower in (
        (50, 2.0, 1.0),
        (80, 3.0, 2.0),
        (95, 4.0, 3.0),
    ):
        for side, distance in (("positive", upper), ("negative", lower)):
            rows.append(
                {
                    "Date": "20260505",
                    "ValueCode": "2330",
                    "QuoteCode": "CDF",
                    "anchor_model_id": "anchor_model",
                    "candidate_id": "Q",
                    "boundary_quantile": quantile,
                    "tod_bucket": "0905_1000",
                    "side": side,
                    "boundary_distance_bp": distance,
                    "source_asof_date": source_asof,
                }
            )
    return pl.from_dicts(rows)


def _day_with_moving_anchor(
    basis_values: list[float],
    anchor_values: list[float],
) -> pl.DataFrame:
    if len(basis_values) != len(anchor_values):
        raise ValueError("basis and anchor fixtures must have equal length")
    seconds = list(range(15_600))
    basis: list[float | None] = [0.0] * len(seconds)
    anchor = [0.0] * len(seconds)
    eligible = [False] * len(seconds)
    for offset, (basis_value, anchor_value) in enumerate(
        zip(basis_values, anchor_values, strict=True)
    ):
        basis[300 + offset] = basis_value
        anchor[300 + offset] = anchor_value
        eligible[300 + offset] = True
    return pl.DataFrame(
        {
            "Date": ["20260505"] * len(seconds),
            "ValueCode": ["2330"] * len(seconds),
            "QuoteCode": ["CDF"] * len(seconds),
            "timestamp": [
                datetime(2026, 5, 5, 9, 0) + timedelta(seconds=value)  # noqa: DTZ001
                for value in seconds
            ],
            "seconds_from_open": seconds,
            "basis_mid_bp": basis,
            "analysis_eligible": eligible,
            "anchor": anchor,
        }
    )


class ConvergenceFactsTest(unittest.TestCase):
    def test_future_anchor_drift_alone_cannot_create_frozen_lower_hit(self) -> None:
        facts = build_post_touch_convergence_facts(
            _day_with_moving_anchor(
                [-1.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 0.0, 0.0, 5.0, 10.0],
            ),
            _boundaries(),
            anchor_column="anchor",
            anchor_model_id="anchor_model",
        )

        q95 = facts.filter(pl.col("boundary_quantile") == 95).row(0, named=True)
        self.assertEqual(q95["touch_second"], 302)
        self.assertAlmostEqual(q95["touch_anchor_basis_bp"], 0.0)
        self.assertAlmostEqual(q95["frozen_center_basis_bp"], 0.0)
        self.assertAlmostEqual(q95["frozen_independent_lower_basis_bp"], -3.0)
        self.assertEqual(
            q95["convergence_reference_semantics"],
            "frozen_anchor_at_upper_touch",
        )
        self.assertFalse(q95["center_hit"])
        self.assertFalse(q95["independent_lower_confirmed_hit"])
        self.assertEqual(q95["floor_frontier_distance_bp"], [])
        self.assertTrue(q95["right_censored"])

    def test_tracks_center_and_following_negative_floor(self) -> None:
        facts = build_post_touch_convergence_facts(
            _day([-1.0, 1.0, 2.0, 4.0, 2.0, -0.5, -2.5, -1.0, 0.5]),
            _boundaries(),
            anchor_column="anchor",
            anchor_model_id="anchor_model",
        )
        q80 = facts.filter(pl.col("boundary_quantile") == 80).row(0, named=True)
        self.assertEqual(q80["touch_second"], 303)
        self.assertEqual(q80["center_hit_second"], 305)
        self.assertAlmostEqual(q80["observed_post_touch_floor_bp"], 2.5)
        self.assertEqual(q80["floor_frontier_distance_bp"], [0.5, 2.5])
        self.assertEqual(q80["floor_frontier_hit_second"], [305, 306])
        self.assertTrue(q80["negative_cycle_completed"])
        self.assertTrue(q80["independent_lower_confirmed_hit"])
        self.assertEqual(q80["independent_lower_hit_second"], 306)
        self.assertEqual(q80["independent_lower_time_from_touch_seconds"], 3)
        self.assertEqual(q80["independent_lower_time_from_center_seconds"], 1)
        self.assertFalse(q80["right_censored"])

    def test_gap_after_touch_is_unknown_not_miss(self) -> None:
        facts = build_post_touch_convergence_facts(
            _day(
                [-1.0, 1.0, 3.0, 4.0, 1.0, None, -5.0],
                [True, True, True, True, True, False, True],
            ),
            _boundaries(),
            anchor_column="anchor",
            anchor_model_id="anchor_model",
        )
        q95 = facts.filter(pl.col("boundary_quantile") == 95).row(0, named=True)
        self.assertTrue(q95["right_censored"])
        self.assertEqual(q95["right_censor_reason"], "eligibility_gap")
        self.assertTrue(q95["independent_lower_unknown"])
        self.assertFalse(q95["independent_lower_known_miss"])

    def test_rejects_target_day_boundary_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly before"):
            build_post_touch_convergence_facts(
                _day([-1.0, 1.0, 4.0, -1.0, 1.0]),
                _boundaries("20260505"),
                anchor_column="anchor",
                anchor_model_id="anchor_model",
            )

    def test_excludes_positive_path_already_active_after_gap(self) -> None:
        facts = build_post_touch_convergence_facts(
            _day([3.0, 4.0, -1.0, 1.0, 3.0, -1.0]),
            _boundaries(),
            anchor_column="anchor",
            anchor_model_id="anchor_model",
        )
        self.assertGreater(facts.height, 0)
        self.assertTrue((facts["episode_start_second"] == 303).all())


if __name__ == "__main__":
    unittest.main()
