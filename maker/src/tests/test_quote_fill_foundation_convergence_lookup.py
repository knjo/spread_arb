"""Focused contracts for D-safe conditional convergence lookups."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from maker.src.quote_fill.foundation_convergence_lookup import (
    REGISTERED_LOOKUPS,
    ConditionalLookupSpec,
    _build_outcome_conditioned_control_convergence_predictions,
    build_conditional_convergence_predictions,
    build_control_convergence_predictions_from_boundaries,
    build_registered_conditional_convergence_grid,
    build_registered_conditional_convergence_grid_prepared,
    first_hit_second_for_threshold,
    prepare_convergence_history,
    resolve_convergence_fallback,
    score_convergence_predictions,
)


def _dates(count: int, *, start: str = "20260501") -> list[str]:
    first = datetime.strptime(start, "%Y%m%d")  # noqa: DTZ007
    return [
        (first + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(count)
    ]


def _fact(
    date_text: str,
    sequence: int,
    floor: float,
    *,
    completed: bool = True,
    anchor: str = "anchor_a",
    candidate: str = "Q2",
    quantile: int = 80,
    frontier_distances: list[float] | None = None,
    frontier_seconds: list[int] | None = None,
) -> dict[str, object]:
    trade_date = datetime.strptime(date_text, "%Y%m%d")  # noqa: DTZ007
    source_asof = (trade_date - timedelta(days=1)).strftime("%Y%m%d")
    distances = (
        frontier_distances
        if frontier_distances is not None
        else ([floor] if floor > 0 else [])
    )
    seconds = (
        frontier_seconds
        if frontier_seconds is not None
        else ([1_010] if floor > 0 else [])
    )
    return {
        "Date": date_text,
        "ValueCode": "2330",
        "QuoteCode": f"Q{date_text[4:6]}",
        "anchor_model_id": anchor,
        "candidate_id": candidate,
        "tod_bucket": "0905_1000",
        "boundary_quantile": quantile,
        "episode_sequence": sequence,
        "source_asof_date": source_asof,
        "upper_distance_bp": 8.0,
        "independent_lower_distance_bp": 2.0,
        "touch_second": 1_000,
        "touch_residual_bp": 8.0,
        "touch_basis_mid_bp": 8.0,
        "touch_anchor_basis_bp": 0.0,
        "frozen_center_basis_bp": 0.0,
        "frozen_independent_lower_basis_bp": -2.0,
        "convergence_reference_semantics": "frozen_anchor_at_upper_touch",
        "center_hit": True,
        "center_hit_second": 1_005,
        "path_end_second": 1_030,
        "observed_post_touch_floor_bp": floor,
        "floor_frontier_distance_bp": distances,
        "floor_frontier_hit_second": seconds,
        "negative_cycle_completed": completed,
        "right_censored": not completed,
        "right_censor_reason": None if completed else "session_cutoff",
    }


def _target(date_text: str, *, anchor: str = "anchor_a") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [date_text],
            "ValueCode": ["2330"],
            "QuoteCode": ["QTARGET"],
            "anchor_model_id": [anchor],
            "candidate_id": ["Q2"],
            "tod_bucket": ["0905_1000"],
            "boundary_quantile": [80],
        }
    )


class ConditionalLookupTests(unittest.TestCase):
    def test_prepared_registered_grid_matches_scalar_oracle_exactly(self) -> None:
        sessions = _dates(63)
        target_date = sessions[61]
        rows: list[dict[str, object]] = []
        for day_index, date_text in enumerate(sessions[:61]):
            for sequence in range(1, 5):
                rows.append(
                    _fact(
                        date_text,
                        sequence,
                        float((day_index * 3 + sequence) % 11) + 0.25,
                        completed=not (sequence == 4 and day_index % 2 == 0),
                    )
                )
        for day_index, date_text in enumerate(sessions[49:61]):
            rows.extend(
                _fact(
                    date_text,
                    sequence,
                    float(day_index + sequence) + 20.5,
                    anchor="anchor_b",
                )
                for sequence in range(1, 3)
            )
        history = pl.from_dicts(rows, infer_schema_length=None)
        targets = pl.concat(
            [
                _target(target_date, anchor="anchor_a"),
                _target(target_date, anchor="anchor_b"),
            ],
            how="vertical",
        )
        oracle = pl.concat(
            [
                build_conditional_convergence_predictions(
                    history,
                    targets,
                    sessions,
                    spec=spec,
                )
                for spec in REGISTERED_LOOKUPS
            ],
            how="vertical",
        ).sort(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "anchor_model_id",
                "candidate_id",
                "tod_bucket",
                "boundary_quantile",
                "convergence_candidate_id",
                "lookup_id",
            ]
        )

        legacy_grid = build_registered_conditional_convergence_grid(
            history,
            targets,
            sessions,
        )
        assert_frame_equal(legacy_grid, oracle, check_exact=True)

        later_rows = pl.from_dicts(
            [
                _fact(target_date, 1, 999.0),
                _fact(sessions[62], 1, 1_999.0),
            ],
            infer_schema_length=None,
        )
        prepared = prepare_convergence_history(
            pl.concat([history, later_rows], how="vertical_relaxed"),
            sessions,
        )
        prepared_grid = build_registered_conditional_convergence_grid_prepared(
            prepared,
            targets,
        )
        assert_frame_equal(prepared_grid, oracle, check_exact=True)

    def test_date_equal_censor_interval_clips_completed_point(self) -> None:
        sessions = _dates(4)
        rows = [_fact(sessions[0], sequence, 10.0) for sequence in range(1, 101)]
        rows.append(_fact(sessions[1], 1, 1.0))
        rows.append(_fact(sessions[2], 1, 5.0, completed=False))
        predictions = build_conditional_convergence_predictions(
            pl.from_dicts(rows, infer_schema_length=None),
            _target(sessions[3]),
            sessions,
            spec=ConditionalLookupSpec("test", 3, 2, 50),
        )

        c2 = predictions.filter(
            pl.col("convergence_candidate_id") == "C2_conditional_reach80"
        ).row(0, named=True)
        c3 = predictions.filter(
            pl.col("convergence_candidate_id") == "C3_conditional_reach50"
        ).row(0, named=True)
        self.assertTrue(c2["native_supported"])
        self.assertAlmostEqual(c2["native_threshold_distance_bp"], 1.0)
        self.assertAlmostEqual(c3["completed_point_bp"], 1.0)
        self.assertAlmostEqual(c3["identified_lower_bp"], 5.0)
        self.assertAlmostEqual(c3["native_threshold_distance_bp"], 5.0)
        self.assertEqual(c3["source_asof_date"], sessions[2])
        self.assertFalse(c3["contains_target_day_outcome"])

    def test_anchor_lineage_prevents_cross_anchor_pooling(self) -> None:
        sessions = _dates(3)
        rows = []
        for sequence in range(1, 6):
            rows.append(_fact(sessions[0], sequence, 1.0, anchor="anchor_a"))
            rows.append(_fact(sessions[0], sequence, 100.0, anchor="anchor_b"))
        predictions = build_conditional_convergence_predictions(
            pl.from_dicts(rows, infer_schema_length=None),
            _target(sessions[2], anchor="anchor_a"),
            sessions,
            spec=ConditionalLookupSpec("test", 2, 1, 1),
        )
        self.assertTrue((predictions["native_threshold_distance_bp"] == 1.0).all())

    def test_history_with_target_day_outcome_fails_closed(self) -> None:
        sessions = _dates(3)
        facts = pl.from_dicts(
            [
                _fact(sessions[0], 1, 1.0),
                _fact(sessions[2], 1, 999.0),
            ],
            infer_schema_length=None,
        )
        with self.assertRaisesRegex(ValueError, "target-day/future"):
            build_conditional_convergence_predictions(
                facts,
                _target(sessions[2]),
                sessions,
                spec=ConditionalLookupSpec("test", 2, 1, 1),
            )

    def test_history_date_must_belong_to_frozen_calendar(self) -> None:
        sessions = _dates(3)
        outside = _fact("20260430", 1, 1.0)
        with self.assertRaisesRegex(ValueError, "absent.*calendar"):
            build_conditional_convergence_predictions(
                pl.from_dicts([outside], infer_schema_length=None),
                _target(sessions[2]),
                sessions,
                spec=ConditionalLookupSpec("test", 2, 1, 1),
            )

    def test_trail20_can_use_explicit_trail60_fallback(self) -> None:
        sessions = _dates(61)
        rows: list[dict[str, object]] = []
        for date_text in sessions[:40]:
            rows.extend(
                _fact(date_text, sequence, float(sequence)) for sequence in range(1, 4)
            )
        for date_text in sessions[40:60]:
            rows.append(_fact(date_text, 1, 1.0, completed=False))
        grid = build_registered_conditional_convergence_grid(
            pl.from_dicts(rows, infer_schema_length=None),
            _target(sessions[60]),
            sessions,
        )
        resolved = resolve_convergence_fallback(
            grid,
            primary_lookup_id="trail20_date_equal",
            fallback_lookup_id="trail60_date_equal",
        )
        self.assertTrue(resolved["effective_supported"].all())
        self.assertTrue(resolved["fallback_used"].all())
        self.assertEqual(
            resolved["effective_lookup_id"].unique().to_list(),
            ["trail60_date_equal"],
        )
        self.assertFalse(resolved["native_supported"].any())


class ConvergenceScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.date = "20260813"
        self.facts = pl.from_dicts(
            [
                _fact(
                    self.date,
                    1,
                    3.0,
                    frontier_distances=[1.0, 3.0],
                    frontier_seconds=[1_010, 1_020],
                ),
                _fact(
                    self.date,
                    2,
                    1.0,
                    completed=False,
                    frontier_distances=[1.0],
                    frontier_seconds=[1_012],
                ),
            ],
            infer_schema_length=None,
        )
        self.boundaries = pl.from_dicts(
            [
                {
                    "Date": self.date,
                    "ValueCode": "2330",
                    "QuoteCode": "Q08",
                    "anchor_model_id": "anchor_a",
                    "candidate_id": "Q2",
                    "tod_bucket": "0905_1000",
                    "boundary_quantile": 80,
                    "side": side,
                    "boundary_distance_bp": distance,
                    "source_asof_date": "20260812",
                    "effective_supported": True,
                    "contains_target_day_outcome": False,
                }
                for side, distance in (("positive", 8.0), ("negative", 2.0))
            ],
            infer_schema_length=None,
        )
        self.controls = build_control_convergence_predictions_from_boundaries(
            self.boundaries
        )

    def test_control_adapter_and_first_hit_censor_bounds(self) -> None:
        controls = self.controls
        self.assertTrue((controls["observable_started"] == 0).all())
        self.assertTrue((~controls["contains_target_day_outcome"]).all())
        self.assertEqual(
            controls["convergence_reference_semantics"].unique().to_list(),
            ["frozen_anchor_at_upper_touch"],
        )
        scored = score_convergence_predictions(self.facts, controls)
        self.assertTrue(scored.path_facts["contains_target_day_outcome"].all())
        self.assertTrue(scored.summary["contains_target_day_outcome"].all())
        self.assertTrue(
            (~scored.summary["prediction_contains_target_day_outcome"]).all()
        )
        c0 = scored.summary.filter(
            pl.col("convergence_candidate_id") == "C0_center"
        ).row(0, named=True)
        c1 = scored.summary.filter(
            pl.col("convergence_candidate_id") == "C1_independent_lower_control"
        ).row(0, named=True)
        self.assertAlmostEqual(c0["reach_lower_bound"], 1.0)
        self.assertAlmostEqual(c0["reach_upper_bound"], 1.0)
        self.assertAlmostEqual(c1["reach_lower_bound"], 0.5)
        self.assertAlmostEqual(c1["reach_upper_bound"], 1.0)
        self.assertEqual(c1["time_to_hit_from_touch_p50_seconds"], 20)
        self.assertEqual(c1["time_to_hit_from_touch_p90_seconds"], 20)
        statuses = scored.path_facts.filter(
            pl.col("convergence_candidate_id") == "C1_independent_lower_control"
        )["hit_status"].to_list()
        self.assertEqual(statuses, ["confirmed_hit", "unknown_censored"])
        c0_path = scored.path_facts.filter(
            pl.col("convergence_candidate_id") == "C0_center"
        ).row(0, named=True)
        self.assertAlmostEqual(c0_path["frozen_exit_basis_bp"], 0.0)
        self.assertAlmostEqual(c0_path["touch_anchor_basis_bp"], 0.0)

    def test_rejects_dynamic_anchor_fact_or_prediction_lineage(self) -> None:
        dynamic_facts = self.facts.with_columns(
            pl.lit("dynamic_anchor_sensitivity").alias(
                "convergence_reference_semantics"
            )
        )
        with self.assertRaisesRegex(ValueError, "path invariants"):
            score_convergence_predictions(dynamic_facts, self.controls)

        dynamic_predictions = self.controls.with_columns(
            pl.lit("dynamic_anchor_sensitivity").alias(
                "convergence_reference_semantics"
            )
        )
        with self.assertRaisesRegex(ValueError, "inconsistent support"):
            score_convergence_predictions(self.facts, dynamic_predictions)

    def test_boundary_controls_include_no_touch_target_lineages(self) -> None:
        boundary_rows: list[dict[str, object]] = []
        for value_code, quote_code, lower in (
            ("2330", "Q08", 2.0),
            ("2317", "QNO_TOUCH", 5.0),
        ):
            for side, distance in (("positive", 8.0), ("negative", lower)):
                boundary_rows.append(
                    {
                        "Date": self.date,
                        "ValueCode": value_code,
                        "QuoteCode": quote_code,
                        "anchor_model_id": "anchor_a",
                        "candidate_id": "Q2",
                        "tod_bucket": "0905_1000",
                        "boundary_quantile": 80,
                        "side": side,
                        "boundary_distance_bp": distance,
                        "source_asof_date": "20260812",
                        "effective_supported": True,
                        "contains_target_day_outcome": False,
                    }
                )
        controls = build_control_convergence_predictions_from_boundaries(
            pl.from_dicts(boundary_rows, infer_schema_length=None)
        )
        self.assertEqual(controls.height, 4)
        self.assertEqual(set(controls["ValueCode"]), {"2317", "2330"})
        self.assertFalse(controls["contains_target_day_outcome"].any())

        fact_controls = _build_outcome_conditioned_control_convergence_predictions(
            self.facts
        )
        self.assertTrue(fact_controls["contains_target_day_outcome"].all())
        with self.assertRaisesRegex(ValueError, "inconsistent support"):
            score_convergence_predictions(self.facts, fact_controls)
        parity_columns = [
            column
            for column in controls.columns
            if column != "contains_target_day_outcome"
        ]
        assert_frame_equal(
            controls.filter(pl.col("ValueCode") == "2330").select(parity_columns),
            fact_controls.select(parity_columns),
            check_exact=True,
        )
        scored = score_convergence_predictions(self.facts, controls)
        no_touch = scored.summary.filter(pl.col("ValueCode") == "2317")
        self.assertEqual(no_touch.height, 2)
        self.assertTrue((no_touch["n_started"] == 0).all())
        self.assertEqual(
            no_touch["outcome_status"].unique().to_list(),
            ["no_observable_touched_path"],
        )

    def test_arbitrary_threshold_uses_first_hit_frontier(self) -> None:
        controls = self.controls
        arbitrary = controls.filter(
            pl.col("convergence_candidate_id") == "C1_independent_lower_control"
        ).with_columns(
            pl.lit("arbitrary_4bp").alias("convergence_candidate_id"),
            pl.lit(4.0).alias("native_threshold_distance_bp"),
            pl.lit(4.0).alias("effective_threshold_distance_bp"),
        )
        scored = score_convergence_predictions(self.facts, arbitrary)
        summary = scored.summary.row(0, named=True)
        self.assertAlmostEqual(summary["reach_lower_bound"], 0.0)
        self.assertAlmostEqual(summary["reach_upper_bound"], 0.5)
        self.assertEqual(summary["known_misses"], 1)
        self.assertEqual(summary["unknown_censored"], 1)

        self.assertEqual(
            first_hit_second_for_threshold(
                threshold_distance_bp=2.0,
                center_hit_second=1_005,
                floor_frontier_distance_bp=[1.0, 3.0],
                floor_frontier_hit_second=[1_010, 1_020],
            ),
            1_020,
        )

    def test_conditional_predictions_join_exact_lineage_for_scoring(self) -> None:
        sessions = _dates(16, start="20260701")
        history_rows = [
            _fact(date_text, sequence, float(sequence))
            for date_text in sessions[:15]
            for sequence in range(1, 5)
        ]
        target_facts = pl.from_dicts(
            [
                {
                    **_fact(
                        sessions[15],
                        1,
                        3.0,
                        frontier_distances=[1.0, 3.0],
                        frontier_seconds=[1_010, 1_020],
                    ),
                    "QuoteCode": "QTARGET",
                }
            ],
            infer_schema_length=None,
        )
        predictions = build_conditional_convergence_predictions(
            pl.from_dicts(history_rows, infer_schema_length=None),
            _target(sessions[15]),
            sessions,
            spec=ConditionalLookupSpec("trail20_test", 20, 15, 50),
        )
        scored = score_convergence_predictions(target_facts, predictions)
        self.assertEqual(scored.summary.height, 2)
        self.assertTrue((scored.summary["reach_lower_bound"] == 1.0).all())
        self.assertEqual(
            set(scored.path_facts["convergence_candidate_id"]),
            {"C2_conditional_reach80", "C3_conditional_reach50"},
        )


if __name__ == "__main__":
    unittest.main()
