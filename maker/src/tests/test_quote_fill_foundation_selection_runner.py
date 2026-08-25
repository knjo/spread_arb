from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.foundation_anchor_selection import (
    ACTIONABLE_MODEL_IDS,
)
from maker.src.quote_fill.foundation_selection_runner import (
    CHECKPOINT_SCHEMA_VERSION,
    COUNT120_DIAGNOSTIC_ID,
    DEFAULT_REGISTRY_PATH,
    EXPECTED_PRIMARY_SESSION_COUNT,
    EXPECTED_SESSION_COUNT,
    SelectionPaths,
    _parser,
    _require_canonical_git_state,
    _selected_anchor_payload,
    _validate_registry_constants,
    aggregate_anchor_daily,
    atomic_publish_checkpoint,
    build_convergence_common_support_sensitivity,
    build_convergence_comparison_predictions,
    build_convergence_support_audit,
    build_file_records,
    build_s1_lookup_views,
    build_selected_q_payload,
    checkpoint_fingerprint,
    load_registry,
    load_selection_sessions,
    resume_date_checkpoint,
    summarize_boundary_candidate_scores,
    summarize_convergence_primary,
    verify_date_checkpoint,
)
from maker.src.quote_fill.foundation_selection_stats import rank_q_candidates


class FoundationSelectionRegistryTest(unittest.TestCase):
    def test_frozen_registry_and_session_calendar_match_executable_contract(
        self,
    ) -> None:
        registry = load_registry(DEFAULT_REGISTRY_PATH)
        sessions = load_selection_sessions()

        self.assertEqual(
            registry.payload["registry_version"],
            "foundation_selection_s05_rebuild_registry_v1",
        )
        self.assertEqual(len(sessions), EXPECTED_SESSION_COUNT)
        self.assertEqual(
            len([date for date in sessions if date >= "20260505"]),
            EXPECTED_PRIMARY_SESSION_COUNT,
        )
        self.assertLess(sessions[-1], "20260814")

    def test_semantic_registry_drift_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload["entry_stop_second"] = 14_401

        with self.assertRaisesRegex(ValueError, "registry constant drift"):
            _validate_registry_constants(payload)

    def test_registry_requires_the_byte_exact_frozen_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            changed = Path(raw_root) / "registry.json"
            changed.write_bytes(DEFAULT_REGISTRY_PATH.read_bytes() + b"\n")

            with self.assertRaisesRegex(ValueError, "registry hash drift"):
                load_registry(changed)

    def test_canonical_context_rejects_a_dirty_source_tree(self) -> None:
        state = {"commit": "abc123", "dirty": True, "status": "?? new.py"}
        with (
            patch(
                "maker.src.quote_fill.foundation_selection_runner._git_state",
                return_value=state,
            ),
            self.assertRaisesRegex(ValueError, "clean committed source tree"),
        ):
            _require_canonical_git_state(True)

        with patch(
            "maker.src.quote_fill.foundation_selection_runner._git_state",
            return_value=state,
        ):
            self.assertEqual(_require_canonical_git_state(False), state)


class FoundationSelectionCheckpointTest(unittest.TestCase):
    def test_fingerprint_is_deterministic_over_input_record_order(self) -> None:
        records = [
            {"path": "/tmp/b", "role": "b", "bytes": 2, "sha256": "bb"},
            {"path": "/tmp/a", "role": "a", "bytes": 1, "sha256": "aa"},
        ]
        kwargs = {
            "phase": "anchor",
            "date": "20260505",
            "registry_sha256": "registry",
            "source_commit": "commit",
            "parameters": {"models": ["a", "b"]},
        }

        first = checkpoint_fingerprint(input_records=records, **kwargs)
        second = checkpoint_fingerprint(
            input_records=list(reversed(records)),
            **kwargs,
        )
        changed = checkpoint_fingerprint(
            input_records=[{**records[0], "sha256": "changed"}, records[1]],
            **kwargs,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_atomic_checkpoint_resumes_only_after_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source.txt"
            source.write_text("source-v1\n", encoding="utf-8")
            records = build_file_records([(source, "fixture")])
            parameters = {"contract": "fixture-v1"}
            fingerprint = checkpoint_fingerprint(
                phase="fixture",
                date="20260505",
                registry_sha256="registry",
                source_commit="commit",
                input_records=records,
                parameters=parameters,
            )
            partition = root / "work" / "Date=20260505"
            marker = atomic_publish_checkpoint(
                partition,
                phase="fixture",
                date="20260505",
                registry_sha256="registry",
                source_commit="commit",
                input_records=records,
                parameters=parameters,
                artifact_values={"facts.parquet": pl.DataFrame({"x": [1, 2]})},
            )

            self.assertEqual(marker["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            resumed = resume_date_checkpoint(
                partition,
                expected_fingerprint=fingerprint,
                resume=True,
            )
            self.assertIsNotNone(resumed)
            with self.assertRaises(FileExistsError):
                resume_date_checkpoint(
                    partition,
                    expected_fingerprint=fingerprint,
                    resume=False,
                )

            pl.DataFrame({"x": [9]}).write_parquet(partition / "facts.parquet")
            with self.assertRaisesRegex(ValueError, "artifact metadata mismatch"):
                verify_date_checkpoint(partition)

    def test_work_root_is_a_deterministic_output_sibling(self) -> None:
        output = Path("/tmp/foundation-selection-result")
        paths = SelectionPaths(output_root=output)

        self.assertEqual(
            paths.resolved_work_root,
            output.with_name(".foundation-selection-result.work"),
        )

    def test_maker_inputs_use_checkout_portable_relative_identity(self) -> None:
        record = build_file_records([(DEFAULT_REGISTRY_PATH, "frozen_registry")])[0]

        self.assertEqual(record["path_scope"], "maker_root")
        self.assertFalse(Path(str(record["path"])).is_absolute())
        self.assertEqual(
            record["path"],
            "src/quote_fill/foundation_selection_registry.json",
        )


class FoundationSelectionAggregationTest(unittest.TestCase):
    @staticmethod
    def _product_day_fixture() -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for date in ("20260505", "20260506"):
            for model_index, model in enumerate(ACTIONABLE_MODEL_IDS):
                for product_index, value_code in enumerate(("1101", "2330")):
                    rows.append(
                        {
                            "Date": date,
                            "month": date[:6],
                            "ValueCode": value_code,
                            "QuoteCode": f"{value_code}F",
                            "model": model,
                            "horizon_id": "future_median_30_300s",
                            "tod_bucket": "all",
                            "mae_bp": float(model_index + product_index * 2),
                            "anchor_tv_over_basis_tv": 0.2 + 0.2 * product_index,
                            "n_common_evaluable": 10 + product_index * 10,
                        }
                    )
        rows.append(
            {
                "Date": "20260505",
                "month": "202605",
                "ValueCode": "1101",
                "QuoteCode": "1101F",
                "model": ACTIONABLE_MODEL_IDS[0],
                "horizon_id": "future_median_10_60s",
                "tod_bucket": "all",
                "mae_bp": 999.0,
                "anchor_tv_over_basis_tv": 999.0,
                "n_common_evaluable": 999,
            }
        )
        return pl.from_dicts(rows, infer_schema_length=None)

    def test_primary_daily_aggregation_is_product_day_equal(self) -> None:
        daily = aggregate_anchor_daily(self._product_day_fixture())
        row = daily.filter(
            (pl.col("Date") == "20260505")
            & (pl.col("model") == ACTIONABLE_MODEL_IDS[0])
        ).row(0, named=True)

        self.assertEqual(row["product_days"], 2)
        self.assertEqual(row["common_seconds"], 30)
        self.assertAlmostEqual(float(row["mae_bp"]), 1.0)
        self.assertAlmostEqual(float(row["tv_ratio"]), 0.3)
        self.assertEqual(
            daily.filter(pl.col("Date") == "20260505")["product_days"].n_unique(),
            1,
        )

    def test_selected_payload_records_rank_one_rank_two_and_count120(self) -> None:
        selection = pl.DataFrame(
            {
                "model": [
                    ACTIONABLE_MODEL_IDS[0],
                    ACTIONABLE_MODEL_IDS[1],
                    ACTIONABLE_MODEL_IDS[2],
                ],
                "selection_rank": [1, 2, None],
                "selection_role": ["winner", "smooth", None],
            }
        )

        payload = _selected_anchor_payload(selection)

        self.assertEqual(
            payload["episode_anchor_ids"],
            [
                ACTIONABLE_MODEL_IDS[0],
                ACTIONABLE_MODEL_IDS[1],
                COUNT120_DIAGNOSTIC_ID,
            ],
        )


class FoundationBoundaryRankingTest(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
        candidates = ["Q1_trail60_date_equal", "Q2_trail20_date_equal"]
        dates = ["20260505", "20260506", "20260507"]
        products = ["1101", "2330"]
        tod_buckets = [
            "0905_1000",
            "1000_1100",
            "1100_1200",
            "1200_1300",
        ]
        mapping_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        calibration_rows: list[dict[str, object]] = []
        for date_index, date in enumerate(dates):
            for product_index, product in enumerate(products):
                quote = f"{product}F"
                mapping_rows.append(
                    {"Date": date, "ValueCode": product, "QuoteCode": quote}
                )
                for candidate_index, candidate in enumerate(candidates):
                    for tod in tod_buckets:
                        for side_index, side in enumerate(("positive", "negative")):
                            for quantile in (50, 80, 95):
                                realized = (
                                    5.0
                                    + product_index * 3.0
                                    + date_index
                                    + side_index * 0.5
                                    + quantile / 10.0
                                )
                                predicted = realized + 0.1 + candidate_index * 0.9
                                common = {
                                    "Date": date,
                                    "ValueCode": product,
                                    "QuoteCode": quote,
                                    "anchor_model_id": "time_ewma_30s",
                                    "candidate_id": candidate,
                                    "tod_bucket": tod,
                                    "boundary_quantile": quantile,
                                    "side": side,
                                    "boundary_distance_bp": predicted,
                                    "native_supported": True,
                                }
                                prediction_rows.append(common)
                                calibration_rows.append(
                                    {
                                        **common,
                                        "predicted_distance_bp": predicted,
                                        "realized_completed_quantile_bp": realized,
                                        "observable_started": 10,
                                        "reach_interval_distance": (
                                            0.01 + candidate_index * 0.01
                                        ),
                                    }
                                )
        return (
            pl.from_dicts(calibration_rows, infer_schema_length=None),
            pl.from_dicts(prediction_rows, infer_schema_length=None),
            pl.from_dicts(mapping_rows, infer_schema_length=None),
            dates,
        )

    def test_q_scores_use_common_support_and_registered_reductions(self) -> None:
        calibration, predictions, mapping, dates = self._fixture()
        candidates = ["Q1_trail60_date_equal", "Q2_trail20_date_equal"]

        facts = summarize_boundary_candidate_scores(
            calibration,
            predictions,
            mapping,
            candidates=candidates,
            sessions=dates,
            minimum_cross_sectional_products=2,
            minimum_temporal_dates=2,
        )

        self.assertEqual(facts.candidate_scores.height, 2)
        self.assertEqual(facts.common_calibration.height, calibration.height)
        self.assertTrue(
            (facts.candidate_scores["native_all_q_coverage"] - 1.0).abs().max() < 1e-12
        )
        q1 = facts.candidate_scores.filter(
            pl.col("candidate_id") == "Q1_trail60_date_equal"
        ).row(0, named=True)
        q2 = facts.candidate_scores.filter(
            pl.col("candidate_id") == "Q2_trail20_date_equal"
        ).row(0, named=True)
        self.assertLess(q1["primary_loss"], q2["primary_loss"])
        self.assertLess(
            q1["amplitude_absolute_error_bp"],
            q2["amplitude_absolute_error_bp"],
        )
        ranking = rank_q_candidates(
            facts.candidate_scores,
            candidate_column="candidate_id",
            simpler_order=candidates,
        )
        self.assertEqual(
            ranking.row(0, named=True)["candidate_id"],
            "Q1_trail60_date_equal",
        )

    def test_selected_q_payload_keeps_final_top_two_and_tod_sources(self) -> None:
        ranking = pl.DataFrame(
            {
                "candidate_id": [
                    "Q3_shape60_level5",
                    "Q2_trail20_date_equal__tod10",
                ],
                "selection_rank": [1, 2],
            }
        )

        payload = build_selected_q_payload(
            ranking,
            base_tod_source_candidates=(
                "Q1_trail60_date_equal",
                "Q2_trail20_date_equal",
            ),
        )

        self.assertEqual(payload["rank1_candidate_id"], "Q3_shape60_level5")
        self.assertEqual(
            payload["rank2_candidate_id"],
            "Q2_trail20_date_equal__tod10",
        )

    def test_native_coverage_uses_full_mapped_tod_denominator(self) -> None:
        calibration, predictions, mapping, dates = self._fixture()
        missing_unit = (
            (pl.col("candidate_id") == "Q2_trail20_date_equal")
            & (pl.col("Date") == dates[0])
            & (pl.col("ValueCode") == "1101")
            & (pl.col("tod_bucket") == "0905_1000")
        )
        predictions = predictions.filter(~missing_unit)

        facts = summarize_boundary_candidate_scores(
            calibration,
            predictions,
            mapping,
            candidates=("Q1_trail60_date_equal", "Q2_trail20_date_equal"),
            sessions=dates,
            minimum_cross_sectional_products=2,
            minimum_temporal_dates=2,
        )
        q2 = facts.candidate_scores.filter(
            pl.col("candidate_id") == "Q2_trail20_date_equal"
        ).row(0, named=True)

        self.assertEqual(q2["native_all_q_units"], 23)
        self.assertEqual(q2["expected_product_day_tod_units"], 24)
        self.assertAlmostEqual(q2["native_all_q_coverage"], 23 / 24)


class FoundationSelectionPublicationViewTest(unittest.TestCase):
    @staticmethod
    def _selected_boundaries() -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for tod in ("0905_1000", "1000_1100", "1100_1200", "1200_1300"):
            for quantile in (50, 80, 95):
                for side in ("positive", "negative"):
                    rows.append(
                        {
                            "Date": "20260813",
                            "ValueCode": "2330",
                            "QuoteCode": "2330F",
                            "anchor_model_id": "time_ewma_30s",
                            "candidate_id": "Q3_shape60_level5",
                            "tod_bucket": tod,
                            "boundary_quantile": quantile,
                            "side": side,
                            "boundary_distance_bp": float(quantile),
                            "source_asof_date": "20260812",
                            "effective_supported": True,
                            "effective_candidate_id": "Q3_shape60_level5",
                            "contains_target_day_outcome": False,
                        }
                    )
        return pl.from_dicts(rows, infer_schema_length=None)

    def test_s1_lookup_views_are_exact_long24_and_wide4(self) -> None:
        long, wide = build_s1_lookup_views(self._selected_boundaries())

        self.assertEqual(long.height, 24)
        self.assertEqual(wide.height, 4)
        self.assertTrue(
            {
                "upper_q50_bp",
                "upper_q80_bp",
                "upper_q95_bp",
                "lower_q50_bp",
                "lower_q80_bp",
                "lower_q95_bp",
            }.issubset(wide.columns)
        )
        self.assertTrue((wide["long_rows_per_product_day"] == 24).all())
        self.assertFalse(wide["contains_target_day_outcome"].any())

    def test_cli_exposes_convergence_and_final_phases(self) -> None:
        parser = _parser()

        self.assertEqual(parser.parse_args(["convergence"]).phase, "convergence")
        self.assertEqual(parser.parse_args(["final"]).phase, "final")


class FoundationConvergenceRunnerTest(unittest.TestCase):
    @staticmethod
    def _prediction(
        lookup_id: str,
        candidate_id: str,
    ) -> dict[str, object]:
        return {
            "Date": "20260813",
            "ValueCode": "2330",
            "QuoteCode": "2330F",
            "anchor_model_id": "time_ewma_30s",
            "candidate_id": "Q3_shape60_level5",
            "tod_bucket": "0905_1000",
            "boundary_quantile": 80,
            "convergence_candidate_id": candidate_id,
            "lookup_id": lookup_id,
            "lookback_sessions": 20 if lookup_id == "trail20_date_equal" else 60,
            "minimum_completed_dates": 1,
            "minimum_completed_paths": 1,
            "target_reach_probability": 0.8,
            "floor_quantile_probability": 0.2,
            "source_asof_date": "20260812",
            "history_start_date": "20260801",
            "history_end_date": "20260812",
            "history_observation_end_date": "20260812",
            "selected_sessions": 10,
            "completed_history_dates": 10,
            "observable_started": 100,
            "completed_paths": 100,
            "right_censored_paths": 0,
            "completed_point_bp": 2.0,
            "identified_lower_bp": 2.0,
            "identified_upper_bp": 2.0,
            "identified_upper_unbounded": False,
            "native_threshold_distance_bp": 2.0,
            "native_supported": True,
            "native_failure_reason": None,
            "effective_threshold_distance_bp": 2.0,
            "effective_supported": True,
            "effective_lookup_id": lookup_id,
            "effective_source_asof_date": "20260812",
            "fallback_lookup_id": None,
            "fallback_used": False,
            "fallback_reason": None,
            "contains_target_day_outcome": False,
        }

    def test_comparison_keeps_both_windows_resolved_route_and_controls(self) -> None:
        from maker.src.quote_fill.foundation_convergence_lookup import (
            CONVERGENCE_PREDICTION_SCHEMA,
        )

        registered = pl.from_dicts(
            [
                self._prediction(lookup, candidate)
                for lookup in ("trail20_date_equal", "trail60_date_equal")
                for candidate in (
                    "C2_conditional_reach80",
                    "C3_conditional_reach50",
                )
            ],
            schema=CONVERGENCE_PREDICTION_SCHEMA,
            infer_schema_length=None,
        )
        resolved = registered.filter(pl.col("lookup_id") == "trail20_date_equal")
        controls = pl.from_dicts(
            [
                self._prediction("structural_zero", "C0_center"),
                self._prediction(
                    "independent_lower_control",
                    "C1_independent_lower_control",
                ),
            ],
            schema=CONVERGENCE_PREDICTION_SCHEMA,
            infer_schema_length=None,
        )

        comparison = build_convergence_comparison_predictions(
            registered,
            resolved,
            controls,
        )

        self.assertEqual(comparison.height, 8)
        self.assertEqual(
            comparison["convergence_candidate_id"].n_unique(),
            8,
        )
        self.assertIn(
            "C2_conditional_reach80__trail60_date_equal",
            comparison["convergence_candidate_id"].to_list(),
        )
        self.assertIn(
            "C3_conditional_reach50__trail20_then_trail60",
            comparison["convergence_candidate_id"].to_list(),
        )

        targets = registered.select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "anchor_model_id",
            "candidate_id",
            "tod_bucket",
            "boundary_quantile",
        ).unique()
        support = build_convergence_support_audit(
            targets,
            registered,
            resolved,
            controls,
        )
        self.assertEqual(support.height, 8)
        self.assertTrue(
            (support["effective_support_coverage"] - 1.0).abs().max() < 1e-12
        )
        self.assertFalse(support["contains_target_day_outcome"].any())

    def test_primary_summary_reports_bounds_without_selecting_winner(self) -> None:
        path_scores = pl.DataFrame(
            {
                "Date": ["20260811", "20260812", "20260813"],
                "ValueCode": ["2330", "2330", "2317"],
                "candidate_id": ["Q3", "Q3", "Q3"],
                "boundary_quantile": [80, 80, 80],
                "convergence_candidate_id": [
                    "C2_conditional_reach80__trail20_then_trail60"
                ]
                * 3,
                "effective_lookup_id": [
                    "trail20_date_equal",
                    "trail60_date_equal",
                    "trail60_date_equal",
                ],
                "confirmed_hit": [True, False, False],
                "known_miss": [False, True, False],
                "unknown_censored": [False, False, True],
                "time_to_hit_from_touch_seconds": [10, None, None],
                "time_to_hit_from_center_seconds": [5, None, None],
            }
        )

        summary = summarize_convergence_primary(path_scores).row(0, named=True)

        self.assertAlmostEqual(summary["reach_lower_bound"], 1 / 3)
        self.assertAlmostEqual(summary["reach_upper_bound"], 2 / 3)
        self.assertEqual(summary["time_to_hit_from_touch_p90_seconds"], 10)
        self.assertEqual(summary["effective_lookup_id_count"], 2)
        self.assertEqual(summary["fallback_effective_paths"], 2)
        self.assertAlmostEqual(summary["fallback_source_share"], 2 / 3)
        self.assertFalse(summary["unique_winner_declared"])

    def test_entry_q_sensitivity_uses_exact_shared_episode_paths(self) -> None:
        rows: list[dict[str, object]] = []
        for candidate, statuses in (
            ("Q_rank1", ("confirmed", "confirmed")),
            ("Q_rank2", ("confirmed", "known")),
        ):
            for sequence, status in enumerate(statuses, start=1):
                rows.append(
                    {
                        "Date": "20260813",
                        "ValueCode": "2330",
                        "QuoteCode": "2330F",
                        "candidate_id": candidate,
                        "tod_bucket": "0905_1000",
                        "boundary_quantile": 80,
                        "episode_sequence": sequence,
                        "convergence_candidate_id": "C0_center",
                        "effective_lookup_id": "structural_zero",
                        "confirmed_hit": status == "confirmed",
                        "known_miss": status == "known",
                        "unknown_censored": False,
                        "time_to_hit_from_touch_seconds": (
                            10 if status == "confirmed" else None
                        ),
                        "time_to_hit_from_center_seconds": (
                            0 if status == "confirmed" else None
                        ),
                    }
                )
        common, review, sensitivity = build_convergence_common_support_sensitivity(
            pl.from_dicts(rows, infer_schema_length=None),
            entry_q_candidate_ids=("Q_rank1", "Q_rank2"),
        )

        self.assertEqual(common.height, 4)
        self.assertEqual(review.height, 2)
        result = sensitivity.row(0, named=True)
        self.assertEqual(result["common_paths"], 2)
        self.assertAlmostEqual(result["rank1_reach_lower_bound"], 1.0)
        self.assertAlmostEqual(result["rank2_reach_lower_bound"], 0.5)
        self.assertAlmostEqual(
            result["rank2_minus_rank1_reach_lower_bound"],
            -0.5,
        )


if __name__ == "__main__":
    unittest.main()
