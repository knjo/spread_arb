"""Tests for the frozen-at-touch S0.5 convergence supplement runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from maker.src.quote_fill.foundation_frozen_convergence_runner import (
    OUTPUT_ARTIFACTS,
    RUNNER_VERSION,
    FrozenConvergencePaths,
    _add_geometry_positive_shares,
    _ensure_fact_date,
    _fact_partition,
    _parser,
    _summarize_frozen_primary,
    _UpstreamContext,
    _validate_destinations,
)
from maker.src.quote_fill.foundation_selection_runner import atomic_publish_checkpoint


class FrozenConvergenceRunnerContractTest(unittest.TestCase):
    def test_default_contract_is_v2_and_publishes_auditable_outputs(self) -> None:
        self.assertTrue(RUNNER_VERSION.endswith("_v2"))
        self.assertIn("frozen_touch_facts.parquet", OUTPUT_ARTIFACTS)
        self.assertIn("frozen_effective_predictions.parquet", OUTPUT_ARTIFACTS)
        self.assertIn("conditional_policy_geometry.parquet", OUTPUT_ARTIFACTS)
        self.assertIn("dynamic_anchor_v1_sensitivity_review.parquet", OUTPUT_ARTIFACTS)

    def test_rejects_output_or_work_nested_under_any_source(self) -> None:
        base = Path("/tmp/frozen-convergence-runner-contract")
        with self.assertRaisesRegex(ValueError, "outside source roots"):
            _validate_destinations(
                FrozenConvergencePaths(
                    source_root=base / "source",
                    source_work_root=base / "legacy-work",
                    daily_root=base / "daily",
                    output_root=base / "source" / "supplement",
                    work_root=base / "new-work",
                )
            )

    def test_rejects_nested_output_and_work_roots(self) -> None:
        base = Path("/tmp/frozen-convergence-runner-contract")
        with self.assertRaisesRegex(ValueError, "must not be nested"):
            _validate_destinations(
                FrozenConvergencePaths(
                    source_root=base / "source",
                    source_work_root=base / "legacy-work",
                    daily_root=base / "daily",
                    output_root=base / "supplement",
                    work_root=base / "supplement" / "work",
                )
            )
        with self.assertRaisesRegex(ValueError, "outside source roots"):
            _validate_destinations(
                FrozenConvergencePaths(
                    source_root=base / "source",
                    source_work_root=base / "legacy-work",
                    daily_root=base / "daily",
                    output_root=base / "supplement",
                    work_root=base / "daily" / "work",
                )
            )

    def test_cli_requires_explicit_publish_execution(self) -> None:
        args = _parser().parse_args(["publish"])
        self.assertFalse(args.execute)
        self.assertEqual(args.phase, "publish")

    def test_plain_c2_id_reports_registered_trail60_fallback(self) -> None:
        path_scores = pl.from_dicts(
            [
                {
                    "Date": "20260801",
                    "ValueCode": "A",
                    "candidate_id": "Q2_trail20_date_equal",
                    "boundary_quantile": 95,
                    "convergence_candidate_id": "C2_conditional_reach80",
                    "effective_lookup_id": "trail20_date_equal",
                    "confirmed_hit": True,
                    "known_miss": False,
                    "unknown_censored": False,
                    "time_to_hit_from_touch_seconds": 3,
                    "time_to_hit_from_center_seconds": 2,
                },
                {
                    "Date": "20260801",
                    "ValueCode": "B",
                    "candidate_id": "Q2_trail20_date_equal",
                    "boundary_quantile": 95,
                    "convergence_candidate_id": "C2_conditional_reach80",
                    "effective_lookup_id": "trail60_date_equal",
                    "confirmed_hit": False,
                    "known_miss": True,
                    "unknown_censored": False,
                    "time_to_hit_from_touch_seconds": None,
                    "time_to_hit_from_center_seconds": None,
                },
            ],
            infer_schema_length=None,
        )
        row = _summarize_frozen_primary(path_scores).row(0, named=True)
        self.assertEqual(row["trail60_effective_paths"], 1)
        self.assertEqual(row["fallback_effective_paths"], 1)
        self.assertEqual(row["fallback_source_share"], 0.5)

    def test_geometry_positive_shares_are_derived_from_supported_rows(self) -> None:
        result = _add_geometry_positive_shares(
            pl.DataFrame(
                {
                    "product_day_tod_rows": [4],
                    "supported_cell_rows": [4],
                    "nominal_same_day_known_cost_positive_rows": [3],
                    "nominal_overnight_known_cost_positive_rows": [1],
                }
            )
        )
        self.assertEqual(
            result.item(0, "nominal_same_day_known_cost_positive_share"),
            0.75,
        )
        self.assertEqual(
            result.item(0, "nominal_overnight_known_cost_positive_share"),
            0.25,
        )

    def test_fact_checkpoint_is_branded_with_supplement_runner(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = FrozenConvergencePaths(work_root=root / "work")
            context = _UpstreamContext(
                marker={},
                boundary_marker={},
                publication={},
                sessions=("20260126",),
                primary_sessions=(),
                registry_sha256="registry",
                anchor_model_id="time_ewma_15s",
            )
            atomic_publish_checkpoint(
                root / "work" / "preflight",
                phase="test_preflight",
                date="test",
                registry_sha256="registry",
                source_commit="commit",
                input_records=[],
                parameters={},
                artifact_values={"fixture.json": {"complete": True}},
                runner_version=RUNNER_VERSION,
            )
            marker = _ensure_fact_date(
                paths,
                context,
                pl.DataFrame(schema={"Date": pl.String}),
                "20260126",
                source_commit="commit",
                initial_git_state={},
                canonical=False,
                resume=True,
            )
            self.assertEqual(marker["runner_version"], RUNNER_VERSION)
            self.assertTrue(
                (_fact_partition(paths, "20260126") / "post_touch_facts.parquet").is_file()
            )


if __name__ == "__main__":
    unittest.main()
