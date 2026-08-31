"""Focused contracts for the frozen cost-aware S1 scenario grid."""

from __future__ import annotations

import unittest

import polars as pl

from ..quote_fill.policy_spec import CONVERGENCE_REFERENCE_SEMANTICS, TOD_BUCKETS
from ..quote_fill.s1_scenario_spec import (
    LOWER_C0,
    LOWER_C2,
    LOWER_C3,
    SCENARIO_GRID_SHA256,
    SCENARIO_IDS,
    S1ScenarioSpec,
    build_s1_scenario_spec_table,
    s1_scenario_spec_table_sha256,
    s1_scenario_specs_from_table,
    s1_scenario_specs_to_table,
    validate_s1_scenario_spec_table,
)

DATE = "20260505"
VALUE_CODE = "2330"
QUOTE_CODE = "CDFE6"
ASOF = "20260504"


def _mother() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "s1_primary": [True],
            "anchor_model_id": ["time_ewma_15s"],
            "contains_target_day_outcome": [False],
        }
    )


def _entry_lookup() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for tod in TOD_BUCKETS:
        for quantile, distance in ((50, 12.5), (80, 20.0), (95, 31.5)):
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "anchor_model_id": "time_ewma_15s",
                    "tod_bucket": tod,
                    "boundary_quantile": quantile,
                    "side": "positive",
                    "candidate_id": "Q2_trail20_date_equal",
                    "effective_supported": True,
                    "boundary_distance_bp": distance,
                    "effective_candidate_id": "Q2_trail20_date_equal",
                    "effective_source_asof_date": ASOF,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _convergence() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    combinations = (
        (95, LOWER_C0, 0.0),
        (95, LOWER_C2, 4.0),
        (80, LOWER_C0, 0.0),
        (50, LOWER_C3, 7.0),
    )
    for tod in TOD_BUCKETS:
        for quantile, candidate, distance in combinations:
            unsupported = (
                tod == TOD_BUCKETS[0]
                and quantile == 95
                and candidate == LOWER_C2
            )
            is_c0 = candidate == LOWER_C0
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "anchor_model_id": "time_ewma_15s",
                    "candidate_id": "Q2_trail20_date_equal",
                    "tod_bucket": tod,
                    "boundary_quantile": quantile,
                    "convergence_candidate_id": candidate,
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "effective_threshold_distance_bp": (
                        None if unsupported else distance
                    ),
                    "effective_supported": not unsupported,
                    "effective_lookup_id": (
                        None
                        if unsupported
                        else "structural_zero"
                        if is_c0
                        else "trail20_date_equal"
                    ),
                    "effective_source_asof_date": None if unsupported else ASOF,
                    "fallback_used": False,
                    "fallback_reason": (
                        "primary=insufficient_completed_paths;fallback=unsupported"
                        if unsupported
                        else None
                    ),
                    "native_failure_reason": (
                        "insufficient_completed_paths" if unsupported else None
                    ),
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


class S1ScenarioSpecTest(unittest.TestCase):
    def test_builds_exact_grid_and_freezes_economic_primitives(self) -> None:
        result = build_s1_scenario_spec_table(
            _mother(), _entry_lookup(), _convergence()
        )

        self.assertEqual(result.height, len(TOD_BUCKETS) * len(SCENARIO_IDS))
        for cell in result.partition_by("entry_tod_bucket"):
            self.assertEqual(set(cell["scenario_id"]), set(SCENARIO_IDS))
        self.assertEqual(
            result["scenario_grid_sha256"].unique().to_list(),
            [SCENARIO_GRID_SHA256],
        )

        control = result.filter(pl.col("scenario_id") == "ctrl_q95_C0_ungated")
        self.assertTrue((control["cost_horizon"] == "ungated").all())
        self.assertTrue(control["safety_floor_bp"].is_null().all())
        self.assertFalse(control["economic_gate_enabled"].any())
        self.assertFalse(control["deployment_shortlist_eligible"].any())

        overnight = result.filter(pl.col("scenario_id") == "q95_C0_on_f0")
        self.assertTrue((overnight["cost_horizon"] == "overnight").all())
        self.assertTrue((overnight["safety_floor_bp"].abs() < 1e-12).all())
        self.assertTrue(overnight["economic_gate_enabled"].all())
        self.assertTrue(overnight["deployment_shortlist_eligible"].all())

        fixed = result.filter(
            pl.col("scenario_id") == "fixed20_sym20_on_f0"
        )
        self.assertTrue(((fixed["upper_distance_bp"] - 20.0).abs() < 1e-12).all())
        self.assertTrue(((fixed["lower_distance_bp"] - 20.0).abs() < 1e-12).all())
        self.assertTrue(fixed["lookup_supported"].all())
        self.assertEqual(fixed["upper_source_id"].unique().to_list(), ["constant_bp:20"])

    def test_unsupported_lower_is_retained_without_cross_lower_fallback(self) -> None:
        result = build_s1_scenario_spec_table(
            _mother(), _entry_lookup(), _convergence()
        )
        row = result.filter(
            (pl.col("scenario_id") == "q95_C2_sd_f5")
            & (pl.col("entry_tod_bucket") == TOD_BUCKETS[0])
        ).row(0, named=True)

        self.assertFalse(row["lower_lookup_supported"])
        self.assertFalse(row["lookup_supported"])
        self.assertEqual(row["lookup_support_reason"], "lower_unsupported")
        self.assertEqual(row["lower_source_id"], LOWER_C2)
        self.assertIsNone(row["lower_distance_bp"])
        self.assertIsNone(row["lower_effective_source_id"])
        self.assertIn("insufficient_completed_paths", row["lower_fallback_reason"])

        same_cell_control = result.filter(
            (pl.col("scenario_id") == "ctrl_q95_C0_ungated")
            & (pl.col("entry_tod_bucket") == TOD_BUCKETS[0])
        ).row(0, named=True)
        self.assertTrue(same_cell_control["lookup_supported"])
        self.assertEqual(same_cell_control["lower_source_id"], LOWER_C0)
        self.assertAlmostEqual(same_cell_control["lower_distance_bp"], 0.0)

    def test_missing_lookup_is_retained_and_duplicates_fail(self) -> None:
        missing = _convergence().filter(
            ~(
                (pl.col("tod_bucket") == TOD_BUCKETS[-1])
                & (pl.col("boundary_quantile") == 50)
                & (pl.col("convergence_candidate_id") == LOWER_C3)
            )
        )
        result = build_s1_scenario_spec_table(_mother(), _entry_lookup(), missing)
        retained = result.filter(
            (pl.col("scenario_id") == "q50_C3_sd_f5")
            & (pl.col("entry_tod_bucket") == TOD_BUCKETS[-1])
        ).row(0, named=True)
        self.assertFalse(retained["lookup_supported"])
        self.assertEqual(retained["lookup_support_reason"], "lower_missing")
        self.assertEqual(result.height, len(TOD_BUCKETS) * len(SCENARIO_IDS))

        with self.assertRaisesRegex(ValueError, "duplicated"):
            build_s1_scenario_spec_table(
                _mother(),
                _entry_lookup(),
                pl.concat([_convergence(), _convergence().head(1)]),
            )

    def test_provenance_lookahead_and_semantics_fail_closed(self) -> None:
        lookahead = _entry_lookup().with_columns(
            pl.lit(DATE).alias("effective_source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            build_s1_scenario_spec_table(_mother(), lookahead, _convergence())

        wrong_semantics = _convergence().with_columns(
            pl.lit("moving_anchor_invalid").alias(
                "convergence_reference_semantics"
            )
        )
        with self.assertRaisesRegex(ValueError, "frozen-anchor semantics"):
            build_s1_scenario_spec_table(
                _mother(), _entry_lookup(), wrong_semantics
            )

        target_outcome = _convergence().with_columns(
            pl.lit(True).alias("contains_target_day_outcome")
        )
        with self.assertRaisesRegex(ValueError, "target-day outcomes"):
            build_s1_scenario_spec_table(
                _mother(), _entry_lookup(), target_outcome
            )

    def test_roundtrip_schema_coverage_and_hash_validation(self) -> None:
        frame = build_s1_scenario_spec_table(
            _mother(), _entry_lookup(), _convergence()
        )
        digest = s1_scenario_spec_table_sha256(frame)
        validate_s1_scenario_spec_table(frame, expected_sha256=digest)
        self.assertEqual(digest, s1_scenario_spec_table_sha256(frame.reverse()))

        specs = s1_scenario_specs_from_table(frame)
        rebuilt = s1_scenario_specs_to_table(tuple(reversed(specs)))
        self.assertTrue(frame.equals(rebuilt))
        self.assertEqual(
            specs[0], S1ScenarioSpec.from_dict(specs[0].to_dict())
        )
        unsupported = next(
            value
            for value in specs
            if value.scenario_id == "q95_C2_sd_f5"
            and value.entry_tod_bucket == TOD_BUCKETS[0]
        )
        self.assertEqual(unsupported.policy_id, unsupported.scenario_id)
        self.assertEqual(unsupported.kind, unsupported.policy_kind)
        self.assertIn("insufficient_completed_paths", unsupported.fallback_reason)

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_s1_scenario_spec_table(frame, expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "exact seven-scenario"):
            validate_s1_scenario_spec_table(frame.slice(1))

        changed_control = frame.with_columns(
            pl.when(pl.col("scenario_id") == "ctrl_q95_C0_ungated")
            .then(True)
            .otherwise(pl.col("deployment_shortlist_eligible"))
            .alias("deployment_shortlist_eligible")
        )
        with self.assertRaisesRegex(ValueError, "scenario definition mismatch"):
            validate_s1_scenario_spec_table(changed_control)


if __name__ == "__main__":
    unittest.main()
