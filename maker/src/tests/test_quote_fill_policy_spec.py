"""Focused contracts for the frozen seven-policy specification table."""

from __future__ import annotations

import unittest
from dataclasses import replace

import polars as pl

from ..quote_fill.policy_spec import (
    CONVERGENCE_REFERENCE_SEMANTICS,
    POLICY_IDS,
    TOD_BUCKETS,
    PolicySpec,
    build_policy_spec_table,
    policy_spec_table_sha256,
    policy_specs_from_table,
    policy_specs_to_table,
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
                    "effective_source_asof_date": ASOF,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _convergence() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for tod in TOD_BUCKETS:
        for quantile in (50, 80, 95):
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "anchor_model_id": "time_ewma_15s",
                    "candidate_id": "Q2_trail20_date_equal",
                    "tod_bucket": tod,
                    "boundary_quantile": quantile,
                    "convergence_candidate_id": "C0_center",
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "effective_threshold_distance_bp": 0.0,
                    "effective_supported": True,
                    "effective_source_asof_date": ASOF,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


class PolicySpecTest(unittest.TestCase):
    def test_builds_exact_seven_policy_grid_and_provenance(self) -> None:
        result = build_policy_spec_table(_mother(), _entry_lookup(), _convergence())

        self.assertEqual(result.height, len(TOD_BUCKETS) * len(POLICY_IDS))
        for cell in result.partition_by("entry_tod_bucket"):
            self.assertEqual(set(cell["policy_id"]), set(POLICY_IDS))

        q95 = result.filter(pl.col("policy_id") == "q95")
        self.assertTrue(((q95["upper_distance_bp"] - 31.5).abs() < 1e-12).all())
        self.assertTrue(((q95["lower_distance_bp"] - 0.0).abs() < 1e-12).all())
        self.assertEqual(
            q95["upper_source_id"].unique().to_list(), ["Q2_trail20_date_equal"]
        )
        self.assertEqual(q95["lower_source_id"].unique().to_list(), ["C0_center"])
        self.assertTrue((q95["combined_source_asof_date"] < q95["Date"]).all())

        fixed = result.filter(pl.col("policy_id") == "fixed25")
        self.assertTrue(((fixed["upper_distance_bp"] - 25.0).abs() < 1e-12).all())
        self.assertTrue(((fixed["lower_distance_bp"] - 25.0).abs() < 1e-12).all())
        self.assertEqual(
            fixed["upper_source_id"].unique().to_list(), ["constant_bp:25"]
        )
        self.assertEqual(fixed["upper_source_asof_date"].null_count(), fixed.height)

    def test_unsupported_missing_and_lookahead_fail_closed(self) -> None:
        unsupported = _convergence().with_columns(
            pl.when(
                (pl.col("tod_bucket") == TOD_BUCKETS[0])
                & (pl.col("boundary_quantile") == 50)
            )
            .then(False)
            .otherwise(pl.col("effective_supported"))
            .alias("effective_supported")
        )
        with self.assertRaisesRegex(ValueError, "missing, unsupported"):
            build_policy_spec_table(_mother(), _entry_lookup(), unsupported)

        missing = _entry_lookup().filter(
            ~(
                (pl.col("tod_bucket") == TOD_BUCKETS[0])
                & (pl.col("boundary_quantile") == 50)
            )
        )
        with self.assertRaisesRegex(ValueError, "missing, unsupported"):
            build_policy_spec_table(_mother(), missing, _convergence())

        lookahead = _entry_lookup().with_columns(
            pl.lit(DATE).alias("effective_source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            build_policy_spec_table(_mother(), lookahead, _convergence())

    def test_roundtrip_and_hash_are_deterministic(self) -> None:
        frame = build_policy_spec_table(_mother(), _entry_lookup(), _convergence())
        specs = policy_specs_from_table(frame)
        rebuilt = policy_specs_to_table(tuple(reversed(specs)))

        self.assertTrue(frame.equals(rebuilt))
        self.assertEqual(
            policy_spec_table_sha256(frame),
            policy_spec_table_sha256(frame.reverse()),
        )
        self.assertEqual(specs[0], PolicySpec.from_dict(specs[0].to_dict()))
        self.assertEqual(
            specs[0].deterministic_sha256,
            PolicySpec.from_dict(specs[0].to_dict()).deterministic_sha256,
        )

        q = next(spec for spec in specs if spec.kind == "quantile")
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            replace(q, combined_source_asof_date=q.Date)

    def test_duplicate_mother_and_wrong_convergence_candidate_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "duplicated"):
            build_policy_spec_table(
                pl.concat([_mother(), _mother()]),
                _entry_lookup(),
                _convergence(),
            )

        wrong_candidate = _convergence().with_columns(
            pl.lit("Q9_not_selected").alias("candidate_id")
        )
        with self.assertRaisesRegex(ValueError, "missing, unsupported"):
            build_policy_spec_table(
                _mother(),
                _entry_lookup(),
                wrong_candidate,
            )


if __name__ == "__main__":
    unittest.main()
