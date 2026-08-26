"""Tests for the immutable C2/C3 conditional-geometry supplement."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.foundation_convergence_lookup import (
    CONVERGENCE_PREDICTION_SCHEMA,
)
from maker.src.quote_fill.foundation_geometry import (
    build_reference_policy_geometry,
)
from maker.src.quote_fill.foundation_selected_geometry import (
    CONDITIONAL_CANDIDATES,
    CONDITIONAL_GEOMETRY_VERSION,
    TOD_BUCKETS,
    build_conditional_lookup_geometry,
)


class FoundationConditionalGeometryTest(unittest.TestCase):
    def test_keeps_full_support_grid_and_builds_only_effective_geometry(self) -> None:
        result = build_conditional_lookup_geometry(
            _mother(),
            _mapping(),
            _boundaries(),
            _convergence_predictions(),
        )

        self.assertEqual(result.support_audit.height, 4 * 3 * 3)
        self.assertEqual(result.geometry_long.height, 4 * 3 * 3 - 1)
        self.assertEqual(result.summary_overall.height, 9)
        self.assertEqual(result.summary_by_month.height, 9)
        self.assertEqual(result.summary_by_tod.height, 4 * 9)
        self.assertEqual(result.support_summary.height, 9 + 9 + 4 * 9)
        self.assertFalse(result.geometry_long["actionable_execution"].any())
        self.assertFalse(result.geometry_long["ev_ready"].any())
        self.assertFalse(result.geometry_long["contains_target_day_outcome"].any())
        self.assertEqual(
            result.geometry_long["conditional_geometry_version"].unique().to_list(),
            [CONDITIONAL_GEOMETRY_VERSION],
        )

        unsupported = result.support_audit.filter(
            (pl.col("tod_bucket") == TOD_BUCKETS[0])
            & (pl.col("boundary_quantile") == 95)
            & (
                pl.col("convergence_candidate_id")
                == "C2_conditional_reach80"
            )
        ).row(0, named=True)
        self.assertFalse(unsupported["conditional_supported"])
        self.assertIsNone(unsupported["source_asof_date"])

        zero = result.geometry_long.filter(
            (pl.col("tod_bucket") == TOD_BUCKETS[0])
            & (pl.col("boundary_quantile") == 50)
            & (pl.col("convergence_candidate_id") == "C0_center")
        ).row(0, named=True)
        self.assertAlmostEqual(zero["lower_distance_bp"], 0.0)
        self.assertAlmostEqual(zero["nominal_band_bp"], zero["upper_distance_bp"])

        fallback = result.geometry_long.filter(
            (pl.col("tod_bucket") == TOD_BUCKETS[0])
            & (pl.col("boundary_quantile") == 80)
            & (
                pl.col("convergence_candidate_id")
                == "C3_conditional_reach50"
            )
        ).row(0, named=True)
        self.assertTrue(fallback["fallback_used"])
        self.assertEqual(fallback["effective_lookup_id"], "trail60_date_equal")
        self.assertEqual(fallback["source_asof_date"], "20260504")
        self.assertAlmostEqual(
            fallback["nominal_same_day_known_cost_margin_bp"],
            fallback["nominal_band_bp"] - fallback["same_day_reference_cost_bp"],
        )

    def test_rejects_incomplete_or_unsafe_effective_predictions(self) -> None:
        incomplete = _convergence_predictions().head(
            _convergence_predictions().height - 1
        )
        with self.assertRaisesRegex(ValueError, "full mother"):
            build_conditional_lookup_geometry(
                _mother(),
                _mapping(),
                _boundaries(),
                incomplete,
            )

        unsafe = _convergence_predictions().with_columns(
            pl.when(pl.col("effective_supported"))
            .then(pl.col("Date"))
            .otherwise(pl.col("effective_source_asof_date"))
            .alias("effective_source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "strictly before Date"):
            build_conditional_lookup_geometry(
                _mother(),
                _mapping(),
                _boundaries(),
                unsafe,
            )

        dynamic = _convergence_predictions().with_columns(
            pl.lit("dynamic_anchor_sensitivity").alias(
                "convergence_reference_semantics"
            )
        )
        with self.assertRaisesRegex(ValueError, "inconsistent support"):
            build_conditional_lookup_geometry(
                _mother(),
                _mapping(),
                _boundaries(),
                dynamic,
            )

    def test_public_reference_primitive_allows_zero_lower_only_explicitly(
        self,
    ) -> None:
        policy = pl.DataFrame(
            {
                "Date": ["20260505"],
                "ValueCode": ["2330"],
                "QuoteCode": ["2330F"],
                "policy_id": ["q50__C2"],
                "policy_kind": ["conditional"],
                "policy_order": [0],
                "source_asof_date": ["20260504"],
                "contains_target_day_outcome": [False],
                "execution_safe_snapshot": [True],
                "spot_ref_price": [100.0],
                "fut_ref_price": [100.0],
                "contract_size": [2_000.0],
                "upper_distance_bp": [20.0],
                "lower_distance_bp": [0.0],
                "nominal_band_bp": [20.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "positive and additive"):
            build_reference_policy_geometry(policy)

        result = build_reference_policy_geometry(
            policy,
            allow_zero_lower=True,
        )
        self.assertEqual(result.height, 1)
        self.assertAlmostEqual(result.item(0, "lower_distance_bp"), 0.0)
        self.assertFalse(result.item(0, "actionable_execution"))


def _mother() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260505"],
            "ValueCode": ["2330"],
            "QuoteCode": ["2330F"],
            "anchor_model_id": ["time_ewma_15s"],
            "candidate_id": ["Q2_trail20_date_equal"],
            "s1_primary": [True],
            "boundary_source_asof_date": ["20260504"],
            "liquidity_source_asof_date": ["20260504"],
            "contains_target_day_outcome": [False],
        }
    )


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260505"],
            "ValueCode": ["2330"],
            "QuoteCode": ["2330F"],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2_000.0],
        }
    )


def _boundaries() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    distances = {50: 10.0, 80: 20.0, 95: 30.0}
    for tod_bucket in TOD_BUCKETS:
        for quantile, distance in distances.items():
            for side in ("positive", "negative"):
                rows.append(
                    {
                        "Date": "20260505",
                        "ValueCode": "2330",
                        "QuoteCode": "2330F",
                        "anchor_model_id": "time_ewma_15s",
                        "candidate_id": "Q2_trail20_date_equal",
                        "tod_bucket": tod_bucket,
                        "boundary_quantile": quantile,
                        "side": side,
                        "boundary_distance_bp": distance,
                        "source_asof_date": "20260504",
                        "effective_supported": True,
                        "contains_target_day_outcome": False,
                    }
                )
    return pl.from_dicts(rows, infer_schema_length=None)


def _convergence_predictions() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for tod_bucket in TOD_BUCKETS:
        for quantile in (50, 80, 95):
            for candidate in CONDITIONAL_CANDIDATES:
                is_c0 = candidate == "C0_center"
                is_c2 = candidate == "C2_conditional_reach80"
                unsupported = (
                    tod_bucket == TOD_BUCKETS[0]
                    and quantile == 95
                    and is_c2
                )
                fallback = (
                    tod_bucket == TOD_BUCKETS[0]
                    and quantile == 80
                    and candidate == "C3_conditional_reach50"
                )
                lower = 0.0 if is_c0 else 2.0 if is_c2 else 5.0
                effective = not unsupported
                native = is_c0 or (effective and not fallback)
                rows.append(
                    {
                        "Date": "20260505",
                        "ValueCode": "2330",
                        "QuoteCode": "2330F",
                        "anchor_model_id": "time_ewma_15s",
                        "candidate_id": "Q2_trail20_date_equal",
                        "tod_bucket": tod_bucket,
                        "boundary_quantile": quantile,
                        "convergence_candidate_id": candidate,
                        "convergence_reference_semantics": (
                            "frozen_anchor_at_upper_touch"
                        ),
                        "lookup_id": (
                            "structural_zero" if is_c0 else "trail20_date_equal"
                        ),
                        "lookback_sessions": 0 if is_c0 else 20,
                        "minimum_completed_dates": 0 if is_c0 else 15,
                        "minimum_completed_paths": 0 if is_c0 else 50,
                        "target_reach_probability": (
                            None if is_c0 else 0.80 if is_c2 else 0.50
                        ),
                        "floor_quantile_probability": (
                            None if is_c0 else 0.20 if is_c2 else 0.50
                        ),
                        "source_asof_date": "20260504",
                        "history_start_date": "20260401",
                        "history_end_date": "20260504",
                        "history_observation_end_date": "20260504",
                        "selected_sessions": 20,
                        "completed_history_dates": 20,
                        "observable_started": 100,
                        "completed_paths": 100,
                        "right_censored_paths": 0,
                        "completed_point_bp": lower,
                        "identified_lower_bp": lower,
                        "identified_upper_bp": lower,
                        "identified_upper_unbounded": False,
                        "native_threshold_distance_bp": lower if native else None,
                        "native_supported": native,
                        "native_failure_reason": (
                            None if native else "insufficient_completed_paths"
                        ),
                        "effective_threshold_distance_bp": (
                            lower if effective else None
                        ),
                        "effective_supported": effective,
                        "effective_lookup_id": (
                            "structural_zero"
                            if is_c0
                            else "trail60_date_equal"
                            if fallback
                            else "trail20_date_equal"
                            if effective
                            else None
                        ),
                        "effective_source_asof_date": (
                            "20260504" if effective else None
                        ),
                        "fallback_lookup_id": (
                            None if is_c0 else "trail60_date_equal"
                        ),
                        "fallback_used": fallback,
                        "fallback_reason": (
                            "insufficient_completed_paths"
                            if fallback or unsupported
                            else None
                        ),
                        "contains_target_day_outcome": False,
                    }
                )
    return pl.from_dicts(
        rows,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    )


if __name__ == "__main__":
    unittest.main()
