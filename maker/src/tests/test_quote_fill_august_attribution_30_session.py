"""Focused contracts for the market-only S0 30-session challenger."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ..quote_fill.august_attribution_30_session_runner import (
    BOUNDARY_ARTIFACT,
    CHALLENGER_CONFIG,
    DEFAULT_CANONICAL_S0_ROOT,
    MEMBERSHIP_ARTIFACT,
    MONTHLY_ARTIFACT,
    PARAMETER_VERSION,
    PRODUCT_DAY_ARTIFACT,
    RUNNER_VERSION,
    SCHEMA_VERSION,
    _canonical_checks_from_payload,
    _input_inventory,
    _publish_bundle,
    _rolling_config_payload,
    _verify_bundle_artifacts,
    _verify_domain_frames,
    build_fixed_challenger_boundaries,
    build_product_day_sensitivity,
    summarize_common_membership_sensitivity,
    summarize_monthly_boundary_sensitivity,
    verify_bundle,
)
from ..quote_width.rolling import RollingBoundaryConfig


def _synthetic_sessions() -> list[str]:
    current = date(2026, 6, 1)
    end = date(2026, 8, 1)
    values: list[str] = []
    while current <= end:
        values.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return values


def _manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260701", "20260701", "20260801", "20260801"],
            "ValueCode": ["A", "B", "A", "B"],
            "QuoteCode": ["QA", "QB", "QA", "QB"],
        }
    ).sort(["Date", "ValueCode"])


def _canonical_coverage() -> pl.DataFrame:
    return _manifest().with_columns(
        pl.lit(10.0).alias("upper_distance_bp"),
        pl.lit(60, dtype=pl.Int64).alias("lookback_sessions"),
        pl.when(pl.col("Date") == "20260701")
        .then(pl.lit("20260630"))
        .otherwise(pl.lit("20260731"))
        .alias("source_asof_date"),
        pl.lit("rolling_60_session_empirical_v1").alias("parameter_version"),
    )


def _challenger_boundaries() -> pl.DataFrame:
    return _manifest().with_columns(
        pl.when(pl.col("ValueCode") == "A")
        .then(pl.lit(8.0))
        .otherwise(None)
        .alias("upper_distance_bp"),
        pl.when(pl.col("ValueCode") == "A")
        .then(pl.lit(8.0))
        .otherwise(None)
        .alias("lower_distance_bp"),
        (pl.col("ValueCode") == "A").alias("adaptive_parameter_valid"),
        (pl.col("ValueCode") == "A").alias("challenger_boundary_valid"),
        pl.when(pl.col("Date") == "20260701")
        .then(pl.lit("20260630"))
        .otherwise(pl.lit("20260731"))
        .alias("source_asof_date"),
        pl.when(pl.col("Date") == "20260701")
        .then(pl.lit("20260601"))
        .otherwise(pl.lit("20260702"))
        .alias("train_start_date"),
        pl.when(pl.col("Date") == "20260701")
        .then(pl.lit("20260630"))
        .otherwise(pl.lit("20260731"))
        .alias("train_end_date"),
        pl.lit(30, dtype=pl.Int64).alias("history_sessions_global"),
        pl.when(pl.col("ValueCode") == "A")
        .then(pl.lit(30, dtype=pl.Int64))
        .otherwise(pl.lit(19, dtype=pl.Int64))
        .alias("history_sessions_product"),
        pl.when(pl.col("ValueCode") == "A")
        .then(pl.lit(20, dtype=pl.Int64))
        .otherwise(pl.lit(19, dtype=pl.Int64))
        .alias("positive_history_dates"),
        pl.when(pl.col("ValueCode") == "A")
        .then(pl.lit(20, dtype=pl.Int64))
        .otherwise(pl.lit(19, dtype=pl.Int64))
        .alias("negative_history_dates"),
        pl.lit(200, dtype=pl.Int64).alias("positive_completed"),
        pl.lit(200, dtype=pl.Int64).alias("negative_completed"),
        pl.lit(95, dtype=pl.Int64).alias("boundary_quantile"),
        pl.lit(30, dtype=pl.Int64).alias("lookback_sessions"),
        pl.lit(20, dtype=pl.Int64).alias("minimum_history_sessions"),
        pl.lit(20, dtype=pl.Int64).alias(
            "minimum_excursion_history_sessions_per_side"
        ),
        pl.lit(PARAMETER_VERSION).alias("parameter_version"),
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
        pl.lit(False).alias("actionable_execution"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(True).alias("fixed_manifest_member"),
        pl.lit(True).alias("market_only"),
    )


def _market_excursions() -> pl.DataFrame:
    specs = [
        ("20260701", "A", "QA", 12.0),
        ("20260701", "A", "QA", 9.0),
        ("20260701", "B", "QB", 12.0),
        ("20260801", "A", "QA", 11.0),
        ("20260801", "A", "QA", 9.0),
        ("20260801", "B", "QB", 5.0),
    ]
    return pl.from_dicts(
        [
            {
                "excursion_id": f"exc-{index}",
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "analysis_window": "entry_primary",
                "primary_observable": True,
                "amplitude_bp": amplitude,
                "upper_distance_bp": 10.0,
                "touch_time_ns": 100 + index if amplitude >= 10.0 else None,
            }
            for index, (date, value_code, quote_code, amplitude) in enumerate(
                specs
            )
        ],
        infer_schema_length=None,
    )


def _synthetic_frames() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    manifest = _manifest()
    boundaries = _challenger_boundaries()
    product_days = build_product_day_sensitivity(
        manifest,
        _canonical_coverage(),
        _market_excursions(),
        boundaries,
    )
    monthly = summarize_monthly_boundary_sensitivity(product_days)
    membership = summarize_common_membership_sensitivity(
        product_days, expected_common_product_count=2
    )
    return boundaries, product_days, monthly, membership


class AugustAttributionThirtySessionTest(unittest.TestCase):
    def test_frozen_config_and_default_refuse_known_bad_v1(self) -> None:
        self.assertEqual(CHALLENGER_CONFIG.lookback_sessions, 30)
        self.assertEqual(CHALLENGER_CONFIG.min_history_sessions, 20)
        self.assertEqual(
            CHALLENGER_CONFIG.min_excursion_history_sessions_per_side, 20
        )
        self.assertEqual(
            CHALLENGER_CONFIG.min_completed_excursions_per_side, 100
        )
        self.assertEqual(CHALLENGER_CONFIG.quantiles, (95,))
        self.assertEqual(CHALLENGER_CONFIG.parameter_version, PARAMETER_VERSION)
        self.assertTrue(DEFAULT_CANONICAL_S0_ROOT.name.endswith("_v2"))

    def test_builder_is_d_minus_one_and_retains_invalid_manifest_rows(self) -> None:
        sessions = ["20260101", "20260102", "20260103"]
        mapping = pl.DataFrame(
            {
                "Date": [
                    "20260101",
                    "20260101",
                    "20260102",
                    "20260102",
                    "20260103",
                    "20260103",
                ],
                "ValueCode": ["A", "B", "A", "B", "A", "B"],
                "QuoteCode": ["QA", "QB", "QA", "QB", "QA", "QB"],
            }
        )
        excursions = pl.DataFrame(
            {
                "Date": [
                    "20260101",
                    "20260101",
                    "20260102",
                    "20260102",
                    "20260103",
                    "20260103",
                ],
                "ValueCode": ["A"] * 6,
                "side": ["positive", "negative"] * 3,
                "amplitude_bp": [10.0, 20.0, 30.0, 40.0, 9999.0, 9999.0],
                "completed": [True] * 6,
            }
        )
        manifest = mapping.filter(pl.col("Date") == "20260102").sort(
            ["Date", "ValueCode"]
        )
        config = RollingBoundaryConfig(
            lookback_sessions=1,
            min_history_sessions=1,
            min_excursion_history_sessions_per_side=1,
            min_completed_excursions_per_side=1,
            quantiles=(95,),
            parameter_version="synthetic_30_challenger",
        )
        result = build_fixed_challenger_boundaries(
            excursions,
            mapping,
            sessions,
            manifest,
            config=config,
        )
        self.assertEqual(result.height, 2)
        valid = result.filter(pl.col("ValueCode") == "A").row(0, named=True)
        invalid = result.filter(pl.col("ValueCode") == "B").row(0, named=True)
        self.assertEqual(valid["source_asof_date"], "20260101")
        self.assertEqual(valid["upper_distance_bp"], 10.0)
        self.assertTrue(valid["challenger_boundary_valid"])
        self.assertFalse(invalid["challenger_boundary_valid"])
        self.assertIsNone(invalid["upper_distance_bp"])

    def test_same_denominator_excludes_invalid_without_zero_filling(self) -> None:
        boundaries, product_days, monthly, membership = _synthetic_frames()
        invalid = product_days.filter(pl.col("ValueCode") == "B")
        self.assertEqual(invalid.height, 2)
        self.assertEqual(
            invalid["canonical_60_touches_common_denominator"].null_count(), 2
        )
        self.assertEqual(
            invalid["challenger_30_touches_common_denominator"].null_count(), 2
        )
        july = monthly.filter(pl.col("month") == "202607").row(0, named=True)
        august = monthly.filter(pl.col("month") == "202608").row(0, named=True)
        for row in (july, august):
            self.assertEqual(row["fixed_manifest_product_days"], 2)
            self.assertEqual(row["common_valid_product_days"], 1)
            self.assertEqual(row["challenger_invalid_product_days"], 1)
            self.assertEqual(row["excluded_observable_excursions"], 1)
            self.assertEqual(row["common_observable_excursions"], 2)
            self.assertEqual(row["canonical_60_touches"], 1)
            self.assertEqual(row["challenger_30_hypothetical_touches"], 2)
            self.assertEqual(row["canonical_60_touch_rate"], 0.5)
            self.assertEqual(
                row["challenger_30_hypothetical_touch_rate"], 1.0
            )
        self.assertEqual(
            product_days["canonical_60_touches_common_denominator"].dtype,
            pl.Int64,
        )
        self.assertEqual(membership["common_product_count"].unique().item(), 2)
        verification = _verify_domain_frames(
            boundaries,
            product_days,
            monthly,
            membership,
            sessions=_synthetic_sessions(),
            expected_product_days=4,
            expected_common_product_count=2,
        )
        self.assertTrue(verification["same_denominator_verified"])
        self.assertEqual(verification["challenger_invalid_product_days"], 2)

        changed_boundaries = boundaries.with_columns(
            pl.when(
                (pl.col("Date") == "20260701")
                & (pl.col("ValueCode") == "A")
            )
            .then(pl.col("upper_distance_bp") + 1.0)
            .otherwise(pl.col("upper_distance_bp"))
            .alias("upper_distance_bp")
        )
        with self.assertRaisesRegex(ValueError, "projection drift"):
            _verify_domain_frames(
                changed_boundaries,
                product_days,
                monthly,
                membership,
                sessions=_synthetic_sessions(),
                expected_product_days=4,
                expected_common_product_count=2,
            )

        zero_filled = product_days.with_columns(
            pl.col("canonical_60_touches_common_denominator").fill_null(0)
        )
        with self.assertRaisesRegex(ValueError, "silently treated as zero"):
            summarize_monthly_boundary_sensitivity(zero_filled)

        stale_d_minus_one = boundaries.with_columns(
            pl.when(pl.col("Date") == "20260701")
            .then(pl.lit("20260629"))
            .otherwise(pl.col("source_asof_date"))
            .alias("source_asof_date"),
            pl.when(pl.col("Date") == "20260701")
            .then(pl.lit("20260629"))
            .otherwise(pl.col("train_end_date"))
            .alias("train_end_date"),
        )
        with self.assertRaisesRegex(ValueError, "exact D-1"):
            _verify_domain_frames(
                stale_d_minus_one,
                product_days,
                monthly,
                membership,
                sessions=_synthetic_sessions(),
                expected_product_days=4,
                expected_common_product_count=2,
            )

    def test_atomic_bundle_hashes_and_recomputes_summaries(self) -> None:
        boundaries, product_days, monthly, membership = _synthetic_frames()
        verification = _verify_domain_frames(
            boundaries,
            product_days,
            monthly,
            membership,
            sessions=_synthetic_sessions(),
            expected_product_days=4,
            expected_common_product_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("synthetic\n", encoding="utf-8")
            destination = root / "bundle"
            run_config = {
                "runner_version": RUNNER_VERSION,
                "schema_version": SCHEMA_VERSION,
                "market_only": True,
                "scheduler_included": False,
                "queue_included": False,
                "makerfill_included": False,
                "shortlist_eligible": False,
                "rolling_boundary_config": _rolling_config_payload(
                    CHALLENGER_CONFIG
                ),
                "fixed_manifest": {
                    "product_days": 4,
                    "jul_aug_common_product_count": 2,
                },
                "history_calendar": {
                    "session_count": len(_synthetic_sessions()),
                    "first_date": _synthetic_sessions()[0],
                    "last_date": _synthetic_sessions()[-1],
                    "dates": _synthetic_sessions(),
                    "source_file_sha256": "0" * 64,
                },
                "canonical_s0_source": {},
                "git": {
                    "commit": "synthetic-debug",
                    "dirty": True,
                    "status": "synthetic",
                },
                "focused_tests": {"status": "skipped_debug"},
                "input_inventory": _input_inventory([source]),
            }
            run_config["canonical_checks"] = _canonical_checks_from_payload(
                run_config,
                verification,
            )
            _publish_bundle(
                destination,
                {
                    BOUNDARY_ARTIFACT: boundaries,
                    PRODUCT_DAY_ARTIFACT: product_days,
                    MONTHLY_ARTIFACT: monthly,
                    MEMBERSHIP_ARTIFACT: membership,
                },
                run_config,
                verification,
                elapsed_seconds=0.0,
            )
            with self.assertRaises(ValueError):
                verify_bundle(destination, verify_inputs=True)
            result = _verify_bundle_artifacts(
                destination,
                verify_inputs=True,
                require_canonical=False,
            )
            self.assertEqual(result["status"], "pass")
            self.assertFalse(result["canonical_eligible"])
            self.assertTrue(result["input_content_rehashed"])
            self.assertFalse(any(root.glob(".bundle.tmp-*")))

            with (destination / MONTHLY_ARTIFACT).open("a", encoding="utf-8") as out:
                out.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "metadata drift"):
                _verify_bundle_artifacts(
                    destination,
                    verify_inputs=False,
                    require_canonical=False,
                )


if __name__ == "__main__":
    unittest.main()
