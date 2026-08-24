"""Tests for trailing-session adaptive boundary snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from maker.src.quote_width.rolling import (
    RollingBoundaryConfig,
    ROLLING_BOUNDARY_SCHEMA_VERSION,
    _json_sha256,
    _make_source_provenance,
    build_rolling_boundary_snapshots,
    load_rolling_boundary_snapshots,
    validate_rolling_boundary_artifact,
    write_rolling_boundary_snapshots,
)


class RollingBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = ["20260101", "20260102", "20260103", "20260104"]
        self.mapping = pl.DataFrame(
            {
                "Date": self.sessions,
                "ValueCode": ["2317"] * 4,
                "QuoteCode": ["DHFA6", "DHFA6", "DHFB6", "DHFB6"],
            }
        )
        rows: list[dict[str, object]] = []
        amplitudes = {
            "20260101": (10.0, 20.0),
            "20260102": (30.0, 40.0),
            "20260103": (1000.0, 1000.0),
        }
        for date, (positive, negative) in amplitudes.items():
            rows.extend(
                [
                    {
                        "Date": date,
                        "ValueCode": "2317",
                        "side": "positive",
                        "amplitude_bp": positive,
                        "completed": True,
                    },
                    {
                        "Date": date,
                        "ValueCode": "2317",
                        "side": "negative",
                        "amplitude_bp": negative,
                        "completed": True,
                    },
                ]
            )
        self.excursions = pl.DataFrame(rows)
        self.config = RollingBoundaryConfig(
            lookback_sessions=2,
            min_history_sessions=2,
            min_excursion_history_sessions_per_side=2,
            min_completed_excursions_per_side=2,
            quantiles=(80,),
            parameter_version="test",
        )

    def test_target_day_is_excluded_and_contract_is_target_contract(self) -> None:
        result = build_rolling_boundary_snapshots(
            self.excursions,
            self.mapping,
            self.sessions,
            self.config,
        )
        target = result.filter(pl.col("Date") == "20260103").row(0, named=True)
        self.assertEqual(target["QuoteCode"], "DHFB6")
        self.assertEqual(target["train_start_date"], "20260101")
        self.assertEqual(target["train_end_date"], "20260102")
        self.assertEqual(target["source_asof_date"], "20260102")
        self.assertEqual(target["upper_distance_bp"], 30.0)
        self.assertEqual(target["lower_distance_bp"], 40.0)
        self.assertTrue(target["adaptive_parameter_valid"])
        self.assertTrue(target["execution_safe_snapshot"])
        self.assertFalse(target["contains_target_day_outcome"])
        self.assertNotIn("trading_turnover", result.columns)

        changed = self.excursions.with_columns(
            pl.when(pl.col("Date") == "20260103")
            .then(pl.lit(9999.0))
            .otherwise(pl.col("amplitude_bp"))
            .alias("amplitude_bp")
        )
        changed_result = build_rolling_boundary_snapshots(
            changed,
            self.mapping,
            self.sessions,
            self.config,
        )
        changed_target = changed_result.filter(
            pl.col("Date") == "20260103"
        ).row(0, named=True)
        self.assertEqual(
            target["upper_distance_bp"], changed_target["upper_distance_bp"]
        )
        self.assertEqual(
            target["lower_distance_bp"], changed_target["lower_distance_bp"]
        )

    def test_window_rolls_by_market_sessions(self) -> None:
        result = build_rolling_boundary_snapshots(
            self.excursions,
            self.mapping,
            self.sessions,
            self.config,
        )
        target = result.filter(pl.col("Date") == "20260104").row(0, named=True)
        self.assertEqual(target["train_start_date"], "20260102")
        self.assertEqual(target["train_end_date"], "20260103")
        self.assertEqual(target["upper_distance_bp"], 1000.0)
        self.assertEqual(target["lower_distance_bp"], 1000.0)

    def test_target_day_recomputes_stale_tick_columns_and_persists_lineage(self) -> None:
        sessions = ["20260703", "20260706"]
        mapping = pl.DataFrame(
            {
                "Date": sessions,
                "ValueCode": ["2317", "2317"],
                "QuoteCode": ["DHFN6", "DHFN6"],
                "spot_ref_price": [2130.0, 2130.0],
                "fut_ref_price": [2130.0, 2130.0],
                # Simulate an old persisted mapping.  Presence must not bypass
                # recomputation under the current versioned ladder.
                "target_ref_future_tick_bp": [999.0, 999.0],
                "target_ref_future_ask_tick_bp": [999.0, 999.0],
                "target_ref_spot_bid_tick_bp": [999.0, 999.0],
                "price_ladder_version": ["stale", "stale"],
            }
        )
        excursions = pl.DataFrame(
            {
                "Date": ["20260703", "20260703"],
                "ValueCode": ["2317", "2317"],
                "side": ["positive", "negative"],
                "amplitude_bp": [20.0, 30.0],
                "completed": [True, True],
            }
        )
        config = RollingBoundaryConfig(
            lookback_sessions=1,
            min_history_sessions=1,
            min_excursion_history_sessions_per_side=1,
            min_completed_excursions_per_side=1,
            quantiles=(80,),
            parameter_version="test",
        )
        result = build_rolling_boundary_snapshots(
            excursions,
            mapping,
            sessions,
            config,
        )
        before = result.filter(pl.col("Date") == "20260703").row(0, named=True)
        after = result.filter(pl.col("Date") == "20260706").row(0, named=True)
        self.assertAlmostEqual(
            before["target_ref_future_tick_bp"],
            10_000 * 5 / 2130,
        )
        self.assertAlmostEqual(
            after["target_ref_future_tick_bp"],
            10_000 * 1 / 2130,
        )
        self.assertEqual(
            after["target_ref_future_tick_bp"],
            after["target_ref_future_ask_tick_bp"],
        )
        self.assertEqual(after["price_ladder_version"], PRICE_LADDER_VERSION)
        self.assertEqual(
            after["future_one_dollar_tick_effective_date"],
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
        )
        self.assertAlmostEqual(
            after["upper_distance_future_ticks"],
            20.0 / (10_000 / 2130),
        )

    def test_config_rejects_unimplemented_ladder_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "price_ladder_version"):
            RollingBoundaryConfig(price_ladder_version="legacy").validate()

    def test_atomic_publication_marker_binds_shape_hash_source_config_and_code(self) -> None:
        snapshots = build_rolling_boundary_snapshots(
            self.excursions,
            self.mapping,
            self.sessions,
            self.config,
        )
        source = _make_source_provenance(
            "synthetic_test_fixture",
            {"fixture_sha256": "a" * 64},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_rolling_boundary_snapshots(
                snapshots,
                output,
                self.config,
                source_provenance=source,
            )
            marker = validate_rolling_boundary_artifact(
                output / "rolling_boundary_snapshots.parquet"
            )
            self.assertTrue(marker["complete"])
            self.assertEqual(
                marker["schema_version"], ROLLING_BOUNDARY_SCHEMA_VERSION
            )
            artifact = marker["artifacts"]["rolling_boundary_snapshots.parquet"]
            self.assertEqual(artifact["rows"], snapshots.height)
            self.assertEqual(artifact["columns"], snapshots.width)
            self.assertEqual(artifact["column_names"], snapshots.columns)
            self.assertEqual(marker["source_provenance"], source)
            self.assertIn(
                "fair_mid/quote_churn.py",
                {entry["module"] for entry in marker["builder_code"]},
            )
            self.assertEqual(
                load_rolling_boundary_snapshots(
                    output / "rolling_boundary_snapshots.parquet"
                ).shape,
                snapshots.shape,
            )

            # A legacy artifact with no publication marker is never trusted.
            (output / "complete.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "publication is incomplete"):
                load_rolling_boundary_snapshots(
                    output / "rolling_boundary_snapshots.parquet"
                )

    def test_writer_rejects_stale_v1_row_lineage_before_marker(self) -> None:
        snapshots = build_rolling_boundary_snapshots(
            self.excursions,
            self.mapping,
            self.sessions,
            self.config,
        ).with_columns(pl.lit("legacy-v1").alias("price_ladder_version"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "row/config lineage"):
                write_rolling_boundary_snapshots(
                    snapshots,
                    output,
                    self.config,
                    source_provenance=_make_source_provenance(
                        "synthetic_test_fixture",
                        {"fixture_sha256": "b" * 64},
                    ),
                )
            self.assertFalse((output / "complete.json").exists())

    def test_loader_rejects_marker_bound_to_different_builder_code(self) -> None:
        snapshots = build_rolling_boundary_snapshots(
            self.excursions,
            self.mapping,
            self.sessions,
            self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_rolling_boundary_snapshots(
                snapshots,
                output,
                self.config,
                source_provenance=_make_source_provenance(
                    "synthetic_test_fixture",
                    {"fixture_sha256": "c" * 64},
                ),
            )
            marker_path = output / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["builder_code"][0]["sha256"] = "0" * 64
            marker["builder_code_sha256"] = _json_sha256(marker["builder_code"])
            marker.pop("marker_payload_sha256")
            marker["marker_payload_sha256"] = _json_sha256(marker)
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "builder code mismatch"):
                validate_rolling_boundary_artifact(
                    output / "rolling_boundary_snapshots.parquet"
                )


if __name__ == "__main__":
    unittest.main()
