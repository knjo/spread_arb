from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill import interim_portfolio_snapshot as snapshot


class InterimPortfolioSnapshotTest(unittest.TestCase):
    def _build_bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        exit_root = (root / "formal-exit").resolve()
        entry_root = (root / "formal-entry").resolve()
        rows: list[dict[str, object]] = []
        for entry_route in snapshot.ENTRY_ROUTES:
            for quantile in (50, 80, 95):
                for rule in ("frozen_center", "frozen_lower"):
                    for exit_route in (
                        "future_bid_spot_taker",
                        "spot_ask_future_taker",
                    ):
                        rows.append(
                            {
                                "Date": "20260102",
                                "entry_route": entry_route,
                                "boundary_quantile": quantile,
                                "exit_rule_id": rule,
                                "exit_route": exit_route,
                                "spot_shares": 2_000,
                                "future_contracts": 1,
                                "one_way_spot_leg_entry_price_notional_twd": (
                                    100_000.0
                                ),
                                "nominal_instant_cancel_v0_branch": (
                                    "flat_same_day"
                                ),
                                "branch_status": "flat_same_day",
                                "gross_cycle_pnl_twd": 1_000.0,
                            }
                        )
        physical = pl.from_dicts(rows, infer_schema_length=None)
        relative_marker = "Date=20260102/ValueCode=1101/complete.json"
        exit_partition = exit_root / "Date=20260102" / "ValueCode=1101"
        entry_partition = entry_root / "Date=20260102" / "ValueCode=1101"
        exit_partition.mkdir(parents=True)
        entry_partition.mkdir(parents=True)
        pl.DataFrame(
            {"Date": ["20260102"], "ValueCode": ["1101"]}
        ).write_parquet(entry_root / snapshot.ENTRY_MANIFEST)
        action_path = entry_partition / snapshot.ENTRY_ACTION_ARTIFACT
        position_path = exit_partition / snapshot.POSITION_ARTIFACT
        pl.DataFrame({"action": [1]}).write_parquet(action_path)
        pl.DataFrame({"position": list(range(24))}).write_parquet(position_path)
        action_sha256 = snapshot._file_sha256(action_path)
        position_sha256 = snapshot._file_sha256(position_path)
        exit_config = {
            "source": {
                "action_source": {
                    "artifact": snapshot.ENTRY_ACTION_ARTIFACT,
                    "sha256": action_sha256,
                }
            }
        }
        exit_config_sha256 = snapshot._canonical_sha256(exit_config)
        exit_marker = {
            "complete": True,
            "Date": "20260102",
            "ValueCode": "1101",
            "runner_version": "test-runner",
            "runner_config_sha256": "b2" * 32,
            "config": exit_config,
            "config_sha256": exit_config_sha256,
            "artifacts": {
                snapshot.POSITION_ARTIFACT: {
                    "sha256": position_sha256,
                    "bytes": position_path.stat().st_size,
                    "rows": 24,
                    "columns": 1,
                }
            },
        }
        marker_path = exit_partition / "complete.json"
        marker_path.write_text(
            json.dumps(exit_marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker_sha256 = snapshot._file_sha256(marker_path)
        inventory_digest = hashlib.sha256()
        inventory_digest.update(relative_marker.encode())
        inventory_digest.update(b"\0")
        inventory_digest.update(bytes.fromhex(marker_sha256))
        inventory_sha256 = inventory_digest.hexdigest()
        config = snapshot.SnapshotConfig(
            marker_count=1,
            snapshot_time="2026-08-20T15:13:10+08:00",
            expected_marker_inventory_sha256=inventory_sha256,
            expected_last_key="20260102/1101",
            cost_sensitivity_bp=19.0,
        )
        daily = snapshot._daily_metrics(
            physical,
            ["20260102"],
            config,
            split_entry_route=False,
        )
        summary = snapshot._policy_summary(
            daily,
            split_entry_route=False,
        )
        route_daily = snapshot._daily_metrics(
            physical,
            ["20260102"],
            config,
            split_entry_route=True,
        )
        route_summary = snapshot._policy_summary(
            route_daily,
            split_entry_route=True,
        )
        inventory = pl.from_dicts(
            [
                {
                    "ordinal": 1,
                    "Date": "20260102",
                    "ValueCode": "1101",
                    "marker_path": str(exit_root / relative_marker),
                    "marker_sha256": marker_sha256,
                    "marker_bytes": marker_path.stat().st_size,
                    "runner_version": "test-runner",
                    "runner_config_sha256": "b2" * 32,
                    "partition_config_sha256": exit_config_sha256,
                    "entry_action_sha256": action_sha256,
                    "position_policy_sha256": position_sha256,
                    "position_policy_bytes": position_path.stat().st_size,
                    "position_policy_rows": 24,
                    "position_policy_columns": 1,
                }
            ],
            infer_schema_length=None,
        )
        inventory.write_csv(bundle / snapshot.INVENTORY_ARTIFACT)
        daily.write_parquet(bundle / snapshot.DAILY_ARTIFACT)
        daily.write_csv(bundle / snapshot.DAILY_CSV_ARTIFACT)
        summary.write_csv(bundle / snapshot.SUMMARY_ARTIFACT)
        route_daily.write_parquet(bundle / snapshot.ENTRY_ROUTE_DAILY_ARTIFACT)
        route_daily.write_csv(bundle / snapshot.ENTRY_ROUTE_DAILY_CSV_ARTIFACT)
        route_summary.write_csv(bundle / snapshot.ENTRY_ROUTE_SUMMARY_ARTIFACT)
        snapshot._plot_diagnostics(
            daily,
            bundle / snapshot.CHART_ARTIFACT,
            config,
        )
        artifact_names = (
            snapshot.INVENTORY_ARTIFACT,
            snapshot.DAILY_ARTIFACT,
            snapshot.DAILY_CSV_ARTIFACT,
            snapshot.SUMMARY_ARTIFACT,
            snapshot.ENTRY_ROUTE_DAILY_ARTIFACT,
            snapshot.ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
            snapshot.ENTRY_ROUTE_SUMMARY_ARTIFACT,
            snapshot.CHART_ARTIFACT,
        )
        config_payload = {
            "marker_count": 1,
            "snapshot_time": config.snapshot_time,
            "expected_marker_inventory_sha256": inventory_sha256,
            "expected_last_key": "20260102/1101",
            "cost_sensitivity_bp": 19.0,
            "snapshot_version": snapshot.SNAPSHOT_VERSION,
            "implementation_sha256": snapshot._file_sha256(
                Path(snapshot.__file__)
            ),
            "exit_root": str(exit_root),
            "entry_root": str(entry_root),
            "output_root": str(bundle.resolve()),
        }
        marker = {
            "complete": True,
            "snapshot_version": snapshot.SNAPSHOT_VERSION,
            "config": config_payload,
            "config_sha256": snapshot._canonical_sha256(config_payload),
            "snapshot_time": config.snapshot_time,
            "input_marker_count": 1,
            "input_marker_inventory_sha256": inventory_sha256,
            "input_last_key": "20260102/1101",
            "entry_product_day_count": 1,
            "exit_runner_config_sha256": "b2" * 32,
            "full_dates": ["20260102"],
            "full_date_count": 1,
            "excluded_partial_dates": [],
            "entry_action_aliases": 6,
            "established_entry_aliases": 6,
            "physical_entry_dependencies": 6,
            "physical_policy_cells": 24,
            "physical_policy_aliases": 24,
            "complete_date_physical_policy_cells": 24,
            "population_audit": {
                "established_entry_aliases": 6,
                "expected_boundary_quantiles": [50, 80, 95],
                "expected_entry_routes": list(snapshot.ENTRY_ROUTES),
                "expected_exit_routes": [
                    "future_bid_spot_taker",
                    "spot_ask_future_taker",
                ],
                "expected_rules": ["frozen_center", "frozen_lower"],
                "policy_rows_per_established_entry": 4,
                "validated_position_rows": 24,
            },
            "daily_rows": daily.height,
            "policy_summary_rows": summary.height,
            "entry_route_daily_rows": route_daily.height,
            "entry_route_summary_rows": route_summary.height,
            **snapshot.SAFETY_SEMANTICS,
            "artifacts": {
                name: snapshot._artifact_metadata(bundle / name)
                for name in artifact_names
            },
        }
        (bundle / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return bundle

    def test_valid_bundle_is_fully_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._build_bundle(Path(temporary))
            payload = snapshot.verify_interim_portfolio_snapshot(bundle)
            self.assertTrue(payload["analysis_only"])
            self.assertEqual(payload["entry_route_daily_rows"], 24)

    def test_verify_rejects_multi_flag_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._build_bundle(Path(temporary))
            marker_path = bundle / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker.update(
                {
                    "pathwise_ev_ready": True,
                    "joint_volume_allocated": True,
                    "full_cost_profile_complete": True,
                    "unresolved_cashflow_imputed": True,
                }
            )
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safety/semantic flags"):
                snapshot.verify_interim_portfolio_snapshot(bundle)

    def test_verify_rejects_coordinated_daily_dtype_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._build_bundle(Path(temporary))
            rewritten = (
                (
                    snapshot.DAILY_ARTIFACT,
                    snapshot.DAILY_CSV_ARTIFACT,
                ),
                (
                    snapshot.ENTRY_ROUTE_DAILY_ARTIFACT,
                    snapshot.ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
                ),
            )
            marker_path = bundle / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for parquet_name, csv_name in rewritten:
                frame = pl.read_parquet(bundle / parquet_name).with_columns(
                    pl.col("entry_positions").cast(pl.Int64)
                )
                frame.write_parquet(bundle / parquet_name)
                frame.write_csv(bundle / csv_name)
                marker["artifacts"][parquet_name] = snapshot._artifact_metadata(
                    bundle / parquet_name
                )
                marker["artifacts"][csv_name] = snapshot._artifact_metadata(
                    bundle / csv_name
                )
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical artifact schema"):
                snapshot.verify_interim_portfolio_snapshot(bundle)

    def test_every_declared_safety_semantic_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._build_bundle(Path(temporary))
            marker_path = bundle / "complete.json"
            original = json.loads(marker_path.read_text(encoding="utf-8"))
            for name, expected in snapshot.SAFETY_SEMANTICS.items():
                with self.subTest(name=name):
                    corrupted = dict(original)
                    corrupted[name] = (
                        not expected
                        if isinstance(expected, bool)
                        else f"{expected}-corrupt"
                    )
                    marker_path.write_text(
                        json.dumps(corrupted, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "safety/semantic flags",
                    ):
                        snapshot.verify_interim_portfolio_snapshot(bundle)
            marker_path.write_text(
                json.dumps(original, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
