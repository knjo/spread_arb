"""S1 source-family and daily input-manifest contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..common.paths import (
    INDIVIDUAL_STOCK_FUTURES_ROOT,
    LEGACY_HFT_DATA_ROOT,
    PIPELINE_STORAGE,
)
from ..quote_fill import s1_production_runner as production_runner
from ..quote_fill.s1_entry_day_runner import S1EntryRunnerPaths
from ..quote_fill.s1_production_runner import (
    S1ProductionRunError,
    _validate_daily_file_records,
    _validate_prepared_source_paths,
)

DATE = "20260505"


def _record(role: str, path: Path) -> dict[str, object]:
    return {
        "path_scope": "absolute",
        "path": str(path),
        "role": role,
    }


def _valid_records() -> list[dict[str, object]]:
    root = Path("/portable/s1_input_contract")
    return [
        _record("causal_fair", root / f"Date={DATE}" / "causal_fair.parquet"),
        _record("mapping", root / f"Date={DATE}" / "mapping.parquet"),
        _record("contract_metadata", root / "metadata" / f"{DATE}_contracts.parquet"),
        _record(
            "spot_raw",
            LEGACY_HFT_DATA_ROOT / "tickData" / f"{DATE}_StockTick.parquet",
        ),
        _record(
            "future_raw",
            INDIVIDUAL_STOCK_FUTURES_ROOT
            / DATE[:4]
            / DATE[4:6]
            / DATE[6:8]
            / "stock_futures.parquet",
        ),
        _record(
            "makerfill",
            LEGACY_HFT_DATA_ROOT / "makerFill" / f"{DATE}_makerFill.parquet",
        ),
    ]


def _prepared_sources(records: list[dict[str, object]]) -> dict[str, Path]:
    source_key_by_role = {
        "causal_fair": "causal_fair",
        "mapping": "mapping",
        "contract_metadata": "contracts",
        "spot_raw": "spot_raw",
        "future_raw": "future_raw",
        "makerfill": "makerfill",
    }
    return {
        source_key_by_role[str(record["role"])]: Path(str(record["path"]))
        for record in records
    }


class S1InputPathTest(unittest.TestCase):
    def test_exact_daily_role_inventory_and_filenames_are_accepted(self) -> None:
        _validate_daily_file_records(_valid_records(), date=DATE)

        canonical = _valid_records()
        canonical[3] = _record(
            "spot_raw",
            PIPELINE_STORAGE.tick_dir / f"{DATE}_StockTick.parquet",
        )
        canonical[5] = _record(
            "makerfill",
            PIPELINE_STORAGE.maker_queue_dir / f"{DATE}_makerFill.parquet",
        )
        _validate_daily_file_records(canonical, date=DATE)

    def test_missing_role_duplicate_role_and_wrong_filename_fail_closed(self) -> None:
        missing = _valid_records()[:-1]
        with self.assertRaisesRegex(S1ProductionRunError, "role inventory"):
            _validate_daily_file_records(missing, date=DATE)

        duplicated = _valid_records()
        duplicated[-1] = _record(
            "spot_raw",
            Path("/portable/duplicate") / f"{DATE}_StockTick.parquet",
        )
        with self.assertRaisesRegex(S1ProductionRunError, "role inventory"):
            _validate_daily_file_records(duplicated, date=DATE)

        wrong = _valid_records()
        wrong[3] = _record("spot_raw", Path("/portable/wrong.parquet"))
        with self.assertRaisesRegex(S1ProductionRunError, "filename"):
            _validate_daily_file_records(wrong, date=DATE)

    def test_txf_is_never_accepted_as_individual_stock_futures(self) -> None:
        with self.assertRaisesRegex(ValueError, "TXF"):
            S1EntryRunnerPaths(future_tick_root=PIPELINE_STORAGE.txf_tick_dir)

        records = _valid_records()
        records[4] = _record(
            "future_raw",
            PIPELINE_STORAGE.txf_tick_dir
            / DATE[:4]
            / DATE[4:6]
            / DATE[6:8]
            / "stock_futures.parquet",
        )
        with self.assertRaisesRegex(S1ProductionRunError, "TXF"):
            _validate_daily_file_records(records, date=DATE)

    def test_entry_path_resolution_rejects_every_non_nas_future_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "NAS root"):
            S1EntryRunnerPaths(
                future_tick_root=Path("/portable/not_individual_futures")
            )

    def test_spot_and_makerfill_manifest_roots_are_exact(self) -> None:
        cases = (
            (3, "spot_raw", f"{DATE}_StockTick.parquet"),
            (5, "makerfill", f"{DATE}_makerFill.parquet"),
        )
        for index, role, filename in cases:
            with self.subTest(role=role):
                records = _valid_records()
                records[index] = _record(role, Path("/custom/raw") / filename)
                with self.assertRaisesRegex(
                    S1ProductionRunError,
                    "approved legacy root",
                ):
                    _validate_daily_file_records(records, date=DATE)

    def test_absolute_manifest_parent_traversal_is_rejected(self) -> None:
        records = _valid_records()
        records[3] = _record(
            "spot_raw",
            PIPELINE_STORAGE.tick_dir
            / ".."
            / PIPELINE_STORAGE.tick_dir.name
            / f"{DATE}_StockTick.parquet",
        )
        with self.assertRaisesRegex(S1ProductionRunError, "parent traversal"):
            _validate_daily_file_records(records, date=DATE)

    def test_raw_manifest_symlink_components_are_rejected_for_every_role(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            legacy = root / "legacy"
            legacy.mkdir()

            real_spot = root / "real_spot"
            real_spot.mkdir()
            (legacy / "tickData").symlink_to(real_spot, target_is_directory=True)
            spot_records = _valid_records()
            spot_records[3] = _record(
                "spot_raw",
                legacy / "tickData" / f"{DATE}_StockTick.parquet",
            )
            with (
                patch.object(
                    production_runner,
                    "LEGACY_HFT_DATA_ROOT",
                    legacy,
                ),
                self.assertRaisesRegex(S1ProductionRunError, "spot_raw.*symlink"),
            ):
                _validate_daily_file_records(spot_records, date=DATE)

            (legacy / "tickData").unlink()
            real_makerfill = root / "real_makerfill"
            real_makerfill.mkdir()
            (legacy / "makerFill").symlink_to(
                real_makerfill,
                target_is_directory=True,
            )
            makerfill_records = _valid_records()
            makerfill_records[5] = _record(
                "makerfill",
                legacy / "makerFill" / f"{DATE}_makerFill.parquet",
            )
            with (
                patch.object(
                    production_runner,
                    "LEGACY_HFT_DATA_ROOT",
                    legacy,
                ),
                self.assertRaisesRegex(S1ProductionRunError, "makerfill.*symlink"),
            ):
                _validate_daily_file_records(makerfill_records, date=DATE)

            futures_root = root / "nas" / "Parquet" / "Ticks"
            futures_root.mkdir(parents=True)
            real_year = root / "real_year"
            real_year.mkdir()
            (futures_root / DATE[:4]).symlink_to(
                real_year,
                target_is_directory=True,
            )
            future_records = _valid_records()
            future_records[4] = _record(
                "future_raw",
                futures_root
                / DATE[:4]
                / DATE[4:6]
                / DATE[6:8]
                / "stock_futures.parquet",
            )
            with (
                patch.object(
                    production_runner,
                    "INDIVIDUAL_STOCK_FUTURES_ROOT",
                    futures_root,
                ),
                self.assertRaisesRegex(S1ProductionRunError, "future_raw.*symlink"),
            ):
                _validate_daily_file_records(future_records, date=DATE)

    def test_prepared_sources_must_match_manifest_role_by_role(self) -> None:
        records = _valid_records()
        source_paths = _prepared_sources(records)
        _validate_prepared_source_paths(source_paths, records, date=DATE)

        source_paths["mapping"] = Path("/portable/other/mapping.parquet")
        with self.assertRaisesRegex(
            S1ProductionRunError,
            "prepared mapping source differs",
        ):
            _validate_prepared_source_paths(source_paths, records, date=DATE)


if __name__ == "__main__":
    unittest.main()
