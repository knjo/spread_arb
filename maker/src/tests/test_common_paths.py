"""Filesystem contracts for maker research inputs."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.pipeline_storage import PipelineStoragePaths, load_pipeline_storage

from ..common import paths as path_contracts
from ..common.paths import (
    HFT_DATA_ROOT,
    HFT_ROOT,
    find_hft_root,
    futures_raw_path,
    resolve_input_file,
    validate_required_mount,
)

DATE = "20260505"


def _temporary_pipeline_storage(
    root: Path,
) -> tuple[PipelineStoragePaths, Path]:
    mount = root / "ssd2"
    data = mount / "Data"
    tick = data / "tickData"
    makerfill = data / "makerFill"
    tick.mkdir(parents=True)
    makerfill.mkdir()
    legacy = root / "legacy_data"
    return (
        replace(
            path_contracts.PIPELINE_STORAGE,
            base_dir=data,
            tick_dir=tick,
            maker_queue_dir=makerfill,
            required_mount=mount,
        ),
        legacy,
    )


class CommonPathsTest(unittest.TestCase):
    def test_live_root_uses_pipeline_storage_instead_of_legacy_data(self) -> None:
        configured = load_pipeline_storage(project_root=HFT_ROOT)
        self.assertEqual(HFT_DATA_ROOT, configured.base_dir)
        self.assertNotEqual(HFT_DATA_ROOT, HFT_ROOT / "data")

    def test_root_discovery_does_not_require_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config").mkdir()
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (root / "config" / "pipeline.yaml").write_text(
                "data_storage: {}\n",
                encoding="utf-8",
            )
            nested = root / "a" / "b"
            nested.mkdir(parents=True)

            self.assertEqual(find_hft_root(nested), root)

    def test_missing_legacy_spot_uses_exact_pipeline_canonical_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            requested = legacy / "tickData" / f"{DATE}_StockTick.parquet"
            canonical = storage.tick_dir / requested.name
            canonical.write_bytes(b"canonical")

            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
                patch.object(
                    path_contracts,
                    "validate_required_mount",
                ) as validate_mount,
            ):
                selected = resolve_input_file(
                    requested,
                    canonical=canonical,
                    role="spot_raw",
                    legacy_parents=(requested.parent,),
                    canonical_required_mount=storage.required_mount,
                )

            self.assertEqual(selected, canonical)
            validate_mount.assert_called_once_with(
                storage.required_mount,
                role="spot_raw",
            )

    def test_existing_legacy_file_wins_and_missing_pair_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            requested = legacy / "makerFill" / f"{DATE}_makerFill.parquet"
            canonical = storage.maker_queue_dir / requested.name
            requested.parent.mkdir(parents=True)
            requested.write_bytes(b"legacy")
            canonical.write_bytes(b"canonical")

            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
                patch.object(
                    path_contracts,
                    "validate_required_mount",
                ) as validate_mount,
            ):
                self.assertEqual(
                    resolve_input_file(
                        requested,
                        canonical=canonical,
                        role="makerfill",
                        legacy_parents=(requested.parent,),
                        canonical_required_mount=storage.required_mount,
                    ),
                    requested,
                )
                validate_mount.assert_not_called()

                requested.unlink()
                canonical.unlink()
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "tried legacy .* and canonical",
                ):
                    resolve_input_file(
                        requested,
                        canonical=canonical,
                        role="makerfill",
                        legacy_parents=(requested.parent,),
                        canonical_required_mount=storage.required_mount,
                    )
                validate_mount.assert_called_once_with(
                    storage.required_mount,
                    role="makerfill",
                )

    def test_existing_custom_spot_and_makerfill_roots_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            cases = (
                (
                    "spot_raw",
                    f"{DATE}_StockTick.parquet",
                    storage.tick_dir,
                    legacy / "tickData",
                ),
                (
                    "makerfill",
                    f"{DATE}_makerFill.parquet",
                    storage.maker_queue_dir,
                    legacy / "makerFill",
                ),
            )
            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
            ):
                for role, filename, canonical_root, approved_legacy in cases:
                    with self.subTest(role=role):
                        requested = root / f"custom_{role}" / filename
                        requested.parent.mkdir()
                        requested.write_bytes(b"custom")
                        canonical = canonical_root / filename
                        canonical.write_bytes(b"canonical")

                        with self.assertRaisesRegex(
                            FileNotFoundError,
                            "fallback is not allowed",
                        ):
                            resolve_input_file(
                                requested,
                                canonical=canonical,
                                role=role,
                                legacy_parents=(approved_legacy,),
                                canonical_required_mount=storage.required_mount,
                            )

    def test_arbitrary_canonical_typo_and_different_basename_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            filename = f"{DATE}_StockTick.parquet"
            requested = legacy / "tickData" / filename
            canonical = storage.tick_dir / filename
            canonical.write_bytes(b"canonical")
            arbitrary = root / "other_ssd" / filename
            arbitrary.parent.mkdir()
            arbitrary.write_bytes(b"wrong source")

            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
            ):
                with self.assertRaisesRegex(ValueError, "pipeline.yaml"):
                    resolve_input_file(
                        requested,
                        canonical=arbitrary,
                        role="spot_raw",
                        legacy_parents=(requested.parent,),
                    )
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "fallback is not allowed",
                ):
                    resolve_input_file(
                        root / "tickDat" / filename,
                        canonical=canonical,
                        role="spot_raw",
                        legacy_parents=(requested.parent,),
                    )
                with self.assertRaisesRegex(ValueError, "filename"):
                    resolve_input_file(
                        legacy / "tickData" / "wrong.parquet",
                        canonical=canonical,
                        role="spot_raw",
                        legacy_parents=(requested.parent,),
                    )

    def test_pipeline_fallback_rejects_symlink_file_and_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            filename = f"{DATE}_StockTick.parquet"
            canonical = storage.tick_dir / filename
            canonical.write_bytes(b"canonical")
            real_legacy = root / "real_legacy_tick"
            real_legacy.mkdir()
            legacy.mkdir()
            (legacy / "tickData").symlink_to(real_legacy, target_is_directory=True)
            requested = legacy / "tickData" / filename

            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
                self.assertRaisesRegex(FileNotFoundError, "symlink"),
            ):
                resolve_input_file(
                    requested,
                    canonical=canonical,
                    role="spot_raw",
                    legacy_parents=(requested.parent,),
                )

            (legacy / "tickData").unlink()
            (legacy / "tickData").mkdir()
            target = root / "target.parquet"
            target.write_bytes(b"target")
            requested.symlink_to(target)
            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
                self.assertRaisesRegex(FileNotFoundError, "symlink"),
            ):
                resolve_input_file(
                    requested,
                    canonical=canonical,
                    role="spot_raw",
                    legacy_parents=(requested.parent,),
                )

    def test_canonical_symlink_is_rejected_even_when_target_is_regular(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            storage, legacy = _temporary_pipeline_storage(root)
            filename = f"{DATE}_makerFill.parquet"
            requested = legacy / "makerFill" / filename
            target = root / "makerfill_target.parquet"
            target.write_bytes(b"target")
            canonical = storage.maker_queue_dir / filename
            canonical.symlink_to(target)

            with (
                patch.object(path_contracts, "PIPELINE_STORAGE", storage),
                patch.object(path_contracts, "LEGACY_HFT_DATA_ROOT", legacy),
                self.assertRaisesRegex(FileNotFoundError, "symlink"),
            ):
                resolve_input_file(
                    requested,
                    canonical=canonical,
                    role="makerfill",
                    legacy_parents=(requested.parent,),
                )

    def test_required_mount_validation_uses_injected_mount_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mountpoint = root / "mount"
            mountpoint.mkdir()
            validate_required_mount(
                mountpoint,
                role="spot_raw",
                mount_checker=lambda path: path == mountpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                validate_required_mount(
                    mountpoint,
                    role="spot_raw",
                    mount_checker=lambda _path: False,
                )
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                validate_required_mount(
                    root / "missing",
                    role="spot_raw",
                    mount_checker=lambda _path: True,
                )

            mount_alias = root / "mount_alias"
            mount_alias.symlink_to(mountpoint, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                validate_required_mount(
                    mount_alias,
                    role="spot_raw",
                    mount_checker=lambda _path: True,
                )

    def test_future_raw_is_exact_nas_partition_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            nas_mount = root / "nas"
            futures_root = nas_mount / "Parquet" / "Ticks"
            expected = futures_root / "2026" / "05" / "05" / "stock_futures.parquet"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"future")

            with (
                patch.object(
                    path_contracts,
                    "INDIVIDUAL_STOCK_FUTURES_ROOT",
                    futures_root,
                ),
                patch.object(
                    path_contracts,
                    "INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT",
                    nas_mount,
                ),
                patch.object(
                    path_contracts,
                    "validate_required_mount",
                ) as validate_mount,
            ):
                self.assertEqual(futures_raw_path(DATE), expected)
                self.assertEqual(
                    resolve_input_file(expected, role="future_raw"),
                    expected,
                )
                validate_mount.assert_called_once_with(
                    nas_mount,
                    role="future_raw",
                )
                with self.assertRaisesRegex(ValueError, "does not allow fallback"):
                    resolve_input_file(
                        expected,
                        canonical=expected,
                        role="future_raw",
                    )

    def test_future_raw_rejects_txf_and_any_non_nas_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            nas_mount = root / "nas"
            futures_root = nas_mount / "Parquet" / "Ticks"
            txf = (
                root
                / "ssd2"
                / "Data"
                / "txfTickData"
                / "2026"
                / "05"
                / "05"
                / "stock_futures.parquet"
            )
            txf.parent.mkdir(parents=True)
            txf.write_bytes(b"txf")

            with (
                patch.object(
                    path_contracts,
                    "INDIVIDUAL_STOCK_FUTURES_ROOT",
                    futures_root,
                ),
                patch.object(
                    path_contracts,
                    "INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT",
                    nas_mount,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "NAS root"):
                    resolve_input_file(txf, role="future_raw")
                with self.assertRaisesRegex(ValueError, "valid YYYY/MM/DD"):
                    resolve_input_file(
                        futures_root / "2026" / "02" / "30" / "stock_futures.parquet",
                        role="future_raw",
                    )


if __name__ == "__main__":
    unittest.main()
