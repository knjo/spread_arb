from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.cross_session_prerequisite import (
    _audit_extension_requirements,
    build_cross_session_prerequisites,
    verify_cross_session_prerequisites,
)


D1 = "20260813"
D2 = "20260814"
VALUE = "2303"
QUOTE = "CCFH6"


def _contracts(ref: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": [QUOTE],
            "ValueCode": [VALUE],
            "contract_size": [2_000.0],
            "decimal_locator": [2],
            "end_date": [date(2026, 8, 19)],
            "fut_ref_price": [ref],
        }
    )


def _requirements() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "candidate_date": [D2],
            "ValueCode": [VALUE],
            "QuoteCode": [QUOTE],
            "expiry_session": ["20260819"],
            "origin_product_days": [1],
            "established_entry_aliases": [2],
        }
    )


class CrossSessionPrerequisiteTests(unittest.TestCase):
    def test_atomic_versioned_root_verifies_and_detects_metadata_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions.txt"
            sessions.write_text(f"{D1}\n", encoding="utf-8")
            calendar = root / "calendar.parquet"
            pl.DataFrame(
                {
                    "QuoteCode": [QUOTE],
                    "expiry_session": ["20260819"],
                    "calendar_version": ["test-v1"],
                }
            ).write_parquet(calendar)
            product_days = root / "product-days.parquet"
            pl.DataFrame({"Date": [D1], "ValueCode": [VALUE]}).write_parquet(
                product_days
            )
            old_metadata = root / "old-metadata"
            old_metadata.mkdir()
            _contracts(99.0).write_parquet(
                old_metadata / f"{D1}_contracts.parquet"
            )
            lineage = pl.DataFrame(
                {
                    "Date": [D1],
                    "ValueCode": [VALUE],
                    "QuoteCode": [QUOTE],
                    "established_entry_aliases": [2],
                    "entry_marker_sha256": ["a" * 64],
                    "entry_config_sha256": ["b" * 64],
                    "action_artifact_sha256": ["c" * 64],
                }
            )
            audit = _requirements().with_columns(
                pl.lit(True).alias("mapping_valid")
            )
            sources = pl.DataFrame(
                {
                    "Date": [D2],
                    "source": ["synthetic"],
                    "path": ["synthetic://source"],
                    "bytes": [None],
                    "mtime_ns": [None],
                    "sha256": ["d" * 64],
                }
            )
            output = root / "versioned-prerequisite"
            with (
                patch(
                    "maker.src.quote_fill.cross_session_prerequisite."
                    "_candidate_requirements",
                    return_value=(_requirements(), lineage),
                ),
                patch(
                    "maker.src.quote_fill.cross_session_prerequisite."
                    "_audit_extension_requirements",
                    return_value=(audit, sources),
                ),
            ):
                result = build_cross_session_prerequisites(
                    base_sessions_path=sessions,
                    extension_sessions=(D2,),
                    contract_calendar_path=calendar,
                    entry_execution_root=root / "entry",
                    product_days_path=product_days,
                    existing_metadata_root=old_metadata,
                    output_root=output,
                    data_root=root / "data",
                    futures_raw_root=root / "futures",
                    basic_loader=lambda _session: _contracts(101.0),
                    resume=False,
                )
            self.assertFalse(result.resumed)
            payload = verify_cross_session_prerequisites(output)
            self.assertEqual(payload["session_count"], 2)
            self.assertEqual(payload["metadata_session_count"], 2)
            self.assertEqual(
                (output / "candidate_sessions.txt").read_text().splitlines(),
                [D1, D2],
            )
            target = output / "metadata" / f"{D2}_contracts.parquet"
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                verify_cross_session_prerequisites(output)

    def test_extension_audit_accepts_day_trade_n_and_binds_exact_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            futures_root = root / "futures"
            metadata_root = root / "metadata"
            for name in ("tickData", "tickFeature", "marketData"):
                (data_root / name).mkdir(parents=True)
            future_path = (
                futures_root / "2026" / "08" / "14" / "stock_futures.parquet"
            )
            future_path.parent.mkdir(parents=True)
            metadata_root.mkdir()
            pl.DataFrame(
                {"ValueCode": [VALUE], "QuoteCode": [QUOTE]}
            ).write_parquet(
                data_root / "tickData" / f"{D2}_StockTick.parquet"
            )
            pl.DataFrame({"QuoteCode": [VALUE]}).write_parquet(
                data_root / "tickFeature" / f"{D2}_tickFeature.parquet"
            )
            pl.DataFrame(
                {
                    "quote_code": [VALUE],
                    "opening_ref_price": [100.0],
                    "allow_day_trade_mark": ["N"],
                }
            ).write_parquet(
                data_root / "marketData" / f"{D2}_marketData.parquet"
            )
            pl.DataFrame(
                {"ValueCode": [VALUE], "QuoteCode": [QUOTE]}
            ).write_parquet(future_path)
            _contracts().write_parquet(
                metadata_root / f"{D2}_contracts.parquet"
            )
            audit, sources = _audit_extension_requirements(
                _requirements(),
                extension=(D2,),
                metadata_root=metadata_root,
                data_root=data_root,
                futures_raw_root=futures_root,
                futures_basic_hashes={D2: "e" * 64},
                validation_pairs=_contracts().select(
                    "ValueCode", "QuoteCode", "end_date"
                ),
            )
            row = audit.row(0, named=True)
            self.assertTrue(row["mapping_valid"])
            self.assertEqual(row["day_trade_mark"], "N")
            self.assertTrue(row["future_raw_exact_events"])
            self.assertTrue(row["spot_raw_events"])
            self.assertTrue(row["spread_clock_rows"])
            self.assertEqual(sources.height, 5)


if __name__ == "__main__":
    unittest.main()
