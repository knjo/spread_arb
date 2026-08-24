"""Publication-contract tests for partitioned daily latent facts."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from maker.src.quote_width.daily_facts import (
    EXPECTED_ARTIFACT_SCHEMAS,
    MIGRATED_DAILY_FACT_SCHEMA_VERSION,
    completed_artifact_paths,
    migrate_legacy_completion_markers,
    validate_completion_marker,
)


class DailyFactMarkerTest(unittest.TestCase):
    def test_legacy_is_fail_closed_until_explicit_migration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "Date=20260102"
            partition.mkdir()
            for name, schema in EXPECTED_ARTIFACT_SCHEMAS.items():
                values: dict[str, object] = {}
                for column, dtype in schema.items():
                    if column == "Date":
                        value: object = "20260102"
                    elif dtype == pl.String:
                        value = "x"
                    elif dtype == pl.Date:
                        value = date(2026, 1, 2)
                    elif isinstance(dtype, pl.Datetime):
                        value = datetime(2026, 1, 2)
                    elif dtype == pl.Boolean:
                        value = False
                    elif dtype.is_integer():
                        value = 1
                    else:
                        value = 1.0
                    values[column] = value
                pl.DataFrame([values], schema=schema).write_parquet(
                    partition / name
                )
            marker = partition / "complete.json"
            marker.write_text(
                json.dumps(
                    {
                        "date": "20260102",
                        "products": 1,
                        "causal_rows": 1,
                        "excursion_rows": 1,
                        "elapsed_seconds": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported daily fact schema"):
                validate_completion_marker(marker)
            self.assertEqual(migrate_legacy_completion_markers(root), 1)
            payload = validate_completion_marker(marker)
            self.assertEqual(
                payload["schema_version"],
                MIGRATED_DAILY_FACT_SCHEMA_VERSION,
            )
            self.assertFalse(payload["legacy_atomic_publish_verified"])
            self.assertEqual(
                completed_artifact_paths(root, "causal_fair.parquet"),
                [partition / "causal_fair.parquet"],
            )

            causal = pl.read_parquet(partition / "causal_fair.parquet")
            pl.concat([causal, causal]).write_parquet(
                partition / "causal_fair.parquet"
            )
            with self.assertRaisesRegex(ValueError, "row-count mismatch"):
                validate_completion_marker(marker)


if __name__ == "__main__":
    unittest.main()
