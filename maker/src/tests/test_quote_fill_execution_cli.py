from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import polars as pl

from maker.src.quote_fill.execution_cli import (
    _available_product_days,
    build_frozen_exit_rules,
)


class FrozenExitRuleTest(unittest.TestCase):
    def _actions(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": ["20260813", "20260813"],
                "ValueCode": ["2317", "2317"],
                "QuoteCode": ["DHFH6", "DHFH6"],
                "boundary_quantile": [50, 80],
                "policy_generation_id": ["p50", "p80"],
                "threshold_basis_bp": [30.0, 42.0],
                "source_asof_date": ["20260812", "20260812"],
            }
        )

    def _boundaries(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": ["20260813", "20260813"],
                "ValueCode": ["2317", "2317"],
                "QuoteCode": ["DHFH6", "DHFH6"],
                "boundary_quantile": [50, 80],
                "upper_distance_bp": [10.0, 20.0],
                "lower_distance_bp": [12.0, 18.0],
                "source_asof_date": ["20260812", "20260812"],
                "execution_safe_snapshot": [True, True],
                "contains_target_day_outcome": [False, False],
            }
        )

    def test_asymmetric_frozen_thresholds_are_entry_causal(self) -> None:
        rules = build_frozen_exit_rules(self._actions(), self._boundaries())
        actual = {
            (row["policy_generation_id"], row["exit_rule_id"]): row[
                "exit_threshold_basis_bp"
            ]
            for row in rules.iter_rows(named=True)
        }
        self.assertEqual(
            actual,
            {
                ("p50", "frozen_center"): 20.0,
                ("p50", "frozen_lower"): 8.0,
                ("p80", "frozen_center"): 22.0,
                ("p80", "frozen_lower"): 4.0,
            },
        )
        self.assertFalse(rules["contains_target_day_outcome"].any())

    def test_source_lineage_and_target_day_outcome_fail_closed(self) -> None:
        mismatch = self._boundaries().with_columns(
            pl.lit("20260811").alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            build_frozen_exit_rules(self._actions(), mismatch)

        unsafe = self._boundaries().with_columns(
            pl.lit(True).alias("contains_target_day_outcome")
        )
        with self.assertRaisesRegex(ValueError, "execution-safe"):
            build_frozen_exit_rules(self._actions(), unsafe)

    def test_common_calendar_keeps_missing_product_day_as_zero_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for date, products in (
                ("20260812", ["2303", "2317"]),
                ("20260813", ["2317"]),
            ):
                partition = root / f"Date={date}"
                partition.mkdir()
                pl.DataFrame({"ValueCode": products}).write_parquet(
                    partition / "mapping.parquet"
                )
            keys, audit = _available_product_days(
                ("20260812", "20260813"),
                ("2303", "2317"),
                daily_root=root,
            )
        self.assertEqual(
            keys,
            (
                ("20260812", "2303"),
                ("20260812", "2317"),
                ("20260813", "2317"),
            ),
        )
        missing = audit.filter(~pl.col("available_daily_mapping")).row(
            0, named=True
        )
        self.assertEqual(missing["Date"], "20260813")
        self.assertEqual(missing["ValueCode"], "2303")
        self.assertEqual(missing["availability_status"], "missing_daily_mapping")


if __name__ == "__main__":
    unittest.main()
