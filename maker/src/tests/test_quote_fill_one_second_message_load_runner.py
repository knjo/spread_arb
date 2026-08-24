"""Focused aggregation tests for the causal one-second load runner."""

from __future__ import annotations

import unittest

import polars as pl

from ..quote_fill.one_second_message_load_runner import (
    ENTRY_CUTOFF_SECOND,
    LoadScenario,
    capacity_summary,
    spread_pair_second_counts,
)


class OneSecondMessageLoadRunnerTest(unittest.TestCase):
    def test_spread_pair_samples_use_exact_clock_then_sequence_fallback(self) -> None:
        frame = pl.DataFrame(
            {
                "ValueCode": ["A", "A", "A", "B", "B", "B"],
                "seconds_from_open": [300, 301, 302, 300, 301, 302],
                "spot_sequence": [10, 11, 12, 20, 20, 21],
                "SpreadPairTotalCount": [1, 1, 2, None, None, None],
            },
            schema_overrides={"SpreadPairTotalCount": pl.UInt32},
        )

        result = spread_pair_second_counts(frame, "20260102")

        self.assertEqual(result["exact_sample_count"].sum(), 2)
        self.assertEqual(result["fallback_sequence_sample_count"].sum(), 2)
        self.assertEqual(result["sample_count"].sum(), 4)

    def test_capacity_separates_cutoff_and_adjacent_bucket_stress(self) -> None:
        scenario = LoadScenario(
            "ab12_entry_until_1300",
            ENTRY_CUTOFF_SECOND,
            (-1, 0),
        )
        per_second = pl.DataFrame(
            {
                "scenario_id": [scenario.scenario_id] * 3,
                "Date": ["20260102"] * 3,
                "second_from_open": [300, 301, ENTRY_CUTOFF_SECOND],
                "submits": [60, 50, 0],
                "cancels": [0, 0, 150],
                "requests": [60, 50, 150],
                "is_cutoff": [False, False, True],
            }
        )

        row = capacity_summary(
            per_second,
            ["20260102"],
            scenarios=(scenario,),
            cap=100,
        ).row(0, named=True)

        self.assertEqual(row["max_intraday_requests"], 60)
        self.assertEqual(row["intraday_seconds_over_cap"], 0)
        self.assertEqual(row["max_adjacent_two_bucket_requests"], 110)
        self.assertEqual(row["adjacent_two_bucket_rows_over_cap"], 1)
        self.assertEqual(row["max_cutoff_requests"], 150)
        self.assertEqual(
            row["minimum_seconds_to_drain_max_cutoff_at_cap"], 2
        )


if __name__ == "__main__":
    unittest.main()
