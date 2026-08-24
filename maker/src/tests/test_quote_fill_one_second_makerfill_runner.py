"""Focused contracts for the causal one-second makerFill screening runner."""

from __future__ import annotations

import math
import unittest
from datetime import date, datetime, timedelta

import polars as pl

from ..quote_fill.one_second_makerfill_runner import (
    SCENARIO_ID,
    aggregate_outcomes,
    attach_q95_boundaries,
    build_candidate_windows,
    label_candidate_windows,
)
from ..quote_fill.targets import absolute_price_tick


def _events() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    prices = (100.0, 99.9, 99.9, 100.0)
    for generation, (submit_second, price) in enumerate(
        zip((300, 310, 320, 330), prices), start=1
    ):
        common = {
            "scenario_id": SCENARIO_ID,
            "Date": "20260102",
            "ValueCode": "2330",
            "absolute_price_tick": absolute_price_tick(price),
            "generation": generation,
            "submit_point_offset": 0 if price == 100.0 else -1,
        }
        rows.append(
            {
                **common,
                "second_from_open": submit_second,
                "kind": "submit",
                "reason": "forward_new_price",
                "is_cutoff": False,
            }
        )
        rows.append(
            {
                **common,
                "second_from_open": submit_second + 5,
                "kind": "cancel",
                "reason": "target_retreat",
                "is_cutoff": False,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _state() -> pl.DataFrame:
    # Source market timestamps are intentionally stored as naive UTC.
    start = datetime(2026, 1, 2, 1, 5, 0)  # noqa: DTZ001
    rows: list[dict[str, object]] = []
    for offset, sequence in zip((0, 10, 20, 30), (10, 11, 12, 13)):
        decision = start + timedelta(seconds=offset)
        snapshot_lag_ms = 900 if sequence == 13 else 100
        rows.append(
            {
                "Date": "20260102",
                "ValueCode": "2330",
                "QuoteCode": "CDFL6",
                "seconds_from_open": 300 + offset,
                "timestamp": decision,
                "spot_recv_time": decision
                - timedelta(milliseconds=snapshot_lag_ms),
                "spot_sequence": sequence,
                "anchor_ewma_120s_bp": 2.0,
                "contract_size": 1000.0,
                "end_date": date(2026, 1, 21),
                "fut_exec_bid": 101.0,
                "spot_bid": 100.0,
                "spot_ask": 100.5,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _boundaries() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260102"],
            "ValueCode": ["2330"],
            "QuoteCode": ["CDFL6"],
            "boundary_quantile": [95],
            "upper_distance_bp": [20.0],
            "lower_distance_bp": [15.0],
            "source_asof_date": ["20251231"],
            "execution_safe_snapshot": [True],
            "contains_target_day_outcome": [False],
        }
    )


def _ticks(state: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in state.iter_rows(named=True):
        sequence = int(row["spot_sequence"])
        rows.append(
            {
                "ValueCode": "2330",
                "ChannelSeq": sequence,
                "RecvTime": row["spot_recv_time"],
                "TransTime": row["spot_recv_time"] + timedelta(hours=8),
                "BidPrice1": 100.0,
                "BidPrice2": 99.8 if sequence == 12 else 99.9,
                "BidLots1": 10,
                "BidLots2": 20,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _makerfill() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": ["2330"] * 4,
            "ChannelSeq": [10, 11, 12, 13],
            "Bid1_FillSeconds": [1.0, 2.0, 2.0, 0.1],
            "Bid2_FillSeconds": [2.0, math.nan, 2.0, 2.0],
        },
        schema_overrides={
            "ChannelSeq": pl.UInt64,
            "Bid1_FillSeconds": pl.Float32,
            "Bid2_FillSeconds": pl.Float32,
        },
    )


class OneSecondMakerFillRunnerTest(unittest.TestCase):
    def test_candidate_windows_keep_snapshot_and_policy_contract(self) -> None:
        windows = build_candidate_windows(_events(), _state())
        result = attach_q95_boundaries(windows, _boundaries())

        self.assertEqual(result.height, 4)
        self.assertEqual(result["physical_order_id"][0], "20260102/2330/1")
        self.assertEqual(result["boundary_quantile"].unique().to_list(), [95])
        self.assertEqual(result["entry_threshold_basis_bp"].unique().to_list(), [22.0])
        self.assertTrue((result["nominal_lifetime_seconds"] == 5).all())
        self.assertTrue((result["boundary_source_asof_date"] < result["Date"]).all())

    def test_exact_rank_and_mixed_clock_labels_fail_closed(self) -> None:
        state = _state()
        windows = attach_q95_boundaries(
            build_candidate_windows(_events(), state), _boundaries()
        )
        outcomes = label_candidate_windows(
            windows,
            _ticks(state),
            _makerfill(),
        ).sort("generation")

        self.assertEqual(
            outcomes["outcome_status"].to_list(),
            [
                "approx_fill_before_nominal_stop",
                "approx_no_fill_through_eod",
                "unsupported_target_not_displayed_l1_l2",
                "unsupported_fill_not_after_submit",
            ],
        )
        self.assertEqual(outcomes["full_fill"].to_list(), [True, False, None, None])
        self.assertEqual(outcomes["outcome_supported"].sum(), 2)
        self.assertTrue(outcomes["snapshot_recv_time_exact_match"].all())
        self.assertFalse(outcomes["fill_cursor_exact"].any())

        summary = aggregate_outcomes(outcomes).row(0, named=True)
        self.assertEqual(summary["candidate_orders"], 4)
        self.assertEqual(summary["outcome_supported_orders"], 2)
        self.assertEqual(summary["unknown_orders"], 2)
        self.assertEqual(summary["approximate_fills"], 1)
        self.assertEqual(summary["approximate_cancels"], 1)
        self.assertEqual(summary["approximate_fill_rate"], 0.5)

    def test_unpaired_generation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one submit and one cancel"):
            build_candidate_windows(
                _events().filter(
                    ~(
                        (pl.col("generation") == 4)
                        & (pl.col("kind") == "cancel")
                    )
                ),
                _state(),
            )

    def test_target_day_boundary_is_rejected(self) -> None:
        lookahead = _boundaries().with_columns(
            pl.lit("20260102").alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "lookahead"):
            attach_q95_boundaries(
                build_candidate_windows(_events(), _state()),
                lookahead,
            )


if __name__ == "__main__":
    unittest.main()
