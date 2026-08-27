"""Contracts for the S1 actual-send legacy makerFill adapter."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl

from ..quote_fill.makerfill_adapter import (
    MakerFillLabelIndex,
    MakerFillObservedSnapshot,
    MakerFillScalarOrder,
    MakerFillSnapshotIndex,
    classify_actual_active_intervals,
    derive_potential_fill_events,
)
from ..quote_fill.targets import absolute_price_tick


def _fixtures() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    base = datetime(2026, 5, 5, 1, 0, 0)  # noqa: DTZ001
    base_ns = int(pl.Series([base]).dt.timestamp("ns")[0])
    orders = pl.from_dicts(
        [
            {
                "raw_order_fact_id": f"raw-{sequence}",
                "Date": "20260505",
                "ValueCode": "2330",
                "QuoteCode": "CDFE6",
                "absolute_price_tick": absolute_price_tick(price),
                "maker_snapshot_channel_seq": sequence,
                "maker_snapshot_recv_time_ns": base_ns + index * 1_000_000_000,
                "actual_new_send_time_ns": base_ns
                + index * 1_000_000_000
                + 100_000_000,
                "nominal_stop_time_ns": 1,
            }
            for index, (sequence, price) in enumerate(
                [(10, 100.0), (11, 100.0), (12, 99.9), (13, 100.0)]
            )
        ],
        infer_schema_length=None,
    )
    ticks = pl.from_dicts(
        [
            {
                "ValueCode": "2330",
                "ChannelSeq": sequence,
                "RecvTime": base + timedelta(seconds=index),
                "TransTime": base + timedelta(hours=8, seconds=index),
                "BidPrice1": 100.0,
                "BidPrice2": 99.8 if sequence == 12 else 99.9,
                "BidLots1": 10,
                "BidLots2": 20,
            }
            for index, sequence in enumerate([10, 11, 12, 13])
        ],
        infer_schema_length=None,
    )
    makerfill = pl.DataFrame(
        {
            "QuoteCode": ["2330"] * 4,
            "ChannelSeq": [10, 11, 12, 13],
            "Bid1_FillSeconds": [1.0, 1.0, 1.0, 0.0],
            "Bid2_FillSeconds": [2.0, math.nan, 1.0, 1.0],
        },
        schema_overrides={
            "ChannelSeq": pl.UInt64,
            "Bid1_FillSeconds": pl.Float32,
            "Bid2_FillSeconds": pl.Float32,
        },
    )
    return orders, ticks, makerfill


def _equivalence_fixtures() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    base = datetime(2026, 5, 5, 1, 0, 0)  # noqa: DTZ001
    base_ns = int(pl.Series([base]).dt.timestamp("ns")[0])
    cases = [
        ("supported", 10, 100.0),
        ("eod", 11, 99.9),
        ("rank", 12, 99.9),
        ("preworking", 13, 100.0),
        ("invalid-negative", 14, 100.0),
        ("missing-maker", 15, 100.0),
        ("invalid-infinite", 16, 100.0),
        ("missing-tick", 99, 100.0),
    ]
    orders = pl.from_dicts(
        [
            {
                "raw_order_fact_id": f"raw-{name}",
                "Date": "20260505",
                "ValueCode": "2330",
                "QuoteCode": "CDFE6",
                "absolute_price_tick": absolute_price_tick(price),
                "maker_snapshot_channel_seq": sequence,
                "maker_snapshot_recv_time_ns": base_ns + index * 1_000_000_000,
                "actual_new_send_time_ns": base_ns
                + index * 1_000_000_000
                + 100_000_000,
            }
            for index, (name, sequence, price) in enumerate(cases)
        ],
        infer_schema_length=None,
    )
    ticks = pl.from_dicts(
        [
            {
                "ValueCode": "2330",
                "ChannelSeq": sequence,
                "RecvTime": base + timedelta(seconds=index),
                "TransTime": base + timedelta(hours=8, seconds=index),
                "BidPrice1": 100.0,
                "BidPrice2": 99.8 if sequence == 12 else 99.9,
                "BidLots1": 10 + index,
                "BidLots2": 20 + index,
            }
            for index, sequence in enumerate(range(10, 17))
        ],
        infer_schema_length=None,
    )
    makerfill = pl.DataFrame(
        {
            "QuoteCode": ["2330"] * 6,
            "ChannelSeq": [10, 11, 12, 13, 14, 16],
            "Bid1_FillSeconds": [1.0, 1.0, 1.0, 0.0, -1.0, math.inf],
            "Bid2_FillSeconds": [2.0, math.nan, 1.0, 1.0, 1.0, 1.0],
        },
        schema_overrides={
            "ChannelSeq": pl.UInt64,
            "Bid1_FillSeconds": pl.Float64,
            "Bid2_FillSeconds": pl.Float64,
        },
    )
    return orders, ticks, makerfill


class MakerFillAdapterTest(unittest.TestCase):
    def test_potential_events_use_actual_new_and_ignore_nominal_stop(self) -> None:
        orders, ticks, makerfill = _fixtures()
        result = derive_potential_fill_events(orders, ticks, makerfill).sort(
            "raw_order_fact_id"
        )

        self.assertEqual(
            result["potential_outcome_status"].to_list(),
            [
                "approx_potential_fill",
                "approx_potential_fill",
                "unsupported_target_not_displayed_l1_l2",
                "unsupported_fill_not_after_actual_new",
            ],
        )
        self.assertEqual(
            result["makerfill_potential_fill"].to_list(),
            [True, True, None, None],
        )
        self.assertEqual(result["entry_fill_truth"].unique().to_list(), ["approximate"])
        self.assertNotIn(
            "nominal_stop_time_ns", result["potential_outcome_status"].name
        )

    def test_actual_interval_is_start_exclusive_and_end_inclusive(self) -> None:
        orders, ticks, makerfill = _fixtures()
        potential = derive_potential_fill_events(orders, ticks, makerfill)
        terminal_by_id = {
            "raw-10": potential.filter(pl.col("raw_order_fact_id") == "raw-10")[
                "makerfill_potential_fill_time_ns"
            ][0],
            "raw-11": orders.filter(pl.col("raw_order_fact_id") == "raw-11")[
                "actual_new_send_time_ns"
            ][0]
            + 500_000_000,
            "raw-12": orders.filter(pl.col("raw_order_fact_id") == "raw-12")[
                "actual_new_send_time_ns"
            ][0]
            + 500_000_000,
            "raw-13": orders.filter(pl.col("raw_order_fact_id") == "raw-13")[
                "actual_new_send_time_ns"
            ][0]
            + 500_000_000,
        }
        resolved = classify_actual_active_intervals(
            potential.with_columns(
                pl.col("raw_order_fact_id")
                .replace_strict(terminal_by_id)
                .cast(pl.Int64)
                .alias("actual_terminal_time_ns")
            )
        ).sort("raw_order_fact_id")

        self.assertEqual(
            resolved["approximate_full_fill_within_actual_interval"].to_list(),
            [True, False, None, None],
        )
        self.assertEqual(
            resolved["actual_interval_outcome_status"].to_list(),
            [
                "approx_fill_within_actual_interval",
                "actual_terminal_before_later_approx_fill",
                "unsupported_target_not_displayed_l1_l2",
                "unsupported_fill_not_after_actual_new",
            ],
        )

    def test_missing_snapshot_and_duplicate_identity_fail_closed(self) -> None:
        orders, ticks, makerfill = _fixtures()
        missing = derive_potential_fill_events(
            orders.head(1), ticks.filter(pl.col("ChannelSeq") != 10), makerfill
        )
        self.assertFalse(missing["outcome_supported"][0])
        self.assertEqual(
            missing["potential_outcome_status"][0],
            "unsupported_missing_raw_tick_snapshot",
        )

        with self.assertRaisesRegex(ValueError, "duplicated"):
            derive_potential_fill_events(
                pl.concat([orders.head(1), orders.head(1)]), ticks, makerfill
            )

    def test_indexed_scalar_events_match_batch_semantics(self) -> None:
        orders, ticks, makerfill = _equivalence_fixtures()
        batch = derive_potential_fill_events(orders, ticks, makerfill)
        batch_by_id = {
            row["raw_order_fact_id"]: row for row in batch.iter_rows(named=True)
        }
        index = MakerFillSnapshotIndex.from_frames(
            ticks.drop("TransTime"),
            makerfill,
        )
        self.assertEqual(index.tick_snapshot_count, 7)
        self.assertEqual(index.makerfill_snapshot_count, 6)
        self.assertEqual(index.product_span_count, 1)
        self.assertGreater(index.estimated_size_bytes, 0)

        status_by_id = {
            "raw-supported": "approx_potential_fill",
            "raw-eod": "approx_no_fill_through_eod",
            "raw-rank": "unsupported_target_not_displayed_l1_l2",
            "raw-preworking": "unsupported_fill_not_after_actual_new",
            "raw-invalid-negative": "unsupported_invalid_fill_seconds",
            "raw-missing-maker": "unsupported_missing_makerfill_key",
            "raw-invalid-infinite": "unsupported_invalid_fill_seconds",
            "raw-missing-tick": "unsupported_missing_raw_tick_snapshot",
        }
        exact_fields = (
            "raw_tick_recv_time_ns",
            "snapshot_bid_lots1",
            "snapshot_bid_lots2",
            "snapshot_recv_time_exact_match",
            "exact_target_rank",
            "initial_displayed_lots",
            "makerfill_column",
            "makerfill_potential_fill_time_ns",
            "makerfill_mapping_exact",
            "outcome_supported",
            "legacy_no_fill_through_eod",
            "makerfill_potential_fill",
            "potential_outcome_status",
            "makerfill_adapter_version",
            "entry_fill_truth",
            "fill_cursor_exact",
            "own_quantity_included",
            "partial_fill_included",
        )
        float_fields = (
            "snapshot_bid_price1",
            "snapshot_bid_price2",
            "target_price",
            "maker_snapshot_age_ms",
            "makerfill_fill_seconds",
        )
        for order_row in orders.iter_rows(named=True):
            scalar = index.derive_potential_fill_event(
                MakerFillScalarOrder.from_mapping(order_row)
            )
            expected = batch_by_id[scalar.raw_order_fact_id]
            self.assertEqual(
                scalar.potential_outcome_status,
                status_by_id[scalar.raw_order_fact_id],
            )
            for field in exact_fields:
                self.assertEqual(
                    getattr(scalar, field),
                    expected[field],
                    f"{scalar.raw_order_fact_id}: {field}",
                )
            for field in float_fields:
                actual_value = getattr(scalar, field)
                expected_value = expected[field]
                if expected_value is None:
                    self.assertIsNone(
                        actual_value,
                        f"{scalar.raw_order_fact_id}: {field}",
                    )
                elif isinstance(expected_value, float) and math.isnan(expected_value):
                    self.assertTrue(
                        isinstance(actual_value, float) and math.isnan(actual_value),
                        f"{scalar.raw_order_fact_id}: {field}",
                    )
                else:
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        places=12,
                        msg=f"{scalar.raw_order_fact_id}: {field}",
                    )

    def test_index_enforces_unique_keys_and_scalar_clock_gates(self) -> None:
        orders, ticks, makerfill = _equivalence_fixtures()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            MakerFillSnapshotIndex.from_frames(
                pl.concat([ticks, ticks.head(1)]),
                makerfill,
            )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            MakerFillSnapshotIndex.from_frames(
                ticks,
                pl.concat([makerfill, makerfill.head(1)]),
            )

        index = MakerFillSnapshotIndex.from_frames(ticks, makerfill)
        supported_row = orders.filter(
            pl.col("raw_order_fact_id") == "raw-supported"
        ).row(0, named=True)
        supported = MakerFillScalarOrder.from_mapping(supported_row)

        recv_mismatch = index.derive_potential_fill_event(
            replace(
                supported,
                maker_snapshot_recv_time_ns=supported.maker_snapshot_recv_time_ns + 1,
            )
        )
        self.assertFalse(recv_mismatch.outcome_supported)
        self.assertEqual(
            recv_mismatch.potential_outcome_status,
            "unsupported_snapshot_recv_time_mismatch",
        )

        snapshot_after_send = index.derive_potential_fill_event(
            replace(
                supported,
                actual_new_send_time_ns=supported.maker_snapshot_recv_time_ns - 1,
            )
        )
        self.assertFalse(snapshot_after_send.outcome_supported)
        self.assertEqual(
            snapshot_after_send.potential_outcome_status,
            "unsupported_snapshot_after_actual_new",
        )

    def test_label_only_index_reuses_actual_send_raw_snapshot(self) -> None:
        orders, ticks, makerfill = _equivalence_fixtures()
        full_index = MakerFillSnapshotIndex.from_frames(ticks, makerfill)
        label_index = MakerFillLabelIndex.from_frame(makerfill)
        self.assertEqual(label_index.snapshot_count, 6)
        self.assertEqual(label_index.product_count, 1)
        self.assertGreater(label_index.estimated_size_bytes, 0)

        tick_by_sequence = {
            row["ChannelSeq"]: row for row in ticks.iter_rows(named=True)
        }
        for order_row in orders.filter(
            pl.col("maker_snapshot_channel_seq") != 99
        ).iter_rows(named=True):
            order = MakerFillScalarOrder.from_mapping(order_row)
            raw = tick_by_sequence[order.maker_snapshot_channel_seq]
            observed = MakerFillObservedSnapshot(
                value_code=raw["ValueCode"],
                channel_seq=raw["ChannelSeq"],
                recv_time_ns=int(
                    pl.Series([raw["RecvTime"]]).dt.timestamp("ns")[0]
                ),
                bid_price1=raw["BidPrice1"],
                bid_price2=raw["BidPrice2"],
                bid_lots1=raw["BidLots1"],
                bid_lots2=raw["BidLots2"],
            )
            expected = full_index.derive_potential_fill_event(order)
            actual = label_index.derive_potential_fill_event(order, observed)
            self.assertEqual(
                actual.potential_outcome_status,
                expected.potential_outcome_status,
            )
            self.assertEqual(actual.exact_target_rank, expected.exact_target_rank)
            self.assertEqual(
                actual.makerfill_potential_fill_time_ns,
                expected.makerfill_potential_fill_time_ns,
            )
            if expected.makerfill_fill_seconds is not None and math.isnan(
                expected.makerfill_fill_seconds
            ):
                self.assertTrue(math.isnan(actual.makerfill_fill_seconds))
            else:
                self.assertEqual(
                    actual.makerfill_fill_seconds,
                    expected.makerfill_fill_seconds,
                )

        order = MakerFillScalarOrder.from_mapping(orders.row(0, named=True))
        raw = ticks.row(0, named=True)
        mismatched = MakerFillObservedSnapshot(
            value_code=raw["ValueCode"],
            channel_seq=raw["ChannelSeq"] + 1,
            recv_time_ns=int(pl.Series([raw["RecvTime"]]).dt.timestamp("ns")[0]),
            bid_price1=raw["BidPrice1"],
            bid_price2=raw["BidPrice2"],
            bid_lots1=raw["BidLots1"],
            bid_lots2=raw["BidLots2"],
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            label_index.derive_potential_fill_event(order, mismatched)

        with self.assertRaisesRegex(ValueError, "duplicated"):
            MakerFillLabelIndex.from_frame(
                pl.concat([makerfill, makerfill.head(1)])
            )

    def test_terminal_must_follow_actual_new(self) -> None:
        orders, ticks, makerfill = _fixtures()
        potential = derive_potential_fill_events(
            orders.head(1), ticks, makerfill
        ).with_columns(
            pl.col("actual_new_send_time_ns").alias("actual_terminal_time_ns")
        )
        with self.assertRaisesRegex(ValueError, "after actual new"):
            classify_actual_active_intervals(potential)


if __name__ == "__main__":
    unittest.main()
