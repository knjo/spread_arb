"""Contracts for the compact S1 physical spot-trade index."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from random import Random

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_spot_trade_adapter import (
    SPOT_TRADE_EVENT_SEQUENCE,
    PhysicalSpotTrade,
    SpotTradeDayIndex,
    build_spot_trade_day_index,
    build_spot_trade_day_index_from_scan,
    physical_spot_trade_source_id,
)

BASE = datetime(2026, 5, 5, 1, 0, 0)  # noqa: DTZ001


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2330", "2603", "2317"],
            "QuoteCode": ["CDFE6", "CZFE6", "DHFE6"],
        }
    )


def _row(
    code: str,
    *,
    second: int,
    channel: int,
    packet: int,
    fill_price: float,
    fill_lots: int,
    trial_match: int = 0,
    raw_value_code: str | None = None,
) -> dict[str, object]:
    return {
        "RecvTime": BASE + timedelta(seconds=second),
        "QuoteCode": code,
        "ValueCode": code if raw_value_code is None else raw_value_code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "TrialMatch": trial_match,
        "FillPrice": fill_price,
        "FillLots": fill_lots,
    }


def _rows() -> pl.DataFrame:
    return pl.from_dicts(
        [
            _row(
                "2330",
                second=0,
                channel=5,
                packet=1,
                fill_price=500.0,
                fill_lots=1,
            ),
            _row(
                "2317",
                second=0,
                channel=20,
                packet=2,
                fill_price=101.0,
                fill_lots=2,
                trial_match=1,
            ),
            _row(
                "2317",
                second=0,
                channel=10,
                packet=9,
                fill_price=100.0,
                fill_lots=3,
            ),
            _row(
                "2317",
                second=0,
                channel=30,
                packet=3,
                fill_price=0.0,
                fill_lots=8,
            ),
            _row(
                "7777",
                second=0,
                channel=1,
                packet=1,
                fill_price=50.0,
                fill_lots=9,
            ),
            _row(
                "2317",
                second=1,
                channel=21,
                packet=4,
                fill_price=102.0,
                fill_lots=1,
            ),
        ],
        infer_schema_length=None,
    )


def _ns(second: int) -> int:
    value = BASE + timedelta(seconds=second)
    return int(pl.Series([value]).dt.timestamp("ns")[0])


class SpotTradeDayIndexTest(unittest.TestCase):
    def test_keeps_positive_trade_rows_with_joint_clock_order_and_provenance(
        self,
    ) -> None:
        index = build_spot_trade_day_index(_rows(), _mapping())

        self.assertEqual(index.product_ids, ("2317", "2330", "2603"))
        self.assertEqual(index.trade_count, 4)
        self.assertEqual(index.retained_trade_count, 4)
        self.assertEqual(index.trade_count_for("2317"), 3)
        self.assertEqual(index.trade_count_for("2330"), 1)
        self.assertEqual(index.trade_count_for("2603"), 0)
        storage = index.storage_frames()
        self.assertEqual(set(storage), {"products", "trades"})
        self.assertEqual(
            index.estimated_size_bytes,
            sum(frame.estimated_size() for frame in storage.values()) + 8 * 8,
        )

        trades = index.trades_at("2317", _ns(0))
        self.assertEqual(len(trades), 2)
        self.assertTrue(all(isinstance(trade, PhysicalSpotTrade) for trade in trades))
        self.assertEqual(
            [trade.cursor for trade in trades],
            [
                EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 0),
                EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 1),
            ],
        )
        self.assertEqual(
            [trade.channel_sequence for trade in trades],
            [10, 20],
        )
        self.assertEqual([trade.packet_sequence for trade in trades], [9, 2])
        self.assertEqual([trade.source_row for trade in trades], [2, 1])
        self.assertEqual([trade.quantity_shares for trade in trades], [3_000, 2_000])
        self.assertEqual([trade.fill_lots for trade in trades], [3, 2])
        self.assertAlmostEqual(trades[0].trade_price, 100.0)
        self.assertAlmostEqual(trades[1].trade_price, 101.0)
        self.assertFalse(trades[0].trial_match)
        self.assertTrue(trades[1].trial_match)
        self.assertEqual(trades[1].quote_code, "DHFE6")
        self.assertEqual(
            trades[0].source_id,
            physical_spot_trade_source_id("2317", "DHFE6", _ns(0), 10, 9, 2),
        )

        other_product = index.trades_at("2330", _ns(0))
        self.assertEqual(len(other_product), 1)
        self.assertEqual(other_product[0].cursor.row_index, 2)
        self.assertEqual(other_product[0].source_row, 0)
        self.assertEqual(index.trades_at("2603", _ns(0)), ())

    def test_next_cursor_is_strict_phase_aware_and_deadline_inclusive(self) -> None:
        index = SpotTradeDayIndex.from_selected_rows(_rows(), _mapping())
        before_trade_phase = EventCursor(_ns(0), 0, 999)
        first = index.next_trade_cursor("2317", before_trade_phase, _ns(0))
        self.assertEqual(first, EventCursor(_ns(0), 1, 0))

        assert first is not None
        second = index.next_trade_cursor("2317", first, _ns(0))
        self.assertEqual(second, EventCursor(_ns(0), 1, 1))

        assert second is not None
        self.assertIsNone(index.next_trade_cursor("2317", second, _ns(0)))
        next_second = index.next_trade_cursor("2317", second, _ns(1))
        self.assertEqual(next_second, EventCursor(_ns(1), 1, 0))

        after_trade_phase = EventCursor(_ns(0), 100, 0)
        self.assertEqual(
            index.next_trade_cursor("2317", after_trade_phase, _ns(1)),
            EventCursor(_ns(1), 1, 0),
        )
        self.assertIsNone(index.next_trade_cursor("2603", before_trade_phase, _ns(10)))
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            index.next_trade_cursor("2317", EventCursor(_ns(1)), _ns(0))

    def test_next_cursor_at_or_above_is_inclusive_and_phase_aware(self) -> None:
        index = SpotTradeDayIndex.from_selected_rows(_rows(), _mapping())
        before_trade_phase = EventCursor(_ns(0), 0, 999)
        self.assertEqual(
            index.next_trade_cursor_at_or_above(
                "2317", before_trade_phase, _ns(0), 100.0
            ),
            EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 0),
        )
        self.assertEqual(
            index.next_trade_cursor_at_or_above(
                "2317", before_trade_phase, _ns(0), 101.0
            ),
            EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 1),
        )
        first = EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 0)
        self.assertEqual(
            index.next_trade_cursor_at_or_above("2317", first, _ns(0), 101.0),
            EventCursor(_ns(0), SPOT_TRADE_EVENT_SEQUENCE, 1),
        )
        self.assertIsNone(
            index.next_trade_cursor_at_or_above("2317", first, _ns(0), 102.0)
        )
        self.assertEqual(
            index.next_trade_cursor_at_or_above("2317", first, _ns(1), 102.0),
            EventCursor(_ns(1), SPOT_TRADE_EVENT_SEQUENCE, 0),
        )
        after_trade_phase = EventCursor(_ns(0), 100, 0)
        self.assertEqual(
            index.next_trade_cursor_at_or_above(
                "2317", after_trade_phase, _ns(1), 100.0
            ),
            EventCursor(_ns(1), SPOT_TRADE_EVENT_SEQUENCE, 0),
        )
        self.assertIsNone(
            index.next_trade_cursor_at_or_above(
                "2603", before_trade_phase, _ns(10), 1.0
            )
        )
        self.assertIsNone(
            index.next_trade_cursor_at_or_above(
                "2317", before_trade_phase, _ns(1), 1_000.0
            )
        )
        for invalid in (float("nan"), float("inf"), 0.0, -1.0, True):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite and positive"),
            ):
                index.next_trade_cursor_at_or_above(
                    "2317", before_trade_phase, _ns(1), invalid
                )
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            index.next_trade_cursor_at_or_above(
                "2317", EventCursor(_ns(1)), _ns(0), 100.0
            )

    def test_price_floor_lookup_matches_brute_force_randomized(self) -> None:
        random = Random(20260827)
        product_ids = ("1001", "1002", "1003")
        mapping = pl.DataFrame(
            {
                "ValueCode": list(product_ids),
                "QuoteCode": [f"quote-{value}" for value in product_ids],
            }
        )
        rows: list[dict[str, object]] = []
        prices = (98.5, 99.0, 100.0, 100.5, 101.0, 103.0)
        for source in range(900):
            product_id = product_ids[random.randrange(len(product_ids))]
            rows.append(
                _row(
                    product_id,
                    second=random.randrange(40),
                    channel=source,
                    packet=random.randrange(10_000),
                    fill_price=prices[random.randrange(len(prices))],
                    fill_lots=1 + random.randrange(4),
                )
            )
        index = SpotTradeDayIndex.from_selected_rows(
            pl.from_dicts(rows, infer_schema_length=None), mapping
        )
        oracle = {
            product_id: tuple(
                trade
                for second in range(40)
                for trade in index.trades_at(product_id, _ns(second))
            )
            for product_id in product_ids
        }
        after_choices: list[EventCursor] = [EventCursor(_ns(0), 0, 0)]
        for trades in oracle.values():
            for trade in trades[::17]:
                after_choices.extend(
                    (
                        EventCursor(trade.cursor.recv_time_ns, 0, 999_999),
                        trade.cursor,
                        EventCursor(trade.cursor.recv_time_ns, 100, 0),
                    )
                )

        for query in range(1_000):
            product_id = product_ids[random.randrange(len(product_ids))]
            after = after_choices[random.randrange(len(after_choices))]
            deadline = _ns(
                random.randrange(
                    max(0, int((after.recv_time_ns - _ns(0)) / 1_000_000_000)),
                    41,
                )
            )
            threshold = (*prices, 97.0, 104.0)[random.randrange(len(prices) + 2)]
            expected = next(
                (
                    trade.cursor
                    for trade in oracle[product_id]
                    if trade.cursor > after
                    and trade.cursor.recv_time_ns <= deadline
                    and trade.trade_price >= threshold
                ),
                None,
            )
            actual = index.next_trade_cursor_at_or_above(
                product_id,
                after,
                deadline,
                threshold,
            )
            self.assertEqual(actual, expected, f"random query {query}")

    def test_large_synthetic_price_queries_find_sparse_tail_match(self) -> None:
        row_count = 20_000
        rows = pl.DataFrame(
            {
                "RecvTime": [
                    BASE + timedelta(microseconds=value) for value in range(row_count)
                ],
                "QuoteCode": ["2317"] * row_count,
                "ValueCode": ["2317"] * row_count,
                "ChannelSeq": list(range(row_count)),
                "PacketSeq": list(range(row_count)),
                "TrialMatch": [0] * row_count,
                "FillPrice": [100.0] * (row_count - 1) + [500.0],
                "FillLots": [1] * row_count,
            }
        )
        index = SpotTradeDayIndex.from_selected_rows(rows, _mapping())
        deadline = int(
            pl.Series([BASE + timedelta(microseconds=row_count - 1)]).dt.timestamp(
                "ns"
            )[0]
        )
        expected = EventCursor(deadline, SPOT_TRADE_EVENT_SEQUENCE, 0)
        after = EventCursor(_ns(0), 0, 0)
        for threshold in (100.5, 499.0, 500.0):
            self.assertEqual(
                index.next_trade_cursor_at_or_above("2317", after, deadline, threshold),
                expected,
            )
        self.assertIsNone(
            index.next_trade_cursor_at_or_above("2317", after, deadline, 500.1)
        )

    def test_lazy_scan_filters_nonpositive_nonfinite_and_unmapped_rows(self) -> None:
        rows = pl.from_dicts(
            [
                _row(
                    "2317",
                    second=0,
                    channel=1,
                    packet=1,
                    fill_price=float("nan"),
                    fill_lots=1,
                ),
                _row(
                    "2317",
                    second=0,
                    channel=2,
                    packet=2,
                    fill_price=100.0,
                    fill_lots=0,
                ),
                _row(
                    "9999",
                    second=0,
                    channel=3,
                    packet=3,
                    fill_price=100.0,
                    fill_lots=1,
                ),
                _row(
                    "2317",
                    second=1,
                    channel=4,
                    packet=4,
                    fill_price=99.5,
                    fill_lots=2,
                ),
            ],
            infer_schema_length=None,
        )
        index = build_spot_trade_day_index_from_scan(rows.lazy(), _mapping())
        self.assertEqual(index.trade_count, 1)
        trade = index.trades_at("2317", _ns(1))[0]
        self.assertEqual(trade.source_row, 3)
        self.assertEqual(trade.quantity_shares, 2_000)
        self.assertAlmostEqual(trade.trade_price, 99.5)

    def test_rejects_undecidable_trade_cursor_and_raw_mapping_mismatch(self) -> None:
        first = _row(
            "2317",
            second=0,
            channel=1,
            packet=1,
            fill_price=100.0,
            fill_lots=1,
        )
        duplicate = pl.from_dicts([first, dict(first)], infer_schema_length=None)
        with self.assertRaisesRegex(ValueError, "undecidable duplicate cursor"):
            build_spot_trade_day_index(duplicate, _mapping())

        mismatch = pl.from_dicts(
            [
                _row(
                    "2317",
                    second=0,
                    channel=1,
                    packet=1,
                    fill_price=100.0,
                    fill_lots=1,
                    raw_value_code="2330",
                )
            ],
            infer_schema_length=None,
        )
        with self.assertRaisesRegex(ValueError, "exact daily mapping"):
            build_spot_trade_day_index(mismatch, _mapping())

    def test_mapping_is_exact_and_queries_reject_unknown_products(self) -> None:
        duplicate_value = pl.DataFrame(
            {"ValueCode": ["2317", "2317"], "QuoteCode": ["A", "B"]}
        )
        with self.assertRaisesRegex(ValueError, "one-to-one on product_id"):
            build_spot_trade_day_index(_rows(), duplicate_value)

        duplicate_quote = pl.DataFrame(
            {"ValueCode": ["2317", "2330"], "QuoteCode": ["A", "A"]}
        )
        with self.assertRaisesRegex(ValueError, "one-to-one on quote_code"):
            build_spot_trade_day_index(_rows(), duplicate_quote)

        index = build_spot_trade_day_index(_rows(), _mapping())
        with self.assertRaisesRegex(ValueError, "exact spot-trade mapping"):
            index.trades_at("9999", _ns(0))

    def test_invalid_cursor_payload_and_schema_are_rejected(self) -> None:
        negative_channel = pl.from_dicts(
            [
                _row(
                    "2317",
                    second=0,
                    channel=-1,
                    packet=1,
                    fill_price=100.0,
                    fill_lots=1,
                )
            ],
            infer_schema_length=None,
        )
        with self.assertRaisesRegex(ValueError, "invalid cursor or payload"):
            build_spot_trade_day_index(negative_channel, _mapping())

        float_lots = _rows().with_columns(pl.col("FillLots").cast(pl.Float64))
        with self.assertRaisesRegex(TypeError, "FillLots must be an integer"):
            build_spot_trade_day_index(float_lots, _mapping())

        non_datetime = _rows().with_columns(pl.col("RecvTime").dt.timestamp("ns"))
        with self.assertRaisesRegex(TypeError, "RecvTime must be a Datetime"):
            build_spot_trade_day_index(non_datetime, _mapping())

    def test_returned_trade_is_immutable_and_source_id_is_checked(self) -> None:
        trade = build_spot_trade_day_index(_rows(), _mapping()).trades_at(
            "2317", _ns(0)
        )[0]
        with self.assertRaises(FrozenInstanceError):
            trade.quantity_shares = 1  # type: ignore[misc]

        with self.assertRaisesRegex(ValueError, "source_id"):
            PhysicalSpotTrade(
                product_id=trade.product_id,
                quote_code=trade.quote_code,
                cursor=trade.cursor,
                trade_price=trade.trade_price,
                quantity_shares=trade.quantity_shares,
                source_id="wrong",
                channel_sequence=trade.channel_sequence,
                packet_sequence=trade.packet_sequence,
                source_row=trade.source_row,
                trial_match=trade.trial_match,
            )


if __name__ == "__main__":
    unittest.main()
