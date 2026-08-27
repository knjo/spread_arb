"""Contracts for the compact raw-book S1 entry bridge."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_raw_book_adapter import build_raw_book_day_index
from ..quote_fill.s1_raw_entry_provider import S1RawEntryBookProvider

BASE = datetime(2026, 5, 5, 1, 0, 0)  # noqa: DTZ001


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.5],
        }
    )


def _row(
    *,
    code: str,
    second: int,
    channel: int,
    packet: int,
    future: bool = False,
    trial: int = 0,
    **updates: object,
) -> dict[str, object]:
    recv_time = BASE + timedelta(seconds=second)
    if future:
        recv_time = recv_time.replace(tzinfo=UTC)
    row: dict[str, object] = {
        "RecvTime": recv_time,
        "QuoteCode": code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "TrialMatch": trial,
        "BestBidPrice": 0,
        "BestBidLots": 0,
        "BestAskPrice": 0,
        "BestAskLots": 0,
    }
    for side in ("Bid", "Ask"):
        for level in range(1, 6):
            row[f"{side}Price{level}"] = 0
            row[f"{side}Lots{level}"] = 0
    if future:
        row["DecimalLocator"] = 2
    row.update(updates)
    return row


def _index():
    spot = pl.from_dicts(
        [
            _row(
                code="2317",
                second=0,
                channel=10,
                packet=1,
                BidPrice1=100.0,
                BidLots1=2,
                BidPrice2=99.5,
                BidLots2=3,
                AskPrice1=101.0,
                AskLots1=4,
            ),
            _row(
                code="2317",
                second=1,
                channel=11,
                packet=2,
                trial=1,
            ),
            _row(
                code="2317",
                second=2,
                channel=12,
                packet=3,
                BestBidPrice=99.0,
                BestBidLots=5,
                BestAskPrice=102.0,
                BestAskLots=6,
            ),
        ],
        infer_schema_length=None,
    )
    future = pl.from_dicts(
        [
            _row(
                code="DHFB6",
                second=0,
                channel=20,
                packet=4,
                future=True,
                BidPrice1=10_000,
                BidLots1=2,
                AskPrice1=10_100,
                AskLots1=3,
            )
        ],
        infer_schema_length=None,
    )
    return build_raw_book_day_index(spot, future, _mapping())


def _ns(second: int) -> int:
    return int(pl.Series([BASE + timedelta(seconds=second)]).dt.timestamp("ns")[0])


class S1RawEntryBookProviderTest(unittest.TestCase):
    def test_formal_source_becomes_exact_makerfill_snapshot(self) -> None:
        provider = S1RawEntryBookProvider(_index())
        snapshot = provider.maker_snapshot_as_of("2317", EventCursor(_ns(0), 100))
        assert snapshot is not None
        self.assertEqual(snapshot.channel_seq, 10)
        self.assertEqual(snapshot.recv_time_ns, _ns(0))
        self.assertAlmostEqual(snapshot.bid_price1 or 0.0, 100.0)
        self.assertAlmostEqual(snapshot.bid_price2 or 0.0, 99.5)
        self.assertEqual(snapshot.bid_lots1, 2)
        self.assertEqual(snapshot.bid_lots2, 3)

        future = provider.state_as_of("future", "2317", EventCursor(_ns(0), 100))
        assert future is not None
        self.assertTrue(future.gate_open)
        self.assertAlmostEqual(future.bids[0].price, 100.0)

    def test_trial_is_not_a_makerfill_snapshot(self) -> None:
        provider = S1RawEntryBookProvider(_index())
        cursor = EventCursor(_ns(1), 100)
        self.assertIsNone(provider.maker_snapshot_as_of("2317", cursor))
        state = provider.state_as_of("spot", "2317", cursor)
        assert state is not None
        self.assertFalse(state.gate_open)

    def test_best_only_source_preserves_identity_but_has_no_l1_rank(self) -> None:
        provider = S1RawEntryBookProvider(_index())
        snapshot = provider.maker_snapshot_as_of("2317", EventCursor(_ns(2), 100))
        assert snapshot is not None
        self.assertEqual(snapshot.channel_seq, 12)
        self.assertIsNone(snapshot.bid_price1)
        self.assertIsNone(snapshot.bid_price2)
        self.assertEqual(snapshot.bid_lots1, 0)
        self.assertEqual(snapshot.bid_lots2, 0)

    def test_unknown_product_fails_closed(self) -> None:
        provider = S1RawEntryBookProvider(_index())
        with self.assertRaisesRegex(ValueError, "exact raw-book mapping"):
            provider.maker_snapshot_as_of("9999", EventCursor(_ns(0)))


if __name__ == "__main__":
    unittest.main()
