from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.raw_tape import (
    extract_trade_tape,
    load_raw_tape_day,
    normalize_future_tape,
    normalize_spot_tape,
)


DATE = "20260128"


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "spot_ref_price": [225.0],
            "fut_ref_price": [226.0],
            "contract_size": [2000.0],
        }
    )


def _raw(market: str) -> pl.DataFrame:
    aware = market == "future"
    recv = [
        datetime(2026, 1, 28, 1, 5, 0, tzinfo=timezone.utc if aware else None),
        datetime(2026, 1, 28, 1, 5, 0, 50_000, tzinfo=timezone.utc if aware else None),
        datetime(2026, 1, 28, 1, 5, 0, 100_000, tzinfo=timezone.utc if aware else None),
        datetime(2026, 1, 28, 1, 5, 0, 150_000, tzinfo=timezone.utc if aware else None),
    ]
    scale = 100 if aware else 1
    data: dict[str, object] = {
        "RecvTime": recv,
        "TransTime": [
            datetime(2026, 1, 28, 9, 4, 59),
            datetime(2026, 1, 28, 9, 5, 0),
            datetime(2026, 1, 28, 9, 5, 0, 50_000),
            datetime(2026, 1, 28, 9, 5, 0, 100_000),
        ],
        "QuoteCode": ["DHFB6" if aware else "2317"] * 4,
        "PacketSeq": [1, 2, 3, 4],
        "ChannelSeq": [10, 11, 12, 13],
        "TrialMatch": [0, 0, 1, 0],
        "TotalFillLots": [0, 0, 1, 1],
        "FillPrice": [0, 0, 101 * scale, 0],
        "FillLots": [0, 0, 2, 0],
        "BestBidPrice": [0, 100 * scale, 0, 102 * scale],
        "BestBidLots": [0, 9, 0, 12],
        "BestAskPrice": [0, 101 * scale, 0, 104 * scale],
        "BestAskLots": [0, 4, 0, 7],
    }
    for side, prices, lots in (
        ("Bid", [100, 99, 98, 97, 0], [5, 4, 3, 2, 0]),
        ("Ask", [102, 103, 104, 105, 0], [6, 5, 4, 3, 0]),
    ):
        for level in range(1, 6):
            value = prices[level - 1] * scale
            # First row is outside session; second is a full snapshot, third a
            # zero-book trade, fourth a Best-only update.
            data[f"{side}Price{level}"] = [value, value, 0, 0]
            data[f"{side}Lots{level}"] = [lots[level - 1], lots[level - 1], 0, 0]
    if aware:
        data["DecimalLocator"] = [2] * 4
    return pl.DataFrame(data)


class RawTapeNormalizationTest(unittest.TestCase):
    def test_future_zero_book_trade_inherits_complete_snapshot(self) -> None:
        states = normalize_future_tape(_raw("future"), _mapping(), DATE)
        self.assertEqual(states.height, 3)  # 09:04:59 was removed
        trade = states.row(1, named=True)
        self.assertFalse(trade["raw_has_book"])
        self.assertEqual(trade["fill_price"], 101.0)
        self.assertEqual(trade["bid_price_1"], 100.0)
        self.assertEqual(trade["ask_price_1"], 102.0)
        self.assertIsNone(trade["bid_price_5"])
        self.assertEqual(trade["book_sequence"], 11)
        self.assertEqual(trade["recv_time"].tzinfo, None)
        expected_ns = int(
            (trade["recv_time"] - datetime(1970, 1, 1)).total_seconds() * 1e9
        )
        self.assertEqual(trade["recv_time_ns"], expected_ns)
        self.assertTrue(trade["trial_match"])

    def test_future_exec_uses_l1_and_best_then_best_only_update(self) -> None:
        states = normalize_future_tape(_raw("future"), _mapping(), DATE)
        first = states.row(0, named=True)
        self.assertEqual(first["exec_bid_price"], 100.0)
        self.assertEqual(first["exec_bid_lots"], 9)  # same price -> max lots
        self.assertEqual(first["exec_ask_price"], 101.0)
        self.assertEqual(first["exec_ask_lots"], 4)
        best_only = states.row(2, named=True)
        self.assertEqual(best_only["bid_price_1"], 100.0)
        self.assertEqual(best_only["best_bid_price"], 102.0)
        self.assertEqual(best_only["exec_bid_price"], 102.0)
        self.assertEqual(best_only["exec_bid_lots"], 12)
        self.assertEqual(best_only["exec_ask_price"], 102.0)
        self.assertEqual(best_only["exec_ask_lots"], 6)

    def test_spot_exec_is_l1_and_trade_projection_retains_cursor(self) -> None:
        states = normalize_spot_tape(_raw("spot"), _mapping(), DATE)
        row = states.row(0, named=True)
        self.assertEqual(row["QuoteCode"], "DHFB6")
        self.assertEqual(row["instrument_code"], "2317")
        self.assertEqual(row["exec_bid_price"], 100.0)
        self.assertEqual(row["exec_bid_lots"], 5)
        self.assertEqual(row["exec_ask_price"], 102.0)
        trades = extract_trade_tape(states)
        self.assertEqual(trades.height, 1)
        self.assertEqual(trades.item(0, "trade_price"), 101.0)
        self.assertEqual(trades.item(0, "trade_lots"), 2)
        self.assertEqual(trades.item(0, "book_sequence"), 11)

    def test_file_loader_filters_exact_mapping_and_reports_zero_book_trade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spot_path = root / "spot.parquet"
            future_path = root / "future.parquet"
            _raw("spot").write_parquet(spot_path)
            _raw("future").write_parquet(future_path)
            day = load_raw_tape_day(
                DATE,
                _mapping(),
                spot_path=spot_path,
                future_path=future_path,
            )
        self.assertEqual(day.spot_states.height, 3)
        self.assertEqual(day.future_states.height, 3)
        self.assertEqual(day.future_trades.height, 1)
        future_audit = day.audit.filter(pl.col("market") == "future")
        self.assertEqual(future_audit.item(0, "zero_book_trade_rows"), 1)
        self.assertEqual(future_audit.item(0, "rows_without_prior_book"), 0)


if __name__ == "__main__":
    unittest.main()
