from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.study import _trade_events
from maker.src.quote_fill.targets import (
    PRICE_LADDER_VERSION,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    price_in_ref_band,
    round_down_to_tick,
    round_up_to_tick,
    target_price_for_basis,
)


class QuoteFillTargetTest(unittest.TestCase):
    def test_four_route_rounding_inequalities(self) -> None:
        upper = 70.0
        lower = 30.0
        fa = target_price_for_basis(
            "future_ask_spot_taker", upper, spot_ask=100.0
        )
        sb = target_price_for_basis(
            "spot_bid_future_taker", upper, fut_exec_bid=101.0
        )
        fb = target_price_for_basis(
            "future_bid_spot_taker", lower, spot_bid=99.5
        )
        sa = target_price_for_basis(
            "spot_ask_future_taker", lower, fut_exec_ask=100.5
        )
        self.assertGreaterEqual(
            effective_basis_bp("future_ask_spot_taker", fa, spot_ask=100.0),
            upper,
        )
        self.assertGreaterEqual(
            effective_basis_bp("spot_bid_future_taker", sb, fut_exec_bid=101.0),
            upper,
        )
        self.assertLessEqual(
            effective_basis_bp("future_bid_spot_taker", fb, spot_bid=99.5),
            lower,
        )
        self.assertLessEqual(
            effective_basis_bp("spot_ask_future_taker", sa, fut_exec_ask=100.5),
            lower,
        )

    def test_ladder_boundaries_and_exact_index(self) -> None:
        self.assertEqual(round_up_to_tick(9.999), 10.0)
        self.assertEqual(round_down_to_tick(10.049), 10.0)
        self.assertEqual(round_up_to_tick(49.99), 50.0)
        self.assertEqual(round_up_to_tick(99.99), 100.0)
        self.assertEqual(round_up_to_tick(499.9), 500.0)
        self.assertEqual(round_up_to_tick(999.9), 1000.0)
        self.assertIsInstance(round_up_to_tick(500.1), float)
        self.assertIsInstance(round_up_to_tick(1000.1), float)
        self.assertEqual(absolute_price_tick(100.0), 2300)
        with self.assertRaises(ValueError):
            absolute_price_tick(100.1)

    def test_stock_future_high_price_ladder_changes_only_on_20260706(self) -> None:
        self.assertEqual(
            PRICE_LADDER_VERSION,
            "tw_stock_spot_v1_future_20260706_v2",
        )
        with self.assertRaisesRegex(ValueError, "off-ladder"):
            absolute_price_tick(
                2131.0,
                market="future",
                session_date="20260703",
            )
        with self.assertRaisesRegex(ValueError, "off-ladder"):
            absolute_price_tick(
                2131.0,
                market="spot",
                session_date="20260706",
            )
        self.assertEqual(
            absolute_price_tick(
                2131.0,
                market="future",
                session_date="20260706",
            ),
            4731,
        )
        self.assertEqual(
            round_up_to_tick(
                2130.1,
                market="future",
                session_date="20260703",
            ),
            2135.0,
        )
        self.assertEqual(
            round_up_to_tick(
                2130.1,
                market="future",
                session_date="20260706",
            ),
            2131.0,
        )

    def test_post_change_future_target_and_trade_share_one_tick_index(self) -> None:
        target = target_price_for_basis(
            "future_ask_spot_taker",
            1.0,
            session_date="20260706",
            spot_ask=2130.0,
        )
        self.assertEqual(target, 2131.0)
        target_tick = absolute_price_tick(
            target,
            market="future",
            session_date="20260706",
        )
        trade = pl.DataFrame(
            {
                "Date": ["20260706"],
                "recv_time_ns": [1],
                "sequence": [2],
                "packet_sequence": [3],
                "trade_price": [2131.0],
                "trade_lots": [1],
            }
        )
        self.assertEqual(
            _trade_events(trade, market="future")[0].price_tick,
            target_tick,
        )

        pre_change = trade.with_columns(pl.lit("20260703").alias("Date"))
        with self.assertRaisesRegex(ValueError, "off-ladder price: 2131.0"):
            _trade_events(pre_change, market="future")

    def test_strict_ref_band(self) -> None:
        self.assertFalse(price_in_ref_band(91.0, 100.0))
        self.assertFalse(price_in_ref_band(108.0, 100.0))
        self.assertTrue(price_in_ref_band(91.01, 100.0))
        self.assertTrue(price_in_ref_band(107.99, 100.0))

    def test_passivity_is_route_specific(self) -> None:
        self.assertTrue(
            is_passive_target(
                "future_ask_spot_taker", 101.0, fut_exec_bid=100.5
            )
        )
        self.assertTrue(
            is_passive_target("spot_bid_future_taker", 99.5, spot_ask=100.0)
        )
        self.assertTrue(
            is_passive_target("future_bid_spot_taker", 99.5, fut_exec_ask=100.0)
        )
        self.assertTrue(
            is_passive_target("spot_ask_future_taker", 100.5, spot_bid=100.0)
        )


if __name__ == "__main__":
    unittest.main()
