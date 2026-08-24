from __future__ import annotations

import unittest

from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.replay import (
    IndependentOrderWindow,
    TradeEvent,
    label_independent_window,
)


def _cursor(value: int) -> EventCursor:
    return EventCursor(value, 0, 0)


class IndependentReplayTest(unittest.TestCase):
    def test_bid_same_price_depletes_known_queue(self) -> None:
        window = IndependentOrderWindow(
            "g1", "bid", 100, _cursor(10), _cursor(20), 5, "retreat"
        )
        result = label_independent_window(
            window,
            [
                TradeEvent(_cursor(11), 100, 2),
                TradeEvent(_cursor(15), 100, 3),
                TradeEvent(_cursor(16), 100, 1),
            ],
        )
        self.assertTrue(result.executable_fill)
        self.assertEqual(result.fill_reason, "queue_depletion")
        self.assertEqual(result.fill_cursor, _cursor(16))

    def test_exact_visible_queue_consumption_is_not_own_fill(self) -> None:
        window = IndependentOrderWindow(
            "g1", "bid", 100, _cursor(10), _cursor(20), 5, "retreat"
        )
        result = label_independent_window(
            window, [TradeEvent(_cursor(11), 100, 5)]
        )
        self.assertFalse(result.executable_fill)
        self.assertTrue(result.touched_before_stop)

    def test_ask_trade_through_is_definite_with_unknown_queue(self) -> None:
        window = IndependentOrderWindow(
            "g1", "ask", 100, _cursor(10), _cursor(20), None, "cutoff"
        )
        result = label_independent_window(
            window,
            [TradeEvent(_cursor(12), 100, 99), TradeEvent(_cursor(13), 101, 1)],
        )
        self.assertTrue(result.executable_fill)
        self.assertEqual(result.fill_reason, "trade_through")
        self.assertFalse(result.queue_known)

    def test_unknown_queue_same_price_is_touch_not_fill(self) -> None:
        window = IndependentOrderWindow(
            "g1", "bid", 100, _cursor(10), _cursor(20), None, "cutoff"
        )
        result = label_independent_window(
            window, [TradeEvent(_cursor(12), 100, 1)]
        )
        self.assertFalse(result.executable_fill)
        self.assertTrue(result.touched_before_stop)
        self.assertEqual(result.same_price_quantity_before_stop, 1)

    def test_start_event_is_excluded_and_stop_event_is_included(self) -> None:
        window = IndependentOrderWindow(
            "g1", "bid", 100, _cursor(10), _cursor(20), 0, "retreat"
        )
        result = label_independent_window(
            window,
            [TradeEvent(_cursor(10), 99, 1), TradeEvent(_cursor(20), 99, 1)],
        )
        self.assertEqual(result.fill_cursor, _cursor(20))

    def test_trade_sorting_is_checked(self) -> None:
        window = IndependentOrderWindow(
            "g1", "bid", 100, _cursor(10), _cursor(20), 1, "cutoff"
        )
        with self.assertRaisesRegex(ValueError, "strictly"):
            label_independent_window(
                window,
                [TradeEvent(_cursor(12), 100, 1), TradeEvent(_cursor(11), 100, 1)],
            )


if __name__ == "__main__":
    unittest.main()
