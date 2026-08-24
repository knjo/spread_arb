from __future__ import annotations

import random
import unittest

from maker.src.quote_fill.indexed_replay import (
    DEFAULT_SHADOW_HORIZONS_MS,
    IndexedTradeReplay,
    label_independent_windows_indexed,
)
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.replay import (
    IndependentOrderWindow,
    TradeEvent,
    label_independent_window,
)


def _cursor(value: int, event_sequence: int = 0) -> EventCursor:
    return EventCursor(value, event_sequence, 0)


def _window(
    generation_id: str,
    side: str,
    price: int,
    start: EventCursor,
    stop: EventCursor,
    queue: int | None,
) -> IndependentOrderWindow:
    return IndependentOrderWindow(
        generation_id,
        side,  # type: ignore[arg-type]
        price,
        start,
        stop,
        queue,
        "target_retreat",
    )


class IndexedReplayTest(unittest.TestCase):
    def test_matches_scalar_bid_and_ask_labels(self) -> None:
        trades = [
            TradeEvent(_cursor(11), 100, 2),
            TradeEvent(_cursor(12), 101, 7),
            TradeEvent(_cursor(13), 100, 3),
            TradeEvent(_cursor(14), 99, 1),
            TradeEvent(_cursor(15), 102, 1),
        ]
        windows = [
            _window("bid-queue", "bid", 100, _cursor(10), _cursor(20), 4),
            _window("bid-unknown", "bid", 100, _cursor(10), _cursor(20), None),
            _window("ask-through", "ask", 100, _cursor(10), _cursor(20), None),
            _window("ask-queue", "ask", 101, _cursor(10), _cursor(14), 6),
        ]
        indexed = IndexedTradeReplay(trades).label_windows(
            windows, shadow_horizons_ms=()
        )

        for window, result in zip(windows, indexed):
            scalar = label_independent_window(window, trades)
            for field in scalar.__dataclass_fields__:
                self.assertEqual(
                    getattr(result, field),
                    getattr(scalar, field),
                    (window.generation_id, field),
                )

    def test_first_fill_wins_and_quantity_stops_at_fill(self) -> None:
        trades = [
            TradeEvent(_cursor(11), 100, 2),
            TradeEvent(_cursor(12), 99, 1),
            TradeEvent(_cursor(13), 100, 50),
        ]
        result = IndexedTradeReplay(trades).label_window(
            _window("g", "bid", 100, _cursor(10), _cursor(20), 5),
            shadow_horizons_ms=(),
        )
        self.assertEqual(result.fill_cursor, _cursor(12))
        self.assertEqual(result.fill_reason, "trade_through")
        self.assertEqual(result.same_price_quantity_before_stop, 2)
        self.assertTrue(result.trade_through_before_stop)

        queue_first_trades = [
            TradeEvent(_cursor(11), 100, 6),
            TradeEvent(_cursor(12), 99, 1),
        ]
        queue_first = IndexedTradeReplay(queue_first_trades).label_window(
            _window("g2", "bid", 100, _cursor(10), _cursor(20), 5),
            shadow_horizons_ms=(),
        )
        self.assertEqual(queue_first.fill_cursor, _cursor(11))
        self.assertEqual(queue_first.fill_reason, "queue_depletion")
        self.assertFalse(queue_first.trade_through_before_stop)

    def test_exact_visible_queue_is_not_a_fill(self) -> None:
        result = IndexedTradeReplay(
            [TradeEvent(_cursor(11), 100, 5)]
        ).label_window(
            _window("g", "bid", 100, _cursor(10), _cursor(20), 5),
            shadow_horizons_ms=(),
        )
        self.assertFalse(result.executable_fill)
        self.assertTrue(result.touched_before_stop)

    def test_quantity_path_distinguishes_partial_full_and_unknown(self) -> None:
        replay = IndexedTradeReplay(
            [
                TradeEvent(_cursor(11), 100, 5),
                TradeEvent(_cursor(12), 100, 1),
                TradeEvent(_cursor(13), 100, 1),
            ]
        )
        window = _window("g", "bid", 100, _cursor(10), _cursor(12), 5)
        partial = replay.label_quantity(window, 2)
        self.assertTrue(partial.any_fill)
        self.assertFalse(partial.full_fill)
        self.assertTrue(partial.partial_fill)
        self.assertEqual(partial.known_filled_quantity_before_stop, 1)
        self.assertEqual(partial.first_fill_cursor, _cursor(12))

        full = replay.label_quantity(
            _window("g", "bid", 100, _cursor(10), _cursor(13), 5), 2
        )
        self.assertTrue(full.full_fill)
        self.assertEqual(full.full_fill_cursor, _cursor(13))

        unknown = replay.label_quantity(
            _window("u", "bid", 100, _cursor(10), _cursor(13), None), 2
        )
        self.assertIsNone(unknown.any_fill)
        self.assertIsNone(unknown.known_filled_quantity_before_stop)

    def test_trade_through_fills_remaining_quantity_with_unknown_queue(self) -> None:
        replay = IndexedTradeReplay([TradeEvent(_cursor(12), 99, 1)])
        label = replay.label_quantity(
            _window("g", "bid", 100, _cursor(10), _cursor(20), None), 2
        )
        self.assertTrue(label.full_fill)
        self.assertEqual(label.known_filled_quantity_before_stop, 2)
        self.assertTrue(label.trade_through_fill)

    def test_unknown_queue_same_price_is_only_a_touch(self) -> None:
        result = IndexedTradeReplay(
            [TradeEvent(_cursor(12), 100, 9)]
        ).label_window(
            _window("g", "bid", 100, _cursor(10), _cursor(20), None),
            shadow_horizons_ms=(),
        )
        self.assertFalse(result.executable_fill)
        self.assertTrue(result.touched_before_stop)
        self.assertEqual(result.same_price_quantity_before_stop, 9)

    def test_active_boundaries_exclude_start_and_include_stop(self) -> None:
        trades = [
            TradeEvent(_cursor(10), 99, 1),
            TradeEvent(_cursor(20), 99, 1),
            TradeEvent(_cursor(21), 99, 1),
        ]
        result = IndexedTradeReplay(trades).label_window(
            _window("g", "bid", 100, _cursor(10), _cursor(20), None),
            shadow_horizons_ms=(),
        )
        self.assertEqual(result.fill_cursor, _cursor(20))

    def test_shadow_horizons_are_time_based_and_stop_exclusive(self) -> None:
        millisecond = 1_000_000
        stop = _cursor(100 * millisecond, 2)
        trades = [
            # Same recv time but later merged sequence is after cancellation.
            TradeEvent(_cursor(100 * millisecond, 3), 100, 1),
            TradeEvent(_cursor(110 * millisecond), 100, 2),
            TradeEvent(_cursor(140 * millisecond), 99, 1),
            TradeEvent(_cursor(600 * millisecond), 98, 1),
            TradeEvent(_cursor(601 * millisecond), 99, 1),
        ]
        result = IndexedTradeReplay(trades).label_window(
            _window("g", "bid", 100, _cursor(90 * millisecond), stop, None)
        )

        shadow_10 = result.shadow(10)
        self.assertTrue(shadow_10.touched_after_stop)
        self.assertFalse(shadow_10.trade_through_after_stop)
        self.assertEqual(shadow_10.first_touch_cursor, _cursor(100 * millisecond, 3))
        self.assertEqual(shadow_10.same_price_quantity_after_stop, 3)

        shadow_50 = result.shadow(50)
        self.assertTrue(shadow_50.trade_through_after_stop)
        self.assertEqual(shadow_50.first_trade_through_cursor, _cursor(140 * millisecond))
        # Exactly +500 ms is included; +501 ms is excluded.
        shadow_500 = result.shadow(500)
        self.assertEqual(
            shadow_500.first_trade_through_cursor, _cursor(140 * millisecond)
        )
        self.assertEqual(
            tuple(label.horizon_ms for label in result.shadow_labels),
            DEFAULT_SHADOW_HORIZONS_MS,
        )

        exact_deadline = IndexedTradeReplay(trades).label_window(
            _window("deadline", "bid", 99, _cursor(90 * millisecond), stop, None),
            shadow_horizons_ms=(499, 500),
        )
        self.assertFalse(exact_deadline.shadow(499).trade_through_after_stop)
        self.assertTrue(exact_deadline.shadow(500).trade_through_after_stop)
        self.assertEqual(
            exact_deadline.shadow(500).first_trade_through_cursor,
            _cursor(600 * millisecond),
        )

    def test_ask_shadow_uses_greater_price_as_trade_through(self) -> None:
        millisecond = 1_000_000
        stop = _cursor(100 * millisecond)
        trades = [
            TradeEvent(_cursor(105 * millisecond), 99, 1),
            TradeEvent(_cursor(106 * millisecond), 100, 1),
            TradeEvent(_cursor(107 * millisecond), 101, 1),
        ]
        shadow = IndexedTradeReplay(trades).label_window(
            _window("g", "ask", 100, _cursor(90 * millisecond), stop, None),
            shadow_horizons_ms=(10,),
        ).shadow(10)
        self.assertEqual(shadow.first_touch_cursor, _cursor(106 * millisecond))
        self.assertEqual(
            shadow.first_trade_through_cursor, _cursor(107 * millisecond)
        )

    def test_horizons_are_validated_deduplicated_and_sorted(self) -> None:
        replay = IndexedTradeReplay([])
        result = replay.label_window(
            _window("g", "bid", 100, _cursor(10), _cursor(20), None),
            shadow_horizons_ms=(50, 10, 50),
        )
        self.assertEqual(
            tuple(label.horizon_ms for label in result.shadow_labels), (10, 50)
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replay.label_window(
                _window("g", "bid", 100, _cursor(10), _cursor(20), None),
                shadow_horizons_ms=(0,),
            )

    def test_sorting_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly"):
            IndexedTradeReplay(
                [
                    TradeEvent(_cursor(12), 100, 1),
                    TradeEvent(_cursor(11), 100, 1),
                ]
            )
        with self.assertRaisesRegex(ValueError, "strictly"):
            IndexedTradeReplay(
                [
                    TradeEvent(_cursor(12), 100, 1),
                    TradeEvent(_cursor(12), 101, 1),
                ]
            )

    def test_batch_wrapper_preserves_window_order(self) -> None:
        trades = [TradeEvent(_cursor(12), 99, 1)]
        windows = [
            _window("second", "bid", 98, _cursor(10), _cursor(20), None),
            _window("first", "bid", 100, _cursor(10), _cursor(20), None),
        ]
        results = label_independent_windows_indexed(
            windows, trades, shadow_horizons_ms=()
        )
        self.assertEqual(
            tuple(result.generation_id for result in results), ("second", "first")
        )
        self.assertFalse(results[0].executable_fill)
        self.assertTrue(results[1].executable_fill)

    def test_randomized_active_labels_match_scalar_reference(self) -> None:
        randomizer = random.Random(20260814)
        trades = [
            TradeEvent(
                _cursor(index * 10),
                randomizer.randint(96, 104),
                randomizer.randint(1, 8),
            )
            for index in range(1, 201)
        ]
        replay = IndexedTradeReplay(trades)
        windows: list[IndependentOrderWindow] = []
        for generation in range(250):
            start_index = randomizer.randint(0, 180)
            stop_index = randomizer.randint(start_index + 1, 201)
            windows.append(
                _window(
                    str(generation),
                    randomizer.choice(("bid", "ask")),
                    randomizer.randint(96, 104),
                    _cursor(start_index * 10),
                    _cursor(stop_index * 10),
                    randomizer.choice((None, 0, 1, 5, 20)),
                )
            )

        indexed = replay.label_windows(windows, shadow_horizons_ms=())
        for window, result in zip(windows, indexed):
            scalar = label_independent_window(window, trades)
            for field in scalar.__dataclass_fields__:
                self.assertEqual(
                    getattr(result, field),
                    getattr(scalar, field),
                    (window.generation_id, field),
                )


if __name__ == "__main__":
    unittest.main()
