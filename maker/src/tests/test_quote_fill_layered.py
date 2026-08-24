"""Synthetic tests for layered SpreadPair-epoch order admission."""

from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor, LayeredSampler


def _cursor(sequence: int, *, recv_time_ns: int = 100) -> EventCursor:
    return EventCursor(recv_time_ns, sequence, 0)


def _ticks(sampler: LayeredSampler) -> list[int]:
    return [order.absolute_price_tick for order in sampler.active_orders]


class EventCursorTest(unittest.TestCase):
    def test_cursor_has_deterministic_lexicographic_order(self) -> None:
        self.assertLess(EventCursor(100, 1, 0), EventCursor(100, 1, 1))
        self.assertLess(EventCursor(100, 1, 1), EventCursor(100, 2, 0))
        self.assertLess(EventCursor(100, 2, 0), EventCursor(101, 0, 0))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EventCursor(100, -1, 0)

    def test_sampler_rejects_duplicate_or_backwards_events(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        sampler.reconcile(_cursor(1), 1, 100)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sampler.reconcile(_cursor(1), 1, 100)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sampler.reconcile(_cursor(0), 1, 100)

    def test_invalid_observation_does_not_consume_its_cursor(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        sampler.reconcile(_cursor(1), 2, 100)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            sampler.reconcile(_cursor(2), 1, 100)

        actions = sampler.reconcile(_cursor(2), 2, 101)
        self.assertEqual(actions[0].reason, "forward_new_price")


class BidLayeredSamplerTest(unittest.TestCase):
    def test_forward_adds_retreat_prunes_and_seen_price_does_not_reopen(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")

        base = sampler.reconcile(_cursor(1), 21, 100)
        self.assertEqual([(item.kind, item.reason) for item in base], [
            ("submit", "new_epoch")
        ])
        self.assertEqual(sampler.reconcile(_cursor(2), 21, 100), ())

        forward = sampler.reconcile(_cursor(3), 21, 101)
        self.assertEqual([(item.kind, item.reason) for item in forward], [
            ("submit", "forward_new_price")
        ])
        self.assertEqual(_ticks(sampler), [100, 101])
        self.assertEqual(sampler.seen_price_ticks, frozenset({100, 101}))

        retreat = sampler.reconcile(_cursor(4), 21, 100)
        self.assertEqual([(item.kind, item.reason) for item in retreat], [
            ("cancel", "target_retreat")
        ])
        self.assertEqual(retreat[0].order.absolute_price_tick, 101)
        self.assertEqual(_ticks(sampler), [100])

        repeated_forward = sampler.reconcile(_cursor(5), 21, 101)
        self.assertEqual(
            [(item.kind, item.reason) for item in repeated_forward],
            [("suppress", "seen_price_in_epoch")],
        )
        self.assertEqual(_ticks(sampler), [100])
        self.assertEqual(len(sampler.all_orders), 2)

    def test_new_epoch_retreat_cancels_first_then_admits_new_base(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        old = sampler.reconcile(_cursor(1), 21, 101)[0].order

        actions = sampler.reconcile(_cursor(2), 22, 100)
        self.assertEqual(
            [(item.kind, item.reason) for item in actions],
            [("cancel", "target_retreat"), ("submit", "new_epoch")],
        )
        self.assertEqual(actions[0].order, old)
        self.assertEqual(actions[1].order.spread_pair_epoch, 22)
        self.assertEqual(_ticks(sampler), [100])

    def test_cross_epoch_same_price_is_an_independent_live_generation(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        first = sampler.reconcile(_cursor(1), 21, 100)[0].order
        second = sampler.reconcile(_cursor(2), 22, 100)[0].order

        self.assertNotEqual(first.generation, second.generation)
        self.assertEqual(first.absolute_price_tick, second.absolute_price_tick)
        self.assertEqual(_ticks(sampler), [100, 100])
        self.assertEqual(
            [order.spread_pair_epoch for order in sampler.active_orders], [21, 22]
        )
        self.assertNotEqual(first.sampling_key, second.sampling_key)

    def test_retreat_to_unseen_price_does_not_submit(self) -> None:
        sampler = LayeredSampler("future_bid_spot_taker")
        sampler.reconcile(_cursor(1), 1, 100)
        sampler.reconcile(_cursor(2), 1, 102)

        actions = sampler.reconcile(_cursor(3), 1, 101)
        self.assertEqual([(item.kind, item.reason) for item in actions], [
            ("cancel", "target_retreat")
        ])
        self.assertEqual(actions[0].order.absolute_price_tick, 102)
        self.assertEqual(_ticks(sampler), [100])
        self.assertNotIn(101, sampler.seen_price_ticks)


class AskLayeredSamplerTest(unittest.TestCase):
    def test_ask_direction_is_the_exact_bid_mirror(self) -> None:
        sampler = LayeredSampler("future_ask_spot_taker")
        sampler.reconcile(_cursor(1), 1, 101)

        forward = sampler.reconcile(_cursor(2), 1, 100)
        self.assertEqual(forward[0].kind, "submit")
        self.assertEqual(_ticks(sampler), [101, 100])

        retreat = sampler.reconcile(_cursor(3), 1, 101)
        self.assertEqual([(item.kind, item.reason) for item in retreat], [
            ("cancel", "target_retreat")
        ])
        self.assertEqual(retreat[0].order.absolute_price_tick, 100)
        self.assertEqual(_ticks(sampler), [101])

        repeated_forward = sampler.reconcile(_cursor(4), 1, 100)
        self.assertEqual(repeated_forward[0].kind, "suppress")
        self.assertEqual(_ticks(sampler), [101])


class GateAndLifecycleTest(unittest.TestCase):
    def test_gate_cancels_every_generation_and_does_not_reopen_seen_price(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        sampler.reconcile(_cursor(1), 1, 100)
        sampler.reconcile(_cursor(2), 1, 101)

        closed = sampler.reconcile(
            _cursor(3), 1, 101, gate_open=False, gate_reason="trial_match"
        )
        self.assertEqual(
            [(item.kind, item.reason) for item in closed],
            [("cancel", "trial_match"), ("cancel", "trial_match")],
        )
        self.assertEqual(sampler.active_orders, ())
        self.assertEqual(sampler.reconcile(_cursor(4), 1, 101), ())

        new_forward = sampler.reconcile(_cursor(5), 1, 102)
        self.assertEqual(new_forward[0].reason, "forward_new_price")
        self.assertEqual(_ticks(sampler), [102])

    def test_gate_closed_on_new_epoch_consumes_no_delayed_base(self) -> None:
        sampler = LayeredSampler("spot_ask_future_taker")
        sampler.reconcile(_cursor(1), 1, 101)
        canceled = sampler.reconcile(
            _cursor(2), 2, None, gate_open=False, gate_reason="book_gate"
        )
        self.assertEqual(canceled[0].reason, "book_gate")
        self.assertEqual(sampler.current_epoch, 2)

        self.assertEqual(sampler.reconcile(_cursor(3), 2, 101), ())
        base = sampler.reconcile(_cursor(4), 3, 101)
        self.assertEqual(base[0].reason, "new_epoch")

    def test_terminal_fill_removes_only_the_selected_generation(self) -> None:
        sampler = LayeredSampler("spot_bid_future_taker")
        first = sampler.reconcile(_cursor(1), 1, 100)[0].order
        second = sampler.reconcile(_cursor(2), 2, 100)[0].order

        terminal = sampler.mark_terminal(_cursor(3), first.generation)
        self.assertEqual((terminal.kind, terminal.reason), ("terminal", "filled"))
        self.assertEqual([order.generation for order in sampler.active_orders], [
            second.generation
        ])

        with self.assertRaisesRegex(ValueError, "not active"):
            sampler.mark_terminal(_cursor(4), first.generation)
        sampler.reconcile(_cursor(4), 2, 101)

    def test_routes_are_scoped_independently_and_epoch_is_monotonic(self) -> None:
        bid = LayeredSampler("spot_bid_future_taker")
        ask = LayeredSampler("spot_ask_future_taker")
        self.assertEqual(bid.reconcile(_cursor(1), 7, 100)[0].kind, "submit")
        self.assertEqual(ask.reconcile(_cursor(1), 7, 100)[0].kind, "submit")
        self.assertNotEqual(bid.stage, ask.stage)

        with self.assertRaisesRegex(ValueError, "monotonic"):
            bid.reconcile(_cursor(2), 6, 100)


if __name__ == "__main__":
    unittest.main()
