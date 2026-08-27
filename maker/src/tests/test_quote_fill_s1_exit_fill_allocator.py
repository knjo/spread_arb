"""Physical printed-volume contracts for the S1 Spot Ask exit route."""

from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor
from ..quote_fill.replay import TradeEvent
from ..quote_fill.s1_exit_fill_allocator import (
    ExitFillAllocationError,
    S1SpotAskFillAllocator,
)
from ..quote_fill.s1_exit_inventory import (
    AggregateOrderMember,
    AggregateOrderTerminal,
    AggregateWorkingOrder,
)


def _order(
    raw_id: str = "raw-1",
    *,
    tick: int = 1_000,
    start_ns: int = 100,
    shares: int = 2_000,
) -> AggregateWorkingOrder:
    return AggregateWorkingOrder(
        candidate_intent_id=f"candidate-{raw_id}",
        raw_order_fact_id=raw_id,
        absolute_price_tick=tick,
        actual_start_cursor=EventCursor(start_ns, 700, 1),
        cancel_state="none",
        members=(AggregateOrderMember("position-1", 50, shares, 0),),
    )


def _terminal(
    order: AggregateWorkingOrder,
    *,
    cursor: EventCursor,
    filled_shares: int,
    reason: str,
) -> AggregateOrderTerminal:
    return AggregateOrderTerminal(
        order.raw_order_fact_id,
        order.absolute_price_tick,
        order.actual_start_cursor,
        cursor,
        reason,  # type: ignore[arg-type]
        (
            AggregateOrderMember(
                "position-1",
                50,
                order.total_shares,
                filled_shares,
            ),
        ),
        False,
    )


class S1SpotAskFillAllocatorTest(unittest.TestCase):
    def test_exposes_lowest_active_target_for_relevant_trade_wakes(self) -> None:
        allocator = S1SpotAskFillAllocator(
            session_date="20260826",
            product_id="2330",
        )
        self.assertIsNone(allocator.minimum_active_target_price_tick)
        allocator.register_order(
            _order("raw-high", tick=1_010),
            initial_queue_ahead_shares=0,
        )
        allocator.register_order(
            _order("raw-low", tick=1_000),
            initial_queue_ahead_shares=0,
        )
        self.assertEqual(allocator.minimum_active_target_price_tick, 1_000)

    def test_physical_fill_ids_are_scoped_by_date_product_and_raw_order(self) -> None:
        identities: list[str] = []
        for session_date, product_id, raw_id in (
            ("20260826", "2330", "raw-a"),
            ("20260826", "2317", "raw-b"),
            ("20260827", "2330", "raw-c"),
        ):
            allocator = S1SpotAskFillAllocator(
                session_date=session_date,
                product_id=product_id,
            )
            allocator.register_order(
                _order(raw_id),
                initial_queue_ahead_shares=0,
            )
            identities.append(
                allocator.on_trade(TradeEvent(EventCursor(110), 1_001, 2_000))[
                    0
                ].fill_id
            )

        self.assertEqual(len(identities), len(set(identities)))

    def test_same_price_volume_consumes_queue_once_then_own_leaves(self) -> None:
        order = _order()
        allocator = S1SpotAskFillAllocator(session_date="20260826", product_id="2330")
        allocator.register_order(order, initial_queue_ahead_shares=1_500)

        self.assertEqual(
            allocator.on_trade(TradeEvent(EventCursor(110), 1_000, 1_000)),
            (),
        )
        first = allocator.on_trade(TradeEvent(EventCursor(120), 1_000, 1_000))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].fill_shares, 500)
        self.assertEqual(first[0].queue_ahead_before_shares, 500)
        self.assertEqual(first[0].queue_ahead_after_shares, 0)
        self.assertTrue(first[0].joint_same_price_volume_allocated)
        second = allocator.on_trade(TradeEvent(EventCursor(130), 1_000, 2_000))
        self.assertEqual(second[0].fill_shares, 1_500)
        self.assertEqual(second[0].leaves_after_shares, 0)
        self.assertEqual(
            sum(fill.fill_shares for fill in allocator.fills),
            order.total_shares,
        )
        allocator.on_order_terminal(
            _terminal(
                order,
                cursor=EventCursor(130, 300, 1),
                filled_shares=2_000,
                reason="filled",
            )
        )

    def test_unknown_queue_only_fills_on_strict_trade_through(self) -> None:
        order = _order()
        allocator = S1SpotAskFillAllocator(session_date="20260826", product_id="2330")
        allocator.register_order(order, initial_queue_ahead_shares=None)
        self.assertEqual(
            allocator.on_trade(TradeEvent(EventCursor(110), 1_000, 10_000)),
            (),
        )
        fill = allocator.on_trade(TradeEvent(EventCursor(120), 1_001, 1))[0]
        self.assertEqual(fill.fill_shares, 2_000)
        self.assertEqual(fill.fill_reason, "trade_through")
        self.assertTrue(fill.trade_through_inference)
        self.assertFalse(fill.joint_same_price_volume_allocated)

    def test_same_cursor_fill_precedes_cancel_and_new_cannot_backfill(self) -> None:
        old = _order()
        allocator = S1SpotAskFillAllocator(session_date="20260826", product_id="2330")
        allocator.register_order(old, initial_queue_ahead_shares=0)
        cursor = EventCursor(200, 300, 1)
        fill = allocator.on_trade(TradeEvent(EventCursor(200, 20, 1), 1_000, 500))[0]
        self.assertEqual(fill.fill_shares, 500)
        allocator.on_order_terminal(
            _terminal(
                old,
                cursor=EventCursor(200, 500, 1),
                filled_shares=500,
                reason="actual_cancelled",
            )
        )

        new = _order("raw-2", start_ns=200, shares=1_500)
        allocator.register_order(new, initial_queue_ahead_shares=0)
        self.assertEqual(len(allocator.fills), 1)
        allocator.assert_working_order(new)
        self.assertLess(cursor, new.actual_start_cursor)

    def test_regression_duplicate_and_terminal_divergence_fail_closed(self) -> None:
        order = _order()
        allocator = S1SpotAskFillAllocator(session_date="20260826", product_id="2330")
        allocator.register_order(order, initial_queue_ahead_shares=0)
        with self.assertRaisesRegex(ExitFillAllocationError, "already"):
            allocator.register_order(order, initial_queue_ahead_shares=0)
        allocator.on_trade(TradeEvent(EventCursor(120), 999, 1))
        with self.assertRaisesRegex(ExitFillAllocationError, "strictly"):
            allocator.on_trade(TradeEvent(EventCursor(120), 999, 1))
        with self.assertRaisesRegex(ExitFillAllocationError, "diverged"):
            allocator.on_order_terminal(
                _terminal(
                    order,
                    cursor=EventCursor(130),
                    filled_shares=1,
                    reason="actual_cancelled",
                )
            )


if __name__ == "__main__":
    unittest.main()
