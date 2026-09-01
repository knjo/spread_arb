"""Contracts for the causal S1 Spot Ask lower target."""

from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_exit_target import build_s1_spot_ask_target
from ..quote_fill.s1_hedge import (
    CausalBookState,
    RawBookCursor,
    RawBookLevel,
)
from ..quote_fill.targets import absolute_price_tick


def _book(
    cursor_ns: int,
    *,
    bids: tuple[tuple[float, int], ...],
    asks: tuple[tuple[float, int], ...],
    gate_open: bool = True,
    reason: str | None = None,
) -> CausalBookState:
    return CausalBookState(
        RawBookCursor(EventCursor(cursor_ns), 1),
        gate_open,
        reason,
        100.0 if gate_open else None,
        tuple(RawBookLevel(*level) for level in bids),
        tuple(RawBookLevel(*level) for level in asks),
    )


class S1SpotAskTargetTest(unittest.TestCase):
    def test_normal_lower_is_maker_ask_plus_future_buy_taker(self) -> None:
        cursor = EventCursor(200)
        spot = _book(
            190,
            bids=((99.9, 2_000),),
            asks=((100.0, 3_000), (100.5, 4_000)),
        )
        future = _book(
            191,
            bids=((100.0, 5),),
            asks=((100.0, 5),),
        )
        target = build_s1_spot_ask_target(
            date="20260505",
            value_code="2330",
            quote_code="CDFE6",
            position_id="position-1",
            scenario_id="q95/spot-ask",
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=0.0,
            frozen_exit_target_price=100.0,
            frozen_exit_absolute_price_tick=absolute_price_tick(100.0),
            spot_book=spot,
            future_book=future,
        )

        self.assertTrue(target.gate_open)
        self.assertEqual(target.target_price, 100.0)
        self.assertEqual(target.future_buy_vwap, 100.0)
        self.assertEqual(target.target_location, "at_ask1")
        self.assertEqual(target.initial_queue_ahead_shares, 3_000)
        self.assertTrue(target.queue_observable)
        self.assertLessEqual(target.effective_exit_basis_bp, 0.0)

    def test_inside_spread_has_zero_queue_and_deeper_unknown_is_explicit(self) -> None:
        cursor = EventCursor(200)
        spot = _book(
            190,
            bids=((99.0, 2_000),),
            asks=((101.0, 3_000),),
        )
        future_inside = _book(
            191,
            bids=((100.0, 5),),
            asks=((100.0, 5),),
        )
        inside = build_s1_spot_ask_target(
            date="20260505",
            value_code="2330",
            quote_code="CDFE6",
            position_id="position-1",
            scenario_id="q95/spot-ask",
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=0.0,
            frozen_exit_target_price=100.0,
            frozen_exit_absolute_price_tick=absolute_price_tick(100.0),
            spot_book=spot,
            future_book=future_inside,
        )
        self.assertEqual(inside.target_location, "inside_spread")
        self.assertEqual(inside.initial_queue_ahead_shares, 0)

        future_deeper = _book(
            191,
            bids=((101.5, 5),),
            asks=((102.0, 5),),
        )
        deeper = build_s1_spot_ask_target(
            date="20260505",
            value_code="2330",
            quote_code="CDFE6",
            position_id="position-1",
            scenario_id="q95/spot-ask",
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=0.0,
            frozen_exit_target_price=102.0,
            frozen_exit_absolute_price_tick=absolute_price_tick(102.0),
            spot_book=spot,
            future_book=future_deeper,
        )
        self.assertEqual(deeper.target_location, "undisplayed_deeper")
        self.assertIsNone(deeper.initial_queue_ahead_shares)
        self.assertFalse(deeper.queue_observable)
        self.assertTrue(deeper.gate_open)

    def test_spot_and_future_hedgeability_both_gate_passive_exit(self) -> None:
        cursor = EventCursor(200)
        trial = _book(
            190,
            bids=(),
            asks=(),
            gate_open=False,
            reason="trial_match",
        )
        future = _book(
            191,
            bids=((100.0, 5),),
            asks=((100.0, 5),),
        )
        target = build_s1_spot_ask_target(
            date="20260505",
            value_code="2330",
            quote_code="CDFE6",
            position_id="position-1",
            scenario_id="q95/spot-ask",
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=0.0,
            frozen_exit_target_price=100.0,
            frozen_exit_absolute_price_tick=absolute_price_tick(100.0),
            spot_book=trial,
            future_book=future,
        )
        self.assertFalse(target.gate_open)
        self.assertEqual(target.gate_reason, "trial_match")
        self.assertEqual(target.target_price, 100.0)
        self.assertEqual(target.absolute_price_tick, absolute_price_tick(100.0))

        spot = _book(
            190,
            bids=((99.9, 2_000),),
            asks=((100.0, 3_000),),
        )
        shallow_future = _book(
            191,
            bids=((100.0, 5),),
            asks=((100.0, 1),),
        )
        shallow = build_s1_spot_ask_target(
            date="20260505",
            value_code="2330",
            quote_code="CDFE6",
            position_id="position-1",
            scenario_id="q95/spot-ask",
            observation_cursor=cursor,
            frozen_exit_threshold_basis_bp=0.0,
            frozen_exit_target_price=100.0,
            frozen_exit_absolute_price_tick=absolute_price_tick(100.0),
            spot_book=spot,
            future_book=shallow_future,
            future_contracts=2,
        )
        self.assertFalse(shallow.gate_open)
        self.assertEqual(shallow.gate_reason, "future_insufficient_depth")
        self.assertIsNone(shallow.future_buy_vwap)
        self.assertIsNone(shallow.effective_exit_basis_bp)

    def test_future_ask_changes_and_missing_book_never_move_frozen_target(self) -> None:
        cursor = EventCursor(200)
        spot = _book(
            190,
            bids=((99.0, 2_000),),
            asks=((101.0, 3_000),),
        )

        def observe(future: CausalBookState | None):
            return build_s1_spot_ask_target(
                date="20260505",
                value_code="2330",
                quote_code="CDFE6",
                position_id="position-1",
                scenario_id="q95/spot-ask",
                observation_cursor=cursor,
                frozen_exit_threshold_basis_bp=500.0,
                frozen_exit_target_price=100.0,
                frozen_exit_absolute_price_tick=absolute_price_tick(100.0),
                spot_book=spot,
                future_book=future,
            )

        at_100 = observe(
            _book(
                191,
                bids=((99.5, 5),),
                asks=((100.0, 5),),
            )
        )
        at_102 = observe(
            _book(
                192,
                bids=((101.5, 5),),
                asks=((102.0, 5),),
            )
        )
        missing = observe(None)

        for target in (at_100, at_102, missing):
            self.assertEqual(target.target_price, 100.0)
            self.assertEqual(
                target.absolute_price_tick,
                absolute_price_tick(100.0),
            )
        self.assertTrue(at_100.gate_open)
        self.assertTrue(at_102.gate_open)
        self.assertFalse(missing.gate_open)
        self.assertEqual(missing.gate_reason, "missing_future_book")
        self.assertNotEqual(
            at_100.effective_exit_basis_bp,
            at_102.effective_exit_basis_bp,
        )
        self.assertIsNone(missing.effective_exit_basis_bp)
        self.assertIsNone(missing.future_book_cursor)

    def test_frozen_price_tick_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "price/tick mismatch"):
            build_s1_spot_ask_target(
                date="20260505",
                value_code="2330",
                quote_code="CDFE6",
                position_id="position-1",
                scenario_id="q95/spot-ask",
                observation_cursor=EventCursor(200),
                frozen_exit_threshold_basis_bp=0.0,
                frozen_exit_target_price=100.0,
                frozen_exit_absolute_price_tick=absolute_price_tick(100.5),
                spot_book=None,
                future_book=None,
            )


if __name__ == "__main__":
    unittest.main()
