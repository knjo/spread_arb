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
            spot_book=spot,
            future_book=future_deeper,
        )
        self.assertEqual(deeper.target_location, "undisplayed_deeper")
        self.assertIsNone(deeper.initial_queue_ahead_shares)
        self.assertFalse(deeper.queue_observable)
        self.assertTrue(deeper.gate_open)

    def test_trial_match_and_future_depth_fail_closed(self) -> None:
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
            spot_book=trial,
            future_book=future,
        )
        self.assertFalse(target.gate_open)
        self.assertEqual(target.gate_reason, "trial_match")
        self.assertIsNone(target.target_price)

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
            spot_book=spot,
            future_book=shallow_future,
            future_contracts=2,
        )
        self.assertFalse(shallow.gate_open)
        self.assertEqual(shallow.gate_reason, "insufficient_depth")


if __name__ == "__main__":
    unittest.main()
