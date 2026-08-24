from __future__ import annotations

import unittest

from maker.src.quote_fill.hedge import (
    BookLevel,
    MakerFillHedgeRequest,
    OppositeBookSnapshot,
    SpotMakerFillEvent,
    label_delayed_taker_hedge,
    label_spot_partial_fill_horizons,
)
from maker.src.quote_fill.layered import EventCursor


MS = 1_000_000


def _cursor(ms: int, sequence: int = 0) -> EventCursor:
    return EventCursor(ms * MS, sequence, 0)


def _book(
    ms: int,
    *,
    sequence: int = 0,
    bids: tuple[tuple[float, int], ...] = ((99.0, 10),),
    asks: tuple[tuple[float, int], ...] = ((101.0, 10),),
    trial_match: bool = False,
    gate_open: bool = True,
    gate_reason: str | None = None,
) -> OppositeBookSnapshot:
    return OppositeBookSnapshot(
        _cursor(ms, sequence),
        tuple(BookLevel(*level) for level in bids),
        tuple(BookLevel(*level) for level in asks),
        trial_match=trial_match,
        gate_open=gate_open,
        gate_reason=gate_reason,
    )


class DelayedHedgeTest(unittest.TestCase):
    def test_equal_recv_time_keeps_cursor_and_time_asof_boundaries(self) -> None:
        result = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g-equal", _cursor(10, 1), 1, "buy", 1),
            [
                _book(10, sequence=0, asks=((100.0, 10),)),
                _book(10, sequence=2, asks=((101.0, 10),)),
                _book(60, sequence=0, asks=((102.0, 10),)),
                _book(60, sequence=2, asks=((103.0, 10),)),
            ],
        )
        self.assertEqual(result.arrival_snapshot_cursor, _cursor(10, 0))
        self.assertEqual(result.decision_snapshot_cursor, _cursor(60, 2))
        self.assertEqual(result.arrival_reference_price, 100.0)
        self.assertEqual(result.executable_vwap_price, 103.0)

    def test_buy_uses_last_causal_book_and_splits_latency_depth(self) -> None:
        request = MakerFillHedgeRequest("g1", _cursor(10), 1, "buy", 6)
        result = label_delayed_taker_hedge(
            request,
            [
                _book(5, asks=((100.0, 10),)),
                _book(40, asks=((101.0, 2), (102.0, 10))),
                _book(60, asks=((103.0, 10),)),
                _book(61, asks=((90.0, 10),)),  # first post-deadline update
            ],
        )
        self.assertTrue(result.hedge_complete)
        self.assertEqual(result.decision_snapshot_cursor, _cursor(60))
        self.assertEqual(result.executable_vwap_price, 103.0)
        self.assertEqual(result.levels_swept, 1)
        self.assertAlmostEqual(result.signed_latency_slippage_bp, 300.0)
        self.assertAlmostEqual(result.signed_depth_slippage_bp, 0.0)
        self.assertAlmostEqual(result.signed_total_slippage_bp, 300.0)

    def test_buy_depth_vwap_and_shortfall_are_not_filled_at_l5(self) -> None:
        snapshots = [
            _book(0, asks=((100.0, 2), (101.0, 3))),
            _book(50, asks=((101.0, 2), (102.0, 3))),
        ]
        complete = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g1", _cursor(0), 1, "buy", 4), snapshots
        )
        self.assertEqual(complete.status, "executable")
        self.assertEqual(complete.executable_vwap_price, 101.5)
        self.assertAlmostEqual(
            complete.signed_depth_slippage_bp, 0.5 / 101.0 * 10_000.0
        )
        self.assertAlmostEqual(complete.signed_total_slippage_bp, 150.0)

        short = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g2", _cursor(0), 1, "buy", 7), snapshots
        )
        self.assertEqual(short.status, "insufficient_depth")
        self.assertEqual(short.available_quantity, 5)
        self.assertEqual(short.executed_quantity, 5)
        self.assertEqual(short.depth_shortfall, 2)
        self.assertEqual(short.partial_vwap_price, 101.6)
        self.assertIsNone(short.executable_vwap_price)
        self.assertIsNone(short.signed_total_slippage_bp)

    def test_sell_slippage_is_adverse_when_bid_falls(self) -> None:
        result = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g1", _cursor(0), 2, "sell", 4),
            [
                _book(0, bids=((100.0, 10),)),
                _book(50, bids=((99.0, 2), (98.0, 3))),
            ],
        )
        self.assertEqual(result.executable_vwap_price, 98.5)
        self.assertAlmostEqual(result.signed_latency_slippage_bp, 100.0)
        self.assertAlmostEqual(
            result.signed_depth_slippage_bp, 0.5 / 99.0 * 10_000.0
        )
        self.assertAlmostEqual(result.signed_total_slippage_bp, 150.0)

    def test_trial_match_and_closed_gate_are_explicit(self) -> None:
        arrival_invalid = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g1", _cursor(0), 1, "buy", 1),
            [_book(0, trial_match=True)],
        )
        self.assertEqual(arrival_invalid.status, "arrival_trial_match")
        self.assertEqual(arrival_invalid.arrival_book_status, "trial_match")

        decision_invalid = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g2", _cursor(0), 1, "sell", 1),
            [
                _book(0),
                _book(50, gate_open=False, gate_reason="spot_trial_gate"),
            ],
        )
        self.assertEqual(decision_invalid.status, "decision_gate_closed")
        self.assertEqual(decision_invalid.gate_reason, "spot_trial_gate")

    def test_no_lookahead_and_stale_book_are_explicit(self) -> None:
        no_arrival = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g1", _cursor(0), 1, "buy", 1),
            [_book(1)],
        )
        self.assertEqual(no_arrival.status, "no_arrival_book")

        stale = label_delayed_taker_hedge(
            MakerFillHedgeRequest("g2", _cursor(100), 1, "buy", 1),
            [_book(0)],
            max_book_age_ns=50 * MS,
        )
        self.assertEqual(stale.status, "stale_arrival_book")

    def test_snapshot_order_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly"):
            label_delayed_taker_hedge(
                MakerFillHedgeRequest("g1", _cursor(0), 1, "buy", 1),
                [_book(10), _book(9)],
            )


class SpotPartialFillTest(unittest.TestCase):
    def test_cumulative_quantity_and_completion_by_horizon(self) -> None:
        fills = [
            SpotMakerFillEvent(_cursor(10), 1000),
            SpotMakerFillEvent(_cursor(60), 500),
            SpotMakerFillEvent(_cursor(110), 500),
        ]
        labels = label_spot_partial_fill_horizons(
            "g1",
            _cursor(10),
            fills,
            futures_equivalent_quantity=2000,
            wait_horizons_ns=(0, 50 * MS, 100 * MS, 500 * MS),
        )
        self.assertEqual(
            [label.cumulative_fill_quantity for label in labels],
            [1000, 1500, 2000, 2000],
        )
        self.assertEqual([label.status for label in labels], [
            "partial",
            "partial",
            "complete",
            "complete",
        ])
        self.assertEqual(labels[2].completion_cursor, _cursor(110))
        self.assertEqual(labels[1].residual_quantity, 500)

    def test_invalid_fill_state_is_counted_and_flagged(self) -> None:
        labels = label_spot_partial_fill_horizons(
            "g1",
            _cursor(10),
            [
                SpotMakerFillEvent(_cursor(10), 1000),
                SpotMakerFillEvent(
                    _cursor(20),
                    1000,
                    trial_match=True,
                    gate_open=False,
                    gate_reason="cross_market_trial",
                ),
            ],
            futures_equivalent_quantity=2000,
            wait_horizons_ns=(10 * MS,),
        )
        label = labels[0]
        self.assertTrue(label.hedge_unit_complete)
        self.assertEqual(label.status, "trial_match_and_gate_closed_fill")
        self.assertEqual(label.trial_match_fill_quantity, 1000)
        self.assertEqual(label.gate_closed_fill_quantity, 1000)
        self.assertEqual(label.gate_reasons, ("cross_market_trial",))

    def test_horizons_and_fill_order_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            label_spot_partial_fill_horizons(
                "g1",
                _cursor(0),
                [],
                futures_equivalent_quantity=2000,
                wait_horizons_ns=(50 * MS, 50 * MS),
            )
        with self.assertRaisesRegex(ValueError, "strictly cursor-sorted"):
            label_spot_partial_fill_horizons(
                "g1",
                _cursor(0),
                [
                    SpotMakerFillEvent(_cursor(2), 1),
                    SpotMakerFillEvent(_cursor(1), 1),
                ],
                futures_equivalent_quantity=2,
                wait_horizons_ns=(10 * MS,),
            )


if __name__ == "__main__":
    unittest.main()
