from __future__ import annotations

import unittest

from maker.src.quote_fill.engine import (
    TargetObservation,
    build_layered_order_windows,
)
from maker.src.quote_fill.layered import EventCursor


def _cursor(value: int) -> EventCursor:
    return EventCursor(value)


def _obs(
    value: int,
    epoch: int,
    target: int | None,
    *,
    gate: bool = True,
    queue: int | None = 10,
) -> TargetObservation:
    return TargetObservation(
        _cursor(value),
        epoch,
        target,
        gate,
        "open" if gate else "trial_match",
        queue,
        "B1",
    )


class LayeredWindowEngineTest(unittest.TestCase):
    def test_forward_layers_retreat_and_cutoff_form_windows(self) -> None:
        result = build_layered_order_windows(
            [
                _obs(10, 1, 100, queue=11),
                _obs(20, 1, 101, queue=12),
                _obs(30, 1, 100, queue=13),
            ],
            route="spot_bid_future_taker",
            policy_id="D/S/p50",
            cutoff_cursor=_cursor(40),
        )
        self.assertEqual(len(result.windows), 2)
        by_price = {item.target_price_tick: item for item in result.windows}
        self.assertEqual(by_price[101].stop_cursor, _cursor(30))
        self.assertEqual(by_price[101].stop_reason, "target_retreat")
        self.assertEqual(by_price[101].initial_queue_ahead, 12)
        self.assertEqual(by_price[100].stop_cursor, _cursor(40))
        self.assertEqual(by_price[100].stop_reason, "session_cutoff")
        self.assertEqual(result.peak_active_layers, 2)

    def test_cross_epoch_same_price_creates_two_generations(self) -> None:
        result = build_layered_order_windows(
            [_obs(10, 1, 100), _obs(20, 2, 100)],
            route="spot_bid_future_taker",
            policy_id="D/S/p50",
            cutoff_cursor=_cursor(30),
        )
        self.assertEqual(len(result.windows), 2)
        self.assertEqual(
            [item.target_price_tick for item in result.windows], [100, 100]
        )
        self.assertEqual(result.peak_active_layers, 2)

    def test_action_level_active_counts_follow_cancel_then_submit(self) -> None:
        result = build_layered_order_windows(
            [_obs(10, 1, 101), _obs(20, 2, 100)],
            route="spot_bid_future_taker",
            policy_id="D/S/p50",
            cutoff_cursor=_cursor(30),
        )
        at_switch = [
            item for item in result.transitions if item.cursor == _cursor(20)
        ]
        self.assertEqual(
            [(item.kind, item.active_layers_after) for item in at_switch],
            [("cancel", 0), ("submit", 1)],
        )
        cutoff = [
            item for item in result.transitions if item.reason == "session_cutoff"
        ]
        self.assertEqual([item.active_layers_after for item in cutoff], [0])

    def test_closed_epoch_has_no_delayed_base_but_cancels_live_layers(self) -> None:
        result = build_layered_order_windows(
            [
                _obs(10, 1, 100),
                _obs(20, 2, None, gate=False),
                _obs(30, 2, 100),
            ],
            route="spot_bid_future_taker",
            policy_id="D/S/p50",
            cutoff_cursor=_cursor(40),
        )
        self.assertEqual(len(result.windows), 1)
        self.assertEqual(result.windows[0].stop_cursor, _cursor(20))
        self.assertEqual(result.windows[0].stop_reason, "trial_match")

    def test_observations_and_cutoff_must_be_causal(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly"):
            build_layered_order_windows(
                [_obs(20, 1, 100), _obs(10, 1, 101)],
                route="spot_bid_future_taker",
                policy_id="p",
                cutoff_cursor=_cursor(30),
            )
        with self.assertRaisesRegex(ValueError, "follow"):
            build_layered_order_windows(
                [_obs(20, 1, 100)],
                route="spot_bid_future_taker",
                policy_id="p",
                cutoff_cursor=_cursor(20),
            )


if __name__ == "__main__":
    unittest.main()
