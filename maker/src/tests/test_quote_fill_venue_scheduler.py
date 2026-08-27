"""Focused contracts for the pure rolling venue scheduler."""

from __future__ import annotations

import unittest

from ..quote_fill.venue_scheduler import (
    ONE_SECOND_NS,
    EventCursor,
    RollingVenueScheduler,
    VenueRequestIntent,
)


def _request(
    request_id: str,
    request_class: str,
    time_ns: int,
    *,
    event_sequence: int = 0,
    row_index: int = 0,
    stable_id: str | None = None,
    risk_subtype: str | None = None,
    cutoff_drain: bool = False,
    maker_side: str | None = None,
    absolute_price_tick: int | None = None,
) -> VenueRequestIntent:
    return VenueRequestIntent(
        request_id=request_id,
        venue="spot",
        request_class=request_class,
        original_cursor=EventCursor(time_ns, event_sequence, row_index),
        stable_id=request_id if stable_id is None else stable_id,
        risk_subtype=(
            "hedge"
            if request_class == "exposed_risk" and risk_subtype is None
            else risk_subtype
        ),
        cutoff_drain=cutoff_drain,
        maker_side=maker_side,
        absolute_price_tick=absolute_price_tick,
    )


class RollingVenueSchedulerTest(unittest.TestCase):
    def test_rolling_window_has_an_open_lower_bound(self) -> None:
        scheduler = RollingVenueScheduler("spot", 2)
        for index in range(3):
            scheduler.enqueue(_request(f"new-{index}", "new", 10))

        first = scheduler.dispatch(EventCursor(10, 5, 0))
        self.assertEqual([row.request_id for row in first], ["new-0", "new-1"])
        self.assertEqual(scheduler.next_token_time_ns(), 10 + ONE_SECOND_NS)
        self.assertEqual(
            scheduler.dispatch(EventCursor(10 + ONE_SECOND_NS - 1, 5, 0)),
            (),
        )

        boundary = scheduler.dispatch(EventCursor(10 + ONE_SECOND_NS, 5, 0))
        self.assertEqual([row.request_id for row in boundary], ["new-2"])
        self.assertEqual(boundary[0].queue_delay_ns, ONE_SECOND_NS)

    def test_class_priority_precedes_original_cursor(self) -> None:
        scheduler = RollingVenueScheduler("spot", 3)
        scheduler.enqueue(_request("old-new", "new", 1))
        scheduler.enqueue(_request("middle-cancel", "cancel", 2))
        scheduler.enqueue(_request("late-risk", "exposed_risk", 3))

        assignments = scheduler.dispatch(EventCursor(4, 5, 0))

        self.assertEqual(
            [row.request_id for row in assignments],
            ["late-risk", "middle-cancel", "old-new"],
        )

    def test_exposed_risk_subtype_priority_precedes_original_cursor(self) -> None:
        scheduler = RollingVenueScheduler("spot", 3)
        scheduler.enqueue(
            _request(
                "old-aggressive",
                "exposed_risk",
                1,
                risk_subtype="aggressive_first_leg",
            )
        )
        scheduler.enqueue(
            _request("middle-hedge", "exposed_risk", 2, risk_subtype="hedge")
        )
        scheduler.enqueue(
            _request(
                "late-rollback",
                "exposed_risk",
                3,
                risk_subtype="emergency_rollback",
            )
        )

        assignments = scheduler.dispatch(EventCursor(4, 5, 0))

        self.assertEqual(
            [row.request_id for row in assignments],
            ["late-rollback", "middle-hedge", "old-aggressive"],
        )

    def test_risk_subtype_is_required_only_for_exposed_risk(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            VenueRequestIntent(
                request_id="risk",
                venue="spot",
                request_class="exposed_risk",
                original_cursor=EventCursor(1),
                stable_id="risk",
            )
        with self.assertRaisesRegex(ValueError, "only exposed_risk"):
            _request("new", "new", 1, risk_subtype="hedge")

    def test_external_send_gate_skips_without_head_of_line_blocking(self) -> None:
        scheduler = RollingVenueScheduler("spot", 2)
        scheduler.enqueue(_request("blocked", "new", 1))
        scheduler.enqueue(_request("eligible", "new", 2))

        first = scheduler.dispatch(
            EventCursor(3),
            send_eligible=lambda request: request.request_id != "blocked",
        )
        self.assertEqual([row.request_id for row in first], ["eligible"])
        self.assertEqual(
            [request.request_id for request in scheduler.pending_requests],
            ["blocked"],
        )

        second = scheduler.dispatch(EventCursor(4), send_eligible=lambda request: True)
        self.assertEqual([row.request_id for row in second], ["blocked"])

    def test_future_higher_priority_risk_does_not_block_ready_hedge(self) -> None:
        scheduler = RollingVenueScheduler("future", 1)
        scheduler.enqueue(
            VenueRequestIntent(
                request_id="future-rollback",
                venue="future",
                request_class="exposed_risk",
                risk_subtype="emergency_rollback",
                original_cursor=EventCursor(20),
                stable_id="future-rollback",
            )
        )
        scheduler.enqueue(
            VenueRequestIntent(
                request_id="ready-hedge",
                venue="future",
                request_class="exposed_risk",
                risk_subtype="hedge",
                original_cursor=EventCursor(10),
                stable_id="ready-hedge",
            )
        )

        assignments = scheduler.dispatch(EventCursor(10))
        self.assertEqual([row.request_id for row in assignments], ["ready-hedge"])

    def test_external_send_gate_must_return_boolean(self) -> None:
        scheduler = RollingVenueScheduler("spot", 1)
        scheduler.enqueue(_request("new", "new", 1))
        with self.assertRaisesRegex(TypeError, "return boolean"):
            scheduler.dispatch(EventCursor(1), send_eligible=lambda request: 1)
        self.assertEqual(scheduler.pending_count, 1)

    def test_same_class_uses_cursor_cutoff_price_then_stable_id(self) -> None:
        scheduler = RollingVenueScheduler("spot", 5)
        scheduler.enqueue(
            _request(
                "later-high",
                "cancel",
                2,
                row_index=1,
                cutoff_drain=True,
                maker_side="bid",
                absolute_price_tick=110,
            )
        )
        for request_id, tick, stable_id, generation in (
            ("low", 100, "z", 1),
            ("high-z", 110, "z", 3),
            ("high-a", 110, "a", 2),
        ):
            scheduler.enqueue(
                _request(
                    request_id,
                    "cancel",
                    1,
                    row_index=generation,
                    stable_id=stable_id,
                    cutoff_drain=True,
                    maker_side="bid",
                    absolute_price_tick=tick,
                )
            )

        assignments = scheduler.dispatch(EventCursor(3, 5, 0))

        self.assertEqual(
            [row.request_id for row in assignments],
            ["high-a", "high-z", "low", "later-high"],
        )

    def test_pending_request_can_be_cancelled_but_id_cannot_be_reused(self) -> None:
        scheduler = RollingVenueScheduler("spot", 1)
        request = _request("stale-new", "new", 10)
        scheduler.enqueue(request)

        self.assertEqual(scheduler.cancel_pending("stale-new"), request)
        self.assertIsNone(scheduler.cancel_pending("stale-new"))
        self.assertEqual(scheduler.pending_count, 0)
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            scheduler.enqueue(request)

    def test_future_intent_and_causal_phase_are_not_sent_early(self) -> None:
        scheduler = RollingVenueScheduler("spot", 1)
        scheduler.enqueue(_request("future", "new", 20, event_sequence=4))

        self.assertEqual(scheduler.next_token_time_ns(), 20)
        self.assertEqual(scheduler.dispatch(EventCursor(20, 3, 0)), ())
        assignment = scheduler.dispatch(EventCursor(20, 5, 0))
        self.assertEqual([row.request_id for row in assignment], ["future"])
        self.assertFalse(assignment[0].was_delayed)

    def test_s0_cutoff_drain_splits_190_bid_cancels_100_and_90(self) -> None:
        scheduler = RollingVenueScheduler("spot", 100)
        drain_time = 14_398 * ONE_SECOND_NS
        for index, tick in enumerate(range(1_000, 1_190)):
            scheduler.enqueue(
                _request(
                    f"cancel-{index:03d}",
                    "cancel",
                    drain_time,
                    cutoff_drain=True,
                    maker_side="bid",
                    absolute_price_tick=tick,
                )
            )

        first = scheduler.dispatch(EventCursor(drain_time, 4, 0))
        second_time = scheduler.next_token_time_ns()
        assert second_time is not None
        second = scheduler.dispatch(EventCursor(second_time, 4, 0))

        self.assertEqual((len(first), len(second)), (100, 90))
        self.assertEqual(first[0].request.absolute_price_tick, 1_189)
        self.assertEqual(first[-1].request.absolute_price_tick, 1_090)
        self.assertEqual(second[0].request.absolute_price_tick, 1_089)
        self.assertEqual(second[-1].request.absolute_price_tick, 1_000)
        self.assertEqual(second_time, drain_time + ONE_SECOND_NS)
        self.assertEqual(scheduler.total_sent, 190)
        self.assertEqual(scheduler.pending_count, 0)
        self.assertEqual(
            [row.send_sequence for row in (*first, *second)],
            list(range(1, 191)),
        )

    def test_validation_rejects_wrong_venue_and_time_regression(self) -> None:
        scheduler = RollingVenueScheduler("spot", 100)
        wrong_venue = VenueRequestIntent(
            request_id="future-new",
            venue="future",
            request_class="new",
            original_cursor=EventCursor(1),
            stable_id="future-new",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            scheduler.enqueue(wrong_venue)

        scheduler.dispatch(EventCursor(10))
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            scheduler.dispatch(EventCursor(9))


if __name__ == "__main__":
    unittest.main()
