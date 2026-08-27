"""Focused contracts for the pure S1 entry intent/working controller."""

from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_entry_controller import (
    EntryControllerCommand,
    S1EntryController,
)
from ..quote_fill.venue_scheduler import (
    RollingVenueScheduler,
    VenueRequestIntent,
    VenueSendAssignment,
)


def _controller(policy: str = "q95") -> S1EntryController:
    return S1EntryController(
        Date="20260505",
        ValueCode="2330",
        QuoteCode="CDFE6",
        route="spot_bid_future_taker",
        stage="entry",
        maker_side="bid",
        policy_id=policy,
    )


def _candidate(command: EntryControllerCommand) -> str:
    candidate_id = command.candidate_intent_id
    assert candidate_id is not None
    return candidate_id


def _venue_request(command: EntryControllerCommand) -> VenueRequestIntent:
    if command.kind not in ("enqueue_new", "enqueue_cancel"):
        raise ValueError("only enqueue commands create venue requests")
    return VenueRequestIntent(
        request_id=command.request_id,
        venue="spot",
        request_class="new" if command.kind == "enqueue_new" else "cancel",
        original_cursor=command.cursor,
        stable_id=command.request_id,
        maker_side="bid",
        absolute_price_tick=command.absolute_price_tick,
    )


def _effect_cursor(
    assignment: VenueSendAssignment,
    event_sequence: int,
) -> EventCursor:
    """Map one frozen assignment to its unique same-timestamp effect cursor."""

    return EventCursor(
        assignment.actual_send_cursor.recv_time_ns,
        event_sequence,
        assignment.send_sequence,
    )


class S1EntryControllerTest(unittest.TestCase):
    def test_delayed_new_creates_no_raw_id_until_atomic_actual_send(self) -> None:
        controller = _controller()
        command = controller.observe(
            EventCursor(10, 1, 0),
            100,
            base_gate_open=True,
            admission_open=True,
        )[0]
        self.assertEqual(command.kind, "enqueue_new")
        self.assertIsNone(command.raw_order_fact_id)
        self.assertEqual(controller.working_orders, ())

        order = controller.on_new_sent(
            _candidate(command),
            EventCursor(20, 6, 7),
            100,
        )
        self.assertEqual(order.actual_start_cursor, EventCursor(20, 6, 7))
        self.assertEqual(order.intent_absolute_price_tick, 100)
        self.assertEqual(order.absolute_price_tick, 100)
        self.assertEqual(len(controller.working_orders), 1)

    def test_raw_identity_uses_actual_facts_and_policy_only_changes_alias(self) -> None:
        q95 = _controller("q95")
        fixed = _controller("fixed20")
        q95_command = q95.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        fixed_command = fixed.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        q95_order = q95.on_new_sent(_candidate(q95_command), EventCursor(20, 6, 1), 100)
        fixed_order = fixed.on_new_sent(
            _candidate(fixed_command), EventCursor(20, 6, 1), 100
        )

        self.assertEqual(q95_order.raw_order_fact_id, fixed_order.raw_order_fact_id)
        self.assertNotEqual(q95_order.policy_alias_id, fixed_order.policy_alias_id)

        changed_tick = _controller("fixed25")
        changed_command = changed_tick.observe(
            EventCursor(10), 101, base_gate_open=True, admission_open=True
        )[0]
        changed_order = changed_tick.on_new_sent(
            _candidate(changed_command), EventCursor(20, 6, 1), 101
        )
        self.assertNotEqual(
            q95_order.raw_order_fact_id, changed_order.raw_order_fact_id
        )

    def test_unsent_latest_desired_coalesces_without_raw_fact(self) -> None:
        controller = _controller()
        first = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        commands = controller.observe(
            EventCursor(11), 101, base_gate_open=True, admission_open=True
        )
        self.assertEqual(
            [command.kind for command in commands],
            ["withdraw_pending_new", "enqueue_new"],
        )
        self.assertNotEqual(
            first.candidate_intent_id,
            controller.pending_candidate_intent_id,
        )
        self.assertEqual(controller.intent_audit[0].status, "coalesced")
        self.assertIsNone(controller.intent_audit[0].raw_order_fact_id)

        cutoff = controller.cutoff(EventCursor(20), unsent_reason="cap_blocked")
        self.assertEqual(cutoff[0].kind, "withdraw_pending_new")
        self.assertEqual(controller.intent_audit[-1].status, "cap_blocked")
        self.assertEqual(controller.working_orders, ())

    def test_forward_adds_layer_but_retreat_only_cancels_above(self) -> None:
        controller = _controller()
        first = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        low = controller.on_new_sent(
            _candidate(first),
            EventCursor(10, 6, 1),
            100,
        )
        forward = controller.observe(
            EventCursor(11), 101, base_gate_open=True, admission_open=True
        )[0]
        high = controller.on_new_sent(
            _candidate(forward),
            EventCursor(11, 6, 2),
            101,
        )
        self.assertEqual(len(controller.working_orders), 2)

        retreat = controller.observe(
            EventCursor(12), 100, base_gate_open=True, admission_open=True
        )
        self.assertEqual([command.kind for command in retreat], ["enqueue_cancel"])
        self.assertEqual(retreat[0].raw_order_fact_id, high.raw_order_fact_id)
        self.assertEqual(
            {order.raw_order_fact_id for order in controller.working_orders},
            {low.raw_order_fact_id, high.raw_order_fact_id},
        )

    def test_gate_close_cancel_persists_without_duplicate(self) -> None:
        controller = _controller()
        new = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        order = controller.on_new_sent(
            _candidate(new),
            EventCursor(10, 6, 1),
            100,
        )
        cancel = controller.observe(
            EventCursor(11), None, base_gate_open=False, admission_open=False
        )
        self.assertEqual(cancel[0].kind, "enqueue_cancel")
        repeated = controller.observe(
            EventCursor(12), None, base_gate_open=False, admission_open=False
        )
        self.assertEqual(repeated, ())
        self.assertEqual(controller.working_orders[0].cancel_state, "pending")

        controller.on_cancel_assigned(order.raw_order_fact_id, EventCursor(13, 2, 1))
        terminal, replacements = controller.on_cancel_sent(
            order.raw_order_fact_id, EventCursor(13, 5, 1)
        )
        self.assertEqual(terminal.terminal_reason, "actual_cancelled")
        self.assertEqual(replacements, ())

    def test_fill_withdraws_pending_cancel_but_frozen_cancel_is_noop(self) -> None:
        pending_controller = _controller()
        new = pending_controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        order = pending_controller.on_new_sent(
            _candidate(new),
            EventCursor(10, 6, 1),
            100,
        )
        pending_controller.observe(
            EventCursor(11), None, base_gate_open=False, admission_open=False
        )
        terminal, commands = pending_controller.on_fill(
            order.raw_order_fact_id, EventCursor(12, 3, 0)
        )
        self.assertEqual(terminal.terminal_reason, "filled")
        self.assertEqual(commands[0].kind, "withdraw_pending_cancel")

        frozen_controller = _controller("fixed20")
        new = frozen_controller.observe(
            EventCursor(20), 100, base_gate_open=True, admission_open=True
        )[0]
        order = frozen_controller.on_new_sent(
            _candidate(new),
            EventCursor(20, 6, 1),
            100,
        )
        frozen_controller.observe(
            EventCursor(21), None, base_gate_open=False, admission_open=False
        )
        frozen_controller.on_cancel_assigned(
            order.raw_order_fact_id, EventCursor(22, 2, 1)
        )
        terminal, commands = frozen_controller.on_fill(
            order.raw_order_fact_id, EventCursor(22, 3, 1)
        )
        self.assertEqual(commands, ())
        self.assertTrue(terminal.frozen_cancel_will_be_noop)
        noop, replacements = frozen_controller.on_cancel_sent(
            order.raw_order_fact_id, EventCursor(22, 5, 1)
        )
        self.assertEqual(noop, terminal)
        self.assertEqual(replacements, ())

    def test_cutoff_and_expiry_do_not_pretend_cancel_is_terminal(self) -> None:
        controller = _controller()
        new = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        controller.on_new_sent(
            _candidate(new),
            EventCursor(10, 6, 1),
            100,
        )
        cutoff = controller.cutoff(EventCursor(20))
        self.assertEqual(cutoff[0].kind, "enqueue_cancel")
        self.assertEqual(len(controller.working_orders), 1)
        terminals, commands = controller.session_expiry(EventCursor(30))
        self.assertEqual(terminals[0].terminal_reason, "session_expired")
        self.assertEqual(commands[0].kind, "withdraw_pending_cancel")
        self.assertEqual(controller.working_orders, ())
        self.assertTrue(controller.session_expired)
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            controller.session_expiry(EventCursor(31))

    def test_atomic_new_validation_rejects_changed_tick_without_advancing(self) -> None:
        controller = _controller()
        command = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        candidate_id = _candidate(command)

        self.assertFalse(controller.can_send_pending_at_tick(candidate_id, 101))
        with self.assertRaisesRegex(ValueError, "actual_tick_differs"):
            controller.on_new_sent(candidate_id, EventCursor(20, 6, 1), 101)

        order = controller.on_new_sent(
            candidate_id,
            EventCursor(20, 6, 1),
            100,
        )
        self.assertEqual(order.absolute_price_tick, 100)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            controller.on_fill(order.raw_order_fact_id, EventCursor(20, 6, 1))

    def test_gate_and_pending_identity_are_part_of_atomic_send_validation(self) -> None:
        controller = _controller()
        command = controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=True
        )[0]
        stale_candidate = _candidate(command)
        controller.observe(
            EventCursor(11), 100, base_gate_open=True, admission_open=False
        )

        self.assertFalse(controller.can_send_pending_at_tick(stale_candidate, 100))
        with self.assertRaisesRegex(ValueError, "admission_closed"):
            controller.on_new_sent(
                stale_candidate,
                EventCursor(12, 6, 1),
                100,
            )

        replacement = controller.observe(
            EventCursor(12, 6, 1),
            100,
            base_gate_open=True,
            admission_open=True,
        )[0]
        self.assertTrue(
            controller.can_send_pending_at_tick(_candidate(replacement), 100)
        )

    def test_failed_validation_does_not_consume_cursor(self) -> None:
        controller = _controller()
        controller.observe(
            EventCursor(10), 100, base_gate_open=True, admission_open=False
        )
        with self.assertRaisesRegex(ValueError, "admission cannot"):
            controller.observe(
                EventCursor(11),
                None,
                base_gate_open=False,
                admission_open=True,
            )

        command = controller.observe(
            EventCursor(11), 100, base_gate_open=True, admission_open=True
        )[0]
        self.assertEqual(command.reason, "became_admission_eligible")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            controller.observe(
                EventCursor(11), 100, base_gate_open=True, admission_open=True
            )

    def test_controller_rejects_non_s1_route_semantics(self) -> None:
        common = {
            "Date": "20260505",
            "ValueCode": "2330",
            "QuoteCode": "CDFE6",
            "policy_id": "q95",
        }
        with self.assertRaisesRegex(ValueError, "spot_bid_future_taker"):
            S1EntryController(
                **common,
                route="future_ask_spot_taker",
                stage="entry",
                maker_side="bid",
            )
        with self.assertRaisesRegex(ValueError, "spot_bid_future_taker"):
            S1EntryController(
                **common,
                route="spot_bid_future_taker",
                stage="exit",
                maker_side="bid",
            )
        with self.assertRaisesRegex(ValueError, "spot_bid_future_taker"):
            S1EntryController(
                **common,
                route="spot_bid_future_taker",
                stage="entry",
                maker_side="ask",  # type: ignore[arg-type]
            )


class S1EntryControllerSchedulerIntegrationTest(unittest.TestCase):
    def test_ineligible_new_uses_no_token_and_coalesced_request_is_removed(
        self,
    ) -> None:
        controller = _controller()
        scheduler = RollingVenueScheduler("spot", 1)
        old = controller.observe(
            EventCursor(10, 1, 0),
            100,
            base_gate_open=True,
            admission_open=True,
        )[0]
        scheduler.enqueue(_venue_request(old))

        blocked = scheduler.dispatch(
            EventCursor(20, 2, 0),
            send_eligible=lambda request: controller.can_send_pending_at_tick(
                _candidate(old),
                101,
            ),
        )
        self.assertEqual(blocked, ())
        self.assertEqual(scheduler.total_sent, 0)
        self.assertEqual(scheduler.pending_count, 1)

        coalesced, latest = controller.observe(
            EventCursor(20, 3, 0),
            101,
            base_gate_open=True,
            admission_open=True,
        )
        self.assertEqual(coalesced.kind, "withdraw_pending_new")
        self.assertEqual(
            scheduler.cancel_pending(coalesced.request_id),
            _venue_request(old),
        )
        scheduler.enqueue(_venue_request(latest))

        assignments = scheduler.dispatch(
            EventCursor(20, 4, 0),
            send_eligible=lambda request: controller.can_send_pending_at_tick(
                _candidate(latest),
                101,
            ),
        )
        self.assertEqual(len(assignments), 1)
        effect_cursor = _effect_cursor(assignments[0], 6)
        order = controller.on_new_sent(
            _candidate(latest),
            effect_cursor,
            101,
        )
        self.assertEqual(effect_cursor.row_index, assignments[0].send_sequence)
        self.assertEqual(order.actual_start_cursor, effect_cursor)
        self.assertEqual(scheduler.total_sent, 1)

    def test_delayed_cancel_replaces_still_desired_tick_after_actual_effect(
        self,
    ) -> None:
        controller = _controller()
        scheduler = RollingVenueScheduler("spot", 100)
        initial = controller.observe(
            EventCursor(10, 1, 0),
            100,
            base_gate_open=True,
            admission_open=True,
        )[0]
        scheduler.enqueue(_venue_request(initial))
        new_assignment = scheduler.dispatch(EventCursor(10, 2, 0))[0]
        old_order = controller.on_new_sent(
            _candidate(initial),
            _effect_cursor(new_assignment, 6),
            100,
        )

        cancel = controller.observe(
            EventCursor(11, 1, 0),
            None,
            base_gate_open=False,
            admission_open=False,
        )[0]
        scheduler.enqueue(_venue_request(cancel))
        reopened = controller.observe(
            EventCursor(12, 1, 0),
            100,
            base_gate_open=True,
            admission_open=True,
        )
        self.assertEqual(reopened, ())

        cancel_assignment = scheduler.dispatch(EventCursor(13, 1, 0))[0]
        controller.on_cancel_assigned(
            old_order.raw_order_fact_id,
            _effect_cursor(cancel_assignment, 2),
        )
        terminal, replacement = controller.on_cancel_sent(
            old_order.raw_order_fact_id,
            _effect_cursor(cancel_assignment, 5),
        )
        self.assertEqual(terminal.terminal_reason, "actual_cancelled")
        self.assertEqual(len(replacement), 1)
        self.assertEqual(replacement[0].reason, "desired_after_actual_cancel")

        scheduler.enqueue(_venue_request(replacement[0]))
        replacement_assignment = scheduler.dispatch(EventCursor(14, 1, 0))[0]
        new_order = controller.on_new_sent(
            _candidate(replacement[0]),
            _effect_cursor(replacement_assignment, 6),
            100,
        )
        self.assertNotEqual(old_order.raw_order_fact_id, new_order.raw_order_fact_id)
        self.assertEqual(len(controller.working_orders), 1)

    def test_assignment_fill_precedes_cancel_effect_and_noop_is_consumed_once(
        self,
    ) -> None:
        controller = _controller()
        scheduler = RollingVenueScheduler("spot", 100)
        initial = controller.observe(
            EventCursor(10, 1, 0),
            100,
            base_gate_open=True,
            admission_open=True,
        )[0]
        scheduler.enqueue(_venue_request(initial))
        new_assignment = scheduler.dispatch(EventCursor(10, 2, 0))[0]
        order = controller.on_new_sent(
            _candidate(initial),
            _effect_cursor(new_assignment, 6),
            100,
        )

        cancel = controller.observe(
            EventCursor(11, 1, 0),
            None,
            base_gate_open=False,
            admission_open=False,
        )[0]
        scheduler.enqueue(_venue_request(cancel))
        cancel_assignment = scheduler.dispatch(EventCursor(12, 1, 0))[0]
        controller.on_cancel_assigned(
            order.raw_order_fact_id,
            _effect_cursor(cancel_assignment, 2),
        )
        terminal, fill_commands = controller.on_fill(
            order.raw_order_fact_id,
            _effect_cursor(cancel_assignment, 3),
        )
        self.assertEqual(fill_commands, ())
        self.assertTrue(terminal.frozen_cancel_will_be_noop)

        noop, replacements = controller.on_cancel_sent(
            order.raw_order_fact_id,
            _effect_cursor(cancel_assignment, 5),
        )
        self.assertEqual(noop, terminal)
        self.assertEqual(replacements, ())
        self.assertEqual(scheduler.total_sent, 2)

        failed_cursor = EventCursor(13, 5, cancel_assignment.send_sequence)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            controller.on_cancel_sent(order.raw_order_fact_id, failed_cursor)
        terminals, expiry_commands = controller.session_expiry(failed_cursor)
        self.assertEqual((terminals, expiry_commands), ((), ()))


if __name__ == "__main__":
    unittest.main()
