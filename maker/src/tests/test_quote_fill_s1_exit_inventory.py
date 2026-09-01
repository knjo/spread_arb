from __future__ import annotations

import unittest
from dataclasses import replace
from time import perf_counter

from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.s1_accounting import PositionEstablishedFact
from maker.src.quote_fill.s1_exit_inventory import (
    ExitDesiredUpdate,
    ExitInventoryError,
    ExitInventoryReplayError,
    S1ExitInventoryController,
    replay_exit_inventory_facts,
)


def cursor(value: int, row: int = 0) -> EventCursor:
    return EventCursor(value, 0, row)


def position_fact(
    position_id: str,
    *,
    established_ns: int = 1,
    established_row: int = 0,
    shares: int = 1000,
    contracts: int = 1,
    sequence: int = 1,
) -> PositionEstablishedFact:
    return PositionEstablishedFact(
        sequence=sequence,
        establishment_id=f"established-{position_id}",
        position_id=position_id,
        value_code="2330",
        scenario_id="scenario-a",
        capacity_id=f"capacity-{position_id}",
        establishment_date="20260825",
        cursor=cursor(established_ns, established_row),
        position_established_ns=established_ns,
        capacity_transition_id=f"paired-{position_id}",
        spot_shares=shares,
        short_future_contracts=contracts,
        short_future_share_equivalent=shares,
        execution_truth="exact",
    )


def controller(*, session_date: str = "20260826") -> S1ExitInventoryController:
    return S1ExitInventoryController(
        Date=session_date,
        ValueCode="2330",
        QuoteCode="2330F",
        scenario_id="scenario-a",
    )


def assign_and_send(
    state: S1ExitInventoryController,
    *,
    assigned_cursor: EventCursor,
    sent_cursor: EventCursor,
) -> str:
    candidate = state.pending_orders[0].candidate_intent_id
    state.on_new_assigned(candidate, assigned_cursor)
    sent = state.on_new_sent(
        candidate,
        sent_cursor,
        state.pending_orders[0].absolute_price_tick,
    )
    assert sent.raw_order_fact_id is not None
    return sent.raw_order_fact_id


class S1ExitInventoryControllerTests(unittest.TestCase):
    def test_followup_identities_are_scoped_across_session_rebuilds(self) -> None:
        identities: list[tuple[str, str, str, str]] = []
        for session_date in ("20260826", "20260827"):
            state = controller(session_date=session_date)
            state.add_paired_position(
                position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
            )
            raw_id = assign_and_send(
                state,
                assigned_cursor=cursor(3),
                sent_cursor=cursor(4),
            )
            filled = state.on_fill(raw_id, cursor(5), 1000)
            hedge = filled.hedge_unit_requests[0]
            timed_out = state.on_hedge_timeout(hedge.request_id, cursor(6))
            identities.append(
                (
                    raw_id,
                    filled.fill_allocations[0].allocation_id,
                    hedge.request_id,
                    timed_out.rollback_requests[0].request_id,
                )
            )

        for index in range(4):
            self.assertNotEqual(identities[0][index], identities[1][index])

    def test_atomic_desired_batch_validates_all_rows_before_one_reconcile(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p1", established_ns=1, sequence=1),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        state.add_paired_position(
            position_fact("p2", established_ns=3, sequence=2),
            absolute_target_tick=101,
            cursor=cursor(4),
        )
        before_positions = state.positions
        before_pending = state.pending_orders
        before_fact_count = len(state.facts)
        with self.assertRaisesRegex(ExitInventoryError, "integral hedge units"):
            state.set_positions_desired(
                (
                    ExitDesiredUpdate("p1", 102, 1000),
                    ExitDesiredUpdate("p2", 102, 500),
                ),
                cursor=cursor(5),
            )
        self.assertEqual(state.positions, before_positions)
        self.assertEqual(state.pending_orders, before_pending)
        self.assertEqual(len(state.facts), before_fact_count)

        changed = state.set_positions_desired(
            (
                ExitDesiredUpdate("p2", 102, 1000),
                ExitDesiredUpdate("p1", 102, 1000),
            ),
            cursor=cursor(5),
        )
        self.assertEqual(changed.operation, "set_desired_batch")
        self.assertEqual(
            [update.position_id for update in changed.desired_updates],
            ["p1", "p2"],
        )
        self.assertEqual(
            [command.kind for command in changed.commands],
            ["withdraw_pending_new", "withdraw_pending_new", "enqueue_new"],
        )
        self.assertEqual(len(state.pending_orders), 1)
        self.assertEqual(state.pending_orders[0].total_shares, 2000)
        self.assertEqual(
            [member.position_id for member in state.pending_orders[0].members],
            ["p1", "p2"],
        )
        state.verify()

    def test_only_valid_causal_exact_pair_fact_is_admitted(self) -> None:
        state = controller()
        source = position_fact("p")
        with self.assertRaisesRegex(TypeError, "PositionEstablishedFact"):
            state.add_paired_position(  # type: ignore[arg-type]
                object(), absolute_target_tick=100, cursor=cursor(2)
            )
        with self.assertRaisesRegex(ExitInventoryError, "paired"):
            state.add_paired_position(
                replace(source, short_future_share_equivalent=500),
                absolute_target_tick=100,
                cursor=cursor(2),
            )
        with self.assertRaisesRegex(ExitInventoryError, "scenario_id"):
            state.add_paired_position(
                replace(source, scenario_id="other"),
                absolute_target_tick=100,
                cursor=cursor(2),
            )
        with self.assertRaisesRegex(ExitInventoryError, "before allocation"):
            state.add_paired_position(
                source, absolute_target_tick=100, cursor=cursor(1)
            )
        admitted = state.add_paired_position(
            source, absolute_target_tick=100, cursor=cursor(2)
        )
        self.assertEqual(len(admitted.commands), 1)
        self.assertEqual(admitted.commands[0].kind, "enqueue_new")
        with self.assertRaisesRegex(ExitInventoryError, "already added"):
            state.add_paired_position(
                source, absolute_target_tick=100, cursor=cursor(3)
            )

    def test_raw_order_exists_only_after_actual_new_send(self) -> None:
        state = controller()
        added = state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        candidate = added.commands[0].candidate_intent_id
        self.assertIsNotNone(candidate)
        self.assertEqual(len(state.working_orders), 0)
        self.assertEqual(state.positions[0].allocation_kind, "candidate")
        assert candidate is not None
        state.on_new_assigned(candidate, cursor(3))
        sent = state.on_new_sent(candidate, cursor(4), 100)
        self.assertIsNotNone(sent.raw_order_fact_id)
        working = state.working_orders[0]
        self.assertEqual(working.actual_start_cursor, cursor(4))
        self.assertEqual(working.members[0].assigned_shares, 1000)
        self.assertEqual(state.positions[0].allocation_kind, "working")
        state.verify()

    def test_candidate_identity_ignores_global_sibling_effect_rows(self) -> None:
        states = (controller(), controller())
        effect_cursors = (
            EventCursor(10, 150, 3),
            EventCursor(10, 150, 1),
        )

        added = tuple(
            state.add_paired_position(
                position_fact("p"),
                absolute_target_tick=100,
                cursor=effect_cursor,
            )
            for state, effect_cursor in zip(states, effect_cursors, strict=True)
        )
        first_commands = tuple(fact.commands[0] for fact in added)
        self.assertNotEqual(first_commands[0].cursor, first_commands[1].cursor)
        self.assertEqual(
            first_commands[0].candidate_intent_id,
            first_commands[1].candidate_intent_id,
        )
        self.assertEqual(first_commands[0].request_id, first_commands[1].request_id)
        self.assertEqual(
            states[0].pending_orders[0].intent_cursor,
            EventCursor(10, 150, 0),
        )
        self.assertEqual(
            states[0].pending_orders[0].intent_cursor,
            states[1].pending_orders[0].intent_cursor,
        )

        raw_ids = tuple(
            assign_and_send(
                state,
                assigned_cursor=EventCursor(10, 600, 1),
                sent_cursor=EventCursor(10, 700, 1),
            )
            for state in states
        )
        self.assertEqual(raw_ids[0], raw_ids[1])
        for state in states:
            state.verify()

    def test_noop_reconcile_does_not_shift_later_candidate_identity(self) -> None:
        states = (controller(), controller())
        for state in states:
            state.add_paired_position(
                position_fact("p"),
                absolute_target_tick=100,
                cursor=cursor(2),
            )
            state.set_position_desired(
                "p",
                absolute_target_tick=None,
                desired_shares=0,
                cursor=EventCursor(10, 150, 1),
            )

        noop = states[1].set_position_desired(
            "p",
            absolute_target_tick=None,
            desired_shares=0,
            cursor=EventCursor(10, 150, 2),
        )
        self.assertEqual(noop.commands, ())
        reopened = (
            states[0].set_position_desired(
                "p",
                absolute_target_tick=100,
                desired_shares=1000,
                cursor=EventCursor(10, 150, 2),
            ),
            states[1].set_position_desired(
                "p",
                absolute_target_tick=100,
                desired_shares=1000,
                cursor=EventCursor(10, 150, 3),
            ),
        )

        commands = tuple(fact.commands[0] for fact in reopened)
        self.assertNotEqual(commands[0].cursor, commands[1].cursor)
        self.assertEqual(
            commands[0].candidate_intent_id, commands[1].candidate_intent_id
        )
        self.assertEqual(commands[0].request_id, commands[1].request_id)
        self.assertEqual(
            states[0].pending_orders[0].intent_cursor,
            EventCursor(10, 150, 0),
        )
        self.assertEqual(
            states[0].pending_orders[0].intent_cursor,
            states[1].pending_orders[0].intent_cursor,
        )
        for state in states:
            state.verify()

    def test_same_phase_reopen_gets_distinct_product_local_emission_rows(self) -> None:
        state = controller()
        first = state.add_paired_position(
            position_fact("p"),
            absolute_target_tick=100,
            cursor=EventCursor(10, 150, 4),
        )
        first_candidate = first.commands[0].candidate_intent_id

        state.set_position_desired(
            "p",
            absolute_target_tick=None,
            desired_shares=0,
            cursor=EventCursor(10, 150, 6),
        )
        reopened = state.set_position_desired(
            "p",
            absolute_target_tick=100,
            desired_shares=1000,
            cursor=EventCursor(10, 150, 9),
        )
        second_candidate = reopened.commands[0].candidate_intent_id
        state.set_position_desired(
            "p",
            absolute_target_tick=None,
            desired_shares=0,
            cursor=EventCursor(10, 150, 10),
        )
        reopened_again = state.set_position_desired(
            "p",
            absolute_target_tick=100,
            desired_shares=1000,
            cursor=EventCursor(10, 150, 11),
        )
        third_candidate = reopened_again.commands[0].candidate_intent_id

        self.assertEqual(len({first_candidate, second_candidate, third_candidate}), 3)
        self.assertEqual(
            state.pending_orders[0].intent_cursor,
            EventCursor(10, 150, 2),
        )
        state.verify()

    def test_prefix_replay_reconstructs_next_candidate_emission_identity(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"),
            absolute_target_tick=100,
            cursor=EventCursor(10, 150, 2),
        )
        state.set_position_desired(
            "p",
            absolute_target_tick=None,
            desired_shares=0,
            cursor=EventCursor(10, 150, 4),
        )
        state.set_position_desired(
            "p",
            absolute_target_tick=None,
            desired_shares=0,
            cursor=EventCursor(10, 150, 5),
        )
        replayed = replay_exit_inventory_facts(
            state.facts,
            Date=state.Date,
            ValueCode=state.ValueCode,
            QuoteCode=state.QuoteCode,
            scenario_id=state.scenario_id,
        )

        expected = state.set_position_desired(
            "p",
            absolute_target_tick=100,
            desired_shares=1000,
            cursor=EventCursor(10, 150, 8),
        )
        actual = replayed.set_position_desired(
            "p",
            absolute_target_tick=100,
            desired_shares=1000,
            cursor=EventCursor(10, 150, 8),
        )

        self.assertEqual(actual, expected)
        self.assertEqual(
            actual.commands[0].request_id,
            expected.commands[0].request_id,
        )
        state.verify()
        replayed.verify()

    def test_candidate_emission_counter_is_committed_by_state_digest(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"),
            absolute_target_tick=100,
            cursor=EventCursor(10, 150, 4),
        )
        state.verify()

        state._candidate_emission_next_row_by_tick[100] += 1

        with self.assertRaisesRegex(
            ExitInventoryReplayError,
            "final state does not match",
        ):
            state.verify()

    def test_same_price_position_join_cancel_replaces_whole_aggregate(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p1"), absolute_target_tick=100, cursor=cursor(2)
        )
        first_raw = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        joined = state.add_paired_position(
            position_fact("p2", established_ns=5, sequence=2),
            absolute_target_tick=100,
            cursor=cursor(6),
        )
        self.assertEqual([item.kind for item in joined.commands], ["enqueue_cancel"])
        self.assertEqual(state.working_orders[0].raw_order_fact_id, first_raw)
        self.assertEqual(
            [item.position_id for item in state.working_orders[0].members], ["p1"]
        )
        self.assertIsNone(state.positions[1].allocation_id)

        state.on_cancel_assigned(first_raw, cursor(7))
        cancelled = state.on_cancel_sent(first_raw, cursor(8))
        self.assertEqual(cancelled.terminals[0].leaves_shares, 1000)
        self.assertEqual(cancelled.commands[0].kind, "enqueue_new")
        self.assertEqual(
            [item.position_id for item in cancelled.commands[0].members],
            ["p1", "p2"],
        )
        replacement_raw = assign_and_send(
            state, assigned_cursor=cursor(9), sent_cursor=cursor(10)
        )
        self.assertNotEqual(replacement_raw, first_raw)
        self.assertEqual(state.working_orders[0].actual_start_cursor, cursor(10))
        self.assertEqual(len(state.working_orders), 1)

    def test_unassigned_new_coalesces_but_assigned_stale_new_must_send_cancel(
        self,
    ) -> None:
        state = controller()
        first = state.add_paired_position(
            position_fact("p", shares=2000, contracts=2),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        first_candidate = first.commands[0].candidate_intent_id
        changed = state.set_position_desired(
            "p",
            absolute_target_tick=101,
            desired_shares=1000,
            cursor=cursor(3),
        )
        self.assertEqual(
            [item.kind for item in changed.commands],
            ["withdraw_pending_new", "enqueue_new"],
        )
        second_candidate = state.pending_orders[0].candidate_intent_id
        self.assertNotEqual(first_candidate, second_candidate)
        state.on_new_assigned(second_candidate, cursor(4))
        stale = state.set_position_desired(
            "p",
            absolute_target_tick=102,
            desired_shares=1000,
            cursor=cursor(5),
        )
        self.assertEqual(stale.commands, ())
        self.assertTrue(state.pending_orders[0].stale)
        sent = state.on_new_sent(second_candidate, cursor(6), 101)
        self.assertEqual([item.kind for item in sent.commands], ["enqueue_cancel"])
        self.assertEqual(state.working_orders[0].cancel_state, "pending")

    def test_move_to_occupied_price_never_double_allocates_inventory(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p1"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_100 = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.add_paired_position(
            position_fact("p2", established_ns=5, sequence=2),
            absolute_target_tick=101,
            cursor=cursor(6),
        )
        raw_101 = assign_and_send(
            state, assigned_cursor=cursor(7), sent_cursor=cursor(8)
        )
        moved = state.set_position_desired(
            "p1",
            absolute_target_tick=101,
            desired_shares=1000,
            cursor=cursor(9),
        )
        self.assertEqual(moved.commands[0].raw_order_fact_id, raw_100)
        self.assertEqual(state.positions[0].allocation_id, raw_100)
        state.on_cancel_assigned(raw_100, cursor(10))
        old_cancelled = state.on_cancel_sent(raw_100, cursor(11))
        self.assertEqual(old_cancelled.commands[0].raw_order_fact_id, raw_101)
        self.assertEqual(old_cancelled.commands[0].kind, "enqueue_cancel")
        self.assertIsNone(state.positions[0].allocation_id)
        self.assertEqual(state.positions[1].allocation_id, raw_101)
        state.on_cancel_assigned(raw_101, cursor(12))
        destination_cancelled = state.on_cancel_sent(raw_101, cursor(13))
        self.assertEqual(destination_cancelled.commands[0].kind, "enqueue_new")
        self.assertEqual(
            [item.position_id for item in destination_cancelled.commands[0].members],
            ["p1", "p2"],
        )
        allocations = [item.allocation_id for item in state.positions]
        self.assertEqual(len(set(allocations)), 1)

    def test_move_between_unsent_prices_rebuilds_destination_in_one_reconcile(
        self,
    ) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p1", established_row=0, sequence=1),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        state.add_paired_position(
            position_fact("p2", established_row=1, sequence=2),
            absolute_target_tick=101,
            cursor=cursor(3),
        )
        moved = state.set_position_desired(
            "p1",
            absolute_target_tick=101,
            desired_shares=1000,
            cursor=cursor(4),
        )
        self.assertEqual(
            sorted(item.kind for item in moved.commands[:2]),
            ["withdraw_pending_new", "withdraw_pending_new"],
        )
        self.assertEqual(moved.commands[-1].kind, "enqueue_new")
        self.assertEqual(len(state.pending_orders), 1)
        self.assertEqual(state.pending_orders[0].absolute_price_tick, 101)
        self.assertEqual(
            [item.position_id for item in state.pending_orders[0].members],
            ["p1", "p2"],
        )

    def test_exact_fill_uses_member_fifo_and_rejects_overfill_atomically(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p-b", established_ns=1, established_row=0, sequence=1),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        state.add_paired_position(
            position_fact("p-a", established_ns=1, established_row=1, sequence=2),
            absolute_target_tick=100,
            cursor=cursor(3),
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(4), sent_cursor=cursor(5)
        )
        first = state.on_fill(raw_id, cursor(6), 1500)
        self.assertEqual(
            [item.position_id for item in first.fill_allocations], ["p-a", "p-b"]
        )
        self.assertEqual(
            [item.allocated_shares for item in first.fill_allocations], [1000, 500]
        )
        self.assertFalse(first.fill_allocations[0].partial_member_fill)
        self.assertTrue(first.fill_allocations[1].partial_member_fill)
        self.assertTrue(
            all(item.execution_truth == "exact" for item in first.fill_allocations)
        )
        self.assertEqual(len(first.hedge_unit_requests), 1)
        with self.assertRaisesRegex(ExitInventoryError, "exceeds"):
            state.on_fill(raw_id, cursor(7), 501)
        final = state.on_fill(raw_id, cursor(7), 500)
        self.assertEqual(len(final.hedge_unit_requests), 1)
        self.assertEqual(final.terminals[0].terminal_reason, "filled")
        self.assertEqual(
            [item.available_spot_shares for item in state.positions], [0, 0]
        )

    def test_partial_fills_emit_integral_hedge_then_exact_spot_rollback(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p", shares=2000, contracts=2),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        first = state.on_fill(raw_id, cursor(5), 600)
        self.assertEqual(first.hedge_unit_requests, ())
        second = state.on_fill(raw_id, cursor(6), 900)
        self.assertEqual(len(second.hedge_unit_requests), 1)
        hedge = second.hedge_unit_requests[0]
        self.assertEqual(hedge.contracts, 1)
        self.assertEqual(hedge.spot_exit_shares, 1000)
        self.assertEqual(sum(item.shares for item in hedge.sources), 1000)
        self.assertEqual(state.positions[0].unhedged_fill_shares, 500)

        requested = state.set_position_desired(
            "p", absolute_target_tick=None, desired_shares=0, cursor=cursor(7)
        )
        self.assertEqual(requested.commands[0].kind, "enqueue_cancel")
        state.on_cancel_assigned(raw_id, cursor(8))
        cancelled = state.on_cancel_sent(raw_id, cursor(9))
        self.assertEqual(len(cancelled.rollback_requests), 1)
        rollback = cancelled.rollback_requests[0]
        self.assertEqual(rollback.rollback_spot_shares, 500)
        self.assertEqual(sum(item.shares for item in rollback.sources), 500)
        self.assertEqual(state.positions[0].rollback_pending_shares, 500)
        self.assertEqual(state.positions[0].unhedged_fill_shares, 0)
        with self.assertRaisesRegex(ExitInventoryError, "rollback-pending"):
            state.set_position_desired(
                "p",
                absolute_target_tick=101,
                desired_shares=500,
                cursor=cursor(10),
            )

    def test_hedge_sent_is_exact_terminal_and_makes_full_position_flat(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        filled = state.on_fill(raw_id, cursor(5), 1000)
        hedge = filled.hedge_unit_requests[0]
        pending = state.positions[0]
        self.assertEqual(pending.hedge_pending_shares, 1000)
        self.assertEqual(pending.resolution_state, "unresolved")
        self.assertEqual(pending.unresolved_reasons, ("hedge_pending",))
        with self.assertRaisesRegex(ExitInventoryError, "strictly increasing"):
            state.on_hedge_sent(hedge.request_id, cursor(5))

        sent = state.on_hedge_sent(hedge.request_id, cursor(6))
        self.assertEqual(sent.hedge_terminals[0].terminal_reason, "sent")
        self.assertEqual(sent.hedge_terminals[0].sources, hedge.sources)
        self.assertEqual(state.pending_hedge_unit_requests, ())
        self.assertEqual(state.positions[0].hedged_exit_shares, 1000)
        self.assertEqual(state.positions[0].unresolved_shares, 0)
        self.assertEqual(state.positions[0].resolution_state, "flat")
        self.assertTrue(state.positions[0].capacity_releasable)
        self.assertEqual(state.unresolved_positions, ())
        self.assertEqual([item.position_id for item in state.flat_positions], ["p"])
        with self.assertRaisesRegex(ExitInventoryError, "already completed"):
            state.on_hedge_sent(hedge.request_id, cursor(7))
        with self.assertRaisesRegex(ExitInventoryError, "unknown pending hedge"):
            state.on_hedge_sent("missing", cursor(7))
        state.verify()

    def test_one_4000_share_fill_emits_two_independent_2000_share_hedges(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p", shares=4000, contracts=2),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        filled = state.on_fill(raw_id, cursor(5), 4000)
        self.assertEqual(len(filled.fill_allocations), 1)
        self.assertEqual(len(filled.hedge_unit_requests), 2)
        self.assertEqual(
            [request.future_share_equivalent for request in filled.hedge_unit_requests],
            [2000, 2000],
        )
        self.assertEqual(
            [
                request.sources[0].allocation_id
                for request in filled.hedge_unit_requests
            ],
            [filled.fill_allocations[0].allocation_id] * 2,
        )
        self.assertEqual(
            [request.sources[0].shares for request in filled.hedge_unit_requests],
            [2000, 2000],
        )
        state.on_hedge_sent(filled.hedge_unit_requests[0].request_id, cursor(6))
        self.assertEqual(state.positions[0].unresolved_shares, 2000)
        state.on_hedge_sent(filled.hedge_unit_requests[1].request_id, cursor(7))
        self.assertEqual(state.positions[0].resolution_state, "flat")
        state.verify()

    def test_hedge_timeout_rolls_back_full_unit_with_identical_sources(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p", shares=4000, contracts=2),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.on_fill(raw_id, cursor(5, 0), 600)
        filled = state.on_fill(raw_id, cursor(5, 1), 1400)
        hedge = filled.hedge_unit_requests[0]
        self.assertEqual([source.shares for source in hedge.sources], [600, 1400])

        timed_out = state.on_hedge_timeout(hedge.request_id, cursor(6))
        terminal = timed_out.hedge_terminals[0]
        rollback = timed_out.rollback_requests[0]
        self.assertEqual(terminal.terminal_reason, "retry_timeout")
        self.assertEqual(terminal.resulting_rollback_request_id, rollback.request_id)
        self.assertEqual(rollback.reason, "hedge_retry_timeout")
        self.assertEqual(rollback.rollback_spot_shares, 2000)
        self.assertEqual(rollback.sources, hedge.sources)
        self.assertEqual(timed_out.commands[0].kind, "enqueue_cancel")
        self.assertEqual(state.positions[0].hedge_pending_shares, 0)
        self.assertEqual(state.positions[0].rollback_pending_shares, 2000)
        with self.assertRaisesRegex(ExitInventoryError, "already completed"):
            state.on_hedge_sent(hedge.request_id, cursor(7))

        restored = state.on_rollback_sent(rollback.request_id, cursor(7))
        self.assertEqual(restored.rollback_terminals[0].restored_available_shares, 2000)
        self.assertEqual(state.positions[0].available_spot_shares, 4000)
        self.assertEqual(state.positions[0].rollback_pending_shares, 0)
        self.assertEqual(
            state.positions[0].unresolved_reasons, ("available_inventory",)
        )
        state.set_position_desired(
            "p", absolute_target_tick=101, desired_shares=2000, cursor=cursor(8)
        )
        with self.assertRaisesRegex(ExitInventoryError, "already completed"):
            state.on_rollback_sent(rollback.request_id, cursor(9))
        state.verify()

    def test_subunit_rollback_sent_restores_oldest_available_inventory(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("old", established_ns=1, sequence=1),
            absolute_target_tick=100,
            cursor=cursor(2),
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.on_fill(raw_id, cursor(5), 400)
        state.set_position_desired(
            "old", absolute_target_tick=None, desired_shares=0, cursor=cursor(6)
        )
        state.on_cancel_assigned(raw_id, cursor(7))
        cancelled = state.on_cancel_sent(raw_id, cursor(8))
        rollback = cancelled.rollback_requests[0]
        state.add_paired_position(
            position_fact("new", established_ns=9, sequence=2),
            absolute_target_tick=101,
            cursor=cursor(10),
        )
        state.on_rollback_sent(rollback.request_id, cursor(11))
        self.assertEqual(state.positions[0].position_id, "old")
        self.assertEqual(state.positions[0].available_spot_shares, 1000)
        state.set_position_desired(
            "old", absolute_target_tick=102, desired_shares=1000, cursor=cursor(12)
        )
        self.assertEqual(state.unresolved_positions[0].position_id, "old")
        state.verify()

    def test_rollback_failed_stays_unresolved_and_can_never_be_requoted(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.on_fill(raw_id, cursor(5), 400)
        state.set_position_desired(
            "p", absolute_target_tick=None, desired_shares=0, cursor=cursor(6)
        )
        state.on_cancel_assigned(raw_id, cursor(7))
        cancelled = state.on_cancel_sent(raw_id, cursor(8))
        rollback = cancelled.rollback_requests[0]
        failed = state.on_rollback_failed(rollback.request_id, cursor(9))
        self.assertEqual(failed.rollback_terminals[0].failed_unresolved_shares, 400)
        view = state.positions[0]
        self.assertEqual(view.rollback_pending_shares, 0)
        self.assertEqual(view.rollback_failed_shares, 400)
        self.assertEqual(view.unresolved_shares, 1000)
        self.assertEqual(view.resolution_state, "unresolved")
        self.assertFalse(view.capacity_releasable)
        self.assertEqual(
            view.unresolved_reasons,
            ("available_inventory", "rollback_failed"),
        )
        self.assertEqual(state.flat_positions, ())
        with self.assertRaisesRegex(ExitInventoryError, "rollback-failed"):
            state.set_position_desired(
                "p", absolute_target_tick=101, desired_shares=600, cursor=cursor(10)
            )
        with self.assertRaisesRegex(ExitInventoryError, "already completed"):
            state.on_rollback_failed(rollback.request_id, cursor(10))
        state.verify()

    def test_session_expiry_withdraws_requests_and_rolls_back_partial(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p1"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.on_fill(raw_id, cursor(5), 300)
        state.add_paired_position(
            position_fact("p2", established_ns=6, sequence=2),
            absolute_target_tick=101,
            cursor=cursor(7),
        )
        expired = state.session_expiry(cursor(8))
        self.assertTrue(state.session_expired)
        self.assertEqual(
            [item.kind for item in expired.commands], ["withdraw_pending_new"]
        )
        self.assertEqual(expired.terminals[0].terminal_reason, "session_expired")
        self.assertEqual(expired.rollback_requests[0].rollback_spot_shares, 300)
        self.assertEqual(len(state.pending_orders), 0)
        self.assertEqual(len(state.working_orders), 0)
        with self.assertRaisesRegex(ExitInventoryError, "after expiry"):
            state.add_paired_position(
                position_fact("p3", established_ns=6, established_row=1),
                absolute_target_tick=100,
                cursor=cursor(9),
            )

    def test_same_cursor_fill_precedes_assigned_cancel_effect(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.set_position_desired(
            "p", absolute_target_tick=None, desired_shares=0, cursor=cursor(5)
        )
        state.on_cancel_assigned(raw_id, cursor(10, 0))
        filled = state.on_fill(raw_id, cursor(10, 1), 1000)
        self.assertTrue(filled.terminals[0].frozen_cancel_will_be_noop)
        self.assertEqual(filled.commands, ())
        noop = state.on_cancel_sent(raw_id, cursor(10, 2))
        self.assertEqual(noop.terminals, ())
        with self.assertRaisesRegex(ExitInventoryError, "already consumed"):
            state.on_cancel_sent(raw_id, cursor(10, 3))

    def test_fill_terminal_withdraws_only_unassigned_cancel(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.set_position_desired(
            "p", absolute_target_tick=None, desired_shares=0, cursor=cursor(5)
        )
        filled = state.on_fill(raw_id, cursor(6), 1000)
        self.assertEqual(
            [item.kind for item in filled.commands], ["withdraw_pending_cancel"]
        )
        self.assertFalse(filled.terminals[0].frozen_cancel_will_be_noop)
        with self.assertRaisesRegex(ExitInventoryError, "not working"):
            state.on_cancel_assigned(raw_id, cursor(7))

    def test_expiry_requires_assigned_callbacks_to_be_drained(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        candidate = state.pending_orders[0].candidate_intent_id
        state.on_new_assigned(candidate, cursor(3))
        with self.assertRaisesRegex(ExitInventoryError, "assigned new"):
            state.session_expiry(cursor(4))
        state.on_new_sent(candidate, cursor(4), 100)
        state.set_position_desired(
            "p", absolute_target_tick=None, desired_shares=0, cursor=cursor(5)
        )
        raw_id = state.working_orders[0].raw_order_fact_id
        state.on_cancel_assigned(raw_id, cursor(6))
        with self.assertRaisesRegex(ExitInventoryError, "assigned cancel"):
            state.session_expiry(cursor(7))

    def test_replay_rejects_allocation_command_and_digest_tampering(self) -> None:
        state = controller()
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        raw_id = assign_and_send(
            state, assigned_cursor=cursor(3), sent_cursor=cursor(4)
        )
        state.on_fill(raw_id, cursor(5), 400)
        state.session_expiry(cursor(6))
        state.verify()
        config = {
            "Date": "20260826",
            "ValueCode": "2330",
            "QuoteCode": "2330F",
            "scenario_id": "scenario-a",
        }

        allocation_facts = list(state.facts)
        fill_index = next(
            index
            for index, fact in enumerate(allocation_facts)
            if fact.operation == "fill"
        )
        fill_fact = allocation_facts[fill_index]
        allocation_facts[fill_index] = replace(
            fill_fact,
            fill_allocations=(
                replace(
                    fill_fact.fill_allocations[0],
                    allocated_shares=399,
                ),
            ),
        )
        with self.assertRaises(ExitInventoryReplayError):
            replay_exit_inventory_facts(allocation_facts, **config)

        command_facts = list(state.facts)
        first = command_facts[0]
        command_facts[0] = replace(
            first,
            commands=(replace(first.commands[0], total_shares=999),),
        )
        with self.assertRaises(ExitInventoryReplayError):
            replay_exit_inventory_facts(command_facts, **config)

        digest_facts = list(state.facts)
        digest_facts[-1] = replace(digest_facts[-1], state_digest="tampered")
        with self.assertRaises(ExitInventoryReplayError):
            replay_exit_inventory_facts(digest_facts, **config)

        reordered_facts = list(state.facts)
        reordered_facts[1], reordered_facts[2] = (
            reordered_facts[2],
            reordered_facts[1],
        )
        with self.assertRaises(ExitInventoryReplayError):
            replay_exit_inventory_facts(reordered_facts, **config)

        dropped_facts = list(state.facts)
        dropped_facts.pop(1)
        with self.assertRaises(ExitInventoryReplayError):
            replay_exit_inventory_facts(dropped_facts, **config)

    def test_incremental_digest_scales_near_linearly_through_10k_callbacks(
        self,
    ) -> None:
        def replay_cancel_cycles(
            cycles: int,
        ) -> tuple[float, S1ExitInventoryController]:
            state = controller()
            next_cursor = 2
            started = perf_counter()
            first = state.add_paired_position(
                position_fact("p"),
                absolute_target_tick=100,
                cursor=cursor(next_cursor),
            )
            self.assertEqual(first.digest_version, 2)
            next_cursor += 1
            raw_id = assign_and_send(
                state,
                assigned_cursor=cursor(next_cursor),
                sent_cursor=cursor(next_cursor + 1),
            )
            next_cursor += 2
            for index in range(cycles):
                state.set_position_desired(
                    "p",
                    absolute_target_tick=101 + index % 2,
                    desired_shares=1000,
                    cursor=cursor(next_cursor),
                )
                next_cursor += 1
                state.on_cancel_assigned(raw_id, cursor(next_cursor))
                next_cursor += 1
                state.on_cancel_sent(raw_id, cursor(next_cursor))
                next_cursor += 1
                candidate_id = state.pending_orders[0].candidate_intent_id
                state.on_new_assigned(candidate_id, cursor(next_cursor))
                next_cursor += 1
                sent = state.on_new_sent(
                    candidate_id,
                    cursor(next_cursor),
                    state.pending_orders[0].absolute_price_tick,
                )
                next_cursor += 1
                assert sent.raw_order_fact_id is not None
                raw_id = sent.raw_order_fact_id
            return perf_counter() - started, state

        short_seconds, short_state = replay_cancel_cycles(400)
        long_seconds, long_state = replay_cancel_cycles(2000)

        self.assertEqual(len(short_state.facts), 2_003)
        self.assertEqual(len(long_state.facts), 10_003)
        self.assertLess(long_seconds, short_seconds * 8 + 0.1)
        long_state.verify()

    def test_integer_quantity_and_full_cursor_are_strict(self) -> None:
        with self.assertRaisesRegex(ExitInventoryError, "valid YYYYMMDD"):
            S1ExitInventoryController(
                Date="20260230",
                ValueCode="2330",
                QuoteCode="2330F",
                scenario_id="scenario-a",
            )
        state = controller()
        with self.assertRaisesRegex(ExitInventoryError, "positive integer"):
            state.add_paired_position(
                position_fact("p"),
                absolute_target_tick=True,  # type: ignore[arg-type]
                cursor=cursor(2),
            )
        state.add_paired_position(
            position_fact("p"), absolute_target_tick=100, cursor=cursor(2)
        )
        with self.assertRaisesRegex(ExitInventoryError, "strictly increasing"):
            state.set_position_desired(
                "p",
                absolute_target_tick=100,
                desired_shares=1000,
                cursor=cursor(2),
            )


if __name__ == "__main__":
    unittest.main()
