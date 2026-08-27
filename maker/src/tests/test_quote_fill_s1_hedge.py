from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_hedge import (
    HEDGE_DELAY_NS,
    HEDGE_RETRY_NS,
    HedgeAttempt,
    HedgeIntent,
    RawBookCursor,
    RawBookEvent,
    RawBookLevel,
    RawBookStateMachine,
    RollbackAttempt,
    RollbackSpec,
    capture_arrival_reference,
    executable_book,
)


def level(price: float, quantity: int = 1) -> RawBookLevel:
    return RawBookLevel(price, quantity)


def event(
    time_ns: int,
    *,
    row: int = 0,
    packet: int = 0,
    trial: bool = False,
    formal: bool = True,
    bids: tuple[RawBookLevel, ...] = (RawBookLevel(99.0, 10),),
    asks: tuple[RawBookLevel, ...] = (RawBookLevel(101.0, 10),),
    best_bid: RawBookLevel | None = None,
    best_ask: RawBookLevel | None = None,
    reference: float | None = 100.0,
) -> RawBookEvent:
    return RawBookEvent(
        RawBookCursor(EventCursor(time_ns, 3, row), packet),
        trial_match=trial,
        formal_book=formal,
        reference_price=reference,
        l1_bids=bids,
        l1_asks=asks,
        best_bid=best_bid,
        best_ask=best_ask,
    )


def intent(*, session_end: int = 20_000_000_000) -> HedgeIntent:
    return HedgeIntent(
        hedge_intent_id="hedge-1",
        trigger_cursor=EventCursor(1_000_000_000, 7, 4),
        side="sell",
        hedge_quantity=1,
        hedge_quantity_unit="future_contracts",
        session_end_time_ns=session_end,
        initiating_first_leg_venue="spot",
        initiating_execution_id="fill-1",
        initiating_first_leg_side="buy",
        initiating_first_leg_quantity=2_000,
        initiating_first_leg_quantity_unit="spot_shares",
    )


def timed_out_rollback() -> RollbackSpec:
    hedge_intent = intent()
    attempt = HedgeAttempt(hedge_intent)
    attempt.observe(
        None,
        evaluation_cursor=EventCursor(hedge_intent.target_time_ns, 3, 0),
    )
    attempt.observe(
        None,
        evaluation_cursor=EventCursor(hedge_intent.deadline_time_ns, 3, 0),
    )
    result = attempt.expire(EventCursor(hedge_intent.deadline_time_ns, 4, 0))
    assert result.rollback is not None
    return result.rollback


class S1HedgeTest(unittest.TestCase):
    def test_arrival_invalid_does_not_short_circuit_valid_t0(self) -> None:
        machine = RawBookStateMachine()
        invalid = machine.ingest(event(1_000_000_000, bids=(), asks=(), packet=41))
        assert invalid is not None
        hedge_intent = intent()
        arrival = capture_arrival_reference(
            invalid,
            reference_cursor=hedge_intent.trigger_cursor,
            side=hedge_intent.side,
            quantity=hedge_intent.hedge_quantity,
            quantity_unit=hedge_intent.hedge_quantity_unit,
        )
        valid = machine.ingest(event(1_000_000_000 + HEDGE_DELAY_NS, row=1))
        assert valid is not None
        attempt = HedgeAttempt(hedge_intent, arrival_reference=arrival)
        result = attempt.observe(valid)
        self.assertEqual(result.status, "send_eligible")
        assert result.arrival_reference is not None
        self.assertFalse(result.arrival_reference.available)
        self.assertEqual(result.arrival_reference.gate_reason, "empty_book_side")
        assert result.arrival_reference.book_cursor is not None
        self.assertEqual(result.arrival_reference.book_cursor.packet_sequence, 41)
        assert result.send_eligible is not None
        self.assertEqual(result.send_eligible.book_cursor.packet_sequence, 0)

    def test_explicit_missing_book_at_t0_preserves_initial_status(self) -> None:
        hedge_intent = intent()
        attempt = HedgeAttempt(hedge_intent)
        at_t0 = EventCursor(hedge_intent.target_time_ns, 9, 0)
        first = attempt.observe(None, evaluation_cursor=at_t0)
        self.assertEqual(first.status, "waiting")
        self.assertEqual(first.initial_gate_reason, "missing_book")
        self.assertEqual(first.last_gate_reason, "missing_book")
        self.assertTrue(first.target_evaluated)

        machine = RawBookStateMachine()
        legal = machine.ingest(event(hedge_intent.target_time_ns + 1, row=1, packet=73))
        assert legal is not None
        second = attempt.observe(legal)
        self.assertEqual(second.status, "send_eligible")
        self.assertEqual(second.initial_gate_reason, "missing_book")
        assert second.send_eligible is not None
        self.assertEqual(second.send_eligible.book_cursor.packet_sequence, 73)

    def test_t0_uses_latest_causal_book_without_requiring_an_update_at_t0(self) -> None:
        machine = RawBookStateMachine()
        state = machine.ingest(event(1_000_000_010))
        assert state is not None
        hedge_intent = intent()
        t0 = EventCursor(hedge_intent.target_time_ns, 9, 0)
        result = HedgeAttempt(hedge_intent).observe(state, evaluation_cursor=t0)
        self.assertEqual(result.status, "send_eligible")
        assert result.send_eligible is not None
        self.assertEqual(result.send_eligible.send_eligible_cursor, t0)
        self.assertLess(
            result.send_eligible.book_cursor.cursor,
            result.send_eligible.send_eligible_cursor,
        )

    def test_trial_match_clears_book_until_formal_reopen(self) -> None:
        machine = RawBookStateMachine()
        machine.ingest(event(10))
        closed = machine.ingest(event(20, trial=True, formal=False))
        assert closed is not None
        self.assertFalse(closed.gate_open)
        self.assertEqual((closed.bids, closed.asks), ((), ()))
        still_closed = machine.ingest(event(30, formal=False))
        assert still_closed is not None
        self.assertFalse(still_closed.gate_open)
        reopened = machine.ingest(event(40, formal=True))
        assert reopened is not None
        self.assertTrue(reopened.gate_open)

    def test_packet_sequence_is_part_of_full_raw_cursor(self) -> None:
        machine = RawBookStateMachine()
        first = machine.ingest(event(10, row=2, packet=7))
        second = machine.ingest(event(10, row=2, packet=8))
        assert first is not None and second is not None
        self.assertEqual(first.book_cursor.packet_sequence, 7)
        self.assertEqual(second.book_cursor.packet_sequence, 8)
        with self.assertRaisesRegex(ValueError, "strictly cursor ordered"):
            machine.ingest(event(10, row=2, packet=8))

    def test_legal_candidate_becomes_terminal_only_after_confirmed_send(self) -> None:
        machine = RawBookStateMachine()
        attempt = HedgeAttempt(intent())
        invalid = machine.ingest(event(1_000_000_000 + HEDGE_DELAY_NS, bids=(), row=1))
        assert invalid is not None
        self.assertEqual(attempt.observe(invalid).status, "waiting")
        valid = machine.ingest(
            event(1_000_000_000 + HEDGE_DELAY_NS + 10, row=2, packet=77)
        )
        assert valid is not None
        result = attempt.observe(valid)
        self.assertEqual(result.status, "send_eligible")
        assert result.send_eligible is not None
        self.assertEqual(result.send_eligible.book_cursor.packet_sequence, 77)
        sent = attempt.confirm_actual_send(result.send_eligible.send_eligible_cursor)
        self.assertEqual(sent.status, "hedge_sent")
        self.assertIsNone(sent.send_eligible)
        self.assertIsNotNone(sent.actual_send)
        later = machine.ingest(
            event(1_000_000_000 + HEDGE_DELAY_NS + 20, row=3, packet=88)
        )
        assert later is not None
        self.assertEqual(attempt.observe(later), sent)

    def test_no_token_candidate_is_revalidated_and_can_become_illegal(self) -> None:
        machine = RawBookStateMachine()
        hedge_intent = intent()
        attempt = HedgeAttempt(hedge_intent)
        legal = machine.ingest(event(hedge_intent.target_time_ns, row=1))
        assert legal is not None
        first = attempt.observe(legal)
        self.assertEqual(first.status, "send_eligible")
        self.assertTrue(first.target_evaluated)
        self.assertIsNone(first.initial_gate_reason)

        invalid = machine.ingest(
            event(hedge_intent.target_time_ns + 10, row=2, bids=())
        )
        assert invalid is not None
        second = attempt.observe(invalid)
        self.assertEqual(second.status, "waiting")
        self.assertIsNone(second.send_eligible)
        with self.assertRaisesRegex(ValueError, "currently legal"):
            attempt.confirm_actual_send(invalid.book_cursor.cursor)

        legal_again = machine.ingest(
            event(hedge_intent.target_time_ns + 20, row=3, packet=9)
        )
        assert legal_again is not None
        third = attempt.observe(legal_again)
        self.assertEqual(third.status, "send_eligible")
        sent = attempt.confirm_actual_send(legal_again.book_cursor.cursor)
        self.assertEqual(sent.status, "hedge_sent")

    def test_swept_l2_outside_band_is_illegal(self) -> None:
        machine = RawBookStateMachine()
        state = machine.ingest(event(10, asks=(level(101.0, 1), level(109.0, 10))))
        assert state is not None
        executable, reason = executable_book(
            state,
            side="buy",
            quantity=2,
            quantity_unit="future_contracts",
        )
        self.assertIsNone(executable)
        self.assertEqual(reason, "swept_level_outside_reference_band")

    def test_exact_five_seconds_is_inclusive(self) -> None:
        hedge_intent = intent()
        machine = RawBookStateMachine()
        initial = machine.ingest(event(hedge_intent.target_time_ns, bids=(), row=1))
        assert initial is not None
        attempt = HedgeAttempt(hedge_intent)
        self.assertEqual(attempt.observe(initial).status, "waiting")
        deadline = machine.ingest(
            event(hedge_intent.target_time_ns + HEDGE_RETRY_NS, row=2)
        )
        assert deadline is not None
        result = attempt.observe(deadline)
        self.assertEqual(result.status, "send_eligible")
        assert result.send_eligible is not None
        sent = attempt.confirm_actual_send(result.send_eligible.send_eligible_cursor)
        self.assertEqual(sent.status, "hedge_sent")

    def test_timeout_cannot_revive_and_produces_rollback(self) -> None:
        hedge_intent = intent()
        attempt = HedgeAttempt(hedge_intent)
        machine = RawBookStateMachine()
        at_t0 = machine.ingest(event(hedge_intent.target_time_ns, bids=(), row=1))
        assert at_t0 is not None
        attempt.observe(at_t0)
        at_deadline = machine.ingest(
            event(hedge_intent.deadline_time_ns, bids=(), row=2)
        )
        assert at_deadline is not None
        attempt.observe(at_deadline)
        timeout_cursor = EventCursor(hedge_intent.deadline_time_ns, 4, 0)
        timeout = attempt.expire(timeout_cursor)
        self.assertEqual(timeout.status, "hedge_retry_timeout")
        self.assertIsNotNone(timeout.rollback)
        assert timeout.rollback is not None
        self.assertEqual(timeout.rollback.source_hedge_intent_id, "hedge-1")
        self.assertEqual(timeout.rollback.initiating_execution_id, "fill-1")
        self.assertEqual(timeout.rollback.side, "sell")
        self.assertEqual(timeout.rollback.trigger_cursor, timeout_cursor)
        self.assertEqual(timeout.rollback.quantity, 2_000)
        self.assertEqual(timeout.rollback.quantity_unit, "spot_shares")
        self.assertNotEqual(
            timeout.rollback.quantity,
            hedge_intent.hedge_quantity,
        )
        late = machine.ingest(event(hedge_intent.deadline_time_ns + 1))
        assert late is not None
        self.assertEqual(attempt.observe(late), timeout)

    def test_arrival_reference_retains_packet_and_computes_nullable_slip(self) -> None:
        hedge_intent = intent()
        machine = RawBookStateMachine()
        arrival_state = machine.ingest(
            event(
                hedge_intent.trigger_cursor.recv_time_ns,
                packet=101,
                bids=(level(99.0, 5),),
            )
        )
        assert arrival_state is not None
        arrival = capture_arrival_reference(
            arrival_state,
            reference_cursor=hedge_intent.trigger_cursor,
            side=hedge_intent.side,
            quantity=hedge_intent.hedge_quantity,
            quantity_unit=hedge_intent.hedge_quantity_unit,
        )
        self.assertTrue(arrival.available)
        assert arrival.book_cursor is not None
        self.assertEqual(arrival.book_cursor.packet_sequence, 101)
        self.assertAlmostEqual(arrival.executable_vwap or 0.0, 99.0)
        self.assertAlmostEqual(
            arrival.adverse_slippage_bp(98.0) or 0.0,
            (99.0 - 98.0) / 99.0 * 10_000.0,
        )

        missing = capture_arrival_reference(
            None,
            reference_cursor=hedge_intent.trigger_cursor,
            side=hedge_intent.side,
            quantity=hedge_intent.hedge_quantity,
            quantity_unit=hedge_intent.hedge_quantity_unit,
        )
        self.assertFalse(missing.available)
        self.assertIsNone(missing.executable_vwap)
        self.assertIsNone(missing.adverse_slippage_bp(98.0))

    def test_rollback_revalidates_and_sells_initiating_2000_spot_shares(self) -> None:
        spec = timed_out_rollback()
        self.assertEqual(spec.quantity, 2_000)
        self.assertEqual(spec.quantity_unit, "spot_shares")
        attempt = RollbackAttempt(spec)
        machine = RawBookStateMachine()
        initial = machine.ingest(
            event(
                spec.trigger_cursor.recv_time_ns,
                row=1,
                packet=201,
                bids=(level(99.0, 2_500),),
                asks=(level(101.0, 2_500),),
            )
        )
        assert initial is not None
        first_cursor = EventCursor(spec.trigger_cursor.recv_time_ns, 5, 0)
        first = attempt.observe(initial, evaluation_cursor=first_cursor)
        self.assertEqual(first.status, "send_eligible")
        assert first.send_eligible is not None
        self.assertEqual(first.send_eligible.requested_quantity, 2_000)
        self.assertEqual(first.send_eligible.requested_quantity_unit, "spot_shares")

        invalid = machine.ingest(
            event(
                spec.trigger_cursor.recv_time_ns + 1,
                row=2,
                bids=(),
                asks=(level(101.0, 2_500),),
            )
        )
        assert invalid is not None
        waiting = attempt.observe(invalid)
        self.assertEqual(waiting.status, "waiting")
        with self.assertRaisesRegex(ValueError, "currently legal"):
            attempt.confirm_actual_send(invalid.book_cursor.cursor)

        legal = machine.ingest(
            event(
                spec.trigger_cursor.recv_time_ns + 2,
                row=3,
                packet=299,
                bids=(level(98.0, 2_000),),
                asks=(level(101.0, 2_500),),
            )
        )
        assert legal is not None
        eligible = attempt.observe(legal)
        assert eligible.send_eligible is not None
        sent = attempt.confirm_actual_send(eligible.send_eligible.send_eligible_cursor)
        self.assertEqual(sent.status, "rollback_sent")
        assert sent.actual_send is not None
        self.assertEqual(sent.actual_send.requested_quantity, 2_000)
        self.assertEqual(sent.actual_send.requested_quantity_unit, "spot_shares")
        self.assertEqual(sent.actual_send.book_cursor.packet_sequence, 299)
        self.assertEqual(attempt.observe(None), sent)

    def test_rollback_deadline_is_inclusive_and_timeout_cannot_recurse(self) -> None:
        spec = timed_out_rollback()
        inclusive = RollbackAttempt(spec)
        inclusive.observe(
            None,
            evaluation_cursor=EventCursor(spec.trigger_cursor.recv_time_ns, 5, 0),
        )
        machine = RawBookStateMachine()
        deadline_book = machine.ingest(
            event(
                spec.deadline_time_ns,
                row=1,
                packet=401,
                bids=(level(97.0, 2_000),),
                asks=(level(101.0, 2_000),),
            )
        )
        assert deadline_book is not None
        at_deadline = inclusive.observe(deadline_book)
        self.assertEqual(at_deadline.status, "send_eligible")
        assert at_deadline.send_eligible is not None
        sent = inclusive.confirm_actual_send(
            at_deadline.send_eligible.send_eligible_cursor
        )
        self.assertEqual(sent.status, "rollback_sent")

        timeout_attempt = RollbackAttempt(spec)
        timeout_attempt.observe(
            None,
            evaluation_cursor=EventCursor(spec.trigger_cursor.recv_time_ns, 5, 0),
        )
        deadline_cursor = EventCursor(spec.deadline_time_ns, 5, 0)
        timeout_attempt.observe(None, evaluation_cursor=deadline_cursor)
        timeout = timeout_attempt.expire(EventCursor(spec.deadline_time_ns, 6, 0))
        self.assertEqual(timeout.status, "rollback_retry_timeout")
        self.assertFalse(hasattr(timeout, "rollback"))
        self.assertEqual(timeout_attempt.observe(deadline_book), timeout)

    def test_early_expire_and_missing_t0_evaluation_are_rejected(self) -> None:
        hedge_intent = intent()
        attempt = HedgeAttempt(hedge_intent)
        with self.assertRaisesRegex(ValueError, "explicit t0"):
            attempt.expire(EventCursor(hedge_intent.deadline_time_ns, 9, 0))

        machine = RawBookStateMachine()
        late_first = machine.ingest(event(hedge_intent.target_time_ns + 1))
        assert late_first is not None
        with self.assertRaisesRegex(ValueError, "exactly at t0"):
            attempt.observe(late_first)

        machine = RawBookStateMachine()
        at_t0 = machine.ingest(event(hedge_intent.target_time_ns - 1))
        assert at_t0 is not None
        attempt.observe(
            at_t0,
            evaluation_cursor=EventCursor(hedge_intent.target_time_ns, 9, 0),
        )
        with self.assertRaisesRegex(ValueError, "deadline-inclusive"):
            attempt.expire(EventCursor(hedge_intent.deadline_time_ns, 9, 0))

    def test_best_and_l1_same_price_take_max_quantity(self) -> None:
        machine = RawBookStateMachine()
        state = machine.ingest(
            event(
                10,
                asks=(level(101.0, 2), level(102.0, 4)),
                best_ask=level(101.0, 5),
            )
        )
        assert state is not None
        self.assertEqual([row.quantity for row in state.asks], [5, 4])
        self.assertAlmostEqual(state.asks[0].price, 101.0)
        self.assertAlmostEqual(state.asks[1].price, 102.0)
        executable, reason = executable_book(
            state,
            side="buy",
            quantity=5,
            quantity_unit="future_contracts",
        )
        self.assertIsNone(reason)
        assert executable is not None
        self.assertAlmostEqual(executable.executable_vwap, 101.0)
        self.assertEqual(executable.levels_swept, 1)


if __name__ == "__main__":
    unittest.main()
