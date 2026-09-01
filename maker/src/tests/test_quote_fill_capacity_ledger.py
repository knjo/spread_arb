from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from unittest.mock import patch

from maker.src.quote_fill.capacity_ledger import (
    CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
    ENTRY_PARTIAL,
    EXIT_IN_PROGRESS,
    HEDGE_PENDING,
    PAIRED_OPEN,
    WORKING_UNFILLED,
    BucketBalances,
    CapacityCodecError,
    CapacityIdentityRegistryReceipt,
    CapacityLedger,
    CapacityLedgerCompactCheckpoint,
    CapacityLedgerError,
    CapacityReplayError,
    CapacityTransitionError,
    decode_capacity_identity_registry_receipt,
    decode_capacity_ledger_compact_checkpoint,
    decode_capacity_transition,
    decode_capacity_transitions,
    encode_capacity_identity_registry_receipt,
    encode_capacity_ledger_compact_checkpoint,
    encode_capacity_transition,
    encode_capacity_transitions,
    replay_capacity_transitions,
)


class CapacityLedgerTest(unittest.TestCase):
    @staticmethod
    def _registry_receipt(
        *,
        transition_count: int,
        admitted_count: int,
        digest_character: str,
    ) -> CapacityIdentityRegistryReceipt:
        return CapacityIdentityRegistryReceipt(
            schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
            transition_count=transition_count,
            admitted_count=admitted_count,
            registry_sha256=digest_character * 64,
        )

    def test_complete_lifecycle_transfers_without_double_counting(self) -> None:
        ledger = CapacityLedger()
        decision = ledger.attempt_new_reservation(
            transition_id="t1",
            timestamp_ns=1,
            capacity_id="order-a",
            product_id="A",
            requested_notional_twd=Decimal(8_000_000),
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(ledger.global_balances.total_committed_notional_twd, 8_000_000)

        ledger.record_entry_partial_fill(
            transition_id="t2",
            timestamp_ns=2,
            capacity_id="order-a",
            filled_notional_twd=3_000_000,
        )
        ledger.release_working_leaves(
            transition_id="t3",
            timestamp_ns=3,
            capacity_id="order-a",
            reason="actual_cancel",
        )
        self.assertEqual(
            ledger.account_balances("order-a").amount(ENTRY_PARTIAL), 3_000_000
        )
        self.assertEqual(ledger.global_balances.total_committed_notional_twd, 3_000_000)

        ledger.move_entry_partial_to_hedge_pending(
            transition_id="t4",
            timestamp_ns=4,
            capacity_id="order-a",
            notional_twd=3_000_000,
        )
        ledger.complete_entry_hedge(
            transition_id="t5",
            timestamp_ns=5,
            capacity_id="order-a",
            notional_twd=3_000_000,
        )
        ledger.begin_exit(
            transition_id="t6",
            timestamp_ns=6,
            capacity_id="order-a",
            notional_twd=3_000_000,
        )
        self.assertEqual(
            ledger.account_balances("order-a").amount(EXIT_IN_PROGRESS),
            3_000_000,
        )
        ledger.complete_exit_hedge(
            transition_id="t7",
            timestamp_ns=7,
            capacity_id="order-a",
            notional_twd=3_000_000,
        )
        self.assertEqual(ledger.global_balances.total_committed_notional_twd, 0)
        replay = ledger.verify()
        self.assertEqual(replay.global_balances.total_committed_notional_twd, 0)
        self.assertEqual(len(ledger.transitions), 7)
        for row in ledger.transitions:
            self.assertLessEqual(
                row.global_after.total_committed_notional_twd, 20_000_000
            )
            self.assertLessEqual(
                row.product_after.total_committed_notional_twd, 10_000_000
            )

    def test_verify_caches_one_successful_replay_until_append(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="order",
            product_id="A",
            requested_notional_twd=100,
        )

        with patch(
            "maker.src.quote_fill.capacity_ledger.replay_capacity_transitions",
            wraps=replay_capacity_transitions,
        ) as replay:
            first = ledger.verify()
            second = ledger.verify()
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(second, first)

            blocked = ledger.attempt_new_reservation(
                transition_id="blocked",
                timestamp_ns=2,
                capacity_id="blocked-order",
                product_id="B",
                requested_notional_twd=1,
            )
            self.assertFalse(blocked.admitted)
            self.assertEqual(blocked.transition.delta, BucketBalances())

            after_append = ledger.verify()
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(after_append, ledger.verify())
            self.assertEqual(replay.call_count, 2)

    def test_verify_cache_returns_defensive_replay_copies(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="order",
            product_id="A",
            requested_notional_twd=50,
        )
        fresh = replay_capacity_transitions(ledger.transitions)

        returned = ledger.verify()
        returned.account_balances.clear()
        returned.account_products["order"] = "tampered"
        returned.product_balances["A"] = BucketBalances()

        cached = ledger.verify()
        self.assertEqual(cached, fresh)
        self.assertIsNot(cached.account_balances, returned.account_balances)
        self.assertIsNot(cached.account_products, returned.account_products)
        self.assertIsNot(cached.product_balances, returned.product_balances)

    def test_verify_cache_does_not_hide_live_state_drift(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="order",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.verify()
        ledger._global_balances = BucketBalances(working_unfilled=49)

        with patch(
            "maker.src.quote_fill.capacity_ledger.replay_capacity_transitions",
            wraps=replay_capacity_transitions,
        ) as replay:
            with self.assertRaisesRegex(
                CapacityReplayError,
                "replayed global balances differ from live state",
            ):
                ledger.verify()
            with self.assertRaises(CapacityReplayError):
                ledger.verify()
            self.assertEqual(replay.call_count, 2)

    def test_full_fill_directly_moves_working_to_hedge_pending(self) -> None:
        ledger = CapacityLedger()
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=10,
            capacity_id="order",
            product_id="A",
            requested_notional_twd=10_000_000,
        )
        row = ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=10,
            capacity_id="order",
        )
        self.assertEqual(row.from_bucket, WORKING_UNFILLED)
        self.assertEqual(row.to_bucket, HEDGE_PENDING)
        self.assertEqual(row.moved_notional_twd, 10_000_000)
        self.assertEqual(row.account_after.working_unfilled, 0)
        self.assertEqual(row.account_after.hedge_pending, 10_000_000)
        self.assertEqual(row.global_before.total_committed_notional_twd, 10_000_000)
        self.assertEqual(row.global_after.total_committed_notional_twd, 10_000_000)

    def test_global_product_and_both_cap_blocks_are_append_only_noops(self) -> None:
        ledger = CapacityLedger()
        ledger.attempt_new_reservation(
            transition_id="a",
            timestamp_ns=1,
            capacity_id="a",
            product_id="A",
            requested_notional_twd=10_000_000,
        )
        product = ledger.attempt_new_reservation(
            transition_id="product",
            timestamp_ns=2,
            capacity_id="b",
            product_id="A",
            requested_notional_twd=1,
        )
        self.assertEqual(product.status, "blocked_product_cap")
        ledger.attempt_new_reservation(
            transition_id="c",
            timestamp_ns=3,
            capacity_id="c",
            product_id="B",
            requested_notional_twd=10_000_000,
        )
        global_only = ledger.attempt_new_reservation(
            transition_id="global",
            timestamp_ns=4,
            capacity_id="d",
            product_id="C",
            requested_notional_twd=1,
        )
        both = ledger.attempt_new_reservation(
            transition_id="both",
            timestamp_ns=5,
            capacity_id="e",
            product_id="A",
            requested_notional_twd=1,
        )
        self.assertEqual(global_only.status, "blocked_global_cap")
        self.assertEqual(both.status, "blocked_both_caps")
        self.assertEqual(
            ledger.global_balances.total_committed_notional_twd, 20_000_000
        )
        for decision in (product, global_only, both):
            self.assertEqual(decision.transition.delta.total_committed_notional_twd, 0)
            self.assertEqual(
                decision.transition.global_before, decision.transition.global_after
            )
        ledger.verify()

    def test_blocked_capacity_id_can_retry_after_release(self) -> None:
        ledger = CapacityLedger(global_cap_twd=10, product_cap_twd=10)
        ledger.attempt_new_reservation(
            transition_id="a1",
            timestamp_ns=1,
            capacity_id="a",
            product_id="A",
            requested_notional_twd=10,
        )
        blocked = ledger.attempt_new_reservation(
            transition_id="b1",
            timestamp_ns=2,
            capacity_id="b",
            product_id="B",
            requested_notional_twd=10,
        )
        self.assertFalse(blocked.admitted)
        ledger.release_working_leaves(
            transition_id="a2",
            timestamp_ns=3,
            capacity_id="a",
            reason="session_expiry",
        )
        retry = ledger.attempt_new_reservation(
            transition_id="b2",
            timestamp_ns=4,
            capacity_id="b",
            product_id="B",
            requested_notional_twd=10,
        )
        self.assertTrue(retry.admitted)
        ledger.verify()

    def test_duplicate_release_overfill_and_duplicate_identity_are_rejected(
        self,
    ) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="order",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.release_working_leaves(
            transition_id="cancel",
            timestamp_ns=2,
            capacity_id="order",
            reason="actual_cancel",
        )
        with self.assertRaises(CapacityTransitionError):
            ledger.release_working_leaves(
                transition_id="cancel-again",
                timestamp_ns=3,
                capacity_id="order",
                reason="actual_cancel",
            )
        with self.assertRaises(CapacityTransitionError):
            ledger.attempt_new_reservation(
                transition_id="reserve-again",
                timestamp_ns=3,
                capacity_id="order",
                product_id="A",
                requested_notional_twd=1,
            )

        other = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        other.attempt_new_reservation(
            transition_id="r",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        with self.assertRaises(CapacityTransitionError):
            other.record_entry_partial_fill(
                transition_id="overfill",
                timestamp_ns=2,
                capacity_id="x",
                filled_notional_twd=51,
            )

    def test_noninteger_money_bad_time_and_out_of_order_time_are_rejected(self) -> None:
        with self.assertRaises(CapacityLedgerError):
            CapacityLedger(global_cap_twd=20_000_000.0)
        with self.assertRaises(CapacityLedgerError):
            CapacityLedger(global_cap_twd=Decimal("20.1"))
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        with self.assertRaises(CapacityLedgerError):
            ledger.attempt_new_reservation(
                transition_id="float",
                timestamp_ns=1,
                capacity_id="x",
                product_id="A",
                requested_notional_twd=10.0,
            )
        ledger.attempt_new_reservation(
            transition_id="ok",
            timestamp_ns=2,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=10,
        )
        with self.assertRaises(CapacityTransitionError):
            ledger.record_entry_full_fill(
                transition_id="past",
                timestamp_ns=1,
                capacity_id="x",
            )

    def test_replay_detects_tampered_before_after_caps_and_sequence(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=60)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="x",
        )
        rows = ledger.transitions
        replay_capacity_transitions(rows)

        bad_before = replace(rows[1], account_before=rows[0].account_before)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((rows[0], bad_before))
        bad_delta = replace(rows[1], delta=rows[0].delta)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((rows[0], bad_delta))
        bad_cap = replace(rows[0], global_cap_twd=40)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((bad_cap,))
        bad_sequence = replace(rows[0], sequence=2)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((bad_sequence,))
        bad_money_type = replace(
            rows[0],
            delta=BucketBalances(working_unfilled=50.0),  # type: ignore[arg-type]
        )
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((bad_money_type,))
        bad_event = replace(rows[1], event_type="exit_started")
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((rows[0], bad_event))

        blocked = CapacityLedger(global_cap_twd=100, product_cap_twd=60)
        decision = blocked.attempt_new_reservation(
            transition_id="fits",
            timestamp_ns=1,
            capacity_id="fits",
            product_id="A",
            requested_notional_twd=50,
        )
        false_block = replace(
            decision.transition,
            admitted=False,
            status="blocked_product_cap",
            moved_notional_twd=0,
            delta=decision.transition.account_before,
            account_after=decision.transition.account_before,
            product_after=decision.transition.product_before,
            global_after=decision.transition.global_before,
        )
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((false_block,))

    def test_flat_row_contains_all_replay_fields(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=50)
        decision = ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        row = decision.transition.as_dict()
        self.assertEqual(row["delta_working_unfilled_twd"], 50)
        self.assertEqual(row["account_after_total_committed_notional_twd"], 50)
        self.assertEqual(row["product_after_total_committed_notional_twd"], 50)
        self.assertEqual(row["global_after_total_committed_notional_twd"], 50)

    def test_same_timestamp_uses_event_sequence_and_row_index_order(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=100,
            event_sequence=5,
            row_index=7,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=100,
            event_sequence=6,
            row_index=0,
            capacity_id="x",
        )
        with self.assertRaises(CapacityTransitionError):
            ledger.record_entry_rollback_failure(
                transition_id="out-of-phase",
                timestamp_ns=100,
                event_sequence=5,
                row_index=8,
                capacity_id="x",
                notional_twd=50,
            )
        failure = ledger.record_entry_rollback_failure(
            transition_id="failure",
            timestamp_ns=100,
            event_sequence=6,
            row_index=1,
            capacity_id="x",
            notional_twd=50,
        )
        self.assertEqual(
            (failure.timestamp_ns, failure.event_sequence, failure.row_index),
            (100, 6, 1),
        )
        replay = ledger.verify()
        self.assertEqual(replay.last_timestamp_ns, 100)
        self.assertEqual(replay.last_event_sequence, 6)
        self.assertEqual(replay.last_row_index, 1)
        flat = failure.as_dict()
        self.assertEqual(flat["event_sequence"], 6)
        self.assertEqual(flat["row_index"], 1)

    def test_replay_detects_cursor_tampering(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=100,
            event_sequence=5,
            row_index=7,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=100,
            event_sequence=6,
            row_index=0,
            capacity_id="x",
        )
        rows = ledger.transitions
        bad_phase = replace(rows[1], event_sequence=4)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((rows[0], bad_phase))
        bad_row = replace(rows[1], event_sequence=5, row_index=6)
        with self.assertRaises(CapacityReplayError):
            replay_capacity_transitions((rows[0], bad_row))

    def test_entry_rollback_success_releases_and_failure_does_not(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill", timestamp_ns=2, capacity_id="x"
        )
        failed = ledger.record_entry_rollback_failure(
            transition_id="rollback-failed",
            timestamp_ns=3,
            capacity_id="x",
            notional_twd=50,
        )
        self.assertEqual(failed.global_before, failed.global_after)
        self.assertEqual(ledger.account_balances("x").hedge_pending, 50)
        completed = ledger.complete_entry_rollback(
            transition_id="rollback-complete",
            timestamp_ns=4,
            capacity_id="x",
            notional_twd=50,
        )
        self.assertEqual(completed.from_bucket, HEDGE_PENDING)
        self.assertIsNone(completed.to_bucket)
        self.assertEqual(ledger.global_balances.total_committed_notional_twd, 0)
        ledger.verify()

    def test_exit_rollback_success_restores_paired_and_failure_does_not(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill", timestamp_ns=2, capacity_id="x"
        )
        ledger.complete_entry_hedge(
            transition_id="entry-hedge",
            timestamp_ns=3,
            capacity_id="x",
            notional_twd=50,
        )
        ledger.begin_exit(
            transition_id="exit", timestamp_ns=4, capacity_id="x", notional_twd=50
        )
        failed = ledger.record_exit_rollback_failure(
            transition_id="rollback-failed",
            timestamp_ns=5,
            capacity_id="x",
            notional_twd=50,
        )
        self.assertEqual(failed.global_before, failed.global_after)
        self.assertEqual(ledger.account_balances("x").exit_in_progress, 50)
        restored = ledger.complete_exit_rollback(
            transition_id="rollback-complete",
            timestamp_ns=6,
            capacity_id="x",
            notional_twd=50,
        )
        self.assertEqual(restored.from_bucket, EXIT_IN_PROGRESS)
        self.assertEqual(restored.to_bucket, PAIRED_OPEN)
        self.assertEqual(ledger.account_balances("x").paired_open, 50)
        self.assertEqual(ledger.global_balances.total_committed_notional_twd, 50)
        ledger.verify()

    def test_checkpoint_restart_matches_uninterrupted_path_and_daily_delta(
        self,
    ) -> None:
        uninterrupted = CapacityLedger(global_cap_twd=150, product_cap_twd=100)
        uninterrupted.attempt_new_reservation(
            transition_id="d1-reserve",
            timestamp_ns=100,
            event_sequence=1,
            capacity_id="carry-a",
            product_id="A",
            requested_notional_twd=80,
        )
        uninterrupted.record_entry_full_fill(
            transition_id="d1-fill",
            timestamp_ns=101,
            capacity_id="carry-a",
        )
        uninterrupted.complete_entry_hedge(
            transition_id="d1-hedge",
            timestamp_ns=102,
            capacity_id="carry-a",
            notional_twd=80,
        )
        blocked = uninterrupted.attempt_new_reservation(
            transition_id="d1-blocked",
            timestamp_ns=103,
            capacity_id="blocked-b",
            product_id="B",
            requested_notional_twd=100,
        )
        self.assertEqual(blocked.status, "blocked_global_cap")

        day_one_count = len(uninterrupted.transitions)
        seed = uninterrupted.to_seed("20260505")
        self.assertEqual(seed.transition_sequence_offset, day_one_count)
        self.assertEqual(seed.global_balances, uninterrupted.global_balances)
        self.assertEqual(seed.product_balances, {"A": BucketBalances(paired_open=80)})
        self.assertEqual(len(seed.seen_transition_ids), day_one_count)
        with self.assertRaises(FrozenInstanceError):
            seed.through_date = "20260506"  # type: ignore[misc]

        restarted = CapacityLedger.from_seed(seed)
        self.assertEqual(restarted.transitions, ())
        self.assertEqual(restarted.account_balances("carry-a").paired_open, 80)
        self.assertEqual(restarted.global_balances, uninterrupted.global_balances)

        for ledger in (uninterrupted, restarted):
            ledger.begin_exit(
                transition_id="d2-exit",
                timestamp_ns=200,
                capacity_id="carry-a",
                notional_twd=80,
            )
            ledger.complete_exit_hedge(
                transition_id="d2-exit-hedge",
                timestamp_ns=201,
                capacity_id="carry-a",
                notional_twd=80,
            )
            admitted = ledger.attempt_new_reservation(
                transition_id="d2-reserve",
                timestamp_ns=202,
                capacity_id="new-b",
                product_id="B",
                requested_notional_twd=100,
            )
            self.assertTrue(admitted.admitted)

        expected_delta = uninterrupted.transitions[day_one_count:]
        self.assertEqual(restarted.transitions, expected_delta)
        self.assertEqual([row.sequence for row in restarted.transitions], [5, 6, 7])
        self.assertEqual(
            restarted.to_seed("20260506"),
            uninterrupted.to_seed("20260506"),
        )
        replay = restarted.verify()
        self.assertEqual(replay.last_sequence, 7)
        self.assertEqual(replay.global_balances.total_committed_notional_twd, 100)

    def test_restore_rejects_prior_identity_and_requires_cursor_advance(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=100,
            event_sequence=5,
            row_index=7,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        seed = ledger.to_seed("20260505")
        restarted = CapacityLedger.from_seed(seed)

        with self.assertRaisesRegex(CapacityTransitionError, "duplicate transition_id"):
            restarted.record_entry_full_fill(
                transition_id="reserve",
                timestamp_ns=101,
                capacity_id="x",
            )
        with self.assertRaisesRegex(CapacityTransitionError, "strictly advance"):
            restarted.record_entry_full_fill(
                transition_id="same-cursor",
                timestamp_ns=100,
                event_sequence=5,
                row_index=7,
                capacity_id="x",
            )
        with self.assertRaisesRegex(CapacityTransitionError, "strictly advance"):
            restarted.record_entry_full_fill(
                transition_id="past-cursor",
                timestamp_ns=100,
                event_sequence=5,
                row_index=6,
                capacity_id="x",
            )

        row = restarted.record_entry_full_fill(
            transition_id="next",
            timestamp_ns=100,
            event_sequence=5,
            row_index=8,
            capacity_id="x",
        )
        self.assertEqual(row.sequence, 2)
        self.assertEqual(
            (row.timestamp_ns, row.event_sequence, row.row_index),
            (100, 5, 8),
        )
        self.assertIn("reserve", restarted.to_seed("20260506").seen_transition_ids)
        restarted.verify()

    def test_checkpoint_detects_corruption_and_date_regression(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        seed = ledger.to_seed("20260505")

        changed_account = replace(
            seed.accounts[0],
            balances=BucketBalances(working_unfilled=49),
        )
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_seed(replace(seed, accounts=(changed_account,)))
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_seed(replace(seed, transition_chain_sha256="0" * 64))
        with self.assertRaisesRegex(CapacityReplayError, "unique"):
            CapacityLedger.from_seed(
                replace(seed, seen_transition_ids=("reserve", "reserve"))
            )
        with self.assertRaisesRegex(CapacityReplayError, "offset"):
            CapacityLedger.from_seed(replace(seed, transition_sequence_offset=2))
        with self.assertRaisesRegex(CapacityReplayError, "checkpoint digest"):
            CapacityLedger.from_seed(replace(seed, checkpoint_sha256="f" * 64))

        restarted = CapacityLedger.from_seed(seed)
        with self.assertRaisesRegex(CapacityLedgerError, "cannot regress"):
            restarted.to_seed("20260504")
        with self.assertRaisesRegex(CapacityLedgerError, "valid YYYYMMDD"):
            ledger.to_seed("20260230")

    def test_compact_checkpoint_two_partitions_matches_uninterrupted_path(
        self,
    ) -> None:
        uninterrupted = CapacityLedger(global_cap_twd=150, product_cap_twd=100)
        uninterrupted.attempt_new_reservation(
            transition_id="d1-carry-reserve",
            timestamp_ns=100,
            capacity_id="carry-a",
            product_id="A",
            requested_notional_twd=80,
        )
        uninterrupted.record_entry_full_fill(
            transition_id="d1-carry-fill",
            timestamp_ns=101,
            capacity_id="carry-a",
        )
        uninterrupted.complete_entry_hedge(
            transition_id="d1-carry-hedge",
            timestamp_ns=102,
            capacity_id="carry-a",
            notional_twd=80,
        )
        uninterrupted.attempt_new_reservation(
            transition_id="d1-closed-reserve",
            timestamp_ns=103,
            capacity_id="closed-b",
            product_id="B",
            requested_notional_twd=50,
        )
        uninterrupted.release_working_leaves(
            transition_id="d1-closed-release",
            timestamp_ns=104,
            capacity_id="closed-b",
            reason="actual_cancel",
        )

        day_one_receipt = self._registry_receipt(
            transition_count=5,
            admitted_count=2,
            digest_character="1",
        )
        day_one_checkpoint = uninterrupted.to_compact_checkpoint(
            "20260505",
            identity_registry_receipt=day_one_receipt,
        )
        self.assertIsInstance(
            day_one_checkpoint,
            CapacityLedgerCompactCheckpoint,
        )
        self.assertEqual(day_one_checkpoint.global_cap_twd, 150)
        self.assertEqual(day_one_checkpoint.product_cap_twd, 100)
        self.assertEqual(
            day_one_checkpoint.identity_registry_receipt.transition_count,
            day_one_checkpoint.transition_sequence_offset,
        )
        self.assertEqual(
            tuple(account.capacity_id for account in day_one_checkpoint.live_accounts),
            ("carry-a",),
        )
        self.assertNotIn(
            "closed-b",
            day_one_checkpoint.account_balances,
        )

        restarted = CapacityLedger.from_compact_checkpoint(day_one_checkpoint)
        self.assertEqual(restarted.transitions, ())
        self.assertEqual(restarted.account_balances("carry-a").paired_open, 80)
        self.assertEqual(restarted.account_balances("closed-b"), BucketBalances())

        for ledger in (uninterrupted, restarted):
            ledger.begin_exit(
                transition_id="d2-carry-exit",
                timestamp_ns=200,
                capacity_id="carry-a",
                notional_twd=80,
            )
            ledger.complete_exit_hedge(
                transition_id="d2-carry-close",
                timestamp_ns=201,
                capacity_id="carry-a",
                notional_twd=80,
            )
            ledger.attempt_new_reservation(
                transition_id="d2-new-reserve",
                timestamp_ns=202,
                capacity_id="carry-c",
                product_id="C",
                requested_notional_twd=60,
            )
            ledger.record_entry_full_fill(
                transition_id="d2-new-fill",
                timestamp_ns=203,
                capacity_id="carry-c",
            )
            ledger.complete_entry_hedge(
                transition_id="d2-new-hedge",
                timestamp_ns=204,
                capacity_id="carry-c",
                notional_twd=60,
            )

        day_two_receipt = self._registry_receipt(
            transition_count=10,
            admitted_count=3,
            digest_character="2",
        )
        restarted_checkpoint = restarted.to_compact_checkpoint(
            "20260506",
            identity_registry_receipt=day_two_receipt,
        )
        uninterrupted_checkpoint = uninterrupted.to_compact_checkpoint(
            "20260506",
            identity_registry_receipt=day_two_receipt,
        )
        self.assertEqual(restarted_checkpoint, uninterrupted_checkpoint)
        self.assertEqual(
            tuple(
                account.capacity_id for account in restarted_checkpoint.live_accounts
            ),
            ("carry-c",),
        )
        restarted_replay = restarted.verify()
        uninterrupted_replay = uninterrupted.verify()
        self.assertEqual(restarted_replay.last_sequence, 10)
        self.assertEqual(uninterrupted_replay.last_sequence, 10)
        self.assertEqual(
            restarted_replay.transition_chain_sha256,
            uninterrupted_replay.transition_chain_sha256,
        )
        self.assertEqual(
            restarted_replay.global_balances,
            uninterrupted_replay.global_balances,
        )
        self.assertEqual(
            restarted_replay.seen_transition_ids,
            tuple(sorted(row.transition_id for row in restarted.transitions)),
        )
        self.assertEqual(len(restarted_replay.seen_transition_ids), 5)

    def test_compact_checkpoint_rejects_tampered_receipt_and_live_state(
        self,
    ) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="x",
        )
        ledger.complete_entry_hedge(
            transition_id="hedge",
            timestamp_ns=3,
            capacity_id="x",
            notional_twd=50,
        )
        receipt = self._registry_receipt(
            transition_count=3,
            admitted_count=1,
            digest_character="a",
        )
        checkpoint = ledger.to_compact_checkpoint(
            "20260505",
            identity_registry_receipt=receipt,
        )

        with self.assertRaisesRegex(CapacityReplayError, "transition count"):
            ledger.to_compact_checkpoint(
                "20260505",
                identity_registry_receipt=replace(receipt, transition_count=2),
            )
        with self.assertRaisesRegex(CapacityReplayError, "admitted count"):
            ledger.to_compact_checkpoint(
                "20260505",
                identity_registry_receipt=replace(receipt, admitted_count=2),
            )

        with self.assertRaisesRegex(CapacityReplayError, "receipt schema"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    identity_registry_receipt=replace(receipt, schema_version=2),
                )
            )
        with self.assertRaisesRegex(CapacityReplayError, "transition count"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    identity_registry_receipt=replace(receipt, transition_count=2),
                )
            )
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    identity_registry_receipt=replace(receipt, admitted_count=2),
                )
            )
        with self.assertRaisesRegex(CapacityReplayError, "SHA-256"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    identity_registry_receipt=replace(
                        receipt,
                        registry_sha256="z" * 64,
                    ),
                )
            )
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    identity_registry_receipt=replace(
                        receipt,
                        registry_sha256="b" * 64,
                    ),
                )
            )

        changed_account = replace(
            checkpoint.live_accounts[0],
            balances=BucketBalances(paired_open=49),
        )
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_compact_checkpoint(
                replace(checkpoint, live_accounts=(changed_account,))
            )
        with self.assertRaisesRegex(CapacityReplayError, "omit closed"):
            CapacityLedger.from_compact_checkpoint(
                replace(
                    checkpoint,
                    live_accounts=(
                        replace(
                            checkpoint.live_accounts[0],
                            balances=BucketBalances(),
                        ),
                    ),
                )
            )
        with self.assertRaisesRegex(CapacityReplayError, "digest mismatch"):
            CapacityLedger.from_compact_checkpoint(
                replace(checkpoint, checkpoint_sha256="f" * 64)
            )

    def test_compact_restart_rejects_local_duplicate_transition_id(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="prior-reserve",
            timestamp_ns=1,
            capacity_id="x",
            product_id="A",
            requested_notional_twd=50,
        )
        checkpoint = ledger.to_compact_checkpoint(
            "20260505",
            identity_registry_receipt=self._registry_receipt(
                transition_count=1,
                admitted_count=1,
                digest_character="c",
            ),
        )
        restarted = CapacityLedger.from_compact_checkpoint(checkpoint)
        restarted.record_entry_full_fill(
            transition_id="local-fill",
            timestamp_ns=2,
            capacity_id="x",
        )
        with self.assertRaisesRegex(CapacityTransitionError, "duplicate transition_id"):
            restarted.complete_entry_hedge(
                transition_id="local-fill",
                timestamp_ns=3,
                capacity_id="x",
                notional_twd=50,
            )
        duplicate_row = replace(
            restarted.transitions[0],
            sequence=3,
            timestamp_ns=3,
        )
        with self.assertRaisesRegex(CapacityReplayError, "unique"):
            replay_capacity_transitions(
                (*restarted.transitions, duplicate_row),
                compact_checkpoint=checkpoint,
            )

    def test_capacity_transition_json_codec_round_trip_is_exact(self) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            event_sequence=2,
            row_index=3,
            capacity_id="position-a",
            product_id="1101",
            requested_notional_twd=80,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="position-a",
        )
        ledger.complete_entry_hedge(
            transition_id="hedge",
            timestamp_ns=3,
            capacity_id="position-a",
            notional_twd=80,
        )

        records = encode_capacity_transitions(ledger.transitions)
        json_records = json.loads(json.dumps(records, allow_nan=False))
        decoded = decode_capacity_transitions(json_records)
        self.assertEqual(decoded, ledger.transitions)
        self.assertEqual(encode_capacity_transitions(decoded), records)
        self.assertEqual(
            replay_capacity_transitions(decoded).transition_chain_sha256,
            ledger.verify().transition_chain_sha256,
        )
        for transition, record in zip(
            ledger.transitions,
            json_records,
            strict=True,
        ):
            self.assertEqual(decode_capacity_transition(record), transition)
            self.assertEqual(
                encode_capacity_transition(decode_capacity_transition(record)),
                record,
            )

    def test_capacity_transition_json_codec_rejects_schema_and_type_drift(
        self,
    ) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        transition = ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="position-a",
            product_id="1101",
            requested_notional_twd=80,
        ).transition
        original = encode_capacity_transition(transition)

        missing = dict(original)
        missing.pop("status")
        with self.assertRaisesRegex(CapacityCodecError, "schema mismatch"):
            decode_capacity_transition(missing)

        extra = dict(original)
        extra["unexpected"] = 1
        with self.assertRaisesRegex(CapacityCodecError, "schema mismatch"):
            decode_capacity_transition(extra)

        for field_name, invalid_value in (
            ("schema_version", True),
            ("sequence", True),
            ("timestamp_ns", float("nan")),
            ("admitted", 1),
            ("from_bucket", "not_a_bucket"),
        ):
            malformed = dict(original)
            malformed[field_name] = invalid_value
            with (
                self.subTest(field_name=field_name),
                self.assertRaises(CapacityCodecError),
            ):
                decode_capacity_transition(malformed)

        malformed_delta = dict(original)
        malformed_delta["delta"] = [0, 0, 0, 0, 0]
        with self.assertRaisesRegex(CapacityCodecError, "JSON object"):
            decode_capacity_transition(malformed_delta)

        bad_status = dict(original)
        bad_status["status"] = "admitted_by_guess"
        with self.assertRaises(CapacityCodecError):
            decode_capacity_transition(bad_status)

        with self.assertRaisesRegex(CapacityCodecError, "JSON array"):
            decode_capacity_transitions((original,))
        with self.assertRaises(CapacityCodecError):
            encode_capacity_transition(replace(transition, sequence=True))

    def test_registry_receipt_json_codec_is_strict_and_exact(self) -> None:
        receipt = self._registry_receipt(
            transition_count=3,
            admitted_count=1,
            digest_character="a",
        )
        record = json.loads(
            json.dumps(
                encode_capacity_identity_registry_receipt(receipt),
                allow_nan=False,
            )
        )
        self.assertEqual(decode_capacity_identity_registry_receipt(record), receipt)
        self.assertEqual(
            encode_capacity_identity_registry_receipt(
                decode_capacity_identity_registry_receipt(record)
            ),
            record,
        )

        for field_name, invalid_value in (
            ("record_type", "capacity_transition"),
            ("schema_version", True),
            ("transition_count", True),
            ("admitted_count", float("inf")),
        ):
            malformed = dict(record)
            malformed[field_name] = invalid_value
            with (
                self.subTest(field_name=field_name),
                self.assertRaises(CapacityCodecError),
            ):
                decode_capacity_identity_registry_receipt(malformed)

        extra = dict(record)
        extra["unexpected"] = None
        with self.assertRaisesRegex(CapacityCodecError, "schema mismatch"):
            decode_capacity_identity_registry_receipt(extra)

    def test_compact_checkpoint_json_codec_round_trip_and_tamper_checks(
        self,
    ) -> None:
        ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            event_sequence=2,
            row_index=3,
            capacity_id="position-a",
            product_id="1101",
            requested_notional_twd=80,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="position-a",
        )
        ledger.complete_entry_hedge(
            transition_id="hedge",
            timestamp_ns=3,
            capacity_id="position-a",
            notional_twd=80,
        )
        checkpoint = ledger.to_compact_checkpoint(
            "20260505",
            identity_registry_receipt=self._registry_receipt(
                transition_count=3,
                admitted_count=1,
                digest_character="b",
            ),
        )
        record = json.loads(
            json.dumps(
                encode_capacity_ledger_compact_checkpoint(checkpoint),
                allow_nan=False,
            )
        )
        decoded = decode_capacity_ledger_compact_checkpoint(record)
        self.assertEqual(decoded, checkpoint)
        self.assertEqual(encode_capacity_ledger_compact_checkpoint(decoded), record)
        CapacityLedger.from_compact_checkpoint(decoded).verify()

        empty = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
        empty_checkpoint = empty.to_compact_checkpoint(
            "20260505",
            identity_registry_receipt=self._registry_receipt(
                transition_count=0,
                admitted_count=0,
                digest_character="c",
            ),
        )
        empty_record = json.loads(
            json.dumps(
                encode_capacity_ledger_compact_checkpoint(empty_checkpoint),
                allow_nan=False,
            )
        )
        self.assertIsNone(empty_record["last_cursor"])
        self.assertEqual(
            decode_capacity_ledger_compact_checkpoint(empty_record),
            empty_checkpoint,
        )

        tuple_accounts = dict(record)
        tuple_accounts["live_accounts"] = tuple(record["live_accounts"])
        with self.assertRaisesRegex(CapacityCodecError, "JSON array"):
            decode_capacity_ledger_compact_checkpoint(tuple_accounts)

        malformed_cursor = dict(record)
        malformed_cursor["last_cursor"] = [3, 0, 0]
        with self.assertRaisesRegex(CapacityCodecError, "JSON object"):
            decode_capacity_ledger_compact_checkpoint(malformed_cursor)

        bool_cap = dict(record)
        bool_cap["global_cap_twd"] = True
        with self.assertRaises(CapacityCodecError):
            decode_capacity_ledger_compact_checkpoint(bool_cap)

        nested_extra = json.loads(json.dumps(record))
        nested_extra["live_accounts"][0]["balances"]["unexpected"] = 0
        with self.assertRaisesRegex(CapacityCodecError, "schema mismatch"):
            decode_capacity_ledger_compact_checkpoint(nested_extra)

        digest_tamper = json.loads(json.dumps(record))
        digest_tamper["live_accounts"][0]["balances"]["paired_open"] = 79
        with self.assertRaisesRegex(CapacityCodecError, "digest mismatch"):
            decode_capacity_ledger_compact_checkpoint(digest_tamper)

        receipt_missing = json.loads(json.dumps(record))
        receipt_missing["identity_registry_receipt"].pop("registry_sha256")
        with self.assertRaisesRegex(CapacityCodecError, "schema mismatch"):
            decode_capacity_ledger_compact_checkpoint(receipt_missing)

    def test_expiry_basis_zero_releases_only_exactly_paired_residual(self) -> None:
        ledger = CapacityLedger(global_cap_twd=1_000, product_cap_twd=1_000)
        ledger.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="paired",
            product_id="1101",
            requested_notional_twd=800,
        )
        ledger.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="paired",
        )
        ledger.complete_entry_hedge(
            transition_id="hedge",
            timestamp_ns=3,
            capacity_id="paired",
            notional_twd=800,
        )
        released = ledger.complete_expiry_basis_zero(
            transition_id="expiry",
            timestamp_ns=4,
            capacity_id="paired",
        )
        self.assertEqual(released.from_bucket, PAIRED_OPEN)
        self.assertIsNone(released.to_bucket)
        self.assertEqual(released.moved_notional_twd, 800)
        self.assertEqual(
            ledger.account_balances("paired").total_committed_notional_twd,
            0,
        )
        ledger.verify()

        in_exit = CapacityLedger(global_cap_twd=1_000, product_cap_twd=1_000)
        in_exit.attempt_new_reservation(
            transition_id="reserve",
            timestamp_ns=1,
            capacity_id="naked",
            product_id="1101",
            requested_notional_twd=800,
        )
        in_exit.record_entry_full_fill(
            transition_id="fill",
            timestamp_ns=2,
            capacity_id="naked",
        )
        in_exit.complete_entry_hedge(
            transition_id="hedge",
            timestamp_ns=3,
            capacity_id="naked",
            notional_twd=800,
        )
        in_exit.begin_exit(
            transition_id="exit",
            timestamp_ns=4,
            capacity_id="naked",
            notional_twd=800,
        )
        with self.assertRaisesRegex(
            CapacityTransitionError,
            "paired-open capacity",
        ):
            in_exit.complete_expiry_basis_zero(
                transition_id="bad-expiry",
                timestamp_ns=5,
                capacity_id="naked",
            )


if __name__ == "__main__":
    unittest.main()
