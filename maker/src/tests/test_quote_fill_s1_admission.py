from __future__ import annotations

import unittest

from ..quote_fill.capacity_ledger import CapacityLedger
from ..quote_fill.s1_admission import CapacityAdmissionPlanner


class CapacityAdmissionPlannerTest(unittest.TestCase):
    def test_shadow_reservations_enforce_global_and_product_caps(self) -> None:
        ledger = CapacityLedger(global_cap_twd=20_000, product_cap_twd=10_000)
        planner = CapacityAdmissionPlanner(ledger)
        first = planner.evaluate(
            request_id="r1",
            capacity_id="c1",
            product_id="2330",
            requested_notional_twd=7_000,
        )
        product_blocked = planner.evaluate(
            request_id="r2",
            capacity_id="c2",
            product_id="2330",
            requested_notional_twd=4_000,
        )
        second = planner.evaluate(
            request_id="r3",
            capacity_id="c3",
            product_id="2317",
            requested_notional_twd=10_000,
        )
        global_blocked = planner.evaluate(
            request_id="r4",
            capacity_id="c4",
            product_id="2454",
            requested_notional_twd=4_000,
        )
        self.assertTrue(first.admitted)
        self.assertEqual(product_blocked.status, "shadow_blocked_product_cap")
        self.assertTrue(second.admitted)
        self.assertEqual(global_blocked.status, "shadow_blocked_global_cap")

        planner.bind_assignment("r1", send_sequence=11)
        planner.bind_assignment("r3", send_sequence=12)
        committed = planner.commit(
            timestamp_ns=100,
            event_sequence=2,
            transition_prefix="cursor-100",
        )
        self.assertEqual(len(committed), 4)
        self.assertEqual(
            ledger.global_balances.total_committed_notional_twd,
            17_000,
        )
        ledger.verify()

    def test_repeated_scheduler_probe_is_idempotent(self) -> None:
        ledger = CapacityLedger(global_cap_twd=20_000, product_cap_twd=10_000)
        planner = CapacityAdmissionPlanner(ledger)
        first = planner.evaluate(
            request_id="r1",
            capacity_id="c1",
            product_id="2330",
            requested_notional_twd=5_000,
        )
        repeated = planner.evaluate(
            request_id="r1",
            capacity_id="c1",
            product_id="2330",
            requested_notional_twd=5_000,
        )
        self.assertIs(first, repeated)
        self.assertEqual(len(planner.plans), 1)
        with self.assertRaisesRegex(ValueError, "changed admission inputs"):
            planner.evaluate(
                request_id="r1",
                capacity_id="c1",
                product_id="2330",
                requested_notional_twd=5_001,
            )

    def test_admitted_plan_requires_assignment_and_ledger_must_not_drift(self) -> None:
        ledger = CapacityLedger(global_cap_twd=20_000, product_cap_twd=10_000)
        planner = CapacityAdmissionPlanner(ledger)
        planner.evaluate(
            request_id="r1",
            capacity_id="c1",
            product_id="2330",
            requested_notional_twd=5_000,
        )
        with self.assertRaisesRegex(RuntimeError, "lack venue assignments"):
            planner.commit(
                timestamp_ns=100,
                event_sequence=2,
                transition_prefix="cursor-100",
            )

        ledger.attempt_new_reservation(
            transition_id="foreign",
            timestamp_ns=99,
            capacity_id="foreign-cap",
            product_id="2317",
            requested_notional_twd=1_000,
        )
        planner.bind_assignment("r1", send_sequence=1)
        with self.assertRaisesRegex(RuntimeError, "changed before"):
            planner.commit(
                timestamp_ns=100,
                event_sequence=2,
                transition_prefix="cursor-100",
            )

    def test_blocked_plan_cannot_bind_assignment(self) -> None:
        ledger = CapacityLedger(global_cap_twd=20_000, product_cap_twd=10_000)
        ledger.attempt_new_reservation(
            transition_id="seed",
            timestamp_ns=1,
            capacity_id="seed-cap",
            product_id="2330",
            requested_notional_twd=10_000,
        )
        planner = CapacityAdmissionPlanner(ledger)
        blocked = planner.evaluate(
            request_id="r1",
            capacity_id="c1",
            product_id="2330",
            requested_notional_twd=1_000,
        )
        self.assertFalse(blocked.admitted)
        with self.assertRaisesRegex(ValueError, "blocked admission"):
            planner.bind_assignment("r1", send_sequence=2)


if __name__ == "__main__":
    unittest.main()
