"""Negative controls ensure the independent auditor catches policy regressions."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from ..backtest.unreserved_audit import UnreservedAudit
from .test_causal_replay import T, decision, row
from .test_unreserved_replay import two_products, second_row, unreserved


def facts(actor):
    positions = {}
    for p in actor.cycles.values():
        r = asdict(p)
        r.update(r.pop("contract"))
        positions[p.id] = r
    return dict(day=actor.day, positions=positions, events=deepcopy(actor.trace),
                cash=deepcopy(actor.cash), ledger=deepcopy(actor.ledger.events), observations={})


class IndependentAuditTest(unittest.TestCase):
    def partial(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 1000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        a.offer(second_row(), decision())
        replay.drain(T+60_000_000)
        return a, facts(a)

    def check(self, a, data):
        return UnreservedAudit(asdict(a.cfg)).check_day(**data)

    def test_valid_partial_booking_and_cancel_pass(self):
        a, data = self.partial()
        self.assertEqual(self.check(a, data), [])

    def test_missing_partial_booking_fails_even_if_other_rows_reconcile(self):
        a, data = self.partial()
        data["events"] = [e for e in data["events"] if e["kind"] != "capital"]
        data["ledger"] = []
        self.assertIn("unbooked_entry_execution", {f["check"] for f in self.check(a, data)})

    def test_missing_capacity_cancel_fails(self):
        a, data = self.partial()
        data["events"] = [e for e in data["events"] if e["kind"] != "cancel_request"]
        self.assertIn("missing_immediate_capacity_cancel", {f["check"] for f in self.check(a, data)})

    def test_duplicate_product_route_fails(self):
        a, data = self.partial()
        e = deepcopy(data["events"][0])
        e["order_id"] += "/duplicate"
        e["id"] += "9"
        data["events"].insert(1, e)
        checks = {f["check"] for f in self.check(a, data)}
        self.assertIn("multiple_working_route_orders", checks)
        self.assertIn("entry_before_previous_hedge_or_cancel", checks)

    def test_release_before_actual_close_fails(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 2000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        replay.drain(T+200_000_000)
        a.force_close(next(iter(a.cycles.values())), 2*T, "test_exit")
        replay.drain(2*T)
        data = facts(a)
        self.assertEqual(self.check(a, data), [])
        data["ledger"][-1]["ns"] = T
        self.assertIn("capital_release_not_at_close", {f["check"] for f in self.check(a, data)})


if __name__ == "__main__":
    unittest.main()
