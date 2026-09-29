"""Actual fills occupy capital; unfilled quotes only pass an admission check."""
from dataclasses import replace
import unittest

from maker.src.ev_lookup_cost.execution import Book
from .test_causal_replay import C, T, DELAY, FakeMarket, decision, engine, row


def two_products(trades=(), **kwargs):
    m = FakeMarket(trades, **kwargs)
    c = replace(C, vc="2317", qc="OTHER")
    m.contracts[c.vc] = c
    m.timelines[c.qc] = m.timelines[C.qc]
    m.limits[c.vc] = m.limits[C.vc]
    for a, old, new in (("S:", C.vc, c.vc), ("F:", C.qc, c.qc)):
        m.b[a + new] = replace(m.b[a + old], instrument=a + new)
        m.marks[a + new] = m.marks[a + old]
    return m


def second_row(ns=T):
    return {**row(ns=ns), "vc": "2317", "qc": "OTHER"}


def unreserved(m, **cfg):
    return engine(m, reserve_on_submit=False, product_cap_frac=1.0, **cfg)


class UnreservedCapitalTest(unittest.TestCase):
    def test_unfilled_quotes_can_sum_above_free_capital(self):
        a, _ = unreserved(two_products(), cap_twd=101000)
        self.assertEqual(a.offer(row(), decision()), "submitted")
        self.assertEqual(a.offer(second_row(), decision()), "submitted")
        self.assertEqual(a.ledger.committed_cents, 0)
        self.assertEqual(a.ledger.events, [])

    def test_partial_fill_books_immediately_and_cancels_larger_pending_order(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 1000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        a.offer(second_row(), decision())
        replay.drain(T+60_000_000)
        self.assertEqual(a.ledger.committed_cents, 5050000)
        orders = {a.cycles[o["pid"]].contract.vc: o for o in a.orders.values()}
        self.assertIsNone(orders[C.vc]["cancel_at"])
        self.assertEqual(orders["2317"]["cancel_at"], T+60_000_000)
        self.assertEqual(a.offer(second_row(T+60_000_001), decision()), "busy")
        replay.drain(T+110_000_000)
        self.assertTrue(orders["2317"]["done"])
        self.assertEqual(a.offer(second_row(T+110_000_001), decision()), "cap")

    def test_capacity_cancel_prevents_fills_after_cancel_arrival(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 2000),
                          ("S:2317", 2*T, 505000, 2000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        a.offer(second_row(), decision())
        replay.drain(2*T)
        self.assertEqual(a.ledger.committed_cents, 10100000)
        self.assertEqual(sum(p.maker_qty for p in a.cycles.values()), 2000)

    def test_cancel_latency_race_retains_both_fills_and_overshoot(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 2000),
                          ("S:2317", T+80_000_000, 505000, 2000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        a.offer(second_row(), decision())
        replay.drain(T+200_000_000)
        self.assertEqual(a.ledger.committed_cents, 20200000)
        self.assertEqual(a.stats["cancel_race_fills"], 1)
        self.assertEqual(a.stats["capital_excess_fills"], 1)
        self.assertEqual(sum(p.state == "paired" for p in a.cycles.values()), 2)
        self.assertEqual(a.offer(row(ns=2*T), decision()), "product_cap")

    def test_partial_rollback_releases_only_at_actual_flat_fill(self):
        m = FakeMarket([("S:2330", T+60_000_000, 505000, 1000)], guard=2*T)
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        replay.drain(2*T+DELAY)
        self.assertEqual(a.ledger.committed_cents, 5050000)
        self.assertEqual(next(iter(a.cycles.values())).state, "entry_rollback")
        replay.drain(2*T+2*DELAY)
        self.assertEqual(a.ledger.committed_cents, 0)
        self.assertEqual(a.ledger.events[-1]["ns"], 2*T+2*DELAY)

    def test_future_scheduled_exit_does_not_fund_new_entry_now(self):
        m = two_products([("S:2330", T+60_000_000, 505000, 2000)])
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        replay.drain(T+200_000_000)
        p = next(iter(a.cycles.values()))
        a.force_close(p, 3*T, "test_exit")
        self.assertEqual(a.offer(second_row(2*T), decision()), "cap")
        replay.drain(3*T-1)
        self.assertEqual(a.ledger.committed_cents, 10100000)
        replay.drain(3*T)
        self.assertEqual(a.ledger.committed_cents, 0)
        self.assertEqual(a.offer(second_row(3*T+1), decision()), "submitted")

    def test_s2_uses_current_ask_then_actual_hedge_cash(self):
        m = FakeMarket([("F:FUT6", T+60_000_000, 525000, 1)])
        a, replay = unreserved(m, cap_twd=103000)
        self.assertEqual(a.offer(row("S2"), decision()), "submitted")
        self.assertEqual(a.ledger.committed_cents, 0)
        replay.drain(T+60_000_000)
        self.assertEqual(a.ledger.committed_cents, 10200000)
        self.assertEqual(a.ledger.events[-1]["estimate_ask"], 510000)
        m.b["S:2330"] = Book("S:2330", T+70_000_000, 2, ((500000, 10000),), ((520000, 10000),), True)
        replay.drain(T+110_000_000)
        self.assertEqual(a.ledger.committed_cents, 10400000)
        self.assertEqual(a.ledger.events[-1]["kind"], "hedged")
        self.assertEqual(a.ledger.events[-1]["excess_cents"], 100000)
        self.assertFalse(a.ledger.events[-1]["estimate"])

    def test_failed_hedge_retains_maker_exposure(self):
        m = FakeMarket([("S:2330", T+60_000_000, 505000, 2000)], fut_qty=0)
        a, replay = unreserved(m, cap_twd=101000)
        a.offer(row(), decision())
        replay.drain(m.end)
        self.assertEqual(a.ledger.committed_cents, 10100000)
        self.assertEqual(next(iter(a.cycles.values())).state, "entry_hedge")


if __name__ == "__main__":
    unittest.main()
