"""Regressions for the 2026-09-09 changes: overnight EV split (P_nx), causal
P_fill table, taker-cross exit rule, and event-driven S2 quote maintenance."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from .causal_lookup import (CLOSE_SECOND, SECOND, ExitRiskObservation, LookupHistory,
                            ResolvedOutcome, open_ns)
from .ev_rules import est_short, should_cross
from .execution import Book, price_i
from .market import Contract
from .portfolio import Portfolio


class EvSplitTests(unittest.TestCase):
    def test_overnight_branch_values_the_actual_exit_rule(self):
        # ab=80, anchor=50 -> eff_u=30, target = anchor-5 = 45. Never same-day.
        self.assertAlmostEqual(est_short(30., 80., 0.0, p_nx=1.0), 80 - 45 - 34)   # 1 bp
        self.assertAlmostEqual(est_short(30., 80., 0.0, p_nx=0.0), 80 - 0 - 34)    # 46 bp legacy
        self.assertAlmostEqual(est_short(30., 80., 0.0, p_nx=0.5), 23.5)
        self.assertAlmostEqual(est_short(30., 80., 1.0, p_nx=0.3), 80 - 45 - 20)   # same-day unchanged

    def test_pnx_and_pfill_tables_are_walk_forward(self):
        h, d0, d1 = LookupHistory(), "20260723", "20260724"
        for i in range(40):
            r = ResolvedOutcome(str(i), "S2", 40., "20260722", d0, open_ns(d0) + 1000 * SECOND, 1., 1.,
                                "maker_exit" if i < 30 else "expiry_basis_zero_accounting")
            h.observe_resolution(r, r.available_ns)
        close = open_ns(d0) + CLOSE_SECOND * SECOND
        for i in range(40):
            o = ExitRiskObservation(f"x{i}", "S2", d0, 1000., 10_000. if i < 20 else None, close)
            with self.assertRaises(ValueError):
                h.observe_exit_risk(o, close - 1)
            h.observe_exit_risk(o, close)
        self.assertEqual(h.freeze(d0, 20e6, {}).pnx, {})     # same session: not yet usable
        h.finish_session(d0)
        s = h.freeze(d1, 20e6, {})
        self.assertAlmostEqual(s.p_nx("S2"), 0.75)
        self.assertAlmostEqual(s.p_nx("S1"), 0.8)             # n < 30 prior
        self.assertAlmostEqual(s.p_fill(0), 0.5)               # 40 at risk, 20 later filled
        self.assertAlmostEqual(s.p_fill(9000), 0.5)            # exits at 10,000 s count as still open at 9,000
        self.assertAlmostEqual(s.p_fill(12_600), 0.5)          # only 20 at risk -> prior
        self.assertEqual(s.pfill[2], (20, 0))

    def test_should_cross_widens_as_fill_chance_vanishes(self):
        self.assertFalse(should_cross(10., 0.5, 0., True))     # threshold 7
        self.assertTrue(should_cross(10., 0.0, 0., True))      # threshold 14
        self.assertFalse(should_cross(10., 0.0, 0., False))    # carry: only lambda pays
        self.assertTrue(should_cross(10., 0.0, 12., False))


class EventPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.day = "20260724"
        self.start = open_ns(self.day)
        self.now = self.start + 300 * SECOND
        self.c = Contract("X", "XHF6", 2000, "20260819", price_i(100), price_i(100))
        self.spot_ask, self.future_ask = 100.0, 101.0

        def book(instrument, ns):
            if instrument.startswith("S:"):
                ask = self.spot_ask
                return Book(instrument, ns, ns, ((price_i(ask - .5), 100_000),),
                            ((price_i(ask), 100_000),), True)
            ask = self.future_ask
            return Book(instrument, ns, ns, ((price_i(ask - 1.), 10),), ((price_i(ask), 10),), True)

        self.market = SimpleNamespace(day=self.day, start=self.start, end=self.start + CLOSE_SECOND * SECOND,
                                      contracts={"X": self.c}, book=book, series={},
                                      signals={"X": {"an": np.zeros(CLOSE_SECOND + 1)}},
                                      pair=lambda c, ns, signal_buffer=True: (book("S:X", ns), book("F:XHF6", ns)))
        self.tasks = []
        self.p = Portfolio("test", 10_000_000, use_ev=True, use_bpday=True, cancel_ms=50)
        self.p.begin(self.market, 0, LookupHistory().freeze(self.day, 1e7, {}),
                     lambda *event: self.tasks.append(event))

    def test_spot_ask_move_cancels_then_requotes_inside(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.assertEqual(self.p.s2_live["X"], "x")
        self.p.book_update("S:X", self.now + 1)               # ask unchanged: still 50 bp -> keep
        self.assertFalse(self.tasks)
        self.spot_ask = 100.4                                  # held basis 10 bp < 20 floor
        self.p.book_update("S:X", self.now + 2)
        cancel_ns, kind, actor, oid = self.tasks.pop()
        self.assertEqual((kind, oid, cancel_ns), ("cancel", "x", self.now + 2 + 50_000_000))
        # A print before the cancel becomes effective still fills and must be hedged.
        self.p.trade("F:XHF6", cancel_ns - 1, 1, price_i(100.5), 1)
        self.assertEqual(self.p.positions["x"].future_sell_qty, 1)
        self.p.cancel("x", cancel_ns)                          # nothing left to cancel
        self.assertEqual(self.p.positions["x"].state, "hedging")
        self.assertNotIn("X", self.p.s2_live)

    def test_requote_moves_back_to_restore_basis_after_cancel(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.spot_ask = 100.4
        self.p.book_update("S:X", self.now + 2)
        cancel_ns = self.tasks.pop()[0]
        self.p.cancel("x", cancel_ns)
        self.assertEqual(self.p.positions["x"].state, "cancelled")
        # Replacement is emitted only after all same-time cancellations finish.
        self.assertNotIn("X", self.p.s2_live)
        self.assertEqual(self.p.ledger.committed_cents, 0)
        self.p._requote_s2("X", cancel_ns)
        pid = self.p.s2_live["X"]
        self.assertTrue(pid.startswith("S2/20260724/X/e"))
        self.assertEqual(self.p.queue.orders[pid].price, price_i(101.0))
        self.assertEqual(self.p.decisions[-1]["intent_id"], pid)
        self.assertGreater(self.p.ledger.committed_cents, 0)

    def test_afternoon_cross_rule_closes_with_taker_kind(self):
        self.p.enable_cross = True
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.p.trade("F:XHF6", self.now + 900_000_000, 1, price_i(100.5), 1)
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        pos = self.p.positions["x"]
        self.assertEqual(pos.state, "paired")
        # 12:00, taker exit basis = future ask / spot bid: make it 0 bp -> d = 5 < 14*(1-0.5).
        self.spot_ask, self.future_ask = 101.0, 100.5
        self.p.second(10_800)
        self.assertTrue(pos.cross_requested)
        ns, kind, actor, pid = self.tasks.pop()
        self.assertEqual(kind, "hedge")
        actor.hedge(pid, ns)                                   # sells spot at bid
        self.assertEqual(pos.spot_sell_qty, 2000)
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)                                   # buys future at ask
        self.assertEqual((pos.state, pos.close_kind), ("closed", "taker_cross"))
        self.assertEqual(self.p.resolved[-1].kind, "taker_cross")
        self.assertEqual(self.p.ledger.committed_cents, 0)

    def test_cross_rule_is_not_applied_in_the_morning(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.p.trade("F:XHF6", self.now + 900_000_000, 1, price_i(100.5), 1)
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.spot_ask, self.future_ask = 101.0, 100.5
        self.p.second(3600)
        self.assertFalse(self.p.positions["x"].cross_requested)
        self.assertFalse(self.tasks)


if __name__ == "__main__":
    unittest.main()
