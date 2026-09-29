"""Regressions for the execution-cost extension (ev_lookup_xc)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from .causal_lookup import CLOSE_SECOND, SECOND, DecayObservation, LookupHistory, open_ns
from .decide import EntryDecider, policy_ev_xc
from .execution import Book, price_i
from .exit_model import forecast_exits, policy_ev
from .market import Contract
from .portfolio import Portfolio


class DecayTableTests(unittest.TestCase):
    def test_entry_and_exit_decay_tables_are_walk_forward_with_fallbacks(self):
        h, d0, d1 = LookupHistory(), "20260723", "20260724"
        close = open_ns(d0) + CLOSE_SECOND * SECOND
        for i in range(40):
            o = DecayObservation(f"e{i}", "S2", "entry", 0, 2, 20.0 if i % 2 else 10.0, d0, close)
            if i == 0:
                with self.assertRaises(ValueError):
                    h.observe_decay(o, close - 1)
            h.observe_decay(o, close)
        with self.assertRaises(ValueError):
            h.observe_decay(DecayObservation("e0", "S2", "entry", 0, 2, 1.0, d0, close), close)
        for i in range(35):
            h.observe_decay(DecayObservation(f"x{i}", "S2", "exit", 1, -1, 8.0, d0, close), close)
        self.assertEqual(dict(h.freeze(d0, 20e6, {}).decay), {})
        h.finish_session(d0)
        s = h.freeze(d1, 20e6, {})
        self.assertAlmostEqual(s.entry_decay("S2", 100, 100.0, margin_bp=3.0), 18.0)   # exact cell
        self.assertAlmostEqual(s.entry_decay("S2", 5000, 100.0, margin_bp=3.0), 18.0)  # time pooled
        self.assertAlmostEqual(s.entry_decay("S1", 100, 10.0, margin_bp=3.0), 18.0)    # all-stream pooled
        self.assertAlmostEqual(s.exit_decay("S2", True, margin_bp=3.0), 11.0)
        self.assertAlmostEqual(s.exit_decay("S1", True, margin_bp=3.0), 11.0)
        self.assertAlmostEqual(s.exit_decay("S2", False, margin_bp=3.0), 8.0)          # prior 5 + margin
        empty = LookupHistory().freeze(d1, 20e6, {})
        self.assertAlmostEqual(empty.entry_decay("S2", 100, 0.0, margin_bp=0.0), 15.0)
        self.assertAlmostEqual(empty.exit_decay("S2", True, margin_bp=0.0), 5.0)

    def test_negative_measured_decay_is_floored_not_credited(self):
        h, d0 = LookupHistory(), "20260723"
        close = open_ns(d0) + CLOSE_SECOND * SECOND
        for i in range(40):
            h.observe_decay(DecayObservation(f"x{i}", "S1", "exit", 1, -1, -9.0, d0, close), close)
        h.finish_session(d0)
        self.assertAlmostEqual(h.freeze("20260724", 20e6, {}).exit_decay("S1", True, margin_bp=3.0), 3.0)


class DecayEvTests(unittest.TestCase):
    def test_decay_is_removed_from_every_route(self):
        f = forecast_exits("20260724", "20260819", "S2", 0.4, {}, {})
        eff_u, ab, d_in, d_sd, d_on = 40.0, 90.0, 18.0, 8.0, 11.0
        expected = (policy_ev(eff_u, ab, f) - d_in * (f.p_sd + f.p_overnight + f.p_expiry)
                    - f.p_sd * d_sd - f.p_overnight * d_on)
        self.assertAlmostEqual(policy_ev_xc(eff_u - d_in, ab - d_in, f, d_sd, d_on), expected)

    def test_decider_uses_priors_and_reports_components(self):
        day = "20260724"
        snapshot = LookupHistory().freeze(day, 1e7, {})
        args = dict(now_ns=open_ns(day) + 300 * SECOND, stream="S2", quote_second=300,
                    eff_u=40.0, ab=40.0, reservation_cents=1000, committed_cents=0,
                    expiry="20260819", spread_bp=100.0)
        plain = EntryDecider(10**9, snapshot).decide(**args)
        costed = EntryDecider(10**9, snapshot, exec_cost=True, decay_margin_bp=3.0).decide(**args)
        self.assertEqual((plain.d_in, plain.d_sd, plain.d_on), (0.0, 0.0, 0.0))
        self.assertEqual((costed.d_in, costed.d_sd, costed.d_on), (18.0, 8.0, 8.0))
        self.assertLess(costed.est_bp, plain.est_bp - 18.0)
        self.assertEqual(costed.p_sd, plain.p_sd)


class XcPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.day = "20260724"
        self.start = open_ns(self.day)
        self.now = self.start + 400 * SECOND
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

    def make(self, **flags):
        p = Portfolio("test", 10_000_000, use_ev=True, use_bpday=False, cancel_ms=50, **flags)
        p.begin(self.market, 0, LookupHistory().freeze(self.day, 1e7, {}), lambda *e: self.tasks.append(e))
        return p

    def paired(self, p):
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                 anchor=0., ab=50., eff_u=50.)
        p.trade("F:XHF6", self.now + 900_000_000, 1, price_i(100.5), 1)
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        pos = p.positions["x"]
        self.assertEqual(pos.state, "paired")
        return pos

    def test_relative_drift_cancels_a_deep_quote_before_the_floor(self):
        p = self.make(event_quotes=True, repeg_drop_bp=10.0)
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                 anchor=0., ab=50., eff_u=50.)
        self.assertEqual(p.positions["x"].quote_ab, 50.0)
        self.spot_ask = 100.15                       # locked basis 35 bp: above floor 20, 15 below quote
        p.book_update("S:X", self.now + 1)
        self.assertEqual(self.tasks[-1][1], "cancel")
        self.assertEqual(p.drift_cancels, 1)
        self.assertEqual(next(t for t in p.trace if t["kind"] == "event_cancel")["reason"], "drift")
        self.tasks.clear()
        legacy = self.make(event_quotes=True, repeg_drop_bp=None)
        legacy.submit(intent_id="y", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        legacy.book_update("S:X", self.now + 1)
        self.assertFalse(self.tasks)

    def test_no_maker_exit_quotes_before_open_delay(self):
        p = self.make(event_quotes=True, exit_open_delay_s=600)
        pos = self.paired(p)                             # paired at 400.95 s
        self.spot_ask, self.future_ask = 100.6, 100.0    # taker exit basis -10 bp <= target -5
        p.second(401)
        self.assertIsNone(pos.exit_order)
        p.exit_open_delay_s = 300
        p.second(402)
        self.assertIsNotNone(pos.exit_order)

    def test_exit_guard_reacts_to_futures_update(self):
        p = self.make(event_quotes=True, exit_event_guard=True)
        pos = self.paired(p)
        self.spot_ask, self.future_ask = 100.6, 100.0
        p.second(401)
        self.assertIsNotNone(pos.exit_order)
        self.tasks.clear()
        self.future_ask = 101.0                          # working sell price now misses the target
        p.book_update("F:XHF6", self.start + 401 * SECOND + 1)
        self.assertEqual(self.tasks[-1][1], "cancel")
        self.assertEqual(p.exit_guard_cancels, 1)

    def test_expiry_carry_policy_skips_target_exit_for_carry(self):
        p = self.make(event_quotes=True, carry_exit="expiry")
        pos = self.paired(p)
        pos.entry_day = "20260723"                       # pretend it was carried in
        self.spot_ask, self.future_ask = 100.6, 100.0
        p.second(401)
        self.assertIsNone(pos.exit_order)

    def test_s1_stream_can_be_disabled(self):
        p = self.make(event_quotes=False, use_s1=False)
        p.offer_s1(dict(ValueCode="X", raw_order_fact_id="s", QuoteCode="XHF6", target_price=99.0,
                        nominal_new_time_ns=self.now, nominal_stop_time_ns=self.now + SECOND), self.now)
        self.assertFalse(p.queue.orders)


if __name__ == "__main__":
    unittest.main()
