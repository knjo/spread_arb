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

    def test_unreserved_quotes_consume_capacity_only_when_filled(self):
        # cap 450k TWD; each S2 fill books ~201k (100.5 x 2000), each quote would reserve 220k at limit-up.
        cy = Contract("Y", "YHF6", 2000, "20260819", price_i(100), price_i(100))
        cz = Contract("Z", "ZHF6", 2000, "20260819", price_i(100), price_i(100))
        self.market.contracts.update(Y=cy, Z=cz)
        for v in ("Y", "Z"):
            self.market.signals[v] = {"an": np.zeros(CLOSE_SECOND + 1)}
        p = Portfolio("test", 450_000, use_ev=True, use_bpday=False, cancel_ms=50,
                      event_quotes=True, reserve_quotes=False)
        p.begin(self.market, 0, LookupHistory().freeze(self.day, 1e7, {}), lambda *e: self.tasks.append(e))
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now, anchor=0., ab=50., eff_u=50.)
        p.submit(intent_id="y", c=cy, stream="S2", price=price_i(100.5), ns=self.now, anchor=0., ab=50., eff_u=50.)
        self.assertEqual(p.ledger.committed_cents, 0)           # two working quotes, nothing reserved
        self.assertEqual(p.unreserved, {"x", "y"})
        p.trade("F:XHF6", self.now + SECOND, 1, price_i(100.5), 1)
        self.assertEqual(p.ledger.committed_cents, price_i(110) * 2000 // 100)   # booked at limit-up
        self.assertIn("y", p.queue.orders)                       # 220k still fits in the remaining 230k
        self.assertFalse(any(t[1] == "cancel" for t in self.tasks))
        p.submit(intent_id="z", c=cz, stream="S2", price=price_i(100.5), ns=self.now + SECOND + 1,
                 anchor=0., ab=50., eff_u=50.)
        self.assertIn("z", p.queue.orders)                       # 201k + 220k <= 450k
        p.trade("F:YHF6", self.now + 2 * SECOND, 2, price_i(100.5), 1)
        self.assertEqual(p.capacity_cancels, 1)                  # z no longer fits: withdrawn
        self.assertEqual(self.tasks[-1][1], "cancel")
        self.assertEqual(self.tasks[-1][3], "z")
        # A print racing the cancel still books, as a capacity overrun.
        p.trade("F:ZHF6", self.now + 2 * SECOND + 1, 3, price_i(100.5), 1)
        self.assertGreater(p.ledger.committed_cents, p.ledger.cap_cents)
        self.assertTrue(any(t["kind"] == "unreserved_fill" and t["overrun_cents"] > 0 for t in p.trace))
        p.submit(intent_id="w", c=self.c, stream="S2", price=price_i(100.5), ns=self.now + 3 * SECOND,
                 anchor=0., ab=50., eff_u=50.)
        self.assertEqual(p.decisions[-1]["reason"], "cap")        # nothing new until back under the cap

    def test_ticket_cap_skips_oversized_entries_before_any_decision(self):
        p = self.make(event_quotes=True, max_ticket_twd=200_000)      # limit-up nominal here is 220k
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                 anchor=0., ab=50., eff_u=50.)
        self.assertFalse(p.queue.orders)
        self.assertFalse(p.decisions)
        self.assertEqual(p.ticket_rejects, 1)
        p.max_ticket_cents = 300_000 * 100
        p.submit(intent_id="y", c=self.c, stream="S2", price=price_i(100.5), ns=self.now + 1,
                 anchor=0., ab=50., eff_u=50.)
        self.assertIn("y", p.queue.orders)

    def test_q_table_gate_uses_realized_bp_per_capital_day(self):
        from .causal_lookup import ResolvedOutcome
        h, d0, d1 = LookupHistory(), "20260723", "20260724"
        for i in range(40):   # U 50 / spread 50 / 09:30 cell: +30 bp over 5 capital-days = 6 bp/day
            r = ResolvedOutcome(f"a{i}", "S2", 90., d0, d0, open_ns(d0) + 1000 * SECOND, 30., .15,
                                "maker_exit", 50., 50., 1800, 5.0)
            h.observe_resolution(r, r.available_ns)
        for i in range(40):   # U 50 / spread 50 / 10:30 cell: +30 bp over 0.5 day = 60 bp/day
            r = ResolvedOutcome(f"b{i}", "S2", 90., d0, d0, open_ns(d0) + 1000 * SECOND, 30., .15,
                                "maker_exit", 50., 50., 5400, 0.5)
            h.observe_resolution(r, r.available_ns)
        h.finish_session(d0)
        s = h.freeze(d1, 20e6, {})
        self.assertAlmostEqual(s.q_bpday("S2", 50., 50., 1800), 6.0)
        self.assertAlmostEqual(s.q_bpday("S2", 50., 50., 5400), 60.0)
        self.assertAlmostEqual(s.q_bpday("S2", 50., 50., 12000), 30. / 2.75)   # time-pooled fallback
        self.assertIsNone(s.q_bpday("S2", 120., 50., 1800))                     # unknown depth cell
        args = dict(now_ns=open_ns(d1) + 1800 * SECOND, stream="S2", quote_second=1800, eff_u=50., ab=90.,
                    reservation_cents=1000, committed_cents=0, expiry="20260819", spread_bp=50.)
        gated = EntryDecider(10**9, s, q_threshold_bpday=12.0).decide(**args)
        self.assertEqual(gated.reason, "qcell")
        self.assertAlmostEqual(gated.q_bpday, 6.0)
        self.assertTrue(EntryDecider(10**9, s, q_threshold_bpday=12.0).decide(**dict(args, now_ns=open_ns(d1) + 5400 * SECOND, quote_second=5400)).admit)
        self.assertTrue(EntryDecider(10**9, s).decide(**args).admit)

    def test_undercut_repegs_inside_and_never_queues_at_a1(self):
        p = self.make(event_quotes=True, chase_undercut=True)
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                 anchor=0., ab=50., eff_u=50.)
        self.future_ask = 100.4                          # someone posted below us: we are no longer first
        self.market.future_to_vc = {"XHF6": "X"}
        p.book_update("F:XHF6", self.now + 1)
        self.assertEqual(self.tasks[-1][1], "cancel")
        self.assertEqual(p.undercut_cancels, 1)
        self.assertEqual(next(t for t in p.trace if t["kind"] == "event_cancel")["reason"], "undercut")
        p.cancel("x", self.tasks[-1][0])
        p._requote_s2("X", self.tasks[-1][0])            # A1 100.4 -> inside is 99.9, basis vs spot 100 = -10 bp: stay out
        self.assertNotIn("X", p.s2_live)
        self.spot_ask = 99.0                             # inside 99.9 vs spot 99.0 = +91 bp: re-peg inside
        p._requote_s2("X", self.tasks[-1][0] + 1)
        pid = p.s2_live["X"]
        self.assertEqual(p.queue.orders[pid].price, price_i(99.9))
        legacy = self.make(event_quotes=True, chase_undercut=False)
        legacy.submit(intent_id="y", c=self.c, stream="S2", price=price_i(100.5), ns=self.now + SECOND,
                      anchor=0., ab=50., eff_u=50.)
        self.tasks.clear()
        legacy.book_update("F:XHF6", self.now + SECOND + 1)
        self.assertFalse(self.tasks)                     # old behaviour: sit above A1

    def test_s1_stream_can_be_disabled(self):
        p = self.make(event_quotes=False, use_s1=False)
        p.offer_s1(dict(ValueCode="X", raw_order_fact_id="s", QuoteCode="XHF6", target_price=99.0,
                        nominal_new_time_ns=self.now, nominal_stop_time_ns=self.now + SECOND), self.now)
        self.assertFalse(p.queue.orders)


if __name__ == "__main__":
    unittest.main()
