"""Behavioral regressions for the five failures found in the v17/v18 audit."""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from .causal_lookup import (SECOND, DayEntryOutcome, LookupHistory, ResolvedOutcome,
                            open_ns)
from .decide import EntryDecider
from .execution import (Book, CapacityLedger, PrintedVolumeQueue, QueueOrder,
                        TakerDepth, price_i, realized_pnl)
from .market import BookSeries, Contract
from .portfolio import Portfolio


class LookupTests(unittest.TestCase):
    def test_same_day_and_future_outcomes_cannot_change_morning_decisions(self):
        h = LookupHistory()
        d0, d1 = "20260723", "20260724"
        for i in range(30):
            r = ResolvedOutcome(str(i), "S2", 40, d0, d0, open_ns(d0) + 1000 * SECOND, -10, 1)
            h.observe_resolution(r, r.available_ns)
        h.finish_session(d0)
        frozen = h.freeze(d1, 20e6, {})
        args = dict(now_ns=open_ns(d1) + 300 * SECOND, stream="S2", quote_second=300,
                    eff_u=40., ab=40., reservation_cents=100_000, committed_cents=0, expiry="20260819")
        before = EntryDecider(2_000_000_000, frozen).decide(**args)
        self.assertEqual(before.reason, "cell")
        for i in range(100):
            r = ResolvedOutcome("future" + str(i), "S2", 40, d1, d1,
                                open_ns(d1) + 10_000 * SECOND, 1000, .15)
            with self.assertRaises(ValueError):
                h.observe_resolution(r, args["now_ns"])
            h.observe_resolution(r, r.available_ns)
        self.assertEqual(EntryDecider(2_000_000_000, frozen).decide(**args), before)
        self.assertEqual(dict(h.freeze(d1, 20e6, {}).cells), dict(frozen.cells))
        with self.assertRaises(TypeError):
            frozen.cells["S2_1"] = (1000, .15, 30)
        h.finish_session(d1)
        self.assertEqual(h.freeze("20260727", 20e6, {}).cells["S2_1"][2], 130)

    def test_carry_resolution_and_psd_must_be_observed(self):
        h = LookupHistory()
        h.finish_session("20260723")
        r = ResolvedOutcome("carry", "S1", 70, "20260723", "20260724",
                            open_ns("20260724") + 12_000 * SECOND, 60, 1)
        h.observe_resolution(r, r.available_ns)
        self.assertFalse(h.freeze("20260724", 20e6, {}).cells)
        bad = DayEntryOutcome("x", "S2", 300, "20260724", open_ns("20260724") + SECOND, True)
        with self.assertRaises(ValueError):
            h.observe_entry_day(bad, bad.available_ns)
        with self.assertRaises(ValueError):
            h.observe_resolution(r, r.available_ns)

    def test_window_uses_last_twenty_completed_resolution_sessions(self):
        h = LookupHistory()
        dates = [f"202606{i:02}" for i in range(1, 23)]
        for i, day in enumerate(dates):
            r = ResolvedOutcome(str(i), "S1", 90, day, day, open_ns(day) + SECOND, 1, .15)
            h.observe_resolution(r, r.available_ns)
            h.finish_session(day)
        snap = h.freeze("20260623", 20e6, {"20260623": 1e12})
        self.assertEqual(snap.cells["S1_3"][2], 20)
        self.assertAlmostEqual(snap.lam_bp, 0)


class CapacityTests(unittest.TestCase):
    def test_pending_and_carry_share_cap_until_both_legs_close(self):
        ledger = CapacityLedger(20_000)
        self.assertTrue(ledger.reserve("carry", 12_000, 1))
        self.assertTrue(ledger.reserve("S2_pending", 8_000, 2))
        self.assertFalse(ledger.reserve("S1_pending", 1, 3))
        with self.assertRaises(ValueError):
            ledger.release("carry", 4, terminal=False)
        self.assertEqual(ledger.committed_cents, 20_000)
        ledger.confirm_nominal("S2_pending", 7000, 5)
        self.assertTrue(ledger.reserve("S1_pending", 1000, 6))
        ledger.release("carry", 10, terminal=True)
        self.assertEqual(ledger.committed_cents, 8000)
        self.assertEqual(ledger.peak_cents, 20_000)

    def test_psd_does_not_predict_available_carry_capacity(self):
        snapshot = LookupHistory().freeze("20260724", 200., {})
        decider = EntryDecider(20_000, snapshot)
        d = decider.decide(now_ns=open_ns("20260724") + 300 * SECOND,
                           stream="S2", quote_second=300, eff_u=100, ab=100,
                           reservation_cents=6000, committed_cents=15_000, expiry="20260819")
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, "cap")
        self.assertFalse(d.slot_free)


class QueueTests(unittest.TestCase):
    def test_37_lots_ahead_plus_40_own_cannot_all_fill_on_41_lots(self):
        q = PrintedVolumeQueue()
        for i in range(20):
            q.add(QueueOrder(str(i), "S:8039", "sell", price_i(253), 2000,
                             1, "exit", str(i)), 37_000)
        fills = q.trade("S:8039", 2, 42, price_i(253), 41_000)
        self.assertEqual(sum(n for _, n in fills), 4000)
        self.assertEqual(len(q.orders), 18)
        # Even a trade above our limit has only its own finite volume.
        fills = q.trade("S:8039", 3, 43, price_i(254), 1000)
        self.assertEqual(sum(n for _, n in fills), 1000)
        self.assertEqual(q.orders["2"].remaining, 1000)
        with self.assertRaises(ValueError):
            q.trade("S:8039", 3, 43, price_i(254), 1000)

    def test_contract_and_actual_time_and_cancel_race(self):
        q = PrintedVolumeQueue()
        o = QueueOrder("S2", "F:CFFE6", "sell", price_i(50.2), 1, 301_000,
                       "entry", "S2", cancel_ns=302_000)
        q.add(o, 0)
        self.assertFalse(q.trade("F:CFFF6", 301_100, 1, price_i(50.2), 1))
        self.assertFalse(q.trade("F:CFFE6", 301_000, 2, price_i(50.2), 1))
        fills = q.trade("F:CFFE6", 301_900, 3, price_i(50.2), 1)
        self.assertEqual(fills[0][1], 1)
        self.assertIsNone(q.cancel("S2", 302_000))

    def test_future_trade_never_fills_already_cancelled_order(self):
        q = PrintedVolumeQueue()
        q.add(QueueOrder("x", "F:A", "sell", 100, 1, 1, "entry", "x"), 0)
        q.cancel("x", 2)
        self.assertFalse(q.trade("F:A", 3, 1, 100, 100))


class ExecutionTests(unittest.TestCase):
    def test_hedge_uses_actual_depth_and_does_not_reuse_snapshot(self):
        b = Book("S:X", 10, 1, ((price_i(100), 2000),),
                 ((price_i(101), 1000), (price_i(102), 1000)), True)
        depth = TakerDepth()
        self.assertEqual(depth.take(b, "buy", 2000, 20), (price_i(203) * 1000, 2000))
        self.assertIsNone(depth.take(b, "buy", 1000, 20))
        with self.assertRaises(ValueError):
            depth.take(b, "buy", 1000, 9)
        pnl, bp = realized_pnl(price_i(100) * 2000, price_i(102) * 2000,
                               price_i(101) * 2000, price_i(103) * 2000,
                               entry_day="20260724", exit_day="20260724")
        self.assertAlmostEqual(pnl, -400)
        self.assertAlmostEqual(bp, -20)

    def test_raw_book_asof_does_not_use_later_part_of_second(self):
        rows = []
        for ns, ask in ((300 * SECOND, 100), (300 * SECOND + 900_000_000, 200)):
            r = dict(ns=ns, seq=ns, formal=True)
            for side in ("Bid", "Ask"):
                for i in range(1, 6):
                    r[f"{side}Price{i}"] = price_i(ask - 1 if side == "Bid" else ask) if i == 1 else 0
                    r[f"{side}Lots{i}"] = 2000 if i == 1 else 0
                r[f"Best{side}Price"] = r[f"{side}Price1"]
                r[f"Best{side}Lots"] = 2000
            rows.append(r)
        series = BookSeries("S:X", pl.from_dicts(rows))
        self.assertEqual(series.at(300 * SECOND).asks[0][0], price_i(100))
        self.assertEqual(series.at(301 * SECOND).asks[0][0], price_i(200))


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.day = "20260724"
        self.start = open_ns(self.day)
        self.now = self.start + 300 * SECOND
        self.c = Contract("X", "XHF6", 2000, "20260819", price_i(100), price_i(100))
        self.spot_ask = 100.0
        self.future_ask = 100.5
        self.future_book_ns = self.now
        def book(instrument, ns):
            spot = instrument.startswith("S:")
            ask = self.spot_ask if spot else self.future_ask
            return Book(instrument, min(ns, self.future_book_ns if not spot else ns), ns,
                        ((price_i(ask - .5), 100_000),), ((price_i(ask), 100_000),), True)
        self.market = SimpleNamespace(day=self.day, start=self.start, end=self.start + 15600 * SECOND,
                                      contracts={"X": self.c}, book=book, series={})
        self.tasks = []
        self.p = Portfolio("test", 1_000_000, use_ev=True, use_bpday=True)
        self.p.begin(self.market, 0, LookupHistory().freeze(self.day, 1e6, {}),
                     lambda *event: self.tasks.append(event))

    def enter_s2(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.4), ns=self.now,
                      anchor=0., ab=40., eff_u=40.)
        self.p.trade("F:XHF6", self.now + 900_000_000, 1, price_i(100.4), 1)
        return self.p.positions["x"]

    def test_adverse_s2_fill_must_buy_spot_and_stay_in_ledger(self):
        pos = self.enter_s2()
        self.spot_ask = 101.
        ns, kind, actor, pid = self.tasks.pop()
        self.assertEqual(ns, self.now + 950_000_000)
        actor.hedge(pid, ns)
        self.assertEqual(pos.state, "paired")
        self.assertLess(pos.actual_ab, 0)
        self.assertEqual(pos.spot_buy_qty, 2000)
        self.assertEqual(pos.spot_buy_cash, price_i(101) * 2000)
        self.assertEqual(self.p.ledger.committed_cents, 20_200_000)
        self.assertEqual(len(self.p.decisions), 1)

    def test_entry_risk_buffer_cannot_veto_mandatory_spot_hedge(self):
        pos = self.enter_s2()
        self.spot_ask = 109.
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.assertEqual(pos.state, "paired")
        self.assertEqual(pos.spot_buy_cash, price_i(109) * 2000)
        self.assertLessEqual(self.p.ledger.committed_cents, 22_000_000)

    def test_nonpositive_basis_is_rejected_before_quote_only(self):
        self.p.submit(intent_id="bad_quote", c=self.c, stream="S1", price=price_i(99), ns=self.now,
                      anchor=-100., ab=-1., eff_u=99.)
        self.assertEqual(self.p.decisions[-1]["reason"], "basis")
        self.assertNotIn("bad_quote", self.p.queue.orders)
        self.assertEqual(self.p.ledger.committed_cents, 0)

    def test_capacity_releases_only_after_actual_second_leg_exit(self):
        pos = self.enter_s2()
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        before = self.p.ledger.committed_cents
        exit_ns = ns + SECOND
        self.p.queue.add(QueueOrder("out", "S:X", "sell", price_i(101), 2000,
                                    exit_ns, "exit", pid), 0)
        pos.exit_order = "out"
        self.p.trade("S:X", exit_ns + SECOND, 2, price_i(101), 2000)
        self.assertEqual(self.p.ledger.committed_cents, before)
        self.future_ask = 102.
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.assertEqual(pos.state, "closed")
        self.assertEqual(self.p.ledger.committed_cents, 0)
        self.assertAlmostEqual(pos.pnl_twd, -1600.)

    def test_exit_guard_cancels_using_current_hedge_then_retains_racing_fill(self):
        pos = self.enter_s2()
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.market.signals = {}
        self.p.cancel_delay = 50_000_000
        quote_ns = self.start + 301 * SECOND
        self.p.queue.add(QueueOrder("out", "S:X", "sell", price_i(101), 2000,
                                    quote_ns, "exit", pid), 0)
        pos.exit_order = "out"
        self.future_ask = 103.
        self.p.second(302)
        cancel_ns, kind, _, oid = self.tasks.pop()
        self.assertEqual(kind, "cancel")
        self.assertEqual(cancel_ns, self.start + 302 * SECOND + 50_000_000)
        self.p.trade("S:X", cancel_ns - 1, 2, price_i(101), 2000)
        self.p.cancel(oid, cancel_ns)
        self.assertEqual(pos.spot_sell_qty, 2000)
        self.assertEqual(pos.hedge_kind, "exit_future")
        self.assertGreater(self.p.ledger.committed_cents, 0)

    def test_partial_s1_cancel_is_explicit_rollback(self):
        self.p.submit(intent_id="s1", c=self.c, stream="S1", price=price_i(99), ns=self.now,
                      anchor=0., ab=100., eff_u=100.)
        self.p.trade("S:X", self.now + 1, 1, price_i(99), 1000)
        self.p.cancel("s1", self.now + 2)
        pos = self.p.positions["s1"]
        self.assertEqual(pos.spot_buy_qty, 1000)
        self.assertGreater(self.p.ledger.committed_cents, 0)
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.assertEqual(pos.state, "closed")
        self.assertEqual(pos.future_sell_qty, 0)
        self.assertEqual(pos.spot_sell_qty, 1000)

    def test_missing_hedge_does_not_erase_maker_fill(self):
        pos = self.enter_s2()
        self.market.book = lambda instrument, ns: None
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        self.assertEqual(pos.future_sell_qty, 1)
        self.assertEqual(pos.state, "hedging")
        self.assertGreater(self.p.ledger.committed_cents, 0)

    def test_exact_expiry_settles_next_session_not_next_month(self):
        pos = self.enter_s2()
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        pos.contract = replace(pos.contract, expiry=self.day)
        next_day = "20260727"
        next_market = SimpleNamespace(day=next_day, start=open_ns(next_day), contracts={})
        snapshot = LookupHistory().freeze(next_day, 1e6, {})
        self.p.begin(next_market, 1, snapshot, lambda *args: None)
        self.assertEqual(pos.close_ns, open_ns(next_day))
        self.assertEqual(pos.state, "closed")
        self.assertEqual(self.p.ledger.committed_cents, 0)
        self.assertAlmostEqual(pos.pnl_twd, 120.)

    def test_carry_cannot_silently_roll_into_another_contract(self):
        self.enter_s2()
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        next_day = "20260727"
        next_market = SimpleNamespace(day=next_day, start=open_ns(next_day),
                                     contracts={"X": replace(self.c, qc="XI6")})
        with self.assertRaisesRegex(ValueError, "missing exact carry"):
            self.p.begin(next_market, 1, LookupHistory().freeze(next_day, 1e6, {}), lambda *args: None)

    def test_removed_entry_name_still_tracks_exact_carry(self):
        pos = self.enter_s2()
        ns, _, actor, pid = self.tasks.pop()
        actor.hedge(pid, ns)
        next_day = "20260727"
        next_market = SimpleNamespace(day=next_day, start=open_ns(next_day), contracts={},
                                     exact={self.c.qc: replace(self.c, spot_ref=price_i(101))})
        self.p.begin(next_market, 1, LookupHistory().freeze(next_day, 1e6, {}), lambda *args: None)
        self.assertEqual(pos.state, "paired")
        self.assertEqual(pos.contract.qc, self.c.qc)
        self.assertGreater(self.p.ledger.committed_cents, 0)


class RawRegressionTests(unittest.TestCase):
    def test_actual_8039_tape_cannot_fill_twenty_exit_units(self):
        path = Path(__file__).resolve().parents[2] / "data/ev_lookup_audit_20260908/sample_8039_20260724_ticks.parquet"
        if not path.exists():
            self.skipTest("local audit tape is unavailable")
        raw = pl.read_parquet(path).with_columns(pl.col("RecvTime").dt.epoch("ns").alias("ns"))
        row = raw.filter(pl.col("ChannelSeq") == 1986698).row(0, named=True)
        q = PrintedVolumeQueue()
        for i in range(20):
            q.add(QueueOrder(str(i), "S:8039", "sell", price_i(row["AskPrice1"]),
                             2000, row["ns"], "exit", str(i)), row["AskLots1"] * 1000)
        stop = open_ns("20260724") + (12 * 60 + 47) * SECOND
        window = raw.filter((pl.col("ns") > row["ns"]) & (pl.col("ns") <= stop)
                             & (pl.col("FillLots") > 0)).sort("ns", "ChannelSeq")
        filled = 0
        for r in window.iter_rows(named=True):
            filled += sum(n for _, n in q.trade("S:8039", r["ns"], r["ChannelSeq"],
                                               price_i(r["FillPrice"]), r["FillLots"] * 1000))
        self.assertEqual(filled, 4000)
        self.assertEqual(len(q.orders), 18)


if __name__ == "__main__":
    unittest.main()
