"""Regression contracts for the execution flaws found in the 2026-09-22 audit."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from maker.src.ev_lookup_cost.execution import Book, QueueOrder
from ..backtest.causal_market import Contract, Timeline
from ..backtest.causal_replay import Actor, Replay, Cycle, DELAY
from ..backtest.policy import PolicyConfig, Decider, StreamingDecider, decide_row
from ..common.books import Prints
from ..ev.ev import Decision, ExitEval, evaluate_exit, horizon, Quote
from ..ev.config import CostConfig
from .test_ev import FakeLookup

T = 1_000_000_000
C = Contract("2330", "FUT6", 2000, "20260715", 500000, 510000)


class FakeMarket:
    def __init__(self, trades=(), guard=T + 2_000_000_000, spot_bid=500000, fut_qty=10):
        self.start, self.end, self.withdraw, self.day = 0, 20*T, 19*T, "20260706"
        self.contracts = {C.vc: C}
        self.corporate_block = set()
        self.limits = {C.vc: (450000, 550000)}
        self.timelines = {C.qc: SimpleNamespace(guard=lambda *a: guard, first_exit=lambda *a: None)}
        self.b = {"S:2330": Book("S:2330", 0, 1, ((spot_bid, 100000),), ((510000, 100000),), True),
                  "F:FUT6": Book("F:FUT6", 0, 1, ((520000, fut_qty),), ((530000, fut_qty),), True)}
        self.prints = {}
        for instrument in {r[0] for r in trades}:
            rs = [r for r in trades if r[0] == instrument]
            self.prints[instrument] = Prints(instrument, np.array([r[1] for r in rs], dtype=np.int64),
                np.arange(len(rs), dtype=np.int64), np.array([r[2] for r in rs]), np.array([r[3] for r in rs]))
        self.marks = {"S:2330": (510000, "fixture"), "F:FUT6": (520000, "fixture")}

    def book(self, instrument, ns):
        return self.b.get(instrument)

    def next_book(self, instrument, ns):
        return None


def row(stream="S1", ns=T):
    return dict(stream=stream, vc=C.vc, qc=C.qc, expiry=C.expiry, quote_ns=ns, quote_second=1800,
                price=505000 if stream == "S1" else 525000, quote_ab=100.0, anchor=30.0,
                scale=20.0, scale_raw=20.0, resid_mid_bp=40.0, e_norm=2.0,
                tick_bp_hedge=20.0, spot_a1=510000, execution_floor_bp=0.0)


def decision():
    v = ExitEval(None, None, 0, 0, 1, 50, 1, 50, 40, 10, 3, route="settle")
    return Decision(True, "ok", v, (v,))


def engine(market, **cfg):
    # Historical full-reservation regressions opt in; new policy tests pass False.
    cfg.setdefault("reserve_on_submit", True)
    actor = Actor("A", PolicyConfig(**cfg))
    replay = Replay(Path("unused"), [actor])
    actor.begin(market, replay)
    return actor, replay


class CausalExecutionTest(unittest.TestCase):
    def test_summary_keeps_missing_equity_explicit(self):
        import polars as pl
        import json
        from ..backtest.causal_replay import build_summary
        with TemporaryDirectory() as temp:
            root=Path(temp)
            pl.DataFrame(dict(day=[20260706,20260707],equity_twd=[0.,None],filled=[0,0],rollbacks=[0,0],
                              open_end=[0,1],unhedged_end=[0,1],missing_marks=[0,1],
                              committed_peak=[0.,100.],committed_end=[0.,100.])).write_csv(root/'A_daily.csv')
            build_summary(root)
            summary=json.loads((root/'summary.json').read_text())['A']
            self.assertIsNone(summary['total_pnl_twd'])
            self.assertIsNone(summary['mtm_drawdown_twd'])
            self.assertEqual(summary['missing_marks'],1)

    def test_unfilled_order_reserves_until_cancel(self):
        a, replay = engine(FakeMarket())
        a.offer(row(), decision())
        self.assertEqual(a.ledger.committed_cents, 10100000)
        replay.drain(T + 2*T + DELAY - 1)
        self.assertEqual(a.ledger.committed_cents, 10100000)
        replay.drain(T + 2*T + DELAY)
        self.assertEqual(a.ledger.committed_cents, 0)

    def test_pre_live_volume_does_not_fill_and_partial_rolls_back_at_current_bid(self):
        # Quote is inside, so no external queue. Only one board lot prints after live.
        m = FakeMarket([("S:2330", T+10_000_000, 505000, 1000),
                        ("S:2330", T+60_000_000, 505000, 1000)], spot_bid=490000)
        a, replay = engine(m)
        a.offer(row(), decision())
        replay.drain(m.end)
        p = next(iter(a.cycles.values()))
        self.assertEqual(p.maker_qty, 1000)
        self.assertIsNone(p.hedge_ns)
        self.assertEqual(p.close_kind, "rollback")
        self.assertAlmostEqual(p.gross, -1500.0)
        self.assertEqual(p.spot_sell_cash, 490000*1000)
        self.assertEqual(a.ledger.committed_cents, 0)

    def test_guard_during_placement_still_allows_cancel_race(self):
        m = FakeMarket([("S:2330", T+55_000_000, 505000, 2000)], guard=T+10_000_000)
        a, replay = engine(m)
        a.offer(row(), decision())
        replay.drain(T+200_000_000)
        p = next(iter(a.cycles.values()))
        self.assertTrue(p.entry_race)
        self.assertEqual(p.state, "paired")

    def test_failed_hedge_keeps_exposure_and_reservation(self):
        m = FakeMarket([("S:2330", T+60_000_000, 505000, 2000)], fut_qty=0)
        a, replay = engine(m)
        a.offer(row(), decision())
        replay.drain(m.end)
        p = next(iter(a.cycles.values()))
        self.assertEqual(p.state, "entry_hedge")
        self.assertEqual(p.sq, 2000)
        self.assertEqual(len(p.tasks), 1)
        self.assertGreater(a.ledger.committed_cents, 0)

    def test_future_entry_reserves_upper_limit_not_future_hedge_cash(self):
        a, _ = engine(FakeMarket())
        a.offer(row("S2"), decision())
        self.assertEqual(a.ledger.committed_cents, 550000 * 2000 // 100)

    def test_two_orders_share_one_print_quantity(self):
        a, _ = engine(FakeMarket())
        a.queue.add(QueueOrder("one", "S:2330", "sell", 510000, 2000, T, "E1", "p1"), 1000)
        a.queue.add(QueueOrder("two", "S:2330", "sell", 510000, 2000, T+1, "E1", "p2"), 1000)
        filled = a.queue.trade("S:2330", T+100, 1, 510000, 3000)
        self.assertEqual(sum(q for _, q in filled), 2000)
        self.assertEqual([o.id for o, _ in filled], ["one"])
        self.assertEqual(a.queue.orders["two"].remaining, 2000)

    def test_hedge_depth_consumed_once(self):
        m = FakeMarket(fut_qty=1)
        a, _ = engine(m)
        b = m.book("F:FUT6", T)
        self.assertIsNotNone(a.depth.take(b, "sell", 1, T))
        self.assertIsNone(a.depth.take(b, "sell", 1, T+1))

    def test_duplicate_hedge_quantities_keep_independent_arrival_times(self):
        a, _ = engine(FakeMarket())
        a.offer(row(), decision())
        p = next(iter(a.cycles.values()))
        a.hedge_task(p, "double_rollback", "S:2330", "buy", 1000, 2*T)
        a.hedge_task(p, "double_rollback", "S:2330", "buy", 1000, 3*T)
        self.assertEqual(len(p.tasks), 2)
        self.assertEqual(sorted(a.hedges.values()), [2*T, 3*T])

    def test_double_exit_books_losing_fill_and_actual_buyback(self):
        trades = [("S:2330", T+60_000_000, 505000, 2000),
                  ("S:2330", 2*T+60_000_000, 505000, 2000),
                  ("F:FUT6", 2*T+70_000_000, 525000, 1)]
        a, replay = engine(FakeMarket(trades))
        a.offer(row(), decision())
        replay.drain(T+200_000_000)
        p = next(iter(a.cycles.values()))
        a.new_order(p, "E1", 505000, 2*T)
        a.new_order(p, "E2", 525000, 2*T)
        replay.drain(4*T)
        self.assertTrue(p.exit_double_risk)
        self.assertEqual((p.sq, p.fq), (0, 0))
        self.assertEqual(p.close_kind, "maker_exit")
        self.assertAlmostEqual(p.gross, -3000.0)
        self.assertEqual(a.ledger.committed_cents, 0)
        self.assertTrue(any(r["purpose"] == "double_rollback" for r in a.cash))

    def test_expiry_missing_mark_keeps_explicit_pending_state(self):
        m = FakeMarket([("S:2330", T+60_000_000, 505000, 2000)])
        m.day = C.expiry
        del m.marks["S:2330"]
        a, replay = engine(m)
        a.offer(row(), decision())
        replay.drain(m.end)
        with TemporaryDirectory() as temp:
            stats = a.finish([], Path(temp))
        p = next(iter(a.cycles.values()))
        self.assertEqual(p.state, "settlement_pending")
        self.assertIsNone(stats["equity_twd"])
        self.assertEqual(stats["missing_marks"], 1)
        self.assertGreater(a.ledger.committed_cents, 0)

    def test_expiry_settles_original_contract_when_entry_grid_is_absent(self):
        m = FakeMarket([("S:2330", T+60_000_000, 505000, 2000)])
        m.day = C.expiry
        a, replay = engine(m)
        a.offer(row(), decision())
        replay.drain(m.end)
        m.timelines = {}
        with TemporaryDirectory() as temp:
            stats = a.finish([], Path(temp))
        self.assertEqual(stats["settled"], 1)
        self.assertEqual(stats["open_end"], 0)
        f = [r for r in a.cash if r["purpose"] == "settlement" and r["instrument"].startswith("F:")]
        self.assertEqual([r["instrument"] for r in f], ["F:FUT6"])


class PolicyRegressionTest(unittest.TestCase):
    def test_cli_explicit_defaults_override_preset(self):
        from ..backtest.replay import parse_args
        _, cfg = parse_args(["--preset", "B", "--exit-routes", "E1", "--abs-target", "zero", "--dyn-cap-frac", "0"])
        self.assertEqual(cfg.exit_routes, ("E1",))
        self.assertEqual(cfg.abs_target_mode, "zero")
        self.assertAlmostEqual(cfg.dyn_cap_frac, 0.)

    def test_cache_keeps_negative_anchor_and_distinct_e_separate(self):
        cfg = PolicyConfig()
        dec = Decider("20260706", FakeLookup(), None, cfg)
        first = row()
        first["anchor"] = 0.1
        self.assertTrue(dec.decide(first).admit)
        bad = {**first, "anchor": -0.1}
        self.assertEqual(dec.decide(bad).reason, "anchor")
        dec.decide({**first, "e_norm": 5.0})
        self.assertEqual(dec.calls, 3)

    def test_explicit_wide_scale_excluded(self):
        r = {**row(), "scale": None, "scale_raw": 79.9}
        self.assertEqual(decide_row(r, "20260706", FakeLookup(), None, PolicyConfig()).reason, "scale_excluded")

    def test_expiry_day_time_and_fee(self):
        q = Quote("S2", 80, 30, 20, 2, 3600)
        cfg = CostConfig()
        h = horizon("20260715", "20260715")
        v = evaluate_exit(None, q, h, None, cfg)
        self.assertAlmostEqual(v.t_days, (15600-3600)/86400)
        self.assertAlmostEqual(v.ev_bp, 80-cfg.d_in_bp("S2")-cfg.d_settle_bp()-20)

    def test_signal_population_retains_products_when_streaming_cache_expires(self):
        d = StreamingDecider("20260706", FakeLookup(), None, PolicyConfig())
        r = row()
        d.decide(r)
        d.decide(r)
        d.decide({**r, "vc": "2317", "qc": "OTHER"})
        d.decide({**r, "quote_second": 1801})
        self.assertEqual(len(d.admitted_scores()), 3)
        self.assertEqual(len(d.cache), 1)


class TimelineRegressionTest(unittest.TestCase):
    def timeline(self, anchor=30., shallow=False, narrow=False):
        from .test_points_s2 import books, prints, T0
        from ..common.paths import CLOSE_SECOND, SECOND, MAKER_WITHDRAW_SECOND
        c = Contract("2330", "CDFG6", 2000, "20260715", 400000, 400000)
        spot_rows = [(0, 400000, 5000, 400500, 20000, True)]
        if shallow:
            spot_rows.append((350., 400000, 5000, 400500, 1000, True))
        fut_rows = [(0, 403000, 3, 404500, 2, True)]
        if narrow:
            fut_rows.append((350., 403500, 3, 404000, 2, True))
        market = SimpleNamespace(start=T0, end=T0+CLOSE_SECOND*SECOND,
            withdraw=T0+MAKER_WITHDRAW_SECOND*SECOND,
            books={"S:2330": books("S:2330", spot_rows), "F:CDFG6": books("F:CDFG6", fut_rows)},
            prints={"F:CDFG6": prints("F:CDFG6", [(400.,403500,1),(600.,403500,1)])})
        grid = SimpleNamespace(anchor=np.full(CLOSE_SECOND,anchor), mid=np.full(CLOSE_SECOND,100.))
        return Timeline(market,c,grid,{"scale":20.,"scale_raw":20.},PolicyConfig()), T0, SECOND

    def test_absolute_entry_not_dropped_by_residual_prefilter(self):
        tl, _, _ = self.timeline(anchor=85.)
        first = tl.row("S2",int(tl.candidates["S2"][0]))
        self.assertGreater(first["quote_ab"],50.)
        self.assertLess(first["quote_ab"]-first["anchor"],10.)

    def test_exit_coverage_after_entry_cutoff_and_repeated_prints(self):
        tl, start, second = self.timeline()
        self.assertIn(start+600*second,tl.ns)
        self.assertEqual(tl.first_exit("E2",100.,start+14001*second),start+14001*second)
        for indices in tl.candidates.values():
            self.assertTrue(np.all(tl.sec[indices] < 14000))

    def test_depth_must_hold_while_s2_order_rests(self):
        tl, start, second = self.timeline(shallow=True)
        trigger = tl.guard("S2",404000,None,start+300*second,PolicyConfig())
        self.assertEqual(trigger,start+350*second)

    def test_resting_guard_uses_submitted_limit_not_new_inside_price(self):
        tl, start, second = self.timeline(narrow=True)
        trigger = tl.guard("S2",404000,None,start+300*second,PolicyConfig())
        self.assertEqual(trigger,tl.withdraw)


if __name__ == "__main__":
    unittest.main()
