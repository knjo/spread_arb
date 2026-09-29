import unittest
from pathlib import Path

import numpy as np
import polars as pl

from ..backtest.policy import PolicyConfig, exit_target_bp
from ..backtest.legacy_replay import Position, Replay, _active_start
from ..common.paths import MAKER_WITHDRAW_SECOND, SECOND, open_ns
from ..ev.ev import ExitEval

DAY = "20260706"
T0 = open_ns(DAY)
MS = 1_000_000


def ns(sec: float) -> int:
    return T0 + int(sec * SECOND)


def exits_table(rows):
    """rows: dict per E1 row with seconds-from-open for the time columns; None stays None."""
    cols = ["quote_ns", "price", "level", "quote_ab", "t_fill_ns", "t_gate_ns", "hedge_ns", "hedge_vwap", "actual_ab",
            "t_above_0", "t_above_5", "t_above_10", "t_above_20", "t_above_30"]
    time_cols = {"quote_ns", "t_fill_ns", "t_gate_ns", "hedge_ns", "t_above_0", "t_above_5", "t_above_10", "t_above_20",
                 "t_above_30"}
    out = {}
    for c in cols:
        vals = [r.get(c) for r in rows]
        if c in time_cols:
            out[c] = np.array([np.nan if v is None else float(ns(v)) for v in vals], dtype=float)
            if c == "quote_ns":
                out[c] = out[c].astype(np.int64)
        else:
            out[c] = np.array([np.nan if v is None else v for v in vals], dtype=float)
    return {"2330": out}


def position(target_bp, quote_day=DAY, hedge_sec=600.0, price=1_000_000, fut=1_005_000, shares=2000):
    return Position("S1/x/2330/1", "2330", "CDFG6", "S1", "20260715", shares, "residual", -1.0, target_bp, 30.0, 10.0,
                    2.0, 0.5, quote_day, ns(599.0), 60.0, 30.0, 10.0, price, ns(599.5), ns(hedge_sec),
                    price * shares, fut, 50.0)


def replay(**kw):
    cfg = PolicyConfig(**kw)
    return Replay(cfg, Path("/tmp/spreadarb-test-unused"), samples=pl.DataFrame())


class ActiveStart(unittest.TestCase):
    def test_group_start_with_duplicate_timestamps(self):
        qn = np.array([10, 20, 20, 30], dtype=np.int64)
        self.assertEqual(_active_start(qn, 5), 0)
        self.assertEqual(_active_start(qn, 20), 1)
        self.assertEqual(_active_start(qn, 25), 1)     # inside the 20-group
        self.assertEqual(_active_start(qn, 30), 3)
        self.assertEqual(_active_start(qn, 99), 3)


class PlanExit(unittest.TestCase):
    def test_guard_already_fired_row_is_skipped_and_loop_terminates(self):
        # row 0: basis <= target but its guard fired at quote time; row 1 (same product, later) fills cleanly
        ex = exits_table([
            dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=700.2, t_gate_ns=None,
                 hedge_ns=700.25, hedge_vwap=1_015_000, actual_ab=49.5, t_above_0=700.0, t_above_5=700.0,
                 t_above_10=700.0, t_above_20=700.0, t_above_30=700.0),
            dict(quote_ns=900.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=930.0, t_gate_ns=None,
                 hedge_ns=930.05, hedge_vwap=1_009_000, actual_ab=-9.9, t_above_0=None, t_above_5=None,
                 t_above_10=None, t_above_20=None, t_above_30=None),
        ])
        r = replay()
        plan = r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0))
        self.assertIsNotNone(plan)
        j, q_t, t_fill, hedge_ns, price, hedge_vwap, qab, realized, cancel_at, route, double = plan
        self.assertEqual((j, route, double), (1, "E1", False))
        self.assertEqual(q_t, ns(900.0) + r.cfg.place_ns)
        self.assertEqual(t_fill, ns(930.0))
        self.assertIsNone(cancel_at)

    def test_cancelled_quote_resumes_from_cancel_effective_and_never_revisits(self):
        # one long segment whose guard fires at 800 s, fill only at 850 s (after the cancel): not filled today
        ex = exits_table([
            dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=850.0, t_gate_ns=None,
                 hedge_ns=850.05, hedge_vwap=1_015_000, actual_ab=49.5, t_above_0=800.0, t_above_5=800.0,
                 t_above_10=800.0, t_above_20=800.0, t_above_30=800.0),
        ])
        r = replay()
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))

    def test_race_fill_inside_cancel_latency_counts(self):
        ex = exits_table([
            dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=800.03, t_gate_ns=None,
                 hedge_ns=800.08, hedge_vwap=1_020_000, actual_ab=99.0, t_above_0=800.0, t_above_5=800.0,
                 t_above_10=800.0, t_above_20=800.0, t_above_30=800.0),
        ])
        r = replay()
        plan = r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0))
        self.assertIsNotNone(plan)
        self.assertEqual(plan[8], ns(800.0))          # cancel_at
        self.assertGreater(plan[2], plan[8])          # filled after the guard: a race

    def test_tolerance_picks_the_rise_bucket(self):
        # target 0, quote basis -12, tol 5 -> need 17 -> rise bucket 10 (fires at 900 s) not 20 (never)
        ex = exits_table([
            dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-12.0, t_fill_ns=950.0, t_gate_ns=None,
                 hedge_ns=950.05, hedge_vwap=1_010_000, actual_ab=0.0, t_above_0=750.0, t_above_5=800.0,
                 t_above_10=900.0, t_above_20=None, t_above_30=None),
        ])
        self.assertIsNone(replay(exit_tol_bp=5.0).plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))
        self.assertIsNotNone(replay(exit_tol_bp=10.0).plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))

    def test_no_row_at_or_below_target(self):
        ex = exits_table([dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=20.0, t_fill_ns=701.0,
                               t_gate_ns=None, hedge_ns=701.05, hedge_vwap=1_010_000, actual_ab=0.0)])
        self.assertIsNone(replay().plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))
        self.assertIsNone(replay().plan_exit(position(target_bp=None), ex, DAY, T0, ns(600.0)))

    def test_cursor_before_quote_start_and_after_withdraw(self):
        r = replay()
        early = exits_table([dict(quote_ns=100.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=400.0,
                                  t_gate_ns=None, hedge_ns=400.05, hedge_vwap=1_010_000, actual_ab=0.0)])
        # E1 rests in the spot queue: a row that started before the cursor (09:05) cannot be joined late
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), early, DAY, T0, T0))
        fast = exits_table([dict(quote_ns=310.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=310.02,
                                 t_gate_ns=None, hedge_ns=310.07, hedge_vwap=1_010_000, actual_ab=0.0)])
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), fast, DAY, T0, T0))      # filled before the quote is live
        ok = exits_table([dict(quote_ns=310.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=400.0,
                               t_gate_ns=None, hedge_ns=400.05, hedge_vwap=1_010_000, actual_ab=0.0)])
        plan = r.plan_exit(position(target_bp=0.0), ok, DAY, T0, T0)
        self.assertEqual(plan[1], ns(310.0) + r.cfg.place_ns)
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), ok, DAY, T0, T0 + MAKER_WITHDRAW_SECOND * SECOND))

    def test_two_routes_pick_the_earlier_fill_and_flag_double_risk(self):
        row = dict(price=1_010_000, level=0, quote_ab=-5.0, t_gate_ns=None, hedge_vwap=1_009_000, actual_ab=-9.9)
        e1 = exits_table([dict(row, quote_ns=700.0, t_fill_ns=800.0, hedge_ns=800.05)])
        e2 = exits_table([dict(row, quote_ns=650.0, t_fill_ns=760.0, hedge_ns=760.05)])
        r = replay(exit_routes=("E1", "E2"))
        plan = r.plan_exit(position(target_bp=0.0), {"E1": e1, "E2": e2}, DAY, T0, ns(600.0))
        self.assertEqual((plan[9], plan[2], plan[10]), ("E2", ns(760.0), False))
        e1b = exits_table([dict(row, quote_ns=700.0, t_fill_ns=760.03, hedge_ns=760.08)])
        r = replay(exit_routes=("E1", "E2"))     # fresh replay: the 760 s E2 fill above is consumed in `r`
        plan = r.plan_exit(position(target_bp=0.0), {"E1": e1b, "E2": e2}, DAY, T0, ns(600.0))
        self.assertEqual((plan[9], plan[10]), ("E2", True))          # E1 would also fill inside the cancel latency

    def test_quote_gate_blocks_and_cancels(self):
        class FakeGates:
            def __init__(self, bad_from):
                self.bad_from = bad_from

            def next_bad(self, leg, vc, t):
                return t if t >= self.bad_from else self.bad_from

        ex = exits_table([dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-5.0, t_fill_ns=800.0, t_gate_ns=None,
                               hedge_ns=800.05, hedge_vwap=1_009_000, actual_ab=-9.9)])
        r = replay()
        r.gates = FakeGates(ns(750.0))           # gate turns bad at 750 s: the quote is cancelled before the 800 s fill
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))
        r.gates = FakeGates(ns(799.98))          # turns bad 20 ms before the fill: inside the cancel latency -> race fill
        plan = r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0))
        self.assertEqual(plan[8], ns(799.98))
        r.gates = FakeGates(ns(100.0))           # already bad when we want to quote
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))


class ExitFillConsumedOnce(unittest.TestCase):
    def test_second_position_queues_behind_the_first(self):
        row = dict(price=1_010_000, level=0, quote_ab=-5.0, t_gate_ns=None, hedge_vwap=1_009_000, actual_ab=-9.9)
        ex = exits_table([dict(row, quote_ns=700.0, t_fill_ns=800.0, hedge_ns=800.05),
                          dict(row, quote_ns=900.0, t_fill_ns=950.0, hedge_ns=950.05)])
        r = replay()
        p1, p2 = position(target_bp=0.0), position(target_bp=0.0)
        self.assertEqual(r.plan_exit(p1, ex, DAY, T0, ns(600.0))[2], ns(800.0))
        self.assertEqual(r.plan_exit(p2, ex, DAY, T0, ns(600.0))[2], ns(950.0))     # 800 s fill already taken
        self.assertIsNone(r.plan_exit(position(target_bp=0.0), ex, DAY, T0, ns(600.0)))


class ClosePnl(unittest.TestCase):
    def test_four_legs_and_fees(self):
        r = replay()
        p = position(target_bp=0.0)               # bought 2000 @ 100.00, sold fut @ 100.50
        r.open_ids.add(p.id)
        r.committed = p.notional_twd
        r.close(p, DAY, "maker_exit", 1_020_000 * 2000, 1_015_000, ns(3600.0))
        self.assertAlmostEqual(p.pnl_spot, 2000 * 2.0)            # +2.00 per share
        self.assertAlmostEqual(p.pnl_fut, 2000 * (-1.0))          # sold 100.50, bought 101.50
        self.assertAlmostEqual(p.fees, 200_000 * 20 / 1e4)        # same day: 20 bp of notional
        self.assertAlmostEqual(p.pnl_net, 4000 - 2000 - 400)
        self.assertAlmostEqual(p.exit_realized_ab, (101.5 / 102.0 - 1) * 1e4)
        self.assertEqual(r.committed, 0.0)
        self.assertNotIn(p.id, r.open_ids)
        q = position(target_bp=0.0, quote_day="20260703")
        r.close(q, DAY, "settlement", 1_000_000 * 2000, 1_000_000, ns(15600))
        self.assertAlmostEqual(q.fees, 200_000 * 34 / 1e4)        # overnight fee


class ExitTarget(unittest.TestCase):
    def test_routes(self):
        ev = ExitEval.__new__(ExitEval)
        for route, x, expect in (("absolute", 0.0, 0.0), ("settle", None, None), ("residual", -0.5, 30.0 - 5.0)):
            object.__setattr__(ev, "route", route)
            object.__setattr__(ev, "x", x)
            self.assertEqual(exit_target_bp(ev, 30.0, 10.0), expect)


class AbsTargetMode(unittest.TestCase):
    def test_higher_of_zero_and_q_target(self):
        def ev_(route, x, ev_bp, score):
            v = ExitEval.__new__(ExitEval)
            for k, val in dict(route=route, x=x, ev_bp=ev_bp, score=score, t_days=1.0).items():
                object.__setattr__(v, k, val)
            return v
        best = ev_("absolute", 0.0, 40.0, 12.0)
        evals = (best, ev_("residual", 0.0, -3.0, -3.0), ev_("residual", -1.0, 6.0, 4.0))
        self.assertEqual(exit_target_bp(best, 50.0, 20.0, evals, "zero"), 0.0)
        self.assertEqual(exit_target_bp(best, 50.0, 20.0, evals, "max_q"), 30.0)          # best residual x = -1
        self.assertEqual(exit_target_bp(best, 10.0, 20.0, evals, "max_q"), 0.0)           # Q target below 0 -> 0
        losing = (best, ev_("residual", 0.0, -3.0, -3.0))
        self.assertEqual(exit_target_bp(best, 50.0, 20.0, losing, "max_q"), 50.0)
        self.assertEqual(exit_target_bp(best, 50.0, 20.0, losing, "max_q_pos"), 0.0)      # Q exit would lose money
        self.assertEqual(exit_target_bp(best, 50.0, 20.0, (), "max_q"), 0.0)


class AdmittedScoresUseBaseHurdle(unittest.TestCase):
    def test_raised_hurdle_does_not_truncate_the_signal_set(self):
        from ..backtest.policy import Decider
        from ..ev.ev import Decision
        def ev_(score):
            v = ExitEval.__new__(ExitEval)
            for k, val in dict(route="absolute", x=0.0, ev_bp=score * 2, score=score, t_days=2.0).items():
                object.__setattr__(v, k, val)
            return v
        dec = Decider("20260706", None, None, PolicyConfig())
        dec.cache = {1: Decision(True, "ok", ev_(50.0), ()), 2: Decision(False, "hurdle", ev_(9.0), ()),   # 9 > base 5.82: rejected only by a raised hurdle
                     3: Decision(False, "hurdle", ev_(2.0), ()), 4: Decision(False, "basis", ev_(80.0), ()), 5: Decision(False, "no_estimate", None, ())}
        self.assertEqual(sorted(dec.admitted_scores()), [9.0, 50.0])


class DynamicHurdle(unittest.TestCase):
    def test_quantile_of_previous_days_and_cap_condition(self):
        base = PolicyConfig().cost.hurdle_bp_per_day
        r = replay(dyn_q=0.8, dyn_window=1)
        self.assertEqual(r.dynamic_hurdle(), (base, 0))                       # no history yet
        r.signal_hist.append(("d1", [1.0, 2.0, 3.0, 4.0, 10.0], 20e6))
        h, n = r.dynamic_hurdle()
        self.assertEqual(n, 5)
        self.assertAlmostEqual(h, max(base, 4.0 + 0.2 * 6.0))                 # numpy linear quantile q80 = 5.2 -> above base
        r.signal_hist.append(("d2", [0.1, 0.2], 1e6))
        self.assertEqual(r.dynamic_hurdle(), (base, 2))                       # q80 below base -> base
        r2 = replay(dyn_q=0.8, dyn_window=2)
        r2.signal_hist = list(r.signal_hist)
        self.assertEqual(r2.dynamic_hurdle()[1], 7)                            # two-day window pools both days
        r3 = replay(dyn_q=0.8, dyn_window=1, dyn_cap_frac=0.8)
        r3.signal_hist = [("d1", [10.0, 20.0, 30.0], 10e6)]
        self.assertEqual(r3.dynamic_hurdle()[0], base)                         # capital was not binding -> base
        r3.signal_hist = [("d1", [10.0, 20.0, 30.0], 19e6)]
        self.assertGreater(r3.dynamic_hurdle()[0], base)


if __name__ == "__main__":
    unittest.main()
