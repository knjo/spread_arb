import unittest
from dataclasses import dataclass

from ..ev import ev
from ..ev.config import CostConfig


@dataclass(frozen=True)
class FakeCell:
    n: int
    p: float
    level: int = 0


class FakeLookup:
    def __init__(self, c0=0.4, c1=0.65, q=0.3, n=500):
        self.c0v, self.c1v, self.qv, self.n = c0, c1, q, n

    def lookup_c0(self, e, x, t_sec, k_days):
        return FakeCell(self.n, self.c0v)

    def lookup_c1(self, e, x, k_days):
        return FakeCell(self.n, self.c1v)

    def lookup_q(self, x, k_days, e=None):
        return FakeCell(self.n, self.qv)


H7 = ev.horizon("20260706", "20260715")   # 7 planned sessions, settlement at expiry close
CFG = CostConfig()


def quote(stream="S2", quote_ab=80.0, anchor=30.0, scale=20.0, t_sec=1800):
    return ev.Quote(stream, quote_ab, anchor, scale, (quote_ab - anchor) / scale, t_sec)


class BranchTest(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for K in (0, 1, 2, 5, 12):
            b = ev.branches(0.4, 0.65, 0.3, K)
            self.assertAlmostEqual(b.p_sd + b.p_on + b.p_never, 1.0, places=12)
            self.assertEqual(len(b.p_later), K)

    def test_expiry_today_has_no_tomorrow(self):
        b = ev.branches(0.4, 0.9, 0.9, 0)
        self.assertEqual((b.p_sd, b.p_later, b.p_never), (0.4, (), 0.6))

    def test_c1_not_below_c0_and_missing_defaults(self):
        b = ev.branches(0.5, 0.3, None, 3)
        self.assertEqual(b.p_later[0], 0.0)
        self.assertAlmostEqual(b.p_never, 0.5)      # q=0 -> everything left settles
        b = ev.branches(0.5, None, 0.5, 3)
        self.assertEqual(b.p_later[0], 0.0)
        self.assertAlmostEqual(b.p_later[1], 0.25)


class HorizonTest(unittest.TestCase):
    def test_horizon_uses_planned_calendar(self):
        self.assertEqual(H7.K, 7)
        self.assertEqual(H7.session_offsets, (1, 2, 3, 4, 7, 8, 9))
        self.assertEqual(H7.settle_offset, 9)
        self.assertEqual(ev.horizon("20260713", "20260715").K, 2)
        self.assertEqual(ev.horizon("20260715", "20260715").K, 0)


class EvalTest(unittest.TestCase):
    def test_c0_one_is_pure_same_day(self):
        v = ev.evaluate_exit(-0.5, quote(), H7, FakeLookup(c0=1.0, c1=1.0), CFG)
        d_in, d_out = CFG.d_in_bp("S2"), CFG.d_out_bp("S2")
        g_x = 80.0 - d_in - (30.0 - 0.5 * 20.0) - d_out
        self.assertAlmostEqual(v.ev_bp, g_x - 20.0)
        self.assertAlmostEqual(v.t_days, (15_600 - 1800) / 86_400)
        self.assertEqual(v.p_never, 0.0)

    def test_nothing_reaches_means_settlement(self):
        v = ev.evaluate_exit(-0.5, quote(), H7, FakeLookup(c0=0.0, c1=0.0, q=0.0), CFG)
        self.assertAlmostEqual(v.p_never, 1.0)
        self.assertAlmostEqual(v.ev_bp, 80.0 - CFG.d_in_bp("S2") - CFG.d_settle_bp() - 34.0)
        self.assertAlmostEqual(v.t_days, 9.0 + (15600 - 1800) / 86400)

    def test_settle_row(self):
        v = ev.evaluate_exit(ev.SETTLE, quote(), H7, None, CFG)
        self.assertEqual((v.x, v.p_never), (None, 1.0))
        self.assertAlmostEqual(v.t_days, 9.0 + (15600 - 1800) / 86400)
        self.assertAlmostEqual(v.ev_bp, 80.0 - CFG.d_in_bp("S2") - CFG.d_settle_bp() - 34.0)

    def test_gates_precede_ev(self):
        evals = ev.evaluate(quote(), H7, FakeLookup(), CFG)
        self.assertEqual(ev.choose(evals, CFG, quote(quote_ab=-5.0)).reason, "basis")
        self.assertEqual(ev.choose(evals, CFG, quote(anchor=-70.0)).reason, "anchor")
        self.assertEqual(ev.choose(evals, CostConfig(min_anchor_bp=-100.0), quote(anchor=-70.0)).reason, "ok")

    def test_monotone_in_quote_ab_and_d_in(self):
        lk = FakeLookup()
        lo = ev.evaluate_exit(-0.5, quote(quote_ab=60.0), H7, lk, CFG)
        hi = ev.evaluate_exit(-0.5, quote(quote_ab=90.0), H7, lk, CFG)
        self.assertLess(lo.ev_bp, hi.ev_bp)
        costly = CostConfig(d_in_base={"S1": 25.7, "S2": 30.0})
        self.assertLess(ev.evaluate_exit(-0.5, quote(), H7, lk, costly).ev_bp, ev.evaluate_exit(-0.5, quote(), H7, lk, CFG).ev_bp)

    def test_execution_floor_takes_max_not_sum(self):
        q = ev.Quote("S2", 80.0, 30.0, 20.0, 2.5, 1800, execution_floor_bp=20.0)
        v = ev.evaluate_exit(-0.5, q, H7, FakeLookup(), CFG)
        self.assertEqual(v.d_in, 20.0)
        q2 = ev.Quote("S2", 80.0, 30.0, 20.0, 2.5, 1800, execution_floor_bp=5.0)
        self.assertEqual(ev.evaluate_exit(-0.5, q2, H7, FakeLookup(), CFG).d_in, 8.9 + 3.0)

    def test_ev_min_floor_blocks_tiny_fast_trades(self):
        fast = ev.ExitEval(-0.25, 25.0, 0.99, 0.0, 0.01, 2.0, 0.05, 40.0, 1.7, 11.9, 3.0)
        slow = ev.ExitEval(-1.0, 10.0, 0.5, 0.4, 0.1, 12.0, 2.0, 6.0, 1.0, 11.9, 3.0)
        d = ev.choose([fast, slow], CostConfig(ev_min_bp=0.0))
        self.assertTrue(d.admit)
        self.assertEqual(d.best.x, -0.25)
        d = ev.choose([fast, slow], CostConfig(ev_min_bp=5.0))
        self.assertTrue(d.admit)
        self.assertEqual(d.best.x, -1.0)
        d = ev.choose([fast, slow], CostConfig(ev_min_bp=5.0, hurdle_bp_per_day=8.0))
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, "hurdle")

    def test_missing_cells_are_flagged_not_hidden(self):
        class Sparse(FakeLookup):
            def lookup_c1(self, e, x, k_days):
                return None

            def lookup_q(self, x, k_days, e=None):
                return None

        v = ev.evaluate_exit(-0.5, quote(), H7, Sparse(), CFG)
        self.assertEqual(v.fallback, "c1=c0,q=0")
        self.assertAlmostEqual(v.p_never, 1.0 - v.p_sd)


if __name__ == "__main__":
    unittest.main()
