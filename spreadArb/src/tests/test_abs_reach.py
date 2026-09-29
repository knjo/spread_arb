import unittest

import polars as pl

from ..ev import abs_reach, ev
from ..ev.config import CostConfig
from .test_ev import FakeLookup, H7, quote


def samples(rows):
    return pl.DataFrame(rows, schema={"Date": pl.String, "ValueCode": pl.String, "QuoteCode": pl.String,
                                      "settle_date": pl.String, "threshold": pl.Float64, "ret_sell": pl.Float64,
                                      "days_to_settle_td": pl.Int64, "days_to_settle_cal": pl.Int64,
                                      "natural_converge": pl.Boolean, "convergence_days_td": pl.Int64,
                                      "convergence_days_cal": pl.Int64, "converge_date": pl.String, "status": pl.String})


def row(date, thr, k, natural, days, conv_date, settle="20260715"):
    return dict(Date=date, ValueCode="2330", QuoteCode="CDFG6", settle_date=settle, threshold=thr, ret_sell=thr + 0.001,
                days_to_settle_td=k, days_to_settle_cal=int(k * 1.4), natural_converge=natural,
                convergence_days_td=days, convergence_days_cal=int(days * 1.4), converge_date=conv_date, status="ok")


class AbsReachTest(unittest.TestCase):
    def setUp(self):
        rows = []
        # threshold 1%, 7 trading days to settle: 40 rows converge day 0, 30 day 1, 20 day 3, 10 settle
        for i in range(40):
            rows.append(row("20260601", 0.01, 7, True, 0, "20260601"))
        for i in range(30):
            rows.append(row("20260601", 0.01, 7, True, 1, "20260602"))
        for i in range(20):
            rows.append(row("20260601", 0.01, 7, True, 3, "20260604"))
        for i in range(10):
            rows.append(row("20260601", 0.01, 7, False, 7, None))
        self.table = abs_reach.AbsTable.fit(samples(rows))

    def test_distribution_and_lookup(self):
        cell = self.table.lookup(120.0, 7)            # 120 bp -> 1% threshold; 7 td -> bucket 6-10
        self.assertEqual(cell.n, 100)
        self.assertAlmostEqual(cell.p_day[0], 0.4)
        self.assertAlmostEqual(cell.p_day[1], 0.3)
        self.assertAlmostEqual(cell.p_day[3], 0.2)
        self.assertAlmostEqual(cell.p_settle, 0.1)
        self.assertIsNone(self.table.lookup(40.0, 7))   # below the lowest threshold: no absolute route
        self.assertEqual(self.table.lookup(120.0, 20).level, 1)   # 16+ bucket has no samples: threshold-only pool
        pooled = abs_reach.AbsTable.fit(samples([row("20260601", 0.01, k, True, 0, "20260601") for k in (1, 4, 8, 12, 20)] * 7), min_n=30)
        self.assertEqual(pooled.lookup(120.0, 20).level, 1)
        self.assertEqual(self.table.lookup(80.0, 7).key[0], 100.0)   # nearest threshold, not the floor

    def test_as_of_uses_seasoned_entries_and_censors_unknowns(self):
        rows = ([row("20260601", 0.01, 7, True, 3, "20260604")] * 40          # converged on day 3
                + [row("20260601", 0.01, 7, False, 7, None)] * 30             # held to settlement 7/15
                + [row("20260601", 0.01, 7, True, 20, "20260630")] * 30)      # natural convergence after MAX_J
        # 6/3: the 6/1 entries are not yet MAX_J + 1 sessions old -> no table at all
        self.assertIsNone(abs_reach.AbsTable.fit(samples(rows), as_of="20260603").lookup(120.0, 7))
        # 6/26: seasoned; day-3 convergence observable; the 7/15 settlement and the 6/30 late convergence are
        # not known yet -> both counted as held to settlement (conservative)
        cell = abs_reach.AbsTable.fit(samples(rows), as_of="20260626").lookup(120.0, 7)
        self.assertEqual(cell.n, 100)
        self.assertAlmostEqual(cell.p_day[3], 0.4)
        self.assertAlmostEqual(cell.p_later, 0.0)
        self.assertAlmostEqual(cell.p_settle, 0.6)
        # 7/20: every outcome known -> late convergence moves to p_later, settlement stays
        cell = abs_reach.AbsTable.fit(samples(rows), as_of="20260720").lookup(120.0, 7)
        self.assertAlmostEqual(cell.p_later, 0.3)
        self.assertAlmostEqual(cell.p_settle, 0.3)

    def test_absolute_route_ev_and_choice(self):
        cfg = CostConfig()
        q = quote(quote_ab=131.9, anchor=120.0)           # ab_eff = 120 after d_in 11.9; residual only 12 bp
        v = ev.evaluate_absolute(q, H7, self.table, cfg)
        g_x, g_0 = 120.0 - cfg.d_out_bp("S2"), 120.0
        expected = (0.4 * (g_x - 20) + 0.3 * (g_x - 34) + 0.2 * (g_x - 34) + 0.1 * (g_0 - cfg.d_settle_bp() - 34))
        self.assertAlmostEqual(v.ev_bp, expected, places=6)
        self.assertEqual(v.route, "absolute")
        self.assertAlmostEqual(v.p_never, 0.1)
        evals = ev.evaluate(q, H7, FakeLookup(c0=0.05, c1=0.08, q=0.02), cfg, abs_table=self.table)
        d = ev.choose(evals, cfg, q)
        self.assertTrue(d.admit)
        self.assertEqual(d.best.route, "absolute")       # residual route is hopeless at 12 bp; absolute carries it
        no_abs = ev.choose(ev.evaluate(q, H7, FakeLookup(c0=0.05, c1=0.08, q=0.02), cfg), cfg, q)
        self.assertNotEqual(no_abs.best.route if no_abs.best else None, "absolute")


if __name__ == "__main__":
    unittest.main()
