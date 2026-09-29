import unittest

import numpy as np
import polars as pl

from ..common.grid import ProductDay
from ..common.paths import CLOSE_SECOND, MAKER_WITHDRAW_SECOND, QUOTE_START_SECOND
from ..ev import reach


def ou_product(day, vc, rng, *, qc="XXFG6", expiry="20260715", theta=0.002, sd=3.0, start=40.0):
    resid = np.empty(CLOSE_SECOND)
    resid[0] = start
    for t in range(1, CLOSE_SECOND):
        resid[t] = resid[t - 1] * (1 - theta) + rng.normal(0.0, sd)
    anchor = np.full(CLOSE_SECOND, 30.0)
    mid = anchor + resid
    return ProductDay(day, vc, qc, expiry, mid, mid + 8.0, anchor, np.ones(CLOSE_SECOND, dtype=bool))


class FactsTest(unittest.TestCase):
    def test_min_today_matches_brute_force_and_stops_at_withdraw(self):
        rng = np.random.default_rng(0)
        p = ou_product("20260706", "2330", rng)
        cols = reach.product_facts(p, None, None, None, None)
        for i in (0, 10, 200):
            t0 = int(cols["t0"][i])
            truth = p.buy[t0 + 1:MAKER_WITHDRAW_SECOND + 1].min() - p.anchor[t0]
            self.assertAlmostEqual(cols["min_today_bp"][i], truth)
        self.assertEqual(cols["d1_state"][0], "missing")
        self.assertEqual(cols["k_days"][0], 9)

    def test_next_sessions_same_contract_only(self):
        rng = np.random.default_rng(1)
        today = ou_product("20260706", "2330", rng)
        d1 = ou_product("20260707", "2330", rng)
        rolled = ou_product("20260708", "2330", rng, qc="XXFH6")
        cols = reach.product_facts(today, d1, "20260707", rolled, "20260708")
        self.assertEqual((cols["d1_state"][0], cols["d2_state"][0]), ("ok", "missing"))
        lo, hi = QUOTE_START_SECOND, MAKER_WITHDRAW_SECOND
        self.assertAlmostEqual(cols["min_d1_bp"][0], d1.buy[lo:hi + 1].min() - today.anchor[int(cols["t0"][0])])
        self.assertTrue(np.isnan(cols["min_d2_bp"][0]))

    def test_expiring_today_has_no_tomorrow(self):
        rng = np.random.default_rng(2)
        today = ou_product("20260715", "2330", rng)
        d1 = ou_product("20260716", "2330", rng)
        cols = reach.product_facts(today, d1, "20260716", None, None)
        self.assertEqual(cols["d1_state"][0], "expired")
        self.assertEqual(cols["k_days"][0], 0)

    def test_ineligible_sampling_points_are_skipped(self):
        rng = np.random.default_rng(3)
        p = ou_product("20260706", "2330", rng)
        p.eligible[:5000] = False
        cols = reach.product_facts(p, None, None, None, None)
        self.assertGreaterEqual(int(cols["t0"].min()), 5000)


def synthetic_facts(days, rng, n=400):
    frames = []
    for i, day in enumerate(days):
        resid = rng.uniform(10.0, 60.0, n)
        # reach probability decreases with entry level; mins are independent draws
        min_today = resid - rng.exponential(25.0, n)
        min_d1 = np.minimum(min_today, resid - rng.exponential(40.0, n))
        min_d2 = resid - rng.exponential(40.0, n)
        frames.append(pl.DataFrame({
            "day": [day] * n, "ValueCode": ["2330"] * n, "QuoteCode": ["XXFG6"] * n, "expiry": ["20260815"] * n,
            "t0": rng.integers(300, 14000, n).astype(np.int32), "resid_bp": resid, "anchor_bp": np.full(n, 30.0),
            "min_today_bp": min_today, "min_d1_bp": min_d1, "min_d2_bp": min_d2,
            "d1_day": [days[i + 1] if i + 1 < len(days) else None] * n,
            "d2_day": [days[i + 2] if i + 2 < len(days) else None] * n,
            "d1_state": ["ok" if i + 1 < len(days) else "missing"] * n,
            "d2_state": ["ok" if i + 2 < len(days) else "missing"] * n,
            "k_days": np.full(n, 30, dtype=np.int32),
        }, schema=reach.FACT_SCHEMA))
    return pl.concat(frames)


class FitTest(unittest.TestCase):
    days = ["20260601", "20260602", "20260603", "20260604", "20260605"]

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.facts = synthetic_facts(self.days, self.rng)
        self.scales = pl.DataFrame({"day": self.days, "ValueCode": ["2330"] * 5, "scale": [20.0] * 5})

    def test_as_of_guard(self):
        with self.assertRaises(AssertionError):
            reach.fit_from_facts(self.facts, self.scales, "20260605")

    def test_label_availability_by_horizon(self):
        table = reach.fit_from_facts(self.facts, self.scales, "20260606", boot=20, min_days=3)
        # today labels: all 5 days; tomorrow labels need d1_day < decision: days 1-4; q needs d2: days 1-3
        x_only = (("x",), (0.0,))
        self.assertEqual(table.c0[x_only].n, 2000)
        self.assertEqual(table.c1[x_only].n, 1600)
        self.assertLess(table.q[x_only].n, 1200)
        self.assertTrue(0.0 <= table.q[x_only].p <= 1.0)
        self.assertIn((("x", "k_b"), (0.0, 3)), table.q)     # pooled-over-e fallback exists (k=30 -> bucket 3)
        self.assertIsNotNone(table.lookup_q(0.0, 30, e=1.2))
        self.assertIsNotNone(table.lookup_q(0.0, 30))

    def test_reach_monotone_in_level_and_target(self):
        table = reach.fit_from_facts(self.facts, self.scales, "20260606", boot=20, min_days=3)
        for x in table.x_grid:
            ps = [table.c0.get((("e_b", "x"), (eb, x))) for eb in (0, 1, 2)]
            ps = [c.p for c in ps if c is not None]
            self.assertEqual(ps, sorted(ps, reverse=True))
        c = table.lookup_c0(1.2, 0.0, 1800, 30)
        d = table.lookup_c0(1.2, -1.0, 1800, 30)
        self.assertGreater(c.p, d.p)
        self.assertGreaterEqual(table.lookup_c1(1.2, 0.0, 30).p, c.p)

    def test_ladder_and_min_n(self):
        thin = synthetic_facts(self.days, self.rng, n=100)   # 500 rows over 12 full-key cells
        table = reach.fit_from_facts(thin, self.scales, "20260606", boot=20, min_n=10_000)
        self.assertIsNone(table.lookup_c0(1.2, 0.0, 1800, 30))
        table = reach.fit_from_facts(thin, self.scales, "20260606", boot=20, min_n=100)
        cell = table.lookup_c0(1.2, 0.0, 1800, 30)
        self.assertGreater(cell.level, 0)          # full (e,x,t,k) key is too thin
        self.assertEqual(cell.key[0], ("e_b", "x"))  # falls to (e_b, x)
        self.assertLessEqual(cell.ci_lo, cell.p)
        self.assertGreaterEqual(cell.ci_hi, cell.p)

    def test_full_mode_pools_every_session(self):
        table = reach.fit_from_facts(self.facts, self.scales, None, boot=20, min_days=3)
        self.assertEqual(table.decision_day, "full")
        x_only = (("x",), (0.0,))
        self.assertEqual(table.c0[x_only].n, 2000)
        self.assertEqual(table.c1[x_only].n, 1600)      # last day has no next session
        with self.assertRaises(AssertionError):
            reach.fit_from_facts(self.facts, self.scales, "20260601", boot=20)

    def test_min_days_falls_back(self):
        table = reach.fit_from_facts(self.facts, self.scales, "20260606", boot=20, min_days=10)
        cell = table.lookup_c0(1.2, 0.0, 1800, 30)
        self.assertIsNone(cell)                     # only 5 sessions anywhere in the ladder
        table = reach.fit_from_facts(self.facts, self.scales, "20260606", boot=20, min_days=5)
        self.assertIsNotNone(table.lookup_c0(1.2, 0.0, 1800, 30))

    def test_buckets(self):
        np.testing.assert_array_equal(reach.k_bucket([0, 1, 3, 4, 10, 11], (0, 3, 10)), [0, 1, 1, 2, 2, 3])
        np.testing.assert_array_equal(reach.k_bucket([0, 1, 2, 3, 4, 10, 21]), [0, 1, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(reach.bucket([0.99, 1.0, 1.5, 2.5, 4.0, 9.0], reach.E_EDGES), [0, 1, 2, 3, 4, 4])
        np.testing.assert_array_equal(reach.bucket([3599, 3600, 8999, 9000], reach.T_EDGES), [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
