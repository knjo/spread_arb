import unittest

import numpy as np
import polars as pl

from ..common.grid import ProductDay
from ..common.paths import CLOSE_SECOND
from ..ev import qlevel


def synthetic_product(day, vc, rng, mean=0.0, sd=15.0):
    resid = rng.normal(mean, sd, CLOSE_SECOND)
    anchor = np.full(CLOSE_SECOND, 30.0)
    return ProductDay(day, vc, "XXFG6", "20260715", anchor + resid, anchor + resid + 10.0, anchor,
                      np.ones(CLOSE_SECOND, dtype=bool))


class QLevelTest(unittest.TestCase):
    def test_histogram_quantiles_match_numpy(self):
        rng = np.random.default_rng(1)
        p = synthetic_product("20260701", "2330", rng)
        h = qlevel.daily_hist({"2330": p})
        qs = qlevel.quantiles_from_hist(h["bin"].to_numpy(), h["count"].to_numpy())
        truth = np.quantile(p.mid - p.anchor, qlevel.QUANTILES)
        np.testing.assert_allclose(qs, truth, atol=1.0)

    def test_table_is_as_of_and_pooled(self):
        rng = np.random.default_rng(2)
        hists = pl.concat([qlevel.daily_hist({"2330": synthetic_product(d, "2330", rng, sd=s)})
                           for d, s in (("20260701", 10.0), ("20260702", 10.0), ("20260703", 100.0))])
        t = qlevel.table_from_hists(hists, "20260703", window=20)
        self.assertEqual(t["n_days"][0], 2)
        self.assertLess(t["scale"][0], 25.0)          # the wide 07/03 day is not visible on 07/03
        self.assertEqual(t["as_of"][0], "20260703")
        self.assertEqual(qlevel.table_from_hists(hists, "20260701").height, 0)

    def test_ineligible_seconds_are_excluded(self):
        rng = np.random.default_rng(3)
        p = synthetic_product("20260701", "2330", rng)
        p.eligible[:] = False
        self.assertEqual(qlevel.daily_hist({"2330": p}).height, 0)


if __name__ == "__main__":
    unittest.main()
