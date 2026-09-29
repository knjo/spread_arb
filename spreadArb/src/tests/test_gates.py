import unittest

import numpy as np

from ..backtest.gates import GateSpec, change_points, gap_ticks, leg_series, spot_shape
from ..common.paths import SECOND, open_ns
from .test_slip import series

T0 = open_ns("20260706")


class Gates(unittest.TestCase):
    def test_gap_and_shape(self):
        # 40 TWD band (tick 0.05 = 500): bid 40.00 / 39.90 -> 2 ticks; ask 40.05 / 40.10 -> 1 tick
        b = series([(10.0, [(400_000, 3), (399_000, 5)], [(400_500, 2), (401_000, 8)])])
        self.assertEqual(gap_ticks(b, "bid").tolist(), [2.0])
        self.assertEqual(gap_ticks(b, "ask").tolist(), [1.0])
        b1_a1b1, a1_a1a5 = spot_shape(b)
        self.assertAlmostEqual(b1_a1b1[0], 3 / 5)
        self.assertAlmostEqual(a1_a1a5[0], 2 / 10)

    def test_leg_series_change_points(self):
        # spot: ask side 1 tick apart throughout; A1 share 0.1 then 0.5; B1_A1B1 0.2 then 0.8
        spot = series([(10.0, [(400_000, 2), (399_500, 5)], [(400_500, 8), (401_000, 72)]),
                       (20.0, [(400_000, 40), (399_500, 5)], [(400_500, 10), (401_000, 10)])])
        # futures: bid gap 1 tick, then 3 ticks; ask gap 1 tick throughout
        fut = series([(5.0, [(401_000, 2), (400_500, 1)], [(401_500, 2), (402_000, 1)]),
                      (15.0, [(401_000, 2), (399_500, 1)], [(401_500, 2), (402_000, 1)])])
        legs = leg_series(spot, fut, GateSpec())
        ns, ok = legs["S2_sell"]
        self.assertEqual(ok.tolist(), [True, False])                       # A1_A1A5 0.1 < 0.2, then 0.5
        ns, ok = legs["S1_buy"]                                             # needs futures bid gap 1 AND B1_A1B1 < 0.3
        self.assertEqual([(int((t - T0) / SECOND), bool(o)) for t, o in zip(ns, ok)], [(5, False), (10, True), (15, False)])
        self.assertEqual(legs["S1_sell"][1].tolist(), [True])
        self.assertEqual(legs["S2_buy"][1].tolist(), [True])

    def test_change_points(self):
        ns, ok = change_points(np.array([1, 2, 3, 4]), np.array([True, True, False, False]))
        self.assertEqual((ns.tolist(), ok.tolist()), ([1, 3], [True, False]))


if __name__ == "__main__":
    unittest.main()
