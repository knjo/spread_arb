import unittest

import numpy as np

from ..common.books import INF, BookSeries
from ..common.paths import SECOND, open_ns
from ..slip.build import _book_state, _second_level

T0 = open_ns("20260706")


def series(rows):
    """rows: (sec, [(bid_px, qty)...], [(ask_px, qty)...])"""
    n = len(rows)
    z = np.zeros((n, 5), dtype=np.int64)
    bp, bq, ap, aq = z.copy(), z.copy(), z.copy(), z.copy()
    for i, (_, bids, asks) in enumerate(rows):
        for k, (p, q) in enumerate(bids):
            bp[i, k], bq[i, k] = p, q
        for k, (p, q) in enumerate(asks):
            ap[i, k], aq[i, k] = p, q
    ns = np.array([T0 + int(r[0] * SECOND) for r in rows], dtype=np.int64)
    zero = np.zeros(n, np.int64)
    return BookSeries("F:X", ns, np.arange(n, dtype=np.int64), np.ones(n, bool), bp, bq, ap, aq, zero, zero, zero, zero)


class SlipBookState(unittest.TestCase):
    def test_second_level_and_gap_in_ticks(self):
        # 40 TWD band: tick 0.05 = 500. bids 40.00 / 39.90 (gap 2 ticks); asks 40.05 / 40.10 (gap 1 tick)
        b = series([(10.0, [(400_000, 3), (399_000, 5)], [(400_500, 2), (401_000, 7)]),
                    (20.0, [(399_500, 1)], [(400_500, 4)])])
        self.assertEqual(_second_level(b.bid_px, b.bid_qty, b.top_bid_px, "bid").tolist(), [399_000, 0])
        self.assertEqual(_second_level(b.ask_px, b.ask_qty, b.top_ask_px, "ask").tolist(), [401_000, INF])
        t = np.array([T0 + 15 * SECOND, T0 + 25 * SECOND])
        bid = _book_state(b, t, "bid", -1, "hedge")
        self.assertEqual(bid["hedge_gap12_ticks"][0], 2.0)
        self.assertTrue(np.isnan(bid["hedge_gap12_ticks"][1]))          # no second level displayed
        self.assertEqual(bid["hedge_near_qty"].tolist(), [3.0, 1.0])
        self.assertEqual(bid["hedge_spread_ticks"].tolist(), [1.0, 2.0])
        ask = _book_state(b, t, "ask", 1, "hedge")
        self.assertEqual(ask["hedge_gap12_ticks"][0], 1.0)
        # adverse momentum for a sell hedge (adverse = -1): mid fell 40.025 -> 40.00 over the last 30 s window
        self.assertAlmostEqual(bid["hedge_mom_30s_bp"][1], (400_250 - 400_000) / 400_250 * 1e4, places=6)


if __name__ == "__main__":
    unittest.main()
