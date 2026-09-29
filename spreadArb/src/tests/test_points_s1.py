import unittest

import numpy as np
import polars as pl

from ..common.books import BookSeries, Prints
from ..common.paths import CLOSE_SECOND, SECOND, open_ns
from ..points import s1
from .test_points_s2 import books, prints

DAY = "20260706"
T0 = open_ns(DAY)

# 10-50 TWD band, tick 0.05. spot 40.00 / 40.05 (B1 5,000 shares, A1 20,000); futures 40.30 / 40.45
# quote ladder: inside 40.05? no (== A1) -> B1 40.00, 39.95
# quote_ab at B1 = (40.30 / 40.00 - 1) = 75 bp, eff = 45 with anchor 30
SPOT = [(0, 400_000, 5000, 400_500, 20_000, True)]
FUT = [(0, 403_000, 3, 404_500, 2, True)]


def inputs(spot, fut, spot_prints, anchor_bp=30.0):
    anchor = np.full(CLOSE_SECOND, anchor_bp)
    return s1.ProductInputs(DAY, "2330", "CDFG6", 2000, "20260715", 400_000, 400_000, spot, fut, spot_prints,
                            anchor, np.zeros(CLOSE_SECOND), 16.0, (360_000, 440_000))


class S1PointsTest(unittest.TestCase):
    def test_ladder_and_quote_basis(self):
        rows = s1.product_rows(inputs(books("S:2330", SPOT), books("F:CDFG6", FUT), None))
        first = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["side"] == "buy"]
        self.assertEqual([r["level"] for r in first], [0, 1])                     # no inside level: B1 + tick == A1
        self.assertEqual([r["price"] for r in first], [400_000, 399_500])
        self.assertAlmostEqual(first[0]["quote_ab"], 75.0, places=6)
        self.assertEqual(first[0]["depth_ahead"], 5000)
        self.assertEqual(first[1]["depth_ahead"], 0)
        wide = books("S:2330", [(0, 400_000, 5000, 401_000, 20_000, True)])       # spread 2 ticks: inside level exists
        rows = [r for r in s1.product_rows(inputs(wide, books("F:CDFG6", FUT), None)) if r["side"] == "buy"]
        self.assertEqual(rows[0]["level"], -1)
        self.assertEqual(rows[0]["price"], 400_500)

    def test_fill_needs_queue_ahead_then_two_lots(self):
        # 5,000 ahead at B1: prints at 40.00 of 3,000 + 2,000 clear the queue, then 1,000 (partial) and 1,000 (full)
        pr = prints("S:2330", [(400.0, 400_000, 3000), (500.0, 400_000, 2000), (600.0, 400_000, 1000),
                               (700.0, 399_500, 1000), (800.0, 400_500, 5000)])
        rows = s1.product_rows(inputs(books("S:2330", SPOT), books("F:CDFG6", FUT), pr))
        b1 = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["level"] == 0 and r["side"] == "buy"][0]
        self.assertEqual(b1["t_partial_ns"], T0 + 600 * SECOND)
        self.assertEqual(b1["t_fill_ns"], T0 + 700 * SECOND)
        self.assertEqual(b1["fill_kind"], "through")                              # completed by a print below P
        deeper = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["level"] == 1 and r["side"] == "buy"][0]
        self.assertEqual(deeper["t_partial_ns"], T0 + 700 * SECOND)              # only the 39.95 print reaches it
        self.assertIsNone(deeper["t_fill_ns"])

    def test_hedge_sells_futures_bid_and_records_decay(self):
        pr = prints("S:2330", [(400.0, 400_000, 7000)])
        fut = books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True)])
        row = [r for r in s1.product_rows(inputs(books("S:2330", SPOT), fut, pr)) if r["level"] == 0 and r["side"] == "buy"][0]
        self.assertEqual(row["t_fill_ns"], T0 + 400 * SECOND)
        self.assertEqual(row["hedge_ns"], row["t_fill_ns"] + 50_000_000)
        self.assertEqual(row["hedge_vwap"], 403_000)
        self.assertAlmostEqual(row["actual_ab"], 75.0, places=6)
        self.assertAlmostEqual(row["d_in_realized"], 0.0, places=6)

    def test_deterioration_from_futures_bid_and_gate(self):
        # futures bid drops 40.30 -> 40.10 at 09:15: eff at B1 falls from 45 to (25 bp - 30) = -5
        fut = books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True), (900.0, 401_000, 3, 404_500, 2, True),
                               (1000.0, 401_000, 3, 404_500, 2, False)])
        rows = s1.product_rows(inputs(books("S:2330", SPOT), fut, None))
        b1 = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["level"] == 0 and r["side"] == "buy"][0]
        self.assertEqual(b1["t_below_0"], T0 + 900 * SECOND)
        self.assertEqual(b1["t_below_20"], T0 + 900 * SECOND)
        self.assertIsNone(b1["t_ab0_ns"])
        self.assertEqual(b1["t_gate_ns"], T0 + 1000 * SECOND)
        self.assertEqual(b1["t_down_20"], T0 + 900 * SECOND)
        self.assertIsNone(b1["t_up_20"])

    def test_walker_partial_is_rollback(self):
        pr = prints("S:2330", [(400.0, 400_000, 6000)])          # clears 5,000 ahead + first lot only
        fut = books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True), (500.0, 401_000, 3, 404_500, 2, True)])
        frame = pl.from_dicts(s1.product_rows(inputs(books("S:2330", SPOT), fut, pr)), schema=s1.SCHEMA)
        w = s1.sequential_fills(frame, residual_bp=25.0, floor_bp=20.0)
        self.assertEqual(w.height, 1)
        self.assertTrue(bool(w["rollback"][0]))
        self.assertEqual(w["t_fill_ns"][0], T0 + 400 * SECOND)


    def test_sell_side_exit_quote(self):
        # anchor 100: sell at A1 40.05 vs futures ask 40.45 -> basis 99.9 bp, eff -0.1 (<= 10, admitted)
        # futures ask rises to 40.60 at 09:15 (+37 bp basis) then a spot buyer lifts 40.05 at 09:20
        fut = books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True), (900.0, 403_000, 3, 406_000, 2, True)])
        pr = prints("S:2330", [(1200.0, 400_500, 3000)])
        rows = [r for r in s1.product_rows(inputs(books("S:2330", SPOT), fut, pr, anchor_bp=100.0)) if r["side"] == "sell"]
        a1 = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["level"] == 0][0]
        self.assertEqual(a1["price"], 400_500)
        self.assertAlmostEqual(a1["quote_ab"], (404_500 / 400_500 - 1) * 1e4)
        self.assertEqual(a1["depth_ahead"], 20_000)
        self.assertIsNone(a1["t_fill_ns"])                                        # 3,000 shares < 20,000 ahead + 2,000
        self.assertEqual(a1["t_above_0"], T0 + 900 * SECOND)
        self.assertEqual(a1["t_above_30"], T0 + 900 * SECOND)
        self.assertIsNone(a1["t_above_0"] if a1["t_above_0"] is None else None)
        self.assertEqual(a1["t_up_20"], T0 + 900 * SECOND)
        # no queue ahead: inside quote at 40.10 with a 2-tick spread is filled by the 40.10 print and hedged at the ask
        wide = books("S:2330", [(0, 400_000, 5000, 401_000, 20_000, True)])
        pr2 = prints("S:2330", [(600.0, 400_500, 2000)])
        inside = [r for r in s1.product_rows(inputs(wide, fut, pr2, anchor_bp=100.0)) if r["side"] == "sell" and r["level"] == -1][0]
        self.assertEqual(inside["price"], 400_500)
        self.assertEqual(inside["t_fill_ns"], T0 + 600 * SECOND)
        self.assertEqual(inside["hedge_vwap"], 404_500)                           # buys futures at the ask
        self.assertAlmostEqual(inside["actual_ab"], (404_500 / 400_500 - 1) * 1e4)

    def test_makerfill_alignment(self):
        # the second book quadruples the displayed B1 size, which opens a new segment carrying that tick's makerFill
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True), (500.0, 400_000, 20_000, 400_500, 20_000, True)])
        mf = np.array([[7.5, 30.0, 11.0, 60.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        inp = inputs(spot, books("F:CDFG6", FUT), None)
        inp.makerfill = mf
        rows = s1.product_rows(inp)
        early = [r for r in rows if r["quote_ns"] == T0 + 300 * SECOND and r["side"] == "buy" and r["level"] == 0][0]
        self.assertAlmostEqual(early["mf_bid1_s"], 7.5)
        self.assertEqual(early["mf_fill_ns"], T0 + 300 * SECOND + int(7.5 * SECOND))
        later = [r for r in rows if r["quote_ns"] >= T0 + 500 * SECOND and r["side"] == "buy" and r["level"] == 0]
        self.assertTrue(later and all(abs(r["mf_bid1_s"] - 1.0) < 1e-6 for r in later))


if __name__ == "__main__":
    unittest.main()
