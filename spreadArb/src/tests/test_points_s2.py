import unittest

import numpy as np

from ..common.books import BookSeries, Prints
from ..common.paths import CLOSE_SECOND, SECOND, open_ns
from ..points import s2

DAY = "20260706"
T0 = open_ns(DAY)


def books(instrument, rows):
    """rows: (seconds_from_open (float), bid_px, bid_qty, ask_px, ask_qty, formal)."""
    n = len(rows)
    ns = np.array([T0 + int(r[0] * SECOND) for r in rows], dtype=np.int64)
    z = np.zeros((n, 5), dtype=np.int64)
    bid_px, bid_qty, ask_px, ask_qty = z.copy(), z.copy(), z.copy(), z.copy()
    for i, r in enumerate(rows):
        bid_px[i, 0], bid_qty[i, 0], ask_px[i, 0], ask_qty[i, 0] = r[1], r[2], r[3], r[4]
        if len(r) > 6:   # optional second ask level
            ask_px[i, 1], ask_qty[i, 1] = r[6]
    return BookSeries(instrument, ns, np.arange(n, dtype=np.int64), np.array([r[5] for r in rows]),
                      bid_px, bid_qty, ask_px, ask_qty, np.zeros(n, np.int64), np.zeros(n, np.int64),
                      np.zeros(n, np.int64), np.zeros(n, np.int64))


def prints(instrument, rows):
    ns = np.array([T0 + int(r[0] * SECOND) for r in rows], dtype=np.int64)
    return Prints(instrument, ns, np.arange(len(rows), dtype=np.int64),
                  np.array([r[1] for r in rows], dtype=np.int64), np.array([r[2] for r in rows], dtype=np.int64))


def inputs(spot, fut, fut_prints, anchor_bp=30.0):
    anchor = np.full(CLOSE_SECOND + 1, anchor_bp)
    return s2.ProductInputs(DAY, "2330", "CDFG6", 2000, "20260715", 400_000, 400_000, spot, fut, fut_prints,
                            anchor, np.zeros(CLOSE_SECOND + 1), 16.0, (360_000, 440_000))


# 10-50 TWD band, tick 0.05. spot 40.00 / 40.05 with 20k shares at ask; futures 40.15 / 40.30 (bid/ask)
# -> quote 40.25 (one tick inside), ab = (40.25/40.05 - 1) = 49.9 bp, eff = 19.9 with anchor 30
SPOT = [(0, 400_000, 5000, 400_500, 20_000, True)]
FUT = [(0, 401_500, 3, 403_000, 2, True)]


class S2PointsTest(unittest.TestCase):
    def test_quote_is_one_tick_inside_and_needs_two_tick_book(self):
        rows = s2.product_rows(inputs(books("S:2330", SPOT), books("F:CDFG6", FUT), None))
        self.assertTrue(rows)
        self.assertEqual(rows[0]["price"], 402_500)
        self.assertAlmostEqual(rows[0]["quote_ab"], (402_500 / 400_500 - 1) * 1e4)
        self.assertAlmostEqual(rows[0]["eff_u"], rows[0]["quote_ab"] - 30.0)
        one_tick = books("F:CDFG6", [(0, 402_500, 3, 403_000, 2, True)])
        self.assertEqual(s2.product_rows(inputs(books("S:2330", SPOT), one_tick, None)), [])

    def test_fill_is_first_print_at_or_above_price_strictly_after_quote(self):
        pr = prints("F:CDFG6", [(300.0, 403_000, 1), (400.0, 402_000, 5), (500.0, 402_500, 1), (600.0, 403_000, 1)])
        rows = s2.product_rows(inputs(books("S:2330", SPOT), books("F:CDFG6", FUT), pr))
        first = rows[0]
        self.assertEqual(first["quote_ns"], T0 + 300 * SECOND)       # the 09:05 print itself cannot fill this quote
        self.assertEqual(first["t_fill_ns"], T0 + 500 * SECOND)      # 40.20 print is below the quote, skipped
        self.assertEqual(first["fill_kind"], "at_price")
        later = [r for r in rows if r["quote_ns"] == T0 + 500 * SECOND][0]   # re-emitted at the print
        self.assertEqual(later["t_fill_ns"], T0 + 600 * SECOND)
        self.assertEqual(later["fill_kind"], "through")

    def test_hedge_sweeps_levels_and_records_decay(self):
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 1000, True, (401_000, 5000))])
        pr = prints("F:CDFG6", [(400.0, 402_500, 1)])
        row = s2.product_rows(inputs(spot, books("F:CDFG6", FUT), pr))[0]
        self.assertEqual(row["hedge_ns"], row["t_fill_ns"] + 50_000_000)
        vwap = (400_500 * 1000 + 401_000 * 1000) / 2000
        self.assertEqual(row["hedge_vwap"], int(round(vwap)))
        self.assertEqual(row["hedge_levels_swept"], 2)
        self.assertAlmostEqual(row["actual_ab"], (402_500 / vwap - 1) * 1e4)
        self.assertAlmostEqual(row["d_in_realized"], row["quote_ab"] - row["actual_ab"])
        self.assertFalse(row["hedge_timeout"])

    def test_hedge_waits_for_depth_and_flags_timeout(self):
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 1000, True),      # only 1,000 shares
                               (410.0, 400_000, 5000, 400_500, 3000, True)])  # depth arrives 10 s later
        pr = prints("F:CDFG6", [(400.0, 402_500, 1)])
        row = s2.product_rows(inputs(spot, books("F:CDFG6", FUT), pr))[0]
        self.assertEqual(row["hedge_ns"], T0 + 410 * SECOND)
        self.assertTrue(row["hedge_timeout"])
        self.assertAlmostEqual(row["hedge_wait_ms"], 10_000.0)

    def test_deterioration_floors_and_cancel_race(self):
        # spot ask rises at 09:10 to 40.17: locked basis 40.25/40.17 -> 19.9 bp, eff -10 -> below every floor
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True),
                               (600.0, 400_000, 5000, 401_700, 20_000, True)])
        pr = prints("F:CDFG6", [(600.03, 402_500, 1)])            # fills 30 ms after the deterioration
        rows = s2.product_rows(inputs(spot, books("F:CDFG6", FUT), pr, anchor_bp=25.0))   # eff 24.9 at quote
        first = rows[0]
        self.assertEqual(first["t_below_0"], T0 + 600 * SECOND)
        self.assertEqual(first["t_below_20"], T0 + 600 * SECOND)
        self.assertEqual(first["t_below_25"], first["quote_ns"])      # eff 24.9 < 25 from the start
        self.assertEqual(first["t_below_30"], first["quote_ns"])
        self.assertTrue(first["cancel_race_20"])
        self.assertIsNone(first["t_gate_ns"])
        late = prints("F:CDFG6", [(600.06, 402_500, 1)])            # 60 ms: cancel would have landed first
        row = s2.product_rows(inputs(spot, books("F:CDFG6", FUT), late, anchor_bp=25.0))[0]
        self.assertFalse(row["cancel_race_20"])

    def test_gate_closes_on_trial_match(self):
        fut = books("F:CDFG6", [(0, 401_500, 3, 403_000, 2, True), (900.0, 401_500, 3, 403_000, 2, False)])
        row = s2.product_rows(inputs(books("S:2330", SPOT), fut, None))[0]
        self.assertEqual(row["t_gate_ns"], T0 + 900 * SECOND)
        self.assertIsNone(row["t_fill_ns"])

    def test_move_columns_and_repeg(self):
        # futures A1 steps up 40.30 -> 40.60 at 09:15 (quotable 40.25 -> 40.55, +74 bp), back down at 09:20
        fut = books("F:CDFG6", [(0, 401_500, 3, 403_000, 2, True), (900.0, 404_500, 3, 406_000, 2, True),
                               (1200.0, 401_500, 3, 403_000, 2, True)])
        pr = prints("F:CDFG6", [(1500.0, 402_500, 1)])
        rows = s2.product_rows(inputs(books("S:2330", SPOT), fut, pr))
        first = rows[0]
        self.assertEqual(first["t_up_20"], T0 + 900 * SECOND)
        self.assertIsNone(first["t_down_20"])
        at_900 = [r for r in rows if r["quote_ns"] == T0 + 900 * SECOND][0]   # quote 40.55
        self.assertEqual(at_900["price"], 405_500)
        self.assertEqual(at_900["t_down_20"], T0 + 1200 * SECOND)
        import polars as pl
        frame = pl.from_dicts(rows, schema=s2.SCHEMA)
        plain = s2.sequential_fills(frame, residual_bp=15.0, floor_bp=10.0)
        self.assertEqual(plain["t_fill_ns"].to_list(), [T0 + 1500 * SECOND])         # sits at 40.25 until the print
        repeg = s2.sequential_fills(frame, residual_bp=15.0, floor_bp=10.0, repeg_bp=20, patience_ns=60 * SECOND)
        # cancelled at 09:15 for a better pair; the 40.55 quote never fills and no later quote fills either
        self.assertEqual(repeg.height, 0)

    def test_buy_side_exit_quote(self):
        # futures 40.15 / 40.30 (3 ticks wide): buy quote at B1+1 = 40.20; spot B1 40.00 -> basis 50 bp;
        # anchor 60 -> eff -10 (admitted); a futures seller hits 40.20 at 09:10; spot bid drops to 39.90 at 09:20
        fut = books("F:CDFG6", [(0, 401_500, 3, 403_000, 2, True)])
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True), (1200.0, 399_000, 5000, 400_500, 20_000, True)])
        pr = prints("F:CDFG6", [(600.0, 402_000, 1)])
        rows = [r for r in s2.product_rows(inputs(spot, fut, pr, anchor_bp=60.0)) if r["side"] == "buy"]
        self.assertTrue(rows)
        first = rows[0]
        self.assertEqual(first["price"], 402_000)
        self.assertAlmostEqual(first["quote_ab"], (402_000 / 400_000 - 1) * 1e4)
        self.assertEqual(first["t_fill_ns"], T0 + 600 * SECOND)
        self.assertEqual(first["hedge_ns"], T0 + 600 * SECOND + 50_000_000)
        self.assertEqual(first["hedge_vwap"], 400_000)                            # sells spot into the bid
        self.assertAlmostEqual(first["actual_ab"], (402_000 / 400_000 - 1) * 1e4)
        self.assertEqual(first["t_above_20"], T0 + 1200 * SECOND)                # 40.20/39.90 = 75 bp, +25
        self.assertIsNone(first["t_above_30"])
        self.assertIsNone(first["t_below_20"])
        # a one-tick futures book has no inside bid
        one_tick = books("F:CDFG6", [(0, 402_500, 3, 403_000, 2, True)])
        self.assertEqual([r for r in s2.product_rows(inputs(spot, one_tick, pr, anchor_bp=60.0)) if r["side"] == "buy"], [])

    def test_walker_takes_one_quote_per_product_until_hedge(self):
        spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True)])
        pr = prints("F:CDFG6", [(400.0, 402_500, 1), (400.02, 402_500, 1), (700.0, 402_500, 1)])
        rows = s2.product_rows(inputs(spot, books("F:CDFG6", FUT), pr))
        import polars as pl
        w = s2.sequential_fills(pl.from_dicts(rows, schema=s2.SCHEMA), residual_bp=15.0, floor_bp=10.0)
        # first quote fills at 400.00, hedge at 400.05; the print at 400.02 is inside the hedge wait, so the
        # next quote can only be taken after 400.05 and fills at 700
        self.assertEqual(w["t_fill_ns"].to_list(), [T0 + 400 * SECOND, T0 + 700 * SECOND])


if __name__ == "__main__":
    unittest.main()
