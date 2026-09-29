"""Ranking must use the decision at the actual quote time, including retries."""
import unittest

import polars as pl

from .report_cost import forecast_rows


class ReportingTests(unittest.TestCase):
    def test_reused_intent_cannot_borrow_an_earlier_or_later_decision(self):
        positions = pl.from_dicts([
            dict(id="same",quote_ns=20,stream="S2",actual_ab=40.,contract={"expiry":"20260819"},
                 entry_day="20260724",close_day="20260727",close_kind="maker_exit",pnl_bp=12.,pnl_twd=240.),
            dict(id="unmatched",quote_ns=30,stream="S2",actual_ab=30.,contract={"expiry":"20260819"},
                 entry_day="20260724",close_day="20260727",close_kind="maker_exit",pnl_bp=-8.,pnl_twd=-160.)])
        quotes = pl.from_dicts([
            dict(intent_id="same",ns=10,stream="S2",est_bp=-5.,reason="ev"),
            dict(intent_id="same",ns=20,stream="S2",est_bp=15.,reason="ok"),
            dict(intent_id="unmatched",ns=40,stream="S2",est_bp=25.,reason="ok")])
        result = forecast_rows(positions,quotes,"20260902").sort("id")
        matched = result.filter(pl.col("id")=="same")
        self.assertAlmostEqual(matched["est_bp"].item(),15.)
        self.assertEqual(matched["reason"].item(),"ok")
        self.assertIsNone(result.filter(pl.col("id")=="unmatched")["est_bp"].item())


if __name__ == "__main__":
    unittest.main()
