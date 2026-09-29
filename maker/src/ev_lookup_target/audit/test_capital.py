import unittest
from pathlib import Path
import tempfile

import polars as pl

from ...ev_lookup_cost.causal_lookup import open_ns
from .capital import DAY_NS, StockCash, intervals, analyze


class CapitalTests(unittest.TestCase):
    def test_partial_buy_and_sell_charge_only_outstanding_stock_cost(self):
        cash=StockCash()
        cash.buy('p',0,100,1000.)
        cash.buy('p',DAY_NS,100,1200.)
        cash.sell('p',2*DAY_NS,100)
        cash.sell('p',3*DAY_NS,100)
        self.assertAlmostEqual(cash.finish(4*DAY_NS)['p'],4300.)
        self.assertEqual(cash.quantity['p'],0)

    def test_stock_sold_before_future_hedge_stops_cash_funding(self):
        cash=StockCash()
        cash.buy('rollback',0,50,500.)
        cash.sell('rollback',DAY_NS,50)
        self.assertAlmostEqual(cash.finish(10*DAY_NS)['rollback'],500.)

    def test_zero_duration_race_still_counts_in_peak(self):
        result=intervals([(1,5_000_000.),(1,-5_000_000.)],0,2,20_000_000.)
        self.assertAlmostEqual(result['peak_twd'],25_000_000.)
        self.assertAlmostEqual(result['over20_seconds'],0.)

    def test_empty_warmup_session_has_zero_funding_without_a_positions_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'Date=20260504/event').mkdir(parents=True)
            rows,areas=analyze(root,'event',['20260504'])
        self.assertFalse(areas)
        self.assertAlmostEqual(rows[0]['funding_2pct_twd'],0.)
        self.assertAlmostEqual(rows[0]['committed_peak_twd'],0.)

    def test_missing_positions_file_cannot_hide_real_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'Date=20260504/event';folder.mkdir(parents=True)
            pl.from_dicts([dict(ns=open_ns('20260504'),delta_cents=100_000_000)])\
                .write_parquet(folder/'ledger.parquet')
            with self.assertRaises(AssertionError):
                analyze(root,'event',['20260504'])


if __name__=='__main__':
    unittest.main()
