"""Cash flow counts actual stock execution; expiry accounting stays separate."""
from pathlib import Path
import tempfile
import unittest

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .event_turnover import daily, summarize


class TurnoverTests(unittest.TestCase):
    def test_actual_market_flow_and_completed_pairs_are_distinct(self):
        days=['20260723','20260724','20260727']
        start=open_ns(days[0])
        positions=pl.from_dicts([
            dict(entry_day=days[0],close_day=days[1],stream='S2',close_kind='maker_exit',
                 spot_buy_cash=200_000*10_000,entry_fill_ns=start,close_ns=start+86400*SECOND),
            dict(entry_day=days[0],close_day=days[2],stream='S1',close_kind='expiry_basis_zero_accounting',
                 spot_buy_cash=100_000*10_000,entry_fill_ns=start,close_ns=start+4*86400*SECOND),
            dict(entry_day=days[0],close_day=None,stream='S1',close_kind=None,
                 spot_buy_cash=50_000*10_000,entry_fill_ns=start,close_ns=None),
        ])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            traces=[[
                dict(kind='quote',stream='S1',purpose='entry',quantity=9000,price=100.),
                dict(kind='cancel',stream='S1',purpose='entry',quantity=9000,price=100.),
                dict(kind='maker_fill',stream='S2',purpose='entry',quantity=1,price=100.5),
                dict(kind='taker_fill',stream='S2',purpose='entry_spot',quantity=2000,price=100.),
                dict(kind='maker_fill',stream='S1',purpose='entry',quantity=1500,price=100.),
                dict(kind='taker_fill',stream='S1',purpose='entry_future',quantity=1,price=100.5),
            ],[
                dict(kind='maker_fill',stream='S2',purpose='exit',quantity=2000,price=101.),
                dict(kind='taker_fill',stream='S2',purpose='exit_future',quantity=1,price=101.),
            ],[dict(kind='expiry_basis_zero_accounting',stream='S1')]]
            for day,trace in zip(days,traces):
                path=root/f'Date={day}'/'event';path.mkdir(parents=True)
                pl.from_dicts(trace).write_parquet(path/'execution.parquet')
            frame=daily(root,'event',days,positions)
        self.assertAlmostEqual(frame['market_stock_buy_cash_twd'].sum(),350_000.)
        self.assertAlmostEqual(frame['market_stock_sell_cash_twd'].sum(),202_000.)
        self.assertAlmostEqual(frame['completed_original_stock_cash_twd'].sum(),300_000.)
        self.assertAlmostEqual(frame['normal_closed_original_stock_cash_twd'].sum(),200_000.)
        self.assertAlmostEqual(frame['expiry_accounting_released_original_stock_cash_twd'].sum(),100_000.)
        summary=summarize(frame,positions,days)
        self.assertEqual(summary['observed_sessions'],3)
        self.assertAlmostEqual(summary['stock_buy_turns_per_20M_per_day'],350_000/3/20_000_000)
        self.assertAlmostEqual(summary['resolved_mean_calendar_days'],2.5)
        self.assertAlmostEqual(summary['resolved_cash_weighted_calendar_days'],2.)
        post=summarize(frame,positions,days,days[1])
        self.assertEqual(post['observed_sessions'],2)
        self.assertAlmostEqual(post['entries_per_day'],0.)
        self.assertAlmostEqual(post['normal_close_cash_turns_per_20M_per_day'],.005)


if __name__=='__main__':
    unittest.main()
