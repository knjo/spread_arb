"""Evaluate real market releases without treating expiry or missing days as fills."""
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .event_release_calibration import write


class ReleaseCalibrationTests(unittest.TestCase):
    def test_market_closures_expiry_unknown_next_day_and_repeated_buckets(self):
        days=['20260723','20260724'];start=open_ns(days[0]);nxt=open_ns(days[1])
        common=dict(stream='S2',entry_day=days[0])
        labels=[dict(common,id='normal',close_kind='maker_exit',close_day=days[1],close_ns=nxt+200*SECOND),
                dict(common,id='expiry',close_kind='expiry_basis_zero_accounting',close_day=days[1],close_ns=nxt),
                dict(common,id='forced',close_kind='corporate_action_exit',close_day=days[0],close_ns=start+200*SECOND),
                dict(common,id='open',close_kind=None,close_day=None,close_ns=None)]
        risk=lambda names:json.dumps([dict(id=name,nominal_twd=1000.,today=100.,next_session=50.) for name in names])
        trace=lambda ns,next_day,names:dict(kind='capacity_forecast',ns=ns,next_session=next_day,positions_json=risk(names))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'report';output.mkdir()
            pl.from_dicts(labels).write_parquet(output/'position_returns.parquet')
            for day,rows in zip(days,[
                [trace(start,days[1],[]),trace(start+100*SECOND,days[1],['normal','expiry','forced','open']),
                 trace(start+110*SECOND,days[1],['normal'])],
                [trace(nxt+100*SECOND,'20260727',['normal','open'])]]):
                folder=root/f'Date={day}'/'event';folder.mkdir(parents=True)
                pl.from_dicts(rows).write_parquet(folder/'execution.parquet')
            result=write(root,'event',days,days,output)
            frame=pl.read_parquet(output/'actual_release_forecasts.parquet')
        self.assertEqual(frame.height,6)
        first=frame.filter(pl.col('day')==days[0])
        self.assertAlmostEqual(first['actual_today_twd'].sum(),1000.)
        self.assertAlmostEqual(first['actual_next_twd'].sum(),1000.)
        self.assertAlmostEqual(first['predicted_today_twd'].sum(),400.)
        self.assertEqual(frame.filter(pl.col('day')==days[1])['actual_next_twd'].null_count(),2)
        for row in result:
            if row['carry']:
                self.assertIsNone(row['predicted_next_fraction'])
                self.assertIsNone(row['actual_next_fraction'])


if __name__=='__main__':
    unittest.main()
