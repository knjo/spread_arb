"""Rebuild each stock hedge from raw books and independently spend displayed depth."""
import argparse
from collections import Counter
import json
from pathlib import Path

import polars as pl

from ...common.paths import market_data_path, resolve_input_file, spot_tick_path
from ...ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ...ev_lookup_cost.market import BookSeries, _raw


def verify(root):
    protocol=json.loads((root/'protocol.json').read_text())
    count=0
    for day in protocol['days']:
        traces={}
        for config in protocol['configs']:
            path=root/f'Date={day}'/config['name']/'execution.parquet'
            if path.exists():
                traces[config['name']]=(pl.scan_parquet(path).filter(pl.col('kind')=='taker_fill')
                    .select('ns','vc','book_ns','book_sequence','quantity','price','purpose').collect())
        codes=sorted({vc for t in traces.values() for vc in t['vc'].to_list()})
        start=open_ns(day)
        raw=_raw(resolve_input_file(spot_tick_path(day),role='spot_raw'),codes,future=False,
                 start=start,end=start+CLOSE_SECOND*SECOND)
        series={g.item(0,'QuoteCode'):BookSeries('S:'+g.item(0,'QuoteCode'),g)
                for g in raw.partition_by('QuoteCode',maintain_order=True)}
        limits={r['quote_code']:(round(r['limit_down_price']*10000),round(r['limit_up_price']*10000))
                for r in pl.read_parquet(market_data_path(day),columns=['quote_code','limit_down_price','limit_up_price'])
                    .filter(pl.col('quote_code').is_in(codes)).iter_rows(named=True)}
        for policy,trace in traces.items():
            used=Counter()
            for t in trace.iter_rows(named=True):
                assert t['purpose']=='entry_spot'
                book=series[t['vc']].at(t['ns'])
                assert book is not None and book.valid() and book.ns<=t['ns']
                assert book.ns==t['book_ns'] and book.sequence==t['book_sequence']
                lower,upper=limits[t['vc']]
                left=2000;cash=0;quantity=0
                # Entry-only paths start every hedge at a fresh two-lot stock
                # requirement. A partial report would fail the current study's
                # all-at50ms result and require explicit residual accounting.
                for price,shown in book.asks:
                    if not lower<=price<=upper:
                        continue
                    key=(book.instrument,book.ns,book.sequence,price)
                    take=min(left,max(0,shown-used[key]))
                    used[key]+=take;quantity+=take;cash+=price*take;left-=take
                    if left==0:
                        break
                assert quantity==t['quantity']==2000,(day,policy,t)
                assert abs(cash/10000-t['quantity']*t['price'])<1e-6,(day,policy,t)
                count+=1
        print(json.dumps(dict(day=day,raw_stock_hedges=count)),flush=True)
    result=dict(passed=True,raw_stock_hedges=count,
        checks='Actual receive-time raw stock book/sequence and visible ask depth independently consumed '
               'once per policy/book/price. Every reported hedge quantity and cash matches.')
    (root/'verification_hedges.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
