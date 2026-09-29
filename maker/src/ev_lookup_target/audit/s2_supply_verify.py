"""Check S2 supply decisions, irreversible fills, delayed hedge cash and raw prints."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.market import previous_tick
from ...ev_lookup_cost.verify_run import verify_raw_fills


def records(path):
    return pl.read_parquet(path).to_dicts() if path.exists() else []


def verify(root,raw=False):
    protocol=json.loads((root/'protocol.json').read_text())
    hashes=json.loads((root.parent/'sources.json').read_text())
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    checks=Counter()
    for day in protocol['days']:
        for config in protocol['configs']:
            folder=root/f'Date={day}'/config['name']
            decisions={d['intent_id']:d for d in records(folder/'decisions.parquet')}
            for d in decisions.values():
                price,bid,ask=(round(d[k]*10_000) for k in ('quote_price','future_bid','future_ask'))
                assert bid<price<ask and price==previous_tick(ask)
                assert d['future_book_ns']<=d['ns'] and d['liquidity_book_ns']<=d['ns']
                assert d['liquidity_a1_multiple']+1e-10>=config['depth_multiple']
                assert d['eff_u']+1e-8>=config['residual_bp'] and d['quote_ab']>0
                assert d['admit'] and d['reason']=='ok'
                checks['decisions']+=1
            trace=records(folder/'execution.parquet')
            quotes={};cancels={};fills={};cash=defaultdict(float);qty=Counter();last=-1
            last_fill={};pending={};live={}
            for t in trace:
                assert t['ns']>=last
                last=t['ns'];pid=t['position_id'];kind=t['kind']
                if kind=='quote':
                    assert pid in decisions and t['ns']==decisions[pid]['ns']
                    assert t['vc'] not in pending and t['vc'] not in live
                    assert t['ns']>=last_fill.get(t['vc'],0)+config['cooldown_seconds']*1_000_000_000
                    live[t['vc']]=pid
                    quotes[pid]=t
                elif kind=='event_cancel':
                    assert t['effective_ns']==t['ns']+50_000_000
                    cancels[pid]=t['effective_ns']
                elif kind=='maker_fill':
                    assert pid not in fills and quotes[pid]['ns']<t['ns']
                    assert t['quantity']==1 and t['purpose']=='entry'
                    if pid in cancels:
                        assert t['ns']<=cancels[pid]
                        checks['retained_races']+=1
                    fills[pid]=t
                    assert live.pop(t['vc'])==pid
                    last_fill[t['vc']]=t['ns'];pending[t['vc']]=pid
                    checks['maker_fills']+=1
                elif kind=='taker_fill':
                    assert t['purpose']=='entry_spot'
                    assert t['book_ns']<=t['ns'] and t['ns']>=fills[pid]['ns']+50_000_000
                    cash[pid]+=t['price']*t['quantity']
                    qty[pid]+=t['quantity']
                    checks['hedge_reports']+=1
                elif kind=='paired':
                    assert pending.pop(t['vc'])==pid and qty[pid]==2000
                elif kind=='cancel':
                    assert live.pop(t['vc'])==pid
            rows=records(folder/'fills.parquet')
            assert {r['id'] for r in rows}==set(fills)
            for r in rows:
                pid=r['id'];f=fills[pid]
                assert r['fill_ns']==f['ns'] and r['stock_shares']==qty[pid]
                assert abs(r['stock_cash_twd']-cash[pid])<1e-6
                if r['hedged_ns'] is not None:
                    assert qty[pid]==2000
                    actual=(f['price']*2000/cash[pid]-1)*10_000
                    assert abs(actual-r['actual_ab'])<1e-7
                    checks['negative_basis']+=actual<0
            ledger=records(folder/'ledger.parquet');total=0;last=-1
            for row in ledger:
                assert row['ns']>=last
                last=row['ns'];total+=row['delta_cents']
                assert total==row['committed_cents'] and total>=0
                if row['kind']=='release':
                    assert row['id'] not in fills,'filled inventory was released without an exit'
            if raw and fills:
                checks['raw_prints']+=verify_raw_fills(day,pl.from_dicts(trace,infer_schema_length=None))
            checks['policy_days']+=1
        print(json.dumps(dict(day=day,**checks)),flush=True)
    result=dict(passed=True,**checks,scope='Entry-only supply chronology and actual stock cash, not portfolio exit/return validation.')
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--raw',action='store_true')
    args=p.parse_args();print(json.dumps(verify(args.root,args.raw),indent=2))
