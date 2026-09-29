"""Reconstruct release credit from event-time paired inventory, never future exits."""
import argparse
from collections import Counter
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from ...ev_lookup_cost.forecast_calendar import known_calendar
from ..release_model import ReleaseSnapshot
from .models import same


def read(path):
    return pl.read_parquet(path).to_dicts() if path.exists() else []


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    copies={name:root/'input_snapshot'/f'{i:03d}_{Path(name).name}'
            for i,name in enumerate(manifest['inputs_sha256'])}
    checks=Counter()
    for config in manifest['configurations']:
        if not config.get('release_credit'):
            continue
        amounts,paired,blocked={},{},set()
        sold={}
        for day in manifest['days']:
            folder=root/f'Date={day}'/config['name']
            positions={r['id']:r for r in read(folder/'positions.parquet')}
            path=next(value for key,value in copies.items() if key.endswith(f'Date={day}/release_snapshot.json'))
            model=ReleaseSnapshot(**json.loads(path.read_text()))
            assert model.day==day and all(d<day for d in model.train_days)
            ledger=read(folder/'ledger.parquet')
            index=0
            exit_orders,crossed={},set()

            def consume(pid,ns,kind,optional=False):
                nonlocal index
                if optional and (index==len(ledger) or (ledger[index]['id'],ledger[index]['ns'],ledger[index]['kind'])!=(pid,ns,kind)):
                    return
                row=ledger[index]
                assert (row['id'],row['ns'],row['kind'])==(pid,ns,kind),(day,pid,kind,row)
                delta=row['delta_cents']
                if kind in {'reserve','unreserved_fill'}:
                    assert pid not in amounts
                    amounts[pid]=delta
                elif kind=='hedged':
                    amounts[pid]+=delta
                elif kind=='release':
                    assert amounts.pop(pid)==-delta
                assert sum(amounts.values())==row['committed_cents']
                index+=1

            for t in read(folder/'execution.parquet'):
                pid,ns,kind=t['position_id'],t['ns'],t['kind']
                if kind=='quote':
                    consume(pid,ns,'reserve',optional=True)
                elif kind=='unreserved_fill':
                    consume(pid,ns,'unreserved_fill')
                elif kind=='paired':
                    consume(pid,ns,'hedged')
                    paired[pid]=positions[pid]
                elif kind=='exit_quote':
                    exit_orders[pid]=t['quantity']
                elif kind=='maker_fill' and t['purpose']=='exit':
                    sold[pid]=sold.get(pid,0)+t['quantity']
                    exit_orders[pid]-=t['quantity']
                    if exit_orders[pid]==0:
                        del exit_orders[pid]
                        paired.pop(pid,None)
                elif kind=='cancel' and t['purpose']=='entry':
                    consume(pid,ns,'release',optional=True)
                elif kind=='cancel' and t['purpose']=='exit':
                    exit_orders.pop(pid,None)
                    if sold.get(pid,0)>0 or pid in crossed:
                        paired.pop(pid,None)
                elif kind=='cross_decision':
                    crossed.add(pid)
                    if pid not in exit_orders:
                        paired.pop(pid,None)
                elif kind=='corporate_risk_exit':
                    paired.pop(pid,None)
                elif kind=='taker_fill' and t['purpose']=='exit_spot_remainder':
                    sold[pid]=sold.get(pid,0)+t['quantity']
                elif t.get('pnl_twd') is not None:
                    consume(pid,ns,'release')
                    paired.pop(pid,None)
                elif kind=='contract_continuity_blocked':
                    blocked.add(pid)
                elif kind=='capacity_forecast':
                    assert sum(amounts.values())==t['committed_cents'], 'forecast used a different live ledger'
                    inputs=[dict(id=key,stream=p['stream'],entry_day=p['entry_day'],expiry=p['contract']['expiry'],
                        premium=p['quote_ab']-p['anchor'],nominal_twd=amounts.get(key,0)/100)
                        for key,p in sorted(paired.items()) if key not in blocked]
                    expected=model.capacity(inputs,second=(ns-open_ns(day))//SECOND,calendar=known_calendar(day))
                    assert round(expected['admission_twd']*100)==t['admission_cap_cents']
                    assert 20_000_000*100<=t['admission_cap_cents']<=25_000_000*100
                    for key in ('forecast_today_twd','forecast_next_session_twd'):
                        assert abs(expected[key]-t[key])<1e-6
                    assert expected['next_session']==t['next_session']
                    same(json.loads(t['positions_json']),expected['positions'])
                    checks['forecasts']+=1
                    checks['position_forecasts']+=len(inputs)
            assert index==len(ledger),(day,'ledger event omitted from chronology')
            # Only the actual state reached by today's events is carried.
            expected_paired={key for key,p in positions.items() if p['state']=='paired'}
            assert set(paired)==expected_paired,(day,'paired inventory disappeared')
            checks['portfolio_days']+=1
    result=dict(passed=True,**checks,
        checks='Replayed quote reservations, irrevocable fills, hedges, cancellations and closes in trace order; '
               'every release-credit forecast uses exactly the paired inventory still present at that event. '
               'Predicted release never removes a ledger position.')
    (root/'verification_capacity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
