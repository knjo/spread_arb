"""Verify event S2 pending-hedge exclusion and immediate post-hedge rechecks."""
import argparse
from collections import Counter
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def rows(path):
    return pl.read_parquet(path).to_dicts() if path.exists() else []


def verify(root,shadow_reference=None):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    checks=Counter()
    for config in manifest['configurations']:
        if not config.get('event_s2'):
            continue
        pending={};paired={};rechecks=set()
        for day in manifest['days']:
            folder=root/f'Date={day}'/config['name']
            positions={p['id']:p for p in rows(folder/'positions.parquet')}
            for t in rows(folder/'execution.parquet'):
                if t['stream']!='S2':
                    continue
                pid,vc,ns,kind=t['position_id'],t['vc'],t['ns'],t['kind']
                if kind=='quote':
                    assert vc not in pending,(day,config['name'],vc,'quoted before stock hedge completed')
                    checks['s2_quotes']+=1
                elif kind=='maker_fill' and t['purpose']=='entry':
                    assert vc not in pending
                    pending[vc]=pid
                    checks['s2_entry_fills']+=1
                elif kind=='paired':
                    assert pending.pop(vc)==pid
                    paired[pid]=ns
                    checks['s2_pairs']+=1
                elif kind=='s2_hedge_complete_recheck':
                    assert paired[pid]==ns and pid not in rechecks
                    rechecks.add(pid)
                    checks['same_timestamp_hedge_rechecks']+=1
                elif t.get('pnl_twd') is not None and pending.get(vc)==pid:
                    del pending[vc]
            expected={p['contract']['vc']:p['id'] for p in positions.values()
                if p['stream']=='S2' and p['state'] not in {'closed','cancelled'}
                and p['entry_fill_ns'] is not None
                and p['spot_buy_qty']<p['future_sell_qty']*p['contract']['shares']}
            assert pending==expected,(day,config['name'],'pending stock hedge state mismatch')
            checks['portfolio_days']+=1
        assert set(paired)==rechecks,'some successful stock hedges did not trigger immediate re-evaluation'
    if shadow_reference is not None:
        reference=json.loads((shadow_reference/'manifest.json').read_text())
        assert reference['days']==manifest['days'] and reference['status']=='completed'
        for day in manifest['days']:
            for kind in ('decisions','execution','ledger','positions','marks'):
                p=Path(f'Date={day}/shadow/{kind}.parquet')
                assert (root/p).exists()==(shadow_reference/p).exists()
                if (root/p).exists():
                    expected=pl.read_parquet(shadow_reference/p)
                    actual=pl.read_parquet(root/p).select(expected.columns)
                    assert_frame_equal(actual,expected,check_exact=False,rel_tol=1e-12,abs_tol=1e-8)
                    checks['shadow_files']+=1
                    checks['shadow_rows']+=actual.height
    result=dict(passed=True,**checks,checks='No new S2 quote while its previous futures fill lacks the stock hedge; '
        'every completed S2 stock hedge triggers a recheck at the same receive timestamp; pending state survives days. '
        'Optional original shadow equivalence covers every saved original field.')
    (root/'verification_event_s2.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--shadow-reference',type=Path)
    a=p.parse_args();print(json.dumps(verify(a.root,a.shadow_reference),indent=2))
