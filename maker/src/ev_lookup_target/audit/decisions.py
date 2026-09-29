"""Revalue recorded quote decisions and inspect irreversible fill chronology."""
import argparse
from collections import Counter
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from ...ev_lookup_cost.forecast_calendar import known_calendar
from ...ev_lookup_cost.audit.cost_tables import verify_priority
from ..q_model import QSnapshot
from .entry_window import DAILY_HURDLE_BP, sessions as entry_window_sessions


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    copies={name:root/'input_snapshot'/f'{i:03d}_{Path(name).name}'
            for i,name in enumerate(manifest['inputs_sha256'])}
    checks=Counter()
    for day in manifest['days']:
        calendar=known_calendar(day)
        path=next(value for key,value in copies.items() if key.endswith(f'Date={day}/q_snapshot.json'))
        q=QSnapshot(**json.loads(path.read_text()))
        assert q.day==day and all(d<day for d in q.train_days)
        for config in manifest['configurations']:
            if config.get('q_policy') is None:
                continue
            folder=root/f'Date={day}'/config['name']
            path=folder/'decisions.parquet'
            rows=pl.read_parquet(path) if path.exists() else pl.DataFrame()
            for d in rows.iter_rows(named=True):
                assert d['q_snapshot_day']==day
                second=(d['ns']-open_ns(day))//SECOND
                assert 0<=second<15600
                verify_priority(d)
                free=d['committed_cents']+d['reservation_cents']<=d['admission_cap_cents']
                assert bool(d['slot_free'])==free
                deep=bool(d.get('deep_shared'))
                if deep:
                    assert config.get('deep_shared') and d['stream']=='S1' and d['parked']
                    assert d['quote_price']<d['deep_stock_bid'] and d['deep_stock_book_ns']<=d['ns']
                    if d['admit'] and not free:
                        checks['deep_quotes_without_room']+=1
                if config.get('max_ticket_twd') is not None:
                    assert d['reservation_cents']<=config['max_ticket_twd']*100
                assert d['q_prior_sessions']==q.observed_prior_sessions
                if q.observed_prior_sessions<20:
                    reason='basis' if d['quote_ab']<=0 else 'q_warmup'
                    assert d['q_entry_key']=='warmup_not_valued'
                else:
                    estimate=q.estimate(stream=d['stream'],quote_second=second,ab=d['quote_ab'],eff_u=d['eff_u'],
                        expiry=d['expiry'],spread_bp=d['quote_spread_bp'],quote_notional_twd=d['reservation_cents']/100,
                        execution_cost_bp=d['execution_cost_bp'],calendar=calendar)
                    mapping=dict(q_net_bp='net_bp',est_bp='net_bp',q_before_funding_bp='before_funding_bp',
                        q_days='expected_calendar_days',q_funding_bp='funding_bp',q_hurdle_bp='capital_hurdle_bp',
                        p_sd='p_sd',p_overnight='p_overnight',p_expiry='p_expiry',p_other='p_other',
                        p_rollback='p_rollback',d_in='d_in',d_sd='d_sd',d_on='d_on')
                    for key,field in mapping.items():
                        assert abs(d[key]-getattr(estimate,field))<1e-7,(day,config['name'],key)
                    assert d['q_entry_samples']==estimate.entry_samples
                    assert d['q_entry_key']==estimate.entry_key
                    required=estimate.capital_hurdle_bp if config['q_policy']=='target' else 0.
                    if config.get('target_clock')=='entry_window':
                        duration=entry_window_sessions(q,stream=d['stream'],quote_second=second,ab=d['quote_ab'],
                            eff_u=d['eff_u'],expiry=d['expiry'],spread_bp=d['quote_spread_bp'],
                            quote_notional_twd=d['reservation_cents']/100,calendar=calendar)
                        required=duration*DAILY_HURDLE_BP
                        assert abs(d['q_entry_window_sessions']-duration)<1e-7
                        assert abs(d['q_entry_window_hurdle_bp']-required)<1e-7
                    assert abs(d['q_required_bp']-required)<1e-7
                    reason=('basis' if d['quote_ab']<=0 else 'q_warmup' if estimate.entry_samples<50 else
                            'q_value' if estimate.net_bp<required else 'cap' if not free and not deep else 'ok')
                assert d['reason']==reason and bool(d['admit'])==(reason=='ok')
                checks['quote_decisions']+=1
            trace_path=folder/'execution.parquet'
            trace=pl.read_parquet(trace_path).to_dicts() if trace_path.exists() else []
            cancel_requests={}
            quotes={}
            quote_prices={}
            last=0
            for t in trace:
                assert t['ns']>=last, 'execution event order moved backwards'
                last=t['ns']
                pid=t['position_id']
                if t['kind']=='quote':
                    quotes[pid]=t['ns']
                    quote_prices[pid]=t['price']
                elif t['kind'] in {'event_cancel','capacity_cancel'}:
                    cancel_requests.setdefault(pid,t['ns'])
                    if t['kind']=='event_cancel':
                        assert t['effective_ns']==t['ns']+manifest['cancel_ms']*1_000_000
                    elif t.get('deep_approach'):
                        assert config.get('deep_shared') and t['stream']=='S1'
                        if t.get('capacity_stock_bid') is not None:
                            assert quote_prices[pid]>=t['capacity_stock_bid']-1e-8
                        if t.get('capacity_stock_book_ns') is not None:
                            assert t['capacity_stock_book_ns']<=t['ns']
                        checks['deep_top_approach_cancels']+=1
                    checks[t['kind']]+=1
                elif t['kind']=='maker_fill':
                    if t['purpose']=='entry':
                        assert quotes[pid]<t['ns'], 'new queue reused simultaneous or earlier print'
                        if pid in cancel_requests:
                            assert t['ns']<=cancel_requests[pid]+manifest['cancel_ms']*1_000_000
                            checks['cancel_race_fills_retained']+=1
                    checks['maker_fills']+=1
        print(json.dumps(dict(day=day,quote_decisions=checks['quote_decisions'])),flush=True)
    result=dict(passed=True,**checks,
        checks='All Q quote values and admission reasons reconstructed from the frozen opening table; '
               'S2 first priority, observable depth, 50% one-tick cost floor and fresh queue chronology checked.')
    (root/'verification_decisions.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(verify(parser.parse_args().root),indent=2))
