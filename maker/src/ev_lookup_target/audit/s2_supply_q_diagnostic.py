"""Revalue every supply quote before joining any resulting fill or hedge label."""
import argparse
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import open_ns, SECOND
from ...ev_lookup_cost.forecast_calendar import known_calendar
from ..q_model import QSnapshot


def run(root,models):
    protocol=json.loads((root/'protocol.json').read_text())
    output=root/'q_diagnostic';output.mkdir(exist_ok=False)
    columns=['intent_id','ns','eff_u','quote_ab','expiry','quote_spread_bp','reservation_cents','execution_cost_bp']
    predictions=[]
    for day in protocol['days']:
        q=QSnapshot(**json.loads((models/f'Date={day}/q_snapshot.json').read_text()))
        assert q.day==day and all(d<day for d in q.train_days)
        calendar=known_calendar(day)
        rows=[]
        for config in protocol['configs']:
            path=root/f'Date={day}'/config['name']/'decisions.parquet'
            if not path.exists():
                continue
            for d in pl.read_parquet(path,columns=columns).iter_rows(named=True):
                estimate=q.estimate(stream='S2',quote_second=(d['ns']-open_ns(day))//SECOND,
                    ab=d['quote_ab'],eff_u=d['eff_u'],expiry=d['expiry'],spread_bp=d['quote_spread_bp'],
                    quote_notional_twd=d['reservation_cents']/100,execution_cost_bp=d['execution_cost_bp'],
                    calendar=calendar) if q.observed_prior_sessions>=20 else None
                ready=estimate is not None and estimate.entry_samples>=50
                rows.append(dict(day=day,policy=config['name'],id=d['intent_id'],quote_ns=d['ns'],
                    q_ready=ready,q_net_bp=estimate.net_bp if estimate else None,
                    q_pass=bool(ready and estimate.net_bp>=0),
                    expected_days=estimate.expected_calendar_days if estimate else None,
                    below_original25bp_support=d['eff_u']<25))
        frame=pl.from_dicts(rows,infer_schema_length=None)
        frame.write_parquet(output/f'quotes_{day}.parquet');predictions.append(frame)
        print(json.dumps(dict(day=day,predicted_quotes=frame.height)),flush=True)
    # All forecasts above have been written before any outcome file is opened.
    pred=pl.concat(predictions,how='diagonal_relaxed')
    observed=pl.concat([pl.read_parquet(p) for day in protocol['days']
        for c in protocol['configs']
        if (p:=root/f'Date={day}'/c['name']/'fills.parquet').exists()],how='diagonal_relaxed')
    paired=observed.join(pred,on=['day','policy','id'],how='left',validate='1:1')
    assert paired['q_pass'].null_count()==0
    paired.write_parquet(output/'fill_diagnostics.parquet')
    summary=paired.group_by('policy').agg(pl.len().alias('observed_supply_fills'),
        pl.col('q_ready').sum().alias('fills_after_q_warmup'),pl.col('q_pass').sum().alias('q_pass_observed_fills'),
        pl.col('actual_premium_bp').filter(pl.col('q_pass')).mean().alias('q_pass_actual_entry_premium_bp'),
        pl.col('q_net_bp').filter(pl.col('q_pass')).mean().alias('q_pass_predicted_full_trade_net_bp'))
    summary.sort('policy').write_csv(output/'summary.csv')
    (output/'manifest.json').write_text(json.dumps(dict(status='completed',quotes=pred.height,
        interpretation='Opening Q table revalues every issued supply quote before reading outcomes. '
        'Joining actual supply fills afterwards is a diagnostic, not Q-gated/capital-constrained execution. '
        'Q-aware cancellations and subsequent quote paths can change which fills occur. '
        '0bp cohorts include quotes outside the original25bp training-policy support.'),indent=2)+'\n')
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('models',type=Path)
    a=p.parse_args();run(a.root,a.models)
