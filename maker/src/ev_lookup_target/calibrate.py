"""Walk forward through daily Q snapshots; evaluate future outcomes only afterwards."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import polars as pl

from ..ev_lookup_cost.forecast_calendar import known_calendar
from .q_model import QSnapshot
from .target import ReturnTarget


def run(facts: Path, output: Path):
    manifest = json.loads((facts/'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('facts incomplete')
    output.mkdir(parents=True, exist_ok=False)
    predictions, endings, folds = [], [], []
    for day in manifest['days']:
        rows = pl.read_parquet(facts/f'Date={day}/risk.parquet')
        entered = rows.filter(pl.col('phase') == 'entry')
        closed = rows.filter(pl.col('event') != 'survive')
        endings.extend(closed.select('id','day','event','elapsed_calendar_days','outcome_net_bp',
                                     'outcome_net_bp_on_quote_capital','outcome_twd').to_dicts())
        for conditional, name in ((False, 'pooled'), (True, 'conditional')):
            snapshot = QSnapshot.fit(facts, day, conditional=conditional)
            folder = output/name
            folder.mkdir(exist_ok=True)
            (folder/f'{day}.json').write_text(json.dumps(snapshot.to_dict(),indent=2)+'\n')
            folds.append(dict(day=day, model=name, train_days=snapshot.train_days,
                              risk_cells=len(snapshot.risk), cost_cells=len(snapshot.costs), entries=entered.height))
            calendar = known_calendar(day)
            for row in entered.iter_rows(named=True):
                q = snapshot.estimate(stream=row['stream'],quote_second=row['quote_second'],
                    ab=row['quote_ab'],eff_u=row['quote_ab']-row['anchor'],expiry=row['expiry'],
                    spread_bp=row['quote_spread_bp'],quote_notional_twd=row['quote_notional_twd'],
                    execution_cost_bp=row['execution_cost_bp'],calendar=calendar)
                predictions.append(dict(id=row['id'],day=day,model=name,stream=row['stream'],
                    expiry=row['expiry'],warmup=snapshot.observed_prior_sessions<20, **asdict(q)))
        print(json.dumps(dict(day=day, entered=entered.height)),flush=True)
    pred = pl.from_dicts(predictions,infer_schema_length=None)
    outcome = pl.from_dicts(endings,infer_schema_length=None).rename({'day':'close_day','event':'actual_event',
                         'elapsed_calendar_days':'actual_calendar_days'})
    if outcome['id'].n_unique() != outcome.height:
        raise AssertionError('one entry closed more than once')
    joined = pred.join(outcome,on='id',how='left',validate='m:1').with_columns(
        pl.when(pl.col('actual_event')=='rollback').then(pl.col('outcome_net_bp_on_quote_capital'))
          .otherwise(pl.col('outcome_net_bp')).alias('actual_before_funding_bp'))
    joined = joined.with_columns((pl.col('actual_before_funding_bp')-
        pl.col('actual_calendar_days')*ReturnTarget().funding_annual_rate*10_000/365).alias('actual_net_bp'),
        (pl.col('expiry')<manifest['days'][-1]).alias('mature_contract'))
    joined.write_parquet(output/'forecasts.parquet')
    mature = joined.filter(pl.col('mature_contract') & ~pl.col('warmup'))
    summary = mature.group_by('model','stream').agg(pl.len().alias('fills'),
        pl.col('actual_event').is_null().sum().alias('unresolved'),
        pl.col('net_bp').mean(),pl.col('actual_net_bp').mean(),
        pl.col('expected_calendar_days').mean(),pl.col('actual_calendar_days').mean(),
        *[pl.col(c).mean() for c in ['p_sd','p_overnight','p_expiry','p_rollback','p_other']],
        ((pl.col('actual_event')=='normal')&(pl.col('day')==pl.col('close_day'))).mean().alias('actual_sd'),
        ((pl.col('actual_event')=='normal')&(pl.col('day')!=pl.col('close_day'))).mean().alias('actual_overnight'),
        (pl.col('actual_event')=='expiry').mean().alias('actual_expiry'),
        (pl.col('actual_event')=='rollback').mean().alias('actual_rollback'))
    summary.write_csv(output/'calibration.csv')
    gates = []
    for field in ('net_bp','surplus_bp'):
        gated = mature.with_columns(pl.lit(field).alias('gate'),(pl.col(field)>=0).alias('admit'))
        gates.append(gated)
    gate_rows = pl.concat(gates).with_columns(pl.col('day').str.slice(0,6).alias('entry_month'))
    gate_rows.group_by('model','stream','gate','admit').agg(pl.len().alias('fills'),pl.col('actual_net_bp').mean(),
        pl.col('actual_calendar_days').mean()).write_csv(output/'selection.csv')
    gate_rows.group_by('model','stream','gate','admit','entry_month').agg(pl.len().alias('fills'),
        pl.col('actual_net_bp').mean(),pl.col('actual_calendar_days').mean()).write_csv(output/'selection_monthly.csv')
    result = dict(status='completed',days=manifest['days'],facts=str(facts.resolve()),folds=folds,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')},
                  interpretation='Causal predictions fitted from prior sessions; future close labels joined only for evaluation. '
                  'Conditional-on-fill shadow diagnostics, not capacity-constrained portfolio returns. '
                  'Warmup is 20 observed sessions. Mature contracts include rollbacks and preserve unresolved outcomes. '
                  'Funding comparison uses full-ticket duration for rollback as a conservative approximation; '
                  'final execution report must use actual cash-time funding.')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('facts',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    print(run(args.facts,args.output))


if __name__ == '__main__':
    main()
