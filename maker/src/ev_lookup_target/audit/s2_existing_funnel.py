"""Count the original S2 universe before confusing it with the capital/Q selection."""
import argparse
import json
from pathlib import Path

import polars as pl


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    facts=root/'q_facts'
    manifest=json.loads((facts/'manifest.json').read_text())
    original=Path(manifest['source'])
    daily=[]
    for day in manifest['days']:
        q=(pl.scan_parquet(facts/f'Date={day}/quotes.parquet').filter(pl.col('stream')=='S2')
            .select('filled','paired_by_close','quote_ab','anchor','execution_cost_bp').collect())
        filled=q.filter(pl.col('filled'))
        daily.append(dict(day=day,shadow_quoted=q.height,shadow_fills=filled.height,
            paired_by_day_end=filled['paired_by_close'].sum(),
            mean_quote_premium_bp=(filled['quote_ab']-filled['anchor']).mean(),
            mean_quote_adverse_scenario_bp=filled['execution_cost_bp'].mean()))
    daily=pl.from_dicts(daily,infer_schema_length=None)
    old=pl.read_csv(original/'cost_guard_20M_daily.csv',schema_overrides={'day':pl.String})
    daily=daily.join(old.select('day',pl.col('fills_s2').alias('old_actual_s2_fills')),on='day',validate='1:1')
    predictions=(pl.read_parquet(root/'q_calibration_current/forecasts.parquet')
        .filter((pl.col('model')=='conditional') & (pl.col('stream')=='S2')))
    selection=predictions.group_by('day').agg(pl.len().alias('predicted_shadow_fills'),
        pl.col('warmup').sum().alias('warmup_fills'),
        ((~pl.col('warmup')) & (pl.col('net_bp')>=0)).sum().alias('net_q_eligible_fills'),
        ((~pl.col('warmup')) & (pl.col('surplus_bp')>=0)).sum().alias('calendar_target_eligible_fills'))
    daily=daily.join(selection,on='day',how='left').sort('day')
    daily.write_csv(output/'daily.csv')
    cols=['shadow_quoted','shadow_fills','old_actual_s2_fills','warmup_fills',
          'net_q_eligible_fills','calendar_target_eligible_fills']
    monthly=daily.with_columns(pl.col('day').str.slice(0,6).alias('month')).group_by('month').agg(
        pl.len().alias('calendar_rows'),*[pl.col(c).sum() for c in cols]).sort('month')
    monthly.write_csv(output/'monthly.csv')
    result=dict(days=len(manifest['days']),observed_days=len(manifest['days'])-len(manifest['outages']),
        counts={c:daily[c].sum() for c in cols},
        scope='Original shadow already has25bp residual, depth5,60-second cooldown and old event-dispatch restrictions. '
              'Repeated submitted/cancelled quotes are not independent fills. Q counts revalue the original '
              'shadow entry universe using prior-only forecasts before outcome joins, without maturity filtering. '
              'Neither shadow nor Q-eligible counts are a capital-constrained executable portfolio. '
              'Observed dates exclude the known outage; warmup is retained explicitly.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.root,a.output)
