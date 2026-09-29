"""Evaluate frozen release probabilities after forecasts; labels never select trades."""
import argparse
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.forecast_calendar import known_calendar
from ..q_facts import day_distance
from ..q_model import settlement_day
from ..release_model import ReleaseSnapshot


def run(models,output):
    manifest=json.loads((models/'manifest.json').read_text())
    assert manifest['status']=='completed'
    facts=json.loads((Path(manifest['facts'])/'manifest.json').read_text())
    days=manifest['days']
    observations={day:pl.read_parquet(models/f'Date={day}/release_observations.parquet') for day in days}
    forecasts=[]
    # Only observable features and frozen prior tables are used in this pass.
    for day in days:
        model=ReleaseSnapshot(**json.loads((models/f'Date={day}/release_snapshot.json').read_text()))
        q=json.loads((models/f'Date={day}/q_snapshot.json').read_text())
        if q['observed_prior_sessions']<20:
            continue
        next_day=settlement_day(day,known_calendar(day))
        for second in (0,3600,7200,10800):
            alive=observations[day].filter((pl.col('start_second')<=second)&(pl.col('end_second')>second))
            for row in alive.iter_rows(named=True):
                today=model.probability(stream=row['stream'],carry=row['carry'],second=second,
                    days_to_expiry=row['days_to_expiry'],premium=row['premium'])
                remaining=row['days_to_expiry']-day_distance(day,next_day)
                nxt=model.probability(stream=row['stream'],carry=True,second=0,
                    days_to_expiry=remaining,premium=row['premium']) if remaining>=0 else {'discounted':0.}
                forecasts.append(dict(day=day,next_day=next_day,id=row['id'],second=second,stream=row['stream'],
                    carry=row['carry'],nominal_twd=row['nominal_twd'],predicted_mean=today['mean'],
                    predicted_discounted=today['discounted'],samples=today['samples'],
                    predicted_next_discounted=(1-today['mean'])*nxt['discounted']))
    # Joining future outcomes is evaluation only, after every prediction exists.
    labels={(day,row['id']):row['closed'] for day,rows in observations.items() for row in rows.iter_rows(named=True)}
    for row in forecasts:
        row['actual_today']=labels[row['day'],row['id']]
        row['actual_next']=(not row['actual_today'] and labels.get((row['next_day'],row['id']),False)
                            if row['next_day'] in days and row['next_day'] not in facts['outages'] else None)
    frame=pl.from_dicts(forecasts,infer_schema_length=None)
    output.mkdir(parents=True,exist_ok=True)
    frame.write_parquet(output/'forecasts.parquet')
    frame=frame.with_columns(*[(pl.col(k)*pl.col('nominal_twd')).alias(k+'_twd') for k in
        ('predicted_mean','predicted_discounted','predicted_next_discounted','actual_today','actual_next')])
    summary=frame.group_by('second','carry','stream').agg(pl.len().alias('observations'),
        pl.col('nominal_twd').sum(),pl.col('samples').min(),
        *[(pl.col(k+'_twd').sum()/pl.col('nominal_twd').sum()).alias(k+'_weighted_fraction') for k in
          ('predicted_mean','predicted_discounted','actual_today')],
        (pl.col('predicted_next_discounted_twd').filter(pl.col('actual_next').is_not_null()).sum()/
         pl.col('nominal_twd').filter(pl.col('actual_next').is_not_null()).sum()).alias('predicted_next_weighted_fraction'),
        (pl.col('actual_next_twd').sum()/pl.col('nominal_twd').filter(pl.col('actual_next').is_not_null()).sum())
         .alias('actual_next_weighted_fraction'))
    summary.write_csv(output/'calibration.csv')
    frame.group_by('day','second').agg(pl.col('nominal_twd').sum(),pl.col('predicted_discounted_twd').sum(),
        pl.col('actual_today_twd').sum()).with_columns(
            ((pl.col('predicted_discounted_twd')-pl.col('actual_today_twd'))/pl.col('nominal_twd'))
            .alias('discounted_overprediction_fraction')).write_csv(output/'daily_errors.csv')
    (output/'manifest.json').write_text(json.dumps(dict(status='completed',rows=len(forecasts),
        interpretation='Prior-only predictions on the independent shadow inventory actually still held at each timestamp; '
            'same-day and next-session labels joined only afterwards. Correlated repeated risk sets are descriptive '
            'calibration, not an independent-sample confidence guarantee or actual portfolio return.'),indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('models',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    print(run(a.models,a.output))
