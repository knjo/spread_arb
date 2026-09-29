"""Evaluate recorded prior-only forecasts against actual portfolio releases."""
import json

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns


def write(root,name,days,available_days,output):
    predictions=[]
    for day in days:
        path=root/f'Date={day}'/name/'execution.parquet'
        if not path.exists():
            continue
        schema=pl.read_parquet_schema(path)
        if 'positions_json' not in schema:
            continue
        forecasts=(pl.scan_parquet(path).filter(pl.col('kind')=='capacity_forecast')
            .select('ns','next_session','positions_json').collect())
        seen=set()
        for forecast in forecasts.iter_rows(named=True):
            # One nonempty observed risk set per five-minute bucket, rather
            # than giving busier event loops more weight. No outcome is read.
            bucket=(forecast['ns']-open_ns(day))//(300*SECOND)
            risk=json.loads(forecast['positions_json'])
            if bucket in seen or not risk:
                continue
            seen.add(bucket)
            for p in risk:
                predictions.append(dict(day=day,ns=forecast['ns'],bucket=bucket,
                    next_session=forecast['next_session'],id=p['id'],nominal_twd=p['nominal_twd'],
                    predicted_today_twd=p['today'],predicted_next_twd=p['next_session']))
    if not predictions:
        return []
    # Labels enter only after all frozen predictions have been collected.
    labels={p['id']:p for p in pl.read_parquet(output/'position_returns.parquet').iter_rows(named=True)}
    for r in predictions:
        p=labels[r['id']]
        assert p['close_ns'] is None or p['close_ns']>r['ns'], 'forecast retained an already closed pair'
        market_close=p['close_ns'] is not None and p['close_kind']!='expiry_basis_zero_accounting'
        today=market_close and p['close_day']==r['day']
        next_known=r['next_session'] in available_days
        nxt=market_close and p['close_day']==r['next_session'] if next_known else None
        r.update(stream=p['stream'],carry=p['entry_day']<r['day'],
            actual_today_twd=r['nominal_twd']*today,
            actual_next_twd=r['nominal_twd']*nxt if nxt is not None else None)
    frame=pl.from_dicts(predictions,infer_schema_length=None)
    frame.write_parquet(output/'actual_release_forecasts.parquet')
    by=['stream','carry']
    summary=frame.group_by(by).agg(pl.len().alias('risk_observations'),pl.col('nominal_twd').sum(),
        *[(pl.col(k).sum()/pl.col('nominal_twd').sum()).alias(k.replace('_twd','_fraction'))
          for k in ('predicted_today_twd','actual_today_twd')],
        *[(pl.col(k).filter(pl.col('actual_next_twd').is_not_null()).sum()/
           pl.col('nominal_twd').filter(pl.col('actual_next_twd').is_not_null()).sum())
          .alias(k.replace('_twd','_fraction')) for k in ('predicted_next_twd','actual_next_twd')])
    summary=summary.with_columns(pl.col(pl.Float64).fill_nan(None))
    summary.write_csv(output/'actual_release_calibration.csv')
    frame.group_by('day','bucket').agg(pl.col('nominal_twd').sum(),
        pl.col('predicted_today_twd').sum(),pl.col('actual_today_twd').sum()).write_csv(output/'actual_release_daily.csv')
    return summary.to_dicts()
