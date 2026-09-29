"""Compare calendar and entry-window hurdles without changing future outcome labels."""
import argparse
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from ...ev_lookup_cost.forecast_calendar import known_calendar
from ..q_model import QSnapshot
from .entry_window import DAILY_HURDLE_BP, ENTRY_SECONDS, sessions


def run(models,calibration,output):
    manifest=json.loads((models/'manifest.json').read_text())
    facts=Path(manifest['facts'])
    predictions=[]
    for day in manifest['days']:
        q=QSnapshot(**json.loads((models/f'Date={day}/q_snapshot.json').read_text()))
        calendar=known_calendar(day)
        # Diagnostic cohort is conditional on a shadow maker fill. The model
        # receives quote features only; actual outcomes are joined afterwards.
        rows=pl.read_parquet(facts/f'Date={day}/risk.parquet').filter(pl.col('phase')=='entry').select(
            'id','stream','quote_second','quote_ab','anchor','expiry','quote_spread_bp','quote_notional_twd')
        for row in rows.iter_rows(named=True):
            duration=sessions(q,stream=row['stream'],quote_second=row['quote_second'],ab=row['quote_ab'],
                eff_u=row['quote_ab']-row['anchor'],expiry=row['expiry'],spread_bp=row['quote_spread_bp'],
                quote_notional_twd=row['quote_notional_twd'],calendar=calendar)
            predictions.append(dict(id=row['id'],expected_entry_window_sessions=duration,
                                    entry_window_hurdle_bp=duration*DAILY_HURDLE_BP))
    frame=pl.from_dicts(predictions)
    outcomes=pl.read_parquet(calibration/'forecasts.parquet').filter(pl.col('model')=='conditional')
    frame=frame.join(outcomes,on='id',validate='1:1')
    starts,ends={},{}
    for day in manifest['days']:
        for row in pl.read_parquet(facts/f'Date={day}/risk.parquet').select(
            'id','phase','event','risk_start_ns','risk_end_ns').iter_rows(named=True):
            if row['phase']=='entry':
                starts[row['id']]=row['risk_start_ns']
            if row['event']!='survive':
                ends[row['id']]=row['risk_end_ns']
    windows=[(open_ns(day),open_ns(day)+ENTRY_SECONDS*SECOND) for day in manifest['days']]
    durations={key:sum(max(0,min(end,right)-max(starts[key],left)) for left,right in windows)/SECOND/ENTRY_SECONDS
               for key,end in ends.items()}
    frame=frame.with_columns(pl.col('id').replace_strict(durations,default=None,return_dtype=pl.Float64)
                            .alias('actual_entry_window_sessions'))
    output.mkdir(parents=True,exist_ok=True)
    frame.write_parquet(output/'forecasts.parquet')
    eligible=frame.filter(pl.col('mature_contract') & ~pl.col('warmup') & (pl.col('entry_samples')>=50))
    summaries=[]
    for gate,condition in [('net',pl.col('net_bp')>=0),('calendar_target',pl.col('surplus_bp')>=0),
                           ('entry_window_target',pl.col('net_bp')>=pl.col('entry_window_hurdle_bp'))]:
        summary=eligible.filter(condition).group_by('stream').agg(pl.len().alias('fills'),
            pl.col('net_bp').mean().alias('predicted_net_bp'),pl.col('actual_net_bp').mean(),
            pl.col('expected_calendar_days').mean(),pl.col('actual_calendar_days').mean(),
            pl.col('expected_entry_window_sessions').mean(),pl.col('actual_entry_window_sessions').mean(),
            pl.col('actual_event').is_null().sum().alias('unresolved')).with_columns(pl.lit(gate).alias('gate'))
        summaries.append(summary)
    result=pl.concat(summaries)
    result.write_csv(output/'summary.csv')
    (output/'manifest.json').write_text(json.dumps(dict(status='completed',quotes=len(predictions),
        entry_window_seconds=ENTRY_SECONDS,daily_target_bp=DAILY_HURDLE_BP,
        interpretation='Predictions first, future labels afterwards. Equal-trade shadow diagnostics; '
            'neither capital-constrained execution nor proof of achieving30%. Funding remains calendar-based.'),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('models',type=Path)
    p.add_argument('calibration',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    print(run(a.models,a.calibration,a.output))
