"""Reconcile all S1 source commands with submits/cancels and prior boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from ...ev_lookup_cost.market import WF
from ...ev_lookup_cost.replay import S1_COLUMNS
from ...quote_fill.one_second_makerfill_runner import tick_index_to_price


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def verify(output):
    folder=WF/'august_attribution_s0_20260824_v2'
    config=json.loads((folder/'run_config.json').read_text())
    path=folder/'raw_order_facts.parquet'
    raw=pl.read_parquet(path)
    keys=['Date','ValueCode','generation']
    required=S1_COLUMNS+['generation','absolute_price_tick','upper_distance_bp','lower_distance_bp']
    assert not any(raw.select(required).null_count().row(0)), 'missing command or boundary metadata'
    assert raw.select(pl.all_horizontal(pl.col('target_price','upper_distance_bp','lower_distance_bp')
        .is_finite()).all()).item(), 'nonfinite command or boundary metadata'
    assert raw.select(keys).n_unique()==raw.height
    assert raw.filter(pl.col('boundary_source_asof_date').is_null()|
        (pl.col('boundary_source_asof_date')>=pl.col('Date'))).is_empty()
    assert S1_COLUMNS==['raw_order_fact_id','Date','ValueCode','QuoteCode','target_price',
        'nominal_new_time_ns','nominal_stop_time_ns','boundary_source_asof_date']
    assert raw.filter((pl.col('target_price')-tick_index_to_price(
        pl.col('absolute_price_tick'),market='spot')).abs()>1e-8).is_empty()
    boundaries=pl.read_parquet(config['paths']['boundary_path']).filter(
        (pl.col('boundary_quantile')==95)&pl.col('execution_safe_snapshot')&
        ~pl.col('contains_target_day_outcome')).select('Date','ValueCode','QuoteCode',
            pl.col('source_asof_date').alias('boundary_check_asof'),
            pl.col('upper_distance_bp').alias('boundary_check_upper'),
            pl.col('lower_distance_bp').alias('boundary_check_lower'))
    checked=raw.join(boundaries,on=['Date','ValueCode','QuoteCode'],how='left',validate='m:1')
    assert not any(checked.select('boundary_check_asof','boundary_check_upper','boundary_check_lower')
        .null_count().row(0)), 'missing prior boundary match'
    assert checked.filter((pl.col('boundary_source_asof_date')!=pl.col('boundary_check_asof'))|
        ((pl.col('upper_distance_bp')-pl.col('boundary_check_upper')).abs()>1e-8)|
        ((pl.col('lower_distance_bp')-pl.col('boundary_check_lower')).abs()>1e-8)).is_empty()
    days=sorted(raw['Date'].unique().to_list())
    evidence=[]
    for day in days:
        message=Path(config['paths']['message_root'])/f'Date={day}'/'message_events.parquet'
        events=pl.read_parquet(message).filter(pl.col('scenario_id')=='ab12_entry_until_1300')
        submits=events.filter((pl.col('kind')=='submit')&(pl.col('second_from_open')<14398))
        cancels=events.filter(pl.col('kind')=='cancel')
        sub=raw.filter(pl.col('Date')==day)
        assert sub.height==submits.height,(day,'submitted orders were filtered')
        assert submits.select(keys).n_unique()==submits.height
        assert sub.join(submits.select(keys),on=keys,how='anti').is_empty()
        events=pl.concat([
            submits.select(*keys,pl.col('absolute_price_tick').alias('check_price_tick'),
                pl.col('second_from_open').alias('check_second'),pl.lit('submit').alias('check_kind')),
            cancels.join(sub.select(keys),on=keys,how='inner',validate='1:1').select(
                *keys,pl.col('absolute_price_tick').alias('check_price_tick'),
                pl.col('second_from_open').alias('check_second'),pl.lit('cancel').alias('check_kind'))])
        assert events.height==sub.height*2
        joined=events.join(sub.select(*keys,'absolute_price_tick','nominal_new_time_ns','nominal_stop_time_ns'),
            on=keys,how='left',validate='m:1').with_columns(
                (pl.lit(open_ns(day))+pl.col('check_second').cast(pl.Int64)*SECOND).alias('check_ns'))
        assert joined.filter((pl.col('absolute_price_tick')!=pl.col('check_price_tick'))|
            (pl.when(pl.col('check_kind')=='submit').then(pl.col('nominal_new_time_ns'))
                .otherwise(pl.col('nominal_stop_time_ns'))!=pl.col('check_ns'))).is_empty(),day
        evidence.append(dict(day=day,commands=sub.height,message_source=str(message),sha256=digest(message)))
    result=dict(passed=True,scope='All existing S1 command rows, original submit/cancel timestamps and prior '
        'q95 boundary metadata. Actual makerFill outcomes and lifecycle fields are not replay admission inputs. '
        'This does not replace the full event replay execution/Q/capacity audit or rebuild every upstream raw feature.',
        source=str(path),sha256=digest(path),source_days=len(days),commands=raw.height,
        replay_loaded_columns=S1_COLUMNS,all_submitted_orders_retained=True,
        nominal_stop_matches_original_cancel_event=True,
        target_price_matches_submitted_price_tick=True,prior_boundary_metadata_matches=True,
        source_legacy_outcome_counts=raw.group_by('legacy_nominal_outcome_status').len().sort(
            'legacy_nominal_outcome_status').to_dicts(),
        lineage_note=config['legacy_input_provenance_caveat'],daily=evidence)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='daily'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
