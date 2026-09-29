"""Allocated-capital return, actual funding, capacity and Q calibration reports."""
import argparse
import json
from pathlib import Path

import polars as pl

from ...ev_lookup.audit.execution_costs import filled_positions
from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .capital import analyze as capital_analysis


def portfolio(root,name,days,available_days,output):
    daily=pl.read_csv(root/f'{name}_daily.csv',schema_overrides={'day':pl.String})
    assert daily['day'].to_list()==days
    capital,areas=capital_analysis(root,name,days)
    cap=pl.from_dicts(capital)
    out=output/name
    out.mkdir(parents=True,exist_ok=True)
    cap.write_csv(out/'capital_daily.csv')
    pos=filled_positions(root,name,days)
    marks_path=root/f'Date={days[-1]}'/name/'marks.parquet'
    marks=(pl.read_parquet(marks_path) if marks_path.exists() else
           pl.DataFrame(schema={'position_id':pl.String,'official_mark_twd':pl.Float64}))
    if marks.is_empty():
        marks=pl.DataFrame(schema={'position_id':pl.String,'official_mark_twd':pl.Float64})
    assert pos.height>0, 'report no-fill portfolios explicitly before using trade calibration'
    pos=pos.join(marks.select('position_id','official_mark_twd'),left_on='id',right_on='position_id',how='left',validate='1:1')
    pos=pos.with_columns(pl.col('id').replace_strict(areas,default=0.,return_dtype=pl.Float64).alias('capital_days_twd'))
    pos=pos.with_columns((pl.col('capital_days_twd')*.02/365).alias('funding_twd'),
        pl.when(pl.col('close_ns').is_not_null()).then(pl.col('pnl_twd')).otherwise(pl.col('official_mark_twd')).alias('equity_twd'))
    assert pos['equity_twd'].null_count()==0, 'unmarked final exposure cannot become zero PnL'
    pos=pos.with_columns((pl.col('equity_twd')-pl.col('funding_twd')).alias('net_equity_twd'))
    pos=pos.with_columns(((pl.col('entry_fill_ns')-pl.col('quote_ns'))/1e6).alias('quote_to_fill_ms'))
    pos.write_parquet(out/'position_returns.parquet')
    funding=pos['funding_twd'].sum()
    equity=pos['equity_twd'].sum()
    assert daily['official_equity_twd'][-1] is not None
    assert abs(equity-daily['official_equity_twd'][-1])<1e-5
    assert abs(funding-cap['funding_2pct_twd'].sum())<1e-6
    net=equity-funding
    n=len(available_days)
    elapsed=(open_ns(days[-1])+15600*SECOND-open_ns(days[0]))/SECOND/86400
    final=daily.row(-1,named=True)
    sample=pos.filter(pl.col('contract').struct.field('expiry')<days[-1])
    indices={d:i for i,d in enumerate(days)}
    normal=sample.filter(pl.col('close_kind')=='maker_exit')
    known=pos.filter(pl.col('close_ns').is_not_null())
    ratios=dict(mature_fills=sample.height,mature_unresolved=sample['close_ns'].null_count(),
        mature_normal_sd_fraction=normal.filter(pl.col('entry_day')==pl.col('close_day')).height/sample.height if sample.height else None,
        mature_normal_by_next_fraction=sum(indices[r['close_day']]-indices[r['entry_day']]<=1
                                        for r in normal.iter_rows(named=True))/sample.height if sample.height else None)
    complete=daily['official_equity_twd'].null_count()==0
    curve=daily.join(cap.select('day','funding_2pct_twd'),on='day').with_columns(
        (pl.col('official_equity_twd')-pl.col('funding_2pct_twd').cum_sum()).alias('net_equity_twd'))
    curve.write_csv(out/'equity_daily.csv')
    peak=drawdown=0.
    for value in curve['net_equity_twd']:
        if value is not None:
            peak=max(peak,value)
            drawdown=max(drawdown,peak-value)
    result=dict(portfolio=name,observed_sessions=n,calendar_sessions=len(days),elapsed_calendar_days=elapsed,
        allocated_capital_twd=20_000_000,realized_twd=daily['realized_twd'].sum(),
        final_marked_inventory_twd=final['official_marked_open_twd'],equity_before_funding_twd=equity,
        actual_cash_funding_2pct_twd=funding,net_equity_twd=net,net_per_observed_day_twd=net/n,
        simple_annual_250_sessions=net/n*250/20_000_000,
        simple_annual_calendar=net/elapsed*365/20_000_000,
        simple_annual_250_on_peak_committed=net/n*250/max(20_000_000,cap['committed_peak_twd'].max()),
        target_daily_twd=24_000.,daily_target_gap_twd=24_000-net/n,
        triggered_entries=pos.height,completed_entries=known.height,
        mean_net_completed_trade_twd=known['net_equity_twd'].mean(),
        target_net_per_completed_trade_twd=24_000*n/known.height if known.height else None,
        expiry_closed=pos.filter(pl.col('close_kind')=='expiry_basis_zero_accounting').height,
        expiry_net_twd=pos.filter(pl.col('close_kind')=='expiry_basis_zero_accounting')['net_equity_twd'].sum(),
        partial_rollbacks=pos.filter(pl.col('close_kind')=='partial_entry_rollback').height,
        mean_stock_cash_twd=cap['mean_stock_cash_twd'].mean(),mean_committed_twd=cap['mean_committed_twd'].mean(),
        peak_stock_cash_twd=cap['stock_cash_peak_twd'].max(),peak_committed_twd=cap['committed_peak_twd'].max(),
        extra_committed_above_20M_twd=max(0.,cap['committed_peak_twd'].max()-20_000_000),
        extra_cash_above_20M_twd=max(0.,cap['stock_cash_peak_twd'].max()-20_000_000),
        committed_over20_intraday_seconds=cap['committed_over20_intraday_seconds'].sum(),
        committed_over25_intraday_seconds=cap['committed_over25_intraday_seconds'].sum(),
        committed_over20_calendar_seconds=cap['committed_over20_calendar_seconds'].sum(),
        committed_over25_calendar_seconds=cap['committed_over25_calendar_seconds'].sum(),
        final_carry_twd=final['carry_twd'],final_unhedged=final['unhedged'],
        missing_daily_official_marks=daily['official_equity_twd'].null_count(),
        max_drawdown_twd=drawdown if complete else None,observed_drawdown_lower_bound_twd=drawdown,**ratios)
    for venue in ('F','S'):
        result.update({f'{venue}_outbound_messages':cap[f'{venue}_outbound_messages'].sum(),
            f'{venue}_max_messages_per_second':cap[f'{venue}_max_messages_per_second'].max(),
            f'{venue}_seconds_above_reference':cap[f'{venue}_seconds_above_reference'].sum(),
            f'{venue}_messages_in_over_reference_seconds':cap[f'{venue}_messages_in_over_reference_seconds'].sum()})
    streams=pos.group_by('stream').agg(pl.len().alias('fills'),pl.col('equity_twd').sum(),
        pl.col('funding_twd').sum(),pl.col('net_equity_twd').sum()).with_columns(
            (pl.col('net_equity_twd')/n).alias('net_per_observed_day_twd'))
    streams.write_csv(out/'streams.csv')
    pos.group_by('stream').agg(pl.len().alias('maker_entries'),
        (pl.col('quote_to_fill_ms')<50).sum().alias('under50ms'),
        pl.col('quote_to_fill_ms').median().alias('median_ms'),
        pl.col('net_equity_twd').filter(pl.col('quote_to_fill_ms')<50).sum().alias('under50ms_net_twd'))\
        .write_csv(out/'quote_latency_exposure.csv')
    monthly=[]
    previous=0.
    for month in sorted({d[:6] for d in days}):
        rows=curve.filter(pl.col('day').str.starts_with(month))
        ending=rows['net_equity_twd'][-1]
        monthly.append(dict(portfolio=name,month=month,
            observed_sessions=sum(d.startswith(month) for d in available_days),
            net_change_twd=ending-previous if ending is not None and previous is not None else None,
            ending_net_equity_twd=ending))
        previous=ending
    pl.from_dicts(monthly).write_csv(out/'monthly.csv')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,pos,curve


def q_calibration(root,name,days,pos,output):
    cols=['intent_id','ns','reservation_cents','q_net_bp','q_days','q_hurdle_bp','q_required_bp','q_entry_samples','q_entry_key',
          'p_sd','p_overnight','p_expiry','p_rollback','p_other']
    frames=[]
    for day in days:
        path=root/f'Date={day}'/name/'decisions.parquet'
        if path.exists():
            frames.append(pl.scan_parquet(path).filter(pl.col('admit')).select(cols).collect())
    decisions=pl.concat(frames,how='diagonal_relaxed')
    joined=pos.join(decisions,left_on=['id','quote_ns'],right_on=['intent_id','ns'],how='left',validate='1:1')
    assert joined['q_net_bp'].null_count()==0, 'filled order must match its exact original quote decision'
    joined=joined.with_columns(pl.when(pl.col('close_kind')=='partial_entry_rollback')
        .then(pl.col('reservation_cents')/100).otherwise(pl.col('spot_buy_cash')/10_000).alias('return_nominal_twd'))
    joined=joined.with_columns((pl.col('contract').struct.field('expiry')<days[-1]).alias('mature'),
        pl.when(pl.col('close_ns').is_not_null()).then(
            (pl.col('pnl_twd')-pl.col('funding_twd'))/pl.col('return_nominal_twd')*10_000
            ).otherwise(None).alias('actual_net_bp'),
        pl.when(pl.col('close_ns').is_not_null()).then(
            (pl.col('close_ns')-pl.col('entry_fill_ns'))/SECOND/86400).otherwise(None).alias('actual_days'))
    joined.write_parquet(output/name/'q_calibration_positions.parquet')
    mature=joined.filter(pl.col('mature'))
    mature.group_by('stream').agg(pl.len().alias('fills'),pl.col('close_ns').is_null().sum().alias('unresolved'),
        pl.col('q_net_bp').mean(),pl.col('actual_net_bp').mean(),pl.col('q_days').mean(),pl.col('actual_days').mean(),
        pl.col('p_sd').mean(),pl.col('p_overnight').mean(),pl.col('p_expiry').mean(),
        ((pl.col('close_kind')=='maker_exit')&(pl.col('entry_day')==pl.col('close_day'))).mean().alias('actual_sd'),
        ((pl.col('close_kind')=='maker_exit')&(pl.col('entry_day')!=pl.col('close_day'))).mean().alias('actual_overnight'),
        (pl.col('close_kind')=='partial_entry_rollback').mean().alias('actual_rollback'),
        (pl.col('close_kind')=='expiry_basis_zero_accounting').mean().alias('actual_expiry')).write_csv(output/name/'q_calibration.csv')


def run(root,output,*,baseline=False):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed', 'no partial-period headline'
    if not baseline:
        for kind in ('full','decisions','capacity'):
            assert json.loads((root/f'verification_{kind}.json').read_text())['passed']
    else:
        assert json.loads((root/'verification_cost.json').read_text())['passed']
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    warm_days=[]
    if not baseline:
        model_manifest=json.loads((root/'input_snapshot/000_manifest.json').read_text())
        if manifest['days']==model_manifest['days']:
            from .shadow_equivalence import verify as verify_shadow
            verify_shadow(root)
        warm_days=[r['day'] for r in model_manifest['counts']
                   if r['observed_prior_sessions']>=20 and r['day'] in manifest['days']]
    for config in manifest['configurations']:
        result,pos,curve=portfolio(root,config['name'],manifest['days'],manifest['available_days'],output)
        if config.get('q_policy'):
            q_calibration(root,config['name'],manifest['days'],pos,output)
            first=warm_days[0]
            before=curve.filter(pl.col('day')<first)
            opening=before['net_equity_twd'][-1] if before.height else 0.
            assert opening is not None
            gain=result['net_equity_twd']-opening
            n=sum(d>=first for d in manifest['available_days'])
            elapsed=(open_ns(manifest['days'][-1])+15600*SECOND-open_ns(first))/SECOND/86400
            result.update(postwarm_first_day=first,postwarm_observed_sessions=n,postwarm_net_change_twd=gain,
                postwarm_net_per_observed_day_twd=gain/n,postwarm_simple_annual_250=gain/n*250/20_000_000,
                postwarm_simple_annual_calendar=gain/elapsed*365/20_000_000,
                postwarm_daily_target_gap_twd=24_000-gain/n)
            (output/config['name']/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            stream_path=output/config['name']/'streams.csv'
            pl.read_csv(stream_path).with_columns((pl.col('net_equity_twd')/n)
                .alias('postwarm_net_per_observed_day_twd')).write_csv(stream_path)
        rows.append(result)
    pl.from_dicts(rows,infer_schema_length=None).write_csv(output/'comparison.csv')
    (output/'comparison.json').write_text(json.dumps(dict(source=str(root.resolve()),summaries=rows,
        funding='2% hypothetical annual rate, original cost of outstanding actual stock inventory from each fill '
                'until its sale; includes partial/rollback/calendar carry. Futures margin and funding of fees '
                'are not separately modeled. Ledger capacity remains committed until both exit legs complete.',
        returns='Whole S1+S2 after transaction costs, realized execution and official final inventory marks; '
                '20M allocated denominator. 250 sessions is a planning convention, not a calendar claim.',
        messages='Outbound quote/cancel/hedge counts are lower bounds: unlogged source-intent cancellations '
                 'that lose a fill race cannot be recovered. 5 future / 100 stock messages per second are '
                 'diagnostic reference levels, not enforced replay throttles. Immediate new queue entry '
                 'and quote-to-fill exposures below50ms are reported separately.'),indent=2)+'\n')
    if not baseline:
        from .plot import draw
        draw(root,output)
        from .compare import maybe_combine
        maybe_combine(Path(model_manifest['facts']).parent)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.output,baseline=args.baseline),indent=2))
