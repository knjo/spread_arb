"""Compare actual profit and turnover under fixed Q and different S2 clocks."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.market import WF
from .event_turnover import write as write_turnover
from .event_release_calibration import write as write_release
from .event_capacity_timing import write as write_timing


def source_coverage(manifest,output):
    source=WF/'august_attribution_s0_20260824_v2/raw_order_facts.parquet'
    digest=hashlib.sha256()
    with source.open('rb') as handle:
        while chunk:=handle.read(1024*1024):
            digest.update(chunk)
    days=pl.scan_parquet(source).select('Date').unique().collect()['Date'].to_list()
    result=dict(s1_commands_source=str(source),sha256=digest.hexdigest(),s1_source_days=sorted(days),
        observed_sessions_without_s1_new_commands=[d for d in manifest['available_days'] if d not in days],
        interpretation='Unchanged exogenous S1 command source ends20260813; later sessions admit only new S2 '
            'and continue existing S1 carry/exits. Availability splits are descriptive, with positions and Q kept continuous.')
    path=output/'source_coverage.json'
    if path.exists():
        assert json.loads(path.read_text())['sha256']==result['sha256'],'S1 command source changed during this comparison'
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def availability_periods(folder,manifest,coverage,first_day):
    curve=pl.read_csv(folder/'equity_daily.csv',schema_overrides={'day':pl.String})
    turn=pl.read_csv(folder/'turnover_daily.csv',schema_overrides={'day':pl.String})
    groups=[]
    for day in manifest['days']:
        if day<first_day:
            continue
        mode='both_entry_sources' if day in coverage['s1_source_days'] else 's2_new_only_with_s1_carry'
        if not groups or groups[-1][0]!=mode:
            groups.append((mode,[]))
        groups[-1][1].append(day)
    result=[]
    for mode,days in groups:
        before=curve.filter(pl.col('day')<days[0])
        opening=before['net_equity_twd'][-1] if before.height else 0.
        ending=curve.filter(pl.col('day')==days[-1])['net_equity_twd'].item()
        n=sum(d in manifest['available_days'] for d in days)
        gain=ending-opening if ending is not None and opening is not None else None
        flow=turn.filter(pl.col('day').is_in(days))
        result.append(dict(mode=mode,first_day=days[0],last_day=days[-1],observed_sessions=n,
            net_change_twd=gain,net_per_observed_day_twd=gain/n if gain is not None and n else None,
            entries=flow['entries'].sum(),s2_entries=flow['s2_entries'].sum(),
            normal_close_cash_turns_per_20M_per_day=flow['normal_closed_original_stock_cash_twd'].sum()/n/20e6 if n else None))
    return result


def verified(root,event=False):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    checks=['full','decisions','capacity','shadow_equivalence']
    if event:
        checks.append('event_s2')
    for check in checks:
        assert json.loads((root/f'verification_{check}.json').read_text())['passed']
    return manifest


def cohort_statistics(folder,days):
    pos=pl.read_parquet(folder/'position_returns.parquet')
    # Only contracts whose original expiry has passed form the mature cohort.
    # Unresolved positions remain in the denominator and are reported.
    mature=pos.filter(pl.col('contract').struct.field('expiry')<days[-1])
    indices={day:i for i,day in enumerate(days)}
    rows=[]
    for stream in ('S1','S2'):
        sub=mature.filter(pl.col('stream')==stream)
        normal=sub.filter(pl.col('close_kind')=='maker_exit')
        nsd=sum(r['entry_day']==r['close_day'] for r in normal.iter_rows(named=True))
        nd1=sum(indices[r['close_day']]-indices[r['entry_day']]<=1 for r in normal.iter_rows(named=True))
        rows.append(dict(stream=stream,mature_entries=sub.height,unresolved=sub['close_ns'].null_count(),
            normal_same_day_fraction=nsd/sub.height if sub.height else None,
            normal_by_next_session_fraction=nd1/sub.height if sub.height else None,
            normal_closed=normal.height,
            normal_net_twd=normal['net_equity_twd'].sum(),
            expiry_closed=sub.filter(pl.col('close_kind')=='expiry_basis_zero_accounting').height))
    return rows


def weighted_calibration(folder):
    frame=pl.read_parquet(folder/'q_calibration_positions.parquet')
    closed=frame.filter(pl.col('mature')&pl.col('close_ns').is_not_null())
    result=closed.group_by('stream').agg(pl.len().alias('mature_closed'),
        pl.col('return_nominal_twd').sum(),pl.col('net_equity_twd').sum().alias('actual_net_twd'),
        ((pl.col('q_net_bp')*pl.col('return_nominal_twd')).sum()/pl.col('return_nominal_twd').sum())
            .alias('predicted_nominal_weighted_bp'),
        (pl.col('net_equity_twd').sum()/pl.col('return_nominal_twd').sum()*10_000)
            .alias('actual_nominal_weighted_bp'))
    result.write_csv(folder/'weighted_q_calibration.csv')
    return result.to_dicts()


def hedge_timing(folder,hedge_ms=50):
    pos=pl.read_parquet(folder/'position_returns.parquet').filter(pl.col('stream')=='S2')
    paired=pos.filter(pl.col('hedged_ns').is_not_null())
    delay=paired['hedged_ns']-paired['entry_fill_ns']
    assert delay.null_count()==0 and not (delay<hedge_ms*1_000_000).any()
    def milliseconds(value):
        return value/1_000_000 if value is not None else None
    return dict(s2_stock_hedged_entries=paired.height,
        s2_entries_without_completed_stock_hedge=pos.height-paired.height,
        s2_configured_hedge_delay_ms=hedge_ms,
        s2_stock_hedge_delay_p50_ms=milliseconds(delay.median()),
        s2_stock_hedge_delay_p95_ms=milliseconds(delay.quantile(.95,interpolation='nearest')),
        s2_stock_hedge_delay_max_ms=milliseconds(delay.max()),
        s2_stock_hedge_later_than_configured=(delay>hedge_ms*1_000_000).sum(),
        s2_stock_hedge_later_than5s=(delay>5_000_000_000).sum())


def plot(sources,output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hft-matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
    colors=['#555555','#0072B2','#D55E00']
    for color,(folder,label) in zip(colors,sources):
        equity=pl.read_csv(folder/'equity_daily.csv',schema_overrides={'day':pl.String})
        turn=pl.read_csv(folder/'turnover_daily.csv',schema_overrides={'day':pl.String})
        cap=pl.read_csv(folder/'capital_daily.csv',schema_overrides={'day':pl.String})
        dates=[datetime.strptime(day,'%Y%m%d') for day in equity['day']]
        axes[0].plot(dates,equity['net_equity_twd'],label=label,color=color)
        axes[1].plot(dates,turn['normal_closed_original_stock_cash_twd'].cum_sum()/20e6,
                     label=label,color=color)
        axes[2].plot(dates,cap['committed_peak_twd']/1e6,label=label,color=color)
    axes[0].set(title='S1 + S2 net equity: costs, final marks and 2% actual stock funding',ylabel='TWD')
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[1].set(title='Cumulative original stock cash closed by normal market exits',ylabel='Turns / allocated 20M')
    axes[2].set(title='Actual daily peak commitment, including cancellation races',ylabel='TWD million')
    axes[2].axhline(20,color='#777777',ls='--',lw=.8)
    axes[2].axhline(25,color='#777777',ls=':',lw=.8)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('Current fixed Q | chronological execution | immediate S2 recheck after stock hedge')
    fig.supxlabel('First20 sessions flat; S1 new commands end Aug13; C8 accounting excluded from normal turnover.',fontsize=9)
    fig.savefig(output/'profit_turnover_capacity.png',dpi=160)
    plt.close(fig)


def run(event_root,old_root,output):
    event=verified(event_root,event=True);old=verified(old_root)
    assert event['days']==old['days'] and event['available_days']==old['available_days']
    assert event['inputs_sha256']==old['inputs_sha256'], 'current-Q comparison requires identical frozen inputs'
    for path in event['sources'].keys() & old['sources'].keys():
        assert event['sources'][path]==old['sources'][path], 'shared execution source changed'
    selected=next(c for c in event['configurations'] if c['name']=='q_event_release25_deep')
    control=next(c for c in old['configurations'] if c['name']=='q_net_release25_deep')
    for key in selected.keys() | control.keys():
        if key not in {'name','event_s2'}:
            assert selected.get(key)==control.get(key),(key,'unmatched S2-clock comparator')
    output.mkdir(parents=True,exist_ok=True)
    coverage=source_coverage(event,output)
    results=[];sources=[];cohorts=[];releases=[];timings=[];periods=[];weighted=[]
    for root,manifest in ((old_root,old),(event_root,event)):
        turns={r['portfolio']:r for r in write_turnover(root,root.parent/'report',manifest)}
        for config in manifest['configurations']:
            name=config['name'];folder=root.parent/'report'/name
            summary=json.loads((folder/'summary.json').read_text())
            cap=pl.read_csv(folder/'capital_daily.csv',schema_overrides={'day':pl.String})
            cap=cap.filter(pl.col('day')>=summary['postwarm_first_day'])
            label=('Original S2 clock, release credit' if root==old_root else
                   'Immediate S2, release credit' if config.get('release_credit') else 'Immediate S2, base 20M')
            row=dict(**summary,**{k:v for k,v in turns[name].items() if k not in summary},
                label=label,source=str(root.resolve()),
                postwarm_mean_stock_cash_twd=cap['mean_stock_cash_twd'].mean(),
                postwarm_stock_cash_utilization_20M=cap['mean_stock_cash_twd'].mean()/20e6,
                postwarm_annual_250_on_peak_committed=summary['postwarm_net_per_observed_day_twd']*250/max(20e6,summary['peak_committed_twd']))
            row.update(hedge_timing(folder,config.get('hedge_ms',50)))
            results.append(row);sources.append((folder,label))
            cohorts.extend(dict(portfolio=name,**r) for r in cohort_statistics(folder,manifest['days']))
            weighted.extend(dict(portfolio=name,**r) for r in weighted_calibration(folder))
            timings.append(write_timing(root,name,manifest,folder,summary['postwarm_first_day']))
            periods.extend(dict(portfolio=name,**r) for r in availability_periods(
                folder,manifest,coverage,summary['postwarm_first_day']))
            if config.get('release_credit'):
                releases.extend(dict(portfolio=name,**r) for r in write_release(
                    root,name,manifest['days'],manifest['available_days'],folder))
    baseline=results[0]
    for row in results:
        row['postwarm_net_per_day_delta_vs_old_twd']=row['postwarm_net_per_observed_day_twd']-baseline['postwarm_net_per_observed_day_twd']
        for key in ('postwarm_entries_per_day','postwarm_s2_entries_per_day','postwarm_normal_close_cash_turns_per_20M_per_day'):
            row[key+'_ratio_vs_old']=row[key]/baseline[key] if baseline[key] else None
    pl.from_dicts(results,infer_schema_length=None).write_csv(output/'policies.csv')
    pl.from_dicts(cohorts,infer_schema_length=None).write_csv(output/'mature_close_rates.csv')
    pl.from_dicts(releases,infer_schema_length=None).write_csv(output/'release_calibration.csv')
    pl.from_dicts(timings,infer_schema_length=None).write_csv(output/'capacity_timing.csv')
    pl.from_dicts(periods,infer_schema_length=None).write_csv(output/'availability_periods.csv')
    pl.from_dicts(weighted,infer_schema_length=None).write_csv(output/'weighted_q_calibration.csv')
    result=dict(status='completed',summaries=results,mature_close_rates=cohorts,release_calibration=releases,
        capacity_timing=timings,source_coverage=coverage,availability_periods=periods,weighted_q_calibration=weighted,
        comparison='Identical frozen opening Q/release inputs, common execution core and dates verified. '
            'Release-credit old/new policies differ only in S2 event timing and immediate recheck on completed stock hedge. '
            'Event base20 removes release credit as a capacity control. No actual future outcome determines admission.',
        accounting='All are whole S1+S2 portfolios, not incremental profit. Net equity includes unresolved inventory marks '
            'and actual stock cash-time funding at 2%; margin funding is not included. Entry stock flow, normal pair-close '
            'turnover and C8 expiry accounting release are separate. Planning annualization uses allocated20M and250sessions.',
        limitations='Exploratory, already studied historical interval; original-policy Q is kept fixed as requested. '
            'Immediate simulated queue entry and no message throttle can matter more with faster S2 quoting. '
            'C8 expiry basis-zero accounting is not raw-size spot execution. Retained races can exceed admission limits. '
            'Mandatory hedges execute the full remaining quantity across L1-L5 or retry with exposure retained; '
            'this model does not take a partial stock hedge when full size is unavailable. Completed S2 hedge delays '
            'and unresolved stock hedges are reported separately from quote-to-maker-fill latency. '
            'Release calibration uses the first nonempty observed risk set per5min; repeats are correlated descriptions, '
            'not independent samples. Unknown/outage next-session outcomes are excluded from both sides of comparison.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(sources,output)
    from .event_assessment import write as write_assessment
    write_assessment(output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('event_root',type=Path);p.add_argument('old_root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(run(a.event_root,a.old_root,a.output),indent=2))
