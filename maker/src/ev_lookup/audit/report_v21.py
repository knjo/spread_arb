"""Summarize completed v21 mode outputs; no execution or model mutations."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import polars as pl

from ..analyze_full_study import analyze
from ..verify_full_study import read_rows
from ..causal_lookup import open_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    summary, reasons, cancellation_rows, calibration, forecasts, messages, selected = [], [], [], [], [], [], []
    hedges = []
    for mode in ("second","event"):
        root = args.root/mode
        analysis = analyze(root)
        manifest = json.loads((root/"manifest.json").read_text())
        snapshots = {(s['day'],s['portfolio']):s for s in json.loads((root/'snapshots.json').read_text())}
        for row in analysis["summaries"]:
            daily = pl.read_csv(root/f"{row['portfolio']}_daily.csv",schema_overrides={'day':pl.String})
            active = daily.filter(pl.col('fills')>0)
            summary.append(dict(mode=mode,**row,active_entry_days=active.height,
                                last_entry_day=active['day'].max()))
        for config in manifest["configurations"]:
            name = config["name"]
            invalid, fills, terminal, quote_predictions = {}, {}, {}, {}
            hedge_ready = {}
            for day in manifest["days"]:
                folder = root/f"Date={day}"/name
                trace = read_rows(folder/"execution.parquet")
                if mode == 'event':
                    messages.append(dict(portfolio=name,**maker_message_load(day,trace)))
                for t in trace:
                    pid = t["position_id"]
                    if t['kind']=='maker_fill':
                        purpose = ('entry_spot' if t['stream']=='S2' else 'entry_future') if t['purpose']=='entry' else 'exit_future'
                        hedge_ready[pid,purpose] = t['ns']
                    if t['kind']=='cancel' and t['purpose']=='entry':
                        hedge_ready[pid,'entry_rollback'] = t['ns']
                    if t['kind']=='taker_fill':
                        # Forced spot liquidations do not all have a recorded
                        # start event. Never infer their start from an older,
                        # unrelated exit-order cancellation. Futures hedge
                        # starts are observable in the actual stock fills.
                        ready = hedge_ready.get((pid,t['purpose']))
                        if ready is not None:
                            hedges.append(dict(mode=mode,portfolio=name,day=day,position_id=pid,
                                purpose=t['purpose'],ready_ns=ready,fill_ns=t['ns'],
                                latency_ms=(t['ns']-ready)/1e6))
                        if t['purpose']=='exit_spot_remainder':
                            hedge_ready[pid,'exit_future'] = t['ns']
                    if t["kind"] == "s2_first_invalid":
                        invalid[pid] = t
                    if t["kind"] == "maker_fill" and t["stream"] == "S2" and t["purpose"] == "entry":
                        fills[pid] = t
                for p in read_rows(folder/"positions.parquet"):
                    if p["entry_fill_ns"] is not None:
                        terminal[p["id"]] = p
                decisions = read_rows(folder/"decisions.parquet")
                if decisions:
                    quote_predictions.update({r['intent_id']:r for r in decisions if r['admit']})
                    d = pl.from_dicts(decisions,infer_schema_length=None)
                    for r in d.group_by('reason').len().to_dicts():
                        reasons.append(dict(mode=mode,portfolio=name,day=day,**r))
                    calibration.append(dict(mode=mode,portfolio=name,day=day,quotes=len(decisions),
                        admitted=sum(d['admit']),hazard_prior_quotes=d.filter(
                            (pl.col('hazard_samples')<30) & (pl.col('remaining_sessions')>0)).height,
                        mean_p_sd=d['p_sd'].mean(),mean_p_overnight=d['p_overnight'].mean(),
                        mean_p_expiry=d['p_expiry'].mean(),mean_p_other=d['p_other'].mean()))
            for pid,p in terminal.items():
                if config['use_bpday']:
                    d = quote_predictions[pid]
                    total, held, n = snapshots[p['entry_day'],name]['cells'].get(d['cell'],[0.,0.,0])
                    selected.append(dict(mode=mode,portfolio=name,position_id=pid,stream=p['stream'],
                        entry_day=p['entry_day'],cell=d['cell'],prior_n=n,
                        source='warmup_n_below30' if n<30 else 'qualified_prior_bpday',
                        closed=p['state']=='closed',pnl_twd=p['pnl_twd'] if p['state']=='closed' else None))
                if not config['split_ev']:
                    continue
                if p['hedged_ns'] is None or not p['future_sell_qty'] or p['contract']['expiry'] >= manifest['days'][-1]:
                    continue
                d = quote_predictions[pid]
                kind = ('unresolved' if p['state']!='closed' else 'sd' if p['close_day']==p['entry_day']
                        else 'overnight' if p['close_kind']=='maker_exit' else 'expiry'
                        if p['close_kind']=='expiry_basis_zero_accounting' else 'other')
                forecasts.append(dict(mode=mode,portfolio=name,stream=p['stream'],position_id=pid,
                    actual_kind=kind,p_sd=d['p_sd'],p_overnight=d['p_overnight'],p_expiry=d['p_expiry'],
                    p_other=d['p_other'],expected_entry_ev_bp=d['est_bp'],
                    realized_bp=p['pnl_bp'] if p['state']=='closed' else None,
                    pnl_twd=p['pnl_twd'] if p['state']=='closed' else None))
            for pid,t in fills.items():
                event = invalid.get(pid)
                p = terminal[pid]
                lead = (t['ns']-event['ns'])/1e6 if event else None
                cancellation_rows.append(dict(mode=mode,portfolio=name,position_id=pid,day=t['day'],vc=t['vc'],
                    invalid_reason=event['reason'] if event else 'no_prior_invalid_event',lead_ms=lead,
                    category=('no_prior_invalid_event' if lead is None else 'avoidable_original_quote' if lead>50 else 'cancel_race'),
                    closed=p['state']=='closed',pnl_twd=p['pnl_twd'] if p['state']=='closed' else None,
                    quote_ab=p['quote_ab'],actual_ab=p['actual_ab'],close_kind=p['close_kind']))
    df = pl.from_dicts(summary,infer_schema_length=None)
    df.write_csv(args.root/'comparison.csv')
    diag = pl.from_dicts(cancellation_rows,infer_schema_length=None)
    diag.write_parquet(args.root/'s2_cancellation_attribution.parquet')
    grouped = diag.group_by('mode','portfolio','category','invalid_reason').agg(
        pl.len().alias('fills'),pl.col('lead_ms').median().alias('median_lead_ms'),
        pl.col('closed').sum().alias('closed'),pl.col('pnl_twd').sum().alias('closed_pnl_twd'),
        (pl.col('actual_ab')<=0).sum().alias('nonpositive_actual_basis'))
    grouped.sort('mode','portfolio','category','invalid_reason').write_csv(args.root/'s2_cancellation_summary.csv')
    pl.from_dicts(reasons).write_csv(args.root/'decision_reasons_daily.csv')
    pl.from_dicts(calibration).write_csv(args.root/'probabilities_daily.csv')
    pl.from_dicts(messages).write_csv(args.root/'futures_maker_message_load.csv')
    hedge_frame = pl.from_dicts(hedges,infer_schema_length=None)
    hedge_frame.write_parquet(args.root/'hedge_latencies.parquet')
    hedge_frame.group_by('mode','portfolio','purpose').agg(pl.len().alias('fills'),
        pl.col('latency_ms').median().alias('median_ms'),pl.col('latency_ms').quantile(.99).alias('p99_ms'),
        pl.col('latency_ms').max().alias('max_ms'),
        (pl.col('latency_ms')>50.000001).sum().alias('later_than_50ms'),
        (pl.col('latency_ms')>5000).sum().alias('later_than_5s')).write_csv(args.root/'hedge_latency_summary.csv')
    selection = pl.from_dicts(selected,infer_schema_length=None)
    selection.write_parquet(args.root/'bpday_fill_sources.parquet')
    selection.group_by('mode','portfolio','source').agg(
        pl.len().alias('maker_triggered_entries'),pl.col('closed').sum().alias('closed'),
        pl.col('pnl_twd').sum().alias('closed_pnl_twd'),pl.col('entry_day').min().alias('first_entry_day'),
        pl.col('entry_day').max().alias('last_entry_day')).write_csv(args.root/'bpday_fill_sources.csv')
    prediction = pl.from_dicts(forecasts,infer_schema_length=None)
    prediction.write_parquet(args.root/'matured_forecasts.parquet')
    prediction.group_by('mode','portfolio','stream').agg(
        pl.len().alias('matured_pairs'),
        pl.col('expected_entry_ev_bp').mean().alias('mean_quote_ev_bp'),
        pl.col('realized_bp').mean().alias('mean_closed_realized_bp'),
        *[pl.col('p_'+k).mean().alias('predicted_'+k) for k in ('sd','overnight','expiry','other')],
        *[(pl.col('actual_kind')==k).mean().alias('actual_'+k) for k in ('sd','overnight','expiry','other','unresolved')]
    ).write_csv(args.root/'forecast_calibration.csv')
    # Old v20 control must remain exactly identical, including inventory marks.
    old = Path(__file__).resolve().parents[3]/'data/ev_lookup_v20_capacity_full_20260908_r4'
    a = pl.read_csv(old/'ev_20M_daily.csv')
    b = pl.read_csv(args.root/'second/legacy_ev_20M_daily.csv')
    cols = [c for c in a.columns if c in b.columns]
    if not a.select(cols).equals(b.select(cols)):
        raise AssertionError('full v20 control differs; investigate before attributing performance')
    baseline = next(r for r in summary if r['mode']=='second' and r['portfolio']=='legacy_ev_20M')
    result = dict(summaries=summary,baseline_exact_daily_match=True,
                  daily_delta_from_v20=[dict(mode=r['mode'],portfolio=r['portfolio'],
                    delta=(r['equity_per_available_day']-baseline['equity_per_available_day'])
                    if r['equity_per_available_day'] is not None else None) for r in summary],
                  cancellation_attribution=grouped.to_dicts(),
                  note='All daily returns are whole-strategy totals, not increments on the old v17 27950/day. Avoidable original-quote PnL is diagnostic, not recoverable profit.',
                  report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    plot_equity(args.root)
    print(df.select('mode','portfolio','equity_per_available_day','mean_paired_twd','peak_committed_twd',
                    'paired_entries','actual_nonpositive_basis','expiry_realized_twd'))


def plot_equity(root: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import numpy as np
    from datetime import datetime
    labels = [('legacy_ev_20M','Old EV','#0072B2'),('split_ev_20M','EV with exit branches','#D55E00'),
              ('split_bpday_20M','EV with exit branches + prior bpday','#009E73')]
    fig,axes = plt.subplots(1,2,figsize=(13,4.8),sharey=True,layout='constrained')
    for ax,mode,title in zip(axes,['second','event'],['1-second order checks','Book-event cancellation and repricing']):
        for name,label,color in labels:
            d=pl.read_csv(root/mode/f'{name}_daily.csv',schema_overrides={'day':pl.String})
            dates=[datetime.strptime(x,'%Y%m%d') for x in d['day']]
            values=[x if x is not None else np.nan for x in d['official_equity_twd']]
            ax.plot(dates,values,label=label,color=color,lw=1.5)
        ax.set_title(title)
        ax.axhline(0,color='#555555',lw=.7)
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.legend(fontsize=8,loc='best')
    axes[0].set_ylabel('Cumulative PnL, including open inventory marks (TWD)')
    fig.suptitle('S1 + S2, TWD 20M reserved capacity; 50ms hedge / cancellation')
    fig.supxlabel('2026 | Before funding costs; gaps preserve unavailable inventory marks; expiry uses C8 accounting',fontsize=9)
    fig.savefig(root/'equity_comparison.png',dpi=160)
    plt.close(fig)


def maker_message_load(day: str, trace: list[dict]) -> dict:
    # These are directly recorded maker requests. Taker execution events are
    # excluded because the trace does not contain their precise wire-send time.
    times=sorted(t['ns'] for t in trace if t['kind']=='event_cancel' or
                 t['kind']=='quote' and t['stream']=='S2')
    window=deque()
    peak=regular=above5=0
    peak_second=None
    start=open_ns(day)
    for t in times:
        while window and window[0] <= t-1_000_000_000:
            window.popleft()
        window.append(t)
        sec=(t-start)/1e9
        if len(window)>peak:
            peak,peak_second=len(window),sec
        if 315<sec<13995:
            regular=max(regular,len(window))
        above5 += len(window)>5
    return dict(day=day,maker_requests=len(times),rolling_1s_peak=peak,peak_second=peak_second,
                regular_peak=regular,arrivals_above5=above5)


if __name__ == '__main__':
    main()
