"""Measure actual stock flow and completed-pair turnover on allocated capital."""
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND


def daily(root,name,days,positions):
    rows=[]
    for day in days:
        path=Path(root)/f'Date={day}'/name/'execution.parquet'
        trace=(pl.scan_parquet(path).filter(pl.col('kind').is_in(['maker_fill','taker_fill']))
               .collect().to_dicts()) if path.exists() else []
        buy=sell=0.
        for t in trace:
            cash=t['quantity']*t['price']
            if ((t['kind']=='maker_fill' and t['stream']=='S1' and t['purpose']=='entry') or
                (t['kind']=='taker_fill' and t['purpose']=='entry_spot')):
                buy+=cash
            elif ((t['kind']=='maker_fill' and t['purpose']=='exit') or
                  (t['kind']=='taker_fill' and t['purpose'] in {'entry_rollback','exit_spot_remainder'})):
                sell+=cash
        closed=positions.filter(pl.col('close_day')==day)
        normal=closed.filter(pl.col('close_kind')=='maker_exit')
        expiry=closed.filter(pl.col('close_kind')=='expiry_basis_zero_accounting')
        entry=positions.filter(pl.col('entry_day')==day)
        rows.append(dict(day=day,portfolio=name,market_stock_buy_cash_twd=buy,market_stock_sell_cash_twd=sell,
            entries=entry.height,s2_entries=entry.filter(pl.col('stream')=='S2').height,
            completed_positions=closed.height,normal_closed_positions=normal.height,
            completed_original_stock_cash_twd=closed['spot_buy_cash'].sum()/10_000,
            normal_closed_original_stock_cash_twd=normal['spot_buy_cash'].sum()/10_000,
            expiry_accounting_released_original_stock_cash_twd=expiry['spot_buy_cash'].sum()/10_000))
    return pl.from_dicts(rows)


def summarize(frame,positions,available_days,first_day=None):
    if first_day is not None:
        frame=frame.filter(pl.col('day')>=first_day)
    days=frame['day'].to_list();n=sum(d in available_days for d in days)
    closed=positions.filter(pl.col('close_day').is_in(days))
    held=((pl.col('close_ns')-pl.col('entry_fill_ns'))/SECOND/86400)
    duration=closed.select(held.mean().alias('mean'),
        ((held*pl.col('spot_buy_cash')).sum()/pl.col('spot_buy_cash').sum()).alias('weighted')).row(0,named=True)
    return dict(observed_sessions=n,entries_per_day=frame['entries'].sum()/n,
        s2_entries_per_day=frame['s2_entries'].sum()/n,completed_positions_per_day=frame['completed_positions'].sum()/n,
        normal_closed_positions_per_day=frame['normal_closed_positions'].sum()/n,
        stock_buy_cash_per_day_twd=frame['market_stock_buy_cash_twd'].sum()/n,
        stock_sell_cash_per_day_twd=frame['market_stock_sell_cash_twd'].sum()/n,
        stock_buy_turns_per_20M_per_day=frame['market_stock_buy_cash_twd'].sum()/n/20_000_000,
        completed_cash_turns_per_20M_per_day=frame['completed_original_stock_cash_twd'].sum()/n/20_000_000,
        normal_close_cash_turns_per_20M_per_day=frame['normal_closed_original_stock_cash_twd'].sum()/n/20_000_000,
        expiry_accounting_cash_per_day_twd=frame['expiry_accounting_released_original_stock_cash_twd'].sum()/n,
        resolved_mean_calendar_days=duration['mean'],resolved_cash_weighted_calendar_days=duration['weighted'])


def write(root,output,manifest):
    results=[]
    for config in manifest['configurations']:
        name=config['name'];folder=output/name
        positions=pl.read_parquet(folder/'position_returns.parquet')
        frame=daily(root,name,manifest['days'],positions)
        frame.write_csv(folder/'turnover_daily.csv')
        summary=json.loads((folder/'summary.json').read_text())
        whole=summarize(frame,positions,manifest['available_days'])
        post=summarize(frame,positions,manifest['available_days'],summary['postwarm_first_day'])
        row=dict(portfolio=name,**whole,**{'postwarm_'+k:v for k,v in post.items()})
        row.update(net_per_day_at_0pct_funding=summary['equity_before_funding_twd']/summary['observed_sessions'],
            net_per_day_at_2pct_funding=summary['net_per_observed_day_twd'],
            net_per_day_at_4pct_funding=(summary['equity_before_funding_twd']-2*summary['actual_cash_funding_2pct_twd'])/summary['observed_sessions'],
            net_per_day_at_6pct_funding=(summary['equity_before_funding_twd']-3*summary['actual_cash_funding_2pct_twd'])/summary['observed_sessions'])
        results.append(row)
    pl.from_dicts(results,infer_schema_length=None).write_csv(output/'turnover.csv')
    return results
