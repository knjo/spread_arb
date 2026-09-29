"""Actual stock cash-time funding, including partial fills and overnight inventory."""
from collections import defaultdict

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .messages import counts as message_counts


DAY_NS=86_400*SECOND


class StockCash:
    def __init__(self):
        self.quantity=defaultdict(int)
        self.cost=defaultdict(float)
        self.last={}
        self.areas=defaultdict(float)
        self.buys=defaultdict(float)
        self.sells=defaultdict(int)

    def advance(self,pid,ns):
        if pid in self.last:
            assert ns>=self.last[pid]
            self.areas[pid]+=self.cost[pid]*(ns-self.last[pid])/DAY_NS
        self.last[pid]=ns

    def buy(self,pid,ns,quantity,cash):
        self.advance(pid,ns)
        assert quantity>0 and cash>0
        self.quantity[pid]+=quantity
        self.cost[pid]+=cash
        self.buys[pid]+=cash
        return cash

    def sell(self,pid,ns,quantity):
        self.advance(pid,ns)
        assert 0<quantity<=self.quantity[pid]
        cost=self.cost[pid]*quantity/self.quantity[pid]
        self.cost[pid]-=cost
        self.quantity[pid]-=quantity
        self.sells[pid]+=quantity
        return -cost

    def finish(self,ns):
        for pid in self.last:
            self.advance(pid,ns)
        return dict(self.areas)


def intervals(events,start,end,opening,thresholds=(20_000_000.,25_000_000.)):
    value,last,area,peak=opening,start,0.,opening
    over=[0. for _ in thresholds]
    for ns,delta in events+[(end,0.)]:
        assert start<=last<=ns<=end
        area+=value*(ns-last)/SECOND
        for i,limit in enumerate(thresholds):
            if value>limit+1e-6:
                over[i]+=(ns-last)/SECOND
        value+=delta
        assert value>=-1e-6
        peak=max(peak,value)
        last=ns
    return dict(ending_twd=value,area_twd_seconds=area,peak_twd=peak,
                over20_seconds=over[0],over25_seconds=over[1])


def analyze(root,actor,days):
    cash=StockCash()
    daily=[]
    opening_cash=opening_committed=0.
    previous_end=open_ns(days[0])
    for day in days:
        folder=root/f'Date={day}'/actor
        assert folder.is_dir(), 'portfolio session folder is missing'
        path=folder/'execution.parquet'
        trace=pl.read_parquet(path).to_dicts() if path.exists() else []
        changes=[]
        for t in trace:
            pid,ns,kind=t['position_id'],t['ns'],t['kind']
            delta=None
            if kind=='maker_fill' and t['stream']=='S1' and t['purpose']=='entry':
                delta=cash.buy(pid,ns,t['quantity'],t['price']*t['quantity'])
            elif kind=='taker_fill' and t['purpose']=='entry_spot':
                delta=cash.buy(pid,ns,t['quantity'],t['price']*t['quantity'])
            elif kind=='maker_fill' and t['purpose']=='exit':
                delta=cash.sell(pid,ns,t['quantity'])
            elif kind=='taker_fill' and t['purpose'] in {'entry_rollback','exit_spot_remainder'}:
                delta=cash.sell(pid,ns,t['quantity'])
            elif kind=='expiry_basis_zero_accounting':
                delta=cash.sell(pid,ns,cash.quantity[pid])
            if delta is not None:
                changes.append((ns,delta))
        path=folder/'ledger.parquet'
        ledger=pl.read_parquet(path).to_dicts() if path.exists() else []
        capacity_changes=[(t['ns'],t['delta_cents']/100) for t in ledger]
        start,end=open_ns(day),open_ns(day)+15600*SECOND
        stock_day=intervals(changes,start,end,opening_cash)
        stock_calendar=intervals(changes,previous_end,end,opening_cash)
        capacity_day=intervals(capacity_changes,start,end,opening_committed)
        capacity_calendar=intervals(capacity_changes,previous_end,end,opening_committed)
        row=dict(day=day,portfolio=actor,stock_cash_peak_twd=stock_day['peak_twd'],
            mean_stock_cash_twd=stock_day['area_twd_seconds']/15600,
            mean_committed_twd=capacity_day['area_twd_seconds']/15600,
            committed_peak_twd=capacity_day['peak_twd'],
            capital_days_twd=stock_calendar['area_twd_seconds']/86400,
            funding_2pct_twd=stock_calendar['area_twd_seconds']/86400*.02/365,
            committed_over20_intraday_seconds=capacity_day['over20_seconds'],
            committed_over25_intraday_seconds=capacity_day['over25_seconds'],
            committed_over20_calendar_seconds=capacity_calendar['over20_seconds'],
            committed_over25_calendar_seconds=capacity_calendar['over25_seconds'])
        row.update(message_counts(day,trace))
        daily.append(row)
        opening_cash,opening_committed=stock_day['ending_twd'],capacity_day['ending_twd']
        previous_end=end
        position_path=folder/'positions.parquet'
        if not position_path.exists():
            assert abs(opening_cash)<1e-6 and abs(opening_committed)<1e-6
            assert not any(t['kind'] in {'maker_fill','taker_fill'} for t in trace), 'filled positions file missing'
        positions=pl.read_parquet(position_path).to_dicts() if position_path.exists() else []
        for p in positions:
            pid=p['id']
            if p['entry_fill_ns'] is None:
                continue
            assert abs(cash.buys[pid]-p['spot_buy_cash']/10_000)<1e-5
            assert cash.sells[pid]==p['spot_sell_qty']
            assert cash.quantity[pid]==p['spot_buy_qty']-p['spot_sell_qty']
    areas=cash.finish(previous_end)
    assert abs(sum(areas.values())-sum(r['capital_days_twd'] for r in daily))<1e-4
    return daily,areas
