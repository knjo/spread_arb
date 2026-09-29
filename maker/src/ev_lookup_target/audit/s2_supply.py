"""Chronological entry-only S2 supply experiments, independent of future outcomes.

These are alternative unlimited-capacity market supply paths, not portfolio
returns. No exits or future exit labels are used. Every futures maker fill
requires its actual delayed stock hedge, including adverse and race fills.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import heapq
import json
from math import isfinite
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl

from ...ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, LookupHistory
from ...ev_lookup_cost.market import MarketDay, previous_tick
from ..inside_portfolio import LiquidityPortfolio


CONFIGS = [
    dict(name='second60_depth5_r25',event_entry=False,cooldown_seconds=60,depth_multiple=5,residual_bp=25),
    dict(name='event60_depth5_r25',event_entry=True,cooldown_seconds=60,depth_multiple=5,residual_bp=25),
    dict(name='event0_depth5_r25',event_entry=True,cooldown_seconds=0,depth_multiple=5,residual_bp=25),
    dict(name='event0_depth1_r25',event_entry=True,cooldown_seconds=0,depth_multiple=1,residual_bp=25),
    dict(name='event0_depth5_r0',event_entry=True,cooldown_seconds=0,depth_multiple=5,residual_bp=0),
    dict(name='event0_depth1_r0',event_entry=True,cooldown_seconds=0,depth_multiple=1,residual_bp=0),
]


class SupplyPortfolio(LiquidityPortfolio):
    def __init__(self,*,name,event_entry,cooldown_seconds,depth_multiple,residual_bp):
        super().__init__(name,10**12,use_ev=False,use_bpday=False,use_s1=False,event_quotes=True,
            hedge_ms=50,cancel_ms=50,reserve_quotes=True,exit_event_guard=False,repeg_drop_bp=10.,
            liquidity_rule=dict(enabled=True,min_a1_multiple=depth_multiple,adverse_probability=.5))
        self.event_entry=event_entry
        self.cooldown_seconds=cooldown_seconds
        self.residual_bp=residual_bp
        self.pending_hedges={}
        self.supply_counts=Counter()
        self._attempt=0

    def _requote_s2(self,vc,ns,intent_id=None):
        if vc in self.s2_live or vc in self.pending_hedges or ns<self.cooldown.get(vc,0):
            return
        sec=int((ns-self.market.start)//SECOND)
        if not 300<=sec<14_000:
            return
        c=self.market.contracts.get(vc)
        if c is None:
            return
        self.supply_counts['idle_evaluations']+=1
        pair=self.market.pair(c,ns)
        if pair is None:
            return
        self.supply_counts['valid_pair']+=1
        spot,future=pair
        anchor=float(self.market.signals[vc]['an'][sec])
        desired=previous_tick(future.asks[0][0])
        if desired<=future.bids[0][0]:
            return
        self.supply_counts['inside_room']+=1
        basis=(desired/spot.asks[0][0]-1)*10_000
        if not isfinite(anchor) or basis<=0 or basis-anchor<self.residual_bp:
            return
        self.supply_counts['basis_gate']+=1
        self._attempt+=1
        self.submit(intent_id=f'S2/{self.day}/{vc}/{self._attempt}',c=c,stream='S2',price=desired,
                    ns=ns,anchor=anchor,ab=basis,eff_u=basis-anchor)

    def book_update(self,instrument,ns):
        vc=instrument[2:] if instrument.startswith('S:') else self.market.future_to_vc.get(instrument[2:])
        if vc is None:
            return
        pid=self.s2_live.get(vc)
        order=self.queue.orders.get(pid)
        if order is not None and order.cancel_ns is None:
            p=self.positions[pid]
            sec=int((ns-self.market.start)//SECOND)
            pair=self.market.pair(p.contract,ns)
            anchor=float(self.market.signals[vc]['an'][min(sec,CLOSE_SECOND)])
            basis=(order.price/pair[0].asks[0][0]-1)*10_000 if pair else None
            reason='market'
            if pair is not None and isfinite(anchor):
                future=pair[1]
                if not future.bids[0][0]<order.price<=future.asks[0][0]:
                    reason='first_priority'
                elif basis<=0 or basis-anchor<self.residual_bp-5:
                    reason='basis'
                elif basis<p.quote_ab-10:
                    reason='drift'
                else:
                    reason=self._risk(p.contract,order.price,ns)['reason']
            if sec>=14_000:
                reason='cutoff'
            if reason!='ok':
                if p.s2_invalid_ns is None:
                    p.s2_invalid_ns=ns
                    self._event(p,ns,'s2_first_invalid',reason=reason,held_basis=basis)
                self._event(p,ns,'event_cancel',reason=reason,held_basis=basis,
                            effective_ns=ns+self.cancel_delay)
                self.request_cancel(pid,ns)
        if self.event_entry:
            self._requote_s2(vc,ns)

    def second(self,sec,**kwargs):
        ns=self.market.start+sec*SECOND
        for vc in list(self.s2_live):
            self.book_update('S:'+vc,ns)
        # All available symbols are considered independently of a shadow fill.
        # Keep the one-second scan in both paths; event_entry adds book updates.
        if 300<=sec<14_000:
            for vc in sorted(self.market.contracts):
                self._requote_s2(vc,ns)

    def cancel(self,oid,ns):
        super().cancel(oid,ns)
        # Independent local replacement after a real effective cancellation.
        p=self.positions.get(oid)
        if p is not None and p.state=='cancelled':
            self._requote_s2(p.contract.vc,ns)

    def trade(self,instrument,ns,sequence,price,qty):
        vc=self.market.future_to_vc.get(instrument[2:]) if instrument.startswith('F:') else None
        pid=self.s2_live.get(vc)
        super().trade(instrument,ns,sequence,price,qty)
        p=self.positions.get(pid)
        if p is not None and p.entry_fill_ns==ns:
            self.cooldown[vc]=ns+self.cooldown_seconds*SECOND
            self.pending_hedges[vc]=pid

    def hedge(self,pid,ns):
        super().hedge(pid,ns)
        p=self.positions[pid]
        if p.state=='paired':
            self.pending_hedges.pop(p.contract.vc,None)
        # No future quote is backdated to the fill or hedge event.


def run_day(day,output,*,products=None):
    output=Path(output)
    folder=output/f'Date={day}'
    folder.mkdir(parents=True,exist_ok=False)
    start=monotonic()
    market=MarketDay(day,output,products=products,depth_events=True)
    actors=[SupplyPortfolio(**config) for config in CONFIGS]
    heap=[]
    serial=0
    def schedule(ns,kind,actor,payload):
        nonlocal serial
        serial+=1
        assert kind in {'hedge','cancel'}
        heapq.heappush(heap,(ns,1 if kind=='hedge' else 2,serial,kind,actor,payload))
    for actor in actors:
        actor.begin(market,0,LookupHistory().freeze(day,10**12,{}),schedule)
    trades=iter(market.trades.filter(pl.col('instrument').str.starts_with('F:')).iter_rows())
    books=iter(market.book_events.iter_rows())
    trade=next(trades,None); book=next(books,None); sec=0
    while True:
        tn=trade[0] if trade else market.end+1
        bn=book[0] if book else market.end+1
        hn=heap[0][0] if heap else market.end+1
        sn=market.start+sec*SECOND if sec<CLOSE_SECOND else market.end+1
        ns=min(tn,bn,hn,sn)
        if ns>market.end:
            break
        if tn<=min(bn,hn,sn):
            t,sequence,instrument,price,qty=trade
            for actor in actors:
                actor.trade(instrument,t,sequence,price,qty)
            trade=next(trades,None)
        elif bn<=min(hn,sn):
            for actor in actors:
                actor.book_update(book[1],bn)
            book=next(books,None)
        elif hn<=sn:
            t,_,_,kind,actor,payload=heapq.heappop(heap)
            getattr(actor,kind)(payload,t)
        else:
            for actor in actors:
                actor.second(sec)
            sec+=1
    rows=[]
    for actor in actors:
        dest=folder/actor.name;dest.mkdir()
        fills=[]
        for p in actor.positions.values():
            if p.entry_fill_ns is None:
                continue
            fills.append(dict(day=day,policy=actor.name,id=p.id,vc=p.contract.vc,qc=p.contract.qc,
                quote_ns=p.quote_ns,fill_ns=p.entry_fill_ns,hedged_ns=p.hedged_ns,quote_ab=p.quote_ab,
                anchor=p.anchor,actual_ab=p.actual_ab if p.hedged_ns is not None else None,
                actual_premium_bp=p.actual_ab-p.anchor if p.hedged_ns is not None else None,
                quote_premium_bp=p.quote_ab-p.anchor,stock_cash_twd=p.spot_buy_cash/10_000,
                stock_shares=p.spot_buy_qty,future_contracts=p.future_sell_qty,
                cancellation_race=p.s2_invalid_ns is not None,state=p.state))
        if fills:
            pl.from_dicts(fills,infer_schema_length=None).write_parquet(dest/'fills.parquet')
        for filename,records in [('execution',actor.trace),('decisions',actor.decisions),('ledger',actor.ledger.events)]:
            if records:
                pl.from_dicts(records,infer_schema_length=None).write_parquet(dest/f'{filename}.parquet')
        paired=[r for r in fills if r['hedged_ns'] is not None]
        row=dict(day=day,policy=actor.name,quotes=sum(t['kind']=='quote' for t in actor.trace),
            fills=len(fills),paired=len(paired),pending_hedges=len(fills)-len(paired),
            race_fills=sum(r['cancellation_race'] for r in fills),
            actual_premium_mean_bp=float(np.mean([r['actual_premium_bp'] for r in paired])) if paired else None,
            actual_premium_ge25=sum(r['actual_premium_bp']>=25 for r in paired),
            negative_actual_basis=sum(r['actual_ab']<0 for r in paired),
            stock_cash_twd=sum(r['stock_cash_twd'] for r in fills),**actor.supply_counts)
        rows.append(row)
    pl.from_dicts(rows,infer_schema_length=None).write_csv(folder/'summary.csv')
    (folder/'inputs.json').write_text(json.dumps(market.inputs,indent=2)+'\n')
    print(json.dumps(dict(day=day,seconds=round(monotonic()-start,2),results=rows)),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--days',nargs='+',required=True)
    parser.add_argument('--products',nargs='+')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps(dict(days=args.days,products=args.products,
        configs=CONFIGS,scope='Entry-only uncapped supply; each day starts flat, no exit/return claims. '
        'All fills retained; mandatory stock hedges use50ms schedule and actual shared depth. '
        'Zero cooldown still waits for completion of the previous stock hedge on that symbol. '
        'New quote queue admission is immediate; same-timestamp prints execute before new quotes. '
        'A0/25bp residual threshold and depth sensitivity are exploratory, not calibrated Q policies. '
        'No future outcomes rank symbols or filter fills.'),indent=2)+'\n')
    for day in args.days:
        run_day(day,args.output,products=args.products)


if __name__=='__main__':
    main()
