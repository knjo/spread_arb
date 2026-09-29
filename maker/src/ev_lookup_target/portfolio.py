"""Chronological Q orders, persistent waiting intents and observed-release credit."""
from functools import lru_cache
import json
from pathlib import Path

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND
from .decide import QEntryDecider
from .inside_portfolio import LiquidityPortfolio
from .q_model import QSnapshot
from .release_model import ReleaseSnapshot


@lru_cache(maxsize=4)
def load_models(root, day):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed' or day not in manifest['days']:
        raise ValueError('model-input build incomplete or day absent')
    folder=root/f'Date={day}'
    q=QSnapshot(**json.loads((folder/'q_snapshot.json').read_text()))
    release=ReleaseSnapshot(**json.loads((folder/'release_snapshot.json').read_text()))
    if q.day!=day or release.day!=day or any(d>=day for d in q.train_days+release.train_days):
        raise AssertionError('future training session in model snapshot')
    return q,release


class TargetPortfolio(LiquidityPortfolio):
    def __init__(self,*args,q_policy=None,model_inputs=None,release_credit=False,base_cap_twd=20_000_000,
                 max_extra_twd=5_000_000,persistent_wait=True,**kwargs):
        super().__init__(*args,**kwargs)
        if self.park_spot:
            raise ValueError('this study uses shared unreserved quotes, not the old unlimited deep-parking policy')
        if (q_policy is not None or release_credit) and model_inputs is None:
            raise ValueError('Q and release policies require frozen inputs')
        self.q_policy,self.model_inputs=q_policy,str(model_inputs) if model_inputs is not None else None
        self.release_credit,self.base_cap_twd,self.max_extra_twd=release_credit,base_cap_twd,max_extra_twd
        self.persistent_wait=persistent_wait
        self.waiting_s2={}
        self._capacity_cached=None
        self._retrying=False

    def begin(self,market,session,snapshot,schedule):
        self.waiting_s2={}
        self._capacity_cached=None
        self._retrying=False
        super().begin(market,session,snapshot,schedule)
        if self.q_policy is not None or self.release_credit:
            q,self.release_snapshot=load_models(self.model_inputs,self.day)
            if self.q_policy is not None:
                self.decider=QEntryDecider(self.decider,q,policy=self.q_policy,calendar=market.calendar)

    def target_admission_limit(self,ns):
        if not self.release_credit:
            return self.ledger.cap_cents
        second=max(0,min(CLOSE_SECOND-1,int((ns-self.market.start)//SECOND)))
        paired=[self.positions[pid] for pid in sorted(self.active)
                if self.positions[pid].state=='paired' and not self.positions[pid].continuity_blocked]
        signature=(second//300,self.ledger.committed_cents,tuple(p.id for p in paired))
        if self._capacity_cached and self._capacity_cached[0]==signature:
            return self._capacity_cached[1]
        inputs=[dict(id=p.id,stream=p.stream,entry_day=p.entry_day,expiry=p.contract.expiry,
                     premium=p.quote_ab-p.anchor,nominal_twd=self.ledger.amounts.get(p.id,0)/100) for p in paired]
        forecast=self.release_snapshot.capacity(inputs,second=second,calendar=self.market.calendar,
                    base_twd=self.base_cap_twd,max_extra_twd=self.max_extra_twd)
        limit=min(self.ledger.cap_cents,round(forecast['admission_twd']*100))
        self._capacity_cached=(signature,limit)
        self.trace.append(dict(day=self.day,ns=ns,position_id=None,stream=None,kind='capacity_forecast',
            committed_cents=self.ledger.committed_cents,admission_cap_cents=limit,
            forecast_today_twd=forecast['forecast_today_twd'],forecast_next_session_twd=forecast['forecast_next_session_twd'],
            next_session=forecast['next_session'],positions_json=json.dumps(forecast['positions'])))
        return limit

    def offer_s1(self,row,ns,*,retry=False):
        if self.persistent_wait and self.use_s1 and not retry:
            self.desired_s1[row['raw_order_fact_id']]=row
        super().offer_s1(row,ns,retry=retry)

    def offer_s2(self,vc,intent_id,sec):
        if self.persistent_wait:
            self.waiting_s2.setdefault(vc,self.market.start+sec*SECOND)
        super().offer_s2(vc,intent_id,sec)

    def _requote_s2(self,vc,ns,intent_id=None):
        if self.persistent_wait:
            self.waiting_s2.setdefault(vc,ns)
        super()._requote_s2(vc,ns,intent_id)

    def retry_waiting(self,ns,*,include_s1=True):
        if not self.persistent_wait or self._retrying or ns>=self.market.start+14_000*SECOND:
            return
        self._retrying=True
        try:
            waiting=[(seen,'S2',vc) for vc,seen in self.waiting_s2.items() if vc not in self.s2_live]
            if include_s1:
                waiting.extend((row['nominal_new_time_ns'],'S1',source) for source,row in self.desired_s1.items()
                               if self.s1_attempts.get(source,source) not in self.queue.orders)
            # Priority uses when the intent became observable, never today's
            # future PnL or a retrospectively selected best trade.
            for _,stream,key in sorted(waiting):
                if stream=='S2':
                    self._requote_s2(key,ns,f'S2/{self.day}/{key}/w{ns}')
                else:
                    row=self.desired_s1.get(key)
                    if row and ns<min(row['nominal_stop_time_ns'],self.market.start+14_000*SECOND):
                        self.offer_s1(row,ns,retry=True)
        finally:
            self._retrying=False

    def second(self,sec,**kwargs):
        super().second(sec,**kwargs)
        # Base processing already retries persistent S1 intents each second.
        self.retry_waiting(self.market.start+sec*SECOND,include_s1=False)

    def hedge(self,pid,ns):
        before=self.ledger.committed_cents
        super().hedge(pid,ns)
        self._enforce_capacity(ns)
        if self.ledger.committed_cents<before:
            self.retry_waiting(ns)

    def cancel(self,oid,ns):
        before=self.ledger.committed_cents
        super().cancel(oid,ns)
        if self.ledger.committed_cents<before:
            self.retry_waiting(ns)

    def finish(self):
        row=super().finish()
        row.update(capacity_cancels=self.capacity_cancels,ticket_rejects=self.ticket_rejects,
                   waiting_s2_products=len(self.waiting_s2),base_cap_twd=self.base_cap_twd)
        return row
