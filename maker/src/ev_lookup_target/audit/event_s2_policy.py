"""Independent event-time S2 entry; completed stock hedge replaces the old CD.

Q, actual shared capacity, S1 deep queues and all exit/hedge rules stay in the
existing execution core. The disabled adapter preserves the old shadow path.
"""
from .deep_policy import DeepSpotPortfolio


class EventS2Portfolio(DeepSpotPortfolio):
    def __init__(self,*args,event_s2=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.event_s2=event_s2
        self.entry_hedges={}
        self.post_hedge_rechecks=0
        self.event_entry_rechecks=0

    def begin(self,*args,**kwargs):
        self.entry_hedges={}
        self.post_hedge_rechecks=self.event_entry_rechecks=0
        super().begin(*args,**kwargs)
        if self.event_s2:
            # The universe is known at the open. A pending symbol is not an
            # admitted order; observable basis/depth/Q/capacity still gate it.
            self.waiting_s2.update({vc:self.market.start for vc in self.market.contracts})
            self.entry_hedges={p.contract.vc:p.id for p in self.positions.values()
                if p.id in self.active and p.stream=='S2' and p.entry_fill_ns is not None
                and p.spot_buy_qty<p.future_sell_qty*p.contract.shares}

    def _requote_s2(self,vc,ns,intent_id=None):
        if not self.event_s2:
            return super()._requote_s2(vc,ns,intent_id)
        if not self.market.start+300_000_000_000<=ns<self.market.start+14_000_000_000_000:
            return
        if vc in self.entry_hedges or vc in self.s2_live:
            return
        self.cooldown.pop(vc,None)
        # Rejections at the same timestamp may see a different ledger after
        # a real release. Keep their decision IDs distinct and reviewable.
        self._counter+=1
        intent=f'S2/{self.day}/{vc}/event{ns}_{self._counter}'
        super()._requote_s2(vc,ns,intent)

    def offer_s2(self,vc,intent_id,sec):
        if not self.event_s2:
            return super().offer_s2(vc,intent_id,sec)
        self._requote_s2(vc,self.market.start+sec*1_000_000_000)

    def book_update(self,instrument,ns):
        super().book_update(instrument,ns)
        if self.event_s2:
            vc=instrument[2:] if instrument.startswith('S:') else self.market.future_to_vc.get(instrument[2:])
            if vc in self.market.contracts:
                self.event_entry_rechecks+=1
                self._requote_s2(vc,ns)

    def trade(self,instrument,ns,sequence,price,quantity):
        vc=self.market.future_to_vc.get(instrument[2:]) if instrument.startswith('F:') else None
        pid=self.s2_live.get(vc) if self.event_s2 else None
        super().trade(instrument,ns,sequence,price,quantity)
        p=self.positions.get(pid)
        if p is not None and p.entry_fill_ns==ns:
            self.cooldown.pop(vc,None)
            self.entry_hedges[vc]=pid

    def hedge(self,pid,ns):
        super().hedge(pid,ns)
        if not self.event_s2:
            return
        p=self.positions[pid]
        vc=p.contract.vc
        if self.entry_hedges.get(vc)==pid and p.state=='paired':
            del self.entry_hedges[vc]
            self.cooldown.pop(vc,None)
            self.post_hedge_rechecks+=1
            self._event(p,ns,'s2_hedge_complete_recheck')
            # Same-ns raw prints were processed before this callback. A new
            # queue here cannot reuse the print that filled the previous lot.
            self._requote_s2(vc,ns)

    def cancel(self,oid,ns):
        super().cancel(oid,ns)
        if self.event_s2:
            p=self.positions.get(oid)
            if p is not None and p.stream=='S2' and p.state=='cancelled':
                self._requote_s2(p.contract.vc,ns)

    def finish(self):
        result=super().finish()
        if self.event_s2:
            result.update(s2_post_hedge_rechecks=self.post_hedge_rechecks,
                          s2_event_entry_rechecks=self.event_entry_rechecks)
        return result
