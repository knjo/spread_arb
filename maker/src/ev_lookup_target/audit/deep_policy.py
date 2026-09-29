"""Keep S1 below B1 in the actual queue; cancel on approach if capacity is absent.

This does not make fills conditional on future free capacity. A direct jump or
a fill during cancellation is irrevocable and stays in the common ledger.
"""
from collections import defaultdict

from ..capacity_policy import admission_limit
from ..portfolio import TargetPortfolio


class DeepQuoteDecision:
    """Bypass only the capacity gate for a currently deep S1 maker request."""
    def __init__(self,delegate):
        self.delegate=delegate

    def __getattr__(self,key):
        return getattr(self.delegate,key)

    def decide(self,**kwargs):
        assert kwargs['stream']=='S1'
        return self.delegate.decide(**dict(kwargs,capacity_required=False))


class DeepSpotPortfolio(TargetPortfolio):
    def __init__(self,*args,deep_shared=False,**kwargs):
        self.deep_shared=deep_shared
        self.stock_entry_orders=defaultdict(set)
        super().__init__(*args,**kwargs)
        if deep_shared and self.reserve_quotes:
            raise ValueError('deep shared policy requires unreserved quotes')

    def begin(self,*args,**kwargs):
        self.stock_entry_orders.clear()
        return super().begin(*args,**kwargs)

    def submit(self,**kwargs):
        if not self.deep_shared or kwargs['stream']!='S1':
            return super().submit(**kwargs)
        instrument='S:'+kwargs['c'].vc
        book=self.market.book(instrument,kwargs['ns'])
        deep=bool(book and book.valid() and kwargs['price']<book.bids[0][0]
                  and self.ledger.committed_cents<=self.ledger.cap_cents)
        decider,parked=self.decider,self.park_spot
        before=len(self.decisions)
        try:
            if deep:
                self.decider=DeepQuoteDecision(decider)
                self.park_spot=True
            super().submit(**kwargs)
        finally:
            self.decider,self.park_spot=decider,parked
            # The legacy parked-order activator reserves on approach. This
            # policy keeps the quote unreserved and uses its own depth guard.
            self.parked.discard(kwargs['intent_id'])
            self.parked_instruments.get(instrument,set()).discard(kwargs['intent_id'])
        if len(self.decisions)>before:
            self.decisions[-1].update(deep_shared=deep,deep_stock_bid=book.bids[0][0]/10_000,
                                     deep_stock_book_ns=book.ns)
        if kwargs['intent_id'] in self.queue.orders:
            self.stock_entry_orders[instrument].add(kwargs['intent_id'])

    def _enforce_candidates(self,ns,ids):
        remaining=None
        books={}
        emergency=self.ledger.committed_cents>self.ledger.cap_cents
        for oid in sorted(ids,key=lambda key:(self.queue.orders[key].placed_ns,key)
                          if key in self.queue.orders else (0,key)):
            order=self.queue.orders.get(oid)
            if order is None or order.cancel_ns is not None or oid not in self.unreserved:
                continue
            p=self.positions[order.position_id]
            if p.stream=='S1':
                if order.instrument not in books:
                    books[order.instrument]=self.market.book(order.instrument,ns)
                book=books[order.instrument]
                if not emergency and book and book.valid() and order.price<book.bids[0][0]:
                    continue
            else:
                book=None
            if remaining is None:
                remaining=admission_limit(self,ns)-self.ledger.committed_cents
            if self.order_reservation(p,order.price)>remaining:
                self.capacity_cancels+=1
                self._event(p,ns,'capacity_cancel',remaining_cents=remaining,
                    deep_approach=p.stream=='S1' and not emergency,capacity_emergency=emergency,
                    capacity_stock_bid=book.bids[0][0]/10_000 if book and book.valid() else None,
                    capacity_stock_book_ns=book.ns if book else None)
                self.request_cancel(oid,ns)

    def _enforce_capacity(self,ns):
        if not self.deep_shared:
            return super()._enforce_capacity(ns)
        self._enforce_candidates(ns,list(self.unreserved))

    def book_update(self,instrument,ns):
        super().book_update(instrument,ns)
        if self.deep_shared and instrument in self.stock_entry_orders:
            self._enforce_candidates(ns,list(self.stock_entry_orders[instrument]))

    def cancel(self,oid,ns):
        super().cancel(oid,ns)
        p=self.positions.get(oid)
        if p and p.stream=='S1' and oid not in self.queue.orders:
            self.stock_entry_orders['S:'+p.contract.vc].discard(oid)

    def trade(self,instrument,ns,sequence,price,quantity):
        super().trade(instrument,ns,sequence,price,quantity)
        if self.deep_shared and instrument in self.stock_entry_orders:
            self.stock_entry_orders[instrument].intersection_update(self.queue.orders)
