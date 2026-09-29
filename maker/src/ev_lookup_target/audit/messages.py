"""Lower bound on outbound load, retaining explicitly logged cancel-race requests.

A source-intent cancellation that loses a race can lack both request and
effective-cancel records. Consequently this is a lower bound, not proof that
an actual outbound throttle would accept every message.
"""
from collections import Counter

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns


def counts(day,trace,cancel_ms=50):
    start=open_ns(day)
    outbound=[]
    requests=set()
    for t in trace:
        kind=t['kind']
        if kind in {'event_cancel','capacity_cancel','exit_guard_cancel'}:
            purpose='exit' if kind=='exit_guard_cancel' else 'entry'
            requests.add((t['position_id'],purpose,t['ns'],t['stream']))
        elif kind=='cancel':
            # Some source-intent/time-limit cancellations have no separate
            # request trace. Their declared delay identifies the send time.
            requests.add((t['position_id'],t['purpose'],t['ns']-cancel_ms*1_000_000,t['stream']))
        elif kind in {'quote','exit_quote'}:
            venue='F' if kind=='quote' and t['stream']=='S2' else 'S'
            outbound.append((t['ns'],venue))
        elif kind=='taker_fill':
            venue='F' if t['purpose'] in {'entry_future','exit_future'} else 'S'
            outbound.append((t['ns'],venue))
    for _,purpose,ns,stream in requests:
        outbound.append((ns,'F' if purpose=='entry' and stream=='S2' else 'S'))
    buckets=Counter()
    for ns,venue in outbound:
        assert start<=ns<=start+15600*SECOND
        buckets[venue,(ns-start)//SECOND]+=1
    result={}
    for venue,reference in [('F',5),('S',100)]:
        values=[n for (v,_),n in buckets.items() if v==venue]
        result.update({f'{venue}_outbound_messages':sum(values),f'{venue}_max_messages_per_second':max(values,default=0),
            f'{venue}_seconds_above_reference':sum(n>reference for n in values),
            f'{venue}_messages_in_over_reference_seconds':sum(n for n in values if n>reference)})
    return result
