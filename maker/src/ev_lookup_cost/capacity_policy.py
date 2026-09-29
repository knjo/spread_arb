"""Observable capacity allocation and parked-stock-order activation."""
from __future__ import annotations

from math import sqrt

from .causal_lookup import SECOND
from .execution import nominal_cents


def release_probability(snapshot, stream: str, carry: bool, second: int) -> float:
    """95% Wilson lower bound; include unresolved right-censored daily risk sets."""
    rows = [r for r in snapshot.capacity_observations if r.stream == stream and r.carry == carry
            and r.start_second <= second < r.end_second]
    n = len(rows)
    if n < 30:
        return 0.0
    probability = sum(r.closed for r in rows) / n
    z = 1.96
    return max(0., (probability + z*z/(2*n)
                    - z*sqrt(probability*(1-probability)/n + z*z/(4*n*n))) / (1+z*z/n))


def admission_limit(actor, ns: int) -> int:
    if actor.overnight_target_cents is None:
        return actor.ledger.cap_cents
    sec = max(0, int((ns - actor.market.start) // SECOND))
    bucket = sec // 300 * 300
    credit = 0
    for pid in actor.active:
        p = actor.positions[pid]
        if p.state != "paired" or p.continuity_blocked or pid not in actor.ledger.amounts:
            continue
        key = (p.stream, p.entry_day < actor.day, bucket)
        if key not in actor.release_cache:
            actor.release_cache[key] = release_probability(actor.decider.snapshot, *key)
        credit += int(actor.ledger.amounts[pid] * actor.release_cache[key])
    return min(actor.ledger.cap_cents, actor.overnight_target_cents + credit)


def activate_parked(actor, ns: int, instrument: str | None = None) -> None:
    """Observed top-of-book proximity starts reservation or delayed cancellation.

    A print can jump straight to a parked price before cancellation. Such fills
    are booked by Portfolio.trade, including any capacity overrun.
    """
    if not actor.park_spot:
        return
    candidates = actor.parked if instrument is None else actor.parked_instruments.get(instrument, set())
    # Capacity activation follows original quote arrival, never set/hash order.
    ordered = sorted(candidates, key=lambda oid: (actor.queue.orders[oid].placed_ns, oid)
                     if oid in actor.queue.orders else (0, oid))
    for oid in ordered:
        o = actor.queue.orders.get(oid)
        if o is None:
            actor.parked.discard(oid)
            if instrument is not None:
                candidates.discard(oid)
            continue
        if instrument is not None and instrument != o.instrument:
            continue
        b = actor.market.book(o.instrument, ns)
        if b is None or not b.valid() or o.price < b.bids[0][0]:
            continue
        p = actor.positions[o.position_id]
        reserve = nominal_cents(o.price, p.contract.shares)
        if actor.ledger.committed_cents + reserve <= admission_limit(actor, ns):
            if not actor.ledger.reserve(p.id, reserve, ns):
                raise AssertionError("park activation exceeded hard reservation limit")
            actor.parked.remove(oid)
            actor.parked_instruments.get(o.instrument, set()).discard(oid)
            actor._event(p, ns, "park_activated", reservation_cents=reserve)
        else:
            actor.request_cancel(oid, ns)
