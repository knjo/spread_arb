"""Independent raw-print queue reconstruction for one order per route.

There is at most one own order on each side of an instrument under the accepted
policy. This permits an independent ahead/remaining calculation without using
the engine's PrintedVolumeQueue implementation.
"""
from collections import Counter, defaultdict

import numpy as np


def submitted_orders(events, positions, mapping):
    result = {}
    for e in events:
        if e["kind"] != "submit":
            continue
        pid, route = e["id"], e["route"]
        vc = pid.split("/")[2]
        qc = positions[pid]["qc"] if pid in positions else mapping[vc]
        result[e["order_id"]] = dict(id=e["order_id"], pid=pid, route=route, price=e["price"],
            qty=e["qty"], remaining=e["qty"], side="buy" if route in ("S1", "E2") else "sell",
            instrument="S:"+vc if route in ("S1", "E1") else "F:"+qc)
    return result


def audit_fifo(events, cash, positions, mapping, books, prints):
    orders = submitted_orders(events, positions, mapping)
    actions = defaultdict(list)
    failures, counts = [], defaultdict(int)
    for ordinal, e in enumerate(events):
        if e["kind"] in ("live", "cancel_effective"):
            o = orders[e["order_id"]]
            actions[o["instrument"]].append((e["ns"], 0 if e["kind"] == "cancel_effective" else 1, ordinal, e))
    predicted = Counter()
    reported = Counter((r["order_id"], r["ns"], r["sequence"], r["qty"]) for r in cash if r["liquidity"] == "maker")
    for instrument, acts in actions.items():
        pr = prints.get(instrument)
        active, j = {}, 0

        def consume(until):
            nonlocal j
            if pr is None:
                return
            end = int(np.searchsorted(pr.ns, until, side="right"))
            if not active:
                j = end
                return
            while j < end:
                ns, seq, price, qty = int(pr.ns[j]), int(pr.seq[j]), int(pr.price[j]), int(pr.qty[j])
                eligible = [o for o in active.values() if o["live_ns"] < ns and
                    (price <= o["price"] if o["side"] == "buy" else price >= o["price"])]
                eligible.sort(key=lambda o: (o["side"] != "sell", o["price"] if o["side"] == "sell" else -o["price"]))
                budget = qty
                for o in eligible:
                    ahead = min(o["ahead"], budget)
                    o["ahead"] -= ahead
                    budget -= ahead
                    fill = min(o["remaining"], budget)
                    if fill:
                        predicted[(o["id"], ns, seq, fill)] += 1
                        o["remaining"] -= fill
                        budget -= fill
                        if not o["remaining"]:
                            del active[o["id"]]
                j += 1
                counts["raw_prints_examined_while_working"] += 1

        for ns, _, _, event in sorted(acts):
            consume(ns)
            oid = event["order_id"]
            if event["kind"] == "cancel_effective":
                active.pop(oid, None)
                continue
            o = orders[oid]
            b = books.get(instrument)
            i = b.index_at(ns) if b is not None else -1
            levels = b.levels(i, "bid" if o["side"] == "buy" else "ask") if i >= 0 else []
            ahead = sum(q for price, q in levels if price == o["price"])
            if event["ahead"] != ahead:
                failures.append(dict(check="raw_queue_ahead", id=o["pid"], ns=ns, calculated=ahead, reported=event["ahead"]))
            if any(other["side"] == o["side"] for other in active.values()):
                failures.append(dict(check="multiple_same_side_working_orders", id=o["pid"], ns=ns))
            if any(other["side"] != o["side"] and
                   (o["price"] >= other["price"] if o["side"] == "buy" else o["price"] <= other["price"])
                   for other in active.values()):
                failures.append(dict(check="crossed_own_post_only_orders", id=o["pid"], ns=ns))
            o.update(ahead=ahead, live_ns=ns)
            active[oid] = o
            counts["raw_queue_placements_checked"] += 1
        if pr is not None and len(pr.ns):
            consume(int(pr.ns[-1]))
    for key, count in (reported-predicted).items():
        failures.append(dict(check="maker_fill_not_supported_by_raw_queue", order_id=key[0], ns=key[1],
                             sequence=key[2], qty=key[3], count=count))
    for key, count in (predicted-reported).items():
        failures.append(dict(check="raw_queue_fill_missing_from_replay", order_id=key[0], ns=key[1],
                             sequence=key[2], qty=key[3], count=count))
    counts["raw_queue_fills_checked"] = sum(predicted.values())
    return dict(counts), failures
