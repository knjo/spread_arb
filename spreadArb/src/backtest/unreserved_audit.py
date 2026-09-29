"""Independent event reconstruction for the user's no-reservation policy.

Uses executed quantities/cash and current raw spot asks, not Actor methods.
Includes negative controls in test_unreserved_audit.py.
"""
from collections import Counter, defaultdict, deque

import numpy as np
import polars as pl

from ..common.books import raw_paths
from ..common.paths import open_ns, SECOND, CLOSE_SECOND


def spot_observations(day, codes):
    """Only the raw spot fields needed to verify observable capital estimates."""
    if not codes:
        return {}
    path, _, _ = raw_paths(day)
    scan = pl.scan_parquet(path).filter(pl.col("QuoteCode").is_in(sorted(codes)))
    bids = [pl.when(pl.col(f"BidLots{i}") > 0).then(pl.col(f"BidPrice{i}")).otherwise(0.) for i in range(1, 6)]
    asks = [pl.when(pl.col(f"AskLots{i}") > 0).then(pl.col(f"AskPrice{i}")).otherwise(float("inf")) for i in range(1, 6)]
    bids.append(pl.when((pl.col("BestBidLots") > 0) & (pl.col("BestBidPrice") > 0)).then(pl.col("BestBidPrice")).otherwise(0.))
    asks.append(pl.when((pl.col("BestAskLots") > 0) & (pl.col("BestAskPrice") > 0)).then(pl.col("BestAskPrice")).otherwise(float("inf")))
    start = open_ns(day)
    df = (scan.select("QuoteCode", pl.col("RecvTime").dt.epoch("ns").alias("ns"),
        pl.col("ChannelSeq").alias("seq"), (pl.col("TrialMatch") == 0).alias("formal"),
        pl.max_horizontal(bids).alias("bid"), pl.min_horizontal(asks).alias("ask"))
        .filter(pl.col("ns").is_between(start-1800*SECOND, start+CLOSE_SECOND*SECOND))
        .with_columns((pl.col("formal") & (pl.col("bid") > 0) & pl.col("ask").is_finite()
                       & (pl.col("bid") < pl.col("ask"))).alias("valid"))
        .sort("QuoteCode", "ns", "seq").collect())
    result = {}
    for g in df.partition_by("QuoteCode", maintain_order=True):
        result[g.item(0, "QuoteCode")] = dict(ns=g["ns"].to_numpy(), seq=g["seq"].to_numpy(),
            ask=g["ask"].to_numpy(), valid=g["valid"].to_numpy())
    return result


class UnreservedAudit:
    def __init__(self, cfg):
        self.cfg = cfg
        self.amounts = {}
        self.entry_cash = defaultdict(int)
        self.previous = {}
        self.counts = defaultdict(int)

    def check_day(self, day, positions, events, cash, ledger, observations):
        failures = []
        orders, inflight, paired = {}, {}, {}
        legs = deque(cash)
        capital_rows = deque(ledger)
        cap = round(self.cfg["cap_twd"] * 100)
        last_entry = {}
        cancel_required = []
        required_updates = []

        def fail(check, **detail):
            failures.append(dict(check=check, day=day, **detail))

        for pid, p in self.previous.items():
            if p["state"] in ("quoting", "entry_hedge", "entry_rollback"):
                inflight[(p["vc"], p["stream"])] = pid
            if p["state"] == "paired" and p["winner"] is None and p["target_bp"] is not None:
                paired[pid] = (p["hedge_ns"], p["quote_ns"], pid)

        def vc_of(pid):
            return pid.split("/")[2]

        def used(vc):
            return sum(money for pid, money in self.amounts.items() if vc_of(pid) == vc)

        def spot_at(vc, ns):
            b = observations.get(vc)
            if b is None:
                return None
            ix = int(np.searchsorted(b["ns"], ns, side="right")) - 1
            if ix < 0 or not b["valid"][ix]:
                return None
            return round(float(b["ask"][ix])*10000), int(b["ns"][ix]), int(b["seq"][ix])

        def full_nominal(o, ns):
            if o["route"] == "S1":
                return (o["price"]*o["qty"]+99)//100
            book = spot_at(o["vc"], ns)
            shares = positions.get(o["id"], {}).get("shares", 2000)
            return (book[0]*shares+99)//100 if book else o["nominal_cents"]

        for e in events:
            kind, ns, pid = e["kind"], e["ns"], e.get("id")
            oid = e.get("order_id")
            if kind == "submit":
                route, vc = e["route"], vc_of(pid)
                if any(not o["done"] and o["vc"] == vc and o["route"] == route for o in orders.values()):
                    fail("multiple_working_route_orders", id=pid, ns=ns, route=route)
                o = {**e, "vc": vc, "filled": 0, "done": False, "cancel_ns": None, "live_ns": None, "rejected": False}
                orders[oid] = o
                if route in ("S1", "S2"):
                    if (vc, route) in inflight:
                        fail("entry_before_previous_hedge_or_cancel", id=pid, ns=ns)
                    inflight[(vc, route)] = pid
                    if e["committed_cents"] != sum(self.amounts.values()):
                        fail("admission_capital_not_current", id=pid, ns=ns)
                    full = full_nominal(o, ns)
                    if full != e["nominal_cents"]:
                        fail("admission_notional_not_observable", id=pid, ns=ns)
                    if full + sum(self.amounts.values()) > cap:
                        fail("entry_exceeds_available_capital", id=pid, ns=ns)
                    if used(vc)+full > max(full, round(cap*self.cfg["product_cap_frac"])):
                        fail("entry_exceeds_product_capital", id=pid, ns=ns)
                else:
                    eligible = [k for k in paired if vc_of(k) == vc]
                    owner = min(eligible, key=paired.get) if eligible else None
                    if owner != pid:
                        fail("exit_not_oldest_eligible_position", id=pid, expected=owner, ns=ns)
                self.counts["route_submissions_checked"] += 1
            elif kind == "live":
                o = orders[oid]
                o["live_ns"] = ns
                if ns != o["ns"] + self.cfg["place_ns"]:
                    fail("placement_latency", id=pid, ns=ns)
            elif kind == "cancel_request":
                o = orders[oid]
                o["cancel_ns"] = ns
            elif kind == "post_only_reject":
                orders[oid]["rejected"] = True
            elif kind == "cancel_effective":
                o = orders[oid]
                if not o["rejected"] and (o["cancel_ns"] is None or ns != o["cancel_ns"]+self.cfg["cancel_ns"]):
                    fail("cancel_latency", id=pid, ns=ns)
                o["done"] = True
            elif kind == "execution":
                if not legs:
                    fail("execution_without_cash", id=pid, ns=ns)
                    continue
                leg = legs.popleft()
                if any(leg[k] != e[k] for k in ("id", "ns", "instrument", "side", "qty", "purpose")):
                    fail("execution_cash_event_mismatch", id=pid, ns=ns)
                purpose = leg["purpose"]
                if leg["liquidity"] == "maker":
                    o = orders[leg["order_id"]]
                    if o["done"] or o["live_ns"] is None or ns <= o["live_ns"]:
                        fail("fill_without_live_order", id=pid, ns=ns)
                    if o["cancel_ns"] is not None and ns > o["cancel_ns"]+self.cfg["cancel_ns"]:
                        fail("fill_after_cancel_effective", id=pid, ns=ns)
                    o["filled"] += leg["qty"]
                    if o["filled"] > o["qty"]:
                        fail("overfilled_order", id=pid, ns=ns)
                    o["done"] = o["filled"] == o["qty"]
                if purpose == "S1":
                    self.entry_cash[pid] += leg["cash"]
                if purpose == "entry" and leg["instrument"].startswith("S:"):
                    self.entry_cash[pid] += leg["cash"]
                if purpose in ("S1", "S2", "entry"):
                    last_entry[pid] = leg
                    required_updates.append((pid, ns, "hedged" if purpose == "entry" else "maker_fill"))
                if purpose in ("E1", "E2", "forced", "settlement"):
                    paired.pop(pid, None)
                if purpose in ("entry", "exit_rollback"):
                    p = positions[pid]
                    if p["target_bp"] is not None:
                        paired[pid] = (p["hedge_ns"], p["quote_ns"], pid)
                    if purpose == "entry":
                        inflight.pop((p["vc"], p["stream"]), None)
            elif kind == "capital":
                if not capital_rows:
                    fail("capital_event_without_ledger", id=pid, ns=ns)
                    continue
                r = capital_rows.popleft()
                capital_kind = e["capital_kind"]
                if (r["id"], r["ns"], r["kind"]) != (pid, ns, capital_kind):
                    fail("capital_event_ledger_mismatch", id=pid, ns=ns)
                leg = last_entry.get(pid)
                if not leg or leg["ns"] != ns or (capital_kind == "maker_fill") != (leg["liquidity"] == "maker"):
                    fail("capital_not_at_actual_entry_execution", id=pid, ns=ns)
                p = positions[pid]
                if capital_kind == "maker_fill" and p["stream"] == "S2":
                    book = spot_at(p["vc"], ns)
                    cents = (book[0]*p["shares"]+99)//100 if book else p["quote_nominal_cents"]
                    if book and (r.get("estimate_ask"), r.get("estimate_book_ns"), r.get("estimate_book_seq")) != book:
                        fail("future_or_wrong_s2_capital_estimate", id=pid, ns=ns)
                    self.counts["s2_raw_capital_estimates_checked"] += 1
                else:
                    cents = (self.entry_cash[pid]+99)//100
                delta = cents - self.amounts.get(pid, 0)
                old_total = sum(self.amounts.values())
                self.amounts[pid] = cents
                total = sum(self.amounts.values())
                if (e["delta_cents"], e["amount_cents"], e["committed_cents"]) != (delta, cents, total):
                    fail("capital_not_actual_cash_or_estimate", id=pid, ns=ns)
                if (r["delta_cents"], r["committed_cents"]) != (delta, total):
                    fail("actual_capital_ledger_mismatch", id=pid, ns=ns)
                if r.get("excess_cents") != max(0, total-cap):
                    fail("capital_excess_amount", id=pid, ns=ns)
                if total > cap and delta > 0:
                    if capital_kind == "hedged" and p["stream"] == "S2":
                        self.counts["hedge_repricing_over_cap"] += 1
                    else:
                        o = orders[leg["order_id"]]
                        in_cancel = o["cancel_ns"] is not None and o["cancel_ns"] <= ns <= o["cancel_ns"]+self.cfg["cancel_ns"]
                        repriced = p["stream"] == "S2" and cents > o["nominal_cents"]
                        if not (in_cancel or repriced):
                            fail("unexplained_capital_overshoot", id=pid, ns=ns, prior=old_total, current=total)
                        self.counts["cancel_race_over_cap" if in_cancel else "maker_estimate_repricing_over_cap"] += 1
                for pending_oid, o in orders.items():
                    if o["route"] not in ("S1", "S2") or o["done"] or o["cancel_ns"] is not None:
                        continue
                    full = full_nominal(o, ns)
                    needed = (o["price"]*(o["qty"]-o["filled"])+99)//100 if o["route"] == "S1" else full
                    if needed > cap-total or used(o["vc"])+needed > max(full, round(cap*self.cfg["product_cap_frac"])):
                        cancel_required.append((pending_oid, ns))
                self.counts["actual_capital_updates_checked"] += 1
            elif kind == "closed":
                amount = self.amounts.pop(pid, 0)
                if amount:
                    if not capital_rows:
                        fail("closed_without_capital_release", id=pid, ns=ns)
                    else:
                        r = capital_rows.popleft()
                        if (r["id"], r["ns"], r["kind"], r["delta_cents"], r["committed_cents"]) != \
                                (pid, ns, "release", -amount, sum(self.amounts.values())):
                            fail("capital_release_not_at_close", id=pid, ns=ns)
                paired.pop(pid, None)
                stream = pid.split("/")[0]
                if inflight.get((vc_of(pid), stream)) == pid:
                    inflight.pop((vc_of(pid), stream))

        for oid, ns in cancel_required:
            if orders[oid]["cancel_ns"] != ns:
                fail("missing_immediate_capacity_cancel", id=orders[oid]["id"], ns=ns, order_id=oid)
            self.counts["mandatory_capacity_cancels_checked"] += 1
        if legs or capital_rows:
            fail("unmatched_cash_or_capital_rows", cash=len(legs), capital=len(capital_rows))
        booked = Counter((r["id"], r["ns"], r["kind"]) for r in ledger if r["kind"] in ("maker_fill", "hedged"))
        for (pid, ns, kind), count in (Counter(required_updates)-booked).items():
            fail("unbooked_entry_execution", id=pid, ns=ns, kind=kind, count=count)
        self.previous = {pid: p for pid, p in positions.items() if p["state"] != "closed"}
        return failures
