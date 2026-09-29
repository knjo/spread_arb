"""Independent quote lifetime and current-book price checks; no Actor calls."""
from collections import defaultdict
from math import isfinite, isnan

import polars as pl

from ..common.books import previous_tick, tick_i
from ..common.paths import grid_path, SECOND


def audit_refresh(events, cash, cfg, end):
    ttl = cfg.get("quote_refresh_ns")
    if ttl is None:  # Archived policy did not have a refresh deadline.
        return {}, []
    orders, closed, counts, failures = {}, {}, defaultdict(int), []
    for e in events:
        kind, ns, oid = e["kind"], e["ns"], e.get("order_id")
        if kind == "submit":
            prior = e.get("replaces_order_id")
            if prior:
                old = orders.get(prior)
                if (old is None or old.get("cancelled", end+1) >= ns
                        or closed.get(old["id"], end+1) >= ns):
                    failures.append(dict(check="refresh_before_old_entry_flat", order_id=oid, ns=ns))
                counts["entry_replacements_checked"] += 1
            orders[oid] = {**e, "filled": 0}
        elif kind == "closed":
            closed[e["id"]] = ns
        elif oid in orders:
            o = orders[oid]
            if kind == "live":
                o["live"] = ns
            elif kind == "cancel_request":
                o["cancel_request"] = ns
            elif kind == "cancel_effective":
                o["cancelled"] = ns
            elif kind == "quote_expired":
                deadline = o.get("live", end+1) + ttl
                if ns != deadline or e.get("deadline_ns") != deadline:
                    failures.append(dict(check="wrong_quote_deadline", order_id=oid, ns=ns))
                counts["quote_expirations_checked"] += 1
    for r in cash:
        if r.get("liquidity") != "maker":
            continue
        o = orders[r["order_id"]]
        o["filled"] += r["qty"]
        if o["filled"] >= o["qty"]:
            o["full_ns"] = r["ns"]
        if r["ns"] > o["live"] + ttl + cfg["cancel_ns"]:
            failures.append(dict(check="fill_after_quote_deadline", order_id=r["order_id"], ns=r["ns"]))
    for oid, o in orders.items():
        if "live" not in o:
            continue
        counts["quote_lifetimes_checked"] += 1
        deadline = o["live"]+ttl
        if min(o.get("full_ns", end+1), o.get("cancelled", end+1), end) <= deadline:
            continue
        if o.get("cancel_request", end+1) > deadline:
            failures.append(dict(check="missing_quote_deadline_cancel", order_id=oid, deadline=deadline))
        if min(o.get("full_ns", end+1), o.get("cancelled", end+1)) > deadline+cfg["cancel_ns"]:
            failures.append(dict(check="quote_survived_cancel_latency", order_id=oid, deadline=deadline))
    return dict(counts), failures


def audit_quote_prices(events, orders, books):
    """All sends, including refreshed exits, must quote the then-current BBO."""
    failures, count = [], 0
    for e in events:
        if e["kind"] != "submit":
            continue
        o = orders[e["order_id"]]
        book = books[o["instrument"]]
        i = book.index_at(e["ns"])
        if i < 0:
            failures.append(dict(check="quote_before_raw_book", order_id=e["order_id"]))
            continue
        route = e["route"]
        if route in ("S1", "E2"):
            px = int(book.top_bid_px[i])
            expected = px if route == "S1" else px + tick_i(px)
        else:
            px = int(book.top_ask_px[i])
            expected = px if route == "E1" else previous_tick(px)
        count += 1
        if e["price"] != expected:
            failures.append(dict(check="quote_not_current_price", order_id=e["order_id"],
                                 ns=e["ns"], calculated=expected, reported=e["price"]))
    return {"raw_quote_prices_checked": count}, failures


def signal_grid(day, rows):
    if not rows:
        return {}
    keys = pl.from_dicts(rows, infer_schema_length=None).select("vc", "qc", "quote_second").unique()
    values = (pl.scan_parquet(grid_path(day)).select(
        pl.col("ValueCode").alias("vc"), pl.col("QuoteCode").alias("qc"),
        pl.col("seconds_from_open").alias("quote_second"),
        pl.col("anchor_ewma_120s_bp").alias("anchor"), pl.col("basis_mid_bp").alias("mid"))
        .join(keys.lazy(), on=["vc", "qc", "quote_second"], how="inner").collect())
    return {(r["vc"], r["qc"], r["quote_second"]): (r["anchor"], r["mid"])
            for r in values.iter_rows(named=True)}


def audit_decision_inputs(rows, books, grid, start):
    """Rebuild observable EV inputs instead of trusting saved decision fields."""
    failures = []
    for r in rows:
        ns = r["ns"]
        sec = (ns-start)//SECOND
        spot, fut = books.get("S:"+r["vc"]), books.get("F:"+r["qc"])
        si, fi = spot.index_at(ns) if spot is not None else -1, fut.index_at(ns) if fut is not None else -1
        if si < 0 or fi < 0:
            failures.append(dict(check="decision_before_raw_book", ns=ns, vc=r["vc"]))
            continue
        sb, sa = int(spot.top_bid_px[si]), int(spot.top_ask_px[si])
        fb, fa = int(fut.top_bid_px[fi]), int(fut.top_ask_px[fi])
        price = sb if r["stream"] == "S1" else previous_tick(fa)
        hedge = fb if r["stream"] == "S1" else sa
        basis = (fb/price-1)*10000 if r["stream"] == "S1" else (price/sa-1)*10000
        expected = dict(quote_second=sec, price=price, spot_a1=sa, quote_ab=basis,
            tick_bp_hedge=tick_i(hedge)/hedge*10000,
            execution_floor_bp=tick_i(hedge)/hedge*5000 if r["stream"] == "S2" else 0.)
        inputs = grid.get((r["vc"], r["qc"], sec))
        if inputs is None:
            failures.append(dict(check="decision_missing_grid", ns=ns, vc=r["vc"]))
        else:
            anchor, mid = inputs
            anchor = float("nan") if anchor is None else anchor
            mid = float("nan") if mid is None else mid
            expected.update(anchor=anchor, resid_mid_bp=mid-anchor,
                            e_norm=(mid-anchor)/(r["scale"] or 1.))
        for key, value in expected.items():
            actual = r.get(key)
            # Absolute/settlement EV can legitimately omit the normalized
            # residual. Preserve the raw missing value; never invent a number.
            if key in ("resid_mid_bp", "e_norm") and actual is not None and isnan(actual) and isnan(value):
                continue
            if actual is None or value is None or not isfinite(actual) or not isfinite(value) or abs(actual-value) > 1e-8:
                failures.append(dict(check="raw_decision_input", ns=ns, vc=r["vc"], stream=r["stream"],
                    field=key, calculated=value if value is None or isfinite(value) else str(value),
                    reported=actual if actual is None or isfinite(actual) else str(actual)))
    return {"raw_decision_inputs_checked": len(rows)}, failures
