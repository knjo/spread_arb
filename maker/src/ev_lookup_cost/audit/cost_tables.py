"""Rebuild costs independently from shadow cash, including exact daily provenance."""
from collections import defaultdict
import json

from ..verify_full_study import read_rows


def lookup(snap, keys, prior, margin):
    for key in keys:
        n, total, _ = snap["decay"].get("|".join(map(str,key)), (0,0.,0.))
        if n >= 30:
            return max(0.,total/n)+margin
    return prior+margin


def quote_costs(snap, d, config):
    stream, second = d["stream"], (d["ns"]-snap["cutoff_ns"])//1_000_000_000
    time = 0 if second < 3600 else 1 if second < 9000 else 2
    spread = d["quote_spread_bp"]
    sb = 0 if spread < 30 else 1 if spread < 80 else 2
    margin = config["decay_margin_bp"]
    entry = lookup(snap, [("entry",stream,time,sb),("entry",stream,-1,sb),
                         ("entry",stream,-1,-1),("entry","*",-1,-1)], 15., margin)
    exits = [lookup(snap, [("exit",stream,b,-1),("exit","*",b,-1)], 5., margin) for b in (0,1)]
    return max(entry,d.get("execution_cost_bp",0.)), *exits


def tick(px):
    return 100 if px < 100000 else 500 if px < 500000 else 1000 if px < 1000000 else 5000 if px < 5000000 else 10000 if px < 10000000 else 50000


def verify_priority(d):
    if d["stream"] != "S2":
        return
    price, bid, ask = (round(d[k]*10_000) for k in ("quote_price","future_bid","future_ask"))
    if price != ask-tick(ask-1) or not bid < price < ask or d["future_book_ns"] > d["ns"]:
        raise AssertionError("new S2 order lost first priority or used a future book")
    if d["liquidity_book_ns"] > d["ns"] or d["liquidity_a1_shares"] < 5*d["liquidity_hedge_shares"]:
        raise AssertionError("future/insufficient hedge depth at entry")
    raw = d["quote_price"]/d["liquidity_a1_price"]*10_000
    scenario = .5*d["quote_price"]*(1/d["liquidity_depth_vwap"]+1/d["liquidity_adverse_vwap"])*10_000
    if abs(d["execution_cost_bp"]-(raw-scenario))>1e-7:
        raise AssertionError("incorrect fixed scenario floor")


def verify_tables(root):
    manifest = json.loads((root/"manifest.json").read_text())
    snapshots = json.loads((root/"snapshots.json").read_text())
    expected_rows, seen_entry = [], set()
    verified_snapshots = 0
    for day in manifest["days"]:
        for snap in (s for s in snapshots if s["day"] == day):
            expected = defaultdict(lambda:[0,0.,0.])
            for r in expected_rows:
                if r["day"] not in snap["train_days"] or r["available_ns"] >= snap["cutoff_ns"]:
                    continue
                kind, stream, bucket, spread = (r[k] for k in ("kind","stream","bucket","spread"))
                keys = ([(kind,stream,bucket,spread),(kind,stream,-1,spread),(kind,stream,-1,-1),(kind,"*",-1,-1)]
                        if kind == "entry" else [(kind,stream,bucket,-1),(kind,"*",bucket,-1)])
                for key in keys:
                    x = expected["|".join(map(str,key))]
                    x[0] += 1; x[1] += r["bp"]; x[2] += r["bp"]**2
            if set(expected) != set(snap["decay"]):
                raise AssertionError("cost table has missing or future keys")
            for key, wanted in expected.items():
                if any(abs(a-b)>max(1e-7,abs(b)*1e-12) for a,b in zip(snap["decay"][key],wanted)):
                    raise AssertionError("cost table does not reconcile to prior shadow cash")
            verified_snapshots += 1
        folder = root/f"Date={day}"
        positions = read_rows(folder/"shadow/positions.parquet")
        rows = []
        cutoff = next(s["cutoff_ns"] for s in snapshots if s["day"] == day)+15_600_000_000_000
        if day not in manifest["data_outage_days"]:
            for p in positions:
                if p["continuity_blocked"] or p["stream"] not in {"S1","S2"}:
                    continue
                if p["hedged_ns"] is not None and p["actual_ab"] is not None and p["id"] not in seen_entry:
                    if p["hedged_ns"] > cutoff:
                        raise AssertionError("hedge outcome recorded before hedge execution")
                    second,spread = p["quote_second"],p["quote_spread_bp"]
                    rows.append(dict(position_id=p["id"], stream=p["stream"], kind="entry",
                        bucket=0 if second<3600 else 1 if second<9000 else 2,
                        spread=0 if spread<30 else 1 if spread<80 else 2,
                        bp=p["quote_ab"]-(p["future_sell_cash"]/p["spot_buy_cash"]-1)*10_000,
                        day=day,available_ns=cutoff))
                    seen_entry.add(p["id"])
                if p["close_day"] == day and p["close_kind"] == "maker_exit":
                    rows.append(dict(position_id=p["id"],stream=p["stream"],kind="exit",
                        bucket=int(p["entry_day"]!=day),spread=-1,
                        bp=(p["future_buy_cash"]-p["spot_sell_cash"])/p["spot_buy_cash"]*10_000-(p["anchor"]-5),
                        day=day,available_ns=cutoff))
        actual = {(r["kind"],r["position_id"]):r for r in read_rows(folder/"decay_observations.parquet")}
        if len(actual) != len(rows):
            raise AssertionError("missing or duplicate daily cost observations")
        for r in rows:
            got = actual[r["kind"],r["position_id"]]
            if any(got[k]!=v for k,v in r.items() if k != "bp") or abs(got["bp"]-r["bp"])>1e-7:
                raise AssertionError("daily cost observation differs from actual shadow cash/timing")
        expected_rows.extend(rows)
    return dict(snapshots=verified_snapshots, observations=len(expected_rows), entry_observations=len(seen_entry))
