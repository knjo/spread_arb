"""Reconstruct the daily exit risk tables and audit cancellation races from outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path

import polars as pl

from .verify_full_study import read_rows, verify


def verify_model(root: Path, raw_days: set[str]) -> dict:
    base = verify(root, raw_days)
    manifest = json.loads((root/"manifest.json").read_text())
    snapshots = {(s["day"],s["portfolio"]):s for s in json.loads((root/"snapshots.json").read_text())}
    calendar_spec = manifest["forecast_calendar"]
    configs = {c["name"]:c for c in manifest["configurations"]}
    configs["shadow"] = dict(split_ev=manifest["configurations"][0]["split_ev"],
                             event_quotes=manifest["configurations"][0]["event_quotes"])
    history = []
    checked_decisions = 0
    counts = defaultdict(lambda:defaultdict(int))
    risk_checks = 0
    for day in manifest["days"]:
        closed = set(calendar_spec["planned_holidays"])
        closed.update(r["day"] for r in calendar_spec["extra_closures"] if r["publication_day"] < day)
        for name, config in configs.items():
            snap = snapshots[day,name]
            expected = {}
            for r in history:
                if r["day"] not in snap["train_days"] or r["available_ns"] >= snap["cutoff_ns"]:
                    continue
                if not r["carry"] or r["start_second"] > 0 or not r.get("expiry") or r["day"] > r["expiry"]:
                    continue
                if r.get("close_kind") == "expiry_basis_zero_accounting":
                    continue
                remaining = (datetime.strptime(r["expiry"],"%Y%m%d")-datetime.strptime(r["day"],"%Y%m%d")).days
                b = 0 if remaining <= 0 else 1 if remaining <= 3 else 2 if remaining <= 10 else 3
                normal = r["closed"] and r["close_kind"] == "maker_exit"
                other = r["closed"] and not normal
                for k in ("all",r["stream"],f'{r["stream"]}:{b}'):
                    x = expected.setdefault(k,[0,0,0,0.])
                    x[0] += 1; x[1] += int(normal); x[2] += int(other)
                    x[3] += r["net_bp"] if other else 0.
            if set(expected) != set(snap["exit_hazards"]):
                raise AssertionError("exit hazard keys used unavailable or omitted risk observations")
            for k,values in expected.items():
                if any(abs(a-b)>1e-8 for a,b in zip(values,snap["exit_hazards"][k])):
                    raise AssertionError("exit risk denominator or outcome classification mismatch")
            risk_checks += 1
            folder = root/f"Date={day}"/name
            positions = {p["id"]:p for p in read_rows(folder/"positions.parquet")}
            for d in read_rows(folder/"decisions.parquet"):
                if "expiry" not in d:
                    continue  # pre-final smoke trace; full study must have this field
                second = (d["ns"]-snap["cutoff_ns"])//1_000_000_000
                bucket = 0 if second < 3600 else 1 if second < 9000 else 2
                n, nsd = snap["psd"].get(f'{d["stream"]}_{bucket}', (0,0))
                p = nsd/n if n >= 200 else .68
                survival, on, other, other_value = 1-p, 0., 0., 0.
                current = datetime.strptime(day,"%Y%m%d")+timedelta(days=1)
                expiry = datetime.strptime(d["expiry"],"%Y%m%d")
                if not calendar_spec["start"] <= day <= d["expiry"] <= calendar_spec["end"]:
                    raise AssertionError("forecast exceeds verified calendar coverage")
                while current <= expiry:
                    date = current.strftime("%Y%m%d")
                    if current.weekday()<5 and date not in closed:
                        rem = (expiry-current).days
                        b = 0 if rem <= 0 else 1 if rem <= 3 else 2 if rem <= 10 else 3
                        x = next((expected[k] for k in (f'{d["stream"]}:{b}',d["stream"],"all")
                                  if k in expected and expected[k][0]>=30),None)
                        q,qo,mean = (x[1]/x[0],x[2]/x[0],x[3]/x[2] if x[2] else 0.) if x else (.5,0.,0.)
                        on += survival*q
                        other += survival*qo
                        other_value += survival*qo*mean
                        survival *= 1-q-qo
                    current += timedelta(days=1)
                est = (p*(d["eff_u"]+5-20)+on*(d["eff_u"]+5-34)+survival*(d["quote_ab"]-34)+other_value
                       if config["split_ev"] else p*(d["eff_u"]+5-20)+(1-p)*(d["quote_ab"]-34))
                est -= d.get("execution_cost_bp", 0.0)
                for actual, wanted in [(d["est_bp"],est),(d["p_sd"],p),(d["p_overnight"],on),
                                       (d["p_expiry"],survival),(d["p_other"],other)]:
                    if abs(actual-wanted)>1e-7:
                        raise AssertionError("quote EV is not reconstructed from the morning risk table")
                checked_decisions += 1
            trace = read_rows(folder/"execution.parquet")
            invalid = {t["position_id"]:t for t in trace if t["kind"] == "s2_first_invalid"}
            cancellation = {t["position_id"]:t for t in trace if t["kind"] == "event_cancel"}
            for event in trace:
                if event["kind"] == "s2_invalid_fill":
                    request = invalid[event["position_id"]]
                    lead = event["ns"]-request["ns"]
                    if lead != event["lead_ns"] or lead < 0:
                        raise AssertionError("invalidation occurs after its attributed fill")
                    possible = lead > manifest["cancel_ms"]*1_000_000
                    if bool(event["cancel_could_arrive"]) != possible:
                        raise AssertionError("incorrect cancellation race classification")
                    if config["event_quotes"] and possible:
                        raise AssertionError("an event-cancelled order filled after cancellation could take effect")
                    counts[name]["avoidable_original_fill" if possible else "race_fill"] += 1
                if event["kind"] == "maker_fill" and event["stream"] == "S2" and event["purpose"] == "entry":
                    counts[name]["s2_fills"] += 1
                    p = positions[event["position_id"]]
                    if p["future_sell_qty"] <= 0 or p["state"] == "cancelled":
                        raise AssertionError("maker execution disappeared after cancellation")
                if event["kind"] == "event_cancel":
                    if event["effective_ns"]-event["ns"] != manifest["cancel_ms"]*1_000_000:
                        raise AssertionError("cancel latency changed")
                    counts[name]["event_cancels"] += 1
        history += read_rows(root/f"Date={day}"/"capacity_observations.parquet")
    result = dict(passed=True, risk_snapshots=risk_checks, decisions=checked_decisions,
                  cancellation_counts={k:dict(v) for k,v in counts.items()}, base=base)
    (root/"verification_v21.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--raw-days",nargs="*",default=[])
    a = p.parse_args()
    print(json.dumps(verify_model(a.root,set(a.raw_days)),indent=2))
