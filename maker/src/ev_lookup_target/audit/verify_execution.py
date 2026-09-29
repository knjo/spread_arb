"""Independent full-output checks, including truthful parked-order overruns."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from ...ev_lookup_cost.ev_rules import cell_of
from ...ev_lookup_cost.verify_run import verify_raw_fills


def read_rows(path: Path) -> list[dict]:
    return pl.read_parquet(path).to_dicts() if path.exists() else []


def verify(root: Path, raw_days: set[str]) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise AssertionError("full study has not finished")
    source_top = Path(__file__).resolve().parents[2]
    for name, digest in manifest["sources"].items():
        stored = root / "source_snapshot" / Path(name).relative_to(source_top)
        if hashlib.sha256(stored.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"source snapshot mismatch: {name}")
    input_files = {}
    for i, (name, digest) in enumerate(manifest["inputs_sha256"].items()):
        stored = root / "input_snapshot" / f"{i:03d}_{Path(name).name}"
        input_files[name] = stored
        if hashlib.sha256(stored.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"input snapshot mismatch: {name}")
    configs = {r["name"]:r for r in manifest["configurations"]}
    configs["shadow"] = dict(cap_twd=10**12, park_spot=False)
    snapshots = json.loads((root / "snapshots.json").read_text())
    action_file = next(p for name,p in input_files.items() if name.endswith("announcements/index.json"))
    actions = [r for r in json.loads(action_file.read_text()) if "error" not in r]
    balances, totals, clocks, peaks = defaultdict(dict), defaultdict(int), defaultdict(int), defaultdict(int)
    history, risk_history = [], []
    checks = dict(days=len(manifest["days"]), portfolio_days=0, maker_fill_events=0,
                  raw_prints_checked=0, overrun_fill_events=defaultdict(int),
                  unreserved_fill_events=defaultdict(int), adverse_pairs=0, expiry_positions=0)
    for day_index, day in enumerate(manifest["days"]):
        start, end = open_ns(day), open_ns(day)+15600*SECOND
        blocked = {r["vc"] for r in actions if r["announce_day"] < day < r["effective_day"]}
        for name, config in configs.items():
            folder = root / f"Date={day}" / name
            if not folder.exists():
                raise AssertionError(f"missing portfolio day {name} {day}")
            checks["portfolio_days"] += 1
            positions = {r["id"]:r for r in read_rows(folder / "positions.parquet")}
            trace = read_rows(folder / "execution.parquet")
            events_by_position = defaultdict(list)
            for event in trace:
                events_by_position[event["position_id"]].append(event)
                if not start <= event["ns"] <= end:
                    raise AssertionError("execution outside replay session")
                if event["kind"] == "maker_fill":
                    checks["maker_fill_events"] += 1
                if event["kind"] == "taker_fill" and event["book_ns"] > event["ns"]:
                    raise AssertionError("future hedge book")
                if event["kind"] == "quote" and event["vc"] in blocked:
                    raise AssertionError("entry ignored published corporate-action restriction")
                if event["kind"] == "expiry_basis_zero_accounting":
                    checks["expiry_positions"] += 1
                    if positions[event["position_id"]]["contract"]["expiry"] >= day or event["ns"] != start:
                        raise AssertionError("wrong original-contract expiry date")
            for d in read_rows(folder / "decisions.parquet"):
                if d["admit"] and (d["quote_ab"] <= 0 or d["vc"] in blocked):
                    raise AssertionError("invalid quote admission")
                if d["admit"] and not d.get("parked",False):
                    if d["committed_cents"]+d["reservation_cents"] > d["admission_cap_cents"]:
                        raise AssertionError("admitted reserved quote exceeded observable admission limit")
            cap = config["cap_twd"]*100
            for event in read_rows(folder / "ledger.parquet"):
                pid, ns, kind, delta = event["id"], event["ns"], event["kind"], event["delta_cents"]
                if ns < clocks[name] or not start <= ns <= end:
                    raise AssertionError("capacity clock moved backwards")
                clocks[name] = ns
                if kind in {"reserve", "unreserved_fill"}:
                    if pid in balances[name] or delta <= 0:
                        raise AssertionError("duplicate capacity reservation")
                    balances[name][pid] = delta
                elif kind == "hedged":
                    if pid not in balances[name] or delta > 0:
                        raise AssertionError("hedge cannot enlarge pre-reserved nominal")
                    balances[name][pid] += delta
                elif kind == "release":
                    if balances[name].pop(pid, None) != -delta:
                        raise AssertionError("released amount differs from actual reservation")
                    p = positions[pid]
                    if p["state"] not in {"closed", "cancelled"}:
                        raise AssertionError("released nonterminal position")
                    if p["state"] == "closed" and ns != p["close_ns"]:
                        raise AssertionError("released capital before both exit legs")
                else:
                    raise AssertionError(f"unknown ledger event {kind}")
                prior = totals[name]
                totals[name] += delta
                peaks[name] = max(peaks[name],totals[name])
                if totals[name] != event["committed_cents"] or totals[name] != sum(balances[name].values()):
                    raise AssertionError("ledger does not reconcile across days")
                if kind == "unreserved_fill":
                    checks["unreserved_fill_events"][name] += 1
                    if (not config.get("park_spot") and config.get("reserve_quotes", True)) or not any(t["kind"] == "maker_fill" and t["ns"] == ns
                                                              for t in events_by_position[pid]):
                        raise AssertionError("overrun is not backed by an irrevocable maker fill")
                    checks["overrun_fill_events"][name] += int(totals[name] > cap)
                if totals[name] > cap and not (kind == "unreserved_fill" or prior > cap and delta <= 0):
                    raise AssertionError("capacity overrun was created by voluntary admission")
            expected_active = {pid for pid,p in positions.items() if p["state"] not in {"closed","cancelled"}}
            if set(balances[name]) != expected_active:
                raise AssertionError("carry or unhedged exposure disappeared from capacity ledger")
            if day in manifest["data_outage_days"] and trace:
                if any(t["kind"] in {"maker_fill","taker_fill","quote","expiry_basis_zero_accounting"} for t in trace):
                    raise AssertionError("fabricated execution on the data-outage session")
            for p in positions.values():
                if p["entry_day"] == day and p["actual_ab"] is not None and p["actual_ab"] <= 0:
                    checks["adverse_pairs"] += 1
                if p["hedged_ns"] is not None and p["hedged_ns"] < p["entry_fill_ns"]:
                    raise AssertionError("hedge precedes first-leg fill")
                if p["state"] == "closed":
                    if p["close_ns"] <= p["entry_fill_ns"]:
                        raise AssertionError("exit precedes entry")
                    if p["spot_buy_qty"] != p["spot_sell_qty"] or p["future_sell_qty"] != p["future_buy_qty"]:
                        raise AssertionError("closed position retains exposure")
                    gross=(p["spot_sell_cash"]-p["spot_buy_cash"]+p["future_sell_cash"]-p["future_buy_cash"])/10_000
                    fee=p["spot_buy_cash"]/1e8*(20 if p["entry_day"] == day else 34)
                    if abs(p["pnl_twd"]-(gross-fee)) > 1e-6:
                        raise AssertionError("PnL does not reconcile to four legs")
                    if name == "shadow":
                        history.append(p)
            if day in raw_days and any(t["kind"] == "maker_fill" for t in trace):
                checks["raw_prints_checked"] += verify_raw_fills(day,pl.from_dicts(trace,infer_schema_length=None))
        risk_history += read_rows(root / f"Date={day}" / "capacity_observations.parquet")
        for snap in [s for s in snapshots if s["day"] == day]:
            if snap["train_days"] != manifest["days"][max(0,day_index-20):day_index]:
                raise AssertionError("training window contains an unavailable session")
            expected = {}
            for p in history:
                if p["close_day"] in snap["train_days"] and p["close_ns"] < snap["cutoff_ns"]:
                    values=expected.setdefault(cell_of(p["stream"],p["quote_ab"]),[0.,0.,0])
                    values[0] += p["pnl_bp"]
                    values[1] += max(manifest["days"].index(p["close_day"])-p["entry_session"],.15)
                    values[2] += 1
            if set(expected) != set(snap["cells"]):
                raise AssertionError("morning lookup keys mismatch")
            for cell,values in expected.items():
                if any(abs(a-b)>1e-6 for a,b in zip(values,snap["cells"][cell])):
                    raise AssertionError("morning lookup used future/current-day outcomes")
            n=sum(r["day"] in snap["train_days"] and r["available_ns"] < snap["cutoff_ns"] for r in risk_history)
            if n != snap["capacity_observations_count"]:
                raise AssertionError("adaptive capacity used unavailable outcomes")
    checks.update(passed=True, peak_committed_twd={n:v/100 for n,v in peaks.items()},
                  final_committed_twd={n:v/100 for n,v in totals.items()},
                  interpretation="shared-quote ceilings are admission limits; actual race overruns are measured, not discarded")
    (root / "verification_full.json").write_text(json.dumps(checks,indent=2)+"\n")
    return checks


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--raw-days",nargs="*",default=[])
    args=parser.parse_args()
    print(json.dumps(verify(args.root,set(args.raw_days)),indent=2))
