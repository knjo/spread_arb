"""Reproduce v17/v18 allocation, then isolate date-safe bpday sensitivity.

All variants share unchanged, outcome-bearing execution facts. Even the date-safe
variants are allocation diagnostics, not certified live-executable strategies.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import polars as pl

from audit_common import OUT, SNAPSHOT, days

sys.path.insert(0, str(SNAPSHOT))
from decide import EntryDecider
from ev_rules import cell_of, cell_gate


def ordered_candidates(rows: list[dict], reverse_ties: bool = False) -> list[dict]:
    return sorted(rows, key=lambda r: (r["t0"],
        (0 if r["strm"] == "S1" else 1) if reverse_ties else (0 if r["strm"] == "S2" else 1),
        r["vc"], r.get("source_order", r["cid"])))


def table_snapshot(rows: list[dict], di: int, didx: dict, prior: bool) -> dict:
    result = defaultdict(lambda: [0.0, 0.0, 0])
    lower, upper = (di - 20, di - 1) if prior else (di - 19, di)
    for r in rows:
        rd = r["shadow_day"]
        if rd is None or not lower <= didx[rd] <= upper:
            continue
        held = max(didx[rd] - didx[r["day0"]], 0.15)
        s = result[cell_of(r["strm"], r["eb"])]
        s[0] += r["shadow_bp"]
        s[1] += held
        s[2] += 1
    return dict(result)


def replay(rows: list[dict], name: str, cap: float, *, ev: bool, gate: bool,
           prior: bool, stream: str | None = None, reverse_ties: bool = False,
           causal_slot: bool = False, strict_release: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    calendar = days()
    didx = {d: i for i, d in enumerate(calendar)}
    dec = EntryDecider(cap)
    by_day = {d: ordered_candidates([r for r in rows if r["day0"] == d], reverse_ties) for d in calendar}
    opened = []
    last_rejected = 0.0
    daily, trace = [], []
    for di, day in enumerate(calendar):
        dec.cell_sum = table_snapshot(rows, di, didx, prior)
        dec.start_day(last_rejected)
        last_rejected = 0.0
        live_carry = [r for r in opened if r["live_day"] != day]
        releasing = [r for r in opened if r["live_day"] == day]
        gross = sum(r["ntl"] for r in live_carry) + sum(r["ntl"] for r in releasing)
        release = sorted((r["live_te"], r["ntl"]) for r in releasing)
        ri = 0
        pending = []
        pnl = sum(r["live_bp"] * 1e-4 * r["ntl"] for r in opened if r["live_day"] == day)
        expected = gross if causal_slot else sum(r["ntl"] for r in live_carry)
        opened = live_carry
        fills = rejected = sd = 0
        flow = 0.0
        gross_peak = gross
        for r in by_day[day]:
            t = r["t0"]
            while ri < len(release) and (release[ri][0] < t if strict_release else release[ri][0] <= t):
                _, value = release[ri]
                gross -= value
                if causal_slot:
                    expected -= value
                ri += 1
            to_release = [p for p in pending if p[0] < t or (not strict_release and p[0] == t)]
            for _, value, reserved in to_release:
                gross -= value
                if causal_slot:
                    expected -= reserved
            pending = [p for p in pending if p[0] >= t] if strict_release else [p for p in pending if p[0] > t]
            decision = dec.decide(r["strm"], t, r["eu"], r["eb"], r["ntl"], gross, expected)
            if stream is not None and r["strm"] != stream:
                reason = "stream"
            elif ev and decision.est_bp < dec.lam * 0.6:
                reason = "ev"
            elif gate and not cell_gate(*dec.cell_sum.get(cell_of(r["strm"], r["eb"]), [0.0, 0.0, 0])):
                reason = "cell"
            elif gross + r["ntl"] > cap:
                reason = "cap"
            else:
                reason = "ok"
            if ev and gate and stream is None:
                assert reason == decision.reason
            trace.append({"variant": name, "cid": r["cid"], "day": day, "t": t, "vc": r["vc"],
                "strm": r["strm"], "reason": reason, "p_sd": decision.p_sd,
                "lam": dec.lam, "est": decision.est_bp, "slot_free": decision.slot_free,
                "gross_before": gross, "expected_overnight": expected, "ntl": r["ntl"]})
            if reason != "ok":
                if reason == "cap":
                    rejected += 1
                    last_rejected += max(decision.est_bp, 0.0) * 1e-4 * r["ntl"]
                continue
            expected_piece = r["ntl"] * (1 - decision.p_sd)
            expected += expected_piece
            fills += 1
            flow += r["ntl"]
            gross += r["ntl"]
            gross_peak = max(gross_peak, gross)
            if r["live_day"] == day:
                sd += 1
                pnl += r["live_bp"] * 1e-4 * r["ntl"]
                pending.append((r["live_te"], r["ntl"], expected_piece))
            else:
                opened.append(r)
        daily.append({"day": day, "fills": fills, "cap_rej": rejected, "sd": sd, "flow_M": flow / 1e6,
                      "twd": pnl, "carry_M": sum(r["ntl"] for r in opened) / 1e6,
                      "gross_peak_M": gross_peak / 1e6})
        for r in by_day[day]:
            dec.update_psd(r["strm"], r["t0"], r["live_type"] == "sd")
    return pl.from_dicts(daily), pl.from_dicts(trace)


def independent_capacity_audit(trace: pl.DataFrame, rows: dict, cap: float) -> dict:
    """Independently sweep actual accepted-position opens/closes, including carry."""
    calendar = days()
    didx = {d: i for i, d in enumerate(calendar)}
    events = []
    accepted = trace.filter(pl.col("reason") == "ok")["cid"].to_list()
    for order, cid in enumerate(accepted):
        r = rows[cid]
        events.append((didx[r["day0"]], r["t0"], 1, order, r["ntl"], cid))
        if r["live_day"] is not None:
            events.append((didx[r["live_day"]], r["live_te"], 0, order, -r["ntl"], cid))
    events.sort()
    balance = peak = product_peak = 0.0
    negatives = violations = 0
    products = defaultdict(float)
    active = set()
    invalid_close = 0
    for _, _, phase, _, delta, cid in events:
        r = rows[cid]
        if phase == 0:
            if cid not in active:
                invalid_close += 1
            active.discard(cid)
        else:
            active.add(cid)
        balance += delta
        products[r["vc"]] += delta
        peak = max(peak, balance)
        product_peak = max(product_peak, products[r["vc"]])
        negatives += int(balance < -1e-5)
        violations += int(balance > cap + 1e-5)
    return {"accepted": len(accepted), "peak_M": peak / 1e6, "product_peak_M": product_peak / 1e6,
            "cap_violations": violations, "negative_balance_events": negatives,
            "close_before_open": invalid_close, "final_open": len(active), "final_balance_M": balance / 1e6}


def compare_snapshot(actual: pl.DataFrame, path) -> dict:
    saved = pl.read_csv(path, schema_overrides={"day": pl.String})
    joined = actual.join(saved, on="day", suffix="_saved")
    differences = {c: float((joined[c] - joined[c + "_saved"]).abs().max())
                   for c in ("fills", "cap_rej", "sd", "flow_M", "twd", "carry_M")}
    return {"days": joined.height, "max_abs_differences": differences,
            "matched": all(v < 1e-5 for v in differences.values())}


def main() -> None:
    frame = pl.read_parquet(OUT / "resolved_candidates.parquet")
    rows = frame.to_dicts()
    from audit_common import WF
    raw = pl.read_parquet(WF / "august_attribution_s0_20260824_v2/raw_order_facts.parquet").with_row_index("source_order")
    raw = raw.filter(pl.col("outcome_supported") & pl.col("approximate_fill_time_ns").is_not_null())
    mapping = {}
    for r in raw.to_dicts():
        t = (r["approximate_fill_time_ns"] % 86400_000_000_000 - 3600_000_000_000) // 1_000_000_000
        mapping.setdefault((r["Date"], r["ValueCode"], t), []).append(r)
    cs = pl.read_parquet(WF / "daily/Date=20260813/causal_fair.parquet", columns=["ValueCode", "contract_size"]).drop_nulls().unique()
    sizes = dict(cs.iter_rows())
    for r in rows:
        if r["strm"] == "S1":
            matches = [x for x in mapping[(r["day0"], r["vc"], r["t0"])]
                       if abs(x["target_price"] * sizes.get(r["vc"], 2000.0) - r["ntl"]) < 1e-5]
            assert matches
            r["source_order"] = matches[0]["source_order"]
    keyed = {r["cid"]: r for r in rows}
    configs = [
        ("v18_reference_20", 20e6, dict(ev=True, gate=True, prior=False)),
        ("v18_reference_50", 50e6, dict(ev=True, gate=True, prior=False)),
        ("v18_reference_100", 100e6, dict(ev=True, gate=True, prior=False)),
        ("v17_reference_20", 20e6, dict(ev=False, gate=True, prior=False)),
        ("v18_prior20", 20e6, dict(ev=True, gate=True, prior=True)),
        ("v18_prior20_causal_slot", 20e6, dict(ev=True, gate=True, prior=True, causal_slot=True)),
        ("v17_prior20", 20e6, dict(ev=False, gate=True, prior=True)),
        ("fcfs", 20e6, dict(ev=False, gate=False, prior=True)),
        ("S1_only", 20e6, dict(ev=False, gate=False, prior=True, stream="S1")),
        ("S2_only", 20e6, dict(ev=False, gate=False, prior=True, stream="S2")),
        ("v18_reverse_ties", 20e6, dict(ev=True, gate=True, prior=False, reverse_ties=True)),
        ("v18_strict_release", 20e6, dict(ev=True, gate=True, prior=False, strict_release=True)),
    ]
    reports, month_rows = {}, []
    for name, cap, kwargs in configs:
        daily, trace = replay(rows, name, cap, **kwargs)
        daily.write_csv(OUT / f"{name}_daily.csv")
        trace.write_parquet(OUT / f"{name}_trace.parquet")
        report = {"pnl_mean": daily["twd"].mean(), "pnl_sum": daily["twd"].sum(),
                  "carry_mean_M": daily["carry_M"].mean(),
                  "capacity_audit": independent_capacity_audit(trace, keyed, cap)}
        if "reference" in name:
            snap = name.replace("_reference", "") + "_daily.csv"
            report["snapshot_comparison"] = compare_snapshot(daily, OUT / snap)
        for r in daily.with_columns(pl.col("day").str.slice(0, 6).alias("month")).group_by("month").agg(
                pl.col("twd").mean(), pl.len()).sort("month").to_dicts():
            month_rows.append({"variant": name, **r})
        reports[name] = report
        print(name, json.dumps(report), flush=True)
    (OUT / "allocation_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    pl.from_dicts(month_rows).write_csv(OUT / "allocation_monthly.csv")


if __name__ == "__main__":
    main()
