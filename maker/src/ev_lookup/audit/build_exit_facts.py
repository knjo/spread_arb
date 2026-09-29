"""Resolve archived candidate entries with the frozen v18 mexit implementation.

This deliberately preserves v18 execution assumptions. It is an allocation audit,
not a replacement execution backtest. No candidate_dump terminal label is used.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from audit_common import OUT, candidates, days, day_books, next_exp, source_function, source_exit_detail


def build() -> None:
    calendar = days()
    frame = candidates()
    records = {r["cid"]: r | {"exp": next_exp(r["day0"])} for r in frame.to_dicts()}
    by_day = {d: [r for r in records.values() if r["day0"] == d] for d in calendar}
    active = []
    outcomes = {}
    details = []
    for di, day in enumerate(calendar):
        before = time.monotonic()
        today = by_day[day]
        check = [r for r in active if day <= r["exp"]] + today
        products = sorted({r["vc"] for r in check})
        books, mfa = day_books(day, products)
        mexit = source_function("mexit", {"np": np, "books": books, "mfa": mfa, "RC": 15600})
        keep = []
        for row in active + today:
            cid = row["cid"]
            if day > row["exp"]:
                out = outcomes.setdefault(cid, {})
                out.update(live_day=day, live_te=0, live_type="expiry", live_bp=row["eb"] - 34.0)
                if "shadow_day" not in out:
                    out.update(shadow_day=day, shadow_bp=row["eb"] - 34.0)
                continue
            start = row["t0"] + 1 if row["day0"] == day else 0
            detail = source_exit_detail(row, start, books, mfa, mexit)
            details.append({"cid": cid, "day": day, "vc": row["vc"], **detail})
            te = detail["te"]
            if te is None:
                keep.append(row)
                continue
            sd = row["day0"] == day
            pnl = row["eu"] + 5.0 - (20.0 if sd else 34.0)
            out = outcomes.setdefault(cid, {})
            out.update(shadow_day=day, shadow_bp=pnl)
            if day == row["exp"]:
                keep.append(row)
            else:
                out.update(live_day=day, live_te=te, live_type="sd" if sd else "maker34", live_bp=pnl)
        active = keep
        print(day, "candidates", len(today), "carry", len(active), "seconds", round(time.monotonic() - before, 2), flush=True)
    rows = []
    for cid, row in records.items():
        result = row | {"live_day": None, "live_te": None, "live_type": "open", "live_bp": None,
                        "shadow_day": None, "shadow_bp": None} | outcomes.get(cid, {})
        rows.append(result)
    pl.from_dicts(rows, infer_schema_length=None).write_parquet(OUT / "resolved_candidates.parquet")
    pl.from_dicts(details, infer_schema_length=None).write_parquet(OUT / "exit_details.parquet")
    (OUT / "exit_build_summary.json").write_text(json.dumps({"candidates": len(rows), "days": len(calendar),
        "open_final": len(active), "exit_evaluations": len(details)}, indent=2) + "\n")


if __name__ == "__main__":
    build()
