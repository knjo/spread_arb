"""Independent quote-buffer arithmetic, inside priority, raw depth and ledger audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ..execution import price_i
from ..market import BookSeries, _raw
from ..verify_full_study import read_rows
from ..verify_v21 import verify_model


def tick(px):
    return 100 if px < 100000 else 500 if px < 500000 else 1000 if px < 1000000 else 5000 if px < 5000000 else 10000 if px < 10000000 else 50000


def check(root, raw_days):
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("planned_days", manifest["days"]) != manifest["days"]:
        raise AssertionError("The planned full continuation has not finished")
    base = verify_model(root, raw_days)
    config = manifest["configurations"][0]
    rule = config["liquidity_rule"]
    counts = defaultdict(int)
    for day in manifest["days"]:
        folders = [root/f"Date={day}"/name for name in ("shadow", config["name"])]
        raw_samples = []
        for folder in folders:
            sample_limit = len(raw_samples)+1250
            trace = read_rows(folder/"execution.parquet")
            taken = defaultdict(list)
            for t in trace:
                if t["kind"] == "taker_fill" and t.get("purpose") == "entry_spot":
                    taken[t["vc"], t["book_ns"]].append(t["ns"])
            for d in read_rows(folder/"decisions.parquet"):
                if d["stream"] != "S2":
                    if abs(d.get("execution_cost_bp", 0.0)) > 1e-10:
                        raise AssertionError("S1 received S2 execution buffer")
                    continue
                counts["s2_decisions"] += 1
                quote, bid, ask = map(price_i, (d["quote_price"], d["future_bid"], d["future_ask"]))
                if quote != ask-tick(ask-1) or not bid < quote < ask or d["future_book_ns"] > d["ns"]:
                    raise AssertionError("new S2 quote did not improve current A1 by one tick")
                if not rule["enabled"]:
                    if abs(d["execution_cost_bp"]) > 1e-10:
                        raise AssertionError("inside control received a liquidity buffer")
                    continue
                if d["liquidity_book_ns"] > d["ns"]:
                    raise AssertionError("future spot depth used at quote")
                if d["liquidity_a1_shares"] < rule["min_a1_multiple"]*d["liquidity_hedge_shares"]:
                    raise AssertionError("quote passed insufficient A1 coverage")
                if abs(d["liquidity_a1_multiple"]-d["liquidity_a1_shares"]/d["liquidity_hedge_shares"]) > 1e-10:
                    raise AssertionError("incorrect shares/contract hedge ratio")
                raw = (d["quote_price"]/d["liquidity_a1_price"]-1)*10000
                full = (d["quote_price"]/d["liquidity_depth_vwap"]-1)*10000
                adverse = (d["quote_price"]/d["liquidity_adverse_vwap"]-1)*10000
                wanted = raw-((1-rule["adverse_probability"])*full+rule["adverse_probability"]*adverse)
                if abs(d["execution_cost_bp"]-wanted) > 1e-7:
                    raise AssertionError("incorrect fixed-scenario buffer")
                counts["buffer_checks"] += 1
                # Quotes with no prior internal consumption of this snapshot
                # have a direct independent raw-depth reconstruction. Samples
                # with shared consumption still receive the complete algebra,
                # capacity and four-leg checks above and in verify_model.
                if (day in raw_days and len(raw_samples) < sample_limit
                        and not any(ns <= d["ns"] for ns in taken[d["vc"], d["liquidity_book_ns"]])):
                    raw_samples.append(d)
            counts["depth_rejections"] += sum(t["kind"] == "s2_liquidity_reject" for t in trace)
            counts["quantity_cancels"] += sum(t["kind"] == "event_cancel" and t["reason"] == "a1_depth" for t in trace)
        if raw_samples:
            from ..causal_lookup import open_ns, CLOSE_SECOND, SECOND
            paths = json.loads((root/f"Date={day}"/"inputs.json").read_text())
            # The spot raw path is also recorded by the standard raw-fill audit.
            from ...common.paths import spot_tick_path, resolve_input_file
            path = resolve_input_file(spot_tick_path(day), role="spot_raw")
            if str(path) not in paths:
                raise AssertionError("raw quote audit input differs from replay input")
            raw = _raw(path, sorted({d["vc"] for d in raw_samples}), future=False,
                       start=open_ns(day), end=open_ns(day)+CLOSE_SECOND*SECOND)
            series = {g.item(0, "QuoteCode"):BookSeries("S:"+g.item(0, "QuoteCode"), g)
                      for g in raw.partition_by("QuoteCode", maintain_order=True)}
            for d in raw_samples:
                book = series[d["vc"]].at(d["ns"])
                if book.ns != d["liquidity_book_ns"] or book.sequence != d["liquidity_book_sequence"]:
                    raise AssertionError("recorded liquidity is not the raw as-of snapshot")
                if book.asks[0][1] != d["liquidity_a1_shares"]:
                    raise AssertionError("A1 quantity differs from raw depth")
                left, cash, stressed_cash = d["liquidity_hedge_shares"], 0, 0
                for px, qty in book.asks:
                    take = min(left, qty)
                    stress = px
                    for _ in range(rule["adverse_ticks"]):
                        stress += tick(stress)
                    cash += px*take
                    stressed_cash += stress*take
                    left -= take
                    if not left:
                        break
                if left:
                    raise AssertionError("quote falsely claimed full-quantity raw depth")
                scale = d["liquidity_hedge_shares"]*10000
                if (abs(cash/scale-d["liquidity_depth_vwap"]) > 1e-8
                        or abs(stressed_cash/scale-d["liquidity_adverse_vwap"]) > 1e-8):
                    raise AssertionError("quote VWAP is not reconstructed from raw levels")
                counts["raw_liquidity_quotes"] += 1
    result = dict(passed=True, counts=dict(counts), base=base)
    (root/"verification_liquidity.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--raw-days", nargs="*", default=[])
    a = p.parse_args()
    print(json.dumps(check(a.root, set(a.raw_days)), indent=2))
