"""Bounded raw-tape checks for S2 timing, contract identity and exit volume."""
from __future__ import annotations

import ast
import json
from collections import defaultdict

import numpy as np
import polars as pl

from audit_common import OUT, WF, SNAPSHOT, candidates, source_function


def synthetic_ordering_probe() -> dict:
    source = ast.parse((SNAPSHOT / "stacked_walkforward_backtest.py").read_text())
    loop = next(n for n in ast.walk(source) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == "vc" and isinstance(n.iter, ast.Name) and n.iter.id == "univ")
    namespace = {"np": np, "univ": ["TEST"], "CS": {"TEST": 2000.0}, "fills": [],
        "CUT": 305, "U": 25.0, "FLOOR": 20.0,
        "books": {"TEST": {"spot_ask": np.full(400, 50.0), "fut_bid": np.full(400, 50.05),
            "fut_ask": np.full(400, 50.3), "anchor_ewma_120s_bp": np.zeros(400)}},
        "trades": {"TEST": (np.array([301.9]), np.array([50.2]))}}
    source_function("tick_of", namespace)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "frozen_s2_loop", "exec"), namespace)
    assert len(namespace["fills"]) == 1
    recorded = namespace["fills"][0][0]
    assert recorded == 301
    return {"s2_recorded_second": recorded, "s2_actual_print_second": 301.9,
        "earlier_s1_actual_second": 301.1, "source_same_second_winner": "S2",
        "actual_first_fill": "S1", "advanced_seconds": 301.9 - recorded}


def raw_s2_day(day: str) -> pl.DataFrame:
    base = pl.read_parquet(WF / "daily/Date=20260813/causal_fair.parquet",
        columns=["ValueCode", "QuoteCode", "contract_size"]).drop_nulls().unique()
    families = {r["ValueCode"]: r["QuoteCode"][:3] for r in base.to_dicts()}
    sizes = {r["ValueCode"]: r["contract_size"] for r in base.to_dicts()}
    cols = ["ValueCode", "seconds_from_open", "spot_ask", "fut_bid", "fut_ask", "anchor_ewma_120s_bp", "QuoteCode"]
    cf = pl.read_parquet(WF / f"daily/Date={day}/causal_fair.parquet", columns=cols)
    books = {}
    for group in cf.partition_by("ValueCode", maintain_order=True):
        vc = group.item(0, "ValueCode")
        if vc not in families:
            continue
        b = {}
        for col in cols[2:]:
            series = group[col]
            if series.dtype.is_float():
                series = series.fill_nan(None)
            b[col] = series.forward_fill().to_numpy()
        books[vc] = b
    fu = pl.scan_parquet(f"/mnt/NAS/Parquet/Ticks/2026/{day[4:6]}/{day[6:8]}/stock_futures.parquet").filter(
        (pl.col("TrialMatch") == 0) & pl.col("ValueCode").is_in(list(books))
    ).select("ValueCode", "QuoteCode", "RecvTime", "FillPrice", "FillLots", "DecimalLocator", "TotalFillLots").collect()
    fu = fu.with_columns(pl.col("ValueCode").replace_strict(families).alias("family")).filter(
        pl.col("QuoteCode").str.slice(0, 3) == pl.col("family"))
    volumes = fu.group_by("ValueCode", "QuoteCode").agg(pl.col("TotalFillLots").max()).sort("TotalFillLots")
    front = dict(volumes.group_by("ValueCode", maintain_order=True).last().select("ValueCode", "QuoteCode").iter_rows())
    open_ns = int(np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}T01:00:00", "ns").astype(np.int64))
    fu = fu.filter(pl.col("FillLots") > 0).with_columns(
        ((pl.col("RecvTime").cast(pl.Int64) - open_ns) / 1e9).alias("sec"),
        (pl.col("FillPrice") * pl.lit(10.0).pow(-pl.col("DecimalLocator").cast(pl.Float64))).alias("px"))
    trades = {}
    for group in fu.partition_by(["ValueCode", "QuoteCode"], maintain_order=True):
        ts = group["sec"].to_numpy()
        assert np.all(ts[1:] >= ts[:-1])
        trades[(group.item(0, "ValueCode"), group.item(0, "QuoteCode"))] = (ts, group["px"].to_numpy())
    tick = source_function("tick_of", {})
    rows = []
    for vc, b in books.items():
        selected = front[vc]
        if (vc, selected) not in trades:
            continue
        ts, px = trades[(vc, selected)]
        sa, fb, fa, anchor = b["spot_ask"], b["fut_bid"], b["fut_ask"], b["anchor_ewma_120s_bp"]
        held, pt, t, submit_ab, submit_eu = np.nan, 0, 300, np.nan, np.nan
        while t < 14000:
            valid = not (np.isnan(fa[t]) or np.isnan(fb[t]) or np.isnan(sa[t]) or np.isnan(anchor[t]) or sa[t] <= 0 or fb[t] <= 0)
            cond = False
            if valid:
                desired = round(fa[t] - tick(fa[t]), 4)
                desired_ab = (desired / sa[t] - 1) * 1e4
                cond = desired > fb[t] + 1e-9 and desired_ab - anchor[t] >= 25 and desired_ab > 0
            if np.isnan(held):
                if cond:
                    held, pt, submit_ab, submit_eu = desired, t, desired_ab, desired_ab - anchor[t]
                t += 1
                continue
            crossed = valid and held <= fb[t] + 1e-9
            lo, hi = np.searchsorted(ts, max(pt + 1, t), side="right"), np.searchsorted(ts, t + 1, side="right")
            matched = np.flatnonzero(px[lo:hi] >= held - 1e-9)
            if crossed or len(matched):
                actual_print = None if crossed else float(ts[lo + matched[0]])
                ab = (held / sa[t] - 1) * 1e4 if sa[t] > 0 else np.nan
                model_contract = b["QuoteCode"][t]
                mts, mpx = trades.get((vc, model_contract), (np.array([]), np.array([])))
                ml, mh = np.searchsorted(mts, max(pt + 1, t), side="right"), np.searchsorted(mts, t + 1, side="right")
                matching_contract_print = bool(np.any(mpx[ml:mh] >= held - 1e-9))
                rows.append({"day": day, "vc": vc, "t": t, "quote_t": pt, "held_px": held,
                    "ab": float(ab), "eu": float(ab - anchor[t]), "ntl": float(sa[t] * sizes.get(vc, 2000.0)),
                    "submit_ab": float(submit_ab), "submit_eu": float(submit_eu),
                    "source_kept": bool(not np.isnan(ab) and ab > 0), "trigger": "book_cross" if crossed else "print",
                    "actual_print_sec": actual_print, "model_contract": model_contract,
                    "print_contract": selected, "matching_contract_print": matching_contract_print})
                held = np.nan
                t += 60
                continue
            held_ab = (held / sa[t] - 1) * 1e4 if valid else np.nan
            if np.isnan(held_ab) or held_ab - anchor[t] < 20 or held_ab <= 0:
                if cond:
                    held, pt, submit_ab, submit_eu = desired, t, desired_ab, desired_ab - anchor[t]
                else:
                    held = np.nan
            t += 1
    result = pl.from_dicts(rows, infer_schema_length=None)
    expected = candidates().filter((pl.col("day0") == day) & (pl.col("strm") == "S2"))
    kept = result.filter(pl.col("source_kept"))
    compare = kept.join(expected, left_on=["vc", "t"], right_on=["vc", "t0"], suffix="_saved")
    assert kept.height == expected.height == compare.height
    assert (compare["ab"] - compare["eb"]).abs().max() < 1e-7
    result.write_parquet(OUT / f"raw_s2_{day}.parquet")
    return result


def exit_volume_probe() -> dict:
    raw = pl.read_parquet(OUT / "sample_8039_20260724_ticks.parquet")
    row = raw.filter(pl.col("ChannelSeq") == 1986698).row(0, named=True)
    end = np.datetime64("2026-07-24T01:12:47", "us")
    window = raw.filter((pl.col("ChannelSeq") > 1986698) & (pl.col("RecvTime") <= pl.lit(end)) &
                        (pl.col("FillPrice") >= row["AskPrice1"] - 1e-8) & (pl.col("FillLots") > 0))
    return {"day": "20260724", "product": "8039", "label_sequence": 1986698,
            "source_ask": row["AskPrice1"], "source_displayed_lots": row["AskLots1"],
            "paper_exit_future_units": 20, "paper_exit_spot_lots": 40,
            "all_observed_ge_ask_trade_lots_until_paper_exit": window["FillLots"].sum(),
            "trade_rows": window.height}


def main() -> None:
    summary = {"synthetic_ordering": synthetic_ordering_probe(), "exit_volume": exit_volume_probe(), "raw_samples": {}}
    for day in ("20260520", "20260706", "20260724"):
        result = raw_s2_day(day)
        prints = result.filter(pl.col("trigger") == "print")
        mismatched = prints.filter(pl.col("model_contract") != pl.col("print_contract"))
        summary["raw_samples"][day] = {"total_potential_fills": result.height,
            "kept": result["source_kept"].sum(), "discarded_post_fill": result.filter(~pl.col("source_kept")).height,
            "print_fills": prints.height, "mismatched_print_contract": mismatched.height,
            "mismatched_without_matching_quote_contract_print": mismatched.filter(~pl.col("matching_contract_print")).height,
            "median_advanced_seconds": prints.select((pl.col("actual_print_sec") - pl.col("t")).median()).item()}
        print(day, summary["raw_samples"][day], flush=True)
    (OUT / "execution_probes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
