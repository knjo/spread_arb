"""Read-only inputs and frozen-source helpers for the 2026-09-08 audit."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl

MAKER = Path(__file__).resolve().parents[3]
OUT = MAKER / "data/ev_lookup_audit_20260908"
WF = MAKER / "data/walkforward"
SNAPSHOT = OUT / "source_snapshot"
EXT = Path("/tmp/claude-1000/-home-kevin-Project-HFT/41972a34-5d7e-4f48-b275-3f9588a394bc/scratchpad/ext_daily")
EXPIRY = ("20260520", "20260617", "20260715", "20260819", "20260916", "20261021")


def candidates() -> pl.DataFrame:
    return pl.read_csv(
        OUT / "v15_candidates.csv",
        schema_overrides={"day0": pl.String, "res_day": pl.String, "vc": pl.String},
    ).with_row_index("cid")


def days() -> list[str]:
    return pl.read_csv(OUT / "v18_20_daily.csv", schema_overrides={"day": pl.String})["day"].to_list()


def next_exp(day: str) -> str:
    return next((d for d in EXPIRY if d > day), "20261118")


def source_function(name: str, namespace: dict) -> object:
    path = SNAPSHOT / "stacked_walkforward_backtest.py"
    module = ast.parse(path.read_text())
    node = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name)
    fragment = ast.Module(body=[node], type_ignores=[])
    exec(compile(fragment, str(path), "exec"), namespace)
    return namespace[name]


def day_books(day: str, products: list[str]) -> tuple[dict, dict]:
    path = WF / f"daily/Date={day}/causal_fair.parquet"
    if not path.exists():
        path = EXT / f"{day}.parquet"
    schema = pl.read_parquet_schema(path)
    cols = ["ValueCode", "seconds_from_open", "basis_buy_taker_bp", "spot_sequence", "spot_ask", "fut_ask"]
    if "fut_exec_ask" in schema:
        cols.append("fut_exec_ask")
    frame = pl.scan_parquet(path).filter(pl.col("ValueCode").is_in(products)).select(cols).collect()
    frame = frame.filter(pl.col("seconds_from_open") <= 15600)
    books = {}
    for group in frame.partition_by("ValueCode", maintain_order=True):
        code = group.item(0, "ValueCode")
        assert group["seconds_from_open"].to_list() == list(range(group.height)), (day, code, "non-contiguous grid")
        books[code] = {}
        for col in cols[2:]:
            series = group[col]
            if series.dtype.is_float():
                series = series.fill_nan(None)
            books[code][col] = series.forward_fill().to_numpy()
    mk = pl.scan_parquet(f"/media/kevin/SSD2/Data/makerFill/{day}_makerFill.parquet").filter(
        pl.col("QuoteCode").is_in(products)
    ).select("QuoteCode", "ChannelSeq", "Ask1_FillSeconds").collect()
    mfa = {}
    for group in mk.partition_by("QuoteCode", maintain_order=True):
        code = group.item(0, "QuoteCode")
        seq = group["ChannelSeq"].to_numpy()
        assert np.all(seq[1:] >= seq[:-1]), (day, code, "unsorted makerFill")
        mfa[code] = (seq, group["Ask1_FillSeconds"].to_numpy())
    return books, mfa


def source_exit_detail(row: dict, start: int, books: dict, mfa: dict, mexit) -> dict:
    vc, target = row["vc"], row["eb"] - row["eu"] - 5.0
    te = mexit(vc, target, start)
    detail = {"te": te, "first_touch": None, "chosen_touch": None, "skipped_nan_touches": 0,
              "maker_sequence": None, "label_seconds": None, "spot_exit_ask": None,
              "future_ask_te": None, "future_ask_next_second": None}
    if vc not in books:
        return detail
    b = books[vc]
    touches = np.flatnonzero(b["basis_buy_taker_bp"][start:] <= target) + start
    if len(touches) == 0:
        return detail
    detail["first_touch"] = int(touches[0])
    if vc not in mfa:
        return detail
    sq, fs = mfa[vc]
    js = np.searchsorted(sq, b["spot_sequence"][touches], side="right") - 1
    valid = js >= 0
    valid[valid] &= ~np.isnan(fs[js[valid]])
    if not valid.any():
        return detail
    j = int(np.flatnonzero(valid)[0])
    chosen = int(touches[j])
    detail.update(chosen_touch=chosen, skipped_nan_touches=j,
                  maker_sequence=int(sq[js[j]]), label_seconds=float(fs[js[j]]),
                  spot_exit_ask=float(b["spot_ask"][chosen]))
    expected = chosen + int(fs[js[j]]) + 1
    assert te == (expected if expected <= 15480 else None)
    if te is not None:
        fa = b.get("fut_exec_ask", b["fut_ask"])
        detail["future_ask_te"] = float(fa[te])
        detail["future_ask_next_second"] = float(fa[min(te + 1, len(fa) - 1)])
    return detail
