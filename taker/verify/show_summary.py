"""把 summary CSV 用表格顯示，並加一欄『不留倉淨利(直接加總收斂事件真實pnl)』。

「不留倉」欄改用正確算法：直接 Σ(收斂事件的真實 potential_pnl)，
不是「總淨利 × 收斂率」(那是錯的近似：假設收斂/不收斂平均淨利相同，但大單傾向不收斂故偏差)。
執行：uv run python verify/show_summary.py
"""
import os
import sys
import glob

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report_first as stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "out", "stats")
OUT = os.path.join(ROOT, "out")


def converged_pnl_by_thr(exit_type="B", mode="2"):
    """從事實 CSV 直接加總每閾值『收斂事件』的真實 pnl（億）。"""
    frames = []
    for f in sorted(glob.glob(os.path.join(OUT, "events_*_tick.csv"))):
        if "_m" in os.path.basename(f):
            continue
        df = pl.read_csv(f)
        if "entry_fut_bid" in df.columns:
            frames.append(df)
    ev = pl.concat(frames, how="diagonal")
    sub = ev.filter((pl.col("exit_type") == exit_type) & (pl.col("potential_lots") > 0))
    sub = stats.enrich(sub, mode=mode)
    sub = sub.filter(pl.col("converged"))   # 只加收斂(真平掉)的
    return {round(t, 4): v / 1e8 for t, v in
            sub.group_by("threshold").agg(pl.col("potential_pnl").sum()).rows()}


for path in sorted(glob.glob(os.path.join(STATS, "summary_*.csv"))):
    name = os.path.basename(path)
    df = pl.read_csv(path)
    # 從檔名推 exit_type / mode（summary_m2B_tick → mode2, B）
    et = "B" if "B" in name else "A"
    md = "2" if "m2" in name else "1"
    cpnl = converged_pnl_by_thr(et, md)
    # 對每列(閾值)填入該閾值的收斂直接加總(三口徑共用同一閾值值；本表只有全量列為主)
    df = df.with_columns(
        pl.col("threshold").map_elements(
            lambda t: round(cpnl.get(round(t, 4), 0.0), 3), return_dtype=pl.Float64
        ).alias("不留倉淨利_億")
    )
    print(f"\n===== {name}（不留倉淨利 = 直接加總收斂事件真實pnl）=====")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200,
                   tbl_hide_dataframe_shape=True):
        print(df)
