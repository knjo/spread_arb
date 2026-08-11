"""pivot 輸出（METHODOLOGY.md 步驟 10）。

對「多日累積的事件指標表」(含 net_ret) 做彙整：
  列 = 結算日距離分組、欄 = 價差閥值，
  每格 = {發生次數, 平均潛在部位, 平均淨獲利率, 日內收斂率, 平均最大發散}。
折價(價差買)與溢價(價差賣)各出一張表（由呼叫端分別帶入該方向的事件表）。
"""
from __future__ import annotations

import polars as pl

# 結算日距離分組（天）
DEFAULT_SETTLE_BINS = [0, 3, 7, 14, 21, 9999]
DEFAULT_SETTLE_LABELS = ["0-2", "3-6", "7-13", "14-20", "21+"]


def summarize(events: pl.DataFrame,
              settle_bins=None, settle_labels=None) -> pl.DataFrame:
    """單一閥值、單方向的事件表 → 依結算日距離分組彙整每格指標。

    events 需含：days_to_settle, potential_value, net_ret, converged, diverge_max。
    回傳每個結算日距離分組一列。
    """
    bins = settle_bins or DEFAULT_SETTLE_BINS
    labels = settle_labels or DEFAULT_SETTLE_LABELS

    g = events.with_columns(
        # left_closed=True：days=3 落 "3-6"、7 落 "7-13"、14 落 "14-20"、21 落 "21+"
        # （預設右閉會 off-by-one：3 落 "0-2" 等，審計 V03-1）
        pl.col("days_to_settle").cut(bins[1:-1], labels=labels, left_closed=True)
        .alias("settle_bucket")
    )
    # 只打第一檔(量=min兩腳第一檔)，不吃穿、無滑價，金額即實際可做到的數字。
    # 單一閥值內加總乾淨(各筆獨立不重複)；「不同閥值之間」不可相加(同一波會重複)。
    return (g.group_by("settle_bucket")
            .agg([
                pl.len().alias("發生次數"),
                pl.col("net_ret").mean().alias("平均淨獲利率"),
                pl.col("converged").mean().alias("日內收斂率"),
                pl.col("diverge_max").mean().alias("平均最大發散"),
                pl.col("potential_value").mean().alias("平均部位"),
                pl.col("potential_pnl").mean().alias("平均獲利"),
                pl.col("potential_pnl").sum().alias("總獲利"),
            ])
            .sort("settle_bucket"))


def pivot_by_threshold(events_by_thr: dict[float, pl.DataFrame]) -> pl.DataFrame:
    """多閥值彙整成一張 pivot：列=結算日距離分組、欄=各閥值的指標。

    events_by_thr: {閥值: 該閥值的事件指標表(含 net_ret 等)}。
    每個閥值算 summarize 後，欄位前綴閥值，再依 settle_bucket 對齊合併。
    """
    out = None
    for thr in sorted(events_by_thr):
        s = summarize(events_by_thr[thr])
        s = s.rename({c: f"{thr:.1%}_{c}" for c in s.columns if c != "settle_bucket"})
        out = s if out is None else out.join(s, on="settle_bucket", how="full", coalesce=True)
    return out.sort("settle_bucket") if out is not None else pl.DataFrame()
