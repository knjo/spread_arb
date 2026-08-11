"""第3層：價差計算 + 事件狀態機（METHODOLOGY.md 步驟 5-6）。

價差（taker-taker）:
  價差買 = 賣現 B1、買期(取優賣)  → 期貨折價賺   spread_buy  = spot_bid - fut_ask
  價差賣 = 買現 A1、賣期(取優買)  → 期貨溢價賺   spread_sell = fut_bid  - spot_ask
獲利率(毛) 分母統一用「期 A1」= fut_ask（取優後的賣價）。

事件（一波）: 獲利率(毛) >= 閥值 持續算同一次事件，< 閥值 重置。每事件取第一筆。
"""
from __future__ import annotations

import polars as pl

from .preprocess import TIME_COL

# 方向常數
SIDE_BUY = "價差買"    # 期貨折價：賣現買期
SIDE_SELL = "價差賣"   # 期貨溢價：買現賣期


def calc_spreads(df: pl.DataFrame) -> pl.DataFrame:
    """在 as-of join 後的 df 上算兩個方向的價差與毛獲利率。

    需要欄位：fut_ask, fut_bid, spot_ask, spot_bid（皆已還原真實價，0/null=無報價）。
    分母統一 fut_ask（期 A1，取優賣）。fut_ask 無效時獲利率為 null。
    """
    denom = pl.col("fut_ask")  # 分母統一期 A1

    return df.with_columns([
        # 價差買：賣現 B1(spot_bid) - 買期 取優賣(fut_ask)
        (pl.col("spot_bid") - pl.col("fut_ask")).alias("spread_buy"),
        # 價差賣：賣期 取優買(fut_bid) - 買現 A1(spot_ask)
        (pl.col("fut_bid") - pl.col("spot_ask")).alias("spread_sell"),
    ]).with_columns([
        pl.when((pl.col("spot_bid") > 0) & (denom > 0))
          .then((pl.col("spot_bid") - pl.col("fut_ask")) / denom)
          .otherwise(None).alias("ret_buy"),
        pl.when((pl.col("spot_ask") > 0) & (pl.col("fut_bid") > 0) & (denom > 0))
          .then((pl.col("fut_bid") - pl.col("spot_ask")) / denom)
          .otherwise(None).alias("ret_sell"),
    ])


def _ret_col(side: str) -> str:
    """進場（同邊）毛獲利率欄位：價差賣→ret_sell、價差買→ret_buy。"""
    return "ret_sell" if side == SIDE_SELL else "ret_buy"


def _opp_ret_col(side: str) -> str:
    """出場（反邊）毛獲利率欄位 —— 平倉那一刀打的方向。
    收斂定義（交易員）：進場價差賣 → 平倉走反邊「價差買」(買回期A1、賣現B1)，
    看 ret_buy 是否回到 >=0（平倉不倒貼＝收斂）。價差買對稱看 ret_sell。"""
    return "ret_buy" if side == SIDE_SELL else "ret_sell"


def tag_events(df: pl.DataFrame, side: str, threshold: float,
               exit_threshold: float = 0.0) -> pl.DataFrame:
    """標記事件：進出場用不同門檻（遲滯/hysteresis），避免在閥值邊緣抖動被切碎。

    狀態定義：
      進場  ret >= threshold              → 開啟一波事件
      持續  ret >  exit_threshold(預設0)  → 仍在同一波（價差還在就不算結束）
      結束  ret <= exit_threshold 且非null → 唯一的重置
      null  缺報價                         → 維持前一狀態，不切斷
    預設 exit_threshold=0：只要價差 ret>0（期貨仍溢價/折價）就算同一波，
    符合「只要價差還在就不算結束」+「先估寬一點」。

    新增欄：
      is_signal  該列是否達進場閥值
      in_event   是否處於事件進行中
      is_first   該列是否為一波事件的第一筆
      event_id   事件序號（同商品內，每進入一波 +1；非事件列為 null）
    需 df 已按 (ValueCode, 時間欄 TIME_COL) 排序。
    """
    # 事件分組鍵：以「合約」為主體（QuoteCode）。曾誤用 ValueCode 分組，
    # 同標的多合約(標準+小型)的報價流互相開/關事件 → 單日爆出 2 萬筆假事件。
    key = "QuoteCode" if "QuoteCode" in df.columns else "ValueCode"
    ret = pl.col(_ret_col(side))
    df = df.sort([key, TIME_COL])

    # raw 事件邊界：進場=1（開啟）、結束=-1、其餘(持續或null)=0（維持）
    # 用 0 代表「不主動改變狀態」，再靠 forward-fill 把進場/結束狀態延續下去。
    boundary = (
        pl.when(ret >= threshold).then(1)            # 進場
          .when(ret <= exit_threshold).then(-1)      # 明確結束（非 null 且 <=出場門檻）
          .otherwise(0)                              # 持續(exit<ret<entry) 或 null → 維持
    )
    # 持續/ null 列(boundary=0)沿用前一個「明確邊界」狀態
    df = df.with_columns(boundary.alias("_b"))
    df = df.with_columns(
        pl.when(pl.col("_b") != 0).then(pl.col("_b")).otherwise(None)
          .forward_fill().over(key).fill_null(-1).alias("_state")  # 預設事件外(-1)
    )

    df = df.with_columns([
        (ret >= threshold).fill_null(False).alias("is_signal"),
        (pl.col("_state") == 1).alias("in_event"),
    ])
    prev_in = pl.col("in_event").shift(1).over(key).fill_null(False)
    df = df.with_columns(
        (pl.col("in_event") & ~prev_in).alias("is_first")
    )
    df = df.with_columns(
        pl.when(pl.col("in_event"))
          .then(pl.col("is_first").cum_sum().over(key))
          .otherwise(None).alias("event_id")
    ).drop("_b", "_state")
    return df


def first_ticks(df: pl.DataFrame, side: str, threshold: float) -> pl.DataFrame:
    """回傳每一波事件的「第一筆」（越早套越好），即事件清單。"""
    tagged = tag_events(df, side, threshold)
    return tagged.filter(pl.col("is_first"))
