"""第2層：取優價 + as-of join（METHODOLOGY.md 步驟 3-4）。

取優價：期貨買賣腳都要參考「明掛一檔」與「最佳衍生一檔」取優價。
價格 0 = 無報價（SDK 用 0 代 null），不可進 min/max，需先轉 null 排除。
"""
from __future__ import annotations

import warnings

import polars as pl

from .preprocess import TIME_COL


def _nz(col: str) -> pl.Expr:
    """把 0（無報價填充值）轉成 null，使其不參與 min/max。"""
    return pl.when(pl.col(col) > 0).then(pl.col(col)).otherwise(None)


def best_quotes(fut: pl.DataFrame) -> pl.DataFrame:
    """為期貨算出取優賣 / 取優買，及對應可成交量。

    取優賣（買期腳）= min(明掛 AskPrice1, 衍生 BestAskPrice)，排除 0。
    取優買（賣期腳）= max(明掛 BidPrice1, 衍生 BestBidPrice)，排除 0。
    對應量：取到哪一檔（明掛 or 衍生），就帶那一檔的量；同價時取量較大者。
    若兩邊都無報價 → 該腳價格 null（該筆該方向無套利）。
    """
    ask1, bask = _nz("AskPrice1"), _nz("BestAskPrice")
    bid1, bbid = _nz("BidPrice1"), _nz("BestBidPrice")

    fut = fut.with_columns([
        pl.min_horizontal(ask1, bask).alias("fut_ask"),   # 取優賣（買期成交價）
        pl.max_horizontal(bid1, bbid).alias("fut_bid"),   # 取優買（賣期成交價）
    ])

    # 對應量：取優價來自明掛或衍生，挑出對應的量；若兩邊同價取較大量
    fut = fut.with_columns([
        pl.when(pl.col("fut_ask").is_null()).then(0)
          .when((_nz("AskPrice1") == pl.col("fut_ask")) & (_nz("BestAskPrice") == pl.col("fut_ask")))
          .then(pl.max_horizontal("AskLots1", "BestAskLots"))
          .when(_nz("AskPrice1") == pl.col("fut_ask")).then(pl.col("AskLots1"))
          .otherwise(pl.col("BestAskLots")).alias("fut_ask_lots"),

        pl.when(pl.col("fut_bid").is_null()).then(0)
          .when((_nz("BidPrice1") == pl.col("fut_bid")) & (_nz("BestBidPrice") == pl.col("fut_bid")))
          .then(pl.max_horizontal("BidLots1", "BestBidLots"))
          .when(_nz("BidPrice1") == pl.col("fut_bid")).then(pl.col("BidLots1"))
          .otherwise(pl.col("BestBidLots")).alias("fut_bid_lots"),
    ])
    # 期貨原始報價序號（回 tick 定位用）
    if "ChannelSeq" in fut.columns:
        fut = fut.with_columns(pl.col("ChannelSeq").alias("fut_chseq"))
    return fut


def asof_join(fut: pl.DataFrame, spot: pl.DataFrame) -> pl.DataFrame:
    """以期貨為主體，按 TransTime 往回找最近一筆同 ValueCode 的現貨（上一刻現貨）。

    現貨欄改前綴 spot_，保留要用的 A1/B1 價量。as-of join 需先各自依時間排序。
    E06：限 tolerance="60s"——往回最近一筆現貨若距期貨時間 >60 秒，視為過時、
      不對入（現貨欄 spot_* 全 null）。下游須丟棄現貨欄為 null 的列（見 caller）。
      無 tolerance 時過時現貨會被當「最新」對到（最久 2.2 小時），回測虛賺、實單致命。
    """
    spot_cols = [
        TIME_COL, "ValueCode",
        # 現貨自己的時間戳另存一欄：as-of join 用 on=TIME_COL，現貨的 TIME_COL 會被期貨時間
        #   覆蓋掉，故改名 spot_time 才能保留「這筆現貨報價的真實時間」(E08：供量現貨 staleness)。
        pl.col(TIME_COL).alias("spot_time"),
        pl.col("AskPrice1").alias("spot_ask"), pl.col("AskLots1").alias("spot_ask_lots"),
        pl.col("BidPrice1").alias("spot_bid"), pl.col("BidLots1").alias("spot_bid_lots"),
    ]
    if "ChannelSeq" in spot.columns:
        spot_cols.append(pl.col("ChannelSeq").alias("spot_chseq"))  # 現貨報價序號(回 tick 定位)
    spot_sel = spot.select(spot_cols).sort(TIME_COL)

    fut = fut.sort(TIME_COL)

    # 有 by 分組時 polars 無法驗證排序、會跳 UserWarning；我們已 sort，故抑制此提醒。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        return fut.join_asof(
            spot_sel,
            on=TIME_COL,
            by="ValueCode",
            strategy="backward",   # 往回找 <= 期貨時間的最近一筆現貨
            tolerance="60s",       # E06：限 60 秒內，過時現貨不當「最新」(超過→現貨欄 null，下游丟棄)
        )
