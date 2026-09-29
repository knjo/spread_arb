"""第1層：讀檔 + 還原價格 + 篩近月（METHODOLOGY.md 步驟 1-2）。

對外吃日期一律 int/str(yyyymmdd)/date/datetime，原樣傳給 SDK，
自己要算日曆時才用 contract.to_date 轉。
"""
from __future__ import annotations

import polars as pl

from .contract import near_month_code

try:  # 本地資料來源（SSD2 現貨 / NAS 股期）；由 taker/ 目錄執行時可匯入
    import data_paths as _dp
except ImportError:  # pragma: no cover - 只在非 taker 目錄執行時發生
    _dp = None

# 價格還原 scale（df 無 DecimalLocator 欄，分析端用常數，METHODOLOGY.md 步驟 1）
SPOT_SCALE = 10000   # 現貨 ÷10000
FUT_SCALE = 100      # 股期 ÷100

# 時間軸：用 RecvTime（系統實際收到的時間＝真正打得到的時間線），
# 而非 TransTime（交易所撮合時間，封包到你手上有延遲，用它對齊=偷看未收到的資訊）。
# RecvTime 為 UTC，前處理時轉台北時間(naive)供全管線使用。要換回 TransTime 改這行即可。
TIME_COL = "RecvTime"

# 所有「價格」欄位（量/Lots 不動）
PRICE_COLS = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + ["FillPrice", "BestBidPrice", "BestAskPrice"]
)


def _drop_trial_match(df: pl.DataFrame) -> pl.DataFrame:
    """白名單：只留 TrialMatch==0（正式撮合、可成交）。
    其餘皆為緩搓（1=緩搓, 2=緩搓且趨漲, 3=緩搓且趨跌，及任何未知值），
    都不能成交，留著會產生假價差、切碎事件 → 一律排除。"""
    if "TrialMatch" in df.columns:
        return df.filter(pl.col("TrialMatch") == 0)
    return df


def _restore_prices(df: pl.DataFrame, scale: int) -> pl.DataFrame:
    """把放大整數的價格欄全部 ÷scale 還原成真實價（元，float）。
    只轉 df 裡存在且仍是整數的價格欄（SSD2 現貨檔已是 float 真實價，不可再除）。"""
    cols = [c for c in PRICE_COLS if c in df.columns and not df.schema[c].is_float()]
    if not cols:
        return df
    return df.with_columns([(pl.col(c) / scale).alias(c) for c in cols])


def _mark_quote_fill(df: pl.DataFrame) -> pl.DataFrame:
    """不 drop 成交 tick，改加 flag 欄（原始事實，與事件無關，屬前處理）。

    舊版會 filter 掉純成交 tick；但胃納量(capacity)要靠「事件打開後、達 taker 價的
    成交量」累加，那些成交列被 drop 就永遠進不了流 → 改成全份留著、加兩個布林旗標：
      - is_quote：有委託簿(五檔)的報價列。下游價差只在報價列算，成交列 ret 設 null，
        靠 tag_events 對 null 的「維持前狀態、不切事件」自動落回所屬事件區間。
      - is_fill ：有成交的列(FillPrice>0 且 FillLots>0)。胃納量累加用。
    證交所現貨：成交與五檔同一 row（有成交則 is_quote 與 is_fill 可同真）；
    期貨：成交與五檔分開 row（成交列五檔全 0 → is_quote=false、is_fill=true）。"""
    has_book = (pl.col("BidPrice1") > 0) | (pl.col("AskPrice1") > 0)
    has_fill = (pl.col("FillPrice") > 0) & (pl.col("FillLots") > 0)
    return df.with_columns([
        has_book.alias("is_quote"),
        has_fill.alias("is_fill"),
    ])


def drop_price0_with_lots(df: pl.DataFrame) -> pl.DataFrame:
    """漲跌停異常：任一腳 Price==0 但 Lots>0（有單只是價揭示成0）→ 整筆濾掉。
    注意與「Price==0 且 Lots==0 = 無報價」分開：這裡是有量卻價0，視為漲跌停。"""
    bad = (
        ((pl.col("BidPrice1") == 0) & (pl.col("BidLots1") > 0))
        | ((pl.col("AskPrice1") == 0) & (pl.col("AskLots1") > 0))
    )
    return df.filter(~bad)


def filter_day_tradable(df: pl.DataFrame) -> pl.DataFrame:
    """只留可當沖標的（allow_day_trade_mark=="X"；含排除處置股）。
    需先 join_basic 帶入 allow_day_trade_mark 欄。"""
    if "allow_day_trade_mark" not in df.columns:
        return df
    return df.filter(pl.col("allow_day_trade_mark") == "X")


LIMIT_PCT = 0.09   # 參考價 ±9% 安全邊界（逼近漲跌停就不可靠）


def filter_price_limit(df: pl.DataFrame, ref_col: str = "ref_price") -> pl.DataFrame:
    """漲跌停過濾（統一規則）：報價超出 參考價×(1±9%) → 濾掉。
    現貨與期貨同一套邏輯，只是參考價各取各的（現貨 opening_ref_price、期貨 ref_price）。
    需先把參考價 join 進來成 ref_col 欄。只對 >0 的有效報價判斷（0 是無報價）。

    C4 修正：不只判明掛 A1/B1，連衍生 Best 價(BestAskPrice/BestBidPrice)一併判。
      因為下游 best_quotes 會用 min/max(A1, BestAsk) 取「更優」價當成交價，
      若 Best 價觸漲跌停而沒擋，會被當成可成交的優價用到事件裡（期貨 Best 腳漏過濾）。
      Best 欄只有期貨有；現貨無此欄時自動跳過該腳。"""
    if ref_col not in df.columns:
        return df
    up = pl.col(ref_col) * (1 + LIMIT_PCT)
    dn = pl.col(ref_col) * (1 - LIMIT_PCT)

    def _out(col: str):
        return (pl.col(col) > 0) & ((pl.col(col) >= up) | (pl.col(col) <= dn))

    hit = _out("AskPrice1") | _out("BidPrice1")
    # 衍生 Best 價（期貨才有）：best_quotes 會取為成交價，故同樣須檢漲跌停
    for bc in ("BestAskPrice", "BestBidPrice"):
        if bc in df.columns:
            hit = hit | _out(bc)
    return df.filter(~hit)


# 時段截止：13:20 之後的報價不看（交易員拍板）。
# 原因：股票 13:25 進收盤集合競價、大概率打不到；期貨交易到 13:45，
# 不設限的話 13:20~13:45 的期貨報價會一直對到打不到的現貨 → 幽靈事件(審計 V02)。
SESSION_CUTOFF = (13, 20)


def _clean(df: pl.DataFrame, scale: int) -> pl.DataFrame:
    """前處理：濾緩搓 → 還原價格 → 標 is_quote/is_fill(不 drop 成交) → 時間軸轉台北 → 13:20截止。"""
    df = _drop_trial_match(df)
    df = _restore_prices(df, scale)
    df = _mark_quote_fill(df)
    # 時間軸統一：TIME_COL(RecvTime, UTC) → 台北時間(naive)，輸出時間才不會差 8 小時。
    # NAS 期貨為 tz-aware、SSD2 現貨為 naive UTC，兩種都處理。
    if df.schema[TIME_COL].time_zone is not None:
        df = df.with_columns(
            pl.col(TIME_COL).dt.convert_time_zone("Asia/Taipei").dt.replace_time_zone(None)
        )
    else:
        df = df.with_columns(
            pl.col(TIME_COL).dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Taipei")
              .dt.replace_time_zone(None)
        )
    df = df.with_columns(pl.col(TIME_COL).cast(pl.Datetime("us")))
    # 用 dt.time() 直接比，不可用 hour()*60+minute()：dt.hour() 回 Int8，
    # ×60 不升型會無聲溢位(780→12)，導致過濾完全失效（實際踩過）
    return df.filter(pl.col(TIME_COL).dt.time() < pl.time(*SESSION_CUTOFF))


def _load_kw(date, code) -> dict:
    kw = {"date": date}
    if code is not None:
        kw["code"] = code      # SDK 支援：str | list[str]，撈時直接過濾代號
    return kw


def _code_list(code) -> list[str] | None:
    if code is None:
        return None
    return [code] if isinstance(code, str) else list(code)


def load_spot(tw, date, code=None) -> pl.DataFrame:
    """讀現貨 ticks 並還原價格。tw=None → 讀 SSD2 {date}_StockTick.parquet（預設）；
    tw 給 sdk_core.TwTicks 則走舊 NAS SDK。code：股票代號(str|list)，不傳=全市場。"""
    if tw is None:
        if _dp is None:
            raise RuntimeError("data_paths 不可匯入：請從 taker/ 目錄執行")
        df = _dp.scan_spot_ticks(str(date), _code_list(code)).collect()
    else:
        df = tw.get_stock_round_only(**_load_kw(date, code))
    return _clean(df, SPOT_SCALE)


def load_futures(tw, date, code=None) -> pl.DataFrame:
    """讀股期 ticks 並還原價格（÷100）。tw=None → 讀 NAS YYYY/MM/DD/stock_futures.parquet；
    code：股期合約代號(str|list，如 CCFF6)，不傳=全部合約。"""
    if tw is None:
        if _dp is None:
            raise RuntimeError("data_paths 不可匯入：請從 taker/ 目錄執行")
        df = _dp.scan_futures_ticks(str(date), _code_list(code)).collect()
    else:
        df = tw.get_stock_futures_only(**_load_kw(date, code))
    return _clean(df, FUT_SCALE)


def filter_standard_contract(fut: pl.DataFrame) -> pl.DataFrame:
    """只留標準個股期貨：QuoteCode 第三碼 == 'F'。

    第三碼非 F（如 '1'，形如 __1F6）是跨期/組合等特殊合約，
    每月靠近結算日才出現，其 contract_size 是小數（如 1794.1176，因非單一個股、
    乘數無意義），不可當個股期貨算 → 整批剔除。
    （注意：小型個股期貨第三碼也是 F，不受此過濾影響、照常保留。）"""
    return fut.filter(pl.col("QuoteCode").str.slice(2, 1) == "F")


def filter_near_month(fut: pl.DataFrame, date) -> pl.DataFrame:
    """只留近月標準合約：先剔除非標準合約(第三碼非F)，再留末兩碼==near_month_code。"""
    fut = filter_standard_contract(fut)
    code = near_month_code(date)
    return fut.filter(pl.col("QuoteCode").str.slice(-2) == code)


def to_minute_bars(df: pl.DataFrame, by: str = "ValueCode",
                   lag: bool = False) -> pl.DataFrame:
    """壓成 1 分 K：每商品、每分鐘取「最後一筆」委託簿快照（METHODOLOGY.md 步驟 1）。

    低頻版用分 K，取每分鐘最後一筆（報價類資料慣例＝該分鐘結束能掛到的價，非 OHLC）。
    by：分組鍵（現貨/期貨皆用 ValueCode；期貨已篩近月，同 ValueCode 只剩一個合約）。
    lag：是否把時間戳往後推 1 分鐘。用於現貨 —— 「以期貨為主體對齊上一刻現貨」，
         這一分鐘的期貨要對齊『前一分鐘』的現貨，故現貨壓 K 後標籤 +1 分，
         讓 08:31 的期貨 as-of 對到 08:30 的現貨。
    需 df 已前處理（濾緩搓、還原價、濾成交 tick）。
    """
    bars = (
        df.sort(TIME_COL)
          .with_columns(pl.col(TIME_COL).dt.truncate("1m").alias("minute"))
          .group_by(by, "minute")
          .last()
    )
    ts = pl.col("minute") + pl.duration(minutes=1) if lag else pl.col("minute")
    return bars.with_columns(ts.alias(TIME_COL)).sort([by, TIME_COL])
