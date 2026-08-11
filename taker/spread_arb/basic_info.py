"""個股/期貨基本面（參考價、可當沖）載入與 join。

現貨：TwMarketData.get_equity_basic_info(date, ins_type='stock')，回 pandas。
期貨：StrategyMySQLLoader.get_futures_basic_info(date)（當初未包進 SDK，直連 MySQL），回 pandas。
價格為真實價（不用 ÷scale）。

漲跌停過濾統一用「參考價 ±9%」（見 preprocess.filter_price_limit），
所以這裡只需把參考價(+現貨可當沖標記) join 進 ticks 當新欄位，向量化過濾。
"""
from __future__ import annotations

import polars as pl

# 現貨基本面要帶進 ticks 的欄位（quote_code 是對齊鍵）
_SPOT_KEEP = ["quote_code", "opening_ref_price", "allow_day_trade_mark"]


def load_stock_basic(tw_md, date) -> pl.DataFrame:
    """讀現貨個股基本面（pandas → polars），只留要用的欄位。"""
    pdf = tw_md.get_equity_basic_info(date=date, ins_type="stock")
    df = pl.from_pandas(pdf)
    return df.select([c for c in _SPOT_KEEP if c in df.columns])


def load_futures_basic(mysql_loader, date) -> pl.DataFrame:
    """讀期貨基本面（MySQL taifex_pib_view，pandas → polars）。
    含 contract_size（契約乘數：標準2000/小型100/調整契約可能帶小數如1794.1176）。
    注意：view 的 ref_price 已是真實價（實測 1303 南亞=104），不可再除；
    decimal_locator 描述該商品 tick 餵價縮放（期貨=2 即÷100，對應 FUT_SCALE），
    僅為 metadata，與 ref_price 無關（曾誤除→±9%全滅→單日0事件）。"""
    pdf = mysql_loader.get_futures_basic_info(date=date)
    return pl.from_pandas(pdf)


def _warn_join_miss(joined: pl.DataFrame, key_col: str, check_col: str, what: str) -> None:
    """L6 防呆：left join 後 check_col 為 null＝該標的 join 不到基本面，
    下游過濾會把它整天靜默丟掉。數出失配標的數＋筆數並告警，避免無聲漏失。"""
    miss = joined.filter(pl.col(check_col).is_null())
    if miss.height:
        codes = miss[key_col].unique().to_list()
        print(f"⚠️ L6：{what} 有 {len(codes)} 檔 join 不到基本面（{miss.height:,} 筆，"
              f"下游會靜默丟棄）：{codes[:10]}{' …' if len(codes) > 10 else ''}")


def join_spot_basic(spot: pl.DataFrame, basic: pl.DataFrame) -> pl.DataFrame:
    """現貨 ticks join 基本面：basic.quote_code ↔ ticks.ValueCode。
    帶入 opening_ref_price（→ 改名 ref_price 供統一過濾）、allow_day_trade_mark。"""
    basic = basic.rename({"opening_ref_price": "ref_price"})
    joined = spot.join(basic, left_on="ValueCode", right_on="quote_code", how="left")
    _warn_join_miss(joined, "ValueCode", "ref_price", "現貨基本面")
    return joined


def join_futures_basic(fut: pl.DataFrame, basic: pl.DataFrame) -> pl.DataFrame:
    """期貨 ticks join 基本面：basic.quote_code(合約碼) ↔ ticks.QuoteCode。
    帶入 ref_price（已還原真實價，供±9%過濾）與 contract_size（逐合約乘數）。"""
    basic = basic.select(["quote_code", "ref_price", "contract_size"])
    joined = fut.join(basic, left_on="QuoteCode", right_on="quote_code", how="left")
    _warn_join_miss(joined, "QuoteCode", "ref_price", "期貨基本面")
    return joined
