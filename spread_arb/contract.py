"""合約代碼解析與結算日工具。

所有對外吃日期的 function 都接受 int/str(yyyymmdd)/date/datetime，
進來第一件事用 to_date() 統一成 date 再運算（見 METHODOLOGY.md 步驟 1-2）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

# 月份碼：A=1月 … L=12月
_MONTH_CODES = "ABCDEFGHIJKL"


def to_date(d: int | str | date | datetime) -> date:
    """統一把日期參數轉成 date。接受 20260608 / "20260608" / date / datetime。"""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y%m%d").date()


def third_wednesday(year: int, month: int) -> date:
    """某年月的第三個星期三（台股期貨名目結算日，未含假日順延）。"""
    first = date(year, month, 1)
    # 第一個星期三：weekday() 週三=2
    offset = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=offset)
    return first_wed + timedelta(days=14)  # 第三個 = 第一個 + 兩週


# --- 結算日假日順延（審計 V03-3：2026/2/18 撞春節，實際結算 2/23）---
_is_trade_day_fn = None
_settle_cache: dict = {}


def set_trade_day_fn(fn):
    """注入交易日判斷函式（如 mysql.is_trade_day，吃 yyyymmdd 字串回 bool）。
    設定後，結算日逢非交易日自動順延至下一交易日；未設定維持純第三週三（舊行為）。"""
    global _is_trade_day_fn
    _is_trade_day_fn = fn
    _settle_cache.clear()


def _actual_settlement(year: int, month: int) -> date:
    """實際結算日 = 第三個星期三，逢非交易日順延至下一交易日（需先 set_trade_day_fn）。"""
    d = third_wednesday(year, month)
    if _is_trade_day_fn is None:
        return d
    key = (year, month)
    if key in _settle_cache:
        return _settle_cache[key]
    dd = d
    for _ in range(10):  # 最多順延10天（春節連假足夠）
        try:
            if _is_trade_day_fn(dd.strftime("%Y%m%d")):
                break
        except Exception as e:
            # L5 防呆：日曆查詢失敗 → 回退名目第三週三（保守、不中斷回測），
            #   但大聲告警，避免「實單對錯近月合約」這種無聲後果。
            print(f"⚠️ L5：{year}-{month:02d} 結算日順延查詢失敗（{e}），回退名目第三週三 "
                  f"{d}。實單前須確認日曆服務正常、勿用此 fallback 下單。")
            dd = d
            break
        dd += timedelta(days=1)
    _settle_cache[key] = dd
    return dd


def month_code(month: int) -> str:
    """月份(1-12) → 月份碼(A-L)。"""
    return _MONTH_CODES[month - 1]


def near_month_ym(d: int | str | date | datetime) -> tuple[int, int]:
    """用資料日期推近月合約的 (到期年, 到期月)。

    規則（METHODOLOGY.md 步驟 2 近月合約）：
      資料日 <= 當月結算日 → 近月 = 當月（結算日當天整天仍算當月）
      資料日 >  當月結算日 → 近月 = 下月（跨年時月+1進位、年+1）
    """
    d = to_date(d)
    settle = _actual_settlement(d.year, d.month)   # 含假日順延
    if d <= settle:
        return d.year, d.month
    # 已過結算 → 下月
    if d.month == 12:
        return d.year + 1, 1
    return d.year, d.month + 1


def near_month_code(d: int | str | date | datetime) -> str:
    """近月合約的月份+年尾數碼，例如 20260608 → "F6"。"""
    y, m = near_month_ym(d)
    return f"{month_code(m)}{y % 10}"


def settlement_date(d: int | str | date | datetime) -> date:
    """近月合約的結算日（第三個星期三，逢假日順延至下一交易日）。"""
    y, m = near_month_ym(d)
    return _actual_settlement(y, m)


def days_to_settlement(d: int | str | date | datetime) -> int:
    """距近月結算日的天數 = 結算日 − 資料日。"""
    d = to_date(d)
    return (settlement_date(d) - d).days


def parse_contract_suffix(quote_code: str) -> tuple[str, int]:
    """期貨 QuoteCode 末兩碼 → (月份碼, 年尾數)。例如 "CCFF6" → ("F", 6)。"""
    return quote_code[-2], int(quote_code[-1])
