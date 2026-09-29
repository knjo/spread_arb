"""Taker 研究線的資料來源與輸出路徑（2026-09-07 起改讀 SSD2 / NAS，不再經 sdk_core）。

來源契約：
  現貨 ticks      : pipeline.yaml data_storage.tick_dir      → {date}_StockTick.parquet
                    （價格已還原成真實價 float；RecvTime 為 naive UTC）
  股期 ticks      : /mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet
                    （價格為放大整數 ÷100；RecvTime 為 tz-aware UTC[ns]；含全部月份）
  現貨基本面      : pipeline.yaml data_storage.market_dir   → {date}_marketData.parquet
  期貨基本面/日曆 : MySQL（ProductInfo.taifex_pib_view、Common.calendar_view）
  研究輸出        : data_storage.base_dir / stockfuture     （原 HFT/data/stockfuture 搬到 SSD2）

所有腳本一律從這裡取路徑，不再各自寫 PROJECT_ROOT / "data"。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import polars as pl


def find_hft_root(start: Path | None = None) -> Path:
    """以 pyproject.toml + config/pipeline.yaml 定位 HFT 專案根目錄。"""
    origin = Path(__file__).resolve() if start is None else Path(start).resolve()
    search = origin if origin.is_dir() else origin.parent
    for cand in (search, *search.parents):
        if (cand / "pyproject.toml").is_file() and (cand / "config" / "pipeline.yaml").is_file():
            return cand
    raise RuntimeError("could not locate the HFT project root")


HFT_ROOT = find_hft_root()
if str(HFT_ROOT) not in sys.path:
    sys.path.insert(0, str(HFT_ROOT))

from src.pipeline_storage import load_pipeline_storage  # noqa: E402

STORAGE = load_pipeline_storage(project_root=HFT_ROOT)
DATA_ROOT = STORAGE.base_dir                     # /media/kevin/SSD2/Data
REQUIRED_MOUNT = STORAGE.required_mount          # /media/kevin/SSD2
MARKET_DIR = STORAGE.market_dir
SPOT_TICK_DIR = STORAGE.tick_dir
TICK_FEATURE_DIR = STORAGE.tick_feature_dir
PREMARKET_DIR = STORAGE.pre_market_dir
STOCKFUTURE_DIR = DATA_ROOT / "stockfuture"      # 本研究線所有中間檔/回測輸出
PLOT_DIR = STOCKFUTURE_DIR / "plots"

FUT_NAS_ROOT = Path("/mnt/NAS/Parquet/Ticks")
FUT_NAS_MOUNT = Path("/mnt/NAS")

SPOT_TICK_SUFFIX = "_StockTick.parquet"
MARKET_SUFFIX = "_marketData.parquet"

MYSQL_HOST_DEFAULT = "192.168.1.187"


# ---------------------------------------------------------------- 路徑
def require_mount(mount: Path | None, label: str) -> None:
    """外接儲存沒掛就直接失敗，避免在系統碟無聲重建資料夾。"""
    if mount is None:
        return
    if not mount.is_dir() or not os.path.ismount(mount):
        raise RuntimeError(f"{label} mount unavailable: {mount}")


def ensure_output_dir(path: Path = STOCKFUTURE_DIR) -> Path:
    require_mount(REQUIRED_MOUNT, "SSD2 data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def market_data_path(date: str) -> Path:
    return MARKET_DIR / f"{date}{MARKET_SUFFIX}"


def spot_tick_path(date: str) -> Path:
    return SPOT_TICK_DIR / f"{date}{SPOT_TICK_SUFFIX}"


def futures_raw_path(date: str) -> Path:
    d = datetime.strptime(date, "%Y%m%d")
    return FUT_NAS_ROOT / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d") / "stock_futures.parquet"


def available_dates(directory: Path, suffix: str, start: str | None = None,
                    end: str | None = None) -> list[str]:
    out = []
    for p in sorted(directory.glob(f"*{suffix}")):
        d = p.name[: -len(suffix)]
        if len(d) != 8 or not d.isdigit():
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(d)
    return out


def market_dates(start: str | None = None, end: str | None = None) -> list[str]:
    return available_dates(MARKET_DIR, MARKET_SUFFIX, start, end)


def spot_tick_dates(start: str | None = None, end: str | None = None) -> list[str]:
    return available_dates(SPOT_TICK_DIR, SPOT_TICK_SUFFIX, start, end)


# ---------------------------------------------------------------- 時間軸
def to_taipei_naive(col: str = "RecvTime", dtype: pl.DataType | None = None) -> pl.Expr:
    """把 RecvTime 轉成台北 naive datetime[us]。

    tz-aware（NAS 期貨 UTC[ns]）→ convert_time_zone；naive（SSD2 現貨，pipeline 存的是 UTC）
    → 視為 UTC 再轉。dtype 給原欄型別以決定走哪條路。
    """
    expr = pl.col(col)
    if dtype is not None and getattr(dtype, "time_zone", None) is not None:
        expr = expr.dt.convert_time_zone("Asia/Taipei").dt.replace_time_zone(None)
    else:
        expr = expr.dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Taipei") \
                   .dt.replace_time_zone(None)
    return expr.cast(pl.Datetime("us")).alias(col)


def normalize_recv_time(df: pl.DataFrame, col: str = "RecvTime") -> pl.DataFrame:
    if col not in df.columns:
        raise ValueError(f"missing {col}")
    return df.with_columns(to_taipei_naive(col, df.schema[col]))


# ---------------------------------------------------------------- tick 讀取
def scan_spot_ticks(date: str, codes: list[str] | None = None) -> pl.LazyFrame:
    """SSD2 現貨 ticks（價格已是真實價、RecvTime naive UTC）。codes=None 全市場。"""
    path = spot_tick_path(date)
    if not path.exists():
        raise FileNotFoundError(f"spot ticks not found: {path}")
    lf = pl.scan_parquet(path)
    if codes is not None:
        lf = lf.filter(pl.col("ValueCode").is_in(list(codes)))
    return lf


def scan_futures_ticks(date: str, quote_codes: list[str] | None = None) -> pl.LazyFrame:
    """NAS 股期 ticks（放大整數價、RecvTime tz-aware UTC）。quote_codes=None 全部合約。"""
    require_mount(FUT_NAS_MOUNT, "NAS")
    path = futures_raw_path(date)
    if not path.exists():
        raise FileNotFoundError(f"stock futures ticks not found: {path}")
    lf = pl.scan_parquet(path)
    if quote_codes is not None:
        lf = lf.filter(pl.col("QuoteCode").is_in(list(quote_codes)))
    return lf


# ---------------------------------------------------------------- 基本面 / 日曆（MySQL）
def _mysql_engine():
    from sqlalchemy import create_engine

    host = os.getenv("MYSQL_HOST") or MYSQL_HOST_DEFAULT
    return create_engine(f"mysql+pymysql://data.admin:automated@{host}:3306", pool_pre_ping=True)


def _mysql_query(sql: str, params: dict | None = None) -> pl.DataFrame:
    import pandas as pd
    from sqlalchemy import text

    with _mysql_engine().begin() as conn:
        pdf = pd.read_sql(text(sql), conn, params=params or {})
    return pl.from_pandas(pdf)


def load_futures_basic(date: str | int) -> pl.DataFrame:
    """期貨基本面（quote_code/value_code/ref_price/contract_size/decimal_locator/end_date）。
    ref_price 已是真實價，不可再除。"""
    df = _mysql_query(
        """
        SELECT quote_code, value_code, ref_price, contract_size, decimal_locator, end_date
        FROM ProductInfo.taifex_pib_view
        WHERE date = :date AND prod_kind = 'stock' AND ins_type = 'futures'
        """,
        {"date": int(date)},
    )
    if df.height == 0:
        raise RuntimeError(f"{date}: no futures basic info")
    return df


def is_trade_day(date: str | int) -> bool:
    df = _mysql_query(
        "SELECT DayType FROM Common.calendar_view WHERE date = :date", {"date": int(date)}
    )
    if df.height == 0:
        raise RuntimeError(f"calendar_view 查無 {date}")
    return df["DayType"][0] == "TradeDay"


def load_futures_settle_price(date: str | int) -> pl.DataFrame:
    return _mysql_query(
        """
        SELECT quote_code, settlement_price
        FROM MarketInfo.taifex_futures_trades_daily
        WHERE date = :date AND trading_session = 'day' AND char_length(quote_code) = 5
        """,
        {"date": int(date)},
    )


def load_spot_basic(date: str | int) -> pl.DataFrame:
    """現貨基本面（取代 TwMarketData.get_equity_basic_info）：
    回 quote_code / opening_ref_price / allow_day_trade_mark，只留股票類。"""
    path = market_data_path(str(date))
    if not path.exists():
        raise FileNotFoundError(f"marketData not found: {path}")
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    keep = [c for c in ("quote_code", "opening_ref_price", "allow_day_trade_mark", "ins_type")
            if c in cols]
    df = lf.select(keep).collect()
    if "ins_type" in df.columns:
        df = df.filter(pl.col("ins_type") == "stock").drop("ins_type")
    return df.unique(subset=["quote_code"])
