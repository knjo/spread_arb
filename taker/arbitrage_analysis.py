"""Tick-level futures/spot arbitrage feature builder and first-entry analysis.

This script is intentionally separate from main.py so first-entry samples and
second-entry/capacity samples do not get mixed while we inspect assumptions.

Examples:
  uv run python src/research/futures_spot_spread/taker/arbitrage_analysis.py fetch \
    -s 20260609 --fut-code CCFF6

  uv run python src/research/futures_spot_spread/taker/arbitrage_analysis.py features \
    -s 20260609 --fut-code CCFF6 --code 2303

  uv run python src/research/futures_spot_spread/taker/arbitrage_analysis.py first \
    -s 20260609 -e 20260617
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from spread_arb import contract as contract_calendar
from spread_arb.contract import near_month_code, settlement_date


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKET_DIR = PROJECT_ROOT / "data" / "marketData"
OUT_DIR = PROJECT_ROOT / "data" / "stockfuture"

SPOT_SCALE = 10000
FUT_SCALE = 100
TIME_COL = "RecvTime"
THRESHOLDS = [0.005, 0.0075, 0.01, 0.0125]
STANDARD_CONTRACT_SIZE = 2000.0
CONTRACT_SIZE_EPS = 1e-6
REF_LIMIT_PCT = 0.09
SESSION_END_EXCLUSIVE = (13, 25, 0)  # keep ticks through the 13:24 minute only

PRICE_COLS = (
    [f"BidPrice{i}" for i in range(1, 6)]
    + [f"AskPrice{i}" for i in range(1, 6)]
    + ["FillPrice", "BestBidPrice", "BestAskPrice"]
)

FUT_RAW_TEMPLATE = "{date}_stockfuture.parquet"
SPOT_RAW_TEMPLATE = "{date}_spot_for_stockfuture.parquet"
FEATURE_TEMPLATE = "{date}_arbitrage_features.parquet"


def _is_standard_contract_expr() -> pl.Expr:
    return (pl.col("contract_size") - STANDARD_CONTRACT_SIZE).abs() <= CONTRACT_SIZE_EPS


def _date_range(start: str, end: str | None) -> list[str]:
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end or start, "%Y%m%d").date()
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _split_codes(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _raw_future_path(date: str) -> Path:
    return OUT_DIR / FUT_RAW_TEMPLATE.format(date=date)


def _raw_spot_path(date: str) -> Path:
    return OUT_DIR / SPOT_RAW_TEMPLATE.format(date=date)


def _feature_path(date: str) -> Path:
    return OUT_DIR / FEATURE_TEMPLATE.format(date=date)


def _restore_prices(df: pl.DataFrame, scale: int) -> pl.DataFrame:
    cols = [c for c in PRICE_COLS if c in df.columns]
    return df.with_columns([(pl.col(c) / scale).alias(c) for c in cols])


def _normalize_time(df: pl.DataFrame) -> pl.DataFrame:
    if TIME_COL not in df.columns:
        raise ValueError(f"missing {TIME_COL}")
    return df.with_columns(
        pl.col(TIME_COL).dt.convert_time_zone("Asia/Taipei").dt.replace_time_zone(None)
    )


def _after_0900_expr(col: str = "TransTime") -> pl.Expr:
    return pl.col(col).dt.time() > pl.time(9, 0, 0)


def _before_close_auction_expr(col: str = "TransTime") -> pl.Expr:
    return pl.col(col).dt.time() < pl.time(*SESSION_END_EXCLUSIVE)


def _session_time_expr(col: str = "TransTime") -> pl.Expr:
    return _after_0900_expr(col) & _before_close_auction_expr(col)


def _has_book_expr() -> pl.Expr:
    return (pl.col("BidPrice1") > 0) | (pl.col("AskPrice1") > 0)


def _within_ref_limit_expr(ref_col: str, price_cols: list[str]) -> pl.Expr:
    up = pl.col(ref_col) * (1 + REF_LIMIT_PCT)
    dn = pl.col(ref_col) * (1 - REF_LIMIT_PCT)
    ok = pl.col(ref_col).is_not_null() & (pl.col(ref_col) > 0)
    for col in price_cols:
        ok = ok & (
            (pl.col(col) <= 0)
            | ((pl.col(col) < up) & (pl.col(col) > dn))
        )
    return ok


def _not_opening_fill_expr(columns: list[str]) -> pl.Expr:
    if "TotalFillLots" not in columns or "FillLots" not in columns:
        return pl.lit(True)
    return ~(
        (pl.col("FillLots") > 0)
        & (pl.col("TotalFillLots") == pl.col("FillLots"))
    )


def _nz(col: str) -> pl.Expr:
    return pl.when(pl.col(col) > 0).then(pl.col(col)).otherwise(None)


def _add_futures_best_quotes(fut: pl.DataFrame) -> pl.DataFrame:
    ask1 = _nz("AskPrice1")
    bid1 = _nz("BidPrice1")
    best_ask = _nz("BestAskPrice") if "BestAskPrice" in fut.columns else pl.lit(None)
    best_bid = _nz("BestBidPrice") if "BestBidPrice" in fut.columns else pl.lit(None)
    fut = fut.with_columns([
        pl.min_horizontal(ask1, best_ask).alias("fut_ask"),
        pl.max_horizontal(bid1, best_bid).alias("fut_bid"),
    ])
    return fut.with_columns([
        pl.when(pl.col("fut_ask").is_null()).then(0)
          .when(("BestAskPrice" in fut.columns) & (_nz("BestAskPrice") == pl.col("fut_ask")))
          .then(pl.col("BestAskLots") if "BestAskLots" in fut.columns else pl.col("AskLots1"))
          .otherwise(pl.col("AskLots1")).alias("fut_ask_lots"),
        pl.when(pl.col("fut_bid").is_null()).then(0)
          .when(("BestBidPrice" in fut.columns) & (_nz("BestBidPrice") == pl.col("fut_bid")))
          .then(pl.col("BestBidLots") if "BestBidLots" in fut.columns else pl.col("BidLots1"))
          .otherwise(pl.col("BidLots1")).alias("fut_bid_lots"),
    ])


def _load_futures_basic(date: str) -> pl.DataFrame:
    """Load stock-futures product metadata without depending on python-dotenv."""
    try:
        from mysql import StrategyMySQLLoader

        pdf = StrategyMySQLLoader().get_futures_basic_info(date=int(date))
        return pl.from_pandas(pdf)
    except Exception:
        from sqlalchemy import create_engine, text
        import pandas as pd

        db_host = os.getenv("MYSQL_HOST") or "192.168.1.187"
        db_url = f"mysql+pymysql://data.admin:automated@{db_host}:3306"
        query = text(
            """
            SELECT quote_code, value_code, ref_price, contract_size,
                   decimal_locator, end_date
            FROM ProductInfo.taifex_pib_view
            WHERE date = :date
              AND prod_kind = 'stock'
              AND ins_type = 'futures'
            """
        )
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.begin() as conn:
            pdf = pd.read_sql(query, conn, params={"date": int(date)})
        if pdf.empty:
            raise RuntimeError(f"{date}: no futures basic info")
        return pl.from_pandas(pdf)


def _join_futures_basic(fut: pl.DataFrame, date: str) -> pl.DataFrame:
    basic = _load_futures_basic(date).select([
        pl.col("quote_code").alias("_basic_quote_code"),
        pl.col("value_code").alias("_basic_value_code"),
        pl.col("ref_price").cast(pl.Float64).alias("fut_ref_price"),
        pl.col("contract_size").cast(pl.Float64),
    ])
    fut = fut.drop([
        c for c in ("contract_size", "fut_ref_price", "ref_price", "_basic_value_code")
        if c in fut.columns
    ])
    joined = fut.join(basic, left_on="QuoteCode", right_on="_basic_quote_code", how="left")
    if "ValueCode" not in joined.columns and "_basic_value_code" in joined.columns:
        joined = joined.rename({"_basic_value_code": "ValueCode"})
    else:
        joined = joined.drop([c for c in ("_basic_value_code",) if c in joined.columns])
    miss = joined.filter(pl.col("contract_size").is_null() | pl.col("fut_ref_price").is_null())
    if miss.height:
        codes = miss["QuoteCode"].unique().head(10).to_list()
        print(
            f"{date}: warning {miss.height:,} futures ticks missing contract metadata; "
            f"examples={codes}"
        )
    return joined


def _load_market_tradable(date: str) -> pl.DataFrame:
    path = MARKET_DIR / f"{date}_marketData.parquet"
    if not path.exists():
        raise FileNotFoundError(f"marketData not found: {path}")
    df = pl.read_parquet(path)
    code_col = "quote_code" if "quote_code" in df.columns else "QuoteCode"
    day_col = "day_trade_table" if "day_trade_table" in df.columns else "allow_day_trade_mark"
    keep = [code_col, day_col]
    if "opening_ref_price" in df.columns:
        keep.append("opening_ref_price")
    for c in ("ins_type", "market", "stock_name"):
        if c in df.columns:
            keep.append(c)
    rename = {code_col: "ValueCode", day_col: "day_trade_mark"}
    if "opening_ref_price" in keep:
        rename["opening_ref_price"] = "spot_ref_price"
    df = df.select(keep).rename(rename)
    return (
        df.with_columns(pl.col("day_trade_mark").cast(pl.Utf8).str.to_uppercase())
          .filter(pl.col("day_trade_mark") == "X")
          .unique(subset=["ValueCode"])
    )


def _fetch_futures(date: str, fut_codes: list[str] | None, force: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_future_path(date)
    if path.exists() and not force:
        return path
    from sdk_core import TwTicks

    tw = TwTicks()
    df = tw.get_stock_futures_only(date=int(date), code=fut_codes)
    if df.height == 0:
        raise RuntimeError(f"{date}: no stock-futures ticks fetched")
    df = _join_futures_basic(df, date)
    near = near_month_code(date)
    df = df.filter(
        (pl.col("QuoteCode").str.slice(2, 1) == "F")
        & (pl.col("QuoteCode").str.slice(-2) == near)
        & _is_standard_contract_expr()
    )
    if df.height == 0:
        raise RuntimeError(f"{date}: no near-month standard contract_size=2000 futures ticks")
    df.write_parquet(path)
    return path


def _fetch_spot(date: str, codes: list[str], force: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_spot_path(date)
    if path.exists() and not force:
        return path
    from sdk_core import TwTicks

    tw = TwTicks()
    df = tw.get_stock_round_only(date=int(date), code=codes)
    if df.height == 0:
        raise RuntimeError(f"{date}: no spot ticks fetched for {len(codes)} codes")
    df.write_parquet(path)
    return path


def _load_spot_raw(date: str, codes: list[str], force_fetch: bool) -> pl.DataFrame:
    path = _fetch_spot(date, codes, force_fetch)
    spot = _restore_prices(_normalize_time(pl.read_parquet(path)), SPOT_SCALE)
    spot = spot.filter(_session_time_expr(TIME_COL))
    if "TransTime" in spot.columns:
        spot = spot.filter(_session_time_expr("TransTime"))
    return spot


def _add_spot_ref(spot: pl.DataFrame, spot_ref: pl.DataFrame) -> pl.DataFrame:
    if "spot_ref_price" in spot.columns:
        return spot
    return spot.join(spot_ref, on="ValueCode", how="left")


def fetch_samples(args: argparse.Namespace) -> None:
    fut_codes = _split_codes(args.fut_code)
    for date in _date_range(args.start_date, args.end_date):
        path = _fetch_futures(date, fut_codes, args.force)
        print(f"{date}: stock-futures ticks -> {path}")


def _load_futures_prepared(date: str, fut_codes: list[str] | None, code_filter: list[str] | None,
                           force_fetch: bool) -> pl.DataFrame:
    path = _fetch_futures(date, fut_codes, force_fetch)
    fut = pl.read_parquet(path)
    if "contract_size" not in fut.columns or "fut_ref_price" not in fut.columns:
        fut = _join_futures_basic(fut, date)
    fut = _restore_prices(_normalize_time(fut), FUT_SCALE)
    if "TrialMatch" in fut.columns:
        fut = fut.filter(pl.col("TrialMatch") == 0)
    fut = fut.filter(_session_time_expr(TIME_COL))
    if "TransTime" in fut.columns:
        fut = fut.filter(_session_time_expr("TransTime"))
    if code_filter is not None:
        fut = fut.filter(pl.col("ValueCode").is_in(code_filter))
    if fut_codes is not None:
        fut = fut.filter(pl.col("QuoteCode").is_in(fut_codes))
    if not getattr(sys.modules[__name__], "_ALLOW_ALL_MONTHS", False):
        fut = fut.filter(
            (pl.col("QuoteCode").str.slice(2, 1) == "F")
            & (pl.col("QuoteCode").str.slice(-2) == near_month_code(date))
        )
    fut = fut.filter(pl.col("contract_size").is_not_null() & _is_standard_contract_expr())
    fut = fut.filter(_has_book_expr())
    return _add_futures_best_quotes(fut).with_columns(
        _within_ref_limit_expr(
            "fut_ref_price",
            ["BidPrice1", "AskPrice1", "BestBidPrice", "BestAskPrice"],
        ).alias("fut_price_ok")
    ).sort(["QuoteCode", TIME_COL])


def _prepare_futures(date: str, fut_codes: list[str] | None, code_filter: list[str] | None,
                     force_fetch: bool) -> pl.DataFrame:
    return _load_futures_prepared(date, fut_codes, code_filter, force_fetch).filter(
        pl.col("fut_price_ok")
    )


def _prepare_futures_state(date: str, fut_codes: list[str] | None, code_filter: list[str] | None,
                           force_fetch: bool) -> pl.DataFrame:
    return (
        _load_futures_prepared(date, fut_codes, code_filter, force_fetch)
        .select([
            "QuoteCode",
            pl.col(TIME_COL).alias("fut_state_time"),
            pl.col("BidPrice1").alias("fut_bid1_state"),
            pl.col("AskPrice1").alias("fut_ask1_state"),
            "fut_price_ok",
        ])
        .sort(["QuoteCode", "fut_state_time"])
    )


def _prepare_spot(date: str, codes: list[str], force_fetch: bool,
                  spot_ref: pl.DataFrame) -> pl.DataFrame:
    spot = _add_spot_ref(_load_spot_raw(date, codes, force_fetch), spot_ref)
    if "TrialMatch" in spot.columns:
        spot = spot.filter(pl.col("TrialMatch") == 0)
    spot = spot.filter(_not_opening_fill_expr(spot.columns))
    return (
        spot.filter(_has_book_expr())
            .with_columns(
                _within_ref_limit_expr("spot_ref_price", ["BidPrice1", "AskPrice1"])
                .alias("spot_price_ok")
            )
            .filter(pl.col("spot_price_ok"))
            .select([
                "ValueCode", TIME_COL,
                "spot_ref_price",
                *([pl.col("TransTime").alias("spot_trans_time")] if "TransTime" in spot.columns else []),
                *([pl.col("TrialMatch").alias("spot_trial_match")] if "TrialMatch" in spot.columns else []),
                *([pl.col("ChannelSeq").alias("spot_chseq")] if "ChannelSeq" in spot.columns else []),
                pl.col("AskPrice1").alias("spot_ask"),
                pl.col("BidPrice1").alias("spot_bid"),
                pl.col("AskLots1").alias("spot_ask_lots"),
                pl.col("BidLots1").alias("spot_bid_lots"),
            ])
            .sort(["ValueCode", TIME_COL])
    )


def _prepare_spot_state(date: str, codes: list[str], force_fetch: bool,
                        spot_ref: pl.DataFrame) -> pl.DataFrame | None:
    spot = _add_spot_ref(_load_spot_raw(date, codes, force_fetch), spot_ref)
    if "TrialMatch" not in spot.columns:
        return None
    return (
        spot.with_columns(
            _within_ref_limit_expr("spot_ref_price", ["BidPrice1", "AskPrice1"])
            .alias("spot_price_ok")
        )
        .select([
            "ValueCode",
            pl.col(TIME_COL).alias("spot_state_time"),
            pl.col("TrialMatch").cast(pl.Int64).alias("spot_trial_state"),
            "spot_price_ok",
        ])
        .sort(["ValueCode", "spot_state_time"])
    )


def _join_asof(left: pl.DataFrame, right: pl.DataFrame, *, left_on: str, right_on: str,
               by: str, strategy: str = "backward", tolerance: str | None = None) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        return left.join_asof(
            right,
            left_on=left_on,
            right_on=right_on,
            by=by,
            strategy=strategy,
            tolerance=tolerance,
        )


def build_features_for_day(date: str, args: argparse.Namespace) -> Path:
    code_filter = _split_codes(args.code)
    fut_codes = _split_codes(args.fut_code)
    setattr(sys.modules[__name__], "_ALLOW_ALL_MONTHS", not args.near_month_only)

    tradable = _load_market_tradable(date)
    tradable_codes = set(tradable["ValueCode"].to_list())
    if code_filter is not None:
        tradable_codes &= set(code_filter)
    if not tradable_codes:
        raise RuntimeError(f"{date}: no day-tradable marketData codes after filters")
    spot_ref = tradable.select(["ValueCode", "spot_ref_price"]).filter(
        pl.col("spot_ref_price").is_not_null() & (pl.col("spot_ref_price") > 0)
    )

    fut = _prepare_futures(date, fut_codes, sorted(tradable_codes), args.force_fetch)
    fut_state = _prepare_futures_state(date, fut_codes, sorted(tradable_codes), args.force_fetch)
    if fut.height == 0:
        raise RuntimeError(f"{date}: no eligible futures quote ticks")
    mapping = fut.select("ValueCode", "QuoteCode").unique().sort(["ValueCode", "QuoteCode"])
    expected_suffix = near_month_code(date)
    bad_month = mapping.filter(
        (pl.col("QuoteCode").str.slice(2, 1) != "F")
        | (pl.col("QuoteCode").str.slice(-2) != expected_suffix)
    )
    if bad_month.height:
        raise ValueError(f"{date}: non-near-month futures survived filter: {bad_month}")
    value_codes = sorted(set(fut["ValueCode"].to_list()) & tradable_codes)
    spot = _prepare_spot(date, value_codes, args.force_fetch, spot_ref)
    if spot.height == 0:
        raise RuntimeError(f"{date}: no eligible spot quote ticks")
    spot_state = _prepare_spot_state(date, value_codes, args.force_fetch, spot_ref)

    spot_aligned = spot.rename({TIME_COL: "spot_time"})
    base = fut.select([
        pl.lit(date).alias("Date"),
        "ValueCode", "QuoteCode", "contract_size", TIME_COL,
        "fut_ref_price",
        *([pl.col("TransTime").alias("fut_trans_time")] if "TransTime" in fut.columns else []),
        *([pl.col("ChannelSeq").alias("fut_chseq")] if "ChannelSeq" in fut.columns else []),
        "BidPrice1", "AskPrice1", "BidLots1", "AskLots1",
        "fut_bid", "fut_ask", "fut_bid_lots", "fut_ask_lots",
    ]).rename({
        "BidPrice1": "fut_bid1",
        "AskPrice1": "fut_ask1",
        "BidLots1": "fut_bid1_lots",
        "AskLots1": "fut_ask1_lots",
    }).sort(["ValueCode", TIME_COL])

    if spot_state is not None:
        base = _join_asof(
            base,
            spot_state,
            left_on=TIME_COL,
            right_on="spot_state_time",
            by="ValueCode",
            strategy="backward",
        ).filter(
            (pl.col("spot_trial_state").fill_null(1) == 0)
            & pl.col("spot_price_ok").fill_null(False)
        )

    feat = _join_asof(
        base,
        spot_aligned,
        left_on=TIME_COL,
        right_on="spot_time",
        by="ValueCode",
        strategy="backward",
        tolerance="60s",
    )
    time_dtype = fut.schema[TIME_COL]
    feat = feat.filter(
        (pl.col("spot_bid") > 0) & (pl.col("spot_ask") > 0)
        & (pl.col("fut_bid") > 0) & (pl.col("fut_ask") > 0)
    ).with_columns([
        ((pl.col("fut_bid") - pl.col("spot_ask")) / pl.col("fut_ask")).alias("ret_sell"),
        ((pl.col("spot_bid") - pl.col("fut_ask")) / pl.col("fut_ask")).alias("ret_buy"),
        ((pl.col("fut_bid1") - pl.col("spot_ask")) / pl.col("fut_bid1")).alias("ret_sell_raw_b1"),
        (pl.col(TIME_COL) + pl.duration(milliseconds=50)).cast(time_dtype).alias("target_fut_50ms"),
        (pl.col(TIME_COL) + pl.duration(milliseconds=100)).cast(time_dtype).alias("target_spot_100ms"),
    ])

    if spot_state is not None:
        spot_state_100ms = spot_state.rename({
            "spot_state_time": "spot_state_time_100ms",
            "spot_trial_state": "spot_trial_state_100ms",
            "spot_price_ok": "spot_price_ok_100ms",
        })
        feat = _join_asof(
            feat.sort(["ValueCode", "target_spot_100ms"]),
            spot_state_100ms,
            left_on="target_spot_100ms",
            right_on="spot_state_time_100ms",
            by="ValueCode",
            strategy="backward",
        ).filter(
            (pl.col("spot_trial_state_100ms").fill_null(1) == 0)
            & pl.col("spot_price_ok_100ms").fill_null(False)
        )

    fut_after = fut_state.select([
        "QuoteCode",
        pl.col("fut_state_time").alias("fut_after_time"),
        pl.col("fut_bid1_state").alias("fut_bid1_50ms"),
        pl.col("fut_ask1_state").alias("fut_ask1_50ms"),
        pl.col("fut_price_ok").alias("fut_price_ok_50ms"),
    ]).sort(["QuoteCode", "fut_after_time"])
    feat = _join_asof(
        feat.sort(["QuoteCode", "target_fut_50ms"]),
        fut_after,
        left_on="target_fut_50ms",
        right_on="fut_after_time",
        by="QuoteCode",
        strategy="backward",
    ).filter(
        pl.col("fut_price_ok_50ms").fill_null(False)
    )

    spot_after = spot.select([
        "ValueCode",
        pl.col(TIME_COL).alias("spot_after_time"),
        pl.col("spot_bid").alias("spot_bid_100ms"),
        pl.col("spot_ask").alias("spot_ask_100ms"),
    ]).sort(["ValueCode", "spot_after_time"])
    feat = _join_asof(
        feat.sort(["ValueCode", "target_spot_100ms"]),
        spot_after,
        left_on="target_spot_100ms",
        right_on="spot_after_time",
        by="ValueCode",
        strategy="backward",
    )

    feat = feat.with_columns([
        (
            (pl.col("fut_bid1") > 0)
            & (pl.col("fut_bid1_50ms") >= pl.col("fut_bid1"))
        ).alias("fut_entry_success_50ms"),
        pl.when(pl.col("fut_bid1") > 0)
          .then((pl.col("fut_bid1") - pl.col("spot_ask_100ms")) / pl.col("fut_bid1"))
          .otherwise(None)
          .alias("lock_ret_100ms"),
        pl.when(pl.col("fut_bid1") > 0)
          .then((pl.col("spot_ask_100ms") - pl.col("spot_ask")) / pl.col("fut_bid1"))
          .otherwise(None)
          .alias("spot_ask_slippage_rate_100ms"),
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _feature_path(date)
    feat.write_parquet(out)
    print(
        f"{date}: features {feat.height:,} rows, "
        f"{mapping.height} ValueCode/QuoteCode pairs (near month {expected_suffix}) -> {out}"
    )
    return out


def build_features(args: argparse.Namespace) -> None:
    for date in _date_range(args.start_date, args.end_date):
        try:
            build_features_for_day(date, args)
        except Exception as exc:
            if args.skip_errors:
                print(f"{date}: skipped ({type(exc).__name__}: {exc})")
            else:
                raise


def _load_features(start: str, end: str | None) -> pl.DataFrame:
    frames = []
    for date in _date_range(start, end):
        path = _feature_path(date)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        raise FileNotFoundError("no feature files found; run the features command first")
    return pl.concat(frames, how="diagonal_relaxed")


def _threshold_entries(features: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for thr in THRESHOLDS:
        first = (
            features.filter((pl.col("ret_sell") >= thr) & (pl.col("fut_bid1") > 0))
            .sort(["Date", "ValueCode", TIME_COL, "QuoteCode"])
            .group_by(["Date", "ValueCode"], maintain_order=True)
            .first()
            .with_columns(pl.lit(thr).alias("threshold"))
        )
        if first.height:
            parts.append(first)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _add_settlement_cols(entries: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for date in entries["Date"].unique().to_list():
        settle = settlement_date(int(date)).strftime("%Y%m%d")
        d0 = datetime.strptime(date, "%Y%m%d").date()
        ds = datetime.strptime(settle, "%Y%m%d").date()
        rows.append((date, settle, (ds - d0).days))
    lookup = pl.DataFrame(rows, schema=["Date", "settlement_date", "days_to_settle"], orient="row")
    return entries.join(lookup, on="Date", how="left")


def _classify_convergence(entries: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    conv = (
        features.filter(pl.col("ret_buy") >= 0)
        .select([
            "QuoteCode",
            pl.col("Date").alias("converge_date"),
            pl.col(TIME_COL).alias("converge_time"),
        ])
        .sort(["QuoteCode", "converge_time"])
    )
    ent = entries.rename({TIME_COL: "entry_time"}).sort(["QuoteCode", "entry_time"])
    out = _join_asof(
        ent,
        conv,
        left_on="entry_time",
        right_on="converge_time",
        by="QuoteCode",
        strategy="forward",
    )
    out = _add_settlement_cols(out)
    out = out.with_columns([
        pl.when(pl.col("converge_date") == pl.col("Date"))
          .then(pl.lit("當日收斂"))
          .when(
              pl.col("converge_date").is_not_null()
              & (pl.col("converge_date") < pl.col("settlement_date"))
          )
          .then(pl.lit("跨日收斂"))
          .otherwise(pl.lit("沒收斂（結算日才收斂）"))
          .alias("status"),
        pl.when(pl.col("converge_date").is_not_null())
          .then(
              (
                  pl.col("converge_date").str.strptime(pl.Date, "%Y%m%d")
                  - pl.col("Date").str.strptime(pl.Date, "%Y%m%d")
              ).dt.total_days()
          )
          .otherwise(pl.col("days_to_settle"))
          .alias("convergence_days"),
    ])
    return out


def first_entry_analysis(args: argparse.Namespace) -> None:
    try:
        from mysql import StrategyMySQLLoader

        mysql = StrategyMySQLLoader()
        contract_calendar.set_trade_day_fn(mysql.is_trade_day)
    except Exception as exc:
        print(f"settlement calendar fallback to nominal third Wednesday ({type(exc).__name__}: {exc})")

    features = _load_features(args.start_date, args.end_date)
    entries = _threshold_entries(features)
    if entries.height == 0:
        print("no first entries found for configured thresholds")
        return
    entries = _classify_convergence(entries, features)

    detail_path = OUT_DIR / "first_entry_detail.parquet"
    entries.write_parquet(detail_path)

    total_by_thr = entries.group_by(["ValueCode", "threshold"]).agg(pl.len().alias("_total"))
    summary = (
        entries.group_by(["ValueCode", "threshold", "status"])
        .agg([
            pl.len().alias("筆數"),
            pl.col("convergence_days").mean().alias("平均收斂天數"),
            pl.col("fut_entry_success_50ms").mean().alias("期貨方進場成功率"),
            pl.col("lock_ret_100ms")
              .filter(pl.col("fut_entry_success_50ms"))
              .mean()
              .alias("成功樣本_鎖單價差率"),
            pl.col("spot_ask_slippage_rate_100ms")
              .filter(pl.col("fut_entry_success_50ms"))
              .mean()
              .alias("成功樣本_現貨A1滑價率"),
        ])
        .join(total_by_thr, on=["ValueCode", "threshold"], how="left")
        .with_columns((pl.col("筆數") / pl.col("_total")).alias("比例"))
        .drop("_total")
        .sort(["ValueCode", "threshold", "status"])
    )

    threshold_summary = (
        entries.group_by(["ValueCode", "threshold"])
        .agg([
            pl.len().alias("筆數"),
            pl.col("fut_entry_success_50ms").mean().alias("期貨方進場成功率"),
            pl.col("lock_ret_100ms")
              .filter(pl.col("fut_entry_success_50ms"))
              .mean()
              .alias("成功樣本_鎖單價差率"),
            pl.col("spot_ask_slippage_rate_100ms")
              .filter(pl.col("fut_entry_success_50ms"))
              .mean()
              .alias("成功樣本_現貨A1滑價率"),
        ])
        .sort(["ValueCode", "threshold"])
    )

    summary_path = OUT_DIR / "first_entry_status_summary.csv"
    threshold_path = OUT_DIR / "first_entry_threshold_summary.csv"
    summary.write_csv(summary_path)
    threshold_summary.write_csv(threshold_path)

    print(f"detail -> {detail_path}")
    print(f"status summary -> {summary_path}")
    print(f"threshold summary -> {threshold_path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=260):
        print("\n===== First-entry status summary =====")
        print(summary.with_columns([
            (pl.col("threshold") * 100).round(3).alias("threshold_pct"),
            (pl.col("比例") * 100).round(2).alias("比例_pct"),
            (pl.col("期貨方進場成功率") * 100).round(2).alias("期貨方進場成功率_pct"),
            (pl.col("成功樣本_鎖單價差率") * 100).round(4).alias("成功樣本_鎖單價差率_pct"),
            (pl.col("成功樣本_現貨A1滑價率") * 100).round(4).alias("成功樣本_現貨A1滑價率_pct"),
        ]).select(
            "ValueCode", "threshold_pct", "status", "筆數", "比例_pct", "平均收斂天數",
            "期貨方進場成功率_pct", "成功樣本_鎖單價差率_pct", "成功樣本_現貨A1滑價率_pct",
        ))


def _period_tag(start: str, end: str | None) -> str:
    return f"{start}_{end or start}"


def _position_control_detail_path(start: str, end: str | None) -> Path:
    return (
        OUT_DIR
        / f"first_entry_reentry_detail_{_period_tag(start, end)}"
          "_ref9_symbol_open_position_control_10m.parquet"
    )


def _volume_reentry_tag(start: str, end: str | None, volume_lots: int,
                        add_lots: int, max_notional: float) -> str:
    max_m = int(max_notional / 1_000_000)
    return (
        f"{_period_tag(start, end)}_ref9_symbol_open"
        f"_volume{volume_lots}_add{add_lots}_position{max_m}m"
    )


def _default_volume_additions_path(start: str, end: str | None) -> Path:
    tag = _volume_reentry_tag(start, end, volume_lots=10, add_lots=1, max_notional=10_000_000)
    return OUT_DIR / f"first_entry_volume_reentry_additions_{tag}.parquet"


def _backtest_tag(start: str, end: str | None, fee_bp: float, include_volume: bool) -> str:
    vol = "base_plus_volume10" if include_volume else "base_only"
    fee = str(fee_bp).replace(".", "p")
    return f"{_period_tag(start, end)}_ref9_symbol_open_{vol}_fee{fee}bp"


def _ref_limit_cols_expr(ref_col: str, price_cols: list[str]) -> pl.Expr:
    up = pl.col(ref_col) * (1 + REF_LIMIT_PCT)
    dn = pl.col(ref_col) * (1 - REF_LIMIT_PCT)
    ok = pl.col(ref_col).is_not_null() & (pl.col(ref_col) > 0)
    for col in price_cols:
        ok = ok & pl.col(col).is_not_null() & (pl.col(col) > dn) & (pl.col(col) < up)
    return ok


def _load_volume_trade_candidates(
    date: str,
    quote_codes: list[str],
    refs: pl.DataFrame,
    max_feature_stale_ms: float | None,
) -> pl.DataFrame:
    feature_path = _feature_path(date)
    future_path = _raw_future_path(date)
    if not feature_path.exists() or not future_path.exists() or not quote_codes:
        return pl.DataFrame()

    feature_cols = [
        "Date", "ValueCode", "QuoteCode", TIME_COL,
        "contract_size",
        "fut_bid1", "fut_ask1", "fut_bid", "fut_ask", "fut_bid1_lots",
        "spot_ask", "spot_bid",
        "ret_sell", "ret_buy",
        "target_fut_50ms", "fut_after_time", "fut_bid1_50ms", "fut_ask1_50ms",
        "target_spot_100ms", "spot_after_time", "spot_ask_100ms", "spot_bid_100ms",
        "fut_entry_success_50ms", "spot_trial_state", "spot_trial_state_100ms",
    ]
    available_feature_cols = set(pl.read_parquet_schema(feature_path).keys())
    feature_cols = [c for c in feature_cols if c in available_feature_cols]
    features = (
        pl.scan_parquet(feature_path)
        .filter(pl.col("QuoteCode").is_in(quote_codes))
        .select(feature_cols)
        .rename({TIME_COL: "feature_time"})
        .collect()
    )
    if features.height == 0:
        return pl.DataFrame()
    features = features.sort(["QuoteCode", "feature_time"])

    raw_cols = ["RecvTime", "TransTime", "QuoteCode", "FillLots", "TrialMatch", "contract_size"]
    available_raw_cols = set(pl.read_parquet_schema(future_path).keys())
    raw_cols = [c for c in raw_cols if c in available_raw_cols]
    trades = _normalize_time(pl.read_parquet(future_path, columns=raw_cols))
    trades = trades.filter(
        pl.col("QuoteCode").is_in(quote_codes)
        & (pl.col("FillLots") > 0)
        & _session_time_expr(TIME_COL)
    )
    if "TransTime" in trades.columns:
        trades = trades.filter(_session_time_expr("TransTime"))
    if "TrialMatch" in trades.columns:
        trades = trades.filter(pl.col("TrialMatch") == 0)
    if "contract_size" in trades.columns:
        trades = trades.filter(_is_standard_contract_expr())
    trades = (
        trades.select([
            "QuoteCode",
            pl.col(TIME_COL).alias("trade_time"),
            pl.col("FillLots").cast(pl.Int64),
        ])
        .sort(["QuoteCode", "trade_time"])
    )
    if trades.height == 0:
        return pl.DataFrame()

    joined = _join_asof(
        trades,
        features,
        left_on="trade_time",
        right_on="feature_time",
        by="QuoteCode",
        strategy="backward",
    )
    joined = joined.filter(pl.col("feature_time").is_not_null()).with_columns(
        ((pl.col("trade_time") - pl.col("feature_time")).dt.total_microseconds() / 1000)
        .alias("feature_gap_ms")
    )
    if max_feature_stale_ms is not None:
        joined = joined.filter(pl.col("feature_gap_ms") <= max_feature_stale_ms)

    ref_keys = refs.filter(pl.col("Date") == date).select([
        "Date", "ValueCode", "QuoteCode", "spot_ref_price", "fut_ref_price",
    ]).unique()
    joined = joined.join(ref_keys, on=["Date", "ValueCode", "QuoteCode"], how="inner")
    return (
        joined.filter(
            (pl.col("ret_sell") >= min(THRESHOLDS))
            & (pl.col("spot_trial_state").fill_null(1) == 0)
            & (pl.col("spot_trial_state_100ms").fill_null(1) == 0)
            & _ref_limit_cols_expr(
                "spot_ref_price",
                ["spot_bid", "spot_ask", "spot_bid_100ms", "spot_ask_100ms"],
            )
            & _ref_limit_cols_expr(
                "fut_ref_price",
                ["fut_bid1", "fut_ask1", "fut_bid1_50ms", "fut_ask1_50ms"],
            )
        )
        .select([
            "Date", "ValueCode", "QuoteCode", "trade_time", "feature_time",
            "feature_gap_ms", "FillLots", "contract_size",
            "fut_bid1", "fut_ask1", "fut_bid", "fut_ask", "fut_bid1_lots",
            "spot_ask", "spot_bid", "ret_sell", "ret_buy",
            "target_fut_50ms", "fut_after_time", "fut_bid1_50ms", "fut_ask1_50ms",
            "target_spot_100ms", "spot_after_time", "spot_ask_100ms", "spot_bid_100ms",
            "fut_entry_success_50ms", "spot_ref_price", "fut_ref_price",
        ])
        .sort(["QuoteCode", "trade_time"])
    )


def _spot_tick_size_expr(price_col: str) -> pl.Expr:
    price = pl.col(price_col)
    return (
        pl.when(price < 10).then(0.01)
        .when(price < 50).then(0.05)
        .when(price < 100).then(0.1)
        .when(price < 500).then(0.5)
        .when(price < 1000).then(1.0)
        .otherwise(5.0)
    )


def _summarize_volume_reentry(base: pl.DataFrame, additions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    base_summary = (
        base.group_by("threshold")
        .agg([
            pl.len().alias("base_samples"),
            pl.col("trade_fut_lots_10m").sum().alias("base_lots"),
            pl.col("position_10m_notional_twd").sum().alias("base_notional_twd"),
            pl.col("fut_entry_success_50ms").sum().alias("base_executable_samples"),
            pl.col("position_10m_notional_twd")
              .filter(pl.col("fut_entry_success_50ms"))
              .sum()
              .alias("base_executable_notional_twd"),
        ])
    )
    if additions.height:
        add_summary = (
            additions.group_by("threshold")
            .agg([
                pl.len().alias("add_samples"),
                pl.col("trade_fut_lots").sum().alias("add_lots"),
                pl.col("trade_notional_twd").sum().alias("add_notional_twd"),
                pl.col("fut_entry_success_50ms").sum().alias("add_executable_samples"),
                pl.col("trade_notional_twd")
                  .filter(pl.col("fut_entry_success_50ms"))
                  .sum()
                  .alias("add_executable_notional_twd"),
                pl.col("spot_slippage_bp")
                  .filter(pl.col("fut_entry_success_50ms"))
                  .mean()
                  .alias("add_exec_spot_slippage_bp_mean"),
                (
                    (
                        pl.col("spot_slippage_bp")
                        * pl.col("trade_notional_twd")
                    )
                    .filter(pl.col("fut_entry_success_50ms"))
                    .sum()
                    / pl.col("trade_notional_twd")
                      .filter(pl.col("fut_entry_success_50ms"))
                      .sum()
                ).alias("add_exec_spot_slippage_bp_amt_wavg"),
                pl.col("feature_gap_ms").mean().alias("feature_gap_ms_mean"),
                pl.col("feature_gap_ms").quantile(0.9).alias("feature_gap_ms_p90"),
            ])
        )
    else:
        add_summary = pl.DataFrame({
            "threshold": THRESHOLDS,
            "add_samples": [0] * len(THRESHOLDS),
            "add_lots": [0] * len(THRESHOLDS),
            "add_notional_twd": [0.0] * len(THRESHOLDS),
            "add_executable_samples": [0] * len(THRESHOLDS),
            "add_executable_notional_twd": [0.0] * len(THRESHOLDS),
            "add_exec_spot_slippage_bp_mean": [None] * len(THRESHOLDS),
            "add_exec_spot_slippage_bp_amt_wavg": [None] * len(THRESHOLDS),
            "feature_gap_ms_mean": [None] * len(THRESHOLDS),
            "feature_gap_ms_p90": [None] * len(THRESHOLDS),
        })
    summary = (
        base_summary.join(add_summary, on="threshold", how="full", coalesce=True)
        .with_columns([
            (pl.col("base_samples") + pl.col("add_samples")).alias("total_samples"),
            (pl.col("base_lots") + pl.col("add_lots")).alias("total_lots"),
            (pl.col("base_notional_twd") + pl.col("add_notional_twd")).alias("total_notional_twd"),
            (
                pl.col("base_executable_samples")
                + pl.col("add_executable_samples")
            ).alias("total_executable_samples"),
            (
                pl.col("base_executable_notional_twd")
                + pl.col("add_executable_notional_twd")
            ).alias("total_executable_notional_twd"),
        ])
        .sort("threshold")
    )

    if additions.height:
        status_summary = (
            additions.group_by(["threshold", "status_full"])
            .agg([
                pl.len().alias("add_samples"),
                pl.col("trade_notional_twd").sum().alias("add_notional_twd"),
                pl.col("fut_entry_success_50ms").sum().alias("add_executable_samples"),
                pl.col("trade_notional_twd")
                  .filter(pl.col("fut_entry_success_50ms"))
                  .sum()
                  .alias("add_executable_notional_twd"),
            ])
            .sort(["threshold", "status_full"])
        )
    else:
        status_summary = pl.DataFrame()
    return summary, status_summary


def _plot_volume_reentry(summary: pl.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plot ({type(exc).__name__}: {exc})")
        return

    pdf = summary.with_columns((pl.col("threshold") * 100).alias("threshold_pct")).to_pandas()
    labels = [f"{x:.2f}%" for x in pdf["threshold_pct"]]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, pdf["base_samples"], label="base")
    axes[0].bar(x, pdf["add_samples"], bottom=pdf["base_samples"], label="volume add")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_title("Samples")
    axes[0].legend()

    base_yi = pdf["base_notional_twd"] / 100_000_000
    add_yi = pdf["add_notional_twd"] / 100_000_000
    axes[1].bar(x, base_yi, label="base")
    axes[1].bar(x, add_yi, bottom=base_yi, label="volume add")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_title("Notional (100M TWD)")
    axes[1].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def volume_reentry_analysis(args: argparse.Namespace) -> None:
    base_path = Path(args.base_path) if args.base_path else _position_control_detail_path(
        args.start_date, args.end_date
    )
    if not base_path.exists():
        raise FileNotFoundError(f"base position-control detail not found: {base_path}")

    base = (
        pl.read_parquet(base_path)
        .filter(pl.col("position_control_accept"))
        .with_row_index("base_event_id")
    )
    period_start = datetime.strptime(args.start_date, "%Y%m%d")
    period_end_exclusive = datetime.strptime(args.end_date or args.start_date, "%Y%m%d") + timedelta(days=1)
    base = base.filter(
        (pl.col("entry_time") < period_end_exclusive)
        & (pl.col("position_release_time") >= period_start)
    )
    if base.height == 0:
        raise RuntimeError("no accepted base events in position-control detail")
    refs = base.select([
        "Date", "ValueCode", "QuoteCode", "spot_ref_price", "fut_ref_price",
    ]).unique()

    dates = _date_range(args.start_date, args.end_date)
    additions: list[dict] = []
    max_notional = float(args.max_notional)
    volume_lots = int(args.volume_lots)
    add_lots = int(args.add_lots)

    base_rows = base.sort(["threshold", "ValueCode", "entry_time"]).iter_rows(named=True)
    events = list(base_rows)
    candidates_by_date: dict[str, pl.DataFrame] = {}

    for date in dates:
        day0 = datetime.strptime(date, "%Y%m%d")
        day1 = day0 + timedelta(days=1)
        active_codes = sorted({
            row["QuoteCode"]
            for row in events
            if row["entry_time"] < day1 and row["position_release_time"] >= day0
        })
        if not active_codes:
            continue
        cand = _load_volume_trade_candidates(
            date,
            active_codes,
            refs,
            args.max_feature_stale_ms,
        )
        candidates_by_date[date] = cand
        print(f"{date}: volume trade candidates {cand.height:,} for {len(active_codes)} contracts")

    for row in events:
        entry_time = row["entry_time"]
        release_time = row["position_release_time"]
        threshold = float(row["threshold"])
        quote_code = row["QuoteCode"]
        value_code = row["ValueCode"]
        used_notional = float(row["position_10m_notional_twd"] or 0.0)
        next_trigger_lots = volume_lots
        cumulative_fill_lots = 0

        start_date = entry_time.strftime("%Y%m%d")
        end_date = release_time.strftime("%Y%m%d")
        for date in _date_range(start_date, end_date):
            cand = candidates_by_date.get(date)
            if cand is None or cand.height == 0:
                continue
            window = cand.filter(
                (pl.col("QuoteCode") == quote_code)
                & (pl.col("ValueCode") == value_code)
                & (pl.col("trade_time") > entry_time)
                & (pl.col("trade_time") < release_time)
                & (pl.col("ret_sell") >= threshold)
            )
            if window.height == 0:
                continue
            for c in window.iter_rows(named=True):
                cumulative_fill_lots += int(c["FillLots"] or 0)
                while cumulative_fill_lots >= next_trigger_lots:
                    trade_notional = float(c["fut_bid1"]) * STANDARD_CONTRACT_SIZE * add_lots
                    if used_notional + trade_notional <= max_notional:
                        used_notional += trade_notional
                        spot_tick = (
                            0.01 if c["spot_ask"] < 10 else
                            0.05 if c["spot_ask"] < 50 else
                            0.1 if c["spot_ask"] < 100 else
                            0.5 if c["spot_ask"] < 500 else
                            1.0 if c["spot_ask"] < 1000 else
                            5.0
                        )
                        spot_slip = float(c["spot_ask_100ms"]) - float(c["spot_ask"])
                        additions.append({
                            "base_event_id": row["base_event_id"],
                            "Date": c["Date"],
                            "base_Date": row["Date"],
                            "ValueCode": value_code,
                            "QuoteCode": quote_code,
                            "threshold": threshold,
                            "add_seq": next_trigger_lots // volume_lots,
                            "entry_time": c["trade_time"],
                            "feature_time": c["feature_time"],
                            "feature_gap_ms": c["feature_gap_ms"],
                            "FillLots": c["FillLots"],
                            "cumulative_fill_lots": cumulative_fill_lots,
                            "trigger_lots": next_trigger_lots,
                            "trade_fut_lots": add_lots,
                            "trade_notional_twd": trade_notional,
                            "open_notional_after_twd": used_notional,
                            "fut_bid1": c["fut_bid1"],
                            "fut_ask1": c["fut_ask1"],
                            "spot_ask": c["spot_ask"],
                            "spot_bid": c["spot_bid"],
                            "ret_sell": c["ret_sell"],
                            "target_fut_50ms": c["target_fut_50ms"],
                            "fut_after_time": c["fut_after_time"],
                            "fut_bid1_50ms": c["fut_bid1_50ms"],
                            "fut_ask1_50ms": c["fut_ask1_50ms"],
                            "target_spot_100ms": c["target_spot_100ms"],
                            "spot_after_time": c["spot_after_time"],
                            "spot_ask_100ms": c["spot_ask_100ms"],
                            "spot_bid_100ms": c["spot_bid_100ms"],
                            "fut_entry_success_50ms": c["fut_entry_success_50ms"],
                            "spot_slippage_ticks": spot_slip / spot_tick,
                            "spot_slippage_bp": spot_slip / float(c["spot_ask"]) * 10000,
                            "position_release_time": release_time,
                            "status_full": row["status_full"],
                        })
                    next_trigger_lots += volume_lots

    if additions:
        additions_df = pl.DataFrame(additions).with_columns([
            pl.col("fut_entry_success_50ms").cast(pl.Boolean),
            pl.col("trade_fut_lots").cast(pl.Int64),
        ])
    else:
        additions_df = pl.DataFrame(schema={
            "base_event_id": pl.UInt32,
            "Date": pl.Utf8,
            "base_Date": pl.Utf8,
            "ValueCode": pl.Utf8,
            "QuoteCode": pl.Utf8,
            "threshold": pl.Float64,
            "add_seq": pl.Int64,
            "entry_time": pl.Datetime("ns"),
            "trade_fut_lots": pl.Int64,
            "trade_notional_twd": pl.Float64,
            "fut_entry_success_50ms": pl.Boolean,
            "status_full": pl.Utf8,
        })

    tag = _volume_reentry_tag(
        args.start_date, args.end_date, volume_lots, add_lots, max_notional
    )
    additions_path = OUT_DIR / f"first_entry_volume_reentry_additions_{tag}.parquet"
    summary_path = OUT_DIR / f"first_entry_volume_reentry_summary_{tag}.csv"
    status_path = OUT_DIR / f"first_entry_volume_reentry_status_summary_{tag}.csv"
    plot_path = OUT_DIR / "plots" / f"first_entry_volume_reentry_{tag}.png"

    additions_df.write_parquet(additions_path)
    summary, status_summary = _summarize_volume_reentry(base, additions_df)
    summary.write_csv(summary_path)
    if status_summary.height:
        status_summary.write_csv(status_path)
    _plot_volume_reentry(summary, plot_path)

    print(f"additions -> {additions_path}")
    print(f"summary -> {summary_path}")
    if status_summary.height:
        print(f"status summary -> {status_path}")
    print(f"plot -> {plot_path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=240):
        print(summary.with_columns([
            (pl.col("threshold") * 100).round(3).alias("threshold_pct"),
            (pl.col("base_notional_twd") / 100_000_000).round(2).alias("base_億"),
            (pl.col("add_notional_twd") / 100_000_000).round(2).alias("add_億"),
            (pl.col("total_notional_twd") / 100_000_000).round(2).alias("total_億"),
        ]).select([
            "threshold_pct", "base_samples", "add_samples", "total_samples",
            "base_億", "add_億", "total_億",
            "base_executable_samples", "add_executable_samples", "total_executable_samples",
            "add_exec_spot_slippage_bp_mean", "add_exec_spot_slippage_bp_amt_wavg",
            "feature_gap_ms_mean", "feature_gap_ms_p90",
        ]))


def _base_backtest_entries(base_path: Path, start: str, end: str | None,
                           fee_bp: float) -> pl.DataFrame:
    if not base_path.exists():
        raise FileNotFoundError(f"base position-control detail not found: {base_path}")
    period_start = datetime.strptime(start, "%Y%m%d")
    period_end_exclusive = datetime.strptime(end or start, "%Y%m%d") + timedelta(days=1)
    return (
        pl.scan_parquet(base_path)
        .filter(
            pl.col("position_control_accept")
            & pl.col("fut_entry_success_50ms")
            & (pl.col("trade_fut_lots_10m") > 0)
            & pl.col("spot_ask_100ms").is_not_null()
            & (pl.col("entry_time") < period_end_exclusive)
            & (pl.col("position_release_time") >= period_start)
        )
        .select([
            pl.lit("base").alias("source"),
            pl.col("row_id").cast(pl.Int64).alias("source_event_id"),
            "Date", "ValueCode", "QuoteCode", "threshold", "status_full",
            "entry_time",
            pl.col("position_release_time").alias("exit_time"),
            pl.col("trade_fut_lots_10m").cast(pl.Int64).alias("fut_lots"),
            pl.col("fut_bid1").alias("entry_fut_sell_price"),
            pl.col("spot_ask_100ms").alias("entry_spot_buy_price"),
            pl.col("spot_ask").alias("signal_spot_ask"),
            pl.col("fut_bid1_50ms"),
            pl.col("target_fut_50ms"),
            pl.col("target_spot_100ms"),
        ])
        .collect()
        .with_columns(pl.lit(float(fee_bp)).alias("fee_bp"))
    )


def _volume_backtest_entries(additions_path: Path, start: str, end: str | None,
                             fee_bp: float) -> pl.DataFrame:
    if not additions_path.exists():
        return pl.DataFrame()
    period_start = datetime.strptime(start, "%Y%m%d")
    period_end_exclusive = datetime.strptime(end or start, "%Y%m%d") + timedelta(days=1)
    return (
        pl.scan_parquet(additions_path)
        .filter(
            pl.col("fut_entry_success_50ms")
            & (pl.col("trade_fut_lots") > 0)
            & pl.col("spot_ask_100ms").is_not_null()
            & (pl.col("entry_time") < period_end_exclusive)
            & (pl.col("position_release_time") >= period_start)
        )
        .select([
            pl.lit("volume_add").alias("source"),
            pl.col("base_event_id").cast(pl.Int64).alias("source_event_id"),
            "Date", "ValueCode", "QuoteCode", "threshold", "status_full",
            "entry_time",
            pl.col("position_release_time").alias("exit_time"),
            pl.col("trade_fut_lots").cast(pl.Int64).alias("fut_lots"),
            pl.col("fut_bid1").alias("entry_fut_sell_price"),
            pl.col("spot_ask_100ms").alias("entry_spot_buy_price"),
            pl.col("spot_ask").alias("signal_spot_ask"),
            pl.col("fut_bid1_50ms"),
            pl.col("target_fut_50ms"),
            pl.col("target_spot_100ms"),
        ])
        .collect()
        .with_columns(pl.lit(float(fee_bp)).alias("fee_bp"))
    )


def _add_backtest_pnl_cols(entries: pl.DataFrame) -> pl.DataFrame:
    if entries.height == 0:
        return entries
    return (
        entries
        .with_columns([
            pl.col("entry_time").dt.strftime("%Y%m%d").alias("entry_date"),
            pl.col("exit_time").dt.strftime("%Y%m%d").alias("exit_date"),
            (pl.col("fut_lots") * STANDARD_CONTRACT_SIZE).alias("shares"),
            (
                pl.col("entry_fut_sell_price")
                * pl.col("fut_lots")
                * STANDARD_CONTRACT_SIZE
            ).alias("fut_notional_twd"),
            (
                pl.col("entry_spot_buy_price")
                * pl.col("fut_lots")
                * STANDARD_CONTRACT_SIZE
            ).alias("entry_notional_twd"),
        ])
        .with_columns([
            (
                (pl.col("entry_fut_sell_price") - pl.col("entry_spot_buy_price"))
                / pl.col("entry_fut_sell_price")
                * 10000
            ).alias("entry_spread_bp_bid_basis"),
            (
                (pl.col("entry_fut_sell_price") - pl.col("entry_spot_buy_price"))
                / pl.col("entry_spot_buy_price")
                * 10000
            ).alias("entry_spread_bp_ask_basis"),
            (
                (pl.col("entry_fut_sell_price") - pl.col("entry_spot_buy_price"))
                * pl.col("fut_lots")
                * STANDARD_CONTRACT_SIZE
            ).alias("gross_pnl_twd"),
            (
                pl.col("fee_bp")
                / 10000
                * pl.col("entry_spot_buy_price")
                * pl.col("fut_lots")
                * STANDARD_CONTRACT_SIZE
            ).alias("fee_twd"),
        ])
        .with_columns([
            (pl.col("gross_pnl_twd") - pl.col("fee_twd")).alias("net_pnl_twd"),
            (
                (pl.col("gross_pnl_twd") - pl.col("fee_twd"))
                / pl.col("entry_notional_twd")
                * 10000
            ).alias("net_pnl_bp_spot_notional"),
        ])
        .sort(["threshold", "entry_time", "ValueCode", "source"])
    )


def _build_backtest_daily(entries: pl.DataFrame, start: str, end: str | None) -> pl.DataFrame:
    dates = _date_range(start, end)
    thresholds = sorted(entries["threshold"].unique().to_list()) if entries.height else THRESHOLDS
    realized = (
        entries.group_by(["threshold", "exit_date"])
        .agg([
            pl.len().alias("exit_trades"),
            pl.col("net_pnl_twd").sum().alias("daily_net_pnl_twd"),
            pl.col("gross_pnl_twd").sum().alias("daily_gross_pnl_twd"),
            pl.col("fee_twd").sum().alias("daily_fee_twd"),
            pl.col("entry_notional_twd").sum().alias("exit_notional_twd"),
        ])
        .rename({"exit_date": "Date"})
    ) if entries.height else pl.DataFrame()
    entered = (
        entries.group_by(["threshold", "entry_date"])
        .agg([
            pl.len().alias("entry_trades"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
        ])
        .rename({"entry_date": "Date"})
    ) if entries.height else pl.DataFrame()

    entry_rows = entries.select([
        "threshold", "entry_time", "exit_time", "entry_notional_twd"
    ]).to_dicts()
    rows = []
    for threshold in thresholds:
        threshold_rows = [r for r in entry_rows if r["threshold"] == threshold]
        for date in dates:
            day_start = datetime.strptime(date, "%Y%m%d")
            day_end = day_start.replace(hour=13, minute=25)
            start_open = sum(
                r["entry_notional_twd"]
                for r in threshold_rows
                if r["entry_time"] < day_start and r["exit_time"] > day_start
            )
            deltas = []
            for r in threshold_rows:
                if day_start <= r["entry_time"] <= day_end:
                    deltas.append((r["entry_time"], r["entry_notional_twd"]))
                if day_start <= r["exit_time"] <= day_end:
                    deltas.append((r["exit_time"], -r["entry_notional_twd"]))
            open_now = start_open
            peak = start_open
            for _, delta in sorted(deltas, key=lambda x: x[0]):
                open_now += delta
                peak = max(peak, open_now)
            eod_open = sum(
                r["entry_notional_twd"]
                for r in threshold_rows
                if r["entry_time"] <= day_end and r["exit_time"] > day_end
            )
            rows.append({
                "Date": date,
                "threshold": threshold,
                "eod_open_notional_twd": eod_open,
                "peak_open_notional_twd": peak,
            })
    pos = pl.DataFrame(rows) if rows else pl.DataFrame()
    daily = pos
    if realized.height:
        daily = daily.join(realized, on=["threshold", "Date"], how="left")
    if entered.height:
        daily = daily.join(entered, on=["threshold", "Date"], how="left")
    fill_cols = [
        "exit_trades", "daily_net_pnl_twd", "daily_gross_pnl_twd", "daily_fee_twd",
        "exit_notional_twd", "entry_trades", "entry_notional_twd",
    ]
    daily = daily.with_columns([
        pl.col(c).fill_null(0) for c in fill_cols if c in daily.columns
    ])
    return (
        daily.sort(["threshold", "Date"])
        .with_columns(
            pl.col("daily_net_pnl_twd")
            .cum_sum()
            .over("threshold")
            .alias("cum_net_pnl_twd")
        )
    )


def _plot_backtest_daily(daily: pl.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plot ({type(exc).__name__}: {exc})")
        return
    pdf = daily.with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d")
    ).to_pandas()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for threshold, group in pdf.groupby("threshold"):
        label = f"{threshold * 100:.2f}%"
        axes[0].plot(group["Date"], group["cum_net_pnl_twd"] / 1e8, label=label)
        axes[1].plot(group["Date"], group["eod_open_notional_twd"] / 1e8, label=label)
    axes[0].set_title("Cumulative Realized Net PnL")
    axes[0].set_ylabel("100M TWD")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_title("End-of-day Open Notional")
    axes[1].set_ylabel("100M TWD")
    axes[1].grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def backtest_analysis(args: argparse.Namespace) -> None:
    base_path = Path(args.base_path) if args.base_path else _position_control_detail_path(
        args.start_date, args.end_date
    )
    additions_path = (
        Path(args.additions_path)
        if args.additions_path
        else _default_volume_additions_path(args.start_date, args.end_date)
    )
    base = _base_backtest_entries(base_path, args.start_date, args.end_date, args.fee_bp)
    parts = [base]
    include_volume = not args.no_volume_additions and additions_path.exists()
    if include_volume:
        additions = _volume_backtest_entries(
            additions_path, args.start_date, args.end_date, args.fee_bp
        )
        if additions.height:
            parts.append(additions)
    entries = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    entries = _add_backtest_pnl_cols(entries)
    if entries.height == 0:
        print("no executable entries for backtest")
        return

    daily = _build_backtest_daily(entries, args.start_date, args.end_date)
    tag = _backtest_tag(
        args.start_date, args.end_date, args.fee_bp, include_volume
    )
    trades_path = OUT_DIR / f"backtest_trades_{tag}.parquet"
    trades_csv_path = OUT_DIR / f"backtest_trades_{tag}.csv"
    daily_path = OUT_DIR / f"backtest_daily_{tag}.csv"
    summary_path = OUT_DIR / f"backtest_summary_{tag}.csv"
    plot_path = OUT_DIR / "plots" / f"backtest_daily_{tag}_en.png"

    entries.write_parquet(trades_path)
    entries.write_csv(trades_csv_path)
    daily.write_csv(daily_path)
    summary = (
        entries.group_by("threshold")
        .agg([
            pl.len().alias("trades"),
            pl.col("fut_lots").sum().alias("fut_lots"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            pl.col("gross_pnl_twd").sum().alias("gross_pnl_twd"),
            pl.col("fee_twd").sum().alias("fee_twd"),
            pl.col("net_pnl_twd").sum().alias("net_pnl_twd"),
            (
                pl.col("net_pnl_twd").sum() / pl.col("entry_notional_twd").sum() * 10000
            ).alias("net_pnl_bp"),
            pl.col("entry_spread_bp_bid_basis").mean().alias("entry_spread_bp_mean"),
            (
                (pl.col("entry_spread_bp_bid_basis") * pl.col("entry_notional_twd")).sum()
                / pl.col("entry_notional_twd").sum()
            ).alias("entry_spread_bp_amt_wavg"),
        ])
        .sort("threshold")
    )
    summary.write_csv(summary_path)
    _plot_backtest_daily(daily, plot_path)

    print(f"trades parquet -> {trades_path}")
    print(f"trades csv -> {trades_csv_path}")
    print(f"daily -> {daily_path}")
    print(f"summary -> {summary_path}")
    print(f"plot -> {plot_path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=240):
        print(summary.with_columns([
            (pl.col("threshold") * 100).round(3).alias("threshold_pct"),
            (pl.col("entry_notional_twd") / 1e8).round(2).alias("notional_億"),
            (pl.col("gross_pnl_twd") / 1e4).round(2).alias("gross_萬"),
            (pl.col("fee_twd") / 1e4).round(2).alias("fee_萬"),
            (pl.col("net_pnl_twd") / 1e4).round(2).alias("net_萬"),
            pl.col("net_pnl_bp").round(2),
            pl.col("entry_spread_bp_mean").round(2),
            pl.col("entry_spread_bp_amt_wavg").round(2),
        ]).select([
            "threshold_pct", "trades", "fut_lots", "notional_億",
            "gross_萬", "fee_萬", "net_萬", "net_pnl_bp",
            "entry_spread_bp_mean", "entry_spread_bp_amt_wavg",
        ]))


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stock futures/spot arbitrage tick analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="cache stock-futures ticks under data/stockfuture")
    fetch.add_argument("-s", "--start-date", required=True)
    fetch.add_argument("-e", "--end-date")
    fetch.add_argument("--fut-code", help="comma-separated futures QuoteCode filter")
    fetch.add_argument("--force", action="store_true")

    feat = sub.add_parser("features", help="build tick-level arbitrage features")
    feat.add_argument("-s", "--start-date", required=True)
    feat.add_argument("-e", "--end-date")
    feat.add_argument("--code", help="comma-separated stock ValueCode filter")
    feat.add_argument("--fut-code", help="comma-separated futures QuoteCode filter")
    feat.add_argument("--force-fetch", action="store_true")
    feat.add_argument("--skip-errors", action="store_true")
    feat.add_argument("--all-months", dest="near_month_only", action="store_false")
    feat.set_defaults(near_month_only=True)

    first = sub.add_parser("first", help="first-entry threshold/status analysis")
    first.add_argument("-s", "--start-date", required=True)
    first.add_argument("-e", "--end-date")

    volume = sub.add_parser(
        "volume-reentry",
        help="add one-lot entries whenever futures trades accumulate by N lots while divergence persists",
    )
    volume.add_argument("-s", "--start-date", required=True)
    volume.add_argument("-e", "--end-date")
    volume.add_argument("--base-path", help="position-control detail parquet")
    volume.add_argument("--volume-lots", type=int, default=10)
    volume.add_argument("--add-lots", type=int, default=1)
    volume.add_argument("--max-notional", type=float, default=10_000_000)
    volume.add_argument(
        "--max-feature-stale-ms",
        type=float,
        default=100.0,
        help="drop trade ticks whose latest feature state is older than this many ms; use -1 to disable",
    )

    backtest = sub.add_parser(
        "backtest",
        help="build executable-entry trade ledger and daily PnL/position curves",
    )
    backtest.add_argument("-s", "--start-date", required=True)
    backtest.add_argument("-e", "--end-date")
    backtest.add_argument("--base-path", help="position-control detail parquet")
    backtest.add_argument("--additions-path", help="volume-reentry additions parquet")
    backtest.add_argument("--fee-bp", type=float, default=38.0)
    backtest.add_argument("--no-volume-additions", action="store_true")

    return p


def main() -> None:
    args = arg_parser().parse_args()
    if args.cmd == "fetch":
        fetch_samples(args)
    elif args.cmd == "features":
        build_features(args)
    elif args.cmd == "first":
        first_entry_analysis(args)
    elif args.cmd == "volume-reentry":
        if args.max_feature_stale_ms is not None and args.max_feature_stale_ms < 0:
            args.max_feature_stale_ms = None
        volume_reentry_analysis(args)
    elif args.cmd == "backtest":
        backtest_analysis(args)


if __name__ == "__main__":
    main()
