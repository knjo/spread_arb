"""Previous-day basis baseline for futures/spot arbitrage filters.

The filter tested here is:

    entry spread bp > max(previous-day mean basis bp, 0) + threshold bp

For normal days the previous-day basis is read from the previous day's
near-month feature file.  On the first trading day after settlement, today's
near-month contract was yesterday's next-month contract, so this script fetches
that specific previous-day futures contract and aligns it to previous-day spot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

import arbitrage_analysis as aa


from data_paths import STOCKFUTURE_DIR as OUT_DIR, scan_futures_ticks

BASIS_FUT_DIR = OUT_DIR / "prev_basis_futures"


def _feature_dates(start: str, end: str | None) -> list[str]:
    end = end or start
    out = []
    for path in sorted(OUT_DIR.glob("*_arbitrage_features.parquet")):
        date = path.name[:8]
        if start <= date <= end:
            out.append(date)
    return out


def _all_feature_dates() -> list[str]:
    return sorted(path.name[:8] for path in OUT_DIR.glob("*_arbitrage_features.parquet"))


def _prev_feature_date(date: str, all_dates: list[str]) -> str | None:
    prev = [d for d in all_dates if d < date]
    return prev[-1] if prev else None


def _current_mapping(date: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(aa._feature_path(date))
        .select(["Date", "ValueCode", "QuoteCode"])
        .unique()
        .collect()
        .sort(["ValueCode", "QuoteCode"])
    )


def _feature_basis_means(dates: list[str]) -> pl.DataFrame:
    paths = [aa._feature_path(date) for date in dates if aa._feature_path(date).exists()]
    if not paths:
        return pl.DataFrame()
    return (
        pl.scan_parquet(paths)
        .select(["Date", "ValueCode", "QuoteCode", "ret_sell"])
        .group_by(["Date", "ValueCode", "QuoteCode"])
        .agg([
            (pl.col("ret_sell").mean() * 10000).alias("prev_basis_bp_mean"),
            pl.len().alias("prev_basis_rows"),
        ])
        .collect()
    )


def _baseline_from_previous_feature(date: str, prev_date: str, mapping: pl.DataFrame) -> pl.DataFrame:
    quote_codes = mapping["QuoteCode"].to_list()
    prev_feature = aa._feature_path(prev_date)
    if not prev_feature.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(prev_feature)
        .filter(pl.col("QuoteCode").is_in(quote_codes))
        .group_by(["ValueCode", "QuoteCode"])
        .agg([
            (pl.col("ret_sell").mean() * 10000).alias("prev_basis_bp_mean"),
            pl.len().alias("prev_basis_rows"),
        ])
        .collect()
        .with_columns([
            pl.lit(date).alias("Date"),
            pl.lit(prev_date).alias("prev_date"),
            pl.lit("prev_near_feature").alias("basis_source"),
        ])
        .select([
            "Date", "prev_date", "ValueCode", "QuoteCode",
            "prev_basis_bp_mean", "prev_basis_rows", "basis_source",
        ])
    )


def _fetch_previous_futures_for_current_contracts(
    date: str,
    prev_date: str,
    quote_codes: list[str],
) -> Path:
    BASIS_FUT_DIR.mkdir(parents=True, exist_ok=True)
    path = BASIS_FUT_DIR / f"{prev_date}_for_{date}_stockfuture.parquet"
    if path.exists():
        return path
    print(f"{date}: fetch previous-day current contracts {len(quote_codes)} codes on {prev_date}")
    raw = scan_futures_ticks(prev_date, quote_codes).collect()
    if raw.height == 0:
        raw.write_parquet(path)
        return path
    raw = aa._join_futures_basic(raw, prev_date)
    raw = raw.filter(
        pl.col("QuoteCode").is_in(quote_codes)
        & (pl.col("QuoteCode").str.slice(2, 1) == "F")
        & pl.col("contract_size").is_not_null()
        & aa._is_standard_contract_expr()
    )
    raw.write_parquet(path)
    return path


def _baseline_from_previous_far_contract(date: str, prev_date: str, mapping: pl.DataFrame) -> pl.DataFrame:
    quote_codes = mapping["QuoteCode"].to_list()
    path = _fetch_previous_futures_for_current_contracts(date, prev_date, quote_codes)
    if not path.exists():
        return pl.DataFrame()
    raw = pl.read_parquet(path)
    if raw.height == 0:
        return pl.DataFrame()

    fut = aa._restore_prices(aa._normalize_time(raw), aa.FUT_SCALE)
    if "TrialMatch" in fut.columns:
        fut = fut.filter(pl.col("TrialMatch") == 0)
    fut = fut.filter(aa._session_time_expr(aa.TIME_COL))
    if "TransTime" in fut.columns:
        fut = fut.filter(aa._session_time_expr("TransTime"))
    fut = (
        aa._add_futures_best_quotes(fut.filter(aa._has_book_expr()))
        .select([
            "ValueCode", "QuoteCode", aa.TIME_COL,
            "fut_bid", "fut_ask",
        ])
        .filter((pl.col("fut_bid") > 0) & (pl.col("fut_ask") > 0))
        .sort(["ValueCode", aa.TIME_COL])
    )
    if fut.height == 0:
        return pl.DataFrame()

    value_codes = sorted(mapping["ValueCode"].unique().to_list())
    tradable = aa._load_market_tradable(prev_date)
    spot_ref = tradable.select(["ValueCode", "spot_ref_price"]).filter(
        pl.col("spot_ref_price").is_not_null() & (pl.col("spot_ref_price") > 0)
    )
    spot = aa._prepare_spot(prev_date, value_codes, False, spot_ref).rename({
        aa.TIME_COL: "spot_time"
    })
    if spot.height == 0:
        return pl.DataFrame()

    aligned = aa._join_asof(
        fut,
        spot,
        left_on=aa.TIME_COL,
        right_on="spot_time",
        by="ValueCode",
        strategy="backward",
        tolerance="60s",
    ).filter(
        (pl.col("spot_ask") > 0)
        & (pl.col("fut_bid") > 0)
        & (pl.col("fut_ask") > 0)
    )
    if aligned.height == 0:
        return pl.DataFrame()
    return (
        aligned.with_columns(
            ((pl.col("fut_bid") - pl.col("spot_ask")) / pl.col("fut_ask") * 10000)
            .alias("basis_bp")
        )
        .group_by(["ValueCode", "QuoteCode"])
        .agg([
            pl.col("basis_bp").mean().alias("prev_basis_bp_mean"),
            pl.len().alias("prev_basis_rows"),
        ])
        .with_columns([
            pl.lit(date).alias("Date"),
            pl.lit(prev_date).alias("prev_date"),
            pl.lit("prev_far_fetch").alias("basis_source"),
        ])
        .select([
            "Date", "prev_date", "ValueCode", "QuoteCode",
            "prev_basis_bp_mean", "prev_basis_rows", "basis_source",
        ])
    )


def build_prev_basis(start: str, end: str | None) -> pl.DataFrame:
    dates = _feature_dates(start, end)
    all_dates = _all_feature_dates()
    prev_dates = sorted({d for date in dates if (d := _prev_feature_date(date, all_dates))})
    print(f"scan previous feature basis once for {len(prev_dates)} dates", flush=True)
    feature_basis = _feature_basis_means(prev_dates)
    parts = []
    for date in dates:
        prev_date = _prev_feature_date(date, all_dates)
        if prev_date is None:
            print(f"{date}: no previous feature date; baseline missing", flush=True)
            continue
        mapping = _current_mapping(date)
        if feature_basis.height:
            prev_near = (
                mapping.join(
                    feature_basis.filter(pl.col("Date") == prev_date).drop("Date"),
                    on=["ValueCode", "QuoteCode"],
                    how="inner",
                )
                .with_columns([
                    pl.lit(prev_date).alias("prev_date"),
                    pl.lit("prev_near_feature").alias("basis_source"),
                ])
                .select([
                    "Date", "prev_date", "ValueCode", "QuoteCode",
                    "prev_basis_bp_mean", "prev_basis_rows", "basis_source",
                ])
            )
        else:
            prev_near = _baseline_from_previous_feature(date, prev_date, mapping)
        got_codes = set(prev_near["QuoteCode"].to_list()) if prev_near.height else set()
        missing = mapping.filter(~pl.col("QuoteCode").is_in(got_codes))
        parts_for_date = []
        if prev_near.height:
            parts_for_date.append(prev_near)
        if missing.height:
            far = _baseline_from_previous_far_contract(date, prev_date, missing)
            if far.height:
                parts_for_date.append(far)
        if parts_for_date:
            day = pl.concat(parts_for_date, how="diagonal_relaxed")
            parts.append(day)
            miss_n = mapping.height - day.select(["ValueCode", "QuoteCode"]).unique().height
            print(
                f"{date}: prev basis {day.height:,}/{mapping.height:,} pairs "
                f"from {prev_date}; missing={miss_n}"
                ,
                flush=True,
            )
        else:
            print(f"{date}: no previous basis collected", flush=True)
    if not parts:
        return pl.DataFrame()
    return (
        pl.concat(parts, how="diagonal_relaxed")
        .with_columns([
            pl.col("prev_basis_bp_mean").clip(0, None).alias("prev_basis_bp_clip0"),
        ])
        .sort(["Date", "ValueCode", "QuoteCode"])
    )


def adjusted_backtest_summary(basis: pl.DataFrame, start: str, end: str | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    trade_path = OUT_DIR / (
        f"backtest_trades_{start}_{end or start}_ref9_symbol_open_"
        "base_plus_volume10_fee38p0bp.parquet"
    )
    if not trade_path.exists():
        raise FileNotFoundError(f"backtest trade ledger missing: {trade_path}")
    trades = pl.read_parquet(trade_path)
    joined = (
        trades.join(
            basis.select([
                "Date", "ValueCode", "QuoteCode",
                "prev_date", "prev_basis_bp_mean", "prev_basis_bp_clip0", "basis_source",
            ]),
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
        )
        .with_columns([
            pl.col("prev_basis_bp_clip0").fill_null(0).alias("prev_basis_bp_clip0"),
            (pl.col("threshold") * 10000).alias("threshold_bp"),
        ])
        .with_columns([
            (pl.col("prev_basis_bp_clip0") + pl.col("threshold_bp"))
            .alias("adjusted_required_bp"),
            (
                pl.col("entry_spread_bp_bid_basis")
                >= (pl.col("prev_basis_bp_clip0") + pl.col("threshold_bp"))
            ).alias("pass_adjusted_basis_filter"),
        ])
    )
    summary = (
        joined.group_by("threshold")
        .agg([
            pl.len().alias("original_trades"),
            pl.col("entry_notional_twd").sum().alias("original_notional_twd"),
            pl.col("pass_adjusted_basis_filter").sum().alias("kept_trades"),
            pl.col("entry_notional_twd")
              .filter(pl.col("pass_adjusted_basis_filter"))
              .sum()
              .alias("kept_notional_twd"),
            pl.col("net_pnl_twd")
              .filter(pl.col("pass_adjusted_basis_filter"))
              .sum()
              .alias("kept_net_pnl_twd"),
            pl.col("prev_basis_bp_clip0").mean().alias("avg_prev_basis_bp_clip0"),
            pl.col("adjusted_required_bp").mean().alias("avg_adjusted_required_bp"),
        ])
        .with_columns([
            (pl.col("threshold") * 100).round(2).alias("threshold_pct"),
            (pl.col("kept_trades") / pl.col("original_trades") * 100).alias("kept_trade_pct"),
            (pl.col("kept_notional_twd") / pl.col("original_notional_twd") * 100)
            .alias("kept_notional_pct"),
            (pl.col("original_notional_twd") / 1e8).alias("original_notional_yi"),
            (pl.col("kept_notional_twd") / 1e8).alias("kept_notional_yi"),
            (pl.col("kept_net_pnl_twd") / 1e4).alias("kept_net_pnl_wan"),
            (pl.col("kept_net_pnl_twd") / pl.col("kept_notional_twd") * 10000)
            .alias("kept_net_pnl_bp"),
        ])
        .sort("threshold")
    )
    return joined, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start-date", required=True)
    parser.add_argument("-e", "--end-date")
    args = parser.parse_args()

    tag = f"{args.start_date}_{args.end_date or args.start_date}"
    basis = build_prev_basis(args.start_date, args.end_date)
    basis_path = OUT_DIR / f"previous_day_basis_baseline_{tag}.parquet"
    basis_csv = OUT_DIR / f"previous_day_basis_baseline_{tag}.csv"
    basis.write_parquet(basis_path)
    basis.write_csv(basis_csv)
    print(f"basis parquet -> {basis_path}")
    print(f"basis csv -> {basis_csv}")

    adjusted, summary = adjusted_backtest_summary(basis, args.start_date, args.end_date)
    detail_path = OUT_DIR / f"backtest_trades_adjusted_prev_basis_filter_{tag}.parquet"
    summary_path = OUT_DIR / f"backtest_summary_adjusted_prev_basis_filter_{tag}.csv"
    adjusted.write_parquet(detail_path)
    summary.write_csv(summary_path)
    print(f"adjusted detail -> {detail_path}")
    print(f"adjusted summary -> {summary_path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=240):
        print(summary.select([
            "threshold_pct", "original_trades", "kept_trades",
            "kept_trade_pct", "original_notional_yi", "kept_notional_yi",
            "kept_notional_pct", "avg_prev_basis_bp_clip0",
            "avg_adjusted_required_bp", "kept_net_pnl_wan", "kept_net_pnl_bp",
        ]).with_columns([
            pl.col("kept_trade_pct").round(2),
            pl.col("original_notional_yi").round(2),
            pl.col("kept_notional_yi").round(2),
            pl.col("kept_notional_pct").round(2),
            pl.col("avg_prev_basis_bp_clip0").round(2),
            pl.col("avg_adjusted_required_bp").round(2),
            pl.col("kept_net_pnl_wan").round(2),
            pl.col("kept_net_pnl_bp").round(2),
        ]))


if __name__ == "__main__":
    main()
