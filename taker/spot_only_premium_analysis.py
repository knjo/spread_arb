"""Spot-only outcomes after futures-premium signals.

Policy:
* entry: buy spot at the signal spot A1 (`spot_ask` / `signal_spot_ask`)
* next-day exits: sell at next trading day's open or close from marketData
* convergence exit: sell passively at spot A1 at the existing convergence /
  settlement release timestamp, reconstructed from arbitrage feature files

Costs are intentionally excluded; this is a gross return diagnostic.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from spread_arb.contract import settlement_date, to_date


from data_paths import MARKET_DIR, STOCKFUTURE_DIR as DATA_DIR
START = "20260126"
END = "20260629"
CONTRACT_SIZE = 2000

BASE_PATH = DATA_DIR / "first_entry_reentry_detail_20260126_20260629_ref9_symbol_open_position_control_10m.parquet"
ADD_PATH = DATA_DIR / "first_entry_volume_reentry_additions_20260126_20260629_ref9_symbol_open_volume10_add1_position10m.parquet"
LEDGER_PATH = DATA_DIR / "backtest_trades_20260126_20260629_ref9_symbol_open_base_plus_volume10_fee38p0bp.parquet"

OUT_DETAIL = DATA_DIR / f"spot_only_premium_detail_{START}_{END}.parquet"
OUT_SUMMARY = DATA_DIR / f"spot_only_premium_summary_{START}_{END}.csv"
OUT_REPORT = DATA_DIR / f"spot_only_premium_report_{START}_{END}.md"
OUT_OVERNIGHT_REPORT = DATA_DIR / f"spot_only_overnight_actionable_report_{START}_{END}.md"
OUT_OVERNIGHT_SUMMARY = DATA_DIR / f"spot_only_overnight_actionable_summary_{START}_{END}.csv"


def _date_files(directory: Path, suffix: str) -> list[str]:
    out = []
    for p in directory.glob(f"*{suffix}"):
        name = p.name
        if len(name) >= 8 and name[:8].isdigit():
            out.append(name[:8])
    return sorted(set(out))


def _next_market_date_map() -> pl.DataFrame:
    dates = _date_files(MARKET_DIR, "_marketData.parquet")
    rows = [(d, next((x for x in dates if x > d), None)) for d in dates]
    return pl.DataFrame(rows, schema=["Date", "next_trade_date"], orient="row")


def _load_market_prices(dates: list[str]) -> pl.DataFrame:
    parts = []
    for date in sorted(set(d for d in dates if d)):
        path = MARKET_DIR / f"{date}_marketData.parquet"
        if not path.exists():
            continue
        parts.append(
            pl.scan_parquet(path)
            .select(
                pl.lit(date).alias("next_trade_date"),
                pl.col("quote_code").cast(pl.Utf8).alias("ValueCode"),
                pl.col("open_price").cast(pl.Float64).alias("next_open_price"),
                pl.col("close_price").cast(pl.Float64).alias("next_close_price"),
            )
            .filter(
                (pl.col("next_open_price") > 0)
                & (pl.col("next_close_price") > 0)
            )
            .collect()
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def _load_all_stage3_attempts() -> pl.DataFrame:
    base = (
        pl.read_parquet(BASE_PATH)
        .filter(
            pl.col("position_control_accept")
            & (pl.col("trade_fut_lots_10m") > 0)
        )
        .select(
            pl.lit("all_stage3_attempts").alias("scope"),
            pl.lit("base_or_reentry").alias("source"),
            "Date",
            "ValueCode",
            "QuoteCode",
            "threshold",
            "status_full",
            pl.col("entry_time").dt.cast_time_unit("us").alias("entry_time"),
            pl.col("position_release_time").dt.cast_time_unit("us").alias("exit_time"),
            pl.col("trade_fut_lots_10m").cast(pl.Int64).alias("fut_lots_equiv"),
            pl.col("spot_ask").cast(pl.Float64).alias("entry_spot_price"),
        )
    )
    adds = (
        pl.read_parquet(ADD_PATH)
        .filter(pl.col("trade_fut_lots") > 0)
        .select(
            pl.lit("all_stage3_attempts").alias("scope"),
            pl.lit("volume_add").alias("source"),
            "Date",
            "ValueCode",
            "QuoteCode",
            "threshold",
            "status_full",
            pl.col("entry_time").dt.cast_time_unit("us").alias("entry_time"),
            pl.col("position_release_time").dt.cast_time_unit("us").alias("exit_time"),
            pl.col("trade_fut_lots").cast(pl.Int64).alias("fut_lots_equiv"),
            pl.col("spot_ask").cast(pl.Float64).alias("entry_spot_price"),
        )
    )
    return pl.concat([base, adds], how="diagonal_relaxed").with_row_index("spot_trade_id")


def _load_executable_ledger() -> pl.DataFrame:
    return (
        pl.read_parquet(LEDGER_PATH)
        .select(
            pl.lit("futures_first_executable_ledger").alias("scope"),
            "source",
            "Date",
            "ValueCode",
            "QuoteCode",
            "threshold",
            "status_full",
            "entry_time",
            "exit_time",
            pl.col("fut_lots").cast(pl.Int64).alias("fut_lots_equiv"),
            pl.col("signal_spot_ask").cast(pl.Float64).alias("entry_spot_price"),
        )
        .with_row_index("spot_trade_id")
    )


def _attach_next_day_prices(df: pl.DataFrame) -> pl.DataFrame:
    mapped = df.join(_next_market_date_map(), on="Date", how="left")
    dates = mapped["next_trade_date"].drop_nulls().unique().to_list()
    prices = _load_market_prices(dates)
    if prices.is_empty():
        return mapped.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("next_open_price"),
            pl.lit(None, dtype=pl.Float64).alias("next_close_price"),
        )
    return mapped.join(prices, on=["next_trade_date", "ValueCode"], how="left")


def _add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    trade_dates = sorted(df["Date"].unique().to_list())
    first = to_date(trade_dates[0]) - timedelta(days=70)
    last = to_date(trade_dates[-1]) + timedelta(days=70)
    settlement_dates = sorted(
        {
            settlement_date(first + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((last - first).days + 1)
        }
    )
    rows = []
    for date in trade_dates:
        d0 = to_date(date)
        cur_settle = settlement_date(date)
        prior_settles = [to_date(d) for d in settlement_dates if d < date]
        prev_settle = max(prior_settles) if prior_settles else None
        calendar_days_after_prev_settle = None if prev_settle is None else (d0 - prev_settle).days
        calendar_days_to_settle = (cur_settle - d0).days
        rows.append((date, calendar_days_after_prev_settle, calendar_days_to_settle))

    cal = pl.DataFrame(
        rows,
        schema=["Date", "calendar_days_after_prev_settle", "calendar_days_to_settle"],
        orient="row",
    )
    return (
        df.join(cal, on="Date", how="left")
        .with_columns(
            pl.when(pl.col("calendar_days_after_prev_settle").is_null())
            .then(pl.lit("pre_first_settle"))
            .when(pl.col("calendar_days_after_prev_settle") <= 2)
            .then(pl.lit("D0-D2_after_prev_settle"))
            .when(pl.col("calendar_days_after_prev_settle") <= 5)
            .then(pl.lit("D3-D5_after_prev_settle"))
            .when(pl.col("calendar_days_after_prev_settle") <= 10)
            .then(pl.lit("D6-D10_after_prev_settle"))
            .otherwise(pl.lit("D11+_after_prev_settle"))
            .alias("after_prev_settle_bucket"),
            pl.when(pl.col("calendar_days_to_settle").is_null())
            .then(pl.lit("unknown_to_settle"))
            .when(pl.col("calendar_days_to_settle") <= 2)
            .then(pl.lit("D0-D2_to_settle"))
            .when(pl.col("calendar_days_to_settle") <= 5)
            .then(pl.lit("D3-D5_to_settle"))
            .when(pl.col("calendar_days_to_settle") <= 10)
            .then(pl.lit("D6-D10_to_settle"))
            .otherwise(pl.lit("D11+_to_settle"))
            .alias("to_settle_bucket"),
            pl.when(pl.col("entry_time").dt.time() < pl.time(9, 5))
            .then(pl.lit("09:00-09:05"))
            .when(pl.col("entry_time").dt.time() < pl.time(9, 30))
            .then(pl.lit("09:05-09:30"))
            .when(pl.col("entry_time").dt.time() < pl.time(10, 30))
            .then(pl.lit("09:30-10:30"))
            .when(pl.col("entry_time").dt.time() < pl.time(12, 0))
            .then(pl.lit("10:30-12:00"))
            .otherwise(pl.lit("12:00-13:24"))
            .alias("entry_time_bucket"),
        )
    )


def _load_exit_spot_ask_for_date(date: str, day: pl.DataFrame) -> pl.DataFrame:
    path = DATA_DIR / f"{date}_arbitrage_features.parquet"
    if not path.exists() or day.is_empty():
        return day.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("converge_exit_spot_ask"),
            pl.lit(None, dtype=pl.Datetime("us")).alias("converge_exit_quote_time"),
            pl.lit("missing_feature_file").alias("converge_exit_source"),
        )
    quotes = (
        pl.scan_parquet(path)
        .select(
            "ValueCode",
            pl.col("RecvTime").dt.cast_time_unit("us").alias("quote_time"),
            pl.col("spot_ask").cast(pl.Float64).alias("converge_exit_spot_ask"),
        )
        .filter(pl.col("converge_exit_spot_ask") > 0)
        .sort(["ValueCode", "quote_time"])
        .collect()
    )
    joined = day.sort(["ValueCode", "exit_time"]).join_asof(
        quotes,
        left_on="exit_time",
        right_on="quote_time",
        by="ValueCode",
        strategy="backward",
    )
    return joined.with_columns(
        pl.col("quote_time").alias("converge_exit_quote_time"),
        pl.when(pl.col("converge_exit_spot_ask").is_not_null())
        .then(pl.lit("feature_asof_backward"))
        .otherwise(pl.lit("missing_quote"))
        .alias("converge_exit_source"),
    ).drop("quote_time")


def _attach_convergence_spot_ask(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("exit_time").dt.strftime("%Y%m%d").alias("exit_date"))
    parts = []
    for date in sorted(df["exit_date"].unique().to_list()):
        parts.append(_load_exit_spot_ask_for_date(date, df.filter(pl.col("exit_date") == date)))
    return pl.concat(parts, how="diagonal_relaxed") if parts else df


def _add_pnl(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("fut_lots_equiv") * CONTRACT_SIZE).cast(pl.Float64).alias("shares"),
        )
        .with_columns(
            (pl.col("entry_spot_price") * pl.col("shares")).alias("entry_notional_twd"),
            ((pl.col("next_open_price") - pl.col("entry_spot_price")) * pl.col("shares")).alias("next_open_pnl_twd"),
            ((pl.col("next_close_price") - pl.col("entry_spot_price")) * pl.col("shares")).alias("next_close_pnl_twd"),
            ((pl.col("converge_exit_spot_ask") - pl.col("entry_spot_price")) * pl.col("shares")).alias("converge_ask_pnl_twd"),
        )
        .with_columns(
            (pl.col("next_open_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_open_pnl_bp"),
            (pl.col("next_close_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_close_pnl_bp"),
            (pl.col("converge_ask_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("converge_ask_pnl_bp"),
        )
    )


def _summarize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["scope", "threshold", "status_full"])
        .agg(
            pl.len().alias("samples"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            pl.col("next_open_price").is_not_null().sum().alias("next_open_priced_samples"),
            pl.col("next_close_price").is_not_null().sum().alias("next_close_priced_samples"),
            pl.col("converge_exit_spot_ask").is_not_null().sum().alias("converge_priced_samples"),
            pl.col("next_open_pnl_twd").sum().alias("next_open_pnl_twd"),
            pl.col("next_close_pnl_twd").sum().alias("next_close_pnl_twd"),
            pl.col("converge_ask_pnl_twd").sum().alias("converge_ask_pnl_twd"),
            pl.when(pl.col("next_open_pnl_twd") > 0).then(1).otherwise(0).sum().alias("next_open_win_samples"),
            pl.when(pl.col("next_close_pnl_twd") > 0).then(1).otherwise(0).sum().alias("next_close_win_samples"),
            pl.when(pl.col("converge_ask_pnl_twd") > 0).then(1).otherwise(0).sum().alias("converge_win_samples"),
        )
        .with_columns(
            (pl.col("next_open_priced_samples") / pl.col("samples") * 100).alias("next_open_priced_pct"),
            (pl.col("next_close_priced_samples") / pl.col("samples") * 100).alias("next_close_priced_pct"),
            (pl.col("converge_priced_samples") / pl.col("samples") * 100).alias("converge_priced_pct"),
            (pl.col("next_open_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_open_pnl_bp"),
            (pl.col("next_close_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_close_pnl_bp"),
            (pl.col("converge_ask_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("converge_ask_pnl_bp"),
            (pl.col("next_open_win_samples") / pl.col("next_open_priced_samples") * 100).alias("next_open_win_pct"),
            (pl.col("next_close_win_samples") / pl.col("next_close_priced_samples") * 100).alias("next_close_win_pct"),
            (pl.col("converge_win_samples") / pl.col("converge_priced_samples") * 100).alias("converge_win_pct"),
        )
        .sort(["scope", "threshold", "status_full"])
    )


def _fmt_yi(v: float | None) -> str:
    return "" if v is None else f"{v / 1e8:.2f}"


def _fmt_wan(v: float | None) -> str:
    return "" if v is None else f"{v / 1e4:.2f}"


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def _md_table(summary: pl.DataFrame, scope: str) -> str:
    view = summary.filter(pl.col("scope") == scope)
    headers = [
        "threshold_pct",
        "status",
        "samples",
        "notional_yi",
        "next_open_bp",
        "next_open_wan",
        "next_open_win_pct",
        "next_close_bp",
        "next_close_wan",
        "next_close_win_pct",
        "conv_ask_bp",
        "conv_ask_wan",
        "conv_win_pct",
        "conv_priced_pct",
    ]
    lines = [
        f"## {scope}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in view.iter_rows(named=True):
        vals = [
            f"{r['threshold'] * 100:.2f}",
            r["status_full"],
            str(r["samples"]),
            _fmt_yi(r["entry_notional_twd"]),
            _fmt(r["next_open_pnl_bp"]),
            _fmt_wan(r["next_open_pnl_twd"]),
            _fmt(r["next_open_win_pct"]),
            _fmt(r["next_close_pnl_bp"]),
            _fmt_wan(r["next_close_pnl_twd"]),
            _fmt(r["next_close_win_pct"]),
            _fmt(r["converge_ask_pnl_bp"]),
            _fmt_wan(r["converge_ask_pnl_twd"]),
            _fmt(r["converge_win_pct"]),
            _fmt(r["converge_priced_pct"]),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(detail: pl.DataFrame, summary: pl.DataFrame) -> None:
    lines = [
        "# Spot-only Premium Signal Analysis",
        "",
        "Entry buys spot at the observed signal A1. PnL is gross and excludes fees/tax.",
        "`conv_ask` exits by posting/selling at spot A1 at the existing convergence or settlement release timestamp; fill probability is not modeled.",
        "",
        _md_table(summary, "all_stage3_attempts"),
        "",
        _md_table(summary, "futures_first_executable_ledger"),
        "",
        "## Unpriced Convergence/Settlement A1 Exits",
        "",
    ]
    missing = (
        detail.filter(pl.col("converge_exit_source") != "feature_asof_backward")
        .group_by(["scope", "converge_exit_source", "exit_date"])
        .agg(pl.len().alias("samples"), pl.col("entry_notional_twd").sum().alias("notional_twd"))
        .sort(["scope", "converge_exit_source", "exit_date"])
    )
    if missing.is_empty():
        lines.append("All convergence exits were priced.")
    else:
        lines.extend(["| scope | source | exit_date | samples | notional_yi |", "| --- | --- | --- | --- | --- |"])
        for r in missing.iter_rows(named=True):
            lines.append(
                f"| {r['scope']} | {r['converge_exit_source']} | {r['exit_date']} | "
                f"{r['samples']} | {_fmt_yi(r['notional_twd'])} |"
            )
    lines.extend(["", "## Outputs", "", f"- `{OUT_DETAIL}`", f"- `{OUT_SUMMARY}`"])
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overnight_group_summary(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    return (
        df.group_by(["scope", "threshold", pl.col(group_col).alias("bucket")])
        .agg(
            pl.len().alias("samples"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            pl.col("next_open_pnl_twd").sum().alias("next_open_pnl_twd"),
            pl.col("next_close_pnl_twd").sum().alias("next_close_pnl_twd"),
            pl.when(pl.col("next_open_pnl_twd") > 0).then(1).otherwise(0).sum().alias("next_open_win_samples"),
            pl.when(pl.col("next_close_pnl_twd") > 0).then(1).otherwise(0).sum().alias("next_close_win_samples"),
        )
        .with_columns(
            pl.lit(group_col).alias("group"),
            (pl.col("next_open_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_open_bp"),
            (pl.col("next_close_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("next_close_bp"),
            (pl.col("next_open_win_samples") / pl.col("samples") * 100).alias("next_open_win_pct"),
            (pl.col("next_close_win_samples") / pl.col("samples") * 100).alias("next_close_win_pct"),
        )
        .select(
            "scope",
            "threshold",
            "group",
            "bucket",
            "samples",
            "entry_notional_twd",
            "next_open_pnl_twd",
            "next_open_bp",
            "next_open_win_pct",
            "next_close_pnl_twd",
            "next_close_bp",
            "next_close_win_pct",
        )
    )


def _write_overnight_actionable_report(detail: pl.DataFrame) -> None:
    all_bucket = detail.with_columns(pl.lit("all").alias("all_bucket"))
    actionable = pl.concat(
        [
            _overnight_group_summary(all_bucket, "all_bucket"),
            _overnight_group_summary(detail, "after_prev_settle_bucket"),
            _overnight_group_summary(detail, "to_settle_bucket"),
            _overnight_group_summary(detail, "entry_time_bucket"),
        ],
        how="diagonal_relaxed",
    ).sort(["scope", "threshold", "group", "bucket"])
    actionable.write_csv(OUT_OVERNIGHT_SUMMARY)

    attribution = _overnight_group_summary(detail, "status_full").with_columns(
        pl.lit("post_trade_attribution_only").alias("group")
    )

    def table(df: pl.DataFrame, scope: str, group: str, thresholds: tuple[float, ...] = (0.005, 0.0075)) -> str:
        view = df.filter((pl.col("scope") == scope) & (pl.col("group") == group) & (pl.col("threshold").is_in(thresholds)))
        headers = [
            "threshold_pct",
            "bucket",
            "samples",
            "notional_yi",
            "next_open_bp",
            "next_open_wan",
            "next_open_win_pct",
            "next_close_bp",
            "next_close_wan",
            "next_close_win_pct",
        ]
        lines = [
            f"### {group}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in view.iter_rows(named=True):
            vals = [
                f"{r['threshold'] * 100:.2f}",
                r["bucket"],
                str(r["samples"]),
                _fmt_yi(r["entry_notional_twd"]),
                _fmt(r["next_open_bp"]),
                _fmt_wan(r["next_open_pnl_twd"]),
                _fmt(r["next_open_win_pct"]),
                _fmt(r["next_close_bp"]),
                _fmt_wan(r["next_close_pnl_twd"]),
                _fmt(r["next_close_win_pct"]),
            ]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Spot-only Overnight Actionable View",
        "",
        "This report removes future-known convergence labels from the main overnight view.",
        "Buckets are based on information known at entry time: threshold, calendar position, and entry time.",
        "",
        "## all_stage3_attempts",
        "",
        table(actionable, "all_stage3_attempts", "all_bucket"),
        "",
        table(actionable, "all_stage3_attempts", "after_prev_settle_bucket"),
        "",
        table(actionable, "all_stage3_attempts", "to_settle_bucket"),
        "",
        table(actionable, "all_stage3_attempts", "entry_time_bucket"),
        "",
        "## futures_first_executable_ledger",
        "",
        table(actionable, "futures_first_executable_ledger", "all_bucket"),
        "",
        table(actionable, "futures_first_executable_ledger", "after_prev_settle_bucket"),
        "",
        table(actionable, "futures_first_executable_ledger", "to_settle_bucket"),
        "",
        table(actionable, "futures_first_executable_ledger", "entry_time_bucket"),
        "",
        "## Post-trade Attribution Only",
        "",
        "The following status split is not actionable for an overnight trade; it is only included to explain where the overnight PnL came from.",
        "",
        table(attribution, "all_stage3_attempts", "post_trade_attribution_only"),
        "",
        "## Outputs",
        "",
        f"- `{OUT_OVERNIGHT_SUMMARY}`",
    ]
    OUT_OVERNIGHT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades = pl.concat([_load_all_stage3_attempts(), _load_executable_ledger()], how="diagonal_relaxed")
    detail = _add_calendar_features(_add_pnl(_attach_convergence_spot_ask(_attach_next_day_prices(trades))))
    summary = _summarize(detail)
    detail.write_parquet(OUT_DETAIL)
    summary.write_csv(OUT_SUMMARY)
    _write_report(detail, summary)
    _write_overnight_actionable_report(detail)
    print(f"wrote {OUT_REPORT}")
    print(f"wrote {OUT_OVERNIGHT_REPORT}")


if __name__ == "__main__":
    main()
