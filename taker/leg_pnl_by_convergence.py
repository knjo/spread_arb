"""Split futures-spot backtest PnL by leg and convergence status.

The main backtest ledger stores entry prices and a release/exit timestamp, but
its gross PnL is the locked entry spread.  This script reconstructs markable
exit prices from the daily arbitrage feature files:

* long spot exits at spot bid
* short futures exits at futures ask

Rows whose exit day has no feature file remain explicitly unpriced.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "stockfuture"
START = "20260126"
END = "20260629"
TRADES_PATH = DATA_DIR / (
    "backtest_trades_20260126_20260629_"
    "ref9_symbol_open_base_plus_volume10_fee38p0bp.parquet"
)
OUT_DETAIL = DATA_DIR / f"backtest_leg_pnl_by_status_detail_{START}_{END}.parquet"
OUT_SUMMARY = DATA_DIR / f"backtest_leg_pnl_by_status_summary_{START}_{END}.csv"
OUT_REPORT = DATA_DIR / f"backtest_leg_pnl_by_status_report_{START}_{END}.md"


def _fmt_yi(v: float | None) -> str:
    return "" if v is None else f"{v / 1e8:.2f}"


def _fmt_wan(v: float | None) -> str:
    return "" if v is None else f"{v / 1e4:.2f}"


def _fmt_bp(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def _load_exit_quotes_for_date(date: str, trades: pl.DataFrame) -> pl.DataFrame:
    feature_path = DATA_DIR / f"{date}_arbitrage_features.parquet"
    if not feature_path.exists() or trades.is_empty():
        return trades.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("exit_spot_sell_price"),
            pl.lit(None, dtype=pl.Float64).alias("exit_fut_buy_price"),
            pl.lit(None, dtype=pl.Datetime("us")).alias("exit_quote_time"),
            pl.lit(False).alias("exit_priced"),
            pl.lit("missing_feature_file").alias("exit_price_source"),
        )

    quotes = (
        pl.scan_parquet(feature_path)
        .select(
            "ValueCode",
            "QuoteCode",
            pl.col("RecvTime").dt.cast_time_unit("us").alias("quote_time"),
            pl.col("spot_bid").alias("exit_spot_sell_price"),
            pl.col("fut_ask").alias("exit_fut_buy_price"),
            pl.col("ret_buy").alias("exit_ret_buy_signal"),
        )
        .filter(
            pl.col("exit_spot_sell_price").is_not_null()
            & pl.col("exit_fut_buy_price").is_not_null()
            & (pl.col("exit_spot_sell_price") > 0)
            & (pl.col("exit_fut_buy_price") > 0)
        )
        .sort(["ValueCode", "QuoteCode", "quote_time"])
        .collect()
    )

    joined = trades.sort(["ValueCode", "QuoteCode", "exit_time"]).join_asof(
        quotes,
        left_on="exit_time",
        right_on="quote_time",
        by=["ValueCode", "QuoteCode"],
        strategy="backward",
    )
    return joined.with_columns(
        pl.col("quote_time").alias("exit_quote_time"),
        (
            pl.col("exit_spot_sell_price").is_not_null()
            & pl.col("exit_fut_buy_price").is_not_null()
        ).alias("exit_priced"),
        pl.when(
            pl.col("exit_spot_sell_price").is_not_null()
            & pl.col("exit_fut_buy_price").is_not_null()
        )
        .then(pl.lit("feature_asof_backward"))
        .otherwise(pl.lit("missing_quote"))
        .alias("exit_price_source"),
    ).drop("quote_time")


def attach_exit_prices(trades: pl.DataFrame) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for date in sorted(trades["exit_date"].unique().to_list()):
        day_trades = trades.filter(pl.col("exit_date") == date)
        parts.append(_load_exit_quotes_for_date(date, day_trades))
    return pl.concat(parts, how="diagonal_relaxed") if parts else trades


def add_leg_pnl(priced: pl.DataFrame) -> pl.DataFrame:
    return (
        priced.with_columns(
        pl.when(pl.col("exit_priced"))
        .then((pl.col("exit_spot_sell_price") - pl.col("entry_spot_buy_price")) * pl.col("shares"))
        .otherwise(None)
        .alias("spot_leg_pnl_twd"),
        pl.when(pl.col("exit_priced"))
        .then((pl.col("entry_fut_sell_price") - pl.col("exit_fut_buy_price")) * pl.col("shares"))
        .otherwise(None)
        .alias("fut_leg_pnl_twd"),
        )
        .with_columns(
            (pl.col("spot_leg_pnl_twd") + pl.col("fut_leg_pnl_twd")).alias("mark_gross_pnl_twd"),
            (pl.col("spot_leg_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("spot_leg_pnl_bp"),
            (pl.col("fut_leg_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("fut_leg_pnl_bp"),
        )
        .with_columns(
            (pl.col("mark_gross_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("mark_gross_pnl_bp"),
            (
                (pl.col("mark_gross_pnl_twd") - pl.col("fee_twd"))
                / pl.col("entry_notional_twd")
                * 10000
            ).alias("mark_net_pnl_bp"),
        )
    )


def summarize(detail: pl.DataFrame) -> pl.DataFrame:
    return (
        detail.group_by(["threshold", "status_full"])
        .agg(
            pl.len().alias("samples"),
            pl.col("exit_priced").sum().alias("priced_samples"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            pl.when(pl.col("exit_priced"))
            .then(pl.col("entry_notional_twd"))
            .otherwise(0)
            .sum()
            .alias("priced_notional_twd"),
            pl.col("spot_leg_pnl_twd").sum().alias("spot_leg_pnl_twd"),
            pl.col("fut_leg_pnl_twd").sum().alias("fut_leg_pnl_twd"),
            pl.col("mark_gross_pnl_twd").sum().alias("mark_gross_pnl_twd"),
            pl.col("gross_pnl_twd").sum().alias("locked_spread_gross_pnl_twd"),
            pl.when(pl.col("exit_priced") & (pl.col("spot_leg_pnl_twd") > 0))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("spot_win_samples"),
            pl.when(pl.col("exit_priced") & (pl.col("fut_leg_pnl_twd") > 0))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("fut_win_samples"),
        )
        .with_columns(
            (pl.col("priced_samples") / pl.col("samples") * 100).alias("priced_sample_pct"),
            (pl.col("priced_notional_twd") / pl.col("entry_notional_twd") * 100).alias("priced_notional_pct"),
            (pl.col("spot_leg_pnl_twd") / pl.col("priced_notional_twd") * 10000).alias("spot_leg_pnl_bp"),
            (pl.col("fut_leg_pnl_twd") / pl.col("priced_notional_twd") * 10000).alias("fut_leg_pnl_bp"),
            (pl.col("mark_gross_pnl_twd") / pl.col("priced_notional_twd") * 10000).alias("mark_gross_pnl_bp"),
            (
                pl.col("locked_spread_gross_pnl_twd")
                / pl.col("entry_notional_twd")
                * 10000
            ).alias("locked_spread_gross_bp"),
            (pl.col("spot_win_samples") / pl.col("priced_samples") * 100).alias("spot_win_pct"),
            (pl.col("fut_win_samples") / pl.col("priced_samples") * 100).alias("fut_win_pct"),
        )
        .sort(["threshold", "status_full"])
    )


def _md_table(summary: pl.DataFrame) -> str:
    headers = [
        "threshold_pct",
        "status",
        "samples",
        "priced_pct",
        "notional_yi",
        "priced_notional_yi",
        "spot_pnl_wan",
        "fut_pnl_wan",
        "gross_pnl_wan",
        "spot_bp",
        "fut_bp",
        "gross_bp",
        "spot_win_pct",
        "fut_win_pct",
        "locked_spread_bp",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in summary.iter_rows(named=True):
        vals = [
            f"{r['threshold'] * 100:.2f}",
            r["status_full"],
            str(r["samples"]),
            f"{r['priced_sample_pct']:.2f}",
            _fmt_yi(r["entry_notional_twd"]),
            _fmt_yi(r["priced_notional_twd"]),
            _fmt_wan(r["spot_leg_pnl_twd"]),
            _fmt_wan(r["fut_leg_pnl_twd"]),
            _fmt_wan(r["mark_gross_pnl_twd"]),
            _fmt_bp(r["spot_leg_pnl_bp"]),
            _fmt_bp(r["fut_leg_pnl_bp"]),
            _fmt_bp(r["mark_gross_pnl_bp"]),
            _fmt_bp(r["spot_win_pct"]),
            _fmt_bp(r["fut_win_pct"]),
            _fmt_bp(r["locked_spread_gross_bp"]),
        ]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _write_report(detail: pl.DataFrame, summary: pl.DataFrame) -> None:
    source_summary = (
        detail.group_by(["exit_price_source", "exit_date"])
        .agg(pl.len().alias("samples"), pl.col("entry_notional_twd").sum().alias("notional_twd"))
        .sort(["exit_price_source", "exit_date"])
    )
    missing = source_summary.filter(pl.col("exit_price_source") != "feature_asof_backward")
    lines = [
        "# Backtest Leg PnL By Convergence Status",
        "",
        (
            "Exit is reconstructed as `sell spot at spot_bid` and `buy futures at fut_ask` "
            "from the daily arbitrage feature file using as-of backward on `exit_time`."
        ),
        "",
        "The original backtest `gross_pnl_twd` is the locked entry spread; "
        "`gross_bp` below is the markable two-leg exit result on rows with exit quotes.",
        "",
        _md_table(summary),
        "",
        "## Unpriced Exits",
        "",
    ]
    if missing.is_empty():
        lines.append("All exits were priced.")
    else:
        lines.extend(
            [
                "| source | exit_date | samples | notional_yi |",
                "| --- | --- | --- | --- |",
            ]
        )
        for r in missing.iter_rows(named=True):
            lines.append(
                f"| {r['exit_price_source']} | {r['exit_date']} | {r['samples']} | "
                f"{_fmt_yi(r['notional_twd'])} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT_DETAIL}`",
            f"- `{OUT_SUMMARY}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades = (
        pl.read_parquet(TRADES_PATH)
        .with_row_index("trade_row_id")
        .with_columns(
            pl.col("threshold").cast(pl.Float64),
            pl.col("shares").cast(pl.Float64),
            pl.col("entry_notional_twd").cast(pl.Float64),
            pl.col("fee_twd").cast(pl.Float64),
            pl.col("gross_pnl_twd").cast(pl.Float64),
        )
    )
    detail = add_leg_pnl(attach_exit_prices(trades))
    summary = summarize(detail)
    detail.write_parquet(OUT_DETAIL)
    summary.write_csv(OUT_SUMMARY)
    _write_report(detail, summary)
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
