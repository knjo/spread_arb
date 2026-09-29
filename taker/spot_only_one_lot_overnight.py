"""One-lot spot-only overnight test after futures premium thresholds.

Rules:
* Signal: `ret_sell >= threshold` in daily arbitrage features.
* Entry: buy one board lot (1,000 shares) at spot A1 after 100ms.
* Per-threshold-symbol daily cap: keep buying until entry notional would exceed
  TWD 2M.
* Exit: sell all next trading day at stock open price.
* Cost: subtract 34bp of sell notional.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


from data_paths import MARKET_DIR, PLOT_DIR, STOCKFUTURE_DIR as DATA_DIR

START = "20260126"
END = "20260629"
BOARD_LOT_SHARES = 1000
SYMBOL_CAP_TWD = 2_000_000.0
SELL_COST_BP = 34.0
THRESHOLDS = (0.005, 0.0075)

OUT_DETAIL = DATA_DIR / f"spot_only_one_lot_overnight_detail_{START}_{END}.parquet"
OUT_DAILY = DATA_DIR / f"spot_only_one_lot_overnight_daily_{START}_{END}.csv"
OUT_SYMBOL = DATA_DIR / f"spot_only_one_lot_overnight_symbol_{START}_{END}.csv"
OUT_REPORT = DATA_DIR / f"spot_only_one_lot_overnight_report_{START}_{END}.md"
OUT_POS_PLOT = PLOT_DIR / f"spot_only_one_lot_overnight_position_thr0p50_0p75_{START}_{END}.png"
OUT_PNL_PLOT = PLOT_DIR / f"spot_only_one_lot_overnight_pnl_thr0p50_0p75_{START}_{END}.png"
OUT_CUM_PLOT = PLOT_DIR / f"spot_only_one_lot_overnight_cumulative_pnl_thr0p50_0p75_{START}_{END}.png"


def _date_files(directory: Path, suffix: str) -> list[str]:
    dates = []
    for path in directory.glob(f"*{suffix}"):
        if path.name[:8].isdigit():
            dates.append(path.name[:8])
    return sorted(set(dates))


def _next_trade_map() -> pl.DataFrame:
    dates = _date_files(MARKET_DIR, "_marketData.parquet")
    rows = [(date, next((d for d in dates if d > date), None)) for date in dates]
    return pl.DataFrame(rows, schema=["Date", "next_trade_date"], orient="row")


def _load_next_open(dates: list[str]) -> pl.DataFrame:
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
                pl.col("open_price").cast(pl.Float64).alias("exit_open_price"),
            )
            .filter(pl.col("exit_open_price") > 0)
            .collect()
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def _load_entries_for_date(date: str) -> pl.DataFrame:
    path = DATA_DIR / f"{date}_arbitrage_features.parquet"
    if not path.exists():
        return pl.DataFrame()

    base = (
        pl.scan_parquet(path)
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            pl.col("RecvTime").dt.cast_time_unit("us").alias("signal_time"),
            pl.col("target_spot_100ms").dt.cast_time_unit("us").alias("entry_time"),
            "ret_sell",
            "spot_ask",
            "spot_ask_100ms",
            "spot_trial_state_100ms",
        )
        .filter(
            (pl.col("ret_sell") >= min(THRESHOLDS))
            & (pl.col("spot_ask_100ms") > 0)
            & (pl.col("spot_trial_state_100ms") == 0)
        )
        .with_columns(
            (pl.col("spot_ask_100ms") * BOARD_LOT_SHARES).alias("entry_notional_twd"),
        )
        .collect()
    )
    parts = []
    for threshold in THRESHOLDS:
        parts.append(
            base.filter(pl.col("ret_sell") >= threshold)
            .with_columns(pl.lit(threshold).alias("threshold"))
            .sort(["threshold", "ValueCode", "signal_time"])
            .with_columns(
                pl.col("entry_notional_twd")
                .cum_sum()
                .over(["threshold", "Date", "ValueCode"])
                .alias("cum_entry_notional_twd")
            )
            .filter(pl.col("cum_entry_notional_twd") <= SYMBOL_CAP_TWD)
            .with_columns(
                pl.lit(BOARD_LOT_SHARES).alias("shares"),
                pl.lit(SYMBOL_CAP_TWD).alias("symbol_cap_twd"),
                pl.lit(SELL_COST_BP).alias("sell_cost_bp"),
            )
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def build_trades() -> pl.DataFrame:
    feature_dates = [d for d in _date_files(DATA_DIR, "_arbitrage_features.parquet") if START <= d <= END]
    parts = []
    for date in feature_dates:
        day = _load_entries_for_date(date)
        if not day.is_empty():
            parts.append(day)
    trades = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    if trades.is_empty():
        return trades

    mapped = trades.join(_next_trade_map(), on="Date", how="left")
    next_open = _load_next_open(mapped["next_trade_date"].drop_nulls().unique().to_list())
    return (
        mapped.join(next_open, on=["next_trade_date", "ValueCode"], how="left")
        .with_columns(
            (pl.col("exit_open_price") * pl.col("shares")).alias("exit_sell_notional_twd"),
            (pl.col("exit_open_price") * pl.col("shares") * SELL_COST_BP / 10000).alias("sell_cost_twd"),
        )
        .with_columns(
            (
                (pl.col("exit_open_price") - pl.col("spot_ask_100ms")) * pl.col("shares")
                - pl.col("sell_cost_twd")
            ).alias("net_pnl_twd"),
            (
                (
                    (pl.col("exit_open_price") - pl.col("spot_ask_100ms")) * pl.col("shares")
                    - pl.col("sell_cost_twd")
                )
                / pl.col("entry_notional_twd")
                * 10000
            ).alias("net_pnl_bp"),
        )
        .with_row_index("trade_id")
    )


def _summaries(trades: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    priced = trades.filter(pl.col("exit_open_price").is_not_null())
    daily = (
        priced.group_by(["threshold", "Date"])
        .agg(
            pl.len().alias("trades"),
            pl.col("ValueCode").n_unique().alias("symbols"),
            pl.col("entry_notional_twd").sum().alias("overnight_position_twd"),
            pl.col("exit_sell_notional_twd").sum().alias("exit_sell_notional_twd"),
            pl.col("sell_cost_twd").sum().alias("sell_cost_twd"),
            pl.col("net_pnl_twd").sum().alias("net_pnl_twd"),
        )
        .with_columns(
            (pl.col("net_pnl_twd") / pl.col("overnight_position_twd") * 10000).alias("net_pnl_bp"),
        )
        .sort(["threshold", "Date"])
        .with_columns(pl.col("net_pnl_twd").cum_sum().over("threshold").alias("cum_net_pnl_twd"))
    )
    symbol = (
        priced.group_by(["threshold", "ValueCode"])
        .agg(
            pl.len().alias("trades"),
            pl.col("Date").n_unique().alias("trade_days"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            pl.col("sell_cost_twd").sum().alias("sell_cost_twd"),
            pl.col("net_pnl_twd").sum().alias("net_pnl_twd"),
            (pl.col("net_pnl_twd") > 0).mean().mul(100).alias("trade_win_pct"),
        )
        .with_columns(
            (pl.col("net_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("net_pnl_bp"),
        )
        .sort(["threshold", "net_pnl_twd"], descending=[False, True])
    )
    return daily, symbol


def _plot_daily(daily: pl.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = daily.to_pandas()
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.38
    dates = sorted(pdf["Date"].unique())
    x = list(range(len(dates)))
    for i, threshold in enumerate(THRESHOLDS):
        sub = daily.filter(pl.col("threshold") == threshold).sort("Date").to_pandas()
        offset = (i - 0.5) * width
        ax.bar([v + offset for v in x], sub["overnight_position_twd"] / 1e8, width=width, label=f"{threshold:.2%}")
    ax.set_title("Spot-only Overnight Position")
    ax.set_ylabel("Entry notional (TWD 100M)")
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_POS_PLOT, dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    for i, threshold in enumerate(THRESHOLDS):
        sub = daily.filter(pl.col("threshold") == threshold).sort("Date").to_pandas()
        offset = (i - 0.5) * width
        ax1.bar([v + offset for v in x], sub["net_pnl_twd"] / 1e4, width=width, label=f"{threshold:.2%} daily")
    ax1.set_ylabel("Daily net PnL (TWD 10K)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates)
    ax1.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax2 = ax1.twinx()
    for threshold in THRESHOLDS:
        sub = daily.filter(pl.col("threshold") == threshold).sort("Date").to_pandas()
        ax2.plot(x, sub["cum_net_pnl_twd"] / 1e4, linewidth=2, label=f"{threshold:.2%} cumulative")
    ax2.set_ylabel("Cumulative net PnL (TWD 10K)")
    ax1.set_title("Spot-only Overnight PnL")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNL_PLOT, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for threshold in THRESHOLDS:
        sub = daily.filter(pl.col("threshold") == threshold).sort("Date").to_pandas()
        ax.plot(x, sub["cum_net_pnl_twd"] / 1e4, linewidth=2.2, label=f"{threshold:.2%}")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("Spot-only Overnight Cumulative Net PnL")
    ax.set_ylabel("Cumulative net PnL (TWD 10K)")
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CUM_PLOT, dpi=160)
    plt.close(fig)


def _fmt_wan(v: float | None) -> str:
    return "" if v is None else f"{v / 1e4:.2f}"


def _fmt_yi(v: float | None) -> str:
    return "" if v is None else f"{v / 1e8:.2f}"


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def _write_report(trades: pl.DataFrame, daily: pl.DataFrame, symbol: pl.DataFrame) -> None:
    priced = trades.filter(pl.col("exit_open_price").is_not_null())
    overall = (
        priced.group_by("threshold")
        .agg(
        pl.len().alias("trades"),
        pl.col("Date").n_unique().alias("trade_days"),
        pl.col("ValueCode").n_unique().alias("symbols"),
        pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
        pl.col("exit_sell_notional_twd").sum().alias("exit_sell_notional_twd"),
        pl.col("sell_cost_twd").sum().alias("sell_cost_twd"),
        pl.col("net_pnl_twd").sum().alias("net_pnl_twd"),
        (pl.col("net_pnl_twd") > 0).mean().mul(100).alias("trade_win_pct"),
        )
        .with_columns((pl.col("net_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("net_pnl_bp"))
        .sort("threshold")
    )

    symbol_stats = symbol.group_by("threshold").agg(
        pl.len().alias("symbols"),
        (pl.col("net_pnl_twd") > 0).mean().mul(100).alias("profitable_symbol_pct"),
        pl.col("net_pnl_bp").median().alias("symbol_bp_median"),
        pl.col("net_pnl_bp").mean().alias("symbol_bp_mean"),
    ).sort("threshold")

    def symbol_table(df: pl.DataFrame) -> list[str]:
        lines = [
            "| ValueCode | trades | trade_days | notional_yi | net_pnl_wan | net_pnl_bp | trade_win_pct |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in df.iter_rows(named=True):
            lines.append(
                f"| {r['ValueCode']} | {r['trades']} | {r['trade_days']} | {_fmt_yi(r['entry_notional_twd'])} | "
                f"{_fmt_wan(r['net_pnl_twd'])} | {_fmt(r['net_pnl_bp'])} | {_fmt(r['trade_win_pct'])} |"
            )
        return lines

    overall_with_stats = overall.join(symbol_stats, on="threshold", how="left")
    overall_lines = [
        "| threshold_pct | trades | trade_days | symbols | entry_notional_yi | sell_cost_wan | net_pnl_wan | net_pnl_bp | trade_win_pct | profitable_symbol_pct | symbol_bp_median |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in overall_with_stats.iter_rows(named=True):
        overall_lines.append(
            f"| {r['threshold'] * 100:.2f} | {r['trades']} | {r['trade_days']} | {r['symbols']} | "
            f"{_fmt_yi(r['entry_notional_twd'])} | {_fmt_wan(r['sell_cost_twd'])} | "
            f"{_fmt_wan(r['net_pnl_twd'])} | {_fmt(r['net_pnl_bp'])} | "
            f"{_fmt(r['trade_win_pct'])} | {_fmt(r['profitable_symbol_pct'])} | "
            f"{_fmt(r['symbol_bp_median'])} |"
        )

    lines = [
        "# Spot-only One-lot Overnight Test",
        "",
        "Rules: `ret_sell >= threshold`; buy 1,000 shares at spot A1 after 100ms; cap TWD 2M per `threshold x Date x ValueCode`; sell next trading day at open; subtract 34bp of sell notional.",
        "",
        "## Overall",
        "",
        *overall_lines,
        "",
    ]
    for threshold in THRESHOLDS:
        view = symbol.filter(pl.col("threshold") == threshold)
        lines.extend(
            [
                f"## Best Symbols {threshold:.2%}",
                "",
                *symbol_table(view.sort("net_pnl_twd", descending=True).head(20)),
                "",
                f"## Worst Symbols {threshold:.2%}",
                "",
                *symbol_table(view.sort("net_pnl_twd").head(20)),
                "",
            ]
        )
    lines.extend([
        "## Plots",
        "",
        f"- `{OUT_POS_PLOT}`",
        f"- `{OUT_PNL_PLOT}`",
        f"- `{OUT_CUM_PLOT}`",
        "",
        "## Outputs",
        "",
        f"- `{OUT_DETAIL}`",
        f"- `{OUT_DAILY}`",
        f"- `{OUT_SYMBOL}`",
    ])
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades = build_trades()
    if trades.is_empty():
        raise RuntimeError("no trades generated")
    daily, symbol = _summaries(trades)
    trades.write_parquet(OUT_DETAIL)
    daily.write_csv(OUT_DAILY)
    symbol.write_csv(OUT_SYMBOL)
    _plot_daily(daily)
    _write_report(trades, daily, symbol)
    print(f"wrote {OUT_REPORT}")
    print(f"wrote {OUT_POS_PLOT}")
    print(f"wrote {OUT_PNL_PLOT}")


if __name__ == "__main__":
    main()
