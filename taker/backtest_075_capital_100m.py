"""Scale the preserved 0.75% backtest path to a TWD 100M capital budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


from data_paths import MARKET_DIR, PLOT_DIR, STOCKFUTURE_DIR as DATA_DIR

START = "20260126"
END = "20260629"
THRESHOLD = 0.0075
CAPITAL_TWD = 100_000_000.0
TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-12

SOURCE_DAILY = (
    DATA_DIR
    / "backtest_daily_20260126_20260629_ref9_symbol_open_"
    "base_plus_volume10_fee38p0bp.csv"
)
SOURCE_SUMMARY = (
    DATA_DIR
    / "backtest_summary_20260126_20260629_ref9_symbol_open_"
    "base_plus_volume10_fee38p0bp.csv"
)
OUT_DAILY = DATA_DIR / f"backtest_thr0p75_capital100m_daily_{START}_{END}.csv"
OUT_METRICS = DATA_DIR / f"backtest_thr0p75_capital100m_metrics_{START}_{END}.csv"
OUT_REPORT = DATA_DIR / f"backtest_thr0p75_capital100m_report_{START}_{END}.md"
OUT_PLOT = PLOT_DIR / f"backtest_thr0p75_capital100m_{START}_{END}.png"


def _trading_dates() -> list[str]:
    return sorted(
        path.name[:8]
        for path in MARKET_DIR.glob("*_marketData.parquet")
        if START <= path.name[:8] <= END
    )


def _load_source() -> tuple[pl.DataFrame, pl.DataFrame]:
    daily = (
        pl.read_csv(SOURCE_DAILY)
        .filter((pl.col("threshold") - THRESHOLD).abs() < EPSILON)
        .with_columns(pl.col("Date").cast(pl.Utf8))
        .filter(pl.col("Date").is_in(_trading_dates()))
        .sort("Date")
    )
    summary = pl.read_csv(SOURCE_SUMMARY).filter(
        (pl.col("threshold") - THRESHOLD).abs() < EPSILON
    )
    if daily.is_empty() or summary.height != 1:
        raise RuntimeError("missing preserved 0.75% backtest rows")
    return daily, summary


def _scale_path(daily: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    max_peak = float(daily["peak_open_notional_twd"].max())
    if max_peak <= 0:
        raise RuntimeError("invalid source peak position")
    scale = CAPITAL_TWD / max_peak
    money_columns = [
        "eod_open_notional_twd",
        "peak_open_notional_twd",
        "daily_net_pnl_twd",
        "daily_gross_pnl_twd",
        "daily_fee_twd",
        "exit_notional_twd",
        "entry_notional_twd",
    ]
    scaled = daily.with_columns(
        [(pl.col(column) * scale).alias(column) for column in money_columns]
    )
    return (
        scaled.with_columns(
            pl.col("daily_net_pnl_twd").cum_sum().alias("cum_net_pnl_twd")
        )
        .with_columns(
            (CAPITAL_TWD + pl.col("cum_net_pnl_twd")).alias("equity_twd"),
            (
                pl.col("eod_open_notional_twd") / CAPITAL_TWD * 100
            ).alias("eod_utilization_pct"),
            (
                pl.col("peak_open_notional_twd") / CAPITAL_TWD * 100
            ).alias("peak_utilization_pct"),
        ),
        scale,
    )


def _metrics(
    source_daily: pl.DataFrame,
    source_summary: pl.DataFrame,
    scaled: pl.DataFrame,
    scale: float,
) -> pl.DataFrame:
    daily_returns = scaled["daily_net_pnl_twd"].to_numpy() / CAPITAL_TWD
    realized_pnl = float(scaled["daily_net_pnl_twd"].sum())
    locked_total_pnl = float(source_summary["net_pnl_twd"][0]) * scale
    equity = scaled["equity_twd"].to_numpy()
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=1))
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    period_return = realized_pnl / CAPITAL_TWD
    annualized_return = (1 + period_return) ** (
        TRADING_DAYS_PER_YEAR / scaled.height
    ) - 1

    return pl.DataFrame(
        {
            "threshold_pct": [THRESHOLD * 100],
            "capital_twd": [CAPITAL_TWD],
            "sessions": [scaled.height],
            "static_scale_pct": [scale * 100],
            "unscaled_max_peak_yi": [
                float(source_daily["peak_open_notional_twd"].max()) / 1e8
            ],
            "realized_net_pnl_twd": [realized_pnl],
            "realized_period_return_pct": [period_return * 100],
            "annualized_return_pct": [annualized_return * 100],
            "locked_total_net_pnl_twd": [locked_total_pnl],
            "locked_total_return_pct": [locked_total_pnl / CAPITAL_TWD * 100],
            "realized_daily_sharpe": [sharpe],
            "max_drawdown_pct": [float(drawdown.min()) * 100],
            "avg_eod_utilization_pct": [
                float(scaled["eod_utilization_pct"].mean())
            ],
            "median_eod_utilization_pct": [
                float(scaled["eod_utilization_pct"].median())
            ],
            "p95_eod_utilization_pct": [
                float(scaled["eod_utilization_pct"].quantile(0.95))
            ],
            "max_eod_utilization_pct": [
                float(scaled["eod_utilization_pct"].max())
            ],
            "avg_peak_utilization_pct": [
                float(scaled["peak_utilization_pct"].mean())
            ],
            "p95_peak_utilization_pct": [
                float(scaled["peak_utilization_pct"].quantile(0.95))
            ],
            "max_peak_utilization_pct": [
                float(scaled["peak_utilization_pct"].max())
            ],
        }
    )


def _plot(daily: pl.DataFrame, metrics: pl.DataFrame) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = daily["Date"].str.strptime(pl.Date, "%Y%m%d").to_list()
    cumulative_pnl_m = daily["cum_net_pnl_twd"].to_numpy() / 1e6
    eod_m = daily["eod_open_notional_twd"].to_numpy() / 1e6
    peak_m = daily["peak_open_notional_twd"].to_numpy() / 1e6
    pnl_m = daily["daily_net_pnl_twd"].to_numpy() / 1e6
    row = metrics.row(0, named=True)

    teal = "#087E8B"
    coral = "#D95D39"
    amber = "#E9A23B"
    ink = "#1E2933"
    muted = "#66727D"
    grid = "#DDE3E8"
    background = "#F7F9FA"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": grid,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": ink,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0, 0.72], "hspace": 0.22},
    )
    fig.patch.set_facecolor(background)
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=grid, linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)

    axes[0].plot(dates, cumulative_pnl_m, color=teal, linewidth=2.4)
    axes[0].fill_between(
        dates,
        0,
        cumulative_pnl_m,
        color=teal,
        alpha=0.12,
    )
    axes[0].axhline(0, color=muted, linewidth=1)
    axes[0].set_ylabel("TWD million")
    axes[0].set_title("Cumulative realized net PnL")
    axes[0].annotate(
        f"+{cumulative_pnl_m[-1]:.2f}M",
        xy=(dates[-1], cumulative_pnl_m[-1]),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        color=teal,
        fontweight="bold",
    )

    axes[1].bar(
        dates,
        eod_m,
        width=0.9,
        color=teal,
        alpha=0.78,
        label="EOD open position",
    )
    axes[1].plot(
        dates,
        peak_m,
        color=amber,
        linewidth=1.7,
        label="Intraday peak",
    )
    axes[1].axhline(
        CAPITAL_TWD / 1e6,
        color=coral,
        linewidth=1.3,
        linestyle="--",
        label="100M budget",
    )
    axes[1].set_ylabel("TWD million")
    axes[1].set_title("Capital utilization")
    axes[1].set_ylim(0, 108)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    pnl_colors = np.where(pnl_m >= 0, teal, coral)
    axes[2].bar(dates, pnl_m, width=0.9, color=pnl_colors, alpha=0.85)
    axes[2].axhline(0, color=muted, linewidth=0.8)
    axes[2].set_ylabel("TWD million")
    axes[2].set_title("Daily realized net PnL")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[2].set_xlabel("2026")

    fig.suptitle(
        "0.75% Futures-Spot Arbitrage | TWD 100M",
        x=0.07,
        y=0.982,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.945,
        (
            f"Static scale {row['static_scale_pct']:.2f}%  |  "
            f"Realized return {row['realized_period_return_pct']:.2f}%  |  "
            f"Sharpe {row['realized_daily_sharpe']:.2f}  |  "
            f"Avg EOD use {row['avg_eod_utilization_pct']:.1f}%"
        ),
        color=muted,
        fontsize=11,
    )
    fig.text(
        0.07,
        0.018,
        (
            "Fee assumption: 38 bp. PnL is recognized on convergence/settlement; "
            "the Sharpe ratio uses realized exit-day returns, not daily mark-to-market."
        ),
        color=muted,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.075)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_report(metrics: pl.DataFrame) -> None:
    columns = metrics.columns
    row = metrics.row(0, named=True)
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        "| " + " | ".join(_format_value(row[column]) for column in columns) + " |",
    ]
    lines = [
        "# 0.75% Backtest With TWD 100M",
        "",
        f"Period: {START} to {END}. Fee assumption: 38 bp.",
        "",
        "Method: statically scale the preserved backtest by the factor required "
        "to make its historical maximum intraday open notional equal TWD 100M. "
        "This is not an event-level capital queue simulation.",
        "",
        "The realized Sharpe ratio uses exit-day realized PnL over 99 trading "
        "sessions. Because the source backtest has no daily mark-to-market PnL, "
        "this Sharpe is not directly comparable with a conventional MTM Sharpe.",
        "",
        "## Metrics",
        "",
        *table,
        "",
        "## Files",
        "",
        f"- `{OUT_DAILY}`",
        f"- `{OUT_METRICS}`",
        f"- `{OUT_PLOT}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_daily, source_summary = _load_source()
    scaled, scale = _scale_path(source_daily)
    metrics = _metrics(source_daily, source_summary, scaled, scale)
    scaled.write_csv(OUT_DAILY)
    metrics.write_csv(OUT_METRICS)
    _plot(scaled, metrics)
    _write_report(metrics)
    print(metrics)
    print(f"plot -> {OUT_PLOT}")
    print(f"report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
