"""Build the compact policy comparison figure used by the 2026-08-21 report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl


ROOT = Path(__file__).resolve().parents[3]
NORMAL_DAILY = (
    ROOT
    / "maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only/cap_daily.parquet"
)
AGGRESSIVE_DAILY = (
    ROOT
    / "maker/data/walkforward/aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix/controller_daily_control.parquet"
)
OUTPUT = (
    ROOT
    / "maker/doc/quote_fill/assets/portfolio_policy_comparison_20260821.png"
)


def normal_series(cap: float) -> pl.DataFrame:
    return (
        pl.read_parquet(NORMAL_DAILY)
        .filter(
            (pl.col("entry_cutoff_variant") == "normal_cutoff_1300")
            & (pl.col("hard_intraday_cap_twd") == cap)
        )
        .select(
            pl.col("Date").str.strptime(pl.Date, "%Y%m%d"),
            "realized_net_pnl_twd",
            pl.col("eod_outstanding_one_way_notional_twd").alias("carry_twd"),
            pl.col("accepted_entry_one_way_turnover_twd").alias("entry_twd"),
        )
        .sort("Date")
        .with_columns(
            pl.col("realized_net_pnl_twd").cum_sum().alias("cumulative_net_twd"),
            pl.col("entry_twd").rolling_mean(window_size=5, min_samples=1).alias(
                "entry_rolling_5_twd"
            ),
        )
    )


def aggressive_series(cap: float) -> pl.DataFrame:
    return (
        pl.read_parquet(AGGRESSIVE_DAILY)
        .filter(
            pl.col("primary_conservative_result")
            & (pl.col("portfolio_cap_twd") == cap)
        )
        .select(
            pl.col("Date").str.strptime(pl.Date, "%Y%m%d"),
            "realized_net_pnl_twd",
            pl.col("eod_active_notional_twd").alias("carry_twd"),
            pl.col("accepted_entry_one_way_turnover_twd").alias("entry_twd"),
        )
        .sort("Date")
        .with_columns(
            pl.col("realized_net_pnl_twd").cum_sum().alias("cumulative_net_twd"),
            pl.col("entry_twd").rolling_mean(window_size=5, min_samples=1).alias(
                "entry_rolling_5_twd"
            ),
        )
    )


def plot_line(ax: plt.Axes, frame: pl.DataFrame, column: str, **kwargs: object) -> None:
    ax.plot(frame["Date"].to_list(), (frame[column] / 1_000_000).to_list(), **kwargs)


def main() -> None:
    series = [
        ("Normal 20M", normal_series(20_000_000.0), "#3b82f6", "-"),
        ("Normal 30M", normal_series(30_000_000.0), "#64748b", "--"),
        ("Aggressive 30M / target 20M", aggressive_series(30_000_000.0), "#f97316", "-"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "AB1/2 portfolio policy comparison — 63 sessions",
        fontsize=15,
        fontweight="bold",
    )

    for label, frame, color, linestyle in series:
        plot_line(
            axes[0],
            frame,
            "cumulative_net_twd",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
        plot_line(
            axes[1],
            frame,
            "carry_twd",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )
        plot_line(
            axes[2],
            frame,
            "entry_rolling_5_twd",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )

    axes[0].set_ylabel("Cumulative net (TWD M)")
    axes[1].set_ylabel("13:20 carry (TWD M)")
    axes[2].set_ylabel("Entry turnover (TWD M)\n5-session mean")
    axes[1].axhline(20.0, color="#dc2626", linestyle=":", linewidth=1.5, label="20M target")

    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.axvline(date(2026, 8, 1), color="#94a3b8", linestyle=":", linewidth=1.0)
        ax.legend(loc="upper left", frameon=False, ncols=2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].set_xlabel("Trading date (vertical dotted line: 2026-08-01)")
    fig.text(
        0.5,
        0.01,
        "After-cost realized P&L only; no carry MTM. Replay exposure ends at 13:20. "
        "Retrospective 45-product raw cohort.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
