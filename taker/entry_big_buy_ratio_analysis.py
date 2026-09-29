"""Study entry slippage versus intraday buy size normalized by premarket history.

The denominator is ``big_buy_lots`` from the same trade day's preMarketData.
That file is built from the previous trading day's ticks, so the ratio is
available before the entry signal occurs.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


from data_paths import (
    DATA_ROOT as DATA_DIR,
    PLOT_DIR,
    PREMARKET_DIR,
    SPOT_TICK_DIR as TICK_DATA_DIR,
    STOCKFUTURE_DIR,
    TICK_FEATURE_DIR,
)

DEFAULT_INPUT = (
    STOCKFUTURE_DIR
    / "entry_slippage_a1a2_1tick_tickbp_detail_20260126_20260629_thr0p50_0p75.parquet"
)
DEFAULT_CUTOFFS = [
    0.0,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.75,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    8.0,
    10.0,
]
THRESHOLD = 0.0075
FLOAT_EPSILON = 1e-9
MAX_FEATURE_LAG_MS = 60_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--max-feature-lag-ms", type=float, default=MAX_FEATURE_LAG_MS)
    parser.add_argument("--window", type=int, choices=[10, 30], default=10)
    return parser.parse_args()


def load_events(path: Path, threshold: float) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .with_row_index("event_id")
        .filter((pl.col("threshold") - threshold).abs() < FLOAT_EPSILON)
        .with_columns(pl.col("entry_time").cast(pl.Datetime("us")))
        .collect()
    )


def load_daily_feature_state(
    date: str, value_codes: list[str], feature_column: str
) -> pl.DataFrame:
    tick_path = TICK_DATA_DIR / f"{date}_StockTick.parquet"
    feature_path = TICK_FEATURE_DIR / f"{date}_tickFeature.parquet"
    if not tick_path.exists() or not feature_path.exists():
        return pl.DataFrame()

    ticks = (
        pl.scan_parquet(tick_path)
        .select(
            "RecvTime",
            "QuoteCode",
            "ValueCode",
            "ChannelSeq",
            "TrialMatch",
            "marketOpen",
        )
        .filter(pl.col("ValueCode").is_in(value_codes))
        .filter((pl.col("TrialMatch") == 0) & pl.col("marketOpen"))
    )
    features = pl.scan_parquet(feature_path).select(
        "QuoteCode", "ChannelSeq", feature_column
    )
    return (
        ticks.join(features, on=["QuoteCode", "ChannelSeq"], how="inner")
        .select(
            "ValueCode",
            (pl.col("RecvTime").cast(pl.Datetime("us")) + timedelta(hours=8)).alias(
                "feature_time"
            ),
            pl.col(feature_column).cast(pl.Float64).alias("l1_buy_biggest_lots"),
        )
        .sort("ValueCode", "feature_time")
        .collect()
    )


def load_premarket(date: str, value_codes: list[str]) -> pl.DataFrame:
    path = PREMARKET_DIR / f"{date}_preMarketData.parquet"
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .select(
            pl.col("QuoteCode").alias("ValueCode"),
            pl.col("big_buy_lots").cast(pl.Float64),
        )
        .filter(pl.col("ValueCode").is_in(value_codes))
        .unique("ValueCode", keep="last")
        .collect()
    )


def enrich_events(
    events: pl.DataFrame, max_feature_lag_ms: float, feature_column: str
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    dates = events["Date"].unique().sort().to_list()
    for index, date in enumerate(dates, start=1):
        daily_events = events.filter(pl.col("Date") == date)
        value_codes = daily_events["ValueCode"].unique().to_list()
        feature_state = load_daily_feature_state(date, value_codes, feature_column)
        premarket = load_premarket(date, value_codes)
        if feature_state.is_empty() or premarket.is_empty():
            print(f"{date}: skipped, missing feature state or preMarketData")
            continue

        joined = (
            daily_events.sort("ValueCode", "entry_time")
            .join_asof(
                feature_state,
                left_on="entry_time",
                right_on="feature_time",
                by="ValueCode",
                strategy="backward",
            )
            .join(premarket, on="ValueCode", how="left")
            .with_columns(
                (
                    (pl.col("entry_time") - pl.col("feature_time"))
                    .dt.total_microseconds()
                    / 1000.0
                ).alias("feature_lag_ms")
            )
            .with_columns(
                pl.when(
                    (pl.col("feature_lag_ms") <= max_feature_lag_ms)
                    & (pl.col("big_buy_lots") > 0)
                )
                .then(pl.col("l1_buy_biggest_lots") / pl.col("big_buy_lots"))
                .otherwise(None)
                .alias("big_buy_ratio")
            )
        )
        frames.append(joined)
        if index % 10 == 0 or index == len(dates):
            print(f"joined {index}/{len(dates)} dates")

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def weighted_mean_expr(value: str, weight: str) -> pl.Expr:
    return (
        (pl.col(value) * pl.col(weight)).sum() / pl.col(weight).sum()
    )


def summarize_cutoffs(enriched: pl.DataFrame, cutoffs: list[float]) -> pl.DataFrame:
    valid = enriched.filter(pl.col("big_buy_ratio").is_not_null())
    baseline_n = valid.height
    baseline_notional = valid["entry_notional_twd"].sum()
    baseline_wslip = valid.select(
        weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
    ).item()
    dates = valid["Date"].unique().sort().to_list()
    split_date = dates[len(dates) // 2]
    baseline_first = valid.filter(pl.col("Date") < split_date).select(
        weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
    ).item()
    baseline_second = valid.filter(pl.col("Date") >= split_date).select(
        weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
    ).item()

    baseline_daily = (
        valid.group_by("Date")
        .agg(
            weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias(
                "baseline_daily_wslip_bp"
            )
        )
    )
    rows: list[dict[str, float | int | str]] = []
    scenarios: list[tuple[str, pl.DataFrame]] = [("unfiltered", valid)]
    scenarios.extend(
        (f"ratio_le_{cutoff:g}", valid.filter(pl.col("big_buy_ratio") <= cutoff))
        for cutoff in cutoffs
    )
    for scenario, sample in scenarios:
        if sample.is_empty():
            continue
        notional = sample["entry_notional_twd"].sum()
        wslip = sample.select(
            weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
        ).item()
        daily = (
            sample.group_by("Date")
            .agg(
                weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias(
                    "filtered_daily_wslip_bp"
                )
            )
            .join(baseline_daily, on="Date", how="left")
            .with_columns(
                (
                    pl.col("baseline_daily_wslip_bp")
                    - pl.col("filtered_daily_wslip_bp")
                ).alias("daily_improvement_bp")
            )
        )
        first = sample.filter(pl.col("Date") < split_date)
        second = sample.filter(pl.col("Date") >= split_date)
        first_wslip = first.select(
            weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
        ).item()
        second_wslip = second.select(
            weighted_mean_expr("spot_slip_bp", "entry_notional_twd").alias("value")
        ).item()
        rows.append(
            {
                "scenario": scenario,
                "ratio_cutoff": (
                    float("inf")
                    if scenario == "unfiltered"
                    else float(scenario.removeprefix("ratio_le_"))
                ),
                "samples": sample.height,
                "sample_retention_pct": sample.height / baseline_n * 100.0,
                "notional_yi": notional / 100_000_000.0,
                "notional_retention_pct": notional / baseline_notional * 100.0,
                "symbols": sample["ValueCode"].n_unique(),
                "dates": sample["Date"].n_unique(),
                "slip_rate_pct": sample["did_slip"].mean() * 100.0,
                "spot_slip_bp_mean": sample["spot_slip_bp"].mean(),
                "spot_slip_bp_median": sample["spot_slip_bp"].median(),
                "spot_slip_bp_wavg": wslip,
                "wavg_improvement_bp": baseline_wslip - wslip,
                "daily_improvement_bp_median": daily[
                    "daily_improvement_bp"
                ].median(),
                "daily_improvement_positive_pct": (
                    (daily["daily_improvement_bp"] > 0).mean() * 100.0
                ),
                "first_half_improvement_bp": baseline_first - first_wslip,
                "second_half_improvement_bp": baseline_second - second_wslip,
            }
        )
    return pl.DataFrame(rows)


def ratio_distribution(enriched: pl.DataFrame) -> pl.DataFrame:
    valid = enriched.filter(pl.col("big_buy_ratio").is_not_null())
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0]
    return pl.DataFrame(
        {
            "quantile": quantiles,
            "big_buy_ratio": [
                valid["big_buy_ratio"].quantile(q, interpolation="linear")
                for q in quantiles
            ],
        }
    )


def plot_summary(summary: pl.DataFrame, path: Path) -> None:
    plot_df = summary.filter(pl.col("scenario") != "unfiltered").sort("ratio_cutoff")
    x = plot_df["ratio_cutoff"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    axes[0].plot(x, plot_df["spot_slip_bp_wavg"], marker="o", color="#1565c0")
    axes[0].set_title("Weighted spot slippage")
    axes[0].set_xlabel("L1 buy max / previous-day big buy lots cutoff")
    axes[0].set_ylabel("Basis points")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        plot_df["notional_retention_pct"],
        marker="o",
        color="#00897b",
        label="Notional retained",
    )
    axes[1].plot(
        x,
        plot_df["sample_retention_pct"],
        marker="s",
        color="#ef6c00",
        label="Samples retained",
    )
    axes[1].set_title("Capacity retained")
    axes[1].set_xlabel("Ratio cutoff")
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pl.DataFrame, decimals: int) -> str:
    columns = frame.columns
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.iter_rows():
        cells = []
        for value in row:
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    summary: pl.DataFrame,
    distribution: pl.DataFrame,
    enriched: pl.DataFrame,
    output_path: Path,
    plot_path: Path,
    window: int,
) -> None:
    valid = enriched.filter(pl.col("big_buy_ratio").is_not_null())
    coverage = valid.height / enriched.height * 100.0 if enriched.height else 0.0
    display = summary.with_columns(
        pl.when(pl.col("ratio_cutoff").is_infinite())
        .then(None)
        .otherwise(pl.col("ratio_cutoff"))
        .alias("ratio_cutoff")
    )
    lines = [
        "# Previous-day Big-buy Ratio Entry Slippage",
        "",
        "Universe: 0.75% futures-first executable entries with spot A1-A2 equal to one tick.",
        "",
        f"`big_buy_ratio = L1_BuyBiggestLots_{window} / preMarketData.big_buy_lots`.",
        "The preMarketData field is computed from the previous trading day and is known before entry.",
        "The latest stock feature state is accepted up to 60 seconds old, matching the original factor screen.",
        "",
        f"Feature and denominator coverage: {valid.height}/{enriched.height} ({coverage:.2f}%).",
        "",
        "## Cutoff Sweep",
        "",
        markdown_table(display, decimals=2),
        "",
        "## Ratio Distribution",
        "",
        markdown_table(distribution, decimals=3),
        "",
        "## Plot",
        "",
        f"- {plot_path}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    events = load_events(args.input, args.threshold)
    feature_column = f"L1_BuyBiggestLots_{args.window}"
    enriched = enrich_events(events, args.max_feature_lag_ms, feature_column)
    if enriched.is_empty():
        raise RuntimeError("No events could be enriched")

    date_tag = f"{events['Date'].min()}_{events['Date'].max()}"
    feature_tag = f"l1buy{args.window}"
    detail_path = STOCKFUTURE_DIR / f"entry_big_buy_ratio_{feature_tag}_detail_{date_tag}_thr0p75.parquet"
    summary_path = STOCKFUTURE_DIR / f"entry_big_buy_ratio_{feature_tag}_cutoff_summary_{date_tag}_thr0p75.csv"
    report_path = STOCKFUTURE_DIR / f"entry_big_buy_ratio_{feature_tag}_report_{date_tag}_thr0p75.md"
    plot_path = PLOT_DIR / f"entry_big_buy_ratio_{feature_tag}_cutoff_{date_tag}_thr0p75.png"

    summary = summarize_cutoffs(enriched, DEFAULT_CUTOFFS)
    distribution = ratio_distribution(enriched)
    enriched.write_parquet(detail_path, compression="zstd")
    summary.write_csv(summary_path)
    plot_summary(summary, plot_path)
    write_report(summary, distribution, enriched, report_path, plot_path, args.window)
    print(summary)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
