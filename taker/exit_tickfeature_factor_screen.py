"""Screen stock tickFeature signals for exit convergence execution quality.

This joins stock tickFeature states to convergence exit points and screens
single-factor relationships with 30ms taker-taker exit slippage.
"""
from __future__ import annotations

import argparse
import math
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import tickfeature_factor_screen as tfs


from data_paths import (
    DATA_ROOT as DATA_DIR,
    PLOT_DIR,
    SPOT_TICK_DIR as TICK_DATA_DIR,
    STOCKFUTURE_DIR,
    TICK_FEATURE_DIR,
)

DEFAULT_EVENTS = STOCKFUTURE_DIR / "exit_convergence_30ms_events_20260126_20260629.parquet"
DEFAULT_THRESHOLDS = [0.005, 0.0075]


def _threshold_tag(thresholds: list[float]) -> str:
    return "_".join(f"{x * 100:.2f}".replace(".", "p") for x in thresholds)


def _daily_feature_frame(date: str, value_codes: list[str], feature_cols: list[str]) -> pl.DataFrame:
    tick_path = TICK_DATA_DIR / f"{date}_StockTick.parquet"
    feature_path = TICK_FEATURE_DIR / f"{date}_tickFeature.parquet"
    if not tick_path.exists() or not feature_path.exists():
        return pl.DataFrame()
    tick = (
        pl.scan_parquet(tick_path)
        .select(
            [
                "RecvTime",
                "TransTime",
                "QuoteCode",
                "ValueCode",
                "ChannelSeq",
                "TrialMatch",
                "marketOpen",
                "AskPrice1",
                "BidPrice1",
            ]
        )
        .filter(pl.col("ValueCode").is_in(value_codes))
        .filter((pl.col("TrialMatch") == 0) & pl.col("marketOpen"))
        .filter((pl.col("AskPrice1") > 0) & (pl.col("BidPrice1") > 0))
    )
    tick_feature = pl.scan_parquet(feature_path).select(["QuoteCode", "ChannelSeq", *feature_cols])
    return (
        tick.join(tick_feature, on=["QuoteCode", "ChannelSeq"], how="inner")
        .select(
            [
                "ValueCode",
                (pl.col("RecvTime").cast(pl.Datetime("us")) + timedelta(hours=8)).alias(
                    "feature_recv_time"
                ),
                pl.col("TransTime").cast(pl.Datetime("us")).alias("feature_trans_time"),
                pl.col("QuoteCode").alias("stock_quote_code"),
                pl.col("ChannelSeq").alias("stock_channel_seq"),
                *feature_cols,
            ]
        )
        .sort(["ValueCode", "feature_recv_time"])
        .collect()
    )


def load_exit_points(events_path: Path, thresholds: list[float], unique_exit_points: bool) -> pl.DataFrame:
    thresholds_bp = [int(round(x * 10000)) for x in thresholds]
    events = (
        pl.scan_parquet(events_path)
        .with_row_index("exit_row_id")
        .with_columns((pl.col("threshold") * 10000).round(0).cast(pl.Int64).alias("_threshold_bp"))
        .filter(pl.col("_threshold_bp").is_in(thresholds_bp))
        .filter(pl.col("exit_30ms_valid").fill_null(False))
        .drop("_threshold_bp")
        .with_columns(
            [
                pl.col("Date").cast(pl.Utf8),
                pl.col("converge_time").cast(pl.Datetime("us")).alias("exit_signal_time"),
                pl.col("target_exit_30ms").cast(pl.Datetime("us")).alias("exit_exec_time_30ms"),
                (pl.col("exit_30ms_nonnegative").cast(pl.Int8)).alias(
                    "exit_30ms_nonnegative_y"
                ),
                (pl.col("signal_exit_bp") - pl.col("exit_30ms_bp")).alias(
                    "exit_slippage_loss_bp"
                ),
                (-pl.col("exit_30ms_bp")).alias("exit_negative_cost_bp"),
            ]
        )
        .collect()
    )
    if unique_exit_points:
        events = events.sort(["threshold", "QuoteCode", "converge_time", "exit_row_id"]).unique(
            subset=["threshold", "QuoteCode", "converge_time"], keep="first", maintain_order=True
        )
    return events


def build_enriched_exit_points(events: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in events["Date"].unique().sort().to_list():
        ev = events.filter(pl.col("Date") == date)
        value_codes = ev["ValueCode"].unique().to_list()
        feature_state = _daily_feature_frame(date, value_codes, feature_cols)
        if feature_state.height == 0:
            continue
        joined = (
            ev.sort(["ValueCode", "exit_signal_time"])
            .join_asof(
                feature_state,
                left_on="exit_signal_time",
                right_on="feature_recv_time",
                by="ValueCode",
                strategy="backward",
            )
            .with_columns(
                (
                    (pl.col("exit_signal_time") - pl.col("feature_recv_time"))
                    .dt.total_microseconds()
                    .truediv(1000.0)
                ).alias("feature_lag_ms")
            )
        )
        frames.append(joined)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _daily_ic_table(
    pdf: pd.DataFrame, feature_cols: list[str], target_cols: list[str], min_daily_rows: int
) -> pd.DataFrame:
    rows = []
    grouped = pdf.groupby(["threshold", "Date"], sort=True)
    for (threshold, date), day in grouped:
        if len(day) < min_daily_rows:
            continue
        for target in target_cols:
            y = day[target].to_numpy(dtype=np.float64)
            if np.nanstd(y) <= 0:
                continue
            for feature in feature_cols:
                x = day[feature].to_numpy(dtype=np.float64)
                ic = tfs._spearman_ic(x, y)
                if np.isnan(ic):
                    continue
                rows.append(
                    {
                        "threshold": threshold,
                        "Date": date,
                        "target": target,
                        "feature": feature,
                        "n": int((~(np.isnan(x) | np.isnan(y))).sum()),
                        "daily_ic": ic,
                    }
                )
    return pd.DataFrame(rows)


def _quintile_spread(
    pdf: pd.DataFrame, feature_cols: list[str], target_cols: list[str], min_rows: int
) -> pd.DataFrame:
    rows = []
    for threshold, sub in pdf.groupby("threshold", sort=True):
        for target in target_cols:
            y_all = sub[target].to_numpy(dtype=np.float64)
            for feature in feature_cols:
                xy = sub[[feature, target]].dropna()
                if len(xy) < min_rows or xy[feature].nunique() < 5:
                    continue
                try:
                    bucket = pd.qcut(xy[feature], q=5, labels=False, duplicates="drop")
                except ValueError:
                    continue
                xy = xy.assign(bucket=bucket)
                if xy["bucket"].nunique() < 2:
                    continue
                low = xy.loc[xy["bucket"] == xy["bucket"].min(), target].to_numpy(dtype=np.float64)
                high = xy.loc[xy["bucket"] == xy["bucket"].max(), target].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "threshold": threshold,
                        "target": target,
                        "feature": feature,
                        "n": len(xy),
                        "low_mean": float(np.nanmean(low)),
                        "high_mean": float(np.nanmean(high)),
                        "high_minus_low": float(np.nanmean(high) - np.nanmean(low)),
                        "low_median": float(np.nanmedian(low)),
                        "high_median": float(np.nanmedian(high)),
                        "high_minus_low_median": float(np.nanmedian(high) - np.nanmedian(low)),
                        "pooled_ic": tfs._spearman_ic(
                            sub[feature].to_numpy(dtype=np.float64), y_all
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_factors(
    enriched: pl.DataFrame,
    feature_cols: list[str],
    *,
    max_lag_ms: float,
    min_daily_rows: int,
    min_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = [
        "exit_30ms_bp",
        "exit_slippage_loss_bp",
        "exit_30ms_nonnegative_y",
        "exit_30ms_minus_signal_bp",
    ]
    keep_cols = [
        "exit_row_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "threshold",
        "exit_signal_time",
        "feature_lag_ms",
        *target_cols,
        *feature_cols,
    ]
    pdf = (
        enriched.select([c for c in keep_cols if c in enriched.columns])
        .filter(pl.col("feature_lag_ms").is_not_null() & (pl.col("feature_lag_ms") <= max_lag_ms))
        .to_pandas()
    )
    pdf["threshold"] = (pdf["threshold"] * 100).round(2)
    daily_ic = _daily_ic_table(pdf, feature_cols, target_cols, min_daily_rows)
    qspread = _quintile_spread(pdf, feature_cols, target_cols, min_rows)
    if daily_ic.empty:
        return daily_ic, qspread, pd.DataFrame()
    pooled = qspread[["threshold", "target", "feature", "pooled_ic", "n"]].drop_duplicates()
    summary = (
        daily_ic.groupby(["threshold", "target", "feature"], as_index=False)
        .agg(
            valid_days=("daily_ic", "count"),
            sample_n_median=("n", "median"),
            daily_ic_mean=("daily_ic", "mean"),
            daily_ic_std=("daily_ic", "std"),
            daily_ic_abs_mean=("daily_ic", lambda x: float(np.nanmean(np.abs(x)))),
        )
        .merge(pooled, on=["threshold", "target", "feature"], how="left")
    )
    summary["daily_ic_ir"] = summary["daily_ic_mean"] / summary["daily_ic_std"].replace(0, np.nan)
    sign_lookup = summary.set_index(["threshold", "target", "feature"])["pooled_ic"].to_dict()
    stable_rows = []
    for key, grp in daily_ic.groupby(["threshold", "target", "feature"], sort=False):
        pooled_ic = sign_lookup.get(key, np.nan)
        if np.isnan(pooled_ic) or abs(pooled_ic) <= 1e-12:
            stable = np.nan
        else:
            stable = float((np.sign(grp["daily_ic"]) == np.sign(pooled_ic)).mean())
        stable_rows.append((*key, stable))
    stability = pd.DataFrame(stable_rows, columns=["threshold", "target", "feature", "sign_stability"])
    summary = summary.merge(stability, on=["threshold", "target", "feature"], how="left")
    summary["factor_score"] = (
        summary["daily_ic_mean"].abs()
        * summary["sign_stability"].fillna(0)
        * np.minimum(summary["valid_days"] / 20.0, 1.0)
    )
    summary = summary.merge(
        qspread[
            [
                "threshold",
                "target",
                "feature",
                "high_minus_low",
                "low_mean",
                "high_mean",
                "low_median",
                "high_median",
                "high_minus_low_median",
            ]
        ],
        on=["threshold", "target", "feature"],
        how="left",
    )
    return daily_ic, qspread, summary.sort_values(
        ["threshold", "target", "factor_score"], ascending=[True, True, False]
    )


def plot_top_features(summary: pd.DataFrame, tag: str, top_n: int) -> list[Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if summary.empty:
        return paths
    for (threshold, target), sub in summary.groupby(["threshold", "target"], sort=True):
        top = sub.sort_values("factor_score", ascending=False).head(top_n)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
        colors = np.where(top["daily_ic_mean"] >= 0, "#2563eb", "#dc2626")
        ax.barh(top["feature"], top["daily_ic_mean"], color=colors)
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean daily Spearman IC")
        ax.set_title(f"Exit factors threshold {threshold:.2f}% target {target}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = (
            PLOT_DIR
            / f"exit_tickfeature_factor_ic_{tag}_thr{str(threshold).replace('.', 'p')}_{target}.png"
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    enriched: pl.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
    plot_paths: list[Path],
) -> None:
    sample = (
        enriched.with_columns((pl.col("threshold") * 100).round(2).alias("threshold_pct"))
        .group_by("threshold_pct")
        .agg(
            [
                pl.len().alias("asof_rows"),
                (pl.col("feature_lag_ms") <= args.max_lag_ms).sum().alias("screen_rows"),
                pl.col("feature_lag_ms").median().alias("lag_ms_median"),
                pl.col("feature_lag_ms").quantile(0.95).alias("lag_ms_p95"),
                (pl.col("exit_30ms_nonnegative_y").mean() * 100).alias(
                    "nonnegative_pct_raw"
                ),
                pl.col("exit_30ms_bp").median().alias("exit_30ms_bp_median_raw"),
            ]
        )
        .sort("threshold_pct")
        .to_pandas()
    )
    lines = [
        "# Exit TickFeature Factor Screen",
        "",
        f"Exit events: `{args.events_path}`",
        f"Unique exit point mode: `{not args.event_level}`",
        f"Thresholds: {', '.join(f'{x * 100:.2f}%' for x in args.thresholds)}",
        f"Feature lag used for IC: <= {args.max_lag_ms:,.0f} ms",
        "",
        "Feature states are joined as-of to `converge_time`, before the 30ms execution measurement.",
        "",
        "Target direction: higher `exit_30ms_bp` and `exit_30ms_nonnegative_y` are better; lower `exit_slippage_loss_bp` is better.",
        "",
        "## Sample",
        "",
        _to_markdown_table(sample.round(4)),
        "",
        "## Top Factors",
        "",
    ]
    for target in [
        "exit_30ms_bp",
        "exit_slippage_loss_bp",
        "exit_30ms_nonnegative_y",
    ]:
        top = (
            summary[summary["target"] == target]
            .sort_values(["threshold", "factor_score"], ascending=[True, False])
            .groupby("threshold")
            .head(10)
        )
        if top.empty:
            continue
        show_cols = [
            "threshold",
            "target",
            "feature",
            "valid_days",
            "daily_ic_mean",
            "pooled_ic",
            "sign_stability",
            "low_mean",
            "high_mean",
            "high_minus_low",
            "factor_score",
        ]
        lines.extend([f"### {target}", "", _to_markdown_table(top[show_cols].round(4)), ""])
    lines.extend(["## Files", ""])
    for label, path in paths.items():
        lines.append(f"- {label}: `{path}`")
    for path in plot_paths:
        lines.append(f"- plot: `{path}`")
    paths["report"].write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20260126")
    parser.add_argument("--end-date", default="20260629")
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--max-lag-ms", type=float, default=60_000.0)
    parser.add_argument("--min-daily-rows", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=150)
    parser.add_argument("--top-n-plot", type=int, default=15)
    parser.add_argument("--event-level", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unique_exit_points = not args.event_level
    tag = (
        f"{args.start_date}_{args.end_date}_thr{_threshold_tag(args.thresholds)}_"
        f"{'unique_exit' if unique_exit_points else 'event_level'}"
    )
    events = load_exit_points(args.events_path, args.thresholds, unique_exit_points)
    events = events.filter((pl.col("Date") >= args.start_date) & (pl.col("Date") <= args.end_date))
    if events.height == 0:
        raise SystemExit("No exit events matched filters.")
    feature_cols = tfs.candidate_tickfeature_columns(events["Date"].max())
    enriched = build_enriched_exit_points(events, feature_cols)
    if enriched.height == 0:
        raise SystemExit("No enriched rows produced.")
    daily_ic, qspread, summary = summarize_factors(
        enriched,
        feature_cols,
        max_lag_ms=args.max_lag_ms,
        min_daily_rows=args.min_daily_rows,
        min_rows=args.min_rows,
    )
    paths = {
        "enriched": STOCKFUTURE_DIR / f"exit_tickfeature_enriched_{tag}.parquet",
        "daily_ic": STOCKFUTURE_DIR / f"exit_tickfeature_daily_ic_{tag}.csv",
        "quintile": STOCKFUTURE_DIR / f"exit_tickfeature_quintile_spread_{tag}.csv",
        "summary": STOCKFUTURE_DIR / f"exit_tickfeature_factor_summary_{tag}.csv",
        "report": STOCKFUTURE_DIR / f"exit_tickfeature_factor_report_{tag}.md",
    }
    enriched.write_parquet(paths["enriched"])
    daily_ic.to_csv(paths["daily_ic"], index=False)
    qspread.to_csv(paths["quintile"], index=False)
    summary.to_csv(paths["summary"], index=False)
    plot_paths = plot_top_features(summary, tag, args.top_n_plot)
    write_report(summary, enriched, args, paths, plot_paths)
    print(f"exit events: {events.height:,}")
    print(f"enriched rows: {enriched.height:,}")
    print(f"feature cols: {len(feature_cols):,}")
    print(f"daily IC rows: {len(daily_ic):,}")
    print(f"summary rows: {len(summary):,}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
