"""Join stock tickFeature states to futures/spot arbitrage triggers and screen IC.

The join is intentionally event-time only:

1. tickFeature rows are first joined to stock tickData to recover RecvTime and
   TransTime by QuoteCode x ChannelSeq.
2. Arbitrage trigger rows are as-of joined to the latest stock feature state by
   ValueCode and RecvTime.
3. Single-factor diagnostics are computed for 0.50% and 0.75% thresholds by
   daily Spearman IC plus simple quintile spreads.

Example:
  uv run python src/research/futures_spot_spread/taker/tickfeature_factor_screen.py \
    --start-date 20260126 --end-date 20260629
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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
STOCKFUTURE_DIR = DATA_DIR / "stockfuture"
TICK_DATA_DIR = DATA_DIR / "tickData"
TICK_FEATURE_DIR = DATA_DIR / "tickFeature"
PLOT_DIR = STOCKFUTURE_DIR / "plots"

DEFAULT_TRADES = (
    STOCKFUTURE_DIR
    / "backtest_trades_20260126_20260629_ref9_symbol_open_base_plus_volume10_fee38p0bp.parquet"
)

DEFAULT_THRESHOLDS = [0.005, 0.0075]
KEY_COLS = {"QuoteCode", "ChannelSeq"}
LEAKY_PREFIXES = ("FutureAsk1_", "FutureBid1_", "midEdge_")
LEAKY_COLS = {"TakerSell_CloseBP", "TakerBuy_CloseBP"}


def _threshold_tag(thresholds: list[float]) -> str:
    return "_".join(f"{x * 100:.2f}".replace(".", "p") for x in thresholds)


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]
    n = len(x_valid)
    if n < 3:
        return np.nan
    if np.nanstd(x_valid) <= 0 or np.nanstd(y_valid) <= 0:
        return np.nan
    rank_x = pd.Series(x_valid).rank(method="average").to_numpy(dtype=np.float64)
    rank_y = pd.Series(y_valid).rank(method="average").to_numpy(dtype=np.float64)
    rx = rank_x - rank_x.mean()
    ry = rank_y - rank_y.mean()
    denom = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return np.nan
    return float(np.sum(rx * ry) / denom)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = ~(np.isnan(values) | np.isnan(weights)) & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _is_safe_feature(name: str, dtype: pl.DataType) -> bool:
    if name in KEY_COLS or name in LEAKY_COLS:
        return False
    if any(name.startswith(prefix) for prefix in LEAKY_PREFIXES):
        return False
    return dtype.is_numeric()


def candidate_tickfeature_columns(sample_date: str) -> list[str]:
    path = TICK_FEATURE_DIR / f"{sample_date}_tickFeature.parquet"
    schema = pl.scan_parquet(path).collect_schema()
    return [name for name, dtype in schema.items() if _is_safe_feature(name, dtype)]


def load_events(trades_path: Path, thresholds: list[float]) -> pl.DataFrame:
    thresholds_bp = [int(round(x * 10000)) for x in thresholds]
    events = (
        pl.scan_parquet(trades_path)
        .with_row_index("event_row_id")
        .with_columns((pl.col("threshold") * 10000).round(0).cast(pl.Int64).alias("_threshold_bp"))
        .filter(pl.col("_threshold_bp").is_in(thresholds_bp))
        .drop("_threshold_bp")
        .with_columns([
            ((pl.col("entry_spot_buy_price") - pl.col("signal_spot_ask"))
             / pl.col("signal_spot_ask") * 10000).alias("spot_slippage_bp"),
            (pl.col("status_full") == "當日收斂").cast(pl.Int8).alias("same_day_y"),
            (pl.col("status_full") == "跨日收斂").cast(pl.Int8).alias("cross_day_y"),
            pl.col("status_full").str.contains("沒收斂").cast(pl.Int8).alias("settlement_y"),
            (
                pl.col("exit_date").str.strptime(pl.Date, "%Y%m%d")
                - pl.col("entry_date").str.strptime(pl.Date, "%Y%m%d")
            ).dt.total_days().alias("holding_calendar_days"),
        ])
        .collect()
    )
    return events


def _daily_feature_frame(date: str, value_codes: list[str], feature_cols: list[str]) -> pl.DataFrame:
    tick_path = TICK_DATA_DIR / f"{date}_StockTick.parquet"
    feature_path = TICK_FEATURE_DIR / f"{date}_tickFeature.parquet"
    if not tick_path.exists() or not feature_path.exists():
        return pl.DataFrame()

    tick = (
        pl.scan_parquet(tick_path)
        .select([
            "RecvTime",
            "TransTime",
            "QuoteCode",
            "ValueCode",
            "ChannelSeq",
            "TrialMatch",
            "marketOpen",
            "AskPrice1",
            "BidPrice1",
        ])
        .filter(pl.col("ValueCode").is_in(value_codes))
        .filter((pl.col("TrialMatch") == 0) & pl.col("marketOpen"))
        .filter((pl.col("AskPrice1") > 0) & (pl.col("BidPrice1") > 0))
    )
    tick_feature = pl.scan_parquet(feature_path).select(["QuoteCode", "ChannelSeq", *feature_cols])
    return (
        tick.join(tick_feature, on=["QuoteCode", "ChannelSeq"], how="inner")
        .select([
            "ValueCode",
            (pl.col("RecvTime").cast(pl.Datetime("us")) + timedelta(hours=8)).alias("feature_recv_time"),
            pl.col("TransTime").cast(pl.Datetime("us")).alias("feature_trans_time"),
            pl.col("QuoteCode").alias("stock_quote_code"),
            pl.col("ChannelSeq").alias("stock_channel_seq"),
            *feature_cols,
        ])
        .sort(["ValueCode", "feature_recv_time"])
        .collect()
    )


def build_enriched_events(events: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in events["Date"].unique().sort().to_list():
        ev = events.filter(pl.col("Date") == date)
        value_codes = ev["ValueCode"].unique().to_list()
        feature_state = _daily_feature_frame(date, value_codes, feature_cols)
        if feature_state.height == 0:
            continue
        joined = ev.with_columns(
            pl.col("entry_time").cast(pl.Datetime("us")).alias("entry_time_us")
        ).sort(["ValueCode", "entry_time_us"]).join_asof(
            feature_state,
            left_on="entry_time_us",
            right_on="feature_recv_time",
            by="ValueCode",
            strategy="backward",
        )
        frames.append(
            joined.with_columns([
                (
                    (pl.col("entry_time_us") - pl.col("feature_recv_time"))
                    .dt.total_microseconds()
                    .truediv(1000.0)
                ).alias("feature_lag_ms"),
                pl.when(pl.col("TickSize") > 0)
                .then((pl.col("entry_spot_buy_price") - pl.col("signal_spot_ask")) / pl.col("TickSize"))
                .otherwise(None)
                .alias("spot_slippage_ticks"),
            ]).drop("entry_time_us")
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _daily_ic_table(pdf: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
                    min_daily_rows: int) -> pd.DataFrame:
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
                ic = _spearman_ic(x, y)
                if np.isnan(ic):
                    continue
                rows.append({
                    "threshold": threshold,
                    "Date": date,
                    "target": target,
                    "feature": feature,
                    "n": int((~(np.isnan(x) | np.isnan(y))).sum()),
                    "daily_ic": ic,
                })
    return pd.DataFrame(rows)


def _quintile_spread(pdf: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
                     min_rows: int) -> pd.DataFrame:
    rows = []
    for threshold, sub in pdf.groupby("threshold", sort=True):
        weights = sub["entry_notional_twd"].to_numpy(dtype=np.float64)
        for target in target_cols:
            y_all = sub[target].to_numpy(dtype=np.float64)
            for feature in feature_cols:
                xy = sub[[feature, target, "entry_notional_twd"]].dropna()
                if len(xy) < min_rows or xy[feature].nunique() < 5:
                    continue
                try:
                    bucket = pd.qcut(xy[feature], q=5, labels=False, duplicates="drop")
                except ValueError:
                    continue
                xy = xy.assign(bucket=bucket)
                if xy["bucket"].nunique() < 2:
                    continue
                lo = xy.loc[xy["bucket"] == xy["bucket"].min()]
                hi = xy.loc[xy["bucket"] == xy["bucket"].max()]
                lo_y = lo[target].to_numpy(dtype=np.float64)
                hi_y = hi[target].to_numpy(dtype=np.float64)
                lo_w = lo["entry_notional_twd"].to_numpy(dtype=np.float64)
                hi_w = hi["entry_notional_twd"].to_numpy(dtype=np.float64)
                rows.append({
                    "threshold": threshold,
                    "target": target,
                    "feature": feature,
                    "n": len(xy),
                    "low_mean": float(np.nanmean(lo_y)),
                    "high_mean": float(np.nanmean(hi_y)),
                    "high_minus_low": float(np.nanmean(hi_y) - np.nanmean(lo_y)),
                    "low_wmean": _weighted_mean(lo_y, lo_w),
                    "high_wmean": _weighted_mean(hi_y, hi_w),
                    "high_minus_low_w": _weighted_mean(hi_y, hi_w) - _weighted_mean(lo_y, lo_w),
                    "pooled_ic": _spearman_ic(
                        sub[feature].to_numpy(dtype=np.float64),
                        y_all,
                    ),
                    "weight_sum_yi": float(weights.sum() / 100_000_000),
                })
    return pd.DataFrame(rows)


def summarize_factors(enriched: pl.DataFrame, feature_cols: list[str], *, max_lag_ms: float,
                      min_daily_rows: int, min_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = ["spot_slippage_bp", "spot_slippage_ticks", "same_day_y", "settlement_y", "holding_calendar_days"]
    keep_cols = [
        "event_row_id",
        "Date",
        "ValueCode",
        "threshold",
        "entry_time",
        "entry_notional_twd",
        "status_full",
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
        qspread[[
            "threshold", "target", "feature", "high_minus_low",
            "high_minus_low_w", "low_mean", "high_mean", "low_wmean", "high_wmean",
        ]],
        on=["threshold", "target", "feature"],
        how="left",
    )
    return daily_ic, qspread, summary.sort_values(
        ["threshold", "target", "factor_score"], ascending=[True, True, False]
    )


def plot_top_features(summary: pd.DataFrame, out_dir: Path, tag: str, top_n: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if summary.empty:
        return paths
    for (threshold, target), sub in summary.groupby(["threshold", "target"], sort=True):
        top = sub.sort_values("factor_score", ascending=False).head(top_n)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
        colors = np.where(top["daily_ic_mean"] >= 0, "#2f6fbb", "#c24b3a")
        ax.barh(top["feature"], top["daily_ic_mean"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean daily Spearman IC")
        ax.set_title(f"Top factors threshold {threshold:.2f}% target {target}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        target_tag = target.replace("_", "-")
        threshold_tag = f"{threshold:.2f}".replace(".", "p")
        path = out_dir / f"tickfeature_factor_ic_{tag}_thr{threshold_tag}_{target_tag}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    cols = list(frame.columns)
    rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_markdown_report(summary: pd.DataFrame, enriched: pl.DataFrame, args: argparse.Namespace,
                          paths: dict[str, Path], plot_paths: list[Path]) -> None:
    sample = (
        enriched.with_columns((pl.col("threshold") * 100).round(2).alias("threshold_pct"))
        .group_by("threshold_pct")
        .agg([
            pl.len().alias("asof_rows"),
            (pl.col("feature_lag_ms") <= args.max_lag_ms).sum().alias("screen_rows"),
            pl.col("feature_lag_ms").median().alias("lag_ms_median"),
            pl.col("feature_lag_ms").quantile(0.95).alias("lag_ms_p95"),
        ])
        .sort("threshold_pct")
        .to_pandas()
    )
    lines = [
        "# TickFeature Factor Screen",
        "",
        f"Trades: `{args.trades_path}`",
        f"Thresholds: {', '.join(f'{x * 100:.2f}%' for x in args.thresholds)}",
        f"Feature lag used for IC: <= {args.max_lag_ms:,.0f} ms",
        "",
        "## Sample",
        "",
        _to_markdown_table(sample),
        "",
        "## Top Factors",
        "",
    ]
    for target in [
        "spot_slippage_bp",
        "spot_slippage_ticks",
        "same_day_y",
        "settlement_y",
        "holding_calendar_days",
    ]:
        top = (
            summary[summary["target"] == target]
            .sort_values(["threshold", "factor_score"], ascending=[True, False])
            .groupby("threshold")
            .head(8)
        )
        if top.empty:
            continue
        show_cols = [
            "threshold", "target", "feature", "valid_days", "daily_ic_mean",
            "pooled_ic", "sign_stability", "high_minus_low", "factor_score",
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
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--max-lag-ms", type=float, default=60_000.0)
    parser.add_argument("--min-daily-rows", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--top-n-plot", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = f"{args.start_date}_{args.end_date}_thr{_threshold_tag(args.thresholds)}"
    events = load_events(args.trades_path, args.thresholds)
    events = events.filter((pl.col("Date") >= args.start_date) & (pl.col("Date") <= args.end_date))
    if events.height == 0:
        raise SystemExit("No events matched the requested date/threshold filters.")

    feature_cols = candidate_tickfeature_columns(events["Date"].max())
    enriched = build_enriched_events(events, feature_cols)
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
        "enriched": STOCKFUTURE_DIR / f"arbitrage_tickfeature_enriched_{tag}.parquet",
        "daily_ic": STOCKFUTURE_DIR / f"arbitrage_tickfeature_daily_ic_{tag}.csv",
        "quintile": STOCKFUTURE_DIR / f"arbitrage_tickfeature_quintile_spread_{tag}.csv",
        "summary": STOCKFUTURE_DIR / f"arbitrage_tickfeature_factor_summary_{tag}.csv",
        "report": STOCKFUTURE_DIR / f"arbitrage_tickfeature_factor_report_{tag}.md",
    }
    enriched.write_parquet(paths["enriched"])
    daily_ic.to_csv(paths["daily_ic"], index=False)
    qspread.to_csv(paths["quintile"], index=False)
    summary.to_csv(paths["summary"], index=False)
    plot_paths = plot_top_features(summary, PLOT_DIR, tag, args.top_n_plot)
    write_markdown_report(summary, enriched, args, paths, plot_paths)

    print(f"enriched rows: {enriched.height:,}")
    print(f"feature cols: {len(feature_cols):,}")
    print(f"daily IC rows: {len(daily_ic):,}")
    print(f"summary rows: {len(summary):,}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
