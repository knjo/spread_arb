"""Analyze executable exit spreads after futures/spot divergence events.

The entry-side studies use futures premium events (`ret_sell >= threshold`).
This script studies the other side: after a divergence has occurred, find the
next convergence point (`ret_buy >= 0`) and measure the taker-taker exit spread
30ms later using futures ask and spot bid.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import arbitrage_analysis as arb


from data_paths import PLOT_DIR, STOCKFUTURE_DIR as OUT_DIR
THRESHOLDS = [0.005, 0.0075, 0.01, 0.0125]


def _date_range(start: str, end: str) -> list[str]:
    return arb._date_range(start, end)


def _feature_path(date: str) -> Path:
    return OUT_DIR / f"{date}_arbitrage_features.parquet"


def _join_asof(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str,
    strategy: str = "backward",
    tolerance: str | None = None,
) -> pl.DataFrame:
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


def _tag_divergence_events(features: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return one row per futures-premium event (`ret_sell >= threshold`)."""
    key = "QuoteCode"
    df = features.sort([key, arb.TIME_COL])
    boundary = (
        pl.when(pl.col("ret_sell") >= threshold).then(1)
        .when(pl.col("ret_sell") <= 0).then(-1)
        .otherwise(0)
    )
    df = df.with_columns(boundary.alias("_event_boundary"))
    df = df.with_columns(
        pl.when(pl.col("_event_boundary") != 0)
        .then(pl.col("_event_boundary"))
        .otherwise(None)
        .forward_fill()
        .over(key)
        .fill_null(-1)
        .alias("_event_state")
    )
    df = df.with_columns((pl.col("_event_state") == 1).alias("in_divergence_event"))
    prev_in = pl.col("in_divergence_event").shift(1).over(key).fill_null(False)
    df = df.with_columns((pl.col("in_divergence_event") & ~prev_in).alias("is_event_start"))
    return (
        df.filter(pl.col("is_event_start"))
        .with_columns(pl.lit(threshold).alias("threshold"))
        .select(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "threshold",
                pl.col(arb.TIME_COL).alias("event_start_time"),
                pl.col("ret_sell").alias("event_start_ret_sell"),
                "fut_bid",
                "fut_ask",
                "spot_bid",
                "spot_ask",
            ]
        )
    )


def _exit_candidates(features: pl.DataFrame) -> pl.DataFrame:
    return (
        features.filter(pl.col("ret_buy") >= 0)
        .select(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                pl.col(arb.TIME_COL).alias("converge_time"),
                pl.col("ret_buy").alias("signal_exit_ret_buy"),
                pl.col("spot_bid").alias("signal_spot_bid"),
                pl.col("fut_ask").alias("signal_fut_ask"),
                pl.col("spot_time").alias("signal_spot_time"),
            ]
        )
        .sort(["QuoteCode", "converge_time"])
    )


def _exit_30ms_quotes(date: str, events: pl.DataFrame, force_fetch: bool) -> pl.DataFrame:
    if events.height == 0:
        return events

    value_codes = sorted(events["ValueCode"].unique().to_list())
    quote_codes = sorted(events["QuoteCode"].unique().to_list())
    tradable = arb._load_market_tradable(date)
    spot_ref = tradable.select(["ValueCode", "spot_ref_price"]).filter(
        pl.col("ValueCode").is_in(value_codes)
        & pl.col("spot_ref_price").is_not_null()
        & (pl.col("spot_ref_price") > 0)
    )

    fut_state = (
        arb._load_futures_prepared(date, quote_codes, value_codes, force_fetch)
        .filter(pl.col("fut_price_ok"))
        .select(
            [
                "QuoteCode",
                pl.col(arb.TIME_COL).alias("fut_after_time_30ms"),
                pl.col("fut_ask").alias("fut_ask_30ms"),
                pl.col("fut_bid").alias("fut_bid_30ms"),
            ]
        )
        .sort(["QuoteCode", "fut_after_time_30ms"])
    )
    spot = arb._prepare_spot(date, value_codes, force_fetch, spot_ref).select(
        [
            "ValueCode",
            pl.col(arb.TIME_COL).alias("spot_after_time_30ms"),
            pl.col("spot_bid").alias("spot_bid_30ms"),
            pl.col("spot_ask").alias("spot_ask_30ms"),
        ]
    )
    spot_state = arb._prepare_spot_state(date, value_codes, force_fetch, spot_ref)
    if spot_state is not None:
        spot_state = spot_state.rename(
            {
                "spot_state_time": "spot_state_time_30ms",
                "spot_trial_state": "spot_trial_state_30ms",
                "spot_price_ok": "spot_price_ok_30ms",
            }
        )

    time_dtype = events.schema["converge_time"]
    out = events.with_columns(
        (pl.col("converge_time") + pl.duration(milliseconds=30))
        .cast(time_dtype)
        .alias("target_exit_30ms")
    )
    out = _join_asof(
        out.sort(["QuoteCode", "target_exit_30ms"]),
        fut_state,
        left_on="target_exit_30ms",
        right_on="fut_after_time_30ms",
        by="QuoteCode",
    )
    out = _join_asof(
        out.sort(["ValueCode", "target_exit_30ms"]),
        spot,
        left_on="target_exit_30ms",
        right_on="spot_after_time_30ms",
        by="ValueCode",
    )
    if spot_state is not None:
        out = _join_asof(
            out.sort(["ValueCode", "target_exit_30ms"]),
            spot_state,
            left_on="target_exit_30ms",
            right_on="spot_state_time_30ms",
            by="ValueCode",
        )

    valid_spot_state = (
        (pl.col("spot_trial_state_30ms").fill_null(1) == 0)
        & pl.col("spot_price_ok_30ms").fill_null(False)
        if "spot_trial_state_30ms" in out.columns
        else pl.lit(True)
    )
    return out.with_columns(
        [
            (
                valid_spot_state
                & (pl.col("spot_bid_30ms") > 0)
                & (pl.col("fut_ask_30ms") > 0)
            ).alias("exit_30ms_valid"),
            ((pl.col("spot_bid_30ms") - pl.col("fut_ask_30ms"))).alias(
                "exit_spread_price_30ms"
            ),
            (
                (pl.col("spot_bid_30ms") - pl.col("fut_ask_30ms"))
                / pl.col("fut_ask_30ms")
            ).alias("exit_ret_buy_30ms"),
        ]
    ).with_columns(
        [
            (pl.col("signal_exit_ret_buy") * 10000).alias("signal_exit_bp"),
            (pl.col("exit_ret_buy_30ms") * 10000).alias("exit_30ms_bp"),
            (
                (pl.col("exit_ret_buy_30ms") - pl.col("signal_exit_ret_buy"))
                * 10000
            ).alias("exit_30ms_minus_signal_bp"),
            (pl.col("exit_ret_buy_30ms") >= 0).alias("exit_30ms_nonnegative"),
        ]
    )


def analyze_day(date: str, force_fetch: bool) -> pl.DataFrame:
    path = _feature_path(date)
    if not path.exists():
        return pl.DataFrame()
    features = pl.read_parquet(path)
    if features.height == 0:
        return pl.DataFrame()

    conv = _exit_candidates(features)
    parts = []
    for threshold in THRESHOLDS:
        starts = _tag_divergence_events(features, threshold)
        if starts.height == 0:
            continue
        event_level = _join_asof(
            starts.sort(["QuoteCode", "event_start_time"]),
            conv,
            left_on="event_start_time",
            right_on="converge_time",
            by="QuoteCode",
            strategy="forward",
        ).filter(pl.col("converge_time").is_not_null())
        if event_level.height:
            parts.append(event_level)
    if not parts:
        return pl.DataFrame()
    events = pl.concat(parts, how="diagonal_relaxed")
    events = events.with_row_index("event_row_id")
    return _exit_30ms_quotes(date, events, force_fetch)


def summarize(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    valid = pl.col("exit_30ms_valid")
    agg_exprs = [
        pl.len().alias("events"),
        pl.col("converge_time").n_unique().alias("unique_exit_points"),
        (valid.mean() * 100).alias("exit_30ms_valid_pct"),
        (pl.col("exit_30ms_nonnegative").filter(valid).mean() * 100).alias(
            "exit_30ms_nonnegative_pct"
        ),
        pl.col("signal_exit_bp").mean().alias("signal_exit_bp_mean"),
        pl.col("signal_exit_bp").median().alias("signal_exit_bp_median"),
        pl.col("exit_30ms_bp").filter(valid).mean().alias("exit_30ms_bp_mean"),
        pl.col("exit_30ms_bp").filter(valid).median().alias("exit_30ms_bp_median"),
        pl.col("exit_30ms_bp").filter(valid).quantile(0.1).alias("exit_30ms_bp_p10"),
        pl.col("exit_30ms_bp").filter(valid).quantile(0.9).alias("exit_30ms_bp_p90"),
        pl.col("exit_30ms_minus_signal_bp")
        .filter(valid)
        .mean()
        .alias("exit_30ms_minus_signal_bp_mean"),
        pl.col("exit_30ms_minus_signal_bp")
        .filter(valid)
        .median()
        .alias("exit_30ms_minus_signal_bp_median"),
    ]
    event_summary = (
        events.group_by("threshold")
        .agg(agg_exprs)
        .with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .sort("threshold")
    )

    unique = events.unique(
        subset=["threshold", "QuoteCode", "converge_time"], keep="first"
    )
    unique_summary = (
        unique.group_by("threshold")
        .agg(agg_exprs)
        .with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .sort("threshold")
    )
    return event_summary, unique_summary


def write_report(
    events: pl.DataFrame,
    event_summary: pl.DataFrame,
    unique_summary: pl.DataFrame,
    tag: str,
) -> dict[str, Path]:
    out_events = OUT_DIR / f"exit_convergence_30ms_events_{tag}.parquet"
    out_event_summary = OUT_DIR / f"exit_convergence_30ms_event_summary_{tag}.csv"
    out_unique_summary = OUT_DIR / f"exit_convergence_30ms_unique_summary_{tag}.csv"
    out_md = OUT_DIR / f"exit_convergence_30ms_report_{tag}.md"
    events.write_parquet(out_events)
    event_summary.write_csv(out_event_summary)
    unique_summary.write_csv(out_unique_summary)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"exit_convergence_30ms_bp_hist_{tag}.png"
    pdf = events.filter(pl.col("exit_30ms_valid")).select(
        ["threshold", "exit_30ms_bp"]
    ).to_pandas()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, threshold in zip(axes.ravel(), THRESHOLDS):
        s = pdf[pdf["threshold"] == threshold]["exit_30ms_bp"]
        ax.hist(s, bins=80, range=(-30, 30), color="#2563eb", alpha=0.85)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_title(f"threshold {threshold * 100:.2f}%")
        ax.grid(True, alpha=0.25)
    fig.supxlabel("Exit ret_buy 30ms (bp)")
    fig.supylabel("Count")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    def fmt(x: object, nd: int = 2) -> str:
        if x is None:
            return ""
        return f"{float(x):.{nd}f}"

    def table(df: pl.DataFrame) -> list[str]:
        cols = [
            "threshold_pct",
            "events",
            "unique_exit_points",
            "exit_30ms_valid_pct",
            "exit_30ms_nonnegative_pct",
            "signal_exit_bp_median",
            "exit_30ms_bp_median",
            "exit_30ms_bp_mean",
            "exit_30ms_bp_p10",
            "exit_30ms_bp_p90",
            "exit_30ms_minus_signal_bp_median",
        ]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in df.select(cols).iter_rows(named=True):
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(row["threshold_pct"]),
                        str(row["events"]),
                        str(row["unique_exit_points"]),
                        fmt(row["exit_30ms_valid_pct"]),
                        fmt(row["exit_30ms_nonnegative_pct"]),
                        fmt(row["signal_exit_bp_median"]),
                        fmt(row["exit_30ms_bp_median"]),
                        fmt(row["exit_30ms_bp_mean"]),
                        fmt(row["exit_30ms_bp_p10"]),
                        fmt(row["exit_30ms_bp_p90"]),
                        fmt(row["exit_30ms_minus_signal_bp_median"]),
                    ]
                )
                + " |"
            )
        return lines

    lines = [
        "# Exit Convergence 30ms Taker-taker Report",
        "",
        "Divergence is futures premium `ret_sell >= threshold`. Exit convergence is the first subsequent `ret_buy >= 0` after each divergence event start.",
        "",
        "`signal_exit_bp` is the observed convergence spread at the signal row: `(spot_bid - fut_ask) / fut_ask * 10000`.",
        "`exit_30ms_bp` is the executable-style spread 30ms later using spot bid and futures ask.",
        "",
        "## Event-level",
        "",
        *table(event_summary),
        "",
        "## Unique Exit Points",
        "",
        *table(unique_summary),
        "",
        "## Plot",
        "",
        f"- {plot_path}",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    return {
        "events": out_events,
        "event_summary": out_event_summary,
        "unique_summary": out_unique_summary,
        "report": out_md,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20260126")
    parser.add_argument("--end-date", default="20260629")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for date in _date_range(args.start_date, args.end_date):
        try:
            day = analyze_day(date, args.force_fetch)
        except Exception as exc:
            if args.skip_errors:
                print(f"{date}: skipped ({type(exc).__name__}: {exc})")
                continue
            raise
        if day.height:
            frames.append(day)
            print(f"{date}: exit convergence events {day.height:,}")
    if not frames:
        raise RuntimeError("no exit convergence events found")
    events = pl.concat(frames, how="diagonal_relaxed")
    event_summary, unique_summary = summarize(events)
    tag = f"{args.start_date}_{args.end_date}"
    paths = write_report(events, event_summary, unique_summary, tag)
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(event_summary)
    print(unique_summary)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
