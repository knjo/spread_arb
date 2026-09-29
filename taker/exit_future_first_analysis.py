"""Analyze futures-first exit execution at convergence points.

For a long-spot/short-future position, exit is buy futures and sell spot.
This script mirrors the entry-side execution assumption:

1. Observe a convergence point (`ret_buy >= 0`).
2. Try to buy futures at the observed futures ask. It is considered fillable if
   50ms later the futures ask is still <= the observed ask.
3. If futures fills, sell spot at the spot bid observed 100ms after the signal.

The futures leg has no price slippage under this assumption; the residual exit
slippage is the spot bid movement over the 100ms window.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import polars as pl

import arbitrage_analysis as arb


from data_paths import STOCKFUTURE_DIR as OUT_DIR
DEFAULT_EVENTS = OUT_DIR / "exit_convergence_30ms_events_20260126_20260629.parquet"


def _join_asof(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str,
    strategy: str = "backward",
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
        )


def _add_future_first_quotes(date: str, events: pl.DataFrame, force_fetch: bool) -> pl.DataFrame:
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
                pl.col(arb.TIME_COL).alias("fut_after_time_50ms"),
                pl.col("fut_ask").alias("fut_ask_50ms"),
                pl.col("fut_bid").alias("fut_bid_50ms"),
            ]
        )
        .sort(["QuoteCode", "fut_after_time_50ms"])
    )
    spot = arb._prepare_spot(date, value_codes, force_fetch, spot_ref).select(
        [
            "ValueCode",
            pl.col(arb.TIME_COL).alias("spot_after_time_100ms"),
            pl.col("spot_bid").alias("spot_bid_100ms"),
            pl.col("spot_ask").alias("spot_ask_100ms"),
        ]
    )
    spot_state = arb._prepare_spot_state(date, value_codes, force_fetch, spot_ref)
    if spot_state is not None:
        spot_state = spot_state.rename(
            {
                "spot_state_time": "spot_state_time_100ms",
                "spot_trial_state": "spot_trial_state_100ms",
                "spot_price_ok": "spot_price_ok_100ms",
            }
        )

    time_dtype = events.schema["converge_time"]
    out = events.with_columns(
        [
            (pl.col("converge_time") + pl.duration(milliseconds=50))
            .cast(time_dtype)
            .alias("target_fut_50ms"),
            (pl.col("converge_time") + pl.duration(milliseconds=100))
            .cast(time_dtype)
            .alias("target_spot_100ms"),
        ]
    )
    out = _join_asof(
        out.sort(["QuoteCode", "target_fut_50ms"]),
        fut_state,
        left_on="target_fut_50ms",
        right_on="fut_after_time_50ms",
        by="QuoteCode",
    )
    out = _join_asof(
        out.sort(["ValueCode", "target_spot_100ms"]),
        spot,
        left_on="target_spot_100ms",
        right_on="spot_after_time_100ms",
        by="ValueCode",
    )
    if spot_state is not None:
        out = _join_asof(
            out.sort(["ValueCode", "target_spot_100ms"]),
            spot_state,
            left_on="target_spot_100ms",
            right_on="spot_state_time_100ms",
            by="ValueCode",
        )

    valid_spot_state = (
        (pl.col("spot_trial_state_100ms").fill_null(1) == 0)
        & pl.col("spot_price_ok_100ms").fill_null(False)
        if "spot_trial_state_100ms" in out.columns
        else pl.lit(True)
    )
    return out.with_columns(
        [
            (
                (pl.col("signal_fut_ask") > 0)
                & (pl.col("fut_ask_50ms") > 0)
                & (pl.col("fut_ask_50ms") <= pl.col("signal_fut_ask"))
            ).alias("fut_exit_success_50ms"),
            (
                valid_spot_state
                & (pl.col("spot_bid_100ms") > 0)
                & (pl.col("signal_fut_ask") > 0)
            ).alias("spot_exit_valid_100ms"),
        ]
    ).with_columns(
        [
            (
                pl.col("fut_exit_success_50ms") & pl.col("spot_exit_valid_100ms")
            ).alias("future_first_exit_valid"),
            (
                (pl.col("spot_bid_100ms") - pl.col("signal_fut_ask"))
                / pl.col("signal_fut_ask")
            ).alias("future_first_exit_ret_100ms"),
            (
                (pl.col("spot_bid_100ms") - pl.col("signal_fut_ask"))
                / pl.col("signal_fut_ask")
                * 10000
            ).alias("future_first_exit_bp_100ms"),
            (
                (pl.col("signal_spot_bid") - pl.col("spot_bid_100ms"))
                / pl.col("signal_fut_ask")
                * 10000
            ).alias("spot_bid_slippage_bp_100ms"),
        ]
    ).with_columns(
        (pl.col("future_first_exit_ret_100ms") >= 0).alias(
            "future_first_exit_nonnegative"
        )
    )


def analyze(events_path: Path, start: str, end: str, force_fetch: bool) -> pl.DataFrame:
    events = (
        pl.read_parquet(events_path)
        .filter((pl.col("Date") >= start) & (pl.col("Date") <= end))
    )
    frames = []
    for date in events["Date"].unique().sort().to_list():
        day = events.filter(pl.col("Date") == date)
        enriched = _add_future_first_quotes(date, day, force_fetch)
        if enriched.height:
            frames.append(enriched)
            print(f"{date}: future-first exit points {enriched.height:,}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def summarize(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    valid = pl.col("future_first_exit_valid")
    agg = [
        pl.len().alias("events"),
        pl.col("converge_time").n_unique().alias("unique_exit_points"),
        (pl.col("fut_exit_success_50ms").mean() * 100).alias("fut_success_50ms_pct"),
        (pl.col("spot_exit_valid_100ms").mean() * 100).alias("spot_valid_100ms_pct"),
        (valid.mean() * 100).alias("future_first_valid_pct"),
        (pl.col("future_first_exit_nonnegative").filter(valid).mean() * 100).alias(
            "future_first_nonnegative_pct"
        ),
        pl.col("future_first_exit_bp_100ms")
        .filter(valid)
        .mean()
        .alias("future_first_exit_bp_mean"),
        pl.col("future_first_exit_bp_100ms")
        .filter(valid)
        .median()
        .alias("future_first_exit_bp_median"),
        pl.col("future_first_exit_bp_100ms")
        .filter(valid)
        .quantile(0.1)
        .alias("future_first_exit_bp_p10"),
        pl.col("future_first_exit_bp_100ms")
        .filter(valid)
        .quantile(0.9)
        .alias("future_first_exit_bp_p90"),
        pl.col("spot_bid_slippage_bp_100ms")
        .filter(valid)
        .mean()
        .alias("spot_bid_slip_bp_mean"),
        pl.col("spot_bid_slippage_bp_100ms")
        .filter(valid)
        .median()
        .alias("spot_bid_slip_bp_median"),
    ]
    event_summary = (
        df.group_by("threshold")
        .agg(agg)
        .with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .sort("threshold")
    )
    unique = df.unique(subset=["threshold", "QuoteCode", "converge_time"], keep="first")
    unique_summary = (
        unique.group_by("threshold")
        .agg(agg)
        .with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .sort("threshold")
    )
    return event_summary, unique_summary


def write_report(df: pl.DataFrame, event_summary: pl.DataFrame, unique_summary: pl.DataFrame, tag: str) -> None:
    out_events = OUT_DIR / f"exit_future_first_100ms_events_{tag}.parquet"
    out_event_summary = OUT_DIR / f"exit_future_first_100ms_event_summary_{tag}.csv"
    out_unique_summary = OUT_DIR / f"exit_future_first_100ms_unique_summary_{tag}.csv"
    out_md = OUT_DIR / f"exit_future_first_100ms_report_{tag}.md"
    df.write_parquet(out_events)
    event_summary.write_csv(out_event_summary)
    unique_summary.write_csv(out_unique_summary)

    cols = [
        "threshold_pct",
        "events",
        "unique_exit_points",
        "fut_success_50ms_pct",
        "future_first_valid_pct",
        "future_first_nonnegative_pct",
        "future_first_exit_bp_median",
        "future_first_exit_bp_mean",
        "spot_bid_slip_bp_median",
        "spot_bid_slip_bp_mean",
    ]

    def fmt(x: object) -> str:
        if x is None:
            return ""
        return f"{float(x):.2f}"

    def table(frame: pl.DataFrame) -> list[str]:
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in frame.select(cols).iter_rows(named=True):
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(row["threshold_pct"]),
                        str(row["events"]),
                        str(row["unique_exit_points"]),
                        fmt(row["fut_success_50ms_pct"]),
                        fmt(row["future_first_valid_pct"]),
                        fmt(row["future_first_nonnegative_pct"]),
                        fmt(row["future_first_exit_bp_median"]),
                        fmt(row["future_first_exit_bp_mean"]),
                        fmt(row["spot_bid_slip_bp_median"]),
                        fmt(row["spot_bid_slip_bp_mean"]),
                    ]
                )
                + " |"
            )
        return lines

    lines = [
        "# Futures-first Exit 100ms Spot Slippage Report",
        "",
        "Assumption: after a convergence signal, buy futures first at signal futures ask if 50ms later futures ask is still <= signal ask. Then sell spot at 100ms spot bid.",
        "",
        "Futures leg has no price slippage in this assumption. Residual slippage is spot bid movement over 100ms.",
        "",
        "## Event-level",
        "",
        *table(event_summary),
        "",
        "## Unique Exit Points",
        "",
        *table(unique_summary),
        "",
        "## Files",
        "",
        f"- events: `{out_events}`",
        f"- event_summary: `{out_event_summary}`",
        f"- unique_summary: `{out_unique_summary}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"report: {out_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--start-date", default="20260126")
    parser.add_argument("--end-date", default="20260629")
    parser.add_argument("--force-fetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = analyze(args.events_path, args.start_date, args.end_date, args.force_fetch)
    if df.height == 0:
        raise SystemExit("No future-first exit rows produced.")
    event_summary, unique_summary = summarize(df)
    tag = f"{args.start_date}_{args.end_date}"
    write_report(df, event_summary, unique_summary, tag)
    print(event_summary)
    print(unique_summary)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
