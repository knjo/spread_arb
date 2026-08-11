"""Analyze entry spot-book conditions for futures-first arbitrage trades.

The entry policy is futures-first: sell futures if fillable, then buy spot at
the future spot ask state. This script studies whether the signal spot A1 size
and the A1/A2 ask gap explain the 100ms spot ask slippage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT / "data" / "stockfuture"
DEFAULT_TRADES = (
    OUT_DIR / "backtest_trades_20260126_20260629_ref9_symbol_open_base_plus_volume10_fee38p0bp.parquet"
)


def tick_size_expr(price: pl.Expr) -> pl.Expr:
    """TWSE stock tick size by price level."""
    return (
        pl.when(price < 10).then(0.01)
        .when(price < 50).then(0.05)
        .when(price < 100).then(0.1)
        .when(price < 500).then(0.5)
        .when(price < 1000).then(1.0)
        .otherwise(5.0)
    )


def load_signal_book_for_day(date: str, value_codes: list[str]) -> pl.DataFrame:
    feature_path = OUT_DIR / f"{date}_arbitrage_features.parquet"
    spot_path = OUT_DIR / f"{date}_spot_for_stockfuture.parquet"
    if not feature_path.exists() or not spot_path.exists():
        return pl.DataFrame()

    feature = (
        pl.scan_parquet(feature_path)
        .filter(pl.col("ValueCode").is_in(value_codes))
        .select(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                pl.col("RecvTime").alias("feature_entry_time_ns"),
                "spot_time",
                "spot_ask",
                "spot_ask_lots",
                "spot_ask_100ms",
            ]
        )
    )
    spot = (
        pl.scan_parquet(spot_path)
        .filter(pl.col("ValueCode").is_in(value_codes))
        .select(
            [
                "ValueCode",
                pl.col("RecvTime")
                .dt.convert_time_zone("Asia/Taipei")
                .dt.replace_time_zone(None)
                .alias("spot_time"),
                (pl.col("AskPrice1") / 10000).alias("raw_ask1"),
                (pl.col("AskPrice2") / 10000).alias("raw_ask2"),
                pl.col("AskLots1").alias("raw_ask_lots1"),
                pl.col("AskLots2").alias("raw_ask_lots2"),
            ]
        )
    )
    return (
        feature.join(
            spot,
            on=["ValueCode", "spot_time"],
            how="left",
        )
        .with_columns(
            [
                tick_size_expr(pl.col("spot_ask")).alias("spot_tick_size"),
                (pl.col("raw_ask2") - pl.col("spot_ask")).alias("ask1_ask2_gap"),
            ]
        )
        .with_columns(
            [
                (pl.col("ask1_ask2_gap") / pl.col("spot_tick_size")).alias(
                    "ask1_ask2_gap_ticks"
                ),
                (
                    (pl.col("raw_ask2") > 0)
                    & ((pl.col("ask1_ask2_gap") - pl.col("spot_tick_size")).abs() < 1e-9)
                ).alias("ask_gap_1tick"),
            ]
        )
        .collect()
    )


def build_enriched_trades(trades_path: Path, thresholds: list[float], start: str, end: str) -> pl.DataFrame:
    threshold_bp = [int(round(x * 10000)) for x in thresholds]
    trades = (
        pl.read_parquet(trades_path)
        .with_row_index("trade_row_id")
        .with_columns((pl.col("threshold") * 10000).round(0).cast(pl.Int64).alias("_thr_bp"))
        .filter(pl.col("_thr_bp").is_in(threshold_bp))
        .filter((pl.col("Date") >= start) & (pl.col("Date") <= end))
        .with_columns(
            [
                pl.col("entry_time").cast(pl.Datetime("ns")).alias("entry_time_ns"),
                ((pl.col("entry_spot_buy_price") - pl.col("signal_spot_ask"))
                 / pl.col("signal_spot_ask") * 10000).alias("spot_ask_slip_bp_100ms"),
            ]
        )
        .drop("_thr_bp")
    )
    frames: list[pl.DataFrame] = []
    for date in trades["Date"].unique().sort().to_list():
        day_trades = trades.filter(pl.col("Date") == date)
        book = load_signal_book_for_day(date, day_trades["ValueCode"].unique().to_list())
        if book.height == 0:
            continue
        joined = day_trades.sort(["ValueCode", "entry_time_ns"]).join_asof(
            book.sort(["ValueCode", "feature_entry_time_ns"]),
            left_on="entry_time_ns",
            right_on="feature_entry_time_ns",
            by="ValueCode",
            strategy="nearest",
            tolerance="1ms",
        )
        frames.append(joined)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def add_lot_buckets(df: pl.DataFrame, cuts: list[int]) -> pl.DataFrame:
    lot = pl.col("spot_ask_lots")
    expr = pl.lit(f">{cuts[-1]}")
    for cut in reversed(cuts):
        expr = pl.when(lot <= cut).then(pl.lit(f"<={cut}")).otherwise(expr)
    return df.with_columns(expr.alias("spot_ask_lots_bucket"))


def summarize(enriched: pl.DataFrame, cuts: list[int]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    enriched = add_lot_buckets(enriched, cuts).with_columns(
        [
            (pl.col("threshold") * 100).alias("threshold_pct"),
            (pl.col("source") == "base").alias("is_base"),
            (pl.col("source") == "volume_add").alias("is_volume_add"),
        ]
    )
    common_aggs = [
        pl.len().alias("samples"),
        (pl.col("entry_notional_twd").sum() / 1e8).alias("notional_yi"),
        pl.col("spot_ask_slip_bp_100ms").mean().alias("slip_bp_mean"),
        pl.col("spot_ask_slip_bp_100ms").median().alias("slip_bp_median"),
        pl.col("spot_ask_slip_bp_100ms").quantile(0.9).alias("slip_bp_p90"),
        ((pl.col("spot_ask_slip_bp_100ms") * pl.col("entry_notional_twd")).sum()
         / pl.col("entry_notional_twd").sum()).alias("slip_bp_wavg"),
    ]
    by_lots = (
        enriched.group_by(["threshold_pct", "spot_ask_lots_bucket"])
        .agg(common_aggs)
        .with_columns(
            pl.col("spot_ask_lots_bucket")
            .str.replace("<=", "")
            .str.replace(">", "999999")
            .cast(pl.Int64)
            .alias("_bucket_order")
        )
        .sort(["threshold_pct", "_bucket_order"])
        .drop("_bucket_order")
    )
    by_gap = (
        enriched.group_by(["threshold_pct", "ask_gap_1tick"])
        .agg(
            common_aggs
            + [
                pl.col("ask1_ask2_gap_ticks").median().alias("gap_ticks_median"),
                pl.col("ask1_ask2_gap_ticks").quantile(0.9).alias("gap_ticks_p90"),
            ]
        )
        .sort(["threshold_pct", "ask_gap_1tick"])
    )
    combo = (
        enriched.group_by(["threshold_pct", "spot_ask_lots_bucket", "ask_gap_1tick"])
        .agg(common_aggs)
        .with_columns(
            pl.col("spot_ask_lots_bucket")
            .str.replace("<=", "")
            .str.replace(">", "999999")
            .cast(pl.Int64)
            .alias("_bucket_order")
        )
        .sort(["threshold_pct", "_bucket_order", "ask_gap_1tick"])
        .drop("_bucket_order")
    )
    return by_lots, by_gap, combo


def write_report(
    enriched: pl.DataFrame,
    by_lots: pl.DataFrame,
    by_gap: pl.DataFrame,
    combo: pl.DataFrame,
    tag: str,
) -> dict[str, Path]:
    out_enriched = OUT_DIR / f"entry_spot_book_slippage_enriched_{tag}.parquet"
    out_lots = OUT_DIR / f"entry_spot_book_slippage_by_a1_lots_{tag}.csv"
    out_gap = OUT_DIR / f"entry_spot_book_slippage_by_ask_gap_{tag}.csv"
    out_combo = OUT_DIR / f"entry_spot_book_slippage_by_lots_gap_{tag}.csv"
    out_md = OUT_DIR / f"entry_spot_book_slippage_report_{tag}.md"
    plot_path = OUT_DIR / "plots" / f"entry_spot_book_slippage_by_a1_lots_{tag}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    enriched.write_parquet(out_enriched)
    by_lots.write_csv(out_lots)
    by_gap.write_csv(out_gap)
    combo.write_csv(out_combo)

    pdf = by_lots.to_pandas()
    fig, ax = plt.subplots(figsize=(11, 5))
    for threshold, sub in pdf.groupby("threshold_pct"):
        ax.plot(sub["spot_ask_lots_bucket"], sub["slip_bp_wavg"], marker="o", label=f"{threshold:.2f}%")
    ax.set_title("Entry spot ask slippage by signal A1 lots")
    ax.set_xlabel("Signal spot A1 lots bucket")
    ax.set_ylabel("Notional-weighted spot ask slippage (bp)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    def table(df: pl.DataFrame) -> list[str]:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in df.iter_rows(named=True):
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.2f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    sample = (
        enriched.with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .group_by("threshold_pct")
        .agg(
            [
                pl.len().alias("samples"),
                (pl.col("entry_notional_twd").sum() / 1e8).alias("notional_yi"),
                pl.col("spot_ask_lots").is_not_null().mean().mul(100).alias("book_join_pct"),
                pl.col("spot_ask_slip_bp_100ms").median().alias("slip_bp_median"),
                ((pl.col("spot_ask_slip_bp_100ms") * pl.col("entry_notional_twd")).sum()
                 / pl.col("entry_notional_twd").sum()).alias("slip_bp_wavg"),
            ]
        )
        .sort("threshold_pct")
    )
    lines = [
        "# Entry Spot Book Slippage",
        "",
        "Base: futures-first executable trades. Spot leg buys at 100ms ask; slippage is `(entry_spot_buy_price - signal_spot_ask) / signal_spot_ask * 10000`.",
        "",
        "A1/A2 one-tick gap uses TWSE stock tick size by signal A1 price.",
        "",
        "## Sample",
        "",
        *table(sample),
        "",
        "## By Signal A1 Lots",
        "",
        *table(by_lots),
        "",
        "## By Ask A1-A2 Gap",
        "",
        *table(by_gap),
        "",
        "## By A1 Lots And Gap",
        "",
        *table(combo),
        "",
        "## Files",
        "",
        f"- enriched: `{out_enriched}`",
        f"- by_lots: `{out_lots}`",
        f"- by_gap: `{out_gap}`",
        f"- combo: `{out_combo}`",
        f"- plot: `{plot_path}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    return {
        "enriched": out_enriched,
        "by_lots": out_lots,
        "by_gap": out_gap,
        "combo": out_combo,
        "report": out_md,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--start-date", default="20260126")
    parser.add_argument("--end-date", default="20260629")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.005, 0.0075])
    parser.add_argument("--a1-lot-cuts", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched = build_enriched_trades(
        args.trades_path, args.thresholds, args.start_date, args.end_date
    )
    if enriched.height == 0:
        raise SystemExit("No enriched trades produced.")
    by_lots, by_gap, combo = summarize(enriched, args.a1_lot_cuts)
    tag = (
        f"{args.start_date}_{args.end_date}_"
        + "_".join(f"thr{x * 100:.2f}".replace(".", "p") for x in args.thresholds)
    )
    paths = write_report(enriched, by_lots, by_gap, combo, tag)
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(by_lots)
    print(by_gap)


if __name__ == "__main__":
    main()
