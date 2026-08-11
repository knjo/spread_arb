"""Previous-day cutoff candidate filter for futures/spot arbitrage.

This is the tradable variant of the slippage feature filter:

* Build yesterday's full-market candidate p80 cutoff for selected stock
  tickFeature columns.
* Today, reject a candidate if its feature is greater than yesterday's cutoff.
* Re-run per-name 10m position control and volume re-entry from the filtered
  candidate stream.

The first date has no previous cutoff and is left unfiltered.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

import candidate_filter_analysis as cfa
from arbitrage_analysis import (
    OUT_DIR,
    STANDARD_CONTRACT_SIZE,
    _add_backtest_pnl_cols,
    _build_backtest_daily,
    _date_range,
    _load_volume_trade_candidates,
    _period_tag,
    _position_control_detail_path,
)
from candidate_filter_analysis import (
    DEFAULT_THRESHOLDS,
    FILTER_FEATURES,
    _addition_entries,
    _base_entries,
    _compute_position_control,
    _join_stock_features,
    _summary,
)


def _cutoff_col(feature: str) -> str:
    return f"{feature}_prev_p80"


def _feature_tag(features: list[str]) -> str:
    aliases = {
        "L1_BuyBiggestLots_10": "L1Buy10",
        "TickBP": "TickBP",
        "CTC_TimeSpan_100": "CTC100",
        "MD_ElaspeTime_100": "MDE100",
    }
    return "_".join(aliases.get(feature, feature) for feature in features)


def _build_prevday_cutoffs(enriched: pl.DataFrame) -> pl.DataFrame:
    daily = (
        enriched.group_by(["Date", "threshold"])
        .agg([pl.col(feature).quantile(0.8).alias(_cutoff_col(feature)) for feature in FILTER_FEATURES])
        .sort(["threshold", "Date"])
    )
    dates = sorted(enriched["Date"].unique().to_list())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    return (
        daily.with_columns(pl.col("Date").replace(next_date, default=None).alias("Date"))
        .filter(pl.col("Date").is_not_null())
    )


def _apply_prevday_filter(df: pl.DataFrame, cutoffs: pl.DataFrame) -> pl.DataFrame:
    joined = df.join(cutoffs, on=["Date", "threshold"], how="left")
    filter_expr = pl.lit(False)
    has_cutoff = pl.lit(False)
    for feature in FILTER_FEATURES:
        c = _cutoff_col(feature)
        has_cutoff = has_cutoff | pl.col(c).is_not_null()
        filter_expr = filter_expr | (
            pl.col(c).is_not_null()
            & pl.col(feature).is_not_null()
            & (pl.col(feature) > pl.col(c))
        )
    return joined.with_columns([
        has_cutoff.alias("has_prevday_cutoff"),
        filter_expr.alias("candidate_filtered"),
    ])


def _prepare_base(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source_path = Path(args.base_path) if args.base_path else _position_control_detail_path(args.start_date, args.end_date)
    source = (
        pl.read_parquet(source_path)
        .filter(pl.col("threshold").is_in(args.thresholds))
        .with_columns(pl.col("row_id").cast(pl.Int64))
    )
    enriched = _join_stock_features(source, "entry_time").filter(
        pl.col("stock_feature_lag_ms").is_not_null()
        & (pl.col("stock_feature_lag_ms") <= args.max_feature_lag_ms)
    )
    cutoffs = _build_prevday_cutoffs(enriched)
    filtered = _apply_prevday_filter(enriched, cutoffs)
    controlled = _compute_position_control(
        filtered.filter(~pl.col("candidate_filtered")),
        args.max_notional,
    )
    return filtered, controlled, cutoffs


def _volume_candidates_with_prev_filter(
    cand: pl.DataFrame,
    thresholds: list[float],
    cutoffs: pl.DataFrame,
    max_feature_lag_ms: float,
) -> pl.DataFrame:
    if cand.height == 0:
        return cand
    threshold_df = pl.DataFrame({"threshold": thresholds})
    expanded = cand.join(threshold_df, how="cross").filter(pl.col("ret_sell") >= pl.col("threshold"))
    if expanded.height == 0:
        return expanded
    enriched = _join_stock_features(expanded, "trade_time").filter(
        pl.col("stock_feature_lag_ms").is_not_null()
        & (pl.col("stock_feature_lag_ms") <= max_feature_lag_ms)
    )
    return _apply_prevday_filter(enriched, cutoffs)


def _build_volume_additions(base: pl.DataFrame, cutoffs: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    accepted = (
        base.filter(pl.col("position_control_accept"))
        .with_row_index("base_event_id")
        .sort(["threshold", "ValueCode", "entry_time"])
    )
    if accepted.height == 0:
        return pl.DataFrame()

    refs = accepted.select(["Date", "ValueCode", "QuoteCode", "spot_ref_price", "fut_ref_price"]).unique()
    events = list(accepted.iter_rows(named=True))
    candidates_by_date: dict[str, pl.DataFrame] = {}
    for date in _date_range(args.start_date, args.end_date):
        day0 = datetime.strptime(date, "%Y%m%d")
        day1 = day0 + timedelta(days=1)
        active_codes = sorted({
            row["QuoteCode"]
            for row in events
            if row["entry_time"] < day1 and row["position_release_time"] >= day0
        })
        if not active_codes:
            continue
        cand = _load_volume_trade_candidates(date, active_codes, refs, args.max_feature_stale_ms)
        if cand.height:
            candidates_by_date[date] = _volume_candidates_with_prev_filter(
                cand, args.thresholds, cutoffs, args.max_feature_lag_ms
            )

    additions: list[dict] = []
    for row in events:
        entry_time = row["entry_time"]
        release_time = row["position_release_time"]
        threshold = float(row["threshold"])
        quote_code = row["QuoteCode"]
        value_code = row["ValueCode"]
        used_notional = float(row["position_10m_notional_twd"] or 0.0)
        next_trigger_lots = int(args.volume_lots)
        cumulative_fill_lots = 0
        for date in _date_range(entry_time.strftime("%Y%m%d"), release_time.strftime("%Y%m%d")):
            cand = candidates_by_date.get(date)
            if cand is None or cand.height == 0:
                continue
            window = cand.filter(
                (pl.col("QuoteCode") == quote_code)
                & (pl.col("ValueCode") == value_code)
                & (pl.col("threshold") == threshold)
                & (~pl.col("candidate_filtered"))
                & (pl.col("trade_time") > entry_time)
                & (pl.col("trade_time") < release_time)
            )
            if window.height == 0:
                continue
            for c in window.sort("trade_time").iter_rows(named=True):
                cumulative_fill_lots += int(c["FillLots"] or 0)
                while cumulative_fill_lots >= next_trigger_lots:
                    trade_notional = float(c["fut_bid1"]) * STANDARD_CONTRACT_SIZE * int(args.add_lots)
                    if used_notional + trade_notional <= float(args.max_notional):
                        used_notional += trade_notional
                        spot_tick = (
                            0.01 if c["spot_ask"] < 10 else
                            0.05 if c["spot_ask"] < 50 else
                            0.1 if c["spot_ask"] < 100 else
                            0.5 if c["spot_ask"] < 500 else
                            1.0 if c["spot_ask"] < 1000 else
                            5.0
                        )
                        spot_slip = float(c["spot_ask_100ms"]) - float(c["spot_ask"])
                        additions.append({
                            "base_event_id": row["base_event_id"],
                            "Date": c["Date"],
                            "base_Date": row["Date"],
                            "ValueCode": value_code,
                            "QuoteCode": quote_code,
                            "threshold": threshold,
                            "add_seq": next_trigger_lots // int(args.volume_lots),
                            "entry_time": c["trade_time"],
                            "feature_time": c["feature_time"],
                            "feature_gap_ms": c["feature_gap_ms"],
                            "stock_feature_lag_ms": c["stock_feature_lag_ms"],
                            "FillLots": c["FillLots"],
                            "cumulative_fill_lots": cumulative_fill_lots,
                            "trigger_lots": next_trigger_lots,
                            "trade_fut_lots": int(args.add_lots),
                            "trade_notional_twd": trade_notional,
                            "open_notional_after_twd": used_notional,
                            "fut_bid1": c["fut_bid1"],
                            "fut_ask1": c["fut_ask1"],
                            "spot_ask": c["spot_ask"],
                            "spot_bid": c["spot_bid"],
                            "ret_sell": c["ret_sell"],
                            "target_fut_50ms": c["target_fut_50ms"],
                            "fut_after_time": c["fut_after_time"],
                            "fut_bid1_50ms": c["fut_bid1_50ms"],
                            "fut_ask1_50ms": c["fut_ask1_50ms"],
                            "target_spot_100ms": c["target_spot_100ms"],
                            "spot_after_time": c["spot_after_time"],
                            "spot_ask_100ms": c["spot_ask_100ms"],
                            "spot_bid_100ms": c["spot_bid_100ms"],
                            "fut_entry_success_50ms": c["fut_entry_success_50ms"],
                            "spot_slippage_ticks": spot_slip / spot_tick,
                            "spot_slippage_bp": spot_slip / float(c["spot_ask"]) * 10000,
                            "position_release_time": release_time,
                            "status_full": row["status_full"],
                        })
                    next_trigger_lots += int(args.volume_lots)
    return pl.DataFrame(additions) if additions else pl.DataFrame()


def _candidate_layer_summary(base_ranked: pl.DataFrame, base_controlled: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    raw = (
        base_ranked.group_by("threshold")
        .agg([
            pl.len().alias("raw_candidates"),
            pl.col("candidate_filtered").sum().alias("raw_filtered"),
            (pl.col("candidate_filtered").mean() * 100).alias("raw_filtered_pct"),
            pl.col("has_prevday_cutoff").mean().mul(100).alias("has_cutoff_pct"),
        ])
    )
    accepted_all = (
        base_controlled.group_by("threshold")
        .agg([
            pl.col("position_control_accept").sum().alias("accepted_all"),
            (pl.col("position_10m_notional_twd").filter(pl.col("position_control_accept")).sum() / 100_000_000)
            .alias("accepted_all_yi"),
        ])
    )
    executable = (
        entries.group_by("threshold")
        .agg([
            pl.len().alias("executable_trades"),
            (pl.col("entry_notional_twd").sum() / 100_000_000).alias("executable_yi"),
        ])
    )
    return (
        raw.join(accepted_all, on="threshold", how="left")
        .join(executable, on="threshold", how="left")
        .with_columns((pl.col("threshold") * 100).alias("threshold_pct"))
        .sort("threshold")
    )


def _write_report(path: Path, summary: pl.DataFrame, layer: pl.DataFrame, args: argparse.Namespace) -> None:
    def table(df: pl.DataFrame) -> list[str]:
        pdf = df.to_pandas().round(2)
        lines = ["| " + " | ".join(pdf.columns) + " |", "| " + " | ".join(["---"] * len(pdf.columns)) + " |"]
        for _, row in pdf.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.to_list()) + " |")
        return lines

    lines = [
        "# Previous-day Candidate Filter",
        "",
        f"Filter: reject today's candidate if any selected feature is greater than yesterday's full-market p80 cutoff for the same threshold. Features: {', '.join(FILTER_FEATURES)}.",
        "Cutoffs are built from base/re-entry candidates and shifted by trading date. First date is unfiltered.",
        f"Period: {args.start_date} to {args.end_date}",
        "",
        "## Backtest",
        "",
        *table(summary.select([
            "threshold_pct", "samples", "entry_notional_yi", "spot_slippage_bp_wavg",
            "entry_spread_bp_wavg", "same_day_pct", "settlement_pct",
            "net_pnl_wan", "net_pnl_bp",
        ])),
        "",
        "## Layers",
        "",
        *table(layer.select([
            "threshold_pct", "raw_candidates", "raw_filtered", "raw_filtered_pct",
            "has_cutoff_pct", "accepted_all", "accepted_all_yi",
            "executable_trades", "executable_yi",
        ])),
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="20260126")
    p.add_argument("--end-date", default="20260629")
    p.add_argument("--base-path")
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--max-notional", type=float, default=10_000_000)
    p.add_argument("--volume-lots", type=int, default=10)
    p.add_argument("--add-lots", type=int, default=1)
    p.add_argument("--max-feature-stale-ms", type=float, default=1000.0)
    p.add_argument("--max-feature-lag-ms", type=float, default=60_000.0)
    p.add_argument("--fee-bp", type=float, default=38.0)
    p.add_argument("--filter-features", nargs="+", default=FILTER_FEATURES)
    return p.parse_args()


def main() -> None:
    global FILTER_FEATURES
    args = parse_args()
    FILTER_FEATURES = list(args.filter_features)
    cfa.FILTER_FEATURES = list(args.filter_features)
    tag = f"{_period_tag(args.start_date, args.end_date)}_thr0p50_0p75_prevday_p80_{_feature_tag(FILTER_FEATURES)}"
    base_ranked, base_controlled, cutoffs = _prepare_base(args)
    additions = _build_volume_additions(base_controlled, cutoffs, args)
    entries = pl.concat([
        _base_entries(base_controlled, args.fee_bp),
        _addition_entries(additions, args.fee_bp),
    ], how="diagonal_relaxed")
    entries = _add_backtest_pnl_cols(entries)
    daily = _build_backtest_daily(entries, args.start_date, args.end_date)
    summary = _summary(entries)
    layer = _candidate_layer_summary(base_ranked, base_controlled, entries)

    outputs = {
        "cutoffs": OUT_DIR / f"prevday_candidate_filter_cutoffs_{tag}.csv",
        "base_ranked": OUT_DIR / f"prevday_candidate_filter_base_ranked_{tag}.parquet",
        "base_controlled": OUT_DIR / f"prevday_candidate_filter_base_position_control_{tag}.parquet",
        "additions": OUT_DIR / f"prevday_candidate_filter_volume_additions_{tag}.parquet",
        "trades": OUT_DIR / f"prevday_candidate_filter_backtest_trades_{tag}.parquet",
        "daily": OUT_DIR / f"prevday_candidate_filter_backtest_daily_{tag}.csv",
        "summary": OUT_DIR / f"prevday_candidate_filter_backtest_summary_{tag}.csv",
        "layers": OUT_DIR / f"prevday_candidate_filter_layers_{tag}.csv",
        "report": OUT_DIR / f"prevday_candidate_filter_report_{tag}.md",
    }
    cutoffs.write_csv(outputs["cutoffs"])
    base_ranked.write_parquet(outputs["base_ranked"])
    base_controlled.write_parquet(outputs["base_controlled"])
    additions.write_parquet(outputs["additions"])
    entries.write_parquet(outputs["trades"])
    daily.write_csv(outputs["daily"])
    summary.write_csv(outputs["summary"])
    layer.write_csv(outputs["layers"])
    _write_report(outputs["report"], summary, layer, args)

    for label, path in outputs.items():
        print(f"{label}: {path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=220):
        print(summary.select([
            "threshold_pct", "samples", "entry_notional_yi", "spot_slippage_bp_wavg",
            "entry_spread_bp_wavg", "same_day_pct", "settlement_pct",
            "net_pnl_wan", "net_pnl_bp",
        ]))
        print(layer)


if __name__ == "__main__":
    main()
