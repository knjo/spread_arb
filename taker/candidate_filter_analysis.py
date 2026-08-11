"""Candidate-stage tickFeature filters for the stock-futures arbitrage ledger.

This reruns the 10m per-name position control after filtering candidates, rather
than removing trades from the finished ledger. It is meant for quick research on
filters discovered by ``tickfeature_factor_screen.py``.

Example:
  uv run python src/research/futures_spot_spread/taker/candidate_filter_analysis.py \
    --start-date 20260126 --end-date 20260629
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

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
from tickfeature_factor_screen import _daily_feature_frame


FILTER_FEATURES = ["L1_BuyBiggestLots_10", "TickBP"]
DEFAULT_THRESHOLDS = [0.005, 0.0075]


def _top20_flag_name(feature: str) -> str:
    return f"{feature}_top20"


def _filtered_flag_expr() -> pl.Expr:
    expr = pl.lit(False)
    for feature in FILTER_FEATURES:
        expr = expr | pl.col(_top20_flag_name(feature)).fill_null(False)
    return expr


def _add_daily_rank_top20(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    out = df.with_row_index("_rank_row")
    exprs: list[pl.Expr] = []
    for feature in FILTER_FEATURES:
        valid = pl.col(feature).is_not_null()
        n = valid.cast(pl.Int64).sum().over(by)
        rank = pl.col(feature).rank("ordinal").over(by)
        exprs.append(
            pl.when(valid & (n >= 5))
            .then((rank / n) >= 0.8)
            .otherwise(False)
            .alias(_top20_flag_name(feature))
        )
    return out.with_columns(exprs).drop("_rank_row")


def _join_stock_features(events: pl.DataFrame, time_col: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for date in events["Date"].unique().sort().to_list():
        day = events.filter(pl.col("Date") == date)
        feature_state = _daily_feature_frame(date, day["ValueCode"].unique().to_list(), FILTER_FEATURES)
        if feature_state.height == 0:
            frames.append(day.with_columns([pl.lit(None).alias(c) for c in FILTER_FEATURES]))
            continue
        joined = (
            day.with_columns(pl.col(time_col).cast(pl.Datetime("us")).alias("_join_time"))
            .sort(["ValueCode", "_join_time"])
            .join_asof(
                feature_state.select([
                    "ValueCode",
                    "feature_recv_time",
                    *FILTER_FEATURES,
                ]),
                left_on="_join_time",
                right_on="feature_recv_time",
                by="ValueCode",
                strategy="backward",
            )
            .with_columns(
                ((pl.col("_join_time") - pl.col("feature_recv_time")).dt.total_microseconds() / 1000)
                .alias("stock_feature_lag_ms")
            )
            .drop("_join_time")
        )
        frames.append(joined)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _compute_position_control(candidates: pl.DataFrame, max_notional: float) -> pl.DataFrame:
    rows = []
    active: dict[tuple[float, str], list[tuple[datetime, float]]] = {}
    sorted_rows = candidates.sort(["threshold", "ValueCode", "entry_time", "row_id"]).iter_rows(named=True)
    for row in sorted_rows:
        key = (float(row["threshold"]), row["ValueCode"])
        entry_time = row["entry_time"]
        release_time = row["position_release_time"]
        fut_bid1 = float(row["fut_bid1"] or 0.0)
        raw_lots = int(row["fut_bid1_lots"] or 0)
        still_open = [
            (exit_time, notional)
            for exit_time, notional in active.get(key, [])
            if exit_time > entry_time
        ]
        used_notional = sum(notional for _, notional in still_open)
        max_lots = math.floor(max(0.0, max_notional - used_notional) / (fut_bid1 * STANDARD_CONTRACT_SIZE)) if fut_bid1 > 0 else 0
        trade_lots = max(0, min(raw_lots, max_lots))
        trade_notional = trade_lots * fut_bid1 * STANDARD_CONTRACT_SIZE
        accept = trade_lots > 0
        if accept:
            still_open.append((release_time, trade_notional))
        active[key] = still_open
        out = dict(row)
        out.update({
            "position_control_accept": accept,
            "max_fut_lots_by_10m": max_lots,
            "trade_fut_lots_10m": trade_lots,
            "position_10m_notional_twd": trade_notional,
            "open_notional_before_twd": used_notional,
            "open_notional_after_twd": used_notional + trade_notional,
        })
        rows.append(out)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _prepare_base_candidates(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    source_path = Path(args.base_path) if args.base_path else _position_control_detail_path(args.start_date, args.end_date)
    source = (
        pl.read_parquet(source_path)
        .filter(pl.col("threshold").is_in(args.thresholds))
        .with_columns(pl.col("row_id").cast(pl.Int64))
    )
    enriched = _join_stock_features(source, "entry_time")
    enriched = enriched.filter(
        pl.col("stock_feature_lag_ms").is_not_null()
        & (pl.col("stock_feature_lag_ms") <= args.max_feature_lag_ms)
    )
    ranked = _add_daily_rank_top20(enriched, ["Date", "threshold"]).with_columns(
        _filtered_flag_expr().alias("candidate_filtered")
    )
    filtered = ranked.filter(~pl.col("candidate_filtered"))
    controlled = _compute_position_control(filtered, args.max_notional)
    return ranked, controlled


def _candidate_threshold_flags(cand: pl.DataFrame, thresholds: list[float]) -> pl.DataFrame:
    if cand.height == 0:
        return cand
    threshold_df = pl.DataFrame({"threshold": thresholds})
    expanded = cand.join(threshold_df, how="cross").filter(pl.col("ret_sell") >= pl.col("threshold"))
    if expanded.height == 0:
        return expanded
    expanded = _join_stock_features(expanded, "trade_time")
    expanded = expanded.filter(
        pl.col("stock_feature_lag_ms").is_not_null()
        & (pl.col("stock_feature_lag_ms") <= 60_000)
    )
    expanded = _add_daily_rank_top20(expanded, ["Date", "threshold"]).with_columns(
        _filtered_flag_expr().alias("candidate_filtered")
    )
    return expanded


def _build_volume_additions(base: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    accepted = (
        base.filter(pl.col("position_control_accept"))
        .with_row_index("base_event_id")
        .sort(["threshold", "ValueCode", "entry_time"])
    )
    if accepted.height == 0:
        return pl.DataFrame()
    refs = accepted.select(["Date", "ValueCode", "QuoteCode", "spot_ref_price", "fut_ref_price"]).unique()
    events = list(accepted.iter_rows(named=True))
    dates = _date_range(args.start_date, args.end_date)
    candidates_by_date: dict[str, pl.DataFrame] = {}
    for date in dates:
        day0 = datetime.strptime(date, "%Y%m%d")
        day1 = day0 + timedelta(days=1)
        active_codes = sorted({
            row["QuoteCode"]
            for row in events
            if row["entry_time"] < day1 and row["position_release_time"] >= day0
        })
        if not active_codes:
            continue
        cand = _load_volume_trade_candidates(
            date,
            active_codes,
            refs,
            args.max_feature_stale_ms,
        )
        if cand.height == 0:
            continue
        candidates_by_date[date] = _candidate_threshold_flags(cand, args.thresholds)

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


def _base_entries(base: pl.DataFrame, fee_bp: float) -> pl.DataFrame:
    return (
        base.filter(
            pl.col("position_control_accept")
            & pl.col("fut_entry_success_50ms")
            & (pl.col("trade_fut_lots_10m") > 0)
            & pl.col("spot_ask_100ms").is_not_null()
        )
        .select([
            pl.lit("base").alias("source"),
            pl.col("row_id").cast(pl.Int64).alias("source_event_id"),
            "Date", "ValueCode", "QuoteCode", "threshold", "status_full",
            "entry_time",
            pl.col("position_release_time").alias("exit_time"),
            pl.col("trade_fut_lots_10m").cast(pl.Int64).alias("fut_lots"),
            pl.col("fut_bid1").alias("entry_fut_sell_price"),
            pl.col("spot_ask_100ms").alias("entry_spot_buy_price"),
            pl.col("spot_ask").alias("signal_spot_ask"),
            "fut_bid1_50ms", "target_fut_50ms", "target_spot_100ms",
        ])
        .with_columns(pl.lit(float(fee_bp)).alias("fee_bp"))
    )


def _addition_entries(additions: pl.DataFrame, fee_bp: float) -> pl.DataFrame:
    if additions.height == 0:
        return pl.DataFrame()
    return (
        additions.filter(
            pl.col("fut_entry_success_50ms")
            & (pl.col("trade_fut_lots") > 0)
            & pl.col("spot_ask_100ms").is_not_null()
        )
        .select([
            pl.lit("volume_add").alias("source"),
            pl.col("base_event_id").cast(pl.Int64).alias("source_event_id"),
            "Date", "ValueCode", "QuoteCode", "threshold", "status_full",
            "entry_time",
            pl.col("position_release_time").alias("exit_time"),
            pl.col("trade_fut_lots").cast(pl.Int64).alias("fut_lots"),
            pl.col("fut_bid1").alias("entry_fut_sell_price"),
            pl.col("spot_ask_100ms").alias("entry_spot_buy_price"),
            pl.col("spot_ask").alias("signal_spot_ask"),
            "fut_bid1_50ms", "target_fut_50ms", "target_spot_100ms",
        ])
        .with_columns(pl.lit(float(fee_bp)).alias("fee_bp"))
    )


def _summary(entries: pl.DataFrame) -> pl.DataFrame:
    return (
        entries.with_columns([
            ((pl.col("entry_spot_buy_price") - pl.col("signal_spot_ask")) / pl.col("signal_spot_ask") * 10000)
            .alias("spot_slippage_bp"),
            (pl.col("status_full") == "當日收斂").cast(pl.Int8).alias("same_day_y"),
            (pl.col("status_full") == "跨日收斂").cast(pl.Int8).alias("cross_day_y"),
            pl.col("status_full").str.contains("沒收斂").cast(pl.Int8).alias("settlement_y"),
        ])
        .group_by("threshold")
        .agg([
            pl.len().alias("samples"),
            pl.col("entry_notional_twd").sum().alias("entry_notional_twd"),
            (pl.col("spot_slippage_bp") * pl.col("entry_notional_twd")).sum()
            .truediv(pl.col("entry_notional_twd").sum())
            .alias("spot_slippage_bp_wavg"),
            (pl.col("entry_spread_bp_bid_basis") * pl.col("entry_notional_twd")).sum()
            .truediv(pl.col("entry_notional_twd").sum())
            .alias("entry_spread_bp_wavg"),
            (pl.col("same_day_y").mean() * 100).alias("same_day_pct"),
            (pl.col("cross_day_y").mean() * 100).alias("cross_day_pct"),
            (pl.col("settlement_y").mean() * 100).alias("settlement_pct"),
            pl.col("net_pnl_twd").sum().alias("net_pnl_twd"),
        ])
        .with_columns([
            (pl.col("threshold") * 100).alias("threshold_pct"),
            (pl.col("entry_notional_twd") / 100_000_000).alias("entry_notional_yi"),
            (pl.col("net_pnl_twd") / 10_000).alias("net_pnl_wan"),
            (pl.col("net_pnl_twd") / pl.col("entry_notional_twd") * 10000).alias("net_pnl_bp"),
        ])
        .sort("threshold")
    )


def _write_markdown(path: Path, summary: pl.DataFrame, args: argparse.Namespace) -> None:
    show = summary.select([
        "threshold_pct", "samples", "entry_notional_yi", "spot_slippage_bp_wavg",
        "entry_spread_bp_wavg", "same_day_pct", "cross_day_pct",
        "settlement_pct", "net_pnl_wan", "net_pnl_bp",
    ]).to_pandas().round(2)
    lines = [
        "# Candidate-stage Slippage Top20 Filter",
        "",
        "Filter: remove daily rank top20 of `L1_BuyBiggestLots_10` or `TickBP` before position control.",
        f"Period: {args.start_date} to {args.end_date}",
        "",
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_list()) + " |")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = f"{_period_tag(args.start_date, args.end_date)}_thr0p50_0p75_remove_L1_TickBP_top20_candidate"
    ranked_base, controlled_base = _prepare_base_candidates(args)
    additions = _build_volume_additions(controlled_base, args)
    entries = pl.concat([
        _base_entries(controlled_base, args.fee_bp),
        _addition_entries(additions, args.fee_bp),
    ], how="diagonal_relaxed")
    entries = _add_backtest_pnl_cols(entries)
    daily = _build_backtest_daily(entries, args.start_date, args.end_date)
    summary = _summary(entries)

    outputs = {
        "ranked_base": OUT_DIR / f"candidate_filter_base_ranked_{tag}.parquet",
        "controlled_base": OUT_DIR / f"candidate_filter_base_position_control_{tag}.parquet",
        "additions": OUT_DIR / f"candidate_filter_volume_additions_{tag}.parquet",
        "trades": OUT_DIR / f"candidate_filter_backtest_trades_{tag}.parquet",
        "daily": OUT_DIR / f"candidate_filter_backtest_daily_{tag}.csv",
        "summary": OUT_DIR / f"candidate_filter_backtest_summary_{tag}.csv",
        "report": OUT_DIR / f"candidate_filter_backtest_summary_{tag}.md",
    }
    ranked_base.write_parquet(outputs["ranked_base"])
    controlled_base.write_parquet(outputs["controlled_base"])
    additions.write_parquet(outputs["additions"])
    entries.write_parquet(outputs["trades"])
    daily.write_csv(outputs["daily"])
    summary.write_csv(outputs["summary"])
    _write_markdown(outputs["report"], summary, args)

    for label, path in outputs.items():
        print(f"{label}: {path}")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=220):
        print(summary.select([
            "threshold_pct", "samples", "entry_notional_yi", "spot_slippage_bp_wavg",
            "entry_spread_bp_wavg", "same_day_pct", "settlement_pct",
            "net_pnl_wan", "net_pnl_bp",
        ]))


if __name__ == "__main__":
    main()
