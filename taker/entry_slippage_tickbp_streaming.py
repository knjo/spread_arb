"""Rebuild the executable entry ledger with bounded daily storage.

The full quote stream is used one trading day at a time. Each day is reduced to
re-entry events, convergence transitions, and futures-fill candidates before
the large futures/spot/feature files are removed.
"""

from __future__ import annotations

import argparse
import gc
import math
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

import arbitrage_analysis as arb


START = "20260126"
END = "20260629"
THRESHOLDS = [0.005, 0.0075]
MAX_NOTIONAL = 10_000_000.0
VOLUME_STEP_LOTS = 10
ADD_LOTS = 1
MAX_FEATURE_STALE_MS = 100.0
HISTORICAL_LEDGER_SAMPLES = {
    0.005: 29_718,
    0.0075: 14_134,
}

WORK_DIR = arb.OUT_DIR / ".tickbp_compact_work"
OUT_DETAIL = (
    arb.OUT_DIR
    / f"entry_slippage_a1a2_1tick_tickbp_detail_{START}_{END}_thr0p50_0p75.parquet"
)
OUT_SUMMARY = (
    arb.OUT_DIR
    / f"entry_slippage_a1a2_1tick_tickbp_summary_{START}_{END}_thr0p50_0p75.csv"
)
OUT_REPORT = (
    arb.OUT_DIR
    / f"entry_slippage_a1a2_1tick_tickbp_report_{START}_{END}_thr0p50_0p75.md"
)
_REF_CACHE: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}


def _trading_dates(start: str, end: str) -> list[str]:
    return sorted(
        path.name[:8]
        for path in arb.MARKET_DIR.glob("*_marketData.parquet")
        if path.name[:8].isdigit() and start <= path.name[:8] <= end
    )


def _compact_paths(date: str) -> tuple[Path, Path, Path]:
    return (
        WORK_DIR / f"{date}_events.parquet",
        WORK_DIR / f"{date}_convergence.parquet",
        WORK_DIR / f"{date}_fills.parquet",
    )


def _remove_daily_cache(date: str) -> None:
    for path in (
        arb._feature_path(date),
        arb._raw_future_path(date),
        arb._raw_spot_path(date),
    ):
        path.unlink(missing_ok=True)


def _spot_ask2(date: str) -> pl.DataFrame:
    path = arb._raw_spot_path(date)
    return (
        pl.scan_parquet(path)
        .select("ValueCode", "RecvTime", "AskPrice2")
        .with_columns(
            pl.col("RecvTime")
            .dt.convert_time_zone("Asia/Taipei")
            .dt.replace_time_zone(None)
            .alias("spot_time"),
            (pl.col("AskPrice2") / arb.SPOT_SCALE).alias("signal_spot_ask2"),
        )
        .select("ValueCode", "spot_time", "signal_spot_ask2")
        .unique(["ValueCode", "spot_time"], keep="last")
        .collect()
    )


def _convergence_transitions(features: pl.DataFrame) -> pl.DataFrame:
    return (
        features.sort(["QuoteCode", arb.TIME_COL])
        .with_columns(
            pl.col("ret_buy").shift(1).over("QuoteCode").alias("_prev_ret_buy")
        )
        .filter(
            (pl.col("ret_buy") >= 0)
            & (
                pl.col("_prev_ret_buy").is_null()
                | (pl.col("_prev_ret_buy") < 0)
            )
        )
        .select(
            "QuoteCode",
            pl.col("Date").alias("converge_date"),
            pl.col(arb.TIME_COL).alias("converge_time"),
        )
        .sort(["QuoteCode", "converge_time"])
    )


def _select_reentries(
    candidates: pl.DataFrame,
    convergence: pl.DataFrame,
) -> pl.DataFrame:
    candidates = candidates.with_row_index("_candidate_row")
    convergence_by_quote: dict[str, np.ndarray] = {}
    for key, group in convergence.group_by("QuoteCode"):
        quote_code = key[0] if isinstance(key, tuple) else key
        convergence_by_quote[quote_code] = np.array(
            group["converge_time"].to_list(),
            dtype="datetime64[ns]",
        )

    selected: list[tuple[int, float, int]] = []
    for threshold in THRESHOLDS:
        threshold_candidates = candidates.filter(
            pl.col("ret_sell") >= threshold
        ).sort(["ValueCode", arb.TIME_COL, "QuoteCode"])
        for _, group in threshold_candidates.group_by(
            "ValueCode",
            maintain_order=True,
        ):
            times = np.array(
                group[arb.TIME_COL].to_list(),
                dtype="datetime64[ns]",
            )
            quote_codes = group["QuoteCode"].to_list()
            row_ids = group["_candidate_row"].to_list()
            previous_convergence = np.datetime64(
                "1900-01-01T00:00:00.000000000"
            )
            cycle = 0
            while True:
                entry_index = int(
                    np.searchsorted(times, previous_convergence, side="right")
                )
                if entry_index >= len(times):
                    break
                cycle += 1
                entry_time = times[entry_index]
                quote_code = quote_codes[entry_index]
                convergence_times = convergence_by_quote.get(quote_code)
                has_same_day_convergence = False
                if convergence_times is not None and len(convergence_times):
                    convergence_index = int(
                        np.searchsorted(
                            convergence_times,
                            entry_time,
                            side="left",
                        )
                    )
                    if convergence_index < len(convergence_times):
                        has_same_day_convergence = True
                        previous_convergence = convergence_times[convergence_index]
                selected.append((int(row_ids[entry_index]), threshold, cycle))
                if not has_same_day_convergence:
                    break

    if not selected:
        return pl.DataFrame()
    selected_rows = pl.DataFrame(
        selected,
        schema=["_candidate_row", "threshold", "cycle_id"],
        orient="row",
    )
    return (
        candidates.join(selected_rows, on="_candidate_row", how="inner")
        .drop("_candidate_row")
        .rename({arb.TIME_COL: "entry_time"})
    )


def _futures_fill_candidates(
    date: str,
    features: pl.DataFrame,
) -> pl.DataFrame:
    feature_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        arb.TIME_COL,
        "contract_size",
        "spot_time",
        "spot_ask",
        "spot_bid",
        "spot_ask_lots",
        "ret_sell",
        "ret_buy",
        "fut_bid1",
        "fut_ask1",
        "fut_bid1_lots",
        "target_fut_50ms",
        "fut_after_time",
        "fut_bid1_50ms",
        "fut_ask1_50ms",
        "target_spot_100ms",
        "spot_after_time",
        "spot_ask_100ms",
        "spot_bid_100ms",
        "fut_entry_success_50ms",
    ]
    feature_state = (
        features.select(feature_columns)
        .rename({arb.TIME_COL: "feature_time"})
        .sort(["QuoteCode", "feature_time"])
    )
    raw_schema = pl.read_parquet_schema(arb._raw_future_path(date))
    raw_columns = [
        column
        for column in [
            "RecvTime",
            "TransTime",
            "QuoteCode",
            "FillLots",
            "TrialMatch",
            "contract_size",
        ]
        if column in raw_schema
    ]
    fills = arb._normalize_time(
        pl.read_parquet(arb._raw_future_path(date), columns=raw_columns)
    ).filter(
        (pl.col("FillLots") > 0)
        & arb._session_time_expr(arb.TIME_COL)
    )
    if "TransTime" in fills.columns:
        fills = fills.filter(arb._session_time_expr("TransTime"))
    if "TrialMatch" in fills.columns:
        fills = fills.filter(pl.col("TrialMatch") == 0)
    if "contract_size" in fills.columns:
        fills = fills.filter(arb._is_standard_contract_expr())
    fills = fills.select(
        "QuoteCode",
        pl.col(arb.TIME_COL).alias("trade_time"),
        pl.col("FillLots").cast(pl.Int64),
    ).sort(["QuoteCode", "trade_time"])
    if fills.is_empty():
        return pl.DataFrame()
    return (
        arb._join_asof(
            fills,
            feature_state,
            left_on="trade_time",
            right_on="feature_time",
            by="QuoteCode",
            strategy="backward",
        )
        .filter(pl.col("feature_time").is_not_null())
        .with_columns(
            (
                (pl.col("trade_time") - pl.col("feature_time"))
                .dt.total_microseconds()
                / 1000
            ).alias("feature_gap_ms")
        )
        .filter(
            (pl.col("feature_gap_ms") <= MAX_FEATURE_STALE_MS)
            & (pl.col("ret_sell") >= min(THRESHOLDS))
        )
        .sort(["QuoteCode", "trade_time"])
    )


def process_day(date: str) -> None:
    event_path, convergence_path, fill_path = _compact_paths(date)
    if event_path.exists() and convergence_path.exists() and fill_path.exists():
        print(f"{date}: compact data already exists")
        return

    build_args = SimpleNamespace(
        code=None,
        fut_code=None,
        near_month_only=True,
        force_fetch=False,
    )
    try:
        arb.build_features_for_day(date, build_args)
        features = pl.read_parquet(arb._feature_path(date))
        ask2 = _spot_ask2(date)
        convergence = _convergence_transitions(features)
        candidates = features.filter(
            (pl.col("ret_sell") >= min(THRESHOLDS))
            & (pl.col("fut_bid1") > 0)
        )
        events = _select_reentries(candidates, convergence)
        fills = _futures_fill_candidates(date, features)
        if not events.is_empty():
            events = events.join(
                ask2,
                on=["ValueCode", "spot_time"],
                how="left",
            )
        if not fills.is_empty():
            fills = fills.join(
                ask2,
                on=["ValueCode", "spot_time"],
                how="left",
            )
        events.write_parquet(event_path)
        convergence.write_parquet(convergence_path)
        fills.write_parquet(fill_path)
        print(
            f"{date}: compact events={events.height:,}, "
            f"convergence={convergence.height:,}, fills={fills.height:,}"
        )
    finally:
        _remove_daily_cache(date)
        gc.collect()


def _classify_convergence(events: pl.DataFrame) -> pl.DataFrame:
    convergence_parts = [
        pl.read_parquet(path)
        for path in sorted(WORK_DIR.glob("*_convergence.parquet"))
    ]
    convergence = pl.concat(
        convergence_parts,
        how="diagonal_relaxed",
    ).sort(["QuoteCode", "converge_time"])

    settlement_rows = []
    for date in events["Date"].unique().sort().to_list():
        settlement = arb.settlement_date(int(date)).strftime("%Y%m%d")
        settlement_rows.append((date, settlement))
    events = (
        events.with_row_index("row_id")
        .join(
            pl.DataFrame(
                settlement_rows,
                schema=["Date", "settlement_date"],
                orient="row",
            ),
            on="Date",
            how="left",
        )
        .sort(["QuoteCode", "entry_time"])
    )
    joined = arb._join_asof(
        events,
        convergence,
        left_on="entry_time",
        right_on="converge_time",
        by="QuoteCode",
        strategy="forward",
    )
    joined = joined.with_columns(
        pl.when(pl.col("ret_buy") >= 0)
        .then(pl.col("Date"))
        .when(pl.col("converge_date") <= pl.col("settlement_date"))
        .then(pl.col("converge_date"))
        .otherwise(None)
        .alias("converge_date_full"),
        pl.when(pl.col("ret_buy") >= 0)
        .then(pl.col("entry_time"))
        .when(pl.col("converge_date") <= pl.col("settlement_date"))
        .then(pl.col("converge_time"))
        .otherwise(None)
        .alias("converge_time_full"),
    )
    settlement_close = (
        (pl.col("settlement_date") + " 13:24:59")
        .str.strptime(pl.Datetime("ns"), "%Y%m%d %H:%M:%S")
    )
    return joined.with_columns(
        pl.coalesce("converge_time_full", settlement_close).alias(
            "position_release_time"
        ),
        pl.when(pl.col("converge_date_full") == pl.col("Date"))
        .then(pl.lit("same_day"))
        .when(
            pl.col("converge_date_full").is_not_null()
            & (pl.col("converge_date_full") < pl.col("settlement_date"))
        )
        .then(pl.lit("cross_day"))
        .otherwise(pl.lit("settlement"))
        .alias("status_full"),
    )


def _strict_ref_filter(date: str, frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if date not in _REF_CACHE:
        market = (
            pl.read_parquet(
                arb.MARKET_DIR / f"{date}_marketData.parquet",
                columns=["quote_code", "opening_ref_price"],
            )
            .select(
                pl.col("quote_code").alias("ValueCode"),
                pl.col("opening_ref_price").cast(pl.Float64).alias("spot_ref_price"),
            )
            .filter(pl.col("spot_ref_price") > 0)
            .unique("ValueCode")
        )
        futures = (
            arb._load_futures_basic(date)
            .select(
                pl.col("quote_code").alias("QuoteCode"),
                pl.col("ref_price").cast(pl.Float64).alias("fut_ref_price"),
            )
            .filter(pl.col("fut_ref_price") > 0)
            .unique("QuoteCode")
        )
        _REF_CACHE[date] = (market, futures)
    market, futures = _REF_CACHE[date]
    return (
        frame.drop(
            [
                column
                for column in ("spot_ref_price", "fut_ref_price")
                if column in frame.columns
            ]
        )
        .join(market, on="ValueCode", how="inner")
        .join(futures, on="QuoteCode", how="inner")
        .filter(
            arb._ref_limit_cols_expr(
                "spot_ref_price",
                ["spot_bid", "spot_ask", "spot_bid_100ms", "spot_ask_100ms"],
            )
            & arb._ref_limit_cols_expr(
                "fut_ref_price",
                ["fut_bid1", "fut_ask1", "fut_bid1_50ms", "fut_ask1_50ms"],
            )
        )
    )


def _position_control(events: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for _, group in events.sort(
        ["threshold", "ValueCode", "entry_time", "QuoteCode"]
    ).group_by(["threshold", "ValueCode"], maintain_order=True):
        open_until = datetime(1900, 1, 1)
        for row in group.iter_rows(named=True):
            entry_time = row["entry_time"]
            fut_bid1 = float(row["fut_bid1"] or 0)
            raw_lots = int(row["fut_bid1_lots"] or 0)
            max_lots = (
                math.floor(MAX_NOTIONAL / (fut_bid1 * arb.STANDARD_CONTRACT_SIZE))
                if fut_bid1 > 0
                else 0
            )
            trade_lots = max(0, min(raw_lots, max_lots))
            accepted = entry_time > open_until and trade_lots > 0
            if accepted:
                open_until = row["position_release_time"]
            out = dict(row)
            out.update(
                {
                    "position_control_accept": accepted,
                    "trade_fut_lots_10m": trade_lots if accepted else 0,
                    "position_10m_notional_twd": (
                        trade_lots
                        * fut_bid1
                        * arb.STANDARD_CONTRACT_SIZE
                        if accepted
                        else 0.0
                    ),
                }
            )
            rows.append(out)
    return pl.DataFrame(rows)


def _volume_additions(base: pl.DataFrame) -> pl.DataFrame:
    additions: list[dict] = []
    accepted = base.filter(pl.col("position_control_accept"))
    allowed_keys = accepted.select(
        "Date",
        "ValueCode",
        "QuoteCode",
    ).unique()
    fills_by_date = {}
    for path in sorted(WORK_DIR.glob("*_fills.parquet")):
        date = path.name[:8]
        date_keys = allowed_keys.filter(pl.col("Date") == date)
        if date_keys.is_empty():
            continue
        fills_by_date[date] = _strict_ref_filter(
            date,
            pl.read_parquet(path),
        ).join(
            date_keys,
            on=["Date", "ValueCode", "QuoteCode"],
            how="inner",
        )
    for row in accepted.iter_rows(named=True):
        entry_time = row["entry_time"]
        release_time = row["position_release_time"]
        threshold = float(row["threshold"])
        used_notional = float(row["position_10m_notional_twd"])
        cumulative_fill_lots = 0
        next_trigger_lots = VOLUME_STEP_LOTS
        date = entry_time.date()
        while date <= release_time.date():
            date_text = date.strftime("%Y%m%d")
            fills = fills_by_date.get(date_text)
            if fills is not None and not fills.is_empty():
                window = fills.filter(
                    (pl.col("QuoteCode") == row["QuoteCode"])
                    & (pl.col("ValueCode") == row["ValueCode"])
                    & (pl.col("trade_time") > entry_time)
                    & (pl.col("trade_time") < release_time)
                    & (pl.col("ret_sell") >= threshold)
                )
                for fill in window.iter_rows(named=True):
                    cumulative_fill_lots += int(fill["FillLots"] or 0)
                    while cumulative_fill_lots >= next_trigger_lots:
                        trade_notional = (
                            float(fill["fut_bid1"])
                            * arb.STANDARD_CONTRACT_SIZE
                            * ADD_LOTS
                        )
                        if used_notional + trade_notional <= MAX_NOTIONAL:
                            used_notional += trade_notional
                            additions.append(
                                {
                                    "source": "volume_add",
                                    "source_event_id": row["row_id"],
                                    "Date": fill["Date"],
                                    "ValueCode": fill["ValueCode"],
                                    "QuoteCode": fill["QuoteCode"],
                                    "threshold": threshold,
                                    "entry_time": fill["trade_time"],
                                    "feature_gap_ms": fill["feature_gap_ms"],
                                    "fut_lots": ADD_LOTS,
                                    "entry_fut_sell_price": fill["fut_bid1"],
                                    "signal_spot_ask": fill["spot_ask"],
                                    "signal_spot_ask2": fill["signal_spot_ask2"],
                                    "signal_spot_ask_lots": fill["spot_ask_lots"],
                                    "entry_spot_buy_price": fill["spot_ask_100ms"],
                                    "fut_entry_success_50ms": fill[
                                        "fut_entry_success_50ms"
                                    ],
                                }
                            )
                        next_trigger_lots += VOLUME_STEP_LOTS
            date += timedelta(days=1)
    return pl.DataFrame(additions) if additions else pl.DataFrame()


def _executable_ledger(base: pl.DataFrame, additions: pl.DataFrame) -> pl.DataFrame:
    base_ledger = (
        base.filter(
            pl.col("position_control_accept")
            & pl.col("fut_entry_success_50ms")
            & (pl.col("trade_fut_lots_10m") > 0)
            & (pl.col("spot_ask_100ms") > 0)
        )
        .select(
            pl.lit("base").alias("source"),
            pl.col("row_id").alias("source_event_id"),
            "Date",
            "ValueCode",
            "QuoteCode",
            "threshold",
            "entry_time",
            pl.col("trade_fut_lots_10m").alias("fut_lots"),
            pl.col("fut_bid1").alias("entry_fut_sell_price"),
            pl.col("spot_ask").alias("signal_spot_ask"),
            "signal_spot_ask2",
            pl.col("spot_ask_lots").alias("signal_spot_ask_lots"),
            pl.col("spot_ask_100ms").alias("entry_spot_buy_price"),
            "fut_entry_success_50ms",
        )
    )
    executable_additions = (
        additions.filter(
            pl.col("fut_entry_success_50ms")
            & (pl.col("fut_lots") > 0)
            & (pl.col("entry_spot_buy_price") > 0)
        )
        if not additions.is_empty()
        else pl.DataFrame()
    )
    parts = [base_ledger]
    if not executable_additions.is_empty():
        parts.append(executable_additions)
    return pl.concat(parts, how="diagonal_relaxed").with_columns(
        (
            pl.col("entry_spot_buy_price")
            * pl.col("fut_lots")
            * arb.STANDARD_CONTRACT_SIZE
        ).alias("entry_notional_twd"),
        (
            (pl.col("entry_spot_buy_price") - pl.col("signal_spot_ask"))
            / pl.col("signal_spot_ask")
            * 10000
        ).alias("spot_slip_bp"),
    )


def _tick_size(price: pl.Expr) -> pl.Expr:
    return (
        pl.when(price < 10).then(0.01)
        .when(price < 50).then(0.05)
        .when(price < 100).then(0.1)
        .when(price < 500).then(0.5)
        .when(price < 1000).then(1.0)
        .otherwise(5.0)
    )


def _add_tick_groups(ledger: pl.DataFrame) -> pl.DataFrame:
    tick_size = _tick_size(pl.col("signal_spot_ask"))
    return (
        ledger.with_columns(
            tick_size.alias("tick_size"),
            (tick_size / pl.col("signal_spot_ask") * 10000).alias("tick_bp"),
        )
        .with_columns(
            (
                (
                    pl.col("signal_spot_ask2")
                    - pl.col("signal_spot_ask")
                    - pl.col("tick_size")
                ).abs()
                < 1e-9
            ).alias("ask_gap_1tick"),
            (pl.col("spot_slip_bp") > 1e-9).alias("did_slip"),
        )
        .with_columns(
            pl.when(pl.col("tick_bp") <= 20)
            .then(pl.lit("<=20"))
            .when((pl.col("tick_bp") > 20) & (pl.col("tick_bp") <= 30))
            .then(pl.lit("20-30"))
            .when((pl.col("tick_bp") > 30) & (pl.col("tick_bp") <= 50))
            .then(pl.lit("30-50"))
            .otherwise(None)
            .alias("tickbp_bucket")
        )
    )


def _summarize(detail: pl.DataFrame) -> pl.DataFrame:
    return (
        detail.filter(
            pl.col("ask_gap_1tick")
            & pl.col("tickbp_bucket").is_not_null()
        )
        .group_by(["threshold", "tickbp_bucket"])
        .agg(
            pl.len().alias("samples"),
            (pl.col("entry_notional_twd").sum() / 1e8).alias("notional_yi"),
            pl.col("did_slip").mean().mul(100).alias("slip_event_pct"),
            pl.col("spot_slip_bp").mean().alias("slip_bp_mean_all"),
            pl.col("spot_slip_bp").median().alias("slip_bp_median_all"),
            pl.col("spot_slip_bp").quantile(0.9).alias("slip_bp_p90_all"),
            pl.col("spot_slip_bp")
            .filter(pl.col("did_slip"))
            .mean()
            .alias("slip_bp_mean_when_slipped"),
            (
                (pl.col("spot_slip_bp") * pl.col("entry_notional_twd")).sum()
                / pl.col("entry_notional_twd").sum()
            ).alias("slip_bp_notional_wavg"),
            pl.col("tick_bp").mean().alias("tick_bp_mean"),
        )
        .with_columns(
            pl.when(pl.col("tickbp_bucket") == "<=20")
            .then(0)
            .when(pl.col("tickbp_bucket") == "20-30")
            .then(1)
            .otherwise(2)
            .alias("_order"),
            (pl.col("threshold") * 100).alias("threshold_pct"),
        )
        .sort(["threshold", "_order"])
        .drop("_order")
        .select(
            "threshold_pct",
            "tickbp_bucket",
            "samples",
            "notional_yi",
            "slip_event_pct",
            "slip_bp_mean_all",
            "slip_bp_median_all",
            "slip_bp_p90_all",
            "slip_bp_mean_when_slipped",
            "slip_bp_notional_wavg",
            "tick_bp_mean",
        )
    )


def _markdown_table(frame: pl.DataFrame) -> list[str]:
    columns = frame.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.iter_rows(named=True):
        values = [
            f"{row[column]:.2f}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def run(start: str, end: str) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dates = _trading_dates(start, end)
    for index, date in enumerate(dates, start=1):
        process_day(date)
        print(f"progress {index}/{len(dates)}")

    events = pl.concat(
        [
            _strict_ref_filter(path.name[:8], pl.read_parquet(path))
            for path in sorted(WORK_DIR.glob("*_events.parquet"))
        ],
        how="diagonal_relaxed",
    )
    base = _position_control(_classify_convergence(events))
    additions = _volume_additions(base)
    ledger = _add_tick_groups(_executable_ledger(base, additions))
    summary = _summarize(ledger)

    validation = (
        ledger.group_by("threshold")
        .agg(pl.len().alias("samples"))
        .sort("threshold")
        .with_columns(
            pl.col("threshold")
            .replace_strict(HISTORICAL_LEDGER_SAMPLES)
            .alias("historical_samples")
        )
        .with_columns(
            (pl.col("samples") / pl.col("historical_samples") * 100)
            .alias("coverage_pct")
        )
    )
    detail = ledger.filter(
        pl.col("ask_gap_1tick")
        & pl.col("tickbp_bucket").is_not_null()
    )
    detail.write_parquet(OUT_DETAIL)
    summary.write_csv(OUT_SUMMARY)
    lines = [
        "# Entry Slippage By Stock TickBP",
        "",
        "Scope: bounded reconstruction of the futures-first executable ledger; "
        "A1-A2 equals one TWSE tick.",
        "",
        "The historical trade-detail parquet had already been removed. The validation "
        "table therefore reports reconstruction coverage against the preserved "
        "historical backtest counts; this report does not replace historical totals.",
        "",
        "`slip_event_pct` is the percentage with 100ms spot A1 worse than signal A1.",
        "`slip_bp_mean_when_slipped` conditions on positive slippage.",
        "",
        "## Ledger Validation",
        "",
        *_markdown_table(
            validation.with_columns(
                (pl.col("threshold") * 100).alias("threshold_pct")
            ).select(
                "threshold_pct",
                "samples",
                "historical_samples",
                "coverage_pct",
            )
        ),
        "",
        "## TickBP Summary",
        "",
        *_markdown_table(summary),
        "",
        "## Files",
        "",
        f"- `{OUT_DETAIL}`",
        f"- `{OUT_SUMMARY}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(validation)
    print(summary)
    print(f"report -> {OUT_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=START)
    parser.add_argument("--end-date", default=END)
    parser.add_argument("--day", help="process one date only")
    args = parser.parse_args()
    if args.day:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        process_day(args.day)
        return
    run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
