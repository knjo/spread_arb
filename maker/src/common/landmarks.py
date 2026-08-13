"""Causal one-second futures/spot basis landmark builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import polars as pl

from .contracts import load_contract_mapping
from .paths import DEFAULT_OUTPUT_ROOT, futures_raw_path, spot_tick_path


REF_LOWER_RETURN = -0.09
REF_UPPER_RETURN = 0.08
REF_COMPARISON_EPS_RATIO = 1e-12
SESSION_OPEN = time(9, 0)
SESSION_END = time(13, 20)
AGE_THRESHOLDS_MS = (100, 250, 500, 1000, 5000)


@dataclass(frozen=True)
class LandmarkBuildResult:
    landmarks: pl.DataFrame
    audit: pl.DataFrame
    mapping: pl.DataFrame


def _session_expr(column: str) -> pl.Expr:
    current = pl.col(column).dt.time()
    return (current >= pl.lit(SESSION_OPEN)) & (current < pl.lit(SESSION_END))


def _utc_naive_ns(column: str, timezone_aware: bool) -> pl.Expr:
    value = pl.col(column)
    if timezone_aware:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return value.cast(pl.Datetime("ns"))


def _raw_price(column: str) -> pl.Expr:
    divisor = pl.lit(10.0).pow(pl.col("DecimalLocator").cast(pl.Float64))
    return (pl.col(column).cast(pl.Float64) / divisor).alias(column)


def _load_spot_events(date: str, mapping: pl.DataFrame) -> pl.DataFrame:
    path = spot_tick_path(date)
    if not path.exists():
        raise FileNotFoundError(path)
    codes = mapping["ValueCode"].to_list()
    references = mapping.select("ValueCode", "spot_ref_price")
    return (
        pl.scan_parquet(path)
        .filter(pl.col("ValueCode").is_in(codes) & _session_expr("TransTime"))
        .select(
            _utc_naive_ns("RecvTime", timezone_aware=False).alias("recv_time"),
            pl.col("TransTime").cast(pl.Datetime("us")).alias("trans_time"),
            pl.col("ValueCode").cast(pl.String),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("sequence"),
            pl.col("TrialMatch").cast(pl.Int16).alias("trial_match"),
            pl.col("BidPrice1").cast(pl.Float64).alias("bid"),
            pl.col("AskPrice1").cast(pl.Float64).alias("ask"),
            pl.col("BidLots1").cast(pl.Int64).alias("bid_lots"),
            pl.col("AskLots1").cast(pl.Int64).alias("ask_lots"),
        )
        .join(references.lazy(), on="ValueCode", how="inner")
        .collect()
        .sort(["ValueCode", "recv_time", "sequence"])
    )


def _load_future_events(date: str, mapping: pl.DataFrame) -> pl.DataFrame:
    path = futures_raw_path(date)
    if not path.exists():
        raise FileNotFoundError(path)
    quote_codes = mapping["QuoteCode"].to_list()
    references = mapping.select(
        "QuoteCode", "ValueCode", "fut_ref_price", "contract_size", "end_date"
    )
    return (
        pl.scan_parquet(path)
        .filter(pl.col("QuoteCode").is_in(quote_codes) & _session_expr("TransTime"))
        .select(
            _utc_naive_ns("RecvTime", timezone_aware=True).alias("recv_time"),
            pl.col("TransTime").cast(pl.Datetime("us")).alias("trans_time"),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("sequence"),
            pl.col("TrialMatch").cast(pl.Int16).alias("trial_match"),
            pl.col("DecimalLocator").cast(pl.Int16),
            _raw_price("BidPrice1"),
            _raw_price("AskPrice1"),
            _raw_price("BestBidPrice"),
            _raw_price("BestAskPrice"),
            pl.col("BidLots1").cast(pl.Int64).alias("bid_lots"),
            pl.col("AskLots1").cast(pl.Int64).alias("ask_lots"),
            pl.col("BestBidLots").cast(pl.Int64).alias("best_bid_lots"),
            pl.col("BestAskLots").cast(pl.Int64).alias("best_ask_lots"),
        )
        .rename(
            {
                "BidPrice1": "bid",
                "AskPrice1": "ask",
                "BestBidPrice": "best_bid",
                "BestAskPrice": "best_ask",
            }
        )
        .with_columns(
            pl.max_horizontal(
                pl.when(pl.col("bid") > 0).then(pl.col("bid")).otherwise(None),
                pl.when(pl.col("best_bid") > 0)
                .then(pl.col("best_bid"))
                .otherwise(None),
            ).alias("exec_bid"),
            pl.min_horizontal(
                pl.when(pl.col("ask") > 0).then(pl.col("ask")).otherwise(None),
                pl.when(pl.col("best_ask") > 0)
                .then(pl.col("best_ask"))
                .otherwise(None),
            ).alias("exec_ask"),
        )
        .with_columns(
            pl.when(pl.col("exec_bid").is_null())
            .then(0)
            .when(
                (pl.col("bid") == pl.col("exec_bid"))
                & (pl.col("best_bid") == pl.col("exec_bid"))
            )
            .then(pl.max_horizontal("bid_lots", "best_bid_lots"))
            .when(pl.col("bid") == pl.col("exec_bid"))
            .then(pl.col("bid_lots"))
            .otherwise(pl.col("best_bid_lots"))
            .alias("exec_bid_lots"),
            pl.when(pl.col("exec_ask").is_null())
            .then(0)
            .when(
                (pl.col("ask") == pl.col("exec_ask"))
                & (pl.col("best_ask") == pl.col("exec_ask"))
            )
            .then(pl.max_horizontal("ask_lots", "best_ask_lots"))
            .when(pl.col("ask") == pl.col("exec_ask"))
            .then(pl.col("ask_lots"))
            .otherwise(pl.col("best_ask_lots"))
            .alias("exec_ask_lots"),
        )
        .join(references.lazy(), on="QuoteCode", how="inner")
        .collect()
        .sort(["ValueCode", "recv_time", "sequence"])
    )


def _last_per_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.sort(["ValueCode", "recv_time", "sequence"])
        .group_by(["ValueCode", "recv_time"], maintain_order=True)
        .agg(pl.exclude("ValueCode", "recv_time").last())
    )


def _book_state(events: pl.DataFrame, prefix: str) -> pl.DataFrame:
    columns = [
        "ValueCode",
        "recv_time",
        "sequence",
        "bid",
        "ask",
        "bid_lots",
        "ask_lots",
    ]
    columns.extend(
        column
        for column in ("exec_bid", "exec_ask", "exec_bid_lots", "exec_ask_lots")
        if column in events.columns
    )
    optional = [
        column
        for column in ("spot_ref_price", "fut_ref_price", "contract_size", "end_date")
        if column in events.columns
    ]
    has_quote = (pl.col("bid") > 0) | (pl.col("ask") > 0)
    state = _last_per_timestamp(events.filter(has_quote).select(columns + optional))
    rename = {
        column: f"{prefix}_{column}"
        for column in (
            "recv_time",
            "sequence",
            "bid",
            "ask",
            "bid_lots",
            "ask_lots",
            "exec_bid",
            "exec_ask",
            "exec_bid_lots",
            "exec_ask_lots",
        )
        if column in state.columns
    }
    return state.rename(rename).sort(["ValueCode", f"{prefix}_recv_time"])


def _trial_transitions(events: pl.DataFrame, prefix: str) -> pl.DataFrame:
    transitions = (
        events.select("ValueCode", "recv_time", "sequence", "trial_match")
        .sort(["ValueCode", "recv_time", "sequence"])
        .with_columns(
            pl.col("trial_match").shift(1).over("ValueCode").alias("previous_trial")
        )
        .filter(
            pl.col("previous_trial").is_null()
            | (pl.col("trial_match") != pl.col("previous_trial"))
        )
        .drop("previous_trial")
    )
    transitions = _last_per_timestamp(transitions)
    return transitions.rename(
        {
            "recv_time": f"{prefix}_trial_time",
            "sequence": f"{prefix}_trial_sequence",
            "trial_match": f"{prefix}_trial_match",
        }
    ).sort(["ValueCode", f"{prefix}_trial_time"])


def _session_grid(date: str, mapping: pl.DataFrame, interval: str) -> pl.DataFrame:
    parsed = datetime.strptime(date, "%Y%m%d")
    local_start = datetime.combine(parsed.date(), SESSION_OPEN)
    local_end = datetime.combine(parsed.date(), SESSION_END)
    utc_start = local_start - timedelta(hours=8)
    utc_end = local_end - timedelta(hours=8)
    timestamps = pl.datetime_range(
        utc_start,
        utc_end,
        interval=interval,
        closed="left",
        time_unit="ns",
        eager=True,
    )
    grid = mapping.select("ValueCode", "QuoteCode").with_columns(
        pl.lit(date).alias("Date")
    ).join(pl.DataFrame({"timestamp": timestamps}), how="cross")
    return grid.with_columns(
        (pl.col("timestamp") + pl.duration(hours=8)).alias("local_timestamp"),
        ((pl.col("timestamp") - pl.lit(utc_start)).dt.total_seconds()).cast(pl.Int32)
        .alias("seconds_from_open"),
    ).sort(["ValueCode", "timestamp"])


def _join_state(
    grid: pl.DataFrame,
    state: pl.DataFrame,
    right_on: str,
) -> pl.DataFrame:
    return grid.join_asof(
        state,
        left_on="timestamp",
        right_on=right_on,
        by="ValueCode",
        strategy="backward",
        check_sortedness=False,
    )


def _ref_ok(
    prefix: str,
    reference: str,
    price_suffixes: tuple[str, ...] = ("bid", "ask"),
) -> pl.Expr:
    epsilon = pl.col(reference).abs() * REF_COMPARISON_EPS_RATIO
    lower = pl.col(reference) * (1 + REF_LOWER_RETURN) + epsilon
    upper = pl.col(reference) * (1 + REF_UPPER_RETURN) - epsilon
    result = (
        pl.col(reference).is_not_null()
        & (pl.col(reference) > 0)
    )
    for suffix in price_suffixes:
        result = result & (pl.col(f"{prefix}_{suffix}") > lower) & (
            pl.col(f"{prefix}_{suffix}") < upper
        )
    return result


def _formal_after_trial(prefix: str) -> pl.Expr:
    book_after_transition = (
        (pl.col(f"{prefix}_recv_time") > pl.col(f"{prefix}_trial_time"))
        | (
            (pl.col(f"{prefix}_recv_time") == pl.col(f"{prefix}_trial_time"))
            & (pl.col(f"{prefix}_sequence") >= pl.col(f"{prefix}_trial_sequence"))
        )
    )
    return (
        (pl.col(f"{prefix}_trial_match") == 0) & book_after_transition
    ).fill_null(False)


def _add_eligibility_and_basis(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        ((pl.col("timestamp") - pl.col("spot_recv_time")).dt.total_nanoseconds() / 1_000_000)
        .alias("spot_age_ms"),
        ((pl.col("timestamp") - pl.col("fut_recv_time")).dt.total_nanoseconds() / 1_000_000)
        .alias("fut_age_ms"),
        (
            (pl.col("spot_recv_time") - pl.col("fut_recv_time"))
            .dt.total_nanoseconds()
            .abs()
            / 1_000_000
        ).alias("leg_skew_ms"),
        ((pl.col("spot_bid") + pl.col("spot_ask")) / 2).alias("spot_mid"),
        ((pl.col("fut_bid") + pl.col("fut_ask")) / 2).alias("fut_mid"),
        _ref_ok("spot", "spot_ref_price").alias("spot_ref_ok"),
        _ref_ok(
            "fut",
            "fut_ref_price",
            ("bid", "ask", "exec_bid", "exec_ask"),
        ).alias("fut_ref_ok"),
        _formal_after_trial("spot").alias("spot_formal"),
        _formal_after_trial("fut").alias("fut_formal"),
        (
            (pl.col("spot_bid") > 0)
            & (pl.col("spot_ask") > 0)
            & (pl.col("spot_bid_lots") > 0)
            & (pl.col("spot_ask_lots") > 0)
            & (pl.col("spot_bid") <= pl.col("spot_ask"))
        ).fill_null(False).alias("spot_book_ok"),
        (
            (pl.col("fut_bid") > 0)
            & (pl.col("fut_ask") > 0)
            & (pl.col("fut_bid_lots") > 0)
            & (pl.col("fut_ask_lots") > 0)
            & (pl.col("fut_bid") <= pl.col("fut_ask"))
        ).fill_null(False).alias("fut_book_ok"),
        (
            (pl.col("fut_exec_bid") > 0)
            & (pl.col("fut_exec_ask") > 0)
            & (pl.col("fut_exec_bid_lots") > 0)
            & (pl.col("fut_exec_ask_lots") > 0)
            & (pl.col("fut_exec_bid") <= pl.col("fut_exec_ask"))
        ).fill_null(False).alias("fut_exec_book_ok"),
    )
    frame = frame.with_columns(
        (
            pl.col("spot_formal")
            & pl.col("fut_formal")
            & pl.col("spot_book_ok")
            & pl.col("fut_book_ok")
            & pl.col("fut_exec_book_ok")
            & pl.col("spot_ref_ok")
            & pl.col("fut_ref_ok")
        ).fill_null(False).alias("eligible_base")
    )
    frame = frame.with_columns(
        *[
            (
                pl.col("eligible_base")
                & (pl.col("spot_age_ms") <= threshold)
                & (pl.col("fut_age_ms") <= threshold)
            ).fill_null(False).alias(f"eligible_{threshold}ms")
            for threshold in AGE_THRESHOLDS_MS
        ]
    )
    return frame.with_columns(
        pl.when(pl.col("eligible_base"))
        .then((pl.col("fut_mid") / pl.col("spot_mid") - 1) * 10_000)
        .otherwise(None)
        .alias("basis_mid_bp"),
        pl.when(pl.col("eligible_base"))
        .then((pl.col("fut_exec_bid") / pl.col("spot_ask") - 1) * 10_000)
        .otherwise(None)
        .alias("basis_sell_taker_bp"),
        pl.when(pl.col("eligible_base"))
        .then((pl.col("fut_exec_ask") / pl.col("spot_bid") - 1) * 10_000)
        .otherwise(None)
        .alias("basis_buy_taker_bp"),
        (pl.col("spot_mid") / pl.col("spot_ref_price") - 1).alias("spot_ref_return"),
        (pl.col("fut_mid") / pl.col("fut_ref_price") - 1).alias("fut_ref_return"),
    )


def _audit_frame(
    date: str,
    spot_events: pl.DataFrame,
    future_events: pl.DataFrame,
    landmarks: pl.DataFrame,
) -> pl.DataFrame:
    def clock_offset_ms(events: pl.DataFrame, quantile: float) -> float | None:
        result = events.select(
            (
                (
                    pl.col("recv_time")
                    - (pl.col("trans_time") - pl.duration(hours=8))
                ).dt.total_microseconds()
                / 1000
            ).quantile(quantile).alias("offset_ms")
        ).item()
        return float(result) if result is not None else None

    metrics: dict[str, object] = {
        "Date": date,
        "symbols": landmarks["ValueCode"].n_unique(),
        "spot_event_rows": spot_events.height,
        "future_event_rows": future_events.height,
        "landmark_rows": landmarks.height,
        "spot_clock_offset_p50_ms": clock_offset_ms(spot_events, 0.50),
        "spot_clock_offset_p99_ms": clock_offset_ms(spot_events, 0.99),
        "future_clock_offset_p50_ms": clock_offset_ms(future_events, 0.50),
        "future_clock_offset_p99_ms": clock_offset_ms(future_events, 0.99),
    }
    for column in (
        "eligible_base",
        *[f"eligible_{threshold}ms" for threshold in AGE_THRESHOLDS_MS],
    ):
        metrics[f"{column}_rows"] = int(landmarks[column].sum())
    metrics["spot_trial_block_rows"] = int((~landmarks["spot_formal"].fill_null(False)).sum())
    metrics["future_trial_block_rows"] = int((~landmarks["fut_formal"].fill_null(False)).sum())
    metrics["spot_ref_block_rows"] = int((~landmarks["spot_ref_ok"].fill_null(False)).sum())
    metrics["future_ref_block_rows"] = int((~landmarks["fut_ref_ok"].fill_null(False)).sum())
    return pl.DataFrame([metrics])


def build_landmarks(
    date: str,
    value_codes: list[str] | None = None,
    interval: str = "1s",
    cache_dir: Path | None = None,
    mapping_override: pl.DataFrame | None = None,
) -> LandmarkBuildResult:
    """Build causal fixed-grid basis landmarks for one date."""
    cache_dir = cache_dir or (DEFAULT_OUTPUT_ROOT / "metadata")
    if mapping_override is None:
        mapping = load_contract_mapping(
            date,
            value_codes=value_codes,
            cache_dir=cache_dir,
        )
    else:
        if value_codes is not None:
            raise ValueError("value_codes cannot be combined with mapping_override")
        mapping = mapping_override
    spot_events = _load_spot_events(date, mapping)
    future_events = _load_future_events(date, mapping)
    if spot_events.height == 0 or future_events.height == 0:
        raise RuntimeError(f"{date}: empty spot or futures event set")

    grid = _session_grid(date, mapping, interval)
    for state, right_on in (
        (_book_state(spot_events, "spot"), "spot_recv_time"),
        (_trial_transitions(spot_events, "spot"), "spot_trial_time"),
        (_book_state(future_events, "fut"), "fut_recv_time"),
        (_trial_transitions(future_events, "fut"), "fut_trial_time"),
    ):
        grid = _join_state(grid, state, right_on)
    landmarks = _add_eligibility_and_basis(grid)
    audit = _audit_frame(date, spot_events, future_events, landmarks)
    return LandmarkBuildResult(landmarks=landmarks, audit=audit, mapping=mapping)
