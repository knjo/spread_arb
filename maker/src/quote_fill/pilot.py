"""Sparse WP02 base-epoch intent builder.

This module deliberately stops before raw maker-fill replay.  It joins the
existing spot ``SpreadPairTotalCount`` clock to the one-second causal fair
panel and the D-1-safe adaptive snapshot, then materialises the two entry-route
targets for p50/p80.  No forward fair labels are loaded.
"""

from __future__ import annotations

import argparse
from datetime import time
import math
from pathlib import Path
from typing import Iterable

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .targets import (
    ROUTE_SPECS,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    price_in_ref_band,
    price_to_tick_index,
    target_price_for_basis,
)


DEFAULT_DATES = (
    "20260128",
    "20260223",
    "20260318",
    "20260420",
    "20260609",
    "20260617",
    "20260720",
    "20260811",
)
DEFAULT_SYMBOLS = ("2303", "2317", "2603", "2881")
ENTRY_ROUTES = ("future_ask_spot_taker", "spot_bid_future_taker")
BOUNDARY_QUANTILES = (50, 80)
SESSION_START = time(9, 5)
SESSION_END = time(13, 20)

DEFAULT_FAIR_PANEL_PATH = MAKER_ROOT / "data" / "fair_mid" / "fair_anchor_panel.parquet"
DEFAULT_SNAPSHOT_PATH = (
    MAKER_ROOT
    / "data"
    / "quote_width"
    / "adaptive"
    / "adaptive_parameter_snapshot_by_day_symbol.csv"
)


def _require_columns(frame: pl.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _normalise_symbols(symbols: Iterable[str]) -> list[str]:
    result = sorted({str(symbol) for symbol in symbols})
    if not result:
        raise ValueError("symbols must not be empty")
    return result


def load_spot_feature_state(
    date: str,
    symbols: Iterable[str],
    *,
    data_root: Path = HFT_DATA_ROOT,
) -> pl.DataFrame:
    """Exact-join market-open spot ticks and their causal feature rows."""
    symbols = _normalise_symbols(symbols)
    tick_path = data_root / "tickData" / f"{date}_StockTick.parquet"
    feature_path = data_root / "tickFeature" / f"{date}_tickFeature.parquet"
    for path in (tick_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(path)

    tick = (
        pl.scan_parquet(tick_path)
        .filter(pl.col("QuoteCode").is_in(symbols) & pl.col("marketOpen"))
        .select(
            pl.lit(str(date)).alias("Date"),
            pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("spot_channel_seq"),
            pl.col("RecvTime").cast(pl.Datetime("ns")).alias("event_recv_time"),
            pl.col("TransTime").cast(pl.Datetime("us")).alias("event_trans_time"),
            pl.col("TrialMatch").cast(pl.Int16).alias("spot_trial_match"),
            pl.col("RefPrice").cast(pl.Float64).alias("spot_ref_price_current"),
            pl.col("BidPrice1").cast(pl.Float64).alias("spot_bid"),
            pl.col("BidPrice2").cast(pl.Float64).alias("spot_bid_2"),
            pl.col("AskPrice1").cast(pl.Float64).alias("spot_ask"),
            pl.col("AskPrice2").cast(pl.Float64).alias("spot_ask_2"),
            pl.col("BidLots1").cast(pl.Int64).alias("spot_bid_lots"),
            pl.col("AskLots1").cast(pl.Int64).alias("spot_ask_lots"),
        )
    )
    feature = (
        pl.scan_parquet(feature_path)
        .filter(pl.col("QuoteCode").is_in(symbols))
        .select(
            pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("spot_channel_seq"),
            pl.col("SpreadPairID").cast(pl.Int64).alias("spread_pair_id"),
            pl.col("SpreadPairSeq").cast(pl.Int64).alias("spread_pair_seq"),
            pl.col("SpreadPairTotalCount")
            .cast(pl.Int64)
            .alias("spread_pair_epoch"),
            pl.col("SpreadCountAtSameCount")
            .cast(pl.Int64)
            .alias("spread_count_at_same_count"),
        )
    )
    return (
        tick.join(
            feature,
            on=["ValueCode", "spot_channel_seq"],
            how="inner",
            validate="1:1",
        )
        .collect(engine="streaming")
        .sort(["ValueCode", "event_recv_time", "spot_channel_seq"])
    )


def load_causal_fair_state(
    date: str,
    symbols: Iterable[str],
    *,
    fair_panel_path: Path = DEFAULT_FAIR_PANEL_PATH,
) -> pl.DataFrame:
    """Load only current/causal columns from the WP01 one-second panel."""
    if not fair_panel_path.exists():
        raise FileNotFoundError(fair_panel_path)
    symbols = _normalise_symbols(symbols)
    # Keep this explicit allowlist: the source also contains future_* labels.
    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "spot_recv_time",
        "spot_sequence",
        "fut_recv_time",
        "fut_sequence",
        "spot_ref_price",
        "fut_ref_price",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "fut_exec_ask",
        "fut_bid_lots",
        "fut_ask_lots",
        "fut_exec_bid_lots",
        "fut_exec_ask_lots",
        "spot_age_ms",
        "fut_age_ms",
        "leg_skew_ms",
        "spot_formal",
        "fut_formal",
        "spot_book_ok",
        "fut_book_ok",
        "fut_exec_book_ok",
        "spot_ref_ok",
        "fut_ref_ok",
        "eligible_base",
        "analysis_eligible",
        "basis_mid_bp",
        "anchor_ewma_120s_bp",
    ]
    return (
        pl.scan_parquet(fair_panel_path)
        .filter(
            (pl.col("Date").cast(pl.String) == str(date))
            & pl.col("ValueCode").cast(pl.String).is_in(symbols)
        )
        .select(columns)
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("timestamp").cast(pl.Datetime("ns")).alias("fair_timestamp"),
        )
        .drop("timestamp")
        .collect(engine="streaming")
        .sort(["ValueCode", "fair_timestamp"])
    )


def load_adaptive_snapshot(
    date: str,
    symbols: Iterable[str],
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    quantiles: Iterable[int] = BOUNDARY_QUANTILES,
) -> pl.DataFrame:
    """Load p50/p80 D-1-safe boundaries with identifiers kept as strings."""
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)
    symbols = _normalise_symbols(symbols)
    quantiles = sorted({int(value) for value in quantiles})
    snapshot = pl.read_csv(
        snapshot_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "prior_date": pl.String,
            "source_asof_date": pl.String,
        },
        infer_schema_length=10_000,
    )
    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "boundary_role",
        "upper_distance_bp",
        "adaptive_parameter_valid",
        "parameter_version",
        "source_asof_date",
        "contains_target_day_outcome",
    ]
    _require_columns(snapshot, columns, str(snapshot_path))
    result = snapshot.filter(
        (pl.col("Date") == str(date))
        & pl.col("ValueCode").is_in(symbols)
        & pl.col("boundary_quantile").is_in(quantiles)
    ).select(columns)
    if result.select("Date", "ValueCode", "QuoteCode", "boundary_quantile").n_unique() != result.height:
        raise ValueError("adaptive snapshot contains duplicate candidate keys")
    if result.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("adaptive snapshot contains target-day outcomes")
    return result.sort(["ValueCode", "QuoteCode", "boundary_quantile"])


def _extract_base_epochs(spot_state: pl.DataFrame) -> pl.DataFrame:
    required = [
        "Date",
        "ValueCode",
        "spot_channel_seq",
        "event_recv_time",
        "event_trans_time",
        "spread_pair_id",
        "spread_pair_epoch",
    ]
    _require_columns(spot_state, required, "spot feature state")
    ordered = spot_state.sort(
        ["Date", "ValueCode", "event_recv_time", "spot_channel_seq"]
    ).with_columns(
        pl.col("spread_pair_epoch")
        .shift(1)
        .over(["Date", "ValueCode"])
        .alias("_previous_epoch")
    )
    current_time = pl.col("event_trans_time").dt.time()
    return (
        ordered.filter(
            (pl.col("spread_pair_id") > 0)
            & pl.col("spread_pair_epoch").is_not_null()
            & pl.col("_previous_epoch").is_not_null()
            & (pl.col("spread_pair_epoch") != pl.col("_previous_epoch"))
            & (current_time >= pl.lit(SESSION_START))
            & (current_time < pl.lit(SESSION_END))
        )
        .drop("_previous_epoch")
        .with_columns(
            pl.concat_str(
                "Date",
                "ValueCode",
                pl.col("spread_pair_epoch").cast(pl.String),
                separator="/",
            ).alias("base_event_id")
        )
    )


def _safe_bool(value: object) -> bool:
    return value is True


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _target_rank(route: str, target: float, row: dict[str, object]) -> str:
    if route == "future_ask_spot_taker":
        bbo = float(row["fut_ask"])
        if target < bbo and not math.isclose(target, bbo, abs_tol=1e-8):
            return "inside"
        if math.isclose(target, bbo, abs_tol=1e-8):
            return "A1"
        return "behind_A1"
    bid1 = float(row["spot_bid"])
    bid2 = row.get("spot_bid_2")
    if target > bid1 and not math.isclose(target, bid1, abs_tol=1e-8):
        return "inside"
    if math.isclose(target, bid1, abs_tol=1e-8):
        return "B1"
    if _finite_positive(bid2) and math.isclose(target, float(bid2), abs_tol=1e-8):
        return "B2"
    return "behind_B1"


def _materialise_route_rows(joined: pl.DataFrame) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        fair_available = _finite_positive(row.get("fut_exec_bid")) and _finite_positive(
            row.get("spot_ask")
        )
        upper = row.get("upper_distance_bp")
        anchor = row.get("anchor_ewma_120s_bp")
        threshold = (
            float(anchor) + float(upper)
            if isinstance(anchor, (int, float))
            and isinstance(upper, (int, float))
            and math.isfinite(float(anchor))
            and math.isfinite(float(upper))
            else None
        )
        spot_book_ok = (
            _finite_positive(row.get("spot_bid"))
            and _finite_positive(row.get("spot_ask"))
            and _finite_positive(row.get("spot_bid_lots"))
            and _finite_positive(row.get("spot_ask_lots"))
            and float(row["spot_bid"]) <= float(row["spot_ask"])
        )
        spot_ref_ok = spot_book_ok and price_in_ref_band(
            float(row["spot_bid"]), float(row["spot_ref_price_current"])
        ) and price_in_ref_band(
            float(row["spot_ask"]), float(row["spot_ref_price_current"])
        )
        common_gates = {
            "spot_formal_current": row.get("spot_trial_match") == 0,
            "spot_book_ok_current": spot_book_ok,
            "spot_ref_ok_current": spot_ref_ok,
            "fair_state_available": fair_available,
            "fair_analysis_eligible": _safe_bool(row.get("analysis_eligible")),
            "future_formal": _safe_bool(row.get("fut_formal")),
            "future_book_ok": _safe_bool(row.get("fut_book_ok"))
            and _safe_bool(row.get("fut_exec_book_ok")),
            "future_ref_ok": _safe_bool(row.get("fut_ref_ok")),
            "adaptive_parameter_valid": _safe_bool(
                row.get("adaptive_parameter_valid")
            ),
        }
        for route in ENTRY_ROUTES:
            spec = ROUTE_SPECS[route]
            target: float | None = None
            target_tick: int | None = None
            effective: float | None = None
            passive = False
            target_ref_ok = False
            target_rank: str | None = None
            offset_ticks: int | None = None
            if threshold is not None and fair_available:
                try:
                    target = target_price_for_basis(
                        route,
                        threshold,
                        session_date=str(row["Date"]),
                        spot_ask=float(row["spot_ask"]),
                        fut_exec_bid=float(row["fut_exec_bid"]),
                    )
                    target_tick = absolute_price_tick(
                        target,
                        market=spec.maker_market,
                        session_date=str(row["Date"]),
                    )
                    effective = effective_basis_bp(
                        route,
                        target,
                        spot_ask=float(row["spot_ask"]),
                        fut_exec_bid=float(row["fut_exec_bid"]),
                    )
                    passive = is_passive_target(
                        route,
                        target,
                        spot_ask=float(row["spot_ask"]),
                        fut_exec_bid=float(row["fut_exec_bid"]),
                    )
                    reference = (
                        row.get("fut_ref_price")
                        if spec.maker_market == "future"
                        else row.get("spot_ref_price_current")
                    )
                    target_ref_ok = _finite_positive(reference) and price_in_ref_band(
                        target, float(reference)
                    )
                    bbo = (
                        row.get("fut_ask")
                        if route == "future_ask_spot_taker"
                        else row.get("spot_bid")
                    )
                    if _finite_positive(bbo):
                        bbo_tick = int(
                            round(
                                price_to_tick_index(
                                    float(bbo),
                                    market=spec.maker_market,
                                    session_date=str(row["Date"]),
                                )
                            )
                        )
                        offset_ticks = (
                            target_tick - bbo_tick
                            if spec.maker_side == "ask"
                            else bbo_tick - target_tick
                        )
                        target_rank = _target_rank(route, target, row)
                except ValueError:
                    target = None
                    target_tick = None
                    effective = None

            route_gates = {
                **common_gates,
                "target_available": target is not None,
                "target_passive": passive,
                "target_ref_ok": target_ref_ok,
            }
            gate_open = all(route_gates.values())
            reason_order = (
                ("spot_formal_current", "spot_trial_match"),
                ("spot_book_ok_current", "spot_book_gate"),
                ("spot_ref_ok_current", "spot_ref_gate"),
                ("fair_state_available", "missing_fair_state"),
                ("fair_analysis_eligible", "fair_ineligible"),
                ("future_formal", "future_trial_match"),
                ("future_book_ok", "future_book_gate"),
                ("future_ref_ok", "future_ref_gate"),
                ("adaptive_parameter_valid", "invalid_adaptive_parameter"),
                ("target_available", "target_unavailable"),
                ("target_passive", "target_not_passive"),
                ("target_ref_ok", "target_ref_gate"),
            )
            admission_reason = "admitted" if gate_open else next(
                reason for gate, reason in reason_order if not route_gates[gate]
            )
            records.append(
                {
                    **row,
                    "route": route,
                    "stage": spec.stage,
                    "maker_market": spec.maker_market,
                    "maker_side": spec.maker_side,
                    "threshold_basis_bp": threshold,
                    "absolute_target_price": target,
                    "absolute_target_tick": target_tick,
                    "effective_basis_bp": effective,
                    "target_offset_ticks": offset_ticks,
                    "target_rank": target_rank,
                    **route_gates,
                    "gate_open": gate_open,
                    "admission_reason": admission_reason,
                }
            )
    # Some product-days start with invalid/null adaptive rows and only expose
    # finite candidate values after Polars' default 100-row inference window.
    # Full inference keeps those diagnostic rows without order-dependent dtype
    # failures.
    result = (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame()
    )
    if result.is_empty():
        return result
    # Python row materialisation otherwise silently lowers nanosecond inputs
    # to microseconds.  Raw merged replay uses recv_time_ns as its canonical
    # clock, but the diagnostic frame must still preserve the declared type.
    timestamp_columns = [
        name for name in ("event_recv_time", "fair_timestamp")
        if name in result.columns
    ]
    return result.with_columns(
        *(pl.col(name).cast(pl.Datetime("ns")) for name in timestamp_columns)
    )


def build_base_epoch_intents_from_frames(
    spot_state: pl.DataFrame,
    fair_state: pl.DataFrame,
    adaptive_snapshot: pl.DataFrame,
    *,
    quantiles: Iterable[int] = BOUNDARY_QUANTILES,
) -> pl.DataFrame:
    """Pure frame-level builder used by tests and the filesystem runner."""
    base = _extract_base_epochs(spot_state)
    if base.is_empty():
        return pl.DataFrame()
    _require_columns(
        fair_state,
        ["Date", "ValueCode", "QuoteCode", "fair_timestamp"],
        "causal fair state",
    )
    _require_columns(
        adaptive_snapshot,
        [
            "Date",
            "ValueCode",
            "QuoteCode",
            "boundary_quantile",
            "upper_distance_bp",
            "adaptive_parameter_valid",
        ],
        "adaptive snapshot",
    )
    joined = base.sort(["ValueCode", "event_recv_time"]).join_asof(
        fair_state.sort(["ValueCode", "fair_timestamp"]),
        left_on="event_recv_time",
        right_on="fair_timestamp",
        by="ValueCode",
        strategy="backward",
        check_sortedness=False,
        suffix="_fair",
    )
    quantile_frame = pl.DataFrame(
        {"boundary_quantile": sorted({int(value) for value in quantiles})}
    )
    joined = joined.join(quantile_frame, how="cross").join(
        adaptive_snapshot,
        on=["Date", "ValueCode", "QuoteCode", "boundary_quantile"],
        how="left",
        validate="m:1",
        suffix="_snapshot",
    )
    result = _materialise_route_rows(joined)
    if result.is_empty():
        return result
    return result.sort(
        [
            "Date",
            "ValueCode",
            "event_recv_time",
            "spot_channel_seq",
            "boundary_quantile",
            "route",
        ]
    )


def build_base_epoch_intents(
    date: str,
    symbols: Iterable[str],
    *,
    data_root: Path = HFT_DATA_ROOT,
    fair_panel_path: Path = DEFAULT_FAIR_PANEL_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> pl.DataFrame:
    """Load one product-day set and build sparse p50/p80 entry intents."""
    symbols = _normalise_symbols(symbols)
    return build_base_epoch_intents_from_frames(
        load_spot_feature_state(date, symbols, data_root=data_root),
        load_causal_fair_state(
            date, symbols, fair_panel_path=fair_panel_path
        ),
        load_adaptive_snapshot(
            date, symbols, snapshot_path=snapshot_path
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse WP02 base intents")
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_base_epoch_intents(args.date, args.symbols)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(args.output)
    print(
        result.group_by("route", "gate_open")
        .agg(pl.len().alias("rows"))
        .sort("route", "gate_open")
    )


if __name__ == "__main__":
    main()
