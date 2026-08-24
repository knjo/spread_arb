"""End-to-end WP02 independent-event raw fill pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

import polars as pl

from .engine import build_layered_order_windows
from .indexed_replay import IndexedTradeReplay
from .layered import EventCursor
from .merged import (
    build_merged_target_input,
    observations_for_policy,
    session_cutoff_cursor,
)
from .pilot import BOUNDARY_QUANTILES, DEFAULT_DATES, DEFAULT_SYMBOLS, ENTRY_ROUTES
from .replay import TradeEvent
from .targets import ROUTE_SPECS, absolute_price_tick


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "quote_fill"


@dataclass(frozen=True)
class RawFillStudyResult:
    target_observations: pl.DataFrame
    order_aliases: pl.DataFrame
    raw_order_facts: pl.DataFrame
    by_day_symbol: pl.DataFrame
    summary: pl.DataFrame
    audit: pl.DataFrame


def run_raw_fill_study(
    dates: Iterable[str],
    symbols: Iterable[str],
    *,
    output_dir: Path | None = None,
) -> RawFillStudyResult:
    """Run the independent-event fill study one date at a time."""

    dates = [str(value) for value in dates]
    symbols = [str(value) for value in symbols]
    observation_frames: list[pl.DataFrame] = []
    alias_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    # Process one product-day at a time.  The raw files are large and a
    # four-product normalized tape can exceed the research workstation's
    # memory even though each independent replay is small.
    for date in dates:
        for requested_value_code in symbols:
            try:
                merged = build_merged_target_input(date, [requested_value_code])
            except ValueError as error:
                if "no exact fair-panel mappings" in str(error):
                    continue
                raise
            observation_frames.append(merged.observations)
            audit_frames.append(
                merged.audit.join(
                    merged.raw_tape.audit.group_by("Date", "ValueCode").agg(
                        pl.col("event_rows").sum().alias("raw_rows"),
                        pl.col("trade_rows").sum().alias("raw_trades"),
                        pl.col("zero_book_trade_rows")
                        .sum()
                        .alias("zero_book_trade_rows"),
                        pl.col("rows_without_prior_book")
                        .sum()
                        .alias("rows_without_prior_book"),
                    ),
                    on=["Date", "ValueCode"],
                    how="left",
                )
            )
            for value_code in merged.mapping["ValueCode"].to_list():
                for route in ENTRY_ROUTES:
                    market = ROUTE_SPECS[route].maker_market
                    tape = (
                        merged.raw_tape.future_trades
                        if market == "future"
                        else merged.raw_tape.spot_trades
                    ).filter(pl.col("ValueCode") == value_code)
                    trade_index = IndexedTradeReplay(
                        _trade_events(tape, market=market)
                    )
                    for quantile in BOUNDARY_QUANTILES:
                        observations = observations_for_policy(
                            merged.observations,
                            date=date,
                            value_code=value_code,
                            route=route,
                            boundary_quantile=quantile,
                        )
                        if not observations:
                            continue
                        cutoff = session_cutoff_cursor(date)
                        observations = tuple(
                            item for item in observations if item.cursor < cutoff
                        )
                        if not observations:
                            continue
                        policy_id = f"{date}/{value_code}/{route}/q{quantile}"
                        built = build_layered_order_windows(
                            observations,
                            route=route,
                            policy_id=policy_id,
                            cutoff_cursor=cutoff,
                        )
                        alias_frames.append(
                            _label_policy_windows(
                                merged.observations,
                                built.windows,
                                trade_index,
                                date=date,
                                value_code=value_code,
                                route=route,
                                boundary_quantile=quantile,
                                peak_nominal_layers=built.peak_active_layers,
                            )
                        )

    observations = _concat(observation_frames)
    aliases = _concat(alias_frames)
    audit = _concat(audit_frames)
    raw_facts = summarize_raw_order_facts(aliases)
    by_day = summarize_fill_by_day_symbol(aliases)
    summary = summarize_fill_study(by_day)
    result = RawFillStudyResult(
        observations,
        aliases,
        raw_facts,
        by_day,
        summary,
        audit,
    )
    if output_dir is not None:
        write_raw_fill_study(result, output_dir, dates=dates, symbols=symbols)
    return result


def _trade_events(
    frame: pl.DataFrame,
    *,
    market: Literal["spot", "future"],
) -> tuple[TradeEvent, ...]:
    priority = 1 if market == "future" else 2
    return tuple(
        TradeEvent(
            EventCursor(
                int(row["recv_time_ns"]),
                priority,
                int(row["sequence"]),
            ),
            absolute_price_tick(
                float(row["trade_price"]),
                market=market,
                session_date=(
                    str(row["Date"])
                    if row.get("Date") is not None
                    else None
                ),
            ),
            int(row["trade_lots"]),
        )
        for row in frame.sort(
            ["recv_time_ns", "sequence", "packet_sequence"]
        ).iter_rows(named=True)
    )


def _label_policy_windows(
    observations_frame: pl.DataFrame,
    windows: tuple,
    replay: IndexedTradeReplay,
    *,
    date: str,
    value_code: str,
    route: str,
    boundary_quantile: int,
    peak_nominal_layers: int,
) -> pl.DataFrame:
    if not windows:
        return pl.DataFrame()
    policy_rows = observations_frame.filter(
        (pl.col("Date") == date)
        & (pl.col("ValueCode") == value_code)
        & (pl.col("route") == route)
        & (pl.col("boundary_quantile") == boundary_quantile)
        & pl.col("gate_open")
    )
    start_metadata = {
        (
            int(row["recv_time_ns"]),
            int(row["cursor_event_sequence"]),
            int(row["cursor_row_index"]),
            int(row["absolute_target_tick"]),
        ): row
        for row in policy_rows.iter_rows(named=True)
    }
    intended_quantity = 1 if ROUTE_SPECS[route].maker_market == "future" else 2
    first_labels = replay.label_windows(windows)
    records: list[dict[str, object]] = []
    for window, first_label in zip(windows, first_labels):
        quantity = replay.label_quantity(window, intended_quantity)
        key = (
            window.start_cursor.recv_time_ns,
            window.start_cursor.event_sequence,
            window.start_cursor.row_index,
            window.target_price_tick,
        )
        metadata = start_metadata[key]
        raw_id = _raw_fact_id(
            date,
            value_code,
            str(metadata["QuoteCode"]),
            route,
            int(metadata["spread_pair_epoch"]),
            window.target_price_tick,
            window.start_cursor,
        )
        full_cursor = quantity.full_fill_cursor
        terminal_cursor = full_cursor or window.stop_cursor
        if quantity.full_fill is True:
            terminal_reason = "full_fill"
        elif quantity.partial_fill:
            terminal_reason = f"partial_then_{window.stop_reason}"
        elif quantity.full_fill is None:
            terminal_reason = f"unknown_queue_then_{window.stop_reason}"
        else:
            terminal_reason = window.stop_reason
        shadows = {
            f"shadow_touch_{label.horizon_ms}ms": label.touched_after_stop
            for label in first_label.shadow_labels
        }
        shadows.update(
            {
                f"shadow_through_{label.horizon_ms}ms": (
                    label.trade_through_after_stop
                )
                for label in first_label.shadow_labels
            }
        )
        records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": metadata["QuoteCode"],
                "route": route,
                "maker_market": ROUTE_SPECS[route].maker_market,
                "maker_side": ROUTE_SPECS[route].maker_side,
                "boundary_quantile": boundary_quantile,
                "boundary_role": metadata.get("boundary_role"),
                "parameter_version": metadata.get("parameter_version"),
                "source_asof_date": metadata.get("source_asof_date"),
                "spread_pair_epoch": metadata["spread_pair_epoch"],
                "raw_order_fact_id": raw_id,
                "policy_generation_id": window.generation_id,
                "target_price_tick": window.target_price_tick,
                "target_price": metadata["absolute_target_price"],
                "target_rank_at_submit": metadata.get("target_rank"),
                "threshold_basis_bp": metadata.get("threshold_basis_bp"),
                "effective_basis_bp": metadata.get("effective_basis_bp"),
                "initial_queue_ahead": window.initial_queue_ahead,
                "queue_known": quantity.queue_known,
                "intended_quantity": intended_quantity,
                "submit_recv_time_ns": window.start_cursor.recv_time_ns,
                "submit_event_sequence": window.start_cursor.event_sequence,
                "submit_row_index": window.start_cursor.row_index,
                "nominal_stop_recv_time_ns": window.stop_cursor.recv_time_ns,
                "nominal_stop_reason": window.stop_reason,
                "first_fill_recv_time_ns": (
                    quantity.first_fill_cursor.recv_time_ns
                    if quantity.first_fill_cursor is not None
                    else None
                ),
                "first_fill_event_sequence": (
                    quantity.first_fill_cursor.event_sequence
                    if quantity.first_fill_cursor is not None
                    else None
                ),
                "first_fill_row_index": (
                    quantity.first_fill_cursor.row_index
                    if quantity.first_fill_cursor is not None
                    else None
                ),
                "full_fill_recv_time_ns": (
                    full_cursor.recv_time_ns if full_cursor is not None else None
                ),
                "full_fill_event_sequence": (
                    full_cursor.event_sequence if full_cursor is not None else None
                ),
                "full_fill_row_index": (
                    full_cursor.row_index if full_cursor is not None else None
                ),
                "known_filled_quantity": quantity.known_filled_quantity_before_stop,
                "any_fill": quantity.any_fill,
                "full_fill": quantity.full_fill,
                "partial_fill": quantity.partial_fill,
                "trade_through_fill": quantity.trade_through_fill,
                "fill_reason": first_label.fill_reason,
                "terminal_recv_time_ns": terminal_cursor.recv_time_ns,
                "terminal_reason": terminal_reason,
                "cancel_required": quantity.full_fill is not True,
                "lifetime_ms": (
                    terminal_cursor.recv_time_ns
                    - window.start_cursor.recv_time_ns
                )
                / 1_000_000.0,
                "time_first_to_full_ms": (
                    (full_cursor.recv_time_ns - quantity.first_fill_cursor.recv_time_ns)
                    / 1_000_000.0
                    if full_cursor is not None
                    and quantity.first_fill_cursor is not None
                    else None
                ),
                "spot_book_age_ms_at_submit": metadata.get("spot_book_age_ms"),
                "future_book_age_ms_at_submit": metadata.get(
                    "future_book_age_ms"
                ),
                "peak_nominal_layers_policy_day": peak_nominal_layers,
                "independent_event_label": True,
                "joint_volume_allocated": False,
                **shadows,
            }
        )
    return pl.from_dicts(records, infer_schema_length=None)


def _raw_fact_id(
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
    epoch: int,
    target_tick: int,
    cursor: EventCursor,
) -> str:
    identity = (
        f"{date}|{value_code}|{quote_code}|{route}|{epoch}|{target_tick}|"
        f"{cursor.recv_time_ns}|{cursor.event_sequence}|{cursor.row_index}"
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]


def summarize_raw_order_facts(aliases: pl.DataFrame) -> pl.DataFrame:
    if aliases.is_empty():
        return pl.DataFrame()
    return aliases.group_by("raw_order_fact_id").agg(
        pl.col("Date").first(),
        pl.col("ValueCode").first(),
        pl.col("QuoteCode").first(),
        pl.col("route").first(),
        pl.col("maker_market").first(),
        pl.col("maker_side").first(),
        pl.col("spread_pair_epoch").first(),
        pl.col("target_price_tick").first(),
        pl.col("target_price").first(),
        pl.col("submit_recv_time_ns").first(),
        pl.col("initial_queue_ahead").first(),
        pl.len().alias("policy_alias_count"),
        pl.col("boundary_quantile").sort().alias("boundary_quantile_aliases"),
        pl.col("nominal_stop_recv_time_ns").min().alias("earliest_alias_stop_ns"),
        pl.col("nominal_stop_recv_time_ns").max().alias("latest_alias_stop_ns"),
    ).sort(["Date", "ValueCode", "route", "submit_recv_time_ns"])


def summarize_fill_by_day_symbol(aliases: pl.DataFrame) -> pl.DataFrame:
    if aliases.is_empty():
        return pl.DataFrame()
    group = ["Date", "ValueCode", "route", "boundary_quantile"]
    return aliases.group_by(group).agg(
        pl.len().alias("orders"),
        pl.col("raw_order_fact_id").n_unique().alias("unique_raw_order_facts"),
        pl.col("queue_known").sum().alias("queue_known_orders"),
        pl.col("any_fill").is_not_null().sum().alias("any_fill_observed_n"),
        pl.col("any_fill").fill_null(False).sum().alias("any_fills"),
        pl.col("full_fill").is_not_null().sum().alias("full_fill_observed_n"),
        pl.col("full_fill").fill_null(False).sum().alias("full_fills"),
        pl.col("partial_fill").sum().alias("partial_fills"),
        pl.col("cancel_required").sum().alias("cancel_required_orders"),
        (pl.col("nominal_stop_reason") == "target_retreat")
        .sum()
        .alias("nominal_target_retreat_stops"),
        (pl.col("nominal_stop_reason") == "session_cutoff")
        .sum()
        .alias("nominal_session_cutoff_stops"),
        pl.col("trade_through_fill").sum().alias("trade_through_fills"),
        pl.col("lifetime_ms").median().alias("lifetime_ms_p50"),
        pl.col("lifetime_ms").quantile(0.95).alias("lifetime_ms_p95"),
        pl.col("time_first_to_full_ms")
        .drop_nulls()
        .median()
        .alias("time_first_to_full_ms_p50"),
        pl.col("peak_nominal_layers_policy_day")
        .max()
        .alias("peak_nominal_layers"),
        *[
            pl.col(f"shadow_through_{horizon}ms")
            .sum()
            .alias(f"shadow_through_{horizon}ms")
            for horizon in (10, 50, 100, 500)
        ],
    ).with_columns(
        (pl.col("any_fills") / pl.col("orders")).alias(
            "p_any_fill_all_lower_bound"
        ),
        (pl.col("any_fills") / pl.col("any_fill_observed_n"))
        .fill_nan(None)
        .alias("p_any_fill_known_queue"),
        (pl.col("full_fills") / pl.col("orders")).alias(
            "p_full_fill_all_lower_bound"
        ),
        (pl.col("full_fills") / pl.col("full_fill_observed_n"))
        .fill_nan(None)
        .alias("p_full_fill_known_queue"),
        (pl.col("cancel_required_orders") / pl.col("orders")).alias(
            "cancel_required_rate"
        ),
    ).sort(group)


def summarize_fill_study(by_day: pl.DataFrame) -> pl.DataFrame:
    if by_day.is_empty():
        return pl.DataFrame()
    group = ["ValueCode", "route", "boundary_quantile"]
    return by_day.group_by(group).agg(
        pl.col("Date").n_unique().alias("dates"),
        pl.len().alias("pair_days"),
        pl.col("orders").sum().alias("orders"),
        pl.col("unique_raw_order_facts").sum().alias("unique_raw_order_facts"),
        pl.col("any_fills").sum().alias("any_fills"),
        pl.col("full_fills").sum().alias("full_fills"),
        pl.col("partial_fills").sum().alias("partial_fills"),
        pl.col("cancel_required_orders").sum().alias("cancel_required_orders"),
        pl.col("p_any_fill_all_lower_bound")
        .median()
        .alias("pair_median_p_any_fill_lower_bound"),
        pl.col("p_full_fill_all_lower_bound")
        .median()
        .alias("pair_median_p_full_fill_lower_bound"),
        pl.col("cancel_required_rate")
        .median()
        .alias("pair_median_cancel_required_rate"),
        pl.col("peak_nominal_layers").median().alias("pair_median_peak_layers"),
        pl.col("peak_nominal_layers").max().alias("max_peak_layers"),
    ).with_columns(
        (pl.col("any_fills") / pl.col("orders")).alias(
            "event_weighted_p_any_fill_lower_bound"
        ),
        (pl.col("full_fills") / pl.col("orders")).alias(
            "event_weighted_p_full_fill_lower_bound"
        ),
        (pl.col("cancel_required_orders") / pl.col("orders")).alias(
            "event_weighted_cancel_required_rate"
        ),
    ).sort(group)


def write_raw_fill_study(
    result: RawFillStudyResult,
    output_dir: Path,
    *,
    dates: list[str],
    symbols: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.target_observations.write_parquet(
        output_dir / "target_observations.parquet"
    )
    result.order_aliases.write_parquet(output_dir / "order_aliases.parquet")
    result.raw_order_facts.write_parquet(output_dir / "raw_order_facts.parquet")
    result.by_day_symbol.write_csv(output_dir / "fill_by_day_symbol.csv")
    result.summary.write_csv(output_dir / "fill_summary.csv")
    result.audit.write_csv(output_dir / "raw_replay_audit.csv")
    config = {
        "dates": dates,
        "symbols": symbols,
        "spread_pair_clock": "SpreadPairTotalCount",
        "boundary_quantiles": list(BOUNDARY_QUANTILES),
        "entry_routes": list(ENTRY_ROUTES),
        "fixed_bp_actionable": False,
        "independent_event_label": True,
        "joint_volume_allocated": False,
        "queue_first_fill_rule": "same_price_volume > initial_visible_queue",
        "shadow_cancel_latency_ms": [10, 50, 100, 500],
        "execution_state": "raw_receive_time_asof",
        "fair_anchor_state": "one_second_ewma120",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [frame for frame in frames if not frame.is_empty()]
    return (
        pl.concat(nonempty, how="diagonal_relaxed")
        if nonempty
        else pl.DataFrame()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WP02 raw maker-fill pilot")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_raw_fill_study(
        args.dates,
        args.symbols,
        output_dir=args.output_dir,
    )
    print(result.summary)


if __name__ == "__main__":
    main()
