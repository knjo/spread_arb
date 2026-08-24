"""Spot-maker partial-fill timing diagnostics from indexed raw replay."""

from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl

from .indexed_replay import IndexedTradeReplay
from .replay import IndependentOrderWindow


DEFAULT_PARTIAL_HORIZONS_MS: tuple[int, ...] = (50, 250, 1000, 2000, 5000)


def summarize_spot_partial_completion(
    aliases: pl.DataFrame,
    *,
    horizons_ms: Sequence[int] = DEFAULT_PARTIAL_HORIZONS_MS,
) -> pl.DataFrame:
    """Report time from first own lot to a two-lot hedge unit.

    Every order with a first fill stays in the denominator.  Reaching the
    nominal cancellation boundary while still partial is a known competing
    failure for the moving-quote policy, not missing data: that generation
    can no longer accumulate its second maker lot after cancellation.
    """

    required = {
        "route",
        "boundary_quantile",
        "raw_order_fact_id",
        "any_fill",
        "full_fill",
        "partial_fill",
        "first_fill_recv_time_ns",
        "full_fill_recv_time_ns",
        "nominal_stop_recv_time_ns",
    }
    missing = sorted(required - set(aliases.columns))
    if missing:
        raise ValueError(f"order aliases missing columns: {missing}")
    horizons = tuple(int(value) for value in horizons_ms)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_ms must be positive")
    if any(right <= left for left, right in zip(horizons, horizons[1:])):
        raise ValueError("horizons_ms must be strictly increasing")

    spot = aliases.filter(
        (pl.col("route") == "spot_bid_future_taker")
        & (pl.col("any_fill") == True)  # noqa: E712
    )
    if spot.is_empty():
        return pl.DataFrame()
    rows: list[dict[str, object]] = []
    for quantile in sorted(spot["boundary_quantile"].unique().to_list()):
        selected = spot.filter(pl.col("boundary_quantile") == quantile)
        for horizon_ms in horizons:
            horizon_ns = horizon_ms * 1_000_000
            complete = (
                (pl.col("full_fill") == True)  # noqa: E712
                & (
                    pl.col("full_fill_recv_time_ns")
                    - pl.col("first_fill_recv_time_ns")
                    <= horizon_ns
                )
            )
            values = selected.select(
                pl.len().alias("any_fill_orders"),
                pl.col("full_fill").fill_null(False).sum().alias("eventual_full"),
                complete.sum().alias("complete_two_lots"),
            ).row(0, named=True)
            denominator = int(values["any_fill_orders"])
            values.update(
                {
                    "boundary_quantile": int(quantile),
                    "horizon_ms": horizon_ms,
                    "p_complete_two_lots_given_any_fill": (
                        int(values["complete_two_lots"]) / denominator
                        if denominator
                        else None
                    ),
                }
            )
            rows.append(values)
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["boundary_quantile", "horizon_ms"]
    )


def build_quantity_path(
    replay: IndexedTradeReplay,
    windows: Iterable[IndependentOrderWindow],
    *,
    intended_quantity: int = 2,
) -> pl.DataFrame:
    """Expose the exact first/full cursor path available from indexed replay.

    V0 stores at most the first and full cumulative-quantity milestones.  It
    is sufficient for the standard two-spot-lot contract used by this pilot;
    larger quantities will need incremental allocation in WP05.
    """

    records: list[dict[str, object]] = []
    for window in windows:
        label = replay.label_quantity(window, intended_quantity)
        first = label.first_fill_cursor
        full = label.full_fill_cursor
        if first is not None:
            records.append(
                {
                    "generation_id": window.generation_id,
                    "fill_recv_time_ns": first.recv_time_ns,
                    "fill_event_sequence": first.event_sequence,
                    "fill_row_index": first.row_index,
                    "cumulative_fill_quantity": 1,
                }
            )
        if full is not None and full != first:
            records.append(
                {
                    "generation_id": window.generation_id,
                    "fill_recv_time_ns": full.recv_time_ns,
                    "fill_event_sequence": full.event_sequence,
                    "fill_row_index": full.row_index,
                    "cumulative_fill_quantity": intended_quantity,
                }
            )
    return (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame()
    )
