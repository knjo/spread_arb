"""Causal merged-state target observations for the WP02 raw pilot.

The one-second panel supplies only the fair anchor.  Every execution decision
(opposite taker price, maker BBO, queue, TrialMatch and reference gate) is
recomputed from the raw spot/futures tapes at its exact receive cursor.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import heapq
import math
from pathlib import Path
from typing import Iterable, Iterator, Literal

import polars as pl

from ..common.paths import HFT_DATA_ROOT
from .engine import TargetObservation
from .layered import EventCursor
from .pilot import (
    BOUNDARY_QUANTILES,
    DEFAULT_FAIR_PANEL_PATH,
    DEFAULT_SNAPSHOT_PATH,
    ENTRY_ROUTES,
    load_adaptive_snapshot,
    load_causal_fair_state,
    load_spot_feature_state,
)
from .raw_tape import RawTapeDay, load_raw_tape_day
from .targets import (
    ROUTE_SPECS,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    price_in_ref_band,
    target_price_for_basis,
)


SESSION_START = time(9, 5)
SESSION_END = time(13, 20)


@dataclass(frozen=True)
class MergedTargetStudyInput:
    date: str
    mapping: pl.DataFrame
    raw_tape: RawTapeDay
    observations: pl.DataFrame
    audit: pl.DataFrame


@dataclass
class _MarketState:
    row: dict[str, object] | None = None
    formal_after_trial: bool = False
    last_trial_match: bool | None = None

    def update(self, row: dict[str, object]) -> None:
        trial = bool(row["trial_match"])
        has_raw_book = bool(row["raw_has_book"])
        if trial:
            self.formal_after_trial = False
        elif self.last_trial_match is True:
            self.formal_after_trial = has_raw_book
        elif not self.formal_after_trial and has_raw_book:
            self.formal_after_trial = True
        self.last_trial_match = trial
        self.row = row


@dataclass(frozen=True)
class _Event:
    cursor: EventCursor
    source: Literal["anchor", "future", "spot"]
    row: dict[str, object]


def load_replay_mapping(
    date: str,
    symbols: Iterable[str],
    *,
    fair_panel_path: Path = DEFAULT_FAIR_PANEL_PATH,
) -> pl.DataFrame:
    """Recover the exact already-researched contract pairs for one day."""

    symbols = sorted({str(value) for value in symbols})
    required = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
    ]
    mapping = (
        pl.scan_parquet(fair_panel_path)
        .filter(
            (pl.col("Date").cast(pl.String) == str(date))
            & pl.col("ValueCode").cast(pl.String).is_in(symbols)
        )
        .select(required)
        .group_by("Date", "ValueCode", "QuoteCode")
        .agg(
            pl.col("spot_ref_price").drop_nulls().first(),
            pl.col("fut_ref_price").drop_nulls().first(),
            pl.col("contract_size").drop_nulls().first(),
        )
        .filter(
            pl.col("spot_ref_price").is_not_null()
            & pl.col("fut_ref_price").is_not_null()
            & pl.col("contract_size").is_not_null()
        )
        .collect()
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
        )
    )
    if mapping.is_empty():
        raise ValueError(f"{date}: no exact fair-panel mappings")
    duplicate = mapping.group_by("ValueCode").len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError(f"{date}: mapping is not one-to-one: {duplicate.to_dicts()}")
    return mapping.drop("Date").sort(["ValueCode", "QuoteCode"])


def build_merged_target_input(
    date: str,
    symbols: Iterable[str],
    *,
    fair_panel_path: Path = DEFAULT_FAIR_PANEL_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    data_root: Path = HFT_DATA_ROOT,
    quantiles: Iterable[int] = BOUNDARY_QUANTILES,
) -> MergedTargetStudyInput:
    """Load one day and construct sparse raw target/gate observations."""

    symbols = sorted({str(value) for value in symbols})
    mapping = load_replay_mapping(
        date, symbols, fair_panel_path=fair_panel_path
    )
    tape = load_raw_tape_day(date, mapping)
    feature_state = load_spot_feature_state(date, symbols, data_root=data_root)
    fair = load_causal_fair_state(
        date, symbols, fair_panel_path=fair_panel_path
    )
    snapshot = load_adaptive_snapshot(
        date,
        symbols,
        snapshot_path=snapshot_path,
        quantiles=quantiles,
    )
    observations, audit = build_merged_target_observations_from_frames(
        date,
        tape.spot_states,
        tape.future_states,
        feature_state,
        fair,
        snapshot,
    )
    return MergedTargetStudyInput(date, mapping, tape, observations, audit)


def build_merged_target_observations_from_frames(
    date: str,
    spot_states: pl.DataFrame,
    future_states: pl.DataFrame,
    spot_feature_state: pl.DataFrame,
    fair_state: pl.DataFrame,
    adaptive_snapshot: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pure frame-level merged-state builder used by tests and the runner."""

    clock_full = _attach_spread_clock(spot_states, spot_feature_state)
    clock = _thin_market_events(clock_full, include_base_epoch=True)
    future_events = _thin_market_events(
        future_states, include_base_epoch=False
    )
    boundary_by_product = _boundary_lookup(adaptive_snapshot)
    fair_by_product = {
        str(value_code): group.sort("fair_timestamp")
        for (value_code,), group in fair_state.group_by("ValueCode")
    }
    spot_by_product = {
        str(value_code): group.sort(["recv_time", "sequence", "packet_sequence"])
        for (value_code,), group in clock.group_by("ValueCode")
    }
    future_by_product = {
        str(value_code): group.sort(["recv_time", "sequence", "packet_sequence"])
        for (value_code,), group in future_events.group_by("ValueCode")
    }
    spot_full_by_product = {
        str(value_code): group.sort(["recv_time", "sequence", "packet_sequence"])
        for (value_code,), group in clock_full.group_by("ValueCode")
    }
    future_full_by_product = {
        str(value_code): group.sort(["recv_time", "sequence", "packet_sequence"])
        for (value_code,), group in future_states.group_by("ValueCode")
    }
    value_codes = sorted(
        set(spot_by_product)
        & set(future_by_product)
        & set(spot_full_by_product)
        & set(future_full_by_product)
    )
    records: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for value_code in value_codes:
        product_records, product_audit = _build_product_observations(
            str(date),
            value_code,
            spot_by_product[value_code],
            future_by_product[value_code],
            fair_by_product.get(value_code, pl.DataFrame()),
            boundary_by_product.get(value_code, ()),
        )
        _refresh_maker_rank_and_queue_from_full_state(
            product_records,
            spot_full_by_product[value_code],
            future_full_by_product[value_code],
        )
        records.extend(product_records)
        audit_rows.append(product_audit)

    observations = (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame()
    )
    if not observations.is_empty():
        observations = observations.sort(
            [
                "Date",
                "ValueCode",
                "boundary_quantile",
                "route",
                "recv_time_ns",
                "cursor_event_sequence",
                "cursor_row_index",
            ]
        )
    return observations, pl.from_dicts(audit_rows, infer_schema_length=None)


def observations_for_policy(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    route: str,
    boundary_quantile: int,
) -> tuple[TargetObservation, ...]:
    """Convert one sparse policy slice to the pure layered engine input."""

    # A mapped product-day can legitimately have no causal anchor at all
    # (for example after an upstream eligibility outage).  The replay still
    # needs a zero-order support row for every route/q, not a schema error.
    if frame.is_empty():
        return ()

    selected = frame.filter(
        (pl.col("Date") == str(date))
        & (pl.col("ValueCode") == str(value_code))
        & (pl.col("route") == route)
        & (pl.col("boundary_quantile") == int(boundary_quantile))
    ).sort(["recv_time_ns", "cursor_event_sequence", "cursor_row_index"])
    return tuple(
        TargetObservation(
            cursor=EventCursor(
                int(row["recv_time_ns"]),
                int(row["cursor_event_sequence"]),
                int(row["cursor_row_index"]),
            ),
            spread_pair_epoch=int(row["spread_pair_epoch"]),
            absolute_target_tick=(
                int(row["absolute_target_tick"])
                if row["absolute_target_tick"] is not None
                else None
            ),
            gate_open=bool(row["gate_open"]),
            gate_reason=str(row["admission_reason"]),
            initial_queue_ahead=(
                int(row["initial_queue_ahead"])
                if row["initial_queue_ahead"] is not None
                else None
            ),
            target_rank=(
                str(row["target_rank"])
                if row["target_rank"] is not None
                else None
            ),
            source=str(row["event_source"]),
        )
        for row in selected.iter_rows(named=True)
    )


def session_cutoff_cursor(date: str) -> EventCursor:
    parsed = datetime.strptime(str(date), "%Y%m%d")
    local = datetime.combine(parsed.date(), SESSION_END)
    utc = local - timedelta(hours=8)
    return EventCursor(_datetime_ns(utc), 3, 0)


def _attach_spread_clock(
    spot_states: pl.DataFrame,
    feature_state: pl.DataFrame,
) -> pl.DataFrame:
    features = feature_state.select(
        "Date",
        "ValueCode",
        pl.col("spot_channel_seq").alias("sequence"),
        "spread_pair_id",
        "spread_pair_seq",
        "spread_pair_epoch",
        "spread_count_at_same_count",
    ).sort(["Date", "ValueCode", "sequence"])
    features = features.with_columns(
        pl.col("spread_pair_epoch")
        .shift(1)
        .over(["Date", "ValueCode"])
        .alias("_previous_epoch")
    ).with_columns(
        (
            (pl.col("spread_pair_id") > 0)
            & pl.col("spread_pair_epoch").is_not_null()
            & pl.col("_previous_epoch").is_not_null()
            & (pl.col("spread_pair_epoch") != pl.col("_previous_epoch"))
        ).alias("base_epoch_transition")
    ).drop("_previous_epoch")
    joined = spot_states.join(
        features,
        on=["Date", "ValueCode", "sequence"],
        how="left",
        validate="1:1",
    )
    missing = joined.filter(pl.col("spread_pair_epoch").is_null())
    if missing.height:
        raise ValueError(
            "spot raw/feature clock join is incomplete: "
            f"{missing.select('Date', 'ValueCode', 'sequence').head(5).to_dicts()}"
        )
    return joined


def _thin_market_events(
    frame: pl.DataFrame,
    *,
    include_base_epoch: bool,
) -> pl.DataFrame:
    """Keep only updates capable of changing a target or safety gate.

    Positive-to-positive depth changes do not change the quote action and are
    rejoined as-of when a generation is emitted.  This reduction is essential
    for liquid products such as 2303, whose raw spot tape has hundreds of
    thousands of lots-only rows.
    """

    group = ["Date", "ValueCode"]
    signature_columns = [
        "trial_match",
        "bid_price_1",
        "ask_price_1",
        "exec_bid_price",
        "exec_ask_price",
        "ref_price",
    ]
    working = frame.sort(
        ["Date", "ValueCode", "recv_time", "sequence", "packet_sequence"]
    ).with_columns(
        *[
            (
                pl.col(column).cast(pl.String).fill_null("__NULL__")
                != pl.col(column)
                .cast(pl.String)
                .fill_null("__NULL__")
                .shift(1)
            )
            .over(group)
            .fill_null(True)
            .alias(f"_changed_{column}")
            for column in signature_columns
        ],
        *[
            (
                (pl.col(column).fill_null(0) > 0)
                != (pl.col(column).fill_null(0).shift(1) > 0)
            )
            .over(group)
            .fill_null(True)
            .alias(f"_changed_{column}_positive")
            for column in (
                "bid_lots_1",
                "ask_lots_1",
                "exec_bid_lots",
                "exec_ask_lots",
            )
        ],
        pl.col("trial_match")
        .shift(1)
        .over(group)
        .fill_null(False)
        .alias("_previous_trial"),
        pl.col("trial_match")
        .cast(pl.Int64)
        .cum_sum()
        .over(group)
        .alias("_trial_epoch"),
    ).with_columns(
        pl.when((~pl.col("trial_match")) & pl.col("raw_has_book"))
        .then(1)
        .otherwise(0)
        .cum_sum()
        .over(group + ["_trial_epoch"])
        .alias("_formal_raw_book_ordinal")
    )
    change_columns = [
        name for name in working.columns if name.startswith("_changed_")
    ]
    keep = pl.any_horizontal(*[pl.col(name) for name in change_columns])
    # Retain the first formal book when a trial row is immediately followed by
    # formal matching even if the price itself did not change.
    keep = keep | (pl.col("_previous_trial") & pl.col("raw_has_book"))
    # A zero-book formal row can immediately follow TrialMatch while carrying
    # forward the old normalized prices/lots.  Retain the first *actual* raw
    # formal book anywhere after the latest trial epoch so the formal-book gate
    # can reopen without needing an unrelated price or depth-sign change.
    keep = keep | (
        (pl.col("_trial_epoch") > 0)
        & (~pl.col("trial_match"))
        & pl.col("raw_has_book")
        & (pl.col("_formal_raw_book_ordinal") == 1)
    )
    if include_base_epoch:
        keep = keep | pl.col("base_epoch_transition").fill_null(False)
    return working.filter(keep).drop(
        change_columns
        + ["_previous_trial", "_trial_epoch", "_formal_raw_book_ordinal"]
    )


def _refresh_maker_rank_and_queue_from_full_state(
    records: list[dict[str, object]],
    spot_full: pl.DataFrame,
    future_full: pl.DataFrame,
) -> None:
    """Replace sparse-state queue labels with exact causal full-tape labels.

    Target-event thinning intentionally removes positive-to-positive lots-only
    updates.  Those updates must not create a new quote generation, but they do
    change how much displayed quantity is in front of an order submitted later
    because of an anchor or opposite-market event.  For each observation we
    therefore look up its maker state on the full raw tape using the same total
    cursor order as ``_merged_events`` and recompute both rank and queue.
    """

    frames = {"spot": spot_full, "future": future_full}
    priorities = {"spot": 2, "future": 1}
    for market, maker in frames.items():
        selected = [
            record for record in records if record["maker_market"] == market
        ]
        if not selected:
            continue
        maker = maker.sort(
            ["recv_time_ns", "sequence", "packet_sequence"]
        )
        recv_times = [int(value) for value in maker["recv_time_ns"].to_list()]
        sequences = [int(value) for value in maker["sequence"].to_list()]
        row_cache: dict[int, dict[str, object]] = {}
        for record in selected:
            target = record.get("absolute_target_price")
            if not _finite(target):
                record["target_rank"] = None
                record["initial_queue_ahead"] = None
                continue
            cursor = EventCursor(
                int(record["recv_time_ns"]),
                int(record["cursor_event_sequence"]),
                int(record["cursor_row_index"]),
            )
            row_index = _latest_market_state_index(
                recv_times,
                sequences,
                maker_priority=priorities[market],
                observation=cursor,
            )
            if row_index is None:
                record["target_rank"] = None
                record["initial_queue_ahead"] = None
                continue
            row = row_cache.get(row_index)
            if row is None:
                row = maker.row(row_index, named=True)
                row_cache[row_index] = row
            rank, queue = _rank_and_queue(
                float(target),
                str(record["maker_side"]),
                row,
            )
            record["target_rank"] = rank
            record["initial_queue_ahead"] = queue


def _latest_market_state_index(
    recv_times: list[int],
    sequences: list[int],
    *,
    maker_priority: int,
    observation: EventCursor,
) -> int | None:
    """Return the last maker row not later than an observation cursor."""

    if not recv_times:
        return None
    left = bisect_left(recv_times, observation.recv_time_ns)
    right = bisect_right(recv_times, observation.recv_time_ns)
    if maker_priority < observation.event_sequence:
        index = right - 1
    elif maker_priority > observation.event_sequence:
        index = left - 1
    else:
        index = bisect_right(
            sequences,
            observation.row_index,
            left,
            right,
        ) - 1
        if index < left:
            index = left - 1
    return index if index >= 0 else None


def _boundary_lookup(
    snapshot: pl.DataFrame,
) -> dict[str, tuple[dict[str, object], ...]]:
    return {
        str(value_code): tuple(group.sort("boundary_quantile").iter_rows(named=True))
        for (value_code,), group in snapshot.group_by("ValueCode")
    }


def _build_product_observations(
    date: str,
    value_code: str,
    spot: pl.DataFrame,
    future: pl.DataFrame,
    fair: pl.DataFrame,
    boundaries: tuple[dict[str, object], ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    spot_market = _MarketState()
    future_market = _MarketState()
    current_anchor: float | None = None
    current_epoch: int | None = None
    activated = False
    last_signature: dict[tuple[int, str], tuple[object, ...]] = {}
    records: list[dict[str, object]] = []
    raw_events = anchor_events = 0
    state_observations = base_transitions = 0

    events = _merged_events(spot, future, fair)
    for event in events:
        raw_events += event.source != "anchor"
        anchor_events += event.source == "anchor"
        base_transition = False
        if event.source == "spot":
            spot_market.update(event.row)
            current_epoch = int(event.row["spread_pair_epoch"])
            base_transition = bool(event.row["base_epoch_transition"])
            if base_transition:
                activated = True
                base_transitions += 1
        elif event.source == "future":
            future_market.update(event.row)
        else:
            anchor = event.row.get("anchor_ewma_120s_bp")
            current_anchor = (
                float(anchor) if _finite(anchor) else None
            )

        if not activated or current_epoch is None:
            continue
        for boundary in boundaries:
            quantile = int(boundary["boundary_quantile"])
            for route in ENTRY_ROUTES:
                target = _route_state(
                    route,
                    current_anchor,
                    boundary,
                    spot_market,
                    future_market,
                    session_date=date,
                )
                key = (quantile, route)
                signature = (
                    current_epoch,
                    target["absolute_target_tick"],
                    target["gate_open"],
                    target["admission_reason"],
                )
                if not base_transition and signature == last_signature.get(key):
                    continue
                last_signature[key] = signature
                records.append(
                    {
                        "Date": date,
                        "ValueCode": value_code,
                        "QuoteCode": boundary["QuoteCode"],
                        "boundary_quantile": quantile,
                        "boundary_role": boundary.get("boundary_role"),
                        "upper_distance_bp": boundary.get("upper_distance_bp"),
                        "parameter_version": boundary.get("parameter_version"),
                        "source_asof_date": boundary.get("source_asof_date"),
                        "route": route,
                        "stage": ROUTE_SPECS[route].stage,
                        "maker_market": ROUTE_SPECS[route].maker_market,
                        "maker_side": ROUTE_SPECS[route].maker_side,
                        "spread_pair_epoch": current_epoch,
                        "base_epoch_transition": base_transition,
                        "event_source": event.source,
                        "recv_time_ns": event.cursor.recv_time_ns,
                        "cursor_event_sequence": event.cursor.event_sequence,
                        "cursor_row_index": event.cursor.row_index,
                        "anchor_ewma_120s_bp": current_anchor,
                        **target,
                    }
                )
                state_observations += 1
    return records, {
        "Date": date,
        "ValueCode": value_code,
        "raw_market_events": raw_events,
        "anchor_events": anchor_events,
        "base_epoch_transitions": base_transitions,
        "sparse_policy_observations": state_observations,
    }


def _merged_events(
    spot: pl.DataFrame,
    future: pl.DataFrame,
    fair: pl.DataFrame,
) -> Iterator[_Event]:
    sources = (
        _event_iterator(fair, "anchor", 0),
        _event_iterator(future, "future", 1),
        _event_iterator(spot, "spot", 2),
    )
    return heapq.merge(*sources, key=lambda event: event.cursor)


def _event_iterator(
    frame: pl.DataFrame,
    source: Literal["anchor", "future", "spot"],
    priority: int,
) -> Iterator[_Event]:
    if frame.is_empty():
        return
    for index, row in enumerate(frame.iter_rows(named=True)):
        if source == "anchor":
            timestamp = row["fair_timestamp"]
            recv_ns = _datetime_ns(timestamp)
            row_index = index
        else:
            recv_ns = int(row["recv_time_ns"])
            row_index = int(row["sequence"])
        yield _Event(EventCursor(recv_ns, priority, row_index), source, row)


def _route_state(
    route: str,
    anchor: float | None,
    boundary: dict[str, object],
    spot_state: _MarketState,
    future_state: _MarketState,
    *,
    session_date: str | None = None,
) -> dict[str, object]:
    spot = spot_state.row
    future = future_state.row
    upper = boundary.get("upper_distance_bp")
    threshold = (
        anchor + float(upper)
        if anchor is not None and _finite(upper)
        else None
    )
    gates: list[tuple[bool, str]] = [
        (bool(boundary.get("adaptive_parameter_valid") is True), "invalid_adaptive_parameter"),
        (anchor is not None, "missing_anchor"),
        (spot is not None, "missing_spot_state"),
        (future is not None, "missing_future_state"),
    ]
    if spot is not None:
        gates.extend(
            [
                (not bool(spot["trial_match"]), "spot_trial_match"),
                (spot_state.formal_after_trial, "spot_formal_book_gate"),
                (_book_ok(spot), "spot_book_gate"),
                (_ref_ok(spot, include_exec=False), "spot_ref_gate"),
            ]
        )
    if future is not None:
        gates.extend(
            [
                (not bool(future["trial_match"]), "future_trial_match"),
                (future_state.formal_after_trial, "future_formal_book_gate"),
                (_book_ok(future), "future_book_gate"),
                (_exec_book_ok(future), "future_exec_book_gate"),
                (_ref_ok(future, include_exec=True), "future_ref_gate"),
            ]
        )

    target_price: float | None = None
    target_tick: int | None = None
    effective: float | None = None
    passive = False
    target_ref_ok = False
    target_rank: str | None = None
    queue_ahead: int | None = None
    if threshold is not None and spot is not None and future is not None:
        try:
            target_price = target_price_for_basis(
                route,
                threshold,
                session_date=session_date,
                spot_ask=float(spot["exec_ask_price"]),
                fut_exec_bid=float(future["exec_bid_price"]),
            )
            target_tick = absolute_price_tick(
                target_price,
                market=ROUTE_SPECS[route].maker_market,
                session_date=session_date,
            )
            effective = effective_basis_bp(
                route,
                target_price,
                spot_ask=float(spot["exec_ask_price"]),
                fut_exec_bid=float(future["exec_bid_price"]),
            )
            passive = is_passive_target(
                route,
                target_price,
                spot_ask=float(spot["exec_ask_price"]),
                fut_exec_bid=float(future["exec_bid_price"]),
            )
            maker = future if ROUTE_SPECS[route].maker_market == "future" else spot
            target_ref_ok = price_in_ref_band(
                target_price, float(maker["ref_price"])
            )
            target_rank, queue_ahead = _rank_and_queue(
                target_price,
                ROUTE_SPECS[route].maker_side,
                maker,
            )
        except (TypeError, ValueError):
            target_price = target_tick = effective = None

    gates.extend(
        [
            (target_price is not None, "target_unavailable"),
            (passive, "target_not_passive"),
            (target_ref_ok, "target_ref_gate"),
        ]
    )
    gate_open = all(valid for valid, _ in gates)
    reason = "admitted" if gate_open else next(
        reason for valid, reason in gates if not valid
    )
    return {
        "threshold_basis_bp": threshold,
        "absolute_target_price": target_price,
        "absolute_target_tick": target_tick,
        "effective_basis_bp": effective,
        "target_rank": target_rank,
        "initial_queue_ahead": queue_ahead,
        "gate_open": gate_open,
        "admission_reason": reason,
        "spot_book_age_ms": _book_age_ms(spot),
        "future_book_age_ms": _book_age_ms(future),
    }


def _book_ok(row: dict[str, object]) -> bool:
    values = (
        row.get("bid_price_1"),
        row.get("ask_price_1"),
        row.get("bid_lots_1"),
        row.get("ask_lots_1"),
    )
    return all(_positive(value) for value in values) and float(
        row["bid_price_1"]
    ) <= float(row["ask_price_1"])


def _exec_book_ok(row: dict[str, object]) -> bool:
    values = (
        row.get("exec_bid_price"),
        row.get("exec_ask_price"),
        row.get("exec_bid_lots"),
        row.get("exec_ask_lots"),
    )
    return all(_positive(value) for value in values) and float(
        row["exec_bid_price"]
    ) <= float(row["exec_ask_price"])


def _ref_ok(row: dict[str, object], *, include_exec: bool) -> bool:
    reference = row.get("ref_price")
    if not _positive(reference):
        return False
    fields = ["bid_price_1", "ask_price_1"]
    if include_exec:
        fields.extend(["exec_bid_price", "exec_ask_price"])
    return all(
        _positive(row.get(field))
        and price_in_ref_band(float(row[field]), float(reference))
        for field in fields
    )


def _rank_and_queue(
    target: float,
    side: Literal["bid", "ask"],
    row: dict[str, object],
) -> tuple[str, int | None]:
    candidates: list[int] = []
    for level in range(1, 6):
        price = row.get(f"{side}_price_{level}")
        lots = row.get(f"{side}_lots_{level}")
        if _positive(price) and math.isclose(target, float(price), abs_tol=1e-8):
            if _positive(lots):
                candidates.append(int(lots))
    best_price = row.get(f"best_{side}_price")
    best_lots = row.get(f"best_{side}_lots")
    if _positive(best_price) and math.isclose(
        target, float(best_price), abs_tol=1e-8
    ) and _positive(best_lots):
        candidates.append(int(best_lots))
    if candidates:
        visible = [
            row.get(f"{side}_price_{level}") for level in range(1, 6)
        ]
        level = next(
            (
                index
                for index, value in enumerate(visible, 1)
                if _positive(value)
                and math.isclose(target, float(value), abs_tol=1e-8)
            ),
            1,
        )
        return (f"{side.upper()}{level}", max(candidates))

    bid = row.get("exec_bid_price")
    ask = row.get("exec_ask_price")
    if _positive(bid) and _positive(ask) and float(bid) < target < float(ask):
        return "inside", 0
    return "behind_visible", None


def _book_age_ms(row: dict[str, object] | None) -> float | None:
    if row is None or row.get("book_recv_time_ns") is None:
        return None
    return (
        int(row["recv_time_ns"]) - int(row["book_recv_time_ns"])
    ) / 1_000_000.0


def _positive(value: object) -> bool:
    return _finite(value) and float(value) > 0


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _datetime_ns(value: object) -> int:
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be datetime")
    epoch = datetime(1970, 1, 1)
    return int((value - epoch).total_seconds() * 1_000_000_000)
