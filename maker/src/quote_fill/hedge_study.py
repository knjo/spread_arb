"""WP03 raw-tape delayed hedge and spot partial-fill study.

This module is the data-frame orchestration layer around :mod:`.hedge`.  It
deliberately consumes the *raw-fact* identity emitted by WP02 so coincident
p50/p80 policy aliases do not duplicate a physical maker fill.

Quantity conventions are explicit throughout:

* ``future_ask_spot_taker``: one filled futures contract is hedged by buying
  two spot board lots after 50 ms;
* ``spot_bid_future_taker``: two filled spot board lots are hedged by selling
  one futures contract after 50 ms;
* one spot board lot is assumed to be 1,000 shares and the selected stock
  futures contract is required to represent 2,000 shares.

The opposite-market state is selected causally.  In the merged tape, futures
events have priority 1 and spot events priority 2.  Consequently a spot event
with the same receive timestamp as a futures maker fill is after that fill and
is excluded from the arrival reference; a futures event at the timestamp of a
spot maker fill is included.  At the 50 ms deadline all events at the deadline
timestamp are observable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Sequence

import polars as pl

from .hedge import (
    DEFAULT_HEDGE_DELAY_NS,
    BookLevel,
    MakerFillHedgeRequest,
    OppositeBookSnapshot,
    SpotMakerFillEvent,
    label_delayed_taker_hedge,
    label_spot_partial_fill_horizons,
)
from .layered import EventCursor
from .raw_tape import RawTapeDay


FUTURE_ASK_ROUTE = "future_ask_spot_taker"
SPOT_BID_ROUTE = "spot_bid_future_taker"
SUPPORTED_ENTRY_ROUTES = (FUTURE_ASK_ROUTE, SPOT_BID_ROUTE)

SPOT_LOT_SHARES = 1_000
CONTRACT_SHARES = 2_000
SPOT_HEDGE_LOTS = 2
FUTURE_HEDGE_CONTRACTS = 1
PARTIAL_HORIZONS_MS: tuple[int, ...] = (50, 250, 1_000, 2_000, 5_000)

_MARKET_PRIORITY = {"future": 1, "spot": 2}
_FILL_CURSOR_COLUMNS = (
    "full_fill_recv_time_ns",
    "full_fill_event_sequence",
    "full_fill_row_index",
)
_QUANTITY_PATH_COLUMNS = (
    "raw_order_fact_id",
    "fill_recv_time_ns",
    "fill_event_sequence",
    "fill_row_index",
    "fill_quantity",
)


@dataclass(frozen=True)
class HedgeStudyResult:
    """Materialized WP03 labels and aggregate diagnostics."""

    hedge_facts: pl.DataFrame
    hedge_by_day_symbol: pl.DataFrame
    hedge_summary: pl.DataFrame
    spot_partial_horizons: pl.DataFrame
    spot_partial_summary: pl.DataFrame
    audit: pl.DataFrame


@dataclass(frozen=True)
class _HedgeFactRequest:
    raw_order_fact_id: str
    date: str
    value_code: str
    quote_code: str
    route: str
    fill_cursor: EventCursor
    maker_fill_quantity: int
    full_fill_boundary_quantiles: tuple[int, ...]
    full_fill_policy_alias_count: int
    all_policy_alias_count: int

    @property
    def maker_market(self) -> Literal["future", "spot"]:
        return "future" if self.route == FUTURE_ASK_ROUTE else "spot"

    @property
    def opposite_market(self) -> Literal["future", "spot"]:
        return "spot" if self.route == FUTURE_ASK_ROUTE else "future"

    @property
    def hedge_side(self) -> Literal["buy", "sell"]:
        return "buy" if self.route == FUTURE_ASK_ROUTE else "sell"

    @property
    def hedge_quantity(self) -> int:
        return (
            SPOT_HEDGE_LOTS
            if self.route == FUTURE_ASK_ROUTE
            else FUTURE_HEDGE_CONTRACTS
        )


@dataclass(frozen=True)
class _StateHit:
    snapshot: OppositeBookSnapshot
    state_recv_time_ns: int
    book_recv_time_ns: int | None


class _RawStateIndex:
    """Compact as-of index over one product-market raw state frame."""

    def __init__(self, states: pl.DataFrame, market: Literal["spot", "future"]):
        _require_columns(
            states,
            {
                "recv_time_ns",
                "sequence",
                "trial_match",
                "ref_price",
                "book_state_available",
                "book_recv_time_ns",
                *(f"{side}_price_{level}" for side in ("bid", "ask") for level in range(1, 6)),
                *(f"{side}_lots_{level}" for side in ("bid", "ask") for level in range(1, 6)),
            },
            f"raw {market} states",
        )
        sort_columns = ["recv_time_ns", "sequence"]
        if "packet_sequence" in states.columns:
            sort_columns.append("packet_sequence")
        self.market = market
        self.states = states.sort(sort_columns)
        self.recv_times = self.states.get_column("recv_time_ns")

    def latest(self, query_time_ns: int) -> _StateHit | None:
        """Return the last raw market state with receive time <= query."""

        position = self.recv_times.search_sorted(int(query_time_ns), side="right") - 1
        if position < 0:
            return None
        row = self.states.row(position, named=True)
        snapshot = _snapshot_from_state(row, self.market)
        book_ns = _optional_int(row.get("book_recv_time_ns"))
        return _StateHit(snapshot, int(row["recv_time_ns"]), book_ns)


def run_hedge_study(
    order_aliases: pl.DataFrame,
    raw_order_facts: pl.DataFrame,
    raw_tapes: RawTapeDay | Iterable[RawTapeDay],
    *,
    quantity_paths: pl.DataFrame | None = None,
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS,
    output_dir: Path | None = None,
) -> HedgeStudyResult:
    """Label unique WP02 full fills against exact raw opposite books.

    ``order_aliases`` must carry the complete full-fill ``EventCursor`` in
    ``full_fill_recv_time_ns``, ``full_fill_event_sequence`` and
    ``full_fill_row_index``.  Receive time alone is intentionally rejected:
    at an equal timestamp the futures/spot source priority determines whether
    an opposite-market update is observable.

    Optional ``quantity_paths`` is a long table of known own spot fills with
    the five columns in :data:`_QUANTITY_PATH_COLUMNS`.  When absent, partial
    fill outputs are empty rather than inferred from aggregate quantities.
    """

    if isinstance(raw_tapes, RawTapeDay):
        tapes = (raw_tapes,)
    else:
        tapes = tuple(raw_tapes)
    if not tapes:
        raise ValueError("raw_tapes cannot be empty")
    if isinstance(hedge_delay_ns, bool) or not isinstance(hedge_delay_ns, int) or hedge_delay_ns < 0:
        raise ValueError("hedge_delay_ns must be a non-negative integer")
    dates = [str(tape.date) for tape in tapes]
    if len(dates) != len(set(dates)):
        raise ValueError("raw_tapes must contain at most one RawTapeDay per date")

    requests = _prepare_hedge_requests(order_aliases, raw_order_facts)
    request_dates = {request.date for request in requests}
    missing_dates = sorted(request_dates - set(dates))
    if missing_dates:
        raise ValueError(f"missing RawTapeDay for full-fill dates: {missing_dates}")

    fact_records: list[dict[str, object]] = []
    tape_by_date = {str(tape.date): tape for tape in tapes}
    index_cache: dict[tuple[str, str, str], _RawStateIndex] = {}
    contract_cache: dict[tuple[str, str], float] = {}
    for request in requests:
        tape = tape_by_date[request.date]
        contract_key = (request.date, request.value_code)
        if contract_key not in contract_cache:
            contract_cache[contract_key] = _contract_size_for_request(request, tape)
        index_key = (
            request.date,
            request.value_code,
            request.opposite_market,
        )
        if index_key not in index_cache:
            states = (
                tape.spot_states
                if request.opposite_market == "spot"
                else tape.future_states
            ).filter(pl.col("ValueCode").cast(pl.String) == request.value_code)
            if states.is_empty():
                raise ValueError(
                    f"missing {request.opposite_market} raw states for "
                    f"{request.date}/{request.value_code}"
                )
            index_cache[index_key] = _RawStateIndex(
                states, request.opposite_market
            )
        fact_records.append(
            _label_one_request(
                request,
                index_cache[index_key],
                contract_size=contract_cache[contract_key],
                hedge_delay_ns=hedge_delay_ns,
            )
        )
    hedge_facts = _from_records(fact_records, _hedge_fact_schema())
    by_day = summarize_hedges(hedge_facts, ["Date", "ValueCode", "route"])
    summary = summarize_hedges(hedge_facts, ["ValueCode", "route"])

    partial_horizons = build_spot_partial_diagnostics(
        order_aliases,
        quantity_paths,
        horizons_ms=PARTIAL_HORIZONS_MS,
    )
    partial_summary = summarize_spot_partial(partial_horizons)
    audit = _build_audit(
        order_aliases,
        raw_order_facts,
        requests,
        hedge_facts,
        partial_horizons,
        quantity_paths,
        tapes,
        hedge_delay_ns,
    )
    result = HedgeStudyResult(
        hedge_facts,
        by_day,
        summary,
        partial_horizons,
        partial_summary,
        audit,
    )
    if output_dir is not None:
        write_hedge_study(result, Path(output_dir), hedge_delay_ns=hedge_delay_ns)
    return result


def executable_levels_from_state(
    row: dict[str, object],
    market: Literal["spot", "future"],
    side: Literal["bid", "ask"],
) -> tuple[BookLevel, ...]:
    """Merge/deduplicate the executable Best and L1--L5 price levels.

    Futures ``Best*`` can sit inside the regular L1.  A same-price Best/L1
    duplicate represents alternative views of the same displayed queue and is
    merged with ``max(quantity)``, never summed.  At most the best five unique
    prices are retained.  Spot uses the native L1--L5 ladder only.
    """

    if market not in ("spot", "future"):
        raise ValueError("market must be 'spot' or 'future'")
    if side not in ("bid", "ask"):
        raise ValueError("side must be 'bid' or 'ask'")
    candidates: list[tuple[float, int]] = []
    if market == "future":
        _append_level(
            candidates,
            row.get(f"best_{side}_price"),
            row.get(f"best_{side}_lots"),
        )
    for level in range(1, 6):
        _append_level(
            candidates,
            row.get(f"{side}_price_{level}"),
            row.get(f"{side}_lots_{level}"),
        )

    # Normalized prices are stable floats.  Rounding the dictionary key avoids
    # a feed representation artifact creating two economically equal levels.
    by_price: dict[float, tuple[float, int]] = {}
    for price, quantity in candidates:
        key = round(price, 8)
        previous = by_price.get(key)
        if previous is None or quantity > previous[1]:
            by_price[key] = (price, quantity)
    ordered = sorted(
        by_price.values(), key=lambda item: item[0], reverse=side == "bid"
    )[:5]
    return tuple(BookLevel(price, quantity) for price, quantity in ordered)


def build_spot_partial_diagnostics(
    order_aliases: pl.DataFrame,
    quantity_paths: pl.DataFrame | None,
    *,
    horizons_ms: Sequence[int] = PARTIAL_HORIZONS_MS,
) -> pl.DataFrame:
    """Label time-to-two-lot paths for unique spot-maker raw facts.

    ``quantity_paths`` is intentionally required for non-empty output.  An
    aggregate ``known_filled_quantity`` does not identify when the second lot
    arrived and is therefore not expanded into a fictional path.
    """

    if quantity_paths is None or quantity_paths.is_empty():
        return pl.DataFrame(schema=_partial_schema())
    _require_columns(quantity_paths, set(_QUANTITY_PATH_COLUMNS), "quantity paths")
    _require_columns(
        order_aliases,
        {"raw_order_fact_id", "Date", "ValueCode", "QuoteCode", "route"},
        "order aliases",
    )
    horizons = tuple(int(value) for value in horizons_ms)
    if not horizons or any(value < 0 for value in horizons):
        raise ValueError("horizons_ms must contain non-negative values")
    if any(right <= left for left, right in zip(horizons, horizons[1:])):
        raise ValueError("horizons_ms must be strictly increasing and unique")

    spot_aliases = order_aliases.filter(pl.col("route") == SPOT_BID_ROUTE)
    metadata = _representative_metadata(spot_aliases)
    records: list[dict[str, object]] = []
    paths = quantity_paths.filter(
        pl.col("raw_order_fact_id").cast(pl.String).is_in(list(metadata))
    ).unique(subset=list(_QUANTITY_PATH_COLUMNS), keep="first")
    for raw_id, frame in _partition_by_string(paths, "raw_order_fact_id"):
        if raw_id not in metadata:
            continue
        sort_columns = [
            "fill_recv_time_ns",
            "fill_event_sequence",
            "fill_row_index",
        ]
        frame = frame.sort(sort_columns)
        fills: list[SpotMakerFillEvent] = []
        for row in frame.iter_rows(named=True):
            quantity = _required_positive_int(row.get("fill_quantity"), "fill_quantity")
            trial = bool(row.get("trial_match", False))
            gate_open = bool(row.get("gate_open", True))
            gate_reason = row.get("gate_reason")
            if not gate_open and not gate_reason:
                gate_reason = "quantity_path_gate_closed"
            fills.append(
                SpotMakerFillEvent(
                    _event_cursor_from_path(row),
                    quantity,
                    trial_match=trial,
                    gate_open=gate_open,
                    gate_reason=str(gate_reason) if gate_reason is not None else None,
                )
            )
        if not fills:
            continue
        labels = label_spot_partial_fill_horizons(
            raw_id,
            fills[0].cursor,
            fills,
            futures_equivalent_quantity=SPOT_HEDGE_LOTS,
            wait_horizons_ns=tuple(value * 1_000_000 for value in horizons),
        )
        meta = metadata[raw_id]
        for label in labels:
            completion = label.completion_cursor
            records.append(
                {
                    "Date": meta["Date"],
                    "ValueCode": meta["ValueCode"],
                    "QuoteCode": meta["QuoteCode"],
                    "route": SPOT_BID_ROUTE,
                    "raw_order_fact_id": raw_id,
                    "horizon_ms": label.horizon_ns // 1_000_000,
                    "status": label.status,
                    "spot_fill_quantity": label.cumulative_fill_quantity,
                    "spot_fill_quantity_unit": "spot_board_lot",
                    "spot_fill_share_quantity": label.cumulative_fill_quantity * SPOT_LOT_SHARES,
                    "required_spot_quantity": label.futures_equivalent_quantity,
                    "required_spot_quantity_unit": "spot_board_lot",
                    "residual_spot_quantity": label.residual_quantity,
                    "complete_two_lots": label.hedge_unit_complete,
                    "incremental_fill_count": label.incremental_fill_count,
                    "first_fill_recv_time_ns": fills[0].cursor.recv_time_ns,
                    "completion_recv_time_ns": completion.recv_time_ns if completion else None,
                    "completion_event_sequence": completion.event_sequence if completion else None,
                    "completion_row_index": completion.row_index if completion else None,
                    "trial_match_fill_quantity": label.trial_match_fill_quantity,
                    "gate_closed_fill_quantity": label.gate_closed_fill_quantity,
                    "gate_reasons": list(label.gate_reasons),
                }
            )
    return _from_records(records, _partial_schema()).sort(
        ["Date", "ValueCode", "raw_order_fact_id", "horizon_ms"]
    )


def summarize_hedges(frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Aggregate completion, depth, latency and total slippage separately."""

    if frame.is_empty():
        return pl.DataFrame()
    valid_decision = pl.col("status").is_in(["executable", "insufficient_depth"])
    executable = pl.col("status") == "executable"
    invalid_statuses = [
        "arrival_trial_match",
        "arrival_gate_closed",
        "decision_trial_match",
        "decision_gate_closed",
    ]
    return (
        frame.group_by(group_columns)
        .agg(
            pl.len().alias("hedge_events"),
            executable.sum().alias("executable_events"),
            (pl.col("status") == "insufficient_depth").sum().alias("insufficient_depth_events"),
            pl.col("status").is_in(invalid_statuses).sum().alias("invalid_state_events"),
            (~valid_decision).sum().alias("unpriced_events"),
            pl.col("available_quantity").filter(valid_decision).median().alias("available_quantity_p50"),
            pl.col("depth_shortfall").filter(valid_decision).quantile(0.95).alias("depth_shortfall_p95"),
            pl.col("arrival_book_age_ms").drop_nulls().quantile(0.95).alias("arrival_book_age_ms_p95"),
            pl.col("decision_book_age_ms").drop_nulls().quantile(0.95).alias("decision_book_age_ms_p95"),
            pl.col("signed_latency_slippage_bp").filter(valid_decision).median().alias("latency_slippage_bp_p50"),
            pl.col("signed_latency_slippage_bp").filter(valid_decision).quantile(0.80).alias("latency_slippage_bp_p80"),
            pl.col("signed_latency_slippage_bp").filter(valid_decision).quantile(0.95).alias("latency_slippage_bp_p95"),
            pl.col("signed_depth_slippage_bp").filter(executable).median().alias("depth_slippage_bp_p50"),
            pl.col("signed_depth_slippage_bp").filter(executable).quantile(0.80).alias("depth_slippage_bp_p80"),
            pl.col("signed_depth_slippage_bp").filter(executable).quantile(0.95).alias("depth_slippage_bp_p95"),
            pl.col("signed_total_slippage_bp").filter(executable).median().alias("total_slippage_bp_p50"),
            pl.col("signed_total_slippage_bp").filter(executable).quantile(0.80).alias("total_slippage_bp_p80"),
            pl.col("signed_total_slippage_bp").filter(executable).quantile(0.95).alias("total_slippage_bp_p95"),
        )
        .with_columns(
            (pl.col("executable_events") / pl.col("hedge_events")).alias("executable_rate"),
            (pl.col("insufficient_depth_events") / pl.col("hedge_events")).alias("insufficient_depth_rate"),
            (pl.col("invalid_state_events") / pl.col("hedge_events")).alias("invalid_state_rate"),
        )
        .sort(group_columns)
    )


def summarize_spot_partial(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    group = ["ValueCode", "horizon_ms"]
    return (
        frame.group_by(group)
        .agg(
            pl.len().alias("spot_fill_generations"),
            pl.col("complete_two_lots").sum().alias("complete_two_lot_generations"),
            (pl.col("status") == "partial").sum().alias("partial_generations"),
            (pl.col("status") == "no_fill").sum().alias("no_additional_fill_generations"),
            pl.col("residual_spot_quantity").median().alias("residual_spot_lots_p50"),
        )
        .with_columns(
            (pl.col("complete_two_lot_generations") / pl.col("spot_fill_generations")).alias("two_lot_completion_rate")
        )
        .sort(group)
    )


def write_hedge_study(
    result: HedgeStudyResult,
    output_dir: Path,
    *,
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS,
) -> None:
    """Write WP03 artifacts without overwriting WP02's ``config.json``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.hedge_facts.write_parquet(output_dir / "hedge_facts.parquet")
    result.hedge_by_day_symbol.write_csv(output_dir / "hedge_by_day_symbol.csv")
    result.hedge_summary.write_csv(output_dir / "hedge_summary.csv")
    result.spot_partial_horizons.write_parquet(output_dir / "spot_partial_horizons.parquet")
    result.spot_partial_summary.write_csv(output_dir / "spot_partial_summary.csv")
    result.audit.write_csv(output_dir / "hedge_audit.csv")
    config = {
        "hedge_delay_ms": hedge_delay_ns / 1_000_000.0,
        "arrival_reference": "latest_opposite_raw_state_causally_at_or_before_fill_cursor",
        "decision_book": "latest_opposite_raw_state_at_or_before_fill_plus_delay",
        "future_ask_hedge": "buy_2_spot_board_lots",
        "spot_bid_hedge": "sell_1_future_contract_after_full_2_spot_lots",
        "spot_board_lot_shares": SPOT_LOT_SHARES,
        "required_contract_shares": CONTRACT_SHARES,
        "future_best_l1_merge": "same_price_max_quantity_then_best_5_unique_levels",
        "partial_horizons_ms": list(PARTIAL_HORIZONS_MS),
        "policy_aliases_deduplicated_by": "raw_order_fact_id",
    }
    (output_dir / "hedge_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare_hedge_requests(
    aliases: pl.DataFrame, raw_facts: pl.DataFrame
) -> tuple[_HedgeFactRequest, ...]:
    required_alias = {
        "raw_order_fact_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "maker_market",
        "intended_quantity",
        "boundary_quantile",
        "full_fill",
        *_FILL_CURSOR_COLUMNS,
    }
    _require_columns(aliases, required_alias, "order aliases")
    _require_columns(raw_facts, {"raw_order_fact_id"}, "raw order facts")
    fact_ids = set(raw_facts.get_column("raw_order_fact_id").cast(pl.String).to_list())
    alias_ids = set(aliases.get_column("raw_order_fact_id").cast(pl.String).to_list())
    missing_facts = sorted(alias_ids - fact_ids)
    if missing_facts:
        raise ValueError(f"order aliases missing raw fact rows: {missing_facts[:5]}")

    eligible = aliases.filter(
        pl.col("route").is_in(SUPPORTED_ENTRY_ROUTES)
        & (pl.col("full_fill") == True)  # noqa: E712
        & pl.col("full_fill_recv_time_ns").is_not_null()
    )
    all_alias_counts = {
        str(row["raw_order_fact_id"]): int(row["len"])
        for row in aliases.group_by("raw_order_fact_id").len().iter_rows(named=True)
    }
    requests: list[_HedgeFactRequest] = []
    for raw_id, group in _partition_by_string(eligible, "raw_order_fact_id"):
        rows = list(group.iter_rows(named=True))
        first = rows[0]
        _require_one_value(rows, ("Date", "ValueCode", "QuoteCode", "route", "maker_market", "intended_quantity"), raw_id)
        cursor_values = {
            tuple(int(row[column]) for column in _FILL_CURSOR_COLUMNS)
            for row in rows
        }
        if len(cursor_values) != 1:
            raise ValueError(f"inconsistent full-fill cursor for raw fact {raw_id}")
        route = str(first["route"])
        expected_market = "future" if route == FUTURE_ASK_ROUTE else "spot"
        if str(first["maker_market"]) != expected_market:
            raise ValueError(f"maker market does not match route for raw fact {raw_id}")
        expected_quantity = 1 if route == FUTURE_ASK_ROUTE else 2
        if int(first["intended_quantity"]) != expected_quantity:
            raise ValueError(
                f"unexpected maker quantity for {raw_id}: expected {expected_quantity}"
            )
        recv_ns, event_sequence, row_index = next(iter(cursor_values))
        expected_priority = _MARKET_PRIORITY[expected_market]
        if event_sequence != expected_priority:
            raise ValueError(
                f"full-fill source priority for {raw_id} is {event_sequence}, expected {expected_priority}"
            )
        requests.append(
            _HedgeFactRequest(
                raw_order_fact_id=raw_id,
                date=str(first["Date"]),
                value_code=str(first["ValueCode"]),
                quote_code=str(first["QuoteCode"]),
                route=route,
                fill_cursor=EventCursor(recv_ns, event_sequence, row_index),
                maker_fill_quantity=expected_quantity,
                full_fill_boundary_quantiles=tuple(sorted({int(row["boundary_quantile"]) for row in rows})),
                full_fill_policy_alias_count=len(rows),
                all_policy_alias_count=all_alias_counts[raw_id],
            )
        )
    return tuple(sorted(requests, key=lambda item: (item.date, item.value_code, item.fill_cursor, item.route)))


def _label_one_request(
    request: _HedgeFactRequest,
    index: _RawStateIndex,
    *,
    contract_size: float,
    hedge_delay_ns: int,
) -> dict[str, object]:
    # Same-timestamp source ordering is material only at arrival.  All spot
    # events (priority 2) are after a futures fill (priority 1); all futures
    # events are before a spot fill at the same timestamp.
    opposite_priority = _MARKET_PRIORITY[request.opposite_market]
    arrival_query_ns = request.fill_cursor.recv_time_ns
    if opposite_priority > request.fill_cursor.event_sequence:
        arrival_query_ns -= 1
    arrival = index.latest(arrival_query_ns)
    decision_ns = request.fill_cursor.recv_time_ns + hedge_delay_ns
    decision = index.latest(decision_ns)
    snapshots = _unique_snapshots(arrival, decision)
    hedge_request = MakerFillHedgeRequest(
        request.raw_order_fact_id,
        request.fill_cursor,
        request.maker_fill_quantity,
        request.hedge_side,
        request.hedge_quantity,
        delay_ns=hedge_delay_ns,
    )
    label = label_delayed_taker_hedge(hedge_request, snapshots)
    record = asdict(label)
    # Replace nested cursor dictionaries with stable scalar columns.
    for key in (
        "maker_fill_cursor",
        "arrival_snapshot_cursor",
        "decision_snapshot_cursor",
    ):
        record.pop(key, None)
    arrival_cursor = label.arrival_snapshot_cursor
    decision_cursor = label.decision_snapshot_cursor
    record.update(
        {
            "Date": request.date,
            "ValueCode": request.value_code,
            "QuoteCode": request.quote_code,
            "route": request.route,
            "raw_order_fact_id": request.raw_order_fact_id,
            "maker_market": request.maker_market,
            "maker_fill_quantity_unit": "future_contract" if request.maker_market == "future" else "spot_board_lot",
            "opposite_market": request.opposite_market,
            "hedge_quantity_unit": "spot_board_lot" if request.opposite_market == "spot" else "future_contract",
            "spot_board_lot_shares": SPOT_LOT_SHARES,
            "contract_size_shares": int(contract_size),
            "hedge_share_equivalent": CONTRACT_SHARES,
            "full_fill_boundary_quantiles": list(request.full_fill_boundary_quantiles),
            "full_fill_policy_alias_count": request.full_fill_policy_alias_count,
            "all_policy_alias_count": request.all_policy_alias_count,
            "maker_fill_recv_time_ns": request.fill_cursor.recv_time_ns,
            "maker_fill_event_sequence": request.fill_cursor.event_sequence,
            "maker_fill_row_index": request.fill_cursor.row_index,
            "arrival_snapshot_recv_time_ns": arrival_cursor.recv_time_ns if arrival_cursor else None,
            "arrival_snapshot_event_sequence": arrival_cursor.event_sequence if arrival_cursor else None,
            "arrival_snapshot_row_index": arrival_cursor.row_index if arrival_cursor else None,
            "decision_snapshot_recv_time_ns": decision_cursor.recv_time_ns if decision_cursor else None,
            "decision_snapshot_event_sequence": decision_cursor.event_sequence if decision_cursor else None,
            "decision_snapshot_row_index": decision_cursor.row_index if decision_cursor else None,
            "arrival_book_recv_time_ns": arrival.book_recv_time_ns if arrival else None,
            "decision_book_recv_time_ns": decision.book_recv_time_ns if decision else None,
            "arrival_state_age_ms": _state_age_ms(request.fill_cursor.recv_time_ns, arrival),
            "arrival_book_age_ms": _book_age_ms(request.fill_cursor.recv_time_ns, arrival),
            "decision_state_age_ms": _state_age_ms(decision_ns, decision),
            "decision_book_age_ms": _book_age_ms(decision_ns, decision),
        }
    )
    return record


def _contract_size_for_request(
    request: _HedgeFactRequest, tape: RawTapeDay
) -> float:
    mapping = tape.mapping.filter(
        pl.col("ValueCode").cast(pl.String) == request.value_code
    )
    if mapping.height != 1:
        raise ValueError(
            f"{request.date}/{request.value_code} needs exactly one raw-tape mapping"
        )
    contract_size = float(mapping.item(0, "contract_size"))
    if not math.isclose(
        contract_size, CONTRACT_SHARES, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"{request.date}/{request.value_code} contract_size={contract_size}; "
            f"WP03 quantity contract requires {CONTRACT_SHARES} shares"
        )
    return contract_size


def _snapshot_from_state(
    row: dict[str, object], market: Literal["spot", "future"]
) -> OppositeBookSnapshot:
    bids = executable_levels_from_state(row, market, "bid")
    asks = executable_levels_from_state(row, market, "ask")
    gate_open, gate_reason = _hedge_gate(row, bids, asks)
    return OppositeBookSnapshot(
        EventCursor(
            int(row["recv_time_ns"]),
            _MARKET_PRIORITY[market],
            int(row["sequence"]),
        ),
        bids,
        asks,
        trial_match=bool(row.get("trial_match", False)),
        gate_open=gate_open,
        gate_reason=gate_reason,
    )


def _hedge_gate(
    row: dict[str, object],
    bids: tuple[BookLevel, ...],
    asks: tuple[BookLevel, ...],
) -> tuple[bool, str | None]:
    if not bool(row.get("book_state_available", False)):
        return False, "no_prior_book"
    explicit = row.get("gate_open")
    if isinstance(explicit, bool) and not explicit:
        return False, str(row.get("gate_reason") or "raw_state_gate_closed")
    if not bids or not asks or bids[0].price > asks[0].price:
        return False, "invalid_executable_book"
    reference = row.get("ref_price")
    if not _positive_number(reference):
        return False, "missing_ref_price"
    lower = float(reference) * 0.91
    upper = float(reference) * 1.08
    if not (lower < bids[0].price < upper and lower < asks[0].price < upper):
        return False, "ref_price_band"
    return True, None


def _unique_snapshots(
    arrival: _StateHit | None, decision: _StateHit | None
) -> tuple[OppositeBookSnapshot, ...]:
    values: dict[EventCursor, OppositeBookSnapshot] = {}
    for hit in (arrival, decision):
        if hit is not None:
            values[hit.snapshot.cursor] = hit.snapshot
    return tuple(values[cursor] for cursor in sorted(values))


def _event_cursor_from_path(row: dict[str, object]) -> EventCursor:
    return EventCursor(
        int(row["fill_recv_time_ns"]),
        int(row["fill_event_sequence"]),
        int(row["fill_row_index"]),
    )


def _representative_metadata(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw_id, group in _partition_by_string(frame, "raw_order_fact_id"):
        rows = list(group.iter_rows(named=True))
        _require_one_value(rows, ("Date", "ValueCode", "QuoteCode", "route"), raw_id)
        result[raw_id] = rows[0]
    return result


def _partition_by_string(
    frame: pl.DataFrame, column: str
) -> Iterable[tuple[str, pl.DataFrame]]:
    if frame.is_empty():
        return ()
    return (
        (str(key[0] if isinstance(key, tuple) else key), group)
        for key, group in frame.partition_by(column, as_dict=True).items()
    )


def _require_one_value(
    rows: list[dict[str, object]], columns: Sequence[str], identity: str
) -> None:
    for column in columns:
        values = {row.get(column) for row in rows}
        if len(values) != 1:
            raise ValueError(f"inconsistent {column} for raw fact {identity}")


def _append_level(
    output: list[tuple[float, int]], price: object, quantity: object
) -> None:
    if not _positive_number(price) or not _positive_number(quantity):
        return
    integer_quantity = int(quantity)
    if not math.isclose(float(quantity), integer_quantity, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"book quantity is not integral: {quantity}")
    output.append((float(price), integer_quantity))


def _state_age_ms(query_ns: int, hit: _StateHit | None) -> float | None:
    return (
        (query_ns - hit.state_recv_time_ns) / 1_000_000.0
        if hit is not None
        else None
    )


def _book_age_ms(query_ns: int, hit: _StateHit | None) -> float | None:
    return (
        (query_ns - hit.book_recv_time_ns) / 1_000_000.0
        if hit is not None and hit.book_recv_time_ns is not None
        else None
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _required_positive_int(value: object, name: str) -> int:
    if not _positive_number(value):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if not math.isclose(float(value), result, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _build_audit(
    aliases: pl.DataFrame,
    raw_facts: pl.DataFrame,
    requests: tuple[_HedgeFactRequest, ...],
    hedge_facts: pl.DataFrame,
    partial: pl.DataFrame,
    quantity_paths: pl.DataFrame | None,
    tapes: tuple[RawTapeDay, ...],
    hedge_delay_ns: int,
) -> pl.DataFrame:
    supported = aliases.filter(pl.col("route").is_in(SUPPORTED_ENTRY_ROUTES))
    full = supported.filter(pl.col("full_fill") == True)  # noqa: E712
    return pl.from_dicts(
        [
            {
                "input_alias_rows": aliases.height,
                "input_raw_fact_rows": raw_facts.height,
                "supported_full_fill_alias_rows": full.height,
                "unique_full_fill_raw_facts": len(requests),
                "policy_alias_duplicates_collapsed": full.height - len(requests),
                "hedge_fact_rows": hedge_facts.height,
                "raw_tape_days": len(tapes),
                "hedge_delay_ms": hedge_delay_ns / 1_000_000.0,
                "exact_full_fill_cursor_available": all(column in aliases.columns for column in _FILL_CURSOR_COLUMNS),
                "quantity_paths_available": quantity_paths is not None and not quantity_paths.is_empty(),
                "partial_generation_rows": partial.get_column("raw_order_fact_id").n_unique() if not partial.is_empty() else 0,
                "quantity_unit_contract": "1 future contract = 2 spot board lots = 2000 shares",
                "arrival_equal_timestamp_rule": "future_before_spot",
            }
        ],
        infer_schema_length=None,
    )


def _hedge_fact_schema() -> dict[str, pl.DataType]:
    # This schema is primarily used when a selected day has no fills.  Nonempty
    # records are inferred so all detailed label fields remain available.
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "raw_order_fact_id": pl.String,
        "status": pl.String,
        "hedge_side": pl.String,
        "maker_fill_recv_time_ns": pl.Int64,
        "decision_time_ns": pl.Int64,
        "requested_hedge_quantity": pl.Int64,
        "hedge_quantity_unit": pl.String,
    }


def _partial_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "raw_order_fact_id": pl.String,
        "horizon_ms": pl.Int64,
        "status": pl.String,
        "spot_fill_quantity": pl.Int64,
        "spot_fill_quantity_unit": pl.String,
        "spot_fill_share_quantity": pl.Int64,
        "required_spot_quantity": pl.Int64,
        "required_spot_quantity_unit": pl.String,
        "residual_spot_quantity": pl.Int64,
        "complete_two_lots": pl.Boolean,
        "incremental_fill_count": pl.Int64,
        "first_fill_recv_time_ns": pl.Int64,
        "completion_recv_time_ns": pl.Int64,
        "completion_event_sequence": pl.Int64,
        "completion_row_index": pl.Int64,
        "trial_match_fill_quantity": pl.Int64,
        "gate_closed_fill_quantity": pl.Int64,
        "gate_reasons": pl.List(pl.String),
    }


def _from_records(
    records: list[dict[str, object]], empty_schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    return (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame(schema=empty_schema)
    )


def _require_columns(
    frame: pl.DataFrame, required: set[str], source: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def parse_args() -> argparse.Namespace:
    """Parse a memory-bounded one-product-day research run."""

    parser = argparse.ArgumentParser(description="Run WP03 50ms hedge replay")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--quote-fill-dir", type=Path, default=Path("maker/data/quote_fill")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Load one product-day, keeping large raw tapes out of pooled memory."""

    args = parse_args()
    aliases = pl.read_parquet(args.quote_fill_dir / "order_aliases.parquet").filter(
        (pl.col("Date") == str(args.date))
        & (pl.col("ValueCode") == str(args.symbol))
    )
    facts = pl.read_parquet(args.quote_fill_dir / "raw_order_facts.parquet").filter(
        (pl.col("Date") == str(args.date))
        & (pl.col("ValueCode") == str(args.symbol))
    )
    if aliases.is_empty():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return

    # Local imports avoid coupling the pure orchestration helpers to WP01.
    from .merged import load_replay_mapping
    from .raw_tape import load_raw_tape_day

    mapping = load_replay_mapping(str(args.date), [str(args.symbol)])
    tape = load_raw_tape_day(str(args.date), mapping)
    result = run_hedge_study(aliases, facts, tape, output_dir=args.output_dir)
    print(result.hedge_summary)


if __name__ == "__main__":
    main()
