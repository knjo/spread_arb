"""Compact event-driven maker execution evaluator prototype.

The formal execution runner deliberately preserves every sparse target-state
change plus shadow/candidate diagnostics.  This prototype is a bounded,
lower-output/lower-query alternative once a product-day is loaded:

* candidates are admitted only at a ``spread_pair_epoch`` transition or a
  new forward absolute tick inside the epoch;
* each candidate's first retreat/gate/cutoff stop is precomputed from the
  change-point series in linear time, rather than rediscovering it while
  rolling every order;
* an indexed MBP quantity query is made exactly once for every generation
  that is not handled by the optional legacy spot ``makerFill`` adapter;
* only compact order outcomes, material lifecycle changes, and audit rows are
  returned.

The legacy ``makerFill`` path is an explicitly approximate EOD sanity label.
The primary spot v1 accepts only A1/A2/B1/B2 mappings and leaves all other
spot ranks unknown/fail-closed; futures maker continues to use the general
indexed replay.  The module is intentionally independent of every frozen
formal publication root.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping

import polars as pl

from ..common.paths import HFT_DATA_ROOT
from ..quote_width.rolling import DEFAULT_QUANTILES
from .engine import TargetObservation
from .hedge import DEFAULT_HEDGE_DELAY_NS
from .hedge_study import run_hedge_study
from .indexed_replay import IndexedTradeReplay, QuantityFillLabel
from .layered import EventCursor, LayeredSampler
from .merged import MergedTargetStudyInput, _rank_and_queue, session_cutoff_cursor
from .pilot import ENTRY_ROUTES
from .replay import IndependentOrderWindow
from .study import _trade_events, summarize_raw_order_facts
from .targets import ROUTE_SPECS, tick_index_to_price


COMPACT_EVALUATOR_VERSION = "compact_execution_prototype_v2_first_passage"
ONE_SECOND_NS = 1_000_000_000
_SYNTHETIC_DECISION_PRIORITY = 3
_SPOT_PRIORITY = 2
_FUTURE_PRIORITY = 1
_MAKER_FILL_COLUMNS: Mapping[str, str] = {
    "ASK1": "Ask1_FillSeconds",
    "ASK2": "Ask2_FillSeconds",
    "BID1": "Bid1_FillSeconds",
    "BID2": "Bid2_FillSeconds",
}
_REQUIRED_OBSERVATION_COLUMNS = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "boundary_quantile",
    "spread_pair_epoch",
    "base_epoch_transition",
    "event_source",
    "recv_time_ns",
    "cursor_event_sequence",
    "cursor_row_index",
    "absolute_target_price",
    "absolute_target_tick",
    "target_rank",
    "initial_queue_ahead",
    "gate_open",
    "admission_reason",
    "maker_market",
    "maker_side",
}
_OUTCOME_BASE_SCHEMA: Mapping[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "boundary_quantile": pl.Int64,
    "spread_pair_epoch": pl.Int64,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "target_price_tick": pl.Int64,
    "target_price": pl.Float64,
    "target_rank_at_submit": pl.String,
    "initial_queue_ahead": pl.Int64,
    "queue_known": pl.Boolean,
    "intended_quantity": pl.Int64,
    "submit_recv_time_ns": pl.Int64,
    "submit_event_sequence": pl.Int64,
    "submit_row_index": pl.Int64,
    "decision_reason": pl.String,
    "sampling_mode": pl.String,
    "source_recv_time_ns": pl.Int64,
    "source_event_sequence": pl.Int64,
    "source_row_index": pl.Int64,
    "nominal_stop_recv_time_ns": pl.Int64,
    "nominal_stop_event_sequence": pl.Int64,
    "nominal_stop_row_index": pl.Int64,
    "nominal_stop_reason": pl.String,
    "nominal_stop_type": pl.String,
    "first_fill_recv_time_ns": pl.Int64,
    "first_fill_event_sequence": pl.Int64,
    "first_fill_row_index": pl.Int64,
    "full_fill_recv_time_ns": pl.Int64,
    "full_fill_event_sequence": pl.Int64,
    "full_fill_row_index": pl.Int64,
    "known_filled_quantity": pl.Int64,
    "any_fill": pl.Boolean,
    "full_fill": pl.Boolean,
    "partial_fill": pl.Boolean,
    "trade_through_fill": pl.Boolean,
    "terminal_recv_time_ns": pl.Int64,
    "terminal_reason": pl.String,
    "cancel_required": pl.Boolean,
    "outcome_supported": pl.Boolean,
    "outcome_status": pl.String,
    "fill_backend": pl.String,
    "fill_backend_reason": pl.String,
    "makerfill_mapping_exact": pl.Boolean,
    "fill_outcome_exact_within_model": pl.Boolean,
    "fill_cursor_exact": pl.Boolean,
    "indexed_quantity_query_count": pl.Int64,
    "makerfill_fill_seconds": pl.Float64,
    "makerfill_candidate_fill_ns": pl.Int64,
    "maker_snapshot_sequence": pl.Int64,
    "maker_snapshot_recv_time_ns": pl.Int64,
    "maker_snapshot_exact_at_decision": pl.Boolean,
    "peak_nominal_layers_policy_day": pl.Int64,
    "independent_event_label": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
}


@dataclass(frozen=True)
class CompactEvaluatorConfig:
    """Explicit prototype clock, routes, and fill backend."""

    routes: tuple[str, ...] = tuple(ENTRY_ROUTES)
    boundary_quantiles: tuple[int, ...] = tuple(DEFAULT_QUANTILES)
    sampling_mode: Literal["event_change_first_passage"] = (
        "event_change_first_passage"
    )
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS
    fill_backend: Literal["indexed", "hybrid_makerfill"] = "hybrid_makerfill"

    def validate(self) -> None:
        if self.sampling_mode != "event_change_first_passage":
            raise ValueError(
                "only sampling_mode='event_change_first_passage' is implemented"
            )
        if (
            isinstance(self.hedge_delay_ns, bool)
            or not isinstance(self.hedge_delay_ns, int)
            or self.hedge_delay_ns < 0
        ):
            raise ValueError("hedge_delay_ns must be a non-negative integer")
        if not self.routes or len(self.routes) != len(set(self.routes)):
            raise ValueError("routes must be unique and non-empty")
        unknown_routes = sorted(set(self.routes) - set(ROUTE_SPECS))
        if unknown_routes:
            raise ValueError(f"unknown routes: {unknown_routes}")
        if (
            not self.boundary_quantiles
            or len(self.boundary_quantiles) != len(set(self.boundary_quantiles))
        ):
            raise ValueError("boundary_quantiles must be unique and non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.boundary_quantiles
        ):
            raise ValueError("boundary_quantiles must be positive integers")
        if self.fill_backend not in {"indexed", "hybrid_makerfill"}:
            raise ValueError("unsupported fill_backend")


@dataclass(frozen=True)
class CompactDecision:
    """One retained decision plus the full submit metadata used by replay."""

    observation: TargetObservation
    metadata: Mapping[str, object]
    decision_reason: Literal[
        "spread_pair_epoch_transition", "forward_new_absolute_tick"
    ]


@dataclass(frozen=True)
class CompactCandidate:
    """One admitted order and its precomputed first-passage stop."""

    decision: CompactDecision
    generation: int
    stop_cursor: EventCursor
    stop_type: Literal["target_retreat", "gate_invalid", "session_cutoff"]
    stop_reason: str


@dataclass(frozen=True)
class CompactEvaluationResult:
    order_outcomes: pl.DataFrame
    state_changes: pl.DataFrame
    audit: pl.DataFrame


@dataclass(frozen=True)
class _CompactFill:
    backend: str
    backend_reason: str
    makerfill_mapping_exact: bool
    outcome_exact: bool
    fill_cursor_exact: bool
    indexed_query_count: int
    queue_known: bool
    first_fill_cursor: EventCursor | None
    full_fill_cursor: EventCursor | None
    known_filled_quantity: int | None
    any_fill: bool | None
    full_fill: bool | None
    partial_fill: bool | None
    trade_through_fill: bool | None
    makerfill_fill_seconds: float | None
    makerfill_candidate_fill_ns: int | None


class _MarketStateIndex:
    """Immutable exact-cursor as-of index over normalized raw states."""

    def __init__(self, states: pl.DataFrame, market: str) -> None:
        if market not in {"spot", "future"}:
            raise ValueError(f"unknown market: {market}")
        required = {"recv_time_ns", "sequence", "packet_sequence"}
        _require_columns(states, required, f"{market} states")
        self.market = market
        self.priority = _SPOT_PRIORITY if market == "spot" else _FUTURE_PRIORITY
        self.states = states.sort(
            ["recv_time_ns", "sequence", "packet_sequence"]
        )
        self.recv_times = tuple(
            int(value) for value in self.states.get_column("recv_time_ns")
        )
        self.sequences = tuple(
            int(value) for value in self.states.get_column("sequence")
        )

    def latest(self, cursor: EventCursor) -> dict[str, object] | None:
        if not self.recv_times:
            return None
        left = bisect_left(self.recv_times, cursor.recv_time_ns)
        right = bisect_right(self.recv_times, cursor.recv_time_ns)
        if self.priority < cursor.event_sequence:
            position = right - 1
        elif self.priority > cursor.event_sequence:
            position = left - 1
        else:
            position = bisect_right(
                self.sequences,
                cursor.row_index,
                left,
                right,
            ) - 1
            if position < left:
                position = left - 1
        if position < 0:
            return None
        return self.states.row(position, named=True)


class MakerFillFastAdapter:
    """Cached legacy A1/A2/B1/B2 ``FillSeconds`` adapter.

    One cached spot snapshot is selected causally as-of the candidate cursor;
    its four FillSeconds values are never reset or shifted at later decisions.
    A target beyond level two or a missing/unusable snapshot fails closed.
    Even a direct mapping has ``outcome_exact=False``: legacy ``FillSeconds``
    is Float32, EOD-looking, and does not expose the exact trade cursor or a
    two-lot partial path.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        required = {"QuoteCode", "ChannelSeq", *_MAKER_FILL_COLUMNS.values()}
        _require_columns(frame, required, "makerFill frame")
        duplicate = frame.group_by("QuoteCode", "ChannelSeq").len().filter(
            pl.col("len") != 1
        )
        if duplicate.height:
            raise ValueError("makerFill snapshot keys are duplicated")
        nonnumeric = [
            column
            for column in _MAKER_FILL_COLUMNS.values()
            if not frame.schema[column].is_numeric()
        ]
        if nonnumeric:
            raise ValueError(
                f"makerFill FillSeconds columns must be numeric: {nonnumeric}"
            )
        self._snapshots = {
            (str(row["QuoteCode"]), int(row["ChannelSeq"])): {
                column: row[column] for column in _MAKER_FILL_COLUMNS.values()
            }
            for row in frame.iter_rows(named=True)
        }

    @classmethod
    def load(
        cls,
        date: str,
        value_code: str,
        *,
        data_root: Path = HFT_DATA_ROOT,
    ) -> "MakerFillFastAdapter":
        path = Path(data_root) / "makerFill" / f"{date}_makerFill.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("QuoteCode").cast(pl.String) == str(value_code))
            .select("QuoteCode", "ChannelSeq", *_MAKER_FILL_COLUMNS.values())
            .collect(engine="streaming")
        )
        if frame.is_empty():
            raise ValueError(f"{path}: no makerFill rows for {value_code}")
        return cls(frame)

    def label(
        self,
        window: IndependentOrderWindow,
        metadata: Mapping[str, object],
        *,
        value_code: str,
        intended_quantity: int,
    ) -> _CompactFill | None:
        if str(metadata.get("maker_market")) != "spot":
            return None
        rank = metadata.get("target_rank")
        column = _MAKER_FILL_COLUMNS.get(str(rank))
        if column is None:
            return None
        sequence = metadata.get("maker_snapshot_sequence")
        snapshot_time_ns = metadata.get("maker_snapshot_recv_time_ns")
        if sequence is None or snapshot_time_ns is None:
            return None
        snapshot = self._snapshots.get((str(value_code), int(sequence)))
        if snapshot is None:
            return None
        raw_seconds = snapshot[column]
        if raw_seconds is None:
            return None
        try:
            numeric_seconds = float(raw_seconds)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric_seconds):
            # Legacy NaN alone is the documented no-fill-through-EOD sentinel.
            seconds = None
        elif not math.isfinite(numeric_seconds):
            return None
        else:
            seconds = numeric_seconds
        candidate_ns: int | None = None
        candidate: EventCursor | None = None
        if seconds is not None:
            if seconds < 0:
                return None
            # FillSeconds is anchored to this exact cached spot snapshot, not
            # reset at every decision.  If its implied fill predates the
            # candidate submit, the legacy label is unusable and must remain
            # unknown rather than being shifted forward.
            candidate_ns = int(snapshot_time_ns) + int(
                round(seconds * ONE_SECOND_NS)
            )
            # The active interval is (start, stop].  A zero-second or
            # pre-submit legacy label cannot establish a causal fill, so the
            # hybrid v1 leaves the outcome unknown/fail-closed.
            if candidate_ns <= window.start_cursor.recv_time_ns:
                return None
            candidate_cursor = EventCursor(candidate_ns, _SPOT_PRIORITY, 0)
            if candidate_cursor <= window.stop_cursor:
                candidate = candidate_cursor
        full = candidate is not None
        return _CompactFill(
            backend="makerfill_fast",
            backend_reason="exact_asof_snapshot_key_l1_l2_approx_outcome",
            makerfill_mapping_exact=True,
            outcome_exact=False,
            fill_cursor_exact=False,
            indexed_query_count=0,
            queue_known=window.initial_queue_ahead is not None,
            first_fill_cursor=candidate,
            full_fill_cursor=candidate,
            known_filled_quantity=intended_quantity if full else 0,
            any_fill=full,
            full_fill=full,
            partial_fill=False,
            trade_through_fill=None,
            makerfill_fill_seconds=seconds,
            makerfill_candidate_fill_ns=candidate_ns,
        )


def load_makerfill_fast_adapter(
    date: str,
    value_code: str,
    *,
    data_root: Path = HFT_DATA_ROOT,
) -> MakerFillFastAdapter:
    """Load one product-day legacy adapter exactly once."""

    return MakerFillFastAdapter.load(
        date,
        value_code,
        data_root=Path(data_root),
    )


def build_compact_candidates(
    observations: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    route: str,
    boundary_quantile: int,
    cutoff_cursor: EventCursor,
    maker_state_index: _MarketStateIndex | None = None,
) -> tuple[CompactCandidate, ...]:
    """Admit layered candidates and label first-passage stops in ``O(N)``.

    The admission scan is the existing :class:`LayeredSampler`: a new epoch
    admits a base order and a same-epoch forward unseen absolute tick admits a
    new layer.  Stop lookup is separate and vector-like.  A reverse monotonic
    stack computes the next strictly smaller target for bids or strictly
    greater target for asks; a second reverse scan computes the next invalid
    gate.  Each candidate takes the earlier cursor or session cutoff.
    """

    if not isinstance(cutoff_cursor, EventCursor):
        raise TypeError("cutoff_cursor must be an EventCursor")
    if observations.is_empty():
        return ()
    _require_columns(
        observations,
        _REQUIRED_OBSERVATION_COLUMNS,
        "target observations",
    )
    selected = observations.filter(
        (pl.col("Date").cast(pl.String) == str(date))
        & (pl.col("ValueCode").cast(pl.String) == str(value_code))
        & (pl.col("route") == str(route))
        & (pl.col("boundary_quantile") == int(boundary_quantile))
    ).sort(["recv_time_ns", "cursor_event_sequence", "cursor_row_index"])
    rows: list[dict[str, object]] = [
        row
        for row in selected.iter_rows(named=True)
        if _row_cursor(row) < cutoff_cursor
    ]
    if not rows:
        return ()
    state_observations: list[TargetObservation] = []
    for row in rows:
        cursor = _row_cursor(row)
        if maker_state_index is not None:
            _refresh_decision_maker_state(row, cursor, maker_state_index)
        state_observations.append(_target_observation(row, cursor))

    maker_side = ROUTE_SPECS[route].maker_side
    next_retreat = _next_retreat_indices(state_observations, maker_side)
    next_invalid = _next_gate_invalid_indices(state_observations)

    sampler = LayeredSampler(route)
    admissions: list[tuple[int, int, str]] = []
    for index, observation in enumerate(state_observations):
        actions = sampler.reconcile(
            observation.cursor,
            observation.spread_pair_epoch,
            observation.absolute_target_tick,
            gate_open=observation.gate_open,
            gate_reason=observation.gate_reason,
        )
        for action in actions:
            if action.kind != "submit":
                continue
            assert action.order is not None
            reason = (
                "spread_pair_epoch_transition"
                if action.reason == "new_epoch"
                else "forward_new_absolute_tick"
            )
            admissions.append((index, action.order.generation, reason))

    candidates: list[CompactCandidate] = []
    policy_prefix = f"{date}/{value_code}/{route}/q{boundary_quantile}/compact"
    for index, generation, reason in admissions:
        observation = state_observations[index]
        metadata = rows[index]
        metadata.update(
            {
                "source_recv_time_ns": observation.cursor.recv_time_ns,
                "source_event_sequence": observation.cursor.event_sequence,
                "source_row_index": observation.cursor.row_index,
                "decision_reason": reason,
                "sampling_mode": "event_change_first_passage",
                "policy_generation_id": f"{policy_prefix}/{generation}",
            }
        )
        stop_cursor = cutoff_cursor
        stop_type: Literal[
            "target_retreat", "gate_invalid", "session_cutoff"
        ] = "session_cutoff"
        stop_reason = "session_cutoff"
        retreat_index = next_retreat[index]
        invalid_index = next_invalid[index]
        if retreat_index is not None:
            retreat_cursor = state_observations[retreat_index].cursor
            if retreat_cursor < stop_cursor:
                stop_cursor = retreat_cursor
                stop_type = "target_retreat"
                stop_reason = "target_retreat"
        if invalid_index is not None:
            invalid_cursor = state_observations[invalid_index].cursor
            if invalid_cursor <= stop_cursor:
                stop_cursor = invalid_cursor
                stop_type = "gate_invalid"
                stop_reason = (
                    state_observations[invalid_index].gate_reason
                    or "gate_invalid"
                )
        metadata["first_passage_stop_type"] = stop_type
        metadata["first_passage_stop_reason"] = stop_reason
        decision = CompactDecision(
            observation=observation,
            metadata=metadata,
            decision_reason=reason,  # type: ignore[arg-type]
        )
        candidates.append(
            CompactCandidate(
                decision=decision,
                generation=generation,
                stop_cursor=stop_cursor,
                stop_type=stop_type,
                stop_reason=stop_reason,
            )
        )
    return tuple(candidates)


def _next_retreat_indices(
    observations: list[TargetObservation],
    maker_side: Literal["bid", "ask"],
) -> tuple[int | None, ...]:
    """First strict less-aggressive target via one reverse monotonic stack."""

    answer: list[int | None] = [None] * len(observations)
    stack: list[int] = []
    for index in range(len(observations) - 1, -1, -1):
        tick = observations[index].absolute_target_tick
        if not observations[index].gate_open or tick is None:
            continue
        if maker_side == "bid":
            while stack and observations[stack[-1]].absolute_target_tick >= tick:  # type: ignore[operator]
                stack.pop()
        else:
            while stack and observations[stack[-1]].absolute_target_tick <= tick:  # type: ignore[operator]
                stack.pop()
        answer[index] = stack[-1] if stack else None
        stack.append(index)
    return tuple(answer)


def _next_gate_invalid_indices(
    observations: list[TargetObservation],
) -> tuple[int | None, ...]:
    answer: list[int | None] = [None] * len(observations)
    next_invalid: int | None = None
    for index in range(len(observations) - 1, -1, -1):
        answer[index] = next_invalid
        if not observations[index].gate_open:
            next_invalid = index
    return tuple(answer)


def _target_observation(
    metadata: Mapping[str, object], cursor: EventCursor
) -> TargetObservation:
    return TargetObservation(
        cursor=cursor,
        spread_pair_epoch=int(metadata["spread_pair_epoch"]),
        absolute_target_tick=_optional_int(metadata.get("absolute_target_tick")),
        gate_open=bool(metadata["gate_open"]),
        gate_reason=str(metadata["admission_reason"]),
        initial_queue_ahead=_optional_int(metadata.get("initial_queue_ahead")),
        target_rank=(
            str(metadata["target_rank"])
            if metadata.get("target_rank") is not None
            else None
        ),
        source=str(metadata.get("event_source", "market_state")),
    )


def evaluate_compact_product_day(
    merged: MergedTargetStudyInput,
    config: CompactEvaluatorConfig = CompactEvaluatorConfig(),
    *,
    makerfill_adapter: MakerFillFastAdapter | None = None,
) -> CompactEvaluationResult:
    """Evaluate one loaded product-day without writing formal artifacts."""

    config.validate()
    if merged.mapping.height != 1:
        raise ValueError("compact evaluator requires one mapped product-day")
    date = str(merged.date)
    value_code = str(merged.mapping.item(0, "ValueCode"))
    quote_code = str(merged.mapping.item(0, "QuoteCode"))
    cutoff = session_cutoff_cursor(date)

    state_indexes = {
        "spot": _MarketStateIndex(
            merged.raw_tape.spot_states.filter(
                pl.col("ValueCode").cast(pl.String) == value_code
            ),
            "spot",
        ),
        "future": _MarketStateIndex(
            merged.raw_tape.future_states.filter(
                pl.col("ValueCode").cast(pl.String) == value_code
            ),
            "future",
        ),
    }
    indexed_markets = {
        ROUTE_SPECS[route].maker_market
        for route in config.routes
        if config.fill_backend == "indexed"
        or ROUTE_SPECS[route].maker_market == "future"
    }
    trade_indexes = {
        market: IndexedTradeReplay(
            _trade_events(
                (
                    merged.raw_tape.spot_trades
                    if market == "spot"
                    else merged.raw_tape.future_trades
                ).filter(pl.col("ValueCode").cast(pl.String) == value_code),
                market=market,  # type: ignore[arg-type]
            )
        )
        for market in indexed_markets
    }

    outcome_records: list[dict[str, object]] = []
    nominal_transition_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    for route in config.routes:
        spec = ROUTE_SPECS[route]
        for quantile in config.boundary_quantiles:
            candidates = build_compact_candidates(
                merged.observations,
                date=date,
                value_code=value_code,
                route=route,
                boundary_quantile=quantile,
                cutoff_cursor=cutoff,
                maker_state_index=state_indexes[spec.maker_market],
            )
            windows = tuple(
                IndependentOrderWindow(
                    generation_id=str(
                        candidate.decision.metadata["policy_generation_id"]
                    ),
                    maker_side=spec.maker_side,
                    target_price_tick=int(
                        candidate.decision.observation.absolute_target_tick
                    ),
                    start_cursor=candidate.decision.observation.cursor,
                    stop_cursor=candidate.stop_cursor,
                    initial_queue_ahead=(
                        candidate.decision.observation.initial_queue_ahead
                    ),
                    stop_reason=candidate.stop_reason,
                )
                for candidate in candidates
            )
            indexed_queries = makerfill_direct = excluded = 0
            peak_layers = _peak_candidate_layers(candidates)
            for candidate, window in zip(candidates, windows):
                metadata = candidate.decision.metadata
                intended_quantity = 1 if spec.maker_market == "future" else 2
                fill = _select_fill(
                    window,
                    metadata,
                    value_code=value_code,
                    intended_quantity=intended_quantity,
                    config=config,
                    makerfill_adapter=makerfill_adapter,
                    replay=trade_indexes.get(spec.maker_market),
                )
                indexed_queries += fill.indexed_query_count
                makerfill_direct += fill.backend == "makerfill_fast"
                excluded += fill.backend == "excluded_fail_closed"
                outcome_records.append(
                    _outcome_record(
                        date=date,
                        value_code=value_code,
                        quote_code=quote_code,
                        route=route,
                        quantile=quantile,
                        window=window,
                        metadata=metadata,
                        fill=fill,
                        intended_quantity=intended_quantity,
                        peak_active_layers=peak_layers,
                    )
                )
                nominal_transition_records.extend(
                    _candidate_nominal_state_records(
                        candidate,
                        date=date,
                        value_code=value_code,
                        quote_code=quote_code,
                        route=route,
                        quantile=quantile,
                        sampling_mode=config.sampling_mode,
                    )
                )
            audit_records.append(
                {
                    "scope": "policy",
                    "Date": date,
                    "ValueCode": value_code,
                    "QuoteCode": quote_code,
                    "route": route,
                    "boundary_quantile": quantile,
                    "source_sparse_observations": _policy_observation_count(
                        merged.observations,
                        date=date,
                        value_code=value_code,
                        route=route,
                        boundary_quantile=quantile,
                        cutoff=cutoff,
                    ),
                    "first_passage_candidates": len(candidates),
                    "epoch_decisions": sum(
                        item.decision.decision_reason
                        == "spread_pair_epoch_transition"
                        for item in candidates
                    ),
                    "forward_tick_decisions": sum(
                        item.decision.decision_reason
                        == "forward_new_absolute_tick"
                        for item in candidates
                    ),
                    "submitted_generations": len(windows),
                    "indexed_quantity_queries": indexed_queries,
                    "makerfill_direct_mappings": makerfill_direct,
                    "spot_fail_closed_exclusions": excluded,
                    "peak_nominal_layers": peak_layers,
                    "sampling_mode": config.sampling_mode,
                    "fill_backend": config.fill_backend,
                    "pathwise_ev_ready": False,
                    "joint_volume_allocated": False,
                }
            )

    outcomes = _normalize_outcomes(_frame(outcome_records, _empty_outcomes))
    nominal = _frame(nominal_transition_records, _empty_state_changes)
    state_changes = _materialize_actual_state_changes(nominal, outcomes)
    outcomes = _attach_hedges(
        outcomes,
        merged,
        hedge_delay_ns=config.hedge_delay_ns,
    )
    audit = pl.from_dicts(audit_records, infer_schema_length=None)
    if not audit.is_empty():
        total = audit.select(
            pl.lit("total").alias("scope"),
            pl.lit(date).alias("Date"),
            pl.lit(value_code).alias("ValueCode"),
            pl.lit(quote_code).alias("QuoteCode"),
            pl.lit("__all__").alias("route"),
            pl.lit(None, dtype=pl.Int64).alias("boundary_quantile"),
            *[
                pl.col(column).sum().alias(column)
                for column in (
                    "source_sparse_observations",
                    "first_passage_candidates",
                    "epoch_decisions",
                    "forward_tick_decisions",
                    "submitted_generations",
                    "indexed_quantity_queries",
                    "makerfill_direct_mappings",
                    "spot_fail_closed_exclusions",
                )
            ],
            pl.col("peak_nominal_layers").max().alias("peak_nominal_layers"),
            pl.lit(config.sampling_mode).alias("sampling_mode"),
            pl.lit(config.fill_backend).alias("fill_backend"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("joint_volume_allocated"),
        )
        audit = pl.concat([audit, total], how="vertical_relaxed")
    _validate_compact_result(outcomes, state_changes, audit, config)
    return CompactEvaluationResult(outcomes, state_changes, audit)


def summarize_compact_q(order_outcomes: pl.DataFrame) -> pl.DataFrame:
    """Summarize the auditable route/q/rank universe without alias inflation.

    Legacy makerFill rows remain explicitly approximate.  Unsupported spot
    ranks stay in the total candidate universe but outside the supported
    denominator, and every headline alias count is paired with its unique
    physical raw-order count.
    """

    if order_outcomes.is_empty():
        return pl.DataFrame()
    required = {
        "Date",
        "ValueCode",
        "route",
        "boundary_quantile",
        "target_rank_at_submit",
        "raw_order_fact_id",
        "fill_backend",
        "outcome_supported",
        "any_fill",
        "full_fill",
        "partial_fill",
        "cancel_required",
        "submit_recv_time_ns",
        "full_fill_recv_time_ns",
        "makerfill_fill_seconds",
        "hedge_estimate_available",
        "hedge_estimate_executable_vwap_price",
        "hedge_estimate_signed_total_slippage_bp",
        "hedge_label_observed",
        "hedge_executable",
    }
    _require_columns(order_outcomes, required, "compact order outcomes")
    supported = pl.col("outcome_supported")
    makerfill = pl.col("fill_backend") == "makerfill_fast"
    approx_full = makerfill & (pl.col("full_fill") == True)  # noqa: E712
    approx_cancel = makerfill & (pl.col("full_fill") == False)  # noqa: E712
    supported_full = supported & (pl.col("full_fill") == True)  # noqa: E712
    supported_partial = supported & (pl.col("partial_fill") == True)  # noqa: E712
    supported_any = supported & (pl.col("any_fill") == True)  # noqa: E712
    supported_no_fill = supported & (pl.col("any_fill") == False)  # noqa: E712
    supported_cancel = supported & (pl.col("cancel_required") == True)  # noqa: E712
    estimate = supported_full & pl.col("hedge_estimate_available")
    wait_seconds = (
        pl.col("full_fill_recv_time_ns") - pl.col("submit_recv_time_ns")
    ) / ONE_SECOND_NS
    groups = ["route", "boundary_quantile", "target_rank_at_submit"]
    result = (
        order_outcomes.group_by(groups)
        .agg(
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.len().alias("candidate_policy_aliases"),
            pl.col("raw_order_fact_id")
            .n_unique()
            .alias("candidate_unique_physical_orders"),
            supported.sum().alias("supported_outcome_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported)
            .n_unique()
            .alias("supported_unique_physical_orders"),
            (~supported).sum().alias("unsupported_unknown_aliases"),
            pl.col("raw_order_fact_id")
            .filter(~supported)
            .n_unique()
            .alias("unsupported_unique_physical_orders"),
            makerfill.sum().alias("makerfill_approx_aliases"),
            approx_full.sum().alias("makerfill_approx_full_aliases"),
            pl.col("raw_order_fact_id")
            .filter(approx_full)
            .n_unique()
            .alias("makerfill_approx_full_unique_physical"),
            approx_cancel.sum().alias("makerfill_approx_cancel_aliases"),
            pl.col("makerfill_fill_seconds")
            .filter(approx_full)
            .quantile(0.5)
            .alias("makerfill_approx_wait_seconds_p50"),
            pl.col("makerfill_fill_seconds")
            .filter(approx_full)
            .quantile(0.9)
            .alias("makerfill_approx_wait_seconds_p90"),
            supported_full.sum().alias("supported_full_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported_full)
            .n_unique()
            .alias("supported_full_fill_unique_physical"),
            supported_partial.sum().alias("supported_partial_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported_partial)
            .n_unique()
            .alias("supported_partial_fill_unique_physical"),
            supported_any.sum().alias("supported_any_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported_any)
            .n_unique()
            .alias("supported_any_fill_unique_physical"),
            supported_no_fill.sum().alias("supported_no_fill_aliases"),
            pl.col("raw_order_fact_id")
            .filter(supported_no_fill)
            .n_unique()
            .alias("supported_no_fill_unique_physical"),
            supported_cancel.sum().alias("supported_cancel_required_aliases"),
            wait_seconds
            .filter(supported_full)
            .quantile(0.5)
            .alias("supported_fill_wait_seconds_p50"),
            wait_seconds
            .filter(supported_full)
            .quantile(0.9)
            .alias("supported_fill_wait_seconds_p90"),
            estimate.sum().alias("hedge_estimate_available_aliases"),
            pl.col("hedge_estimate_executable_vwap_price")
            .filter(estimate)
            .median()
            .alias("hedge_estimate_vwap_p50"),
            pl.col("hedge_estimate_signed_total_slippage_bp")
            .filter(estimate)
            .quantile(0.5)
            .alias("hedge_estimate_total_slippage_bp_p50"),
            pl.col("hedge_estimate_signed_total_slippage_bp")
            .filter(estimate)
            .quantile(0.9)
            .alias("hedge_estimate_total_slippage_bp_p90"),
            pl.col("hedge_label_observed")
            .sum()
            .alias("exact_hedge_observed_aliases"),
            pl.col("hedge_executable")
            .sum()
            .alias("exact_hedge_executable_aliases"),
        )
        .with_columns(
            (
                pl.col("candidate_policy_aliases")
                - pl.col("candidate_unique_physical_orders")
            ).alias("policy_alias_rows_above_unique_physical"),
            (
                pl.col("makerfill_approx_full_aliases")
                / pl.col("makerfill_approx_aliases")
            ).fill_nan(None).alias("makerfill_approx_full_rate"),
            (
                pl.col("hedge_estimate_available_aliases")
                / pl.col("supported_full_fill_aliases")
            ).fill_nan(None).alias("hedge_estimate_coverage_given_full"),
            (
                pl.col("supported_full_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_full_fill_rate"),
            (
                pl.col("supported_partial_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_partial_fill_rate"),
            (
                pl.col("supported_any_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_any_fill_rate"),
            (
                pl.col("supported_no_fill_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_no_fill_rate"),
            (
                pl.col("supported_cancel_required_aliases")
                / pl.col("supported_outcome_aliases")
            ).fill_nan(None).alias("supported_cancel_required_rate"),
            pl.lit(True).alias("full_partial_no_fill_partition_exact"),
            pl.lit(True).alias("partial_is_cancel_required_subset"),
            pl.col("target_rank_at_submit")
            .is_in(sorted(_MAKER_FILL_COLUMNS))
            .alias("makerfill_primary_l1_l2_rank"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("joint_volume_allocated"),
        )
        .sort(groups)
    )
    return result


def _refresh_decision_maker_state(
    metadata: dict[str, object],
    cursor: EventCursor,
    index: _MarketStateIndex,
) -> None:
    row = index.latest(cursor)
    metadata["maker_snapshot_sequence"] = None
    metadata["maker_snapshot_recv_time_ns"] = None
    metadata["maker_snapshot_exact_at_decision"] = False
    target = _finite_float(metadata.get("absolute_target_price"))
    if row is None or target is None:
        metadata["target_rank"] = None
        metadata["initial_queue_ahead"] = None
        return
    rank, queue = _rank_and_queue(target, str(metadata["maker_side"]), row)
    metadata["target_rank"] = rank
    metadata["initial_queue_ahead"] = queue
    metadata["maker_snapshot_sequence"] = int(row["sequence"])
    metadata["maker_snapshot_recv_time_ns"] = int(row["recv_time_ns"])
    metadata["maker_snapshot_exact_at_decision"] = (
        index.market == "spot"
        and cursor.event_sequence == _SPOT_PRIORITY
        and cursor.recv_time_ns == int(row["recv_time_ns"])
        and cursor.row_index == int(row["sequence"])
    )


def _indexed_fill(
    replay: IndexedTradeReplay,
    window: IndependentOrderWindow,
    intended_quantity: int,
    *,
    fallback_reason: str,
) -> _CompactFill:
    quantity: QuantityFillLabel = replay.label_quantity(
        window, intended_quantity
    )
    return _CompactFill(
        backend="indexed_trade_replay",
        backend_reason=fallback_reason,
        makerfill_mapping_exact=False,
        outcome_exact=True,
        fill_cursor_exact=True,
        indexed_query_count=1,
        queue_known=quantity.queue_known,
        first_fill_cursor=quantity.first_fill_cursor,
        full_fill_cursor=quantity.full_fill_cursor,
        known_filled_quantity=quantity.known_filled_quantity_before_stop,
        any_fill=quantity.any_fill,
        full_fill=quantity.full_fill,
        partial_fill=quantity.partial_fill,
        trade_through_fill=quantity.trade_through_fill,
        makerfill_fill_seconds=None,
        makerfill_candidate_fill_ns=None,
    )


def _excluded_fill(reason: str) -> _CompactFill:
    return _CompactFill(
        backend="excluded_fail_closed",
        backend_reason=reason,
        makerfill_mapping_exact=False,
        outcome_exact=False,
        fill_cursor_exact=False,
        indexed_query_count=0,
        queue_known=False,
        first_fill_cursor=None,
        full_fill_cursor=None,
        known_filled_quantity=None,
        any_fill=None,
        full_fill=None,
        partial_fill=None,
        trade_through_fill=None,
        makerfill_fill_seconds=None,
        makerfill_candidate_fill_ns=None,
    )


def _select_fill(
    window: IndependentOrderWindow,
    metadata: Mapping[str, object],
    *,
    value_code: str,
    intended_quantity: int,
    config: CompactEvaluatorConfig,
    makerfill_adapter: MakerFillFastAdapter | None,
    replay: IndexedTradeReplay | None,
) -> _CompactFill:
    market = str(metadata.get("maker_market"))
    if config.fill_backend == "indexed":
        if replay is None:
            raise ValueError("indexed backend requires a maker trade index")
        return _indexed_fill(
            replay,
            window,
            intended_quantity,
            fallback_reason="indexed_backend_requested",
        )
    if market == "future":
        if replay is None:
            raise ValueError("future maker requires a maker trade index")
        return _indexed_fill(
            replay,
            window,
            intended_quantity,
            fallback_reason="future_maker_not_supported_by_makerfill",
        )
    rank = str(metadata.get("target_rank"))
    if rank not in _MAKER_FILL_COLUMNS:
        return _excluded_fill("spot_rank_outside_makerfill_l1_l2")
    if makerfill_adapter is None:
        return _excluded_fill("makerfill_adapter_unavailable")
    mapped = makerfill_adapter.label(
        window,
        metadata,
        value_code=value_code,
        intended_quantity=intended_quantity,
    )
    if mapped is None:
        return _excluded_fill("makerfill_snapshot_or_horizon_unusable")
    return mapped


def _outcome_record(
    *,
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
    quantile: int,
    window: IndependentOrderWindow,
    metadata: Mapping[str, object],
    fill: _CompactFill,
    intended_quantity: int,
    peak_active_layers: int,
) -> dict[str, object]:
    first = fill.first_fill_cursor
    full = fill.full_fill_cursor
    terminal = full or window.stop_cursor
    outcome_supported = fill.backend != "excluded_fail_closed"
    if not outcome_supported:
        terminal_reason = "unsupported_fill_outcome"
    elif fill.full_fill is True and fill.fill_cursor_exact:
        terminal_reason = "full_fill"
    elif fill.full_fill is True:
        terminal_reason = "approx_full_fill"
    elif fill.partial_fill is True:
        terminal_reason = f"partial_then_{window.stop_reason}"
    elif fill.full_fill is None:
        terminal_reason = f"unknown_queue_then_{window.stop_reason}"
    else:
        terminal_reason = window.stop_reason
    epoch = int(metadata["spread_pair_epoch"])
    raw_id = _compact_raw_id(
        date,
        value_code,
        quote_code,
        route,
        epoch,
        window.target_price_tick,
        window.start_cursor,
    )
    target_price = _finite_float(metadata.get("absolute_target_price"))
    if target_price is None:
        target_price = tick_index_to_price(
            window.target_price_tick,
            market=ROUTE_SPECS[route].maker_market,
            session_date=date,
        )
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "route": route,
        "maker_market": ROUTE_SPECS[route].maker_market,
        "maker_side": ROUTE_SPECS[route].maker_side,
        "boundary_quantile": quantile,
        "spread_pair_epoch": epoch,
        "raw_order_fact_id": raw_id,
        "policy_generation_id": window.generation_id,
        "target_price_tick": window.target_price_tick,
        "target_price": target_price,
        "target_rank_at_submit": metadata.get("target_rank"),
        "initial_queue_ahead": window.initial_queue_ahead,
        "queue_known": fill.queue_known,
        "intended_quantity": intended_quantity,
        "submit_recv_time_ns": window.start_cursor.recv_time_ns,
        "submit_event_sequence": window.start_cursor.event_sequence,
        "submit_row_index": window.start_cursor.row_index,
        "decision_reason": metadata.get("decision_reason"),
        "sampling_mode": "event_change_first_passage",
        "source_recv_time_ns": metadata.get("source_recv_time_ns"),
        "source_event_sequence": metadata.get("source_event_sequence"),
        "source_row_index": metadata.get("source_row_index"),
        "nominal_stop_recv_time_ns": window.stop_cursor.recv_time_ns,
        "nominal_stop_event_sequence": window.stop_cursor.event_sequence,
        "nominal_stop_row_index": window.stop_cursor.row_index,
        "nominal_stop_reason": window.stop_reason,
        "nominal_stop_type": metadata.get("first_passage_stop_type"),
        "first_fill_recv_time_ns": first.recv_time_ns if first else None,
        "first_fill_event_sequence": first.event_sequence if first else None,
        "first_fill_row_index": first.row_index if first else None,
        "full_fill_recv_time_ns": full.recv_time_ns if full else None,
        "full_fill_event_sequence": full.event_sequence if full else None,
        "full_fill_row_index": full.row_index if full else None,
        "known_filled_quantity": fill.known_filled_quantity,
        "any_fill": fill.any_fill,
        "full_fill": fill.full_fill,
        "partial_fill": fill.partial_fill,
        "trade_through_fill": fill.trade_through_fill,
        "terminal_recv_time_ns": terminal.recv_time_ns,
        "terminal_reason": terminal_reason,
        "cancel_required": (
            None if not outcome_supported else fill.full_fill is not True
        ),
        "outcome_supported": outcome_supported,
        "outcome_status": (
            "indexed_mbp_observed"
            if fill.backend == "indexed_trade_replay"
            else (
                "makerfill_eod_approximation"
                if fill.backend == "makerfill_fast"
                else "unsupported_unknown"
            )
        ),
        "fill_backend": fill.backend,
        "fill_backend_reason": fill.backend_reason,
        "makerfill_mapping_exact": fill.makerfill_mapping_exact,
        "fill_outcome_exact_within_model": fill.outcome_exact,
        "fill_cursor_exact": fill.fill_cursor_exact,
        "indexed_quantity_query_count": fill.indexed_query_count,
        "makerfill_fill_seconds": fill.makerfill_fill_seconds,
        "makerfill_candidate_fill_ns": fill.makerfill_candidate_fill_ns,
        "maker_snapshot_sequence": metadata.get("maker_snapshot_sequence"),
        "maker_snapshot_recv_time_ns": metadata.get(
            "maker_snapshot_recv_time_ns"
        ),
        "maker_snapshot_exact_at_decision": metadata.get(
            "maker_snapshot_exact_at_decision"
        ),
        "peak_nominal_layers_policy_day": peak_active_layers,
        "independent_event_label": True,
        "joint_volume_allocated": False,
        "pathwise_ev_ready": False,
    }


def _candidate_nominal_state_records(
    candidate: CompactCandidate,
    *,
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
    quantile: int,
    sampling_mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    observation = candidate.decision.observation
    generation_id = str(
        candidate.decision.metadata["policy_generation_id"]
    )
    common = {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "route": route,
        "boundary_quantile": quantile,
        "policy_generation_id": generation_id,
        "spread_pair_epoch": observation.spread_pair_epoch,
        "target_price_tick": observation.absolute_target_tick,
        "sampling_mode": sampling_mode,
        "fill_backend": None,
        "fill_cursor_exact": None,
        "outcome_supported": True,
    }
    submit = {
        **common,
        "change_kind": "submit",
        "change_reason": candidate.decision.decision_reason,
        "change_recv_time_ns": observation.cursor.recv_time_ns,
        "change_event_sequence": observation.cursor.event_sequence,
        "change_row_index": observation.cursor.row_index,
        "decision_reason": candidate.decision.decision_reason,
    }
    cancel = {
        **common,
        "change_kind": "cancel",
        "change_reason": candidate.stop_reason,
        "change_recv_time_ns": candidate.stop_cursor.recv_time_ns,
        "change_event_sequence": candidate.stop_cursor.event_sequence,
        "change_row_index": candidate.stop_cursor.row_index,
        "decision_reason": candidate.stop_reason,
    }
    return submit, cancel


def _peak_candidate_layers(candidates: tuple[CompactCandidate, ...]) -> int:
    events: list[tuple[EventCursor, int]] = []
    for candidate in candidates:
        events.append((candidate.decision.observation.cursor, 1))
        events.append((candidate.stop_cursor, -1))
    # Stops are effective at their first-passage cursor before any later
    # admission at that same cursor.
    events.sort(key=lambda item: (item[0], item[1]))
    active = peak = 0
    for _, delta in events:
        active += delta
        if active < 0:
            raise ValueError("candidate stop precedes its submit")
        peak = max(peak, active)
    if active != 0:
        raise ValueError("candidate layer sweep did not close at cutoff")
    return peak


def _attach_hedges(
    outcomes: pl.DataFrame,
    merged: MergedTargetStudyInput,
    *,
    hedge_delay_ns: int,
) -> pl.DataFrame:
    eligible_routes = set(ENTRY_ROUTES)
    aliases = outcomes.filter(pl.col("route").is_in(sorted(eligible_routes)))
    if aliases.is_empty():
        hedge = pl.DataFrame()
    else:
        raw_facts = summarize_raw_order_facts(aliases)
        hedge = run_hedge_study(
            aliases,
            raw_facts,
            merged.raw_tape,
            hedge_delay_ns=hedge_delay_ns,
        ).hedge_facts
    if hedge.is_empty():
        joined = outcomes.with_columns(*_empty_hedge_columns())
    else:
        selected = hedge.select(
            "raw_order_fact_id",
            pl.col("status").alias("hedge_estimate_status"),
            pl.col("decision_time_ns").alias("hedge_estimate_decision_time_ns"),
            pl.col("executable_vwap_price").alias(
                "hedge_estimate_executable_vwap_price"
            ),
            pl.col("executed_quantity").alias(
                "hedge_estimate_executed_quantity"
            ),
            pl.col("depth_shortfall").alias(
                "hedge_estimate_depth_shortfall"
            ),
            pl.col("decision_book_age_ms").alias(
                "hedge_estimate_decision_book_age_ms"
            ),
            pl.col("signed_latency_slippage_bp").alias(
                "hedge_estimate_signed_latency_slippage_bp"
            ),
            pl.col("signed_depth_slippage_bp").alias(
                "hedge_estimate_signed_depth_slippage_bp"
            ),
            pl.col("signed_total_slippage_bp").alias(
                "hedge_estimate_signed_total_slippage_bp"
            ),
        )
        joined = outcomes.join(
            selected,
            on="raw_order_fact_id",
            how="left",
            validate="m:1",
        )
    joined = joined.with_columns(
        (
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("hedge_estimate_status").is_not_null()
        ).fill_null(False).alias("hedge_estimate_available"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_status"))
        .otherwise(None)
        .alias("hedge_status"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_decision_time_ns"))
        .otherwise(None)
        .alias("hedge_decision_time_ns"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_executable_vwap_price"))
        .otherwise(None)
        .alias("hedge_executable_vwap_price"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_executed_quantity"))
        .otherwise(None)
        .alias("hedge_executed_quantity"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_depth_shortfall"))
        .otherwise(None)
        .alias("hedge_depth_shortfall"),
        pl.when(
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
        )
        .then(pl.col("hedge_estimate_decision_book_age_ms"))
        .otherwise(None)
        .alias("hedge_decision_book_age_ms"),
    ).with_columns(
        (
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
            & pl.col("hedge_status").is_not_null()
        ).fill_null(False).alias("hedge_label_observed"),
        (
            (pl.col("full_fill") == True)  # noqa: E712
            & pl.col("fill_cursor_exact")
            & (pl.col("hedge_status") == "executable")
        ).fill_null(False).alias("hedge_executable"),
        pl.when(pl.col("full_fill") == True)  # noqa: E712
        .then(pl.col("fill_cursor_exact"))
        .otherwise(None)
        .alias("hedge_input_fill_cursor_exact"),
    )
    return joined


def _materialize_actual_state_changes(
    nominal: pl.DataFrame,
    outcomes: pl.DataFrame,
) -> pl.DataFrame:
    """Replace post-fill nominal cancels and add partial/full terminals."""

    if outcomes.is_empty():
        return nominal
    records = nominal.to_dicts() if not nominal.is_empty() else []
    full_by_generation = {
        str(row["policy_generation_id"]): EventCursor(
            int(row["full_fill_recv_time_ns"]),
            int(row["full_fill_event_sequence"]),
            int(row["full_fill_row_index"]),
        )
        for row in outcomes.filter(pl.col("full_fill_recv_time_ns").is_not_null())
        .select(
            "policy_generation_id",
            "full_fill_recv_time_ns",
            "full_fill_event_sequence",
            "full_fill_row_index",
        )
        .iter_rows(named=True)
    }
    retained: list[dict[str, object]] = []
    for row in records:
        generation = row.get("policy_generation_id")
        if row["change_kind"] == "cancel" and generation is not None:
            full = full_by_generation.get(str(generation))
            cursor = EventCursor(
                int(row["change_recv_time_ns"]),
                int(row["change_event_sequence"]),
                int(row["change_row_index"]),
            )
            if full is not None and full <= cursor:
                continue
        retained.append(row)
    for row in outcomes.iter_rows(named=True):
        first_ns = row.get("first_fill_recv_time_ns")
        full_ns = row.get("full_fill_recv_time_ns")
        if first_ns is not None and (
            full_ns is None
            or (
                int(first_ns),
                int(row["first_fill_event_sequence"]),
                int(row["first_fill_row_index"]),
            )
            < (
                int(full_ns),
                int(row["full_fill_event_sequence"]),
                int(row["full_fill_row_index"]),
            )
        ):
            retained.append(
                _fill_state_record(row, "partial_fill", "first_partial_fill")
            )
        if full_ns is not None:
            retained.append(_fill_state_record(row, "full_fill", "full_fill"))
    if not retained:
        return _empty_state_changes()
    frame = pl.from_dicts(retained, infer_schema_length=None).sort(
        [
            "change_recv_time_ns",
            "change_event_sequence",
            "change_row_index",
            "route",
            "boundary_quantile",
            "policy_generation_id",
            "change_kind",
        ]
    )
    # Recompute actual active-layer count independently per policy.  Partial
    # fills remain active; full fills and cancels remove a generation once.
    output: list[dict[str, object]] = []
    for _, group in frame.group_by(
        ["Date", "ValueCode", "route", "boundary_quantile"],
        maintain_order=True,
    ):
        active: set[str] = set()
        for row in group.iter_rows(named=True):
            generation = row.get("policy_generation_id")
            if row["change_kind"] == "submit" and generation is not None:
                active.add(str(generation))
            elif row["change_kind"] in {
                "cancel",
                "full_fill",
                "approx_full_fill",
            } and generation is not None:
                active.discard(str(generation))
            row["active_layers_after"] = len(active)
            output.append(row)
    return pl.from_dicts(output, infer_schema_length=None).sort(
        [
            "Date",
            "ValueCode",
            "route",
            "boundary_quantile",
            "change_recv_time_ns",
            "change_event_sequence",
            "change_row_index",
            "policy_generation_id",
            "change_kind",
        ]
    )


def _fill_state_record(
    outcome: Mapping[str, object],
    kind: str,
    reason: str,
) -> dict[str, object]:
    prefix = "first_fill" if kind == "partial_fill" else "full_fill"
    exact = bool(outcome["fill_cursor_exact"])
    actual_kind = (
        "approx_full_fill" if kind == "full_fill" and not exact else kind
    )
    actual_reason = (
        "makerfill_approx_full_fill"
        if actual_kind == "approx_full_fill"
        else reason
    )
    return {
        "Date": outcome["Date"],
        "ValueCode": outcome["ValueCode"],
        "QuoteCode": outcome["QuoteCode"],
        "route": outcome["route"],
        "boundary_quantile": outcome["boundary_quantile"],
        "policy_generation_id": outcome["policy_generation_id"],
        "change_kind": actual_kind,
        "change_reason": actual_reason,
        "change_recv_time_ns": outcome[f"{prefix}_recv_time_ns"],
        "change_event_sequence": outcome[f"{prefix}_event_sequence"],
        "change_row_index": outcome[f"{prefix}_row_index"],
        "spread_pair_epoch": outcome["spread_pair_epoch"],
        "target_price_tick": outcome["target_price_tick"],
        "decision_reason": "fill_replay",
        "sampling_mode": outcome["sampling_mode"],
        "fill_backend": outcome["fill_backend"],
        "fill_cursor_exact": exact,
        "outcome_supported": outcome["outcome_supported"],
    }


def _validate_compact_result(
    outcomes: pl.DataFrame,
    state_changes: pl.DataFrame,
    audit: pl.DataFrame,
    config: CompactEvaluatorConfig,
) -> None:
    if not outcomes.is_empty():
        if outcomes.select("policy_generation_id").n_unique() != outcomes.height:
            raise ValueError("compact policy generation ids are duplicated")
        invalid_query_count = outcomes.filter(
            ~(
                (
                    (pl.col("fill_backend") == "indexed_trade_replay")
                    & (pl.col("indexed_quantity_query_count") == 1)
                )
                | (
                    (pl.col("fill_backend") == "makerfill_fast")
                    & (pl.col("indexed_quantity_query_count") == 0)
                )
                | (
                    (pl.col("fill_backend") == "excluded_fail_closed")
                    & (pl.col("indexed_quantity_query_count") == 0)
                )
            )
        )
        if invalid_query_count.height:
            raise ValueError("compact fill query-count contract is violated")
        invalid_makerfill = outcomes.filter(
            (pl.col("fill_backend") == "makerfill_fast")
            & (
                (~pl.col("makerfill_mapping_exact"))
                | pl.col("fill_outcome_exact_within_model")
                | pl.col("fill_cursor_exact")
            )
        )
        if invalid_makerfill.height:
            raise ValueError("makerFill approximation flags are incoherent")
        invalid_indexed = outcomes.filter(
            (pl.col("fill_backend") == "indexed_trade_replay")
            & (
                (~pl.col("fill_outcome_exact_within_model"))
                | (~pl.col("fill_cursor_exact"))
            )
        )
        if invalid_indexed.height:
            raise ValueError("indexed replay exactness flags are incoherent")
        invalid_excluded = outcomes.filter(
            (pl.col("fill_backend") == "excluded_fail_closed")
            & (
                pl.col("any_fill").is_not_null()
                | pl.col("full_fill").is_not_null()
                | pl.col("cancel_required").is_not_null()
                | pl.col("outcome_supported")
                | pl.col("makerfill_mapping_exact")
                | pl.col("fill_outcome_exact_within_model")
            )
        )
        if invalid_excluded.height:
            raise ValueError("fail-closed spot exclusions cannot carry outcomes")
        invalid_observed_hedge = outcomes.filter(
            pl.col("hedge_label_observed")
            & (~pl.col("fill_cursor_exact"))
        )
        if invalid_observed_hedge.height:
            raise ValueError(
                "observed hedge labels require an exact indexed fill cursor"
            )
        invalid_support = outcomes.filter(
            (
                (pl.col("fill_backend") == "excluded_fail_closed")
                == pl.col("outcome_supported")
            )
        )
        if invalid_support.height:
            raise ValueError("row-level supported denominator flag is incoherent")
        bad_delay = outcomes.filter(
            pl.col("hedge_label_observed")
            & (
                pl.col("hedge_decision_time_ns")
                != pl.col("full_fill_recv_time_ns") + config.hedge_delay_ns
            )
        )
        if bad_delay.height:
            raise ValueError("hedge decision is not exactly full_fill+delay")
    if not state_changes.is_empty():
        allowed = {
            "submit",
            "cancel",
            "partial_fill",
            "full_fill",
            "approx_full_fill",
        }
        actual = set(state_changes.get_column("change_kind").to_list())
        if not actual <= allowed:
            raise ValueError(f"unexpected compact state changes: {actual - allowed}")
        if state_changes.filter(pl.col("active_layers_after") < 0).height:
            raise ValueError("active layer count cannot be negative")
    if audit.is_empty():
        raise ValueError("compact evaluator must emit auditable policy rows")


def _policy_observation_count(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    route: str,
    boundary_quantile: int,
    cutoff: EventCursor,
) -> int:
    selected = frame.filter(
        (pl.col("Date").cast(pl.String) == date)
        & (pl.col("ValueCode").cast(pl.String) == value_code)
        & (pl.col("route") == route)
        & (pl.col("boundary_quantile") == boundary_quantile)
    )
    return sum(
        _row_cursor(row) < cutoff for row in selected.iter_rows(named=True)
    )


def _compact_raw_id(
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


def _row_cursor(row: Mapping[str, object]) -> EventCursor:
    return EventCursor(
        int(row["recv_time_ns"]),
        int(row["cursor_event_sequence"]),
        int(row["cursor_row_index"]),
    )


def _ceiling_boundary(value: int, interval: int) -> int:
    return ((value + interval - 1) // interval) * interval


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _require_columns(
    frame: pl.DataFrame, required: Iterable[str], source: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _frame(
    records: list[dict[str, object]],
    empty_factory,
) -> pl.DataFrame:
    return (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else empty_factory()
    )


def _normalize_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(_OUTCOME_BASE_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"compact outcome construction missing columns: {missing}")
    return frame.select(
        *[
            pl.col(column).cast(dtype, strict=False).alias(column)
            for column, dtype in _OUTCOME_BASE_SCHEMA.items()
        ]
    )


def _empty_hedge_columns() -> tuple[pl.Expr, ...]:
    return (
        pl.lit(None, dtype=pl.String).alias("hedge_estimate_status"),
        pl.lit(None, dtype=pl.Int64).alias("hedge_estimate_decision_time_ns"),
        pl.lit(None, dtype=pl.Float64).alias(
            "hedge_estimate_executable_vwap_price"
        ),
        pl.lit(None, dtype=pl.Int64).alias(
            "hedge_estimate_executed_quantity"
        ),
        pl.lit(None, dtype=pl.Int64).alias("hedge_estimate_depth_shortfall"),
        pl.lit(None, dtype=pl.Float64).alias(
            "hedge_estimate_decision_book_age_ms"
        ),
        pl.lit(None, dtype=pl.Float64).alias(
            "hedge_estimate_signed_latency_slippage_bp"
        ),
        pl.lit(None, dtype=pl.Float64).alias(
            "hedge_estimate_signed_depth_slippage_bp"
        ),
        pl.lit(None, dtype=pl.Float64).alias(
            "hedge_estimate_signed_total_slippage_bp"
        ),
    )


def _empty_outcomes() -> pl.DataFrame:
    return pl.DataFrame(schema=_OUTCOME_BASE_SCHEMA)


def _empty_state_changes() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "boundary_quantile": pl.Int64,
            "policy_generation_id": pl.String,
            "change_kind": pl.String,
            "change_reason": pl.String,
            "change_recv_time_ns": pl.Int64,
            "change_event_sequence": pl.Int64,
            "change_row_index": pl.Int64,
            "spread_pair_epoch": pl.Int64,
            "target_price_tick": pl.Int64,
            "decision_reason": pl.String,
            "sampling_mode": pl.String,
            "fill_backend": pl.String,
            "fill_cursor_exact": pl.Boolean,
            "outcome_supported": pl.Boolean,
            "active_layers_after": pl.Int64,
        }
    )
