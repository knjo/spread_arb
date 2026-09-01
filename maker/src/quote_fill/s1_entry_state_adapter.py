"""Actual-cursor S1 entry-state resolver over 1 Hz decisions and raw books."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Protocol

import polars as pl

from .layered import EventCursor
from .policy_spec import TOD_BUCKETS, PolicySpec
from .s1_day_state import S1_POLICY_DECISION_SIGNATURE_COLUMNS
from .s1_economic_gate import (
    UNGATED_CONTROL_RULE,
    S1EconomicGateEstimate,
    S1EconomicGateRule,
    evaluate_s1_entry_economics,
)
from .s1_event_loop import PHASE_ASSIGN, ActualSendMakerSnapshot, EntryObservation
from .s1_hedge import CausalBookState, executable_book
from .s1_scenario_spec import S1ScenarioSpec
from .s1_target import S1_ROUTE, actual_send_tod_bucket, build_s1_spot_bid_target
from .targets import price_in_ref_band

Venue = Literal["spot", "future"]

_DECISION_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "decision_time_ns",
    "analysis_eligible",
    "selected_anchor_bp",
    "contract_size",
)


class EntryBookProvider(Protocol):
    def state_as_of(
        self,
        venue: Venue,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None: ...

    def maker_snapshot_as_of(
        self,
        product_id: str,
        cursor: EventCursor,
    ) -> ActualSendMakerSnapshot | None: ...


@dataclass(frozen=True, slots=True)
class _PackedNullableColumn:
    values: object = field(repr=False)
    valid: object = field(repr=False)

    def value_at(self, position: int) -> object | None:
        if not bool(self.valid[position]):
            return None
        return _python_scalar(self.values[position])


@dataclass(frozen=True, slots=True)
class _PackedDecisionSeries:
    date: str
    value_code: str
    quote_code: str
    times: object = field(repr=False)
    analysis_eligible: _PackedNullableColumn = field(repr=False)
    anchor_basis_bp: _PackedNullableColumn = field(repr=False)
    contract_size: _PackedNullableColumn = field(repr=False)


@dataclass(frozen=True, slots=True)
class _DecisionTable:
    frame: pl.DataFrame = field(repr=False)
    spans: Mapping[str, tuple[int, int]] = field(repr=False)
    packed: Mapping[str, _PackedDecisionSeries] = field(repr=False)
    _last_positions: dict[str, int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _lookup_counts: dict[str, int] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=lambda: {"hits": 0, "misses": 0},
    )

    @classmethod
    def from_frame(cls, common_day: pl.DataFrame) -> _DecisionTable:
        missing = sorted(set(_DECISION_COLUMNS) - set(common_day.columns))
        if missing:
            raise ValueError(f"common decision day missing columns: {missing}")
        selected = common_day.select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("decision_time_ns").cast(pl.Int64),
            pl.col("analysis_eligible").cast(pl.Boolean),
            pl.col("selected_anchor_bp").cast(pl.Float64),
            pl.col("contract_size").cast(pl.Float64),
        )
        if selected.is_empty():
            raise ValueError("common decision day cannot be empty")
        invalid = selected.filter(
            pl.col("Date").is_null()
            | (pl.col("Date").str.len_chars() != 8)
            | pl.col("ValueCode").is_null()
            | (pl.col("ValueCode").str.len_chars() == 0)
            | pl.col("QuoteCode").is_null()
            | (pl.col("QuoteCode").str.len_chars() == 0)
            | pl.col("decision_time_ns").is_null()
        )
        if not invalid.is_empty():
            raise ValueError("common decision day has invalid identity or cursor rows")
        ordered = selected.sort("ValueCode", "decision_time_ns").rechunk()
        if ordered.select("ValueCode", "decision_time_ns").n_unique() != ordered.height:
            raise ValueError("common decision day keys are duplicated")
        spans: dict[str, tuple[int, int]] = {}
        offset = 0
        for product_id, length in (
            ordered.group_by("ValueCode", maintain_order=True).len().iter_rows()
        ):
            spans[product_id] = (offset, length)
            offset += length
        packed: dict[str, _PackedDecisionSeries] = {}
        for product_id, (offset, length) in spans.items():
            rows = ordered.slice(offset, length)
            dates = rows["Date"].unique().to_list()
            quote_codes = rows["QuoteCode"].unique().to_list()
            if len(dates) != 1 or len(quote_codes) != 1:
                raise ValueError("decision product identity changes within one span")
            packed[product_id] = _PackedDecisionSeries(
                date=str(dates[0]),
                value_code=product_id,
                quote_code=str(quote_codes[0]),
                times=_readonly_array(rows["decision_time_ns"]),
                analysis_eligible=_packed_nullable_column(
                    rows["analysis_eligible"], False
                ),
                anchor_basis_bp=_packed_nullable_column(
                    rows["selected_anchor_bp"], 0.0
                ),
                contract_size=_packed_nullable_column(rows["contract_size"], 0.0),
            )
        return cls(
            ordered,
            MappingProxyType(spans),
            MappingProxyType(packed),
        )

    def row_as_of(
        self,
        product_id: str,
        cursor: EventCursor,
    ) -> tuple[object, ...] | None:
        series = self.packed.get(product_id)
        if series is None:
            raise ValueError(f"unknown product_id: {product_id}")
        position = self._last_positions.get(product_id)
        if position is not None and _decision_position_contains(
            series,
            position,
            cursor.recv_time_ns,
        ):
            self._lookup_counts["hits"] += 1
        else:
            self._lookup_counts["misses"] += 1
            position = (
                int(series.times.searchsorted(cursor.recv_time_ns, side="right")) - 1
            )
            self._last_positions[product_id] = position
        if position < 0:
            return None
        return (
            series.date,
            series.value_code,
            series.quote_code,
            int(series.times[position]),
            series.analysis_eligible.value_at(position),
            series.anchor_basis_bp.value_at(position),
            series.contract_size.value_at(position),
        )

    @property
    def lookup_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._lookup_counts))


class S1EntryStateAdapter:
    """Resolve sparse desired observations and atomic actual-send state."""

    def __init__(
        self,
        common_day: pl.DataFrame,
        specs: Sequence[PolicySpec | S1ScenarioSpec],
        *,
        date: str,
        policy_id: str,
        book_provider: EntryBookProvider,
    ) -> None:
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError("date must be YYYYMMDD")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")
        if not hasattr(book_provider, "state_as_of") or not hasattr(
            book_provider,
            "maker_snapshot_as_of",
        ):
            raise TypeError("book_provider does not implement the entry book protocol")
        self.date = date
        self.policy_id = policy_id
        self.book_provider = book_provider
        self._decisions = _DecisionTable.from_frame(common_day)
        dates = self._decisions.frame["Date"].unique().to_list()
        if dates != [date]:
            raise ValueError("common decision day does not match adapter date")
        self._identity = {
            row[0]: row[1]
            for row in self._decisions.frame.select("ValueCode", "QuoteCode")
            .unique()
            .iter_rows()
        }
        if len(self._identity) != len(self._decisions.spans):
            raise ValueError("ValueCode to QuoteCode mapping is not one-to-one")
        values = tuple(specs)
        if not values or any(
            not isinstance(spec, (PolicySpec, S1ScenarioSpec)) for spec in values
        ):
            raise TypeError("specs must contain PolicySpec or S1ScenarioSpec values")
        self._specs: dict[
            tuple[str, str], PolicySpec | S1ScenarioSpec
        ] = {}
        self._economic_estimates: list[S1EconomicGateEstimate] = []
        for spec in values:
            if spec.Date != date or spec.policy_id != policy_id:
                raise ValueError("PolicySpec date/policy does not match adapter")
            if self._identity.get(spec.ValueCode) != spec.QuoteCode:
                raise ValueError("PolicySpec product mapping does not match common day")
            key = (spec.ValueCode, spec.entry_tod_bucket)
            if key in self._specs:
                raise ValueError("PolicySpec product/TOD cell is duplicated")
            self._specs[key] = spec
        expected = {
            (product_id, bucket)
            for product_id in self._identity
            for bucket in TOD_BUCKETS
        }
        if set(self._specs) != expected:
            raise ValueError("adapter requires exact four-TOD PolicySpec coverage")

    @property
    def product_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._identity))

    @property
    def decision_lookup_counts(self) -> Mapping[str, int]:
        return self._decisions.lookup_counts

    @property
    def economic_gate_estimates(self) -> tuple[S1EconomicGateEstimate, ...]:
        return tuple(self._economic_estimates)

    def current_state(
        self,
        product_id: str,
        assignment_cursor: EventCursor,
    ) -> EntryObservation | None:
        if not isinstance(assignment_cursor, EventCursor):
            raise TypeError("assignment_cursor must be an EventCursor")
        row = self._decisions.row_as_of(product_id, assignment_cursor)
        if row is None:
            return None
        (
            row_date,
            value_code,
            quote_code,
            _,
            analysis_eligible,
            anchor_basis_bp,
            contract_size,
        ) = row
        if row_date != self.date or value_code != product_id:
            raise RuntimeError("decision index identity diverged")
        try:
            _, tod_bucket = actual_send_tod_bucket(self.date, assignment_cursor)
        except ValueError:
            return None
        spec = self._specs[(product_id, tod_bucket)]
        if isinstance(spec, S1ScenarioSpec) and not spec.lookup_supported:
            return EntryObservation(
                assignment_cursor,
                product_id,
                None,
                None,
                None,
                False,
                False,
                "lookup_unsupported:" + spec.lookup_support_reason,
                None,
                None,
                None,
            )
        snapshot = self.book_provider.maker_snapshot_as_of(
            product_id,
            assignment_cursor,
        )
        if snapshot is not None and snapshot.value_code != product_id:
            raise ValueError("maker snapshot product identity mismatch")
        spot = self.book_provider.state_as_of(
            "spot",
            product_id,
            assignment_cursor,
        )
        future = self.book_provider.state_as_of(
            "future",
            product_id,
            assignment_cursor,
        )
        gate_reason, base_gate_book_wake_venues = _base_gate_reason(
            bool(analysis_eligible),
            anchor_basis_bp,
            contract_size,
            spot,
            future,
            assignment_cursor,
        )
        if gate_reason is not None:
            return EntryObservation(
                assignment_cursor,
                product_id,
                None,
                None,
                None,
                False,
                False,
                gate_reason,
                snapshot,
                None,
                base_gate_book_wake_venues=base_gate_book_wake_venues,
            )
        assert spot is not None and future is not None
        future_exec, future_reason = executable_book(
            future,
            side="sell",
            quantity=1,
            quantity_unit="future_contracts",
            send_eligible_cursor=assignment_cursor,
        )
        if future_exec is None:
            return EntryObservation(
                assignment_cursor,
                product_id,
                None,
                None,
                None,
                False,
                False,
                future_reason or "future_entry_book_closed",
                snapshot,
                None,
                base_gate_book_wake_venues=frozenset(("future",)),
            )
        future_exit_exec, future_exit_reason = executable_book(
            future,
            side="buy",
            quantity=1,
            quantity_unit="future_contracts",
            send_eligible_cursor=assignment_cursor,
        )
        if future_exit_exec is None:
            return EntryObservation(
                assignment_cursor,
                product_id,
                None,
                None,
                None,
                False,
                False,
                future_exit_reason or "future_exit_proxy_book_closed",
                snapshot,
                None,
                base_gate_book_wake_venues=frozenset(("future",)),
            )
        target = build_s1_spot_bid_target(
            spec,
            date=self.date,
            value_code=product_id,
            quote_code=str(quote_code),
            route=S1_ROUTE,
            actual_new_send_cursor=assignment_cursor,
            causal_anchor_basis_bp=float(anchor_basis_bp),
            fut_exec_bid=future_exec.executable_vwap,
            fut_exec_ask=future_exit_exec.executable_vwap,
            spot_bid=spot.bids[0].price,
            spot_ask=spot.asks[0].price,
            contract_size_shares=float(contract_size),
        )
        reference = spot.reference_price
        assert reference is not None
        base_open = target.passive_target and price_in_ref_band(
            target.target_price,
            float(reference),
        )
        if not base_open:
            reason = (
                "target_not_passive"
                if not target.passive_target
                else "target_outside_reference_band"
            )
            return EntryObservation(
                assignment_cursor,
                product_id,
                target.absolute_price_tick,
                target.target_price,
                None,
                False,
                False,
                reason,
                snapshot,
                target.frozen_exit_threshold_basis_bp,
                frozen_exit_target_price=target.frozen_exit_target_price,
                frozen_exit_absolute_price_tick=(
                    target.frozen_exit_absolute_price_tick
                ),
            )
        economics = evaluate_s1_entry_economics(
            target,
            observation_cursor=assignment_cursor,
            spot_reference_price=float(reference),
            rule=_economic_rule(spec),
            evaluation_stage=(
                "actual_send_refresh"
                if assignment_cursor.event_sequence == PHASE_ASSIGN
                else "decision_observation"
            ),
        )
        self._economic_estimates.append(economics)
        ab12 = _ab12_exact(target.target_price, snapshot, spot)
        admission = ab12 and economics.gate_open
        if not economics.gate_open:
            reason = "economic_gate:" + economics.reason
        elif not ab12:
            reason = "target_not_exact_raw_bid1_bid2"
        else:
            reason = "eligible"
        return EntryObservation(
            source_cursor=assignment_cursor,
            product_id=product_id,
            absolute_price_tick=target.absolute_price_tick,
            target_price=target.target_price,
            reservation_notional_twd=(
                target.reservation_notional_twd if admission else None
            ),
            base_gate_open=True,
            admission_open=admission,
            gate_reason=reason,
            maker_snapshot=snapshot,
            frozen_exit_threshold_basis_bp=(
                target.frozen_exit_threshold_basis_bp
            ),
            economic_estimate=economics,
            frozen_exit_target_price=target.frozen_exit_target_price,
            frozen_exit_absolute_price_tick=(
                target.frozen_exit_absolute_price_tick
            ),
        )

    def iter_observations(
        self,
        state_changes: pl.DataFrame,
    ) -> Iterator[EntryObservation]:
        """Resolve chronological sparse 1 Hz desired-state change cursors."""

        required = {"Date", "ValueCode", "decision_time_ns"}
        missing = sorted(required - set(state_changes.columns))
        if missing:
            raise ValueError(f"state changes missing columns: {missing}")
        signature_columns = set(S1_POLICY_DECISION_SIGNATURE_COLUMNS)
        present_signature_columns = signature_columns.intersection(
            state_changes.columns
        )
        if present_signature_columns and present_signature_columns != signature_columns:
            missing_signature = sorted(signature_columns - present_signature_columns)
            raise ValueError(
                "state changes contain a partial policy signature: "
                f"missing={missing_signature}"
            )
        has_policy_signature = present_signature_columns == signature_columns
        ordered = state_changes.select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("decision_time_ns").cast(pl.Int64),
            *(
                pl.col(column)
                for column in S1_POLICY_DECISION_SIGNATURE_COLUMNS
                if has_policy_signature
            ),
        ).sort("decision_time_ns", "ValueCode")
        previous: tuple[int, str] | None = None
        row_index_by_time: dict[int, int] = {}
        for row in ordered.iter_rows():
            row_date, product_id, timestamp_ns = row[:3]
            if row_date != self.date:
                raise ValueError("state changes contain another Date")
            if product_id not in self._identity:
                raise ValueError("state changes contain an unknown product")
            key = (timestamp_ns, product_id)
            if previous is not None and key <= previous:
                raise ValueError("state-change cursors are duplicated or regress")
            previous = key
            row_index = row_index_by_time.get(timestamp_ns, 0) + 1
            row_index_by_time[timestamp_ns] = row_index
            cursor = EventCursor(timestamp_ns, 10, row_index)
            state = self.current_state(product_id, cursor)
            if state is not None:
                yield replace(
                    state,
                    policy_state_signature=(
                        tuple(row[3:]) if has_policy_signature else None
                    ),
                )


def _economic_rule(
    spec: PolicySpec | S1ScenarioSpec,
) -> S1EconomicGateRule:
    if isinstance(spec, PolicySpec):
        return UNGATED_CONTROL_RULE
    return S1EconomicGateRule(
        cost_horizon=spec.cost_horizon,
        safety_floor_bp=spec.safety_floor_bp,
        economic_gate_enabled=spec.economic_gate_enabled,
        deployment_shortlist_eligible=spec.deployment_shortlist_eligible,
    )


def merge_entry_events(
    observations: Iterable[EntryObservation],
    other_events: Iterable[object],
) -> Iterator[object]:
    """Merge two already-sorted streams without materializing either input."""

    left = iter(observations)
    right = iter(other_events)
    left_value = next(left, None)
    right_value = next(right, None)
    while left_value is not None or right_value is not None:
        if right_value is None or (
            left_value is not None
            and _event_cursor(left_value) <= _event_cursor(right_value)
        ):
            yield left_value
            left_value = next(left, None)
        else:
            yield right_value
            right_value = next(right, None)


def _base_gate_reason(
    analysis_eligible: bool,
    anchor_basis_bp: object,
    contract_size: object,
    spot: CausalBookState | None,
    future: CausalBookState | None,
    cursor: EventCursor,
) -> tuple[str | None, frozenset[Venue]]:
    if not analysis_eligible:
        return "analysis_ineligible", frozenset()
    if not _finite(anchor_basis_bp):
        return "invalid_anchor", frozenset()
    if not _positive_integral(contract_size):
        return "invalid_contract_size", frozenset()
    for venue, state in (("spot", spot), ("future", future)):
        if state is None:
            return f"missing_{venue}_book", frozenset((venue,))
        if state.book_cursor.cursor > cursor:
            raise ValueError(f"{venue} book cannot follow the assignment cursor")
        if not state.gate_open:
            return (
                state.gate_reason or f"{venue}_gate_closed",
                frozenset((venue,)),
            )
        reference = state.reference_price
        if not _finite_positive(reference):
            return f"invalid_{venue}_reference", frozenset((venue,))
        if not state.bids or not state.asks:
            return f"empty_{venue}_book", frozenset((venue,))
        if state.bids[0].price > state.asks[0].price:
            return f"crossed_{venue}_book", frozenset((venue,))
        if not (
            price_in_ref_band(state.bids[0].price, float(reference))
            and price_in_ref_band(state.asks[0].price, float(reference))
        ):
            return (
                f"{venue}_bbo_outside_reference_band",
                frozenset((venue,)),
            )
    return None, frozenset()


def _ab12_exact(
    target_price: float,
    snapshot: ActualSendMakerSnapshot | None,
    spot: CausalBookState,
) -> bool:
    if snapshot is None:
        return False
    raw_levels = (
        (snapshot.bid_price1, snapshot.bid_lots1),
        (snapshot.bid_price2, snapshot.bid_lots2),
    )
    raw_match = any(
        price is not None
        and lots is not None
        and lots > 0
        and math.isclose(
            target_price,
            price,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        for price, lots in raw_levels
    )
    canonical_match = any(
        math.isclose(
            target_price,
            level.price,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        for level in spot.bids[:2]
    )
    return raw_match and canonical_match


def _event_cursor(value: object) -> EventCursor:
    if isinstance(value, EntryObservation):
        return value.source_cursor
    source = getattr(value, "source_cursor", None)
    if isinstance(source, EventCursor):
        return source
    event = getattr(value, "event", None)
    book_cursor = getattr(event, "book_cursor", None)
    cursor = getattr(book_cursor, "cursor", None)
    if isinstance(cursor, EventCursor):
        return cursor
    raise TypeError("merged event has no EventCursor")


def _readonly_array(series: pl.Series) -> object:
    values = series.to_numpy()
    values.setflags(write=False)
    return values


def _packed_nullable_column(
    series: pl.Series,
    null_fill: object,
) -> _PackedNullableColumn:
    return _PackedNullableColumn(
        _readonly_array(series.fill_null(null_fill)),
        _readonly_array(series.is_not_null()),
    )


def _python_scalar(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _decision_position_contains(
    series: _PackedDecisionSeries,
    position: int,
    timestamp_ns: int,
) -> bool:
    if position < 0:
        return len(series.times) == 0 or int(series.times[0]) > timestamp_ns
    if int(series.times[position]) > timestamp_ns:
        return False
    next_position = position + 1
    return (
        next_position >= len(series.times)
        or int(series.times[next_position]) > timestamp_ns
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _finite_positive(value: object) -> bool:
    return _finite(value) and float(value) > 0


def _positive_integral(value: object) -> bool:
    return _finite_positive(value) and math.isclose(
        float(value), round(float(value)), abs_tol=1e-9
    )


__all__ = ["EntryBookProvider", "S1EntryStateAdapter", "merge_entry_events"]
