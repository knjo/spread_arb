"""Strict raw-tape labels for positions carried beyond the entry session.

This module deliberately starts from an already established position and an
exit-policy fact whose strict branch is ``carry_at_eod_*``.  It does not turn
``cancel_race_unknown`` (or any other unknown branch) into a carry position.

The ordinary overnight benchmark crosses the first jointly executable raw
spot bid and futures ask in a later session.  Both books are selected in one
causal merged receive-time order and the futures leg must retain the entry's
exact ``QuoteCode``.  A different contract is evidence that an explicit roll
would be required; it is never substituted automatically.

The two taker legs use the same causal cursor with zero added execution
latency, so this is explicitly an optimistic benchmark rather than a latency-
matched execution replay.  A missing candidate-session tape is a terminal
censor: the labeler never skips an unobserved day and resumes later.

Expiry is a separate branch.  It can be priced only from a caller-supplied,
final settlement fact plus an executable spot unwind.  No settlement price is
estimated here.  Roll cashflows, fees, taxes, financing and carrying costs are
also outside this labeler, so every output remains ``pathwise_ev_ready=False``
even when its gross terminal cashflow is known.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal, Mapping, Sequence

import polars as pl

from .ev_surface import DEFAULT_COST_COLUMNS
from .hedge_study import executable_levels_from_state
from .layered import EventCursor
from .raw_tape import RawTapeDay


STRICT_CARRY_BRANCHES = frozenset(
    {"carry_at_eod_cancel_unconfirmed", "carry_at_eod_no_admission"}
)

OVERNIGHT_EXIT = "overnight_exit"
EXPIRY_SETTLEMENT = "expiry_settlement"

_MARKET_PRIORITY: Mapping[str, int] = {"future": 1, "spot": 2}

_POLICY_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "position_status",
    "position_established_ns",
    "branch_status",
    "needs_next_session_label",
}

_ENTRY_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "policy_generation_id",
    "raw_order_fact_id",
    "full_fill",
    "entry_hedge_status",
    "entry_hedge_contract_size_shares",
    "entry_spot_price",
    "entry_future_price",
}

_CALENDAR_REQUIRED = {
    "QuoteCode",
    "expiry_session",
    "calendar_version",
}

_SETTLEMENT_REQUIRED = {
    "QuoteCode",
    "expiry_session",
    "settlement_price",
    "settlement_time_ns",
    "settlement_source_version",
    "settlement_final",
}

_STATE_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "recv_time_ns",
    "sequence",
    "trial_match",
    "raw_has_book",
    "book_state_available",
    "book_recv_time_ns",
    *(f"{side}_price_{level}" for side in ("bid", "ask") for level in range(1, 6)),
    *(f"{side}_lots_{level}" for side in ("bid", "ask") for level in range(1, 6)),
}


@dataclass(frozen=True)
class OvernightCarryConfig:
    """Frozen gross-close benchmark semantics.

    ``max_carry_sessions`` is an explicit research horizon, not a hidden
    holding-period assumption.  Sessions after the exact contract's expiry
    are never searched.  ``max_book_age_ns=None`` records book age without
    imposing an arbitrary freshness cutoff; callers can freeze a finite bound
    in a later policy version.
    """

    max_carry_sessions: int = 1
    max_book_age_ns: int | None = None
    spot_board_lot_shares: int = 1_000
    futures_contracts: int = 1
    added_exit_latency_ns: int = 0
    policy_version: str = "first_joint_raw_taker_close_exact_contract_v1"
    calendar_required: bool = True

    def validate(self) -> None:
        for name in ("max_carry_sessions", "spot_board_lot_shares", "futures_contracts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.futures_contracts != 1:
            raise ValueError(
                "v1 supports exactly one futures contract; quantity scaling is not implemented"
            )
        if (
            isinstance(self.added_exit_latency_ns, bool)
            or not isinstance(self.added_exit_latency_ns, int)
            or self.added_exit_latency_ns != 0
        ):
            raise ValueError(
                "v1 is an optimistic same-cursor zero-added-latency benchmark"
            )
        if self.max_book_age_ns is not None and (
            isinstance(self.max_book_age_ns, bool)
            or not isinstance(self.max_book_age_ns, int)
            or self.max_book_age_ns < 0
        ):
            raise ValueError("max_book_age_ns must be a non-negative integer or None")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        if self.calendar_required is not True:
            raise ValueError(
                "calendar_required must remain true: expiry cannot be inferred safely"
            )


@dataclass(frozen=True)
class OvernightCarryResult:
    """Strict carry labels plus one-row branch/support audit."""

    labels: pl.DataFrame
    audit: pl.DataFrame


@dataclass
class _FormalMarketState:
    row: dict[str, object] | None = None
    formal_after_trial: bool = False
    last_trial_match: bool | None = None

    def update(self, row: dict[str, object]) -> None:
        trial = bool(row["trial_match"])
        raw_book = bool(row["raw_has_book"])
        if trial:
            self.formal_after_trial = False
        elif self.last_trial_match is True:
            self.formal_after_trial = raw_book
        elif not self.formal_after_trial and raw_book:
            self.formal_after_trial = True
        self.last_trial_match = trial
        self.row = row


@dataclass(frozen=True)
class _ExecutableClose:
    date: str
    decision_cursor: EventCursor
    spot_cursor: EventCursor
    future_cursor: EventCursor | None
    spot_price: float
    future_price: float | None
    spot_levels_swept: int
    future_levels_swept: int | None
    spot_book_age_ms: float
    future_book_age_ms: float | None


def build_strict_overnight_carry_labels(
    position_policy_facts: pl.DataFrame,
    entry_action_facts: pl.DataFrame,
    raw_tapes: RawTapeDay | Iterable[RawTapeDay],
    sessions: Sequence[str] | Iterable[str],
    contract_calendar: pl.DataFrame,
    *,
    settlement_facts: pl.DataFrame | None = None,
    config: OvernightCarryConfig = OvernightCarryConfig(),
) -> OvernightCarryResult:
    """Price strict carry branches without inventing roll or settlement.

    The returned unit is one ``exit_policy_trial_id``.  Exit rules/routes are
    alternative policies and remain separate even when they share the same
    physical entry and therefore the same overnight close observation.

    Unknown exit-maker branches are excluded, not relabelled.  A caller that
    wants the nominal instant-cancel V0 population must first materialize it as
    a separate policy fact table; mixing nominal and strict branch columns in
    this function is intentionally unsupported.
    """

    config.validate()
    _require(position_policy_facts, _POLICY_REQUIRED, "position policy facts")
    _require(entry_action_facts, _ENTRY_REQUIRED, "entry action facts")
    _require(contract_calendar, _CALENDAR_REQUIRED, "contract calendar")
    sessions_tuple = _normalise_sessions(sessions)
    tapes = _normalise_tapes(raw_tapes)
    calendar = _normalise_calendar(contract_calendar)
    settlements = _normalise_settlements(settlement_facts)

    if position_policy_facts.select("exit_policy_trial_id").n_unique() != position_policy_facts.height:
        raise ValueError("position policy facts must be unique by exit_policy_trial_id")
    if entry_action_facts.select("policy_generation_id").n_unique() != entry_action_facts.height:
        raise ValueError("entry action facts must be unique by policy_generation_id")

    carry = position_policy_facts.filter(
        pl.col("branch_status").is_in(sorted(STRICT_CARRY_BRANCHES))
    )
    if carry.is_empty():
        labels = pl.DataFrame(schema=_label_schema())
        return OvernightCarryResult(labels, _audit(position_policy_facts, carry, labels))
    invalid_carry = carry.filter(
        (pl.col("position_status") != "position_established")
        | ~pl.col("needs_next_session_label").fill_null(False)
    )
    if invalid_carry.height:
        raise ValueError(
            "strict carry rows must be established positions requiring a next-session label"
        )

    joined = carry.join(
        entry_action_facts,
        left_on="entry_policy_generation_id",
        right_on="policy_generation_id",
        how="left",
        suffix="_entry",
        validate="m:1",
    )
    _validate_entry_join(joined)
    dependency_keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_raw_order_fact_id",
    ]
    _validate_physical_entry_dependencies(joined, dependency_keys)
    joined = joined.with_columns(
        pl.len()
        .over(dependency_keys)
        .cast(pl.Int64)
        .alias("_physical_entry_strict_carry_policy_alias_count")
    )

    session_index = {date: index for index, date in enumerate(sessions_tuple)}
    unknown_dates = sorted(set(str(value) for value in joined["Date"]) - set(session_index))
    if unknown_dates:
        raise ValueError(f"position dates absent from session calendar: {unknown_dates[:5]}")

    tape_by_date = {str(tape.date): tape for tape in tapes}
    records: list[dict[str, object]] = []
    close_cache: dict[tuple[object, ...], tuple[str, _ExecutableClose | None, str | None]] = {}
    spot_cache: dict[tuple[object, ...], tuple[str, _ExecutableClose | None, str | None]] = {}
    for row in joined.iter_rows(named=True):
        quote_code = str(row["QuoteCode"])
        calendar_row = calendar.get(quote_code)
        if calendar_row is None:
            raise ValueError(f"missing contract calendar row for exact QuoteCode {quote_code}")
        date = str(row["Date"])
        expiry = str(calendar_row["expiry_session"])
        if date > expiry:
            record = _base_record(row, calendar_row, config)
            record.update(
                _unresolved_fields(
                    status="invalid_position_after_expiry",
                    terminal_branch=None,
                    transition_branch="invalid_after_expiry",
                    reason="entry_date_after_exact_contract_expiry",
                )
            )
            records.append(record)
            continue

        future_dates = _future_sessions(
            date,
            sessions_tuple,
            session_index,
            config.max_carry_sessions,
        )
        if date == expiry:
            settlement = settlements.get((quote_code, expiry))
            records.append(
                _label_expiry_path(
                    row,
                    calendar_row,
                    settlement,
                    future_dates,
                    tape_by_date,
                    spot_cache,
                    config,
                )
            )
            continue

        if not future_dates:
            record = _base_record(row, calendar_row, config)
            record.update(
                _unresolved_fields(
                    status="unresolved_no_next_session",
                    terminal_branch=None,
                    transition_branch="observation_horizon_censor",
                    reason="no_later_session_in_supplied_calendar",
                )
            )
            records.append(record)
            continue

        cache_key = (
            date,
            str(row["ValueCode"]),
            quote_code,
            int(row["entry_hedge_contract_size_shares"]),
            tuple(future_dates),
            config.max_book_age_ns,
        )
        if cache_key not in close_cache:
            close_cache[cache_key] = _search_exact_contract_close(
                value_code=str(row["ValueCode"]),
                quote_code=quote_code,
                contract_size_shares=int(row["entry_hedge_contract_size_shares"]),
                candidate_dates=future_dates,
                expiry_session=expiry,
                tape_by_date=tape_by_date,
                config=config,
            )
        status, close, reason = close_cache[cache_key]
        if close is None and status == "expiry_settlement_required":
            settlement = settlements.get((quote_code, expiry))
            records.append(
                _label_expiry_path(
                    row,
                    calendar_row,
                    settlement,
                    tuple(value for value in future_dates if value > expiry),
                    tape_by_date,
                    spot_cache,
                    config,
                )
            )
            continue
        if close is None:
            record = _base_record(row, calendar_row, config)
            transition = (
                "roll_substitution_forbidden"
                if status == "roll_substitution_forbidden"
                else "same_exact_contract_unresolved"
            )
            record.update(
                _unresolved_fields(
                    status=status,
                    terminal_branch=None,
                    transition_branch=transition,
                    reason=reason,
                )
            )
            records.append(record)
            continue
        records.append(
            _priced_record(
                row,
                calendar_row,
                close,
                status="overnight_exit",
                terminal_branch=OVERNIGHT_EXIT,
                transition_branch="same_exact_contract",
                settlement=None,
                config=config,
            )
        )

    labels = _from_records(records, _label_schema()).sort(
        ["Date", "ValueCode", "entry_route", "exit_rule_id", "exit_route", "exit_policy_trial_id"]
    )
    if labels.select("carry_label_id").n_unique() != labels.height:
        raise ValueError("duplicate carry_label_id generated")
    return OvernightCarryResult(labels, _audit(position_policy_facts, carry, labels))


def _label_expiry_path(
    row: Mapping[str, object],
    calendar_row: Mapping[str, object],
    settlement: Mapping[str, object] | None,
    future_dates: tuple[str, ...],
    tape_by_date: Mapping[str, RawTapeDay],
    cache: dict[tuple[object, ...], tuple[str, _ExecutableClose | None, str | None]],
    config: OvernightCarryConfig,
) -> dict[str, object]:
    base = _base_record(row, calendar_row, config)
    if settlement is None:
        base.update(
            _unresolved_fields(
                status="expiry_settlement_unpriced",
                terminal_branch=None,
                transition_branch="expiry_settlement_required",
                reason="no_final_exact_contract_settlement_fact",
            )
        )
        return base
    if settlement.get("settlement_final") is not True:
        raise ValueError("settlement facts used for pricing must be final")
    if not future_dates:
        base.update(
            _unresolved_fields(
                status="expiry_settlement_missing_spot_exit",
                terminal_branch=None,
                transition_branch="expiry_settlement",
                reason="settlement_known_but_no_later_spot_session_in_calendar",
            )
        )
        return base

    cache_key = (
        str(row["Date"]),
        str(row["ValueCode"]),
        int(row["entry_hedge_contract_size_shares"]),
        tuple(future_dates),
        config.max_book_age_ns,
        int(settlement["settlement_time_ns"]),
    )
    if cache_key not in cache:
        cache[cache_key] = _search_spot_close(
            value_code=str(row["ValueCode"]),
            contract_size_shares=int(row["entry_hedge_contract_size_shares"]),
            candidate_dates=future_dates,
            tape_by_date=tape_by_date,
            earliest_book_time_ns=int(settlement["settlement_time_ns"]),
            config=config,
        )
    status, spot_close, reason = cache[cache_key]
    if spot_close is None:
        base.update(
            _unresolved_fields(
                status="expiry_settlement_missing_spot_exit",
                terminal_branch=None,
                transition_branch="expiry_settlement",
                reason=reason or status,
            )
        )
        return base
    settlement_price = float(settlement["settlement_price"])
    settlement_time_ns = int(settlement["settlement_time_ns"])
    combined = _ExecutableClose(
        date=spot_close.date,
        decision_cursor=EventCursor(
            max(spot_close.decision_cursor.recv_time_ns, settlement_time_ns),
            spot_close.decision_cursor.event_sequence,
            spot_close.decision_cursor.row_index,
        ),
        spot_cursor=spot_close.spot_cursor,
        future_cursor=EventCursor(settlement_time_ns, 0, 0),
        spot_price=spot_close.spot_price,
        future_price=settlement_price,
        spot_levels_swept=spot_close.spot_levels_swept,
        future_levels_swept=None,
        spot_book_age_ms=spot_close.spot_book_age_ms,
        future_book_age_ms=None,
    )
    return _priced_record(
        row,
        calendar_row,
        combined,
        status="expiry_settlement",
        terminal_branch=EXPIRY_SETTLEMENT,
        transition_branch="expiry_settlement",
        settlement=settlement,
        config=config,
    )


def _search_exact_contract_close(
    *,
    value_code: str,
    quote_code: str,
    contract_size_shares: int,
    candidate_dates: tuple[str, ...],
    expiry_session: str,
    tape_by_date: Mapping[str, RawTapeDay],
    config: OvernightCarryConfig,
) -> tuple[str, _ExecutableClose | None, str | None]:
    no_joint_book = False
    reached_expiry = False
    for date in candidate_dates:
        if date > expiry_session:
            return (
                "expiry_settlement_required",
                None,
                "search_horizon_crossed_exact_contract_expiry",
            )
        reached_expiry |= date == expiry_session
        tape = tape_by_date.get(date)
        if tape is None:
            return (
                "unresolved_missing_raw_tape",
                None,
                "first_missing_candidate_session_censors_path; later_sessions_not_examined",
            )
        spot = tape.spot_states.filter(
            (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
        )
        future = tape.future_states.filter(
            (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
        )
        if future.is_empty():
            different_contract_present = tape.future_states.filter(
                (pl.col("ValueCode").cast(pl.String) == value_code)
                & (pl.col("QuoteCode").cast(pl.String) != quote_code)
            ).height > 0
            if date == expiry_session:
                return (
                    "expiry_settlement_required",
                    None,
                    "exact_contract_absent_on_expiry_session; settlement_fact_required",
                )
            if different_contract_present:
                return (
                    "roll_substitution_forbidden",
                    None,
                    "first_candidate_session_has_only_a_different_QuoteCode; later_sessions_not_examined",
                )
            return (
                "unresolved_exact_contract_absent",
                None,
                "entry_QuoteCode_absent_from_first_candidate_session; later_sessions_not_examined",
            )
        if spot.is_empty():
            return (
                "unresolved_spot_tape_absent",
                None,
                "spot_instrument_absent_from_first_candidate_session; later_sessions_not_examined",
            )
        close = _first_joint_close(
            date,
            value_code,
            quote_code,
            spot,
            future,
            contract_size_shares,
            config,
        )
        if close is not None:
            return "overnight_exit", close, None
        no_joint_book = True
    if reached_expiry:
        return (
            "expiry_settlement_required",
            None,
            "no_exact_contract_taker_close_before_expiry_settlement",
        )
    if no_joint_book:
        return (
            "unresolved_no_joint_executable_book",
            None,
            "exact_contract_present_but_no_joint_executable_raw_book",
        )
    return "unresolved_no_joint_executable_book", None, "no_executable_close_observed"


def _search_spot_close(
    *,
    value_code: str,
    contract_size_shares: int,
    candidate_dates: tuple[str, ...],
    tape_by_date: Mapping[str, RawTapeDay],
    earliest_book_time_ns: int,
    config: OvernightCarryConfig,
) -> tuple[str, _ExecutableClose | None, str | None]:
    for candidate_index, date in enumerate(candidate_dates):
        tape = tape_by_date.get(date)
        if tape is None:
            return (
                "unresolved_missing_raw_tape",
                None,
                "first_missing_candidate_spot_session_censors_path; later_sessions_not_examined",
            )
        spot = tape.spot_states.filter(pl.col("ValueCode").cast(pl.String) == value_code)
        if spot.is_empty():
            return (
                "unresolved_spot_tape_absent",
                None,
                "spot_instrument_absent_from_first_candidate_session; later_sessions_not_examined",
            )
        close = _first_spot_close(
            date,
            value_code,
            spot,
            contract_size_shares,
            earliest_book_time_ns=(
                earliest_book_time_ns if candidate_index == 0 else 0
            ),
            config=config,
        )
        if close is not None:
            return "expiry_settlement", close, None
    return (
        "unresolved_no_executable_spot_book",
        None,
        "no_executable_spot_bid_for_settled_position",
    )


def _first_joint_close(
    date: str,
    value_code: str,
    quote_code: str,
    spot: pl.DataFrame,
    future: pl.DataFrame,
    contract_size_shares: int,
    config: OvernightCarryConfig,
) -> _ExecutableClose | None:
    _validate_state_frame(spot, "spot", date, value_code, quote_code)
    _validate_state_frame(future, "future", date, value_code, quote_code)
    spot_lots = _spot_lots(contract_size_shares, config.spot_board_lot_shares)
    events: list[tuple[EventCursor, Literal["future", "spot"], dict[str, object]]] = []
    for market, frame in (("future", future), ("spot", spot)):
        priority = _MARKET_PRIORITY[market]
        for row in frame.iter_rows(named=True):
            events.append((_cursor(row, priority), market, row))
    events.sort(key=lambda value: value[0])
    states = {"future": _FormalMarketState(), "spot": _FormalMarketState()}
    for cursor, market, row in events:
        states[market].update(row)
        spot_row = states["spot"].row
        future_row = states["future"].row
        if spot_row is None or future_row is None:
            continue
        if not _state_eligible(states["spot"], cursor.recv_time_ns, config.max_book_age_ns):
            continue
        if not _state_eligible(states["future"], cursor.recv_time_ns, config.max_book_age_ns):
            continue
        if not _book_sane(spot_row, "spot") or not _book_sane(future_row, "future"):
            continue
        spot_levels = executable_levels_from_state(spot_row, "spot", "bid")
        future_levels = executable_levels_from_state(future_row, "future", "ask")
        spot_vwap = _vwap(spot_levels, spot_lots)
        future_vwap = _vwap(future_levels, config.futures_contracts)
        if spot_vwap is None or future_vwap is None:
            continue
        spot_price, spot_swept = spot_vwap
        future_price, future_swept = future_vwap
        spot_cursor = _cursor(spot_row, _MARKET_PRIORITY["spot"])
        future_cursor = _cursor(future_row, _MARKET_PRIORITY["future"])
        return _ExecutableClose(
            date=date,
            decision_cursor=cursor,
            spot_cursor=spot_cursor,
            future_cursor=future_cursor,
            spot_price=spot_price,
            future_price=future_price,
            spot_levels_swept=spot_swept,
            future_levels_swept=future_swept,
            spot_book_age_ms=_book_age_ms(cursor.recv_time_ns, spot_row),
            future_book_age_ms=_book_age_ms(cursor.recv_time_ns, future_row),
        )
    return None


def _first_spot_close(
    date: str,
    value_code: str,
    spot: pl.DataFrame,
    contract_size_shares: int,
    earliest_book_time_ns: int,
    config: OvernightCarryConfig,
) -> _ExecutableClose | None:
    _require(spot, _STATE_REQUIRED, "raw spot state")
    for column, expected in (("Date", date), ("ValueCode", value_code)):
        values = set(str(value) for value in spot[column].unique())
        if values != {expected}:
            raise ValueError(
                f"raw spot state is not exact on {column}: {values} != {{{expected}}}"
            )
    spot_lots = _spot_lots(contract_size_shares, config.spot_board_lot_shares)
    state = _FormalMarketState()
    ordered = spot.sort(["recv_time_ns", "sequence"])
    for row in ordered.iter_rows(named=True):
        cursor = _cursor(row, _MARKET_PRIORITY["spot"])
        state.update(row)
        if cursor.recv_time_ns < earliest_book_time_ns:
            continue
        book_recv_time_ns = row.get("book_recv_time_ns")
        if (
            book_recv_time_ns is None
            or int(book_recv_time_ns) < earliest_book_time_ns
        ):
            # A later trade/event can carry a forward-filled pre-settlement
            # book.  That stale quote is not a post-settlement spot unwind.
            continue
        if not _state_eligible(state, cursor.recv_time_ns, config.max_book_age_ns):
            continue
        if not _book_sane(row, "spot"):
            continue
        levels = executable_levels_from_state(row, "spot", "bid")
        result = _vwap(levels, spot_lots)
        if result is None:
            continue
        price, swept = result
        return _ExecutableClose(
            date=date,
            decision_cursor=cursor,
            spot_cursor=cursor,
            future_cursor=None,
            spot_price=price,
            future_price=None,
            spot_levels_swept=swept,
            future_levels_swept=None,
            spot_book_age_ms=_book_age_ms(cursor.recv_time_ns, row),
            future_book_age_ms=None,
        )
    return None


def _priced_record(
    row: Mapping[str, object],
    calendar_row: Mapping[str, object],
    close: _ExecutableClose,
    *,
    status: str,
    terminal_branch: str,
    transition_branch: str,
    settlement: Mapping[str, object] | None,
    config: OvernightCarryConfig,
) -> dict[str, object]:
    assert close.future_price is not None
    shares = int(row["entry_hedge_contract_size_shares"])
    entry_spot = float(row["entry_spot_price"])
    entry_future = float(row["entry_future_price"])
    gross = shares * (
        (close.spot_price - entry_spot) + (entry_future - close.future_price)
    )
    notional = shares * entry_spot
    established = int(row["position_established_ns"])
    terminal_ns = close.decision_cursor.recv_time_ns
    record = _base_record(row, calendar_row, config)
    record.update(
        {
            "label_status": status,
            "terminal_branch": terminal_branch,
            "transition_branch": transition_branch,
            "outcome_status": "known",
            "unresolved_reason": None,
            "exit_date": close.date,
            "label_end_date": close.date,
            "exit_decision_time_ns": terminal_ns,
            "exit_spot_snapshot_time_ns": close.spot_cursor.recv_time_ns,
            "exit_future_snapshot_time_ns": (
                close.future_cursor.recv_time_ns if close.future_cursor else None
            ),
            "exit_spot_price": close.spot_price,
            "exit_future_price": close.future_price,
            "exit_spot_levels_swept": close.spot_levels_swept,
            "exit_future_levels_swept": close.future_levels_swept,
            "exit_spot_book_age_ms": close.spot_book_age_ms,
            "exit_future_book_age_ms": close.future_book_age_ms,
            "settlement_price": (
                float(settlement["settlement_price"]) if settlement is not None else None
            ),
            "settlement_time_ns": (
                int(settlement["settlement_time_ns"]) if settlement is not None else None
            ),
            "settlement_source_version": (
                str(settlement["settlement_source_version"])
                if settlement is not None
                else None
            ),
            "gross_cycle_pnl_twd": gross,
            "normalization_notional_twd": notional,
            "filled_cashflow_before_cost_bp": gross / notional * 10_000.0,
            "holding_seconds": (terminal_ns - established) / 1_000_000_000.0,
            "terminal_cashflow_priced": True,
            "counterfactual_executable": True,
            "execution_benchmark_role": (
                "official_settlement_plus_optimistic_zero_added_latency_spot_taker"
                if settlement is not None
                else "optimistic_zero_added_latency_same_cursor_taker_taker"
            ),
            "book_freshness_status": (
                "configured_limit_passed"
                if config.max_book_age_ns is not None
                else "ungated_age_recorded"
            ),
            "needs_next_session_label": False,
        }
    )
    return record


def _base_record(
    row: Mapping[str, object],
    calendar_row: Mapping[str, object],
    config: OvernightCarryConfig,
) -> dict[str, object]:
    identity = (
        str(row["exit_policy_trial_id"]),
        config.policy_version,
        str(calendar_row["calendar_version"]),
    )
    return {
        "carry_label_id": "carry-" + _digest(identity),
        "Date": str(row["Date"]),
        "ValueCode": str(row["ValueCode"]),
        "QuoteCode": str(row["QuoteCode"]),
        "entry_route": str(row["entry_route"]),
        "entry_policy_generation_id": str(row["entry_policy_generation_id"]),
        "entry_raw_order_fact_id": str(row["entry_raw_order_fact_id"]),
        "exit_rule_id": str(row["exit_rule_id"]),
        "exit_route": str(row["exit_route"]),
        "exit_policy_trial_id": str(row["exit_policy_trial_id"]),
        "source_branch_status": str(row["branch_status"]),
        "position_established_ns": int(row["position_established_ns"]),
        "entry_spot_price": float(row["entry_spot_price"]),
        "entry_future_price": float(row["entry_future_price"]),
        "contract_size_shares": int(row["entry_hedge_contract_size_shares"]),
        "physical_entry_dependency_id": "entry-dependency-"
        + _digest(
            (
                str(row["Date"]),
                str(row["ValueCode"]),
                str(row["QuoteCode"]),
                str(row["entry_route"]),
                str(row["entry_raw_order_fact_id"]),
            )
        ),
        "physical_entry_strict_carry_policy_alias_count": int(
            row["_physical_entry_strict_carry_policy_alias_count"]
        ),
        "physical_entry_coverage_weight": 1.0
        / int(row["_physical_entry_strict_carry_policy_alias_count"]),
        "policy_observation_weight": 1.0,
        "physical_entry_alias_nonindependent": int(
            row["_physical_entry_strict_carry_policy_alias_count"]
        )
        > 1,
        "cross_q_rule_route_additive": False,
        "sampling_unit": "exit_policy_trial_alternative",
        "expiry_session": str(calendar_row["expiry_session"]),
        "calendar_version": str(calendar_row["calendar_version"]),
        "carry_policy_version": config.policy_version,
        "max_carry_sessions": config.max_carry_sessions,
        "max_book_age_ns": config.max_book_age_ns,
        "max_book_age_enforced": config.max_book_age_ns is not None,
        "added_exit_latency_ns": config.added_exit_latency_ns,
        "latency_matched_to_exit_maker": False,
        "exact_quote_code_required": True,
        "roll_attempted": False,
        "roll_policy_version": None,
        "new_quote_code": None,
        "fee_cost_bp": None,
        "tax_cost_bp": None,
        "financing_cost_bp": None,
        "overnight_cost_bp": None,
        "cancel_cost_bp": None,
        "emergency_cost_bp": None,
        "cost_profile_version": None,
        "fees_tax_included": False,
        "pathwise_ev_ready": False,
    }


def _unresolved_fields(
    *,
    status: str,
    terminal_branch: str | None,
    transition_branch: str,
    reason: str | None,
) -> dict[str, object]:
    return {
        "label_status": status,
        "terminal_branch": terminal_branch,
        "transition_branch": transition_branch,
        "outcome_status": "censored" if not status.startswith("invalid_") else "unknown",
        "unresolved_reason": reason,
        "exit_date": None,
        "label_end_date": None,
        "exit_decision_time_ns": None,
        "exit_spot_snapshot_time_ns": None,
        "exit_future_snapshot_time_ns": None,
        "exit_spot_price": None,
        "exit_future_price": None,
        "exit_spot_levels_swept": None,
        "exit_future_levels_swept": None,
        "exit_spot_book_age_ms": None,
        "exit_future_book_age_ms": None,
        "settlement_price": None,
        "settlement_time_ns": None,
        "settlement_source_version": None,
        "gross_cycle_pnl_twd": None,
        "normalization_notional_twd": None,
        "filled_cashflow_before_cost_bp": None,
        "holding_seconds": None,
        "terminal_cashflow_priced": False,
        "counterfactual_executable": False,
        "execution_benchmark_role": (
            "official_settlement_plus_optimistic_zero_added_latency_spot_taker"
            if transition_branch.startswith("expiry_settlement")
            else "optimistic_zero_added_latency_same_cursor_taker_taker"
        ),
        "book_freshness_status": "not_priced",
        "needs_next_session_label": True,
    }


def _validate_entry_join(frame: pl.DataFrame) -> None:
    missing = frame.filter(pl.col("raw_order_fact_id").is_null())
    if missing.height:
        raise ValueError("strict carry policies are missing entry action facts")
    identity_pairs = (
        ("Date", "Date_entry"),
        ("ValueCode", "ValueCode_entry"),
        ("QuoteCode", "QuoteCode_entry"),
        ("entry_route", "route"),
        ("entry_raw_order_fact_id", "raw_order_fact_id"),
    )
    for left, right in identity_pairs:
        if right not in frame.columns:
            # Polars retains an unsuffixed right key if its name did not collide.
            raise ValueError(f"entry join did not retain required identity column {right}")
        bad = frame.filter(
            pl.col(left).cast(pl.String) != pl.col(right).cast(pl.String)
        )
        if bad.height:
            raise ValueError(f"position policy and entry action disagree on {left}")
    invalid = frame.filter(
        ~pl.col("full_fill").fill_null(False)
        | (pl.col("entry_hedge_status") != "executable")
        | pl.col("entry_spot_price").is_null()
        | pl.col("entry_future_price").is_null()
        | (pl.col("entry_hedge_contract_size_shares") <= 0)
    )
    if invalid.height:
        raise ValueError("strict carry requires a full entry and executable priced hedge")


def _validate_physical_entry_dependencies(
    frame: pl.DataFrame, dependency_keys: list[str]
) -> None:
    value_columns = [
        "position_established_ns",
        "entry_spot_price",
        "entry_future_price",
        "entry_hedge_contract_size_shares",
    ]
    consistency = frame.group_by(dependency_keys).agg(
        *[
            pl.col(column).n_unique().alias(f"_{column}_n")
            for column in value_columns
        ]
    )
    bad = consistency.filter(
        pl.any_horizontal(
            *[pl.col(f"_{column}_n") != 1 for column in value_columns]
        )
    )
    if bad.height:
        raise ValueError(
            "policy aliases sharing one physical entry disagree on position cashflow identity"
        )


def _validate_state_frame(
    frame: pl.DataFrame,
    market: str,
    date: str,
    value_code: str,
    quote_code: str,
) -> None:
    _require(frame, _STATE_REQUIRED, f"raw {market} state")
    for column, expected in (
        ("Date", date),
        ("ValueCode", value_code),
        ("QuoteCode", quote_code),
    ):
        values = set(str(value) for value in frame[column].unique())
        if values != {expected}:
            raise ValueError(
                f"raw {market} state is not exact on {column}: {values} != {{{expected}}}"
            )


def _state_eligible(
    state: _FormalMarketState,
    query_time_ns: int,
    max_book_age_ns: int | None,
) -> bool:
    row = state.row
    if row is None:
        return False
    if bool(row["trial_match"]) or not state.formal_after_trial:
        return False
    if not bool(row["book_state_available"]):
        return False
    book_time = row.get("book_recv_time_ns")
    if book_time is None:
        return False
    age = query_time_ns - int(book_time)
    if age < 0:
        raise ValueError("book receive time is after the causal decision cursor")
    return max_book_age_ns is None or age <= max_book_age_ns


def _book_sane(row: Mapping[str, object], market: Literal["spot", "future"]) -> bool:
    bids = executable_levels_from_state(dict(row), market, "bid")
    asks = executable_levels_from_state(dict(row), market, "ask")
    return bool(bids and asks and bids[0].price <= asks[0].price)


def _vwap(levels: Sequence[object], requested: int) -> tuple[float, int] | None:
    remaining = requested
    notional = 0.0
    swept = 0
    for level in levels:
        quantity = int(getattr(level, "quantity"))
        take = min(remaining, quantity)
        if take <= 0:
            continue
        notional += float(getattr(level, "price")) * take
        remaining -= take
        swept += 1
        if remaining == 0:
            return notional / requested, swept
    return None


def _spot_lots(contract_size_shares: int, board_lot_shares: int) -> int:
    if contract_size_shares <= 0 or contract_size_shares % board_lot_shares:
        raise ValueError(
            "contract_size_shares must be a positive multiple of spot_board_lot_shares"
        )
    return contract_size_shares // board_lot_shares


def _cursor(row: Mapping[str, object], priority: int) -> EventCursor:
    return EventCursor(
        int(row["recv_time_ns"]),
        priority,
        int(row["sequence"]),
    )


def _book_age_ms(query_time_ns: int, row: Mapping[str, object]) -> float:
    return (query_time_ns - int(row["book_recv_time_ns"])) / 1_000_000.0


def _normalise_sessions(sessions: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in sessions)
    if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("sessions must be non-empty, unique and ascending")
    if any(len(value) != 8 or not value.isdigit() for value in values):
        raise ValueError("sessions must use YYYYMMDD strings")
    return values


def _normalise_tapes(raw_tapes: RawTapeDay | Iterable[RawTapeDay]) -> tuple[RawTapeDay, ...]:
    tapes = (raw_tapes,) if isinstance(raw_tapes, RawTapeDay) else tuple(raw_tapes)
    dates = [str(tape.date) for tape in tapes]
    if len(dates) != len(set(dates)):
        raise ValueError("raw_tapes must contain at most one tape per date")
    return tapes


def _normalise_calendar(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    selected = frame.select(sorted(_CALENDAR_REQUIRED)).with_columns(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("expiry_session").cast(pl.String),
        pl.col("calendar_version").cast(pl.String),
    )
    if selected.select("QuoteCode").n_unique() != selected.height:
        raise ValueError("contract calendar must be unique by exact QuoteCode")
    invalid = selected.filter(
        (pl.col("expiry_session").str.len_chars() != 8)
        | (~pl.col("expiry_session").str.contains(r"^\d{8}$"))
        | (pl.col("calendar_version").is_null())
        | (pl.col("calendar_version").str.len_chars() == 0)
    )
    if invalid.height:
        raise ValueError("contract calendar contains invalid expiry/version values")
    return {str(row["QuoteCode"]): row for row in selected.iter_rows(named=True)}


def _normalise_settlements(
    frame: pl.DataFrame | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if frame is None:
        return {}
    _require(frame, _SETTLEMENT_REQUIRED, "settlement facts")
    selected = frame.select(sorted(_SETTLEMENT_REQUIRED)).with_columns(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("expiry_session").cast(pl.String),
        pl.col("settlement_price").cast(pl.Float64),
        pl.col("settlement_time_ns").cast(pl.Int64),
        pl.col("settlement_source_version").cast(pl.String),
        pl.col("settlement_final").cast(pl.Boolean),
    )
    key = ["QuoteCode", "expiry_session"]
    if selected.select(key).n_unique() != selected.height:
        raise ValueError("settlement facts must be unique by QuoteCode/expiry_session")
    invalid = selected.filter(
        (~pl.col("settlement_price").is_finite())
        | (pl.col("settlement_price") <= 0)
        | (pl.col("settlement_time_ns") < 0)
        | (~pl.col("settlement_final"))
        | (pl.col("settlement_source_version").is_null())
        | (pl.col("settlement_source_version").str.len_chars() == 0)
    )
    if invalid.height:
        raise ValueError("settlement facts must be positive, final and versioned")
    return {
        (str(row["QuoteCode"]), str(row["expiry_session"])): row
        for row in selected.iter_rows(named=True)
    }


def _future_sessions(
    date: str,
    sessions: tuple[str, ...],
    index: Mapping[str, int],
    maximum: int,
) -> tuple[str, ...]:
    start = int(index[date]) + 1
    return sessions[start : start + maximum]


def _audit(
    all_policies: pl.DataFrame,
    carry: pl.DataFrame,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    status_counts = (
        {
            str(row["label_status"]): int(row["len"])
            for row in labels.group_by("label_status").len().iter_rows(named=True)
        }
        if labels.height
        else {}
    )
    return pl.DataFrame(
        {
            "position_policy_rows": [all_policies.height],
            "strict_carry_policy_rows": [carry.height],
            "excluded_noncarry_or_unknown_rows": [all_policies.height - carry.height],
            "label_rows": [labels.height],
            "priced_terminal_rows": [status_counts.get("overnight_exit", 0) + status_counts.get("expiry_settlement", 0)],
            "overnight_exit_rows": [status_counts.get("overnight_exit", 0)],
            "expiry_settlement_rows": [status_counts.get("expiry_settlement", 0)],
            "expiry_settlement_unpriced_rows": [status_counts.get("expiry_settlement_unpriced", 0)],
            "roll_substitution_forbidden_rows": [status_counts.get("roll_substitution_forbidden", 0)],
            "unresolved_rows": [labels.height - status_counts.get("overnight_exit", 0) - status_counts.get("expiry_settlement", 0)],
            "fees_tax_complete_rows": [0],
            "pathwise_ev_ready_rows": [0],
        }
    )


def _label_schema() -> dict[str, pl.DataType]:
    return {
        "carry_label_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "exit_policy_trial_id": pl.String,
        "source_branch_status": pl.String,
        "position_established_ns": pl.Int64,
        "entry_spot_price": pl.Float64,
        "entry_future_price": pl.Float64,
        "contract_size_shares": pl.Int64,
        "physical_entry_dependency_id": pl.String,
        "physical_entry_strict_carry_policy_alias_count": pl.Int64,
        "physical_entry_coverage_weight": pl.Float64,
        "policy_observation_weight": pl.Float64,
        "physical_entry_alias_nonindependent": pl.Boolean,
        "cross_q_rule_route_additive": pl.Boolean,
        "sampling_unit": pl.String,
        "expiry_session": pl.String,
        "calendar_version": pl.String,
        "carry_policy_version": pl.String,
        "max_carry_sessions": pl.Int64,
        "max_book_age_ns": pl.Int64,
        "max_book_age_enforced": pl.Boolean,
        "added_exit_latency_ns": pl.Int64,
        "latency_matched_to_exit_maker": pl.Boolean,
        "exact_quote_code_required": pl.Boolean,
        "roll_attempted": pl.Boolean,
        "roll_policy_version": pl.String,
        "new_quote_code": pl.String,
        "label_status": pl.String,
        "terminal_branch": pl.String,
        "transition_branch": pl.String,
        "outcome_status": pl.String,
        "unresolved_reason": pl.String,
        "exit_date": pl.String,
        "label_end_date": pl.String,
        "exit_decision_time_ns": pl.Int64,
        "exit_spot_snapshot_time_ns": pl.Int64,
        "exit_future_snapshot_time_ns": pl.Int64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "exit_spot_levels_swept": pl.Int64,
        "exit_future_levels_swept": pl.Int64,
        "exit_spot_book_age_ms": pl.Float64,
        "exit_future_book_age_ms": pl.Float64,
        "settlement_price": pl.Float64,
        "settlement_time_ns": pl.Int64,
        "settlement_source_version": pl.String,
        "gross_cycle_pnl_twd": pl.Float64,
        "normalization_notional_twd": pl.Float64,
        "filled_cashflow_before_cost_bp": pl.Float64,
        "holding_seconds": pl.Float64,
        "terminal_cashflow_priced": pl.Boolean,
        "counterfactual_executable": pl.Boolean,
        "execution_benchmark_role": pl.String,
        "book_freshness_status": pl.String,
        "needs_next_session_label": pl.Boolean,
        **{name: pl.Float64 for name in DEFAULT_COST_COLUMNS},
        "cost_profile_version": pl.String,
        "fees_tax_included": pl.Boolean,
        "pathwise_ev_ready": pl.Boolean,
    }


def _from_records(
    records: list[dict[str, object]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=schema)
    expected = set(schema)
    for record in records:
        if set(record) != expected:
            raise ValueError(
                "overnight carry record schema mismatch: "
                f"missing={sorted(expected - set(record))}, "
                f"extra={sorted(set(record) - expected)}"
            )
    return pl.from_dicts(records, schema=schema, infer_schema_length=None)


def _require(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _digest(parts: Sequence[object]) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
