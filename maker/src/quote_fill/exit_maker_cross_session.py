"""Cross-session continuation for frozen maker-exit alternatives.

The ordinary :mod:`.exit_maker_study` replay is deliberately scoped to one
product-day.  This module composes those product-day replays without carrying
an exchange order or its displayed queue across a session boundary.  An open
long-spot/short-future position is carried; every following session creates a
new day-order sampler and reuses the *entry-time frozen* center/lower basis
threshold.

The output population is conditional on an actually established position:
entry no-fills and unhedged entry fills are excluded from ``policy_outcomes``
and retained only in the audit.  The four center/lower x maker-route rows are
counterfactual alternatives.  They are never jointly volume allocated.

Book age is recorded as a diagnostic.  It is not an alpha rule and is never
passed as a hard gate to the product-day replay.  Likewise, this module does
not silently turn an exact-contract carry into a calendar-roll trade.  A
missing exact QuoteCode, partial maker exit, incomplete taker hedge, explicit
cancel-race ambiguity, missing intermediate session, or unpriced expiry is a
terminal censor rather than an invented cash flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Literal, Mapping, Sequence

import polars as pl

from .engine import IntentTransition
from .exit_maker import (
    EXIT_MAKER_ROUTE_CONTRACTS,
    FUTURE_BID_EXIT_ROUTE,
    SUPPORTED_EXIT_MAKER_ROUTES,
    ExitMakerObservation,
    ExitMakerOcoProjection,
    ExitMakerOrderOutcome,
    _index_exit_maker_opposite_snapshots,
)
from .exit_maker_study import (
    ExitMakerProductDayReplayCache,
    ExitMakerProductDayResult,
    ExitMakerStudyConfig,
    _build_timeline,
    _index_observation_timeline,
    _opposite_snapshots,
    _replay_physical_policy,
    _trade_replay,
    replay_exit_maker_product_day,
    replay_exit_maker_product_day_to_artifacts,
)
from .layered import EventCursor
from .merged import session_cutoff_cursor
from .raw_tape import RawTapeDay
from .replay import IndependentOrderWindow


OutcomeType = Literal["terminal", "censored"]
CancelSemantics = Literal["strict", "nominal_instant_cancel_v0"]
SessionDispositionStatus = Literal[
    "flat_maker_taker",
    "carry_no_admission",
    "carry_known_zero_fill",
    "unknown_fill",
    "unknown_partial_fill",
    "unknown_exit_hedge_incomplete",
    "unknown_oco_cancel_race",
]

_ACTION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "full_fill",
    "entry_hedge_status",
    "entry_hedge_decision_time_ns",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_future_price",
    "entry_spot_price",
    "entry_hedge_contract_size_shares",
}
_EXIT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "exit_rule_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
}
_CALENDAR_REQUIRED = {"QuoteCode", "expiry_session", "calendar_version"}
_CARRY_BRANCHES = {
    "carry_at_eod_no_admission",
    "carry_at_eod_cancel_unconfirmed",
    "carry_no_exact_contract_events",
}


@dataclass(frozen=True)
class CrossSessionExitMakerConfig:
    """Immutable assumptions for one cross-session continuation study."""

    hedge_delay_ns: int = 50_000_000
    expected_exit_rule_ids: tuple[str, ...] = (
        "frozen_center",
        "frozen_lower",
    )
    book_age_diagnostic_threshold_ns: int = 1_000_000_000
    cancel_semantics: CancelSemantics = "strict"
    lifecycle_policy_version: str = "frozen_exit_maker_cross_session_v1"

    def validate(self) -> None:
        if (
            isinstance(self.hedge_delay_ns, bool)
            or not isinstance(self.hedge_delay_ns, int)
            or self.hedge_delay_ns < 0
        ):
            raise ValueError("hedge_delay_ns must be a non-negative integer")
        if (
            isinstance(self.book_age_diagnostic_threshold_ns, bool)
            or not isinstance(self.book_age_diagnostic_threshold_ns, int)
            or self.book_age_diagnostic_threshold_ns < 0
        ):
            raise ValueError(
                "book_age_diagnostic_threshold_ns must be non-negative"
            )
        if (
            not self.expected_exit_rule_ids
            or len(set(self.expected_exit_rule_ids))
            != len(self.expected_exit_rule_ids)
            or any(not value for value in self.expected_exit_rule_ids)
        ):
            raise ValueError("expected_exit_rule_ids must be non-empty and unique")
        if not self.lifecycle_policy_version:
            raise ValueError("lifecycle_policy_version cannot be empty")
        if self.cancel_semantics not in (
            "strict",
            "nominal_instant_cancel_v0",
        ):
            raise ValueError(
                "cancel_semantics must be 'strict' or "
                "'nominal_instant_cancel_v0'"
            )


@dataclass(frozen=True)
class CrossSessionExitMakerSession:
    """One candidate session with explicit point-in-time reference lineage.

    ``spot_ref_price`` and ``future_ref_price`` are intentionally separate
    from ``raw_tape.mapping``.  The overnight loader historically could keep
    entry-day references while selecting an exact later-day QuoteCode.  The
    cross-session replay overwrites the selected tape's reference values with
    these candidate-session values before running any reference-band gate.
    """

    date: str
    raw_tape: RawTapeDay
    spread_pair_clock: pl.DataFrame
    spot_ref_price: float
    future_ref_price: float
    ref_price_source_date: str
    ref_price_source_version: str
    session_start_time_ns: int | None = None
    cutoff_cursor: EventCursor | None = None

    def validate(self) -> None:
        date = _date(self.date, "session date")
        if not isinstance(self.raw_tape, RawTapeDay):
            raise TypeError("raw_tape must be a RawTapeDay")
        if str(self.raw_tape.date) != date:
            raise ValueError("raw_tape date must match candidate session date")
        for name, value in (
            ("spot_ref_price", self.spot_ref_price),
            ("future_ref_price", self.future_ref_price),
        ):
            if not _positive(value):
                raise ValueError(f"{name} must be finite and positive")
        if _date(self.ref_price_source_date, "ref_price_source_date") != date:
            raise ValueError(
                "candidate-session RefPrice source date must equal session date"
            )
        if not self.ref_price_source_version:
            raise ValueError("ref_price_source_version cannot be empty")
        if self.session_start_time_ns is not None and (
            isinstance(self.session_start_time_ns, bool)
            or not isinstance(self.session_start_time_ns, int)
            or self.session_start_time_ns < 0
        ):
            raise ValueError("session_start_time_ns must be non-negative or None")
        if self.cutoff_cursor is not None and not isinstance(
            self.cutoff_cursor, EventCursor
        ):
            raise TypeError("cutoff_cursor must be an EventCursor or None")


@dataclass(frozen=True)
class CrossSessionExitMakerResult:
    """Mutually exclusive terminal/censor labels and supporting diagnostics."""

    policy_outcomes: pl.DataFrame
    session_attempts: pl.DataFrame
    candidate_aliases: pl.DataFrame
    observations: pl.DataFrame
    transitions: pl.DataFrame
    audit: pl.DataFrame

    def frames(self) -> Mapping[str, pl.DataFrame]:
        return {
            "cross_session_exit_policy_outcomes": self.policy_outcomes,
            "cross_session_exit_session_attempts": self.session_attempts,
            "cross_session_exit_candidate_aliases": self.candidate_aliases,
            "cross_session_exit_observations": self.observations,
            "cross_session_exit_transitions": self.transitions,
            "cross_session_exit_audit": self.audit,
        }


@dataclass(frozen=True)
class ExitMakerSessionReplay:
    """One policy's session-local replay with no entry-Date mutation.

    This is the pure adapter for callers which already hold an established
    entry cash-flow fact.  It owns one fresh daily sampler; the entry prices
    are deliberately absent because they are only needed after a terminal
    exit is selected and joined back to the original filled entry.
    """

    session_date: str
    value_code: str
    quote_code: str
    route: str
    threshold_basis_bp: float
    physical_policy_id: str
    observations: tuple[ExitMakerObservation, ...]
    windows: tuple[IndependentOrderWindow, ...]
    transitions: tuple[IntentTransition, ...]
    outcomes: tuple[ExitMakerOrderOutcome, ...]
    projection: ExitMakerOcoProjection
    candidate_session_keys: tuple[str, ...]
    day_order_queue_reset: bool = True
    book_age_gate_enforced: bool = False


@dataclass(frozen=True)
class ExitMakerSessionDisposition:
    """Fail-closed state transition derived from a session-local replay."""

    status: SessionDispositionStatus
    terminal: bool
    carry_full_position: bool
    censored: bool
    winner_generation_id: str | None
    cancel_semantics: CancelSemantics
    nominal_cancel_model_assumption: bool
    pathwise_ev_ready: bool = False


def replay_exit_maker_session_policy(
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    *,
    value_code: str,
    exact_quote_code: str,
    route: str,
    threshold_basis_bp: float,
    start_time_ns: int,
    physical_policy_id: str,
    cutoff_cursor: EventCursor | None = None,
    config: CrossSessionExitMakerConfig = CrossSessionExitMakerConfig(),
) -> ExitMakerSessionReplay:
    """Replay one frozen exit policy in exactly one candidate session.

    No entry action is fabricated for the candidate Date.  The caller passes
    only the session-local market inputs and the frozen threshold.  The raw
    tape must already contain candidate-session RefPrice values; the exact
    entry QuoteCode is validated and no later contract is substituted.  This
    adapter is also useful to a future day-batched runner which wants to share
    raw indexes while preserving entry cash flows in a separate table.
    """

    config.validate()
    if not isinstance(raw_tape, RawTapeDay):
        raise TypeError("raw_tape must be a RawTapeDay")
    if route not in SUPPORTED_EXIT_MAKER_ROUTES:
        raise ValueError(f"unsupported exit maker route: {route}")
    if not _finite(threshold_basis_bp):
        raise ValueError("threshold_basis_bp must be finite")
    if (
        isinstance(start_time_ns, bool)
        or not isinstance(start_time_ns, int)
        or start_time_ns < 0
    ):
        raise ValueError("start_time_ns must be a non-negative integer")
    if not physical_policy_id:
        raise ValueError("physical_policy_id cannot be empty")
    _require(raw_tape.mapping, {"ValueCode", "QuoteCode"}, "raw tape mapping")
    exact_mapping = raw_tape.mapping.filter(
        (pl.col("ValueCode").cast(pl.String) == str(value_code))
        & (pl.col("QuoteCode").cast(pl.String) == str(exact_quote_code))
    )
    if exact_mapping.height != 1:
        raise ValueError("session replay requires the exact entry QuoteCode")
    for name, frame in (
        ("spot", raw_tape.spot_states),
        ("future", raw_tape.future_states),
    ):
        if "ValueCode" not in frame.columns or "QuoteCode" not in frame.columns:
            raise ValueError(f"raw {name} states lack exact-contract identity")
        selected = frame.filter(
            pl.col("ValueCode").cast(pl.String) == str(value_code)
        )
        if selected.is_empty() or set(
            map(str, selected["QuoteCode"].to_list())
        ) != {str(exact_quote_code)}:
            raise ValueError(
                f"raw {name} states do not contain only the exact entry QuoteCode"
            )
    cutoff = cutoff_cursor or session_cutoff_cursor(str(raw_tape.date))
    if start_time_ns >= cutoff.recv_time_ns:
        raise ValueError("start_time_ns must precede the session cutoff")

    timeline = _build_timeline(raw_tape, spread_pair_clock, str(value_code))
    timeline_index = _index_observation_timeline(timeline)
    contract = EXIT_MAKER_ROUTE_CONTRACTS[route]
    maker_replay = _trade_replay(
        raw_tape.future_trades
        if contract.maker_market == "future"
        else raw_tape.spot_trades,
        contract.maker_market,
    )
    opposite_index = _index_exit_maker_opposite_snapshots(
        _opposite_snapshots(timeline, contract.opposite_market)
    )
    replay = _replay_physical_policy(
        timeline_index,
        route=route,
        threshold_basis_bp=float(threshold_basis_bp),
        start_time_ns=start_time_ns,
        cutoff_cursor=cutoff,
        physical_policy_id=physical_policy_id,
        maker_replay=maker_replay,
        opposite_snapshot_index=opposite_index,
        outcome_replay_cache={},
        config=ExitMakerStudyConfig(
            hedge_delay_ns=config.hedge_delay_ns,
            max_book_age_ns=None,
            expected_exit_rule_ids=config.expected_exit_rule_ids,
            exit_lifecycle_policy_version=(
                f"{config.lifecycle_policy_version}/pure_session"
            ),
            exit_queue_scenario="displayed_queue_independent_daily_reset_v1",
            instant_cancel_v0=True,
        ),
    )
    epochs = {
        transition.generation_id: transition.spread_pair_epoch
        for transition in replay.transitions
        if transition.kind == "submit" and transition.generation_id is not None
    }
    keys = tuple(
        f"{raw_tape.date}:{epochs[window.generation_id]}:{window.generation_id}"
        for window in replay.windows
    )
    return ExitMakerSessionReplay(
        session_date=str(raw_tape.date),
        value_code=str(value_code),
        quote_code=str(exact_quote_code),
        route=route,
        threshold_basis_bp=float(threshold_basis_bp),
        physical_policy_id=physical_policy_id,
        observations=replay.observations,
        windows=replay.windows,
        transitions=replay.transitions,
        outcomes=replay.outcomes,
        projection=replay.projection,
        candidate_session_keys=keys,
    )


def classify_exit_maker_session(
    replay: ExitMakerSessionReplay,
    *,
    cancel_semantics: CancelSemantics = "strict",
) -> ExitMakerSessionDisposition:
    """Classify whether a full paired position is flat, carried, or unknown."""

    if not isinstance(replay, ExitMakerSessionReplay):
        raise TypeError("replay must be an ExitMakerSessionReplay")
    if cancel_semantics not in ("strict", "nominal_instant_cancel_v0"):
        raise ValueError("invalid cancel_semantics")
    projection = replay.projection
    outcome_by_id = {item.generation_id: item for item in replay.outcomes}
    window_by_id = {item.generation_id: item for item in replay.windows}
    winner_id = projection.winner_generation_id
    if winner_id is None:
        if not replay.windows:
            status: SessionDispositionStatus = "carry_no_admission"
        elif any(
            item.known_filled_maker_quantity is None for item in replay.outcomes
        ):
            status = "unknown_fill"
        elif any(
            (item.known_filled_maker_quantity or 0) > 0
            for item in replay.outcomes
        ):
            status = "unknown_partial_fill"
        else:
            status = "carry_known_zero_fill"
        return ExitMakerSessionDisposition(
            status=status,
            terminal=False,
            carry_full_position=status in (
                "carry_no_admission",
                "carry_known_zero_fill",
            ),
            censored=status not in (
                "carry_no_admission",
                "carry_known_zero_fill",
            ),
            winner_generation_id=None,
            cancel_semantics=cancel_semantics,
            nominal_cancel_model_assumption=(
                cancel_semantics == "nominal_instant_cancel_v0"
            ),
        )

    winner = outcome_by_id[winner_id]
    if winner.branch_status != "flat_same_day" or not winner.position_flat:
        if "hedge_incomplete" in winner.branch_status:
            status = "unknown_exit_hedge_incomplete"
        elif winner.known_filled_maker_quantity is None:
            status = "unknown_fill"
        else:
            status = "unknown_partial_fill"
        return ExitMakerSessionDisposition(
            status=status,
            terminal=False,
            carry_full_position=False,
            censored=True,
            winner_generation_id=winner_id,
            cancel_semantics=cancel_semantics,
            nominal_cancel_model_assumption=(
                cancel_semantics == "nominal_instant_cancel_v0"
            ),
        )

    winner_cursor = projection.winner_full_fill_cursor
    assert winner_cursor is not None
    active_sibling = any(
        member.disposition == "sibling_cancel_required"
        for member in projection.members
    )
    prior_unacked = any(
        window.generation_id != winner_id
        and window.start_cursor < winner_cursor
        and window.stop_cursor <= winner_cursor
        and not outcome_by_id[window.generation_id].cancel_ack_observed
        for window in window_by_id.values()
    )
    ambiguous = (
        not projection.position_projection_safe
        or active_sibling
        or prior_unacked
    )
    if ambiguous and cancel_semantics == "strict":
        return ExitMakerSessionDisposition(
            status="unknown_oco_cancel_race",
            terminal=False,
            carry_full_position=False,
            censored=True,
            winner_generation_id=winner_id,
            cancel_semantics=cancel_semantics,
            nominal_cancel_model_assumption=False,
        )
    return ExitMakerSessionDisposition(
        status="flat_maker_taker",
        terminal=True,
        carry_full_position=False,
        censored=False,
        winner_generation_id=winner_id,
        cancel_semantics=cancel_semantics,
        nominal_cancel_model_assumption=(
            cancel_semantics == "nominal_instant_cancel_v0"
        ),
    )


def replay_cross_session_exit_maker(
    execution_action_facts: pl.DataFrame,
    frozen_exit_facts: pl.DataFrame,
    candidate_sessions: Sequence[CrossSessionExitMakerSession]
    | Iterable[CrossSessionExitMakerSession],
    trading_sessions: Sequence[str] | Iterable[str],
    contract_calendar: pl.DataFrame,
    config: CrossSessionExitMakerConfig = CrossSessionExitMakerConfig(),
    *,
    same_day_result: ExitMakerProductDayResult | None = None,
    same_day_book_age_gate_enforced: bool = False,
    shared_day_replay_caches: dict[
        tuple[str, str, str], ExitMakerProductDayReplayCache
    ]
    | None = None,
    shared_day_policy_facts: dict[tuple[object, ...], pl.DataFrame]
    | None = None,
    day_policy_spool_root: Path | None = None,
    retain_session_artifacts: bool = True,
) -> CrossSessionExitMakerResult:
    """Continue frozen center/lower maker exits over consecutive sessions.

    If ``same_day_result`` is supplied, its entry-day terminal/carry facts are
    consumed directly and later days begin with only the clean carry rows.
    The seed must have been produced without a book-age hard gate.  Otherwise
    the entry day is replayed from the matching candidate-session input.

    ``shared_day_policy_facts`` is a process-local cache for two classifiers
    consuming the exact same input objects (normally strict and nominal).  Its
    key includes object identity, the active policy set and all physical replay
    settings, so it never substitutes data across independently loaded runs.
    When ``day_policy_spool_root`` is supplied, the large product-day support
    frames are streamed to a temporary directory and only the small terminal
    policy table is read back.  That bounded path requires
    ``retain_session_artifacts=False``.

    The function never averages entry no-fills into policy PnL.  Every
    established entry alias yields exactly one row per frozen rule and exit
    route, and every such row ends in exactly one of ``terminal`` or
    ``censored``.
    """

    config.validate()
    if not isinstance(retain_session_artifacts, bool):
        raise TypeError("retain_session_artifacts must be boolean")
    if day_policy_spool_root is not None and retain_session_artifacts:
        raise ValueError(
            "day_policy_spool_root requires retain_session_artifacts=False"
        )
    if shared_day_policy_facts is not None and retain_session_artifacts:
        raise ValueError(
            "shared_day_policy_facts requires retain_session_artifacts=False"
        )
    _require(execution_action_facts, _ACTION_REQUIRED, "execution action facts")
    if same_day_result is not None and same_day_book_age_gate_enforced:
        raise ValueError(
            "same_day_result with a book-age hard gate cannot seed the primary replay"
        )
    # ``full_fill`` is intentionally null on unsupported/non-order action
    # rows.  Admission is alias-local: only an explicit true is a full fill;
    # null must behave like false here.  Keep the status comparison nullable,
    # however, so a full-filled alias with a missing physical hedge status is
    # still rejected below rather than silently treated as non-executable.
    full = (pl.col("full_fill") == True).fill_null(False)  # noqa: E712
    observed_expected = full & pl.col("entry_hedge_status").is_not_null()
    executable_expected = full & (
        pl.col("entry_hedge_status") == "executable"
    )
    incoherent_admission = execution_action_facts.filter(
        (
            pl.col("entry_hedge_label_observed")
            != observed_expected
        ).fill_null(True)
        | (
            pl.col("entry_hedge_executable")
            != executable_expected
        ).fill_null(True)
    )
    if incoherent_admission.height:
        raise ValueError(
            "entry alias-local hedge admission facts are incoherent"
        )
    identity = _entry_identity(execution_action_facts)
    entry_date, value_code, quote_code = identity
    all_entry_rows = execution_action_facts.height
    established = execution_action_facts.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
        & (pl.col("entry_hedge_executable") == True)  # noqa: E712
    )
    excluded_entry_rows = all_entry_rows - established.height
    if established.is_empty():
        return _empty_result(
            entry_date,
            value_code,
            quote_code,
            all_entry_rows=all_entry_rows,
            excluded_entry_rows=excluded_entry_rows,
            config=config,
        )
    _validate_established_entries(established)
    _require(frozen_exit_facts, _EXIT_REQUIRED, "frozen exit facts")
    rules = _frozen_rule_lookup(
        established,
        frozen_exit_facts,
        identity=identity,
        expected=config.expected_exit_rule_ids,
    )
    definitions = _policy_definitions(
        established,
        rules,
        config=config,
    )
    expected_policy_ids = set(definitions)

    calendar_sessions = _normalise_trading_sessions(trading_sessions)
    if entry_date not in calendar_sessions:
        raise ValueError("entry Date is absent from trading_sessions")
    expiry_session, calendar_version = _contract_expiry(
        contract_calendar,
        quote_code,
        entry_date,
    )
    entry_index = calendar_sessions.index(entry_date)
    if expiry_session in calendar_sessions:
        expiry_index: int | None = calendar_sessions.index(expiry_session)
    elif expiry_session > calendar_sessions[-1]:
        # A normal walk-forward slice often ends before the live contract's
        # expiry.  That is an observation-horizon censor, not a calendar
        # validation failure.
        expiry_index = None
    else:
        raise ValueError(
            "expiry_session is inside the observed date span but absent from "
            "trading_sessions"
        )
    if expiry_index is not None and expiry_index < entry_index:
        raise ValueError("contract expiry cannot precede the entry session")

    sessions = tuple(candidate_sessions)
    by_date: dict[str, CrossSessionExitMakerSession] = {}
    for item in sessions:
        if not isinstance(item, CrossSessionExitMakerSession):
            raise TypeError(
                "candidate_sessions must contain CrossSessionExitMakerSession values"
            )
        item.validate()
        if item.date in by_date:
            raise ValueError(f"duplicate candidate session: {item.date}")
        if item.date not in calendar_sessions:
            raise ValueError(f"candidate session absent from calendar: {item.date}")
        if calendar_sessions.index(item.date) < entry_index:
            raise ValueError("candidate sessions cannot precede the entry Date")
        by_date[item.date] = item

    same_day_policy: pl.DataFrame | None = None
    if same_day_result is not None:
        same_day_policy = same_day_result.position_policy_facts
        _validate_seed_policy_rows(
            same_day_policy,
            definitions,
            identity=identity,
        )

    last_input_index = max(
        (calendar_sessions.index(value) for value in by_date),
        default=entry_index,
    )
    horizon_index = (
        last_input_index
        if expiry_index is None
        else min(last_input_index, expiry_index)
    )
    if same_day_result is not None:
        horizon_index = max(horizon_index, entry_index)

    active = set(expected_policy_ids)
    final_records: dict[str, dict[str, object]] = {}
    attempt_records: list[dict[str, object]] = []
    candidate_frames: list[pl.DataFrame] = []
    observation_frames: list[pl.DataFrame] = []
    transition_frames: list[pl.DataFrame] = []
    sessions_attempted: dict[str, int] = {key: 0 for key in definitions}
    last_observed: dict[str, str | None] = {key: None for key in definitions}

    loop_start = entry_index
    if same_day_policy is not None:
        same_day_active = set(active)
        active = _consume_session_result(
            same_day_policy,
            definitions=definitions,
            active=active,
            session_date=entry_date,
            session_ordinal=0,
            calendar_version=calendar_version,
            expiry_session=expiry_session,
            config=config,
            prepared_tape=None,
            ref_source_version=None,
            final_records=final_records,
            attempt_records=attempt_records,
            sessions_attempted=sessions_attempted,
            last_observed=last_observed,
        )
        if retain_session_artifacts:
            same_day_candidates, same_day_observations, same_day_transitions = (
                _active_session_artifacts(
                    same_day_result,
                    active_policy_ids=same_day_active,
                )
            )
            candidate_frames.append(
                _annotate_artifact(
                    same_day_candidates,
                    entry_date,
                    ref_source_date=None,
                    ref_source_version=None,
                )
            )
            observation_frames.append(
                _annotate_artifact(
                    same_day_observations,
                    entry_date,
                    ref_source_date=None,
                    ref_source_version=None,
                )
            )
            transition_frames.append(
                _annotate_artifact(
                    same_day_transitions,
                    entry_date,
                    ref_source_date=None,
                    ref_source_version=None,
                )
            )
        loop_start += 1

    for session_index in range(loop_start, horizon_index + 1):
        if not active:
            break
        session_date = calendar_sessions[session_index]
        session_ordinal = session_index - entry_index
        session_input = by_date.get(session_date)
        if session_input is None:
            _censor_active(
                active,
                definitions,
                status="right_censored_missing_raw_session",
                detail=f"no raw/clock input for trading session {session_date}",
                session_date=session_date,
                calendar_version=calendar_version,
                expiry_session=expiry_session,
                config=config,
                final_records=final_records,
                sessions_attempted=sessions_attempted,
                last_observed=last_observed,
            )
            active.clear()
            break

        prepared, integrity_status = _prepare_candidate_tape(
            session_input,
            value_code=value_code,
            exact_quote_code=quote_code,
        )
        if prepared is None:
            if integrity_status == "carry_no_exact_contract_events":
                active_definitions = {
                    policy_id: definitions[policy_id]
                    for policy_id in active
                }
                no_admission_rows = _no_admission_policy_rows(
                    active_definitions,
                    session_date=session_date,
                )
                active = _consume_session_result(
                    no_admission_rows,
                    definitions=definitions,
                    active=active,
                    session_date=session_date,
                    session_ordinal=session_ordinal,
                    calendar_version=calendar_version,
                    expiry_session=expiry_session,
                    config=config,
                    prepared_tape=None,
                    ref_source_version=session_input.ref_price_source_version,
                    final_records=final_records,
                    attempt_records=attempt_records,
                    sessions_attempted=sessions_attempted,
                    last_observed=last_observed,
                )
                if (
                    active
                    and expiry_index is not None
                    and session_index == expiry_index
                ):
                    _censor_active(
                        active,
                        definitions,
                        status="right_censored_expiry_settlement_unpriced",
                        detail=(
                            "exact contract reached expiry without a maker/taker "
                            "exit or priced settlement plus spot unwind"
                        ),
                        session_date=session_date,
                        calendar_version=calendar_version,
                        expiry_session=expiry_session,
                        config=config,
                        final_records=final_records,
                        sessions_attempted=sessions_attempted,
                        last_observed=last_observed,
                    )
                    active.clear()
                continue
            _censor_active(
                active,
                definitions,
                status=integrity_status or "invalid_session_data",
                detail=(
                    "candidate session does not contain the exact entry contract"
                    if integrity_status == "roll_substitution_forbidden"
                    else "candidate session raw tape failed exact-pair validation"
                ),
                session_date=session_date,
                calendar_version=calendar_version,
                expiry_session=expiry_session,
                config=config,
                final_records=final_records,
                sessions_attempted=sessions_attempted,
                last_observed=last_observed,
            )
            active.clear()
            break

        session_active = set(active)
        active_definitions = {
            policy_id: definitions[policy_id]
            for policy_id in session_active
        }
        active_entry_policy_ids = {
            str(item["entry_policy_generation_id"])
            for item in active_definitions.values()
        }
        active_entries = established.filter(
            pl.col("policy_generation_id").cast(pl.String).is_in(
                sorted(active_entry_policy_ids)
            )
        )
        session_actions = _actions_for_session(
            active_entries,
            session_date=session_date,
            entry_date=entry_date,
            start_time_ns=_session_start_time(session_input, prepared),
        )
        session_exits = _exit_facts_for_session(
            frozen_exit_facts,
            policy_ids=active_entry_policy_ids,
            session_date=session_date,
        )
        day_config = ExitMakerStudyConfig(
            hedge_delay_ns=config.hedge_delay_ns,
            max_book_age_ns=None,
            expected_exit_rule_ids=config.expected_exit_rule_ids,
            exit_lifecycle_policy_version=(
                f"{config.lifecycle_policy_version}/day_order"
            ),
            exit_queue_scenario="displayed_queue_independent_daily_reset_v1",
            instant_cancel_v0=True,
        )
        day_result: ExitMakerProductDayResult | None = None
        policy_cache_key = _day_policy_facts_cache_key(
            execution_action_facts,
            frozen_exit_facts,
            session_input,
            session_active=session_active,
            value_code=value_code,
            quote_code=quote_code,
            start_time_ns=_session_start_time(session_input, prepared),
            day_config=day_config,
        )
        policy_rows = (
            None
            if shared_day_policy_facts is None
            else shared_day_policy_facts.get(policy_cache_key)
        )
        try:
            if policy_rows is None and day_policy_spool_root is not None:
                policy_rows = _spool_day_policy_facts(
                    session_actions,
                    session_exits,
                    prepared,
                    session_input.spread_pair_clock,
                    day_config,
                    cutoff_cursor=session_input.cutoff_cursor,
                    active_exit_policy_trial_ids=sorted(session_active),
                    spool_root=Path(day_policy_spool_root),
                )
            elif policy_rows is None:
                day_replay_cache = None
                if shared_day_replay_caches is not None:
                    cache_key = (session_date, value_code, quote_code)
                    day_replay_cache = shared_day_replay_caches.setdefault(
                        cache_key, ExitMakerProductDayReplayCache()
                    )
                day_result = replay_exit_maker_product_day(
                    session_actions,
                    session_exits,
                    prepared,
                    session_input.spread_pair_clock,
                    day_config,
                    cutoff_cursor=session_input.cutoff_cursor,
                    active_exit_policy_trial_ids=sorted(session_active),
                    replay_cache=day_replay_cache,
                )
                policy_rows = day_result.position_policy_facts
            if shared_day_policy_facts is not None and (
                policy_cache_key not in shared_day_policy_facts
            ):
                shared_day_policy_facts[policy_cache_key] = policy_rows
            if retain_session_artifacts and day_result is None:
                raise ValueError(
                    "cached policy-only replay cannot retain session artifacts"
                )
        except ValueError as error:
            detail = str(error)
            failure_status = (
                "right_censored_missing_spread_clock"
                if "SpreadPairTotalCount clock" in detail
                or "SpreadPairTotalCount must cover" in detail
                else "invalid_session_data"
            )
            _censor_active(
                active,
                definitions,
                status=failure_status,
                detail=detail,
                session_date=session_date,
                calendar_version=calendar_version,
                expiry_session=expiry_session,
                config=config,
                final_records=final_records,
                sessions_attempted=sessions_attempted,
                last_observed=last_observed,
            )
            active.clear()
            break

        assert policy_rows is not None
        _validate_day_policy_rows(
            policy_rows,
            active_definitions,
            session_date=session_date,
        )
        if retain_session_artifacts:
            assert day_result is not None
            day_candidates, day_observations, day_transitions = (
                _active_session_artifacts(
                    day_result,
                    active_policy_ids=session_active,
                )
            )
            candidate_frames.append(
                _annotate_artifact(
                    day_candidates,
                    session_date,
                    ref_source_date=session_input.ref_price_source_date,
                    ref_source_version=session_input.ref_price_source_version,
                )
            )
            observation_frames.append(
                _annotate_artifact(
                    day_observations,
                    session_date,
                    ref_source_date=session_input.ref_price_source_date,
                    ref_source_version=session_input.ref_price_source_version,
                )
            )
            transition_frames.append(
                _annotate_artifact(
                    day_transitions,
                    session_date,
                    ref_source_date=session_input.ref_price_source_date,
                    ref_source_version=session_input.ref_price_source_version,
                )
            )
        active = _consume_session_result(
            policy_rows,
            definitions=definitions,
            active=active,
            session_date=session_date,
            session_ordinal=session_ordinal,
            calendar_version=calendar_version,
            expiry_session=expiry_session,
            config=config,
            prepared_tape=prepared,
            ref_source_version=session_input.ref_price_source_version,
            final_records=final_records,
            attempt_records=attempt_records,
            sessions_attempted=sessions_attempted,
            last_observed=last_observed,
        )
        if active and expiry_index is not None and session_index == expiry_index:
            _censor_active(
                active,
                definitions,
                status="right_censored_expiry_settlement_unpriced",
                detail=(
                    "exact contract reached expiry without a maker/taker exit; "
                    "no final settlement plus spot-unwind cash flow was supplied"
                ),
                session_date=session_date,
                calendar_version=calendar_version,
                expiry_session=expiry_session,
                config=config,
                final_records=final_records,
                sessions_attempted=sessions_attempted,
                last_observed=last_observed,
            )
            active.clear()

    if active:
        last_date = calendar_sessions[horizon_index]
        status = (
            "right_censored_expiry_settlement_unpriced"
            if expiry_index is not None and horizon_index == expiry_index
            else "right_censored_observation_end"
        )
        detail = (
            "exact contract reached expiry without a priced terminal path"
            if status == "right_censored_expiry_settlement_unpriced"
            else "position remains open after the final supplied candidate session"
        )
        _censor_active(
            active,
            definitions,
            status=status,
            detail=detail,
            session_date=last_date,
            calendar_version=calendar_version,
            expiry_session=expiry_session,
            config=config,
            final_records=final_records,
            sessions_attempted=sessions_attempted,
            last_observed=last_observed,
        )
        active.clear()

    if set(final_records) != expected_policy_ids:
        raise AssertionError("cross-session replay did not emit one outcome per policy")
    outcomes = _records_frame(
        [final_records[key] for key in sorted(final_records)],
        _outcome_schema(),
    )
    if outcomes.select("exit_policy_trial_id").n_unique() != outcomes.height:
        raise AssertionError("cross-session policy outcomes are not mutually exclusive")
    if outcomes.filter(
        ~pl.col("outcome_type").is_in(["terminal", "censored"])
    ).height:
        raise AssertionError("every cross-session policy must terminate or censor")

    attempts = _records_frame(attempt_records, _attempt_schema())
    candidate_aliases = _concat_artifacts(candidate_frames)
    observations = _concat_artifacts(observation_frames)
    transitions = _concat_artifacts(transition_frames)
    terminal_count = outcomes.filter(pl.col("outcome_type") == "terminal").height
    censor_count = outcomes.height - terminal_count
    audit = pl.from_dicts(
        [
            {
                "Date": entry_date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "all_entry_policy_alias_rows": all_entry_rows,
                "established_entry_policy_alias_rows": established.height,
                "excluded_entry_no_fill_or_unhedged_rows": excluded_entry_rows,
                "primary_policy_outcome_rows": outcomes.height,
                "terminal_policy_rows": terminal_count,
                "censored_policy_rows": censor_count,
                "candidate_alias_rows": candidate_aliases.height,
                "session_attempt_rows": attempts.height,
                "expected_exit_rules": len(config.expected_exit_rule_ids),
                "expected_exit_routes": len(SUPPORTED_EXIT_MAKER_ROUTES),
                "no_fill_entries_excluded_from_primary": True,
                "frozen_threshold_reestimated_after_entry": False,
                "session_scoped_layered_sampler": True,
                "day_order_and_queue_reset_each_session": True,
                "candidate_identity_includes_session_date_and_epoch": True,
                "exact_quote_code_roll_substitution_forbidden": True,
                "candidate_session_ref_price_required": True,
                "book_age_gate_enforced": False,
                "book_age_diagnostic_only": True,
                "cancel_semantics": config.cancel_semantics,
                "nominal_cancel_model_assumption": (
                    config.cancel_semantics == "nominal_instant_cancel_v0"
                ),
                "maker_exit_alternatives_nonadditive": True,
                "joint_volume_allocated": False,
                "forced_next_session_taker_taker_used": False,
                "calendar_version": calendar_version,
                "expiry_session": expiry_session,
                "lifecycle_policy_version": config.lifecycle_policy_version,
            }
        ],
        infer_schema_length=None,
    )
    return CrossSessionExitMakerResult(
        outcomes,
        attempts,
        candidate_aliases,
        observations,
        transitions,
        audit,
    )


def _day_policy_facts_cache_key(
    execution_action_facts: pl.DataFrame,
    frozen_exit_facts: pl.DataFrame,
    session: CrossSessionExitMakerSession,
    *,
    session_active: set[str],
    value_code: str,
    quote_code: str,
    start_time_ns: int,
    day_config: ExitMakerStudyConfig,
) -> tuple[object, ...]:
    """Bind one process-local policy replay without rescanning frame content.

    Object identity is intentional: this cache exists only to let two outcome
    classifiers consume the exact same already-loaded candidate session.  It
    is neither serialized nor accepted as evidence that independently loaded
    data are equal.  Candidate-cache markers and upstream partition markers
    provide the persistent content-integrity contract in the formal runner.
    """

    cutoff = session.cutoff_cursor
    cutoff_key = (
        None
        if cutoff is None
        else (
            cutoff.recv_time_ns,
            cutoff.event_sequence,
            cutoff.row_index,
        )
    )
    return (
        id(execution_action_facts),
        id(frozen_exit_facts),
        id(session),
        session.date,
        str(value_code),
        str(quote_code),
        float(session.spot_ref_price),
        float(session.future_ref_price),
        int(start_time_ns),
        cutoff_key,
        tuple(sorted(session_active)),
        day_config.hedge_delay_ns,
        day_config.max_book_age_ns,
        tuple(day_config.expected_exit_rule_ids),
        day_config.exit_lifecycle_policy_version,
        day_config.exit_queue_scenario,
        day_config.instant_cancel_v0,
    )


def _spool_day_policy_facts(
    execution_action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    config: ExitMakerStudyConfig,
    *,
    cutoff_cursor: EventCursor | None,
    active_exit_policy_trial_ids: Sequence[str],
    spool_root: Path,
) -> pl.DataFrame:
    """Run the bounded artifact sink and retain only policy classifications."""

    root = Path(spool_root)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".cross-day-policy.tmp-", dir=root)
    )
    try:
        artifacts = replay_exit_maker_product_day_to_artifacts(
            execution_action_facts,
            exit_facts,
            raw_tape,
            spread_pair_clock,
            config,
            artifact_directory=stage,
            cutoff_cursor=cutoff_cursor,
            active_exit_policy_trial_ids=active_exit_policy_trial_ids,
            # Retaining physical replay objects would defeat the bounded
            # artifact path.  The policy-fact cache above performs the only
            # cross-classifier reuse required by this runner.
            replay_cache=None,
        )
        paths = artifacts.artifact_paths()
        try:
            policy_path = paths["exit_maker_position_policy_facts"]
        except KeyError as error:
            raise ValueError(
                "spooled day replay omitted position policy facts"
            ) from error
        return pl.read_parquet(policy_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _consume_session_result(
    policy_rows: pl.DataFrame,
    *,
    definitions: Mapping[str, dict[str, object]],
    active: set[str],
    session_date: str,
    session_ordinal: int,
    calendar_version: str,
    expiry_session: str,
    config: CrossSessionExitMakerConfig,
    prepared_tape: RawTapeDay | None,
    ref_source_version: str | None,
    final_records: dict[str, dict[str, object]],
    attempt_records: list[dict[str, object]],
    sessions_attempted: dict[str, int],
    last_observed: dict[str, str | None],
) -> set[str]:
    rows = {
        str(row["exit_policy_trial_id"]): row
        for row in policy_rows.iter_rows(named=True)
    }
    next_active: set[str] = set()
    for policy_id in sorted(active):
        try:
            row = rows[policy_id]
        except KeyError as error:
            raise ValueError(
                f"candidate session is missing active policy {policy_id}"
            ) from error
        classification, status, detail = _classify_session_policy(
            row,
            cancel_semantics=config.cancel_semantics,
        )
        sessions_attempted[policy_id] += 1
        last_observed[policy_id] = session_date
        ages = _winner_book_age_diagnostic(
            row,
            prepared_tape,
            threshold_ns=config.book_age_diagnostic_threshold_ns,
            terminal=classification == "terminal",
        )
        attempt_records.append(
            _attempt_record(
                definitions[policy_id],
                row,
                classification=classification,
                classification_status=status,
                session_date=session_date,
                session_ordinal=session_ordinal,
                ref_source_version=ref_source_version,
                ages=ages,
            )
        )
        if classification == "carry":
            next_active.add(policy_id)
            continue
        if classification == "terminal":
            final_records[policy_id] = _terminal_record(
                definitions[policy_id],
                row,
                session_date=session_date,
                session_ordinal=session_ordinal,
                sessions_attempted=sessions_attempted[policy_id],
                calendar_version=calendar_version,
                expiry_session=expiry_session,
                config=config,
                ages=ages,
            )
            continue
        final_records[policy_id] = _censor_record(
            definitions[policy_id],
            status=status,
            detail=detail,
            session_date=session_date,
            sessions_attempted=sessions_attempted[policy_id],
            calendar_version=calendar_version,
            expiry_session=expiry_session,
            config=config,
        )
    return next_active


def _classify_session_policy(
    row: Mapping[str, object],
    *,
    cancel_semantics: CancelSemantics = "strict",
) -> tuple[Literal["carry", "terminal", "censored"], str, str | None]:
    """Map one product-day fact to a fail-closed cross-session transition."""

    if cancel_semantics not in ("strict", "nominal_instant_cancel_v0"):
        raise ValueError("invalid cancel_semantics")
    strict_branch = str(row.get("branch_status"))
    if cancel_semantics == "nominal_instant_cancel_v0":
        nominal = row.get("nominal_instant_cancel_v0_branch")
        branch = str(nominal) if nominal is not None else strict_branch
    else:
        branch = strict_branch
    if branch == "flat_same_day":
        complete = (
            (
                row.get("terminal_outcome") is True
                or cancel_semantics == "nominal_instant_cancel_v0"
            )
            and row.get("exit_hedge_status") == "executable"
            and _positive(row.get("exit_spot_price"))
            and _positive(row.get("exit_future_price"))
            and _finite(row.get("gross_cycle_pnl_twd"))
        )
        if complete:
            return "terminal", "maker_taker_exit", None
        return (
            "censored",
            "inconsistent_terminal_path",
            "flat branch lacks a complete delayed hedge or priced cash flow",
        )
    if branch == "cancel_race_unknown":
        return (
            "censored",
            "cancel_race_unknown",
            "cancel/fill ordering is not identified by an exchange cancel ACK",
        )
    if "partial_fill" in branch:
        return (
            "censored",
            "partial_exit_fill_unknown_position",
            "partial maker exit changes inventory and cannot restart as a full pair",
        )
    if "hedge_incomplete" in branch:
        return (
            "censored",
            "exit_hedge_incomplete",
            "maker fill completed without a complete delayed taker hedge",
        )
    if "fill_unknown" in branch:
        return (
            "censored",
            "maker_fill_state_unknown",
            "maker fill quantity is not identified at the session boundary",
        )
    if branch in _CARRY_BRANCHES and (
        row.get("needs_next_session_label") is True
        or branch == "carry_no_exact_contract_events"
        or cancel_semantics == "nominal_instant_cancel_v0"
    ):
        return "carry", branch, None
    return (
        "censored",
        "unsupported_session_branch",
        f"product-day exit branch cannot be continued safely: {branch}",
    )


def _prepare_candidate_tape(
    session: CrossSessionExitMakerSession,
    *,
    value_code: str,
    exact_quote_code: str,
) -> tuple[RawTapeDay | None, str | None]:
    """Select the exact contract and inject candidate-day RefPrice values."""

    mapping = session.raw_tape.mapping
    if "ValueCode" not in mapping.columns or "QuoteCode" not in mapping.columns:
        return None, "invalid_session_data"
    value_rows = mapping.filter(
        pl.col("ValueCode").cast(pl.String) == value_code
    )
    exact = value_rows.filter(
        pl.col("QuoteCode").cast(pl.String) == exact_quote_code
    )
    if exact.height != 1:
        # Other contracts can be present in the raw source, but they are never
        # substituted for the entry QuoteCode.  Zero exact-contract events is
        # simply a no-admission day for the still-open position.
        return None, "carry_no_exact_contract_events"

    def selected(frame: pl.DataFrame, market: str) -> pl.DataFrame | None:
        if frame.is_empty():
            return frame
        required = {"Date", "ValueCode", "QuoteCode", "ref_price"}
        if not required.issubset(frame.columns):
            return None
        result = frame.filter(
            (pl.col("Date").cast(pl.String) == session.date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == exact_quote_code)
        )
        reference = (
            session.spot_ref_price
            if market == "spot"
            else session.future_ref_price
        )
        return result.with_columns(pl.lit(float(reference)).alias("ref_price"))

    spot_states = selected(session.raw_tape.spot_states, "spot")
    future_states = selected(session.raw_tape.future_states, "future")
    spot_trades = selected(session.raw_tape.spot_trades, "spot")
    future_trades = selected(session.raw_tape.future_trades, "future")
    if any(
        item is None
        for item in (spot_states, future_states, spot_trades, future_trades)
    ):
        return None, "invalid_session_data"
    assert spot_states is not None and future_states is not None
    assert spot_trades is not None and future_trades is not None
    if spot_states.is_empty() or future_states.is_empty():
        return None, "carry_no_exact_contract_events"
    selected_mapping = exact.with_columns(
        pl.lit(float(session.spot_ref_price)).alias("spot_ref_price"),
        pl.lit(float(session.future_ref_price)).alias("fut_ref_price"),
    )
    return (
        RawTapeDay(
            date=session.date,
            mapping=selected_mapping,
            spot_states=spot_states,
            future_states=future_states,
            spot_trades=spot_trades,
            future_trades=future_trades,
            audit=session.raw_tape.audit,
        ),
        None,
    )


def _actions_for_session(
    established: pl.DataFrame,
    *,
    session_date: str,
    entry_date: str,
    start_time_ns: int,
) -> pl.DataFrame:
    expressions: list[pl.Expr] = [pl.lit(session_date).alias("Date")]
    if session_date != entry_date:
        expressions.append(
            pl.lit(start_time_ns).cast(pl.Int64).alias(
                "entry_hedge_decision_time_ns"
            )
        )
    return established.with_columns(*expressions)


def _exit_facts_for_session(
    exits: pl.DataFrame,
    *,
    policy_ids: set[str],
    session_date: str,
) -> pl.DataFrame:
    return exits.filter(
        pl.col("policy_generation_id").cast(pl.String).is_in(policy_ids)
    ).with_columns(pl.lit(session_date).alias("Date"))


def _no_admission_policy_rows(
    definitions: Mapping[str, Mapping[str, object]],
    *,
    session_date: str,
) -> pl.DataFrame:
    """Build a known-zero-order day when the exact contract has no events."""

    return pl.from_dicts(
        [
            {
                "Date": session_date,
                "exit_policy_trial_id": policy_id,
                "exit_threshold_basis_bp": definition[
                    "frozen_exit_threshold_basis_bp"
                ],
                "branch_status": "carry_no_exact_contract_events",
                "nominal_instant_cancel_v0_branch": (
                    "carry_no_exact_contract_events"
                ),
                "terminal_outcome": False,
                "needs_next_session_label": True,
                "raw_candidate_count": 0,
                "exit_decision_time_ns": None,
                "exit_maker_price": None,
                "exit_hedge_status": None,
                "exit_hedge_price": None,
                "exit_spot_price": None,
                "exit_future_price": None,
                "exit_basis_bp": None,
                "gross_cycle_pnl_twd": None,
                "strict_ev_ready": False,
            }
            for policy_id, definition in sorted(definitions.items())
        ],
        infer_schema_length=None,
    )


def _session_start_time(
    session: CrossSessionExitMakerSession,
    tape: RawTapeDay,
) -> int:
    if session.session_start_time_ns is not None:
        value = session.session_start_time_ns
    else:
        minima = []
        for frame in (tape.spot_states, tape.future_states):
            if not frame.is_empty() and "recv_time_ns" in frame.columns:
                minima.append(int(frame["recv_time_ns"].min()))
        if not minima:
            raise ValueError("candidate session has no market-state receive time")
        first = min(minima)
        value = first - 1 if first > 0 else 0
    cutoff = session.cutoff_cursor
    if cutoff is not None and value >= cutoff.recv_time_ns:
        raise ValueError("candidate session start must precede cutoff")
    return value


def _active_session_artifacts(
    result: ExitMakerProductDayResult,
    *,
    active_policy_ids: set[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Keep support rows attributable to policies active at session open.

    The product-day engine deliberately evaluates every established policy
    alternative.  In a multi-session composition some alternatives may have
    terminated on an earlier day, so blindly appending the whole later-day
    result would publish unused counterfactual paths.  Candidate aliases carry
    the trial id directly; observations/transitions carry a physical-policy id
    and are linked through the active position-policy rows.
    """

    if not active_policy_ids:
        raise ValueError("active_policy_ids cannot be empty")
    positions = result.position_policy_facts
    _require(
        positions,
        {"exit_policy_trial_id", "physical_exit_policy_id"},
        "position policy facts used for active support filtering",
    )
    active_positions = positions.filter(
        pl.col("exit_policy_trial_id").cast(pl.String).is_in(
            sorted(active_policy_ids)
        )
    )
    present = set(
        active_positions["exit_policy_trial_id"].cast(pl.String).to_list()
    )
    if present != active_policy_ids:
        missing = sorted(active_policy_ids - present)
        raise ValueError(
            f"active policies missing from product-day support: {missing}"
        )

    physical_to_trials: dict[str, list[str]] = {}
    for row in active_positions.select(
        "physical_exit_policy_id", "exit_policy_trial_id"
    ).iter_rows(named=True):
        physical = str(row["physical_exit_policy_id"])
        trial = str(row["exit_policy_trial_id"])
        physical_to_trials.setdefault(physical, []).append(trial)
    physical_to_trials = {
        key: sorted(set(values)) for key, values in physical_to_trials.items()
    }

    candidates = result.candidate_aliases
    _require(
        candidates,
        {"exit_policy_trial_id"},
        "candidate aliases used for active support filtering",
    )
    candidates = candidates.filter(
        pl.col("exit_policy_trial_id").cast(pl.String).is_in(
            sorted(active_policy_ids)
        )
    ).with_columns(
        pl.col("exit_policy_trial_id")
        .cast(pl.String)
        .map_elements(lambda value: [value], return_dtype=pl.List(pl.String))
        .alias("active_exit_policy_trial_ids"),
        pl.lit(True).alias("active_policy_session"),
    )

    def physical_support(frame: pl.DataFrame, label: str) -> pl.DataFrame:
        _require(
            frame,
            {"physical_exit_policy_id"},
            f"{label} used for active support filtering",
        )
        filtered = frame.filter(
            pl.col("physical_exit_policy_id")
            .cast(pl.String)
            .is_in(sorted(physical_to_trials))
        ).with_columns(
            pl.col("physical_exit_policy_id")
            .cast(pl.String)
            .map_elements(
                lambda value: physical_to_trials.get(value),
                return_dtype=pl.List(pl.String),
            )
            .alias("active_exit_policy_trial_ids"),
            pl.lit(True).alias("active_policy_session"),
        )
        if filtered.filter(
            pl.col("active_exit_policy_trial_ids").is_null()
            | (pl.col("active_exit_policy_trial_ids").list.len() == 0)
            | (pl.col("active_policy_session") != True)  # noqa: E712
        ).height:
            raise AssertionError(
                f"{label} contains rows not traceable to an active attempt"
            )
        return filtered

    if candidates.filter(
        pl.col("active_exit_policy_trial_ids").is_null()
        | (pl.col("active_exit_policy_trial_ids").list.len() == 0)
        | (pl.col("active_policy_session") != True)  # noqa: E712
    ).height:
        raise AssertionError(
            "candidate aliases contain rows not traceable to an active attempt"
        )
    return (
        candidates,
        physical_support(result.observations, "observations"),
        physical_support(result.transitions, "transitions"),
    )


def _annotate_artifact(
    frame: pl.DataFrame,
    session_date: str,
    *,
    ref_source_date: str | None,
    ref_source_version: str | None,
) -> pl.DataFrame:
    result = frame.with_columns(
        pl.lit(session_date).alias("session_date"),
        pl.lit(ref_source_date).cast(pl.String).alias("ref_price_source_date"),
        pl.lit(ref_source_version)
        .cast(pl.String)
        .alias("ref_price_source_version"),
        pl.lit(True).alias("day_order_queue_reset"),
        pl.lit(False).alias("joint_volume_allocated_cross_session"),
    )
    if "spread_pair_epoch" in result.columns:
        result = result.with_columns(
            pl.concat_str(
                pl.lit(session_date),
                pl.lit(":"),
                pl.col("spread_pair_epoch").cast(pl.String),
            ).alias("session_epoch_key")
        )
    else:
        result = result.with_columns(
            pl.lit(None).cast(pl.String).alias("session_epoch_key")
        )
    if "exit_raw_candidate_fact_id" in result.columns:
        result = result.with_columns(
            pl.struct("exit_raw_candidate_fact_id", "session_epoch_key")
            .map_elements(
                lambda value: _digest(
                    session_date,
                    value.get("session_epoch_key"),
                    value.get("exit_raw_candidate_fact_id"),
                ),
                return_dtype=pl.String,
            )
            .alias("cross_session_candidate_id")
        )
    else:
        result = result.with_columns(
            pl.lit(None).cast(pl.String).alias("cross_session_candidate_id")
        )
    return result


def _winner_book_age_diagnostic(
    row: Mapping[str, object],
    tape: RawTapeDay | None,
    *,
    threshold_ns: int,
    terminal: bool,
) -> dict[str, object]:
    base: dict[str, object] = {
        "arrival_book_age_ns": None,
        "decision_book_age_ns": None,
        "max_book_age_ns": None,
        "book_age_diagnostic_status": "not_available",
        "book_age_gate_enforced": False,
    }
    if tape is None or not terminal:
        return base
    fill_ns = _optional_int(row.get("oco_winner_fill_recv_time_ns"))
    decision_ns = _optional_int(row.get("exit_decision_time_ns"))
    route = str(row.get("exit_route"))
    if fill_ns is None or decision_ns is None:
        return base
    if route == FUTURE_BID_EXIT_ROUTE:
        states = tape.spot_states
        same_time_visible = False
    else:
        states = tape.future_states
        same_time_visible = True
    arrival_book_ns = _latest_real_book_time(
        states,
        fill_ns,
        inclusive=same_time_visible,
    )
    decision_book_ns = _latest_real_book_time(states, decision_ns, inclusive=True)
    arrival_age = (
        None if arrival_book_ns is None else max(0, fill_ns - arrival_book_ns)
    )
    decision_age = (
        None if decision_book_ns is None else max(0, decision_ns - decision_book_ns)
    )
    values = [value for value in (arrival_age, decision_age) if value is not None]
    maximum = max(values) if values else None
    if maximum is None:
        status = "not_available"
    elif maximum > threshold_ns:
        status = "diagnostic_over_threshold"
    else:
        status = "diagnostic_within_threshold"
    return {
        "arrival_book_age_ns": arrival_age,
        "decision_book_age_ns": decision_age,
        "max_book_age_ns": maximum,
        "book_age_diagnostic_status": status,
        "book_age_gate_enforced": False,
    }


def _latest_real_book_time(
    states: pl.DataFrame,
    query_ns: int,
    *,
    inclusive: bool,
) -> int | None:
    if states.is_empty() or "recv_time_ns" not in states.columns:
        return None
    predicate = (
        pl.col("recv_time_ns") <= query_ns
        if inclusive
        else pl.col("recv_time_ns") < query_ns
    )
    if "raw_has_book" in states.columns:
        predicate = predicate & pl.col("raw_has_book").fill_null(False)
    selected = states.filter(predicate).sort(
        [column for column in ("recv_time_ns", "sequence", "packet_sequence") if column in states.columns]
    )
    if selected.is_empty():
        return None
    row = selected.row(-1, named=True)
    value = _optional_int(row.get("book_recv_time_ns"))
    return value if value is not None else int(row["recv_time_ns"])


def _classify_age(ages: Mapping[str, object]) -> tuple[object, ...]:
    return (
        ages["arrival_book_age_ns"],
        ages["decision_book_age_ns"],
        ages["max_book_age_ns"],
        ages["book_age_diagnostic_status"],
        ages["book_age_gate_enforced"],
    )


def _attempt_record(
    definition: Mapping[str, object],
    row: Mapping[str, object],
    *,
    classification: str,
    classification_status: str,
    session_date: str,
    session_ordinal: int,
    ref_source_version: str | None,
    ages: Mapping[str, object],
) -> dict[str, object]:
    arrival_age, decision_age, maximum_age, age_status, age_gate = _classify_age(ages)
    return {
        "exit_policy_trial_id": definition["exit_policy_trial_id"],
        "entry_policy_generation_id": definition["entry_policy_generation_id"],
        "entry_raw_order_fact_id": definition["entry_raw_order_fact_id"],
        "exit_rule_id": definition["exit_rule_id"],
        "exit_route": definition["exit_route"],
        "session_date": session_date,
        "session_ordinal": session_ordinal,
        "session_sampler_id": _digest(
            definition["exit_policy_trial_id"], session_date, "sampler"
        ),
        "carried_in": session_ordinal > 0,
        "day_order_queue_reset": True,
        "frozen_exit_threshold_basis_bp": definition[
            "frozen_exit_threshold_basis_bp"
        ],
        "observed_exit_threshold_basis_bp": row.get(
            "exit_threshold_basis_bp"
        ),
        "frozen_threshold_unchanged": math.isclose(
            float(definition["frozen_exit_threshold_basis_bp"]),
            float(row["exit_threshold_basis_bp"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "raw_candidate_count": _optional_int(row.get("raw_candidate_count")) or 0,
        "branch_status": str(row.get("branch_status")),
        "session_transition": classification,
        "session_transition_status": classification_status,
        "exit_hedge_status": row.get("exit_hedge_status"),
        "exit_spot_price": row.get("exit_spot_price"),
        "exit_future_price": row.get("exit_future_price"),
        "gross_cycle_pnl_twd": row.get("gross_cycle_pnl_twd"),
        "arrival_book_age_ns": arrival_age,
        "decision_book_age_ns": decision_age,
        "max_book_age_ns": maximum_age,
        "book_age_diagnostic_status": age_status,
        "book_age_gate_enforced": age_gate,
        "ref_price_source_version": ref_source_version,
        "joint_volume_allocated": False,
    }


def _terminal_record(
    definition: Mapping[str, object],
    row: Mapping[str, object],
    *,
    session_date: str,
    session_ordinal: int,
    sessions_attempted: int,
    calendar_version: str,
    expiry_session: str,
    config: CrossSessionExitMakerConfig,
    ages: Mapping[str, object],
) -> dict[str, object]:
    arrival_age, decision_age, maximum_age, age_status, age_gate = _classify_age(ages)
    terminal_branch = (
        "same_day_target_exit"
        if session_ordinal == 0
        else "cross_session_maker_exit"
    )
    pathwise_ready = (
        bool(row.get("strict_ev_ready", False))
        if config.cancel_semantics == "strict"
        else False
    )
    return {
        **definition,
        "outcome_type": "terminal",
        "outcome_status": terminal_branch,
        "terminal_branch": terminal_branch,
        "unresolved_reason": None,
        "censor_detail": None,
        "filled_entry_outcome_category": "completed",
        "filled_entry_terminal_date": session_date,
        "filled_entry_unresolved_reason": None,
        "terminal_date": session_date,
        "terminal_reason": terminal_branch,
        "terminal_session_date": session_date,
        "last_observed_session_date": session_date,
        "terminal_session_ordinal": session_ordinal,
        "sessions_attempted": sessions_attempted,
        "overnight_boundaries_crossed": session_ordinal,
        "exit_decision_time_ns": row.get("exit_decision_time_ns"),
        "exit_maker_price": row.get("exit_maker_price"),
        "exit_hedge_status": row.get("exit_hedge_status"),
        "exit_hedge_price": row.get("exit_hedge_price"),
        "exit_spot_price": row.get("exit_spot_price"),
        "exit_future_price": row.get("exit_future_price"),
        "exit_basis_bp": row.get("exit_basis_bp"),
        "gross_cycle_pnl_twd": row.get("gross_cycle_pnl_twd"),
        "terminal_cashflow_priced": True,
        "arrival_book_age_ns": arrival_age,
        "decision_book_age_ns": decision_age,
        "max_book_age_ns": maximum_age,
        "book_age_diagnostic_status": age_status,
        "book_age_gate_enforced": age_gate,
        "calendar_version": calendar_version,
        "expiry_session": expiry_session,
        "lifecycle_policy_version": config.lifecycle_policy_version,
        "strict_ev_ready": bool(row.get("strict_ev_ready", False)),
        "pathwise_ev_ready": pathwise_ready,
    }


def _censor_record(
    definition: Mapping[str, object],
    *,
    status: str,
    detail: str | None,
    session_date: str,
    sessions_attempted: int,
    calendar_version: str,
    expiry_session: str,
    config: CrossSessionExitMakerConfig,
) -> dict[str, object]:
    category = _filled_entry_category(status)
    return {
        **definition,
        "outcome_type": "censored",
        "outcome_status": status,
        "terminal_branch": None,
        "unresolved_reason": status,
        "censor_detail": detail,
        "filled_entry_outcome_category": category,
        "filled_entry_terminal_date": None,
        "filled_entry_unresolved_reason": status,
        "terminal_date": None,
        "terminal_reason": detail or status,
        "terminal_session_date": None,
        "last_observed_session_date": session_date,
        "terminal_session_ordinal": None,
        "sessions_attempted": sessions_attempted,
        "overnight_boundaries_crossed": None,
        "exit_decision_time_ns": None,
        "exit_maker_price": None,
        "exit_hedge_status": None,
        "exit_hedge_price": None,
        "exit_spot_price": None,
        "exit_future_price": None,
        "exit_basis_bp": None,
        "gross_cycle_pnl_twd": None,
        "terminal_cashflow_priced": False,
        "arrival_book_age_ns": None,
        "decision_book_age_ns": None,
        "max_book_age_ns": None,
        "book_age_diagnostic_status": "not_available",
        "book_age_gate_enforced": False,
        "calendar_version": calendar_version,
        "expiry_session": expiry_session,
        "lifecycle_policy_version": config.lifecycle_policy_version,
        "strict_ev_ready": False,
        "pathwise_ev_ready": False,
    }


def _filled_entry_category(status: str) -> str:
    if status == "right_censored_observation_end":
        return "still_open"
    if status.startswith("right_censored_"):
        return "censored"
    return "unknown"


def _censor_active(
    active: set[str],
    definitions: Mapping[str, dict[str, object]],
    *,
    status: str,
    detail: str,
    session_date: str,
    calendar_version: str,
    expiry_session: str,
    config: CrossSessionExitMakerConfig,
    final_records: dict[str, dict[str, object]],
    sessions_attempted: Mapping[str, int],
    last_observed: Mapping[str, str | None],
) -> None:
    for policy_id in sorted(active):
        final_records[policy_id] = _censor_record(
            definitions[policy_id],
            status=status,
            detail=detail,
            session_date=last_observed[policy_id] or session_date,
            sessions_attempted=sessions_attempted[policy_id],
            calendar_version=calendar_version,
            expiry_session=expiry_session,
            config=config,
        )


def _entry_identity(frame: pl.DataFrame) -> tuple[str, str, str]:
    identity = frame.select("Date", "ValueCode", "QuoteCode").unique()
    if identity.height != 1:
        raise ValueError("cross-session exit requires one exact entry product-day")
    date, value_code, quote_code = map(str, identity.row(0))
    return _date(date, "entry Date"), value_code, quote_code


def _validate_established_entries(frame: pl.DataFrame) -> None:
    if frame.select("policy_generation_id").n_unique() != frame.height:
        raise ValueError("established entry policy IDs must be unique")
    for row in frame.iter_rows(named=True):
        for name in (
            "entry_hedge_decision_time_ns",
            "entry_future_price",
            "entry_spot_price",
            "entry_hedge_contract_size_shares",
        ):
            if not _positive(row.get(name)):
                raise ValueError(f"established entry has invalid {name}")
        if not math.isclose(
            float(row["entry_hedge_contract_size_shares"]),
            2_000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("maker-exit V1 requires 2,000 shares per future")


def _frozen_rule_lookup(
    established: pl.DataFrame,
    exits: pl.DataFrame,
    *,
    identity: tuple[str, str, str],
    expected: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, object]]:
    entry_date, value_code, quote_code = identity
    selected = exits.filter(
        pl.col("policy_generation_id").cast(pl.String).is_in(
            list(map(str, established["policy_generation_id"].to_list()))
        )
    )
    if selected.select("policy_generation_id", "exit_rule_id").n_unique() != selected.height:
        raise ValueError("frozen exit facts contain duplicate policy/rule keys")
    expected_ids = set(map(str, established["policy_generation_id"].to_list()))
    if set(map(str, selected["policy_generation_id"].to_list())) != expected_ids:
        raise ValueError("frozen exit facts do not cover every established entry")
    if "exit_rule_contains_target_day_outcome" in selected.columns and selected.filter(
        pl.col("exit_rule_contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("frozen exit rule contains target-day outcomes")
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for policy_id in sorted(expected_ids):
        group = selected.filter(
            pl.col("policy_generation_id").cast(pl.String) == policy_id
        )
        observed = set(map(str, group["exit_rule_id"].to_list()))
        if observed != set(expected) or group.height != len(expected):
            raise ValueError(
                f"{policy_id}: frozen exits must contain exactly {list(expected)}"
            )
        for row in group.iter_rows(named=True):
            if tuple(map(str, (row["Date"], row["ValueCode"], row["QuoteCode"]))) != (
                entry_date,
                value_code,
                quote_code,
            ):
                raise ValueError("frozen exit identity differs from entry identity")
            if not _finite(row.get("exit_threshold_basis_bp")):
                raise ValueError("frozen exit threshold must be finite")
            source = _date(
                str(row.get("exit_rule_source_asof_date")),
                "exit_rule_source_asof_date",
            )
            if source >= entry_date:
                raise ValueError("frozen exit source must be strictly before entry Date")
            lookup[(policy_id, str(row["exit_rule_id"]))] = row
    return lookup


def _policy_definitions(
    established: pl.DataFrame,
    rules: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    config: CrossSessionExitMakerConfig,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for action in established.sort("policy_generation_id").iter_rows(named=True):
        entry_policy = str(action["policy_generation_id"])
        raw_entry = str(action["raw_order_fact_id"])
        for exit_rule in config.expected_exit_rule_ids:
            rule = rules[(entry_policy, exit_rule)]
            threshold = float(rule["exit_threshold_basis_bp"])
            for route in SUPPORTED_EXIT_MAKER_ROUTES:
                policy_id = f"{entry_policy}/exit/{exit_rule}/{route}"
                result[policy_id] = {
                    "Date": str(action["Date"]),
                    "ValueCode": str(action["ValueCode"]),
                    "QuoteCode": str(action["QuoteCode"]),
                    "entry_route": str(action["route"]),
                    "entry_policy_generation_id": entry_policy,
                    "entry_raw_order_fact_id": raw_entry,
                    "exit_rule_id": exit_rule,
                    "exit_route": route,
                    "exit_policy_trial_id": policy_id,
                    "alternative_set_id": f"{raw_entry}/maker-exit-alternatives",
                    "frozen_exit_threshold_basis_bp": threshold,
                    "exit_rule_source_asof_date": str(
                        rule["exit_rule_source_asof_date"]
                    ),
                    "entry_spot_price": float(action["entry_spot_price"]),
                    "entry_future_price": float(action["entry_future_price"]),
                    "entry_contract_size_shares": int(
                        action["entry_hedge_contract_size_shares"]
                    ),
                    "primary_entry_population": "full_fill_and_executable_hedge",
                    "entry_no_fill_in_primary": False,
                    "frozen_threshold_reestimated_after_entry": False,
                    "maker_exit_alternative_nonadditive": True,
                    "joint_volume_allocated": False,
                    "cancel_semantics": config.cancel_semantics,
                    "nominal_cancel_model_assumption": (
                        config.cancel_semantics == "nominal_instant_cancel_v0"
                    ),
                }
    return result


def _validate_seed_policy_rows(
    rows: pl.DataFrame,
    definitions: Mapping[str, Mapping[str, object]],
    *,
    identity: tuple[str, str, str],
) -> None:
    _validate_day_policy_rows(rows, definitions, session_date=identity[0])
    observed_identity = rows.select("Date", "ValueCode", "QuoteCode").unique()
    if observed_identity.height != 1 or tuple(map(str, observed_identity.row(0))) != identity:
        raise ValueError("same-day seed identity differs from the established entry")


def _validate_day_policy_rows(
    rows: pl.DataFrame,
    definitions: Mapping[str, Mapping[str, object]],
    *,
    session_date: str,
) -> None:
    required = {
        "Date",
        "exit_policy_trial_id",
        "exit_threshold_basis_bp",
        "branch_status",
        "terminal_outcome",
        "needs_next_session_label",
        "exit_hedge_status",
        "exit_spot_price",
        "exit_future_price",
        "gross_cycle_pnl_twd",
    }
    _require(rows, required, "product-day exit policy facts")
    if set(map(str, rows["exit_policy_trial_id"].to_list())) != set(definitions):
        raise ValueError("product-day exit policies differ from frozen alternatives")
    if rows.select("exit_policy_trial_id").n_unique() != rows.height:
        raise ValueError("product-day exit policy IDs are not unique")
    if set(map(str, rows["Date"].to_list())) != {session_date}:
        raise ValueError("product-day exit facts have the wrong session Date")
    for row in rows.iter_rows(named=True):
        policy_id = str(row["exit_policy_trial_id"])
        expected = float(definitions[policy_id]["frozen_exit_threshold_basis_bp"])
        observed = row.get("exit_threshold_basis_bp")
        if not _finite(observed) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("candidate session reestimated a frozen exit threshold")


def _contract_expiry(
    calendar: pl.DataFrame,
    quote_code: str,
    entry_date: str,
) -> tuple[str, str]:
    _require(calendar, _CALENDAR_REQUIRED, "contract calendar")
    selected = calendar.filter(
        pl.col("QuoteCode").cast(pl.String) == quote_code
    )
    if selected.height != 1:
        raise ValueError("contract calendar must contain one exact QuoteCode row")
    row = selected.row(0, named=True)
    expiry = _date(str(row["expiry_session"]), "expiry_session")
    if expiry < entry_date:
        raise ValueError("expiry_session cannot precede entry Date")
    version = str(row["calendar_version"])
    if not version:
        raise ValueError("calendar_version cannot be empty")
    return expiry, version


def _normalise_trading_sessions(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(_date(str(value), "trading session") for value in values)
    if not result:
        raise ValueError("trading_sessions cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError("trading_sessions contain duplicates")
    if tuple(sorted(result)) != result:
        raise ValueError("trading_sessions must be strictly increasing")
    return result


def _empty_result(
    date: str,
    value_code: str,
    quote_code: str,
    *,
    all_entry_rows: int,
    excluded_entry_rows: int,
    config: CrossSessionExitMakerConfig,
) -> CrossSessionExitMakerResult:
    audit = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "all_entry_policy_alias_rows": all_entry_rows,
                "established_entry_policy_alias_rows": 0,
                "excluded_entry_no_fill_or_unhedged_rows": excluded_entry_rows,
                "primary_policy_outcome_rows": 0,
                "terminal_policy_rows": 0,
                "censored_policy_rows": 0,
                "candidate_alias_rows": 0,
                "session_attempt_rows": 0,
                "expected_exit_rules": len(config.expected_exit_rule_ids),
                "expected_exit_routes": len(SUPPORTED_EXIT_MAKER_ROUTES),
                "no_fill_entries_excluded_from_primary": True,
                "frozen_threshold_reestimated_after_entry": False,
                "session_scoped_layered_sampler": True,
                "day_order_and_queue_reset_each_session": True,
                "candidate_identity_includes_session_date_and_epoch": True,
                "exact_quote_code_roll_substitution_forbidden": True,
                "candidate_session_ref_price_required": True,
                "book_age_gate_enforced": False,
                "book_age_diagnostic_only": True,
                "cancel_semantics": config.cancel_semantics,
                "nominal_cancel_model_assumption": (
                    config.cancel_semantics == "nominal_instant_cancel_v0"
                ),
                "maker_exit_alternatives_nonadditive": True,
                "joint_volume_allocated": False,
                "forced_next_session_taker_taker_used": False,
                "calendar_version": None,
                "expiry_session": None,
                "lifecycle_policy_version": config.lifecycle_policy_version,
            }
        ],
        infer_schema_length=None,
    )
    return CrossSessionExitMakerResult(
        pl.DataFrame(schema=_outcome_schema()),
        pl.DataFrame(schema=_attempt_schema()),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        audit,
    )


def _outcome_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "exit_policy_trial_id": pl.String,
        "alternative_set_id": pl.String,
        "frozen_exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "entry_spot_price": pl.Float64,
        "entry_future_price": pl.Float64,
        "entry_contract_size_shares": pl.Int64,
        "primary_entry_population": pl.String,
        "entry_no_fill_in_primary": pl.Boolean,
        "frozen_threshold_reestimated_after_entry": pl.Boolean,
        "maker_exit_alternative_nonadditive": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
        "cancel_semantics": pl.String,
        "nominal_cancel_model_assumption": pl.Boolean,
        "outcome_type": pl.String,
        "outcome_status": pl.String,
        "terminal_branch": pl.String,
        "unresolved_reason": pl.String,
        "censor_detail": pl.String,
        "filled_entry_outcome_category": pl.String,
        "filled_entry_terminal_date": pl.String,
        "filled_entry_unresolved_reason": pl.String,
        "terminal_date": pl.String,
        "terminal_reason": pl.String,
        "terminal_session_date": pl.String,
        "last_observed_session_date": pl.String,
        "terminal_session_ordinal": pl.Int64,
        "sessions_attempted": pl.Int64,
        "overnight_boundaries_crossed": pl.Int64,
        "exit_decision_time_ns": pl.Int64,
        "exit_maker_price": pl.Float64,
        "exit_hedge_status": pl.String,
        "exit_hedge_price": pl.Float64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "exit_basis_bp": pl.Float64,
        "gross_cycle_pnl_twd": pl.Float64,
        "terminal_cashflow_priced": pl.Boolean,
        "arrival_book_age_ns": pl.Int64,
        "decision_book_age_ns": pl.Int64,
        "max_book_age_ns": pl.Int64,
        "book_age_diagnostic_status": pl.String,
        "book_age_gate_enforced": pl.Boolean,
        "calendar_version": pl.String,
        "expiry_session": pl.String,
        "lifecycle_policy_version": pl.String,
        "strict_ev_ready": pl.Boolean,
        "pathwise_ev_ready": pl.Boolean,
    }


def _attempt_schema() -> dict[str, pl.DataType]:
    return {
        "exit_policy_trial_id": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "session_date": pl.String,
        "session_ordinal": pl.Int64,
        "session_sampler_id": pl.String,
        "carried_in": pl.Boolean,
        "day_order_queue_reset": pl.Boolean,
        "frozen_exit_threshold_basis_bp": pl.Float64,
        "observed_exit_threshold_basis_bp": pl.Float64,
        "frozen_threshold_unchanged": pl.Boolean,
        "raw_candidate_count": pl.Int64,
        "branch_status": pl.String,
        "session_transition": pl.String,
        "session_transition_status": pl.String,
        "exit_hedge_status": pl.String,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "gross_cycle_pnl_twd": pl.Float64,
        "arrival_book_age_ns": pl.Int64,
        "decision_book_age_ns": pl.Int64,
        "max_book_age_ns": pl.Int64,
        "book_age_diagnostic_status": pl.String,
        "book_age_gate_enforced": pl.Boolean,
        "ref_price_source_version": pl.String,
        "joint_volume_allocated": pl.Boolean,
    }


def _records_frame(
    records: list[dict[str, object]],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=schema)
    observed = set(records[0])
    expected = set(schema)
    if observed != expected:
        raise AssertionError(
            "record/schema columns disagree: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return pl.from_dicts(records, infer_schema_length=None).select(
        *(pl.col(name).cast(dtype).alias(name) for name, dtype in schema.items())
    )


def _concat_artifacts(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    material = [frame for frame in frames if frame.width]
    if not material:
        return pl.DataFrame()
    return pl.concat(material, how="diagonal_relaxed")


def _date(value: str, name: str) -> str:
    result = str(value)
    if len(result) != 8 or not result.isdigit():
        raise ValueError(f"{name} must be YYYYMMDD")
    return result


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _positive(value: object) -> bool:
    return _finite(value) and float(value) > 0


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _optional_int(value: object) -> int | None:
    return int(value) if _finite(value) else None


def _digest(*parts: object) -> str:
    payload = "|".join(map(str, parts)).encode()
    return "cross-exit-" + hashlib.sha256(payload).hexdigest()[:24]
