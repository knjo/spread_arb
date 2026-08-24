"""Analysis-only 13:00 dynamic B1/A1 maker-exit challenger.

This module deliberately reuses the existing exit-maker primitives without
changing the frozen same-day producer.  For one already-established paired
long-spot/short-future position it starts, at 13:00 Taipei session time, two
dynamic passive pegs:

* buy the future at B1, then sell two spot lots by taker after 50 ms; and
* sell two spot lots at A1, then buy one future by taker after 50 ms.

Each route uses :class:`~maker.src.quote_fill.layered.LayeredSampler` through
``build_layered_order_windows``.  An unchanged absolute price keeps its old
displayed queue, a more-aggressive target adds a layer without deleting the
old one, and a retreat cancels layers which are now ahead of the target.  All
generations from both routes are finally projected through one
earliest-full-fill-wins OCO.  Only that winner can be counted as the nominal
close; independent sibling fills remain cancel-race diagnostics.

The tape has no cancel ACK.  Consequently ``nominal_instant_cancel_v0`` is
the primary analysis view and the strict view is ``cancel_race_unknown``
whenever a winning fill has an active sibling.  Candidate fills also remain
independent and do not allocate shared market volume.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Iterable, Mapping

import polars as pl

from .engine import (
    LayeredWindowBuildResult,
    TargetObservation,
    build_layered_order_windows,
)
from .exit_maker import (
    EXIT_MAKER_ROUTE_CONTRACTS,
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
    ExitMakerOcoProjection,
    ExitMakerOrderOutcome,
    PairedExitPosition,
    project_earliest_full_fill_oco,
    replay_exit_maker_windows,
)
from .exit_maker_report import _validate_entry_action_execution_contract
from .exit_maker_study import (
    _CombinedPoint,
    _build_timeline,
    _opposite_snapshots,
    _rank_and_queue,
    _raw_ref_gate,
    _required_vwap,
    _trade_replay,
    _validate_physical_entry_identity,
)
from .hedge import DEFAULT_HEDGE_DELAY_NS
from .hedge_study import (
    FUTURE_HEDGE_CONTRACTS,
    SPOT_HEDGE_LOTS,
    executable_levels_from_state,
)
from .layered import EventCursor
from .merged import session_cutoff_cursor
from .raw_tape import RawTapeDay
from .targets import absolute_price_tick


AGGRESSIVE_1300_POLICY_VERSION = "aggressive_1300_dynamic_b1a1_oco_v1"
AGGRESSIVE_1300_START_BEFORE_SESSION_END_NS = 20 * 60 * 1_000_000_000
AGGRESSIVE_1300_ROUTES = (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE)

_INVENTORY_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "any_fill",
    "full_fill",
    "partial_fill",
    "full_fill_recv_time_ns",
    "entry_hedge_status",
    "entry_hedge_decision_time_ns",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_future_price",
    "entry_spot_price",
    "entry_hedge_contract_size_shares",
}

_SELECTED_POSITION_PATH_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_raw_order_fact_id": pl.String,
    "position_established_ns": pl.Int64,
    "policy_path_id": pl.String,
    "exit_policy_trial_id": pl.String,
    "terminal_date": pl.String,
    "exit_decision_time_ns": pl.Int64,
    "terminal_cashflow_priced": pl.Boolean,
    "outcome_type": pl.String,
    "outcome_status": pl.String,
    "last_observed_session_date": pl.String,
    "outstanding_interval_end_exclusive": pl.String,
    "entry_no_fill_included": pl.Boolean,
    "gross_zero_imputation": pl.Boolean,
}

_TEMPLATE_POSITION_ID = "__aggressive_1300_market_template__"


@dataclass(frozen=True)
class Aggressive1300ExitConfig:
    """Frozen semantics for the analysis-only challenger."""

    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS
    max_book_age_ns: int | None = None
    policy_version: str = AGGRESSIVE_1300_POLICY_VERSION
    peg_semantics: str = "layered_dynamic_best_price_forward_keep_retreat_cancel_v1"
    queue_scenario: str = "displayed_queue_at_each_new_absolute_price_v1"
    oco_semantics: str = "cross_route_earliest_full_fill_wins_v1"
    cancel_model: str = "nominal_instant_cancel_v0"

    def validate(self) -> None:
        if self.hedge_delay_ns != DEFAULT_HEDGE_DELAY_NS:
            raise ValueError("aggressive 13:00 challenger requires an exact 50 ms hedge")
        if self.max_book_age_ns is not None and (
            isinstance(self.max_book_age_ns, bool)
            or not isinstance(self.max_book_age_ns, int)
            or self.max_book_age_ns < 0
        ):
            raise ValueError("max_book_age_ns must be non-negative or None")
        expected = {
            "policy_version": AGGRESSIVE_1300_POLICY_VERSION,
            "peg_semantics": "layered_dynamic_best_price_forward_keep_retreat_cancel_v1",
            "queue_scenario": "displayed_queue_at_each_new_absolute_price_v1",
            "oco_semantics": "cross_route_earliest_full_fill_wins_v1",
            "cancel_model": "nominal_instant_cancel_v0",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must be {value!r}")


@dataclass(frozen=True)
class Aggressive1300InventorySelection:
    """Physical paired positions established by the 13:00 entry cutoff."""

    inventory: pl.DataFrame
    audit: pl.DataFrame


@dataclass(frozen=True)
class Aggressive1300OpenInventorySelection:
    """One selected normal path per physical position still open at 13:00."""

    inventory: pl.DataFrame
    audit: pl.DataFrame


@dataclass(frozen=True)
class Aggressive1300ExitResult:
    """One physical position's two-route dynamic replay and OCO projection."""

    date: str
    value_code: str
    quote_code: str
    position_id: str
    scheduled_start_cursor: EventCursor
    actual_first_submit_cursor: EventCursor | None
    cutoff_cursor: EventCursor
    observations_by_route: Mapping[str, tuple[TargetObservation, ...]]
    builds_by_route: Mapping[str, LayeredWindowBuildResult]
    outcomes_by_route: Mapping[str, tuple[ExitMakerOrderOutcome, ...]]
    projection: ExitMakerOcoProjection
    nominal_branch_status: str
    strict_branch_status: str
    active_sibling_cancel_count: int
    prior_unacked_cancel_count: int
    nominal_close_count: int
    cancel_ack_observed: bool = False
    joint_volume_allocated: bool = False
    strict_ev_ready: bool = False

    @property
    def winner_outcome(self) -> ExitMakerOrderOutcome | None:
        winner = self.projection.winner_generation_id
        if winner is None:
            return None
        for outcomes in self.outcomes_by_route.values():
            for outcome in outcomes:
                if outcome.generation_id == winner:
                    return outcome
        raise AssertionError("OCO winner is absent from replay outcomes")

    @property
    def winner_route(self) -> str | None:
        winner = self.winner_outcome
        return None if winner is None else winner.route

    @property
    def nominal_terminal(self) -> bool:
        return self.nominal_branch_status == "flat_same_day"

    @property
    def strict_terminal(self) -> bool:
        return self.strict_branch_status == "flat_same_day"

    def summary_record(self) -> dict[str, object]:
        """Return one non-additive physical-position summary row."""

        winner = self.winner_outcome
        winner_cursor = self.projection.winner_full_fill_cursor
        sibling_after_cancel = sum(
            member.independent_full_fill_after_cancel_request
            for member in self.projection.members
        )
        sibling_partial_before = sum(
            member.partial_fill_before_winner for member in self.projection.members
        )
        return {
            "Date": self.date,
            "ValueCode": self.value_code,
            "QuoteCode": self.quote_code,
            "position_id": self.position_id,
            "policy_version": AGGRESSIVE_1300_POLICY_VERSION,
            "scheduled_start_recv_time_ns": self.scheduled_start_cursor.recv_time_ns,
            "actual_first_submit_recv_time_ns": (
                None
                if self.actual_first_submit_cursor is None
                else self.actual_first_submit_cursor.recv_time_ns
            ),
            "entry_after_1300_allowed": False,
            "dynamic_b1a1_peg": True,
            "future_candidate_generations": len(
                self.builds_by_route[FUTURE_BID_EXIT_ROUTE].windows
            ),
            "spot_candidate_generations": len(
                self.builds_by_route[SPOT_ASK_EXIT_ROUTE].windows
            ),
            "winner_generation_id": self.projection.winner_generation_id,
            "winner_route": None if winner is None else winner.route,
            "winner_full_fill_recv_time_ns": (
                None if winner_cursor is None else winner_cursor.recv_time_ns
            ),
            "winner_hedge_delay_ns": (
                None
                if winner is None or not winner.hedge_attempts
                else winner.hedge_attempts[0].decision_time_ns
                - winner.hedge_attempts[0].maker_fill_cursor.recv_time_ns
            ),
            "winner_hedge_status": (
                None
                if winner is None or not winner.hedge_attempts
                else winner.hedge_attempts[0].status
            ),
            "active_sibling_cancel_count": self.active_sibling_cancel_count,
            "prior_unacked_cancel_count_before_winner": self.prior_unacked_cancel_count,
            "sibling_full_fill_after_cancel_request_count": sibling_after_cancel,
            "sibling_partial_before_winner_count": sibling_partial_before,
            "oco_position_projection_safe": self.projection.position_projection_safe,
            "nominal_branch_status": self.nominal_branch_status,
            "strict_branch_status": self.strict_branch_status,
            "nominal_close_count": self.nominal_close_count,
            "duplicate_close_prevented": self.nominal_close_count <= 1,
            "cancel_ack_observed": False,
            "cancel_model": "nominal_instant_cancel_v0",
            "joint_volume_allocated": False,
            "strict_ev_ready": False,
        }


@dataclass(frozen=True)
class Aggressive1300BatchResult:
    """Cached market replay projected to unique open physical positions.

    The rows are independent counterfactual labels.  The same market template
    can be reused by many positions, so they are deliberately marked unsafe to
    sum and never claim joint volume allocation.
    """

    position_outcomes: pl.DataFrame
    audit: pl.DataFrame
    market_template: Aggressive1300ExitResult | None


@dataclass(frozen=True)
class _Aggressive1300CacheEntry:
    raw_tape: RawTapeDay
    spread_pair_clock: pl.DataFrame
    result: Aggressive1300ExitResult


@dataclass
class Aggressive1300ProductDayReplayCache:
    """Process-local cache of one B1/A1 market replay per product-day.

    ``source_cache_key`` is an explicit caller-supplied immutable-source key.
    It is not a substitute for a formal source-marker verifier.  Object
    identity is included as a second guard, so a key cannot accidentally reuse
    a template built from a different in-memory tape.
    """

    _entries: dict[tuple[object, ...], _Aggressive1300CacheEntry] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    hits: int = 0
    misses: int = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self,
        raw_tape: RawTapeDay,
        spread_pair_clock: pl.DataFrame,
        *,
        source_cache_key: str,
        value_code: str,
        starting_position: PairedExitPosition,
        config: Aggressive1300ExitConfig,
    ) -> tuple[Aggressive1300ExitResult, bool]:
        if not isinstance(source_cache_key, str) or not source_cache_key.strip():
            raise ValueError("source_cache_key must be a non-empty immutable-source key")
        key = (
            source_cache_key,
            id(raw_tape),
            id(spread_pair_clock),
            str(raw_tape.date),
            str(value_code),
            starting_position,
            config,
        )
        cached = self._entries.get(key)
        if cached is not None:
            if cached.raw_tape is not raw_tape or cached.spread_pair_clock is not spread_pair_clock:
                raise AssertionError("aggressive 13:00 cache object identity collision")
            self.hits += 1
            return cached.result, True
        start = aggressive_1300_start_cursor(str(raw_tape.date))
        result = replay_aggressive_1300_product_day(
            raw_tape,
            spread_pair_clock,
            position_id=_TEMPLATE_POSITION_ID,
            position_established_recv_time_ns=start.recv_time_ns - 1,
            position_open_at_start=True,
            value_code=str(value_code),
            starting_position=starting_position,
            config=config,
        )
        self._entries[key] = _Aggressive1300CacheEntry(
            raw_tape=raw_tape,
            spread_pair_clock=spread_pair_clock,
            result=result,
        )
        self.misses += 1
        return result, False


def aggressive_1300_start_cursor(date: str) -> EventCursor:
    """Derive 13:00 from the existing 13:20 Taipei session contract.

    ``session_cutoff_cursor`` is the repository authority for converting the
    local session timestamp to the UTC-naive receive-time nanosecond axis.  By
    subtracting twenty minutes, this module cannot silently introduce a second
    timezone convention.
    """

    cutoff = session_cutoff_cursor(str(date))
    start_ns = cutoff.recv_time_ns - AGGRESSIVE_1300_START_BEFORE_SESSION_END_NS
    if start_ns < 0:
        raise ValueError("derived 13:00 cursor is negative")
    return EventCursor(start_ns, cutoff.event_sequence, cutoff.row_index)


def select_aggressive_1300_inventory(
    execution_actions: pl.DataFrame,
) -> Aggressive1300InventorySelection:
    """Collapse policy aliases to paired entries established before 13:00.

    Admission uses alias-local ``full_fill`` plus observed/executable hedge
    booleans.  A physical entry is retained at most once even when q aliases
    share ``raw_order_fact_id``.  This entry-only helper does **not** prove that
    a position is still open; production callers must additionally use
    :func:`select_open_positions_at_1300` with one selected normal path.
    """

    missing = sorted(_INVENTORY_REQUIRED - set(execution_actions.columns))
    if missing:
        raise ValueError(f"execution actions missing 13:00 inventory columns: {missing}")
    if execution_actions.is_empty():
        return Aggressive1300InventorySelection(
            _empty_inventory(),
            _inventory_audit(
                date=None,
                value_code=None,
                all_aliases=0,
                established_aliases=0,
                established_by_start_aliases=0,
                established_after_start_aliases=0,
                partial_before_start_aliases=0,
                physical_positions=0,
            ),
        )

    identities = execution_actions.select("Date", "ValueCode", "QuoteCode").unique()
    if identities.height != 1:
        raise ValueError("13:00 inventory selection requires one product-day")
    date, value_code, quote_code = map(str, identities.row(0))
    validated = _validate_entry_action_execution_contract(execution_actions)
    _validate_physical_entry_identity(validated)
    start = aggressive_1300_start_cursor(date)
    established = validated.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
        & (pl.col("entry_hedge_executable") == True)  # noqa: E712
    )
    by_start = established.filter(
        pl.col("entry_hedge_decision_time_ns") < start.recv_time_ns
    )
    after_start = established.filter(
        pl.col("entry_hedge_decision_time_ns") >= start.recv_time_ns
    )
    partial_before = validated.filter(
        (pl.col("partial_fill") == True)  # noqa: E712
        & pl.col("first_fill_recv_time_ns").is_not_null()
        & (pl.col("first_fill_recv_time_ns") <= start.recv_time_ns)
    ) if "first_fill_recv_time_ns" in validated.columns else validated.head(0)

    records: list[dict[str, object]] = []
    for (raw_order_fact_id,), group in by_start.sort(
        "policy_generation_id"
    ).group_by("raw_order_fact_id", maintain_order=True):
        first = group.row(0, named=True)
        aliases = tuple(map(str, group["policy_generation_id"].to_list()))
        records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "entry_raw_order_fact_id": str(raw_order_fact_id),
                "entry_route": str(first["route"]),
                "position_established_recv_time_ns": int(
                    first["entry_hedge_decision_time_ns"]
                ),
                "entry_future_price": float(first["entry_future_price"]),
                "entry_spot_price": float(first["entry_spot_price"]),
                "entry_hedge_contract_size_shares": int(
                    first["entry_hedge_contract_size_shares"]
                ),
                "eligible_policy_alias_count": len(aliases),
                "eligible_policy_generation_ids": list(aliases),
                "entry_cutoff_status": "paired_entry_established_before_1300",
            }
        )
    inventory = _inventory_from_records(records)
    audit = _inventory_audit(
        date=date,
        value_code=value_code,
        all_aliases=validated.height,
        established_aliases=established.height,
        established_by_start_aliases=by_start.height,
        established_after_start_aliases=after_start.height,
        partial_before_start_aliases=partial_before.height,
        physical_positions=inventory.height,
    )
    return Aggressive1300InventorySelection(inventory, audit)


def select_open_positions_at_1300(
    selected_position_paths: pl.DataFrame,
    *,
    target_date: str,
) -> Aggressive1300OpenInventorySelection:
    """Select physical positions which are genuinely open at 13:00.

    The input must already contain exactly one selected *normal* policy path
    per physical entry.  This function does not pick a q/rule/route winner.
    It only evaluates that selected path's state at ``target_date`` 13:00:

    * entries originating after ``target_date`` do not yet exist;
    * target-day entries established at or after 13:00 are forbidden;
    * priced normal terminals before or at 13:00 are already flat;
    * prior-day open positions are retained, including products with no new
      target-day entry; and
    * unpriced paths are retained only when their immutable observation window
      reaches ``target_date``.

    The result is analysis-only and remains one row per physical entry.  It
    never treats policy aliases as additional inventory.
    """

    target = _session_date_string(target_date, name="target_date")
    start_ns = aggressive_1300_start_cursor(target).recv_time_ns
    paths = _validate_selected_position_paths(selected_position_paths)
    if paths.is_empty():
        return Aggressive1300OpenInventorySelection(
            _empty_open_inventory(),
            _open_inventory_audit(target_date=target),
        )

    counters = {
        "selected_physical_paths": paths.height,
        "not_yet_originated_paths_excluded": 0,
        "prior_terminal_paths_excluded": 0,
        "target_day_entries_at_or_after_1300_excluded": 0,
        "normal_terminal_by_1300_paths_excluded": 0,
        "observation_ended_before_target_paths_excluded": 0,
        "target_day_open_positions": 0,
        "carried_open_positions": 0,
        "normal_terminal_after_1300_positions": 0,
        "unresolved_observed_through_target_positions": 0,
    }
    records: list[dict[str, object]] = []
    for item in paths.sort(
        ["Date", "ValueCode", "entry_raw_order_fact_id"]
    ).iter_rows(named=True):
        origin = str(item["Date"])
        priced = bool(item["terminal_cashflow_priced"])
        terminal_date = (
            None if item["terminal_date"] is None else str(item["terminal_date"])
        )
        terminal_ns = item["exit_decision_time_ns"]
        established_ns = int(item["position_established_ns"])
        last_observed = str(item["last_observed_session_date"])

        if origin > target:
            counters["not_yet_originated_paths_excluded"] += 1
            continue
        if priced and terminal_date is not None and terminal_date < target:
            counters["prior_terminal_paths_excluded"] += 1
            continue
        if origin == target and established_ns >= start_ns:
            counters["target_day_entries_at_or_after_1300_excluded"] += 1
            continue
        if (
            priced
            and terminal_date == target
            and terminal_ns is not None
            and int(terminal_ns) <= start_ns
        ):
            counters["normal_terminal_by_1300_paths_excluded"] += 1
            continue
        if not priced and last_observed < target:
            counters["observation_ended_before_target_paths_excluded"] += 1
            continue

        carried = origin < target
        normal_terminal_after = bool(
            priced
            and terminal_date == target
            and terminal_ns is not None
            and int(terminal_ns) > start_ns
        )
        unresolved = not priced
        if carried:
            counters["carried_open_positions"] += 1
        else:
            counters["target_day_open_positions"] += 1
        if normal_terminal_after:
            counters["normal_terminal_after_1300_positions"] += 1
        if unresolved:
            counters["unresolved_observed_through_target_positions"] += 1

        records.append(
            {
                "Date": target,
                "ValueCode": str(item["ValueCode"]),
                "QuoteCode": str(item["QuoteCode"]),
                "position_id": str(item["entry_raw_order_fact_id"]),
                "entry_raw_order_fact_id": str(item["entry_raw_order_fact_id"]),
                "entry_origin_date": origin,
                "position_established_recv_time_ns": established_ns,
                "source_selected_policy_path_id": str(item["policy_path_id"]),
                "source_exit_policy_trial_id": str(item["exit_policy_trial_id"]),
                "normal_terminal_cashflow_priced": priced,
                "normal_terminal_date": terminal_date,
                "normal_exit_decision_time_ns": (
                    None if terminal_ns is None else int(terminal_ns)
                ),
                "normal_outcome_type": str(item["outcome_type"]),
                "normal_outcome_status": str(item["outcome_status"]),
                "normal_last_observed_session_date": last_observed,
                "normal_outstanding_interval_end_exclusive": str(
                    item["outstanding_interval_end_exclusive"]
                ),
                "inventory_origin": (
                    "carried_from_prior_session"
                    if carried
                    else "target_day_entry_before_1300"
                ),
                "normal_path_status_at_1300": (
                    "normal_terminal_after_1300"
                    if normal_terminal_after
                    else (
                        "unresolved_observed_through_target"
                        if unresolved
                        else "terminal_in_future_session"
                    )
                ),
                "open_at_1300": True,
                "new_entry_at_or_after_1300": False,
                "source_selected_path_unique": True,
                "joint_volume_allocated": False,
            }
        )

    inventory = _open_inventory_from_records(records)
    audit = _open_inventory_audit(
        target_date=target,
        **counters,
        open_physical_positions=inventory.height,
    )
    return Aggressive1300OpenInventorySelection(inventory, audit)


def replay_aggressive_1300_inventory_batch(
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    open_inventory: pl.DataFrame,
    *,
    source_cache_key: str,
    value_code: str | None = None,
    cache: Aggressive1300ProductDayReplayCache | None = None,
    starting_position: PairedExitPosition = PairedExitPosition(),
    config: Aggressive1300ExitConfig = Aggressive1300ExitConfig(),
) -> Aggressive1300BatchResult:
    """Replay once per product-day and label every unique open position.

    This is the cache-backed batch entry point.  It avoids rebuilding the raw
    trade/book indexes for each position.  Because the replay template is
    shared, output rows are independent counterfactual labels and explicitly
    cannot be summed into a portfolio fill or P&L claim.
    """

    config.validate()
    inventory = _validate_open_inventory(open_inventory)
    date = str(raw_tape.date)
    if inventory.is_empty():
        return Aggressive1300BatchResult(
            _empty_batch_outcomes(),
            _batch_audit(
                date=date,
                value_code=value_code,
                quote_code=None,
                source_cache_key=source_cache_key,
                physical_positions=0,
                cache_hit=False,
                templates_built=0,
                template=None,
            ),
            None,
        )

    identities = inventory.select("Date", "ValueCode", "QuoteCode").unique()
    if identities.height != 1:
        raise ValueError("batch inventory must contain exactly one product-day")
    inventory_date, inventory_value, quote_code = map(str, identities.row(0))
    if inventory_date != date:
        raise ValueError("open inventory Date disagrees with raw tape date")
    selected_value = inventory_value if value_code is None else str(value_code)
    if selected_value != inventory_value:
        raise ValueError("value_code disagrees with open inventory")

    replay_cache = cache if cache is not None else Aggressive1300ProductDayReplayCache()
    before_misses = replay_cache.misses
    template, cache_hit = replay_cache.get_or_build(
        raw_tape,
        spread_pair_clock,
        source_cache_key=source_cache_key,
        value_code=selected_value,
        starting_position=starting_position,
        config=config,
    )
    if template.quote_code != quote_code:
        raise ValueError("open inventory QuoteCode disagrees with raw tape mapping")

    template_summary = template.summary_record()
    records: list[dict[str, object]] = []
    for item in inventory.sort("position_id").iter_rows(named=True):
        position_id = str(item["position_id"])
        summary = dict(template_summary)
        summary["position_id"] = position_id
        summary["winner_generation_id"] = _position_generation_id(
            template_summary["winner_generation_id"], position_id
        )
        records.append(
            {
                "entry_origin_date": str(item["entry_origin_date"]),
                "entry_raw_order_fact_id": str(item["entry_raw_order_fact_id"]),
                "position_established_recv_time_ns": int(
                    item["position_established_recv_time_ns"]
                ),
                "source_selected_policy_path_id": str(
                    item["source_selected_policy_path_id"]
                ),
                "source_exit_policy_trial_id": str(
                    item["source_exit_policy_trial_id"]
                ),
                "inventory_origin": str(item["inventory_origin"]),
                "normal_path_status_at_1300": str(
                    item["normal_path_status_at_1300"]
                ),
                "source_cache_key": source_cache_key,
                "market_replay_template_shared": True,
                "counterfactual_position_label": True,
                "position_outcomes_safe_to_sum": False,
                "formal_ev_ready": False,
                **summary,
            }
        )
    outcomes = pl.from_dicts(
        records,
        schema=_batch_outcome_schema(),
        strict=True,
    ).sort("position_id")
    if outcomes["position_id"].n_unique() != outcomes.height:
        raise AssertionError("batch replay emitted duplicate physical position labels")
    audit = _batch_audit(
        date=date,
        value_code=selected_value,
        quote_code=quote_code,
        source_cache_key=source_cache_key,
        physical_positions=outcomes.height,
        cache_hit=cache_hit,
        templates_built=replay_cache.misses - before_misses,
        template=template,
    )
    return Aggressive1300BatchResult(outcomes, audit, template)


def replay_aggressive_1300_product_day(
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    *,
    position_id: str,
    position_established_recv_time_ns: int,
    position_open_at_start: bool,
    value_code: str | None = None,
    starting_position: PairedExitPosition = PairedExitPosition(),
    config: Aggressive1300ExitConfig = Aggressive1300ExitConfig(),
) -> Aggressive1300ExitResult:
    """Replay the dynamic two-route challenger for one physical position."""

    config.validate()
    if not isinstance(raw_tape, RawTapeDay):
        raise TypeError("raw_tape must be a RawTapeDay")
    if not position_id:
        raise ValueError("position_id cannot be empty")
    if position_open_at_start is not True:
        raise ValueError("position must be proven open at the 13:00 replay start")
    if (
        isinstance(position_established_recv_time_ns, bool)
        or not isinstance(position_established_recv_time_ns, int)
        or position_established_recv_time_ns < 0
    ):
        raise ValueError("position_established_recv_time_ns must be non-negative")

    date = str(raw_tape.date)
    scheduled_start = aggressive_1300_start_cursor(date)
    cutoff = session_cutoff_cursor(date)
    if position_established_recv_time_ns >= scheduled_start.recv_time_ns:
        raise ValueError("13:00 entry cutoff prohibits this at-or-later position")

    mapping = raw_tape.mapping
    if value_code is None:
        values = mapping["ValueCode"].cast(pl.String).unique().to_list()
        if len(values) != 1:
            raise ValueError("value_code is required for a multi-product raw tape")
        selected_value = str(values[0])
    else:
        selected_value = str(value_code)
    selected_mapping = mapping.filter(
        pl.col("ValueCode").cast(pl.String) == selected_value
    )
    if selected_mapping.height != 1:
        raise ValueError("raw tape mapping must contain one selected ValueCode")
    quote_code = str(selected_mapping.row(0, named=True)["QuoteCode"])

    timeline = _build_timeline(raw_tape, spread_pair_clock, selected_value)
    observations_by_route = {
        route: _dynamic_peg_observations(
            timeline,
            date=date,
            route=route,
            start_cursor=scheduled_start,
            cutoff_cursor=cutoff,
        )
        for route in AGGRESSIVE_1300_ROUTES
    }
    builds_by_route = {
        route: build_layered_order_windows(
            observations_by_route[route],
            route=route,
            policy_id=f"{position_id}/{AGGRESSIVE_1300_POLICY_VERSION}/{route}",
            cutoff_cursor=cutoff,
        )
        for route in AGGRESSIVE_1300_ROUTES
    }

    trade_replays = {
        "spot": _trade_replay(raw_tape.spot_trades, "spot"),
        "future": _trade_replay(raw_tape.future_trades, "future"),
    }
    opposite_snapshots = {
        market: _opposite_snapshots(timeline, market)
        for market in ("spot", "future")
    }
    outcomes_by_route: dict[str, tuple[ExitMakerOrderOutcome, ...]] = {}
    pairs = []
    for route in AGGRESSIVE_1300_ROUTES:
        contract = EXIT_MAKER_ROUTE_CONTRACTS[route]
        windows = builds_by_route[route].windows
        outcomes = replay_exit_maker_windows(
            windows,
            route=route,
            maker_replay=trade_replays[contract.maker_market],
            opposite_snapshots=opposite_snapshots[contract.opposite_market],
            eod_cursor=cutoff,
            starting_position=starting_position,
            hedge_delay_ns=config.hedge_delay_ns,
            max_book_age_ns=config.max_book_age_ns,
        )
        outcomes_by_route[route] = outcomes
        pairs.extend(zip(windows, outcomes))
    projection = project_earliest_full_fill_oco(pairs)
    outcome_by_id = {
        outcome.generation_id: outcome
        for outcomes in outcomes_by_route.values()
        for outcome in outcomes
    }
    winner = (
        None
        if projection.winner_generation_id is None
        else outcome_by_id[projection.winner_generation_id]
    )
    nominal_branch = _nominal_branch(projection, winner, outcome_by_id.values())
    active_siblings = sum(
        member.disposition == "sibling_cancel_required"
        for member in projection.members
    )
    window_by_id = {
        window.generation_id: window
        for build in builds_by_route.values()
        for window in build.windows
    }
    prior_unacked = 0
    if projection.winner_full_fill_cursor is not None:
        winner_cursor = projection.winner_full_fill_cursor
        for generation_id, window in window_by_id.items():
            if generation_id == projection.winner_generation_id:
                continue
            outcome = outcome_by_id[generation_id]
            if (
                window.start_cursor < winner_cursor
                and window.stop_cursor <= winner_cursor
                and not outcome.cancel_ack_observed
            ):
                prior_unacked += 1
    strict_branch = (
        "cancel_race_unknown"
        if nominal_branch == "flat_same_day"
        and (
            active_siblings > 0
            or not projection.position_projection_safe
            or prior_unacked > 0
        )
        else nominal_branch
    )
    nominal_close_count = int(nominal_branch == "flat_same_day")
    if nominal_close_count > 1:
        raise AssertionError("cross-route OCO produced duplicate position closes")
    submits = [
        window.start_cursor
        for build in builds_by_route.values()
        for window in build.windows
    ]
    return Aggressive1300ExitResult(
        date=date,
        value_code=selected_value,
        quote_code=quote_code,
        position_id=position_id,
        scheduled_start_cursor=scheduled_start,
        actual_first_submit_cursor=min(submits) if submits else None,
        cutoff_cursor=cutoff,
        observations_by_route=observations_by_route,
        builds_by_route=builds_by_route,
        outcomes_by_route=outcomes_by_route,
        projection=projection,
        nominal_branch_status=nominal_branch,
        strict_branch_status=strict_branch,
        active_sibling_cancel_count=active_siblings,
        prior_unacked_cancel_count=prior_unacked,
        nominal_close_count=nominal_close_count,
    )


def _dynamic_peg_observations(
    timeline: tuple[_CombinedPoint, ...],
    *,
    date: str,
    route: str,
    start_cursor: EventCursor,
    cutoff_cursor: EventCursor,
) -> tuple[TargetObservation, ...]:
    """Build sparse B1/A1 peg states while retaining LayeredSampler semantics."""

    if route not in AGGRESSIVE_1300_ROUTES:
        raise ValueError(f"unsupported aggressive exit route: {route}")
    if start_cursor >= cutoff_cursor or not timeline:
        return ()
    cursors = tuple(point.cursor for point in timeline)
    seed_index = bisect_right(cursors, start_cursor) - 1
    first_later = bisect_right(cursors, start_cursor)
    stop = bisect_left(cursors, cutoff_cursor)

    points: list[tuple[EventCursor, _CombinedPoint]] = []
    if seed_index >= 0:
        points.append((start_cursor, timeline[seed_index]))
    points.extend((timeline[index].cursor, timeline[index]) for index in range(first_later, stop))

    observations: list[TargetObservation] = []
    last: TargetObservation | None = None
    last_signature: tuple[object, ...] | None = None
    for cursor, point in points:
        observation = _aggressive_observation_at_point(
            point,
            cursor=cursor,
            date=date,
            route=route,
        )
        if observation is None:
            if last is None:
                continue
            epoch = (
                point.spot.spread_pair_epoch
                if point.spot is not None
                and point.spot.spread_pair_epoch is not None
                else last.spread_pair_epoch
            )
            observation = TargetObservation(
                cursor=cursor,
                spread_pair_epoch=epoch,
                absolute_target_tick=last.absolute_target_tick,
                gate_open=False,
                gate_reason="raw_state_unavailable",
                initial_queue_ahead=None,
                target_rank="unavailable",
                source="aggressive_1300_dynamic_b1a1_peg",
            )
        last = observation
        signature = (
            observation.spread_pair_epoch,
            observation.absolute_target_tick,
            observation.gate_open,
            observation.gate_reason,
        )
        if signature == last_signature:
            continue
        observations.append(observation)
        last_signature = signature
    return tuple(observations)


def _aggressive_observation_at_point(
    point: _CombinedPoint,
    *,
    cursor: EventCursor,
    date: str,
    route: str,
) -> TargetObservation | None:
    if point.spot is None or point.future is None:
        return None
    spot = point.spot.row
    future = point.future.row
    epoch = point.spot.spread_pair_epoch
    if epoch is None:
        return None
    maker_point = point.future if route == FUTURE_BID_EXIT_ROUTE else point.spot
    maker = maker_point.row
    maker_side = EXIT_MAKER_ROUTE_CONTRACTS[route].maker_side
    price_value = maker.get("exec_bid_price" if maker_side == "bid" else "exec_ask_price")
    maker_price = _positive_float(price_value)
    spot_bid = _positive_float(spot.get("exec_bid_price"))
    spot_ask = _positive_float(spot.get("exec_ask_price"))
    future_bid = _positive_float(future.get("exec_bid_price"))
    future_ask = _positive_float(future.get("exec_ask_price"))
    if maker_price is None or None in (spot_bid, spot_ask, future_bid, future_ask):
        return None
    assert spot_bid is not None and spot_ask is not None
    assert future_bid is not None and future_ask is not None
    valid_bbo = spot_bid < spot_ask and future_bid < future_ask
    spot_vwap = _required_vwap(
        executable_levels_from_state(spot, "spot", "bid"),
        SPOT_HEDGE_LOTS,
    )
    future_vwap = _required_vwap(
        executable_levels_from_state(future, "future", "ask"),
        FUTURE_HEDGE_CONTRACTS,
    )
    rank, queue = _rank_and_queue(maker_price, maker_side, maker)
    expected_rank = "BID1" if maker_side == "bid" else "ASK1"
    gates = (
        (point.spot.spread_pair_active, "spread_pair_clock_inactive"),
        (not bool(spot.get("trial_match", False)), "spot_trial_match"),
        (point.spot.formal_after_trial, "spot_formal_book_gate"),
        (not bool(future.get("trial_match", False)), "future_trial_match"),
        (point.future.formal_after_trial, "future_formal_book_gate"),
        (_raw_ref_gate(spot), "spot_ref_gate"),
        (_raw_ref_gate(future), "future_ref_gate"),
        (valid_bbo, "crossed_or_locked_executable_bbo"),
        (spot_vwap is not None, "spot_insufficient_bid_depth"),
        (future_vwap is not None, "future_insufficient_ask_depth"),
        (rank == expected_rank and queue is not None, "maker_b1a1_queue_unknown"),
    )
    gate_open = all(valid for valid, _ in gates)
    reason = "open" if gate_open else next(reason for valid, reason in gates if not valid)
    return TargetObservation(
        cursor=cursor,
        spread_pair_epoch=int(epoch),
        absolute_target_tick=absolute_price_tick(
            maker_price,
            market=EXIT_MAKER_ROUTE_CONTRACTS[route].maker_market,
            session_date=date,
        ),
        gate_open=gate_open,
        gate_reason=reason,
        initial_queue_ahead=queue if gate_open else None,
        target_rank=rank,
        source="aggressive_1300_dynamic_b1a1_peg",
    )


def _nominal_branch(
    projection: ExitMakerOcoProjection,
    winner: ExitMakerOrderOutcome | None,
    outcomes: Iterable[ExitMakerOrderOutcome],
) -> str:
    values = tuple(outcomes)
    if winner is not None:
        if not projection.position_projection_safe:
            return "oco_inventory_ambiguous"
        return winner.branch_status
    if not values:
        return "carry_at_eod_no_admission"
    if any(item.known_filled_maker_quantity is None for item in values):
        return "fill_unknown_at_eod"
    if any((item.known_filled_maker_quantity or 0) > 0 for item in values):
        return "partial_fill_carry_at_eod"
    return "carry_at_eod_cancel_unconfirmed"


def _validate_selected_position_paths(paths: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(_SELECTED_POSITION_PATH_SCHEMA) - set(paths.columns))
    if missing:
        raise ValueError(f"selected position paths missing columns: {missing}")
    wrong = [
        f"{name}:{paths.schema[name]}!={dtype}"
        for name, dtype in _SELECTED_POSITION_PATH_SCHEMA.items()
        if paths.schema[name] != dtype
    ]
    if wrong:
        raise ValueError(f"selected position path dtypes disagree: {wrong}")
    selected = paths.select(list(_SELECTED_POSITION_PATH_SCHEMA))
    if selected.is_empty():
        return selected

    nonnull = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_raw_order_fact_id",
        "position_established_ns",
        "policy_path_id",
        "exit_policy_trial_id",
        "terminal_cashflow_priced",
        "outcome_type",
        "outcome_status",
        "last_observed_session_date",
        "outstanding_interval_end_exclusive",
        "entry_no_fill_included",
        "gross_zero_imputation",
    ]
    nulls = {name: selected[name].null_count() for name in nonnull}
    nulls = {name: count for name, count in nulls.items() if count}
    if nulls:
        raise ValueError(f"selected position paths contain null contract fields: {nulls}")
    for name in (
        "ValueCode",
        "QuoteCode",
        "entry_raw_order_fact_id",
        "policy_path_id",
        "exit_policy_trial_id",
        "outcome_type",
        "outcome_status",
    ):
        if selected.filter(pl.col(name).str.strip_chars() == "").height:
            raise ValueError(f"selected position paths contain empty {name}")
    for name in (
        "Date",
        "terminal_date",
        "last_observed_session_date",
        "outstanding_interval_end_exclusive",
    ):
        for value in selected[name].drop_nulls().unique().to_list():
            _session_date_string(value, name=name)
    if selected.filter(pl.col("position_established_ns") < 0).height:
        raise ValueError("position_established_ns must be non-negative")
    if selected["entry_raw_order_fact_id"].n_unique() != selected.height:
        raise ValueError(
            "selected normal paths must contain exactly one row per physical entry"
        )
    if selected["policy_path_id"].n_unique() != selected.height:
        raise ValueError("selected normal policy_path_id must be unique")
    if selected.filter(
        pl.col("entry_no_fill_included")
        | pl.col("gross_zero_imputation")
    ).height:
        raise ValueError("selected path universe includes imputed/no-fill entry rows")

    priced = pl.col("terminal_cashflow_priced") == True  # noqa: E712
    if selected.filter(
        priced
        & (
            pl.col("terminal_date").is_null()
            | pl.col("exit_decision_time_ns").is_null()
            | (pl.col("outcome_type") != "terminal")
        )
    ).height:
        raise ValueError("priced normal paths require coherent terminal fields")
    if selected.filter(
        (~priced)
        & (
            pl.col("terminal_date").is_not_null()
            | pl.col("exit_decision_time_ns").is_not_null()
            | (pl.col("outcome_type") != "censored")
        )
    ).height:
        raise ValueError("unpriced normal paths must remain coherently censored")
    if selected.filter(
        pl.col("exit_decision_time_ns").is_not_null()
        & (pl.col("exit_decision_time_ns") < pl.col("position_established_ns"))
    ).height:
        raise ValueError("normal terminal precedes position establishment")
    if selected.filter(
        pl.col("last_observed_session_date") < pl.col("Date")
    ).height:
        raise ValueError("last observation precedes entry origin date")
    if selected.filter(
        pl.col("terminal_date").is_not_null()
        & (pl.col("terminal_date") > pl.col("last_observed_session_date"))
    ).height:
        raise ValueError("normal terminal is beyond last observed session")
    return selected


def _session_date_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} must be a YYYYMMDD string")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date: {value}") from exc
    return value


def _open_inventory_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "position_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "entry_origin_date": pl.String,
        "position_established_recv_time_ns": pl.Int64,
        "source_selected_policy_path_id": pl.String,
        "source_exit_policy_trial_id": pl.String,
        "normal_terminal_cashflow_priced": pl.Boolean,
        "normal_terminal_date": pl.String,
        "normal_exit_decision_time_ns": pl.Int64,
        "normal_outcome_type": pl.String,
        "normal_outcome_status": pl.String,
        "normal_last_observed_session_date": pl.String,
        "normal_outstanding_interval_end_exclusive": pl.String,
        "inventory_origin": pl.String,
        "normal_path_status_at_1300": pl.String,
        "open_at_1300": pl.Boolean,
        "new_entry_at_or_after_1300": pl.Boolean,
        "source_selected_path_unique": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
    }


def _empty_open_inventory() -> pl.DataFrame:
    return pl.DataFrame(schema=_open_inventory_schema())


def _open_inventory_from_records(records: list[dict[str, object]]) -> pl.DataFrame:
    if not records:
        return _empty_open_inventory()
    return pl.from_dicts(
        records,
        schema=_open_inventory_schema(),
        strict=True,
    ).sort(["ValueCode", "position_id"])


def _validate_open_inventory(inventory: pl.DataFrame) -> pl.DataFrame:
    expected = _open_inventory_schema()
    if inventory.columns != list(expected):
        raise ValueError("open inventory ordered schema is not canonical")
    wrong = [
        f"{name}:{inventory.schema[name]}!={dtype}"
        for name, dtype in expected.items()
        if inventory.schema[name] != dtype
    ]
    if wrong:
        raise ValueError(f"open inventory dtypes are not canonical: {wrong}")
    if inventory.is_empty():
        return inventory
    if inventory.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        nullable = {"normal_terminal_date", "normal_exit_decision_time_ns"}
        unexpected = {
            name: inventory[name].null_count()
            for name in inventory.columns
            if name not in nullable and inventory[name].null_count()
        }
        if unexpected:
            raise ValueError(f"open inventory contains null contract fields: {unexpected}")
    for name in (
        "position_id",
        "entry_raw_order_fact_id",
        "source_selected_policy_path_id",
        "source_exit_policy_trial_id",
    ):
        if inventory.filter(pl.col(name).str.strip_chars() == "").height:
            raise ValueError(f"open inventory contains empty {name}")
    if inventory["position_id"].n_unique() != inventory.height:
        raise ValueError("open inventory contains duplicate physical position_id")
    if inventory.filter(
        (pl.col("position_id") != pl.col("entry_raw_order_fact_id"))
        | (pl.col("open_at_1300") != True).fill_null(True)  # noqa: E712
        | (pl.col("new_entry_at_or_after_1300") != False).fill_null(True)  # noqa: E712
        | (pl.col("source_selected_path_unique") != True).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("open inventory safety flags or physical identity are invalid")
    if inventory["Date"].n_unique() != 1:
        raise ValueError("open inventory must contain one target date")
    target = _session_date_string(inventory.item(0, "Date"), name="Date")
    start_ns = aggressive_1300_start_cursor(target).recv_time_ns
    for name in (
        "entry_origin_date",
        "normal_terminal_date",
        "normal_last_observed_session_date",
        "normal_outstanding_interval_end_exclusive",
    ):
        for value in inventory[name].drop_nulls().unique().to_list():
            _session_date_string(value, name=name)
    if inventory.filter(pl.col("entry_origin_date") > target).height:
        raise ValueError("open inventory contains a not-yet-originated position")
    if inventory.filter(
        (pl.col("entry_origin_date") == target)
        & (pl.col("position_established_recv_time_ns") >= start_ns)
    ).height:
        raise ValueError("open inventory contains a target-day entry at or after 13:00")

    priced = pl.col("normal_terminal_cashflow_priced") == True  # noqa: E712
    if inventory.filter(
        priced
        & (
            pl.col("normal_terminal_date").is_null()
            | pl.col("normal_exit_decision_time_ns").is_null()
            | (pl.col("normal_outcome_type") != "terminal")
        )
    ).height:
        raise ValueError("open inventory has incoherent priced normal terminal")
    if inventory.filter(
        (~priced)
        & (
            pl.col("normal_terminal_date").is_not_null()
            | pl.col("normal_exit_decision_time_ns").is_not_null()
            | (pl.col("normal_outcome_type") != "censored")
        )
    ).height:
        raise ValueError("open inventory has incoherent censored normal path")
    if inventory.filter(
        priced
        & (
            (pl.col("normal_terminal_date") < target)
            | (
                (pl.col("normal_terminal_date") == target)
                & (pl.col("normal_exit_decision_time_ns") <= start_ns)
            )
        )
    ).height:
        raise ValueError("open inventory contains a position already flat by 13:00")
    if inventory.filter(
        (~priced) & (pl.col("normal_last_observed_session_date") < target)
    ).height:
        raise ValueError("open inventory contains an unobserved carried position")
    if inventory.filter(
        (
            (pl.col("entry_origin_date") < target)
            & (pl.col("inventory_origin") != "carried_from_prior_session")
        )
        | (
            (pl.col("entry_origin_date") == target)
            & (pl.col("inventory_origin") != "target_day_entry_before_1300")
        )
    ).height:
        raise ValueError("open inventory origin label disagrees with entry date")
    expected_status = (
        pl.when(~priced)
        .then(pl.lit("unresolved_observed_through_target"))
        .when(pl.col("normal_terminal_date") == target)
        .then(pl.lit("normal_terminal_after_1300"))
        .otherwise(pl.lit("terminal_in_future_session"))
    )
    if inventory.filter(
        pl.col("normal_path_status_at_1300") != expected_status
    ).height:
        raise ValueError("open inventory normal-path state label is not reproducible")
    return inventory


def _open_inventory_audit(
    *,
    target_date: str,
    selected_physical_paths: int = 0,
    not_yet_originated_paths_excluded: int = 0,
    prior_terminal_paths_excluded: int = 0,
    target_day_entries_at_or_after_1300_excluded: int = 0,
    normal_terminal_by_1300_paths_excluded: int = 0,
    observation_ended_before_target_paths_excluded: int = 0,
    target_day_open_positions: int = 0,
    carried_open_positions: int = 0,
    normal_terminal_after_1300_positions: int = 0,
    unresolved_observed_through_target_positions: int = 0,
    open_physical_positions: int = 0,
) -> pl.DataFrame:
    excluded = (
        not_yet_originated_paths_excluded
        + prior_terminal_paths_excluded
        + target_day_entries_at_or_after_1300_excluded
        + normal_terminal_by_1300_paths_excluded
        + observation_ended_before_target_paths_excluded
    )
    if excluded + open_physical_positions != selected_physical_paths:
        raise AssertionError("13:00 inventory audit does not partition selected paths")
    return pl.from_dicts(
        [
            {
                "Date": target_date,
                "selected_physical_paths": selected_physical_paths,
                "not_yet_originated_paths_excluded": not_yet_originated_paths_excluded,
                "prior_terminal_paths_excluded": prior_terminal_paths_excluded,
                "target_day_entries_at_or_after_1300_excluded": target_day_entries_at_or_after_1300_excluded,
                "normal_terminal_by_1300_paths_excluded": normal_terminal_by_1300_paths_excluded,
                "observation_ended_before_target_paths_excluded": observation_ended_before_target_paths_excluded,
                "target_day_open_positions": target_day_open_positions,
                "carried_open_positions": carried_open_positions,
                "normal_terminal_after_1300_positions": normal_terminal_after_1300_positions,
                "unresolved_observed_through_target_positions": unresolved_observed_through_target_positions,
                "open_physical_positions_at_1300": open_physical_positions,
                "new_entries_at_or_after_1300_admitted": 0,
                "selected_paths_are_one_per_physical": True,
                "joint_volume_allocated": False,
            }
        ],
        schema={
            "Date": pl.String,
            "selected_physical_paths": pl.Int64,
            "not_yet_originated_paths_excluded": pl.Int64,
            "prior_terminal_paths_excluded": pl.Int64,
            "target_day_entries_at_or_after_1300_excluded": pl.Int64,
            "normal_terminal_by_1300_paths_excluded": pl.Int64,
            "observation_ended_before_target_paths_excluded": pl.Int64,
            "target_day_open_positions": pl.Int64,
            "carried_open_positions": pl.Int64,
            "normal_terminal_after_1300_positions": pl.Int64,
            "unresolved_observed_through_target_positions": pl.Int64,
            "open_physical_positions_at_1300": pl.Int64,
            "new_entries_at_or_after_1300_admitted": pl.Int64,
            "selected_paths_are_one_per_physical": pl.Boolean,
            "joint_volume_allocated": pl.Boolean,
        },
        strict=True,
    )


def _batch_outcome_schema() -> dict[str, pl.DataType]:
    return {
        "entry_origin_date": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "position_established_recv_time_ns": pl.Int64,
        "source_selected_policy_path_id": pl.String,
        "source_exit_policy_trial_id": pl.String,
        "inventory_origin": pl.String,
        "normal_path_status_at_1300": pl.String,
        "source_cache_key": pl.String,
        "market_replay_template_shared": pl.Boolean,
        "counterfactual_position_label": pl.Boolean,
        "position_outcomes_safe_to_sum": pl.Boolean,
        "formal_ev_ready": pl.Boolean,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "position_id": pl.String,
        "policy_version": pl.String,
        "scheduled_start_recv_time_ns": pl.Int64,
        "actual_first_submit_recv_time_ns": pl.Int64,
        "entry_after_1300_allowed": pl.Boolean,
        "dynamic_b1a1_peg": pl.Boolean,
        "future_candidate_generations": pl.Int64,
        "spot_candidate_generations": pl.Int64,
        "winner_generation_id": pl.String,
        "winner_route": pl.String,
        "winner_full_fill_recv_time_ns": pl.Int64,
        "winner_hedge_delay_ns": pl.Int64,
        "winner_hedge_status": pl.String,
        "active_sibling_cancel_count": pl.Int64,
        "prior_unacked_cancel_count_before_winner": pl.Int64,
        "sibling_full_fill_after_cancel_request_count": pl.Int64,
        "sibling_partial_before_winner_count": pl.Int64,
        "oco_position_projection_safe": pl.Boolean,
        "nominal_branch_status": pl.String,
        "strict_branch_status": pl.String,
        "nominal_close_count": pl.Int64,
        "duplicate_close_prevented": pl.Boolean,
        "cancel_ack_observed": pl.Boolean,
        "cancel_model": pl.String,
        "joint_volume_allocated": pl.Boolean,
        "strict_ev_ready": pl.Boolean,
    }


def _empty_batch_outcomes() -> pl.DataFrame:
    return pl.DataFrame(schema=_batch_outcome_schema())


def _position_generation_id(value: object, position_id: str) -> str | None:
    if value is None:
        return None
    generation_id = str(value)
    prefix = f"{_TEMPLATE_POSITION_ID}/"
    if not generation_id.startswith(prefix):
        raise AssertionError("cached market template generation id has wrong prefix")
    return f"{position_id}/{generation_id[len(prefix):]}"


def _batch_audit(
    *,
    date: str,
    value_code: str | None,
    quote_code: str | None,
    source_cache_key: str,
    physical_positions: int,
    cache_hit: bool,
    templates_built: int,
    template: Aggressive1300ExitResult | None,
) -> pl.DataFrame:
    if not isinstance(source_cache_key, str) or not source_cache_key.strip():
        raise ValueError("source_cache_key must be non-empty")
    return pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "source_cache_key": source_cache_key,
                "open_physical_position_rows": physical_positions,
                "market_templates_built_this_call": templates_built,
                "market_template_cache_hit": cache_hit,
                "market_template_future_generations": (
                    0
                    if template is None
                    else len(template.builds_by_route[FUTURE_BID_EXIT_ROUTE].windows)
                ),
                "market_template_spot_generations": (
                    0
                    if template is None
                    else len(template.builds_by_route[SPOT_ASK_EXIT_ROUTE].windows)
                ),
                "market_template_nominal_status": (
                    None if template is None else template.nominal_branch_status
                ),
                "market_template_strict_status": (
                    None if template is None else template.strict_branch_status
                ),
                "one_market_replay_per_product_day": templates_built <= 1,
                "position_labels_are_counterfactual": True,
                "position_rows_safe_to_sum": False,
                "joint_volume_allocated": False,
                "portfolio_terminal_metric_available": False,
                "formal_ev_ready": False,
            }
        ],
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "source_cache_key": pl.String,
            "open_physical_position_rows": pl.Int64,
            "market_templates_built_this_call": pl.Int64,
            "market_template_cache_hit": pl.Boolean,
            "market_template_future_generations": pl.Int64,
            "market_template_spot_generations": pl.Int64,
            "market_template_nominal_status": pl.String,
            "market_template_strict_status": pl.String,
            "one_market_replay_per_product_day": pl.Boolean,
            "position_labels_are_counterfactual": pl.Boolean,
            "position_rows_safe_to_sum": pl.Boolean,
            "joint_volume_allocated": pl.Boolean,
            "portfolio_terminal_metric_available": pl.Boolean,
            "formal_ev_ready": pl.Boolean,
        },
        strict=True,
    )


def _inventory_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "entry_route": pl.String,
        "position_established_recv_time_ns": pl.Int64,
        "entry_future_price": pl.Float64,
        "entry_spot_price": pl.Float64,
        "entry_hedge_contract_size_shares": pl.Int64,
        "eligible_policy_alias_count": pl.Int64,
        "eligible_policy_generation_ids": pl.List(pl.String),
        "entry_cutoff_status": pl.String,
    }


def _empty_inventory() -> pl.DataFrame:
    return pl.DataFrame(schema=_inventory_schema())


def _inventory_from_records(records: list[dict[str, object]]) -> pl.DataFrame:
    if not records:
        return _empty_inventory()
    return pl.from_dicts(records, schema=_inventory_schema(), strict=True).sort(
        "entry_raw_order_fact_id"
    )


def _inventory_audit(
    *,
    date: str | None,
    value_code: str | None,
    all_aliases: int,
    established_aliases: int,
    established_by_start_aliases: int,
    established_after_start_aliases: int,
    partial_before_start_aliases: int,
    physical_positions: int,
) -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "all_policy_aliases": all_aliases,
                "established_policy_aliases": established_aliases,
                "established_by_1300_policy_aliases": established_by_start_aliases,
                "established_after_1300_policy_aliases_excluded": established_after_start_aliases,
                "partial_before_1300_policy_aliases_not_in_paired_inventory": partial_before_start_aliases,
                "physical_paired_positions_at_1300": physical_positions,
                "post_1300_new_entries_admitted": 0,
                "entry_cutoff_cancel_ack_observed": False,
                "entry_partial_inventory_fully_resolved": partial_before_start_aliases == 0,
            }
        ],
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "all_policy_aliases": pl.Int64,
            "established_policy_aliases": pl.Int64,
            "established_by_1300_policy_aliases": pl.Int64,
            "established_after_1300_policy_aliases_excluded": pl.Int64,
            "partial_before_1300_policy_aliases_not_in_paired_inventory": pl.Int64,
            "physical_paired_positions_at_1300": pl.Int64,
            "post_1300_new_entries_admitted": pl.Int64,
            "entry_cutoff_cancel_ack_observed": pl.Boolean,
            "entry_partial_inventory_fully_resolved": pl.Boolean,
        },
        strict=True,
    )


def _positive_float(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return None
