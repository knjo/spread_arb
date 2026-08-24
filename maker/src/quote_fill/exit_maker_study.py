"""Frame-level product-day study for maker-then-taker exits.

This module is the data-frame orchestration layer around :mod:`.exit_maker`.
It intentionally keeps three different units of observation separate:

* ``policy_support`` retains every entry-policy alias crossed with every
  required D-1 exit rule and both maker-exit routes, including policies which
  cannot admit an order;
* ``raw_candidate_facts`` contains independent displayed-queue replay facts
  keyed to the physical entry position; and
* ``position_policy_facts`` applies an earliest-full-fill-wins OCO projection
  so overlapping independent candidates cannot be added together as if one
  two-lot/one-contract position could be closed repeatedly.

The cancel model is deliberately narrow.  A target retreat or session cutoff
is an instantaneous *cancel request* in V0, but no cancel ACK or cancel/fill
race is observed.  Consequently every output keeps ``strict_ev_ready=False``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping

import polars as pl

from .engine import IntentTransition
from .exit_maker import (
    EXIT_MAKER_ROUTE_CONTRACTS,
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
    SUPPORTED_EXIT_MAKER_ROUTES,
    ExitMakerObservation,
    ExitMakerOrderOutcome,
    ExitMakerOcoProjection,
    _index_exit_maker_opposite_snapshots,
    _replay_exit_maker_windows_indexed,
    build_exit_maker_order_windows,
    make_exit_maker_observation,
    project_earliest_full_fill_oco,
)
from .hedge import (
    BookLevel,
    OppositeBookSnapshot,
    _IndexedOppositeBookSnapshots,
)
from .hedge_study import (
    CONTRACT_SHARES,
    FUTURE_HEDGE_CONTRACTS,
    SPOT_HEDGE_LOTS,
    executable_levels_from_state,
)
from .indexed_replay import IndexedTradeReplay
from .layered import EventCursor
from .merged import session_cutoff_cursor
from .raw_tape import RawTapeDay
from .replay import IndependentOrderWindow, TradeEvent
from .targets import absolute_price_tick, price_in_ref_band, tick_index_to_price


_MARKET_PRIORITY: Mapping[str, int] = {"future": 1, "spot": 2}
_EXPECTED_EXIT_RULE_IDS = ("frozen_center", "frozen_lower")
_ARTIFACT_FRAME_NAMES = (
    "exit_maker_policy_support",
    "exit_maker_observations",
    "exit_maker_transitions",
    "exit_maker_candidate_aliases",
    "exit_maker_raw_candidate_facts",
    "exit_maker_position_policy_facts",
    "exit_maker_audit",
)

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
_CLOCK_REQUIRED = {
    "Date",
    "ValueCode",
    "spot_channel_seq",
    "spread_pair_epoch",
}


@dataclass(frozen=True)
class ExitMakerStudyConfig:
    """Fixed assumptions for one frame-level maker-exit replay."""

    hedge_delay_ns: int = 50_000_000
    max_book_age_ns: int | None = None
    expected_exit_rule_ids: tuple[str, ...] = _EXPECTED_EXIT_RULE_IDS
    exit_lifecycle_policy_version: str = "exit_maker_layered_oco_v0"
    exit_queue_scenario: str = "displayed_queue_independent_v0"
    instant_cancel_v0: bool = True

    def validate(self) -> None:
        if (
            isinstance(self.hedge_delay_ns, bool)
            or not isinstance(self.hedge_delay_ns, int)
            or self.hedge_delay_ns < 0
        ):
            raise ValueError("hedge_delay_ns must be a non-negative integer")
        if self.max_book_age_ns is not None and (
            isinstance(self.max_book_age_ns, bool)
            or not isinstance(self.max_book_age_ns, int)
            or self.max_book_age_ns < 0
        ):
            raise ValueError("max_book_age_ns must be non-negative or None")
        if (
            not self.expected_exit_rule_ids
            or len(set(self.expected_exit_rule_ids))
            != len(self.expected_exit_rule_ids)
            or any(not value for value in self.expected_exit_rule_ids)
        ):
            raise ValueError("expected_exit_rule_ids must be non-empty and unique")
        if not self.exit_lifecycle_policy_version:
            raise ValueError("exit_lifecycle_policy_version cannot be empty")
        if not self.exit_queue_scenario:
            raise ValueError("exit_queue_scenario cannot be empty")
        if self.instant_cancel_v0 is not True:
            raise ValueError("only the explicit instant_cancel_v0 model is supported")

    def as_dict(self) -> dict[str, object]:
        """Return a stable, serializable configuration payload for runners."""

        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ExitMakerProductDayResult:
    """Materialized exit-maker facts for one exact product-day."""

    policy_support: pl.DataFrame
    observations: pl.DataFrame
    transitions: pl.DataFrame
    candidate_aliases: pl.DataFrame
    raw_candidate_facts: pl.DataFrame
    position_policy_facts: pl.DataFrame
    audit: pl.DataFrame

    def frames(self) -> Mapping[str, pl.DataFrame]:
        """Expose stable artifact names without coupling callers to ordering."""

        return {
            "exit_maker_policy_support": self.policy_support,
            "exit_maker_observations": self.observations,
            "exit_maker_transitions": self.transitions,
            "exit_maker_candidate_aliases": self.candidate_aliases,
            "exit_maker_raw_candidate_facts": self.raw_candidate_facts,
            "exit_maker_position_policy_facts": self.position_policy_facts,
            "exit_maker_audit": self.audit,
        }


@dataclass(frozen=True)
class ExitMakerProductDayArtifacts:
    """Seven already-materialized Parquet artifacts in one atomic stage."""

    paths: tuple[tuple[str, Path], ...]

    def artifact_paths(self) -> Mapping[str, Path]:
        return dict(self.paths)


@dataclass(frozen=True)
class _MarketPoint:
    cursor: EventCursor
    market: Literal["spot", "future"]
    row: dict[str, object]
    formal_after_trial: bool
    spread_pair_epoch: int | None = None
    spread_pair_active: bool = False


@dataclass(frozen=True)
class _CombinedPoint:
    cursor: EventCursor
    spot: _MarketPoint | None
    future: _MarketPoint | None


@dataclass(frozen=True)
class _ObservationTimelineIndex:
    """One product-day's full/target timelines and reusable cursor indexes."""

    full_timeline: tuple[_CombinedPoint, ...]
    target_timeline: tuple[_CombinedPoint, ...]
    full_cursors: tuple[EventCursor, ...]
    target_cursors: tuple[EventCursor, ...]


@dataclass(frozen=True)
class _PhysicalReplay:
    observations: tuple[ExitMakerObservation, ...]
    windows: tuple[IndependentOrderWindow, ...]
    transitions: tuple[IntentTransition, ...]
    outcomes: tuple[ExitMakerOrderOutcome, ...]
    projection: ExitMakerOcoProjection


@dataclass(frozen=True)
class _OutcomeReplayTemplate:
    """ID-bearing replay result reusable by structurally identical windows."""

    windows: tuple[IndependentOrderWindow, ...]
    outcomes: tuple[ExitMakerOrderOutcome, ...]


@dataclass
class ExitMakerProductDayReplayCache:
    """Physical replay cache shareable across equivalent policy classifiers."""

    physical: dict[tuple[object, ...], _PhysicalReplay] = field(
        default_factory=dict
    )
    outcomes: dict[tuple[object, ...], _OutcomeReplayTemplate] = field(
        default_factory=dict
    )
    input_signature: tuple[object, ...] | None = None


@dataclass(frozen=True)
class ExitMakerReplayTuning:
    """Non-semantic memory bounds for one product-day replay.

    The values affect only batching and cache retention.  They must never alter
    the rows returned by :func:`replay_exit_maker_product_day`.
    """

    record_chunk_rows: int = 100_000
    physical_cache_max_entries: int = 128
    outcome_cache_max_entries: int = 2

    def validate(self) -> None:
        for name in (
            "record_chunk_rows",
            "physical_cache_max_entries",
            "outcome_cache_max_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.record_chunk_rows <= 0:
            raise ValueError("record_chunk_rows must be positive")
        if self.physical_cache_max_entries < 0:
            raise ValueError("physical_cache_max_entries must be non-negative")
        if self.outcome_cache_max_entries < 0:
            raise ValueError("outcome_cache_max_entries must be non-negative")


@dataclass(frozen=True)
class _PolicyTrialConsumer:
    """One policy-trial alias consuming a shared physical replay."""

    action: dict[str, object]
    common: dict[str, object]
    exit_rule_id: str
    threshold_basis_bp: float
    source_asof_date: str
    physical_exit_policy_id: str


@dataclass
class _RecordChunkAccumulator:
    """Bound Python records and retain frames or spool typed Parquet chunks."""

    schema: Mapping[str, pl.DataType]
    chunk_rows: int
    spool_directory: Path | None = None
    pending: list[dict[str, object]] = field(default_factory=list)
    chunks: list[pl.DataFrame] = field(default_factory=list)
    chunk_paths: list[Path] = field(default_factory=list)
    row_count: int = 0

    def add(self, record: dict[str, object]) -> None:
        self.pending.append(record)
        self.row_count += 1
        if len(self.pending) >= self.chunk_rows:
            self.flush()

    def extend(self, records: Iterable[dict[str, object]]) -> None:
        # Do not first materialize an arbitrarily large iterable into
        # ``pending``.  Keeping the hard bound here also protects callers which
        # accidentally pass a sequence much larger than ``chunk_rows``.
        for record in records:
            self.add(record)

    def flush(self) -> None:
        if not self.pending:
            return
        frame = _from_records(self.pending, self.schema)
        if self.spool_directory is None:
            self.chunks.append(frame)
        else:
            self.spool_directory.mkdir(parents=True, exist_ok=True)
            path = self.spool_directory / f"part-{len(self.chunk_paths):08d}.parquet"
            frame.write_parquet(path)
            self.chunk_paths.append(path)
        self.pending = []

    def finish(self) -> pl.DataFrame:
        if self.spool_directory is not None:
            raise RuntimeError("spooled accumulator must finish_to_parquet")
        self.flush()
        if not self.chunks:
            return pl.DataFrame(schema=self.schema)
        if len(self.chunks) == 1:
            return self.chunks.pop()
        result = pl.concat(self.chunks, how="vertical", rechunk=False)
        self.chunks = []
        return result

    def finish_to_parquet(
        self,
        destination: Path,
        *,
        sort_by: list[str],
        descending: list[bool] | None = None,
    ) -> None:
        """Stream sorted chunks into one artifact and release the spool."""

        if self.spool_directory is None:
            raise RuntimeError("in-memory accumulator cannot finish to parquet")
        self.flush()
        destination = Path(destination)
        if not self.chunk_paths:
            pl.DataFrame(schema=self.schema).write_parquet(destination)
        else:
            (
                pl.scan_parquet(
                    [str(path) for path in self.chunk_paths],
                    low_memory=True,
                    cache=False,
                )
                .sort(
                    sort_by,
                    descending=(False if descending is None else descending),
                )
                .sink_parquet(destination, engine="streaming")
            )
        for path in self.chunk_paths:
            path.unlink()
        self.chunk_paths = []
        if self.spool_directory.is_dir():
            self.spool_directory.rmdir()


def replay_exit_maker_product_day(
    execution_action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    config: ExitMakerStudyConfig = ExitMakerStudyConfig(),
    *,
    cutoff_cursor: EventCursor | None = None,
    active_exit_policy_trial_ids: Iterable[str] | None = None,
    replay_cache: ExitMakerProductDayReplayCache | None = None,
    replay_tuning: ExitMakerReplayTuning = ExitMakerReplayTuning(),
) -> ExitMakerProductDayResult:
    """Replay both maker-exit routes for one exact product-day.

    ``exit_facts`` is used only as the already-materialized D-1 threshold
    lineage keyed by ``policy_generation_id`` and ``exit_rule_id``.  Its
    taker/taker terminal outcome is not reused by this counterfactual study.
    The function fails closed on missing rules, target-day lineage, identity
    mismatch, or an incomplete SpreadPairTotalCount clock.
    """

    result = _replay_exit_maker_product_day(
        execution_action_facts,
        exit_facts,
        raw_tape,
        spread_pair_clock,
        config,
        cutoff_cursor=cutoff_cursor,
        active_exit_policy_trial_ids=active_exit_policy_trial_ids,
        replay_cache=replay_cache,
        replay_tuning=replay_tuning,
        artifact_directory=None,
    )
    if not isinstance(result, ExitMakerProductDayResult):
        raise AssertionError("in-memory replay returned artifact paths")
    return result


def replay_exit_maker_product_day_to_artifacts(
    execution_action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    config: ExitMakerStudyConfig = ExitMakerStudyConfig(),
    *,
    artifact_directory: Path,
    cutoff_cursor: EventCursor | None = None,
    active_exit_policy_trial_ids: Iterable[str] | None = None,
    replay_cache: ExitMakerProductDayReplayCache | None = None,
    replay_tuning: ExitMakerReplayTuning = ExitMakerReplayTuning(),
) -> ExitMakerProductDayArtifacts:
    """Replay directly into seven Parquet artifacts in an empty stage."""

    destination = Path(artifact_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"exit-maker artifact directory must be empty: {destination}"
        )
    result = _replay_exit_maker_product_day(
        execution_action_facts,
        exit_facts,
        raw_tape,
        spread_pair_clock,
        config,
        cutoff_cursor=cutoff_cursor,
        active_exit_policy_trial_ids=active_exit_policy_trial_ids,
        replay_cache=replay_cache,
        replay_tuning=replay_tuning,
        artifact_directory=destination,
    )
    if not isinstance(result, ExitMakerProductDayArtifacts):
        raise AssertionError("artifact replay returned in-memory frames")
    return result


def _replay_exit_maker_product_day(
    execution_action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    config: ExitMakerStudyConfig,
    *,
    cutoff_cursor: EventCursor | None,
    active_exit_policy_trial_ids: Iterable[str] | None,
    replay_cache: ExitMakerProductDayReplayCache | None,
    replay_tuning: ExitMakerReplayTuning,
    artifact_directory: Path | None,
) -> ExitMakerProductDayResult | ExitMakerProductDayArtifacts:
    """Shared replay implementation for frame and bounded artifact sinks."""

    config.validate()
    replay_tuning.validate()
    requested_policy_ids: frozenset[str] | None = None
    if active_exit_policy_trial_ids is not None:
        requested_values = tuple(
            str(value) for value in active_exit_policy_trial_ids
        )
        if (
            not requested_values
            or len(requested_values) != len(set(requested_values))
            or any(not value for value in requested_values)
        ):
            raise ValueError(
                "active_exit_policy_trial_ids must be nonempty and unique"
            )
        requested_policy_ids = frozenset(requested_values)
    if not isinstance(raw_tape, RawTapeDay):
        raise TypeError("raw_tape must be a RawTapeDay")
    if cutoff_cursor is not None and not isinstance(cutoff_cursor, EventCursor):
        raise TypeError("cutoff_cursor must be an EventCursor or None")
    if execution_action_facts.is_empty():
        if requested_policy_ids is not None:
            raise ValueError(
                "active exit policies require nonempty execution actions"
            )
        return _result_or_artifacts(
            _empty_result(raw_tape, config), artifact_directory
        )

    _require(execution_action_facts, _ACTION_REQUIRED, "execution action facts")
    _require(spread_pair_clock, _CLOCK_REQUIRED, "SpreadPairTotalCount clock")
    identity = _validate_product_day_identity(
        execution_action_facts, exit_facts, raw_tape
    )
    date, value_code, quote_code = identity
    cutoff = cutoff_cursor or session_cutoff_cursor(date)
    if replay_cache is not None:
        if not isinstance(replay_cache, ExitMakerProductDayReplayCache):
            raise TypeError(
                "replay_cache must be an ExitMakerProductDayReplayCache"
            )
        _trim_lru(
            replay_cache.physical,
            replay_tuning.physical_cache_max_entries,
        )
        # Outcome templates are useful only while constructing one day's
        # physical policies.  A retained physical replay already owns its
        # outcomes, so keeping templates across calls duplicates the largest
        # object graph without enabling an additional cache hit.
        replay_cache.outcomes.clear()
        mapping_row = raw_tape.mapping.filter(
            (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
        ).row(0, named=True)
        signature = (
            date,
            value_code,
            quote_code,
            float(mapping_row["spot_ref_price"]),
            float(mapping_row["fut_ref_price"]),
            cutoff.recv_time_ns,
            cutoff.event_sequence,
            cutoff.row_index,
            config.hedge_delay_ns,
            config.max_book_age_ns,
            config.instant_cancel_v0,
            _physical_replay_input_signature(raw_tape, spread_pair_clock),
        )
        if replay_cache.input_signature is None:
            replay_cache.input_signature = signature
        elif replay_cache.input_signature != signature:
            raise ValueError("shared exit-maker replay cache input mismatch")
    if execution_action_facts.select("policy_generation_id").n_unique() != execution_action_facts.height:
        raise ValueError("execution action facts must be unique by policy_generation_id")
    _validate_physical_entry_identity(execution_action_facts)
    established_actions = execution_action_facts.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_status") == "executable")
    )
    if established_actions.is_empty():
        if requested_policy_ids is not None:
            raise ValueError(
                "active exit policies require an established entry"
            )
        return _result_or_artifacts(
            _zero_established_result(
                date,
                value_code,
                quote_code,
                all_entry_aliases=execution_action_facts.height,
                config=config,
            ),
            artifact_directory,
        )
    _require(exit_facts, _EXIT_REQUIRED, "exit facts with D-1 lineage")
    if exit_facts.is_empty():
        raise ValueError("established entry aliases require D-1 exit facts")
    _validate_product_day_identity(established_actions, exit_facts, raw_tape)
    established_policy_ids = established_actions["policy_generation_id"].to_list()
    established_exits = exit_facts.filter(
        pl.col("policy_generation_id").is_in(established_policy_ids)
    )
    rules_by_policy = _validate_exit_rules(
        established_actions,
        established_exits,
        expected=config.expected_exit_rule_ids,
        date=date,
    )
    all_policy_ids = {
        (
            f"{entry_policy_id}/exit/{exit_rule_id}/{exit_route}"
        )
        for entry_policy_id in map(
            str, established_actions["policy_generation_id"].to_list()
        )
        for exit_rule_id in config.expected_exit_rule_ids
        for exit_route in SUPPORTED_EXIT_MAKER_ROUTES
    }
    if requested_policy_ids is not None:
        unknown = sorted(requested_policy_ids - all_policy_ids)
        if unknown:
            raise ValueError(
                "active exit policy ids are absent from established entries: "
                f"{unknown[:5]}"
            )
        materialized_policy_ids = requested_policy_ids
    else:
        materialized_policy_ids = frozenset(all_policy_ids)

    full_timeline = _build_timeline(raw_tape, spread_pair_clock, value_code)
    timeline_index = _index_observation_timeline(full_timeline)
    trade_replays = {
        "spot": _trade_replay(raw_tape.spot_trades, "spot"),
        "future": _trade_replay(raw_tape.future_trades, "future"),
    }
    opposite_snapshot_indexes = {
        market: _index_exit_maker_opposite_snapshots(
            _opposite_snapshots(full_timeline, market)
        )
        for market in ("spot", "future")
    }

    spool_root = (
        None
        if artifact_directory is None
        else artifact_directory / ".chunks"
    )
    support_accumulator = _RecordChunkAccumulator(
        _support_schema(), replay_tuning.record_chunk_rows
    )
    observation_accumulator = _RecordChunkAccumulator(
        _observation_schema(),
        replay_tuning.record_chunk_rows,
        spool_directory=(
            None if spool_root is None else spool_root / "observations"
        ),
    )
    transition_accumulator = _RecordChunkAccumulator(
        _transition_schema(),
        replay_tuning.record_chunk_rows,
        spool_directory=(
            None if spool_root is None else spool_root / "transitions"
        ),
    )
    alias_accumulator = _RecordChunkAccumulator(
        _candidate_alias_schema(),
        replay_tuning.record_chunk_rows,
        spool_directory=(
            None if spool_root is None else spool_root / "candidate_aliases"
        ),
    )
    candidate_accumulator = _RecordChunkAccumulator(
        _candidate_schema(),
        replay_tuning.record_chunk_rows,
        spool_directory=(
            None if spool_root is None else spool_root / "raw_candidate_facts"
        ),
    )
    policy_accumulator = _RecordChunkAccumulator(
        _position_policy_schema(), replay_tuning.record_chunk_rows
    )
    physical_cache = replay_cache.physical if replay_cache is not None else None
    # This cache deduplicates structurally identical center/lower paths inside
    # one call only.  Cross-call reuse comes from the bounded physical cache;
    # retaining both would hold the same outcomes twice.
    outcome_replay_cache: dict[
        tuple[object, ...], _OutcomeReplayTemplate
    ] = {}

    # Build the complete, small policy-trial denominator before replaying raw
    # market paths.  Consumers sharing one physical key are processed together,
    # so the usually enormous replay object can be discarded immediately after
    # all of its aliases have been emitted.
    physical_consumers: dict[
        tuple[object, ...], list[_PolicyTrialConsumer]
    ] = {}
    actions = established_actions.sort("policy_generation_id")
    for action in actions.iter_rows(named=True):
        entry_policy_id = str(action["policy_generation_id"])
        raw_entry_id = str(action["raw_order_fact_id"])
        entry_route = str(action["route"])
        position_status = _position_status(action)
        established_ns = _optional_int(action.get("entry_hedge_decision_time_ns"))
        for exit_rule_id in config.expected_exit_rule_ids:
            rule = rules_by_policy[(entry_policy_id, exit_rule_id)]
            threshold = float(rule["exit_threshold_basis_bp"])
            source_asof = str(rule["exit_rule_source_asof_date"])
            for exit_route in SUPPORTED_EXIT_MAKER_ROUTES:
                exit_policy_trial_id = (
                    f"{entry_policy_id}/exit/{exit_rule_id}/{exit_route}"
                )
                if exit_policy_trial_id not in materialized_policy_ids:
                    continue
                common = {
                    "Date": date,
                    "ValueCode": value_code,
                    "QuoteCode": quote_code,
                    "entry_route": entry_route,
                    "entry_policy_generation_id": entry_policy_id,
                    "entry_raw_order_fact_id": raw_entry_id,
                    "exit_rule_id": exit_rule_id,
                    "exit_route": exit_route,
                    "exit_policy_trial_id": exit_policy_trial_id,
                    "exit_threshold_basis_bp": threshold,
                    "exit_rule_source_asof_date": source_asof,
                    "exit_lifecycle_policy_version": config.exit_lifecycle_policy_version,
                    "exit_queue_scenario": config.exit_queue_scenario,
                    "position_status": position_status,
                    "position_established_ns": established_ns,
                }
                if position_status != "position_established":
                    raise AssertionError("established action filter admitted no position")
                assert established_ns is not None
                physical_key = (
                    raw_entry_id,
                    established_ns,
                    exit_rule_id,
                    exit_route,
                    round(threshold, 12),
                )
                physical_policy_id = _physical_policy_id(physical_key)
                physical_consumers.setdefault(physical_key, []).append(
                    _PolicyTrialConsumer(
                        action=action,
                        common=common,
                        exit_rule_id=exit_rule_id,
                        threshold_basis_bp=threshold,
                        source_asof_date=source_asof,
                        physical_exit_policy_id=physical_policy_id,
                    )
                )

    # Group one raw entry/route contiguously.  All rule variants for a raw
    # candidate can then be collapsed and released before advancing to the
    # next physical position, instead of retaining day-wide variants.
    ordered_physical_items = sorted(
        physical_consumers.items(),
        key=lambda item: (
            str(item[0][0]),
            str(item[1][0].common["exit_route"]),
            int(item[0][1]),
            str(item[0][2]),
            float(item[0][4]),
        ),
    )
    candidate_group: tuple[str, str] | None = None
    candidate_variant_accumulator = _RecordChunkAccumulator(
        _candidate_variant_schema(), replay_tuning.record_chunk_rows
    )
    for physical_key, consumers in ordered_physical_items:
        representative = consumers[0]
        common = representative.common
        action = representative.action
        exit_rule_id = representative.exit_rule_id
        exit_route = str(common["exit_route"])
        raw_entry_id = str(common["entry_raw_order_fact_id"])
        threshold = representative.threshold_basis_bp
        source_asof = representative.source_asof_date
        physical_policy_id = representative.physical_exit_policy_id
        established_ns = _optional_int(common["position_established_ns"])
        assert established_ns is not None
        next_candidate_group = (raw_entry_id, exit_route)
        if candidate_group is not None and next_candidate_group != candidate_group:
            if candidate_variant_accumulator.row_count:
                candidate_accumulator.extend(
                    _collapse_raw_candidate_frame(
                        candidate_variant_accumulator.finish()
                    ).iter_rows(named=True)
                )
                candidate_variant_accumulator = _RecordChunkAccumulator(
                    _candidate_variant_schema(), replay_tuning.record_chunk_rows
                )
        candidate_group = next_candidate_group

        replay = (
            None
            if physical_cache is None
            else _lru_get(physical_cache, physical_key)
        )
        if replay is None:
            replay = _replay_physical_policy(
                timeline_index,
                route=exit_route,
                threshold_basis_bp=threshold,
                start_time_ns=established_ns,
                cutoff_cursor=cutoff,
                physical_policy_id=physical_policy_id,
                maker_replay=trade_replays[
                    EXIT_MAKER_ROUTE_CONTRACTS[exit_route].maker_market
                ],
                opposite_snapshot_index=opposite_snapshot_indexes[
                    EXIT_MAKER_ROUTE_CONTRACTS[exit_route].opposite_market
                ],
                outcome_replay_cache=outcome_replay_cache,
                outcome_cache_max_entries=(
                    replay_tuning.outcome_cache_max_entries
                ),
                config=config,
            )
            if physical_cache is not None:
                _lru_put(
                    physical_cache,
                    physical_key,
                    replay,
                    replay_tuning.physical_cache_max_entries,
                )

        admission = "admitted" if replay.windows else "no_admission"
        peak_active_layers = _peak_active_layers(replay.transitions)
        candidate_ids = _candidate_ids(
            replay,
            entry_raw_order_fact_id=raw_entry_id,
            exit_route=exit_route,
        )
        outcome_by_generation = {
            item.generation_id: item for item in replay.outcomes
        }

        for consumer in consumers:
            consumer_common = consumer.common
            exit_policy_trial_id = str(
                consumer_common["exit_policy_trial_id"]
            )
            support_accumulator.add(
                {
                    **consumer_common,
                    "admission_status": admission,
                    "observation_count": len(replay.observations),
                    "raw_candidate_count": len(replay.windows),
                    "peak_active_layers": peak_active_layers,
                }
            )
            for window in replay.windows:
                outcome = outcome_by_generation[window.generation_id]
                attempt = (
                    outcome.hedge_attempts[0]
                    if outcome.hedge_attempts
                    else None
                )
                alias_accumulator.add(
                    {
                        **consumer_common,
                        "physical_exit_policy_id": physical_policy_id,
                        "exit_raw_candidate_fact_id": candidate_ids[
                            window.generation_id
                        ][0],
                        "exit_policy_candidate_id": window.generation_id,
                        "exit_candidate_alias_id": (
                            f"{exit_policy_trial_id}/"
                            f"{window.generation_id.rsplit('/', 1)[-1]}"
                        ),
                        "spread_pair_epoch": candidate_ids[
                            window.generation_id
                        ][1],
                        "target_price_tick": window.target_price_tick,
                        "submit_recv_time_ns": window.start_cursor.recv_time_ns,
                        "nominal_cancel_request_recv_time_ns": window.stop_cursor.recv_time_ns,
                        "nominal_stop_reason": window.stop_reason,
                        "known_filled_maker_quantity": outcome.known_filled_maker_quantity,
                        "any_fill": outcome.any_fill,
                        "full_fill": outcome.full_fill,
                        "partial_fill": outcome.partial_fill,
                        "independent_branch_status": outcome.branch_status,
                        "cancel_required": outcome.cancel_required,
                        "cancel_ack_observed": outcome.cancel_ack_observed,
                        "cancel_race_modeled": outcome.cancel_race_modeled,
                        "exit_hedge_status": (
                            attempt.status if attempt is not None else None
                        ),
                    }
                )
            policy_accumulator.add(
                _projected_policy_record(
                    consumer_common,
                    consumer.action,
                    replay,
                    physical_policy_id,
                    config,
                )
            )

        observation_accumulator.extend(
            _observation_records(
                replay.observations,
                Date=date,
                ValueCode=value_code,
                QuoteCode=quote_code,
                entry_raw_order_fact_id=raw_entry_id,
                exit_rule_id=exit_rule_id,
                physical_exit_policy_id=physical_policy_id,
            )
        )
        transition_accumulator.extend(
            _transition_records(
                replay.transitions,
                Date=date,
                ValueCode=value_code,
                QuoteCode=quote_code,
                entry_raw_order_fact_id=raw_entry_id,
                exit_rule_id=exit_rule_id,
                exit_route=exit_route,
                physical_exit_policy_id=physical_policy_id,
                candidate_ids=candidate_ids,
            )
        )
        candidate_variant_accumulator.extend(
            _candidate_records(
                replay,
                action,
                date=date,
                value_code=value_code,
                quote_code=quote_code,
                exit_rule_id=exit_rule_id,
                threshold=threshold,
                source_asof=source_asof,
                physical_exit_policy_id=physical_policy_id,
                candidate_ids=candidate_ids,
                config=config,
            )
        )

        # With no explicit cross-call cache, this is now the last strong
        # reference to the physical replay.  The next loop iteration can reuse
        # its memory instead of retaining the entire product-day graph.
        del outcome_by_generation, candidate_ids, replay

    if candidate_variant_accumulator.row_count:
        candidate_accumulator.extend(
            _collapse_raw_candidate_frame(
                candidate_variant_accumulator.finish()
            ).iter_rows(named=True)
        )

    policy_support = support_accumulator.finish().sort(
        ["entry_policy_generation_id", "exit_rule_id", "exit_route"]
    )
    policies = policy_accumulator.finish().sort(
        ["entry_policy_generation_id", "exit_rule_id", "exit_route"]
    )
    expected_trials = len(materialized_policy_ids)
    if policy_support.height != expected_trials or policies.height != expected_trials:
        raise AssertionError("exit maker denominator lost a policy trial")
    artifact_paths = (
        None
        if artifact_directory is None
        else _artifact_path_map(artifact_directory)
    )
    if artifact_paths is None:
        observations = observation_accumulator.finish().sort(
            [
                "physical_exit_policy_id",
                "recv_time_ns",
                "event_sequence",
                "row_index",
            ]
        )
        transitions = transition_accumulator.finish().sort(
            [
                "physical_exit_policy_id",
                "recv_time_ns",
                "event_sequence",
                "row_index",
                "spread_pair_epoch",
                "kind",
                "exit_policy_candidate_id",
                "reason",
                "target_price_tick",
                "active_layers_after",
            ],
            descending=[
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        aliases = alias_accumulator.finish().sort(
            ["exit_policy_trial_id", "exit_raw_candidate_fact_id"]
        )
        candidates = candidate_accumulator.finish().sort(
            [
                "entry_raw_order_fact_id",
                "exit_route",
                "submit_recv_time_ns",
                "spread_pair_epoch",
                "exit_raw_candidate_fact_id",
            ],
            descending=[False, False, False, True, False],
        )
        _validate_candidate_fact_uniqueness(candidates)
        alias_rows = aliases.height
        candidate_rows = candidates.height
    else:
        observation_accumulator.finish_to_parquet(
            artifact_paths["exit_maker_observations"],
            sort_by=[
                "physical_exit_policy_id",
                "recv_time_ns",
                "event_sequence",
                "row_index",
            ],
        )
        transition_accumulator.finish_to_parquet(
            artifact_paths["exit_maker_transitions"],
            sort_by=[
                "physical_exit_policy_id",
                "recv_time_ns",
                "event_sequence",
                "row_index",
                "spread_pair_epoch",
                "kind",
                "exit_policy_candidate_id",
                "reason",
                "target_price_tick",
                "active_layers_after",
            ],
            descending=[
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        alias_accumulator.finish_to_parquet(
            artifact_paths["exit_maker_candidate_aliases"],
            sort_by=["exit_policy_trial_id", "exit_raw_candidate_fact_id"],
        )
        candidate_accumulator.finish_to_parquet(
            artifact_paths["exit_maker_raw_candidate_facts"],
            sort_by=[
                "entry_raw_order_fact_id",
                "exit_route",
                "submit_recv_time_ns",
                "spread_pair_epoch",
                "exit_raw_candidate_fact_id",
            ],
            descending=[False, False, False, True, False],
        )
        _validate_candidate_artifact_uniqueness(
            artifact_paths["exit_maker_raw_candidate_facts"]
        )
        alias_rows = alias_accumulator.row_count
        candidate_rows = candidate_accumulator.row_count
    audit = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "all_entry_aliases": execution_action_facts.height,
                "all_entry_policy_aliases": execution_action_facts.height,
                "entry_policy_aliases": established_actions.height,
                "expected_exit_rules": len(config.expected_exit_rule_ids),
                "expected_exit_routes": len(SUPPORTED_EXIT_MAKER_ROUTES),
                "expected_policy_trials": expected_trials,
                "materialized_policy_trials": policy_support.height,
                "position_established_trials": policy_support.filter(
                    pl.col("position_status") == "position_established"
                ).height,
                "no_admission_trials": policy_support.filter(
                    pl.col("admission_status") == "no_admission"
                ).height,
                "unique_physical_entry_positions": established_actions[
                    "raw_order_fact_id"
                ].n_unique(),
                "physical_exit_policy_replays": len(physical_consumers),
                "raw_candidate_facts": candidate_rows,
                "candidate_alias_rows": alias_rows,
                "oco_winner_trials": policies["oco_winner_generation_id"].is_not_null().sum(),
                "flat_same_day_trials": policies.filter(
                    pl.col("branch_status") == "flat_same_day"
                ).height,
                "cancel_ack_observed_rows": 0,
                "cancel_race_modeled_rows": 0,
                "strict_ev_ready_rows": 0,
                "instant_cancel_v0": True,
                "d_minus_one_lineage_validated": True,
                "joint_volume_allocated": False,
            }
        ],
        infer_schema_length=None,
    )
    if artifact_paths is not None:
        policy_support.write_parquet(
            artifact_paths["exit_maker_policy_support"]
        )
        policies.write_parquet(
            artifact_paths["exit_maker_position_policy_facts"]
        )
        audit.write_parquet(artifact_paths["exit_maker_audit"])
        if spool_root is not None and spool_root.is_dir():
            spool_root.rmdir()
        return ExitMakerProductDayArtifacts(tuple(artifact_paths.items()))
    return ExitMakerProductDayResult(
        policy_support,
        observations,
        transitions,
        aliases,
        candidates,
        policies,
        audit,
    )


def _artifact_path_map(directory: Path) -> dict[str, Path]:
    return {
        name: Path(directory) / f"{name}.parquet"
        for name in _ARTIFACT_FRAME_NAMES
    }


def _validate_candidate_fact_uniqueness(frame: pl.DataFrame) -> None:
    if frame["exit_raw_candidate_fact_id"].n_unique() != frame.height:
        raise AssertionError("canonical exit raw candidate identity collided")


def _validate_candidate_artifact_uniqueness(path: Path) -> None:
    counts = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.col("exit_raw_candidate_fact_id").n_unique().alias("unique_ids"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if int(counts["rows"]) != int(counts["unique_ids"]):
        raise AssertionError("canonical exit raw candidate identity collided")


def _result_or_artifacts(
    result: ExitMakerProductDayResult,
    artifact_directory: Path | None,
) -> ExitMakerProductDayResult | ExitMakerProductDayArtifacts:
    if artifact_directory is None:
        return result
    paths = _artifact_path_map(artifact_directory)
    frames = result.frames()
    if tuple(frames) != _ARTIFACT_FRAME_NAMES:
        raise AssertionError("exit-maker artifact frame ordering changed")
    for name, frame in frames.items():
        frame.write_parquet(paths[name])
    return ExitMakerProductDayArtifacts(tuple(paths.items()))


def _validate_product_day_identity(
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    tape: RawTapeDay,
) -> tuple[str, str, str]:
    identities = actions.select("Date", "ValueCode", "QuoteCode").unique()
    if identities.height != 1:
        raise ValueError("exit maker study requires one exact product-day slice")
    date, value_code, quote_code = map(str, identities.row(0))
    if str(tape.date) != date:
        raise ValueError("raw tape Date does not match action facts")
    mapping = tape.mapping.filter(
        (pl.col("ValueCode").cast(pl.String) == value_code)
        & (pl.col("QuoteCode").cast(pl.String) == quote_code)
    )
    if mapping.height != 1:
        raise ValueError("raw tape mapping does not match action identity")
    if not exits.is_empty():
        exit_identity = exits.select("Date", "ValueCode", "QuoteCode").unique()
        if exit_identity.height != 1 or tuple(map(str, exit_identity.row(0))) != (
            date,
            value_code,
            quote_code,
        ):
            raise ValueError("exit facts do not match action product-day identity")
    return date, value_code, quote_code


def _validate_physical_entry_identity(actions: pl.DataFrame) -> None:
    # ``full_fill`` is intentionally policy-specific: aliases at the same
    # absolute entry price can have different nominal stop cursors.  A print
    # after the shorter policy's stop may therefore fill one alias but not the
    # other.  Only aliases which actually establish a hedged position must
    # agree on the resulting physical prices/cursor.
    base_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
    ]
    inconsistent = actions.group_by("raw_order_fact_id").agg(
        *(pl.col(column).n_unique().alias(column) for column in base_columns)
    ).filter(pl.any_horizontal(*(pl.col(column) != 1 for column in base_columns)))
    if inconsistent.height:
        raise ValueError("entry aliases disagree on physical order identity")

    opened = actions.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_status") == "executable")
    )
    if opened.is_empty():
        return
    position_columns = [
        "entry_hedge_decision_time_ns",
        "entry_future_price",
        "entry_spot_price",
        "entry_hedge_contract_size_shares",
    ]
    inconsistent = opened.group_by("raw_order_fact_id").agg(
        *(pl.col(column).n_unique().alias(column) for column in position_columns)
    ).filter(
        pl.any_horizontal(*(pl.col(column) != 1 for column in position_columns))
    )
    if inconsistent.height:
        raise ValueError("entry aliases disagree on physical position identity")


def _validate_exit_rules(
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    *,
    expected: tuple[str, ...],
    date: str,
) -> dict[tuple[str, str], dict[str, object]]:
    key = ["policy_generation_id", "exit_rule_id"]
    if exits.select(key).n_unique() != exits.height:
        raise ValueError("exit facts contain duplicate policy/rule keys")
    action_ids = set(map(str, actions["policy_generation_id"].to_list()))
    exit_ids = set(map(str, exits["policy_generation_id"].to_list()))
    if action_ids != exit_ids:
        raise ValueError("exit facts must cover exactly every entry policy alias")
    if "exit_rule_contains_target_day_outcome" in exits.columns and exits.filter(
        pl.col("exit_rule_contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("exit rule lineage contains target-day outcomes")
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    expected_set = set(expected)
    for (policy_id,), group in exits.group_by("policy_generation_id", maintain_order=True):
        observed = set(map(str, group["exit_rule_id"].to_list()))
        if observed != expected_set or group.height != len(expected):
            raise ValueError(
                f"{policy_id}: exit facts must contain exactly {list(expected)}"
            )
        for row in group.iter_rows(named=True):
            threshold = row.get("exit_threshold_basis_bp")
            source = str(row.get("exit_rule_source_asof_date"))
            if not _finite(threshold):
                raise ValueError("exit thresholds must be finite")
            if len(source) != 8 or not source.isdigit() or source >= date:
                raise ValueError(
                    "exit_rule_source_asof_date must be strictly before Date"
                )
            policy = str(row["policy_generation_id"])
            rule = str(row["exit_rule_id"])
            lookup[(policy, rule)] = row
    return lookup


def _position_status(action: dict[str, object]) -> str:
    if action.get("full_fill") is not True:
        return "entry_not_full_fill"
    if action.get("entry_hedge_status") != "executable":
        return "entry_hedge_not_executable"
    required = (
        action.get("entry_hedge_decision_time_ns"),
        action.get("entry_future_price"),
        action.get("entry_spot_price"),
        action.get("entry_hedge_contract_size_shares"),
    )
    if not all(_finite(value) and float(value) > 0 for value in required):
        raise ValueError("an executable entry hedge has incomplete physical prices")
    if not math.isclose(
        float(action["entry_hedge_contract_size_shares"]),
        CONTRACT_SHARES,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"exit V0 requires {CONTRACT_SHARES} shares per futures contract"
        )
    return "position_established"


def _build_timeline(
    tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
    value_code: str,
) -> tuple[_CombinedPoint, ...]:
    clock = spread_pair_clock.filter(
        (pl.col("Date").cast(pl.String) == str(tape.date))
        & (pl.col("ValueCode").cast(pl.String) == value_code)
    )
    spot = tape.spot_states.filter(
        pl.col("ValueCode").cast(pl.String) == value_code
    )
    future = tape.future_states.filter(
        pl.col("ValueCode").cast(pl.String) == value_code
    )
    if spot.is_empty() or future.is_empty():
        raise ValueError("raw tape is missing spot or future states")
    if clock.select("spot_channel_seq").n_unique() != clock.height:
        raise ValueError("SpreadPairTotalCount clock contains duplicate spot keys")
    clock_by_sequence = {
        int(row["spot_channel_seq"]): row
        for row in clock.iter_rows(named=True)
    }
    spot_sequences = set(map(int, spot["sequence"].to_list()))
    if spot_sequences != set(clock_by_sequence):
        raise ValueError("SpreadPairTotalCount clock must cover every raw spot state")
    spot_points = _market_points(
        spot,
        "spot",
        clock_by_sequence=clock_by_sequence,
    )
    future_points = _market_points(future, "future", clock_by_sequence=None)
    events = sorted((*spot_points, *future_points), key=lambda item: item.cursor)
    for previous, current in zip(events, events[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("raw market event cursors must be strictly ordered")
    current_spot: _MarketPoint | None = None
    current_future: _MarketPoint | None = None
    combined: list[_CombinedPoint] = []
    for event in events:
        if event.market == "spot":
            current_spot = event
        else:
            current_future = event
        combined.append(_CombinedPoint(event.cursor, current_spot, current_future))
    return tuple(combined)


def _thin_target_timeline(
    timeline: tuple[_CombinedPoint, ...],
) -> tuple[_CombinedPoint, ...]:
    """Keep only events capable of changing admission or a legal target.

    Raw rows are still consumed in exact merged cursor order.  Trade-only and
    queue-lots-only rows with identical executable BBO/VWAP/gate/epoch state
    cannot submit or cancel a layer, so retaining millions of copies would
    change only artifact size, not the replay.  The current full raw timeline
    remains available separately for maker prints and 50 ms hedge snapshots.
    """

    thinned: list[_CombinedPoint] = []
    previous: tuple[object, ...] | None = None
    for point in timeline:
        signature = _target_state_signature(point)
        if signature != previous:
            thinned.append(point)
            previous = signature
    return tuple(thinned)


def _index_observation_timeline(
    full_timeline: tuple[_CombinedPoint, ...],
) -> _ObservationTimelineIndex:
    """Precompute cursor arrays once instead of once per physical policy."""

    target_timeline = _thin_target_timeline(full_timeline)
    return _ObservationTimelineIndex(
        full_timeline=full_timeline,
        target_timeline=target_timeline,
        full_cursors=tuple(point.cursor for point in full_timeline),
        target_cursors=tuple(point.cursor for point in target_timeline),
    )


def _target_state_signature(point: _CombinedPoint) -> tuple[object, ...]:
    def market_state(
        current: _MarketPoint | None,
        market: Literal["spot", "future"],
    ) -> tuple[object, ...]:
        if current is None:
            return (None,)
        row = current.row
        required_side = "bid" if market == "spot" else "ask"
        required_quantity = (
            SPOT_HEDGE_LOTS if market == "spot" else FUTURE_HEDGE_CONTRACTS
        )
        vwap = _required_vwap(
            executable_levels_from_state(row, market, required_side),
            required_quantity,
        )
        return (
            bool(row.get("trial_match", False)),
            current.formal_after_trial,
            _positive_float(row.get("ref_price")),
            _positive_float(row.get("exec_bid_price")),
            _positive_float(row.get("exec_ask_price")),
            vwap,
            _raw_ref_gate(row),
        )

    spot_clock = (
        None,
        False,
    ) if point.spot is None else (
        point.spot.spread_pair_epoch,
        point.spot.spread_pair_active,
    )
    return (
        *spot_clock,
        *market_state(point.spot, "spot"),
        *market_state(point.future, "future"),
    )


def _market_points(
    frame: pl.DataFrame,
    market: Literal["spot", "future"],
    *,
    clock_by_sequence: Mapping[int, dict[str, object]] | None,
) -> tuple[_MarketPoint, ...]:
    _require(
        frame,
        {
            "recv_time_ns",
            "sequence",
            "packet_sequence",
            "trial_match",
            "raw_has_book",
            "book_state_available",
            "ref_price",
            "exec_bid_price",
            "exec_ask_price",
        },
        f"raw {market} states",
    )
    formal = False
    previous_trial: bool | None = None
    points: list[_MarketPoint] = []
    previous_epoch: int | None = None
    ordered = frame.sort(["recv_time_ns", "sequence", "packet_sequence"])
    for row in ordered.iter_rows(named=True):
        trial = bool(row.get("trial_match", False))
        raw_has_book = bool(row.get("raw_has_book", False))
        if trial:
            formal = False
        elif previous_trial is True:
            formal = raw_has_book
        elif not formal and raw_has_book:
            formal = True
        previous_trial = trial
        epoch: int | None = None
        active = False
        if market == "spot":
            assert clock_by_sequence is not None
            clock = clock_by_sequence[int(row["sequence"])]
            epoch_value = clock.get("spread_pair_epoch")
            if isinstance(epoch_value, bool) or not isinstance(epoch_value, int):
                raise ValueError("spread_pair_epoch must be an integer")
            epoch = int(epoch_value)
            if epoch < 0 or (
                previous_epoch is not None and epoch < previous_epoch
            ):
                raise ValueError("SpreadPairTotalCount must be non-negative and monotonic")
            previous_epoch = epoch
            pair_id = clock.get("spread_pair_id")
            active = (int(pair_id) > 0) if _finite(pair_id) else epoch > 0
        points.append(
            _MarketPoint(
                EventCursor(
                    int(row["recv_time_ns"]),
                    _MARKET_PRIORITY[market],
                    int(row["sequence"]),
                ),
                market,
                row,
                formal,
                epoch,
                active,
            )
        )
    return tuple(points)


def _replay_physical_policy(
    timeline_index: _ObservationTimelineIndex,
    *,
    route: str,
    threshold_basis_bp: float,
    start_time_ns: int,
    cutoff_cursor: EventCursor,
    physical_policy_id: str,
    maker_replay: IndexedTradeReplay,
    opposite_snapshot_index: _IndexedOppositeBookSnapshots,
    outcome_replay_cache: dict[
        tuple[object, ...], _OutcomeReplayTemplate
    ],
    config: ExitMakerStudyConfig,
    outcome_cache_max_entries: int = 128,
) -> _PhysicalReplay:
    observations = _build_observations(
        timeline_index,
        route=route,
        threshold_basis_bp=threshold_basis_bp,
        start_time_ns=start_time_ns,
        cutoff_cursor=cutoff_cursor,
    )
    built = build_exit_maker_order_windows(
        observations,
        route=route,
        policy_id=physical_policy_id,
        cutoff_cursor=cutoff_cursor,
    )
    replay_signature = _window_replay_signature(route, built.windows)
    template = _lru_get(outcome_replay_cache, replay_signature)
    if template is None:
        outcomes = _replay_exit_maker_windows_indexed(
            built.windows,
            route=route,
            maker_replay=maker_replay,
            opposite_snapshot_index=opposite_snapshot_index,
            eod_cursor=cutoff_cursor,
            hedge_delay_ns=config.hedge_delay_ns,
            max_book_age_ns=config.max_book_age_ns,
        )
        _lru_put(
            outcome_replay_cache,
            replay_signature,
            _OutcomeReplayTemplate(built.windows, outcomes),
            outcome_cache_max_entries,
        )
    else:
        outcomes = _rebind_exit_maker_outcomes(
            template.windows,
            template.outcomes,
            built.windows,
        )
    projection = project_earliest_full_fill_oco(zip(built.windows, outcomes))
    return _PhysicalReplay(
        observations,
        built.windows,
        built.transitions,
        outcomes,
        projection,
    )


def _window_replay_signature(
    route: str,
    windows: tuple[IndependentOrderWindow, ...],
) -> tuple[object, ...]:
    """Return the complete ID-free input to maker-fill and hedge replay."""

    return (
        route,
        tuple(
            (
                window.maker_side,
                window.target_price_tick,
                window.start_cursor,
                window.stop_cursor,
                window.initial_queue_ahead,
                window.stop_reason,
            )
            for window in windows
        ),
    )


def _rebind_exit_maker_outcomes(
    template_windows: tuple[IndependentOrderWindow, ...],
    template_outcomes: tuple[ExitMakerOrderOutcome, ...],
    windows: tuple[IndependentOrderWindow, ...],
) -> tuple[ExitMakerOrderOutcome, ...]:
    """Copy an identical structural replay while replacing every policy ID."""

    if len(template_windows) != len(template_outcomes) or len(windows) != len(
        template_windows
    ):
        raise AssertionError("cached exit-maker replay has inconsistent lengths")
    rebound: list[ExitMakerOrderOutcome] = []
    for template_window, template_outcome, window in zip(
        template_windows,
        template_outcomes,
        windows,
    ):
        if _window_replay_signature("template", (template_window,)) != (
            _window_replay_signature("template", (window,))
        ):
            raise AssertionError("cached exit-maker window structure changed")
        if template_outcome.generation_id != template_window.generation_id:
            raise AssertionError("cached exit-maker window/outcome IDs diverged")
        quantity_fill = replace(
            template_outcome.quantity_fill,
            generation_id=window.generation_id,
        )
        attempts = []
        for attempt in template_outcome.hedge_attempts:
            expected_attempt_id = (
                f"{template_window.generation_id}/exit-hedge-{attempt.unit_start}"
            )
            if attempt.generation_id != expected_attempt_id:
                raise AssertionError("cached exit hedge attempt ID is malformed")
            attempt_id = f"{window.generation_id}/exit-hedge-{attempt.unit_start}"
            execution = attempt.execution
            if execution is not None:
                if execution.generation_id != attempt.generation_id:
                    raise AssertionError("cached exit hedge execution ID diverged")
                execution = replace(execution, generation_id=attempt_id)
            attempts.append(
                replace(
                    attempt,
                    generation_id=attempt_id,
                    execution=execution,
                )
            )
        rebound.append(
            replace(
                template_outcome,
                generation_id=window.generation_id,
                quantity_fill=quantity_fill,
                hedge_attempts=tuple(attempts),
            )
        )
    return tuple(rebound)


def _build_observations(
    timeline_index: _ObservationTimelineIndex,
    *,
    route: str,
    threshold_basis_bp: float,
    start_time_ns: int,
    cutoff_cursor: EventCursor,
) -> tuple[ExitMakerObservation, ...]:
    start_cursor = EventCursor(start_time_ns, 3, 0)
    if start_cursor >= cutoff_cursor or not timeline_index.full_timeline:
        return ()
    seed_index = bisect_right(timeline_index.full_cursors, start_cursor) - 1
    target_start = bisect_right(timeline_index.target_cursors, start_cursor)
    target_stop = bisect_left(timeline_index.target_cursors, cutoff_cursor)

    def relevant_points() -> Iterable[tuple[EventCursor, _CombinedPoint]]:
        if seed_index >= 0:
            yield start_cursor, timeline_index.full_timeline[seed_index]
        for index in range(target_start, target_stop):
            point = timeline_index.target_timeline[index]
            yield point.cursor, point

    observations: list[ExitMakerObservation] = []
    last: ExitMakerObservation | None = None
    last_signature: tuple[object, ...] | None = None
    for cursor, point in relevant_points():
        observation, failure = _observation_at_point(
            point,
            cursor=cursor,
            route=route,
            threshold_basis_bp=threshold_basis_bp,
        )
        if observation is None:
            if last is not None:
                epoch = (
                    point.spot.spread_pair_epoch
                    if point.spot is not None
                    and point.spot.spread_pair_epoch is not None
                    else last.spread_pair_epoch
                )
                observation = replace(
                    last,
                    cursor=cursor,
                    spread_pair_epoch=epoch,
                    gate_open=False,
                    gate_reason=failure or "raw_state_unavailable",
                    initial_queue_ahead=None,
                    target_rank="unavailable",
                )
            else:
                continue
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


def _observation_at_point(
    point: _CombinedPoint,
    *,
    cursor: EventCursor,
    route: str,
    threshold_basis_bp: float,
) -> tuple[ExitMakerObservation | None, str | None]:
    if point.spot is None:
        return None, "missing_spot_state"
    if point.future is None:
        return None, "missing_future_state"
    spot = point.spot.row
    future = point.future.row
    session_date = (
        str(spot["Date"])
        if spot.get("Date") is not None
        else (
            str(future["Date"])
            if future.get("Date") is not None
            else None
        )
    )
    if (
        spot.get("Date") is not None
        and future.get("Date") is not None
        and str(spot["Date"]) != str(future["Date"])
    ):
        return None, "spot_future_session_date_mismatch"
    epoch = point.spot.spread_pair_epoch
    if epoch is None:
        return None, "missing_spread_pair_epoch"
    spot_bid = _positive_float(spot.get("exec_bid_price"))
    spot_ask = _positive_float(spot.get("exec_ask_price"))
    future_bid = _positive_float(future.get("exec_bid_price"))
    future_ask = _positive_float(future.get("exec_ask_price"))
    if None in (spot_bid, spot_ask, future_bid, future_ask):
        return None, "invalid_executable_bbo"
    assert spot_bid is not None and spot_ask is not None
    assert future_bid is not None and future_ask is not None
    if spot_bid >= spot_ask or future_bid >= future_ask:
        return None, "crossed_or_locked_executable_bbo"
    spot_vwap = _required_vwap(
        executable_levels_from_state(spot, "spot", "bid"), SPOT_HEDGE_LOTS
    )
    future_vwap = _required_vwap(
        executable_levels_from_state(future, "future", "ask"),
        FUTURE_HEDGE_CONTRACTS,
    )
    gates = [
        (point.spot.spread_pair_active, "spread_pair_clock_inactive"),
        (not bool(spot.get("trial_match", False)), "spot_trial_match"),
        (point.spot.formal_after_trial, "spot_formal_book_gate"),
        (not bool(future.get("trial_match", False)), "future_trial_match"),
        (point.future.formal_after_trial, "future_formal_book_gate"),
        (_raw_ref_gate(spot), "spot_ref_gate"),
        (_raw_ref_gate(future), "future_ref_gate"),
        (spot_vwap is not None, "spot_insufficient_bid_depth"),
        (future_vwap is not None, "future_insufficient_ask_depth"),
    ]
    gate_open = all(valid for valid, _ in gates)
    reason = "open" if gate_open else next(
        reason for valid, reason in gates if not valid
    )
    # Closed observations still need finite prices so the pure target helper
    # can carry a legal absolute target into the cancellation transition.
    spot_exec = spot_vwap if spot_vwap is not None else spot_bid
    future_exec = future_vwap if future_vwap is not None else future_ask
    maker = future if route == FUTURE_BID_EXIT_ROUTE else spot
    reference = _positive_float(maker.get("ref_price"))
    if reference is None:
        return None, "missing_maker_ref_price"
    observation = make_exit_maker_observation(
        route,
        cursor,
        epoch,
        threshold_basis_bp,
        session_date=session_date,
        spot_bid=spot_bid,
        spot_ask=spot_ask,
        future_bid=future_bid,
        future_ask=future_ask,
        spot_sell_exec_price=spot_exec,
        future_buy_exec_price=future_exec,
        maker_reference_price=reference,
        initial_queue_ahead=None,
        raw_gate_open=gate_open,
        raw_gate_reason=reason,
    )
    maker_side = EXIT_MAKER_ROUTE_CONTRACTS[route].maker_side
    rank, queue = _rank_and_queue(observation.target_price, maker_side, maker)
    return replace(observation, initial_queue_ahead=queue, target_rank=rank), None


def _trade_replay(trades: pl.DataFrame, market: Literal["spot", "future"]) -> IndexedTradeReplay:
    selected = trades.sort(["recv_time_ns", "sequence", "packet_sequence"])
    events = tuple(
        TradeEvent(
            EventCursor(
                int(row["recv_time_ns"]),
                _MARKET_PRIORITY[market],
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
        for row in selected.iter_rows(named=True)
    )
    return IndexedTradeReplay(events)


def _opposite_snapshots(
    timeline: tuple[_CombinedPoint, ...],
    market: Literal["spot", "future"],
) -> tuple[OppositeBookSnapshot, ...]:
    snapshots: list[OppositeBookSnapshot] = []
    last_formal: bool | None = None
    for point in timeline:
        current = point.spot if market == "spot" else point.future
        if current is None or current.cursor != point.cursor:
            continue
        row = current.row
        state_transition = last_formal is None or current.formal_after_trial != last_formal
        last_formal = current.formal_after_trial
        if not (
            bool(row.get("raw_has_book", False))
            or bool(row.get("trial_match", False))
            or state_transition
        ):
            continue
        bids = executable_levels_from_state(row, market, "bid")
        asks = executable_levels_from_state(row, market, "ask")
        gate_open = (
            current.formal_after_trial
            and not bool(row.get("trial_match", False))
            and _raw_ref_gate(row)
            and bool(bids)
            and bool(asks)
            and bids[0].price < asks[0].price
        )
        if bool(row.get("trial_match", False)):
            reason = "trial_match"
        elif not current.formal_after_trial:
            reason = "formal_book_gate"
        elif not _raw_ref_gate(row):
            reason = "ref_price_gate"
        elif not bids or not asks or bids[0].price >= asks[0].price:
            reason = "invalid_executable_book"
        else:
            reason = None
        snapshots.append(
            OppositeBookSnapshot(
                current.cursor,
                bids,
                asks,
                trial_match=bool(row.get("trial_match", False)),
                gate_open=gate_open,
                gate_reason=reason,
            )
        )
    return tuple(snapshots)


def _candidate_records(
    replay: _PhysicalReplay,
    action: dict[str, object],
    *,
    date: str,
    value_code: str,
    quote_code: str,
    exit_rule_id: str,
    threshold: float,
    source_asof: str,
    physical_exit_policy_id: str,
    candidate_ids: Mapping[str, tuple[str, int]],
    config: ExitMakerStudyConfig,
) -> Iterable[dict[str, object]]:
    windows = {window.generation_id: window for window in replay.windows}
    members = {member.generation_id: member for member in replay.projection.members}
    for outcome in replay.outcomes:
        window = windows[outcome.generation_id]
        member = members[outcome.generation_id]
        raw_candidate_id, spread_pair_epoch = candidate_ids[outcome.generation_id]
        attempt = outcome.hedge_attempts[0] if outcome.hedge_attempts else None
        execution = attempt.execution if attempt is not None else None
        exit_prices = _exit_prices(outcome, window, session_date=date)
        gross = _gross_cycle_pnl(action, *exit_prices)
        fill = outcome.quantity_fill
        yield {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "entry_raw_order_fact_id": str(action["raw_order_fact_id"]),
                "exit_rule_id": exit_rule_id,
                "exit_route": outcome.route,
                "physical_exit_policy_id": physical_exit_policy_id,
                "exit_raw_candidate_fact_id": raw_candidate_id,
                "exit_policy_candidate_id": outcome.generation_id,
                "spread_pair_epoch": spread_pair_epoch,
                "exit_threshold_basis_bp": threshold,
                "exit_rule_source_asof_date": source_asof,
                "target_price_tick": window.target_price_tick,
                "target_price": tick_index_to_price(
                    window.target_price_tick,
                    market=EXIT_MAKER_ROUTE_CONTRACTS[
                        outcome.route
                    ].maker_market,
                    session_date=date,
                ),
                "intended_maker_quantity": outcome.intended_maker_quantity,
                "submit_recv_time_ns": window.start_cursor.recv_time_ns,
                "submit_event_sequence": window.start_cursor.event_sequence,
                "submit_row_index": window.start_cursor.row_index,
                "nominal_cancel_request_recv_time_ns": window.stop_cursor.recv_time_ns,
                "nominal_cancel_request_event_sequence": window.stop_cursor.event_sequence,
                "nominal_cancel_request_row_index": window.stop_cursor.row_index,
                "nominal_stop_reason": window.stop_reason,
                "initial_queue_ahead": window.initial_queue_ahead,
                "queue_known": fill.queue_known,
                "known_filled_maker_quantity": outcome.known_filled_maker_quantity,
                "remaining_maker_quantity": outcome.remaining_maker_quantity,
                "any_fill": fill.any_fill,
                "full_fill": fill.full_fill,
                "partial_fill": fill.partial_fill,
                "first_fill_recv_time_ns": _cursor_field(fill.first_fill_cursor, "recv_time_ns"),
                "full_fill_recv_time_ns": _cursor_field(fill.full_fill_cursor, "recv_time_ns"),
                "trade_through_fill": fill.trade_through_fill,
                "independent_branch_status": outcome.branch_status,
                "cancel_required": outcome.cancel_required,
                "instant_cancel_v0": config.instant_cancel_v0,
                "cancel_ack_observed": outcome.cancel_ack_observed,
                "cancel_race_modeled": outcome.cancel_race_modeled,
                "oco_disposition": member.disposition,
                "oco_cancel_request_recv_time_ns": _cursor_field(
                    member.oco_cancel_request_cursor, "recv_time_ns"
                ),
                "independent_full_fill_after_oco_cancel_request": member.independent_full_fill_after_cancel_request,
                "partial_fill_before_oco_winner": member.partial_fill_before_winner,
                "selected_for_position": member.selected_for_position,
                "exit_maker_price": exit_prices[1] if outcome.route == FUTURE_BID_EXIT_ROUTE else exit_prices[0],
                "exit_hedge_status": attempt.status if attempt is not None else None,
                "exit_hedge_decision_time_ns": attempt.decision_time_ns if attempt is not None else None,
                "exit_hedge_requested_quantity": attempt.hedge_quantity if attempt is not None else None,
                "exit_hedge_executed_quantity": attempt.executed_hedge_quantity if attempt is not None else None,
                "exit_hedge_price": (
                    execution.executable_vwap_price if execution is not None else None
                ),
                "exit_hedge_signed_latency_slippage_bp": (
                    execution.signed_latency_slippage_bp if execution is not None else None
                ),
                "exit_hedge_signed_depth_slippage_bp": (
                    execution.signed_depth_slippage_bp if execution is not None else None
                ),
                "exit_hedge_signed_total_slippage_bp": (
                    execution.signed_total_slippage_bp if execution is not None else None
                ),
                "exit_spot_price": exit_prices[0],
                "exit_future_price": exit_prices[1],
                "exit_basis_bp": _basis(*exit_prices),
                "gross_cycle_pnl_twd": gross,
                "residual_spot_long_lots": (
                    outcome.residual_position.spot_long_lots
                    if outcome.residual_position is not None else None
                ),
                "residual_future_short_contracts": (
                    outcome.residual_position.future_short_contracts
                    if outcome.residual_position is not None else None
                ),
                "joint_volume_allocated": outcome.joint_volume_allocated,
                "strict_ev_ready": outcome.pathwise_ev_ready,
            }


def _projected_policy_record(
    common: dict[str, object],
    action: dict[str, object],
    replay: _PhysicalReplay,
    physical_exit_policy_id: str,
    config: ExitMakerStudyConfig,
) -> dict[str, object]:
    projection = replay.projection
    outcome_by_id = {item.generation_id: item for item in replay.outcomes}
    window_by_id = {item.generation_id: item for item in replay.windows}
    winner = (
        outcome_by_id.get(projection.winner_generation_id)
        if projection.winner_generation_id is not None
        else None
    )
    window = (
        window_by_id.get(projection.winner_generation_id)
        if projection.winner_generation_id is not None
        else None
    )
    candidate_ids = _candidate_ids(
        replay,
        entry_raw_order_fact_id=str(action["raw_order_fact_id"]),
        exit_route=str(common["exit_route"]),
    )
    winner_identity = (
        candidate_ids.get(projection.winner_generation_id)
        if projection.winner_generation_id is not None
        else None
    )
    attempt = winner.hedge_attempts[0] if winner is not None and winner.hedge_attempts else None
    execution = attempt.execution if attempt is not None else None
    if winner is not None:
        nominal_branch = winner.branch_status
    elif not replay.windows:
        nominal_branch = "carry_at_eod_no_admission"
    else:
        nominal_branch = _no_winner_branch(replay.outcomes)
    active_sibling_cancel_count = sum(
        member.disposition == "sibling_cancel_required"
        for member in projection.members
    )
    prior_unacked_cancel_count = 0
    if projection.winner_full_fill_cursor is not None:
        winner_cursor = projection.winner_full_fill_cursor
        for candidate_window in replay.windows:
            if candidate_window.generation_id == projection.winner_generation_id:
                continue
            candidate_outcome = outcome_by_id[candidate_window.generation_id]
            if (
                candidate_window.start_cursor < winner_cursor
                and candidate_window.stop_cursor <= winner_cursor
                and not candidate_outcome.cancel_ack_observed
            ):
                prior_unacked_cancel_count += 1
    nominal_flat = nominal_branch == "flat_same_day"
    cancel_ambiguous = nominal_flat and (
        not projection.position_projection_safe
        or active_sibling_cancel_count > 0
        or prior_unacked_cancel_count > 0
    )
    if cancel_ambiguous:
        branch = "cancel_race_unknown"
    else:
        branch = nominal_branch
    exit_prices = (
        _exit_prices(
            winner,
            window,
            session_date=str(common["Date"]),
        )
        if winner is not None and window is not None
        else (None, None)
    )
    gross = _gross_cycle_pnl(action, *exit_prices)
    contract = EXIT_MAKER_ROUTE_CONTRACTS[str(common["exit_route"])]
    flat = branch == "flat_same_day"
    return {
        **common,
        "physical_exit_policy_id": physical_exit_policy_id,
        "exit_style": "maker_taker",
        "exit_maker_quantity": (
            1 if contract.maker_market == "future" else SPOT_HEDGE_LOTS
        ),
        "exit_hedge_quantity": contract.opposite_units_per_hedge,
        "raw_candidate_count": len(replay.windows),
        "oco_winner_generation_id": projection.winner_generation_id,
        "oco_winner_raw_candidate_fact_id": (
            winner_identity[0] if winner_identity is not None else None
        ),
        "oco_winner_spread_pair_epoch": (
            winner_identity[1] if winner_identity is not None else None
        ),
        "oco_winner_target_price_tick": (
            window.target_price_tick if window is not None else None
        ),
        "oco_winner_submit_recv_time_ns": (
            window.start_cursor.recv_time_ns if window is not None else None
        ),
        "oco_winner_submit_event_sequence": (
            window.start_cursor.event_sequence if window is not None else None
        ),
        "oco_winner_submit_row_index": (
            window.start_cursor.row_index if window is not None else None
        ),
        "oco_winner_raw_identity_excludes_rule": winner_identity is not None,
        "oco_winner_fill_recv_time_ns": _cursor_field(
            projection.winner_full_fill_cursor, "recv_time_ns"
        ),
        "oco_same_cursor_winner_count": len(projection.same_cursor_winner_candidates),
        "oco_active_sibling_cancel_count": active_sibling_cancel_count,
        "prior_unacked_cancel_count_before_winner": prior_unacked_cancel_count,
        "oco_position_projection_safe": projection.position_projection_safe,
        "nominal_instant_cancel_v0_branch": nominal_branch,
        "branch_status": branch,
        "terminal_outcome": flat and projection.position_projection_safe,
        "needs_next_session_label": not flat,
        "exit_decision_time_ns": attempt.decision_time_ns if attempt is not None else None,
        "exit_maker_price": (
            exit_prices[1]
            if common["exit_route"] == FUTURE_BID_EXIT_ROUTE
            else exit_prices[0]
        ),
        "exit_hedge_status": attempt.status if attempt is not None else None,
        "exit_hedge_price": execution.executable_vwap_price if execution is not None else None,
        "exit_hedge_signed_latency_slippage_bp": (
            execution.signed_latency_slippage_bp if execution is not None else None
        ),
        "exit_hedge_signed_depth_slippage_bp": (
            execution.signed_depth_slippage_bp if execution is not None else None
        ),
        "exit_hedge_signed_total_slippage_bp": (
            execution.signed_total_slippage_bp if execution is not None else None
        ),
        "exit_spot_price": exit_prices[0],
        "exit_future_price": exit_prices[1],
        "exit_basis_bp": _basis(*exit_prices),
        "gross_cycle_pnl_twd": gross,
        "instant_cancel_v0": config.instant_cancel_v0,
        "cancel_ack_observed": projection.cancel_ack_observed,
        "cancel_race_modeled": projection.cancel_race_modeled,
        "joint_volume_allocated": False,
        "strict_ev_ready": projection.strict_ev_ready,
    }


def _unopened_position_policy(
    common: dict[str, object],
    action: dict[str, object],
    config: ExitMakerStudyConfig,
) -> dict[str, object]:
    contract = EXIT_MAKER_ROUTE_CONTRACTS[str(common["exit_route"])]
    return {
        **common,
        "physical_exit_policy_id": None,
        "exit_style": "maker_taker",
        "exit_maker_quantity": 1 if contract.maker_market == "future" else 2,
        "exit_hedge_quantity": contract.opposite_units_per_hedge,
        "raw_candidate_count": 0,
        "oco_winner_generation_id": None,
        "oco_winner_fill_recv_time_ns": None,
        "oco_same_cursor_winner_count": 0,
        "oco_position_projection_safe": False,
        "branch_status": "not_opened_or_unhedged",
        "terminal_outcome": True,
        "needs_next_session_label": False,
        "exit_decision_time_ns": None,
        "exit_maker_price": None,
        "exit_hedge_status": None,
        "exit_hedge_price": None,
        "exit_hedge_signed_latency_slippage_bp": None,
        "exit_hedge_signed_depth_slippage_bp": None,
        "exit_hedge_signed_total_slippage_bp": None,
        "exit_spot_price": None,
        "exit_future_price": None,
        "exit_basis_bp": None,
        "gross_cycle_pnl_twd": None,
        "instant_cancel_v0": config.instant_cancel_v0,
        "cancel_ack_observed": False,
        "cancel_race_modeled": False,
        "joint_volume_allocated": False,
        "strict_ev_ready": False,
    }


def _exit_prices(
    outcome: ExitMakerOrderOutcome | None,
    window: IndependentOrderWindow | None,
    *,
    session_date: str | None = None,
) -> tuple[float | None, float | None]:
    if outcome is None or window is None or not outcome.position_flat:
        return None, None
    attempt = outcome.hedge_attempts[0] if outcome.hedge_attempts else None
    execution = attempt.execution if attempt is not None else None
    if execution is None or not execution.hedge_complete:
        return None, None
    maker_price = tick_index_to_price(
        window.target_price_tick,
        market=EXIT_MAKER_ROUTE_CONTRACTS[outcome.route].maker_market,
        session_date=session_date,
    )
    if outcome.route == FUTURE_BID_EXIT_ROUTE:
        return execution.executable_vwap_price, maker_price
    return maker_price, execution.executable_vwap_price


def _gross_cycle_pnl(
    action: dict[str, object],
    exit_spot: float | None,
    exit_future: float | None,
) -> float | None:
    if exit_spot is None or exit_future is None:
        return None
    return (
        float(action["entry_future_price"])
        - float(action["entry_spot_price"])
        + exit_spot
        - exit_future
    ) * float(action["entry_hedge_contract_size_shares"])


def _basis(spot: float | None, future: float | None) -> float | None:
    if spot is None or future is None:
        return None
    return (future / spot - 1.0) * 10_000.0


def _no_winner_branch(outcomes: Iterable[ExitMakerOrderOutcome]) -> str:
    values = tuple(outcomes)
    if any(item.known_filled_maker_quantity is None for item in values):
        return "fill_unknown_at_eod"
    if any((item.known_filled_maker_quantity or 0) > 0 for item in values):
        return "partial_fill_carry_at_eod"
    return "carry_at_eod_cancel_unconfirmed"


def _observation_records(
    observations: tuple[ExitMakerObservation, ...],
    **identity: object,
) -> Iterable[dict[str, object]]:
    for item in observations:
        yield {
            **identity,
            "exit_route": item.route,
            "recv_time_ns": item.cursor.recv_time_ns,
            "event_sequence": item.cursor.event_sequence,
            "row_index": item.cursor.row_index,
            "spread_pair_epoch": item.spread_pair_epoch,
            "threshold_basis_bp": item.threshold_basis_bp,
            "spot_sell_arrival_vwap_price": item.spot_sell_exec_price,
            "future_buy_arrival_vwap_price": item.future_buy_exec_price,
            "target_price": item.target_price,
            "target_price_tick": item.absolute_target_tick,
            "effective_basis_bp": item.effective_basis_bp,
            "passive_clamped": item.passive_clamped,
            "threshold_already_taker_executable": item.threshold_already_taker_executable,
            "target_rank": item.target_rank,
            "initial_queue_ahead": item.initial_queue_ahead,
            "gate_open": item.gate_open,
            "gate_reason": item.gate_reason,
        }


def _transition_records(
    transitions: tuple[IntentTransition, ...],
    *,
    exit_route: str,
    candidate_ids: Mapping[str, tuple[str, int]],
    **identity: object,
) -> Iterable[dict[str, object]]:
    for item in transitions:
        yield {
            **identity,
            "exit_route": exit_route,
            "exit_raw_candidate_fact_id": (
                candidate_ids[item.generation_id][0]
                if item.generation_id is not None else None
            ),
            "exit_policy_candidate_id": item.generation_id,
            "kind": item.kind,
            "reason": item.reason,
            "recv_time_ns": item.cursor.recv_time_ns,
            "event_sequence": item.cursor.event_sequence,
            "row_index": item.cursor.row_index,
            "spread_pair_epoch": item.spread_pair_epoch,
            "target_price_tick": item.target_price_tick,
            "target_rank": item.target_rank,
            "active_layers_after": item.active_layers_after,
        }


def _peak_active_layers(transitions: tuple[IntentTransition, ...]) -> int:
    return max((item.active_layers_after for item in transitions), default=0)


def _required_vwap(levels: tuple[BookLevel, ...], quantity: int) -> float | None:
    remaining = quantity
    notional = 0.0
    for level in levels:
        take = min(remaining, level.quantity)
        notional += take * level.price
        remaining -= take
        if remaining == 0:
            return notional / quantity
    return None


def _raw_ref_gate(row: dict[str, object]) -> bool:
    reference = _positive_float(row.get("ref_price"))
    bid = _positive_float(row.get("exec_bid_price"))
    ask = _positive_float(row.get("exec_ask_price"))
    return (
        reference is not None
        and bid is not None
        and ask is not None
        and bid < ask
        and price_in_ref_band(bid, reference)
        and price_in_ref_band(ask, reference)
    )


def _rank_and_queue(
    target: float,
    side: Literal["bid", "ask"],
    row: dict[str, object],
) -> tuple[str, int | None]:
    candidates: list[int] = []
    visible_level: int | None = None
    for level in range(1, 6):
        price = _positive_float(row.get(f"{side}_price_{level}"))
        lots = _optional_int(row.get(f"{side}_lots_{level}"))
        if price is not None and math.isclose(target, price, abs_tol=1e-8):
            visible_level = visible_level or level
            if lots is not None and lots > 0:
                candidates.append(lots)
    best_price = _positive_float(row.get(f"best_{side}_price"))
    best_lots = _optional_int(row.get(f"best_{side}_lots"))
    if best_price is not None and math.isclose(target, best_price, abs_tol=1e-8):
        visible_level = visible_level or 1
        if best_lots is not None and best_lots > 0:
            candidates.append(best_lots)
    if candidates:
        return f"{side.upper()}{visible_level or 1}", max(candidates)
    bid = _positive_float(row.get("exec_bid_price"))
    ask = _positive_float(row.get("exec_ask_price"))
    if bid is not None and ask is not None and bid < target < ask:
        return "inside", 0
    return "behind_visible", None


def _candidate_ids(
    replay: _PhysicalReplay,
    *,
    entry_raw_order_fact_id: str,
    exit_route: str,
) -> dict[str, tuple[str, int]]:
    epochs = {
        transition.generation_id: transition.spread_pair_epoch
        for transition in replay.transitions
        if transition.kind == "submit" and transition.generation_id is not None
    }
    result: dict[str, tuple[str, int]] = {}
    for window in replay.windows:
        try:
            epoch = epochs[window.generation_id]
        except KeyError as error:
            raise AssertionError("exit candidate is missing its submit epoch") from error
        components = (
            entry_raw_order_fact_id,
            exit_route,
            epoch,
            window.target_price_tick,
            window.start_cursor.recv_time_ns,
            window.start_cursor.event_sequence,
            window.start_cursor.row_index,
        )
        digest = hashlib.sha256(
            "|".join(map(str, components)).encode()
        ).hexdigest()[:24]
        result[window.generation_id] = (f"exit-raw-{digest}", epoch)
    return result


def _collapse_raw_candidate_records(
    records: list[dict[str, object]],
) -> pl.DataFrame:
    """Collapse rule aliases onto one canonical raw exit-order identity.

    The canonical key deliberately excludes q alias, rule id and threshold.
    When several policies share a submit, the longest nominal observation
    horizon is retained as the representative raw trace; every policy-specific
    stop/fill label remains available in ``candidate_aliases``.
    """

    frame = _from_records(records, _candidate_variant_schema())
    return _collapse_raw_candidate_frame(frame)


def _collapse_raw_candidate_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse an already-columnar candidate-variant frame exactly once."""

    if frame.is_empty():
        return pl.DataFrame(schema=_candidate_schema())
    expected = set(_candidate_variant_schema())
    observed = set(frame.columns)
    if observed != expected:
        raise AssertionError(
            "candidate variant/schema columns disagree: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    immutable = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_raw_order_fact_id",
        "exit_route",
        "spread_pair_epoch",
        "target_price_tick",
        "target_price",
        "intended_maker_quantity",
        "submit_recv_time_ns",
        "submit_event_sequence",
        "submit_row_index",
        "initial_queue_ahead",
    ]
    inconsistent = frame.group_by("exit_raw_candidate_fact_id").agg(
        *(pl.col(column).n_unique().alias(column) for column in immutable)
    ).filter(pl.any_horizontal(*(pl.col(column) != 1 for column in immutable)))
    if inconsistent.height:
        raise AssertionError("canonical exit raw candidate identity is inconsistent")
    variants = frame.group_by("exit_raw_candidate_fact_id").agg(
        pl.len().alias("policy_variant_count"),
        pl.col("exit_rule_id").sort().unique().alias("exit_rule_ids"),
    )
    representative = (
        frame.sort(
            [
                "exit_raw_candidate_fact_id",
                "nominal_cancel_request_recv_time_ns",
                "exit_policy_candidate_id",
            ],
            descending=[False, True, False],
        )
        .unique(subset=["exit_raw_candidate_fact_id"], keep="first")
        .join(variants, on="exit_raw_candidate_fact_id", how="left", validate="1:1")
        .with_columns(
            pl.lit("max_nominal_policy_horizon").alias("raw_fact_replay_horizon")
        )
    )
    return representative.with_columns(
        pl.col("policy_variant_count").cast(pl.Int64),
        pl.col("exit_rule_ids").cast(pl.List(pl.String)),
        pl.col("raw_fact_replay_horizon").cast(pl.String),
    ).sort(["entry_raw_order_fact_id", "exit_route", "submit_recv_time_ns"])


def _physical_policy_id(key: tuple[object, ...]) -> str:
    digest = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:20]
    return f"exit-physical-{digest}"


def _physical_replay_input_signature(
    raw_tape: RawTapeDay,
    spread_pair_clock: pl.DataFrame,
) -> tuple[tuple[str, str], ...]:
    """Fingerprint every frame which can affect a cached physical replay.

    The work is deliberately performed only for an explicit cross-call cache.
    Normal same-day runner calls pass no cache and therefore pay no hashing
    cost.  Row order is included because it is part of raw event semantics.
    """

    frames = (
        ("mapping", raw_tape.mapping),
        ("spot_states", raw_tape.spot_states),
        ("future_states", raw_tape.future_states),
        ("spot_trades", raw_tape.spot_trades),
        ("future_trades", raw_tape.future_trades),
        ("audit", raw_tape.audit),
        ("spread_pair_clock", spread_pair_clock),
    )
    return tuple(
        (name, _frame_content_sha256(frame)) for name, frame in frames
    )


def _frame_content_sha256(frame: pl.DataFrame) -> str:
    """Return a bounded-memory deterministic schema, order and value digest."""

    digest = hashlib.sha256()
    digest.update(f"{frame.height}:{frame.width}\n".encode("utf-8"))
    for name, dtype in frame.schema.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(dtype).encode("utf-8"))
        digest.update(b"\n")
    if frame.height:
        # Polars computes this without expanding rows to Python objects.  The
        # enclosing SHA binds both the typed schema and ordered row hashes.
        row_hashes = frame.hash_rows(
            seed=0x243F6A8885A308D3,
            seed_1=0x13198A2E03707344,
            seed_2=0xA4093822299F31D0,
            seed_3=0x082EFA98EC4E6C89,
        )
        digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def _trim_lru(cache: dict[tuple[object, ...], object], max_entries: int) -> None:
    """Apply a new hard retention bound before looking up any old entry."""

    if max_entries == 0:
        cache.clear()
        return
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def _lru_get(
    cache: dict[tuple[object, ...], object],
    key: tuple[object, ...],
) -> object | None:
    """Return and refresh one insertion-ordered cache entry."""

    try:
        value = cache.pop(key)
    except KeyError:
        return None
    cache[key] = value
    return value


def _lru_put(
    cache: dict[tuple[object, ...], object],
    key: tuple[object, ...],
    value: object,
    max_entries: int,
) -> None:
    """Insert one value while bounding retained replay object graphs."""

    cache.pop(key, None)
    if max_entries == 0:
        return
    cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def _cursor_field(cursor: EventCursor | None, field: str) -> int | None:
    return None if cursor is None else int(getattr(cursor, field))


def _positive_float(value: object) -> float | None:
    return float(value) if _finite(value) and float(value) > 0 else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _from_records(
    records: list[dict[str, object]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=schema)
    frame = pl.from_dicts(records, infer_schema_length=None)
    expected = set(schema)
    observed = set(frame.columns)
    if observed != expected:
        raise AssertionError(
            "record/schema columns disagree: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return frame.select(
        *(pl.col(column).cast(dtype).alias(column) for column, dtype in schema.items())
    )


def _empty_result(
    tape: RawTapeDay, config: ExitMakerStudyConfig
) -> ExitMakerProductDayResult:
    if tape.mapping.height == 1:
        value_code = str(tape.mapping.item(0, "ValueCode"))
        quote_code = str(tape.mapping.item(0, "QuoteCode"))
    else:
        value_code = quote_code = None
    return _zero_established_result(
        str(tape.date),
        value_code,
        quote_code,
        all_entry_aliases=0,
        config=config,
        lineage_validated=False,
    )


def _zero_established_result(
    date: str,
    value_code: str | None,
    quote_code: str | None,
    *,
    all_entry_aliases: int,
    config: ExitMakerStudyConfig,
    lineage_validated: bool = True,
) -> ExitMakerProductDayResult:
    """Return an exact-identity zero conditional-exit denominator."""

    audit = pl.from_dicts(
        [
            {
                "Date": str(date),
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "all_entry_aliases": all_entry_aliases,
                "all_entry_policy_aliases": all_entry_aliases,
                "entry_policy_aliases": 0,
                "expected_exit_rules": len(config.expected_exit_rule_ids),
                "expected_exit_routes": len(SUPPORTED_EXIT_MAKER_ROUTES),
                "expected_policy_trials": 0,
                "materialized_policy_trials": 0,
                "position_established_trials": 0,
                "no_admission_trials": 0,
                "unique_physical_entry_positions": 0,
                "physical_exit_policy_replays": 0,
                "raw_candidate_facts": 0,
                "candidate_alias_rows": 0,
                "oco_winner_trials": 0,
                "flat_same_day_trials": 0,
                "cancel_ack_observed_rows": 0,
                "cancel_race_modeled_rows": 0,
                "strict_ev_ready_rows": 0,
                "instant_cancel_v0": True,
                "d_minus_one_lineage_validated": lineage_validated,
                "joint_volume_allocated": False,
            }
        ],
        infer_schema_length=None,
    )
    return ExitMakerProductDayResult(
        pl.DataFrame(schema=_support_schema()),
        pl.DataFrame(schema=_observation_schema()),
        pl.DataFrame(schema=_transition_schema()),
        pl.DataFrame(schema=_candidate_alias_schema()),
        pl.DataFrame(schema=_candidate_schema()),
        pl.DataFrame(schema=_position_policy_schema()),
        audit,
    )


def _identity_schema() -> dict[str, pl.DataType]:
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
    }


def _support_schema() -> dict[str, pl.DataType]:
    return {
        **_identity_schema(),
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "exit_lifecycle_policy_version": pl.String,
        "exit_queue_scenario": pl.String,
        "position_status": pl.String,
        "position_established_ns": pl.Int64,
        "admission_status": pl.String,
        "observation_count": pl.Int64,
        "raw_candidate_count": pl.Int64,
        "peak_active_layers": pl.Int64,
    }


def _position_policy_schema() -> dict[str, pl.DataType]:
    return {
        **_identity_schema(),
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "exit_lifecycle_policy_version": pl.String,
        "exit_queue_scenario": pl.String,
        "position_status": pl.String,
        "position_established_ns": pl.Int64,
        "physical_exit_policy_id": pl.String,
        "exit_style": pl.String,
        "exit_maker_quantity": pl.Int64,
        "exit_hedge_quantity": pl.Int64,
        "raw_candidate_count": pl.Int64,
        "oco_winner_generation_id": pl.String,
        "oco_winner_raw_candidate_fact_id": pl.String,
        "oco_winner_spread_pair_epoch": pl.Int64,
        "oco_winner_target_price_tick": pl.Int64,
        "oco_winner_submit_recv_time_ns": pl.Int64,
        "oco_winner_submit_event_sequence": pl.Int64,
        "oco_winner_submit_row_index": pl.Int64,
        "oco_winner_raw_identity_excludes_rule": pl.Boolean,
        "oco_winner_fill_recv_time_ns": pl.Int64,
        "oco_same_cursor_winner_count": pl.Int64,
        "oco_active_sibling_cancel_count": pl.Int64,
        "prior_unacked_cancel_count_before_winner": pl.Int64,
        "oco_position_projection_safe": pl.Boolean,
        "nominal_instant_cancel_v0_branch": pl.String,
        "branch_status": pl.String,
        "terminal_outcome": pl.Boolean,
        "needs_next_session_label": pl.Boolean,
        "exit_decision_time_ns": pl.Int64,
        "exit_maker_price": pl.Float64,
        "exit_hedge_status": pl.String,
        "exit_hedge_price": pl.Float64,
        "exit_hedge_signed_latency_slippage_bp": pl.Float64,
        "exit_hedge_signed_depth_slippage_bp": pl.Float64,
        "exit_hedge_signed_total_slippage_bp": pl.Float64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "exit_basis_bp": pl.Float64,
        "gross_cycle_pnl_twd": pl.Float64,
        "instant_cancel_v0": pl.Boolean,
        "cancel_ack_observed": pl.Boolean,
        "cancel_race_modeled": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
        "strict_ev_ready": pl.Boolean,
    }


def _observation_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "physical_exit_policy_id": pl.String,
        "exit_route": pl.String,
        "recv_time_ns": pl.Int64,
        "event_sequence": pl.Int64,
        "row_index": pl.Int64,
        "spread_pair_epoch": pl.Int64,
        "threshold_basis_bp": pl.Float64,
        "spot_sell_arrival_vwap_price": pl.Float64,
        "future_buy_arrival_vwap_price": pl.Float64,
        "target_price": pl.Float64,
        "target_price_tick": pl.Int64,
        "effective_basis_bp": pl.Float64,
        "passive_clamped": pl.Boolean,
        "threshold_already_taker_executable": pl.Boolean,
        "target_rank": pl.String,
        "initial_queue_ahead": pl.Int64,
        "gate_open": pl.Boolean,
        "gate_reason": pl.String,
    }


def _transition_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "physical_exit_policy_id": pl.String,
        "exit_route": pl.String,
        "exit_raw_candidate_fact_id": pl.String,
        "exit_policy_candidate_id": pl.String,
        "kind": pl.String,
        "reason": pl.String,
        "recv_time_ns": pl.Int64,
        "event_sequence": pl.Int64,
        "row_index": pl.Int64,
        "spread_pair_epoch": pl.Int64,
        "target_price_tick": pl.Int64,
        "target_rank": pl.String,
        "active_layers_after": pl.Int64,
    }


def _candidate_alias_schema() -> dict[str, pl.DataType]:
    return {
        **_identity_schema(),
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "exit_lifecycle_policy_version": pl.String,
        "exit_queue_scenario": pl.String,
        "position_status": pl.String,
        "position_established_ns": pl.Int64,
        "physical_exit_policy_id": pl.String,
        "exit_raw_candidate_fact_id": pl.String,
        "exit_policy_candidate_id": pl.String,
        "exit_candidate_alias_id": pl.String,
        "spread_pair_epoch": pl.Int64,
        "target_price_tick": pl.Int64,
        "submit_recv_time_ns": pl.Int64,
        "nominal_cancel_request_recv_time_ns": pl.Int64,
        "nominal_stop_reason": pl.String,
        "known_filled_maker_quantity": pl.Int64,
        "any_fill": pl.Boolean,
        "full_fill": pl.Boolean,
        "partial_fill": pl.Boolean,
        "independent_branch_status": pl.String,
        "cancel_required": pl.Boolean,
        "cancel_ack_observed": pl.Boolean,
        "cancel_race_modeled": pl.Boolean,
        "exit_hedge_status": pl.String,
    }


def _candidate_variant_schema() -> dict[str, pl.DataType]:
    # Policy-specific traces before canonical raw-order collapse.
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "physical_exit_policy_id": pl.String,
        "exit_raw_candidate_fact_id": pl.String,
        "exit_policy_candidate_id": pl.String,
        "spread_pair_epoch": pl.Int64,
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "target_price_tick": pl.Int64,
        "target_price": pl.Float64,
        "intended_maker_quantity": pl.Int64,
        "submit_recv_time_ns": pl.Int64,
        "submit_event_sequence": pl.Int64,
        "submit_row_index": pl.Int64,
        "nominal_cancel_request_recv_time_ns": pl.Int64,
        "nominal_cancel_request_event_sequence": pl.Int64,
        "nominal_cancel_request_row_index": pl.Int64,
        "nominal_stop_reason": pl.String,
        "initial_queue_ahead": pl.Int64,
        "queue_known": pl.Boolean,
        "known_filled_maker_quantity": pl.Int64,
        "remaining_maker_quantity": pl.Int64,
        "any_fill": pl.Boolean,
        "full_fill": pl.Boolean,
        "partial_fill": pl.Boolean,
        "first_fill_recv_time_ns": pl.Int64,
        "full_fill_recv_time_ns": pl.Int64,
        "trade_through_fill": pl.Boolean,
        "independent_branch_status": pl.String,
        "cancel_required": pl.Boolean,
        "instant_cancel_v0": pl.Boolean,
        "cancel_ack_observed": pl.Boolean,
        "cancel_race_modeled": pl.Boolean,
        "oco_disposition": pl.String,
        "oco_cancel_request_recv_time_ns": pl.Int64,
        "independent_full_fill_after_oco_cancel_request": pl.Boolean,
        "partial_fill_before_oco_winner": pl.Boolean,
        "selected_for_position": pl.Boolean,
        "exit_maker_price": pl.Float64,
        "exit_hedge_status": pl.String,
        "exit_hedge_decision_time_ns": pl.Int64,
        "exit_hedge_requested_quantity": pl.Int64,
        "exit_hedge_executed_quantity": pl.Int64,
        "exit_hedge_price": pl.Float64,
        "exit_hedge_signed_latency_slippage_bp": pl.Float64,
        "exit_hedge_signed_depth_slippage_bp": pl.Float64,
        "exit_hedge_signed_total_slippage_bp": pl.Float64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "exit_basis_bp": pl.Float64,
        "gross_cycle_pnl_twd": pl.Float64,
        "residual_spot_long_lots": pl.Int64,
        "residual_future_short_contracts": pl.Int64,
        "joint_volume_allocated": pl.Boolean,
        "strict_ev_ready": pl.Boolean,
    }


def _candidate_schema() -> dict[str, pl.DataType]:
    # Persisted canonical facts, including rule-alias provenance.
    return {
        **_candidate_variant_schema(),
        "policy_variant_count": pl.Int64,
        "exit_rule_ids": pl.List(pl.String),
        "raw_fact_replay_horizon": pl.String,
    }
