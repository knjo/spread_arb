"""Atomic runner for frozen maker exits continued across market sessions.

This runner is intentionally disjoint from the historical forced-next-session
taker/taker benchmark.  It consumes immutable entry-execution and same-day
exit-maker partitions, verifies their completion markers and artifact hashes,
then carries only an established paired position into fresh day-order maker
replays.

Formal CLI runs also carry a verified prerequisite identity through the runner
config, every partition marker and the root manifest.  The programmatic API
keeps that identity optional for isolated tests, while validating it before
creating any output root whenever supplied.

Candidate-session normalization uses that session's point-in-time spot and
futures reference prices.  The spot reference lookup deliberately does not
apply the new-entry ``allow_day_trade_mark`` filter: an already-owned spot leg
may be sold to close even when the candidate day is marked ``N``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT, market_data_path
from .execution_runner import load_spread_pair_clock_for_raw_day
from .exit_maker_cross_session import (
    CrossSessionExitMakerConfig,
    CrossSessionExitMakerResult,
    CrossSessionExitMakerSession,
    _attempt_schema,
    _empty_result,
    _outcome_schema,
    replay_cross_session_exit_maker,
)
from .exit_maker import SUPPORTED_EXIT_MAKER_ROUTES
from .exit_maker_study import ExitMakerProductDayResult
from .raw_tape import RawTapeDay, load_raw_tape_day


CROSS_SESSION_RUNNER_VERSION = (
    "frozen_exit_maker_cross_session_runner_v8_fixed_output_schemas"
)
CROSS_SESSION_PREREQUISITE_BINDING_VERSION = (
    "cross_session_prerequisite_binding_v1"
)
CANDIDATE_CACHE_SCHEMA_VERSION = "cross_session_candidate_cache_v1"
DEFAULT_ENTRY_EXECUTION_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "execution_narrow_60d"
)
DEFAULT_EXIT_MAKER_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "exit_maker_narrow_60d"
)
DEFAULT_CROSS_SESSION_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "exit_maker_cross_session_narrow"
)
DEFAULT_CONTRACT_METADATA_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "daily" / "metadata"
)
DEFAULT_FUTURES_RAW_ROOT = Path("/mnt/NAS/Parquet/Ticks")
CROSS_SESSION_MANIFEST_NAME = "cross_session_partition_manifest.parquet"

_ACTION_ARTIFACT = "execution_action_facts.parquet"
_EXIT_RULE_ARTIFACT = "exit_facts.parquet"
_TARGET_AUDIT_ARTIFACT = "target_audit.parquet"
_EXIT_ARTIFACTS = {
    "policy_support": "exit_maker_policy_support.parquet",
    "observations": "exit_maker_observations.parquet",
    "transitions": "exit_maker_transitions.parquet",
    "candidate_aliases": "exit_maker_candidate_aliases.parquet",
    "raw_candidate_facts": "exit_maker_raw_candidate_facts.parquet",
    "position_policy_facts": "exit_maker_position_policy_facts.parquet",
    "audit": "exit_maker_audit.parquet",
}
_STRICT_OUTCOMES = "cross_session_strict_policy_outcomes.parquet"
_STRICT_ATTEMPTS = "cross_session_strict_session_attempts.parquet"
_NOMINAL_OUTCOMES = "cross_session_nominal_policy_outcomes.parquet"
_NOMINAL_ATTEMPTS = "cross_session_nominal_session_attempts.parquet"
_AUDIT_ARTIFACT = "cross_session_runner_audit.parquet"
_OUTPUT_ARTIFACTS = (
    _STRICT_OUTCOMES,
    _STRICT_ATTEMPTS,
    _NOMINAL_OUTCOMES,
    _NOMINAL_ATTEMPTS,
    _AUDIT_ARTIFACT,
)
_CANDIDATE_CACHE_ARTIFACTS = {
    "mapping": "mapping.parquet",
    "spot_states": "spot_states.parquet",
    "future_states": "future_states.parquet",
    "spot_trades": "spot_trades.parquet",
    "future_trades": "future_trades.parquet",
    "raw_audit": "raw_audit.parquet",
    "spread_pair_clock": "spread_pair_clock.parquet",
}
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")

_RUNNER_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "cancel_semantics": pl.String,
    "policy_outcome_rows": pl.Int64,
    "session_attempt_rows": pl.Int64,
    "completed_rows": pl.Int64,
    "same_day_completed_rows": pl.Int64,
    "cross_session_completed_rows": pl.Int64,
    "still_open_rows": pl.Int64,
    "censored_rows": pl.Int64,
    "unknown_rows": pl.Int64,
    "candidate_session_count": pl.Int64,
    "loaded_candidate_session_count": pl.Int64,
    "loaded_candidate_sessions": pl.List(pl.String),
    "load_failure_status": pl.String,
    "load_failure_detail": pl.String,
    "old_forced_taker_taker_used": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
}

_OUTPUT_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    _STRICT_OUTCOMES: _outcome_schema(),
    _STRICT_ATTEMPTS: _attempt_schema(),
    _NOMINAL_OUTCOMES: _outcome_schema(),
    _NOMINAL_ATTEMPTS: _attempt_schema(),
    _AUDIT_ARTIFACT: _RUNNER_AUDIT_SCHEMA,
}


@dataclass(frozen=True)
class CrossSessionRunnerConfig:
    """Frozen source, replay and output semantics for one result root."""

    hedge_delay_ns: int = 50_000_000
    book_age_diagnostic_threshold_ns: int = 1_000_000_000
    expected_exit_rule_ids: tuple[str, ...] = (
        "frozen_center",
        "frozen_lower",
    )
    action_source_artifact: str = _ACTION_ARTIFACT
    exit_rule_source_artifact: str = _EXIT_RULE_ARTIFACT
    target_audit_source_artifact: str = _TARGET_AUDIT_ARTIFACT
    exit_maker_source_artifacts: tuple[str, ...] = tuple(
        _EXIT_ARTIFACTS.values()
    )
    candidate_reference_policy: str = (
        "candidate_session_marketData_spot_plus_exact_futures_metadata_v1"
    )
    candidate_spot_day_trade_policy: str = (
        "existing_long_spot_exit_all_marks_X_Y_N_no_entry_filter_v1"
    )
    raw_source_integrity: str = "path_size_mtime_ns_fingerprint_v1"
    prerequisite_identity: Mapping[str, object] | None = None
    runner_version: str = CROSS_SESSION_RUNNER_VERSION

    def validate(self) -> None:
        for name, value in (
            ("hedge_delay_ns", self.hedge_delay_ns),
            (
                "book_age_diagnostic_threshold_ns",
                self.book_age_diagnostic_threshold_ns,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.hedge_delay_ns != 50_000_000:
            raise ValueError("formal cross-session runner requires a 50 ms hedge delay")
        if tuple(self.expected_exit_rule_ids) != (
            "frozen_center",
            "frozen_lower",
        ):
            raise ValueError("runner requires frozen_center and frozen_lower")
        if set(self.exit_maker_source_artifacts) != set(
            _EXIT_ARTIFACTS.values()
        ):
            raise ValueError("runner must bind all seven exit-maker artifacts")
        for name in (
            "action_source_artifact",
            "exit_rule_source_artifact",
            "target_audit_source_artifact",
            "candidate_reference_policy",
            "candidate_spot_day_trade_policy",
            "raw_source_integrity",
            "runner_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for value in (
            self.action_source_artifact,
            self.exit_rule_source_artifact,
            self.target_audit_source_artifact,
            *self.exit_maker_source_artifacts,
        ):
            if Path(value).name != value:
                raise ValueError("source artifact names must be plain filenames")
        if self.runner_version != CROSS_SESSION_RUNNER_VERSION:
            raise ValueError(
                f"runner_version must be {CROSS_SESSION_RUNNER_VERSION!r}"
            )
        _normalise_prerequisite_identity(self.prerequisite_identity)

    def cross_config(self, cancel_semantics: str) -> CrossSessionExitMakerConfig:
        return CrossSessionExitMakerConfig(
            hedge_delay_ns=self.hedge_delay_ns,
            expected_exit_rule_ids=self.expected_exit_rule_ids,
            book_age_diagnostic_threshold_ns=(
                self.book_age_diagnostic_threshold_ns
            ),
            cancel_semantics=cancel_semantics,  # type: ignore[arg-type]
            lifecycle_policy_version=(
                "frozen_exit_maker_cross_session_runner_v2"
            ),
        )


@dataclass(frozen=True)
class CrossSessionPartitionSource:
    date: str
    value_code: str
    entry_partition: Path
    exit_partition: Path
    action_path: Path
    exit_rule_path: Path
    target_audit_path: Path
    exit_paths: Mapping[str, Path]
    entry_marker_sha256: str
    exit_marker_sha256: str
    entry_runner_version: str
    exit_runner_version: str
    entry_config_sha256: str
    exit_config_sha256: str
    artifact_sha256: Mapping[str, str]

    def payload(self, config: CrossSessionRunnerConfig) -> dict[str, object]:
        return {
            "Date": self.date,
            "ValueCode": self.value_code,
            "entry_execution": {
                "partition": str(self.entry_partition),
                "runner_version": self.entry_runner_version,
                "config_sha256": self.entry_config_sha256,
                "marker_sha256": self.entry_marker_sha256,
                "artifacts": {
                    config.action_source_artifact: self.artifact_sha256[
                        config.action_source_artifact
                    ],
                    config.exit_rule_source_artifact: self.artifact_sha256[
                        config.exit_rule_source_artifact
                    ],
                    config.target_audit_source_artifact: self.artifact_sha256[
                        config.target_audit_source_artifact
                    ],
                },
            },
            "same_day_exit_maker": {
                "partition": str(self.exit_partition),
                "runner_version": self.exit_runner_version,
                "config_sha256": self.exit_config_sha256,
                "marker_sha256": self.exit_marker_sha256,
                "artifacts": {
                    name: self.artifact_sha256[name]
                    for name in config.exit_maker_source_artifacts
                },
            },
        }


@dataclass(frozen=True)
class CandidateSessionRequirement:
    """One immutable exact-contract candidate session needed by an origin."""

    date: str
    value_code: str
    quote_code: str
    source_fingerprint: Mapping[str, object]

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.date, self.value_code, self.quote_code


@dataclass(frozen=True)
class CandidateSessionFailure:
    status: str
    detail: str


@dataclass(frozen=True)
class _CandidateCacheHandle:
    requirement: CandidateSessionRequirement
    cache_partition: Path | None = None
    failure: CandidateSessionFailure | None = None


@dataclass(frozen=True)
class _CrossSessionOriginPlan:
    date: str
    value_code: str
    source: CrossSessionPartitionSource
    quote_code: str
    established_count: int
    candidate_dates: tuple[str, ...]
    raw_fingerprints: tuple[Mapping[str, object], ...]
    partition_payload: Mapping[str, object]
    config_sha256: str
    partition: Path
    resumed: bool


CandidateSessionLoader = Callable[
    [str, str, str], CrossSessionExitMakerSession
]
CandidateSourceValidator = Callable[[CandidateSessionRequirement], None]


def run_cross_session_exit_replay(
    product_days: Iterable[tuple[str, str]],
    *,
    sessions: Sequence[str] | Iterable[str],
    contract_calendar: pl.DataFrame,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    output_root: Path = DEFAULT_CROSS_SESSION_ROOT,
    config: CrossSessionRunnerConfig = CrossSessionRunnerConfig(),
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    contract_metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
    resume: bool = True,
    candidate_session_loader: CandidateSessionLoader | None = None,
    candidate_cache_enabled: bool = True,
    candidate_cache_root: Path | None = None,
    candidate_cache_max_entries: int = 32,
) -> pl.DataFrame:
    """Replay strict and nominal-V0 cross-session exits atomically."""

    config.validate()
    prerequisite_identity = _normalise_prerequisite_identity(
        config.prerequisite_identity
    )
    if not isinstance(candidate_cache_enabled, bool):
        raise TypeError("candidate_cache_enabled must be boolean")
    if (
        isinstance(candidate_cache_max_entries, bool)
        or not isinstance(candidate_cache_max_entries, int)
        or candidate_cache_max_entries <= 0
    ):
        raise ValueError("candidate_cache_max_entries must be positive")
    session_values = _normalise_sessions(sessions)
    session_index = {value: index for index, value in enumerate(session_values)}
    keys = tuple((str(date), str(value)) for date, value in product_days)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("product_days must be nonempty and unique")
    for date, value_code in keys:
        _validate_partition_key(date, value_code)
        if date not in session_index:
            raise ValueError(f"origin Date absent from sessions: {date}")

    entry_root = Path(entry_execution_root).resolve()
    exit_root = Path(exit_maker_root).resolve()
    destination = Path(output_root).resolve()
    data_source_root = Path(data_root).resolve()
    futures_source_root = Path(futures_raw_root).resolve()
    metadata_source_root = Path(contract_metadata_root).resolve()
    cache_destination = (
        Path(candidate_cache_root).resolve()
        if candidate_cache_root is not None
        else (
            destination.parent
            / f"{destination.name}_candidate_session_cache"
        ).resolve()
    )
    protected_roots = [
        entry_root,
        exit_root,
        data_source_root,
        futures_source_root,
        destination,
    ]
    if prerequisite_identity is not None:
        protected_roots.append(Path(str(prerequisite_identity["root"])))
    # Validate both possible write roots even when the cache is disabled.  A
    # later resume/config change must never turn a source tree into a cache or
    # output destination merely because the original run skipped cache I/O.
    _validate_distinct_roots(*protected_roots, cache_destination)
    destination.mkdir(parents=True, exist_ok=True)

    global_inputs = {
        "sessions": {
            "count": len(session_values),
            "first": session_values[0],
            "last": session_values[-1],
            "content_sha256": _canonical_sha256(list(session_values)),
        },
        "contract_calendar_content_sha256": _frame_sha256(contract_calendar),
        "prerequisite": prerequisite_identity,
    }
    runner_payload = _runner_payload(config, global_inputs)
    runner_sha = _canonical_sha256(runner_payload)
    _preflight_output_root(destination, runner_sha)

    plans: list[_CrossSessionOriginPlan] = []
    requirements: dict[
        tuple[str, str, str], CandidateSessionRequirement
    ] = {}
    for date, value_code in keys:
        source = verify_cross_session_sources(
            date,
            value_code,
            entry_execution_root=entry_root,
            exit_maker_root=exit_root,
            config=config,
        )
        actions = pl.read_parquet(source.action_path)
        exits = pl.read_parquet(source.exit_rule_path)
        policy_support = pl.read_parquet(source.exit_paths["policy_support"])
        position_policy_facts = pl.read_parquet(
            source.exit_paths["position_policy_facts"]
        )
        from .exit_maker_report import (
            _validate_entry_action_execution_contract,
            _validate_position_rule_lineage,
        )

        actions = _validate_entry_action_execution_contract(
            actions,
            expected_hedge_delay_ns=config.hedge_delay_ns,
        )
        _validate_frame_identity(actions, date, value_code, "entry actions")
        _validate_frame_identity(exits, date, value_code, "frozen exit facts")
        origin_offset = session_index.get(date)
        if origin_offset is None or origin_offset == 0:
            raise ValueError(
                f"candidate session calendar lacks D-1 predecessor for {date}"
            )
        frozen_rule_lineage = _crossvalidate_frozen_rule_lineage(
            actions,
            exits,
            policy_support,
            position_policy_facts,
            origin_date=date,
            expected_source_asof_date=session_values[origin_offset - 1],
            expected_exit_rule_ids=config.expected_exit_rule_ids,
        )
        position_rule_lineage = _validate_position_rule_lineage(
            position_policy_facts,
            exits,
            expected_session_predecessors={
                date: session_values[origin_offset - 1]
            },
        )
        quote_code = _source_quote_code(
            source.action_path,
            source.target_audit_path,
            date,
            value_code,
        )
        established_count = _established_entry_count(actions)
        candidate_dates = (
            _candidate_dates(
                date,
                quote_code,
                session_values,
                session_index,
                contract_calendar,
            )
            if established_count
            else ()
        )
        raw_fingerprints = (
            _candidate_source_fingerprints(
                candidate_dates,
                data_root=data_source_root,
                futures_raw_root=futures_source_root,
                contract_metadata_root=metadata_source_root,
                custom_loader=candidate_session_loader is not None,
            )
            if established_count
            else []
        )
        partition_payload = {
            "runner": runner_payload,
            "source": source.payload(config),
            **global_inputs,
            "exact_quote_code": quote_code,
            "established_entry_rows": established_count,
            "frozen_rule_lineage": frozen_rule_lineage,
            "frozen_rule_lineage_crossvalidated": True,
            "same_day_position_rule_lineage": position_rule_lineage,
            "same_day_position_rule_lineage_crossvalidated": True,
            "candidate_sessions": list(candidate_dates),
            "candidate_source_fingerprints": raw_fingerprints,
            "candidate_raw_io_skipped_zero_established": established_count == 0,
        }
        config_sha = _canonical_sha256(partition_payload)
        partition = destination / f"Date={date}" / f"ValueCode={value_code}"
        resumed = partition.exists()
        if resumed:
            if not resume:
                raise FileExistsError(partition)
            verify_cross_session_output_partition(
                partition,
                expected_config_sha256=config_sha,
                expected_runner_config_sha256=runner_sha,
            )
        plans.append(
            _CrossSessionOriginPlan(
                date=date,
                value_code=value_code,
                source=source,
                quote_code=quote_code,
                established_count=established_count,
                candidate_dates=tuple(candidate_dates),
                raw_fingerprints=tuple(raw_fingerprints),
                partition_payload=partition_payload,
                config_sha256=config_sha,
                partition=partition,
                resumed=resumed,
            )
        )
        if not resumed:
            for candidate_date, fingerprint in zip(
                candidate_dates, raw_fingerprints, strict=True
            ):
                requirement = CandidateSessionRequirement(
                    date=candidate_date,
                    value_code=value_code,
                    quote_code=quote_code,
                    source_fingerprint=dict(fingerprint),
                )
                previous = requirements.get(requirement.identity)
                if previous is not None and _canonical_sha256(
                    previous.source_fingerprint
                ) != _canonical_sha256(requirement.source_fingerprint):
                    raise ValueError(
                        "candidate source fingerprint changed within preflight: "
                        f"{requirement.identity}"
                    )
                requirements[requirement.identity] = requirement

    handles: dict[tuple[str, str, str], _CandidateCacheHandle] = {}
    repository: _CandidateSessionRepository | None = None
    if candidate_cache_enabled and requirements:
        handles = _prepare_candidate_session_cache(
            tuple(requirements.values()),
            cache_root=cache_destination,
            config=config,
            data_root=data_source_root,
            futures_raw_root=futures_source_root,
            contract_metadata_root=metadata_source_root,
            custom_loader=candidate_session_loader,
        )
        source_validator: CandidateSourceValidator | None = None
        if candidate_session_loader is None:

            def validate_candidate_source(
                requirement: CandidateSessionRequirement,
            ) -> None:
                _assert_candidate_batch_sources_unchanged(
                    (requirement,),
                    data_root=data_source_root,
                    futures_raw_root=futures_source_root,
                    contract_metadata_root=metadata_source_root,
                )

            source_validator = validate_candidate_source

        repository = _CandidateSessionRepository(
            handles,
            custom_loader=candidate_session_loader,
            max_entries=candidate_cache_max_entries,
            source_validator=source_validator,
        )

    execution_plans = (
        sorted(plans, key=lambda item: (item.value_code, item.date))
        if candidate_cache_enabled
        else plans
    )
    for plan in execution_plans:
        if plan.resumed:
            continue
        date = plan.date
        value_code = plan.value_code
        source = verify_cross_session_sources(
            date,
            value_code,
            entry_execution_root=entry_root,
            exit_maker_root=exit_root,
            config=config,
        )
        if _canonical_sha256(source.payload(config)) != _canonical_sha256(
            plan.source.payload(config)
        ):
            raise ValueError(
                f"cross-session source changed after preflight: {date}/{value_code}"
            )
        actions = pl.read_parquet(source.action_path)
        exits = pl.read_parquet(source.exit_rule_path)
        _validate_frame_identity(actions, date, value_code, "entry actions")
        _validate_frame_identity(exits, date, value_code, "frozen exit facts")
        quote_code = plan.quote_code
        established_count = plan.established_count
        candidate_dates = plan.candidate_dates
        partition_payload = plan.partition_payload
        config_sha = plan.config_sha256
        partition = plan.partition

        same_day = (
            _load_same_day_result(source, policy_only=True)
            if established_count
            else None
        )

        loaded_sessions: list[CrossSessionExitMakerSession] = []
        load_failure_status: str | None = None
        load_failure_detail: str | None = None
        for candidate_date in candidate_dates:
            if repository is not None:
                resolved = repository.get(
                    (candidate_date, value_code, quote_code)
                )
                if isinstance(resolved, CandidateSessionFailure):
                    load_failure_status = resolved.status
                    load_failure_detail = resolved.detail
                    break
                loaded_sessions.append(resolved)
                continue
            try:
                item = (
                    candidate_session_loader(
                        candidate_date, value_code, quote_code
                    )
                    if candidate_session_loader is not None
                    else load_cross_session_candidate(
                        candidate_date,
                        value_code,
                        quote_code,
                        data_root=data_source_root,
                        futures_raw_root=futures_source_root,
                        contract_metadata_root=metadata_source_root,
                    )
                )
                item.validate()
                loaded_sessions.append(item)
            except (FileNotFoundError, ValueError) as error:
                failure = _candidate_failure(error)
                load_failure_status = failure.status
                load_failure_detail = failure.detail
                break

        results: dict[str, CrossSessionExitMakerResult] = {}
        # Both classifiers consume the same physical day replay.  Persist only
        # the compact position-policy facts between them; the large support
        # frames are spooled and discarded because this runner does not
        # publish them.
        shared_day_policy_facts: dict[tuple[object, ...], pl.DataFrame] = {}
        for semantics in ("strict", "nominal_instant_cancel_v0"):
            cross_config = config.cross_config(semantics)
            if established_count == 0:
                result = _empty_result(
                    date,
                    value_code,
                    quote_code,
                    all_entry_rows=actions.height,
                    excluded_entry_rows=actions.height,
                    config=cross_config,
                )
            else:
                result = replay_cross_session_exit_maker(
                    actions,
                    exits,
                    loaded_sessions,
                    session_values,
                    contract_calendar,
                    cross_config,
                    same_day_result=same_day,
                    same_day_book_age_gate_enforced=False,
                    shared_day_policy_facts=shared_day_policy_facts,
                    day_policy_spool_root=destination,
                    retain_session_artifacts=False,
                )
                if load_failure_status is not None:
                    result = _replace_open_horizon_censor(
                        result,
                        status=load_failure_status,
                        detail=load_failure_detail or load_failure_status,
                    )
            results[semantics] = result

        audit = _runner_audit(
            date,
            value_code,
            quote_code,
            results,
            candidate_dates=candidate_dates,
            loaded_sessions=tuple(item.date for item in loaded_sessions),
            load_failure_status=load_failure_status,
            load_failure_detail=load_failure_detail,
        )
        frames = {
            _STRICT_OUTCOMES: results["strict"].policy_outcomes,
            _STRICT_ATTEMPTS: results["strict"].session_attempts,
            _NOMINAL_OUTCOMES: results[
                "nominal_instant_cancel_v0"
            ].policy_outcomes,
            _NOMINAL_ATTEMPTS: results[
                "nominal_instant_cancel_v0"
            ].session_attempts,
            _AUDIT_ARTIFACT: audit,
        }
        _publish_partition(
            partition,
            date=date,
            value_code=value_code,
            frames=frames,
            partition_config=partition_payload,
            config_sha256=config_sha,
            runner_config_sha256=runner_sha,
        )

    return _rebuild_root_manifest(destination, runner_sha)


class _CandidateSessionRepository:
    """Resolve immutable candidate sessions through a bounded in-memory LRU."""

    def __init__(
        self,
        handles: Mapping[tuple[str, str, str], _CandidateCacheHandle],
        *,
        custom_loader: CandidateSessionLoader | None,
        max_entries: int,
        source_validator: CandidateSourceValidator | None,
    ) -> None:
        self._handles = dict(handles)
        self._custom_loader = custom_loader
        self._max_entries = max_entries
        self._source_validator = source_validator
        self._sessions: OrderedDict[
            tuple[str, str, str], CrossSessionExitMakerSession
        ] = OrderedDict()
        self._failures: dict[
            tuple[str, str, str], CandidateSessionFailure
        ] = {}

    def get(
        self, identity: tuple[str, str, str]
    ) -> CrossSessionExitMakerSession | CandidateSessionFailure:
        try:
            handle = self._handles[identity]
        except KeyError as error:
            raise ValueError(
                f"candidate requirement was absent from preflight: {identity}"
            ) from error
        # The cache key binds the five source stat fingerprints observed at
        # preflight.  Re-stat on every repository hit, including in-memory and
        # cached-failure hits, so a source mutation between preflight and use
        # cannot silently reuse the old derived session.
        if self._source_validator is not None:
            self._source_validator(handle.requirement)
        if identity in self._failures:
            return self._failures[identity]
        cached = self._sessions.get(identity)
        if cached is not None:
            self._sessions.move_to_end(identity)
            return cached
        if handle.failure is not None:
            self._failures[identity] = handle.failure
            return handle.failure
        if handle.cache_partition is not None:
            # Cache corruption is an infrastructure failure, not a market-data
            # censor.  Let validation errors fail the run instead of relabeling
            # a position as if the source session were invalid.
            item = _load_candidate_cache_partition(
                handle.requirement,
                handle.cache_partition,
            )
            # Cover a mutation concurrent with the cache artifact read as
            # well as the wider preflight-to-hit interval.
            if self._source_validator is not None:
                self._source_validator(handle.requirement)
        else:
            if self._custom_loader is None:
                raise AssertionError("candidate cache handle has no resolver")
            try:
                item = self._custom_loader(*identity)
                item.validate()
            except (FileNotFoundError, ValueError) as error:
                failure = _candidate_failure(error)
                self._failures[identity] = failure
                return failure
        self._sessions[identity] = item
        self._sessions.move_to_end(identity)
        while len(self._sessions) > self._max_entries:
            self._sessions.popitem(last=False)
        return item


def _prepare_candidate_session_cache(
    requirements: Sequence[CandidateSessionRequirement],
    *,
    cache_root: Path,
    config: CrossSessionRunnerConfig,
    data_root: Path,
    futures_raw_root: Path,
    contract_metadata_root: Path,
    custom_loader: CandidateSessionLoader | None,
) -> dict[tuple[str, str, str], _CandidateCacheHandle]:
    """Materialize each uncached candidate Date with one multi-product raw scan."""

    selected = _validate_candidate_requirements(requirements)
    if custom_loader is not None:
        return {
            item.identity: _CandidateCacheHandle(requirement=item)
            for item in selected
        }

    root = Path(cache_root)
    implementation = _candidate_cache_implementation_sources()
    handles: dict[tuple[str, str, str], _CandidateCacheHandle] = {}
    misses_by_date: dict[str, list[CandidateSessionRequirement]] = {}
    for item in selected:
        partition = _candidate_cache_partition(
            root,
            item,
            config=config,
            implementation_sources=implementation,
        )
        if partition.exists():
            payload = _verify_candidate_cache_partition(
                item,
                partition,
                config=config,
                implementation_sources=implementation,
            )
            failure = _failure_from_cache_payload(payload)
            handles[item.identity] = _CandidateCacheHandle(
                requirement=item,
                cache_partition=None if failure is not None else partition,
                failure=failure,
            )
        else:
            misses_by_date.setdefault(item.date, []).append(item)

    for date in sorted(misses_by_date):
        batch = tuple(
            sorted(
                misses_by_date[date],
                key=lambda item: (item.value_code, item.quote_code),
            )
        )
        _assert_candidate_batch_sources_unchanged(
            batch,
            data_root=data_root,
            futures_raw_root=futures_raw_root,
            contract_metadata_root=contract_metadata_root,
        )
        resolved = _load_candidate_date_batch(
            batch,
            data_root=data_root,
            futures_raw_root=futures_raw_root,
            contract_metadata_root=contract_metadata_root,
        )
        _assert_candidate_batch_sources_unchanged(
            batch,
            data_root=data_root,
            futures_raw_root=futures_raw_root,
            contract_metadata_root=contract_metadata_root,
        )
        if set(resolved) != {item.identity for item in batch}:
            raise AssertionError("candidate date batch did not resolve every request")
        for item in batch:
            partition = _candidate_cache_partition(
                root,
                item,
                config=config,
                implementation_sources=implementation,
            )
            value = resolved[item.identity]
            _publish_candidate_cache_partition(
                item,
                value,
                partition,
                config=config,
                implementation_sources=implementation,
            )
            failure = value if isinstance(value, CandidateSessionFailure) else None
            handles[item.identity] = _CandidateCacheHandle(
                requirement=item,
                cache_partition=None if failure is not None else partition,
                failure=failure,
            )
    return handles


def _assert_candidate_batch_sources_unchanged(
    requirements: Sequence[CandidateSessionRequirement],
    *,
    data_root: Path,
    futures_raw_root: Path,
    contract_metadata_root: Path,
) -> None:
    expected = requirements[0].source_fingerprint
    required_sources = {
        "spot_raw",
        "future_raw",
        "spread_clock",
        "spot_reference",
        "future_reference",
    }
    if not required_sources.issubset(expected):
        return
    current = _candidate_source_fingerprints(
        (requirements[0].date,),
        data_root=data_root,
        futures_raw_root=futures_raw_root,
        contract_metadata_root=contract_metadata_root,
        custom_loader=False,
    )[0]
    if _canonical_sha256(current) != _canonical_sha256(expected):
        raise ValueError(
            f"candidate source changed after preflight: {requirements[0].date}"
        )


def _validate_candidate_requirements(
    requirements: Sequence[CandidateSessionRequirement],
) -> tuple[CandidateSessionRequirement, ...]:
    selected = tuple(requirements)
    identities = [item.identity for item in selected]
    if not selected or len(identities) != len(set(identities)):
        raise ValueError("candidate requirements must be nonempty and unique")
    by_date: dict[str, list[CandidateSessionRequirement]] = {}
    for item in selected:
        _date(item.date, "candidate Date")
        _validate_partition_key(item.date, item.value_code)
        if not _SAFE_KEY.fullmatch(item.quote_code):
            raise ValueError(f"unsafe candidate QuoteCode: {item.quote_code!r}")
        if str(item.source_fingerprint.get("Date")) != item.date:
            raise ValueError("candidate source fingerprint Date mismatch")
        by_date.setdefault(item.date, []).append(item)
    for date, rows in by_date.items():
        fingerprints = {
            _canonical_sha256(item.source_fingerprint) for item in rows
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"candidate source fingerprints disagree within Date {date}"
            )
        value_quotes: dict[str, set[str]] = {}
        quote_values: dict[str, set[str]] = {}
        for item in rows:
            value_quotes.setdefault(item.value_code, set()).add(item.quote_code)
            quote_values.setdefault(item.quote_code, set()).add(item.value_code)
        conflicts = {
            value: sorted(quotes)
            for value, quotes in value_quotes.items()
            if len(quotes) != 1
        }
        reverse_conflicts = {
            quote: sorted(values)
            for quote, values in quote_values.items()
            if len(values) != 1
        }
        if conflicts or reverse_conflicts:
            raise ValueError(
                "candidate Date cannot be batched without exact-contract "
                f"substitution: Date={date}, ValueCode={conflicts}, "
                f"QuoteCode={reverse_conflicts}"
            )
    return tuple(sorted(selected, key=lambda item: item.identity))


def _candidate_cache_implementation_sources() -> dict[str, str]:
    module_root = Path(__file__).parent
    return {
        name: _file_sha256(module_root / name)
        for name in (
            "exit_maker_cross_session_runner.py",
            "raw_tape.py",
            "execution_runner.py",
        )
    }


def _candidate_cache_contract(
    requirement: CandidateSessionRequirement,
    *,
    config: CrossSessionRunnerConfig,
    implementation_sources: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "Date": requirement.date,
        "ValueCode": requirement.value_code,
        "QuoteCode": requirement.quote_code,
        "source_fingerprint": dict(requirement.source_fingerprint),
        "implementation_sources": dict(implementation_sources),
        "candidate_reference_policy": config.candidate_reference_policy,
        "candidate_spot_day_trade_policy": (
            config.candidate_spot_day_trade_policy
        ),
    }


def _candidate_cache_partition(
    root: Path,
    requirement: CandidateSessionRequirement,
    *,
    config: CrossSessionRunnerConfig,
    implementation_sources: Mapping[str, str],
) -> Path:
    contract = _candidate_cache_contract(
        requirement,
        config=config,
        implementation_sources=implementation_sources,
    )
    key = _canonical_sha256(contract)
    return (
        Path(root)
        / f"Date={requirement.date}"
        / f"ValueCode={requirement.value_code}"
        / f"QuoteCode={requirement.quote_code}"
        / f"Key={key}"
    )


def _verify_candidate_cache_partition(
    requirement: CandidateSessionRequirement,
    partition: Path,
    *,
    config: CrossSessionRunnerConfig,
    implementation_sources: Mapping[str, str],
) -> dict[str, object]:
    expected = _candidate_cache_contract(
        requirement,
        config=config,
        implementation_sources=implementation_sources,
    )
    payload = _verify_candidate_cache_payload(partition, requirement)
    if payload.get("config") != expected or payload.get(
        "config_sha256"
    ) != _canonical_sha256(expected):
        raise ValueError(f"candidate cache config mismatch: {partition}")
    if payload.get("cache_key_sha256") != _canonical_sha256(expected):
        raise ValueError(f"candidate cache key mismatch: {partition}")
    return payload


def _verify_candidate_cache_payload(
    partition: Path,
    requirement: CandidateSessionRequirement,
) -> dict[str, object]:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileExistsError(f"candidate cache is incomplete: {partition}")
    payload = _read_json(marker)
    if payload.get("complete") is not True or payload.get(
        "schema_version"
    ) != CANDIDATE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"candidate cache marker mismatch: {partition}")
    if tuple(
        str(payload.get(name)) for name in ("Date", "ValueCode", "QuoteCode")
    ) != requirement.identity:
        raise ValueError(f"candidate cache identity mismatch: {partition}")
    contract = payload.get("config")
    if not isinstance(contract, dict) or payload.get(
        "config_sha256"
    ) != _canonical_sha256(contract):
        raise ValueError(f"candidate cache embedded config mismatch: {partition}")
    if payload.get("cache_key_sha256") != _canonical_sha256(contract):
        raise ValueError(f"candidate cache embedded key mismatch: {partition}")
    status = payload.get("cache_status")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"candidate cache artifacts are invalid: {partition}")
    if status == "success":
        if set(artifacts) != set(_CANDIDATE_CACHE_ARTIFACTS.values()):
            raise ValueError(f"candidate cache artifact set mismatch: {partition}")
        _verify_artifacts(Path(partition), artifacts)
    elif status == "failure":
        if artifacts or not isinstance(payload.get("failure"), dict):
            raise ValueError(f"candidate failure cache is invalid: {partition}")
    else:
        raise ValueError(f"candidate cache status is invalid: {partition}")
    return payload


def _failure_from_cache_payload(
    payload: Mapping[str, object],
) -> CandidateSessionFailure | None:
    if payload.get("cache_status") != "failure":
        return None
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        raise ValueError("candidate cache failure payload is invalid")
    status = failure.get("status")
    detail = failure.get("detail")
    if not isinstance(status, str) or not status or not isinstance(detail, str):
        raise ValueError("candidate cache failure fields are invalid")
    return CandidateSessionFailure(status=status, detail=detail)


def _publish_candidate_cache_partition(
    requirement: CandidateSessionRequirement,
    value: CrossSessionExitMakerSession | CandidateSessionFailure,
    partition: Path,
    *,
    config: CrossSessionRunnerConfig,
    implementation_sources: Mapping[str, str],
) -> None:
    contract = _candidate_cache_contract(
        requirement,
        config=config,
        implementation_sources=implementation_sources,
    )
    key = _canonical_sha256(contract)
    partition.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=partition.parent)
    )
    try:
        artifacts: dict[str, dict[str, object]] = {}
        failure_payload: dict[str, str] | None = None
        session_payload: dict[str, object] | None = None
        if isinstance(value, CandidateSessionFailure):
            status = "failure"
            failure_payload = asdict(value)
        else:
            status = "success"
            value.validate()
            if (
                value.date,
                str(value.raw_tape.mapping.item(0, "ValueCode")),
                str(value.raw_tape.mapping.item(0, "QuoteCode")),
            ) != requirement.identity:
                raise ValueError("candidate cache session identity mismatch")
            frames = {
                "mapping": value.raw_tape.mapping,
                "spot_states": value.raw_tape.spot_states,
                "future_states": value.raw_tape.future_states,
                "spot_trades": value.raw_tape.spot_trades,
                "future_trades": value.raw_tape.future_trades,
                "raw_audit": value.raw_tape.audit,
                "spread_pair_clock": value.spread_pair_clock,
            }
            for field, filename in _CANDIDATE_CACHE_ARTIFACTS.items():
                frame = frames[field]
                path = stage / filename
                frame.write_parquet(path)
                artifacts[filename] = _artifact_metadata(path, frame)
            session_payload = {
                "spot_ref_price": value.spot_ref_price,
                "future_ref_price": value.future_ref_price,
                "ref_price_source_date": value.ref_price_source_date,
                "ref_price_source_version": value.ref_price_source_version,
                "session_start_time_ns": value.session_start_time_ns,
                "cutoff_cursor": (
                    asdict(value.cutoff_cursor)
                    if value.cutoff_cursor is not None
                    else None
                ),
            }
        marker = {
            "complete": True,
            "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
            "Date": requirement.date,
            "ValueCode": requirement.value_code,
            "QuoteCode": requirement.quote_code,
            "cache_status": status,
            "cache_key_sha256": key,
            "config": contract,
            "config_sha256": key,
            "artifacts": artifacts,
            "session": session_payload,
            "failure": failure_payload,
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if partition.exists():
            raise FileExistsError(partition)
        stage.replace(partition)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _load_candidate_cache_partition(
    requirement: CandidateSessionRequirement,
    partition: Path,
) -> CrossSessionExitMakerSession:
    payload = _verify_candidate_cache_payload(partition, requirement)
    if payload.get("cache_status") != "success":
        raise ValueError("failure cache cannot be loaded as a candidate session")
    session = payload.get("session")
    if not isinstance(session, dict):
        raise ValueError(f"candidate cache session metadata is invalid: {partition}")
    cutoff_payload = session.get("cutoff_cursor")
    cutoff = None
    if cutoff_payload is not None:
        if not isinstance(cutoff_payload, dict):
            raise ValueError("candidate cache cutoff cursor is invalid")
        from .layered import EventCursor

        cutoff = EventCursor(**cutoff_payload)
    frames = {
        field: pl.read_parquet(Path(partition) / filename)
        for field, filename in _CANDIDATE_CACHE_ARTIFACTS.items()
    }
    item = CrossSessionExitMakerSession(
        date=requirement.date,
        raw_tape=RawTapeDay(
            date=requirement.date,
            mapping=frames["mapping"],
            spot_states=frames["spot_states"],
            future_states=frames["future_states"],
            spot_trades=frames["spot_trades"],
            future_trades=frames["future_trades"],
            audit=frames["raw_audit"],
        ),
        spread_pair_clock=frames["spread_pair_clock"],
        spot_ref_price=float(session["spot_ref_price"]),
        future_ref_price=float(session["future_ref_price"]),
        ref_price_source_date=str(session["ref_price_source_date"]),
        ref_price_source_version=str(session["ref_price_source_version"]),
        session_start_time_ns=(
            int(session["session_start_time_ns"])
            if session.get("session_start_time_ns") is not None
            else None
        ),
        cutoff_cursor=cutoff,
    )
    item.validate()
    return item


def _load_candidate_date_batch(
    requirements: Sequence[CandidateSessionRequirement],
    *,
    data_root: Path,
    futures_raw_root: Path,
    contract_metadata_root: Path,
) -> dict[
    tuple[str, str, str],
    CrossSessionExitMakerSession | CandidateSessionFailure,
]:
    """Open the large spot/futures feeds once for one candidate Date."""

    selected = _validate_candidate_requirements(requirements)
    dates = {item.date for item in selected}
    if len(dates) != 1:
        raise ValueError("candidate date batch requires exactly one Date")
    date = next(iter(dates))
    resolved: dict[
        tuple[str, str, str],
        CrossSessionExitMakerSession | CandidateSessionFailure,
    ] = {}
    mappings: dict[tuple[str, str, str], pl.DataFrame] = {}
    day_trade_marks: dict[tuple[str, str, str], str] = {}
    for item in selected:
        try:
            mapping, mark = load_candidate_session_mapping(
                date,
                item.value_code,
                item.quote_code,
                data_root=data_root,
                contract_metadata_root=contract_metadata_root,
            )
            mappings[item.identity] = mapping
            day_trade_marks[item.identity] = mark
        except (FileNotFoundError, ValueError) as error:
            resolved[item.identity] = _candidate_failure(error)
    pending = [item for item in selected if item.identity in mappings]
    if not pending:
        return resolved

    combined_mapping = pl.concat(
        [mappings[item.identity] for item in pending],
        how="vertical_relaxed",
    ).sort(["ValueCode", "QuoteCode"])
    spot_path, future_path = _raw_paths(
        date,
        data_root=data_root,
        futures_raw_root=futures_raw_root,
    )
    try:
        tape = load_raw_tape_day(
            date,
            combined_mapping,
            spot_path=spot_path,
            future_path=future_path,
        )
    except (FileNotFoundError, ValueError) as error:
        failure = _candidate_failure(error)
        for item in pending:
            resolved[item.identity] = failure
        return resolved

    product_tapes = {
        item.identity: _slice_candidate_batch_tape(tape, item)
        for item in pending
    }
    clocks: dict[tuple[str, str, str], pl.DataFrame] = {}
    clock_ready = [
        item
        for item in pending
        if not product_tapes[item.identity].spot_states.is_empty()
    ]
    if clock_ready:
        try:
            combined_clock = load_spread_pair_clock_for_raw_day(
                date,
                [item.value_code for item in clock_ready],
                tape,
                data_root=data_root,
            )
            for item in clock_ready:
                clocks[item.identity] = combined_clock.filter(
                    (pl.col("Date").cast(pl.String) == date)
                    & (
                        pl.col("ValueCode").cast(pl.String)
                        == item.value_code
                    )
                )
        except FileNotFoundError as error:
            failure = _candidate_failure(error)
            for item in clock_ready:
                resolved[item.identity] = failure
        except ValueError:
            # A missing clock for one symbol must not censor every symbol in
            # the date batch.  Isolate only this derived-clock fallback; the
            # large raw spot/futures feeds remain single-scan.
            for item in clock_ready:
                try:
                    clocks[item.identity] = load_spread_pair_clock_for_raw_day(
                        date,
                        [item.value_code],
                        product_tapes[item.identity],
                        data_root=data_root,
                    )
                except (FileNotFoundError, ValueError) as error:
                    resolved[item.identity] = _candidate_failure(error)

    for item in pending:
        if item.identity in resolved:
            continue
        if item.identity not in clocks:
            try:
                clocks[item.identity] = load_spread_pair_clock_for_raw_day(
                    date,
                    [item.value_code],
                    product_tapes[item.identity],
                    data_root=data_root,
                )
            except (FileNotFoundError, ValueError) as error:
                resolved[item.identity] = _candidate_failure(error)
                continue
        row = mappings[item.identity].row(0, named=True)
        session = CrossSessionExitMakerSession(
            date=date,
            raw_tape=product_tapes[item.identity],
            spread_pair_clock=clocks[item.identity],
            spot_ref_price=float(row["spot_ref_price"]),
            future_ref_price=float(row["fut_ref_price"]),
            ref_price_source_date=date,
            ref_price_source_version=(
                "candidate_marketData+exact_contract_metadata_v1;"
                f"day_trade_mark={day_trade_marks[item.identity]}"
            ),
        )
        try:
            session.validate()
        except ValueError as error:
            resolved[item.identity] = _candidate_failure(error)
        else:
            resolved[item.identity] = session
    return resolved


def _slice_candidate_batch_tape(
    tape: RawTapeDay,
    requirement: CandidateSessionRequirement,
) -> RawTapeDay:
    def selected(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        required = {"Date", "ValueCode", "QuoteCode"}
        if not required.issubset(frame.columns):
            raise ValueError("candidate batch frame lacks exact-contract identity")
        return frame.filter(
            (pl.col("Date").cast(pl.String) == requirement.date)
            & (
                pl.col("ValueCode").cast(pl.String)
                == requirement.value_code
            )
            & (
                pl.col("QuoteCode").cast(pl.String)
                == requirement.quote_code
            )
        )

    mapping = tape.mapping.filter(
        (pl.col("ValueCode").cast(pl.String) == requirement.value_code)
        & (pl.col("QuoteCode").cast(pl.String) == requirement.quote_code)
    )
    if mapping.height != 1:
        raise ValueError("candidate batch lost the exact contract mapping")
    audit = tape.audit
    if not audit.is_empty():
        audit = audit.filter(
            (pl.col("ValueCode").cast(pl.String) == requirement.value_code)
            & (pl.col("QuoteCode").cast(pl.String) == requirement.quote_code)
        )
    return RawTapeDay(
        date=requirement.date,
        mapping=mapping,
        spot_states=selected(tape.spot_states),
        future_states=selected(tape.future_states),
        spot_trades=selected(tape.spot_trades),
        future_trades=selected(tape.future_trades),
        audit=audit,
    )


def _candidate_failure(
    error: FileNotFoundError | ValueError,
) -> CandidateSessionFailure:
    detail = str(error)
    if isinstance(error, FileNotFoundError):
        status = "right_censored_missing_raw_session"
    elif "clock" in detail.lower() or "SpreadPair" in detail:
        status = "right_censored_missing_spread_clock"
    elif "reference" in detail.lower() or "metadata" in detail.lower():
        status = "right_censored_missing_candidate_reference"
    else:
        status = "right_censored_invalid_candidate_session"
    return CandidateSessionFailure(status=status, detail=detail)


def load_cross_session_candidate(
    date: str,
    value_code: str,
    exact_quote_code: str,
    *,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    contract_metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
) -> CrossSessionExitMakerSession:
    """Load one exact candidate day without filtering day-trade mark ``N``."""

    mapping, day_trade_mark = load_candidate_session_mapping(
        date,
        value_code,
        exact_quote_code,
        data_root=Path(data_root),
        contract_metadata_root=Path(contract_metadata_root),
    )
    spot_path, future_path = _raw_paths(
        date,
        data_root=Path(data_root),
        futures_raw_root=Path(futures_raw_root),
    )
    if not spot_path.is_file():
        raise FileNotFoundError(spot_path)
    if not future_path.is_file():
        raise FileNotFoundError(future_path)
    tape = load_raw_tape_day(
        date,
        mapping,
        spot_path=spot_path,
        future_path=future_path,
    )
    clock = load_spread_pair_clock_for_raw_day(
        date,
        [value_code],
        tape,
        data_root=Path(data_root),
    )
    row = mapping.row(0, named=True)
    return CrossSessionExitMakerSession(
        date=date,
        raw_tape=tape,
        spread_pair_clock=clock,
        spot_ref_price=float(row["spot_ref_price"]),
        future_ref_price=float(row["fut_ref_price"]),
        ref_price_source_date=date,
        ref_price_source_version=(
            "candidate_marketData+exact_contract_metadata_v1;"
            f"day_trade_mark={day_trade_mark}"
        ),
    )


def load_candidate_session_mapping(
    date: str,
    value_code: str,
    exact_quote_code: str,
    *,
    data_root: Path = HFT_DATA_ROOT,
    contract_metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
) -> tuple[pl.DataFrame, str]:
    """Build candidate-day refs for an existing position, accepting X/Y/N."""

    date = _date(date, "candidate Date")
    contract_path = Path(contract_metadata_root) / f"{date}_contracts.parquet"
    market_path = Path(data_root) / "marketData" / f"{date}_marketData.parquet"
    for path in (contract_path, market_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract_scan = pl.scan_parquet(contract_path)
    _require_scan(
        contract_scan,
        {"ValueCode", "QuoteCode", "fut_ref_price", "contract_size"},
        str(contract_path),
    )
    contract = (
        contract_scan.filter(
            (pl.col("ValueCode").cast(pl.String) == str(value_code))
            & (pl.col("QuoteCode").cast(pl.String) == str(exact_quote_code))
        )
        .select(
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("fut_ref_price").cast(pl.Float64),
            pl.col("contract_size").cast(pl.Float64),
        )
        .collect()
    )
    if contract.height != 1:
        raise ValueError(
            f"{date}: exact futures reference metadata is unavailable for "
            f"{value_code}/{exact_quote_code}"
        )
    market_scan = pl.scan_parquet(market_path)
    _require_scan(
        market_scan,
        {"quote_code", "opening_ref_price", "allow_day_trade_mark"},
        str(market_path),
    )
    # Do not call common.contracts.load_spot_reference here: that function is
    # correctly entry-oriented and filters N.  Exit inventory has a different
    # admissibility contract.
    spot = (
        market_scan.filter(
            pl.col("quote_code").cast(pl.String) == str(value_code)
        )
        .select(
            pl.col("quote_code").cast(pl.String).alias("ValueCode"),
            pl.col("opening_ref_price").cast(pl.Float64).alias(
                "spot_ref_price"
            ),
            pl.col("allow_day_trade_mark").cast(pl.String).alias(
                "day_trade_mark"
            ),
        )
        .collect()
    )
    if spot.height == 0:
        raise ValueError(f"{date}: candidate spot reference is unavailable")
    distinct = spot.select(
        "ValueCode", "spot_ref_price", "day_trade_mark"
    ).unique()
    if distinct.height != 1:
        raise ValueError(f"{date}: conflicting candidate spot reference rows")
    spot_row = distinct.row(0, named=True)
    if not _positive(spot_row["spot_ref_price"]):
        raise ValueError(f"{date}: candidate spot reference is invalid")
    contract_row = contract.row(0, named=True)
    if not _positive(contract_row["fut_ref_price"]) or not _positive(
        contract_row["contract_size"]
    ):
        raise ValueError(f"{date}: candidate futures reference is invalid")
    mark = str(spot_row["day_trade_mark"]).upper()
    if not mark:
        raise ValueError(f"{date}: candidate day-trade mark is missing")
    mapping = pl.DataFrame(
        {
            "ValueCode": [str(value_code)],
            "QuoteCode": [str(exact_quote_code)],
            "spot_ref_price": [float(spot_row["spot_ref_price"])],
            "fut_ref_price": [float(contract_row["fut_ref_price"])],
            "contract_size": [float(contract_row["contract_size"])],
        }
    )
    return mapping, mark


def verify_cross_session_sources(
    date: str,
    value_code: str,
    *,
    entry_execution_root: Path,
    exit_maker_root: Path,
    config: CrossSessionRunnerConfig,
) -> CrossSessionPartitionSource:
    """Verify both markers and every selected upstream artifact hash."""

    entry_partition = (
        Path(entry_execution_root) / f"Date={date}" / f"ValueCode={value_code}"
    )
    exit_partition = (
        Path(exit_maker_root) / f"Date={date}" / f"ValueCode={value_code}"
    )
    entry_payload, entry_marker_hash = _verify_source_marker(
        entry_partition,
        date,
        value_code,
        # ``target_audit`` retains the exact contract identity when a valid
        # product-day has zero action rows.  It is therefore part of the
        # execution-safe source contract, not an optional diagnostic.
        (
            config.action_source_artifact,
            config.exit_rule_source_artifact,
            config.target_audit_source_artifact,
        ),
    )
    exit_payload, exit_marker_hash = _verify_source_marker(
        exit_partition,
        date,
        value_code,
        config.exit_maker_source_artifacts,
    )
    entry_artifacts = entry_payload["artifacts"]
    exit_artifacts = exit_payload["artifacts"]
    assert isinstance(entry_artifacts, dict) and isinstance(exit_artifacts, dict)
    entry_config = entry_payload.get("config")
    exit_config = exit_payload.get("config")
    if not isinstance(entry_config, dict) or not isinstance(exit_config, dict):
        raise ValueError("entry/exit marker lacks embedded config")
    if entry_config.get("hedge_delay_ns") != config.hedge_delay_ns:
        raise ValueError(
            "entry source hedge delay disagrees with cross-session runner"
        )
    exit_runner = exit_config.get("runner")
    if (
        not isinstance(exit_runner, dict)
        or exit_runner.get("hedge_delay_ns") != config.hedge_delay_ns
    ):
        raise ValueError(
            "same-day exit source hedge delay disagrees with cross-session runner"
        )
    hashes = {
        name: str(metadata["sha256"])
        for name, metadata in {**entry_artifacts, **exit_artifacts}.items()
        if isinstance(metadata, dict) and "sha256" in metadata
    }
    # The exit-maker marker already binds the exact action/rule hashes.  Check
    # that relationship again instead of trusting two individually valid but
    # unrelated roots.
    source = exit_config.get("source")
    if not isinstance(source, dict):
        raise ValueError("exit-maker marker lacks upstream source lineage")
    action_source = source.get("action_source")
    rule_source = source.get("exit_rule_source")
    if not isinstance(action_source, dict) or not isinstance(rule_source, dict):
        raise ValueError("exit-maker upstream source lineage is invalid")
    if source.get("upstream_config_sha256") != entry_payload.get(
        "config_sha256"
    ):
        raise ValueError(
            "exit-maker upstream config identity differs from entry partition"
        )
    if action_source.get("sha256") != hashes[config.action_source_artifact]:
        raise ValueError("exit-maker action lineage differs from entry partition")
    if rule_source.get("sha256") != hashes[config.exit_rule_source_artifact]:
        raise ValueError("exit-maker rule lineage differs from entry partition")
    return CrossSessionPartitionSource(
        date=date,
        value_code=value_code,
        entry_partition=entry_partition,
        exit_partition=exit_partition,
        action_path=entry_partition / config.action_source_artifact,
        exit_rule_path=entry_partition / config.exit_rule_source_artifact,
        target_audit_path=(
            entry_partition / config.target_audit_source_artifact
        ),
        exit_paths={
            field: exit_partition / name
            for field, name in _EXIT_ARTIFACTS.items()
        },
        entry_marker_sha256=entry_marker_hash,
        exit_marker_sha256=exit_marker_hash,
        entry_runner_version=str(entry_payload.get("runner_version", "unknown")),
        exit_runner_version=str(exit_payload.get("runner_version", "unknown")),
        entry_config_sha256=str(entry_payload["config_sha256"]),
        exit_config_sha256=str(exit_payload["config_sha256"]),
        artifact_sha256=hashes,
    )


def discover_cross_session_product_days(
    entry_sessions: Sequence[str],
    symbols: Sequence[str],
    *,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
) -> tuple[tuple[tuple[str, str], ...], pl.DataFrame]:
    dates = tuple(str(value) for value in entry_sessions)
    values = tuple(str(value) for value in symbols)
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("entry_sessions must be nonempty and unique")
    if not values or len(values) != len(set(values)):
        raise ValueError("symbols must be nonempty and unique")
    rows: list[dict[str, object]] = []
    keys: list[tuple[str, str]] = []
    for date in dates:
        for value_code in values:
            _validate_partition_key(date, value_code)
            entry_marker = (
                Path(entry_execution_root)
                / f"Date={date}"
                / f"ValueCode={value_code}"
                / "complete.json"
            )
            exit_marker = (
                Path(exit_maker_root)
                / f"Date={date}"
                / f"ValueCode={value_code}"
                / "complete.json"
            )
            available = entry_marker.is_file() and exit_marker.is_file()
            if available:
                keys.append((date, value_code))
            rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "entry_complete": entry_marker.is_file(),
                    "exit_maker_complete": exit_marker.is_file(),
                    "available": available,
                }
            )
    return tuple(keys), pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )


def verify_cross_session_output_partition(
    partition: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_runner_config_sha256: str | None = None,
) -> dict[str, object]:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileExistsError(f"cross-session partition is incomplete: {partition}")
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"cross-session marker is not complete: {partition}")
    if payload.get("runner_version") != CROSS_SESSION_RUNNER_VERSION:
        raise ValueError(f"cross-session runner version mismatch: {partition}")
    if expected_config_sha256 is not None and payload.get(
        "config_sha256"
    ) != expected_config_sha256:
        raise ValueError(f"cross-session partition config mismatch: {partition}")
    config = payload.get("config")
    if not isinstance(config, dict) or _canonical_sha256(config) != payload.get(
        "config_sha256"
    ):
        raise ValueError(f"cross-session embedded config hash mismatch: {partition}")
    embedded_runner_sha, prerequisite = _embedded_runner_binding(
        config,
        payload.get("prerequisite"),  # type: ignore[arg-type]
        source=partition,
    )
    if payload.get("runner_config_sha256") != embedded_runner_sha:
        raise ValueError(
            f"cross-session embedded runner config hash mismatch: {partition}"
        )
    if (
        expected_runner_config_sha256 is not None
        and embedded_runner_sha != expected_runner_config_sha256
    ):
        raise ValueError(f"cross-session runner config mismatch: {partition}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_OUTPUT_ARTIFACTS):
        raise ValueError(f"cross-session artifact set mismatch: {partition}")
    _verify_artifacts(Path(partition), artifacts)
    _verify_output_artifact_schemas(Path(partition), artifacts)
    semantics = payload.get("fact_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"cross-session marker lacks fact semantics: {partition}")
    required_true = (
        "entry_full_fill_and_hedge_only",
        "frozen_threshold_across_sessions",
        "fresh_day_order_and_queue_each_session",
        "exact_quote_code_no_roll",
        "candidate_day_reference_prices",
        "day_trade_mark_n_allowed_for_exit",
        "book_age_diagnostic_only",
        "strict_and_nominal_outputs_separate",
        "old_forced_taker_taker_unchanged",
        "day_policy_replay_disk_spooled",
        "strict_nominal_physical_replay_shared",
        "unpublished_session_support_not_retained",
        "same_day_support_hash_verified_not_loaded",
        "embedded_runner_config_sha256_recomputed",
        "all_prerequisite_identity_copies_consistent",
        "output_and_cache_disjoint_from_raw_source_roots",
        "entry_action_fill_and_hedge_facts_coherent",
        "entry_action_hedge_delay_bound_to_entry_and_exit_markers",
        "exact_session_predecessor_exit_rule_source",
        "complete_center_lower_x_two_exit_route_population",
    )
    if any(semantics.get(name) is not True for name in required_true):
        raise ValueError(f"cross-session marker semantics mismatch: {partition}")
    if semantics.get("formal_prerequisite_bound") is not (
        prerequisite is not None
    ):
        raise ValueError(
            f"cross-session prerequisite semantic mismatch: {partition}"
        )
    return _manifest_row(Path(partition), payload)


def _load_same_day_result(
    source: CrossSessionPartitionSource, *, policy_only: bool = False
) -> ExitMakerProductDayResult:
    if not isinstance(policy_only, bool):
        raise TypeError("policy_only must be boolean")
    if policy_only:
        # Source verification has already checked all seven marker-bound
        # artifacts.  The formal cross runner publishes no same-day support
        # frames, so loading observations/transitions/candidates here only
        # recreates their memory cost without changing a terminal decision.
        empty = pl.DataFrame()
        return ExitMakerProductDayResult(
            policy_support=empty,
            observations=empty,
            transitions=empty,
            candidate_aliases=empty,
            raw_candidate_facts=empty,
            position_policy_facts=pl.read_parquet(
                source.exit_paths["position_policy_facts"]
            ),
            audit=empty,
        )
    return ExitMakerProductDayResult(
        policy_support=pl.read_parquet(source.exit_paths["policy_support"]),
        observations=pl.read_parquet(source.exit_paths["observations"]),
        transitions=pl.read_parquet(source.exit_paths["transitions"]),
        candidate_aliases=pl.read_parquet(source.exit_paths["candidate_aliases"]),
        raw_candidate_facts=pl.read_parquet(
            source.exit_paths["raw_candidate_facts"]
        ),
        position_policy_facts=pl.read_parquet(
            source.exit_paths["position_policy_facts"]
        ),
        audit=pl.read_parquet(source.exit_paths["audit"]),
    )


def _established_entry_count(actions: pl.DataFrame) -> int:
    """Count paired entries without requiring outcome columns on typed empties."""

    if actions.is_empty():
        return 0
    _require(
        actions,
        {
            "full_fill",
            "entry_hedge_label_observed",
            "entry_hedge_executable",
        },
        "nonempty entry actions",
    )
    return actions.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
        & (pl.col("entry_hedge_executable") == True)  # noqa: E712
    ).height


def _crossvalidate_frozen_rule_lineage(
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    policy_support: pl.DataFrame,
    position_policy_facts: pl.DataFrame,
    *,
    origin_date: str,
    expected_source_asof_date: str,
    expected_exit_rule_ids: Sequence[str],
) -> dict[str, object]:
    """Prove that replay thresholds are the frozen, prior-date rule facts.

    ``exit_facts`` also contains target-day outcomes, so its blanket
    ``contains_target_day_outcome`` flag cannot certify the rule fields by
    itself.  This check isolates the immutable rule columns, requires a
    complete rule grid for every entry alias, and cross-checks the established
    subset against the independently materialised same-day maker support.
    """

    origin = _date(origin_date, "origin Date")
    expected_source = _date(
        expected_source_asof_date,
        "expected D-1 exit-rule source",
    )
    if expected_source >= origin:
        raise ValueError("expected exit-rule source must precede origin Date")
    rule_ids = tuple(str(value) for value in expected_exit_rule_ids)
    if not rule_ids or len(rule_ids) != len(set(rule_ids)):
        raise ValueError("expected exit rule ids must be nonempty and unique")

    action_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "policy_generation_id",
        "raw_order_fact_id",
    ]
    minimal_action = {
        *action_key,
        "full_fill",
        "entry_hedge_label_observed",
        "entry_hedge_executable",
    }
    if actions.is_empty():
        if (
            not exits.is_empty()
            or not policy_support.is_empty()
            or not position_policy_facts.is_empty()
        ):
            raise ValueError(
                "empty entry actions require empty exit rules and same-day policies"
            )
        return {
            "entry_action_aliases": 0,
            "established_entry_aliases": 0,
            "frozen_rule_rows": 0,
            "same_day_policy_support_rows": 0,
            "same_day_position_policy_rows": 0,
            "expected_exit_rule_ids": list(rule_ids),
            "expected_exit_routes": list(SUPPORTED_EXIT_MAKER_ROUTES),
            "source_asof_dates": [],
            "expected_source_asof_date": expected_source,
            "exact_session_predecessor": True,
            "lineage_sha256": _canonical_sha256([]),
        }

    _require(actions, minimal_action, "entry actions for frozen-rule lineage")
    if actions.select(action_key).null_count().row(0) != (0,) * len(action_key):
        raise ValueError("entry action lineage keys must be non-null")
    action_aliases = actions.select(action_key).unique(maintain_order=True)
    if action_aliases.height != actions.height:
        raise ValueError("entry action policy aliases are not unique")

    established = (
        actions.filter(
            (pl.col("full_fill") == True)  # noqa: E712
            & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
            & (pl.col("entry_hedge_executable") == True)  # noqa: E712
        )
        .select(action_key)
        .unique(maintain_order=True)
    )
    # Upstream intentionally does not materialise frozen exit-rule facts when
    # there is no paired position to close.  Such partitions remain valid and
    # must avoid both synthetic rules and candidate-day raw I/O.
    no_established = established.is_empty()
    if no_established and (
        not policy_support.is_empty() or not position_policy_facts.is_empty()
    ):
        raise ValueError(
            "same-day policies exist without an established paired entry"
        )
    if no_established and exits.is_empty():
        return {
            "entry_action_aliases": action_aliases.height,
            "established_entry_aliases": 0,
            "frozen_rule_rows": 0,
            "same_day_policy_support_rows": 0,
            "same_day_position_policy_rows": 0,
            "expected_exit_rule_ids": list(rule_ids),
            "expected_exit_routes": list(SUPPORTED_EXIT_MAKER_ROUTES),
            "source_asof_dates": [],
            "expected_source_asof_date": expected_source,
            "exact_session_predecessor": True,
            "lineage_sha256": _canonical_sha256([]),
        }

    rule_columns = {
        *action_key,
        "exit_rule_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
    }
    _require(exits, rule_columns, "frozen exit-rule facts")
    selected_rules = exits.select(sorted(rule_columns))
    if selected_rules.null_count().row(0) != (0,) * selected_rules.width:
        raise ValueError("frozen exit-rule lineage contains null values")
    invalid_threshold = selected_rules.filter(
        (~pl.col("exit_threshold_basis_bp").is_finite())
    )
    if not invalid_threshold.is_empty():
        raise ValueError("frozen exit-rule threshold must be finite")
    invalid_source = selected_rules.filter(
        pl.col("exit_rule_source_asof_date").cast(pl.String)
        != pl.lit(expected_source)
    )
    if not invalid_source.is_empty():
        raise ValueError(
            "frozen exit-rule source must equal the exact session predecessor"
        )

    rule_key = [*action_key, "exit_rule_id"]
    if selected_rules.select(rule_key).n_unique() != selected_rules.height:
        raise ValueError("frozen exit-rule rows are not unique")
    expected_rules = action_aliases.join(
        pl.DataFrame({"exit_rule_id": list(rule_ids)}), how="cross"
    )
    actual_rule_keys = selected_rules.select(rule_key)
    if (
        actual_rule_keys.join(expected_rules, on=rule_key, how="anti").height
        or expected_rules.join(actual_rule_keys, on=rule_key, how="anti").height
    ):
        raise ValueError("frozen exit-rule grid differs from entry aliases")

    if no_established:
        lineage = selected_rules.select(
            *rule_key,
            "exit_threshold_basis_bp",
            "exit_rule_source_asof_date",
        ).sort(rule_key)
        return {
            "entry_action_aliases": action_aliases.height,
            "established_entry_aliases": 0,
            "frozen_rule_rows": lineage.height,
            "same_day_policy_support_rows": 0,
            "same_day_position_policy_rows": 0,
            "expected_exit_rule_ids": list(rule_ids),
            "expected_exit_routes": list(SUPPORTED_EXIT_MAKER_ROUTES),
            "source_asof_dates": sorted(
                str(value)
                for value in lineage["exit_rule_source_asof_date"]
                .unique()
                .to_list()
            ),
            "expected_source_asof_date": expected_source,
            "exact_session_predecessor": True,
            "lineage_sha256": _frame_sha256(lineage),
        }

    support_columns = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
        "exit_route",
        "exit_policy_trial_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "position_status",
        "position_established_ns",
    }
    _require(
        policy_support,
        support_columns,
        "same-day exit-maker frozen-rule support",
    )
    support = policy_support.select(sorted(support_columns))
    if support.null_count().row(0) != (0,) * support.width:
        raise ValueError("same-day frozen-rule support contains null lineage")
    support_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
        "exit_route",
    ]
    if support.select(support_key).n_unique() != support.height:
        raise ValueError("same-day frozen-rule support rows are not unique")
    expected_trial_id = pl.concat_str(
        pl.col("entry_policy_generation_id"),
        pl.lit("/exit/"),
        pl.col("exit_rule_id"),
        pl.lit("/"),
        pl.col("exit_route"),
    )
    if support.filter(
        (pl.col("exit_policy_trial_id") != expected_trial_id).fill_null(True)
        | (pl.col("position_status") != "position_established").fill_null(True)
    ).height:
        raise ValueError(
            "same-day support has an invalid trial identity or position status"
        )
    expected_support = (
        established.rename(
            {
                "route": "entry_route",
                "policy_generation_id": "entry_policy_generation_id",
                "raw_order_fact_id": "entry_raw_order_fact_id",
            }
        )
        .join(pl.DataFrame({"exit_rule_id": list(rule_ids)}), how="cross")
        .join(
            pl.DataFrame(
                {"exit_route": list(SUPPORTED_EXIT_MAKER_ROUTES)}
            ),
            how="cross",
        )
    )
    actual_support_keys = support.select(support_key)
    if (
        actual_support_keys.join(
            expected_support, on=support_key, how="anti"
        ).height
        or expected_support.join(
            actual_support_keys, on=support_key, how="anti"
        ).height
    ):
        raise ValueError(
            "same-day frozen-rule support grid differs from established entries"
        )

    rule_for_support = selected_rules.rename(
        {
            "route": "entry_route",
            "policy_generation_id": "entry_policy_generation_id",
            "raw_order_fact_id": "entry_raw_order_fact_id",
            "exit_threshold_basis_bp": "expected_threshold_basis_bp",
            "exit_rule_source_asof_date": "expected_source_asof_date",
        }
    ).select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
        "expected_threshold_basis_bp",
        "expected_source_asof_date",
    )
    checked = support.join(
        rule_for_support,
        on=[
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "entry_policy_generation_id",
            "entry_raw_order_fact_id",
            "exit_rule_id",
        ],
        how="left",
        validate="m:1",
    )
    mismatch = checked.filter(
        (
            (
                pl.col("exit_threshold_basis_bp")
                - pl.col("expected_threshold_basis_bp")
            ).abs()
            > 1e-12
        )
        | (
            pl.col("exit_rule_source_asof_date")
            != pl.col("expected_source_asof_date")
        )
        | pl.col("expected_threshold_basis_bp").is_null()
        | pl.col("expected_source_asof_date").is_null()
    )
    if not mismatch.is_empty():
        raise ValueError(
            "same-day support threshold/source differs from frozen exit rules"
        )

    expected_cursor = actions.filter(
        (pl.col("full_fill") == True)  # noqa: E712
        & (pl.col("entry_hedge_label_observed") == True)  # noqa: E712
        & (pl.col("entry_hedge_executable") == True)  # noqa: E712
    ).select(
        pl.col("Date"),
        pl.col("ValueCode"),
        pl.col("QuoteCode"),
        pl.col("route").alias("entry_route"),
        pl.col("policy_generation_id").alias(
            "entry_policy_generation_id"
        ),
        pl.col("raw_order_fact_id").alias("entry_raw_order_fact_id"),
        pl.col("entry_hedge_decision_time_ns").alias(
            "_expected_position_established_ns"
        ),
    )
    checked_cursor = support.join(
        expected_cursor,
        on=[
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "entry_policy_generation_id",
            "entry_raw_order_fact_id",
        ],
        how="left",
        validate="m:1",
    )
    if checked_cursor.filter(
        (
            pl.col("position_established_ns")
            != pl.col("_expected_position_established_ns")
        ).fill_null(True)
    ).height:
        raise ValueError(
            "same-day support position cursor differs from entry hedge cursor"
        )

    position_projection_columns = [
        *support_key,
        "exit_policy_trial_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "position_status",
        "position_established_ns",
    ]
    _require(
        position_policy_facts,
        set(position_projection_columns),
        "same-day position policy facts",
    )
    position_projection = position_policy_facts.select(
        position_projection_columns
    )
    if (
        position_projection.select("exit_policy_trial_id").n_unique()
        != position_projection.height
    ):
        raise ValueError("same-day position policy trial IDs are not unique")
    support_projection = support.select(position_projection_columns)
    if (
        support_projection.join(
            position_projection,
            on=position_projection_columns,
            how="anti",
        ).height
        or position_projection.join(
            support_projection,
            on=position_projection_columns,
            how="anti",
        ).height
    ):
        raise ValueError(
            "same-day position facts differ from frozen-rule support lineage"
        )

    lineage = selected_rules.select(
        *rule_key,
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
    ).sort(rule_key)
    source_dates = sorted(
        str(value)
        for value in lineage["exit_rule_source_asof_date"].unique().to_list()
    )
    return {
        "entry_action_aliases": action_aliases.height,
        "established_entry_aliases": established.height,
        "frozen_rule_rows": lineage.height,
        "same_day_policy_support_rows": policy_support.height,
        "same_day_position_policy_rows": position_policy_facts.height,
        "expected_exit_rule_ids": list(rule_ids),
        "expected_exit_routes": list(SUPPORTED_EXIT_MAKER_ROUTES),
        "source_asof_dates": source_dates,
        "expected_source_asof_date": expected_source,
        "exact_session_predecessor": True,
        "lineage_sha256": _frame_sha256(lineage),
    }


def _replace_open_horizon_censor(
    result: CrossSessionExitMakerResult,
    *,
    status: str,
    detail: str,
) -> CrossSessionExitMakerResult:
    open_path = (
        (pl.col("outcome_type") == "censored")
        & (pl.col("filled_entry_outcome_category") == "still_open")
        & (pl.col("outcome_status") == "right_censored_observation_end")
    )
    outcomes = result.policy_outcomes.with_columns(
        pl.when(open_path).then(pl.lit(status)).otherwise(pl.col("outcome_status")).alias("outcome_status"),
        pl.when(open_path).then(pl.lit(status)).otherwise(pl.col("unresolved_reason")).alias("unresolved_reason"),
        pl.when(open_path).then(pl.lit(detail)).otherwise(pl.col("censor_detail")).alias("censor_detail"),
        pl.when(open_path).then(pl.lit("censored")).otherwise(pl.col("filled_entry_outcome_category")).alias("filled_entry_outcome_category"),
        pl.when(open_path).then(pl.lit(status)).otherwise(pl.col("filled_entry_unresolved_reason")).alias("filled_entry_unresolved_reason"),
        pl.when(open_path).then(pl.lit(detail)).otherwise(pl.col("terminal_reason")).alias("terminal_reason"),
    )
    return CrossSessionExitMakerResult(
        outcomes,
        result.session_attempts,
        result.candidate_aliases,
        result.observations,
        result.transitions,
        result.audit,
    )


def _runner_audit(
    date: str,
    value_code: str,
    quote_code: str,
    results: Mapping[str, CrossSessionExitMakerResult],
    *,
    candidate_dates: tuple[str, ...],
    loaded_sessions: tuple[str, ...],
    load_failure_status: str | None,
    load_failure_detail: str | None,
) -> pl.DataFrame:
    records = []
    for semantics in ("strict", "nominal_instant_cancel_v0"):
        result = results[semantics]
        outcomes = result.policy_outcomes
        categories = {
            str(row["filled_entry_outcome_category"]): int(row["len"])
            for row in outcomes.group_by("filled_entry_outcome_category")
            .len()
            .iter_rows(named=True)
        }
        same = outcomes.filter(
            pl.col("terminal_branch") == "same_day_target_exit"
        ).height
        cross = outcomes.filter(
            pl.col("terminal_branch") == "cross_session_maker_exit"
        ).height
        records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "cancel_semantics": semantics,
                "policy_outcome_rows": outcomes.height,
                "session_attempt_rows": result.session_attempts.height,
                "completed_rows": categories.get("completed", 0),
                "same_day_completed_rows": same,
                "cross_session_completed_rows": cross,
                "still_open_rows": categories.get("still_open", 0),
                "censored_rows": categories.get("censored", 0),
                "unknown_rows": categories.get("unknown", 0),
                "candidate_session_count": len(candidate_dates),
                "loaded_candidate_session_count": len(loaded_sessions),
                "loaded_candidate_sessions": list(loaded_sessions),
                "load_failure_status": load_failure_status,
                "load_failure_detail": load_failure_detail,
                "old_forced_taker_taker_used": False,
                "pathwise_ev_ready": False,
            }
        )
    return pl.from_dicts(
        records,
        schema=_RUNNER_AUDIT_SCHEMA,
        strict=True,
    )


def _schema_metadata(
    schema: Mapping[str, pl.DataType],
) -> list[dict[str, str]]:
    return [
        {"name": name, "dtype": str(dtype)}
        for name, dtype in schema.items()
    ]


def _validate_output_frame_schemas(
    frames: Mapping[str, pl.DataFrame],
) -> None:
    for name in _OUTPUT_ARTIFACTS:
        expected = list(_OUTPUT_SCHEMAS[name].items())
        actual = list(frames[name].schema.items())
        if actual != expected:
            raise ValueError(
                f"cross-session output schema mismatch for {name}: "
                f"expected={expected}, actual={actual}"
            )


def _verify_output_artifact_schemas(
    partition: Path,
    artifacts: Mapping[str, object],
) -> None:
    required_metadata = {"rows", "columns", "bytes", "sha256", "schema"}
    for name in _OUTPUT_ARTIFACTS:
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict) or set(metadata) != required_metadata:
            raise ValueError(
                f"cross-session artifact metadata mismatch for {name}: "
                f"{partition}"
            )
        expected_schema = _OUTPUT_SCHEMAS[name]
        if metadata.get("schema") != _schema_metadata(expected_schema):
            raise ValueError(
                f"cross-session declared schema mismatch for {name}: "
                f"{partition}"
            )
        actual = list(pl.read_parquet_schema(partition / name).items())
        expected = list(expected_schema.items())
        if actual != expected:
            raise ValueError(
                f"cross-session artifact schema mismatch for {name}: "
                f"expected={expected}, actual={actual}, partition={partition}"
            )


def _publish_partition(
    partition: Path,
    *,
    date: str,
    value_code: str,
    frames: Mapping[str, pl.DataFrame],
    partition_config: Mapping[str, object],
    config_sha256: str,
    runner_config_sha256: str,
) -> None:
    if set(frames) != set(_OUTPUT_ARTIFACTS):
        raise ValueError("cross-session output frame set is incomplete")
    _validate_output_frame_schemas(frames)
    embedded_runner_sha, prerequisite = _embedded_runner_binding(
        partition_config,
        partition_config.get("prerequisite"),  # type: ignore[arg-type]
        source=partition,
    )
    if embedded_runner_sha != runner_config_sha256:
        raise ValueError(
            f"cross-session embedded runner config hash mismatch: {partition}"
        )
    if _canonical_sha256(partition_config) != config_sha256:
        raise ValueError(
            f"cross-session embedded config hash mismatch: {partition}"
        )
    partition.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=partition.parent)
    )
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name in _OUTPUT_ARTIFACTS:
            frame = frames[name]
            path = stage / name
            frame.write_parquet(path)
            artifacts[name] = {
                **_artifact_metadata(path, frame),
                "schema": _schema_metadata(_OUTPUT_SCHEMAS[name]),
            }
        marker = {
            "complete": True,
            "Date": date,
            "ValueCode": value_code,
            "runner_version": CROSS_SESSION_RUNNER_VERSION,
            "runner_config_sha256": runner_config_sha256,
            "prerequisite": prerequisite,
            "config": dict(partition_config),
            "config_sha256": config_sha256,
            "artifacts": artifacts,
            "fact_semantics": {
                "entry_full_fill_and_hedge_only": True,
                "frozen_threshold_across_sessions": True,
                "fresh_day_order_and_queue_each_session": True,
                "exact_quote_code_no_roll": True,
                "candidate_day_reference_prices": True,
                "day_trade_mark_n_allowed_for_exit": True,
                "book_age_diagnostic_only": True,
                "strict_and_nominal_outputs_separate": True,
                "nominal_instant_cancel_v0_model_assumption": True,
                "pathwise_ev_ready": False,
                "old_forced_taker_taker_unchanged": True,
                "raw_content_sha256_bound": False,
                "candidate_cache_source_restat_before_and_after_read": True,
                "day_policy_replay_disk_spooled": True,
                "strict_nominal_physical_replay_shared": True,
                "unpublished_session_support_not_retained": True,
                "same_day_support_hash_verified_not_loaded": True,
                "embedded_runner_config_sha256_recomputed": True,
                "all_prerequisite_identity_copies_consistent": True,
                "output_and_cache_disjoint_from_raw_source_roots": True,
                "entry_action_fill_and_hedge_facts_coherent": True,
                "entry_action_hedge_delay_bound_to_entry_and_exit_markers": True,
                "exact_session_predecessor_exit_rule_source": True,
                "complete_center_lower_x_two_exit_route_population": True,
                "formal_prerequisite_bound": prerequisite is not None,
            },
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if partition.exists():
            raise FileExistsError(partition)
        stage.replace(partition)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _preflight_output_root(root: Path, runner_sha: str) -> None:
    for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json")):
        verify_cross_session_output_partition(
            marker.parent,
            expected_runner_config_sha256=runner_sha,
        )


def _rebuild_root_manifest(root: Path, runner_sha: str) -> pl.DataFrame:
    rows = [
        verify_cross_session_output_partition(
            marker.parent,
            expected_runner_config_sha256=runner_sha,
        )
        for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json"))
    ]
    manifest = (
        pl.from_dicts(rows, infer_schema_length=None).sort(["Date", "ValueCode"])
        if rows
        else pl.DataFrame()
    )
    if not manifest.is_empty():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".cross_session_manifest.",
            suffix=".tmp.parquet",
            dir=root,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.unlink()
            manifest.write_parquet(temporary)
            temporary.replace(root / CROSS_SESSION_MANIFEST_NAME)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def _manifest_row(partition: Path, payload: Mapping[str, object]) -> dict[str, object]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    prerequisite = _normalise_prerequisite_identity(
        payload.get("prerequisite")  # type: ignore[arg-type]
    )
    return {
        "Date": str(payload["Date"]),
        "ValueCode": str(payload["ValueCode"]),
        "partition": str(partition),
        "runner_config_sha256": str(payload["runner_config_sha256"]),
        "config_sha256": str(payload["config_sha256"]),
        "strict_policy_outcome_rows": int(artifacts[_STRICT_OUTCOMES]["rows"]),
        "nominal_policy_outcome_rows": int(artifacts[_NOMINAL_OUTCOMES]["rows"]),
        "strict_session_attempt_rows": int(artifacts[_STRICT_ATTEMPTS]["rows"]),
        "nominal_session_attempt_rows": int(artifacts[_NOMINAL_ATTEMPTS]["rows"]),
        "audit_rows": int(artifacts[_AUDIT_ARTIFACT]["rows"]),
        "prerequisite_root": (
            str(prerequisite["root"]) if prerequisite is not None else None
        ),
        "prerequisite_marker_sha256": (
            str(prerequisite["marker_sha256"])
            if prerequisite is not None
            else None
        ),
        "prerequisite_marker_payload_sha256": (
            str(prerequisite["marker_payload_sha256"])
            if prerequisite is not None
            else None
        ),
        "prerequisite_config_sha256": (
            str(prerequisite["config_sha256"])
            if prerequisite is not None
            else None
        ),
        "prerequisite_source_identity_sha256": (
            str(prerequisite["source_identity_sha256"])
            if prerequisite is not None
            else None
        ),
        "complete": True,
    }


def _verify_source_marker(
    partition: Path,
    date: str,
    value_code: str,
    required_artifacts: Sequence[str],
) -> tuple[dict[str, object], str]:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"upstream partition is not complete: {marker}")
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"upstream completion marker is false: {marker}")
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"upstream marker identity mismatch: {marker}")
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"upstream marker config is invalid: {marker}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"upstream config hash mismatch: {marker}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"upstream marker artifacts are invalid: {marker}")
    missing = sorted(set(required_artifacts) - set(artifacts))
    if missing:
        raise ValueError(f"upstream marker is missing artifacts: {missing}")
    selected = {name: artifacts[name] for name in required_artifacts}
    _verify_artifacts(Path(partition), selected)
    return payload, _file_sha256(marker)


def _verify_artifacts(partition: Path, artifacts: Mapping[str, object]) -> None:
    for filename, metadata in artifacts.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise ValueError(f"invalid artifact declaration: {partition}/{filename}")
        path = partition / filename
        if not path.is_file() or _file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"artifact hash mismatch or missing: {path}")
        schema = pl.read_parquet_schema(path)
        if metadata.get("columns") is not None and len(schema) != int(
            metadata["columns"]
        ):
            raise ValueError(f"artifact column count mismatch: {path}")
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        if metadata.get("rows") is not None and rows != int(metadata["rows"]):
            raise ValueError(f"artifact row count mismatch: {path}")
        if metadata.get("bytes") is not None and path.stat().st_size != int(
            metadata["bytes"]
        ):
            raise ValueError(f"artifact byte count mismatch: {path}")


def _candidate_dates(
    origin: str,
    quote_code: str,
    sessions: tuple[str, ...],
    index: Mapping[str, int],
    contract_calendar: pl.DataFrame,
) -> tuple[str, ...]:
    required = {"QuoteCode", "expiry_session", "calendar_version"}
    _require(contract_calendar, required, "contract calendar")
    selected = contract_calendar.filter(
        pl.col("QuoteCode").cast(pl.String) == quote_code
    )
    if selected.height != 1:
        raise ValueError("contract calendar must contain one exact QuoteCode row")
    expiry = _date(str(selected.item(0, "expiry_session")), "expiry_session")
    start = int(index[origin]) + 1
    return tuple(value for value in sessions[start:] if value <= expiry)


def _candidate_source_fingerprints(
    dates: tuple[str, ...],
    *,
    data_root: Path,
    futures_raw_root: Path,
    contract_metadata_root: Path,
    custom_loader: bool,
) -> list[dict[str, object]]:
    if custom_loader:
        return [
            {
                "Date": date,
                "source": "injected_candidate_session_loader",
                "content_integrity_bound": False,
            }
            for date in dates
        ]
    rows = []
    for date in dates:
        spot, future = _raw_paths(
            date, data_root=data_root, futures_raw_root=futures_raw_root
        )
        paths = {
            "spot_raw": spot,
            "future_raw": future,
            "spread_clock": data_root / "tickFeature" / f"{date}_tickFeature.parquet",
            "spot_reference": data_root / "marketData" / f"{date}_marketData.parquet",
            "future_reference": contract_metadata_root / f"{date}_contracts.parquet",
        }
        rows.append(
            {
                "Date": date,
                **{name: _stat_fingerprint(path) for name, path in paths.items()},
                "content_integrity_bound": False,
            }
        )
    return rows


def _raw_paths(
    date: str,
    *,
    data_root: Path,
    futures_raw_root: Path,
) -> tuple[Path, Path]:
    return (
        data_root / "tickData" / f"{date}_StockTick.parquet",
        futures_raw_root
        / date[:4]
        / date[4:6]
        / date[6:8]
        / "stock_futures.parquet",
    )


def _stat_fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_quote_code(
    action_path: Path,
    target_audit_path: Path,
    date: str,
    value_code: str,
) -> str:
    frame = pl.read_parquet(
        action_path, columns=["Date", "ValueCode", "QuoteCode"]
    )
    _validate_frame_identity(frame, date, value_code, "entry action identity")
    values = frame["QuoteCode"].drop_nulls().cast(pl.String).unique().to_list()
    if not values:
        audit = pl.read_parquet(
            target_audit_path,
            columns=["Date", "ValueCode", "QuoteCode"],
        )
        _validate_frame_identity(
            audit, date, value_code, "entry target-audit identity"
        )
        values = (
            audit["QuoteCode"].drop_nulls().cast(pl.String).unique().to_list()
        )
    if len(values) != 1:
        raise ValueError(
            "entry source must retain one exact QuoteCode in actions or target audit"
        )
    return str(values[0])


def _validate_frame_identity(
    frame: pl.DataFrame,
    date: str,
    value_code: str,
    source: str,
) -> None:
    _require(frame, {"Date", "ValueCode"}, source)
    if frame.is_empty():
        return
    if set(frame["Date"].cast(pl.String).drop_nulls()) != {date} or set(
        frame["ValueCode"].cast(pl.String).drop_nulls()
    ) != {value_code}:
        raise ValueError(f"{source} identity differs from {date}/{value_code}")


def _normalise_prerequisite_identity(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return one immutable JSON binding for a verified prerequisite root."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("prerequisite_identity must be a mapping or None")
    try:
        normalised = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("prerequisite_identity must be finite JSON") from error
    if not isinstance(normalised, dict):
        raise ValueError("prerequisite_identity must be a JSON object")
    required = {
        "binding_version",
        "root",
        "schema_version",
        "marker_sha256",
        "marker_payload_sha256",
        "config_sha256",
        "source_identity",
        "source_identity_sha256",
    }
    if set(normalised) != required:
        raise ValueError("prerequisite_identity field set mismatch")
    if normalised["binding_version"] != (
        CROSS_SESSION_PREREQUISITE_BINDING_VERSION
    ):
        raise ValueError("prerequisite binding version mismatch")
    root = normalised["root"]
    schema_version = normalised["schema_version"]
    if (
        not isinstance(root, str)
        or not root
        or not Path(root).is_absolute()
        or not isinstance(schema_version, str)
        or not schema_version
    ):
        raise ValueError("prerequisite root/schema identity is invalid")
    for name in (
        "marker_sha256",
        "marker_payload_sha256",
        "config_sha256",
        "source_identity_sha256",
    ):
        digest = normalised[name]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"prerequisite {name} is not a SHA-256 digest")
    source_identity = normalised["source_identity"]
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("prerequisite source_identity is invalid")
    if _canonical_sha256(source_identity) != normalised[
        "source_identity_sha256"
    ]:
        raise ValueError("prerequisite source identity hash mismatch")
    return normalised


def _embedded_runner_binding(
    config: Mapping[str, object],
    marker_prerequisite: Mapping[str, object] | None,
    *,
    source: object,
) -> tuple[str, dict[str, object] | None]:
    """Recompute the embedded runner identity and reconcile every lineage copy."""

    runner = config.get("runner")
    if not isinstance(runner, dict):
        raise ValueError(
            f"cross-session embedded runner config is invalid: {source}"
        )
    global_inputs = runner.get("global_inputs")
    if not isinstance(global_inputs, dict):
        raise ValueError(
            f"cross-session embedded runner global inputs are invalid: {source}"
        )
    prerequisite = _normalise_prerequisite_identity(marker_prerequisite)
    copies = (
        _normalise_prerequisite_identity(
            config.get("prerequisite")  # type: ignore[arg-type]
        ),
        _normalise_prerequisite_identity(
            runner.get("prerequisite_identity")  # type: ignore[arg-type]
        ),
        _normalise_prerequisite_identity(
            global_inputs.get("prerequisite")  # type: ignore[arg-type]
        ),
    )
    if any(copy != prerequisite for copy in copies):
        raise ValueError(
            f"cross-session prerequisite lineage mismatch: {source}"
        )
    return _canonical_sha256(runner), prerequisite


def _runner_payload(
    config: CrossSessionRunnerConfig,
    global_inputs: Mapping[str, object],
) -> dict[str, object]:
    payload = asdict(config)
    payload["global_inputs"] = dict(global_inputs)
    module_root = Path(__file__).parent
    payload["implementation_sources"] = {
        name: _file_sha256(module_root / name)
        for name in (
            "exit_maker_cross_session_runner.py",
            "exit_maker_cross_session.py",
            "exit_maker_report.py",
            "exit_maker_study.py",
            "exit_maker.py",
            "raw_tape.py",
            "execution_runner.py",
            "hedge.py",
            "layered.py",
            "targets.py",
        )
    }
    return payload


def _artifact_metadata(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _normalise_sessions(values: Iterable[str]) -> tuple[str, ...]:
    sessions = tuple(_date(str(value), "session") for value in values)
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("sessions must be nonempty and unique")
    if sessions != tuple(sorted(sessions)):
        raise ValueError("sessions must be ascending")
    return sessions


def _validate_partition_key(date: str, value_code: str) -> None:
    _date(date, "partition Date")
    if not _SAFE_KEY.fullmatch(value_code):
        raise ValueError(f"unsafe ValueCode partition key: {value_code}")


def _validate_distinct_roots(*roots: Path) -> None:
    values = [Path(value).resolve() for value in roots]
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("source, prerequisite, output and cache roots must be disjoint")


def _frame_sha256(frame: pl.DataFrame) -> str:
    columns = sorted(frame.columns)
    selected = frame.select(columns)
    if columns and not selected.is_empty():
        try:
            selected = selected.sort(columns, nulls_last=True)
        except Exception:
            pass
    return hashlib.sha256(selected.write_json().encode()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _require_scan(scan: pl.LazyFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(scan.collect_schema().names()))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _date(value: str, name: str) -> str:
    result = str(value)
    if len(result) != 8 or not result.isdigit():
        raise ValueError(f"{name} must be YYYYMMDD")
    return result
