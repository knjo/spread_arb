"""Atomic, resumable day-batched runner for the exit-maker study.

This runner is deliberately separate from :mod:`execution_runner`.  The
existing execution partitions are immutable inputs: their completion marker
and artifact hashes are verified before the entry and frozen-exit facts are
opened.  Raw spot/futures tape is then loaded once for all pending products
on a date and each product-day study result is published through an atomic
directory rename.

The persisted config hash is product-day specific.  In addition to the
runner policy it binds the exact upstream action-fact and exit-rule hashes,
so resume cannot silently combine a replay with edited entry inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

from .exit_maker_allocator import (
    EXIT_MAKER_ALLOCATOR_ENV_NAME,
    EXIT_MAKER_ALLOCATOR_ENV_VALUE,
    EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE,
    validate_exit_maker_allocator_launch,
)

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from ..quote_width.rolling import DEFAULT_QUANTILES
from .execution_runner import (
    DEFAULT_ROLLING_BOUNDARY_PATH,
    DEFAULT_WALKFORWARD_DAILY_ROOT,
    EXECUTION_RUNNER_VERSION,
    ExecutionRunnerConfig,
    WalkForwardExecutionDayBatch,
    _config_payload as _execution_config_payload,
    load_walkforward_execution_day_batch,
)
from .exit_maker import SUPPORTED_EXIT_MAKER_ROUTES
from .hedge import DEFAULT_HEDGE_DELAY_NS
from .raw_tape import RawTapeDay
from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


EXIT_MAKER_RUNNER_VERSION = (
    "exit_maker_product_day_v4_spooled_decay100ms_launch_guard"
)
DEFAULT_ENTRY_EXECUTION_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "execution_narrow_60d"
)
DEFAULT_EXIT_MAKER_OUTPUT_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "exit_maker_narrow_60d"
)
EXIT_MAKER_MANIFEST_NAME = "exit_maker_partition_manifest.parquet"

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_ACTION_ARTIFACT = "execution_action_facts.parquet"
_EXIT_RULE_ARTIFACT = "exit_facts.parquet"
_RESULT_FRAME_KEYS: tuple[str, ...] = (
    "exit_maker_policy_support",
    "exit_maker_observations",
    "exit_maker_transitions",
    "exit_maker_candidate_aliases",
    "exit_maker_raw_candidate_facts",
    "exit_maker_position_policy_facts",
    "exit_maker_audit",
)


@dataclass(frozen=True)
class ExitMakerRunnerConfig:
    """Frozen replay and lineage semantics written into every partition.

    The strings are intentionally explicit rather than implicit code
    defaults.  A lifecycle, cancellation, OCO, or volume-allocation change
    therefore changes the hash even when timing parameters stay unchanged.
    """

    routes: tuple[str, ...] = tuple(SUPPORTED_EXIT_MAKER_ROUTES)
    boundary_quantiles: tuple[int, ...] = tuple(DEFAULT_QUANTILES)
    exit_rule_ids: tuple[str, ...] = ("frozen_center", "frozen_lower")
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS
    max_book_age_ns: int | None = None
    action_source_artifact: str = _ACTION_ARTIFACT
    exit_rule_source_kind: str = "entry_execution_exit_facts_v1"
    exit_rule_source_artifact: str = _EXIT_RULE_ARTIFACT
    lifecycle_policy_version: str = "exit_maker_layered_oco_v0"
    queue_scenario: str = "displayed_queue_independent_v0"
    cancel_model: str = "nominal_instant_cancel_v0"
    oco_semantics: str = "earliest_full_fill_wins_cancel_request_only_v0"
    allocation_semantics: str = (
        "independent_candidates_no_joint_volume_allocation_v0"
    )
    allocator_env_name: str = EXIT_MAKER_ALLOCATOR_ENV_NAME
    allocator_env_value: str = EXIT_MAKER_ALLOCATOR_ENV_VALUE
    allocator_runtime_value: str = EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE
    runner_version: str = EXIT_MAKER_RUNNER_VERSION

    def validate(self) -> None:
        if tuple(self.routes) != tuple(SUPPORTED_EXIT_MAKER_ROUTES):
            raise ValueError(
                "exit-maker v1 routes must be the two frozen supported routes"
            )
        if (
            not self.boundary_quantiles
            or len(self.boundary_quantiles) != len(set(self.boundary_quantiles))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value >= 100
                for value in self.boundary_quantiles
            )
        ):
            raise ValueError(
                "boundary_quantiles must be unique integers between 0 and 100"
            )
        if (
            not self.exit_rule_ids
            or len(self.exit_rule_ids) != len(set(self.exit_rule_ids))
            or any(
                not isinstance(value, str) or not value
                for value in self.exit_rule_ids
            )
        ):
            raise ValueError("exit_rule_ids must be nonempty and unique")
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
        for name in (
            "action_source_artifact",
            "exit_rule_source_kind",
            "exit_rule_source_artifact",
            "lifecycle_policy_version",
            "queue_scenario",
            "cancel_model",
            "oco_semantics",
            "allocation_semantics",
            "allocator_env_name",
            "allocator_env_value",
            "allocator_runtime_value",
            "runner_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if Path(self.action_source_artifact).name != self.action_source_artifact:
            raise ValueError("action_source_artifact must be a plain filename")
        if (
            Path(self.exit_rule_source_artifact).name
            != self.exit_rule_source_artifact
        ):
            raise ValueError("exit_rule_source_artifact must be a plain filename")
        if self.cancel_model != "nominal_instant_cancel_v0":
            raise ValueError("exit-maker v1 supports nominal_instant_cancel_v0 only")
        if (
            self.allocator_env_name != EXIT_MAKER_ALLOCATOR_ENV_NAME
            or self.allocator_env_value != EXIT_MAKER_ALLOCATOR_ENV_VALUE
            or self.allocator_runtime_value
            != EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE
        ):
            raise ValueError(
                "exit-maker allocator contract must use the frozen jemalloc "
                "100ms decay configuration"
            )
        if self.runner_version != EXIT_MAKER_RUNNER_VERSION:
            raise ValueError(
                f"runner_version must be {EXIT_MAKER_RUNNER_VERSION!r}"
            )

    def study_config(self):
        """Build the study-layer config without duplicating replay logic."""

        from .exit_maker_study import ExitMakerStudyConfig

        return ExitMakerStudyConfig(
            hedge_delay_ns=self.hedge_delay_ns,
            max_book_age_ns=self.max_book_age_ns,
            expected_exit_rule_ids=self.exit_rule_ids,
            exit_lifecycle_policy_version=self.lifecycle_policy_version,
            exit_queue_scenario=self.queue_scenario,
            instant_cancel_v0=self.cancel_model == "nominal_instant_cancel_v0",
        )


@dataclass(frozen=True)
class EntryPartitionSource:
    """Verified upstream files and hashes used by one product-day."""

    date: str
    value_code: str
    partition: Path
    action_path: Path
    exit_rule_path: Path
    upstream_runner_version: str
    upstream_config_sha256: str
    action_sha256: str
    exit_rule_sha256: str

    def config_payload(self, config: ExitMakerRunnerConfig) -> dict[str, object]:
        return {
            "Date": self.date,
            "ValueCode": self.value_code,
            "upstream_runner_version": self.upstream_runner_version,
            "upstream_config_sha256": self.upstream_config_sha256,
            "action_source": {
                "artifact": config.action_source_artifact,
                "sha256": self.action_sha256,
            },
            "exit_rule_source": {
                "kind": config.exit_rule_source_kind,
                "artifact": config.exit_rule_source_artifact,
                "sha256": self.exit_rule_sha256,
            },
        }


DayBatchLoader = Callable[..., WalkForwardExecutionDayBatch]
ProductDayReplayer = Callable[..., object]


def run_exit_maker_replay(
    product_days: Iterable[tuple[str, str]],
    *,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    output_root: Path = DEFAULT_EXIT_MAKER_OUTPUT_ROOT,
    config: ExitMakerRunnerConfig = ExitMakerRunnerConfig(),
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_snapshot_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    data_root: Path = HFT_DATA_ROOT,
    resume: bool = True,
    day_batch_loader: DayBatchLoader | None = None,
    product_day_replayer: ProductDayReplayer | None = None,
) -> pl.DataFrame:
    """Replay and atomically publish selected exit-maker product-days.

    Complete products are verified before deciding which symbols still need
    the day's raw tape.  Consequently a fully resumed day performs no raw
    I/O, while all pending symbols on a date share one normalized tape load.
    """

    config.validate()
    validate_exit_maker_allocator_launch()
    _validate_allocator_runtime(config)
    keys = [(str(date), str(value_code)) for date, value_code in product_days]
    if not keys:
        raise ValueError("product_days must be nonempty")
    if len(keys) != len(set(keys)):
        raise ValueError("product_days contains duplicate keys")
    for date, value_code in keys:
        _validate_partition_key(date, value_code)

    entry_root = Path(entry_execution_root)
    root = Path(output_root)
    _validate_disjoint_roots(entry_root, root)
    root.mkdir(parents=True, exist_ok=True)

    runner_payload = _runner_config_payload(config)
    runner_config_sha256 = _canonical_sha256(runner_payload)
    _preflight_output_root(root, runner_config_sha256)

    by_date: dict[str, list[str]] = {}
    for date, value_code in keys:
        by_date.setdefault(date, []).append(value_code)

    batch_loader = day_batch_loader or load_walkforward_execution_day_batch
    artifact_replayer: Callable[..., object] | None = None
    if product_day_replayer is None:
        from .exit_maker_study import (
            replay_exit_maker_product_day_to_artifacts,
        )

        artifact_replayer = replay_exit_maker_product_day_to_artifacts
    study_config = config.study_config()

    for date, value_codes in by_date.items():
        pending: list[tuple[str, EntryPartitionSource, dict[str, object], str]] = []
        for value_code in value_codes:
            source = verify_entry_partition(
                entry_root / f"Date={date}" / f"ValueCode={value_code}",
                date=date,
                value_code=value_code,
                config=config,
            )
            partition_payload = {
                "runner": runner_payload,
                "source": source.config_payload(config),
            }
            config_sha256 = _canonical_sha256(partition_payload)
            partition = root / f"Date={date}" / f"ValueCode={value_code}"
            if partition.exists():
                if not resume:
                    raise FileExistsError(partition)
                _verify_output_partition(
                    partition,
                    expected_config_sha256=config_sha256,
                    expected_runner_config_sha256=runner_config_sha256,
                )
            else:
                pending.append(
                    (value_code, source, partition_payload, config_sha256)
                )

        if not pending:
            continue
        pending_codes = tuple(item[0] for item in pending)
        batch = batch_loader(
            date,
            pending_codes,
            daily_root=Path(daily_root),
            boundary_snapshot_path=Path(boundary_snapshot_path),
            data_root=Path(data_root),
            quantiles=config.boundary_quantiles,
        )
        _validate_day_batch(batch, date, pending_codes, config.boundary_quantiles)

        for value_code, source, partition_payload, config_sha256 in pending:
            raw_tape = _slice_raw_tape_product(batch.raw_tape, value_code)
            spread_clock = _slice_product_frame(
                batch.spot_feature_state,
                value_code,
                source="spread-pair clock",
            )
            action_facts = pl.read_parquet(source.action_path)
            exit_facts = pl.read_parquet(source.exit_rule_path)
            _validate_input_frame_identity(
                action_facts, date, value_code, "execution action facts"
            )
            _validate_input_frame_identity(
                exit_facts, date, value_code, "exit-rule facts"
            )
            partition = root / f"Date={date}" / f"ValueCode={value_code}"
            if artifact_replayer is not None:
                _replay_and_publish_artifact_partition(
                    partition,
                    date=date,
                    value_code=value_code,
                    action_facts=action_facts,
                    exit_facts=exit_facts,
                    raw_tape=raw_tape,
                    spread_clock=spread_clock,
                    study_config=study_config,
                    artifact_replayer=artifact_replayer,
                    partition_config=partition_payload,
                    config_sha256=config_sha256,
                    runner_config_sha256=runner_config_sha256,
                )
            else:
                assert product_day_replayer is not None
                result = product_day_replayer(
                    action_facts,
                    exit_facts,
                    raw_tape,
                    spread_clock,
                    study_config,
                )
                frames = _result_frames(result)
                _publish_output_partition(
                    partition,
                    date=date,
                    value_code=value_code,
                    frames=frames,
                    partition_config=partition_payload,
                    config_sha256=config_sha256,
                    runner_config_sha256=runner_config_sha256,
                )
                del result, frames
            del raw_tape, spread_clock, action_facts, exit_facts
        del batch

    return _rebuild_root_manifest(root, runner_config_sha256)


def _validate_allocator_runtime(config: ExitMakerRunnerConfig) -> None:
    """Fail before output or raw I/O unless jemalloc decay is process-bound."""

    observed = os.environ.get(config.allocator_env_name)
    if observed != config.allocator_runtime_value:
        raise RuntimeError(
            "exit-maker requires the process-start allocator environment "
            f"{config.allocator_env_name}={config.allocator_env_value!r}; "
            f"expected Polars runtime value {config.allocator_runtime_value!r}, "
            f"observed {observed!r}"
        )


def verify_entry_partition(
    partition: Path,
    *,
    date: str,
    value_code: str,
    config: ExitMakerRunnerConfig,
) -> EntryPartitionSource:
    """Verify the immutable entry partition before trusting rule inputs."""

    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"entry partition is not complete: {marker}")
    payload = _read_json_object(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"entry completion marker is not complete: {marker}")
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"entry completion marker identity mismatch: {marker}")
    upstream_config = payload.get("config")
    upstream_sha = payload.get("config_sha256")
    if not isinstance(upstream_config, dict) or not isinstance(upstream_sha, str):
        raise ValueError(f"entry completion marker has invalid config: {marker}")
    if _canonical_sha256(upstream_config) != upstream_sha:
        raise ValueError(f"entry config hash mismatch: {marker}")
    expected_upstream_config = _expected_entry_config_payload()
    expected_upstream_sha = _canonical_sha256(expected_upstream_config)
    if payload.get("runner_version") != EXECUTION_RUNNER_VERSION:
        raise ValueError(
            f"entry runner version is not current {EXECUTION_RUNNER_VERSION}: "
            f"{marker}"
        )
    if (
        upstream_config != expected_upstream_config
        or upstream_sha != expected_upstream_sha
    ):
        raise ValueError(f"entry config is not the current v3 ladder contract: {marker}")
    if (
        upstream_config.get("price_ladder_version") != PRICE_LADDER_VERSION
        or upstream_config.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        or upstream_config.get("runner_version") != EXECUTION_RUNNER_VERSION
    ):
        raise ValueError(f"entry config ladder lineage mismatch: {marker}")
    fact_semantics = payload.get("fact_semantics")
    if not isinstance(fact_semantics, dict) or (
        fact_semantics.get("price_ladder_version") != PRICE_LADDER_VERSION
        or fact_semantics.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        raise ValueError(f"entry fact-semantics ladder lineage mismatch: {marker}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"entry completion marker has invalid artifacts: {marker}")
    _verify_declared_artifacts(Path(partition), artifacts)
    required = (config.action_source_artifact, config.exit_rule_source_artifact)
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(f"entry partition is missing required artifacts: {missing}")
    action_meta = artifacts[config.action_source_artifact]
    exit_meta = artifacts[config.exit_rule_source_artifact]
    assert isinstance(action_meta, dict) and isinstance(exit_meta, dict)
    return EntryPartitionSource(
        date=date,
        value_code=value_code,
        partition=Path(partition),
        action_path=Path(partition) / config.action_source_artifact,
        exit_rule_path=Path(partition) / config.exit_rule_source_artifact,
        upstream_runner_version=str(payload.get("runner_version", "unknown")),
        upstream_config_sha256=upstream_sha,
        action_sha256=str(action_meta["sha256"]),
        exit_rule_sha256=str(exit_meta["sha256"]),
    )


def discover_entry_product_days(
    sessions: Sequence[str],
    symbols: Sequence[str],
    *,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
) -> tuple[tuple[tuple[str, str], ...], pl.DataFrame]:
    """Return complete entry partitions and an explicit requested-grid audit."""

    root = Path(entry_execution_root)
    session_values = tuple(str(value) for value in sessions)
    symbol_values = tuple(str(value) for value in symbols)
    if not session_values or len(session_values) != len(set(session_values)):
        raise ValueError("sessions must be nonempty and unique")
    if not symbol_values or len(symbol_values) != len(set(symbol_values)):
        raise ValueError("symbols must be nonempty and unique")
    rows: list[dict[str, object]] = []
    keys: list[tuple[str, str]] = []
    for date in session_values:
        for value_code in symbol_values:
            _validate_partition_key(date, value_code)
            partition = root / f"Date={date}" / f"ValueCode={value_code}"
            marker = partition / "complete.json"
            if partition.exists() and not marker.is_file():
                raise FileExistsError(
                    f"entry partition exists without complete.json: {partition}"
                )
            available = marker.is_file()
            if available:
                keys.append((date, value_code))
            rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "requested": True,
                    "available_entry_partition": available,
                    "availability_status": (
                        "available" if available else "missing_entry_partition"
                    ),
                }
            )
    audit = pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )
    return tuple(keys), audit


def _runner_config_payload(config: ExitMakerRunnerConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["routes"] = list(config.routes)
    payload["boundary_quantiles"] = list(config.boundary_quantiles)
    payload["exit_rule_ids"] = list(config.exit_rule_ids)
    module_root = Path(__file__).parent
    payload["implementation_sources"] = {
        name: _file_sha256(module_root / name)
        for name in (
            "execution_runner.py",
            "execution_facts.py",
            "exit_maker_allocator.py",
            "exit_maker_runner.py",
            "exit_maker.py",
            "exit_maker_study.py",
            "hedge.py",
            "targets.py",
        )
    }
    expected_entry_config = _expected_entry_config_payload()
    payload["upstream_entry_contract"] = {
        "runner_version": EXECUTION_RUNNER_VERSION,
        "config_sha256": _canonical_sha256(expected_entry_config),
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
        "execution_runner_source_sha256": payload["implementation_sources"][
            "execution_runner.py"
        ],
        "execution_facts_source_sha256": payload["implementation_sources"][
            "execution_facts.py"
        ],
        "exit_maker_runner_source_sha256": payload["implementation_sources"][
            "exit_maker_runner.py"
        ],
        "targets_source_sha256": payload["implementation_sources"]["targets.py"],
    }
    return payload


def _expected_entry_config_payload() -> dict[str, object]:
    config = ExecutionRunnerConfig()
    config.validate()
    return _execution_config_payload(config)


def _result_frames(result: object) -> dict[str, pl.DataFrame]:
    frame_method = getattr(result, "frames", None)
    if not callable(frame_method):
        raise TypeError("exit-maker result must expose a callable frames()")
    materialized = frame_method()
    if not isinstance(materialized, Mapping):
        raise TypeError("exit-maker result frames() must return a mapping")
    if set(materialized) != set(_RESULT_FRAME_KEYS):
        raise ValueError("exit-maker result frames() returned an unexpected schema")
    frames: dict[str, pl.DataFrame] = {}
    for key in _RESULT_FRAME_KEYS:
        value = materialized[key]
        if not isinstance(value, pl.DataFrame):
            raise TypeError(
                f"exit-maker result frame {key!r} must be a polars DataFrame"
            )
        frames[f"{key}.parquet"] = value
    return frames


def _publish_output_partition(
    partition: Path,
    *,
    date: str,
    value_code: str,
    frames: Mapping[str, pl.DataFrame],
    partition_config: Mapping[str, object],
    config_sha256: str,
    runner_config_sha256: str,
) -> dict[str, object]:
    parent = partition.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=parent))
    committed = False
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = stage / filename
            frame.write_parquet(path)
            artifacts[filename] = {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        complete = _commit_output_stage(
            stage,
            partition,
            date=date,
            value_code=value_code,
            artifacts=artifacts,
            partition_config=partition_config,
            config_sha256=config_sha256,
            runner_config_sha256=runner_config_sha256,
        )
        committed = True
    finally:
        if not committed:
            shutil.rmtree(stage, ignore_errors=True)
    return _manifest_row(partition, complete)


def _replay_and_publish_artifact_partition(
    partition: Path,
    *,
    date: str,
    value_code: str,
    action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    raw_tape: RawTapeDay,
    spread_clock: pl.DataFrame,
    study_config: object,
    artifact_replayer: Callable[..., object],
    partition_config: Mapping[str, object],
    config_sha256: str,
    runner_config_sha256: str,
) -> dict[str, object]:
    """Replay directly inside the atomic stage and publish marker last."""

    parent = partition.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=parent))
    committed = False
    try:
        result = artifact_replayer(
            action_facts,
            exit_facts,
            raw_tape,
            spread_clock,
            study_config,
            artifact_directory=stage,
        )
        paths = _result_artifact_paths(result, stage)
        artifacts = {
            path.name: _parquet_artifact_metadata(path)
            for path in paths.values()
        }
        complete = _commit_output_stage(
            stage,
            partition,
            date=date,
            value_code=value_code,
            artifacts=artifacts,
            partition_config=partition_config,
            config_sha256=config_sha256,
            runner_config_sha256=runner_config_sha256,
        )
        committed = True
    finally:
        if not committed:
            shutil.rmtree(stage, ignore_errors=True)
    return _manifest_row(partition, complete)


def _result_artifact_paths(
    result: object,
    stage: Path,
) -> dict[str, Path]:
    path_method = getattr(result, "artifact_paths", None)
    if not callable(path_method):
        raise TypeError(
            "spooled exit-maker result must expose callable artifact_paths()"
        )
    materialized = path_method()
    if not isinstance(materialized, Mapping):
        raise TypeError("exit-maker artifact_paths() must return a mapping")
    if set(materialized) != set(_RESULT_FRAME_KEYS):
        raise ValueError(
            "exit-maker artifact_paths() returned an unexpected schema"
        )
    expected_paths = {
        key: stage / f"{key}.parquet" for key in _RESULT_FRAME_KEYS
    }
    for key, expected in expected_paths.items():
        observed = Path(materialized[key])
        if observed != expected or not observed.is_file():
            raise ValueError(f"invalid spooled exit-maker artifact: {observed}")
    if set(stage.iterdir()) != set(expected_paths.values()):
        raise ValueError("spooled exit-maker stage contains unexpected files")
    return expected_paths


def _parquet_artifact_metadata(path: Path) -> dict[str, object]:
    schema = pl.read_parquet_schema(path)
    rows = int(
        pl.scan_parquet(path)
        .select(pl.len().alias("rows"))
        .collect(engine="streaming")
        .item()
    )
    return {
        "rows": rows,
        "columns": len(schema),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _commit_output_stage(
    stage: Path,
    partition: Path,
    *,
    date: str,
    value_code: str,
    artifacts: Mapping[str, Mapping[str, object]],
    partition_config: Mapping[str, object],
    config_sha256: str,
    runner_config_sha256: str,
) -> dict[str, object]:
    complete = {
        "complete": True,
        "Date": date,
        "ValueCode": value_code,
        "runner_version": EXIT_MAKER_RUNNER_VERSION,
        "runner_config_sha256": runner_config_sha256,
        "config": partition_config,
        "config_sha256": config_sha256,
        "artifacts": dict(artifacts),
        "fact_semantics": {
            "exit_rule_source_hash_bound": True,
            "entry_action_source_hash_bound": True,
            "raw_tape_batch_unit": "selected_date",
            "old_entry_partition_mutated": False,
            "pathwise_ev_ready": False,
        },
    }
    (stage / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if partition.exists():
        raise FileExistsError(partition)
    stage.replace(partition)
    return complete


def _verify_output_partition(
    partition: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_runner_config_sha256: str | None = None,
) -> dict[str, object]:
    marker = partition / "complete.json"
    if not marker.is_file():
        raise FileExistsError(
            "existing exit-maker partition is incomplete and will not be "
            f"overwritten: {partition}"
        )
    payload = _read_json_object(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"exit-maker marker is not complete: {partition}")
    expected_date = partition.parent.name.removeprefix("Date=")
    expected_value_code = partition.name.removeprefix("ValueCode=")
    if (
        payload.get("Date") != expected_date
        or payload.get("ValueCode") != expected_value_code
    ):
        raise ValueError(f"exit-maker partition identity mismatch: {partition}")
    if payload.get("runner_version") != EXIT_MAKER_RUNNER_VERSION:
        raise ValueError(f"exit-maker runner version mismatch: {partition}")
    if (
        expected_config_sha256 is not None
        and payload.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError(f"exit-maker partition config mismatch: {partition}")
    if (
        expected_runner_config_sha256 is not None
        and payload.get("runner_config_sha256")
        != expected_runner_config_sha256
    ):
        raise ValueError(f"exit-maker runner config mismatch: {partition}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"exit-maker partition config is invalid: {partition}")
    if _canonical_sha256(config_payload) != payload.get("config_sha256"):
        raise ValueError(f"exit-maker partition config hash mismatch: {partition}")
    embedded_runner = config_payload.get("runner")
    if not isinstance(embedded_runner, dict):
        raise ValueError(
            f"exit-maker embedded runner config is invalid: {partition}"
        )
    if _canonical_sha256(embedded_runner) != payload.get(
        "runner_config_sha256"
    ):
        raise ValueError(
            f"exit-maker embedded runner config hash mismatch: {partition}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"exit-maker artifact manifest is invalid: {partition}")
    expected_names = {f"{key}.parquet" for key in _RESULT_FRAME_KEYS}
    if set(artifacts) != expected_names:
        raise ValueError(f"exit-maker artifact set mismatch: {partition}")
    _verify_declared_artifacts(partition, artifacts)
    return _manifest_row(partition, payload)


def _preflight_output_root(root: Path, runner_config_sha256: str) -> None:
    """Reject mixed semantics before publishing any new partition."""

    for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json")):
        _verify_output_partition(
            marker.parent,
            expected_runner_config_sha256=runner_config_sha256,
        )


def _rebuild_root_manifest(
    root: Path, runner_config_sha256: str
) -> pl.DataFrame:
    rows = [
        _verify_output_partition(
            marker.parent,
            expected_runner_config_sha256=runner_config_sha256,
        )
        for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json"))
    ]
    manifest = (
        pl.from_dicts(rows, infer_schema_length=None).sort(["Date", "ValueCode"])
        if rows
        else pl.DataFrame()
    )
    if manifest.is_empty():
        return manifest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".exit_maker_partition_manifest.",
        suffix=".tmp.parquet",
        dir=root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        manifest.write_parquet(temporary)
        temporary.replace(root / EXIT_MAKER_MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _manifest_row(
    partition: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"invalid artifact manifest: {partition}")
    counts = {
        filename.removesuffix(".parquet") + "_rows": int(metadata["rows"])
        for filename, metadata in artifacts.items()
        if isinstance(metadata, dict)
    }
    config_payload = payload.get("config")
    source_payload = (
        config_payload.get("source")
        if isinstance(config_payload, dict)
        else None
    )
    action_source = (
        source_payload.get("action_source")
        if isinstance(source_payload, dict)
        else None
    )
    exit_rule_source = (
        source_payload.get("exit_rule_source")
        if isinstance(source_payload, dict)
        else None
    )
    return {
        "Date": str(payload["Date"]),
        "ValueCode": str(payload["ValueCode"]),
        "partition": str(partition),
        "runner_config_sha256": str(payload["runner_config_sha256"]),
        "config_sha256": str(payload["config_sha256"]),
        "action_source_sha256": (
            str(action_source.get("sha256"))
            if isinstance(action_source, dict)
            else None
        ),
        "exit_rule_source_kind": (
            str(exit_rule_source.get("kind"))
            if isinstance(exit_rule_source, dict)
            else None
        ),
        "exit_rule_source_sha256": (
            str(exit_rule_source.get("sha256"))
            if isinstance(exit_rule_source, dict)
            else None
        ),
        "complete": True,
        **counts,
    }


def _verify_declared_artifacts(
    partition: Path, artifacts: Mapping[str, object]
) -> None:
    for filename, metadata in artifacts.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise ValueError(f"invalid artifact declaration: {partition}/{filename}")
        path = partition / filename
        expected_hash = metadata.get("sha256")
        if not path.is_file() or not isinstance(expected_hash, str):
            raise ValueError(f"partition artifact is missing: {path}")
        for field in ("rows", "columns", "bytes"):
            value = metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"partition artifact metadata is invalid: {path}"
                )
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"partition artifact byte size mismatch: {path}")
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"partition artifact hash mismatch: {path}")


def _slice_raw_tape_product(tape: RawTapeDay, value_code: str) -> RawTapeDay:
    return RawTapeDay(
        date=str(tape.date),
        mapping=_slice_product_frame(tape.mapping, value_code, source="raw mapping"),
        spot_states=_slice_product_frame(
            tape.spot_states, value_code, source="raw spot states"
        ),
        future_states=_slice_product_frame(
            tape.future_states, value_code, source="raw future states"
        ),
        spot_trades=_slice_product_frame(
            tape.spot_trades,
            value_code,
            source="raw spot trades",
            allow_empty=True,
        ),
        future_trades=_slice_product_frame(
            tape.future_trades,
            value_code,
            source="raw future trades",
            allow_empty=True,
        ),
        audit=_slice_product_frame(tape.audit, value_code, source="raw tape audit"),
    )


def _slice_product_frame(
    frame: pl.DataFrame,
    value_code: str,
    *,
    source: str,
    allow_empty: bool = False,
) -> pl.DataFrame:
    if "ValueCode" not in frame.columns:
        raise ValueError(f"{source} is missing ValueCode")
    selected = frame.filter(pl.col("ValueCode").cast(pl.String) == value_code)
    if selected.is_empty() and not allow_empty:
        raise ValueError(f"{source} has no rows for {value_code}")
    return selected


def _validate_day_batch(
    batch: WalkForwardExecutionDayBatch,
    date: str,
    value_codes: tuple[str, ...],
    quantiles: tuple[int, ...],
) -> None:
    if not isinstance(batch, WalkForwardExecutionDayBatch):
        raise TypeError("day_batch_loader must return WalkForwardExecutionDayBatch")
    if str(batch.date) != date or tuple(batch.value_codes) != value_codes:
        raise ValueError("day-batch loader returned different date or products")
    if tuple(batch.boundary_quantiles) != tuple(quantiles):
        raise ValueError("day-batch loader returned different boundary quantiles")


def _validate_input_frame_identity(
    frame: pl.DataFrame, date: str, value_code: str, source: str
) -> None:
    required = {"Date", "ValueCode"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing identity columns: {sorted(missing)}")
    if frame.is_empty():
        return
    dates = set(frame["Date"].cast(pl.String).drop_nulls().to_list())
    values = set(frame["ValueCode"].cast(pl.String).drop_nulls().to_list())
    if dates != {date} or values != {value_code}:
        raise ValueError(f"{source} identity does not match {date}/{value_code}")


def _validate_partition_key(date: str, value_code: str) -> None:
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"invalid YYYYMMDD partition date: {date}")
    if not _SAFE_KEY.fullmatch(value_code):
        raise ValueError(f"unsafe ValueCode partition key: {value_code}")


def _validate_disjoint_roots(entry_root: Path, output_root: Path) -> None:
    source = entry_root.resolve()
    destination = output_root.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("entry and exit-maker output roots must be disjoint")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON marker: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON marker must contain an object: {path}")
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
