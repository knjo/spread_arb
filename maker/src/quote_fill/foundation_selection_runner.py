"""Staged orchestration for the rebuilt S0.5 foundation selection study.

This runner builds only the frozen, development-only foundation-selection
study.  It never reuses the legacy boundary table or its q-dependent gates.
The dependency chain is::

    preflight -> anchor -> entry episodes -> boundary -> convergence -> publish

All input dates come from one frozen, content-hashed 131-session calendar.  A
canonical call additionally requires one clean committed source revision.
Every date-local or phase-level checkpoint is atomic and its fingerprint binds
the registry, source commit, source files, and phase parameters.  Anchor,
episode, and convergence work resumes per Date.  The boundary orchestration
API currently returns only a complete 131-session panel, so its base and
incremental-TOD checkpoints resume at phase granularity and print that limit
before starting.  Resume always verifies content instead of trusting that a
directory merely exists.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from ..common.paths import MAKER_ROOT
from .foundation_anchor_selection import (
    ACTIONABLE_MODEL_IDS,
    ANCHOR_CANDIDATES,
    FUTURE_HORIZONS,
    MODEL_COLUMNS,
    evaluate_anchor_selection_day,
    materialize_anchor_column,
)
from .foundation_boundary_selection import (
    DEFAULT_TOD_BUCKETS,
    annotate_expiry_dte,
    extract_entry_window_excursions,
)
from .foundation_cohort_selection import (
    SPOT_BID_ROUTE,
    build_anchor_product_day_support,
    build_s1_mother_cohort,
)
from .foundation_revalidation_runner import (
    DEFAULT_DAILY_ROOT,
    DEFAULT_LIQUIDITY_PATH,
    DEFAULT_SESSIONS_PATH,
    EXPECTED_PRIMARY_SESSION_COUNT,
    EXPECTED_SESSION_COUNT,
    PRIMARY_START_DATE,
    PROTECTED_FORWARD_START_DATE,
    SOURCE_END_DATE,
    SOURCE_START_DATE,
    _artifact_metadata,
    _canonical_sha256,
    _git_state,
    _sha256_file,
    _write_frame,
    _write_json,
    load_frozen_sessions,
    validate_named_daily_partitions,
)
from .foundation_selection_stats import (
    daily_cross_sectional_spearman,
    filter_common_support,
    native_support_summary,
    rank_q_candidates,
    select_anchor_top_two,
    temporal_product_spearman,
)

RUNNER_VERSION: Final = "foundation_selection_s05_rebuild_runner_v2"
LEGACY_RUNNER_VERSION_V1: Final = "foundation_selection_s05_rebuild_runner_v1"
CHECKPOINT_SCHEMA_VERSION: Final = "foundation_selection_atomic_checkpoint_v1"
EXPECTED_REGISTRY_VERSION: Final = "foundation_selection_s05_rebuild_registry_v2"
EXPECTED_REGISTRY_SHA256: Final = (
    "bd6ddda5fde082e5cb66638b87785ce81215c6b322ba80b1cef1729ef73f0958"
)
EXPECTED_SESSION_LIST_SHA256: Final = (
    "4781a479f4c04b6d53fe206035a89c1bb7a83ecc8d2796266aac6467bb7bc6cd"
)
PRIMARY_HORIZON_ID: Final = "future_median_30_300s"
ALL_TOD_BUCKET: Final = "all"
COUNT120_DIAGNOSTIC_ID: Final = "count_ewma_120obs"
ANCHOR_BOOTSTRAP_REPLICATES: Final = 5_000
ANCHOR_BOOTSTRAP_BLOCK_SESSIONS: Final = 5
ANCHOR_BOOTSTRAP_SEED: Final = 20_260_825
RAW_RANK1_BOUNDARY_CANDIDATES: Final = (
    "Q0_trail60_event_pooled",
    "Q1_trail60_date_equal",
    "Q2_trail20_date_equal",
    "Q5_prev1_expiry_dte5",
    "Q6_prev2_expiry_dte5",
)
DIAGNOSTIC_BOUNDARY_CANDIDATES: Final = (
    "Q1_trail60_date_equal",
    "Q2_trail20_date_equal",
)
SELECTABLE_BASE_BOUNDARY_CANDIDATES: Final = (
    "Q1_trail60_date_equal",
    "Q2_trail20_date_equal",
    "Q3_shape60_level5",
    "Q4_shape60_level10",
    "Q6_prev2_expiry_dte5",
)
BOUNDARY_SIMPLICITY_ORDER: Final = SELECTABLE_BASE_BOUNDARY_CANDIDATES
BOUNDARY_UNIT_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "tod_bucket",
    "boundary_quantile",
    "side",
)
BOUNDARY_EPISODE_COLUMNS: Final = (
    "episode_id",
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "side",
    "start_seconds_from_open",
    "end_seconds_from_open",
    "observed_amplitude_bp",
    "left_censored",
    "right_censored",
    "right_censor_reason",
    "completed_center_return",
    "tod_bucket",
    "expiry_date",
    "calendar_dte",
)
SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY: Final = 24
CONVERGENCE_PRIMARY_LOOKUP_ID: Final = "trail20_date_equal"
CONVERGENCE_FALLBACK_LOOKUP_ID: Final = "trail60_date_equal"
CONVERGENCE_REGISTERED_LOOKUPS: Final = (
    CONVERGENCE_PRIMARY_LOOKUP_ID,
    CONVERGENCE_FALLBACK_LOOKUP_ID,
)
CONVERGENCE_BASE_CANDIDATES: Final = (
    "C2_conditional_reach80",
    "C3_conditional_reach50",
)
CONVERGENCE_CONTROL_CANDIDATES: Final = (
    "C0_center",
    "C1_independent_lower_control",
)

DEFAULT_REGISTRY_PATH: Final = Path(__file__).with_name(
    "foundation_selection_registry.json"
)
DEFAULT_OUTPUT_ROOT: Final = (
    MAKER_ROOT / "data" / "walkforward" / "foundation_selection_s05_rebuild_20260826_v2"
)
LEGACY_OUTPUT_ROOT_V1: Final = (
    MAKER_ROOT / "data" / "walkforward" / "foundation_selection_s05_rebuild_20260826_v1"
)

ANCHOR_READ_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "seconds_from_open",
    "end_date",
    "basis_mid_bp",
    "eligible_base",
)
EPISODE_READ_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "timestamp",
    "seconds_from_open",
    "end_date",
    "basis_mid_bp",
    "eligible_base",
    "analysis_eligible",
)
MAPPING_READ_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "contract_size",
    "decimal_locator",
    "end_date",
    "fut_ref_price",
    "spot_ref_price",
    "day_trade_mark",
    "trading_turnover",
    "ins_type",
)
CONVERGENCE_FACT_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "anchor_model_id": pl.String,
        "candidate_id": pl.String,
        "boundary_quantile": pl.Int64,
        "episode_sequence": pl.Int64,
        "episode_start_second": pl.Int64,
        "tod_bucket": pl.String,
        "upper_distance_bp": pl.Float64,
        "independent_lower_distance_bp": pl.Float64,
        "source_asof_date": pl.String,
        "touch_second": pl.Int64,
        "touch_timestamp": pl.Datetime("ns"),
        "touch_residual_bp": pl.Float64,
        "touch_basis_mid_bp": pl.Float64,
        "touch_anchor_basis_bp": pl.Float64,
        "frozen_center_basis_bp": pl.Float64,
        "frozen_independent_lower_basis_bp": pl.Float64,
        "convergence_reference_semantics": pl.String,
        "center_hit": pl.Boolean,
        "center_hit_second": pl.Int64,
        "time_to_center_seconds": pl.Int64,
        "observed_post_touch_floor_bp": pl.Float64,
        "floor_frontier_distance_bp": pl.List(pl.Float64),
        "floor_frontier_hit_second": pl.List(pl.Int64),
        "negative_cycle_completed": pl.Boolean,
        "right_censored": pl.Boolean,
        "right_censor_reason": pl.String,
        "path_end_second": pl.Int64,
        "independent_lower_confirmed_hit": pl.Boolean,
        "independent_lower_hit_second": pl.Int64,
        "independent_lower_time_from_touch_seconds": pl.Int64,
        "independent_lower_time_from_center_seconds": pl.Int64,
        "independent_lower_known_miss": pl.Boolean,
        "independent_lower_unknown": pl.Boolean,
    }
)


@dataclass(frozen=True)
class SelectionPaths:
    """Frozen paths for one staged foundation-selection run."""

    sessions_path: Path = DEFAULT_SESSIONS_PATH
    daily_root: Path = DEFAULT_DAILY_ROOT
    registry_path: Path = DEFAULT_REGISTRY_PATH
    liquidity_path: Path = DEFAULT_LIQUIDITY_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    work_root: Path | None = None

    @property
    def resolved_work_root(self) -> Path:
        """Return the deterministic work directory beside ``output_root``."""

        if self.work_root is not None:
            return Path(self.work_root)
        output = Path(self.output_root)
        return output.with_name(f".{output.name}.work")


@dataclass(frozen=True)
class RegistrySnapshot:
    """Validated frozen registry payload and its full-content digest."""

    path: Path
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class RunContext:
    """Validated immutable lineage shared by every stage."""

    paths: SelectionPaths
    registry: RegistrySnapshot
    sessions: tuple[str, ...]
    primary_sessions: tuple[str, ...]
    git_state: Mapping[str, object]


@dataclass(frozen=True)
class BoundaryRankingFacts:
    """Auditable reductions used by the q-candidate lexicographic rank."""

    candidate_scores: pl.DataFrame
    common_calibration: pl.DataFrame
    monthly_q_side_loss: pl.DataFrame
    daily_spearman: pl.DataFrame
    temporal_spearman: pl.DataFrame
    support_summary: pl.DataFrame
    turnover_by_date: pl.DataFrame


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"registry missing field: {'.'.join(keys)}")
        value = value[key]
    return value


def _validate_registry_constants(payload: Mapping[str, object]) -> None:
    """Reject semantic drift between the registry and executable contracts."""

    expected_scalars: tuple[tuple[tuple[str, ...], object], ...] = (
        (("registry_version",), EXPECTED_REGISTRY_VERSION),
        (("frozen_on",), "2026-08-26"),
        (("development_only",), True),
        (("source_start",), SOURCE_START_DATE),
        (("source_end",), SOURCE_END_DATE),
        (("protected_forward_start",), PROTECTED_FORWARD_START_DATE),
        (("primary_start",), PRIMARY_START_DATE),
        (("analysis_start_second",), 300),
        (("entry_stop_second",), 14_400),
        (("anchor", "primary_horizon", "id"), PRIMARY_HORIZON_ID),
        (("anchor", "selection", "bootstrap", "block_sessions"), 5),
        (("anchor", "selection", "bootstrap", "replicates"), 5_000),
        (("anchor", "selection", "bootstrap", "seed"), 20_260_825),
        (
            ("boundary", "selection", "temporal_product_spearman_minimum_dates"),
            20,
        ),
        (("convergence", "tracking_end_second"), 15_600),
        (
            ("convergence", "reference_semantics"),
            "frozen_anchor_at_upper_touch",
        ),
        (
            ("convergence", "frozen_center_formula"),
            "selected_anchor_bp_at_upper_touch",
        ),
        (
            ("convergence", "frozen_lower_formula"),
            "frozen_center_basis_bp_minus_threshold_distance_bp",
        ),
        (
            ("convergence", "dynamic_anchor_results_role"),
            "noncanonical_sensitivity_only",
        ),
    )
    mismatches = []
    for keys, expected in expected_scalars:
        actual = _nested(payload, *keys)
        if actual != expected:
            mismatches.append(f"{'.'.join(keys)}={actual!r}, expected {expected!r}")
    if mismatches:
        raise ValueError("registry constant drift: " + "; ".join(mismatches))

    registry_candidates = [
        *list(_nested(payload, "anchor", "actionable_candidates")),
        *list(_nested(payload, "anchor", "controls")),
    ]
    candidate_contract = {
        str(row["id"]): str(row["kind"])
        for row in registry_candidates
        if isinstance(row, Mapping)
    }
    executable_contract = {
        candidate.model_id: candidate.kind for candidate in ANCHOR_CANDIDATES
    }
    if candidate_contract != executable_contract:
        raise ValueError("registry anchor candidates differ from executable candidates")
    executable_q_eligibility = {
        candidate.model_id: candidate.q_eligible for candidate in ANCHOR_CANDIDATES
    }
    explicit_q_drift = [
        str(row["id"])
        for row in registry_candidates
        if isinstance(row, Mapping)
        and "q_eligible" in row
        and bool(row["q_eligible"]) != executable_q_eligibility[str(row["id"])]
    ]
    if explicit_q_drift:
        raise ValueError(
            "registry q-eligibility differs from executable candidates: "
            f"{explicit_q_drift}"
        )

    horizon_rows = [
        _nested(payload, "anchor", "primary_horizon"),
        *list(_nested(payload, "anchor", "sensitivity_horizons")),
    ]
    valid_horizon_rows = [row for row in horizon_rows if isinstance(row, Mapping)]
    if len(valid_horizon_rows) != len(FUTURE_HORIZONS):
        raise ValueError("registry future horizons differ from executable horizons")
    horizon_drift = False
    for row, horizon in zip(
        valid_horizon_rows,
        FUTURE_HORIZONS,
        strict=True,
    ):
        horizon_drift = horizon_drift or (
            str(row["id"]) != horizon.horizon_id
            or int(row["start_seconds"]) != horizon.start_seconds
            or int(row["end_seconds"]) != horizon.end_seconds
            or not math.isclose(
                float(row["minimum_coverage"]),
                horizon.minimum_coverage,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    if horizon_drift:
        raise ValueError("registry future horizons differ from executable horizons")

    registry_tod = [
        (
            str(row["id"]),
            int(row["start_second"]),
            int(row["end_second"]),
        )
        for row in list(_nested(payload, "boundary", "tod_buckets"))
        if isinstance(row, Mapping)
    ]
    executable_tod = [
        (bucket.id, bucket.start_second, bucket.end_second)
        for bucket in DEFAULT_TOD_BUCKETS
    ]
    if registry_tod != executable_tod:
        raise ValueError("registry TOD buckets differ from executable buckets")

    from .foundation_convergence_lookup import REGISTERED_LOOKUPS

    registry_windows = tuple(
        (
            int(window),
            int(_nested(payload, "convergence", "minimum_history_dates")[str(window)]),
            int(
                _nested(payload, "convergence", "minimum_completed_paths")[str(window)]
            ),
        )
        for window in _nested(payload, "convergence", "history_windows_sessions")
    )
    executable_windows = tuple(
        (
            spec.lookback_sessions,
            spec.minimum_completed_dates,
            spec.minimum_completed_paths,
        )
        for spec in REGISTERED_LOOKUPS
    )
    if registry_windows != executable_windows:
        raise ValueError("registry convergence windows differ from executable lookups")
    registry_convergence_candidates = tuple(
        str(row["id"])
        for row in _nested(payload, "convergence", "candidates")
        if isinstance(row, Mapping)
    )
    if registry_convergence_candidates != (
        *CONVERGENCE_CONTROL_CANDIDATES,
        *CONVERGENCE_BASE_CANDIDATES,
    ):
        raise ValueError(
            "registry convergence candidates differ from executable candidates"
        )
    if tuple(_nested(payload, "convergence", "right_censor_reasons")) != (
        "eligibility_gap",
        "session_cutoff",
    ):
        raise ValueError("registry convergence censor reasons drifted")


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> RegistrySnapshot:
    """Load only the byte-exact frozen registry and validate its constants."""

    registry_path = Path(path)
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ValueError(f"registry must be a regular file: {registry_path}")
    digest = _sha256_file(registry_path)
    if digest != EXPECTED_REGISTRY_SHA256:
        raise ValueError(
            "foundation selection registry hash drift: "
            f"{digest} != {EXPECTED_REGISTRY_SHA256}"
        )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("foundation selection registry must be a JSON object")
    _validate_registry_constants(payload)
    return RegistrySnapshot(registry_path.resolve(), digest, payload)


def load_selection_sessions(path: Path = DEFAULT_SESSIONS_PATH) -> tuple[str, ...]:
    """Load and content-lock the explicit 131/71-session development set."""

    sessions = tuple(load_frozen_sessions(Path(path)))
    digest = _canonical_sha256(sessions)
    if digest != EXPECTED_SESSION_LIST_SHA256:
        raise ValueError(
            "frozen session list content drift: "
            f"{digest} != {EXPECTED_SESSION_LIST_SHA256}"
        )
    if len(sessions) != EXPECTED_SESSION_COUNT:
        raise ValueError("frozen source session count drift")
    primary = tuple(date for date in sessions if date >= PRIMARY_START_DATE)
    if len(primary) != EXPECTED_PRIMARY_SESSION_COUNT:
        raise ValueError("frozen primary session count drift")
    if any(date >= PROTECTED_FORWARD_START_DATE for date in sessions):
        raise ValueError("protected-forward Date entered the frozen session list")
    return sessions


def _require_canonical_git_state(canonical: bool) -> Mapping[str, object]:
    state = _git_state()
    if canonical and bool(state.get("dirty")):
        raise ValueError(
            "canonical foundation selection requires a clean committed source "
            f"tree: {state.get('status', '')}"
        )
    commit = str(state.get("commit", ""))
    if canonical and not commit:
        raise ValueError("canonical foundation selection requires a git commit")
    return state


def _normalise_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalised = [dict(record) for record in records]
    required = {"path", "role", "bytes", "sha256"}
    for record in normalised:
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"input record missing fields: {missing}")
        record.setdefault("path_scope", "absolute")
        if record["path_scope"] not in {"maker_root", "absolute"}:
            raise ValueError(f"unknown input path scope: {record['path_scope']!r}")
    return sorted(
        normalised,
        key=lambda row: (
            str(row["path_scope"]),
            str(row["path"]),
            str(row["role"]),
        ),
    )


def _portable_path_record(path: Path) -> tuple[str, str]:
    """Return a checkout-portable maker path or an explicit external path."""

    resolved = Path(path).resolve()
    maker_root = MAKER_ROOT.resolve()
    try:
        relative = resolved.relative_to(maker_root)
    except ValueError:
        return "absolute", str(resolved)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe maker-root-relative input path: {relative}")
    return "maker_root", relative.as_posix()


def _resolve_file_record_path(record: Mapping[str, object]) -> Path:
    """Resolve one persisted scoped record in the current checkout."""

    scope = str(record.get("path_scope", "absolute"))
    stored = Path(str(record["path"]))
    if scope == "maker_root":
        if stored.is_absolute() or ".." in stored.parts:
            raise ValueError(f"unsafe maker-root input record path: {stored}")
        return MAKER_ROOT / stored
    if scope == "absolute":
        if not stored.is_absolute():
            raise ValueError(f"absolute input record is not absolute: {stored}")
        return stored
    raise ValueError(f"unknown input path scope: {scope!r}")


def build_file_records(
    paths_and_roles: Sequence[tuple[Path, str]],
) -> list[dict[str, object]]:
    """Hash inputs with maker-root-relative identities when possible."""

    roles_by_path: dict[tuple[str, str], str] = {}
    for raw_path, raw_role in paths_and_roles:
        unresolved = Path(raw_path)
        if unresolved.is_symlink():
            raise ValueError(f"checkpoint input must not be a symlink: {unresolved}")
        scope, path = _portable_path_record(unresolved)
        role = str(raw_role)
        previous = roles_by_path.get((scope, path))
        if previous is not None and previous != role:
            raise ValueError(f"input path has conflicting roles: {path}")
        roles_by_path[(scope, path)] = role
    records: list[dict[str, object]] = []
    for (scope, path_text), role in sorted(roles_by_path.items()):
        record = {"path_scope": scope, "path": path_text}
        path = _resolve_file_record_path(record)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint input must be a regular file: {path}")
        records.append(
            {
                "path_scope": scope,
                "path": path_text,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _assert_file_records_current(
    records: Sequence[Mapping[str, object]],
) -> None:
    for record in _normalise_records(records):
        path = _resolve_file_record_path(record)
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"checkpoint input content drift: {path}")


def checkpoint_fingerprint(
    *,
    phase: str,
    date: str,
    registry_sha256: str,
    source_commit: str,
    input_records: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
    runner_version: str = RUNNER_VERSION,
    session_list_sha256: str = EXPECTED_SESSION_LIST_SHA256,
) -> str:
    """Return the canonical fingerprint for one phase/Date computation."""

    payload = {
        "runner_version": str(runner_version),
        "phase": str(phase),
        "date": str(date),
        "registry_sha256": str(registry_sha256),
        "session_list_sha256": str(session_list_sha256),
        "source_commit": str(source_commit),
        "input_records": _normalise_records(input_records),
        "parameters": dict(parameters),
    }
    return _canonical_sha256(payload)


def _checkpoint_marker_payload(
    *,
    phase: str,
    date: str,
    registry_sha256: str,
    source_commit: str,
    input_records: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    runner_version: str = RUNNER_VERSION,
    checkpoint_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    session_list_sha256: str = EXPECTED_SESSION_LIST_SHA256,
) -> dict[str, object]:
    fingerprint = checkpoint_fingerprint(
        phase=phase,
        date=date,
        registry_sha256=registry_sha256,
        source_commit=source_commit,
        input_records=input_records,
        parameters=parameters,
        runner_version=runner_version,
        session_list_sha256=session_list_sha256,
    )
    payload: dict[str, object] = {
        "schema_version": str(checkpoint_schema_version),
        "runner_version": str(runner_version),
        "phase": phase,
        "date": date,
        "registry_sha256": registry_sha256,
        "session_list_sha256": str(session_list_sha256),
        "source_commit": source_commit,
        "input_records": _normalise_records(input_records),
        "parameters": dict(parameters),
        "checkpoint_fingerprint": fingerprint,
        "artifacts": {name: dict(value) for name, value in artifacts.items()},
        "complete": True,
    }
    payload["marker_payload_sha256"] = _canonical_sha256(payload)
    return payload


def verify_date_checkpoint(
    partition: Path,
    *,
    expected_fingerprint: str | None = None,
    verify_inputs: bool = False,
    expected_runner_version: str = RUNNER_VERSION,
    expected_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    expected_session_list_sha256: str = EXPECTED_SESSION_LIST_SHA256,
) -> Mapping[str, object]:
    """Verify marker self-hash, artifact hashes, and optional input lineage."""

    root = Path(partition)
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"checkpoint marker is missing: {marker_path}")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint marker is not an object: {marker_path}")
    marker_digest = payload.pop("marker_payload_sha256", None)
    if marker_digest != _canonical_sha256(payload):
        raise ValueError(f"checkpoint marker self-hash mismatch: {marker_path}")
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(f"unsupported checkpoint schema: {marker_path}")
    if payload.get("runner_version") != expected_runner_version:
        raise ValueError(f"checkpoint runner version drift: {marker_path}")
    if payload.get("session_list_sha256") != expected_session_list_sha256:
        raise ValueError(f"checkpoint session-list drift: {marker_path}")
    if payload.get("complete") is not True:
        raise ValueError(f"checkpoint is not complete: {marker_path}")
    fingerprint = checkpoint_fingerprint(
        phase=str(payload["phase"]),
        date=str(payload["date"]),
        registry_sha256=str(payload["registry_sha256"]),
        source_commit=str(payload["source_commit"]),
        input_records=list(payload["input_records"]),
        parameters=dict(payload["parameters"]),
        runner_version=expected_runner_version,
        session_list_sha256=expected_session_list_sha256,
    )
    if payload.get("checkpoint_fingerprint") != fingerprint:
        raise ValueError(f"checkpoint fingerprint self-mismatch: {marker_path}")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(f"checkpoint lineage mismatch: {marker_path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"checkpoint has no declared artifacts: {marker_path}")
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "complete.json"
    }
    if actual_files != set(artifacts):
        raise ValueError(f"checkpoint artifact set mismatch: {marker_path}")
    for name, expected in artifacts.items():
        if Path(str(name)).name != str(name):
            raise ValueError(f"unsafe checkpoint artifact name: {name!r}")
        actual = _artifact_metadata(root / str(name))
        if actual != expected:
            raise ValueError(
                f"checkpoint artifact metadata mismatch: {root / str(name)}"
            )
    if verify_inputs:
        _assert_file_records_current(list(payload["input_records"]))
    payload["marker_payload_sha256"] = marker_digest
    return payload


def verify_legacy_v1_date_checkpoint(
    partition: Path,
    *,
    expected_fingerprint: str | None = None,
    verify_inputs: bool = False,
) -> Mapping[str, object]:
    """Verify only the byte-explicit ace2669/v1 checkpoint contract."""

    return verify_date_checkpoint(
        partition,
        expected_fingerprint=expected_fingerprint,
        verify_inputs=verify_inputs,
        expected_runner_version=LEGACY_RUNNER_VERSION_V1,
        expected_schema_version="foundation_selection_atomic_checkpoint_v1",
        expected_session_list_sha256=EXPECTED_SESSION_LIST_SHA256,
    )


def _write_checkpoint_artifact(path: Path, value: object) -> None:
    if isinstance(value, pl.DataFrame):
        _write_frame(value, path)
        return
    if path.suffix != ".json":
        raise TypeError(f"non-frame checkpoint artifacts must be JSON: {path}")
    _write_json(path, value)


def atomic_publish_checkpoint(
    partition: Path,
    *,
    phase: str,
    date: str,
    registry_sha256: str,
    source_commit: str,
    input_records: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
    artifact_values: Mapping[str, object],
    runner_version: str = RUNNER_VERSION,
    checkpoint_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    session_list_sha256: str = EXPECTED_SESSION_LIST_SHA256,
) -> Mapping[str, object]:
    """Publish a complete checkpoint directory with one atomic rename."""

    destination = Path(partition)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not artifact_values:
        raise ValueError("checkpoint must contain at least one artifact")
    for name in artifact_values:
        if Path(name).name != name or name == "complete.json":
            raise ValueError(f"unsafe checkpoint artifact name: {name!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        _assert_file_records_current(input_records)
        for name in sorted(artifact_values):
            _write_checkpoint_artifact(temporary / name, artifact_values[name])
        artifacts = {
            name: _artifact_metadata(temporary / name)
            for name in sorted(artifact_values)
        }
        _assert_file_records_current(input_records)
        marker = _checkpoint_marker_payload(
            phase=phase,
            date=date,
            registry_sha256=registry_sha256,
            source_commit=source_commit,
            input_records=input_records,
            parameters=parameters,
            artifacts=artifacts,
            runner_version=runner_version,
            checkpoint_schema_version=checkpoint_schema_version,
            session_list_sha256=session_list_sha256,
        )
        _write_json(temporary / "complete.json", marker)
        expected = str(marker["checkpoint_fingerprint"])
        verify_date_checkpoint(
            temporary,
            expected_fingerprint=expected,
            expected_runner_version=runner_version,
            expected_schema_version=checkpoint_schema_version,
            expected_session_list_sha256=session_list_sha256,
        )
        os.replace(temporary, destination)
        return verify_date_checkpoint(
            destination,
            expected_fingerprint=expected,
            expected_runner_version=runner_version,
            expected_schema_version=checkpoint_schema_version,
            expected_session_list_sha256=session_list_sha256,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resume_date_checkpoint(
    partition: Path,
    *,
    expected_fingerprint: str,
    resume: bool,
) -> Mapping[str, object] | None:
    """Return a verified marker when resumable, otherwise signal pending."""

    path = Path(partition)
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(path)
    return verify_date_checkpoint(
        path,
        expected_fingerprint=expected_fingerprint,
        verify_inputs=False,
    )


def _daily_partition(paths: SelectionPaths, date: str) -> Path:
    if date >= PROTECTED_FORWARD_START_DATE:
        raise ValueError(f"protected forward partition requested: {date}")
    return Path(paths.daily_root) / f"Date={date}"


def _context(
    paths: SelectionPaths,
    *,
    canonical: bool,
) -> RunContext:
    registry = load_registry(paths.registry_path)
    sessions = load_selection_sessions(paths.sessions_path)
    state = _require_canonical_git_state(canonical)
    primary = tuple(date for date in sessions if date >= PRIMARY_START_DATE)
    return RunContext(paths, registry, sessions, primary, state)


def _preflight_input_records(context: RunContext) -> list[dict[str, object]]:
    inputs = [
        (context.paths.sessions_path, "frozen_session_calendar"),
        (context.paths.registry_path, "frozen_selection_registry"),
    ]
    inputs.extend(
        (
            _daily_partition(context.paths, date) / "complete.json",
            "daily_completion_marker",
        )
        for date in context.sessions
    )
    return build_file_records(inputs)


def _preflight_parameters(context: RunContext) -> dict[str, object]:
    return {
        "source_start": SOURCE_START_DATE,
        "source_end": SOURCE_END_DATE,
        "primary_start": PRIMARY_START_DATE,
        "protected_forward_start": PROTECTED_FORWARD_START_DATE,
        "session_count": len(context.sessions),
        "primary_session_count": len(context.primary_sessions),
        "validation": "named_partitions_only",
    }


def run_preflight(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Validate the registry, git state, calendar, and all named daily inputs."""

    selected_paths = paths or SelectionPaths()
    context = _context(selected_paths, canonical=canonical)
    audit_rows, _ = validate_named_daily_partitions(
        selected_paths.daily_root,
        context.sessions,
    )
    records = _preflight_input_records(context)
    parameters = _preflight_parameters(context)
    fingerprint = checkpoint_fingerprint(
        phase="preflight",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
    )
    partition = selected_paths.resolved_work_root / "preflight"
    resumed = resume_date_checkpoint(
        partition,
        expected_fingerprint=fingerprint,
        resume=resume,
    )
    if resumed is not None:
        if canonical and _git_state() != context.git_state:
            raise ValueError("source tree changed during canonical preflight")
        return resumed
    if canonical and _git_state() != context.git_state:
        raise ValueError("source tree changed during canonical preflight")
    result = atomic_publish_checkpoint(
        partition,
        phase="preflight",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
        artifact_values={
            "input_audit.parquet": pl.from_dicts(
                audit_rows,
                infer_schema_length=None,
            )
        },
    )
    if canonical and _git_state() != context.git_state:
        raise ValueError("source tree changed during canonical preflight")
    return result


def _anchor_parameters() -> dict[str, object]:
    return {
        "read_columns": list(ANCHOR_READ_COLUMNS),
        "models": [candidate.model_id for candidate in ANCHOR_CANDIDATES],
        "horizons": [horizon.horizon_id for horizon in FUTURE_HORIZONS],
        "product_day_weighting": "eligible_one_second_occupancy_equal",
        "comparison_support": "all_actionable_common_seconds",
    }


def _anchor_input_records(
    context: RunContext,
    date: str,
) -> list[dict[str, object]]:
    daily = _daily_partition(context.paths, date)
    return build_file_records(
        [
            (daily / "complete.json", "daily_completion_marker"),
            (daily / "causal_fair.parquet", "daily_causal_fair_input"),
        ]
    )


def _anchor_date_partition(context: RunContext, date: str) -> Path:
    return context.paths.resolved_work_root / "anchor" / "dates" / f"Date={date}"


def aggregate_anchor_daily(
    product_day_stats: pl.DataFrame,
    *,
    primary_horizon_id: str = PRIMARY_HORIZON_ID,
) -> pl.DataFrame:
    """Reduce primary/all-TOD facts to product-day-equal daily model means.

    ``product_days`` is deliberately retained as the month-level selection
    weight.  Thus Dates contribute in proportion to their mapped product-day
    population within a month, while the selector subsequently gives each
    month equal weight.
    """

    required = {
        "Date",
        "month",
        "ValueCode",
        "QuoteCode",
        "model",
        "horizon_id",
        "tod_bucket",
        "mae_bp",
        "anchor_tv_over_basis_tv",
        "n_common_evaluable",
    }
    missing = sorted(required - set(product_day_stats.columns))
    if missing:
        raise ValueError(f"anchor product-day facts missing columns: {missing}")
    scoped = product_day_stats.filter(
        (pl.col("horizon_id") == primary_horizon_id)
        & (pl.col("tod_bucket") == ALL_TOD_BUCKET)
        & pl.col("mae_bp").is_not_null()
        & pl.col("mae_bp").is_finite()
        & pl.col("anchor_tv_over_basis_tv").is_not_null()
        & pl.col("anchor_tv_over_basis_tv").is_finite()
        & (pl.col("n_common_evaluable") > 0)
    )
    if scoped.is_empty():
        raise ValueError("anchor primary/all-TOD scope is empty")
    duplicate = (
        scoped.group_by(["Date", "ValueCode", "QuoteCode", "model"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("anchor primary scope has duplicate product-day models")
    daily = (
        scoped.group_by(["Date", "month", "model"])
        .agg(
            pl.len().alias("product_days"),
            pl.col("n_common_evaluable").sum().alias("common_seconds"),
            pl.col("mae_bp").mean().alias("mae_bp"),
            pl.col("anchor_tv_over_basis_tv").mean().alias("tv_ratio"),
        )
        .sort(["Date", "model"])
    )
    actionable = daily.filter(pl.col("model").is_in(ACTIONABLE_MODEL_IDS))
    if actionable["model"].n_unique() != len(ACTIONABLE_MODEL_IDS):
        raise ValueError("daily anchor panel is missing actionable models")
    weight_drift = (
        actionable.group_by("Date")
        .agg(
            pl.col("model").n_unique().alias("models"),
            pl.col("product_days").min().alias("minimum_product_days"),
            pl.col("product_days").max().alias("maximum_product_days"),
        )
        .filter(
            (pl.col("models") != len(ACTIONABLE_MODEL_IDS))
            | (pl.col("minimum_product_days") != pl.col("maximum_product_days"))
        )
    )
    if weight_drift.height:
        raise ValueError(
            "actionable anchor candidates do not share Date/product-day weights"
        )
    return daily


def _load_anchor_product_day(
    context: RunContext,
) -> pl.DataFrame:
    paths = [
        _anchor_date_partition(context, date) / "product_day.parquet"
        for date in context.primary_sessions
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"anchor Date checkpoint is incomplete: {path}")
    return pl.concat(
        [pl.read_parquet(path) for path in paths],
        how="vertical_relaxed",
    )


def _anchor_selection_parameters(
    *,
    bootstrap_replicates: int,
) -> dict[str, object]:
    return {
        "primary_horizon_id": PRIMARY_HORIZON_ID,
        "tod_bucket": ALL_TOD_BUCKET,
        "actionable_models": list(ACTIONABLE_MODEL_IDS),
        "controls": [
            candidate.model_id
            for candidate in ANCHOR_CANDIDATES
            if candidate.model_id not in ACTIONABLE_MODEL_IDS
        ],
        "weight_column": "product_days",
        "block_sessions": ANCHOR_BOOTSTRAP_BLOCK_SESSIONS,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": ANCHOR_BOOTSTRAP_SEED,
        "top_two_rule": (
            "raw_mae_winner_plus_smoothest_actionable_within_paired_one_se"
        ),
    }


def _anchor_selection_input_records(context: RunContext) -> list[dict[str, object]]:
    return build_file_records(
        [
            (
                _anchor_date_partition(context, date) / "complete.json",
                "anchor_date_checkpoint_marker",
            )
            for date in context.primary_sessions
        ]
    )


def _selected_anchor_payload(selection: pl.DataFrame) -> dict[str, object]:
    required = {"model", "selection_rank", "selection_role"}
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(f"anchor selection missing columns: {missing}")
    ranked = selection.filter(pl.col("selection_rank").is_not_null()).sort(
        "selection_rank"
    )
    if ranked.height != 2 or ranked["selection_rank"].to_list() != [1, 2]:
        raise ValueError("anchor selection must contain exact ranks one and two")
    rank1 = str(ranked.row(0, named=True)["model"])
    rank2 = str(ranked.row(1, named=True)["model"])
    if (
        rank1 == rank2
        or rank1 not in ACTIONABLE_MODEL_IDS
        or rank2 not in ACTIONABLE_MODEL_IDS
    ):
        raise ValueError("selected anchor identities are not a distinct top two")
    return {
        "rank1": rank1,
        "rank2": rank2,
        "count120_diagnostic": COUNT120_DIAGNOSTIC_ID,
        "episode_anchor_ids": [rank1, rank2, COUNT120_DIAGNOSTIC_ID],
        "rank1_role": str(ranked.row(0, named=True)["selection_role"]),
        "rank2_role": str(ranked.row(1, named=True)["selection_role"]),
        "q_primary_anchor": rank1,
        "q_diagnostic_anchors": [rank2, COUNT120_DIAGNOSTIC_ID],
        "development_only": True,
    }


def _anchor_selection_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "anchor" / "selection"


def run_anchor_phase(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
    bootstrap_replicates: int = ANCHOR_BOOTSTRAP_REPLICATES,
) -> Mapping[str, object]:
    """Evaluate 71 primary sessions and select two actionable anchors."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if canonical and bootstrap_replicates != ANCHOR_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "canonical anchor selection requires the registry-frozen "
            f"{ANCHOR_BOOTSTRAP_REPLICATES} bootstrap replicates"
        )
    selected_paths = paths or SelectionPaths()
    run_preflight(selected_paths, resume=resume, canonical=canonical)
    context = _context(selected_paths, canonical=canonical)
    initial_git_state = dict(context.git_state)
    parameters = _anchor_parameters()
    source_commit = str(context.git_state["commit"])
    for index, date in enumerate(context.primary_sessions, start=1):
        records = _anchor_input_records(context, date)
        fingerprint = checkpoint_fingerprint(
            phase="anchor",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=parameters,
        )
        partition = _anchor_date_partition(context, date)
        if (
            resume_date_checkpoint(
                partition,
                expected_fingerprint=fingerprint,
                resume=resume,
            )
            is not None
        ):
            print(
                f"anchor {index}/{len(context.primary_sessions)} {date} resumed",
                flush=True,
            )
            continue
        causal_path = _daily_partition(selected_paths, date) / "causal_fair.parquet"
        day = pl.read_parquet(causal_path, columns=list(ANCHOR_READ_COLUMNS))
        evaluated = evaluate_anchor_selection_day(day)
        atomic_publish_checkpoint(
            partition,
            phase="anchor",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=parameters,
            artifact_values={
                "product_day.parquet": evaluated.product_day_stats,
                "support.parquet": evaluated.support_stats,
            },
        )
        del day, evaluated
        gc.collect()
        print(
            f"anchor {index}/{len(context.primary_sessions)} {date} complete",
            flush=True,
        )

    selection_parameters = _anchor_selection_parameters(
        bootstrap_replicates=bootstrap_replicates
    )
    selection_records = _anchor_selection_input_records(context)
    selection_fingerprint = checkpoint_fingerprint(
        phase="anchor_selection",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=selection_records,
        parameters=selection_parameters,
    )
    selection_partition = _anchor_selection_partition(context)
    resumed = resume_date_checkpoint(
        selection_partition,
        expected_fingerprint=selection_fingerprint,
        resume=resume,
    )
    if resumed is None:
        product_day = _load_anchor_product_day(context)
        daily = aggregate_anchor_daily(product_day)
        selection = select_anchor_top_two(
            daily,
            actionable_models=ACTIONABLE_MODEL_IDS,
            controls=tuple(
                candidate.model_id
                for candidate in ANCHOR_CANDIDATES
                if candidate.model_id not in ACTIONABLE_MODEL_IDS
            ),
            mae_column="mae_bp",
            tv_column="tv_ratio",
            weight_column="product_days",
            block_sessions=ANCHOR_BOOTSTRAP_BLOCK_SESSIONS,
            replicates=bootstrap_replicates,
            seed=ANCHOR_BOOTSTRAP_SEED,
        )
        selected = _selected_anchor_payload(selection)
        resumed = atomic_publish_checkpoint(
            selection_partition,
            phase="anchor_selection",
            date="primary_71_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=selection_records,
            parameters=selection_parameters,
            artifact_values={
                "anchor_daily.parquet": daily,
                "anchor_selection.parquet": selection,
                "selected_anchors.json": selected,
            },
        )
    if canonical and _git_state() != initial_git_state:
        raise ValueError("source tree changed during canonical anchor phase")
    return resumed


def load_selected_anchor_ids(
    paths: SelectionPaths | None = None,
    *,
    expected_registry_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> tuple[str, str, str]:
    """Load the exact rank1, rank2, and count120 episode anchor order."""

    selected_paths = paths or SelectionPaths()
    partition = selected_paths.resolved_work_root / "anchor" / "selection"
    marker = verify_date_checkpoint(partition)
    if (
        expected_registry_sha256 is not None
        and marker["registry_sha256"] != expected_registry_sha256
    ):
        raise ValueError("selected anchor registry lineage mismatch")
    if (
        expected_source_commit is not None
        and marker["source_commit"] != expected_source_commit
    ):
        raise ValueError("selected anchor source-commit lineage mismatch")
    payload = json.loads(
        (partition / "selected_anchors.json").read_text(encoding="utf-8")
    )
    ids = tuple(str(value) for value in payload.get("episode_anchor_ids", []))
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("selected anchor checkpoint lacks three distinct anchors")
    if ids[2] != COUNT120_DIAGNOSTIC_ID:
        raise ValueError("selected anchor checkpoint lost the count120 diagnostic")
    if ids[0] not in ACTIONABLE_MODEL_IDS or ids[1] not in ACTIONABLE_MODEL_IDS:
        raise ValueError("selected anchor checkpoint has a non-actionable top two")
    return ids[0], ids[1], ids[2]


def _episode_parameters(anchor_ids: Sequence[str]) -> dict[str, object]:
    return {
        "read_columns": list(EPISODE_READ_COLUMNS),
        "mapping_columns": list(MAPPING_READ_COLUMNS),
        "anchor_ids": list(anchor_ids),
        "analysis_start_second": 300,
        "entry_stop_second": 14_400,
        "active_at_entry_stop": "right_censored_entry_stop",
        "post_entry_extrema_forbidden": True,
        "expiry_join": "exact_Date_ValueCode_QuoteCode",
    }


def _episode_input_records(
    context: RunContext,
    date: str,
) -> list[dict[str, object]]:
    daily = _daily_partition(context.paths, date)
    return build_file_records(
        [
            (daily / "complete.json", "daily_completion_marker"),
            (daily / "causal_fair.parquet", "daily_causal_fair_input"),
            (daily / "mapping.parquet", "daily_exact_contract_mapping"),
            (
                _anchor_selection_partition(context) / "complete.json",
                "selected_anchor_checkpoint_marker",
            ),
        ]
    )


def _episode_date_partition(context: RunContext, date: str) -> Path:
    return context.paths.resolved_work_root / "episodes" / "dates" / f"Date={date}"


def _anchor_role_expression(anchor_ids: Sequence[str]) -> pl.Expr:
    return (
        pl.when(pl.col("anchor_model_id") == anchor_ids[0])
        .then(pl.lit("rank1_primary"))
        .when(pl.col("anchor_model_id") == anchor_ids[1])
        .then(pl.lit("rank2_diagnostic"))
        .when(pl.col("anchor_model_id") == anchor_ids[2])
        .then(pl.lit("count120_diagnostic"))
        .otherwise(pl.lit("unknown"))
    )


def _episode_index_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "episodes" / "index"


def _episode_index_parameters(anchor_ids: Sequence[str]) -> dict[str, object]:
    return {
        "anchor_ids": list(anchor_ids),
        "expected_date_checkpoints": EXPECTED_SESSION_COUNT,
        "index_order": "frozen_session_calendar",
    }


def _episode_index_records(context: RunContext) -> list[dict[str, object]]:
    return build_file_records(
        [
            (
                _episode_date_partition(context, date) / "complete.json",
                "episode_date_checkpoint_marker",
            )
            for date in context.sessions
        ]
    )


def _episode_manifest(
    context: RunContext,
    markers: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    rows = []
    for date, marker in zip(context.sessions, markers, strict=True):
        artifact = dict(marker["artifacts"])["episodes.parquet"]
        rows.append(
            {
                "Date": date,
                "stage": ("history" if date < PRIMARY_START_DATE else "primary"),
                "episode_rows": int(artifact["rows"]),
                "checkpoint_fingerprint": str(marker["checkpoint_fingerprint"]),
                "partition": str(_episode_date_partition(context, date)),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def run_episode_phase(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Extract anchor-aware entry episodes for all 131 frozen sessions."""

    selected_paths = paths or SelectionPaths()
    run_preflight(selected_paths, resume=resume, canonical=canonical)
    context = _context(selected_paths, canonical=canonical)
    initial_git_state = dict(context.git_state)
    source_commit = str(context.git_state["commit"])
    anchor_ids = load_selected_anchor_ids(
        selected_paths,
        expected_registry_sha256=context.registry.sha256,
        expected_source_commit=source_commit,
    )
    parameters = _episode_parameters(anchor_ids)
    markers: list[Mapping[str, object]] = []
    for index, date in enumerate(context.sessions, start=1):
        records = _episode_input_records(context, date)
        fingerprint = checkpoint_fingerprint(
            phase="episodes",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=parameters,
        )
        partition = _episode_date_partition(context, date)
        marker = resume_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            resume=resume,
        )
        if marker is not None:
            markers.append(marker)
            print(
                f"episodes {index}/{len(context.sessions)} {date} resumed",
                flush=True,
            )
            continue
        daily = _daily_partition(selected_paths, date)
        day = pl.read_parquet(
            daily / "causal_fair.parquet",
            columns=list(EPISODE_READ_COLUMNS),
        )
        mapping = pl.read_parquet(
            daily / "mapping.parquet",
            columns=list(MAPPING_READ_COLUMNS),
        )
        episode_parts: list[pl.DataFrame] = []
        anchor_support: pl.DataFrame | None = None
        for anchor_id in anchor_ids:
            materialized = materialize_anchor_column(day, anchor_id)
            if anchor_id == anchor_ids[0]:
                anchor_support = build_anchor_product_day_support(
                    materialized,
                    anchor_column=MODEL_COLUMNS[anchor_id],
                    anchor_model_id=anchor_id,
                )
            episodes = extract_entry_window_excursions(
                materialized,
                anchor_column=MODEL_COLUMNS[anchor_id],
                anchor_model_id=anchor_id,
            )
            episode_parts.append(episodes)
            del materialized, episodes
            gc.collect()
        episode_facts = annotate_expiry_dte(
            pl.concat(episode_parts, how="vertical_relaxed"),
            mapping,
        ).with_columns(
            _anchor_role_expression(anchor_ids).alias("anchor_role"),
            pl.lit(True).alias("contains_target_day_outcome"),
            pl.lit("after_session_close").alias("outcome_available"),
        )
        if anchor_support is None:
            raise AssertionError("rank1 anchor support was not materialized")
        marker = atomic_publish_checkpoint(
            partition,
            phase="episodes",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=parameters,
            artifact_values={
                "episodes.parquet": episode_facts,
                "anchor_support.parquet": anchor_support,
            },
        )
        markers.append(marker)
        del day, mapping, episode_parts, episode_facts, anchor_support
        gc.collect()
        print(
            f"episodes {index}/{len(context.sessions)} {date} complete",
            flush=True,
        )

    index_parameters = _episode_index_parameters(anchor_ids)
    index_records = _episode_index_records(context)
    index_fingerprint = checkpoint_fingerprint(
        phase="episode_index",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=index_records,
        parameters=index_parameters,
    )
    index_partition = _episode_index_partition(context)
    resumed = resume_date_checkpoint(
        index_partition,
        expected_fingerprint=index_fingerprint,
        resume=resume,
    )
    if resumed is None:
        resumed = atomic_publish_checkpoint(
            index_partition,
            phase="episode_index",
            date="frozen_131_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=index_records,
            parameters=index_parameters,
            artifact_values={
                "episode_manifest.parquet": _episode_manifest(context, markers)
            },
        )
    if canonical and _git_state() != initial_git_state:
        raise ValueError("source tree changed during canonical episode phase")
    return resumed


def _reject_unfrozen_checkpoint_dates(
    root: Path,
    *,
    allowed_dates: Sequence[str],
) -> None:
    if not root.is_dir():
        return
    allowed = set(allowed_dates)
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("Date="):
            continue
        date = child.name.removeprefix("Date=")
        if date >= PROTECTED_FORWARD_START_DATE:
            raise ValueError(
                f"protected-forward checkpoint is present in work root: {child}"
            )
        if date not in allowed:
            raise ValueError(
                f"unfrozen checkpoint Date is present in work root: {child}"
            )


def verify_work(
    paths: SelectionPaths | None = None,
    *,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Verify every explicitly frozen checkpoint that currently exists."""

    selected_paths = paths or SelectionPaths()
    context = _context(selected_paths, canonical=canonical)
    source_commit = str(context.git_state["commit"])
    preflight_records = _preflight_input_records(context)
    preflight_fingerprint = checkpoint_fingerprint(
        phase="preflight",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=preflight_records,
        parameters=_preflight_parameters(context),
    )
    verify_date_checkpoint(
        selected_paths.resolved_work_root / "preflight",
        expected_fingerprint=preflight_fingerprint,
        verify_inputs=True,
    )

    anchor_root = selected_paths.resolved_work_root / "anchor" / "dates"
    episode_root = selected_paths.resolved_work_root / "episodes" / "dates"
    _reject_unfrozen_checkpoint_dates(
        anchor_root,
        allowed_dates=context.primary_sessions,
    )
    _reject_unfrozen_checkpoint_dates(
        episode_root,
        allowed_dates=context.sessions,
    )

    anchor_parameters = _anchor_parameters()
    verified_anchor_dates = 0
    for date in context.primary_sessions:
        partition = _anchor_date_partition(context, date)
        if not partition.exists():
            continue
        records = _anchor_input_records(context, date)
        expected = checkpoint_fingerprint(
            phase="anchor",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=anchor_parameters,
        )
        verify_date_checkpoint(
            partition,
            expected_fingerprint=expected,
            verify_inputs=True,
        )
        verified_anchor_dates += 1

    selection_partition = _anchor_selection_partition(context)
    selection_complete = selection_partition.exists()
    anchor_ids: tuple[str, str, str] | None = None
    if selection_complete:
        if verified_anchor_dates != len(context.primary_sessions):
            raise ValueError("anchor selection exists before all Date checkpoints")
        marker_records = _anchor_selection_input_records(context)
        marker = verify_date_checkpoint(
            selection_partition,
            verify_inputs=True,
        )
        if canonical and dict(marker["parameters"]) != (
            _anchor_selection_parameters(
                bootstrap_replicates=ANCHOR_BOOTSTRAP_REPLICATES
            )
        ):
            raise ValueError(
                "canonical anchor selection parameters differ from registry"
            )
        expected = checkpoint_fingerprint(
            phase="anchor_selection",
            date="primary_71_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=marker_records,
            parameters=dict(marker["parameters"]),
        )
        if marker["checkpoint_fingerprint"] != expected:
            raise ValueError("anchor selection checkpoint lineage mismatch")
        anchor_ids = load_selected_anchor_ids(
            selected_paths,
            expected_registry_sha256=context.registry.sha256,
            expected_source_commit=source_commit,
        )

    verified_episode_dates = 0
    if episode_root.exists() and anchor_ids is None:
        raise ValueError("episode checkpoints exist without anchor selection")
    if anchor_ids is not None:
        episode_parameters = _episode_parameters(anchor_ids)
        for date in context.sessions:
            partition = _episode_date_partition(context, date)
            if not partition.exists():
                continue
            records = _episode_input_records(context, date)
            expected = checkpoint_fingerprint(
                phase="episodes",
                date=date,
                registry_sha256=context.registry.sha256,
                source_commit=source_commit,
                input_records=records,
                parameters=episode_parameters,
            )
            verify_date_checkpoint(
                partition,
                expected_fingerprint=expected,
                verify_inputs=True,
            )
            verified_episode_dates += 1

    episode_index = _episode_index_partition(context)
    episode_index_complete = episode_index.exists()
    if episode_index_complete:
        if anchor_ids is None or verified_episode_dates != len(context.sessions):
            raise ValueError("episode index exists before all Date checkpoints")
        index_records = _episode_index_records(context)
        expected = checkpoint_fingerprint(
            phase="episode_index",
            date="frozen_131_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=index_records,
            parameters=_episode_index_parameters(anchor_ids),
        )
        verify_date_checkpoint(
            episode_index,
            expected_fingerprint=expected,
            verify_inputs=True,
        )
    boundary_checkpoints = (
        ("inputs", _boundary_inputs_partition(context), "boundary_inputs"),
        ("base_panel", _boundary_base_partition(context), "boundary_base_panel"),
        (
            "base_ranking",
            _boundary_base_ranking_partition(context),
            "boundary_base_ranking",
        ),
        ("final_panel", _boundary_final_partition(context), "boundary_final_panel"),
        (
            "selection",
            _boundary_selection_partition(context),
            "boundary_selection",
        ),
    )
    boundary_status: dict[str, bool] = {}
    prior_complete = True
    for name, partition, expected_phase in boundary_checkpoints:
        exists = partition.exists()
        boundary_status[f"boundary_{name}_complete"] = exists
        if exists and not prior_complete:
            raise ValueError(f"boundary checkpoint {name} exists before its dependency")
        if exists:
            marker = verify_date_checkpoint(partition, verify_inputs=True)
            if (
                marker["phase"] != expected_phase
                or marker["registry_sha256"] != context.registry.sha256
                or marker["source_commit"] != source_commit
            ):
                raise ValueError(f"boundary checkpoint lineage mismatch: {partition}")
        prior_complete = prior_complete and exists
    convergence_status: dict[str, object] = {
        "convergence_fact_dates_complete": 0,
        "convergence_fact_dates_expected": len(context.sessions),
        "convergence_facts_index_complete": False,
        "convergence_prediction_dates_complete": 0,
        "convergence_prediction_dates_expected": len(context.sessions),
        "convergence_predictions_index_complete": False,
        "convergence_review_complete": False,
        "final_publish_complete": False,
    }
    facts_root = selected_paths.resolved_work_root / "convergence" / "facts" / "dates"
    predictions_root = (
        selected_paths.resolved_work_root / "convergence" / "predictions" / "dates"
    )
    _reject_unfrozen_checkpoint_dates(facts_root, allowed_dates=context.sessions)
    _reject_unfrozen_checkpoint_dates(
        predictions_root,
        allowed_dates=context.sessions,
    )
    convergence_exists = facts_root.exists() or predictions_root.exists()
    if convergence_exists and not boundary_status["boundary_selection_complete"]:
        raise ValueError("convergence checkpoints exist before boundary selection")
    if boundary_status["boundary_selection_complete"]:
        selected_anchor, q_ids, _ = _load_selected_q_context(context)
        fact_parameters = _convergence_fact_parameters(selected_anchor, q_ids)
        fact_count = 0
        for date in context.sessions:
            partition = _convergence_facts_date_partition(context, date)
            if not partition.exists():
                continue
            records = _convergence_fact_records(context, date)
            expected = checkpoint_fingerprint(
                phase="convergence_facts",
                date=date,
                registry_sha256=context.registry.sha256,
                source_commit=source_commit,
                input_records=records,
                parameters=fact_parameters,
            )
            verify_date_checkpoint(
                partition,
                expected_fingerprint=expected,
                verify_inputs=True,
            )
            fact_count += 1
        convergence_status["convergence_fact_dates_complete"] = fact_count

        facts_index = _convergence_facts_index_partition(context)
        facts_index_complete = facts_index.exists()
        if facts_index_complete:
            if fact_count != len(context.sessions):
                raise ValueError("convergence facts index exists before all Dates")
            records = build_file_records(
                [
                    (
                        _convergence_facts_date_partition(context, date)
                        / "complete.json",
                        "convergence_facts_date_marker",
                    )
                    for date in context.sessions
                ]
            )
            parameters = {
                **fact_parameters,
                "expected_date_checkpoints": EXPECTED_SESSION_COUNT,
            }
            expected = checkpoint_fingerprint(
                phase="convergence_facts_index",
                date="frozen_131_sessions",
                registry_sha256=context.registry.sha256,
                source_commit=source_commit,
                input_records=records,
                parameters=parameters,
            )
            verify_date_checkpoint(
                facts_index,
                expected_fingerprint=expected,
                verify_inputs=True,
            )
        convergence_status["convergence_facts_index_complete"] = facts_index_complete

        prediction_parameters = _convergence_prediction_parameters(
            selected_anchor,
            q_ids,
        )
        prediction_count = 0
        for date in context.sessions:
            partition = _convergence_prediction_date_partition(context, date)
            if not partition.exists():
                continue
            if not facts_index_complete:
                raise ValueError("convergence predictions exist before the facts index")
            records = _convergence_prediction_records(context, date)
            expected = checkpoint_fingerprint(
                phase="convergence_predictions",
                date=date,
                registry_sha256=context.registry.sha256,
                source_commit=source_commit,
                input_records=records,
                parameters=prediction_parameters,
            )
            verify_date_checkpoint(
                partition,
                expected_fingerprint=expected,
                verify_inputs=True,
            )
            prediction_count += 1
        convergence_status["convergence_prediction_dates_complete"] = prediction_count

        predictions_index = _convergence_prediction_index_partition(context)
        predictions_index_complete = predictions_index.exists()
        if predictions_index_complete:
            if prediction_count != len(context.sessions):
                raise ValueError(
                    "convergence predictions index exists before all Dates"
                )
            records = build_file_records(
                [
                    (
                        _convergence_prediction_date_partition(context, date)
                        / "complete.json",
                        "convergence_predictions_date_marker",
                    )
                    for date in context.sessions
                ]
            )
            parameters = {
                **prediction_parameters,
                "expected_date_checkpoints": EXPECTED_SESSION_COUNT,
            }
            expected = checkpoint_fingerprint(
                phase="convergence_predictions_index",
                date="frozen_131_sessions",
                registry_sha256=context.registry.sha256,
                source_commit=source_commit,
                input_records=records,
                parameters=parameters,
            )
            verify_date_checkpoint(
                predictions_index,
                expected_fingerprint=expected,
                verify_inputs=True,
            )
        convergence_status["convergence_predictions_index_complete"] = (
            predictions_index_complete
        )

        review = _convergence_review_partition(context)
        review_complete = review.exists()
        if review_complete:
            if not predictions_index_complete:
                raise ValueError("convergence review exists before predictions index")
            marker = verify_date_checkpoint(review, verify_inputs=True)
            if (
                marker["phase"] != "convergence_review"
                or marker["registry_sha256"] != context.registry.sha256
                or marker["source_commit"] != source_commit
            ):
                raise ValueError("convergence review checkpoint lineage mismatch")
        convergence_status["convergence_review_complete"] = review_complete

        final_complete = selected_paths.output_root.exists()
        if final_complete:
            if not review_complete:
                raise ValueError("final publication exists before convergence review")
            marker = verify_date_checkpoint(
                selected_paths.output_root,
                verify_inputs=True,
            )
            if (
                marker["phase"] != "final_publish"
                or marker["registry_sha256"] != context.registry.sha256
                or marker["source_commit"] != source_commit
            ):
                raise ValueError("final publication checkpoint lineage mismatch")
        convergence_status["final_publish_complete"] = final_complete
    return {
        "runner_version": RUNNER_VERSION,
        "registry_sha256": context.registry.sha256,
        "source_commit": source_commit,
        "preflight_complete": True,
        "anchor_dates_complete": verified_anchor_dates,
        "anchor_dates_expected": len(context.primary_sessions),
        "anchor_selection_complete": selection_complete,
        "episode_dates_complete": verified_episode_dates,
        "episode_dates_expected": len(context.sessions),
        "episode_index_complete": episode_index_complete,
        **boundary_status,
        **convergence_status,
        "protected_forward_start": PROTECTED_FORWARD_START_DATE,
        "development_only": True,
    }


def _require_boundary_columns(
    frame: pl.DataFrame,
    columns: Sequence[str] | set[str],
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _normalise_boundary_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in candidates))
    if not result or len(result) != len(candidates):
        raise ValueError("boundary candidates must be nonempty and unique")
    return result


def _boundary_native_coverage(
    predictions: pl.DataFrame,
    target_mapping: pl.DataFrame,
    *,
    candidates: Sequence[str],
) -> pl.DataFrame:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "candidate_id",
        "tod_bucket",
        "boundary_quantile",
        "side",
        "boundary_distance_bp",
        "native_supported",
    }
    _require_boundary_columns(predictions, required, "boundary predictions")
    _require_boundary_columns(
        target_mapping,
        {"Date", "ValueCode", "QuoteCode"},
        "boundary target mapping",
    )
    candidate_ids = _normalise_boundary_candidates(candidates)
    mapping = target_mapping.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    duplicate = (
        mapping.group_by(["Date", "ValueCode", "QuoteCode"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("boundary target mapping contains duplicate contracts")
    mapping = mapping.unique()
    expected_units = mapping.height * len(DEFAULT_TOD_BUCKETS)
    if expected_units <= 0:
        raise ValueError("boundary target mapping is empty")
    panel = predictions.filter(
        pl.col("candidate_id").cast(pl.String).is_in(candidate_ids)
    ).with_columns(
        pl.col("candidate_id").cast(pl.String),
        pl.col("native_supported").fill_null(False).cast(pl.Boolean),
        pl.col("boundary_distance_bp").cast(pl.Float64),
    )
    cell_keys = [
        "candidate_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "tod_bucket",
    ]
    cells = (
        panel.group_by(cell_keys)
        .agg(
            pl.len().alias("rows"),
            pl.col("boundary_quantile").n_unique().alias("quantiles"),
            pl.col("side").n_unique().alias("sides"),
            pl.col("native_supported").all().alias("all_native"),
            (
                pl.col("boundary_distance_bp").is_not_null()
                & pl.col("boundary_distance_bp").is_finite()
                & (pl.col("boundary_distance_bp") > 0)
            )
            .all()
            .alias("all_finite_positive"),
        )
        .with_columns(
            (
                (pl.col("rows") == 6)
                & (pl.col("quantiles") == 3)
                & (pl.col("sides") == 2)
                & pl.col("all_native")
                & pl.col("all_finite_positive")
            ).alias("native_all_q")
        )
    )
    rows: list[dict[str, object]] = []
    for candidate in candidate_ids:
        native = cells.filter(
            (pl.col("candidate_id") == candidate) & pl.col("native_all_q")
        ).height
        rows.append(
            {
                "candidate_id": candidate,
                "native_all_q_units": native,
                "expected_product_day_tod_units": expected_units,
                "native_all_q_coverage": native / expected_units,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("candidate_id")


def _boundary_turnover(
    predictions: pl.DataFrame,
    *,
    candidates: Sequence[str],
    sessions: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidate_ids = _normalise_boundary_candidates(candidates)
    session_order = {str(date): index for index, date in enumerate(sessions)}
    panel = (
        predictions.filter(pl.col("candidate_id").cast(pl.String).is_in(candidate_ids))
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("candidate_id").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("boundary_distance_bp").cast(pl.Float64),
        )
        .filter(
            pl.col("boundary_distance_bp").is_not_null()
            & pl.col("boundary_distance_bp").is_finite()
            & (pl.col("boundary_distance_bp") > 0)
            & pl.col("Date").is_in(tuple(session_order))
        )
        .with_columns(
            pl.col("Date")
            .replace_strict(session_order, return_dtype=pl.Int32)
            .alias("_session_index")
        )
    )
    group_keys = [
        "candidate_id",
        "ValueCode",
        "tod_bucket",
        "boundary_quantile",
        "side",
    ]
    ordered = panel.sort([*group_keys, "_session_index"]).with_columns(
        pl.col("_session_index").diff().over(group_keys).alias("_session_difference"),
        pl.col("boundary_distance_bp")
        .log()
        .diff()
        .abs()
        .over(group_keys)
        .alias("_absolute_log_ratio"),
    )
    by_date = (
        ordered.filter(pl.col("_session_difference") == 1)
        .group_by(["candidate_id", "Date"])
        .agg(
            pl.col("_absolute_log_ratio")
            .median()
            .alias("date_median_absolute_log_ratio"),
            pl.len().alias("adjacent_boundary_rows"),
        )
        .sort(["candidate_id", "Date"])
    )
    summary = by_date.group_by("candidate_id").agg(
        pl.col("date_median_absolute_log_ratio").mean().alias("boundary_turnover"),
        pl.col("Date").n_unique().alias("turnover_dates"),
        pl.col("adjacent_boundary_rows").sum(),
    )
    return by_date, summary


def summarize_boundary_candidate_scores(
    calibration: pl.DataFrame,
    predictions: pl.DataFrame,
    target_mapping: pl.DataFrame,
    *,
    candidates: Sequence[str],
    sessions: Sequence[str],
    allow_primary_selection: Mapping[str, bool] | None = None,
    minimum_cross_sectional_products: int = 20,
    minimum_temporal_dates: int = 20,
) -> BoundaryRankingFacts:
    """Apply the frozen common-support and Date/month-equal q score reduction."""

    candidate_ids = _normalise_boundary_candidates(candidates)
    required_calibration = {
        *BOUNDARY_UNIT_COLUMNS,
        "candidate_id",
        "predicted_distance_bp",
        "realized_completed_quantile_bp",
        "observable_started",
        "reach_interval_distance",
    }
    _require_boundary_columns(
        calibration,
        required_calibration,
        "boundary calibration",
    )
    panel = calibration.filter(
        pl.col("candidate_id").cast(pl.String).is_in(candidate_ids)
        & (pl.col("observable_started") > 0)
    ).with_columns(
        pl.col("candidate_id").cast(pl.String),
        pl.col("Date").cast(pl.String),
        pl.col("reach_interval_distance").cast(pl.Float64),
        pl.col("predicted_distance_bp").cast(pl.Float64),
        pl.col("realized_completed_quantile_bp").cast(pl.Float64),
    )
    common = filter_common_support(
        panel,
        candidates=candidate_ids,
        unit_columns=BOUNDARY_UNIT_COLUMNS,
        candidate_column="candidate_id",
        required_value_columns=("reach_interval_distance",),
    ).with_columns(pl.col("Date").str.slice(0, 6).alias("month"))
    if common.is_empty():
        raise ValueError("selectable q candidates have no common calibration support")

    daily_q_side = common.group_by(
        ["candidate_id", "month", "Date", "boundary_quantile", "side"]
    ).agg(
        pl.col("reach_interval_distance")
        .mean()
        .alias("date_product_day_tod_equal_loss"),
        pl.len().alias("common_product_day_tod_units"),
    )
    monthly_q_side = (
        daily_q_side.group_by(["candidate_id", "month", "boundary_quantile", "side"])
        .agg(
            pl.col("date_product_day_tod_equal_loss")
            .mean()
            .alias("month_date_equal_loss"),
            pl.col("Date").n_unique().alias("dates"),
            pl.col("common_product_day_tod_units").sum(),
        )
        .sort(["candidate_id", "month", "boundary_quantile", "side"])
    )
    primary_loss = monthly_q_side.group_by("candidate_id").agg(
        pl.col("month_date_equal_loss").max().alias("primary_loss")
    )
    worst_month = (
        monthly_q_side.group_by(["candidate_id", "month"])
        .agg(pl.col("month_date_equal_loss").mean().alias("month_q_side_equal_loss"))
        .group_by("candidate_id")
        .agg(pl.col("month_q_side_equal_loss").max().alias("worst_month_loss"))
    )

    amplitude_common = filter_common_support(
        common.drop("month"),
        candidates=candidate_ids,
        unit_columns=BOUNDARY_UNIT_COLUMNS,
        candidate_column="candidate_id",
        required_value_columns=(
            "predicted_distance_bp",
            "realized_completed_quantile_bp",
        ),
    ).with_columns(
        (pl.col("predicted_distance_bp") - pl.col("realized_completed_quantile_bp"))
        .abs()
        .alias("amplitude_absolute_error_row_bp")
    )
    amplitude = (
        amplitude_common.group_by(["candidate_id", "Date"])
        .agg(
            pl.col("amplitude_absolute_error_row_bp")
            .mean()
            .alias("date_amplitude_absolute_error_bp")
        )
        .group_by("candidate_id")
        .agg(
            pl.col("date_amplitude_absolute_error_bp")
            .mean()
            .alias("amplitude_absolute_error_bp"),
            pl.col("Date").n_unique().alias("amplitude_dates"),
        )
    )

    daily_spearman = daily_cross_sectional_spearman(
        amplitude_common,
        predicted_column="predicted_distance_bp",
        realized_column="realized_completed_quantile_bp",
        extra_group_columns=(
            "candidate_id",
            "tod_bucket",
            "boundary_quantile",
            "side",
        ),
        minimum_products=minimum_cross_sectional_products,
    )
    daily_spearman_summary = (
        daily_spearman.filter(
            pl.col("daily_cross_product_spearman").is_not_null()
            & pl.col("daily_cross_product_spearman").is_finite()
        )
        .group_by("candidate_id")
        .agg(
            pl.col("daily_cross_product_spearman")
            .mean()
            .alias("daily_cross_product_spearman"),
            pl.len().alias("finite_daily_spearman_cells"),
        )
    )
    temporal_spearman = temporal_product_spearman(
        amplitude_common,
        predicted_column="predicted_distance_bp",
        realized_column="realized_completed_quantile_bp",
        extra_group_columns=(
            "candidate_id",
            "tod_bucket",
            "boundary_quantile",
            "side",
        ),
        product_columns=("ValueCode",),
        minimum_dates=minimum_temporal_dates,
    )
    temporal_summary = (
        temporal_spearman.filter(
            pl.col("temporal_product_spearman").is_not_null()
            & pl.col("temporal_product_spearman").is_finite()
        )
        .group_by("candidate_id")
        .agg(
            pl.col("temporal_product_spearman")
            .mean()
            .alias("mean_temporal_product_spearman"),
            pl.len().alias("finite_temporal_spearman_cells"),
        )
    )
    coverage = _boundary_native_coverage(
        predictions,
        target_mapping,
        candidates=candidate_ids,
    )
    common_unit_keys = [*BOUNDARY_UNIT_COLUMNS, "candidate_id"]
    turnover_predictions = predictions.join(
        common.select(*common_unit_keys).unique(),
        on=common_unit_keys,
        how="semi",
    )
    turnover_by_date, turnover = _boundary_turnover(
        turnover_predictions,
        candidates=candidate_ids,
        sessions=sessions,
    )
    support = native_support_summary(
        panel,
        candidates=candidate_ids,
        unit_columns=BOUNDARY_UNIT_COLUMNS,
        candidate_column="candidate_id",
        required_value_columns=("reach_interval_distance",),
    )

    selection_flags = allow_primary_selection or {
        candidate: True for candidate in candidate_ids
    }
    unknown_flags = sorted(set(candidate_ids) - set(selection_flags))
    if unknown_flags:
        raise ValueError(f"allow_primary_selection missing candidates: {unknown_flags}")
    flags = pl.from_dicts(
        [
            {
                "candidate_id": candidate,
                "allow_primary_selection": bool(selection_flags[candidate]),
            }
            for candidate in candidate_ids
        ],
        infer_schema_length=None,
    )
    scores = (
        primary_loss.join(worst_month, on="candidate_id", how="inner")
        .join(amplitude, on="candidate_id", how="inner")
        .join(coverage, on="candidate_id", how="inner")
        .join(daily_spearman_summary, on="candidate_id", how="inner")
        .join(temporal_summary, on="candidate_id", how="left")
        .join(turnover, on="candidate_id", how="inner")
        .join(support, on="candidate_id", how="inner")
        .join(flags, on="candidate_id", how="inner")
        .sort("candidate_id")
    )
    if scores.height != len(candidate_ids):
        missing_scores = sorted(
            set(candidate_ids) - set(scores["candidate_id"].to_list())
        )
        raise ValueError(
            f"boundary score reduction lacks finite candidate metrics: {missing_scores}"
        )
    return BoundaryRankingFacts(
        candidate_scores=scores,
        common_calibration=common.sort([*BOUNDARY_UNIT_COLUMNS, "candidate_id"]),
        monthly_q_side_loss=monthly_q_side,
        daily_spearman=daily_spearman,
        temporal_spearman=temporal_spearman,
        support_summary=support,
        turnover_by_date=turnover_by_date,
    )


def build_selected_q_payload(
    ranking: pl.DataFrame,
    *,
    base_tod_source_candidates: Sequence[str],
) -> dict[str, object]:
    """Freeze final q rank1/rank2 identities with their TOD-source lineage."""

    _require_boundary_columns(
        ranking,
        {"candidate_id", "selection_rank"},
        "boundary candidate ranking",
    )
    ranked = ranking.sort("selection_rank")
    if ranked.height < 2 or ranked["selection_rank"].head(2).to_list() != [1, 2]:
        raise ValueError("boundary ranking must contain exact ranks one and two")
    rank1 = str(ranked.row(0, named=True)["candidate_id"])
    rank2 = str(ranked.row(1, named=True)["candidate_id"])
    tod_sources = _normalise_boundary_candidates(base_tod_source_candidates)
    if len(tod_sources) != 2:
        raise ValueError("TOD adaptation requires exact top-two base candidates")
    return {
        "rank1_candidate_id": rank1,
        "rank2_candidate_id": rank2,
        "base_tod_source_candidates": list(tod_sources),
        "selected_candidate_ids": [rank1, rank2],
        "selection_scope": "rank1_anchor_primary_71_development_sessions",
        "contains_target_day_outcome": True,
        "prediction_contract": "strict_D_safe_effective_boundary",
        "development_only": True,
    }


def _boundary_inputs_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "boundary" / "inputs"


def _boundary_base_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "boundary" / "base_panel"


def _boundary_base_ranking_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "boundary" / "base_ranking"


def _boundary_final_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "boundary" / "final_panel"


def _boundary_selection_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "boundary" / "selection"


def _boundary_input_records(context: RunContext) -> list[dict[str, object]]:
    records: list[tuple[Path, str]] = [
        (
            _episode_index_partition(context) / "complete.json",
            "episode_index_checkpoint_marker",
        ),
        (
            _anchor_selection_partition(context) / "complete.json",
            "selected_anchor_checkpoint_marker",
        ),
    ]
    for date in context.sessions:
        daily = _daily_partition(context.paths, date)
        records.extend(
            [
                (daily / "complete.json", "daily_completion_marker"),
                (daily / "mapping.parquet", "daily_exact_contract_mapping"),
            ]
        )
    return build_file_records(records)


def _boundary_input_parameters(anchor_ids: Sequence[str]) -> dict[str, object]:
    return {
        "target_dates": [SOURCE_START_DATE, SOURCE_END_DATE],
        "session_count": EXPECTED_SESSION_COUNT,
        "selected_anchor_id": anchor_ids[0],
        "diagnostic_anchor_ids": list(anchor_ids[1:]),
        "mapping_columns": list(MAPPING_READ_COLUMNS),
        "episode_columns": list(BOUNDARY_EPISODE_COLUMNS),
        "protected_forward_start": PROTECTED_FORWARD_START_DATE,
    }


def _verify_episode_checkpoints_for_boundary(
    context: RunContext,
    anchor_ids: Sequence[str],
) -> None:
    index_records = _episode_index_records(context)
    index_expected = checkpoint_fingerprint(
        phase="episode_index",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=index_records,
        parameters=_episode_index_parameters(anchor_ids),
    )
    verify_date_checkpoint(
        _episode_index_partition(context),
        expected_fingerprint=index_expected,
    )
    for date in context.sessions:
        verify_date_checkpoint(_episode_date_partition(context, date))


def _build_boundary_inputs(
    context: RunContext,
    anchor_ids: Sequence[str],
    *,
    resume: bool,
) -> Mapping[str, object]:
    records = _boundary_input_records(context)
    parameters = _boundary_input_parameters(anchor_ids)
    fingerprint = checkpoint_fingerprint(
        phase="boundary_inputs",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
    )
    partition = _boundary_inputs_partition(context)
    resumed = resume_date_checkpoint(
        partition,
        expected_fingerprint=fingerprint,
        resume=resume,
    )
    if resumed is not None:
        return resumed
    mapping_parts: list[pl.DataFrame] = []
    support_parts: list[pl.DataFrame] = []
    for date in context.sessions:
        daily = _daily_partition(context.paths, date)
        mapping_parts.append(
            pl.read_parquet(
                daily / "mapping.parquet",
                columns=list(MAPPING_READ_COLUMNS),
            )
        )
        support_parts.append(
            pl.read_parquet(
                _episode_date_partition(context, date) / "anchor_support.parquet"
            )
        )
    mapping = pl.concat(mapping_parts, how="vertical_relaxed").sort(
        ["Date", "ValueCode", "QuoteCode"]
    )
    duplicate = (
        mapping.group_by(["Date", "ValueCode", "QuoteCode"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("frozen target mapping contains duplicate exact contracts")
    anchor_support = pl.concat(support_parts, how="vertical_relaxed").sort(
        ["Date", "ValueCode", "QuoteCode"]
    )
    if set(anchor_support["anchor_model_id"].unique().to_list()) != {anchor_ids[0]}:
        raise ValueError("anchor-support facts are not selected-rank1 only")
    return atomic_publish_checkpoint(
        partition,
        phase="boundary_inputs",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
        artifact_values={
            "target_mapping.parquet": mapping,
            "anchor_support.parquet": anchor_support,
        },
    )


def _load_boundary_episodes(
    context: RunContext,
    anchor_ids: Sequence[str],
) -> pl.DataFrame:
    parts = [
        pl.read_parquet(
            _episode_date_partition(context, date) / "episodes.parquet",
            columns=list(BOUNDARY_EPISODE_COLUMNS),
        )
        for date in context.sessions
    ]
    episodes = pl.concat(parts, how="vertical_relaxed").filter(
        pl.col("anchor_model_id").is_in(tuple(anchor_ids))
    )
    actual = set(episodes["anchor_model_id"].unique().to_list())
    if actual != set(anchor_ids):
        raise ValueError(
            f"boundary episode facts lack selected anchors: {sorted(set(anchor_ids) - actual)}"
        )
    return episodes.sort(
        ["Date", "anchor_model_id", "ValueCode", "start_seconds_from_open"]
    )


def _build_boundary_candidate_panel(
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str],
    anchor_model_id: str,
    tod_source_candidates: Sequence[str] = (),
) -> object:
    """Single runner-to-orchestration integration point."""

    from .foundation_boundary_orchestration import (
        BoundaryOrchestrationConfig,
        build_boundary_candidate_panel,
    )

    return build_boundary_candidate_panel(
        episodes,
        sessions,
        target_mapping,
        target_dates=target_dates,
        anchor_model_id=anchor_model_id,
        tod_source_candidates=tod_source_candidates,
        config=BoundaryOrchestrationConfig(),
    )


def _add_tod_boundary_variants(
    base_partition: Path,
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    *,
    tod_source_candidates: Sequence[str],
) -> object:
    """Rehydrate the base checkpoint and fit only incremental TOD variants."""

    from .foundation_boundary_orchestration import (
        BoundaryCandidatePanel,
        BoundaryOrchestrationConfig,
        add_tod_variants,
    )

    meta = json.loads(
        (Path(base_partition) / "pipeline_meta.json").read_text(encoding="utf-8")
    )
    if not isinstance(meta, Mapping):
        raise TypeError("base boundary pipeline metadata must be an object")
    base_panel = BoundaryCandidatePanel(
        raw_predictions=pl.read_parquet(
            Path(base_partition) / "raw_predictions.parquet"
        ),
        predictions=pl.read_parquet(Path(base_partition) / "predictions.parquet"),
        calibration=pl.read_parquet(Path(base_partition) / "calibration.parquet"),
        multipliers=pl.read_parquet(Path(base_partition) / "multipliers.parquet"),
        support_audit=pl.read_parquet(Path(base_partition) / "support_audit.parquet"),
        build_dates=tuple(str(value) for value in meta["build_dates"]),
        target_dates=tuple(str(value) for value in meta["target_dates"]),
        anchor_model_id=str(meta["anchor_model_id"]),
    )
    return add_tod_variants(
        base_panel,
        episodes,
        sessions,
        tod_source_candidates=tod_source_candidates,
        config=BoundaryOrchestrationConfig(),
    )


def _complete_native_diagnostic_predictions(raw: pl.DataFrame) -> pl.DataFrame:
    if raw.is_empty():
        return raw
    cell_keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "anchor_model_id",
        "candidate_id",
        "tod_bucket",
    ]
    valid = raw.filter(
        pl.col("native_supported").fill_null(False)
        & pl.col("boundary_distance_bp").is_not_null()
        & pl.col("boundary_distance_bp").is_finite()
        & (pl.col("boundary_distance_bp") > 0)
        & pl.col("source_asof_date").is_not_null()
    )
    complete = (
        valid.group_by(cell_keys)
        .agg(
            pl.len().alias("rows"),
            pl.col("boundary_quantile").n_unique().alias("quantiles"),
            pl.col("side").n_unique().alias("sides"),
        )
        .filter(
            (pl.col("rows") == 6) & (pl.col("quantiles") == 3) & (pl.col("sides") == 2)
        )
    )
    return valid.join(
        complete.select(*cell_keys),
        on=cell_keys,
        how="semi",
    ).sort([*BOUNDARY_UNIT_COLUMNS, "candidate_id"])


def _build_diagnostic_boundary_panel(
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str],
    anchor_ids: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build exactly Q1/Q2 for rank2 and count120 diagnostic anchors."""

    from .foundation_boundary_adaptation import score_boundary_predictions
    from .foundation_boundary_batch import build_boundary_batch_predictions

    raw_parts: list[pl.DataFrame] = []
    prediction_parts: list[pl.DataFrame] = []
    calibration_parts: list[pl.DataFrame] = []
    for anchor_id in anchor_ids:
        anchor_episodes = episodes.filter(pl.col("anchor_model_id") == anchor_id)
        raw = build_boundary_batch_predictions(
            anchor_episodes,
            sessions,
            target_mapping,
            target_dates=target_dates,
            candidate_ids=DIAGNOSTIC_BOUNDARY_CANDIDATES,
        )
        predictions = _complete_native_diagnostic_predictions(raw)
        if not predictions.is_empty():
            prediction_dates = predictions["Date"].unique().to_list()
            score_episodes = anchor_episodes.filter(
                pl.col("Date").is_in(prediction_dates)
            )
            calibration_parts.append(
                score_boundary_predictions(predictions, score_episodes)
            )
        raw_parts.append(raw)
        prediction_parts.append(predictions)
    raw_all = pl.concat(raw_parts, how="diagonal_relaxed")
    predictions_all = pl.concat(prediction_parts, how="diagonal_relaxed")
    if calibration_parts:
        calibration_all = pl.concat(calibration_parts, how="diagonal_relaxed")
    else:
        from .foundation_boundary_adaptation import SCORE_SCHEMA

        calibration_all = pl.DataFrame(schema=SCORE_SCHEMA)
    return raw_all, predictions_all, calibration_all


def _boundary_pipeline_parameters(
    *,
    rank1_anchor_id: str,
    diagnostic_anchor_ids: Sequence[str],
    tod_source_candidates: Sequence[str],
) -> dict[str, object]:
    return {
        "rank1_anchor_id": rank1_anchor_id,
        "rank1_raw_candidates": list(RAW_RANK1_BOUNDARY_CANDIDATES),
        "rank1_adapted_candidates": [
            "Q3_shape60_level5",
            "Q4_shape60_level10",
        ],
        "q6_fallback": "Q1_trail60_date_equal",
        "diagnostic_anchor_ids": list(diagnostic_anchor_ids),
        "diagnostic_candidates": list(DIAGNOSTIC_BOUNDARY_CANDIDATES),
        "tod_source_candidates": list(tod_source_candidates),
        "tod_suffix": "__tod10",
        "target_dates": [SOURCE_START_DATE, SOURCE_END_DATE],
        "minimum_completed_per_side": {"50": 100, "80": 200, "95": 400},
        "q3": {"sessions": 5, "minimum_dates": 4, "minimum_cells": 500},
        "q4": {"sessions": 10, "minimum_dates": 8, "minimum_cells": 1_000},
        "tod": {"sessions": 10, "minimum_dates": 8, "minimum_cells": 100},
        "legacy_q_fallback_allowed": False,
        "checkpoint_granularity": "full_131_session_panel",
        "intermediate_date_resume_available": False,
        "orchestration_progress_callback_available": False,
    }


def _pipeline_artifacts(panel: object) -> dict[str, object]:
    required = (
        "raw_predictions",
        "predictions",
        "calibration",
        "multipliers",
        "support_audit",
        "build_dates",
        "target_dates",
        "anchor_model_id",
    )
    missing = [name for name in required if not hasattr(panel, name)]
    if missing:
        raise TypeError(f"boundary orchestration result missing fields: {missing}")
    return {
        "raw_predictions.parquet": panel.raw_predictions,
        "predictions.parquet": panel.predictions,
        "calibration.parquet": panel.calibration,
        "multipliers.parquet": panel.multipliers,
        "support_audit.parquet": panel.support_audit,
        "pipeline_meta.json": {
            "anchor_model_id": str(panel.anchor_model_id),
            "build_dates": list(panel.build_dates),
            "target_dates": list(panel.target_dates),
            "contains_target_day_outcome": bool(panel.calibration.height > 0),
        },
    }


def _top_two_candidate_ids(ranking: pl.DataFrame) -> tuple[str, str]:
    _require_boundary_columns(
        ranking,
        {"candidate_id", "selection_rank"},
        "boundary ranking",
    )
    selected = ranking.sort("selection_rank").head(2)
    if selected.height != 2 or selected["selection_rank"].to_list() != [1, 2]:
        raise ValueError("boundary ranking lacks exact top-two candidates")
    ids = tuple(str(value) for value in selected["candidate_id"].to_list())
    if len(set(ids)) != 2:
        raise ValueError("boundary ranking top two are not distinct")
    return ids[0], ids[1]


def _boundary_ranking_artifacts(
    facts: BoundaryRankingFacts,
    ranking: pl.DataFrame,
    selected_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate_scores.parquet": facts.candidate_scores,
        "candidate_ranking.parquet": ranking,
        "common_calibration.parquet": facts.common_calibration,
        "monthly_q_side_loss.parquet": facts.monthly_q_side_loss,
        "daily_spearman.parquet": facts.daily_spearman,
        "temporal_spearman.parquet": facts.temporal_spearman,
        "support_summary.parquet": facts.support_summary,
        "turnover_by_date.parquet": facts.turnover_by_date,
        "selected_q.json": dict(selected_payload),
    }


def run_boundary_phase(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Build, causally adapt, score, and rank the frozen boundary grid."""

    selected_paths = paths or SelectionPaths()
    run_preflight(selected_paths, resume=resume, canonical=canonical)
    context = _context(selected_paths, canonical=canonical)
    initial_git_state = dict(context.git_state)
    source_commit = str(context.git_state["commit"])
    anchor_ids = load_selected_anchor_ids(
        selected_paths,
        expected_registry_sha256=context.registry.sha256,
        expected_source_commit=source_commit,
    )
    _verify_episode_checkpoints_for_boundary(context, anchor_ids)
    _build_boundary_inputs(context, anchor_ids, resume=resume)
    target_mapping = pl.read_parquet(
        _boundary_inputs_partition(context) / "target_mapping.parquet"
    )
    primary_mapping = target_mapping.filter(
        pl.col("Date").is_in(context.primary_sessions)
    )
    episodes = _load_boundary_episodes(context, anchor_ids)
    shared_records = build_file_records(
        [
            (
                _boundary_inputs_partition(context) / "complete.json",
                "boundary_input_checkpoint_marker",
            ),
            (
                _episode_index_partition(context) / "complete.json",
                "episode_index_checkpoint_marker",
            ),
        ]
    )

    base_parameters = _boundary_pipeline_parameters(
        rank1_anchor_id=anchor_ids[0],
        diagnostic_anchor_ids=anchor_ids[1:],
        tod_source_candidates=(),
    )
    base_fingerprint = checkpoint_fingerprint(
        phase="boundary_base_panel",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=shared_records,
        parameters=base_parameters,
    )
    base_partition = _boundary_base_partition(context)
    base_marker = resume_date_checkpoint(
        base_partition,
        expected_fingerprint=base_fingerprint,
        resume=resume,
    )
    if base_marker is None:
        print(
            "boundary base panel started (131 sessions; atomic marker is written "
            "only after the orchestration call returns)",
            flush=True,
        )
        rank1_episodes = episodes.filter(pl.col("anchor_model_id") == anchor_ids[0])
        base_panel = _build_boundary_candidate_panel(
            rank1_episodes,
            context.sessions,
            target_mapping,
            target_dates=context.sessions,
            anchor_model_id=anchor_ids[0],
        )
        print(
            "boundary rank1 base panel complete; diagnostic anchors started",
            flush=True,
        )
        diagnostic_raw, diagnostic_predictions, diagnostic_calibration = (
            _build_diagnostic_boundary_panel(
                episodes,
                context.sessions,
                target_mapping,
                target_dates=context.sessions,
                anchor_ids=anchor_ids[1:],
            )
        )
        artifacts = _pipeline_artifacts(base_panel)
        artifacts.update(
            {
                "diagnostic_raw_predictions.parquet": diagnostic_raw,
                "diagnostic_predictions.parquet": diagnostic_predictions,
                "diagnostic_calibration.parquet": diagnostic_calibration,
            }
        )
        base_marker = atomic_publish_checkpoint(
            base_partition,
            phase="boundary_base_panel",
            date="frozen_131_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=shared_records,
            parameters=base_parameters,
            artifact_values=artifacts,
        )
        print("boundary base panel checkpoint published", flush=True)

    base_ranking_records = build_file_records(
        [
            (
                base_partition / "complete.json",
                "boundary_base_panel_checkpoint_marker",
            )
        ]
    )
    base_ranking_parameters = {
        "candidate_ids": list(SELECTABLE_BASE_BOUNDARY_CANDIDATES),
        "selection_dates": list(context.primary_sessions),
        "common_support": "all_selectable_base_candidates",
        "score_reduction": "registry_lexicographic",
    }
    base_ranking_fingerprint = checkpoint_fingerprint(
        phase="boundary_base_ranking",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=base_ranking_records,
        parameters=base_ranking_parameters,
    )
    base_ranking_partition = _boundary_base_ranking_partition(context)
    base_ranking_marker = resume_date_checkpoint(
        base_ranking_partition,
        expected_fingerprint=base_ranking_fingerprint,
        resume=resume,
    )
    if base_ranking_marker is None:
        base_predictions = pl.read_parquet(
            base_partition / "predictions.parquet"
        ).filter(pl.col("Date").is_in(context.primary_sessions))
        base_calibration = pl.read_parquet(
            base_partition / "calibration.parquet"
        ).filter(pl.col("Date").is_in(context.primary_sessions))
        base_facts = summarize_boundary_candidate_scores(
            base_calibration,
            base_predictions,
            primary_mapping,
            candidates=SELECTABLE_BASE_BOUNDARY_CANDIDATES,
            sessions=context.primary_sessions,
        )
        base_ranking = rank_q_candidates(
            base_facts.candidate_scores,
            candidate_column="candidate_id",
            simpler_order=BOUNDARY_SIMPLICITY_ORDER,
        )
        base_top_two = _top_two_candidate_ids(base_ranking)
        base_payload = {
            "rank1_candidate_id": base_top_two[0],
            "rank2_candidate_id": base_top_two[1],
            "tod_source_candidates": list(base_top_two),
            "selection_scope": "rank1_anchor_primary_71_base_candidates",
            "selection_uses_development_outcomes": True,
            "development_only": True,
        }
        base_ranking_marker = atomic_publish_checkpoint(
            base_ranking_partition,
            phase="boundary_base_ranking",
            date="primary_71_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=base_ranking_records,
            parameters=base_ranking_parameters,
            artifact_values=_boundary_ranking_artifacts(
                base_facts,
                base_ranking,
                base_payload,
            ),
        )
    base_payload = json.loads(
        (base_ranking_partition / "selected_q.json").read_text(encoding="utf-8")
    )
    tod_sources = tuple(str(value) for value in base_payload["tod_source_candidates"])
    if len(tod_sources) != 2:
        raise ValueError("base boundary selection did not freeze two TOD sources")

    final_records = build_file_records(
        [
            *[
                (_resolve_file_record_path(record), str(record["role"]))
                for record in shared_records
            ],
            (
                base_ranking_partition / "complete.json",
                "boundary_base_ranking_checkpoint_marker",
            ),
        ]
    )
    final_parameters = _boundary_pipeline_parameters(
        rank1_anchor_id=anchor_ids[0],
        diagnostic_anchor_ids=anchor_ids[1:],
        tod_source_candidates=tod_sources,
    )
    final_fingerprint = checkpoint_fingerprint(
        phase="boundary_final_panel",
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=final_records,
        parameters=final_parameters,
    )
    final_partition = _boundary_final_partition(context)
    final_marker = resume_date_checkpoint(
        final_partition,
        expected_fingerprint=final_fingerprint,
        resume=resume,
    )
    if final_marker is None:
        print(
            "boundary incremental TOD panel started (131 sessions; base raw/Q3/Q4 "
            "artifacts are reused)",
            flush=True,
        )
        final_panel = _add_tod_boundary_variants(
            base_partition,
            episodes.filter(pl.col("anchor_model_id") == anchor_ids[0]),
            context.sessions,
            tod_source_candidates=tod_sources,
        )
        artifacts = _pipeline_artifacts(final_panel)
        artifacts.update(
            {
                "diagnostic_raw_predictions.parquet": pl.read_parquet(
                    base_partition / "diagnostic_raw_predictions.parquet"
                ),
                "diagnostic_predictions.parquet": pl.read_parquet(
                    base_partition / "diagnostic_predictions.parquet"
                ),
                "diagnostic_calibration.parquet": pl.read_parquet(
                    base_partition / "diagnostic_calibration.parquet"
                ),
            }
        )
        final_marker = atomic_publish_checkpoint(
            final_partition,
            phase="boundary_final_panel",
            date="frozen_131_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=final_records,
            parameters=final_parameters,
            artifact_values=artifacts,
        )
        print("boundary final TOD checkpoint published", flush=True)

    selection_records = build_file_records(
        [
            (
                final_partition / "complete.json",
                "boundary_final_panel_checkpoint_marker",
            ),
            (
                base_ranking_partition / "complete.json",
                "boundary_base_ranking_checkpoint_marker",
            ),
        ]
    )
    final_candidates = (
        *SELECTABLE_BASE_BOUNDARY_CANDIDATES,
        *(f"{candidate}__tod10" for candidate in tod_sources),
    )
    selection_parameters = {
        "candidate_ids": list(final_candidates),
        "selection_dates": list(context.primary_sessions),
        "base_tod_source_candidates": list(tod_sources),
        "common_support": "all_selectable_final_candidates",
        "score_reduction": "registry_lexicographic",
    }
    selection_fingerprint = checkpoint_fingerprint(
        phase="boundary_selection",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=selection_records,
        parameters=selection_parameters,
    )
    selection_partition = _boundary_selection_partition(context)
    selection_marker = resume_date_checkpoint(
        selection_partition,
        expected_fingerprint=selection_fingerprint,
        resume=resume,
    )
    if selection_marker is None:
        final_predictions = pl.read_parquet(
            final_partition / "predictions.parquet"
        ).filter(pl.col("Date").is_in(context.primary_sessions))
        final_calibration = pl.read_parquet(
            final_partition / "calibration.parquet"
        ).filter(pl.col("Date").is_in(context.primary_sessions))
        final_facts = summarize_boundary_candidate_scores(
            final_calibration,
            final_predictions,
            primary_mapping,
            candidates=final_candidates,
            sessions=context.primary_sessions,
        )
        ranking = rank_q_candidates(
            final_facts.candidate_scores,
            candidate_column="candidate_id",
            simpler_order=final_candidates,
        )
        selected_payload = build_selected_q_payload(
            ranking,
            base_tod_source_candidates=tod_sources,
        )
        selected_payload.update(
            {
                "rank1_anchor_model_id": anchor_ids[0],
                "rank2_anchor_diagnostic_id": anchor_ids[1],
                "count120_anchor_diagnostic_id": anchor_ids[2],
                "selected_boundary_rows_per_product_day": 24,
                "selection_uses_development_outcomes": True,
            }
        )
        selection_marker = atomic_publish_checkpoint(
            selection_partition,
            phase="boundary_selection",
            date="primary_71_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=selection_records,
            parameters=selection_parameters,
            artifact_values=_boundary_ranking_artifacts(
                final_facts,
                ranking,
                selected_payload,
            ),
        )
    if canonical and _git_state() != initial_git_state:
        raise ValueError("source tree changed during canonical boundary phase")
    return selection_marker


def _load_selected_q_context(
    context: RunContext,
) -> tuple[str, tuple[str, str], Mapping[str, object]]:
    """Load the frozen rank1 anchor and final top-two entry-q candidates."""

    verify_date_checkpoint(
        _boundary_selection_partition(context),
        verify_inputs=True,
    )
    payload = json.loads(
        (_boundary_selection_partition(context) / "selected_q.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, Mapping):
        raise TypeError("selected-q payload must be a JSON object")
    anchor_id = str(payload.get("rank1_anchor_model_id", ""))
    q_ids = tuple(str(value) for value in payload.get("selected_candidate_ids", []))
    if anchor_id not in ACTIONABLE_MODEL_IDS:
        raise ValueError("selected-q payload has an invalid rank1 anchor")
    if len(q_ids) != 2 or len(set(q_ids)) != 2:
        raise ValueError("selected-q payload must contain two distinct candidates")
    if q_ids != (
        str(payload.get("rank1_candidate_id", "")),
        str(payload.get("rank2_candidate_id", "")),
    ):
        raise ValueError("selected-q candidate order differs from rank1/rank2")
    if int(payload.get("selected_boundary_rows_per_product_day", -1)) != (
        SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY
    ):
        raise ValueError("selected-q payload lost the 24-cell boundary contract")
    return anchor_id, (q_ids[0], q_ids[1]), payload


def select_convergence_boundary_predictions(
    predictions: pl.DataFrame,
    *,
    date: str,
    anchor_model_id: str,
    candidate_ids: Sequence[str],
) -> pl.DataFrame:
    """Return complete D-safe 24-cell panels for the selected q candidates."""

    candidates = _normalise_boundary_candidates(candidate_ids)
    required = {
        *BOUNDARY_UNIT_COLUMNS,
        "candidate_id",
        "boundary_distance_bp",
        "source_asof_date",
    }
    _require_boundary_columns(
        predictions,
        required,
        "selected convergence boundaries",
    )
    panel = predictions.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("side").cast(pl.String),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("boundary_distance_bp").cast(pl.Float64),
    ).filter(
        (pl.col("Date") == str(date))
        & (pl.col("anchor_model_id") == str(anchor_model_id))
        & pl.col("candidate_id").is_in(candidates)
    )
    if "effective_supported" in panel.columns:
        panel = panel.filter(
            pl.col("effective_supported").fill_null(False).cast(pl.Boolean)
        )
    invalid = panel.filter(
        ~pl.col("tod_bucket").is_in(tuple(bucket.id for bucket in DEFAULT_TOD_BUCKETS))
        | ~pl.col("boundary_quantile").is_in((50, 80, 95))
        | ~pl.col("side").is_in(("positive", "negative"))
        | pl.col("boundary_distance_bp").is_null()
        | ~pl.col("boundary_distance_bp").is_finite()
        | (pl.col("boundary_distance_bp") <= 0.0)
        | pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if invalid.height:
        raise ValueError("selected convergence boundaries contain unsafe rows")
    if (
        "contains_target_day_outcome" in panel.columns
        and panel["contains_target_day_outcome"].fill_null(True).any()
    ):
        raise ValueError("selected convergence boundaries contain target outcomes")
    keys = [*BOUNDARY_UNIT_COLUMNS, "candidate_id"]
    duplicate = panel.group_by(keys).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("selected convergence boundaries contain duplicate cells")
    if not panel.is_empty():
        complete = panel.group_by(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "anchor_model_id",
                "candidate_id",
            ]
        ).agg(
            pl.len().alias("rows"),
            pl.col("tod_bucket").n_unique().alias("tod_buckets"),
            pl.col("boundary_quantile").n_unique().alias("quantiles"),
            pl.col("side").n_unique().alias("sides"),
        )
        incomplete = complete.filter(
            (pl.col("rows") != SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY)
            | (pl.col("tod_buckets") != len(DEFAULT_TOD_BUCKETS))
            | (pl.col("quantiles") != 3)
            | (pl.col("sides") != 2)
        )
        if incomplete.height:
            raise ValueError("selected convergence boundary panel is not 24-cell")
    return panel.sort(keys)


def build_s1_lookup_views(
    selected_boundaries: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build unambiguous rank1 long and 4-row-per-product-day wide lookups."""

    if selected_boundaries.is_empty():
        raise ValueError("S1 selected boundary lookup must not be empty")
    anchors = selected_boundaries["anchor_model_id"].unique().to_list()
    candidates = selected_boundaries["candidate_id"].unique().to_list()
    if len(anchors) != 1 or len(candidates) != 1:
        raise ValueError("S1 lookup must contain exactly one anchor and q candidate")
    long = selected_boundaries.sort(
        [
            "Date",
            "ValueCode",
            "QuoteCode",
            "tod_bucket",
            "boundary_quantile",
            "side",
        ]
    )
    keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "anchor_model_id",
        "candidate_id",
        "tod_bucket",
    ]
    paired = long.group_by([*keys, "boundary_quantile"]).agg(
        pl.col("boundary_distance_bp")
        .filter(pl.col("side") == "positive")
        .first()
        .alias("upper_distance_bp"),
        pl.col("boundary_distance_bp")
        .filter(pl.col("side") == "negative")
        .first()
        .alias("lower_distance_bp"),
        pl.col("source_asof_date").min().alias("source_asof_start_date"),
        pl.col("source_asof_date").max().alias("source_asof_end_date"),
    )
    wide = paired.pivot(
        on="boundary_quantile",
        index=keys,
        values=[
            "upper_distance_bp",
            "lower_distance_bp",
            "source_asof_start_date",
            "source_asof_end_date",
        ],
    )
    rename = {
        f"{side}_{value}": f"{prefix}_q{quantile}_{suffix}"
        for side, prefix, suffix in (
            ("upper_distance_bp", "upper", "bp"),
            ("lower_distance_bp", "lower", "bp"),
            ("source_asof_start_date", "source_asof_start", "date"),
            ("source_asof_end_date", "source_asof_end", "date"),
        )
        for value, quantile in (("50", 50), ("80", 80), ("95", 95))
        if f"{side}_{value}" in wide.columns
    }
    wide = wide.rename(rename)
    required_values = [
        f"{side}_q{quantile}_bp"
        for side in ("upper", "lower")
        for quantile in (50, 80, 95)
    ]
    missing = sorted(set(required_values) - set(wide.columns))
    if missing:
        raise ValueError(f"S1 wide lookup lacks q/side columns: {missing}")
    invalid = wide.filter(
        pl.any_horizontal(
            *(
                pl.col(column).is_null()
                | ~pl.col(column).cast(pl.Float64).is_finite()
                | (pl.col(column).cast(pl.Float64) <= 0.0)
                for column in required_values
            )
        )
    )
    if invalid.height:
        raise ValueError("S1 wide lookup has incomplete q/side cells")
    expected_wide_rows = long.select(
        "Date", "ValueCode", "QuoteCode"
    ).unique().height * len(DEFAULT_TOD_BUCKETS)
    if wide.height != expected_wide_rows:
        raise ValueError("S1 wide lookup is not four TOD rows per product-day")
    wide = wide.with_columns(
        pl.lit(SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY).alias(
            "long_rows_per_product_day"
        ),
        pl.lit(False).alias("contains_target_day_outcome"),
    ).sort(keys)
    return long, wide


def _convergence_facts_date_partition(
    context: RunContext,
    date: str,
) -> Path:
    return (
        context.paths.resolved_work_root
        / "convergence"
        / "facts"
        / "dates"
        / f"Date={date}"
    )


def _convergence_facts_index_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "convergence" / "facts" / "index"


def _convergence_prediction_date_partition(
    context: RunContext,
    date: str,
) -> Path:
    return (
        context.paths.resolved_work_root
        / "convergence"
        / "predictions"
        / "dates"
        / f"Date={date}"
    )


def _convergence_prediction_index_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "convergence" / "predictions" / "index"


def _convergence_review_partition(context: RunContext) -> Path:
    return context.paths.resolved_work_root / "convergence" / "review"


def _convergence_fact_parameters(
    anchor_model_id: str,
    candidate_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "anchor_model_id": str(anchor_model_id),
        "entry_q_candidate_ids": list(candidate_ids),
        "read_columns": list(EPISODE_READ_COLUMNS),
        "entry_touch_stop_second_exclusive": 14_400,
        "tracking_end_second": 15_600,
        "boundary_rows_per_supported_product_day": (
            SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY
        ),
        "touch_definition": "first_causal_selected_upper_touch",
        "convergence_reference_semantics": "frozen_anchor_at_upper_touch",
        "frozen_center_formula": "selected_anchor_bp_at_upper_touch",
        "frozen_lower_formula": "frozen_center_basis_bp-threshold_distance_bp",
        "path_contract": (
            "frozen_center_then_immediately_following_frozen_negative_cycle"
        ),
        "right_censor_reasons": ["eligibility_gap", "session_cutoff"],
    }


def _convergence_fact_records(
    context: RunContext,
    date: str,
) -> list[dict[str, object]]:
    daily = _daily_partition(context.paths, date)
    return build_file_records(
        [
            (daily / "complete.json", "daily_completion_marker"),
            (daily / "causal_fair.parquet", "daily_causal_fair_input"),
            (
                _boundary_final_partition(context) / "complete.json",
                "boundary_final_panel_checkpoint_marker",
            ),
            (
                _boundary_selection_partition(context) / "complete.json",
                "boundary_selection_checkpoint_marker",
            ),
        ]
    )


def _convergence_index_manifest(
    context: RunContext,
    *,
    kind: str,
    artifact_name: str,
) -> pl.DataFrame:
    if kind not in {"facts", "predictions"}:
        raise ValueError("convergence manifest kind must be facts or predictions")
    rows: list[dict[str, object]] = []
    for date in context.sessions:
        partition = (
            _convergence_facts_date_partition(context, date)
            if kind == "facts"
            else _convergence_prediction_date_partition(context, date)
        )
        marker = verify_date_checkpoint(partition)
        artifact = dict(marker["artifacts"])[artifact_name]
        rows.append(
            {
                "Date": date,
                "stage": "history" if date < PRIMARY_START_DATE else "primary",
                "rows": int(artifact["rows"]),
                "checkpoint_fingerprint": str(marker["checkpoint_fingerprint"]),
                "partition": str(partition),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _empty_convergence_predictions() -> pl.DataFrame:
    from .foundation_convergence_lookup import CONVERGENCE_PREDICTION_SCHEMA

    return pl.DataFrame(schema=CONVERGENCE_PREDICTION_SCHEMA)


def _empty_convergence_path_scores() -> pl.DataFrame:
    from .foundation_convergence_lookup import CONVERGENCE_PATH_SCORE_SCHEMA

    return pl.DataFrame(schema=CONVERGENCE_PATH_SCORE_SCHEMA)


def _empty_convergence_summary() -> pl.DataFrame:
    from .foundation_convergence_lookup import CONVERGENCE_PREDICTION_SCHEMA

    schema = dict(CONVERGENCE_PREDICTION_SCHEMA)
    schema.update(
        {
            "prediction_contains_target_day_outcome": pl.Boolean,
            "contains_target_day_outcome": pl.Boolean,
            "n_started": pl.Int64,
            "confirmed_hits": pl.Int64,
            "known_misses": pl.Int64,
            "unknown_censored": pl.Int64,
            "reach_lower_bound": pl.Float64,
            "reach_upper_bound": pl.Float64,
            "first_hit_observations": pl.Int64,
            "time_to_hit_from_touch_p50_seconds": pl.Int64,
            "time_to_hit_from_touch_p90_seconds": pl.Int64,
            "time_to_hit_from_center_p50_seconds": pl.Int64,
            "time_to_hit_from_center_p90_seconds": pl.Int64,
            "time_quantile_method": pl.String,
            "outcome_status": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)


def build_convergence_comparison_predictions(
    registered: pl.DataFrame,
    resolved: pl.DataFrame,
    controls: pl.DataFrame,
) -> pl.DataFrame:
    """Give trail20, trail60, resolved, and control thresholds unique IDs."""

    from .foundation_convergence_lookup import (
        THRESHOLD_KEYS,
        validate_convergence_prediction_lineage,
    )

    frames: list[pl.DataFrame] = []
    if not registered.is_empty():
        lookup_ids = set(registered["lookup_id"].unique().to_list())
        if lookup_ids != set(CONVERGENCE_REGISTERED_LOOKUPS):
            raise ValueError("registered convergence lookup set drift")
        unknown = set(registered["convergence_candidate_id"].unique().to_list()) - set(
            CONVERGENCE_BASE_CANDIDATES
        )
        if unknown:
            raise ValueError(f"unknown registered convergence candidates: {unknown}")
        frames.append(
            registered.with_columns(
                pl.concat_str(
                    "convergence_candidate_id",
                    pl.lit("__"),
                    "lookup_id",
                ).alias("convergence_candidate_id")
            )
        )
    if not resolved.is_empty():
        if set(resolved["lookup_id"].unique().to_list()) != {
            CONVERGENCE_PRIMARY_LOOKUP_ID
        }:
            raise ValueError("resolved convergence rows lost primary lookup lineage")
        frames.append(
            resolved.with_columns(
                pl.concat_str(
                    "convergence_candidate_id",
                    pl.lit("__trail20_then_trail60"),
                ).alias("convergence_candidate_id")
            )
        )
    if not controls.is_empty():
        unknown_controls = set(
            controls["convergence_candidate_id"].unique().to_list()
        ) - set(CONVERGENCE_CONTROL_CANDIDATES)
        if unknown_controls:
            raise ValueError(f"unknown convergence controls: {unknown_controls}")
        frames.append(controls)
    if not frames:
        return _empty_convergence_predictions()
    result = pl.concat(frames, how="vertical_relaxed").sort(THRESHOLD_KEYS)
    duplicate = result.group_by(THRESHOLD_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("convergence comparison predictions are not unique")
    validate_convergence_prediction_lineage(result)
    return result


def build_convergence_support_audit(
    targets: pl.DataFrame,
    registered: pl.DataFrame,
    resolved: pl.DataFrame,
    controls: pl.DataFrame,
) -> pl.DataFrame:
    """Measure native/effective threshold support against all target cells."""

    from .foundation_convergence_lookup import TARGET_LINEAGE_KEYS

    _require_boundary_columns(targets, TARGET_LINEAGE_KEYS, "convergence targets")
    schema = {
        "Date": pl.String,
        "candidate_id": pl.String,
        "convergence_candidate_id": pl.String,
        "prediction_role": pl.String,
        "lookup_id": pl.String,
        "expected_target_units": pl.Int64,
        "prediction_rows": pl.Int64,
        "native_supported_units": pl.Int64,
        "effective_supported_units": pl.Int64,
        "fallback_used_units": pl.Int64,
        "native_support_coverage": pl.Float64,
        "effective_support_coverage": pl.Float64,
        "contains_target_day_outcome": pl.Boolean,
    }
    if targets.is_empty():
        return pl.DataFrame(schema=schema)
    target = targets.select(*TARGET_LINEAGE_KEYS).unique()
    duplicate = targets.group_by(TARGET_LINEAGE_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("convergence support targets contain duplicate lineage")
    denominators = {
        (str(date), str(candidate)): int(count)
        for date, candidate, count in target.group_by(["Date", "candidate_id"])
        .len()
        .iter_rows()
    }
    records: list[dict[str, object]] = []
    frame_specs = (
        ("registered_native", registered, False),
        ("resolved_effective", resolved, True),
        ("structural_control", controls, False),
    )
    for role, frame, resolved_role in frame_specs:
        if frame.is_empty():
            continue
        required = {
            *TARGET_LINEAGE_KEYS,
            "convergence_candidate_id",
            "lookup_id",
            "native_supported",
            "effective_supported",
            "fallback_used",
            "contains_target_day_outcome",
        }
        _require_boundary_columns(frame, required, f"{role} convergence predictions")
        if frame["contains_target_day_outcome"].fill_null(True).any():
            raise ValueError(f"{role} convergence predictions contain target outcomes")
        extra = frame.join(target, on=list(TARGET_LINEAGE_KEYS), how="anti")
        if extra.height:
            raise ValueError(f"{role} convergence predictions exceed target lineage")
        group_keys = ["Date", "candidate_id", "convergence_candidate_id", "lookup_id"]
        for raw_key, group in frame.group_by(group_keys, maintain_order=True):
            date, candidate, convergence_candidate, lookup = map(str, raw_key)
            expected = denominators[(date, candidate)]
            native = int(group["native_supported"].fill_null(False).sum())
            effective = int(group["effective_supported"].fill_null(False).sum())
            fallback = int(group["fallback_used"].fill_null(False).sum())
            records.append(
                {
                    "Date": date,
                    "candidate_id": candidate,
                    "convergence_candidate_id": convergence_candidate,
                    "prediction_role": role,
                    "lookup_id": ("trail20_then_trail60" if resolved_role else lookup),
                    "expected_target_units": expected,
                    "prediction_rows": group.height,
                    "native_supported_units": native,
                    "effective_supported_units": effective,
                    "fallback_used_units": fallback,
                    "native_support_coverage": native / expected,
                    "effective_support_coverage": effective / expected,
                    "contains_target_day_outcome": False,
                }
            )
    return pl.from_dicts(records, schema=schema, infer_schema_length=None).sort(
        [
            "Date",
            "candidate_id",
            "convergence_candidate_id",
            "prediction_role",
            "lookup_id",
        ]
    )


def _inverse_empirical_seconds(
    values: Sequence[int],
    probability: float,
) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def _summarize_convergence_paths(
    path_scores: pl.DataFrame,
    *,
    lineage_stratified: bool,
) -> pl.DataFrame:
    """Pool hit bounds, optionally retaining effective-lookup strata."""

    required = {
        "Date",
        "ValueCode",
        "candidate_id",
        "boundary_quantile",
        "convergence_candidate_id",
        "effective_lookup_id",
        "confirmed_hit",
        "known_miss",
        "unknown_censored",
        "time_to_hit_from_touch_seconds",
        "time_to_hit_from_center_seconds",
    }
    _require_boundary_columns(path_scores, required, "convergence path scores")
    schema = {
        "candidate_id": pl.String,
        "boundary_quantile": pl.Int64,
        "convergence_candidate_id": pl.String,
        **({"effective_lookup_id": pl.String} if lineage_stratified else {}),
        "effective_lookup_ids": pl.String,
        "effective_lookup_id_count": pl.Int64,
        "trail20_effective_paths": pl.Int64,
        "trail60_effective_paths": pl.Int64,
        "other_effective_paths": pl.Int64,
        "fallback_effective_paths": pl.Int64,
        "fallback_source_share": pl.Float64,
        "dates": pl.Int64,
        "products": pl.Int64,
        "n_started": pl.Int64,
        "confirmed_hits": pl.Int64,
        "known_misses": pl.Int64,
        "unknown_censored": pl.Int64,
        "reach_lower_bound": pl.Float64,
        "reach_upper_bound": pl.Float64,
        "time_to_hit_from_touch_p50_seconds": pl.Int64,
        "time_to_hit_from_touch_p90_seconds": pl.Int64,
        "time_to_hit_from_center_p50_seconds": pl.Int64,
        "time_to_hit_from_center_p90_seconds": pl.Int64,
        "contains_target_day_outcome": pl.Boolean,
        "unique_winner_declared": pl.Boolean,
    }
    if path_scores.is_empty():
        return pl.DataFrame(schema=schema)
    base_keys = (
        "candidate_id",
        "boundary_quantile",
        "convergence_candidate_id",
    )
    keys = (*base_keys, "effective_lookup_id") if lineage_stratified else base_keys
    rows: list[dict[str, object]] = []
    for raw_key, group in path_scores.group_by(keys, maintain_order=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        started = group.height
        confirmed = int(group["confirmed_hit"].sum())
        known = int(group["known_miss"].sum())
        unknown = int(group["unknown_censored"].sum())
        if started != confirmed + known + unknown:
            raise ValueError("convergence hit-status counts do not partition paths")
        touch_times = [
            int(value) for value in group["time_to_hit_from_touch_seconds"].drop_nulls()
        ]
        center_times = [
            int(value)
            for value in group["time_to_hit_from_center_seconds"].drop_nulls()
        ]
        lookup_ids = sorted(
            str(value) for value in group["effective_lookup_id"].drop_nulls().unique()
        )
        trail20_paths = group.filter(
            pl.col("effective_lookup_id") == CONVERGENCE_PRIMARY_LOOKUP_ID
        ).height
        trail60_paths = group.filter(
            pl.col("effective_lookup_id") == CONVERGENCE_FALLBACK_LOOKUP_ID
        ).height
        other_paths = started - trail20_paths - trail60_paths
        convergence_id = str(group.item(0, "convergence_candidate_id"))
        is_resolved = convergence_id.endswith("__trail20_then_trail60")
        fallback_paths = trail60_paths if is_resolved else 0
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "effective_lookup_ids": ",".join(lookup_ids),
                "effective_lookup_id_count": len(lookup_ids),
                "trail20_effective_paths": trail20_paths,
                "trail60_effective_paths": trail60_paths,
                "other_effective_paths": other_paths,
                "fallback_effective_paths": fallback_paths,
                "fallback_source_share": (
                    fallback_paths / started if is_resolved else None
                ),
                "dates": group["Date"].n_unique(),
                "products": group["ValueCode"].n_unique(),
                "n_started": started,
                "confirmed_hits": confirmed,
                "known_misses": known,
                "unknown_censored": unknown,
                "reach_lower_bound": confirmed / started,
                "reach_upper_bound": (confirmed + unknown) / started,
                "time_to_hit_from_touch_p50_seconds": _inverse_empirical_seconds(
                    touch_times, 0.50
                ),
                "time_to_hit_from_touch_p90_seconds": _inverse_empirical_seconds(
                    touch_times, 0.90
                ),
                "time_to_hit_from_center_p50_seconds": _inverse_empirical_seconds(
                    center_times, 0.50
                ),
                "time_to_hit_from_center_p90_seconds": _inverse_empirical_seconds(
                    center_times, 0.90
                ),
                "contains_target_day_outcome": True,
                "unique_winner_declared": False,
            }
        )
    return pl.from_dicts(rows, schema=schema, infer_schema_length=None).sort(keys)


def summarize_convergence_primary(path_scores: pl.DataFrame) -> pl.DataFrame:
    """Pool each policy across fallback sources; never select a winner."""

    return _summarize_convergence_paths(
        path_scores,
        lineage_stratified=False,
    )


def summarize_convergence_primary_lineage(
    path_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Retain effective lookup strata as a separate fallback-lineage audit."""

    return _summarize_convergence_paths(
        path_scores,
        lineage_stratified=True,
    )


def build_convergence_common_support_sensitivity(
    path_scores: pl.DataFrame,
    *,
    entry_q_candidate_ids: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compare top-two entry-q policies on exact shared touched episodes."""

    candidates = _normalise_boundary_candidates(entry_q_candidate_ids)
    if len(candidates) != 2:
        raise ValueError("convergence common support requires exact top-two q IDs")
    unit_keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "tod_bucket",
        "boundary_quantile",
        "episode_sequence",
        "convergence_candidate_id",
    ]
    _require_boundary_columns(
        path_scores,
        {*unit_keys, "candidate_id"},
        "convergence common-support path scores",
    )
    panel = path_scores.filter(pl.col("candidate_id").is_in(candidates))
    duplicate = (
        panel.group_by([*unit_keys, "candidate_id"]).len().filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("convergence common-support paths contain duplicates")
    common_keys = (
        panel.group_by(unit_keys)
        .agg(pl.col("candidate_id").n_unique().alias("candidate_count"))
        .filter(pl.col("candidate_count") == len(candidates))
    )
    common = panel.join(
        common_keys.select(*unit_keys),
        on=unit_keys,
        how="semi",
    )
    review = summarize_convergence_primary(common).with_columns(
        pl.lit(True).alias("exact_top2_entry_q_common_support"),
        pl.lit(",".join(candidates)).alias("common_entry_q_candidate_ids"),
    )
    sensitivity_schema = {
        "boundary_quantile": pl.Int64,
        "convergence_candidate_id": pl.String,
        "rank1_entry_q_candidate_id": pl.String,
        "rank2_entry_q_candidate_id": pl.String,
        "common_paths": pl.Int64,
        "rank1_reach_lower_bound": pl.Float64,
        "rank2_reach_lower_bound": pl.Float64,
        "rank2_minus_rank1_reach_lower_bound": pl.Float64,
        "rank1_reach_upper_bound": pl.Float64,
        "rank2_reach_upper_bound": pl.Float64,
        "rank2_minus_rank1_reach_upper_bound": pl.Float64,
        "rank1_touch_p50_seconds": pl.Int64,
        "rank2_touch_p50_seconds": pl.Int64,
        "rank2_minus_rank1_touch_p50_seconds": pl.Int64,
        "rank1_fallback_source_share": pl.Float64,
        "rank2_fallback_source_share": pl.Float64,
        "contains_target_day_outcome": pl.Boolean,
        "unique_winner_declared": pl.Boolean,
    }
    if review.is_empty():
        sensitivity = pl.DataFrame(schema=sensitivity_schema)
    else:
        metrics = [
            "n_started",
            "reach_lower_bound",
            "reach_upper_bound",
            "time_to_hit_from_touch_p50_seconds",
            "fallback_source_share",
        ]
        join_keys = ["boundary_quantile", "convergence_candidate_id"]
        rank1 = review.filter(pl.col("candidate_id") == candidates[0]).select(
            *join_keys,
            *(pl.col(value).alias(f"rank1_{value}") for value in metrics),
        )
        rank2 = review.filter(pl.col("candidate_id") == candidates[1]).select(
            *join_keys,
            *(pl.col(value).alias(f"rank2_{value}") for value in metrics),
        )
        paired = rank1.join(rank2, on=join_keys, how="inner", validate="1:1")
        if paired.filter(pl.col("rank1_n_started") != pl.col("rank2_n_started")).height:
            raise AssertionError("exact common support has unequal entry-q path counts")
        sensitivity = paired.select(
            *join_keys,
            pl.lit(candidates[0]).alias("rank1_entry_q_candidate_id"),
            pl.lit(candidates[1]).alias("rank2_entry_q_candidate_id"),
            pl.col("rank1_n_started").alias("common_paths"),
            pl.col("rank1_reach_lower_bound"),
            pl.col("rank2_reach_lower_bound"),
            (
                pl.col("rank2_reach_lower_bound") - pl.col("rank1_reach_lower_bound")
            ).alias("rank2_minus_rank1_reach_lower_bound"),
            pl.col("rank1_reach_upper_bound"),
            pl.col("rank2_reach_upper_bound"),
            (
                pl.col("rank2_reach_upper_bound") - pl.col("rank1_reach_upper_bound")
            ).alias("rank2_minus_rank1_reach_upper_bound"),
            pl.col("rank1_time_to_hit_from_touch_p50_seconds").alias(
                "rank1_touch_p50_seconds"
            ),
            pl.col("rank2_time_to_hit_from_touch_p50_seconds").alias(
                "rank2_touch_p50_seconds"
            ),
            (
                pl.col("rank2_time_to_hit_from_touch_p50_seconds")
                - pl.col("rank1_time_to_hit_from_touch_p50_seconds")
            ).alias("rank2_minus_rank1_touch_p50_seconds"),
            pl.col("rank1_fallback_source_share"),
            pl.col("rank2_fallback_source_share"),
            pl.lit(True).alias("contains_target_day_outcome"),
            pl.lit(False).alias("unique_winner_declared"),
        ).cast(sensitivity_schema)
    return common, review, sensitivity


def _convergence_prediction_parameters(
    anchor_model_id: str,
    candidate_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "anchor_model_id": str(anchor_model_id),
        "entry_q_candidate_ids": list(candidate_ids),
        "registered_lookups": {
            "trail20_date_equal": {
                "lookback_sessions": 20,
                "minimum_completed_dates": 15,
                "minimum_completed_paths": 50,
            },
            "trail60_date_equal": {
                "lookback_sessions": 60,
                "minimum_completed_dates": 40,
                "minimum_completed_paths": 100,
            },
        },
        "fallback_chain": [
            CONVERGENCE_PRIMARY_LOOKUP_ID,
            CONVERGENCE_FALLBACK_LOOKUP_ID,
        ],
        "history_preparation": "validate_and_index_all131_facts_once",
        "prediction_history_slice": "strict_prior_maximum60_sessions",
        "conditional_candidates": list(CONVERGENCE_BASE_CANDIDATES),
        "controls": list(CONVERGENCE_CONTROL_CANDIDATES),
        "convergence_reference_semantics": "frozen_anchor_at_upper_touch",
        "dynamic_anchor_results_role": "noncanonical_sensitivity_only",
        "control_prediction_source": "selected_D_safe_boundary_cells_not_touch_facts",
        "support_denominator": "all_selected_boundary_target_lineage_cells",
        "entry_q_sensitivity_support": (
            "exact_common_Date_product_tod_q_episode_policy_paths"
        ),
        "score_dates": "primary_71_only",
        "unique_winner_declared": False,
    }


def _convergence_prediction_records(
    context: RunContext,
    date: str,
) -> list[dict[str, object]]:
    target_index = context.sessions.index(date)
    records: list[tuple[Path, str]] = [
        (
            _boundary_final_partition(context) / "complete.json",
            "boundary_final_panel_checkpoint_marker",
        ),
        (
            _boundary_selection_partition(context) / "complete.json",
            "boundary_selection_checkpoint_marker",
        ),
    ]
    records.extend(
        (
            _convergence_facts_date_partition(context, fact_date) / "complete.json",
            (
                "target_convergence_fact_marker"
                if fact_date == date
                else "prior_convergence_fact_marker"
            ),
        )
        for fact_date in context.sessions[: target_index + 1]
    )
    return build_file_records(records)


def _publish_convergence_index(
    context: RunContext,
    *,
    kind: str,
    artifact_name: str,
    parameters: Mapping[str, object],
    resume: bool,
) -> Mapping[str, object]:
    if kind == "facts":
        partition = _convergence_facts_index_partition(context)
        partitions = [
            _convergence_facts_date_partition(context, date)
            for date in context.sessions
        ]
        phase = "convergence_facts_index"
    elif kind == "predictions":
        partition = _convergence_prediction_index_partition(context)
        partitions = [
            _convergence_prediction_date_partition(context, date)
            for date in context.sessions
        ]
        phase = "convergence_predictions_index"
    else:
        raise ValueError("convergence index kind must be facts or predictions")
    records = build_file_records(
        [
            (item / "complete.json", f"convergence_{kind}_date_marker")
            for item in partitions
        ]
    )
    fingerprint = checkpoint_fingerprint(
        phase=phase,
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
    )
    marker = resume_date_checkpoint(
        partition,
        expected_fingerprint=fingerprint,
        resume=resume,
    )
    if marker is not None:
        return marker
    return atomic_publish_checkpoint(
        partition,
        phase=phase,
        date="frozen_131_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=str(context.git_state["commit"]),
        input_records=records,
        parameters=parameters,
        artifact_values={
            f"{kind}_manifest.parquet": _convergence_index_manifest(
                context,
                kind=kind,
                artifact_name=artifact_name,
            )
        },
    )


def run_convergence_phase(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Build top-two-q post-touch facts and report C0/C1/C2/C3 results."""

    from .foundation_convergence_lookup import (
        build_control_convergence_predictions_from_boundaries,
        build_registered_conditional_convergence_grid_prepared,
        prepare_convergence_history,
        resolve_convergence_fallback,
        score_convergence_predictions,
    )
    from .foundation_convergence_selection import (
        build_post_touch_convergence_facts,
    )

    selected_paths = paths or SelectionPaths()
    run_boundary_phase(selected_paths, resume=resume, canonical=canonical)
    context = _context(selected_paths, canonical=canonical)
    initial_git_state = dict(context.git_state)
    source_commit = str(context.git_state["commit"])
    anchor_id, q_ids, _ = _load_selected_q_context(context)
    final_predictions = pl.read_parquet(
        _boundary_final_partition(context) / "predictions.parquet"
    )
    fact_parameters = _convergence_fact_parameters(anchor_id, q_ids)
    for index, date in enumerate(context.sessions, start=1):
        records = _convergence_fact_records(context, date)
        fingerprint = checkpoint_fingerprint(
            phase="convergence_facts",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=fact_parameters,
        )
        partition = _convergence_facts_date_partition(context, date)
        marker = resume_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            resume=resume,
        )
        if marker is not None:
            print(
                f"convergence facts {index}/{len(context.sessions)} {date} resumed",
                flush=True,
            )
            continue
        boundaries = select_convergence_boundary_predictions(
            final_predictions,
            date=date,
            anchor_model_id=anchor_id,
            candidate_ids=q_ids,
        )
        if boundaries.is_empty():
            facts = pl.DataFrame(schema=CONVERGENCE_FACT_SCHEMA)
        else:
            day = pl.read_parquet(
                _daily_partition(selected_paths, date) / "causal_fair.parquet",
                columns=list(EPISODE_READ_COLUMNS),
            )
            materialized = materialize_anchor_column(day, anchor_id)
            facts = build_post_touch_convergence_facts(
                materialized,
                boundaries,
                anchor_column="selected_anchor_bp",
                anchor_model_id=anchor_id,
            )
            del day, materialized
        atomic_publish_checkpoint(
            partition,
            phase="convergence_facts",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=fact_parameters,
            artifact_values={"post_touch_facts.parquet": facts},
        )
        del boundaries, facts
        gc.collect()
        print(
            f"convergence facts {index}/{len(context.sessions)} {date} complete",
            flush=True,
        )

    _publish_convergence_index(
        context,
        kind="facts",
        artifact_name="post_touch_facts.parquet",
        parameters={
            **fact_parameters,
            "expected_date_checkpoints": EXPECTED_SESSION_COUNT,
        },
        resume=resume,
    )
    prediction_parameters = _convergence_prediction_parameters(anchor_id, q_ids)
    all_facts: pl.DataFrame | None = None
    prepared_history: object | None = None
    for index, date in enumerate(context.sessions, start=1):
        records = _convergence_prediction_records(context, date)
        fingerprint = checkpoint_fingerprint(
            phase="convergence_predictions",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=prediction_parameters,
        )
        partition = _convergence_prediction_date_partition(context, date)
        marker = resume_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            resume=resume,
        )
        if marker is not None:
            print(
                f"convergence predictions {index}/{len(context.sessions)} "
                f"{date} resumed",
                flush=True,
            )
            continue
        boundaries = select_convergence_boundary_predictions(
            final_predictions,
            date=date,
            anchor_model_id=anchor_id,
            candidate_ids=q_ids,
        )
        targets = (
            boundaries.filter(pl.col("side") == "positive")
            .select(
                "Date",
                "ValueCode",
                "QuoteCode",
                "anchor_model_id",
                "candidate_id",
                "tod_bucket",
                "boundary_quantile",
            )
            .unique()
        )
        target_facts = pl.DataFrame(schema=CONVERGENCE_FACT_SCHEMA)
        if targets.is_empty():
            registered = _empty_convergence_predictions()
            resolved = _empty_convergence_predictions()
            controls = _empty_convergence_predictions()
            comparison = _empty_convergence_predictions()
            path_scores = _empty_convergence_path_scores()
            summary = _empty_convergence_summary()
        else:
            if all_facts is None:
                all_facts = pl.concat(
                    [
                        pl.read_parquet(
                            _convergence_facts_date_partition(context, fact_date)
                            / "post_touch_facts.parquet"
                        )
                        for fact_date in context.sessions
                    ],
                    how="vertical_relaxed",
                )
                prepared_history = prepare_convergence_history(
                    all_facts,
                    list(context.sessions),
                )
            if prepared_history is None:
                raise AssertionError("prepared convergence history is unavailable")
            target_facts = all_facts.filter(pl.col("Date") == date)
            registered = build_registered_conditional_convergence_grid_prepared(
                prepared_history,
                targets,
            )
            resolved = resolve_convergence_fallback(
                registered,
                primary_lookup_id=CONVERGENCE_PRIMARY_LOOKUP_ID,
                fallback_lookup_id=CONVERGENCE_FALLBACK_LOOKUP_ID,
            )
            controls = build_control_convergence_predictions_from_boundaries(boundaries)
            comparison = build_convergence_comparison_predictions(
                registered,
                resolved,
                controls,
            )
            score = score_convergence_predictions(target_facts, comparison)
            path_scores = score.path_facts
            summary = score.summary
        effective = pl.concat(
            [resolved, controls],
            how="vertical_relaxed",
        ).sort(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "candidate_id",
                "tod_bucket",
                "boundary_quantile",
                "convergence_candidate_id",
            ]
        )
        support_audit = build_convergence_support_audit(
            targets,
            registered,
            resolved,
            controls,
        )
        atomic_publish_checkpoint(
            partition,
            phase="convergence_predictions",
            date=date,
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=records,
            parameters=prediction_parameters,
            artifact_values={
                "registered_predictions.parquet": registered,
                "resolved_predictions.parquet": resolved,
                "control_predictions.parquet": controls,
                "effective_predictions.parquet": effective,
                "comparison_predictions.parquet": comparison,
                "support_audit.parquet": support_audit,
                "path_scores.parquet": path_scores,
                "summary.parquet": summary,
            },
        )
        del (
            boundaries,
            targets,
            target_facts,
            registered,
            resolved,
            controls,
            comparison,
            effective,
            support_audit,
            path_scores,
            summary,
        )
        gc.collect()
        print(
            f"convergence predictions {index}/{len(context.sessions)} {date} complete",
            flush=True,
        )

    _publish_convergence_index(
        context,
        kind="predictions",
        artifact_name="comparison_predictions.parquet",
        parameters={
            **prediction_parameters,
            "expected_date_checkpoints": EXPECTED_SESSION_COUNT,
        },
        resume=resume,
    )
    review_records = build_file_records(
        [
            (
                _convergence_prediction_index_partition(context) / "complete.json",
                "convergence_predictions_index_marker",
            ),
            (
                _convergence_facts_index_partition(context) / "complete.json",
                "convergence_facts_index_marker",
            ),
        ]
    )
    review_parameters = {
        **prediction_parameters,
        "review_dates": list(context.primary_sessions),
        "selection": "report_variants_without_unique_winner",
    }
    review_fingerprint = checkpoint_fingerprint(
        phase="convergence_review",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=review_records,
        parameters=review_parameters,
    )
    review_partition = _convergence_review_partition(context)
    review_marker = resume_date_checkpoint(
        review_partition,
        expected_fingerprint=review_fingerprint,
        resume=resume,
    )
    if review_marker is None:
        primary_summary = pl.concat(
            [
                pl.read_parquet(
                    _convergence_prediction_date_partition(context, date)
                    / "summary.parquet"
                )
                for date in context.primary_sessions
            ],
            how="vertical_relaxed",
        )
        primary_path_scores = pl.concat(
            [
                pl.read_parquet(
                    _convergence_prediction_date_partition(context, date)
                    / "path_scores.parquet"
                )
                for date in context.primary_sessions
            ],
            how="vertical_relaxed",
        )
        primary_support_audit = pl.concat(
            [
                pl.read_parquet(
                    _convergence_prediction_date_partition(context, date)
                    / "support_audit.parquet"
                )
                for date in context.primary_sessions
            ],
            how="vertical_relaxed",
        )
        review = summarize_convergence_primary(primary_path_scores)
        lineage_review = summarize_convergence_primary_lineage(primary_path_scores)
        common_paths, common_review, common_sensitivity = (
            build_convergence_common_support_sensitivity(
                primary_path_scores,
                entry_q_candidate_ids=q_ids,
            )
        )
        review_marker = atomic_publish_checkpoint(
            review_partition,
            phase="convergence_review",
            date="primary_71_sessions",
            registry_sha256=context.registry.sha256,
            source_commit=source_commit,
            input_records=review_records,
            parameters=review_parameters,
            artifact_values={
                "primary_summary.parquet": primary_summary,
                "primary_path_scores.parquet": primary_path_scores,
                "candidate_review.parquet": review,
                "candidate_lineage_review.parquet": lineage_review,
                "primary_support_audit.parquet": primary_support_audit,
                "common_support_path_scores.parquet": common_paths,
                "common_support_candidate_review.parquet": common_review,
                "common_support_entry_q_sensitivity.parquet": (common_sensitivity),
                "selection.json": {
                    "entry_q_candidate_ids": list(q_ids),
                    "conditional_variants": [
                        f"{candidate}__{lookup}"
                        for candidate in CONVERGENCE_BASE_CANDIDATES
                        for lookup in CONVERGENCE_REGISTERED_LOOKUPS
                    ],
                    "resolved_variants": [
                        f"{candidate}__trail20_then_trail60"
                        for candidate in CONVERGENCE_BASE_CANDIDATES
                    ],
                    "controls": list(CONVERGENCE_CONTROL_CANDIDATES),
                    "unique_winner_declared": False,
                    "user_review_required": True,
                    "development_only": True,
                },
            },
        )
    if canonical and _git_state() != initial_git_state:
        raise ValueError("source tree changed during canonical convergence phase")
    return review_marker


def publish_final_bundle(
    paths: SelectionPaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Publish the selected S1 mother and all review-critical S0.5 facts."""

    selected_paths = paths or SelectionPaths()
    run_convergence_phase(selected_paths, resume=resume, canonical=canonical)
    context = _context(selected_paths, canonical=canonical)
    initial_git_state = dict(context.git_state)
    source_commit = str(context.git_state["commit"])
    anchor_id, q_ids, q_payload = _load_selected_q_context(context)
    liquidity_marker = selected_paths.liquidity_path.parent / "complete.json"
    records = build_file_records(
        [
            (
                selected_paths.resolved_work_root / "preflight" / "complete.json",
                "preflight_checkpoint_marker",
            ),
            (
                _boundary_selection_partition(context) / "complete.json",
                "boundary_selection_checkpoint_marker",
            ),
            (
                _boundary_base_ranking_partition(context) / "complete.json",
                "boundary_base_ranking_checkpoint_marker",
            ),
            (
                _boundary_final_partition(context) / "complete.json",
                "boundary_final_panel_checkpoint_marker",
            ),
            (
                _boundary_inputs_partition(context) / "complete.json",
                "boundary_inputs_checkpoint_marker",
            ),
            (
                _anchor_selection_partition(context) / "complete.json",
                "anchor_selection_checkpoint_marker",
            ),
            (
                _episode_index_partition(context) / "complete.json",
                "episode_index_checkpoint_marker",
            ),
            (
                _convergence_facts_index_partition(context) / "complete.json",
                "convergence_facts_index_marker",
            ),
            (
                _convergence_prediction_index_partition(context) / "complete.json",
                "convergence_predictions_index_marker",
            ),
            (
                _convergence_review_partition(context) / "complete.json",
                "convergence_review_checkpoint_marker",
            ),
            (liquidity_marker, "liquidity_completion_marker"),
            (selected_paths.liquidity_path, "q_independent_liquidity_source"),
        ]
    )
    parameters = {
        "selected_anchor_model_id": anchor_id,
        "selected_entry_q_candidate_id": q_ids[0],
        "diagnostic_entry_q_candidate_id": q_ids[1],
        "primary_dates": list(context.primary_sessions),
        "route": SPOT_BID_ROUTE,
        "mother_boundary_rows_per_product_day": (
            SELECTED_BOUNDARY_ROWS_PER_PRODUCT_DAY
        ),
        "legacy_boundary_allowed": False,
        "legacy_q_dependent_liquidity_gate_allowed": False,
        "selected_geometry_version": "foundation_selected_lookup_geometry_v1",
        "selected_geometry_policies": [
            "q50",
            "q80",
            "q95",
            "fixed15",
            "fixed20",
            "fixed25",
            "fixed30",
        ],
        "selected_geometry_adverse_sensitivity_bp": [10.0, 20.0, 30.0],
        "selected_geometry_readiness": "known_cost_reference_not_ev",
        "development_only": True,
    }
    fingerprint = checkpoint_fingerprint(
        phase="final_publish",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
    )
    marker = resume_date_checkpoint(
        selected_paths.output_root,
        expected_fingerprint=fingerprint,
        resume=resume,
    )
    if marker is not None:
        return marker

    from .foundation_selected_geometry import (
        SELECTED_GEOMETRY_VERSION,
        build_selected_lookup_geometry,
    )
    from .universe_manifest import load_verified_liquidity_sources

    if parameters["selected_geometry_version"] != SELECTED_GEOMETRY_VERSION:
        raise ValueError("selected geometry version differs from publish fingerprint")

    inputs_partition = _boundary_inputs_partition(context)
    target_mapping = pl.read_parquet(inputs_partition / "target_mapping.parquet")
    anchor_support = pl.read_parquet(inputs_partition / "anchor_support.parquet")
    boundary_predictions = pl.read_parquet(
        _boundary_final_partition(context) / "predictions.parquet"
    )
    selected_boundaries = boundary_predictions.filter(
        (pl.col("Date").is_in(context.primary_sessions))
        & (pl.col("anchor_model_id") == anchor_id)
        & (pl.col("candidate_id") == q_ids[0])
    )
    # Validate the full primary panel, including every Date with native support.
    for date in context.primary_sessions:
        select_convergence_boundary_predictions(
            selected_boundaries,
            date=date,
            anchor_model_id=anchor_id,
            candidate_ids=(q_ids[0],),
        )
    lookup_long, lookup_wide = build_s1_lookup_views(selected_boundaries)
    liquidity_bundle = load_verified_liquidity_sources(
        selected_paths.liquidity_path.parent
    )
    cohort = build_s1_mother_cohort(
        target_mapping,
        anchor_support,
        selected_boundaries,
        liquidity_bundle.rolling_screen,
        selected_anchor_model_id=anchor_id,
        selected_candidate_id=q_ids[0],
        primary_dates=context.primary_sessions,
        route=SPOT_BID_ROUTE,
    )
    mother = cohort.mother.with_columns(
        pl.lit(True).alias("development_selected"),
        pl.lit(True).alias("selection_contains_development_outcomes"),
        pl.lit(False).alias("contains_target_day_outcome"),
    )
    geometry = build_selected_lookup_geometry(
        mother,
        target_mapping,
        lookup_long,
    )
    anchor_product_day = _load_anchor_product_day(context)
    anchor_selection_support = pl.concat(
        [
            pl.read_parquet(_anchor_date_partition(context, date) / "support.parquet")
            for date in context.primary_sessions
        ],
        how="vertical_relaxed",
    )
    convergence_facts = pl.concat(
        [
            pl.read_parquet(
                _convergence_facts_date_partition(context, date)
                / "post_touch_facts.parquet"
            )
            for date in context.sessions
        ],
        how="vertical_relaxed",
    )
    convergence_registered = pl.concat(
        [
            pl.read_parquet(
                _convergence_prediction_date_partition(context, date)
                / "registered_predictions.parquet"
            )
            for date in context.sessions
        ],
        how="vertical_relaxed",
    )
    convergence_effective = pl.concat(
        [
            pl.read_parquet(
                _convergence_prediction_date_partition(context, date)
                / "effective_predictions.parquet"
            )
            for date in context.sessions
        ],
        how="vertical_relaxed",
    )
    convergence_comparison = pl.concat(
        [
            pl.read_parquet(
                _convergence_prediction_date_partition(context, date)
                / "comparison_predictions.parquet"
            )
            for date in context.sessions
        ],
        how="vertical_relaxed",
    )
    marker = atomic_publish_checkpoint(
        selected_paths.output_root,
        phase="final_publish",
        date="primary_71_sessions",
        registry_sha256=context.registry.sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        artifact_values={
            "input_audit.parquet": pl.read_parquet(
                selected_paths.resolved_work_root / "preflight" / "input_audit.parquet"
            ),
            "preflight_lineage.json": json.loads(
                (
                    selected_paths.resolved_work_root / "preflight" / "complete.json"
                ).read_text(encoding="utf-8")
            ),
            "target_mapping.parquet": target_mapping,
            "anchor_support.parquet": anchor_support,
            "anchor_daily.parquet": pl.read_parquet(
                _anchor_selection_partition(context) / "anchor_daily.parquet"
            ),
            "anchor_selection_product_day.parquet": anchor_product_day,
            "anchor_selection_support.parquet": anchor_selection_support,
            "anchor_selection.parquet": pl.read_parquet(
                _anchor_selection_partition(context) / "anchor_selection.parquet"
            ),
            "selected_anchors.json": json.loads(
                (
                    _anchor_selection_partition(context) / "selected_anchors.json"
                ).read_text(encoding="utf-8")
            ),
            "selected_boundary_predictions.parquet": lookup_long,
            "s1_lookup_long.parquet": lookup_long,
            "s1_lookup_wide.parquet": lookup_wide,
            "boundary_candidate_ranking.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "candidate_ranking.parquet"
            ),
            "boundary_candidate_scores.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "candidate_scores.parquet"
            ),
            "boundary_common_calibration.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "common_calibration.parquet"
            ),
            "boundary_monthly_q_side_loss.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "monthly_q_side_loss.parquet"
            ),
            "boundary_daily_spearman.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "daily_spearman.parquet"
            ),
            "boundary_temporal_spearman.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "temporal_spearman.parquet"
            ),
            "boundary_support_summary.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "support_summary.parquet"
            ),
            "boundary_turnover_by_date.parquet": pl.read_parquet(
                _boundary_selection_partition(context) / "turnover_by_date.parquet"
            ),
            "boundary_base_candidate_scores.parquet": pl.read_parquet(
                _boundary_base_ranking_partition(context) / "candidate_scores.parquet"
            ),
            "boundary_base_candidate_ranking.parquet": pl.read_parquet(
                _boundary_base_ranking_partition(context) / "candidate_ranking.parquet"
            ),
            "boundary_base_selected_q.json": json.loads(
                (
                    _boundary_base_ranking_partition(context) / "selected_q.json"
                ).read_text(encoding="utf-8")
            ),
            "selected_q.json": dict(q_payload),
            "episode_manifest.parquet": pl.read_parquet(
                _episode_index_partition(context) / "episode_manifest.parquet"
            ),
            "boundary_adaptation_multipliers.parquet": pl.read_parquet(
                _boundary_final_partition(context) / "multipliers.parquet"
            ),
            "boundary_effective_support_audit.parquet": pl.read_parquet(
                _boundary_final_partition(context) / "support_audit.parquet"
            ),
            "boundary_diagnostic_predictions.parquet": pl.read_parquet(
                _boundary_final_partition(context) / "diagnostic_predictions.parquet"
            ),
            "boundary_diagnostic_calibration.parquet": pl.read_parquet(
                _boundary_final_partition(context) / "diagnostic_calibration.parquet"
            ),
            "boundary_pipeline_meta.json": json.loads(
                (_boundary_final_partition(context) / "pipeline_meta.json").read_text(
                    encoding="utf-8"
                )
            ),
            "convergence_facts.parquet": convergence_facts,
            "convergence_registered_predictions.parquet": (convergence_registered),
            "convergence_effective_predictions.parquet": convergence_effective,
            "convergence_comparison_predictions.parquet": (convergence_comparison),
            "convergence_primary_summary.parquet": pl.read_parquet(
                _convergence_review_partition(context) / "primary_summary.parquet"
            ),
            "convergence_primary_path_scores.parquet": pl.read_parquet(
                _convergence_review_partition(context) / "primary_path_scores.parquet"
            ),
            "convergence_candidate_review.parquet": pl.read_parquet(
                _convergence_review_partition(context) / "candidate_review.parquet"
            ),
            "convergence_candidate_lineage_review.parquet": pl.read_parquet(
                _convergence_review_partition(context)
                / "candidate_lineage_review.parquet"
            ),
            "convergence_primary_support_audit.parquet": pl.read_parquet(
                _convergence_review_partition(context) / "primary_support_audit.parquet"
            ),
            "convergence_common_support_path_scores.parquet": pl.read_parquet(
                _convergence_review_partition(context)
                / "common_support_path_scores.parquet"
            ),
            "convergence_common_support_candidate_review.parquet": (
                pl.read_parquet(
                    _convergence_review_partition(context)
                    / "common_support_candidate_review.parquet"
                )
            ),
            "convergence_common_support_entry_q_sensitivity.parquet": (
                pl.read_parquet(
                    _convergence_review_partition(context)
                    / "common_support_entry_q_sensitivity.parquet"
                )
            ),
            "convergence_selection.json": json.loads(
                (_convergence_review_partition(context) / "selection.json").read_text(
                    encoding="utf-8"
                )
            ),
            "s1_mother.parquet": mother,
            "s1_mother_funnel.parquet": cohort.funnel,
            "liquidity_invariance_audit.parquet": (cohort.liquidity_invariance_audit),
            "selected_policy_geometry.parquet": geometry.geometry_long,
            "selected_geometry_summary_overall.parquet": (geometry.summary_overall),
            "selected_geometry_summary_by_month.parquet": (geometry.summary_by_month),
            "selected_geometry_summary_by_tod.parquet": geometry.summary_by_tod,
            "publication.json": {
                "runner_version": RUNNER_VERSION,
                "registry_sha256": context.registry.sha256,
                "source_commit": source_commit,
                "selected_anchor_model_id": anchor_id,
                "selected_entry_q_candidate_id": q_ids[0],
                "diagnostic_entry_q_candidate_id": q_ids[1],
                "convergence_unique_winner_declared": False,
                "convergence_reference_semantics": (
                    "frozen_anchor_at_upper_touch"
                ),
                "dynamic_anchor_results_role": "noncanonical_sensitivity_only",
                "deployment_baseline_approved": False,
                "development_only": True,
                "diagnostic_scope": "all_131_sessions",
                "primary_scope": "primary_71_sessions",
                "protected_forward_start": PROTECTED_FORWARD_START_DATE,
            },
        },
    )
    if canonical and _git_state() != initial_git_state:
        raise ValueError("source tree changed during canonical final publication")
    return marker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "preflight",
            "anchor",
            "episodes",
            "boundary",
            "convergence",
            "final",
            "all",
            "verify-only",
        ),
    )
    parser.add_argument("--sessions-path", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--allow-dirty-development-run",
        action="store_true",
        help="disable the canonical clean-tree guard; outputs remain noncanonical",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=ANCHOR_BOOTSTRAP_REPLICATES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one explicitly requested phase; imports never execute data work."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.phase != "verify-only" and not args.execute:
        parser.error(f"phase {args.phase!r} requires explicit --execute")
    paths = SelectionPaths(
        sessions_path=args.sessions_path,
        daily_root=args.daily_root,
        registry_path=args.registry_path,
        liquidity_path=args.liquidity,
        output_root=args.output_root,
        work_root=args.work_root,
    )
    canonical = not args.allow_dirty_development_run
    resume = not args.no_resume
    if args.phase == "preflight":
        result = run_preflight(paths, resume=resume, canonical=canonical)
    elif args.phase == "anchor":
        result = run_anchor_phase(
            paths,
            resume=resume,
            canonical=canonical,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.phase == "episodes":
        result = run_episode_phase(paths, resume=resume, canonical=canonical)
    elif args.phase == "boundary":
        result = run_boundary_phase(paths, resume=resume, canonical=canonical)
    elif args.phase == "convergence":
        result = run_convergence_phase(paths, resume=resume, canonical=canonical)
    elif args.phase == "final":
        result = publish_final_bundle(paths, resume=resume, canonical=canonical)
    elif args.phase == "all":
        run_anchor_phase(
            paths,
            resume=resume,
            canonical=canonical,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        run_episode_phase(paths, resume=resume, canonical=canonical)
        result = publish_final_bundle(paths, resume=resume, canonical=canonical)
    else:
        result = verify_work(
            paths,
            canonical=canonical,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
