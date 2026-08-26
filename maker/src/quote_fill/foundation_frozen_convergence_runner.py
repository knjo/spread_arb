"""Rebuild S0.5 convergence with the exit reference frozen at upper touch.

The canonical ace2669/v1 foundation bundle remains immutable.  Its validated
anchor, boundary selection, and S1 mother are reused, but none of its
moving-anchor convergence outcomes enter this runner.  Selected-Q2 boundaries
are migrated from the content-hashed v1 boundary checkpoint, post-touch facts
are rebuilt from the frozen 131-session causal inputs, and C0/C2/C3 reference
geometry is published in a separate atomic v2 supplement.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from ..common.paths import MAKER_ROOT
from .foundation_anchor_selection import materialize_anchor_column
from .foundation_convergence_lookup import (
    CONVERGENCE_REFERENCE_SEMANTICS,
    TARGET_LINEAGE_KEYS,
    build_control_convergence_predictions_from_boundaries,
    build_registered_conditional_convergence_grid_prepared,
    prepare_convergence_history,
    resolve_convergence_fallback,
    score_convergence_predictions,
    validate_convergence_prediction_lineage,
)
from .foundation_convergence_selection import build_post_touch_convergence_facts
from .foundation_revalidation_runner import (
    DEFAULT_DAILY_ROOT,
    DEFAULT_SESSIONS_PATH,
    PRIMARY_START_DATE,
    PROTECTED_FORWARD_START_DATE,
)
from .foundation_selected_geometry import (
    CONDITIONAL_CANDIDATES,
    CONDITIONAL_GEOMETRY_VERSION,
    build_conditional_lookup_geometry,
)
from .foundation_selection_runner import (
    CONVERGENCE_FACT_SCHEMA,
    CONVERGENCE_FALLBACK_LOOKUP_ID,
    CONVERGENCE_PRIMARY_LOOKUP_ID,
    DEFAULT_REGISTRY_PATH,
    EPISODE_READ_COLUMNS,
    EXPECTED_PRIMARY_SESSION_COUNT,
    EXPECTED_REGISTRY_SHA256,
    EXPECTED_SESSION_COUNT,
    LEGACY_OUTPUT_ROOT_V1,
    _git_state,
    _require_canonical_git_state,
    atomic_publish_checkpoint,
    build_convergence_support_audit,
    build_file_records,
    checkpoint_fingerprint,
    load_registry,
    load_selection_sessions,
    select_convergence_boundary_predictions,
    summarize_convergence_primary,
    verify_date_checkpoint,
    verify_legacy_v1_date_checkpoint,
)

RUNNER_VERSION: Final = "foundation_frozen_touch_convergence_supplement_runner_v2"
PHASE: Final = "frozen_touch_convergence_supplement"
DATE_SCOPE: Final = "primary_71_sessions"
SELECTED_ENTRY_Q_ID: Final = "Q2_trail20_date_equal"
LEGACY_DYNAMIC_SEMANTICS: Final = "dynamic_anchor_per_second_sensitivity"
EPS_BP: Final = 1e-9
EXPECTED_LEGACY_SOURCE_COMMIT: Final = (
    "ace2669dbc3c725a94baa035494b9bef10701304"
)
EXPECTED_LEGACY_REGISTRY_SHA256: Final = (
    "00b0147db5054b126046ef2d59b2ca2a43ad26e78d695a88fed13ffd8f151ec1"
)
EXPECTED_LEGACY_FINAL_FINGERPRINT: Final = (
    "3425776cc651c26c8f1ec47eddfeffd9f127c6d03f96a8cab42fd14d2c35983c"
)
EXPECTED_LEGACY_FINAL_PAYLOAD_SHA256: Final = (
    "6bb4c2ee97e77e59912ec57c90396af3ac92bb269ddacdae394b0feccc642be4"
)
EXPECTED_LEGACY_BOUNDARY_FINGERPRINT: Final = (
    "784d40b868f5fbbe894b574c2c650be515f9c7c5c55496496ff833705197a560"
)
EXPECTED_LEGACY_BOUNDARY_PAYLOAD_SHA256: Final = (
    "5e551e00f9c2d13226cab6d4fd77d93639c3a17283c12750d59f0a6b0143b751"
)
RESOLVED_FROZEN_CANDIDATES: Final = frozenset(
    {"C2_conditional_reach80", "C3_conditional_reach50"}
)

DEFAULT_SOURCE_ROOT: Final = LEGACY_OUTPUT_ROOT_V1
DEFAULT_SOURCE_WORK_ROOT: Final = LEGACY_OUTPUT_ROOT_V1.with_name(
    f".{LEGACY_OUTPUT_ROOT_V1.name}.work"
)
DEFAULT_OUTPUT_ROOT: Final = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "foundation_selection_s05_frozen_convergence_20260826_v2"
)
DEFAULT_WORK_ROOT: Final = DEFAULT_OUTPUT_ROOT.with_name(
    f".{DEFAULT_OUTPUT_ROOT.name}.work"
)

UPSTREAM_FILES: Final = (
    ("complete.json", "legacy_v1_final_marker"),
    ("publication.json", "legacy_v1_publication"),
    ("s1_mother.parquet", "legacy_v1_s1_mother"),
    ("target_mapping.parquet", "legacy_v1_target_mapping"),
    ("s1_lookup_long.parquet", "legacy_v1_selected_entry_lookup"),
    (
        "convergence_candidate_review.parquet",
        "legacy_v1_dynamic_anchor_sensitivity",
    ),
)
OUTPUT_ARTIFACTS: Final = (
    "selected_q2_boundary_predictions.parquet",
    "boundary_migration_audit.parquet",
    "frozen_touch_facts.parquet",
    "frozen_registered_predictions.parquet",
    "frozen_effective_predictions.parquet",
    "frozen_primary_path_scores.parquet",
    "frozen_primary_cell_reach.parquet",
    "frozen_primary_reach_summary.parquet",
    "frozen_primary_reach_by_month.parquet",
    "frozen_primary_prediction_support.parquet",
    "dynamic_anchor_v1_sensitivity_review.parquet",
    "conditional_geometry_support_audit.parquet",
    "conditional_policy_geometry.parquet",
    "conditional_geometry_summary_overall.parquet",
    "conditional_geometry_summary_by_month.parquet",
    "conditional_geometry_summary_by_tod.parquet",
    "conditional_geometry_support_summary.parquet",
    "publication.json",
)


@dataclass(frozen=True)
class FrozenConvergencePaths:
    """Immutable v1 sources, raw daily inputs, work area, and v2 output."""

    source_root: Path = DEFAULT_SOURCE_ROOT
    source_work_root: Path = DEFAULT_SOURCE_WORK_ROOT
    sessions_path: Path = DEFAULT_SESSIONS_PATH
    daily_root: Path = DEFAULT_DAILY_ROOT
    registry_path: Path = DEFAULT_REGISTRY_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    work_root: Path = DEFAULT_WORK_ROOT

    @property
    def legacy_boundary_root(self) -> Path:
        return Path(self.source_work_root) / "boundary" / "final_panel"

    @property
    def legacy_fact_root(self) -> Path:
        return Path(self.source_work_root) / "convergence" / "facts" / "dates"


@dataclass(frozen=True)
class _UpstreamContext:
    marker: Mapping[str, object]
    boundary_marker: Mapping[str, object]
    publication: Mapping[str, object]
    sessions: tuple[str, ...]
    primary_sessions: tuple[str, ...]
    registry_sha256: str
    anchor_model_id: str


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_destinations(paths: FrozenConvergencePaths) -> None:
    destinations = (Path(paths.output_root), Path(paths.work_root))
    sources = (
        Path(paths.source_root),
        Path(paths.source_work_root),
        Path(paths.daily_root),
    )
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("output_root and work_root must differ")
    if _is_within(destinations[0], destinations[1]) or _is_within(
        destinations[1], destinations[0]
    ):
        raise ValueError("output_root and work_root must not be nested")
    for destination in destinations:
        for source in sources:
            if _is_within(destination, source):
                raise ValueError("supplement destinations must remain outside source roots")


def _require_unchanged_git_state(
    initial: Mapping[str, object],
    *,
    canonical: bool,
) -> None:
    if canonical and dict(_git_state()) != dict(initial):
        raise ValueError("source tree changed during frozen convergence publication")


def _load_context(paths: FrozenConvergencePaths) -> _UpstreamContext:
    registry = load_registry(paths.registry_path)
    if registry.sha256 != EXPECTED_REGISTRY_SHA256:
        raise ValueError("current frozen-convergence registry hash drifted")
    sessions = tuple(load_selection_sessions(paths.sessions_path))
    primary = tuple(date for date in sessions if date >= PRIMARY_START_DATE)
    if len(sessions) != EXPECTED_SESSION_COUNT or (
        len(primary) != EXPECTED_PRIMARY_SESSION_COUNT
    ):
        raise ValueError("frozen session cardinality drifted")
    if any(date >= PROTECTED_FORWARD_START_DATE for date in sessions):
        raise ValueError("protected-forward Date entered the source calendar")

    marker = verify_legacy_v1_date_checkpoint(paths.source_root, verify_inputs=False)
    boundary_marker = verify_legacy_v1_date_checkpoint(
        paths.legacy_boundary_root,
        verify_inputs=False,
    )
    if marker.get("phase") != "final_publish" or (
        boundary_marker.get("phase") != "boundary_final_panel"
    ):
        raise ValueError("legacy v1 checkpoint phase drifted")
    publication_path = Path(paths.source_root) / "publication.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    if not isinstance(publication, dict):
        raise TypeError("legacy publication.json must contain an object")
    required = {
        "registry_sha256",
        "source_commit",
        "selected_anchor_model_id",
        "selected_entry_q_candidate_id",
        "development_only",
        "primary_scope",
        "protected_forward_start",
    }
    missing = sorted(required - set(publication))
    if missing:
        raise ValueError(f"legacy publication metadata missing: {missing}")
    if (
        publication["development_only"] is not True
        or publication["primary_scope"] != DATE_SCOPE
        or publication["protected_forward_start"] != PROTECTED_FORWARD_START_DATE
        or publication["selected_entry_q_candidate_id"] != SELECTED_ENTRY_Q_ID
        or publication["registry_sha256"] != marker["registry_sha256"]
        or publication["source_commit"] != marker["source_commit"]
        or boundary_marker["source_commit"] != marker["source_commit"]
        or boundary_marker["registry_sha256"] != marker["registry_sha256"]
    ):
        raise ValueError("legacy foundation lineage or scope drifted")
    pinned = {
        "legacy source commit": (
            marker["source_commit"],
            EXPECTED_LEGACY_SOURCE_COMMIT,
        ),
        "legacy registry": (
            marker["registry_sha256"],
            EXPECTED_LEGACY_REGISTRY_SHA256,
        ),
        "legacy final fingerprint": (
            marker["checkpoint_fingerprint"],
            EXPECTED_LEGACY_FINAL_FINGERPRINT,
        ),
        "legacy final marker payload": (
            marker["marker_payload_sha256"],
            EXPECTED_LEGACY_FINAL_PAYLOAD_SHA256,
        ),
        "legacy boundary fingerprint": (
            boundary_marker["checkpoint_fingerprint"],
            EXPECTED_LEGACY_BOUNDARY_FINGERPRINT,
        ),
        "legacy boundary marker payload": (
            boundary_marker["marker_payload_sha256"],
            EXPECTED_LEGACY_BOUNDARY_PAYLOAD_SHA256,
        ),
    }
    drifted = [name for name, (actual, expected) in pinned.items() if actual != expected]
    if drifted:
        raise ValueError(f"canonical legacy v1 source drifted: {drifted}")
    return _UpstreamContext(
        marker=marker,
        boundary_marker=boundary_marker,
        publication=publication,
        sessions=sessions,
        primary_sessions=primary,
        registry_sha256=registry.sha256,
        anchor_model_id=str(publication["selected_anchor_model_id"]),
    )


def _preflight_paths_and_roles(
    paths: FrozenConvergencePaths,
) -> list[tuple[Path, str]]:
    records = [
        (Path(paths.registry_path), "frozen_v2_registry"),
        (Path(paths.sessions_path), "frozen_session_calendar"),
        *[
            (Path(paths.source_root) / name, role)
            for name, role in UPSTREAM_FILES
        ],
        (
            paths.legacy_boundary_root / "complete.json",
            "legacy_v1_boundary_checkpoint_marker",
        ),
        (
            paths.legacy_boundary_root / "predictions.parquet",
            "legacy_v1_boundary_predictions",
        ),
    ]
    return records


def _select_boundary_panel(
    paths: FrozenConvergencePaths,
    context: _UpstreamContext,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    source = pl.read_parquet(paths.legacy_boundary_root / "predictions.parquet")
    parts = [
        panel
        for date in context.sessions
        if not (
            panel := select_convergence_boundary_predictions(
                source,
                date=date,
                anchor_model_id=context.anchor_model_id,
                candidate_ids=(SELECTED_ENTRY_Q_ID,),
            )
        ).is_empty()
    ]
    if not parts:
        raise ValueError("legacy boundary checkpoint has no selected-Q2 support")
    selected = pl.concat(parts, how="vertical_relaxed").sort(
        [
            "Date",
            "ValueCode",
            "QuoteCode",
            "candidate_id",
            "tod_bucket",
            "boundary_quantile",
            "side",
        ]
    )
    lookup = (
        pl.scan_parquet(Path(paths.source_root) / "s1_lookup_long.parquet")
        .filter(
            (pl.col("anchor_model_id") == context.anchor_model_id)
            & (pl.col("candidate_id") == SELECTED_ENTRY_Q_ID)
        )
        .collect()
    )
    keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "anchor_model_id",
        "candidate_id",
        "tod_bucket",
        "boundary_quantile",
        "side",
    ]
    actual_primary = selected.filter(pl.col("Date") >= PRIMARY_START_DATE)
    if actual_primary.join(lookup.select(keys), on=keys, how="anti").height or (
        lookup.join(actual_primary.select(keys), on=keys, how="anti").height
    ):
        raise ValueError("legacy work boundary panel differs from published S1 lookup keys")
    paired = actual_primary.select(
        *keys,
        pl.col("boundary_distance_bp").alias("work_distance_bp"),
        pl.col("source_asof_date").alias("work_source_asof_date"),
    ).join(
        lookup.select(
            *keys,
            pl.col("boundary_distance_bp").alias("published_distance_bp"),
            pl.col("source_asof_date").alias("published_source_asof_date"),
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if paired.filter(
        (
            pl.col("work_distance_bp") - pl.col("published_distance_bp")
        ).abs()
        > EPS_BP
    ).height or paired.filter(
        pl.col("work_source_asof_date") != pl.col("published_source_asof_date")
    ).height:
        raise ValueError("legacy work and published boundary values differ")
    audit = pl.DataFrame(
        {
            "selected_anchor_model_id": [context.anchor_model_id],
            "selected_entry_q_candidate_id": [SELECTED_ENTRY_Q_ID],
            "history_boundary_rows": [selected.height],
            "history_boundary_dates": [selected["Date"].n_unique()],
            "history_product_days": [
                selected.select("Date", "ValueCode", "QuoteCode").unique().height
            ],
            "primary_boundary_rows": [actual_primary.height],
            "published_lookup_rows": [lookup.height],
            "primary_key_match": [True],
            "primary_value_match_tolerance_bp": [EPS_BP],
            "contains_target_day_outcome": [False],
            "protected_forward_read": [False],
        }
    )
    return selected, audit


def _preflight_partition(paths: FrozenConvergencePaths) -> Path:
    return Path(paths.work_root) / "preflight"


def _ensure_preflight(
    paths: FrozenConvergencePaths,
    context: _UpstreamContext,
    *,
    source_commit: str,
    initial_git_state: Mapping[str, object],
    canonical: bool,
    resume: bool,
) -> Mapping[str, object]:
    records = build_file_records(_preflight_paths_and_roles(paths))
    parameters = {
        "runner_version": RUNNER_VERSION,
        "migration": "legacy_v1_selected_Q2_boundary_checkpoint_to_frozen_v2",
        "legacy_final_checkpoint_fingerprint": context.marker[
            "checkpoint_fingerprint"
        ],
        "legacy_final_marker_payload_sha256": context.marker[
            "marker_payload_sha256"
        ],
        "legacy_boundary_checkpoint_fingerprint": context.boundary_marker[
            "checkpoint_fingerprint"
        ],
        "legacy_boundary_marker_payload_sha256": context.boundary_marker[
            "marker_payload_sha256"
        ],
        "selected_anchor_model_id": context.anchor_model_id,
        "selected_entry_q_candidate_id": SELECTED_ENTRY_Q_ID,
        "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
        "source_sessions": len(context.sessions),
        "primary_sessions": len(context.primary_sessions),
    }
    fingerprint = checkpoint_fingerprint(
        phase="frozen_convergence_preflight",
        date="frozen_131_sessions",
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        runner_version=RUNNER_VERSION,
    )
    partition = _preflight_partition(paths)
    if partition.exists():
        if not resume:
            raise FileExistsError(partition)
        marker = verify_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            verify_inputs=True,
            expected_runner_version=RUNNER_VERSION,
        )
        _require_unchanged_git_state(initial_git_state, canonical=canonical)
        return marker
    selected, audit = _select_boundary_panel(paths, context)
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    marker = atomic_publish_checkpoint(
        partition,
        phase="frozen_convergence_preflight",
        date="frozen_131_sessions",
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        artifact_values={
            "selected_q2_boundary_predictions.parquet": selected,
            "boundary_migration_audit.parquet": audit,
            "upstream_lineage.json": {
                "legacy_source_commit": context.marker["source_commit"],
                "legacy_final_checkpoint_fingerprint": context.marker[
                    "checkpoint_fingerprint"
                ],
                "legacy_final_marker_payload_sha256": context.marker[
                    "marker_payload_sha256"
                ],
                "legacy_boundary_checkpoint_fingerprint": context.boundary_marker[
                    "checkpoint_fingerprint"
                ],
                "legacy_boundary_marker_payload_sha256": context.boundary_marker[
                    "marker_payload_sha256"
                ],
                "legacy_dynamic_convergence_used_for_fit_or_selection": False,
            },
        },
        runner_version=RUNNER_VERSION,
    )
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    return marker


def _fact_partition(paths: FrozenConvergencePaths, date: str) -> Path:
    return Path(paths.work_root) / "facts" / "dates" / f"Date={date}"


def _daily_partition(paths: FrozenConvergencePaths, date: str) -> Path:
    if date >= PROTECTED_FORWARD_START_DATE:
        raise ValueError(f"protected-forward partition requested: {date}")
    return Path(paths.daily_root) / f"Date={date}"


def _legacy_fact_partition(paths: FrozenConvergencePaths, date: str) -> Path:
    return paths.legacy_fact_root / f"Date={date}"


def _validate_legacy_daily_lineage(
    paths: FrozenConvergencePaths,
    date: str,
    current_daily_records: Sequence[Mapping[str, object]],
) -> None:
    """Prove rebuilt raw inputs equal the bytes used by legacy Q2 facts."""

    marker = verify_legacy_v1_date_checkpoint(
        _legacy_fact_partition(paths, date),
        verify_inputs=True,
    )
    if (
        marker.get("phase") != "convergence_facts"
        or marker.get("date") != date
        or marker.get("source_commit") != EXPECTED_LEGACY_SOURCE_COMMIT
        or marker.get("registry_sha256") != EXPECTED_LEGACY_REGISTRY_SHA256
    ):
        raise ValueError(f"legacy convergence-fact lineage drifted: {date}")
    roles = {"daily_completion_marker", "daily_causal_fair_input"}
    legacy_by_role = {
        str(record["role"]): record
        for record in marker["input_records"]
        if str(record.get("role")) in roles
    }
    current_by_role = {
        str(record["role"]): record
        for record in current_daily_records
        if str(record.get("role")) in roles
    }
    if set(legacy_by_role) != roles or set(current_by_role) != roles:
        raise ValueError(f"legacy/current daily input roles are incomplete: {date}")
    identity = ("path", "path_scope", "bytes", "sha256")
    for role in sorted(roles):
        if any(
            legacy_by_role[role].get(key) != current_by_role[role].get(key)
            for key in identity
        ):
            raise ValueError(f"daily input differs from legacy Q2 source: {date} {role}")


def _ensure_fact_date(
    paths: FrozenConvergencePaths,
    context: _UpstreamContext,
    boundaries: pl.DataFrame,
    date: str,
    *,
    source_commit: str,
    initial_git_state: Mapping[str, object],
    canonical: bool,
    resume: bool,
) -> Mapping[str, object]:
    day_boundaries = boundaries.filter(pl.col("Date") == date)
    input_paths = [
        (_preflight_partition(paths) / "complete.json", "frozen_preflight_marker")
    ]
    if not day_boundaries.is_empty():
        daily = _daily_partition(paths, date)
        input_paths.extend(
            [
                (daily / "complete.json", "daily_completion_marker"),
                (daily / "causal_fair.parquet", "daily_causal_fair_input"),
                (
                    _legacy_fact_partition(paths, date) / "complete.json",
                    "legacy_v1_convergence_fact_marker",
                ),
            ]
        )
    records = build_file_records(input_paths)
    if not day_boundaries.is_empty():
        _validate_legacy_daily_lineage(paths, date, records)
    parameters = {
        "runner_version": RUNNER_VERSION,
        "selected_anchor_model_id": context.anchor_model_id,
        "selected_entry_q_candidate_id": SELECTED_ENTRY_Q_ID,
        "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
        "upper_touch_stop_second_exclusive": 14_400,
        "tracking_end_second": 15_600,
        "boundary_rows": day_boundaries.height,
        "read_columns": list(EPISODE_READ_COLUMNS),
    }
    fingerprint = checkpoint_fingerprint(
        phase="frozen_touch_facts",
        date=date,
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        runner_version=RUNNER_VERSION,
    )
    partition = _fact_partition(paths, date)
    if partition.exists():
        if not resume:
            raise FileExistsError(partition)
        marker = verify_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            verify_inputs=True,
            expected_runner_version=RUNNER_VERSION,
        )
        _require_unchanged_git_state(initial_git_state, canonical=canonical)
        return marker
    if day_boundaries.is_empty():
        facts = pl.DataFrame(schema=CONVERGENCE_FACT_SCHEMA)
    else:
        day = pl.read_parquet(
            _daily_partition(paths, date) / "causal_fair.parquet",
            columns=list(EPISODE_READ_COLUMNS),
        )
        materialized = materialize_anchor_column(day, context.anchor_model_id)
        facts = build_post_touch_convergence_facts(
            materialized,
            day_boundaries,
            anchor_column="selected_anchor_bp",
            anchor_model_id=context.anchor_model_id,
        )
        del day, materialized
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    marker = atomic_publish_checkpoint(
        partition,
        phase="frozen_touch_facts",
        date=date,
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        artifact_values={"post_touch_facts.parquet": facts},
        runner_version=RUNNER_VERSION,
    )
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    return marker


def _facts_index_partition(paths: FrozenConvergencePaths) -> Path:
    return Path(paths.work_root) / "facts" / "index"


def _ensure_facts_index(
    paths: FrozenConvergencePaths,
    context: _UpstreamContext,
    *,
    source_commit: str,
    initial_git_state: Mapping[str, object],
    canonical: bool,
    resume: bool,
) -> Mapping[str, object]:
    input_paths = [
        path_and_role
        for date in context.sessions
        for path_and_role in (
            (
                _fact_partition(paths, date) / "complete.json",
                "frozen_touch_fact_date_marker",
            ),
            (
                _fact_partition(paths, date) / "post_touch_facts.parquet",
                "frozen_touch_fact_date_artifact",
            ),
        )
    ]
    records = build_file_records(input_paths)
    parameters = {
        "runner_version": RUNNER_VERSION,
        "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
        "expected_date_checkpoints": len(context.sessions),
    }
    fingerprint = checkpoint_fingerprint(
        phase="frozen_touch_facts_index",
        date="frozen_131_sessions",
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        runner_version=RUNNER_VERSION,
    )
    partition = _facts_index_partition(paths)
    if partition.exists():
        if not resume:
            raise FileExistsError(partition)
        marker = verify_date_checkpoint(
            partition,
            expected_fingerprint=fingerprint,
            verify_inputs=True,
            expected_runner_version=RUNNER_VERSION,
        )
        _require_unchanged_git_state(initial_git_state, canonical=canonical)
        return marker
    manifest_rows = []
    for date in context.sessions:
        marker = verify_date_checkpoint(
            _fact_partition(paths, date),
            expected_runner_version=RUNNER_VERSION,
        )
        artifact = marker["artifacts"]["post_touch_facts.parquet"]
        manifest_rows.append(
            {
                "Date": date,
                "rows": int(artifact["rows"]),
                "bytes": int(artifact["bytes"]),
                "sha256": str(artifact["sha256"]),
                "marker_payload_sha256": str(marker["marker_payload_sha256"]),
                "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
            }
        )
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    marker = atomic_publish_checkpoint(
        partition,
        phase="frozen_touch_facts_index",
        date="frozen_131_sessions",
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        artifact_values={"facts_manifest.parquet": pl.from_dicts(manifest_rows)},
        runner_version=RUNNER_VERSION,
    )
    _require_unchanged_git_state(initial_git_state, canonical=canonical)
    return marker


def _load_all_facts(
    paths: FrozenConvergencePaths,
    context: _UpstreamContext,
) -> pl.DataFrame:
    parts = []
    for date in context.sessions:
        verify_date_checkpoint(
            _fact_partition(paths, date),
            verify_inputs=True,
            expected_runner_version=RUNNER_VERSION,
        )
        parts.append(
            pl.read_parquet(_fact_partition(paths, date) / "post_touch_facts.parquet")
        )
    return pl.concat(
        parts,
        how="vertical_relaxed",
    ).sort(
        [
            "Date",
            "ValueCode",
            "QuoteCode",
            "candidate_id",
            "boundary_quantile",
            "episode_sequence",
        ]
    )


def _empty_with_schema(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _build_predictions_and_scores(
    context: _UpstreamContext,
    boundaries: pl.DataFrame,
    facts: pl.DataFrame,
    primary_mother: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    from .foundation_convergence_lookup import (
        CONVERGENCE_PATH_SCORE_SCHEMA,
        CONVERGENCE_PREDICTION_SCHEMA,
    )

    prepared = prepare_convergence_history(facts, context.sessions)
    registered_parts: list[pl.DataFrame] = []
    effective_parts: list[pl.DataFrame] = []
    path_parts: list[pl.DataFrame] = []
    cell_parts: list[pl.DataFrame] = []
    support_parts: list[pl.DataFrame] = []
    mother_keys = primary_mother.select("Date", "ValueCode", "QuoteCode").unique()
    for index, date in enumerate(context.primary_sessions, start=1):
        day_mother = mother_keys.filter(pl.col("Date") == date)
        day_boundaries = boundaries.filter(pl.col("Date") == date).join(
            day_mother,
            on=["Date", "ValueCode", "QuoteCode"],
            how="semi",
        )
        expected_boundary_rows = day_mother.height * 4 * 3 * 2
        if day_boundaries.height != expected_boundary_rows:
            raise ValueError(f"selected-Q2 boundary grid differs from S1 mother: {date}")
        targets = (
            day_boundaries.filter(pl.col("side") == "positive")
            .select(*TARGET_LINEAGE_KEYS)
            .unique()
        )
        if targets.is_empty():
            raise ValueError(f"primary Date has no selected-Q2 targets: {date}")
        registered = build_registered_conditional_convergence_grid_prepared(
            prepared,
            targets,
        )
        resolved = resolve_convergence_fallback(
            registered,
            primary_lookup_id=CONVERGENCE_PRIMARY_LOOKUP_ID,
            fallback_lookup_id=CONVERGENCE_FALLBACK_LOOKUP_ID,
        )
        controls = build_control_convergence_predictions_from_boundaries(
            day_boundaries
        )
        effective = pl.concat([resolved, controls], how="vertical_relaxed").sort(
            [*TARGET_LINEAGE_KEYS, "convergence_candidate_id"]
        )
        score = score_convergence_predictions(
            facts.filter(pl.col("Date") == date).join(
                day_mother,
                on=["Date", "ValueCode", "QuoteCode"],
                how="semi",
            ),
            effective,
        )
        support = build_convergence_support_audit(
            targets,
            registered,
            resolved,
            controls,
        )
        registered_parts.append(registered)
        effective_parts.append(effective)
        path_parts.append(score.path_facts)
        cell_parts.append(score.summary)
        support_parts.append(support)
        if index == 1 or index % 10 == 0 or index == len(context.primary_sessions):
            print(
                f"frozen predictions {index}/{len(context.primary_sessions)} {date}",
                flush=True,
            )
    registered = (
        pl.concat(registered_parts, how="vertical_relaxed")
        if registered_parts
        else _empty_with_schema(CONVERGENCE_PREDICTION_SCHEMA)
    )
    effective = (
        pl.concat(effective_parts, how="vertical_relaxed")
        if effective_parts
        else _empty_with_schema(CONVERGENCE_PREDICTION_SCHEMA)
    )
    paths = (
        pl.concat(path_parts, how="vertical_relaxed")
        if path_parts
        else _empty_with_schema(CONVERGENCE_PATH_SCORE_SCHEMA)
    )
    cells = pl.concat(cell_parts, how="vertical_relaxed")
    support = pl.concat(support_parts, how="vertical_relaxed")
    return registered, effective, paths, cells, support


def _summarize_frozen_primary(path_scores: pl.DataFrame) -> pl.DataFrame:
    """Summarize plain-ID C2/C3 as the registered trail20→trail60 route."""

    summary = summarize_convergence_primary(path_scores)
    is_resolved = pl.col("convergence_candidate_id").is_in(
        list(RESOLVED_FROZEN_CANDIDATES)
    )
    return summary.with_columns(
        pl.when(is_resolved)
        .then(pl.col("trail60_effective_paths"))
        .otherwise(pl.col("fallback_effective_paths"))
        .alias("fallback_effective_paths"),
        pl.when(is_resolved)
        .then(pl.col("trail60_effective_paths") / pl.col("n_started"))
        .otherwise(pl.col("fallback_source_share"))
        .alias("fallback_source_share"),
    )


def _reach_by_month(path_scores: pl.DataFrame) -> pl.DataFrame:
    parts = []
    months = sorted(
        path_scores.select(pl.col("Date").str.slice(0, 6).alias("month"))[
            "month"
        ]
        .unique()
        .to_list()
    )
    for month in months:
        review = _summarize_frozen_primary(
            path_scores.filter(pl.col("Date").str.starts_with(month))
        ).with_columns(pl.lit(month).alias("month"))
        parts.append(review)
    return pl.concat(parts, how="vertical_relaxed")


def _validate_frozen_fact_reference(facts: pl.DataFrame) -> None:
    if facts.is_empty():
        raise ValueError("frozen convergence facts are empty")
    if facts.filter(
        (pl.col("convergence_reference_semantics") != CONVERGENCE_REFERENCE_SEMANTICS)
        | (
            (
                pl.col("touch_anchor_basis_bp")
                - pl.col("frozen_center_basis_bp")
            ).abs()
            > EPS_BP
        )
        | (
            (
                pl.col("frozen_center_basis_bp")
                - pl.col("independent_lower_distance_bp")
                - pl.col("frozen_independent_lower_basis_bp")
            ).abs()
            > EPS_BP
        )
    ).height:
        raise ValueError("frozen fact reference formulas drifted")


def _validate_frozen_path_reference(path_scores: pl.DataFrame) -> None:
    if path_scores.filter(
        (pl.col("convergence_reference_semantics") != CONVERGENCE_REFERENCE_SEMANTICS)
        | (
            (
                pl.col("frozen_center_basis_bp")
                - pl.col("threshold_distance_bp")
                - pl.col("frozen_exit_basis_bp")
            ).abs()
            > EPS_BP
        )
    ).height:
        raise ValueError("frozen path-score exit formula drifted")


def _validate_frozen_frames(
    facts: pl.DataFrame,
    effective: pl.DataFrame,
    path_scores: pl.DataFrame,
    reach: pl.DataFrame,
    support_audit: pl.DataFrame,
    geometry: pl.DataFrame,
    *,
    primary_mother_rows: int,
) -> None:
    _validate_frozen_fact_reference(facts)
    validate_convergence_prediction_lineage(effective)
    expected_effective_rows = primary_mother_rows * 4 * 3 * 4
    if effective.height != expected_effective_rows:
        raise ValueError("effective prediction grid differs from the S1 mother")
    if effective.filter(pl.col("contains_target_day_outcome")).height:
        raise ValueError("D-safe effective predictions contain target-day outcomes")
    duplicate = effective.group_by(
        [*TARGET_LINEAGE_KEYS, "convergence_candidate_id"]
    ).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("effective frozen predictions contain duplicate keys")
    if set(effective["convergence_candidate_id"].unique().to_list()) != {
        "C0_center",
        "C1_independent_lower_control",
        "C2_conditional_reach80",
        "C3_conditional_reach50",
    }:
        raise ValueError("effective frozen prediction candidate set drifted")
    c0 = effective.filter(pl.col("convergence_candidate_id") == "C0_center")
    if c0.filter(
        ~pl.col("effective_supported")
        | (pl.col("effective_threshold_distance_bp").abs() > 1e-12)
        | (pl.col("effective_lookup_id") != "structural_zero")
    ).height:
        raise ValueError("C0 is not a full structural-zero prediction")
    _validate_frozen_path_reference(path_scores)
    if path_scores.filter(~pl.col("contains_target_day_outcome")).height:
        raise ValueError("evaluation path scores lack target-day outcome labeling")
    if set(reach["convergence_candidate_id"].unique().to_list()) != {
        "C0_center",
        "C1_independent_lower_control",
        "C2_conditional_reach80",
        "C3_conditional_reach50",
    }:
        raise ValueError("frozen reach summary candidate set drifted")
    if reach.filter(~pl.col("contains_target_day_outcome")).height:
        raise ValueError("reach summary lacks target-day outcome labeling")
    resolved_reach = reach.filter(
        pl.col("convergence_candidate_id").is_in(
            list(RESOLVED_FROZEN_CANDIDATES)
        )
    )
    if resolved_reach.filter(
        (pl.col("fallback_effective_paths") != pl.col("trail60_effective_paths"))
        | (
            (
                pl.col("fallback_source_share")
                - pl.col("trail60_effective_paths") / pl.col("n_started")
            ).abs()
            > 1e-12
        )
    ).height:
        raise ValueError("resolved C2/C3 fallback summary drifted")
    expected_support_rows = primary_mother_rows * 4 * 3 * len(
        CONDITIONAL_CANDIDATES
    )
    if support_audit.height != expected_support_rows:
        raise ValueError("conditional support audit denominator drifted")
    if support_audit.filter(pl.col("contains_target_day_outcome")).height:
        raise ValueError("conditional support audit contains target-day outcomes")
    if geometry.filter(pl.col("contains_target_day_outcome")).height:
        raise ValueError("conditional geometry contains target-day outcomes")
    denominator = primary_mother_rows * 4
    support_counts = support_audit.group_by(
        "boundary_quantile", "convergence_candidate_id"
    ).len()
    if support_counts.height != 3 * len(CONDITIONAL_CANDIDATES) or (
        support_counts.filter(pl.col("len") != denominator).height
    ):
        raise ValueError("conditional policy mother denominators drifted")
    supported_keys = support_audit.filter(pl.col("conditional_supported")).select(
        *TARGET_LINEAGE_KEYS,
        "convergence_candidate_id",
    )
    geometry_keys = geometry.select(
        *TARGET_LINEAGE_KEYS,
        "convergence_candidate_id",
    )
    if geometry_keys.join(
        supported_keys,
        on=[*TARGET_LINEAGE_KEYS, "convergence_candidate_id"],
        how="anti",
    ).height or supported_keys.join(
        geometry_keys,
        on=[*TARGET_LINEAGE_KEYS, "convergence_candidate_id"],
        how="anti",
    ).height:
        raise ValueError("conditional geometry differs from supported cells")


def _dynamic_sensitivity(paths: FrozenConvergencePaths) -> pl.DataFrame:
    return (
        pl.read_parquet(
            Path(paths.source_root) / "convergence_candidate_review.parquet"
        )
        .filter(pl.col("candidate_id") == SELECTED_ENTRY_Q_ID)
        .with_columns(
            pl.lit(LEGACY_DYNAMIC_SEMANTICS).alias(
                "convergence_reference_semantics"
            ),
            pl.lit(False).alias("canonical_decision_evidence"),
            pl.lit("superseded_by_frozen_touch_v2").alias("result_role"),
        )
    )


def _add_geometry_positive_shares(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "product_day_tod_rows",
        "supported_cell_rows",
        "nominal_same_day_known_cost_positive_rows",
        "nominal_overnight_known_cost_positive_rows",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"geometry summary lacks positive-share inputs: {missing}")
    if frame.filter(
        pl.col("product_day_tod_rows").fill_null(0)
        != pl.col("supported_cell_rows")
    ).height:
        raise ValueError("geometry rows differ from supported-cell denominator")
    return frame.with_columns(
        pl.when(pl.col("supported_cell_rows") > 0)
        .then(
            pl.col("nominal_same_day_known_cost_positive_rows")
            / pl.col("supported_cell_rows")
        )
        .otherwise(None)
        .alias("nominal_same_day_known_cost_positive_share"),
        pl.when(pl.col("supported_cell_rows") > 0)
        .then(
            pl.col("nominal_overnight_known_cost_positive_rows")
            / pl.col("supported_cell_rows")
        )
        .otherwise(None)
        .alias("nominal_overnight_known_cost_positive_share"),
    )


def _final_paths_and_roles(
    paths: FrozenConvergencePaths,
    boundary_dates: Sequence[str],
    sessions: Sequence[str],
) -> list[tuple[Path, str]]:
    records = _preflight_paths_and_roles(paths)
    records.extend(
        [
            (
                _facts_index_partition(paths) / "complete.json",
                "frozen_touch_facts_index_marker",
            ),
            (
                _facts_index_partition(paths) / "facts_manifest.parquet",
                "frozen_touch_facts_manifest",
            ),
        ]
    )
    for date in sessions:
        records.extend(
            [
                (
                    _fact_partition(paths, date) / "complete.json",
                    "frozen_touch_fact_date_marker",
                ),
                (
                    _fact_partition(paths, date) / "post_touch_facts.parquet",
                    "frozen_touch_fact_date_artifact",
                ),
            ]
        )
    for date in boundary_dates:
        daily = _daily_partition(paths, date)
        records.extend(
            [
                (daily / "complete.json", "daily_completion_marker"),
                (daily / "causal_fair.parquet", "daily_causal_fair_input"),
                (
                    _legacy_fact_partition(paths, date) / "complete.json",
                    "legacy_v1_convergence_fact_marker",
                ),
            ]
        )
    return records


def _publication_parameters(
    context: _UpstreamContext,
    *,
    canonical: bool,
) -> dict[str, object]:
    return {
        "runner_version": RUNNER_VERSION,
        "conditional_geometry_version": CONDITIONAL_GEOMETRY_VERSION,
        "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
        "frozen_reference_proxy": "anchor_at_first_legal_upper_touch",
        "not_exact_s1_submit_tick": True,
        "selected_anchor_model_id": context.anchor_model_id,
        "selected_entry_q_candidate_id": SELECTED_ENTRY_Q_ID,
        "registered_lookups": [
            CONVERGENCE_PRIMARY_LOOKUP_ID,
            CONVERGENCE_FALLBACK_LOOKUP_ID,
        ],
        "resolved_lookup_route": "trail20_date_equal_then_trail60_date_equal",
        "published_candidates": list(CONDITIONAL_CANDIDATES),
        "scored_candidates": [
            "C0_center",
            "C1_independent_lower_control",
            "C2_conditional_reach80",
            "C3_conditional_reach50",
        ],
        "geometry_candidates": list(CONDITIONAL_CANDIDATES),
        "legacy_dynamic_convergence_used_for_fit_or_selection": False,
        "legacy_dynamic_sensitivity_copied": True,
        "legacy_dynamic_results_role": "sensitivity_only",
        "source_sessions": len(context.sessions),
        "primary_sessions": len(context.primary_sessions),
        "facts_scope": "131_session_selected_Q2_boundary_history",
        "prediction_and_reach_scope": "71_session_s1_primary_mother",
        "geometry_scope": "71_session_s1_primary_mother",
        "protected_forward_start": PROTECTED_FORWARD_START_DATE,
        "canonical_source_tree": canonical,
        "development_only": True,
        "deployment_baseline_approved": False,
        "actionable_execution": False,
        "ev_ready": False,
    }


def publish_frozen_convergence_supplement(
    paths: FrozenConvergencePaths | None = None,
    *,
    resume: bool = True,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Rebuild frozen facts and atomically publish the complete v2 supplement."""

    selected_paths = paths or FrozenConvergencePaths()
    _validate_destinations(selected_paths)
    initial_state = dict(_require_canonical_git_state(canonical))
    source_commit = str(initial_state.get("commit", ""))
    context = _load_context(selected_paths)
    _ensure_preflight(
        selected_paths,
        context,
        source_commit=source_commit,
        initial_git_state=initial_state,
        canonical=canonical,
        resume=resume,
    )
    boundaries = pl.read_parquet(
        _preflight_partition(selected_paths)
        / "selected_q2_boundary_predictions.parquet"
    )
    for index, date in enumerate(context.sessions, start=1):
        existed = _fact_partition(selected_paths, date).exists()
        _ensure_fact_date(
            selected_paths,
            context,
            boundaries,
            date,
            source_commit=source_commit,
            initial_git_state=initial_state,
            canonical=canonical,
            resume=resume,
        )
        if index == 1 or index % 10 == 0 or index == len(context.sessions):
            action = "resumed" if existed and resume else "complete"
            print(
                f"frozen facts {index}/{len(context.sessions)} {date} {action}",
                flush=True,
            )
    _ensure_facts_index(
        selected_paths,
        context,
        source_commit=source_commit,
        initial_git_state=initial_state,
        canonical=canonical,
        resume=resume,
    )

    boundary_dates = sorted(boundaries["Date"].unique().to_list())
    records = build_file_records(
        _final_paths_and_roles(selected_paths, boundary_dates, context.sessions)
    )
    parameters = _publication_parameters(context, canonical=canonical)
    fingerprint = checkpoint_fingerprint(
        phase=PHASE,
        date=DATE_SCOPE,
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        runner_version=RUNNER_VERSION,
    )
    if Path(selected_paths.output_root).exists():
        if not resume:
            raise FileExistsError(selected_paths.output_root)
        marker = verify_date_checkpoint(
            selected_paths.output_root,
            expected_fingerprint=fingerprint,
            verify_inputs=True,
            expected_runner_version=RUNNER_VERSION,
        )
        _require_unchanged_git_state(initial_state, canonical=canonical)
        verify_frozen_convergence_supplement(selected_paths, canonical=canonical)
        return marker

    facts = _load_all_facts(selected_paths, context)
    mother = pl.read_parquet(Path(selected_paths.source_root) / "s1_mother.parquet")
    primary_mother = mother.filter(pl.col("s1_primary"))
    registered, effective, path_scores, cell_reach, prediction_support = (
        _build_predictions_and_scores(
            context,
            boundaries,
            facts,
            primary_mother,
        )
    )
    reach = _summarize_frozen_primary(path_scores)
    reach_by_month = _reach_by_month(path_scores)
    mapping = pl.read_parquet(Path(selected_paths.source_root) / "target_mapping.parquet")
    lookup = pl.read_parquet(Path(selected_paths.source_root) / "s1_lookup_long.parquet")
    geometry_result = build_conditional_lookup_geometry(
        mother,
        mapping,
        lookup,
        effective,
    )
    primary_mother_rows = primary_mother.height
    _validate_frozen_frames(
        facts,
        effective,
        path_scores,
        reach,
        geometry_result.support_audit,
        geometry_result.geometry_long,
        primary_mother_rows=primary_mother_rows,
    )
    migration_audit = pl.read_parquet(
        _preflight_partition(selected_paths) / "boundary_migration_audit.parquet"
    )
    dynamic_sensitivity = _dynamic_sensitivity(selected_paths)
    if dynamic_sensitivity.is_empty() or dynamic_sensitivity.filter(
        pl.col("canonical_decision_evidence")
        | (pl.col("result_role") != "superseded_by_frozen_touch_v2")
        | (pl.col("convergence_reference_semantics") != LEGACY_DYNAMIC_SEMANTICS)
    ).height:
        raise ValueError("legacy dynamic sensitivity quarantine drifted")
    support_overall = geometry_result.support_summary.filter(
        pl.col("summary_scope") == "overall"
    )
    geometry_overall = _add_geometry_positive_shares(
        geometry_result.summary_overall.filter(
            pl.col("summary_scope") == "overall"
        ).join(
            support_overall.select("policy_id", "supported_cell_rows"),
            on="policy_id",
            how="inner",
            validate="1:1",
        )
    )
    publication = {
        **parameters,
        "supplement_source_commit": source_commit,
        "registry_sha256": context.registry_sha256,
        "legacy_source_commit": context.marker["source_commit"],
        "legacy_registry_sha256": context.marker["registry_sha256"],
        "legacy_final_checkpoint_fingerprint": context.marker[
            "checkpoint_fingerprint"
        ],
        "legacy_final_marker_payload_sha256": context.marker[
            "marker_payload_sha256"
        ],
        "legacy_boundary_checkpoint_fingerprint": context.boundary_marker[
            "checkpoint_fingerprint"
        ],
        "legacy_boundary_marker_payload_sha256": context.boundary_marker[
            "marker_payload_sha256"
        ],
        "boundary_rows": boundaries.height,
        "frozen_fact_rows": facts.height,
        "registered_prediction_rows": registered.height,
        "effective_prediction_rows": effective.height,
        "path_score_rows": path_scores.height,
        "cell_reach_rows": cell_reach.height,
        "prediction_support_rows": prediction_support.height,
        "dynamic_sensitivity_rows": dynamic_sensitivity.height,
        "primary_mother_rows": primary_mother_rows,
        "conditional_support_rows": geometry_result.support_audit.height,
        "conditional_geometry_rows": geometry_result.geometry_long.height,
        "conditional_geometry_overall_rows": (
            geometry_result.summary_overall.height
        ),
        "conditional_geometry_month_rows": geometry_result.summary_by_month.height,
        "conditional_geometry_tod_rows": geometry_result.summary_by_tod.height,
        "conditional_support_summary_rows": geometry_result.support_summary.height,
        "reach_summary": reach.select(
            "boundary_quantile",
            "convergence_candidate_id",
            "n_started",
            "confirmed_hits",
            "known_misses",
            "unknown_censored",
            "reach_lower_bound",
            "reach_upper_bound",
            "time_to_hit_from_touch_p50_seconds",
            "time_to_hit_from_touch_p90_seconds",
            "fallback_source_share",
        ).to_dicts(),
        "conditional_support_summary": support_overall.select(
            "policy_id",
            "mother_cell_rows",
            "supported_cell_rows",
            "conditional_support_coverage",
            "fallback_supported_rows",
            "fallback_share_supported",
            "zero_lower_supported_rows",
        ).to_dicts(),
        "conditional_geometry_summary": geometry_overall.select(
            "policy_id",
            "product_day_tod_rows",
            "lower_distance_bp_p50",
            "nominal_same_day_known_cost_margin_bp_p50",
            "nominal_same_day_known_cost_positive_share",
            "nominal_overnight_known_cost_margin_bp_p50",
            "nominal_overnight_known_cost_positive_share",
        ).to_dicts(),
        "prepublish_validation": "passed",
        "selection_contains_development_outcomes": True,
        "contains_target_day_outcome": True,
        "facts_and_scores_contain_target_day_outcome": True,
        "predictions_contain_target_day_outcome": False,
        "geometry_contains_target_day_outcome": False,
        "reference_diagnostic": True,
    }
    _require_unchanged_git_state(initial_state, canonical=canonical)
    marker = atomic_publish_checkpoint(
        selected_paths.output_root,
        phase=PHASE,
        date=DATE_SCOPE,
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        artifact_values={
            "selected_q2_boundary_predictions.parquet": boundaries,
            "boundary_migration_audit.parquet": migration_audit,
            "frozen_touch_facts.parquet": facts,
            "frozen_registered_predictions.parquet": registered,
            "frozen_effective_predictions.parquet": effective,
            "frozen_primary_path_scores.parquet": path_scores,
            "frozen_primary_cell_reach.parquet": cell_reach,
            "frozen_primary_reach_summary.parquet": reach,
            "frozen_primary_reach_by_month.parquet": reach_by_month,
            "frozen_primary_prediction_support.parquet": prediction_support,
            "dynamic_anchor_v1_sensitivity_review.parquet": dynamic_sensitivity,
            "conditional_geometry_support_audit.parquet": (
                geometry_result.support_audit
            ),
            "conditional_policy_geometry.parquet": geometry_result.geometry_long,
            "conditional_geometry_summary_overall.parquet": (
                geometry_result.summary_overall
            ),
            "conditional_geometry_summary_by_month.parquet": (
                geometry_result.summary_by_month
            ),
            "conditional_geometry_summary_by_tod.parquet": (
                geometry_result.summary_by_tod
            ),
            "conditional_geometry_support_summary.parquet": (
                geometry_result.support_summary
            ),
            "publication.json": publication,
        },
        runner_version=RUNNER_VERSION,
    )
    _require_unchanged_git_state(initial_state, canonical=canonical)
    del facts, registered, effective, path_scores, cell_reach, prediction_support
    gc.collect()
    return marker


def verify_frozen_convergence_supplement(
    paths: FrozenConvergencePaths | None = None,
    *,
    canonical: bool = True,
) -> Mapping[str, object]:
    """Verify content hashes, raw inputs, lineage, support, and readiness flags."""

    selected_paths = paths or FrozenConvergencePaths()
    _validate_destinations(selected_paths)
    state = dict(_require_canonical_git_state(canonical))
    context = _load_context(selected_paths)
    publication_path = Path(selected_paths.output_root) / "publication.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    if not isinstance(publication, dict):
        raise TypeError("frozen supplement publication must be an object")
    source_commit = str(publication.get("supplement_source_commit", ""))
    if not source_commit:
        raise ValueError("frozen supplement source commit is missing")
    boundaries = pl.read_parquet(
        Path(selected_paths.output_root) / "selected_q2_boundary_predictions.parquet"
    )
    boundary_dates = sorted(boundaries["Date"].unique().to_list())
    records = build_file_records(
        _final_paths_and_roles(selected_paths, boundary_dates, context.sessions)
    )
    parameters = _publication_parameters(context, canonical=canonical)
    fingerprint = checkpoint_fingerprint(
        phase=PHASE,
        date=DATE_SCOPE,
        registry_sha256=context.registry_sha256,
        source_commit=source_commit,
        input_records=records,
        parameters=parameters,
        runner_version=RUNNER_VERSION,
    )
    marker = verify_date_checkpoint(
        selected_paths.output_root,
        expected_fingerprint=fingerprint,
        verify_inputs=True,
        expected_runner_version=RUNNER_VERSION,
    )
    if set(marker["artifacts"]) != set(OUTPUT_ARTIFACTS):
        raise ValueError("frozen supplement artifact set drifted")
    if (
        publication.get("registry_sha256") != context.registry_sha256
        or publication.get("legacy_final_checkpoint_fingerprint")
        != context.marker["checkpoint_fingerprint"]
        or publication.get("legacy_final_marker_payload_sha256")
        != context.marker["marker_payload_sha256"]
        or publication.get("legacy_boundary_checkpoint_fingerprint")
        != context.boundary_marker["checkpoint_fingerprint"]
        or publication.get("legacy_boundary_marker_payload_sha256")
        != context.boundary_marker["marker_payload_sha256"]
        or publication.get("convergence_reference_semantics")
        != CONVERGENCE_REFERENCE_SEMANTICS
        or publication.get("legacy_dynamic_convergence_used_for_fit_or_selection")
        is not False
        or publication.get("legacy_dynamic_sensitivity_copied") is not True
        or publication.get("development_only") is not True
        or publication.get("deployment_baseline_approved") is not False
        or publication.get("actionable_execution") is not False
        or publication.get("ev_ready") is not False
        or publication.get("prepublish_validation") != "passed"
        or publication.get("contains_target_day_outcome") is not True
        or publication.get("facts_and_scores_contain_target_day_outcome") is not True
        or publication.get("predictions_contain_target_day_outcome") is not False
        or publication.get("geometry_contains_target_day_outcome") is not False
    ):
        raise ValueError("frozen supplement publication lineage drifted")
    root = Path(selected_paths.output_root)
    facts = pl.read_parquet(root / "frozen_touch_facts.parquet")
    effective = pl.read_parquet(root / "frozen_effective_predictions.parquet")
    path_scores = pl.read_parquet(root / "frozen_primary_path_scores.parquet")
    reach = pl.read_parquet(root / "frozen_primary_reach_summary.parquet")
    support = pl.read_parquet(root / "conditional_geometry_support_audit.parquet")
    geometry = pl.read_parquet(root / "conditional_policy_geometry.parquet")
    dynamic_sensitivity = pl.read_parquet(
        root / "dynamic_anchor_v1_sensitivity_review.parquet"
    )
    if dynamic_sensitivity.is_empty() or dynamic_sensitivity.filter(
        pl.col("canonical_decision_evidence")
        | (pl.col("result_role") != "superseded_by_frozen_touch_v2")
        | (pl.col("convergence_reference_semantics") != LEGACY_DYNAMIC_SEMANTICS)
    ).height:
        raise ValueError("published dynamic sensitivity quarantine drifted")
    primary_mother_rows = (
        pl.scan_parquet(Path(selected_paths.source_root) / "s1_mother.parquet")
        .filter(pl.col("s1_primary"))
        .select(pl.len())
        .collect()
        .item()
    )
    _validate_frozen_frames(
        facts,
        effective,
        path_scores,
        reach,
        support,
        geometry,
        primary_mother_rows=int(primary_mother_rows),
    )
    if (
        int(publication["boundary_rows"]) != boundaries.height
        or int(publication["frozen_fact_rows"]) != facts.height
        or int(publication["effective_prediction_rows"]) != effective.height
        or int(publication["path_score_rows"]) != path_scores.height
        or int(publication["conditional_support_rows"]) != support.height
        or int(publication["conditional_geometry_rows"]) != geometry.height
    ):
        raise ValueError("frozen supplement publication row counts drifted")
    artifact_count_fields = {
        "registered_prediction_rows": "frozen_registered_predictions.parquet",
        "cell_reach_rows": "frozen_primary_cell_reach.parquet",
        "prediction_support_rows": "frozen_primary_prediction_support.parquet",
        "dynamic_sensitivity_rows": "dynamic_anchor_v1_sensitivity_review.parquet",
        "conditional_geometry_overall_rows": (
            "conditional_geometry_summary_overall.parquet"
        ),
        "conditional_geometry_month_rows": (
            "conditional_geometry_summary_by_month.parquet"
        ),
        "conditional_geometry_tod_rows": "conditional_geometry_summary_by_tod.parquet",
        "conditional_support_summary_rows": (
            "conditional_geometry_support_summary.parquet"
        ),
    }
    for field, artifact_name in artifact_count_fields.items():
        if int(publication[field]) != int(marker["artifacts"][artifact_name]["rows"]):
            raise ValueError(f"frozen supplement row count drifted: {field}")
    return {
        "runner_version": RUNNER_VERSION,
        "checkpoint_fingerprint": marker["checkpoint_fingerprint"],
        "marker_payload_sha256": marker["marker_payload_sha256"],
        "source_commit": source_commit,
        "verifier_commit": str(state.get("commit", "")),
        "registry_sha256": context.registry_sha256,
        "legacy_checkpoint_fingerprint": context.marker["checkpoint_fingerprint"],
        "boundary_rows": boundaries.height,
        "frozen_fact_rows": facts.height,
        "effective_prediction_rows": effective.height,
        "path_score_rows": path_scores.height,
        "conditional_support_rows": support.height,
        "conditional_geometry_rows": geometry.height,
        "convergence_reference_semantics": CONVERGENCE_REFERENCE_SEMANTICS,
        "development_only": True,
        "deployment_baseline_approved": False,
        "actionable_execution": False,
        "ev_ready": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("publish", "verify-only"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-work-root",
        type=Path,
        default=DEFAULT_SOURCE_WORK_ROOT,
    )
    parser.add_argument("--sessions-path", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Publish or independently verify the frozen convergence supplement."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.phase == "publish" and not args.execute:
        parser.error("phase 'publish' requires explicit --execute")
    paths = FrozenConvergencePaths(
        source_root=args.source_root,
        source_work_root=args.source_work_root,
        sessions_path=args.sessions_path,
        daily_root=args.daily_root,
        registry_path=args.registry_path,
        output_root=args.output_root,
        work_root=args.work_root,
    )
    canonical = not args.allow_dirty_development_run
    if args.phase == "publish":
        result = publish_frozen_convergence_supplement(
            paths,
            resume=not args.no_resume,
            canonical=canonical,
        )
    else:
        result = verify_frozen_convergence_supplement(paths, canonical=canonical)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
