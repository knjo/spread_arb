"""Source-bound orchestration for the current-ladder D+1 carry benchmark.

This module is intentionally only an adapter around the existing
``overnight_carry_runner`` and ``overnight_carry_report`` producers.  It fixes
the research policy to the first jointly executable exact-contract spot-bid /
future-ask close in the next observed session, with both books no older than
one second.  It does not reimplement carry labels or report arithmetic.

The entry cohort comes from the immutable entry and same-day root manifests.
It is never inferred by taking the last 60 rows of the longer prerequisite
calendar.  The latter is used only as the observation calendar, so the final
entry date can still reach its true D+1 session.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .cross_session_prerequisite import verify_cross_session_prerequisites
from .execution_runner import (
    _verify_complete_partition as _verify_execution_output_partition,
)
from .overnight_carry import OvernightCarryConfig
from . import overnight_carry_report as _carry_report
from .overnight_carry_report import run_overnight_carry_report
from .exit_maker_runner import (
    _verify_output_partition as _verify_exit_maker_output_partition,
)
from .overnight_carry_runner import (
    DEFAULT_FUTURES_RAW_ROOT,
    OVERNIGHT_CARRY_MANIFEST_NAME,
    OvernightCarryRunnerConfig,
    _OUTPUT_ARTIFACTS as _OVERNIGHT_OUTPUT_ARTIFACTS,
    _candidate_raw_fingerprints,
    _frame_sha256 as _runner_frame_sha256,
    _runner_payload as _build_runner_payload,
    run_overnight_carry_replay,
    verify_overnight_output_partition,
    verify_overnight_sources,
)


ORCHESTRATOR_VERSION = "current_ladder_d1_fresh1s_overnight_v1"
BUNDLE_SCHEMA_VERSION = "current_ladder_d1_fresh1s_overnight_bundle_v1"
POLICY_VERSION = (
    "first_joint_raw_taker_close_exact_contract_d1_fresh_le1s_"
    "current_ladder_v1"
)

DEFAULT_PREREQUISITE_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "cross_session_prerequisites_v1_20260819"
)
DEFAULT_ENTRY_ROOT = MAKER_ROOT / "data" / "walkforward" / "execution_narrow_60d"
DEFAULT_EXIT_ROOT = MAKER_ROOT / "data" / "walkforward" / "exit_maker_narrow_60d"
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "overnight_carry_narrow_60d_d1_fresh1s_current_ladder"
)

ENTRY_MANIFEST_NAME = "execution_partition_manifest.parquet"
EXIT_MANIFEST_NAME = "exit_maker_partition_manifest.parquet"
PREREQUISITE_MARKER_NAME = "complete.json"
SESSION_ARTIFACT_NAME = "candidate_sessions.txt"
CALENDAR_ARTIFACT_NAME = "exact_contract_calendar_v1.parquet"
UNIVERSE_ARTIFACT_NAME = "gap3_product_day_universe.parquet"
COMPLETION_MARKER_NAME = "gap3_complete.json"
REPORT_DIRECTORY_NAME = "report_60_sessions"

FRESH_BOOK_MAX_AGE_NS = 1_000_000_000
MAX_CARRY_SESSIONS = 1

_ENTRY_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "partition": pl.String,
    "config_sha256": pl.String,
    "complete": pl.Boolean,
    "order_aliases_rows": pl.Int64,
    "raw_order_facts_rows": pl.Int64,
    "hedge_facts_rows": pl.Int64,
    "execution_action_facts_rows": pl.Int64,
    "execution_daily_facts_rows": pl.Int64,
    "exit_facts_rows": pl.Int64,
    "exit_daily_facts_rows": pl.Int64,
    "policy_audit_rows": pl.Int64,
    "target_audit_rows": pl.Int64,
    "raw_tape_audit_rows": pl.Int64,
}

_EXIT_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "partition": pl.String,
    "runner_config_sha256": pl.String,
    "config_sha256": pl.String,
    "action_source_sha256": pl.String,
    "exit_rule_source_kind": pl.String,
    "exit_rule_source_sha256": pl.String,
    "complete": pl.Boolean,
    "exit_maker_audit_rows": pl.Int64,
    "exit_maker_candidate_aliases_rows": pl.Int64,
    "exit_maker_observations_rows": pl.Int64,
    "exit_maker_policy_support_rows": pl.Int64,
    "exit_maker_position_policy_facts_rows": pl.Int64,
    "exit_maker_raw_candidate_facts_rows": pl.Int64,
    "exit_maker_transitions_rows": pl.Int64,
}

_OUTPUT_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "partition": pl.String,
    "runner_config_sha256": pl.String,
    "config_sha256": pl.String,
    "overnight_carry_labels_rows": pl.Int64,
    "overnight_carry_audit_rows": pl.Int64,
    "complete": pl.Boolean,
}

_UNIVERSE_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "requested": pl.Boolean,
    "available_entry_partition": pl.Boolean,
    "available_exit_partition": pl.Boolean,
    "selected_for_replay": pl.Boolean,
    "availability_status": pl.String,
    "next_session": pl.String,
}

_MARKER_KEYS = {
    "schema_version",
    "orchestrator_version",
    "complete",
    "analysis_only",
    "gross_only",
    "pathwise_ev_ready",
    "joint_volume_allocated",
    "exact_contract_no_roll",
    "d1_only",
    "first_joint_taker_taker",
    "fresh_book_max_age_ns",
    "entry_dates_derived_from_manifests",
    "candidate_calendar_used_as_entry_selector",
    "source_binding",
    "runner_config_sha256",
    "implementation_sources",
    "artifacts",
}


@dataclass(frozen=True)
class CurrentLadderSourceContract:
    """Immutable source anchors and exact formal cohort cardinalities."""

    entry_manifest_sha256: str
    exit_manifest_sha256: str
    prerequisite_marker_sha256: str
    candidate_sessions_sha256: str
    contract_calendar_sha256: str
    entry_config_sha256: str
    exit_runner_config_sha256: str
    entry_partition_marker_inventory_sha256: str
    exit_partition_marker_inventory_sha256: str
    consumed_upstream_source_inventory_sha256: str
    expected_entry_dates: int
    expected_products: int
    expected_available_product_days: int
    expected_grid_product_days: int
    expected_missing_product_days: int
    expected_candidate_sessions: int

    def validate(self) -> None:
        for name in (
            "entry_manifest_sha256",
            "exit_manifest_sha256",
            "prerequisite_marker_sha256",
            "candidate_sessions_sha256",
            "contract_calendar_sha256",
            "entry_config_sha256",
            "exit_runner_config_sha256",
            "entry_partition_marker_inventory_sha256",
            "exit_partition_marker_inventory_sha256",
            "consumed_upstream_source_inventory_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "expected_entry_dates",
            "expected_products",
            "expected_available_product_days",
            "expected_grid_product_days",
            "expected_candidate_sessions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.expected_missing_product_days, bool)
            or not isinstance(self.expected_missing_product_days, int)
            or self.expected_missing_product_days < 0
        ):
            raise ValueError("expected_missing_product_days must be non-negative")
        if (
            self.expected_entry_dates * self.expected_products
            != self.expected_grid_product_days
            or self.expected_grid_product_days
            - self.expected_available_product_days
            != self.expected_missing_product_days
        ):
            raise ValueError("source contract grid cardinalities are incoherent")


FORMAL_SOURCE_CONTRACT = CurrentLadderSourceContract(
    entry_manifest_sha256=(
        "ba8436a6a085fe8b8716d67294e9202e9d7fe2ba4a874245dd13d1b9a64efe04"
    ),
    exit_manifest_sha256=(
        "b99d8b2f11354e150f02c990f178237e44078fc68dabc84f81317ce1f7aff6d8"
    ),
    prerequisite_marker_sha256=(
        "cd0c66bf24ab56eb657974adcefe28bec9ff0276fd31230a65a6f451efa661d3"
    ),
    candidate_sessions_sha256=(
        "1b46ca094485be18044aeafca51ceb39c7538a1d435af128331705a43f79b347"
    ),
    contract_calendar_sha256=(
        "dd93f3e24e72f190c02c52b83f541a45201a8c1b6f7b8761c7492ebbe33179d6"
    ),
    entry_config_sha256=(
        "b0a26ee2f6cb42decb8272742e1bd248cb6a550d64896f4a5fb9d3974874ab29"
    ),
    exit_runner_config_sha256=(
        "3947d8225d7799ab8efb9f55335ecb2c578241c1f5316cf077295ff7af737751"
    ),
    entry_partition_marker_inventory_sha256=(
        "76e659c891544a27fa763f434526e070e64694c3dd9feba9ad6e77c7bfbc7f60"
    ),
    exit_partition_marker_inventory_sha256=(
        "2115e95f071b615dcae3186d4e9be9631f24dd0348146ce1e8d5145b1ee6e516"
    ),
    consumed_upstream_source_inventory_sha256=(
        "0d8930b606299e03d3809efa1a2878f987df243ec87c1aaab519351f6918b974"
    ),
    expected_entry_dates=60,
    expected_products=45,
    expected_available_product_days=2_687,
    expected_grid_product_days=2_700,
    expected_missing_product_days=13,
    expected_candidate_sessions=135,
)


@dataclass(frozen=True)
class CurrentLadderOvernightPlan:
    """Fully verified, small orchestration inputs for the unchanged runner."""

    entry_root: Path
    exit_root: Path
    prerequisite_root: Path
    data_root: Path
    futures_raw_root: Path
    entry_dates: tuple[str, ...]
    products: tuple[str, ...]
    product_days: tuple[tuple[str, str], ...]
    candidate_sessions: tuple[str, ...]
    next_session_by_entry_date: Mapping[str, str]
    contract_calendar: pl.DataFrame
    universe: pl.DataFrame
    source_binding: Mapping[str, object]


def build_current_ladder_overnight_plan(
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    exit_root: Path = DEFAULT_EXIT_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    source_contract: CurrentLadderSourceContract = FORMAL_SOURCE_CONTRACT,
) -> CurrentLadderOvernightPlan:
    """Verify the immutable roots and derive the exact entry cohort."""

    source_contract.validate()
    entry_root = _canonical_root(entry_root, "entry execution root")
    exit_root = _canonical_root(exit_root, "same-day exit-maker root")
    prerequisite_root = _canonical_root(prerequisite_root, "prerequisite root")
    if len({entry_root, exit_root, prerequisite_root}) != 3:
        raise ValueError("entry, exit and prerequisite roots must be distinct")
    data_root = _canonical_root(data_root, "spot raw-data root")
    futures_raw_root = _canonical_root(futures_raw_root, "futures raw-data root")
    if source_contract == FORMAL_SOURCE_CONTRACT:
        formal_data_root = _canonical_root(HFT_DATA_ROOT, "formal spot raw-data root")
        formal_futures_root = _canonical_root(
            DEFAULT_FUTURES_RAW_ROOT,
            "formal futures raw-data root",
        )
        if data_root != formal_data_root:
            raise ValueError("formal data root override is forbidden")
        if futures_raw_root != formal_futures_root:
            raise ValueError("formal futures root override is forbidden")

    entry_manifest_path = entry_root / ENTRY_MANIFEST_NAME
    exit_manifest_path = exit_root / EXIT_MANIFEST_NAME
    prerequisite_marker_path = prerequisite_root / PREREQUISITE_MARKER_NAME
    sessions_path = prerequisite_root / SESSION_ARTIFACT_NAME
    calendar_path = prerequisite_root / CALENDAR_ARTIFACT_NAME
    for path, source in (
        (entry_manifest_path, "entry manifest"),
        (exit_manifest_path, "exit manifest"),
        (prerequisite_marker_path, "prerequisite marker"),
        (sessions_path, "candidate sessions"),
        (calendar_path, "exact contract calendar"),
    ):
        _require_regular_file(path, source)

    actual_hashes = {
        "entry_manifest_sha256": _file_sha256(entry_manifest_path),
        "exit_manifest_sha256": _file_sha256(exit_manifest_path),
        "prerequisite_marker_sha256": _file_sha256(prerequisite_marker_path),
        "candidate_sessions_sha256": _file_sha256(sessions_path),
        "contract_calendar_sha256": _file_sha256(calendar_path),
    }
    for name, actual in actual_hashes.items():
        if actual != getattr(source_contract, name):
            raise ValueError(f"formal source hash mismatch: {name}")

    entry_manifest = _read_exact_manifest(
        entry_manifest_path, _ENTRY_MANIFEST_SCHEMA, "entry manifest"
    )
    exit_manifest = _read_exact_manifest(
        exit_manifest_path, _EXIT_MANIFEST_SCHEMA, "exit manifest"
    )
    _validate_manifest_rows(entry_manifest, entry_root, "entry manifest")
    _validate_manifest_rows(exit_manifest, exit_root, "exit manifest")

    entry_configs = set(entry_manifest["config_sha256"].to_list())
    exit_runner_configs = set(exit_manifest["runner_config_sha256"].to_list())
    if entry_configs != {source_contract.entry_config_sha256}:
        raise ValueError("entry manifest does not bind the current price-ladder config")
    if exit_runner_configs != {source_contract.exit_runner_config_sha256}:
        raise ValueError("exit manifest does not bind the current price-ladder runner")

    entry_keys = _manifest_keys(entry_manifest)
    exit_keys = _manifest_keys(exit_manifest)
    if entry_keys != exit_keys:
        raise ValueError("entry and exit formal manifest keys differ")
    if len(entry_keys) != source_contract.expected_available_product_days:
        raise ValueError("formal available product-day count mismatch")
    _verify_exact_partition_tree(entry_root, entry_keys, source="entry execution")
    _verify_exact_partition_tree(exit_root, exit_keys, source="same-day exit-maker")
    upstream_inventory = _verify_upstream_partition_inventories(
        entry_manifest,
        exit_manifest,
        entry_root=entry_root,
        exit_root=exit_root,
        entry_config_sha256=source_contract.entry_config_sha256,
        exit_runner_config_sha256=source_contract.exit_runner_config_sha256,
        overnight_config=current_ladder_runner_config(),
    )
    for name in (
        "entry_partition_marker_inventory_sha256",
        "exit_partition_marker_inventory_sha256",
        "consumed_upstream_source_inventory_sha256",
    ):
        if upstream_inventory[name] != getattr(source_contract, name):
            raise ValueError(f"formal upstream partition inventory mismatch: {name}")

    entry_dates = tuple(sorted({date for date, _ in entry_keys}))
    products = tuple(sorted({value for _, value in entry_keys}))
    if len(entry_dates) != source_contract.expected_entry_dates:
        raise ValueError("formal entry-session count mismatch")
    if len(products) != source_contract.expected_products:
        raise ValueError("formal product count mismatch")

    candidate_sessions = _load_sessions(sessions_path)
    if len(candidate_sessions) != source_contract.expected_candidate_sessions:
        raise ValueError("candidate-session calendar count mismatch")
    session_index = {date: index for index, date in enumerate(candidate_sessions)}
    next_session: dict[str, str] = {}
    for date in entry_dates:
        index = session_index.get(date)
        if index is None:
            raise ValueError(f"entry date absent from prerequisite calendar: {date}")
        if index + 1 >= len(candidate_sessions):
            raise ValueError(f"entry date lacks a D+1 candidate session: {date}")
        next_session[date] = candidate_sessions[index + 1]

    key_set = set(entry_keys)
    universe_rows: list[dict[str, object]] = []
    for date in entry_dates:
        for value_code in products:
            available = (date, value_code) in key_set
            universe_rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "requested": True,
                    "available_entry_partition": available,
                    "available_exit_partition": available,
                    "selected_for_replay": available,
                    "availability_status": "available" if available else "unavailable",
                    "next_session": next_session[date],
                }
            )
    universe = pl.from_dicts(
        universe_rows, schema=_UNIVERSE_SCHEMA, infer_schema_length=None
    ).sort(["Date", "ValueCode"])
    if universe.height != source_contract.expected_grid_product_days:
        raise ValueError("formal date-by-product grid count mismatch")
    missing = universe.filter(~pl.col("selected_for_replay"))
    if missing.height != source_contract.expected_missing_product_days:
        raise ValueError("formal unavailable grid-cell count mismatch")
    if int(universe["selected_for_replay"].sum()) != len(entry_keys):
        raise ValueError("formal universe selected-key count mismatch")

    prerequisite = verify_cross_session_prerequisites(prerequisite_root)
    _validate_prerequisite_binding(
        prerequisite,
        prerequisite_root=prerequisite_root,
        entry_root=entry_root,
        entry_manifest_path=entry_manifest_path,
        entry_manifest_sha256=actual_hashes["entry_manifest_sha256"],
        sessions_path=sessions_path,
        sessions_sha256=actual_hashes["candidate_sessions_sha256"],
        calendar_path=calendar_path,
        calendar_sha256=actual_hashes["contract_calendar_sha256"],
    )
    calendar = pl.read_parquet(calendar_path)
    expected_calendar_schema = {
        "QuoteCode": pl.String,
        "expiry_session": pl.String,
        "calendar_version": pl.String,
    }
    if list(calendar.schema.items()) != list(expected_calendar_schema.items()):
        raise ValueError("exact contract calendar schema mismatch")
    if calendar.is_empty() or calendar.select("QuoteCode").n_unique() != calendar.height:
        raise ValueError("exact contract calendar QuoteCode identity is not unique")

    binding: dict[str, object] = {
        "entry_root": str(entry_root),
        "exit_root": str(exit_root),
        "prerequisite_root": str(prerequisite_root),
        "data_root": str(data_root),
        "futures_raw_root": str(futures_raw_root),
        **actual_hashes,
        "prerequisite_config_sha256": prerequisite.get("config_sha256"),
        "entry_config_sha256": source_contract.entry_config_sha256,
        "exit_runner_config_sha256": source_contract.exit_runner_config_sha256,
        "entry_session_count": len(entry_dates),
        "product_count": len(products),
        "available_product_day_count": len(entry_keys),
        "grid_product_day_count": universe.height,
        "missing_product_day_count": missing.height,
        "candidate_session_count": len(candidate_sessions),
        **upstream_inventory,
        "entry_dates_sha256": _canonical_sha256(list(entry_dates)),
        "products_sha256": _canonical_sha256(list(products)),
        "available_keys_sha256": _canonical_sha256(
            [[date, value] for date, value in entry_keys]
        ),
        "missing_keys_sha256": _canonical_sha256(
            missing.select("Date", "ValueCode").to_dicts()
        ),
        "d1_mapping_sha256": _canonical_sha256(next_session),
    }
    binding["source_binding_sha256"] = _canonical_sha256(binding)
    return CurrentLadderOvernightPlan(
        entry_root=entry_root,
        exit_root=exit_root,
        prerequisite_root=prerequisite_root,
        data_root=data_root,
        futures_raw_root=futures_raw_root,
        entry_dates=entry_dates,
        products=products,
        product_days=entry_keys,
        candidate_sessions=candidate_sessions,
        next_session_by_entry_date=next_session,
        contract_calendar=calendar,
        universe=universe,
        source_binding=binding,
    )


def current_ladder_runner_config() -> OvernightCarryRunnerConfig:
    """Return the non-overridable D+1/fresh<=1s legacy runner config."""

    return OvernightCarryRunnerConfig(
        carry=OvernightCarryConfig(
            max_carry_sessions=MAX_CARRY_SESSIONS,
            max_book_age_ns=FRESH_BOOK_MAX_AGE_NS,
            policy_version=POLICY_VERSION,
        )
    )


def run_current_ladder_overnight(
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    exit_root: Path = DEFAULT_EXIT_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    resume: bool = True,
    source_contract: CurrentLadderSourceContract = FORMAL_SOURCE_CONTRACT,
) -> dict[str, object]:
    """Run the unchanged carry producer/report and publish a completion marker."""

    destination = Path(output_root)
    if destination.is_symlink():
        raise ValueError("overnight output root must not be a symlink")
    canonical_inputs = (
        _canonical_root(entry_root, "entry execution root"),
        _canonical_root(exit_root, "same-day exit-maker root"),
        _canonical_root(prerequisite_root, "prerequisite root"),
        _canonical_root(data_root, "spot raw-data root"),
        _canonical_root(futures_raw_root, "futures raw-data root"),
    )
    _validate_output_disjoint(
        destination.resolve(strict=False),
        canonical_inputs,
    )
    entry_root, exit_root, prerequisite_root, data_root, futures_raw_root = (
        canonical_inputs
    )
    marker_path = destination / COMPLETION_MARKER_NAME
    if marker_path.is_file():
        if not resume:
            raise FileExistsError(
                f"completed output already exists and resume is disabled: {marker_path}"
            )
        return verify_current_ladder_overnight_bundle(
            destination,
            entry_root=entry_root,
            exit_root=exit_root,
            prerequisite_root=prerequisite_root,
            data_root=data_root,
            futures_raw_root=futures_raw_root,
            source_contract=source_contract,
        )
    plan = build_current_ladder_overnight_plan(
        entry_root=entry_root,
        exit_root=exit_root,
        prerequisite_root=prerequisite_root,
        data_root=data_root,
        futures_raw_root=futures_raw_root,
        source_contract=source_contract,
    )
    config = current_ladder_runner_config()
    expected_runner_payload = _expected_runner_payload(plan, config)
    expected_runner_sha = _canonical_sha256(expected_runner_payload)
    _verify_partial_output_tree(destination, plan.product_days)
    manifest = run_overnight_carry_replay(
        plan.product_days,
        sessions=plan.candidate_sessions,
        contract_calendar=plan.contract_calendar,
        settlement_facts=None,
        exit_maker_root=plan.exit_root,
        entry_execution_root=plan.entry_root,
        output_root=destination,
        config=config,
        data_root=plan.data_root,
        futures_raw_root=plan.futures_raw_root,
        resume=resume,
    )
    _validate_output_manifest(
        manifest,
        destination,
        plan.product_days,
        expected_runner_sha,
    )
    _verify_exact_partition_tree(
        destination,
        plan.product_days,
        source="overnight output",
    )
    _atomic_write_parquet(plan.universe, destination / UNIVERSE_ARTIFACT_NAME)
    report_dir = destination / REPORT_DIRECTORY_NAME
    run_overnight_carry_report(
        destination,
        output_dir=report_dir,
        sessions=len(plan.entry_dates),
        value_codes=plan.products,
        require_exact_sessions=True,
        require_balanced_product_days=False,
        validate_hashes=True,
    )

    artifacts = {
        OVERNIGHT_CARRY_MANIFEST_NAME: _parquet_metadata(
            destination / OVERNIGHT_CARRY_MANIFEST_NAME
        ),
        UNIVERSE_ARTIFACT_NAME: _parquet_metadata(
            destination / UNIVERSE_ARTIFACT_NAME
        ),
        f"{REPORT_DIRECTORY_NAME}/report_complete.json": _file_metadata(
            report_dir / "report_complete.json"
        ),
    }
    marker: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "complete": True,
        "analysis_only": True,
        "gross_only": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "exact_contract_no_roll": True,
        "d1_only": True,
        "first_joint_taker_taker": True,
        "fresh_book_max_age_ns": FRESH_BOOK_MAX_AGE_NS,
        "entry_dates_derived_from_manifests": True,
        "candidate_calendar_used_as_entry_selector": False,
        "source_binding": dict(plan.source_binding),
        "runner_config_sha256": expected_runner_sha,
        "implementation_sources": _implementation_sources(),
        "artifacts": artifacts,
    }
    _validate_marker_payload(marker, plan, expected_runner_sha)
    _verify_exact_output_tree(
        destination,
        plan.product_days,
        require_completion_marker=False,
    )
    _atomic_write_json(marker, marker_path)
    return verify_current_ladder_overnight_bundle(
        destination,
        entry_root=plan.entry_root,
        exit_root=plan.exit_root,
        prerequisite_root=plan.prerequisite_root,
        data_root=plan.data_root,
        futures_raw_root=plan.futures_raw_root,
        source_contract=source_contract,
    )


def verify_current_ladder_overnight_bundle(
    output_root: Path,
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    exit_root: Path = DEFAULT_EXIT_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    source_contract: CurrentLadderSourceContract = FORMAL_SOURCE_CONTRACT,
    deep_report_rebuild: bool = True,
) -> dict[str, object]:
    """Public verifier for source roots, partitions, report and exact inventory."""

    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("overnight output root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"overnight output root does not exist: {root}") from exc
    if not root.is_dir():
        raise NotADirectoryError(f"overnight output root is not a directory: {root}")
    canonical_inputs = (
        _canonical_root(entry_root, "entry execution root"),
        _canonical_root(exit_root, "same-day exit-maker root"),
        _canonical_root(prerequisite_root, "prerequisite root"),
        _canonical_root(data_root, "spot raw-data root"),
        _canonical_root(futures_raw_root, "futures raw-data root"),
    )
    _validate_output_disjoint(root, canonical_inputs)
    entry_root, exit_root, prerequisite_root, data_root, futures_raw_root = (
        canonical_inputs
    )
    marker_path = root / COMPLETION_MARKER_NAME
    _require_regular_file(marker_path, "gap3 completion marker")
    marker = _read_json_object(marker_path)
    plan = build_current_ladder_overnight_plan(
        entry_root=entry_root,
        exit_root=exit_root,
        prerequisite_root=prerequisite_root,
        data_root=data_root,
        futures_raw_root=futures_raw_root,
        source_contract=source_contract,
    )
    config = current_ladder_runner_config()
    expected_runner_payload = _expected_runner_payload(plan, config)
    expected_runner_sha = _canonical_sha256(expected_runner_payload)
    _validate_marker_payload(marker, plan, expected_runner_sha)
    if marker.get("implementation_sources") != _implementation_sources():
        raise ValueError("gap3 implementation source hashes changed")

    artifacts = marker.get("artifacts")
    assert isinstance(artifacts, dict)
    expected_artifacts = {
        OVERNIGHT_CARRY_MANIFEST_NAME,
        UNIVERSE_ARTIFACT_NAME,
        f"{REPORT_DIRECTORY_NAME}/report_complete.json",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("gap3 artifact inventory mismatch")
    for relative, declaration in artifacts.items():
        if not isinstance(declaration, dict):
            raise ValueError(f"invalid gap3 artifact declaration: {relative}")
        _verify_file_metadata(root / relative, declaration)

    universe = pl.read_parquet(root / UNIVERSE_ARTIFACT_NAME)
    if list(universe.schema.items()) != list(_UNIVERSE_SCHEMA.items()):
        raise ValueError("gap3 universe schema mismatch")
    if not universe.equals(plan.universe, null_equal=True):
        raise ValueError("gap3 universe differs from immutable manifest derivation")

    manifest = pl.read_parquet(root / OVERNIGHT_CARRY_MANIFEST_NAME)
    _verify_exact_output_tree(
        root,
        plan.product_days,
        require_completion_marker=True,
    )
    _validate_output_manifest(
        manifest, root, plan.product_days, expected_runner_sha
    )
    manifest_rows = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in manifest.iter_rows(named=True)
    }
    raw_fingerprints_by_entry_date = {
        date: _candidate_raw_fingerprints(
            (plan.next_session_by_entry_date[date],),
            data_root=plan.data_root,
            futures_raw_root=plan.futures_raw_root,
            custom_loader=False,
        )
        for date in plan.entry_dates
    }
    for date, value_code in plan.product_days:
        partition = root / f"Date={date}" / f"ValueCode={value_code}"
        verified = verify_overnight_output_partition(
            partition,
            expected_runner_config_sha256=expected_runner_sha,
        )
        row = manifest_rows[(date, value_code)]
        _verify_output_manifest_row_parity(
            verified,
            row,
            source=f"{date}/{value_code}",
        )
        payload = _read_json_object(partition / "complete.json")
        partition_config = payload.get("config")
        if not isinstance(partition_config, dict):
            raise ValueError(f"partition config is invalid: {partition}")
        if partition_config.get("runner") != expected_runner_payload:
            raise ValueError(f"partition runner payload is not current gap3: {partition}")
        if partition_config.get("candidate_sessions") != [
            plan.next_session_by_entry_date[date]
        ]:
            raise ValueError(f"partition does not bind exact D+1 session: {partition}")
        if partition_config.get("raw_source_fingerprints") != (
            raw_fingerprints_by_entry_date[date]
        ):
            raise ValueError(f"partition raw source fingerprint drift: {partition}")
        current_source = verify_overnight_sources(
            date,
            value_code,
            exit_maker_root=plan.exit_root,
            entry_execution_root=plan.entry_root,
            config=config,
        )
        if partition_config.get("source") != current_source.payload(config):
            raise ValueError(f"partition upstream source hashes changed: {partition}")

    report_dir = root / REPORT_DIRECTORY_NAME
    _verify_report_marker(report_dir, plan, expected_runner_sha)
    if deep_report_rebuild:
        run_overnight_carry_report(
            root,
            output_dir=report_dir,
            sessions=len(plan.entry_dates),
            value_codes=plan.products,
            require_exact_sessions=True,
            require_balanced_product_days=False,
            validate_hashes=True,
        )
    return marker


def _validate_prerequisite_binding(
    payload: Mapping[str, object],
    *,
    prerequisite_root: Path,
    entry_root: Path,
    entry_manifest_path: Path,
    entry_manifest_sha256: str,
    sessions_path: Path,
    sessions_sha256: str,
    calendar_path: Path,
    calendar_sha256: str,
) -> None:
    config = payload.get("config")
    artifacts = payload.get("artifacts")
    if not isinstance(config, dict) or not isinstance(artifacts, dict):
        raise ValueError("prerequisite marker lacks config/artifact binding")
    declared_entry_root = config.get("entry_execution_root")
    declared_product_days = config.get("product_days_path")
    if not isinstance(declared_entry_root, str) or (
        Path(declared_entry_root).resolve(strict=True) != entry_root
    ):
        raise ValueError("prerequisite entry root differs from supplied formal root")
    if not isinstance(declared_product_days, str) or (
        Path(declared_product_days).resolve(strict=True) != entry_manifest_path
    ):
        raise ValueError("prerequisite entry manifest path binding mismatch")
    if config.get("product_days_sha256") != entry_manifest_sha256:
        raise ValueError("prerequisite entry manifest hash binding mismatch")
    if config.get("contract_calendar_sha256") != calendar_sha256:
        raise ValueError("prerequisite calendar config hash binding mismatch")
    for name, path, digest in (
        (SESSION_ARTIFACT_NAME, sessions_path, sessions_sha256),
        (CALENDAR_ARTIFACT_NAME, calendar_path, calendar_sha256),
    ):
        declaration = artifacts.get(name)
        if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
            raise ValueError(f"prerequisite artifact hash binding mismatch: {name}")
        if int(declaration.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"prerequisite artifact byte binding mismatch: {name}")
    if prerequisite_root != (prerequisite_root / ".").resolve():
        raise AssertionError("prerequisite root canonicalization changed")


def _expected_runner_payload(
    plan: CurrentLadderOvernightPlan,
    config: OvernightCarryRunnerConfig,
) -> dict[str, object]:
    global_inputs = {
        "session_calendar": {
            "count": len(plan.candidate_sessions),
            "first": plan.candidate_sessions[0],
            "last": plan.candidate_sessions[-1],
            "sha256": _canonical_sha256(list(plan.candidate_sessions)),
        },
        "contract_calendar_content_sha256": _runner_frame_sha256(
            plan.contract_calendar
        ),
        "settlement_facts_content_sha256": None,
    }
    return _build_runner_payload(config, global_inputs)


def _validate_marker_payload(
    marker: Mapping[str, object],
    plan: CurrentLadderOvernightPlan,
    expected_runner_sha: str,
) -> None:
    if set(marker) != _MARKER_KEYS:
        raise ValueError("gap3 completion marker keys mismatch")
    expected_scalars = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "complete": True,
        "analysis_only": True,
        "gross_only": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "exact_contract_no_roll": True,
        "d1_only": True,
        "first_joint_taker_taker": True,
        "fresh_book_max_age_ns": FRESH_BOOK_MAX_AGE_NS,
        "entry_dates_derived_from_manifests": True,
        "candidate_calendar_used_as_entry_selector": False,
        "runner_config_sha256": expected_runner_sha,
    }
    for name, expected in expected_scalars.items():
        if marker.get(name) != expected:
            raise ValueError(f"gap3 marker semantic mismatch: {name}")
    if marker.get("source_binding") != dict(plan.source_binding):
        raise ValueError("gap3 marker source binding mismatch")
    if not isinstance(marker.get("artifacts"), dict):
        raise ValueError("gap3 marker artifact declaration is invalid")


def _validate_output_manifest(
    manifest: pl.DataFrame,
    output_root: Path,
    expected_keys: Sequence[tuple[str, str]],
    expected_runner_sha: str,
) -> None:
    if list(manifest.schema.items()) != list(_OUTPUT_MANIFEST_SCHEMA.items()):
        raise ValueError("overnight root manifest schema mismatch")
    if not manifest.equals(manifest.sort(["Date", "ValueCode"]), null_equal=True):
        raise ValueError("overnight root manifest is not canonically sorted")
    actual_keys = _manifest_keys(manifest)
    if actual_keys != tuple(expected_keys):
        raise ValueError("overnight root manifest key inventory mismatch")
    if manifest.filter(
        (pl.col("complete") != True)  # noqa: E712
        | (pl.col("runner_config_sha256") != expected_runner_sha)
    ).height:
        raise ValueError("overnight root manifest completion/runner mismatch")
    for row in manifest.iter_rows(named=True):
        expected_partition = (
            Path(output_root)
            / f"Date={row['Date']}"
            / f"ValueCode={row['ValueCode']}"
        ).resolve()
        if _resolve_declared_path(str(row["partition"])) != expected_partition:
            raise ValueError("overnight root manifest partition path mismatch")


def _verify_output_manifest_row_parity(
    rebuilt: Mapping[str, object],
    declared: Mapping[str, object],
    *,
    source: str,
) -> None:
    if set(rebuilt) != set(_OUTPUT_MANIFEST_SCHEMA):
        raise ValueError(f"overnight partition rebuilt manifest schema mismatch: {source}")
    for column in _OUTPUT_MANIFEST_SCHEMA:
        if column == "partition":
            equal = _resolve_declared_path(str(rebuilt[column])) == (
                _resolve_declared_path(str(declared[column]))
            )
        else:
            equal = rebuilt[column] == declared[column]
        if not equal:
            raise ValueError(
                f"overnight root manifest differs from partition: {source}/{column}"
            )


def _verify_report_marker(
    report_dir: Path,
    plan: CurrentLadderOvernightPlan,
    expected_runner_sha: str,
) -> None:
    marker_path = Path(report_dir) / "report_complete.json"
    _require_regular_file(marker_path, "overnight report marker")
    marker = _read_json_object(marker_path)
    declared_payload_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if not isinstance(declared_payload_sha, str) or (
        _canonical_sha256(unhashed) != declared_payload_sha
    ):
        raise ValueError("overnight report marker payload hash mismatch")
    if (
        marker.get("complete") is not True
        or marker.get("report_version") != _carry_report.REPORT_VERSION
        or int(marker.get("selected_session_count", -1)) != len(plan.entry_dates)
        or int(marker.get("selected_product_count", -1)) != len(plan.products)
        or int(marker.get("selected_product_day_count", -1)) != len(plan.product_days)
        or int(marker.get("expected_product_day_count", -1)) != plan.universe.height
        or int(marker.get("missing_product_day_count", -1))
        != plan.universe.height - len(plan.product_days)
        or marker.get("runner_config_sha256") != expected_runner_sha
        or marker.get("gross_only") is not True
        or marker.get("pathwise_ev_ready") is not False
    ):
        raise ValueError("overnight report marker scope/semantics mismatch")
    if tuple(marker.get("selected_dates", ())) != plan.entry_dates:
        raise ValueError("overnight report selected dates differ from manifest cohort")
    if tuple(marker.get("value_codes", ())) != plan.products:
        raise ValueError("overnight report products differ from manifest cohort")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        _carry_report.REPORT_ARTIFACTS
    ):
        raise ValueError("overnight report artifact inventory mismatch")
    for name, declaration in artifacts.items():
        if not isinstance(declaration, dict):
            raise ValueError(f"invalid overnight report artifact: {name}")
        _carry_report._validate_one_artifact(  # noqa: SLF001
            Path(report_dir) / name,
            declaration,
            validate_hashes=True,
        )


def _read_exact_manifest(
    path: Path,
    schema: Mapping[str, pl.DataType],
    source: str,
) -> pl.DataFrame:
    actual_schema = pl.read_parquet_schema(path)
    if list(actual_schema.items()) != list(schema.items()):
        raise ValueError(f"{source} ordered schema mismatch")
    frame = pl.read_parquet(path)
    if frame.is_empty():
        raise ValueError(f"{source} must be nonempty")
    if not frame.equals(frame.sort(["Date", "ValueCode"]), null_equal=True):
        raise ValueError(f"{source} is not canonically sorted")
    if frame.select("Date", "ValueCode").n_unique() != frame.height:
        raise ValueError(f"{source} contains duplicate product-day keys")
    return frame


def _verify_upstream_partition_inventories(
    entry_manifest: pl.DataFrame,
    exit_manifest: pl.DataFrame,
    *,
    entry_root: Path,
    exit_root: Path,
    entry_config_sha256: str,
    exit_runner_config_sha256: str,
    overnight_config: OvernightCarryRunnerConfig,
) -> dict[str, object]:
    """Rebuild both frozen manifests and bind marker/artifact inventories."""

    entry_rows = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in entry_manifest.iter_rows(named=True)
    }
    exit_rows = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in exit_manifest.iter_rows(named=True)
    }
    entry_inventory: list[dict[str, object]] = []
    exit_inventory: list[dict[str, object]] = []
    consumed_source_inventory: list[dict[str, object]] = []
    for date, value_code in _manifest_keys(entry_manifest):
        entry_partition = (
            entry_root / f"Date={date}" / f"ValueCode={value_code}"
        )
        exit_partition = exit_root / f"Date={date}" / f"ValueCode={value_code}"

        rebuilt_entry = _verify_execution_output_partition(
            entry_partition,
            entry_config_sha256,
        )
        _verify_manifest_row_parity(
            rebuilt_entry,
            entry_rows[(date, value_code)],
            _ENTRY_MANIFEST_SCHEMA,
            source=f"entry partition {date}/{value_code}",
        )
        rebuilt_exit = _verify_exit_maker_output_partition(
            exit_partition,
            expected_config_sha256=str(
                exit_rows[(date, value_code)]["config_sha256"]
            ),
            expected_runner_config_sha256=exit_runner_config_sha256,
        )
        _verify_manifest_row_parity(
            rebuilt_exit,
            exit_rows[(date, value_code)],
            _EXIT_MANIFEST_SCHEMA,
            source=f"exit partition {date}/{value_code}",
        )
        consumed_source = verify_overnight_sources(
            date,
            value_code,
            exit_maker_root=exit_root,
            entry_execution_root=entry_root,
            config=overnight_config,
        )
        consumed_source_inventory.append(consumed_source.payload(overnight_config))
        entry_inventory.append(
            _partition_marker_inventory(entry_partition, date, value_code)
        )
        exit_inventory.append(
            _partition_marker_inventory(exit_partition, date, value_code)
        )

    entry_digest = _canonical_sha256(entry_inventory)
    exit_digest = _canonical_sha256(exit_inventory)
    return {
        "entry_partition_verified_count": len(entry_inventory),
        "exit_partition_verified_count": len(exit_inventory),
        "entry_partition_marker_inventory_sha256": entry_digest,
        "exit_partition_marker_inventory_sha256": exit_digest,
        "consumed_upstream_source_inventory_sha256": _canonical_sha256(
            consumed_source_inventory
        ),
        "upstream_partition_inventory_sha256": _canonical_sha256(
            {
                "entry": entry_digest,
                "exit": exit_digest,
            }
        ),
    }


def _verify_manifest_row_parity(
    rebuilt: Mapping[str, object],
    declared: Mapping[str, object],
    schema: Mapping[str, pl.DataType],
    *,
    source: str,
) -> None:
    if set(rebuilt) != set(schema):
        raise ValueError(f"{source} rebuilt manifest schema mismatch")
    for column in schema:
        if column == "partition":
            equal = _resolve_declared_path(str(rebuilt[column])) == (
                _resolve_declared_path(str(declared[column]))
            )
        else:
            equal = rebuilt[column] == declared[column]
        if not equal:
            raise ValueError(f"{source} differs from root manifest: {column}")


def _partition_marker_inventory(
    partition: Path,
    date: str,
    value_code: str,
) -> dict[str, object]:
    marker = partition / "complete.json"
    _require_regular_file(marker, "verified upstream completion marker")
    payload = _read_json_object(marker)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"verified upstream marker lacks artifacts: {marker}")
    for filename, metadata in artifacts.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise ValueError(f"invalid upstream artifact declaration: {marker}")
        _validate_sha256(metadata.get("sha256"), f"{marker}:{filename}")
    return {
        "Date": date,
        "ValueCode": value_code,
        "complete_marker_sha256": _file_sha256(marker),
        "artifact_declarations_sha256": _canonical_sha256(artifacts),
    }


def _verify_exact_partition_tree(
    root: Path,
    expected_keys: Sequence[tuple[str, str]],
    *,
    source: str,
) -> None:
    """Reject missing, extra, incomplete, or symlinked Date/ValueCode partitions."""

    expected_by_date: dict[str, set[str]] = {}
    for date, value_code in expected_keys:
        expected_by_date.setdefault(date, set()).add(value_code)

    actual_by_date: dict[str, set[str]] = {}
    for date_entry in sorted(root.glob("Date=*")):
        if date_entry.is_symlink() or not date_entry.is_dir():
            raise ValueError(f"{source} date partition is not a regular directory")
        date = date_entry.name.removeprefix("Date=")
        _validate_partition_key(date, "inventory")
        products: set[str] = set()
        for product_entry in sorted(date_entry.glob("ValueCode=*")):
            if product_entry.is_symlink() or not product_entry.is_dir():
                raise ValueError(
                    f"{source} product partition is not a regular directory"
                )
            value_code = product_entry.name.removeprefix("ValueCode=")
            _validate_partition_key(date, value_code)
            products.add(value_code)
        actual_by_date[date] = products

    if actual_by_date != expected_by_date:
        expected = {
            (date, value_code)
            for date, products in expected_by_date.items()
            for value_code in products
        }
        actual = {
            (date, value_code)
            for date, products in actual_by_date.items()
            for value_code in products
        }
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"{source} partition inventory mismatch; missing={missing}, extra={extra}"
        )


def _validate_output_disjoint(
    destination: Path,
    input_roots: Sequence[Path],
) -> None:
    for source in input_roots:
        if (
            destination == source
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)
        ):
            raise ValueError(
                "output root must be fully disjoint from every input root: "
                f"output={destination}, input={source}"
            )


def _verify_partial_output_tree(
    root: Path,
    expected_keys: Sequence[tuple[str, str]],
) -> None:
    """Reject foreign entries in a resumable root before the runner writes."""

    root = Path(root)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError("partial overnight output root must be a regular directory")
    expected_by_date: dict[str, set[str]] = {}
    for date, value_code in expected_keys:
        expected_by_date.setdefault(date, set()).add(value_code)
    allowed_root_names = {
        OVERNIGHT_CARRY_MANIFEST_NAME,
        UNIVERSE_ARTIFACT_NAME,
        REPORT_DIRECTORY_NAME,
        *(f"Date={date}" for date in expected_by_date),
    }
    actual_root_names = {path.name for path in root.iterdir()}
    foreign_root = actual_root_names - allowed_root_names
    if foreign_root:
        raise ValueError(
            f"partial overnight output has foreign root entries: "
            f"{sorted(foreign_root)[:5]}"
        )
    for filename in (OVERNIGHT_CARRY_MANIFEST_NAME, UNIVERSE_ARTIFACT_NAME):
        path = root / filename
        if path.exists() or path.is_symlink():
            _require_regular_file(path, "partial overnight root artifact")

    allowed_partition_files = {"complete.json", *_OVERNIGHT_OUTPUT_ARTIFACTS}
    for date, products in expected_by_date.items():
        date_dir = root / f"Date={date}"
        if not date_dir.exists() and not date_dir.is_symlink():
            continue
        if date_dir.is_symlink() or not date_dir.is_dir():
            raise ValueError(f"invalid partial overnight date directory: {date_dir}")
        allowed_products = {f"ValueCode={value}" for value in products}
        actual_products = {path.name for path in date_dir.iterdir()}
        foreign_products = actual_products - allowed_products
        if foreign_products:
            raise ValueError(
                f"partial overnight date has foreign entries: {date}; "
                f"{sorted(foreign_products)[:5]}"
            )
        for product_name in actual_products:
            partition = date_dir / product_name
            if partition.is_symlink() or not partition.is_dir():
                raise ValueError(f"invalid partial overnight partition: {partition}")
            actual_files = {path.name for path in partition.iterdir()}
            foreign_files = actual_files - allowed_partition_files
            if foreign_files:
                raise ValueError(
                    f"partial overnight partition has foreign entries: {partition}; "
                    f"{sorted(foreign_files)[:5]}"
                )
            for filename in actual_files:
                _require_regular_file(
                    partition / filename,
                    "partial overnight partition artifact",
                )

    report_dir = root / REPORT_DIRECTORY_NAME
    if report_dir.exists() or report_dir.is_symlink():
        if report_dir.is_symlink() or not report_dir.is_dir():
            raise ValueError("partial overnight report directory is invalid")
        allowed_report_files = {
            "report_complete.json",
            *_carry_report.REPORT_ARTIFACTS,
        }
        actual_report_files = {path.name for path in report_dir.iterdir()}
        foreign_report = actual_report_files - allowed_report_files
        if foreign_report:
            raise ValueError(
                "partial overnight report has foreign entries: "
                f"{sorted(foreign_report)[:5]}"
            )
        for filename in actual_report_files:
            _require_regular_file(
                report_dir / filename,
                "partial overnight report artifact",
            )


def _verify_exact_output_tree(
    root: Path,
    expected_keys: Sequence[tuple[str, str]],
    *,
    require_completion_marker: bool,
) -> None:
    """Verify every allowed output entry, including non-partition artifacts."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("overnight output root must be a regular directory")
    expected_by_date: dict[str, set[str]] = {}
    for date, value_code in expected_keys:
        expected_by_date.setdefault(date, set()).add(value_code)
    expected_root_names = {
        OVERNIGHT_CARRY_MANIFEST_NAME,
        UNIVERSE_ARTIFACT_NAME,
        REPORT_DIRECTORY_NAME,
        *(f"Date={date}" for date in expected_by_date),
    }
    if require_completion_marker:
        expected_root_names.add(COMPLETION_MARKER_NAME)
    actual_root_names = {path.name for path in root.iterdir()}
    if actual_root_names != expected_root_names:
        raise ValueError(
            "overnight output root inventory mismatch; "
            f"missing={sorted(expected_root_names - actual_root_names)[:5]}, "
            f"extra={sorted(actual_root_names - expected_root_names)[:5]}"
        )

    for filename in (OVERNIGHT_CARRY_MANIFEST_NAME, UNIVERSE_ARTIFACT_NAME):
        _require_regular_file(root / filename, "overnight root artifact")
    if require_completion_marker:
        _require_regular_file(
            root / COMPLETION_MARKER_NAME,
            "gap3 completion marker",
        )

    expected_partition_files = {
        "complete.json",
        *_OVERNIGHT_OUTPUT_ARTIFACTS,
    }
    for date, products in expected_by_date.items():
        date_dir = root / f"Date={date}"
        if date_dir.is_symlink() or not date_dir.is_dir():
            raise ValueError(f"overnight output invalid date directory: {date_dir}")
        expected_product_names = {f"ValueCode={value}" for value in products}
        actual_product_names = {path.name for path in date_dir.iterdir()}
        if actual_product_names != expected_product_names:
            raise ValueError(
                f"overnight output date inventory mismatch: {date}; "
                f"missing={sorted(expected_product_names - actual_product_names)[:5]}, "
                f"extra={sorted(actual_product_names - expected_product_names)[:5]}"
            )
        for value_code in products:
            partition = date_dir / f"ValueCode={value_code}"
            if partition.is_symlink() or not partition.is_dir():
                raise ValueError(f"invalid overnight output partition: {partition}")
            actual_files = {path.name for path in partition.iterdir()}
            if actual_files != expected_partition_files:
                raise ValueError(
                    f"overnight output partition file inventory mismatch: {partition}; "
                    f"missing={sorted(expected_partition_files - actual_files)}, "
                    f"extra={sorted(actual_files - expected_partition_files)}"
                )
            for filename in expected_partition_files:
                _require_regular_file(
                    partition / filename,
                    "overnight partition artifact",
                )

    report_dir = root / REPORT_DIRECTORY_NAME
    if report_dir.is_symlink() or not report_dir.is_dir():
        raise ValueError("overnight report directory is invalid")
    expected_report_files = {
        "report_complete.json",
        *_carry_report.REPORT_ARTIFACTS,
    }
    actual_report_files = {path.name for path in report_dir.iterdir()}
    if actual_report_files != expected_report_files:
        raise ValueError(
            "overnight report file inventory mismatch; "
            f"missing={sorted(expected_report_files - actual_report_files)}, "
            f"extra={sorted(actual_report_files - expected_report_files)}"
        )
    for filename in expected_report_files:
        _require_regular_file(report_dir / filename, "overnight report artifact")


def _validate_manifest_rows(frame: pl.DataFrame, root: Path, source: str) -> None:
    if frame.filter(pl.col("complete") != True).height:  # noqa: E712
        raise ValueError(f"{source} contains incomplete rows")
    for row in frame.iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        _validate_partition_key(date, value_code)
        expected = (root / f"Date={date}" / f"ValueCode={value_code}").resolve()
        if _resolve_declared_path(str(row["partition"])) != expected:
            raise ValueError(f"{source} partition path mismatch: {date}/{value_code}")


def _manifest_keys(frame: pl.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(date), str(value))
        for date, value in frame.select("Date", "ValueCode").iter_rows()
    )


def _load_sessions(path: Path) -> tuple[str, ...]:
    sessions = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        not sessions
        or sessions != tuple(sorted(sessions))
        or len(sessions) != len(set(sessions))
        or any(len(value) != 8 or not value.isdigit() for value in sessions)
    ):
        raise ValueError("candidate sessions must be unique ascending YYYYMMDD")
    return sessions


def _canonical_root(path: Path, source: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{source} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(raw) from exc
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _require_regular_file(path: Path, source: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{source} is not a regular non-symlink file: {path}")


def _resolve_declared_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def _validate_partition_key(date: str, value_code: str) -> None:
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"invalid YYYYMMDD partition date: {date}")
    if not value_code or any(character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_.-" for character in value_code):
        raise ValueError(f"unsafe ValueCode partition key: {value_code}")


def _implementation_sources() -> dict[str, str]:
    module_root = Path(__file__).parent
    paths = (
        Path(__file__),
        module_root / "cross_session_prerequisite.py",
        module_root / "current_ladder_overnight_cli.py",
        module_root / "execution_runner.py",
        module_root / "exit_maker_runner.py",
        module_root / "overnight_carry_runner.py",
        module_root / "overnight_carry.py",
        module_root / "overnight_carry_report.py",
    )
    return {str(path): _file_sha256(path) for path in paths}


def _parquet_metadata(path: Path) -> dict[str, object]:
    _require_regular_file(path, "published parquet artifact")
    frame = pl.read_parquet(path)
    return {
        "kind": "parquet",
        "rows": frame.height,
        "columns": frame.width,
        "schema": [[name, str(dtype)] for name, dtype in frame.schema.items()],
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_metadata(path: Path) -> dict[str, object]:
    _require_regular_file(path, "published file artifact")
    return {
        "kind": "json",
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _verify_file_metadata(path: Path, declaration: Mapping[str, object]) -> None:
    _require_regular_file(path, "declared gap3 artifact")
    if declaration.get("kind") == "parquet":
        expected_keys = {"kind", "rows", "columns", "schema", "bytes", "sha256"}
        if set(declaration) != expected_keys:
            raise ValueError(f"parquet metadata keys mismatch: {path}")
        schema = pl.read_parquet_schema(path)
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        if (
            int(declaration.get("rows", -1)) != rows
            or int(declaration.get("columns", -1)) != len(schema)
            or declaration.get("schema")
            != [[name, str(dtype)] for name, dtype in schema.items()]
        ):
            raise ValueError(f"parquet rows/schema metadata mismatch: {path}")
    elif declaration.get("kind") == "json":
        if set(declaration) != {"kind", "bytes", "sha256"}:
            raise ValueError(f"JSON metadata keys mismatch: {path}")
    else:
        raise ValueError(f"unsupported artifact metadata kind: {path}")
    if (
        int(declaration.get("bytes", -1)) != path.stat().st_size
        or declaration.get("sha256") != _file_sha256(path)
    ):
        raise ValueError(f"artifact byte/hash metadata mismatch: {path}")


def _atomic_write_parquet(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        current = pl.read_parquet(destination)
        if not current.equals(frame, null_equal=True):
            raise FileExistsError(f"existing artifact differs: {destination}")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.unlink()
        frame.write_parquet(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"existing marker differs: {destination}")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, source: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{source} must be a lowercase SHA-256")
    return rendered


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "COMPLETION_MARKER_NAME",
    "CurrentLadderOvernightPlan",
    "CurrentLadderSourceContract",
    "DEFAULT_ENTRY_ROOT",
    "DEFAULT_EXIT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PREREQUISITE_ROOT",
    "FORMAL_SOURCE_CONTRACT",
    "FRESH_BOOK_MAX_AGE_NS",
    "MAX_CARRY_SESSIONS",
    "ORCHESTRATOR_VERSION",
    "POLICY_VERSION",
    "build_current_ladder_overnight_plan",
    "current_ladder_runner_config",
    "run_current_ladder_overnight",
    "verify_current_ladder_overnight_bundle",
]
