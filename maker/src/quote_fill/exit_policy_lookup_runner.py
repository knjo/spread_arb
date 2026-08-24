"""Validated runner and atomic publisher for exit-policy lookup checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import polars as pl

from .exit_maker_report import load_exit_maker_partition_inputs
from .exit_maker_terminal import ExitMakerTerminalConfig
from .exit_policy_lookup import (
    ExitPolicyLabelTables,
    ExitPolicyLookupConfig,
    build_daily_prequential_lookup,
    build_exit_policy_label_tables,
)
from .overnight_carry_report import load_overnight_carry_partition_inputs


CHECKPOINT_RUNNER_VERSION = "conditional_exit_policy_lookup_checkpoint_v1"
DEFAULT_EXIT_MAKER_ROOT = Path("maker/data/walkforward/exit_maker_narrow_60d")
DEFAULT_ENTRY_EXECUTION_ROOT = Path("maker/data/walkforward/execution_narrow_60d")
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/exit_policy_lookup_checkpoint_60d"
)


@dataclass(frozen=True)
class ExitPolicyLookupCheckpoint:
    """Published facts, daily lookups, audit summaries, and lineage metadata."""

    labels: ExitPolicyLabelTables
    strict_lookup: pl.DataFrame
    nominal_v0_lookup: pl.DataFrame
    strict_state_lookup: pl.DataFrame
    nominal_v0_state_lookup: pl.DataFrame
    lookup_status_summary: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


def run_exit_policy_lookup_checkpoint(
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    *,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    overnight_root: Path | None = None,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
    include_causal_state: bool = True,
    lookup_config: ExitPolicyLookupConfig = ExitPolicyLookupConfig(),
    terminal_config: ExitMakerTerminalConfig = ExitMakerTerminalConfig(
        lifecycle_policy_version="layered_execution_observation_v1",
        queue_scenario="displayed_queue_independent_candidate_v1",
    ),
) -> ExitPolicyLookupCheckpoint:
    """Load immutable roots, build D-safe tables, and publish to a new root."""

    lookup_config.validate()
    terminal_config.validate()
    products = tuple(str(value) for value in value_codes) if value_codes else None
    inputs = load_exit_maker_partition_inputs(
        Path(exit_maker_root),
        Path(entry_execution_root),
        sessions=sessions,
        value_codes=products,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=require_balanced_product_days,
        validate_hashes=validate_hashes,
    )
    selected_dates = [str(value) for value in inputs.metadata["selected_dates"]]
    exit_partition_marker_manifest_sha256 = _partition_marker_manifest_sha256(
        inputs.coverage
    )
    overnight_labels: pl.DataFrame | None = None
    overnight_metadata: Mapping[str, object] | None = None
    if overnight_root is not None:
        overnight = load_overnight_carry_partition_inputs(
            Path(overnight_root),
            sessions=len(selected_dates),
            value_codes=products,
            require_exact_sessions=True,
            require_balanced_product_days=require_balanced_product_days,
            validate_hashes=validate_hashes,
        )
        overnight_dates = [str(value) for value in overnight.metadata["selected_dates"]]
        if overnight_dates != selected_dates:
            raise ValueError("overnight and exit-maker selected dates differ")
        _validate_matching_coverage(inputs.coverage, overnight.coverage)
        overnight_labels = overnight.labels
        overnight_metadata = overnight.metadata

    labels = build_exit_policy_label_tables(
        inputs.action_facts,
        inputs.position_policy_facts,
        terminal_config=terminal_config,
        assumed_non_price_cycle_cost_bp=(
            lookup_config.assumed_non_price_cycle_cost_bp
        ),
        overnight_labels=overnight_labels,
    )
    strict_lookup = build_daily_prequential_lookup(
        labels.strict,
        selected_dates,
        lookup_config,
    )
    nominal_lookup = build_daily_prequential_lookup(
        labels.nominal_v0,
        selected_dates,
        lookup_config,
    )
    strict_state = (
        build_daily_prequential_lookup(
            labels.strict,
            selected_dates,
            lookup_config,
            include_causal_state=True,
        )
        if include_causal_state
        else pl.DataFrame()
    )
    nominal_state = (
        build_daily_prequential_lookup(
            labels.nominal_v0,
            selected_dates,
            lookup_config,
            include_causal_state=True,
        )
        if include_causal_state
        else pl.DataFrame()
    )
    summary = _lookup_status_summary(strict_lookup, nominal_lookup)
    metadata: dict[str, object] = {
        "runner_version": CHECKPOINT_RUNNER_VERSION,
        "exit_maker_root": str(Path(exit_maker_root)),
        "entry_execution_root": str(Path(entry_execution_root)),
        "overnight_root": (
            str(Path(overnight_root)) if overnight_root is not None else None
        ),
        "output_dir": str(Path(output_dir)),
        "selected_dates": selected_dates,
        "selected_session_count": len(selected_dates),
        "selected_product_day_count": inputs.metadata[
            "selected_product_day_count"
        ],
        "missing_product_day_count": inputs.metadata["missing_product_day_count"],
        "value_codes": inputs.metadata["value_codes"],
        "strict_policy_label_rows": labels.strict.height,
        "nominal_v0_policy_label_rows": labels.nominal_v0.height,
        "strict_lookup_rows": strict_lookup.height,
        "nominal_v0_lookup_rows": nominal_lookup.height,
        "strict_state_lookup_rows": strict_state.height,
        "nominal_v0_state_lookup_rows": nominal_state.height,
        "primary_lookup_key": [
            "ValueCode",
            "entry_route",
            "entry_q",
            "exit_rule_id",
            "exit_route",
        ],
        "causal_state_lookup_optional": True,
        "causal_state_lookup_emitted": include_causal_state,
        "lookup_config": asdict(lookup_config),
        "terminal_config": asdict(terminal_config),
        "exit_input_metadata": dict(inputs.metadata),
        "exit_partition_marker_manifest_sha256": (
            exit_partition_marker_manifest_sha256
        ),
        "overnight_input_metadata": (
            dict(overnight_metadata) if overnight_metadata is not None else None
        ),
        "overnight_labels_attached": overnight_labels is not None,
        "strict_cancel_race_kept_unknown": True,
        "nominal_instant_cancel_v0_is_model_assumption": True,
        "nominal_instant_cancel_v0_analysis_only": True,
        "conditional_on_established_entry": True,
        "entry_fill_probability_included": False,
        "actual_four_leg_prices_in_gross": True,
        "price_slippage_subtracted_again": False,
        "flat_assumed_cost_is_non_price_sensitivity_only": True,
        "unknown_and_censored_retained": True,
        "unmatured_or_null_maturity_labels_retained_as_pending": True,
        "unknown_imputed_as_zero_for_ev": False,
        "zero_for_unpriced_output_is_sensitivity_only": True,
        "finite_unresolved_cashflow_limits_assumed": False,
        "unresolved_cashflow_bounds_are_null": True,
        "all_ev_ready_false": True,
        "all_expected_ev_null": True,
        "policy_rows_safe_to_sum_across_q_rule_route": False,
        "joint_volume_allocated": False,
        "implementation_sources": _implementation_sources(),
    }
    checkpoint = ExitPolicyLookupCheckpoint(
        labels=labels,
        strict_lookup=strict_lookup,
        nominal_v0_lookup=nominal_lookup,
        strict_state_lookup=strict_state,
        nominal_v0_state_lookup=nominal_state,
        lookup_status_summary=summary,
        coverage=inputs.coverage,
        metadata=metadata,
    )
    _validate_checkpoint(checkpoint)
    _publish_checkpoint(checkpoint, Path(output_dir))
    return checkpoint


def _lookup_status_summary(
    strict: pl.DataFrame, nominal: pl.DataFrame
) -> pl.DataFrame:
    frames = [frame for frame in (strict, nominal) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(
            schema={
                "asof_date": pl.String,
                "scenario": pl.String,
                "ev_status": pl.String,
                "lookup_cells": pl.Int64,
                "policy_paths": pl.Int64,
                "known_paths": pl.Int64,
                "censored_paths": pl.Int64,
                "unknown_paths": pl.Int64,
                "pending_labels": pl.Int64,
            }
        )
    return pl.concat(frames, how="diagonal_relaxed").group_by(
        ["asof_date", "scenario", "ev_status"]
    ).agg(
        pl.len().cast(pl.Int64).alias("lookup_cells"),
        pl.col("n_policy_paths").sum().cast(pl.Int64).alias("policy_paths"),
        pl.col("n_outcome_known").sum().cast(pl.Int64).alias("known_paths"),
        pl.col("n_outcome_censored")
        .sum()
        .cast(pl.Int64)
        .alias("censored_paths"),
        pl.col("n_outcome_unknown")
        .sum()
        .cast(pl.Int64)
        .alias("unknown_paths"),
        pl.col("n_labels_pending")
        .sum()
        .cast(pl.Int64)
        .alias("pending_labels"),
    ).sort(["asof_date", "scenario", "ev_status"])


def _validate_matching_coverage(exit_coverage: pl.DataFrame, overnight: pl.DataFrame) -> None:
    keys = ["Date", "ValueCode", "partition_complete"]
    left = exit_coverage.select(keys).sort(keys[:2])
    right = overnight.select(keys).sort(keys[:2])
    if not left.equals(right):
        raise ValueError("overnight and exit-maker product-day coverage differ")


def _validate_checkpoint(checkpoint: ExitPolicyLookupCheckpoint) -> None:
    if checkpoint.labels.strict.height != checkpoint.labels.nominal_v0.height:
        raise ValueError("strict and nominal V0 label populations differ")
    for name, frame in (
        ("strict lookup", checkpoint.strict_lookup),
        ("nominal lookup", checkpoint.nominal_v0_lookup),
        ("strict state lookup", checkpoint.strict_state_lookup),
        ("nominal state lookup", checkpoint.nominal_v0_state_lookup),
    ):
        if frame.is_empty():
            continue
        if frame.filter(pl.col("ev_ready")).height:
            raise ValueError(f"{name} unexpectedly contains ev_ready=true")
        if frame.filter(pl.col("expected_net_cashflow_bp").is_not_null()).height:
            raise ValueError(f"{name} unexpectedly priced production EV")
        if frame.filter(pl.col("contains_target_day_outcome")).height:
            raise ValueError(f"{name} contains target-day outcomes")
        if frame.filter(pl.col("label_cutoff_date") >= pl.col("asof_date")).height:
            raise ValueError(f"{name} violates label maturity cutoff")


def _publish_checkpoint(
    checkpoint: ExitPolicyLookupCheckpoint,
    destination: Path,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"checkpoint output already exists; choose a new path: {destination}"
        )
    artifacts: dict[str, pl.DataFrame] = {
        "strict_policy_labels.parquet": checkpoint.labels.strict,
        "nominal_v0_policy_labels.parquet": checkpoint.labels.nominal_v0,
        "daily_prequential_strict_lookup.parquet": checkpoint.strict_lookup,
        "daily_prequential_nominal_v0_lookup.parquet": (
            checkpoint.nominal_v0_lookup
        ),
        "daily_prequential_strict_state_lookup.parquet": (
            checkpoint.strict_state_lookup
        ),
        "daily_prequential_nominal_v0_state_lookup.parquet": (
            checkpoint.nominal_v0_state_lookup
        ),
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        artifact_metadata: dict[str, object] = {}
        for name, frame in artifacts.items():
            path = stage / name
            frame.write_parquet(path)
            artifact_metadata[name] = {
                "rows": frame.height,
                "columns": len(frame.columns),
                "sha256": _file_sha256(path),
            }
        checkpoint.lookup_status_summary.write_csv(stage / "lookup_status_summary.csv")
        checkpoint.coverage.write_csv(stage / "coverage.csv")
        for name in ("lookup_status_summary.csv", "coverage.csv"):
            path = stage / name
            artifact_metadata[name] = {
                "rows": (
                    checkpoint.lookup_status_summary.height
                    if name.startswith("lookup")
                    else checkpoint.coverage.height
                ),
                "sha256": _file_sha256(path),
            }
        payload = {
            **dict(checkpoint.metadata),
            "artifacts": artifact_metadata,
            "complete": True,
        }
        (stage / "checkpoint_complete.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)


def _implementation_sources() -> Mapping[str, str]:
    module_root = Path(__file__).parent
    names = (
        "exit_policy_lookup.py",
        "exit_policy_lookup_runner.py",
        "exit_maker_terminal.py",
        "ev_surface.py",
        "exit_maker_report.py",
        "overnight_carry_report.py",
    )
    return {name: _file_sha256(module_root / name) for name in names}


def _partition_marker_manifest_sha256(coverage: pl.DataFrame) -> str:
    records: list[dict[str, str]] = []
    for row in coverage.filter(pl.col("partition_complete")).sort(
        ["Date", "ValueCode"]
    ).iter_rows(named=True):
        partition = row.get("partition")
        if partition is None:
            raise ValueError("complete exit coverage row is missing partition path")
        marker = Path(str(partition)) / "complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"missing exit partition marker: {marker}")
        records.append(
            {
                "Date": str(row["Date"]),
                "ValueCode": str(row["ValueCode"]),
                "marker_sha256": _file_sha256(marker),
            }
        )
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
