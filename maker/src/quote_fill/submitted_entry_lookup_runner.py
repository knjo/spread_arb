"""Validated runner for the per-submitted-entry-quote lookup checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import polars as pl

from .exit_maker_report import load_exit_maker_partition_inputs
from .submitted_entry_lookup import (
    SubmittedEntryLookupConfig,
    build_daily_submitted_entry_lookup,
    build_submitted_entry_policy_labels,
    submitted_label_category_counts,
)


CHECKPOINT_RUNNER_VERSION = "submitted_entry_quote_lookup_checkpoint_v1"
DEFAULT_EXIT_MAKER_ROOT = Path("maker/data/walkforward/exit_maker_narrow_60d")
DEFAULT_ENTRY_EXECUTION_ROOT = Path("maker/data/walkforward/execution_narrow_60d")
DEFAULT_CONDITIONAL_CHECKPOINT_ROOT = Path(
    "maker/data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight_fresh1s"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/submitted_entry_quote_lookup_checkpoint_60d_fresh1s"
)


@dataclass(frozen=True)
class SubmittedEntryLookupCheckpoint:
    labels: pl.DataFrame
    lookup: pl.DataFrame
    state_lookup: pl.DataFrame
    category_counts: pl.DataFrame
    lookup_status_summary: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


def run_submitted_entry_lookup_checkpoint(
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    conditional_checkpoint_root: Path = DEFAULT_CONDITIONAL_CHECKPOINT_ROOT,
    *,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
    include_causal_state: bool = False,
    lookup_config: SubmittedEntryLookupConfig = SubmittedEntryLookupConfig(),
) -> SubmittedEntryLookupCheckpoint:
    """Load immutable facts, build D-safe diagnostics, and publish atomically."""

    lookup_config.validate()
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
    conditional_labels, conditional_marker, conditional_marker_sha = (
        _load_conditional_nominal_labels(
            Path(conditional_checkpoint_root),
            validate_hashes=validate_hashes,
        )
    )
    conditional_dates = [str(value) for value in conditional_marker["selected_dates"]]
    if conditional_dates != selected_dates:
        raise ValueError("conditional checkpoint and exit inputs selected dates differ")
    conditional_values = sorted(
        str(value) for value in conditional_marker.get("value_codes", [])
    )
    input_values = sorted(str(value) for value in inputs.metadata["value_codes"])
    if conditional_values != input_values:
        raise ValueError("conditional checkpoint and exit inputs value codes differ")

    labels = build_submitted_entry_policy_labels(
        inputs.action_facts,
        inputs.taker_exit_facts,
        conditional_labels,
        lookup_config,
    )
    lookup = build_daily_submitted_entry_lookup(
        labels,
        selected_dates,
        lookup_config,
    )
    state_lookup = (
        build_daily_submitted_entry_lookup(
            labels,
            selected_dates,
            lookup_config,
            include_causal_state=True,
        )
        if include_causal_state
        else pl.DataFrame()
    )
    category_counts = submitted_label_category_counts(labels)
    status = _lookup_status_summary(lookup)
    metadata: dict[str, object] = {
        "runner_version": CHECKPOINT_RUNNER_VERSION,
        "exit_maker_root": str(Path(exit_maker_root)),
        "entry_execution_root": str(Path(entry_execution_root)),
        "conditional_checkpoint_root": str(Path(conditional_checkpoint_root)),
        "conditional_checkpoint_marker_sha256": conditional_marker_sha,
        "conditional_nominal_label_artifact_sha256": conditional_marker[
            "artifacts"
        ]["nominal_v0_policy_labels.parquet"]["sha256"],
        "conditional_checkpoint_overnight_root": conditional_marker.get(
            "overnight_root"
        ),
        "output_dir": str(Path(output_dir)),
        "selected_dates": selected_dates,
        "selected_session_count": len(selected_dates),
        "selected_product_day_count": inputs.metadata[
            "selected_product_day_count"
        ],
        "missing_product_day_count": inputs.metadata["missing_product_day_count"],
        "value_codes": inputs.metadata["value_codes"],
        "entry_policy_alias_rows": inputs.action_facts.height,
        "entry_raw_order_rows": inputs.action_facts.select(
            "raw_order_fact_id"
        ).n_unique(),
        "conditional_established_entry_policy_rows": conditional_labels.height,
        "submitted_policy_path_rows": labels.height,
        "lookup_rows": lookup.height,
        "state_lookup_rows": state_lookup.height,
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
        "exit_input_metadata": dict(inputs.metadata),
        "exit_partition_marker_manifest_sha256": (
            _partition_marker_manifest_sha256(inputs.coverage)
        ),
        "scenario": "per_submitted_entry_quote_nominal_v0",
        "sampling_unit": "entry_policy_alias_x_available_exit_policy_path",
        "conditional_on_established_entry": False,
        "entry_fill_probability_included": True,
        "no_fill_gross_zero_is_nominal_instant_cancel_v0_assumption": True,
        "no_fill_cancel_cost_missing": True,
        "full_hedged_paths_inherit_conditional_fresh1s_checkpoint": True,
        "partial_fill_hedge_and_exit_unknown_retained_null": True,
        "unassigned_exit_policy_aliases_retained_once": True,
        "unassigned_exit_policies_not_fabricated_or_duplicated": True,
        "flat_completed_cycle_cost_sensitivity_analysis_only": True,
        "branch_completed_cycle_cost_sensitivity_analysis_only": True,
        "branch_cost_sensitivity_not_claimed_as_formal_fee_schedule": True,
        "flat_cost_applied_once_only_to_completed_cycle": True,
        "branch_cost_applied_once_only_to_completed_cycle": True,
        "actual_execution_prices_and_slippage_already_in_gross": True,
        "price_slippage_subtracted_again": False,
        "unknown_censored_and_pending_retained": True,
        "zero_for_unpriced_output_is_sensitivity_only": True,
        "zero_for_unpriced_output_is_ev": False,
        "finite_unresolved_cashflow_limits_assumed": False,
        "unresolved_cashflow_bounds_are_null": True,
        "all_ev_ready_false": True,
        "all_expected_ev_null": True,
        "policy_rows_safe_to_sum_across_q_rule_route": False,
        "joint_volume_allocated": False,
        "implementation_sources": _implementation_sources(),
    }
    checkpoint = SubmittedEntryLookupCheckpoint(
        labels=labels,
        lookup=lookup,
        state_lookup=state_lookup,
        category_counts=category_counts,
        lookup_status_summary=status,
        coverage=inputs.coverage,
        metadata=metadata,
    )
    _validate_checkpoint(checkpoint)
    _publish_checkpoint(checkpoint, Path(output_dir))
    return checkpoint


def _load_conditional_nominal_labels(
    root: Path,
    *,
    validate_hashes: bool,
) -> tuple[pl.DataFrame, Mapping[str, object], str]:
    marker_path = root / "checkpoint_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"missing conditional checkpoint marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("complete") is not True:
        raise ValueError("conditional checkpoint is not complete")
    if marker.get("conditional_on_established_entry") is not True:
        raise ValueError("conditional checkpoint has unexpected denominator")
    if marker.get("nominal_instant_cancel_v0_is_model_assumption") is not True:
        raise ValueError("conditional checkpoint lacks nominal V0 declaration")
    artifact_name = "nominal_v0_policy_labels.parquet"
    artifact_meta = marker.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact_meta, dict):
        raise ValueError("conditional checkpoint lacks nominal V0 artifact metadata")
    artifact = root / artifact_name
    if not artifact.is_file():
        raise FileNotFoundError(f"missing conditional label artifact: {artifact}")
    if validate_hashes and _file_sha256(artifact) != artifact_meta.get("sha256"):
        raise ValueError("conditional nominal V0 label artifact hash mismatch")
    labels = pl.read_parquet(artifact)
    if labels.height != int(artifact_meta["rows"]):
        raise ValueError("conditional nominal V0 label row count mismatch")
    return labels, marker, _file_sha256(marker_path)


def _lookup_status_summary(lookup: pl.DataFrame) -> pl.DataFrame:
    if lookup.is_empty():
        return pl.DataFrame()
    return lookup.group_by(["asof_date", "ev_status"]).agg(
        pl.len().cast(pl.Int64).alias("lookup_cells"),
        pl.col("n_policy_paths").sum().cast(pl.Int64).alias("policy_paths"),
        pl.col("n_entry_policy_aliases")
        .sum()
        .cast(pl.Int64)
        .alias("entry_policy_alias_path_denominator"),
        pl.col("n_outcome_known").sum().cast(pl.Int64).alias("known_paths"),
        pl.col("n_outcome_censored")
        .sum()
        .cast(pl.Int64)
        .alias("censored_paths"),
        pl.col("n_outcome_unknown")
        .sum()
        .cast(pl.Int64)
        .alias("unknown_paths"),
        pl.col("n_labels_pending").sum().cast(pl.Int64).alias("pending_labels"),
    ).sort(["asof_date", "ev_status"])


def _validate_checkpoint(checkpoint: SubmittedEntryLookupCheckpoint) -> None:
    if checkpoint.labels.height == 0:
        raise ValueError("submitted checkpoint unexpectedly has no labels")
    for name, frame in (
        ("lookup", checkpoint.lookup),
        ("state lookup", checkpoint.state_lookup),
    ):
        if frame.is_empty():
            continue
        if frame.filter(pl.col("ev_ready")).height:
            raise ValueError(f"{name} unexpectedly marks EV ready")
        if frame.filter(pl.col("expected_net_cashflow_bp").is_not_null()).height:
            raise ValueError(f"{name} unexpectedly contains EV")
        if frame.filter(pl.col("contains_target_day_outcome")).height:
            raise ValueError(f"{name} contains target-day outcomes")
        if frame.filter(pl.col("label_cutoff_date") >= pl.col("asof_date")).height:
            raise ValueError(f"{name} violates maturity cutoff")
    if checkpoint.lookup.is_empty():
        raise ValueError("submitted checkpoint unexpectedly has no lookup rows")


def _publish_checkpoint(
    checkpoint: SubmittedEntryLookupCheckpoint,
    destination: Path,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"checkpoint output already exists; choose a new path: {destination}"
        )
    artifacts: dict[str, pl.DataFrame] = {
        "submitted_entry_policy_labels.parquet": checkpoint.labels,
        "daily_prequential_submitted_entry_lookup.parquet": checkpoint.lookup,
        "daily_prequential_submitted_entry_state_lookup.parquet": (
            checkpoint.state_lookup
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
        csv_artifacts = {
            "submitted_label_category_counts.csv": checkpoint.category_counts,
            "lookup_status_summary.csv": checkpoint.lookup_status_summary,
            "coverage.csv": checkpoint.coverage,
        }
        for name, frame in csv_artifacts.items():
            path = stage / name
            frame.write_csv(path)
            artifact_metadata[name] = {
                "rows": frame.height,
                "columns": len(frame.columns),
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
        "submitted_entry_lookup.py",
        "submitted_entry_lookup_runner.py",
        "exit_policy_lookup.py",
        "exit_maker_report.py",
    )
    return {name: _file_sha256(module_root / name) for name in names}


def _partition_marker_manifest_sha256(coverage: pl.DataFrame) -> str:
    records: list[dict[str, str]] = []
    for row in coverage.filter(pl.col("partition_complete")).sort(
        ["Date", "ValueCode"]
    ).iter_rows(named=True):
        partition = row.get("partition")
        if partition is None:
            raise ValueError("complete exit coverage row lacks partition")
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
