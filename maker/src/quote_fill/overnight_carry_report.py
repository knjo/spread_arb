"""Audited aggregation for partitioned strict overnight-carry labels.

The input unit is an atomic ``Date=.../ValueCode=...`` partition containing
exactly one label artifact, one audit artifact, and a complete marker.  This
module intentionally validates those public files directly; it does not
depend on any runner-private verification helper.

Policy labels remain alternative q/rule/route observations.  A separate
physical-fact table deduplicates ``physical_entry_dependency_id`` so the same
entry cashflow cannot accidentally acquire extra statistical weight merely
because several policy aliases refer to it.  All reported PnL is gross,
before cost.  Missing fee, tax, financing, overnight, cancel, and emergency
costs remain null and every table is explicitly not pathwise-EV-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from .ev_surface import DEFAULT_COST_COLUMNS
from .overnight_carry import STRICT_CARRY_BRANCHES


REPORT_VERSION = "overnight_carry_partition_report_v1"
INPUT_ARTIFACTS = (
    "overnight_carry_labels.parquet",
    "overnight_carry_audit.parquet",
)
REPORT_ARTIFACTS = (
    "combined_overnight_carry_labels.parquet",
    "combined_overnight_carry_audits.parquet",
    "overnight_carry_physical_facts.parquet",
    "overnight_carry_policy_summary.parquet",
    "overnight_carry_status_summary.parquet",
)

POLICY_SUMMARY_KEYS = [
    "ValueCode",
    "entry_route",
    "exit_rule_id",
    "exit_route",
    "carry_policy_version",
]
STATUS_SUMMARY_KEYS = ["label_status", "terminal_branch", "outcome_status"]

_LABEL_REQUIRED = {
    "carry_label_id",
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "source_branch_status",
    "position_established_ns",
    "entry_spot_price",
    "entry_future_price",
    "contract_size_shares",
    "physical_entry_dependency_id",
    "physical_entry_strict_carry_policy_alias_count",
    "physical_entry_coverage_weight",
    "policy_observation_weight",
    "physical_entry_alias_nonindependent",
    "cross_q_rule_route_additive",
    "sampling_unit",
    "expiry_session",
    "calendar_version",
    "carry_policy_version",
    "max_carry_sessions",
    "max_book_age_ns",
    "max_book_age_enforced",
    "added_exit_latency_ns",
    "latency_matched_to_exit_maker",
    "exact_quote_code_required",
    "roll_attempted",
    "roll_policy_version",
    "new_quote_code",
    "label_status",
    "terminal_branch",
    "transition_branch",
    "outcome_status",
    "unresolved_reason",
    "exit_date",
    "label_end_date",
    "exit_decision_time_ns",
    "exit_spot_snapshot_time_ns",
    "exit_future_snapshot_time_ns",
    "exit_spot_price",
    "exit_future_price",
    "exit_spot_levels_swept",
    "exit_future_levels_swept",
    "exit_spot_book_age_ms",
    "exit_future_book_age_ms",
    "settlement_price",
    "settlement_time_ns",
    "settlement_source_version",
    "gross_cycle_pnl_twd",
    "normalization_notional_twd",
    "filled_cashflow_before_cost_bp",
    "holding_seconds",
    "terminal_cashflow_priced",
    "counterfactual_executable",
    "execution_benchmark_role",
    "book_freshness_status",
    "needs_next_session_label",
    *DEFAULT_COST_COLUMNS,
    "cost_profile_version",
    "fees_tax_included",
    "pathwise_ev_ready",
}

_AUDIT_REQUIRED = {
    "position_policy_rows",
    "strict_carry_policy_rows",
    "excluded_noncarry_or_unknown_rows",
    "label_rows",
    "priced_terminal_rows",
    "overnight_exit_rows",
    "expiry_settlement_rows",
    "expiry_settlement_unpriced_rows",
    "roll_substitution_forbidden_rows",
    "unresolved_rows",
    "fees_tax_complete_rows",
    "pathwise_ev_ready_rows",
}

_PHYSICAL_VALUE_COLUMNS = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_raw_order_fact_id",
    "position_established_ns",
    "entry_spot_price",
    "entry_future_price",
    "contract_size_shares",
    "expiry_session",
    "calendar_version",
    "carry_policy_version",
    "max_carry_sessions",
    "max_book_age_ns",
    "max_book_age_enforced",
    "added_exit_latency_ns",
    "latency_matched_to_exit_maker",
    "exact_quote_code_required",
    "roll_attempted",
    "roll_policy_version",
    "new_quote_code",
    "label_status",
    "terminal_branch",
    "transition_branch",
    "outcome_status",
    "unresolved_reason",
    "exit_date",
    "label_end_date",
    "exit_decision_time_ns",
    "exit_spot_snapshot_time_ns",
    "exit_future_snapshot_time_ns",
    "exit_spot_price",
    "exit_future_price",
    "exit_spot_levels_swept",
    "exit_future_levels_swept",
    "exit_spot_book_age_ms",
    "exit_future_book_age_ms",
    "settlement_price",
    "settlement_time_ns",
    "settlement_source_version",
    "gross_cycle_pnl_twd",
    "normalization_notional_twd",
    "filled_cashflow_before_cost_bp",
    "holding_seconds",
    "terminal_cashflow_priced",
    "counterfactual_executable",
    "execution_benchmark_role",
    "book_freshness_status",
    "needs_next_session_label",
]


@dataclass(frozen=True)
class OvernightCarryPartitionInputs:
    """Validated and combined product-day artifacts."""

    labels: pl.DataFrame
    audits: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class OvernightCarryReport:
    """Report tables with policy and physical sampling units kept separate."""

    combined_labels: pl.DataFrame
    combined_audits: pl.DataFrame
    physical_facts: pl.DataFrame
    policy_summary: pl.DataFrame
    status_summary: pl.DataFrame
    metadata: Mapping[str, object]


def load_overnight_carry_partition_inputs(
    overnight_root: Path,
    *,
    sessions: int | None = None,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
) -> OvernightCarryPartitionInputs:
    """Discover complete partitions and validate marker/artifact integrity."""

    root = Path(overnight_root)
    _validate_session_request(sessions)
    requested_products = _normalise_products(value_codes)
    records = _discover_markers(root, requested_products)
    if not records:
        raise FileNotFoundError(f"no complete overnight-carry partitions under {root}")

    available_dates = sorted({str(record["Date"]) for record in records})
    if sessions is not None and require_exact_sessions and len(available_dates) < sessions:
        raise ValueError(
            f"requested {sessions} complete sessions, found {len(available_dates)}"
        )
    selected_dates = available_dates[-sessions:] if sessions is not None else available_dates
    selected = [record for record in records if str(record["Date"]) in selected_dates]
    products = list(
        requested_products
        if requested_products is not None
        else sorted({str(record["ValueCode"]) for record in selected})
    )
    selected_by_key = {
        (str(record["Date"]), str(record["ValueCode"])): record
        for record in selected
    }
    if len(selected_by_key) != len(selected):
        raise ValueError("duplicate complete overnight-carry marker for a product-day")

    coverage_rows: list[dict[str, object]] = []
    for date in selected_dates:
        for value_code in products:
            record = selected_by_key.get((date, value_code))
            coverage_rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "partition_complete": record is not None,
                    "partition": str(record["partition"]) if record else None,
                }
            )
    coverage = pl.from_dicts(coverage_rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )
    missing = coverage.filter(~pl.col("partition_complete"))
    if require_balanced_product_days and missing.height:
        examples = [
            f"{row['Date']}/{row['ValueCode']}"
            for row in missing.head(5).iter_rows(named=True)
        ]
        raise ValueError("incomplete date-by-product grid; missing " + ", ".join(examples))

    label_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    label_schema: Mapping[str, pl.DataType] | None = None
    audit_schema: Mapping[str, pl.DataType] | None = None
    runner_versions: set[str] = set()
    runner_config_hashes: set[str] = set()
    marker_manifest: list[dict[str, str]] = []
    for record in selected:
        partition = Path(record["partition"])
        payload = record["payload"]
        assert isinstance(payload, dict)
        marker_info = _validate_partition_marker(
            payload,
            partition,
            validate_hashes=validate_hashes,
        )
        runner_version = payload.get("runner_version")
        if runner_version is not None:
            runner_versions.add(str(runner_version))
        runner_hash = marker_info.get("runner_config_sha256")
        if runner_hash is not None:
            runner_config_hashes.add(str(runner_hash))

        labels_path = partition / INPUT_ARTIFACTS[0]
        audit_path = partition / INPUT_ARTIFACTS[1]
        current_label_schema = pl.read_parquet_schema(labels_path)
        current_audit_schema = pl.read_parquet_schema(audit_path)
        label_schema = _same_schema(label_schema, current_label_schema, labels_path)
        audit_schema = _same_schema(audit_schema, current_audit_schema, audit_path)
        labels = pl.read_parquet(labels_path)
        audit = pl.read_parquet(audit_path)
        date = str(record["Date"])
        value_code = str(record["ValueCode"])
        _validate_partition_frames(labels, audit, date, value_code, partition)
        label_frames.append(labels)
        audit_frames.append(
            audit.with_columns(
                pl.lit(date).alias("Date"),
                pl.lit(value_code).alias("ValueCode"),
                pl.lit(str(partition)).alias("source_partition"),
            )
        )
        marker_manifest.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "marker_sha256": _file_sha256(Path(record["marker"])),
            }
        )

    if len(runner_versions) > 1:
        raise ValueError("selected overnight partitions do not share one runner version")
    if len(runner_config_hashes) > 1:
        raise ValueError("selected overnight partitions do not share one runner config")
    root_manifest_validated = _validate_root_manifest(root, selected)

    labels = pl.concat(label_frames, how="vertical", rechunk=True)
    audits = pl.concat(audit_frames, how="vertical", rechunk=True)
    _validate_combined_labels(labels)
    audits = _add_output_invariants(audits)
    metadata: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "overnight_root": str(root),
        "selected_session_count": len(selected_dates),
        "selected_dates": selected_dates,
        "selected_product_count": len(products),
        "selected_product_day_count": len(selected),
        "expected_product_day_count": len(selected_dates) * len(products),
        "missing_product_day_count": missing.height,
        "balanced_product_day_grid": missing.is_empty(),
        "value_codes": products,
        "runner_version": next(iter(runner_versions)) if runner_versions else None,
        "runner_config_sha256": (
            next(iter(runner_config_hashes)) if runner_config_hashes else None
        ),
        "partition_hashes_validated": validate_hashes,
        "root_manifest_validated": root_manifest_validated,
        "partition_marker_manifest_sha256": _canonical_sha256(marker_manifest),
    }
    return OvernightCarryPartitionInputs(
        labels=labels,
        audits=audits,
        coverage=coverage,
        metadata=metadata,
    )


def build_overnight_carry_report(
    inputs: OvernightCarryPartitionInputs,
) -> OvernightCarryReport:
    """Build gross-only policy summaries and deduplicated physical facts."""

    _require(inputs.labels, _LABEL_REQUIRED, "overnight carry labels")
    _require(inputs.audits, _AUDIT_REQUIRED, "overnight carry audits")
    labels = inputs.labels.sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "exit_rule_id",
            "exit_route",
            "exit_policy_trial_id",
        ]
    )
    _validate_combined_labels(labels)
    audits = _add_output_invariants(inputs.audits)
    physical = _build_physical_facts(labels)
    policy = _build_policy_summary(labels)
    status = _build_status_summary(labels)
    metadata = dict(inputs.metadata)
    module_root = Path(__file__).parent
    report_sources = {
        name: _file_sha256(module_root / name)
        for name in (
            "overnight_carry_report.py",
            "overnight_carry.py",
            "ev_surface.py",
        )
    }
    metadata.update(
        {
            "report_semantics": "strict_overnight_gross_policy_and_physical_v1",
            "policy_label_rows": labels.height,
            "physical_entry_rows": physical.height,
            "policy_sampling_unit": "exit_policy_trial_alternative",
            "physical_sampling_unit": "physical_entry_dependency_id",
            "gross_only": True,
            "net_or_after_cost_statistics_present": False,
            "cost_columns_all_null": True,
            "fees_tax_included": False,
            "pathwise_ev_ready": False,
            "cross_q_rule_route_additive": False,
            "boundary_quantile_field_available": False,
            "policy_summary_rows_safe_to_sum_across_q_rule_route": False,
            "physical_entry_deduplicated": True,
            "report_implementation_sources": report_sources,
            "report_implementation_sha256": _canonical_sha256(report_sources),
        }
    )
    return OvernightCarryReport(
        combined_labels=labels,
        combined_audits=audits,
        physical_facts=physical,
        policy_summary=policy,
        status_summary=status,
        metadata=metadata,
    )


def run_overnight_carry_report(
    overnight_root: Path,
    *,
    output_dir: Path | None = None,
    sessions: int | None = None,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
) -> OvernightCarryReport:
    """Validate, build, and atomically publish one overnight report."""

    inputs = load_overnight_carry_partition_inputs(
        overnight_root,
        sessions=sessions,
        value_codes=value_codes,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=require_balanced_product_days,
        validate_hashes=validate_hashes,
    )
    report = build_overnight_carry_report(inputs)
    count = int(inputs.metadata["selected_session_count"])
    source_digest = str(inputs.metadata["partition_marker_manifest_sha256"])
    implementation_digest = str(report.metadata["report_implementation_sha256"])
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(overnight_root)
        / (
            f"report_{count}_sessions_"
            f"{source_digest[:10]}_{implementation_digest[:10]}"
        )
    )
    _publish_report(report, destination)
    return report


def _build_physical_facts(labels: pl.DataFrame) -> pl.DataFrame:
    key = "physical_entry_dependency_id"
    if labels.is_empty():
        base = labels.select([key, *_PHYSICAL_VALUE_COLUMNS, *DEFAULT_COST_COLUMNS])
        return _add_output_invariants(
            base.with_columns(
                pl.lit(None, dtype=pl.String).alias("cost_profile_version"),
                pl.lit(False).alias("fees_tax_included"),
                pl.lit(0, dtype=pl.Int64).alias("policy_alias_rows"),
                pl.lit(0, dtype=pl.Int64).alias("entry_policy_alias_count"),
                pl.lit(0, dtype=pl.Int64).alias("exit_rule_count"),
                pl.lit(0, dtype=pl.Int64).alias("exit_route_count"),
                pl.lit(0, dtype=pl.Int64).alias("source_branch_count"),
                pl.lit("physical_entry_dependency", dtype=pl.String).alias(
                    "report_sampling_unit"
                ),
            )
        )

    consistency = labels.group_by(key).agg(
        *[
            pl.col(column).n_unique().alias(f"_{column}_n")
            for column in _PHYSICAL_VALUE_COLUMNS
        ]
    )
    bad = consistency.filter(
        pl.any_horizontal(
            *[pl.col(f"_{column}_n") != 1 for column in _PHYSICAL_VALUE_COLUMNS]
        )
    )
    if bad.height:
        raise ValueError(
            "policy aliases sharing physical_entry_dependency_id disagree on physical outcome"
        )

    counts = labels.group_by(key).agg(
        pl.len().cast(pl.Int64).alias("policy_alias_rows"),
        pl.col("entry_policy_generation_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("entry_policy_alias_count"),
        pl.col("exit_rule_id").n_unique().cast(pl.Int64).alias("exit_rule_count"),
        pl.col("exit_route").n_unique().cast(pl.Int64).alias("exit_route_count"),
        pl.col("source_branch_status")
        .n_unique()
        .cast(pl.Int64)
        .alias("source_branch_count"),
    )
    representatives = (
        labels.sort(
            [key, "entry_policy_generation_id", "exit_rule_id", "exit_route"]
        )
        .unique(subset=[key], keep="first", maintain_order=True)
        .select([key, *_PHYSICAL_VALUE_COLUMNS, *DEFAULT_COST_COLUMNS, "cost_profile_version", "fees_tax_included"])
    )
    return _add_output_invariants(
        representatives.join(counts, on=key, how="left", validate="1:1")
        .with_columns(
            pl.lit("physical_entry_dependency").alias("report_sampling_unit")
        )
        .sort(["Date", "ValueCode", key])
    )


def _build_policy_summary(labels: pl.DataFrame) -> pl.DataFrame:
    priced = pl.col("terminal_cashflow_priced")
    result = labels.group_by(POLICY_SUMMARY_KEYS).agg(
        pl.len().cast(pl.Int64).alias("policy_label_rows"),
        pl.col("Date").n_unique().cast(pl.Int64).alias("sessions_with_labels"),
        pl.col("QuoteCode").n_unique().cast(pl.Int64).alias("quote_codes"),
        pl.col("entry_policy_generation_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("entry_policy_aliases"),
        pl.col("physical_entry_dependency_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("unique_physical_entries"),
        priced.sum().cast(pl.Int64).alias("priced_policy_rows"),
        (pl.col("outcome_status") == "censored")
        .sum()
        .cast(pl.Int64)
        .alias("censored_policy_rows"),
        (pl.col("outcome_status") == "unknown")
        .sum()
        .cast(pl.Int64)
        .alias("unknown_policy_rows"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .min()
        .alias("gross_before_cost_bp_min"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .quantile(0.05)
        .alias("gross_before_cost_bp_p05"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .median()
        .alias("gross_before_cost_bp_p50"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .quantile(0.95)
        .alias("gross_before_cost_bp_p95"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .max()
        .alias("gross_before_cost_bp_max"),
        (pl.col("filled_cashflow_before_cost_bp") > 0)
        .filter(priced)
        .sum()
        .cast(pl.Int64)
        .alias("positive_gross_policy_rows"),
        pl.col("gross_cycle_pnl_twd")
        .filter(priced)
        .median()
        .alias("gross_cycle_pnl_twd_p50"),
        pl.col("holding_seconds")
        .filter(priced)
        .median()
        .alias("holding_seconds_p50"),
        pl.col("holding_seconds")
        .filter(priced)
        .quantile(0.95)
        .alias("holding_seconds_p95"),
    )
    result = result.with_columns(
        pl.when(pl.col("policy_label_rows") > 0)
        .then(pl.col("priced_policy_rows") / pl.col("policy_label_rows"))
        .otherwise(None)
        .alias("priced_policy_rate"),
        pl.when(pl.col("priced_policy_rows") > 0)
        .then(pl.col("positive_gross_policy_rows") / pl.col("priced_policy_rows"))
        .otherwise(None)
        .alias("positive_gross_rate_given_priced"),
        pl.lit("exit_policy_trial_alternative").alias("report_sampling_unit"),
        pl.lit("known_priced_policy_labels_gross_before_cost").alias(
            "gross_statistic_scope"
        ),
    )
    return _add_output_invariants(result).sort(POLICY_SUMMARY_KEYS)


def _build_status_summary(labels: pl.DataFrame) -> pl.DataFrame:
    priced = pl.col("terminal_cashflow_priced")
    result = labels.group_by(STATUS_SUMMARY_KEYS).agg(
        pl.len().cast(pl.Int64).alias("policy_label_rows"),
        pl.col("physical_entry_dependency_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("unique_physical_entries"),
        priced.sum().cast(pl.Int64).alias("priced_policy_rows"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .quantile(0.05)
        .alias("gross_before_cost_bp_p05"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .median()
        .alias("gross_before_cost_bp_p50"),
        pl.col("filled_cashflow_before_cost_bp")
        .filter(priced)
        .quantile(0.95)
        .alias("gross_before_cost_bp_p95"),
    ).with_columns(
        pl.lit("status_across_policy_alternatives").alias("report_sampling_unit"),
        pl.lit("known_priced_policy_labels_gross_before_cost").alias(
            "gross_statistic_scope"
        ),
    )
    return _add_output_invariants(result).sort(STATUS_SUMMARY_KEYS)


def _validate_partition_frames(
    labels: pl.DataFrame,
    audit: pl.DataFrame,
    date: str,
    value_code: str,
    partition: Path,
) -> None:
    _require(labels, _LABEL_REQUIRED, f"overnight labels {partition}")
    _require(audit, _AUDIT_REQUIRED, f"overnight audit {partition}")
    if audit.height != 1:
        raise ValueError(f"overnight audit must contain one row: {partition}")
    for column, expected in (("Date", date), ("ValueCode", value_code)):
        values = {str(value) for value in labels[column].unique()}
        if values and values != {expected}:
            raise ValueError(
                f"overnight label partition identity mismatch on {column}: {partition}"
            )
        if column in audit.columns:
            audit_values = {str(value) for value in audit[column].unique()}
            if audit_values != {expected}:
                raise ValueError(
                    f"overnight audit partition identity mismatch on {column}: {partition}"
                )
    if int(audit.item(0, "label_rows")) != labels.height:
        raise ValueError(f"overnight audit label_rows mismatch: {partition}")
    if int(audit.item(0, "strict_carry_policy_rows")) != labels.height:
        raise ValueError(f"overnight audit strict carry count mismatch: {partition}")
    if int(audit.item(0, "fees_tax_complete_rows")) != 0:
        raise ValueError(f"overnight audit unexpectedly reports complete costs: {partition}")
    if int(audit.item(0, "pathwise_ev_ready_rows")) != 0:
        raise ValueError(f"overnight audit unexpectedly reports EV-ready rows: {partition}")
    priced_rows = int(labels["terminal_cashflow_priced"].sum() or 0)
    status_counts = {
        str(row["label_status"]): int(row["len"])
        for row in labels.group_by("label_status").len().iter_rows(named=True)
    }
    expected_audit_counts = {
        "priced_terminal_rows": priced_rows,
        "overnight_exit_rows": status_counts.get("overnight_exit", 0),
        "expiry_settlement_rows": status_counts.get("expiry_settlement", 0),
        "expiry_settlement_unpriced_rows": status_counts.get(
            "expiry_settlement_unpriced", 0
        ),
        "roll_substitution_forbidden_rows": status_counts.get(
            "roll_substitution_forbidden", 0
        ),
        "unresolved_rows": labels.height - priced_rows,
    }
    for column, expected in expected_audit_counts.items():
        if int(audit.item(0, column)) != expected:
            raise ValueError(f"overnight audit {column} mismatch: {partition}")
    if int(audit.item(0, "position_policy_rows")) != int(
        audit.item(0, "strict_carry_policy_rows")
    ) + int(audit.item(0, "excluded_noncarry_or_unknown_rows")):
        raise ValueError(f"overnight audit policy population mismatch: {partition}")


def _validate_combined_labels(labels: pl.DataFrame) -> None:
    _require(labels, _LABEL_REQUIRED, "combined overnight labels")
    if labels.select("carry_label_id").n_unique() != labels.height:
        raise ValueError("combined overnight labels contain duplicate carry_label_id")
    if labels.select("exit_policy_trial_id").n_unique() != labels.height:
        raise ValueError("combined overnight labels contain duplicate exit_policy_trial_id")
    if labels.filter(pl.col("physical_entry_dependency_id").is_null()).height:
        raise ValueError("overnight labels contain a null physical dependency id")
    non_strict = labels.filter(
        ~pl.col("source_branch_status").is_in(sorted(STRICT_CARRY_BRANCHES))
        | ~pl.col("exact_quote_code_required").fill_null(False)
        | pl.col("roll_attempted").fill_null(True)
        | pl.col("roll_policy_version").is_not_null()
        | pl.col("new_quote_code").is_not_null()
    )
    if non_strict.height:
        raise ValueError(
            "overnight report accepts strict carry labels with exact QuoteCode "
            "and no roll only"
        )
    _validate_output_invariants(labels, "overnight labels")
    if labels["terminal_cashflow_priced"].null_count():
        raise ValueError("overnight labels contain null terminal pricing status")
    invalid_outcome = labels.filter(
        pl.col("outcome_status").is_null()
        | ~pl.col("outcome_status").is_in(["known", "censored", "unknown"])
    )
    if invalid_outcome.height:
        raise ValueError("overnight labels contain an invalid outcome_status")
    invalid_weight = labels.filter(
        pl.col("policy_observation_weight").is_null()
        | ~pl.col("policy_observation_weight").is_finite()
        | (pl.col("policy_observation_weight") != 1.0)
        | pl.col("physical_entry_coverage_weight").is_null()
        | ~pl.col("physical_entry_coverage_weight").is_finite()
        | (pl.col("physical_entry_coverage_weight") <= 0)
    )
    if invalid_weight.height:
        raise ValueError("overnight labels contain invalid policy/dependency weights")
    weights = labels.group_by("physical_entry_dependency_id").agg(
        pl.len().cast(pl.Int64).alias("_actual_aliases"),
        pl.col("physical_entry_strict_carry_policy_alias_count")
        .n_unique()
        .alias("_declared_values"),
        pl.col("physical_entry_strict_carry_policy_alias_count")
        .first()
        .cast(pl.Int64)
        .alias("_declared_aliases"),
        pl.col("physical_entry_coverage_weight").sum().alias("_weight_sum"),
    )
    bad_weights = weights.filter(
        (pl.col("_declared_values") != 1)
        | (pl.col("_actual_aliases") != pl.col("_declared_aliases"))
        | ((pl.col("_weight_sum") - 1.0).abs() > 1e-9)
    )
    if bad_weights.height:
        raise ValueError("physical-entry alias counts/coverage weights are inconsistent")

    priced = pl.col("terminal_cashflow_priced")
    malformed_priced = labels.filter(
        priced
        & (
            (pl.col("outcome_status") != "known")
            | pl.col("gross_cycle_pnl_twd").is_null()
            | ~pl.col("gross_cycle_pnl_twd").is_finite()
            | pl.col("normalization_notional_twd").is_null()
            | ~pl.col("normalization_notional_twd").is_finite()
            | (pl.col("normalization_notional_twd") <= 0)
            | pl.col("filled_cashflow_before_cost_bp").is_null()
            | ~pl.col("filled_cashflow_before_cost_bp").is_finite()
        )
    )
    if malformed_priced.height:
        raise ValueError("priced overnight labels lack finite gross cashflow/notional")
    inconsistent_bp = labels.filter(priced).filter(
        (
            pl.col("filled_cashflow_before_cost_bp")
            - pl.col("gross_cycle_pnl_twd")
            / pl.col("normalization_notional_twd")
            * 10_000.0
        ).abs()
        > 1e-9
        * pl.max_horizontal(
            pl.lit(1.0), pl.col("filled_cashflow_before_cost_bp").abs()
        )
    )
    if inconsistent_bp.height:
        raise ValueError("overnight gross bp disagrees with gross cashflow/notional")
    invalid_holding = labels.filter(
        priced
        & (
            pl.col("holding_seconds").is_null()
            | ~pl.col("holding_seconds").is_finite()
            | (pl.col("holding_seconds") < 0)
        )
    )
    if invalid_holding.height:
        raise ValueError("priced overnight labels contain invalid holding_seconds")
    malformed_unpriced = labels.filter(
        ~priced
        & pl.any_horizontal(
            pl.col("gross_cycle_pnl_twd").is_not_null(),
            pl.col("normalization_notional_twd").is_not_null(),
            pl.col("filled_cashflow_before_cost_bp").is_not_null(),
        )
    )
    if malformed_unpriced.height:
        raise ValueError("unpriced overnight labels unexpectedly contain gross cashflow")


def _add_output_invariants(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.lit(None, dtype=pl.Float64).alias(column) for column in DEFAULT_COST_COLUMNS
    ]
    expressions.extend(
        [
            pl.lit(None, dtype=pl.String).alias("cost_profile_version"),
            pl.lit(False).alias("fees_tax_included"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("cross_q_rule_route_additive"),
        ]
    )
    return frame.with_columns(*expressions)


def _validate_output_invariants(frame: pl.DataFrame, source: str) -> None:
    for column in DEFAULT_COST_COLUMNS:
        if column not in frame.columns or frame[column].null_count() != frame.height:
            raise ValueError(f"{source} must keep {column} entirely null")
    if "cost_profile_version" not in frame.columns or frame[
        "cost_profile_version"
    ].null_count() != frame.height:
        raise ValueError(f"{source} must keep cost_profile_version entirely null")
    for column in (
        "fees_tax_included",
        "pathwise_ev_ready",
        "cross_q_rule_route_additive",
    ):
        if column not in frame.columns or frame[column].null_count() or frame[column].any():
            raise ValueError(f"{source} must keep {column}=false")


def _discover_markers(
    root: Path,
    requested_products: tuple[str, ...] | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json")):
        date = marker.parent.parent.name.removeprefix("Date=")
        value_code = marker.parent.name.removeprefix("ValueCode=")
        if requested_products is not None and value_code not in requested_products:
            continue
        payload = _read_json_object(marker)
        if payload.get("complete") is not True:
            continue
        if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
            raise ValueError(f"overnight marker identity mismatch: {marker}")
        records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": marker.parent,
                "marker": marker,
                "payload": payload,
            }
        )
    return records


def _validate_partition_marker(
    payload: Mapping[str, object],
    partition: Path,
    *,
    validate_hashes: bool,
) -> Mapping[str, object]:
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"invalid overnight config marker: {partition}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"overnight config hash mismatch: {partition}")

    runner_version = payload.get("runner_version")
    runner_sha = payload.get("runner_config_sha256")
    if not isinstance(runner_version, str) or not runner_version:
        raise ValueError(f"invalid overnight runner version: {partition}")
    if not isinstance(runner_sha, str):
        raise ValueError(f"invalid overnight runner config hash: {partition}")
    runner_config = config.get("runner")
    if not isinstance(runner_config, dict):
        runner_config = config.get("runner_config")
    if not isinstance(runner_config, dict):
        raise ValueError(f"overnight marker lacks runner config payload: {partition}")
    if _canonical_sha256(runner_config) != runner_sha:
        raise ValueError(f"overnight runner config hash mismatch: {partition}")

    declared_marker_sha = payload.get("marker_payload_sha256")
    if declared_marker_sha is not None:
        if not isinstance(declared_marker_sha, str):
            raise ValueError(f"invalid overnight marker payload hash: {partition}")
        unhashed = dict(payload)
        unhashed.pop("marker_payload_sha256", None)
        if _canonical_sha256(unhashed) != declared_marker_sha:
            raise ValueError(f"overnight marker payload hash mismatch: {partition}")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(INPUT_ARTIFACTS):
        raise ValueError(f"overnight artifact set mismatch: {partition}")
    for name in INPUT_ARTIFACTS:
        declared = artifacts[name]
        if not isinstance(declared, dict):
            raise ValueError(f"invalid artifact declaration: {partition}/{name}")
        _validate_one_artifact(
            partition / name,
            declared,
            validate_hashes=validate_hashes,
        )
    semantics = payload.get("fact_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"overnight marker lacks fact_semantics: {partition}")
    required_true = (
        "strict_carry_branches_only",
        "cancel_race_unknown_excluded",
        "exact_quote_code_no_roll",
        "first_missing_candidate_censors",
        "expiry_requires_final_settlement",
    )
    if any(semantics.get(name) is not True for name in required_true):
        raise ValueError(
            f"overnight marker lacks strict carry/exact-contract semantics: {partition}"
        )
    if semantics.get("cost_columns_null") is not True:
        raise ValueError(f"overnight marker does not freeze null costs: {partition}")
    if semantics.get("pathwise_ev_ready") is not False:
        raise ValueError(f"overnight marker does not freeze EV readiness: {partition}")
    return {
        "runner_config_sha256": runner_sha,
        "config_sha256": config_sha,
    }


def _validate_one_artifact(
    path: Path,
    declared: Mapping[str, object],
    *,
    validate_hashes: bool,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = declared.get("sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(f"artifact lacks sha256: {path}")
    if validate_hashes and _file_sha256(path) != expected_hash:
        raise ValueError(f"artifact hash mismatch: {path}")
    expected_bytes = declared.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ValueError(f"artifact lacks byte count: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"artifact byte count mismatch: {path}")
    schema = pl.read_parquet_schema(path)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    if int(declared.get("rows", -1)) != rows:
        raise ValueError(f"artifact row count mismatch: {path}")
    if int(declared.get("columns", -1)) != len(schema):
        raise ValueError(f"artifact column count mismatch: {path}")


def _validate_root_manifest(
    root: Path,
    selected: Sequence[Mapping[str, object]],
) -> bool:
    manifest_path = root / "overnight_carry_partition_manifest.parquet"
    if not manifest_path.is_file():
        return False
    manifest = pl.read_parquet(manifest_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    required = {
        "Date",
        "ValueCode",
        "runner_config_sha256",
        "config_sha256",
        "complete",
    }
    _require(manifest, required, "overnight root manifest")
    if manifest.select("Date", "ValueCode").n_unique() != manifest.height:
        raise ValueError("overnight root manifest has duplicate product-days")
    lookup = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in manifest.iter_rows(named=True)
    }
    for record in selected:
        key = (str(record["Date"]), str(record["ValueCode"]))
        row = lookup.get(key)
        payload = record["payload"]
        assert isinstance(payload, dict)
        if row is None or row["complete"] is not True:
            raise ValueError(f"overnight root manifest missing complete partition {key}")
        if (
            str(row["runner_config_sha256"])
            != str(payload.get("runner_config_sha256"))
            or str(row["config_sha256"]) != str(payload.get("config_sha256"))
        ):
            raise ValueError(f"overnight root manifest hash mismatch for {key}")
    return True


def _publish_report(report: OvernightCarryReport, destination: Path) -> None:
    destination = Path(destination)
    if destination.exists():
        _verify_existing_report(report, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    tables = {
        REPORT_ARTIFACTS[0]: report.combined_labels,
        REPORT_ARTIFACTS[1]: report.combined_audits,
        REPORT_ARTIFACTS[2]: report.physical_facts,
        REPORT_ARTIFACTS[3]: report.policy_summary,
        REPORT_ARTIFACTS[4]: report.status_summary,
    }
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name, frame in tables.items():
            _validate_output_invariants(frame, f"report artifact {name}")
            path = stage / name
            frame.write_parquet(path)
            artifacts[name] = {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        marker = dict(report.metadata)
        marker.update(
            {
                "complete": True,
                "report_version": REPORT_VERSION,
                "artifacts": artifacts,
            }
        )
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "report_complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _verify_existing_report(
    report: OvernightCarryReport, destination: Path
) -> None:
    """Make an identical report publication idempotent, but reject staleness."""

    marker_path = destination / "report_complete.json"
    if not marker_path.is_file():
        raise FileExistsError(
            f"existing report directory is incomplete: {destination}"
        )
    marker = _read_json_object(marker_path)
    declared_marker_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if not isinstance(declared_marker_sha, str) or _canonical_sha256(
        unhashed
    ) != declared_marker_sha:
        raise ValueError(f"existing report marker hash mismatch: {destination}")
    for key, expected in report.metadata.items():
        if marker.get(key) != expected:
            raise FileExistsError(
                "existing report was built from different partitions or "
                f"semantics: {destination}"
            )
    if marker.get("report_version") != REPORT_VERSION or marker.get("complete") is not True:
        raise ValueError(f"existing report marker is incompatible: {destination}")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REPORT_ARTIFACTS):
        raise ValueError(f"existing report artifact set mismatch: {destination}")
    for name in REPORT_ARTIFACTS:
        declaration = artifacts[name]
        if not isinstance(declaration, dict):
            raise ValueError(f"invalid existing report artifact: {destination / name}")
        _validate_one_artifact(
            destination / name,
            declaration,
            validate_hashes=True,
        )


def _same_schema(
    expected: Mapping[str, pl.DataType] | None,
    current: Mapping[str, pl.DataType],
    path: Path,
) -> Mapping[str, pl.DataType]:
    if expected is not None and dict(expected) != dict(current):
        raise ValueError(f"overnight artifact schema drift: {path}")
    return current


def _validate_session_request(sessions: int | None) -> None:
    if sessions is not None and (
        isinstance(sessions, bool) or not isinstance(sessions, int) or sessions <= 0
    ):
        raise ValueError("sessions must be a positive integer or None")


def _normalise_products(value_codes: Iterable[str] | None) -> tuple[str, ...] | None:
    if value_codes is None:
        return None
    values = tuple(dict.fromkeys(str(value) for value in value_codes))
    if not values:
        raise ValueError("value_codes must be nonempty when supplied")
    return values


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON marker: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON marker must contain an object: {path}")
    return payload


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
