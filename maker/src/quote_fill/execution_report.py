"""Support-balanced reports for partitioned raw maker execution replay.

The partition runner deliberately persists both policy aliases and physical
``raw_order_fact_id`` rows.  This module keeps those two populations separate:

* submitted policy aliases are an audit count only;
* entry, cancel, fill, hedge, and exit rates use one physical raw order per
  product-day / route / boundary quantile; and
* daily-balanced rates average product-day rates with explicit support counts,
  so a high-activity day cannot silently dominate the product table.

The same physical order may legitimately implement more than one quantile
when legal prices round together.  It therefore appears once inside each q
action, is flagged as cross-q shared, and must never be summed across q as if
the action rows were independent portfolio orders.

The report remains an independent-candidate execution diagnostic.  It does
not add shared visible-volume allocation, fees/tax, overnight realization, or
portfolio inventory, and consequently never labels itself pathwise-EV ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..quote_width.rolling import load_rolling_boundary_snapshots
from .hedge_study import FUTURE_ASK_ROUTE, SPOT_BID_ROUTE
from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)

EXPECTED_HEDGE_DELAY_NS = 50_000_000
DAILY_ACTION_KEY = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "boundary_quantile",
]
PRODUCT_ACTION_KEY = ["ValueCode", "route", "boundary_quantile"]
RAW_ACTION_KEY = [*DAILY_ACTION_KEY, "raw_order_fact_id"]
EXIT_RAW_ACTION_KEY = [*RAW_ACTION_KEY, "exit_rule_id"]
GEOMETRY_BASE_KEY = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]

_ENTRY_REQUIRED = {
    *DAILY_ACTION_KEY,
    "lookup_action_id",
    "raw_order_fact_id",
    "policy_generation_id",
    "any_fill",
    "full_fill",
    "partial_fill",
    "cancel_required",
}
_RAW_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
}
_HEDGE_REQUIRED = {
    "raw_order_fact_id",
    "status",
    "decision_book_age_ms",
    "signed_total_slippage_bp",
    "depth_shortfall",
}
_SUPPORT_REQUIRED = {*DAILY_ACTION_KEY, "lookup_action_id"}
_EXIT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "exit_rule_id",
    "branch_status",
    "same_day_exit",
    "overnight_carry",
    "gross_cycle_pnl_twd",
    "eod_liquidation_gross_pnl_twd",
}
_BOUNDARY_REQUIRED = {
    *GEOMETRY_BASE_KEY,
    "boundary_role",
    "upper_distance_bp",
    "lower_distance_bp",
    "adaptive_parameter_valid",
    "fut_ref_price",
    "spot_ref_price",
    "contract_size",
    "target_ref_future_ask_tick_bp",
    "target_ref_future_tick_bp",
    "target_ref_spot_bid_tick_bp",
    "source_asof_date",
    "parameter_version",
    "price_ladder_version",
    "future_one_dollar_tick_effective_date",
    "execution_safe_snapshot",
    "contains_target_day_outcome",
}

_EXPECTED_BOUNDARY_ROLES = {
    50: "rolling_latent_candidate",
    80: "rolling_latent_candidate",
    95: "tail_diagnostic",
}

_PHYSICAL_CONSTANT_COLUMNS = (
    "any_fill",
    "full_fill",
    "partial_fill",
    "cancel_required",
)
_EXIT_CONSTANT_COLUMNS = (
    "branch_status",
    "same_day_exit",
    "overnight_carry",
    "gross_cycle_pnl_twd",
    "eod_liquidation_gross_pnl_twd",
)
_BRANCHES = (
    "same_day_taker_exit",
    "carry_at_eod",
    "no_entry_fill",
    "partial_entry_unhedged",
    "entry_fill_unknown",
    "entry_hedge_unpriceable",
)


@dataclass(frozen=True)
class PartitionedExecutionInputs:
    """Validated facts loaded from complete product-day partitions."""

    action_facts: pl.DataFrame
    raw_order_facts: pl.DataFrame
    hedge_facts: pl.DataFrame
    daily_support: pl.DataFrame
    exit_facts: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class NarrowExecutionReport:
    """Detailed support and product-level execution checkpoint tables."""

    coverage: pl.DataFrame
    daily_entry: pl.DataFrame
    product_entry: pl.DataFrame
    daily_exit: pl.DataFrame
    product_exit: pl.DataFrame
    daily_geometry: pl.DataFrame
    product_geometry: pl.DataFrame
    lookup_checkpoint: pl.DataFrame
    metadata: Mapping[str, object]


def build_narrow_execution_report(
    action_facts: pl.DataFrame,
    raw_order_facts: pl.DataFrame,
    hedge_facts: pl.DataFrame,
    daily_support: pl.DataFrame,
    exit_facts: pl.DataFrame | None = None,
    boundary_snapshot: pl.DataFrame | None = None,
    *,
    coverage: pl.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> NarrowExecutionReport:
    """Build physical-order and product-day-balanced execution tables.

    ``daily_support`` is the zero-preserving grid written by the execution
    runner.  Every complete product-day must contain the same action axes.
    Detailed alias rows are collapsed only *within* q; a raw order shared by
    q80/q95 remains visible in both action rows and is explicitly flagged.
    """

    _require(action_facts, _ENTRY_REQUIRED, "execution action facts")
    _require(raw_order_facts, _RAW_REQUIRED, "raw order facts")
    _require(hedge_facts, _HEDGE_REQUIRED, "hedge facts")
    _require(daily_support, _SUPPORT_REQUIRED, "daily execution support")
    exit_facts = exit_facts if exit_facts is not None else pl.DataFrame()
    if not exit_facts.is_empty():
        _require(exit_facts, _EXIT_REQUIRED, "exit facts")
    if boundary_snapshot is not None:
        _require(boundary_snapshot, _BOUNDARY_REQUIRED, "boundary snapshot")

    report_metadata: dict[str, object] = dict(metadata or {})
    hedge_delay_ns = int(report_metadata.get("hedge_delay_ns", EXPECTED_HEDGE_DELAY_NS))
    if hedge_delay_ns != EXPECTED_HEDGE_DELAY_NS:
        raise ValueError(
            f"execution report requires a 50 ms hedge, got {hedge_delay_ns} ns"
        )

    support = _validate_and_normalise_support(daily_support)
    _validate_fact_universes(action_facts, raw_order_facts, hedge_facts, support)
    physical = _physical_entry_actions(action_facts, hedge_facts)
    daily_entry = _daily_entry_table(action_facts, physical, support).with_columns(
        pl.lit(hedge_delay_ns / 1_000_000.0).alias("hedge_delay_ms")
    )
    product_entry = _product_entry_table(daily_entry, physical).with_columns(
        pl.lit(hedge_delay_ns / 1_000_000.0).alias("hedge_delay_ms")
    )
    daily_exit, product_exit = _exit_tables(
        action_facts,
        physical,
        support,
        exit_facts,
    )
    daily_geometry, product_geometry = _geometry_tables(
        support,
        boundary_snapshot,
    )
    lookup_checkpoint = _lookup_checkpoint(product_entry, product_geometry)

    report_coverage = (
        coverage.sort(["Date", "ValueCode"])
        if coverage is not None
        else support.select("Date", "ValueCode", "QuoteCode")
        .unique()
        .with_columns(pl.lit(True).alias("partition_complete"))
        .sort(["Date", "ValueCode"])
    )
    report_metadata.setdefault("hedge_delay_ns", hedge_delay_ns)
    report_metadata.update(
        {
            "report_semantics": (
                "physical_raw_order_daily_balanced_with_optional_d1_geometry_v2"
            ),
            "physical_rate_unit": (
                "Date/ValueCode/QuoteCode/route/boundary_quantile/raw_order_fact_id"
            ),
            "policy_aliases_used_as_rate_denominator": False,
            "cross_quantile_rows_safe_to_sum": False,
            "joint_volume_allocated": False,
            "fees_tax_included": False,
            "overnight_realized_pnl_included": False,
            "pathwise_ev_ready": False,
            "entry_rows": daily_entry.height,
            "product_action_rows": product_entry.height,
            "exit_rows": daily_exit.height,
            "product_exit_rows": product_exit.height,
            "boundary_snapshot_included": boundary_snapshot is not None,
            "daily_geometry_rows": daily_geometry.height,
            "product_geometry_rows": product_geometry.height,
            "lookup_checkpoint_rows": lookup_checkpoint.height,
        }
    )
    return NarrowExecutionReport(
        coverage=report_coverage,
        daily_entry=daily_entry,
        product_entry=product_entry,
        daily_exit=daily_exit,
        product_exit=product_exit,
        daily_geometry=daily_geometry,
        product_geometry=product_geometry,
        lookup_checkpoint=lookup_checkpoint,
        metadata=report_metadata,
    )


def load_partitioned_execution_inputs(
    execution_root: Path,
    *,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = True,
    validate_hashes: bool = True,
    expected_hedge_delay_ns: int = EXPECTED_HEDGE_DELAY_NS,
) -> PartitionedExecutionInputs:
    """Load the latest complete sessions from a partitioned replay root.

    Completion markers, row counts, artifact hashes, one frozen runner config,
    the requested number of dates, and the date-by-product grid are validated
    before any report is returned.  Set the two ``require_*`` flags false only
    for an explicitly labelled development checkpoint.
    """

    root = Path(execution_root)
    if sessions is not None and (
        isinstance(sessions, bool) or not isinstance(sessions, int) or sessions <= 0
    ):
        raise ValueError("sessions must be a positive integer or None")
    requested_products = (
        tuple(dict.fromkeys(str(value) for value in value_codes))
        if value_codes is not None
        else None
    )
    if requested_products is not None and not requested_products:
        raise ValueError("value_codes must be nonempty when supplied")

    marker_records: list[dict[str, object]] = []
    for marker_path in sorted(root.glob("Date=*/ValueCode=*/complete.json")):
        date = marker_path.parent.parent.name.removeprefix("Date=")
        value_code = marker_path.parent.name.removeprefix("ValueCode=")
        if requested_products is not None and value_code not in requested_products:
            continue
        payload = json.loads(marker_path.read_text())
        if payload.get("complete") is not True:
            continue
        if (
            str(payload.get("Date")) != date
            or str(payload.get("ValueCode")) != value_code
        ):
            raise ValueError(f"partition marker identity mismatch: {marker_path}")
        marker_records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": marker_path.parent,
                "marker": marker_path,
                "payload": payload,
            }
        )
    if not marker_records:
        raise FileNotFoundError(f"no complete execution partitions under {root}")

    available_dates = sorted({str(record["Date"]) for record in marker_records})
    if (
        sessions is not None
        and require_exact_sessions
        and len(available_dates) < sessions
    ):
        raise ValueError(
            f"requested {sessions} complete sessions, found {len(available_dates)}"
        )
    selected_dates = (
        available_dates[-sessions:] if sessions is not None else available_dates
    )
    selected = [
        record for record in marker_records if str(record["Date"]) in selected_dates
    ]
    products = list(
        requested_products
        if requested_products is not None
        else sorted({str(record["ValueCode"]) for record in selected})
    )
    selected_by_key = {
        (str(record["Date"]), str(record["ValueCode"])): record for record in selected
    }
    if len(selected_by_key) != len(selected):
        raise ValueError("duplicate complete marker for a product-day")

    coverage_records: list[dict[str, object]] = []
    for date in selected_dates:
        for value_code in products:
            record = selected_by_key.get((date, value_code))
            coverage_records.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "partition_complete": record is not None,
                    "partition": (
                        str(record["partition"]) if record is not None else None
                    ),
                }
            )
    coverage = pl.from_dicts(coverage_records, infer_schema_length=None)
    missing = coverage.filter(~pl.col("partition_complete"))
    if require_balanced_product_days and missing.height:
        examples = [
            f"{row['Date']}/{row['ValueCode']}"
            for row in missing.head(5).iter_rows(named=True)
        ]
        raise ValueError(
            "incomplete date-by-product grid; missing " + ", ".join(examples)
        )

    required_artifacts = (
        "execution_action_facts.parquet",
        "raw_order_facts.parquet",
        "hedge_facts.parquet",
        "execution_daily_facts.parquet",
        "exit_facts.parquet",
    )
    loaded: dict[str, list[pl.DataFrame]] = {name: [] for name in required_artifacts}
    config_hashes: set[str] = set()
    runner_versions: set[str] = set()
    hedge_delays: set[int] = set()
    for record in selected:
        payload = record["payload"]
        assert isinstance(payload, dict)
        config_hashes.add(str(payload.get("config_sha256")))
        runner_versions.add(str(payload.get("runner_version")))
        config = payload.get("config")
        if not isinstance(config, dict) or "hedge_delay_ns" not in config:
            raise ValueError(f"marker missing hedge delay: {record['marker']}")
        hedge_delays.add(int(config["hedge_delay_ns"]))
        artifact_meta = payload.get("artifacts")
        if not isinstance(artifact_meta, dict):
            raise TypeError(f"marker missing artifact metadata: {record['marker']}")
        partition = Path(record["partition"])
        for name in required_artifacts:
            declared = artifact_meta.get(name)
            path = partition / name
            if not isinstance(declared, dict) or not path.is_file():
                raise ValueError(f"complete marker missing {name}: {partition}")
            if validate_hashes and _sha256(path) != str(declared.get("sha256")):
                raise ValueError(f"artifact hash mismatch: {path}")
            frame = pl.read_parquet(path)
            if frame.height != int(declared.get("rows", -1)):
                raise ValueError(f"artifact row count mismatch: {path}")
            if frame.width != int(declared.get("columns", -1)):
                raise ValueError(f"artifact column count mismatch: {path}")
            loaded[name].append(frame)

    if len(config_hashes) != 1 or len(runner_versions) != 1 or len(hedge_delays) != 1:
        raise ValueError("selected partitions do not share one frozen runner config")
    hedge_delay_ns = next(iter(hedge_delays))
    if hedge_delay_ns != expected_hedge_delay_ns:
        raise ValueError(
            f"expected {expected_hedge_delay_ns} ns hedge delay, got {hedge_delay_ns}"
        )

    def combine(name: str) -> pl.DataFrame:
        frames = loaded[name]
        return (
            pl.concat(frames, how="diagonal_relaxed", rechunk=True)
            if frames
            else pl.DataFrame()
        )

    metadata: dict[str, object] = {
        "execution_root": str(root),
        "selected_session_count": len(selected_dates),
        "selected_product_count": len(products),
        "selected_product_day_count": len(selected),
        "selected_dates": selected_dates,
        "value_codes": products,
        "config_sha256": next(iter(config_hashes)),
        "runner_version": next(iter(runner_versions)),
        "hedge_delay_ns": hedge_delay_ns,
        "partition_hashes_validated": validate_hashes,
        "balanced_product_day_grid": missing.is_empty(),
    }
    return PartitionedExecutionInputs(
        action_facts=combine("execution_action_facts.parquet"),
        raw_order_facts=combine("raw_order_facts.parquet"),
        hedge_facts=combine("hedge_facts.parquet"),
        daily_support=combine("execution_daily_facts.parquet"),
        exit_facts=combine("exit_facts.parquet"),
        coverage=coverage,
        metadata=metadata,
    )


def run_narrow_execution_report(
    execution_root: Path,
    *,
    output_dir: Path | None = None,
    boundary_snapshot_path: Path | None = None,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = True,
    validate_hashes: bool = True,
) -> NarrowExecutionReport:
    """Load complete partitions, build all tables, and atomically write CSVs."""

    inputs = load_partitioned_execution_inputs(
        execution_root,
        sessions=sessions,
        value_codes=value_codes,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=require_balanced_product_days,
        validate_hashes=validate_hashes,
    )
    boundary_snapshot = (
        load_boundary_snapshot_for_report(
            boundary_snapshot_path,
            inputs.daily_support,
        )
        if boundary_snapshot_path is not None
        else None
    )
    metadata = dict(inputs.metadata)
    if boundary_snapshot_path is not None:
        metadata.update(
            {
                "boundary_snapshot_path": str(boundary_snapshot_path),
                "boundary_snapshot_sha256": _sha256(boundary_snapshot_path),
            }
        )
    report = build_narrow_execution_report(
        inputs.action_facts,
        inputs.raw_order_facts,
        inputs.hedge_facts,
        inputs.daily_support,
        inputs.exit_facts,
        boundary_snapshot,
        coverage=inputs.coverage,
        metadata=metadata,
    )
    count = int(inputs.metadata["selected_session_count"])
    destination = (
        Path(output_dir)
        if output_dir
        else Path(execution_root) / f"report_{count}_sessions"
    )
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "coverage.csv": report.coverage,
        "daily_entry.csv": report.daily_entry,
        "product_action_entry.csv": report.product_entry,
        "daily_exit.csv": report.daily_exit,
        "product_action_exit.csv": report.product_exit,
        "daily_geometry.csv": report.daily_geometry,
        "product_action_geometry.csv": report.product_geometry,
        "product_action_lookup_checkpoint.csv": report.lookup_checkpoint,
    }
    artifact_meta: dict[str, dict[str, object]] = {}
    for name, frame in tables.items():
        path = destination / name
        _atomic_write_csv(frame, path)
        artifact_meta[name] = {
            "rows": frame.height,
            "columns": frame.width,
            "sha256": _sha256(path),
        }
    metadata_payload = dict(report.metadata)
    metadata_payload["artifacts"] = artifact_meta
    _atomic_write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        destination / "report_complete.json",
    )
    return report


def load_boundary_snapshot_for_report(
    boundary_snapshot_path: Path,
    daily_support: pl.DataFrame,
) -> pl.DataFrame:
    """Read only selected product-days from a rolling boundary snapshot."""

    _require(daily_support, _SUPPORT_REQUIRED, "daily execution support")
    path = Path(boundary_snapshot_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    snapshot = load_rolling_boundary_snapshots(path)
    schema_names = set(snapshot.columns)
    missing = sorted(_BOUNDARY_REQUIRED - schema_names)
    if missing:
        raise ValueError(f"boundary snapshot missing columns: {missing}")
    dates = daily_support["Date"].cast(pl.String).unique().to_list()
    products = daily_support["ValueCode"].cast(pl.String).unique().to_list()
    quantiles = daily_support["boundary_quantile"].cast(pl.Int64).unique().to_list()
    return (
        snapshot.filter(
            pl.col("Date").cast(pl.String).is_in(dates)
            & pl.col("ValueCode").cast(pl.String).is_in(products)
            & pl.col("boundary_quantile").cast(pl.Int64).is_in(quantiles)
        )
        .select(sorted(_BOUNDARY_REQUIRED))
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("source_asof_date").cast(pl.String),
        )
    )


def _validate_and_normalise_support(frame: pl.DataFrame) -> pl.DataFrame:
    support = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("lookup_action_id").cast(pl.String),
    )
    if support.select(DAILY_ACTION_KEY).n_unique() != support.height:
        raise ValueError("daily support must be unique by product-day/route/q")
    action_id_conflicts = (
        support.group_by("boundary_quantile")
        .agg(pl.col("lookup_action_id").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
    )
    if action_id_conflicts.height:
        raise ValueError("one boundary quantile maps to multiple lookup actions")

    product_days = support.select("Date", "ValueCode", "QuoteCode").unique()
    action_axes = support.select(
        "route", "boundary_quantile", "lookup_action_id"
    ).unique()
    expected = product_days.join(action_axes, how="cross").select(
        *DAILY_ACTION_KEY, "lookup_action_id"
    )
    missing = expected.join(
        support.select(*DAILY_ACTION_KEY),
        on=DAILY_ACTION_KEY,
        how="anti",
    )
    if missing.height:
        raise ValueError("daily execution support omits zero-order action cells")
    return support.sort(DAILY_ACTION_KEY)


def _validate_fact_universes(
    action: pl.DataFrame,
    raw: pl.DataFrame,
    hedge: pl.DataFrame,
    support: pl.DataFrame,
) -> None:
    if action.select("policy_generation_id").n_unique() != action.height:
        raise ValueError("policy_generation_id must be unique")
    if raw.select("raw_order_fact_id").n_unique() != raw.height:
        raise ValueError("raw_order_fact_id must be unique in raw facts")
    if hedge.select("raw_order_fact_id").n_unique() != hedge.height:
        raise ValueError("hedge facts must be unique by raw_order_fact_id")
    action_raw = action.select("raw_order_fact_id").unique()
    persisted_raw = raw.select("raw_order_fact_id").unique()
    if action_raw.join(persisted_raw, on="raw_order_fact_id", how="anti").height:
        raise ValueError("execution alias references an absent raw order fact")
    if persisted_raw.join(action_raw, on="raw_order_fact_id", how="anti").height:
        raise ValueError("raw order fact has no policy alias")
    raw_identity_columns = ("Date", "ValueCode", "QuoteCode", "route")
    action_identity = action.group_by("raw_order_fact_id").agg(
        *[
            pl.col(column).n_unique().alias(f"_n_{column}")
            for column in raw_identity_columns
        ],
        *[
            pl.col(column).first().cast(pl.String).alias(f"_action_{column}")
            for column in raw_identity_columns
        ],
    )
    if action_identity.filter(
        pl.any_horizontal(
            *[pl.col(f"_n_{column}") != 1 for column in raw_identity_columns]
        )
    ).height:
        raise ValueError("raw order aliases disagree on physical identity")
    raw_identity = raw.select("raw_order_fact_id", *raw_identity_columns).join(
        action_identity,
        on="raw_order_fact_id",
        how="left",
        validate="1:1",
    )
    if raw_identity.filter(
        pl.any_horizontal(
            *[
                pl.col(column).cast(pl.String) != pl.col(f"_action_{column}")
                for column in raw_identity_columns
            ]
        )
    ).height:
        raise ValueError("raw order fact identity disagrees with its aliases")
    if (
        hedge.select("raw_order_fact_id")
        .join(persisted_raw, on="raw_order_fact_id", how="anti")
        .height
    ):
        raise ValueError("hedge fact references an absent raw order fact")
    action_cells = action.select(*DAILY_ACTION_KEY).unique()
    if action_cells.join(
        support.select(*DAILY_ACTION_KEY),
        on=DAILY_ACTION_KEY,
        how="anti",
    ).height:
        raise ValueError("execution action falls outside the daily support grid")


def _physical_entry_actions(
    action: pl.DataFrame,
    hedge: pl.DataFrame,
) -> pl.DataFrame:
    _assert_constant_within(
        action,
        RAW_ACTION_KEY,
        _PHYSICAL_CONSTANT_COLUMNS,
        "same-q aliases disagree on physical entry outcome",
    )
    physical = action.group_by(RAW_ACTION_KEY).agg(
        pl.len().alias("policy_aliases_for_raw_action"),
        *[
            pl.col(column).first().alias(column)
            for column in _PHYSICAL_CONSTANT_COLUMNS
        ],
    )
    shared = action.group_by(
        "Date", "ValueCode", "QuoteCode", "route", "raw_order_fact_id"
    ).agg(pl.col("boundary_quantile").n_unique().alias("quantile_alias_count"))
    hedge_selected = hedge.select(
        "raw_order_fact_id",
        pl.col("status").alias("hedge_status"),
        pl.col("decision_book_age_ms").alias("hedge_decision_book_age_ms"),
        pl.col("signed_total_slippage_bp").alias("hedge_total_slippage_bp"),
        pl.col("depth_shortfall").alias("hedge_depth_shortfall"),
    )
    return (
        physical.join(
            shared,
            on=["Date", "ValueCode", "QuoteCode", "route", "raw_order_fact_id"],
            how="left",
            validate="m:1",
        )
        .join(hedge_selected, on="raw_order_fact_id", how="left", validate="m:1")
        .with_columns(
            (pl.col("quantile_alias_count") > 1).alias("shared_across_quantiles")
        )
        .sort(RAW_ACTION_KEY)
    )


def _daily_entry_table(
    action: pl.DataFrame,
    physical: pl.DataFrame,
    support: pl.DataFrame,
) -> pl.DataFrame:
    alias_counts = action.group_by(DAILY_ACTION_KEY).agg(
        pl.len().alias("submitted_policy_aliases")
    )
    full = pl.col("full_fill") == True
    hedge_labeled = full & pl.col("hedge_status").is_not_null()
    hedge_executable = full & (pl.col("hedge_status") == "executable")
    age_observed = full & pl.col("hedge_decision_book_age_ms").is_not_null()
    physical_daily = physical.group_by(DAILY_ACTION_KEY).agg(
        pl.len().alias("physical_raw_order_facts"),
        pl.col("shared_across_quantiles").sum().alias("cross_q_shared_raw_order_facts"),
        pl.col("cancel_required")
        .fill_null(False)
        .sum()
        .alias("cancel_required_raw_orders"),
        full.fill_null(False).sum().alias("full_fill_raw_orders"),
        pl.col("partial_fill").fill_null(False).sum().alias("partial_fill_raw_orders"),
        pl.col("any_fill").is_null().sum().alias("unknown_fill_raw_orders"),
        hedge_labeled.fill_null(False).sum().alias("hedge_labeled_full_fills"),
        hedge_executable.fill_null(False).sum().alias("hedge_executable_full_fills"),
        age_observed.fill_null(False).sum().alias("hedge_book_age_observed_full_fills"),
        (age_observed & (pl.col("hedge_decision_book_age_ms") <= 100.0))
        .fill_null(False)
        .sum()
        .alias("hedge_book_fresh_le100ms_full_fills"),
        (age_observed & (pl.col("hedge_decision_book_age_ms") <= 1_000.0))
        .fill_null(False)
        .sum()
        .alias("hedge_book_fresh_le1000ms_full_fills"),
        (
            full
            & (
                (pl.col("hedge_status") == "insufficient_depth")
                | (pl.col("hedge_depth_shortfall").fill_null(0) > 0)
            )
        )
        .fill_null(False)
        .sum()
        .alias("hedge_depth_shortfall_full_fills"),
        pl.col("hedge_decision_book_age_ms")
        .filter(age_observed)
        .quantile(0.50, interpolation="nearest")
        .alias("hedge_book_age_ms_p50"),
        pl.col("hedge_decision_book_age_ms")
        .filter(age_observed)
        .quantile(0.95, interpolation="nearest")
        .alias("hedge_book_age_ms_p95"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.50, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p50"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.80, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p80"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.95, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p95"),
    )
    keep_support = [*DAILY_ACTION_KEY, "lookup_action_id"]
    for column in (
        "target_observations",
        "submitted_generations",
        "unique_raw_order_facts",
    ):
        if column in support.columns:
            keep_support.append(column)
    result = (
        support.select(*keep_support)
        .join(alias_counts, on=DAILY_ACTION_KEY, how="left", validate="1:1")
        .join(physical_daily, on=DAILY_ACTION_KEY, how="left", validate="1:1")
    )
    count_columns = [
        "submitted_policy_aliases",
        "physical_raw_order_facts",
        "cross_q_shared_raw_order_facts",
        "cancel_required_raw_orders",
        "full_fill_raw_orders",
        "partial_fill_raw_orders",
        "unknown_fill_raw_orders",
        "hedge_labeled_full_fills",
        "hedge_executable_full_fills",
        "hedge_book_age_observed_full_fills",
        "hedge_book_fresh_le100ms_full_fills",
        "hedge_book_fresh_le1000ms_full_fills",
        "hedge_depth_shortfall_full_fills",
    ]
    result = result.with_columns(
        *[pl.col(column).fill_null(0).cast(pl.Int64) for column in count_columns]
    )
    _validate_reported_counts(result)
    return result.with_columns(
        _safe_rate(
            "submitted_policy_aliases",
            "physical_raw_order_facts",
            "policy_aliases_per_physical_raw_order",
        ),
        _safe_rate(
            "cancel_required_raw_orders",
            "physical_raw_order_facts",
            "cancel_required_rate_physical",
        ),
        _safe_rate(
            "full_fill_raw_orders",
            "physical_raw_order_facts",
            "full_fill_rate_physical",
        ),
        _safe_rate(
            "partial_fill_raw_orders",
            "physical_raw_order_facts",
            "partial_fill_rate_physical",
        ),
        _safe_rate(
            "hedge_executable_full_fills",
            "full_fill_raw_orders",
            "hedge_executable_rate_given_full_fill",
        ),
        _safe_rate(
            "hedge_book_fresh_le100ms_full_fills",
            "hedge_book_age_observed_full_fills",
            "hedge_book_fresh_le100ms_rate",
        ),
        _safe_rate(
            "hedge_book_fresh_le1000ms_full_fills",
            "hedge_book_age_observed_full_fills",
            "hedge_book_fresh_le1000ms_rate",
        ),
        pl.lit(False).alias("aliases_are_independent_rate_samples"),
        pl.lit(False).alias("cross_q_rows_safe_to_sum"),
        pl.lit(False).alias("pathwise_ev_ready"),
    ).sort(DAILY_ACTION_KEY)


def _product_entry_table(
    daily: pl.DataFrame,
    physical: pl.DataFrame,
) -> pl.DataFrame:
    count_columns = (
        "submitted_policy_aliases",
        "physical_raw_order_facts",
        "cross_q_shared_raw_order_facts",
        "cancel_required_raw_orders",
        "full_fill_raw_orders",
        "partial_fill_raw_orders",
        "unknown_fill_raw_orders",
        "hedge_labeled_full_fills",
        "hedge_executable_full_fills",
        "hedge_book_age_observed_full_fills",
        "hedge_book_fresh_le100ms_full_fills",
        "hedge_book_fresh_le1000ms_full_fills",
        "hedge_depth_shortfall_full_fills",
    )
    summary = daily.group_by(PRODUCT_ACTION_KEY).agg(
        pl.col("Date").n_unique().alias("sessions_in_grid"),
        pl.col("QuoteCode").n_unique().alias("quote_codes_in_window"),
        pl.len().alias("product_days_in_grid"),
        (pl.col("physical_raw_order_facts") > 0)
        .sum()
        .alias("product_days_with_submissions"),
        (pl.col("full_fill_raw_orders") > 0).sum().alias("product_days_with_full_fill"),
        (pl.col("hedge_executable_full_fills") > 0)
        .sum()
        .alias("product_days_with_executable_hedge"),
        *[pl.col(column).sum().alias(column) for column in count_columns],
        pl.col("physical_raw_order_facts")
        .filter(pl.col("physical_raw_order_facts") > 0)
        .median()
        .alias("physical_raw_orders_per_active_day_p50"),
        pl.col("cancel_required_rate_physical")
        .drop_nulls()
        .len()
        .alias("daily_cancel_rate_support_product_days"),
        pl.col("cancel_required_rate_physical")
        .mean()
        .alias("daily_balanced_cancel_required_rate"),
        pl.col("full_fill_rate_physical")
        .drop_nulls()
        .len()
        .alias("daily_fill_rate_support_product_days"),
        pl.col("full_fill_rate_physical").mean().alias("daily_balanced_full_fill_rate"),
        pl.col("partial_fill_rate_physical")
        .mean()
        .alias("daily_balanced_partial_fill_rate"),
        pl.col("hedge_executable_rate_given_full_fill")
        .drop_nulls()
        .len()
        .alias("daily_hedge_rate_support_product_days"),
        pl.col("hedge_executable_rate_given_full_fill")
        .mean()
        .alias("daily_balanced_hedge_executable_rate_given_full_fill"),
        pl.col("hedge_book_fresh_le100ms_rate")
        .drop_nulls()
        .len()
        .alias("daily_freshness_rate_support_product_days"),
        pl.col("hedge_book_fresh_le100ms_rate")
        .mean()
        .alias("daily_balanced_hedge_book_fresh_le100ms_rate"),
        pl.col("hedge_book_fresh_le1000ms_rate")
        .mean()
        .alias("daily_balanced_hedge_book_fresh_le1000ms_rate"),
    )
    full = pl.col("full_fill") == True
    hedge_executable = full & (pl.col("hedge_status") == "executable")
    age_observed = full & pl.col("hedge_decision_book_age_ms").is_not_null()
    distributions = physical.group_by(PRODUCT_ACTION_KEY).agg(
        pl.col("hedge_decision_book_age_ms")
        .filter(age_observed)
        .quantile(0.50, interpolation="nearest")
        .alias("hedge_book_age_ms_p50"),
        pl.col("hedge_decision_book_age_ms")
        .filter(age_observed)
        .quantile(0.95, interpolation="nearest")
        .alias("hedge_book_age_ms_p95"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.50, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p50"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.80, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p80"),
        pl.col("hedge_total_slippage_bp")
        .filter(hedge_executable)
        .quantile(0.95, interpolation="nearest")
        .alias("hedge_total_slippage_bp_p95"),
    )
    return (
        summary.join(distributions, on=PRODUCT_ACTION_KEY, how="left", validate="1:1")
        .with_columns(
            _safe_rate(
                "submitted_policy_aliases",
                "physical_raw_order_facts",
                "policy_aliases_per_physical_raw_order",
            ),
            _safe_rate(
                "cancel_required_raw_orders",
                "physical_raw_order_facts",
                "pooled_cancel_required_rate_physical",
            ),
            _safe_rate(
                "full_fill_raw_orders",
                "physical_raw_order_facts",
                "pooled_full_fill_rate_physical",
            ),
            _safe_rate(
                "partial_fill_raw_orders",
                "physical_raw_order_facts",
                "pooled_partial_fill_rate_physical",
            ),
            _safe_rate(
                "hedge_executable_full_fills",
                "full_fill_raw_orders",
                "pooled_hedge_executable_rate_given_full_fill",
            ),
            _safe_rate(
                "hedge_book_fresh_le100ms_full_fills",
                "hedge_book_age_observed_full_fills",
                "pooled_hedge_book_fresh_le100ms_rate",
            ),
            _safe_rate(
                "hedge_book_fresh_le1000ms_full_fills",
                "hedge_book_age_observed_full_fills",
                "pooled_hedge_book_fresh_le1000ms_rate",
            ),
            pl.lit("physical_raw_order_within_q").alias("rate_sample_unit"),
            pl.lit(False).alias("cross_q_rows_safe_to_sum"),
            pl.lit(False).alias("joint_volume_allocated"),
            pl.lit(False).alias("pathwise_ev_ready"),
        )
        .sort(PRODUCT_ACTION_KEY)
    )


def _geometry_tables(
    support: pl.DataFrame,
    boundary_snapshot: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if boundary_snapshot is None:
        return pl.DataFrame(schema=_empty_daily_geometry_schema()), pl.DataFrame(
            schema=_empty_product_geometry_schema()
        )

    boundary = boundary_snapshot.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("boundary_role").cast(pl.String),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("parameter_version").cast(pl.String),
        pl.col("price_ladder_version").cast(pl.String),
        pl.col("future_one_dollar_tick_effective_date").cast(pl.String),
        pl.col("adaptive_parameter_valid").cast(pl.Boolean),
        pl.col("execution_safe_snapshot").cast(pl.Boolean),
        pl.col("contains_target_day_outcome").cast(pl.Boolean),
    )
    expected = support.select(*GEOMETRY_BASE_KEY).unique()
    wanted = expected.select("Date", "ValueCode", "boundary_quantile").unique()
    selected = boundary.join(
        wanted,
        on=["Date", "ValueCode", "boundary_quantile"],
        how="inner",
        validate="m:1",
    )
    no_contract_key = ["Date", "ValueCode", "boundary_quantile"]
    duplicate = selected.group_by(no_contract_key).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("boundary snapshot has duplicate target contract rows")
    resolved = expected.rename({"QuoteCode": "_expected_QuoteCode"}).join(
        selected,
        on=no_contract_key,
        how="left",
        validate="1:1",
    )
    if resolved.filter(pl.col("QuoteCode").is_null()).height:
        raise ValueError("boundary snapshot is missing a selected product-day/q")
    if resolved.filter(pl.col("QuoteCode") != pl.col("_expected_QuoteCode")).height:
        raise ValueError("boundary snapshot does not use the exact execution contract")
    selected = resolved.drop("_expected_QuoteCode")

    unsafe = selected.filter(
        (pl.col("execution_safe_snapshot") != True)
        | pl.col("execution_safe_snapshot").is_null()
        | (pl.col("contains_target_day_outcome") != False)
        | pl.col("contains_target_day_outcome").is_null()
    )
    if unsafe.height:
        raise ValueError("boundary snapshot is not execution-safe and D-1-only")
    stale_ladder = selected.filter(
        (pl.col("price_ladder_version") != PRICE_LADDER_VERSION)
        | pl.col("price_ladder_version").is_null()
        | (
            pl.col("future_one_dollar_tick_effective_date")
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        )
        | pl.col("future_one_dollar_tick_effective_date").is_null()
        | (
            (pl.col("target_ref_future_tick_bp")
             - pl.col("target_ref_future_ask_tick_bp"))
            .abs()
            > 1e-12
        )
    )
    if stale_ladder.height:
        raise ValueError("boundary snapshot price ladder lineage mismatch")
    malformed_source = selected.filter(
        pl.col("source_asof_date").is_null()
        | (~pl.col("source_asof_date").str.contains(r"^\d{8}$"))
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if malformed_source.height:
        raise ValueError("boundary source_asof_date must be strictly before Date")
    inconsistent_source = (
        selected.group_by("Date", "ValueCode", "QuoteCode")
        .agg(pl.col("source_asof_date").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
    )
    if inconsistent_source.height:
        raise ValueError("one product-day must share one boundary source_asof_date")

    positive_identity_columns = (
        "fut_ref_price",
        "spot_ref_price",
        "contract_size",
        "target_ref_future_tick_bp",
        "target_ref_future_ask_tick_bp",
        "target_ref_spot_bid_tick_bp",
    )
    invalid_identity = selected.filter(
        pl.any_horizontal(
            *[
                pl.col(column).is_null()
                | (~pl.col(column).cast(pl.Float64).is_finite())
                | (pl.col(column).cast(pl.Float64) <= 0)
                for column in positive_identity_columns
            ]
        )
    )
    if invalid_identity.height:
        raise ValueError("boundary snapshot has invalid contract/tick identity")
    invalid_version = selected.filter(
        pl.col("parameter_version").is_null()
        | (pl.col("parameter_version").str.len_chars() == 0)
    )
    if invalid_version.height:
        raise ValueError("boundary parameter_version must be nonempty")
    version_conflict = (
        selected.group_by("Date", "ValueCode", "QuoteCode")
        .agg(pl.col("parameter_version").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
    )
    if version_conflict.height:
        raise ValueError("one product-day must share one boundary parameter version")

    role_conflict = (
        selected.group_by("ValueCode", "boundary_quantile")
        .agg(pl.col("boundary_role").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
    )
    if role_conflict.height or selected["boundary_role"].null_count():
        raise ValueError("boundary role must be stable by product/q")
    role_rows = selected.select("boundary_quantile", "boundary_role").unique()
    for row in role_rows.iter_rows(named=True):
        expected_role = _EXPECTED_BOUNDARY_ROLES.get(int(row["boundary_quantile"]))
        if expected_role is not None and row["boundary_role"] != expected_role:
            raise ValueError(
                f"q{row['boundary_quantile']} must retain role {expected_role}"
            )

    geometry_valid = (
        (pl.col("adaptive_parameter_valid") == True)
        & pl.col("upper_distance_bp").is_not_null()
        & pl.col("upper_distance_bp").is_finite()
        & (pl.col("upper_distance_bp") > 0)
        & pl.col("lower_distance_bp").is_not_null()
        & pl.col("lower_distance_bp").is_finite()
        & (pl.col("lower_distance_bp") > 0)
    )
    selected = selected.with_columns(
        geometry_valid.fill_null(False).alias("geometry_valid"),
        (
            pl.col("target_ref_future_ask_tick_bp")
            / pl.col("target_ref_spot_bid_tick_bp")
        ).alias("future_to_spot_tick_bp_ratio"),
        (pl.col("upper_distance_bp") / pl.col("target_ref_future_ask_tick_bp")).alias(
            "upper_distance_future_ticks"
        ),
        (pl.col("lower_distance_bp") / pl.col("target_ref_future_ask_tick_bp")).alias(
            "lower_distance_future_ticks"
        ),
        (pl.col("upper_distance_bp") / pl.col("target_ref_spot_bid_tick_bp")).alias(
            "upper_distance_spot_ticks"
        ),
        (pl.col("lower_distance_bp") / pl.col("target_ref_spot_bid_tick_bp")).alias(
            "lower_distance_spot_ticks"
        ),
    )
    action_grid = support.select(*DAILY_ACTION_KEY).unique()
    daily = action_grid.join(
        selected,
        on=GEOMETRY_BASE_KEY,
        how="left",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("route") == FUTURE_ASK_ROUTE)
        .then(pl.col("target_ref_future_ask_tick_bp"))
        .when(pl.col("route") == SPOT_BID_ROUTE)
        .then(pl.col("target_ref_spot_bid_tick_bp"))
        .otherwise(None)
        .alias("maker_route_tick_bp"),
        pl.when(pl.col("route") == FUTURE_ASK_ROUTE)
        .then(pl.col("upper_distance_future_ticks"))
        .when(pl.col("route") == SPOT_BID_ROUTE)
        .then(pl.col("upper_distance_spot_ticks"))
        .otherwise(None)
        .alias("upper_distance_maker_ticks"),
        pl.when(pl.col("route") == FUTURE_ASK_ROUTE)
        .then(pl.col("lower_distance_future_ticks"))
        .when(pl.col("route") == SPOT_BID_ROUTE)
        .then(pl.col("lower_distance_spot_ticks"))
        .otherwise(None)
        .alias("lower_distance_maker_ticks"),
        (pl.col("boundary_role") == "tail_diagnostic").alias("tail_diagnostic_only"),
    )
    if daily.filter(pl.col("maker_route_tick_bp").is_null()).height:
        raise ValueError("geometry report encountered an unsupported execution route")

    metric_columns = (
        "upper_distance_bp",
        "lower_distance_bp",
        "target_ref_future_ask_tick_bp",
        "target_ref_spot_bid_tick_bp",
        "future_to_spot_tick_bp_ratio",
        "upper_distance_future_ticks",
        "lower_distance_future_ticks",
        "upper_distance_spot_ticks",
        "lower_distance_spot_ticks",
        "maker_route_tick_bp",
        "upper_distance_maker_ticks",
        "lower_distance_maker_ticks",
    )
    product_key = [*PRODUCT_ACTION_KEY, "boundary_role"]
    quantile_expressions: list[pl.Expr] = []
    for column in metric_columns:
        for quantile, suffix in ((0.50, "p50"), (0.80, "p80"), (0.95, "p95")):
            quantile_expressions.append(
                pl.col(column)
                .filter(pl.col("geometry_valid"))
                .quantile(quantile, interpolation="nearest")
                .alias(f"{column}_{suffix}")
            )
    product = (
        daily.group_by(product_key)
        .agg(
            pl.col("Date").n_unique().alias("geometry_sessions_in_grid"),
            pl.len().alias("geometry_product_days_in_grid"),
            pl.col("geometry_valid").sum().alias("geometry_valid_product_days"),
            (~pl.col("geometry_valid")).sum().alias("geometry_invalid_product_days"),
            pl.col("QuoteCode").n_unique().alias("geometry_quote_codes_in_window"),
            pl.col("parameter_version")
            .n_unique()
            .alias("geometry_parameter_versions_in_window"),
            pl.col("source_asof_date").min().alias("geometry_source_asof_min"),
            pl.col("source_asof_date").max().alias("geometry_source_asof_max"),
            *quantile_expressions,
        )
        .with_columns(
            _safe_rate(
                "geometry_valid_product_days",
                "geometry_product_days_in_grid",
                "geometry_valid_product_day_rate",
            ),
            (pl.col("boundary_role") == "tail_diagnostic").alias(
                "tail_diagnostic_only"
            ),
            (
                pl.col("geometry_valid_product_days")
                == pl.col("geometry_product_days_in_grid")
            ).alias("geometry_complete_for_window"),
            pl.lit(True).alias("execution_safe_snapshot_validated"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit(False).alias("pathwise_ev_ready"),
        )
    )
    return daily.sort(DAILY_ACTION_KEY), product.sort(PRODUCT_ACTION_KEY)


def _lookup_checkpoint(
    product_entry: pl.DataFrame,
    product_geometry: pl.DataFrame,
) -> pl.DataFrame:
    if product_geometry.is_empty():
        return pl.DataFrame(schema=_empty_lookup_checkpoint_schema())
    if (
        product_geometry.select(PRODUCT_ACTION_KEY).n_unique()
        != product_geometry.height
    ):
        raise ValueError("product geometry must be unique by product/route/q")
    result = product_entry.join(
        product_geometry,
        on=PRODUCT_ACTION_KEY,
        how="left",
        validate="1:1",
        suffix="_geometry",
    )
    if result.filter(pl.col("boundary_role").is_null()).height:
        raise ValueError("product entry action does not resolve to D-1 geometry")
    return result.with_columns(
        pl.lit("entry_execution_plus_d1_geometry").alias("lookup_checkpoint_layer"),
        pl.lit(False).alias("lookup_checkpoint_is_final_ev"),
        pl.lit(False).alias("pathwise_ev_ready"),
    ).sort(PRODUCT_ACTION_KEY)


def _exit_tables(
    action: pl.DataFrame,
    physical: pl.DataFrame,
    support: pl.DataFrame,
    exit_facts: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if exit_facts.is_empty():
        return pl.DataFrame(schema=_empty_daily_exit_schema()), pl.DataFrame(
            schema=_empty_product_exit_schema()
        )
    if (
        exit_facts.select("policy_generation_id", "exit_rule_id").n_unique()
        != exit_facts.height
    ):
        raise ValueError("exit facts must be unique by policy generation and rule")
    metadata = action.select(
        "policy_generation_id",
        *DAILY_ACTION_KEY,
        "raw_order_fact_id",
    ).rename(
        {
            column: f"_action_{column}"
            for column in [*DAILY_ACTION_KEY, "raw_order_fact_id"]
        }
    )
    joined = exit_facts.join(
        metadata,
        on="policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined.filter(pl.col("_action_Date").is_null()).height:
        raise ValueError("exit fact does not resolve to an execution alias")
    for column in [*DAILY_ACTION_KEY[:-1], "raw_order_fact_id"]:
        if joined.filter(
            pl.col(column).cast(pl.String)
            != pl.col(f"_action_{column}").cast(pl.String)
        ).height:
            raise ValueError(f"exit fact {column} disagrees with execution alias")
    joined = joined.with_columns(
        pl.col("_action_boundary_quantile").cast(pl.Int64).alias("boundary_quantile")
    )
    _assert_constant_within(
        joined,
        EXIT_RAW_ACTION_KEY,
        _EXIT_CONSTANT_COLUMNS,
        "same-q exit aliases disagree on physical branch",
    )
    physical_exit = joined.group_by(EXIT_RAW_ACTION_KEY).agg(
        pl.len().alias("exit_policy_aliases_for_raw_action"),
        *[pl.col(column).first().alias(column) for column in _EXIT_CONSTANT_COLUMNS],
    )
    invalid_branches = physical_exit.filter(~pl.col("branch_status").is_in(_BRANCHES))
    if invalid_branches.height:
        raise ValueError("exit facts contain an unknown branch status")

    rules = physical_exit.select("exit_rule_id").unique().sort("exit_rule_id")
    grid = support.select(*DAILY_ACTION_KEY).join(rules, how="cross")
    entry_denominator = physical.group_by(DAILY_ACTION_KEY).agg(
        pl.len().alias("physical_entry_raw_orders")
    )
    aliases = joined.group_by([*DAILY_ACTION_KEY, "exit_rule_id"]).agg(
        pl.len().alias("exit_policy_alias_facts")
    )
    branch_daily = physical_exit.group_by([*DAILY_ACTION_KEY, "exit_rule_id"]).agg(
        pl.len().alias("exit_rule_physical_raw_facts"),
        *[
            (pl.col("branch_status") == branch)
            .sum()
            .alias(f"branch_{branch}_raw_orders")
            for branch in _BRANCHES
        ],
        pl.col("gross_cycle_pnl_twd")
        .filter(pl.col("branch_status") == "same_day_taker_exit")
        .quantile(0.50, interpolation="nearest")
        .alias("same_day_gross_cycle_pnl_twd_p50"),
        pl.col("eod_liquidation_gross_pnl_twd")
        .filter(pl.col("branch_status") == "carry_at_eod")
        .quantile(0.50, interpolation="nearest")
        .alias("carry_eod_liquidation_gross_pnl_twd_p50"),
    )
    daily = (
        grid.join(entry_denominator, on=DAILY_ACTION_KEY, how="left", validate="m:1")
        .join(
            aliases, on=[*DAILY_ACTION_KEY, "exit_rule_id"], how="left", validate="1:1"
        )
        .join(
            branch_daily,
            on=[*DAILY_ACTION_KEY, "exit_rule_id"],
            how="left",
            validate="1:1",
        )
    )
    count_columns = [
        "physical_entry_raw_orders",
        "exit_policy_alias_facts",
        "exit_rule_physical_raw_facts",
        *[f"branch_{branch}_raw_orders" for branch in _BRANCHES],
    ]
    daily = (
        daily.with_columns(
            *[pl.col(column).fill_null(0).cast(pl.Int64) for column in count_columns]
        )
        .with_columns(
            (
                pl.col("branch_same_day_taker_exit_raw_orders")
                + pl.col("branch_carry_at_eod_raw_orders")
            ).alias("opened_and_hedged_raw_orders")
        )
        .with_columns(
            _safe_rate(
                "exit_rule_physical_raw_facts",
                "physical_entry_raw_orders",
                "exit_fact_coverage_rate",
            ),
            _safe_rate(
                "branch_same_day_taker_exit_raw_orders",
                "physical_entry_raw_orders",
                "same_day_exit_rate_per_physical_entry",
            ),
            _safe_rate(
                "branch_carry_at_eod_raw_orders",
                "physical_entry_raw_orders",
                "carry_rate_per_physical_entry",
            ),
            _safe_rate(
                "branch_same_day_taker_exit_raw_orders",
                "opened_and_hedged_raw_orders",
                "same_day_exit_rate_given_opened_and_hedged",
            ),
            _safe_rate(
                "branch_carry_at_eod_raw_orders",
                "opened_and_hedged_raw_orders",
                "carry_rate_given_opened_and_hedged",
            ),
            pl.lit(False).alias("cross_q_rows_safe_to_sum"),
            pl.lit(False).alias("pathwise_ev_ready"),
        )
        .sort([*DAILY_ACTION_KEY, "exit_rule_id"])
    )

    product_key = [*PRODUCT_ACTION_KEY, "exit_rule_id"]
    count_sum = [
        "physical_entry_raw_orders",
        "exit_policy_alias_facts",
        "exit_rule_physical_raw_facts",
        "opened_and_hedged_raw_orders",
        *[f"branch_{branch}_raw_orders" for branch in _BRANCHES],
    ]
    product = daily.group_by(product_key).agg(
        pl.col("Date").n_unique().alias("sessions_in_grid"),
        pl.len().alias("product_days_in_grid"),
        (pl.col("physical_entry_raw_orders") > 0)
        .sum()
        .alias("product_days_with_entry_submissions"),
        (pl.col("exit_rule_physical_raw_facts") > 0)
        .sum()
        .alias("product_days_with_exit_facts"),
        (pl.col("opened_and_hedged_raw_orders") > 0)
        .sum()
        .alias("product_days_with_opened_and_hedged"),
        *[pl.col(column).sum().alias(column) for column in count_sum],
        pl.col("exit_fact_coverage_rate")
        .drop_nulls()
        .len()
        .alias("daily_exit_coverage_support_product_days"),
        pl.col("exit_fact_coverage_rate")
        .mean()
        .alias("daily_balanced_exit_fact_coverage_rate"),
        pl.col("same_day_exit_rate_per_physical_entry")
        .drop_nulls()
        .len()
        .alias("daily_exit_rate_support_product_days"),
        pl.col("same_day_exit_rate_per_physical_entry")
        .mean()
        .alias("daily_balanced_same_day_exit_rate_per_physical_entry"),
        pl.col("same_day_exit_rate_given_opened_and_hedged")
        .drop_nulls()
        .len()
        .alias("daily_opened_exit_rate_support_product_days"),
        pl.col("same_day_exit_rate_given_opened_and_hedged")
        .mean()
        .alias("daily_balanced_same_day_exit_rate_given_opened_and_hedged"),
    )
    distributions = physical_exit.group_by(product_key).agg(
        pl.col("gross_cycle_pnl_twd")
        .filter(pl.col("branch_status") == "same_day_taker_exit")
        .quantile(0.50, interpolation="nearest")
        .alias("same_day_gross_cycle_pnl_twd_p50"),
        pl.col("gross_cycle_pnl_twd")
        .filter(pl.col("branch_status") == "same_day_taker_exit")
        .quantile(0.95, interpolation="nearest")
        .alias("same_day_gross_cycle_pnl_twd_p95"),
        pl.col("eod_liquidation_gross_pnl_twd")
        .filter(pl.col("branch_status") == "carry_at_eod")
        .quantile(0.50, interpolation="nearest")
        .alias("carry_eod_liquidation_gross_pnl_twd_p50"),
    )
    product = (
        product.join(distributions, on=product_key, how="left", validate="1:1")
        .with_columns(
            _safe_rate(
                "exit_rule_physical_raw_facts",
                "physical_entry_raw_orders",
                "pooled_exit_fact_coverage_rate",
            ),
            _safe_rate(
                "branch_same_day_taker_exit_raw_orders",
                "physical_entry_raw_orders",
                "pooled_same_day_exit_rate_per_physical_entry",
            ),
            _safe_rate(
                "branch_carry_at_eod_raw_orders",
                "physical_entry_raw_orders",
                "pooled_carry_rate_per_physical_entry",
            ),
            _safe_rate(
                "branch_same_day_taker_exit_raw_orders",
                "opened_and_hedged_raw_orders",
                "pooled_same_day_exit_rate_given_opened_and_hedged",
            ),
            pl.lit(False).alias("fees_tax_included"),
            pl.lit(False).alias("overnight_realized_pnl_included"),
            pl.lit(False).alias("cross_q_rows_safe_to_sum"),
            pl.lit(False).alias("pathwise_ev_ready"),
        )
        .sort(product_key)
    )
    return daily, product


def _validate_reported_counts(frame: pl.DataFrame) -> None:
    for reported, actual in (
        ("submitted_generations", "submitted_policy_aliases"),
        ("unique_raw_order_facts", "physical_raw_order_facts"),
    ):
        if (
            reported in frame.columns
            and frame.filter(
                pl.col(reported).fill_null(0).cast(pl.Int64) != pl.col(actual)
            ).height
        ):
            raise ValueError(
                f"daily support {reported} disagrees with detailed {actual}"
            )


def _assert_constant_within(
    frame: pl.DataFrame,
    key: Sequence[str],
    columns: Sequence[str],
    message: str,
) -> None:
    inconsistent = (
        frame.group_by(list(key))
        .agg(*[pl.col(column).n_unique().alias(column) for column in columns])
        .filter(pl.any_horizontal(*[pl.col(column) != 1 for column in columns]))
    )
    if inconsistent.height:
        raise ValueError(message)


def _safe_rate(numerator: str, denominator: str, output: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(output)
    )


def _empty_daily_exit_schema() -> Mapping[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "exit_rule_id": pl.String,
        "physical_entry_raw_orders": pl.Int64,
        "exit_rule_physical_raw_facts": pl.Int64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_daily_geometry_schema() -> Mapping[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "boundary_role": pl.String,
        "geometry_valid": pl.Boolean,
        "upper_distance_bp": pl.Float64,
        "lower_distance_bp": pl.Float64,
        "maker_route_tick_bp": pl.Float64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_product_geometry_schema() -> Mapping[str, pl.DataType]:
    return {
        "ValueCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "boundary_role": pl.String,
        "geometry_product_days_in_grid": pl.Int64,
        "geometry_valid_product_days": pl.Int64,
        "upper_distance_bp_p50": pl.Float64,
        "lower_distance_bp_p50": pl.Float64,
        "maker_route_tick_bp_p50": pl.Float64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_lookup_checkpoint_schema() -> Mapping[str, pl.DataType]:
    return {
        "ValueCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "boundary_role": pl.String,
        "lookup_checkpoint_layer": pl.String,
        "lookup_checkpoint_is_final_ev": pl.Boolean,
        "pathwise_ev_ready": pl.Boolean,
    }


def _empty_product_exit_schema() -> Mapping[str, pl.DataType]:
    return {
        "ValueCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "exit_rule_id": pl.String,
        "physical_entry_raw_orders": pl.Int64,
        "exit_rule_physical_raw_facts": pl.Int64,
        "pathwise_ev_ready": pl.Boolean,
    }


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(frame: pl.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_csv(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(text: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a physical-order, daily-balanced execution report"
    )
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--boundary-snapshot", type=Path)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--allow-incomplete-product-grid", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_narrow_execution_report(
        args.execution_root,
        output_dir=args.output_dir,
        boundary_snapshot_path=args.boundary_snapshot,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        require_balanced_product_days=not args.allow_incomplete_product_grid,
        validate_hashes=not args.skip_hash_validation,
    )
    print(report.product_entry)
    if not report.product_geometry.is_empty():
        print(report.lookup_checkpoint)
    if not report.product_exit.is_empty():
        print(report.product_exit)


if __name__ == "__main__":
    main()
