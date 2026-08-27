"""Resumable canonical runner for the 71-session, seven-policy S1 study.

The runner is date-major so one prepared physical market day is shared by all
seven counterfactual policies and then released.  Every policy/date partition
is atomically published, capacity identities are committed to an exact SQLite
registry, and only a compact live-capacity checkpoint crosses the day boundary.

This is the Spot-Bid approximate entry screen with the first normal exit route
integrated (Spot Ask exact printed-volume maker fill, then Future buy taker).
It is not the S5 exact-entry calibration and cannot be labelled deploy-ready.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

from ..common.paths import HFT_ROOT, MAKER_ROOT
from .capacity_ledger import (
    DEFAULT_GLOBAL_CAP_TWD,
    DEFAULT_PRODUCT_CAP_TWD,
    CapacityLedger,
    CapacityLedgerCompactCheckpoint,
    CapacityTransition,
    decode_capacity_ledger_compact_checkpoint,
    decode_capacity_transition,
    encode_capacity_identity_registry_receipt,
    encode_capacity_ledger_compact_checkpoint,
    encode_capacity_transition,
    replay_capacity_transitions,
)
from .policy_spec import (
    ANCHOR_MODEL_ID,
    ENTRY_CANDIDATE_ID,
    LOWER_CANDIDATE_ID,
    POLICY_IDS,
    POLICY_SPEC_VERSION,
    load_policy_spec_table,
    policy_spec_table_sha256,
)
from .s1_accounting import (
    AccountingFact,
    ExecutedLeg,
    ExpiryAccountingMark,
    TerminalRealizedAccounting,
    decode_accounting_fact,
    encode_accounting_fact,
)
from .s1_accounting_bridge import S1AccountingBridge, S1AccountingProduct
from .s1_bundle_artifacts import read_s1_bundle_partition, write_s1_bundle_partition
from .s1_capacity_identity_registry import S1CapacityIdentityRegistry
from .s1_cross_ledger_verifier import verify_s1_accounting_capacity_links
from .s1_daily_diagnostics import (
    build_s1_daily_diagnostics,
    validate_s1_daily_diagnostics,
)
from .s1_entry_day_runner import (
    PreparedS1EntryDay,
    S1EntryDayRun,
    S1EntryDaySummary,
    S1EntryRunnerPaths,
    prepare_s1_entry_day,
    run_s1_entry_policy,
)
from .s1_event_loop import (
    S1CarryContractBinding,
    S1CarryPosition,
    decode_s1_carry_contract_binding,
    decode_s1_carry_position,
    encode_s1_carry_contract_binding,
    encode_s1_carry_position,
)
from .s1_performance import (
    S1DailyReplaySummary,
    S1ScenarioPerformance,
    aggregate_s1_scenario_metrics,
    build_s1_daily_replay_summary_from_risk_events,
)
from .s1_ranking import S1ShortlistResult, rank_s1_shortlist
from .transaction_costs import TransactionCostProfile

PRODUCTION_RUNNER_VERSION: Final = "s1_spot_bid_71x7_partitioned_v2"
RUN_CONFIG_SCHEMA_VERSION: Final = "s1_spot_bid_run_config_v1"
FINAL_BUNDLE_SCHEMA_VERSION: Final = "s1_spot_bid_complete_v1"
ROUTE_ID: Final = "spot_bid_future_taker__spot_ask_future_taker_exit"
ENTRY_FILL_TRUTH: Final = "approximate"
EXIT_FILL_TRUTH: Final = "exact_indexed_print_volume"
DEVELOPMENT_END_DATE: Final = "20260813"
DEFAULT_OUTPUT_ROOT: Final = (
    MAKER_ROOT / "data" / "walkforward" / "s1_spot_bid_joint_20260827_v1"
)
DEFAULT_REPORT_PATH: Final = (
    MAKER_ROOT / "doc" / "quote_fill" / "POLICY_COMPARISON_SPOT_BID_20260827.md"
)
CAPACITY_REGISTRY_FILENAME: Final = "capacity_identity_registry.sqlite"
GENESIS_PARTITION_SHA256: Final = hashlib.sha256(
    b"s1-policy-date-partition-genesis-v1"
).hexdigest()
TAIPEI: Final = ZoneInfo("Asia/Taipei")

S1_DEVELOPMENT_DATES: Final = (
    "20260505",
    "20260506",
    "20260507",
    "20260508",
    "20260511",
    "20260512",
    "20260513",
    "20260514",
    "20260515",
    "20260518",
    "20260519",
    "20260520",
    "20260521",
    "20260522",
    "20260525",
    "20260526",
    "20260527",
    "20260528",
    "20260529",
    "20260601",
    "20260602",
    "20260603",
    "20260604",
    "20260605",
    "20260608",
    "20260609",
    "20260610",
    "20260611",
    "20260612",
    "20260615",
    "20260616",
    "20260617",
    "20260618",
    "20260622",
    "20260623",
    "20260624",
    "20260625",
    "20260626",
    "20260629",
    "20260630",
    "20260701",
    "20260702",
    "20260703",
    "20260706",
    "20260707",
    "20260708",
    "20260709",
    "20260713",
    "20260714",
    "20260715",
    "20260716",
    "20260717",
    "20260720",
    "20260721",
    "20260722",
    "20260723",
    "20260724",
    "20260727",
    "20260728",
    "20260729",
    "20260730",
    "20260731",
    "20260803",
    "20260804",
    "20260805",
    "20260806",
    "20260807",
    "20260810",
    "20260811",
    "20260812",
    "20260813",
)


class S1ProductionRunError(RuntimeError):
    """Canonical S1 bundle state or replay output is invalid."""


@dataclass(frozen=True, slots=True)
class S1ProductionConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    report_path: Path = DEFAULT_REPORT_PATH
    dates: tuple[str, ...] = S1_DEVELOPMENT_DATES
    paths: S1EntryRunnerPaths = field(default_factory=S1EntryRunnerPaths)
    source_commit: str = ""
    verify_completed_input_content: bool = False

    def __post_init__(self) -> None:
        if tuple(self.dates) != S1_DEVELOPMENT_DATES:
            raise S1ProductionRunError(
                "canonical S1 run requires the frozen 71-session date list"
            )
        if not isinstance(self.output_root, Path) or not isinstance(
            self.report_path, Path
        ):
            raise TypeError("output_root and report_path must be Paths")
        if not isinstance(self.paths, S1EntryRunnerPaths):
            raise TypeError("paths must be S1EntryRunnerPaths")
        if self.source_commit and not _is_git_sha(self.source_commit):
            raise S1ProductionRunError("source_commit must be a git SHA")
        if not isinstance(self.verify_completed_input_content, bool):
            raise TypeError("verify_completed_input_content must be boolean")


@dataclass(slots=True)
class _PolicyRuntimeState:
    accounting: S1AccountingBridge
    checkpoint: CapacityLedgerCompactCheckpoint | None = None
    carry: tuple[S1CarryPosition, ...] = ()
    carry_bindings: tuple[S1CarryContractBinding, ...] = ()
    daily_summaries: list[S1EntryDaySummary] | None = None
    daily_risk_summaries: list[S1DailyReplaySummary] | None = None
    daily_diagnostics: list[dict[str, object]] | None = None
    previous_partition_sha256: str = GENESIS_PARTITION_SHA256
    accounting_fact_count: int = 0

    def __post_init__(self) -> None:
        self.daily_summaries = []
        self.daily_risk_summaries = []
        self.daily_diagnostics = []


@dataclass(frozen=True, slots=True)
class S1ProductionResult:
    output_root: Path
    report_path: Path
    run_config_sha256: str
    complete_sha256: str
    performance: tuple[S1ScenarioPerformance, ...]
    shortlist: S1ShortlistResult
    resumed_partitions: int
    executed_partitions: int


def _semantic_run_config(
    config: S1ProductionConfig,
    *,
    source_commit: str,
) -> dict[str, object]:
    static_paths = (
        config.paths.mother_path,
        config.paths.entry_lookup_path,
        config.paths.convergence_path,
    )
    before = _file_stats(static_paths)
    policy_table = load_policy_spec_table(
        config.paths.mother_path,
        config.paths.entry_lookup_path,
        config.paths.convergence_path,
    )
    dates = tuple(policy_table.select("Date").unique().sort("Date")["Date"].to_list())
    if dates != S1_DEVELOPMENT_DATES:
        raise S1ProductionRunError("policy table dates differ from frozen S1 dates")
    policy_sha = policy_spec_table_sha256(policy_table)
    mother = pl.read_parquet(config.paths.mother_path)
    primary = mother.filter(pl.col("s1_primary").fill_null(False))
    if (
        primary.height != 15_638
        or primary.select("Date").n_unique() != 71
        or primary.select("ValueCode").n_unique() != 244
    ):
        raise S1ProductionRunError("S1 mother 15,638/71/244 contract drifted")
    cost = TransactionCostProfile()
    cost.validate()
    static_records = _build_file_records(
        (
            (config.paths.mother_path, "s1_mother"),
            (config.paths.entry_lookup_path, "entry_lookup"),
            (config.paths.convergence_path, "frozen_c0_lower"),
        )
    )
    if before != _file_stats(static_paths):
        raise S1ProductionRunError("static source changed while run config was built")
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "runner_version": PRODUCTION_RUNNER_VERSION,
        "source_commit": source_commit,
        "dates": list(S1_DEVELOPMENT_DATES),
        "policy_ids": list(POLICY_IDS),
        "route_id": ROUTE_ID,
        "entry_route": "Spot Bid maker -> Future sell taker",
        "normal_exit_route": "Spot Ask maker -> Future buy taker",
        "entry_fill_truth": ENTRY_FILL_TRUTH,
        "entry_fill_cursor_exact": False,
        "own_quantity_included": False,
        "partial_entry_fill_included": False,
        "exit_fill_truth": EXIT_FILL_TRUTH,
        "anchor_model_id": ANCHOR_MODEL_ID,
        "entry_boundary_id": ENTRY_CANDIDATE_ID,
        "q_lower_scheme": LOWER_CANDIDATE_ID,
        "q_lower_status": "development_default_pending_production_freeze",
        "policy_spec_version": POLICY_SPEC_VERSION,
        "policy_spec_sha256": policy_sha,
        "mother_primary_product_days": 15_638,
        "mother_sessions": 71,
        "mother_products": 244,
        "global_cap_twd": DEFAULT_GLOBAL_CAP_TWD,
        "product_cap_twd": DEFAULT_PRODUCT_CAP_TWD,
        "per_product_fraction": 0.5,
        "capacity_notional_semantics": "one_way_spot_equivalent",
        "spot_request_limit_per_rolling_second": 100,
        "future_request_limit_per_rolling_second": 5,
        "hedge_delay_ms": 50,
        "hedge_retry_window_ms": 5_000,
        "expiry_semantics": "paired_basis_zero_at_official_spot_close",
        "naked_unresolved_policy": "fail_closed_no_cross_day_carry",
        "cost_profile": asdict(cost),
        "static_inputs": static_records,
        "protected_forward_start_date": "20260814",
    }


def _build_accounting_catalog(
    config: S1ProductionConfig,
) -> tuple[S1AccountingProduct, ...]:
    mother_keys = (
        pl.scan_parquet(config.paths.mother_path)
        .filter(pl.col("s1_primary").fill_null(False))
        .select("Date", "ValueCode", "QuoteCode")
        .collect()
    )
    rows: list[pl.DataFrame] = []
    for date in S1_DEVELOPMENT_DATES:
        mapping_path = config.paths.mapping_path(date)
        if not mapping_path.is_file():
            raise FileNotFoundError(f"missing mapping input: {mapping_path}")
        keys = mother_keys.filter(pl.col("Date") == date)
        rows.append(
            pl.scan_parquet(mapping_path)
            .with_columns(pl.col("Date").cast(pl.String))
            .join(keys.lazy(), on=["Date", "ValueCode", "QuoteCode"], how="inner")
            .select(
                pl.col("ValueCode").cast(pl.String),
                pl.col("contract_size").cast(pl.Int64),
            )
            .unique()
            .collect()
        )
    catalog_frame = pl.concat(rows).unique().sort("ValueCode")
    conflicts = (
        catalog_frame.group_by("ValueCode")
        .agg(pl.col("contract_size").n_unique().alias("sizes"))
        .filter(pl.col("sizes") != 1)
    )
    if not conflicts.is_empty():
        raise S1ProductionRunError("contract size changes across S1 dates")
    products = tuple(
        S1AccountingProduct(
            product_id=str(row["ValueCode"]),
            value_code=str(row["ValueCode"]),
            contract_size_shares=int(row["contract_size"]),
        )
        for row in catalog_frame.iter_rows(named=True)
    )
    if len(products) != 244:
        raise S1ProductionRunError("accounting catalog must contain 244 products")
    return products


def _daily_source_records(
    config: S1ProductionConfig,
    date: str,
) -> list[dict[str, object]]:
    return _build_file_records(
        (
            (config.paths.causal_path(date), "causal_fair"),
            (config.paths.mapping_path(date), "mapping"),
            (config.paths.contracts_path(date), "contract_metadata"),
            (config.paths.spot_raw_path(date), "spot_raw"),
            (config.paths.future_raw_path(date), "future_raw"),
            (config.paths.makerfill_path(date), "makerfill"),
        )
    )


def _date_manifest_path(root: Path, date: str) -> Path:
    return root / "input_manifests" / f"Date={date}.json"


def _ensure_date_input_manifest(
    config: S1ProductionConfig,
    *,
    date: str,
    run_config_sha256: str,
) -> tuple[dict[str, object], str]:
    path = _date_manifest_path(config.output_root, date)
    if path.exists() or path.is_symlink():
        manifest = _read_canonical_json_object(path)
        expected_keys = {
            "schema_version",
            "date",
            "run_config_sha256",
            "input_records",
        }
        if set(manifest) != expected_keys:
            raise S1ProductionRunError("date input manifest schema drifted")
        if (
            manifest["schema_version"] != "s1_date_input_manifest_v1"
            or manifest["date"] != date
            or manifest["run_config_sha256"] != run_config_sha256
        ):
            raise S1ProductionRunError("date input manifest lineage drifted")
        records = _validated_file_records(manifest["input_records"])
        if config.verify_completed_input_content:
            _verify_file_records_stable(records)
        return manifest, _sha256_file(path)

    before = _file_stats(_daily_source_paths(config, date))
    records = _daily_source_records(config, date)
    after = _file_stats(_daily_source_paths(config, date))
    if before != after:
        raise S1ProductionRunError("daily source changed while it was hashed")
    manifest: dict[str, object] = {
        "schema_version": "s1_date_input_manifest_v1",
        "date": date,
        "run_config_sha256": run_config_sha256,
        "input_records": records,
    }
    _atomic_write_canonical_json(path, manifest)
    return manifest, _sha256_file(path)


def _daily_source_paths(
    config: S1ProductionConfig,
    date: str,
) -> tuple[Path, ...]:
    return (
        config.paths.causal_path(date),
        config.paths.mapping_path(date),
        config.paths.contracts_path(date),
        config.paths.spot_raw_path(date),
        config.paths.future_raw_path(date),
        config.paths.makerfill_path(date),
    )


def _partition_path(root: Path, date: str, policy_id: str) -> Path:
    return root / "partitions" / f"Date={date}" / f"policy={policy_id}"


def _partition_marker_sha256(path: Path) -> str:
    return _sha256_file(path / "complete.json")


def _lineage_record(
    *,
    date_input_manifest_sha256: str,
    previous_global_partition_sha256: str,
    state: _PolicyRuntimeState,
) -> dict[str, object]:
    receipt = (
        None
        if state.checkpoint is None
        else encode_capacity_identity_registry_receipt(
            state.checkpoint.identity_registry_receipt
        )
    )
    return {
        "date_input_manifest_sha256": date_input_manifest_sha256,
        "previous_global_partition_sha256": previous_global_partition_sha256,
        "previous_policy_partition_sha256": state.previous_partition_sha256,
        "previous_capacity_registry_receipt": receipt,
        "accounting_fact_start_index": state.accounting_fact_count,
        "carry_in_count": len(state.carry),
    }


def _summary_from_record(record: Mapping[str, object]) -> S1EntryDaySummary:
    expected = {field.name for field in fields(S1EntryDaySummary)}
    if not isinstance(record, Mapping) or set(record) != expected:
        raise S1ProductionRunError("daily summary schema mismatch")
    try:
        result = S1EntryDaySummary(**dict(record))
    except (TypeError, ValueError) as error:
        raise S1ProductionRunError("daily summary cannot be reconstructed") from error
    _validate_daily_summary_types(result)
    return result


def _risk_summary_record(summary: S1DailyReplaySummary) -> dict[str, object]:
    return {
        "date": summary.date,
        "policy_id": summary.scenario_id,
        "all_risk_hedge_priced_numerator": (summary.all_risk_hedge_priced_numerator),
        "all_risk_hedge_priced_denominator": (
            summary.all_risk_hedge_priced_denominator
        ),
    }


def _risk_summary_from_record(
    record: Mapping[str, object],
) -> S1DailyReplaySummary:
    expected = {
        "date",
        "policy_id",
        "all_risk_hedge_priced_numerator",
        "all_risk_hedge_priced_denominator",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise S1ProductionRunError("daily risk summary schema mismatch")
    if (
        type(record["all_risk_hedge_priced_numerator"]) is not int
        or type(record["all_risk_hedge_priced_denominator"]) is not int
    ):
        raise S1ProductionRunError("daily risk summary counts must be integers")
    return S1DailyReplaySummary(
        date=str(record["date"]),
        scenario_id=str(record["policy_id"]),
        all_risk_hedge_priced_numerator=int(record["all_risk_hedge_priced_numerator"]),
        all_risk_hedge_priced_denominator=int(
            record["all_risk_hedge_priced_denominator"]
        ),
    )


def _validate_daily_summary_types(summary: S1EntryDaySummary) -> None:
    for dataclass_field in fields(summary):
        value = getattr(summary, dataclass_field.name)
        annotation = str(dataclass_field.type)
        if "int" in annotation and "None" not in annotation and type(value) is not int:
            raise S1ProductionRunError(
                f"daily summary {dataclass_field.name} must be an integer"
            )
        if "bool" in annotation and type(value) is not bool:
            raise S1ProductionRunError(
                f"daily summary {dataclass_field.name} must be boolean"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise S1ProductionRunError(
                f"daily summary {dataclass_field.name} must be finite"
            )


def _validate_prepared_day(
    prepared: PreparedS1EntryDay,
    *,
    date: str,
    catalog: Mapping[str, S1AccountingProduct],
) -> None:
    if not isinstance(prepared, PreparedS1EntryDay) or prepared.date != date:
        raise S1ProductionRunError("prepare_day returned the wrong date/type")
    if prepared.policy_ids != POLICY_IDS:
        raise S1ProductionRunError("prepared day lacks the canonical seven policies")
    entry_product_ids = frozenset(prepared.entry_product_ids)
    for product in prepared.products:
        if product.product_id != product.value_code or product.future_contracts != 1:
            raise S1ProductionRunError("prepared product identity/contract drifted")
        if product.product_id not in entry_product_ids:
            continue
        try:
            accounting_product = catalog[product.product_id]
        except KeyError as error:
            raise S1ProductionRunError(
                "prepared entry product is outside accounting catalog"
            ) from error
        if (
            accounting_product.value_code != product.value_code
            or accounting_product.contract_size_shares != product.contract_size_shares
        ):
            raise S1ProductionRunError(
                "prepared entry product/accounting contract drifted"
            )


def _validate_opening_carry(
    state: _PolicyRuntimeState,
    prepared: PreparedS1EntryDay,
) -> None:
    if len(state.carry) != len(state.carry_bindings):
        raise S1ProductionRunError("carry and contract bindings diverged")
    products = {product.product_id: product for product in prepared.products}
    for position, binding in zip(state.carry, state.carry_bindings, strict=True):
        try:
            actual = S1CarryContractBinding.from_product(products[position.product_id])
        except KeyError as error:
            raise S1ProductionRunError(
                "opening carry product is absent from daily mapping"
            ) from error
        if actual != binding or position.quote_code != binding.quote_code:
            raise S1ProductionRunError("opening carry contract mapping changed")


def _carry_bindings(
    carry: Sequence[S1CarryPosition],
    prepared: PreparedS1EntryDay,
) -> tuple[S1CarryContractBinding, ...]:
    products = {product.product_id: product for product in prepared.products}
    result: list[S1CarryContractBinding] = []
    for position in carry:
        try:
            product = products[position.product_id]
        except KeyError as error:
            raise S1ProductionRunError(
                "carry_out references unknown product"
            ) from error
        result.append(S1CarryContractBinding.from_product(product))
    return tuple(result)


def _required_exit_only_bindings(
    states: Mapping[str, _PolicyRuntimeState],
    *,
    policy_ids: Sequence[str],
) -> tuple[S1CarryContractBinding, ...]:
    by_product: dict[str, S1CarryContractBinding] = {}
    for policy_id in policy_ids:
        try:
            bindings = states[policy_id].carry_bindings
        except KeyError as error:
            raise S1ProductionRunError(
                "required carry policy state is missing"
            ) from error
        for binding in bindings:
            existing = by_product.get(binding.product_id)
            if existing is not None and existing != binding:
                raise S1ProductionRunError(
                    "policies require conflicting exit-only contract bindings"
                )
            by_product[binding.product_id] = binding
    return tuple(by_product[product_id] for product_id in sorted(by_product))


def run_s1_production_bundle(
    config: S1ProductionConfig,
    *,
    max_new_partitions: int | None = None,
) -> S1ProductionResult | None:
    """Resume or execute the canonical 71 x 7 bundle.

    ``max_new_partitions`` is an operational checkpoint stop, not a research
    parameter.  A finite value returns ``None`` after atomically publishing
    that many new partitions; resuming uses the identical run fingerprint.
    """

    return _run_s1_production_bundle(
        config,
        max_new_partitions=max_new_partitions,
        verification_only=False,
    )


def verify_s1_production_bundle(
    config: S1ProductionConfig,
) -> S1ProductionResult:
    """Read and deeply verify one already-complete canonical bundle.

    This path never prepares a market day, executes a policy, creates a
    registry, or fills a missing artifact.  It first verifies the existing
    final marker and every hash it names, then replays all partition facts and
    capacity transitions through the normal final verifier.
    """

    if not isinstance(config, S1ProductionConfig):
        raise TypeError("config must be S1ProductionConfig")
    verified_config = replace(config, verify_completed_input_content=True)
    result = _run_s1_production_bundle(
        verified_config,
        max_new_partitions=None,
        verification_only=True,
    )
    if result is None:
        raise AssertionError("verification-only replay cannot stop early")
    return result


def _run_s1_production_bundle(
    config: S1ProductionConfig,
    *,
    max_new_partitions: int | None,
    verification_only: bool,
) -> S1ProductionResult | None:
    if not isinstance(config, S1ProductionConfig):
        raise TypeError("config must be S1ProductionConfig")
    if not isinstance(verification_only, bool):
        raise TypeError("verification_only must be boolean")
    if max_new_partitions is not None and (
        type(max_new_partitions) is not int or max_new_partitions <= 0
    ):
        raise ValueError("max_new_partitions must be a positive integer or None")
    if verification_only and max_new_partitions is not None:
        raise ValueError("verification-only replay cannot execute partitions")
    if verification_only:
        existing_run_config = _preflight_complete_bundle(config)
        source_commit = _recorded_source_commit(
            existing_run_config,
            expected=config.source_commit,
        )
    else:
        if (config.output_root / "complete.json").exists() or (
            config.output_root / "complete.json"
        ).is_symlink():
            raise S1ProductionRunError(
                "bundle already has a final marker; use the verify command"
            )
        source_commit = _git_source_commit(expected=config.source_commit)
    run_config = _semantic_run_config(config, source_commit=source_commit)
    run_config_path = config.output_root / "run_config.json"
    run_config_sha256 = _ensure_run_config(run_config_path, run_config)
    run_config_fingerprint = _canonical_sha256(run_config)
    if run_config_fingerprint != run_config_sha256:
        raise S1ProductionRunError("run config fingerprint/bytes hash diverged")
    accounting_products = _build_accounting_catalog(config)
    catalog = {product.product_id: product for product in accounting_products}
    if not verification_only:
        config.output_root.mkdir(parents=True, exist_ok=True)
    registry_path = config.output_root / CAPACITY_REGISTRY_FILENAME
    with S1CapacityIdentityRegistry(
        registry_path,
        readonly=verification_only,
    ) as registry:
        (
            states,
            previous_global_sha256,
            resumed_partitions,
            next_coordinate_index,
        ) = _load_resume_prefix(
            config,
            run_config_fingerprint=run_config_fingerprint,
            run_config_sha256=run_config_sha256,
            accounting_products=accounting_products,
            registry=registry,
        )
        coordinates = tuple(
            (date, policy_id)
            for date in S1_DEVELOPMENT_DATES
            for policy_id in POLICY_IDS
        )
        if verification_only and next_coordinate_index != len(coordinates):
            raise S1ProductionRunError(
                "verification requires all 497 canonical partitions"
            )
        executed = 0
        current_date: str | None = None
        prepared: PreparedS1EntryDay | None = None
        date_manifest_sha256 = ""
        for date, policy_id in coordinates[next_coordinate_index:]:
            if current_date != date:
                del prepared
                gc.collect()
                _manifest, date_manifest_sha256 = _ensure_date_input_manifest(
                    config,
                    date=date,
                    run_config_sha256=run_config_sha256,
                )
                manifest_records = _validated_file_records(_manifest["input_records"])
                _verify_file_records_stable(manifest_records)
                remaining_policy_ids = POLICY_IDS[POLICY_IDS.index(policy_id) :]
                required_exit_only_bindings = _required_exit_only_bindings(
                    states,
                    policy_ids=remaining_policy_ids,
                )
                prepared = prepare_s1_entry_day(
                    date,
                    paths=config.paths,
                    required_exit_only_bindings=required_exit_only_bindings,
                )
                _verify_file_records_stable(manifest_records)
                _validate_prepared_day(prepared, date=date, catalog=catalog)
                current_date = date
            assert prepared is not None
            state = states[policy_id]
            _validate_opening_carry(state, prepared)
            lineage = _lineage_record(
                date_input_manifest_sha256=date_manifest_sha256,
                previous_global_partition_sha256=previous_global_sha256,
                state=state,
            )
            previous_global_sha256 = _execute_policy_date_partition(
                config=config,
                prepared=prepared,
                policy_id=policy_id,
                state=state,
                lineage=lineage,
                run_config_fingerprint=run_config_fingerprint,
                run_config_sha256=run_config_sha256,
                registry=registry,
                policy_runner=run_s1_entry_policy,
            )
            executed += 1
            _print_progress(
                date=date,
                policy_id=policy_id,
                completed=resumed_partitions + executed,
                total=len(coordinates),
            )
            if max_new_partitions is not None and executed >= max_new_partitions:
                return None
        del prepared
        gc.collect()
        registry_receipts = registry.verify()

    result = _finalize_production_bundle(
        config=config,
        run_config=run_config,
        run_config_sha256=run_config_sha256,
        states=states,
        registry_receipts=registry_receipts,
        resumed_partitions=resumed_partitions,
        executed_partitions=executed,
    )
    return result


def _load_resume_prefix(
    config: S1ProductionConfig,
    *,
    run_config_fingerprint: str,
    run_config_sha256: str,
    accounting_products: tuple[S1AccountingProduct, ...],
    registry: S1CapacityIdentityRegistry,
) -> tuple[dict[str, _PolicyRuntimeState], str, int, int]:
    facts_by_policy: dict[str, list[AccountingFact]] = {
        policy_id: [] for policy_id in POLICY_IDS
    }
    states = {
        policy_id: _PolicyRuntimeState(
            accounting=S1AccountingBridge(
                default_date=S1_DEVELOPMENT_DATES[0],
                scenario_id=policy_id,
                products=accounting_products,
                execution_date_resolver=_taipei_execution_date,
            )
        )
        for policy_id in POLICY_IDS
    }
    previous_global = GENESIS_PARTITION_SHA256
    coordinates = tuple(
        (date, policy_id) for date in S1_DEVELOPMENT_DATES for policy_id in POLICY_IDS
    )
    resumed = 0
    manifest_sha_by_date: dict[str, str] = {}
    for index, (date, policy_id) in enumerate(coordinates):
        partition = _partition_path(config.output_root, date, policy_id)
        if not partition.exists() and not partition.is_symlink():
            for later_date, later_policy in coordinates[index + 1 :]:
                later = _partition_path(config.output_root, later_date, later_policy)
                if later.exists() or later.is_symlink():
                    raise S1ProductionRunError(
                        "bundle partitions are not one canonical date-major prefix"
                    )
            break
        manifest_sha = manifest_sha_by_date.get(date)
        if manifest_sha is None:
            _manifest, manifest_sha = _ensure_date_input_manifest(
                config,
                date=date,
                run_config_sha256=run_config_sha256,
            )
            manifest_sha_by_date[date] = manifest_sha
        state = states[policy_id]
        lineage = _lineage_record(
            date_input_manifest_sha256=manifest_sha,
            previous_global_partition_sha256=previous_global,
            state=state,
        )
        records = read_s1_bundle_partition(
            partition,
            expected_run_config_fingerprint=run_config_fingerprint,
            expected_run_config_sha256=run_config_sha256,
            expected_date=date,
            expected_policy_id=policy_id,
            expected_lineage=lineage,
        )
        facts_delta = tuple(
            decode_accounting_fact(record) for record in records.accounting_fact_records
        )
        transitions = tuple(
            decode_capacity_transition(record)
            for record in records.capacity_transition_records
        )
        checkpoint = decode_capacity_ledger_compact_checkpoint(
            records.compact_checkpoint_record
        )
        _verify_resumed_capacity_partition(
            prior=state.checkpoint,
            transitions=transitions,
            checkpoint=checkpoint,
        )
        verify_s1_accounting_capacity_links(facts_delta, transitions)
        summary = _summary_from_record(records.daily_summary)
        risk_summary = _risk_summary_from_record(records.daily_risk_summary)
        diagnostics = validate_s1_daily_diagnostics(records.daily_diagnostics)
        if summary.date != date or summary.policy_id != policy_id:
            raise S1ProductionRunError("resumed daily summary identity drifted")
        if risk_summary.date != date or risk_summary.scenario_id != policy_id:
            raise S1ProductionRunError("resumed risk summary identity drifted")
        carries = tuple(
            decode_s1_carry_position(record) for record in records.carry_records
        )
        bindings = tuple(
            decode_s1_carry_contract_binding(record)
            for record in records.carry_binding_records
        )
        if len(carries) != len(bindings):
            raise S1ProductionRunError("resumed carry/binding count differs")
        facts_by_policy[policy_id].extend(facts_delta)
        expected_count = state.accounting_fact_count + len(facts_delta)
        if any(
            fact.sequence != sequence
            for sequence, fact in enumerate(
                facts_delta, start=state.accounting_fact_count + 1
            )
        ):
            raise S1ProductionRunError("accounting fact sequence is not contiguous")
        state.accounting_fact_count = expected_count
        state.checkpoint = checkpoint
        state.carry = carries
        state.carry_bindings = bindings
        assert state.daily_summaries is not None
        assert state.daily_risk_summaries is not None
        assert state.daily_diagnostics is not None
        state.daily_summaries.append(summary)
        state.daily_risk_summaries.append(risk_summary)
        state.daily_diagnostics.append(diagnostics)
        state.previous_partition_sha256 = _partition_marker_sha256(partition)
        previous_global = state.previous_partition_sha256
        resumed += 1
    else:
        index = len(coordinates)

    for policy_id, state in states.items():
        state.accounting = S1AccountingBridge.from_facts(
            default_date=S1_DEVELOPMENT_DATES[0],
            scenario_id=policy_id,
            products=accounting_products,
            facts=facts_by_policy[policy_id],
            execution_date_resolver=_taipei_execution_date,
        )
        if state.accounting.fact_count != state.accounting_fact_count:
            raise S1ProductionRunError("resumed accounting fact count drifted")

    receipts = registry.verify()
    if not isinstance(receipts, dict):
        raise S1ProductionRunError("capacity registry verification shape drifted")
    for policy_id, state in states.items():
        artifact_receipt = (
            None
            if state.checkpoint is None
            else state.checkpoint.identity_registry_receipt
        )
        registry_receipt = receipts.get(policy_id)
        if registry_receipt == artifact_receipt:
            continue
        # A crash may commit exactly the next missing partition to SQLite before
        # its atomic artifact rename.  The idempotent replay path resolves it.
        if resumed < len(coordinates) and registry_receipt is not None:
            next_date, next_policy = coordinates[resumed]
            tip = registry.latest_tip(policy_id)
            previous_partition_date = (
                None if state.checkpoint is None else state.checkpoint.through_date
            )
            if (
                policy_id == next_policy
                and tip is not None
                and tip.partition_date == next_date
                and tip.previous_partition_date == previous_partition_date
                and tip.previous_receipt == artifact_receipt
                and tip.receipt == registry_receipt
            ):
                continue
        raise S1ProductionRunError(
            f"capacity registry/artifact lineage differs for {policy_id}"
        )
    return states, previous_global, resumed, resumed


def _execute_policy_date_partition(
    *,
    config: S1ProductionConfig,
    prepared: PreparedS1EntryDay,
    policy_id: str,
    state: _PolicyRuntimeState,
    lineage: Mapping[str, object],
    run_config_fingerprint: str,
    run_config_sha256: str,
    registry: S1CapacityIdentityRegistry,
    policy_runner: Callable[..., S1EntryDayRun],
) -> str:
    ledger = (
        CapacityLedger()
        if state.checkpoint is None
        else CapacityLedger.from_compact_checkpoint(state.checkpoint)
    )
    if (
        ledger.global_cap_twd != DEFAULT_GLOBAL_CAP_TWD
        or ledger.product_cap_twd != DEFAULT_PRODUCT_CAP_TWD
    ):
        raise S1ProductionRunError("capacity ledger is not the frozen 20M/10M config")
    accounting_start = state.accounting.fact_count
    run = policy_runner(
        prepared,
        policy_id,
        normal_exit_enabled=True,
        accounting_adapter=state.accounting,
        capacity_ledger=ledger,
        carry_in=state.carry,
        entry_enabled_product_ids=frozenset(prepared.entry_product_ids),
    )
    _validate_production_daily_run(
        run=run,
        prepared=prepared,
        policy_id=policy_id,
        opening_carry=state.carry,
        ledger=ledger,
    )
    facts_delta = state.accounting.facts_since(accounting_start)
    transitions = ledger.transitions
    verify_s1_accounting_capacity_links(facts_delta, transitions)
    risk_summary = build_s1_daily_replay_summary_from_risk_events(
        run.result.risk_events,
        date=prepared.date,
        scenario_id=policy_id,
    )
    diagnostics = build_s1_daily_diagnostics(run, ledger)
    prior_receipt = (
        None if state.checkpoint is None else state.checkpoint.identity_registry_receipt
    )
    receipt = registry.commit_partition(
        policy_id,
        prepared.date,
        transitions,
        prior_receipt,
        verify_full_history_before_commit=False,
    )
    checkpoint = ledger.to_compact_checkpoint(
        prepared.date,
        identity_registry_receipt=receipt,
    )
    carry = tuple(run.result.carry_out)
    bindings = _carry_bindings(carry, prepared)
    partition = _partition_path(config.output_root, prepared.date, policy_id)
    write_s1_bundle_partition(
        partition,
        run_config_fingerprint=run_config_fingerprint,
        run_config_sha256=run_config_sha256,
        date=prepared.date,
        policy_id=policy_id,
        lineage=lineage,
        daily_summary=run.summary.as_dict(),
        daily_risk_summary=_risk_summary_record(risk_summary),
        daily_diagnostics=diagnostics,
        accounting_fact_records=(encode_accounting_fact(fact) for fact in facts_delta),
        capacity_transition_records=(
            encode_capacity_transition(transition) for transition in transitions
        ),
        compact_checkpoint_record=encode_capacity_ledger_compact_checkpoint(checkpoint),
        carry_records=tuple(encode_s1_carry_position(value) for value in carry),
        carry_binding_records=tuple(
            encode_s1_carry_contract_binding(value) for value in bindings
        ),
    )
    marker_sha = _partition_marker_sha256(partition)
    state.checkpoint = checkpoint
    state.carry = carry
    state.carry_bindings = bindings
    state.accounting_fact_count = state.accounting.fact_count
    state.previous_partition_sha256 = marker_sha
    assert state.daily_summaries is not None
    assert state.daily_risk_summaries is not None
    assert state.daily_diagnostics is not None
    state.daily_summaries.append(run.summary)
    state.daily_risk_summaries.append(risk_summary)
    state.daily_diagnostics.append(diagnostics)
    del facts_delta, transitions, run, ledger
    gc.collect()
    return marker_sha


def _validate_production_daily_run(
    *,
    run: S1EntryDayRun,
    prepared: PreparedS1EntryDay,
    policy_id: str,
    opening_carry: Sequence[S1CarryPosition],
    ledger: CapacityLedger,
) -> None:
    if not isinstance(run, S1EntryDayRun):
        raise TypeError("policy runner must return S1EntryDayRun")
    result = run.result
    if run.summary.date != prepared.date or run.summary.policy_id != policy_id:
        raise S1ProductionRunError("daily run identity drifted")
    if not result.normal_exit_enabled:
        raise S1ProductionRunError("canonical S1 replay requires normal exit")
    if result.carry_in != tuple(opening_carry):
        raise S1ProductionRunError("daily carry_in differs from portfolio state")
    opening_products = frozenset(value.product_id for value in opening_carry)
    if not opening_products.issubset(result.entry_disabled_product_ids):
        raise S1ProductionRunError("opening carry product was not exit-only")
    if any(order.product_id in opening_products for order in result.orders):
        raise S1ProductionRunError("opening carry product reopened during session")
    if result.capacity_transitions != ledger.transitions:
        raise S1ProductionRunError("daily run/ledger transition deltas differ")
    ledger.verify()
    naked = tuple(
        position
        for position in result.positions
        if position.state
        in (
            "entry_hedge_timeout_unresolved",
            "exit_rollback_failed_unresolved",
        )
    )
    if naked:
        identities = tuple(
            (position.position_id, position.product_id, position.state)
            for position in naked
        )
        raise S1ProductionRunError(
            "naked unresolved state requires an explicit next-day resolution "
            f"policy before S1 can continue: {identities}"
        )
    carry_ids = {position.position_id for position in result.carry_out}
    paired_ids = {
        position.position_id
        for position in result.positions
        if position.state == "paired_open"
    }
    if carry_ids != paired_ids:
        raise S1ProductionRunError("carry_out is not exactly the paired-open state")
    carry_capacity_ids = {position.capacity_id for position in result.carry_out}
    carry_notional = sum(
        ledger.account_balances(capacity_id).total_committed_notional_twd
        for capacity_id in carry_capacity_ids
    )
    if carry_notional != ledger.global_balances.total_committed_notional_twd:
        raise S1ProductionRunError(
            "live committed capacity is not exactly paired carry"
        )
    if any(
        getattr(ledger.global_balances, bucket) != 0
        for bucket in (
            "working_unfilled",
            "entry_partial",
            "hedge_pending",
            "exit_in_progress",
        )
    ):
        raise S1ProductionRunError("session ended with non-paired capacity bucket")


def _verify_resumed_capacity_partition(
    *,
    prior: CapacityLedgerCompactCheckpoint | None,
    transitions: Sequence[CapacityTransition],
    checkpoint: CapacityLedgerCompactCheckpoint,
) -> None:
    replay = replay_capacity_transitions(
        transitions,
        compact_checkpoint=prior,
    )
    live_account_balances = {
        capacity_id: balances
        for capacity_id, balances in replay.account_balances.items()
        if balances.total_committed_notional_twd != 0
    }
    live_account_products = {
        capacity_id: replay.account_products[capacity_id]
        for capacity_id in live_account_balances
    }
    live_product_balances = {
        product_id: balances
        for product_id, balances in replay.product_balances.items()
        if balances.total_committed_notional_twd != 0
    }
    comparisons = {
        "account_balances": (live_account_balances, checkpoint.account_balances),
        "account_products": (live_account_products, checkpoint.account_products),
        "product_balances": (live_product_balances, checkpoint.product_balances),
        "global_balances": (replay.global_balances, checkpoint.global_balances),
        "last_sequence": (replay.last_sequence, checkpoint.transition_sequence_offset),
        "transition_chain_sha256": (
            replay.transition_chain_sha256,
            checkpoint.transition_chain_sha256,
        ),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            raise S1ProductionRunError(
                f"resumed capacity partition/checkpoint {name} differs"
            )
    replay_cursor = (
        None
        if replay.last_timestamp_ns is None
        else (
            replay.last_timestamp_ns,
            replay.last_event_sequence,
            replay.last_row_index,
        )
    )
    if replay_cursor != checkpoint.last_cursor:
        raise S1ProductionRunError(
            "resumed capacity partition/checkpoint cursor differs"
        )
    receipt = checkpoint.identity_registry_receipt
    if receipt.transition_count != checkpoint.transition_sequence_offset:
        raise S1ProductionRunError("checkpoint registry transition count differs")


def _finalize_production_bundle(
    *,
    config: S1ProductionConfig,
    run_config: Mapping[str, object],
    run_config_sha256: str,
    states: Mapping[str, _PolicyRuntimeState],
    registry_receipts: object,
    resumed_partitions: int,
    executed_partitions: int,
) -> S1ProductionResult:
    if not isinstance(registry_receipts, dict) or set(registry_receipts) != set(
        POLICY_IDS
    ):
        raise S1ProductionRunError("final capacity registry policy set differs")
    performances: list[S1ScenarioPerformance] = []
    diagnostics_by_policy: dict[str, dict[str, object]] = {}
    entry_month_by_policy: dict[str, list[dict[str, object]]] = {}
    for policy_id in POLICY_IDS:
        state = states[policy_id]
        if state.checkpoint is None:
            raise S1ProductionRunError("final policy lacks a capacity checkpoint")
        if registry_receipts[policy_id] != (state.checkpoint.identity_registry_receipt):
            raise S1ProductionRunError("final registry/checkpoint receipt differs")
        accounting = state.accounting.verify()
        facts = accounting.facts
        assert state.daily_summaries is not None
        assert state.daily_risk_summaries is not None
        assert state.daily_diagnostics is not None
        if tuple(summary.date for summary in state.daily_summaries) != (
            S1_DEVELOPMENT_DATES
        ):
            raise S1ProductionRunError("final daily summary dates are incomplete")
        performance = aggregate_s1_scenario_metrics(
            facts,
            S1_DEVELOPMENT_DATES,
            scenario_id=policy_id,
            daily_replay_summaries=state.daily_risk_summaries,
            cursor_date_resolver=_taipei_execution_date,
        )
        diagnostics = _aggregate_daily_diagnostics(
            state.daily_diagnostics,
            state.daily_summaries,
        )
        if (
            performance.entry_fill_exact != 0
            or performance.entry_fill_approximate != performance.entry_positions
        ):
            raise S1ProductionRunError(
                "Spot-Bid production screen contains non-approximate entry truth"
            )
        if diagnostics["actual_active_entry_fills"] != performance.entry_positions:
            raise S1ProductionRunError(
                "diagnostic entry fills differ from accounting entry positions"
            )
        conserved_positions = (
            performance.same_day_exit_maker_flat
            + performance.cross_day_exit_maker_flat
            + performance.expiry_marks
            + performance.entry_rollbacks
            + performance.other_executable_terminals
            + performance.open_or_unresolved
        )
        if conserved_positions != performance.entry_positions:
            raise S1ProductionRunError("entry terminal/open outcomes do not conserve")
        performances.append(performance)
        diagnostics_by_policy[policy_id] = diagnostics
        entry_month_by_policy[policy_id] = _entry_month_rows(facts)
    performance_tuple = tuple(performances)
    shortlist = rank_s1_shortlist(
        tuple(value.to_approx_screen_scenario_metrics() for value in performance_tuple)
    )
    result_record = {
        "schema_version": "s1_spot_bid_results_v1",
        "run_config_sha256": run_config_sha256,
        "entry_fill_truth": ENTRY_FILL_TRUTH,
        "capacity_attestation": {
            "global_cap_twd": DEFAULT_GLOBAL_CAP_TWD,
            "product_cap_twd": DEFAULT_PRODUCT_CAP_TWD,
            "all_capacity_partitions_verified": True,
            "exact_identity_registry_verified": True,
            "naked_unresolved_fail_closed": True,
        },
        "performance": [_performance_record(value) for value in performance_tuple],
        "entry_cohort_month": entry_month_by_policy,
        "diagnostics": diagnostics_by_policy,
        "shortlist": _shortlist_record(shortlist),
    }
    daily_record = {
        "schema_version": "s1_spot_bid_daily_v1",
        "rows": [
            summary.as_dict()
            for policy_id in POLICY_IDS
            for summary in states[policy_id].daily_summaries or ()
        ],
        "risk_rows": [
            _risk_summary_record(summary)
            for policy_id in POLICY_IDS
            for summary in states[policy_id].daily_risk_summaries or ()
        ],
    }
    results_path = config.output_root / "results.json"
    daily_path = config.output_root / "daily_metrics.json"
    _atomic_write_canonical_json(results_path, result_record, replace_exact=True)
    _atomic_write_canonical_json(daily_path, daily_record, replace_exact=True)
    report = _render_report(
        run_config=run_config,
        run_config_sha256=run_config_sha256,
        performance=performance_tuple,
        shortlist=shortlist,
        diagnostics=diagnostics_by_policy,
        entry_month=entry_month_by_policy,
    )
    _atomic_write_text(config.report_path, report, replace_exact=True)
    partition_records = [
        {
            "date": date,
            "policy_id": policy_id,
            "complete_sha256": _partition_marker_sha256(
                _partition_path(config.output_root, date, policy_id)
            ),
        }
        for date in S1_DEVELOPMENT_DATES
        for policy_id in POLICY_IDS
    ]
    input_records = [
        {
            "date": date,
            "manifest_sha256": _sha256_file(
                _date_manifest_path(config.output_root, date)
            ),
        }
        for date in S1_DEVELOPMENT_DATES
    ]
    complete = {
        "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
        "complete": True,
        "run_config_sha256": run_config_sha256,
        "results_sha256": _sha256_file(results_path),
        "daily_metrics_sha256": _sha256_file(daily_path),
        "capacity_identity_registry_sha256": _sha256_file(
            config.output_root / CAPACITY_REGISTRY_FILENAME
        ),
        "report_path": _portable_path(config.report_path),
        "report_sha256": _sha256_file(config.report_path),
        "partition_count": len(partition_records),
        "expected_partition_count": len(S1_DEVELOPMENT_DATES) * len(POLICY_IDS),
        "partitions": partition_records,
        "input_manifests": input_records,
    }
    complete["marker_payload_sha256"] = _canonical_sha256(complete)
    complete_path = config.output_root / "complete.json"
    _atomic_write_canonical_json(complete_path, complete, replace_exact=True)
    complete_sha = _sha256_file(complete_path)
    return S1ProductionResult(
        output_root=config.output_root,
        report_path=config.report_path,
        run_config_sha256=run_config_sha256,
        complete_sha256=complete_sha,
        performance=performance_tuple,
        shortlist=shortlist,
        resumed_partitions=resumed_partitions,
        executed_partitions=executed_partitions,
    )


def _performance_record(value: S1ScenarioPerformance) -> dict[str, object]:
    record = _json_safe(asdict(value))
    assert isinstance(record, dict)
    record.update(
        {
            "reporting_sessions": value.reporting_sessions,
            "approx_screen_completion_numerator_20m": (
                value.approx_screen_completion_numerator
            ),
            "approx_screen_completion_denominator_20m": (
                value.approx_screen_completion_denominator
            ),
            "approx_screen_completion_rate_20m": (
                None
                if value.approx_screen_completion_rate is None
                else str(value.approx_screen_completion_rate)
            ),
            "mean_daily_net_twd_20m": str(value.mean_daily_net_twd),
            "entry_fill_truth": ENTRY_FILL_TRUTH,
            "all_risk_executable_send_numerator": value.hedge_priced_numerator,
            "all_risk_created_denominator": value.hedge_priced_denominator,
            "all_risk_executable_send_coverage_source": (
                value.hedge_pricing_coverage_source
            ),
        }
    )
    return record


def _shortlist_record(value: S1ShortlistResult) -> dict[str, object]:
    return {
        "completion_champion": value.completion_champion.scenario_id,
        "net_champion": value.net_champion.scenario_id,
        "selected": [row.scenario_id for row in value.selected],
        "second_selection_source": value.second_selection_source,
        "pareto_frontier": [row.scenario_id for row in value.pareto_frontier],
        "completion_ranking": [row.scenario_id for row in value.completion_ranking],
        "net_ranking": [row.scenario_id for row in value.net_ranking],
    }


def _aggregate_daily_diagnostics(
    records: Sequence[Mapping[str, object]],
    summaries: Sequence[S1EntryDaySummary],
) -> dict[str, object]:
    if len(records) != len(S1_DEVELOPMENT_DATES) or len(summaries) != len(
        S1_DEVELOPMENT_DATES
    ):
        raise S1ProductionRunError("diagnostic aggregation requires 71 sessions")
    validated = tuple(validate_s1_daily_diagnostics(record) for record in records)
    policy_ids = {str(record["policy_id"]) for record in validated}
    if len(policy_ids) != 1:
        raise S1ProductionRunError("diagnostics mix policies")

    counter_names = (
        "target_rank_counts",
        "makerfill_outcome_counts",
        "candidate_terminal_counts",
        "order_terminal_counts",
        "admission_status_counts",
        "request_counts",
        "execution_role_counts",
        "exit_physical_fill_reason_counts",
    )
    counters: dict[str, Counter[str]] = {name: Counter() for name in counter_names}
    fill_latency: list[float] = []
    risk: dict[tuple[str, str], dict[str, object]] = {}
    for record in validated:
        for name in counter_names:
            counters[name].update(record[name])  # type: ignore[arg-type]
        fill_latency.extend(record["active_fill_latency_ms"])  # type: ignore[arg-type]
        for group in record["risk_groups"]:  # type: ignore[union-attr]
            key = (str(group["stage"]), str(group["risk_kind"]))
            aggregate = risk.setdefault(
                key,
                {
                    "stage": key[0],
                    "risk_kind": key[1],
                    "created": 0,
                    "actual_send": 0,
                    "timeout": 0,
                    "on_time": 0,
                    "delayed": 0,
                    "arrival_reference_available": 0,
                    "arrival_reference_denominator": 0,
                    "delay_ms": [],
                    "adverse_slippage_bp": [],
                    "initial_gate_reason_counts": Counter(),
                },
            )
            for name in (
                "created",
                "actual_send",
                "timeout",
                "on_time",
                "delayed",
                "arrival_reference_available",
                "arrival_reference_denominator",
            ):
                aggregate[name] = int(aggregate[name]) + int(group[name])
            aggregate["delay_ms"].extend(group["delay_ms"])  # type: ignore[union-attr]
            aggregate["adverse_slippage_bp"].extend(  # type: ignore[union-attr]
                group["adverse_slippage_bp"]
            )
            aggregate["initial_gate_reason_counts"].update(  # type: ignore[union-attr]
                group["initial_gate_reason_counts"]
            )

    risk_rows: list[dict[str, object]] = []
    for key in sorted(risk):
        aggregate = risk[key]
        delays = sorted(float(value) for value in aggregate.pop("delay_ms"))
        slips = sorted(float(value) for value in aggregate.pop("adverse_slippage_bp"))
        gate_counts = aggregate.pop("initial_gate_reason_counts")
        risk_rows.append(
            {
                **aggregate,
                "delay_ms_p50": _quantile(delays, 0.50),
                "delay_ms_p95": _quantile(delays, 0.95),
                "delay_ms_max": max(delays, default=None),
                "adverse_slippage_bp_p50": _quantile(slips, 0.50),
                "adverse_slippage_bp_p95": _quantile(slips, 0.95),
                "adverse_slippage_bp_max": max(slips, default=None),
                "adverse_slippage_sample_count": len(slips),
                "initial_gate_reason_counts": {
                    name: int(gate_counts[name]) for name in sorted(gate_counts)
                },
            }
        )
    fill_latency.sort()
    return {
        "policy_id": next(iter(policy_ids)),
        "reporting_sessions": len(validated),
        **{
            name: {key: int(counters[name][key]) for key in sorted(counters[name])}
            for name in counter_names
        },
        "active_fill_latency_ms_p50": _quantile(fill_latency, 0.50),
        "active_fill_latency_ms_p95": _quantile(fill_latency, 0.95),
        "active_fill_latency_ms_max": max(fill_latency, default=None),
        "risk_groups": risk_rows,
        "sent_entry_orders": sum(row.sent_entry_orders for row in summaries),
        "makerfill_supported_orders": sum(
            row.makerfill_supported_orders for row in summaries
        ),
        "actual_active_entry_fills": sum(
            row.actual_active_entry_fills for row in summaries
        ),
        "spot_requests_sent": sum(row.spot_requests_sent for row in summaries),
        "future_requests_sent": sum(row.future_requests_sent for row in summaries),
        "mean_daily_spot_requests_sent": sum(
            row.spot_requests_sent for row in summaries
        )
        / len(summaries),
        "mean_daily_future_requests_sent": sum(
            row.future_requests_sent for row in summaries
        )
        / len(summaries),
        "spot_rolling_request_peak": max(
            row.spot_rolling_request_peak for row in summaries
        ),
        "future_rolling_request_peak": max(
            row.future_rolling_request_peak for row in summaries
        ),
        "global_cap_peak_twd": max(row.global_cap_peak_twd for row in summaries),
        "product_cap_peak_twd": max(row.product_cap_peak_twd for row in summaries),
        "reservation_attempts": sum(row.admission_checks for row in summaries),
        "cap_blocked_attempts": sum(row.blocked_admission_checks for row in summaries),
        "suppressed_redundant_blocked_admission_probes": sum(
            row.suppressed_redundant_blocked_admission_probes for row in summaries
        ),
        "carry_notional_days_twd": sum(
            int(record["carry_out_notional_twd"]) for record in validated
        ),
        "final_carry_positions": int(validated[-1]["carry_out_positions"]),
        "final_carry_notional_twd": int(validated[-1]["carry_out_notional_twd"]),
        "naked_unresolved_positions_max": max(
            int(record["naked_unresolved_positions"]) for record in validated
        ),
        "naked_unresolved_notional_twd_max": max(
            int(record["naked_unresolved_notional_twd"]) for record in validated
        ),
        "global_committed_bucket_peaks_twd": {
            bucket: max(
                int(record["global_committed_peak_twd"][bucket])  # type: ignore[index]
                for record in validated
            )
            for bucket in (
                "working_unfilled",
                "entry_partial",
                "hedge_pending",
                "paired_open",
                "exit_in_progress",
                "total_committed_notional_twd",
            )
        },
    }


def _entry_month_rows(facts: Sequence[AccountingFact]) -> list[dict[str, object]]:
    entries = {
        fact.position_id: fact
        for fact in facts
        if isinstance(fact, ExecutedLeg) and fact.route_role == "entry_maker"
    }
    if len(entries) != sum(
        isinstance(fact, ExecutedLeg) and fact.route_role == "entry_maker"
        for fact in facts
    ):
        raise S1ProductionRunError("entry maker position IDs are duplicated")
    terminals = {
        fact.position_id: fact
        for fact in facts
        if isinstance(fact, (TerminalRealizedAccounting, ExpiryAccountingMark))
    }
    months = ("202605", "202606", "202607", "202608")
    result: list[dict[str, object]] = []
    for month in months:
        selected = tuple(
            (position_id, entry)
            for position_id, entry in entries.items()
            if entry.execution_date.startswith(month)
        )
        counts: Counter[str] = Counter()
        net = Decimal(0)
        for position_id, entry in selected:
            terminal = terminals.get(position_id)
            if terminal is None:
                counts["open_at_horizon"] += 1
                continue
            net += Decimal(str(terminal.realized_net_twd))
            if isinstance(terminal, ExpiryAccountingMark):
                counts["expiry"] += 1
            elif terminal.terminal_outcome == "exit_maker_flat":
                counts[
                    "same_day"
                    if terminal.terminal_date == entry.execution_date
                    else "cross_day"
                ] += 1
            elif terminal.terminal_outcome in (
                "entry_emergency_rollback_flat",
                "entry_partial_rollback_flat",
            ):
                counts["entry_rollback"] += 1
            else:
                counts["other_terminal"] += 1
        result.append(
            {
                "entry_month": month,
                "entry_positions": len(selected),
                "same_day": counts["same_day"],
                "cross_day": counts["cross_day"],
                "expiry": counts["expiry"],
                "entry_rollback": counts["entry_rollback"],
                "other_terminal": counts["other_terminal"],
                "open_at_horizon": counts["open_at_horizon"],
                "terminal_net_twd": str(net),
            }
        )
    return result


def _render_report(
    *,
    run_config: Mapping[str, object],
    run_config_sha256: str,
    performance: Sequence[S1ScenarioPerformance],
    shortlist: S1ShortlistResult,
    diagnostics: Mapping[str, Mapping[str, object]],
    entry_month: Mapping[str, Sequence[Mapping[str, object]]],
) -> str:
    lines = [
        "# S1 Spot Bid 七組 policy：20M 聯合事件回放",
        "",
        "日期：2026-08-27",
        "",
        "## 結論",
        "",
        (
            "本表是 71 個 development sessions、共同 15,638 product-days、"
            "global 20M／單檔 10M reservation cap 下的完整 chronological screen。"
            "Entry 使用 legacy makerFill approximate full-fill；normal exit 使用 Spot Ask "
            "printed-volume FIFO，再以 Future buy taker 完成。因此它可作策略研究 baseline，"
            "但不是 exact-entry 或 production exchange-level 部署證明。"
        ),
        "",
        (
            f"Completion champion：`{shortlist.completion_champion.scenario_id}`；"
            f"net champion：`{shortlist.net_champion.scenario_id}`；"
            "S2 shortlist："
            + ", ".join(f"`{row.scenario_id}`" for row in shortlist.selected)
            + "。"
        ),
        "",
        "## 七組主比較",
        "",
        (
            "| policy | approx entry fills | hedge success | same-day | "
            "approx completion 20M | cross-day | expiry | rollback | "
            "other terminal | open@horizon | terminal coverage | executable net TWD | "
            "expiry net TWD | total net TWD | mean/day TWD | all-risk sent/created |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ]
    for value in performance:
        lines.append(
            "| "
            + " | ".join(
                (
                    value.scenario_id,
                    str(value.entry_fill_approximate),
                    (
                        f"{value.entry_hedge_success_numerator}/"
                        f"{value.entry_hedge_success_denominator}"
                    ),
                    str(value.same_day_exit_maker_flat),
                    _percent(
                        value.approx_screen_completion_numerator,
                        value.approx_screen_completion_denominator,
                    ),
                    str(value.cross_day_exit_maker_flat),
                    str(value.expiry_marks),
                    str(value.entry_rollbacks),
                    str(value.other_executable_terminals),
                    str(value.open_or_unresolved),
                    _percent(
                        value.terminal_coverage_numerator,
                        value.terminal_coverage_denominator,
                    ),
                    _money(value.executable_terminal_net_twd),
                    _money(value.expiry_mark_net_twd),
                    _money(value.total_net_twd),
                    _money(value.mean_daily_net_twd),
                    _percent(
                        value.hedge_priced_numerator,
                        value.hedge_priced_denominator,
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "`open@horizon` 在本 run 已由 fail-closed invariant 證明全是 paired carry；"
                "naked unresolved 若出現會中止整個 run，不會被混入此欄。未平倉執行成本另列，"
                "不混入 terminal realized net。"
            ),
            (
                "`all-risk sent/created` 的分母是 entry/exit hedge 與 rollback 的唯一完整"
                " lifecycle；分子是取得合法 executable book 並實際送出的 lifecycle。"
            ),
            "",
            "## 未平倉已發生成本",
            "",
            "| policy | open positions | commission TWD | tax TWD | actual cost TWD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for value in performance:
        lines.append(
            f"| {value.scenario_id} | {value.open_or_unresolved} | "
            f"{_money(value.open_execution_actual_commission_twd)} | "
            f"{_money(value.open_execution_actual_tax_twd)} | "
            f"{_money(value.open_execution_actual_cost_twd)} |"
        )
    lines.extend(
        [
            "",
            "## 執行與容量診斷",
            "",
            (
                "| policy | sent orders | supported | fills | fill latency p50/p95 ms | "
                "spot req/day | future req/day | rolling peak S/F | cap peak G/P TWD | "
                "carry notional-days | final carry TWD | naked max TWD |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_id in POLICY_IDS:
        row = diagnostics[policy_id]
        lines.append(
            "| "
            + " | ".join(
                (
                    policy_id,
                    str(row["sent_entry_orders"]),
                    str(row["makerfill_supported_orders"]),
                    str(row["actual_active_entry_fills"]),
                    (
                        f"{_number(row['active_fill_latency_ms_p50'])}/"
                        f"{_number(row['active_fill_latency_ms_p95'])}"
                    ),
                    _number(row["mean_daily_spot_requests_sent"]),
                    _number(row["mean_daily_future_requests_sent"]),
                    (
                        f"{row['spot_rolling_request_peak']}/"
                        f"{row['future_rolling_request_peak']}"
                    ),
                    f"{row['global_cap_peak_twd']}/{row['product_cap_peak_twd']}",
                    str(row["carry_notional_days_twd"]),
                    str(row["final_carry_notional_twd"]),
                    str(row["naked_unresolved_notional_twd_max"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Entry cohort 月表",
            "",
            (
                "| policy | entry month | entries | same-day | cross-day | expiry | "
                "rollback | other terminal | open@horizon | terminal net TWD |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_id in POLICY_IDS:
        for row in entry_month[policy_id]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        policy_id,
                        str(row["entry_month"]),
                        str(row["entry_positions"]),
                        str(row["same_day"]),
                        str(row["cross_day"]),
                        str(row["expiry"]),
                        str(row["entry_rollback"]),
                        str(row["other_terminal"]),
                        str(row["open_at_horizon"]),
                        str(row["terminal_net_twd"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Cashflow calendar 月表",
            "",
            (
                "| policy | month | executable terminals | expiry marks | "
                "executable net TWD | expiry net TWD | total net TWD |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for value in performance:
        for row in value.monthly_net:
            lines.append(
                f"| {value.scenario_id} | {row.calendar_key} | "
                f"{row.executable_terminal_count} | {row.expiry_mark_count} | "
                f"{_money(row.executable_terminal_net_twd)} | "
                f"{_money(row.expiry_mark_net_twd)} | "
                f"{_money(row.total_net_twd)} |"
            )
    lines.extend(
        [
            "",
            "## 排名與可重現性",
            "",
            "- Completion ranking："
            + " → ".join(
                f"`{row.scenario_id}`" for row in shortlist.completion_ranking
            ),
            "- Net ranking："
            + " → ".join(f"`{row.scenario_id}`" for row in shortlist.net_ranking),
            "- Pareto frontier："
            + ", ".join(f"`{row.scenario_id}`" for row in shortlist.pareto_frontier),
            f"- Run config SHA-256：`{run_config_sha256}`。",
            f"- Source commit：`{run_config['source_commit']}`。",
            (
                "- Policy spec：`time_ewma_15s` + `Q2_trail20_date_equal`; q lower "
                "目前明標 `C0_center` development default，不能冒稱已凍結 production lower。"
            ),
            (
                "- 每個 policy/date partition 都有 deterministic gzip JSONL、compact capacity "
                "checkpoint、SQLite exact identity registry 與 atomic complete marker；最終 verifier "
                "重播 accounting、capacity 與 cross-ledger links。"
            ),
            "",
            "## 限制",
            "",
            (
                "- Spot Bid entry 看不到 own quantity、partial fill、cancel ACK 與 joint volume allocation；"
                "S5 exact calibration 前只能稱 approximate screen。"
            ),
            "- 目前未計 overnight financing、borrow、margin opportunity cost；不能把未建模成本當 0。",
            "- 本結果使用 development-selected lookup；2026-08-14 起 protected forward 未讀。",
            (
                "- C0 是 completion-oriented development default。若要部署 baseline，仍須凍結 lower "
                "與 unsupported 行為，並完成 exact-entry、forward、風控與實盤 shadow 驗證。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "null"
    return f"{100.0 * numerator / denominator:.4f}% ({numerator}/{denominator})"


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _number(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _preflight_complete_bundle(
    config: S1ProductionConfig,
) -> dict[str, object]:
    """Verify the final marker and every named artifact without writing."""

    root = config.output_root
    _require_real_directory(root, "bundle root")
    complete_path = root / "complete.json"
    complete = _read_canonical_json_object(complete_path)
    expected_keys = {
        "schema_version",
        "complete",
        "run_config_sha256",
        "results_sha256",
        "daily_metrics_sha256",
        "capacity_identity_registry_sha256",
        "report_path",
        "report_sha256",
        "partition_count",
        "expected_partition_count",
        "partitions",
        "input_manifests",
        "marker_payload_sha256",
    }
    if set(complete) != expected_keys:
        raise S1ProductionRunError("final marker schema drifted")
    if (
        complete["schema_version"] != FINAL_BUNDLE_SCHEMA_VERSION
        or complete["complete"] is not True
    ):
        raise S1ProductionRunError("final marker identity/status drifted")
    marker_payload_sha256 = complete["marker_payload_sha256"]
    if not isinstance(marker_payload_sha256, str) or not _is_sha256(
        marker_payload_sha256
    ):
        raise S1ProductionRunError("final marker payload SHA-256 is invalid")
    unsigned = dict(complete)
    del unsigned["marker_payload_sha256"]
    if _canonical_sha256(unsigned) != marker_payload_sha256:
        raise S1ProductionRunError("final marker payload hash differs")

    expected_partition_count = len(S1_DEVELOPMENT_DATES) * len(POLICY_IDS)
    if (
        type(complete["partition_count"]) is not int
        or type(complete["expected_partition_count"]) is not int
        or complete["partition_count"] != expected_partition_count
        or complete["expected_partition_count"] != expected_partition_count
    ):
        raise S1ProductionRunError("final marker partition count differs")

    partitions_root = root / "partitions"
    _require_real_directory(partitions_root, "partition root")
    expected_partitions: list[dict[str, object]] = []
    for date in S1_DEVELOPMENT_DATES:
        _require_real_directory(
            partitions_root / f"Date={date}",
            f"partition date {date}",
        )
        for policy_id in POLICY_IDS:
            partition = _partition_path(root, date, policy_id)
            _require_real_directory(
                partition,
                f"partition {date}/{policy_id}",
            )
            expected_partitions.append(
                {
                    "date": date,
                    "policy_id": policy_id,
                    "complete_sha256": _partition_marker_sha256(partition),
                }
            )
    if complete["partitions"] != expected_partitions:
        raise S1ProductionRunError("final marker partition list/hash differs")

    manifests_root = root / "input_manifests"
    _require_real_directory(manifests_root, "input manifest root")
    for date in S1_DEVELOPMENT_DATES:
        _require_real_file(
            _date_manifest_path(root, date),
            f"input manifest {date}",
        )
    expected_manifests = [
        {
            "date": date,
            "manifest_sha256": _sha256_file(_date_manifest_path(root, date)),
        }
        for date in S1_DEVELOPMENT_DATES
    ]
    if complete["input_manifests"] != expected_manifests:
        raise S1ProductionRunError("final marker input manifest list/hash differs")

    report_record = complete["report_path"]
    if not isinstance(report_record, dict) or set(report_record) != {
        "path_scope",
        "path",
    }:
        raise S1ProductionRunError("final marker report path is invalid")
    if report_record != _portable_path(config.report_path):
        raise S1ProductionRunError("requested report path differs from final marker")

    artifact_checks = (
        (root / "run_config.json", "run_config_sha256", "run config"),
        (root / "results.json", "results_sha256", "results"),
        (root / "daily_metrics.json", "daily_metrics_sha256", "daily metrics"),
        (
            root / CAPACITY_REGISTRY_FILENAME,
            "capacity_identity_registry_sha256",
            "capacity identity registry",
        ),
        (config.report_path, "report_sha256", "report"),
    )
    for path, hash_key, label in artifact_checks:
        _require_real_file(path, label)
        expected_sha256 = complete[hash_key]
        if not isinstance(expected_sha256, str) or not _is_sha256(expected_sha256):
            raise S1ProductionRunError(f"final marker {label} SHA-256 is invalid")
        if _sha256_file(path) != expected_sha256:
            raise S1ProductionRunError(f"final marker {label} hash differs")
    return _read_canonical_json_object(root / "run_config.json")


def _recorded_source_commit(
    run_config: Mapping[str, object],
    *,
    expected: str,
) -> str:
    source_commit = run_config.get("source_commit")
    if not isinstance(source_commit, str) or not _is_git_sha(source_commit):
        raise S1ProductionRunError("recorded source commit is invalid")
    if expected and expected != source_commit:
        raise S1ProductionRunError("expected source commit differs from bundle")
    return source_commit


def _require_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise S1ProductionRunError(f"{label} must be an existing real directory")


def _require_real_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise S1ProductionRunError(f"{label} must be an existing real file")


def _ensure_run_config(path: Path, record: Mapping[str, object]) -> str:
    payload = _canonical_json_bytes(record)
    expected_sha = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise S1ProductionRunError("existing run_config.json differs")
        return expected_sha
    _atomic_write_bytes(path, payload)
    return expected_sha


def _atomic_write_canonical_json(
    path: Path,
    value: object,
    *,
    replace_exact: bool = False,
) -> None:
    _atomic_write_bytes(
        path,
        _canonical_json_bytes(value),
        accept_existing_exact=replace_exact,
    )


def _atomic_write_text(
    path: Path,
    value: str,
    *,
    replace_exact: bool = False,
) -> None:
    if not isinstance(value, str):
        raise TypeError("text artifact must be a string")
    _atomic_write_bytes(
        path,
        value.encode("utf-8"),
        accept_existing_exact=replace_exact,
    )


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    accept_existing_exact: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        if (
            accept_existing_exact
            and not destination.is_symlink()
            and destination.is_file()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_canonical_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise S1ProductionRunError(f"JSON artifact must be a real file: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S1ProductionRunError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise S1ProductionRunError(f"non-canonical JSON object: {path}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_safe(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise S1ProductionRunError("JSON value is non-finite")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise S1ProductionRunError("JSON Decimal is non-finite")
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _build_file_records(
    paths_and_roles: Iterable[tuple[Path, str]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path, role in paths_and_roles:
        selected = Path(path)
        if selected.is_symlink() or not selected.is_file():
            raise FileNotFoundError(f"input must be a regular file: {selected}")
        scope, portable = _portable_path_record(selected)
        identity = (scope, portable)
        if identity in seen:
            raise S1ProductionRunError("input path is duplicated")
        seen.add(identity)
        records.append(
            {
                "path_scope": scope,
                "path": portable,
                "role": str(role),
                "bytes": selected.stat().st_size,
                "sha256": _sha256_file(selected),
            }
        )
    return sorted(records, key=lambda row: (str(row["role"]), str(row["path"])))


def _validated_file_records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise S1ProductionRunError("input_records must be a JSON array")
    keys = {"path_scope", "path", "role", "bytes", "sha256"}
    records: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != keys:
            raise S1ProductionRunError("input record schema mismatch")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise S1ProductionRunError("input record bytes is invalid")
        if not _is_sha256(str(row["sha256"])):
            raise S1ProductionRunError("input record SHA-256 is invalid")
        _resolve_portable_path(row)
        records.append(dict(row))
    if records != sorted(records, key=lambda row: (str(row["role"]), str(row["path"]))):
        raise S1ProductionRunError("input records are not sorted")
    return records


def _verify_file_records(records: Sequence[Mapping[str, object]]) -> None:
    for record in records:
        path = _resolve_portable_path(record)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise S1ProductionRunError(f"input file content drifted: {path}")


def _verify_file_records_stable(records: Sequence[Mapping[str, object]]) -> None:
    paths = tuple(_resolve_portable_path(record) for record in records)
    before = _file_stats(paths)
    _verify_file_records(records)
    if before != _file_stats(paths):
        raise S1ProductionRunError(
            "input files changed while their content was verified"
        )


def _portable_path_record(path: Path) -> tuple[str, str]:
    resolved = path.resolve()
    for scope, root in (
        ("maker_root", MAKER_ROOT.resolve()),
        ("hft_root", HFT_ROOT.resolve()),
    ):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return scope, relative.as_posix()
    return "absolute", str(resolved)


def _portable_path(path: Path) -> dict[str, str]:
    scope, value = _portable_path_record(path)
    return {"path_scope": scope, "path": value}


def _resolve_portable_path(record: Mapping[str, object]) -> Path:
    scope = record.get("path_scope")
    raw = Path(str(record.get("path")))
    if scope == "absolute":
        if not raw.is_absolute():
            raise S1ProductionRunError("absolute input path is not absolute")
        return raw
    if raw.is_absolute() or ".." in raw.parts:
        raise S1ProductionRunError("relative input path is unsafe")
    if scope == "maker_root":
        return MAKER_ROOT / raw
    if scope == "hft_root":
        return HFT_ROOT / raw
    raise S1ProductionRunError("input path scope is invalid")


def _file_stats(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    result: list[tuple[str, int, int]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"input must be a regular file: {path}")
        stat = path.stat()
        result.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _taipei_execution_date(cursor: object) -> str:
    recv_time_ns = getattr(cursor, "recv_time_ns", None)
    if type(recv_time_ns) is not int:
        raise TypeError("cursor must expose integer recv_time_ns")
    seconds, _nanoseconds = divmod(recv_time_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).astimezone(TAIPEI).strftime("%Y%m%d")


def _git_source_commit(*, expected: str = "") -> str:
    repository = MAKER_ROOT.parent
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "maker/src"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise S1ProductionRunError("canonical run requires committed maker/src source")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _is_git_sha(commit):
        raise S1ProductionRunError("git did not return a source commit")
    if expected and expected != commit:
        raise S1ProductionRunError("expected source commit differs from current HEAD")
    return commit


def _is_git_sha(value: str) -> bool:
    return len(value) in (40, 64) and all(
        character in "0123456789abcdef" for character in value
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _print_progress(*, date: str, policy_id: str, completed: int, total: int) -> None:
    print(
        json.dumps(
            {
                "event": "s1_partition_complete",
                "date": date,
                "policy_id": policy_id,
                "completed": completed,
                "total": total,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--max-new-partitions", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "verify" and args.max_new_partitions is not None:
        raise S1ProductionRunError("verify cannot stop after new partitions")
    config = S1ProductionConfig(
        output_root=args.output_root,
        report_path=args.report_path,
        source_commit=args.source_commit,
        verify_completed_input_content=(args.verify_inputs or args.command == "verify"),
    )
    result = (
        verify_s1_production_bundle(config)
        if args.command == "verify"
        else run_s1_production_bundle(
            config,
            max_new_partitions=args.max_new_partitions,
        )
    )
    if result is None:
        return 0
    print(
        json.dumps(
            {
                "event": "s1_bundle_complete",
                "output_root": str(result.output_root),
                "report_path": str(result.report_path),
                "run_config_sha256": result.run_config_sha256,
                "complete_sha256": result.complete_sha256,
                "completion_champion": (
                    result.shortlist.completion_champion.scenario_id
                ),
                "net_champion": result.shortlist.net_champion.scenario_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REPORT_PATH",
    "PRODUCTION_RUNNER_VERSION",
    "S1_DEVELOPMENT_DATES",
    "S1ProductionConfig",
    "S1ProductionResult",
    "S1ProductionRunError",
    "run_s1_production_bundle",
    "verify_s1_production_bundle",
]
