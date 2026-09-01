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
import base64
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
from time import perf_counter
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

from ..common.paths import (
    HFT_ROOT,
    INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT,
    INDIVIDUAL_STOCK_FUTURES_ROOT,
    INPUT_PATH_RESOLUTION_POLICY_VERSION,
    LEGACY_HFT_DATA_ROOT,
    MAKER_ROOT,
    PIPELINE_STORAGE,
    validate_required_mount,
)
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
from .s1_day_state import (
    ENTRY_STOP_SECOND,
    EXIT_STOP_SECOND,
    S1_POLICY_DECISION_SIGNATURE_COLUMNS,
    SESSION_END_SECOND,
)
from .s1_economic_gate import S1EconomicGateAudit, S1EconomicGateEstimate
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
from .s1_exit_target import FUTURE_BUY_HEADROOM_TICKS
from .s1_hedge import HEDGE_DELAY_NS, HEDGE_RETRY_NS
from .s1_open_position_valuation import (
    COMMON_HORIZON_ASOF_ID,
    COMMON_HORIZON_CURSOR,
    COMMON_HORIZON_DATE,
    VALUATION_METHOD_ID,
    S1CommonHorizonBooks,
    S1CommonHorizonValuationResult,
    encode_s1_open_position_valuations,
    value_s1_common_horizon_open_positions,
)
from .s1_performance import (
    S1DailyReplaySummary,
    S1ScenarioPerformance,
    aggregate_s1_scenario_metrics,
    build_s1_daily_replay_summary_from_risk_events,
)
from .s1_publication_gate import (
    REQUIRED_UNMODELED_COST_IDS,
    S1EntryFunnelFacts,
    S1LookupDenominatorFacts,
    S1ModeledCostAccountingFacts,
    S1OpenPositionValuation,
    S1PositionAccountingFacts,
    S1PublicationClaims,
    S1PublicationDecision,
    S1PublicationScenarioFacts,
    S1UnavailableCost,
    evaluate_s1_publication_gate,
)
from .s1_ranking import S1ShortlistResult, ScenarioMetrics, rank_s1_shortlist
from .s1_scenario_spec import (
    CELL_KEYS,
    SCENARIO_BY_ID,
    SCENARIO_DEFINITIONS,
    SCENARIO_GRID_SHA256,
    SCENARIO_IDS,
    SCENARIO_SPEC_VERSION,
    load_s1_scenario_spec_table,
    s1_scenario_spec_table_sha256,
)
from .transaction_costs import TransactionCostProfile

PRODUCTION_RUNNER_VERSION: Final = (
    "s1_spot_bid_cost_aware_71x7_v8_entry_decision_clock"
)
RUN_CONFIG_SCHEMA_VERSION: Final = (
    "s1_spot_bid_run_config_v7_entry_decision_clock"
)
FINAL_BUNDLE_SCHEMA_VERSION: Final = (
    "s1_spot_bid_complete_v7_entry_decision_clock"
)
RESULTS_SCHEMA_VERSION: Final = "s1_spot_bid_results_v7_entry_decision_clock"
DAILY_METRICS_SCHEMA_VERSION: Final = "s1_spot_bid_daily_v3_entry_decision_clock"
COMMON_POPULATION_SCHEMA_VERSION: Final = "s1_common_population_date_value_quote_tod_v1"
VERIFICATION_SCHEMA_VERSION: Final = "s1_production_verification_v2_source_bound"
VERIFIER_VERSION: Final = "s1_production_deep_verifier_v5_entry_decision_clock"
VERIFICATION_FILENAME: Final = "verification.json"
ROUTE_ID: Final = "spot_bid_future_taker__spot_ask_future_taker_exit"
ENTRY_FILL_TRUTH: Final = "approximate"
EXIT_FILL_TRUTH: Final = "exact_indexed_print_volume"
DEVELOPMENT_END_DATE: Final = "20260813"
DEFAULT_OUTPUT_ROOT: Final = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "s1_spot_bid_cost_aware_20260901_v4_entry_decision_clock"
)
DEFAULT_REPORT_PATH: Final = (
    MAKER_ROOT
    / "doc"
    / "quote_fill"
    / "POLICY_COMPARISON_SPOT_BID_20260901_V4_ENTRY_DECISION_CLOCK.md"
)
CAPACITY_REGISTRY_FILENAME: Final = "capacity_identity_registry.sqlite"
GENESIS_PARTITION_SHA256: Final = hashlib.sha256(
    b"s1-policy-date-partition-genesis-v1"
).hexdigest()
TAIPEI: Final = ZoneInfo("Asia/Taipei")
PIPELINE_CONFIG_PATH: Final = HFT_ROOT / "config" / "pipeline.yaml"
PIPELINE_RESOLVER_PATH: Final = HFT_ROOT / "src" / "pipeline_storage.py"
DAILY_INPUT_ROLES: Final = frozenset(
    {
        "causal_fair",
        "mapping",
        "contract_metadata",
        "spot_raw",
        "future_raw",
        "makerfill",
    }
)

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
    common_horizon_opening_bindings: tuple[S1CarryContractBinding, ...] | None = None
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
    verifier_source_commit: str
    performance: tuple[S1ScenarioPerformance, ...]
    open_valuations: tuple[S1CommonHorizonValuationResult, ...]
    shortlist: S1ShortlistResult | None
    publication_decision: S1PublicationDecision
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
        PIPELINE_CONFIG_PATH,
        PIPELINE_RESOLVER_PATH,
    )
    before = _file_stats(static_paths)
    storage_contract = _semantic_storage_contract()
    scenario_table = load_s1_scenario_spec_table(
        config.paths.mother_path,
        config.paths.entry_lookup_path,
        config.paths.convergence_path,
    )
    dates = tuple(scenario_table.select("Date").unique().sort("Date")["Date"].to_list())
    if dates != S1_DEVELOPMENT_DATES:
        raise S1ProductionRunError("scenario table dates differ from frozen S1 dates")
    scenario_sha = s1_scenario_spec_table_sha256(scenario_table)
    common_population = _common_population_record(scenario_table)
    scenario_support = (
        scenario_table.group_by("scenario_id")
        .agg(
            pl.len().alias("common_cells"),
            pl.col("lookup_supported").sum().alias("lookup_supported_cells"),
        )
        .with_columns(
            (pl.col("common_cells") - pl.col("lookup_supported_cells")).alias(
                "lookup_unsupported_cells"
            )
        )
        .sort("scenario_id")
        .to_dicts()
    )
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
            (config.paths.convergence_path, "frozen_scenario_lower_lookup"),
            (PIPELINE_CONFIG_PATH, "pipeline_storage_config"),
            (PIPELINE_RESOLVER_PATH, "pipeline_storage_resolver"),
        )
    )
    source_snapshots = {
        "config": _embedded_source_record(PIPELINE_CONFIG_PATH),
        "resolver": _embedded_source_record(PIPELINE_RESOLVER_PATH),
    }
    if before != _file_stats(static_paths):
        raise S1ProductionRunError("static source changed while run config was built")
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "runner_version": PRODUCTION_RUNNER_VERSION,
        "source_commit": source_commit,
        "dates": list(S1_DEVELOPMENT_DATES),
        "policy_ids": list(SCENARIO_IDS),
        "route_id": ROUTE_ID,
        "entry_route": "Spot Bid maker -> Future sell taker",
        "normal_exit_route": "Spot Ask maker -> Future buy taker",
        "exit_target_freeze_semantics": (
            "actual_new_send_frozen_absolute_spot_ask_price_and_tick"
        ),
        "exit_target_repricing_after_entry": False,
        "exit_pre_fill_hedgeability_gate": {
            "required": True,
            "maker_venue": "spot",
            "hedge_venue": "future",
            "hedge_side": "buy",
            "quantity": "one_position_future_contracts",
            "wake_venues": ["spot", "future"],
            "future_buy_upper_band_headroom_ticks": FUTURE_BUY_HEADROOM_TICKS,
            "semantics": (
                "passive_spot_exit_requires_current_causal_full_depth_future_buy;"
                "actual_fill_still_uses_independent_t0_plus_retry_b6"
            ),
        },
        "exit_safety_cutoff": {
            "seconds_from_open": EXIT_STOP_SECOND,
            "research_horizon_seconds_from_open": SESSION_END_SECOND,
            "reserve_seconds": SESSION_END_SECOND - EXIT_STOP_SECOND,
            "drain_barrier_ns_from_open": (
                SESSION_END_SECOND * 1_000_000_000
                - HEDGE_DELAY_NS
                - 2 * HEDGE_RETRY_NS
            ),
            "drain_barrier_clock": "13:19:49.950 Asia/Taipei",
            "passive_orders_must_be_terminal_at_barrier": True,
            "risk_deadlines_clipped_to_research_horizon": True,
            "maximum_aggregate_product_orders": 244,
            "spot_cancel_request_limit_per_second": 100,
            "hedge_delay_ms": 50,
            "hedge_retry_ms": 5_000,
            "rollback_retry_ms": 5_000,
            "same_cursor_fill_precedes_cancel": True,
            "rule_id": "exit_maker_pre_horizon_risk_drain_v1",
        },
        "entry_fill_truth": ENTRY_FILL_TRUTH,
        "entry_fill_cursor_exact": False,
        "own_quantity_included": False,
        "partial_entry_fill_included": False,
        "entry_decision_clock": {
            "policy_clock": "causal_1hz_panel_relative_state_changes",
            "policy_decision_signature_columns": list(
                S1_POLICY_DECISION_SIGNATURE_COLUMNS
            ),
            "continuous_anchor_or_margin_in_signature": False,
            "actual_send_refresh": "exact_causal_raw_book_state",
            "actual_send_refresh_authoritative": True,
            "raw_entry_recovery_clock": (
                "route_visible_spot_bid_future_sell_buy_state_changes"
            ),
            "raw_entry_recovery_venues": ["spot", "future"],
            "capacity_blocked_monitor": (
                "product_policy_generation_until_send_or_policy_supersession"
            ),
            "stale_raw_wake_policy": "generation_invalidated_fail_closed",
            "risk_route_clock": "generic_effective_raw_book_changes",
            "panel_raw_reconstruction_equivalence_claimed": False,
        },
        "exit_fill_truth": EXIT_FILL_TRUTH,
        "anchor_model_id": ANCHOR_MODEL_ID,
        "entry_boundary_id": ENTRY_CANDIDATE_ID,
        "q_lower_scheme": "scenario_specific_C0_C2_C3_or_fixed20",
        "q_lower_status": "frozen_cost_aware_grid",
        "scenario_spec_version": SCENARIO_SPEC_VERSION,
        "scenario_grid_sha256": SCENARIO_GRID_SHA256,
        "scenario_definitions": [
            definition.to_dict() for definition in SCENARIO_DEFINITIONS
        ],
        "scenario_spec_sha256": scenario_sha,
        "common_population": common_population,
        "scenario_lookup_support": scenario_support,
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
        "input_storage_contract": storage_contract,
        "pipeline_storage_source_snapshots": source_snapshots,
        "static_inputs": static_records,
        "protected_forward_start_date": "20260814",
        "common_horizon_open_valuation": {
            "accounting_scope": "final_unsealed_positions_after_complete_71_day_replay",
            "market_book_asof_id": COMMON_HORIZON_ASOF_ID,
            "market_book_cursor": {
                "recv_time_ns": COMMON_HORIZON_CURSOR.recv_time_ns,
                "event_sequence": COMMON_HORIZON_CURSOR.event_sequence,
                "row_index": COMMON_HORIZON_CURSOR.row_index,
            },
            "valuation_method_id": VALUATION_METHOD_ID,
            "spot_liquidation": "full_aggregate_quantity_executable_sell_vwap",
            "future_liquidation": "full_aggregate_quantity_executable_buy_vwap",
            "same_product_depth_semantics": "aggregate_before_executable_vwap",
            "book_staleness_semantics": (
                "last_causal_state_asof_cursor_no_future_rows;"
                "explicit_clear_or_trial_remains_closed"
            ),
            "remaining_costs_included": True,
            "unpriced_semantics": "fail_closed_blocks_economic_ranking_and_s2",
            "protected_forward_consumed": False,
        },
    }


def _common_population_record(scenario_table: pl.DataFrame) -> dict[str, object]:
    """Hash the exact common Date/ValueCode/QuoteCode/TOD population."""

    population = scenario_table.select(*CELL_KEYS).unique().sort(CELL_KEYS)
    if population.height != 15_638 * 4:
        raise S1ProductionRunError(
            "common scenario population must contain 62,552 cells"
        )
    digest = hashlib.sha256()
    for row in population.iter_rows(named=True):
        digest.update(_canonical_json_bytes(row))
        digest.update(b"\n")
    return {
        "schema_version": COMMON_POPULATION_SCHEMA_VERSION,
        "key_columns": list(CELL_KEYS),
        "cell_count": population.height,
        "hash_semantics": "sha256_concatenated_canonical_json_lines_v1",
        "sha256": digest.hexdigest(),
    }


def _embedded_source_record(path: Path) -> dict[str, object]:
    """Embed exact source bytes so outer dirty files remain reconstructible."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise S1ProductionRunError(f"source snapshot must be a real file: {selected}")
    payload = selected.read_bytes()
    return {
        "path": _portable_path(selected),
        "encoding": "base64",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload_base64": base64.b64encode(payload).decode("ascii"),
    }


def _semantic_storage_contract() -> dict[str, object]:
    """Freeze external storage roles without conflating TXF and stock futures."""

    validate_required_mount(PIPELINE_STORAGE.required_mount, role="pipeline_spot")
    validate_required_mount(
        INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT,
        role="individual_stock_futures",
    )
    if PIPELINE_STORAGE.required_mount is None:
        raise S1ProductionRunError("pipeline storage requires an explicit mount")
    if not _path_is_within(
        PIPELINE_STORAGE.base_dir,
        PIPELINE_STORAGE.required_mount,
    ):
        raise S1ProductionRunError("pipeline base_dir is outside required_mount")
    if not _path_is_within(
        INDIVIDUAL_STOCK_FUTURES_ROOT,
        INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT,
    ):
        raise S1ProductionRunError(
            "individual stock-futures root is outside its required mount"
        )
    if (
        INDIVIDUAL_STOCK_FUTURES_ROOT.resolve()
        == PIPELINE_STORAGE.txf_tick_dir.resolve()
    ):
        raise S1ProductionRunError(
            "individual stock futures cannot use the TXF tick directory"
        )
    return {
        "resolution_policy_version": INPUT_PATH_RESOLUTION_POLICY_VERSION,
        "pipeline_base_dir": str(PIPELINE_STORAGE.base_dir.resolve()),
        "pipeline_required_mount": str(PIPELINE_STORAGE.required_mount.resolve()),
        "spot_tick_root": str(PIPELINE_STORAGE.tick_dir.resolve()),
        "makerfill_root": str(PIPELINE_STORAGE.maker_queue_dir.resolve()),
        "txf_tick_root": str(PIPELINE_STORAGE.txf_tick_dir.resolve()),
        "individual_stock_futures_root": str(INDIVIDUAL_STOCK_FUTURES_ROOT.resolve()),
        "individual_stock_futures_required_mount": str(
            INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT.resolve()
        ),
        "future_source_family": "individual_stock_futures",
        "txf_substitution_allowed": False,
        "legacy_spot_tick_root": str((LEGACY_HFT_DATA_ROOT / "tickData").resolve()),
        "legacy_makerfill_root": str((LEGACY_HFT_DATA_ROOT / "makerFill").resolve()),
        "legacy_fallback_requires_exact_filename": True,
        "custom_root_fallback_allowed": False,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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
    records = _build_file_records(
        (
            (config.paths.causal_path(date), "causal_fair"),
            (config.paths.mapping_path(date), "mapping"),
            (config.paths.contracts_path(date), "contract_metadata"),
            (config.paths.spot_raw_path(date), "spot_raw"),
            (config.paths.future_raw_path(date), "future_raw"),
            (config.paths.makerfill_path(date), "makerfill"),
        )
    )
    _validate_daily_file_records(records, date=date)
    return records


def _date_manifest_path(root: Path, date: str) -> Path:
    return root / "input_manifests" / f"Date={date}.json"


def _ensure_date_input_manifest(
    config: S1ProductionConfig,
    *,
    date: str,
    run_config_sha256: str,
    require_existing: bool = False,
) -> tuple[dict[str, object], str]:
    if not isinstance(require_existing, bool):
        raise TypeError("require_existing must be boolean")
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
        _validate_daily_file_records(records, date=date)
        if config.verify_completed_input_content:
            _verify_file_records_stable(records)
        return manifest, _sha256_file(path)

    if require_existing:
        raise S1ProductionRunError(
            f"verification requires existing date input manifest: {date}"
        )
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


def _validate_daily_file_records(
    records: Sequence[Mapping[str, object]],
    *,
    date: str,
) -> None:
    """Require one exact source family per daily role."""

    if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    roles = [str(record.get("role")) for record in records]
    if len(roles) != len(DAILY_INPUT_ROLES) or set(roles) != DAILY_INPUT_ROLES:
        raise S1ProductionRunError("daily input manifest role inventory drifted")
    if len(roles) != len(set(roles)):
        raise S1ProductionRunError("daily input manifest roles are duplicated")
    expected_names = {
        "causal_fair": "causal_fair.parquet",
        "mapping": "mapping.parquet",
        "contract_metadata": f"{date}_contracts.parquet",
        "spot_raw": f"{date}_StockTick.parquet",
        "future_raw": "stock_futures.parquet",
        "makerfill": f"{date}_makerFill.parquet",
    }
    by_role = {str(record["role"]): record for record in records}
    for role, expected_name in expected_names.items():
        path = _resolve_portable_path(by_role[role])
        if path.name != expected_name:
            raise S1ProductionRunError(
                f"daily {role} filename differs from its exact contract"
            )

    raw_paths = {
        role: _resolve_portable_path(by_role[role])
        for role in ("spot_raw", "makerfill", "future_raw")
    }
    for role, path in raw_paths.items():
        if any(component.is_symlink() for component in (path, *path.parents)):
            raise S1ProductionRunError(f"daily {role} path cannot contain a symlink")

    allowed_spot_parents = {
        PIPELINE_STORAGE.tick_dir.resolve(),
        (LEGACY_HFT_DATA_ROOT / "tickData").resolve(),
    }
    spot = raw_paths["spot_raw"]
    if spot.parent not in allowed_spot_parents:
        raise S1ProductionRunError(
            "spot_raw must use the exact pipeline or approved legacy root"
        )

    allowed_makerfill_parents = {
        PIPELINE_STORAGE.maker_queue_dir.resolve(),
        (LEGACY_HFT_DATA_ROOT / "makerFill").resolve(),
    }
    makerfill = raw_paths["makerfill"]
    if makerfill.parent not in allowed_makerfill_parents:
        raise S1ProductionRunError(
            "makerfill must use the exact pipeline or approved legacy root"
        )

    future = raw_paths["future_raw"]
    if _path_is_within(future, PIPELINE_STORAGE.txf_tick_dir):
        raise S1ProductionRunError("future_raw cannot resolve to TXF tick data")
    expected_future = (
        INDIVIDUAL_STOCK_FUTURES_ROOT.resolve()
        / date[:4]
        / date[4:6]
        / date[6:8]
        / "stock_futures.parquet"
    )
    if future != expected_future:
        raise S1ProductionRunError(
            "future_raw must use the exact individual-stock-futures NAS partition"
        )


_PREPARED_SOURCE_KEY_BY_DAILY_ROLE: Final = {
    "causal_fair": "causal_fair",
    "mapping": "mapping",
    "contract_metadata": "contracts",
    "spot_raw": "spot_raw",
    "future_raw": "future_raw",
    "makerfill": "makerfill",
}


def _validate_prepared_source_paths(
    source_paths: Mapping[str, Path],
    records: Sequence[Mapping[str, object]],
    *,
    date: str,
) -> None:
    """Bind the files loaded by ``prepare_s1_entry_day`` to its manifest."""

    if not isinstance(source_paths, Mapping):
        raise TypeError("prepared source_paths must be a mapping")
    _validate_daily_file_records(records, date=date)
    by_role = {str(record["role"]): record for record in records}
    for role, source_key in _PREPARED_SOURCE_KEY_BY_DAILY_ROLE.items():
        actual = source_paths.get(source_key)
        if not isinstance(actual, Path):
            raise S1ProductionRunError(
                f"prepared source path is missing or invalid for {role}"
            )
        expected = _resolve_portable_path(by_role[role])
        if actual.resolve(strict=False) != expected.resolve(strict=False):
            raise S1ProductionRunError(
                f"prepared {role} source differs from its input manifest"
            )


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
    if prepared.policy_ids != SCENARIO_IDS:
        raise S1ProductionRunError("prepared day lacks the canonical seven policies")
    expected_expiry_ns = (
        prepared.day_open_time_ns + SESSION_END_SECOND * 1_000_000_000
    )
    expected_barrier_ns = expected_expiry_ns - (
        HEDGE_DELAY_NS + 2 * HEDGE_RETRY_NS
    )
    if (
        prepared.entry_cutoff_time_ns
        != prepared.day_open_time_ns + ENTRY_STOP_SECOND * 1_000_000_000
        or prepared.exit_cutoff_time_ns
        != prepared.day_open_time_ns + EXIT_STOP_SECOND * 1_000_000_000
        or prepared.exit_drain_barrier_time_ns != expected_barrier_ns
        or prepared.session_expiry_time_ns != expected_expiry_ns
    ):
        raise S1ProductionRunError("prepared production clock contract drifted")
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


def _capture_common_horizon_opening_bindings(
    state: _PolicyRuntimeState,
    *,
    date: str,
) -> None:
    """Freeze pre-8/13 carry identities needed to rebuild the mark universe."""

    if date != COMMON_HORIZON_DATE:
        return
    current = tuple(state.carry_bindings)
    frozen = state.common_horizon_opening_bindings
    if frozen is None:
        state.common_horizon_opening_bindings = current
    elif frozen != current:
        raise S1ProductionRunError(
            "common-horizon opening carry bindings changed while resuming"
        )


def _merge_common_horizon_bindings(
    states: Mapping[str, _PolicyRuntimeState],
) -> tuple[S1CarryContractBinding, ...]:
    by_product: dict[str, S1CarryContractBinding] = {}
    for policy_id in SCENARIO_IDS:
        state = states[policy_id]
        bindings = state.common_horizon_opening_bindings
        if bindings is None:
            raise S1ProductionRunError(
                f"{policy_id} lacks frozen common-horizon opening bindings"
            )
        for binding in bindings:
            existing = by_product.get(binding.product_id)
            if existing is not None and existing != binding:
                raise S1ProductionRunError(
                    "common-horizon policies require conflicting contract bindings"
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

    This path never executes a policy, creates a registry, or fills a missing
    artifact.  It first verifies the existing final marker and every hash it
    names, replays all partition facts/capacity transitions, and read-only
    rebuilds the 8/13 books needed for the common-horizon mark.  Every final
    artifact must already exist with exactly the reconstructed bytes.
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
    _write_verification_attestation(verified_config, result)
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
        verifier_source_commit = _git_source_commit(expected=source_commit)
    else:
        if (config.output_root / "complete.json").exists() or (
            config.output_root / "complete.json"
        ).is_symlink():
            raise S1ProductionRunError(
                "bundle already has a final marker; use the verify command"
            )
        source_commit = _git_source_commit(expected=config.source_commit)
        verifier_source_commit = source_commit
    run_config = _semantic_run_config(config, source_commit=source_commit)
    run_config_path = config.output_root / "run_config.json"
    run_config_sha256 = _ensure_run_config(
        run_config_path,
        run_config,
        require_existing_exact=verification_only,
    )
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
            verification_only=verification_only,
        )
        coordinates = tuple(
            (date, policy_id)
            for date in S1_DEVELOPMENT_DATES
            for policy_id in SCENARIO_IDS
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
                prepare_started = perf_counter()
                _print_runtime_progress(
                    event="s1_date_prepare_start",
                    date=date,
                    completed=resumed_partitions + executed,
                    total=len(coordinates),
                )
                _manifest, date_manifest_sha256 = _ensure_date_input_manifest(
                    config,
                    date=date,
                    run_config_sha256=run_config_sha256,
                )
                manifest_records = _validated_file_records(_manifest["input_records"])
                _verify_file_records_stable(manifest_records)
                remaining_policy_ids = SCENARIO_IDS[SCENARIO_IDS.index(policy_id) :]
                required_exit_only_bindings = _required_exit_only_bindings(
                    states,
                    policy_ids=remaining_policy_ids,
                )
                prepared = prepare_s1_entry_day(
                    date,
                    paths=config.paths,
                    required_exit_only_bindings=required_exit_only_bindings,
                )
                _validate_prepared_source_paths(
                    prepared.source_paths,
                    manifest_records,
                    date=date,
                )
                _verify_file_records_stable(manifest_records)
                _validate_prepared_day(prepared, date=date, catalog=catalog)
                current_date = date
                _print_runtime_progress(
                    event="s1_date_prepare_complete",
                    date=date,
                    completed=resumed_partitions + executed,
                    total=len(coordinates),
                    elapsed_seconds=perf_counter() - prepare_started,
                    sparse_state_rows=sum(
                        frame.height
                        for frame in prepared.state_changes_by_policy.values()
                    ),
                    raw_events=prepared.raw_books.retained_event_count,
                    entry_route_raw_changes=(
                        prepared.raw_books.entry_quote_change_count
                    ),
                )
            assert prepared is not None
            state = states[policy_id]
            _capture_common_horizon_opening_bindings(state, date=date)
            _validate_opening_carry(state, prepared)
            lineage = _lineage_record(
                date_input_manifest_sha256=date_manifest_sha256,
                previous_global_partition_sha256=previous_global_sha256,
                state=state,
            )
            partition_started = perf_counter()
            _print_runtime_progress(
                event="s1_partition_start",
                date=date,
                policy_id=policy_id,
                completed=resumed_partitions + executed,
                total=len(coordinates),
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
                elapsed_seconds=perf_counter() - partition_started,
            )
            if max_new_partitions is not None and executed >= max_new_partitions:
                return None
        del prepared
        gc.collect()
        registry_receipts = registry.verify()

    open_valuations = _build_common_horizon_valuations(
        config=config,
        run_config_sha256=run_config_sha256,
        states=states,
        catalog=catalog,
        verification_only=verification_only,
    )

    result = _finalize_production_bundle(
        config=config,
        run_config=run_config,
        run_config_sha256=run_config_sha256,
        states=states,
        open_valuations=open_valuations,
        verification_only=verification_only,
        verifier_source_commit=verifier_source_commit,
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
    verification_only: bool,
) -> tuple[dict[str, _PolicyRuntimeState], str, int, int]:
    if not isinstance(verification_only, bool):
        raise TypeError("verification_only must be boolean")
    facts_by_policy: dict[str, list[AccountingFact]] = {
        policy_id: [] for policy_id in SCENARIO_IDS
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
        for policy_id in SCENARIO_IDS
    }
    previous_global = GENESIS_PARTITION_SHA256
    coordinates = tuple(
        (date, policy_id) for date in S1_DEVELOPMENT_DATES for policy_id in SCENARIO_IDS
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
                require_existing=verification_only,
            )
            manifest_sha_by_date[date] = manifest_sha
        state = states[policy_id]
        _capture_common_horizon_opening_bindings(state, date=date)
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
        economic_estimates = tuple(
            S1EconomicGateEstimate.from_record(record)
            for record in records.economic_gate_estimate_records
        )
        economic_audits = tuple(
            S1EconomicGateAudit.from_record(record)
            for record in records.economic_gate_event_records
        )
        _validate_economic_gate_partition(
            economic_estimates,
            economic_audits,
            date=date,
            policy_id=policy_id,
            summary=summary,
            diagnostics=diagnostics,
        )
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


def _validate_economic_gate_partition(
    estimates: Sequence[S1EconomicGateEstimate],
    audits: Sequence[S1EconomicGateAudit],
    *,
    date: str,
    policy_id: str,
    summary: S1EntryDaySummary,
    diagnostics: Mapping[str, object] | None = None,
) -> None:
    """Verify persisted decision/actual-send estimates and dispatch audits."""

    estimate_values = tuple(estimates)
    values = tuple(audits)
    if any(
        estimate.date != date or estimate.policy_id != policy_id
        for estimate in estimate_values
    ):
        raise S1ProductionRunError("economic-gate estimate identity drifted")
    if any(
        audit.estimate.date != date or audit.estimate.policy_id != policy_id
        for audit in values
    ):
        raise S1ProductionRunError("economic-gate audit identity drifted")
    estimate_keys = [
        (
            estimate.evaluation_stage,
            estimate.product_id,
            estimate.observation_cursor,
        )
        for estimate in estimate_values
    ]
    if len(estimate_keys) != len(set(estimate_keys)):
        raise S1ProductionRunError("economic-gate estimate keys are duplicated")
    keys = [
        (
            audit.estimate.product_id,
            audit.estimate.observation_cursor,
        )
        for audit in values
    ]
    if len(keys) != len(set(keys)):
        raise S1ProductionRunError("economic-gate actual-send keys are duplicated")
    decision_estimates = tuple(
        value
        for value in estimate_values
        if value.evaluation_stage == "decision_observation"
    )
    actual_send_estimates = tuple(
        value
        for value in estimate_values
        if value.evaluation_stage == "actual_send_refresh"
    )
    if len(decision_estimates) != summary.decision_economic_checks:
        raise S1ProductionRunError("decision economic-gate estimate count differs")
    if sum(value.gate_open for value in decision_estimates) != (
        summary.decision_economic_gate_open
    ):
        raise S1ProductionRunError("decision economic-gate open count differs")
    if sum(not value.gate_open for value in decision_estimates) != (
        summary.decision_economic_gate_closed
    ):
        raise S1ProductionRunError("decision economic-gate closed count differs")
    if len(actual_send_estimates) != summary.actual_send_economic_checks:
        raise S1ProductionRunError("actual-send economic-gate estimate count differs")
    if sum(value.gate_open for value in actual_send_estimates) != (
        summary.actual_send_economic_gate_open
    ):
        raise S1ProductionRunError("actual-send economic-gate open count differs")
    if sum(not value.gate_open for value in actual_send_estimates) != (
        summary.actual_send_economic_gate_closed
    ):
        raise S1ProductionRunError("actual-send economic-gate closed count differs")
    actual_by_key = {
        (estimate.product_id, estimate.observation_cursor): estimate
        for estimate in actual_send_estimates
    }
    if any(
        actual_by_key.get(
            (audit.estimate.product_id, audit.estimate.observation_cursor)
        )
        != audit.estimate
        for audit in values
    ):
        raise S1ProductionRunError(
            "economic-gate audit estimate is not persisted exactly"
        )
    if len(actual_by_key) != len(values):
        raise S1ProductionRunError("actual-send estimate/audit populations differ")
    outcome_counts = Counter(audit.dispatch_outcome for audit in values)
    if len(values) != summary.actual_send_economic_checks:
        raise S1ProductionRunError("economic-gate audit count differs from summary")
    if outcome_counts["sent"] != summary.sent_entry_orders:
        raise S1ProductionRunError("economic-gate sent count differs from summary")
    if (
        outcome_counts["economic_gate_blocked"]
        != summary.actual_send_economic_gate_closed
    ):
        raise S1ProductionRunError("economic-gate blocked count differs from summary")
    if (
        outcome_counts["not_sent_after_gate_pass"]
        != summary.actual_send_not_sent_after_gate_pass
    ):
        raise S1ProductionRunError("post-gate non-send count differs from summary")
    if (
        outcome_counts["sent"] + outcome_counts["not_sent_after_gate_pass"]
        != summary.actual_send_economic_gate_open
    ):
        raise S1ProductionRunError("economic-gate open count differs from summary")
    sent_ids = [
        audit.raw_order_fact_id for audit in values if audit.dispatch_outcome == "sent"
    ]
    if len(sent_ids) != len(set(sent_ids)):
        raise S1ProductionRunError("economic-gate sent raw-order IDs are duplicated")
    if diagnostics is not None:
        expected_counters = {
            "economic_gate_decision_status_counts": Counter(
                value.status for value in decision_estimates
            ),
            "economic_gate_actual_send_status_counts": Counter(
                value.status for value in actual_send_estimates
            ),
            "economic_gate_dispatch_outcome_counts": outcome_counts,
        }
        for name, expected in expected_counters.items():
            actual = diagnostics.get(name)
            if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
                raise S1ProductionRunError(
                    f"economic-gate persisted stream differs from diagnostics: {name}"
                )


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
    _validate_economic_gate_partition(
        run.economic_gate_estimates,
        run.economic_gate_audits,
        date=prepared.date,
        policy_id=policy_id,
        summary=run.summary,
        diagnostics=diagnostics,
    )
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
        economic_gate_estimate_records=(
            estimate.to_record() for estimate in run.economic_gate_estimates
        ),
        economic_gate_event_records=(
            audit.to_record() for audit in run.economic_gate_audits
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
    _validate_economic_gate_partition(
        run.economic_gate_estimates,
        run.economic_gate_audits,
        date=prepared.date,
        policy_id=policy_id,
        summary=run.summary,
    )
    if not result.normal_exit_enabled:
        raise S1ProductionRunError("canonical S1 replay requires normal exit")
    if not result.exit_cutoff_applied or not result.exit_drain_barrier_applied:
        raise S1ProductionRunError(
            "canonical S1 replay requires the exit cutoff and drain barrier"
        )
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


def _build_common_horizon_valuations(
    *,
    config: S1ProductionConfig,
    run_config_sha256: str,
    states: Mapping[str, _PolicyRuntimeState],
    catalog: Mapping[str, S1AccountingProduct],
    verification_only: bool,
) -> tuple[S1CommonHorizonValuationResult, ...]:
    """Rebuild 8/13 books and mark only positions still open after all facts."""

    if not isinstance(verification_only, bool):
        raise TypeError("verification_only must be boolean")
    if DEVELOPMENT_END_DATE != COMMON_HORIZON_DATE:
        raise S1ProductionRunError(
            "development end date differs from common valuation horizon"
        )
    required_bindings = _merge_common_horizon_bindings(states)
    manifest, _manifest_sha256 = _ensure_date_input_manifest(
        config,
        date=COMMON_HORIZON_DATE,
        run_config_sha256=run_config_sha256,
        require_existing=verification_only,
    )
    manifest_records = _validated_file_records(manifest["input_records"])
    _verify_file_records_stable(manifest_records)
    prepared = prepare_s1_entry_day(
        COMMON_HORIZON_DATE,
        paths=config.paths,
        required_exit_only_bindings=required_bindings,
    )
    _validate_prepared_source_paths(
        prepared.source_paths,
        manifest_records,
        date=COMMON_HORIZON_DATE,
    )
    _verify_file_records_stable(manifest_records)
    _validate_prepared_day(prepared, date=COMMON_HORIZON_DATE, catalog=catalog)
    if prepared.session_expiry_time_ns != COMMON_HORIZON_CURSOR.recv_time_ns:
        raise S1ProductionRunError(
            "prepared 8/13 session close differs from valuation book horizon"
        )
    adapter = prepared.raw_books.as_risk_book_adapter()
    books_by_value_code = {
        product.value_code: S1CommonHorizonBooks(
            spot=adapter.state_as_of(
                "spot",
                product.product_id,
                COMMON_HORIZON_CURSOR,
            ),
            future=adapter.state_as_of(
                "future",
                product.product_id,
                COMMON_HORIZON_CURSOR,
            ),
        )
        for product in prepared.products
    }
    results = tuple(
        value_s1_common_horizon_open_positions(
            states[policy_id].accounting.verify().facts,
            scenario_id=policy_id,
            books_by_value_code=books_by_value_code,
        )
        for policy_id in SCENARIO_IDS
    )
    del prepared, adapter, books_by_value_code
    gc.collect()
    return results


def _validate_common_horizon_accounting(
    performance: S1ScenarioPerformance,
    valuation: S1CommonHorizonValuationResult,
    diagnostics: Mapping[str, object],
) -> tuple[int, int]:
    """Conserve final paired/naked populations and open execution costs."""

    if performance.scenario_id != valuation.scenario_id:
        raise S1ProductionRunError("performance/common-horizon scenario differs")
    final_paired_open = len(valuation.rows)
    final_naked = performance.open_or_unresolved - final_paired_open
    if final_naked < 0:
        raise S1ProductionRunError(
            "final paired-open count exceeds open-or-unresolved accounting"
        )
    final_carry = diagnostics.get("final_carry_positions")
    diagnostic_final_naked = diagnostics.get("final_naked_unresolved_positions")
    for name, value in (
        ("final_carry_positions", final_carry),
        ("final_naked_unresolved_positions", diagnostic_final_naked),
    ):
        if type(value) is not int or value < 0:
            raise S1ProductionRunError(
                f"common-horizon diagnostic {name} is invalid"
            )
    if final_paired_open != final_carry:
        raise S1ProductionRunError(
            "valuation paired-open count differs from final carry diagnostics"
        )
    if final_naked != diagnostic_final_naked:
        raise S1ProductionRunError(
            "final naked count differs between accounting and diagnostics"
        )
    if final_naked != 0:
        raise S1ProductionRunError(
            "canonical S1 final horizon contains naked unresolved positions"
        )
    incurred_commission = sum(
        (row.incurred_commission_twd for row in valuation.rows),
        start=Decimal(0),
    )
    incurred_tax = sum(
        (row.incurred_tax_twd for row in valuation.rows),
        start=Decimal(0),
    )
    comparisons = (
        (
            "commission",
            incurred_commission,
            performance.open_execution_actual_commission_twd,
        ),
        ("tax", incurred_tax, performance.open_execution_actual_tax_twd),
        (
            "total",
            valuation.incurred_open_execution_cost_twd,
            performance.open_execution_actual_cost_twd,
        ),
    )
    for name, actual, expected in comparisons:
        if abs(actual - expected) > Decimal("0.000001"):
            raise S1ProductionRunError(
                f"common-horizon open {name} differs from performance accounting"
            )
    return final_paired_open, final_naked


def _finalize_production_bundle(
    *,
    config: S1ProductionConfig,
    run_config: Mapping[str, object],
    run_config_sha256: str,
    states: Mapping[str, _PolicyRuntimeState],
    open_valuations: Sequence[S1CommonHorizonValuationResult],
    verification_only: bool,
    verifier_source_commit: str,
    registry_receipts: object,
    resumed_partitions: int,
    executed_partitions: int,
) -> S1ProductionResult:
    if not isinstance(verification_only, bool):
        raise TypeError("verification_only must be boolean")
    if not _is_git_sha(verifier_source_commit):
        raise S1ProductionRunError("verifier_source_commit must be a git SHA")
    if not isinstance(registry_receipts, dict) or set(registry_receipts) != set(
        SCENARIO_IDS
    ):
        raise S1ProductionRunError("final capacity registry policy set differs")
    valuation_tuple = tuple(open_valuations)
    if tuple(value.scenario_id for value in valuation_tuple) != SCENARIO_IDS:
        raise S1ProductionRunError(
            "common-horizon valuation policy order/set differs"
        )
    valuation_by_policy = {
        value.scenario_id: value for value in valuation_tuple
    }
    performances: list[S1ScenarioPerformance] = []
    diagnostics_by_policy: dict[str, dict[str, object]] = {}
    entry_month_by_policy: dict[str, list[dict[str, object]]] = {}
    for policy_id in SCENARIO_IDS:
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
        valuation = valuation_by_policy[policy_id]
        diagnostics = _aggregate_daily_diagnostics(
            state.daily_diagnostics,
            state.daily_summaries,
        )
        _validate_common_horizon_accounting(performance, valuation, diagnostics)
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
    common_population_sha256 = _run_config_common_population_sha256(run_config)
    publication_facts = tuple(
        _publication_scenario_facts(
            performance,
            diagnostics_by_policy[performance.scenario_id],
            open_valuation=valuation_by_policy[performance.scenario_id].aggregate,
            paired_open_positions=len(
                valuation_by_policy[performance.scenario_id].rows
            ),
            naked_unresolved_positions=(
                performance.open_or_unresolved
                - len(valuation_by_policy[performance.scenario_id].rows)
            ),
            common_population_sha256=common_population_sha256,
        )
        for performance in performance_tuple
    )
    descriptive_decision = evaluate_s1_publication_gate(
        publication_facts,
        S1PublicationClaims(),
    )
    descriptive_decision.require_publishable()
    shortlist: S1ShortlistResult | None = None
    claims = S1PublicationClaims()
    if descriptive_decision.economic_ranking_allowed:
        publication_by_policy = {
            value.scenario_id: value for value in publication_facts
        }
        deployment_performance = tuple(
            value
            for value in performance_tuple
            if SCENARIO_BY_ID[value.scenario_id].deployment_shortlist_eligible
        )
        shortlist = rank_s1_shortlist(
            tuple(
                ScenarioMetrics(
                    scenario_id=value.scenario_id,
                    completion_numerator=(
                        value.approx_screen_completion_numerator
                    ),
                    completion_denominator=(
                        value.approx_screen_completion_denominator
                    ),
                    total_net_twd=_required_economic_ranking_net(
                        publication_by_policy[value.scenario_id]
                    ),
                    reporting_sessions=value.reporting_sessions,
                    hedge_priced_numerator=value.hedge_priced_numerator,
                    hedge_priced_denominator=value.hedge_priced_denominator,
                )
                for value in deployment_performance
            )
        )
        claims = S1PublicationClaims(
            completion_champion_id=shortlist.completion_champion.scenario_id,
            net_champion_id=shortlist.net_champion.scenario_id,
            pareto_frontier_ids=tuple(
                value.scenario_id for value in shortlist.pareto_frontier
            ),
            s2_shortlist_ids=tuple(value.scenario_id for value in shortlist.selected),
        )
    publication_decision = evaluate_s1_publication_gate(publication_facts, claims)
    publication_decision.require_publishable()
    result_record = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_config_sha256": run_config_sha256,
        "entry_fill_truth": ENTRY_FILL_TRUTH,
        "capacity_attestation": {
            "global_cap_twd": DEFAULT_GLOBAL_CAP_TWD,
            "product_cap_twd": DEFAULT_PRODUCT_CAP_TWD,
            "all_capacity_partitions_verified": True,
            "exact_identity_registry_verified": True,
            "naked_unresolved_fail_closed": True,
        },
        "performance": [
            _performance_record(value, valuation_by_policy[value.scenario_id])
            for value in performance_tuple
        ],
        "open_position_valuations": [
            _open_valuation_result_record(valuation_by_policy[policy_id])
            for policy_id in SCENARIO_IDS
        ],
        "entry_cohort_month": entry_month_by_policy,
        "diagnostics": diagnostics_by_policy,
        "publication_gate": _publication_decision_record(publication_decision),
        "shortlist": _shortlist_record(shortlist, publication_decision),
    }
    daily_record = {
        "schema_version": DAILY_METRICS_SCHEMA_VERSION,
        "rows": [
            summary.as_dict()
            for policy_id in SCENARIO_IDS
            for summary in states[policy_id].daily_summaries or ()
        ],
        "risk_rows": [
            _risk_summary_record(summary)
            for policy_id in SCENARIO_IDS
            for summary in states[policy_id].daily_risk_summaries or ()
        ],
    }
    results_path = config.output_root / "results.json"
    daily_path = config.output_root / "daily_metrics.json"
    _publish_or_verify_canonical_json(
        results_path,
        result_record,
        verification_only=verification_only,
    )
    _publish_or_verify_canonical_json(
        daily_path,
        daily_record,
        verification_only=verification_only,
    )
    report = _render_report(
        run_config=run_config,
        run_config_sha256=run_config_sha256,
        performance=performance_tuple,
        open_valuations=valuation_by_policy,
        shortlist=shortlist,
        publication_decision=publication_decision,
        diagnostics=diagnostics_by_policy,
        entry_month=entry_month_by_policy,
    )
    _publish_or_verify_text(
        config.report_path,
        report,
        verification_only=verification_only,
    )
    partition_records = [
        {
            "date": date,
            "policy_id": policy_id,
            "complete_sha256": _partition_marker_sha256(
                _partition_path(config.output_root, date, policy_id)
            ),
        }
        for date in S1_DEVELOPMENT_DATES
        for policy_id in SCENARIO_IDS
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
        "expected_partition_count": len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS),
        "partitions": partition_records,
        "input_manifests": input_records,
    }
    complete["marker_payload_sha256"] = _canonical_sha256(complete)
    complete_path = config.output_root / "complete.json"
    _publish_or_verify_canonical_json(
        complete_path,
        complete,
        verification_only=verification_only,
    )
    complete_sha = _sha256_file(complete_path)
    return S1ProductionResult(
        output_root=config.output_root,
        report_path=config.report_path,
        run_config_sha256=run_config_sha256,
        complete_sha256=complete_sha,
        verifier_source_commit=verifier_source_commit,
        performance=performance_tuple,
        open_valuations=valuation_tuple,
        shortlist=shortlist,
        publication_decision=publication_decision,
        resumed_partitions=resumed_partitions,
        executed_partitions=executed_partitions,
    )


def _publication_scenario_facts(
    performance: S1ScenarioPerformance,
    diagnostics: Mapping[str, object],
    *,
    open_valuation: S1OpenPositionValuation | None,
    paired_open_positions: int,
    naked_unresolved_positions: int,
    common_population_sha256: str,
) -> S1PublicationScenarioFacts:
    """Adapt verifier-backed aggregates to the fail-closed publication gate."""

    definition = SCENARIO_BY_ID[performance.scenario_id]

    def count(name: str) -> int:
        value = diagnostics.get(name)
        if type(value) is not int or value < 0:
            raise S1ProductionRunError(
                f"publication diagnostic {name} must be a non-negative integer"
            )
        return value

    common_cells = count("common_lookup_cells")
    supported_cells = count("lookup_supported_cells")
    unsupported_cells = count("lookup_unsupported_cells")
    sent_orders = count("sent_entry_orders")
    makerfill_supported = count("makerfill_supported_orders")
    if (
        type(paired_open_positions) is not int
        or paired_open_positions < 0
        or type(naked_unresolved_positions) is not int
        or naked_unresolved_positions < 0
        or paired_open_positions + naked_unresolved_positions
        != performance.open_or_unresolved
    ):
        raise S1ProductionRunError(
            "publication final paired/naked population does not conserve"
        )
    unmodeled_reasons = {
        "futures_margin_opportunity_cost": "margin_funding_cost_not_modeled",
        "overnight_financing": "financing_rate_and_funding_contract_not_modeled",
        "spot_borrow_cost": "borrow_availability_and_fee_not_modeled",
    }
    return S1PublicationScenarioFacts(
        scenario_id=performance.scenario_id,
        reporting_sessions=performance.reporting_sessions,
        cost_horizon=definition.cost_horizon,
        economic_gate_enabled=definition.economic_gate_enabled,
        deployment_shortlist_eligible=definition.deployment_shortlist_eligible,
        lookup=S1LookupDenominatorFacts(
            common_mother_product_days=15_638,
            tod_bucket_count=4,
            expected_policy_cells=15_638 * 4,
            policy_spec_rows=common_cells,
            supported_policy_cells=supported_cells,
            unsupported_policy_cells=unsupported_cells,
            reporting_denominator_cells=common_cells,
            unsupported_no_trade_cells=unsupported_cells,
            common_population_sha256=common_population_sha256,
        ),
        funnel=S1EntryFunnelFacts(
            candidate_intents=count("candidate_intents"),
            reservation_attempts=count("reservation_attempts"),
            admitted_orders=count("admitted_orders"),
            cap_blocked_attempts=count("cap_blocked_attempts"),
            blocked_candidate_intents=count("blocked_candidate_intents"),
            sent_orders=sent_orders,
            makerfill_supported_orders=makerfill_supported,
            makerfill_unsupported_orders=count("makerfill_unsupported_orders"),
            filled_orders=count("actual_active_entry_fills"),
            actual_cancelled_orders=count("actual_cancelled_orders"),
            session_expired_orders=count("session_expired_orders"),
            unknown_terminal_orders=count("unknown_terminal_orders"),
        ),
        accounting=S1PositionAccountingFacts(
            entry_positions=performance.entry_positions,
            same_day_exit_maker_flat=performance.same_day_exit_maker_flat,
            cross_day_exit_maker_flat=performance.cross_day_exit_maker_flat,
            expiry_marks=performance.expiry_marks,
            entry_rollbacks=performance.entry_rollbacks,
            other_executable_terminals=performance.other_executable_terminals,
            paired_open_positions=paired_open_positions,
            naked_unresolved_positions=naked_unresolved_positions,
            open_or_unresolved=performance.open_or_unresolved,
            terminal_coverage_numerator=performance.terminal_coverage_numerator,
            terminal_coverage_denominator=performance.terminal_coverage_denominator,
        ),
        costs=S1ModeledCostAccountingFacts(
            executable_terminal_gross_pnl_twd=(
                performance.executable_terminal_gross_pnl_twd
            ),
            expiry_mark_gross_pnl_twd=performance.expiry_mark_gross_pnl_twd,
            terminal_gross_pnl_twd=performance.terminal_gross_pnl_twd,
            executable_terminal_commission_twd=(
                performance.executable_terminal_commission_twd
            ),
            executable_terminal_tax_twd=performance.executable_terminal_tax_twd,
            expiry_mark_commission_twd=performance.expiry_mark_commission_twd,
            expiry_mark_tax_twd=performance.expiry_mark_tax_twd,
            terminal_modeled_direct_cost_twd=(
                performance.terminal_modeled_direct_cost_twd
            ),
            executable_terminal_net_twd=performance.executable_terminal_net_twd,
            expiry_mark_net_twd=performance.expiry_mark_net_twd,
            total_net_twd=performance.total_net_twd,
            terminal_executed_turnover_twd=(performance.terminal_executed_turnover_twd),
            open_executed_turnover_twd=performance.open_executed_turnover_twd,
            total_executed_turnover_twd=performance.total_executed_turnover_twd,
            open_execution_actual_commission_twd=(
                performance.open_execution_actual_commission_twd
            ),
            open_execution_actual_tax_twd=(performance.open_execution_actual_tax_twd),
            open_execution_actual_cost_twd=performance.open_execution_actual_cost_twd,
        ),
        unmodeled_costs=tuple(
            S1UnavailableCost(
                cost_id=cost_id,
                amount_twd=None,
                unavailable_reason=unmodeled_reasons[cost_id],
            )
            for cost_id in REQUIRED_UNMODELED_COST_IDS
        ),
        open_valuation=open_valuation,
        economic_ranking_net_twd=(
            performance.total_net_twd
            if paired_open_positions == 0
            else (
                None
                if open_valuation is None
                else performance.total_net_twd + open_valuation.net_mark_pnl_twd
            )
        ),
        hedge_priced_numerator=performance.hedge_priced_numerator,
        hedge_priced_denominator=performance.hedge_priced_denominator,
    )


def _required_economic_ranking_net(
    value: S1PublicationScenarioFacts,
) -> Decimal:
    result = value.economic_ranking_net_twd
    if result is None:
        raise S1ProductionRunError(
            "publication gate allowed ranking without an economic net"
        )
    return result


def _run_config_common_population_sha256(
    run_config: Mapping[str, object],
) -> str:
    record = run_config.get("common_population")
    expected_keys = {
        "schema_version",
        "key_columns",
        "cell_count",
        "hash_semantics",
        "sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise S1ProductionRunError("run config common population record drifted")
    if (
        record["schema_version"] != COMMON_POPULATION_SCHEMA_VERSION
        or record["key_columns"] != list(CELL_KEYS)
        or record["cell_count"] != 15_638 * 4
        or record["hash_semantics"] != "sha256_concatenated_canonical_json_lines_v1"
    ):
        raise S1ProductionRunError("run config common population identity drifted")
    sha256 = record["sha256"]
    if not isinstance(sha256, str) or not _is_sha256(sha256):
        raise S1ProductionRunError("run config common population SHA-256 is invalid")
    return sha256


def _open_valuation_aggregate_record(
    value: S1OpenPositionValuation | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "position_count": value.position_count,
        "valuation_method_id": value.valuation_method_id,
        "valuation_asof_id": value.valuation_asof_id,
        "comparable_across_scenarios": value.comparable_across_scenarios,
        "gross_mark_pnl_twd": str(value.gross_mark_pnl_twd),
        "remaining_exit_cost_twd": str(value.remaining_exit_cost_twd),
        "net_mark_pnl_twd": str(value.net_mark_pnl_twd),
    }


def _open_valuation_result_record(
    value: S1CommonHorizonValuationResult,
) -> dict[str, object]:
    reason_counts = Counter(
        row.unpriced_reason for row in value.rows if row.unpriced_reason is not None
    )
    all_priced = all(row.priced for row in value.rows)
    spot_liquidation_notional = (
        sum(
            (row.spot_liquidation_cashflow_twd for row in value.rows),
            start=Decimal(0),
        )
        if all_priced
        else None
    )
    future_liquidation_notional = (
        sum(
            (
                row.future_exit_vwap * row.future_share_equivalent
                for row in value.rows
            ),
            start=Decimal(0),
        )
        if all_priced
        else None
    )
    return {
        "scenario_id": value.scenario_id,
        "valuation_method_id": VALUATION_METHOD_ID,
        "valuation_asof_id": COMMON_HORIZON_ASOF_ID,
        "final_open_position_count": len(value.rows),
        "priced_position_count": sum(row.priced for row in value.rows),
        "unpriced_position_count": sum(not row.priced for row in value.rows),
        "unpriced_reason_counts": dict(sorted(reason_counts.items())),
        "incurred_open_execution_cost_twd": str(
            value.incurred_open_execution_cost_twd
        ),
        "spot_liquidation_notional_twd": (
            None
            if spot_liquidation_notional is None
            else str(spot_liquidation_notional)
        ),
        "future_liquidation_notional_twd": (
            None
            if future_liquidation_notional is None
            else str(future_liquidation_notional)
        ),
        "publishable_aggregate": _open_valuation_aggregate_record(value.aggregate),
        "rows": encode_s1_open_position_valuations(value.rows),
    }


def _performance_record(
    value: S1ScenarioPerformance,
    open_valuation: S1CommonHorizonValuationResult,
) -> dict[str, object]:
    if value.scenario_id != open_valuation.scenario_id:
        raise S1ProductionRunError("performance/open valuation scenario differs")
    aggregate = open_valuation.aggregate
    economic_net = (
        None
        if aggregate is None
        else value.total_net_twd + aggregate.net_mark_pnl_twd
    )
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
            "mean_daily_terminal_realized_net_twd_20m": str(value.mean_daily_net_twd),
            "mean_daily_terminal_realized_net_scope": (
                "executable_terminals_and_expiry_marks_only;"
                "open_positions_reported_in_separate_common_horizon_mark"
            ),
            "terminal_accounting_net_twd": str(value.total_net_twd),
            "common_horizon_open_gross_mark_pnl_twd": (
                None if aggregate is None else str(aggregate.gross_mark_pnl_twd)
            ),
            "common_horizon_open_incurred_cost_twd": str(
                open_valuation.incurred_open_execution_cost_twd
            ),
            "common_horizon_open_remaining_exit_cost_twd": (
                None
                if aggregate is None
                else str(aggregate.remaining_exit_cost_twd)
            ),
            "common_horizon_open_net_mark_pnl_twd": (
                None if aggregate is None else str(aggregate.net_mark_pnl_twd)
            ),
            "economic_ranking_net_twd": (
                None if economic_net is None else str(economic_net)
            ),
            "mean_daily_economic_ranking_net_twd_20m": (
                None
                if economic_net is None
                else str(economic_net / value.reporting_sessions)
            ),
            "common_horizon_priced_position_count": sum(
                row.priced for row in open_valuation.rows
            ),
            "common_horizon_unpriced_position_count": sum(
                not row.priced for row in open_valuation.rows
            ),
            "terminal_net_bp_of_executed_turnover": (
                None
                if value.terminal_net_bp_of_turnover is None
                else str(value.terminal_net_bp_of_turnover)
            ),
            "cost_scope": "modeled_direct_execution_costs_only",
            "open_positions_comparably_valued": aggregate is not None,
            "unmodeled_costs": {
                "overnight_financing_twd": {
                    "value": None,
                    "reason": "financing_rate_and_funding_contract_not_modeled",
                },
                "borrow_cost_twd": {
                    "value": None,
                    "reason": "borrow_availability_and_fee_not_modeled",
                },
                "futures_margin_opportunity_cost_twd": {
                    "value": None,
                    "reason": "margin_funding_cost_not_modeled",
                },
                "live_reject_latency_impact_twd": {
                    "value": None,
                    "reason": "requires_live_execution_telemetry",
                },
            },
            "entry_fill_truth": ENTRY_FILL_TRUTH,
            "all_risk_executable_send_numerator": value.hedge_priced_numerator,
            "all_risk_created_denominator": value.hedge_priced_denominator,
            "all_risk_executable_send_coverage_source": (
                value.hedge_pricing_coverage_source
            ),
        }
    )
    return record


def _shortlist_record(
    value: S1ShortlistResult | None,
    publication_decision: S1PublicationDecision,
) -> dict[str, object]:
    if value is None:
        return {
            "available": False,
            "reason": "publication_gate_economic_ranking_blocked",
            "blockers": sorted(
                {
                    *publication_decision.economic_ranking_blockers,
                    *publication_decision.shortlist_blockers,
                }
            ),
            "completion_champion": None,
            "net_champion": None,
            "selected": [],
            "second_selection_source": None,
            "pareto_frontier": [],
            "completion_ranking": [],
            "net_ranking": [],
        }
    return {
        "available": True,
        "reason": None,
        "blockers": [],
        "completion_champion": value.completion_champion.scenario_id,
        "net_champion": value.net_champion.scenario_id,
        "selected": [row.scenario_id for row in value.selected],
        "second_selection_source": value.second_selection_source,
        "pareto_frontier": [row.scenario_id for row in value.pareto_frontier],
        "completion_ranking": [row.scenario_id for row in value.completion_ranking],
        "net_ranking": [row.scenario_id for row in value.net_ranking],
    }


def _publication_decision_record(
    value: S1PublicationDecision,
) -> dict[str, object]:
    record = _json_safe(asdict(value))
    assert isinstance(record, dict)
    return record


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
        "economic_gate_decision_status_counts",
        "economic_gate_actual_send_status_counts",
        "economic_gate_dispatch_outcome_counts",
        "candidate_terminal_counts",
        "order_terminal_counts",
        "admission_status_counts",
        "request_counts",
        "execution_role_counts",
        "exit_physical_fill_reason_counts",
        "exit_desired_withdrawal_reason_counts",
    )
    counters: dict[str, Counter[str]] = {name: Counter() for name in counter_names}
    fill_latency: list[float] = []
    economic_margin_bp: list[float] = []
    economic_cost_twd: list[float] = []
    risk: dict[tuple[str, str], dict[str, object]] = {}
    for record in validated:
        for name in counter_names:
            counters[name].update(record[name])  # type: ignore[arg-type]
        fill_latency.extend(record["active_fill_latency_ms"])  # type: ignore[arg-type]
        economic_margin_bp.extend(  # type: ignore[arg-type]
            record["economic_gate_sent_expected_margin_bp"]
        )
        economic_cost_twd.extend(  # type: ignore[arg-type]
            record["economic_gate_sent_modeled_cost_twd"]
        )
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
    economic_margin_bp.sort()
    economic_cost_twd.sort()
    lookup_cells = sum(row.lookup_cells for row in summaries)
    lookup_supported = sum(row.lookup_supported_cells for row in summaries)
    lookup_unsupported = sum(row.lookup_unsupported_cells for row in summaries)
    sent_orders = sum(row.sent_entry_orders for row in summaries)
    makerfill_supported = sum(row.makerfill_supported_orders for row in summaries)
    active_fills = sum(row.actual_active_entry_fills for row in summaries)
    actual_cancelled = sum(row.actual_cancelled_orders for row in summaries)
    session_expired = sum(row.session_expired_orders for row in summaries)
    unknown_terminals = sent_orders - active_fills - actual_cancelled - session_expired
    reservation_attempts = sum(row.admission_checks for row in summaries)
    cap_blocked_attempts = sum(row.blocked_admission_checks for row in summaries)
    admitted_orders = reservation_attempts - cap_blocked_attempts
    if lookup_cells != 15_638 * 4:
        raise S1ProductionRunError("scenario common lookup denominator drifted")
    if lookup_supported + lookup_unsupported != lookup_cells:
        raise S1ProductionRunError("lookup supported/unsupported cells do not conserve")
    if any(
        row.decision_economic_gate_open + row.decision_economic_gate_closed
        != row.decision_economic_checks
        for row in summaries
    ):
        raise S1ProductionRunError("decision economic-gate funnel does not conserve")
    if any(
        row.actual_send_economic_gate_open + row.actual_send_economic_gate_closed
        != row.actual_send_economic_checks
        for row in summaries
    ):
        raise S1ProductionRunError("actual-send economic-gate funnel does not conserve")
    if active_fills > makerfill_supported or makerfill_supported > sent_orders:
        raise S1ProductionRunError("makerFill support/fill funnel does not conserve")
    if unknown_terminals < 0:
        raise S1ProductionRunError("entry order terminal funnel over-counted")
    if admitted_orders != sent_orders:
        raise S1ProductionRunError("capacity-admitted orders differ from sent orders")
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
        "economic_gate_sent_expected_margin_bp_count": len(economic_margin_bp),
        "economic_gate_sent_expected_margin_bp_p50": _quantile(
            economic_margin_bp,
            0.50,
        ),
        "economic_gate_sent_expected_margin_bp_p95": _quantile(
            economic_margin_bp,
            0.95,
        ),
        "economic_gate_sent_modeled_cost_twd_count": len(economic_cost_twd),
        "economic_gate_sent_modeled_cost_twd_p50": _quantile(
            economic_cost_twd,
            0.50,
        ),
        "economic_gate_sent_modeled_cost_twd_p95": _quantile(
            economic_cost_twd,
            0.95,
        ),
        "risk_groups": risk_rows,
        "common_lookup_cells": lookup_cells,
        "lookup_supported_cells": lookup_supported,
        "lookup_unsupported_cells": lookup_unsupported,
        "lookup_support_rate": _safe_rate(lookup_supported, lookup_cells),
        "decision_economic_checks": sum(
            row.decision_economic_checks for row in summaries
        ),
        "decision_economic_gate_open": sum(
            row.decision_economic_gate_open for row in summaries
        ),
        "decision_economic_gate_closed": sum(
            row.decision_economic_gate_closed for row in summaries
        ),
        "actual_send_economic_checks": sum(
            row.actual_send_economic_checks for row in summaries
        ),
        "actual_send_economic_gate_open": sum(
            row.actual_send_economic_gate_open for row in summaries
        ),
        "actual_send_economic_gate_closed": sum(
            row.actual_send_economic_gate_closed for row in summaries
        ),
        "actual_send_not_sent_after_gate_pass": sum(
            row.actual_send_not_sent_after_gate_pass for row in summaries
        ),
        "sent_entry_orders": sent_orders,
        "makerfill_supported_orders": makerfill_supported,
        "actual_active_entry_fills": active_fills,
        "candidate_intents": sum(row.candidate_intents for row in summaries),
        "blocked_candidate_intents": sum(
            row.blocked_candidate_intents for row in summaries
        ),
        "admitted_orders": admitted_orders,
        "makerfill_unsupported_orders": sent_orders - makerfill_supported,
        "actual_cancelled_orders": actual_cancelled,
        "session_expired_orders": session_expired,
        "unknown_terminal_orders": unknown_terminals,
        "makerfill_support_rate_of_sent": _safe_rate(
            makerfill_supported,
            sent_orders,
        ),
        "actual_active_fill_rate_of_supported": _safe_rate(
            active_fills,
            makerfill_supported,
        ),
        "actual_active_fill_rate_of_sent": _safe_rate(
            active_fills,
            sent_orders,
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
        "reservation_attempts": reservation_attempts,
        "cap_blocked_attempts": cap_blocked_attempts,
        "suppressed_redundant_blocked_admission_probes": sum(
            row.suppressed_redundant_blocked_admission_probes for row in summaries
        ),
        "reused_incidental_cap_blocked_entry_states": sum(
            row.reused_incidental_cap_blocked_entry_states for row in summaries
        ),
        "carry_notional_days_twd": sum(
            int(record["carry_out_notional_twd"]) for record in validated
        ),
        "final_carry_positions": int(validated[-1]["carry_out_positions"]),
        "final_carry_notional_twd": int(validated[-1]["carry_out_notional_twd"]),
        "final_naked_unresolved_positions": int(
            validated[-1]["naked_unresolved_positions"]
        ),
        "final_naked_unresolved_notional_twd": int(
            validated[-1]["naked_unresolved_notional_twd"]
        ),
        "naked_unresolved_positions_max": max(
            int(record["naked_unresolved_positions"]) for record in validated
        ),
        "naked_unresolved_notional_twd_max": max(
            int(record["naked_unresolved_notional_twd"]) for record in validated
        ),
        "exit_cutoff_sessions": sum(
            bool(record["exit_cutoff_applied"]) for record in validated
        ),
        "exit_drain_barrier_sessions": sum(
            bool(record["exit_drain_barrier_applied"]) for record in validated
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
    open_valuations: Mapping[str, S1CommonHorizonValuationResult],
    shortlist: S1ShortlistResult | None,
    publication_decision: S1PublicationDecision,
    diagnostics: Mapping[str, Mapping[str, object]],
    entry_month: Mapping[str, Sequence[Mapping[str, object]]],
) -> str:
    if set(open_valuations) != set(SCENARIO_IDS):
        raise S1ProductionRunError("report open-valuation policy set differs")
    if (shortlist is None) == publication_decision.economic_ranking_allowed:
        raise S1ProductionRunError(
            "shortlist availability disagrees with publication gate"
        )
    if shortlist is not None and not publication_decision.s2_shortlist_allowed:
        raise S1ProductionRunError("published shortlist is not authorized by the gate")
    common_population_sha256 = _run_config_common_population_sha256(run_config)
    if shortlist is None:
        conclusion = (
            "Publication gate 允許描述性統計，但未授權 economic ranking、Pareto 與 S2 "
            "shortlist；本報告不發布 completion/net champion。"
        )
    else:
        conclusion = (
            f"Completion champion：`{shortlist.completion_champion.scenario_id}`；"
            f"net champion：`{shortlist.net_champion.scenario_id}`；S2 shortlist："
            + ", ".join(f"`{row.scenario_id}`" for row in shortlist.selected)
            + "。排名只涵蓋六組 deployment-eligible scenarios；ungated control 僅作描述對照。"
        )
    lines = [
        "# S1 Spot Bid 七組 scenario：20M 聯合事件回放",
        "",
        "日期：2026-09-01",
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
        conclusion,
        "",
        "## 七組主比較",
        "",
        (
            "| scenario | approx entry fills | same-day | approx completion 20M | "
            "final open | final carry committed TWD | priced/unpriced | terminal gross TWD | modeled direct cost TWD | "
            "terminal net TWD | mean daily terminal realized net TWD | open gross mark TWD | "
            "open incurred cost TWD | remaining exit cost TWD | open net mark TWD | "
            "economic ranking net TWD | mean daily marked net TWD |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ]
    for value in performance:
        valuation = open_valuations[value.scenario_id]
        aggregate = valuation.aggregate
        economic_net = (
            None
            if aggregate is None
            else value.total_net_twd + aggregate.net_mark_pnl_twd
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    value.scenario_id,
                    str(value.entry_fill_approximate),
                    str(value.same_day_exit_maker_flat),
                    _percent(
                        value.approx_screen_completion_numerator,
                        value.approx_screen_completion_denominator,
                    ),
                    str(value.open_or_unresolved),
                    str(diagnostics[value.scenario_id]["final_carry_notional_twd"]),
                    (
                        f"{sum(row.priced for row in valuation.rows)}/"
                        f"{sum(not row.priced for row in valuation.rows)}"
                    ),
                    _money(value.terminal_gross_pnl_twd),
                    _money(value.terminal_modeled_direct_cost_twd),
                    _money(value.total_net_twd),
                    _money(value.mean_daily_net_twd),
                    _optional_money(
                        None if aggregate is None else aggregate.gross_mark_pnl_twd
                    ),
                    _money(valuation.incurred_open_execution_cost_twd),
                    _optional_money(
                        None
                        if aggregate is None
                        else aggregate.remaining_exit_cost_twd
                    ),
                    _optional_money(
                        None if aggregate is None else aggregate.net_mark_pnl_twd
                    ),
                    _optional_money(economic_net),
                    _optional_money(
                        None
                        if economic_net is None
                        else economic_net / value.reporting_sessions
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "`terminal net` 只涵蓋 executable terminals 與 expiry marks；"
                "`open net mark` 是最終仍未平 paired positions 使用 2026-08-13 13:20 "
                "共同 raw-book 時點的全量 Spot sell／Future buy executable VWAP，"
                "並扣除已發生與剩餘 exit costs。兩者相加才是 economic ranking net。"
            ),
            (
                "`mean daily marked net` 是 economic ranking net 除以 71 reporting sessions；"
                "它是 marked economics，不是 realized cashflow。任何 final open 缺合法或足量 book，"
                "該 scenario 會顯示 `null`，publication gate 會封鎖 economic ranking 與 S2 shortlist。"
            ),
            (
                "Gross、commission、tax、modeled direct cost、net 與 turnover 均由 executed legs "
                "重建並通過 conservation gate；未建模成本不以 0 代填。"
            ),
            "",
            "## Common-horizon valuation 可用性",
            "",
            "| scenario | final open | priced | unpriced | unpriced reasons |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for policy_id in SCENARIO_IDS:
        valuation = open_valuations[policy_id]
        reasons = Counter(
            row.unpriced_reason
            for row in valuation.rows
            if row.unpriced_reason is not None
        )
        reason_text = (
            "none"
            if not reasons
            else ", ".join(
                f"`{reason}`={count}" for reason, count in sorted(reasons.items())
            )
        )
        lines.append(
            f"| {policy_id} | {len(valuation.rows)} | "
            f"{sum(row.priced for row in valuation.rows)} | "
            f"{sum(not row.priced for row in valuation.rows)} | {reason_text} |"
        )
    lines.extend(
        [
            "",
            "## 查表、成本 gate 與成交漏斗",
            "",
            (
                "| scenario | cost horizon/floor bp | lookup supported/common | decision gate open/checks | "
                "actual-send gate open/checks | candidates | ever-capacity-blocked candidates | "
                "reservation attempts | cap-blocked attempts | admitted | sent | "
                "makerFill supported/sent | fills/supported | fills/sent | "
                "expected margin bp p50/p95 | modeled cost TWD p50/p95 |"
            ),
            (
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|"
            ),
        ]
    )
    for policy_id in SCENARIO_IDS:
        row = diagnostics[policy_id]
        definition = SCENARIO_BY_ID[policy_id]
        horizon_floor = (
            "ungated"
            if definition.safety_floor_bp is None
            else f"{definition.cost_horizon}/{definition.safety_floor_bp:g}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    policy_id,
                    horizon_floor,
                    f"{row['lookup_supported_cells']}/{row['common_lookup_cells']}",
                    f"{row['decision_economic_gate_open']}/{row['decision_economic_checks']}",
                    f"{row['actual_send_economic_gate_open']}/{row['actual_send_economic_checks']}",
                    str(row["candidate_intents"]),
                    str(row["blocked_candidate_intents"]),
                    str(row["reservation_attempts"]),
                    str(row["cap_blocked_attempts"]),
                    str(row["admitted_orders"]),
                    str(row["sent_entry_orders"]),
                    f"{row['makerfill_supported_orders']}/{row['sent_entry_orders']}",
                    f"{row['actual_active_entry_fills']}/{row['makerfill_supported_orders']}",
                    f"{row['actual_active_entry_fills']}/{row['sent_entry_orders']}",
                    (
                        f"{_number(row['economic_gate_sent_expected_margin_bp_p50'])}/"
                        f"{_number(row['economic_gate_sent_expected_margin_bp_p95'])}"
                    ),
                    (
                        f"{_number(row['economic_gate_sent_modeled_cost_twd_p50'])}/"
                        f"{_number(row['economic_gate_sent_modeled_cost_twd_p95'])}"
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Target location/rank 診斷",
            "",
            "| scenario | exact target-rank counts | makerFill assessment denominator |",
            "|---|---|---:|",
        ]
    )
    for policy_id in SCENARIO_IDS:
        target_counts = _nonnegative_count_items(
            diagnostics[policy_id]["target_rank_counts"],
            f"{policy_id} target_rank_counts",
        )
        lines.append(
            f"| {policy_id} | {_count_items_text(target_counts)} | "
            f"{sum(count for _, count in target_counts)} |"
        )
    lines.extend(
        [
            "",
            (
                "Target-rank 分母是 makerFill assessments；它不等於 sent orders、"
                "makerFill-supported orders 或 active fills，這些分母仍在上一表分開列示。"
            ),
            "",
            "## Exit pre-fill risk guard 診斷",
            "",
            "| scenario | cutoff/barrier sessions | desired-withdrawal reasons | withdrawals |",
            "|---|---:|---|---:|",
        ]
    )
    for policy_id in SCENARIO_IDS:
        withdrawal_counts = _nonnegative_count_items(
            diagnostics[policy_id]["exit_desired_withdrawal_reason_counts"],
            f"{policy_id} exit_desired_withdrawal_reason_counts",
        )
        lines.append(
            f"| {policy_id} | {diagnostics[policy_id]['exit_cutoff_sessions']}/"
            f"{diagnostics[policy_id]['exit_drain_barrier_sessions']} | "
            f"{_count_items_text(withdrawal_counts)} | "
            f"{sum(count for _, count in withdrawal_counts)} |"
        )
    lines.extend(
        [
            "",
            (
                "`gate:future_*` 表示被動 Spot exit 因當下 Future buy 不可完整執行而撤回；"
                "`safety_cutoff` 是 13:19:45 的固定事前撤單。actual cancel effect 前的真實 "
                "Spot fill 仍成立並走獨立 B6 hedge／rollback；任何最終裸腿仍 fail-closed。"
            ),
            "",
            "## 執行與容量診斷",
            "",
            (
                "| scenario | fill latency p50/p95 ms | all-risk sent/created | "
                "spot req/day | future req/day | rolling peak S/F | cap peak G/P TWD | "
                "carry notional-days | final carry TWD | naked max TWD |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_id in SCENARIO_IDS:
        row = diagnostics[policy_id]
        lines.append(
            "| "
            + " | ".join(
                (
                    policy_id,
                    (
                        f"{_number(row['active_fill_latency_ms_p50'])}/"
                        f"{_number(row['active_fill_latency_ms_p95'])}"
                    ),
                    _percent(
                        next(
                            value.hedge_priced_numerator
                            for value in performance
                            if value.scenario_id == policy_id
                        ),
                        next(
                            value.hedge_priced_denominator
                            for value in performance
                            if value.scenario_id == policy_id
                        ),
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
            "## Hedge／rollback delay 與 slippage",
            "",
            (
                "| scenario | stage:risk | on-time | delayed | timeout | actual-send/created | "
                "arrival reference available/denominator | delay ms p50/p95/max | "
                "adverse slippage bp p50/p95/max | slippage samples/actual-send |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_id in SCENARIO_IDS:
        groups = diagnostics[policy_id]["risk_groups"]
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
            raise S1ProductionRunError(f"{policy_id} risk_groups must be a sequence")
        if not groups:
            lines.append(
                f"| {policy_id} | none observed | 0 | 0 | 0 | 0/0 | 0/0 | "
                "null/null/null | null/null/null | 0/0 |"
            )
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                raise S1ProductionRunError(f"{policy_id} risk group must be a mapping")
            lines.append(
                "| "
                + " | ".join(
                    (
                        policy_id,
                        f"{group['stage']}:{group['risk_kind']}",
                        str(group["on_time"]),
                        str(group["delayed"]),
                        str(group["timeout"]),
                        f"{group['actual_send']}/{group['created']}",
                        (
                            f"{group['arrival_reference_available']}/"
                            f"{group['arrival_reference_denominator']}"
                        ),
                        (
                            f"{_number(group['delay_ms_p50'])}/"
                            f"{_number(group['delay_ms_p95'])}/"
                            f"{_number(group['delay_ms_max'])}"
                        ),
                        (
                            f"{_number(group['adverse_slippage_bp_p50'])}/"
                            f"{_number(group['adverse_slippage_bp_p95'])}/"
                            f"{_number(group['adverse_slippage_bp_max'])}"
                        ),
                        (
                            f"{group['adverse_slippage_sample_count']}/"
                            f"{group['actual_send']}"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "Risk 分母分開解讀：on-time + delayed = actual-send；actual-send + timeout = "
                "created；arrival-reference 與 adverse-slippage sample 各自使用表內分母，"
                "不可拿 created 代替。"
            ),
            "",
            "## Global committed capacity bucket peaks",
            "",
            (
                "| scenario | working unfilled | entry partial | hedge pending | paired open | "
                "exit in progress | total committed |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    capacity_buckets = (
        "working_unfilled",
        "entry_partial",
        "hedge_pending",
        "paired_open",
        "exit_in_progress",
        "total_committed_notional_twd",
    )
    for policy_id in SCENARIO_IDS:
        bucket_items = dict(
            _nonnegative_count_items(
                diagnostics[policy_id]["global_committed_bucket_peaks_twd"],
                f"{policy_id} global_committed_bucket_peaks_twd",
            )
        )
        if set(bucket_items) != set(capacity_buckets):
            raise S1ProductionRunError(
                f"{policy_id} committed capacity bucket set drifted"
            )
        lines.append(
            "| "
            + " | ".join(
                (policy_id, *(str(bucket_items[name]) for name in capacity_buckets))
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "各 bucket 是 71 sessions 內各自的 global peak，未必發生在同一時點，"
                "因此不可相加；`total committed` 是獨立觀測到的總承諾峰值。"
            ),
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
    for policy_id in SCENARIO_IDS:
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
            "## Publication gate",
            "",
            f"- Descriptive publication allowed：`{str(publication_decision.descriptive_publication_allowed).lower()}`。",
            f"- Completion ranking allowed：`{str(publication_decision.completion_ranking_allowed).lower()}`。",
            f"- Economic ranking allowed：`{str(publication_decision.economic_ranking_allowed).lower()}`。",
            f"- S2 shortlist allowed：`{str(publication_decision.s2_shortlist_allowed).lower()}`。",
            "- Economic blockers："
            + _markdown_values(publication_decision.economic_ranking_blockers),
            "- S2-only blockers："
            + _markdown_values(publication_decision.shortlist_blockers),
            "- Unavailable-cost disclosures：",
        ]
    )
    lines.extend(
        f"  - `{value}`" for value in publication_decision.unavailable_cost_disclosures
    )
    lines.extend(["", "## 排名與可重現性", ""])
    if shortlist is None:
        lines.append(
            "- Completion/net ranking、Pareto frontier 與 S2 shortlist 未發布；詳見 publication gate blockers。"
        )
    else:
        lines.extend(
            [
                "- Completion ranking："
                + " → ".join(
                    f"`{row.scenario_id}`" for row in shortlist.completion_ranking
                ),
                "- Net ranking："
                + " → ".join(f"`{row.scenario_id}`" for row in shortlist.net_ranking),
                "- Pareto frontier："
                + ", ".join(
                    f"`{row.scenario_id}`" for row in shortlist.pareto_frontier
                ),
            ]
        )
    lines.extend(
        [
            f"- Run config SHA-256：`{run_config_sha256}`。",
            f"- Source commit：`{run_config['source_commit']}`。",
            f"- Common population SHA-256：`{common_population_sha256}`。",
            (
                "- Scenario spec 已凍結為七組 cost-aware grid：Q95/Q80/Q50、"
                "C0/C2/C3 與 fixed20；unsupported lookup cells 保留共同分母並明確 no-trade。"
            ),
            (
                "- 每筆 entry 在 actual-new send 同時鎖定絕對 Spot Bid entry 與 Spot Ask exit "
                "price/tick；後續 Future Ask 更新不會移動 exit target。"
            ),
            (
                "- 每個 policy/date partition 都有 deterministic gzip JSONL、compact capacity "
                "checkpoint、SQLite exact identity registry 與 atomic complete marker；最終 verifier "
                "重播 decision/actual-send economic estimates、accounting、capacity 與 cross-ledger links。"
            ),
            "",
            "## 限制",
            "",
            (
                "- Spot Bid entry 看不到 own quantity、partial fill、cancel ACK 與 joint volume allocation；"
                "S5 exact calibration 前只能稱 approximate screen。"
            ),
            (
                "- 目前未計 overnight financing、spot borrow、futures margin opportunity cost "
                "與 live reject/latency impact；不能把未建模成本當 0。"
            ),
            "- 本結果使用 development-selected lookup；2026-08-14 起 protected forward 未讀。",
            (
                "- Frozen grid 與 unsupported=no-trade 已足以重啟可重現研究；部署 baseline 仍須完成 "
                "exact-entry calibration、protected forward、風控與實盤 shadow 驗證。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_values(values: Sequence[str]) -> str:
    return "none" if not values else ", ".join(f"`{value}`" for value in values)


def _nonnegative_count_items(
    value: object,
    label: str,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise S1ProductionRunError(f"{label} must be a mapping")
    items: list[tuple[str, int]] = []
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise S1ProductionRunError(f"{label} keys must be non-empty strings")
        if type(count) is not int or count < 0:
            raise S1ProductionRunError(f"{label} counts must be non-negative integers")
        items.append((key, count))
    return tuple(sorted(items))


def _count_items_text(items: Sequence[tuple[str, int]]) -> str:
    return (
        "none observed"
        if not items
        else ", ".join(f"`{key}`={count}" for key, count in items)
    )


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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("rate counts must be integers")
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise S1ProductionRunError("rate counts are invalid")
    return None if denominator == 0 else numerator / denominator


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "null"
    return f"{100.0 * numerator / denominator:.4f}% ({numerator}/{denominator})"


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _optional_money(value: Decimal | None) -> str:
    return "null" if value is None else _money(value)


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

    expected_partition_count = len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS)
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
        for policy_id in SCENARIO_IDS:
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


def _write_verification_attestation(
    config: S1ProductionConfig,
    result: S1ProductionResult,
) -> None:
    """Persist a deterministic attestation only after deep verification succeeds."""

    if config.verify_completed_input_content is not True:
        raise S1ProductionRunError(
            "verification attestation requires verified input content"
        )
    if result.output_root != config.output_root or result.report_path != config.report_path:
        raise S1ProductionRunError("verification result paths differ from request")
    expected_partition_count = len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS)
    if (
        result.executed_partitions != 0
        or result.resumed_partitions != expected_partition_count
    ):
        raise S1ProductionRunError(
            "verification attestation requires a complete read-only replay"
        )

    complete_path = config.output_root / "complete.json"
    run_config_path = config.output_root / "run_config.json"
    results_path = config.output_root / "results.json"
    critical_paths = (
        complete_path,
        run_config_path,
        results_path,
        config.report_path,
    )
    for path, label in (
        (complete_path, "complete marker"),
        (run_config_path, "run config"),
        (results_path, "results"),
        (config.report_path, "report"),
    ):
        _require_real_file(path, label)
    before = _file_stats(critical_paths)
    complete = _read_canonical_json_object(complete_path)
    if (
        complete.get("schema_version") != FINAL_BUNDLE_SCHEMA_VERSION
        or complete.get("complete") is not True
    ):
        raise S1ProductionRunError(
            "verified complete marker changed before attestation"
        )
    run_config = _read_canonical_json_object(run_config_path)
    bundle_source_commit = _recorded_source_commit(
        run_config,
        expected=config.source_commit,
    )
    verifier_source_commit = _git_source_commit(
        expected=result.verifier_source_commit
    )
    if verifier_source_commit != bundle_source_commit:
        raise S1ProductionRunError(
            "verifier source commit differs from bundle source commit"
        )
    complete_sha256 = _sha256_file(complete_path)
    run_config_sha256 = _sha256_file(run_config_path)
    results_sha256 = _sha256_file(results_path)
    report_sha256 = _sha256_file(config.report_path)
    partition_count = complete.get("partition_count")
    if (
        complete_sha256 != result.complete_sha256
        or run_config_sha256 != result.run_config_sha256
        or complete.get("run_config_sha256") != run_config_sha256
        or complete.get("results_sha256") != results_sha256
        or complete.get("report_sha256") != report_sha256
    ):
        raise S1ProductionRunError(
            "verified result/artifact hashes changed before attestation"
        )
    if type(partition_count) is not int or partition_count != expected_partition_count:
        raise S1ProductionRunError(
            "verified partition count changed before attestation"
        )
    if before != _file_stats(critical_paths):
        raise S1ProductionRunError(
            "verified artifacts changed while attestation was built"
        )
    record = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "verification_status": "verified",
        "verify_inputs": True,
        "complete_sha256": complete_sha256,
        "run_config_sha256": run_config_sha256,
        "bundle_source_commit": bundle_source_commit,
        "verifier_source_commit": verifier_source_commit,
        "partition_count": partition_count,
        "results_sha256": results_sha256,
        "report_sha256": report_sha256,
    }
    _atomic_write_canonical_json(
        config.output_root / VERIFICATION_FILENAME,
        record,
        replace_exact=True,
    )


def _require_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise S1ProductionRunError(f"{label} must be an existing real directory")


def _require_real_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise S1ProductionRunError(f"{label} must be an existing real file")


def _ensure_run_config(
    path: Path,
    record: Mapping[str, object],
    *,
    require_existing_exact: bool = False,
) -> str:
    if not isinstance(require_existing_exact, bool):
        raise TypeError("require_existing_exact must be boolean")
    payload = _canonical_json_bytes(record)
    expected_sha = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise S1ProductionRunError("existing run_config.json differs")
        return expected_sha
    if require_existing_exact:
        raise S1ProductionRunError(
            "verification requires existing exact run_config.json"
        )
    _atomic_write_bytes(path, payload)
    return expected_sha


def _publish_or_verify_canonical_json(
    path: Path,
    value: object,
    *,
    verification_only: bool,
) -> None:
    _publish_or_verify_bytes(
        path,
        _canonical_json_bytes(value),
        verification_only=verification_only,
    )


def _publish_or_verify_text(
    path: Path,
    value: str,
    *,
    verification_only: bool,
) -> None:
    if not isinstance(value, str):
        raise TypeError("text artifact must be a string")
    _publish_or_verify_bytes(
        path,
        value.encode("utf-8"),
        verification_only=verification_only,
    )


def _publish_or_verify_bytes(
    path: Path,
    payload: bytes,
    *,
    verification_only: bool,
) -> None:
    if not isinstance(verification_only, bool):
        raise TypeError("verification_only must be boolean")
    if verification_only:
        _require_real_file(path, "verification target artifact")
        if path.read_bytes() != payload:
            raise S1ProductionRunError(
                f"deep verification reconstructed different artifact bytes: {path}"
            )
        return
    _atomic_write_bytes(path, payload, accept_existing_exact=True)


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
    if ".." in raw.parts:
        raise S1ProductionRunError("input path contains parent traversal")
    if scope == "absolute":
        if not raw.is_absolute():
            raise S1ProductionRunError("absolute input path is not absolute")
        return raw
    if raw.is_absolute():
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


def _print_runtime_progress(*, event: str, **fields: object) -> None:
    if not isinstance(event, str) or not event:
        raise ValueError("progress event must be a non-empty string")
    print(
        json.dumps(
            {
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _print_progress(
    *,
    date: str,
    policy_id: str,
    completed: int,
    total: int,
    elapsed_seconds: float,
) -> None:
    _print_runtime_progress(
        event="s1_partition_complete",
        date=date,
        policy_id=policy_id,
        completed=completed,
        total=total,
        elapsed_seconds=elapsed_seconds,
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
        json.dumps(_bundle_complete_event_record(result), sort_keys=True),
        flush=True,
    )
    return 0


def _bundle_complete_event_record(result: S1ProductionResult) -> dict[str, object]:
    shortlist = result.shortlist
    return {
        "event": "s1_bundle_complete",
        "output_root": str(result.output_root),
        "report_path": str(result.report_path),
        "run_config_sha256": result.run_config_sha256,
        "complete_sha256": result.complete_sha256,
        "shortlist_available": shortlist is not None,
        "completion_champion": (
            None if shortlist is None else shortlist.completion_champion.scenario_id
        ),
        "net_champion": (
            None if shortlist is None else shortlist.net_champion.scenario_id
        ),
        "economic_ranking_allowed": (
            result.publication_decision.economic_ranking_allowed
        ),
        "s2_shortlist_allowed": result.publication_decision.s2_shortlist_allowed,
        "economic_ranking_blockers": list(
            result.publication_decision.economic_ranking_blockers
        ),
        "shortlist_blockers": list(result.publication_decision.shortlist_blockers),
    }


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COMMON_POPULATION_SCHEMA_VERSION",
    "DAILY_METRICS_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REPORT_PATH",
    "PRODUCTION_RUNNER_VERSION",
    "RESULTS_SCHEMA_VERSION",
    "S1_DEVELOPMENT_DATES",
    "VERIFICATION_FILENAME",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFIER_VERSION",
    "S1ProductionConfig",
    "S1ProductionResult",
    "S1ProductionRunError",
    "run_s1_production_bundle",
    "verify_s1_production_bundle",
]
