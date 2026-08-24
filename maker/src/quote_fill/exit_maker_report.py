"""Audited reports for partitioned maker/taker exit replay.

The report keeps three deliberately different units apart:

* a policy trial is one established entry alias x D-1 exit rule x exit route;
* a policy alias is one candidate-order label attached to such a trial; and
* a raw candidate is one canonical physical exit order, deduplicated by its
  rule-free ``exit_raw_candidate_fact_id`` inside each requested q action.

Rows for different q aliases, exit rules, and exit routes are alternative
policies.  They are useful side-by-side but are not additive portfolio
observations.  Cancellation statistics are request-only diagnostics because
the replay has no exchange cancel acknowledgement feed.

The 19 bp table is an explicitly analysis-only non-price cost sensitivity.
Actual maker and delayed-taker prices are already in gross cycle PnL and are
not subtracted a second time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT


EXPECTED_HEDGE_DELAY_NS = 50_000_000
DEFAULT_SESSION_CALENDAR_PATH = (
    MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
)
DEFAULT_NON_PRICE_COST_BP = 19.0
REPORT_VERSION = "exit_maker_partition_report_v1"
NON_PRICE_COST_SCOPE = "non_price_roundtrip_fees_tax_commission_only"

_EXIT_ARTIFACTS = (
    "exit_maker_policy_support.parquet",
    "exit_maker_observations.parquet",
    "exit_maker_transitions.parquet",
    "exit_maker_candidate_aliases.parquet",
    "exit_maker_raw_candidate_facts.parquet",
    "exit_maker_position_policy_facts.parquet",
    "exit_maker_audit.parquet",
)
_LOADED_EXIT_ARTIFACTS = (
    "exit_maker_policy_support.parquet",
    "exit_maker_candidate_aliases.parquet",
    "exit_maker_raw_candidate_facts.parquet",
    "exit_maker_position_policy_facts.parquet",
)
_ENTRY_ACTION_ARTIFACT = "execution_action_facts.parquet"
_ENTRY_EXIT_ARTIFACT = "exit_facts.parquet"

DAILY_POLICY_KEY = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "boundary_quantile",
    "exit_rule_id",
    "exit_route",
]
PRODUCT_POLICY_KEY = [
    "ValueCode",
    "entry_route",
    "boundary_quantile",
    "exit_rule_id",
    "exit_route",
]

_SUPPORT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "position_status",
    "admission_status",
    "raw_candidate_count",
}
_ALIAS_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "exit_raw_candidate_fact_id",
    "exit_candidate_alias_id",
    "cancel_required",
    "any_fill",
    "full_fill",
    "partial_fill",
    "exit_hedge_status",
}
_RAW_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_raw_order_fact_id",
    "exit_route",
    "exit_raw_candidate_fact_id",
    "spread_pair_epoch",
    "target_price_tick",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
    "cancel_required",
    "cancel_ack_observed",
    "any_fill",
    "full_fill",
    "partial_fill",
    "exit_hedge_status",
}
_POSITION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "position_status",
    "position_established_ns",
    "nominal_instant_cancel_v0_branch",
    "branch_status",
    "terminal_outcome",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
    "exit_hedge_status",
    "exit_hedge_signed_latency_slippage_bp",
    "exit_hedge_signed_depth_slippage_bp",
    "exit_hedge_signed_total_slippage_bp",
    "oco_active_sibling_cancel_count",
    "prior_unacked_cancel_count_before_winner",
    "cancel_ack_observed",
    "joint_volume_allocated",
}
_ACTION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "parameter_version",
    "boundary_quantile",
    "lookup_action_id",
    "raw_order_fact_id",
    "policy_generation_id",
    "full_fill",
    "entry_hedge_status",
    "submit_recv_time_ns",
    "entry_spot_price",
    "entry_hedge_contract_size_shares",
    "entry_hedge_signed_latency_slippage_bp",
    "entry_hedge_signed_depth_slippage_bp",
    "entry_hedge_signed_total_slippage_bp",
    "entry_hedge_decision_book_age_ms",
}
_TAKER_EXIT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "policy_generation_id",
    "exit_rule_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "branch_status",
    "terminal_outcome",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
    "exit_added_latency_ns",
}

_NOMINAL_SAME_DAY = "flat_same_day"
_NOMINAL_CARRY = {
    "carry_at_eod_cancel_unconfirmed",
    "carry_at_eod_no_admission",
    "no_fill_before_cancel_request",
}
_STRICT_CARRY = {
    "carry_at_eod_cancel_unconfirmed",
    "carry_at_eod_no_admission",
}


@dataclass(frozen=True)
class ExitMakerPartitionInputs:
    """Validated exit partitions and their immutable entry-side sources."""

    policy_support: pl.DataFrame
    candidate_aliases: pl.DataFrame
    raw_candidate_facts: pl.DataFrame
    position_policy_facts: pl.DataFrame
    action_facts: pl.DataFrame
    taker_exit_facts: pl.DataFrame
    coverage: pl.DataFrame
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ExitMakerReport:
    """Materialized report tables with explicit sampling-unit semantics."""

    coverage: pl.DataFrame
    daily_policy: pl.DataFrame
    product_policy: pl.DataFrame
    daily_strict_branch: pl.DataFrame
    product_strict_branch: pl.DataFrame
    policy_threshold_lineage: pl.DataFrame
    matched_pairs: pl.DataFrame
    matched_summary: pl.DataFrame
    metadata: Mapping[str, object]


def load_exit_maker_partition_inputs(
    exit_maker_root: Path,
    entry_execution_root: Path,
    *,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
    expected_hedge_delay_ns: int = EXPECTED_HEDGE_DELAY_NS,
    session_calendar: Sequence[str] | None = None,
    require_root_manifest: bool = True,
) -> ExitMakerPartitionInputs:
    """Load complete product-days and validate marker/source lineage.

    Session coverage is date based.  A known missing product-day can remain in
    the coverage grid when ``require_balanced_product_days`` is false, but an
    exact 60 distinct sessions is still required by default.
    """

    root = Path(exit_maker_root)
    entry_root = Path(entry_execution_root)
    if (
        isinstance(expected_hedge_delay_ns, bool)
        or not isinstance(expected_hedge_delay_ns, int)
        or expected_hedge_delay_ns != EXPECTED_HEDGE_DELAY_NS
    ):
        raise ValueError(
            f"formal entry/exit hedge delay must be {EXPECTED_HEDGE_DELAY_NS} ns"
        )
    if not isinstance(require_root_manifest, bool):
        raise TypeError("require_root_manifest must be boolean")
    _validate_session_request(sessions)
    requested_products = _normalise_products(value_codes)
    records = _discover_markers(root, requested_products)
    if not records:
        raise FileNotFoundError(f"no complete exit-maker partitions under {root}")

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
        (str(record["Date"]), str(record["ValueCode"])): record for record in selected
    }
    if len(selected_by_key) != len(selected):
        raise ValueError("duplicate complete exit-maker marker for a product-day")

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

    loaded_exit: dict[str, list[pl.DataFrame]] = {
        name: [] for name in _LOADED_EXIT_ARTIFACTS
    }
    actions: list[pl.DataFrame] = []
    exits: list[pl.DataFrame] = []
    runner_hashes: set[str] = set()
    runner_payloads: set[str] = set()
    entry_config_hashes: set[str] = set()
    taker_exit_grid_values: set[int] = set()
    taker_exit_max_book_age_values: set[int] = set()
    runner_versions: set[str] = set()
    entry_hedge_delay_values: set[int] = set()
    exit_rule_id_sets: set[tuple[str, ...]] = set()
    exit_route_sets: set[tuple[str, ...]] = set()
    for record in selected:
        payload = record["payload"]
        assert isinstance(payload, dict)
        partition = Path(record["partition"])
        config = _validate_exit_marker(
            payload,
            partition,
            validate_hashes=validate_hashes,
        )
        runner = config["runner"]
        source = config["source"]
        assert isinstance(runner, dict) and isinstance(source, dict)
        runner_hashes.add(str(payload["runner_config_sha256"]))
        runner_payloads.add(_canonical_json(runner))
        runner_versions.add(str(payload.get("runner_version")))
        hedge_delay = int(runner.get("hedge_delay_ns", -1))
        if hedge_delay != expected_hedge_delay_ns:
            raise ValueError(
                f"expected {expected_hedge_delay_ns} ns exit hedge delay, got {hedge_delay}"
            )
        exit_rule_id_sets.add(
            tuple(sorted(str(value) for value in runner.get("exit_rule_ids", ())))
        )
        exit_route_sets.add(
            tuple(sorted(str(value) for value in runner.get("routes", ())))
        )
        for name in _LOADED_EXIT_ARTIFACTS:
            loaded_exit[name].append(pl.read_parquet(partition / name))

        date = str(record["Date"])
        value_code = str(record["ValueCode"])
        entry_partition = entry_root / f"Date={date}" / f"ValueCode={value_code}"
        action, taker_exit, entry_config_hash, entry_config = _load_and_validate_entry_source(
            entry_partition,
            source,
            date=date,
            value_code=value_code,
            validate_hashes=validate_hashes,
        )
        actions.append(action)
        exits.append(taker_exit)
        entry_config_hashes.add(entry_config_hash)
        entry_hedge_delay = entry_config.get("hedge_delay_ns")
        if (
            isinstance(entry_hedge_delay, bool)
            or not isinstance(entry_hedge_delay, int)
            or entry_hedge_delay != expected_hedge_delay_ns
        ):
            raise ValueError(
                "entry action marker hedge delay disagrees with the formal "
                f"{expected_hedge_delay_ns} ns contract"
            )
        entry_hedge_delay_values.add(entry_hedge_delay)
        actions[-1] = _validate_entry_action_execution_contract(
            action,
            expected_hedge_delay_ns=expected_hedge_delay_ns,
        )
        exit_path = entry_config.get("exit_path")
        if not isinstance(exit_path, dict):
            raise ValueError("entry runner config is missing the T/T exit_path")
        grid_ns = exit_path.get("grid_ns")
        max_book_age_ns = exit_path.get("max_book_age_ns")
        if (
            isinstance(grid_ns, bool)
            or not isinstance(grid_ns, int)
            or grid_ns <= 0
            or isinstance(max_book_age_ns, bool)
            or not isinstance(max_book_age_ns, int)
            or max_book_age_ns < 0
        ):
            raise ValueError("entry T/T exit grid/book-age config is invalid")
        taker_exit_grid_values.add(grid_ns)
        taker_exit_max_book_age_values.add(max_book_age_ns)

    if len(runner_hashes) != 1 or len(runner_payloads) != 1 or len(runner_versions) != 1:
        raise ValueError("selected exit-maker partitions do not share one frozen runner config")
    if len(entry_config_hashes) != 1:
        raise ValueError("selected entry partitions do not share one frozen runner config")
    if entry_hedge_delay_values != {expected_hedge_delay_ns}:
        raise ValueError("selected entry partitions do not share the formal hedge delay")
    if len(exit_rule_id_sets) != 1 or len(exit_route_sets) != 1:
        raise ValueError("selected exit-maker partitions disagree on rule/route grids")
    if next(iter(exit_rule_id_sets)) != ("frozen_center", "frozen_lower"):
        raise ValueError(
            "formal exit-maker runner must declare frozen_center and frozen_lower"
        )
    if next(iter(exit_route_sets)) != (
        "future_bid_spot_taker",
        "spot_ask_future_taker",
    ):
        raise ValueError("formal exit-maker runner must declare both exit routes")
    if len(taker_exit_grid_values) != 1 or len(taker_exit_max_book_age_values) != 1:
        raise ValueError("selected T/T baselines do not share one exit-grid config")
    root_manifest = _validate_root_manifest(
        root,
        selected,
        validate_hashes=validate_hashes,
        required=require_root_manifest,
    )

    def combine(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
        return (
            pl.concat(frames, how="diagonal_relaxed", rechunk=True)
            if frames
            else pl.DataFrame()
        )

    combined_positions = combine(
        loaded_exit["exit_maker_position_policy_facts.parquet"]
    )
    combined_actions = combine(actions)
    combined_exits = combine(exits)
    predecessor_map = _expected_session_predecessors(
        session_calendar,
        selected_dates,
    )
    position_lineage = _validate_position_rule_lineage(
        combined_positions,
        combined_exits,
        expected_session_predecessors=predecessor_map,
    )

    metadata: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "exit_maker_root": str(root),
        "entry_execution_root": str(entry_root),
        "selected_session_count": len(selected_dates),
        "selected_dates": selected_dates,
        "selected_product_count": len(products),
        "selected_product_day_count": len(selected),
        "expected_product_day_count": len(selected_dates) * len(products),
        "missing_product_day_count": missing.height,
        "balanced_product_day_grid": missing.is_empty(),
        "value_codes": products,
        "runner_config_sha256": next(iter(runner_hashes)),
        "entry_config_sha256": next(iter(entry_config_hashes)),
        "runner_version": next(iter(runner_versions)),
        "hedge_delay_ns": expected_hedge_delay_ns,
        "entry_action_hedge_delay_ns": next(iter(entry_hedge_delay_values)),
        "exit_maker_hedge_delay_ns": expected_hedge_delay_ns,
        "entry_exit_hedge_delay_match": True,
        "entry_action_hedge_delay_rows_validated": combined_actions.filter(
            pl.col("full_fill") == True  # noqa: E712
        ).height,
        "exit_rule_ids": list(next(iter(exit_rule_id_sets))),
        "exit_routes": list(next(iter(exit_route_sets))),
        "session_predecessors": predecessor_map,
        "session_predecessor_validation": (
            "exact_calendar" if predecessor_map is not None else "strictly_prior_only"
        ),
        "position_rule_lineage": position_lineage,
        "position_policy_to_bound_entry_exit_facts_crossvalidated": True,
        "entry_action_and_exit_facts_hash_lineage_validated": validate_hashes,
        **root_manifest,
        "taker_taker_exit_grid_ns": next(iter(taker_exit_grid_values)),
        "taker_taker_exit_max_book_age_ns": next(
            iter(taker_exit_max_book_age_values)
        ),
        "partition_hashes_validated": validate_hashes,
    }
    return ExitMakerPartitionInputs(
        policy_support=combine(loaded_exit["exit_maker_policy_support.parquet"]),
        candidate_aliases=combine(
            loaded_exit["exit_maker_candidate_aliases.parquet"]
        ),
        raw_candidate_facts=combine(
            loaded_exit["exit_maker_raw_candidate_facts.parquet"]
        ),
        position_policy_facts=combined_positions,
        action_facts=combined_actions,
        taker_exit_facts=combined_exits,
        coverage=coverage,
        metadata=metadata,
    )


def build_exit_maker_report(
    inputs: ExitMakerPartitionInputs,
    *,
    assumed_non_price_cost_bp: float = DEFAULT_NON_PRICE_COST_BP,
) -> ExitMakerReport:
    """Build daily and product/q/rule/route maker-exit diagnostics."""

    cost_bp = _validate_cost(assumed_non_price_cost_bp)
    _require(inputs.policy_support, _SUPPORT_REQUIRED, "exit maker policy support")
    _require(inputs.candidate_aliases, _ALIAS_REQUIRED, "exit maker candidate aliases")
    _require(inputs.raw_candidate_facts, _RAW_REQUIRED, "exit maker raw candidates")
    _require(inputs.position_policy_facts, _POSITION_REQUIRED, "exit maker position policies")
    _require(inputs.action_facts, _ACTION_REQUIRED, "entry execution actions")
    if not inputs.taker_exit_facts.is_empty():
        _require(inputs.taker_exit_facts, _TAKER_EXIT_REQUIRED, "taker/taker exit facts")

    support, aliases, candidates, policies = _normalise_and_validate_facts(inputs)
    policy_daily = _aggregate_policy(policies, DAILY_POLICY_KEY, cost_bp)
    policy_product = _aggregate_policy(policies, PRODUCT_POLICY_KEY, cost_bp)
    candidate_daily = _aggregate_candidates(candidates, DAILY_POLICY_KEY)
    candidate_product = _aggregate_candidates(candidates, PRODUCT_POLICY_KEY)
    alias_daily = _aggregate_aliases(aliases, DAILY_POLICY_KEY)
    alias_product = _aggregate_aliases(aliases, PRODUCT_POLICY_KEY)

    daily = _finish_summary(
        policy_daily.join(candidate_daily, on=DAILY_POLICY_KEY, how="left", validate="1:1")
        .join(alias_daily, on=DAILY_POLICY_KEY, how="left", validate="1:1"),
        DAILY_POLICY_KEY,
    )
    product = _finish_summary(
        policy_product.join(
            candidate_product, on=PRODUCT_POLICY_KEY, how="left", validate="1:1"
        ).join(alias_product, on=PRODUCT_POLICY_KEY, how="left", validate="1:1"),
        PRODUCT_POLICY_KEY,
    )
    daily_branch = _strict_branch_table(policies, DAILY_POLICY_KEY)
    product_branch = _strict_branch_table(policies, PRODUCT_POLICY_KEY)
    threshold_lineage = _threshold_lineage_table(policies)
    pairs, matched = _matched_comparison(
        policies,
        inputs.taker_exit_facts,
        cost_bp=cost_bp,
        taker_exit_grid_ns=inputs.metadata.get("taker_taker_exit_grid_ns"),
    )

    taker_latency_ns: int | None = None
    if not inputs.taker_exit_facts.is_empty():
        latency_values = (
            inputs.taker_exit_facts["exit_added_latency_ns"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        if len(latency_values) != 1:
            raise ValueError("T/T baseline must have one exit_added_latency_ns")
        taker_latency_ns = int(latency_values[0])
    latency_matched = taker_latency_ns == EXPECTED_HEDGE_DELAY_NS
    taker_role = (
        "optimistic_zero_added_latency_benchmark"
        if taker_latency_ns == 0
        else "observed_added_latency_benchmark"
        if taker_latency_ns is not None
        else "unavailable"
    )

    metadata = dict(inputs.metadata)
    metadata.update(
        {
            "report_semantics": "conditional_exit_maker_policy_and_canonical_candidate_v1",
            "policy_trial_unit": "entry_alias_x_exit_rule_x_exit_route",
            "raw_candidate_unit": "canonical_exit_raw_candidate_fact_id_within_q_action",
            "candidate_alias_unit": "exit_policy_trial_x_candidate_generation",
            "policy_trial_count": support.height,
            "canonical_raw_candidate_count": inputs.raw_candidate_facts.height,
            "candidate_alias_count": aliases.height,
            "assumed_non_price_cycle_cost_bp": cost_bp,
            "cost_scope": NON_PRICE_COST_SCOPE,
            "cost_sensitivity_analysis_only": True,
            "price_slippage_already_in_gross": True,
            "cancel_ack_observation_available": False,
            "cancel_rate_is_cancel_request_only": True,
            "exit_rule_rows_safe_to_sum": False,
            "exit_route_rows_safe_to_sum": False,
            "boundary_quantile_alias_rows_safe_to_sum": False,
            "matched_exit_style_comparison_available": not pairs.is_empty(),
            "threshold_lineage_rows": threshold_lineage.height,
            "threshold_lineage_semantics": "frozen_prior_session_exit_rule_per_policy_trial",
            "d_minus_one_threshold_lineage_valid": (
                threshold_lineage.is_empty()
                or threshold_lineage["source_strictly_precedes_target_date"].all()
            ),
            "matched_taker_baseline_shared_across_exit_routes": True,
            "taker_taker_exit_added_latency_ns": taker_latency_ns,
            "maker_exit_hedge_delay_ns": EXPECTED_HEDGE_DELAY_NS,
            "latency_matched": latency_matched,
            "taker_baseline_role": taker_role,
            "paired_gross_delta_is_latency_matched_comparison": latency_matched,
            "joint_volume_allocated": False,
            "pathwise_ev_ready": False,
        }
    )
    return ExitMakerReport(
        coverage=inputs.coverage.sort(["Date", "ValueCode"]),
        daily_policy=daily.sort(DAILY_POLICY_KEY),
        product_policy=product.sort(PRODUCT_POLICY_KEY),
        daily_strict_branch=daily_branch,
        product_strict_branch=product_branch,
        policy_threshold_lineage=threshold_lineage,
        matched_pairs=pairs,
        matched_summary=matched,
        metadata=metadata,
    )


def run_exit_maker_report(
    exit_maker_root: Path,
    entry_execution_root: Path,
    *,
    output_dir: Path | None = None,
    sessions: int | None = 60,
    value_codes: Iterable[str] | None = None,
    require_exact_sessions: bool = True,
    require_balanced_product_days: bool = False,
    validate_hashes: bool = True,
    assumed_non_price_cost_bp: float = DEFAULT_NON_PRICE_COST_BP,
    session_calendar_path: Path = DEFAULT_SESSION_CALENDAR_PATH,
) -> ExitMakerReport:
    """Validate, build, and publish one report directory atomically."""

    inputs = load_exit_maker_partition_inputs(
        exit_maker_root,
        entry_execution_root,
        sessions=sessions,
        value_codes=value_codes,
        require_exact_sessions=require_exact_sessions,
        require_balanced_product_days=require_balanced_product_days,
        validate_hashes=validate_hashes,
        session_calendar=_load_session_calendar(session_calendar_path),
        require_root_manifest=True,
    )
    report = build_exit_maker_report(
        inputs,
        assumed_non_price_cost_bp=assumed_non_price_cost_bp,
    )
    count = int(inputs.metadata["selected_session_count"])
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(exit_maker_root) / f"report_{count}_sessions"
    )
    _publish_report(report, destination)
    return report


def _normalise_and_validate_facts(
    inputs: ExitMakerPartitionInputs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    actions = inputs.action_facts.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    )
    if actions.select("policy_generation_id").n_unique() != actions.height:
        raise ValueError("entry actions must be unique by policy_generation_id")
    action_meta = actions.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        pl.col("boundary_quantile"),
        pl.col("lookup_action_id").alias("entry_lookup_action_id"),
        pl.col("parameter_version").alias("entry_parameter_version"),
        pl.col("submit_recv_time_ns").alias("entry_submit_recv_time_ns"),
        pl.col("entry_spot_price"),
        pl.col("entry_hedge_contract_size_shares"),
        pl.col("entry_hedge_signed_latency_slippage_bp"),
        pl.col("entry_hedge_signed_depth_slippage_bp"),
        pl.col("entry_hedge_signed_total_slippage_bp"),
        pl.col("entry_hedge_decision_book_age_ms"),
        pl.col("full_fill").alias("entry_full_fill"),
        pl.col("entry_hedge_status"),
        pl.col("entry_hedge_label_observed"),
        pl.col("entry_hedge_executable"),
    )
    action_identity = actions.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        pl.col("Date").alias("_action_Date"),
        pl.col("ValueCode").alias("_action_ValueCode"),
        pl.col("QuoteCode").alias("_action_QuoteCode"),
        pl.col("route").alias("_action_entry_route"),
        pl.col("raw_order_fact_id").alias("_action_entry_raw_order_fact_id"),
    )

    support = inputs.policy_support.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    aliases = inputs.candidate_aliases.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    raw = inputs.raw_candidate_facts.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    policies = inputs.position_policy_facts.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    for name, frame in (("support", support), ("candidate aliases", aliases), ("position policies", policies)):
        if frame.select("exit_policy_trial_id").n_unique() != frame.height and name != "candidate aliases":
            raise ValueError(f"{name} must be unique by exit_policy_trial_id")
    if aliases.select("exit_candidate_alias_id").n_unique() != aliases.height:
        raise ValueError("candidate aliases must be unique by exit_candidate_alias_id")
    if raw.select("exit_raw_candidate_fact_id").n_unique() != raw.height:
        raise ValueError("raw candidates must be unique by canonical raw id")

    support = support.join(action_identity, on="entry_policy_generation_id", how="left", validate="m:1")
    _validate_joined_action_identity(support)
    support = support.drop([name for name in support.columns if name.startswith("_action_")])
    support = support.join(action_meta, on="entry_policy_generation_id", how="left", validate="m:1")
    if support.filter(
        (pl.col("position_status") != "position_established")
        | (pl.col("entry_full_fill") != True)  # noqa: E712
        | (pl.col("entry_hedge_label_observed") != True)  # noqa: E712
        | (pl.col("entry_hedge_executable") != True)  # noqa: E712
    ).height:
        raise ValueError("exit-maker support must contain established entry aliases only")

    policies = policies.join(action_identity, on="entry_policy_generation_id", how="left", validate="m:1")
    _validate_joined_action_identity(policies)
    policies = policies.drop([name for name in policies.columns if name.startswith("_action_")])
    policies = policies.join(action_meta, on="entry_policy_generation_id", how="left", validate="m:1")
    aliases = aliases.join(action_identity, on="entry_policy_generation_id", how="left", validate="m:1")
    _validate_joined_action_identity(aliases)
    aliases = aliases.drop([name for name in aliases.columns if name.startswith("_action_")])
    aliases = aliases.join(
        action_meta.select("entry_policy_generation_id", "boundary_quantile"),
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )

    support_ids = set(support["exit_policy_trial_id"].to_list())
    policy_ids = set(policies["exit_policy_trial_id"].to_list())
    if support_ids != policy_ids:
        raise ValueError("support and position policy trial universes differ")
    if not set(aliases["exit_policy_trial_id"].to_list()).issubset(support_ids):
        raise ValueError("candidate alias does not resolve to a policy trial")
    expected_alias_counts = support.select(
        "exit_policy_trial_id", pl.col("raw_candidate_count").cast(pl.Int64)
    )
    actual_alias_counts = aliases.group_by("exit_policy_trial_id").agg(
        pl.len().alias("_actual_raw_candidate_count")
    )
    bad_counts = expected_alias_counts.join(
        actual_alias_counts, on="exit_policy_trial_id", how="left", validate="1:1"
    ).with_columns(pl.col("_actual_raw_candidate_count").fill_null(0)).filter(
        pl.col("raw_candidate_count") != pl.col("_actual_raw_candidate_count")
    )
    if bad_counts.height:
        raise ValueError("support raw_candidate_count disagrees with candidate aliases")

    raw_ids = set(raw["exit_raw_candidate_fact_id"].to_list())
    alias_raw_ids = set(aliases["exit_raw_candidate_fact_id"].to_list())
    if raw_ids != alias_raw_ids:
        raise ValueError("raw candidate and candidate-alias universes differ")
    if raw.filter(pl.col("cancel_ack_observed") == True).height:  # noqa: E712
        raise ValueError("report v1 expects cancel acknowledgement to be unavailable")
    if policies.filter(pl.col("cancel_ack_observed") == True).height:  # noqa: E712
        raise ValueError("report v1 expects policy cancel acknowledgement to be unavailable")
    if policies.filter(pl.col("joint_volume_allocated") == True).height:  # noqa: E712
        raise ValueError("report v1 is an independent-candidate study")
    _validate_canonical_raw_ids(raw)

    # A canonical raw fact retains the longest nominal policy horizon.  Its
    # stop/fill fields must not be copied onto a shorter center/lower policy.
    # Candidate aliases carry the policy-specific replay label; first prove
    # that aliases inside one q/rule/route/raw cell agree, then deduplicate.
    candidate_outcome_columns = [
        "cancel_required",
        "any_fill",
        "full_fill",
        "partial_fill",
        "exit_hedge_status",
    ]
    if "exit_hedge_decision_book_age_ms" in aliases.columns:
        candidate_outcome_columns.append("exit_hedge_decision_book_age_ms")
    candidate_cell_key = [*DAILY_POLICY_KEY, "exit_raw_candidate_fact_id"]
    conflicts = aliases.group_by(candidate_cell_key).agg(
        *(
            pl.col(column).n_unique().alias(column)
            for column in candidate_outcome_columns
        )
    ).filter(
        pl.any_horizontal(
            *(pl.col(column) != 1 for column in candidate_outcome_columns)
        )
    )
    if conflicts.height:
        raise ValueError(
            "policy-specific candidate outcomes disagree inside one q/rule/route/raw cell"
        )
    candidates = aliases.select(
        *candidate_cell_key,
        *candidate_outcome_columns,
    ).unique(subset=candidate_cell_key, keep="first")
    if "exit_hedge_decision_book_age_ms" not in candidates.columns:
        candidates = candidates.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("exit_hedge_decision_book_age_ms")
        )

    support_lineage = support.select(
        "exit_policy_trial_id",
        "admission_status",
        pl.col("exit_threshold_basis_bp").alias("_support_exit_threshold_basis_bp"),
        pl.col("exit_rule_source_asof_date").alias(
            "_support_exit_rule_source_asof_date"
        ),
    )
    policies = policies.join(
        support_lineage,
        on="exit_policy_trial_id",
        how="left",
        validate="1:1",
    )
    lineage_mismatch = policies.filter(
        (
            pl.col("exit_threshold_basis_bp")
            != pl.col("_support_exit_threshold_basis_bp")
        ).fill_null(True)
        | (
            pl.col("exit_rule_source_asof_date").cast(pl.String)
            != pl.col("_support_exit_rule_source_asof_date").cast(pl.String)
        ).fill_null(True)
    )
    if lineage_mismatch.height:
        raise ValueError("policy support and position facts disagree on exit-rule lineage")
    policies = policies.drop(
        "_support_exit_threshold_basis_bp",
        "_support_exit_rule_source_asof_date",
    )
    policies = _add_threshold_lineage_metrics(policies)
    policies = _add_policy_metrics(policies)
    return support, aliases, candidates, policies


def _add_threshold_lineage_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    invalid_threshold = frame.filter(
        pl.col("exit_threshold_basis_bp").is_null()
        | ~pl.col("exit_threshold_basis_bp").is_finite()
    )
    if invalid_threshold.height:
        raise ValueError("exit threshold lineage must be finite")
    try:
        enriched = frame.with_columns(
            pl.col("Date")
            .str.to_date(format="%Y%m%d", strict=True)
            .alias("_target_date"),
            pl.col("exit_rule_source_asof_date")
            .cast(pl.String)
            .str.to_date(format="%Y%m%d", strict=True)
            .alias("_source_date"),
        ).with_columns(
            (pl.col("_target_date") - pl.col("_source_date"))
            .dt.total_days()
            .cast(pl.Int64)
            .alias("exit_rule_source_age_calendar_days")
        )
    except Exception as error:
        raise ValueError("exit-rule source Date must use YYYYMMDD") from error
    if enriched.filter(pl.col("exit_rule_source_age_calendar_days") <= 0).height:
        raise ValueError("exit-rule source must strictly precede target Date")
    return enriched.with_columns(
        (pl.col("exit_rule_source_age_calendar_days") > 0).alias(
            "source_strictly_precedes_target_date"
        )
    ).drop("_target_date", "_source_date")


def _add_policy_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    nominal = pl.col("nominal_instant_cancel_v0_branch")
    same = nominal == _NOMINAL_SAME_DAY
    carry = nominal.is_in(sorted(_NOMINAL_CARRY))
    notional = pl.col("entry_spot_price") * pl.col("entry_hedge_contract_size_shares")
    valid_notional = notional.is_finite() & (notional > 0)
    malformed = frame.filter(
        same
        & (
            pl.col("gross_cycle_pnl_twd").is_null()
            | ~pl.col("gross_cycle_pnl_twd").is_finite()
            | ~valid_notional
            | (pl.col("exit_hedge_status") != "executable")
        )
    )
    if malformed.height:
        raise ValueError("nominal flat_same_day trial lacks executable hedge/gross notional")
    invalid_time = frame.filter(
        same
        & (
            pl.col("exit_decision_time_ns").is_null()
            | (pl.col("exit_decision_time_ns") < pl.col("position_established_ns"))
            | (pl.col("exit_decision_time_ns") < pl.col("entry_submit_recv_time_ns"))
        )
    )
    if invalid_time.height:
        raise ValueError("nominal flat_same_day trial has invalid holding/capital time")
    return frame.with_columns(
        pl.when(same)
        .then(pl.lit("same_day"))
        .when(carry)
        .then(pl.lit("carry"))
        .otherwise(pl.lit("unknown"))
        .alias("nominal_outcome_category"),
        pl.when(pl.col("branch_status") == "flat_same_day")
        .then(pl.lit("same_day"))
        .when(pl.col("branch_status").is_in(sorted(_STRICT_CARRY)))
        .then(pl.lit("carry"))
        .otherwise(pl.lit("unknown"))
        .alias("strict_outcome_category"),
        pl.when(same)
        .then(pl.col("gross_cycle_pnl_twd") / notional * 10_000.0)
        .otherwise(None)
        .alias("nominal_gross_cycle_bp"),
        pl.when(same)
        .then((pl.col("exit_decision_time_ns") - pl.col("position_established_ns")) / 1_000_000.0)
        .otherwise(None)
        .alias("nominal_holding_time_ms"),
        pl.when(same)
        .then((pl.col("exit_decision_time_ns") - pl.col("entry_submit_recv_time_ns")) / 1_000_000_000.0)
        .otherwise(None)
        .alias("nominal_capital_time_seconds"),
    )


def _aggregate_policy(
    policies: pl.DataFrame,
    keys: Sequence[str],
    cost_bp: float,
) -> pl.DataFrame:
    same = pl.col("nominal_outcome_category") == "same_day"
    exit_executable = same & (pl.col("exit_hedge_status") == "executable")
    entry_age_observed = pl.col("entry_hedge_decision_book_age_ms").is_not_null()
    result = policies.group_by(list(keys)).agg(
        pl.len().alias("established_policy_trials"),
        pl.col("Date").n_unique().alias("sessions_with_trials"),
        pl.col("entry_raw_order_fact_id").n_unique().alias("unique_physical_entry_positions"),
        pl.col("exit_threshold_basis_bp").count().alias(
            "exit_threshold_support_trials"
        ),
        pl.col("exit_threshold_basis_bp").n_unique().alias(
            "exit_threshold_unique_values"
        ),
        pl.col("exit_threshold_basis_bp").min().alias(
            "exit_threshold_basis_bp_min"
        ),
        pl.col("exit_threshold_basis_bp").quantile(0.10).alias(
            "exit_threshold_basis_bp_p10"
        ),
        pl.col("exit_threshold_basis_bp").median().alias(
            "exit_threshold_basis_bp_p50"
        ),
        pl.col("exit_threshold_basis_bp").quantile(0.90).alias(
            "exit_threshold_basis_bp_p90"
        ),
        pl.col("exit_threshold_basis_bp").max().alias(
            "exit_threshold_basis_bp_max"
        ),
        pl.col("exit_rule_source_asof_date").count().alias(
            "exit_rule_source_support_trials"
        ),
        pl.col("exit_rule_source_asof_date").n_unique().alias(
            "exit_rule_source_unique_dates"
        ),
        pl.col("exit_rule_source_asof_date").min().alias(
            "exit_rule_source_asof_date_min"
        ),
        pl.col("exit_rule_source_asof_date").max().alias(
            "exit_rule_source_asof_date_max"
        ),
        pl.col("entry_parameter_version").count().alias(
            "entry_parameter_version_support_trials"
        ),
        pl.col("entry_parameter_version").n_unique().alias(
            "entry_parameter_version_unique_values"
        ),
        pl.col("entry_parameter_version").min().alias(
            "entry_parameter_version_min"
        ),
        pl.col("entry_parameter_version").max().alias(
            "entry_parameter_version_max"
        ),
        pl.struct(
            "entry_parameter_version",
            "exit_rule_id",
            "exit_rule_source_asof_date",
            "exit_threshold_basis_bp",
        ).n_unique().alias("frozen_exit_policy_signatures"),
        pl.col("exit_rule_source_age_calendar_days").median().alias(
            "exit_rule_source_age_days_p50"
        ),
        pl.col("exit_rule_source_age_calendar_days").quantile(0.90).alias(
            "exit_rule_source_age_days_p90"
        ),
        pl.col("source_strictly_precedes_target_date").sum().alias(
            "d_minus_one_lineage_valid_trials"
        ),
        (pl.col("admission_status") == "admitted").sum().alias("admitted_policy_trials"),
        (pl.col("oco_active_sibling_cancel_count") > 0).sum().alias("trials_with_active_sibling_cancel"),
        pl.col("oco_active_sibling_cancel_count").sum().alias("active_sibling_cancel_requests"),
        (pl.col("prior_unacked_cancel_count_before_winner") > 0).sum().alias("trials_with_prior_unacked_cancel"),
        pl.col("prior_unacked_cancel_count_before_winner").sum().alias("prior_unacked_cancel_requests"),
        (pl.col("nominal_outcome_category") == "same_day").sum().alias("nominal_same_day_trials"),
        (pl.col("nominal_outcome_category") == "carry").sum().alias("nominal_carry_trials"),
        (pl.col("nominal_outcome_category") == "unknown").sum().alias("nominal_unknown_trials"),
        (pl.col("strict_outcome_category") == "same_day").sum().alias("strict_same_day_trials"),
        (pl.col("strict_outcome_category") == "carry").sum().alias("strict_carry_trials"),
        (pl.col("strict_outcome_category") == "unknown").sum().alias("strict_unknown_trials"),
        pl.col("nominal_gross_cycle_bp").drop_nulls().median().alias("nominal_same_day_gross_bp_p50"),
        pl.col("nominal_gross_cycle_bp").drop_nulls().quantile(0.10).alias("nominal_same_day_gross_bp_p10"),
        pl.col("nominal_gross_cycle_bp").drop_nulls().quantile(0.90).alias("nominal_same_day_gross_bp_p90"),
        (pl.col("nominal_gross_cycle_bp") - cost_bp).drop_nulls().median().alias("nominal_same_day_gross_minus_cost_bp_p50"),
        ((pl.col("nominal_gross_cycle_bp") - cost_bp) > 0).filter(same).sum().alias("nominal_same_day_positive_after_cost_trials"),
        pl.col("nominal_holding_time_ms").drop_nulls().median().alias("nominal_same_day_holding_ms_p50"),
        pl.col("nominal_holding_time_ms").drop_nulls().quantile(0.90).alias("nominal_same_day_holding_ms_p90"),
        pl.col("nominal_capital_time_seconds").drop_nulls().median().alias("nominal_same_day_capital_seconds_p50"),
        pl.col("nominal_capital_time_seconds").drop_nulls().quantile(0.90).alias("nominal_same_day_capital_seconds_p90"),
        pl.col("entry_hedge_signed_total_slippage_bp").count().alias("entry_hedge_slippage_observations"),
        pl.col("entry_hedge_signed_latency_slippage_bp").drop_nulls().median().alias("entry_hedge_latency_slippage_bp_p50"),
        pl.col("entry_hedge_signed_latency_slippage_bp").drop_nulls().quantile(0.95).alias("entry_hedge_latency_slippage_bp_p95"),
        pl.col("entry_hedge_signed_depth_slippage_bp").drop_nulls().median().alias("entry_hedge_depth_slippage_bp_p50"),
        pl.col("entry_hedge_signed_depth_slippage_bp").drop_nulls().quantile(0.95).alias("entry_hedge_depth_slippage_bp_p95"),
        pl.col("entry_hedge_signed_total_slippage_bp").drop_nulls().median().alias("entry_hedge_total_slippage_bp_p50"),
        pl.col("entry_hedge_signed_total_slippage_bp").drop_nulls().quantile(0.90).alias("entry_hedge_total_slippage_bp_p90"),
        pl.col("entry_hedge_signed_total_slippage_bp").drop_nulls().quantile(0.95).alias("entry_hedge_total_slippage_bp_p95"),
        entry_age_observed.sum().alias("entry_hedge_book_age_observations"),
        (entry_age_observed & (pl.col("entry_hedge_decision_book_age_ms") <= 100.0)).sum().alias("entry_hedge_fresh_le_100ms_observations"),
        (entry_age_observed & (pl.col("entry_hedge_decision_book_age_ms") <= 1_000.0)).sum().alias("entry_hedge_fresh_le_1s_observations"),
        pl.col("exit_hedge_signed_total_slippage_bp").filter(exit_executable).count().alias("nominal_same_day_exit_hedge_slippage_observations"),
        pl.col("exit_hedge_signed_latency_slippage_bp").filter(exit_executable).drop_nulls().median().alias("nominal_same_day_exit_hedge_latency_slippage_bp_p50"),
        pl.col("exit_hedge_signed_latency_slippage_bp").filter(exit_executable).drop_nulls().quantile(0.95).alias("nominal_same_day_exit_hedge_latency_slippage_bp_p95"),
        pl.col("exit_hedge_signed_depth_slippage_bp").filter(exit_executable).drop_nulls().median().alias("nominal_same_day_exit_hedge_depth_slippage_bp_p50"),
        pl.col("exit_hedge_signed_depth_slippage_bp").filter(exit_executable).drop_nulls().quantile(0.95).alias("nominal_same_day_exit_hedge_depth_slippage_bp_p95"),
        pl.col("exit_hedge_signed_total_slippage_bp").filter(exit_executable).drop_nulls().median().alias("nominal_same_day_exit_hedge_total_slippage_bp_p50"),
        pl.col("exit_hedge_signed_total_slippage_bp").filter(exit_executable).drop_nulls().quantile(0.90).alias("nominal_same_day_exit_hedge_total_slippage_bp_p90"),
        pl.col("exit_hedge_signed_total_slippage_bp").filter(exit_executable).drop_nulls().quantile(0.95).alias("nominal_same_day_exit_hedge_total_slippage_bp_p95"),
    )
    denominator = pl.col("established_policy_trials")
    same_denom = pl.col("nominal_same_day_trials")
    return result.with_columns(
        (pl.col("admitted_policy_trials") / denominator).alias("admission_rate"),
        (
            pl.col("d_minus_one_lineage_valid_trials") / denominator
        ).alias("d_minus_one_lineage_valid_rate"),
        (pl.col("exit_threshold_unique_values") > 1).alias(
            "exit_threshold_varies_within_cell"
        ),
        pl.when(pl.col("exit_threshold_unique_values") == 1)
        .then(pl.col("exit_threshold_basis_bp_min"))
        .otherwise(None)
        .alias("exit_threshold_basis_bp_exact"),
        (pl.col("exit_rule_source_unique_dates") > 1).alias(
            "exit_rule_source_varies_within_cell"
        ),
        pl.when(pl.col("exit_rule_source_unique_dates") == 1)
        .then(pl.col("exit_rule_source_asof_date_min"))
        .otherwise(None)
        .alias("exit_rule_source_asof_date_exact"),
        pl.when(pl.col("entry_parameter_version_unique_values") == 1)
        .then(pl.col("entry_parameter_version_min"))
        .otherwise(None)
        .alias("entry_parameter_version_exact"),
        (pl.col("trials_with_active_sibling_cancel") / denominator).alias("active_sibling_cancel_trial_rate"),
        (pl.col("trials_with_prior_unacked_cancel") / denominator).alias("prior_unacked_cancel_trial_rate"),
        (pl.col("nominal_same_day_trials") / denominator).alias("nominal_same_day_rate"),
        (pl.col("nominal_carry_trials") / denominator).alias("nominal_carry_rate"),
        (pl.col("nominal_unknown_trials") / denominator).alias("nominal_unknown_rate"),
        (pl.col("strict_same_day_trials") / denominator).alias("strict_same_day_rate"),
        (pl.col("strict_carry_trials") / denominator).alias("strict_carry_rate"),
        (pl.col("strict_unknown_trials") / denominator).alias("strict_unknown_rate"),
        pl.when(same_denom > 0)
        .then(pl.col("nominal_same_day_positive_after_cost_trials") / same_denom)
        .otherwise(None)
        .alias("nominal_same_day_positive_after_cost_rate"),
        pl.when(pl.col("entry_hedge_book_age_observations") > 0)
        .then(
            pl.col("entry_hedge_fresh_le_100ms_observations")
            / pl.col("entry_hedge_book_age_observations")
        )
        .otherwise(None)
        .alias("entry_hedge_fresh_le_100ms_rate"),
        pl.when(pl.col("entry_hedge_book_age_observations") > 0)
        .then(
            pl.col("entry_hedge_fresh_le_1s_observations")
            / pl.col("entry_hedge_book_age_observations")
        )
        .otherwise(None)
        .alias("entry_hedge_fresh_le_1s_rate"),
        pl.lit(cost_bp).alias("assumed_non_price_cycle_cost_bp"),
        pl.lit(NON_PRICE_COST_SCOPE).alias("cost_scope"),
        pl.lit(True).alias("entry_and_exit_hedge_slippage_already_in_gross"),
        pl.lit(False).alias("hedge_slippage_subtracted_again"),
    )


def _aggregate_candidates(candidates: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    age_observed = pl.col("exit_hedge_decision_book_age_ms").is_not_null()
    executable = pl.col("exit_hedge_status") == "executable"
    fill_known = pl.col("any_fill").is_not_null()
    result = candidates.group_by(list(keys)).agg(
        pl.len().alias("unique_canonical_raw_candidates"),
        pl.col("cancel_required").sum().alias("raw_cancel_required_candidates"),
        fill_known.sum().alias("raw_fill_known_candidates"),
        (~fill_known).sum().alias("raw_fill_unknown_candidates"),
        pl.col("any_fill").sum().alias("raw_any_fill_candidates"),
        pl.col("full_fill").sum().alias("raw_full_fill_candidates"),
        pl.col("partial_fill").sum().alias("raw_partial_fill_candidates"),
        executable.sum().alias("exit_hedge_executable_candidates"),
        age_observed.sum().alias("exit_hedge_book_age_observed_candidates"),
        (executable & age_observed & (pl.col("exit_hedge_decision_book_age_ms") <= 100.0)).sum().alias("exit_hedge_fresh_le_100ms_candidates"),
        (executable & age_observed & (pl.col("exit_hedge_decision_book_age_ms") <= 1_000.0)).sum().alias("exit_hedge_fresh_le_1s_candidates"),
    )
    denominator = pl.col("unique_canonical_raw_candidates")
    known = pl.col("raw_fill_known_candidates")
    full = pl.col("raw_full_fill_candidates")
    observed = pl.col("exit_hedge_book_age_observed_candidates")
    return result.with_columns(
        (pl.col("raw_cancel_required_candidates") / denominator).alias("raw_cancel_required_rate"),
        (pl.col("raw_fill_unknown_candidates") / denominator).alias("raw_fill_unknown_rate"),
        (pl.col("raw_any_fill_candidates") / denominator).alias("raw_any_fill_rate_all_candidates_lower_bound"),
        (pl.col("raw_full_fill_candidates") / denominator).alias("raw_full_fill_rate_all_candidates_lower_bound"),
        (pl.col("raw_partial_fill_candidates") / denominator).alias("raw_partial_fill_rate_all_candidates_lower_bound"),
        pl.when(known > 0).then(pl.col("raw_any_fill_candidates") / known).otherwise(None).alias("raw_any_fill_rate_given_fill_known"),
        pl.when(known > 0).then(pl.col("raw_full_fill_candidates") / known).otherwise(None).alias("raw_full_fill_rate_given_fill_known"),
        pl.when(known > 0).then(pl.col("raw_partial_fill_candidates") / known).otherwise(None).alias("raw_partial_fill_rate_given_fill_known"),
        (pl.col("exit_hedge_executable_candidates") / denominator).alias("exit_hedge_executable_rate_per_candidate"),
        pl.when(full > 0).then(pl.col("exit_hedge_executable_candidates") / full).otherwise(None).alias("exit_hedge_executable_rate_per_full_fill"),
        pl.when(observed > 0).then(pl.col("exit_hedge_fresh_le_100ms_candidates") / observed).otherwise(None).alias("exit_hedge_fresh_le_100ms_rate_when_observed"),
        pl.when(observed > 0).then(pl.col("exit_hedge_fresh_le_1s_candidates") / observed).otherwise(None).alias("exit_hedge_fresh_le_1s_rate_when_observed"),
        (observed > 0).alias("exit_hedge_book_age_available"),
    )


def _aggregate_aliases(aliases: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    return aliases.group_by(list(keys)).agg(
        pl.len().alias("policy_candidate_aliases"),
        pl.col("exit_candidate_alias_id").n_unique().alias("unique_policy_candidate_aliases"),
    )


def _finish_summary(frame: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    integer_columns = [
        "unique_canonical_raw_candidates",
        "raw_cancel_required_candidates",
        "raw_fill_known_candidates",
        "raw_fill_unknown_candidates",
        "raw_any_fill_candidates",
        "raw_full_fill_candidates",
        "raw_partial_fill_candidates",
        "exit_hedge_executable_candidates",
        "exit_hedge_book_age_observed_candidates",
        "exit_hedge_fresh_le_100ms_candidates",
        "exit_hedge_fresh_le_1s_candidates",
        "policy_candidate_aliases",
        "unique_policy_candidate_aliases",
    ]
    expressions: list[pl.Expr] = []
    for column in integer_columns:
        if column in frame.columns:
            expressions.append(pl.col(column).fill_null(0).cast(pl.Int64))
    result = frame.with_columns(*expressions)
    denominator = pl.col("unique_canonical_raw_candidates")
    known = pl.col("raw_fill_known_candidates")
    full = pl.col("raw_full_fill_candidates")
    observed = pl.col("exit_hedge_book_age_observed_candidates")
    result = result.with_columns(
        pl.when(denominator > 0).then(pl.col("raw_cancel_required_candidates") / denominator).otherwise(None).alias("raw_cancel_required_rate"),
        pl.when(denominator > 0).then(pl.col("raw_fill_unknown_candidates") / denominator).otherwise(None).alias("raw_fill_unknown_rate"),
        pl.when(denominator > 0).then(pl.col("raw_any_fill_candidates") / denominator).otherwise(None).alias("raw_any_fill_rate_all_candidates_lower_bound"),
        pl.when(denominator > 0).then(pl.col("raw_full_fill_candidates") / denominator).otherwise(None).alias("raw_full_fill_rate_all_candidates_lower_bound"),
        pl.when(denominator > 0).then(pl.col("raw_partial_fill_candidates") / denominator).otherwise(None).alias("raw_partial_fill_rate_all_candidates_lower_bound"),
        pl.when(known > 0).then(pl.col("raw_any_fill_candidates") / known).otherwise(None).alias("raw_any_fill_rate_given_fill_known"),
        pl.when(known > 0).then(pl.col("raw_full_fill_candidates") / known).otherwise(None).alias("raw_full_fill_rate_given_fill_known"),
        pl.when(known > 0).then(pl.col("raw_partial_fill_candidates") / known).otherwise(None).alias("raw_partial_fill_rate_given_fill_known"),
        pl.when(denominator > 0).then(pl.col("exit_hedge_executable_candidates") / denominator).otherwise(None).alias("exit_hedge_executable_rate_per_candidate"),
        pl.when(full > 0).then(pl.col("exit_hedge_executable_candidates") / full).otherwise(None).alias("exit_hedge_executable_rate_per_full_fill"),
        pl.when(observed > 0).then(pl.col("exit_hedge_fresh_le_100ms_candidates") / observed).otherwise(None).alias("exit_hedge_fresh_le_100ms_rate_when_observed"),
        pl.when(observed > 0).then(pl.col("exit_hedge_fresh_le_1s_candidates") / observed).otherwise(None).alias("exit_hedge_fresh_le_1s_rate_when_observed"),
        (observed > 0).alias("exit_hedge_book_age_available"),
        pl.lit(EXPECTED_HEDGE_DELAY_NS / 1_000_000.0).alias("exit_hedge_delay_ms"),
        pl.lit(False).alias("cancel_ack_observation_available"),
        pl.lit(True).alias("raw_cancel_rate_is_request_only"),
        pl.lit("any_fill_non_null").alias("raw_fill_known_definition"),
        pl.lit(True).alias("all_candidate_fill_rates_are_lower_bounds"),
        pl.lit(True).alias("alternative_policy_analysis"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("boundary_quantile_alias_rows_safe_to_sum"),
        pl.lit(False).alias("joint_volume_allocated"),
    )
    return result.sort(list(keys))


def _strict_branch_table(policies: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    denominators = policies.group_by(list(keys)).agg(
        pl.len().alias("established_policy_trials")
    )
    result = policies.group_by([*keys, "branch_status"]).agg(
        pl.len().alias("strict_branch_trials"),
        pl.col("terminal_outcome").sum().alias("strict_terminal_outcome_trials"),
    ).join(denominators, on=list(keys), how="left", validate="m:1")
    return result.with_columns(
        (pl.col("strict_branch_trials") / pl.col("established_policy_trials")).alias("strict_branch_rate"),
        pl.lit(False).alias("cancel_ack_observation_available"),
        pl.lit(False).alias("strict_branch_rows_safe_to_sum_across_exit_policies"),
    ).sort([*keys, "branch_status"])


def _threshold_lineage_table(policies: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "boundary_quantile",
        "entry_lookup_action_id",
        "entry_parameter_version",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
        "exit_route",
        "exit_policy_trial_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "exit_rule_source_age_calendar_days",
        "source_strictly_precedes_target_date",
    ]
    return policies.select(*columns).with_columns(
        pl.concat_str(
            [
                "entry_parameter_version",
                "exit_rule_id",
                "exit_rule_source_asof_date",
                pl.col("exit_threshold_basis_bp").cast(pl.String),
            ],
            separator="|",
        ).alias("frozen_exit_policy_signature"),
        pl.lit("frozen_prior_session_exit_rule").alias(
            "threshold_lineage_semantics"
        ),
        pl.lit(True).alias("alternative_policy_analysis"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("boundary_quantile_alias_rows_safe_to_sum"),
    ).sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "boundary_quantile",
            "exit_rule_id",
            "exit_route",
            "entry_policy_generation_id",
        ]
    )


def _matched_comparison(
    policies: pl.DataFrame,
    taker_exits: pl.DataFrame,
    *,
    cost_bp: float,
    taker_exit_grid_ns: object,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if policies.is_empty() or taker_exits.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    if (
        isinstance(taker_exit_grid_ns, bool)
        or not isinstance(taker_exit_grid_ns, int)
        or taker_exit_grid_ns <= 0
    ):
        raise ValueError("matched T/T comparison requires its positive exit grid_ns")
    baseline_key = ["policy_generation_id", "exit_rule_id"]
    if taker_exits.select(baseline_key).n_unique() != taker_exits.height:
        raise ValueError("taker/taker facts must be unique by entry alias and exit rule")
    taker_latencies = taker_exits["exit_added_latency_ns"].drop_nulls().unique()
    if taker_latencies.len() != 1 or taker_exits["exit_added_latency_ns"].null_count():
        raise ValueError("T/T baseline must have one explicit exit_added_latency_ns")
    baseline = taker_exits.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        "exit_rule_id",
        pl.col("Date").cast(pl.String).alias("_tt_Date"),
        pl.col("ValueCode").cast(pl.String).alias("_tt_ValueCode"),
        pl.col("QuoteCode").cast(pl.String).alias("_tt_QuoteCode"),
        pl.col("route").alias("_tt_entry_route"),
        pl.col("exit_threshold_basis_bp").alias(
            "taker_taker_exit_threshold_basis_bp"
        ),
        pl.col("exit_rule_source_asof_date").alias(
            "taker_taker_exit_rule_source_asof_date"
        ),
        pl.col("branch_status").alias("taker_taker_branch_status"),
        pl.col("terminal_outcome").alias("taker_taker_terminal_outcome"),
        pl.col("exit_decision_time_ns").alias("taker_taker_exit_decision_time_ns"),
        pl.col("gross_cycle_pnl_twd").alias("taker_taker_gross_cycle_pnl_twd"),
        pl.col("exit_added_latency_ns").alias("taker_taker_exit_added_latency_ns"),
    )
    pairs = policies.join(
        baseline,
        on=["entry_policy_generation_id", "exit_rule_id"],
        how="left",
        validate="m:1",
    )
    if pairs.filter(pl.col("_tt_Date").is_null()).height:
        raise ValueError("maker policy is missing its same-alias/rule T/T baseline")
    for left, right in (
        ("Date", "_tt_Date"),
        ("ValueCode", "_tt_ValueCode"),
        ("QuoteCode", "_tt_QuoteCode"),
        ("entry_route", "_tt_entry_route"),
    ):
        if pairs.filter((pl.col(left).cast(pl.String) != pl.col(right).cast(pl.String)).fill_null(True)).height:
            raise ValueError(f"maker/T/T matched identity disagrees on {left}")
    signature_mismatch = pairs.filter(
        (
            pl.col("exit_threshold_basis_bp")
            != pl.col("taker_taker_exit_threshold_basis_bp")
        ).fill_null(True)
        | (
            pl.col("exit_rule_source_asof_date").cast(pl.String)
            != pl.col("taker_taker_exit_rule_source_asof_date").cast(pl.String)
        ).fill_null(True)
    )
    if signature_mismatch.height:
        raise ValueError("maker/T/T matched exit-rule threshold/source disagree")

    notional = pl.col("entry_spot_price") * pl.col("entry_hedge_contract_size_shares")
    maker_same = pl.col("nominal_outcome_category") == "same_day"
    tt_same = (
        (pl.col("taker_taker_branch_status") == "same_day_taker_exit")
        & (pl.col("taker_taker_terminal_outcome") == True)  # noqa: E712
        & pl.col("taker_taker_gross_cycle_pnl_twd").is_not_null()
    )
    pairs = pairs.with_columns(
        pl.concat_str(
            ["Date", "ValueCode", "entry_policy_generation_id", "exit_rule_id"],
            separator="|",
        ).alias("taker_taker_baseline_ref_id"),
        maker_same.alias("maker_nominal_same_day_completed"),
        tt_same.alias("taker_taker_same_day_completed"),
        pl.when(tt_same)
        .then(pl.col("taker_taker_gross_cycle_pnl_twd") / notional * 10_000.0)
        .otherwise(None)
        .alias("taker_taker_gross_cycle_bp"),
    ).with_columns(
        (pl.col("maker_nominal_same_day_completed") & pl.col("taker_taker_same_day_completed")).alias("both_same_day_completed"),
        (pl.col("nominal_gross_cycle_bp") - cost_bp).alias("maker_nominal_gross_minus_cost_bp"),
        (pl.col("taker_taker_gross_cycle_bp") - cost_bp).alias("taker_taker_gross_minus_cost_bp"),
    ).with_columns(
        pl.when(pl.col("both_same_day_completed"))
        .then(pl.col("nominal_gross_cycle_bp") - pl.col("taker_taker_gross_cycle_bp"))
        .otherwise(None)
        .alias("paired_maker_minus_taker_gross_bp"),
    )
    counts = pairs.group_by("taker_taker_baseline_ref_id").agg(
        pl.len().alias("maker_exit_route_matches_for_baseline"),
        pl.col("exit_route").n_unique().alias("unique_maker_exit_routes_for_baseline"),
    )
    pairs = pairs.join(counts, on="taker_taker_baseline_ref_id", how="left", validate="m:1").with_columns(
        (1.0 / pl.col("maker_exit_route_matches_for_baseline")).alias("baseline_reference_weight"),
        pl.lit(True).alias("alternative_policy_comparison"),
        pl.lit(True).alias("taker_baseline_shared_across_exit_routes"),
        pl.lit(False).alias("route_copy_is_independent_baseline"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("boundary_quantile_alias_rows_safe_to_sum"),
        pl.lit(cost_bp).alias("assumed_non_price_cycle_cost_bp"),
        pl.lit(NON_PRICE_COST_SCOPE).alias("cost_scope"),
        pl.lit(EXPECTED_HEDGE_DELAY_NS).alias("maker_exit_hedge_delay_ns"),
        pl.lit(taker_exit_grid_ns).alias("taker_taker_exit_grid_ns"),
        (
            pl.col("taker_taker_exit_added_latency_ns")
            == EXPECTED_HEDGE_DELAY_NS
        ).alias("latency_matched"),
        pl.when(pl.col("taker_taker_exit_added_latency_ns") == 0)
        .then(pl.lit("optimistic_zero_added_latency_benchmark"))
        .otherwise(pl.lit("observed_added_latency_benchmark"))
        .alias("taker_baseline_role"),
        pl.lit(True).alias("maker_exit_hedge_slippage_already_in_gross"),
        pl.lit(False).alias("maker_exit_hedge_slippage_subtracted_again"),
    )
    pair_columns = [
        "Date", "ValueCode", "QuoteCode", "entry_route", "boundary_quantile",
        "entry_lookup_action_id", "entry_policy_generation_id", "entry_raw_order_fact_id",
        "exit_rule_id", "exit_route", "exit_policy_trial_id", "taker_taker_baseline_ref_id",
        "branch_status", "nominal_instant_cancel_v0_branch", "taker_taker_branch_status",
        "exit_threshold_basis_bp", "exit_rule_source_asof_date",
        "taker_taker_exit_threshold_basis_bp",
        "taker_taker_exit_rule_source_asof_date",
        "maker_nominal_same_day_completed", "taker_taker_same_day_completed", "both_same_day_completed",
        "nominal_gross_cycle_bp", "taker_taker_gross_cycle_bp",
        "exit_hedge_signed_total_slippage_bp",
        "maker_nominal_gross_minus_cost_bp", "taker_taker_gross_minus_cost_bp",
        "paired_maker_minus_taker_gross_bp", "baseline_reference_weight",
        "maker_exit_route_matches_for_baseline", "unique_maker_exit_routes_for_baseline",
        "alternative_policy_comparison", "taker_baseline_shared_across_exit_routes",
        "route_copy_is_independent_baseline", "exit_route_rows_safe_to_sum",
        "exit_rule_rows_safe_to_sum", "boundary_quantile_alias_rows_safe_to_sum",
        "assumed_non_price_cycle_cost_bp", "cost_scope",
        "taker_taker_exit_added_latency_ns", "maker_exit_hedge_delay_ns",
        "taker_taker_exit_grid_ns",
        "latency_matched", "taker_baseline_role",
        "maker_exit_hedge_slippage_already_in_gross",
        "maker_exit_hedge_slippage_subtracted_again",
    ]
    pairs = pairs.select(pair_columns).sort(
        ["Date", "ValueCode", "entry_route", "boundary_quantile", "exit_rule_id", "exit_route", "entry_policy_generation_id"]
    )
    summary = pairs.group_by(PRODUCT_POLICY_KEY).agg(
        pl.len().alias("matched_policy_pairs"),
        pl.col("taker_taker_baseline_ref_id").n_unique().alias("unique_taker_taker_baseline_refs"),
        pl.col("baseline_reference_weight").sum().alias("baseline_effective_observations"),
        pl.col("maker_nominal_same_day_completed").sum().alias("maker_nominal_same_day_pairs"),
        pl.col("taker_taker_same_day_completed").sum().alias("taker_taker_same_day_pairs"),
        pl.col("both_same_day_completed").sum().alias("both_same_day_pairs"),
        pl.col("paired_maker_minus_taker_gross_bp").drop_nulls().median().alias("paired_maker_minus_taker_gross_bp_p50"),
        pl.col("paired_maker_minus_taker_gross_bp").drop_nulls().quantile(0.10).alias("paired_maker_minus_taker_gross_bp_p10"),
        pl.col("paired_maker_minus_taker_gross_bp").drop_nulls().quantile(0.90).alias("paired_maker_minus_taker_gross_bp_p90"),
        pl.col("taker_taker_exit_added_latency_ns").first().alias("taker_taker_exit_added_latency_ns"),
        pl.col("maker_exit_hedge_delay_ns").first().alias("maker_exit_hedge_delay_ns"),
        pl.col("taker_taker_exit_grid_ns").first().alias("taker_taker_exit_grid_ns"),
        pl.col("latency_matched").all().alias("latency_matched"),
        pl.col("taker_baseline_role").first().alias("taker_baseline_role"),
    ).with_columns(
        (pl.col("maker_nominal_same_day_pairs") / pl.col("matched_policy_pairs")).alias("maker_nominal_same_day_rate"),
        (pl.col("taker_taker_same_day_pairs") / pl.col("matched_policy_pairs")).alias("taker_taker_same_day_rate"),
        (pl.col("both_same_day_pairs") / pl.col("matched_policy_pairs")).alias("both_same_day_rate"),
        pl.lit(True).alias("alternative_policy_comparison"),
        pl.lit(True).alias("taker_baseline_shared_across_exit_routes"),
        pl.lit(False).alias("exit_route_rows_safe_to_sum"),
        pl.lit(False).alias("exit_rule_rows_safe_to_sum"),
        pl.lit(False).alias("boundary_quantile_alias_rows_safe_to_sum"),
    ).sort(PRODUCT_POLICY_KEY)
    return pairs, summary


def _validate_joined_action_identity(frame: pl.DataFrame) -> None:
    if frame.filter(pl.col("_action_Date").is_null()).height:
        raise ValueError("exit-maker fact does not resolve to an entry action alias")
    for left, right in (
        ("Date", "_action_Date"),
        ("ValueCode", "_action_ValueCode"),
        ("QuoteCode", "_action_QuoteCode"),
        ("entry_route", "_action_entry_route"),
        ("entry_raw_order_fact_id", "_action_entry_raw_order_fact_id"),
    ):
        if frame.filter((pl.col(left).cast(pl.String) != pl.col(right).cast(pl.String)).fill_null(True)).height:
            raise ValueError(f"exit-maker and entry action identities disagree on {left}")


def _validate_canonical_raw_ids(raw: pl.DataFrame) -> None:
    for row in raw.iter_rows(named=True):
        components = (
            str(row["entry_raw_order_fact_id"]),
            str(row["exit_route"]),
            int(row["spread_pair_epoch"]),
            int(row["target_price_tick"]),
            int(row["submit_recv_time_ns"]),
            int(row["submit_event_sequence"]),
            int(row["submit_row_index"]),
        )
        digest = hashlib.sha256("|".join(map(str, components)).encode("utf-8")).hexdigest()[:24]
        expected = f"exit-raw-{digest}"
        if str(row["exit_raw_candidate_fact_id"]) != expected:
            raise ValueError("exit raw candidate id is not the canonical rule-free identity")


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
        payload = _read_json(marker)
        if payload.get("complete") is not True:
            continue
        if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
            raise ValueError(f"exit-maker marker identity mismatch: {marker}")
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


def _validate_exit_marker(
    payload: Mapping[str, object],
    partition: Path,
    *,
    validate_hashes: bool,
) -> Mapping[str, object]:
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"invalid exit-maker config marker: {partition}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"exit-maker config hash mismatch: {partition}")
    runner = config.get("runner")
    source = config.get("source")
    if not isinstance(runner, dict) or not isinstance(source, dict):
        raise ValueError(f"exit-maker marker lacks runner/source config: {partition}")
    if _canonical_sha256(runner) != payload.get("runner_config_sha256"):
        raise ValueError(f"exit-maker runner config hash mismatch: {partition}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_EXIT_ARTIFACTS):
        raise ValueError(f"exit-maker artifact set mismatch: {partition}")
    _validate_artifacts(partition, artifacts, validate_hashes=validate_hashes)
    return config


def _load_and_validate_entry_source(
    partition: Path,
    exit_source: Mapping[str, object],
    *,
    date: str,
    value_code: str,
    validate_hashes: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, str, Mapping[str, object]]:
    marker = partition / "complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"entry source partition is incomplete: {marker}")
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"entry source marker is incomplete: {marker}")
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"entry source marker identity mismatch: {marker}")
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"entry source config is invalid: {marker}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"entry source config hash mismatch: {marker}")
    if str(exit_source.get("upstream_config_sha256")) != config_sha:
        raise ValueError(f"exit-maker source binds a different entry config: {marker}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"entry source artifact manifest is invalid: {marker}")
    source_specs = (
        ("action_source", _ENTRY_ACTION_ARTIFACT),
        ("exit_rule_source", _ENTRY_EXIT_ARTIFACT),
    )
    frames: list[pl.DataFrame] = []
    for source_name, expected_name in source_specs:
        declared_source = exit_source.get(source_name)
        if not isinstance(declared_source, dict):
            raise ValueError(f"exit-maker marker lacks {source_name}: {marker}")
        artifact_name = declared_source.get("artifact")
        if artifact_name != expected_name:
            raise ValueError(f"unexpected bound entry artifact for {source_name}: {artifact_name}")
        declared = artifacts.get(expected_name)
        if not isinstance(declared, dict):
            raise ValueError(f"entry marker lacks bound artifact: {expected_name}")
        if str(declared_source.get("sha256")) != str(declared.get("sha256")):
            raise ValueError(f"exit-maker source hash disagrees with entry marker: {expected_name}")
        path = partition / expected_name
        _validate_one_artifact(path, declared, validate_hashes=validate_hashes)
        frames.append(pl.read_parquet(path))
    return frames[0], frames[1], config_sha, config


def _validate_entry_action_execution_contract(
    actions: pl.DataFrame,
    *,
    expected_hedge_delay_ns: int = EXPECTED_HEDGE_DELAY_NS,
) -> pl.DataFrame:
    """Validate fill flags and derive the observed entry hedge delay."""

    contract_schema = {
        "raw_order_fact_id": pl.String,
        "any_fill": pl.Boolean,
        "full_fill": pl.Boolean,
        "partial_fill": pl.Boolean,
        "full_fill_recv_time_ns": pl.Int64,
        "entry_hedge_status": pl.String,
        "entry_hedge_decision_time_ns": pl.Int64,
        "entry_hedge_label_observed": pl.Boolean,
        "entry_hedge_executable": pl.Boolean,
    }
    if actions.is_empty():
        # Real zero-action partitions intentionally use the narrow execution
        # schema and omit outcome/hedge columns altogether.  Add only typed
        # empty columns here; nonempty partitions remain strictly required and
        # validated below.
        return actions.with_columns(
            *(
                pl.col(name).cast(dtype)
                if name in actions.columns
                else pl.lit(None, dtype=dtype).alias(name)
                for name, dtype in contract_schema.items()
            ),
            pl.lit(None, dtype=pl.Int64).alias(
                "entry_action_hedge_delay_ns"
            )
        )
    _require(actions, contract_schema, "entry execution actions")

    any_true = pl.col("any_fill").fill_null(False)
    full_true = pl.col("full_fill").fill_null(False)
    partial_true = pl.col("partial_fill").fill_null(False)
    invalid_fill = actions.filter(
        (full_true & (~any_true | partial_true))
        | (partial_true & (~any_true | full_true))
        | (
            (pl.col("any_fill") == False)  # noqa: E712
            & (
                (pl.col("full_fill") != False).fill_null(True)  # noqa: E712
                | (pl.col("partial_fill") != False).fill_null(True)  # noqa: E712
            )
        )
        | (
            pl.col("any_fill").is_null()
            & (pl.col("full_fill").is_not_null() | partial_true)
        )
        | (any_true & ~(full_true | partial_true))
    )
    if invalid_fill.height:
        raise ValueError(
            "entry action fill flags are contradictory: full/partial/any_fill"
        )

    observed_delay = (
        pl.col("entry_hedge_decision_time_ns")
        - pl.col("full_fill_recv_time_ns")
    )
    executable_from_status = pl.col("entry_hedge_status") == "executable"
    invalid_hedge = actions.filter(
        full_true
        & (
            pl.col("full_fill_recv_time_ns").is_null()
            | pl.col("entry_hedge_decision_time_ns").is_null()
            | (observed_delay != expected_hedge_delay_ns).fill_null(True)
            | (
                pl.col("entry_hedge_label_observed") != True  # noqa: E712
            ).fill_null(True)
            | pl.col("entry_hedge_status").is_null()
            | (
                pl.col("entry_hedge_executable")
                != executable_from_status
            ).fill_null(True)
        )
    )
    if invalid_hedge.height:
        raise ValueError(
            "full-filled entry action has incoherent 50 ms hedge facts"
        )
    # The status/cursor belong to the canonical physical raw order.  Aliases
    # with different stop cursors may share that raw order, so a no-fill alias
    # can retain the status/cursor established by a longer-lived full-fill
    # alias.  Admission is therefore governed by the explicit alias-local
    # label booleans, which must remain false on every non-full-filled alias.
    invalid_nonfull_hedge = actions.filter(
        ~full_true
        & (
            pl.col("full_fill_recv_time_ns").is_not_null()
            | (
                pl.col("entry_hedge_label_observed") != False  # noqa: E712
            ).fill_null(True)
            | (
                pl.col("entry_hedge_executable") != False  # noqa: E712
            ).fill_null(True)
        )
    )
    if invalid_nonfull_hedge.height:
        raise ValueError(
            "non-full-filled entry action carries contradictory hedge facts"
        )

    raw_groups = actions.group_by("raw_order_fact_id").agg(
        pl.col("entry_hedge_status")
        .drop_nulls()
        .n_unique()
        .alias("_physical_status_values"),
        pl.col("entry_hedge_decision_time_ns")
        .drop_nulls()
        .n_unique()
        .alias("_physical_decision_values"),
        pl.col("entry_hedge_status")
        .is_not_null()
        .sum()
        .alias("_physical_status_rows"),
        pl.col("entry_hedge_decision_time_ns")
        .is_not_null()
        .sum()
        .alias("_physical_decision_rows"),
        full_true.sum().alias("_full_aliases"),
        pl.col("full_fill_recv_time_ns")
        .filter(full_true)
        .drop_nulls()
        .n_unique()
        .alias("_full_fill_cursor_values"),
    )
    invalid_raw_groups = raw_groups.filter(
        pl.col("raw_order_fact_id").is_null()
        | (pl.col("_physical_status_values") > 1)
        | (pl.col("_physical_decision_values") > 1)
        | (pl.col("_full_fill_cursor_values") > 1)
        | (
            (pl.col("_physical_status_rows") > 0)
            != (pl.col("_physical_decision_rows") > 0)
        )
        | (
            (
                (pl.col("_physical_status_rows") > 0)
                | (pl.col("_physical_decision_rows") > 0)
            )
            & (pl.col("_full_aliases") == 0)
        )
    )
    if invalid_raw_groups.height:
        raise ValueError(
            "entry aliases disagree on canonical physical hedge facts"
        )
    return actions.with_columns(
        pl.when(full_true)
        .then(observed_delay)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("entry_action_hedge_delay_ns")
    )


def _expected_session_predecessors(
    session_calendar: Sequence[str] | None,
    target_dates: Sequence[str],
) -> dict[str, str] | None:
    if session_calendar is None:
        return None
    sessions = tuple(str(value) for value in session_calendar)
    if (
        not sessions
        or sessions != tuple(sorted(sessions))
        or len(sessions) != len(set(sessions))
        or any(len(value) != 8 or not value.isdigit() for value in sessions)
    ):
        raise ValueError("session calendar must be unique ascending YYYYMMDD")
    index = {value: offset for offset, value in enumerate(sessions)}
    predecessors: dict[str, str] = {}
    for date in target_dates:
        offset = index.get(str(date))
        if offset is None or offset == 0:
            raise ValueError(
                f"session calendar lacks a predecessor for target Date {date}"
            )
        predecessors[str(date)] = sessions[offset - 1]
    return predecessors


def _validate_position_rule_lineage(
    positions: pl.DataFrame,
    entry_exit_facts: pl.DataFrame,
    *,
    expected_session_predecessors: Mapping[str, str] | None,
) -> dict[str, object]:
    """Bind every same-day position rule to immutable entry ``exit_facts``."""

    if positions.is_empty() and entry_exit_facts.is_empty():
        return {
            "validated_position_rows": 0,
            "bound_entry_exit_rule_rows": 0,
            "expected_session_predecessor_exact": (
                expected_session_predecessors is not None
            ),
            "bound_rule_lineage_sha256": _canonical_sha256([]),
        }

    _require(
        positions,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "entry_policy_generation_id",
            "entry_raw_order_fact_id",
            "exit_rule_id",
            "exit_threshold_basis_bp",
            "exit_rule_source_asof_date",
        },
        "exit-maker position policies",
    )
    _require(
        entry_exit_facts,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "policy_generation_id",
            "raw_order_fact_id",
            "exit_rule_id",
            "exit_threshold_basis_bp",
            "exit_rule_source_asof_date",
        },
        "bound entry exit facts",
    )
    position_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "exit_rule_id",
    ]
    exit_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "policy_generation_id",
        "raw_order_fact_id",
        "exit_rule_id",
    ]
    positions_normalized = positions.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("exit_rule_source_asof_date").cast(pl.String),
    )
    exit_rules = entry_exit_facts.select(
        *exit_key,
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
    ).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("exit_rule_source_asof_date").cast(pl.String),
    )
    if exit_rules.select(exit_key).n_unique() != exit_rules.height:
        raise ValueError("bound entry exit facts duplicate a frozen rule key")
    for frame, label in (
        (positions_normalized, "same-day position"),
        (exit_rules, "bound entry exit fact"),
    ):
        if frame.filter(
            pl.col("exit_rule_source_asof_date").is_null()
            | (
                pl.col("exit_rule_source_asof_date")
                >= pl.col("Date")
            )
        ).height:
            raise ValueError(
                f"{label} exit_rule_source_asof_date must precede Date"
            )
        if expected_session_predecessors is not None:
            expected = pl.DataFrame(
                {
                    "Date": list(expected_session_predecessors),
                    "_expected_session_predecessor": list(
                        expected_session_predecessors.values()
                    ),
                }
            )
            checked = frame.join(expected, on="Date", how="left", validate="m:1")
            if checked.filter(
                pl.col("_expected_session_predecessor").is_null()
                | (
                    pl.col("exit_rule_source_asof_date")
                    != pl.col("_expected_session_predecessor")
                )
            ).height:
                raise ValueError(
                    f"{label} exit_rule_source_asof_date is not the session predecessor"
                )

    expected_rules = exit_rules.rename(
        {
            "route": "entry_route",
            "policy_generation_id": "entry_policy_generation_id",
            "raw_order_fact_id": "entry_raw_order_fact_id",
            "exit_threshold_basis_bp": "_bound_exit_threshold_basis_bp",
            "exit_rule_source_asof_date": "_bound_exit_rule_source_asof_date",
        }
    )
    checked = positions_normalized.join(
        expected_rules,
        on=position_key,
        how="left",
        validate="m:1",
    )
    if checked.filter(
        pl.col("_bound_exit_threshold_basis_bp").is_null()
        | ~pl.col("_bound_exit_threshold_basis_bp").is_finite()
        | pl.col("exit_threshold_basis_bp").is_null()
        | ~pl.col("exit_threshold_basis_bp").is_finite()
        | (
            pl.col("exit_threshold_basis_bp")
            != pl.col("_bound_exit_threshold_basis_bp")
        ).fill_null(True)
        | (
            pl.col("exit_rule_source_asof_date")
            != pl.col("_bound_exit_rule_source_asof_date")
        ).fill_null(True)
    ).height:
        raise ValueError(
            "same-day position identity/threshold/source differs from bound entry exit_facts"
        )
    lineage = expected_rules.select(
        *position_key,
        "_bound_exit_threshold_basis_bp",
        "_bound_exit_rule_source_asof_date",
    ).sort(position_key)
    return {
        "validated_position_rows": positions.height,
        "bound_entry_exit_rule_rows": exit_rules.height,
        "expected_session_predecessor_exact": (
            expected_session_predecessors is not None
        ),
        "bound_rule_lineage_sha256": _canonical_sha256(lineage.to_dicts()),
    }


def _validate_root_manifest(
    root: Path,
    selected: Sequence[Mapping[str, object]],
    *,
    validate_hashes: bool,
    required: bool,
) -> dict[str, object]:
    manifest_path = root / "exit_maker_partition_manifest.parquet"
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(
                f"formal exit-maker root manifest does not exist: {manifest_path}"
            )
        return {
            "same_day_root_manifest_required": False,
            "same_day_root_manifest_crossvalidated": False,
            "same_day_root_manifest_path": None,
            "same_day_root_manifest_sha256": None,
        }
    manifest = pl.read_parquet(manifest_path).with_columns(
        pl.col("Date").cast(pl.String), pl.col("ValueCode").cast(pl.String)
    )
    required_columns = {
        "Date",
        "ValueCode",
        "runner_config_sha256",
        "config_sha256",
        "complete",
    }
    _require(manifest, required_columns, "exit-maker root manifest")
    if manifest.select("Date", "ValueCode").n_unique() != manifest.height:
        raise ValueError("exit-maker root manifest has duplicate product-days")
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
            raise ValueError(f"root manifest missing complete partition {key}")
        if (
            str(row["runner_config_sha256"]) != str(payload["runner_config_sha256"])
            or str(row["config_sha256"]) != str(payload["config_sha256"])
        ):
            raise ValueError(f"root manifest hash mismatch for {key}")
    # The manifest itself has no completion sidecar.  Its content-to-marker
    # checks above are the relevant integrity test; ``validate_hashes`` is
    # accepted to keep this validator's call contract explicit.
    _ = validate_hashes
    return {
        "same_day_root_manifest_required": required,
        "same_day_root_manifest_crossvalidated": True,
        "same_day_root_manifest_path": str(manifest_path),
        "same_day_root_manifest_sha256": _file_sha256(manifest_path),
        "same_day_root_manifest_rows": manifest.height,
    }


def _load_session_calendar(path: Path) -> tuple[str, ...]:
    calendar_path = Path(path)
    if not calendar_path.is_file():
        raise FileNotFoundError(calendar_path)
    sessions = tuple(
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not sessions:
        raise ValueError("session calendar is empty")
    # Reuse the exact predecessor helper for ordering/date validation.
    _expected_session_predecessors(sessions, sessions[1:])
    return sessions


def _validate_artifacts(
    partition: Path,
    artifacts: Mapping[str, object],
    *,
    validate_hashes: bool,
) -> None:
    for name, declared in artifacts.items():
        if Path(name).name != name or not isinstance(declared, dict):
            raise ValueError(f"invalid artifact declaration: {partition}/{name}")
        _validate_one_artifact(
            partition / name,
            declared,
            validate_hashes=validate_hashes,
        )


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
    schema = pl.read_parquet_schema(path)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    if int(declared.get("rows", -1)) != rows:
        raise ValueError(f"artifact row count mismatch: {path}")
    if int(declared.get("columns", -1)) != len(schema):
        raise ValueError(f"artifact column count mismatch: {path}")


def _publish_report(report: ExitMakerReport, destination: Path) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    tables = {
        "coverage.csv": report.coverage,
        "daily_exit_maker_policy.csv": report.daily_policy,
        "product_exit_maker_policy.csv": report.product_policy,
        "daily_exit_maker_strict_branch.csv": report.daily_strict_branch,
        "product_exit_maker_strict_branch.csv": report.product_strict_branch,
        "exit_policy_threshold_lineage.csv": report.policy_threshold_lineage,
        "matched_exit_style_pairs.csv": report.matched_pairs,
        "matched_exit_style_summary.csv": report.matched_summary,
    }
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name, frame in tables.items():
            path = stage / name
            frame.write_csv(path)
            artifacts[name] = {
                "rows": frame.height,
                "columns": frame.width,
                "sha256": _file_sha256(path),
            }
        metadata = dict(report.metadata)
        metadata["complete"] = True
        metadata["artifacts"] = artifacts
        marker = stage / "report_complete.json"
        marker.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


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


def _validate_cost(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("assumed non-price cost must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("assumed non-price cost must be finite and non-negative")
    return result


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON marker: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON marker must be an object: {path}")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--require-balanced-product-days", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument(
        "--session-calendar",
        type=Path,
        default=DEFAULT_SESSION_CALENDAR_PATH,
        help="Immutable ordered session calendar used for exact D-1 validation",
    )
    parser.add_argument("--assumed-non-price-cost-bp", type=float, default=DEFAULT_NON_PRICE_COST_BP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_exit_maker_report(
        args.exit_root,
        args.entry_root,
        output_dir=args.output_dir,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        require_balanced_product_days=args.require_balanced_product_days,
        validate_hashes=not args.skip_hash_validation,
        assumed_non_price_cost_bp=args.assumed_non_price_cost_bp,
        session_calendar_path=args.session_calendar,
    )
    print(json.dumps(dict(report.metadata), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests.
    raise SystemExit(main())
