"""Source-bound analysis bundle for the 13:00 dynamic B1/A1 challenger.

This module does not modify the frozen formal exit producers.  It turns one
market replay template per product-day into a controller-ready, non-additive
position overlay.  The overlay is intentionally not a portfolio execution
claim: cache labels do not allocate shared market volume and the candidate
cache is only artifact-hash-bound, while its upstream raw fingerprint is
stat/mtime based.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl

from .aggressive_1300_exit import (
    AGGRESSIVE_1300_POLICY_VERSION,
    aggressive_1300_start_cursor,
    replay_aggressive_1300_product_day,
    select_open_positions_at_1300,
)
from .combined_cost_cap_sweep import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_UNIVERSE_ROOT,
    TransactionCostProfile,
    _partition_manifest,
    _read_bound_partition_artifact,
    load_verified_source,
)
from .exit_maker import EXIT_MAKER_ROUTE_CONTRACTS
from .merged import session_cutoff_cursor
from .raw_tape import RawTapeDay
from .targets import absolute_price_tick, tick_index_to_price


ANALYSIS_VERSION = (
    "aggressive_1300_daily_inventory_templates_v3_post13_timeline_fix"
)
BUNDLE_SCHEMA_VERSION = "aggressive_1300_analysis_bundle_v3_post13_timeline_fix"
SEED_TEMPLATE_SHA256 = "68505c2ff20cbc8cc90e7f871006121372b88bade24d947341195e642a28c9da"
SEED_TEMPLATE_ROWS = 427
SEED_GENERATOR_SHA256 = "734fa4211e1871db408b5dc716a991a8e2a65b62c23d3278086c5e0499a5fa49"
SEED_SOURCE_COMPLETE_SHA256 = "8d6f0a3b51189e84d7e96d36ab50400a46f087b379cf407b157d2f028ace55c4"
FORMAL_SESSION_CALENDAR_SHA256 = "54f9b4f75b8cca800721f3c7d2c6b14134df1dd6a3e700c7951a371c1e2ba980"
FORMAL_SESSION_CALENDAR_COUNT = 63
EXPECTED_CACHE_IMPLEMENTATION = {
    "execution_runner.py": "6d9b64a15d8e5539155b0560d87d27a195cf0c67614795bf3c73210fe8fe4825",
    "exit_maker_cross_session_runner.py": "361f599a7006ebfea09b255df4ca3d79808134417bd952d275a42fb04a638d80",
    "raw_tape.py": "7d8a19376e59f413c0fb6a5565116000f1c41ef0693112c325bfc8013111802a",
}
EXPECTED_CACHE_ARTIFACTS = (
    "future_states.parquet",
    "future_trades.parquet",
    "mapping.parquet",
    "raw_audit.parquet",
    "spot_states.parquet",
    "spot_trades.parquet",
    "spread_pair_clock.parquet",
)

DEFAULT_SEED_TEMPLATE = Path(
    "maker/data/walkforward/aggressive_1300_seed_templates_20260821_v1/"
    "market_exit_templates.parquet"
)
DEFAULT_SUPPLEMENTAL_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close"
)
SUPPLEMENTAL_COMPLETE_SHA256 = "402d050e6fed75c38cea540fd31dd09994f6a24ef91a5411ea4e715e14cb691d"
SUPPLEMENTAL_MARKER_PAYLOAD_SHA256 = (
    "173a98dab1d029b8d3e2f523eabf52f050c8b539a890f1b6aed05c7f71b6f337"
)
SUPPLEMENTAL_PATH_SHA256 = "63742b773f32cdb4851fe686aea8c1dd8349d27b42be7460045dd73ec23cfda6"
SUPPLEMENTAL_PATH_ROWS = 3672
UPSTREAM_SUPPLEMENTAL_V2_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback"
)
UPSTREAM_SUPPLEMENTAL_V2_COMPLETE_SHA256 = (
    "f0902805f2cd555340e09d38232ba0c3a694a56df2ff11043f338cfdfff3c0f2"
)
UPSTREAM_SUPPLEMENTAL_V2_MARKER_PAYLOAD_SHA256 = (
    "c7c4860f94064b438aea671182b93012dbb6cbbbd833914daf42c683ea9f0566"
)
UPSTREAM_SUPPLEMENTAL_V2_PATH_SHA256 = (
    "38329217a7504ca219ee83a016a99c63208bfd32de17ec4a64de8967ab1752cf"
)
EXPIRY_CLOSE_FACT_ROOT = Path(
    "maker/data/walkforward/expiry_daily_close_facts_20260821_v1"
)
EXPIRY_CLOSE_FACT_COMPLETE_SHA256 = (
    "813d2b484214f58cbe8f07815d1999ee097f206955b128ec26602aee0ddfdd1b"
)
EXPIRY_CLOSE_FACT_MARKER_PAYLOAD_SHA256 = (
    "737f4f44e125545d821aac990ea108ac9db5f7b29e8e31d10508ec9e9f32c04c"
)
EXPIRY_DAILY_CLOSE_FACT_SHA256 = (
    "9e6879c3104e17f4cb2baa5ba305a90c316ec63de00c05436cfddd9a9bc7f868"
)
NORMAL_CONTROL_ROOT = Path(
    "maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only"
)
NORMAL_CONTROL_COMPLETE_SHA256 = (
    "156715157532d99e75aa194ae5babd8a8e79d89c86aefd59b21a03569bd51c3c"
)
NORMAL_CONTROL_CAP_SUMMARY_SHA256 = (
    "4bc664221cc60a908a1a654d9d085bc72712501a72ad85318439f7ee80ade8f3"
)
DEFAULT_CACHE_ROOT = Path(
    "maker/data/walkforward/"
    "exit_maker_cross_session_narrow_60d_candidate_session_cache_v8"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/"
    "aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix"
)

ARTIFACTS = {
    "market_exit_templates.parquet": "market_templates",
    "daily_open_inventory.parquet": "daily_open_inventory",
    "position_template_overlay.parquet": "position_template_overlay",
    "missing_product_days.parquet": "missing_product_days",
    "cache_source_inventory.parquet": "cache_source_inventory",
    "entry_price_source_inventory.parquet": "entry_price_source_inventory",
    "template_branch_summary.parquet": "template_branch_summary",
    "coverage_summary.parquet": "coverage_summary",
    "controller_contract.parquet": "controller_contract",
    "controller_position_outcomes.parquet": "controller_position_outcomes",
    "controller_daily_control.parquet": "controller_daily_control",
    "controller_scenario_summary.parquet": "controller_scenario_summary",
}


@dataclass(frozen=True)
class Aggressive1300AnalysisConfig:
    portfolio_caps_twd: tuple[float, ...] = (
        10_000_000.0,
        20_000_000.0,
        30_000_000.0,
        40_000_000.0,
        50_000_000.0,
    )
    eod_target_ceiling_twd: float = 20_000_000.0
    per_product_cap_fraction: float = 0.30
    seed_template_sha256: str = SEED_TEMPLATE_SHA256
    policy_version: str = AGGRESSIVE_1300_POLICY_VERSION
    controller_ordering: str = (
        "winner_exit_decision_time_ns_then_position_established_ns_then_"
        "ValueCode_then_policy_path_id"
    )

    def validate(self) -> None:
        caps = tuple(float(value) for value in self.portfolio_caps_twd)
        if (
            not caps
            or any(not math.isfinite(value) or value <= 0 for value in caps)
            or caps != tuple(sorted(set(caps)))
        ):
            raise ValueError("portfolio caps must be finite, positive, unique, and sorted")
        if (
            not math.isfinite(float(self.eod_target_ceiling_twd))
            or self.eod_target_ceiling_twd <= 0
        ):
            raise ValueError("EOD target ceiling must be finite and positive")
        if (
            not math.isfinite(float(self.per_product_cap_fraction))
            or self.per_product_cap_fraction <= 0
            or self.per_product_cap_fraction > 1
        ):
            raise ValueError("per-product cap fraction must be in (0, 1]")
        if self.seed_template_sha256 != SEED_TEMPLATE_SHA256:
            raise ValueError("seed template identity changed")
        if self.policy_version != AGGRESSIVE_1300_POLICY_VERSION:
            raise ValueError("aggressive policy version changed")
        if self.controller_ordering != (
            "winner_exit_decision_time_ns_then_position_established_ns_then_"
            "ValueCode_then_policy_path_id"
        ):
            raise ValueError("controller ordering changed")


def _config_payload(config: Aggressive1300AnalysisConfig) -> dict[str, object]:
    config.validate()
    return {
        "portfolio_caps_twd": [float(value) for value in config.portfolio_caps_twd],
        "eod_target_ceiling_twd": float(config.eod_target_ceiling_twd),
        "per_product_cap_fraction": float(config.per_product_cap_fraction),
        "seed_template_sha256": config.seed_template_sha256,
        "policy_version": config.policy_version,
        "controller_ordering": config.controller_ordering,
    }


def _fact_semantics(
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> dict[str, object]:
    config.validate()
    return {
        "analysis_only": True,
        "daily_carry_inventory_rebuilt": True,
        "full_session_calendar_bound": True,
        "entry_origin_day_1300_cutoff_enforced": True,
        "forbidden_post1300_entries_do_not_advance_normal_exit_clock": True,
        "normal_terminal_at_or_before_1300_excluded": True,
        "normal_and_aggressive_exit_timestamp_competition": True,
        "normal_wins_equal_aggressive_decision_timestamp": True,
        "portfolio_target_cancel_clock": (
            "winner_exit_decision_time_ns_full_fill_plus_50ms"
        ),
        "cross_product_fills_within_50ms_cancel_race_modeled": False,
        "entry_wins_equal_normal_exit_timestamp_at_admission": True,
        "capacity_readmission_replayed_after_early_release": True,
        "held_product_with_start_of_day_carry_is_exit_only_all_session": True,
        "per_product_hard_cap_fraction": float(config.per_product_cap_fraction),
        "dynamic_b1a1_layered_peg": True,
        "nominal_cancel_model_only": True,
        "strict_cancel_ack_identified": False,
        "target_measurement_time": "13:20 Asia/Taipei study cutoff",
        "study_cutoff_is_twse_close": False,
        "extra_ten_minutes_to_twse_close_evaluated": False,
        "post_expiry_close_inventory_accounting_published": True,
        "post_expiry_close_is_full_1330_market_replay": False,
        "supplemental_v3_terminal_cashflows_all_priced": True,
        "expiry_close_uses_official_daily_close": True,
        "expiry_close_uses_official_settlement": False,
        "normal_control_10m_20m_identity_verified": True,
        "trade_turnover_metrics_published": True,
        "exit_one_way_turnover_basis": "actual_spot_exit_price_times_shares",
        "runtime_daily_liquidity_admission_gate_applied": False,
        "joint_volume_allocated": False,
        "position_labels_safe_to_sum": False,
        "optimistic_template_reuse_is_joint_volume_overcount_upper_bound": True,
        "all_close_pnl_published": False,
        "actual_notional_target_attainment_published": True,
        "formal_transaction_cost_profile_applied": True,
        "realized_pnl_only_no_mtm": True,
        "upstream_raw_content_integrity_bound": False,
        "universe_selection_d_safe_go": False,
        "formal_ev_ready": False,
        "production_strategy_go": False,
    }


@dataclass(frozen=True)
class Aggressive1300AnalysisResult:
    market_templates: pl.DataFrame
    daily_open_inventory: pl.DataFrame
    position_template_overlay: pl.DataFrame
    missing_product_days: pl.DataFrame
    cache_source_inventory: pl.DataFrame
    entry_price_source_inventory: pl.DataFrame
    template_branch_summary: pl.DataFrame
    coverage_summary: pl.DataFrame
    controller_contract: pl.DataFrame
    controller_position_outcomes: pl.DataFrame
    controller_daily_control: pl.DataFrame
    controller_scenario_summary: pl.DataFrame


def build_daily_open_inventory(
    selected_paths_with_entry_prices: pl.DataFrame,
    *,
    session_dates: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Expand selected physical paths to every observed date they are open.

    The frozen core selector correctly evaluates normal terminal state at a
    target date.  This post-only layer additionally enforces the strategy-wide
    rule that an entry established at or after 13:00 on its *origin* day never
    existed, including on later carry dates.
    """

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_raw_order_fact_id",
        "position_established_ns",
        "policy_path_id",
        "exit_policy_trial_id",
        "terminal_date",
        "exit_decision_time_ns",
        "terminal_cashflow_priced",
        "outcome_type",
        "outcome_status",
        "last_observed_session_date",
        "outstanding_interval_end_exclusive",
        "supplemental_terminal_resolution",
        "model_imputed_full_carry_on_unknown",
        "double_exit_bias_possible",
        "expiry_uses_last_valid_session_mark",
        "expiry_uses_last_observed_session_mark",
        "expiry_mark_uses_trade_fallback",
        "spot_expiry_mark_is_executable_bbo",
        "future_expiry_mark_is_executable_bbo",
        "expiry_mark_is_official_close",
        "expiry_mark_is_official_settlement",
        "entry_no_fill_included",
        "gross_zero_imputation",
        "entry_policy_generation_id",
        "entry_route",
        "boundary_quantile",
        "normalization_notional_twd",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
    }
    missing = sorted(required - set(selected_paths_with_entry_prices.columns))
    if missing:
        raise ValueError(f"daily inventory source missing columns: {missing}")
    source = selected_paths_with_entry_prices
    if source.is_empty() or source["policy_path_id"].n_unique() != source.height:
        raise ValueError("daily inventory source paths must be unique and nonempty")
    dates = _session_calendar_from_paths(source) if session_dates is None else _validate_session_dates(
        session_dates
    )
    projection = source.select(
        "policy_path_id",
        "entry_policy_generation_id",
        "entry_route",
        "boundary_quantile",
        "normalization_notional_twd",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
    )
    frames: list[pl.DataFrame] = []
    core_audits: list[pl.DataFrame] = []
    late_origin_ids: set[str] = set()
    for item in source.select(
        "entry_raw_order_fact_id", "Date", "position_established_ns"
    ).iter_rows(named=True):
        origin = str(item["Date"])
        if int(item["position_established_ns"]) >= aggressive_1300_start_cursor(
            origin
        ).recv_time_ns:
            late_origin_ids.add(str(item["entry_raw_order_fact_id"]))
    for target_date in dates:
        selected = select_open_positions_at_1300(source, target_date=target_date)
        core_audits.append(selected.audit)
        frame = selected.inventory.filter(
            ~pl.col("entry_raw_order_fact_id").is_in(sorted(late_origin_ids))
        ).join(
            projection,
            left_on="source_selected_policy_path_id",
            right_on="policy_path_id",
            how="left",
            validate="m:1",
        )
        if frame.height:
            frames.append(frame)
    if not frames:
        raise ValueError("daily 13:00 inventory is empty")
    inventory = pl.concat(frames, how="vertical_relaxed").sort(
        ["Date", "ValueCode", "position_established_recv_time_ns", "position_id"]
    )
    invalid = inventory.filter(
        pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "normalization_notional_twd",
            ).is_null()
        )
        | ~pl.all_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "normalization_notional_twd",
            ).is_finite()
        )
        | pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "normalization_notional_twd",
            )
            <= 0
        )
        | (
            (
                pl.col("entry_spot_price")
                * pl.col("entry_contract_size_shares")
                - pl.col("normalization_notional_twd")
            ).abs()
            > 1e-6
        )
    )
    if invalid.height:
        raise ValueError("daily inventory exact entry price/notional is invalid")
    if inventory.select("Date", "position_id").unique().height != inventory.height:
        raise ValueError("daily inventory duplicated a physical position-day")
    late_daily_rows = sum(
        selected.inventory.filter(
            pl.col("entry_raw_order_fact_id").is_in(sorted(late_origin_ids))
        ).height
        for selected in (
            select_open_positions_at_1300(source, target_date=date) for date in dates
        )
    )
    audit = pl.from_dicts(
        [
            {
                "source_selected_paths": source.height,
                "target_dates": len(dates),
                "origin_day_entries_at_or_after_1300_excluded": len(late_origin_ids),
                "daily_inventory_rows_excluded_from_late_origin_entries": late_daily_rows,
                "daily_open_position_rows": inventory.height,
                "daily_open_product_days": inventory.select("Date", "ValueCode").unique().height,
                "carried_position_rows": inventory.filter(
                    pl.col("inventory_origin") == "carried_from_prior_session"
                ).height,
                "target_day_position_rows": inventory.filter(
                    pl.col("inventory_origin") == "target_day_entry_before_1300"
                ).height,
                "post_1300_new_entries_admitted": 0,
                "normal_terminal_at_or_before_1300_included": 0,
                "daily_inventory_source_complete": True,
            }
        ],
        schema={
            "source_selected_paths": pl.Int64,
            "target_dates": pl.Int64,
            "origin_day_entries_at_or_after_1300_excluded": pl.Int64,
            "daily_inventory_rows_excluded_from_late_origin_entries": pl.Int64,
            "daily_open_position_rows": pl.Int64,
            "daily_open_product_days": pl.Int64,
            "carried_position_rows": pl.Int64,
            "target_day_position_rows": pl.Int64,
            "post_1300_new_entries_admitted": pl.Int64,
            "normal_terminal_at_or_before_1300_included": pl.Int64,
            "daily_inventory_source_complete": pl.Boolean,
        },
        strict=True,
    )
    return inventory, audit


def _session_calendar_from_paths(paths: pl.DataFrame) -> tuple[str, ...]:
    columns = (
        "Date",
        "terminal_date",
        "last_observed_session_date",
    )
    missing = [name for name in columns if name not in paths.columns]
    if missing:
        raise ValueError(f"session calendar source missing columns: {missing}")
    dates: set[str] = set()
    for name in columns:
        dates.update(str(value) for value in paths[name].drop_nulls().to_list())
    return _validate_session_dates(sorted(dates))


def _validate_session_dates(session_dates: Sequence[str]) -> tuple[str, ...]:
    dates = tuple(str(value) for value in session_dates)
    if (
        not dates
        or dates != tuple(sorted(set(dates)))
        or any(len(value) != 8 or not value.isdigit() for value in dates)
    ):
        raise ValueError("session calendar must be unique sorted YYYYMMDD values")
    return dates


def _session_calendar_sha256(session_dates: Sequence[str]) -> str:
    return _sha256_text(
        json.dumps(list(_validate_session_dates(session_dates)), separators=(",", ":"))
    )


def build_position_template_overlay(
    daily_open_inventory: pl.DataFrame,
    market_templates: pl.DataFrame,
) -> pl.DataFrame:
    """Attach one independent market template to each physical position-day."""

    keys = ["Date", "ValueCode", "QuoteCode"]
    if market_templates.select(keys).unique().height != market_templates.height:
        raise ValueError("market templates must be unique by product-day")
    prepared_templates = market_templates.rename(
        {"joint_volume_allocated": "market_template_joint_volume_allocated"}
    )
    template_columns = [name for name in prepared_templates.columns if name not in keys]
    overlay = daily_open_inventory.join(
        prepared_templates,
        on=keys,
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("market_template_id").is_not_null().alias("market_template_available"),
        (
            pl.col("market_template_id").is_not_null()
            & (pl.col("nominal_branch_status") == "flat_same_day")
            & (pl.col("oco_position_projection_safe") == True).fill_null(False)  # noqa: E712
            & (pl.col("winner_hedge_status") == "executable").fill_null(False)
        ).alias("controller_nominal_close_eligible"),
        pl.lit(False).alias("joint_volume_allocated_for_positions"),
        pl.lit(False).alias("position_close_is_actual_execution_claim"),
        pl.lit(False).alias("portfolio_controller_formal_go"),
        pl.lit(True).alias("controller_fifo_fields_exposed"),
    )
    if overlay.height != daily_open_inventory.height:
        raise AssertionError("position-template overlay changed the inventory denominator")
    if overlay.filter(
        pl.col("joint_volume_allocated_for_positions")
        | pl.col("position_close_is_actual_execution_claim")
        | pl.col("portfolio_controller_formal_go")
    ).height:
        raise AssertionError("position overlay safety flags changed")
    # Keep deterministic source fields first; template fields follow unchanged.
    return overlay.select(
        *daily_open_inventory.columns,
        *template_columns,
        "market_template_available",
        "controller_nominal_close_eligible",
        "joint_volume_allocated_for_positions",
        "position_close_is_actual_execution_claim",
        "portfolio_controller_formal_go",
        "controller_fifo_fields_exposed",
    ).sort(["Date", "ValueCode", "position_established_recv_time_ns", "position_id"])


def build_controller_contract(
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> pl.DataFrame:
    """Publish the controller contract without inventing scenario outcomes."""

    config.validate()
    records = []
    for cap in config.portfolio_caps_twd:
        target = min(float(cap), float(config.eod_target_ceiling_twd))
        for capacity_assumption in _CAPACITY_ASSUMPTIONS:
            conservative = capacity_assumption == _CONSERVATIVE_CAPACITY
            records.append(
                {
                "scenario_id": _scenario_id(float(cap), target, capacity_assumption),
                "capacity_assumption": capacity_assumption,
                "primary_conservative_result": conservative,
                "intraday_notional_cap_twd": float(cap),
                "per_product_cap_fraction": config.per_product_cap_fraction,
                "per_product_cap_twd": float(cap) * config.per_product_cap_fraction,
                "eod_target_notional_twd": target,
                "target_measurement_time": "13:20 Asia/Taipei study cutoff",
                "study_cutoff_is_twse_close": False,
                "extra_ten_minutes_to_twse_close_evaluated": False,
                "new_entries_at_or_after_1300_allowed": False,
                "normal_exit_competes_by_actual_timestamp": True,
                "entry_vs_normal_equal_timestamp_order": "entry_before_normal_exit",
                "normal_wins_equal_aggressive_decision_timestamp": True,
                "portfolio_target_cancel_clock": (
                    "winner_exit_decision_time_ns_full_fill_plus_50ms"
                ),
                "cross_product_fills_within_50ms_cancel_race_modeled": False,
                "runtime_daily_liquidity_admission_gate_applied": False,
                "held_product_exit_only_gate_applied": True,
                "held_product_definition": "active_carry_from_prior_date_at_start_of_session",
                "eligible_close_ordering": config.controller_ordering,
                "stop_rule": "cancel_remaining_when_active_one_way_entry_notional_le_target",
                "indivisible_position_target_undershoot_allowed": True,
                "same_timestamp_fifo_tie_fields_available": True,
                "winner_event_sequence_available": False,
                "accepted_active_inventory_source_bound": True,
                "one_market_template_capacity_per_product_day": conservative,
                "joint_volume_overcount_upper_bound": not conservative,
                "cross_scenario_template_capacity_reuse_counterfactual": True,
                "joint_volume_allocated": False,
                "actual_target_attainment_metrics_published": True,
                "all_close_pnl_is_stress_bound_only": True,
                "controller_formal_go": False,
                }
            )
    return pl.from_dicts(
        records,
        schema={
            "scenario_id": pl.String,
            "capacity_assumption": pl.String,
            "primary_conservative_result": pl.Boolean,
            "intraday_notional_cap_twd": pl.Float64,
            "per_product_cap_fraction": pl.Float64,
            "per_product_cap_twd": pl.Float64,
            "eod_target_notional_twd": pl.Float64,
            "target_measurement_time": pl.String,
            "study_cutoff_is_twse_close": pl.Boolean,
            "extra_ten_minutes_to_twse_close_evaluated": pl.Boolean,
            "new_entries_at_or_after_1300_allowed": pl.Boolean,
            "normal_exit_competes_by_actual_timestamp": pl.Boolean,
            "entry_vs_normal_equal_timestamp_order": pl.String,
            "normal_wins_equal_aggressive_decision_timestamp": pl.Boolean,
            "portfolio_target_cancel_clock": pl.String,
            "cross_product_fills_within_50ms_cancel_race_modeled": pl.Boolean,
            "runtime_daily_liquidity_admission_gate_applied": pl.Boolean,
            "held_product_exit_only_gate_applied": pl.Boolean,
            "held_product_definition": pl.String,
            "eligible_close_ordering": pl.String,
            "stop_rule": pl.String,
            "indivisible_position_target_undershoot_allowed": pl.Boolean,
            "same_timestamp_fifo_tie_fields_available": pl.Boolean,
            "winner_event_sequence_available": pl.Boolean,
            "accepted_active_inventory_source_bound": pl.Boolean,
            "one_market_template_capacity_per_product_day": pl.Boolean,
            "joint_volume_overcount_upper_bound": pl.Boolean,
            "cross_scenario_template_capacity_reuse_counterfactual": pl.Boolean,
            "joint_volume_allocated": pl.Boolean,
            "actual_target_attainment_metrics_published": pl.Boolean,
            "all_close_pnl_is_stress_bound_only": pl.Boolean,
            "controller_formal_go": pl.Boolean,
        },
        strict=True,
    ).sort("intraday_notional_cap_twd", "primary_conservative_result", descending=[False, True])


_CONSERVATIVE_CAPACITY = "conservative_one_observed_template_fill_per_product_day"
_OPTIMISTIC_CAPACITY = "optimistic_template_reuse_per_position_joint_volume_overcount"
_CAPACITY_ASSUMPTIONS = (_CONSERVATIVE_CAPACITY, _OPTIMISTIC_CAPACITY)


def _scenario_id(cap: float, target: float, capacity_assumption: str) -> str:
    token = "unique" if capacity_assumption == _CONSERVATIVE_CAPACITY else "reuse_upper"
    return (
        f"hard_cap_{int(cap)}_eod_target_{int(target)}_{token}_"
        "aggressive_1300_diagnostic_v2_post13_timeline_fix"
    )


def _cycle_transaction_cost(
    *,
    entry_date: str,
    terminal_date: str,
    entry_spot_price: float,
    exit_spot_price: float,
    entry_future_price: float,
    exit_future_price: float,
    shares: int,
    profile: TransactionCostProfile = TransactionCostProfile(),
) -> float:
    profile.validate()
    commission_rate = profile.spot_commission_bp_per_side / 10_000.0
    spot_tax_bp = profile.spot_sell_tax_bp * (
        profile.same_day_spot_sell_tax_multiplier
        if entry_date == terminal_date
        else 1.0
    )
    futures_tax_rate = profile.futures_tax_bp_per_side / 10_000.0
    return (
        entry_spot_price * shares * commission_rate
        + exit_spot_price * shares * commission_rate
        + exit_spot_price * shares * spot_tax_bp / 10_000.0
        + entry_future_price * shares * futures_tax_rate
        + exit_future_price * shares * futures_tax_rate
        + profile.futures_round_trip_commission_twd
    )


def build_hard_cap_controller_results(
    selected_paths_with_entry_prices: pl.DataFrame,
    market_templates: pl.DataFrame,
    *,
    session_dates: Sequence[str] | None = None,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Replay hard-cap admission and allocate each observed market fill once.

    Each hard-cap scenario is mutually exclusive.  Before 13:00, positions are
    admitted in immutable entry-time order while normal terminals release the
    cap.  At 13:00 the source-bound dynamic B1/A1 template is available at most
    once per product-day, so one observed winner can close at most one accepted
    physical position.  Product-day winners are consumed by decision timestamp
    and deterministic position FIFO until active one-way entry notional is at
    or below ``min(cap, eod_target_ceiling)``.  All unallocated positions keep
    their selected normal continuation/expiry.

    This is an actual deterministic application of the published template
    capacity, but remains a diagnostic rather than formal EV: the cache's raw
    upstream fingerprint is stat/mtime based and same-nanosecond winner event
    sequence is unavailable.
    """

    config.validate()
    validate_market_templates(market_templates)
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "boundary_quantile",
        "entry_raw_order_fact_id",
        "entry_policy_generation_id",
        "policy_path_id",
        "position_established_ns",
        "normalization_notional_twd",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        "terminal_cashflow_priced",
        "terminal_date",
        "exit_decision_time_ns",
        "exit_spot_price",
        "exit_future_price",
        "gross_cycle_pnl_twd",
        "gross_cycle_bp",
        "outcome_type",
        "outcome_status",
        "last_observed_session_date",
        "outstanding_interval_end_exclusive",
        "supplemental_terminal_resolution",
        "model_imputed_full_carry_on_unknown",
        "double_exit_bias_possible",
        "expiry_uses_last_valid_session_mark",
        "expiry_uses_last_observed_session_mark",
        "expiry_mark_uses_trade_fallback",
        "spot_expiry_mark_is_executable_bbo",
        "future_expiry_mark_is_executable_bbo",
        "expiry_mark_is_official_close",
        "expiry_mark_is_official_settlement",
    }
    missing = sorted(required - set(selected_paths_with_entry_prices.columns))
    if missing:
        raise ValueError(f"hard-cap controller source missing columns: {missing}")
    paths = selected_paths_with_entry_prices.sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )
    if (
        paths.is_empty()
        or paths["entry_raw_order_fact_id"].n_unique() != paths.height
        or paths["policy_path_id"].n_unique() != paths.height
    ):
        raise ValueError("hard-cap source must contain unique physical paths")
    invalid = paths.filter(
        pl.col("terminal_cashflow_priced").is_null()
        |
        pl.any_horizontal(
            pl.col(
                "Date",
                "ValueCode",
                "QuoteCode",
                "entry_raw_order_fact_id",
                "entry_policy_generation_id",
                "policy_path_id",
                "entry_route",
                "outcome_type",
                "outcome_status",
                "last_observed_session_date",
                "outstanding_interval_end_exclusive",
                "supplemental_terminal_resolution",
            ).is_null()
        )
        | pl.any_horizontal(
            pl.col(
                "Date",
                "ValueCode",
                "QuoteCode",
                "entry_raw_order_fact_id",
                "entry_policy_generation_id",
                "policy_path_id",
                "entry_route",
                "outcome_type",
                "outcome_status",
                "last_observed_session_date",
                "outstanding_interval_end_exclusive",
                "supplemental_terminal_resolution",
            ).str.len_chars()
            == 0
        )
        | pl.any_horizontal(
            pl.col(
                "model_imputed_full_carry_on_unknown",
                "double_exit_bias_possible",
                "expiry_uses_last_valid_session_mark",
                "expiry_uses_last_observed_session_mark",
                "expiry_mark_uses_trade_fallback",
            ).is_null()
        )
        | (
            pl.col("supplemental_terminal_resolution").str.starts_with("expiry_")
            != pl.all_horizontal(
                pl.col(
                    "spot_expiry_mark_is_executable_bbo",
                    "future_expiry_mark_is_executable_bbo",
                    "expiry_mark_is_official_close",
                    "expiry_mark_is_official_settlement",
                ).is_not_null()
            )
        ).fill_null(True)
        | (
            ~pl.col("supplemental_terminal_resolution").str.starts_with("expiry_")
            & pl.any_horizontal(
                pl.col(
                    "spot_expiry_mark_is_executable_bbo",
                    "future_expiry_mark_is_executable_bbo",
                    "expiry_mark_is_official_close",
                    "expiry_mark_is_official_settlement",
                ).is_not_null()
            )
        )
        | pl.any_horizontal(
            pl.col(
                "normalization_notional_twd",
                "entry_spot_price",
                "entry_future_price",
            ).is_null()
        )
        | ~pl.all_horizontal(
            pl.col(
                "normalization_notional_twd",
                "entry_spot_price",
                "entry_future_price",
            ).is_finite()
        )
        | pl.any_horizontal(
            pl.col(
                "normalization_notional_twd",
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
            )
            <= 0
        )
        | (
            (
                pl.col("entry_spot_price") * pl.col("entry_contract_size_shares")
                - pl.col("normalization_notional_twd")
            ).abs()
            > 1e-6
        )
        | (
            pl.col("terminal_cashflow_priced")
            & pl.any_horizontal(
                pl.col(
                    "terminal_date",
                    "exit_decision_time_ns",
                    "exit_spot_price",
                    "exit_future_price",
                    "gross_cycle_pnl_twd",
                    "gross_cycle_bp",
                ).is_null()
            )
        )
        | (
            pl.col("terminal_cashflow_priced")
            & (
                ~pl.all_horizontal(
                    pl.col(
                        "exit_spot_price",
                        "exit_future_price",
                        "gross_cycle_pnl_twd",
                        "gross_cycle_bp",
                    ).is_finite()
                )
                | (pl.col("exit_spot_price") <= 0)
                | (pl.col("exit_future_price") <= 0)
                | (
                    (
                        pl.col("entry_contract_size_shares")
                        * (
                            pl.col("exit_spot_price")
                            - pl.col("entry_spot_price")
                            + pl.col("entry_future_price")
                            - pl.col("exit_future_price")
                        )
                        - pl.col("gross_cycle_pnl_twd")
                    ).abs()
                    > 1e-6
                )
                | (
                    (
                        pl.col("gross_cycle_pnl_twd")
                        / pl.col("normalization_notional_twd")
                        * 10_000.0
                        - pl.col("gross_cycle_bp")
                    ).abs()
                    > 1e-6
                )
            )
        )
    )
    if invalid.height:
        raise ValueError(f"hard-cap controller source invariant failed: {invalid.height}")
    for row in paths.filter(pl.col("terminal_cashflow_priced")).select(
        "Date",
        "position_established_ns",
        "terminal_date",
        "exit_decision_time_ns",
    ).iter_rows(named=True):
        if (str(row["terminal_date"]), int(row["exit_decision_time_ns"])) < (
            str(row["Date"]),
            int(row["position_established_ns"]),
        ):
            raise ValueError("hard-cap controller terminal precedes entry")

    template_by_key = {
        (str(row["Date"]), str(row["ValueCode"]), str(row["QuoteCode"])): row
        for row in market_templates.iter_rows(named=True)
    }
    dates = (
        _session_calendar_from_paths(paths)
        if session_dates is None
        else _validate_session_dates(session_dates)
    )
    if any(str(row["Date"]) not in dates for row in paths.select("Date").iter_rows(named=True)):
        raise ValueError("hard-cap source origin is outside the session calendar")
    full_daily_inventory, _ = build_daily_open_inventory(
        paths, session_dates=dates
    )
    template_keys = market_templates.select("Date", "ValueCode", "QuoteCode")
    full_missing = (
        full_daily_inventory.group_by(["Date", "ValueCode", "QuoteCode"])
        .agg(pl.len().alias("open_positions"))
        .join(template_keys, on=["Date", "ValueCode", "QuoteCode"], how="anti")
    )
    full_missing_product_days = full_missing.height
    full_missing_position_rows = int(full_missing["open_positions"].sum() or 0)
    source_records = [dict(row) for row in paths.iter_rows(named=True)]
    position_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    scenario_specs = [
        (cap_value, capacity_assumption)
        for cap_value in config.portfolio_caps_twd
        for capacity_assumption in _CAPACITY_ASSUMPTIONS
    ]
    for cap_value, capacity_assumption in scenario_specs:
        cap = float(cap_value)
        target = min(cap, float(config.eod_target_ceiling_twd))
        conservative_capacity = capacity_assumption == _CONSERVATIVE_CAPACITY
        scenario_id = _scenario_id(cap, target, capacity_assumption)
        states: dict[str, dict[str, object]] = {}
        active: set[str] = set()
        aggressive_closes: dict[str, dict[str, object]] = {}

        for sequence, source in enumerate(source_records, start=1):
            position_id = str(source["entry_raw_order_fact_id"])
            states[position_id] = {
                "source": source,
                "candidate_sequence": sequence,
                "accepted": False,
                "admission_status": None,
                "portfolio_cap_blocked": False,
                "per_product_cap_blocked": False,
                "held_product_exit_only_blocked": False,
                "active_before_admission_twd": None,
                "active_same_product_before_admission_twd": None,
                "active_after_admission_twd": None,
                "active_same_product_after_admission_twd": None,
                "missing_product_day_exposures": 0,
                "nonflat_template_product_day_exposures": 0,
                "template_capacity_not_selected_exposures": 0,
            }

        entries_by_date: dict[str, list[dict[str, object]]] = {date: [] for date in dates}
        for source in source_records:
            entries_by_date[str(source["Date"])].append(source)

        def active_notional() -> float:
            return sum(
                float(states[position_id]["source"]["normalization_notional_twd"])
                for position_id in active
            )

        def active_product_notional(value_code: str) -> float:
            return sum(
                float(states[position_id]["source"]["normalization_notional_twd"])
                for position_id in active
                if str(states[position_id]["source"]["ValueCode"]) == value_code
            )

        def release_normal(
            date: str,
            recv_time_ns: int | None,
            *,
            inclusive: bool,
            include_expiry: bool,
        ) -> tuple[int, float]:
            released: list[str] = []
            for position_id in sorted(active):
                source = states[position_id]["source"]
                priced = bool(source["terminal_cashflow_priced"])
                due = False
                if priced:
                    terminal_date = str(source["terminal_date"])
                    terminal_ns = int(source["exit_decision_time_ns"])
                    if terminal_date < date:
                        due = True
                    elif terminal_date == date and recv_time_ns is not None:
                        due = terminal_ns <= recv_time_ns if inclusive else terminal_ns < recv_time_ns
                elif include_expiry:
                    # An unpriced observation boundary is not a cashflow and
                    # therefore cannot release cap.  Formal production input is
                    # the 3672/3672 priced supplemental v2 overlay; the public
                    # helper remains conservative for any unpriced fixture.
                    due = False
                if due:
                    released.append(position_id)
            notional = sum(
                float(states[position_id]["source"]["normalization_notional_twd"])
                for position_id in released
            )
            active.difference_update(released)
            return len(released), notional

        for date in dates:
            start_ns = aggressive_1300_start_cursor(date).recv_time_ns
            cutoff_ns = session_cutoff_cursor(date).recv_time_ns
            released_prior_count, released_prior_notional = release_normal(
                date, None, inclusive=True, include_expiry=True
            )
            held_products_exit_only = {
                str(states[position_id]["source"]["ValueCode"]) for position_id in active
            }
            accepted_today = forbidden_today = 0
            accepted_entry_one_way_turnover_today = 0.0
            accepted_entry_two_leg_turnover_today = 0.0
            portfolio_only_blocked_today = product_only_blocked_today = 0
            both_blocked_today = 0
            held_product_blocked_today = 0
            normal_releases_before_start_count = 0
            normal_release_notional_before_start = 0.0
            pre_start_entries = [
                source
                for source in entries_by_date[date]
                if int(source["position_established_ns"]) < start_ns
            ]
            forbidden_post_start_entries = [
                source
                for source in entries_by_date[date]
                if int(source["position_established_ns"]) >= start_ns
            ]
            for source in pre_start_entries:
                position_id = str(source["entry_raw_order_fact_id"])
                entry_ns = int(source["position_established_ns"])
                released_count, released_notional = release_normal(
                    date, entry_ns, inclusive=False, include_expiry=False
                )
                normal_releases_before_start_count += released_count
                normal_release_notional_before_start += released_notional
                before = active_notional()
                notional = float(source["normalization_notional_twd"])
                value_code = str(source["ValueCode"])
                before_product = active_product_notional(value_code)
                portfolio_blocked = before + notional > cap + 1e-9
                product_blocked = (
                    before_product + notional
                    > cap * float(config.per_product_cap_fraction) + 1e-9
                )
                state = states[position_id]
                state["active_before_admission_twd"] = before
                state["active_same_product_before_admission_twd"] = before_product
                state["portfolio_cap_blocked"] = portfolio_blocked
                state["per_product_cap_blocked"] = product_blocked
                if value_code in held_products_exit_only:
                    status = "held_product_exit_only_blocked"
                    state["portfolio_cap_blocked"] = False
                    state["per_product_cap_blocked"] = False
                    state["held_product_exit_only_blocked"] = True
                    held_product_blocked_today += 1
                elif portfolio_blocked and product_blocked:
                    status = "portfolio_and_product_hard_caps_blocked"
                    both_blocked_today += 1
                elif portfolio_blocked:
                    status = "portfolio_hard_cap_blocked"
                    portfolio_only_blocked_today += 1
                elif product_blocked:
                    status = "per_product_hard_cap_blocked"
                    product_only_blocked_today += 1
                else:
                    status = "accepted_before_1300"
                    state["accepted"] = True
                    active.add(position_id)
                    accepted_today += 1
                    shares = int(source["entry_contract_size_shares"])
                    accepted_entry_one_way_turnover_today += notional
                    accepted_entry_two_leg_turnover_today += shares * (
                        float(source["entry_spot_price"])
                        + float(source["entry_future_price"])
                    )
                state["admission_status"] = status
                state["active_after_admission_twd"] = active_notional()
                state["active_same_product_after_admission_twd"] = active_product_notional(
                    value_code
                )

            released_to_start_count, released_to_start_notional = release_normal(
                date, start_ns, inclusive=True, include_expiry=False
            )
            normal_releases_before_start_count += released_to_start_count
            normal_release_notional_before_start += released_to_start_notional
            active_at_start = sorted(active)
            active_start_notional = active_notional()
            # A post-13:00 entry is not an event in the admitted strategy.  In
            # particular it must not advance the normal-exit clock past the
            # 13:00 inventory snapshot before normal/aggressive competition.
            # Preserve a useful audit snapshot on each forbidden candidate
            # while leaving the active set and all cap flags unchanged.
            for source in forbidden_post_start_entries:
                position_id = str(source["entry_raw_order_fact_id"])
                value_code = str(source["ValueCode"])
                state = states[position_id]
                state["admission_status"] = "forbidden_new_entry_at_or_after_1300"
                state["portfolio_cap_blocked"] = False
                state["per_product_cap_blocked"] = False
                state["held_product_exit_only_blocked"] = False
                state["active_before_admission_twd"] = active_start_notional
                state["active_same_product_before_admission_twd"] = (
                    active_product_notional(value_code)
                )
                state["active_after_admission_twd"] = active_start_notional
                state["active_same_product_after_admission_twd"] = (
                    active_product_notional(value_code)
                )
                forbidden_today += 1
            max_active_product_start_notional = max(
                (active_product_notional(str(states[item]["source"]["ValueCode"])) for item in active),
                default=0.0,
            )
            grouped: dict[tuple[str, str, str], list[str]] = {}
            for position_id in active_at_start:
                source = states[position_id]["source"]
                key = (date, str(source["ValueCode"]), str(source["QuoteCode"]))
                grouped.setdefault(key, []).append(position_id)

            missing_keys = 0
            missing_positions = 0
            missing_notional = 0.0
            nonflat_keys = 0
            nonflat_positions = 0
            normal_equal_timestamp_positions = 0
            candidate_rows: list[tuple[int, int, str, str, dict[str, object]]] = []
            for key, position_ids in sorted(grouped.items()):
                template = template_by_key.get(key)
                if template is None:
                    missing_keys += 1
                    missing_positions += len(position_ids)
                    for position_id in position_ids:
                        state = states[position_id]
                        state["missing_product_day_exposures"] = int(
                            state["missing_product_day_exposures"]
                        ) + 1
                        missing_notional += float(
                            state["source"]["normalization_notional_twd"]
                        )
                    continue
                if (
                    template["nominal_branch_status"] != "flat_same_day"
                    or template["winner_hedge_status"] != "executable"
                    or template["oco_position_projection_safe"] is not True
                ):
                    nonflat_keys += 1
                    nonflat_positions += len(position_ids)
                    for position_id in position_ids:
                        state = states[position_id]
                        state["nonflat_template_product_day_exposures"] = int(
                            state["nonflat_template_product_day_exposures"]
                        ) + 1
                    continue
                decision = int(template["exit_decision_time_ns"])
                still_open = []
                for position_id in position_ids:
                    source = states[position_id]["source"]
                    if bool(source["terminal_cashflow_priced"]) and str(
                        source["terminal_date"]
                    ) == date and int(source["exit_decision_time_ns"]) <= decision:
                        if int(source["exit_decision_time_ns"]) == decision:
                            normal_equal_timestamp_positions += 1
                        continue
                    still_open.append(position_id)
                if not still_open:
                    continue
                still_open.sort(
                    key=lambda position_id: (
                        int(states[position_id]["source"]["position_established_ns"]),
                        str(states[position_id]["source"]["ValueCode"]),
                        str(states[position_id]["source"]["policy_path_id"]),
                    )
                )
                admitted_candidates = still_open[:1] if conservative_capacity else still_open
                if conservative_capacity:
                    for position_id in still_open[1:]:
                        state = states[position_id]
                        state["template_capacity_not_selected_exposures"] = int(
                            state["template_capacity_not_selected_exposures"]
                        ) + 1
                for selected in admitted_candidates:
                    source = states[selected]["source"]
                    candidate_rows.append(
                        (
                            decision,
                            int(source["position_established_ns"]),
                            str(source["ValueCode"]),
                            str(source["policy_path_id"]),
                            template,
                        )
                    )

            allocated_count = 0
            allocated_notional = 0.0
            target_cancelled_templates = 0
            normal_preempted_templates = 0
            normal_releases_during_controller = 0
            normal_release_notional_during_controller = 0.0
            for decision, _, _, policy_path_id, template in sorted(candidate_rows):
                released_count, released_notional = release_normal(
                    date, decision, inclusive=True, include_expiry=False
                )
                normal_releases_during_controller += released_count
                normal_release_notional_during_controller += released_notional
                if active_notional() <= target + 1e-9:
                    target_cancelled_templates += 1
                    continue
                matching = [
                    position_id
                    for position_id in active
                    if str(states[position_id]["source"]["policy_path_id"])
                    == policy_path_id
                ]
                if len(matching) != 1:
                    normal_preempted_templates += 1
                    continue
                position_id = matching[0]
                source = states[position_id]["source"]
                notional = float(source["normalization_notional_twd"])
                shares = int(source["entry_contract_size_shares"])
                exit_spot = float(template["exit_spot_price"])
                exit_future = float(template["exit_future_price"])
                gross = shares * (
                    exit_spot
                    - float(source["entry_spot_price"])
                    + float(source["entry_future_price"])
                    - exit_future
                )
                aggressive_closes[position_id] = {
                    "market_template_id": str(template["market_template_id"]),
                    "template_source": str(template["template_source"]),
                    "terminal_date": date,
                    "decision_time_ns": decision,
                    "winner_route": str(template["winner_route"]),
                    "exit_spot_price": exit_spot,
                    "exit_future_price": exit_future,
                    "gross_cycle_pnl_twd": gross,
                    "gross_cycle_bp": gross / notional * 10_000.0,
                }
                active.remove(position_id)
                allocated_count += 1
                allocated_notional += notional

            active_after_controller_count = len(active)
            active_after_controller_notional = active_notional()
            max_active_product_after_controller_notional = max(
                (active_product_notional(str(states[item]["source"]["ValueCode"])) for item in active),
                default=0.0,
            )
            released_to_eod_count, released_to_eod_notional = release_normal(
                date, cutoff_ns, inclusive=True, include_expiry=False
            )
            eod_active_count = len(active)
            eod_notional = active_notional()
            max_eod_active_product_notional = max(
                (active_product_notional(str(states[item]["source"]["ValueCode"])) for item in active),
                default=0.0,
            )
            post_expiry_close_ids = [
                position_id
                for position_id in sorted(active)
                if bool(states[position_id]["source"]["terminal_cashflow_priced"])
                and str(states[position_id]["source"]["terminal_date"]) == date
                and int(states[position_id]["source"]["exit_decision_time_ns"])
                > cutoff_ns
                and str(
                    states[position_id]["source"]["supplemental_terminal_resolution"]
                )
                == "expiry_same_day_two_leg_close_price"
            ]
            post_expiry_close_notional = sum(
                float(states[position_id]["source"]["normalization_notional_twd"])
                for position_id in post_expiry_close_ids
            )
            active.difference_update(post_expiry_close_ids)
            post_expiry_active_notional = active_notional()
            max_post_expiry_active_product_notional = max(
                (
                    active_product_notional(
                        str(states[item]["source"]["ValueCode"])
                    )
                    for item in active
                ),
                default=0.0,
            )
            daily_rows.append(
                {
                    "scenario_id": scenario_id,
                    "capacity_assumption": capacity_assumption,
                    "primary_conservative_result": conservative_capacity,
                    "joint_volume_overcount_upper_bound": not conservative_capacity,
                    "portfolio_cap_twd": cap,
                    "per_product_cap_fraction": config.per_product_cap_fraction,
                    "per_product_cap_twd": cap * config.per_product_cap_fraction,
                    "eod_target_notional_twd": target,
                    "target_measurement_time": "13:20 Asia/Taipei study cutoff",
                    "study_cutoff_is_twse_close": False,
                    "extra_ten_minutes_to_twse_close_evaluated": False,
                    "Date": date,
                    "accepted_entries_before_1300": accepted_today,
                    "accepted_entry_one_way_turnover_twd": (
                        accepted_entry_one_way_turnover_today
                    ),
                    "accepted_entry_two_leg_turnover_twd": (
                        accepted_entry_two_leg_turnover_today
                    ),
                    "portfolio_only_blocked_entries_before_1300": portfolio_only_blocked_today,
                    "product_only_blocked_entries_before_1300": product_only_blocked_today,
                    "both_caps_blocked_entries_before_1300": both_blocked_today,
                    "held_product_exit_only_blocked_entries": held_product_blocked_today,
                    "held_product_exit_only_products": len(held_products_exit_only),
                    "forbidden_new_entries_at_or_after_1300": forbidden_today,
                    "normal_or_expiry_releases_before_day": released_prior_count,
                    "normal_or_expiry_release_notional_before_day_twd": released_prior_notional,
                    "normal_releases_before_1300": normal_releases_before_start_count,
                    "normal_release_notional_before_1300_twd": (
                        normal_release_notional_before_start
                    ),
                    "active_positions_at_1300": len(active_at_start),
                    "active_notional_at_1300_twd": active_start_notional,
                    "max_active_same_product_notional_at_1300_twd": max_active_product_start_notional,
                    "missing_cache_product_days_at_1300": missing_keys,
                    "missing_cache_position_rows_at_1300": missing_positions,
                    "missing_cache_notional_at_1300_twd": missing_notional,
                    "nonflat_template_product_days_at_1300": nonflat_keys,
                    "nonflat_template_position_rows_at_1300": nonflat_positions,
                    "unique_flat_template_candidates": len(candidate_rows),
                    "aggressive_closes_allocated": allocated_count,
                    "aggressive_close_notional_twd": allocated_notional,
                    "target_reached_cancelled_templates": target_cancelled_templates,
                    "normal_terminal_preempted_templates": normal_preempted_templates,
                    "normal_wins_equal_aggressive_timestamp_positions": (
                        normal_equal_timestamp_positions
                    ),
                    "normal_releases_during_aggressive_controller": normal_releases_during_controller,
                    "normal_release_notional_during_aggressive_controller_twd": normal_release_notional_during_controller,
                    "active_positions_after_aggressive_controller": active_after_controller_count,
                    "active_notional_after_aggressive_controller_twd": active_after_controller_notional,
                    "max_active_same_product_notional_after_controller_twd": max_active_product_after_controller_notional,
                    "normal_releases_after_controller_to_eod": released_to_eod_count,
                    "normal_release_notional_after_controller_to_eod_twd": released_to_eod_notional,
                    "eod_active_positions": eod_active_count,
                    "eod_active_notional_twd": eod_notional,
                    "max_eod_active_same_product_notional_twd": max_eod_active_product_notional,
                    "controller_target_attained": active_after_controller_notional
                    <= target + 1e-9,
                    "eod_target_attained": eod_notional <= target + 1e-9,
                    "eod_target_excess_notional_twd": max(0.0, eod_notional - target),
                    "post_expiry_close_positions": len(post_expiry_close_ids),
                    "post_expiry_close_notional_twd": post_expiry_close_notional,
                    "post_expiry_close_active_positions": len(active),
                    "post_expiry_close_active_notional_twd": post_expiry_active_notional,
                    "max_post_expiry_close_active_same_product_notional_twd": (
                        max_post_expiry_active_product_notional
                    ),
                    "post_expiry_close_target_attained": (
                        post_expiry_active_notional <= target + 1e-9
                    ),
                    "post_expiry_close_target_excess_notional_twd": max(
                        0.0, post_expiry_active_notional - target
                    ),
                    "post_expiry_close_is_full_1330_market_replay": False,
                    "realized_completed_positions": 0,
                    "realized_gross_pnl_twd": 0.0,
                    "realized_transaction_cost_twd": 0.0,
                    "realized_net_pnl_twd": 0.0,
                    "realized_net_loss_positions": 0,
                    "realized_exit_one_way_spot_turnover_twd": 0.0,
                    "realized_exit_paired_two_leg_turnover_twd": 0.0,
                    "one_capacity_per_market_template_enforced": conservative_capacity,
                    "missing_cache_positions_counted_as_closed": False,
                    "joint_volume_allocated": False,
                    "formal_ev_ready": False,
                }
            )

        scenario_positions: list[dict[str, object]] = []
        for position_id, state in sorted(
            states.items(), key=lambda item: int(item[1]["candidate_sequence"])
        ):
            source = state["source"]
            accepted = bool(state["accepted"])
            aggressive = aggressive_closes.get(position_id)
            if aggressive is not None:
                terminal_source = "aggressive_1300_template_capacity"
                terminal_priced = True
                terminal_date = aggressive["terminal_date"]
                terminal_ns = aggressive["decision_time_ns"]
                exit_spot = aggressive["exit_spot_price"]
                exit_future = aggressive["exit_future_price"]
                gross = aggressive["gross_cycle_pnl_twd"]
                gross_bp = aggressive["gross_cycle_bp"]
            elif accepted and bool(source["terminal_cashflow_priced"]):
                terminal_source = "selected_normal_continuation"
                terminal_priced = True
                terminal_date = str(source["terminal_date"])
                terminal_ns = int(source["exit_decision_time_ns"])
                exit_spot = float(source["exit_spot_price"])
                exit_future = float(source["exit_future_price"])
                gross = float(source["gross_cycle_pnl_twd"])
                gross_bp = float(source["gross_cycle_bp"])
            elif accepted:
                terminal_source = "selected_normal_continuation_unresolved_or_expiry"
                terminal_priced = False
                terminal_date = terminal_ns = exit_spot = exit_future = None
                gross = gross_bp = None
            else:
                terminal_source = "entry_not_admitted"
                terminal_priced = False
                terminal_date = terminal_ns = exit_spot = exit_future = None
                gross = gross_bp = None
            if terminal_priced:
                transaction_cost = _cycle_transaction_cost(
                    entry_date=str(source["Date"]),
                    terminal_date=str(terminal_date),
                    entry_spot_price=float(source["entry_spot_price"]),
                    exit_spot_price=float(exit_spot),
                    entry_future_price=float(source["entry_future_price"]),
                    exit_future_price=float(exit_future),
                    shares=int(source["entry_contract_size_shares"]),
                )
                net_pnl = float(gross) - transaction_cost
                net_bp = net_pnl / float(source["normalization_notional_twd"]) * 10_000.0
                net_profitable = net_pnl > 0
            else:
                transaction_cost = net_pnl = net_bp = None
                net_profitable = None
            scenario_positions.append(
                {
                    "scenario_id": scenario_id,
                    "capacity_assumption": capacity_assumption,
                    "primary_conservative_result": conservative_capacity,
                    "joint_volume_overcount_upper_bound": not conservative_capacity,
                    "portfolio_cap_twd": cap,
                    "per_product_cap_fraction": config.per_product_cap_fraction,
                    "per_product_cap_twd": cap * config.per_product_cap_fraction,
                    "eod_target_notional_twd": target,
                    "candidate_sequence": int(state["candidate_sequence"]),
                    "Date": str(source["Date"]),
                    "ValueCode": str(source["ValueCode"]),
                    "QuoteCode": str(source["QuoteCode"]),
                    "entry_route": str(source["entry_route"]),
                    "boundary_quantile": int(source["boundary_quantile"]),
                    "position_id": position_id,
                    "policy_path_id": str(source["policy_path_id"]),
                    "entry_policy_generation_id": str(source["entry_policy_generation_id"]),
                    "position_established_ns": int(source["position_established_ns"]),
                    "normalization_notional_twd": float(source["normalization_notional_twd"]),
                    "entry_spot_price": float(source["entry_spot_price"]),
                    "entry_future_price": float(source["entry_future_price"]),
                    "entry_contract_size_shares": int(source["entry_contract_size_shares"]),
                    "accepted": accepted,
                    "admission_status": str(state["admission_status"]),
                    "portfolio_cap_blocked": bool(state["portfolio_cap_blocked"]),
                    "per_product_cap_blocked": bool(state["per_product_cap_blocked"]),
                    "held_product_exit_only_blocked": bool(
                        state["held_product_exit_only_blocked"]
                    ),
                    "active_notional_before_admission_twd": float(
                        state["active_before_admission_twd"]
                    ),
                    "active_same_product_notional_before_admission_twd": float(
                        state["active_same_product_before_admission_twd"]
                    ),
                    "active_notional_after_admission_twd": float(
                        state["active_after_admission_twd"]
                    ),
                    "active_same_product_notional_after_admission_twd": float(
                        state["active_same_product_after_admission_twd"]
                    ),
                    "missing_cache_product_day_exposures": int(
                        state["missing_product_day_exposures"]
                    ),
                    "nonflat_template_product_day_exposures": int(
                        state["nonflat_template_product_day_exposures"]
                    ),
                    "template_capacity_not_selected_exposures": int(
                        state["template_capacity_not_selected_exposures"]
                    ),
                    "aggressive_close_allocated": aggressive is not None,
                    "aggressive_market_template_id": (
                        None if aggressive is None else aggressive["market_template_id"]
                    ),
                    "aggressive_template_source": (
                        None if aggressive is None else aggressive["template_source"]
                    ),
                    "aggressive_winner_route": (
                        None if aggressive is None else aggressive["winner_route"]
                    ),
                    "scenario_terminal_source": terminal_source,
                    "scenario_terminal_cashflow_priced": terminal_priced,
                    "scenario_terminal_date": terminal_date,
                    "scenario_exit_decision_time_ns": terminal_ns,
                    "scenario_exit_spot_price": exit_spot,
                    "scenario_exit_future_price": exit_future,
                    "scenario_gross_cycle_pnl_twd": gross,
                    "scenario_gross_cycle_bp": gross_bp,
                    "scenario_transaction_cost_twd": transaction_cost,
                    "scenario_net_cycle_pnl_twd": net_pnl,
                    "scenario_net_cycle_pnl_bp": net_bp,
                    "scenario_net_profitable": net_profitable,
                    "normal_terminal_cashflow_priced": bool(
                        source["terminal_cashflow_priced"]
                    ),
                    "normal_terminal_date": (
                        None if source["terminal_date"] is None else str(source["terminal_date"])
                    ),
                    "normal_exit_decision_time_ns": (
                        None
                        if source["exit_decision_time_ns"] is None
                        else int(source["exit_decision_time_ns"])
                    ),
                    "normal_outcome_type": str(source["outcome_type"]),
                    "normal_outcome_status": str(source["outcome_status"]),
                    "normal_outstanding_interval_end_exclusive": str(
                        source["outstanding_interval_end_exclusive"]
                    ),
                    "supplemental_terminal_resolution": str(
                        source["supplemental_terminal_resolution"]
                    ),
                    "model_imputed_full_carry_on_unknown": bool(
                        source["model_imputed_full_carry_on_unknown"]
                    ),
                    "double_exit_bias_possible": bool(source["double_exit_bias_possible"]),
                    "expiry_uses_last_valid_session_mark": bool(
                        source["expiry_uses_last_valid_session_mark"]
                    ),
                    "expiry_uses_last_observed_session_mark": bool(
                        source["expiry_uses_last_observed_session_mark"]
                    ),
                    "expiry_mark_uses_trade_fallback": bool(
                        source["expiry_mark_uses_trade_fallback"]
                    ),
                    "spot_expiry_mark_is_executable_bbo": _optional_bool(
                        source["spot_expiry_mark_is_executable_bbo"]
                    ),
                    "future_expiry_mark_is_executable_bbo": _optional_bool(
                        source["future_expiry_mark_is_executable_bbo"]
                    ),
                    "expiry_mark_is_official_close": _optional_bool(
                        source["expiry_mark_is_official_close"]
                    ),
                    "expiry_mark_is_official_settlement": _optional_bool(
                        source["expiry_mark_is_official_settlement"]
                    ),
                    "one_capacity_per_market_template_enforced": conservative_capacity,
                    "cross_scenario_capacity_reuse_counterfactual": True,
                    "missing_cache_position_counted_as_closed": False,
                    "joint_volume_allocated": False,
                    "transaction_cost_profile_id": TransactionCostProfile().profile_id,
                    "transaction_cost_point_identified": terminal_priced,
                    "transaction_cost_profile_complete_for_priced_cycles": terminal_priced,
                    "realized_pnl_only_no_mtm": True,
                    "formal_ev_ready": False,
                    "production_strategy_go": False,
                }
            )
        position_rows.extend(scenario_positions)

        accepted_positions = [row for row in scenario_positions if row["accepted"]]
        completed_positions = [
            row for row in accepted_positions if row["scenario_terminal_cashflow_priced"]
        ]
        aggressive_positions = [
            row for row in accepted_positions if row["aggressive_close_allocated"]
        ]
        scenario_daily = [row for row in daily_rows if row["scenario_id"] == scenario_id]
        realized_by_date: dict[str, list[dict[str, object]]] = {}
        for row in completed_positions:
            realized_by_date.setdefault(str(row["scenario_terminal_date"]), []).append(row)
        cumulative_net = 0.0
        running_peak = 0.0
        max_drawdown = 0.0
        negative_realized_days = 0
        for row in scenario_daily:
            realized = realized_by_date.get(str(row["Date"]), [])
            gross_day = sum(float(item["scenario_gross_cycle_pnl_twd"]) for item in realized)
            cost_day = sum(float(item["scenario_transaction_cost_twd"]) for item in realized)
            net_day = sum(float(item["scenario_net_cycle_pnl_twd"]) for item in realized)
            exit_one_way_day = sum(
                int(item["entry_contract_size_shares"])
                * float(item["scenario_exit_spot_price"])
                for item in realized
            )
            exit_two_leg_day = sum(
                int(item["entry_contract_size_shares"])
                * (
                    float(item["scenario_exit_spot_price"])
                    + float(item["scenario_exit_future_price"])
                )
                for item in realized
            )
            row["realized_completed_positions"] = len(realized)
            row["realized_gross_pnl_twd"] = gross_day
            row["realized_transaction_cost_twd"] = cost_day
            row["realized_net_pnl_twd"] = net_day
            row["realized_net_loss_positions"] = sum(
                float(item["scenario_net_cycle_pnl_twd"]) < 0 for item in realized
            )
            row["realized_exit_one_way_spot_turnover_twd"] = exit_one_way_day
            row["realized_exit_paired_two_leg_turnover_twd"] = exit_two_leg_day
            if net_day < 0:
                negative_realized_days += 1
            cumulative_net += net_day
            running_peak = max(running_peak, cumulative_net)
            max_drawdown = max(max_drawdown, running_peak - cumulative_net)
        completed_notional = sum(
            float(row["normalization_notional_twd"]) for row in completed_positions
        )
        completed_gross = sum(
            float(row["scenario_gross_cycle_pnl_twd"]) for row in completed_positions
        )
        completed_cost = sum(
            float(row["scenario_transaction_cost_twd"]) for row in completed_positions
        )
        completed_net = sum(
            float(row["scenario_net_cycle_pnl_twd"]) for row in completed_positions
        )
        accepted_entry_one_way_turnover = sum(
            float(row["normalization_notional_twd"]) for row in accepted_positions
        )
        accepted_entry_two_leg_turnover = sum(
            int(row["entry_contract_size_shares"])
            * (float(row["entry_spot_price"]) + float(row["entry_future_price"]))
            for row in accepted_positions
        )
        realized_exit_one_way_turnover = sum(
            int(row["entry_contract_size_shares"])
            * float(row["scenario_exit_spot_price"])
            for row in completed_positions
        )
        realized_exit_two_leg_turnover = sum(
            int(row["entry_contract_size_shares"])
            * (
                float(row["scenario_exit_spot_price"])
                + float(row["scenario_exit_future_price"])
            )
            for row in completed_positions
        )
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "capacity_assumption": capacity_assumption,
                "primary_conservative_result": conservative_capacity,
                "joint_volume_overcount_upper_bound": not conservative_capacity,
                "portfolio_cap_twd": cap,
                "per_product_cap_fraction": config.per_product_cap_fraction,
                "per_product_cap_twd": cap * config.per_product_cap_fraction,
                "eod_target_notional_twd": target,
                "target_measurement_time": "13:20 Asia/Taipei study cutoff",
                "study_cutoff_is_twse_close": False,
                "extra_ten_minutes_to_twse_close_evaluated": False,
                "runtime_daily_liquidity_admission_gate_applied": False,
                "candidate_entries": len(scenario_positions),
                "accepted_entries": len(accepted_positions),
                "portfolio_only_blocked_entries": sum(
                    row["admission_status"] == "portfolio_hard_cap_blocked"
                    for row in scenario_positions
                ),
                "product_only_blocked_entries": sum(
                    row["admission_status"] == "per_product_hard_cap_blocked"
                    for row in scenario_positions
                ),
                "both_caps_blocked_entries": sum(
                    row["admission_status"] == "portfolio_and_product_hard_caps_blocked"
                    for row in scenario_positions
                ),
                "held_product_exit_only_blocked_entries": sum(
                    row["admission_status"] == "held_product_exit_only_blocked"
                    for row in scenario_positions
                ),
                "forbidden_new_entries_at_or_after_1300": sum(
                    row["admission_status"]
                    == "forbidden_new_entry_at_or_after_1300"
                    for row in scenario_positions
                ),
                "aggressive_closed_positions": len(aggressive_positions),
                "aggressive_closed_notional_twd": sum(
                    float(row["normalization_notional_twd"])
                    for row in aggressive_positions
                ),
                "normal_continuation_completed_positions": sum(
                    row["scenario_terminal_source"] == "selected_normal_continuation"
                    for row in accepted_positions
                ),
                "normal_continuation_unresolved_or_expiry_positions": sum(
                    row["scenario_terminal_source"]
                    == "selected_normal_continuation_unresolved_or_expiry"
                    for row in accepted_positions
                ),
                "normal_wins_equal_aggressive_timestamp_positions": sum(
                    int(row["normal_wins_equal_aggressive_timestamp_positions"])
                    for row in scenario_daily
                ),
                "accepted_model_imputed_full_carry_positions": sum(
                    bool(row["model_imputed_full_carry_on_unknown"])
                    for row in accepted_positions
                ),
                "accepted_double_exit_bias_possible_positions": sum(
                    bool(row["double_exit_bias_possible"])
                    for row in accepted_positions
                ),
                "accepted_expiry_trade_fallback_positions": sum(
                    bool(row["expiry_mark_uses_trade_fallback"])
                    for row in accepted_positions
                ),
                "accepted_expiry_official_close_positions": sum(
                    row["expiry_mark_is_official_close"] is True
                    for row in accepted_positions
                ),
                "accepted_expiry_official_settlement_positions": sum(
                    row["expiry_mark_is_official_settlement"] is True
                    for row in accepted_positions
                ),
                "completed_scenario_positions": len(completed_positions),
                "completed_scenario_gross_pnl_twd": completed_gross,
                "completed_scenario_transaction_cost_twd": completed_cost,
                "completed_scenario_net_pnl_twd": completed_net,
                "completed_scenario_notional_twd": completed_notional,
                "accepted_entry_one_way_turnover_twd": (
                    accepted_entry_one_way_turnover
                ),
                "accepted_entry_paired_two_leg_turnover_twd": (
                    accepted_entry_two_leg_turnover
                ),
                "realized_exit_one_way_spot_turnover_twd": (
                    realized_exit_one_way_turnover
                ),
                "realized_exit_paired_two_leg_turnover_twd": (
                    realized_exit_two_leg_turnover
                ),
                "mean_daily_accepted_entry_one_way_turnover_twd": (
                    accepted_entry_one_way_turnover / len(scenario_daily)
                ),
                "mean_daily_accepted_entry_paired_two_leg_turnover_twd": (
                    accepted_entry_two_leg_turnover / len(scenario_daily)
                ),
                "mean_daily_realized_exit_one_way_spot_turnover_twd": (
                    realized_exit_one_way_turnover / len(scenario_daily)
                ),
                "mean_daily_realized_exit_paired_two_leg_turnover_twd": (
                    realized_exit_two_leg_turnover / len(scenario_daily)
                ),
                "exit_one_way_turnover_basis": "actual_spot_exit_price_times_shares",
                "completed_scenario_weighted_gross_bp": (
                    completed_gross / completed_notional
                    * 10_000.0
                    if completed_notional
                    else None
                ),
                "completed_scenario_weighted_cost_bp": (
                    completed_cost / completed_notional * 10_000.0
                    if completed_notional
                    else None
                ),
                "completed_scenario_weighted_net_bp": (
                    completed_net / completed_notional * 10_000.0
                    if completed_notional
                    else None
                ),
                "net_profitable_positions": sum(
                    float(row["scenario_net_cycle_pnl_twd"]) > 0
                    for row in completed_positions
                ),
                "net_loss_positions": sum(
                    float(row["scenario_net_cycle_pnl_twd"]) < 0
                    for row in completed_positions
                ),
                "net_zero_positions": sum(
                    float(row["scenario_net_cycle_pnl_twd"]) == 0
                    for row in completed_positions
                ),
                "negative_realized_days": negative_realized_days,
                "realized_net_pnl_max_drawdown_twd": max_drawdown,
                "full_universe_missing_product_days": full_missing_product_days,
                "full_universe_missing_position_day_rows": full_missing_position_rows,
                "accepted_missing_cache_position_day_exposures": sum(
                    int(row["missing_cache_position_rows_at_1300"])
                    for row in scenario_daily
                ),
                "accepted_missing_cache_notional_day_exposures_twd": sum(
                    float(row["missing_cache_notional_at_1300_twd"])
                    for row in scenario_daily
                ),
                "days_controller_target_attained": sum(
                    bool(row["controller_target_attained"]) for row in scenario_daily
                ),
                "days_eod_target_attained": sum(
                    bool(row["eod_target_attained"]) for row in scenario_daily
                ),
                "max_eod_target_excess_notional_twd": max(
                    (float(row["eod_target_excess_notional_twd"]) for row in scenario_daily),
                    default=0.0,
                ),
                "post_expiry_close_positions": sum(
                    int(row["post_expiry_close_positions"])
                    for row in scenario_daily
                ),
                "post_expiry_close_notional_twd": sum(
                    float(row["post_expiry_close_notional_twd"])
                    for row in scenario_daily
                ),
                "days_post_expiry_close_target_attained": sum(
                    bool(row["post_expiry_close_target_attained"])
                    for row in scenario_daily
                ),
                "max_post_expiry_close_target_excess_notional_twd": max(
                    (
                        float(row["post_expiry_close_target_excess_notional_twd"])
                        for row in scenario_daily
                    ),
                    default=0.0,
                ),
                "post_expiry_close_is_full_1330_market_replay": False,
                "market_template_capacity_reused_within_scenario": not conservative_capacity,
                "cross_scenario_results_mutually_exclusive": True,
                "upstream_raw_content_integrity_bound": False,
                "transaction_cost_profile_id": TransactionCostProfile().profile_id,
                "transaction_cost_profile_complete_for_priced_cycles": True,
                "realized_pnl_only_no_mtm": True,
                "formal_ev_ready": False,
                "production_strategy_go": False,
            }
        )

    positions = pl.from_dicts(
        position_rows, schema=_controller_position_schema(), strict=True
    ).sort(["portfolio_cap_twd", "candidate_sequence"])
    daily = pl.from_dicts(daily_rows, schema=_controller_daily_schema(), strict=True).sort(
        ["portfolio_cap_twd", "Date"]
    )
    summary = pl.from_dicts(
        summary_rows, schema=_controller_summary_schema(), strict=True
    ).sort("portfolio_cap_twd")
    _validate_controller_results(positions, daily, summary, config)
    return positions, daily, summary


def _controller_position_schema() -> dict[str, pl.DataType]:
    return {
        "scenario_id": pl.String,
        "capacity_assumption": pl.String,
        "primary_conservative_result": pl.Boolean,
        "joint_volume_overcount_upper_bound": pl.Boolean,
        "portfolio_cap_twd": pl.Float64,
        "per_product_cap_fraction": pl.Float64,
        "per_product_cap_twd": pl.Float64,
        "eod_target_notional_twd": pl.Float64,
        "candidate_sequence": pl.Int64,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "boundary_quantile": pl.Int64,
        "position_id": pl.String,
        "policy_path_id": pl.String,
        "entry_policy_generation_id": pl.String,
        "position_established_ns": pl.Int64,
        "normalization_notional_twd": pl.Float64,
        "entry_spot_price": pl.Float64,
        "entry_future_price": pl.Float64,
        "entry_contract_size_shares": pl.Int64,
        "accepted": pl.Boolean,
        "admission_status": pl.String,
        "portfolio_cap_blocked": pl.Boolean,
        "per_product_cap_blocked": pl.Boolean,
        "held_product_exit_only_blocked": pl.Boolean,
        "active_notional_before_admission_twd": pl.Float64,
        "active_same_product_notional_before_admission_twd": pl.Float64,
        "active_notional_after_admission_twd": pl.Float64,
        "active_same_product_notional_after_admission_twd": pl.Float64,
        "missing_cache_product_day_exposures": pl.Int64,
        "nonflat_template_product_day_exposures": pl.Int64,
        "template_capacity_not_selected_exposures": pl.Int64,
        "aggressive_close_allocated": pl.Boolean,
        "aggressive_market_template_id": pl.String,
        "aggressive_template_source": pl.String,
        "aggressive_winner_route": pl.String,
        "scenario_terminal_source": pl.String,
        "scenario_terminal_cashflow_priced": pl.Boolean,
        "scenario_terminal_date": pl.String,
        "scenario_exit_decision_time_ns": pl.Int64,
        "scenario_exit_spot_price": pl.Float64,
        "scenario_exit_future_price": pl.Float64,
        "scenario_gross_cycle_pnl_twd": pl.Float64,
        "scenario_gross_cycle_bp": pl.Float64,
        "scenario_transaction_cost_twd": pl.Float64,
        "scenario_net_cycle_pnl_twd": pl.Float64,
        "scenario_net_cycle_pnl_bp": pl.Float64,
        "scenario_net_profitable": pl.Boolean,
        "normal_terminal_cashflow_priced": pl.Boolean,
        "normal_terminal_date": pl.String,
        "normal_exit_decision_time_ns": pl.Int64,
        "normal_outcome_type": pl.String,
        "normal_outcome_status": pl.String,
        "normal_outstanding_interval_end_exclusive": pl.String,
        "supplemental_terminal_resolution": pl.String,
        "model_imputed_full_carry_on_unknown": pl.Boolean,
        "double_exit_bias_possible": pl.Boolean,
        "expiry_uses_last_valid_session_mark": pl.Boolean,
        "expiry_uses_last_observed_session_mark": pl.Boolean,
        "expiry_mark_uses_trade_fallback": pl.Boolean,
        "spot_expiry_mark_is_executable_bbo": pl.Boolean,
        "future_expiry_mark_is_executable_bbo": pl.Boolean,
        "expiry_mark_is_official_close": pl.Boolean,
        "expiry_mark_is_official_settlement": pl.Boolean,
        "one_capacity_per_market_template_enforced": pl.Boolean,
        "cross_scenario_capacity_reuse_counterfactual": pl.Boolean,
        "missing_cache_position_counted_as_closed": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
        "transaction_cost_profile_id": pl.String,
        "transaction_cost_point_identified": pl.Boolean,
        "transaction_cost_profile_complete_for_priced_cycles": pl.Boolean,
        "realized_pnl_only_no_mtm": pl.Boolean,
        "formal_ev_ready": pl.Boolean,
        "production_strategy_go": pl.Boolean,
    }


def _controller_daily_schema() -> dict[str, pl.DataType]:
    return {
        "scenario_id": pl.String,
        "capacity_assumption": pl.String,
        "primary_conservative_result": pl.Boolean,
        "joint_volume_overcount_upper_bound": pl.Boolean,
        "portfolio_cap_twd": pl.Float64,
        "per_product_cap_fraction": pl.Float64,
        "per_product_cap_twd": pl.Float64,
        "eod_target_notional_twd": pl.Float64,
        "target_measurement_time": pl.String,
        "study_cutoff_is_twse_close": pl.Boolean,
        "extra_ten_minutes_to_twse_close_evaluated": pl.Boolean,
        "Date": pl.String,
        "accepted_entries_before_1300": pl.Int64,
        "accepted_entry_one_way_turnover_twd": pl.Float64,
        "accepted_entry_two_leg_turnover_twd": pl.Float64,
        "portfolio_only_blocked_entries_before_1300": pl.Int64,
        "product_only_blocked_entries_before_1300": pl.Int64,
        "both_caps_blocked_entries_before_1300": pl.Int64,
        "held_product_exit_only_blocked_entries": pl.Int64,
        "held_product_exit_only_products": pl.Int64,
        "forbidden_new_entries_at_or_after_1300": pl.Int64,
        "normal_or_expiry_releases_before_day": pl.Int64,
        "normal_or_expiry_release_notional_before_day_twd": pl.Float64,
        "normal_releases_before_1300": pl.Int64,
        "normal_release_notional_before_1300_twd": pl.Float64,
        "active_positions_at_1300": pl.Int64,
        "active_notional_at_1300_twd": pl.Float64,
        "max_active_same_product_notional_at_1300_twd": pl.Float64,
        "missing_cache_product_days_at_1300": pl.Int64,
        "missing_cache_position_rows_at_1300": pl.Int64,
        "missing_cache_notional_at_1300_twd": pl.Float64,
        "nonflat_template_product_days_at_1300": pl.Int64,
        "nonflat_template_position_rows_at_1300": pl.Int64,
        "unique_flat_template_candidates": pl.Int64,
        "aggressive_closes_allocated": pl.Int64,
        "aggressive_close_notional_twd": pl.Float64,
        "target_reached_cancelled_templates": pl.Int64,
        "normal_terminal_preempted_templates": pl.Int64,
        "normal_wins_equal_aggressive_timestamp_positions": pl.Int64,
        "normal_releases_during_aggressive_controller": pl.Int64,
        "normal_release_notional_during_aggressive_controller_twd": pl.Float64,
        "active_positions_after_aggressive_controller": pl.Int64,
        "active_notional_after_aggressive_controller_twd": pl.Float64,
        "max_active_same_product_notional_after_controller_twd": pl.Float64,
        "normal_releases_after_controller_to_eod": pl.Int64,
        "normal_release_notional_after_controller_to_eod_twd": pl.Float64,
        "eod_active_positions": pl.Int64,
        "eod_active_notional_twd": pl.Float64,
        "max_eod_active_same_product_notional_twd": pl.Float64,
        "controller_target_attained": pl.Boolean,
        "eod_target_attained": pl.Boolean,
        "eod_target_excess_notional_twd": pl.Float64,
        "post_expiry_close_positions": pl.Int64,
        "post_expiry_close_notional_twd": pl.Float64,
        "post_expiry_close_active_positions": pl.Int64,
        "post_expiry_close_active_notional_twd": pl.Float64,
        "max_post_expiry_close_active_same_product_notional_twd": pl.Float64,
        "post_expiry_close_target_attained": pl.Boolean,
        "post_expiry_close_target_excess_notional_twd": pl.Float64,
        "post_expiry_close_is_full_1330_market_replay": pl.Boolean,
        "realized_completed_positions": pl.Int64,
        "realized_gross_pnl_twd": pl.Float64,
        "realized_transaction_cost_twd": pl.Float64,
        "realized_net_pnl_twd": pl.Float64,
        "realized_net_loss_positions": pl.Int64,
        "realized_exit_one_way_spot_turnover_twd": pl.Float64,
        "realized_exit_paired_two_leg_turnover_twd": pl.Float64,
        "one_capacity_per_market_template_enforced": pl.Boolean,
        "missing_cache_positions_counted_as_closed": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
        "formal_ev_ready": pl.Boolean,
    }


def _controller_summary_schema() -> dict[str, pl.DataType]:
    return {
        "scenario_id": pl.String,
        "capacity_assumption": pl.String,
        "primary_conservative_result": pl.Boolean,
        "joint_volume_overcount_upper_bound": pl.Boolean,
        "portfolio_cap_twd": pl.Float64,
        "per_product_cap_fraction": pl.Float64,
        "per_product_cap_twd": pl.Float64,
        "eod_target_notional_twd": pl.Float64,
        "target_measurement_time": pl.String,
        "study_cutoff_is_twse_close": pl.Boolean,
        "extra_ten_minutes_to_twse_close_evaluated": pl.Boolean,
        "runtime_daily_liquidity_admission_gate_applied": pl.Boolean,
        "candidate_entries": pl.Int64,
        "accepted_entries": pl.Int64,
        "portfolio_only_blocked_entries": pl.Int64,
        "product_only_blocked_entries": pl.Int64,
        "both_caps_blocked_entries": pl.Int64,
        "held_product_exit_only_blocked_entries": pl.Int64,
        "forbidden_new_entries_at_or_after_1300": pl.Int64,
        "aggressive_closed_positions": pl.Int64,
        "aggressive_closed_notional_twd": pl.Float64,
        "normal_continuation_completed_positions": pl.Int64,
        "normal_continuation_unresolved_or_expiry_positions": pl.Int64,
        "normal_wins_equal_aggressive_timestamp_positions": pl.Int64,
        "accepted_model_imputed_full_carry_positions": pl.Int64,
        "accepted_double_exit_bias_possible_positions": pl.Int64,
        "accepted_expiry_trade_fallback_positions": pl.Int64,
        "accepted_expiry_official_close_positions": pl.Int64,
        "accepted_expiry_official_settlement_positions": pl.Int64,
        "completed_scenario_positions": pl.Int64,
        "completed_scenario_gross_pnl_twd": pl.Float64,
        "completed_scenario_transaction_cost_twd": pl.Float64,
        "completed_scenario_net_pnl_twd": pl.Float64,
        "completed_scenario_notional_twd": pl.Float64,
        "accepted_entry_one_way_turnover_twd": pl.Float64,
        "accepted_entry_paired_two_leg_turnover_twd": pl.Float64,
        "realized_exit_one_way_spot_turnover_twd": pl.Float64,
        "realized_exit_paired_two_leg_turnover_twd": pl.Float64,
        "mean_daily_accepted_entry_one_way_turnover_twd": pl.Float64,
        "mean_daily_accepted_entry_paired_two_leg_turnover_twd": pl.Float64,
        "mean_daily_realized_exit_one_way_spot_turnover_twd": pl.Float64,
        "mean_daily_realized_exit_paired_two_leg_turnover_twd": pl.Float64,
        "exit_one_way_turnover_basis": pl.String,
        "completed_scenario_weighted_gross_bp": pl.Float64,
        "completed_scenario_weighted_cost_bp": pl.Float64,
        "completed_scenario_weighted_net_bp": pl.Float64,
        "net_profitable_positions": pl.Int64,
        "net_loss_positions": pl.Int64,
        "net_zero_positions": pl.Int64,
        "negative_realized_days": pl.Int64,
        "realized_net_pnl_max_drawdown_twd": pl.Float64,
        "full_universe_missing_product_days": pl.Int64,
        "full_universe_missing_position_day_rows": pl.Int64,
        "accepted_missing_cache_position_day_exposures": pl.Int64,
        "accepted_missing_cache_notional_day_exposures_twd": pl.Float64,
        "days_controller_target_attained": pl.Int64,
        "days_eod_target_attained": pl.Int64,
        "max_eod_target_excess_notional_twd": pl.Float64,
        "post_expiry_close_positions": pl.Int64,
        "post_expiry_close_notional_twd": pl.Float64,
        "days_post_expiry_close_target_attained": pl.Int64,
        "max_post_expiry_close_target_excess_notional_twd": pl.Float64,
        "post_expiry_close_is_full_1330_market_replay": pl.Boolean,
        "market_template_capacity_reused_within_scenario": pl.Boolean,
        "cross_scenario_results_mutually_exclusive": pl.Boolean,
        "upstream_raw_content_integrity_bound": pl.Boolean,
        "transaction_cost_profile_id": pl.String,
        "transaction_cost_profile_complete_for_priced_cycles": pl.Boolean,
        "realized_pnl_only_no_mtm": pl.Boolean,
        "formal_ev_ready": pl.Boolean,
        "production_strategy_go": pl.Boolean,
    }


def _validate_controller_results(
    positions: pl.DataFrame,
    daily: pl.DataFrame,
    summary: pl.DataFrame,
    config: Aggressive1300AnalysisConfig,
) -> None:
    expected_schemas = (
        (positions, _controller_position_schema(), "position"),
        (daily, _controller_daily_schema(), "daily"),
        (summary, _controller_summary_schema(), "summary"),
    )
    for frame, schema, label in expected_schemas:
        if frame.columns != list(schema) or frame.schema != pl.Schema(schema):
            raise ValueError(f"controller {label} ordered schema changed")
    if summary.height != len(config.portfolio_caps_twd) * len(_CAPACITY_ASSUMPTIONS):
        raise ValueError("controller scenario summary denominator changed")
    if positions.height != (
        positions["scenario_id"].n_unique()
        * positions.filter(pl.col("scenario_id") == positions.item(0, "scenario_id")).height
    ):
        raise ValueError("controller position scenario grid is incomplete")
    if positions.select("scenario_id", "position_id").unique().height != positions.height:
        raise ValueError("controller duplicated a scenario physical position")
    if daily.select("scenario_id", "Date").unique().height != daily.height:
        raise ValueError("controller duplicated a scenario date")
    invalid_positions = positions.filter(
        pl.any_horizontal(
            pl.col(
                "scenario_id",
                "capacity_assumption",
                "Date",
                "ValueCode",
                "QuoteCode",
                "entry_route",
                "position_id",
                "policy_path_id",
                "entry_policy_generation_id",
                "admission_status",
                "scenario_terminal_source",
                "normal_outcome_type",
                "normal_outcome_status",
                "normal_outstanding_interval_end_exclusive",
                "supplemental_terminal_resolution",
                "transaction_cost_profile_id",
            ).is_null()
        )
        | pl.any_horizontal(
            pl.col(
                "scenario_id",
                "capacity_assumption",
                "Date",
                "ValueCode",
                "QuoteCode",
                "entry_route",
                "position_id",
                "policy_path_id",
                "entry_policy_generation_id",
                "admission_status",
                "scenario_terminal_source",
                "normal_outcome_type",
                "normal_outcome_status",
                "normal_outstanding_interval_end_exclusive",
                "supplemental_terminal_resolution",
                "transaction_cost_profile_id",
            ).str.len_chars()
            == 0
        )
        | pl.any_horizontal(
            pl.col(
                "model_imputed_full_carry_on_unknown",
                "double_exit_bias_possible",
                "expiry_uses_last_valid_session_mark",
                "expiry_uses_last_observed_session_mark",
                "expiry_mark_uses_trade_fallback",
            ).is_null()
        )
        | (
            pl.col("supplemental_terminal_resolution").str.starts_with("expiry_")
            != pl.all_horizontal(
                pl.col(
                    "spot_expiry_mark_is_executable_bbo",
                    "future_expiry_mark_is_executable_bbo",
                    "expiry_mark_is_official_close",
                    "expiry_mark_is_official_settlement",
                ).is_not_null()
            )
        ).fill_null(True)
        | (
            ~pl.col("supplemental_terminal_resolution").str.starts_with("expiry_")
            & pl.any_horizontal(
                pl.col(
                    "spot_expiry_mark_is_executable_bbo",
                    "future_expiry_mark_is_executable_bbo",
                    "expiry_mark_is_official_close",
                    "expiry_mark_is_official_settlement",
                ).is_not_null()
            )
        )
        | (pl.col("transaction_cost_profile_id") != TransactionCostProfile().profile_id)
        | (pl.col("accepted") & (pl.col("admission_status") != "accepted_before_1300"))
        | (~pl.col("accepted") & (pl.col("admission_status") == "accepted_before_1300"))
        | (
            pl.col("portfolio_cap_blocked")
            != pl.col("admission_status").is_in(
                [
                    "portfolio_hard_cap_blocked",
                    "portfolio_and_product_hard_caps_blocked",
                ]
            )
        )
        | (
            pl.col("per_product_cap_blocked")
            != pl.col("admission_status").is_in(
                [
                    "per_product_hard_cap_blocked",
                    "portfolio_and_product_hard_caps_blocked",
                ]
            )
        )
        | (
            pl.col("held_product_exit_only_blocked")
            != (pl.col("admission_status") == "held_product_exit_only_blocked")
        )
        | (
            pl.col("aggressive_close_allocated")
            & (
                ~pl.col("accepted")
                | ~pl.col("scenario_terminal_cashflow_priced")
                | (pl.col("scenario_terminal_source")
                   != "aggressive_1300_template_capacity")
                | pl.any_horizontal(
                    pl.col(
                        "aggressive_market_template_id",
                        "aggressive_template_source",
                        "aggressive_winner_route",
                        "scenario_terminal_date",
                        "scenario_exit_decision_time_ns",
                        "scenario_exit_spot_price",
                        "scenario_exit_future_price",
                        "scenario_gross_cycle_pnl_twd",
                        "scenario_gross_cycle_bp",
                    ).is_null()
                )
            )
        )
        | (
            ~pl.col("aggressive_close_allocated")
            & pl.any_horizontal(
                pl.col(
                    "aggressive_market_template_id",
                    "aggressive_template_source",
                    "aggressive_winner_route",
                ).is_not_null()
            )
        )
        | (
            pl.col("accepted")
            & ~pl.col("aggressive_close_allocated")
            & pl.col("normal_terminal_cashflow_priced")
            & (
                (pl.col("scenario_terminal_source") != "selected_normal_continuation")
                | ~pl.col("scenario_terminal_cashflow_priced")
                | (pl.col("scenario_terminal_date") != pl.col("normal_terminal_date"))
                | (
                    pl.col("scenario_exit_decision_time_ns")
                    != pl.col("normal_exit_decision_time_ns")
                )
            )
        )
        | (
            pl.col("accepted")
            & ~pl.col("aggressive_close_allocated")
            & ~pl.col("normal_terminal_cashflow_priced")
            & (
                (pl.col("scenario_terminal_source")
                 != "selected_normal_continuation_unresolved_or_expiry")
                | pl.col("scenario_terminal_cashflow_priced")
            )
        )
        | (
            ~pl.col("accepted")
            & (
                (pl.col("scenario_terminal_source") != "entry_not_admitted")
                | pl.col("scenario_terminal_cashflow_priced")
                | pl.col("aggressive_close_allocated")
            )
        )
        | (
            pl.col("one_capacity_per_market_template_enforced")
            != pl.col("primary_conservative_result")
        ).fill_null(True)
        | (
            pl.col("joint_volume_overcount_upper_bound")
            == pl.col("primary_conservative_result")
        ).fill_null(True)
        | (pl.col("cross_scenario_capacity_reuse_counterfactual") != True).fill_null(True)  # noqa: E712
        | (pl.col("missing_cache_position_counted_as_closed") != False).fill_null(True)  # noqa: E712
        | pl.col("joint_volume_allocated")
        | (
            pl.col("transaction_cost_point_identified")
            != pl.col("scenario_terminal_cashflow_priced")
        ).fill_null(True)
        | (
            pl.col("transaction_cost_profile_complete_for_priced_cycles")
            != pl.col("scenario_terminal_cashflow_priced")
        ).fill_null(True)
        | (pl.col("realized_pnl_only_no_mtm") != True).fill_null(True)  # noqa: E712
        | (
            pl.col("scenario_terminal_cashflow_priced")
            & (
                pl.any_horizontal(
                    pl.col(
                        "scenario_terminal_date",
                        "scenario_exit_decision_time_ns",
                        "scenario_exit_spot_price",
                        "scenario_exit_future_price",
                        "scenario_gross_cycle_pnl_twd",
                        "scenario_gross_cycle_bp",
                        "scenario_transaction_cost_twd",
                        "scenario_net_cycle_pnl_twd",
                        "scenario_net_cycle_pnl_bp",
                        "scenario_net_profitable",
                    ).is_null()
                )
                | (
                    (
                        pl.col("scenario_gross_cycle_pnl_twd")
                        - pl.col("scenario_transaction_cost_twd")
                        - pl.col("scenario_net_cycle_pnl_twd")
                    ).abs()
                    > 1e-6
                )
                | (
                    (
                        pl.col("entry_contract_size_shares")
                        * (
                            pl.col("scenario_exit_spot_price")
                            - pl.col("entry_spot_price")
                            + pl.col("entry_future_price")
                            - pl.col("scenario_exit_future_price")
                        )
                        - pl.col("scenario_gross_cycle_pnl_twd")
                    ).abs()
                    > 1e-6
                )
                | (
                    (
                        pl.col("scenario_gross_cycle_pnl_twd")
                        / pl.col("normalization_notional_twd")
                        * 10_000.0
                        - pl.col("scenario_gross_cycle_bp")
                    ).abs()
                    > 1e-6
                )
                | (
                    (
                        pl.col("scenario_net_cycle_pnl_twd")
                        / pl.col("normalization_notional_twd")
                        * 10_000.0
                        - pl.col("scenario_net_cycle_pnl_bp")
                    ).abs()
                    > 1e-6
                )
                | (
                    pl.col("scenario_net_profitable")
                    != (pl.col("scenario_net_cycle_pnl_twd") > 0)
                )
                | ~pl.all_horizontal(
                    pl.col(
                        "scenario_exit_spot_price",
                        "scenario_exit_future_price",
                        "scenario_gross_cycle_pnl_twd",
                        "scenario_gross_cycle_bp",
                        "scenario_transaction_cost_twd",
                        "scenario_net_cycle_pnl_twd",
                        "scenario_net_cycle_pnl_bp",
                    ).is_finite()
                )
                | (pl.col("scenario_exit_spot_price") <= 0)
                | (pl.col("scenario_exit_future_price") <= 0)
                | (pl.col("scenario_transaction_cost_twd") <= 0)
            )
        )
        | (
            ~pl.col("scenario_terminal_cashflow_priced")
            & pl.any_horizontal(
                pl.col(
                    "scenario_terminal_date",
                    "scenario_exit_decision_time_ns",
                    "scenario_exit_spot_price",
                    "scenario_exit_future_price",
                    "scenario_gross_cycle_pnl_twd",
                    "scenario_gross_cycle_bp",
                    "scenario_transaction_cost_twd",
                    "scenario_net_cycle_pnl_twd",
                    "scenario_net_cycle_pnl_bp",
                    "scenario_net_profitable",
                ).is_not_null()
            )
        )
        | pl.col("formal_ev_ready")
        | pl.col("production_strategy_go")
    )
    if invalid_positions.height:
        raise ValueError("controller position lifecycle/safety invariant failed")
    for row in positions.filter(pl.col("scenario_terminal_cashflow_priced")).select(
        "Date",
        "position_established_ns",
        "scenario_terminal_date",
        "scenario_exit_decision_time_ns",
        "entry_spot_price",
        "scenario_exit_spot_price",
        "entry_future_price",
        "scenario_exit_future_price",
        "entry_contract_size_shares",
        "scenario_transaction_cost_twd",
    ).iter_rows(named=True):
        if (
            str(row["scenario_terminal_date"]),
            int(row["scenario_exit_decision_time_ns"]),
        ) < (str(row["Date"]), int(row["position_established_ns"])):
            raise ValueError("controller scenario terminal precedes entry")
        expected_cost = _cycle_transaction_cost(
            entry_date=str(row["Date"]),
            terminal_date=str(row["scenario_terminal_date"]),
            entry_spot_price=float(row["entry_spot_price"]),
            exit_spot_price=float(row["scenario_exit_spot_price"]),
            entry_future_price=float(row["entry_future_price"]),
            exit_future_price=float(row["scenario_exit_future_price"]),
            shares=int(row["entry_contract_size_shares"]),
        )
        if not math.isclose(
            float(row["scenario_transaction_cost_twd"]),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("controller transaction cost differs from formal profile")
    allocated = positions.filter(
        pl.col("aggressive_close_allocated") & pl.col("primary_conservative_result")
    )
    if allocated.select("scenario_id", "aggressive_market_template_id").unique().height != allocated.height:
        raise ValueError("controller reused a market template within a scenario")
    if daily.filter(
        (
            pl.col("one_capacity_per_market_template_enforced")
            != pl.col("primary_conservative_result")
        ).fill_null(True)
        | (
            pl.col("joint_volume_overcount_upper_bound")
            == pl.col("primary_conservative_result")
        ).fill_null(True)
        | (pl.col("missing_cache_positions_counted_as_closed") != False).fill_null(True)  # noqa: E712
        | pl.col("joint_volume_allocated")
        | pl.col("formal_ev_ready")
        | (pl.col("eod_target_excess_notional_twd") < -1e-9)
        | (pl.col("post_expiry_close_target_excess_notional_twd") < -1e-9)
        | pl.col("post_expiry_close_is_full_1330_market_replay")
        | (pl.col("eod_active_notional_twd") > pl.col("portfolio_cap_twd") + 1e-6)
        | (
            pl.col("post_expiry_close_active_notional_twd")
            > pl.col("eod_active_notional_twd") + 1e-6
        )
        | (
            (
                pl.col("eod_active_notional_twd")
                - pl.col("post_expiry_close_notional_twd")
                - pl.col("post_expiry_close_active_notional_twd")
            ).abs()
            > 1e-6
        )
        | (
            pl.col("eod_active_positions")
            - pl.col("post_expiry_close_positions")
            != pl.col("post_expiry_close_active_positions")
        )
        | (
            pl.col("controller_target_attained")
            != (
                pl.col("active_notional_after_aggressive_controller_twd")
                <= pl.col("eod_target_notional_twd") + 1e-9
            )
        )
        | (
            pl.col("eod_target_attained")
            != (
                pl.col("eod_active_notional_twd")
                <= pl.col("eod_target_notional_twd") + 1e-9
            )
        )
        | (
            pl.col("post_expiry_close_target_attained")
            != (
                pl.col("post_expiry_close_active_notional_twd")
                <= pl.col("eod_target_notional_twd") + 1e-9
            )
        )
        | (
            (
                pl.col("eod_target_excess_notional_twd")
                - (
                    pl.col("eod_active_notional_twd")
                    - pl.col("eod_target_notional_twd")
                ).clip(lower_bound=0.0)
            ).abs()
            > 1e-6
        )
        | (
            (
                pl.col("post_expiry_close_target_excess_notional_twd")
                - (
                    pl.col("post_expiry_close_active_notional_twd")
                    - pl.col("eod_target_notional_twd")
                ).clip(lower_bound=0.0)
            ).abs()
            > 1e-6
        )
        | (
            pl.col("max_active_same_product_notional_at_1300_twd")
            > pl.col("portfolio_cap_twd") * float(config.per_product_cap_fraction)
            + 1e-6
        )
        | (
            pl.col("max_active_same_product_notional_after_controller_twd")
            > pl.col("portfolio_cap_twd") * float(config.per_product_cap_fraction)
            + 1e-6
        )
        | (
            pl.col("max_eod_active_same_product_notional_twd")
            > pl.col("portfolio_cap_twd") * float(config.per_product_cap_fraction)
            + 1e-6
        )
        | (
            pl.col("max_post_expiry_close_active_same_product_notional_twd")
            > pl.col("portfolio_cap_twd") * float(config.per_product_cap_fraction)
            + 1e-6
        )
    ).height:
        raise ValueError("controller daily cap/safety invariant failed")
    if summary.filter(
        (
            pl.col("market_template_capacity_reused_within_scenario")
            != pl.col("joint_volume_overcount_upper_bound")
        ).fill_null(True)
        | (
            pl.col("primary_conservative_result")
            == pl.col("joint_volume_overcount_upper_bound")
        ).fill_null(True)
        | (pl.col("cross_scenario_results_mutually_exclusive") != True).fill_null(True)  # noqa: E712
        | pl.col("upstream_raw_content_integrity_bound")
        | (pl.col("transaction_cost_profile_complete_for_priced_cycles") != True).fill_null(True)  # noqa: E712
        | (pl.col("realized_pnl_only_no_mtm") != True).fill_null(True)  # noqa: E712
        | pl.col("post_expiry_close_is_full_1330_market_replay")
        | (
            pl.col("exit_one_way_turnover_basis")
            != "actual_spot_exit_price_times_shares"
        )
        | pl.col("formal_ev_ready")
        | pl.col("production_strategy_go")
    ).height:
        raise ValueError("controller summary safety invariant failed")
    for row in summary.iter_rows(named=True):
        scenario = str(row["scenario_id"])
        scenario_positions = positions.filter(pl.col("scenario_id") == scenario)
        scenario_daily = daily.filter(pl.col("scenario_id") == scenario).sort("Date")
        accepted = scenario_positions.filter(pl.col("accepted"))
        completed = accepted.filter(pl.col("scenario_terminal_cashflow_priced"))
        aggressive = accepted.filter(pl.col("aggressive_close_allocated"))
        daily_by_date = {
            str(item["Date"]): item for item in scenario_daily.iter_rows(named=True)
        }
        realized_by_date: dict[str, list[dict[str, object]]] = {}
        for item in completed.iter_rows(named=True):
            realized_by_date.setdefault(str(item["scenario_terminal_date"]), []).append(item)
        if set(realized_by_date) - set(daily_by_date):
            raise ValueError("controller realized terminal is outside the session calendar")
        cumulative_net = 0.0
        running_peak = 0.0
        expected_max_drawdown = 0.0
        expected_negative_days = 0
        for date, daily_item in daily_by_date.items():
            realized = realized_by_date.get(date, [])
            admitted = accepted.filter(pl.col("Date") == date)
            entry_one_way = float(
                admitted["normalization_notional_twd"].sum() or 0.0
            )
            entry_two_leg = sum(
                int(item["entry_contract_size_shares"])
                * (
                    float(item["entry_spot_price"])
                    + float(item["entry_future_price"])
                )
                for item in admitted.iter_rows(named=True)
            )
            gross = sum(float(item["scenario_gross_cycle_pnl_twd"]) for item in realized)
            cost = sum(float(item["scenario_transaction_cost_twd"]) for item in realized)
            net = sum(float(item["scenario_net_cycle_pnl_twd"]) for item in realized)
            loss_positions = sum(
                float(item["scenario_net_cycle_pnl_twd"]) < 0 for item in realized
            )
            exit_one_way = sum(
                int(item["entry_contract_size_shares"])
                * float(item["scenario_exit_spot_price"])
                for item in realized
            )
            exit_two_leg = sum(
                int(item["entry_contract_size_shares"])
                * (
                    float(item["scenario_exit_spot_price"])
                    + float(item["scenario_exit_future_price"])
                )
                for item in realized
            )
            if (
                int(daily_item["realized_completed_positions"]) != len(realized)
                or int(daily_item["accepted_entries_before_1300"])
                != admitted.height
                or int(daily_item["realized_net_loss_positions"]) != loss_positions
                or not math.isclose(
                    float(daily_item["accepted_entry_one_way_turnover_twd"]),
                    entry_one_way,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(daily_item["accepted_entry_two_leg_turnover_twd"]),
                    entry_two_leg,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(daily_item["realized_gross_pnl_twd"]),
                    gross,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(daily_item["realized_transaction_cost_twd"]),
                    cost,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(daily_item["realized_net_pnl_twd"]),
                    net,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(daily_item["realized_exit_one_way_spot_turnover_twd"]),
                    exit_one_way,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(
                        daily_item[
                            "realized_exit_paired_two_leg_turnover_twd"
                        ]
                    ),
                    exit_two_leg,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("controller daily realized PnL is not reproducible")
            if net < 0:
                expected_negative_days += 1
            cumulative_net += net
            running_peak = max(running_peak, cumulative_net)
            expected_max_drawdown = max(
                expected_max_drawdown, running_peak - cumulative_net
            )

        completed_notional = float(completed["normalization_notional_twd"].sum() or 0.0)
        completed_gross = float(completed["scenario_gross_cycle_pnl_twd"].sum() or 0.0)
        completed_cost = float(completed["scenario_transaction_cost_twd"].sum() or 0.0)
        completed_net = float(completed["scenario_net_cycle_pnl_twd"].sum() or 0.0)
        accepted_entry_one_way = float(
            accepted["normalization_notional_twd"].sum() or 0.0
        )
        accepted_entry_two_leg = sum(
            int(item["entry_contract_size_shares"])
            * (float(item["entry_spot_price"]) + float(item["entry_future_price"]))
            for item in accepted.iter_rows(named=True)
        )
        realized_exit_one_way = sum(
            int(item["entry_contract_size_shares"])
            * float(item["scenario_exit_spot_price"])
            for item in completed.iter_rows(named=True)
        )
        realized_exit_two_leg = sum(
            int(item["entry_contract_size_shares"])
            * (
                float(item["scenario_exit_spot_price"])
                + float(item["scenario_exit_future_price"])
            )
            for item in completed.iter_rows(named=True)
        )
        expected_ints = {
            "candidate_entries": scenario_positions.height,
            "accepted_entries": accepted.height,
            "portfolio_only_blocked_entries": scenario_positions.filter(
                pl.col("admission_status") == "portfolio_hard_cap_blocked"
            ).height,
            "product_only_blocked_entries": scenario_positions.filter(
                pl.col("admission_status") == "per_product_hard_cap_blocked"
            ).height,
            "both_caps_blocked_entries": scenario_positions.filter(
                pl.col("admission_status")
                == "portfolio_and_product_hard_caps_blocked"
            ).height,
            "held_product_exit_only_blocked_entries": scenario_positions.filter(
                pl.col("admission_status") == "held_product_exit_only_blocked"
            ).height,
            "forbidden_new_entries_at_or_after_1300": scenario_positions.filter(
                pl.col("admission_status") == "forbidden_new_entry_at_or_after_1300"
            ).height,
            "aggressive_closed_positions": aggressive.height,
            "normal_continuation_completed_positions": accepted.filter(
                pl.col("scenario_terminal_source") == "selected_normal_continuation"
            ).height,
            "normal_continuation_unresolved_or_expiry_positions": accepted.filter(
                pl.col("scenario_terminal_source")
                == "selected_normal_continuation_unresolved_or_expiry"
            ).height,
            "normal_wins_equal_aggressive_timestamp_positions": int(
                scenario_daily[
                    "normal_wins_equal_aggressive_timestamp_positions"
                ].sum()
                or 0
            ),
            "accepted_model_imputed_full_carry_positions": accepted.filter(
                pl.col("model_imputed_full_carry_on_unknown")
            ).height,
            "accepted_double_exit_bias_possible_positions": accepted.filter(
                pl.col("double_exit_bias_possible")
            ).height,
            "accepted_expiry_trade_fallback_positions": accepted.filter(
                pl.col("expiry_mark_uses_trade_fallback")
            ).height,
            "accepted_expiry_official_close_positions": accepted.filter(
                pl.col("expiry_mark_is_official_close") == True  # noqa: E712
            ).height,
            "accepted_expiry_official_settlement_positions": accepted.filter(
                pl.col("expiry_mark_is_official_settlement") == True  # noqa: E712
            ).height,
            "completed_scenario_positions": completed.height,
            "net_profitable_positions": completed.filter(
                pl.col("scenario_net_cycle_pnl_twd") > 0
            ).height,
            "net_loss_positions": completed.filter(
                pl.col("scenario_net_cycle_pnl_twd") < 0
            ).height,
            "net_zero_positions": completed.filter(
                pl.col("scenario_net_cycle_pnl_twd") == 0
            ).height,
            "negative_realized_days": expected_negative_days,
            "days_controller_target_attained": scenario_daily.filter(
                pl.col("controller_target_attained")
            ).height,
            "days_eod_target_attained": scenario_daily.filter(
                pl.col("eod_target_attained")
            ).height,
            "post_expiry_close_positions": int(
                scenario_daily["post_expiry_close_positions"].sum() or 0
            ),
            "days_post_expiry_close_target_attained": scenario_daily.filter(
                pl.col("post_expiry_close_target_attained")
            ).height,
            "accepted_missing_cache_position_day_exposures": int(
                scenario_daily["missing_cache_position_rows_at_1300"].sum() or 0
            ),
        }
        expected_floats = {
            "aggressive_closed_notional_twd": float(
                aggressive["normalization_notional_twd"].sum() or 0.0
            ),
            "completed_scenario_gross_pnl_twd": completed_gross,
            "completed_scenario_transaction_cost_twd": completed_cost,
            "completed_scenario_net_pnl_twd": completed_net,
            "completed_scenario_notional_twd": completed_notional,
            "accepted_entry_one_way_turnover_twd": accepted_entry_one_way,
            "accepted_entry_paired_two_leg_turnover_twd": accepted_entry_two_leg,
            "realized_exit_one_way_spot_turnover_twd": realized_exit_one_way,
            "realized_exit_paired_two_leg_turnover_twd": realized_exit_two_leg,
            "mean_daily_accepted_entry_one_way_turnover_twd": (
                accepted_entry_one_way / scenario_daily.height
            ),
            "mean_daily_accepted_entry_paired_two_leg_turnover_twd": (
                accepted_entry_two_leg / scenario_daily.height
            ),
            "mean_daily_realized_exit_one_way_spot_turnover_twd": (
                realized_exit_one_way / scenario_daily.height
            ),
            "mean_daily_realized_exit_paired_two_leg_turnover_twd": (
                realized_exit_two_leg / scenario_daily.height
            ),
            "accepted_missing_cache_notional_day_exposures_twd": float(
                scenario_daily["missing_cache_notional_at_1300_twd"].sum() or 0.0
            ),
            "realized_net_pnl_max_drawdown_twd": expected_max_drawdown,
            "max_eod_target_excess_notional_twd": float(
                scenario_daily["eod_target_excess_notional_twd"].max() or 0.0
            ),
            "post_expiry_close_notional_twd": float(
                scenario_daily["post_expiry_close_notional_twd"].sum() or 0.0
            ),
            "max_post_expiry_close_target_excess_notional_twd": float(
                scenario_daily[
                    "post_expiry_close_target_excess_notional_twd"
                ].max()
                or 0.0
            ),
        }
        for name, expected in expected_ints.items():
            if int(row[name]) != expected:
                raise ValueError(f"controller scenario summary is not reproducible: {name}")
        for name, expected in expected_floats.items():
            if not math.isclose(
                float(row[name]), expected, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"controller scenario summary is not reproducible: {name}")
        expected_weighted = {
            "completed_scenario_weighted_gross_bp": completed_gross,
            "completed_scenario_weighted_cost_bp": completed_cost,
            "completed_scenario_weighted_net_bp": completed_net,
        }
        for name, numerator in expected_weighted.items():
            actual = row[name]
            if completed_notional == 0:
                if actual is not None:
                    raise ValueError(f"controller empty weighted metric is populated: {name}")
            elif actual is None or not math.isclose(
                float(actual),
                numerator / completed_notional * 10_000.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"controller weighted metric is not reproducible: {name}")
    controls = summary.filter(
        pl.col("portfolio_cap_twd") <= float(config.eod_target_ceiling_twd)
    )
    if controls.filter(pl.col("aggressive_closed_positions") != 0).height:
        raise ValueError("10M/20M identity controls unexpectedly used aggressive exits")
    position_identity_columns = [
        "position_id",
        "accepted",
        "admission_status",
        "portfolio_cap_blocked",
        "per_product_cap_blocked",
        "held_product_exit_only_blocked",
        "scenario_terminal_source",
        "scenario_terminal_cashflow_priced",
        "scenario_terminal_date",
        "scenario_exit_decision_time_ns",
        "scenario_exit_spot_price",
        "scenario_exit_future_price",
        "scenario_gross_cycle_pnl_twd",
        "scenario_gross_cycle_bp",
        "scenario_transaction_cost_twd",
        "scenario_net_cycle_pnl_twd",
        "scenario_net_cycle_pnl_bp",
        "scenario_net_profitable",
    ]
    daily_identity_columns = [
        "Date",
        "accepted_entries_before_1300",
        "accepted_entry_one_way_turnover_twd",
        "accepted_entry_two_leg_turnover_twd",
        "portfolio_only_blocked_entries_before_1300",
        "product_only_blocked_entries_before_1300",
        "both_caps_blocked_entries_before_1300",
        "held_product_exit_only_blocked_entries",
        "forbidden_new_entries_at_or_after_1300",
        "active_positions_at_1300",
        "active_notional_at_1300_twd",
        "active_positions_after_aggressive_controller",
        "active_notional_after_aggressive_controller_twd",
        "eod_active_positions",
        "eod_active_notional_twd",
        "post_expiry_close_positions",
        "post_expiry_close_notional_twd",
        "post_expiry_close_active_positions",
        "post_expiry_close_active_notional_twd",
        "realized_completed_positions",
        "realized_gross_pnl_twd",
        "realized_transaction_cost_twd",
        "realized_net_pnl_twd",
        "realized_net_loss_positions",
        "realized_exit_one_way_spot_turnover_twd",
        "realized_exit_paired_two_leg_turnover_twd",
    ]
    for cap in controls["portfolio_cap_twd"].unique().to_list():
        by_cap_positions = positions.filter(pl.col("portfolio_cap_twd") == cap)
        conservative_positions = by_cap_positions.filter(
            pl.col("primary_conservative_result")
        ).select(position_identity_columns).sort("position_id")
        optimistic_positions = by_cap_positions.filter(
            ~pl.col("primary_conservative_result")
        ).select(position_identity_columns).sort("position_id")
        by_cap_daily = daily.filter(pl.col("portfolio_cap_twd") == cap)
        conservative_daily = by_cap_daily.filter(
            pl.col("primary_conservative_result")
        ).select(daily_identity_columns).sort("Date")
        optimistic_daily = by_cap_daily.filter(
            ~pl.col("primary_conservative_result")
        ).select(daily_identity_columns).sort("Date")
        if (
            not conservative_positions.equals(optimistic_positions, null_equal=True)
            or not conservative_daily.equals(optimistic_daily, null_equal=True)
        ):
            raise ValueError("10M/20M control differs from same-rule normal baseline")


def load_all_exact_entry_prices(
    paths: pl.DataFrame,
    *,
    execution_manifest_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load marker/hash-verified entry prices for every selected physical path."""

    required = {
        "Date",
        "ValueCode",
        "entry_policy_generation_id",
        "entry_route",
        "position_established_ns",
        "normalization_notional_twd",
        "policy_path_id",
    }
    missing = sorted(required - set(paths.columns))
    if missing:
        raise ValueError(f"entry-price path source missing columns: {missing}")
    manifest = _partition_manifest(Path(execution_manifest_path))
    frames: list[pl.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for group in paths.partition_by(["Date", "ValueCode"], maintain_order=True):
        date = str(group.item(0, "Date"))
        value = str(group.item(0, "ValueCode"))
        identifiers = group["entry_policy_generation_id"].unique().to_list()
        frame, inventory = _read_bound_partition_artifact(
            manifest,
            (date, value),
            filename="execution_action_facts.parquet",
            source_kind="aggressive_1300_entry_execution_action",
            columns=(
                "policy_generation_id",
                "route",
                "entry_spot_price",
                "entry_future_price",
                "entry_hedge_contract_size_shares",
                "entry_hedge_decision_time_ns",
                "full_fill",
                "entry_hedge_label_observed",
                "entry_hedge_executable",
            ),
            filter_column="policy_generation_id",
            identifiers=identifiers,
        )
        frames.append(
            frame.rename(
                {
                    "policy_generation_id": "entry_policy_generation_id",
                    "route": "exact_entry_route",
                    "entry_hedge_contract_size_shares": "entry_contract_size_shares",
                    "entry_hedge_decision_time_ns": "exact_position_established_ns",
                }
            )
        )
        inventory_rows.append(inventory)
    facts = pl.concat(frames, how="vertical_relaxed")
    if (
        facts.height != paths.height
        or facts["entry_policy_generation_id"].n_unique() != paths.height
    ):
        raise ValueError("exact entry-price facts are not one-to-one")
    base = paths.drop(
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        strict=False,
    )
    joined = base.join(
        facts,
        on="entry_policy_generation_id",
        how="left",
        validate="1:1",
        suffix="_exact",
    )
    invalid = joined.filter(
        pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exact_position_established_ns",
            ).is_null()
        )
        | ~pl.all_horizontal(
            pl.col("entry_spot_price", "entry_future_price").is_finite()
        )
        | pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
            )
            <= 0
        )
        | (pl.col("exact_entry_route") != pl.col("entry_route"))
        | (pl.col("exact_position_established_ns") != pl.col("position_established_ns"))
        | (pl.col("full_fill") != True).fill_null(True)  # noqa: E712
        | (pl.col("entry_hedge_label_observed") != True).fill_null(True)  # noqa: E712
        | (pl.col("entry_hedge_executable") != True).fill_null(True)  # noqa: E712
        | (
            (
                pl.col("entry_spot_price")
                * pl.col("entry_contract_size_shares")
                - pl.col("normalization_notional_twd")
            ).abs()
            > 1e-6
        )
    )
    if invalid.height:
        raise ValueError(
            "exact entry-price facts disagree with selected paths: "
            f"invalid={invalid.height}"
        )
    enriched = joined.drop(
        "exact_entry_route",
        "exact_position_established_ns",
        "full_fill",
        "entry_hedge_label_observed",
        "entry_hedge_executable",
    )
    inventory = pl.from_dicts(
        inventory_rows, schema=_entry_price_source_inventory_schema(), strict=True
    ).sort(["Date", "ValueCode"])
    return enriched, inventory


def load_supplemental_normal_paths(
    root: Path,
    *,
    base_paths: pl.DataFrame,
    base_metadata: Mapping[str, object],
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load frozen v3 expiry-close paths and canonicalize stale path labels.

    V3 overlays 171 expiry paths on the source-bound v2 continuation bundle
    using paired MarketInfo daily close facts.  Both v3 and its v2/close-fact
    parents are path/hash bound here.  The controller treats the resulting
    terminal cashflow/date/time/prices as authoritative while retaining every
    explicit approximation and expiry provenance field.
    """

    source_root = Path(root)
    if source_root.is_symlink():
        raise ValueError("supplemental source root cannot be a symlink")
    source_root = source_root.resolve()
    if source_root != DEFAULT_SUPPLEMENTAL_ROOT.resolve():
        raise ValueError("supplemental v3 root identity changed")
    marker_path = source_root / "complete.json"
    artifact_path = source_root / "supplemental_paths.parquet"
    if (
        marker_path.is_symlink()
        or artifact_path.is_symlink()
        or _file_sha256(marker_path) != SUPPLEMENTAL_COMPLETE_SHA256
    ):
        raise ValueError("supplemental v3 marker identity changed")
    marker = _read_json(marker_path)
    declaration = marker.get("artifacts", {}).get("supplemental_paths.parquet")
    if (
        marker.get("complete") is not True
        or marker.get("analysis_only") is not True
        or marker.get("production_strategy_go") is not False
        or marker.get("runner_version")
        != "supplemental_imputed_full_carry_grouped_runner_v3_expiry_daily_close"
        or marker.get("marker_payload_sha256")
        != SUPPLEMENTAL_MARKER_PAYLOAD_SHA256
        or marker.get("model_imputed_full_carry_on_unknown") is not True
        or marker.get("state_sampling_approximate") is not True
        or marker.get("double_exit_bias_possible") is not True
        or marker.get("expiry_mark_is_executable_bbo") is not False
        or marker.get("expiry_mark_is_official_close") is not True
        or marker.get("expiry_mark_is_official_settlement") is not False
        or marker.get("future_settlement_price_used") is not False
        or marker.get("expiry_terminal_resolution")
        != "expiry_same_day_two_leg_close_price"
        or int(marker.get("expiry_paths_repriced", -1)) != 171
        or int(marker.get("expiry_pair_marks", -1)) != 45
        or not math.isclose(
            float(marker.get("gross_pnl_delta_vs_v2_twd", math.nan)),
            293_700.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not isinstance(declaration, dict)
        or declaration.get("sha256") != SUPPLEMENTAL_PATH_SHA256
        or int(declaration.get("rows", -1)) != SUPPLEMENTAL_PATH_ROWS
        or _file_sha256(artifact_path) != SUPPLEMENTAL_PATH_SHA256
        or artifact_path.stat().st_size != int(declaration.get("bytes", -1))
        or {name: str(dtype) for name, dtype in pl.read_parquet_schema(artifact_path).items()}
        != declaration.get("schema")
    ):
        raise ValueError("supplemental v3 source contract changed")

    upstream_root = UPSTREAM_SUPPLEMENTAL_V2_ROOT.resolve()
    upstream_marker_path = upstream_root / "complete.json"
    upstream_artifact_path = upstream_root / "supplemental_paths.parquet"
    if (
        str(marker.get("upstream_root")) != str(upstream_root)
        or marker.get("upstream_complete_sha256")
        != UPSTREAM_SUPPLEMENTAL_V2_COMPLETE_SHA256
        or marker.get("upstream_marker_payload_sha256")
        != UPSTREAM_SUPPLEMENTAL_V2_MARKER_PAYLOAD_SHA256
        or upstream_marker_path.is_symlink()
        or upstream_artifact_path.is_symlink()
        or _file_sha256(upstream_marker_path)
        != UPSTREAM_SUPPLEMENTAL_V2_COMPLETE_SHA256
        or _file_sha256(upstream_artifact_path)
        != UPSTREAM_SUPPLEMENTAL_V2_PATH_SHA256
    ):
        raise ValueError("supplemental v3 upstream v2 identity changed")
    upstream_marker = _read_json(upstream_marker_path)
    if (
        upstream_marker.get("marker_payload_sha256")
        != UPSTREAM_SUPPLEMENTAL_V2_MARKER_PAYLOAD_SHA256
        or upstream_marker.get("runner_version")
        != "supplemental_imputed_full_carry_grouped_runner_v2_trade_fallback"
        or upstream_marker.get("model_imputed_full_carry_on_unknown") is not True
        or upstream_marker.get("state_sampling_approximate") is not True
        or upstream_marker.get("target_cancel_clock_exact") is not False
        or upstream_marker.get("delayed_hedge_snapshot_exact") is not False
        or upstream_marker.get("double_exit_bias_possible") is not True
    ):
        raise ValueError("supplemental upstream v2 contract changed")
    sources = upstream_marker.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("supplemental upstream v2 source lineage is missing")
    lineage_pairs = {
        "source_root": "source_root",
        "source_complete_sha256": "source_complete_sha256",
        "execution_root": "execution_root",
        "execution_manifest_sha256": "execution_manifest_sha256",
        "same_day_exit_root": "same_day_exit_root",
        "same_day_exit_manifest_sha256": "same_day_exit_manifest_sha256",
        "cross_session_exit_root": "cross_session_exit_root",
        "cross_session_exit_manifest_sha256": "cross_session_exit_manifest_sha256",
        "universe_root": "universe_root",
        "universe_complete_sha256": "universe_complete_sha256",
    }
    for supplemental_name, base_name in lineage_pairs.items():
        if str(sources.get(supplemental_name)) != str(base_metadata.get(base_name)):
            raise ValueError(
                f"supplemental v2 lineage disagrees with formal source: {supplemental_name}"
            )

    close_root = EXPIRY_CLOSE_FACT_ROOT.resolve()
    close_marker_path = close_root / "complete.json"
    close_artifact_path = close_root / "daily_close_facts.parquet"
    if (
        str(marker.get("close_fact_root")) != str(close_root)
        or marker.get("close_fact_complete_sha256")
        != EXPIRY_CLOSE_FACT_COMPLETE_SHA256
        or marker.get("close_fact_marker_payload_sha256")
        != EXPIRY_CLOSE_FACT_MARKER_PAYLOAD_SHA256
        or close_marker_path.is_symlink()
        or close_artifact_path.is_symlink()
        or _file_sha256(close_marker_path) != EXPIRY_CLOSE_FACT_COMPLETE_SHA256
        or _file_sha256(close_artifact_path) != EXPIRY_DAILY_CLOSE_FACT_SHA256
    ):
        raise ValueError("supplemental v3 expiry close-fact identity changed")
    close_marker = _read_json(close_marker_path)
    if (
        close_marker.get("complete") is not True
        or close_marker.get("schema_version") != "expiry_daily_close_facts_v1"
        or close_marker.get("marker_payload_sha256")
        != EXPIRY_CLOSE_FACT_MARKER_PAYLOAD_SHA256
        or int(close_marker.get("pair_count", -1)) != 45
        or int(close_marker.get("paired_close_coverage", -1)) != 45
        or close_marker.get("future_settlement_price_used") is not False
        or close_marker.get("upstream_complete_sha256")
        != UPSTREAM_SUPPLEMENTAL_V2_COMPLETE_SHA256
        or close_marker.get("upstream_marker_payload_sha256")
        != UPSTREAM_SUPPLEMENTAL_V2_MARKER_PAYLOAD_SHA256
        or str(close_marker.get("upstream_root")) != str(upstream_root)
    ):
        raise ValueError("expiry close-fact transitive contract changed")
    frame = pl.read_parquet(artifact_path)
    if frame.height != SUPPLEMENTAL_PATH_ROWS:
        raise ValueError("supplemental v3 path denominator changed")
    upstream_frame = pl.read_parquet(upstream_artifact_path)
    if frame.schema != upstream_frame.schema or upstream_frame.height != frame.height:
        raise ValueError("supplemental v3/v2 ordered schema or denominator changed")
    immutable = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "boundary_quantile",
        "entry_raw_order_fact_id",
        "entry_policy_generation_id",
        "policy_path_id",
        "position_established_ns",
        "normalization_notional_twd",
    ]
    left = base_paths.select(immutable).sort("policy_path_id")
    right = frame.select(immutable).sort("policy_path_id")
    if left.schema != right.schema or not left.equals(right, null_equal=True):
        raise ValueError("supplemental v3 immutable entry identity changed")
    expiry = frame.filter(
        pl.col("supplemental_terminal_resolution")
        == "expiry_same_day_two_leg_close_price"
    ).sort("policy_path_id")
    if expiry.height != 171 or expiry["policy_path_id"].n_unique() != 171:
        raise ValueError("supplemental v3 expiry path denominator changed")
    expiry_ids = expiry["policy_path_id"].to_list()
    v3_nonexpiry = frame.filter(~pl.col("policy_path_id").is_in(expiry_ids)).sort(
        "policy_path_id"
    )
    v2_nonexpiry = upstream_frame.filter(
        ~pl.col("policy_path_id").is_in(expiry_ids)
    ).sort("policy_path_id")
    if not v3_nonexpiry.equals(v2_nonexpiry, null_equal=True):
        raise ValueError("supplemental v3 changed a non-expiry upstream path")
    close_facts = pl.read_parquet(close_artifact_path)
    if (
        close_facts.height != 45
        or close_facts.select("Date", "ValueCode", "QuoteCode").unique().height != 45
    ):
        raise ValueError("expiry daily close-fact coverage changed")
    joined_expiry = expiry.join(
        close_facts.select(
            pl.col("Date").alias("terminal_date"),
            "ValueCode",
            "QuoteCode",
            "spot_close_price",
            "future_close_price",
            "future_settlement_price_used",
        ),
        on=["terminal_date", "ValueCode", "QuoteCode"],
        how="left",
        validate="m:1",
    )
    expiry_invalid = joined_expiry.filter(
        pl.any_horizontal(
            pl.col("spot_close_price", "future_close_price").is_null()
        )
        | (pl.col("exit_spot_price") != pl.col("spot_close_price"))
        | (pl.col("exit_future_price") != pl.col("future_close_price"))
        | (pl.col("future_settlement_price_used") != False).fill_null(True)  # noqa: E712
        | (
            pl.col("exit_decision_time_ns")
            != pl.col("terminal_date").map_elements(
                lambda date: aggressive_1300_start_cursor(str(date)).recv_time_ns
                + 30 * 60 * 1_000_000_000,
                return_dtype=pl.Int64,
            )
        )
        | (pl.col("terminal_reason")
           != "expiry_same_day_two_leg_close_price_forced_flat")
        | (pl.col("outcome_status")
           != "expiry_same_day_two_leg_close_price_forced_flat")
        | (pl.col("exact_price_source")
           != "MarketInfo.twse_security_trades_daily.close_price+"
              "MarketInfo.taifex_futures_trades_daily.close_price")
        | pl.col("expiry_uses_last_valid_session_mark")
        | pl.col("expiry_uses_last_observed_session_mark")
        | pl.col("expiry_mark_uses_trade_fallback")
        | (pl.col("spot_expiry_mark_is_executable_bbo") != False).fill_null(True)  # noqa: E712
        | (pl.col("future_expiry_mark_is_executable_bbo") != False).fill_null(True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_close") != True).fill_null(True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_settlement") != False).fill_null(True)  # noqa: E712
    )
    if expiry_invalid.height:
        raise ValueError(
            f"supplemental v3 expiry daily-close semantics changed: {expiry_invalid.height}"
        )
    v2_sorted = upstream_frame.sort("policy_path_id")
    v3_sorted = frame.sort("policy_path_id")
    gross_delta = float(
        (v3_sorted["gross_cycle_pnl_twd"] - v2_sorted["gross_cycle_pnl_twd"]).sum()
    )
    if not math.isclose(gross_delta, 293_700.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("supplemental v3 gross delta versus v2 changed")
    canonical = frame.with_columns(
        pl.lit("terminal").alias("outcome_type"),
        pl.col("supplemental_terminal_resolution").alias("outcome_status"),
        pl.when(pl.col("last_observed_session_date") < pl.col("terminal_date"))
        .then(pl.col("terminal_date"))
        .otherwise(pl.col("last_observed_session_date"))
        .alias("last_observed_session_date"),
    )
    invalid = canonical.filter(
        (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
        | pl.any_horizontal(
            pl.col(
                "terminal_date",
                "exit_decision_time_ns",
                "exit_spot_price",
                "exit_future_price",
                "gross_cycle_pnl_twd",
                "gross_cycle_bp",
            ).is_null()
        )
        | (pl.col("exit_decision_time_ns") < pl.col("position_established_ns"))
        | (pl.col("terminal_date") < pl.col("Date"))
        | (
            (
                pl.col("entry_contract_size_shares")
                * (
                    pl.col("exit_spot_price")
                    - pl.col("entry_spot_price")
                    + pl.col("entry_future_price")
                    - pl.col("exit_future_price")
                )
                - pl.col("gross_cycle_pnl_twd")
            ).abs()
            > 1e-6
        )
    )
    if invalid.height:
        raise ValueError(f"supplemental v3 terminal facts are incoherent: {invalid.height}")
    metadata = {
        "supplemental_root": str(source_root),
        "supplemental_complete_sha256": SUPPLEMENTAL_COMPLETE_SHA256,
        "supplemental_marker_payload_sha256": marker.get("marker_payload_sha256"),
        "supplemental_paths_sha256": SUPPLEMENTAL_PATH_SHA256,
        "supplemental_paths_rows": frame.height,
        "supplemental_paths_columns": frame.width,
        "supplemental_terminal_cashflow_priced_rows": canonical.filter(
            pl.col("terminal_cashflow_priced")
        ).height,
        "supplemental_controller_canonical_adapter": (
            "v3_terminal_fields_authoritative_stale_outcome_and_observation_labels_rewritten_v1"
        ),
        "supplemental_upstream_v2_root": str(upstream_root),
        "supplemental_upstream_v2_complete_sha256": (
            UPSTREAM_SUPPLEMENTAL_V2_COMPLETE_SHA256
        ),
        "supplemental_upstream_v2_paths_sha256": (
            UPSTREAM_SUPPLEMENTAL_V2_PATH_SHA256
        ),
        "supplemental_expiry_close_fact_root": str(close_root),
        "supplemental_expiry_close_fact_complete_sha256": (
            EXPIRY_CLOSE_FACT_COMPLETE_SHA256
        ),
        "supplemental_expiry_daily_close_fact_sha256": (
            EXPIRY_DAILY_CLOSE_FACT_SHA256
        ),
        "supplemental_expiry_daily_close_paths": expiry.height,
        "supplemental_expiry_daily_close_gross_delta_vs_v2_twd": gross_delta,
        "supplemental_expiry_mark_is_official_close": True,
        "supplemental_expiry_mark_is_official_settlement": False,
        "supplemental_expiry_mark_is_executable_bbo": False,
        "supplemental_state_sampling_approximate": True,
        "supplemental_target_cancel_clock_exact": False,
        "supplemental_delayed_hedge_snapshot_exact": False,
        "supplemental_model_imputed_full_carry_on_unknown": True,
        "supplemental_double_exit_bias_possible": True,
    }
    return canonical, metadata


def validate_normal_control_identity(
    controller_summary: pl.DataFrame,
    controller_daily: pl.DataFrame,
    *,
    root: Path = NORMAL_CONTROL_ROOT,
) -> dict[str, object]:
    """Bind and reconcile 10M/20M controls to the normal-only v3 sweep."""

    source_root = Path(root)
    if source_root.is_symlink() or source_root.resolve() != NORMAL_CONTROL_ROOT.resolve():
        raise ValueError("normal control root identity changed")
    source_root = source_root.resolve()
    marker_path = source_root / "complete.json"
    summary_path = source_root / "cap_summary.parquet"
    if (
        marker_path.is_symlink()
        or summary_path.is_symlink()
        or _file_sha256(marker_path) != NORMAL_CONTROL_COMPLETE_SHA256
        or _file_sha256(summary_path) != NORMAL_CONTROL_CAP_SUMMARY_SHA256
    ):
        raise ValueError("normal control artifact identity changed")
    marker = _read_json(marker_path)
    declaration = marker.get("artifacts", {}).get("cap_summary.parquet")
    if (
        marker.get("complete") is not True
        or marker.get("analysis_only") is not True
        or marker.get("production_strategy_go") is not False
        or marker.get("schema_version")
        != "normal_carry_cap_sweep_v2_expiry_close_opening_carry_exit_only"
        or marker.get("opening_carry_product_exit_only_policy") is not True
        or marker.get("opening_carry_exit_only_is_distinct_from_d_safe_universe_gate")
        is not True
        or float(marker.get("per_product_cap_fraction", math.nan)) != 0.30
        or int(marker.get("session_count", -1)) != FORMAL_SESSION_CALENDAR_COUNT
        or int(marker.get("priced_paths", -1)) != SUPPLEMENTAL_PATH_ROWS
        or int(marker.get("unresolved_paths", -1)) != 0
        or str(marker.get("upstream_root")) != str(DEFAULT_SUPPLEMENTAL_ROOT.resolve())
        or marker.get("upstream_complete_sha256") != SUPPLEMENTAL_COMPLETE_SHA256
        or marker.get("upstream_marker_payload_sha256")
        != SUPPLEMENTAL_MARKER_PAYLOAD_SHA256
        or not isinstance(declaration, dict)
        or declaration.get("sha256") != NORMAL_CONTROL_CAP_SUMMARY_SHA256
        or int(declaration.get("rows", -1)) != 10
        or summary_path.stat().st_size != int(declaration.get("bytes", -1))
    ):
        raise ValueError("normal control source contract changed")
    normal = pl.read_parquet(summary_path).filter(
        pl.col("scenario_id").str.starts_with("normal_cutoff_1300_hard_")
        & pl.col("hard_intraday_cap_twd").is_in([10_000_000.0, 20_000_000.0])
    ).sort("hard_intraday_cap_twd")
    controls = controller_summary.filter(
        pl.col("primary_conservative_result")
        & pl.col("portfolio_cap_twd").is_in([10_000_000.0, 20_000_000.0])
    ).sort("portfolio_cap_twd")
    if normal.height != 2 or controls.height != 2:
        raise ValueError("10M/20M control denominator changed")
    normal_by_cap = {
        float(item["hard_intraday_cap_twd"]): item
        for item in normal.iter_rows(named=True)
    }
    for item in controls.iter_rows(named=True):
        cap = float(item["portfolio_cap_twd"])
        expected = normal_by_cap[cap]
        integer_pairs = {
            "candidate_entries": "candidate_paths",
            "accepted_entries": "accepted_paths",
            "held_product_exit_only_blocked_entries": (
                "rejected_opening_carry_exit_only"
            ),
            "forbidden_new_entries_at_or_after_1300": "rejected_entry_cutoff",
            "portfolio_only_blocked_entries": "rejected_portfolio_cap",
            "product_only_blocked_entries": "rejected_product_cap",
            "both_caps_blocked_entries": "rejected_both_caps",
            "completed_scenario_positions": "completed_exits",
            "net_profitable_positions": "winning_exits",
            "net_loss_positions": "losing_exits",
            "net_zero_positions": "flat_exits",
            "negative_realized_days": "negative_realized_days",
        }
        float_pairs = {
            "completed_scenario_notional_twd": "accepted_entry_one_way_turnover_twd",
            "completed_scenario_gross_pnl_twd": "realized_gross_pnl_twd",
            "completed_scenario_transaction_cost_twd": (
                "realized_transaction_cost_twd"
            ),
            "completed_scenario_net_pnl_twd": "realized_net_pnl_twd",
            "completed_scenario_weighted_gross_bp": (
                "realized_gross_bp_on_entry_turnover"
            ),
            "completed_scenario_weighted_cost_bp": (
                "realized_cost_bp_on_entry_turnover"
            ),
            "completed_scenario_weighted_net_bp": (
                "realized_net_bp_on_entry_turnover"
            ),
            "realized_net_pnl_max_drawdown_twd": "max_realized_drawdown_twd",
        }
        for actual_name, expected_name in integer_pairs.items():
            if int(item[actual_name]) != int(expected[expected_name]):
                raise ValueError(
                    f"aggressive identity control differs from normal v3: {cap}/{actual_name}"
                )
        for actual_name, expected_name in float_pairs.items():
            if not math.isclose(
                float(item[actual_name]),
                float(expected[expected_name]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"aggressive identity control differs from normal v3: {cap}/{actual_name}"
                )
        if (
            int(
                controller_daily.filter(
                    (pl.col("scenario_id") == item["scenario_id"])
                ).height
            )
            != int(expected["session_count"])
        ):
            raise ValueError("aggressive identity control session grid changed")
    return {
        "normal_control_root": str(source_root),
        "normal_control_complete_sha256": NORMAL_CONTROL_COMPLETE_SHA256,
        "normal_control_cap_summary_sha256": NORMAL_CONTROL_CAP_SUMMARY_SHA256,
        "normal_control_identity_caps_twd": [10_000_000.0, 20_000_000.0],
        "normal_control_identity_metrics_verified": sorted(
            [*integer_pairs, *float_pairs]
        ),
        "normal_control_identity_verified": True,
    }


def cache_index(cache_root: Path) -> dict[tuple[str, str, str], Path]:
    """Index exact cache partitions, rejecting duplicate or symlinked roots."""

    root = Path(cache_root).resolve()
    result: dict[tuple[str, str, str], Path] = {}
    for marker in root.glob("Date=*/ValueCode=*/QuoteCode=*/Key=*/complete.json"):
        if marker.is_symlink() or any(
            parent.is_symlink() for parent in marker.parents if parent != root.parent
        ):
            raise ValueError(f"cache source contains a symlink: {marker}")
        payload = _read_json(marker)
        key = (
            str(payload.get("Date")),
            str(payload.get("ValueCode")),
            str(payload.get("QuoteCode")),
        )
        if key in result:
            raise ValueError(f"cache product-day is duplicated: {key}")
        result[key] = marker.parent
    if not result:
        raise ValueError("candidate-session cache is empty")
    return result


def validate_cache_sources(
    cache_root: Path,
    required_product_days: pl.DataFrame,
    *,
    rehash_artifacts: bool = True,
) -> tuple[dict[tuple[str, str, str], Path], pl.DataFrame]:
    """Validate cache marker/config and exact seven artifact hashes."""

    keys = ["Date", "ValueCode", "QuoteCode"]
    if required_product_days.select(keys).unique().height != required_product_days.height:
        raise ValueError("required product-day cache keys are duplicated")
    index = cache_index(cache_root)
    rows: list[dict[str, object]] = []
    selected: dict[tuple[str, str, str], Path] = {}
    for item in required_product_days.sort(keys).iter_rows(named=True):
        key = tuple(str(item[name]) for name in keys)
        root = index.get(key)
        if root is None:
            continue
        marker_path = root / "complete.json"
        marker = _read_json(marker_path)
        config = marker.get("config")
        artifacts = marker.get("artifacts")
        if (
            marker.get("complete") is not True
            or marker.get("cache_status") != "success"
            or marker.get("schema_version") != "cross_session_candidate_cache_v1"
            or not isinstance(config, dict)
            or config.get("schema_version") != "cross_session_candidate_cache_v1"
            or config.get("candidate_reference_policy")
            != "candidate_session_marketData_spot_plus_exact_futures_metadata_v1"
            or config.get("candidate_spot_day_trade_policy")
            != "existing_long_spot_exit_all_marks_X_Y_N_no_entry_filter_v1"
            or config.get("implementation_sources") != EXPECTED_CACHE_IMPLEMENTATION
            or config.get("source_fingerprint", {}).get("content_integrity_bound")
            is not False
            or marker.get("failure") is not None
            or not isinstance(artifacts, dict)
            or tuple(sorted(artifacts)) != tuple(sorted(EXPECTED_CACHE_ARTIFACTS))
        ):
            raise ValueError(f"candidate cache marker contract changed: {marker_path}")
        cache_key = str(marker.get("cache_key_sha256"))
        if (
            cache_key != str(marker.get("config_sha256"))
            or root.name != f"Key={cache_key}"
            or tuple(str(marker.get(name)) for name in keys) != key
        ):
            raise ValueError(f"candidate cache identity changed: {marker_path}")
        fingerprint_json = json.dumps(
            config["source_fingerprint"], sort_keys=True, separators=(",", ":")
        )
        marker_sha = _file_sha256(marker_path)
        for filename in EXPECTED_CACHE_ARTIFACTS:
            declaration = artifacts[filename]
            path = root / filename
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(declaration.get("bytes", -1))
                or (
                    rehash_artifacts
                    and _file_sha256(path) != declaration.get("sha256")
                )
            ):
                raise ValueError(f"candidate cache artifact changed: {path}")
            schema = pl.read_parquet_schema(path)
            if (
                pl.scan_parquet(path).select(pl.len()).collect().item()
                != int(declaration.get("rows", -1))
                or len(schema) != int(declaration.get("columns", -1))
            ):
                raise ValueError(f"candidate cache artifact shape changed: {path}")
            rows.append(
                {
                    "Date": key[0],
                    "ValueCode": key[1],
                    "QuoteCode": key[2],
                    "cache_key_sha256": cache_key,
                    "cache_complete_path": str(marker_path.resolve()),
                    "cache_complete_sha256": marker_sha,
                    "artifact_name": filename,
                    "artifact_path": str(path.resolve()),
                    "artifact_sha256": str(declaration["sha256"]),
                    "artifact_bytes": int(declaration["bytes"]),
                    "artifact_rows": int(declaration["rows"]),
                    "artifact_columns": int(declaration["columns"]),
                    "raw_source_fingerprint_json": fingerprint_json,
                    "raw_source_fingerprint_sha256": _sha256_text(fingerprint_json),
                    "cache_artifact_content_hash_verified": rehash_artifacts,
                    "upstream_raw_content_integrity_bound": False,
                }
            )
        selected[key] = root
    inventory = pl.from_dicts(rows, schema=_cache_inventory_schema(), strict=True).sort(
        [*keys, "artifact_name"]
    ) if rows else pl.DataFrame(schema=_cache_inventory_schema())
    return selected, inventory


def _cache_inventory_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "cache_key_sha256": pl.String,
        "cache_complete_path": pl.String,
        "cache_complete_sha256": pl.String,
        "artifact_name": pl.String,
        "artifact_path": pl.String,
        "artifact_sha256": pl.String,
        "artifact_bytes": pl.Int64,
        "artifact_rows": pl.Int64,
        "artifact_columns": pl.Int64,
        "raw_source_fingerprint_json": pl.String,
        "raw_source_fingerprint_sha256": pl.String,
        "cache_artifact_content_hash_verified": pl.Boolean,
        "upstream_raw_content_integrity_bound": pl.Boolean,
    }


def _entry_price_source_inventory_schema() -> dict[str, pl.DataType]:
    return {
        "source_kind": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "partition_complete_path": pl.String,
        "partition_complete_sha256": pl.String,
        "artifact_path": pl.String,
        "artifact_sha256": pl.String,
        "artifact_bytes": pl.Int64,
        "artifact_rows": pl.Int64,
        "artifact_columns": pl.Int64,
    }


def load_seed_templates(path: Path = DEFAULT_SEED_TEMPLATE) -> pl.DataFrame:
    source = Path(path)
    marker_path = source.parent / "complete.json"
    if (
        not source.is_file()
        or source.is_symlink()
        or marker_path.is_symlink()
        or _file_sha256(marker_path) != SEED_SOURCE_COMPLETE_SHA256
        or _file_sha256(source) != SEED_TEMPLATE_SHA256
    ):
        raise ValueError("aggressive 13:00 seed template identity changed")
    marker = _read_json(marker_path)
    unhashed = dict(marker)
    marker_payload_sha = unhashed.pop("marker_payload_sha256", None)
    declaration = marker.get("artifacts", {}).get(source.name)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != "aggressive_1300_seed_template_source_v1"
        or marker.get("analysis_only") is not True
        or marker.get("generator_sha256") != SEED_GENERATOR_SHA256
        or marker_payload_sha != _canonical_sha256(unhashed)
        or not isinstance(declaration, dict)
        or declaration.get("sha256") != SEED_TEMPLATE_SHA256
        or int(declaration.get("rows", -1)) != SEED_TEMPLATE_ROWS
        or int(declaration.get("bytes", -1)) != source.stat().st_size
    ):
        raise ValueError("aggressive seed source marker contract changed")
    frame = pl.read_parquet(source)
    if frame.height != SEED_TEMPLATE_ROWS:
        raise ValueError("aggressive seed template row count changed")
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "position_id",
        "policy_version",
        "scheduled_start_recv_time_ns",
        "actual_first_submit_recv_time_ns",
        "entry_after_1300_allowed",
        "dynamic_b1a1_peg",
        "future_candidate_generations",
        "spot_candidate_generations",
        "winner_generation_id",
        "winner_route",
        "winner_full_fill_recv_time_ns",
        "winner_hedge_delay_ns",
        "winner_hedge_status",
        "active_sibling_cancel_count",
        "prior_unacked_cancel_count_before_winner",
        "sibling_full_fill_after_cancel_request_count",
        "sibling_partial_before_winner_count",
        "oco_position_projection_safe",
        "nominal_branch_status",
        "strict_branch_status",
        "nominal_close_count",
        "duplicate_close_prevented",
        "cancel_ack_observed",
        "cancel_model",
        "joint_volume_allocated",
        "strict_ev_ready",
        "winner_maker_price",
        "winner_taker_vwap_price",
        "exit_decision_time_ns",
        "exit_spot_price",
        "exit_future_price",
        "replay_error",
        "positions",
    }
    if set(frame.columns) != required:
        raise ValueError("aggressive seed template schema changed")
    if frame["replay_error"].null_count() != frame.height:
        raise ValueError("aggressive seed contains replay errors")
    records = [
        _normalize_template_record(item, template_source="audited_seed_template_v1")
        for item in frame.iter_rows(named=True)
    ]
    normalized = pl.from_dicts(
        records, schema=_market_template_schema(), strict=True
    ).sort(["Date", "ValueCode"])
    validate_market_templates(normalized)
    return normalized


def validate_market_templates(templates: pl.DataFrame) -> None:
    schema = _market_template_schema()
    if templates.columns != list(schema) or templates.schema != pl.Schema(schema):
        raise ValueError("market template ordered schema/dtypes are not canonical")
    if templates.is_empty():
        raise ValueError("market templates cannot be empty")
    keys = ["Date", "ValueCode", "QuoteCode"]
    if templates.select(keys).unique().height != templates.height:
        raise ValueError("market templates duplicate a product-day")
    starts = {
        date: aggressive_1300_start_cursor(date).recv_time_ns
        for date in templates["Date"].unique().to_list()
    }
    cutoffs = {
        date: session_cutoff_cursor(date).recv_time_ns
        for date in templates["Date"].unique().to_list()
    }
    invalid = templates.with_columns(
        pl.col("Date").replace_strict(starts, return_dtype=pl.Int64).alias("_start"),
        pl.col("Date").replace_strict(cutoffs, return_dtype=pl.Int64).alias("_cutoff"),
    ).filter(
        (pl.col("policy_version") != AGGRESSIVE_1300_POLICY_VERSION)
        | (pl.col("scheduled_start_recv_time_ns") != pl.col("_start"))
        | (pl.col("entry_after_1300_allowed") != False).fill_null(True)  # noqa: E712
        | (pl.col("dynamic_b1a1_peg") != True).fill_null(True)  # noqa: E712
        | (pl.col("duplicate_close_prevented") != True).fill_null(True)  # noqa: E712
        | (pl.col("cancel_ack_observed") != False).fill_null(True)  # noqa: E712
        | (pl.col("cancel_model") != "nominal_instant_cancel_v0")
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
        | (pl.col("strict_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("nominal_close_count") < 0)
        | (pl.col("nominal_close_count") > 1)
        | (
            pl.col("nominal_close_count")
            != (pl.col("nominal_branch_status") == "flat_same_day").cast(pl.Int64)
        )
        | pl.col("actual_first_submit_recv_time_ns").is_not_null()
        & (pl.col("actual_first_submit_recv_time_ns") < pl.col("_start"))
        | pl.col("actual_first_submit_recv_time_ns").is_not_null()
        & (pl.col("actual_first_submit_recv_time_ns") > pl.col("_cutoff"))
        | pl.col("winner_full_fill_recv_time_ns").is_not_null()
        & pl.col("actual_first_submit_recv_time_ns").is_null()
        | pl.col("winner_full_fill_recv_time_ns").is_not_null()
        & (
            pl.col("winner_full_fill_recv_time_ns")
            < pl.col("actual_first_submit_recv_time_ns")
        )
        | pl.col("winner_full_fill_recv_time_ns").is_not_null()
        & (pl.col("winner_full_fill_recv_time_ns") > pl.col("_cutoff"))
        | pl.col("exit_decision_time_ns").is_not_null()
        & (pl.col("exit_decision_time_ns") > pl.col("_cutoff"))
        | pl.col("winner_hedge_delay_ns").is_not_null()
        & (pl.col("winner_hedge_delay_ns") != 50_000_000)
        | pl.col("exit_decision_time_ns").is_not_null()
        & (
            pl.col("exit_decision_time_ns")
            != pl.col("winner_full_fill_recv_time_ns") + 50_000_000
        )
    )
    if invalid.height:
        raise ValueError(f"market template lifecycle invariant failed: {invalid.height}")
    flat = templates.filter(pl.col("nominal_branch_status") == "flat_same_day")
    if flat.filter(
        pl.any_horizontal(
            pl.col(
                "winner_route",
                "winner_full_fill_recv_time_ns",
                "winner_hedge_delay_ns",
                "winner_hedge_status",
                "winner_maker_price",
                "winner_taker_vwap_price",
                "exit_decision_time_ns",
                "exit_spot_price",
                "exit_future_price",
            ).is_null()
        )
        | (pl.col("winner_hedge_status") != "executable")
        | (pl.col("oco_position_projection_safe") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("flat market template lacks a coherent executable winner")
    for item in templates.filter(pl.col("winner_route").is_not_null()).iter_rows(
        named=True
    ):
        route = str(item["winner_route"])
        if route not in EXIT_MAKER_ROUTE_CONTRACTS:
            raise ValueError(f"unknown market template winner route: {route}")
        maker_market = EXIT_MAKER_ROUTE_CONTRACTS[route].maker_market
        price = item["winner_maker_price"]
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError("market template maker price is invalid")
        tick = absolute_price_tick(
            float(price), market=maker_market, session_date=str(item["Date"])
        )
        if tick_index_to_price(
            tick, market=maker_market, session_date=str(item["Date"])
        ) != float(price):
            raise ValueError("market template maker price is off the legal ladder")
        if route == "future_bid_spot_taker":
            pair = (item["exit_future_price"], item["exit_spot_price"])
        else:
            pair = (item["exit_spot_price"], item["exit_future_price"])
        if pair != (item["winner_maker_price"], item["winner_taker_vwap_price"]):
            raise ValueError("market template exit-leg price mapping changed")


def replay_missing_cached_templates(
    required_product_days: pl.DataFrame,
    existing_templates: pl.DataFrame,
    cache_roots: Mapping[tuple[str, str, str], Path],
) -> pl.DataFrame:
    """Replay required cache-backed product-days not present in the seed."""

    keys = ["Date", "ValueCode", "QuoteCode"]
    missing = required_product_days.select(keys).join(
        existing_templates.select(keys), on=keys, how="anti"
    )
    records: list[dict[str, object]] = []
    for item in missing.sort(keys).iter_rows(named=True):
        key = tuple(str(item[name]) for name in keys)
        root = cache_roots.get(key)
        if root is None:
            continue
        result = _replay_cache_product_day(root, key[0], key[1])
        record = result.summary_record()
        winner = result.winner_outcome
        if winner is not None:
            window = next(
                window
                for build in result.builds_by_route.values()
                for window in build.windows
                if window.generation_id == winner.generation_id
            )
            attempt = winner.hedge_attempts[0] if winner.hedge_attempts else None
            execution = None if attempt is None else attempt.execution
            maker_market = EXIT_MAKER_ROUTE_CONTRACTS[winner.route].maker_market
            maker_price = tick_index_to_price(
                window.target_price_tick,
                market=maker_market,
                session_date=key[0],
            )
            taker_price = None if execution is None else execution.executable_vwap_price
            decision_time = None if attempt is None else attempt.decision_time_ns
            if winner.route == "future_bid_spot_taker":
                exit_future, exit_spot = maker_price, taker_price
            else:
                exit_spot, exit_future = maker_price, taker_price
        else:
            maker_price = taker_price = decision_time = None
            exit_spot = exit_future = None
        record.update(
            {
                "winner_maker_price": maker_price,
                "winner_taker_vwap_price": taker_price,
                "exit_decision_time_ns": decision_time,
                "exit_spot_price": exit_spot,
                "exit_future_price": exit_future,
            }
        )
        records.append(
            _normalize_template_record(
                record, template_source="cache_replay_for_carry_inventory_v1"
            )
        )
        del result
        gc.collect()
    if not records:
        return pl.DataFrame(schema=_market_template_schema())
    result = pl.from_dicts(
        records, schema=_market_template_schema(), strict=True
    ).sort(keys)
    validate_market_templates(result)
    return result


def _replay_cache_product_day(root: Path, date: str, value_code: str):
    start = aggressive_1300_start_cursor(date).recv_time_ns

    def trim(frame: pl.DataFrame) -> pl.DataFrame:
        before = frame.filter(pl.col("recv_time_ns") <= start).tail(1)
        after = frame.filter(pl.col("recv_time_ns") > start)
        return pl.concat([before, after], how="vertical_relaxed")

    spot_states = trim(pl.read_parquet(root / "spot_states.parquet"))
    future_states = trim(pl.read_parquet(root / "future_states.parquet"))
    spot_trades = pl.read_parquet(root / "spot_trades.parquet").filter(
        pl.col("recv_time_ns") > start
    )
    future_trades = pl.read_parquet(root / "future_trades.parquet").filter(
        pl.col("recv_time_ns") > start
    )
    clock = pl.read_parquet(root / "spread_pair_clock.parquet").join(
        spot_states.select(pl.col("sequence").alias("spot_channel_seq")),
        on="spot_channel_seq",
        how="semi",
    )
    raw = RawTapeDay(
        date=date,
        mapping=pl.read_parquet(root / "mapping.parquet"),
        spot_states=spot_states,
        future_states=future_states,
        spot_trades=spot_trades,
        future_trades=future_trades,
        audit=pl.read_parquet(root / "raw_audit.parquet"),
    )
    return replay_aggressive_1300_product_day(
        raw,
        clock,
        position_id=f"market-template/{date}/{value_code}",
        position_established_recv_time_ns=0,
        position_open_at_start=True,
        value_code=value_code,
    )


def _normalize_template_record(
    item: Mapping[str, object], *, template_source: str
) -> dict[str, object]:
    date = str(item["Date"])
    value = str(item["ValueCode"])
    quote = str(item["QuoteCode"])
    generation = item.get("winner_generation_id")
    if generation is None:
        generation_suffix = None
    else:
        text = str(generation)
        route_token = "/aggressive_1300_dynamic_b1a1_oco_v1/"
        offset = text.find(route_token)
        if offset < 0:
            raise ValueError("winner generation id lacks the policy suffix")
        generation_suffix = text[offset + 1 :]
    return {
        "Date": date,
        "ValueCode": value,
        "QuoteCode": quote,
        "market_template_id": f"{ANALYSIS_VERSION}/{date}/{value}/{quote}",
        "template_source": template_source,
        "policy_version": str(item["policy_version"]),
        "scheduled_start_recv_time_ns": int(item["scheduled_start_recv_time_ns"]),
        "actual_first_submit_recv_time_ns": _optional_int(
            item.get("actual_first_submit_recv_time_ns")
        ),
        "entry_after_1300_allowed": bool(item["entry_after_1300_allowed"]),
        "dynamic_b1a1_peg": bool(item["dynamic_b1a1_peg"]),
        "future_candidate_generations": int(item["future_candidate_generations"]),
        "spot_candidate_generations": int(item["spot_candidate_generations"]),
        "winner_generation_suffix": generation_suffix,
        "winner_route": _optional_str(item.get("winner_route")),
        "winner_full_fill_recv_time_ns": _optional_int(
            item.get("winner_full_fill_recv_time_ns")
        ),
        "winner_hedge_delay_ns": _optional_int(item.get("winner_hedge_delay_ns")),
        "winner_hedge_status": _optional_str(item.get("winner_hedge_status")),
        "active_sibling_cancel_count": int(item["active_sibling_cancel_count"]),
        "prior_unacked_cancel_count_before_winner": int(
            item["prior_unacked_cancel_count_before_winner"]
        ),
        "sibling_full_fill_after_cancel_request_count": int(
            item["sibling_full_fill_after_cancel_request_count"]
        ),
        "sibling_partial_before_winner_count": int(
            item["sibling_partial_before_winner_count"]
        ),
        "oco_position_projection_safe": bool(item["oco_position_projection_safe"]),
        "nominal_branch_status": str(item["nominal_branch_status"]),
        "strict_branch_status": str(item["strict_branch_status"]),
        "nominal_close_count": int(item["nominal_close_count"]),
        "duplicate_close_prevented": bool(item["duplicate_close_prevented"]),
        "cancel_ack_observed": bool(item["cancel_ack_observed"]),
        "cancel_model": str(item["cancel_model"]),
        "joint_volume_allocated": bool(item["joint_volume_allocated"]),
        "strict_ev_ready": bool(item["strict_ev_ready"]),
        "winner_maker_price": _optional_float(item.get("winner_maker_price")),
        "winner_taker_vwap_price": _optional_float(
            item.get("winner_taker_vwap_price")
        ),
        "exit_decision_time_ns": _optional_int(item.get("exit_decision_time_ns")),
        "exit_spot_price": _optional_float(item.get("exit_spot_price")),
        "exit_future_price": _optional_float(item.get("exit_future_price")),
        "market_template_is_single_position_counterfactual": True,
        "market_template_safe_to_multiply_by_positions": False,
        "upstream_raw_content_integrity_bound": False,
        "formal_ev_ready": False,
    }


def _market_template_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "market_template_id": pl.String,
        "template_source": pl.String,
        "policy_version": pl.String,
        "scheduled_start_recv_time_ns": pl.Int64,
        "actual_first_submit_recv_time_ns": pl.Int64,
        "entry_after_1300_allowed": pl.Boolean,
        "dynamic_b1a1_peg": pl.Boolean,
        "future_candidate_generations": pl.Int64,
        "spot_candidate_generations": pl.Int64,
        "winner_generation_suffix": pl.String,
        "winner_route": pl.String,
        "winner_full_fill_recv_time_ns": pl.Int64,
        "winner_hedge_delay_ns": pl.Int64,
        "winner_hedge_status": pl.String,
        "active_sibling_cancel_count": pl.Int64,
        "prior_unacked_cancel_count_before_winner": pl.Int64,
        "sibling_full_fill_after_cancel_request_count": pl.Int64,
        "sibling_partial_before_winner_count": pl.Int64,
        "oco_position_projection_safe": pl.Boolean,
        "nominal_branch_status": pl.String,
        "strict_branch_status": pl.String,
        "nominal_close_count": pl.Int64,
        "duplicate_close_prevented": pl.Boolean,
        "cancel_ack_observed": pl.Boolean,
        "cancel_model": pl.String,
        "joint_volume_allocated": pl.Boolean,
        "strict_ev_ready": pl.Boolean,
        "winner_maker_price": pl.Float64,
        "winner_taker_vwap_price": pl.Float64,
        "exit_decision_time_ns": pl.Int64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "market_template_is_single_position_counterfactual": pl.Boolean,
        "market_template_safe_to_multiply_by_positions": pl.Boolean,
        "upstream_raw_content_integrity_bound": pl.Boolean,
        "formal_ev_ready": pl.Boolean,
    }


def build_analysis_result(
    selected_paths_with_entry_prices: pl.DataFrame,
    market_templates: pl.DataFrame,
    cache_source_inventory: pl.DataFrame,
    entry_price_source_inventory: pl.DataFrame,
    inventory_audit: pl.DataFrame,
    *,
    session_dates: Sequence[str] | None = None,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> Aggressive1300AnalysisResult:
    config.validate()
    validate_market_templates(market_templates)
    dates = (
        _session_calendar_from_paths(selected_paths_with_entry_prices)
        if session_dates is None
        else _validate_session_dates(session_dates)
    )
    daily, rebuilt_audit = build_daily_open_inventory(
        selected_paths_with_entry_prices, session_dates=dates
    )
    if not rebuilt_audit.equals(inventory_audit, null_equal=True):
        raise ValueError("daily inventory audit differs on rebuild")
    overlay = build_position_template_overlay(daily, market_templates)
    keys = ["Date", "ValueCode", "QuoteCode"]
    required = daily.group_by(keys).agg(pl.len().alias("open_positions"))
    missing = required.join(market_templates.select(keys), on=keys, how="anti").with_columns(
        pl.lit("candidate_session_cache_unavailable").alias("missing_reason"),
        pl.lit(False).alias("market_template_available"),
        pl.lit(False).alias("direct_raw_replay_performed"),
        pl.lit(False).alias("formal_ev_ready"),
    ).sort(keys)
    covered = overlay.filter(pl.col("market_template_available"))
    branch = (
        covered.group_by(
            "template_source",
            "nominal_branch_status",
            "strict_branch_status",
            "winner_route",
        )
        .agg(
            pl.col("market_template_id").n_unique().alias("product_days"),
            pl.len().alias("independent_position_labels"),
            pl.col("controller_nominal_close_eligible").sum().cast(pl.Int64).alias(
                "nominal_close_eligible_position_labels"
            ),
            pl.col("normalization_notional_twd").sum().alias(
                "independent_position_label_notional_twd"
            ),
        )
        .with_columns(
            pl.lit(False).alias("position_labels_safe_to_sum"),
            pl.lit(False).alias("joint_volume_allocated"),
            pl.lit(False).alias("formal_ev_ready"),
        )
        .sort(
            [
                "template_source",
                "nominal_branch_status",
                "strict_branch_status",
                "winner_route",
            ],
            nulls_last=True,
        )
    )
    inventory_row = inventory_audit.row(0, named=True)
    coverage = pl.from_dicts(
        [
            {
                **inventory_row,
                "required_product_days": required.height,
                "covered_product_days": market_templates.height,
                "missing_product_days": missing.height,
                "covered_position_rows": covered.height,
                "missing_position_rows": overlay.filter(
                    ~pl.col("market_template_available")
                ).height,
                "seed_template_product_days": market_templates.filter(
                    pl.col("template_source") == "audited_seed_template_v1"
                ).height,
                "carry_cache_replay_product_days": market_templates.filter(
                    pl.col("template_source")
                    == "cache_replay_for_carry_inventory_v1"
                ).height,
                "nominal_flat_template_product_days": market_templates.filter(
                    pl.col("nominal_branch_status") == "flat_same_day"
                ).height,
                "nominal_flat_independent_position_labels": covered.filter(
                    pl.col("controller_nominal_close_eligible")
                ).height,
                "formal_session_calendar_count": len(dates),
                "formal_session_calendar_sha256": _session_calendar_sha256(dates),
                "normal_terminal_priced_source_paths": selected_paths_with_entry_prices.filter(
                    pl.col("terminal_cashflow_priced")
                ).height,
                "normal_terminal_unpriced_source_paths": selected_paths_with_entry_prices.filter(
                    ~pl.col("terminal_cashflow_priced")
                ).height,
                "target_measurement_time": "13:20 Asia/Taipei study cutoff",
                "study_cutoff_is_twse_close": False,
                "extra_ten_minutes_to_twse_close_evaluated": False,
                "seed_universe_was_same_origin_day_only": True,
                "daily_carry_inventory_rebuilt": True,
                "held_product_exit_only_gate_applied": True,
                "runtime_daily_liquidity_admission_gate_applied": False,
                "cache_artifact_hashes_verified": bool(
                    cache_source_inventory.height
                    and cache_source_inventory["cache_artifact_content_hash_verified"].all()
                ),
                "upstream_raw_content_integrity_bound": False,
                "joint_volume_allocated": False,
                "all_close_pnl_published": False,
                "actual_notional_target_attainment_published": True,
                "transaction_cost_profile_id": TransactionCostProfile().profile_id,
                "transaction_cost_results_published": True,
                "realized_pnl_only_no_mtm": True,
                "universe_selection_d_safe_go": False,
                "formal_ev_ready": False,
                "production_strategy_go": False,
            }
        ],
        infer_schema_length=None,
    )
    controller_positions, controller_daily, controller_summary = (
        build_hard_cap_controller_results(
            selected_paths_with_entry_prices,
            market_templates,
            session_dates=dates,
            config=config,
        )
    )
    result = Aggressive1300AnalysisResult(
        market_templates=market_templates,
        daily_open_inventory=daily,
        position_template_overlay=overlay,
        missing_product_days=missing,
        cache_source_inventory=cache_source_inventory,
        entry_price_source_inventory=entry_price_source_inventory,
        template_branch_summary=branch,
        coverage_summary=coverage,
        controller_contract=build_controller_contract(config),
        controller_position_outcomes=controller_positions,
        controller_daily_control=controller_daily,
        controller_scenario_summary=controller_summary,
    )
    validate_analysis_result(result, config=config)
    return result


def validate_analysis_result(
    result: Aggressive1300AnalysisResult,
    *,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> None:
    config.validate()
    validate_market_templates(result.market_templates)
    if (
        result.coverage_summary.height != 1
        or result.controller_contract.height
        != len(config.portfolio_caps_twd) * len(_CAPACITY_ASSUMPTIONS)
    ):
        raise ValueError("analysis summary/controller scenario denominator changed")
    coverage = result.coverage_summary.row(0, named=True)
    first_scenario_id = result.controller_position_outcomes.item(0, "scenario_id")
    first_scenario_positions = result.controller_position_outcomes.filter(
        pl.col("scenario_id") == first_scenario_id
    )
    calendar_dates = (
        result.controller_daily_control.filter(pl.col("scenario_id") == first_scenario_id)[
            "Date"
        ]
        .unique()
        .sort()
        .to_list()
    )
    expected = {
        "daily_open_position_rows": result.daily_open_inventory.height,
        "required_product_days": result.daily_open_inventory.select(
            "Date", "ValueCode", "QuoteCode"
        ).unique().height,
        "covered_product_days": result.market_templates.height,
        "missing_product_days": result.missing_product_days.height,
        "covered_position_rows": result.position_template_overlay.filter(
            pl.col("market_template_available")
        ).height,
        "missing_position_rows": result.position_template_overlay.filter(
            ~pl.col("market_template_available")
        ).height,
        "formal_session_calendar_count": len(calendar_dates),
        "normal_terminal_priced_source_paths": first_scenario_positions.filter(
            pl.col("normal_terminal_cashflow_priced")
        ).height,
        "normal_terminal_unpriced_source_paths": first_scenario_positions.filter(
            ~pl.col("normal_terminal_cashflow_priced")
        ).height,
    }
    for name, value in expected.items():
        if int(coverage[name]) != value:
            raise ValueError(f"coverage summary {name} is not reproducible")
    if (
        int(coverage["required_product_days"])
        != int(coverage["covered_product_days"])
        + int(coverage["missing_product_days"])
        or int(coverage["daily_open_position_rows"])
        != int(coverage["covered_position_rows"])
        + int(coverage["missing_position_rows"])
    ):
        raise ValueError("coverage denominator does not partition")
    if (
        coverage["formal_session_calendar_sha256"]
        != _session_calendar_sha256(calendar_dates)
        or coverage["target_measurement_time"]
        != "13:20 Asia/Taipei study cutoff"
        or coverage["study_cutoff_is_twse_close"] is not False
        or coverage["extra_ten_minutes_to_twse_close_evaluated"] is not False
        or coverage["held_product_exit_only_gate_applied"] is not True
        or coverage["runtime_daily_liquidity_admission_gate_applied"] is not False
        or coverage["transaction_cost_profile_id"]
        != TransactionCostProfile().profile_id
        or coverage["transaction_cost_results_published"] is not True
        or coverage["realized_pnl_only_no_mtm"] is not True
    ):
        raise ValueError("analysis coverage calendar/controller/cost contract changed")
    false_flags = (
        "upstream_raw_content_integrity_bound",
        "joint_volume_allocated",
        "all_close_pnl_published",
        "universe_selection_d_safe_go",
        "formal_ev_ready",
        "production_strategy_go",
    )
    if any(coverage[name] is not False for name in false_flags):
        raise ValueError("analysis coverage safety flags changed")
    if coverage["actual_notional_target_attainment_published"] is not True:
        raise ValueError("controller target-attainment publication flag changed")
    expected_contract = build_controller_contract(config)
    if (
        result.controller_contract.schema != expected_contract.schema
        or not result.controller_contract.equals(expected_contract, null_equal=True)
    ):
        raise ValueError("controller contract changed")
    if result.position_template_overlay.height != result.daily_open_inventory.height:
        raise ValueError("position template overlay denominator changed")
    if result.position_template_overlay.filter(
        pl.col("joint_volume_allocated_for_positions")
        | pl.col("position_close_is_actual_execution_claim")
        | pl.col("portfolio_controller_formal_go")
    ).height:
        raise ValueError("position template overlay claims unsupported execution truth")
    if result.cache_source_inventory.height != result.market_templates.height * len(
        EXPECTED_CACHE_ARTIFACTS
    ):
        raise ValueError("cache source inventory does not cover every template")
    if result.cache_source_inventory.schema != pl.Schema(_cache_inventory_schema()):
        raise ValueError("cache source inventory ordered schema changed")
    cache_keys = ["Date", "ValueCode", "QuoteCode"]
    cache_groups = result.cache_source_inventory.group_by(cache_keys).agg(
        pl.col("artifact_name").sort().alias("artifact_names"),
        pl.col("cache_complete_sha256").n_unique().alias("marker_hashes"),
        pl.col("cache_key_sha256").n_unique().alias("cache_keys"),
    )
    if (
        cache_groups.height != result.market_templates.height
        or cache_groups.filter(
            (pl.col("artifact_names") != pl.lit(sorted(EXPECTED_CACHE_ARTIFACTS)))
            | (pl.col("marker_hashes") != 1)
            | (pl.col("cache_keys") != 1)
        ).height
        or not cache_groups.select(cache_keys).sort(cache_keys).equals(
            result.market_templates.select(cache_keys).sort(cache_keys),
            null_equal=True,
        )
    ):
        raise ValueError("cache source inventory product-day/artifact grid changed")
    if result.cache_source_inventory.filter(
        ~pl.col("cache_artifact_content_hash_verified")
        | pl.col("upstream_raw_content_integrity_bound")
    ).height:
        raise ValueError("cache source inventory safety contract changed")
    entry_inventory = result.entry_price_source_inventory
    if (
        entry_inventory.schema != pl.Schema(_entry_price_source_inventory_schema())
        or entry_inventory.is_empty()
        or entry_inventory.select("Date", "ValueCode").unique().height
        != entry_inventory.height
        or entry_inventory.filter(
            (pl.col("source_kind") != "aggressive_1300_entry_execution_action")
            | pl.any_horizontal(
                pl.col(
                    "partition_complete_path",
                    "partition_complete_sha256",
                    "artifact_path",
                    "artifact_sha256",
                ).is_null()
            )
            | (pl.col("artifact_bytes") <= 0)
            | (pl.col("artifact_rows") < 0)
            | (pl.col("artifact_columns") <= 0)
        ).height
    ):
        raise ValueError("entry-price source inventory contract changed")
    _validate_controller_results(
        result.controller_position_outcomes,
        result.controller_daily_control,
        result.controller_scenario_summary,
        config,
    )


def prepare_analysis_result(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
    supplemental_root: Path = DEFAULT_SUPPLEMENTAL_ROOT,
    execution_manifest_path: Path | None = None,
    seed_template_path: Path = DEFAULT_SEED_TEMPLATE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> tuple[Aggressive1300AnalysisResult, dict[str, object]]:
    """Verify formal sources, replay carry-only cache keys, and build tables."""

    config.validate()
    verified = load_verified_source(Path(source_root), universe_root=Path(universe_root))
    supplemental_paths, supplemental_metadata = load_supplemental_normal_paths(
        Path(supplemental_root),
        base_paths=verified.paths,
        base_metadata=verified.metadata,
    )
    if execution_manifest_path is None:
        execution_manifest_path = Path(str(verified.metadata["execution_root"])) / (
            "execution_partition_manifest.parquet"
        )
    enriched, entry_inventory = load_all_exact_entry_prices(
        supplemental_paths,
        execution_manifest_path=Path(execution_manifest_path),
    )
    session_dates = _session_calendar_from_paths(enriched)
    if (
        len(session_dates) != FORMAL_SESSION_CALENDAR_COUNT
        or _session_calendar_sha256(session_dates) != FORMAL_SESSION_CALENDAR_SHA256
    ):
        raise ValueError("formal 63-session calendar identity changed")
    daily, inventory_audit = build_daily_open_inventory(
        enriched, session_dates=session_dates
    )
    required = daily.group_by(["Date", "ValueCode", "QuoteCode"]).agg(
        pl.len().alias("open_positions")
    )
    cache_roots, cache_inventory = validate_cache_sources(
        cache_root, required, rehash_artifacts=True
    )
    seed = load_seed_templates(seed_template_path)
    if seed.join(
        required.select("Date", "ValueCode", "QuoteCode"),
        on=["Date", "ValueCode", "QuoteCode"],
        how="anti",
    ).height:
        raise ValueError("seed template contains a product-day outside daily inventory")
    additions = replay_missing_cached_templates(required, seed, cache_roots)
    templates = pl.concat([seed, additions], how="vertical").sort(
        ["Date", "ValueCode", "QuoteCode"]
    )
    validate_market_templates(templates)
    # Cache inventory was built for all required keys that exist, exactly the
    # same set as seed+new templates.
    template_keys = templates.select("Date", "ValueCode", "QuoteCode")
    cache_inventory = cache_inventory.join(
        template_keys,
        on=["Date", "ValueCode", "QuoteCode"],
        how="semi",
    ).sort(["Date", "ValueCode", "QuoteCode", "artifact_name"])
    result = build_analysis_result(
        enriched,
        templates,
        cache_inventory,
        entry_inventory,
        inventory_audit,
        session_dates=session_dates,
        config=config,
    )
    normal_control_metadata = validate_normal_control_identity(
        result.controller_scenario_summary,
        result.controller_daily_control,
    )
    metadata = {
        **dict(verified.metadata),
        **supplemental_metadata,
        **normal_control_metadata,
        "formal_session_calendar": list(session_dates),
        "formal_session_calendar_count": len(session_dates),
        "formal_session_calendar_sha256": _session_calendar_sha256(session_dates),
        "seed_template_path": str(Path(seed_template_path).resolve()),
        "seed_template_sha256": _file_sha256(Path(seed_template_path)),
        "seed_template_rows": seed.height,
        "seed_generator_sha256": SEED_GENERATOR_SHA256,
        "candidate_cache_root": str(Path(cache_root).resolve()),
        "candidate_cache_selected_product_days": templates.height,
        "candidate_cache_inventory_digest": _frame_digest(cache_inventory),
        "entry_price_source_inventory_digest": _frame_digest(entry_inventory),
    }
    return result, metadata


def publish_analysis_bundle(
    output_root: Path,
    result: Aggressive1300AnalysisResult,
    source_metadata: Mapping[str, object],
    *,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
    verify_source_rebuild: bool = False,
) -> None:
    """Atomically publish twelve analysis artifacts plus a self-hashed marker."""

    config.validate()
    validate_analysis_result(result, config=config)
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        declarations: dict[str, dict[str, object]] = {}
        for filename, attribute in ARTIFACTS.items():
            frame = getattr(result, attribute)
            path = stage / filename
            frame.write_parquet(path)
            declarations[filename] = _frame_declaration(path, frame)
        marker: dict[str, object] = {
            "complete": True,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "config": _config_payload(config),
            "sources": json.loads(json.dumps(dict(source_metadata), sort_keys=True)),
            "implementation_sources": _implementation_sources(),
            "fact_semantics": _fact_semantics(config),
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_analysis_bundle(stage, verify_sources=verify_source_rebuild)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_analysis_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
    supplemental_root: Path = DEFAULT_SUPPLEMENTAL_ROOT,
    seed_template_path: Path = DEFAULT_SEED_TEMPLATE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    config: Aggressive1300AnalysisConfig = Aggressive1300AnalysisConfig(),
) -> Aggressive1300AnalysisResult:
    if Path(output_root).exists():
        raise FileExistsError(output_root)
    result, metadata = prepare_analysis_result(
        source_root=source_root,
        universe_root=universe_root,
        supplemental_root=supplemental_root,
        seed_template_path=seed_template_path,
        cache_root=cache_root,
        config=config,
    )
    publish_analysis_bundle(
        output_root,
        result,
        metadata,
        config=config,
        verify_source_rebuild=True,
    )
    return result


def verify_analysis_bundle(
    output_root: Path,
    *,
    verify_sources: bool = True,
) -> dict[str, object]:
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("aggressive analysis root is missing or symlinked")
    expected_files = {*ARTIFACTS, "complete.json"}
    if {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("aggressive analysis root file set changed")
    marker_path = root / "complete.json"
    if marker_path.is_symlink():
        raise ValueError("aggressive analysis marker cannot be a symlink")
    marker = _read_json(marker_path)
    unhashed = dict(marker)
    declared_marker_sha = unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or marker.get("analysis_version") != ANALYSIS_VERSION
        or declared_marker_sha != _canonical_sha256(unhashed)
        or set(marker) != {
            "complete",
            "schema_version",
            "analysis_version",
            "config",
            "sources",
            "implementation_sources",
            "fact_semantics",
            "artifacts",
            "marker_payload_sha256",
        }
    ):
        raise ValueError("aggressive analysis marker is invalid")
    if marker.get("implementation_sources") != _implementation_sources():
        raise ValueError("aggressive analysis implementation identity changed")
    expected_config = _config_payload(Aggressive1300AnalysisConfig())
    if marker.get("config") != expected_config:
        raise ValueError("aggressive analysis formal config changed")
    semantics = marker.get("fact_semantics")
    expected_semantics = _fact_semantics()
    if semantics != expected_semantics:
        raise ValueError("aggressive analysis safety semantics changed")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACTS):
        raise ValueError("aggressive analysis artifact set changed")
    frames: dict[str, pl.DataFrame] = {}
    for filename in ARTIFACTS:
        path = root / filename
        declaration = artifacts[filename]
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"sha256", "bytes", "rows", "columns", "schema"}
            or not path.is_file()
            or path.is_symlink()
            or _file_sha256(path) != declaration.get("sha256")
            or path.stat().st_size != int(declaration.get("bytes", -1))
        ):
            raise ValueError(f"aggressive analysis artifact changed: {filename}")
        frame = pl.read_parquet(path)
        if (
            frame.height != int(declaration.get("rows", -1))
            or frame.width != int(declaration.get("columns", -1))
            or {name: str(dtype) for name, dtype in frame.schema.items()}
            != declaration.get("schema")
        ):
            raise ValueError(f"aggressive analysis artifact schema changed: {filename}")
        frames[filename] = frame
    result = Aggressive1300AnalysisResult(
        **{attribute: frames[filename] for filename, attribute in ARTIFACTS.items()}
    )
    validate_analysis_result(result)
    if verify_sources:
        sources = marker.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("aggressive analysis source metadata is missing")
        if (
            _file_sha256(Path(str(sources["seed_template_path"])))
            != sources.get("seed_template_sha256")
            or sources.get("seed_template_sha256") != SEED_TEMPLATE_SHA256
            or int(sources.get("seed_template_rows", -1)) != SEED_TEMPLATE_ROWS
        ):
            raise ValueError("seed template source changed")
        cache_inventory = result.cache_source_inventory
        for item in cache_inventory.iter_rows(named=True):
            marker_source = Path(str(item["cache_complete_path"]))
            artifact_source = Path(str(item["artifact_path"]))
            if (
                marker_source.is_symlink()
                or artifact_source.is_symlink()
                or _file_sha256(marker_source) != item["cache_complete_sha256"]
                or _file_sha256(artifact_source) != item["artifact_sha256"]
                or artifact_source.stat().st_size != int(item["artifact_bytes"])
            ):
                raise ValueError("cache source inventory rehash failed")
        if _frame_digest(cache_inventory) != sources.get(
            "candidate_cache_inventory_digest"
        ):
            raise ValueError("cache source inventory digest changed")
        if _frame_digest(result.entry_price_source_inventory) != sources.get(
            "entry_price_source_inventory_digest"
        ):
            raise ValueError("entry price source inventory digest changed")
        execution_root = Path(str(sources["execution_root"])).resolve()
        for item in result.entry_price_source_inventory.iter_rows(named=True):
            marker_source = Path(str(item["partition_complete_path"]))
            artifact_source = Path(str(item["artifact_path"]))
            expected_partition = (
                execution_root
                / f"Date={item['Date']}"
                / f"ValueCode={item['ValueCode']}"
            )
            if (
                marker_source.is_symlink()
                or artifact_source.is_symlink()
                or marker_source.resolve() != expected_partition / "complete.json"
                or artifact_source.resolve()
                != expected_partition / "execution_action_facts.parquet"
                or _file_sha256(marker_source) != item["partition_complete_sha256"]
                or _file_sha256(artifact_source) != item["artifact_sha256"]
                or artifact_source.stat().st_size != int(item["artifact_bytes"])
                or len(pl.read_parquet_schema(artifact_source))
                != int(item["artifact_columns"])
                or pl.scan_parquet(artifact_source).select(pl.len()).collect().item()
                != int(item["artifact_rows"])
            ):
                raise ValueError("entry-price source inventory rehash failed")
        rebuilt, rebuilt_sources = prepare_analysis_result(
            source_root=Path(str(sources["source_root"])),
            universe_root=Path(str(sources["universe_root"])),
            supplemental_root=Path(str(sources["supplemental_root"])),
            execution_manifest_path=execution_root
            / "execution_partition_manifest.parquet",
            seed_template_path=Path(str(sources["seed_template_path"])),
            cache_root=Path(str(sources["candidate_cache_root"])),
            config=Aggressive1300AnalysisConfig(),
        )
        if json.loads(json.dumps(rebuilt_sources, sort_keys=True)) != sources:
            raise ValueError("aggressive analysis source metadata is not reproducible")
        for filename, attribute in ARTIFACTS.items():
            actual = getattr(result, attribute)
            expected = getattr(rebuilt, attribute)
            if actual.schema != expected.schema or not actual.equals(
                expected, null_equal=True
            ):
                raise ValueError(
                    f"aggressive analysis source rebuild differs: {filename}"
                )
    return marker


def _implementation_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "aggressive_1300_analysis.py",
        "aggressive_1300_analysis_cli.py",
        "aggressive_1300_exit.py",
        "combined_cost_cap_sweep.py",
        "exit_maker.py",
        "exit_maker_study.py",
        "engine.py",
        "layered.py",
        "hedge.py",
        "raw_tape.py",
        "targets.py",
    )
    return {name: _file_sha256(root / name) for name in names}


def _frame_declaration(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": frame.height,
        "columns": frame.width,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
    }


def _frame_digest(frame: pl.DataFrame) -> str:
    payload = {
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "rows": frame.to_dicts(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected optional integer, got {value!r}")
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected optional float, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"expected positive finite optional float, got {value!r}")
    return result


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"expected optional boolean, got {value!r}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected optional nonempty string, got {value!r}")
    return value
