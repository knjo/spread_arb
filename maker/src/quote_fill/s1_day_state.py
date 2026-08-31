"""Vectorized one-second decision state for the S1 Spot Bid entry route.

The daily causal panel is shared by all seven policy replays.  This module
materializes the frozen ``time_ewma_15s`` anchor, attaches one immutable
``PolicySpec`` per product/TOD cell, and derives the actual-cursor target and
AB1/2 admission geometry.  It deliberately does not create request, order,
fill, position, or capacity facts; those belong to the chronological loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from ..fair_mid.quote_churn import (
    price_to_tick_index,
    round_down_to_tick,
)
from .foundation_anchor_selection import materialize_anchor_column
from .policy_spec import ANCHOR_MODEL_ID, TOD_BUCKETS, PolicySpec
from .s1_scenario_spec import S1ScenarioSpec

SESSION_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
SESSION_END_SECOND: Final = 15_600
PRICE_EPSILON: Final = 1e-8

_PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
_CELL_KEYS: Final = (*_PRODUCT_KEYS, "entry_tod_bucket")

S1_COMMON_STATE_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "seconds_from_open",
    "decision_time_ns",
    "maker_snapshot_recv_time_ns",
    "maker_snapshot_channel_seq",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
    "end_date",
    "spot_bid",
    "spot_ask",
    "fut_exec_bid",
    "fut_exec_ask",
    "analysis_eligible",
    "selected_anchor_bp",
    "selected_anchor_model",
)

S1_DAY_STATE_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "seconds_from_open",
    "decision_time_ns",
    "maker_snapshot_recv_time_ns",
    "maker_snapshot_channel_seq",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
    "end_date",
    "spot_bid",
    "spot_ask",
    "fut_exec_bid",
    "fut_exec_ask",
    "analysis_eligible",
    "selected_anchor_bp",
    "selected_anchor_model",
    "entry_tod_bucket",
    "policy_id",
    "upper_distance_bp",
    "lower_distance_bp",
    "entry_threshold_multiplier",
    "entry_threshold_basis_bp",
    "frozen_exit_threshold_basis_bp_at_observation",
    "target_price",
    "absolute_price_tick",
    "spot_bid_tick",
    "point_offset",
    "target_location",
    "contract_size_integral",
    "passive_target",
    "target_in_reference_band",
    "base_gate_open",
    "ab12_admission_open",
    "gate_reason",
)


def materialize_s1_common_day(day: pl.DataFrame) -> pl.DataFrame:
    """Materialize the selected anchor once for all seven policy replays."""

    if not isinstance(day, pl.DataFrame):
        raise TypeError("day must be a Polars DataFrame")
    common = (
        materialize_anchor_column(day, ANCHOR_MODEL_ID)
        .filter(
            pl.col("seconds_from_open").is_between(
                SESSION_START_SECOND,
                ENTRY_STOP_SECOND - 1,
                closed="both",
            )
        )
        .with_columns(
            pl.col("timestamp").dt.timestamp("ns").alias("decision_time_ns"),
            pl.col("spot_recv_time")
            .dt.timestamp("ns")
            .alias("maker_snapshot_recv_time_ns"),
            pl.col("spot_sequence").cast(pl.UInt64).alias("maker_snapshot_channel_seq"),
        )
        .select(S1_COMMON_STATE_COLUMNS)
        .sort(["ValueCode", "seconds_from_open"])
    )
    products = common.select(_PRODUCT_KEYS).unique().height
    expected_rows = products * (ENTRY_STOP_SECOND - SESSION_START_SECOND)
    if common.height != expected_rows:
        raise ValueError(
            "causal day does not contain the complete S1 common grid: "
            f"expected={expected_rows}, actual={common.height}"
        )
    if common.select("ValueCode", "seconds_from_open").n_unique() != common.height:
        raise ValueError("S1 common day keys are duplicated")
    return common


def build_s1_policy_day_state(
    day: pl.DataFrame,
    specs: Sequence[PolicySpec | S1ScenarioSpec],
    *,
    policy_id: str,
    _sparse_only: bool = False,
) -> pl.DataFrame:
    """Build the complete 09:05--13:00 actual-send state for one policy.

    ``day`` must contain the complete 00:00--13:19:59 one-second grid used by
    the frozen anchor implementation.  ``specs`` must contain exactly one row
    for every product and each of the four S1 TOD cells.  Unsupported or
    missing cells fail closed instead of disappearing in an inner join.
    """

    if not isinstance(day, pl.DataFrame):
        raise TypeError("day must be a Polars DataFrame")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id must be a non-empty string")
    if not isinstance(_sparse_only, bool):
        raise TypeError("_sparse_only must be boolean")
    values = tuple(specs)
    if not values:
        raise ValueError("specs cannot be empty")
    if any(not isinstance(spec, (PolicySpec, S1ScenarioSpec)) for spec in values):
        raise TypeError("specs must contain PolicySpec or S1ScenarioSpec values")
    if any(spec.policy_id != policy_id for spec in values):
        raise ValueError("all specs must match policy_id")

    common = (
        day
        if set(S1_COMMON_STATE_COLUMNS).issubset(day.columns)
        else materialize_s1_common_day(day)
    )
    _require_columns(common, set(S1_COMMON_STATE_COLUMNS), "S1 common day")
    date_values = common.select(pl.col("Date").cast(pl.String)).unique()["Date"]
    if len(date_values) != 1:
        raise ValueError("day must contain exactly one Date")
    date = str(date_values[0])
    if any(spec.Date != date for spec in values):
        raise ValueError("PolicySpec Date does not match the causal day")

    spec_rows: list[dict[str, object]] = []
    for spec in values:
        row = spec.to_dict()
        row["policy_id"] = spec.policy_id
        row.setdefault("lookup_supported", True)
        row.setdefault("lookup_support_reason", "supported")
        spec_rows.append(row)
    spec_frame = pl.from_dicts(spec_rows, infer_schema_length=None)
    duplicate = spec_frame.group_by(_CELL_KEYS).len().filter(pl.col("len") != 1)
    if not duplicate.is_empty():
        raise ValueError("PolicySpec cells are duplicated")
    spec_products = spec_frame.select(_PRODUCT_KEYS).unique()
    day_products = common.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    ).unique()
    if not spec_products.sort(_PRODUCT_KEYS).equals(day_products.sort(_PRODUCT_KEYS)):
        raise ValueError("PolicySpec products do not exactly match the causal day")
    expected_rows = day_products.height * (ENTRY_STOP_SECOND - SESSION_START_SECOND)
    if common.height != expected_rows:
        raise ValueError(
            "causal day does not contain the complete S1 decision grid: "
            f"expected={expected_rows}, actual={common.height}"
        )
    if common.select("ValueCode", "seconds_from_open").n_unique() != common.height:
        raise ValueError("S1 common day keys are duplicated")
    coverage = spec_frame.group_by(_PRODUCT_KEYS).agg(
        pl.len().alias("rows"),
        pl.col("entry_tod_bucket").n_unique().alias("tod_cells"),
        pl.col("entry_tod_bucket").sort().alias("tod_values"),
    )
    invalid_coverage = coverage.filter(
        (pl.col("rows") != len(TOD_BUCKETS))
        | (pl.col("tod_cells") != len(TOD_BUCKETS))
        | (pl.col("tod_values") != sorted(TOD_BUCKETS))
    )
    if not invalid_coverage.is_empty():
        raise ValueError("each product requires the exact four S1 TOD specs")

    decision = (
        common.lazy()
        .with_columns(_tod_bucket().alias("entry_tod_bucket"))
        .join(
            spec_frame.lazy(),
            on=list(_CELL_KEYS),
            how="left",
            validate="m:1",
        )
    )

    anchor_valid = (
        pl.col("selected_anchor_bp").is_not_null()
        & pl.col("selected_anchor_bp").is_finite()
    )
    future_valid = (
        pl.col("fut_exec_bid").is_not_null()
        & pl.col("fut_exec_bid").is_finite()
        & (pl.col("fut_exec_bid") > 0)
    )
    multiplier = (
        1.0 + (pl.col("selected_anchor_bp") + pl.col("upper_distance_bp")) / 10_000.0
    )
    raw_target = pl.col("fut_exec_bid") / multiplier
    target_geometry_valid = (
        anchor_valid
        & future_valid
        & multiplier.is_finite()
        & (multiplier > 0)
        & raw_target.is_finite()
        & (raw_target > 0)
    )
    target_price = (
        pl.when(target_geometry_valid)
        .then(round_down_to_tick(raw_target, market="spot"))
        .otherwise(None)
    )
    target_tick = (
        pl.when(target_geometry_valid)
        .then(price_to_tick_index(target_price, market="spot").round(0))
        .otherwise(None)
        .cast(pl.Int64)
    )
    spot_bid_valid = (
        pl.col("spot_bid").is_not_null()
        & pl.col("spot_bid").is_finite()
        & (pl.col("spot_bid") > 0)
    )
    spot_bid_tick = (
        pl.when(spot_bid_valid)
        .then(price_to_tick_index(pl.col("spot_bid"), market="spot").round(0))
        .otherwise(None)
        .cast(pl.Int64)
    )

    result = (
        decision.with_columns(
            multiplier.alias("entry_threshold_multiplier"),
            (pl.col("selected_anchor_bp") + pl.col("upper_distance_bp")).alias(
                "entry_threshold_basis_bp"
            ),
            (pl.col("selected_anchor_bp") - pl.col("lower_distance_bp")).alias(
                "frozen_exit_threshold_basis_bp_at_observation"
            ),
            target_price.alias("target_price"),
            target_tick.alias("absolute_price_tick"),
            spot_bid_tick.alias("spot_bid_tick"),
        )
        .with_columns(
            (pl.col("absolute_price_tick") - pl.col("spot_bid_tick")).alias(
                "point_offset"
            ),
            (
                pl.col("contract_size").is_not_null()
                & pl.col("contract_size").is_finite()
                & (pl.col("contract_size") > 0)
                & (
                    (pl.col("contract_size") - pl.col("contract_size").round(0)).abs()
                    < 1e-9
                )
            )
            .fill_null(False)
            .alias("contract_size_integral"),
            (
                pl.col("target_price").is_not_null()
                & pl.col("spot_ask").is_not_null()
                & pl.col("spot_ask").is_finite()
                & (pl.col("spot_ask") > 0)
                & (pl.col("target_price") < pl.col("spot_ask"))
            )
            .fill_null(False)
            .alias("passive_target"),
            (
                pl.col("target_price").is_not_null()
                & pl.col("spot_ref_price").is_not_null()
                & pl.col("spot_ref_price").is_finite()
                & (pl.col("spot_ref_price") > 0)
                & (pl.col("target_price") > pl.col("spot_ref_price") * 0.91)
                & (pl.col("target_price") < pl.col("spot_ref_price") * 1.08)
            )
            .fill_null(False)
            .alias("target_in_reference_band"),
        )
        .with_columns(
            _target_location().alias("target_location"),
            (
                pl.col("analysis_eligible").fill_null(False)
                & anchor_valid
                & target_geometry_valid
                & pl.col("passive_target")
                & pl.col("target_in_reference_band")
                & pl.col("contract_size_integral")
                & pl.col("lookup_supported").fill_null(False)
                & ~pl.col("contains_target_day_outcome").fill_null(True)
            )
            .fill_null(False)
            .alias("base_gate_open"),
        )
        .with_columns(
            (pl.col("base_gate_open") & pl.col("point_offset").is_in([-1, 0]))
            .fill_null(False)
            .alias("ab12_admission_open"),
            _gate_reason().alias("gate_reason"),
        )
    )
    if _sparse_only:
        result = _thin_s1_policy_state_changes_lazy(result)
    collected = (
        result.select(S1_DAY_STATE_COLUMNS)
        .sort(["ValueCode", "seconds_from_open"])
        .collect(engine="streaming")
    )
    if _sparse_only:
        _validate_sparse_output(
            collected,
            policy_id=policy_id,
            expected_products=day_products.height,
        )
    else:
        _validate_output(collected, policy_id=policy_id, expected_rows=expected_rows)
    return collected


def build_s1_policy_state_changes(
    day: pl.DataFrame,
    specs: Sequence[PolicySpec | S1ScenarioSpec],
    *,
    policy_id: str,
) -> pl.DataFrame:
    """Build only controller-relevant changes without materializing full state."""

    return build_s1_policy_day_state(
        day,
        specs,
        policy_id=policy_id,
        _sparse_only=True,
    )


def thin_s1_policy_state_changes(frame: pl.DataFrame) -> pl.DataFrame:
    """Retain only controller-relevant changes while preserving snapshots."""

    required = {
        "ValueCode",
        "seconds_from_open",
        "absolute_price_tick",
        "base_gate_open",
        "ab12_admission_open",
    }
    _require_columns(frame, required, "S1 policy day state")
    return _thin_s1_policy_state_changes_lazy(frame.lazy()).collect(engine="streaming")


def weight_s1_policy_state_changes(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the number of product-seconds represented by each sparse row."""

    required = {"ValueCode", "seconds_from_open", "policy_id"}
    _require_columns(frame, required, "S1 sparse policy state")
    if frame.is_empty():
        raise ValueError("S1 sparse policy state is empty")
    weighted = (
        frame.sort(["ValueCode", "seconds_from_open"])
        .with_columns(
            pl.col("seconds_from_open")
            .shift(-1)
            .over("ValueCode")
            .fill_null(ENTRY_STOP_SECOND)
            .alias("_next_second")
        )
        .with_columns(
            (pl.col("_next_second") - pl.col("seconds_from_open"))
            .cast(pl.UInt32)
            .alias("represented_product_seconds")
        )
        .drop("_next_second")
    )
    if weighted.filter(pl.col("represented_product_seconds") == 0).height:
        raise ValueError("S1 sparse state contains a non-positive interval")
    expected_per_product = ENTRY_STOP_SECOND - SESSION_START_SECOND
    invalid_totals = (
        weighted.group_by("ValueCode")
        .agg(pl.col("represented_product_seconds").sum().alias("represented_seconds"))
        .filter(pl.col("represented_seconds") != expected_per_product)
    )
    if not invalid_totals.is_empty():
        raise ValueError("S1 sparse state does not cover the complete entry window")
    return weighted


def _thin_s1_policy_state_changes_lazy(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    ordered = frame.sort(["ValueCode", "seconds_from_open"])
    with_previous = ordered.with_columns(
        pl.col("seconds_from_open").shift(1).over("ValueCode").alias("_prev_second"),
        pl.col("absolute_price_tick").shift(1).over("ValueCode").alias("_prev_tick"),
        pl.col("base_gate_open").shift(1).over("ValueCode").alias("_prev_gate"),
        pl.col("ab12_admission_open")
        .shift(1)
        .over("ValueCode")
        .alias("_prev_admission"),
    )
    changed = (
        pl.col("_prev_second").is_null()
        | _different(pl.col("absolute_price_tick"), pl.col("_prev_tick"))
        | _different(pl.col("base_gate_open"), pl.col("_prev_gate"))
        | _different(pl.col("ab12_admission_open"), pl.col("_prev_admission"))
    )
    return with_previous.filter(changed).drop(
        "_prev_second", "_prev_tick", "_prev_gate", "_prev_admission"
    )


def _tod_bucket() -> pl.Expr:
    second = pl.col("seconds_from_open")
    return (
        pl.when(second < 3_600)
        .then(pl.lit("0905_1000"))
        .when(second < 7_200)
        .then(pl.lit("1000_1100"))
        .when(second < 10_800)
        .then(pl.lit("1100_1200"))
        .otherwise(pl.lit("1200_1300"))
    )


def _target_location() -> pl.Expr:
    offset = pl.col("point_offset")
    return (
        pl.when(~pl.col("passive_target"))
        .then(pl.lit("not_passive"))
        .when(offset > 0)
        .then(pl.lit("inside_spread"))
        .when(offset == 0)
        .then(pl.lit("BID1"))
        .when(offset == -1)
        .then(pl.lit("BID2_BY_TICK"))
        .when(offset.is_between(-4, -2, closed="both"))
        .then(pl.lit("BID3_TO_BID5_BY_TICK"))
        .when(offset < -4)
        .then(pl.lit("DEEPER_THAN_BID5_BY_TICK"))
        .otherwise(pl.lit("invalid"))
    )


def _gate_reason() -> pl.Expr:
    return (
        pl.when(~pl.col("lookup_supported").fill_null(False))
        .then(
            pl.concat_str(
                pl.lit("lookup_unsupported:"),
                pl.col("lookup_support_reason").fill_null("unknown"),
            )
        )
        .when(~pl.col("analysis_eligible").fill_null(False))
        .then(pl.lit("input_gate_closed"))
        .when(
            pl.col("selected_anchor_bp").is_null()
            | ~pl.col("selected_anchor_bp").is_finite()
        )
        .then(pl.lit("anchor_missing"))
        .when(pl.col("absolute_price_tick").is_null())
        .then(pl.lit("invalid_target_geometry"))
        .when(~pl.col("passive_target"))
        .then(pl.lit("target_not_passive"))
        .when(~pl.col("target_in_reference_band"))
        .then(pl.lit("target_outside_reference_band"))
        .when(~pl.col("contract_size_integral"))
        .then(pl.lit("invalid_contract_size"))
        .when(pl.col("contains_target_day_outcome").fill_null(True))
        .then(pl.lit("lookahead_provenance"))
        .otherwise(pl.lit("eligible"))
    )


def _different(current: pl.Expr, previous: pl.Expr) -> pl.Expr:
    return (current.is_null() != previous.is_null()) | (current != previous).fill_null(
        False
    )


def _require_columns(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _validate_output(
    frame: pl.DataFrame, *, policy_id: str, expected_rows: int
) -> None:
    if frame.height != expected_rows:
        raise ValueError("S1 policy day state row count drifted")
    if frame.select("ValueCode", "seconds_from_open").n_unique() != frame.height:
        raise ValueError("S1 policy day state keys are duplicated")
    if frame.filter(pl.col("policy_id") != policy_id).height:
        raise ValueError("S1 policy day state contains another policy")
    _validate_state_invariants(frame)


def _validate_sparse_output(
    frame: pl.DataFrame,
    *,
    policy_id: str,
    expected_products: int,
) -> None:
    if frame.is_empty():
        raise ValueError("S1 sparse policy state is empty")
    if frame.select("ValueCode", "seconds_from_open").n_unique() != frame.height:
        raise ValueError("S1 sparse policy state keys are duplicated")
    if frame.filter(
        pl.col("policy_id").is_null() | (pl.col("policy_id") != policy_id)
    ).height:
        raise ValueError("S1 sparse policy state contains another policy")
    first_rows = frame.group_by("ValueCode").agg(
        pl.col("seconds_from_open").min().alias("first_second")
    )
    if (
        first_rows.height != expected_products
        or first_rows.filter(pl.col("first_second") != SESSION_START_SECOND).height
    ):
        raise ValueError("S1 sparse policy state lost a product's initial state")
    _validate_state_invariants(frame)


def _validate_state_invariants(frame: pl.DataFrame) -> None:
    invalid_sendable = frame.filter(
        pl.col("ab12_admission_open")
        & (
            ~pl.col("base_gate_open")
            | ~pl.col("point_offset").is_in([-1, 0])
            | pl.col("absolute_price_tick").is_null()
        )
    )
    if not invalid_sendable.is_empty():
        raise ValueError("AB1/2 admission invariants failed")
    invalid_contract = frame.filter(
        pl.col("contract_size_integral")
        & ~((pl.col("contract_size") - pl.col("contract_size").round(0)).abs() < 1e-9)
    )
    if not invalid_contract.is_empty():
        raise ValueError("integral contract-size flag drifted")


__all__ = [
    "ENTRY_STOP_SECOND",
    "S1_COMMON_STATE_COLUMNS",
    "S1_DAY_STATE_COLUMNS",
    "SESSION_END_SECOND",
    "SESSION_START_SECOND",
    "build_s1_policy_day_state",
    "build_s1_policy_state_changes",
    "materialize_s1_common_day",
    "thin_s1_policy_state_changes",
    "weight_s1_policy_state_changes",
]
