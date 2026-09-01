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
    round_up_to_tick,
)
from .foundation_anchor_selection import materialize_anchor_column
from .policy_spec import ANCHOR_MODEL_ID, TOD_BUCKETS, PolicySpec
from .s1_scenario_spec import S1ScenarioSpec
from .transaction_costs import TransactionCostProfile

SESSION_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
# Stop normal exit-maker quoting before the 13:20 research horizon.  The
# 15-second reserve covers a conservative three-second drain for at most 244
# aggregate product cancels at 100 Spot requests/s, the 50 ms hedge delay,
# five seconds of hedge retry, five seconds of rollback retry, and sub-second
# scheduler/phase slack.  This is a risk-lifecycle bound, not an outcome-tuned
# exit parameter.
EXIT_STOP_SECOND: Final = 15_585
SESSION_END_SECOND: Final = 15_600
PRICE_EPSILON: Final = 1e-8
_DECISION_COST_PROFILE: Final = TransactionCostProfile()
_DECISION_COST_PROFILE.validate()

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
    "lookup_supported",
    "lookup_support_reason",
    "cost_horizon",
    "safety_floor_bp",
    "economic_gate_enabled",
    "deployment_shortlist_eligible",
    "upper_distance_bp",
    "lower_distance_bp",
    "entry_threshold_multiplier",
    "entry_threshold_basis_bp",
    "frozen_exit_threshold_basis_bp_at_observation",
    "target_price",
    "absolute_price_tick",
    "frozen_exit_target_price_at_observation",
    "frozen_exit_absolute_price_tick_at_observation",
    "spot_bid_tick",
    "point_offset",
    "target_location",
    "contract_size_integral",
    "passive_target",
    "target_in_reference_band",
    "base_gate_open",
    "ab12_admission_open",
    "gate_reason",
    "reservation_notional_twd_at_observation",
    "decision_selected_expected_margin_bp",
    "decision_economic_status",
    "decision_economic_reason",
    "decision_economic_gate_open",
)

# These are the complete decision outputs projected from the causal 1 Hz
# panel.  Raw-book changes have their own exact event clock; continuous anchor
# or margin values therefore do not belong in this panel-relative key once all
# rounded order prices and gate outcomes are equal.  This key does not assert
# that the landmark builder and RawBookDayIndex reconstruct identical books;
# the actual-send raw refresh remains authoritative.
S1_POLICY_DECISION_SIGNATURE_COLUMNS: Final = (
    "entry_tod_bucket",
    "lookup_supported",
    "lookup_support_reason",
    "absolute_price_tick",
    "frozen_exit_absolute_price_tick_at_observation",
    "reservation_notional_twd_at_observation",
    "base_gate_open",
    "ab12_admission_open",
    "gate_reason",
    "decision_economic_status",
    "decision_economic_reason",
    "decision_economic_gate_open",
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
        row.setdefault("cost_horizon", "ungated")
        row.setdefault("safety_floor_bp", None)
        row.setdefault("economic_gate_enabled", False)
        row.setdefault("deployment_shortlist_eligible", False)
        spec_rows.append(row)
    spec_frame = pl.from_dicts(spec_rows, infer_schema_length=None).with_columns(
        pl.col("lookup_supported").cast(pl.Boolean),
        pl.col("lookup_support_reason").cast(pl.String),
        pl.col("cost_horizon").cast(pl.String),
        pl.col("safety_floor_bp").cast(pl.Float64),
        pl.col("economic_gate_enabled").cast(pl.Boolean),
        pl.col("deployment_shortlist_eligible").cast(pl.Boolean),
    )
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
    frozen_exit_multiplier = (
        1.0 + (pl.col("selected_anchor_bp") - pl.col("lower_distance_bp")) / 10_000.0
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
    future_exit_valid = (
        pl.col("fut_exec_ask").is_not_null()
        & pl.col("fut_exec_ask").is_finite()
        & (pl.col("fut_exec_ask") > 0)
        & future_valid
        & (pl.col("fut_exec_bid") <= pl.col("fut_exec_ask"))
    )
    frozen_exit_geometry_valid = (
        anchor_valid
        & future_exit_valid
        & frozen_exit_multiplier.is_finite()
        & (frozen_exit_multiplier > 0)
    )
    raw_frozen_exit_target = pl.col("fut_exec_ask") / frozen_exit_multiplier
    frozen_exit_geometry_valid = (
        frozen_exit_geometry_valid
        & raw_frozen_exit_target.is_finite()
        & (raw_frozen_exit_target > 0)
    )
    frozen_exit_target_price = (
        pl.when(frozen_exit_geometry_valid)
        .then(round_up_to_tick(raw_frozen_exit_target, market="spot"))
        .otherwise(None)
    )
    frozen_exit_target_tick = (
        pl.when(frozen_exit_geometry_valid)
        .then(
            price_to_tick_index(
                frozen_exit_target_price,
                market="spot",
            ).round(0)
        )
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
            frozen_exit_target_price.alias("frozen_exit_target_price_at_observation"),
            frozen_exit_target_tick.alias(
                "frozen_exit_absolute_price_tick_at_observation"
            ),
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
        .with_columns(_decision_economic_expressions())
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
        *S1_POLICY_DECISION_SIGNATURE_COLUMNS,
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


def _decision_economic_expressions() -> tuple[pl.Expr, ...]:
    """Mirror ``evaluate_s1_entry_economics`` on the causal 1 Hz snapshot.

    Only the selected margin and discrete outcome are materialized.  The raw
    actual-send refresh remains authoritative.  This panel-relative projection
    only determines candidate one-second policy wakes; it is not a substitute
    for validating raw-book reconstruction at those boundaries.
    """

    profile = _DECISION_COST_PROFILE
    shares = pl.col("contract_size")
    entry_spot = pl.col("target_price")
    exit_spot = pl.col("frozen_exit_target_price_at_observation")
    entry_future = pl.col("fut_exec_bid")
    exit_future = pl.col("fut_exec_ask")
    spot_reference = pl.col("spot_ref_price")
    spot_bid = pl.col("spot_bid")

    reservation_valid = (
        pl.col("contract_size_integral")
        & entry_spot.is_not_null()
        & entry_spot.is_finite()
        & (entry_spot > 0)
    ).fill_null(False)
    reservation = (
        pl.when(reservation_valid)
        .then((entry_spot * shares).ceil())
        .otherwise(None)
        .cast(pl.Int64)
    )
    economic_inputs_valid = (
        pl.col("base_gate_open")
        & reservation_valid
        & exit_spot.is_not_null()
        & exit_spot.is_finite()
        & (exit_spot > 0)
        & entry_future.is_not_null()
        & entry_future.is_finite()
        & (entry_future > 0)
        & exit_future.is_not_null()
        & exit_future.is_finite()
        & (exit_future > 0)
        & (entry_future <= exit_future)
        & spot_reference.is_not_null()
        & spot_reference.is_finite()
        & (spot_reference > 0)
        & spot_bid.is_not_null()
        & spot_bid.is_finite()
        & (spot_bid > 0)
        & pl.col("cost_horizon").is_in(("ungated", "same_day", "overnight"))
        & (
            (~pl.col("economic_gate_enabled") & pl.col("safety_floor_bp").is_null())
            | (
                pl.col("economic_gate_enabled")
                & pl.col("safety_floor_bp").is_not_null()
                & pl.col("safety_floor_bp").is_finite()
                & (pl.col("safety_floor_bp") >= 0)
            )
        )
    ).fill_null(False)

    spot_commission_rate = profile.spot_commission_bp_per_side / 10_000.0
    spot_tax_rate = profile.spot_sell_tax_bp / 10_000.0
    future_tax_rate = profile.futures_tax_bp_per_side / 10_000.0
    spot_entry_commission = entry_spot * shares * spot_commission_rate
    spot_exit_commission = exit_spot * shares * spot_commission_rate
    same_day_spot_tax = (
        exit_spot * shares * spot_tax_rate * profile.same_day_spot_sell_tax_multiplier
    )
    overnight_spot_tax = exit_spot * shares * spot_tax_rate
    future_entry_tax = entry_future * shares * future_tax_rate
    future_exit_tax = exit_future * shares * future_tax_rate
    future_entry_commission = pl.lit(profile.futures_commission_twd_per_side)
    future_exit_commission = pl.lit(profile.futures_commission_twd_per_side)
    same_day_cost = (
        spot_entry_commission
        + spot_exit_commission
        + same_day_spot_tax
        + future_entry_tax
        + future_exit_tax
        + future_entry_commission
        + future_exit_commission
    )
    overnight_cost = (
        spot_entry_commission
        + spot_exit_commission
        + overnight_spot_tax
        + future_entry_tax
        + future_exit_tax
        + future_entry_commission
        + future_exit_commission
    )
    gross = shares * ((exit_spot - entry_spot) + (entry_future - exit_future))
    normalization = entry_spot * shares
    same_day_margin_bp = 10_000.0 * (gross - same_day_cost) / normalization
    overnight_margin_bp = 10_000.0 * (gross - overnight_cost) / normalization
    selected_margin_bp = (
        pl.when(pl.col("cost_horizon") == "same_day")
        .then(same_day_margin_bp)
        .when(pl.col("cost_horizon") == "overnight")
        .then(overnight_margin_bp)
        .otherwise(None)
    )
    exit_passive = exit_spot > spot_bid
    exit_in_band = (exit_spot > spot_reference * 0.91) & (
        exit_spot < spot_reference * 1.08
    )
    enabled_gate_open = selected_margin_bp > pl.col("safety_floor_bp")

    gate_open = (
        pl.when(~economic_inputs_valid)
        .then(None)
        .when(~pl.col("economic_gate_enabled"))
        .then(True)
        .when(~exit_passive)
        .then(False)
        .when(~exit_in_band)
        .then(False)
        .otherwise(enabled_gate_open)
        .cast(pl.Boolean)
    )
    status = (
        pl.when(~economic_inputs_valid)
        .then(None)
        .when(~pl.col("economic_gate_enabled"))
        .then(pl.lit("ungated_priced"))
        .when(~exit_passive | ~exit_in_band)
        .then(pl.lit("route_ineligible"))
        .when(enabled_gate_open)
        .then(pl.lit("eligible"))
        .otherwise(pl.lit("below_floor"))
    )
    reason = (
        pl.when(~economic_inputs_valid)
        .then(None)
        .when(~pl.col("economic_gate_enabled"))
        .then(pl.lit("ungated_control"))
        .when(~exit_passive)
        .then(pl.lit("frozen_exit_target_not_passive"))
        .when(~exit_in_band)
        .then(pl.lit("frozen_exit_target_outside_reference_band"))
        .when(enabled_gate_open)
        .then(pl.lit("eligible"))
        .otherwise(pl.lit("expected_margin_not_above_floor"))
    )
    return (
        reservation.alias("reservation_notional_twd_at_observation"),
        pl.when(economic_inputs_valid)
        .then(selected_margin_bp)
        .otherwise(None)
        .alias("decision_selected_expected_margin_bp"),
        status.alias("decision_economic_status"),
        reason.alias("decision_economic_reason"),
        gate_open.alias("decision_economic_gate_open"),
    )


def _thin_s1_policy_state_changes_lazy(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    ordered = frame.sort(["ValueCode", "seconds_from_open"])
    previous_names = {
        column: f"_prev_decision_{index}"
        for index, column in enumerate(S1_POLICY_DECISION_SIGNATURE_COLUMNS)
    }
    with_previous = ordered.with_columns(
        pl.col("seconds_from_open").shift(1).over("ValueCode").alias("_prev_second"),
        *(
            pl.col(column).shift(1).over("ValueCode").alias(previous_names[column])
            for column in S1_POLICY_DECISION_SIGNATURE_COLUMNS
        ),
    )
    changed = pl.col("_prev_second").is_null()
    for column in S1_POLICY_DECISION_SIGNATURE_COLUMNS:
        changed = changed | _different(
            pl.col(column),
            pl.col(previous_names[column]),
        )
    return with_previous.filter(changed).drop("_prev_second", *previous_names.values())


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
    "EXIT_STOP_SECOND",
    "S1_COMMON_STATE_COLUMNS",
    "S1_DAY_STATE_COLUMNS",
    "S1_POLICY_DECISION_SIGNATURE_COLUMNS",
    "SESSION_END_SECOND",
    "SESSION_START_SECOND",
    "build_s1_policy_day_state",
    "build_s1_policy_state_changes",
    "materialize_s1_common_day",
    "thin_s1_policy_state_changes",
    "weight_s1_policy_state_changes",
]
