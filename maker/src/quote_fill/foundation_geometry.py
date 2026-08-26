"""D-safe reference geometry for the seven foundation policies.

This module is intentionally narrower than an execution lookup.  It combines
one broad, pre-open-safe product-day cohort with q50/q80/q95 rolling boundary
rows, adds the four symmetric fixed-BP controls, and evaluates every policy at
opening reference prices on the legal price ladder.

The resulting prices are diagnostics, not orders.  There is no intraday BBO,
maker fill, 50 ms hedge, passive clamp, terminal path, or expected value in
this layer.  In particular, historical TTBand columns may be carried through
for audit but are never deducted from the geometry: the reference entry uses a
future bid proxy and the reference exit uses a future ask proxy directly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
    absolute_price_tick,
    effective_basis_bp,
    price_in_ref_band,
    target_price_for_basis,
    tick_index_to_price,
)
from .transaction_costs import TransactionCostProfile

FOUNDATION_GEOMETRY_VERSION = "foundation_reference_geometry_v1"
KEYS = ("Date", "ValueCode", "QuoteCode")
QUANTILE_POLICIES = (50, 80, 95)
DEFAULT_ADVERSE_SENSITIVITY_BP = (10.0, 20.0, 30.0)
GEOMETRY_EPS = 1e-7


@dataclass(frozen=True)
class FoundationPolicySpec:
    policy_id: str
    policy_kind: str
    boundary_quantile: int | None
    fixed_distance_bp: float | None
    policy_order: int


FOUNDATION_POLICY_SPECS = (
    FoundationPolicySpec("q50", "rolling_quantile", 50, None, 0),
    FoundationPolicySpec("q80", "rolling_quantile", 80, None, 1),
    FoundationPolicySpec("q95", "rolling_quantile", 95, None, 2),
    FoundationPolicySpec("fixed15", "fixed_symmetric", None, 15.0, 3),
    FoundationPolicySpec("fixed20", "fixed_symmetric", None, 20.0, 4),
    FoundationPolicySpec("fixed25", "fixed_symmetric", None, 25.0, 5),
    FoundationPolicySpec("fixed30", "fixed_symmetric", None, 30.0, 6),
)


def build_policy_geometry(
    broad_cohort: pl.DataFrame,
    boundary_wide: pl.DataFrame,
    *,
    cost_profile: TransactionCostProfile | None = None,
    adverse_sensitivity_bp: Iterable[float] = DEFAULT_ADVERSE_SENSITIVITY_BP,
) -> pl.DataFrame:
    """Build seven-policy pre-open reference geometry.

    ``boundary_wide`` may be the current long rolling-boundary publication
    (one row per quantile) or a truly wide table with columns such as
    ``q50_upper_distance_bp`` and ``q50_lower_distance_bp``.  Every cohort key
    must have valid q50, q80, and q95 rows.  This common-coverage contract keeps
    the four fixed controls on exactly the same product-days as the q policies.
    """

    cohort = _prepare_cohort(broad_cohort)
    boundary_sample = boundary_wide.join(
        cohort.select(list(KEYS)),
        on=list(KEYS),
        how="semi",
    )
    reference_columns = {"spot_ref_price", "fut_ref_price", "contract_size"}
    missing_references = reference_columns - set(boundary_sample.columns)
    if missing_references:
        _require(cohort, missing_references, "broad cohort reference metadata")
        boundary_sample = boundary_sample.join(
            cohort.select(*KEYS, *sorted(missing_references)),
            on=list(KEYS),
            how="left",
            validate="m:1",
        )
    boundaries = _normalise_boundaries(boundary_sample)
    if boundaries.filter(~pl.col("adaptive_parameter_valid")).height:
        raise ValueError("common geometry requires valid q50/q80/q95 boundaries")
    _validate_common_boundary_coverage(cohort, boundaries)

    overlap = (set(cohort.columns) & set(boundaries.columns)) - set(KEYS)
    if overlap:
        cohort = cohort.rename(
            {column: f"cohort_{column}" for column in sorted(overlap)}
        )

    quantile_rows = cohort.join(
        boundaries,
        on=list(KEYS),
        how="inner",
        validate="1:m",
    ).with_columns(
        pl.concat_str(
            pl.lit("q"),
            pl.col("boundary_quantile").cast(pl.String),
        ).alias("policy_id"),
        pl.lit("rolling_quantile").alias("policy_kind"),
        pl.col("boundary_quantile")
        .replace_strict({50: 0, 80: 1, 95: 2})
        .cast(pl.Int64)
        .alias("policy_order"),
        pl.lit(None, dtype=pl.Float64).alias("fixed_distance_bp"),
    )

    fixed_base = quantile_rows.filter(pl.col("boundary_quantile") == 50)
    fixed_frames: list[pl.DataFrame] = []
    for spec in FOUNDATION_POLICY_SPECS[3:]:
        assert spec.fixed_distance_bp is not None
        fixed_frames.append(
            fixed_base.with_columns(
                pl.lit(spec.policy_id).alias("policy_id"),
                pl.lit(spec.policy_kind).alias("policy_kind"),
                pl.lit(spec.policy_order).cast(pl.Int64).alias("policy_order"),
                pl.lit(None, dtype=pl.Int64).alias("boundary_quantile"),
                pl.lit(spec.fixed_distance_bp).alias("fixed_distance_bp"),
                pl.lit(spec.fixed_distance_bp).alias("upper_distance_bp"),
                pl.lit(spec.fixed_distance_bp).alias("lower_distance_bp"),
                pl.lit("fixed_symmetric_control").alias("boundary_role"),
            )
        )

    policies = pl.concat(
        [quantile_rows, *fixed_frames],
        how="diagonal_relaxed",
    ).with_columns(
        (pl.col("upper_distance_bp") + pl.col("lower_distance_bp")).alias(
            "nominal_band_bp"
        )
    )
    result = build_reference_policy_geometry(
        policies,
        cost_profile=cost_profile,
        adverse_sensitivity_bp=adverse_sensitivity_bp,
    )
    _validate_output_coverage(result, cohort.height)
    return result.sort([*KEYS, "policy_order"])


def build_reference_policy_geometry(
    policies: pl.DataFrame,
    *,
    cost_profile: TransactionCostProfile | None = None,
    adverse_sensitivity_bp: Iterable[float] = DEFAULT_ADVERSE_SENSITIVITY_BP,
    allow_zero_lower: bool = False,
) -> pl.DataFrame:
    """Add exact opening-reference cost geometry to prepared policy rows.

    This is the reusable row-level primitive beneath the seven-policy adapter.
    Callers must provide one already identified policy per row, including its
    D-safe source lineage, reference prices, and upper/lower distances.  The
    normal foundation lookup keeps both distances strictly positive.  A
    conditional post-entry convergence target may legitimately resolve to the
    center, so such callers can explicitly allow a zero lower distance without
    weakening the original seven-policy contract.
    """

    required = {
        *KEYS,
        "policy_id",
        "policy_kind",
        "policy_order",
        "source_asof_date",
        "contains_target_day_outcome",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
        "upper_distance_bp",
        "lower_distance_bp",
        "nominal_band_bp",
    }
    _require(policies, required, "prepared policy rows")
    if policies.is_empty():
        raise ValueError("prepared policy rows must not be empty")
    prepared = policies.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("policy_id").cast(pl.String),
        pl.col("policy_kind").cast(pl.String),
        pl.col("policy_order").cast(pl.Int64),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("contains_target_day_outcome").fill_null(True).cast(pl.Boolean),
        pl.col("spot_ref_price").cast(pl.Float64),
        pl.col("fut_ref_price").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("lower_distance_bp").cast(pl.Float64),
        pl.col("nominal_band_bp").cast(pl.Float64),
    )
    _validate_d_safe(prepared, "prepared policy rows")
    if prepared.filter(
        ~pl.col("spot_ref_price").is_finite()
        | (pl.col("spot_ref_price") <= 0.0)
        | ~pl.col("fut_ref_price").is_finite()
        | (pl.col("fut_ref_price") <= 0.0)
        | ~pl.col("contract_size").is_finite()
        | (pl.col("contract_size") <= 0.0)
    ).height:
        raise ValueError("policy reference prices and contract size must be positive")
    _validate_policy_distances(prepared, allow_zero_lower=allow_zero_lower)
    profile = cost_profile or TransactionCostProfile()
    profile.validate()
    scenarios = _normalise_adverse_scenarios(adverse_sensitivity_bp)
    return _add_reference_geometry(
        prepared,
        profile,
        scenarios,
        allow_zero_lower=allow_zero_lower,
    )


def summarize_policy_geometry(frame: pl.DataFrame) -> pl.DataFrame:
    """Summarize reference geometry without promoting it to an EV table."""

    required = {
        "policy_id",
        "policy_kind",
        "policy_order",
        "Date",
        "ValueCode",
        "nominal_band_bp",
        "rounded_reference_band_bp",
        "same_day_reference_cost_bp",
        "overnight_reference_cost_bp",
        "same_day_known_cost_margin_bp",
        "overnight_known_cost_margin_bp",
        "nominal_same_day_known_cost_margin_bp",
        "nominal_overnight_known_cost_margin_bp",
        "reference_entry_inside_ref_band",
        "reference_exit_inside_ref_band",
        "actionable_execution",
        "ev_ready",
    }
    _require(frame, required, "policy geometry")
    if frame.filter(
        pl.col("actionable_execution").fill_null(True)
        | pl.col("ev_ready").fill_null(True)
    ).height:
        raise ValueError("foundation geometry cannot be actionable or EV-ready")
    adverse_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("same_day_margin_after_")
        and column.endswith("bp_adverse_bp")
    )
    aggregations: list[pl.Expr] = [
        pl.len().alias("product_days"),
        pl.col("Date").n_unique().alias("sessions"),
        pl.col("ValueCode").n_unique().alias("products"),
        pl.col("nominal_band_bp").median().alias("nominal_band_bp_p50"),
        pl.col("upper_distance_bp").median().alias("upper_distance_bp_p50"),
        pl.col("lower_distance_bp").median().alias("lower_distance_bp_p50"),
        pl.col("rounded_reference_band_bp")
        .median()
        .alias("rounded_reference_band_bp_p50"),
        pl.col("same_day_reference_cost_bp")
        .median()
        .alias("same_day_reference_cost_bp_p50"),
        pl.col("overnight_reference_cost_bp")
        .median()
        .alias("overnight_reference_cost_bp_p50"),
        pl.col("same_day_known_cost_margin_bp")
        .median()
        .alias("same_day_known_cost_margin_bp_p50"),
        pl.col("overnight_known_cost_margin_bp")
        .median()
        .alias("overnight_known_cost_margin_bp_p50"),
        pl.col("nominal_same_day_known_cost_margin_bp")
        .median()
        .alias("nominal_same_day_known_cost_margin_bp_p50"),
        pl.col("nominal_overnight_known_cost_margin_bp")
        .median()
        .alias("nominal_overnight_known_cost_margin_bp_p50"),
        (pl.col("same_day_known_cost_margin_bp") > 0)
        .sum()
        .alias("same_day_known_cost_positive_rows"),
        (pl.col("overnight_known_cost_margin_bp") > 0)
        .sum()
        .alias("overnight_known_cost_positive_rows"),
        (pl.col("nominal_same_day_known_cost_margin_bp") > 0)
        .sum()
        .alias("nominal_same_day_known_cost_positive_rows"),
        (pl.col("nominal_overnight_known_cost_margin_bp") > 0)
        .sum()
        .alias("nominal_overnight_known_cost_positive_rows"),
        (
            pl.col("reference_entry_inside_ref_band")
            & pl.col("reference_exit_inside_ref_band")
        )
        .sum()
        .alias("both_reference_targets_inside_ref_band_rows"),
        *[
            (pl.col(column) > 0).sum().alias(f"{column}_positive_rows")
            for column in adverse_columns
        ],
    ]
    return (
        frame.group_by(["policy_id", "policy_kind", "policy_order"])
        .agg(*aggregations)
        .with_columns(
            pl.lit(True).alias("reference_diagnostic"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
        )
        .sort("policy_order")
    )


def _prepare_cohort(frame: pl.DataFrame) -> pl.DataFrame:
    _require(frame, set(KEYS), "broad cohort")
    if frame.is_empty():
        raise ValueError("broad cohort must not be empty")
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if result.select(list(KEYS)).n_unique() != result.height:
        raise ValueError("broad cohort must be unique by product-day contract")
    _validate_d_safe(result, "broad cohort")
    return result


def _normalise_boundaries(frame: pl.DataFrame) -> pl.DataFrame:
    _require(
        frame,
        {
            *KEYS,
            "source_asof_date",
            "contains_target_day_outcome",
            "spot_ref_price",
            "fut_ref_price",
            "contract_size",
        },
        "boundary table",
    )
    _validate_d_safe(frame, "boundary table")
    if "boundary_quantile" in frame.columns:
        _require(
            frame,
            {
                "upper_distance_bp",
                "lower_distance_bp",
                "adaptive_parameter_valid",
            },
            "long boundary table",
        )
        result = frame.filter(
            pl.col("boundary_quantile").is_in(QUANTILE_POLICIES)
        ).with_columns(pl.col("boundary_quantile").cast(pl.Int64))
    else:
        result = _melt_wide_boundaries(frame)

    result = result.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("spot_ref_price").cast(pl.Float64),
        pl.col("fut_ref_price").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("lower_distance_bp").cast(pl.Float64),
        pl.col("adaptive_parameter_valid").fill_null(False).cast(pl.Boolean),
    )
    key = [*KEYS, "boundary_quantile"]
    if result.select(key).n_unique() != result.height:
        raise ValueError("boundary table contains duplicate quantile keys")
    if result.filter(
        ~pl.col("spot_ref_price").is_finite()
        | (pl.col("spot_ref_price") <= 0)
        | ~pl.col("fut_ref_price").is_finite()
        | (pl.col("fut_ref_price") <= 0)
        | ~pl.col("contract_size").is_finite()
        | (pl.col("contract_size") <= 0)
    ).height:
        raise ValueError("boundary reference prices and contract size must be positive")
    if (
        "price_ladder_version" in result.columns
        and result.filter(pl.col("price_ladder_version") != PRICE_LADDER_VERSION).height
    ):
        raise ValueError("boundary price ladder version is stale")
    return result


def _melt_wide_boundaries(frame: pl.DataFrame) -> pl.DataFrame:
    pairs: dict[int, tuple[str, str]] = {}
    policy_columns: set[str] = set()
    for quantile in QUANTILE_POLICIES:
        upper = _first_column(
            frame,
            (
                f"q{quantile}_upper_distance_bp",
                f"upper_distance_bp_q{quantile}",
                f"upper_distance_bp_{quantile}",
            ),
        )
        lower = _first_column(
            frame,
            (
                f"q{quantile}_lower_distance_bp",
                f"lower_distance_bp_q{quantile}",
                f"lower_distance_bp_{quantile}",
            ),
        )
        if upper is None or lower is None:
            raise ValueError(
                f"wide boundary table lacks q{quantile} upper/lower distances"
            )
        pairs[quantile] = (upper, lower)
        policy_columns.update((upper, lower))
        policy_columns.update(
            column
            for column in (
                f"q{quantile}_adaptive_parameter_valid",
                f"adaptive_parameter_valid_q{quantile}",
                f"q{quantile}_boundary_role",
                f"boundary_role_q{quantile}",
            )
            if column in frame.columns
        )
    policy_columns.update(
        column
        for column in (
            "adaptive_parameter_valid",
            "all_boundary_supported",
            "boundary_role",
        )
        if column in frame.columns
    )
    metadata = [column for column in frame.columns if column not in policy_columns]
    parts: list[pl.DataFrame] = []
    for quantile, (upper, lower) in pairs.items():
        valid_column = _first_column(
            frame,
            (
                f"q{quantile}_adaptive_parameter_valid",
                f"adaptive_parameter_valid_q{quantile}",
                "adaptive_parameter_valid",
                "all_boundary_supported",
            ),
        )
        role_column = _first_column(
            frame,
            (
                f"q{quantile}_boundary_role",
                f"boundary_role_q{quantile}",
                "boundary_role",
            ),
        )
        parts.append(
            frame.select(
                *metadata,
                pl.lit(quantile).cast(pl.Int64).alias("boundary_quantile"),
                pl.col(upper).alias("upper_distance_bp"),
                pl.col(lower).alias("lower_distance_bp"),
                (
                    pl.col(valid_column) if valid_column is not None else pl.lit(True)
                ).alias("adaptive_parameter_valid"),
                (
                    pl.col(role_column)
                    if role_column is not None
                    else pl.lit(
                        "tail_diagnostic"
                        if quantile == 95
                        else "rolling_latent_candidate"
                    )
                ).alias("boundary_role"),
            )
        )
    return pl.concat(parts, how="diagonal_relaxed")


def _add_reference_geometry(
    policies: pl.DataFrame,
    profile: TransactionCostProfile,
    scenarios: tuple[float, ...],
    *,
    allow_zero_lower: bool,
) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    selected = policies.select(
        "Date",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
        "upper_distance_bp",
        "lower_distance_bp",
        "nominal_band_bp",
    )
    for row in selected.iter_rows(named=True):
        date = str(row["Date"])
        spot_ref = _positive(row["spot_ref_price"], "spot_ref_price")
        future_ref = _positive(row["fut_ref_price"], "fut_ref_price")
        shares = _positive(row["contract_size"], "contract_size")
        upper_distance = _positive(
            row["upper_distance_bp"],
            "upper_distance_bp",
        )
        lower_distance = (
            _nonnegative(row["lower_distance_bp"], "lower_distance_bp")
            if allow_zero_lower
            else _positive(row["lower_distance_bp"], "lower_distance_bp")
        )
        nominal_band = _positive(row["nominal_band_bp"], "nominal_band_bp")

        anchor = (future_ref / spot_ref - 1.0) * 10_000.0
        upper = anchor + upper_distance
        lower = anchor - lower_distance
        entry_target = target_price_for_basis(
            "spot_bid_future_taker",
            upper,
            session_date=date,
            fut_exec_bid=future_ref,
        )
        exit_target = target_price_for_basis(
            "spot_ask_future_taker",
            lower,
            session_date=date,
            fut_exec_ask=future_ref,
        )
        entry_tick = absolute_price_tick(
            entry_target,
            market="spot",
            session_date=date,
        )
        exit_tick = absolute_price_tick(
            exit_target,
            market="spot",
            session_date=date,
        )
        future_tick = absolute_price_tick(
            future_ref,
            market="future",
            session_date=date,
        )
        future_next = tick_index_to_price(
            future_tick + 1,
            market="future",
            session_date=date,
        )
        entry_effective = effective_basis_bp(
            "spot_bid_future_taker",
            entry_target,
            fut_exec_bid=future_ref,
        )
        exit_effective = effective_basis_bp(
            "spot_ask_future_taker",
            exit_target,
            fut_exec_ask=future_ref,
        )
        if entry_effective + GEOMETRY_EPS < upper:
            raise AssertionError("reference entry rounding violated upper threshold")
        if exit_effective > lower + GEOMETRY_EPS:
            raise AssertionError("reference exit rounding violated lower threshold")
        rounded_band = entry_effective - exit_effective
        if rounded_band + GEOMETRY_EPS < nominal_band:
            raise AssertionError("rounded reference band is below nominal band")

        normalization_notional = shares * spot_ref
        gross_twd = shares * (exit_target - entry_target)
        gross_bp = gross_twd / normalization_notional * 10_000.0
        same_day_cost = profile.paired_cycle_cost_twd(
            entry_spot_price=entry_target,
            exit_spot_price=exit_target,
            entry_future_price=future_ref,
            exit_future_price=future_ref,
            shares=shares,
            contracts=1.0,
            same_day=True,
        )
        overnight_cost = profile.paired_cycle_cost_twd(
            entry_spot_price=entry_target,
            exit_spot_price=exit_target,
            entry_future_price=future_ref,
            exit_future_price=future_ref,
            shares=shares,
            contracts=1.0,
            same_day=False,
        )
        same_day_cost_bp = same_day_cost / normalization_notional * 10_000.0
        overnight_cost_bp = overnight_cost / normalization_notional * 10_000.0
        same_day_margin = gross_twd - same_day_cost
        overnight_margin = gross_twd - overnight_cost
        record: dict[str, object] = {
            "reference_anchor_basis_bp": anchor,
            "reference_upper_basis_bp": upper,
            "reference_lower_basis_bp": lower,
            "reference_entry_spot_bid_target_price": entry_target,
            "reference_exit_spot_ask_target_price": exit_target,
            "reference_entry_absolute_spot_tick": entry_tick,
            "reference_exit_absolute_spot_tick": exit_tick,
            "reference_entry_effective_basis_bp": entry_effective,
            "reference_exit_effective_basis_bp": exit_effective,
            "reference_entry_rounding_excess_bp": entry_effective - upper,
            "reference_exit_rounding_excess_bp": lower - exit_effective,
            "rounded_reference_band_bp": rounded_band,
            "reference_rounding_total_excess_bp": rounded_band - nominal_band,
            "reference_entry_inside_ref_band": price_in_ref_band(
                entry_target,
                spot_ref,
            ),
            "reference_exit_inside_ref_band": price_in_ref_band(
                exit_target,
                spot_ref,
            ),
            "reference_future_next_tick_price": future_next,
            "reference_future_tick_bp": (
                (future_next - future_ref) / spot_ref * 10_000.0
            ),
            "reference_normalization_notional_twd": normalization_notional,
            "reference_gross_cycle_pnl_twd": gross_twd,
            "reference_gross_cycle_pnl_bp": gross_bp,
            "same_day_reference_cost_twd": same_day_cost,
            "same_day_reference_cost_bp": same_day_cost_bp,
            "overnight_reference_cost_twd": overnight_cost,
            "overnight_reference_cost_bp": overnight_cost_bp,
            # Keep the pre-rounding screen separate from the conservative
            # tick-rounded reference.  Outward rounding raises conditional
            # capture but can reduce touch/fill probability in S1.
            "nominal_same_day_known_cost_margin_bp": (nominal_band - same_day_cost_bp),
            "nominal_overnight_known_cost_margin_bp": (
                nominal_band - overnight_cost_bp
            ),
            "same_day_known_cost_margin_twd": same_day_margin,
            "same_day_known_cost_margin_bp": (
                same_day_margin / normalization_notional * 10_000.0
            ),
            "overnight_known_cost_margin_twd": overnight_margin,
            "overnight_known_cost_margin_bp": (
                overnight_margin / normalization_notional * 10_000.0
            ),
        }
        for adverse in scenarios:
            label = _scenario_label(adverse)
            adverse_twd = normalization_notional * adverse / 10_000.0
            record[f"same_day_margin_after_{label}bp_adverse_twd"] = (
                same_day_margin - adverse_twd
            )
            record[f"same_day_margin_after_{label}bp_adverse_bp"] = (
                same_day_margin / normalization_notional * 10_000.0 - adverse
            )
            record[f"overnight_margin_after_{label}bp_adverse_twd"] = (
                overnight_margin - adverse_twd
            )
            record[f"overnight_margin_after_{label}bp_adverse_bp"] = (
                overnight_margin / normalization_notional * 10_000.0 - adverse
            )
        records.append(record)

    derived = pl.from_dicts(records, infer_schema_length=None)
    return policies.hstack(derived).with_columns(
        pl.lit(FOUNDATION_GEOMETRY_VERSION).alias("geometry_version"),
        pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version"),
        pl.lit(FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE).alias(
            "future_one_dollar_tick_effective_date"
        ),
        pl.lit(profile.profile_id).alias("transaction_cost_profile_id"),
        pl.lit("opening_refs_only_no_intraday_bbo").alias(
            "reference_geometry_semantics"
        ),
        pl.lit(True).alias("reference_diagnostic"),
        pl.lit(False).alias("intraday_bbo_available"),
        pl.lit(False).alias("passive_clamp_applied"),
        pl.lit(False).alias("tt_band_deducted_as_cost"),
        pl.lit(False).alias("actionable_execution"),
        pl.lit(False).alias("ev_ready"),
    )


def _validate_d_safe(frame: pl.DataFrame, label: str) -> None:
    source_columns = [
        column
        for column in frame.columns
        if column == "source_asof_date" or column.endswith("_source_asof_date")
    ]
    if not source_columns:
        raise ValueError(f"{label} lacks source_asof_date lineage")
    _require(frame, {"Date", "contains_target_day_outcome"}, label)
    unsafe = pl.col("contains_target_day_outcome").fill_null(True)
    if "execution_safe_snapshot" in frame.columns:
        unsafe = unsafe | (~pl.col("execution_safe_snapshot").fill_null(False))
    for column in source_columns:
        unsafe = (
            unsafe
            | pl.col(column).is_null()
            | (pl.col(column).cast(pl.String) >= pl.col("Date").cast(pl.String))
        )
    if frame.filter(unsafe).height:
        raise ValueError(f"{label} is not strictly pre-open safe")


def _validate_common_boundary_coverage(
    cohort: pl.DataFrame,
    boundaries: pl.DataFrame,
) -> None:
    cohort_keys = cohort.select(list(KEYS))
    sample = boundaries.join(
        cohort_keys,
        on=list(KEYS),
        how="semi",
    )
    coverage = sample.group_by(list(KEYS)).agg(
        pl.col("boundary_quantile").n_unique().alias("quantiles"),
    )
    if (
        coverage.height != cohort.height
        or coverage.filter(pl.col("quantiles") != len(QUANTILE_POLICIES)).height
    ):
        raise ValueError("every cohort key must have common q50/q80/q95 coverage")


def _validate_policy_distances(
    frame: pl.DataFrame,
    *,
    allow_zero_lower: bool = False,
) -> None:
    invalid_lower = (
        pl.col("lower_distance_bp") < 0.0
        if allow_zero_lower
        else pl.col("lower_distance_bp") <= 0.0
    )
    if frame.filter(
        ~pl.col("upper_distance_bp").is_finite()
        | ~pl.col("lower_distance_bp").is_finite()
        | (pl.col("upper_distance_bp") <= 0)
        | invalid_lower
        | (
            (
                pl.col("nominal_band_bp")
                - pl.col("upper_distance_bp")
                - pl.col("lower_distance_bp")
            ).abs()
            > GEOMETRY_EPS
        )
    ).height:
        qualifier = "positive/nonnegative" if allow_zero_lower else "positive"
        raise ValueError(f"policy distances must be {qualifier} and additive")


def _validate_output_coverage(frame: pl.DataFrame, cohort_rows: int) -> None:
    expected = {spec.policy_id for spec in FOUNDATION_POLICY_SPECS}
    actual = set(frame["policy_id"].to_list())
    if actual != expected or frame.height != cohort_rows * len(expected):
        raise AssertionError("seven-policy common coverage invariant failed")
    counts = frame.group_by("policy_id").len()
    if counts.filter(pl.col("len") != cohort_rows).height:
        raise AssertionError("policy coverage differs across the common cohort")
    if frame.filter(
        pl.col("actionable_execution")
        | pl.col("ev_ready")
        | pl.col("tt_band_deducted_as_cost")
        | pl.col("passive_clamp_applied")
    ).height:
        raise AssertionError("reference geometry readiness flags are invalid")


def _normalise_adverse_scenarios(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("adverse sensitivity values must be nonempty and unique")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("adverse sensitivity values must be finite and positive")
    return result


def _scenario_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _first_column(frame: pl.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and positive")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _require(frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
