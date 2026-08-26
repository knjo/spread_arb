"""Known-cost reference geometry for the selected S0.5 lookup.

The rebuilt S0.5 lookup is indexed by product-day, time-of-day bucket,
quantile, and side, while :mod:`foundation_geometry` intentionally accepts
one q50/q80/q95 triplet per product-day.  This adapter pairs the selected
positive and negative lookup sides within each time-of-day bucket and calls
the existing seven-policy geometry builder one bucket at a time.

The publication remains a development-only reference diagnostic.  Opening
reference prices, the legal price ladder, and known fees/taxes are present;
intraday BBO, maker fills, taker hedge paths, and expected value are not.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import polars as pl

from .foundation_cohort_selection import (
    EXPECTED_BOUNDARY_ROWS,
    PRIMARY_QUANTILES,
    SIDES,
    TOD_BUCKETS,
)
from .foundation_convergence_lookup import (
    TARGET_LINEAGE_KEYS,
    validate_convergence_prediction_lineage,
)
from .foundation_geometry import (
    FOUNDATION_POLICY_SPECS,
    build_policy_geometry,
    build_reference_policy_geometry,
    summarize_policy_geometry,
)
from .transaction_costs import TransactionCostProfile

SELECTED_GEOMETRY_VERSION: Final = "foundation_selected_lookup_geometry_v1"
CONDITIONAL_GEOMETRY_VERSION: Final = "foundation_conditional_convergence_geometry_v2"
CONDITIONAL_CANDIDATES: Final = (
    "C0_center",
    "C2_conditional_reach80",
    "C3_conditional_reach50",
)
CONDITIONAL_PRIMARY_LOOKUP_ID: Final = "trail20_date_equal"
CONDITIONAL_FALLBACK_LOOKUP_ID: Final = "trail60_date_equal"
CONDITIONAL_POLICY_KIND: Final = "conditional_convergence_resolved"
PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
LOOKUP_CELL_KEYS: Final = (
    *PRODUCT_KEYS,
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
    "side",
)
PAIRED_BOUNDARY_KEYS: Final = (
    *PRODUCT_KEYS,
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
)
REQUIRED_MAPPING_METADATA: Final = (
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
)
EXPECTED_POLICIES: Final = len(FOUNDATION_POLICY_SPECS)


@dataclass(frozen=True)
class SelectedLookupGeometryResult:
    """Long geometry and non-EV summaries for the selected lookup."""

    geometry_long: pl.DataFrame
    summary_overall: pl.DataFrame
    summary_by_month: pl.DataFrame
    summary_by_tod: pl.DataFrame


@dataclass(frozen=True)
class ConditionalLookupGeometryResult:
    """Frozen-at-touch C0/C2/C3 support, geometry, and non-EV summaries."""

    support_audit: pl.DataFrame
    geometry_long: pl.DataFrame
    summary_overall: pl.DataFrame
    summary_by_month: pl.DataFrame
    summary_by_tod: pl.DataFrame
    support_summary: pl.DataFrame


def build_selected_lookup_geometry(
    s1_mother: pl.DataFrame,
    target_mapping: pl.DataFrame,
    selected_boundaries: pl.DataFrame,
    *,
    cost_profile: TransactionCostProfile | None = None,
    adverse_sensitivity_bp: Iterable[float] = (10.0, 20.0, 30.0),
) -> SelectedLookupGeometryResult:
    """Build seven policies for every primary product-day and TOD bucket.

    ``s1_mother`` may include failed funnel rows; only rows with
    ``s1_primary=True`` enter the common geometry support.  The selected
    boundary panel must provide exactly four TOD x three q x two side cells
    for each such product-day.  Positive-side distances become the reference
    upper threshold and negative-side distances become the reference lower
    threshold.
    """

    primary = _prepare_primary_mother(s1_mother)
    mapping = _prepare_target_mapping(target_mapping)
    cohort = _enrich_primary_cohort(primary, mapping)
    lookup = _prepare_selected_boundaries(selected_boundaries, primary)
    paired = _pair_lookup_sides(lookup)

    parts: list[pl.DataFrame] = []
    for tod_bucket in TOD_BUCKETS:
        bucket = paired.filter(pl.col("tod_bucket") == tod_bucket)
        if bucket.height != primary.height * len(PRIMARY_QUANTILES):
            raise ValueError(f"selected lookup lacks complete {tod_bucket} q coverage")
        parts.append(
            build_policy_geometry(
                cohort,
                bucket,
                cost_profile=cost_profile,
                adverse_sensitivity_bp=adverse_sensitivity_bp,
            )
        )

    geometry = (
        pl.concat(parts, how="diagonal_relaxed")
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("Date").cast(pl.String).str.slice(0, 6).alias("month"),
            pl.lit(SELECTED_GEOMETRY_VERSION).alias("selected_geometry_version"),
            pl.lit("s1_primary_product_day_tod_policy").alias("geometry_unit"),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit(True).alias("reference_diagnostic"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
        )
        .sort([*PRODUCT_KEYS, "tod_bucket", "policy_order"])
    )
    _validate_geometry(geometry, primary)

    summary_overall = _summarize_scope(
        geometry,
        scope="overall",
        value="all",
    )
    summary_by_month = pl.concat(
        [
            _summarize_scope(
                geometry.filter(pl.col("month") == month),
                scope="month",
                value=month,
            ).with_columns(pl.lit(month).alias("month"))
            for month in sorted(geometry["month"].unique().to_list())
        ],
        how="vertical_relaxed",
    )
    summary_by_tod = pl.concat(
        [
            _summarize_scope(
                geometry.filter(pl.col("tod_bucket") == tod_bucket),
                scope="tod_bucket",
                value=tod_bucket,
            ).with_columns(pl.lit(tod_bucket).alias("tod_bucket"))
            for tod_bucket in TOD_BUCKETS
        ],
        how="vertical_relaxed",
    )
    return SelectedLookupGeometryResult(
        geometry_long=geometry,
        summary_overall=summary_overall,
        summary_by_month=summary_by_month,
        summary_by_tod=summary_by_tod,
    )


def build_conditional_lookup_geometry(
    s1_mother: pl.DataFrame,
    target_mapping: pl.DataFrame,
    selected_boundaries: pl.DataFrame,
    convergence_effective_predictions: pl.DataFrame,
    *,
    cost_profile: TransactionCostProfile | None = None,
    adverse_sensitivity_bp: Iterable[float] = (10.0, 20.0, 30.0),
) -> ConditionalLookupGeometryResult:
    """Build frozen-at-touch C0/C2/C3 geometry on the complete S1 mother.

    The support audit retains every primary product-day x TOD x entry-q x
    conditional-candidate cell, including unsupported lookups.  Exact
    opening-reference geometry is built only where the frozen trail20 lookup
    or its explicit trail60 fallback is effective.  Unsupported q95 cells do
    not remove otherwise supported q50/q80 cells.
    """

    primary = _prepare_primary_mother(s1_mother)
    mapping = _prepare_target_mapping(target_mapping)
    lookup = _prepare_selected_boundaries(selected_boundaries, primary)
    predictions = _prepare_conditional_predictions(
        convergence_effective_predictions,
        primary,
    )
    entry = (
        lookup.filter(pl.col("side") == "positive")
        .select(
            *TARGET_LINEAGE_KEYS,
            pl.col("boundary_distance_bp").alias("upper_distance_bp"),
            pl.col("source_asof_date").alias("entry_source_asof_date"),
        )
        .sort(list(TARGET_LINEAGE_KEYS))
    )
    prediction = predictions.rename(
        {
            "source_asof_date": "convergence_native_source_asof_date",
            "contains_target_day_outcome": ("convergence_contains_target_day_outcome"),
        }
    )
    support = (
        entry.join(
            prediction,
            on=list(TARGET_LINEAGE_KEYS),
            how="inner",
            validate="1:m",
        )
        .with_columns(
            pl.col("effective_supported")
            .fill_null(False)
            .alias("conditional_supported"),
            pl.when(pl.col("effective_supported").fill_null(False))
            .then(
                pl.max_horizontal(
                    "entry_source_asof_date",
                    "effective_source_asof_date",
                )
            )
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("source_asof_date"),
            pl.when(pl.col("effective_supported").fill_null(False))
            .then(pl.lit("supported"))
            .otherwise(
                pl.coalesce(
                    "fallback_reason",
                    "native_failure_reason",
                    pl.lit("unsupported"),
                )
            )
            .alias("conditional_support_reason"),
            pl.concat_str(
                pl.lit("q"),
                pl.col("boundary_quantile").cast(pl.String),
                pl.lit("__"),
                pl.col("convergence_candidate_id"),
            ).alias("policy_id"),
            pl.lit(CONDITIONAL_POLICY_KIND).alias("policy_kind"),
            (
                pl.col("boundary_quantile").replace_strict({50: 0, 80: 3, 95: 6})
                + pl.col("convergence_candidate_id").replace_strict(
                    {
                        "C0_center": 0,
                        "C2_conditional_reach80": 1,
                        "C3_conditional_reach50": 2,
                    }
                )
            )
            .cast(pl.Int64)
            .alias("policy_order"),
            pl.lit(True).alias("s1_primary"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.col("Date").str.slice(0, 6).alias("month"),
            pl.lit(CONDITIONAL_GEOMETRY_VERSION).alias("conditional_geometry_version"),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit("frozen_anchor_at_upper_touch").alias(
                "exit_reference_semantics"
            ),
            pl.lit("touch_anchor_basis_bp-lower_distance_bp").alias(
                "frozen_exit_basis_formula"
            ),
        )
        .sort(
            [
                *PRODUCT_KEYS,
                "tod_bucket",
                "boundary_quantile",
                "policy_order",
            ]
        )
    )
    _validate_conditional_support(support, primary)

    references = mapping.select(*PRODUCT_KEYS, *REQUIRED_MAPPING_METADATA)
    policies = (
        support.filter(pl.col("conditional_supported"))
        .join(
            references,
            on=list(PRODUCT_KEYS),
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("effective_threshold_distance_bp").alias("lower_distance_bp"),
            (
                pl.col("upper_distance_bp") + pl.col("effective_threshold_distance_bp")
            ).alias("nominal_band_bp"),
            pl.lit(True).alias("execution_safe_snapshot"),
            pl.lit(True).alias("adaptive_parameter_valid"),
            pl.lit("conditional_post_entry_exit").alias("boundary_role"),
        )
    )
    geometry = (
        build_reference_policy_geometry(
            policies,
            cost_profile=cost_profile,
            adverse_sensitivity_bp=adverse_sensitivity_bp,
            allow_zero_lower=True,
        )
        .with_columns(
            pl.lit(CONDITIONAL_GEOMETRY_VERSION).alias("conditional_geometry_version"),
            pl.lit("s1_primary_product_day_tod_entry_q_convergence_candidate").alias(
                "geometry_unit"
            ),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit("frozen_anchor_at_upper_touch").alias(
                "exit_reference_semantics"
            ),
            pl.lit("touch_anchor_basis_bp-lower_distance_bp").alias(
                "frozen_exit_basis_formula"
            ),
        )
        .sort(
            [
                *PRODUCT_KEYS,
                "tod_bucket",
                "boundary_quantile",
                "policy_order",
            ]
        )
    )
    _validate_conditional_geometry(geometry, support)
    summary_overall = _summarize_conditional_scope(
        geometry,
        support,
        scope="overall",
        value="all",
    )
    summary_by_month = pl.concat(
        [
            _summarize_conditional_scope(
                geometry.filter(pl.col("month") == month),
                support.filter(pl.col("month") == month),
                scope="month",
                value=month,
            ).with_columns(pl.lit(month).alias("month"))
            for month in sorted(support["month"].unique().to_list())
        ],
        how="vertical_relaxed",
    )
    summary_by_tod = pl.concat(
        [
            _summarize_conditional_scope(
                geometry.filter(pl.col("tod_bucket") == tod_bucket),
                support.filter(pl.col("tod_bucket") == tod_bucket),
                scope="tod_bucket",
                value=tod_bucket,
            ).with_columns(pl.lit(tod_bucket).alias("tod_bucket"))
            for tod_bucket in TOD_BUCKETS
        ],
        how="vertical_relaxed",
    )
    support_summary = pl.concat(
        [
            _summarize_conditional_support(
                support,
                scope="overall",
                value="all",
            ),
            *[
                _summarize_conditional_support(
                    support.filter(pl.col("month") == month),
                    scope="month",
                    value=month,
                )
                for month in sorted(support["month"].unique().to_list())
            ],
            *[
                _summarize_conditional_support(
                    support.filter(pl.col("tod_bucket") == tod_bucket),
                    scope="tod_bucket",
                    value=tod_bucket,
                )
                for tod_bucket in TOD_BUCKETS
            ],
        ],
        how="vertical_relaxed",
    )
    return ConditionalLookupGeometryResult(
        support_audit=support,
        geometry_long=geometry,
        summary_overall=summary_overall,
        summary_by_month=summary_by_month,
        summary_by_tod=summary_by_tod,
        support_summary=support_summary,
    )


def _prepare_primary_mother(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        *PRODUCT_KEYS,
        "anchor_model_id",
        "candidate_id",
        "s1_primary",
        "contains_target_day_outcome",
    }
    _require(frame, required, "S1 mother")
    primary = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("s1_primary").fill_null(False).cast(pl.Boolean),
        pl.col("contains_target_day_outcome").fill_null(True).cast(pl.Boolean),
    ).filter(pl.col("s1_primary"))
    if primary.is_empty():
        raise ValueError("S1 mother has no primary product-days")
    _validate_unique(primary, PRODUCT_KEYS, "S1 primary mother")
    if primary["contains_target_day_outcome"].any():
        raise ValueError("S1 primary mother contains target-day outcomes")
    if primary["anchor_model_id"].n_unique() != 1:
        raise ValueError("S1 primary mother must have one selected anchor")
    if primary["candidate_id"].n_unique() != 1:
        raise ValueError("S1 primary mother must have one selected q candidate")
    _validate_source_dates(primary, "S1 primary mother", require_lineage=True)
    return primary.sort(list(PRODUCT_KEYS))


def _prepare_target_mapping(frame: pl.DataFrame) -> pl.DataFrame:
    _require(
        frame,
        {*PRODUCT_KEYS, *REQUIRED_MAPPING_METADATA},
        "target mapping",
    )
    mapping = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        *(pl.col(column).cast(pl.Float64) for column in REQUIRED_MAPPING_METADATA),
    )
    _validate_unique(mapping, PRODUCT_KEYS, "target mapping")
    invalid = pl.any_horizontal(
        *(
            pl.col(column).is_null()
            | ~pl.col(column).is_finite()
            | (pl.col(column) <= 0.0)
            for column in REQUIRED_MAPPING_METADATA
        )
    )
    if mapping.filter(invalid).height:
        raise ValueError("target mapping reference metadata must be positive")
    return mapping


def _enrich_primary_cohort(
    primary: pl.DataFrame,
    mapping: pl.DataFrame,
) -> pl.DataFrame:
    overlap = (set(primary.columns) & set(mapping.columns)) - set(PRODUCT_KEYS)
    if overlap:
        raise ValueError(
            f"S1 mother and target mapping have ambiguous columns: {sorted(overlap)}"
        )
    cohort = primary.join(
        mapping,
        on=list(PRODUCT_KEYS),
        how="left",
        validate="1:1",
    )
    missing = pl.any_horizontal(
        *(pl.col(column).is_null() for column in REQUIRED_MAPPING_METADATA)
    )
    if cohort.filter(missing).height:
        raise ValueError("target mapping lacks S1 primary product-days")
    if cohort.height != primary.height:
        raise AssertionError("target mapping changed S1 primary row count")
    return cohort


def _prepare_selected_boundaries(
    frame: pl.DataFrame,
    primary: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        *LOOKUP_CELL_KEYS,
        "boundary_distance_bp",
        "source_asof_date",
        "contains_target_day_outcome",
    }
    _require(frame, required, "selected boundaries")
    lookup = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("side").cast(pl.String),
        pl.col("boundary_distance_bp").cast(pl.Float64),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("contains_target_day_outcome").fill_null(True).cast(pl.Boolean),
    )
    identities = primary.select(
        *PRODUCT_KEYS,
        "anchor_model_id",
        "candidate_id",
    )
    lookup = lookup.join(
        identities,
        on=[*PRODUCT_KEYS, "anchor_model_id", "candidate_id"],
        how="semi",
    )
    if "effective_supported" in lookup.columns:
        effective = pl.col("effective_supported").fill_null(False)
    elif "native_supported" in lookup.columns:
        effective = pl.col("native_supported").fill_null(False)
    else:
        effective = pl.lit(True)
    lookup = lookup.with_columns(effective.cast(pl.Boolean).alias("_effective"))
    _validate_unique(lookup, LOOKUP_CELL_KEYS, "selected boundaries")
    invalid = lookup.filter(
        ~pl.col("tod_bucket").is_in(TOD_BUCKETS)
        | ~pl.col("boundary_quantile").is_in(PRIMARY_QUANTILES)
        | ~pl.col("side").is_in(SIDES)
        | ~pl.col("_effective")
        | pl.col("boundary_distance_bp").is_null()
        | ~pl.col("boundary_distance_bp").is_finite()
        | (pl.col("boundary_distance_bp") <= 0.0)
        | pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
        | pl.col("contains_target_day_outcome")
    )
    if invalid.height:
        raise ValueError("selected boundaries contain invalid or unsafe cells")
    support = lookup.group_by(list(PRODUCT_KEYS)).agg(
        pl.len().alias("rows"),
        pl.col("tod_bucket").n_unique().alias("tod_buckets"),
        pl.col("boundary_quantile").n_unique().alias("quantiles"),
        pl.col("side").n_unique().alias("sides"),
    )
    incomplete = support.filter(
        (pl.col("rows") != EXPECTED_BOUNDARY_ROWS)
        | (pl.col("tod_buckets") != len(TOD_BUCKETS))
        | (pl.col("quantiles") != len(PRIMARY_QUANTILES))
        | (pl.col("sides") != len(SIDES))
    )
    if support.height != primary.height or incomplete.height:
        raise ValueError("selected boundaries are not complete 24-cell lookups")
    return lookup.sort(list(LOOKUP_CELL_KEYS))


def _pair_lookup_sides(frame: pl.DataFrame) -> pl.DataFrame:
    upper = frame.filter(pl.col("side") == "positive").select(
        *PAIRED_BOUNDARY_KEYS,
        pl.col("boundary_distance_bp").alias("upper_distance_bp"),
        pl.col("source_asof_date").alias("upper_source_asof_date"),
    )
    lower = frame.filter(pl.col("side") == "negative").select(
        *PAIRED_BOUNDARY_KEYS,
        pl.col("boundary_distance_bp").alias("lower_distance_bp"),
        pl.col("source_asof_date").alias("lower_source_asof_date"),
    )
    paired = upper.join(
        lower,
        on=list(PAIRED_BOUNDARY_KEYS),
        how="inner",
        validate="1:1",
    ).with_columns(
        pl.max_horizontal(
            "upper_source_asof_date",
            "lower_source_asof_date",
        ).alias("source_asof_date"),
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
        pl.lit(True).alias("adaptive_parameter_valid"),
        pl.lit("selected_s05_lookup").alias("boundary_role"),
    )
    if paired.height * len(SIDES) != frame.height:
        raise AssertionError("selected boundary side pairing changed coverage")
    _validate_source_dates(paired, "paired selected boundaries", require_lineage=True)
    return paired.sort(list(PAIRED_BOUNDARY_KEYS))


def _summarize_scope(
    frame: pl.DataFrame,
    *,
    scope: str,
    value: str,
) -> pl.DataFrame:
    summary = summarize_policy_geometry(frame).rename(
        {"product_days": "product_day_tod_rows"}
    )
    coverage = frame.group_by(["policy_id", "policy_kind", "policy_order"]).agg(
        pl.struct(list(PRODUCT_KEYS)).n_unique().alias("product_days"),
        pl.col("tod_bucket").n_unique().alias("tod_buckets"),
        pl.col("month").n_unique().alias("months"),
        pl.col("source_asof_date").min().alias("source_asof_start_date"),
        pl.col("source_asof_date").max().alias("source_asof_end_date"),
    )
    anchor_model_id = _one_value(frame, "anchor_model_id")
    candidate_id = _one_value(frame, "candidate_id")
    return (
        summary.join(
            coverage,
            on=["policy_id", "policy_kind", "policy_order"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            pl.lit(scope).alias("summary_scope"),
            pl.lit(value).alias("summary_value"),
            pl.lit(anchor_model_id).alias("anchor_model_id"),
            pl.lit(candidate_id).alias("candidate_id"),
            pl.lit(SELECTED_GEOMETRY_VERSION).alias("selected_geometry_version"),
            pl.lit("product_day_tod_rows").alias("summary_weighting_unit"),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit(True).alias("reference_diagnostic"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
        )
        .sort("policy_order")
    )


def _prepare_conditional_predictions(
    frame: pl.DataFrame,
    primary: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        *TARGET_LINEAGE_KEYS,
        "convergence_candidate_id",
        "convergence_reference_semantics",
        "lookup_id",
        "target_reach_probability",
        "source_asof_date",
        "native_threshold_distance_bp",
        "native_supported",
        "native_failure_reason",
        "effective_threshold_distance_bp",
        "effective_supported",
        "effective_lookup_id",
        "effective_source_asof_date",
        "fallback_lookup_id",
        "fallback_used",
        "fallback_reason",
        "contains_target_day_outcome",
    }
    _require(frame, required, "effective convergence predictions")
    identities = primary.select(
        *PRODUCT_KEYS,
        "anchor_model_id",
        "candidate_id",
    )
    result = (
        frame.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("anchor_model_id").cast(pl.String),
            pl.col("candidate_id").cast(pl.String),
            pl.col("tod_bucket").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("convergence_candidate_id").cast(pl.String),
            pl.col("lookup_id").cast(pl.String),
            pl.col("effective_supported").fill_null(False).cast(pl.Boolean),
            pl.col("fallback_used").fill_null(False).cast(pl.Boolean),
            pl.col("contains_target_day_outcome").fill_null(True).cast(pl.Boolean),
        )
        .join(
            identities,
            on=[*PRODUCT_KEYS, "anchor_model_id", "candidate_id"],
            how="semi",
        )
        .filter(pl.col("convergence_candidate_id").is_in(CONDITIONAL_CANDIDATES))
    )
    if result.is_empty():
        raise ValueError("effective convergence predictions have no C0/C2/C3 rows")
    validate_convergence_prediction_lineage(result)
    key = [*TARGET_LINEAGE_KEYS, "convergence_candidate_id"]
    _validate_unique(result, key, "effective C2/C3 predictions")
    expected_rows = (
        primary.height
        * len(TOD_BUCKETS)
        * len(PRIMARY_QUANTILES)
        * len(CONDITIONAL_CANDIDATES)
    )
    if result.height != expected_rows:
        raise ValueError("effective C0/C2/C3 predictions do not cover the full mother")
    if set(result["convergence_candidate_id"].unique().to_list()) != set(
        CONDITIONAL_CANDIDATES
    ):
        raise ValueError("effective convergence predictions lack C0, C2, or C3")
    _validate_resolved_prediction_state(result)
    is_c0 = pl.col("convergence_candidate_id") == "C0_center"
    invalid_route = result.filter(
        (
            is_c0
            & (
                (pl.col("lookup_id") != "structural_zero")
                | ~pl.col("effective_supported")
                | (pl.col("effective_lookup_id") != "structural_zero")
                | (pl.col("effective_threshold_distance_bp").abs() > 1e-12)
                | pl.col("fallback_lookup_id").is_not_null()
                | pl.col("fallback_used")
            )
        )
        | (
            ~is_c0
            & (
                (pl.col("lookup_id") != CONDITIONAL_PRIMARY_LOOKUP_ID)
                | (pl.col("fallback_lookup_id") != CONDITIONAL_FALLBACK_LOOKUP_ID)
                | (
                    pl.col("effective_supported")
                    & ~pl.col("effective_lookup_id").is_in(
                        [
                            CONDITIONAL_PRIMARY_LOOKUP_ID,
                            CONDITIONAL_FALLBACK_LOOKUP_ID,
                        ]
                    )
                )
                | (
                    pl.col("fallback_used")
                    & (
                        pl.col("effective_lookup_id")
                        != CONDITIONAL_FALLBACK_LOOKUP_ID
                    )
                )
                | (
                    pl.col("effective_supported")
                    & ~pl.col("fallback_used")
                    & (
                        pl.col("effective_lookup_id")
                        != CONDITIONAL_PRIMARY_LOOKUP_ID
                    )
                )
            )
        )
    )
    if invalid_route.height:
        raise ValueError(
            "C0/C2/C3 predictions do not follow structural/resolved routes"
        )
    reach_error = result.filter(
        (
            (pl.col("convergence_candidate_id") == "C2_conditional_reach80")
            & ((pl.col("target_reach_probability") - 0.80).abs() > 1e-12)
        )
        | (
            (pl.col("convergence_candidate_id") == "C3_conditional_reach50")
            & ((pl.col("target_reach_probability") - 0.50).abs() > 1e-12)
        )
    )
    if reach_error.height:
        raise ValueError("C2/C3 target reach probabilities have drifted")
    return result.sort(key)


def _validate_resolved_prediction_state(frame: pl.DataFrame) -> None:
    """Fail closed on structural-C0 and native-first fallback state drift."""

    for row in frame.iter_rows(named=True):
        candidate = str(row["convergence_candidate_id"])
        native = bool(row["native_supported"])
        effective = bool(row["effective_supported"])
        fallback = bool(row["fallback_used"])
        native_distance = row["native_threshold_distance_bp"]
        effective_distance = row["effective_threshold_distance_bp"]
        if candidate == "C0_center":
            valid = (
                row["lookup_id"] == "structural_zero"
                and native
                and effective
                and not fallback
                and row["effective_lookup_id"] == "structural_zero"
                and row["fallback_lookup_id"] is None
                and native_distance is not None
                and effective_distance is not None
                and abs(float(native_distance)) <= 1e-12
                and abs(float(effective_distance)) <= 1e-12
                and row["source_asof_date"] == row["effective_source_asof_date"]
            )
            if not valid:
                raise ValueError("C0 structural prediction state is inconsistent")
            continue

        if candidate not in {
            "C2_conditional_reach80",
            "C3_conditional_reach50",
        }:
            raise ValueError(f"unexpected conditional candidate: {candidate}")
        if row["lookup_id"] != CONDITIONAL_PRIMARY_LOOKUP_ID or (
            row["fallback_lookup_id"] != CONDITIONAL_FALLBACK_LOOKUP_ID
        ):
            raise ValueError("conditional lookup route is inconsistent")
        if native:
            valid = (
                effective
                and not fallback
                and row["effective_lookup_id"] == CONDITIONAL_PRIMARY_LOOKUP_ID
                and native_distance is not None
                and effective_distance is not None
                and abs(float(native_distance) - float(effective_distance)) <= 1e-12
                and row["source_asof_date"] == row["effective_source_asof_date"]
                and row["fallback_reason"] is None
            )
        elif fallback:
            valid = (
                effective
                and row["effective_lookup_id"] == CONDITIONAL_FALLBACK_LOOKUP_ID
                and effective_distance is not None
                and row["effective_source_asof_date"] is not None
                and row["fallback_reason"] is not None
            )
        else:
            valid = (
                not effective
                and effective_distance is None
                and row["effective_lookup_id"] is None
                and row["effective_source_asof_date"] is None
                and row["fallback_reason"] is not None
            )
        if not valid:
            raise ValueError("conditional native/fallback/effective state is inconsistent")


def _conditional_policy_groups() -> list[str]:
    return [
        "policy_id",
        "policy_kind",
        "policy_order",
        "boundary_quantile",
        "convergence_candidate_id",
    ]


def _summarize_conditional_scope(
    geometry: pl.DataFrame,
    support: pl.DataFrame,
    *,
    scope: str,
    value: str,
) -> pl.DataFrame:
    metrics = summarize_policy_geometry(geometry).rename(
        {"product_days": "product_day_tod_rows"}
    )
    groups = _conditional_policy_groups()
    policy_metadata = support.select(groups).unique()
    coverage = geometry.group_by(groups).agg(
        pl.struct(list(PRODUCT_KEYS)).n_unique().alias("product_days"),
        pl.col("tod_bucket").n_unique().alias("tod_buckets"),
        pl.col("month").n_unique().alias("months"),
        pl.col("source_asof_date").min().alias("source_asof_start_date"),
        pl.col("source_asof_date").max().alias("source_asof_end_date"),
    )
    support_counts = _conditional_support_counts(support)
    anchor_model_id = _one_value(support, "anchor_model_id")
    candidate_id = _one_value(support, "candidate_id")
    return (
        policy_metadata.join(
            support_counts,
            on=groups,
            how="inner",
            validate="1:1",
        )
        .join(
            metrics,
            on=["policy_id", "policy_kind", "policy_order"],
            how="left",
            validate="1:1",
        )
        .join(coverage, on=groups, how="left", validate="1:1")
        .with_columns(
            pl.lit(scope).alias("summary_scope"),
            pl.lit(value).alias("summary_value"),
            pl.lit(anchor_model_id).alias("anchor_model_id"),
            pl.lit(candidate_id).alias("candidate_id"),
            pl.lit(CONDITIONAL_GEOMETRY_VERSION).alias("conditional_geometry_version"),
            pl.lit("supported_product_day_tod_cells").alias("summary_weighting_unit"),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit(True).alias("reference_diagnostic"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
        )
        .sort("policy_order")
    )


def _conditional_support_counts(frame: pl.DataFrame) -> pl.DataFrame:
    groups = _conditional_policy_groups()
    return (
        frame.group_by(groups)
        .agg(
            pl.len().alias("mother_cell_rows"),
            pl.col("conditional_supported").sum().alias("supported_cell_rows"),
            (pl.col("conditional_supported") & pl.col("fallback_used"))
            .sum()
            .alias("fallback_supported_rows"),
            (
                pl.col("conditional_supported")
                & (pl.col("effective_threshold_distance_bp").abs() <= 1e-12)
            )
            .sum()
            .alias("zero_lower_supported_rows"),
        )
        .with_columns(
            (pl.col("supported_cell_rows") / pl.col("mother_cell_rows")).alias(
                "conditional_support_coverage"
            ),
            pl.when(pl.col("supported_cell_rows") > 0)
            .then(pl.col("fallback_supported_rows") / pl.col("supported_cell_rows"))
            .otherwise(None)
            .alias("fallback_share_supported"),
        )
    )


def _summarize_conditional_support(
    frame: pl.DataFrame,
    *,
    scope: str,
    value: str,
) -> pl.DataFrame:
    anchor_model_id = _one_value(frame, "anchor_model_id")
    candidate_id = _one_value(frame, "candidate_id")
    return (
        _conditional_support_counts(frame)
        .with_columns(
            pl.lit(scope).alias("summary_scope"),
            pl.lit(value).alias("summary_value"),
            pl.lit(anchor_model_id).alias("anchor_model_id"),
            pl.lit(candidate_id).alias("candidate_id"),
            pl.lit(CONDITIONAL_GEOMETRY_VERSION).alias("conditional_geometry_version"),
            pl.lit(True).alias("development_only"),
            pl.lit(True).alias("selection_contains_development_outcomes"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
        )
        .sort("policy_order")
    )


def _validate_conditional_support(
    frame: pl.DataFrame,
    primary: pl.DataFrame,
) -> None:
    expected_rows = (
        primary.height
        * len(TOD_BUCKETS)
        * len(PRIMARY_QUANTILES)
        * len(CONDITIONAL_CANDIDATES)
    )
    if frame.height != expected_rows:
        raise AssertionError("conditional support audit changed the S1 mother grid")
    key = [*TARGET_LINEAGE_KEYS, "convergence_candidate_id"]
    _validate_unique(frame, key, "conditional support audit")
    expected_policies = len(PRIMARY_QUANTILES) * len(CONDITIONAL_CANDIDATES)
    coverage = frame.group_by([*PRODUCT_KEYS, "tod_bucket"]).agg(
        pl.len().alias("rows"),
        pl.col("policy_id").n_unique().alias("policies"),
    )
    if (
        coverage.height != primary.height * len(TOD_BUCKETS)
        or coverage.filter(
            (pl.col("rows") != expected_policies)
            | (pl.col("policies") != expected_policies)
        ).height
    ):
        raise AssertionError("conditional support audit lacks a mother TOD cell")
    if frame.filter(
        ~pl.col("s1_primary")
        | pl.col("contains_target_day_outcome")
        | pl.col("convergence_contains_target_day_outcome")
        | (
            pl.col("convergence_reference_semantics")
            != "frozen_anchor_at_upper_touch"
        )
        | (pl.col("exit_reference_semantics") != "frozen_anchor_at_upper_touch")
        | (pl.col("conditional_supported") != pl.col("effective_supported"))
        | pl.col("entry_source_asof_date").is_null()
        | (pl.col("entry_source_asof_date") >= pl.col("Date"))
        | (
            pl.col("conditional_supported")
            & (
                pl.col("source_asof_date").is_null()
                | (pl.col("source_asof_date") >= pl.col("Date"))
                | pl.col("effective_threshold_distance_bp").is_null()
                | ~pl.col("effective_threshold_distance_bp").is_finite()
                | (pl.col("effective_threshold_distance_bp") < 0.0)
            )
        )
    ).height:
        raise ValueError("conditional support audit contains unsafe cells")


def _validate_conditional_geometry(
    frame: pl.DataFrame,
    support: pl.DataFrame,
) -> None:
    expected_rows = int(support["conditional_supported"].sum())
    if frame.height != expected_rows:
        raise AssertionError("conditional geometry differs from effective support")
    key = [*TARGET_LINEAGE_KEYS, "convergence_candidate_id"]
    _validate_unique(frame, key, "conditional geometry")
    expected_policy_ids = set(
        support.filter(pl.col("conditional_supported"))["policy_id"]
        .unique()
        .to_list()
    )
    if set(frame["policy_id"].unique().to_list()) != expected_policy_ids:
        raise AssertionError("conditional geometry lacks a supported policy")
    if frame.filter(
        ~pl.col("conditional_supported")
        | ~pl.col("effective_supported")
        | (pl.col("upper_distance_bp") <= 0.0)
        | (pl.col("lower_distance_bp") < 0.0)
        | ~pl.col("development_only")
        | ~pl.col("selection_contains_development_outcomes")
        | pl.col("contains_target_day_outcome")
        | (pl.col("exit_reference_semantics") != "frozen_anchor_at_upper_touch")
        | (
            pl.col("frozen_exit_basis_formula")
            != "touch_anchor_basis_bp-lower_distance_bp"
        )
        | ~pl.col("reference_diagnostic")
        | pl.col("actionable_execution")
        | pl.col("ev_ready")
    ).height:
        raise AssertionError("conditional geometry readiness labels are invalid")
    _validate_source_dates(frame, "conditional geometry", require_lineage=True)


def _validate_geometry(frame: pl.DataFrame, primary: pl.DataFrame) -> None:
    expected_rows = primary.height * len(TOD_BUCKETS) * EXPECTED_POLICIES
    if frame.height != expected_rows:
        raise AssertionError("selected geometry row count violates common support")
    keys = [*PRODUCT_KEYS, "tod_bucket", "policy_id"]
    _validate_unique(frame, keys, "selected lookup geometry")
    support = frame.group_by([*PRODUCT_KEYS, "tod_bucket"]).agg(
        pl.col("policy_id").n_unique().alias("policies")
    )
    if (
        support.height != primary.height * len(TOD_BUCKETS)
        or support.filter(pl.col("policies") != EXPECTED_POLICIES).height
    ):
        raise AssertionError("seven-policy TOD coverage differs across the mother")
    if set(frame["tod_bucket"].unique().to_list()) != set(TOD_BUCKETS):
        raise AssertionError("selected geometry lacks a frozen TOD bucket")
    if frame.filter(
        ~pl.col("s1_primary")
        | ~pl.col("development_only")
        | ~pl.col("selection_contains_development_outcomes")
        | pl.col("contains_target_day_outcome")
        | ~pl.col("reference_diagnostic")
        | pl.col("actionable_execution")
        | pl.col("ev_ready")
    ).height:
        raise AssertionError("selected geometry readiness labels are invalid")
    _validate_source_dates(frame, "selected lookup geometry", require_lineage=True)


def _validate_source_dates(
    frame: pl.DataFrame,
    source: str,
    *,
    require_lineage: bool,
) -> None:
    columns = [
        column
        for column in frame.columns
        if column == "source_asof_date" or column.endswith("_source_asof_date")
    ]
    if require_lineage and not columns:
        raise ValueError(f"{source} lacks source-asof lineage")
    for column in columns:
        if frame.filter(
            pl.col(column).is_null()
            | (pl.col(column).cast(pl.String) >= pl.col("Date"))
        ).height:
            raise ValueError(f"{source} has unsafe {column}")


def _validate_unique(
    frame: pl.DataFrame,
    keys: tuple[str, ...] | list[str],
    source: str,
) -> None:
    if frame.select(list(keys)).n_unique() != frame.height:
        raise ValueError(f"{source} contains duplicate keys")


def _one_value(frame: pl.DataFrame, column: str) -> str:
    values = frame[column].drop_nulls().unique().to_list()
    if len(values) != 1:
        raise ValueError(f"geometry scope requires exactly one {column}")
    return str(values[0])


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


__all__ = [
    "CONDITIONAL_CANDIDATES",
    "CONDITIONAL_GEOMETRY_VERSION",
    "SELECTED_GEOMETRY_VERSION",
    "ConditionalLookupGeometryResult",
    "SelectedLookupGeometryResult",
    "build_conditional_lookup_geometry",
    "build_selected_lookup_geometry",
]
