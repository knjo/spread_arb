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
from .foundation_geometry import (
    FOUNDATION_POLICY_SPECS,
    build_policy_geometry,
    summarize_policy_geometry,
)
from .transaction_costs import TransactionCostProfile

SELECTED_GEOMETRY_VERSION: Final = "foundation_selected_lookup_geometry_v1"
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
    "SELECTED_GEOMETRY_VERSION",
    "SelectedLookupGeometryResult",
    "build_selected_lookup_geometry",
]
