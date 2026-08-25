"""Outcome-free S1 mother-cohort construction for the rebuilt S0.5 study.

The mother cohort is deliberately assembled from three independent facts:

* a causal selected anchor exists during the target session;
* the selected lookup has every effective q/side/TOD cell before that session;
* the existing liquidity publication passes only its q-independent history and
  hard-data gates.

Legacy ``boundary_parameter_gate``, ``support_gate`` and
``pre_replay_candidate`` columns may be present in the liquidity source, but
they are neither read nor copied.  This prevents the incumbent q table from
silently selecting the universe for a newly rebuilt lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
PRIMARY_QUANTILES: Final = (50, 80, 95)
SIDES: Final = ("positive", "negative")
TOD_BUCKETS: Final = (
    "0905_1000",
    "1000_1100",
    "1100_1200",
    "1200_1300",
)
ENTRY_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
SPOT_BID_ROUTE: Final = "spot_bid_future_taker"
EXPECTED_BOUNDARY_ROWS: Final = len(PRIMARY_QUANTILES) * len(SIDES) * len(TOD_BUCKETS)
FORBIDDEN_OUTPUT_FIELDS: Final = frozenset(
    {
        "boundary_parameter_gate",
        "support_gate",
        "pre_replay_candidate",
        "latent_band_to_tt_band_ratio",
        "entry_upper_to_tt_band_ratio",
    }
)


@dataclass(frozen=True)
class CohortSelectionResult:
    """Published mother rows and their auditable funnel."""

    mother: pl.DataFrame
    funnel: pl.DataFrame
    liquidity_invariance_audit: pl.DataFrame


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _normalise_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )


def _validate_unique(frame: pl.DataFrame, keys: Sequence[str], source: str) -> None:
    duplicate = frame.group_by(list(keys)).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError(f"{source} contains duplicate keys")


def build_anchor_product_day_support(
    day: pl.DataFrame,
    *,
    anchor_column: str,
    anchor_model_id: str,
) -> pl.DataFrame:
    """Summarize target-day causal anchor availability without a pass threshold.

    A product-day is supported when at least one analysis-eligible second in
    the frozen entry window has a finite selected anchor.  Coverage is retained
    for diagnostics; it is not converted into an outcome-tuned cutoff.
    """

    if not str(anchor_model_id).strip():
        raise ValueError("anchor_model_id must not be empty")
    _require(
        day,
        {
            *PRODUCT_KEYS,
            "seconds_from_open",
            "analysis_eligible",
            anchor_column,
        },
        "causal anchor day",
    )
    frame = _normalise_keys(day).with_columns(
        pl.col("seconds_from_open").cast(pl.Int32),
        pl.col("analysis_eligible").fill_null(False).cast(pl.Boolean),
        pl.col(anchor_column).cast(pl.Float64).alias("_anchor_bp"),
    )
    window = frame.filter(
        (pl.col("seconds_from_open") >= ENTRY_START_SECOND)
        & (pl.col("seconds_from_open") < ENTRY_STOP_SECOND)
    ).with_columns(
        (
            pl.col("analysis_eligible")
            & pl.col("_anchor_bp").is_not_null()
            & pl.col("_anchor_bp").is_finite()
        ).alias("_anchor_eligible")
    )
    if window.is_empty():
        raise ValueError("causal anchor day has no frozen entry-window rows")
    result = (
        window.group_by(list(PRODUCT_KEYS))
        .agg(
            pl.col("analysis_eligible").sum().alias("analysis_eligible_seconds"),
            pl.col("_anchor_eligible").sum().alias("selected_anchor_native_seconds"),
        )
        .with_columns(
            pl.lit(str(anchor_model_id)).alias("anchor_model_id"),
            (pl.col("selected_anchor_native_seconds") > 0).alias(
                "selected_anchor_support"
            ),
            pl.when(pl.col("analysis_eligible_seconds") > 0)
            .then(
                pl.col("selected_anchor_native_seconds")
                / pl.col("analysis_eligible_seconds")
            )
            .otherwise(None)
            .alias("selected_anchor_coverage_rate"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
        .sort(list(PRODUCT_KEYS))
    )
    return result


def build_s1_mother_cohort(
    target_mapping: pl.DataFrame,
    anchor_support: pl.DataFrame,
    selected_boundaries: pl.DataFrame,
    liquidity: pl.DataFrame,
    *,
    selected_anchor_model_id: str,
    selected_candidate_id: str,
    primary_dates: Sequence[str] | None = None,
    route: str = SPOT_BID_ROUTE,
    boundary_column: str = "boundary_distance_bp",
) -> CohortSelectionResult:
    """Build the frozen q-independent product-day mother for S1.

    The returned ``mother`` contains all mapped product-days and an explicit
    ``s1_primary`` flag.  Callers must not silently discard failed rows before
    publishing the funnel.
    """

    if not str(selected_anchor_model_id).strip():
        raise ValueError("selected_anchor_model_id must not be empty")
    if not str(selected_candidate_id).strip():
        raise ValueError("selected_candidate_id must not be empty")
    _require(target_mapping, set(PRODUCT_KEYS), "target mapping")
    mapping = _normalise_keys(target_mapping).select(*PRODUCT_KEYS).unique()
    _validate_unique(mapping, PRODUCT_KEYS, "target mapping")
    if primary_dates is not None:
        dates = tuple(str(value) for value in primary_dates)
        if not dates or dates != tuple(sorted(set(dates))):
            raise ValueError("primary_dates must be nonempty, unique, and sorted")
        mapping = mapping.filter(pl.col("Date").is_in(dates))
        absent = sorted(set(dates) - set(mapping["Date"].unique().to_list()))
        if absent:
            raise ValueError(f"target mapping lacks primary dates: {absent}")
    if mapping.is_empty():
        raise ValueError("target mapping is empty after date selection")
    mapping = mapping.with_columns(pl.lit(True).alias("mapping_support"))

    _require(
        anchor_support,
        {
            *PRODUCT_KEYS,
            "anchor_model_id",
            "selected_anchor_support",
            "selected_anchor_native_seconds",
            "analysis_eligible_seconds",
            "selected_anchor_coverage_rate",
        },
        "anchor support",
    )
    anchors = _normalise_keys(anchor_support).with_columns(
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("selected_anchor_support").fill_null(False).cast(pl.Boolean),
    )
    anchors = anchors.filter(
        pl.col("anchor_model_id") == selected_anchor_model_id
    ).select(
        *PRODUCT_KEYS,
        "anchor_model_id",
        "selected_anchor_support",
        "selected_anchor_native_seconds",
        "analysis_eligible_seconds",
        "selected_anchor_coverage_rate",
    )
    _validate_unique(anchors, PRODUCT_KEYS, "selected anchor support")

    boundary_support = _boundary_support(
        selected_boundaries,
        anchor_model_id=selected_anchor_model_id,
        candidate_id=selected_candidate_id,
        boundary_column=boundary_column,
    )
    liquidity_support, liquidity_audit = _q_independent_liquidity(
        liquidity,
        route=route,
    )

    mother = (
        mapping.join(anchors, on=list(PRODUCT_KEYS), how="left", validate="1:1")
        .join(
            boundary_support,
            on=list(PRODUCT_KEYS),
            how="left",
            validate="1:1",
        )
        .join(
            liquidity_support,
            on=list(PRODUCT_KEYS),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("anchor_model_id")
            .fill_null(selected_anchor_model_id)
            .alias("anchor_model_id"),
            pl.col("candidate_id")
            .fill_null(selected_candidate_id)
            .alias("candidate_id"),
            pl.col("selected_anchor_support")
            .fill_null(False)
            .alias("selected_anchor_support"),
            pl.col("selected_all_q_support")
            .fill_null(False)
            .alias("selected_all_q_support"),
            pl.col("long_history_gate").fill_null(False),
            pl.col("recent_history_gate").fill_null(False),
            pl.col("hard_data_gate").fill_null(False),
        )
        .with_columns(
            (pl.col("long_history_gate") & pl.col("recent_history_gate")).alias(
                "q_independent_liquidity_history"
            ),
            pl.col("hard_data_gate").alias("q_independent_hard_data_gate"),
        )
        .with_columns(
            (
                pl.col("q_independent_liquidity_history")
                & pl.col("q_independent_hard_data_gate")
            ).alias("q_independent_liquidity_gate")
        )
        .with_columns(
            (
                pl.col("mapping_support")
                & pl.col("selected_anchor_support")
                & pl.col("selected_all_q_support")
                & pl.col("q_independent_liquidity_gate")
            ).alias("s1_primary"),
            pl.lit(route).alias("route"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
        .sort(list(PRODUCT_KEYS))
    )
    forbidden = sorted(FORBIDDEN_OUTPUT_FIELDS & set(mother.columns))
    if forbidden:
        raise AssertionError(f"mother cohort leaked forbidden fields: {forbidden}")
    funnel = _cohort_funnel(mother)
    return CohortSelectionResult(
        mother=mother,
        funnel=funnel,
        liquidity_invariance_audit=liquidity_audit,
    )


def _boundary_support(
    frame: pl.DataFrame,
    *,
    anchor_model_id: str,
    candidate_id: str,
    boundary_column: str,
) -> pl.DataFrame:
    required = {
        *PRODUCT_KEYS,
        "anchor_model_id",
        "candidate_id",
        "tod_bucket",
        "boundary_quantile",
        "side",
        "source_asof_date",
        boundary_column,
    }
    _require(frame, required, "selected boundaries")
    panel = (
        _normalise_keys(frame)
        .with_columns(
            pl.col("anchor_model_id").cast(pl.String),
            pl.col("candidate_id").cast(pl.String),
            pl.col("tod_bucket").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("side").cast(pl.String),
            pl.col("source_asof_date").cast(pl.String),
            pl.col(boundary_column).cast(pl.Float64).alias("_boundary_bp"),
        )
        .filter(
            (pl.col("anchor_model_id") == anchor_model_id)
            & (pl.col("candidate_id") == candidate_id)
        )
    )
    if panel.is_empty():
        raise ValueError("selected boundaries contain no requested candidate")
    if (
        "contains_target_day_outcome" in panel.columns
        and panel["contains_target_day_outcome"].fill_null(True).any()
    ):
        raise ValueError("selected boundaries contain target-day outcomes")
    if panel["source_asof_date"].is_null().any():
        raise ValueError("selected boundary source date must not be null")
    if panel.filter(pl.col("source_asof_date") >= pl.col("Date")).height:
        raise ValueError("selected boundary source must be strictly before Date")
    full_keys = [
        *PRODUCT_KEYS,
        "anchor_model_id",
        "candidate_id",
        "tod_bucket",
        "boundary_quantile",
        "side",
    ]
    _validate_unique(panel, full_keys, "selected boundaries")
    effective = (
        pl.col("effective_supported").fill_null(False)
        if "effective_supported" in panel.columns
        else pl.col("native_supported").fill_null(False)
        if "native_supported" in panel.columns
        else pl.lit(True)
    )
    panel = panel.with_columns(
        (
            effective
            & pl.col("_boundary_bp").is_not_null()
            & pl.col("_boundary_bp").is_finite()
            & (pl.col("_boundary_bp") > 0)
            & pl.col("tod_bucket").is_in(TOD_BUCKETS)
            & pl.col("boundary_quantile").is_in(PRIMARY_QUANTILES)
            & pl.col("side").is_in(SIDES)
        ).alias("_valid_effective_row")
    )
    return (
        panel.group_by(list(PRODUCT_KEYS))
        .agg(
            pl.first("candidate_id").alias("candidate_id"),
            pl.len().alias("selected_boundary_rows"),
            pl.col("tod_bucket").n_unique().alias("selected_tod_buckets"),
            pl.col("boundary_quantile").n_unique().alias("selected_boundary_quantiles"),
            pl.col("side").n_unique().alias("selected_boundary_sides"),
            pl.col("_valid_effective_row").all().alias("_all_rows_valid"),
            pl.col("source_asof_date").min().alias("boundary_source_start_date"),
            pl.col("source_asof_date").max().alias("boundary_source_asof_date"),
        )
        .with_columns(
            (
                (pl.col("selected_boundary_rows") == EXPECTED_BOUNDARY_ROWS)
                & (pl.col("selected_tod_buckets") == len(TOD_BUCKETS))
                & (pl.col("selected_boundary_quantiles") == len(PRIMARY_QUANTILES))
                & (pl.col("selected_boundary_sides") == len(SIDES))
                & pl.col("_all_rows_valid")
            ).alias("selected_all_q_support")
        )
        .drop("_all_rows_valid")
    )


def _q_independent_liquidity(
    frame: pl.DataFrame,
    *,
    route: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    gate_columns = ("long_history_gate", "recent_history_gate", "hard_data_gate")
    _require(
        frame,
        {
            *PRODUCT_KEYS,
            "route",
            "boundary_quantile",
            "source_asof_date",
            *gate_columns,
        },
        "liquidity",
    )
    panel = (
        _normalise_keys(frame)
        .with_columns(
            pl.col("route").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("source_asof_date").cast(pl.String),
            *(
                pl.col(column).fill_null(False).cast(pl.Boolean)
                for column in gate_columns
            ),
        )
        .filter(
            (pl.col("route") == route) & pl.col("boundary_quantile").is_in([50, 80])
        )
    )
    if panel.is_empty():
        raise ValueError("liquidity has no frozen route q50/q80 rows")
    _validate_unique(
        panel,
        [*PRODUCT_KEYS, "route", "boundary_quantile"],
        "liquidity q rows",
    )
    if panel["source_asof_date"].is_null().any():
        raise ValueError("liquidity source date must not be null")
    if panel.filter(pl.col("source_asof_date") >= pl.col("Date")).height:
        raise ValueError("liquidity source must be strictly before Date")
    audit = (
        panel.group_by(list(PRODUCT_KEYS))
        .agg(
            pl.len().alias("rows"),
            pl.col("boundary_quantile").n_unique().alias("quantiles"),
            pl.col("source_asof_date").n_unique().alias("source_asof_values"),
            *(
                pl.col(column).n_unique().alias(f"{column}_values")
                for column in gate_columns
            ),
        )
        .with_columns(
            (
                (pl.col("rows") == 2)
                & (pl.col("quantiles") == 2)
                & (pl.col("source_asof_values") == 1)
                & pl.all_horizontal(
                    *(pl.col(f"{column}_values") == 1 for column in gate_columns)
                )
            ).alias("q50_q80_invariant")
        )
    )
    if not audit["q50_q80_invariant"].all():
        raise ValueError("q50/q80 liquidity history or hard-data gates differ")
    support = panel.group_by(list(PRODUCT_KEYS)).agg(
        pl.first("source_asof_date").alias("liquidity_source_asof_date"),
        *(pl.first(column).alias(column) for column in gate_columns),
    )
    return support, audit.sort(list(PRODUCT_KEYS))


def _cohort_funnel(mother: pl.DataFrame) -> pl.DataFrame:
    stages = (
        ("mapping_support", pl.col("mapping_support")),
        (
            "selected_anchor_support",
            pl.col("mapping_support") & pl.col("selected_anchor_support"),
        ),
        (
            "selected_effective_all_q_support",
            pl.col("mapping_support")
            & pl.col("selected_anchor_support")
            & pl.col("selected_all_q_support"),
        ),
        ("q_independent_liquidity_gate", pl.col("s1_primary")),
    )
    rows: list[dict[str, object]] = []
    initial = mother.height
    for order, (stage, expression) in enumerate(stages, start=1):
        selected = mother.filter(expression)
        rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "product_days": selected.height,
                "dates": selected["Date"].n_unique(),
                "products": selected["ValueCode"].n_unique(),
                "share_of_mapping": selected.height / initial,
                "contains_target_day_outcome": False,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


__all__ = [
    "FORBIDDEN_OUTPUT_FIELDS",
    "SPOT_BID_ROUTE",
    "CohortSelectionResult",
    "build_anchor_product_day_support",
    "build_s1_mother_cohort",
]
