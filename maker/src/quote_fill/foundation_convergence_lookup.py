"""D-safe conditional convergence lookups and censor-aware scoring.

The entry boundary and the post-entry convergence target are different
objects.  This module consumes the q-specific post-touch path facts produced
by :mod:`foundation_convergence_selection` and builds causal lower-distance
lookups without reading files or discovering sessions.

History is pooled across historical ``QuoteCode`` values by ``ValueCode`` so a
contract roll does not silently change the product prior.  The target keeps
its exact ``QuoteCode``.  Every fitted lookup is additionally keyed by anchor,
entry-q candidate, frozen episode-start TOD bucket, and entry quantile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final

import polars as pl

from .foundation_boundary_selection import (
    censor_identified_quantile_interval,
    clip_completed_quantile_to_interval,
    date_equal_completed_quantile,
)
from .foundation_convergence_selection import CONVERGENCE_REFERENCE_SEMANTICS

CHANGE_EPS_BP: Final = 1e-9
ENTRY_STOP_SECOND: Final = 14_400
PRIMARY_QUANTILES: Final = (50, 80, 95)

TARGET_LINEAGE_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
)
HISTORY_LINEAGE_KEYS: Final = (
    "ValueCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
)
PATH_FACT_KEYS: Final = (
    *TARGET_LINEAGE_KEYS,
    "episode_sequence",
)
THRESHOLD_KEYS: Final = (
    *TARGET_LINEAGE_KEYS,
    "convergence_candidate_id",
)


@dataclass(frozen=True)
class ConditionalLookupSpec:
    """One frozen trailing-session conditional-floor estimator."""

    lookup_id: str
    lookback_sessions: int
    minimum_completed_dates: int
    minimum_completed_paths: int


REGISTERED_LOOKUPS: Final = (
    ConditionalLookupSpec("trail20_date_equal", 20, 15, 50),
    ConditionalLookupSpec("trail60_date_equal", 60, 40, 100),
)

CONDITIONAL_CANDIDATES: Final = (
    ("C2_conditional_reach80", 0.80, 0.20),
    ("C3_conditional_reach50", 0.50, 0.50),
)


@dataclass(frozen=True)
class ConvergenceScoreResult:
    """Per-path first-hit facts and product-day censor-bound summaries."""

    path_facts: pl.DataFrame
    summary: pl.DataFrame


@dataclass(frozen=True)
class PreparedConvergenceHistory:
    """Validated post-touch facts partitioned by the frozen session calendar.

    Construction performs the expensive frontier and path validation once.
    The aligned tuple is also a D-safe date index: a target lookup can slice
    exactly its preceding market sessions without rescanning later facts.
    """

    sessions: tuple[str, ...]
    facts_by_session: tuple[pl.DataFrame, ...]
    fact_count: int


CONVERGENCE_PREDICTION_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "anchor_model_id": pl.String,
        "candidate_id": pl.String,
        "tod_bucket": pl.String,
        "boundary_quantile": pl.Int64,
        "convergence_candidate_id": pl.String,
        "convergence_reference_semantics": pl.String,
        "lookup_id": pl.String,
        "lookback_sessions": pl.Int64,
        "minimum_completed_dates": pl.Int64,
        "minimum_completed_paths": pl.Int64,
        "target_reach_probability": pl.Float64,
        "floor_quantile_probability": pl.Float64,
        "source_asof_date": pl.String,
        "history_start_date": pl.String,
        "history_end_date": pl.String,
        "history_observation_end_date": pl.String,
        "selected_sessions": pl.Int64,
        "completed_history_dates": pl.Int64,
        "observable_started": pl.Int64,
        "completed_paths": pl.Int64,
        "right_censored_paths": pl.Int64,
        "completed_point_bp": pl.Float64,
        "identified_lower_bp": pl.Float64,
        "identified_upper_bp": pl.Float64,
        "identified_upper_unbounded": pl.Boolean,
        "native_threshold_distance_bp": pl.Float64,
        "native_supported": pl.Boolean,
        "native_failure_reason": pl.String,
        "effective_threshold_distance_bp": pl.Float64,
        "effective_supported": pl.Boolean,
        "effective_lookup_id": pl.String,
        "effective_source_asof_date": pl.String,
        "fallback_lookup_id": pl.String,
        "fallback_used": pl.Boolean,
        "fallback_reason": pl.String,
        "contains_target_day_outcome": pl.Boolean,
    }
)

CONVERGENCE_PATH_SCORE_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "anchor_model_id": pl.String,
        "candidate_id": pl.String,
        "tod_bucket": pl.String,
        "boundary_quantile": pl.Int64,
        "episode_sequence": pl.Int64,
        "convergence_candidate_id": pl.String,
        "convergence_reference_semantics": pl.String,
        "effective_lookup_id": pl.String,
        "effective_source_asof_date": pl.String,
        "threshold_distance_bp": pl.Float64,
        "frozen_exit_basis_bp": pl.Float64,
        "touch_second": pl.Int64,
        "touch_anchor_basis_bp": pl.Float64,
        "frozen_center_basis_bp": pl.Float64,
        "center_hit_second": pl.Int64,
        "path_end_second": pl.Int64,
        "observed_post_touch_floor_bp": pl.Float64,
        "negative_cycle_completed": pl.Boolean,
        "right_censored": pl.Boolean,
        "right_censor_reason": pl.String,
        "first_hit_second": pl.Int64,
        "time_to_hit_from_touch_seconds": pl.Int64,
        "time_to_hit_from_center_seconds": pl.Int64,
        "confirmed_hit": pl.Boolean,
        "known_miss": pl.Boolean,
        "unknown_censored": pl.Boolean,
        "hit_status": pl.String,
        "contains_target_day_outcome": pl.Boolean,
    }
)


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_date(value: object, source: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {value!r}") from error


def _normalise_sessions(sessions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(str(value) for value in sessions)
    if not result:
        raise ValueError("frozen market-session calendar must not be empty")
    for value in result:
        _parse_date(value, "market session")
    if len(set(result)) != len(result):
        raise ValueError("frozen market-session calendar contains duplicates")
    if result != tuple(sorted(result)):
        raise ValueError("frozen market-session calendar must be strictly ordered")
    return result


def _normalise_targets(targets: pl.DataFrame) -> pl.DataFrame:
    _require(targets, set(TARGET_LINEAGE_KEYS), "convergence targets")
    result = targets.select(*TARGET_LINEAGE_KEYS).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    )
    if result.is_empty():
        raise ValueError("convergence targets must not be empty")
    target_dates = result["Date"].unique().to_list()
    if len(target_dates) != 1:
        raise ValueError("convergence lookup accepts exactly one target Date")
    _parse_date(target_dates[0], "target Date")
    invalid = result.filter(
        pl.any_horizontal(
            pl.col("ValueCode").is_null(),
            pl.col("QuoteCode").is_null(),
            pl.col("anchor_model_id").is_null(),
            pl.col("candidate_id").is_null(),
            pl.col("tod_bucket").is_null(),
            ~pl.col("boundary_quantile").is_in(PRIMARY_QUANTILES),
            pl.col("ValueCode").str.len_chars() == 0,
            pl.col("QuoteCode").str.len_chars() == 0,
            pl.col("anchor_model_id").str.len_chars() == 0,
            pl.col("candidate_id").str.len_chars() == 0,
            pl.col("tod_bucket").str.len_chars() == 0,
        )
    )
    if invalid.height:
        raise ValueError("convergence targets contain invalid lineage keys")
    duplicate = result.group_by(TARGET_LINEAGE_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("convergence targets contain duplicate lineage keys")
    return result.sort(TARGET_LINEAGE_KEYS)


def _normalise_facts(facts: pl.DataFrame, source: str) -> pl.DataFrame:
    required = {
        *PATH_FACT_KEYS,
        "source_asof_date",
        "upper_distance_bp",
        "independent_lower_distance_bp",
        "touch_second",
        "touch_residual_bp",
        "touch_basis_mid_bp",
        "touch_anchor_basis_bp",
        "frozen_center_basis_bp",
        "frozen_independent_lower_basis_bp",
        "convergence_reference_semantics",
        "center_hit",
        "center_hit_second",
        "path_end_second",
        "observed_post_touch_floor_bp",
        "floor_frontier_distance_bp",
        "floor_frontier_hit_second",
        "negative_cycle_completed",
        "right_censored",
        "right_censor_reason",
    }
    _require(facts, required, source)
    result = facts.select(*required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("episode_sequence").cast(pl.Int64),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("independent_lower_distance_bp").cast(pl.Float64),
        pl.col("touch_second").cast(pl.Int64),
        pl.col("touch_residual_bp").cast(pl.Float64),
        pl.col("touch_basis_mid_bp").cast(pl.Float64),
        pl.col("touch_anchor_basis_bp").cast(pl.Float64),
        pl.col("frozen_center_basis_bp").cast(pl.Float64),
        pl.col("frozen_independent_lower_basis_bp").cast(pl.Float64),
        pl.col("convergence_reference_semantics").cast(pl.String),
        pl.col("center_hit").fill_null(False).cast(pl.Boolean),
        pl.col("center_hit_second").cast(pl.Int64),
        pl.col("path_end_second").cast(pl.Int64),
        pl.col("observed_post_touch_floor_bp").cast(pl.Float64),
        pl.col("floor_frontier_distance_bp").cast(pl.List(pl.Float64)),
        pl.col("floor_frontier_hit_second").cast(pl.List(pl.Int64)),
        pl.col("negative_cycle_completed").fill_null(False).cast(pl.Boolean),
        pl.col("right_censored").fill_null(False).cast(pl.Boolean),
        pl.col("right_censor_reason").cast(pl.String),
    )
    duplicate = result.group_by(PATH_FACT_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError(f"{source} contains duplicate post-touch path keys")
    invalid = result.filter(
        pl.any_horizontal(
            ~pl.col("boundary_quantile").is_in(PRIMARY_QUANTILES),
            pl.col("episode_sequence") <= 0,
            pl.col("upper_distance_bp").is_null(),
            ~pl.col("upper_distance_bp").is_finite(),
            pl.col("upper_distance_bp") <= 0.0,
            pl.col("independent_lower_distance_bp").is_null(),
            ~pl.col("independent_lower_distance_bp").is_finite(),
            pl.col("independent_lower_distance_bp") <= 0.0,
            pl.col("touch_basis_mid_bp").is_null(),
            ~pl.col("touch_basis_mid_bp").is_finite(),
            pl.col("touch_residual_bp").is_null(),
            ~pl.col("touch_residual_bp").is_finite(),
            pl.col("touch_anchor_basis_bp").is_null(),
            ~pl.col("touch_anchor_basis_bp").is_finite(),
            pl.col("frozen_center_basis_bp").is_null(),
            ~pl.col("frozen_center_basis_bp").is_finite(),
            pl.col("frozen_independent_lower_basis_bp").is_null(),
            ~pl.col("frozen_independent_lower_basis_bp").is_finite(),
            pl.col("convergence_reference_semantics")
            != CONVERGENCE_REFERENCE_SEMANTICS,
            (
                pl.col("touch_anchor_basis_bp")
                - pl.col("frozen_center_basis_bp")
            ).abs()
            > CHANGE_EPS_BP,
            (
                pl.col("touch_basis_mid_bp")
                - pl.col("touch_anchor_basis_bp")
                - pl.col("touch_residual_bp")
            ).abs()
            > CHANGE_EPS_BP,
            pl.col("touch_residual_bp") + CHANGE_EPS_BP
            < pl.col("upper_distance_bp"),
            (
                pl.col("frozen_center_basis_bp")
                - pl.col("independent_lower_distance_bp")
                - pl.col("frozen_independent_lower_basis_bp")
            ).abs()
            > CHANGE_EPS_BP,
            pl.col("observed_post_touch_floor_bp").is_null(),
            ~pl.col("observed_post_touch_floor_bp").is_finite(),
            pl.col("observed_post_touch_floor_bp") < 0.0,
            pl.col("touch_second") >= ENTRY_STOP_SECOND,
            pl.col("path_end_second") < pl.col("touch_second"),
            pl.col("center_hit") != pl.col("center_hit_second").is_not_null(),
            pl.col("negative_cycle_completed") == pl.col("right_censored"),
            pl.col("negative_cycle_completed") & ~pl.col("center_hit"),
        )
    )
    if invalid.height:
        raise ValueError(f"{source} violates convergence path invariants")

    for row in result.iter_rows(named=True):
        date = str(row["Date"])
        source_asof = str(row["source_asof_date"])
        if _parse_date(source_asof, "source_asof_date") >= _parse_date(
            date, "fact Date"
        ):
            raise ValueError(f"{source} requires source_asof_date < Date")
        distances = list(row["floor_frontier_distance_bp"] or [])
        seconds = list(row["floor_frontier_hit_second"] or [])
        if len(distances) != len(seconds):
            raise ValueError(f"{source} frontier lists have unequal lengths")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in distances
        ):
            raise ValueError(f"{source} frontier distances must be finite and positive")
        if any(
            float(right) <= float(left) + CHANGE_EPS_BP
            for left, right in pairwise(distances)
        ):
            raise ValueError(f"{source} frontier distances must strictly increase")
        if any(right <= left for left, right in pairwise(seconds)):
            raise ValueError(f"{source} frontier hit seconds must strictly increase")
        center = row["center_hit_second"]
        if any(
            second < int(row["touch_second"])
            or second > int(row["path_end_second"])
            or (center is not None and second < int(center))
            for second in seconds
        ):
            raise ValueError(f"{source} frontier seconds are outside the path")
        expected_floor = float(distances[-1]) if distances else 0.0
        if (
            abs(float(row["observed_post_touch_floor_bp"]) - expected_floor)
            > CHANGE_EPS_BP
        ):
            raise ValueError(f"{source} frontier does not reproduce observed floor")
    return result.sort(PATH_FACT_KEYS)


def _adapt_floor_sample(facts: pl.DataFrame) -> pl.DataFrame:
    return facts.select(
        "Date",
        pl.col("observed_post_touch_floor_bp").alias("observed_amplitude_bp"),
        pl.lit(False).alias("left_censored"),
        "right_censored",
        pl.col("negative_cycle_completed").alias("completed_center_return"),
    )


def prepare_convergence_history(
    history_facts: pl.DataFrame,
    sessions: list[str] | tuple[str, ...],
) -> PreparedConvergenceHistory:
    """Validate all facts once and align them to the frozen session calendar.

    Unlike the scalar builder, a prepared history may contain the target Date
    and later Dates.  Prepared prediction slices only calendar positions before
    its target, so those rows are indexed for later calls but can never enter a
    target lookup.
    """

    calendar = _normalise_sessions(sessions)
    facts = _normalise_facts(history_facts, "post-touch history")
    return _prepare_normalised_convergence_history(facts, calendar)


def _prepare_normalised_convergence_history(
    facts: pl.DataFrame,
    calendar: tuple[str, ...],
) -> PreparedConvergenceHistory:
    calendar_set = set(calendar)
    unknown_history_dates = sorted(set(facts["Date"].to_list()) - calendar_set)
    if unknown_history_dates:
        raise ValueError(
            "post-touch history contains Dates absent from the frozen "
            f"market-session calendar: {unknown_history_dates[:5]}"
        )
    raw_partitions = facts.partition_by(
        "Date",
        as_dict=True,
        maintain_order=True,
    )
    by_date = {
        str(key[0] if isinstance(key, tuple) else key): partition.sort(
            [*HISTORY_LINEAGE_KEYS, "Date", "episode_sequence"]
        )
        for key, partition in raw_partitions.items()
    }
    empty = facts.head(0)
    return PreparedConvergenceHistory(
        sessions=calendar,
        facts_by_session=tuple(by_date.get(date, empty) for date in calendar),
        fact_count=facts.height,
    )


def _empty_lineage_quantiles(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    schema = {column: frame.schema[column] for column in HISTORY_LINEAGE_KEYS}
    schema.update({column: pl.Float64 for column in columns})
    return pl.DataFrame(schema=schema)


def _date_equal_lineage_quantiles(
    facts: pl.DataFrame,
    *,
    probabilities: tuple[float, ...],
    output_prefix: str,
    completed_only: bool,
    upper_endpoints: bool,
) -> pl.DataFrame:
    """Compute weighted inverse quantiles for every lineage in three scans.

    Each lineage-Date has unit mass and its paths split that mass equally.
    Sorting is global by lineage and endpoint; cumulative weights still reset
    per lineage, which is exactly the scalar date-equal estimator.
    """

    output_columns = tuple(
        f"{output_prefix}_{round(probability * 100)}" for probability in probabilities
    )
    sample = (
        facts.filter(pl.col("negative_cycle_completed")) if completed_only else facts
    )
    if sample.is_empty():
        return _empty_lineage_quantiles(facts, output_columns)
    date_keys = [*HISTORY_LINEAGE_KEYS, "Date"]
    endpoint = (
        pl.when(pl.col("right_censored"))
        .then(pl.lit(math.inf))
        .otherwise(pl.col("observed_post_touch_floor_bp"))
        if upper_endpoints
        else pl.col("observed_post_touch_floor_bp")
    )
    ordered = (
        sample.select(
            *HISTORY_LINEAGE_KEYS,
            "Date",
            endpoint.cast(pl.Float64).alias("_endpoint"),
        )
        .with_columns((1.0 / pl.len().over(date_keys)).alias("_weight"))
        .sort([*HISTORY_LINEAGE_KEYS, "_endpoint"])
        .with_columns(
            pl.col("_weight")
            .cum_sum()
            .over(HISTORY_LINEAGE_KEYS)
            .alias("_cumulative_weight"),
            pl.col("_weight").sum().over(HISTORY_LINEAGE_KEYS).alias("_total_weight"),
        )
    )
    outputs: pl.DataFrame | None = None
    for probability, output_column in zip(
        probabilities,
        output_columns,
        strict=True,
    ):
        crossing = (
            ordered.filter(
                pl.col("_cumulative_weight") + 1e-12
                >= pl.lit(probability) * pl.col("_total_weight")
            )
            .group_by(HISTORY_LINEAGE_KEYS, maintain_order=True)
            .agg(pl.col("_endpoint").first().alias(output_column))
        )
        outputs = (
            crossing
            if outputs is None
            else outputs.join(
                crossing,
                on=list(HISTORY_LINEAGE_KEYS),
                how="full",
                coalesce=True,
                validate="1:1",
            )
        )
    if outputs is None:
        return _empty_lineage_quantiles(facts, output_columns)
    return outputs


def _prepared_lookup_cells(
    selected: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    target_lineages = targets.select(*HISTORY_LINEAGE_KEYS).unique()
    sample = selected.join(
        target_lineages,
        on=list(HISTORY_LINEAGE_KEYS),
        how="semi",
    )
    if sample.is_empty():
        stats_schema = {
            column: targets.schema[column] for column in HISTORY_LINEAGE_KEYS
        }
        stats_schema.update(
            {
                "history_observation_end_date": pl.String,
                "completed_history_dates": pl.Int64,
                "observable_started": pl.Int64,
                "completed_paths": pl.Int64,
                "right_censored_paths": pl.Int64,
            }
        )
        stats = pl.DataFrame(schema=stats_schema)
    else:
        stats = sample.group_by(
            HISTORY_LINEAGE_KEYS,
            maintain_order=True,
        ).agg(
            pl.col("Date").max().alias("history_observation_end_date"),
            pl.col("Date")
            .filter(pl.col("negative_cycle_completed"))
            .n_unique()
            .cast(pl.Int64)
            .alias("completed_history_dates"),
            pl.len().cast(pl.Int64).alias("observable_started"),
            pl.col("negative_cycle_completed")
            .sum()
            .cast(pl.Int64)
            .alias("completed_paths"),
            pl.col("right_censored").sum().cast(pl.Int64).alias("right_censored_paths"),
        )
    probabilities = tuple(
        floor_quantile for _, _, floor_quantile in CONDITIONAL_CANDIDATES
    )
    completed = _date_equal_lineage_quantiles(
        sample,
        probabilities=probabilities,
        output_prefix="completed_point",
        completed_only=True,
        upper_endpoints=False,
    )
    lower = _date_equal_lineage_quantiles(
        sample,
        probabilities=probabilities,
        output_prefix="identified_lower",
        completed_only=False,
        upper_endpoints=False,
    )
    upper = _date_equal_lineage_quantiles(
        sample,
        probabilities=probabilities,
        output_prefix="identified_upper_endpoint",
        completed_only=False,
        upper_endpoints=True,
    )
    count_columns = (
        "completed_history_dates",
        "observable_started",
        "completed_paths",
        "right_censored_paths",
    )
    return (
        targets.join(
            stats,
            on=list(HISTORY_LINEAGE_KEYS),
            how="left",
            validate="m:1",
        )
        .join(
            completed,
            on=list(HISTORY_LINEAGE_KEYS),
            how="left",
            validate="m:1",
        )
        .join(
            lower,
            on=list(HISTORY_LINEAGE_KEYS),
            how="left",
            validate="m:1",
        )
        .join(
            upper,
            on=list(HISTORY_LINEAGE_KEYS),
            how="left",
            validate="m:1",
        )
        .with_columns(
            *[pl.col(column).fill_null(0).cast(pl.Int64) for column in count_columns]
        )
    )


def _registered_records_from_cells(
    cells: pl.DataFrame,
    *,
    spec: ConditionalLookupSpec,
    prior_dates: tuple[str, ...],
) -> list[dict[str, object]]:
    source_asof = prior_dates[-1] if prior_dates else None
    history_start = prior_dates[0] if prior_dates else None
    history_end = prior_dates[-1] if prior_dates else None
    records: list[dict[str, object]] = []
    for target_row in cells.iter_rows(named=True):
        for candidate_id, target_reach, floor_quantile in CONDITIONAL_CANDIDATES:
            quantile_suffix = round(floor_quantile * 100)
            completed_point = target_row[f"completed_point_{quantile_suffix}"]
            lower = target_row[f"identified_lower_{quantile_suffix}"]
            upper_endpoint = target_row[f"identified_upper_endpoint_{quantile_suffix}"]
            upper_unbounded = upper_endpoint is not None and math.isinf(
                float(upper_endpoint)
            )
            upper = None if upper_unbounded else upper_endpoint
            clipped: float | None = None
            if completed_point is not None:
                clipped = float(completed_point)
                if lower is not None:
                    clipped = max(clipped, float(lower))
                if upper is not None:
                    clipped = min(clipped, float(upper))
            completed_dates = int(target_row["completed_history_dates"])
            completed_paths = int(target_row["completed_paths"])
            reasons: list[str] = []
            if source_asof is None:
                reasons.append("no_prior_market_session")
            if completed_dates < spec.minimum_completed_dates:
                reasons.append(
                    "insufficient_completed_dates:"
                    f"{completed_dates}<{spec.minimum_completed_dates}"
                )
            if completed_paths < spec.minimum_completed_paths:
                reasons.append(
                    "insufficient_completed_paths:"
                    f"{completed_paths}<{spec.minimum_completed_paths}"
                )
            if clipped is None or not math.isfinite(clipped):
                reasons.append("unavailable_finite_point")
            elif clipped < 0.0:
                reasons.append("negative_point")
            native_supported = not reasons
            native_distance = clipped if native_supported else None
            records.append(
                {
                    **{column: target_row[column] for column in TARGET_LINEAGE_KEYS},
                    "convergence_candidate_id": candidate_id,
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "lookup_id": spec.lookup_id,
                    "lookback_sessions": spec.lookback_sessions,
                    "minimum_completed_dates": spec.minimum_completed_dates,
                    "minimum_completed_paths": spec.minimum_completed_paths,
                    "target_reach_probability": target_reach,
                    "floor_quantile_probability": floor_quantile,
                    "source_asof_date": source_asof,
                    "history_start_date": history_start,
                    "history_end_date": history_end,
                    "history_observation_end_date": target_row[
                        "history_observation_end_date"
                    ],
                    "selected_sessions": len(prior_dates),
                    "completed_history_dates": completed_dates,
                    "observable_started": int(target_row["observable_started"]),
                    "completed_paths": completed_paths,
                    "right_censored_paths": int(target_row["right_censored_paths"]),
                    "completed_point_bp": completed_point,
                    "identified_lower_bp": lower,
                    "identified_upper_bp": upper,
                    "identified_upper_unbounded": upper_unbounded,
                    "native_threshold_distance_bp": native_distance,
                    "native_supported": native_supported,
                    "native_failure_reason": (";".join(reasons) if reasons else None),
                    "effective_threshold_distance_bp": native_distance,
                    "effective_supported": native_supported,
                    "effective_lookup_id": (
                        spec.lookup_id if native_supported else None
                    ),
                    "effective_source_asof_date": (
                        source_asof if native_supported else None
                    ),
                    "fallback_lookup_id": None,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    return records


def build_registered_conditional_convergence_grid_prepared(
    prepared: PreparedConvergenceHistory,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Build trail20/trail60 C2/C3 from one reusable validated history."""

    if not isinstance(prepared, PreparedConvergenceHistory):
        raise TypeError("prepared must be a PreparedConvergenceHistory")
    target = _normalise_targets(targets)
    target_date = str(target.item(0, "Date"))
    if target_date not in prepared.sessions:
        raise ValueError("target Date is absent from frozen market-session calendar")
    target_position = prepared.sessions.index(target_date)
    maximum_lookback = max(spec.lookback_sessions for spec in REGISTERED_LOOKUPS)
    maximum_start = max(0, target_position - maximum_lookback)
    maximum_frames = prepared.facts_by_session[maximum_start:target_position]
    nonempty_frames = [frame for frame in maximum_frames if frame.height]
    selected_maximum = (
        pl.concat(nonempty_frames, how="vertical_relaxed")
        if nonempty_frames
        else prepared.facts_by_session[0].head(0)
    )
    records: list[dict[str, object]] = []
    for spec in REGISTERED_LOOKUPS:
        window_start = max(0, target_position - spec.lookback_sessions)
        prior_dates = prepared.sessions[window_start:target_position]
        selected = (
            selected_maximum.filter(pl.col("Date").is_in(prior_dates))
            if prior_dates
            else selected_maximum.head(0)
        )
        cells = _prepared_lookup_cells(selected, target)
        records.extend(
            _registered_records_from_cells(
                cells,
                spec=spec,
                prior_dates=prior_dates,
            )
        )
    result = pl.from_dicts(
        records,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort([*THRESHOLD_KEYS, "lookup_id"])
    validate_convergence_prediction_lineage(result)
    return result


def build_conditional_convergence_predictions(
    history_facts: pl.DataFrame,
    targets: pl.DataFrame,
    sessions: list[str] | tuple[str, ...],
    *,
    spec: ConditionalLookupSpec,
) -> pl.DataFrame:
    """Fit C2/C3 for one target Date from an exact frozen session window."""

    if spec.lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    if spec.minimum_completed_dates <= 0 or spec.minimum_completed_paths <= 0:
        raise ValueError("conditional lookup minima must be positive")
    calendar = _normalise_sessions(sessions)
    target = _normalise_targets(targets)
    target_date = str(target.item(0, "Date"))
    if target_date not in calendar:
        raise ValueError("target Date is absent from frozen market-session calendar")
    history = _normalise_facts(history_facts, "post-touch history")
    unknown_history_dates = sorted(set(history["Date"].to_list()) - set(calendar))
    if unknown_history_dates:
        raise ValueError(
            "post-touch history contains Dates absent from the frozen "
            f"market-session calendar: {unknown_history_dates[:5]}"
        )
    nonprior = history.filter(pl.col("Date") >= target_date)
    if nonprior.height:
        raise ValueError("post-touch history contains target-day/future outcomes")

    target_position = calendar.index(target_date)
    prior_dates = calendar[
        max(0, target_position - spec.lookback_sessions) : target_position
    ]
    selected = history.filter(pl.col("Date").is_in(prior_dates))
    history_by_lineage: dict[tuple[object, ...], pl.DataFrame] = {
        tuple(key): group
        for key, group in selected.group_by(HISTORY_LINEAGE_KEYS, maintain_order=True)
    }
    source_asof = prior_dates[-1] if prior_dates else None
    history_start = prior_dates[0] if prior_dates else None
    history_end = prior_dates[-1] if prior_dates else None

    records: list[dict[str, object]] = []
    for target_row in target.iter_rows(named=True):
        lineage = tuple(target_row[column] for column in HISTORY_LINEAGE_KEYS)
        sample = history_by_lineage.get(lineage, selected.head(0))
        completed = sample.filter(pl.col("negative_cycle_completed"))
        completed_dates = completed["Date"].n_unique() if completed.height else 0
        completed_paths = completed.height
        observation_end = str(sample["Date"].max()) if sample.height else None
        adapted = _adapt_floor_sample(sample)
        for candidate_id, target_reach, floor_quantile in CONDITIONAL_CANDIDATES:
            completed_point = date_equal_completed_quantile(adapted, floor_quantile)
            interval = censor_identified_quantile_interval(
                adapted,
                floor_quantile,
                date_equal=True,
            )
            clipped = clip_completed_quantile_to_interval(
                completed_point,
                interval,
            )
            reasons: list[str] = []
            if source_asof is None:
                reasons.append("no_prior_market_session")
            if completed_dates < spec.minimum_completed_dates:
                reasons.append(
                    "insufficient_completed_dates:"
                    f"{completed_dates}<{spec.minimum_completed_dates}"
                )
            if completed_paths < spec.minimum_completed_paths:
                reasons.append(
                    "insufficient_completed_paths:"
                    f"{completed_paths}<{spec.minimum_completed_paths}"
                )
            if clipped is None or not math.isfinite(float(clipped)):
                reasons.append("unavailable_finite_point")
            elif float(clipped) < 0.0:
                reasons.append("negative_point")
            native_supported = not reasons
            native_distance = float(clipped) if native_supported else None
            records.append(
                {
                    **{column: target_row[column] for column in TARGET_LINEAGE_KEYS},
                    "convergence_candidate_id": candidate_id,
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "lookup_id": spec.lookup_id,
                    "lookback_sessions": spec.lookback_sessions,
                    "minimum_completed_dates": spec.minimum_completed_dates,
                    "minimum_completed_paths": spec.minimum_completed_paths,
                    "target_reach_probability": target_reach,
                    "floor_quantile_probability": floor_quantile,
                    "source_asof_date": source_asof,
                    "history_start_date": history_start,
                    "history_end_date": history_end,
                    "history_observation_end_date": observation_end,
                    "selected_sessions": len(prior_dates),
                    "completed_history_dates": completed_dates,
                    "observable_started": interval.observable_started,
                    "completed_paths": interval.completed_count,
                    "right_censored_paths": interval.right_censored_count,
                    "completed_point_bp": completed_point,
                    "identified_lower_bp": interval.lower_bp,
                    "identified_upper_bp": interval.upper_bp,
                    "identified_upper_unbounded": interval.upper_unbounded,
                    "native_threshold_distance_bp": native_distance,
                    "native_supported": native_supported,
                    "native_failure_reason": ";".join(reasons) if reasons else None,
                    "effective_threshold_distance_bp": native_distance,
                    "effective_supported": native_supported,
                    "effective_lookup_id": spec.lookup_id if native_supported else None,
                    "effective_source_asof_date": (
                        source_asof if native_supported else None
                    ),
                    "fallback_lookup_id": None,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    result = pl.from_dicts(
        records,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort([*THRESHOLD_KEYS, "lookup_id"])
    validate_convergence_prediction_lineage(result)
    return result


def build_registered_conditional_convergence_grid(
    history_facts: pl.DataFrame,
    targets: pl.DataFrame,
    sessions: list[str] | tuple[str, ...],
) -> pl.DataFrame:
    """Build both registered lookups, preserving the prior-only scalar API."""

    calendar = _normalise_sessions(sessions)
    target = _normalise_targets(targets)
    target_date = str(target.item(0, "Date"))
    history = _normalise_facts(history_facts, "post-touch history")
    unknown_history_dates = sorted(set(history["Date"].to_list()) - set(calendar))
    if unknown_history_dates:
        raise ValueError(
            "post-touch history contains Dates absent from the frozen "
            f"market-session calendar: {unknown_history_dates[:5]}"
        )
    if history.filter(pl.col("Date") >= target_date).height:
        raise ValueError("post-touch history contains target-day/future outcomes")
    prepared = _prepare_normalised_convergence_history(history, calendar)
    return build_registered_conditional_convergence_grid_prepared(
        prepared,
        target,
    )


def validate_convergence_prediction_lineage(predictions: pl.DataFrame) -> None:
    """Reject target leakage and internally inconsistent effective support."""

    required = {
        *THRESHOLD_KEYS,
        "lookup_id",
        "convergence_reference_semantics",
        "source_asof_date",
        "native_threshold_distance_bp",
        "native_supported",
        "effective_threshold_distance_bp",
        "effective_supported",
        "effective_lookup_id",
        "effective_source_asof_date",
        "contains_target_day_outcome",
    }
    _require(predictions, required, "convergence predictions")
    invalid = predictions.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
        | (
            pl.col("convergence_reference_semantics").fill_null("")
            != CONVERGENCE_REFERENCE_SEMANTICS
        )
        | (
            pl.col("native_supported").fill_null(False)
            & (
                pl.col("native_threshold_distance_bp").is_null()
                | ~pl.col("native_threshold_distance_bp").is_finite()
                | (pl.col("native_threshold_distance_bp") < 0.0)
                | pl.col("source_asof_date").is_null()
            )
        )
        | (
            pl.col("effective_supported").fill_null(False)
            & (
                pl.col("effective_threshold_distance_bp").is_null()
                | ~pl.col("effective_threshold_distance_bp").is_finite()
                | (pl.col("effective_threshold_distance_bp") < 0.0)
                | pl.col("effective_lookup_id").is_null()
                | pl.col("effective_source_asof_date").is_null()
            )
        )
    )
    if invalid.height:
        raise ValueError("convergence predictions contain inconsistent support")
    for row in predictions.iter_rows(named=True):
        target = _parse_date(row["Date"], "prediction Date")
        for column in ("source_asof_date", "effective_source_asof_date"):
            value = row[column]
            if value is not None and _parse_date(value, column) >= target:
                raise ValueError(f"{column} must be strictly before Date")


def resolve_convergence_fallback(
    predictions: pl.DataFrame,
    *,
    primary_lookup_id: str,
    fallback_lookup_id: str,
) -> pl.DataFrame:
    """Resolve one explicit native-first fallback chain for C2/C3.

    The returned row retains the primary candidate's native diagnostics while
    making the source of its effective threshold explicit.  No fallback order
    is inferred by this module; callers must state it.
    """

    validate_convergence_prediction_lineage(predictions)
    primary = predictions.filter(pl.col("lookup_id") == str(primary_lookup_id))
    fallback = predictions.filter(pl.col("lookup_id") == str(fallback_lookup_id))
    for name, frame in (("primary", primary), ("fallback", fallback)):
        duplicate = frame.group_by(THRESHOLD_KEYS).len().filter(pl.col("len") != 1)
        if duplicate.height:
            raise ValueError(f"{name} convergence lookup has duplicate target keys")
    primary_keys = set(primary.select(THRESHOLD_KEYS).iter_rows())
    fallback_keys = set(fallback.select(THRESHOLD_KEYS).iter_rows())
    if not primary_keys or primary_keys != fallback_keys:
        raise ValueError("primary and fallback convergence lookup keys must match")
    fallback_rows = {
        tuple(row[column] for column in THRESHOLD_KEYS): row
        for row in fallback.iter_rows(named=True)
    }
    records: list[dict[str, object]] = []
    for row in primary.iter_rows(named=True):
        key = tuple(row[column] for column in THRESHOLD_KEYS)
        other = fallback_rows[key]
        output = dict(row)
        output["fallback_lookup_id"] = str(fallback_lookup_id)
        if bool(row["native_supported"]):
            output.update(
                {
                    "effective_threshold_distance_bp": row[
                        "native_threshold_distance_bp"
                    ],
                    "effective_supported": True,
                    "effective_lookup_id": str(primary_lookup_id),
                    "effective_source_asof_date": row["source_asof_date"],
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            )
        elif bool(other["native_supported"]):
            output.update(
                {
                    "effective_threshold_distance_bp": other[
                        "native_threshold_distance_bp"
                    ],
                    "effective_supported": True,
                    "effective_lookup_id": str(fallback_lookup_id),
                    "effective_source_asof_date": other["source_asof_date"],
                    "fallback_used": True,
                    "fallback_reason": row["native_failure_reason"],
                }
            )
        else:
            output.update(
                {
                    "effective_threshold_distance_bp": None,
                    "effective_supported": False,
                    "effective_lookup_id": None,
                    "effective_source_asof_date": None,
                    "fallback_used": False,
                    "fallback_reason": (
                        f"primary={row['native_failure_reason']};"
                        f"fallback={other['native_failure_reason']}"
                    ),
                }
            )
        records.append(output)
    result = pl.from_dicts(
        records,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort(THRESHOLD_KEYS)
    validate_convergence_prediction_lineage(result)
    return result


def _build_outcome_conditioned_control_convergence_predictions(
    facts: pl.DataFrame,
) -> pl.DataFrame:
    """Reconstruct C0/C1 from touched facts for test parity only.

    Row availability is conditioned on a target-day upper touch, so this is
    not a prediction builder and must never feed scoring or publication.  The
    boundary-driven public adapter below is the only D-safe control builder.
    """

    paths = _normalise_facts(facts, "control post-touch facts")
    records: list[dict[str, object]] = []
    for _, group in paths.group_by(TARGET_LINEAGE_KEYS, maintain_order=True):
        first = group.row(0, named=True)
        lower_min = float(group["independent_lower_distance_bp"].min())
        lower_max = float(group["independent_lower_distance_bp"].max())
        if lower_max - lower_min > CHANGE_EPS_BP:
            raise ValueError("independent lower is not frozen within target lineage")
        source_values = group["source_asof_date"].unique().to_list()
        if len(source_values) != 1:
            raise ValueError("entry boundary lineage varies within target path facts")
        source_asof = str(source_values[0])
        base = {column: first[column] for column in TARGET_LINEAGE_KEYS}
        for candidate_id, lookup_id, distance, target_reach in (
            ("C0_center", "structural_zero", 0.0, None),
            (
                "C1_independent_lower_control",
                "independent_lower_control",
                lower_min,
                1.0 - int(first["boundary_quantile"]) / 100.0,
            ),
        ):
            records.append(
                {
                    **base,
                    "convergence_candidate_id": candidate_id,
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "lookup_id": lookup_id,
                    "lookback_sessions": 0,
                    "minimum_completed_dates": 0,
                    "minimum_completed_paths": 0,
                    "target_reach_probability": target_reach,
                    "floor_quantile_probability": None,
                    "source_asof_date": source_asof,
                    "history_start_date": None,
                    "history_end_date": None,
                    "history_observation_end_date": None,
                    "selected_sessions": 0,
                    "completed_history_dates": 0,
                    # C0/C1 are adapters, not target-outcome-fitted lookups.
                    # Target path counts belong only in the later score table.
                    "observable_started": 0,
                    "completed_paths": 0,
                    "right_censored_paths": 0,
                    "completed_point_bp": None,
                    "identified_lower_bp": None,
                    "identified_upper_bp": None,
                    "identified_upper_unbounded": False,
                    "native_threshold_distance_bp": distance,
                    "native_supported": True,
                    "native_failure_reason": None,
                    "effective_threshold_distance_bp": distance,
                    "effective_supported": True,
                    "effective_lookup_id": lookup_id,
                    "effective_source_asof_date": source_asof,
                    "fallback_lookup_id": None,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "contains_target_day_outcome": True,
                }
            )
    result = pl.from_dicts(
        records,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort(THRESHOLD_KEYS)
    return result


def build_control_convergence_predictions_from_boundaries(
    boundary_predictions_day: pl.DataFrame,
    *,
    boundary_value_column: str = "boundary_distance_bp",
) -> pl.DataFrame:
    """Build C0/C1 for every D-safe boundary lineage, touched or untouched.

    Controls are prediction-time objects.  Deriving their row availability
    from post-touch facts would condition the grid on whether the upper path
    touched later that day.  This adapter instead pairs the positive target
    row with its negative-side distance using only the frozen boundary table.
    """

    required = {
        *TARGET_LINEAGE_KEYS,
        "side",
        boundary_value_column,
        "source_asof_date",
    }
    _require(
        boundary_predictions_day,
        required,
        "control boundary predictions",
    )
    if boundary_predictions_day.is_empty():
        return pl.DataFrame(schema=CONVERGENCE_PREDICTION_SCHEMA)
    if (
        "contains_target_day_outcome" in boundary_predictions_day.columns
        and boundary_predictions_day["contains_target_day_outcome"]
        .fill_null(True)
        .any()
    ):
        raise ValueError("control boundaries contain target-day outcomes")
    if (
        "effective_supported" in boundary_predictions_day.columns
        and not boundary_predictions_day["effective_supported"].fill_null(False).all()
    ):
        raise ValueError("control boundaries contain unsupported cells")
    panel = boundary_predictions_day.select(*required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("side").cast(pl.String),
        pl.col(boundary_value_column).cast(pl.Float64).alias("_distance_bp"),
        pl.col("source_asof_date").cast(pl.String),
    )
    dates = panel["Date"].unique().to_list()
    if len(dates) != 1:
        raise ValueError("control boundary predictions require exactly one Date")
    target_date = _parse_date(dates[0], "control boundary Date")
    invalid = panel.filter(
        pl.any_horizontal(
            pl.col("ValueCode").is_null(),
            pl.col("QuoteCode").is_null(),
            pl.col("anchor_model_id").is_null(),
            pl.col("candidate_id").is_null(),
            pl.col("tod_bucket").is_null(),
            ~pl.col("boundary_quantile").is_in(PRIMARY_QUANTILES),
            ~pl.col("side").is_in(("positive", "negative")),
            pl.col("_distance_bp").is_null(),
            ~pl.col("_distance_bp").is_finite(),
            pl.col("_distance_bp") <= 0.0,
            pl.col("source_asof_date").is_null(),
        )
    )
    if invalid.height:
        raise ValueError("control boundary predictions contain invalid cells")
    unsafe_sources = [
        value
        for value in panel["source_asof_date"].unique().to_list()
        if _parse_date(value, "source_asof_date") >= target_date
    ]
    if unsafe_sources:
        raise ValueError("control boundary source_asof_date must be before Date")
    duplicate = (
        panel.group_by([*TARGET_LINEAGE_KEYS, "side"]).len().filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("control boundary predictions contain duplicate cells")
    paired = panel.group_by(
        TARGET_LINEAGE_KEYS,
        maintain_order=True,
    ).agg(
        pl.len().alias("_rows"),
        pl.col("side").n_unique().alias("_sides"),
        pl.col("_distance_bp")
        .filter(pl.col("side") == "negative")
        .first()
        .alias("_negative_distance_bp"),
        pl.col("source_asof_date").n_unique().alias("_source_count"),
        pl.col("source_asof_date").first().alias("source_asof_date"),
    )
    incomplete = paired.filter(
        (pl.col("_rows") != 2)
        | (pl.col("_sides") != 2)
        | pl.col("_negative_distance_bp").is_null()
        | (pl.col("_source_count") != 1)
    )
    if incomplete.height:
        raise ValueError(
            "control boundary predictions require matched positive/negative lineage"
        )
    records: list[dict[str, object]] = []
    for row in paired.iter_rows(named=True):
        source_asof = str(row["source_asof_date"])
        lower = float(row["_negative_distance_bp"])
        base = {column: row[column] for column in TARGET_LINEAGE_KEYS}
        for candidate_id, lookup_id, distance, target_reach in (
            ("C0_center", "structural_zero", 0.0, None),
            (
                "C1_independent_lower_control",
                "independent_lower_control",
                lower,
                1.0 - int(row["boundary_quantile"]) / 100.0,
            ),
        ):
            records.append(
                {
                    **base,
                    "convergence_candidate_id": candidate_id,
                    "convergence_reference_semantics": (
                        CONVERGENCE_REFERENCE_SEMANTICS
                    ),
                    "lookup_id": lookup_id,
                    "lookback_sessions": 0,
                    "minimum_completed_dates": 0,
                    "minimum_completed_paths": 0,
                    "target_reach_probability": target_reach,
                    "floor_quantile_probability": None,
                    "source_asof_date": source_asof,
                    "history_start_date": None,
                    "history_end_date": None,
                    "history_observation_end_date": None,
                    "selected_sessions": 0,
                    "completed_history_dates": 0,
                    "observable_started": 0,
                    "completed_paths": 0,
                    "right_censored_paths": 0,
                    "completed_point_bp": None,
                    "identified_lower_bp": None,
                    "identified_upper_bp": None,
                    "identified_upper_unbounded": False,
                    "native_threshold_distance_bp": distance,
                    "native_supported": True,
                    "native_failure_reason": None,
                    "effective_threshold_distance_bp": distance,
                    "effective_supported": True,
                    "effective_lookup_id": lookup_id,
                    "effective_source_asof_date": source_asof,
                    "fallback_lookup_id": None,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "contains_target_day_outcome": False,
                }
            )
    result = pl.from_dicts(
        records,
        schema=CONVERGENCE_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort(THRESHOLD_KEYS)
    validate_convergence_prediction_lineage(result)
    return result


def first_hit_second_for_threshold(
    *,
    threshold_distance_bp: float,
    center_hit_second: int | None,
    floor_frontier_distance_bp: list[float] | tuple[float, ...],
    floor_frontier_hit_second: list[int] | tuple[int, ...],
) -> int | None:
    """Return the first observed center/lower hit for any nonnegative threshold."""

    threshold = float(threshold_distance_bp)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold_distance_bp must be finite and nonnegative")
    if threshold <= CHANGE_EPS_BP:
        return center_hit_second
    distances = list(floor_frontier_distance_bp)
    seconds = list(floor_frontier_hit_second)
    if len(distances) != len(seconds):
        raise ValueError("floor frontier lists have unequal lengths")
    return next(
        (
            int(second)
            for distance, second in zip(distances, seconds, strict=True)
            if float(distance) + CHANGE_EPS_BP >= threshold
        ),
        None,
    )


def _inverse_empirical(values: list[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(float(probability) * len(ordered)))
    return ordered[rank - 1]


def score_convergence_predictions(
    target_facts: pl.DataFrame,
    threshold_predictions: pl.DataFrame,
) -> ConvergenceScoreResult:
    """Score C0/C1/C2/C3 or arbitrary thresholds with censor bounds.

    ``threshold_predictions`` must contain one effective row per target lineage
    and threshold id.  Callers can therefore use the registered lookup rows or
    supply another D-safe nonnegative threshold under a distinct
    ``convergence_candidate_id``.  First-hit latency is measured from both the
    upper touch and the first center return.  Summary p50/p90 use the inverse
    empirical order statistic among confirmed hits.
    """

    facts = _normalise_facts(target_facts, "target post-touch facts")
    validate_convergence_prediction_lineage(threshold_predictions)
    duplicate = (
        threshold_predictions.group_by(THRESHOLD_KEYS).len().filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError(
            "threshold predictions require one effective row per target threshold"
        )
    supported = threshold_predictions.filter(pl.col("effective_supported"))
    joined = facts.join(
        supported.select(
            *THRESHOLD_KEYS,
            "effective_lookup_id",
            "effective_source_asof_date",
            "effective_threshold_distance_bp",
        ),
        on=list(TARGET_LINEAGE_KEYS),
        how="inner",
        validate="m:m",
    )
    path_records: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        threshold = float(row["effective_threshold_distance_bp"])
        hit_second = first_hit_second_for_threshold(
            threshold_distance_bp=threshold,
            center_hit_second=row["center_hit_second"],
            floor_frontier_distance_bp=row["floor_frontier_distance_bp"] or [],
            floor_frontier_hit_second=row["floor_frontier_hit_second"] or [],
        )
        confirmed = hit_second is not None
        known_miss = bool(row["negative_cycle_completed"]) and not confirmed
        unknown = bool(row["right_censored"]) and not confirmed
        if sum((confirmed, known_miss, unknown)) != 1:
            raise ValueError("target path does not have one identified hit status")
        center_second = row["center_hit_second"]
        path_records.append(
            {
                **{column: row[column] for column in PATH_FACT_KEYS},
                "convergence_candidate_id": row["convergence_candidate_id"],
                "convergence_reference_semantics": (
                    CONVERGENCE_REFERENCE_SEMANTICS
                ),
                "effective_lookup_id": row["effective_lookup_id"],
                "effective_source_asof_date": row["effective_source_asof_date"],
                "threshold_distance_bp": threshold,
                "frozen_exit_basis_bp": (
                    float(row["frozen_center_basis_bp"]) - threshold
                ),
                "touch_second": row["touch_second"],
                "touch_anchor_basis_bp": row["touch_anchor_basis_bp"],
                "frozen_center_basis_bp": row["frozen_center_basis_bp"],
                "center_hit_second": center_second,
                "path_end_second": row["path_end_second"],
                "observed_post_touch_floor_bp": row["observed_post_touch_floor_bp"],
                "negative_cycle_completed": row["negative_cycle_completed"],
                "right_censored": row["right_censored"],
                "right_censor_reason": row["right_censor_reason"],
                "first_hit_second": hit_second,
                "time_to_hit_from_touch_seconds": (
                    None
                    if hit_second is None
                    else int(hit_second) - int(row["touch_second"])
                ),
                "time_to_hit_from_center_seconds": (
                    None
                    if hit_second is None or center_second is None
                    else int(hit_second) - int(center_second)
                ),
                "confirmed_hit": confirmed,
                "known_miss": known_miss,
                "unknown_censored": unknown,
                "hit_status": (
                    "confirmed_hit"
                    if confirmed
                    else "known_miss"
                    if known_miss
                    else "unknown_censored"
                ),
                "contains_target_day_outcome": True,
            }
        )
    path_frame = (
        pl.from_dicts(
            path_records,
            schema=CONVERGENCE_PATH_SCORE_SCHEMA,
            infer_schema_length=None,
        ).sort([*THRESHOLD_KEYS, "episode_sequence"])
        if path_records
        else pl.DataFrame(schema=CONVERGENCE_PATH_SCORE_SCHEMA)
    )

    records_by_threshold: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in path_records:
        key = tuple(row[column] for column in THRESHOLD_KEYS)
        records_by_threshold.setdefault(key, []).append(row)
    summary_records: list[dict[str, object]] = []
    for prediction in threshold_predictions.iter_rows(named=True):
        key = tuple(prediction[column] for column in THRESHOLD_KEYS)
        rows = records_by_threshold.get(key, [])
        started = len(rows)
        confirmed = sum(bool(row["confirmed_hit"]) for row in rows)
        known = sum(bool(row["known_miss"]) for row in rows)
        unknown = sum(bool(row["unknown_censored"]) for row in rows)
        touch_times = [
            int(row["time_to_hit_from_touch_seconds"])
            for row in rows
            if row["time_to_hit_from_touch_seconds"] is not None
        ]
        center_times = [
            int(row["time_to_hit_from_center_seconds"])
            for row in rows
            if row["time_to_hit_from_center_seconds"] is not None
        ]
        effective = bool(prediction["effective_supported"])
        summary_records.append(
            {
                **prediction,
                "prediction_contains_target_day_outcome": prediction[
                    "contains_target_day_outcome"
                ],
                "contains_target_day_outcome": started > 0,
                "n_started": started,
                "confirmed_hits": confirmed,
                "known_misses": known,
                "unknown_censored": unknown,
                "reach_lower_bound": (
                    confirmed / started if effective and started else None
                ),
                "reach_upper_bound": (
                    (confirmed + unknown) / started if effective and started else None
                ),
                "first_hit_observations": len(touch_times),
                "time_to_hit_from_touch_p50_seconds": _inverse_empirical(
                    touch_times, 0.50
                ),
                "time_to_hit_from_touch_p90_seconds": _inverse_empirical(
                    touch_times, 0.90
                ),
                "time_to_hit_from_center_p50_seconds": _inverse_empirical(
                    center_times, 0.50
                ),
                "time_to_hit_from_center_p90_seconds": _inverse_empirical(
                    center_times, 0.90
                ),
                "time_quantile_method": "inverse_empirical",
                "outcome_status": (
                    "unsupported_threshold"
                    if not effective
                    else "observable"
                    if started
                    else "no_observable_touched_path"
                ),
            }
        )
    summary = pl.from_dicts(summary_records, infer_schema_length=None).sort(
        THRESHOLD_KEYS
    )
    return ConvergenceScoreResult(path_facts=path_frame, summary=summary)


__all__ = [
    "CONDITIONAL_CANDIDATES",
    "CONVERGENCE_PATH_SCORE_SCHEMA",
    "CONVERGENCE_PREDICTION_SCHEMA",
    "CONVERGENCE_REFERENCE_SEMANTICS",
    "HISTORY_LINEAGE_KEYS",
    "PATH_FACT_KEYS",
    "REGISTERED_LOOKUPS",
    "TARGET_LINEAGE_KEYS",
    "THRESHOLD_KEYS",
    "ConditionalLookupSpec",
    "ConvergenceScoreResult",
    "PreparedConvergenceHistory",
    "build_conditional_convergence_predictions",
    "build_control_convergence_predictions_from_boundaries",
    "build_registered_conditional_convergence_grid",
    "build_registered_conditional_convergence_grid_prepared",
    "first_hit_second_for_threshold",
    "prepare_convergence_history",
    "resolve_convergence_fallback",
    "score_convergence_predictions",
    "validate_convergence_prediction_lineage",
]
