"""Causal orchestration for the frozen S0.5 boundary candidate panel.

The raw indexed engine intentionally stops at Q0/Q1/Q2/Q5/Q6.  This module
materialises the effective Q6 -> Q1 fallback, fits the sequential Q3/Q4 level
scales, and optionally fits the development-selected ``__tod10`` variants.
Every calibration window is expressed in positions of the supplied frozen
market-session calendar.  Target-day outcomes are used only after predictions
for that target have been finalised.

The public ``predictions`` output contains only complete, positive, effective
q50/q80/q95 x positive/negative cells.  It can therefore be passed directly to
``score_boundary_predictions`` and the runner's frozen common-support ranker.
Unsupported raw rows remain available in ``raw_predictions`` and aggregate
native/effective coverage remains available in ``support_audit``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

from .foundation_boundary_adaptation import (
    MULTIPLIER_SCHEMA,
    PREDICTION_KEYS,
    SCORE_SCHEMA,
    SIDES,
    TOD_BUCKET_IDS,
    apply_boundary_multipliers,
    fit_side_specific_multipliers,
    fit_tod_bucket_multipliers,
    score_boundary_predictions,
)
from .foundation_boundary_batch import (
    RAW_BATCH_CANDIDATES,
    BoundaryBatchEngine,
)
from .foundation_boundary_selection import (
    MINIMUM_COMPLETED_PER_SIDE,
    PRIMARY_QUANTILES,
)

Q1_ID: Final = "Q1_trail60_date_equal"
Q3_ID: Final = "Q3_shape60_level5"
Q4_ID: Final = "Q4_shape60_level10"
Q6_ID: Final = "Q6_prev2_expiry_dte5"
TOD_SUFFIX: Final = "__tod10"

ORCHESTRATION_MULTIPLIER_SCHEMA: Final = pl.Schema(
    {
        **dict(MULTIPLIER_SCHEMA.items()),
        "scale_native_supported": pl.Boolean,
        "output_candidate_id": pl.String,
        "fallback_reason": pl.String,
    }
)

CELL_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
)
ROW_MATCH_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "tod_bucket",
    "boundary_quantile",
    "side",
)
SIDE_CELL_KEYS: Final = (*CELL_KEYS, "side")


@dataclass(frozen=True)
class BoundaryOrchestrationConfig:
    """Frozen adaptation support rules, overridable only for focused tests."""

    q3_level_sessions: int = 5
    q3_minimum_level_dates: int = 4
    q3_minimum_observable_cells_per_side: int = 500
    q4_level_sessions: int = 10
    q4_minimum_level_dates: int = 8
    q4_minimum_observable_cells_per_side: int = 1_000
    tod_level_sessions: int = 10
    tod_minimum_level_dates: int = 8
    tod_minimum_observable_cells_per_bucket_side: int = 100

    def __post_init__(self) -> None:
        values = (
            self.q3_level_sessions,
            self.q3_minimum_level_dates,
            self.q3_minimum_observable_cells_per_side,
            self.q4_level_sessions,
            self.q4_minimum_level_dates,
            self.q4_minimum_observable_cells_per_side,
            self.tod_level_sessions,
            self.tod_minimum_level_dates,
            self.tod_minimum_observable_cells_per_bucket_side,
        )
        if any(value <= 0 for value in values):
            raise ValueError("boundary orchestration minima must be positive")
        if self.q3_minimum_level_dates > self.q3_level_sessions:
            raise ValueError("Q3 minimum dates exceed its level window")
        if self.q4_minimum_level_dates > self.q4_level_sessions:
            raise ValueError("Q4 minimum dates exceed its level window")
        if self.tod_level_sessions != 10:
            raise ValueError("TOD adaptation is frozen to exactly ten sessions")
        if self.tod_minimum_level_dates > self.tod_level_sessions:
            raise ValueError("TOD minimum dates exceed its level window")


@dataclass(frozen=True)
class BoundaryCandidatePanel:
    """Artifacts emitted by one anchor's deterministic boundary build."""

    raw_predictions: pl.DataFrame
    predictions: pl.DataFrame
    calibration: pl.DataFrame
    multipliers: pl.DataFrame
    support_audit: pl.DataFrame
    build_dates: tuple[str, ...]
    target_dates: tuple[str, ...]
    anchor_model_id: str

    @property
    def effective_predictions(self) -> pl.DataFrame:
        """Explicit alias for the score/rank-ready ``predictions`` panel."""

        return self.predictions


@dataclass(frozen=True)
class _AdaptationSpec:
    output_candidate_id: str
    output_candidate_kind: str
    source_candidate_id: str
    sessions: int
    minimum_dates: int
    minimum_cells: int
    scope: str


def _empty_score() -> pl.DataFrame:
    return pl.DataFrame(schema=SCORE_SCHEMA)


def _empty_multipliers() -> pl.DataFrame:
    return pl.DataFrame(schema=ORCHESTRATION_MULTIPLIER_SCHEMA)


def _normalise_requested_dates(
    target_dates: Sequence[str],
    sessions: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    calendar = tuple(str(value) for value in sessions)
    if not calendar or calendar != tuple(sorted(calendar)):
        raise ValueError("sessions must be a nonempty sorted frozen calendar")
    if len(calendar) != len(set(calendar)):
        raise ValueError("sessions must not contain duplicates")
    requested = tuple(str(value) for value in target_dates)
    if (
        not requested
        or requested != tuple(sorted(requested))
        or len(requested) != len(set(requested))
    ):
        raise ValueError("target_dates must be nonempty, unique, and sorted")
    unknown = sorted(set(requested) - set(calendar))
    if unknown:
        raise ValueError(f"target_dates are outside the calendar: {unknown}")
    return calendar, requested


def _mapping_dates(target_mapping: pl.DataFrame) -> set[str]:
    if "Date" not in target_mapping.columns:
        raise ValueError("target mapping missing Date")
    return set(target_mapping["Date"].cast(pl.String).to_list())


def _validated_tod_sources(values: Sequence[str]) -> tuple[str, ...]:
    sources = tuple(str(value) for value in values)
    if len(sources) != len(set(sources)):
        raise ValueError("TOD source candidates must not contain duplicates")
    unknown = sorted(
        set(sources) - {Q1_ID, "Q2_trail20_date_equal", Q3_ID, Q4_ID, Q6_ID}
    )
    if unknown:
        raise ValueError(f"unsupported TOD source candidates: {unknown}")
    if len(sources) > 2:
        raise ValueError("TOD variants are restricted to the frozen top two sources")
    return sources


def _dependency_dates(
    calendar: tuple[str, ...],
    requested: tuple[str, ...],
    available_mapping_dates: set[str],
    *,
    warmup_sessions: int,
) -> tuple[str, ...]:
    index = {value: position for position, value in enumerate(calendar)}
    needed: set[str] = set()
    for target in requested:
        position = index[target]
        start = max(0, position - warmup_sessions)
        needed.update(calendar[start : position + 1])
    result = tuple(
        value
        for value in calendar
        if value in needed and value in available_mapping_dates
    )
    missing_targets = sorted(set(requested) - set(result))
    if missing_targets:
        raise ValueError(f"target mapping lacks requested dates: {missing_targets}")
    return result


def _episodes_for_dates(
    episodes: pl.DataFrame,
    dates: Sequence[str],
    anchor_model_id: str,
) -> pl.DataFrame:
    if "Date" not in episodes.columns or "anchor_model_id" not in episodes.columns:
        raise ValueError("episodes require Date and anchor_model_id")
    return episodes.filter(
        pl.col("Date").cast(pl.String).is_in(dates)
        & (pl.col("anchor_model_id").cast(pl.String) == anchor_model_id)
    )


def _side_support(frame: pl.DataFrame) -> pl.DataFrame:
    """Require a complete monotone q triplet without coupling both sides."""

    return (
        frame.group_by(list(SIDE_CELL_KEYS))
        .agg(
            pl.len().alias("__rows"),
            pl.col("boundary_quantile").n_unique().alias("__quantiles"),
            pl.col("native_supported").fill_null(False).all().alias("__all_native"),
            (
                pl.col("boundary_distance_bp").is_not_null()
                & pl.col("boundary_distance_bp").is_finite()
                & (pl.col("boundary_distance_bp") > 0)
            )
            .all()
            .alias("__all_positive"),
        )
        .with_columns(
            (
                (pl.col("__rows") == len(PRIMARY_QUANTILES))
                & (pl.col("__quantiles") == len(PRIMARY_QUANTILES))
                & pl.col("__all_native")
                & pl.col("__all_positive")
            ).alias("__side_native")
        )
    )


def _materialise_nonfallback(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    support = _side_support(frame).select(*SIDE_CELL_KEYS, "__side_native")
    return (
        frame.join(
            support,
            on=list(SIDE_CELL_KEYS),
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("native_supported").fill_null(False).alias("raw_native_supported"),
            pl.col("boundary_distance_bp").alias("raw_boundary_distance_bp"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("boundary_distance_bp"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("native_boundary_distance_bp"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("source_asof_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_source_asof_date"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("history_start_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_history_start_date"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("history_end_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_history_end_date"),
            pl.col("__side_native").fill_null(False).alias("native_supported"),
            pl.col("__side_native").fill_null(False).alias("effective_supported"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("candidate_id"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("effective_candidate_id"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("source_asof_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("effective_source_asof_date"),
            pl.lit(False).alias("fallback_used"),
        )
        .drop("__side_native")
    )


def _materialise_base_effective(raw_day: pl.DataFrame) -> pl.DataFrame:
    """Apply whole-cell support and the sole raw-family fallback, Q6 -> Q1."""

    non_q6 = _materialise_nonfallback(raw_day.filter(pl.col("candidate_id") != Q6_ID))
    q6_raw = raw_day.filter(pl.col("candidate_id") == Q6_ID)
    if q6_raw.is_empty():
        return non_q6.sort(list(PREDICTION_KEYS))
    q6_support = _side_support(q6_raw).select(*SIDE_CELL_KEYS, "__side_native")
    q6 = (
        q6_raw.join(
            q6_support,
            on=list(SIDE_CELL_KEYS),
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("native_supported").fill_null(False).alias("raw_native_supported"),
            pl.col("boundary_distance_bp").alias("raw_boundary_distance_bp"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("boundary_distance_bp"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("native_boundary_distance_bp"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("source_asof_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_source_asof_date"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("history_start_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_history_start_date"),
            pl.when(pl.col("__side_native").fill_null(False))
            .then(pl.col("history_end_date"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("native_history_end_date"),
            pl.col("__side_native").fill_null(False).alias("native_supported"),
        )
        .drop("__side_native")
    )
    q1 = non_q6.filter(pl.col("candidate_id") == Q1_ID)
    fallback_columns = (
        "boundary_distance_bp",
        "source_asof_date",
        "history_start_date",
        "history_end_date",
        "effective_supported",
        "effective_candidate_id",
        "effective_source_asof_date",
    )
    fallback = q1.select(
        *ROW_MATCH_KEYS,
        *(pl.col(name).alias(f"__q1_{name}") for name in fallback_columns),
    )
    joined = q6.join(
        fallback,
        on=list(ROW_MATCH_KEYS),
        how="left",
        validate="1:1",
    )
    q6_native = pl.col("native_supported")
    q1_effective = pl.col("__q1_effective_supported").fill_null(False)
    use_q1 = ~q6_native & q1_effective
    effective = q6_native | use_q1
    existing_reason = pl.col("fallback_reason").fill_null("native_Q6_unsupported")
    q6 = joined.with_columns(
        pl.when(q6_native)
        .then(pl.col("boundary_distance_bp"))
        .when(use_q1)
        .then(pl.col("__q1_boundary_distance_bp"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("boundary_distance_bp"),
        pl.when(q6_native)
        .then(pl.col("source_asof_date"))
        .when(use_q1)
        .then(pl.col("__q1_source_asof_date"))
        .otherwise(pl.col("source_asof_date"))
        .alias("source_asof_date"),
        pl.when(q6_native)
        .then(pl.col("history_start_date"))
        .when(use_q1)
        .then(pl.col("__q1_history_start_date"))
        .otherwise(pl.col("history_start_date"))
        .alias("history_start_date"),
        pl.when(q6_native)
        .then(pl.col("history_end_date"))
        .when(use_q1)
        .then(pl.col("__q1_history_end_date"))
        .otherwise(pl.col("history_end_date"))
        .alias("history_end_date"),
        effective.alias("effective_supported"),
        pl.when(q6_native)
        .then(pl.lit(Q6_ID))
        .when(use_q1)
        .then(pl.col("__q1_effective_candidate_id"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("effective_candidate_id"),
        pl.when(q6_native)
        .then(pl.col("source_asof_date"))
        .when(use_q1)
        .then(pl.col("__q1_effective_source_asof_date"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("effective_source_asof_date"),
        use_q1.alias("fallback_used"),
        pl.when(q6_native)
        .then(pl.lit(None, dtype=pl.String))
        .when(use_q1)
        .then(pl.concat_str([pl.lit("Q6_to_Q1:"), existing_reason]))
        .otherwise(pl.concat_str([pl.lit("Q6_and_Q1_unavailable:"), existing_reason]))
        .alias("fallback_reason"),
    ).drop([name for name in joined.columns if name.startswith("__q1_")])
    return pl.concat([non_q6, q6], how="diagonal_relaxed").sort(list(PREDICTION_KEYS))


def _effective_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    valid = frame.filter(
        pl.col("effective_supported").fill_null(False)
        & pl.col("boundary_distance_bp").is_not_null()
        & pl.col("boundary_distance_bp").is_finite()
        & (pl.col("boundary_distance_bp") > 0)
        & pl.col("source_asof_date").is_not_null()
    )
    complete = (
        valid.group_by(list(CELL_KEYS))
        .agg(
            pl.len().alias("__rows"),
            pl.col("boundary_quantile").n_unique().alias("__quantiles"),
            pl.col("side").n_unique().alias("__sides"),
        )
        .filter(
            (pl.col("__rows") == len(PRIMARY_QUANTILES) * len(SIDES))
            & (pl.col("__quantiles") == len(PRIMARY_QUANTILES))
            & (pl.col("__sides") == len(SIDES))
        )
    )
    return valid.join(
        complete.select(*CELL_KEYS),
        on=list(CELL_KEYS),
        how="semi",
    ).sort(list(PREDICTION_KEYS))


def _adaptation_input(source: pl.DataFrame) -> pl.DataFrame:
    """Flatten an adapted source while retaining its immediate lineage."""

    expressions: list[pl.Expr] = [
        pl.col("candidate_id").alias("upstream_candidate_id"),
        pl.col("native_supported").alias("upstream_native_supported"),
        pl.col("effective_candidate_id").alias("upstream_effective_candidate_id"),
        pl.col("effective_source_asof_date").alias(
            "upstream_effective_source_asof_date"
        ),
        pl.col("fallback_used").fill_null(False).alias("upstream_fallback_used"),
        pl.col("fallback_reason").alias("upstream_fallback_reason"),
    ]
    if "base_candidate_id" in source.columns:
        expressions.append(
            pl.col("base_candidate_id").alias("upstream_base_candidate_id")
        )
    if "adaptation_source_asof_date" in source.columns:
        expressions.append(
            pl.col("adaptation_source_asof_date").alias(
                "upstream_adaptation_source_asof_date"
            )
        )
    if "scale_native_supported" in source.columns:
        expressions.append(
            pl.col("scale_native_supported").alias("upstream_scale_native_supported")
        )
    for name in (
        "native_boundary_distance_bp",
        "native_source_asof_date",
        "native_history_start_date",
        "native_history_end_date",
    ):
        if name in source.columns:
            expressions.append(pl.col(name).alias(f"upstream_{name}"))
    result = source.with_columns(*expressions)
    conflicts = [
        name
        for name in result.columns
        if name.startswith(("base_", "adaptation_", "scale_native_supported"))
        or name
        in {
            "fallback_multiplier_used",
            "output_candidate_id",
            "source_candidate_id",
        }
    ]
    return result.drop(conflicts)


def _fallback_adaptation(
    source: pl.DataFrame,
    spec: _AdaptationSpec,
    reason: str,
) -> pl.DataFrame:
    base = _adaptation_input(source)
    return base.with_columns(
        pl.lit(spec.output_candidate_id).alias("candidate_id"),
        pl.lit(spec.output_candidate_kind).alias("candidate_kind"),
        pl.lit(False).alias("native_supported"),
        pl.lit(None, dtype=pl.Float64).alias("native_boundary_distance_bp"),
        pl.lit(None, dtype=pl.String).alias("native_source_asof_date"),
        pl.lit(None, dtype=pl.String).alias("native_history_start_date"),
        pl.lit(None, dtype=pl.String).alias("native_history_end_date"),
        pl.lit(True).alias("effective_supported"),
        pl.col("upstream_effective_candidate_id").alias("effective_candidate_id"),
        pl.col("upstream_effective_source_asof_date").alias(
            "effective_source_asof_date"
        ),
        pl.lit(True).alias("fallback_used"),
        pl.lit(spec.source_candidate_id).alias("fallback_candidate_id"),
        pl.lit(False).alias("scale_native_supported"),
        pl.lit(True).alias("fallback_multiplier_used"),
        pl.lit(1.0).alias("adaptation_multiplier"),
        pl.lit(spec.scope).alias("adaptation_scope"),
        pl.lit(reason).alias("fallback_reason"),
        pl.lit(spec.source_candidate_id).alias("source_candidate_id"),
        pl.lit(False).alias("contains_target_day_outcome"),
    ).sort(list(PREDICTION_KEYS))


def _expected_fit_error(error: ValueError) -> bool:
    text = str(error)
    return (
        "no observable non-left-censored episodes" in text
        or "no observable score rows" in text
        or "incomplete observable q cells" in text
    )


def _build_adapted_candidate(
    *,
    target_date: str,
    source: pl.DataFrame,
    predictions_by_date: Mapping[str, pl.DataFrame],
    episodes: pl.DataFrame,
    calendar: tuple[str, ...],
    spec: _AdaptationSpec,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if source.is_empty():
        return source, _empty_multipliers()
    target_index = calendar.index(target_date)
    if target_index < spec.sessions:
        return (
            _fallback_adaptation(source, spec, "insufficient_prior_sessions"),
            _empty_multipliers(),
        )
    window = calendar[target_index - spec.sessions : target_index]
    history_parts: list[pl.DataFrame] = []
    for date_text in window:
        day = predictions_by_date.get(date_text)
        if day is None:
            return (
                _fallback_adaptation(source, spec, "missing_prior_prediction_date"),
                _empty_multipliers(),
            )
        part = day.filter(pl.col("candidate_id") == spec.source_candidate_id)
        if part.is_empty():
            return (
                _fallback_adaptation(source, spec, "missing_prior_source_support"),
                _empty_multipliers(),
            )
        history_parts.append(part)
    history = pl.concat(history_parts, how="diagonal_relaxed")
    if set(history["Date"].unique().to_list()) != set(window):
        return (
            _fallback_adaptation(source, spec, "incomplete_prior_source_window"),
            _empty_multipliers(),
        )
    window_episodes = _episodes_for_dates(
        episodes,
        window,
        str(source["anchor_model_id"][0]),
    )
    try:
        if spec.scope == "global_side":
            multipliers = fit_side_specific_multipliers(
                history,
                window_episodes,
                target_date=target_date,
                calibration_dates=window,
            )
        elif spec.scope == "tod_side":
            multipliers = fit_tod_bucket_multipliers(
                history,
                window_episodes,
                target_date=target_date,
                calibration_dates=window,
            )
        else:
            raise ValueError(f"unknown adaptation scope: {spec.scope}")
    except ValueError as error:
        if not _expected_fit_error(error):
            raise
        return (
            _fallback_adaptation(source, spec, "unobservable_calibration_window"),
            _empty_multipliers(),
        )

    scale_native = (
        (pl.col("effective_calibration_dates") >= spec.minimum_dates)
        # Registry training_unit includes q, so this is the q-row count rather
        # than observable_episode_cells (which deliberately deduplicates q).
        & (pl.col("observable_score_rows") >= spec.minimum_cells)
    )
    multipliers = multipliers.with_columns(
        scale_native.alias("scale_native_supported"),
        pl.lit(spec.output_candidate_id).alias("output_candidate_id"),
        pl.when(scale_native)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit("insufficient_scale_support"))
        .alias("fallback_reason"),
    )
    applied = apply_boundary_multipliers(
        _adaptation_input(source),
        multipliers.select(MULTIPLIER_SCHEMA.names()),
        output_candidate_id=spec.output_candidate_id,
        output_candidate_kind=spec.output_candidate_kind,
    )
    join_keys = ["Date", "anchor_model_id", "side"]
    if spec.scope == "tod_side":
        join_keys.append("tod_bucket")
    support = multipliers.select(*join_keys, "scale_native_supported")
    joined = applied.join(support, on=join_keys, how="left", validate="m:1")
    native_scale = pl.col("scale_native_supported").fill_null(False) & pl.col(
        "upstream_native_supported"
    ).fill_null(False)
    base_history_start = (
        pl.col("base_history_start_date")
        if "base_history_start_date" in joined.columns
        else pl.lit(None, dtype=pl.String)
    )
    base_history_end = (
        pl.col("base_history_end_date")
        if "base_history_end_date" in joined.columns
        else pl.col("base_source_asof_date")
    )
    result = joined.with_columns(
        pl.when(native_scale)
        .then(pl.col("boundary_distance_bp"))
        .otherwise(pl.col("base_boundary_distance_bp"))
        .alias("boundary_distance_bp"),
        pl.when(native_scale)
        .then(pl.col("source_asof_date"))
        .otherwise(pl.col("base_source_asof_date"))
        .alias("source_asof_date"),
        pl.when(native_scale)
        .then(pl.col("history_start_date"))
        .otherwise(base_history_start)
        .alias("history_start_date"),
        pl.when(native_scale)
        .then(pl.col("history_end_date"))
        .otherwise(base_history_end)
        .alias("history_end_date"),
        native_scale.alias("native_supported"),
        pl.when(native_scale)
        .then(pl.col("boundary_distance_bp"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("native_boundary_distance_bp"),
        pl.when(native_scale)
        .then(pl.col("source_asof_date"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("native_source_asof_date"),
        pl.when(native_scale)
        .then(pl.col("history_start_date"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("native_history_start_date"),
        pl.when(native_scale)
        .then(pl.col("history_end_date"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("native_history_end_date"),
        pl.lit(True).alias("effective_supported"),
        pl.when(native_scale)
        .then(pl.lit(spec.output_candidate_id))
        .otherwise(pl.col("upstream_effective_candidate_id"))
        .alias("effective_candidate_id"),
        pl.when(native_scale)
        .then(pl.col("source_asof_date"))
        .otherwise(pl.col("upstream_effective_source_asof_date"))
        .alias("effective_source_asof_date"),
        (~native_scale | pl.col("upstream_fallback_used").fill_null(False)).alias(
            "fallback_used"
        ),
        pl.lit(spec.source_candidate_id).alias("fallback_candidate_id"),
        (~native_scale).alias("fallback_multiplier_used"),
        pl.when(native_scale)
        .then(pl.col("adaptation_multiplier"))
        .otherwise(pl.lit(1.0))
        .alias("adaptation_multiplier"),
        pl.when(native_scale)
        .then(pl.col("upstream_fallback_reason"))
        .otherwise(pl.lit("insufficient_scale_or_source_support"))
        .alias("fallback_reason"),
        pl.lit(spec.source_candidate_id).alias("source_candidate_id"),
        pl.lit(False).alias("contains_target_day_outcome"),
    )
    return result.sort(list(PREDICTION_KEYS)), multipliers


def _mapped_support_denominator(
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str],
    anchor_model_id: str,
) -> pl.DataFrame:
    required = {"Date", "ValueCode", "QuoteCode"}
    missing = sorted(required - set(target_mapping.columns))
    if missing:
        raise ValueError(f"target mapping missing support columns: {missing}")
    mapping = target_mapping.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    ).filter(pl.col("Date").is_in(target_dates))
    duplicate = (
        mapping.group_by(["Date", "ValueCode", "QuoteCode"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("target mapping has duplicate support denominator rows")
    return (
        mapping.group_by("Date")
        .agg((pl.len() * len(TOD_BUCKET_IDS)).alias("mapped_product_day_tod_units"))
        .with_columns(pl.lit(anchor_model_id).alias("anchor_model_id"))
    )


def _support_audit_from_denominator(
    predictions: pl.DataFrame,
    denominator: pl.DataFrame,
    *,
    candidate_ids: Sequence[str],
) -> pl.DataFrame:
    candidates = tuple(dict.fromkeys(str(value) for value in candidate_ids))
    if not candidates:
        raise ValueError("support audit candidate_ids must not be empty")
    grid = denominator.join(pl.DataFrame({"candidate_id": candidates}), how="cross")
    if predictions.is_empty():
        observed = grid.head(0)
    else:
        cell = (
            predictions.group_by(list(CELL_KEYS))
            .agg(
                pl.len().alias("rows"),
                pl.col("boundary_quantile").n_unique().alias("quantiles"),
                pl.col("side").n_unique().alias("sides"),
                pl.col("native_supported").fill_null(False).all().alias("all_native"),
                pl.col("effective_supported")
                .fill_null(False)
                .all()
                .alias("all_effective"),
                pl.col("fallback_used").fill_null(False).any().alias("any_fallback"),
            )
            .with_columns(
                (
                    (pl.col("rows") == len(PRIMARY_QUANTILES) * len(SIDES))
                    & (pl.col("quantiles") == len(PRIMARY_QUANTILES))
                    & (pl.col("sides") == len(SIDES))
                ).alias("complete_unit")
            )
        )
        observed = cell.group_by(["Date", "anchor_model_id", "candidate_id"]).agg(
            (pl.col("complete_unit") & pl.col("all_native"))
            .sum()
            .alias("native_all_q_units"),
            (pl.col("complete_unit") & pl.col("all_effective"))
            .sum()
            .alias("effective_all_q_units"),
            (pl.col("complete_unit") & pl.col("all_effective") & pl.col("any_fallback"))
            .sum()
            .alias("fallback_effective_units"),
            pl.len().alias("product_day_tod_units_with_rows"),
        )
    result = (
        grid.join(
            observed,
            on=["Date", "anchor_model_id", "candidate_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("native_all_q_units").fill_null(0).cast(pl.Int64),
            pl.col("effective_all_q_units").fill_null(0).cast(pl.Int64),
            pl.col("fallback_effective_units").fill_null(0).cast(pl.Int64),
            pl.col("product_day_tod_units_with_rows").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (
                pl.col("native_all_q_units") / pl.col("mapped_product_day_tod_units")
            ).alias("native_all_q_coverage"),
            (
                pl.col("effective_all_q_units") / pl.col("mapped_product_day_tod_units")
            ).alias("effective_all_q_coverage"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
    )
    invalid = result.filter(
        (pl.col("native_all_q_units") > pl.col("effective_all_q_units"))
        | (pl.col("effective_all_q_units") > pl.col("mapped_product_day_tod_units"))
    )
    if invalid.height:
        raise ValueError("support audit unit counts violate native/effective bounds")
    return result.sort(["Date", "anchor_model_id", "candidate_id"])


def _support_audit(
    predictions: pl.DataFrame,
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str],
    anchor_model_id: str,
    candidate_ids: Sequence[str],
) -> pl.DataFrame:
    denominator = _mapped_support_denominator(
        target_mapping,
        target_dates=target_dates,
        anchor_model_id=anchor_model_id,
    )
    return _support_audit_from_denominator(
        predictions,
        denominator,
        candidate_ids=candidate_ids,
    )


def build_boundary_candidate_panel(
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str],
    anchor_model_id: str | None = None,
    tod_source_candidates: Sequence[str] = (),
    minimum_completed_per_side: Mapping[int, int] = MINIMUM_COMPLETED_PER_SIDE,
    config: BoundaryOrchestrationConfig | None = None,
) -> BoundaryCandidatePanel:
    """Build one anchor's raw/effective/calibration boundary artifact panel.

    ``tod_source_candidates`` must be the already-frozen top-two development
    sources.  The orchestrator never re-ranks or peeks at target outcomes to
    choose them.
    """

    resolved_config = config or BoundaryOrchestrationConfig()
    calendar, requested = _normalise_requested_dates(target_dates, sessions)
    tod_sources = _validated_tod_sources(tod_source_candidates)

    engine = BoundaryBatchEngine(episodes, calendar)
    resolved_anchor = engine.anchor_model_id
    if anchor_model_id is not None and str(anchor_model_id) != resolved_anchor:
        raise ValueError(
            "anchor_model_id does not match the canonical episode facts: "
            f"{anchor_model_id!r} != {resolved_anchor!r}"
        )
    warmup = max(
        resolved_config.q3_level_sessions,
        resolved_config.q4_level_sessions,
    )
    if tod_sources:
        warmup += resolved_config.tod_level_sessions
    build_dates = _dependency_dates(
        calendar,
        requested,
        _mapping_dates(target_mapping),
        warmup_sessions=warmup,
    )
    raw = engine.predict(
        target_mapping,
        target_dates=build_dates,
        candidate_ids=RAW_BATCH_CANDIDATES,
        quantiles=PRIMARY_QUANTILES,
        minimum_completed_per_side=minimum_completed_per_side,
    )

    raw_by_date = {
        str(raw_date): frame
        for (raw_date,), frame in raw.partition_by(
            "Date", as_dict=True, maintain_order=True
        ).items()
    }
    predictions_by_date: dict[str, pl.DataFrame] = {}
    multiplier_parts: list[pl.DataFrame] = []
    all_prediction_parts: list[pl.DataFrame] = []
    all_audit_parts: list[pl.DataFrame] = []
    for target_date in build_dates:
        base = _materialise_base_effective(raw_by_date[target_date])
        day_audit_parts = [base]
        day_effective = _effective_predictions(base)
        for spec in (
            _AdaptationSpec(
                Q3_ID,
                "trail60_product_shape_global_level_scale",
                Q1_ID,
                resolved_config.q3_level_sessions,
                resolved_config.q3_minimum_level_dates,
                resolved_config.q3_minimum_observable_cells_per_side,
                "global_side",
            ),
            _AdaptationSpec(
                Q4_ID,
                "trail60_product_shape_global_level_scale",
                Q1_ID,
                resolved_config.q4_level_sessions,
                resolved_config.q4_minimum_level_dates,
                resolved_config.q4_minimum_observable_cells_per_side,
                "global_side",
            ),
        ):
            source = day_effective.filter(
                pl.col("candidate_id") == spec.source_candidate_id
            )
            adapted, scales = _build_adapted_candidate(
                target_date=target_date,
                source=source,
                predictions_by_date=predictions_by_date,
                episodes=episodes,
                calendar=calendar,
                spec=spec,
            )
            if not adapted.is_empty():
                day_effective = pl.concat(
                    [day_effective, adapted], how="diagonal_relaxed"
                )
                day_audit_parts.append(adapted)
            if not scales.is_empty():
                multiplier_parts.append(scales)

        for source_candidate in tod_sources:
            tod_spec = _AdaptationSpec(
                f"{source_candidate}{TOD_SUFFIX}",
                "tod10_side_level_scale",
                source_candidate,
                resolved_config.tod_level_sessions,
                resolved_config.tod_minimum_level_dates,
                resolved_config.tod_minimum_observable_cells_per_bucket_side,
                "tod_side",
            )
            source = day_effective.filter(pl.col("candidate_id") == source_candidate)
            adapted, scales = _build_adapted_candidate(
                target_date=target_date,
                source=source,
                predictions_by_date=predictions_by_date,
                episodes=episodes,
                calendar=calendar,
                spec=tod_spec,
            )
            if not adapted.is_empty():
                day_effective = pl.concat(
                    [day_effective, adapted], how="diagonal_relaxed"
                )
                day_audit_parts.append(adapted)
            if not scales.is_empty():
                multiplier_parts.append(scales)

        day_effective = _effective_predictions(day_effective)
        predictions_by_date[target_date] = day_effective
        all_prediction_parts.append(day_effective)
        all_audit_parts.append(pl.concat(day_audit_parts, how="diagonal_relaxed"))

    all_predictions = pl.concat(all_prediction_parts, how="diagonal_relaxed").sort(
        list(PREDICTION_KEYS)
    )
    predictions = all_predictions.filter(pl.col("Date").is_in(requested))
    prediction_dates = tuple(predictions["Date"].unique().sort().to_list())
    target_episodes = _episodes_for_dates(
        episodes,
        prediction_dates,
        resolved_anchor,
    )
    calibration = (
        score_boundary_predictions(predictions, target_episodes)
        if not predictions.is_empty()
        else _empty_score()
    )
    multipliers = (
        pl.concat(multiplier_parts, how="diagonal_relaxed")
        .sort(["Date", "anchor_model_id", "output_candidate_id", "side", "tod_bucket"])
        .filter(pl.col("Date").is_in(requested))
        if multiplier_parts
        else _empty_multipliers()
    )
    raw_targets = raw.filter(pl.col("Date").is_in(requested))
    audit_predictions = pl.concat(all_audit_parts, how="diagonal_relaxed").filter(
        pl.col("Date").is_in(requested)
    )
    audit_candidates = (
        *RAW_BATCH_CANDIDATES,
        Q3_ID,
        Q4_ID,
        *(f"{value}{TOD_SUFFIX}" for value in tod_sources),
    )
    return BoundaryCandidatePanel(
        raw_predictions=raw_targets,
        predictions=predictions,
        calibration=calibration,
        multipliers=multipliers,
        support_audit=_support_audit(
            audit_predictions,
            target_mapping,
            target_dates=requested,
            anchor_model_id=resolved_anchor,
            candidate_ids=audit_candidates,
        ),
        build_dates=build_dates,
        target_dates=requested,
        anchor_model_id=resolved_anchor,
    )


def add_tod_variants(
    base_panel: BoundaryCandidatePanel,
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    *,
    tod_source_candidates: Sequence[str],
    config: BoundaryOrchestrationConfig | None = None,
) -> BoundaryCandidatePanel:
    """Add frozen top-two TOD variants without rebuilding raw/Q3/Q4 facts.

    This is the canonical second pass after development ranking.  ``base_panel``
    must cover the desired target dates and contain the effective source
    candidates.  Its raw predictions, base calibration, base multipliers, and
    complete mapped-support denominator are reused verbatim.
    """

    resolved_config = config or BoundaryOrchestrationConfig()
    calendar, requested = _normalise_requested_dates(
        base_panel.target_dates,
        sessions,
    )
    sources = _validated_tod_sources(tod_source_candidates)
    if not sources:
        raise ValueError("add_tod_variants requires one or two frozen sources")
    output_ids = tuple(f"{value}{TOD_SUFFIX}" for value in sources)
    existing = set(base_panel.predictions["candidate_id"].unique().to_list())
    collision = sorted(set(output_ids) & existing)
    if collision:
        raise ValueError(f"base panel already contains TOD variants: {collision}")
    if base_panel.anchor_model_id == "":
        raise ValueError("base panel anchor_model_id must not be empty")
    prediction_anchors = set(
        base_panel.predictions["anchor_model_id"].unique().to_list()
    )
    if prediction_anchors and prediction_anchors != {base_panel.anchor_model_id}:
        raise ValueError("base panel predictions do not match its anchor_model_id")

    predictions_by_date = {
        str(raw_date): frame
        for (raw_date,), frame in base_panel.predictions.partition_by(
            "Date", as_dict=True, maintain_order=True
        ).items()
    }
    tod_prediction_parts: list[pl.DataFrame] = []
    tod_audit_parts: list[pl.DataFrame] = []
    tod_multiplier_parts: list[pl.DataFrame] = []
    for target_date in requested:
        base_day = predictions_by_date.get(target_date)
        if base_day is None:
            continue
        day = base_day
        for source_candidate in sources:
            spec = _AdaptationSpec(
                f"{source_candidate}{TOD_SUFFIX}",
                "tod10_side_level_scale",
                source_candidate,
                resolved_config.tod_level_sessions,
                resolved_config.tod_minimum_level_dates,
                resolved_config.tod_minimum_observable_cells_per_bucket_side,
                "tod_side",
            )
            source = base_day.filter(pl.col("candidate_id") == source_candidate)
            adapted, scales = _build_adapted_candidate(
                target_date=target_date,
                source=source,
                predictions_by_date=predictions_by_date,
                episodes=episodes,
                calendar=calendar,
                spec=spec,
            )
            if not adapted.is_empty():
                adapted = _effective_predictions(adapted)
                day = pl.concat([day, adapted], how="diagonal_relaxed")
                tod_prediction_parts.append(adapted)
                tod_audit_parts.append(adapted)
            if not scales.is_empty():
                tod_multiplier_parts.append(scales)
        predictions_by_date[target_date] = day

    tod_predictions = (
        pl.concat(tod_prediction_parts, how="diagonal_relaxed").sort(
            list(PREDICTION_KEYS)
        )
        if tod_prediction_parts
        else base_panel.predictions.head(0)
    )
    predictions = pl.concat(
        [base_panel.predictions, tod_predictions], how="diagonal_relaxed"
    ).sort(list(PREDICTION_KEYS))
    if tod_predictions.is_empty():
        tod_calibration = _empty_score()
    else:
        tod_dates = tuple(tod_predictions["Date"].unique().sort().to_list())
        tod_episodes = _episodes_for_dates(
            episodes,
            tod_dates,
            base_panel.anchor_model_id,
        )
        tod_calibration = score_boundary_predictions(
            tod_predictions,
            tod_episodes,
        )
    calibration = pl.concat(
        [base_panel.calibration, tod_calibration], how="vertical_relaxed"
    ).sort(list(PREDICTION_KEYS))
    tod_multipliers = (
        pl.concat(tod_multiplier_parts, how="diagonal_relaxed").filter(
            pl.col("Date").is_in(requested)
        )
        if tod_multiplier_parts
        else _empty_multipliers()
    )
    multipliers = pl.concat(
        [base_panel.multipliers, tod_multipliers], how="diagonal_relaxed"
    ).sort(["Date", "anchor_model_id", "output_candidate_id", "side", "tod_bucket"])

    required_audit = {
        "Date",
        "anchor_model_id",
        "mapped_product_day_tod_units",
    }
    missing_audit = sorted(required_audit - set(base_panel.support_audit.columns))
    if missing_audit:
        raise ValueError(
            f"base support audit lacks mapped denominator: {missing_audit}"
        )
    denominator = base_panel.support_audit.select(
        "Date",
        "anchor_model_id",
        "mapped_product_day_tod_units",
    ).unique()
    duplicate_denominator = (
        denominator.group_by(["Date", "anchor_model_id"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_denominator.height:
        raise ValueError("base support audit has inconsistent mapped denominators")
    tod_audit_predictions = (
        pl.concat(tod_audit_parts, how="diagonal_relaxed")
        if tod_audit_parts
        else tod_predictions
    )
    tod_support = _support_audit_from_denominator(
        tod_audit_predictions,
        denominator,
        candidate_ids=output_ids,
    )
    support_audit = pl.concat(
        [base_panel.support_audit, tod_support], how="diagonal_relaxed"
    ).sort(["Date", "anchor_model_id", "candidate_id"])
    duplicate_support = (
        support_audit.group_by(["Date", "anchor_model_id", "candidate_id"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_support.height:
        raise ValueError("incremental support audit contains duplicate candidates")
    return BoundaryCandidatePanel(
        raw_predictions=base_panel.raw_predictions,
        predictions=predictions,
        calibration=calibration,
        multipliers=multipliers,
        support_audit=support_audit,
        build_dates=base_panel.build_dates,
        target_dates=requested,
        anchor_model_id=base_panel.anchor_model_id,
    )


__all__ = [
    "ORCHESTRATION_MULTIPLIER_SCHEMA",
    "Q1_ID",
    "Q3_ID",
    "Q4_ID",
    "Q6_ID",
    "TOD_SUFFIX",
    "BoundaryCandidatePanel",
    "BoundaryOrchestrationConfig",
    "add_tod_variants",
    "build_boundary_candidate_panel",
]
