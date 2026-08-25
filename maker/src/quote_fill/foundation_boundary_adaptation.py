"""Causal boundary scoring and level/TOD multiplier adaptation.

This module is deliberately filesystem-free.  It consumes long, already
causal boundary predictions and censor-aware residual episodes.  Episode TOD
is always derived from ``start_seconds_from_open`` and stays attached to that
episode; an end time can never move an episode into another bucket.

The multiplier objective follows the frozen S0.5 registry.  For every grid
point it first averages product cells within Date and quantile, then gives
q50/q80/q95 equal weight within Date, and finally gives every effective Date
equal weight.  The first tie-break is absolute error to the midpoint of the
reach-probability censor interval.  ``abs(scale - 1)`` is second.  A final
smaller-scale tie-break exists only to make otherwise identical results
deterministic.

Only supported predictions belong here: null/non-positive boundaries or rows
marked ``native_supported=False`` are rejected instead of being silently
dropped, unless the caller has materialized explicit effective fallback
support/candidate/as-of fields.  Q6/Q7 fallback resolution and Q7 history
union/deduplication are intentionally upstream contracts, not inferred here.
Likewise, calibration callers must state the exact prior session window, and
all supplied prediction/episode dates must be inside it.  TOD fitting uses the
same q-shared 0.50--1.50 grid as global fitting; it has no implicit minimum or
fallback, so an unobservable anchor/candidate/side/TOD group fails closed.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Final

import polars as pl

PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
PREDICTION_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
    "side",
)
CELL_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
)
SIDES: Final = ("positive", "negative")
PRIMARY_QUANTILES: Final = (50, 80, 95)
NOMINAL_TAIL_PROBABILITY: Final = {50: 0.50, 80: 0.20, 95: 0.05}
TOD_BUCKETS: Final = (
    ("0905_1000", 300, 3_600),
    ("1000_1100", 3_600, 7_200),
    ("1100_1200", 7_200, 10_800),
    ("1200_1300", 10_800, 14_400),
)
TOD_BUCKET_IDS: Final = tuple(value[0] for value in TOD_BUCKETS)
DEFAULT_SCALE_GRID: Final = tuple(round(0.50 + index * 0.01, 2) for index in range(101))
TOD_CALIBRATION_SESSIONS: Final = 10
FLOAT_EPSILON: Final = 1e-12


SCORE_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "anchor_model_id": pl.String,
        "candidate_id": pl.String,
        "tod_bucket": pl.String,
        "boundary_quantile": pl.Int64,
        "side": pl.String,
        "boundary_distance_bp": pl.Float64,
        "predicted_distance_bp": pl.Float64,
        "source_asof_date": pl.String,
        "nominal_tail_probability": pl.Float64,
        "observable_started": pl.Int64,
        "completed_count": pl.Int64,
        "right_censored_count": pl.Int64,
        "confirmed_hits": pl.Int64,
        "known_nonhits": pl.Int64,
        "unknown_nonhits": pl.Int64,
        "left_censored_count": pl.Int64,
        "left_censored_confirmed_hits": pl.Int64,
        "reach_lower_bound": pl.Float64,
        "reach_upper_bound": pl.Float64,
        "reach_interval_signed_bias": pl.Float64,
        "reach_interval_distance": pl.Float64,
        "reach_midpoint_error": pl.Float64,
        "realized_amplitude_lower_bp": pl.Float64,
        "realized_amplitude_upper_bp": pl.Float64,
        "realized_amplitude_upper_unbounded": pl.Boolean,
        "realized_completed_quantile_bp": pl.Float64,
        "amplitude_interval_signed_error_bp": pl.Float64,
        "amplitude_interval_distance_bp": pl.Float64,
        "amplitude_midpoint_error_bp": pl.Float64,
        "outcome_status": pl.String,
        "episode_tod_assignment": pl.String,
        "native_supported": pl.Boolean,
        "effective_supported": pl.Boolean,
        "effective_candidate_id": pl.String,
        "prediction_contains_target_day_outcome": pl.Boolean,
        "score_contains_same_day_outcome": pl.Boolean,
    }
)


MULTIPLIER_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "anchor_model_id": pl.String,
        "base_candidate_id": pl.String,
        "side": pl.String,
        "tod_bucket": pl.String,
        "scale_scope": pl.String,
        "multiplier": pl.Float64,
        "source_asof_date": pl.String,
        "calibration_start_date": pl.String,
        "calibration_end_date": pl.String,
        "calibration_window_dates": pl.String,
        "calibration_sessions": pl.Int64,
        "effective_calibration_dates": pl.Int64,
        "observable_score_rows": pl.Int64,
        "observable_episode_cells": pl.Int64,
        "objective_reach_interval_distance": pl.Float64,
        "tie_break_reach_midpoint_error": pl.Float64,
        "mean_amplitude_interval_distance_bp": pl.Float64,
        "grid_start": pl.Float64,
        "grid_end": pl.Float64,
        "grid_step": pl.Float64,
        "grid_points": pl.Int64,
        "prediction_history_asof_date": pl.String,
        "contains_target_day_outcome": pl.Boolean,
    }
)


@dataclass(frozen=True)
class _EpisodeCell:
    completed: tuple[float, ...] = ()
    right_censored: tuple[float, ...] = ()
    left_censored: tuple[float, ...] = ()

    @property
    def observable_started(self) -> int:
        return len(self.completed) + len(self.right_censored)


@dataclass(frozen=True)
class _Metric:
    observable_started: int
    completed_count: int
    right_censored_count: int
    confirmed_hits: int
    known_nonhits: int
    unknown_nonhits: int
    left_censored_count: int
    left_censored_confirmed_hits: int
    reach_lower_bound: float | None
    reach_upper_bound: float | None
    reach_signed_bias: float | None
    reach_distance: float | None
    reach_midpoint_error: float | None
    amplitude_lower_bp: float | None
    amplitude_upper_bp: float | None
    amplitude_upper_unbounded: bool
    completed_quantile_bp: float | None
    amplitude_signed_error_bp: float | None
    amplitude_distance_bp: float | None
    amplitude_midpoint_error_bp: float | None


@dataclass(frozen=True)
class _Objective:
    interval_distance: float
    midpoint_error: float
    amplitude_interval_distance_bp: float
    effective_dates: int
    observable_score_rows: int
    observable_episode_cells: int


def score_boundary_predictions(
    predictions: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    value_column: str = "boundary_distance_bp",
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Score long prediction cells with censor-aware target-day facts.

    Non-left-censored completed episodes below the boundary are known misses.
    Non-left-censored right-censored episodes below it are unknown, while any
    observed path that reached it is a confirmed hit.  Thus the product-day
    reach interval is ``[confirmed / n, (confirmed + unknown) / n]``.

    The realized amplitude interval is the empirical q interval obtained by
    treating a right-censored amplitude as ``[observed, infinity)``.  Distance
    and midpoint lineage are reported alongside the reach calibration fields.
    """

    quantiles = _validated_quantiles(expected_quantiles)
    normal_predictions = _normalise_predictions(
        predictions,
        value_column=value_column,
        expected_quantiles=quantiles,
    )
    normal_episodes = _normalise_episodes(episodes)
    prediction_dates = set(normal_predictions["Date"].unique().to_list())
    extra_episode_dates = sorted(
        set(normal_episodes["Date"].unique().to_list()) - prediction_dates
    )
    if extra_episode_dates:
        raise ValueError(
            "episodes contain dates outside the prediction target set: "
            f"{extra_episode_dates[:5]}"
        )
    episode_cells = _episode_cells(normal_episodes)

    records: list[dict[str, object]] = []
    for prediction in normal_predictions.iter_rows(named=True):
        episode_key = _episode_key(prediction)
        facts = episode_cells.get(episode_key, _EpisodeCell())
        quantile = int(prediction["boundary_quantile"])
        boundary = float(prediction["_prediction_distance_bp"])
        metric = _score_metric(
            facts,
            boundary_bp=boundary,
            quantile=quantile,
        )
        records.append(
            {
                "Date": str(prediction["Date"]),
                "ValueCode": str(prediction["ValueCode"]),
                "QuoteCode": str(prediction["QuoteCode"]),
                "anchor_model_id": str(prediction["anchor_model_id"]),
                "candidate_id": str(prediction["candidate_id"]),
                "tod_bucket": str(prediction["tod_bucket"]),
                "boundary_quantile": quantile,
                "side": str(prediction["side"]),
                "boundary_distance_bp": boundary,
                "predicted_distance_bp": boundary,
                "source_asof_date": str(prediction["source_asof_date"]),
                "nominal_tail_probability": NOMINAL_TAIL_PROBABILITY[quantile],
                "observable_started": metric.observable_started,
                "completed_count": metric.completed_count,
                "right_censored_count": metric.right_censored_count,
                "confirmed_hits": metric.confirmed_hits,
                "known_nonhits": metric.known_nonhits,
                "unknown_nonhits": metric.unknown_nonhits,
                "left_censored_count": metric.left_censored_count,
                "left_censored_confirmed_hits": (metric.left_censored_confirmed_hits),
                "reach_lower_bound": metric.reach_lower_bound,
                "reach_upper_bound": metric.reach_upper_bound,
                "reach_interval_signed_bias": metric.reach_signed_bias,
                "reach_interval_distance": metric.reach_distance,
                "reach_midpoint_error": metric.reach_midpoint_error,
                "realized_amplitude_lower_bp": metric.amplitude_lower_bp,
                "realized_amplitude_upper_bp": metric.amplitude_upper_bp,
                "realized_amplitude_upper_unbounded": (
                    metric.amplitude_upper_unbounded
                ),
                "realized_completed_quantile_bp": (metric.completed_quantile_bp),
                "amplitude_interval_signed_error_bp": (
                    metric.amplitude_signed_error_bp
                ),
                "amplitude_interval_distance_bp": (metric.amplitude_distance_bp),
                "amplitude_midpoint_error_bp": (metric.amplitude_midpoint_error_bp),
                "outcome_status": (
                    "observable"
                    if metric.observable_started > 0
                    else "no_observable_episode"
                ),
                "episode_tod_assignment": "episode_start_half_open_frozen",
                "native_supported": bool(prediction["_native_supported"]),
                "effective_supported": bool(prediction["_effective_supported"]),
                "effective_candidate_id": str(prediction["_effective_candidate_id"]),
                "prediction_contains_target_day_outcome": False,
                "score_contains_same_day_outcome": True,
            }
        )

    result = pl.from_dicts(
        records,
        schema=SCORE_SCHEMA,
        infer_schema_length=None,
    )
    return result.sort(PREDICTION_KEYS)


def fit_side_specific_multipliers(
    predictions: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    target_date: str | date,
    calibration_dates: Sequence[str | date],
    value_column: str = "boundary_distance_bp",
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Fit one causal global multiplier per base candidate and side.

    ``calibration_dates`` is the exact ordered market-session window.  The
    function rejects target/future rows and also rejects supplied prediction or
    episode dates outside that window instead of filtering them.
    """

    return _fit_multipliers(
        predictions,
        episodes,
        target_date=target_date,
        calibration_dates=calibration_dates,
        value_column=value_column,
        expected_quantiles=expected_quantiles,
        scope="global_side",
    )


def fit_tod_bucket_multipliers(
    predictions: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    target_date: str | date,
    calibration_dates: Sequence[str | date],
    value_column: str = "boundary_distance_bp",
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Fit one side-specific multiplier per frozen TOD bucket.

    The registry fixes this adaptation window at the prior ten supplied market
    sessions.  Exactly ten unique, increasing dates are therefore required.
    """

    if len(calibration_dates) != TOD_CALIBRATION_SESSIONS:
        raise ValueError("TOD calibration requires exactly ten prior market sessions")
    return _fit_multipliers(
        predictions,
        episodes,
        target_date=target_date,
        calibration_dates=calibration_dates,
        value_column=value_column,
        expected_quantiles=expected_quantiles,
        scope="tod_side",
    )


def apply_boundary_multipliers(
    predictions: pl.DataFrame,
    multipliers: pl.DataFrame,
    *,
    output_candidate_id: str,
    output_candidate_kind: str | None = None,
    value_column: str = "boundary_distance_bp",
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Apply a complete fitted scale table to one target candidate.

    Final ``source_asof_date`` and ``history_end_date`` are the later of the
    base prediction lineage and multiplier lineage.  Base lineage is retained
    in explicit ``base_*`` columns.
    """

    quantiles = _validated_quantiles(expected_quantiles)
    normal_predictions = _normalise_predictions(
        predictions,
        value_column=value_column,
        expected_quantiles=quantiles,
    )
    candidates = normal_predictions["candidate_id"].unique().to_list()
    if len(candidates) != 1:
        raise ValueError("apply_boundary_multipliers requires one base candidate")
    anchors = normal_predictions["anchor_model_id"].unique().to_list()
    if len(anchors) != 1:
        raise ValueError("apply_boundary_multipliers requires one anchor model")
    target_dates = normal_predictions["Date"].unique().to_list()
    if len(target_dates) != 1:
        raise ValueError("apply_boundary_multipliers requires one target Date")
    if not str(output_candidate_id).strip():
        raise ValueError("output_candidate_id must not be empty")

    normal_multipliers = _normalise_multiplier_rows(multipliers)
    multiplier_dates = normal_multipliers["Date"].unique().to_list()
    if multiplier_dates != target_dates:
        raise ValueError("multiplier Date must exactly match prediction Date")
    base_candidate = str(candidates[0])
    multiplier_candidates = normal_multipliers["base_candidate_id"].unique().to_list()
    if multiplier_candidates != [base_candidate]:
        raise ValueError("multiplier base candidate does not match predictions")
    multiplier_anchors = normal_multipliers["anchor_model_id"].unique().to_list()
    if multiplier_anchors != anchors:
        raise ValueError("multiplier anchor model does not match predictions")

    scope_values = normal_multipliers["scale_scope"].unique().to_list()
    if len(scope_values) != 1:
        raise ValueError("multiplier table must contain exactly one scale scope")
    scope = str(scope_values[0])
    join_keys = ["Date", "anchor_model_id", "base_candidate_id", "side"]
    expected_multiplier_cells = len(SIDES)
    if scope == "global_side":
        if normal_multipliers["tod_bucket"].is_not_null().any():
            raise ValueError("global side multipliers must have null TOD buckets")
    elif scope == "tod_side":
        join_keys.append("tod_bucket")
        expected_multiplier_cells *= len(TOD_BUCKET_IDS)
        actual = set(normal_multipliers["tod_bucket"].drop_nulls().to_list())
        if actual != set(TOD_BUCKET_IDS):
            raise ValueError("TOD multiplier table is missing frozen buckets")
    else:
        raise ValueError(f"unknown multiplier scale_scope: {scope}")
    if normal_multipliers.height != expected_multiplier_cells:
        raise ValueError("multiplier table has incomplete side/TOD coverage")

    added_columns = {
        "base_candidate_id",
        "base_boundary_distance_bp",
        "base_source_asof_date",
        "adaptation_scope",
        "adaptation_multiplier",
        "adaptation_source_asof_date",
        "adaptation_calibration_start_date",
        "adaptation_calibration_end_date",
        "adaptation_calibration_sessions",
    }
    collisions = sorted(added_columns & set(predictions.columns))
    if collisions:
        raise ValueError(
            f"predictions already contain adaptation columns: {collisions}"
        )

    drop_temporary = normal_predictions.drop(
        "_prediction_distance_bp",
        "_native_supported",
        "_effective_supported",
        "_effective_candidate_id",
    )
    rename_map = {
        "candidate_id": "base_candidate_id",
        value_column: "base_boundary_distance_bp",
        "source_asof_date": "base_source_asof_date",
    }
    if "history_start_date" in drop_temporary.columns:
        rename_map["history_start_date"] = "base_history_start_date"
    if "history_end_date" in drop_temporary.columns:
        rename_map["history_end_date"] = "base_history_end_date"
    if "candidate_kind" in drop_temporary.columns:
        rename_map["candidate_kind"] = "base_candidate_kind"
    base = drop_temporary.rename(rename_map)

    scale_columns = [
        *join_keys,
        pl.col("multiplier").alias("adaptation_multiplier"),
        pl.col("scale_scope").alias("adaptation_scope"),
        pl.col("source_asof_date").alias("adaptation_source_asof_date"),
        pl.col("calibration_start_date").alias("adaptation_calibration_start_date"),
        pl.col("calibration_end_date").alias("adaptation_calibration_end_date"),
        pl.col("calibration_sessions").alias("adaptation_calibration_sessions"),
    ]
    scale_table = normal_multipliers.select(*scale_columns)
    joined = base.join(
        scale_table,
        on=join_keys,
        how="left",
        validate="m:1",
    )
    if joined.height != base.height or joined["adaptation_multiplier"].null_count():
        raise ValueError("multiplier table does not cover every prediction row")

    final_source = (
        pl.when(
            pl.col("base_source_asof_date") >= pl.col("adaptation_source_asof_date")
        )
        .then(pl.col("base_source_asof_date"))
        .otherwise(pl.col("adaptation_source_asof_date"))
    )
    expressions: list[pl.Expr] = [
        pl.lit(str(output_candidate_id)).alias("candidate_id"),
        (pl.col("base_boundary_distance_bp") * pl.col("adaptation_multiplier")).alias(
            value_column
        ),
        final_source.alias("source_asof_date"),
    ]
    if "contains_target_day_outcome" in joined.columns:
        expressions.append(pl.lit(False).alias("contains_target_day_outcome"))
    if output_candidate_kind is not None:
        if not str(output_candidate_kind).strip():
            raise ValueError("output_candidate_kind must not be empty")
        expressions.append(pl.lit(str(output_candidate_kind)).alias("candidate_kind"))
    elif "base_candidate_kind" in joined.columns:
        expressions.append(pl.col("base_candidate_kind").alias("candidate_kind"))
    if "base_history_start_date" in joined.columns:
        history_start = (
            pl.when(
                pl.col("base_history_start_date").is_null()
                | (
                    pl.col("adaptation_calibration_start_date")
                    < pl.col("base_history_start_date")
                )
            )
            .then(pl.col("adaptation_calibration_start_date"))
            .otherwise(pl.col("base_history_start_date"))
        )
        expressions.append(history_start.alias("history_start_date"))
    if "base_history_end_date" in joined.columns:
        expressions.append(final_source.alias("history_end_date"))

    result = joined.with_columns(*expressions)
    invalid_scaled = result.filter(
        pl.col(value_column).is_null()
        | ~pl.col(value_column).is_finite()
        | (pl.col(value_column) <= 0)
    )
    if invalid_scaled.height:
        raise ValueError("scaled boundary distances must be finite and positive")
    _normalise_predictions(
        result,
        value_column=value_column,
        expected_quantiles=quantiles,
    )
    return result.sort(PREDICTION_KEYS)


def _fit_multipliers(
    predictions: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    target_date: str | date,
    calibration_dates: Sequence[str | date],
    value_column: str,
    expected_quantiles: Sequence[int],
    scope: str,
) -> pl.DataFrame:
    quantiles = _validated_quantiles(expected_quantiles)
    target_text = _normalise_date_value(target_date, "target_date")
    window = _validated_calibration_dates(calibration_dates, target_text)
    normal_predictions = _normalise_predictions(
        predictions,
        value_column=value_column,
        expected_quantiles=quantiles,
    )
    normal_episodes = _normalise_episodes(episodes)
    prediction_dates = set(normal_predictions["Date"].unique().to_list())
    if prediction_dates != set(window):
        raise ValueError(
            "prediction dates must exactly equal the supplied calibration window"
        )
    episode_dates = set(normal_episodes["Date"].unique().to_list())
    extra_episode_dates = sorted(episode_dates - set(window))
    if extra_episode_dates:
        raise ValueError(
            "episode dates fall outside the supplied calibration window: "
            f"{extra_episode_dates[:5]}"
        )
    episode_cells = _episode_cells(normal_episodes)

    group_columns = ["anchor_model_id", "candidate_id", "side"]
    if scope == "tod_side":
        group_columns.append("tod_bucket")
    elif scope != "global_side":
        raise ValueError(f"unknown fit scope: {scope}")

    records: list[dict[str, object]] = []
    for raw_key, group in normal_predictions.group_by(
        group_columns,
        maintain_order=True,
    ):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        anchor_model_id = str(key[0])
        candidate_id = str(key[1])
        side = str(key[2])
        tod_bucket = str(key[3]) if scope == "tod_side" else None
        rows = group.iter_rows(named=True)
        prediction_rows = list(rows)
        best_scale: float | None = None
        best_objective: _Objective | None = None
        for scale in DEFAULT_SCALE_GRID:
            objective = _scale_objective(
                prediction_rows,
                episode_cells,
                scale=scale,
                expected_quantiles=quantiles,
            )
            if _objective_is_better(
                scale,
                objective,
                best_scale,
                best_objective,
            ):
                best_scale = scale
                best_objective = objective
        if best_scale is None or best_objective is None:
            raise ValueError(
                "calibration group has no observable non-left-censored episodes: "
                f"anchor={anchor_model_id}, candidate={candidate_id}, "
                f"side={side}, tod={tod_bucket}"
            )
        records.append(
            {
                "Date": target_text,
                "anchor_model_id": anchor_model_id,
                "base_candidate_id": candidate_id,
                "side": side,
                "tod_bucket": tod_bucket,
                "scale_scope": scope,
                "multiplier": best_scale,
                "source_asof_date": window[-1],
                "calibration_start_date": window[0],
                "calibration_end_date": window[-1],
                "calibration_window_dates": ",".join(window),
                "calibration_sessions": len(window),
                "effective_calibration_dates": best_objective.effective_dates,
                "observable_score_rows": best_objective.observable_score_rows,
                "observable_episode_cells": (best_objective.observable_episode_cells),
                "objective_reach_interval_distance": (best_objective.interval_distance),
                "tie_break_reach_midpoint_error": (best_objective.midpoint_error),
                "mean_amplitude_interval_distance_bp": (
                    best_objective.amplitude_interval_distance_bp
                ),
                "grid_start": DEFAULT_SCALE_GRID[0],
                "grid_end": DEFAULT_SCALE_GRID[-1],
                "grid_step": 0.01,
                "grid_points": len(DEFAULT_SCALE_GRID),
                "prediction_history_asof_date": max(
                    str(value) for value in group["source_asof_date"].unique().to_list()
                ),
                "contains_target_day_outcome": False,
            }
        )

    result = pl.from_dicts(
        records,
        schema=MULTIPLIER_SCHEMA,
        infer_schema_length=None,
    ).sort(["anchor_model_id", "base_candidate_id", "side", "tod_bucket"])
    _normalise_multiplier_rows(result)
    return result


def _scale_objective(
    predictions: Sequence[Mapping[str, object]],
    episode_cells: Mapping[tuple[str, str, str, str, str, str], _EpisodeCell],
    *,
    scale: float,
    expected_quantiles: Sequence[int],
) -> _Objective:
    distance_by_date_q: dict[tuple[str, int], list[float]] = {}
    midpoint_by_date_q: dict[tuple[str, int], list[float]] = {}
    amplitude_by_date_q: dict[tuple[str, int], list[float]] = {}
    observable_cells: set[tuple[str, str, str, str, str, str]] = set()
    observable_rows = 0
    for prediction in predictions:
        facts = episode_cells.get(_episode_key(prediction), _EpisodeCell())
        if facts.observable_started <= 0:
            continue
        quantile = int(prediction["boundary_quantile"])
        boundary = float(prediction["_prediction_distance_bp"]) * scale
        metric = _score_metric(
            facts,
            boundary_bp=boundary,
            quantile=quantile,
        )
        if (
            metric.reach_distance is None
            or metric.reach_midpoint_error is None
            or metric.amplitude_distance_bp is None
        ):
            raise AssertionError("observable metric unexpectedly lacks a loss")
        date_q = (str(prediction["Date"]), quantile)
        distance_by_date_q.setdefault(date_q, []).append(metric.reach_distance)
        midpoint_by_date_q.setdefault(date_q, []).append(metric.reach_midpoint_error)
        amplitude_by_date_q.setdefault(date_q, []).append(metric.amplitude_distance_bp)
        observable_cells.add(_episode_key(prediction))
        observable_rows += 1

    if not distance_by_date_q:
        raise ValueError("calibration group has no observable score rows")
    dates = sorted({key[0] for key in distance_by_date_q})
    date_distance: list[float] = []
    date_midpoint: list[float] = []
    date_amplitude: list[float] = []
    expected = set(expected_quantiles)
    for date_text in dates:
        available = {
            quantile
            for date_value, quantile in distance_by_date_q
            if date_value == date_text
        }
        if available != expected:
            raise ValueError(
                "an effective calibration Date has incomplete observable q cells"
            )
        q_distance = [
            math.fsum(distance_by_date_q[(date_text, quantile)])
            / len(distance_by_date_q[(date_text, quantile)])
            for quantile in expected_quantiles
        ]
        q_midpoint = [
            math.fsum(midpoint_by_date_q[(date_text, quantile)])
            / len(midpoint_by_date_q[(date_text, quantile)])
            for quantile in expected_quantiles
        ]
        q_amplitude = [
            math.fsum(amplitude_by_date_q[(date_text, quantile)])
            / len(amplitude_by_date_q[(date_text, quantile)])
            for quantile in expected_quantiles
        ]
        date_distance.append(math.fsum(q_distance) / len(q_distance))
        date_midpoint.append(math.fsum(q_midpoint) / len(q_midpoint))
        date_amplitude.append(math.fsum(q_amplitude) / len(q_amplitude))
    return _Objective(
        interval_distance=math.fsum(date_distance) / len(date_distance),
        midpoint_error=math.fsum(date_midpoint) / len(date_midpoint),
        amplitude_interval_distance_bp=(
            math.fsum(date_amplitude) / len(date_amplitude)
        ),
        effective_dates=len(dates),
        observable_score_rows=observable_rows,
        observable_episode_cells=len(observable_cells),
    )


def _objective_is_better(
    scale: float,
    objective: _Objective,
    best_scale: float | None,
    best: _Objective | None,
) -> bool:
    if best_scale is None or best is None:
        return True
    if objective.interval_distance < best.interval_distance - FLOAT_EPSILON:
        return True
    if abs(objective.interval_distance - best.interval_distance) > FLOAT_EPSILON:
        return False
    if objective.midpoint_error < best.midpoint_error - FLOAT_EPSILON:
        return True
    if abs(objective.midpoint_error - best.midpoint_error) > FLOAT_EPSILON:
        return False
    distance_to_one = abs(scale - 1.0)
    best_distance_to_one = abs(best_scale - 1.0)
    if distance_to_one < best_distance_to_one - FLOAT_EPSILON:
        return True
    if abs(distance_to_one - best_distance_to_one) > FLOAT_EPSILON:
        return False
    return scale < best_scale - FLOAT_EPSILON


def _score_metric(
    facts: _EpisodeCell,
    *,
    boundary_bp: float,
    quantile: int,
) -> _Metric:
    completed_below = bisect.bisect_left(facts.completed, boundary_bp)
    right_below = bisect.bisect_left(facts.right_censored, boundary_bp)
    left_below = bisect.bisect_left(facts.left_censored, boundary_bp)
    completed_hits = len(facts.completed) - completed_below
    right_hits = len(facts.right_censored) - right_below
    confirmed_hits = completed_hits + right_hits
    observable = facts.observable_started
    reach_lower: float | None = None
    reach_upper: float | None = None
    reach_signed: float | None = None
    reach_distance: float | None = None
    reach_midpoint: float | None = None
    nominal = NOMINAL_TAIL_PROBABILITY[quantile]
    if observable > 0:
        reach_lower = confirmed_hits / observable
        reach_upper = (confirmed_hits + right_below) / observable
        reach_signed = _interval_signed_error(
            nominal,
            reach_lower,
            reach_upper,
        )
        reach_distance = abs(reach_signed)
        reach_midpoint = abs((reach_lower + reach_upper) / 2.0 - nominal)

    probability = quantile / 100.0
    primary_observed = tuple(sorted((*facts.completed, *facts.right_censored)))
    amplitude_lower = _inverse_empirical_quantile(primary_observed, probability)
    upper_endpoints = tuple(
        sorted((*facts.completed, *(math.inf for _ in facts.right_censored)))
    )
    raw_upper = _inverse_empirical_quantile(upper_endpoints, probability)
    upper_unbounded = raw_upper is not None and math.isinf(raw_upper)
    amplitude_upper = None if upper_unbounded else raw_upper
    completed_quantile = _inverse_empirical_quantile(
        facts.completed,
        probability,
    )
    amplitude_signed: float | None = None
    amplitude_distance: float | None = None
    amplitude_midpoint: float | None = None
    if amplitude_lower is not None:
        amplitude_signed = _amplitude_interval_signed_error(
            boundary_bp,
            amplitude_lower,
            amplitude_upper,
        )
        amplitude_distance = abs(amplitude_signed)
        if amplitude_upper is not None:
            amplitude_midpoint = abs(
                boundary_bp - (amplitude_lower + amplitude_upper) / 2.0
            )

    return _Metric(
        observable_started=observable,
        completed_count=len(facts.completed),
        right_censored_count=len(facts.right_censored),
        confirmed_hits=confirmed_hits,
        known_nonhits=completed_below,
        unknown_nonhits=right_below,
        left_censored_count=len(facts.left_censored),
        left_censored_confirmed_hits=len(facts.left_censored) - left_below,
        reach_lower_bound=reach_lower,
        reach_upper_bound=reach_upper,
        reach_signed_bias=reach_signed,
        reach_distance=reach_distance,
        reach_midpoint_error=reach_midpoint,
        amplitude_lower_bp=amplitude_lower,
        amplitude_upper_bp=amplitude_upper,
        amplitude_upper_unbounded=upper_unbounded,
        completed_quantile_bp=completed_quantile,
        amplitude_signed_error_bp=amplitude_signed,
        amplitude_distance_bp=amplitude_distance,
        amplitude_midpoint_error_bp=amplitude_midpoint,
    )


def _interval_signed_error(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return upper - value
    return 0.0


def _amplitude_interval_signed_error(
    prediction: float,
    lower: float,
    upper: float | None,
) -> float:
    if prediction < lower:
        return prediction - lower
    if upper is not None and prediction > upper:
        return prediction - upper
    return 0.0


def _inverse_empirical_quantile(
    ordered_values: Sequence[float],
    probability: float,
) -> float | None:
    if not ordered_values:
        return None
    index = max(0, math.ceil(probability * len(ordered_values)) - 1)
    return float(ordered_values[index])


def _episode_cells(
    episodes: pl.DataFrame,
) -> dict[tuple[str, str, str, str, str, str], _EpisodeCell]:
    cells: dict[tuple[str, str, str, str, str, str], _EpisodeCell] = {}
    eligible = episodes.filter(pl.col("tod_bucket").is_not_null())
    group_columns = [*PRODUCT_KEYS, "anchor_model_id", "tod_bucket", "side"]
    for raw_key, group in eligible.group_by(group_columns, maintain_order=True):
        key = tuple(str(value) for value in raw_key)
        completed = tuple(
            sorted(
                float(value)
                for value in group.filter(
                    ~pl.col("left_censored") & pl.col("completed_center_return")
                )["observed_amplitude_bp"].to_list()
            )
        )
        right = tuple(
            sorted(
                float(value)
                for value in group.filter(
                    ~pl.col("left_censored") & pl.col("right_censored")
                )["observed_amplitude_bp"].to_list()
            )
        )
        left = tuple(
            sorted(
                float(value)
                for value in group.filter(pl.col("left_censored"))[
                    "observed_amplitude_bp"
                ].to_list()
            )
        )
        cells[key] = _EpisodeCell(completed, right, left)
    return cells


def _episode_key(
    row: Mapping[str, object],
) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["Date"]),
        str(row["ValueCode"]),
        str(row["QuoteCode"]),
        str(row["anchor_model_id"]),
        str(row["tod_bucket"]),
        str(row["side"]),
    )


def _normalise_predictions(
    predictions: pl.DataFrame,
    *,
    value_column: str,
    expected_quantiles: Sequence[int],
) -> pl.DataFrame:
    required = {*PREDICTION_KEYS, "source_asof_date", value_column}
    _require_columns(predictions, required, "boundary predictions")
    if predictions.is_empty():
        raise ValueError("boundary predictions must not be empty")
    result = _normalise_date_column(predictions, "Date", "boundary predictions")
    result = _normalise_date_column(
        result,
        "source_asof_date",
        "boundary prediction source_asof_date",
    ).with_columns(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("side").cast(pl.String),
        pl.col(value_column).cast(pl.Float64).alias("_prediction_distance_bp"),
    )
    invalid = result.filter(
        pl.any_horizontal(
            pl.col("ValueCode").is_null(),
            pl.col("QuoteCode").is_null(),
            pl.col("anchor_model_id").is_null(),
            pl.col("anchor_model_id").str.len_chars() == 0,
            pl.col("candidate_id").is_null(),
            pl.col("candidate_id").str.len_chars() == 0,
            ~pl.col("side").is_in(SIDES),
            ~pl.col("tod_bucket").is_in(TOD_BUCKET_IDS),
            ~pl.col("boundary_quantile").is_in(expected_quantiles),
            pl.col("_prediction_distance_bp").is_null(),
            ~pl.col("_prediction_distance_bp").is_finite(),
            pl.col("_prediction_distance_bp") <= 0,
        )
    )
    if invalid.height:
        raise ValueError("boundary predictions contain invalid keys or distances")
    if "native_supported" in result.columns:
        result = result.with_columns(
            pl.col("native_supported")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("_native_supported")
        )
    else:
        result = result.with_columns(pl.lit(True).alias("_native_supported"))
    if "effective_supported" in result.columns:
        result = result.with_columns(
            pl.col("effective_supported")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("_effective_supported")
        )
    else:
        result = result.with_columns(
            pl.col("_native_supported").alias("_effective_supported")
        )
    if not result["_effective_supported"].all():
        raise ValueError("boundary adaptation accepts only effective-supported rows")

    fallback_used = result.filter(
        ~pl.col("_native_supported") & pl.col("_effective_supported")
    )
    if fallback_used.height:
        fallback_required = {
            "effective_candidate_id",
            "effective_source_asof_date",
        }
        _require_columns(
            result,
            fallback_required,
            "effective fallback predictions",
        )
        result = result.with_columns(
            pl.when(
                pl.col("_native_supported")
                & pl.col("effective_source_asof_date").is_null()
            )
            .then(pl.col("source_asof_date"))
            .otherwise(pl.col("effective_source_asof_date"))
            .alias("effective_source_asof_date")
        )
        result = _normalise_date_column(
            result,
            "effective_source_asof_date",
            "effective fallback predictions",
        )
        invalid_fallback = result.filter(
            (~pl.col("_native_supported"))
            & (
                pl.col("effective_candidate_id").cast(pl.String).is_null()
                | (
                    pl.col("effective_candidate_id").cast(pl.String).str.len_chars()
                    == 0
                )
                | (pl.col("effective_source_asof_date") != pl.col("source_asof_date"))
            )
        )
        if invalid_fallback.height:
            raise ValueError(
                "fallback rows need explicit effective candidate/support/asof lineage"
            )
    if "effective_candidate_id" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("effective_candidate_id").is_not_null())
            .then(pl.col("effective_candidate_id").cast(pl.String))
            .otherwise(pl.col("candidate_id"))
            .alias("_effective_candidate_id")
        )
    else:
        result = result.with_columns(
            pl.col("candidate_id").alias("_effective_candidate_id")
        )
    if (
        "contains_target_day_outcome" in result.columns
        and result["contains_target_day_outcome"].fill_null(True).any()
    ):
        raise ValueError("prediction rows contain target-day outcomes")
    unsafe = result.filter(pl.col("source_asof_date") >= pl.col("Date"))
    if unsafe.height:
        raise ValueError("prediction source_asof_date must be strictly before Date")
    duplicate = result.group_by(PREDICTION_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("boundary predictions contain duplicate long keys")

    expected_q = tuple(expected_quantiles)
    for key, cell in result.group_by(CELL_KEYS, maintain_order=True):
        if set(cell["side"].unique().to_list()) != set(SIDES):
            raise ValueError(f"prediction cell has incomplete sides: {key}")
        for side in SIDES:
            side_cell = cell.filter(pl.col("side") == side).sort("boundary_quantile")
            quantiles = tuple(side_cell["boundary_quantile"].to_list())
            if quantiles != expected_q:
                raise ValueError(f"prediction cell has incomplete q values: {key}")
            sources = side_cell["source_asof_date"].unique().to_list()
            if len(sources) != 1:
                raise ValueError("q rows within a prediction side need one lineage")
            distances = side_cell["_prediction_distance_bp"].to_list()
            if any(
                later + FLOAT_EPSILON < earlier
                for earlier, later in pairwise(distances)
            ):
                raise ValueError("boundary distances must be monotone by q")
    return result.sort(PREDICTION_KEYS)


def _normalise_episodes(episodes: pl.DataFrame) -> pl.DataFrame:
    required = {
        "episode_id",
        *PRODUCT_KEYS,
        "anchor_model_id",
        "side",
        "start_seconds_from_open",
        "observed_amplitude_bp",
        "left_censored",
        "right_censored",
        "completed_center_return",
    }
    _require_columns(episodes, required, "episode facts")
    if episodes.is_empty():
        return episodes.with_columns(
            pl.lit(None, dtype=pl.String).alias("tod_bucket")
        ).head(0)
    result = _normalise_date_column(episodes, "Date", "episode facts").with_columns(
        pl.col("episode_id").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("side").cast(pl.String),
        pl.col("start_seconds_from_open").cast(pl.Int64),
        pl.col("observed_amplitude_bp").cast(pl.Float64),
        pl.col("left_censored").cast(pl.Boolean),
        pl.col("right_censored").cast(pl.Boolean),
        pl.col("completed_center_return").cast(pl.Boolean),
    )
    invalid = result.filter(
        pl.any_horizontal(
            pl.col("episode_id").is_null(),
            pl.col("episode_id").str.len_chars() == 0,
            pl.col("ValueCode").is_null(),
            pl.col("QuoteCode").is_null(),
            pl.col("anchor_model_id").is_null(),
            pl.col("anchor_model_id").str.len_chars() == 0,
            ~pl.col("side").is_in(SIDES),
            pl.col("start_seconds_from_open").is_null(),
            pl.col("observed_amplitude_bp").is_null(),
            ~pl.col("observed_amplitude_bp").is_finite(),
            pl.col("observed_amplitude_bp") < 0,
            pl.col("left_censored").is_null(),
            pl.col("right_censored").is_null(),
            pl.col("completed_center_return").is_null(),
            pl.col("right_censored") == pl.col("completed_center_return"),
        )
    )
    if invalid.height:
        raise ValueError("episode facts violate key, amplitude, or censor invariants")
    duplicate_id = (
        result.group_by(["anchor_model_id", "episode_id"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_id.height:
        raise ValueError("episode facts contain duplicate episode_id values")
    duplicate_start = (
        result.group_by(
            [
                *PRODUCT_KEYS,
                "anchor_model_id",
                "side",
                "start_seconds_from_open",
            ]
        )
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_start.height:
        raise ValueError("episode facts contain duplicate product-side starts")

    assigned = [
        _tod_bucket_for_second(int(value))
        for value in result["start_seconds_from_open"].to_list()
    ]
    if "tod_bucket" in result.columns:
        supplied = result["tod_bucket"].cast(pl.String).to_list()
        mismatches = [
            index
            for index, (provided, derived) in enumerate(zip(supplied, assigned))
            if provided is not None and provided != derived
        ]
        if mismatches:
            raise ValueError(
                "supplied episode TOD disagrees with frozen start-time assignment"
            )
        result = result.drop("tod_bucket")
    return result.with_columns(pl.Series("tod_bucket", assigned, dtype=pl.String)).sort(
        [*PRODUCT_KEYS, "anchor_model_id", "start_seconds_from_open", "side"]
    )


def _normalise_multiplier_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    required = {
        "Date",
        "anchor_model_id",
        "base_candidate_id",
        "side",
        "tod_bucket",
        "scale_scope",
        "multiplier",
        "source_asof_date",
        "calibration_start_date",
        "calibration_end_date",
        "calibration_sessions",
        "contains_target_day_outcome",
    }
    _require_columns(multipliers, required, "multipliers")
    if multipliers.is_empty():
        raise ValueError("multiplier table must not be empty")
    result = multipliers
    for column in (
        "Date",
        "source_asof_date",
        "calibration_start_date",
        "calibration_end_date",
    ):
        result = _normalise_date_column(result, column, "multipliers")
    result = result.with_columns(
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("base_candidate_id").cast(pl.String),
        pl.col("side").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("scale_scope").cast(pl.String),
        pl.col("multiplier").cast(pl.Float64),
        pl.col("calibration_sessions").cast(pl.Int64),
        pl.col("contains_target_day_outcome").cast(pl.Boolean),
    )
    invalid = result.filter(
        pl.any_horizontal(
            pl.col("anchor_model_id").is_null(),
            pl.col("anchor_model_id").str.len_chars() == 0,
            pl.col("base_candidate_id").is_null(),
            pl.col("base_candidate_id").str.len_chars() == 0,
            ~pl.col("side").is_in(SIDES),
            ~pl.col("scale_scope").is_in(["global_side", "tod_side"]),
            pl.col("multiplier").is_null(),
            ~pl.col("multiplier").is_finite(),
            pl.col("multiplier") < DEFAULT_SCALE_GRID[0] - FLOAT_EPSILON,
            pl.col("multiplier") > DEFAULT_SCALE_GRID[-1] + FLOAT_EPSILON,
            pl.col("calibration_sessions") <= 0,
            pl.col("contains_target_day_outcome").fill_null(True),
            pl.col("calibration_start_date") > pl.col("calibration_end_date"),
            pl.col("calibration_end_date") >= pl.col("Date"),
            pl.col("source_asof_date") >= pl.col("Date"),
            pl.col("source_asof_date") != pl.col("calibration_end_date"),
        )
    )
    if invalid.height:
        raise ValueError("multiplier rows violate causal or numeric invariants")
    off_grid = [
        float(value)
        for value in result["multiplier"].to_list()
        if abs(round((float(value) - 0.50) / 0.01) * 0.01 + 0.50 - float(value))
        > FLOAT_EPSILON
    ]
    if off_grid:
        raise ValueError("multiplier values must lie on the frozen 0.01 grid")
    duplicate_keys = [
        "Date",
        "anchor_model_id",
        "base_candidate_id",
        "side",
        "tod_bucket",
    ]
    duplicate = result.group_by(duplicate_keys).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("multiplier table contains duplicate scale keys")
    return result.sort(duplicate_keys)


def _validated_quantiles(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if result != PRIMARY_QUANTILES:
        raise ValueError("S0.5 adaptation requires complete q50/q80/q95 cells")
    return result


def _validated_calibration_dates(
    values: Sequence[str | date],
    target_date: str,
) -> tuple[str, ...]:
    result = tuple(_normalise_date_value(value, "calibration date") for value in values)
    if not result:
        raise ValueError("calibration_dates must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("calibration_dates contain duplicates")
    if tuple(sorted(result)) != result:
        raise ValueError("calibration_dates must be strictly increasing sessions")
    if any(value >= target_date for value in result):
        raise ValueError("calibration dates must be strictly before target_date")
    return result


def _tod_bucket_for_second(second: int) -> str | None:
    for bucket, start, end in TOD_BUCKETS:
        if start <= second < end:
            return bucket
    return None


def _normalise_date_column(
    frame: pl.DataFrame,
    column: str,
    source: str,
) -> pl.DataFrame:
    _require_columns(frame, {column}, source)
    values = [
        _normalise_date_value(value, f"{source} {column}")
        for value in frame[column].to_list()
    ]
    return frame.with_columns(pl.Series(column, values, dtype=pl.String))


def _normalise_date_value(value: object, source: str) -> str:
    if value is None:
        raise ValueError(f"{source} must not be null")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value)
    for format_text in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_text).strftime("%Y%m%d")  # noqa: DTZ007
        except ValueError:
            continue
    raise ValueError(f"{source} must be YYYYMMDD or YYYY-MM-DD: {value!r}")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


__all__ = [
    "DEFAULT_SCALE_GRID",
    "MULTIPLIER_SCHEMA",
    "NOMINAL_TAIL_PROBABILITY",
    "PREDICTION_KEYS",
    "PRIMARY_QUANTILES",
    "SCORE_SCHEMA",
    "TOD_BUCKETS",
    "TOD_CALIBRATION_SESSIONS",
    "apply_boundary_multipliers",
    "fit_side_specific_multipliers",
    "fit_tod_bucket_multipliers",
    "score_boundary_predictions",
]
