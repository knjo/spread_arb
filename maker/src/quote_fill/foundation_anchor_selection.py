"""Pure, leakage-safe anchor selection for the rebuilt frozen S0.5 registry.

The functions in this module accept exactly one complete one-second session.
They do not read files, choose a winning model, or publish artifacts.  Model
selection remains a runner concern; this layer only materializes the frozen
candidate/label definitions and compact common-support product-day facts.

Unlike the legacy S0.5 controls, ``time_*`` candidates decay or roll over
elapsed wall-clock time.  ``count_*`` candidates deliberately remain legal-
observation-count controls so eligibility gaps have measurably different
semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import polars as pl

MAX_DEVELOPMENT_DATE: Final = "20260813"
ANALYSIS_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
SESSION_END_SECOND: Final = 15_600
EXPECTED_PRODUCT_ROWS: Final = SESSION_END_SECOND
GROUP_KEYS: Final = ("Date", "ValueCode", "QuoteCode")


@dataclass(frozen=True)
class AnchorCandidate:
    """One candidate frozen in ``foundation_selection_registry.json``."""

    model_id: str
    kind: str
    parameter: int | None
    q_eligible: bool


@dataclass(frozen=True)
class FutureMedianHorizon:
    """One inclusive future-median label window."""

    horizon_id: str
    start_seconds: int
    end_seconds: int
    minimum_coverage: float

    @property
    def window_rows(self) -> int:
        return self.end_seconds - self.start_seconds + 1

    @property
    def minimum_rows(self) -> int:
        return math.ceil(self.window_rows * self.minimum_coverage)


@dataclass(frozen=True)
class AnchorSelectionDayEvaluation:
    """Compact common-support outputs for a single market session."""

    product_day_stats: pl.DataFrame
    support_stats: pl.DataFrame


ANCHOR_CANDIDATES: Final = (
    AnchorCandidate("time_ewma_15s", "wall_clock_ewma", 15, True),
    AnchorCandidate("time_ewma_30s", "wall_clock_ewma", 30, True),
    AnchorCandidate("time_ewma_60s", "wall_clock_ewma", 60, True),
    AnchorCandidate("time_ewma_120s", "wall_clock_ewma", 120, True),
    AnchorCandidate("time_median_60s", "wall_clock_rolling_median", 60, True),
    AnchorCandidate("time_median_120s", "wall_clock_rolling_median", 120, True),
    AnchorCandidate("count_ewma_30obs", "legal_observation_ewma", 30, False),
    AnchorCandidate("count_ewma_120obs", "legal_observation_ewma", 120, True),
    AnchorCandidate("persistence", "current_legal_basis", None, False),
    AnchorCandidate("open_5m", "frozen_opening_median", 300, True),
)
ACTIONABLE_MODEL_IDS: Final = tuple(
    candidate.model_id for candidate in ANCHOR_CANDIDATES[:6]
)

FUTURE_HORIZONS: Final = (
    FutureMedianHorizon("future_median_30_300s", 30, 300, 0.90),
    FutureMedianHorizon("future_median_10_60s", 10, 60, 0.90),
    FutureMedianHorizon("future_median_300_900s", 300, 900, 0.90),
)

MODEL_COLUMNS: Final = {
    candidate.model_id: f"anchor_{candidate.model_id}_bp"
    for candidate in ANCHOR_CANDIDATES
}
LABEL_COLUMNS: Final = {
    horizon.horizon_id: f"label_{horizon.horizon_id}_bp" for horizon in FUTURE_HORIZONS
}
COVERAGE_COLUMNS: Final = {
    horizon.horizon_id: f"label_{horizon.horizon_id}_coverage"
    for horizon in FUTURE_HORIZONS
}
SAMPLE_COUNT_COLUMNS: Final = {
    horizon.horizon_id: f"label_{horizon.horizon_id}_sample_count"
    for horizon in FUTURE_HORIZONS
}

_CANDIDATE_BY_ID: Final = {
    candidate.model_id: candidate for candidate in ANCHOR_CANDIDATES
}
_ROLLING_MINIMUM_SAMPLES: Final = {
    "time_median_60s": 30,
    "time_median_120s": 60,
}


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_date(
    value: str,
    *,
    source: str,
    enforce_development_cutoff: bool,
) -> datetime:
    text = str(value)
    try:
        parsed = datetime.strptime(text, "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {text!r}") from error
    if enforce_development_cutoff and text > MAX_DEVELOPMENT_DATE:
        raise ValueError(
            f"{source} {text} exceeds locked development cutoff {MAX_DEVELOPMENT_DATE}"
        )
    return parsed


def _tod_bucket() -> pl.Expr:
    seconds = pl.col("seconds_from_open")
    return (
        pl.when(seconds < 300)
        .then(pl.lit("outside_selection"))
        .when(seconds < 3_600)
        .then(pl.lit("0905_1000"))
        .when(seconds < 7_200)
        .then(pl.lit("1000_1100"))
        .when(seconds < 10_800)
        .then(pl.lit("1100_1200"))
        .when(seconds < 14_400)
        .then(pl.lit("1200_1300"))
        .otherwise(pl.lit("outside_selection"))
    )


def _dte_bucket() -> pl.Expr:
    dte = pl.col("calendar_dte")
    return (
        pl.when(dte.is_null())
        .then(pl.lit("unknown"))
        .when(dte == 0)
        .then(pl.lit("00"))
        .when(dte <= 2)
        .then(pl.lit("01-02"))
        .when(dte <= 5)
        .then(pl.lit("03-05"))
        .when(dte <= 10)
        .then(pl.lit("06-10"))
        .when(dte <= 20)
        .then(pl.lit("11-20"))
        .when(dte <= 40)
        .then(pl.lit("21-40"))
        .otherwise(pl.lit("41+"))
    )


def _normalise_complete_day(
    day: pl.DataFrame,
    *,
    enforce_development_cutoff: bool,
) -> pl.DataFrame:
    """Validate and normalize one exact 00..15599 product-second grid."""

    if day.is_empty():
        raise ValueError("anchor selection day must not be empty")
    _require(
        day,
        {
            *GROUP_KEYS,
            "seconds_from_open",
            "end_date",
            "basis_mid_bp",
            "eligible_base",
        },
        "anchor selection day",
    )
    date_values = day.select(pl.col("Date").cast(pl.String)).unique()["Date"].to_list()
    if len(date_values) != 1:
        raise ValueError(
            f"anchor selection day must contain exactly one Date, got {date_values[:5]}"
        )
    date_text = str(date_values[0])
    trade_date = _parse_date(
        date_text,
        source="anchor selection day Date",
        enforce_development_cutoff=enforce_development_cutoff,
    ).date()
    frame = day.with_columns(
        pl.lit(date_text).alias("Date"),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("seconds_from_open").cast(pl.Int32),
        pl.col("end_date").cast(pl.Date),
        pl.col("basis_mid_bp").cast(pl.Float64),
        pl.col("eligible_base").fill_null(False).cast(pl.Boolean),
    ).sort([*GROUP_KEYS, "seconds_from_open"])
    invalid_key = frame.filter(
        pl.col("ValueCode").is_null()
        | pl.col("QuoteCode").is_null()
        | (pl.col("ValueCode").str.len_chars() == 0)
        | (pl.col("QuoteCode").str.len_chars() == 0)
    )
    if invalid_key.height:
        raise ValueError("anchor selection day has null or empty product keys")
    duplicate = (
        frame.group_by([*GROUP_KEYS, "seconds_from_open"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError(
            "anchor selection day has duplicate product-second rows: "
            f"{duplicate.head(5).to_dicts()}"
        )
    grid_audit = frame.group_by(GROUP_KEYS).agg(
        pl.len().alias("rows"),
        pl.col("seconds_from_open").n_unique().alias("unique_seconds"),
        pl.col("seconds_from_open").min().alias("first_second"),
        pl.col("seconds_from_open").max().alias("last_second"),
    )
    incomplete = grid_audit.filter(
        (pl.col("rows") != EXPECTED_PRODUCT_ROWS)
        | (pl.col("unique_seconds") != EXPECTED_PRODUCT_ROWS)
        | (pl.col("first_second") != 0)
        | (pl.col("last_second") != SESSION_END_SECOND - 1)
    )
    if incomplete.height:
        raise ValueError(
            "anchor selection day requires the complete 0..15599 one-second "
            f"grid for every product: {incomplete.head(5).to_dicts()}"
        )
    inconsistent_expiry = (
        frame.group_by(GROUP_KEYS)
        .agg(pl.col("end_date").drop_nulls().n_unique().alias("expiry_values"))
        .filter(pl.col("expiry_values") > 1)
    )
    if inconsistent_expiry.height:
        raise ValueError(
            "anchor selection day has inconsistent end_date within a contract"
        )
    frame = frame.with_columns(
        pl.col("end_date").drop_nulls().first().over(GROUP_KEYS).alias("end_date")
    )
    if frame.filter(pl.col("end_date") < pl.lit(trade_date)).height:
        raise ValueError("anchor selection day contains an already-expired contract")
    legal_basis = (
        pl.col("eligible_base")
        & pl.col("basis_mid_bp").is_not_null()
        & pl.col("basis_mid_bp").is_finite()
    ).fill_null(False)
    frame = frame.with_columns(
        pl.when(legal_basis)
        .then(pl.col("basis_mid_bp"))
        .otherwise(None)
        .alias("basis_eval_bp"),
        (
            legal_basis
            & (pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
            & (pl.col("seconds_from_open") < ENTRY_STOP_SECOND)
        ).alias("anchor_selection_eligible"),
        (
            pl.lit(trade_date).cast(pl.Datetime("us"))
            + pl.duration(seconds=pl.col("seconds_from_open"))
        ).alias("_anchor_wall_time"),
        pl.lit(date_text[:6]).alias("month"),
        (pl.col("end_date") - pl.lit(trade_date))
        .dt.total_days()
        .cast(pl.Int16)
        .alias("calendar_dte"),
        _tod_bucket().alias("tod_bucket"),
    ).with_columns(_dte_bucket().alias("dte_bucket"))
    if "analysis_eligible" not in frame.columns:
        frame = frame.with_columns(
            (
                legal_basis & (pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
            ).alias("analysis_eligible")
        )
    return frame


def _candidate_expression(candidate: AnchorCandidate) -> pl.Expr:
    basis = pl.col("basis_eval_bp")
    if candidate.kind == "wall_clock_ewma":
        assert candidate.parameter is not None
        return basis.ewm_mean_by(
            "_anchor_wall_time", half_life=f"{candidate.parameter}s"
        ).over(GROUP_KEYS)
    if candidate.kind == "wall_clock_rolling_median":
        assert candidate.parameter is not None
        return basis.rolling_median_by(
            "_anchor_wall_time",
            window_size=f"{candidate.parameter}s",
            min_samples=_ROLLING_MINIMUM_SAMPLES[candidate.model_id],
            closed="right",
        ).over(GROUP_KEYS)
    if candidate.kind == "legal_observation_ewma":
        assert candidate.parameter is not None
        return basis.ewm_mean(
            half_life=candidate.parameter,
            adjust=False,
            ignore_nulls=True,
        ).over(GROUP_KEYS)
    if candidate.kind == "current_legal_basis":
        return basis
    if candidate.kind == "frozen_opening_median":
        opening = (
            pl.when(pl.col("seconds_from_open") < ANALYSIS_START_SECOND)
            .then(basis)
            .otherwise(None)
            .median()
            .over(GROUP_KEYS)
        )
        return (
            pl.when(pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
            .then(opening)
            .otherwise(None)
        )
    raise ValueError(f"unsupported anchor candidate kind: {candidate.kind}")


def _add_candidates(
    frame: pl.DataFrame,
    model_ids: tuple[str, ...],
) -> pl.DataFrame:
    unknown = sorted(set(model_ids) - set(_CANDIDATE_BY_ID))
    if unknown:
        raise ValueError(f"unknown anchor model ids: {unknown}")
    return frame.with_columns(
        *(
            _candidate_expression(_CANDIDATE_BY_ID[model_id]).alias(
                MODEL_COLUMNS[model_id]
            )
            for model_id in model_ids
        )
    )


def _add_future_labels(frame: pl.DataFrame) -> pl.DataFrame:
    valid_future = pl.col("basis_eval_bp").is_not_null()
    raw_expressions: list[pl.Expr] = []
    for horizon in FUTURE_HORIZONS:
        raw_expressions.extend(
            [
                pl.col("basis_eval_bp")
                .rolling_median(
                    horizon.window_rows,
                    min_samples=horizon.minimum_rows,
                )
                .shift(-horizon.end_seconds)
                .over(GROUP_KEYS)
                .alias(f"_raw_{LABEL_COLUMNS[horizon.horizon_id]}"),
                valid_future.cast(pl.UInt16)
                .rolling_sum(
                    horizon.window_rows,
                    min_samples=horizon.window_rows,
                )
                .shift(-horizon.end_seconds)
                .over(GROUP_KEYS)
                .alias(SAMPLE_COUNT_COLUMNS[horizon.horizon_id]),
            ]
        )
    result = frame.with_columns(*raw_expressions)
    final_expressions: list[pl.Expr] = []
    raw_columns: list[str] = []
    for horizon in FUTURE_HORIZONS:
        label = LABEL_COLUMNS[horizon.horizon_id]
        coverage = COVERAGE_COLUMNS[horizon.horizon_id]
        sample_count = SAMPLE_COUNT_COLUMNS[horizon.horizon_id]
        raw = f"_raw_{label}"
        raw_columns.append(raw)
        final_expressions.extend(
            [
                (pl.col(sample_count) / horizon.window_rows).alias(coverage),
                pl.when(pl.col(sample_count) >= horizon.minimum_rows)
                .then(pl.col(raw))
                .otherwise(None)
                .alias(label),
            ]
        )
    return result.with_columns(*final_expressions).drop(raw_columns)


def build_anchor_selection_grid(day: pl.DataFrame) -> pl.DataFrame:
    """Build every frozen causal anchor and all three future labels."""

    frame = _normalise_complete_day(day, enforce_development_cutoff=True)
    frame = _add_candidates(
        frame, tuple(candidate.model_id for candidate in ANCHOR_CANDIDATES)
    )
    return _add_future_labels(frame)


def materialize_anchor_column(
    day: pl.DataFrame,
    model_id: str,
) -> pl.DataFrame:
    """Materialize one selected anchor without expanding future labels.

    All source columns, including ``timestamp`` and ``analysis_eligible``, are
    retained.  ``selected_anchor_bp`` is a stable runner-facing alias; the
    model-specific column is retained so downstream excursion metadata remains
    explicit.
    """

    if model_id not in _CANDIDATE_BY_ID:
        raise ValueError(f"unknown anchor model id: {model_id}")
    frame = _add_candidates(
        _normalise_complete_day(day, enforce_development_cutoff=False),
        (model_id,),
    )
    column = MODEL_COLUMNS[model_id]
    return frame.with_columns(
        pl.col(column).alias("selected_anchor_bp"),
        pl.lit(model_id).alias("selected_anchor_model"),
    ).drop("_anchor_wall_time")


def _finite(column: str) -> pl.Expr:
    return pl.col(column).is_not_null() & pl.col(column).is_finite()


def _add_common_support_columns(frame: pl.DataFrame) -> pl.DataFrame:
    actionable_anchors = pl.all_horizontal(
        *(_finite(MODEL_COLUMNS[model_id]) for model_id in ACTIONABLE_MODEL_IDS)
    )
    expressions: list[pl.Expr] = []
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        label_gate = (
            pl.col("anchor_selection_eligible")
            & _finite(LABEL_COLUMNS[horizon_id])
            & (pl.col(COVERAGE_COLUMNS[horizon_id]) >= horizon.minimum_coverage)
        ).fill_null(False)
        expressions.extend(
            [
                label_gate.alias(f"_label_gate_{horizon_id}"),
                (label_gate & actionable_anchors)
                .fill_null(False)
                .alias(f"_common_gate_{horizon_id}"),
            ]
        )
    result = frame.with_columns(*expressions)
    lagged: list[pl.Expr] = [
        pl.col("seconds_from_open").diff().over(GROUP_KEYS).alias("_second_diff"),
        pl.col("tod_bucket").shift(1).over(GROUP_KEYS).alias("_previous_tod_bucket"),
    ]
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        lagged.append(
            pl.col(f"_common_gate_{horizon_id}")
            .shift(1)
            .over(GROUP_KEYS)
            .fill_null(False)
            .alias(f"_previous_common_gate_{horizon_id}")
        )
    return result.with_columns(*lagged)


_META_KEYS: Final = (
    "Date",
    "month",
    "ValueCode",
    "QuoteCode",
    "calendar_dte",
    "dte_bucket",
)


def _scope_keys(by_tod: bool) -> list[str]:
    return [*_META_KEYS, "tod_bucket"] if by_tod else list(_META_KEYS)


def _scope_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        (pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
        & (pl.col("seconds_from_open") < ENTRY_STOP_SECOND)
    )


def _support_summary(frame: pl.DataFrame, *, by_tod: bool) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.len().alias("n_grid"),
        pl.col("anchor_selection_eligible").sum().alias("n_current_base_eligible"),
    ]
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        expressions.extend(
            [
                pl.col(f"_label_gate_{horizon_id}")
                .sum()
                .alias(f"n_label_eligible__{horizon_id}"),
                pl.col(f"_common_gate_{horizon_id}")
                .sum()
                .alias(f"n_common_evaluable__{horizon_id}"),
                pl.col(COVERAGE_COLUMNS[horizon_id])
                .filter(pl.col("anchor_selection_eligible"))
                .mean()
                .alias(f"mean_future_coverage__{horizon_id}"),
            ]
        )
    wide = _scope_frame(frame).group_by(_scope_keys(by_tod)).agg(expressions)
    parts: list[pl.DataFrame] = []
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        part = wide.select(
            *_scope_keys(by_tod),
            "n_grid",
            "n_current_base_eligible",
            pl.col(f"n_label_eligible__{horizon_id}").alias("n_label_eligible"),
            pl.col(f"n_common_evaluable__{horizon_id}").alias("n_common_evaluable"),
            pl.col(f"mean_future_coverage__{horizon_id}").alias("mean_future_coverage"),
        )
        if not by_tod:
            part = part.with_columns(pl.lit("all").alias("tod_bucket"))
        part = part.with_columns(
            pl.lit(horizon_id).alias("horizon_id"),
            pl.lit(horizon.start_seconds).cast(pl.Int16).alias("horizon_start_seconds"),
            pl.lit(horizon.end_seconds).cast(pl.Int16).alias("horizon_end_seconds"),
            pl.lit(horizon.minimum_coverage).alias("minimum_coverage"),
            pl.when(pl.col("n_current_base_eligible") > 0)
            .then(pl.col("n_label_eligible") / pl.col("n_current_base_eligible"))
            .otherwise(None)
            .alias("label_coverage_rate"),
            pl.when(pl.col("n_label_eligible") > 0)
            .then(pl.col("n_common_evaluable") / pl.col("n_label_eligible"))
            .otherwise(None)
            .alias("all_model_common_rate"),
        )
        parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed")


def _model_scope_summary(
    frame: pl.DataFrame,
    candidate: AnchorCandidate,
    *,
    by_tod: bool,
) -> pl.DataFrame:
    model_column = MODEL_COLUMNS[candidate.model_id]
    working = frame.with_columns(
        pl.col(model_column).diff().over(GROUP_KEYS).alias("_anchor_delta_bp"),
        pl.col("basis_eval_bp").diff().over(GROUP_KEYS).alias("_basis_delta_bp"),
        _finite(model_column)
        .shift(1)
        .over(GROUP_KEYS)
        .fill_null(False)
        .alias("_previous_model_finite"),
    )
    expressions: list[pl.Expr] = [
        pl.len().alias("n_grid"),
        pl.col("anchor_selection_eligible").sum().alias("n_current_base_eligible"),
    ]
    metric_names = (
        "n_label_eligible",
        "n_native_evaluable",
        "n_common_evaluable",
        "error_sum_bp",
        "abs_error_sum_bp",
        "squared_error_sum_bp2",
        "p95_abs_error_bp",
        "max_abs_error_bp",
        "n_adjacent_common",
        "anchor_tv_bp",
        "basis_tv_bp",
    )
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        label_gate = pl.col(f"_label_gate_{horizon_id}")
        native = label_gate & _finite(model_column)
        common = pl.col(f"_common_gate_{horizon_id}") & _finite(model_column)
        error = pl.col(model_column) - pl.col(LABEL_COLUMNS[horizon_id])
        adjacent = (
            common
            & pl.col(f"_previous_common_gate_{horizon_id}")
            & pl.col("_previous_model_finite")
            & (pl.col("_second_diff") == 1)
        )
        if by_tod:
            adjacent = adjacent & (
                pl.col("_previous_tod_bucket") == pl.col("tod_bucket")
            )
        suffix = f"__{horizon_id}"
        expressions.extend(
            [
                label_gate.sum().alias(f"n_label_eligible{suffix}"),
                native.sum().alias(f"n_native_evaluable{suffix}"),
                common.sum().alias(f"n_common_evaluable{suffix}"),
                error.filter(common).sum().alias(f"error_sum_bp{suffix}"),
                error.abs().filter(common).sum().alias(f"abs_error_sum_bp{suffix}"),
                error.pow(2)
                .filter(common)
                .sum()
                .alias(f"squared_error_sum_bp2{suffix}"),
                error.abs()
                .filter(common)
                .quantile(0.95, interpolation="linear")
                .alias(f"p95_abs_error_bp{suffix}"),
                error.abs().filter(common).max().alias(f"max_abs_error_bp{suffix}"),
                adjacent.sum().alias(f"n_adjacent_common{suffix}"),
                pl.col("_anchor_delta_bp")
                .abs()
                .filter(adjacent)
                .sum()
                .alias(f"anchor_tv_bp{suffix}"),
                pl.col("_basis_delta_bp")
                .abs()
                .filter(adjacent)
                .sum()
                .alias(f"basis_tv_bp{suffix}"),
            ]
        )
    wide = _scope_frame(working).group_by(_scope_keys(by_tod)).agg(expressions)
    parts: list[pl.DataFrame] = []
    for horizon in FUTURE_HORIZONS:
        horizon_id = horizon.horizon_id
        part = wide.select(
            *_scope_keys(by_tod),
            "n_grid",
            "n_current_base_eligible",
            *(
                pl.col(f"{metric}__{horizon_id}").alias(metric)
                for metric in metric_names
            ),
        )
        if not by_tod:
            part = part.with_columns(pl.lit("all").alias("tod_bucket"))
        part = part.with_columns(
            pl.lit(candidate.model_id).alias("model"),
            pl.lit(candidate.kind).alias("model_kind"),
            pl.lit(candidate.q_eligible).alias("q_eligible"),
            pl.lit(horizon_id).alias("horizon_id"),
            pl.when(pl.col("n_common_evaluable") > 0)
            .then(pl.col("error_sum_bp") / pl.col("n_common_evaluable"))
            .otherwise(None)
            .alias("mean_error_bp"),
            pl.when(pl.col("n_common_evaluable") > 0)
            .then(pl.col("abs_error_sum_bp") / pl.col("n_common_evaluable"))
            .otherwise(None)
            .alias("mae_bp"),
            pl.when(pl.col("n_common_evaluable") > 0)
            .then(
                (pl.col("squared_error_sum_bp2") / pl.col("n_common_evaluable")).sqrt()
            )
            .otherwise(None)
            .alias("rmse_bp"),
            pl.when(pl.col("n_label_eligible") > 0)
            .then(pl.col("n_native_evaluable") / pl.col("n_label_eligible"))
            .otherwise(None)
            .alias("native_support_rate"),
            pl.when(pl.col("n_label_eligible") > 0)
            .then(pl.col("n_common_evaluable") / pl.col("n_label_eligible"))
            .otherwise(None)
            .alias("common_support_rate"),
            pl.when(pl.col("basis_tv_bp") > 0)
            .then(pl.col("anchor_tv_bp") / pl.col("basis_tv_bp"))
            .otherwise(None)
            .alias("anchor_tv_over_basis_tv"),
        )
        parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed")


def evaluate_anchor_selection_day(
    day: pl.DataFrame,
) -> AnchorSelectionDayEvaluation:
    """Return exact product-day facts on one common base-label support.

    Every model for a given horizon/TOD cell uses the same rows.  Native model
    availability is intentionally not allowed to change a comparison sample.
    The sufficient statistics retain exact error sums for equal-weight
    product-day/month aggregation and direct product-day p95 diagnostics.
    """

    frame = _add_common_support_columns(build_anchor_selection_grid(day))
    support = pl.concat(
        [
            _support_summary(frame, by_tod=False),
            _support_summary(frame, by_tod=True),
        ],
        how="diagonal_relaxed",
    ).sort([*GROUP_KEYS, "horizon_id", "tod_bucket"])
    model_parts: list[pl.DataFrame] = []
    for candidate in ANCHOR_CANDIDATES:
        model_parts.extend(
            [
                _model_scope_summary(frame, candidate, by_tod=False),
                _model_scope_summary(frame, candidate, by_tod=True),
            ]
        )
    product_day = pl.concat(model_parts, how="diagonal_relaxed").sort(
        [*GROUP_KEYS, "model", "horizon_id", "tod_bucket"]
    )
    return AnchorSelectionDayEvaluation(
        product_day_stats=product_day,
        support_stats=support,
    )


__all__ = [
    "ACTIONABLE_MODEL_IDS",
    "ANCHOR_CANDIDATES",
    "FUTURE_HORIZONS",
    "LABEL_COLUMNS",
    "MODEL_COLUMNS",
    "AnchorSelectionDayEvaluation",
    "build_anchor_selection_grid",
    "evaluate_anchor_selection_day",
    "materialize_anchor_column",
]
