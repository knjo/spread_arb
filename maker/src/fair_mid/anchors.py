"""Causal fair-mid anchor candidates and forward evaluation labels."""

from __future__ import annotations

import polars as pl


GROUP_KEYS = ["Date", "ValueCode"]
CENTER_START_SECONDS = 30
CENTER_END_SECONDS = 300
CENTER_MIN_COVERAGE = 0.90
LOCAL_LABEL_RADIUS_SECONDS = 2
HORIZONS_SECONDS = (30, 60, 180, 300)
ANCHOR_COLUMNS = (
    "anchor_persistence_bp",
    "anchor_open_5m_bp",
    "anchor_ewma_30s_bp",
    "anchor_ewma_120s_bp",
    "anchor_ewma_300s_bp",
    "anchor_rolling_median_300s_bp",
    "anchor_expanding_median_bp",
)


def _future_window_median(
    column: str,
    start_seconds: int,
    end_seconds: int,
    min_coverage: float,
) -> pl.Expr:
    window = end_seconds - start_seconds + 1
    min_samples = max(1, int(window * min_coverage + 0.999999))
    return (
        pl.col(column)
        .rolling_median(window, min_samples=min_samples)
        .shift(-end_seconds)
        .over(GROUP_KEYS)
    )


def _future_window_coverage(
    column: str,
    start_seconds: int,
    end_seconds: int,
) -> pl.Expr:
    window = end_seconds - start_seconds + 1
    return (
        pl.col(column)
        .is_not_null()
        .cast(pl.UInt16)
        .rolling_sum(window, min_samples=window)
        .shift(-end_seconds)
        .over(GROUP_KEYS)
        / window
    )


def add_forward_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Add fixed-grid future labels without using event-frequency weighting."""
    expressions: list[pl.Expr] = [
        _future_window_median(
            "basis_eval_bp",
            CENTER_START_SECONDS,
            CENTER_END_SECONDS,
            CENTER_MIN_COVERAGE,
        ).alias("future_center_bp"),
        _future_window_coverage(
            "basis_eval_bp", CENTER_START_SECONDS, CENTER_END_SECONDS
        ).alias("future_center_coverage"),
    ]
    local_window = LOCAL_LABEL_RADIUS_SECONDS * 2 + 1
    for horizon in HORIZONS_SECONDS:
        end_offset = horizon + LOCAL_LABEL_RADIUS_SECONDS
        expressions.extend(
            [
                (
                    pl.col("basis_eval_bp")
                    .rolling_median(local_window, min_samples=3)
                    .shift(-end_offset)
                    .over(GROUP_KEYS)
                ).alias(f"future_{horizon}s_bp"),
                (
                    pl.col("basis_eval_bp")
                    .is_not_null()
                    .cast(pl.UInt8)
                    .rolling_sum(local_window, min_samples=local_window)
                    .shift(-end_offset)
                    .over(GROUP_KEYS)
                    / local_window
                ).alias(f"future_{horizon}s_coverage"),
            ]
        )
    return frame.with_columns(expressions)


def add_anchor_candidates(frame: pl.DataFrame) -> pl.DataFrame:
    """Add simple causal baselines for the first fair-mid pilot."""
    opening_condition = pl.col("seconds_from_open") < 300
    opening_anchor = (
        pl.when(opening_condition)
        .then(pl.col("basis_eval_bp"))
        .otherwise(None)
        .median()
        .over(GROUP_KEYS)
    )
    return frame.with_columns(
        pl.col("basis_eval_bp").alias("anchor_persistence_bp"),
        pl.when(pl.col("seconds_from_open") >= 300)
        .then(opening_anchor)
        .otherwise(None)
        .alias("anchor_open_5m_bp"),
        pl.col("basis_eval_bp")
        .ewm_mean(half_life=30, adjust=False, ignore_nulls=True)
        .over(GROUP_KEYS)
        .alias("anchor_ewma_30s_bp"),
        pl.col("basis_eval_bp")
        .ewm_mean(half_life=120, adjust=False, ignore_nulls=True)
        .over(GROUP_KEYS)
        .alias("anchor_ewma_120s_bp"),
        pl.col("basis_eval_bp")
        .ewm_mean(half_life=300, adjust=False, ignore_nulls=True)
        .over(GROUP_KEYS)
        .alias("anchor_ewma_300s_bp"),
        pl.col("basis_eval_bp")
        .rolling_median(300, min_samples=60)
        .over(GROUP_KEYS)
        .alias("anchor_rolling_median_300s_bp"),
        pl.col("basis_eval_bp")
        .rolling_median(15_300, min_samples=300)
        .over(GROUP_KEYS)
        .alias("anchor_expanding_median_bp"),
    )


def prepare_fair_panel(
    landmarks: pl.DataFrame,
    primary_age_ms: int | None = None,
) -> pl.DataFrame:
    """Create an ordered candidate/label panel from causal basis landmarks."""
    return add_forward_labels(
        prepare_causal_fair_panel(
            landmarks,
            primary_age_ms=primary_age_ms,
        )
    )


def prepare_causal_fair_panel(
    landmarks: pl.DataFrame,
    primary_age_ms: int | None = None,
) -> pl.DataFrame:
    """Create anchors and eligibility without materialising future labels.

    This is the scalable daily input for rolling boundaries and raw quote
    replay.  The pilot scorer can call :func:`prepare_fair_panel` when forward
    evaluation labels are actually required.
    """
    eligible_column = (
        "eligible_base" if primary_age_ms is None else f"eligible_{primary_age_ms}ms"
    )
    if eligible_column not in landmarks.columns:
        raise ValueError(f"missing freshness column: {eligible_column}")
    required = set(GROUP_KEYS + ["timestamp", "basis_mid_bp", eligible_column])
    missing = sorted(required - set(landmarks.columns))
    if missing:
        raise ValueError(f"landmarks missing columns: {missing}")

    panel = landmarks.sort(GROUP_KEYS + ["timestamp"]).with_columns(
        pl.when(pl.col(eligible_column))
        .then(pl.col("basis_mid_bp"))
        .otherwise(None)
        .alias("basis_eval_bp"),
        (
            pl.col(eligible_column)
            & (pl.col("seconds_from_open") >= 300)
        ).alias("analysis_eligible"),
    )
    return add_anchor_candidates(panel)
