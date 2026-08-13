"""Accuracy, stability, and mean-reversion metrics for fair-mid anchors."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .anchors import ANCHOR_COLUMNS, CENTER_MIN_COVERAGE, GROUP_KEYS


MODEL_NAMES = {
    column: column.removeprefix("anchor_").removesuffix("_bp")
    for column in ANCHOR_COLUMNS
}
CHANGE_EPS_BP = 1e-9
ACTIONABLE_MOVE_BP = 1.0


@dataclass(frozen=True)
class FairMetricResult:
    metrics_by_model: pl.DataFrame
    metrics_by_day_symbol: pl.DataFrame
    residual_bins: pl.DataFrame
    coverage: pl.DataFrame


@dataclass(frozen=True)
class EndpointFreshnessResult:
    summary: pl.DataFrame
    residual_bins: pl.DataFrame


@dataclass(frozen=True)
class DelayedReversionResult:
    summary: pl.DataFrame
    residual_bins: pl.DataFrame


def _residual_bucket() -> pl.Expr:
    absolute = pl.col("residual_bp").abs()
    return (
        pl.when(absolute < 5)
        .then(pl.lit("00_05bp"))
        .when(absolute < 10)
        .then(pl.lit("05_10bp"))
        .when(absolute < 20)
        .then(pl.lit("10_20bp"))
        .when(absolute < 40)
        .then(pl.lit("20_40bp"))
        .otherwise(pl.lit("40bp_plus"))
    )


def candidate_long(panel: pl.DataFrame) -> pl.DataFrame:
    """Convert wide candidate columns into an auditable model panel."""
    base_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "analysis_eligible",
        "basis_eval_bp",
        "future_center_bp",
        "future_center_coverage",
        "future_300s_bp",
        "future_300s_coverage",
    ]
    base_columns.extend(
        column
        for column in panel.columns
        if column not in base_columns
        and (
            column == "leg_skew_ms"
            or column == "eligible_base"
            or column.startswith("eligible_") and column.endswith("ms")
        )
    )
    missing = sorted(set(base_columns + list(ANCHOR_COLUMNS)) - set(panel.columns))
    if missing:
        raise ValueError(f"fair panel missing columns: {missing}")

    frames = []
    for column, model in MODEL_NAMES.items():
        frames.append(
            panel.select(
                *base_columns,
                pl.lit(model).alias("model"),
                pl.col(column).alias("anchor_bp"),
            )
        )
    long = pl.concat(frames).sort(["model", *GROUP_KEYS, "timestamp"])
    long = long.with_columns(
        (pl.col("anchor_bp") - pl.col("future_center_bp")).alias("error_bp"),
        (pl.col("basis_eval_bp") - pl.col("anchor_bp")).alias("residual_bp"),
        pl.col("anchor_bp")
        .diff()
        .over(["model", *GROUP_KEYS])
        .alias("anchor_change_bp"),
        pl.col("basis_eval_bp")
        .diff()
        .over(["model", *GROUP_KEYS])
        .alias("basis_change_bp"),
    )
    return long.with_columns(
        pl.col("error_bp").abs().alias("abs_error_bp"),
        pl.when(pl.col("residual_bp").abs() > CHANGE_EPS_BP)
        .then(
            pl.col("residual_bp").sign()
            * (pl.col("basis_eval_bp") - pl.col("future_300s_bp"))
        )
        .otherwise(None)
        .alias("mean_reversion_300s_bp"),
        _residual_bucket().alias("residual_bucket"),
    )


def _metric_aggregations() -> list[pl.Expr]:
    return [
        pl.len().alias("n"),
        pl.col("Date").n_unique().alias("dates"),
        pl.col("ValueCode").n_unique().alias("symbols"),
        pl.col("error_bp").mean().alias("bias_bp"),
        pl.col("abs_error_bp").mean().alias("mae_bp"),
        pl.col("abs_error_bp").quantile(0.80, interpolation="nearest").alias("p80_abs_error_bp"),
        pl.col("abs_error_bp").quantile(0.95, interpolation="nearest").alias("p95_abs_error_bp"),
        pl.col("anchor_change_bp").abs().sum().alias("anchor_tv_bp"),
        pl.col("basis_change_bp").abs().sum().alias("basis_tv_bp"),
        (pl.col("anchor_change_bp").abs() > CHANGE_EPS_BP)
        .sum()
        .alias("anchor_nonzero_changes"),
        (pl.col("anchor_change_bp").abs() >= ACTIONABLE_MOVE_BP)
        .sum()
        .alias("anchor_moves_ge_1bp"),
        pl.col("anchor_change_bp").abs().quantile(0.95, interpolation="nearest").alias("p95_anchor_change_bp"),
        pl.col("mean_reversion_300s_bp").count().alias("n_reversion_300s"),
        pl.col("mean_reversion_300s_bp").mean().alias("mean_reversion_300s_bp"),
        (pl.col("mean_reversion_300s_bp") > 0).mean().alias("p_toward_anchor_300s"),
        (pl.col("mean_reversion_300s_bp").abs() <= CHANGE_EPS_BP)
        .mean()
        .alias("p_flat_300s"),
        (pl.col("mean_reversion_300s_bp") < -CHANGE_EPS_BP)
        .mean()
        .alias("p_away_from_anchor_300s"),
        (
            (pl.col("mean_reversion_300s_bp") > CHANGE_EPS_BP).sum()
            / (pl.col("mean_reversion_300s_bp").abs() > CHANGE_EPS_BP).sum()
        ).alias("p_toward_given_move_300s"),
    ]


def _finish_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("basis_tv_bp") > CHANGE_EPS_BP)
        .then(pl.col("anchor_tv_bp") / pl.col("basis_tv_bp"))
        .otherwise(None)
        .alias("tv_ratio"),
        (pl.col("anchor_nonzero_changes") / (pl.col("n") / 60))
        .alias("anchor_nonzero_changes_per_minute"),
        (pl.col("anchor_moves_ge_1bp") / (pl.col("n") / 60))
        .alias("anchor_moves_ge_1bp_per_minute"),
    )


def summarize_fair_panel(
    panel: pl.DataFrame,
    eligibility_column: str | None = None,
    max_leg_skew_ms: int | None = None,
) -> FairMetricResult:
    """Summarize all anchor candidates on common causal evaluation rules."""
    long = candidate_long(panel)
    if eligibility_column is None:
        eligibility = pl.col("analysis_eligible")
    else:
        if eligibility_column not in long.columns:
            raise ValueError(f"missing evaluation eligibility: {eligibility_column}")
        eligibility = (
            pl.col(eligibility_column) & (pl.col("seconds_from_open") >= 300)
        )
    if max_leg_skew_ms is not None:
        eligibility = eligibility & (pl.col("leg_skew_ms") <= max_leg_skew_ms)
    evaluation = long.filter(
        eligibility
        & pl.col("anchor_bp").is_not_null()
        & pl.col("future_center_bp").is_not_null()
        & (pl.col("future_center_coverage") >= CENTER_MIN_COVERAGE)
    )
    by_model = _finish_metrics(
        evaluation.group_by("model").agg(_metric_aggregations()).sort("model")
    )
    by_day_symbol = _finish_metrics(
        evaluation.group_by(["model", "Date", "ValueCode", "QuoteCode"])
        .agg(_metric_aggregations())
        .sort(["model", "Date", "ValueCode"])
    )
    residual_rows = evaluation.filter(
        pl.col("future_300s_bp").is_not_null()
        & (pl.col("future_300s_coverage") >= 0.60)
        & (pl.col("residual_bp").abs() > CHANGE_EPS_BP)
    )
    residual_bins = (
        residual_rows.group_by(["model", "residual_bucket"])
        .agg(
            pl.len().alias("n"),
            pl.col("residual_bp").abs().mean().alias("mean_abs_residual_bp"),
            pl.col("mean_reversion_300s_bp").mean().alias("mean_reversion_300s_bp"),
            (pl.col("mean_reversion_300s_bp") > 0).mean().alias("p_toward_anchor_300s"),
            (pl.col("mean_reversion_300s_bp").abs() <= CHANGE_EPS_BP)
            .mean()
            .alias("p_flat_300s"),
            (
                (pl.col("mean_reversion_300s_bp") > CHANGE_EPS_BP).sum()
                / (pl.col("mean_reversion_300s_bp").abs() > CHANGE_EPS_BP).sum()
            ).alias("p_toward_given_move_300s"),
        )
        .sort(["model", "residual_bucket"])
    )
    coverage_columns = [
        column
        for column in panel.columns
        if column == "eligible_base" or column.startswith("eligible_") and column.endswith("ms")
    ]
    coverage = (
        panel.group_by("Date")
        .agg(
            pl.len().alias("landmark_rows"),
            pl.col("ValueCode").n_unique().alias("symbols"),
            *[pl.col(column).mean().alias(f"{column}_rate") for column in coverage_columns],
            pl.col("future_center_bp").is_not_null().mean().alias("future_center_label_rate"),
            pl.col("analysis_eligible").sum().alias("current_eligible_rows"),
            (
                pl.col("analysis_eligible")
                & pl.col("future_center_bp").is_not_null()
                & (pl.col("future_center_coverage") >= CENTER_MIN_COVERAGE)
            ).sum().alias("evaluable_rows"),
        )
        .sort("Date")
    )
    coverage = coverage.with_columns(
        (pl.col("evaluable_rows") / pl.col("current_eligible_rows"))
        .alias("label_rate_given_current_eligible")
    )
    return FairMetricResult(
        metrics_by_model=by_model,
        metrics_by_day_symbol=by_day_symbol,
        residual_bins=residual_bins,
        coverage=coverage,
    )


def _with_exact_future_state(
    panel: pl.DataFrame,
    horizon_seconds: int,
    suffix: str,
) -> pl.DataFrame:
    """Join the state at an exact future timestamp, independent of grid interval."""
    if horizon_seconds <= 0:
        raise ValueError("future horizon must be positive")
    state_columns = (
        "basis_mid_bp",
        "leg_skew_ms",
        "eligible_base",
        "eligible_100ms",
        "eligible_1000ms",
    )
    join_time = f"_future_timestamp_{suffix}"
    future = panel.select(
        *GROUP_KEYS,
        pl.col("timestamp").cast(pl.Datetime("ns")).alias(join_time),
        *[
            pl.col(column).alias(f"{column}_{suffix}")
            for column in state_columns
        ],
    )
    return (
        panel.with_columns(
            (
                pl.col("timestamp").dt.epoch("ns")
                + horizon_seconds * 1_000_000_000
            )
            .cast(pl.Datetime("ns"))
            .alias(join_time)
        )
        .join(
            future,
            on=[*GROUP_KEYS, join_time],
            how="left",
            validate="m:1",
        )
        .drop(join_time)
    )


def _summarize_reversion_rows(
    rows: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    grouping = ["sample"]
    if "residual_side" in rows.columns:
        grouping.append("residual_side")
    direction_count = (
        pl.col("mean_reversion_bp").abs() > CHANGE_EPS_BP
    ).sum()
    summary = (
        rows.group_by(grouping)
        .agg(
            pl.len().alias("n"),
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("symbols"),
            pl.col("mean_reversion_bp").mean().alias("mean_reversion_bp"),
            pl.col("mean_reversion_bp").median().alias("median_reversion_bp"),
            (pl.col("mean_reversion_bp") > CHANGE_EPS_BP)
            .mean()
            .alias("p_toward_anchor"),
            (pl.col("mean_reversion_bp").abs() <= CHANGE_EPS_BP)
            .mean()
            .alias("p_flat"),
            (
                (pl.col("mean_reversion_bp") > CHANGE_EPS_BP).sum()
                / direction_count
            ).alias("p_toward_given_move"),
        )
        .sort(grouping)
    )
    residual_bins = (
        rows.group_by([*grouping, "residual_bucket"])
        .agg(
            pl.len().alias("n"),
            pl.col("residual_bp").abs().mean().alias("mean_abs_residual_bp"),
            pl.col("mean_reversion_bp").mean().alias("mean_reversion_bp"),
            (pl.col("mean_reversion_bp") > CHANGE_EPS_BP)
            .mean()
            .alias("p_toward_anchor"),
            (pl.col("mean_reversion_bp").abs() <= CHANGE_EPS_BP)
            .mean()
            .alias("p_flat"),
            (
                (pl.col("mean_reversion_bp") > CHANGE_EPS_BP).sum()
                / (pl.col("mean_reversion_bp").abs() > CHANGE_EPS_BP).sum()
            ).alias("p_toward_given_move"),
        )
        .sort([*grouping, "residual_bucket"])
    )
    return summary, residual_bins


def summarize_fresh_endpoint_reversion(
    panel: pl.DataFrame,
    anchor_column: str = "anchor_ewma_120s_bp",
    horizon_seconds: int = 300,
) -> EndpointFreshnessResult:
    """Require both current and future endpoints to satisfy the same freshness gate."""
    required = {
        *GROUP_KEYS,
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        "leg_skew_ms",
        "eligible_base",
        "eligible_100ms",
        "eligible_1000ms",
        anchor_column,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"fair panel missing endpoint columns: {missing}")

    specs = (
        ("both_endpoints_base", "eligible_base", None),
        ("both_endpoints_age_1000ms", "eligible_1000ms", None),
        ("both_endpoints_age_1000ms_skew_100ms", "eligible_1000ms", 100),
        ("both_endpoints_age_100ms", "eligible_100ms", None),
    )
    endpoint_panel = _with_exact_future_state(
        panel,
        horizon_seconds=horizon_seconds,
        suffix="endpoint",
    )
    frames: list[pl.DataFrame] = []
    for sample, eligibility_column, max_skew_ms in specs:
        current_eligible = pl.col(eligibility_column)
        future_eligible = pl.col(f"{eligibility_column}_endpoint")
        if max_skew_ms is not None:
            current_eligible = current_eligible & (
                pl.col("leg_skew_ms") <= max_skew_ms
            )
            future_eligible = future_eligible & (
                pl.col("leg_skew_ms_endpoint") <= max_skew_ms
            )
        frame = (
            endpoint_panel.with_columns(
                (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias(
                    "residual_bp"
                ),
                (current_eligible & future_eligible)
                .fill_null(False)
                .alias("endpoint_eligible"),
            )
            .filter(
                pl.col("endpoint_eligible")
                & (pl.col("seconds_from_open") >= 300)
                & pl.col(anchor_column).is_not_null()
                & pl.col("basis_mid_bp").is_not_null()
                & pl.col("basis_mid_bp_endpoint").is_not_null()
                & (pl.col("residual_bp").abs() > CHANGE_EPS_BP)
            )
            .with_columns(
                (
                    pl.col("residual_bp").sign()
                    * (
                        pl.col("basis_mid_bp")
                        - pl.col("basis_mid_bp_endpoint")
                    )
                ).alias("mean_reversion_bp"),
                _residual_bucket().alias("residual_bucket"),
                pl.lit(sample).alias("sample"),
            )
            .select(
                "sample",
                "Date",
                "ValueCode",
                "residual_bucket",
                "residual_bp",
                "mean_reversion_bp",
            )
        )
        frames.append(frame)

    endpoints = pl.concat(frames)
    summary, residual_bins = _summarize_reversion_rows(endpoints)
    return EndpointFreshnessResult(summary=summary, residual_bins=residual_bins)


def summarize_delayed_reversion(
    panel: pl.DataFrame,
    anchor_column: str = "anchor_ewma_120s_bp",
    start_seconds: int = 30,
    end_seconds: int = 300,
) -> DelayedReversionResult:
    """Measure movement after a delay so the outcome does not reuse current basis."""
    if start_seconds >= end_seconds:
        raise ValueError("delayed reversion requires start before end")
    required = {
        *GROUP_KEYS,
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        "leg_skew_ms",
        "eligible_base",
        "eligible_100ms",
        "eligible_1000ms",
        anchor_column,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"fair panel missing delayed reversion columns: {missing}")

    delayed = _with_exact_future_state(panel, start_seconds, "start")
    delayed = _with_exact_future_state(delayed, end_seconds, "end")
    specs = (
        ("three_points_base", "eligible_base", None),
        ("three_points_age_1000ms", "eligible_1000ms", None),
        ("three_points_age_1000ms_skew_100ms", "eligible_1000ms", 100),
        ("three_points_age_100ms", "eligible_100ms", None),
    )
    frames: list[pl.DataFrame] = []
    for sample, eligibility_column, max_skew_ms in specs:
        eligibility = (
            pl.col(eligibility_column)
            & pl.col(f"{eligibility_column}_start")
            & pl.col(f"{eligibility_column}_end")
        )
        if max_skew_ms is not None:
            eligibility = eligibility & (
                (pl.col("leg_skew_ms") <= max_skew_ms)
                & (pl.col("leg_skew_ms_start") <= max_skew_ms)
                & (pl.col("leg_skew_ms_end") <= max_skew_ms)
            )
        frame = (
            delayed.with_columns(
                (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias(
                    "residual_bp"
                ),
                eligibility.fill_null(False).alias("delayed_eligible"),
            )
            .filter(
                pl.col("delayed_eligible")
                & (pl.col("seconds_from_open") >= 300)
                & pl.col(anchor_column).is_not_null()
                & pl.col("basis_mid_bp_start").is_not_null()
                & pl.col("basis_mid_bp_end").is_not_null()
                & (pl.col("residual_bp").abs() > CHANGE_EPS_BP)
            )
            .with_columns(
                (
                    pl.col("residual_bp").sign()
                    * (
                        pl.col("basis_mid_bp_start")
                        - pl.col("basis_mid_bp_end")
                    )
                ).alias("mean_reversion_bp"),
                _residual_bucket().alias("residual_bucket"),
                pl.when(pl.col("residual_bp") > 0)
                .then(pl.lit("positive"))
                .otherwise(pl.lit("negative"))
                .alias("residual_side"),
                pl.lit(sample).alias("sample"),
            )
            .select(
                "sample",
                "Date",
                "ValueCode",
                "residual_side",
                "residual_bucket",
                "residual_bp",
                "mean_reversion_bp",
            )
        )
        frames.append(frame)

    directional_rows = pl.concat(frames)
    rows = pl.concat(
        [
            directional_rows,
            directional_rows.with_columns(
                pl.lit("all").alias("residual_side")
            ),
        ]
    )
    summary, residual_bins = _summarize_reversion_rows(rows)
    rename = {
        "mean_reversion_bp": "mean_signal_consistent_move_bp",
        "median_reversion_bp": "median_signal_consistent_move_bp",
        "p_toward_anchor": "p_signal_consistent",
        "p_toward_given_move": "p_signal_consistent_given_move",
    }
    return DelayedReversionResult(
        summary=summary.rename(rename),
        residual_bins=residual_bins.rename(
            {
                key: value
                for key, value in rename.items()
                if key in residual_bins.columns
            }
        ),
    )
