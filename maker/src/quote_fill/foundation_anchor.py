"""Single-session, leakage-safe anchor validation sufficient statistics.

This module intentionally has no file-system runner.  It accepts one complete
one-second session at a time, builds the frozen anchor candidates, and returns
small product-day sufficient-statistic tables.  Callers must keep the
2026-08-14 and later locked-forward sessions outside this development study.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import polars as pl

MAX_DEVELOPMENT_DATE: Final = "20260813"
ANALYSIS_START_SECOND: Final = 300
SESSION_END_SECOND: Final = 15_600
FUTURE_CENTER_START_SECOND: Final = 30
FUTURE_CENTER_END_SECOND: Final = 300
FUTURE_CENTER_WINDOW_ROWS: Final = 271
FUTURE_CENTER_MIN_ROWS: Final = 244
FUTURE_CENTER_MIN_COVERAGE: Final = 0.90
NONOVERLAP_LOCKOUT_SECONDS: Final = 300
PRIOR_MIN_COVERAGE: Final = 0.80
CHANGE_EPS_BP: Final = 1e-9

GROUP_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
SUMMARY_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "model",
    "freshness_sample",
    "stratum_family",
    "stratum_value",
)
STRATUM_FAMILIES: Final = ("overall", "tod", "dte")

MODEL_COLUMNS: Final = {
    "persistence": "anchor_persistence_bp",
    "open_5m": "anchor_open_5m_bp",
    "ewma_30s": "anchor_ewma_30s_bp",
    "ewma_120s": "anchor_ewma_120s_bp",
    "ewma_300s": "anchor_ewma_300s_bp",
    "rolling_median_300s": "anchor_rolling_median_300s_bp",
    "expanding_median": "anchor_expanding_median_bp",
    "prior_seeded_ewma_120s": "anchor_prior_seeded_ewma_120s_bp",
}
BASELINE_MODEL: Final = "ewma_120s"
BASELINE_COLUMN: Final = MODEL_COLUMNS[BASELINE_MODEL]

FRESHNESS_SAMPLES: Final = (
    "base",
    "age_1000ms",
    "age_1000ms_skew_100ms",
    "age_100ms",
)
# Model selection is evaluated on the base legal-book sample.  Freshness is a
# robustness test of the production baseline, not a second model-selection
# grid; crossing every challenger with every gate would add 21 redundant
# full-market scans per day and invite selection on a sparse Cartesian table.
STRICT_FRESHNESS_MODELS: Final = (BASELINE_MODEL,)


@dataclass(frozen=True)
class AnchorDayEvaluation:
    """Compact outputs for one evaluated market session."""

    anchor_daily: pl.DataFrame
    delayed_daily: pl.DataFrame
    coverage: pl.DataFrame


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_date(value: str, *, source: str) -> datetime:
    text = str(value)
    try:
        parsed = datetime.strptime(text, "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {text!r}") from error
    if text > MAX_DEVELOPMENT_DATE:
        raise ValueError(
            f"{source} {text} exceeds locked development cutoff "
            f"{MAX_DEVELOPMENT_DATE}"
        )
    return parsed


def _single_day_text(frame: pl.DataFrame, *, source: str) -> str:
    _require(frame, {"Date"}, source)
    dates = frame.select(pl.col("Date").cast(pl.String)).unique()["Date"].to_list()
    if len(dates) != 1:
        raise ValueError(f"{source} must contain exactly one Date, got {dates[:5]}")
    date_text = str(dates[0])
    _parse_date(date_text, source=f"{source}.Date")
    return date_text


def _normalise_grid(
    frame: pl.DataFrame,
    *,
    source: str,
    required: set[str],
) -> pl.DataFrame:
    if frame.is_empty():
        raise ValueError(f"{source} must not be empty")
    _require(frame, required | set(GROUP_KEYS) | {"seconds_from_open"}, source)
    date_text = _single_day_text(frame, source=source)
    normalised = frame.with_columns(
        pl.lit(date_text).alias("Date"),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("seconds_from_open").cast(pl.Int32),
    ).sort([*GROUP_KEYS, "seconds_from_open"])
    duplicate = (
        normalised.group_by([*GROUP_KEYS, "seconds_from_open"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError(
            f"{source} has duplicate product-second rows: "
            f"{duplicate.head(5).to_dicts()}"
        )
    invalid_second = normalised.filter(
        (pl.col("seconds_from_open") < 0)
        | (pl.col("seconds_from_open") >= SESSION_END_SECOND)
    )
    if invalid_second.height:
        raise ValueError(f"{source} contains seconds outside the regular session")
    gap = (
        normalised.with_columns(
            pl.col("seconds_from_open")
            .diff()
            .over(GROUP_KEYS)
            .alias("_second_diff")
        )
        .filter(
            pl.col("_second_diff").is_not_null()
            & (pl.col("_second_diff") != 1)
        )
    )
    if gap.height:
        raise ValueError(
            f"{source} is not a complete one-second grid within a product: "
            f"{gap.select(*GROUP_KEYS, 'seconds_from_open', '_second_diff').head(5).to_dicts()}"
        )
    return normalised


def summarize_prior_day(
    day: pl.DataFrame,
    target_date: str,
    prior_date: str,
) -> pl.DataFrame:
    """Summarize an exact-QuoteCode prior common session.

    ``day`` must already contain the exact target contracts on ``prior_date``;
    it must not silently substitute that prior session's nearest contract.
    Missing target pairs therefore remain absent and naturally fall back to the
    current-only EWMA when joined by :func:`evaluate_anchor_day`.
    """

    target = _parse_date(target_date, source="target_date")
    prior = _parse_date(prior_date, source="prior_date")
    if prior >= target:
        raise ValueError("prior_date must be earlier than target_date")
    frame = _normalise_grid(
        day,
        source="prior day",
        required={"basis_mid_bp", "eligible_base"},
    ).with_columns(
        pl.col("basis_mid_bp").cast(pl.Float64),
        pl.col("eligible_base").fill_null(False).cast(pl.Boolean),
    )
    if _single_day_text(frame, source="prior day") != prior_date:
        raise ValueError("prior day Date does not match prior_date")
    sample = frame.filter(pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
    eligible = (
        pl.col("eligible_base")
        & pl.col("basis_mid_bp").is_not_null()
        & pl.col("basis_mid_bp").is_finite()
    )
    result = (
        sample.group_by(GROUP_KEYS)
        .agg(
            pl.len().alias("prior_grid_rows"),
            eligible.sum().alias("prior_eligible_rows"),
            pl.col("basis_mid_bp")
            .filter(eligible)
            .mean()
            .alias("prior_mean_bp"),
        )
        .with_columns(
            pl.lit(target_date).alias("Date"),
            pl.lit(prior_date).alias("prior_date"),
            pl.lit((target - prior).days).cast(pl.Int16).alias("calendar_gap_days"),
            (
                pl.col("prior_eligible_rows") / pl.col("prior_grid_rows")
            ).alias("prior_coverage"),
        )
        .with_columns(
            (
                (pl.col("prior_coverage") >= PRIOR_MIN_COVERAGE)
                & pl.col("prior_mean_bp").is_not_null()
                & pl.col("prior_mean_bp").is_finite()
            )
            .fill_null(False)
            .alias("prior_valid")
        )
        .select(
            "Date",
            "prior_date",
            "calendar_gap_days",
            "ValueCode",
            "QuoteCode",
            "prior_grid_rows",
            "prior_eligible_rows",
            "prior_coverage",
            "prior_mean_bp",
            "prior_valid",
        )
        .sort(["Date", "ValueCode", "QuoteCode"])
    )
    return result


def _prepare_current_day(day: pl.DataFrame) -> pl.DataFrame:
    required = {
        "end_date",
        "basis_mid_bp",
        "eligible_base",
        "eligible_1000ms",
        "eligible_100ms",
        "leg_skew_ms",
    }
    frame = _normalise_grid(day, source="anchor day", required=required)
    frame = frame.with_columns(
        pl.col("end_date").cast(pl.Date),
        pl.col("basis_mid_bp").cast(pl.Float64),
        pl.col("eligible_base").fill_null(False).cast(pl.Boolean),
        pl.col("eligible_1000ms").fill_null(False).cast(pl.Boolean),
        pl.col("eligible_100ms").fill_null(False).cast(pl.Boolean),
        pl.col("leg_skew_ms").cast(pl.Float64),
    )
    inconsistent_expiry = (
        frame.group_by(GROUP_KEYS)
        .agg(
            pl.col("end_date")
            .drop_nulls()
            .n_unique()
            .alias("expiry_values")
        )
        .filter(pl.col("expiry_values") != 1)
    )
    if inconsistent_expiry.height:
        raise ValueError("anchor day has inconsistent end_date within a contract")
    frame = frame.with_columns(
        pl.col("end_date")
        .drop_nulls()
        .first()
        .over(GROUP_KEYS)
        .alias("end_date")
    )
    trade_date = _parse_date(
        _single_day_text(frame, source="anchor day"), source="anchor day Date"
    ).date()
    expired = frame.filter(pl.col("end_date") < pl.lit(trade_date))
    if expired.height:
        raise ValueError("anchor day contains an already-expired contract")
    return frame.select(
        *GROUP_KEYS,
        "seconds_from_open",
        "end_date",
        "basis_mid_bp",
        "eligible_base",
        "eligible_1000ms",
        "eligible_100ms",
        "leg_skew_ms",
    )


def _join_priors(
    frame: pl.DataFrame,
    priors: pl.DataFrame | None,
) -> pl.DataFrame:
    if priors is None or priors.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.String).alias("prior_date"),
            pl.lit(None, dtype=pl.Int16).alias("calendar_gap_days"),
            pl.lit(0, dtype=pl.UInt32).alias("prior_grid_rows"),
            pl.lit(0, dtype=pl.UInt32).alias("prior_eligible_rows"),
            pl.lit(None, dtype=pl.Float64).alias("prior_coverage"),
            pl.lit(None, dtype=pl.Float64).alias("prior_mean_bp"),
            pl.lit(False).alias("prior_valid"),
        )
    required = {
        *GROUP_KEYS,
        "prior_date",
        "calendar_gap_days",
        "prior_grid_rows",
        "prior_eligible_rows",
        "prior_coverage",
        "prior_mean_bp",
        "prior_valid",
    }
    _require(priors, required, "prior summary")
    current_date = _single_day_text(frame, source="anchor day")
    selected = priors.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    ).filter(pl.col("Date") == current_date)
    duplicate = selected.group_by(GROUP_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("prior summary has duplicate exact-contract rows")
    joined = frame.join(selected, on=GROUP_KEYS, how="left", validate="m:1")
    return joined.with_columns(
        pl.col("prior_valid").fill_null(False),
        pl.col("prior_grid_rows").fill_null(0),
        pl.col("prior_eligible_rows").fill_null(0),
    )


def _add_seeded_ewma(frame: pl.DataFrame) -> pl.DataFrame:
    seed = (
        frame.group_by(GROUP_KEYS)
        .agg(
            pl.when(pl.col("prior_valid"))
            .then(pl.col("prior_mean_bp"))
            .otherwise(None)
            .drop_nulls()
            .first()
            .alias("_ewma_input_bp")
        )
        .with_columns(
            pl.lit(-1, dtype=pl.Int32).alias("seconds_from_open"),
            pl.lit(True).alias("_is_seed"),
        )
    )
    observations = frame.select(
        *GROUP_KEYS,
        "seconds_from_open",
        pl.col("basis_eval_bp").alias("_ewma_input_bp"),
        pl.lit(False).alias("_is_seed"),
    )
    seeded = (
        pl.concat([seed, observations], how="diagonal_relaxed")
        .sort([*GROUP_KEYS, "seconds_from_open"])
        .with_columns(
            pl.col("_ewma_input_bp")
            .ewm_mean(half_life=120, adjust=False, ignore_nulls=True)
            .over(GROUP_KEYS)
            .alias("_seeded_ewma_120s_bp")
        )
        .filter(~pl.col("_is_seed"))
        .select(*GROUP_KEYS, "seconds_from_open", "_seeded_ewma_120s_bp")
    )
    return (
        frame.join(
            seeded,
            on=[*GROUP_KEYS, "seconds_from_open"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("prior_valid"))
            .then(pl.col("_seeded_ewma_120s_bp"))
            .otherwise(pl.col(BASELINE_COLUMN))
            .alias("anchor_prior_seeded_ewma_120s_bp")
        )
        .drop("_seeded_ewma_120s_bp")
    )


def build_anchor_candidates(
    day: pl.DataFrame,
    priors: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build all frozen anchor candidates for one complete session."""

    frame = _prepare_current_day(day)
    base_gate = (
        pl.col("eligible_base")
        & pl.col("basis_mid_bp").is_not_null()
        & pl.col("basis_mid_bp").is_finite()
    ).fill_null(False)
    frame = frame.with_columns(
        pl.when(base_gate)
        .then(pl.col("basis_mid_bp"))
        .otherwise(None)
        .alias("basis_eval_bp")
    )
    opening_median = (
        pl.when(pl.col("seconds_from_open") < ANALYSIS_START_SECOND)
        .then(pl.col("basis_eval_bp"))
        .otherwise(None)
        .median()
        .over(GROUP_KEYS)
    )
    frame = frame.with_columns(
        pl.col("basis_eval_bp").alias("anchor_persistence_bp"),
        pl.when(pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
        .then(opening_median)
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
        .rolling_median(SESSION_END_SECOND, min_samples=300)
        .over(GROUP_KEYS)
        .alias("anchor_expanding_median_bp"),
    )
    return _add_seeded_ewma(_join_priors(frame, priors))


def _freshness_gate(sample: str) -> pl.Expr:
    valid_basis = (
        pl.col("basis_mid_bp").is_not_null()
        & pl.col("basis_mid_bp").is_finite()
    )
    if sample == "base":
        gate = pl.col("eligible_base")
    elif sample == "age_1000ms":
        gate = pl.col("eligible_1000ms")
    elif sample == "age_1000ms_skew_100ms":
        gate = pl.col("eligible_1000ms") & (pl.col("leg_skew_ms") <= 100)
    elif sample == "age_100ms":
        gate = pl.col("eligible_100ms")
    else:
        raise ValueError(f"unknown freshness sample: {sample}")
    return (gate & valid_basis).fill_null(False)


def add_freshness_labels(
    candidates: pl.DataFrame,
    freshness_sample: str,
) -> pl.DataFrame:
    """Add same-gate future-center and exact delayed endpoint labels."""

    if freshness_sample not in FRESHNESS_SAMPLES:
        raise ValueError(f"unknown freshness sample: {freshness_sample}")
    gate = _freshness_gate(freshness_sample)
    frame = candidates.with_columns(
        gate.alias("current_fresh_gate"),
        pl.when(gate)
        .then(pl.col("basis_mid_bp"))
        .otherwise(None)
        .alias("_fresh_basis_bp"),
        pl.lit(freshness_sample).alias("freshness_sample"),
    )
    frame = frame.with_columns(
        pl.col("_fresh_basis_bp")
        .rolling_median(
            FUTURE_CENTER_WINDOW_ROWS,
            min_samples=FUTURE_CENTER_MIN_ROWS,
        )
        .shift(-FUTURE_CENTER_END_SECOND)
        .over(GROUP_KEYS)
        .alias("future_center_bp"),
        (
            pl.col("current_fresh_gate")
            .cast(pl.UInt16)
            .rolling_sum(
                FUTURE_CENTER_WINDOW_ROWS,
                min_samples=FUTURE_CENTER_WINDOW_ROWS,
            )
            .shift(-FUTURE_CENTER_END_SECOND)
            .over(GROUP_KEYS)
            / FUTURE_CENTER_WINDOW_ROWS
        ).alias("future_center_coverage"),
        pl.col("_fresh_basis_bp")
        .shift(-FUTURE_CENTER_START_SECOND)
        .over(GROUP_KEYS)
        .alias("basis_t30_bp"),
        pl.col("_fresh_basis_bp")
        .shift(-FUTURE_CENTER_END_SECOND)
        .over(GROUP_KEYS)
        .alias("basis_t300_bp"),
        pl.col("current_fresh_gate")
        .shift(-FUTURE_CENTER_START_SECOND)
        .over(GROUP_KEYS)
        .fill_null(False)
        .alias("_gate_t30"),
        pl.col("current_fresh_gate")
        .shift(-FUTURE_CENTER_END_SECOND)
        .over(GROUP_KEYS)
        .fill_null(False)
        .alias("_gate_t300"),
        pl.col("seconds_from_open")
        .shift(-FUTURE_CENTER_START_SECOND)
        .over(GROUP_KEYS)
        .alias("_second_t30"),
        pl.col("seconds_from_open")
        .shift(-FUTURE_CENTER_END_SECOND)
        .over(GROUP_KEYS)
        .alias("_second_t300"),
    )
    return frame.with_columns(
        (
            pl.col("_gate_t30")
            & (
                pl.col("_second_t30")
                == pl.col("seconds_from_open") + FUTURE_CENTER_START_SECOND
            )
            & pl.col("basis_t30_bp").is_not_null()
            & pl.col("basis_t30_bp").is_finite()
        )
        .fill_null(False)
        .alias("t30_gate_ok"),
        (
            pl.col("_gate_t300")
            & (
                pl.col("_second_t300")
                == pl.col("seconds_from_open") + FUTURE_CENTER_END_SECOND
            )
            & pl.col("basis_t300_bp").is_not_null()
            & pl.col("basis_t300_bp").is_finite()
        )
        .fill_null(False)
        .alias("t300_gate_ok"),
    )


def _expected_stage(date_text: str) -> str:
    if date_text <= "20260504":
        return "history"
    if date_text <= "20260531":
        return "fine_tune"
    if date_text <= "20260630":
        return "confirmation"
    return "pseudo_holdout"


def _tod_bucket() -> pl.Expr:
    seconds = pl.col("seconds_from_open")
    return (
        pl.when(seconds < 900)
        .then(pl.lit("09:05-09:15"))
        .when(seconds < 3_600)
        .then(pl.lit("09:15-10:00"))
        .when(seconds < 10_800)
        .then(pl.lit("10:00-12:00"))
        .when(seconds < 14_400)
        .then(pl.lit("12:00-13:00"))
        .when(seconds < 15_300)
        .then(pl.lit("13:00-13:15"))
        .otherwise(pl.lit("13:15-13:20_horizon_censored"))
    )


def _dte_bucket() -> pl.Expr:
    dte = pl.col("calendar_dte")
    return (
        pl.when(dte == 0)
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


def _add_strata(frame: pl.DataFrame, stage: str) -> pl.DataFrame:
    date_text = _single_day_text(frame, source="anchor day")
    expected = _expected_stage(date_text)
    if stage != expected:
        raise ValueError(
            f"stage {stage!r} does not match fixed stage {expected!r} for {date_text}"
        )
    trade_date = _parse_date(date_text, source="anchor day Date").date()
    result = frame.with_columns(
        (pl.col("end_date") - pl.lit(trade_date))
        .dt.total_days()
        .cast(pl.Int16)
        .alias("calendar_dte"),
        _tod_bucket().alias("tod_bucket"),
        pl.lit(stage).alias("stage"),
        pl.lit(date_text[:6]).alias("month"),
    )
    return result.with_columns(_dte_bucket().alias("dte_bucket"))


def _with_stratum(frame: pl.DataFrame, family: str) -> pl.DataFrame:
    if family == "overall":
        value = pl.lit("all")
    elif family == "tod":
        value = pl.col("tod_bucket")
    elif family == "dte":
        value = pl.col("dte_bucket")
    else:
        raise ValueError(f"unknown stratum family: {family}")
    return frame.with_columns(
        pl.lit(family).alias("stratum_family"),
        value.alias("stratum_value"),
    )


def _metadata_aggregations() -> list[pl.Expr]:
    return [
        pl.col("stage").first().alias("stage"),
        pl.col("month").first().alias("month"),
        pl.col("calendar_dte").first().alias("calendar_dte"),
    ]


def _finite(column: str) -> pl.Expr:
    return pl.col(column).is_not_null() & pl.col(column).is_finite()


def _model_evaluation_frame(
    labels: pl.DataFrame,
    *,
    model: str,
    anchor_column: str,
) -> pl.DataFrame:
    previous_gate = pl.col("current_fresh_gate").shift(1).over(GROUP_KEYS)
    previous_second = pl.col("seconds_from_open").shift(1).over(GROUP_KEYS)
    previous_anchor = pl.col(anchor_column).shift(1).over(GROUP_KEYS)
    previous_basis = pl.col("basis_mid_bp").shift(1).over(GROUP_KEYS)
    frame = labels.with_columns(
        pl.lit(model).alias("model"),
        pl.col(anchor_column).alias("anchor_bp"),
        previous_gate.fill_null(False).alias("_previous_gate"),
        previous_second.alias("_previous_second"),
        previous_anchor.alias("_previous_anchor_bp"),
        previous_basis.alias("_previous_basis_bp"),
    )
    current_ok = pl.col("current_fresh_gate")
    anchor_ok = _finite("anchor_bp")
    horizon_ok = pl.col("seconds_from_open") <= (
        SESSION_END_SECOND - 1 - FUTURE_CENTER_END_SECOND
    )
    label_ok = (
        _finite("future_center_bp")
        & (pl.col("future_center_coverage") >= FUTURE_CENTER_MIN_COVERAGE)
    ).fill_null(False)
    evaluable = current_ok & anchor_ok & horizon_ok & label_ok
    baseline_ok = _finite(BASELINE_COLUMN)
    pairwise = evaluable & baseline_ok
    adjacent = (
        current_ok
        & pl.col("_previous_gate")
        & anchor_ok
        & _finite("_previous_anchor_bp")
        & _finite("basis_mid_bp")
        & _finite("_previous_basis_bp")
        & (pl.col("_previous_second") >= ANALYSIS_START_SECOND)
        & (pl.col("seconds_from_open") - pl.col("_previous_second") == 1)
    ).fill_null(False)
    return (
        frame.with_columns(
            (pl.col("anchor_bp") - pl.col("future_center_bp")).alias("_error_bp"),
            (
                pl.col(BASELINE_COLUMN) - pl.col("future_center_bp")
            ).alias("_baseline_error_bp"),
            evaluable.fill_null(False).alias("_evaluable"),
            pairwise.fill_null(False).alias("_pairwise_common"),
            adjacent.alias("_adjacent_legal"),
            (~current_ok).fill_null(True).alias("_censor_current_gate"),
            (current_ok & ~anchor_ok)
            .fill_null(False)
            .alias("_censor_anchor"),
            (current_ok & anchor_ok & ~horizon_ok)
            .fill_null(False)
            .alias("_censor_horizon"),
            (current_ok & anchor_ok & horizon_ok & ~label_ok)
            .fill_null(False)
            .alias("_censor_future_coverage"),
        )
        .filter(pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
    )


def _summarize_model(
    labels: pl.DataFrame,
    *,
    model: str,
    anchor_column: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frame = _model_evaluation_frame(
        labels,
        model=model,
        anchor_column=anchor_column,
    )
    coverage_parts: list[pl.DataFrame] = []
    anchor_parts: list[pl.DataFrame] = []
    for family in STRATUM_FAMILIES:
        stratum = _with_stratum(frame, family)
        coverage_parts.append(
            stratum.group_by(SUMMARY_KEYS).agg(
                *_metadata_aggregations(),
                pl.len().alias("n_grid"),
                pl.col("current_fresh_gate").sum().alias("n_current_eligible"),
                (
                    pl.col("current_fresh_gate") & _finite("anchor_bp")
                ).sum().alias("n_anchor_available"),
                (
                    pl.col("current_fresh_gate")
                    & _finite("anchor_bp")
                    & (
                        pl.col("seconds_from_open")
                        <= SESSION_END_SECOND - 1 - FUTURE_CENTER_END_SECOND
                    )
                ).sum().alias("n_horizon_possible"),
                pl.col("_evaluable").sum().alias("n_future_center_supported"),
                pl.col("_censor_current_gate")
                .sum()
                .alias("censor_current_gate_closed"),
                pl.col("_censor_anchor").sum().alias("censor_anchor_unavailable"),
                pl.col("_censor_horizon")
                .sum()
                .alias("censor_horizon_after_session"),
                pl.col("_censor_future_coverage")
                .sum()
                .alias("censor_future_window_coverage_lt_90pct"),
                pl.col("_evaluable").sum().alias("evaluable"),
                pl.col("prior_valid").first().alias("prior_valid"),
                pl.col("prior_coverage")
                .drop_nulls()
                .first()
                .alias("prior_coverage"),
            )
        )
        anchor_parts.append(
            stratum.group_by(SUMMARY_KEYS).agg(
                *_metadata_aggregations(),
                pl.col("_evaluable").sum().alias("n_evaluable"),
                pl.col("_error_bp")
                .filter(pl.col("_evaluable"))
                .sum()
                .alias("sum_error_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_evaluable"))
                .sum()
                .alias("sum_abs_error_bp"),
                pl.col("_error_bp")
                .filter(pl.col("_evaluable"))
                .mean()
                .alias("bias_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_evaluable"))
                .mean()
                .alias("mae_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_evaluable"))
                .quantile(0.50, interpolation="nearest")
                .alias("p50_abs_error_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_evaluable"))
                .quantile(0.80, interpolation="nearest")
                .alias("p80_abs_error_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_evaluable"))
                .quantile(0.95, interpolation="nearest")
                .alias("p95_abs_error_bp"),
                pl.col("_pairwise_common").sum().alias("n_pairwise_common"),
                pl.col("_error_bp")
                .filter(pl.col("_pairwise_common"))
                .sum()
                .alias("pairwise_model_sum_error_bp"),
                pl.col("_error_bp")
                .abs()
                .filter(pl.col("_pairwise_common"))
                .sum()
                .alias("pairwise_model_sum_abs_error_bp"),
                pl.col("_baseline_error_bp")
                .filter(pl.col("_pairwise_common"))
                .sum()
                .alias("pairwise_baseline_sum_error_bp"),
                pl.col("_baseline_error_bp")
                .abs()
                .filter(pl.col("_pairwise_common"))
                .sum()
                .alias("pairwise_baseline_sum_abs_error_bp"),
                pl.col("_adjacent_legal")
                .sum()
                .alias("n_adjacent_legal_pairs"),
                (pl.col("anchor_bp") - pl.col("_previous_anchor_bp"))
                .abs()
                .filter(pl.col("_adjacent_legal"))
                .sum()
                .alias("anchor_tv_bp"),
                (pl.col("basis_mid_bp") - pl.col("_previous_basis_bp"))
                .abs()
                .filter(pl.col("_adjacent_legal"))
                .sum()
                .alias("basis_tv_bp"),
            )
        )
    coverage = pl.concat(coverage_parts, how="vertical_relaxed").sort(SUMMARY_KEYS)
    censor_total = (
        pl.col("censor_current_gate_closed")
        + pl.col("censor_anchor_unavailable")
        + pl.col("censor_horizon_after_session")
        + pl.col("censor_future_window_coverage_lt_90pct")
        + pl.col("evaluable")
    )
    if coverage.filter(censor_total != pl.col("n_grid")).height:
        raise AssertionError("future-center censor counts are not exhaustive")

    anchor = (
        pl.concat(anchor_parts, how="vertical_relaxed")
        .with_columns(
            pl.when(pl.col("n_pairwise_common") > 0)
            .then(
                pl.col("pairwise_model_sum_abs_error_bp")
                / pl.col("n_pairwise_common")
                - pl.col("pairwise_baseline_sum_abs_error_bp")
                / pl.col("n_pairwise_common")
            )
            .otherwise(None)
            .alias("pairwise_delta_mae_vs_ewma120_bp"),
            pl.when(pl.col("basis_tv_bp") > CHANGE_EPS_BP)
            .then(pl.col("anchor_tv_bp") / pl.col("basis_tv_bp"))
            .otherwise(None)
            .alias("tv_ratio"),
        )
        .sort(SUMMARY_KEYS)
    )
    delayed = _summarize_delayed_model(frame)
    return anchor, delayed, coverage


def select_nonoverlap_rows(
    rows: pl.DataFrame,
    *,
    lockout_seconds: int = NONOVERLAP_LOCKOUT_SECONDS,
) -> pl.DataFrame:
    """Greedily retain product signals separated by at least ``lockout_seconds``."""

    if lockout_seconds <= 0:
        raise ValueError("lockout_seconds must be positive")
    required = {*GROUP_KEYS, "seconds_from_open"}
    _require(rows, required, "nonoverlap rows")
    if rows.is_empty():
        return rows
    identity = [
        column
        for column in (
            *GROUP_KEYS,
            "model",
            "freshness_sample",
        )
        if column in rows.columns
    ]
    ordered = rows.sort([*identity, "seconds_from_open"])
    selected: list[pl.DataFrame] = []
    for _, group in ordered.group_by(identity, maintain_order=True):
        seconds = group["seconds_from_open"].to_list()
        positions: list[int] = []
        cursor = 0
        while cursor < len(seconds):
            positions.append(cursor)
            cursor = bisect_left(
                seconds,
                int(seconds[cursor]) + lockout_seconds,
                lo=cursor + 1,
            )
        mask = [False] * group.height
        for position in positions:
            mask[position] = True
        selected.append(group.filter(pl.Series("_keep", mask)))
    return pl.concat(selected, how="vertical_relaxed")


def _summarize_delayed_model(frame: pl.DataFrame) -> pl.DataFrame:
    positive_current = (
        pl.col("current_fresh_gate")
        & _finite("anchor_bp")
        & _finite("basis_mid_bp")
        & ((pl.col("basis_mid_bp") - pl.col("anchor_bp")) > CHANGE_EPS_BP)
    ).fill_null(False)
    horizon_ok = pl.col("seconds_from_open") <= (
        SESSION_END_SECOND - 1 - FUTURE_CENTER_END_SECOND
    )
    t30_ok = pl.col("t30_gate_ok")
    t300_ok = pl.col("t300_gate_ok")
    occupancy = positive_current & horizon_ok & t30_ok & t300_ok
    delayed = frame.with_columns(
        positive_current.alias("_positive_current"),
        (positive_current & ~horizon_ok)
        .fill_null(False)
        .alias("_delayed_censor_horizon"),
        (positive_current & horizon_ok & ~t30_ok)
        .fill_null(False)
        .alias("_delayed_censor_t30"),
        (positive_current & horizon_ok & t30_ok & ~t300_ok)
        .fill_null(False)
        .alias("_delayed_censor_t300"),
        occupancy.fill_null(False).alias("_positive_occupancy"),
        (pl.col("basis_t30_bp") - pl.col("basis_t300_bp")).alias(
            "signal_consistent_move_bp"
        ),
    )
    occupancy_rows = delayed.filter(pl.col("_positive_occupancy")).select(
        *GROUP_KEYS,
        "model",
        "freshness_sample",
        "stage",
        "month",
        "calendar_dte",
        "tod_bucket",
        "dte_bucket",
        "seconds_from_open",
        "signal_consistent_move_bp",
    )
    nonoverlap = select_nonoverlap_rows(occupancy_rows)
    result_parts: list[pl.DataFrame] = []
    for family in STRATUM_FAMILIES:
        base = _with_stratum(delayed, family).group_by(SUMMARY_KEYS).agg(
            *_metadata_aggregations(),
            pl.col("_positive_current").sum().alias("n_positive_current"),
            pl.col("_delayed_censor_horizon")
            .sum()
            .alias("delayed_censor_horizon_after_session"),
            pl.col("_delayed_censor_t30")
            .sum()
            .alias("delayed_censor_t30_gate_closed"),
            pl.col("_delayed_censor_t300")
            .sum()
            .alias("delayed_censor_t300_gate_closed"),
            pl.col("_positive_occupancy").sum().alias("n_positive_occupancy"),
            pl.col("signal_consistent_move_bp")
            .filter(pl.col("_positive_occupancy"))
            .sum()
            .alias("occupancy_sum_signal_consistent_move_bp"),
            pl.col("signal_consistent_move_bp")
            .filter(pl.col("_positive_occupancy"))
            .mean()
            .alias("occupancy_mean_signal_consistent_move_bp"),
            pl.col("signal_consistent_move_bp")
            .filter(pl.col("_positive_occupancy"))
            .median()
            .alias("occupancy_median_signal_consistent_move_bp"),
            (pl.col("signal_consistent_move_bp") > CHANGE_EPS_BP)
            .filter(pl.col("_positive_occupancy"))
            .sum()
            .alias("occupancy_signal_consistent_count"),
            (pl.col("signal_consistent_move_bp").abs() <= CHANGE_EPS_BP)
            .filter(pl.col("_positive_occupancy"))
            .sum()
            .alias("occupancy_flat_count"),
        )
        if nonoverlap.is_empty():
            selected = base.select(*SUMMARY_KEYS).with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("n_positive_nonoverlap"),
                pl.lit(0.0).alias("nonoverlap_sum_signal_consistent_move_bp"),
                pl.lit(None, dtype=pl.Float64).alias(
                    "nonoverlap_mean_signal_consistent_move_bp"
                ),
                pl.lit(None, dtype=pl.Float64).alias(
                    "nonoverlap_median_signal_consistent_move_bp"
                ),
                pl.lit(0, dtype=pl.UInt32).alias(
                    "nonoverlap_signal_consistent_count"
                ),
                pl.lit(0, dtype=pl.UInt32).alias("nonoverlap_flat_count"),
            )
        else:
            selected = _with_stratum(nonoverlap, family).group_by(
                SUMMARY_KEYS
            ).agg(
                pl.len().alias("n_positive_nonoverlap"),
                pl.col("signal_consistent_move_bp")
                .sum()
                .alias("nonoverlap_sum_signal_consistent_move_bp"),
                pl.col("signal_consistent_move_bp")
                .mean()
                .alias("nonoverlap_mean_signal_consistent_move_bp"),
                pl.col("signal_consistent_move_bp")
                .median()
                .alias("nonoverlap_median_signal_consistent_move_bp"),
                (pl.col("signal_consistent_move_bp") > CHANGE_EPS_BP)
                .sum()
                .alias("nonoverlap_signal_consistent_count"),
                (pl.col("signal_consistent_move_bp").abs() <= CHANGE_EPS_BP)
                .sum()
                .alias("nonoverlap_flat_count"),
            )
        result_parts.append(
            base.join(selected, on=SUMMARY_KEYS, how="left", validate="1:1")
        )
    result = pl.concat(result_parts, how="vertical_relaxed")
    count_columns = (
        "n_positive_nonoverlap",
        "nonoverlap_signal_consistent_count",
        "nonoverlap_flat_count",
    )
    result = result.with_columns(
        *[pl.col(column).fill_null(0) for column in count_columns],
        pl.col("nonoverlap_sum_signal_consistent_move_bp").fill_null(0.0),
        pl.when(pl.col("n_positive_occupancy") > 0)
        .then(
            pl.col("occupancy_signal_consistent_count")
            / pl.col("n_positive_occupancy")
        )
        .otherwise(None)
        .alias("occupancy_p_signal_consistent"),
        pl.when(pl.col("n_positive_nonoverlap") > 0)
        .then(
            pl.col("nonoverlap_signal_consistent_count")
            / pl.col("n_positive_nonoverlap")
        )
        .otherwise(None)
        .alias("nonoverlap_p_signal_consistent"),
    )
    delayed_total = (
        pl.col("delayed_censor_horizon_after_session")
        + pl.col("delayed_censor_t30_gate_closed")
        + pl.col("delayed_censor_t300_gate_closed")
        + pl.col("n_positive_occupancy")
    )
    if result.filter(delayed_total != pl.col("n_positive_current")).height:
        raise AssertionError("delayed positive censor counts are not exhaustive")
    return result.sort(SUMMARY_KEYS)


def evaluate_anchor_day(
    day: pl.DataFrame,
    priors: pl.DataFrame | None,
    stage: str,
) -> AnchorDayEvaluation:
    """Evaluate one day without concatenating it with any other session."""

    candidates = _add_strata(build_anchor_candidates(day, priors), stage)
    anchor_frames: list[pl.DataFrame] = []
    delayed_frames: list[pl.DataFrame] = []
    coverage_frames: list[pl.DataFrame] = []
    for freshness_sample in FRESHNESS_SAMPLES:
        labels = add_freshness_labels(candidates, freshness_sample)
        model_names = (
            tuple(MODEL_COLUMNS)
            if freshness_sample == "base"
            else STRICT_FRESHNESS_MODELS
        )
        for model in model_names:
            anchor_column = MODEL_COLUMNS[model]
            anchor, delayed, coverage = _summarize_model(
                labels,
                model=model,
                anchor_column=anchor_column,
            )
            anchor_frames.append(anchor)
            delayed_frames.append(delayed)
            coverage_frames.append(coverage)
    return AnchorDayEvaluation(
        anchor_daily=pl.concat(anchor_frames, how="vertical_relaxed").sort(
            SUMMARY_KEYS
        ),
        delayed_daily=pl.concat(delayed_frames, how="vertical_relaxed").sort(
            SUMMARY_KEYS
        ),
        coverage=pl.concat(coverage_frames, how="vertical_relaxed").sort(
            SUMMARY_KEYS
        ),
    )
