"""Pure statistics contracts for causal foundation-model selection.

The functions in this module operate on already materialized, development-only
tables.  They do not read files, discover sessions, or mutate external state.
Dates are the resampling unit: product rows must first be reduced to one value
per candidate and Date before calling the paired bootstrap or anchor selector.

The anchor selector expects the following long-form daily schema by default::

    Date: str, month: str, model: str, mae_bp: float, tv_ratio: float

All actionable models are compared on their common finite-Date support.  MAE
and smoothness are first averaged within month and then equally across months.
The raw MAE winner is retained, plus the smoothest actionable model whose
paired loss difference is no larger than one bootstrap standard error.  A
configured control is never selectable, even if it is accidentally included
in ``actionable_models``.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Literal

import polars as pl

DEFAULT_BLOCK_SESSIONS = 5
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_825
FLOAT_EPSILON = 1e-12

type SortDirection = Literal["min", "max"]

Q_LEXICOGRAPHIC_CRITERIA: tuple[tuple[str, SortDirection], ...] = (
    ("primary_loss", "min"),
    ("worst_month_loss", "min"),
    ("amplitude_absolute_error_bp", "min"),
    ("native_all_q_coverage", "max"),
    ("daily_cross_product_spearman", "max"),
    ("boundary_turnover", "min"),
)


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str] | set[str],
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _normalise_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in values))
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _ordered_quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def moving_whole_date_block_indices(
    session_count: int,
    *,
    block_sessions: int = DEFAULT_BLOCK_SESSIONS,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic moving-block samples over ordered sessions.

    A block is a contiguous run in the supplied *session sequence*, not in
    calendar-day arithmetic.  Blocks never wrap from the final session back to
    the first.  Each replicate contains ``session_count`` indices; the final
    sampled block is truncated if necessary.  The same returned indices must
    be applied to every model to preserve pairing.
    """

    if session_count <= 0:
        raise ValueError("session_count must be positive")
    if block_sessions <= 0:
        raise ValueError("block_sessions must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    effective_block = min(block_sessions, session_count)
    start_count = session_count - effective_block + 1
    rng = random.Random(seed)
    samples: list[tuple[int, ...]] = []
    for _ in range(replicates):
        selected: list[int] = []
        while len(selected) < session_count:
            start = rng.randrange(start_count)
            selected.extend(range(start, start + effective_block))
        samples.append(tuple(selected[:session_count]))
    return tuple(samples)


def _validate_unique_candidate_dates(
    frame: pl.DataFrame,
    *,
    date_column: str,
    candidate_column: str,
    source: str,
) -> None:
    duplicate = (
        frame.group_by([date_column, candidate_column]).len().filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError(
            f"{source} must have one row per candidate-Date: "
            f"{duplicate.head(5).to_dicts()}"
        )


def _finite_common_daily_panel(
    daily: pl.DataFrame,
    *,
    candidates: Sequence[str],
    date_column: str,
    candidate_column: str,
    metric_columns: Sequence[str],
    extra_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Return finite rows on the exact common Date support of candidates."""

    candidate_ids = _normalise_ids(candidates, label="candidates")
    required = {
        date_column,
        candidate_column,
        *metric_columns,
        *extra_columns,
    }
    _require_columns(daily, required, "daily candidate panel")
    panel = daily.filter(
        pl.col(candidate_column).cast(pl.String).is_in(candidate_ids)
    ).with_columns(
        pl.col(date_column).cast(pl.String),
        pl.col(candidate_column).cast(pl.String),
        *[pl.col(column).cast(pl.Float64) for column in metric_columns],
    )
    _validate_unique_candidate_dates(
        panel,
        date_column=date_column,
        candidate_column=candidate_column,
        source="daily candidate panel",
    )

    missing_candidates = sorted(
        set(candidate_ids) - set(panel[candidate_column].unique().to_list())
    )
    if missing_candidates:
        raise ValueError(
            f"daily candidate panel missing candidates: {missing_candidates}"
        )

    finite = panel
    for column in metric_columns:
        finite = finite.filter(
            pl.col(column).is_not_null() & pl.col(column).is_finite()
        )
    support = (
        finite.group_by(date_column)
        .agg(pl.col(candidate_column).n_unique().alias("_candidate_count"))
        .filter(pl.col("_candidate_count") == len(candidate_ids))
        .select(date_column)
    )
    common = finite.join(support, on=date_column, how="inner")
    if common.is_empty():
        raise ValueError("candidates have no common finite Date support")

    expected_rows = common[date_column].n_unique() * len(candidate_ids)
    if common.height != expected_rows:
        raise ValueError("common Date panel is not a complete candidate rectangle")
    return common.sort([date_column, candidate_column])


def paired_moving_whole_date_block_bootstrap(
    daily: pl.DataFrame,
    *,
    reference: str,
    candidates: Sequence[str] | None = None,
    date_column: str = "Date",
    candidate_column: str = "model",
    metric_column: str = "mae_bp",
    weight_column: str | None = None,
    strata_column: str | None = None,
    block_sessions: int = DEFAULT_BLOCK_SESSIONS,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pl.DataFrame:
    """Bootstrap paired candidate-minus-reference Date-level loss differences.

    Input is one finite loss per candidate-Date.  When ``strata_column`` is
    supplied (``month`` for anchor selection), each stratum is independently
    resampled to its original session count and replicate statistics are
    equally weighted across strata.  This retains both five-session serial
    blocks and the registered month-equal objective.
    """

    if candidates is None:
        _require_columns(daily, [candidate_column], "daily candidate panel")
        candidate_ids = tuple(
            sorted(str(value) for value in daily[candidate_column].unique().to_list())
        )
    else:
        candidate_ids = _normalise_ids(candidates, label="candidates")
    reference_id = str(reference)
    if reference_id not in candidate_ids:
        raise ValueError("reference must be included in candidates")

    extra_values = [strata_column] if strata_column is not None else []
    if weight_column is not None:
        extra_values.append(weight_column)
    extra = tuple(extra_values)
    panel = _finite_common_daily_panel(
        daily,
        candidates=candidate_ids,
        date_column=date_column,
        candidate_column=candidate_column,
        metric_columns=(metric_column,),
        extra_columns=extra,
    )
    if strata_column is not None:
        inconsistent = (
            panel.group_by(date_column)
            .agg(pl.col(strata_column).n_unique().alias("_strata"))
            .filter(pl.col("_strata") != 1)
        )
        if inconsistent.height:
            raise ValueError("each Date must map to exactly one bootstrap stratum")
    if weight_column is not None:
        panel = panel.with_columns(pl.col(weight_column).cast(pl.Float64))
        invalid_weight = panel.filter(
            pl.col(weight_column).is_null()
            | ~pl.col(weight_column).is_finite()
            | (pl.col(weight_column) <= 0)
        )
        if invalid_weight.height:
            raise ValueError("bootstrap weights must be finite and positive")
        inconsistent_weight = (
            panel.group_by(date_column)
            .agg(
                (pl.col(weight_column).max() - pl.col(weight_column).min()).alias(
                    "_weight_range"
                )
            )
            .filter(pl.col("_weight_range").abs() > FLOAT_EPSILON)
        )
        if inconsistent_weight.height:
            raise ValueError("paired candidates must have the same weight per Date")

    ordered_dates = sorted(str(value) for value in panel[date_column].unique())
    date_position = {date: index for index, date in enumerate(ordered_dates)}
    value_by_model: dict[str, list[float]] = {
        candidate: [math.nan] * len(ordered_dates) for candidate in candidate_ids
    }
    weight_by_date = [1.0] * len(ordered_dates)
    stratum_by_date: list[str] | None = (
        [""] * len(ordered_dates) if strata_column is not None else None
    )
    for row in panel.iter_rows(named=True):
        date = str(row[date_column])
        index = date_position[date]
        candidate = str(row[candidate_column])
        value_by_model[candidate][index] = float(row[metric_column])
        if weight_column is not None:
            weight_by_date[index] = float(row[weight_column])
        if stratum_by_date is not None:
            stratum_by_date[index] = str(row[strata_column])

    strata_indices: dict[str, list[int]]
    if stratum_by_date is None:
        strata_indices = {"all": list(range(len(ordered_dates)))}
    else:
        strata_indices = {}
        for index, stratum in enumerate(stratum_by_date):
            strata_indices.setdefault(stratum, []).append(index)

    sampled_indices_by_replicate: list[list[list[int]]] = [
        [] for _ in range(replicates)
    ]
    for stratum_number, stratum in enumerate(sorted(strata_indices)):
        indices = strata_indices[stratum]
        stratum_seed = seed + stratum_number * 1_000_003
        samples = moving_whole_date_block_indices(
            len(indices),
            block_sessions=block_sessions,
            replicates=replicates,
            seed=stratum_seed,
        )
        for replicate, local_sample in enumerate(samples):
            sampled_indices_by_replicate[replicate].append(
                [indices[index] for index in local_sample]
            )

    reference_values = value_by_model[reference_id]
    rows: list[dict[str, object]] = []
    for candidate in candidate_ids:
        candidate_values = value_by_model[candidate]
        observed_by_stratum: list[float] = []
        for indices in strata_indices.values():
            numerator = math.fsum(
                (candidate_values[index] - reference_values[index])
                * weight_by_date[index]
                for index in indices
            )
            denominator = math.fsum(weight_by_date[index] for index in indices)
            observed_by_stratum.append(numerator / denominator)
        observed_delta = statistics.fmean(observed_by_stratum)

        bootstrap_deltas: list[float] = []
        for stratum_samples in sampled_indices_by_replicate:
            replicate_stratum_means = [
                math.fsum(
                    (candidate_values[index] - reference_values[index])
                    * weight_by_date[index]
                    for index in sampled_indices
                )
                / math.fsum(weight_by_date[index] for index in sampled_indices)
                for sampled_indices in stratum_samples
            ]
            bootstrap_deltas.append(statistics.fmean(replicate_stratum_means))
        paired_se = (
            statistics.stdev(bootstrap_deltas) if len(bootstrap_deltas) >= 2 else 0.0
        )
        rows.append(
            {
                candidate_column: candidate,
                "reference_model": reference_id,
                "observed_delta_mae_bp": observed_delta,
                "bootstrap_mean_delta_mae_bp": statistics.fmean(bootstrap_deltas),
                "paired_se_mae_bp": paired_se,
                "ci95_low_delta_mae_bp": _ordered_quantile(bootstrap_deltas, 0.025),
                "ci95_high_delta_mae_bp": _ordered_quantile(bootstrap_deltas, 0.975),
                "common_dates": len(ordered_dates),
                "equal_weight_strata": len(strata_indices),
                "block_sessions": block_sessions,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "resampling_unit": "moving_whole_Date_block",
                "paired_resampling": True,
                "within_stratum_weighting": (
                    "weighted" if weight_column is not None else "Date_equal"
                ),
                "weight_column": weight_column,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(candidate_column)


def month_equal_anchor_scores(
    daily: pl.DataFrame,
    *,
    actionable_models: Sequence[str],
    controls: Sequence[str] = (),
    date_column: str = "Date",
    month_column: str = "month",
    model_column: str = "model",
    mae_column: str = "mae_bp",
    tv_column: str = "tv_ratio",
    weight_column: str | None = None,
) -> pl.DataFrame:
    """Score actionable anchors on common Dates with equal month weights."""

    control_ids = {str(value) for value in controls}
    candidate_ids = tuple(
        candidate
        for candidate in _normalise_ids(actionable_models, label="actionable_models")
        if candidate not in control_ids
    )
    if not candidate_ids:
        raise ValueError("no selectable actionable models remain after controls")
    extra_columns = [month_column]
    if weight_column is not None:
        extra_columns.append(weight_column)
    panel = _finite_common_daily_panel(
        daily,
        candidates=candidate_ids,
        date_column=date_column,
        candidate_column=model_column,
        metric_columns=(mae_column, tv_column),
        extra_columns=tuple(extra_columns),
    )
    inconsistent_month = (
        panel.group_by(date_column)
        .agg(pl.col(month_column).n_unique().alias("_months"))
        .filter(pl.col("_months") != 1)
    )
    if inconsistent_month.height:
        raise ValueError("each Date must map to exactly one month")

    if weight_column is None:
        panel = panel.with_columns(pl.lit(1.0).alias("_selection_weight"))
    else:
        panel = panel.with_columns(
            pl.col(weight_column).cast(pl.Float64).alias("_selection_weight")
        )
        invalid_weight = panel.filter(
            pl.col("_selection_weight").is_null()
            | ~pl.col("_selection_weight").is_finite()
            | (pl.col("_selection_weight") <= 0)
        )
        if invalid_weight.height:
            raise ValueError("anchor selection weights must be finite and positive")
        inconsistent_weight = (
            panel.group_by(date_column)
            .agg(
                (
                    pl.col("_selection_weight").max()
                    - pl.col("_selection_weight").min()
                ).alias("_weight_range")
            )
            .filter(pl.col("_weight_range").abs() > FLOAT_EPSILON)
        )
        if inconsistent_weight.height:
            raise ValueError("anchor candidates must have the same weight per Date")
    monthly = (
        panel.group_by([model_column, month_column])
        .agg(
            (pl.col(mae_column) * pl.col("_selection_weight"))
            .sum()
            .alias("_weighted_mae_sum"),
            (pl.col(tv_column) * pl.col("_selection_weight"))
            .sum()
            .alias("_weighted_tv_sum"),
            pl.col("_selection_weight").sum().alias("_month_weight"),
            pl.col(date_column).n_unique().alias("_month_dates"),
        )
        .with_columns(
            (pl.col("_weighted_mae_sum") / pl.col("_month_weight")).alias(
                "_monthly_mae_bp"
            ),
            (pl.col("_weighted_tv_sum") / pl.col("_month_weight")).alias(
                "_monthly_tv_ratio"
            ),
        )
    )
    return (
        monthly.group_by(model_column)
        .agg(
            pl.col("_monthly_mae_bp").mean().alias("month_equal_mae_bp"),
            pl.col("_monthly_tv_ratio").mean().alias("month_equal_tv_ratio"),
            pl.col(month_column).n_unique().alias("months"),
            pl.col("_month_dates").sum().alias("common_dates"),
            pl.col("_month_weight").sum().alias("common_weight"),
        )
        .with_columns(pl.lit(True).alias("selectable_actionable"))
        .sort(["month_equal_mae_bp", "month_equal_tv_ratio", model_column])
    )


def select_anchor_top_two(
    daily: pl.DataFrame,
    *,
    actionable_models: Sequence[str],
    controls: Sequence[str] = (),
    date_column: str = "Date",
    month_column: str = "month",
    model_column: str = "model",
    mae_column: str = "mae_bp",
    tv_column: str = "tv_ratio",
    weight_column: str | None = None,
    block_sessions: int = DEFAULT_BLOCK_SESSIONS,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pl.DataFrame:
    """Apply the registered raw-winner plus paired-one-SE smoothness rule.

    The returned frame contains every selectable actionable model and explicit
    ``selection_rank``/``selection_role`` fields.  Rank 1 is the raw
    month-equal MAE winner.  Rank 2 is the smoothest model within one paired
    standard error; if that is the same model, the next-lowest-MAE actionable
    candidate is used so the result remains a genuine top two.
    """

    scores = month_equal_anchor_scores(
        daily,
        actionable_models=actionable_models,
        controls=controls,
        date_column=date_column,
        month_column=month_column,
        model_column=model_column,
        mae_column=mae_column,
        tv_column=tv_column,
        weight_column=weight_column,
    )
    if scores.height < 2:
        raise ValueError("anchor top-two selection requires two actionable models")
    winner = str(scores.row(0, named=True)[model_column])
    candidate_ids = tuple(str(value) for value in scores[model_column].to_list())

    extra_columns = [month_column]
    if weight_column is not None:
        extra_columns.append(weight_column)
    common_daily = _finite_common_daily_panel(
        daily,
        candidates=candidate_ids,
        date_column=date_column,
        candidate_column=model_column,
        metric_columns=(mae_column, tv_column),
        extra_columns=tuple(extra_columns),
    )
    bootstrap = paired_moving_whole_date_block_bootstrap(
        common_daily,
        reference=winner,
        candidates=candidate_ids,
        date_column=date_column,
        candidate_column=model_column,
        metric_column=mae_column,
        weight_column=weight_column,
        strata_column=month_column,
        block_sessions=block_sessions,
        replicates=replicates,
        seed=seed,
    )
    result = scores.join(bootstrap, on=model_column, how="left").with_columns(
        (
            pl.col("observed_delta_mae_bp")
            <= pl.col("paired_se_mae_bp") + FLOAT_EPSILON
        ).alias("within_paired_one_se")
    )

    eligible = result.filter(pl.col("within_paired_one_se")).sort(
        ["month_equal_tv_ratio", "month_equal_mae_bp", model_column]
    )
    smoothest = str(eligible.row(0, named=True)[model_column])
    if smoothest == winner:
        second = str(
            result.filter(pl.col(model_column) != winner)
            .sort(["month_equal_mae_bp", "month_equal_tv_ratio", model_column])
            .row(0, named=True)[model_column]
        )
        second_role = "next_lowest_mae_after_identical_smoothest"
    else:
        second = smoothest
        second_role = "smoothest_within_paired_one_se"

    return result.with_columns(
        pl.when(pl.col(model_column) == winner)
        .then(pl.lit(1))
        .when(pl.col(model_column) == second)
        .then(pl.lit(2))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("selection_rank"),
        pl.when(pl.col(model_column) == winner)
        .then(pl.lit("raw_month_equal_mae_winner"))
        .when(pl.col(model_column) == second)
        .then(pl.lit(second_role))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("selection_role"),
    ).sort(["selection_rank", "month_equal_mae_bp"], nulls_last=True)


def censor_interval_signed_bias(
    nominal_probability: float,
    reach_lower_bound: float,
    reach_upper_bound: float,
) -> float:
    """Signed distance from a nominal probability to an identified interval.

    Positive values mean even the lower bound exceeds nominal (too many
    reaches); negative values mean even the upper bound is below nominal (too
    few reaches).  A nominal probability inside the interval has zero bias.
    """

    nominal = float(nominal_probability)
    lower = float(reach_lower_bound)
    upper = float(reach_upper_bound)
    if not all(math.isfinite(value) for value in (nominal, lower, upper)):
        raise ValueError("interval inputs must be finite")
    if not 0.0 <= nominal <= 1.0:
        raise ValueError("nominal_probability must be in [0, 1]")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("reach interval must satisfy 0 <= lower <= upper <= 1")
    if nominal < lower:
        return lower - nominal
    if nominal > upper:
        return upper - nominal
    return 0.0


def censor_interval_distance(
    nominal_probability: float,
    reach_lower_bound: float,
    reach_upper_bound: float,
) -> float:
    """Absolute distance from nominal probability to a censor interval."""

    return abs(
        censor_interval_signed_bias(
            nominal_probability,
            reach_lower_bound,
            reach_upper_bound,
        )
    )


def with_censor_interval_loss(
    frame: pl.DataFrame,
    *,
    nominal_column: str = "nominal_probability",
    lower_column: str = "reach_lower_bound",
    upper_column: str = "reach_upper_bound",
    distance_column: str = "censor_interval_distance",
    signed_bias_column: str = "censor_interval_signed_bias",
) -> pl.DataFrame:
    """Append vectorized censor-interval distance and signed-bias columns."""

    _require_columns(
        frame,
        [nominal_column, lower_column, upper_column],
        "censor interval frame",
    )
    casted = frame.with_columns(
        pl.col(nominal_column).cast(pl.Float64),
        pl.col(lower_column).cast(pl.Float64),
        pl.col(upper_column).cast(pl.Float64),
    )
    invalid = casted.filter(
        pl.any_horizontal(
            pl.col(nominal_column).is_null(),
            pl.col(lower_column).is_null(),
            pl.col(upper_column).is_null(),
            ~pl.col(nominal_column).is_finite(),
            ~pl.col(lower_column).is_finite(),
            ~pl.col(upper_column).is_finite(),
            pl.col(nominal_column) < 0.0,
            pl.col(nominal_column) > 1.0,
            pl.col(lower_column) < 0.0,
            pl.col(upper_column) > 1.0,
            pl.col(lower_column) > pl.col(upper_column),
        )
    )
    if invalid.height:
        raise ValueError(
            "invalid censor interval rows: "
            f"{invalid.select(nominal_column, lower_column, upper_column).head(5).to_dicts()}"
        )
    signed = (
        pl.when(pl.col(nominal_column) < pl.col(lower_column))
        .then(pl.col(lower_column) - pl.col(nominal_column))
        .when(pl.col(nominal_column) > pl.col(upper_column))
        .then(pl.col(upper_column) - pl.col(nominal_column))
        .otherwise(0.0)
    )
    return casted.with_columns(
        signed.alias(signed_bias_column),
        signed.abs().alias(distance_column),
    )


def _spearman_summary(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    predicted_column: str,
    realized_column: str,
    minimum_observations: int,
    output_column: str,
) -> pl.DataFrame:
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2")
    _require_columns(
        frame,
        [*group_columns, predicted_column, realized_column],
        "Spearman frame",
    )
    rows: list[dict[str, object]] = []
    for raw_key, group in frame.group_by(list(group_columns), maintain_order=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        valid = group.filter(
            pl.col(predicted_column).is_not_null()
            & pl.col(realized_column).is_not_null()
            & pl.col(predicted_column).cast(pl.Float64).is_finite()
            & pl.col(realized_column).cast(pl.Float64).is_finite()
        )
        correlation: float | None = None
        if valid.height >= minimum_observations:
            value = valid.select(
                pl.corr(
                    predicted_column,
                    realized_column,
                    method="spearman",
                )
            ).item()
            if value is not None and math.isfinite(float(value)):
                correlation = float(value)
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "observations": valid.height,
                output_column: correlation,
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows, infer_schema_length=None).sort(list(group_columns))


def daily_cross_sectional_spearman(
    frame: pl.DataFrame,
    *,
    predicted_column: str,
    realized_column: str,
    extra_group_columns: Sequence[str] = (),
    date_column: str = "Date",
    minimum_products: int = 3,
) -> pl.DataFrame:
    """Compute one cross-product Spearman coefficient per Date and cell."""

    return _spearman_summary(
        frame,
        group_columns=(*extra_group_columns, date_column),
        predicted_column=predicted_column,
        realized_column=realized_column,
        minimum_observations=minimum_products,
        output_column="daily_cross_product_spearman",
    )


def temporal_product_spearman(
    frame: pl.DataFrame,
    *,
    predicted_column: str,
    realized_column: str,
    extra_group_columns: Sequence[str] = (),
    product_columns: Sequence[str] = ("ValueCode", "QuoteCode"),
    minimum_dates: int = 3,
) -> pl.DataFrame:
    """Compute per-product temporal Spearman coefficients across Dates."""

    return _spearman_summary(
        frame,
        group_columns=(*extra_group_columns, *product_columns),
        predicted_column=predicted_column,
        realized_column=realized_column,
        minimum_observations=minimum_dates,
        output_column="temporal_product_spearman",
    )


def common_support_units(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    unit_columns: Sequence[str],
    candidate_column: str = "candidate",
    supported_column: str | None = None,
    required_value_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Return units natively supported by every requested candidate.

    The candidate/unit key must be unique.  Include q, side, and TOD columns in
    ``unit_columns`` whenever they are part of the comparison cell.
    """

    candidate_ids = _normalise_ids(candidates, label="candidates")
    required = {candidate_column, *unit_columns, *required_value_columns}
    if supported_column is not None:
        required.add(supported_column)
    _require_columns(frame, required, "support frame")
    panel = frame.filter(
        pl.col(candidate_column).cast(pl.String).is_in(candidate_ids)
    ).with_columns(pl.col(candidate_column).cast(pl.String))
    duplicate = (
        panel.group_by([candidate_column, *unit_columns])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError(
            "support frame must have one row per candidate-unit; include all "
            "cell keys in unit_columns"
        )
    if supported_column is not None:
        panel = panel.filter(pl.col(supported_column).fill_null(False).cast(pl.Boolean))
    for column in required_value_columns:
        expression = pl.col(column).is_not_null()
        if panel.schema[column].is_numeric():
            expression &= pl.col(column).cast(pl.Float64).is_finite()
        panel = panel.filter(expression)

    return (
        panel.group_by(list(unit_columns))
        .agg(pl.col(candidate_column).n_unique().alias("_supported_candidates"))
        .filter(pl.col("_supported_candidates") == len(candidate_ids))
        .select(*unit_columns)
        .sort(list(unit_columns))
    )


def filter_common_support(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    unit_columns: Sequence[str],
    candidate_column: str = "candidate",
    supported_column: str | None = None,
    required_value_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Filter a long candidate table to the exact common-support units."""

    units = common_support_units(
        frame,
        candidates=candidates,
        unit_columns=unit_columns,
        candidate_column=candidate_column,
        supported_column=supported_column,
        required_value_columns=required_value_columns,
    )
    candidate_ids = _normalise_ids(candidates, label="candidates")
    supported = frame.filter(
        pl.col(candidate_column).cast(pl.String).is_in(candidate_ids)
    )
    if supported_column is not None:
        supported = supported.filter(
            pl.col(supported_column).fill_null(False).cast(pl.Boolean)
        )
    for column in required_value_columns:
        expression = pl.col(column).is_not_null()
        if supported.schema[column].is_numeric():
            expression &= pl.col(column).cast(pl.Float64).is_finite()
        supported = supported.filter(expression)
    return supported.join(units, on=list(unit_columns), how="inner").sort(
        [*unit_columns, candidate_column]
    )


def native_support_summary(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    unit_columns: Sequence[str],
    candidate_column: str = "candidate",
    supported_column: str | None = None,
    required_value_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Count each candidate's native units and shared comparison units."""

    candidate_ids = _normalise_ids(candidates, label="candidates")
    common = common_support_units(
        frame,
        candidates=candidate_ids,
        unit_columns=unit_columns,
        candidate_column=candidate_column,
        supported_column=supported_column,
        required_value_columns=required_value_columns,
    )
    common_count = common.height
    rows: list[dict[str, object]] = []
    for candidate in candidate_ids:
        subset = frame.filter(pl.col(candidate_column).cast(pl.String) == candidate)
        if supported_column is not None:
            subset = subset.filter(
                pl.col(supported_column).fill_null(False).cast(pl.Boolean)
            )
        for column in required_value_columns:
            expression = pl.col(column).is_not_null()
            if subset.schema[column].is_numeric():
                expression &= pl.col(column).cast(pl.Float64).is_finite()
            subset = subset.filter(expression)
        native_count = subset.select(*unit_columns).unique().height
        rows.append(
            {
                candidate_column: candidate,
                "native_support_units": native_count,
                "common_support_units": common_count,
                "common_over_native": (
                    common_count / native_count if native_count else None
                ),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(candidate_column)


def lexicographic_rank(
    scores: pl.DataFrame,
    *,
    candidate_column: str = "candidate",
    criteria: Sequence[tuple[str, SortDirection]],
    simpler_order: Sequence[str] = (),
    rank_column: str = "selection_rank",
) -> pl.DataFrame:
    """Rank one-row-per-candidate scores by exact ordered criteria."""

    if not criteria:
        raise ValueError("criteria must not be empty")
    _require_columns(
        scores,
        [candidate_column, *(column for column, _ in criteria)],
        "lexicographic score table",
    )
    duplicate = scores.group_by(candidate_column).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("lexicographic score table must have one row per candidate")
    invalid_direction = [
        direction for _, direction in criteria if direction not in {"min", "max"}
    ]
    if invalid_direction:
        raise ValueError(f"invalid sort directions: {invalid_direction}")
    metric_columns = [column for column, _ in criteria]
    normalised = scores.with_columns(
        *[pl.col(column).cast(pl.Float64) for column in metric_columns]
    )
    invalid_metrics = normalised.filter(
        pl.any_horizontal(
            *(
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in metric_columns
            )
        )
    )
    if invalid_metrics.height:
        raise ValueError(
            "lexicographic score criteria must be finite for every candidate: "
            f"{invalid_metrics.select(candidate_column, *metric_columns).head(5).to_dicts()}"
        )

    order = {str(candidate): index for index, candidate in enumerate(simpler_order)}
    complexity_default = len(order)
    ranked = normalised.with_columns(
        pl.col(candidate_column)
        .cast(pl.String)
        .replace_strict(order, default=complexity_default)
        .cast(pl.Int64)
        .alias("_simplicity_rank")
    )
    sort_columns = [column for column, _ in criteria]
    descending = [direction == "max" for _, direction in criteria]
    ranked = ranked.sort(
        [*sort_columns, "_simplicity_rank", candidate_column],
        descending=[*descending, False, False],
        nulls_last=True,
    )
    return ranked.with_row_index(rank_column, offset=1).drop("_simplicity_rank")


def rank_q_candidates(
    scores: pl.DataFrame,
    *,
    candidate_column: str = "candidate",
    selectable_column: str = "allow_primary_selection",
    simpler_order: Sequence[str] = (),
) -> pl.DataFrame:
    """Apply the registry's q-candidate lexicographic selection order."""

    _require_columns(scores, [selectable_column], "q score table")
    selectable = scores.filter(
        pl.col(selectable_column).fill_null(False).cast(pl.Boolean)
    )
    if selectable.is_empty():
        raise ValueError("q score table has no selectable candidates")
    return lexicographic_rank(
        selectable,
        candidate_column=candidate_column,
        criteria=Q_LEXICOGRAPHIC_CRITERIA,
        simpler_order=simpler_order,
    )


def select_q_candidate(
    scores: pl.DataFrame,
    *,
    candidate_column: str = "candidate",
    selectable_column: str = "allow_primary_selection",
    simpler_order: Sequence[str] = (),
) -> pl.DataFrame:
    """Return the single highest-ranked q candidate with its score evidence."""

    ranked = rank_q_candidates(
        scores,
        candidate_column=candidate_column,
        selectable_column=selectable_column,
        simpler_order=simpler_order,
    )
    if ranked.is_empty():
        raise ValueError("q score table must not be empty")
    return ranked.head(1)
