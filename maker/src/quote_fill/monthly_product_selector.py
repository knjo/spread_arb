"""Leakage-safe monthly product selector for the short-basis maker study.

The old ``first_wave_45`` cohort is a retrospective research universe: its
membership was selected from realised May--August liquidity and then replayed
over earlier dates.  This module builds a replacement research table whose
membership for calendar month ``M`` uses only labels from the complete month
``M-1``.  A causal route-specific liquidity row is still applied on every
target date.

The broad-universe convergence label is deliberately a proxy.  A positive
q95 excursion is treated as a short-basis opportunity, and the next negative
q95 excursion on the same date is treated as a dynamic full-band return.  It
does not claim a maker fill or a frozen-threshold exit.  Exact execution
approval still requires raw queue, 50 ms hedge, and exit replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import completed_artifact_paths


DEFAULT_BOUNDARY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "rolling_boundaries"
    / "rolling_boundary_snapshots.parquet"
)
DEFAULT_LIQUIDITY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "liquidity"
    / "rolling_liquidity_screen.parquet"
)
DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "monthly_product_selector_causal_v2_20260822"
)

KEYS = ["Date", "ValueCode", "QuoteCode"]


@dataclass(frozen=True)
class MonthlyProductSelectorConfig:
    """Frozen research thresholds for the first monthly selector."""

    boundary_quantile: int = 95
    liquidity_proxy_quantile: int = 80
    entry_route: str = "spot_bid_future_taker"
    minimum_pass_days: int = 10
    minimum_upper_events: int = 20
    minimum_signal_days: int = 10
    minimum_lcb80: float = 0.70
    maximum_wait_upper_bound_p90_seconds: int = 3_600
    wilson_z: float = 1.2815515655446004
    selector_version: str = "monthly_short_basis_selector_causal_v2"

    def validate(self) -> None:
        if not 0 < self.boundary_quantile < 100:
            raise ValueError("boundary_quantile must be in (0, 100)")
        if not 0 < self.liquidity_proxy_quantile < 100:
            raise ValueError("liquidity_proxy_quantile must be in (0, 100)")
        if not self.entry_route:
            raise ValueError("entry_route cannot be empty")
        if (
            self.minimum_pass_days <= 0
            or self.minimum_upper_events <= 0
            or self.minimum_signal_days <= 0
        ):
            raise ValueError("support thresholds must be positive")
        if not 0.0 <= self.minimum_lcb80 <= 1.0:
            raise ValueError("minimum_lcb80 must be in [0, 1]")
        if self.maximum_wait_upper_bound_p90_seconds <= 0:
            raise ValueError("maximum wait must be positive")
        if self.wilson_z <= 0:
            raise ValueError("Wilson z must be positive")


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalise_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )


def _next_month(value: str) -> str:
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"month must use YYYYMM: {value!r}")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month: {value!r}")
    if month == 12:
        return f"{year + 1:04d}01"
    return f"{year:04d}{month + 1:02d}"


def _safe_boundary_rows(
    boundaries: pl.DataFrame,
    config: MonthlyProductSelectorConfig,
) -> pl.DataFrame:
    required = {
        *KEYS,
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "adaptive_parameter_valid",
        "positive_completed_per_session",
        "negative_completed_per_session",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require_columns(boundaries, required, "rolling boundaries")
    result = (
        _normalise_keys(boundaries)
        .filter(
            (pl.col("boundary_quantile") == config.boundary_quantile)
            & pl.col("adaptive_parameter_valid").fill_null(False)
        )
        .with_columns(pl.col("source_asof_date").cast(pl.String))
    )
    unsafe = result.filter(
        pl.any_horizontal([pl.col(column).is_null() for column in KEYS])
        | pl.col("source_asof_date").is_null()
        | ~pl.col("Date").str.contains(r"^\d{8}$").fill_null(False)
        | ~pl.col("source_asof_date").str.contains(r"^\d{8}$").fill_null(False)
        | ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
        | (pl.col("source_asof_date") >= pl.col("Date"))
        | pl.col("upper_distance_bp").is_null()
        | pl.col("lower_distance_bp").is_null()
        | ~pl.col("upper_distance_bp").is_finite()
        | ~pl.col("lower_distance_bp").is_finite()
        | (pl.col("upper_distance_bp") <= 0)
        | (pl.col("lower_distance_bp") <= 0)
    )
    if unsafe.height:
        raise ValueError("rolling boundary input is not D-safe and finite")
    if result.select(KEYS).n_unique() != result.height:
        raise ValueError("rolling boundaries duplicate a product-day")
    return result


def _safe_liquidity_rows(
    liquidity: pl.DataFrame,
    config: MonthlyProductSelectorConfig,
) -> pl.DataFrame:
    required = {
        *KEYS,
        "boundary_quantile",
        "route",
        "pre_replay_candidate",
        "liquidity_gate_status",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require_columns(liquidity, required, "rolling liquidity screen")
    result = (
        _normalise_keys(liquidity)
        .filter(
            (pl.col("boundary_quantile") == config.liquidity_proxy_quantile)
            & (pl.col("route") == config.entry_route)
        )
        .with_columns(pl.col("source_asof_date").cast(pl.String))
    )
    unsafe = result.filter(
        pl.any_horizontal([pl.col(column).is_null() for column in KEYS])
        | pl.col("source_asof_date").is_null()
        | ~pl.col("Date").str.contains(r"^\d{8}$").fill_null(False)
        | ~pl.col("source_asof_date").str.contains(r"^\d{8}$").fill_null(False)
        | ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if unsafe.height:
        raise ValueError("liquidity input is not D-safe")
    if result.select(KEYS).n_unique() != result.height:
        raise ValueError("liquidity screen duplicates a product-day route")
    return result


def build_dynamic_full_band_events(
    boundaries: pl.DataFrame,
    liquidity: pl.DataFrame,
    excursions: pl.DataFrame,
    *,
    config: MonthlyProductSelectorConfig = MonthlyProductSelectorConfig(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build broad-universe q95 events and the eligible product-day base.

    The exact first q95 reach timestamps are not present in the compact
    excursion publication.  We therefore retain a rigorous interval:
    ``wait_lower_bound_seconds`` is lower-start minus upper-end, while
    ``wait_upper_bound_seconds`` is lower-end minus upper-start.  The same-day
    hit indicator itself is valid because the two non-overlapping excursions
    both exceeded their D-1 distances in temporal order.
    """

    config.validate()
    boundary = _safe_boundary_rows(boundaries, config)
    liquid = _safe_liquidity_rows(liquidity, config)
    base = boundary.join(
        liquid.select(
            *KEYS,
            "pre_replay_candidate",
            "liquidity_gate_status",
            pl.col("source_asof_date").alias("liquidity_source_asof_date"),
        ),
        on=KEYS,
        how="inner",
        validate="1:1",
    ).filter(pl.col("pre_replay_candidate").fill_null(False))

    required = {
        *KEYS,
        "side",
        "amplitude_bp",
        "start_seconds_from_open",
        "end_seconds_from_open",
    }
    _require_columns(excursions, required, "daily excursions")
    joined = _normalise_keys(excursions).join(
        base.select(
            *KEYS,
            "upper_distance_bp",
            "lower_distance_bp",
            pl.col("source_asof_date").alias("boundary_source_asof_date"),
            "liquidity_source_asof_date",
        ),
        on=KEYS,
        how="inner",
        validate="m:1",
    )
    upper = (
        joined.filter(
            (pl.col("side") == "positive")
            & (pl.col("amplitude_bp") >= pl.col("upper_distance_bp"))
        )
        .select(
            *KEYS,
            pl.col("start_seconds_from_open").alias("upper_start_seconds"),
            pl.col("end_seconds_from_open").alias("upper_end_seconds"),
            "upper_distance_bp",
            "lower_distance_bp",
            "boundary_source_asof_date",
            "liquidity_source_asof_date",
        )
        .sort([*KEYS, "upper_start_seconds"])
    )
    lower = (
        joined.filter(
            (pl.col("side") == "negative")
            & (pl.col("amplitude_bp") >= pl.col("lower_distance_bp"))
        )
        .select(
            *KEYS,
            pl.col("start_seconds_from_open").alias("lower_start_seconds"),
            pl.col("end_seconds_from_open").alias("lower_end_seconds"),
        )
        .sort([*KEYS, "lower_start_seconds"])
    )
    if upper.is_empty():
        events = upper.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("lower_start_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("lower_end_seconds"),
            pl.lit(False).alias("same_day_dynamic_lower_hit"),
            pl.lit(None, dtype=pl.Int64).alias("start_to_start_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("wait_lower_bound_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("wait_upper_bound_seconds"),
        )
    elif lower.is_empty():
        events = upper.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("lower_start_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("lower_end_seconds"),
            pl.lit(False).alias("same_day_dynamic_lower_hit"),
            pl.lit(None, dtype=pl.Int64).alias("start_to_start_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("wait_lower_bound_seconds"),
            pl.lit(None, dtype=pl.Int64).alias("wait_upper_bound_seconds"),
        )
    else:
        events = upper.join_asof(
            lower,
            left_on="upper_start_seconds",
            right_on="lower_start_seconds",
            by=KEYS,
            strategy="forward",
            allow_exact_matches=False,
            check_sortedness=False,
        ).with_columns(
            pl.col("lower_start_seconds")
            .is_not_null()
            .alias("same_day_dynamic_lower_hit"),
            (pl.col("lower_start_seconds") - pl.col("upper_start_seconds")).alias(
                "start_to_start_seconds"
            ),
            (pl.col("lower_start_seconds") - pl.col("upper_end_seconds"))
            .clip(lower_bound=0)
            .alias("wait_lower_bound_seconds"),
            (pl.col("lower_end_seconds") - pl.col("upper_start_seconds")).alias(
                "wait_upper_bound_seconds"
            ),
        )
    events = events.with_columns(
        pl.col("Date").str.slice(0, 6).alias("observation_month"),
        pl.lit(True).alias("contains_target_day_outcome"),
        pl.lit(False).alias("maker_fill_included"),
        pl.lit("dynamic_excursion_full_band_proxy").alias("label_semantics"),
    ).sort([*KEYS, "upper_start_seconds"])
    base = base.with_columns(
        pl.col("Date").str.slice(0, 6).alias("observation_month")
    ).sort(KEYS)
    return events, base


def _wilson_lower(
    successes: pl.Expr,
    trials: pl.Expr,
    z: float,
) -> pl.Expr:
    p = successes.cast(pl.Float64) / trials.cast(pl.Float64)
    z2 = float(z) ** 2
    return (
        (p + z2 / (2.0 * trials))
        - float(z)
        * ((p * (1.0 - p) / trials) + z2 / (4.0 * trials**2)).sqrt()
    ) / (1.0 + z2 / trials)


def build_monthly_product_metrics(
    events: pl.DataFrame,
    eligible_product_days: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    config: MonthlyProductSelectorConfig = MonthlyProductSelectorConfig(),
) -> pl.DataFrame:
    """Aggregate source-month labels without using the effective month."""

    config.validate()
    _require_columns(
        events,
        {
            "Date",
            "ValueCode",
            "observation_month",
            "same_day_dynamic_lower_hit",
            "wait_upper_bound_seconds",
        },
        "dynamic full-band events",
    )
    _require_columns(
        eligible_product_days,
        {"Date", "ValueCode", "observation_month"},
        "eligible product-day base",
    )
    base = eligible_product_days.group_by(
        ["observation_month", "ValueCode"]
    ).agg(
        pl.col("Date").n_unique().alias("prior_pass_days"),
        pl.col("Date").min().alias("source_month_first_date"),
        pl.col("Date").max().alias("source_month_last_date"),
    )
    event_metrics = events.group_by(["observation_month", "ValueCode"]).agg(
        pl.len().alias("prior_upper_events"),
        pl.col("same_day_dynamic_lower_hit")
        .sum()
        .cast(pl.Int64)
        .alias("prior_same_day_lower_hits"),
        pl.col("wait_upper_bound_seconds")
        .filter(pl.col("same_day_dynamic_lower_hit"))
        .median()
        .alias("prior_wait_upper_bound_p50_seconds"),
        pl.col("wait_upper_bound_seconds")
        .filter(pl.col("same_day_dynamic_lower_hit"))
        .quantile(0.90)
        .alias("prior_wait_upper_bound_p90_seconds"),
    )
    daily_metrics = (
        events.group_by(["observation_month", "ValueCode", "Date"])
        .agg(
            pl.len().alias("daily_upper_events"),
            pl.col("same_day_dynamic_lower_hit")
            .mean()
            .alias("daily_same_day_lower_rate"),
            pl.col("same_day_dynamic_lower_hit")
            .all()
            .alias("all_entries_returned_same_day"),
        )
        .group_by(["observation_month", "ValueCode"])
        .agg(
            pl.len().alias("prior_signal_days"),
            pl.col("daily_same_day_lower_rate")
            .mean()
            .alias("prior_daily_mean_same_day_lower_rate"),
            pl.col("daily_same_day_lower_rate")
            .std(ddof=1)
            .alias("prior_daily_same_day_lower_rate_std"),
            pl.col("all_entries_returned_same_day")
            .mean()
            .alias("prior_all_entries_returned_day_rate"),
        )
    )
    monthly = base.join(
        event_metrics,
        on=["observation_month", "ValueCode"],
        how="left",
        validate="1:1",
    ).join(
        daily_metrics,
        on=["observation_month", "ValueCode"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("prior_upper_events").fill_null(0),
        pl.col("prior_signal_days").fill_null(0),
        pl.col("prior_same_day_lower_hits").fill_null(0),
    )
    monthly = monthly.with_columns(
        pl.when(pl.col("prior_upper_events") > 0)
        .then(
            pl.col("prior_same_day_lower_hits")
            / pl.col("prior_upper_events")
        )
        .otherwise(None)
        .alias("prior_event_same_day_lower_rate"),
        (pl.col("prior_upper_events") / pl.col("prior_pass_days")).alias(
            "prior_upper_events_per_pass_day"
        ),
    ).with_columns(
        pl.when(pl.col("prior_upper_events") > 0)
        .then(
            _wilson_lower(
                pl.col("prior_same_day_lower_hits"),
                pl.col("prior_upper_events"),
                config.wilson_z,
            )
        )
        .otherwise(None)
        .alias("prior_event_same_day_lower_wilson_lcb80"),
        pl.when(pl.col("prior_signal_days") > 0)
        .then(
            (
                pl.col("prior_daily_mean_same_day_lower_rate")
                - config.wilson_z
                * pl.col("prior_daily_same_day_lower_rate_std").fill_null(0.0)
                / pl.col("prior_signal_days").cast(pl.Float64).sqrt()
            ).clip(0.0, 1.0)
        )
        .otherwise(None)
        .alias("prior_daily_same_day_lower_lcb80"),
    )

    safe_boundary = _safe_boundary_rows(boundaries, config).with_columns(
        pl.col("Date").str.slice(0, 6).alias("observation_month"),
        (
            pl.min_horizontal(
                "positive_completed_per_session",
                "negative_completed_per_session",
            )
            / (pl.col("upper_distance_bp") + pl.col("lower_distance_bp"))
        ).alias("cycle_per_band"),
    )
    structural = (
        safe_boundary.sort(["observation_month", "ValueCode", "Date"])
        .group_by(["observation_month", "ValueCode"], maintain_order=True)
        .agg(
            pl.col("Date").last().alias("structural_snapshot_date"),
            pl.col("source_asof_date").last().alias(
                "structural_source_asof_date"
            ),
            pl.col("cycle_per_band").last().alias("prior_cycle_per_band"),
        )
        .with_columns(
            (
                pl.col("prior_cycle_per_band").rank(method="average").over(
                    "observation_month"
                )
                / pl.len().over("observation_month")
            ).alias("prior_cycle_per_band_percentile")
        )
    )
    return (
        monthly.join(
            structural,
            on=["observation_month", "ValueCode"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("observation_month").alias("source_month"),
            pl.lit(config.selector_version).alias("selector_version"),
            pl.lit(True).alias("source_month_outcomes_included"),
            pl.lit(False).alias("effective_month_outcomes_included"),
            pl.lit(False).alias("maker_fill_included"),
        )
        .drop("observation_month")
        .sort(["source_month", "ValueCode"])
    )


def build_monthly_membership(
    monthly_metrics: pl.DataFrame,
    *,
    config: MonthlyProductSelectorConfig = MonthlyProductSelectorConfig(),
    completed_source_months: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Freeze one complete source month's metrics for the following month.

    By default, completion is inferred only after an observation from the
    following month exists.  A live publisher running before that first
    observation must pass an externally verified ``completed_source_months``
    exchange calendar.
    """

    config.validate()
    required = {
        "source_month",
        "ValueCode",
        "source_month_last_date",
        "prior_pass_days",
        "prior_upper_events",
        "prior_signal_days",
        "prior_daily_same_day_lower_lcb80",
        "prior_wait_upper_bound_p90_seconds",
    }
    _require_columns(monthly_metrics, required, "monthly metrics")
    if monthly_metrics.is_empty():
        raise ValueError("monthly metrics contain no source month")
    invalid_source = monthly_metrics.filter(
        pl.col("source_month").is_null()
        | pl.col("ValueCode").is_null()
        | pl.col("source_month_last_date").is_null()
        | ~pl.col("source_month").str.contains(r"^\d{6}$").fill_null(False)
        | ~pl.col("source_month_last_date")
        .str.contains(r"^\d{8}$")
        .fill_null(False)
    )
    if invalid_source.height:
        raise ValueError("monthly metrics contain invalid source chronology")
    months = monthly_metrics["source_month"].unique().sort().to_list()
    if completed_source_months is None:
        observed = set(months)
        completed = tuple(
            month for month in months if _next_month(month) in observed
        )
    else:
        completed = tuple(dict.fromkeys(completed_source_months))
        unknown = sorted(set(completed) - set(months))
        if unknown:
            raise ValueError(f"completed source months absent from metrics: {unknown}")
    if not completed:
        raise ValueError("no complete source month is available for membership")
    month_map = pl.DataFrame(
        {
            "source_month": completed,
            "effective_month": [_next_month(value) for value in completed],
        }
    )
    result = monthly_metrics.join(
        month_map, on="source_month", how="inner", validate="m:1"
    ).with_columns(
        (
            (pl.col("prior_pass_days") >= config.minimum_pass_days)
            & (pl.col("prior_upper_events") >= config.minimum_upper_events)
            & (pl.col("prior_signal_days") >= config.minimum_signal_days)
            & pl.col("prior_daily_same_day_lower_lcb80").is_not_null()
        ).alias("support_gate")
    )
    support_ranks = (
        result.filter(pl.col("support_gate"))
        .sort(
            [
                "effective_month",
                "prior_daily_same_day_lower_lcb80",
                "prior_wait_upper_bound_p90_seconds",
                "prior_upper_events",
                "ValueCode",
            ],
            descending=[False, True, False, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over("effective_month")
            .alias("lcb_rank_in_effective_month")
        )
        .select("effective_month", "ValueCode", "lcb_rank_in_effective_month")
    )
    result = result.join(
        support_ranks,
        on=["effective_month", "ValueCode"],
        how="left",
        validate="1:1",
    ).with_columns(
        (
            pl.col("support_gate")
            & (pl.col("prior_daily_same_day_lower_lcb80") >= config.minimum_lcb80)
            & pl.col("prior_wait_upper_bound_p90_seconds").is_not_null()
            & (
                pl.col("prior_wait_upper_bound_p90_seconds")
                <= config.maximum_wait_upper_bound_p90_seconds
            )
        ).alias("monthly_primary_selected")
    )
    primary_ranks = (
        result.filter(pl.col("monthly_primary_selected"))
        .sort(
            [
                "effective_month",
                "prior_daily_same_day_lower_lcb80",
                "prior_wait_upper_bound_p90_seconds",
                "prior_upper_events",
                "ValueCode",
            ],
            descending=[False, True, False, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over("effective_month")
            .alias("primary_lcb_rank_in_effective_month")
        )
        .select(
            "effective_month", "ValueCode", "primary_lcb_rank_in_effective_month"
        )
    )
    result = result.join(
        primary_ranks,
        on=["effective_month", "ValueCode"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.when(pl.col("monthly_primary_selected"))
        .then(pl.lit("primary_threshold"))
        .when(pl.col("support_gate"))
        .then(pl.lit("supported_not_selected"))
        .otherwise(pl.lit("insufficient_support"))
        .alias("monthly_selection_tier"),
        pl.lit(False).alias("retrospective_multi_window_selection"),
        pl.lit(False).alias("selection_contains_effective_month_outcome"),
        pl.lit(True).alias("runtime_daily_liquidity_gate_required"),
        pl.lit(False).alias("production_universe_approved"),
    )
    bad = result.filter(
        (pl.col("source_month") >= pl.col("effective_month"))
        | (
            pl.col("source_month_last_date").str.slice(0, 6)
            != pl.col("source_month")
        )
    )
    if bad.height:
        raise ValueError("monthly membership violates source/effective chronology")
    return result.sort(
        [
            "effective_month",
            "monthly_primary_selected",
            "primary_lcb_rank_in_effective_month",
        ],
        descending=[False, True, False],
        nulls_last=True,
    )


def build_daily_allowlist(
    membership: pl.DataFrame,
    liquidity: pl.DataFrame,
    *,
    config: MonthlyProductSelectorConfig = MonthlyProductSelectorConfig(),
) -> pl.DataFrame:
    """Combine month-frozen membership with each D's causal runtime gate."""

    config.validate()
    liquid = _safe_liquidity_rows(liquidity, config).with_columns(
        pl.col("Date").str.slice(0, 6).alias("effective_month")
    )
    keep = [
        "effective_month",
        "source_month",
        "ValueCode",
        "source_month_last_date",
        "monthly_primary_selected",
        "monthly_selection_tier",
        "prior_pass_days",
        "prior_upper_events",
        "prior_signal_days",
        "prior_event_same_day_lower_rate",
        "prior_event_same_day_lower_wilson_lcb80",
        "prior_daily_mean_same_day_lower_rate",
        "prior_daily_same_day_lower_lcb80",
        "prior_all_entries_returned_day_rate",
        "prior_wait_upper_bound_p90_seconds",
        "prior_cycle_per_band",
        "prior_cycle_per_band_percentile",
        "selector_version",
    ]
    _require_columns(membership, set(keep), "monthly membership")
    daily = liquid.join(
        membership.select(keep),
        on=["effective_month", "ValueCode"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("monthly_primary_selected").fill_null(False),
        pl.col("pre_replay_candidate").fill_null(False).alias(
            "runtime_liquidity_gate_pass"
        ),
    ).with_columns(
        (
            pl.col("monthly_primary_selected")
            & pl.col("runtime_liquidity_gate_pass")
        ).alias("new_entry_allowed"),
    ).with_columns(
        (~pl.col("new_entry_allowed")).alias("exit_only_if_inventory"),
        pl.lit(False).alias("force_flat_on_membership_removal"),
        pl.lit(False).alias("contains_target_day_outcome"),
        pl.lit(False).alias("production_strategy_go"),
    )
    bad = daily.filter(
        pl.col("new_entry_allowed")
        & (
            pl.col("source_month").is_null()
            | (pl.col("source_month") >= pl.col("effective_month"))
            | (pl.col("source_month_last_date") >= pl.col("Date"))
            | (pl.col("source_asof_date") >= pl.col("Date"))
        )
    )
    if bad.height:
        raise ValueError("daily allowlist contains a non-causal admission")
    return daily.sort(["Date", "ValueCode"])


def evaluate_monthly_membership(
    membership: pl.DataFrame,
    events: pl.DataFrame,
) -> pl.DataFrame:
    """Offline target-month diagnostic; never joined back into membership."""

    if membership.is_empty():
        return pl.DataFrame(
            schema={
                "effective_month": pl.String,
                "policy": pl.String,
                "target_events": pl.Int64,
                "target_same_day_lower_hits": pl.Int64,
                "target_same_day_lower_rate": pl.Float64,
                "contains_effective_month_outcome": pl.Boolean,
                "membership_uses_this_result": pl.Boolean,
            }
        )
    target = events.with_columns(
        pl.col("Date").str.slice(0, 6).alias("effective_month")
    )
    rows: list[dict[str, object]] = []
    for policy_column in ("monthly_primary_selected",):
        selected = membership.filter(pl.col(policy_column)).select(
            "effective_month", "ValueCode"
        )
        joined = target.join(
            selected,
            on=["effective_month", "ValueCode"],
            how="inner",
            validate="m:1",
        )
        for month in membership["effective_month"].unique().sort().to_list():
            part = joined.filter(pl.col("effective_month") == month)
            n = part.height
            hits = (
                int(part["same_day_dynamic_lower_hit"].sum()) if n else 0
            )
            rows.append(
                {
                    "effective_month": month,
                    "policy": policy_column,
                    "target_events": n,
                    "target_same_day_lower_hits": hits,
                    "target_same_day_lower_rate": hits / n if n else None,
                    "contains_effective_month_outcome": True,
                    "membership_uses_this_result": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["effective_month", "policy"]
    )


def _build_events_from_daily_root(
    boundaries: pl.DataFrame,
    liquidity: pl.DataFrame,
    daily_root: Path,
    *,
    config: MonthlyProductSelectorConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Process one daily publication at a time to bound peak memory."""

    paths = completed_artifact_paths(daily_root, "excursions.parquet")
    if not paths:
        raise FileNotFoundError(f"no daily excursions below {daily_root}")
    columns = [
        *KEYS,
        "side",
        "amplitude_bp",
        "start_seconds_from_open",
        "end_seconds_from_open",
    ]
    event_parts: list[pl.DataFrame] = []
    eligible_parts: list[pl.DataFrame] = []
    for path in paths:
        # Excursion publications are wide.  Projection plus day-at-a-time
        # materialisation prevents the 131-day source from expanding to
        # several GB in the selector process.
        excursions = pl.read_parquet(path, columns=columns)
        dates = excursions["Date"].cast(pl.String).unique().to_list()
        day_boundaries = boundaries.filter(
            pl.col("Date").cast(pl.String).is_in(dates)
        )
        day_liquidity = liquidity.filter(
            pl.col("Date").cast(pl.String).is_in(dates)
        )
        events, eligible = build_dynamic_full_band_events(
            day_boundaries,
            day_liquidity,
            excursions,
            config=config,
        )
        if not events.is_empty():
            event_parts.append(events)
        if not eligible.is_empty():
            eligible_parts.append(eligible)
    if not event_parts or not eligible_parts:
        raise ValueError("daily publications produced no eligible q95 events")
    return (
        pl.concat(event_parts, how="vertical_relaxed", rechunk=False).sort(
            [*KEYS, "upper_start_seconds"]
        ),
        pl.concat(eligible_parts, how="vertical_relaxed", rechunk=False).sort(
            KEYS
        ),
    )


def run_monthly_selector(
    *,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
    liquidity_path: Path = DEFAULT_LIQUIDITY_PATH,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: MonthlyProductSelectorConfig = MonthlyProductSelectorConfig(),
    completed_source_months: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build and publish the research tables."""

    config.validate()
    boundaries = pl.read_parquet(boundary_path)
    liquidity = pl.read_parquet(liquidity_path)
    events, eligible = _build_events_from_daily_root(
        boundaries,
        liquidity,
        daily_root,
        config=config,
    )
    metrics = build_monthly_product_metrics(
        events, eligible, boundaries, config=config
    )
    membership = build_monthly_membership(
        metrics,
        config=config,
        completed_source_months=completed_source_months,
    )
    daily = build_daily_allowlist(membership, liquidity, config=config)
    evaluation = evaluate_monthly_membership(membership, events)

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "dynamic_full_band_events.parquet": events,
        "eligible_product_days.parquet": eligible,
        "monthly_product_metrics.parquet": metrics,
        "monthly_membership.parquet": membership,
        "daily_allowlist.parquet": daily,
        "monthly_oos_diagnostic.parquet": evaluation,
    }
    for name, frame in artifacts.items():
        frame.write_parquet(output_root / name)
    membership.filter(pl.col("monthly_primary_selected")).write_csv(
        output_root / "monthly_entry_membership.csv"
    )
    manifest_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "source_month",
        "effective_month",
        "source_month_last_date",
        "source_asof_date",
        "liquidity_gate_status",
        "prior_daily_same_day_lower_lcb80",
        "prior_wait_upper_bound_p90_seconds",
        "selector_version",
    ]
    daily.filter(pl.col("new_entry_allowed")).select(
        manifest_columns
    ).write_csv(output_root / "daily_entry_manifest.csv")
    published_source_months = (
        membership["source_month"].unique().sort().to_list()
    )
    marker = {
        "selector_version": config.selector_version,
        "config": asdict(config),
        "boundary_path": str(boundary_path),
        "liquidity_path": str(liquidity_path),
        "daily_root": str(daily_root),
        "event_rows": events.height,
        "monthly_metric_rows": metrics.height,
        "monthly_membership_rows": membership.height,
        "daily_allowlist_rows": daily.height,
        "published_source_months": published_source_months,
        "latest_observed_source_month": metrics["source_month"].max(),
        "latest_observed_month_published": (
            metrics["source_month"].max() in published_source_months
        ),
        "retrospective_multi_window_selection": False,
        "selection_contains_effective_month_outcome": False,
        "dynamic_full_band_is_execution_fill": False,
        "production_universe_approved": False,
    }
    (output_root / "complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return marker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--completed-source-months",
        help=(
            "comma-separated YYYYMM months verified complete by an external "
            "exchange calendar; otherwise completion requires seeing the next month"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    marker = run_monthly_selector(
        boundary_path=args.boundaries,
        liquidity_path=args.liquidity,
        daily_root=args.daily_root,
        output_root=args.output_root,
        completed_source_months=(
            tuple(value.strip() for value in args.completed_source_months.split(","))
            if args.completed_source_months
            else None
        ),
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
