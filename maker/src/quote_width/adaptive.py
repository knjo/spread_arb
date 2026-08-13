"""Product-adaptive empirical basis boundaries and latent reversion tables.

Every target-day boundary is estimated from data known before that day.  The
tables describe latent 1-second price paths only; they are not maker-fill or EV
tables.  Fixed-BP grids deliberately do not enter this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import polars as pl

from ..common.paths import DEFAULT_OUTPUT_ROOT, MAKER_ROOT
from .cycle import (
    DEFAULT_HORIZONS_SECONDS,
    POLICY_KEYS,
    build_latent_cycles,
    summarize_cycles_by_day_symbol,
)


BOUNDARY_QUANTILES = (50, 80, 95)
LATENT_CANDIDATE_BOUNDARY_QUANTILES = (50, 80)
ANCHOR_MODES = ("frozen_entry", "dynamic")
EXIT_TARGETS = ("center", "adaptive_lower")
OBSERVATION_DELAYS_SECONDS = (1, 30)
MIN_PRIOR_ELIGIBLE_RATE = 0.80
MIN_PRIOR_EXCURSIONS_PER_SIDE = 30
MIN_DISPLAY_PAIR_DAYS = 4
MIN_DISPLAY_STARTED = 100
MIN_DISPLAY_HITS = 30
MIN_REVERSION_ENTRIES = 50
MIN_REVERSION_COMPLETES = 30
MAX_REVERSION_CENSOR_RATE = 0.10


@dataclass(frozen=True)
class AdaptiveBoundaryResult:
    parameter_snapshot: pl.DataFrame
    boundary_validation: pl.DataFrame
    boundary_probability: pl.DataFrame
    boundary_probability_by_state: pl.DataFrame
    cycles: pl.DataFrame
    reversion_by_day_symbol: pl.DataFrame
    reversion_probability: pl.DataFrame
    reversion_probability_by_state: pl.DataFrame
    latent_policy_frontier: pl.DataFrame


def load_daily_parameters(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "prior_date": pl.String,
        },
        infer_schema_length=10_000,
    )


def build_adaptive_boundaries(
    parameters: pl.DataFrame,
    quantiles: Iterable[int] = BOUNDARY_QUANTILES,
) -> pl.DataFrame:
    """Create asymmetric D-1 empirical upper/lower bounds for target day D."""
    frames: list[pl.DataFrame] = []
    base_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "prior_date",
        "calendar_gap_days",
        "target_dte_days",
        "target_tick_bp_bucket",
        "prior_future_spread_bucket",
        "target_ref_future_ask_tick_bp",
        "target_ref_spot_bid_tick_bp",
        "prior_spot_spread_ticks_p50",
        "prior_fut_spread_ticks_p50",
        "prior_tt_band_width_bp_p50",
        "prior_eligible_rate",
        "prior_completed_positive",
        "prior_completed_negative",
        "prior_censored_positive",
        "prior_censored_negative",
    ]
    missing = sorted(set(base_columns) - set(parameters.columns))
    if missing:
        raise ValueError(f"adaptive boundary parameters missing columns: {missing}")
    key_columns = ["Date", "ValueCode", "QuoteCode"]
    if parameters.select(key_columns).unique().height != parameters.height:
        raise ValueError("adaptive boundary parameters contain duplicate target keys")
    parsed_dates = parameters.select(
        pl.col("Date").cast(pl.String).alias("_date_text"),
        pl.col("prior_date").cast(pl.String).alias("_prior_text"),
    ).with_columns(
        pl.col("_date_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("_date"),
        pl.col("_prior_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("_prior"),
    )
    if parsed_dates.filter(
        pl.col("_date").is_null()
        | pl.col("_prior").is_null()
        | (pl.col("_date_text").str.len_chars() != 8)
        | (pl.col("_prior_text").str.len_chars() != 8)
    ).height:
        raise ValueError("adaptive boundary dates must be non-null valid YYYYMMDD")
    if parsed_dates.filter(pl.col("_prior") >= pl.col("_date")).height:
        raise ValueError("adaptive boundary prior_date must be before target Date")

    for quantile in quantiles:
        upper_column = f"prior_p{quantile}_amplitude_bp_positive"
        lower_column = f"prior_p{quantile}_amplitude_bp_negative"
        if upper_column not in parameters.columns or lower_column not in parameters.columns:
            raise ValueError(f"parameters missing p{quantile} asymmetric bounds")
        frames.append(
            parameters.select(
                *base_columns,
                pl.lit(int(quantile)).alias("boundary_quantile"),
                pl.when(
                    pl.lit(int(quantile)).is_in(
                        list(LATENT_CANDIDATE_BOUNDARY_QUANTILES)
                    )
                )
                .then(pl.lit("latent_prior_candidate"))
                .otherwise(pl.lit("tail_diagnostic"))
                .alias("boundary_role"),
                pl.col(upper_column).alias("upper_distance_bp"),
                pl.col(lower_column).alias("lower_distance_bp"),
            )
        )

    return (
        pl.concat(frames, how="vertical")
        .with_columns(
            (
                (pl.col("prior_eligible_rate") >= MIN_PRIOR_ELIGIBLE_RATE)
                & (
                    pl.col("prior_completed_positive")
                    >= MIN_PRIOR_EXCURSIONS_PER_SIDE
                )
                & (
                    pl.col("prior_completed_negative")
                    >= MIN_PRIOR_EXCURSIONS_PER_SIDE
                )
                & pl.col("upper_distance_bp").is_finite()
                & (pl.col("upper_distance_bp") > 0)
                & pl.col("lower_distance_bp").is_finite()
                & (pl.col("lower_distance_bp") > 0)
            )
            .fill_null(False)
            .alias("adaptive_parameter_valid"),
            (
                pl.col("upper_distance_bp")
                / pl.col("target_ref_future_ask_tick_bp")
            ).alias("upper_distance_future_ticks"),
            (
                pl.col("upper_distance_bp")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("upper_distance_spot_ticks"),
            (
                pl.col("lower_distance_bp")
                / pl.col("target_ref_future_ask_tick_bp")
            ).alias("lower_distance_future_ticks"),
            (
                pl.col("lower_distance_bp")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("lower_distance_spot_ticks"),
            (
                pl.col("upper_distance_bp")
                / pl.col("prior_tt_band_width_bp_p50")
            ).alias("upper_distance_tt_bands"),
            (
                pl.col("lower_distance_bp")
                / pl.col("prior_tt_band_width_bp_p50")
            ).alias("lower_distance_tt_bands"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(True).alias("execution_safe_snapshot"),
            pl.lit(False).alias("contains_target_day_outcome"),
            pl.lit("adaptive_empirical_v1").alias("parameter_version"),
            pl.col("prior_date").alias("source_asof_date"),
            pl.lit("latent_price_path_prior").alias("probability_layer"),
        )
        .sort(["Date", "ValueCode", "boundary_quantile"])
    )


def _boundary_supply_for_side(
    boundaries: pl.DataFrame,
    excursions: pl.DataFrame,
    *,
    side: str,
    distance_column: str,
    prefix: str,
) -> pl.DataFrame:
    keys = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    joined = excursions.filter(pl.col("side") == side).join(
        boundaries.filter(
            pl.col("adaptive_parameter_valid")
            & pl.col(distance_column).is_finite()
            & (pl.col(distance_column) > 0)
        ).select(*keys, distance_column),
        on=["Date", "ValueCode", "QuoteCode"],
        how="inner",
        validate="m:m",
    )
    return (
        joined.with_columns(
            (pl.col("amplitude_bp") >= pl.col(distance_column)).alias("_hit"),
            (
                pl.col("completed")
                & (pl.col("amplitude_bp") < pl.col(distance_column))
            ).alias("_known_miss"),
            (
                (~pl.col("completed"))
                & (pl.col("amplitude_bp") < pl.col(distance_column))
            ).alias("_unknown"),
        )
        .group_by(keys)
        .agg(
            pl.len().alias(f"{prefix}_started"),
            pl.col("_hit").sum().alias(f"{prefix}_hits"),
            pl.col("_known_miss").sum().alias(f"{prefix}_known_misses"),
            pl.col("_unknown").sum().alias(f"{prefix}_unknown"),
        )
        .with_columns(
            (
                pl.col(f"{prefix}_hits") / pl.col(f"{prefix}_started")
            ).alias(f"p_{prefix}_reach_all_started_lower_bound"),
            pl.when(
                (
                    pl.col(f"{prefix}_hits")
                    + pl.col(f"{prefix}_known_misses")
                )
                > 0
            )
            .then(
                pl.col(f"{prefix}_hits")
                / (
                    pl.col(f"{prefix}_hits")
                    + pl.col(f"{prefix}_known_misses")
                )
            )
            .otherwise(None)
            .alias(f"p_{prefix}_reach_known_case_conditional"),
        )
    )


def add_boundary_supply(
    boundaries: pl.DataFrame,
    target_excursions: pl.DataFrame,
) -> pl.DataFrame:
    """Attach next-day center-to-boundary supply with explicit censor unknowns."""
    keys = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    upper = _boundary_supply_for_side(
        boundaries,
        target_excursions,
        side="positive",
        distance_column="upper_distance_bp",
        prefix="upper",
    )
    lower = _boundary_supply_for_side(
        boundaries,
        target_excursions,
        side="negative",
        distance_column="lower_distance_bp",
        prefix="lower",
    )
    count_columns = [
        f"{prefix}_{suffix}"
        for prefix in ("upper", "lower")
        for suffix in ("started", "hits", "known_misses", "unknown")
    ]
    return (
        boundaries.join(upper, on=keys, how="left", validate="1:1")
        .join(lower, on=keys, how="left", validate="1:1")
        .with_columns(
            *[pl.col(column).fill_null(0) for column in count_columns],
            pl.lit(False).alias("execution_safe_snapshot"),
            pl.lit(True).alias("contains_target_day_outcome"),
        )
        .sort(keys)
    )


def _summarize_boundary_groups(
    daily: pl.DataFrame,
    group_keys: list[str],
) -> pl.DataFrame:
    valid = pl.col("adaptive_parameter_valid")
    result = daily.group_by(group_keys).agg(
        pl.len().alias("available_pair_days"),
        valid.sum().alias("valid_pair_days"),
        pl.col("Date").filter(valid).n_unique().alias("valid_dates"),
        pl.col("upper_distance_bp").filter(valid).median().alias("upper_bp_p50"),
        pl.col("lower_distance_bp").filter(valid).median().alias("lower_bp_p50"),
        pl.col("upper_distance_future_ticks")
        .filter(valid)
        .median()
        .alias("upper_future_ticks_p50"),
        pl.col("upper_distance_spot_ticks")
        .filter(valid)
        .median()
        .alias("upper_spot_ticks_p50"),
        pl.col("lower_distance_future_ticks")
        .filter(valid)
        .median()
        .alias("lower_future_ticks_p50"),
        pl.col("lower_distance_spot_ticks")
        .filter(valid)
        .median()
        .alias("lower_spot_ticks_p50"),
        pl.col("prior_spot_spread_ticks_p50")
        .filter(valid)
        .median()
        .alias("spot_spread_ticks_p50"),
        pl.col("prior_fut_spread_ticks_p50")
        .filter(valid)
        .median()
        .alias("future_spread_ticks_p50"),
        pl.col("prior_tt_band_width_bp_p50")
        .filter(valid)
        .median()
        .alias("tt_band_bp_p50"),
        *[
            pl.col(f"{prefix}_{suffix}")
            .filter(valid)
            .sum()
            .alias(f"{prefix}_{suffix}")
            for prefix in ("upper", "lower")
            for suffix in ("started", "hits", "known_misses", "unknown")
        ],
        *[
            pl.col(f"p_{prefix}_reach_all_started_lower_bound")
            .filter(valid)
            .median()
            .alias(f"pair_median_p_{prefix}_reach_all_started_lower_bound")
            for prefix in ("upper", "lower")
        ],
    )
    return result.with_columns(
        *[
            pl.when(pl.col(f"{prefix}_started") > 0)
            .then(pl.col(f"{prefix}_hits") / pl.col(f"{prefix}_started"))
            .otherwise(None)
            .alias(f"event_weighted_p_{prefix}_reach_all_started_lower_bound")
            for prefix in ("upper", "lower")
        ],
        (
            (pl.col("valid_pair_days") >= MIN_DISPLAY_PAIR_DAYS)
            & (pl.col("upper_started") >= MIN_DISPLAY_STARTED)
            & (pl.col("lower_started") >= MIN_DISPLAY_STARTED)
            & (pl.col("upper_hits") >= MIN_DISPLAY_HITS)
            & (pl.col("lower_hits") >= MIN_DISPLAY_HITS)
        ).alias("display_support_valid"),
        pl.lit(False).alias("actionable_execution"),
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )


def summarize_boundary_probability(
    daily: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    product = _summarize_boundary_groups(
        daily,
        ["ValueCode", "boundary_quantile", "boundary_role"],
    ).sort(["ValueCode", "boundary_quantile"])
    state_keys = [
        "ValueCode",
        "target_tick_bp_bucket",
        "prior_future_spread_bucket",
        "boundary_quantile",
        "boundary_role",
    ]
    state = _summarize_boundary_groups(daily, state_keys).join(
        product.select(
            "ValueCode",
            "boundary_quantile",
            pl.col("display_support_valid").alias("product_support_valid"),
            pl.col("event_weighted_p_upper_reach_all_started_lower_bound").alias(
                "product_p_upper_reach_all_started_lower_bound"
            ),
            pl.col("event_weighted_p_lower_reach_all_started_lower_bound").alias(
                "product_p_lower_reach_all_started_lower_bound"
            ),
        ),
        on=["ValueCode", "boundary_quantile"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("display_support_valid"))
        .then(pl.lit("product_state"))
        .when(pl.col("product_support_valid"))
        .then(pl.lit("product_all"))
        .otherwise(pl.lit("peer_or_global_required"))
        .alias("fallback_level"),
        pl.when(pl.col("display_support_valid"))
        .then(pl.col("event_weighted_p_upper_reach_all_started_lower_bound"))
        .when(pl.col("product_support_valid"))
        .then(pl.col("product_p_upper_reach_all_started_lower_bound"))
        .otherwise(None)
        .alias("reported_p_upper_reach_all_started_lower_bound"),
        pl.when(pl.col("display_support_valid"))
        .then(pl.col("event_weighted_p_lower_reach_all_started_lower_bound"))
        .when(pl.col("product_support_valid"))
        .then(pl.col("product_p_lower_reach_all_started_lower_bound"))
        .otherwise(None)
        .alias("reported_p_lower_reach_all_started_lower_bound"),
    ).sort(state_keys)
    return product, state


def build_adaptive_cycle_universe(boundaries: pl.DataFrame) -> pl.DataFrame:
    """Build asymmetric +W+ to center/-W- policies without a fixed-BP grid."""
    base = boundaries.filter(pl.col("adaptive_parameter_valid"))
    dimensions = pl.DataFrame(
        [
            {
                "sample": "base",
                "anchor_mode": anchor_mode,
                "exit_target": exit_target,
                "exit_delay_seconds": int(delay),
                "max_horizon_seconds": 0,
            }
            for anchor_mode in ANCHOR_MODES
            for exit_target in EXIT_TARGETS
            for delay in OBSERVATION_DELAYS_SECONDS
        ]
    )
    return (
        base.join(dimensions, how="cross")
        .with_columns(
            pl.lit("adaptive_product_prior").alias("width_family"),
            pl.format("adaptive_q{}", pl.col("boundary_quantile")).alias(
                "width_policy"
            ),
            pl.col("upper_distance_bp").alias("candidate_width_bp"),
            pl.when(pl.col("exit_target") == "center")
            .then(pl.lit(0.0))
            .otherwise(pl.col("lower_distance_bp"))
            .alias("exit_width_bp"),
        )
        .with_columns(
            (pl.col("exit_width_bp") / pl.col("candidate_width_bp")).alias(
                "exit_width_ratio"
            )
        )
        .sort(POLICY_KEYS)
    )


def _summarize_reversion_groups(
    by_day: pl.DataFrame,
    group_keys: list[str],
    horizons_seconds: Iterable[int],
) -> pl.DataFrame:
    horizons = tuple(int(value) for value in horizons_seconds)
    aggregations: list[pl.Expr] = [
        pl.len().alias("pair_days"),
        pl.col("Date").n_unique().alias("dates"),
        (pl.col("latent_entries") > 0).sum().alias("pair_days_with_entries"),
        pl.col("candidate_width_bp").median().alias("upper_bp_p50"),
        pl.col("exit_width_bp").median().alias("exit_distance_bp_p50"),
        pl.col("lower_distance_bp").median().alias("empirical_lower_bp_p50"),
        pl.col("upper_distance_future_ticks")
        .median()
        .alias("upper_future_ticks_p50"),
        pl.col("lower_distance_future_ticks")
        .median()
        .alias("lower_future_ticks_p50"),
        pl.col("latent_entries").sum().alias("latent_entries"),
        pl.col("completed_cycles").sum().alias("completed_cycles"),
        pl.col("censored_cycles").sum().alias("censored_cycles"),
        pl.col("latent_entries").quantile(0.25, interpolation="nearest").alias(
            "entries_per_day_q25"
        ),
        pl.col("latent_entries").median().alias("entries_per_day_p50"),
        pl.col("latent_entries").quantile(0.75, interpolation="nearest").alias(
            "entries_per_day_q75"
        ),
        pl.col("completed_holding_seconds_p50")
        .median()
        .alias("holding_seconds_p50"),
        pl.col("completed_holding_seconds_p80")
        .median()
        .alias("holding_seconds_p80"),
        pl.col("basis_capture_bp_p20").median().alias("basis_capture_bp_p20"),
        pl.col("basis_capture_bp_p50").median().alias("basis_capture_bp_p50"),
        pl.col("anchor_drift_bp_p50").median().alias("anchor_drift_bp_p50"),
        pl.col("anchor_only_exits").sum().alias("anchor_only_exits"),
        pl.col("early_exit_touches").sum().alias("early_exit_touches"),
    ]
    for horizon in horizons:
        aggregations.extend(
            [
                (pl.col(f"n_observed_{horizon}s") > 0)
                .sum()
                .alias(f"pair_days_observed_{horizon}s"),
                pl.col(f"n_observed_{horizon}s")
                .sum()
                .alias(f"n_observed_{horizon}s"),
                pl.col(f"n_complete_within_{horizon}s")
                .sum()
                .alias(f"n_exit_{horizon}s"),
                pl.col(f"p_complete_within_{horizon}s")
                .median()
                .alias(f"pair_median_p_exit_{horizon}s"),
            ]
        )
    result = by_day.group_by(group_keys).agg(*aggregations)
    rates: list[pl.Expr] = [
        pl.when(pl.col("latent_entries") > 0)
        .then(pl.col("completed_cycles") / pl.col("latent_entries"))
        .otherwise(None)
        .alias("p_exit_by_session_cutoff_lower_bound"),
        pl.when(pl.col("latent_entries") > 0)
        .then(pl.col("censored_cycles") / pl.col("latent_entries"))
        .otherwise(None)
        .alias("censor_rate"),
        pl.when(pl.col("completed_cycles") > 0)
        .then(pl.col("anchor_only_exits") / pl.col("completed_cycles"))
        .otherwise(None)
        .alias("anchor_only_exit_rate"),
        pl.when(pl.col("latent_entries") > 0)
        .then(pl.col("early_exit_touches") / pl.col("latent_entries"))
        .otherwise(None)
        .alias("early_exit_touch_rate"),
    ]
    for horizon in horizons:
        rates.append(
            pl.when(pl.col(f"n_observed_{horizon}s") > 0)
            .then(
                pl.col(f"n_exit_{horizon}s")
                / pl.col(f"n_observed_{horizon}s")
            )
            .otherwise(None)
            .alias(f"event_weighted_p_exit_{horizon}s")
        )
    return result.with_columns(
        *rates,
        (
            (pl.col("pair_days") >= MIN_DISPLAY_PAIR_DAYS)
            & (pl.col("latent_entries") >= MIN_REVERSION_ENTRIES)
            & (pl.col("completed_cycles") >= MIN_REVERSION_COMPLETES)
            & (
                pl.col("censored_cycles") / pl.col("latent_entries")
                <= MAX_REVERSION_CENSOR_RATE
            )
        ).fill_null(False).alias("display_support_valid"),
        pl.lit(False).alias("actionable_execution"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )


def summarize_adaptive_reversion(
    by_day: pl.DataFrame,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base_keys = [
        "ValueCode",
        "boundary_quantile",
        "boundary_role",
        "anchor_mode",
        "exit_target",
        "exit_delay_seconds",
    ]
    product = _summarize_reversion_groups(
        by_day,
        base_keys,
        horizons_seconds,
    ).sort(base_keys)
    state_keys = [
        "ValueCode",
        "target_tick_bp_bucket",
        "prior_future_spread_bucket",
        "boundary_quantile",
        "boundary_role",
        "anchor_mode",
        "exit_target",
        "exit_delay_seconds",
    ]
    state = _summarize_reversion_groups(
        by_day,
        state_keys,
        horizons_seconds,
    ).join(
        product.select(
            *base_keys,
            pl.col("display_support_valid").alias("product_support_valid"),
            pl.col("p_exit_by_session_cutoff_lower_bound").alias(
                "product_p_exit_by_session_cutoff_lower_bound"
            ),
        ),
        on=base_keys,
        how="left",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("display_support_valid"))
        .then(pl.lit("product_state"))
        .when(pl.col("product_support_valid"))
        .then(pl.lit("product_all"))
        .otherwise(pl.lit("peer_or_global_required"))
        .alias("fallback_level"),
        pl.when(pl.col("display_support_valid"))
        .then(pl.col("p_exit_by_session_cutoff_lower_bound"))
        .when(pl.col("product_support_valid"))
        .then(pl.col("product_p_exit_by_session_cutoff_lower_bound"))
        .otherwise(None)
        .alias("reported_p_exit_by_session_cutoff_lower_bound"),
    ).sort(state_keys)
    return product, state


def build_latent_policy_frontier(
    boundary_probability: pl.DataFrame,
    reversion_probability: pl.DataFrame,
) -> pl.DataFrame:
    """Join entry-supply and conditional-exit priors without claiming EV.

    This is the compact handoff from the one-second path study.  It deliberately
    contains no fill, hedge, fee, tax, cash-flow, or executable-EV estimate.
    """
    keys = ["ValueCode", "boundary_quantile", "boundary_role"]
    supply_columns = [
        *keys,
        "valid_pair_days",
        "valid_dates",
        "upper_spot_ticks_p50",
        "lower_spot_ticks_p50",
        "spot_spread_ticks_p50",
        "future_spread_ticks_p50",
        "tt_band_bp_p50",
        "upper_started",
        "upper_hits",
        "upper_unknown",
        "pair_median_p_upper_reach_all_started_lower_bound",
        "event_weighted_p_upper_reach_all_started_lower_bound",
        "display_support_valid",
    ]
    return (
        reversion_probability.join(
            boundary_probability.select(*supply_columns).rename(
                {"display_support_valid": "entry_supply_support_valid"}
            ),
            on=keys,
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("display_support_valid").alias(
                "conditional_exit_support_valid"
            ),
            pl.lit("latent_1s_price_path").alias("probability_semantics"),
            pl.lit(False).alias("maker_fill_included"),
            pl.lit(False).alias("hedge_50ms_included"),
            pl.lit(False).alias("fees_and_tax_included"),
            pl.lit(False).alias("ev_ready"),
            pl.lit(False).alias("execution_safe_snapshot"),
            pl.lit(True).alias("contains_target_day_outcome"),
            pl.lit("WP02 fill -> WP03 hedge -> WP04 cashflow").alias(
                "ev_blocked_by"
            ),
        )
        .drop("display_support_valid")
        .sort(
            [
                "ValueCode",
                "boundary_quantile",
                "anchor_mode",
                "exit_target",
                "exit_delay_seconds",
            ]
        )
    )


def run_adaptive_boundary_study(
    panel_path: Path,
    parameter_path: Path,
    excursion_path: Path,
    output_dir: Path,
) -> AdaptiveBoundaryResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(panel_path)
    parameters = load_daily_parameters(parameter_path)
    excursions = pl.read_parquet(excursion_path)
    boundaries = build_adaptive_boundaries(parameters)
    daily = add_boundary_supply(boundaries, excursions)
    boundary_probability, boundary_state = summarize_boundary_probability(daily)
    universe = build_adaptive_cycle_universe(boundaries)
    cycles = build_latent_cycles(panel, universe).with_columns(
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )
    by_day = summarize_cycles_by_day_symbol(cycles, universe).with_columns(
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )
    reversion, reversion_state = summarize_adaptive_reversion(by_day)
    frontier = build_latent_policy_frontier(boundary_probability, reversion)

    boundaries.write_csv(
        output_dir / "adaptive_parameter_snapshot_by_day_symbol.csv"
    )
    daily.write_csv(
        output_dir / "adaptive_boundary_validation_by_day_symbol.csv"
    )
    boundary_probability.write_csv(output_dir / "boundary_probability.csv")
    boundary_state.write_csv(output_dir / "boundary_probability_by_state.csv")
    cycles.write_parquet(output_dir / "adaptive_latent_cycles.parquet")
    by_day.write_csv(output_dir / "adaptive_reversion_by_day_symbol.csv")
    reversion.write_csv(output_dir / "adaptive_reversion_probability.csv")
    reversion_state.write_csv(
        output_dir / "adaptive_reversion_probability_by_state.csv"
    )
    frontier.write_csv(output_dir / "latent_policy_frontier.csv")
    config = {
        "boundary_source": "D-1 same target contract completed excursions",
        "boundary_quantiles": list(BOUNDARY_QUANTILES),
        "latent_candidate_boundary_quantiles": list(
            LATENT_CANDIDATE_BOUNDARY_QUANTILES
        ),
        "upper_definition": "positive residual excursion quantile",
        "lower_definition": "negative residual excursion quantile",
        "fixed_bp_grid_role": "excluded; diagnostic only in cycle.py",
        "anchor_modes": list(ANCHOR_MODES),
        "exit_targets": list(EXIT_TARGETS),
        "observation_delays_seconds": list(OBSERVATION_DELAYS_SECONDS),
        "horizons_seconds": list(DEFAULT_HORIZONS_SECONDS),
        "actionable_execution": False,
        "ev_ready": False,
        "frontier_semantics": "entry supply plus conditional latent exit; not EV",
        "execution_safe_snapshot": "adaptive_parameter_snapshot_by_day_symbol.csv",
        "retrospective_outputs": "every output except adaptive_parameter_snapshot_by_day_symbol.csv and config.json",
        "next_layer": "raw rounded-target maker episode replay",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AdaptiveBoundaryResult(
        parameter_snapshot=boundaries,
        boundary_validation=daily,
        boundary_probability=boundary_probability,
        boundary_probability_by_state=boundary_state,
        cycles=cycles,
        reversion_by_day_symbol=by_day,
        reversion_probability=reversion,
        reversion_probability_by_state=reversion_state,
        latent_policy_frontier=frontier,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build product-adaptive upper/lower latent probability tables."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fair_anchor_panel.parquet",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width" / "daily_product_parameters.csv",
    )
    parser.add_argument(
        "--excursions",
        type=Path,
        default=MAKER_ROOT
        / "data"
        / "quote_width"
        / "target_zero_crossing_excursions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width" / "adaptive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_adaptive_boundary_study(
        args.panel,
        args.parameters,
        args.excursions,
        args.output_dir,
    )
    print(result.boundary_probability)
    print(result.reversion_probability)


if __name__ == "__main__":
    main()
