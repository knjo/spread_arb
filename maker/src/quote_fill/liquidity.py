"""Causal product/route liquidity screen before expensive raw replay.

Wide maker spreads are not automatically bad: they may improve passive edge.
Wide spreads on the *taker hedge* leg, stale books, or insufficient unit depth
are more direct execution problems.  The screen therefore keeps separate
diagnostic flags and only applies a transparent, versioned pre-replay gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import DEFAULT_DAILY_ROOT, completed_artifact_paths
from ..quote_width.rolling import load_rolling_boundary_snapshots
from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


DEFAULT_OUTPUT_DIR = MAKER_ROOT / "data" / "walkforward" / "liquidity"
ENTRY_ROUTES = ("future_ask_spot_taker", "spot_bid_future_taker")
LIQUIDITY_PUBLICATION_SCHEMA_VERSION = (
    "rolling_liquidity_publication_v2_price_ladder_lineage"
)


@dataclass(frozen=True)
class LiquidityScreenConfig:
    lookback_sessions: int = 60
    recent_sessions: int = 20
    min_history_sessions: int = 40
    min_recent_sessions: int = 15
    eligible_median_warning_threshold: float = 0.90
    eligible_q10_warning_threshold: float = 0.70
    min_fresh_1000ms_rate_median: float = 0.01
    min_fresh_5000ms_rate_median: float = 0.05
    min_unit_depth_rate_median: float = 0.50
    min_maker_changed_second_rate_median: float = 0.01
    wide_hedge_spread_p50_ticks: float = 2.0
    wide_hedge_spread_p95_ticks: float = 4.0
    pseudo_validation_start_date: str = "20260701"
    min_stability_sessions: int = 20
    stable_core_rate: float = 0.80
    price_ladder_version: str = PRICE_LADDER_VERSION
    future_one_dollar_tick_effective_date: str = (
        FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    )
    # Keep the screen-policy version stable: this change hardens publication
    # lineage but does not alter cohort selection semantics.  The independent
    # publication schema below rejects every pre-marker/v1 cache.
    screen_version: str = "rolling_liquidity_screen_v6_price_ladder"

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.recent_sessions <= self.lookback_sessions:
            raise ValueError("recent_sessions must be in [1, lookback_sessions]")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError("invalid minimum history sessions")
        if not 1 <= self.min_recent_sessions <= self.recent_sessions:
            raise ValueError("invalid minimum recent sessions")
        if self.min_stability_sessions <= 0:
            raise ValueError("min_stability_sessions must be positive")
        if not 0 <= self.stable_core_rate <= 1:
            raise ValueError("stable_core_rate must be in [0, 1]")
        if self.price_ladder_version != PRICE_LADDER_VERSION:
            raise ValueError(
                "price_ladder_version must match the implemented liquidity "
                "tick geometry"
            )
        if (
            self.future_one_dollar_tick_effective_date
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ):
            raise ValueError(
                "future_one_dollar_tick_effective_date must match the "
                "implemented liquidity tick geometry"
            )
        for value in (
            self.eligible_median_warning_threshold,
            self.eligible_q10_warning_threshold,
            self.min_fresh_1000ms_rate_median,
            self.min_fresh_5000ms_rate_median,
            self.min_unit_depth_rate_median,
            self.min_maker_changed_second_rate_median,
        ):
            if not 0 <= value <= 1:
                raise ValueError("liquidity rate thresholds must be in [0, 1]")


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def summarize_daily_liquidity(causal_panel: pl.DataFrame) -> pl.DataFrame:
    """Summarize equal-weight product/day liquidity from a one-second grid."""

    _require(
        causal_panel,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "seconds_from_open",
            "spot_bid",
            "spot_ask",
            "fut_bid",
            "fut_ask",
            "fut_exec_bid",
            "fut_exec_ask",
            "spot_ask_lots",
            "fut_exec_bid_lots",
            "contract_size",
            "spot_sequence",
            "fut_sequence",
            "eligible_base",
            "eligible_1000ms",
            "eligible_5000ms",
            "basis_sell_taker_bp",
            "basis_buy_taker_bp",
        },
        "causal panel",
    )
    ordered = causal_panel.filter(pl.col("seconds_from_open") >= 300).sort(
        ["Date", "ValueCode", "seconds_from_open"]
    )
    spot_previous = pl.col("spot_sequence").shift(1).over(["Date", "ValueCode"])
    future_previous = pl.col("fut_sequence").shift(1).over(["Date", "ValueCode"])
    sample = ordered.with_columns(
        (
            _price_to_tick_index(pl.col("spot_ask"), market="spot")
            - _price_to_tick_index(pl.col("spot_bid"), market="spot")
        ).round(0).alias("_spot_spread_ticks"),
        (
            _price_to_tick_index(pl.col("fut_exec_ask"), market="future")
            - _price_to_tick_index(pl.col("fut_exec_bid"), market="future")
        ).round(0).alias("_fut_spread_ticks"),
        (
            pl.col("basis_buy_taker_bp") - pl.col("basis_sell_taker_bp")
        ).alias("_tt_band_bp"),
        (
            pl.col("spot_sequence").is_not_null()
            & (spot_previous.is_null() | (pl.col("spot_sequence") != spot_previous))
        ).fill_null(False).alias("_spot_state_changed"),
        (
            pl.col("fut_sequence").is_not_null()
            & (future_previous.is_null() | (pl.col("fut_sequence") != future_previous))
        ).fill_null(False).alias("_fut_state_changed"),
        (
            (pl.col("spot_ask_lots") * 1000 / pl.col("contract_size")) >= 1
        ).alias("_spot_buy_one_unit_depth"),
    )
    eligible = pl.col("eligible_base").fill_null(False)
    return sample.group_by(["Date", "ValueCode", "QuoteCode"]).agg(
        pl.len().alias("grid_rows"),
        eligible.sum().alias("eligible_rows"),
        pl.col("eligible_1000ms").fill_null(False).sum().alias(
            "fresh_1000ms_rows"
        ),
        pl.col("eligible_5000ms").fill_null(False).sum().alias(
            "fresh_5000ms_rows"
        ),
        pl.col("_spot_spread_ticks")
        .filter(eligible)
        .median()
        .alias("spot_spread_ticks_p50"),
        pl.col("_spot_spread_ticks")
        .filter(eligible)
        .quantile(0.95, interpolation="nearest")
        .alias("spot_spread_ticks_p95"),
        pl.col("_fut_spread_ticks")
        .filter(eligible)
        .median()
        .alias("fut_spread_ticks_p50"),
        pl.col("_fut_spread_ticks")
        .filter(eligible)
        .quantile(0.95, interpolation="nearest")
        .alias("fut_spread_ticks_p95"),
        (pl.col("_spot_spread_ticks") <= 1)
        .filter(eligible)
        .mean()
        .alias("p_spot_spread_le1tick"),
        (pl.col("_fut_spread_ticks") <= 2)
        .filter(eligible)
        .mean()
        .alias("p_fut_spread_le2ticks"),
        pl.col("_tt_band_bp").filter(eligible).median().alias("tt_band_bp_p50"),
        pl.col("_tt_band_bp")
        .filter(eligible)
        .quantile(0.95, interpolation="nearest")
        .alias("tt_band_bp_p95"),
        pl.col("_tt_band_bp")
        .filter(pl.col("eligible_1000ms").fill_null(False))
        .median()
        .alias("tt_band_fresh1000_bp_p50"),
        pl.col("_tt_band_bp")
        .filter(pl.col("eligible_5000ms").fill_null(False))
        .median()
        .alias("tt_band_fresh5000_bp_p50"),
        pl.col("_spot_buy_one_unit_depth")
        .filter(eligible)
        .mean()
        .alias("p_spot_buy_one_unit_l1_depth"),
        pl.col("_spot_buy_one_unit_depth")
        .filter(pl.col("eligible_1000ms").fill_null(False))
        .mean()
        .alias("p_spot_one_unit_given_fresh1000"),
        (pl.col("fut_exec_bid_lots") >= 1)
        .filter(eligible)
        .mean()
        .alias("p_fut_bid_depth_ge1contract"),
        (pl.col("fut_exec_bid_lots") >= 1)
        .filter(pl.col("eligible_1000ms").fill_null(False))
        .mean()
        .alias("p_fut_one_unit_given_fresh1000"),
        pl.col("_spot_state_changed").sum().alias("spot_changed_seconds"),
        pl.col("_fut_state_changed").sum().alias("fut_changed_seconds"),
    ).with_columns(
        (pl.col("eligible_rows") / pl.col("grid_rows")).alias(
            "eligible_grid_rate"
        ),
        (pl.col("fresh_1000ms_rows") / pl.col("grid_rows")).alias(
            "fresh_1000ms_rate"
        ),
        (pl.col("fresh_5000ms_rows") / pl.col("grid_rows")).alias(
            "fresh_5000ms_rate"
        ),
        (pl.col("spot_changed_seconds") / (4.25)).alias(
            "spot_changed_seconds_per_hour"
        ),
        (pl.col("fut_changed_seconds") / (4.25)).alias(
            "fut_changed_seconds_per_hour"
        ),
        (pl.col("spot_changed_seconds") / pl.col("grid_rows")).alias(
            "spot_changed_second_rate"
        ),
        (pl.col("fut_changed_seconds") / pl.col("grid_rows")).alias(
            "fut_changed_second_rate"
        ),
        pl.col("Date").alias("label_end_date"),
        pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version"),
        pl.lit(FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE).alias(
            "future_one_dollar_tick_effective_date"
        ),
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    ).sort(["Date", "ValueCode"])


def summarize_partitioned_daily_liquidity(
    daily_root: Path = DEFAULT_DAILY_ROOT,
) -> pl.DataFrame:
    paths = completed_artifact_paths(daily_root, "causal_fair.parquet")
    if not paths:
        raise FileNotFoundError(f"no causal fair partitions below {daily_root}")
    return pl.concat(
        [summarize_daily_liquidity(pl.read_parquet(path)) for path in paths],
        how="diagonal_relaxed",
    ).sort(["Date", "ValueCode"])


def _normalise_sessions(sessions: Iterable[str]) -> list[str]:
    result = sorted({str(value) for value in sessions})
    if not result:
        raise ValueError("session calendar must not be empty")
    parsed = pl.DataFrame({"Date": result}).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_date")
    )
    if parsed.filter(
        pl.col("_date").is_null() | (pl.col("Date").str.len_chars() != 8)
    ).height:
        raise ValueError("session dates must be valid YYYYMMDD")
    return result


def build_rolling_liquidity_screen(
    daily_liquidity: pl.DataFrame,
    rolling_boundaries: pl.DataFrame,
    sessions: Sequence[str],
    config: LiquidityScreenConfig = LiquidityScreenConfig(),
) -> pl.DataFrame:
    """Build route-specific pre-replay candidates with explicit reason flags."""

    config.validate()
    sessions = _normalise_sessions(sessions)
    _require(
        daily_liquidity,
        {
            "Date",
            "ValueCode",
            "eligible_grid_rate",
            "fresh_1000ms_rate",
            "fresh_5000ms_rate",
            "spot_spread_ticks_p50",
            "spot_spread_ticks_p95",
            "fut_spread_ticks_p50",
            "fut_spread_ticks_p95",
            "tt_band_bp_p50",
            "tt_band_bp_p95",
            "tt_band_fresh1000_bp_p50",
            "tt_band_fresh5000_bp_p50",
            "fresh_1000ms_rows",
            "fresh_5000ms_rows",
            "p_spot_buy_one_unit_l1_depth",
            "p_spot_one_unit_given_fresh1000",
            "p_fut_bid_depth_ge1contract",
            "p_fut_one_unit_given_fresh1000",
            "spot_changed_seconds_per_hour",
            "fut_changed_seconds_per_hour",
            "spot_changed_second_rate",
            "fut_changed_second_rate",
        },
        "daily liquidity",
    )
    _require(
        rolling_boundaries,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "boundary_quantile",
            "upper_distance_bp",
            "lower_distance_bp",
            "adaptive_parameter_valid",
            "source_asof_date",
            "contains_target_day_outcome",
            "boundary_role",
            "execution_safe_snapshot",
            "parameter_version",
        },
        "rolling boundaries",
    )
    safe_boundary_columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "boundary_role",
        "upper_distance_bp",
        "lower_distance_bp",
        "adaptive_parameter_valid",
        "source_asof_date",
        "parameter_version",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    ]
    candidate_boundaries = rolling_boundaries.select(
        safe_boundary_columns
    ).filter(
        pl.col("boundary_role").is_in(
            ["rolling_latent_candidate", "adaptive_latent_candidate"]
        )
        & pl.col("source_asof_date").is_not_null()
    )
    if candidate_boundaries.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
        | (~pl.col("execution_safe_snapshot").fill_null(False))
        | pl.col("parameter_version").is_null()
    ).height:
        raise ValueError("rolling boundary input is not execution-safe")
    boundary_key = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    if candidate_boundaries.select(boundary_key).n_unique() != candidate_boundaries.height:
        raise ValueError("rolling boundary input contains duplicate keys")
    dates = candidate_boundaries.select(
        pl.col("Date").cast(pl.String).alias("_target_text"),
        pl.col("source_asof_date").cast(pl.String).alias("_source_text"),
    ).with_columns(
        pl.col("_target_text").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_target"),
        pl.col("_source_text").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_source"),
    )
    if dates.filter(
        pl.col("_target").is_null()
        | pl.col("_source").is_null()
        | (pl.col("_source") >= pl.col("_target"))
    ).height:
        raise ValueError("rolling boundaries must be strictly prior to target Date")
    index = {date: offset for offset, date in enumerate(sessions)}
    unknown_boundary_dates = sorted(
        set(candidate_boundaries["Date"].to_list()) - set(sessions)
    )
    if unknown_boundary_dates:
        raise ValueError(
            "rolling boundary dates absent from session calendar: "
            f"{unknown_boundary_dates[:5]}"
        )
    parts: list[pl.DataFrame] = []
    for date in sorted(candidate_boundaries["Date"].unique().to_list()):
        if date not in index:
            continue
        offset = index[date]
        prior = sessions[max(0, offset - config.lookback_sessions):offset]
        if len(prior) < config.min_history_sessions:
            continue
        recent = prior[-config.recent_sessions:]
        history = daily_liquidity.filter(pl.col("Date").is_in(recent))
        if history.is_empty():
            continue
        metrics = history.group_by("ValueCode").agg(
            pl.col("Date").n_unique().alias("liquidity_recent_sessions"),
            *[
                pl.col(column).median().alias(f"{column}_day_median")
                for column in (
                    "eligible_grid_rate",
                    "fresh_1000ms_rate",
                    "fresh_5000ms_rate",
                    "spot_spread_ticks_p50",
                    "spot_spread_ticks_p95",
                    "fut_spread_ticks_p50",
                    "fut_spread_ticks_p95",
                    "tt_band_bp_p50",
                    "tt_band_bp_p95",
                    "tt_band_fresh1000_bp_p50",
                    "tt_band_fresh5000_bp_p50",
                    "fresh_1000ms_rows",
                    "fresh_5000ms_rows",
                    "p_spot_buy_one_unit_l1_depth",
                    "p_spot_one_unit_given_fresh1000",
                    "p_fut_bid_depth_ge1contract",
                    "p_fut_one_unit_given_fresh1000",
                    "spot_changed_seconds_per_hour",
                    "fut_changed_seconds_per_hour",
                    "spot_changed_second_rate",
                    "fut_changed_second_rate",
                )
            ],
            pl.col("eligible_grid_rate")
            .quantile(0.10, interpolation="nearest")
            .alias("eligible_grid_rate_day_q10"),
        )
        long_support = (
            daily_liquidity.filter(pl.col("Date").is_in(prior))
            .group_by("ValueCode")
            .agg(pl.col("Date").n_unique().alias("liquidity_history_sessions"))
        )
        base = candidate_boundaries.filter(pl.col("Date") == date).join(
            metrics, on="ValueCode", how="left", validate="m:1"
        ).join(long_support, on="ValueCode", how="left", validate="m:1")
        if base.is_empty():
            continue
        parts.extend(
            [
                _route_screen(base, route, prior, config)
                for route in ENTRY_ROUTES
            ]
        )
    return (
        pl.concat(parts, how="vertical")
        .sort(["Date", "ValueCode", "boundary_quantile", "route"])
        if parts
        else pl.DataFrame()
    )


def _route_screen(
    base: pl.DataFrame,
    route: str,
    prior: list[str],
    config: LiquidityScreenConfig,
) -> pl.DataFrame:
    if route == "future_ask_spot_taker":
        hedge_p50 = "spot_spread_ticks_p50_day_median"
        hedge_p95 = "spot_spread_ticks_p95_day_median"
        maker_p50 = "fut_spread_ticks_p50_day_median"
        maker_p95 = "fut_spread_ticks_p95_day_median"
        depth = "p_spot_one_unit_given_fresh1000_day_median"
        hedge_activity = "spot_changed_seconds_per_hour_day_median"
        maker_activity = "fut_changed_second_rate_day_median"
    else:
        hedge_p50 = "fut_spread_ticks_p50_day_median"
        hedge_p95 = "fut_spread_ticks_p95_day_median"
        maker_p50 = "spot_spread_ticks_p50_day_median"
        maker_p95 = "spot_spread_ticks_p95_day_median"
        depth = "p_fut_one_unit_given_fresh1000_day_median"
        hedge_activity = "fut_changed_seconds_per_hour_day_median"
        maker_activity = "spot_changed_second_rate_day_median"
    result = base.with_columns(
        pl.lit(route).alias("route"),
        pl.col(hedge_p50).alias("hedge_spread_ticks_p50"),
        pl.col(hedge_p95).alias("hedge_spread_ticks_p95"),
        pl.col(maker_p50).alias("maker_spread_ticks_p50"),
        pl.col(maker_p95).alias("maker_spread_ticks_p95"),
        pl.col(depth).alias("unit_hedge_depth_rate"),
        pl.col(hedge_activity).alias("hedge_state_changes_per_hour"),
        pl.col(maker_activity).alias("maker_changed_second_rate"),
        pl.coalesce(
            "tt_band_fresh5000_bp_p50_day_median",
            "tt_band_bp_p50_day_median",
        ).alias("tt_band_reference_bp"),
        (
            (pl.col("upper_distance_bp") + pl.col("lower_distance_bp"))
            / pl.coalesce(
                "tt_band_fresh5000_bp_p50_day_median",
                "tt_band_bp_p50_day_median",
            )
        ).alias("latent_band_to_tt_band_ratio"),
    ).with_columns(
        (pl.col("upper_distance_bp") / pl.col("tt_band_reference_bp")).alias(
            "entry_upper_to_tt_band_ratio"
        ),
        (
            pl.col("tt_band_fresh5000_bp_p50_day_median").is_not_null()
            & (
                (
                    pl.col("tt_band_bp_p50_day_median")
                    - pl.col("tt_band_fresh5000_bp_p50_day_median")
                ).abs()
                > 20
            )
        ).fill_null(False).alias("stale_width_suspect"),
    ).with_columns(
        (
            (pl.col("hedge_spread_ticks_p50")
             > config.wide_hedge_spread_p50_ticks)
            | (pl.col("hedge_spread_ticks_p95")
               > config.wide_hedge_spread_p95_ticks)
        ).fill_null(True).alias("wide_hedge_spread_flag"),
        (
            (pl.col("maker_spread_ticks_p50")
             > config.wide_hedge_spread_p50_ticks)
            | (pl.col("maker_spread_ticks_p95")
               > config.wide_hedge_spread_p95_ticks)
        ).fill_null(False).alias("wide_maker_spread_flag"),
        (
            pl.col("liquidity_history_sessions") >= config.min_history_sessions
        ).fill_null(False).alias("long_history_gate"),
        pl.col("adaptive_parameter_valid").fill_null(False).alias(
            "boundary_parameter_gate"
        ),
        (
            pl.col("liquidity_recent_sessions") >= config.min_recent_sessions
        ).fill_null(False).alias("recent_history_gate"),
        (
            pl.col("eligible_grid_rate_day_median")
            >= config.eligible_median_warning_threshold
        ).fill_null(False).alias("eligible_median_warning_ok"),
        (
            pl.col("eligible_grid_rate_day_q10")
            >= config.eligible_q10_warning_threshold
        ).fill_null(False).alias("eligible_q10_warning_ok"),
        (
            pl.col("fresh_1000ms_rate_day_median")
            >= config.min_fresh_1000ms_rate_median
        ).fill_null(False).alias("fresh_1000ms_gate"),
        (
            pl.col("fresh_5000ms_rate_day_median")
            >= config.min_fresh_5000ms_rate_median
        ).fill_null(False).alias("fresh_5000ms_gate"),
        (
            pl.col("unit_hedge_depth_rate")
            >= config.min_unit_depth_rate_median
        ).fill_null(False).alias("depth_gate"),
        (
            pl.col("maker_changed_second_rate")
            >= config.min_maker_changed_second_rate_median
        ).fill_null(False).alias("maker_activity_gate"),
        (
            pl.col("tt_band_reference_bp").is_finite()
            & (pl.col("tt_band_reference_bp") > 0)
        ).fill_null(False).alias("tt_band_data_gate"),
    ).with_columns(
        (
            pl.col("boundary_parameter_gate")
            & pl.col("long_history_gate")
            & pl.col("recent_history_gate")
        ).alias("support_gate"),
        (
            pl.col("fresh_1000ms_gate")
            & pl.col("fresh_5000ms_gate")
            & pl.col("depth_gate")
            & pl.col("maker_activity_gate")
            & pl.col("tt_band_data_gate")
        ).alias("hard_data_gate"),
    ).with_columns(
        (
            pl.col("support_gate")
            & (~pl.col("eligible_q10_warning_ok"))
            & pl.col("eligible_median_warning_ok")
            & pl.col("fresh_1000ms_gate")
            & pl.col("fresh_5000ms_gate")
            & pl.col("depth_gate")
            & pl.col("maker_activity_gate")
            & pl.col("tt_band_data_gate")
        ).alias("eligible_q10_only_warning"),
        (~pl.col("eligible_median_warning_ok")).alias(
            "eligible_median_stress_flag"
        ),
        (
            pl.col("eligible_grid_rate_day_q10").is_not_null()
            & (~pl.col("eligible_q10_warning_ok"))
        ).alias("eligible_q10_stress_flag"),
    ).with_columns(
        pl.when(~pl.col("support_gate"))
        .then(pl.lit("insufficient_support"))
        .when(pl.col("hard_data_gate"))
        .then(pl.lit("pass"))
        .otherwise(pl.lit("known_fail"))
        .alias("liquidity_gate_status"),
        (pl.col("support_gate") & pl.col("hard_data_gate")).alias(
            "pre_replay_candidate"
        ),
        pl.when(~pl.col("support_gate"))
        .then(pl.lit("insufficient_support_exploration"))
        .when(~pl.col("hard_data_gate"))
        .then(pl.lit("known_fail"))
        .when(pl.col("wide_hedge_spread_flag"))
        .then(pl.lit("wide_hedge_cost_test"))
        .when(pl.col("wide_maker_spread_flag"))
        .then(pl.lit("wide_maker_exploration"))
        .otherwise(pl.lit("core_candidate"))
        .alias("replay_tier"),
        (
            pl.col("support_gate") & pl.col("hard_data_gate")
        ).alias("core_replay_eligible"),
        (
            (~pl.col("support_gate"))
            | (pl.col("support_gate") & pl.col("hard_data_gate"))
        ).alias("exploration_replay_eligible"),
        pl.lit(prior[0]).alias("liquidity_train_start_date"),
        pl.lit(prior[-1]).alias("liquidity_train_end_date"),
        pl.lit(config.screen_version).alias("liquidity_screen_version"),
        pl.lit(config.price_ladder_version).alias("price_ladder_version"),
        pl.lit(config.future_one_dollar_tick_effective_date).alias(
            "future_one_dollar_tick_effective_date"
        ),
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
        pl.lit(False).alias("production_universe_approved"),
    )
    return result


def _price_to_tick_index(
    price: pl.Expr,
    *,
    market: Literal["spot", "future"],
) -> pl.Expr:
    """Polars form of the versioned scalar ladder in ``targets``."""

    if market == "spot":
        high_price_boundary = pl.lit(1000.0)
    elif market == "future":
        high_price_boundary = (
            pl.when(
                pl.col("Date").cast(pl.String)
                >= FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            )
            .then(pl.lit(2500.0))
            .otherwise(pl.lit(1000.0))
        )
    else:
        raise ValueError(f"unknown price market: {market}")
    value = price.cast(pl.Float64)
    high_price_index = 3100.0 + (high_price_boundary - 500.0)
    return (
        pl.when(value < 10.0)
        .then(value / 0.01)
        .when(value < 50.0)
        .then(1000.0 + (value - 10.0) / 0.05)
        .when(value < 100.0)
        .then(1800.0 + (value - 50.0) / 0.1)
        .when(value < 500.0)
        .then(2300.0 + (value - 100.0) / 0.5)
        .when(value < high_price_boundary)
        .then(3100.0 + (value - 500.0))
        .otherwise(
            high_price_index + (value - high_price_boundary) / 5.0
        )
    )


def summarize_pseudo_validation_stability(
    screen: pl.DataFrame,
    rolling_boundaries: pl.DataFrame,
    config: LiquidityScreenConfig = LiquidityScreenConfig(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize retrospective route stability and a two-route core universe.

    These tables are research diagnostics, not D-day inputs.  The stable core
    intentionally requires both entry routes to remain in the narrow-spread
    core tier on at least ``stable_core_rate`` of pseudo-validation sessions.
    """

    sample = screen.filter(
        (pl.col("Date") >= config.pseudo_validation_start_date)
        & (pl.col("boundary_quantile") == 50)
    )
    if sample.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    end_date = sample["Date"].max()
    route = (
        sample.group_by(["ValueCode", "route"])
        .agg(
            pl.col("Date").n_unique().alias("product_days"),
            pl.col("support_gate").mean().alias("support_rate"),
            (pl.col("liquidity_gate_status") == "pass")
            .mean()
            .alias("pass_rate"),
            (pl.col("replay_tier") == "core_candidate")
            .mean()
            .alias("core_rate"),
            (pl.col("replay_tier") == "wide_maker_exploration")
            .mean()
            .alias("wide_maker_rate"),
            (pl.col("replay_tier") == "wide_hedge_cost_test")
            .mean()
            .alias("wide_hedge_rate"),
            pl.col("eligible_q10_only_warning")
            .mean()
            .alias("eligibility_q10_warning_rate"),
            pl.col("eligible_median_stress_flag")
            .mean()
            .alias("eligibility_median_warning_rate"),
            pl.col("maker_spread_ticks_p50")
            .median()
            .alias("maker_spread_ticks_p50_day_median"),
            pl.col("maker_spread_ticks_p95")
            .median()
            .alias("maker_spread_ticks_p95_day_median"),
            pl.col("hedge_spread_ticks_p50")
            .median()
            .alias("hedge_spread_ticks_p50_day_median"),
            pl.col("hedge_spread_ticks_p95")
            .median()
            .alias("hedge_spread_ticks_p95_day_median"),
            pl.col("fresh_1000ms_rate_day_median")
            .median()
            .alias("fresh_1000ms_rate_median"),
            pl.col("unit_hedge_depth_rate")
            .median()
            .alias("unit_hedge_depth_rate_median"),
        )
        .with_columns(
            pl.lit(config.pseudo_validation_start_date).alias(
                "pseudo_validation_start_date"
            ),
            pl.lit(end_date).alias("pseudo_validation_end_date"),
            pl.lit(True).alias("contains_target_day_outcome"),
            pl.lit(False).alias("execution_safe_snapshot"),
            pl.lit(False).alias("production_universe_approved"),
        )
        .sort(["route", "ValueCode"])
    )
    qualifying = route.filter(
        (pl.col("product_days") >= config.min_stability_sessions)
        & (pl.col("core_rate") >= config.stable_core_rate)
    )
    stable = (
        qualifying.group_by("ValueCode")
        .agg(
            pl.col("route").n_unique().alias("qualifying_routes"),
            pl.col("product_days").min().alias("min_route_product_days"),
            pl.col("pass_rate").min().alias("min_route_pass_rate"),
            pl.col("core_rate").min().alias("min_route_core_rate"),
        )
        .filter(pl.col("qualifying_routes") == len(ENTRY_ROUTES))
    )
    if stable.is_empty():
        return route, stable
    latest_boundaries = (
        rolling_boundaries.filter(
            (pl.col("Date") == end_date)
            & pl.col("ValueCode").is_in(stable["ValueCode"].to_list())
            & pl.col("boundary_quantile").is_in([50, 80, 95])
        )
        .select(
            "ValueCode",
            "QuoteCode",
            "boundary_quantile",
            "upper_distance_bp",
            "lower_distance_bp",
            "upper_distance_future_ticks",
            "lower_distance_future_ticks",
        )
        .with_columns(
            (pl.col("upper_distance_bp") + pl.col("lower_distance_bp")).alias(
                "full_band_bp"
            ),
            (
                pl.col("upper_distance_future_ticks")
                + pl.col("lower_distance_future_ticks")
            ).alias("full_band_future_ticks"),
        )
        .pivot(
            on="boundary_quantile",
            index=["ValueCode", "QuoteCode"],
            values=[
                "upper_distance_bp",
                "lower_distance_bp",
                "full_band_bp",
                "full_band_future_ticks",
            ],
        )
    )
    stable = (
        stable.join(latest_boundaries, on="ValueCode", how="left", validate="1:1")
        .with_columns(
            pl.lit(config.pseudo_validation_start_date).alias(
                "pseudo_validation_start_date"
            ),
            pl.lit(end_date).alias("pseudo_validation_end_date"),
            pl.lit(config.stable_core_rate).alias("stable_core_rate_threshold"),
            pl.lit(True).alias("retrospective_pseudo_validation"),
            pl.lit(False).alias("production_universe_approved"),
        )
        .sort("ValueCode")
    )
    return route, stable


def run_liquidity_screen_study(
    *,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    rolling_boundary_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: LiquidityScreenConfig = LiquidityScreenConfig(),
    reuse_daily_liquidity: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    config.validate()
    boundary_path = rolling_boundary_path or (
        daily_root.parent / "rolling_boundaries" / "rolling_boundary_snapshots.parquet"
    )
    # Boundary publication and its daily-source generation are validated
    # before either a cached daily summary or the underlying daily partitions
    # are opened.
    boundaries = load_rolling_boundary_snapshots(
        boundary_path,
        expected_daily_root=daily_root,
    )
    cached_daily_path = output_dir / "daily_liquidity.parquet"
    expected_lineage = _liquidity_source_lineage(
        boundary_path=boundary_path,
        daily_root=daily_root,
        config=config,
    )
    daily = (
        _load_reusable_daily_liquidity(
            output_dir,
            expected_lineage=expected_lineage,
            config=config,
        )
        if reuse_daily_liquidity
        else summarize_partitioned_daily_liquidity(daily_root)
    )
    _validate_daily_liquidity_lineage(daily, config)
    sessions = sorted(daily["Date"].unique().to_list())
    screen = build_rolling_liquidity_screen(daily, boundaries, sessions, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        complete_path.unlink()
    funnel = screen.group_by(
        [
            "Date",
            "route",
            "boundary_quantile",
            "liquidity_gate_status",
            "replay_tier",
        ]
    ).agg(
        pl.col("ValueCode").n_unique().alias("products"),
        pl.len().alias("rows"),
    )
    latest_date = screen["Date"].max() if screen.height else None
    latest = (
        screen.filter(
            (pl.col("Date") == latest_date)
            & (pl.col("boundary_quantile") == 50)
        ).sort(["route", "replay_tier", "ValueCode"])
        if latest_date is not None
        else pl.DataFrame()
    )
    product_history = daily.group_by("ValueCode").agg(
        pl.col("Date").n_unique().alias("days"),
        pl.col("spot_spread_ticks_p50").median().alias(
            "spot_spread_ticks_p50_day_median"
        ),
        pl.col("fut_spread_ticks_p50").median().alias(
            "fut_exec_spread_ticks_p50_day_median"
        ),
        pl.col("eligible_grid_rate").median().alias("eligible_rate_day_median"),
        pl.col("fresh_1000ms_rate").median().alias("fresh_1000ms_day_median"),
        pl.col("fresh_5000ms_rate").median().alias("fresh_5000ms_day_median"),
        pl.col("tt_band_fresh5000_bp_p50").median().alias(
            "tt_band_fresh5000_bp_day_median"
        ),
    ).sort("ValueCode")
    route_stability, stable_core = summarize_pseudo_validation_stability(
        screen, boundaries, config
    )
    _atomic_write_parquet(daily, output_dir / "daily_liquidity.parquet")
    _atomic_write_parquet(screen, output_dir / "rolling_liquidity_screen.parquet")
    _atomic_write_csv(funnel, output_dir / "screen_funnel_by_date.csv")
    _atomic_write_csv(latest, output_dir / "latest_q50_route_screen.csv")
    _atomic_write_csv(
        product_history,
        output_dir / "retrospective_product_liquidity_summary.csv",
    )
    _atomic_write_csv(
        route_stability,
        output_dir / "pseudo_validation_route_stability.csv",
    )
    _atomic_write_csv(
        stable_core,
        output_dir / "pseudo_validation_stable_core_products.csv",
    )
    payload = {
        **asdict(config),
        "wide_spread_semantics": (
            "maker/hedge spread and TTBand ratios are strata, not hard gates"
        ),
        "historical_eligibility_semantics": (
            "median/q10 are warning/state-risk features only; current-state "
            "eligibility is the runtime order gate"
        ),
        "daily_fact_maturity": "available after target-day close",
        "production_universe_approved": False,
        "latest_snapshot_date": latest_date,
        "daily_rows": daily.height,
        "screen_rows": screen.height,
        **expected_lineage,
    }
    config_path = output_dir / "config.json"
    _atomic_write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        config_path,
    )
    published = [
        "daily_liquidity.parquet",
        "rolling_liquidity_screen.parquet",
        "screen_funnel_by_date.csv",
        "latest_q50_route_screen.csv",
        "retrospective_product_liquidity_summary.csv",
        "pseudo_validation_route_stability.csv",
        "pseudo_validation_stable_core_products.csv",
        "config.json",
    ]
    frames_by_name = {
        "daily_liquidity.parquet": daily,
        "rolling_liquidity_screen.parquet": screen,
        "screen_funnel_by_date.csv": funnel,
        "latest_q50_route_screen.csv": latest,
        "retrospective_product_liquidity_summary.csv": product_history,
        "pseudo_validation_route_stability.csv": route_stability,
        "pseudo_validation_stable_core_products.csv": stable_core,
    }
    complete = {
        "complete": True,
        "schema_version": LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
        "publication_version": config.screen_version,
        "price_ladder_version": config.price_ladder_version,
        "future_one_dollar_tick_effective_date": (
            config.future_one_dollar_tick_effective_date
        ),
        "config": _liquidity_config_payload(config),
        "source_lineage": expected_lineage,
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256_file(output_dir / name),
                **(
                    {
                        "rows": frames_by_name[name].height,
                        "columns": frames_by_name[name].width,
                        "column_names": frames_by_name[name].columns,
                    }
                    if name in frames_by_name
                    else {}
                ),
            }
            for name in published
        },
    }
    complete["marker_payload_sha256"] = _canonical_json_sha256(complete)
    _atomic_write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        complete_path,
    )
    return daily, screen


def _load_reusable_daily_liquidity(
    output_dir: Path,
    *,
    expected_lineage: Mapping[str, str],
    config: LiquidityScreenConfig,
) -> pl.DataFrame:
    """Read a daily cache only after its complete publication is proven."""

    marker_path = output_dir / "complete.json"
    config_path = output_dir / "config.json"
    daily_path = output_dir / "daily_liquidity.parquet"
    for path in (marker_path, config_path, daily_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"reusable daily liquidity cache is incomplete: {path}"
            )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid liquidity completion marker: {marker_path}") from error
    if not isinstance(marker, dict) or marker.get("complete") is not True:
        raise ValueError(f"liquidity cache marker is incomplete: {marker_path}")
    if marker.get("schema_version") != LIQUIDITY_PUBLICATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported liquidity cache schema: {marker_path}")
    marker_digest = marker.get("marker_payload_sha256")
    digest_payload = dict(marker)
    digest_payload.pop("marker_payload_sha256", None)
    if marker_digest != _canonical_json_sha256(digest_payload):
        raise ValueError(f"liquidity cache marker hash mismatch: {marker_path}")
    if (
        marker.get("publication_version") != config.screen_version
        or marker.get("price_ladder_version") != PRICE_LADDER_VERSION
        or marker.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        or marker.get("config") != _liquidity_config_payload(config)
        or marker.get("source_lineage") != dict(expected_lineage)
    ):
        raise ValueError(f"liquidity cache lineage mismatch: {marker_path}")
    artifacts = marker.get("artifacts")
    entry = artifacts.get("daily_liquidity.parquet") if isinstance(artifacts, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"liquidity cache marker omits daily artifact: {marker_path}")
    if entry.get("bytes") != daily_path.stat().st_size or entry.get(
        "sha256"
    ) != _sha256_file(daily_path):
        raise ValueError(f"liquidity daily cache artifact mismatch: {daily_path}")
    schema = pl.read_parquet_schema(daily_path)
    rows = int(pl.scan_parquet(daily_path).select(pl.len()).collect().item())
    if (
        entry.get("rows") != rows
        or entry.get("columns") != len(schema)
        or entry.get("column_names") != list(schema.names())
    ):
        raise ValueError(f"liquidity daily cache shape mismatch: {daily_path}")
    config_entry = artifacts.get("config.json")
    if not isinstance(config_entry, dict) or config_entry.get(
        "bytes"
    ) != config_path.stat().st_size or config_entry.get(
        "sha256"
    ) != _sha256_file(config_path):
        raise ValueError(f"liquidity cache config artifact mismatch: {config_path}")
    persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(persisted_config, dict) or any(
        persisted_config.get(key) != value
        for key, value in _liquidity_config_payload(config).items()
    ):
        raise ValueError(f"liquidity cache config lineage mismatch: {config_path}")
    daily = pl.read_parquet(daily_path)
    _validate_daily_liquidity_lineage(daily, config)
    return daily


def _validate_daily_liquidity_lineage(
    daily: pl.DataFrame,
    config: LiquidityScreenConfig,
) -> None:
    required = {
        "price_ladder_version",
        "future_one_dollar_tick_effective_date",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily liquidity cache lacks ladder lineage: {missing}")
    if daily.filter(
        (pl.col("price_ladder_version") != config.price_ladder_version)
        | pl.col("price_ladder_version").is_null()
        | (
            pl.col("future_one_dollar_tick_effective_date")
            != config.future_one_dollar_tick_effective_date
        )
        | pl.col("future_one_dollar_tick_effective_date").is_null()
    ).height:
        raise ValueError("daily liquidity row price ladder lineage mismatch")


def _liquidity_config_payload(
    config: LiquidityScreenConfig,
) -> dict[str, object]:
    return asdict(config)


def _liquidity_source_lineage(
    *,
    boundary_path: Path,
    daily_root: Path,
    config: LiquidityScreenConfig,
) -> dict[str, str]:
    marker_path = Path(boundary_path).parent / "complete.json"
    return {
        "boundary_file_sha256": _sha256_file(Path(boundary_path)),
        "boundary_marker_sha256": _sha256_file(marker_path),
        "daily_marker_set_sha256": _daily_marker_set_sha256(Path(daily_root)),
        "liquidity_code_sha256": _liquidity_code_sha256(),
        "config_sha256": _canonical_json_sha256(
            _liquidity_config_payload(config)
        ),
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
    }


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_csv(temporary)
    temporary.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_marker_set_sha256(daily_root: Path) -> str:
    digest = hashlib.sha256()
    markers = sorted(daily_root.glob("Date=*/complete.json"))
    if not markers:
        raise FileNotFoundError(f"no completion markers below {daily_root}")
    for marker in markers:
        digest.update(str(marker.relative_to(daily_root)).encode())
        digest.update(marker.read_bytes())
    return digest.hexdigest()


def _liquidity_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "quote_width" / "daily_facts.py",
        Path(__file__).parents[1] / "quote_width" / "rolling.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rolling route-specific liquidity pre-replay screen"
    )
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--rolling-boundaries", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-daily-liquidity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily, screen = run_liquidity_screen_study(
        daily_root=args.daily_root,
        rolling_boundary_path=args.rolling_boundaries,
        output_dir=args.output_dir,
        reuse_daily_liquidity=args.reuse_daily_liquidity,
    )
    print(daily)
    print(screen)


if __name__ == "__main__":
    main()
