"""Prior-day width parameters and next-day latent basis-excursion tables.

This module deliberately stops before maker-fill inference.  A crossing of the
mid-basis residual is a potential entry state, not evidence that a queued order
would have filled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Iterable

import polars as pl

from ..common.landmarks import (
    REF_COMPARISON_EPS_RATIO,
    REF_LOWER_RETURN,
    REF_UPPER_RETURN,
)
from ..common.paths import DEFAULT_OUTPUT_ROOT, MAKER_ROOT
from ..fair_mid.anchors import prepare_fair_panel
from ..fair_mid.quote_churn import (
    price_to_tick_index,
    round_down_to_tick,
    round_up_to_tick,
    tick_index_to_price,
)
from ..quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


ANCHOR_COLUMN = "anchor_ewma_120s_bp"
DEFAULT_FIXED_WIDTHS_BP = (10.0, 20.0, 30.0, 40.0)
DEFAULT_TICK_MULTIPLIERS = (1.0, 2.0, 3.0)
DEFAULT_TT_BAND_MULTIPLIERS = (0.25, 0.50, 0.75)
DEFAULT_HORIZONS_SECONDS = (60, 120, 300, 600)
OBSERVATION_GAP_SECONDS = 30
SESSION_START_SECONDS = 300
MIN_PRIOR_COMPLETED_EXCURSIONS = 30
MIN_PRIOR_ELIGIBLE_RATE = 0.80
PRICE_EPS = 1e-9
PRIOR_FILE_PATTERN = re.compile(
    r"prior_landmarks_(?P<prior>\d{8})_for_(?P<target>\d{8})\.parquet$"
)


@dataclass(frozen=True)
class WidthTableResult:
    daily_product_parameters: pl.DataFrame
    product_parameter_summary: pl.DataFrame
    next_day_parameter_validation: pl.DataFrame
    validation_summary: pl.DataFrame
    potential_entry_events: pl.DataFrame
    width_policy_table: pl.DataFrame
    width_policy_summary: pl.DataFrame
    route_geometry: pl.DataFrame


def _next_tick(
    price: pl.Expr,
    *,
    market: str,
    session_date: pl.Expr,
) -> pl.Expr:
    index = price_to_tick_index(
        price,
        market=market,
        session_date=session_date,
    ).round(0)
    return tick_index_to_price(
        index + 1,
        market=market,
        session_date=session_date,
    )


def _previous_tick(
    price: pl.Expr,
    *,
    market: str,
    session_date: pl.Expr,
) -> pl.Expr:
    index = price_to_tick_index(
        price,
        market=market,
        session_date=session_date,
    ).round(0)
    return tick_index_to_price(
        index - 1,
        market=market,
        session_date=session_date,
    )


def add_microstructure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Add exact ladder, route-tick, spread, and taker-band state columns."""
    required = {
        "Date",
        "spot_bid",
        "spot_ask",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "basis_sell_taker_bp",
        "basis_buy_taker_bp",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"width table missing microstructure columns: {missing}")

    return frame.with_columns(
        (
            price_to_tick_index(
                pl.col("spot_ask"),
                market="spot",
                session_date=pl.col("Date"),
            )
            - price_to_tick_index(
                pl.col("spot_bid"),
                market="spot",
                session_date=pl.col("Date"),
            )
        ).round(0).alias("spot_spread_ticks"),
        (
            price_to_tick_index(
                pl.col("fut_ask"),
                market="future",
                session_date=pl.col("Date"),
            )
            - price_to_tick_index(
                pl.col("fut_bid"),
                market="future",
                session_date=pl.col("Date"),
            )
        ).round(0).alias("fut_spread_ticks"),
        (
            10_000
            * (
                _next_tick(
                    pl.col("fut_ask"),
                    market="future",
                    session_date=pl.col("Date"),
                )
                - pl.col("fut_ask")
            )
            / pl.col("spot_ask")
        ).alias("future_ask_route_tick_bp"),
        (
            10_000
            * pl.col("fut_exec_bid")
            * (
                1
                / _previous_tick(
                    pl.col("spot_bid"),
                    market="spot",
                    session_date=pl.col("Date"),
                )
                - 1 / pl.col("spot_bid")
            )
        ).alias("spot_bid_route_tick_bp"),
        (
            pl.col("basis_buy_taker_bp")
            - pl.col("basis_sell_taker_bp")
        ).alias("tt_band_width_bp"),
    )


def add_reference_tick_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Add pre-open-known route tick scales using each target day's refs."""
    required = {"Date", "spot_ref_price", "fut_ref_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"width table missing reference columns: {missing}")
    future_tick_bp = (
        10_000
        * (
            _next_tick(
                pl.col("fut_ref_price"),
                market="future",
                session_date=pl.col("Date"),
            )
            - pl.col("fut_ref_price")
        )
        / pl.col("spot_ref_price")
    )
    return frame.with_columns(
        future_tick_bp.alias("target_ref_future_tick_bp"),
        # Backwards-compatible route-specific alias.  Both columns are
        # generated by this one expression and therefore cannot drift.
        future_tick_bp.alias("target_ref_future_ask_tick_bp"),
        (
            10_000
            * pl.col("fut_ref_price")
            * (
                1
                / _previous_tick(
                    pl.col("spot_ref_price"),
                    market="spot",
                    session_date=pl.col("Date"),
                )
                - 1 / pl.col("spot_ref_price")
            )
        ).alias("target_ref_spot_bid_tick_bp"),
        pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version"),
    )


def _group_excursions(
    residuals: list[float | None],
    valid: list[bool | None],
    seconds: list[int] | None = None,
) -> list[dict[str, object]]:
    """Return non-overlapping sign excursions for one ordered fixed grid."""
    excursions: list[dict[str, object]] = []
    active_side: str | None = None
    start_index: int | None = None
    extreme: float | None = None
    previous_residual: float | None = None
    sequence = {"positive": 0, "negative": 0}

    def finish(end_index: int, completed: bool, reason: str) -> None:
        nonlocal active_side, start_index, extreme
        if active_side is None or start_index is None or extreme is None:
            return
        sequence[active_side] += 1
        amplitude = extreme if active_side == "positive" else -extreme
        excursions.append(
            {
                "side": active_side,
                "excursion_sequence": sequence[active_side],
                "start_index": start_index,
                "end_index": end_index,
                "completed": completed,
                "end_reason": reason,
                "amplitude_bp": amplitude,
            }
        )
        active_side = None
        start_index = None
        extreme = None

    for index, (residual, is_valid) in enumerate(zip(residuals, valid, strict=True)):
        timestamp_gap = (
            seconds is not None
            and index > 0
            and seconds[index] - seconds[index - 1] != 1
        )
        if timestamp_gap:
            if active_side is not None:
                finish(index - 1, False, "timestamp_gap")
            previous_residual = None
        if not is_valid or residual is None or not math.isfinite(float(residual)):
            if active_side is not None:
                finish(index - 1, False, "eligibility_gap")
            previous_residual = None
            continue

        value = float(residual)
        if active_side == "positive":
            extreme = max(float(extreme), value)
            if value <= 0:
                finish(index, True, "crossed_center")
        elif active_side == "negative":
            extreme = min(float(extreme), value)
            if value >= 0:
                finish(index, True, "crossed_center")

        # A direct sign change closes one side and starts the other on the same row.
        if active_side is None and previous_residual is not None:
            if previous_residual <= 0 < value:
                active_side = "positive"
                start_index = index
                extreme = value
            elif previous_residual >= 0 > value:
                active_side = "negative"
                start_index = index
                extreme = value
        previous_residual = value

    if active_side is not None:
        finish(len(residuals) - 1, False, "session_cutoff")
    return excursions


def extract_zero_crossing_excursions(
    panel: pl.DataFrame,
    anchor_column: str = ANCHOR_COLUMN,
) -> pl.DataFrame:
    """Extract one non-overlapping residual excursion per center departure."""
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        "analysis_eligible",
        anchor_column,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"width table missing excursion columns: {missing}")

    ordered = panel.sort(["Date", "ValueCode", "timestamp"]).with_columns(
        (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias("_residual_bp")
    )
    rows: list[dict[str, object]] = []
    for key, group in ordered.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        timestamps = group["timestamp"].to_list()
        seconds = group["seconds_from_open"].to_list()
        residuals = group["_residual_bp"].to_list()
        valid = group["analysis_eligible"].to_list()
        for excursion in _group_excursions(residuals, valid, seconds):
            start_index = int(excursion.pop("start_index"))
            end_index = int(excursion.pop("end_index"))
            rows.append(
                {
                    "Date": str(key[0]),
                    "ValueCode": str(key[1]),
                    "QuoteCode": str(key[2]),
                    **excursion,
                    "start_timestamp": timestamps[start_index],
                    "end_timestamp": timestamps[end_index],
                    "start_seconds_from_open": seconds[start_index],
                    "end_seconds_from_open": seconds[end_index],
                    "duration_seconds": seconds[end_index] - seconds[start_index],
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "Date": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
                "side": pl.String,
                "excursion_sequence": pl.Int64,
                "completed": pl.Boolean,
                "end_reason": pl.String,
                "amplitude_bp": pl.Float64,
                "start_timestamp": pl.Datetime("us"),
                "end_timestamp": pl.Datetime("us"),
                "start_seconds_from_open": pl.Int64,
                "end_seconds_from_open": pl.Int64,
                "duration_seconds": pl.Int64,
            }
        )
    return pl.DataFrame(rows).sort(
        ["Date", "ValueCode", "start_timestamp", "side"]
    )


def summarize_excursions(excursions: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Summarize completed amplitudes while retaining censor counts."""
    if excursions.is_empty():
        return pl.DataFrame()
    summary = (
        excursions.group_by(["Date", "ValueCode", "QuoteCode", "side"])
        .agg(
            pl.len().alias("started"),
            pl.col("completed").sum().alias("completed"),
            (~pl.col("completed")).sum().alias("censored"),
            pl.col("amplitude_bp")
            .filter(pl.col("completed"))
            .median()
            .alias("p50_amplitude_bp"),
            pl.col("amplitude_bp")
            .filter(pl.col("completed"))
            .quantile(0.80, interpolation="nearest")
            .alias("p80_amplitude_bp"),
            pl.col("amplitude_bp")
            .filter(pl.col("completed"))
            .quantile(0.95, interpolation="nearest")
            .alias("p95_amplitude_bp"),
            pl.col("duration_seconds")
            .filter(pl.col("completed"))
            .median()
            .alias("p50_duration_seconds"),
        )
        .with_columns(
            (pl.col("completed") / pl.col("started")).alias("completion_rate")
        )
        .pivot(
            on="side",
            index=["Date", "ValueCode", "QuoteCode"],
            values=[
                "started",
                "completed",
                "censored",
                "completion_rate",
                "p50_amplitude_bp",
                "p80_amplitude_bp",
                "p95_amplitude_bp",
                "p50_duration_seconds",
            ],
            separator="_",
        )
    )
    return summary.rename(
        {
            column: f"{prefix}_{column}"
            for column in summary.columns
            if column not in {"Date", "ValueCode", "QuoteCode"}
        }
    )


def _summarize_structure(panel: pl.DataFrame, prefix: str) -> pl.DataFrame:
    frame = add_microstructure_columns(panel)
    sample = frame.filter(pl.col("seconds_from_open") >= SESSION_START_SECONDS)
    groups = ["Date", "ValueCode", "QuoteCode"]
    result = sample.group_by(groups).agg(
        pl.len().alias("grid_rows"),
        pl.col("analysis_eligible").sum().alias("eligible_rows"),
        (
            pl.col("eligible_1000ms")
            & (pl.col("seconds_from_open") >= SESSION_START_SECONDS)
        ).sum().alias("fresh_1000ms_rows"),
        *[
            pl.col(column)
            .filter(pl.col("analysis_eligible"))
            .mean()
            .alias(f"{column}_mean")
            for column in ("spot_spread_ticks", "fut_spread_ticks")
        ],
        *[
            pl.col(column)
            .filter(pl.col("analysis_eligible"))
            .median()
            .alias(f"{column}_p50")
            for column in (
                "spot_spread_ticks",
                "fut_spread_ticks",
                "future_ask_route_tick_bp",
                "spot_bid_route_tick_bp",
                "tt_band_width_bp",
            )
        ],
        *[
            pl.col(column)
            .filter(pl.col("analysis_eligible"))
            .quantile(0.95, interpolation="nearest")
            .alias(f"{column}_p95")
            for column in (
                "spot_spread_ticks",
                "fut_spread_ticks",
                "tt_band_width_bp",
            )
        ],
    ).with_columns(
        (pl.col("eligible_rows") / pl.col("grid_rows")).alias("eligible_rate"),
        (pl.col("fresh_1000ms_rows") / pl.col("grid_rows")).alias(
            "fresh_1000ms_rate"
        ),
    )
    return result.rename(
        {
            column: f"{prefix}_{column}"
            for column in result.columns
            if column not in groups
        }
    )


def _target_reference_table(panel: pl.DataFrame) -> pl.DataFrame:
    with_ticks = add_reference_tick_columns(panel)
    return (
        with_ticks.group_by(["Date", "ValueCode", "QuoteCode"])
        .agg(
            pl.col("spot_ref_price").drop_nulls().first().alias("target_spot_ref_price"),
            pl.col("fut_ref_price").drop_nulls().first().alias("target_fut_ref_price"),
            pl.col("target_ref_future_tick_bp")
            .drop_nulls()
            .first(),
            pl.col("target_ref_future_ask_tick_bp")
            .drop_nulls()
            .first(),
            pl.col("target_ref_spot_bid_tick_bp").drop_nulls().first(),
            pl.col("price_ladder_version").drop_nulls().first(),
            pl.col("end_date").drop_nulls().first().alias("target_end_date"),
        )
        .with_columns(
            (
                pl.col("target_end_date")
                - pl.col("Date").str.strptime(pl.Date, "%Y%m%d")
            ).dt.total_days().alias("target_dte_days"),
            pl.when(pl.col("target_ref_future_tick_bp") <= 15)
            .then(pl.lit("00_le15bp"))
            .when(pl.col("target_ref_future_tick_bp") <= 22)
            .then(pl.lit("01_15to22bp"))
            .when(pl.col("target_ref_future_tick_bp") <= 30)
            .then(pl.lit("02_22to30bp"))
            .otherwise(pl.lit("03_gt30bp"))
            .alias("target_tick_bp_bucket")
        )
    )


def _prior_metadata(path: Path) -> tuple[str, str]:
    match = PRIOR_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized prior landmark filename: {path.name}")
    return match.group("prior"), match.group("target")


def load_prior_panels(prior_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load exact target-contract D-1 landmarks and return panels plus mapping."""
    panels: list[pl.DataFrame] = []
    mapping_rows: list[dict[str, object]] = []
    for path in sorted(prior_dir.glob("prior_landmarks_*_for_*.parquet")):
        prior_date, target_date = _prior_metadata(path)
        panel = prepare_fair_panel(pl.read_parquet(path)).with_columns(
            pl.lit(target_date).alias("TargetDate")
        )
        panels.append(panel)
        mapping_rows.append(
            {
                "Date": prior_date,
                "TargetDate": target_date,
                "calendar_gap_days": (
                    datetime.strptime(target_date, "%Y%m%d").date()
                    - datetime.strptime(prior_date, "%Y%m%d").date()
                ).days,
            }
        )
    if not panels:
        raise FileNotFoundError(f"no prior landmark files under {prior_dir}")
    return pl.concat(panels, how="diagonal_relaxed"), pl.DataFrame(mapping_rows)


def build_daily_product_parameters(
    target_panel: pl.DataFrame,
    prior_panel: pl.DataFrame,
    date_mapping: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the leakage-safe D-1 parameter table and D validation facts."""
    prior_excursions = extract_zero_crossing_excursions(prior_panel)
    target_excursions = extract_zero_crossing_excursions(target_panel)
    prior_excursion_summary = summarize_excursions(prior_excursions, "prior")
    target_excursion_summary = summarize_excursions(target_excursions, "current")
    prior_structure = _summarize_structure(prior_panel, "prior")
    target_structure = _summarize_structure(target_panel, "current")

    prior_facts = (
        prior_structure.join(
            prior_excursion_summary,
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
            validate="1:1",
        )
        .join(date_mapping, on="Date", how="left", validate="m:1")
        .rename({"Date": "prior_date", "TargetDate": "Date"})
    )
    target_refs = _target_reference_table(target_panel)
    parameters = (
        target_refs.join(
            prior_facts,
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("prior_p80_amplitude_bp_positive")
                / pl.col("target_ref_future_tick_bp")
            ).alias("prior_positive_p80_future_ticks"),
            (
                pl.col("prior_p80_amplitude_bp_positive")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("prior_positive_p80_spot_ticks"),
            (
                pl.col("prior_p80_amplitude_bp_positive")
                / pl.col("prior_tt_band_width_bp_p50")
            ).alias("prior_positive_p80_tt_bands"),
            pl.when(pl.col("prior_fut_spread_ticks_p50") <= 1 + PRICE_EPS)
            .then(pl.lit("1tick"))
            .when(pl.col("prior_fut_spread_ticks_p50") <= 2 + PRICE_EPS)
            .then(pl.lit("2ticks"))
            .otherwise(pl.lit("3plus_ticks"))
            .alias("prior_future_spread_bucket"),
            (
                (pl.col("prior_eligible_rate") >= MIN_PRIOR_ELIGIBLE_RATE)
                & (
                    pl.col("prior_completed_positive")
                    >= MIN_PRIOR_COMPLETED_EXCURSIONS
                )
                & pl.col("prior_p80_amplitude_bp_positive").is_not_null()
            ).fill_null(False).alias("prior_parameter_valid"),
            pl.lit("completed_only").alias(
                "prior_excursion_quantile_sample"
            ),
        )
        .sort(["Date", "ValueCode"])
    )

    current_facts = target_excursion_summary.join(
        target_structure,
        on=["Date", "ValueCode", "QuoteCode"],
        how="left",
        validate="1:1",
    )
    validation = (
        parameters.join(
            current_facts,
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
            validate="1:1",
        )
        .join(
            target_excursions.filter(
                (pl.col("side") == "positive") & pl.col("completed")
            )
            .join(
                parameters.select(
                    "Date",
                    "ValueCode",
                    "QuoteCode",
                    "prior_p80_amplitude_bp_positive",
                ),
                on=["Date", "ValueCode", "QuoteCode"],
                how="left",
                validate="m:1",
            )
            .group_by(["Date", "ValueCode", "QuoteCode"])
            .agg(
                (
                    pl.col("amplitude_bp")
                    >= pl.col("prior_p80_amplitude_bp_positive")
                ).mean().alias("current_reach_prior_positive_p80_rate")
            ),
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("current_p80_amplitude_bp_positive")
                - pl.col("prior_p80_amplitude_bp_positive")
            ).alias("positive_p80_error_bp"),
            (
                pl.col("current_p80_amplitude_bp_negative")
                - pl.col("prior_p80_amplitude_bp_negative")
            ).alias("negative_p80_error_bp"),
            (
                pl.col("current_p80_amplitude_bp_positive")
                / pl.col("prior_p80_amplitude_bp_positive")
            ).alias("positive_p80_current_to_prior_ratio"),
            (
                pl.col("current_p80_amplitude_bp_negative")
                / pl.col("prior_p80_amplitude_bp_negative")
            ).alias("negative_p80_current_to_prior_ratio"),
        )
        .sort(["Date", "ValueCode"])
    )
    return parameters, validation, target_excursions


def build_width_candidates(
    parameters: pl.DataFrame,
    fixed_widths_bp: Iterable[float] = DEFAULT_FIXED_WIDTHS_BP,
) -> pl.DataFrame:
    """Expand D-1 parameters into leakage-safe latent diagnostic candidates."""
    base = parameters.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "prior_date",
        "calendar_gap_days",
        "target_dte_days",
        "target_tick_bp_bucket",
        "prior_future_spread_bucket",
        "target_ref_future_tick_bp",
        "target_ref_future_ask_tick_bp",
        "target_ref_spot_bid_tick_bp",
        "price_ladder_version",
        "prior_tt_band_width_bp_p50",
        "prior_p50_amplitude_bp_positive",
        "prior_p80_amplitude_bp_positive",
        "prior_p95_amplitude_bp_positive",
        "prior_parameter_valid",
    )
    frames: list[pl.DataFrame] = []
    for width in fixed_widths_bp:
        frames.append(
            base.with_columns(
                pl.lit(f"fixed_{int(width)}bp").alias("width_policy"),
                pl.lit("fixed_bp").alias("width_family"),
                pl.lit(width).alias("candidate_width_bp"),
            )
        )
    for multiplier in DEFAULT_TICK_MULTIPLIERS:
        label = f"{multiplier:g}x"
        frames.extend(
            [
                base.with_columns(
                    pl.lit(f"future_tick_{label}").alias("width_policy"),
                    pl.lit("future_route_tick").alias("width_family"),
                    (
                        pl.col("target_ref_future_tick_bp") * multiplier
                    ).alias("candidate_width_bp"),
                ),
                base.with_columns(
                    pl.lit(f"spot_tick_{label}").alias("width_policy"),
                    pl.lit("spot_route_tick").alias("width_family"),
                    (
                        pl.col("target_ref_spot_bid_tick_bp") * multiplier
                    ).alias("candidate_width_bp"),
                ),
            ]
        )
    prior_base = base.filter(pl.col("prior_parameter_valid"))
    for multiplier in DEFAULT_TT_BAND_MULTIPLIERS:
        frames.append(
            prior_base.with_columns(
                pl.lit(f"prior_ttband_{multiplier:g}x").alias("width_policy"),
                pl.lit("prior_tt_band").alias("width_family"),
                (
                    pl.col("prior_tt_band_width_bp_p50") * multiplier
                ).alias("candidate_width_bp"),
            )
        )
    for quantile in (50, 80, 95):
        frames.append(
            prior_base.with_columns(
                pl.lit(f"prior_excursion_p{quantile}").alias("width_policy"),
                pl.lit("prior_excursion").alias("width_family"),
                pl.col(f"prior_p{quantile}_amplitude_bp_positive").alias(
                    "candidate_width_bp"
                ),
            )
        )
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .filter(
            pl.col("candidate_width_bp").is_finite()
            & (pl.col("candidate_width_bp") > 0)
        )
        .with_columns(
            (
                pl.col("candidate_width_bp")
                / pl.col("target_ref_future_tick_bp")
            ).alias("candidate_width_future_ticks"),
            (
                pl.col("candidate_width_bp")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("candidate_width_spot_ticks"),
            (
                pl.col("candidate_width_bp")
                / pl.col("prior_tt_band_width_bp_p50")
            ).alias("candidate_width_tt_bands"),
        )
    )


def _outcome_at_horizon(
    residuals: list[float | None],
    valid: list[bool | None],
    seconds: list[int],
    trigger_index: int,
    width_bp: float,
    horizon_seconds: int,
    observation_gap_seconds: int,
) -> tuple[bool | None, bool | None]:
    trigger_second = seconds[trigger_index]
    continuous_limit = trigger_index
    for index in range(trigger_index + 1, len(seconds)):
        offset = seconds[index] - trigger_second
        if offset > horizon_seconds:
            break
        if not valid[index] or residuals[index] is None:
            break
        continuous_limit = index

    center_hit = False
    symmetric_hit = False
    for index in range(trigger_index, continuous_limit + 1):
        offset = seconds[index] - trigger_second
        if offset < observation_gap_seconds:
            continue
        value = float(residuals[index])
        center_hit = center_hit or value <= 0
        symmetric_hit = symmetric_hit or value <= -width_bp

    observed_horizon = (
        seconds[continuous_limit] - trigger_second >= horizon_seconds
    )
    if not observed_horizon:
        return None, None
    return center_hit, symmetric_hit


def build_potential_entry_events(
    panel: pl.DataFrame,
    candidates: pl.DataFrame,
    anchor_column: str = ANCHOR_COLUMN,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
    observation_gap_seconds: int = OBSERVATION_GAP_SECONDS,
) -> pl.DataFrame:
    """Build first-crossing latent entry events and delayed convergence labels."""
    horizon_values = tuple(sorted(set(int(value) for value in horizons_seconds)))
    if not horizon_values or horizon_values[0] < observation_gap_seconds:
        raise ValueError("all horizons must be at least the observation gap")
    ordered = (
        add_microstructure_columns(panel)
        .sort(["Date", "ValueCode", "timestamp"])
        .with_columns(
            (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias("_residual_bp"),
            (
                pl.col("anchor_ewma_120s_bp")
                - pl.col("anchor_ewma_300s_bp")
            ).abs().alias("fast_slow_gap_bp"),
        )
    )
    candidate_map: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for key, group in candidates.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        candidate_map[(str(key[0]), str(key[1]), str(key[2]))] = group.to_dicts()

    rows: list[dict[str, object]] = []
    for key, group in ordered.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        group_key = (str(key[0]), str(key[1]), str(key[2]))
        specs = candidate_map.get(group_key, [])
        if not specs:
            continue
        residuals = group["_residual_bp"].to_list()
        valid = group["analysis_eligible"].to_list()
        seconds = group["seconds_from_open"].to_list()
        timestamps = group["timestamp"].to_list()
        excursions = _group_excursions(residuals, valid, seconds)
        positive = [value for value in excursions if value["side"] == "positive"]
        for excursion in positive:
            start = int(excursion["start_index"])
            end = int(excursion["end_index"])
            for spec in specs:
                width = float(spec["candidate_width_bp"])
                trigger = next(
                    (
                        index
                        for index in range(start, end + 1)
                        if residuals[index] is not None
                        and float(residuals[index]) + PRICE_EPS >= width
                    ),
                    None,
                )
                if trigger is None:
                    continue
                observed_width = float(residuals[trigger])
                record: dict[str, object] = {
                    "Date": group_key[0],
                    "ValueCode": group_key[1],
                    "QuoteCode": group_key[2],
                    "parent_excursion_sequence": excursion["excursion_sequence"],
                    "parent_excursion_completed": excursion["completed"],
                    "width_policy": spec["width_policy"],
                    "width_family": spec["width_family"],
                    "candidate_width_bp": width,
                    "candidate_width_future_ticks": spec[
                        "candidate_width_future_ticks"
                    ],
                    "candidate_width_spot_ticks": spec[
                        "candidate_width_spot_ticks"
                    ],
                    "candidate_width_tt_bands": spec[
                        "candidate_width_tt_bands"
                    ],
                    "trigger_timestamp": timestamps[trigger],
                    "trigger_seconds_from_open": seconds[trigger],
                    "trigger_residual_bp": observed_width,
                    "trigger_overshoot_bp": observed_width - width,
                    "trigger_spot_spread_ticks": group.item(
                        trigger, "spot_spread_ticks"
                    ),
                    "trigger_fut_spread_ticks": group.item(
                        trigger, "fut_spread_ticks"
                    ),
                    "trigger_tt_band_width_bp": group.item(
                        trigger, "tt_band_width_bp"
                    ),
                    "trigger_future_route_tick_bp": group.item(
                        trigger, "future_ask_route_tick_bp"
                    ),
                    "trigger_spot_route_tick_bp": group.item(
                        trigger, "spot_bid_route_tick_bp"
                    ),
                    "trigger_fast_slow_gap_bp": group.item(
                        trigger, "fast_slow_gap_bp"
                    ),
                }
                for horizon in horizon_values:
                    center, symmetric = _outcome_at_horizon(
                        residuals,
                        valid,
                        seconds,
                        trigger,
                        width,
                        horizon,
                        observation_gap_seconds,
                    )
                    record[f"delayed_center_path_hit_{horizon}s"] = center
                    record[f"delayed_symmetric_path_hit_{horizon}s"] = symmetric
                rows.append(record)
    return pl.DataFrame(rows).sort(
        ["Date", "ValueCode", "width_policy", "trigger_timestamp"]
    )


def _validation_summary(validation: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    scopes = [
        ("all_diagnostic", validation),
        ("all_prior_valid", validation.filter(pl.col("prior_parameter_valid"))),
    ] + [
        (f"{key[0]}_diagnostic", group)
        for key, group in validation.group_by("ValueCode", maintain_order=True)
    ] + [
        (f"{key[0]}_prior_valid", group.filter(pl.col("prior_parameter_valid")))
        for key, group in validation.group_by("ValueCode", maintain_order=True)
    ]
    for scope, group in scopes:
        frames.append(
            group.select(
                pl.lit(scope).alias("scope"),
                pl.len().alias("pairs"),
                pl.col("Date").n_unique().alias("dates"),
                pl.corr(
                    "prior_p80_amplitude_bp_positive",
                    "current_p80_amplitude_bp_positive",
                ).alias("positive_p80_prior_current_corr"),
                pl.col("positive_p80_error_bp")
                .abs()
                .median()
                .alias("positive_p80_median_abs_error_bp"),
                pl.col("positive_p80_current_to_prior_ratio")
                .median()
                .alias("positive_p80_median_current_to_prior_ratio"),
                pl.corr(
                    "prior_p80_amplitude_bp_negative",
                    "current_p80_amplitude_bp_negative",
                ).alias("negative_p80_prior_current_corr"),
                pl.col("negative_p80_error_bp")
                .abs()
                .median()
                .alias("negative_p80_median_abs_error_bp"),
                pl.col("current_reach_prior_positive_p80_rate")
                .median()
                .alias("median_current_reach_prior_positive_p80_rate"),
            )
        )
    return pl.concat(frames).sort("scope")


def _product_parameter_summary(validation: pl.DataFrame) -> pl.DataFrame:
    """Make a compact product-facing view of structural and excursion scales."""
    return (
        validation.group_by("ValueCode")
        .agg(
            pl.len().alias("pairs"),
            pl.col("prior_parameter_valid").sum().alias("prior_valid_pairs"),
            pl.col("target_ref_future_ask_tick_bp")
            .quantile(0.10, interpolation="nearest")
            .alias("ref_future_tick_bp_p10"),
            pl.col("target_ref_future_ask_tick_bp")
            .median()
            .alias("ref_future_tick_bp_p50"),
            pl.col("target_ref_future_ask_tick_bp")
            .quantile(0.90, interpolation="nearest")
            .alias("ref_future_tick_bp_p90"),
            pl.col("prior_spot_spread_ticks_mean")
            .median()
            .alias("prior_spot_spread_ticks_mean_pair_median"),
            pl.col("prior_fut_spread_ticks_mean")
            .median()
            .alias("prior_fut_spread_ticks_mean_pair_median"),
            pl.col("prior_tt_band_width_bp_p50")
            .median()
            .alias("prior_tt_band_bp_pair_median"),
            pl.col("prior_p50_amplitude_bp_positive")
            .median()
            .alias("prior_positive_excursion_p50_bp_pair_median"),
            pl.col("prior_p80_amplitude_bp_positive")
            .median()
            .alias("prior_positive_excursion_p80_bp_pair_median"),
            pl.col("prior_p95_amplitude_bp_positive")
            .median()
            .alias("prior_positive_excursion_p95_bp_pair_median"),
            pl.col("prior_positive_p80_future_ticks")
            .median()
            .alias("prior_positive_p80_future_ticks_pair_median"),
            pl.col("positive_p80_error_bp")
            .abs()
            .median()
            .alias("next_day_positive_p80_median_abs_error_bp"),
            pl.corr(
                "prior_p80_amplitude_bp_positive",
                "current_p80_amplitude_bp_positive",
            ).alias("next_day_positive_p80_corr"),
            pl.col("current_reach_prior_positive_p80_rate")
            .median()
            .alias("next_day_reach_prior_p80_rate_pair_median"),
            pl.col("positive_p80_error_bp")
            .filter(pl.col("prior_parameter_valid"))
            .abs()
            .median()
            .alias("valid_next_day_positive_p80_median_abs_error_bp"),
            pl.corr(
                pl.col("prior_p80_amplitude_bp_positive")
                .filter(pl.col("prior_parameter_valid")),
                pl.col("current_p80_amplitude_bp_positive")
                .filter(pl.col("prior_parameter_valid")),
            ).alias("valid_next_day_positive_p80_corr"),
            pl.col("current_reach_prior_positive_p80_rate")
            .filter(pl.col("prior_parameter_valid"))
            .median()
            .alias("valid_next_day_reach_prior_p80_rate_pair_median"),
        )
        .sort("ValueCode")
    )


def _price_in_ref_band(price: pl.Expr, reference: pl.Expr) -> pl.Expr:
    epsilon = reference.abs() * REF_COMPARISON_EPS_RATIO
    return (
        reference.is_not_null()
        & (reference > 0)
        & price.is_not_null()
        & (price > reference * (1 + REF_LOWER_RETURN) + epsilon)
        & (price < reference * (1 + REF_UPPER_RETURN) - epsilon)
    ).fill_null(False)


def summarize_entry_route_geometry(
    panel: pl.DataFrame,
    candidates: pl.DataFrame,
    anchor_column: str = ANCHOR_COLUMN,
) -> pl.DataFrame:
    """Summarize exact rounded entry quotes; no target touch is called a fill."""
    sample = panel.filter(
        pl.col("analysis_eligible") & pl.col(anchor_column).is_not_null()
    )
    keys = ["Date", "ValueCode", "QuoteCode"]
    frames: list[pl.DataFrame] = []
    for policy_key, policy in candidates.group_by(
        ["width_family", "width_policy"], maintain_order=True
    ):
        joined = sample.join(
            policy.select(*keys, "candidate_width_bp"),
            on=keys,
            how="inner",
            validate="m:1",
        ).with_columns(
            (
                pl.col(anchor_column) + pl.col("candidate_width_bp")
            ).alias("_threshold_bp")
        )
        for route in ("future_ask_spot_taker", "spot_bid_future_taker"):
            if route == "future_ask_spot_taker":
                target = round_up_to_tick(
                    pl.col("spot_ask")
                    * (1 + pl.col("_threshold_bp") / 10_000),
                    market="future",
                    session_date=pl.col("Date"),
                )
                maker_bbo = pl.col("fut_ask")
                target_offset = (
                    price_to_tick_index(
                        pl.col("_target_price"),
                        market="future",
                        session_date=pl.col("Date"),
                    )
                    - price_to_tick_index(
                        maker_bbo,
                        market="future",
                        session_date=pl.col("Date"),
                    )
                )
                effective_basis = (
                    10_000
                    * (pl.col("_target_price") / pl.col("spot_ask") - 1)
                )
                route_tick = (
                    10_000
                    * (
                        _next_tick(
                            pl.col("_target_price"),
                            market="future",
                            session_date=pl.col("Date"),
                        )
                        - pl.col("_target_price")
                    )
                    / pl.col("spot_ask")
                )
                passive = pl.col("_target_price") > pl.col("fut_exec_bid")
                legal = _price_in_ref_band(
                    pl.col("_target_price"), pl.col("fut_ref_price")
                )
            else:
                target = round_down_to_tick(
                    pl.col("fut_exec_bid")
                    / (1 + pl.col("_threshold_bp") / 10_000),
                    market="spot",
                    session_date=pl.col("Date"),
                )
                maker_bbo = pl.col("spot_bid")
                target_offset = (
                    price_to_tick_index(
                        maker_bbo,
                        market="spot",
                        session_date=pl.col("Date"),
                    )
                    - price_to_tick_index(
                        pl.col("_target_price"),
                        market="spot",
                        session_date=pl.col("Date"),
                    )
                )
                effective_basis = (
                    10_000
                    * (pl.col("fut_exec_bid") / pl.col("_target_price") - 1)
                )
                route_tick = (
                    10_000
                    * pl.col("fut_exec_bid")
                    * (
                        1
                        / _previous_tick(
                            pl.col("_target_price"),
                            market="spot",
                            session_date=pl.col("Date"),
                        )
                        - 1 / pl.col("_target_price")
                    )
                )
                passive = pl.col("_target_price") < pl.col("spot_ask")
                legal = _price_in_ref_band(
                    pl.col("_target_price"), pl.col("spot_ref_price")
                )
            route_frame = (
                joined.with_columns(target.alias("_target_price"))
                .with_columns(
                    target_offset.round(0).alias("_target_offset_ticks"),
                    (effective_basis - pl.col(anchor_column)).alias(
                        "_effective_width_bp"
                    ),
                    route_tick.alias("_route_tick_bp"),
                    passive.alias("_passive"),
                    legal.alias("_legal"),
                )
                .with_columns(
                    (
                        pl.col("_effective_width_bp")
                        - pl.col("candidate_width_bp")
                    ).alias("_rounding_excess_bp"),
                    (
                        pl.col(anchor_column)
                        + pl.col("_effective_width_bp")
                        - pl.col("basis_sell_taker_bp")
                    ).alias("_maker_vs_sell_tt_improvement_bp"),
                )
            )
            frames.append(
                route_frame.group_by(keys).agg(
                    pl.lit(str(policy_key[0])).alias("width_family"),
                    pl.lit(str(policy_key[1])).alias("width_policy"),
                    pl.lit(route).alias("route"),
                    pl.col("candidate_width_bp").first(),
                    pl.len().alias("state_rows"),
                    pl.col("_legal").mean().alias("target_legal_rate"),
                    pl.col("_passive").mean().alias("target_passive_rate"),
                    (pl.col("_target_offset_ticks") < 0)
                    .mean()
                    .alias("target_inside_spread_rate"),
                    (pl.col("_target_offset_ticks") == 0)
                    .mean()
                    .alias("target_at_bbo_rate"),
                    (pl.col("_target_offset_ticks") > 0)
                    .mean()
                    .alias("target_behind_bbo_rate"),
                    pl.col("_target_offset_ticks")
                    .median()
                    .alias("target_offset_ticks_p50"),
                    pl.col("_target_offset_ticks")
                    .quantile(0.95, interpolation="nearest")
                    .alias("target_offset_ticks_p95"),
                    pl.col("_effective_width_bp")
                    .median()
                    .alias("effective_width_bp_p50"),
                    pl.col("_effective_width_bp")
                    .quantile(0.95, interpolation="nearest")
                    .alias("effective_width_bp_p95"),
                    pl.col("_rounding_excess_bp")
                    .median()
                    .alias("rounding_excess_bp_p50"),
                    pl.col("_route_tick_bp").median().alias("route_tick_bp_p50"),
                    pl.col("_maker_vs_sell_tt_improvement_bp")
                    .median()
                    .alias("maker_vs_sell_tt_improvement_bp_p50"),
                    (
                        pl.col("_effective_width_bp")
                        + PRICE_EPS
                        < pl.col("candidate_width_bp")
                    ).sum().alias("rounding_inequality_violations"),
                )
            )
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version")
        )
        .sort(
            ["Date", "ValueCode", "route", "width_family", "candidate_width_bp"]
        )
    )


def _width_policy_table(
    candidates: pl.DataFrame,
    target_excursions: pl.DataFrame,
    events: pl.DataFrame,
) -> pl.DataFrame:
    positive = target_excursions.filter(pl.col("side") == "positive")
    base = positive.group_by(["Date", "ValueCode", "QuoteCode"]).agg(
        pl.len().alias("positive_excursions_started"),
        pl.col("completed").sum().alias("positive_excursions_completed"),
        (~pl.col("completed")).sum().alias("positive_excursions_censored"),
    )
    if events.is_empty():
        return candidates.join(
            base, on=["Date", "ValueCode", "QuoteCode"], how="left"
        )
    horizons = sorted(
        int(match.group(1))
        for column in events.columns
        if (match := re.fullmatch(r"delayed_center_path_hit_(\d+)s", column))
    )
    aggregations: list[pl.Expr] = [
        pl.len().alias("potential_entry_events"),
        pl.col("parent_excursion_completed").sum().alias(
            "triggered_parent_excursions_completed"
        ),
        pl.col("trigger_spot_spread_ticks").median(),
        pl.col("trigger_fut_spread_ticks").median(),
        pl.col("trigger_tt_band_width_bp").median(),
        pl.col("trigger_fast_slow_gap_bp").median(),
        pl.col("trigger_overshoot_bp").median().alias("trigger_overshoot_bp_p50"),
        pl.col("trigger_overshoot_bp")
        .quantile(0.95, interpolation="nearest")
        .alias("trigger_overshoot_bp_p95"),
    ]
    for horizon in horizons:
        for outcome in ("center", "symmetric"):
            column = f"delayed_{outcome}_path_hit_{horizon}s"
            aggregations.extend(
                [
                    pl.col(column)
                    .count()
                    .alias(f"n_delayed_{outcome}_observed_{horizon}s"),
                    pl.col(column)
                    .mean()
                    .alias(f"p_delayed_{outcome}_path_hit_{horizon}s"),
                ]
            )
    event_summary = events.group_by(
        ["Date", "ValueCode", "QuoteCode", "width_policy", "width_family"]
    ).agg(*aggregations)
    return (
        candidates.join(
            base,
            on=["Date", "ValueCode", "QuoteCode"],
            how="left",
            validate="m:1",
        )
        .join(
            event_summary,
            on=[
                "Date",
                "ValueCode",
                "QuoteCode",
                "width_policy",
                "width_family",
            ],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("potential_entry_events").fill_null(0),
            (
                pl.col("potential_entry_events")
                / pl.col("positive_excursions_started")
            ).alias("reach_rate_all_started_excursions"),
            (
                pl.col("triggered_parent_excursions_completed")
                / pl.col("positive_excursions_completed")
            ).alias("reach_rate_completed_excursions"),
        )
        .sort(["Date", "ValueCode", "width_policy"])
    )


def _width_policy_summary(table: pl.DataFrame) -> pl.DataFrame:
    probability_columns = [
        column
        for column in table.columns
        if column.startswith("p_delayed_center_path_hit_")
        or column.startswith("p_delayed_symmetric_path_hit_")
    ]
    observed_columns = [
        column
        for column in table.columns
        if column.startswith("n_delayed_center_observed_")
    ]
    return (
        table.group_by(["width_family", "width_policy"])
        .agg(
            pl.len().alias("pairs"),
            (pl.col("potential_entry_events") > 0)
            .sum()
            .alias("pairs_with_potential_entries"),
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("symbols"),
            pl.col("candidate_width_bp").median().alias("median_width_bp"),
            pl.col("candidate_width_future_ticks")
            .median()
            .alias("median_width_future_ticks"),
            pl.col("candidate_width_tt_bands")
            .median()
            .alias("median_width_tt_bands"),
            pl.col("prior_parameter_valid").sum().alias("prior_valid_pairs"),
            pl.col("potential_entry_events")
            .median()
            .alias("median_potential_entries_per_pair_day"),
            pl.col("reach_rate_completed_excursions")
            .median()
            .alias("median_reach_rate_completed_excursions"),
            *[
                pl.col(column).median().alias(f"pair_median_{column}")
                for column in probability_columns
            ],
            *[
                (pl.col(column).fill_null(0) > 0)
                .sum()
                .alias(f"pairs_with_{column}")
                for column in observed_columns
            ],
            *[
                pl.col(column).fill_null(0).sum().alias(f"total_{column}")
                for column in observed_columns
            ],
        )
        .sort(["width_family", "median_width_bp", "width_policy"])
    )


def run_width_study(
    panel_path: Path,
    prior_dir: Path,
    output_dir: Path,
) -> WidthTableResult:
    """Run and persist the D-1-to-D width-table pilot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_panel = pl.read_parquet(panel_path)
    prior_panel, mapping = load_prior_panels(prior_dir)
    parameters, validation, target_excursions = build_daily_product_parameters(
        target_panel, prior_panel, mapping
    )
    candidates = build_width_candidates(parameters)
    events = build_potential_entry_events(target_panel, candidates)
    policy_table = _width_policy_table(candidates, target_excursions, events)
    validation_summary = _validation_summary(validation)
    product_summary = _product_parameter_summary(validation)
    policy_summary = _width_policy_summary(policy_table)
    route_geometry = summarize_entry_route_geometry(target_panel, candidates)

    parameters.write_csv(output_dir / "daily_product_parameters.csv")
    product_summary.write_csv(output_dir / "product_parameter_summary.csv")
    validation.write_csv(output_dir / "next_day_parameter_validation.csv")
    validation_summary.write_csv(output_dir / "validation_summary.csv")
    target_excursions.write_parquet(output_dir / "target_zero_crossing_excursions.parquet")
    events.write_parquet(output_dir / "potential_entry_events.parquet")
    policy_table.write_csv(output_dir / "width_policy_by_day_symbol.csv")
    policy_summary.write_csv(output_dir / "width_policy_summary.csv")
    route_geometry.write_csv(output_dir / "entry_route_geometry.csv")
    config = {
        "anchor_column": ANCHOR_COLUMN,
        "fixed_widths_bp": list(DEFAULT_FIXED_WIDTHS_BP),
        "tick_multipliers": list(DEFAULT_TICK_MULTIPLIERS),
        "fixed_bp_role": "diagnostic_only; not a production action shortlist",
        "tick_grid_role": "diagnostic_only; not a production action shortlist",
        "actionable_execution": False,
        "ev_ready": False,
        "tt_band_multipliers": list(DEFAULT_TT_BAND_MULTIPLIERS),
        "horizons_seconds": list(DEFAULT_HORIZONS_SECONDS),
        "observation_gap_seconds": OBSERVATION_GAP_SECONDS,
        "minimum_prior_completed_positive_excursions": MIN_PRIOR_COMPLETED_EXCURSIONS,
        "minimum_prior_eligible_rate": MIN_PRIOR_ELIGIBLE_RATE,
        "entry_side": "positive residual / sell basis first",
        "event_semantics": "first crossing within non-overlapping zero excursion; not maker fill",
        "amplitude_quantile_sample": (
            "completed zero-crossing excursions only; censor counts are retained"
        ),
        "reach_rate_semantics": {
            "all_started": "conservative lower bound; censored non-hits remain in denominator",
            "completed": "completed-only conditional rate; subject to completion selection",
        },
        "path_probability_semantics": (
            "complete-case horizon only; events censored before H have null outcomes"
        ),
        "symmetric_label_semantics": (
            "overlapping path-hit diagnostic; not a position-cycle or turnover estimate"
        ),
        "prior_contract_semantics": "same exact target contract on prior trading day",
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return WidthTableResult(
        daily_product_parameters=parameters,
        product_parameter_summary=product_summary,
        next_day_parameter_validation=validation,
        validation_summary=validation_summary,
        potential_entry_events=events,
        width_policy_table=policy_table,
        width_policy_summary=policy_summary,
        route_geometry=route_geometry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prior-day product width parameters and next-day latent validation."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fair_anchor_panel.parquet",
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "prior_day",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_width_study(args.panel, args.prior_dir, args.output_dir)
    print(result.validation_summary)
    print(result.width_policy_summary)


if __name__ == "__main__":
    main()
