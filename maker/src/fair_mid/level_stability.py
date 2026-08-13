"""Descriptive stability diagnostics for fair-mid level regimes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..common.paths import DEFAULT_OUTPUT_ROOT
from .anchors import CENTER_MIN_COVERAGE
from .metrics import CHANGE_EPS_BP, _with_exact_future_state


QUANTILES = (0.20, 0.40, 0.60, 0.80)
QUINTILE_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")


@dataclass(frozen=True)
class LevelStabilityResult:
    global_abs_level: pl.DataFrame
    local_level: pl.DataFrame
    local_pair_comparison: pl.DataFrame
    fast_slow_gap: pl.DataFrame
    gap_direction: pl.DataFrame


def _cutoffs(frame: pl.DataFrame, column: str) -> tuple[float, ...]:
    values = frame.select(
        *[
            pl.col(column).quantile(value).alias(str(value))
            for value in QUANTILES
        ]
    ).row(0)
    return tuple(float(value) for value in values)


def _quintile(column: str, cutoffs: tuple[float, ...]) -> pl.Expr:
    return (
        pl.when(pl.col(column) <= cutoffs[0])
        .then(pl.lit(QUINTILE_LABELS[0]))
        .when(pl.col(column) <= cutoffs[1])
        .then(pl.lit(QUINTILE_LABELS[1]))
        .when(pl.col(column) <= cutoffs[2])
        .then(pl.lit(QUINTILE_LABELS[2]))
        .when(pl.col(column) <= cutoffs[3])
        .then(pl.lit(QUINTILE_LABELS[3]))
        .otherwise(pl.lit(QUINTILE_LABELS[4]))
    )


def _stability_aggregations() -> list[pl.Expr]:
    return [
        pl.len().alias("n"),
        pl.col("Date").n_unique().alias("dates"),
        pl.struct("Date", "ValueCode").n_unique().alias("date_symbol_pairs"),
        pl.col("anchor_abs_error_bp").mean().alias("mae_bp"),
        pl.col("anchor_abs_error_bp")
        .quantile(0.80, interpolation="nearest")
        .alias("p80_abs_error_bp"),
        pl.col("anchor_abs_error_bp")
        .quantile(0.95, interpolation="nearest")
        .alias("p95_abs_error_bp"),
        pl.col("future_anchor_distance_bp")
        .median()
        .alias("median_future_anchor_distance_bp"),
        pl.col("basis_move_300s_bp").median().alias("median_basis_move_300s_bp"),
    ]


def analyze_level_stability(panel: pl.DataFrame) -> LevelStabilityResult:
    """Compare raw level, local level rank, and fast/slow disagreement regimes."""
    endpoint = _with_exact_future_state(panel, 300, "level_end")
    evaluation = (
        endpoint.filter(
            pl.col("analysis_eligible")
            & pl.col("future_center_bp").is_not_null()
            & (pl.col("future_center_coverage") >= CENTER_MIN_COVERAGE)
            & pl.col("anchor_ewma_120s_bp").is_not_null()
        )
        .with_columns(
            pl.col("anchor_ewma_120s_bp").abs().alias("abs_anchor_level_bp"),
            (
                pl.col("anchor_ewma_120s_bp")
                - pl.col("future_center_bp")
            )
            .abs()
            .alias("anchor_abs_error_bp"),
            (
                pl.col("basis_mid_bp_level_end")
                - pl.col("anchor_ewma_120s_bp")
            )
            .abs()
            .alias("future_anchor_distance_bp"),
            (
                pl.col("basis_mid_bp_level_end") - pl.col("basis_mid_bp")
            )
            .abs()
            .alias("basis_move_300s_bp"),
            (
                pl.col("anchor_ewma_120s_bp")
                - pl.col("anchor_ewma_300s_bp")
            ).alias("fast_slow_gap_bp"),
            (pl.col("seconds_from_open") // 1800).alias("half_hour_block"),
        )
        .with_columns(
            pl.col("fast_slow_gap_bp").abs().alias("abs_fast_slow_gap_bp")
        )
    )

    absolute_cutoffs = _cutoffs(evaluation, "abs_anchor_level_bp")
    global_rows = evaluation.with_columns(
        _quintile("abs_anchor_level_bp", absolute_cutoffs).alias("level_quintile")
    )
    global_abs_level = (
        global_rows.group_by("level_quintile")
        .agg(
            pl.col("abs_anchor_level_bp").min().alias("level_min_bp"),
            pl.col("abs_anchor_level_bp").max().alias("level_max_bp"),
            pl.col("abs_anchor_level_bp").median().alias("median_level_bp"),
            *_stability_aggregations(),
        )
        .sort("level_quintile")
    )

    local_group = ["Date", "ValueCode", "half_hour_block"]
    local_rows = evaluation.with_columns(
        (
            pl.col("anchor_ewma_120s_bp").rank("average").over(local_group)
            / pl.len().over(local_group)
        ).alias("local_level_rank")
    ).with_columns(
        _quintile("local_level_rank", QUANTILES).alias("local_level_quintile")
    )
    local_level = (
        local_rows.group_by("local_level_quintile")
        .agg(
            pl.col("local_level_rank").min().alias("rank_min"),
            pl.col("local_level_rank").max().alias("rank_max"),
            *_stability_aggregations(),
        )
        .sort("local_level_quintile")
    )
    local_pair = (
        local_rows.group_by(["Date", "ValueCode", "local_level_quintile"])
        .agg(
            pl.col("anchor_abs_error_bp")
            .quantile(0.80, interpolation="nearest")
            .alias("p80_abs_error_bp")
        )
        .pivot(
            on="local_level_quintile",
            index=["Date", "ValueCode"],
            values="p80_abs_error_bp",
        )
        .with_columns(
            (pl.col("Q1") - pl.col("Q3")).alias("low_minus_middle_p80_bp"),
            (pl.col("Q5") - pl.col("Q3")).alias("high_minus_middle_p80_bp"),
        )
        .sort(["Date", "ValueCode"])
    )

    gap_cutoffs = _cutoffs(evaluation, "abs_fast_slow_gap_bp")
    gap_rows = evaluation.with_columns(
        _quintile("abs_fast_slow_gap_bp", gap_cutoffs).alias("gap_quintile")
    )
    fast_slow_gap = (
        gap_rows.group_by("gap_quintile")
        .agg(
            pl.col("abs_fast_slow_gap_bp").min().alias("gap_min_bp"),
            pl.col("abs_fast_slow_gap_bp").max().alias("gap_max_bp"),
            pl.col("abs_fast_slow_gap_bp").median().alias("median_gap_bp"),
            *_stability_aggregations(),
        )
        .sort("gap_quintile")
    )

    delayed = _with_exact_future_state(panel, 30, "gap_start")
    delayed = _with_exact_future_state(delayed, 300, "gap_end")
    gap_direction_rows = (
        delayed.with_columns(
            (
                pl.col("anchor_ewma_120s_bp")
                - pl.col("anchor_ewma_300s_bp")
            ).alias("fast_slow_gap_bp")
        )
        .filter(
            (pl.col("seconds_from_open") >= 300)
            & pl.col("eligible_1000ms")
            & pl.col("eligible_1000ms_gap_start")
            & pl.col("eligible_1000ms_gap_end")
            & pl.col("fast_slow_gap_bp").is_not_null()
            & (pl.col("fast_slow_gap_bp").abs() > CHANGE_EPS_BP)
        )
        .with_columns(
            pl.col("fast_slow_gap_bp").abs().alias("abs_fast_slow_gap_bp"),
            (
                pl.col("fast_slow_gap_bp").sign()
                * (
                    pl.col("basis_mid_bp_gap_start")
                    - pl.col("basis_mid_bp_gap_end")
                )
            ).alias("signal_consistent_move_bp"),
            pl.when(pl.col("fast_slow_gap_bp") > 0)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
            .alias("gap_side"),
        )
        .with_columns(
            _quintile("abs_fast_slow_gap_bp", gap_cutoffs).alias("gap_quintile")
        )
    )
    directional = gap_direction_rows.select(
        "Date",
        "ValueCode",
        "gap_quintile",
        "gap_side",
        "signal_consistent_move_bp",
    )
    direction_rows = pl.concat(
        [
            directional,
            directional.with_columns(pl.lit("all").alias("gap_side")),
        ]
    )
    gap_direction = (
        direction_rows.group_by(["gap_quintile", "gap_side"])
        .agg(
            pl.len().alias("n"),
            pl.col("Date").n_unique().alias("dates"),
            pl.struct("Date", "ValueCode").n_unique().alias("date_symbol_pairs"),
            pl.col("signal_consistent_move_bp")
            .mean()
            .alias("mean_signal_consistent_move_bp"),
            (pl.col("signal_consistent_move_bp") > CHANGE_EPS_BP)
            .mean()
            .alias("p_signal_consistent"),
            (pl.col("signal_consistent_move_bp").abs() <= CHANGE_EPS_BP)
            .mean()
            .alias("p_flat"),
            (
                (pl.col("signal_consistent_move_bp") > CHANGE_EPS_BP).sum()
                / (
                    pl.col("signal_consistent_move_bp").abs()
                    > CHANGE_EPS_BP
                ).sum()
            ).alias("p_signal_consistent_given_move"),
        )
        .sort(["gap_quintile", "gap_side"])
    )
    return LevelStabilityResult(
        global_abs_level=global_abs_level,
        local_level=local_level,
        local_pair_comparison=local_pair,
        fast_slow_gap=fast_slow_gap,
        gap_direction=gap_direction,
    )


def run_level_study(panel_path: Path, output_dir: Path) -> LevelStabilityResult:
    """Run and persist all level-stability diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_level_stability(pl.read_parquet(panel_path))
    result.global_abs_level.write_csv(output_dir / "global_abs_level.csv")
    result.local_level.write_csv(output_dir / "local_level_quintiles.csv")
    result.local_pair_comparison.write_csv(
        output_dir / "local_level_pair_comparison.csv"
    )
    result.fast_slow_gap.write_csv(output_dir / "fast_slow_gap.csv")
    result.gap_direction.write_csv(output_dir / "fast_slow_gap_direction.csv")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze absolute and relative fair-mid level stability."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fair_anchor_panel.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "level_stability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_level_study(args.panel, args.output_dir)
    print(result.global_abs_level)
    print(result.local_level)
    print(result.fast_slow_gap)
    print(result.gap_direction)


if __name__ == "__main__":
    main()
