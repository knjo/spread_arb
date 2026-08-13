"""Previous-trading-day basis priors for the fair-mid anchor."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..common.contracts import load_exact_contract_mapping
from ..common.landmarks import build_landmarks
from ..common.paths import DEFAULT_OUTPUT_ROOT, futures_raw_path, spot_tick_path
from .anchors import CENTER_MIN_COVERAGE, GROUP_KEYS


EVALUATION_START_SECONDS = 300
CLOSE_30M_START_SECONDS = 13_800
MIN_PRIOR_COVERAGE = 0.80
BLEND_WEIGHTS = (0.10, 0.20, 0.35, 0.50)
FUTURES_FILE_MIN_BYTES = 1_000_000
PRIOR_COLUMNS = (
    "prior_mean_bp",
    "prior_median_bp",
    "prior_close30_median_bp",
)


def find_previous_common_data_date(target_date: str) -> str:
    """Find the latest earlier date with non-placeholder spot and futures data."""
    current = datetime.strptime(target_date, "%Y%m%d").date() - timedelta(days=1)
    for _ in range(40):
        candidate = current.strftime("%Y%m%d")
        future_path = futures_raw_path(candidate)
        if (
            spot_tick_path(candidate).exists()
            and future_path.exists()
            and future_path.stat().st_size >= FUTURES_FILE_MIN_BYTES
        ):
            return candidate
        current -= timedelta(days=1)
    raise RuntimeError(f"{target_date}: no previous common data date within 40 days")


def summarize_prior_landmarks(
    landmarks: pl.DataFrame,
    target_date: str,
    prior_date: str,
) -> pl.DataFrame:
    """Create time-weighted prior statistics from a legal one-second grid."""
    sample = landmarks.filter(pl.col("seconds_from_open") >= EVALUATION_START_SECONDS)
    return (
        sample.group_by(["ValueCode", "QuoteCode"])
        .agg(
            pl.len().alias("prior_grid_rows"),
            pl.col("eligible_base").sum().alias("prior_eligible_rows"),
            pl.col("eligible_1000ms").sum().alias("prior_fresh_1000ms_rows"),
            pl.col("basis_mid_bp")
            .filter(pl.col("eligible_base"))
            .mean()
            .alias("prior_mean_bp"),
            pl.col("basis_mid_bp")
            .filter(pl.col("eligible_base"))
            .median()
            .alias("prior_median_bp"),
            pl.col("basis_mid_bp")
            .filter(
                pl.col("eligible_base")
                & (pl.col("seconds_from_open") >= CLOSE_30M_START_SECONDS)
            )
            .median()
            .alias("prior_close30_median_bp"),
            pl.col("spot_age_ms")
            .filter(pl.col("eligible_base"))
            .quantile(0.95, interpolation="nearest")
            .alias("prior_spot_age_p95_ms"),
            pl.col("fut_age_ms")
            .filter(pl.col("eligible_base"))
            .quantile(0.95, interpolation="nearest")
            .alias("prior_fut_age_p95_ms"),
        )
        .with_columns(
            pl.lit(target_date).alias("Date"),
            pl.lit(prior_date).alias("prior_date"),
            (
                pl.col("prior_eligible_rows") / pl.col("prior_grid_rows")
            ).alias("prior_coverage"),
            (
                pl.col("prior_fresh_1000ms_rows")
                / pl.col("prior_grid_rows")
            ).alias("prior_fresh_1000ms_coverage"),
            pl.lit(
                (
                    datetime.strptime(target_date, "%Y%m%d").date()
                    - datetime.strptime(prior_date, "%Y%m%d").date()
                ).days
            ).alias("calendar_gap_days"),
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
            "prior_fresh_1000ms_coverage",
            *PRIOR_COLUMNS,
            "prior_spot_age_p95_ms",
            "prior_fut_age_p95_ms",
        )
        .sort(["Date", "ValueCode"])
    )


def _seeded_ewma(
    panel: pl.DataFrame,
    prior_column: str,
    output_column: str,
) -> pl.DataFrame:
    """Prepend one prior observation before the current session EWMA input."""
    timestamp_dtype = panel.schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime):
        raise ValueError("prior panel timestamp must be Datetime")
    time_unit = timestamp_dtype.time_unit
    seed = (
        panel.group_by(GROUP_KEYS)
        .agg(
            pl.col("timestamp").min().alias("timestamp"),
            pl.col(prior_column).drop_nulls().first().alias("ewma_input_bp"),
        )
        .with_columns(
            (
                pl.col("timestamp").dt.epoch(time_unit) - 1
            ).cast(timestamp_dtype).alias("timestamp"),
            pl.lit(True).alias("is_seed"),
        )
    )
    observations = panel.select(
        *GROUP_KEYS,
        "timestamp",
        pl.col("basis_eval_bp").alias("ewma_input_bp"),
        pl.lit(False).alias("is_seed"),
    )
    return (
        pl.concat([seed, observations])
        .sort([*GROUP_KEYS, "timestamp"])
        .with_columns(
            pl.col("ewma_input_bp")
            .ewm_mean(half_life=120, adjust=False, ignore_nulls=True)
            .over(GROUP_KEYS)
            .alias(output_column)
        )
        .filter(~pl.col("is_seed"))
        .select(*GROUP_KEYS, "timestamp", output_column)
    )


def add_prior_candidates(
    panel: pl.DataFrame,
    priors: pl.DataFrame,
) -> pl.DataFrame:
    """Join prior statistics and add seeded and fixed-weight anchor candidates."""
    joined = panel.join(
        priors,
        on=["Date", "ValueCode", "QuoteCode"],
        how="left",
        validate="m:1",
    ).with_columns(
        (
            (pl.col("prior_coverage") >= MIN_PRIOR_COVERAGE)
            & pl.col("prior_mean_bp").is_not_null()
            & pl.col("prior_median_bp").is_not_null()
        ).fill_null(False).alias("prior_valid")
    )
    joined = joined.with_columns(
        *[
            pl.when(pl.col("prior_valid"))
            .then(pl.col(column))
            .otherwise(None)
            .alias(column)
            for column in PRIOR_COLUMNS
        ]
    )
    for prior_column, output_column in (
        ("prior_mean_bp", "anchor_seeded_prior_mean_bp"),
        ("prior_median_bp", "anchor_seeded_prior_median_bp"),
    ):
        seeded = _seeded_ewma(joined, prior_column, output_column)
        joined = joined.join(
            seeded,
            on=[*GROUP_KEYS, "timestamp"],
            how="left",
            validate="1:1",
        )
    return joined.with_columns(
        *[
            (
                pl.col("anchor_ewma_120s_bp") * (1 - weight)
                + pl.col("prior_mean_bp") * weight
            ).alias(f"anchor_blend_prior_mean_{int(weight * 100)}pct_bp")
            for weight in BLEND_WEIGHTS
        ]
    )


def _time_bucket() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_from_open") < 900)
        .then(pl.lit("09:05-09:15"))
        .when(pl.col("seconds_from_open") < 3600)
        .then(pl.lit("09:15-10:00"))
        .otherwise(pl.lit("10:00-13:20"))
    )


def score_prior_candidates(panel: pl.DataFrame) -> pl.DataFrame:
    """Score all candidates on the same current-day label rows."""
    candidates = {
        "current_ewma120": "anchor_ewma_120s_bp",
        "prior_mean_static": "prior_mean_bp",
        "prior_median_static": "prior_median_bp",
        "prior_close30_static": "prior_close30_median_bp",
        "seeded_prior_mean": "anchor_seeded_prior_mean_bp",
        "seeded_prior_median": "anchor_seeded_prior_median_bp",
        **{
            f"blend_prior_mean_{int(weight * 100)}pct":
            f"anchor_blend_prior_mean_{int(weight * 100)}pct_bp"
            for weight in BLEND_WEIGHTS
        },
    }
    common = panel.filter(
        pl.col("analysis_eligible")
        & pl.col("prior_valid")
        & pl.col("future_center_bp").is_not_null()
        & (pl.col("future_center_coverage") >= CENTER_MIN_COVERAGE)
        & pl.all_horizontal(
            [pl.col(column).is_not_null() for column in candidates.values()]
        )
    ).with_columns(_time_bucket().alias("time_bucket"))
    frames = [common.with_columns(pl.lit("all_day").alias("sample"))]
    frames.append(common.with_columns(pl.col("time_bucket").alias("sample")))
    evaluation = pl.concat(frames)

    long = pl.concat(
        [
            evaluation.select(
                "sample",
                "Date",
                "ValueCode",
                "QuoteCode",
                "timestamp",
                "future_center_bp",
                pl.lit(model).alias("model"),
                pl.col(column).alias("anchor_bp"),
            )
            for model, column in candidates.items()
        ]
    ).with_columns(
        (pl.col("anchor_bp") - pl.col("future_center_bp")).alias("error_bp")
    )
    return (
        long.group_by(["sample", "model"])
        .agg(
            pl.len().alias("n"),
            pl.col("Date").n_unique().alias("dates"),
            pl.struct("Date", "ValueCode").n_unique().alias("date_symbol_pairs"),
            pl.col("error_bp").mean().alias("bias_bp"),
            pl.col("error_bp").abs().mean().alias("mae_bp"),
            pl.col("error_bp")
            .abs()
            .quantile(0.80, interpolation="nearest")
            .alias("p80_abs_error_bp"),
            pl.col("error_bp")
            .abs()
            .quantile(0.95, interpolation="nearest")
            .alias("p95_abs_error_bp"),
        )
        .sort(["sample", "model"])
    )


def score_seeded_prior_pairs(panel: pl.DataFrame) -> pl.DataFrame:
    """Create paired baseline-versus-seeded metrics for audit and robustness."""
    common = panel.filter(
        pl.col("analysis_eligible")
        & pl.col("prior_valid")
        & pl.col("future_center_bp").is_not_null()
        & (pl.col("future_center_coverage") >= CENTER_MIN_COVERAGE)
        & pl.col("anchor_ewma_120s_bp").is_not_null()
        & pl.col("anchor_seeded_prior_mean_bp").is_not_null()
    ).with_columns(
        (pl.col("anchor_ewma_120s_bp") - pl.col("future_center_bp"))
        .abs()
        .alias("baseline_abs_error_bp"),
        (
            pl.col("anchor_seeded_prior_mean_bp")
            - pl.col("future_center_bp")
        )
        .abs()
        .alias("seeded_abs_error_bp"),
    )
    frames = [common.with_columns(pl.lit("all_day").alias("sample"))]
    frames.append(
        common.filter(pl.col("seconds_from_open") < 900).with_columns(
            pl.lit("09:05-09:15").alias("sample")
        )
    )
    paired = pl.concat(frames)
    return (
        paired.group_by(["sample", "Date", "ValueCode", "QuoteCode"])
        .agg(
            pl.len().alias("n"),
            pl.col("baseline_abs_error_bp").mean().alias("baseline_mae_bp"),
            pl.col("seeded_abs_error_bp").mean().alias("seeded_mae_bp"),
            pl.col("baseline_abs_error_bp")
            .quantile(0.80, interpolation="nearest")
            .alias("baseline_p80_bp"),
            pl.col("seeded_abs_error_bp")
            .quantile(0.80, interpolation="nearest")
            .alias("seeded_p80_bp"),
            pl.col("baseline_abs_error_bp")
            .quantile(0.95, interpolation="nearest")
            .alias("baseline_p95_bp"),
            pl.col("seeded_abs_error_bp")
            .quantile(0.95, interpolation="nearest")
            .alias("seeded_p95_bp"),
        )
        .with_columns(
            (pl.col("seeded_mae_bp") - pl.col("baseline_mae_bp")).alias(
                "delta_mae_bp"
            ),
            (pl.col("seeded_p80_bp") - pl.col("baseline_p80_bp")).alias(
                "delta_p80_bp"
            ),
            (pl.col("seeded_p95_bp") - pl.col("baseline_p95_bp")).alias(
                "delta_p95_bp"
            ),
        )
        .sort(["sample", "Date", "ValueCode"])
    )


def run_prior_study(
    panel_path: Path,
    output_dir: Path,
) -> pl.DataFrame:
    """Build exact-contract priors, candidates, and comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(panel_path)
    target_pairs = panel.select("Date", "ValueCode", "QuoteCode").unique()
    prior_frames: list[pl.DataFrame] = []
    for target_date in target_pairs["Date"].unique().sort().to_list():
        targets = target_pairs.filter(pl.col("Date") == target_date).select(
            "ValueCode", "QuoteCode"
        )
        prior_date = find_previous_common_data_date(target_date)
        mapping = load_exact_contract_mapping(prior_date, targets)
        result = build_landmarks(
            prior_date,
            interval="1s",
            mapping_override=mapping,
        )
        result.landmarks.write_parquet(
            output_dir / f"prior_landmarks_{prior_date}_for_{target_date}.parquet"
        )
        prior_frames.append(
            summarize_prior_landmarks(
                result.landmarks,
                target_date=target_date,
                prior_date=prior_date,
            )
        )

    priors = pl.concat(prior_frames).sort(["Date", "ValueCode"])
    study_panel = add_prior_candidates(panel, priors)
    metrics = score_prior_candidates(study_panel)
    pair_metrics = score_seeded_prior_pairs(study_panel)
    priors.write_csv(output_dir / "prior_summary.csv")
    study_panel.write_parquet(output_dir / "prior_anchor_panel.parquet")
    metrics.write_csv(output_dir / "prior_metrics.csv")
    pair_metrics.write_csv(output_dir / "prior_metrics_by_pair.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-contract previous-day basis priors."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fair_anchor_panel.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "prior_day",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run_prior_study(args.panel, args.output_dir))


if __name__ == "__main__":
    main()
