"""Command-line pilot for stable futures/spot fair-mid anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ..common.landmarks import build_landmarks
from ..common.paths import DEFAULT_OUTPUT_ROOT
from .anchors import prepare_fair_panel
from .metrics import (
    summarize_delayed_reversion,
    summarize_fair_panel,
    summarize_fresh_endpoint_reversion,
)
from .quote_churn import DEFAULT_OPEN_WIDTH_BP, summarize_quote_churn


DEFAULT_SYMBOLS = ("2303", "2317", "2603", "2881")
FRESHNESS_SENSITIVITY = (100, 250, 500, 1000, 5000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal one-second basis landmarks and compare fair-mid anchors."
    )
    parser.add_argument("--dates", nargs="+", required=True, help="YYYYMMDD dates")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="spot ValueCode list",
    )
    parser.add_argument(
        "--interval",
        default="1s",
        choices=("1s",),
        help="fixed-grid interval; WP01 labels are currently defined for 1s only",
    )
    parser.add_argument(
        "--primary-age-ms",
        type=int,
        default=None,
        help="optional hard age gate for anchor input; default uses every valid formal book",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--open-width-bp", type=float, default=DEFAULT_OPEN_WIDTH_BP)
    return parser.parse_args()


def run_pilot(
    dates: list[str],
    symbols: list[str],
    interval: str,
    primary_age_ms: int | None,
    output_dir: Path,
    open_width_bp: float = DEFAULT_OPEN_WIDTH_BP,
) -> pl.DataFrame:
    if interval != "1s":
        raise ValueError("WP01 forward labels currently require interval='1s'")
    output_dir.mkdir(parents=True, exist_ok=True)
    landmark_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    mapping_frames: list[pl.DataFrame] = []

    for date in dates:
        result = build_landmarks(
            date,
            value_codes=symbols,
            interval=interval,
            cache_dir=output_dir / "metadata",
        )
        result.landmarks.write_parquet(output_dir / f"basis_landmarks_{date}.parquet")
        landmark_frames.append(result.landmarks)
        audit_frames.append(result.audit)
        mapping_frames.append(result.mapping.with_columns(pl.lit(date).alias("Date")))

    landmarks = pl.concat(landmark_frames, how="diagonal_relaxed")
    panel = prepare_fair_panel(landmarks, primary_age_ms=primary_age_ms)
    metrics = summarize_fair_panel(panel)
    endpoint_freshness = summarize_fresh_endpoint_reversion(panel)
    delayed_reversion = summarize_delayed_reversion(panel)
    quote_churn = summarize_quote_churn(panel, open_width_bp=open_width_bp)

    panel.write_parquet(output_dir / "fair_anchor_panel.parquet")
    pl.concat(audit_frames, how="diagonal_relaxed").write_csv(output_dir / "landmark_audit.csv")
    pl.concat(mapping_frames, how="diagonal_relaxed").write_csv(output_dir / "contract_mapping.csv")
    metrics.metrics_by_model.write_csv(output_dir / "metrics_by_model.csv")
    metrics.metrics_by_day_symbol.write_csv(output_dir / "metrics_by_day_symbol.csv")
    metrics.residual_bins.write_csv(output_dir / "residual_bins.csv")
    metrics.coverage.write_csv(output_dir / "coverage.csv")
    endpoint_freshness.summary.write_csv(output_dir / "endpoint_freshness.csv")
    endpoint_freshness.residual_bins.write_csv(
        output_dir / "endpoint_residual_bins.csv"
    )
    delayed_reversion.summary.write_csv(output_dir / "delayed_reversion.csv")
    delayed_reversion.residual_bins.write_csv(
        output_dir / "delayed_reversion_bins.csv"
    )
    quote_churn.by_model_route.write_csv(output_dir / "quote_churn_by_model_route.csv")
    quote_churn.by_day_symbol_route.write_csv(
        output_dir / "quote_churn_by_day_symbol_route.csv"
    )

    sensitivity_models: list[pl.DataFrame] = []
    sensitivity_residuals: list[pl.DataFrame] = []
    sensitivity_specs = [("eligible_base", None, None)] + [
        (f"age_{age_ms}ms", age_ms, None)
        for age_ms in FRESHNESS_SENSITIVITY
    ] + [
        ("age_1000ms_skew_100ms", 1000, 100),
        ("age_1000ms_skew_250ms", 1000, 250),
    ]
    for sample, age_ms, skew_ms in sensitivity_specs:
        eligibility_column = (
            "eligible_base" if age_ms is None else f"eligible_{age_ms}ms"
        )
        sensitivity = summarize_fair_panel(
            panel,
            eligibility_column=eligibility_column,
            max_leg_skew_ms=skew_ms,
        )
        sensitivity_models.append(
            sensitivity.metrics_by_model.with_columns(pl.lit(sample).alias("sample"))
        )
        sensitivity_residuals.append(
            sensitivity.residual_bins.with_columns(pl.lit(sample).alias("sample"))
        )
    pl.concat(sensitivity_models, how="diagonal_relaxed").write_csv(
        output_dir / "metrics_by_sample.csv"
    )
    pl.concat(sensitivity_residuals, how="diagonal_relaxed").write_csv(
        output_dir / "residual_bins_by_sample.csv"
    )
    config = {
        "dates": dates,
        "symbols": symbols,
        "interval": interval,
        "primary_age_ms": primary_age_ms,
        "open_width_bp": open_width_bp,
        "endpoint_diagnostic_seconds": 300,
        "delayed_reversion_start_seconds": 30,
        "delayed_reversion_end_seconds": 300,
    }
    (output_dir / "pilot_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics.metrics_by_model


def main() -> None:
    args = parse_args()
    summary = run_pilot(
        dates=args.dates,
        symbols=args.symbols,
        interval=args.interval,
        primary_age_ms=args.primary_age_ms,
        output_dir=args.output_dir,
        open_width_bp=args.open_width_bp,
    )
    print(summary)


if __name__ == "__main__":
    main()
