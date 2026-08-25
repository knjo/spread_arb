"""Build and verify the S0.5 lookup-foundation revalidation bundle.

The runner deliberately stops at the last development/pseudo-holdout session
(``20260813``).  It validates every named daily partition, processes one day at
a time, and never discovers partitions with a directory glob.  The protected
forward period therefore cannot be pulled into this study by a later data
backfill.

S0.5 evaluates three separate contracts:

* intraday fair-mid anchors, including an exact-contract prior-day seed;
* D-safe rolling excursion boundaries with explicit censor bounds; and
* pre-open seven-policy geometry against known fees and taxes only.

The geometry is a necessary screen before raw execution replay.  It does not
contain maker fills, B6 hedge prices, exits, terminal inventory, or EV.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import validate_completion_marker

RUNNER_VERSION = "foundation_revalidation_s05_v1"
SCHEMA_VERSION = "foundation_revalidation_s05_bundle_v1"
SOURCE_START_DATE = "20260126"
SOURCE_END_DATE = "20260813"
PROTECTED_FORWARD_START_DATE = "20260814"
PRIMARY_START_DATE = "20260505"
EXPECTED_SESSION_COUNT = 131
EXPECTED_PRIMARY_SESSION_COUNT = 71
EXPECTED_CAUSAL_ROWS = 489_216_000
EXPECTED_PRODUCT_DAYS = 31_360
EXPECTED_ALL_BOUNDARY_SUPPORTED_PRODUCT_DAYS = 16_656
EXPECTED_BROAD_PRODUCT_DAYS = 15_935
EXPECTED_BROAD_PRODUCTS = 244

DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
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
DEFAULT_LEGACY_MANIFEST_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "monthly_product_selector_causal_v2_20260822"
    / "daily_entry_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "foundation_revalidation_s05_20260825_v1"
)

ANCHOR_ARTIFACT = "anchor_daily_product.parquet"
ANCHOR_SUMMARY_ARTIFACT = "anchor_summary.csv"
ANCHOR_BOOTSTRAP_ARTIFACT = "anchor_whole_date_bootstrap.csv"
ANCHOR_COVERAGE_ARTIFACT = "anchor_coverage_summary.csv"
DELAYED_ARTIFACT = "delayed_reversion_daily_product.parquet"
DELAYED_SUMMARY_ARTIFACT = "delayed_reversion_summary.csv"
BOUNDARY_WIDE_ARTIFACT = "boundary_predictions_wide.parquet"
BROAD_COHORT_ARTIFACT = "broad_dsafe_cohort.parquet"
CALIBRATION_ARTIFACT = "calibration_product_day.parquet"
CALIBRATION_SUMMARY_ARTIFACT = "calibration_summary.csv"
CALIBRATION_MONTH_ARTIFACT = "calibration_by_month.csv"
CALIBRATION_BOOTSTRAP_ARTIFACT = "calibration_whole_date_bootstrap.csv"
RANK_ARTIFACT = "rank_validation_by_date.csv"
CENSOR_SUMMARY_ARTIFACT = "censor_sensitivity.csv"
GEOMETRY_ARTIFACT = "policy_geometry_preopen.parquet"
GEOMETRY_SUMMARY_ARTIFACT = "geometry_cost_summary.csv"
COHORT_FUNNEL_ARTIFACT = "cohort_funnel.csv"
LEGACY_BRIDGE_ARTIFACT = "legacy_selector_bridge.csv"
INPUT_AUDIT_ARTIFACT = "input_audit.csv"
RUN_CONFIG_ARTIFACT = "run_config.json"
VERIFICATION_ARTIFACT = "verification.json"
COMPLETE_ARTIFACT = "complete.json"
EPISODE_ROOT = "censor_excursions"

TOP_LEVEL_FRAME_ARTIFACTS = (
    ANCHOR_ARTIFACT,
    ANCHOR_SUMMARY_ARTIFACT,
    ANCHOR_BOOTSTRAP_ARTIFACT,
    ANCHOR_COVERAGE_ARTIFACT,
    DELAYED_ARTIFACT,
    DELAYED_SUMMARY_ARTIFACT,
    BOUNDARY_WIDE_ARTIFACT,
    BROAD_COHORT_ARTIFACT,
    CALIBRATION_ARTIFACT,
    CALIBRATION_SUMMARY_ARTIFACT,
    CALIBRATION_MONTH_ARTIFACT,
    CALIBRATION_BOOTSTRAP_ARTIFACT,
    RANK_ARTIFACT,
    CENSOR_SUMMARY_ARTIFACT,
    GEOMETRY_ARTIFACT,
    GEOMETRY_SUMMARY_ARTIFACT,
    COHORT_FUNNEL_ARTIFACT,
    LEGACY_BRIDGE_ARTIFACT,
    INPUT_AUDIT_ARTIFACT,
)


@dataclass(frozen=True)
class FoundationPaths:
    """Frozen input and output locations for one S0.5 run."""

    sessions_path: Path = DEFAULT_SESSIONS_PATH
    daily_root: Path = DEFAULT_DAILY_ROOT
    boundary_path: Path = DEFAULT_BOUNDARY_PATH
    liquidity_path: Path = DEFAULT_LIQUIDITY_PATH
    legacy_manifest_path: Path = DEFAULT_LEGACY_MANIFEST_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT


def load_frozen_sessions(path: Path) -> list[str]:
    """Load the exact development calendar and reject protected-forward rows."""

    rows = [line.strip() for line in Path(path).read_text().splitlines()]
    sessions = [value for value in rows if value]
    if len(sessions) != len(set(sessions)) or sessions != sorted(sessions):
        raise ValueError("session calendar must be unique and sorted")
    if any(len(value) != 8 or not value.isdigit() for value in sessions):
        raise ValueError("session calendar must contain YYYYMMDD values")
    protected = [
        value for value in sessions if value >= PROTECTED_FORWARD_START_DATE
    ]
    if protected:
        raise ValueError(
            "S0.5 session calendar includes protected forward dates: "
            f"{protected[:5]}"
        )
    if (
        len(sessions) != EXPECTED_SESSION_COUNT
        or not sessions
        or sessions[0] != SOURCE_START_DATE
        or sessions[-1] != SOURCE_END_DATE
    ):
        raise ValueError(
            "S0.5 frozen session contract drift: expected "
            f"{EXPECTED_SESSION_COUNT} sessions from {SOURCE_START_DATE} "
            f"through {SOURCE_END_DATE}"
        )
    primary = [value for value in sessions if value >= PRIMARY_START_DATE]
    if len(primary) != EXPECTED_PRIMARY_SESSION_COUNT:
        raise ValueError("S0.5 full-60 primary session count drift")
    return sessions


def stage_for_date(date: str) -> str:
    """Return the frozen development-stage label for a 2026 session."""

    if date < PRIMARY_START_DATE:
        return "history"
    if date <= "20260529":
        return "fine_tune"
    if date <= "20260630":
        return "confirmation"
    if date <= SOURCE_END_DATE:
        return "pseudo_holdout"
    raise ValueError(f"date is outside the S0.5 contract: {date}")


def validate_named_daily_partitions(
    daily_root: Path,
    sessions: Sequence[str],
) -> tuple[list[dict[str, object]], list[Path]]:
    """Validate only explicitly frozen partitions and return their input paths."""

    audits: list[dict[str, object]] = []
    input_paths: list[Path] = []
    causal_rows = 0
    product_days = 0
    for date in sessions:
        if date >= PROTECTED_FORWARD_START_DATE:
            raise ValueError(f"protected forward partition requested: {date}")
        partition = Path(daily_root) / f"Date={date}"
        marker_path = partition / "complete.json"
        marker = validate_completion_marker(marker_path)
        if str(marker.get("date")) != date:
            raise ValueError(f"daily marker Date mismatch: {marker_path}")
        if marker.get("contains_forward_labels") is not False:
            raise ValueError(f"daily causal marker contains forward labels: {date}")
        if marker.get("execution_safe_causal_panel") is not True:
            raise ValueError(f"daily causal marker is not execution safe: {date}")
        causal_path = partition / "causal_fair.parquet"
        excursion_path = partition / "excursions.parquet"
        mapping_path = partition / "mapping.parquet"
        audit_path = partition / "audit.parquet"
        causal_n = int(marker["artifacts"]["causal_fair.parquet"]["rows"])
        excursion_n = int(marker["artifacts"]["excursions.parquet"]["rows"])
        products = int(marker["products"])
        causal_rows += causal_n
        product_days += products
        audits.append(
            {
                "Date": date,
                "stage": stage_for_date(date),
                "products": products,
                "causal_rows": causal_n,
                "legacy_excursion_rows": excursion_n,
                "marker_schema_version": str(marker["schema_version"]),
                "migrated_legacy_marker": bool(
                    marker.get("migrated_legacy_marker", False)
                ),
                "legacy_atomic_publish_verified": bool(
                    marker.get("legacy_atomic_publish_verified", False)
                ),
                "legacy_writer_provenance_verified": bool(
                    marker.get("legacy_writer_provenance_verified", False)
                ),
            }
        )
        input_paths.extend(
            [marker_path, causal_path, excursion_path, mapping_path, audit_path]
        )
    if causal_rows != EXPECTED_CAUSAL_ROWS:
        raise ValueError(
            f"frozen causal row count drift: {causal_rows} != "
            f"{EXPECTED_CAUSAL_ROWS}"
        )
    if product_days != EXPECTED_PRODUCT_DAYS:
        raise ValueError(
            f"frozen product-day count drift: {product_days} != "
            f"{EXPECTED_PRODUCT_DAYS}"
        )
    return audits, input_paths


def whole_date_bootstrap(
    daily: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    seed: int = 20_260_825,
    replicates: int = 1_000,
) -> pl.DataFrame:
    """Date-block bootstrap of product-day equal means.

    The input must already contain one row per independent reporting unit and
    Date.  A replicate resamples whole Dates; every product-day from a sampled
    Date receives the same multiplicity.  Row counts are never interpreted as
    independent one-second observations.
    """

    required = {"Date", *group_columns, *metric_columns}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"bootstrap input missing columns: {missing}")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rows: list[dict[str, object]] = []
    grouped = daily.group_by(list(group_columns), maintain_order=True)
    for raw_key, group in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        dates = sorted(group["Date"].unique().to_list())
        if not dates:
            continue
        by_date = group.group_by("Date").agg(
            *[pl.col(metric).mean().alias(metric) for metric in metric_columns]
        )
        date_values = {
            str(row["Date"]): {
                metric: row[metric] for metric in metric_columns
            }
            for row in by_date.iter_rows(named=True)
        }
        group_seed = seed + int(
            hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:8], 16
        )
        rng = random.Random(group_seed)
        samples: dict[str, list[float]] = {
            metric: [] for metric in metric_columns
        }
        for _ in range(replicates):
            selected = [rng.choice(dates) for _ in dates]
            for metric in metric_columns:
                values = [
                    date_values[date][metric]
                    for date in selected
                    if date_values[date][metric] is not None
                ]
                if values:
                    samples[metric].append(
                        sum(float(value) for value in values) / len(values)
                    )
        for metric in metric_columns:
            observed = [
                date_values[date][metric]
                for date in dates
                if date_values[date][metric] is not None
            ]
            estimates = sorted(samples[metric])
            rows.append(
                {
                    **dict(zip(group_columns, key, strict=True)),
                    "metric": metric,
                    "estimate": (
                        sum(float(value) for value in observed) / len(observed)
                        if observed
                        else None
                    ),
                    "ci95_low": _ordered_quantile(estimates, 0.025),
                    "ci95_high": _ordered_quantile(estimates, 0.975),
                    "dates": len(dates),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "resampling_unit": "whole_Date",
                    "weighting": "Date_equal_after_product_day_equal",
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _ordered_quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be in [0, 1]")
    index = round((len(values) - 1) * probability)
    return float(values[index])


def summarize_calibration(
    calibration: pl.DataFrame,
    *,
    extra_groups: Sequence[str] = (),
) -> pl.DataFrame:
    """Summarize censor-identified reach with explicit weighting."""

    groups = [*extra_groups, "boundary_quantile", "side"]
    required = {
        *groups,
        "Date",
        "n_started",
        "n_hit",
        "n_known_miss",
        "n_unknown_censored",
        "reach_lower_bound",
        "reach_upper_bound",
        "complete_case_reach",
        "predicted_distance_bp",
        "realized_completed_quantile_bp",
    }
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"calibration input missing columns: {missing}")
    result = calibration.group_by(groups).agg(
        pl.len().alias("product_days"),
        (pl.col("n_started") > 0).sum().alias("observable_product_days"),
        (pl.col("n_started") == 0).sum().alias(
            "no_observable_excursion_product_days"
        ),
        pl.col("Date").n_unique().alias("dates"),
        pl.col("ValueCode").n_unique().alias("products"),
        pl.col("n_started").sum().alias("pooled_started"),
        pl.col("n_hit").sum().alias("pooled_hits"),
        pl.col("n_known_miss").sum().alias("pooled_known_misses"),
        pl.col("n_unknown_censored").sum().alias("pooled_unknown_censored"),
        pl.col("reach_lower_bound").mean().alias(
            "product_day_equal_reach_lower_bound"
        ),
        pl.col("reach_upper_bound").mean().alias(
            "product_day_equal_reach_upper_bound"
        ),
        pl.col("complete_case_reach").mean().alias(
            "product_day_equal_complete_case_reach"
        ),
        pl.col("predicted_distance_bp").median().alias(
            "predicted_distance_bp_median"
        ),
        pl.col("realized_completed_quantile_bp").median().alias(
            "realized_completed_quantile_bp_median"
        ),
        (
            pl.col("predicted_distance_bp")
            - pl.col("realized_completed_quantile_bp")
        ).median().alias("predicted_minus_realized_bp_median"),
        (
            pl.col("predicted_distance_bp")
            - pl.col("realized_completed_quantile_bp")
        ).abs().median().alias("predicted_realized_abs_error_bp_median"),
    )
    nominal = (100 - pl.col("boundary_quantile")) / 100
    return result.with_columns(
        pl.when(pl.col("pooled_started") > 0)
        .then(pl.col("pooled_hits") / pl.col("pooled_started"))
        .otherwise(None)
        .alias("event_pooled_reach_lower_bound"),
        pl.when(pl.col("pooled_started") > 0)
        .then(
            (pl.col("pooled_hits") + pl.col("pooled_unknown_censored"))
            / pl.col("pooled_started")
        )
        .otherwise(None)
        .alias("event_pooled_reach_upper_bound"),
        nominal.alias("nominal_tail_probability"),
        (
            pl.col("product_day_equal_reach_lower_bound") - nominal
        ).alias("product_day_equal_lower_calibration_error"),
        (
            pl.col("product_day_equal_reach_upper_bound") - nominal
        ).alias("product_day_equal_upper_calibration_error"),
        pl.lit("product_day_equal_primary_event_pooled_audit").alias(
            "weighting_contract"
        ),
    ).sort(groups)


def summarize_delayed_reversion(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Aggregate positive-residual delayed diagnostics from daily units."""

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    columns = set(lazy.collect_schema().names())
    groups = [
        column
        for column in (
            "model",
            "freshness_sample",
            "stratum_family",
            "stratum_value",
            "stage",
        )
        if column in columns
    ]
    required = {
        *groups,
        "Date",
        "n_positive_occupancy",
        "occupancy_signal_consistent_count",
        "occupancy_flat_count",
        "occupancy_sum_signal_consistent_move_bp",
        "n_positive_nonoverlap",
        "nonoverlap_signal_consistent_count",
        "nonoverlap_flat_count",
        "nonoverlap_sum_signal_consistent_move_bp",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"delayed reversion input missing columns: {missing}")
    outputs: list[pl.DataFrame] = []
    for inference_unit, prefix in (
        ("occupancy_seconds", "occupancy"),
        ("nonoverlap_300s_lockout", "nonoverlap"),
    ):
        outputs.append(
            lazy.group_by(groups)
            .agg(
                pl.col("Date").n_unique().alias("dates"),
                pl.col("ValueCode").n_unique().alias("products"),
                pl.col(f"n_positive_{prefix}").sum().alias("n"),
                pl.col(f"{prefix}_signal_consistent_count")
                .sum()
                .alias("n_consistent"),
                pl.col(f"{prefix}_flat_count").sum().alias("n_flat"),
                pl.col(f"{prefix}_sum_signal_consistent_move_bp")
                .sum()
                .alias("sum_signal_consistent_move_bp"),
            )
            .with_columns(
                (pl.col("n") - pl.col("n_flat")).alias("n_moved"),
                pl.lit(inference_unit).alias("inference_unit"),
            )
        )
    return (
        pl.concat(outputs, how="vertical_relaxed")
        .with_columns(
            pl.when(pl.col("n") > 0)
            .then(pl.col("sum_signal_consistent_move_bp") / pl.col("n"))
            .otherwise(None)
            .alias("mean_signal_consistent_move_bp"),
            pl.when(pl.col("n") > 0)
            .then(pl.col("n_consistent") / pl.col("n"))
            .otherwise(None)
            .alias("p_signal_consistent"),
            pl.when(pl.col("n") > 0)
            .then(pl.col("n_flat") / pl.col("n"))
            .otherwise(None)
            .alias("p_flat"),
            pl.when(pl.col("n_moved") > 0)
            .then(pl.col("n_consistent") / pl.col("n_moved"))
            .otherwise(None)
            .alias("p_signal_consistent_given_move"),
        )
        .sort([*groups, "inference_unit"])
        .collect(engine="streaming")
    )


def summarize_geometry(frame: pl.DataFrame) -> pl.DataFrame:
    """Summarize known-cost reference geometry without promoting it to EV."""

    groups = ["policy_id", "policy_kind"]
    required = {
        *groups,
        "Date",
        "ValueCode",
        "nominal_band_bp",
        "rounded_reference_band_bp",
        "same_day_reference_cost_bp",
        "overnight_reference_cost_bp",
        "same_day_known_cost_margin_bp",
        "overnight_known_cost_margin_bp",
        "nominal_same_day_known_cost_margin_bp",
        "nominal_overnight_known_cost_margin_bp",
        "same_day_margin_after_10bp_adverse_bp",
        "same_day_margin_after_20bp_adverse_bp",
        "same_day_margin_after_30bp_adverse_bp",
        "actionable_execution",
        "ev_ready",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"geometry input missing columns: {missing}")
    if frame.filter(
        pl.col("actionable_execution").fill_null(True)
        | pl.col("ev_ready").fill_null(True)
    ).height:
        raise ValueError("S0.5 geometry cannot be actionable or EV-ready")
    return (
        frame.group_by(groups)
        .agg(
            pl.len().alias("product_days"),
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.col("nominal_band_bp").median().alias(
                "nominal_band_bp_median"
            ),
            pl.col("rounded_reference_band_bp").median().alias(
                "reference_rounded_band_bp_median"
            ),
            pl.col("same_day_reference_cost_bp").median().alias(
                "same_day_reference_cost_bp_median"
            ),
            pl.col("overnight_reference_cost_bp").median().alias(
                "overnight_reference_cost_bp_median"
            ),
            pl.col("same_day_known_cost_margin_bp").median().alias(
                "same_day_known_cost_margin_bp_median"
            ),
            pl.col("overnight_known_cost_margin_bp").median().alias(
                "overnight_known_cost_margin_bp_median"
            ),
            pl.col("nominal_same_day_known_cost_margin_bp").median().alias(
                "nominal_same_day_known_cost_margin_bp_median"
            ),
            pl.col("nominal_overnight_known_cost_margin_bp").median().alias(
                "nominal_overnight_known_cost_margin_bp_median"
            ),
            (pl.col("same_day_known_cost_margin_bp") > 0)
            .sum()
            .alias("same_day_known_cost_positive_product_days"),
            (pl.col("overnight_known_cost_margin_bp") > 0)
            .sum()
            .alias("overnight_known_cost_positive_product_days"),
            (pl.col("nominal_same_day_known_cost_margin_bp") > 0)
            .sum()
            .alias("nominal_same_day_known_cost_positive_product_days"),
            (pl.col("nominal_overnight_known_cost_margin_bp") > 0)
            .sum()
            .alias("nominal_overnight_known_cost_positive_product_days"),
            (pl.col("same_day_margin_after_10bp_adverse_bp") > 0)
            .sum()
            .alias("positive_after_10bp_adverse_product_days"),
            (pl.col("same_day_margin_after_20bp_adverse_bp") > 0)
            .sum()
            .alias("positive_after_20bp_adverse_product_days"),
            (pl.col("same_day_margin_after_30bp_adverse_bp") > 0)
            .sum()
            .alias("positive_after_30bp_adverse_product_days"),
        )
        .with_columns(
            (
                pl.col("same_day_known_cost_positive_product_days")
                / pl.col("product_days")
            ).alias("same_day_known_cost_positive_rate"),
            (
                pl.col("overnight_known_cost_positive_product_days")
                / pl.col("product_days")
            ).alias("overnight_known_cost_positive_rate"),
            pl.lit(False).alias("actionable_execution"),
            pl.lit(False).alias("ev_ready"),
            pl.lit("necessary_pre_replay_screen_not_profitability").alias(
                "interpretation"
            ),
        )
        .sort("policy_id")
    )


def rank_validation_by_date(calibration: pl.DataFrame) -> pl.DataFrame:
    """Daily cross-product rank signal using completed target-day quantiles."""

    rows: list[dict[str, object]] = []
    for key, group in calibration.group_by(
        ["Date", "boundary_quantile", "side"], maintain_order=True
    ):
        valid = group.filter(
            pl.col("predicted_distance_bp").is_finite()
            & pl.col("realized_completed_quantile_bp").is_finite()
        )
        correlation = None
        if valid.height >= 3:
            value = valid.select(
                pl.corr(
                    "predicted_distance_bp",
                    "realized_completed_quantile_bp",
                    method="spearman",
                )
            ).item()
            if value is not None and math.isfinite(float(value)):
                correlation = float(value)
        rows.append(
            {
                "Date": str(key[0]),
                "boundary_quantile": int(key[1]),
                "side": str(key[2]),
                "product_days": group.height,
                "rank_supported_product_days": valid.height,
                "spearman_predicted_vs_realized": correlation,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "boundary_quantile", "side"]
    )


def build_input_inventory(
    paths_and_roles: Iterable[tuple[Path, str]],
) -> dict[str, object]:
    """Full-content-hash every consumed input in deterministic path order."""

    unique: dict[str, str] = {}
    for raw_path, role in paths_and_roles:
        path = str(Path(raw_path).resolve())
        previous = unique.get(path)
        if previous is not None and previous != role:
            raise ValueError(f"input path has conflicting roles: {path}")
        unique[path] = role
    records: list[dict[str, object]] = []
    for path_string, role in sorted(unique.items()):
        path = Path(path_string)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid S0.5 input: {path}")
        records.append(
            {
                "path": path_string,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "hash_scope": "full_content",
            }
        )
    digest = _canonical_sha256(records)
    return {
        "records": records,
        "record_count": len(records),
        "inventory_sha256": digest,
        "hash_scope": "full_content",
    }


def _declared_bundle_inputs(
    marker_path: Path,
    *,
    marker_role: str,
    artifact_role: str,
) -> list[tuple[Path, str]]:
    """Resolve every file a publication marker asks its validator to read."""

    marker_path = Path(marker_path)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"publication marker lacks artifacts: {marker_path}")
    result = [(marker_path, marker_role)]
    for name in sorted(artifacts):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe publication artifact name: {name!r}")
        result.append((marker_path.parent / name, artifact_role))
    return result


def _assert_input_inventory_current(inventory: Mapping[str, object]) -> None:
    records = inventory.get("records")
    if not isinstance(records, list):
        raise TypeError("input inventory records are malformed")
    if inventory.get("inventory_sha256") != _canonical_sha256(records):
        raise ValueError("input inventory self-hash mismatch")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("input inventory record is malformed")
        path = Path(str(record["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"input content drift: {path}")


def _git_state() -> dict[str, object]:
    root = MAKER_ROOT.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _artifact_metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        result.update(
            {
                "rows": int(
                    pl.scan_parquet(path)
                    .select(pl.len())
                    .collect(engine="streaming")
                    .item()
                ),
                "columns": len(schema),
                "column_names": list(schema.names()),
                "schema_sha256": _canonical_sha256(
                    {name: str(dtype) for name, dtype in schema.items()}
                ),
            }
        )
    elif path.suffix == ".csv":
        scan = pl.scan_csv(path)
        schema = scan.collect_schema()
        result.update(
            {
                "rows": int(scan.select(pl.len()).collect().item()),
                "columns": len(schema),
                "column_names": list(schema.names()),
            }
        )
    return result


def _artifact_inventory(root: Path) -> dict[str, dict[str, object]]:
    names = [*TOP_LEVEL_FRAME_ARTIFACTS, RUN_CONFIG_ARTIFACT, VERIFICATION_ARTIFACT]
    episode_paths = sorted((root / EPISODE_ROOT).glob("Date=*/excursions.parquet"))
    paths = [root / name for name in names] + episode_paths
    return {
        str(path.relative_to(root)): _artifact_metadata(path) for path in paths
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.write_parquet(temporary, compression="zstd", statistics=True)
    elif path.suffix == ".csv":
        frame.write_csv(temporary)
    else:
        raise ValueError(f"unsupported frame output: {path}")
    os.replace(temporary, path)


def _sink_lazy_frame(frame: pl.LazyFrame, path: Path) -> None:
    """Stream a partitioned lazy result to one atomically named artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.sink_parquet(
        temporary,
        compression="zstd",
        statistics=True,
        maintain_order=True,
    )
    os.replace(temporary, path)


def _temporary_output_root(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )


def _publish_output(temporary: Path, output_root: Path) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(
            f"refusing to replace existing S0.5 bundle: {output_root}"
        )
    os.replace(temporary, output_root)


def run(paths: FoundationPaths | None = None) -> Mapping[str, object]:
    """Build the canonical bundle.  Integration is defined below."""

    paths = paths or FoundationPaths()

    # Imports are local so pure-function tests can exercise runner utilities
    # without importing the heavier daily analysis modules.
    from ..quote_width.rolling import load_rolling_boundary_snapshots
    from .foundation_anchor import evaluate_anchor_day, summarize_prior_day
    from .foundation_boundary import (
        build_foundation_cohorts,
        calibrate_boundary_product_days,
        extract_censor_aware_excursions,
    )
    from .foundation_geometry import build_policy_geometry
    from .universe_manifest import load_verified_liquidity_sources

    started_at = monotonic()
    sessions = load_frozen_sessions(paths.sessions_path)
    daily_audit, daily_input_paths = validate_named_daily_partitions(
        paths.daily_root,
        sessions,
    )
    git_state = _git_state()
    if git_state["dirty"]:
        raise ValueError(
            "canonical S0.5 run requires a clean committed source tree: "
            f"{git_state['status']}"
        )
    input_paths_and_roles: list[tuple[Path, str]] = [
        (paths.sessions_path, "frozen_session_calendar"),
        (paths.legacy_manifest_path, "diagnostic_legacy_selector_bridge"),
    ]
    input_paths_and_roles.extend(
        _declared_bundle_inputs(
            paths.boundary_path.parent / "complete.json",
            marker_role="rolling_boundary_completion_marker",
            artifact_role="rolling_boundary_declared_artifact",
        )
    )
    input_paths_and_roles.extend(
        _declared_bundle_inputs(
            paths.liquidity_path.parent / "complete.json",
            marker_role="liquidity_completion_marker",
            artifact_role="liquidity_declared_artifact",
        )
    )
    input_paths_and_roles.extend(
        (path, _daily_input_role(path)) for path in daily_input_paths
    )
    # Hash before the first analytical read.  A second full rehash immediately
    # before marker publication rejects mixed-version output if any source
    # changed during the multi-hour daily run.
    print("S0.5 hashing frozen inputs before analysis", flush=True)
    input_inventory = build_input_inventory(input_paths_and_roles)
    print(
        f"S0.5 input hash complete ({monotonic() - started_at:.1f}s)",
        flush=True,
    )
    input_audit = pl.from_dicts(input_inventory["records"])
    boundaries = load_rolling_boundary_snapshots(
        paths.boundary_path,
        expected_daily_root=paths.daily_root,
    )
    liquidity_bundle = load_verified_liquidity_sources(
        paths.liquidity_path.parent,
        expected_daily_root=paths.daily_root,
        expected_boundary_path=paths.boundary_path,
    )
    liquidity = liquidity_bundle.rolling_screen
    cohorts = build_foundation_cohorts(
        boundaries,
        liquidity,
        primary_start=PRIMARY_START_DATE,
        source_end=SOURCE_END_DATE,
    )
    all_supported = cohorts.all_boundary_supported
    broad = cohorts.spot_bid_broad_candidate
    boundary_wide = all_supported
    full60_target_keys = (
        boundaries.filter(
            (pl.col("Date") >= PRIMARY_START_DATE)
            & (pl.col("Date") <= SOURCE_END_DATE)
            & (pl.col("history_sessions_global") == 60)
        )
        .select("Date", "ValueCode", "QuoteCode")
        .unique()
        .sort(["Date", "ValueCode", "QuoteCode"])
    )
    if all_supported.height != EXPECTED_ALL_BOUNDARY_SUPPORTED_PRODUCT_DAYS:
        raise ValueError("all-boundary-supported canonical count drift")
    if (
        broad.height != EXPECTED_BROAD_PRODUCT_DAYS
        or broad["ValueCode"].n_unique() != EXPECTED_BROAD_PRODUCTS
    ):
        raise ValueError("Spot-Bid broad canonical cohort count drift")

    temporary = _temporary_output_root(paths.output_root)
    try:
        anchor_partition_paths: list[Path] = []
        delayed_partition_paths: list[Path] = []
        coverage_partition_paths: list[Path] = []
        calibration_frames: list[pl.DataFrame] = []
        censor_frames: list[pl.DataFrame] = []
        prior_for_day: pl.DataFrame | None = None
        overlay_rows = 0
        legacy_equivalence_failures = 0
        columns = _daily_read_columns()
        for index, date in enumerate(sessions):
            daily_path = (
                paths.daily_root / f"Date={date}" / "causal_fair.parquet"
            )
            day = pl.read_parquet(daily_path, columns=columns)
            anchor_equivalence = _published_anchor_equivalence(day)
            daily_audit[index].update(anchor_equivalence)
            anchor_result = None
            if date >= PRIMARY_START_DATE:
                anchor_result = evaluate_anchor_day(
                    day,
                    priors=prior_for_day,
                    stage=stage_for_date(date),
                )
                anchor_path = (
                    temporary / ".work_anchor" / f"Date={date}.parquet"
                )
                delayed_path = (
                    temporary / ".work_delayed" / f"Date={date}.parquet"
                )
                coverage_path = (
                    temporary / ".work_coverage" / f"Date={date}.parquet"
                )
                _write_frame(anchor_result.anchor_daily, anchor_path)
                _write_frame(anchor_result.delayed_daily, delayed_path)
                _write_frame(anchor_result.coverage, coverage_path)
                anchor_partition_paths.append(anchor_path)
                delayed_partition_paths.append(delayed_path)
                coverage_partition_paths.append(coverage_path)

            episodes = extract_censor_aware_excursions(day)
            overlay_rows += episodes.height
            episode_path = (
                temporary
                / EPISODE_ROOT
                / f"Date={date}"
                / "excursions.parquet"
            )
            _write_frame(episodes, episode_path)
            legacy = pl.read_parquet(
                paths.daily_root / f"Date={date}" / "excursions.parquet"
            )
            if not _legacy_episode_equivalent(episodes, legacy):
                legacy_equivalence_failures += 1
            censor_frames.append(_summarize_episode_censor(episodes))

            if date >= PRIMARY_START_DATE:
                date_keys = all_supported.filter(pl.col("Date") == date)
                date_boundaries = boundaries.join(
                    date_keys.select("Date", "ValueCode", "QuoteCode"),
                    on=["Date", "ValueCode", "QuoteCode"],
                    how="semi",
                )
                calibration_frames.append(
                    calibrate_boundary_product_days(
                        date_boundaries,
                        episodes,
                        date_keys,
                    )
                )

            if index + 1 < len(sessions):
                prior_for_day = summarize_prior_day(
                    day,
                    target_date=sessions[index + 1],
                    prior_date=date,
                )
            del day, episodes, legacy, anchor_result
            gc.collect()
            print(
                f"S0.5 {index + 1}/{len(sessions)} Date={date} "
                f"({monotonic() - started_at:.1f}s)",
                flush=True,
            )

        if legacy_equivalence_failures:
            raise ValueError(
                "new non-left-censored episodes do not reproduce legacy facts "
                f"on {legacy_equivalence_failures} dates"
            )
        if (
            len(anchor_partition_paths) != EXPECTED_PRIMARY_SESSION_COUNT
            or len(delayed_partition_paths) != EXPECTED_PRIMARY_SESSION_COUNT
            or len(coverage_partition_paths) != EXPECTED_PRIMARY_SESSION_COUNT
        ):
            raise ValueError("anchor primary partition count drift")
        anchor_scan = pl.scan_parquet(anchor_partition_paths)
        delayed_scan = pl.scan_parquet(delayed_partition_paths)
        coverage_scan = pl.scan_parquet(coverage_partition_paths)
        _sink_lazy_frame(anchor_scan, temporary / ANCHOR_ARTIFACT)
        _sink_lazy_frame(delayed_scan, temporary / DELAYED_ARTIFACT)
        coverage_date_count = int(
            coverage_scan.select(pl.col("Date").n_unique()).collect().item()
        )
        calibration = pl.concat(calibration_frames, how="diagonal_relaxed")
        censor_summary = pl.concat(censor_frames, how="diagonal_relaxed")

        anchor_summary = _summarize_anchor_daily(anchor_scan)
        anchor_bootstrap = _anchor_bootstrap(
            anchor_scan.filter(
                (pl.col("stratum_family") == "overall")
                & (pl.col("stratum_value") == "all")
            ).collect(engine="streaming")
        )
        anchor_coverage_summary = _summarize_anchor_coverage(coverage_scan)
        delayed_summary = summarize_delayed_reversion(delayed_scan)
        calibration_summary = summarize_calibration(calibration)
        calibration_month = summarize_calibration(
            calibration.with_columns(pl.col("Date").str.slice(0, 6).alias("month")),
            extra_groups=["month"],
        )
        calibration_bootstrap = whole_date_bootstrap(
            calibration,
            group_columns=["boundary_quantile", "side"],
            metric_columns=[
                "reach_lower_bound",
                "reach_upper_bound",
                "complete_case_reach",
            ],
        )
        rank = rank_validation_by_date(calibration)
        # The q-independent cohort intentionally drops reference prices.  The
        # validated long boundary publication supplies target-day opening refs,
        # contract size, and price-ladder lineage for reference geometry.
        geometry = build_policy_geometry(broad, boundaries)
        geometry_summary = summarize_geometry(geometry)
        legacy_bridge = _legacy_selector_bridge(
            broad,
            paths.legacy_manifest_path,
        )
        cohort_funnel = _cohort_funnel(
            full60_target_keys,
            all_supported,
            broad,
            legacy_bridge,
        )

        outputs = {
            ANCHOR_SUMMARY_ARTIFACT: anchor_summary,
            ANCHOR_BOOTSTRAP_ARTIFACT: anchor_bootstrap,
            ANCHOR_COVERAGE_ARTIFACT: anchor_coverage_summary,
            DELAYED_SUMMARY_ARTIFACT: delayed_summary,
            BOUNDARY_WIDE_ARTIFACT: boundary_wide,
            BROAD_COHORT_ARTIFACT: broad,
            CALIBRATION_ARTIFACT: calibration,
            CALIBRATION_SUMMARY_ARTIFACT: calibration_summary,
            CALIBRATION_MONTH_ARTIFACT: calibration_month,
            CALIBRATION_BOOTSTRAP_ARTIFACT: calibration_bootstrap,
            RANK_ARTIFACT: rank,
            CENSOR_SUMMARY_ARTIFACT: censor_summary,
            GEOMETRY_ARTIFACT: geometry,
            GEOMETRY_SUMMARY_ARTIFACT: geometry_summary,
            COHORT_FUNNEL_ARTIFACT: cohort_funnel,
            LEGACY_BRIDGE_ARTIFACT: legacy_bridge,
            INPUT_AUDIT_ARTIFACT: input_audit,
        }
        for name, frame in outputs.items():
            _write_frame(frame, temporary / name)

        verification = _domain_verification(
            sessions=sessions,
            daily_audit=pl.from_dicts(daily_audit),
            anchor_coverage_date_count=coverage_date_count,
            all_supported=all_supported,
            broad=broad,
            calibration=calibration,
            geometry=geometry,
            legacy_bridge=legacy_bridge,
            overlay_rows=overlay_rows,
            legacy_equivalence_failures=legacy_equivalence_failures,
        )
        shutil.rmtree(temporary / ".work_anchor")
        shutil.rmtree(temporary / ".work_delayed")
        shutil.rmtree(temporary / ".work_coverage")
        verification["canonical_checks"]["clean_source_commit"] = not bool(
            git_state["dirty"]
        )
        run_config = {
            "runner_version": RUNNER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "date_contract": {
                "source_start": SOURCE_START_DATE,
                "source_end": SOURCE_END_DATE,
                "protected_forward_start": PROTECTED_FORWARD_START_DATE,
                "primary_start": PRIMARY_START_DATE,
                "sessions": sessions,
                "primary_sessions": [
                    date for date in sessions if date >= PRIMARY_START_DATE
                ],
                "development_only": True,
                "pristine_final": False,
            },
            "paths": {
                key: str(Path(value).resolve())
                for key, value in asdict(paths).items()
            },
            "weighting_contract": {
                "anchor_accuracy": "one_second_occupancy_descriptive",
                "anchor_inference_ci": "whole_Date_block_bootstrap",
                "boundary_primary": "product_day_equal",
                "boundary_audit": "event_pooled",
                "delayed_primary": "300_second_lockout_nonoverlap",
            },
            "geometry_contract": {
                "known_fees_and_taxes_only": True,
                "tt_band_subtracted_again": False,
                "B6_adverse_cost_included": False,
                "maker_fill_included": False,
                "exit_execution_included": False,
                "actionable_execution": False,
                "ev_ready": False,
            },
            "input_inventory": input_inventory,
            "git_state": git_state,
            "verification": verification,
        }
        _write_json(temporary / RUN_CONFIG_ARTIFACT, run_config)
        _write_json(temporary / VERIFICATION_ARTIFACT, verification)
        artifacts = _artifact_inventory(temporary)
        marker_payload: dict[str, object] = {
            "complete": True,
            "runner_version": RUNNER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "artifacts": artifacts,
            "config_sha256": _canonical_sha256(run_config),
            "input_inventory_sha256": input_inventory["inventory_sha256"],
            "verification": verification,
            "canonical_eligible": all(verification["canonical_checks"].values()),
            "development_only": True,
            "pristine_final": False,
        }
        marker_payload["marker_payload_sha256"] = _canonical_sha256(
            marker_payload
        )
        print("S0.5 rehashing frozen inputs before publication", flush=True)
        _assert_input_inventory_current(input_inventory)
        if _git_state() != git_state:
            raise ValueError("source tree changed during canonical S0.5 run")
        _write_json(temporary / COMPLETE_ARTIFACT, marker_payload)
        verify_bundle(temporary, verify_inputs=False)
        _publish_output(temporary, paths.output_root)
        return verify_bundle(paths.output_root, verify_inputs=False)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _daily_read_columns() -> list[str]:
    return [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "end_date",
        "basis_mid_bp",
        "basis_eval_bp",
        "leg_skew_ms",
        "eligible_base",
        "eligible_100ms",
        "eligible_1000ms",
        "analysis_eligible",
        "anchor_ewma_30s_bp",
        "anchor_ewma_120s_bp",
        "anchor_ewma_300s_bp",
        "anchor_rolling_median_300s_bp",
    ]


def _daily_input_role(path: Path) -> str:
    if path.name == "complete.json":
        return "daily_completion_marker"
    if path.name == "causal_fair.parquet":
        return "daily_causal_fair_input"
    if path.name == "excursions.parquet":
        return "legacy_excursion_equivalence_input"
    if path.name == "mapping.parquet":
        return "daily_marker_validation_mapping_input"
    if path.name == "audit.parquet":
        return "daily_marker_validation_audit_input"
    raise ValueError(f"unexpected daily input: {path}")


def _published_anchor_equivalence(day: pl.DataFrame) -> dict[str, object]:
    """Bridge the migrated panel to the canonical anchor and analysis gates."""

    ordered = day.sort(["Date", "ValueCode", "QuoteCode", "timestamp"])
    audit = (
        ordered.with_columns(
            pl.when(
                pl.col("eligible_base").fill_null(False).cast(pl.Boolean)
                & pl.col("basis_mid_bp").is_not_null()
                & pl.col("basis_mid_bp").is_finite()
            )
            .then(pl.col("basis_mid_bp"))
            .otherwise(None)
            .alias("_expected_basis_eval_bp"),
            (
                pl.col("eligible_base").fill_null(False).cast(pl.Boolean)
                & (pl.col("seconds_from_open") >= 300)
            )
            .fill_null(False)
            .alias("_expected_analysis_eligible"),
        )
        .with_columns(
            pl.col("_expected_basis_eval_bp")
            .ewm_mean(half_life=120, adjust=False, ignore_nulls=True)
            .over(["Date", "ValueCode", "QuoteCode"])
            .alias("_recomputed_ewma120_bp")
        )
        .select(
            (
                pl.col("anchor_ewma_120s_bp").is_null()
                != pl.col("_recomputed_ewma120_bp").is_null()
            )
            .sum()
            .alias("null_mismatch_rows"),
            (
                pl.col("basis_eval_bp").is_not_null()
                & ~pl.col("basis_eval_bp").is_finite()
            )
            .sum()
            .alias("basis_eval_nonfinite_rows"),
            (
                pl.col("basis_eval_bp").is_null()
                != pl.col("_expected_basis_eval_bp").is_null()
            )
            .sum()
            .alias("basis_eval_null_mismatch_rows"),
            (
                pl.col("basis_eval_bp") - pl.col("_expected_basis_eval_bp")
            )
            .abs()
            .max()
            .alias("basis_eval_max_abs_difference_bp"),
            (
                ~pl.col("analysis_eligible").eq_missing(
                    pl.col("_expected_analysis_eligible")
                )
            )
            .sum()
            .alias("analysis_eligible_mismatch_rows"),
            (
                pl.col("anchor_ewma_120s_bp")
                - pl.col("_recomputed_ewma120_bp")
            )
            .abs()
            .max()
            .alias("max_abs_difference_bp"),
        )
        .row(0, named=True)
    )
    return {
        "published_basis_eval_nonfinite_rows": int(
            audit["basis_eval_nonfinite_rows"]
        ),
        "published_basis_eval_null_mismatch_rows": int(
            audit["basis_eval_null_mismatch_rows"]
        ),
        "published_basis_eval_max_abs_difference_bp": float(
            audit["basis_eval_max_abs_difference_bp"] or 0.0
        ),
        "published_analysis_eligible_mismatch_rows": int(
            audit["analysis_eligible_mismatch_rows"]
        ),
        "published_ewma120_null_mismatch_rows": int(
            audit["null_mismatch_rows"]
        ),
        "published_ewma120_max_abs_difference_bp": float(
            audit["max_abs_difference_bp"] or 0.0
        ),
    }


def _legacy_episode_equivalent(
    overlay: pl.DataFrame,
    legacy: pl.DataFrame,
) -> bool:
    """Confirm non-left-censored overlay rows reproduce the legacy episodes."""

    adapter = overlay.filter(~pl.col("left_censored")).select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "side",
        pl.col("completed_center_return").alias("completed"),
        pl.col("right_censor_reason")
        .fill_null("crossed_center")
        .alias("end_reason"),
        pl.col("observed_amplitude_bp").alias("amplitude_bp"),
        "start_seconds_from_open",
        "end_seconds_from_open",
    )
    columns = adapter.columns
    if adapter.height != legacy.height:
        return False
    joined = adapter.join(
        legacy.select(columns),
        on=[
            "Date",
            "ValueCode",
            "QuoteCode",
            "side",
            "completed",
            "end_reason",
            "start_seconds_from_open",
            "end_seconds_from_open",
        ],
        how="inner",
        suffix="_legacy",
    )
    return joined.height == legacy.height and not joined.filter(
        (pl.col("amplitude_bp") - pl.col("amplitude_bp_legacy")).abs()
        > 1e-9
    ).height


def _summarize_episode_censor(episodes: pl.DataFrame) -> pl.DataFrame:
    return (
        episodes.with_columns(
            pl.col("Date").str.slice(0, 6).alias("month"),
            pl.when(pl.col("left_censored"))
            .then(pl.lit("left_censored"))
            .when(pl.col("right_censored"))
            .then(pl.lit("right_censored"))
            .otherwise(pl.lit("fully_observed"))
            .alias("censor_class"),
        )
        .group_by(
            [
                "Date",
                "month",
                "side",
                "censor_class",
                "left_censor_reason",
                "right_censor_reason",
            ]
        )
        .agg(
            pl.len().alias("episodes"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.col("observed_amplitude_bp").median().alias(
                "observed_amplitude_bp_median"
            ),
        )
        .sort(["Date", "side", "censor_class"])
    )


def _summarize_anchor_daily(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    columns = set(lazy.collect_schema().names())
    required = {
        "model",
        "freshness_sample",
        "stratum_family",
        "stratum_value",
        "Date",
        "ValueCode",
        "n_pairwise_common",
        "n_evaluable",
        "sum_error_bp",
        "sum_abs_error_bp",
        "pairwise_model_sum_abs_error_bp",
        "pairwise_baseline_sum_abs_error_bp",
        "anchor_tv_bp",
        "basis_tv_bp",
        "n_adjacent_legal_pairs",
        "mae_bp",
        "p80_abs_error_bp",
        "p95_abs_error_bp",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"anchor daily input missing columns: {missing}")
    groups = [
        "model",
        "freshness_sample",
        "stratum_family",
        "stratum_value",
    ]
    return (
        lazy.group_by(groups)
        .agg(
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.col("n_pairwise_common").sum().alias("n_pairwise_common"),
            pl.col("n_evaluable").sum().alias("n_evaluable"),
            pl.col("sum_error_bp").sum().alias("sum_error_bp"),
            pl.col("sum_abs_error_bp").sum().alias("sum_abs_error_bp"),
            pl.col("pairwise_model_sum_abs_error_bp")
            .sum()
            .alias("pairwise_model_sum_abs_error_bp"),
            pl.col("pairwise_baseline_sum_abs_error_bp")
            .sum()
            .alias("pairwise_baseline_sum_abs_error_bp"),
            pl.col("anchor_tv_bp").sum().alias("sum_anchor_tv_bp"),
            pl.col("basis_tv_bp").sum().alias("sum_basis_tv_bp"),
            pl.col("n_adjacent_legal_pairs").sum().alias(
                "adjacent_legal_pairs"
            ),
            pl.col("mae_bp").mean().alias(
                "product_day_equal_mae_bp"
            ),
            pl.col("p80_abs_error_bp").mean().alias(
                "product_day_equal_p80_abs_error_bp"
            ),
            pl.col("p95_abs_error_bp").mean().alias(
                "product_day_equal_p95_abs_error_bp"
            ),
        )
        .with_columns(
            pl.when(pl.col("n_evaluable") > 0)
            .then(pl.col("sum_error_bp") / pl.col("n_evaluable"))
            .otherwise(None)
            .alias("occupancy_weighted_bias_bp"),
            pl.when(pl.col("n_evaluable") > 0)
            .then(pl.col("sum_abs_error_bp") / pl.col("n_evaluable"))
            .otherwise(None)
            .alias("occupancy_weighted_mae_bp"),
            pl.when(pl.col("n_pairwise_common") > 0)
            .then(
                (
                    pl.col("pairwise_model_sum_abs_error_bp")
                    - pl.col("pairwise_baseline_sum_abs_error_bp")
                )
                / pl.col("n_pairwise_common")
            )
            .otherwise(None)
            .alias("pairwise_delta_mae_vs_ewma120_bp"),
            pl.when(pl.col("sum_basis_tv_bp") > 1e-12)
            .then(pl.col("sum_anchor_tv_bp") / pl.col("sum_basis_tv_bp"))
            .otherwise(None)
            .alias("tv_ratio"),
            pl.lit("product_day_equal_primary_occupancy_weighted_audit").alias(
                "weighting_contract"
            ),
        )
        .sort(groups)
        .collect(engine="streaming")
    )


def _anchor_bootstrap(frame: pl.DataFrame) -> pl.DataFrame:
    primary = frame.filter(
        (pl.col("stratum_family") == "overall")
        & (pl.col("stratum_value") == "all")
    )
    return whole_date_bootstrap(
        primary,
        group_columns=["model", "freshness_sample"],
        metric_columns=[
            "mae_bp",
            "p80_abs_error_bp",
            "p95_abs_error_bp",
        ],
    )


def _summarize_anchor_coverage(frame: pl.LazyFrame) -> pl.DataFrame:
    groups = [
        "model",
        "freshness_sample",
        "stratum_family",
        "stratum_value",
    ]
    count_columns = [
        "n_grid",
        "n_current_eligible",
        "n_anchor_available",
        "n_horizon_possible",
        "n_future_center_supported",
        "censor_current_gate_closed",
        "censor_anchor_unavailable",
        "censor_horizon_after_session",
        "censor_future_window_coverage_lt_90pct",
        "evaluable",
    ]
    missing = sorted(
        {"Date", "ValueCode", "prior_valid", *groups, *count_columns}
        - set(frame.collect_schema().names())
    )
    if missing:
        raise ValueError(f"anchor coverage input missing columns: {missing}")
    return (
        frame.group_by(groups)
        .agg(
            pl.col("Date").n_unique().alias("dates"),
            pl.col("ValueCode").n_unique().alias("products"),
            pl.col("prior_valid").fill_null(False).sum().alias(
                "prior_valid_product_days"
            ),
            *[pl.col(column).sum().alias(column) for column in count_columns],
        )
        .with_columns(
            pl.when(pl.col("n_current_eligible") > 0)
            .then(
                pl.col("n_future_center_supported")
                / pl.col("n_current_eligible")
            )
            .otherwise(None)
            .alias("future_center_support_given_current_rate"),
            pl.lit("mutually_exclusive_censor_counts").alias(
                "coverage_contract"
            ),
        )
        .sort(groups)
        .collect(engine="streaming")
    )


def _legacy_selector_bridge(
    broad: pl.DataFrame,
    manifest_path: Path,
) -> pl.DataFrame:
    manifest = pl.read_csv(
        manifest_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
        },
    ).select("Date", "ValueCode", "QuoteCode").unique()
    keys = ["Date", "ValueCode", "QuoteCode"]
    return (
        broad.select(*keys)
        .join(
            manifest.with_columns(pl.lit(True).alias("legacy_selector_member")),
            on=keys,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("legacy_selector_member").fill_null(False),
            pl.lit(False).alias("used_to_construct_broad_cohort"),
        )
        .sort(keys)
    )


def _cohort_funnel(
    boundary_wide: pl.DataFrame,
    all_supported: pl.DataFrame,
    broad: pl.DataFrame,
    legacy_bridge: pl.DataFrame,
) -> pl.DataFrame:
    rows = [
        {
            "stage": "full60_target_mapping",
            "product_days": boundary_wide.height,
            "dates": boundary_wide["Date"].n_unique(),
            "products": boundary_wide["ValueCode"].n_unique(),
            "target_outcome_used": False,
        },
        {
            "stage": "all_q_boundary_supported",
            "product_days": all_supported.height,
            "dates": all_supported["Date"].n_unique(),
            "products": all_supported["ValueCode"].n_unique(),
            "target_outcome_used": False,
        },
        {
            "stage": "spot_bid_broad_D_safe",
            "product_days": broad.height,
            "dates": broad["Date"].n_unique(),
            "products": broad["ValueCode"].n_unique(),
            "target_outcome_used": False,
        },
        {
            "stage": "legacy_monthly_selector_bridge_only",
            "product_days": int(legacy_bridge["legacy_selector_member"].sum()),
            "dates": legacy_bridge.filter(pl.col("legacy_selector_member"))[
                "Date"
            ].n_unique(),
            "products": legacy_bridge.filter(pl.col("legacy_selector_member"))[
                "ValueCode"
            ].n_unique(),
            "target_outcome_used": False,
        },
    ]
    return pl.from_dicts(rows)


def _domain_verification(
    *,
    sessions: Sequence[str],
    daily_audit: pl.DataFrame,
    anchor_coverage_date_count: int,
    all_supported: pl.DataFrame,
    broad: pl.DataFrame,
    calibration: pl.DataFrame,
    geometry: pl.DataFrame,
    legacy_bridge: pl.DataFrame,
    overlay_rows: int,
    legacy_equivalence_failures: int,
) -> dict[str, object]:
    accounting_ok = not calibration.filter(
        pl.col("n_hit")
        + pl.col("n_known_miss")
        + pl.col("n_unknown_censored")
        != pl.col("n_started")
    ).height
    bounds_ok = not calibration.filter(
        (pl.col("reach_lower_bound") < 0)
        | (pl.col("reach_upper_bound") > 1)
        | (
            pl.col("reach_lower_bound")
            > pl.col("reach_upper_bound") + 1e-12
        )
    ).height
    geometry_policy_count = geometry["policy_id"].n_unique()
    checks = {
        "frozen_131_session_contract": (
            len(sessions) == EXPECTED_SESSION_COUNT
            and sessions[0] == SOURCE_START_DATE
            and sessions[-1] == SOURCE_END_DATE
        ),
        "protected_forward_excluded": max(sessions) < PROTECTED_FORWARD_START_DATE,
        "migrated_daily_markers_disclosed": bool(
            daily_audit["migrated_legacy_marker"].all()
        ),
        "published_basis_eval_equivalent": (
            int(daily_audit["published_basis_eval_nonfinite_rows"].sum()) == 0
            and int(
                daily_audit["published_basis_eval_null_mismatch_rows"].sum()
            )
            == 0
            and float(
                daily_audit[
                    "published_basis_eval_max_abs_difference_bp"
                ].max()
            )
            <= 1e-9
        ),
        "published_analysis_gate_equivalent": int(
            daily_audit["published_analysis_eligible_mismatch_rows"].sum()
        )
        == 0,
        "published_recomputed_ewma120_equivalent": (
            int(daily_audit["published_ewma120_null_mismatch_rows"].sum()) == 0
            and float(
                daily_audit[
                    "published_ewma120_max_abs_difference_bp"
                ].max()
            )
            <= 1e-9
        ),
        "anchor_coverage_all_primary_sessions": anchor_coverage_date_count
        == EXPECTED_PRIMARY_SESSION_COUNT,
        "legacy_nonleft_episode_equivalence": legacy_equivalence_failures == 0,
        "censor_overlay_nonempty": overlay_rows > 0,
        "full60_all_boundary_supported_count": all_supported.height
        == EXPECTED_ALL_BOUNDARY_SUPPORTED_PRODUCT_DAYS,
        "full60_broad_cohort_count": broad.height == EXPECTED_BROAD_PRODUCT_DAYS,
        "broad_cohort_71_dates": broad["Date"].n_unique()
        == EXPECTED_PRIMARY_SESSION_COUNT,
        "boundary_censor_accounting": accounting_ok,
        "boundary_identified_bounds": bounds_ok,
        "boundary_no_target_leak": not calibration.filter(
            pl.col("source_asof_date") >= pl.col("Date")
        ).height,
        "seven_policy_common_geometry": geometry_policy_count == 7
        and geometry.height == EXPECTED_BROAD_PRODUCT_DAYS * 7,
        "geometry_not_execution_ready": not geometry.filter(
            pl.col("actionable_execution").fill_null(True)
            | pl.col("ev_ready").fill_null(True)
        ).height,
        "legacy_selector_not_cohort_gate": not legacy_bridge[
            "used_to_construct_broad_cohort"
        ].any(),
    }
    return {
        "canonical_checks": checks,
        "session_count": len(sessions),
        "primary_session_count": len(
            [date for date in sessions if date >= PRIMARY_START_DATE]
        ),
        "daily_causal_rows": int(daily_audit["causal_rows"].sum()),
        "daily_product_days": int(daily_audit["products"].sum()),
        "censor_overlay_rows": overlay_rows,
        "all_boundary_supported_product_days": all_supported.height,
        "broad_product_days": broad.height,
        "broad_products": broad["ValueCode"].n_unique(),
        "calibration_rows": calibration.height,
        "geometry_rows": geometry.height,
        "legacy_selector_members_inside_broad": int(
            legacy_bridge["legacy_selector_member"].sum()
        ),
        "development_only": True,
        "pristine_final": False,
        "actionable_execution": False,
        "ev_ready": False,
    }


def verify_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    verify_inputs: bool = False,
) -> Mapping[str, object]:
    """Verify marker, recursive artifact inventory, and domain invariants."""

    root = Path(output_root)
    marker_path = root / COMPLETE_ARTIFACT
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    digest = marker.get("marker_payload_sha256")
    payload = dict(marker)
    payload.pop("marker_payload_sha256", None)
    if digest != _canonical_sha256(payload):
        raise ValueError("S0.5 complete marker self-hash mismatch")
    if (
        marker.get("complete") is not True
        or marker.get("runner_version") != RUNNER_VERSION
        or marker.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported or incomplete S0.5 bundle")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("S0.5 marker artifact inventory is malformed")
    expected = _artifact_inventory(root)
    if artifacts != expected:
        raise ValueError("S0.5 artifact metadata drift")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = {*expected, COMPLETE_ARTIFACT}
    if actual_files != expected_files:
        raise ValueError(
            "S0.5 bundle file inventory drift: "
            f"{sorted(actual_files ^ expected_files)}"
        )
    run_config = json.loads((root / RUN_CONFIG_ARTIFACT).read_text())
    sessions = run_config.get("date_contract", {}).get("sessions")
    if not isinstance(sessions, list):
        raise TypeError("S0.5 run_config session contract is malformed")
    episode_files = {
        name
        for name in expected
        if name.startswith(f"{EPISODE_ROOT}/Date=")
    }
    expected_episode_files = {
        f"{EPISODE_ROOT}/Date={date}/excursions.parquet" for date in sessions
    }
    if episode_files != expected_episode_files:
        raise ValueError("S0.5 censor episode partition set drift")
    if marker.get("config_sha256") != _canonical_sha256(run_config):
        raise ValueError("S0.5 run_config hash mismatch")
    inventory = run_config.get("input_inventory")
    if not isinstance(inventory, dict):
        raise TypeError("S0.5 input inventory is malformed")
    if marker.get("input_inventory_sha256") != inventory.get(
        "inventory_sha256"
    ):
        raise ValueError("S0.5 input inventory digest drift")
    if verify_inputs:
        _assert_input_inventory_current(inventory)
    verification = json.loads((root / VERIFICATION_ARTIFACT).read_text())
    if verification != marker.get("verification"):
        raise ValueError("S0.5 published verification drift")
    checks = verification.get("canonical_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("S0.5 canonical checks are incomplete or failed")
    if marker.get("canonical_eligible") is not True:
        raise ValueError("S0.5 canonical eligibility drift")
    geometry = pl.read_parquet(root / GEOMETRY_ARTIFACT)
    calibration = pl.read_parquet(root / CALIBRATION_ARTIFACT)
    if geometry.height != EXPECTED_BROAD_PRODUCT_DAYS * 7:
        raise ValueError("S0.5 geometry row count drift")
    if calibration.filter(
        pl.col("n_hit")
        + pl.col("n_known_miss")
        + pl.col("n_unknown_censored")
        != pl.col("n_started")
    ).height:
        raise ValueError("S0.5 calibration censor accounting drift")
    return {
        **verification,
        "canonical_eligible": True,
        "bundle": str(root.resolve()),
        "complete_sha256": _sha256_file(marker_path),
        "marker_payload_sha256": digest,
        "input_content_rehashed": verify_inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY_PATH)
    parser.add_argument(
        "--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_bundle(args.output, verify_inputs=args.verify_inputs),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = run(
        FoundationPaths(
            sessions_path=args.sessions,
            daily_root=args.daily_root,
            boundary_path=args.boundaries,
            liquidity_path=args.liquidity,
            legacy_manifest_path=args.legacy_manifest,
            output_root=args.output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
