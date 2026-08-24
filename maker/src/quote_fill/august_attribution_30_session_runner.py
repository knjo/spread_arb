"""Build the market-only S0 30-session q95 boundary sensitivity.

The challenger deliberately reuses the fixed S0 matched universe and the
canonical S0 raw-market excursions.  It does not replay orders, venue request
limits, makerFill, queue outcomes, or capital.  For every primary observable
excursion, ``amplitude_bp >= challenger_upper`` is therefore only a
hypothetical first-touch-existence label; it is not a reconstructed touch
cursor and must never feed queue attribution or a strategy shortlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import validate_completion_marker
from ..quote_width.rolling import (
    RollingBoundaryConfig,
    build_rolling_boundary_snapshots,
)
from .august_attribution_runner import verify_bundle as verify_canonical_s0_bundle

RUNNER_VERSION = "august_attribution_30_session_market_sensitivity_v1"
SCHEMA_VERSION = "august_attribution_30_session_sensitivity_v1"
PARAMETER_VERSION = "rolling_30_session_challenger_s0_v1"
MANIFEST_SHA256 = (
    "9f1bcddf17eff968ee51e0decdb04736a3747f0665886ce3e4fd26031cfb5891"
)
MANIFEST_KEY_SHA256 = (
    "99185ad2385cc86fff23f5b69a6e9ab5e18c99a74e9640eeb6c3eb09e5426cfa"
)
SESSIONS_SHA256 = (
    "e512c3573423fd9491883a5b8d7927b597688c7328ff49a596497a042d15e7cf"
)
EXPECTED_SESSION_COUNT = 72
EXPECTED_HISTORY_SESSION_COUNT = 131
EXPECTED_PRODUCT_DAY_COUNT = 3_886
EXPECTED_FIRST_DATE = "20260504"
EXPECTED_LAST_DATE = "20260813"
EXPECTED_HISTORY_FIRST_DATE = "20260126"
EXPECTED_COMMON_PRODUCT_COUNT = 24
EXPECTED_CANONICAL_S0_RUN_ID = "august_attribution_s0_20260824_v2"
EXPECTED_INPUT_RECORD_COUNT = 5 + (5 * EXPECTED_HISTORY_SESSION_COUNT)

CHALLENGER_CONFIG = RollingBoundaryConfig(
    lookback_sessions=30,
    min_history_sessions=20,
    min_excursion_history_sessions_per_side=20,
    min_completed_excursions_per_side=100,
    quantiles=(95,),
    parameter_version=PARAMETER_VERSION,
)

DEFAULT_MANIFEST_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "monthly_product_selector_causal_v2_20260822"
    / "daily_entry_manifest.csv"
)
DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
# S0 v1 is intentionally not the default: its raw-state genuine-clear
# semantics were found to require a rebuild before this sensitivity is valid.
DEFAULT_CANONICAL_S0_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "august_attribution_s0_20260824_v2"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "august_attribution_s0_30_session_challenger_20260824_v1"
)

BOUNDARY_ARTIFACT = "rolling_30_q95_boundaries.parquet"
PRODUCT_DAY_ARTIFACT = "product_day_boundary_sensitivity.parquet"
MONTHLY_ARTIFACT = "monthly_boundary_sensitivity.csv"
MEMBERSHIP_ARTIFACT = "membership_sensitivity.csv"
FRAME_ARTIFACTS = (
    BOUNDARY_ARTIFACT,
    PRODUCT_DAY_ARTIFACT,
    MONTHLY_ARTIFACT,
    MEMBERSHIP_ARTIFACT,
)
NONFRAME_ARTIFACTS = ("run_config.json", "verification.json")
FOCUSED_TEST_MODULES = (
    "maker.src.tests.test_quote_width_rolling",
    "maker.src.tests.test_quote_fill_august_attribution_30_session",
)


@dataclass(frozen=True)
class SensitivityPaths:
    """Filesystem inputs and immutable destination for one sensitivity run."""

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    daily_root: Path = DEFAULT_DAILY_ROOT
    sessions_path: Path = DEFAULT_SESSIONS_PATH
    canonical_s0_root: Path = DEFAULT_CANONICAL_S0_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def load_fixed_manifest(path: Path) -> pl.DataFrame:
    """Load the exact frozen S0 population and reject any content drift."""

    if _sha256_file(path) != MANIFEST_SHA256:
        raise ValueError("fixed S0 manifest SHA-256 mismatch")
    manifest = (
        pl.read_csv(
            path,
            schema_overrides={
                "Date": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
            },
        )
        .select("Date", "ValueCode", "QuoteCode")
        .unique()
        .sort(["Date", "ValueCode"])
    )
    dates = manifest["Date"].unique().sort().to_list()
    if (
        manifest.height != EXPECTED_PRODUCT_DAY_COUNT
        or len(dates) != EXPECTED_SESSION_COUNT
        or dates[0] != EXPECTED_FIRST_DATE
        or dates[-1] != EXPECTED_LAST_DATE
        or _manifest_key_sha256(manifest) != MANIFEST_KEY_SHA256
        or manifest.select("Date", "ValueCode", "QuoteCode").n_unique()
        != manifest.height
    ):
        raise ValueError("fixed S0 manifest population drift")
    return manifest


def _manifest_key_sha256(frame: pl.DataFrame) -> str:
    """Hash the sorted exact-contract population independently of file layout."""

    keys = ("Date", "ValueCode", "QuoteCode")
    _require_columns(frame, keys, "fixed manifest key digest")
    rows = sorted(
        tuple(str(row[key]) for key in keys)
        for row in frame.select(*keys).iter_rows(named=True)
    )
    payload = json.dumps(
        rows,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_fixed_manifest_payload() -> dict[str, object]:
    return {
        "sha256": MANIFEST_SHA256,
        "key_sha256": MANIFEST_KEY_SHA256,
        "product_days": EXPECTED_PRODUCT_DAY_COUNT,
        "entry_sessions": EXPECTED_SESSION_COUNT,
        "first_date": EXPECTED_FIRST_DATE,
        "last_date": EXPECTED_LAST_DATE,
        "jul_aug_common_product_count": EXPECTED_COMMON_PRODUCT_COUNT,
    }


def load_session_calendar(path: Path) -> list[str]:
    """Load one strictly ordered YYYYMMDD market-session calendar."""

    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    parsed = pl.DataFrame({"Date": values}).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("parsed")
    )
    if (
        not values
        or values != sorted(values)
        or len(values) != len(set(values))
        or parsed.filter(
            pl.col("parsed").is_null() | (pl.col("Date").str.len_chars() != 8)
        ).height
    ):
        raise ValueError("session calendar must be sorted unique YYYYMMDD")
    return values


def validated_daily_paths(
    daily_root: Path,
    sessions: Sequence[str],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Validate every declared daily marker and return only consumed files."""

    markers, excursion_paths, mapping_paths = _declared_daily_paths(
        daily_root,
        sessions,
    )
    for marker in markers:
        validate_completion_marker(marker)
    return markers, excursion_paths, mapping_paths


def _declared_daily_paths(
    daily_root: Path,
    sessions: Sequence[str],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Resolve the calendar-bound daily file inventory without consuming it."""

    markers = sorted(Path(daily_root).glob("Date=*/complete.json"))
    dates = [marker.parent.name.removeprefix("Date=") for marker in markers]
    if dates != list(sessions):
        raise ValueError("daily partition set differs from session calendar")
    excursion_paths = [marker.parent / "excursions.parquet" for marker in markers]
    mapping_paths = [marker.parent / "mapping.parquet" for marker in markers]
    return markers, excursion_paths, mapping_paths


def load_daily_boundary_inputs(
    excursion_paths: Sequence[Path],
    mapping_paths: Sequence[Path],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the minimum daily-fact columns consumed by the rolling builder."""

    excursions = (
        pl.scan_parquet(list(excursion_paths))
        .select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("side").cast(pl.String),
            pl.col("amplitude_bp").cast(pl.Float64),
            pl.col("completed").cast(pl.Boolean),
        )
        .collect(engine="streaming")
    )
    mapping = (
        pl.scan_parquet(list(mapping_paths))
        .select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
        )
        .collect(engine="streaming")
    )
    if mapping.select("Date", "ValueCode").n_unique() != mapping.height:
        raise ValueError("daily mapping has duplicate product-days")
    if mapping.filter(
        pl.col("QuoteCode").is_null() | (pl.col("QuoteCode") == "")
    ).height:
        raise ValueError("daily mapping contains an empty target contract")
    return excursions, mapping


def build_fixed_challenger_boundaries(
    daily_excursions: pl.DataFrame,
    daily_mapping: pl.DataFrame,
    sessions: Sequence[str],
    manifest: pl.DataFrame,
    *,
    config: RollingBoundaryConfig = CHALLENGER_CONFIG,
) -> pl.DataFrame:
    """Build causal q95 rows, then project without reselecting the fixed universe."""

    config.validate()
    snapshots = build_rolling_boundary_snapshots(
        daily_excursions,
        daily_mapping,
        sessions,
        config,
    ).filter(pl.col("boundary_quantile") == 95)
    keys = ["Date", "ValueCode", "QuoteCode"]
    if snapshots.select(*keys).n_unique() != snapshots.height:
        raise ValueError("30-session q95 snapshots have duplicate exact contracts")
    selected = manifest.join(snapshots, on=keys, how="left", validate="1:1")
    if selected.height != manifest.height or not selected.select(*keys).equals(
        manifest.select(*keys)
    ):
        raise ValueError("30-session boundary projection changed fixed universe")
    required = {"boundary_quantile", "upper_distance_bp"}
    _require_columns(selected, required, "30-session boundaries")
    missing_snapshot = selected.filter(pl.col("boundary_quantile").is_null())
    if not missing_snapshot.is_empty():
        raise ValueError("fixed manifest lacks a 30-session snapshot row")
    availability = (
        pl.col("adaptive_parameter_valid")
        & pl.col("upper_distance_bp").is_finite()
        & (pl.col("upper_distance_bp") > 0)
    ).fill_null(False)
    result = selected.with_columns(
        availability.alias("challenger_boundary_valid"),
        pl.lit(True).alias("fixed_manifest_member"),
        pl.lit(True).alias("market_only"),
    ).sort(keys)
    _validate_challenger_boundary_rows(result, config=config)
    return result


def _validate_challenger_boundary_rows(
    boundaries: pl.DataFrame,
    *,
    config: RollingBoundaryConfig = CHALLENGER_CONFIG,
) -> None:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "adaptive_parameter_valid",
        "challenger_boundary_valid",
        "lookback_sessions",
        "minimum_history_sessions",
        "minimum_excursion_history_sessions_per_side",
        "history_sessions_global",
        "history_sessions_product",
        "positive_history_dates",
        "negative_history_dates",
        "positive_completed",
        "negative_completed",
        "train_start_date",
        "train_end_date",
        "source_asof_date",
        "parameter_version",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
        "actionable_execution",
        "ev_ready",
        "fixed_manifest_member",
        "market_only",
    }
    _require_columns(boundaries, required, "30-session boundaries")
    keys = ["Date", "ValueCode", "QuoteCode"]
    if boundaries.select(*keys).n_unique() != boundaries.height:
        raise ValueError("30-session boundary exact-contract keys are duplicated")
    expected_available = (
        (pl.col("history_sessions_global") >= config.min_history_sessions)
        & (pl.col("history_sessions_product") >= config.min_history_sessions)
        & (
            pl.col("positive_history_dates")
            >= config.min_excursion_history_sessions_per_side
        )
        & (
            pl.col("negative_history_dates")
            >= config.min_excursion_history_sessions_per_side
        )
        & (
            pl.col("positive_completed")
            >= config.min_completed_excursions_per_side
        )
        & (
            pl.col("negative_completed")
            >= config.min_completed_excursions_per_side
        )
        & pl.col("upper_distance_bp").is_finite()
        & (pl.col("upper_distance_bp") > 0)
        & pl.col("lower_distance_bp").is_finite()
        & (pl.col("lower_distance_bp") > 0)
    ).fill_null(False)
    invalid_contract = boundaries.filter(
        (pl.col("boundary_quantile") != 95)
        | (pl.col("lookback_sessions") != config.lookback_sessions)
        | (pl.col("minimum_history_sessions") != config.min_history_sessions)
        | (
            pl.col("minimum_excursion_history_sessions_per_side")
            != config.min_excursion_history_sessions_per_side
        )
        | (pl.col("parameter_version") != config.parameter_version)
        | ~pl.col("execution_safe_snapshot")
        | pl.col("contains_target_day_outcome")
        | pl.col("actionable_execution")
        | pl.col("ev_ready")
        | ~pl.col("fixed_manifest_member")
        | ~pl.col("market_only")
        | pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
        | (pl.col("train_end_date") != pl.col("source_asof_date"))
        | (pl.col("train_start_date") > pl.col("train_end_date"))
        | (pl.col("adaptive_parameter_valid") != expected_available)
        | (pl.col("challenger_boundary_valid") != expected_available)
    )
    if not invalid_contract.is_empty():
        raise ValueError("30-session boundary causal/config contract failed")


def _validate_exact_history_windows(
    boundaries: pl.DataFrame,
    sessions: Sequence[str],
    *,
    config: RollingBoundaryConfig = CHALLENGER_CONFIG,
) -> None:
    """Recompute every D-1 rolling window from the bound session calendar."""

    calendar = [str(value) for value in sessions]
    if (
        not calendar
        or calendar != sorted(calendar)
        or len(calendar) != len(set(calendar))
    ):
        raise ValueError("history calendar must be sorted and unique")
    session_index = {date: index for index, date in enumerate(calendar)}
    columns = [
        "Date",
        "source_asof_date",
        "train_start_date",
        "train_end_date",
        "history_sessions_global",
    ]
    for row in boundaries.select(*columns).unique().iter_rows(named=True):
        target_date = str(row["Date"])
        if target_date not in session_index:
            raise ValueError(
                f"30-session target absent from history calendar: {target_date}"
            )
        target_index = session_index[target_date]
        prior = calendar[
            max(0, target_index - config.lookback_sessions):target_index
        ]
        if not prior:
            raise ValueError(
                f"30-session target has no prior history: {target_date}"
            )
        expected = (
            prior[-1],
            prior[0],
            prior[-1],
            len(prior),
        )
        actual = (
            row["source_asof_date"],
            row["train_start_date"],
            row["train_end_date"],
            row["history_sessions_global"],
        )
        if actual != expected:
            raise ValueError(
                "30-session exact D-1/history-window contract failed: "
                f"{target_date}: expected={expected}, actual={actual}"
            )


def build_product_day_sensitivity(
    manifest: pl.DataFrame,
    canonical_coverage: pl.DataFrame,
    market_excursions: pl.DataFrame,
    challenger_boundaries: pl.DataFrame,
) -> pl.DataFrame:
    """Count 60/30 touches on one explicit common-valid product-day denominator."""

    keys = ["Date", "ValueCode", "QuoteCode"]
    _require_columns(
        canonical_coverage,
        {
            *keys,
            "upper_distance_bp",
            "lookback_sessions",
            "source_asof_date",
            "parameter_version",
        },
        "canonical product-day coverage",
    )
    _require_columns(
        market_excursions,
        {
            "excursion_id",
            *keys,
            "analysis_window",
            "primary_observable",
            "amplitude_bp",
            "upper_distance_bp",
            "touch_time_ns",
        },
        "canonical market excursions",
    )
    _require_columns(
        challenger_boundaries,
        {
            *keys,
            "upper_distance_bp",
            "challenger_boundary_valid",
            "source_asof_date",
            "train_start_date",
            "train_end_date",
            "history_sessions_global",
            "history_sessions_product",
            "positive_completed",
            "negative_completed",
        },
        "challenger boundaries",
    )
    if (
        canonical_coverage.height != manifest.height
        or canonical_coverage.select(*keys).n_unique()
        != canonical_coverage.height
    ):
        raise ValueError("canonical coverage does not match fixed manifest")
    canonical = manifest.join(
        canonical_coverage.select(
            *keys,
            pl.col("upper_distance_bp").alias("canonical_60_upper_bp"),
            pl.col("lookback_sessions").alias("canonical_lookback_sessions"),
            pl.col("source_asof_date").alias("canonical_source_asof_date"),
            pl.col("parameter_version").alias("canonical_parameter_version"),
        ),
        on=keys,
        how="left",
        validate="1:1",
    )
    canonical = canonical.with_columns(
        (
            (pl.col("canonical_lookback_sessions") == 60)
            & pl.col("canonical_60_upper_bp").is_finite()
            & (pl.col("canonical_60_upper_bp") > 0)
            & (pl.col("canonical_source_asof_date") < pl.col("Date"))
        )
        .fill_null(False)
        .alias("canonical_boundary_valid")
    )
    if canonical.filter(~pl.col("canonical_boundary_valid")).height:
        raise ValueError("canonical 60-session coverage is invalid")

    challenger = challenger_boundaries.select(
        *keys,
        pl.col("upper_distance_bp").alias("challenger_30_upper_bp"),
        "challenger_boundary_valid",
        pl.col("source_asof_date").alias("challenger_source_asof_date"),
        pl.col("train_start_date").alias("challenger_train_start_date"),
        pl.col("train_end_date").alias("challenger_train_end_date"),
        pl.col("history_sessions_global").alias(
            "challenger_history_sessions_global"
        ),
        pl.col("history_sessions_product").alias(
            "challenger_history_sessions_product"
        ),
        pl.col("positive_completed").alias("challenger_positive_completed"),
        pl.col("negative_completed").alias("challenger_negative_completed"),
    )
    base = canonical.join(challenger, on=keys, how="left", validate="1:1")
    if base.height != manifest.height:
        raise ValueError("challenger product-day coverage changed fixed universe")

    primary = market_excursions.filter(
        (pl.col("analysis_window") == "entry_primary")
        & pl.col("primary_observable")
    )
    if primary["excursion_id"].n_unique() != primary.height:
        raise ValueError("primary observable excursion IDs are not unique")
    outside = primary.select(*keys).unique().join(
        manifest.select(*keys), on=keys, how="anti"
    )
    if not outside.is_empty():
        raise ValueError("primary excursions contain facts outside fixed manifest")
    invalid_amplitude = primary.filter(
        ~pl.col("amplitude_bp").is_finite() | (pl.col("amplitude_bp") <= 0)
    )
    if not invalid_amplitude.is_empty():
        raise ValueError("primary observable excursion amplitude is invalid")
    canonical_joined = primary.join(
        canonical.select(*keys, "canonical_60_upper_bp"),
        on=keys,
        how="left",
        validate="m:1",
    )
    original_boundary_mismatch = canonical_joined.filter(
        (pl.col("upper_distance_bp") - pl.col("canonical_60_upper_bp")).abs()
        > 1e-12
    )
    if not original_boundary_mismatch.is_empty():
        raise ValueError("canonical excursion and product-day boundary disagree")
    canonical_touch_by_amplitude = (
        pl.col("amplitude_bp") >= pl.col("canonical_60_upper_bp")
    )
    canonical_touch_mismatch = canonical_joined.filter(
        canonical_touch_by_amplitude
        != pl.col("touch_time_ns").is_not_null()
    )
    if not canonical_touch_mismatch.is_empty():
        raise ValueError("canonical touch cursor and amplitude crossing disagree")

    canonical_stats = canonical_joined.group_by(*keys).agg(
        pl.len().cast(pl.Int64).alias("observable_excursions"),
        pl.col("touch_time_ns")
        .is_not_null()
        .sum()
        .cast(pl.Int64)
        .alias("canonical_60_touches_all"),
    )
    challenger_joined = primary.join(
        base.select(
            *keys,
            "challenger_30_upper_bp",
            "challenger_boundary_valid",
        ),
        on=keys,
        how="left",
        validate="m:1",
    ).filter(pl.col("challenger_boundary_valid"))
    challenger_stats = challenger_joined.group_by(*keys).agg(
        (pl.col("amplitude_bp") >= pl.col("challenger_30_upper_bp"))
        .sum()
        .cast(pl.Int64)
        .alias("challenger_30_hypothetical_touches")
    )
    result = (
        base.join(canonical_stats, on=keys, how="left", validate="1:1")
        .join(challenger_stats, on=keys, how="left", validate="1:1")
        .with_columns(
            pl.col("observable_excursions").fill_null(0).cast(pl.Int64),
            pl.col("canonical_60_touches_all").fill_null(0).cast(pl.Int64),
            (
                pl.col("canonical_boundary_valid")
                & pl.col("challenger_boundary_valid").fill_null(False)
            ).alias("comparison_eligible"),
        )
        .with_columns(
            pl.when(pl.col("comparison_eligible"))
            .then(pl.col("canonical_60_touches_all"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("canonical_60_touches_common_denominator"),
            pl.when(pl.col("comparison_eligible"))
            .then(
                pl.col("challenger_30_hypothetical_touches")
                .fill_null(0)
                .cast(pl.Int64)
            )
            .otherwise(None)
            .cast(pl.Int64)
            .alias("challenger_30_touches_common_denominator"),
            pl.col("Date").str.slice(0, 6).alias("month"),
            pl.lit(True).alias("fixed_manifest_member"),
            pl.lit(True).alias("market_only"),
            pl.lit(False).alias("scheduler_included"),
            pl.lit(False).alias("queue_included"),
            pl.lit(False).alias("makerfill_included"),
            pl.lit(False).alias("shortlist_eligible"),
        )
        .with_columns(
            (
                pl.col("canonical_60_touches_common_denominator") > 0
            ).alias("canonical_60_any_touch_common_denominator"),
            (
                pl.col("challenger_30_touches_common_denominator") > 0
            ).alias("challenger_30_any_touch_common_denominator"),
        )
        .drop("challenger_30_hypothetical_touches")
        .sort(keys)
    )
    _validate_product_day_facts(result, manifest)
    return result


def summarize_monthly_boundary_sensitivity(
    product_days: pl.DataFrame,
) -> pl.DataFrame:
    """Compare 60/30 q95 on exactly the same valid product-day denominator."""

    _validate_product_day_facts(product_days)
    rows = [
        _summary_row(month, group)
        for (month,), group in product_days.sort("month").group_by(
            "month", maintain_order=True
        )
    ]
    return pl.from_dicts(rows, infer_schema_length=None).sort("month")


def summarize_common_membership_sensitivity(
    product_days: pl.DataFrame,
    *,
    expected_common_product_count: int | None = EXPECTED_COMMON_PRODUCT_COUNT,
) -> pl.DataFrame:
    """Repeat the Jul/Aug comparison on their fixed common product membership."""

    july = set(
        product_days.filter(pl.col("month") == "202607")["ValueCode"].unique()
    )
    august = set(
        product_days.filter(pl.col("month") == "202608")["ValueCode"].unique()
    )
    common = sorted(july & august)
    if not common:
        raise ValueError("Jul/Aug fixed universe has no common products")
    if (
        expected_common_product_count is not None
        and len(common) != expected_common_product_count
    ):
        raise ValueError("Jul/Aug common product count drift")
    selected = product_days.filter(
        pl.col("month").is_in(["202607", "202608"])
        & pl.col("ValueCode").is_in(common)
    )
    return summarize_monthly_boundary_sensitivity(selected).with_columns(
        pl.lit("jul_aug_common_products").alias("sensitivity_id"),
        pl.lit(len(common), dtype=pl.Int64).alias("common_product_count"),
    ).select(
        "sensitivity_id",
        "common_product_count",
        pl.all().exclude("sensitivity_id", "common_product_count"),
    )


def _summary_row(month: str, product_days: pl.DataFrame) -> dict[str, object]:
    common = product_days.filter(pl.col("comparison_eligible"))
    if common.is_empty():
        raise ValueError(f"{month}: no common-valid product-day denominator")
    excluded = product_days.filter(~pl.col("comparison_eligible"))
    observable = int(common["observable_excursions"].sum() or 0)
    canonical_touches = int(
        common["canonical_60_touches_common_denominator"].sum() or 0
    )
    challenger_touches = int(
        common["challenger_30_touches_common_denominator"].sum() or 0
    )
    product_day_count = common.height
    canonical_any = int(
        common["canonical_60_any_touch_common_denominator"].sum() or 0
    )
    challenger_any = int(
        common["challenger_30_any_touch_common_denominator"].sum() or 0
    )
    canonical_rate = _ratio(canonical_touches, observable)
    challenger_rate = _ratio(challenger_touches, observable)
    canonical_per_pd = _ratio(canonical_touches, product_day_count)
    challenger_per_pd = _ratio(challenger_touches, product_day_count)
    canonical_any_rate = _ratio(canonical_any, product_day_count)
    challenger_any_rate = _ratio(challenger_any, product_day_count)
    row: dict[str, object] = {
        "month": month,
        "session_count": product_days["Date"].n_unique(),
        "fixed_manifest_product_days": product_days.height,
        "common_valid_product_days": product_day_count,
        "challenger_invalid_product_days": excluded.height,
        "comparison_coverage_rate": _ratio(product_day_count, product_days.height),
        "excluded_observable_excursions": int(
            excluded["observable_excursions"].sum() or 0
        ),
        "common_observable_excursions": observable,
        "canonical_60_boundary_p50_bp": _quantile(
            common["canonical_60_upper_bp"], 0.50
        ),
        "canonical_60_boundary_p80_bp": _quantile(
            common["canonical_60_upper_bp"], 0.80
        ),
        "canonical_60_boundary_p95_bp": _quantile(
            common["canonical_60_upper_bp"], 0.95
        ),
        "challenger_30_boundary_p50_bp": _quantile(
            common["challenger_30_upper_bp"], 0.50
        ),
        "challenger_30_boundary_p80_bp": _quantile(
            common["challenger_30_upper_bp"], 0.80
        ),
        "challenger_30_boundary_p95_bp": _quantile(
            common["challenger_30_upper_bp"], 0.95
        ),
        "canonical_60_touches": canonical_touches,
        "challenger_30_hypothetical_touches": challenger_touches,
        "canonical_60_touch_rate": canonical_rate,
        "challenger_30_hypothetical_touch_rate": challenger_rate,
        "touch_rate_change_30_minus_60": _difference(
            challenger_rate, canonical_rate
        ),
        "canonical_60_touches_per_product_day": canonical_per_pd,
        "challenger_30_hypothetical_touches_per_product_day": challenger_per_pd,
        "touches_per_product_day_change_30_minus_60": _difference(
            challenger_per_pd, canonical_per_pd
        ),
        "canonical_60_product_days_with_any_touch": canonical_any,
        "challenger_30_product_days_with_any_touch": challenger_any,
        "canonical_60_any_touch_product_day_rate": canonical_any_rate,
        "challenger_30_any_touch_product_day_rate": challenger_any_rate,
        "any_touch_product_day_rate_change_30_minus_60": _difference(
            challenger_any_rate, canonical_any_rate
        ),
        "market_only": True,
        "hypothetical_touch_existence_only": True,
        "scheduler_included": False,
        "queue_included": False,
        "makerfill_included": False,
        "shortlist_eligible": False,
    }
    for quantile in (50, 80, 95):
        row[f"boundary_p{quantile}_change_30_minus_60_bp"] = _difference(
            row[f"challenger_30_boundary_p{quantile}_bp"],
            row[f"canonical_60_boundary_p{quantile}_bp"],
        )
    return row


def _validate_product_day_facts(
    product_days: pl.DataFrame,
    manifest: pl.DataFrame | None = None,
) -> None:
    keys = ["Date", "ValueCode", "QuoteCode"]
    required = {
        *keys,
        "month",
        "canonical_60_upper_bp",
        "challenger_30_upper_bp",
        "canonical_boundary_valid",
        "challenger_boundary_valid",
        "comparison_eligible",
        "observable_excursions",
        "canonical_60_touches_all",
        "canonical_60_touches_common_denominator",
        "challenger_30_touches_common_denominator",
        "canonical_60_any_touch_common_denominator",
        "challenger_30_any_touch_common_denominator",
        "market_only",
        "scheduler_included",
        "queue_included",
        "makerfill_included",
        "shortlist_eligible",
    }
    _require_columns(product_days, required, "product-day sensitivity")
    if product_days.select(*keys).n_unique() != product_days.height:
        raise ValueError("product-day sensitivity keys are duplicated")
    if manifest is not None and (
        product_days.height != manifest.height
        or not product_days.select(*keys).equals(manifest.select(*keys))
    ):
        raise ValueError("product-day sensitivity changed fixed universe")
    invalid_static = product_days.filter(
        ~pl.col("canonical_boundary_valid")
        | (
            pl.col("comparison_eligible")
            != (
                pl.col("canonical_boundary_valid")
                & pl.col("challenger_boundary_valid")
            )
        )
        | ~pl.col("market_only")
        | pl.col("scheduler_included")
        | pl.col("queue_included")
        | pl.col("makerfill_included")
        | pl.col("shortlist_eligible")
        | (pl.col("month") != pl.col("Date").str.slice(0, 6))
        | (pl.col("observable_excursions") < 0)
        | (pl.col("canonical_60_touches_all") < 0)
        | (
            pl.col("canonical_60_touches_all")
            > pl.col("observable_excursions")
        )
    )
    if not invalid_static.is_empty():
        raise ValueError("product-day sensitivity static invariant failed")
    valid = product_days.filter(pl.col("comparison_eligible"))
    invalid = product_days.filter(~pl.col("comparison_eligible"))
    invalid_valid_counts = valid.filter(
        pl.col("canonical_60_touches_common_denominator").is_null()
        | pl.col("challenger_30_touches_common_denominator").is_null()
        | (
            pl.col("canonical_60_touches_common_denominator")
            != pl.col("canonical_60_touches_all")
        )
        | (pl.col("challenger_30_touches_common_denominator") < 0)
        | (
            pl.col("challenger_30_touches_common_denominator")
            > pl.col("observable_excursions")
        )
        | (
            pl.col("canonical_60_any_touch_common_denominator")
            != (pl.col("canonical_60_touches_common_denominator") > 0)
        )
        | (
            pl.col("challenger_30_any_touch_common_denominator")
            != (pl.col("challenger_30_touches_common_denominator") > 0)
        )
    )
    if not invalid_valid_counts.is_empty():
        raise ValueError("common-valid same-denominator touch invariant failed")
    if invalid.filter(
        pl.col("canonical_60_touches_common_denominator").is_not_null()
        | pl.col("challenger_30_touches_common_denominator").is_not_null()
        | pl.col("canonical_60_any_touch_common_denominator").is_not_null()
        | pl.col("challenger_30_any_touch_common_denominator").is_not_null()
    ).height:
        raise ValueError("invalid boundary row was silently treated as zero touch")


def _verify_domain_frames(
    boundaries: pl.DataFrame,
    product_days: pl.DataFrame,
    monthly: pl.DataFrame,
    membership: pl.DataFrame,
    *,
    sessions: Sequence[str],
    expected_product_days: int,
    expected_common_product_count: int,
    rolling_boundary_reconstruction_verified: bool = False,
    product_day_sensitivity_reconstruction_verified: bool = False,
) -> dict[str, object]:
    keys = ["Date", "ValueCode", "QuoteCode"]
    _validate_challenger_boundary_rows(boundaries)
    _validate_exact_history_windows(boundaries, sessions)
    if (
        boundaries.height != expected_product_days
        or boundaries.select(*keys).n_unique() != boundaries.height
        or product_days.height != expected_product_days
        or not boundaries.select(*keys).equals(product_days.select(*keys))
    ):
        raise ValueError("boundary/product-day fixed-universe coverage drift")
    _validate_product_day_facts(product_days)
    boundary_projection = boundaries.select(
        *keys,
        pl.col("upper_distance_bp").alias("challenger_30_upper_bp"),
        "challenger_boundary_valid",
        pl.col("source_asof_date").alias("challenger_source_asof_date"),
        pl.col("train_start_date").alias("challenger_train_start_date"),
        pl.col("train_end_date").alias("challenger_train_end_date"),
        pl.col("history_sessions_global").alias(
            "challenger_history_sessions_global"
        ),
        pl.col("history_sessions_product").alias(
            "challenger_history_sessions_product"
        ),
        pl.col("positive_completed").alias("challenger_positive_completed"),
        pl.col("negative_completed").alias("challenger_negative_completed"),
    )
    product_projection = product_days.select(*boundary_projection.columns)
    if not _frames_equal_csv_safe(product_projection, boundary_projection):
        raise ValueError("boundary/product-day challenger projection drift")
    recomputed_monthly = summarize_monthly_boundary_sensitivity(product_days)
    if not _frames_equal_csv_safe(monthly, recomputed_monthly):
        raise ValueError("monthly boundary sensitivity drift")
    recomputed_membership = summarize_common_membership_sensitivity(
        product_days,
        expected_common_product_count=expected_common_product_count,
    )
    if not _frames_equal_csv_safe(membership, recomputed_membership):
        raise ValueError("common-membership boundary sensitivity drift")
    invalid = product_days.filter(~pl.col("comparison_eligible"))
    return {
        "status": "pass",
        "market_only": True,
        "scheduler_included": False,
        "queue_included": False,
        "makerfill_included": False,
        "shortlist_eligible": False,
        "fixed_manifest_product_days": product_days.height,
        "challenger_valid_product_days": product_days.filter(
            pl.col("comparison_eligible")
        ).height,
        "challenger_invalid_product_days": invalid.height,
        "excluded_observable_excursions": int(
            invalid["observable_excursions"].sum() or 0
        ),
        "jul_aug_common_product_count": expected_common_product_count,
        "same_denominator_verified": True,
        "invalid_boundary_not_zero_filled_verified": True,
        "causal_boundary_row_contract_verified": True,
        "exact_d_minus_one_30_session_window_verified": True,
        "rolling_boundary_reconstruction_verified": bool(
            rolling_boundary_reconstruction_verified
        ),
        "product_day_sensitivity_reconstruction_verified": bool(
            product_day_sensitivity_reconstruction_verified
        ),
        "summaries_recomputed": True,
        "raw_touch_cursor_reconstructed": False,
        "hypothetical_touch_existence_only": True,
    }


def _verify_rolling_boundary_reconstruction(
    daily_root: Path,
    sessions: Sequence[str],
    manifest: pl.DataFrame,
    published: pl.DataFrame,
) -> bool:
    """Re-read daily facts and reproduce every published q95 boundary value."""

    _, excursion_paths, mapping_paths = validated_daily_paths(
        daily_root,
        sessions,
    )
    daily_excursions, daily_mapping = load_daily_boundary_inputs(
        excursion_paths,
        mapping_paths,
    )
    reconstructed = build_fixed_challenger_boundaries(
        daily_excursions,
        daily_mapping,
        sessions,
        manifest,
    )
    if not _frames_equal_csv_safe(published, reconstructed):
        raise ValueError("30-session rolling boundary reconstruction drift")
    return True


def _verify_product_day_sensitivity_reconstruction(
    canonical_s0_root: Path,
    manifest: pl.DataFrame,
    boundaries: pl.DataFrame,
    published: pl.DataFrame,
) -> bool:
    """Rebuild the common-denominator touch counts from canonical S0 facts."""

    canonical_excursions = pl.read_parquet(
        Path(canonical_s0_root) / "market_excursions.parquet"
    )
    canonical_coverage = pl.read_parquet(
        Path(canonical_s0_root) / "product_day_coverage.parquet"
    )
    reconstructed = build_product_day_sensitivity(
        manifest,
        canonical_coverage,
        canonical_excursions,
        boundaries,
    )
    if not _frames_equal_csv_safe(published, reconstructed):
        raise ValueError("30-session product-day sensitivity reconstruction drift")
    return True


def run(
    paths: SensitivityPaths | None = None,
    *,
    run_focused_tests: bool = True,
) -> Mapping[str, object]:
    """Build one fresh immutable market-only sensitivity bundle."""

    paths = SensitivityPaths() if paths is None else paths
    if paths.output_root.exists():
        raise FileExistsError(paths.output_root)
    started = time.perf_counter()
    git_state = _git_state()
    tests = _run_focused_tests() if run_focused_tests else {
        "status": "skipped_debug",
        "command": None,
        "returncode": None,
        "output_tail": None,
    }
    probe_manifest = load_fixed_manifest(paths.manifest_path)
    probe_sessions = load_session_calendar(paths.sessions_path)
    markers, excursion_paths, mapping_paths = _declared_daily_paths(
        paths.daily_root,
        probe_sessions,
    )
    input_paths = [
        paths.manifest_path,
        paths.sessions_path,
        paths.canonical_s0_root / "complete.json",
        paths.canonical_s0_root / "market_excursions.parquet",
        paths.canonical_s0_root / "product_day_coverage.parquet",
        *markers,
        *excursion_paths,
        *mapping_paths,
        *(marker.parent / "causal_fair.parquet" for marker in markers),
        *(marker.parent / "audit.parquet" for marker in markers),
    ]
    input_inventory = _input_inventory(input_paths)
    manifest = load_fixed_manifest(paths.manifest_path)
    sessions = load_session_calendar(paths.sessions_path)
    if (
        not manifest.equals(probe_manifest, null_equal=True)
        or sessions != probe_sessions
    ):
        raise ValueError("manifest/session calendar changed during inventory probe")
    del probe_manifest, probe_sessions
    target_dates = manifest["Date"].unique().sort().to_list()
    missing_dates = sorted(set(target_dates) - set(sessions))
    if missing_dates:
        raise ValueError(f"fixed target dates absent from calendar: {missing_dates}")

    canonical_verification = verify_canonical_s0_bundle(
        paths.canonical_s0_root,
        verify_inputs=True,
    )
    canonical_marker = json.loads(
        (paths.canonical_s0_root / "complete.json").read_text()
    )
    if (
        canonical_marker.get("run_id") != EXPECTED_CANONICAL_S0_RUN_ID
        or canonical_marker.get("canonical_eligible") is not True
        or canonical_marker.get("dirty") is not False
        or canonical_marker.get("product_day_count")
        != EXPECTED_PRODUCT_DAY_COUNT
        or canonical_verification.get("status") != "pass"
        or canonical_verification.get("canonical_eligible") is not True
        or canonical_verification.get("raw_tape_reconstruction_verified")
        is not True
        or canonical_verification.get("input_content_rehashed") is not True
    ):
        raise ValueError("source S0 bundle is not canonical and verified")

    validated_paths = validated_daily_paths(paths.daily_root, sessions)
    if validated_paths != (markers, excursion_paths, mapping_paths):
        raise ValueError("daily partition paths changed during inventory probe")
    daily_excursions, daily_mapping = load_daily_boundary_inputs(
        excursion_paths, mapping_paths
    )
    boundaries = build_fixed_challenger_boundaries(
        daily_excursions,
        daily_mapping,
        sessions,
        manifest,
    )
    del daily_excursions, daily_mapping
    canonical_excursions = pl.read_parquet(
        paths.canonical_s0_root / "market_excursions.parquet"
    )
    canonical_coverage = pl.read_parquet(
        paths.canonical_s0_root / "product_day_coverage.parquet"
    )
    product_days = build_product_day_sensitivity(
        manifest,
        canonical_coverage,
        canonical_excursions,
        boundaries,
    )
    monthly = summarize_monthly_boundary_sensitivity(product_days)
    membership = summarize_common_membership_sensitivity(product_days)
    rolling_boundary_reconstruction_verified = (
        _verify_rolling_boundary_reconstruction(
            paths.daily_root,
            sessions,
            manifest,
            boundaries,
        )
    )
    product_day_sensitivity_reconstruction_verified = (
        _verify_product_day_sensitivity_reconstruction(
            paths.canonical_s0_root,
            manifest,
            boundaries,
            product_days,
        )
    )
    verification = _verify_domain_frames(
        boundaries,
        product_days,
        monthly,
        membership,
        sessions=sessions,
        expected_product_days=manifest.height,
        expected_common_product_count=EXPECTED_COMMON_PRODUCT_COUNT,
        rolling_boundary_reconstruction_verified=(
            rolling_boundary_reconstruction_verified
        ),
        product_day_sensitivity_reconstruction_verified=(
            product_day_sensitivity_reconstruction_verified
        ),
    )
    _assert_inventory_current(input_inventory)
    if _git_state() != git_state:
        raise ValueError("git state changed during sensitivity build")
    run_config = _run_config(
        paths,
        sessions,
        target_dates,
        manifest,
        canonical_marker,
        canonical_verification,
        git_state,
        tests,
        input_inventory,
        verification,
    )
    marker = _publish_bundle(
        paths.output_root,
        {
            BOUNDARY_ARTIFACT: boundaries,
            PRODUCT_DAY_ARTIFACT: product_days,
            MONTHLY_ARTIFACT: monthly,
            MEMBERSHIP_ARTIFACT: membership,
        },
        run_config,
        verification,
        elapsed_seconds=time.perf_counter() - started,
    )
    if marker.get("canonical_eligible") is True:
        verify_bundle(paths.output_root)
    else:
        _verify_bundle_artifacts(
            paths.output_root,
            verify_inputs=False,
            require_canonical=False,
        )
    return marker


def _run_config(
    paths: SensitivityPaths,
    sessions: Sequence[str],
    target_dates: Sequence[str],
    manifest: pl.DataFrame,
    canonical_marker: Mapping[str, object],
    canonical_verification: Mapping[str, object],
    git_state: Mapping[str, object],
    tests: Mapping[str, object],
    input_inventory: Mapping[str, object],
    verification: Mapping[str, object],
) -> dict[str, object]:
    fixed_manifest = {
        "sha256": MANIFEST_SHA256,
        "key_sha256": _manifest_key_sha256(manifest),
        "product_days": manifest.height,
        "entry_sessions": len(target_dates),
        "first_date": target_dates[0],
        "last_date": target_dates[-1],
        "jul_aug_common_product_count": EXPECTED_COMMON_PRODUCT_COUNT,
    }
    history_calendar = {
        "session_count": len(sessions),
        "first_date": sessions[0],
        "last_date": sessions[-1],
        "dates": list(sessions),
        "source_file_sha256": _inventory_record_for_path(
            input_inventory,
            paths.sessions_path,
        )["sha256"],
    }
    canonical_source = {
        "run_id": canonical_marker.get("run_id"),
        "nested_repo_commit": canonical_marker.get("nested_repo_commit"),
        "marker_payload_sha256": canonical_marker.get(
            "marker_payload_sha256"
        ),
        "complete_sha256": canonical_verification.get("complete_sha256"),
        "market_excursions_sha256": _inventory_record_for_path(
            input_inventory,
            paths.canonical_s0_root / "market_excursions.parquet",
        )["sha256"],
        "product_day_coverage_sha256": _inventory_record_for_path(
            input_inventory,
            paths.canonical_s0_root / "product_day_coverage.parquet",
        )["sha256"],
        "canonical_eligible": canonical_verification.get(
            "canonical_eligible"
        ),
        "dirty": canonical_marker.get("dirty"),
        "verification_status": canonical_verification.get("status"),
        "raw_tape_reconstruction_verified": canonical_verification.get(
            "raw_tape_reconstruction_verified"
        ),
        "input_content_rehashed": canonical_verification.get(
            "input_content_rehashed"
        ),
    }
    result: dict[str, object] = {
        "runner_version": RUNNER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "market_only": True,
        "scheduler_included": False,
        "queue_included": False,
        "makerfill_included": False,
        "capital_included": False,
        "shortlist_eligible": False,
        "purpose": "S0 boundary-lag sensitivity only",
        "hypothetical_touch_contract": (
            "primary_observable amplitude_bp >= 30-session D-1 q95 upper; "
            "existence only, no reconstructed cursor"
        ),
        "comparison_denominator": (
            "fixed-manifest product-days where canonical 60-session and "
            "challenger 30-session q95 are both valid; invalid challenger "
            "rows retained as coverage and excluded from both arms"
        ),
        "rolling_boundary_config": _rolling_config_payload(
            CHALLENGER_CONFIG
        ),
        "fixed_manifest": fixed_manifest,
        "history_calendar": history_calendar,
        "canonical_s0_source": canonical_source,
        "paths": {key: str(value.resolve()) for key, value in asdict(paths).items()},
        "git": dict(git_state),
        "focused_tests": dict(tests),
        "input_inventory": dict(input_inventory),
        "legacy_input_provenance_caveat": (
            "daily partitions use migrated_nonatomic markers; file content "
            "hashes bind mapping/excursion artifacts, but legacy atomic "
            "publish was not verified"
        ),
        "input_hash_contract": "full-content SHA-256 for every consumed input",
    }
    result["canonical_checks"] = _canonical_checks_from_payload(
        result,
        verification,
    )
    return result


def _canonical_checks_from_payload(
    run_config: Mapping[str, object],
    verification: Mapping[str, object],
) -> dict[str, bool]:
    git_state = run_config.get("git")
    tests = run_config.get("focused_tests")
    fixed_manifest = run_config.get("fixed_manifest")
    history = run_config.get("history_calendar")
    source = run_config.get("canonical_s0_source")
    inventory = run_config.get("input_inventory")
    configured_paths = run_config.get("paths")
    commit = git_state.get("commit") if isinstance(git_state, dict) else None
    return {
        "clean_git_worktree": (
            isinstance(git_state, dict) and git_state.get("dirty") is False
        ),
        "focused_tests_passed": (
            isinstance(tests, dict) and tests.get("status") == "pass"
        ),
        "fixed_manifest_verified": (
            fixed_manifest == _expected_fixed_manifest_payload()
        ),
        "history_calendar_bound": _history_calendar_payload_valid(history),
        "fixed_history_calendar_verified": (
            isinstance(history, dict)
            and history.get("session_count") == EXPECTED_HISTORY_SESSION_COUNT
            and history.get("first_date") == EXPECTED_HISTORY_FIRST_DATE
            and history.get("last_date") == EXPECTED_LAST_DATE
            and history.get("source_file_sha256") == SESSIONS_SHA256
        ),
        "complete_input_inventory": _input_inventory_payload_complete(
            inventory
        ),
        "default_source_and_output_paths": _default_paths_verified(
            configured_paths
        ),
        "canonical_s0_source_verified": _canonical_source_payload_valid(
            source,
            expected_commit=commit,
        ),
        "exact_d_minus_one_30_session_window_verified": (
            verification.get(
                "exact_d_minus_one_30_session_window_verified"
            )
            is True
        ),
        "rolling_boundary_values_reconstructed": (
            verification.get("rolling_boundary_reconstruction_verified")
            is True
        ),
        "product_day_touch_counts_reconstructed": (
            verification.get(
                "product_day_sensitivity_reconstruction_verified"
            )
            is True
        ),
    }


def _default_paths_verified(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    expected = asdict(SensitivityPaths())
    return set(payload) == set(expected) and all(
        Path(str(payload[key])).resolve() == Path(value).resolve()
        for key, value in expected.items()
    )


def _input_inventory_payload_complete(payload: object) -> bool:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("records"), list
    ):
        return False
    records = payload["records"]
    return (
        len(records) == EXPECTED_INPUT_RECORD_COUNT
        and len(
            {
                record.get("path")
                for record in records
                if isinstance(record, dict)
            }
        )
        == EXPECTED_INPUT_RECORD_COUNT
        and all(
            isinstance(record, dict)
            and record.get("hash_scope") == "full_content"
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            for record in records
        )
    )


def _history_calendar_payload_valid(payload: object) -> bool:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("dates"), list
    ):
        return False
    dates = payload["dates"]
    return bool(dates) and (
        all(isinstance(date, str) for date in dates)
        and dates == sorted(dates)
        and len(dates) == len(set(dates))
        and payload.get("session_count") == len(dates)
        and payload.get("first_date") == dates[0]
        and payload.get("last_date") == dates[-1]
        and isinstance(payload.get("source_file_sha256"), str)
        and len(payload["source_file_sha256"]) == 64
    )


def _canonical_source_payload_valid(
    payload: object,
    *,
    expected_commit: object,
) -> bool:
    if not isinstance(payload, dict):
        return False
    marker_digest = payload.get("marker_payload_sha256")
    complete_digest = payload.get("complete_sha256")
    market_digest = payload.get("market_excursions_sha256")
    coverage_digest = payload.get("product_day_coverage_sha256")
    return (
        payload.get("run_id") == EXPECTED_CANONICAL_S0_RUN_ID
        and isinstance(expected_commit, str)
        and payload.get("nested_repo_commit") == expected_commit
        and isinstance(marker_digest, str)
        and len(marker_digest) == 64
        and isinstance(complete_digest, str)
        and len(complete_digest) == 64
        and isinstance(market_digest, str)
        and len(market_digest) == 64
        and isinstance(coverage_digest, str)
        and len(coverage_digest) == 64
        and payload.get("canonical_eligible") is True
        and payload.get("dirty") is False
        and payload.get("verification_status") == "pass"
        and payload.get("raw_tape_reconstruction_verified") is True
        and payload.get("input_content_rehashed") is True
    )


def _publish_bundle(
    destination: Path,
    frames: Mapping[str, pl.DataFrame],
    run_config: Mapping[str, object],
    verification: Mapping[str, object],
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    expected = set(FRAME_ARTIFACTS)
    if set(frames) != expected:
        raise ValueError("sensitivity frame artifact inventory drift")
    canonical_checks = run_config.get("canonical_checks")
    if (
        not isinstance(canonical_checks, dict)
        or not canonical_checks
        or any(not isinstance(value, bool) for value in canonical_checks.values())
    ):
        raise ValueError("sensitivity canonical checks are malformed")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
    )
    try:
        for name in FRAME_ARTIFACTS:
            frame = frames[name]
            path = stage / name
            if path.suffix == ".parquet":
                frame.write_parquet(path, compression="zstd", statistics=True)
            else:
                frame.write_csv(path)
        _write_json(stage / "run_config.json", run_config)
        _write_json(stage / "verification.json", verification)
        artifacts = {
            name: _artifact_metadata(stage / name)
            for name in (*FRAME_ARTIFACTS, *NONFRAME_ARTIFACTS)
        }
        marker: dict[str, object] = {
            "complete": True,
            "runner_version": RUNNER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "run_id": destination.name,
            "market_only": True,
            "scheduler_included": False,
            "queue_included": False,
            "makerfill_included": False,
            "shortlist_eligible": False,
            "nested_repo_commit": run_config["git"]["commit"],
            "dirty": run_config["git"]["dirty"],
            "focused_tests": dict(run_config["focused_tests"]),
            "canonical_eligible": all(canonical_checks.values()),
            "canonical_checks": dict(canonical_checks),
            "config_sha256": _canonical_sha256(run_config),
            "input_inventory_sha256": run_config["input_inventory"][
                "inventory_sha256"
            ],
            "verification": dict(verification),
            "artifacts": artifacts,
            "elapsed_seconds": elapsed_seconds,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        _write_json(stage / "complete.json", marker)
        require_canonical = marker["canonical_eligible"] is True
        _verify_bundle_artifacts(
            stage,
            verify_inputs=False,
            require_canonical=require_canonical,
        )
        if require_canonical:
            _assert_inventory_current(run_config["input_inventory"])
            if _git_state() != run_config["git"]:
                raise ValueError(
                    "git state changed during staged sensitivity verification"
                )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return marker


def verify_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    verify_inputs: bool = False,
) -> Mapping[str, object]:
    """Validate one canonical immutable bundle and recompute both summaries."""

    return _verify_bundle_artifacts(
        Path(output_root),
        verify_inputs=verify_inputs,
        require_canonical=True,
    )


def _verify_bundle_artifacts(
    root: Path,
    *,
    verify_inputs: bool,
    require_canonical: bool,
) -> dict[str, object]:
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text())
    marker_digest = marker.get("marker_payload_sha256")
    marker_payload = dict(marker)
    marker_payload.pop("marker_payload_sha256", None)
    if marker_digest != _canonical_sha256(marker_payload):
        raise ValueError("sensitivity complete marker self-hash mismatch")
    if (
        marker.get("complete") is not True
        or marker.get("runner_version") != RUNNER_VERSION
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("market_only") is not True
        or marker.get("scheduler_included") is not False
        or marker.get("queue_included") is not False
        or marker.get("makerfill_included") is not False
        or marker.get("shortlist_eligible") is not False
    ):
        raise ValueError("unsupported or mislabelled sensitivity bundle")
    expected_artifacts = {*FRAME_ARTIFACTS, *NONFRAME_ARTIFACTS}
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("sensitivity artifact inventory drift")
    expected_files = {*expected_artifacts, "complete.json"}
    actual_files = {path.name for path in root.iterdir()}
    if actual_files != expected_files:
        raise ValueError("sensitivity bundle file inventory drift")
    for name, expected_metadata in artifacts.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid sensitivity artifact: {path}")
        if _artifact_metadata(path) != expected_metadata:
            raise ValueError(f"sensitivity artifact metadata drift: {path}")
    run_config = json.loads((root / "run_config.json").read_text())
    if marker.get("config_sha256") != _canonical_sha256(run_config):
        raise ValueError("sensitivity run_config hash drift")
    if (
        run_config.get("runner_version") != RUNNER_VERSION
        or run_config.get("schema_version") != SCHEMA_VERSION
        or run_config.get("market_only") is not True
        or run_config.get("scheduler_included") is not False
        or run_config.get("queue_included") is not False
        or run_config.get("makerfill_included") is not False
        or run_config.get("shortlist_eligible") is not False
        or run_config.get("rolling_boundary_config")
        != _rolling_config_payload(CHALLENGER_CONFIG)
    ):
        raise ValueError("sensitivity run_config semantic drift")
    canonical_checks = run_config.get("canonical_checks")
    if (
        not isinstance(canonical_checks, dict)
        or not canonical_checks
        or any(not isinstance(value, bool) for value in canonical_checks.values())
        or marker.get("canonical_checks") != canonical_checks
    ):
        raise ValueError("sensitivity canonical checks drift")
    canonical_eligible = all(canonical_checks.values())
    git_state = run_config.get("git")
    focused_tests = run_config.get("focused_tests")
    if (
        marker.get("canonical_eligible") is not canonical_eligible
        or not isinstance(git_state, dict)
        or marker.get("nested_repo_commit") != git_state.get("commit")
        or marker.get("dirty") is not git_state.get("dirty")
        or marker.get("focused_tests") != focused_tests
    ):
        raise ValueError("sensitivity marker provenance drift")
    history = run_config.get("history_calendar")
    if not _history_calendar_payload_valid(history):
        raise ValueError("sensitivity history calendar drift")
    sessions = history["dates"]
    inventory = run_config.get("input_inventory")
    _validate_inventory_payload(inventory)
    if marker.get("input_inventory_sha256") != inventory["inventory_sha256"]:
        raise ValueError("sensitivity input inventory marker drift")
    if verify_inputs:
        _assert_inventory_current(inventory)

    boundaries = pl.read_parquet(root / BOUNDARY_ARTIFACT)
    product_days = pl.read_parquet(root / PRODUCT_DAY_ARTIFACT)
    monthly = pl.read_csv(root / MONTHLY_ARTIFACT).with_columns(
        pl.col("month").cast(pl.String)
    )
    membership = pl.read_csv(root / MEMBERSHIP_ARTIFACT).with_columns(
        pl.col("month").cast(pl.String)
    )
    published = json.loads((root / "verification.json").read_text())
    rolling_boundary_reconstruction_verified = False
    product_day_sensitivity_reconstruction_verified = False
    reconstruct_published_facts = require_canonical or (
        published.get("rolling_boundary_reconstruction_verified") is True
        or published.get(
            "product_day_sensitivity_reconstruction_verified"
        )
        is True
    )
    if reconstruct_published_facts:
        configured_paths = run_config.get("paths")
        if not isinstance(configured_paths, dict):
            raise ValueError("canonical sensitivity configured paths are malformed")
        manifest = load_fixed_manifest(Path(configured_paths["manifest_path"]))
        current_sessions = load_session_calendar(
            Path(configured_paths["sessions_path"])
        )
        if current_sessions != sessions:
            raise ValueError("canonical sensitivity session calendar content drift")
        rolling_boundary_reconstruction_verified = (
            _verify_rolling_boundary_reconstruction(
                Path(configured_paths["daily_root"]),
                sessions,
                manifest,
                boundaries,
            )
        )
        product_day_sensitivity_reconstruction_verified = (
            _verify_product_day_sensitivity_reconstruction(
                Path(configured_paths["canonical_s0_root"]),
                manifest,
                boundaries,
                product_days,
            )
        )
    verification = _verify_domain_frames(
        boundaries,
        product_days,
        monthly,
        membership,
        sessions=sessions,
        expected_product_days=int(
            run_config["fixed_manifest"]["product_days"]
        ),
        expected_common_product_count=int(
            run_config["fixed_manifest"]["jul_aug_common_product_count"]
        ),
        rolling_boundary_reconstruction_verified=(
            rolling_boundary_reconstruction_verified
        ),
        product_day_sensitivity_reconstruction_verified=(
            product_day_sensitivity_reconstruction_verified
        ),
    )
    if verification != published or verification != marker.get("verification"):
        raise ValueError("sensitivity verification payload drift")
    expected_checks = _canonical_checks_from_payload(run_config, verification)
    if canonical_checks != expected_checks:
        raise ValueError("sensitivity canonical checks are not reproducible")
    if require_canonical:
        if marker.get("run_id") != DEFAULT_OUTPUT_ROOT.name:
            raise ValueError("canonical sensitivity run-id contract drift")
        _validate_canonical_bundle_contract(
            run_config,
            inventory,
            boundaries,
            product_days,
        )
        if not canonical_eligible:
            raise ValueError("sensitivity bundle is not canonical eligible")
    if verify_inputs:
        _assert_inventory_current(inventory)
    return {
        **verification,
        "canonical_eligible": canonical_eligible,
        "canonical_checks": canonical_checks,
        "bundle": str(root.resolve()),
        "complete_sha256": _sha256_file(marker_path),
        "marker_payload_sha256": marker_digest,
        "input_content_rehashed": verify_inputs,
    }


def _validate_canonical_bundle_contract(
    run_config: Mapping[str, object],
    inventory: Mapping[str, object],
    boundaries: pl.DataFrame,
    product_days: pl.DataFrame,
) -> None:
    """Reject self-described debug populations from the public verifier."""

    if run_config.get("fixed_manifest") != _expected_fixed_manifest_payload():
        raise ValueError("canonical sensitivity fixed-manifest contract drift")
    history = run_config.get("history_calendar")
    if (
        not _history_calendar_payload_valid(history)
        or history.get("session_count") != EXPECTED_HISTORY_SESSION_COUNT
        or history.get("first_date") != EXPECTED_HISTORY_FIRST_DATE
        or history.get("last_date") != EXPECTED_LAST_DATE
        or history.get("source_file_sha256") != SESSIONS_SHA256
        or EXPECTED_FIRST_DATE not in history["dates"]
        or EXPECTED_LAST_DATE not in history["dates"]
    ):
        raise ValueError("canonical sensitivity history-calendar contract drift")
    if (
        _manifest_key_sha256(boundaries) != MANIFEST_KEY_SHA256
        or _manifest_key_sha256(product_days) != MANIFEST_KEY_SHA256
    ):
        raise ValueError("canonical sensitivity exact-contract population drift")
    configured_paths = run_config.get("paths")
    if not isinstance(configured_paths, dict):
        raise ValueError("canonical sensitivity configured paths are malformed")
    manifest_record = _inventory_record_for_path(
        inventory,
        Path(configured_paths["manifest_path"]),
    )
    sessions_record = _inventory_record_for_path(
        inventory,
        Path(configured_paths["sessions_path"]),
    )
    canonical_marker_record = _inventory_record_for_path(
        inventory,
        Path(configured_paths["canonical_s0_root"]) / "complete.json",
    )
    canonical_market_record = _inventory_record_for_path(
        inventory,
        Path(configured_paths["canonical_s0_root"])
        / "market_excursions.parquet",
    )
    canonical_coverage_record = _inventory_record_for_path(
        inventory,
        Path(configured_paths["canonical_s0_root"])
        / "product_day_coverage.parquet",
    )
    if (
        manifest_record.get("sha256") != MANIFEST_SHA256
        or sessions_record.get("sha256") != SESSIONS_SHA256
        or not _input_inventory_payload_complete(inventory)
    ):
        raise ValueError("canonical sensitivity manifest inventory drift")
    source = run_config.get("canonical_s0_source")
    git_state = run_config.get("git")
    expected_commit = (
        git_state.get("commit") if isinstance(git_state, dict) else None
    )
    if not _canonical_source_payload_valid(
        source,
        expected_commit=expected_commit,
    ):
        raise ValueError("canonical sensitivity S0 source contract drift")
    if (
        history["source_file_sha256"] != sessions_record.get("sha256")
        or source["complete_sha256"] != canonical_marker_record.get("sha256")
        or source["market_excursions_sha256"]
        != canonical_market_record.get("sha256")
        or source["product_day_coverage_sha256"]
        != canonical_coverage_record.get("sha256")
    ):
        raise ValueError("canonical sensitivity input-role hash linkage drift")


def _input_inventory(paths: Iterable[Path]) -> dict[str, object]:
    records = [_file_inventory(path) for path in paths]
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("input inventory contains duplicate paths")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "records": records,
        "record_count": len(records),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _inventory_record_for_path(
    inventory: Mapping[str, object],
    path: Path,
) -> Mapping[str, object]:
    records = inventory.get("records")
    if not isinstance(records, list):
        raise TypeError("sensitivity input inventory records are malformed")
    resolved = str(Path(path).resolve())
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("path") == resolved
    ]
    if len(matches) != 1:
        raise ValueError(f"sensitivity input role is not uniquely bound: {path}")
    return matches[0]


def _file_inventory(path: Path) -> dict[str, object]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "hash_scope": "full_content",
        "sha256": _sha256_file(path),
    }


def _validate_inventory_payload(inventory: object) -> None:
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("records"), list
    ):
        raise TypeError("sensitivity input inventory is malformed")
    records = inventory["records"]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("sensitivity input inventory record is malformed")
    if (
        inventory.get("record_count") != len(records)
        or inventory.get("inventory_sha256") != digest
        or len({record.get("path") for record in records}) != len(records)
        or any(
            record.get("hash_scope") != "full_content"
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            for record in records
        )
    ):
        raise ValueError("sensitivity input inventory digest/contract drift")


def _assert_inventory_current(inventory: Mapping[str, object]) -> None:
    _validate_inventory_payload(inventory)
    for record in inventory["records"]:  # type: ignore[index]
        path = Path(record["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sensitivity input disappeared: {path}")
        stat = path.stat()
        if (
            stat.st_size != record["bytes"]
            or stat.st_mtime_ns != record["mtime_ns"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"sensitivity input changed during/after run: {path}")


def _artifact_metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        result.update(
            {
                "rows": int(pl.scan_parquet(path).select(pl.len()).collect().item()),
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


def _rolling_config_payload(
    config: RollingBoundaryConfig,
) -> dict[str, object]:
    payload = asdict(config)
    payload["quantiles"] = list(config.quantiles)
    return payload


def _frames_equal_csv_safe(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    if left.columns != right.columns or left.shape != right.shape:
        return False
    try:
        converted = left.cast(right.schema)
    except (TypeError, ValueError):
        return False
    for name, dtype in right.schema.items():
        left_column = converted[name]
        right_column = right[name]
        if left_column.is_null().to_list() != right_column.is_null().to_list():
            return False
        if dtype in (pl.Float32, pl.Float64):
            difference = (left_column - right_column).abs().fill_null(0.0)
            if bool((difference > 1e-12).any()):
                return False
        elif not left_column.equals(right_column, null_equal=True):
            return False
    return True


def _quantile(series: pl.Series, probability: float) -> float | None:
    if series.is_empty():
        return None
    value = series.quantile(probability, interpolation="nearest")
    return float(value) if value is not None else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _difference(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _require_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _run_focused_tests() -> dict[str, object]:
    command = [sys.executable, "-m", "unittest", *FOCUSED_TEST_MODULES, "-v"]
    result = subprocess.run(
        command,
        cwd=MAKER_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"focused sensitivity tests failed:\n{output[-8000:]}")
    return {
        "status": "pass",
        "command": " ".join(command),
        "modules": list(FOCUSED_TEST_MODULES),
        "returncode": result.returncode,
        "output_tail": output[-4000:],
    }


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
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


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
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument(
        "--canonical-s0-root", type=Path, default=DEFAULT_CANONICAL_S0_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-focused-tests", action="store_true")
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
    marker = run(
        SensitivityPaths(
            manifest_path=args.manifest,
            daily_root=args.daily_root,
            sessions_path=args.sessions,
            canonical_s0_root=args.canonical_s0_root,
            output_root=args.output,
        ),
        run_focused_tests=not args.skip_focused_tests,
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
