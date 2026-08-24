"""Daily walk-forward empirical basis boundaries.

The existing :mod:`adaptive` pilot deliberately used only the immediately
preceding session.  This module implements the production-like alternative:
for target session ``D`` it estimates each product's positive and negative
residual excursion distribution from a trailing window of sessions strictly
before ``D``.  The resulting snapshot is safe to materialise before the open;
it contains no target-day outcome.

The table is still a latent price-path prior.  It does not contain maker fill,
hedge, exit execution, fees, tax, overnight value, or EV.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from .daily_facts import (
    DEFAULT_DAILY_ROOT,
    load_partitioned_excursions,
    load_partitioned_mapping,
)
from .table import add_reference_tick_columns


DEFAULT_QUANTILES = (50, 80, 95)
ROLLING_BOUNDARY_SCHEMA_VERSION = "rolling_boundary_snapshots_v2_price_ladder"
ROLLING_BOUNDARY_ARTIFACT = "rolling_boundary_snapshots.parquet"
ROLLING_BOUNDARY_CSV_ARTIFACT = "rolling_boundary_snapshots.csv"
ROLLING_BOUNDARY_CONFIG_ARTIFACT = "config.json"
ROLLING_BOUNDARY_COMPLETE_MARKER = "complete.json"


@dataclass(frozen=True)
class RollingBoundaryConfig:
    """Frozen estimator settings for one walk-forward run."""

    lookback_sessions: int = 60
    min_history_sessions: int = 40
    min_excursion_history_sessions_per_side: int = 20
    min_completed_excursions_per_side: int = 100
    quantiles: tuple[int, ...] = DEFAULT_QUANTILES
    parameter_version: str = "rolling_60_session_empirical_v1"
    price_ladder_version: str = PRICE_LADDER_VERSION
    future_one_dollar_tick_effective_date: str = (
        FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    )

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_history_sessions must be in [1, lookback_sessions]"
            )
        if not 1 <= self.min_excursion_history_sessions_per_side <= self.lookback_sessions:
            raise ValueError(
                "minimum excursion history sessions must be in "
                "[1, lookback_sessions]"
            )
        if self.min_completed_excursions_per_side <= 0:
            raise ValueError("minimum completed excursions must be positive")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")
        if any(value <= 0 or value >= 100 for value in self.quantiles):
            raise ValueError("quantiles must be strictly between 0 and 100")
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError("quantiles must be unique")
        if self.price_ladder_version != PRICE_LADDER_VERSION:
            raise ValueError(
                "price_ladder_version must match the implemented quote-width "
                f"ladder: {PRICE_LADDER_VERSION}"
            )
        if (
            self.future_one_dollar_tick_effective_date
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ):
            raise ValueError(
                "future_one_dollar_tick_effective_date must match the "
                "implemented quote-width ladder: "
                f"{FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE}"
            )


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


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def build_rolling_boundary_snapshots(
    excursions: pl.DataFrame,
    target_mapping: pl.DataFrame,
    sessions: Sequence[str],
    config: RollingBoundaryConfig = RollingBoundaryConfig(),
) -> pl.DataFrame:
    """Build one pre-open-safe boundary snapshot per target product/day.

    ``excursions`` may contain all historical contracts.  History is pooled by
    ``ValueCode`` because the live target contract can roll; the snapshot keeps
    the target day's exact ``QuoteCode`` from ``target_mapping``.  Contract and
    DTE conditioned challengers should be added as separate state tables rather
    than silently splicing a different target contract into the key.
    """

    config.validate()
    sessions = _normalise_sessions(sessions)
    _require(
        excursions,
        {"Date", "ValueCode", "side", "amplitude_bp", "completed"},
        "excursions",
    )
    _require(
        target_mapping,
        {"Date", "ValueCode", "QuoteCode"},
        "target mapping",
    )
    mapping = target_mapping.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    reference_columns = {"spot_ref_price", "fut_ref_price"}
    derived_tick_columns = {
        "target_ref_future_tick_bp",
        "target_ref_future_ask_tick_bp",
        "target_ref_spot_bid_tick_bp",
        "price_ladder_version",
    }
    if reference_columns.issubset(mapping.columns):
        # Always recompute from target-day refs.  A mapping produced before the
        # 2026-07-06 rule change can contain syntactically valid but stale tick
        # columns; treating their presence as proof of correctness is unsafe.
        mapping = add_reference_tick_columns(mapping)
    elif derived_tick_columns.intersection(mapping.columns):
        raise ValueError(
            "target mapping has derived tick columns but lacks both "
            "spot_ref_price and fut_ref_price required to validate them"
        )
    key = ["Date", "ValueCode"]
    duplicate = mapping.group_by(key).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError(
            "target mapping must contain one contract per product/day: "
            f"{duplicate.head(10).to_dicts()}"
        )
    unknown_dates = sorted(set(mapping["Date"].to_list()) - set(sessions))
    if unknown_dates:
        raise ValueError(
            "target mapping dates are absent from session calendar: "
            f"{unknown_dates[:5]}"
        )

    history = excursions.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("side").cast(pl.String),
        pl.col("completed").fill_null(False).cast(pl.Boolean),
        pl.col("amplitude_bp").cast(pl.Float64),
    ).filter(
        pl.col("side").is_in(["positive", "negative"])
        & pl.col("amplitude_bp").is_finite()
        & (pl.col("amplitude_bp") >= 0)
    )

    rows: list[dict[str, object]] = []
    session_index = {value: index for index, value in enumerate(sessions)}
    for target_date in sorted(mapping["Date"].unique().to_list()):
        target_index = session_index[target_date]
        prior_dates = sessions[
            max(0, target_index - config.lookback_sessions):target_index
        ]
        targets = mapping.filter(pl.col("Date") == target_date)
        if not prior_dates:
            for target in targets.iter_rows(named=True):
                rows.extend(
                    _empty_target_rows(target, target_date, config)
                )
            continue

        window = history.filter(pl.col("Date").is_in(prior_dates))
        availability = (
            mapping.filter(pl.col("Date").is_in(prior_dates))
            .group_by("ValueCode")
            .agg(pl.col("Date").n_unique().alias("_history_sessions"))
        )
        available_sessions = dict(
            availability.select("ValueCode", "_history_sessions").iter_rows()
        )
        grouped = {
            str(value_code): group
            for (value_code,), group in window.group_by("ValueCode")
        }
        train_start = prior_dates[0]
        train_end = prior_dates[-1]
        for target in targets.iter_rows(named=True):
            product = str(target["ValueCode"])
            product_history = grouped.get(product, pl.DataFrame())
            rows.extend(
                _summarize_target(
                    target,
                    product_history,
                    target_date=target_date,
                    train_start=train_start,
                    train_end=train_end,
                    global_history_sessions=len(prior_dates),
                    product_history_sessions=int(
                        available_sessions.get(product, 0)
                    ),
                    config=config,
                )
            )

    result = pl.from_dicts(rows, infer_schema_length=None)
    if result.is_empty():
        return result
    result = result.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("train_start_date").cast(pl.String),
        pl.col("train_end_date").cast(pl.String),
        pl.when(pl.col("history_sessions_product") > 0)
        .then(
            pl.col("positive_completed")
            / pl.col("history_sessions_product")
        )
        .otherwise(None)
        .alias("positive_completed_per_session"),
        pl.when(pl.col("history_sessions_product") > 0)
        .then(
            pl.col("negative_completed")
            / pl.col("history_sessions_product")
        )
        .otherwise(None)
        .alias("negative_completed_per_session"),
    )
    if {
        "target_ref_future_tick_bp",
        "target_ref_spot_bid_tick_bp",
    }.issubset(result.columns):
        result = result.with_columns(
            (
                pl.col("upper_distance_bp")
                / pl.col("target_ref_future_tick_bp")
            ).alias("upper_distance_future_ticks"),
            (
                pl.col("lower_distance_bp")
                / pl.col("target_ref_future_tick_bp")
            ).alias("lower_distance_future_ticks"),
            (
                pl.col("upper_distance_bp")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("upper_distance_spot_ticks"),
            (
                pl.col("lower_distance_bp")
                / pl.col("target_ref_spot_bid_tick_bp")
            ).alias("lower_distance_spot_ticks"),
        )
    return result.sort(["Date", "ValueCode", "boundary_quantile"])


def _empty_target_rows(
    target: dict[str, object],
    target_date: str,
    config: RollingBoundaryConfig,
    *,
    train_start: str | None = None,
    train_end: str | None = None,
    global_history_sessions: int = 0,
    product_history_sessions: int = 0,
) -> list[dict[str, object]]:
    return [
        _snapshot_row(
            target,
            target_date=target_date,
            train_start=train_start,
            train_end=train_end,
            global_history_sessions=global_history_sessions,
            product_history_sessions=product_history_sessions,
            positive_dates=0,
            negative_dates=0,
            positive_started=0,
            negative_started=0,
            positive_completed=0,
            negative_completed=0,
            positive_censored=0,
            negative_censored=0,
            quantile=quantile,
            upper=None,
            lower=None,
            config=config,
        )
        for quantile in config.quantiles
    ]


def _summarize_target(
    target: dict[str, object],
    history: pl.DataFrame,
    *,
    target_date: str,
    train_start: str,
    train_end: str,
    global_history_sessions: int,
    product_history_sessions: int,
    config: RollingBoundaryConfig,
) -> list[dict[str, object]]:
    if history.is_empty():
        return _empty_target_rows(
            target,
            target_date,
            config,
            train_start=train_start,
            train_end=train_end,
            global_history_sessions=global_history_sessions,
            product_history_sessions=product_history_sessions,
        )
    side_stats: dict[str, dict[str, object]] = {}
    for side in ("positive", "negative"):
        sample = history.filter(pl.col("side") == side)
        completed = sample.filter(pl.col("completed"))
        side_stats[side] = {
            "dates": sample["Date"].n_unique(),
            "started": sample.height,
            "completed": completed.height,
            "censored": sample.height - completed.height,
            "amplitudes": completed["amplitude_bp"],
        }
    records: list[dict[str, object]] = []
    for quantile in config.quantiles:
        probability = quantile / 100.0
        positive = side_stats["positive"]
        negative = side_stats["negative"]
        upper = _series_quantile(positive["amplitudes"], probability)
        lower = _series_quantile(negative["amplitudes"], probability)
        records.append(
            _snapshot_row(
                target,
                target_date=target_date,
                train_start=train_start,
                train_end=train_end,
                global_history_sessions=global_history_sessions,
                product_history_sessions=product_history_sessions,
                positive_dates=int(positive["dates"]),
                negative_dates=int(negative["dates"]),
                positive_started=int(positive["started"]),
                negative_started=int(negative["started"]),
                positive_completed=int(positive["completed"]),
                negative_completed=int(negative["completed"]),
                positive_censored=int(positive["censored"]),
                negative_censored=int(negative["censored"]),
                quantile=quantile,
                upper=upper,
                lower=lower,
                config=config,
            )
        )
    return records


def _series_quantile(series: object, probability: float) -> float | None:
    if not isinstance(series, pl.Series) or series.is_empty():
        return None
    value = series.quantile(probability, interpolation="nearest")
    return float(value) if value is not None else None


def _snapshot_row(
    target: dict[str, object],
    *,
    target_date: str,
    train_start: str | None,
    train_end: str | None,
    global_history_sessions: int,
    product_history_sessions: int,
    positive_dates: int,
    negative_dates: int,
    positive_started: int,
    negative_started: int,
    positive_completed: int,
    negative_completed: int,
    positive_censored: int,
    negative_censored: int,
    quantile: int,
    upper: float | None,
    lower: float | None,
    config: RollingBoundaryConfig,
) -> dict[str, object]:
    valid = (
        global_history_sessions >= config.min_history_sessions
        and product_history_sessions >= config.min_history_sessions
        and positive_dates >= config.min_excursion_history_sessions_per_side
        and negative_dates >= config.min_excursion_history_sessions_per_side
        and positive_completed >= config.min_completed_excursions_per_side
        and negative_completed >= config.min_completed_excursions_per_side
        and upper is not None
        and lower is not None
        and upper > 0
        and lower > 0
    )
    # Point-in-time mapping can also carry target-day realised fields such as
    # trading_turnover.  Snapshot output is fail-closed to this pre-open-safe
    # allowlist rather than copying arbitrary mapping columns.
    safe_target_columns = (
        "Date",
        "ValueCode",
        "QuoteCode",
        "contract_size",
        "decimal_locator",
        "end_date",
        "fut_ref_price",
        "spot_ref_price",
        "day_trade_mark",
        "ins_type",
        "target_ref_future_ask_tick_bp",
        "target_ref_future_tick_bp",
        "target_ref_spot_bid_tick_bp",
        "price_ladder_version",
    )
    record = {
        column: target[column]
        for column in safe_target_columns
        if column in target
    }
    record.update(
        {
            "Date": target_date,
            "boundary_quantile": int(quantile),
            "boundary_role": (
                "rolling_latent_candidate"
                if int(quantile) in (50, 80)
                else "tail_diagnostic"
            ),
            "upper_distance_bp": upper,
            "lower_distance_bp": lower,
            "adaptive_parameter_valid": valid,
            "lookback_sessions": config.lookback_sessions,
            "minimum_history_sessions": config.min_history_sessions,
            "minimum_excursion_history_sessions_per_side": (
                config.min_excursion_history_sessions_per_side
            ),
            "history_sessions_global": global_history_sessions,
            "history_sessions_product": product_history_sessions,
            "analysis_supported_sessions": min(positive_dates, negative_dates),
            "positive_history_dates": positive_dates,
            "negative_history_dates": negative_dates,
            "positive_started": positive_started,
            "negative_started": negative_started,
            "positive_completed": positive_completed,
            "negative_completed": negative_completed,
            "positive_censored": positive_censored,
            "negative_censored": negative_censored,
            "train_start_date": train_start,
            "train_end_date": train_end,
            "source_asof_date": train_end,
            "parameter_version": config.parameter_version,
            "price_ladder_version": config.price_ladder_version,
            "future_one_dollar_tick_effective_date": (
                config.future_one_dollar_tick_effective_date
            ),
            "probability_layer": "rolling_latent_price_path_prior",
            "execution_safe_snapshot": True,
            "contains_target_day_outcome": False,
            "actionable_execution": False,
            "ev_ready": False,
        }
    )
    return record


def write_rolling_boundary_snapshots(
    snapshots: pl.DataFrame,
    output_dir: Path,
    config: RollingBoundaryConfig,
    *,
    source_provenance: Mapping[str, object],
) -> None:
    """Atomically publish snapshots, then expose them through one marker.

    The marker is the only publication boundary.  Readers reject a directory
    without it, so an interrupted rebuild cannot expose a mixture of old and
    new ladder artifacts.
    """

    config.validate()
    _validate_snapshot_ladder_rows(snapshots, config)
    source = _validate_source_provenance(source_provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / ROLLING_BOUNDARY_COMPLETE_MARKER
    complete_path.unlink(missing_ok=True)
    parquet_path = output_dir / ROLLING_BOUNDARY_ARTIFACT
    csv_path = output_dir / ROLLING_BOUNDARY_CSV_ARTIFACT
    config_path = output_dir / ROLLING_BOUNDARY_CONFIG_ARTIFACT
    _atomic_write_parquet(snapshots, parquet_path)
    _atomic_write_csv(snapshots, csv_path)
    config_contract = _rolling_config_payload(config)
    config_payload = {
        **config_contract,
        "window_semantics": "last N trading sessions strictly before target Date",
        "history_pooling": (
            "ValueCode across contract rolls; target QuoteCode stays exact"
        ),
        "snapshot_semantics": "pre-open safe latent prior; not execution or EV",
    }
    _atomic_write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        config_path,
    )
    code_provenance = _rolling_code_provenance()
    artifacts = {
        ROLLING_BOUNDARY_ARTIFACT: _frame_artifact_manifest(
            parquet_path,
            snapshots,
        ),
        ROLLING_BOUNDARY_CSV_ARTIFACT: _frame_artifact_manifest(
            csv_path,
            snapshots,
        ),
        ROLLING_BOUNDARY_CONFIG_ARTIFACT: {
            "bytes": config_path.stat().st_size,
            "sha256": _sha256_file(config_path),
        },
    }
    marker = {
        "complete": True,
        "schema_version": ROLLING_BOUNDARY_SCHEMA_VERSION,
        "publication_artifact": ROLLING_BOUNDARY_ARTIFACT,
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
        "config": config_contract,
        "config_sha256": _json_sha256(config_contract),
        "source_provenance": source,
        "builder_code": code_provenance,
        "builder_code_sha256": _json_sha256(code_provenance),
        "artifacts": artifacts,
    }
    marker["marker_payload_sha256"] = _json_sha256(marker)
    _atomic_write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        complete_path,
    )


def validate_rolling_boundary_artifact(
    boundary_path: Path,
    *,
    expected_daily_root: Path | None = None,
) -> dict[str, object]:
    """Validate marker, artifacts, current code, config, and row lineage."""

    path = Path(boundary_path)
    marker_path = path.parent / ROLLING_BOUNDARY_COMPLETE_MARKER
    config_path = path.parent / ROLLING_BOUNDARY_CONFIG_ARTIFACT
    for required in (path, marker_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"rolling boundary publication is incomplete: {required}"
            )
    expected_source_identity = None
    if expected_daily_root is not None:
        expected_source_identity = _partitioned_daily_source_provenance(
            Path(expected_daily_root)
        )["identity_sha256"]
    stats = tuple(
        value
        for required in (path, marker_path, config_path)
        for value in (required.stat().st_size, required.stat().st_mtime_ns)
    )
    payload_json = _validate_rolling_boundary_cached(
        str(path.resolve()),
        str(marker_path.resolve()),
        str(config_path.resolve()),
        *stats,
        expected_source_identity,
        _json_sha256(_rolling_code_provenance()),
    )
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):  # pragma: no cover - guarded in cache
        raise ValueError(f"rolling boundary marker is not an object: {marker_path}")
    return payload


def load_rolling_boundary_snapshots(
    boundary_path: Path,
    *,
    expected_daily_root: Path | None = None,
) -> pl.DataFrame:
    """Load only a completely published, current-version boundary artifact."""

    validate_rolling_boundary_artifact(
        boundary_path,
        expected_daily_root=expected_daily_root,
    )
    return pl.read_parquet(boundary_path)


@lru_cache(maxsize=16)
def _validate_rolling_boundary_cached(
    boundary_path_text: str,
    marker_path_text: str,
    config_path_text: str,
    _boundary_size: int,
    _boundary_mtime_ns: int,
    _marker_size: int,
    _marker_mtime_ns: int,
    _config_size: int,
    _config_mtime_ns: int,
    expected_source_identity: object,
    current_code_sha256: str,
) -> str:
    boundary_path = Path(boundary_path_text)
    marker_path = Path(marker_path_text)
    config_path = Path(config_path_text)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid rolling boundary marker: {marker_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"rolling boundary marker is not an object: {marker_path}")
    if payload.get("complete") is not True:
        raise ValueError(f"rolling boundary marker is incomplete: {marker_path}")
    if payload.get("schema_version") != ROLLING_BOUNDARY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported rolling boundary schema at {marker_path}: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("publication_artifact") != boundary_path.name:
        raise ValueError(f"rolling boundary marker artifact mismatch: {marker_path}")
    marker_digest = payload.get("marker_payload_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("marker_payload_sha256", None)
    if marker_digest != _json_sha256(digest_payload):
        raise ValueError(f"rolling boundary marker hash mismatch: {marker_path}")
    if payload.get("price_ladder_version") != PRICE_LADDER_VERSION or (
        payload.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        raise ValueError(f"rolling boundary marker price ladder mismatch: {marker_path}")

    config_contract = payload.get("config")
    if not isinstance(config_contract, dict):
        raise ValueError(f"rolling boundary marker lacks config: {marker_path}")
    if payload.get("config_sha256") != _json_sha256(config_contract):
        raise ValueError(f"rolling boundary config hash mismatch: {marker_path}")
    _config_from_payload(config_contract).validate()
    try:
        persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid rolling boundary config: {config_path}") from error
    if not isinstance(persisted_config, dict) or any(
        persisted_config.get(key) != value
        for key, value in config_contract.items()
    ):
        raise ValueError(f"rolling boundary config payload mismatch: {config_path}")

    source = _validate_source_provenance(payload.get("source_provenance"))
    if expected_source_identity is not None and (
        source.get("kind") != "partitioned_daily_facts"
        or source.get("identity_sha256") != expected_source_identity
    ):
        raise ValueError(
            "rolling boundary source generation does not match the current "
            "daily fact store"
        )
    current_code = _rolling_code_provenance()
    if payload.get("builder_code") != current_code or payload.get(
        "builder_code_sha256"
    ) != current_code_sha256:
        raise ValueError(f"rolling boundary builder code mismatch: {marker_path}")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"rolling boundary marker lacks artifacts: {marker_path}")
    expected_paths = {
        ROLLING_BOUNDARY_ARTIFACT: boundary_path,
        ROLLING_BOUNDARY_CSV_ARTIFACT: boundary_path.parent
        / ROLLING_BOUNDARY_CSV_ARTIFACT,
        ROLLING_BOUNDARY_CONFIG_ARTIFACT: config_path,
    }
    for name, artifact_path in expected_paths.items():
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"rolling boundary marker omits {name}: {marker_path}")
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"rolling boundary marker missing artifact: {artifact_path}"
            )
        if entry.get("bytes") != artifact_path.stat().st_size or entry.get(
            "sha256"
        ) != _sha256_file(artifact_path):
            raise ValueError(f"rolling boundary artifact hash mismatch: {artifact_path}")

    parquet_entry = artifacts[ROLLING_BOUNDARY_ARTIFACT]
    parquet_rows = int(
        pl.scan_parquet(boundary_path).select(pl.len()).collect().item()
    )
    parquet_columns = pl.read_parquet_schema(boundary_path)
    if (
        parquet_entry.get("rows") != parquet_rows
        or parquet_entry.get("columns") != len(parquet_columns)
        or parquet_entry.get("column_names") != list(parquet_columns.names())
    ):
        raise ValueError(f"rolling boundary artifact shape mismatch: {boundary_path}")
    csv_path = expected_paths[ROLLING_BOUNDARY_CSV_ARTIFACT]
    csv_entry = artifacts[ROLLING_BOUNDARY_CSV_ARTIFACT]
    csv_scan = pl.scan_csv(csv_path)
    csv_schema = csv_scan.collect_schema()
    csv_rows = int(csv_scan.select(pl.len()).collect().item())
    if (
        csv_entry.get("rows") != csv_rows
        or csv_entry.get("columns") != len(csv_schema)
        or csv_entry.get("column_names") != list(csv_schema.names())
        or csv_rows != parquet_rows
    ):
        raise ValueError(f"rolling boundary CSV shape mismatch: {csv_path}")

    rows = pl.read_parquet(
        boundary_path,
        columns=[
            "price_ladder_version",
            "future_one_dollar_tick_effective_date",
            "parameter_version",
            "lookback_sessions",
            "boundary_quantile",
        ],
    )
    _validate_snapshot_ladder_rows(rows, _config_from_payload(config_contract))
    return json.dumps(payload, sort_keys=True)


def _rolling_config_payload(config: RollingBoundaryConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["quantiles"] = list(config.quantiles)
    return payload


def _config_from_payload(payload: Mapping[str, object]) -> RollingBoundaryConfig:
    fields = set(RollingBoundaryConfig.__dataclass_fields__)
    if set(payload) != fields:
        raise ValueError("rolling boundary config fields do not match current schema")
    values = dict(payload)
    quantiles = values.get("quantiles")
    if not isinstance(quantiles, list):
        raise ValueError("rolling boundary config quantiles must be a list")
    values["quantiles"] = tuple(int(value) for value in quantiles)
    try:
        return RollingBoundaryConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid rolling boundary config payload") from error


def _validate_snapshot_ladder_rows(
    snapshots: pl.DataFrame,
    config: RollingBoundaryConfig,
) -> None:
    required = {
        "price_ladder_version",
        "future_one_dollar_tick_effective_date",
        "parameter_version",
        "lookback_sessions",
        "boundary_quantile",
    }
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"rolling boundary rows missing lineage columns: {missing}")
    if snapshots.is_empty():
        raise ValueError("rolling boundary snapshots must not be empty")
    invalid = snapshots.filter(
        (pl.col("price_ladder_version") != PRICE_LADDER_VERSION)
        | pl.col("price_ladder_version").is_null()
        | (
            pl.col("future_one_dollar_tick_effective_date")
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        )
        | pl.col("future_one_dollar_tick_effective_date").is_null()
        | (pl.col("parameter_version") != config.parameter_version)
        | pl.col("parameter_version").is_null()
        | (pl.col("lookback_sessions") != config.lookback_sessions)
        | ~pl.col("boundary_quantile").cast(pl.Int64).is_in(
            list(config.quantiles)
        )
    )
    if invalid.height:
        raise ValueError("rolling boundary row/config lineage mismatch")


def _partitioned_daily_source_provenance(root: Path) -> dict[str, object]:
    root = Path(root)
    markers = sorted(root.glob("Date=*/complete.json"))
    if not markers:
        raise FileNotFoundError(f"no daily completion markers below {root}")
    signature = tuple(
        (
            str(marker.relative_to(root)),
            marker.stat().st_size,
            marker.stat().st_mtime_ns,
        )
        for marker in markers
    )
    return json.loads(
        _partitioned_daily_source_provenance_cached(
            str(root.resolve()),
            signature,
        )
    )


@lru_cache(maxsize=16)
def _partitioned_daily_source_provenance_cached(
    root_text: str,
    signature: tuple[tuple[str, int, int], ...],
) -> str:
    root = Path(root_text)
    entries = [
        {
            "path": relative,
            "bytes": size,
            "sha256": _sha256_file(root / relative),
        }
        for relative, size, _mtime_ns in signature
    ]
    payload = _make_source_provenance(
        "partitioned_daily_facts",
        {
            "completion_marker_count": len(entries),
            "completion_markers_sha256": _json_sha256(entries),
            "first_date": Path(signature[0][0]).parent.name.removeprefix("Date="),
            "last_date": Path(signature[-1][0]).parent.name.removeprefix("Date="),
        },
    )
    return json.dumps(payload, sort_keys=True)


def _file_source_provenance(paths: Mapping[str, Path]) -> dict[str, object]:
    entries: dict[str, object] = {}
    for name, source_path in sorted(paths.items()):
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return _make_source_provenance("explicit_input_files", {"files": entries})


def _make_source_provenance(
    kind: str,
    identity_fields: Mapping[str, object],
) -> dict[str, object]:
    payload = {"bound": True, "kind": kind, **dict(identity_fields)}
    payload["identity_sha256"] = _json_sha256(payload)
    return payload


def _validate_source_provenance(source: object) -> dict[str, object]:
    if not isinstance(source, Mapping):
        raise ValueError("rolling boundary source provenance must be an object")
    payload = dict(source)
    identity = payload.pop("identity_sha256", None)
    if (
        payload.get("bound") is not True
        or not isinstance(payload.get("kind"), str)
        or not payload["kind"]
        or identity != _json_sha256(payload)
    ):
        raise ValueError("rolling boundary source provenance is invalid")
    return {**payload, "identity_sha256": identity}


def _rolling_code_provenance() -> list[dict[str, object]]:
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__),
        root / "quote_width" / "table.py",
        root / "fair_mid" / "quote_churn.py",
        root / "quote_fill" / "targets.py",
    )
    return [
        {
            "module": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _frame_artifact_manifest(
    path: Path,
    frame: pl.DataFrame,
) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "rows": frame.height,
        "columns": frame.width,
        "column_names": frame.columns,
    }


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.write_csv(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def run_partitioned_rolling_boundary_study(
    *,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    output_dir: Path | None = None,
    config: RollingBoundaryConfig = RollingBoundaryConfig(),
) -> pl.DataFrame:
    """Build rolling snapshots directly from the resumable daily fact store."""

    mapping = load_partitioned_mapping(daily_root)
    sessions = sorted(mapping["Date"].cast(pl.String).unique().to_list())
    snapshots = build_rolling_boundary_snapshots(
        load_partitioned_excursions(daily_root),
        mapping,
        sessions,
        config,
    )
    destination = output_dir or daily_root.parent / "rolling_boundaries"
    write_rolling_boundary_snapshots(
        snapshots,
        destination,
        config,
        source_provenance=_partitioned_daily_source_provenance(daily_root),
    )
    return snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trailing-session product-adaptive basis boundaries"
    )
    parser.add_argument("--daily-root", type=Path)
    parser.add_argument("--excursions", type=Path)
    parser.add_argument("--target-mapping", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width" / "rolling",
    )
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=40)
    parser.add_argument("--min-excursions-per-side", type=int, default=100)
    return parser.parse_args()


def _read_frame(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(
        path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
        },
    )


def main() -> None:
    args = parse_args()
    config = RollingBoundaryConfig(
        lookback_sessions=args.lookback_sessions,
        min_history_sessions=args.min_history_sessions,
        min_completed_excursions_per_side=args.min_excursions_per_side,
    )
    if args.daily_root is not None:
        snapshots = run_partitioned_rolling_boundary_study(
            daily_root=args.daily_root,
            output_dir=args.output_dir,
            config=config,
        )
    else:
        if (
            args.excursions is None
            or args.target_mapping is None
            or args.sessions is None
        ):
            raise ValueError(
                "provide --daily-root or all of --excursions, "
                "--target-mapping, --sessions"
            )
        sessions = [
            line.strip()
            for line in args.sessions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshots = build_rolling_boundary_snapshots(
            _read_frame(args.excursions),
            _read_frame(args.target_mapping),
            sessions,
            config,
        )
        write_rolling_boundary_snapshots(
            snapshots,
            args.output_dir,
            config,
            source_provenance=_file_source_provenance(
                {
                    "excursions": args.excursions,
                    "target_mapping": args.target_mapping,
                    "sessions": args.sessions,
                }
            ),
        )
    print(snapshots)


if __name__ == "__main__":
    main()
