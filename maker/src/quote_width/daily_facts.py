"""Partitioned all-market causal fair and excursion facts.

The original pilot concatenated every requested day in memory.  This runner
processes one market day at a time and writes resumable partitions, which is
the prerequisite for a 60-session rolling walk-forward study.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import polars as pl

from ..common.landmarks import build_landmarks
from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT, futures_raw_path
from ..fair_mid.anchors import prepare_causal_fair_panel
from .table import extract_zero_crossing_excursions


DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
MIN_FUTURES_BYTES = 1_000_000
DAILY_FACT_SCHEMA_VERSION = "daily_latent_facts_v2"
MIGRATED_DAILY_FACT_SCHEMA_VERSION = "daily_latent_facts_v2_migrated_nonatomic"
DAILY_BUILD_CONFIG = {
    "interval": "1s",
    "anchor": "causal_ewma_120s_and_controls",
    "analysis_start_seconds": 300,
    "spot_marks": ["X", "Y"],
    "ref_gate": "strict_-9pct_+8pct",
}

EXPECTED_ARTIFACT_SCHEMAS: dict[str, pl.Schema] = {
    "causal_fair.parquet": pl.Schema(
        {
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "timestamp": pl.Datetime("ns"),
            "local_timestamp": pl.Datetime("us"),
            "seconds_from_open": pl.Int32,
            "spot_recv_time": pl.Datetime("ns"),
            "spot_sequence": pl.UInt64,
            "fut_recv_time": pl.Datetime("ns"),
            "fut_sequence": pl.UInt64,
            "spot_ref_price": pl.Float64,
            "fut_ref_price": pl.Float64,
            "contract_size": pl.Float64,
            "end_date": pl.Date,
            "spot_bid": pl.Float64,
            "spot_ask": pl.Float64,
            "spot_bid_lots": pl.Int64,
            "spot_ask_lots": pl.Int64,
            "fut_bid": pl.Float64,
            "fut_ask": pl.Float64,
            "fut_exec_bid": pl.Float64,
            "fut_exec_ask": pl.Float64,
            "fut_bid_lots": pl.Int64,
            "fut_ask_lots": pl.Int64,
            "fut_exec_bid_lots": pl.Int64,
            "fut_exec_ask_lots": pl.Int64,
            "spot_trial_match": pl.Int16,
            "fut_trial_match": pl.Int16,
            "spot_age_ms": pl.Float64,
            "fut_age_ms": pl.Float64,
            "leg_skew_ms": pl.Float64,
            "spot_formal": pl.Boolean,
            "fut_formal": pl.Boolean,
            "spot_book_ok": pl.Boolean,
            "fut_book_ok": pl.Boolean,
            "fut_exec_book_ok": pl.Boolean,
            "spot_ref_ok": pl.Boolean,
            "fut_ref_ok": pl.Boolean,
            "eligible_base": pl.Boolean,
            "eligible_100ms": pl.Boolean,
            "eligible_250ms": pl.Boolean,
            "eligible_500ms": pl.Boolean,
            "eligible_1000ms": pl.Boolean,
            "eligible_5000ms": pl.Boolean,
            "analysis_eligible": pl.Boolean,
            "basis_mid_bp": pl.Float64,
            "basis_sell_taker_bp": pl.Float64,
            "basis_buy_taker_bp": pl.Float64,
            "basis_eval_bp": pl.Float64,
            "anchor_ewma_30s_bp": pl.Float64,
            "anchor_ewma_120s_bp": pl.Float64,
            "anchor_ewma_300s_bp": pl.Float64,
            "anchor_rolling_median_300s_bp": pl.Float64,
        }
    ),
    "excursions.parquet": pl.Schema(
        {
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
    ),
    "mapping.parquet": pl.Schema(
        {
            "QuoteCode": pl.String,
            "ValueCode": pl.String,
            "contract_size": pl.Float64,
            "decimal_locator": pl.Int16,
            "end_date": pl.Date,
            "fut_ref_price": pl.Float64,
            "spot_ref_price": pl.Float64,
            "day_trade_mark": pl.String,
            "trading_turnover": pl.Float64,
            "ins_type": pl.String,
            "Date": pl.String,
        }
    ),
    "audit.parquet": pl.Schema(
        {
            "Date": pl.String,
            "symbols": pl.Int64,
            "spot_event_rows": pl.Int64,
            "future_event_rows": pl.Int64,
            "landmark_rows": pl.Int64,
            "spot_clock_offset_p50_ms": pl.Float64,
            "spot_clock_offset_p99_ms": pl.Float64,
            "future_clock_offset_p50_ms": pl.Float64,
            "future_clock_offset_p99_ms": pl.Float64,
            "eligible_base_rows": pl.Int64,
            "eligible_100ms_rows": pl.Int64,
            "eligible_250ms_rows": pl.Int64,
            "eligible_500ms_rows": pl.Int64,
            "eligible_1000ms_rows": pl.Int64,
            "eligible_5000ms_rows": pl.Int64,
            "spot_trial_block_rows": pl.Int64,
            "future_trial_block_rows": pl.Int64,
            "spot_ref_block_rows": pl.Int64,
            "future_ref_block_rows": pl.Int64,
        }
    ),
}


@dataclass(frozen=True)
class DailyLatentFactResult:
    date: str
    products: int
    causal_rows: int
    excursion_rows: int
    elapsed_seconds: float
    status: str


def discover_common_sessions(
    *,
    data_root: Path = HFT_DATA_ROOT,
    year: str = "2026",
) -> list[str]:
    """Find days with spot tick, spot feature and non-placeholder futures raw."""

    dates: list[str] = []
    pattern = f"{year}*_StockTick.parquet"
    for spot_path in sorted((data_root / "tickData").glob(pattern)):
        date = spot_path.name[:8]
        feature = data_root / "tickFeature" / f"{date}_tickFeature.parquet"
        future = futures_raw_path(date)
        if (
            feature.exists()
            and future.exists()
            and future.stat().st_size >= MIN_FUTURES_BYTES
        ):
            dates.append(date)
    return dates


def _causal_columns(panel: pl.DataFrame) -> list[str]:
    preferred = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "local_timestamp",
        "seconds_from_open",
        "spot_recv_time",
        "spot_sequence",
        "fut_recv_time",
        "fut_sequence",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
        "end_date",
        "spot_bid",
        "spot_ask",
        "spot_bid_lots",
        "spot_ask_lots",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "fut_exec_ask",
        "fut_bid_lots",
        "fut_ask_lots",
        "fut_exec_bid_lots",
        "fut_exec_ask_lots",
        "spot_trial_match",
        "fut_trial_match",
        "spot_age_ms",
        "fut_age_ms",
        "leg_skew_ms",
        "spot_formal",
        "fut_formal",
        "spot_book_ok",
        "fut_book_ok",
        "fut_exec_book_ok",
        "spot_ref_ok",
        "fut_ref_ok",
        "eligible_base",
        "eligible_100ms",
        "eligible_250ms",
        "eligible_500ms",
        "eligible_1000ms",
        "eligible_5000ms",
        "analysis_eligible",
        "basis_mid_bp",
        "basis_sell_taker_bp",
        "basis_buy_taker_bp",
        "basis_eval_bp",
        "anchor_ewma_30s_bp",
        "anchor_ewma_120s_bp",
        "anchor_ewma_300s_bp",
        "anchor_rolling_median_300s_bp",
    ]
    return [column for column in preferred if column in panel.columns]


def build_daily_latent_facts(
    date: str,
    *,
    output_root: Path = DEFAULT_DAILY_ROOT,
    resume: bool = True,
) -> DailyLatentFactResult:
    """Build and persist one all-product market day."""

    date = str(date)
    partition = output_root / f"Date={date}"
    complete_path = partition / "complete.json"
    if resume and complete_path.exists():
        payload = validate_completion_marker(complete_path)
        return DailyLatentFactResult(
            date=date,
            products=int(payload["products"]),
            causal_rows=int(payload["causal_rows"]),
            excursion_rows=int(payload["excursion_rows"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            status="skipped_complete",
        )

    # The completion marker is the publication boundary.  Removing an old
    # marker before a forced rebuild prevents readers from consuming a mix of
    # old and newly written artifacts if the process stops partway through.
    partition.mkdir(parents=True, exist_ok=True)
    if complete_path.exists():
        complete_path.unlink()

    start = perf_counter()
    result = build_landmarks(
        date,
        value_codes=None,
        interval="1s",
        cache_dir=output_root / "metadata",
    )
    causal = prepare_causal_fair_panel(result.landmarks)
    excursions = extract_zero_crossing_excursions(causal)
    compact = causal.select(_causal_columns(causal))
    artifact_frames = {
        "causal_fair.parquet": compact,
        "excursions.parquet": excursions,
        "mapping.parquet": result.mapping.with_columns(pl.lit(date).alias("Date")),
        "audit.parquet": result.audit,
    }
    for name, frame in artifact_frames.items():
        temporary = partition / f".{name}.tmp"
        frame.write_parquet(
            temporary,
            compression="zstd",
            statistics=True,
        )
        temporary.replace(partition / name)
    elapsed = perf_counter() - start
    payload = {
        "date": date,
        "products": result.mapping.height,
        "causal_rows": compact.height,
        "excursion_rows": excursions.height,
        "elapsed_seconds": elapsed,
        "schema_version": DAILY_FACT_SCHEMA_VERSION,
        "builder_code_sha256": _daily_builder_code_sha256(),
        "build_config_sha256": _json_sha256(DAILY_BUILD_CONFIG),
        "source_identity_sha256": _source_identity_sha256(date),
        "artifacts": {
            "causal_fair.parquet": _artifact_manifest(
                partition / "causal_fair.parquet",
                execution_safe_causal_panel=True,
                contains_target_day_outcome=False,
            ),
            "excursions.parquet": _artifact_manifest(
                partition / "excursions.parquet",
                execution_safe_causal_panel=False,
                contains_target_day_outcome=True,
                label_available_after_close=True,
            ),
            "mapping.parquet": _artifact_manifest(
                partition / "mapping.parquet",
                contains_target_day_trading_turnover=True,
                consumer_must_use_preopen_allowlist=True,
            ),
            "audit.parquet": _artifact_manifest(
                partition / "audit.parquet",
                contains_target_day_outcome=False,
            ),
        },
    }
    temporary_complete = partition / ".complete.json.tmp"
    temporary_complete.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_complete.replace(complete_path)
    return DailyLatentFactResult(
        date=date,
        products=result.mapping.height,
        causal_rows=compact.height,
        excursion_rows=excursions.height,
        elapsed_seconds=elapsed,
        status="built",
    )


def run_daily_latent_fact_batch(
    dates: Iterable[str],
    *,
    output_root: Path = DEFAULT_DAILY_ROOT,
    resume: bool = True,
) -> pl.DataFrame:
    rows = [
        build_daily_latent_facts(
            str(date),
            output_root=output_root,
            resume=resume,
        ).__dict__
        for date in dates
    ]
    manifest = pl.from_dicts(rows, infer_schema_length=None)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(output_root / "batch_manifest.csv")
    completed_sessions = []
    for marker in sorted(output_root.glob("Date=*/complete.json")):
        validate_completion_marker(marker)
        completed_sessions.append(marker.parent.name.removeprefix("Date="))
    (output_root.parent / "sessions.txt").write_text(
        "\n".join(completed_sessions) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parquet_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _schema_fingerprint(path: Path) -> str:
    schema = pl.read_parquet_schema(path)
    serialized = "|".join(f"{name}:{dtype}" for name, dtype in schema.items())
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _daily_builder_code_sha256() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).parents[1]
    for path in (
        Path(__file__),
        source_root / "common" / "contracts.py",
        source_root / "common" / "landmarks.py",
        source_root / "fair_mid" / "anchors.py",
        source_root / "quote_width" / "table.py",
    ):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_identity_sha256(date: str) -> str:
    sources = [
        HFT_DATA_ROOT / "tickData" / f"{date}_StockTick.parquet",
        HFT_DATA_ROOT / "tickFeature" / f"{date}_tickFeature.parquet",
        futures_raw_path(date),
    ]
    identity = []
    for path in sources:
        stat = path.stat()
        identity.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return _json_sha256(identity)


def _artifact_manifest(path: Path, **semantics: object) -> dict[str, object]:
    return {
        "rows": _parquet_rows(path),
        "bytes": path.stat().st_size,
        "schema_sha256": _schema_fingerprint(path),
        **semantics,
    }


def validate_completion_marker(
    marker: Path,
    *,
    migrate_legacy: bool = False,
) -> dict[str, object]:
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected_date = marker.parent.name.removeprefix("Date=")
    if str(payload.get("date")) != expected_date:
        raise ValueError(f"completion marker date mismatch: {marker}")
    required = {
        "causal_fair.parquet": int(payload["causal_rows"]),
        "excursions.parquet": int(payload["excursion_rows"]),
        "mapping.parquet": int(payload["products"]),
        "audit.parquet": None,
    }
    for name, expected_rows in required.items():
        path = marker.parent / name
        if not path.exists():
            raise FileNotFoundError(f"completion marker missing artifact: {path}")
        if expected_rows is not None and _parquet_rows(path) != expected_rows:
            raise ValueError(f"completion marker row-count mismatch: {path}")
    version = payload.get("schema_version")
    if version is None and migrate_legacy:
        manifests = {
            "causal_fair.parquet": _artifact_manifest(
                marker.parent / "causal_fair.parquet",
                execution_safe_causal_panel=True,
                contains_target_day_outcome=False,
            ),
            "excursions.parquet": _artifact_manifest(
                marker.parent / "excursions.parquet",
                execution_safe_causal_panel=False,
                contains_target_day_outcome=True,
                label_available_after_close=True,
            ),
            "mapping.parquet": _artifact_manifest(
                marker.parent / "mapping.parquet",
                contains_target_day_trading_turnover=True,
                consumer_must_use_preopen_allowlist=True,
            ),
            "audit.parquet": _artifact_manifest(
                marker.parent / "audit.parquet",
                contains_target_day_outcome=False,
            ),
        }
        payload.update(
            {
                "schema_version": MIGRATED_DAILY_FACT_SCHEMA_VERSION,
                "migrated_legacy_marker": True,
                "legacy_atomic_publish_verified": False,
                "legacy_writer_provenance_verified": False,
                "artifacts": manifests,
            }
        )
        temporary = marker.parent / ".complete.json.migrate.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(marker)
    elif version not in {
        DAILY_FACT_SCHEMA_VERSION,
        MIGRATED_DAILY_FACT_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"unsupported daily fact schema {version!r} at {marker}"
        )
    if payload.get("schema_version") in {
        DAILY_FACT_SCHEMA_VERSION,
        MIGRATED_DAILY_FACT_SCHEMA_VERSION,
    }:
        manifests = payload.get("artifacts")
        if not isinstance(manifests, dict):
            raise ValueError(f"completion marker missing artifact manifest: {marker}")
        if payload.get("schema_version") == DAILY_FACT_SCHEMA_VERSION:
            if payload.get("builder_code_sha256") != _daily_builder_code_sha256():
                raise ValueError(f"daily fact builder code mismatch: {marker}")
            if payload.get("build_config_sha256") != _json_sha256(DAILY_BUILD_CONFIG):
                raise ValueError(f"daily fact build config mismatch: {marker}")
            if payload.get("source_identity_sha256") != _source_identity_sha256(
                expected_date
            ):
                raise ValueError(f"daily fact source identity mismatch: {marker}")
        expected_semantics = {
            "causal_fair.parquet": {
                "execution_safe_causal_panel": True,
                "contains_target_day_outcome": False,
            },
            "excursions.parquet": {
                "execution_safe_causal_panel": False,
                "contains_target_day_outcome": True,
                "label_available_after_close": True,
            },
            "mapping.parquet": {
                "contains_target_day_trading_turnover": True,
                "consumer_must_use_preopen_allowlist": True,
            },
            "audit.parquet": {"contains_target_day_outcome": False},
        }
        for name, expected_rows in required.items():
            entry = manifests.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"completion marker omits {name}: {marker}")
            path = marker.parent / name
            actual_schema = pl.read_parquet_schema(path)
            if actual_schema != EXPECTED_ARTIFACT_SCHEMAS[name]:
                raise ValueError(f"artifact schema contract mismatch: {path}")
            if entry.get("rows") != _parquet_rows(path):
                raise ValueError(f"artifact manifest row mismatch: {path}")
            if entry.get("bytes") != path.stat().st_size:
                raise ValueError(f"artifact manifest size mismatch: {path}")
            if entry.get("schema_sha256") != _schema_fingerprint(path):
                raise ValueError(f"artifact manifest schema mismatch: {path}")
            if any(
                entry.get(key) != value
                for key, value in expected_semantics[name].items()
            ):
                raise ValueError(f"artifact manifest semantics mismatch: {path}")
            if "Date" in actual_schema:
                date_bounds = pl.scan_parquet(path).select(
                    pl.col("Date").cast(pl.String).min().alias("min"),
                    pl.col("Date").cast(pl.String).max().alias("max"),
                ).collect().row(0)
                if date_bounds != (expected_date, expected_date):
                    raise ValueError(f"artifact Date mismatch: {path}")
    return payload


def migrate_legacy_completion_markers(root: Path = DEFAULT_DAILY_ROOT) -> int:
    migrated = 0
    for marker in sorted(root.glob("Date=*/complete.json")):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("schema_version") is None:
            validate_completion_marker(marker, migrate_legacy=True)
            migrated += 1
        else:
            validate_completion_marker(marker)
    return migrated


def completed_artifact_paths(root: Path, artifact: str) -> list[Path]:
    markers = sorted(root.glob("Date=*/complete.json"))
    for marker in markers:
        payload = validate_completion_marker(marker)
        if artifact not in payload["artifacts"]:
            raise ValueError(f"artifact is not published by marker: {artifact}")
    paths = [marker.parent / artifact for marker in markers]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"completed partitions missing {artifact}: {missing[:5]}")
    return paths


def load_partitioned_excursions(root: Path = DEFAULT_DAILY_ROOT) -> pl.DataFrame:
    paths = completed_artifact_paths(root, "excursions.parquet")
    if not paths:
        raise FileNotFoundError(f"no excursion partitions below {root}")
    return pl.scan_parquet(paths).collect(engine="streaming")


def load_partitioned_mapping(root: Path = DEFAULT_DAILY_ROOT) -> pl.DataFrame:
    paths = completed_artifact_paths(root, "mapping.parquet")
    if not paths:
        raise FileNotFoundError(f"no mapping partitions below {root}")
    return pl.scan_parquet(paths).collect(engine="streaming")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable all-market daily causal fair/excursion facts"
    )
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--migrate-legacy-markers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.migrate_legacy_markers:
        migrated = migrate_legacy_completion_markers(args.output_root)
        print(f"migrated {migrated} legacy completion markers")
        return
    dates = args.dates or discover_common_sessions(year=args.year)
    manifest = run_daily_latent_fact_batch(
        dates,
        output_root=args.output_root,
        resume=not args.no_resume,
    )
    print(manifest)


if __name__ == "__main__":
    main()
