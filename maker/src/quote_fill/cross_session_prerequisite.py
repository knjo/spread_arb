"""Versioned candidate-session inputs for the formal cross-session replay.

The 60-entry-session cohort is immutable, but positions opened near the end of
that cohort still require later market sessions through exact-contract expiry.
This builder extends only the *candidate* calendar, publishes a standalone
contract-metadata root, and binds every source needed for the added sessions.
It never mutates the entry cohort, the existing daily metadata cache, or a
completed prerequisite root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import polars as pl

from ..common.contracts import (
    _load_futures_basic_from_existing_loader,
    select_near_standard_contracts,
)
from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT, parse_date
from .exit_maker_cross_session_runner import (
    DEFAULT_FUTURES_RAW_ROOT,
    _canonical_sha256,
    _established_entry_count,
    _file_sha256,
    _source_quote_code,
    _verify_source_marker,
)


PREREQUISITE_SCHEMA_VERSION = "cross_session_prerequisites_v1"
DEFAULT_BASE_SESSIONS = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
DEFAULT_CONTRACT_CALENDAR = (
    MAKER_ROOT / "data" / "walkforward" / "exact_contract_calendar_v1.parquet"
)
DEFAULT_ENTRY_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "execution_narrow_60d"
)
DEFAULT_PRODUCT_DAYS = DEFAULT_ENTRY_ROOT / "execution_partition_manifest.parquet"
DEFAULT_EXISTING_METADATA_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "daily" / "metadata"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "cross_session_prerequisites_v1_20260819"
)
DEFAULT_EXTENSION_SESSIONS = (
    "20260814",
    "20260817",
    "20260818",
    "20260819",
)

BasicLoader = Callable[[str], pl.DataFrame]


@dataclass(frozen=True)
class CrossSessionPrerequisiteResult:
    output_root: Path
    session_count: int
    metadata_session_count: int
    candidate_requirement_count: int
    extension_requirement_count: int
    resumed: bool


def build_cross_session_prerequisites(
    *,
    base_sessions_path: Path = DEFAULT_BASE_SESSIONS,
    extension_sessions: Sequence[str] = DEFAULT_EXTENSION_SESSIONS,
    contract_calendar_path: Path = DEFAULT_CONTRACT_CALENDAR,
    entry_execution_root: Path = DEFAULT_ENTRY_ROOT,
    product_days_path: Path = DEFAULT_PRODUCT_DAYS,
    existing_metadata_root: Path = DEFAULT_EXISTING_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    basic_loader: BasicLoader = _load_futures_basic_from_existing_loader,
    resume: bool = True,
) -> CrossSessionPrerequisiteResult:
    """Publish one immutable candidate calendar plus PIT contract metadata."""

    base_path = Path(base_sessions_path).resolve()
    calendar_path = Path(contract_calendar_path).resolve()
    entry_root = Path(entry_execution_root).resolve()
    product_path = Path(product_days_path).resolve()
    old_metadata_root = Path(existing_metadata_root).resolve()
    destination = Path(output_root).resolve()
    data_root = Path(data_root).resolve()
    futures_raw_root = Path(futures_raw_root).resolve()
    for path in (base_path, calendar_path, product_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not old_metadata_root.is_dir():
        raise FileNotFoundError(old_metadata_root)
    if destination == old_metadata_root or destination.is_relative_to(
        old_metadata_root
    ):
        raise ValueError("output root must be disjoint from existing metadata")
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        payload = verify_cross_session_prerequisites(destination)
        return _result_from_payload(destination, payload, resumed=True)

    base_sessions = _load_sessions(base_path)
    extension = tuple(_date(value, "extension session") for value in extension_sessions)
    if (
        not extension
        or extension != tuple(sorted(extension))
        or len(extension) != len(set(extension))
        or extension[0] <= base_sessions[-1]
    ):
        raise ValueError(
            "extension sessions must be unique, ascending and after the base horizon"
        )
    sessions = base_sessions + extension
    calendar = pl.read_parquet(calendar_path).select(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("expiry_session").cast(pl.String),
        pl.col("calendar_version").cast(pl.String),
    )
    _validate_calendar(calendar, sessions)
    product_days = pl.read_parquet(product_path).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    requirements, entry_lineage = _candidate_requirements(
        product_days,
        sessions=sessions,
        calendar=calendar,
        entry_root=entry_root,
    )
    universe_symbols = sorted(
        str(value) for value in product_days["ValueCode"].unique().to_list()
    )
    last_base_metadata = _validate_contract_frame(
        pl.read_parquet(
            old_metadata_root / f"{base_sessions[-1]}_contracts.parquet"
        ),
        base_sessions[-1],
    )
    extension_validation_pairs = last_base_metadata.filter(
        pl.col("ValueCode").is_in(universe_symbols)
    ).select("ValueCode", "QuoteCode", "end_date")
    if extension_validation_pairs.height != len(universe_symbols):
        raise ValueError(
            "last base-session metadata does not cover the complete entry universe"
        )

    extension_contracts: dict[str, pl.DataFrame] = {}
    extension_basic_hashes: dict[str, str] = {}
    for session in extension:
        basic = basic_loader(session)
        extension_basic_hashes[session] = _frame_sha256(basic)
        contracts = select_near_standard_contracts(
            basic, parse_date(session).date()
        )
        extension_contracts[session] = _validate_contract_frame(
            contracts, session
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        metadata_root = stage / "metadata"
        metadata_root.mkdir()
        metadata_rows: list[dict[str, object]] = []
        for session in sessions:
            target = metadata_root / f"{session}_contracts.parquet"
            if session in extension_contracts:
                contracts = extension_contracts[session]
                contracts.write_parquet(target, compression="zstd", statistics=True)
                source_kind = "ProductInfo.taifex_pib_view_point_in_time"
                source_path = f"mysql://ProductInfo.taifex_pib_view?date={session}"
                source_sha = extension_basic_hashes[session]
            else:
                source = old_metadata_root / f"{session}_contracts.parquet"
                if not source.is_file():
                    raise FileNotFoundError(source)
                contracts = _validate_contract_frame(pl.read_parquet(source), session)
                shutil.copy2(source, target)
                source_kind = "existing_daily_metadata_verified_copy"
                source_path = str(source)
                source_sha = _file_sha256(source)
            metadata_rows.append(
                {
                    "Date": session,
                    "artifact": str(target.relative_to(stage)),
                    "artifact_rows": contracts.height,
                    "artifact_bytes": target.stat().st_size,
                    "artifact_sha256": _file_sha256(target),
                    "source_kind": source_kind,
                    "source_path": source_path,
                    "source_content_sha256": source_sha,
                }
            )

        extension_audit, extension_sources = _audit_extension_requirements(
            requirements,
            extension=extension,
            metadata_root=metadata_root,
            data_root=data_root,
            futures_raw_root=futures_raw_root,
            futures_basic_hashes=extension_basic_hashes,
            validation_pairs=extension_validation_pairs,
        )
        if extension_audit.filter(~pl.col("mapping_valid")).height:
            raise ValueError("extension exact-contract mapping audit failed")

        sessions_path = stage / "candidate_sessions.txt"
        sessions_path.write_text("\n".join(sessions) + "\n", encoding="utf-8")
        calendar_target = stage / "exact_contract_calendar_v1.parquet"
        shutil.copy2(calendar_path, calendar_target)
        frame_artifacts = {
            "candidate_requirements.parquet": requirements,
            "entry_source_lineage.parquet": entry_lineage,
            "metadata_manifest.parquet": pl.from_dicts(
                metadata_rows, infer_schema_length=None
            ).sort("Date"),
            "extension_exact_mapping_audit.parquet": extension_audit,
            "extension_source_manifest.parquet": extension_sources,
        }
        for filename, frame in frame_artifacts.items():
            frame.write_parquet(stage / filename, compression="zstd", statistics=True)

        artifact_paths = {
            "candidate_sessions.txt": sessions_path,
            "exact_contract_calendar_v1.parquet": calendar_target,
            **{name: stage / name for name in frame_artifacts},
        }
        artifacts = {
            name: _artifact_metadata(path)
            for name, path in artifact_paths.items()
        }
        config = {
            "schema_version": PREREQUISITE_SCHEMA_VERSION,
            "base_sessions_path": str(base_path),
            "base_sessions_sha256": _file_sha256(base_path),
            "extension_sessions": list(extension),
            "contract_calendar_path": str(calendar_path),
            "contract_calendar_sha256": _file_sha256(calendar_path),
            "entry_execution_root": str(entry_root),
            "product_days_path": str(product_path),
            "product_days_sha256": _file_sha256(product_path),
            "existing_metadata_root": str(old_metadata_root),
            "data_root": str(data_root),
            "futures_raw_root": str(futures_raw_root),
            "builder_source_sha256": _file_sha256(Path(__file__)),
        }
        marker = {
            "complete": True,
            "schema_version": PREREQUISITE_SCHEMA_VERSION,
            "config": config,
            "config_sha256": _canonical_sha256(config),
            "session_count": len(sessions),
            "base_session_count": len(base_sessions),
            "extension_session_count": len(extension),
            "metadata_session_count": len(metadata_rows),
            "candidate_requirement_count": requirements.height,
            "extension_requirement_count": extension_audit.height,
            "artifacts": artifacts,
            "metadata_artifacts": {
                Path(str(row["artifact"])).name: {
                    "rows": row["artifact_rows"],
                    "bytes": row["artifact_bytes"],
                    "sha256": row["artifact_sha256"],
                }
                for row in metadata_rows
            },
            "fact_semantics": {
                "entry_cohort_unchanged": True,
                "candidate_calendar_extended_only": True,
                "point_in_time_contract_metadata": True,
                "exact_quote_code_no_roll": True,
                "extension_raw_sources_content_hashed": True,
                "extension_market_reference_validated": True,
            },
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(destination)
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verified = verify_cross_session_prerequisites(destination)
    return _result_from_payload(destination, verified, resumed=False)


def verify_cross_session_prerequisites(root: Path) -> dict[str, object]:
    """Verify the completion boundary, every metadata file and all hashes."""

    root = Path(root)
    marker = root / "complete.json"
    if not marker.is_file():
        raise FileExistsError(f"cross-session prerequisite is incomplete: {root}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"invalid prerequisite marker: {marker}")
    if payload.get("schema_version") != PREREQUISITE_SCHEMA_VERSION:
        raise ValueError("prerequisite schema version mismatch")
    config = payload.get("config")
    if not isinstance(config, dict) or payload.get("config_sha256") != _canonical_sha256(
        config
    ):
        raise ValueError("prerequisite config hash mismatch")
    artifacts = payload.get("artifacts")
    metadata = payload.get("metadata_artifacts")
    if not isinstance(artifacts, dict) or not isinstance(metadata, dict):
        raise ValueError("prerequisite artifact declarations are invalid")
    _verify_declared_artifacts(root, artifacts)
    _verify_declared_artifacts(root / "metadata", metadata)
    sessions = _load_sessions(root / "candidate_sessions.txt")
    if len(sessions) != int(payload.get("session_count", -1)):
        raise ValueError("prerequisite session count mismatch")
    if len(metadata) != len(sessions) or len(metadata) != int(
        payload.get("metadata_session_count", -1)
    ):
        raise ValueError("prerequisite metadata coverage mismatch")
    required_true = (
        "entry_cohort_unchanged",
        "candidate_calendar_extended_only",
        "point_in_time_contract_metadata",
        "exact_quote_code_no_roll",
        "extension_raw_sources_content_hashed",
        "extension_market_reference_validated",
    )
    semantics = payload.get("fact_semantics")
    if not isinstance(semantics, dict) or any(
        semantics.get(key) is not True for key in required_true
    ):
        raise ValueError("prerequisite fact semantics mismatch")
    return payload


def _candidate_requirements(
    product_days: pl.DataFrame,
    *,
    sessions: tuple[str, ...],
    calendar: pl.DataFrame,
    entry_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    expiry = {
        str(row["QuoteCode"]): str(row["expiry_session"])
        for row in calendar.iter_rows(named=True)
    }
    index = {value: position for position, value in enumerate(sessions)}
    requirements: dict[tuple[str, str, str, str], dict[str, int]] = {}
    lineage: list[dict[str, object]] = []
    for origin, value_code in product_days.sort(["Date", "ValueCode"]).iter_rows():
        origin = _date(str(origin), "origin Date")
        value_code = str(value_code)
        if origin not in index:
            raise ValueError(f"entry Date absent from candidate sessions: {origin}")
        partition = entry_root / f"Date={origin}" / f"ValueCode={value_code}"
        payload, marker_sha = _verify_source_marker(
            partition,
            origin,
            value_code,
            ("execution_action_facts.parquet", "target_audit.parquet"),
        )
        action_path = partition / "execution_action_facts.parquet"
        actions = pl.read_parquet(action_path)
        established = _established_entry_count(actions)
        if established == 0:
            continue
        quote_code = _source_quote_code(
            action_path,
            partition / "target_audit.parquet",
            origin,
            value_code,
        )
        try:
            expiry_session = _date(expiry[quote_code], "expiry_session")
        except KeyError as error:
            raise ValueError(
                f"exact QuoteCode absent from contract calendar: {quote_code}"
            ) from error
        lineage.append(
            {
                "Date": origin,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "established_entry_aliases": established,
                "entry_marker_sha256": marker_sha,
                "entry_config_sha256": str(payload["config_sha256"]),
                "action_artifact_sha256": _file_sha256(action_path),
            }
        )
        for candidate in sessions[index[origin] + 1 :]:
            if candidate > expiry_session:
                break
            key = (candidate, value_code, quote_code, expiry_session)
            counts = requirements.setdefault(
                key, {"origin_product_days": 0, "established_entry_aliases": 0}
            )
            counts["origin_product_days"] += 1
            counts["established_entry_aliases"] += established
    requirement_rows = [
        {
            "candidate_date": key[0],
            "ValueCode": key[1],
            "QuoteCode": key[2],
            "expiry_session": key[3],
            **counts,
        }
        for key, counts in sorted(requirements.items())
    ]
    requirement_frame = pl.from_dicts(
        requirement_rows, infer_schema_length=None
    ).sort(["candidate_date", "ValueCode", "QuoteCode"])
    lineage_frame = pl.from_dicts(lineage, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )
    return requirement_frame, lineage_frame


def _audit_extension_requirements(
    requirements: pl.DataFrame,
    *,
    extension: tuple[str, ...],
    metadata_root: Path,
    data_root: Path,
    futures_raw_root: Path,
    futures_basic_hashes: Mapping[str, str],
    validation_pairs: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for session in extension:
        required = requirements.filter(
            pl.col("candidate_date") == session
        ).with_columns(pl.lit(True).alias("required_by_established_position"))
        universe = validation_pairs.select(
            pl.lit(session).alias("candidate_date"),
            "ValueCode",
            "QuoteCode",
            pl.col("end_date").dt.strftime("%Y%m%d").alias("expiry_session"),
            pl.lit(0, dtype=pl.Int64).alias("origin_product_days"),
            pl.lit(0, dtype=pl.Int64).alias("established_entry_aliases"),
            pl.lit(False).alias("required_by_established_position"),
        )
        needed = pl.concat([required, universe], how="vertical_relaxed").sort(
            ["ValueCode", "required_by_established_position"],
            descending=[False, True],
        ).unique(subset=["ValueCode", "QuoteCode"], keep="first")
        pairs = needed.select("ValueCode", "QuoteCode").unique()
        contracts = pl.read_parquet(metadata_root / f"{session}_contracts.parquet")
        contract_pairs = set(
            contracts.select("ValueCode", "QuoteCode").iter_rows()
        )
        spot_path = data_root / "tickData" / f"{session}_StockTick.parquet"
        feature_path = data_root / "tickFeature" / f"{session}_tickFeature.parquet"
        market_path = data_root / "marketData" / f"{session}_marketData.parquet"
        future_path = (
            futures_raw_root
            / session[:4]
            / session[4:6]
            / session[6:8]
            / "stock_futures.parquet"
        )
        paths = {
            "spot_raw": spot_path,
            "spread_clock": feature_path,
            "spot_reference": market_path,
            "future_raw": future_path,
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            source_rows.append(
                {
                    "Date": session,
                    "source": label,
                    "path": str(path),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _file_sha256(path),
                }
            )
        source_rows.append(
            {
                "Date": session,
                "source": "futures_basic",
                "path": f"mysql://ProductInfo.taifex_pib_view?date={session}",
                "bytes": None,
                "mtime_ns": None,
                "sha256": futures_basic_hashes[session],
            }
        )

        values = pairs["ValueCode"].to_list() if pairs.height else []
        future_pairs = set(
            pl.scan_parquet(future_path)
            .filter(pl.col("ValueCode").cast(pl.String).is_in(values))
            .select(
                pl.col("ValueCode").cast(pl.String),
                pl.col("QuoteCode").cast(pl.String),
            )
            .unique()
            .collect()
            .iter_rows()
        )
        spot_values = set(
            pl.scan_parquet(spot_path)
            .filter(pl.col("ValueCode").cast(pl.String).is_in(values))
            .select(pl.col("ValueCode").cast(pl.String))
            .unique()
            .collect()
            .get_column("ValueCode")
            .to_list()
        )
        feature_values = set(
            pl.scan_parquet(feature_path)
            .filter(pl.col("QuoteCode").cast(pl.String).is_in(values))
            .select(pl.col("QuoteCode").cast(pl.String))
            .unique()
            .collect()
            .get_column("QuoteCode")
            .to_list()
        )
        market = (
            pl.scan_parquet(market_path)
            .filter(pl.col("quote_code").cast(pl.String).is_in(values))
            .select(
                pl.col("quote_code").cast(pl.String).alias("ValueCode"),
                pl.col("opening_ref_price").cast(pl.Float64).alias(
                    "spot_ref_price"
                ),
                pl.col("allow_day_trade_mark").cast(pl.String).alias(
                    "day_trade_mark"
                ),
            )
            .collect()
            .unique()
        )
        market_lookup: dict[str, tuple[float, str]] = {}
        for value_code in values:
            selected = market.filter(pl.col("ValueCode") == value_code)
            distinct = selected.select("spot_ref_price", "day_trade_mark").unique()
            if distinct.height != 1:
                continue
            spot_ref, mark = distinct.row(0)
            if _positive(spot_ref) and str(mark):
                market_lookup[str(value_code)] = (float(spot_ref), str(mark))

        contract_lookup = {
            (str(row["ValueCode"]), str(row["QuoteCode"])): row
            for row in contracts.iter_rows(named=True)
        }
        for row in needed.iter_rows(named=True):
            pair = (str(row["ValueCode"]), str(row["QuoteCode"]))
            contract = contract_lookup.get(pair)
            reference = market_lookup.get(pair[0])
            metadata_exact = pair in contract_pairs and contract is not None
            market_valid = reference is not None
            audit_rows.append(
                {
                    **row,
                    "metadata_exact_pair": metadata_exact,
                    "future_ref_price": (
                        float(contract["fut_ref_price"])
                        if contract is not None
                        else None
                    ),
                    "contract_size": (
                        float(contract["contract_size"])
                        if contract is not None
                        else None
                    ),
                    "spot_ref_price": reference[0] if reference else None,
                    "day_trade_mark": reference[1] if reference else None,
                    "future_raw_exact_events": pair in future_pairs,
                    "spot_raw_events": pair[0] in spot_values,
                    "spread_clock_rows": pair[0] in feature_values,
                    "mapping_valid": metadata_exact and market_valid,
                }
            )
    audit = pl.from_dicts(audit_rows, infer_schema_length=None).sort(
        ["candidate_date", "ValueCode", "QuoteCode"]
    )
    sources = pl.from_dicts(source_rows, infer_schema_length=None).sort(
        ["Date", "source"]
    )
    return audit, sources


def _validate_contract_frame(frame: pl.DataFrame, session: str) -> pl.DataFrame:
    required = {
        "QuoteCode",
        "ValueCode",
        "contract_size",
        "decimal_locator",
        "end_date",
        "fut_ref_price",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{session}: contract metadata missing columns: {missing}")
    selected = frame.select(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("decimal_locator").cast(pl.Int16),
        pl.col("end_date").cast(pl.Date),
        pl.col("fut_ref_price").cast(pl.Float64),
    ).sort(["ValueCode", "QuoteCode"])
    if selected.is_empty() or selected.null_count().sum_horizontal().item() != 0:
        raise ValueError(f"{session}: contract metadata is empty or null")
    invalid = selected.filter(
        (pl.col("contract_size") <= 0)
        | (pl.col("fut_ref_price") <= 0)
        | (pl.col("end_date") < pl.lit(parse_date(session).date()))
    )
    if invalid.height:
        raise ValueError(f"{session}: contract metadata has invalid values")
    if selected.select("ValueCode").n_unique() != selected.height or selected.select(
        "QuoteCode"
    ).n_unique() != selected.height:
        raise ValueError(f"{session}: contract metadata is not one-to-one")
    return selected


def _validate_calendar(calendar: pl.DataFrame, sessions: tuple[str, ...]) -> None:
    if calendar.is_empty() or calendar.select("QuoteCode").n_unique() != calendar.height:
        raise ValueError("exact contract calendar must be nonempty and unique")
    if calendar.filter(
        ~pl.col("expiry_session").str.contains(r"^\d{8}$")
        | pl.col("calendar_version").is_null()
    ).height:
        raise ValueError("exact contract calendar contains invalid rows")
    if sessions != tuple(sorted(sessions)) or len(sessions) != len(set(sessions)):
        raise ValueError("candidate sessions must be unique and ascending")


def _artifact_metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pl.scan_parquet(path).select(pl.len()).collect().item()
        result["columns"] = len(pl.read_parquet_schema(path))
    return result


def _verify_declared_artifacts(
    root: Path, artifacts: Mapping[str, object]
) -> None:
    for filename, metadata in artifacts.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise ValueError(f"invalid prerequisite artifact declaration: {filename}")
        path = Path(root) / filename
        if not path.is_file() or _file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"prerequisite artifact hash mismatch: {path}")
        if metadata.get("bytes") is not None and path.stat().st_size != int(
            metadata["bytes"]
        ):
            raise ValueError(f"prerequisite artifact byte count mismatch: {path}")
        if path.suffix != ".parquet":
            continue
        schema = pl.read_parquet_schema(path)
        if metadata.get("columns") is not None and len(schema) != int(
            metadata["columns"]
        ):
            raise ValueError(f"prerequisite artifact column mismatch: {path}")
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        if metadata.get("rows") is not None and rows != int(metadata["rows"]):
            raise ValueError(f"prerequisite artifact row count mismatch: {path}")


def _frame_sha256(frame: pl.DataFrame) -> str:
    columns = sorted(frame.columns)
    selected = frame.select(columns)
    if columns and not selected.is_empty():
        selected = selected.sort(columns, nulls_last=True)
    return hashlib.sha256(selected.write_json().encode()).hexdigest()


def _result_from_payload(
    root: Path, payload: Mapping[str, object], *, resumed: bool
) -> CrossSessionPrerequisiteResult:
    return CrossSessionPrerequisiteResult(
        output_root=root,
        session_count=int(payload["session_count"]),
        metadata_session_count=int(payload["metadata_session_count"]),
        candidate_requirement_count=int(payload["candidate_requirement_count"]),
        extension_requirement_count=int(payload["extension_requirement_count"]),
        resumed=resumed,
    )


def _load_sessions(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        not values
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
        or any(len(value) != 8 or not value.isdigit() for value in values)
    ):
        raise ValueError("sessions must be unique ascending YYYYMMDD values")
    return values


def _positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _date(value: str, label: str) -> str:
    result = str(value)
    if len(result) != 8 or not result.isdigit():
        raise ValueError(f"{label} must be YYYYMMDD")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned candidate sessions and PIT contract metadata"
    )
    parser.add_argument("--base-sessions", type=Path, default=DEFAULT_BASE_SESSIONS)
    parser.add_argument(
        "--extension-session", action="append", dest="extension_sessions"
    )
    parser.add_argument(
        "--contract-calendar", type=Path, default=DEFAULT_CONTRACT_CALENDAR
    )
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument("--product-days", type=Path, default=DEFAULT_PRODUCT_DAYS)
    parser.add_argument(
        "--existing-metadata-root",
        type=Path,
        default=DEFAULT_EXISTING_METADATA_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=HFT_DATA_ROOT)
    parser.add_argument("--futures-root", type=Path, default=DEFAULT_FUTURES_RAW_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_cross_session_prerequisites(
        base_sessions_path=args.base_sessions,
        extension_sessions=(
            tuple(args.extension_sessions)
            if args.extension_sessions
            else DEFAULT_EXTENSION_SESSIONS
        ),
        contract_calendar_path=args.contract_calendar,
        entry_execution_root=args.entry_root,
        product_days_path=args.product_days,
        existing_metadata_root=args.existing_metadata_root,
        output_root=args.output,
        data_root=args.data_root,
        futures_raw_root=args.futures_root,
        resume=not args.no_resume,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
