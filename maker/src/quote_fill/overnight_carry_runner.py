"""Atomic, resumable batch runner for strict overnight carry labels.

The exit-maker and entry-execution trees are immutable inputs.  Every
published product-day binds their exact selected artifact hashes, the session
calendar, the exact-contract expiry calendar, optional final settlements and
the raw-file stat fingerprints used for later sessions.

Raw tape is shared across the pending products of one entry date.  The mapping
passed to the normalizer deliberately retains the entry ``QuoteCode`` even if
the next day's preferred contract changed; this runner never substitutes or
prices a roll.  Missing raw files are not skipped: the labeler receives no tape
for that first missing candidate session and censors the path there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .overnight_carry import (
    STRICT_CARRY_BRANCHES,
    OvernightCarryConfig,
    build_strict_overnight_carry_labels,
)
from .raw_tape import RawTapeDay, load_raw_tape_day


OVERNIGHT_CARRY_RUNNER_VERSION = "strict_overnight_carry_product_day_v1"
DEFAULT_ENTRY_EXECUTION_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "execution_narrow_60d"
)
DEFAULT_EXIT_MAKER_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "exit_maker_narrow_60d"
)
DEFAULT_OVERNIGHT_CARRY_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "overnight_carry_narrow_60d"
)
DEFAULT_FUTURES_RAW_ROOT = Path("/mnt/NAS/Parquet/Ticks")
OVERNIGHT_CARRY_MANIFEST_NAME = "overnight_carry_partition_manifest.parquet"

_POSITION_ARTIFACT = "exit_maker_position_policy_facts.parquet"
_ACTION_ARTIFACT = "execution_action_facts.parquet"
_LABEL_ARTIFACT = "overnight_carry_labels.parquet"
_AUDIT_ARTIFACT = "overnight_carry_audit.parquet"
_OUTPUT_ARTIFACTS = (_LABEL_ARTIFACT, _AUDIT_ARTIFACT)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_COMPATIBILITY_MANIFEST = Path(__file__).with_name(
    "overnight_carry_runner_compatibility.json"
)

_EMPTY_ENTRY_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "policy_generation_id": pl.String,
    "raw_order_fact_id": pl.String,
    "full_fill": pl.Boolean,
    "entry_hedge_status": pl.String,
    "entry_hedge_contract_size_shares": pl.Int64,
    "entry_spot_price": pl.Float64,
    "entry_future_price": pl.Float64,
}


@dataclass(frozen=True)
class OvernightCarryRunnerConfig:
    """Frozen input and carry semantics for one replay root."""

    carry: OvernightCarryConfig = OvernightCarryConfig()
    position_source_artifact: str = _POSITION_ARTIFACT
    action_source_artifact: str = _ACTION_ARTIFACT
    raw_mapping_reference_policy: str = (
        "entry_execution_prices_for_normalizer_metadata_only_v1"
    )
    raw_source_integrity: str = "path_size_mtime_ns_fingerprint_v1"
    runner_version: str = OVERNIGHT_CARRY_RUNNER_VERSION

    def validate(self) -> None:
        self.carry.validate()
        for name in (
            "position_source_artifact",
            "action_source_artifact",
            "raw_mapping_reference_policy",
            "raw_source_integrity",
            "runner_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("position_source_artifact", "action_source_artifact"):
            value = getattr(self, name)
            if Path(value).name != value:
                raise ValueError(f"{name} must be a plain filename")
        if self.runner_version != OVERNIGHT_CARRY_RUNNER_VERSION:
            raise ValueError(
                f"runner_version must be {OVERNIGHT_CARRY_RUNNER_VERSION!r}"
            )


@dataclass(frozen=True)
class OvernightCarryPartitionSource:
    date: str
    value_code: str
    exit_partition: Path
    entry_partition: Path
    position_path: Path
    action_path: Path
    exit_marker_sha256: str
    entry_marker_sha256: str
    position_sha256: str
    action_sha256: str
    exit_runner_version: str
    entry_runner_version: str

    def payload(self, config: OvernightCarryRunnerConfig) -> dict[str, object]:
        return {
            "Date": self.date,
            "ValueCode": self.value_code,
            "exit_position_source": {
                "partition": str(self.exit_partition),
                "runner_version": self.exit_runner_version,
                "marker_sha256": self.exit_marker_sha256,
                "artifact": config.position_source_artifact,
                "sha256": self.position_sha256,
            },
            "entry_action_source": {
                "partition": str(self.entry_partition),
                "runner_version": self.entry_runner_version,
                "marker_sha256": self.entry_marker_sha256,
                "artifact": config.action_source_artifact,
                "sha256": self.action_sha256,
            },
        }


RawTapeLoader = Callable[[str, pl.DataFrame], RawTapeDay]


def run_overnight_carry_replay(
    product_days: Iterable[tuple[str, str]],
    *,
    sessions: Sequence[str] | Iterable[str],
    contract_calendar: pl.DataFrame,
    settlement_facts: pl.DataFrame | None = None,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    output_root: Path = DEFAULT_OVERNIGHT_CARRY_ROOT,
    config: OvernightCarryRunnerConfig = OvernightCarryRunnerConfig(),
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    resume: bool = True,
    raw_tape_loader: RawTapeLoader | None = None,
) -> pl.DataFrame:
    """Build strict carry labels and atomically publish product-day facts.

    ``sessions`` is the full ascending observation calendar, not merely the
    selected entry-day window.  This distinction lets the last selected entry
    session use a subsequent raw session when one is supplied.
    """

    config.validate()
    session_values = _normalise_sessions(sessions)
    session_index = {date: index for index, date in enumerate(session_values)}
    keys = tuple((str(date), str(value)) for date, value in product_days)
    if not keys:
        raise ValueError("product_days must be nonempty")
    if len(keys) != len(set(keys)):
        raise ValueError("product_days contains duplicate keys")
    for date, value_code in keys:
        _validate_partition_key(date, value_code)
        if date not in session_index:
            raise ValueError(f"entry date absent from sessions: {date}")

    exit_root = Path(exit_maker_root)
    entry_root = Path(entry_execution_root)
    destination = Path(output_root)
    _validate_distinct_roots(exit_root, entry_root, destination)
    destination.mkdir(parents=True, exist_ok=True)

    session_digest = _canonical_sha256(list(session_values))
    calendar_digest = _frame_sha256(contract_calendar)
    settlement_digest = (
        None if settlement_facts is None else _frame_sha256(settlement_facts)
    )
    global_input_payload = {
        "session_calendar": {
            "count": len(session_values),
            "first": session_values[0],
            "last": session_values[-1],
            "sha256": session_digest,
        },
        "contract_calendar_content_sha256": calendar_digest,
        "settlement_facts_content_sha256": settlement_digest,
    }
    # Global input semantics belong in the runner hash.  This prevents one
    # output root from accumulating partitions built with different session,
    # expiry-calendar, or settlement snapshots.
    runtime_runner_payload = _runner_payload(config, global_input_payload)
    runner_payload, compatibility = _resolve_compatible_runner_payload(
        destination, runtime_runner_payload
    )
    runner_sha = _canonical_sha256(runner_payload)
    _preflight_output_root(destination, runner_sha)

    by_date: dict[str, list[str]] = {}
    for date, value_code in keys:
        by_date.setdefault(date, []).append(value_code)

    for date, value_codes in by_date.items():
        candidate_dates = _candidate_sessions(
            date,
            session_values,
            session_index,
            config.carry.max_carry_sessions,
        )
        raw_fingerprints = _candidate_raw_fingerprints(
            candidate_dates,
            data_root=Path(data_root),
            futures_raw_root=Path(futures_raw_root),
            custom_loader=raw_tape_loader is not None,
        )
        pending: list[
            tuple[
                str,
                OvernightCarryPartitionSource,
                dict[str, object],
                str,
                pl.DataFrame,
                pl.DataFrame,
            ]
        ] = []
        for value_code in value_codes:
            source = verify_overnight_sources(
                date,
                value_code,
                exit_maker_root=exit_root,
                entry_execution_root=entry_root,
                config=config,
            )
            partition = destination / f"Date={date}" / f"ValueCode={value_code}"
            partition_payload: dict[str, object] = {
                "runner": runner_payload,
                "source": source.payload(config),
                **global_input_payload,
                "candidate_sessions": list(candidate_dates),
                "raw_source_fingerprints": raw_fingerprints,
            }
            # Preserve the exact config hash of already-complete predecessor
            # partitions.  Newly published hotfix partitions bind both the
            # compatibility baseline and the actual runtime implementation.
            if compatibility is not None and (
                not partition.exists()
                or _partition_binds_runtime_compatibility(partition)
            ):
                partition_payload.update(
                    {
                        "runtime_implementation_sources": runtime_runner_payload[
                            "implementation_sources"
                        ],
                        "runner_compatibility": compatibility,
                    }
                )
            config_sha = _canonical_sha256(partition_payload)
            if partition.exists():
                if not resume:
                    raise FileExistsError(partition)
                verify_overnight_output_partition(
                    partition,
                    expected_config_sha256=config_sha,
                    expected_runner_config_sha256=runner_sha,
                )
                continue
            positions = pl.read_parquet(source.position_path)
            actions = pl.read_parquet(source.action_path)
            _validate_frame_identity(positions, date, value_code, "position facts")
            _validate_frame_identity(actions, date, value_code, "entry action facts")
            pending.append(
                (
                    value_code,
                    source,
                    partition_payload,
                    config_sha,
                    positions,
                    actions,
                )
            )

        if not pending:
            continue

        mappings = [
            _exact_raw_mapping(positions, actions)
            for _, _, _, _, positions, actions in pending
        ]
        mappings = [frame for frame in mappings if not frame.is_empty()]
        combined_mapping = (
            pl.concat(mappings, how="vertical_relaxed").sort(
                ["ValueCode", "QuoteCode"]
            )
            if mappings
            else pl.DataFrame(schema=_mapping_schema())
        )
        _validate_combined_mapping(combined_mapping)
        tapes: list[RawTapeDay] = []
        if not combined_mapping.is_empty():
            for candidate_date in candidate_dates:
                try:
                    tape = (
                        raw_tape_loader(candidate_date, combined_mapping)
                        if raw_tape_loader is not None
                        else _load_raw_candidate(
                            candidate_date,
                            combined_mapping,
                            data_root=Path(data_root),
                            futures_raw_root=Path(futures_raw_root),
                        )
                    )
                except FileNotFoundError:
                    # First missing candidate is a terminal censor.  Do not
                    # inspect or load any later raw session.
                    break
                _validate_loaded_tape(tape, candidate_date, combined_mapping)
                tapes.append(tape)

        for (
            value_code,
            _source,
            partition_payload,
            config_sha,
            positions,
            actions,
        ) in pending:
            product_tapes = tuple(
                _slice_tape(tape, value_code) for tape in tapes
            )
            result = build_strict_overnight_carry_labels(
                positions,
                (
                    pl.DataFrame(schema=_EMPTY_ENTRY_ACTION_SCHEMA)
                    if positions.is_empty()
                    else actions
                ),
                product_tapes,
                session_values,
                contract_calendar,
                settlement_facts=settlement_facts,
                config=config.carry,
            )
            audit = result.audit.with_columns(
                pl.lit(date).alias("Date"),
                pl.lit(value_code).alias("ValueCode"),
            ).select("Date", "ValueCode", pl.exclude("Date", "ValueCode"))
            partition = destination / f"Date={date}" / f"ValueCode={value_code}"
            _publish_partition(
                partition,
                date=date,
                value_code=value_code,
                labels=result.labels,
                audit=audit,
                partition_config=partition_payload,
                config_sha256=config_sha,
                runner_config_sha256=runner_sha,
            )
        del tapes, pending

    return _rebuild_root_manifest(destination, runner_sha)


def discover_overnight_product_days(
    entry_sessions: Sequence[str],
    symbols: Sequence[str],
    *,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
) -> tuple[tuple[tuple[str, str], ...], pl.DataFrame]:
    """Intersect the requested grid with complete exit and entry partitions."""

    dates = tuple(str(value) for value in entry_sessions)
    values = tuple(str(value) for value in symbols)
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("entry_sessions must be nonempty and unique")
    if not values or len(values) != len(set(values)):
        raise ValueError("symbols must be nonempty and unique")
    rows: list[dict[str, object]] = []
    keys: list[tuple[str, str]] = []
    for date in dates:
        for value_code in values:
            _validate_partition_key(date, value_code)
            exit_partition = (
                Path(exit_maker_root)
                / f"Date={date}"
                / f"ValueCode={value_code}"
            )
            entry_partition = (
                Path(entry_execution_root)
                / f"Date={date}"
                / f"ValueCode={value_code}"
            )
            exit_marker = exit_partition / "complete.json"
            entry_marker = entry_partition / "complete.json"
            for partition, marker, name in (
                (exit_partition, exit_marker, "exit-maker"),
                (entry_partition, entry_marker, "entry"),
            ):
                if partition.exists() and not marker.is_file():
                    raise FileExistsError(
                        f"{name} partition exists without complete.json: {partition}"
                    )
            exit_ok = exit_marker.is_file()
            entry_ok = entry_marker.is_file()
            available = exit_ok and entry_ok
            if available:
                keys.append((date, value_code))
            status = (
                "available"
                if available
                else "missing_exit_and_entry"
                if not exit_ok and not entry_ok
                else "missing_exit_partition"
                if not exit_ok
                else "missing_entry_partition"
            )
            rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "requested": True,
                    "available_exit_partition": exit_ok,
                    "available_entry_partition": entry_ok,
                    "availability_status": status,
                }
            )
    return (
        tuple(keys),
        pl.from_dicts(rows, infer_schema_length=None).sort(
            ["Date", "ValueCode"]
        ),
    )


def verify_overnight_sources(
    date: str,
    value_code: str,
    *,
    exit_maker_root: Path,
    entry_execution_root: Path,
    config: OvernightCarryRunnerConfig,
) -> OvernightCarryPartitionSource:
    """Verify and bind the two immutable upstream artifacts."""

    exit_partition = (
        Path(exit_maker_root) / f"Date={date}" / f"ValueCode={value_code}"
    )
    entry_partition = (
        Path(entry_execution_root) / f"Date={date}" / f"ValueCode={value_code}"
    )
    exit_payload, exit_hash = _verify_source_marker(
        exit_partition,
        date,
        value_code,
        config.position_source_artifact,
    )
    entry_payload, entry_hash = _verify_source_marker(
        entry_partition,
        date,
        value_code,
        config.action_source_artifact,
    )
    exit_meta = exit_payload["artifacts"][config.position_source_artifact]
    entry_meta = entry_payload["artifacts"][config.action_source_artifact]
    return OvernightCarryPartitionSource(
        date=date,
        value_code=value_code,
        exit_partition=exit_partition,
        entry_partition=entry_partition,
        position_path=exit_partition / config.position_source_artifact,
        action_path=entry_partition / config.action_source_artifact,
        exit_marker_sha256=exit_hash,
        entry_marker_sha256=entry_hash,
        position_sha256=str(exit_meta["sha256"]),
        action_sha256=str(entry_meta["sha256"]),
        exit_runner_version=str(exit_payload.get("runner_version", "unknown")),
        entry_runner_version=str(entry_payload.get("runner_version", "unknown")),
    )


def verify_overnight_output_partition(
    partition: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_runner_config_sha256: str | None = None,
) -> dict[str, object]:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileExistsError(
            f"existing overnight partition is incomplete: {partition}"
        )
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"overnight marker is not complete: {partition}")
    if payload.get("runner_version") != OVERNIGHT_CARRY_RUNNER_VERSION:
        raise ValueError(f"overnight runner version mismatch: {partition}")
    if (
        expected_config_sha256 is not None
        and payload.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError(f"overnight partition config mismatch: {partition}")
    if (
        expected_runner_config_sha256 is not None
        and payload.get("runner_config_sha256")
        != expected_runner_config_sha256
    ):
        raise ValueError(f"overnight runner config mismatch: {partition}")
    config = payload.get("config")
    if not isinstance(config, dict) or _canonical_sha256(config) != payload.get(
        "config_sha256"
    ):
        raise ValueError(f"overnight config hash mismatch: {partition}")
    embedded = config.get("runner")
    if not isinstance(embedded, dict) or _canonical_sha256(embedded) != payload.get(
        "runner_config_sha256"
    ):
        raise ValueError(f"overnight embedded runner hash mismatch: {partition}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_OUTPUT_ARTIFACTS):
        raise ValueError(f"overnight artifact set mismatch: {partition}")
    _verify_artifacts(Path(partition), artifacts)
    return _manifest_row(Path(partition), payload)


def _partition_binds_runtime_compatibility(partition: Path) -> bool:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        return False
    payload = _read_json(marker)
    config = payload.get("config")
    return isinstance(config, dict) and "runner_compatibility" in config


def _exact_raw_mapping(
    positions: pl.DataFrame, actions: pl.DataFrame
) -> pl.DataFrame:
    # A zero-established exit partition is a valid product-day.  Its upstream
    # execution action artifact may intentionally use the minimal typed-empty
    # schema because no entry action can be referenced.  Short-circuit before
    # requiring priced-entry columns; non-empty positions remain fail-closed.
    if positions.is_empty():
        return pl.DataFrame(schema=_mapping_schema())
    required_positions = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_policy_generation_id",
        "branch_status",
    }
    required_actions = {
        "policy_generation_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_spot_price",
        "entry_future_price",
        "entry_hedge_contract_size_shares",
    }
    _require(positions, required_positions, "position facts")
    _require(actions, required_actions, "entry action facts")
    carry = positions.filter(
        pl.col("branch_status").is_in(sorted(STRICT_CARRY_BRANCHES))
    )
    if carry.is_empty():
        return pl.DataFrame(schema=_mapping_schema())
    identifiers = carry.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_policy_generation_id",
    ).unique()
    selected = identifiers.join(
        actions.select(sorted(required_actions)),
        left_on="entry_policy_generation_id",
        right_on="policy_generation_id",
        how="left",
        suffix="_entry",
        validate="m:1",
    )
    if selected["entry_spot_price"].null_count() or selected[
        "entry_future_price"
    ].null_count() or selected["entry_hedge_contract_size_shares"].null_count():
        raise ValueError("strict carry rows are missing priced entry actions")
    for left, right in (
        ("Date", "Date_entry"),
        ("ValueCode", "ValueCode_entry"),
        ("QuoteCode", "QuoteCode_entry"),
    ):
        bad = selected.filter(
            pl.col(left).cast(pl.String) != pl.col(right).cast(pl.String)
        )
        if bad.height:
            raise ValueError(f"strict carry action identity mismatch on {left}")
    identity = selected.select("ValueCode", "QuoteCode").unique()
    if identity.height != 1:
        raise ValueError("one product-day must retain one exact QuoteCode")
    contract_sizes = selected[
        "entry_hedge_contract_size_shares"
    ].drop_nulls().unique()
    if len(contract_sizes) != 1 or int(contract_sizes[0]) <= 0:
        raise ValueError("strict carry rows disagree on positive contract size")
    spot_reference = _first_finite_positive(selected["entry_spot_price"])
    future_reference = _first_finite_positive(selected["entry_future_price"])
    row = identity.row(0, named=True)
    return pl.DataFrame(
        {
            "ValueCode": [str(row["ValueCode"])],
            "QuoteCode": [str(row["QuoteCode"])],
            # These references are required normalizer metadata only.  The
            # labeler prices every terminal leg from raw L1--L5 states.
            "spot_ref_price": [spot_reference],
            "fut_ref_price": [future_reference],
            "contract_size": [float(contract_sizes[0])],
        },
        schema=_mapping_schema(),
    )


def _first_finite_positive(series: pl.Series) -> float:
    for value in series:
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    raise ValueError("strict carry entry prices must contain a finite positive value")


def _mapping_schema() -> dict[str, pl.DataType]:
    return {
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "spot_ref_price": pl.Float64,
        "fut_ref_price": pl.Float64,
        "contract_size": pl.Float64,
    }


def _validate_combined_mapping(mapping: pl.DataFrame) -> None:
    if mapping.is_empty():
        return
    for column in ("ValueCode", "QuoteCode"):
        if mapping.select(column).n_unique() != mapping.height:
            raise ValueError(
                f"pending exact-contract mapping is not one-to-one on {column}"
            )


def _load_raw_candidate(
    date: str,
    mapping: pl.DataFrame,
    *,
    data_root: Path,
    futures_raw_root: Path,
) -> RawTapeDay:
    spot_path, future_path = _raw_paths(
        date, data_root=data_root, futures_raw_root=futures_raw_root
    )
    if not spot_path.is_file():
        raise FileNotFoundError(spot_path)
    if not future_path.is_file():
        raise FileNotFoundError(future_path)
    return load_raw_tape_day(
        date,
        mapping,
        spot_path=spot_path,
        future_path=future_path,
    )


def _raw_paths(
    date: str, *, data_root: Path, futures_raw_root: Path
) -> tuple[Path, Path]:
    return (
        Path(data_root) / "tickData" / f"{date}_StockTick.parquet",
        Path(futures_raw_root)
        / date[:4]
        / date[4:6]
        / date[6:8]
        / "stock_futures.parquet",
    )


def _candidate_raw_fingerprints(
    dates: tuple[str, ...],
    *,
    data_root: Path,
    futures_raw_root: Path,
    custom_loader: bool,
) -> list[dict[str, object]]:
    if custom_loader:
        return [
            {
                "Date": date,
                "source": "injected_raw_tape_loader",
                "content_integrity_bound": False,
            }
            for date in dates
        ]
    rows: list[dict[str, object]] = []
    for date in dates:
        spot, future = _raw_paths(
            date, data_root=data_root, futures_raw_root=futures_raw_root
        )
        rows.append(
            {
                "Date": date,
                "spot": _stat_fingerprint(spot),
                "future": _stat_fingerprint(future),
                "content_integrity_bound": False,
            }
        )
    return rows


def _stat_fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validate_loaded_tape(
    tape: RawTapeDay, date: str, mapping: pl.DataFrame
) -> None:
    if not isinstance(tape, RawTapeDay) or str(tape.date) != date:
        raise ValueError("raw_tape_loader returned a different date or type")
    expected = set(
        zip(
            mapping["ValueCode"].cast(pl.String),
            mapping["QuoteCode"].cast(pl.String),
        )
    )
    actual = set(
        zip(
            tape.mapping["ValueCode"].cast(pl.String),
            tape.mapping["QuoteCode"].cast(pl.String),
        )
    )
    if actual != expected:
        raise ValueError("raw_tape_loader mapping differs from exact carry mapping")


def _slice_tape(tape: RawTapeDay, value_code: str) -> RawTapeDay:
    def selected(frame: pl.DataFrame) -> pl.DataFrame:
        if "ValueCode" not in frame.columns:
            return frame
        return frame.filter(pl.col("ValueCode").cast(pl.String) == value_code)

    return RawTapeDay(
        date=str(tape.date),
        mapping=selected(tape.mapping),
        spot_states=selected(tape.spot_states),
        future_states=selected(tape.future_states),
        spot_trades=selected(tape.spot_trades),
        future_trades=selected(tape.future_trades),
        audit=selected(tape.audit),
    )


def _publish_partition(
    partition: Path,
    *,
    date: str,
    value_code: str,
    labels: pl.DataFrame,
    audit: pl.DataFrame,
    partition_config: Mapping[str, object],
    config_sha256: str,
    runner_config_sha256: str,
) -> None:
    partition.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=partition.parent)
    )
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name, frame in ((_LABEL_ARTIFACT, labels), (_AUDIT_ARTIFACT, audit)):
            path = stage / name
            frame.write_parquet(path)
            artifacts[name] = _artifact_metadata(path, frame)
        marker = {
            "complete": True,
            "Date": date,
            "ValueCode": value_code,
            "runner_version": OVERNIGHT_CARRY_RUNNER_VERSION,
            "runner_config_sha256": runner_config_sha256,
            "config": dict(partition_config),
            "config_sha256": config_sha256,
            "artifacts": artifacts,
            "fact_semantics": {
                "strict_carry_branches_only": True,
                "cancel_race_unknown_excluded": True,
                "exact_quote_code_no_roll": True,
                "first_missing_candidate_censors": True,
                "expiry_requires_final_settlement": True,
                "cost_columns_null": True,
                "pathwise_ev_ready": False,
                "raw_content_sha256_bound": False,
            },
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if partition.exists():
            raise FileExistsError(partition)
        stage.replace(partition)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _preflight_output_root(root: Path, runner_sha: str) -> None:
    for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json")):
        verify_overnight_output_partition(
            marker.parent, expected_runner_config_sha256=runner_sha
        )


def _rebuild_root_manifest(root: Path, runner_sha: str) -> pl.DataFrame:
    rows = [
        verify_overnight_output_partition(
            marker.parent, expected_runner_config_sha256=runner_sha
        )
        for marker in sorted(root.glob("Date=*/ValueCode=*/complete.json"))
    ]
    manifest = (
        pl.from_dicts(rows, infer_schema_length=None).sort(["Date", "ValueCode"])
        if rows
        else pl.DataFrame()
    )
    if not manifest.is_empty():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".overnight_carry_manifest.",
            suffix=".tmp.parquet",
            dir=root,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.unlink()
            manifest.write_parquet(temporary)
            temporary.replace(root / OVERNIGHT_CARRY_MANIFEST_NAME)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def _manifest_row(partition: Path, payload: Mapping[str, object]) -> dict[str, object]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    return {
        "Date": str(payload["Date"]),
        "ValueCode": str(payload["ValueCode"]),
        "partition": str(partition),
        "runner_config_sha256": str(payload["runner_config_sha256"]),
        "config_sha256": str(payload["config_sha256"]),
        "overnight_carry_labels_rows": int(artifacts[_LABEL_ARTIFACT]["rows"]),
        "overnight_carry_audit_rows": int(artifacts[_AUDIT_ARTIFACT]["rows"]),
        "complete": True,
    }


def _verify_source_marker(
    partition: Path, date: str, value_code: str, artifact: str
) -> tuple[dict[str, object], str]:
    marker = Path(partition) / "complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"upstream partition is not complete: {marker}")
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"upstream completion marker is false: {marker}")
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"upstream completion marker identity mismatch: {marker}")
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"upstream marker has invalid config: {marker}")
    if _canonical_sha256(config) != config_sha:
        raise ValueError(f"upstream config hash mismatch: {marker}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or artifact not in artifacts:
        raise ValueError(f"upstream marker is missing artifact {artifact}: {marker}")
    _verify_artifacts(Path(partition), {artifact: artifacts[artifact]})
    return payload, _file_sha256(marker)


def _verify_artifacts(partition: Path, artifacts: Mapping[str, object]) -> None:
    for filename, metadata in artifacts.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise ValueError(f"invalid artifact declaration: {partition}/{filename}")
        path = partition / filename
        if not path.is_file():
            raise ValueError(f"artifact is missing: {path}")
        if _file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {path}")
        frame = pl.read_parquet_schema(path)
        expected_columns = metadata.get("columns")
        if expected_columns is not None and len(frame) != int(expected_columns):
            raise ValueError(f"artifact column count mismatch: {path}")
        expected_rows = metadata.get("rows")
        if expected_rows is not None:
            actual_rows = (
                pl.scan_parquet(path).select(pl.len()).collect().item()
            )
            if actual_rows != int(expected_rows):
                raise ValueError(f"artifact row count mismatch: {path}")
        expected_bytes = metadata.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise ValueError(f"artifact byte count mismatch: {path}")


def _artifact_metadata(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _runner_payload(
    config: OvernightCarryRunnerConfig,
    global_input_payload: Mapping[str, object],
) -> dict[str, object]:
    payload = asdict(config)
    payload["global_inputs"] = dict(global_input_payload)
    module_root = Path(__file__).parent
    payload["implementation_sources"] = {
        name: _file_sha256(module_root / name)
        for name in (
            "overnight_carry_runner.py",
            "overnight_carry.py",
            "raw_tape.py",
            "hedge_study.py",
            "layered.py",
            "ev_surface.py",
        )
    }
    if _COMPATIBILITY_MANIFEST.is_file():
        payload["implementation_sources"][_COMPATIBILITY_MANIFEST.name] = (
            _file_sha256(_COMPATIBILITY_MANIFEST)
        )
    return payload


def _resolve_compatible_runner_payload(
    root: Path,
    runtime_payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Adopt one explicitly allowlisted predecessor for an in-place resume.

    Compatibility is deliberately exact and one-way.  It exists only for the
    zero-position/minimal-empty-action hotfix: completed predecessor facts do
    not depend on that branch and remain byte-valid, while new partitions bind
    the actual runtime sources in their product-day config.
    """

    markers = sorted(Path(root).glob("Date=*/ValueCode=*/complete.json"))
    if not markers:
        return dict(runtime_payload), None
    predecessor_payloads: dict[str, dict[str, object]] = {}
    for marker in markers:
        payload = _read_json(marker)
        config = payload.get("config")
        runner = config.get("runner") if isinstance(config, dict) else None
        runner_sha = payload.get("runner_config_sha256")
        if not isinstance(runner, dict) or not isinstance(runner_sha, str):
            raise ValueError(f"invalid existing runner payload: {marker}")
        if _canonical_sha256(runner) != runner_sha:
            raise ValueError(f"existing runner payload hash mismatch: {marker}")
        predecessor_payloads[_canonical_sha256(runner)] = runner
    if len(predecessor_payloads) != 1:
        raise ValueError("overnight output root already mixes runner payloads")
    predecessor_sha, predecessor = next(iter(predecessor_payloads.items()))
    runtime = dict(runtime_payload)
    runtime_sha = _canonical_sha256(runtime)
    if predecessor_sha == runtime_sha:
        return runtime, None

    if not _COMPATIBILITY_MANIFEST.is_file():
        raise ValueError(
            "existing overnight runner differs and no compatibility manifest exists"
        )
    manifest = _read_json(_COMPATIBILITY_MANIFEST)
    transitions = manifest.get("compatible_runner_transitions")
    if not isinstance(transitions, list):
        raise ValueError("invalid overnight runner compatibility manifest")
    predecessor_sources = predecessor.get("implementation_sources")
    runtime_sources = runtime.get("implementation_sources")
    if not isinstance(predecessor_sources, dict) or not isinstance(
        runtime_sources, dict
    ):
        raise ValueError("runner payload lacks implementation source hashes")
    source_name = "overnight_carry_runner.py"
    old_source = predecessor_sources.get(source_name)
    new_source = runtime_sources.get(source_name)
    transition = next(
        (
            value
            for value in transitions
            if isinstance(value, dict)
            and value.get("from_runner_source_sha256") == old_source
            and value.get("to_runner_source_sha256") == new_source
        ),
        None,
    )
    if transition is None:
        raise ValueError(
            "existing overnight runner is not an allowlisted compatible predecessor"
        )

    comparable_predecessor = dict(predecessor)
    comparable_runtime = dict(runtime)
    comparable_predecessor_sources = dict(predecessor_sources)
    comparable_runtime_sources = dict(runtime_sources)
    comparable_predecessor_sources.pop(source_name, None)
    comparable_runtime_sources.pop(source_name, None)
    comparable_predecessor_sources.pop(_COMPATIBILITY_MANIFEST.name, None)
    comparable_runtime_sources.pop(_COMPATIBILITY_MANIFEST.name, None)
    comparable_predecessor["implementation_sources"] = comparable_predecessor_sources
    comparable_runtime["implementation_sources"] = comparable_runtime_sources
    if comparable_predecessor != comparable_runtime:
        raise ValueError(
            "allowlisted runner predecessor differs outside the empty-action hotfix"
        )
    compatibility = {
        "compatibility_policy": str(transition.get("compatibility_policy")),
        "from_runner_config_sha256": predecessor_sha,
        "from_runner_source_sha256": str(old_source),
        "runtime_runner_config_sha256": runtime_sha,
        "runtime_runner_source_sha256": str(new_source),
        "compatibility_manifest_sha256": _file_sha256(_COMPATIBILITY_MANIFEST),
    }
    return predecessor, compatibility


def _candidate_sessions(
    date: str,
    sessions: tuple[str, ...],
    index: Mapping[str, int],
    maximum: int,
) -> tuple[str, ...]:
    start = int(index[date]) + 1
    return sessions[start : start + maximum]


def _normalise_sessions(values: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    sessions = tuple(str(value) for value in values)
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("sessions must be nonempty and unique")
    if sessions != tuple(sorted(sessions)):
        raise ValueError("sessions must be ascending")
    if any(len(value) != 8 or not value.isdigit() for value in sessions):
        raise ValueError("sessions must use YYYYMMDD")
    return sessions


def _validate_frame_identity(
    frame: pl.DataFrame, date: str, value_code: str, source: str
) -> None:
    _require(frame, {"Date", "ValueCode"}, source)
    if frame.is_empty():
        return
    if set(frame["Date"].cast(pl.String).drop_nulls()) != {date} or set(
        frame["ValueCode"].cast(pl.String).drop_nulls()
    ) != {value_code}:
        raise ValueError(f"{source} identity differs from {date}/{value_code}")


def _validate_partition_key(date: str, value_code: str) -> None:
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"invalid YYYYMMDD partition date: {date}")
    if not _SAFE_KEY.fullmatch(value_code):
        raise ValueError(f"unsafe ValueCode partition key: {value_code}")


def _validate_distinct_roots(*roots: Path) -> None:
    resolved = [Path(root).resolve() for root in roots]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    "entry, exit-maker and overnight roots must be disjoint"
                )


def _frame_sha256(frame: pl.DataFrame) -> str:
    columns = sorted(frame.columns)
    selected = frame.select(columns)
    if columns and not selected.is_empty():
        try:
            selected = selected.sort(columns, nulls_last=True)
        except Exception:
            # Calendar and settlement inputs are expected to be scalar types;
            # retain deterministic caller order if an extension type cannot sort.
            pass
    return hashlib.sha256(selected.write_json().encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
