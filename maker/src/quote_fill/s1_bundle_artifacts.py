"""Atomic, streaming JSON artifacts for one S1 policy/date partition.

Accounting and capacity rows may contain tens of millions of records. They are
written and verified as deterministic gzip JSONL streams without materializing
the stream. Small summaries, checkpoints, and carry records use canonical JSON.
Domain decoding remains owned by the accounting/capacity modules.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path

BUNDLE_SCHEMA_VERSION = "s1_policy_date_bundle_v1"
BUNDLE_RUNNER_VERSION = "s1_bundle_artifacts_v2"

_JSON_OBJECT_ARTIFACTS = frozenset(
    {
        "partition_identity.json",
        "daily_summary.json",
        "daily_risk_summary.json",
        "daily_diagnostics.json",
        "compact_checkpoint.json",
    }
)
_JSON_ARRAY_ARTIFACTS = frozenset({"carry.json", "carry_bindings.json"})
_GZIP_JSONL_ARTIFACTS = frozenset(
    {"accounting_facts.jsonl.gz", "capacity_transitions.jsonl.gz"}
)
_PAYLOAD_ARTIFACT_NAMES = frozenset(
    {*_JSON_OBJECT_ARTIFACTS, *_JSON_ARRAY_ARTIFACTS, *_GZIP_JSONL_ARTIFACTS}
)
_PARTITION_FILE_NAMES = frozenset({*_PAYLOAD_ARTIFACT_NAMES, "complete.json"})
_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "runner_version",
        "complete",
        "identity",
        "artifacts",
        "marker_payload_sha256",
    }
)
_ARTIFACT_METADATA_KEYS = frozenset({"bytes", "sha256", "format", "record_count"})
_READ_CHUNK_BYTES = 1024 * 1024


class S1BundleArtifactError(ValueError):
    """An S1 bundle is malformed, drifted, or bound to another lineage."""


@dataclass(frozen=True, slots=True)
class S1GzipJsonlRecords:
    """Verified lazy access to one potentially very large raw-record stream."""

    path: Path
    record_count: int
    sha256: str
    byte_count: int

    def __iter__(self) -> Iterator[dict[str, object]]:
        count = 0
        for record in _iter_canonical_gzip_jsonl(self.path):
            count += 1
            yield record
        if count != self.record_count:
            raise S1BundleArtifactError(
                f"gzip JSONL record-count drift while reading: {self.path}"
            )


@dataclass(frozen=True, slots=True)
class S1BundlePartitionRecords:
    date: str
    policy_id: str
    run_config_fingerprint: str
    run_config_sha256: str
    lineage: dict[str, object]
    daily_summary: dict[str, object]
    daily_risk_summary: dict[str, object]
    daily_diagnostics: dict[str, object]
    accounting_fact_records: S1GzipJsonlRecords
    capacity_transition_records: S1GzipJsonlRecords
    compact_checkpoint_record: dict[str, object]
    carry_records: tuple[dict[str, object], ...]
    carry_binding_records: tuple[dict[str, object], ...]
    complete_marker: dict[str, object]


def write_s1_bundle_partition(
    partition: Path,
    *,
    run_config_fingerprint: str,
    run_config_sha256: str,
    date: str,
    policy_id: str,
    lineage: Mapping[str, object],
    daily_summary: Mapping[str, object],
    daily_risk_summary: Mapping[str, object],
    daily_diagnostics: Mapping[str, object],
    accounting_fact_records: Iterable[Mapping[str, object]],
    capacity_transition_records: Iterable[Mapping[str, object]],
    compact_checkpoint_record: Mapping[str, object],
    carry_records: Sequence[Mapping[str, object]],
    carry_binding_records: Sequence[Mapping[str, object]],
) -> S1BundlePartitionRecords:
    """Stream, fsync, atomically publish, and verify one complete partition."""

    destination = Path(partition)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    identity = _partition_identity(
        run_config_fingerprint=run_config_fingerprint,
        run_config_sha256=run_config_sha256,
        date=date,
        policy_id=policy_id,
        lineage=lineage,
    )
    daily = _bound_summary(
        daily_summary,
        name="daily_summary",
        date=str(identity["date"]),
        policy_id=str(identity["policy_id"]),
    )
    daily_risk = _bound_summary(
        daily_risk_summary,
        name="daily_risk_summary",
        date=str(identity["date"]),
        policy_id=str(identity["policy_id"]),
    )
    diagnostics = _bound_summary(
        daily_diagnostics,
        name="daily_diagnostics",
        date=str(identity["date"]),
        policy_id=str(identity["policy_id"]),
    )
    compact_checkpoint = _json_object(
        compact_checkpoint_record, "compact_checkpoint_record"
    )
    carry = _json_object_records(carry_records, "carry_records")
    carry_bindings = _json_object_records(
        carry_binding_records, "carry_binding_records"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        artifacts: dict[str, dict[str, object]] = {}
        small_artifacts: dict[str, object] = {
            "partition_identity.json": identity,
            "daily_summary.json": daily,
            "daily_risk_summary.json": daily_risk,
            "daily_diagnostics.json": diagnostics,
            "compact_checkpoint.json": compact_checkpoint,
            "carry.json": list(carry),
            "carry_bindings.json": list(carry_bindings),
        }
        for name, value in small_artifacts.items():
            path = temporary / name
            _write_fsynced(path, _canonical_json_bytes(value))
            artifacts[name] = _artifact_metadata(
                path,
                artifact_format="json",
                record_count=(len(value) if isinstance(value, list) else 1),
            )
        artifacts["accounting_facts.jsonl.gz"] = _write_gzip_jsonl_stream(
            temporary / "accounting_facts.jsonl.gz",
            accounting_fact_records,
            name="accounting_fact_records",
        )
        artifacts["capacity_transitions.jsonl.gz"] = _write_gzip_jsonl_stream(
            temporary / "capacity_transitions.jsonl.gz",
            capacity_transition_records,
            name="capacity_transition_records",
        )
        marker_payload: dict[str, object] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "runner_version": BUNDLE_RUNNER_VERSION,
            "complete": True,
            "identity": identity,
            "artifacts": {name: artifacts[name] for name in sorted(artifacts)},
        }
        marker_payload["marker_payload_sha256"] = _canonical_sha256(marker_payload)
        _write_fsynced(
            temporary / "complete.json", _canonical_json_bytes(marker_payload)
        )
        _fsync_directory(temporary)
        _read_verified_partition(temporary, expected_identity=identity)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return _read_verified_partition(destination, expected_identity=identity)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_s1_bundle_partition(
    partition: Path,
    *,
    expected_run_config_fingerprint: str,
    expected_run_config_sha256: str,
    expected_date: str,
    expected_policy_id: str,
    expected_lineage: Mapping[str, object],
) -> S1BundlePartitionRecords:
    """Streaming-verify a partition and return lazy decoded raw-record access."""

    expected_identity = _partition_identity(
        run_config_fingerprint=expected_run_config_fingerprint,
        run_config_sha256=expected_run_config_sha256,
        date=expected_date,
        policy_id=expected_policy_id,
        lineage=expected_lineage,
    )
    return _read_verified_partition(
        Path(partition), expected_identity=expected_identity
    )


def _read_verified_partition(
    partition: Path,
    *,
    expected_identity: dict[str, object],
) -> S1BundlePartitionRecords:
    root = Path(partition)
    _verify_exact_files(root)
    marker = _read_canonical_json_object(root / "complete.json")
    if set(marker) != _MARKER_KEYS:
        raise S1BundleArtifactError("complete marker schema mismatch")
    marker_digest = marker["marker_payload_sha256"]
    unsigned_marker = dict(marker)
    unsigned_marker.pop("marker_payload_sha256")
    if marker_digest != _canonical_sha256(unsigned_marker):
        raise S1BundleArtifactError("complete marker self-hash mismatch")
    if marker["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise S1BundleArtifactError("bundle schema version mismatch")
    if marker["runner_version"] != BUNDLE_RUNNER_VERSION:
        raise S1BundleArtifactError("bundle runner version mismatch")
    if marker["complete"] is not True:
        raise S1BundleArtifactError("bundle is not complete")
    if marker["identity"] != expected_identity:
        raise S1BundleArtifactError("partition identity or lineage mismatch")
    identity = _read_canonical_json_object(root / "partition_identity.json")
    if identity != expected_identity:
        raise S1BundleArtifactError("identity artifact or lineage mismatch")
    metadata = marker["artifacts"]
    if not isinstance(metadata, Mapping) or set(metadata) != _PAYLOAD_ARTIFACT_NAMES:
        raise S1BundleArtifactError("complete marker artifact set mismatch")

    streams: dict[str, S1GzipJsonlRecords] = {}
    for name in sorted(_PAYLOAD_ARTIFACT_NAMES):
        expected = _artifact_metadata_record(metadata[name], name=name)
        path = root / name
        actual_bytes, actual_sha = _hash_file(path)
        if expected["bytes"] != actual_bytes or expected["sha256"] != actual_sha:
            raise S1BundleArtifactError(f"bundle artifact content drift: {name}")
        expected_format = "gzip_jsonl" if name in _GZIP_JSONL_ARTIFACTS else "json"
        if expected["format"] != expected_format:
            raise S1BundleArtifactError(f"bundle artifact format drift: {name}")
        if (
            expected_format == "json"
            and name in _JSON_OBJECT_ARTIFACTS
            and expected["record_count"] != 1
        ):
            raise S1BundleArtifactError(f"bundle artifact record-count drift: {name}")
        if expected_format == "gzip_jsonl":
            count = _verify_gzip_jsonl(path)
            if count != expected["record_count"]:
                raise S1BundleArtifactError(
                    f"bundle artifact record-count drift: {name}"
                )
            streams[name] = S1GzipJsonlRecords(
                path=path,
                record_count=count,
                sha256=actual_sha,
                byte_count=actual_bytes,
            )

    daily = _read_canonical_json_object(root / "daily_summary.json")
    daily_risk = _read_canonical_json_object(root / "daily_risk_summary.json")
    diagnostics = _read_canonical_json_object(root / "daily_diagnostics.json")
    expected_date = str(expected_identity["date"])
    expected_policy = str(expected_identity["policy_id"])
    _assert_summary_binding(
        daily, name="daily_summary", date=expected_date, policy_id=expected_policy
    )
    _assert_summary_binding(
        daily_risk,
        name="daily_risk_summary",
        date=expected_date,
        policy_id=expected_policy,
    )
    _assert_summary_binding(
        diagnostics,
        name="daily_diagnostics",
        date=expected_date,
        policy_id=expected_policy,
    )
    carry = _read_canonical_json_array_objects(root / "carry.json")
    carry_bindings = _read_canonical_json_array_objects(root / "carry_bindings.json")
    for name, count in (
        ("carry.json", len(carry)),
        ("carry_bindings.json", len(carry_bindings)),
    ):
        if (
            _artifact_metadata_record(metadata[name], name=name)["record_count"]
            != count
        ):
            raise S1BundleArtifactError(f"bundle artifact record-count drift: {name}")
    return S1BundlePartitionRecords(
        date=expected_date,
        policy_id=expected_policy,
        run_config_fingerprint=str(expected_identity["run_config_fingerprint"]),
        run_config_sha256=str(expected_identity["run_config_sha256"]),
        lineage=dict(expected_identity["lineage"]),
        daily_summary=daily,
        daily_risk_summary=daily_risk,
        daily_diagnostics=diagnostics,
        accounting_fact_records=streams["accounting_facts.jsonl.gz"],
        capacity_transition_records=streams["capacity_transitions.jsonl.gz"],
        compact_checkpoint_record=_read_canonical_json_object(
            root / "compact_checkpoint.json"
        ),
        carry_records=carry,
        carry_binding_records=carry_bindings,
        complete_marker=marker,
    )


def _write_gzip_jsonl_stream(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    name: str,
) -> dict[str, object]:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
        raise TypeError(f"{name} must be an iterable of JSON objects")
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        ) as compressed:
            for index, value in enumerate(records):
                compressed.write(
                    _canonical_json_bytes(_json_object(value, f"{name}[{index}]"))
                )
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    return _artifact_metadata(path, artifact_format="gzip_jsonl", record_count=count)


def _verify_gzip_jsonl(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(10)
    if len(header) < 10 or header[:2] != b"\x1f\x8b" or header[4:8] != b"\0\0\0\0":
        raise S1BundleArtifactError(f"gzip JSONL header is not deterministic: {path}")
    return sum(1 for _record in _iter_canonical_gzip_jsonl(path))


def _iter_canonical_gzip_jsonl(path: Path) -> Iterator[dict[str, object]]:
    try:
        with gzip.open(path, "rb") as stream:
            for index, line in enumerate(stream):
                if not line.endswith(b"\n"):
                    raise S1BundleArtifactError(
                        f"gzip JSONL record lacks newline at {index}: {path}"
                    )
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise S1BundleArtifactError(
                        f"invalid gzip JSONL record {index}: {path}"
                    ) from error
                record = _json_object(value, f"{path.name}[{index}]")
                if line != _canonical_json_bytes(record):
                    raise S1BundleArtifactError(
                        f"non-canonical gzip JSONL record {index}: {path}"
                    )
                yield record
    except (OSError, EOFError) as error:
        raise S1BundleArtifactError(f"invalid gzip JSONL artifact: {path}") from error


def _artifact_metadata(
    path: Path,
    *,
    artifact_format: str,
    record_count: int,
) -> dict[str, object]:
    byte_count, digest = _hash_file(path)
    return {
        "bytes": byte_count,
        "sha256": digest,
        "format": artifact_format,
        "record_count": record_count,
    }


def _artifact_metadata_record(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_METADATA_KEYS:
        raise S1BundleArtifactError(f"artifact metadata schema mismatch: {name}")
    result = dict(value)
    if type(result["bytes"]) is not int or result["bytes"] < 0:
        raise S1BundleArtifactError(f"artifact bytes metadata invalid: {name}")
    if type(result["record_count"]) is not int or result["record_count"] < 0:
        raise S1BundleArtifactError(f"artifact row-count metadata invalid: {name}")
    _sha256(result["sha256"], f"artifact sha256 {name}")
    if result["format"] not in ("json", "gzip_jsonl"):
        raise S1BundleArtifactError(f"artifact format metadata invalid: {name}")
    return result


def _verify_exact_files(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise S1BundleArtifactError("bundle partition must be a real directory")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != _PARTITION_FILE_NAMES:
        raise S1BundleArtifactError("bundle artifact set mismatch")
    for path in entries.values():
        if path.is_symlink():
            raise S1BundleArtifactError(
                f"bundle artifact must not be a symlink: {path}"
            )
        if not path.is_file():
            raise S1BundleArtifactError(f"bundle artifact must be a file: {path}")


def _partition_identity(
    *,
    run_config_fingerprint: str,
    run_config_sha256: str,
    date: str,
    policy_id: str,
    lineage: Mapping[str, object],
) -> dict[str, object]:
    selected_lineage = _json_object(lineage, "lineage")
    if not selected_lineage:
        raise S1BundleArtifactError("lineage cannot be empty")
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "date": _yyyymmdd(date),
        "policy_id": _identifier(policy_id, "policy_id"),
        "run_config_fingerprint": _sha256(
            run_config_fingerprint, "run_config_fingerprint"
        ),
        "run_config_sha256": _sha256(run_config_sha256, "run_config_sha256"),
        "lineage": selected_lineage,
    }


def _bound_summary(
    value: Mapping[str, object],
    *,
    name: str,
    date: str,
    policy_id: str,
) -> dict[str, object]:
    result = _json_object(value, name)
    _assert_summary_binding(result, name=name, date=date, policy_id=policy_id)
    return result


def _assert_summary_binding(
    value: Mapping[str, object],
    *,
    name: str,
    date: str,
    policy_id: str,
) -> None:
    if value.get("date") != date:
        raise S1BundleArtifactError(f"{name} date mismatch")
    if value.get("policy_id") != policy_id:
        raise S1BundleArtifactError(f"{name} policy_id mismatch")


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _read_canonical_json_object(path: Path) -> dict[str, object]:
    value = _read_canonical_json(path)
    if not isinstance(value, dict):
        raise S1BundleArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _read_canonical_json_array_objects(
    path: Path,
) -> tuple[dict[str, object], ...]:
    value = _read_canonical_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise S1BundleArtifactError(
            f"JSON artifact must be an array of objects: {path}"
        )
    return tuple(value)


def _read_canonical_json(path: Path) -> object:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S1BundleArtifactError(f"invalid JSON artifact: {path}") from error
    result = _json_value(value, path.name)
    if payload != _canonical_json_bytes(result):
        raise S1BundleArtifactError(f"non-canonical JSON artifact: {path}")
    return result


def _json_object(value: object, name: str) -> dict[str, object]:
    result = _json_value(value, name)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _json_object_records(
    values: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of JSON objects")
    return tuple(
        _json_object(value, f"{name}[{index}]") for index, value in enumerate(values)
    )


def _json_value(value: object, path: str) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise S1BundleArtifactError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string JSON object key")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains a non-JSON-safe value")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise S1BundleArtifactError(f"{name} must be a canonical nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    result = _identifier(value, name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise S1BundleArtifactError(f"{name} must be a lowercase SHA-256")
    return result


def _yyyymmdd(value: object) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise S1BundleArtifactError("date must be YYYYMMDD")
    try:
        calendar_date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise S1BundleArtifactError("date must be a valid YYYYMMDD") from error
    return value


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "S1BundleArtifactError",
    "S1BundlePartitionRecords",
    "S1GzipJsonlRecords",
    "read_s1_bundle_partition",
    "write_s1_bundle_partition",
]
