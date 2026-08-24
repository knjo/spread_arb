"""One-shot provenance-only migration for the canonical dynamic hedge bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

MIGRATION_VERSION = "dynamic_future_hedge_provenance_v2_20260822"
DEFAULT_ROOT = Path(
    "maker/data/walkforward/dynamic_future_hedge_causal_v1_20260822"
)
PROVENANCE_COLUMNS = {
    "future_asof_backend",
    "raw_receive_time_regression_detected",
    "regression_check_performed",
}
EXPECTED_LEGACY_FACT_COUNTS = {
    (None, None): 3_622,
    ("pyarrow_sorted_stream", False): 1_235,
    ("polars_global_sort_fallback", True): 2_873,
}
EXPECTED_LEGACY_DAY_COUNTS = {
    (None, None): 35,
    ("pyarrow_sorted_stream", False): 6,
    ("polars_global_sort_fallback", True): 31,
}


def migrate_canonical_dynamic_hedge_bundle(root: Path = DEFAULT_ROOT) -> Path:
    """Build a validated sibling bundle, then recoverably swap it into place."""

    root = Path(root)
    if not (root / "complete.json").is_file():
        raise FileNotFoundError(root / "complete.json")
    original_marker = _read_json(root / "complete.json")
    if original_marker.get("provenance_migration_version") == MIGRATION_VERSION:
        raise ValueError(f"bundle is already migrated: {root}")
    partitions = sorted(
        path for path in root.glob("Date=????????") if path.is_dir()
    )
    if len(partitions) != 72:
        raise ValueError(f"expected 72 day partitions, found {len(partitions)}")

    root_facts_path = root / "future_hedge_facts_all.parquet"
    root_audit_path = root / "daily_audit.parquet"
    for path in (root_facts_path, root_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    root_facts_before = pl.read_parquet(root_facts_path)
    root_audit_before = pl.read_parquet(root_audit_path)
    _assert_legacy_counts(
        root_facts_before, EXPECTED_LEGACY_FACT_COUNTS, "root facts"
    )
    _assert_legacy_counts(
        root_audit_before, EXPECTED_LEGACY_DAY_COUNTS, "root daily audit"
    )
    root_facts_after = migrate_legacy_provenance(root_facts_before)
    root_audit_after = migrate_legacy_provenance(root_audit_before)
    root_checksums = _verify_immutable(root_facts_before, root_facts_after)
    audit_checksums = _verify_immutable(root_audit_before, root_audit_after)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.provenance.", dir=root.parent)
    )
    day_reports: list[dict[str, object]] = []
    migrated_partition_facts: list[pl.DataFrame] = []
    migrated_partition_audits: list[pl.DataFrame] = []
    try:
        for partition in partitions:
            date = partition.name.removeprefix("Date=")
            source_fact_path = partition / "future_hedge_facts.parquet"
            source_audit_path = partition / "audit.parquet"
            source_marker_path = partition / "complete.json"
            for path in (source_fact_path, source_audit_path, source_marker_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            facts_before = pl.read_parquet(source_fact_path)
            audit_before = pl.read_parquet(source_audit_path)
            facts_after = migrate_legacy_provenance(facts_before).select(
                root_facts_after.columns
            )
            audit_after = migrate_legacy_provenance(audit_before).select(
                root_audit_after.columns
            )
            fact_checksums = _verify_immutable(facts_before, facts_after)
            day_audit_checksums = _verify_immutable(audit_before, audit_after)
            _assert_one_day_provenance(date, facts_after, audit_after)

            destination = temporary / partition.name
            destination.mkdir(parents=True)
            fact_path = destination / "future_hedge_facts.parquet"
            audit_path = destination / "audit.parquet"
            facts_after.write_parquet(fact_path)
            audit_after.write_parquet(audit_path)
            provenance = _one_provenance_record(audit_after)
            day_marker = {
                "schema_version": "dynamic_future_hedge_day_v2_provenance",
                "Date": date,
                "analysis_only": True,
                "pathwise_ev_ready": False,
                "provenance_migration_version": MIGRATION_VERSION,
                **provenance,
                "future_hedge_facts_sha256": _sha256(fact_path),
                "audit_sha256": _sha256(audit_path),
                "immutable_logical_sha256": fact_checksums["immutable_after"],
                "price_state_slippage_logical_sha256": fact_checksums[
                    "price_state_slippage_after"
                ],
                "source_complete_json_sha256": _sha256(source_marker_path),
            }
            _write_json(destination / "complete.json", day_marker)
            migrated_partition_facts.append(facts_after)
            migrated_partition_audits.append(audit_after)
            day_reports.append(
                {
                    "Date": date,
                    **provenance,
                    **fact_checksums,
                    "audit_immutable_before": day_audit_checksums[
                        "immutable_before"
                    ],
                    "audit_immutable_after": day_audit_checksums[
                        "immutable_after"
                    ],
                    "fact_rows": facts_after.height,
                }
            )

        _assert_partition_aggregate_identity(
            root_facts_after,
            migrated_partition_facts,
            identity="facts",
        )
        _assert_partition_aggregate_identity(
            root_audit_after,
            migrated_partition_audits,
            identity="daily audit",
        )
        root_facts_after.write_parquet(
            temporary / "future_hedge_facts_all.parquet"
        )
        root_audit_after.write_parquet(temporary / "daily_audit.parquet")
        for name in ("hedge_summary.csv", "status_summary.csv"):
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, temporary / name)

        fact_provenance_counts = _provenance_counts(root_facts_after)
        day_provenance_counts = _provenance_counts(root_audit_after)
        migration_report = {
            "schema_version": "dynamic_future_hedge_provenance_migration_v1",
            "provenance_migration_version": MIGRATION_VERSION,
            "source_root": str(root),
            "facts": root_checksums,
            "daily_audit": audit_checksums,
            "fact_provenance_counts": fact_provenance_counts,
            "day_provenance_counts": day_provenance_counts,
            "price_state_slippage_unchanged": (
                root_checksums["price_state_slippage_before"]
                == root_checksums["price_state_slippage_after"]
            ),
            "all_nonprovenance_columns_unchanged": (
                root_checksums["immutable_before"]
                == root_checksums["immutable_after"]
            ),
            "day_reports": day_reports,
        }
        report_path = temporary / "provenance_migration.json"
        _write_json(report_path, migration_report)

        marker = dict(original_marker)
        marker.update(
            {
                "schema_version": "dynamic_future_hedge_bundle_v2_provenance",
                "provenance_migration_version": MIGRATION_VERSION,
                "future_asof_fact_provenance_counts": fact_provenance_counts,
                "future_asof_day_provenance_counts": day_provenance_counts,
                "price_state_slippage_logical_sha256": root_checksums[
                    "price_state_slippage_after"
                ],
                "all_nonprovenance_logical_sha256": root_checksums[
                    "immutable_after"
                ],
            }
        )
        marker["artifacts"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "complete.json"
        }
        _write_json(temporary / "complete.json", marker)
        _validate_migrated_root(temporary)

        backup = root.with_name(f"{root.name}_pre_provenance_v2_20260822")
        if backup.exists():
            raise FileExistsError(backup)
        root.rename(backup)
        try:
            temporary.rename(root)
        except BaseException:
            backup.rename(root)
            raise
        return backup
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def migrate_legacy_provenance(frame: pl.DataFrame) -> pl.DataFrame:
    """Map only the three known canonical legacy provenance states."""

    if "future_asof_backend" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.String).alias("future_asof_backend")
        )
    if "raw_receive_time_regression_detected" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias(
                "raw_receive_time_regression_detected"
            )
        )
    backend = pl.col("future_asof_backend")
    regression = pl.col("raw_receive_time_regression_detected")
    old_null_stream = backend.is_null() & regression.is_null()
    old_marked_stream = (backend == "pyarrow_sorted_stream") & (
        regression == False
    )
    old_mislabeled_forced = (backend == "polars_global_sort_fallback") & (
        regression == True
    )
    known = old_null_stream | old_marked_stream | old_mislabeled_forced
    if frame.filter(~known.fill_null(False)).height:
        values = frame.select(
            "future_asof_backend", "raw_receive_time_regression_detected"
        ).unique()
        raise ValueError(f"unknown legacy provenance states: {values.to_dicts()}")
    return frame.with_columns(
        pl.when(old_null_stream)
        .then(pl.lit("pyarrow_sorted_stream"))
        .when(old_mislabeled_forced)
        .then(pl.lit("polars_global_sort_forced"))
        .otherwise(backend)
        .alias("future_asof_backend"),
        pl.when(known)
        .then(False)
        .otherwise(regression)
        .cast(pl.Boolean)
        .alias("raw_receive_time_regression_detected"),
        pl.when(old_mislabeled_forced)
        .then(False)
        .when(old_null_stream | old_marked_stream)
        .then(True)
        .otherwise(None)
        .cast(pl.Boolean)
        .alias("regression_check_performed"),
    )


def _assert_legacy_counts(
    frame: pl.DataFrame,
    expected: Mapping[tuple[str | None, bool | None], int],
    identity: str,
) -> None:
    actual = {
        (row["future_asof_backend"], row["raw_receive_time_regression_detected"]): int(
            row["len"]
        )
        for row in frame.group_by(
            "future_asof_backend", "raw_receive_time_regression_detected"
        )
        .len()
        .iter_rows(named=True)
    }
    if actual != dict(expected):
        raise ValueError(f"{identity} legacy provenance mismatch: {actual}")


def _verify_immutable(
    before: pl.DataFrame, after: pl.DataFrame
) -> dict[str, str]:
    immutable_columns = [
        column for column in before.columns if column not in PROVENANCE_COLUMNS
    ]
    if immutable_columns != [
        column for column in after.columns if column not in PROVENANCE_COLUMNS
    ]:
        raise ValueError("non-provenance schema changed during migration")
    before_immutable = before.select(immutable_columns)
    after_immutable = after.select(immutable_columns)
    if not before_immutable.equals(after_immutable, null_equal=True):
        raise ValueError("non-provenance values changed during migration")
    price_columns = [
        column
        for column in immutable_columns
        if "price" in column.lower()
        or "slippage" in column.lower()
        or column
        in {
            "status",
            "available_quantity",
            "executed_quantity",
            "depth_shortfall",
            "levels_swept",
            "decision_book_age_ms",
            "arrival_book_age_ms",
        }
    ]
    return {
        "immutable_before": _logical_checksum(before_immutable),
        "immutable_after": _logical_checksum(after_immutable),
        "price_state_slippage_before": _logical_checksum(
            before.select(price_columns)
        ),
        "price_state_slippage_after": _logical_checksum(after.select(price_columns)),
    }


def _logical_checksum(frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _assert_one_day_provenance(
    date: str, facts: pl.DataFrame, audit: pl.DataFrame
) -> None:
    fact_values = facts.select(sorted(PROVENANCE_COLUMNS)).unique()
    audit_values = audit.select(sorted(PROVENANCE_COLUMNS)).unique()
    if fact_values.height != 1 or audit_values.height != 1:
        raise ValueError(f"{date}: provenance is not constant within day")
    if not fact_values.equals(audit_values, null_equal=True):
        raise ValueError(f"{date}: fact/audit provenance differs")


def _one_provenance_record(frame: pl.DataFrame) -> dict[str, object]:
    row = frame.select(sorted(PROVENANCE_COLUMNS)).unique().row(0, named=True)
    return {str(key): value for key, value in row.items()}


def _assert_partition_aggregate_identity(
    aggregate: pl.DataFrame,
    partitions: Sequence[pl.DataFrame],
    *,
    identity: str,
) -> None:
    combined = pl.concat(partitions, how="vertical")
    if not aggregate.equals(combined, null_equal=True):
        raise ValueError(f"root {identity} differs from ordered day partitions")


def _provenance_counts(frame: pl.DataFrame) -> list[dict[str, object]]:
    return (
        frame.group_by(sorted(PROVENANCE_COLUMNS))
        .len()
        .sort(sorted(PROVENANCE_COLUMNS))
        .to_dicts()
    )


def _validate_migrated_root(root: Path) -> None:
    facts = pl.read_parquet(root / "future_hedge_facts_all.parquet")
    audits = pl.read_parquet(root / "daily_audit.parquet")
    if facts.height != 7_730 or audits.height != 72:
        raise ValueError("migrated aggregate cardinality changed")
    if facts["future_asof_backend"].null_count() or audits[
        "future_asof_backend"
    ].null_count():
        raise ValueError("migrated provenance still has null backend")
    marker = _read_json(root / "complete.json")
    for name, metadata in marker["artifacts"].items():
        path = root / name
        if int(metadata["bytes"]) != path.stat().st_size:
            raise ValueError(f"artifact size mismatch: {path}")
        if str(metadata["sha256"]) != _sha256(path):
            raise ValueError(f"artifact checksum mismatch: {path}")
    for partition in root.glob("Date=????????"):
        marker = _read_json(partition / "complete.json")
        if marker["future_hedge_facts_sha256"] != _sha256(
            partition / "future_hedge_facts.parquet"
        ):
            raise ValueError(f"partition fact checksum mismatch: {partition}")
        if marker["audit_sha256"] != _sha256(partition / "audit.parquet"):
            raise ValueError(f"partition audit checksum mismatch: {partition}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    backup = migrate_canonical_dynamic_hedge_bundle(args.root)
    print(backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
