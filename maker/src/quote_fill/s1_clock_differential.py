"""Exact real-day differential gate for the S1 entry raw-book clock.

The production route uses the entry-quote-derived raw clock.  This diagnostic
runner supports generic-versus-sparse-derived A/B and full-day sparse-derived-
versus-true-full-1Hz-derived B/C gates.  It checkpoints both material outputs
and requires exact component equality.  It is intentionally not part of
routine unit tests because it reads the real development corpus.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Final, Literal
from unittest.mock import patch

import polars as pl

from .capacity_ledger import CapacityLedger
from .s1_accounting import encode_accounting_facts
from .s1_accounting_bridge import S1AccountingBridge, S1AccountingProduct
from .s1_day_state import build_s1_policy_day_state, materialize_s1_common_day
from .s1_entry_day_runner import (
    CAUSAL_INPUT_COLUMNS,
    RUNNER_VERSION,
    PreparedS1EntryDay,
    prepare_s1_entry_day,
    run_s1_entry_policy,
)
from .s1_event_loop import encode_s1_carry_position
from .s1_raw_book_adapter import RawBookRiskAdapter

CHECKPOINT_SCHEMA: Final = "s1-clock-differential-material-v2"
DIFFERENTIAL_RUNNER_VERSION: Final = "s1_clock_differential_v1"
MAKER_ROOT: Final = Path(__file__).resolve().parents[2]
ENTRY_START_SECOND: Final = 300
SESSION_SECONDS: Final = 14_400
ClockMode = Literal["generic_effective", "entry_route_derived"]
CheckpointSide = Literal[
    "generic_effective",
    "entry_route_derived",
    "full_1hz_entry_route_derived",
]
CheckpointRole = Literal[
    "generic_effective",
    "sparse_entry_route_derived",
    "full_1hz_entry_route_derived",
]


class S1ClockDifferentialError(RuntimeError):
    """The differential runner input or checkpoint is invalid."""


class S1ClockDifferentialMismatch(AssertionError):
    """The generic and derived clocks produced different material output."""

    def __init__(self, report: Mapping[str, object]) -> None:
        self.report = dict(report)
        super().__init__(json.dumps(self.report, sort_keys=True))


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    date: str
    policy_id: str
    horizon_minutes: int | None
    enabled_product_ids: tuple[str, ...]
    source_commit: str

    def __post_init__(self) -> None:
        _validated_source_commit(self.source_commit, "source_commit")

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _validated_source_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise ValueError(f"{name} must be a 40- or 64-character commit hash")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a hexadecimal commit hash")
    return normalized


def _git_source_commit(expected: str | None = None) -> str:
    """Return nested maker HEAD only when its source tree is clean."""

    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "src",
        ),
        cwd=MAKER_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise S1ClockDifferentialError("failed to inspect maker source status")
    if status.stdout.strip():
        raise S1ClockDifferentialError(
            "maker/src must be clean before a durable differential replay"
        )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=MAKER_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        raise S1ClockDifferentialError("failed to resolve maker source commit")
    actual = _validated_source_commit(revision.stdout.strip(), "maker HEAD")
    if expected is not None:
        asserted = _validated_source_commit(expected, "--source-commit")
        if actual != asserted:
            raise S1ClockDifferentialError("--source-commit differs from maker HEAD")
    return actual


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _component_metadata(material: Mapping[str, object]) -> dict[str, object]:
    return {
        name: {
            "count": len(value) if hasattr(value, "__len__") else None,
            "sha256": _canonical_sha256(value),
        }
        for name, value in sorted(material.items())
    }


def _first_difference(
    left: object,
    right: object,
    path: str = "$",
) -> dict[str, object] | None:
    if type(left) is not type(right):
        return {
            "path": path,
            "reason": "type",
            "left": type(left).__name__,
            "right": type(right).__name__,
        }
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return {
                "path": path,
                "reason": "keys",
                "left_only": sorted(left_keys - right_keys),
                "right_only": sorted(right_keys - left_keys),
            }
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {
                "path": path,
                "reason": "length",
                "left": len(left),
                "right": len(right),
            }
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_value,
                right_value,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if _canonical_json_bytes(left) != _canonical_json_bytes(right):
        return {
            "path": path,
            "reason": "value",
            "left": repr(left)[:500],
            "right": repr(right)[:500],
        }
    return None


def assert_exact_material_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Require byte-exact canonical JSON equality for every material component."""

    left_names = set(left)
    right_names = set(right)
    if left_names != right_names:
        raise S1ClockDifferentialMismatch(
            {
                "reason": "component_keys",
                "left_only": sorted(left_names - right_names),
                "right_only": sorted(right_names - left_names),
            }
        )
    assertions = tuple(
        {
            "component": name,
            "count": len(left[name]) if hasattr(left[name], "__len__") else None,
            "sha256": _canonical_sha256(left[name]),
            "exact": (
                _canonical_json_bytes(left[name]) == _canonical_json_bytes(right[name])
            ),
        }
        for name in sorted(left)
    )
    mismatched = [row for row in assertions if row["exact"] is not True]
    if mismatched:
        component = str(mismatched[0]["component"])
        raise S1ClockDifferentialMismatch(
            {
                "reason": "material_component_mismatch",
                "mismatched_components": [row["component"] for row in mismatched],
                "component_assertions": [
                    {
                        "component": name,
                        "left_count": (
                            len(left[name]) if hasattr(left[name], "__len__") else None
                        ),
                        "right_count": (
                            len(right[name])
                            if hasattr(right[name], "__len__")
                            else None
                        ),
                        "left_sha256": _canonical_sha256(left[name]),
                        "right_sha256": _canonical_sha256(right[name]),
                    }
                    for name in sorted(left)
                ],
                "first_mismatched_component": component,
                "first_difference": _first_difference(
                    left[component], right[component]
                ),
            }
        )
    return assertions


def write_material_checkpoint(
    path: Path,
    *,
    label: str,
    role: CheckpointRole,
    identity: CheckpointIdentity,
    material: Mapping[str, object],
) -> dict[str, object]:
    """Atomically write deterministic gzip JSON and return its metadata."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": CHECKPOINT_SCHEMA,
        "entry_runner_version": RUNNER_VERSION,
        "differential_runner_version": DIFFERENTIAL_RUNNER_VERSION,
        "label": label,
        "role": role,
        "identity": identity.as_record(),
        "components": _component_metadata(material),
        "material": dict(material),
    }
    payload = _canonical_json_bytes(document)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                compressed.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "document_sha256": hashlib.sha256(payload).hexdigest(),
        "components": document["components"],
    }


def load_material_checkpoint(
    path: Path,
    *,
    expected_identity: CheckpointIdentity | None = None,
    expected_role: CheckpointRole | None = None,
) -> dict[str, object]:
    """Load a checkpoint and verify its schema, identity, and component hashes."""

    try:
        with gzip.open(path, "rb") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise S1ClockDifferentialError(f"invalid checkpoint: {path}") from error
    if not isinstance(document, dict):
        raise S1ClockDifferentialError("checkpoint document must be an object")
    if document.get("schema_version") != CHECKPOINT_SCHEMA:
        raise S1ClockDifferentialError("checkpoint schema version differs")
    if document.get("entry_runner_version") != RUNNER_VERSION:
        raise S1ClockDifferentialError("checkpoint entry runner version differs")
    if document.get("differential_runner_version") != DIFFERENTIAL_RUNNER_VERSION:
        raise S1ClockDifferentialError("checkpoint differential runner version differs")
    if expected_role is not None and document.get("role") != expected_role:
        raise S1ClockDifferentialError("checkpoint role differs")
    if expected_identity is not None and _canonical_json_bytes(
        document.get("identity")
    ) != _canonical_json_bytes(expected_identity.as_record()):
        raise S1ClockDifferentialError("checkpoint identity differs")
    material = document.get("material")
    if not isinstance(material, dict):
        raise S1ClockDifferentialError("checkpoint material must be an object")
    if document.get("components") != _component_metadata(material):
        raise S1ClockDifferentialError("checkpoint component metadata differs")
    return material


def _bounded_prepared(
    prepared: PreparedS1EntryDay,
    policy_id: str,
    horizon_minutes: int | None,
) -> PreparedS1EntryDay:
    if horizon_minutes is None:
        return prepared
    maximum = (SESSION_SECONDS - ENTRY_START_SECOND) // 60
    if not 1 <= horizon_minutes <= maximum:
        raise ValueError(f"horizon_minutes must be in [1, {maximum}] or None")
    cutoff_ns = (
        prepared.day_open_time_ns
        + (ENTRY_START_SECOND + horizon_minutes * 60) * 1_000_000_000
    )
    source = prepared.state_changes_by_policy[policy_id]
    bounded = source.filter(pl.col("decision_time_ns") < cutoff_ns)
    if bounded.is_empty():
        raise S1ClockDifferentialError("bounded policy state is empty")
    missing = set(prepared.entry_product_ids).difference(
        bounded["ValueCode"].cast(pl.String).unique().to_list()
    )
    if missing:
        raise S1ClockDifferentialError(
            f"bounded policy state lost products: {sorted(missing)[:10]}"
        )
    return replace(
        prepared,
        state_changes_by_policy={policy_id: bounded},
        entry_cutoff_time_ns=cutoff_ns,
    )


def _full_1hz_prepared(
    prepared: PreparedS1EntryDay,
    policy_id: str,
) -> PreparedS1EntryDay:
    """Rebuild the true full 1 Hz policy state from the causal fair source."""

    specs = prepared.specs_by_policy[policy_id]
    product_keys = pl.DataFrame(
        {
            "Date": [spec.Date for spec in specs],
            "ValueCode": [spec.ValueCode for spec in specs],
            "QuoteCode": [spec.QuoteCode for spec in specs],
        }
    ).unique(maintain_order=True)
    causal_day = (
        pl.scan_parquet(prepared.source_paths["causal_fair"])
        .join(
            product_keys.lazy(),
            on=["Date", "ValueCode", "QuoteCode"],
            how="inner",
            validate="m:1",
        )
        .select(CAUSAL_INPUT_COLUMNS)
        .collect(engine="streaming")
    )
    common = materialize_s1_common_day(causal_day)
    del causal_day
    gc.collect()
    full_state = build_s1_policy_day_state(
        common,
        specs,
        policy_id=policy_id,
        _sparse_only=False,
    )
    del common
    gc.collect()
    if full_state.select("ValueCode", "decision_time_ns").n_unique() != (
        full_state.height
    ):
        raise S1ClockDifferentialError(
            "full 1 Hz policy-state observation cursors are duplicated"
        )
    if full_state.height != prepared.common_decisions.height:
        raise S1ClockDifferentialError(
            "full 1 Hz policy state differs from the common-grid row count"
        )
    return replace(
        prepared,
        state_changes_by_policy={policy_id: full_state},
    )


def _fresh_accounting(
    prepared: PreparedS1EntryDay, policy_id: str
) -> S1AccountingBridge:
    return S1AccountingBridge(
        default_date=prepared.date,
        scenario_id=policy_id,
        products=tuple(
            S1AccountingProduct(
                product_id=product.product_id,
                value_code=product.value_code,
                contract_size_shares=product.contract_size_shares,
            )
            for product in prepared.products
        ),
    )


def _generic_entry_clock(
    self: RawBookRiskAdapter,
    venue: str,
    product_id: str,
    after_cursor: object,
    deadline_ns: int,
) -> object:
    return self.next_change_cursor(venue, product_id, after_cursor, deadline_ns)


def _final_balances(
    prepared: PreparedS1EntryDay,
    ledger: CapacityLedger,
    transitions: Sequence[object],
) -> dict[str, object]:
    admitted_ids = sorted(
        {
            transition.capacity_id
            for transition in transitions
            if transition.admitted is True
        }
    )
    return {
        "global": asdict(ledger.global_balances),
        "products": {
            product.product_id: asdict(ledger.product_balances(product.product_id))
            for product in sorted(prepared.products, key=lambda item: item.product_id)
        },
        "admitted_accounts": {
            capacity_id: asdict(ledger.account_balances(capacity_id))
            for capacity_id in admitted_ids
        },
    }


def _material_output(
    prepared: PreparedS1EntryDay,
    run: object,
    accounting: S1AccountingBridge,
    ledger: CapacityLedger,
) -> dict[str, object]:
    result = run.result
    estimates = []
    for order in result.orders:
        if order.economic_estimate is None:
            raise S1ClockDifferentialError("sent order lacks an economic estimate")
        estimates.append(order.economic_estimate.to_record())
    return {
        "sent_orders": [asdict(value) for value in result.orders],
        "executions": [asdict(value) for value in result.executions],
        "positions": [asdict(value) for value in result.positions],
        "carry_in": [encode_s1_carry_position(value) for value in result.carry_in],
        "carry_out": [encode_s1_carry_position(value) for value in result.carry_out],
        "accounting_facts": encode_accounting_facts(accounting.facts),
        "sent_economic_estimates": estimates,
        "final_capacity_balances": _final_balances(
            prepared, ledger, result.capacity_transitions
        ),
    }


def _run_material(
    prepared: PreparedS1EntryDay,
    policy_id: str,
    *,
    clock_mode: ClockMode,
    enabled_product_ids: frozenset[str],
) -> tuple[dict[str, object], dict[str, object]]:
    accounting = _fresh_accounting(prepared, policy_id)
    ledger = CapacityLedger()
    started = perf_counter()
    clock = (
        patch.object(
            RawBookRiskAdapter,
            "next_entry_quote_change_cursor",
            new=_generic_entry_clock,
        )
        if clock_mode == "generic_effective"
        else _NullContext()
    )
    with clock:
        run = run_s1_entry_policy(
            prepared,
            policy_id,
            normal_exit_enabled=True,
            accounting_adapter=accounting,
            capacity_ledger=ledger,
            entry_enabled_product_ids=enabled_product_ids,
        )
    material = _material_output(prepared, run, accounting, ledger)
    diagnostic = {
        "clock_mode": clock_mode,
        "seconds": perf_counter() - started,
        "components": _component_metadata(material),
    }
    return material, diagnostic


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def _emit(record: Mapping[str, object]) -> None:
    print(_canonical_json_bytes(dict(record)).decode("ascii"), flush=True)


def _checkpoint_name(
    identity: CheckpointIdentity,
    side: CheckpointSide,
) -> str:
    horizon = (
        "full" if identity.horizon_minutes is None else f"{identity.horizon_minutes}m"
    )
    scope = (
        "all"
        if len(identity.enabled_product_ids) > 1
        else identity.enabled_product_ids[0]
    )
    policy = "".join(
        character if character.isalnum() else "_" for character in identity.policy_id
    )
    return f"s1_clock_diff_{identity.date}_{policy}_{horizon}_{scope}_{side}.json.gz"


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        help="bounded entry window from 09:05; omit for the full day",
    )
    parser.add_argument("--product-id", help="enable entries for one product only")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp"),
    )
    parser.add_argument(
        "--resume-generic",
        type=Path,
        help="reuse a verified generic checkpoint instead of replaying A",
    )
    parser.add_argument(
        "--compare-full-1hz",
        action="store_true",
        help=(
            "compare full-day sparse-derived B with true full-1Hz-derived C "
            "instead of generic-versus-derived A/B"
        ),
    )
    parser.add_argument(
        "--resume-sparse-derived",
        type=Path,
        help="reuse a verified sparse-derived B checkpoint for the B/C gate",
    )
    parser.add_argument(
        "--source-commit",
        help="assert the clean nested maker source commit used by the gate",
    )
    return parser.parse_args(arguments)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.compare_full_1hz and args.horizon_minutes is not None:
        raise ValueError("--compare-full-1hz requires a full-day horizon")
    if args.compare_full_1hz and args.resume_generic is not None:
        raise ValueError("--resume-generic is only valid for the A/B gate")
    if not args.compare_full_1hz and args.resume_sparse_derived is not None:
        raise ValueError(
            "--resume-sparse-derived is only valid with --compare-full-1hz"
        )


def _run_full_1hz_gate(
    args: argparse.Namespace,
    prepared: PreparedS1EntryDay,
    identity: CheckpointIdentity,
    enabled: frozenset[str],
) -> int:
    sparse_path = (
        args.resume_sparse_derived
        if args.resume_sparse_derived is not None
        else args.checkpoint_dir / _checkpoint_name(identity, "entry_route_derived")
    )
    if args.resume_sparse_derived is None:
        material_b, diagnostic_b = _run_material(
            prepared,
            args.policy_id,
            clock_mode="entry_route_derived",
            enabled_product_ids=enabled,
        )
        _emit({"event": "replay_complete", "label": "B_sparse_derived", **diagnostic_b})
        checkpoint_b = write_material_checkpoint(
            sparse_path,
            label="B_sparse_entry_route_derived",
            role="sparse_entry_route_derived",
            identity=identity,
            material=material_b,
        )
        _emit({"event": "checkpoint_written", **checkpoint_b})
        del material_b
        gc.collect()
    else:
        load_material_checkpoint(
            sparse_path,
            expected_identity=identity,
            expected_role="sparse_entry_route_derived",
        )
        _emit({"event": "checkpoint_resumed", "path": str(sparse_path.resolve())})

    full_prepared = _full_1hz_prepared(prepared, args.policy_id)
    del prepared
    gc.collect()
    full_path = args.checkpoint_dir / _checkpoint_name(
        identity, "full_1hz_entry_route_derived"
    )
    material_c, diagnostic_c = _run_material(
        full_prepared,
        args.policy_id,
        clock_mode="entry_route_derived",
        enabled_product_ids=enabled,
    )
    _emit({"event": "replay_complete", "label": "C_full_1hz_derived", **diagnostic_c})
    checkpoint_c = write_material_checkpoint(
        full_path,
        label="C_full_1hz_entry_route_derived",
        role="full_1hz_entry_route_derived",
        identity=identity,
        material=material_c,
    )
    _emit({"event": "checkpoint_written", **checkpoint_c})
    del material_c, full_prepared
    gc.collect()

    material_b = load_material_checkpoint(
        sparse_path,
        expected_identity=identity,
        expected_role="sparse_entry_route_derived",
    )
    material_c = load_material_checkpoint(
        full_path,
        expected_identity=identity,
        expected_role="full_1hz_entry_route_derived",
    )
    assertions = assert_exact_material_equal(material_b, material_c)
    _emit(
        {
            "event": "exact_differential",
            "comparison": "sparse_derived_vs_full_1hz_derived",
            "equal": True,
            "identity": identity.as_record(),
            "components": assertions,
        }
    )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    _validate_cli_args(args)
    source_commit = _git_source_commit(args.source_commit)
    prepared = prepare_s1_entry_day(args.date, policy_ids=(args.policy_id,))
    prepared = _bounded_prepared(prepared, args.policy_id, args.horizon_minutes)
    if args.product_id is None:
        enabled = frozenset(prepared.entry_product_ids)
    elif args.product_id not in prepared.entry_product_ids:
        raise ValueError("product_id is not in the prepared day")
    else:
        enabled = frozenset((args.product_id,))
    identity = CheckpointIdentity(
        date=args.date,
        policy_id=args.policy_id,
        horizon_minutes=args.horizon_minutes,
        enabled_product_ids=tuple(sorted(enabled)),
        source_commit=source_commit,
    )
    if args.compare_full_1hz:
        return _run_full_1hz_gate(args, prepared, identity, enabled)

    generic_path = (
        args.resume_generic
        if args.resume_generic is not None
        else args.checkpoint_dir / _checkpoint_name(identity, "generic_effective")
    )
    if args.resume_generic is None:
        material_a, diagnostic_a = _run_material(
            prepared,
            args.policy_id,
            clock_mode="generic_effective",
            enabled_product_ids=enabled,
        )
        _emit({"event": "replay_complete", **diagnostic_a})
        checkpoint_a = write_material_checkpoint(
            generic_path,
            label="A_generic_effective",
            role="generic_effective",
            identity=identity,
            material=material_a,
        )
        _emit({"event": "checkpoint_written", **checkpoint_a})
        del material_a
        gc.collect()
    else:
        load_material_checkpoint(
            generic_path,
            expected_identity=identity,
            expected_role="generic_effective",
        )
        _emit({"event": "checkpoint_resumed", "path": str(generic_path.resolve())})

    derived_path = args.checkpoint_dir / _checkpoint_name(
        identity, "entry_route_derived"
    )
    material_b, diagnostic_b = _run_material(
        prepared,
        args.policy_id,
        clock_mode="entry_route_derived",
        enabled_product_ids=enabled,
    )
    _emit({"event": "replay_complete", **diagnostic_b})
    checkpoint_b = write_material_checkpoint(
        derived_path,
        label="B_entry_route_derived",
        role="sparse_entry_route_derived",
        identity=identity,
        material=material_b,
    )
    _emit({"event": "checkpoint_written", **checkpoint_b})
    del material_b, prepared
    gc.collect()

    material_a = load_material_checkpoint(
        generic_path,
        expected_identity=identity,
        expected_role="generic_effective",
    )
    material_b = load_material_checkpoint(
        derived_path,
        expected_identity=identity,
        expected_role="sparse_entry_route_derived",
    )
    assertions = assert_exact_material_equal(material_a, material_b)
    _emit(
        {
            "event": "exact_differential",
            "equal": True,
            "identity": identity.as_record(),
            "components": assertions,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except S1ClockDifferentialMismatch as error:
        _emit({"event": "exact_differential", "equal": False, **error.report})
        raise SystemExit(1) from None
    except Exception as error:  # noqa: BLE001 - CLI must emit a structured failure.
        _emit(
            {
                "event": "runner_error",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        raise SystemExit(2) from None
