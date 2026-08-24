"""Publish a bounded benchmark of the fixed-q frozen-fact fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import tempfile
import time

import polars as pl

from .compact_frozen_facts import (
    DEFAULT_EXECUTION_ROOT,
    EXPECTED_EXECUTION_MANIFEST_SHA256,
    EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256,
    FROZEN_COMPACT_VERSION,
    SUPPORTED_RANKS,
    load_frozen_compact_product_days,
    summarize_frozen_compact_universe,
)
from .execution_runner import load_walkforward_execution_product_day


_EXPECTED_ARTIFACTS = {
    "full_q_summary.parquet",
    "full_product_day_audit.parquet",
    "benchmark.json",
}
BENCHMARK_VERSION = "compact_frozen_benchmark_v2_source_recomputed"
BUNDLE_SCHEMA_VERSION = "compact_frozen_benchmark_bundle_v2"
_MARKER_KEYS = {
    "schema_version",
    "analysis_only",
    "formal_roots_mutated",
    "pathwise_ev_ready",
    "joint_volume_allocated",
    "full_order_outcomes_persisted",
    "full_summary_only",
    "actual_exit_opportunity_metrics_absent",
    "actual_fifo_metrics_absent",
    "fifo_experimental_contract_only",
    "fifo_capacity_source_bound",
    "benchmark_observations_are_not_semantic_claims",
    "execution_root",
    "execution_manifest_sha256",
    "partition_source_inventory_sha256",
    "selected_Date",
    "selected_ValueCode",
    "implementation_sources",
    "artifacts",
}
_METRICS_KEYS = {
    "benchmark_version",
    "compact_source_version",
    "Date",
    "ValueCode",
    "timing_seconds",
    "selected_product_day",
    "full_universe",
    "supported_exact_parity",
    "source",
    "memory",
    "safety",
}
_TIMING_KEYS = {
    "selected_product_day_fast_path",
    "full_2687_end_to_end_verified_summary",
    "full_2687_marker_hash_semantic_verification",
    "full_4032586_projection_validation_aggregate",
    "raw_product_day_loader_comparison",
    "selected_fast_path_speedup_vs_raw_loader",
    "raw_replay_or_formal_outputs_recomputed",
}
_MEMORY_NOTE = (
    "peak includes optional raw-loader comparison; the isolated fast-path "
    "/usr/bin/time result should be reported separately"
)
_PARITY_FIELDS = (
    "raw_order_fact_id",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "first_fill_recv_time_ns",
    "first_fill_event_sequence",
    "first_fill_row_index",
    "full_fill_recv_time_ns",
    "full_fill_event_sequence",
    "full_fill_row_index",
    "hedge_label_observed",
    "hedge_executable",
    "hedge_decision_time_ns",
    "hedge_executable_vwap_price",
)


def run_benchmark(
    date: str,
    value_code: str,
    *,
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    measure_raw_loader: bool = True,
) -> tuple[dict[str, object], object, object]:
    date = str(date)
    value_code = str(value_code)
    execution_root = Path(execution_root).resolve()
    selected_started = time.perf_counter()
    selected = load_frozen_compact_product_days(
        execution_root,
        dates=[date],
        value_codes=[value_code],
    )
    selected_seconds = time.perf_counter() - selected_started
    universe = summarize_frozen_compact_universe(execution_root)

    partition = (
        execution_root / f"Date={date}" / f"ValueCode={value_code}"
    )
    source_path = partition / "execution_action_facts.parquet"
    source = pl.read_parquet(source_path)
    parity = _supported_parity(selected.order_outcomes, source)
    raw_loader_seconds = None
    if measure_raw_loader:
        started = time.perf_counter()
        raw = load_walkforward_execution_product_day(date, value_code)
        raw_loader_seconds = time.perf_counter() - started
        del raw

    manifest_path = execution_root / "execution_partition_manifest.parquet"
    metrics: dict[str, object] = {
        "benchmark_version": BENCHMARK_VERSION,
        "compact_source_version": FROZEN_COMPACT_VERSION,
        "Date": date,
        "ValueCode": value_code,
        "timing_seconds": {
            "selected_product_day_fast_path": selected_seconds,
            "full_2687_end_to_end_verified_summary": universe.elapsed_seconds,
            "full_2687_marker_hash_semantic_verification": (
                universe.source_verification_seconds
            ),
            "full_4032586_projection_validation_aggregate": (
                universe.projection_validation_aggregate_seconds
            ),
            "raw_product_day_loader_comparison": raw_loader_seconds,
            "selected_fast_path_speedup_vs_raw_loader": (
                raw_loader_seconds / selected_seconds
                if raw_loader_seconds is not None and selected_seconds
                else None
            ),
            "raw_replay_or_formal_outputs_recomputed": False,
        },
        "selected_product_day": _selected_semantics(selected),
        "full_universe": _full_semantics(universe),
        "supported_exact_parity": parity,
        "source": {
            "execution_root": str(execution_root),
            "execution_manifest": str(manifest_path),
            "execution_manifest_sha256": _sha256(manifest_path),
            "selected_action_facts": str(source_path),
            "selected_action_facts_sha256": _sha256(source_path),
        },
        "memory": {
            "process_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "note": _MEMORY_NOTE,
        },
        "safety": {
            "primary_supported_ranks": list(SUPPORTED_RANKS),
            "unsupported_rank_outcomes_imputed": False,
            "formal_roots_mutated": False,
            "raw_tapes_opened_by_fast_path": False,
            "pathwise_ev_ready": False,
            "joint_volume_allocated": False,
            "exit_outcomes_regenerated": False,
            "independent_entry_event_labels": True,
            "full_order_outcomes_persisted": False,
            "full_summary_only": True,
            "actual_exit_opportunity_metrics_absent": True,
            "actual_fifo_metrics_absent": True,
            "fifo_experimental_contract_only": True,
            "fifo_capacity_source_bound": False,
        },
    }
    return metrics, selected, universe


def publish(
    output: Path,
    metrics: dict[str, object],
    selected,
    universe,
) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    try:
        frames = {
            "full_q_summary.parquet": universe.q_summary,
            "full_product_day_audit.parquet": universe.product_day_audit,
        }
        for filename, frame in frames.items():
            frame.write_parquet(temporary / filename)
        (temporary / "benchmark.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = temporary / filename
            artifacts[filename] = {
                "kind": "parquet",
                "rows": frame.height,
                "columns": frame.width,
                "schema": [
                    [name, str(dtype)] for name, dtype in frame.schema.items()
                ],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        benchmark_path = temporary / "benchmark.json"
        artifacts["benchmark.json"] = {
            "kind": "json",
            "bytes": benchmark_path.stat().st_size,
            "sha256": _sha256(benchmark_path),
        }
        marker = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "analysis_only": True,
            "formal_roots_mutated": False,
            "pathwise_ev_ready": False,
            "joint_volume_allocated": False,
            "full_order_outcomes_persisted": False,
            "full_summary_only": True,
            "actual_exit_opportunity_metrics_absent": True,
            "actual_fifo_metrics_absent": True,
            "fifo_experimental_contract_only": True,
            "fifo_capacity_source_bound": False,
            "benchmark_observations_are_not_semantic_claims": True,
            "execution_root": metrics["source"]["execution_root"],
            "execution_manifest_sha256": metrics["source"][
                "execution_manifest_sha256"
            ],
            "partition_source_inventory_sha256": (
                EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256
            ),
            "selected_Date": metrics["Date"],
            "selected_ValueCode": metrics["ValueCode"],
            "implementation_sources": _implementation_sources(),
            "artifacts": artifacts,
        }
        (temporary / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_benchmark_bundle(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _selected_semantics(selected) -> dict[str, object]:
    audit = selected.audit
    summary = selected.q_summary
    return {
        "candidate_policy_aliases": selected.order_outcomes.height,
        "supported_l1_l2_aliases": int(
            audit["supported_outcome_aliases"].sum()
        ),
        "unsupported_unknown_aliases": int(
            audit["unsupported_unknown_aliases"].sum()
        ),
        "supported_full_fill_aliases": int(
            summary["supported_full_fill_aliases"].sum()
        ),
        "supported_partial_fill_aliases": int(
            summary["supported_partial_fill_aliases"].sum()
        ),
        "supported_any_fill_aliases": int(
            summary["supported_any_fill_aliases"].sum()
        ),
        "supported_no_fill_aliases": int(
            summary["supported_no_fill_aliases"].sum()
        ),
        "supported_cancel_required_aliases": int(
            summary["supported_cancel_required_aliases"].sum()
        ),
        "exact_hedge_observed_aliases": int(
            audit["exact_hedge_observed_aliases"].sum()
        ),
        "exact_hedge_executable_aliases": int(
            audit["exact_hedge_executable_aliases"].sum()
        ),
        "state_change_rows": selected.state_changes.height,
        "q_rank_summary_rows": summary.height,
        "output_estimated_bytes": sum(
            frame.estimated_size()
            for frame in (
                selected.order_outcomes,
                selected.state_changes,
                selected.audit,
                summary,
            )
        ),
    }


def _full_semantics(universe) -> dict[str, object]:
    audit = universe.product_day_audit
    summary = universe.q_summary
    supported = int(summary["supported_outcome_aliases"].sum())
    full = int(summary["supported_full_fill_aliases"].sum())
    partial = int(summary["supported_partial_fill_aliases"].sum())
    any_fill = int(summary["supported_any_fill_aliases"].sum())
    no_fill = int(summary["supported_no_fill_aliases"].sum())
    cancel_required = int(
        summary["supported_cancel_required_aliases"].sum()
    )
    if (
        full + partial + no_fill != supported
        or any_fill != full + partial
        or cancel_required != partial + no_fill
    ):
        raise ValueError("full-universe fill partition is incoherent")
    return {
        "source_partitions": universe.source_partitions,
        "source_action_rows": universe.source_action_rows,
        "supported_l1_l2_aliases": supported,
        "unsupported_unknown_aliases": int(
            summary["unsupported_unknown_aliases"].sum()
        ),
        "supported_full_fill_aliases": full,
        "supported_partial_fill_aliases": partial,
        "supported_any_fill_aliases": any_fill,
        "supported_no_fill_aliases": no_fill,
        "supported_cancel_required_aliases": cancel_required,
        "exact_hedge_observed_aliases": int(
            summary["exact_hedge_observed_aliases"].sum()
        ),
        "exact_hedge_executable_aliases": int(
            summary["exact_hedge_executable_aliases"].sum()
        ),
        "supported_full_fill_rate": full / supported,
        "supported_partial_fill_rate": partial / supported,
        "supported_any_fill_rate": any_fill / supported,
        "supported_no_fill_rate": no_fill / supported,
        "supported_cancel_required_rate": cancel_required / supported,
        "fill_partition_definition": (
            "full + partial_only + no_fill = supported; "
            "cancel_required = partial_only + no_fill"
        ),
        "q_rank_summary_rows": summary.height,
        "product_route_q_audit_rows": audit.height,
    }


def verify_benchmark_bundle(output: Path) -> dict[str, object]:
    """Verify exact inventory, hashes, schemas, and non-readiness flags."""

    output = Path(output)
    marker = _read_json_object(output / "complete.json")
    if set(marker) != _MARKER_KEYS:
        raise ValueError("compact benchmark marker keys mismatch")
    if marker.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("compact benchmark marker schema version mismatch")
    false_flags = (
        "formal_roots_mutated",
        "pathwise_ev_ready",
        "joint_volume_allocated",
        "full_order_outcomes_persisted",
        "fifo_capacity_source_bound",
    )
    true_flags = (
        "analysis_only",
        "full_summary_only",
        "actual_exit_opportunity_metrics_absent",
        "actual_fifo_metrics_absent",
        "fifo_experimental_contract_only",
        "benchmark_observations_are_not_semantic_claims",
    )
    if any(marker.get(field) is not False for field in false_flags) or any(
        marker.get(field) is not True for field in true_flags
    ):
        raise ValueError("compact benchmark marker safety flags are incoherent")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _EXPECTED_ARTIFACTS:
        raise ValueError("compact benchmark artifact inventory mismatch")
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != _EXPECTED_ARTIFACTS | {"complete.json"}:
        raise ValueError("compact benchmark directory has unexpected files")
    for filename, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid artifact metadata: {filename}")
        path = output / filename
        if (
            int(metadata.get("bytes", -1)) != path.stat().st_size
            or metadata.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"compact benchmark artifact mismatch: {filename}")
    if set(artifacts["benchmark.json"]) != {"kind", "bytes", "sha256"} or (
        artifacts["benchmark.json"].get("kind") != "json"
    ):
        raise ValueError("benchmark JSON artifact metadata mismatch")
    implementations = marker.get("implementation_sources")
    if (
        not isinstance(implementations, dict)
        or implementations != _implementation_sources()
    ):
        raise ValueError("compact benchmark implementation source hash mismatch")
    execution_root = Path(str(marker.get("execution_root", "")))
    if execution_root.resolve() != Path(DEFAULT_EXECUTION_ROOT).resolve():
        raise ValueError("compact benchmark execution root identity mismatch")
    if (
        marker.get("partition_source_inventory_sha256")
        != EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256
        or marker.get("execution_manifest_sha256")
        != EXPECTED_EXECUTION_MANIFEST_SHA256
    ):
        raise ValueError("compact benchmark frozen source anchor mismatch")
    metrics = _read_json_object(output / "benchmark.json")
    if set(metrics) != _METRICS_KEYS:
        raise ValueError("compact benchmark metrics keys mismatch")
    safety = metrics.get("safety")
    expected_safety = {
        "primary_supported_ranks": list(SUPPORTED_RANKS),
        "unsupported_rank_outcomes_imputed": False,
        "formal_roots_mutated": False,
        "raw_tapes_opened_by_fast_path": False,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "exit_outcomes_regenerated": False,
        "independent_entry_event_labels": True,
        "full_order_outcomes_persisted": False,
        "full_summary_only": True,
        "actual_exit_opportunity_metrics_absent": True,
        "actual_fifo_metrics_absent": True,
        "fifo_experimental_contract_only": True,
        "fifo_capacity_source_bound": False,
    }
    if (
        safety != expected_safety
        or metrics.get("benchmark_version") != BENCHMARK_VERSION
        or metrics.get("compact_source_version") != FROZEN_COMPACT_VERSION
        or str(metrics.get("Date")) != str(marker.get("selected_Date"))
        or str(metrics.get("ValueCode"))
        != str(marker.get("selected_ValueCode"))
    ):
        raise ValueError("compact benchmark metrics/readiness mismatch")
    _verify_observations(metrics)
    recomputed = summarize_frozen_compact_universe(execution_root)
    q_summary = pl.read_parquet(output / "full_q_summary.parquet")
    audit = pl.read_parquet(output / "full_product_day_audit.parquet")
    _verify_frame_artifact_metadata(
        q_summary, artifacts["full_q_summary.parquet"]
    )
    _verify_frame_artifact_metadata(
        audit, artifacts["full_product_day_audit.parquet"]
    )
    if not q_summary.equals(recomputed.q_summary, null_equal=True):
        raise ValueError("full q summary differs from source-bound recomputation")
    if not audit.equals(recomputed.product_day_audit, null_equal=True):
        raise ValueError("full product-day audit differs from recomputation")
    if metrics.get("full_universe") != _full_semantics(recomputed):
        raise ValueError("full semantic metrics differ from recomputation")
    date = str(marker["selected_Date"])
    value_code = str(marker["selected_ValueCode"])
    selected = load_frozen_compact_product_days(
        execution_root, dates=[date], value_codes=[value_code]
    )
    source_path = (
        execution_root
        / f"Date={date}"
        / f"ValueCode={value_code}"
        / "execution_action_facts.parquet"
    )
    source = pl.read_parquet(source_path)
    if metrics.get("selected_product_day") != _selected_semantics(selected):
        raise ValueError("selected product-day metrics differ from recomputation")
    if metrics.get("supported_exact_parity") != _supported_parity(
        selected.order_outcomes, source
    ):
        raise ValueError("selected exact parity differs from recomputation")
    expected_source = {
        "execution_root": str(execution_root.resolve()),
        "execution_manifest": str(
            execution_root / "execution_partition_manifest.parquet"
        ),
        "execution_manifest_sha256": _sha256(
            execution_root / "execution_partition_manifest.parquet"
        ),
        "selected_action_facts": str(source_path),
        "selected_action_facts_sha256": _sha256(source_path),
    }
    if metrics.get("source") != expected_source:
        raise ValueError("benchmark source lineage differs from recomputation")
    return marker


def _verify_frame_artifact_metadata(
    frame: pl.DataFrame, metadata: object
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid frame artifact metadata")
    if set(metadata) != {
        "kind",
        "rows",
        "columns",
        "schema",
        "bytes",
        "sha256",
    }:
        raise ValueError("frame artifact metadata keys mismatch")
    expected = {
        "kind": "parquet",
        "rows": frame.height,
        "columns": frame.width,
        "schema": [[name, str(dtype)] for name, dtype in frame.schema.items()],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("frame artifact rows/columns/schema mismatch")


def _verify_observations(metrics: dict[str, object]) -> None:
    """Validate benchmark observations without promoting them to facts."""

    timing = metrics.get("timing_seconds")
    if not isinstance(timing, dict) or set(timing) != _TIMING_KEYS:
        raise ValueError("benchmark timing observation keys mismatch")
    if timing.get("raw_replay_or_formal_outputs_recomputed") is not False:
        raise ValueError("benchmark timing scope is incoherent")
    required = (
        "selected_product_day_fast_path",
        "full_2687_end_to_end_verified_summary",
        "full_2687_marker_hash_semantic_verification",
        "full_4032586_projection_validation_aggregate",
    )
    for field in required:
        value = timing.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"invalid benchmark timing observation: {field}")
    raw = timing.get("raw_product_day_loader_comparison")
    speedup = timing.get("selected_fast_path_speedup_vs_raw_loader")
    if raw is None:
        if speedup is not None:
            raise ValueError("benchmark speedup exists without raw-loader timing")
    else:
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0
            or isinstance(speedup, bool)
            or not isinstance(speedup, (int, float))
            or not math.isfinite(float(speedup))
        ):
            raise ValueError("invalid raw-loader timing observation")
        selected = float(timing["selected_product_day_fast_path"])
        if selected <= 0 or not math.isclose(
            float(speedup), float(raw) / selected, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("benchmark speedup arithmetic mismatch")
    memory = metrics.get("memory")
    if (
        not isinstance(memory, dict)
        or set(memory) != {"process_peak_rss_kib", "note"}
        or memory.get("note") != _MEMORY_NOTE
    ):
        raise ValueError("benchmark memory observation metadata mismatch")
    rss = memory.get("process_peak_rss_kib")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss < 0:
        raise ValueError("invalid benchmark memory observation")


def _implementation_sources() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("compact_frozen_facts.py"),
        Path(__file__).with_name("compact_evaluator.py"),
        Path(__file__).with_name("execution_runner.py"),
    )
    return {str(path): _sha256(path) for path in paths}


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _require_columns(
    frame: pl.DataFrame, required: set[str], source: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _supported_parity(
    compact: pl.DataFrame, source: pl.DataFrame
) -> dict[str, object]:
    supported_source = source.filter(
        (
            (pl.col("route") == "future_ask_spot_taker")
            & (pl.col("maker_market") == "future")
            & (pl.col("maker_side") == "ask")
            & pl.col("target_rank_at_submit").is_in(["ASK1", "ASK2"])
        )
        | (
            (pl.col("route") == "spot_bid_future_taker")
            & (pl.col("maker_market") == "spot")
            & (pl.col("maker_side") == "bid")
            & pl.col("target_rank_at_submit").is_in(["BID1", "BID2"])
        )
    ).select(
        "policy_generation_id",
        *[
            (
                pl.col(f"entry_{field}")
                if field
                in {
                    "hedge_label_observed",
                    "hedge_executable",
                    "hedge_decision_time_ns",
                    "hedge_executable_vwap_price",
                }
                else pl.col(field)
            ).alias(f"source_{field}")
            for field in _PARITY_FIELDS
        ],
    )
    supported_compact = compact.filter(pl.col("outcome_supported"))
    joined = supported_compact.join(
        supported_source,
        on="policy_generation_id",
        how="full",
        coalesce=True,
        validate="1:1",
    )
    equal = {
        field: int(
            joined.select(
                pl.col(field).eq_missing(pl.col(f"source_{field}")).sum()
            ).item()
        )
        for field in _PARITY_FIELDS
    }
    return {
        "compact_supported_rows": supported_compact.height,
        "source_supported_rows": supported_source.height,
        "joined_rows": joined.height,
        "field_equal_rows": equal,
        "all_fields_exact": all(value == joined.height for value in equal.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--value-code")
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--skip-raw-loader-comparison", action="store_true")
    args = parser.parse_args()
    if args.verify_only is not None:
        if args.date is not None or args.value_code is not None or args.output is not None:
            parser.error("--verify-only cannot be combined with benchmark arguments")
        print(
            json.dumps(
                verify_benchmark_bundle(args.verify_only),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.date is None or args.value_code is None:
        parser.error("--date and --value-code are required unless --verify-only is used")
    metrics, selected, universe = run_benchmark(
        args.date,
        args.value_code,
        execution_root=args.execution_root,
        measure_raw_loader=not args.skip_raw_loader_comparison,
    )
    if args.output is not None:
        publish(args.output, metrics, selected, universe)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
