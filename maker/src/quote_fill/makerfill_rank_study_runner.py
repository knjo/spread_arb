"""Publish a bounded A/B1--5 legacy-makerFill candidate comparison.

The runner intentionally selects only actual, direct spot-snapshot entry
candidates from the completed execution facts.  It extends the historical
makerFill rule to B3--B5 on those exact snapshots, verifies B1/B2 byte-value
parity against the existing files, and compares both the EOD label and the
user-proposed stop-bounded approximation with the indexed raw truth.

This is an analysis-only diagnostic.  It never mutates the formal execution,
same-day exit, or cross-session roots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import numpy as np
import polars as pl

from .makerfill_rank_study import (
    LEGACY_RANK_STUDY_VERSION,
    LegacyMakerFillRankIndex,
)


DEFAULT_SAMPLE_DATES = (
    "20260603",  # trade/epoch stress
    "20260703",  # before futures tick-ladder change
    "20260706",  # after futures tick-ladder change
    "20260731",  # low workload
    "20260806",  # median workload
)
RANKS = ("BID1", "BID2", "BID3", "BID4", "BID5")
ACTION_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_rank_at_submit",
    "target_price",
    "initial_queue_ahead",
    "intended_quantity",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "full_fill_recv_time_ns",
    "full_fill",
    "entry_hedge_executable",
    "entry_hedge_signed_total_slippage_bp",
)
TICK_COLUMNS = (
    "TransTime",
    "ValueCode",
    "ChannelSeq",
    "marketOpen",
    "FillPrice",
    "FillLots",
    *(f"AskPrice{level}" for level in range(1, 6)),
    *(f"AskLots{level}" for level in range(1, 6)),
    *(f"BidPrice{level}" for level in range(1, 6)),
    *(f"BidLots{level}" for level in range(1, 6)),
)


@dataclass(frozen=True)
class MakerFillRankStudyConfig:
    execution_root: str
    tick_root: str
    makerfill_root: str
    dates: tuple[str, ...] = DEFAULT_SAMPLE_DATES
    analysis_version: str = "selected_direct_spot_candidate_rank_study_v5_epsilon_exact"

    def validate(self) -> None:
        if self.analysis_version != (
            "selected_direct_spot_candidate_rank_study_v5_epsilon_exact"
        ):
            raise ValueError("unsupported analysis_version")
        if not self.dates or len(self.dates) != len(set(self.dates)):
            raise ValueError("dates must be unique and non-empty")
        if tuple(sorted(self.dates)) != self.dates:
            raise ValueError("dates must be sorted")
        for value in self.dates:
            try:
                datetime.strptime(value, "%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"invalid study date: {value}") from exc


@dataclass(frozen=True)
class MakerFillRankStudyResult:
    raw_labels: pl.DataFrame
    alias_comparison: pl.DataFrame
    rank_summary: pl.DataFrame
    audit: pl.DataFrame


def run_makerfill_rank_study(
    config: MakerFillRankStudyConfig,
) -> MakerFillRankStudyResult:
    """Build the selected-date comparison fully in memory."""

    config.validate()
    execution_root = Path(config.execution_root)
    tick_root = Path(config.tick_root)
    makerfill_root = Path(config.makerfill_root)
    raw_parts: list[pl.DataFrame] = []
    alias_parts: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for date in config.dates:
        action_paths = sorted(
            execution_root.glob(
                f"Date={date}/ValueCode=*/execution_action_facts.parquet"
            )
        )
        if not action_paths:
            raise FileNotFoundError(f"no execution actions for {date}")
        # Real zero-action partitions intentionally use a narrow typed-empty
        # schema, so read partitions independently and relax only at concat.
        action_frames = [
            pl.read_parquet(path).select(
                *(
                    pl.col(column).cast(pl.Float64).alias(column)
                    if column
                    in {
                        "target_price",
                        "entry_hedge_signed_total_slippage_bp",
                    }
                    else pl.col(column)
                    for column in ACTION_COLUMNS
                    if column in pl.read_parquet_schema(path)
                )
            )
            for path in action_paths
        ]
        actions = pl.concat(action_frames, how="diagonal_relaxed")
        missing_action_columns = sorted(set(ACTION_COLUMNS) - set(actions.columns))
        if missing_action_columns:
            raise ValueError(
                f"{date}: action facts missing columns {missing_action_columns}"
            )
        selected = actions.filter(
            (pl.col("route") == "spot_bid_future_taker")
            & (pl.col("maker_market") == "spot")
            & (pl.col("maker_side") == "bid")
            & pl.col("target_rank_at_submit").is_in(RANKS)
            & (pl.col("submit_event_sequence") == 2)
        )
        if selected.is_empty():
            raise ValueError(f"{date}: selected direct spot candidates are empty")
        _validate_selected_actions(selected, date)
        _validate_action_aliases(selected, date)
        raw = selected.unique("raw_order_fact_id", keep="first").sort(
            ["ValueCode", "submit_row_index", "raw_order_fact_id"]
        )
        value_codes = sorted(
            str(value) for value in raw.get_column("ValueCode").unique()
        )
        tick_path = tick_root / f"{date}_StockTick.parquet"
        makerfill_path = makerfill_root / f"{date}_makerFill.parquet"
        if not tick_path.is_file() or not makerfill_path.is_file():
            raise FileNotFoundError(
                f"{date}: missing tick or makerFill source"
            )
        ticks = (
            pl.scan_parquet(tick_path)
            .filter(pl.col("ValueCode").cast(pl.String).is_in(value_codes))
            .select(*TICK_COLUMNS)
            .collect(engine="streaming")
        )
        stored = (
            pl.scan_parquet(makerfill_path)
            .filter(pl.col("QuoteCode").cast(pl.String).is_in(value_codes))
            .select(
                pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
                "ChannelSeq",
                "Bid1_FillSeconds",
                "Bid2_FillSeconds",
            )
            .collect(engine="streaming")
        )
        stored_duplicate = stored.group_by("ValueCode", "ChannelSeq").len().filter(
            pl.col("len") != 1
        )
        if stored_duplicate.height:
            raise ValueError(f"{date}: stored makerFill keys are duplicated")
        stored_lookup = {
            (str(row["ValueCode"]), int(row["ChannelSeq"])): row
            for row in stored.iter_rows(named=True)
        }

        labels: list[dict[str, object]] = []
        for value_code in value_codes:
            product_ticks = ticks.filter(
                pl.col("ValueCode").cast(pl.String) == value_code
            )
            if product_ticks.is_empty():
                raise ValueError(f"{date}/{value_code}: tick states are empty")
            index = LegacyMakerFillRankIndex(product_ticks)
            product_raw = raw.filter(
                pl.col("ValueCode").cast(pl.String) == value_code
            )
            for row in product_raw.iter_rows(named=True):
                rank = str(row["target_rank_at_submit"])
                level = int(rank[-1])
                sequence = int(row["submit_row_index"])
                label = index.label(sequence, "bid", level)
                if not label.label_available:
                    raise ValueError(
                        f"{date}/{value_code}/{sequence}/{rank}: "
                        "selected snapshot rank is unavailable"
                    )
                if (
                    label.target_price is None
                    or abs(label.target_price - float(row["target_price"]))
                    >= 1e-8
                    or label.initial_displayed_lots
                    != int(row["initial_queue_ahead"])
                ):
                    raise ValueError(
                        f"{date}/{value_code}/{sequence}/{rank}: "
                        "formal action does not match the direct tick snapshot"
                    )
                stored_seconds: float | None = None
                parity: bool | None = None
                if level <= 2:
                    old = stored_lookup.get((value_code, sequence))
                    if old is None:
                        raise ValueError(
                            f"{date}/{value_code}/{sequence}: missing stored makerFill key"
                        )
                    stored_seconds = _optional_finite(
                        old[f"Bid{level}_FillSeconds"]
                    )
                    parity = _same_optional_float(
                        label.fill_seconds, stored_seconds
                    )
                    if not parity:
                        raise ValueError(
                            f"{date}/{value_code}/{sequence}/{rank}: "
                            "legacy extension does not reproduce stored makerFill"
                        )
                labels.append(
                    {
                        "Date": date,
                        "ValueCode": value_code,
                        "raw_order_fact_id": row["raw_order_fact_id"],
                        "ChannelSeq": sequence,
                        "target_rank_at_submit": rank,
                        "rank_level": level,
                        "rank_group": "L1_2" if level <= 2 else "L3_5",
                        "target_price": label.target_price,
                        "initial_displayed_lots": label.initial_displayed_lots,
                        "legacy_fill_seconds": label.fill_seconds,
                        "legacy_fill_trans_time_us": label.fill_trans_time_us,
                        "legacy_fill_ChannelSeq": label.fill_channel_sequence,
                        "legacy_fill_reason": label.fill_reason,
                        "legacy_eod_positive": label.fill_seconds is not None,
                        "stored_l1_l2_fill_seconds": stored_seconds,
                        "stored_l1_l2_parity": parity,
                        "legacy_outcome_exact": False,
                        "legacy_cancel_bounded": False,
                        "legacy_own_quantity_included": False,
                    }
                )
        raw_labels = pl.from_dicts(labels, infer_schema_length=None)
        aliases = selected.join(
            raw_labels,
            on=["Date", "ValueCode", "raw_order_fact_id"],
            how="left",
            validate="m:1",
        ).with_columns(
            pl.when(pl.col("legacy_fill_seconds").is_not_null())
            .then(
                pl.col("submit_recv_time_ns")
                + (pl.col("legacy_fill_seconds") * 1_000_000_000.0)
                .round(0)
                .cast(pl.Int64)
            )
            .otherwise(None)
            .alias("legacy_mixed_clock_fill_ns")
        ).with_columns(
            (
                pl.col("legacy_mixed_clock_fill_ns").is_not_null()
                & (
                    pl.col("legacy_mixed_clock_fill_ns")
                    > pl.col("submit_recv_time_ns")
                )
                & (
                    pl.col("legacy_mixed_clock_fill_ns")
                    <= pl.col("nominal_stop_recv_time_ns")
                )
            ).alias("legacy_stop_bounded_full_approx"),
            pl.lit(True).alias("in_direct_spot_rank_denominator"),
            pl.lit(False).alias("mixed_clock_outcome_exact"),
            pl.lit(False).alias("pathwise_ev_ready"),
            pl.lit(False).alias("joint_volume_allocated"),
        ).with_columns(
            pl.when(pl.col("legacy_stop_bounded_full_approx") & pl.col("full_fill"))
            .then(pl.lit("TP"))
            .when(pl.col("legacy_stop_bounded_full_approx") & ~pl.col("full_fill"))
            .then(pl.lit("FP"))
            .when(~pl.col("legacy_stop_bounded_full_approx") & pl.col("full_fill"))
            .then(pl.lit("FN"))
            .otherwise(pl.lit("TN"))
            .alias("legacy_vs_indexed_confusion")
        )
        if aliases.filter(pl.col("target_rank_at_submit").is_null()).height:
            raise ValueError(f"{date}: rank join failed")
        raw_parts.append(raw_labels)
        alias_parts.append(aliases)
        audit_rows.append(
            {
                "Date": date,
                "selected_aliases": selected.height,
                "selected_raw_orders": raw.height,
                "tick_rows_source": ticks.height,
                "tick_rows_market_open": ticks.filter(
                    pl.col("marketOpen") == True  # noqa: E712
                ).height,
                "stored_makerfill_rows": stored.height,
                "l1_l2_raw_orders": raw_labels.filter(
                    pl.col("rank_group") == "L1_2"
                ).height,
                "l3_l5_raw_orders": raw_labels.filter(
                    pl.col("rank_group") == "L3_5"
                ).height,
                "stored_l1_l2_parity_failures": raw_labels.filter(
                    pl.col("stored_l1_l2_parity") == False  # noqa: E712
                ).height,
                "analysis_only": True,
                "outcome_exact": False,
                "pathwise_ev_ready": False,
            }
        )

    raw_all = pl.concat(raw_parts, how="vertical_relaxed").sort(
        ["Date", "ValueCode", "ChannelSeq", "raw_order_fact_id"]
    )
    alias_all = pl.concat(alias_parts, how="vertical_relaxed").sort(
        [
            "Date",
            "ValueCode",
            "boundary_quantile",
            "submit_recv_time_ns",
            "policy_generation_id",
        ]
    )
    summary = _rank_summary(alias_all)
    audit = pl.from_dicts(audit_rows, infer_schema_length=None).sort("Date")
    return MakerFillRankStudyResult(raw_all, alias_all, summary, audit)


def publish_makerfill_rank_study(
    output_root: Path,
    config: MakerFillRankStudyConfig,
) -> Path:
    """Atomically publish four facts plus a fail-closed analysis marker."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=output_root.parent
        )
    )
    try:
        source_inventory = _build_source_inventory(config)
        result = run_makerfill_rank_study(config)
        frames = {
            "selected_raw_labels.parquet": result.raw_labels,
            "alias_comparison.parquet": result.alias_comparison,
            "rank_summary.parquet": result.rank_summary,
            "audit.parquet": result.audit,
            "input_inventory.parquet": source_inventory,
        }
        artifact_meta: dict[str, dict[str, object]] = {}
        for name, frame in frames.items():
            path = stage / name
            frame.write_parquet(path, compression="zstd")
            artifact_meta[name] = {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "schema": _schema_payload(frame.schema),
            }
        config_payload = asdict(config)
        implementation_files = _implementation_identity()
        marker = {
            "status": "complete",
            "analysis_version": config.analysis_version,
            "legacy_rank_engine_version": LEGACY_RANK_STUDY_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_payload,
            "config_sha256": _canonical_sha256(config_payload),
            "implementation_files": implementation_files,
            "implementation_identity_sha256": _canonical_sha256(
                implementation_files
            ),
            "artifacts": artifact_meta,
            "safety": {
                "analysis_only": True,
                "legacy_eod_label_is_execution_truth": False,
                "mixed_clock_stop_label_is_exact": False,
                "pathwise_ev_ready": False,
                "joint_volume_allocated": False,
                "cost_profile_complete": False,
                "unsupported_ranks_imputed_no_fill": False,
            },
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.rename(output_root)
        return output_root
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_makerfill_rank_study(output_root: Path) -> dict[str, object]:
    output_root = Path(output_root)
    marker_path = output_root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError("rank study marker is incomplete")
    if marker.get("analysis_version") != (
        "selected_direct_spot_candidate_rank_study_v5_epsilon_exact"
    ):
        raise ValueError("rank study analysis version mismatch")
    if marker.get("legacy_rank_engine_version") != LEGACY_RANK_STUDY_VERSION:
        raise ValueError("rank study engine version mismatch")
    expected_safety = {
        "analysis_only": True,
        "legacy_eod_label_is_execution_truth": False,
        "mixed_clock_stop_label_is_exact": False,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "cost_profile_complete": False,
        "unsupported_ranks_imputed_no_fill": False,
    }
    if marker.get("safety") != expected_safety:
        raise ValueError("rank study safety contract mismatch")
    expected_files = {
        "selected_raw_labels.parquet",
        "alias_comparison.parquet",
        "rank_summary.parquet",
        "audit.parquet",
        "input_inventory.parquet",
    }
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_files:
        raise ValueError("rank study artifact declaration mismatch")
    actual_files = {path.name for path in output_root.iterdir() if path.is_file()}
    if actual_files != expected_files | {"complete.json"}:
        raise ValueError("rank study root file set mismatch")
    for name in sorted(expected_files):
        path = output_root / name
        meta = artifacts[name]
        frame = pl.read_parquet(path)
        if (
            path.stat().st_size != int(meta["bytes"])
            or _file_sha256(path) != str(meta["sha256"])
            or frame.height != int(meta["rows"])
            or frame.width != int(meta["columns"])
            or _schema_payload(frame.schema) != meta.get("schema")
        ):
            raise ValueError(f"rank study artifact mismatch: {name}")
    config = marker.get("config")
    if not isinstance(config, dict) or _canonical_sha256(config) != marker.get(
        "config_sha256"
    ):
        raise ValueError("rank study config hash mismatch")
    study_config = MakerFillRankStudyConfig(
        execution_root=str(config["execution_root"]),
        tick_root=str(config["tick_root"]),
        makerfill_root=str(config["makerfill_root"]),
        dates=tuple(str(value) for value in config["dates"]),
        analysis_version=str(config["analysis_version"]),
    )
    study_config.validate()
    implementation = marker.get("implementation_files")
    if (
        not isinstance(implementation, dict)
        or implementation != _implementation_identity()
        or _canonical_sha256(implementation)
        != marker.get("implementation_identity_sha256")
    ):
        raise ValueError("rank study implementation identity mismatch")

    # Verification is intentionally source-bound, not merely self-attesting:
    # re-hash every selected action/tick/makerFill input and independently
    # recompute the four derived frames with the frozen implementation.
    expected_inventory = _build_source_inventory(study_config)
    actual_inventory = pl.read_parquet(output_root / "input_inventory.parquet")
    _assert_frame_exact(
        actual_inventory, expected_inventory, "input_inventory.parquet"
    )
    recomputed = run_makerfill_rank_study(study_config)
    expected_frames = {
        "selected_raw_labels.parquet": recomputed.raw_labels,
        "alias_comparison.parquet": recomputed.alias_comparison,
        "rank_summary.parquet": recomputed.rank_summary,
        "audit.parquet": recomputed.audit,
    }
    for name, expected in expected_frames.items():
        _assert_frame_exact(pl.read_parquet(output_root / name), expected, name)
    return marker


def _validate_action_aliases(actions: pl.DataFrame, date: str) -> None:
    if actions.select("policy_generation_id").n_unique() != actions.height:
        raise ValueError(f"{date}: duplicate policy_generation_id")
    fixed = (
        actions.group_by("raw_order_fact_id")
        .agg(
            *(
                pl.col(column).n_unique().alias(column)
                for column in (
                    "Date",
                    "ValueCode",
                    "route",
                    "maker_market",
                    "maker_side",
                    "target_rank_at_submit",
                    "target_price",
                    "initial_queue_ahead",
                    "intended_quantity",
                    "submit_recv_time_ns",
                    "submit_event_sequence",
                    "submit_row_index",
                )
            )
        )
        .filter(
            pl.any_horizontal(
                *(pl.col(column) != 1 for column in (
                    "Date",
                    "ValueCode",
                    "route",
                    "maker_market",
                    "maker_side",
                    "target_rank_at_submit",
                    "target_price",
                    "initial_queue_ahead",
                    "intended_quantity",
                    "submit_recv_time_ns",
                    "submit_event_sequence",
                    "submit_row_index",
                ))
            )
        )
    )
    if fixed.height:
        raise ValueError(f"{date}: raw-order aliases disagree on submit state")


def _validate_selected_actions(actions: pl.DataFrame, date: str) -> None:
    """Reject null/coherency drift before computing any confusion cell."""

    invalid = actions.filter(
        pl.col("raw_order_fact_id").is_null()
        | pl.col("policy_generation_id").is_null()
        | pl.col("target_price").is_null()
        | (~pl.col("target_price").is_finite())
        | (pl.col("target_price") <= 0)
        | pl.col("initial_queue_ahead").is_null()
        | (pl.col("initial_queue_ahead") < 0)
        | pl.col("intended_quantity").is_null()
        | (pl.col("intended_quantity") <= 0)
        | pl.col("submit_recv_time_ns").is_null()
        | pl.col("submit_row_index").is_null()
        | pl.col("nominal_stop_recv_time_ns").is_null()
        | (
            pl.col("nominal_stop_recv_time_ns")
            < pl.col("submit_recv_time_ns")
        )
        | pl.col("nominal_stop_reason").is_null()
        | pl.col("full_fill").is_null()
        | pl.col("entry_hedge_executable").is_null()
        | (
            pl.col("full_fill")
            & pl.col("full_fill_recv_time_ns").is_null()
        )
        | (
            (~pl.col("full_fill"))
            & pl.col("full_fill_recv_time_ns").is_not_null()
        )
        | (
            pl.col("full_fill")
            & (
                pl.col("full_fill_recv_time_ns")
                > pl.col("nominal_stop_recv_time_ns")
            )
        )
    )
    if invalid.height:
        raise ValueError(
            f"{date}: selected action facts contain {invalid.height} "
            "null or incoherent execution rows"
        )


def _rank_summary(aliases: pl.DataFrame) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for population, frame in (
        ("alias", aliases),
        (
            "q_raw_order",
            aliases.unique(
                ["Date", "boundary_quantile", "raw_order_fact_id"],
                keep="first",
            ),
        ),
    ):
        for date_scope, scoped in (
            ("by_date", frame),
            ("all_dates", frame.with_columns(pl.lit("__all__").alias("Date"))),
        ):
            for rank_scope, group_keys in (
                (
                    "individual_rank",
                    [
                        "Date",
                        "boundary_quantile",
                        "target_rank_at_submit",
                        "rank_group",
                    ],
                ),
                (
                    "rank_group",
                    ["Date", "boundary_quantile", "rank_group"],
                ),
            ):
                grouped = scoped.group_by(*group_keys).agg(
                    pl.len().alias("n"),
                    pl.col("legacy_eod_positive")
                    .sum()
                    .alias("legacy_eod_positive"),
                    pl.col("legacy_stop_bounded_full_approx")
                    .sum()
                    .alias("legacy_stop_bounded_positive"),
                    pl.col("full_fill").sum().alias("indexed_full_fill"),
                    pl.col("entry_hedge_executable")
                    .sum()
                    .alias("indexed_hedge_executable"),
                    (pl.col("legacy_vs_indexed_confusion") == "TP")
                    .sum()
                    .alias("tp"),
                    (pl.col("legacy_vs_indexed_confusion") == "FP")
                    .sum()
                    .alias("fp"),
                    (pl.col("legacy_vs_indexed_confusion") == "FN")
                    .sum()
                    .alias("fn"),
                    (pl.col("legacy_vs_indexed_confusion") == "TN")
                    .sum()
                    .alias("tn"),
                    pl.col("legacy_fill_seconds")
                    .drop_nulls()
                    .median()
                    .alias("legacy_eod_fill_seconds_p50"),
                    pl.col("entry_hedge_signed_total_slippage_bp")
                    .filter(pl.col("entry_hedge_executable"))
                    .mean()
                    .round(12)
                    .alias("indexed_hedge_slippage_bp_mean"),
                )
                if rank_scope == "rank_group":
                    grouped = grouped.with_columns(
                        pl.col("rank_group").alias("target_rank_at_submit")
                    )
                grouped = grouped.with_columns(
                    pl.lit(population).alias("population_unit"),
                    pl.lit(date_scope).alias("date_scope"),
                    pl.lit(rank_scope).alias("rank_scope"),
                    (100.0 * pl.col("legacy_eod_positive") / pl.col("n"))
                    .alias("legacy_eod_positive_pct"),
                    (100.0 * pl.col("legacy_stop_bounded_positive") / pl.col("n"))
                    .alias("legacy_stop_bounded_positive_pct"),
                    (100.0 * pl.col("indexed_full_fill") / pl.col("n"))
                    .alias("indexed_full_fill_pct"),
                    pl.when(pl.col("tp") + pl.col("fp") > 0)
                    .then(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
                    .otherwise(None)
                    .alias("legacy_stop_precision"),
                    pl.when(pl.col("tp") + pl.col("fn") > 0)
                    .then(pl.col("tp") / (pl.col("tp") + pl.col("fn")))
                    .otherwise(None)
                    .alias("legacy_stop_recall"),
                    pl.lit(False).alias("strategy_defensible"),
                )
                prefix = [
                    "Date",
                    "boundary_quantile",
                    "target_rank_at_submit",
                    "rank_group",
                ]
                grouped = grouped.select(
                    *prefix,
                    *(column for column in grouped.columns if column not in prefix),
                )
                rows.append(grouped)
    return pl.concat(rows, how="vertical_relaxed").sort(
        [
            "date_scope",
            "Date",
            "population_unit",
            "rank_scope",
            "boundary_quantile",
            "target_rank_at_submit",
        ]
    )


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_bits = np.float32(left).view(np.uint32).item()
    right_bits = np.float32(right).view(np.uint32).item()
    return bool(left_bits == right_bits)


def _build_source_inventory(config: MakerFillRankStudyConfig) -> pl.DataFrame:
    """Cryptographically bind every source file selected by the study."""

    records: list[dict[str, object]] = []
    execution_root = Path(config.execution_root)
    tick_root = Path(config.tick_root)
    makerfill_root = Path(config.makerfill_root)
    manifest_path = execution_root / "execution_partition_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    execution_manifest = pl.read_parquet(manifest_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    )
    required_manifest = {
        "Date",
        "ValueCode",
        "partition",
        "config_sha256",
        "complete",
        "execution_action_facts_rows",
    }
    missing_manifest = sorted(
        required_manifest - set(execution_manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            f"execution root manifest missing columns: {missing_manifest}"
        )
    if (
        execution_manifest.select("Date", "ValueCode").n_unique()
        != execution_manifest.height
    ):
        raise ValueError("execution root manifest keys are duplicated")
    records.append(
        _source_record(
            "execution_root_manifest", "__all__", None, manifest_path
        )
    )
    for date in config.dates:
        action_paths = sorted(
            execution_root.glob(
                f"Date={date}/ValueCode=*/execution_action_facts.parquet"
            )
        )
        if not action_paths:
            raise FileNotFoundError(f"no execution actions for {date}")
        for path in action_paths:
            value_code = path.parent.name.removeprefix("ValueCode=")
            action_record = _source_record(
                "execution_action_facts", date, value_code, path
            )
            marker_path = path.parent / "complete.json"
            manifest_row = execution_manifest.filter(
                (pl.col("Date") == date)
                & (pl.col("ValueCode") == value_code)
            )
            _validate_execution_partition_lineage(
                path,
                marker_path,
                date,
                value_code,
                action_record,
                manifest_row,
            )
            records.append(action_record)
            records.append(
                _source_record(
                    "execution_partition_marker",
                    date,
                    value_code,
                    marker_path,
                )
            )
        records.append(
            _source_record(
                "stock_tick",
                date,
                None,
                tick_root / f"{date}_StockTick.parquet",
            )
        )
        records.append(
            _source_record(
                "makerfill_l1_l2",
                date,
                None,
                makerfill_root / f"{date}_makerFill.parquet",
            )
        )
    return pl.from_dicts(
        records,
        schema={
            "source_kind": pl.String,
            "Date": pl.String,
            "ValueCode": pl.String,
            "path": pl.String,
            "bytes": pl.Int64,
            "sha256": pl.String,
            "rows": pl.Int64,
            "columns": pl.Int64,
            "schema_sha256": pl.String,
        },
        infer_schema_length=None,
    ).sort("source_kind", "Date", "ValueCode", "path")


def _validate_execution_partition_lineage(
    action_path: Path,
    marker_path: Path,
    date: str,
    value_code: str,
    action_record: dict[str, object],
    manifest_row: pl.DataFrame,
) -> None:
    """Verify the formal marker and root manifest, not only inventory them."""

    from .execution_runner import EXECUTION_RUNNER_VERSION

    if manifest_row.height != 1:
        raise ValueError(f"{date}/{value_code}: execution manifest key mismatch")
    manifest = manifest_row.row(0, named=True)
    if manifest["complete"] is not True:
        raise ValueError(f"{date}/{value_code}: execution manifest incomplete")
    manifest_partition = Path(str(manifest["partition"]))
    if not manifest_partition.is_absolute():
        manifest_partition = Path.cwd() / manifest_partition
    if manifest_partition.resolve() != action_path.parent.resolve():
        raise ValueError(f"{date}/{value_code}: execution partition path drift")
    if int(manifest["execution_action_facts_rows"]) != int(
        action_record["rows"]
    ):
        raise ValueError(f"{date}/{value_code}: execution manifest row drift")
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("complete") is not True
        or str(marker.get("Date")) != date
        or str(marker.get("ValueCode")) != value_code
        or marker.get("runner_version") != EXECUTION_RUNNER_VERSION
        or marker.get("config_sha256") != manifest["config_sha256"]
    ):
        raise ValueError(f"{date}/{value_code}: execution marker identity drift")
    marker_config = marker.get("config")
    if (
        not isinstance(marker_config, dict)
        or _canonical_sha256(marker_config) != marker.get("config_sha256")
    ):
        raise ValueError(f"{date}/{value_code}: execution marker config drift")
    semantics = marker.get("fact_semantics")
    if (
        not isinstance(semantics, dict)
        or int(semantics.get("hedge_delay_ns", -1)) != 50_000_000
        or semantics.get("independent_event_label") is not True
        or semantics.get("joint_volume_allocated") is not False
    ):
        raise ValueError(f"{date}/{value_code}: execution semantics drift")
    artifacts = marker.get("artifacts")
    action_meta = (
        artifacts.get("execution_action_facts.parquet")
        if isinstance(artifacts, dict)
        else None
    )
    if (
        not isinstance(action_meta, dict)
        or action_meta.get("sha256") != action_record["sha256"]
        or int(action_meta.get("bytes", -1)) != int(action_record["bytes"])
        or int(action_meta.get("rows", -1)) != int(action_record["rows"])
        or int(action_meta.get("columns", -1)) != int(action_record["columns"])
    ):
        raise ValueError(f"{date}/{value_code}: execution action lineage drift")


def _source_record(
    source_kind: str,
    date: str,
    value_code: str | None,
    path: Path,
) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        rows: int | None = int(
            pl.scan_parquet(path)
            .select(pl.len().cast(pl.Int64).alias("rows"))
            .collect(engine="streaming")
            .item()
        )
        columns: int | None = len(schema)
        schema_sha256: str | None = _canonical_sha256(
            _schema_payload(schema)
        )
    else:
        rows = None
        columns = None
        schema_sha256 = None
    return {
        "source_kind": source_kind,
        "Date": str(date),
        "ValueCode": None if value_code is None else str(value_code),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "rows": rows,
        "columns": columns,
        "schema_sha256": schema_sha256,
    }


def _implementation_identity() -> dict[str, dict[str, object]]:
    runner = Path(__file__).resolve()
    project_root = runner.parents[6]
    paths = {
        "rank_index": runner.with_name("makerfill_rank_study.py"),
        "rank_runner": runner,
        "legacy_maker_queue_producer": (
            project_root / "src/features/definitions/maker_queue.py"
        ),
        "legacy_makerfill_driver": project_root / "script/run_makerFill.py",
    }
    identity: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        identity[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return identity


def _schema_payload(schema: pl.Schema) -> list[dict[str, str]]:
    return [
        {"name": name, "dtype": str(dtype)}
        for name, dtype in schema.items()
    ]


def _assert_frame_exact(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    source: str,
) -> None:
    if actual.schema != expected.schema or not actual.equals(expected):
        raise ValueError(f"rank study semantic recomputation mismatch: {source}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
