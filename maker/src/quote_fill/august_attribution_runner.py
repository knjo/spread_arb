"""Run and verify the S0 May--August q95 deterioration attribution.

This is a diagnostic quote-only replay.  Capital is deliberately uncapped;
the Spot venue's rolling 100-request limit is not.  The runner validates the
fixed causal manifest and D-1 q95 rows, derives raw first-touch excursions,
adapts the existing q95 intent stream to actual send cursors, and publishes an
immutable, denominator-explicit bundle.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_MATPLOTLIB_CACHE = Path("/tmp/codex_s0_matplotlib")
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT, futures_raw_path
from .august_attribution import (
    ACTUAL_CANCEL_PHASE,
    ACTUAL_NEW_PHASE,
    APPROXIMATE_FILL_PHASE,
    RAW_FUTURE_PHASE,
    RAW_SPOT_PHASE,
    SCHEMA_VERSION,
    build_decomposition,
    deduplicate_touch_pairs_to_orders,
    extract_positive_excursions,
    link_touch_order_pairs,
    summarize_monthly_attribution,
    summarize_post_touch_by_rank,
)
from .execution_runner import (
    WalkForwardExecutionDayBatch,
    load_walkforward_execution_day_batch,
)
from .venue_scheduler import (
    EventCursor,
    RollingVenueScheduler,
    VenueRequestIntent,
)

RUNNER_VERSION = "august_attribution_raw_touch_quote_only_v2"
SCENARIO_ID = "ab12_entry_until_1300"
ROUTE_ID = "spot_bid_future_taker"
STAGE_ID = "entry"
MAKER_SIDE = "bid"
BOUNDARY_QUANTILE = 95
SPOT_REQUEST_CAP = 100
REQUEST_ASSIGNMENT_PHASE = 4
SESSION_START_SECOND = 300
ENTRY_DRAIN_START_SECOND = 14_398
ENTRY_CUTOFF_SECOND = 14_400
SESSION_END_SECOND = 15_600
ONE_SECOND_NS = 1_000_000_000
MANIFEST_SHA256 = (
    "9f1bcddf17eff968ee51e0decdb04736a3747f0665886ce3e4fd26031cfb5891"
)
EXPECTED_SESSION_COUNT = 72
EXPECTED_PRODUCT_DAY_COUNT = 3_886
EXPECTED_FIRST_DATE = "20260504"
EXPECTED_LAST_DATE = "20260813"
EXPECTED_INPUT_RECORD_COUNT = 2 + (8 * EXPECTED_SESSION_COUNT)

DEFAULT_MANIFEST_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "monthly_product_selector_causal_v2_20260822"
    / "daily_entry_manifest.csv"
)
DEFAULT_BOUNDARY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "rolling_boundaries"
    / "rolling_boundary_snapshots.parquet"
)
DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
DEFAULT_MESSAGE_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "order_message_load_causal_v2_20260822_v2"
    / "spot_message_events"
)
DEFAULT_OUTCOME_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "one_second_makerfill_causal_v2_20260822_v1"
    / "candidate_outcomes"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "august_attribution_s0_20260824_v1"
)

FOCUSED_TEST_MODULES = (
    "maker.src.tests.test_quote_fill_venue_scheduler",
    "maker.src.tests.test_quote_fill_one_second_message_load",
    "maker.src.tests.test_quote_fill_one_second_message_load_runner",
    "maker.src.tests.test_quote_fill_one_second_makerfill_runner",
    "maker.src.tests.test_quote_fill_august_attribution",
)

FRAME_ARTIFACTS = (
    "market_excursions.parquet",
    "raw_order_facts.parquet",
    "touch_order_attribution.parquet",
    "touched_raw_order_facts.parquet",
    "request_assignments.parquet",
    "product_day_coverage.parquet",
    "monthly_attribution.csv",
    "post_touch_fill_by_rank.csv",
    "membership_sensitivity.csv",
    "decomposition.csv",
    "daily_input_audit.csv",
)
NONFRAME_ARTIFACTS = (
    "august_attribution_dual_panel.png",
    "run_config.json",
    "verification.json",
)


@dataclass(frozen=True)
class AttributionPaths:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    boundary_path: Path = DEFAULT_BOUNDARY_PATH
    daily_root: Path = DEFAULT_DAILY_ROOT
    message_root: Path = DEFAULT_MESSAGE_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    data_root: Path = HFT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def load_manifest_and_boundaries(
    manifest_path: Path,
    boundary_path: Path,
    *,
    dates: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the fixed matched universe and its unique safe D-1 q95 rows."""

    if _sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("fixed S0 manifest SHA-256 mismatch")
    manifest = pl.read_csv(
        manifest_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "source_month_last_date": pl.String,
            "source_asof_date": pl.String,
        },
    ).select("Date", "ValueCode", "QuoteCode").unique()
    if dates is not None:
        requested = [str(value) for value in dates]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("dates must be non-empty and unique")
        manifest = manifest.filter(pl.col("Date").is_in(requested))
        missing = sorted(set(requested) - set(manifest["Date"].to_list()))
        if missing:
            raise ValueError(f"requested dates absent from manifest: {missing}")
    manifest = manifest.sort(["Date", "ValueCode"])
    if manifest.is_empty() or (
        manifest.select("Date", "ValueCode", "QuoteCode").n_unique()
        != manifest.height
    ):
        raise ValueError("matched manifest is empty or duplicated")

    boundaries = (
        pl.scan_parquet(boundary_path)
        .filter(
            (pl.col("boundary_quantile") == BOUNDARY_QUANTILE)
            & pl.col("adaptive_parameter_valid").fill_null(False)
            & pl.col("execution_safe_snapshot").fill_null(False)
            & ~pl.col("contains_target_day_outcome").fill_null(True)
        )
        .select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("upper_distance_bp").cast(pl.Float64),
            pl.col("lower_distance_bp").cast(pl.Float64),
            pl.col("source_asof_date").cast(pl.String),
            pl.col("history_sessions_global").cast(pl.Int64),
            pl.col("history_sessions_product").cast(pl.Int64),
            pl.col("lookback_sessions").cast(pl.Int64),
            pl.col("parameter_version").cast(pl.String),
        )
        .collect(engine="streaming")
    )
    keys = ["Date", "ValueCode", "QuoteCode"]
    selected = manifest.join(boundaries, on=keys, how="left", validate="1:1")
    invalid = selected.filter(
        pl.col("upper_distance_bp").is_null()
        | (pl.col("upper_distance_bp") <= 0)
        | pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if not invalid.is_empty():
        raise ValueError("manifest q95 boundary coverage is incomplete or unsafe")
    return manifest, selected.sort(keys)


def build_raw_residual_states(
    batch: WalkForwardExecutionDayBatch,
    boundary_day: pl.DataFrame,
) -> pl.DataFrame:
    """Build sparse raw-event residual observations for one selected day."""

    spot = _add_formal_state(batch.raw_tape.spot_states)
    future = _add_formal_state(batch.raw_tape.future_states)
    spread_clock = batch.spot_feature_state.select(
        "ValueCode",
        pl.col("spot_channel_seq").cast(pl.UInt64).alias("sequence"),
        pl.col("spread_pair_epoch").cast(pl.Int64),
    )
    spot = spot.join(
        spread_clock,
        on=["ValueCode", "sequence"],
        how="left",
        validate="1:1",
    )
    if spot["spread_pair_epoch"].null_count():
        raise ValueError(f"{batch.date}: raw SpreadPair clock join is incomplete")

    spot_events = spot.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        pl.col("recv_time_ns").cast(pl.Int64).alias("cursor_time_ns"),
        pl.lit(RAW_SPOT_PHASE, dtype=pl.Int64).alias(
            "cursor_event_sequence"
        ),
        pl.col("sequence").cast(pl.Int64).alias("cursor_row_index"),
        pl.lit("spot").alias("event_source"),
        pl.col("formal_after_trial").alias("spot_formal"),
        pl.col("bid_price_1").alias("spot_bid"),
        pl.col("ask_price_1").alias("spot_ask"),
        pl.col("bid_lots_1").alias("spot_bid_lots"),
        pl.col("ask_lots_1").alias("spot_ask_lots"),
        pl.col("ref_price").alias("spot_ref_price"),
        "spread_pair_epoch",
    )
    future_events = future.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        pl.col("recv_time_ns").cast(pl.Int64).alias("cursor_time_ns"),
        pl.lit(RAW_FUTURE_PHASE, dtype=pl.Int64).alias(
            "cursor_event_sequence"
        ),
        pl.col("sequence").cast(pl.Int64).alias("cursor_row_index"),
        pl.lit("future").alias("event_source"),
        pl.col("formal_after_trial").alias("future_formal"),
        pl.col("bid_price_1").alias("future_bid"),
        pl.col("ask_price_1").alias("future_ask"),
        pl.col("bid_lots_1").alias("future_bid_lots"),
        pl.col("ask_lots_1").alias("future_ask_lots"),
        pl.col("exec_bid_price").alias("future_exec_bid"),
        pl.col("exec_ask_price").alias("future_exec_ask"),
        pl.col("exec_bid_lots").alias("future_exec_bid_lots"),
        pl.col("exec_ask_lots").alias("future_exec_ask_lots"),
        pl.col("ref_price").alias("future_ref_price"),
    )
    del spot, future, spread_clock
    gc.collect()

    merged = pl.concat(
        [future_events, spot_events],
        how="diagonal_relaxed",
    ).sort(
        [
            "ValueCode",
            "cursor_time_ns",
            "cursor_event_sequence",
            "cursor_row_index",
        ]
    )
    state_columns = [
        name
        for name in merged.columns
        if name.startswith(("spot_", "future_"))
        or name == "spread_pair_epoch"
    ]
    merged = merged.with_columns(
        *(pl.col(name).forward_fill().over("ValueCode") for name in state_columns)
    )
    fair = pl.concat(
        [
            source.causal_fair.select(
                "ValueCode",
                pl.col("fair_timestamp")
                .cast(pl.Int64)
                .alias("fair_time_ns"),
                pl.col("anchor_ewma_120s_bp").cast(pl.Float64),
            )
            for source in batch.sources
        ],
        how="vertical_relaxed",
    ).sort(["ValueCode", "fair_time_ns"])
    merged = merged.join_asof(
        fair,
        left_on="cursor_time_ns",
        right_on="fair_time_ns",
        by="ValueCode",
        strategy="backward",
        check_sortedness=False,
    ).join(
        boundary_day.select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "upper_distance_bp",
        ),
        on=["Date", "ValueCode", "QuoteCode"],
        how="left",
        validate="m:1",
    )
    spot_book_ok = (
        (pl.col("spot_bid") > 0)
        & (pl.col("spot_ask") > 0)
        & (pl.col("spot_bid_lots") > 0)
        & (pl.col("spot_ask_lots") > 0)
        & (pl.col("spot_bid") <= pl.col("spot_ask"))
    ).fill_null(False)
    future_book_ok = (
        (pl.col("future_bid") > 0)
        & (pl.col("future_ask") > 0)
        & (pl.col("future_bid_lots") > 0)
        & (pl.col("future_ask_lots") > 0)
        & (pl.col("future_bid") <= pl.col("future_ask"))
    ).fill_null(False)
    future_exec_ok = (
        (pl.col("future_exec_bid") > 0)
        & (pl.col("future_exec_ask") > 0)
        & (pl.col("future_exec_bid_lots") > 0)
        & (pl.col("future_exec_ask_lots") > 0)
        & (pl.col("future_exec_bid") <= pl.col("future_exec_ask"))
    ).fill_null(False)
    spot_ref_ok = _strict_ref_band(
        "spot_ref_price", ("spot_bid", "spot_ask")
    )
    future_ref_ok = _strict_ref_band(
        "future_ref_price",
        (
            "future_bid",
            "future_ask",
            "future_exec_bid",
            "future_exec_ask",
        ),
    )
    eligible = (
        pl.col("spot_formal").fill_null(False)
        & pl.col("future_formal").fill_null(False)
        & spot_book_ok
        & future_book_ok
        & future_exec_ok
        & spot_ref_ok
        & future_ref_ok
        & pl.col("anchor_ewma_120s_bp").is_finite()
        & pl.col("upper_distance_bp").is_finite()
    ).fill_null(False)
    spot_mid = (pl.col("spot_bid") + pl.col("spot_ask")) / 2.0
    future_mid = (pl.col("future_bid") + pl.col("future_ask")) / 2.0
    merged = merged.with_columns(
        eligible.alias("analysis_eligible_raw"),
        pl.when(eligible)
        .then((future_mid / spot_mid - 1.0) * 10_000.0)
        .otherwise(None)
        .alias("basis_mid_bp"),
    ).with_columns(
        pl.when(pl.col("analysis_eligible_raw"))
        .then(pl.col("basis_mid_bp") - pl.col("anchor_ewma_120s_bp"))
        .otherwise(None)
        .alias("residual_excursion_bp")
    )

    group = ["Date", "ValueCode"]
    changed = merged.with_columns(
        pl.col("analysis_eligible_raw")
        .shift(1)
        .over(group)
        .alias("_previous_eligible"),
        pl.col("residual_excursion_bp")
        .shift(1)
        .over(group)
        .alias("_previous_residual"),
    ).filter(
        pl.col("_previous_eligible").is_null()
        | (
            pl.col("analysis_eligible_raw")
            != pl.col("_previous_eligible")
        )
        | (
            pl.col("analysis_eligible_raw")
            & (
                (
                    pl.col("residual_excursion_bp")
                    - pl.col("_previous_residual")
                ).abs()
                > 1e-12
            ).fill_null(True)
        )
    )
    return changed.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "cursor_time_ns",
        "cursor_event_sequence",
        "cursor_row_index",
        "event_source",
        "analysis_eligible_raw",
        "basis_mid_bp",
        "anchor_ewma_120s_bp",
        "residual_excursion_bp",
        "upper_distance_bp",
        "spread_pair_epoch",
    ).sort(
        [
            "ValueCode",
            "cursor_time_ns",
            "cursor_event_sequence",
            "cursor_row_index",
        ]
    )


def build_quote_only_orders(
    date: str,
    events: pl.DataFrame,
    outcomes: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Replay q95 intents, potential fills, and requests in one event loop.

    At each timestamp, request assignments are frozen first.  A potential
    legacy full fill then terminates an already-working order in phase 3,
    effective cancels apply in phase 5, and assigned new requests become
    working in phase 6.  Thus an unassigned pending cancel can be removed when
    a fill wins, while a cancel already assigned at that timestamp still
    consumes a request and becomes a no-op.
    """

    selected = events.filter(
        (pl.col("scenario_id") == SCENARIO_ID)
        & (pl.col("Date").cast(pl.String) == str(date))
    )
    submits = selected.filter(
        (pl.col("kind") == "submit")
        & (pl.col("second_from_open") < ENTRY_DRAIN_START_SECOND)
    )
    keys = ["Date", "ValueCode", "generation"]
    if submits.select(*keys).n_unique() != submits.height:
        raise ValueError(f"{date}: retained q95 submits are duplicated")
    keep_keys = submits.select(*keys)
    cancels = (
        selected.filter(pl.col("kind") == "cancel")
        .join(
            keep_keys,
            on=["Date", "ValueCode", "generation"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            pl.when(
                pl.col("second_from_open") >= ENTRY_DRAIN_START_SECOND
            )
            .then(pl.lit(ENTRY_DRAIN_START_SECOND))
            .otherwise(pl.col("second_from_open"))
            .cast(pl.Int32)
            .alias("adapter_second_from_open"),
            (
                pl.col("second_from_open") >= ENTRY_DRAIN_START_SECOND
            ).alias("cutoff_drain"),
        )
    )
    if (
        submits.height != cancels.height
        or cancels.select(*keys).n_unique() != cancels.height
    ):
        raise ValueError(f"{date}: retained q95 generations are not paired")

    base_ns = _session_second_ns(date, 0)
    submit_meta = submits.select(
        *keys,
        pl.col("second_from_open").cast(pl.Int32).alias("_new_second"),
        pl.col("reason").alias("_new_reason"),
        pl.col("absolute_price_tick").cast(pl.Int64).alias(
            "_new_price_tick"
        ),
    )
    cancel_meta = cancels.select(
        *keys,
        pl.col("adapter_second_from_open")
        .cast(pl.Int32)
        .alias("_cancel_intent_second"),
        pl.col("reason").alias("_cancel_reason"),
        pl.col("absolute_price_tick").cast(pl.Int64).alias(
            "_cancel_price_tick"
        ),
        "cutoff_drain",
    )
    outcome_day = outcomes.filter(
        pl.col("Date").cast(pl.String) == str(date)
    ).rename({"outcome_status": "legacy_nominal_outcome_status"})
    if outcome_day.select(*keys).n_unique() != outcome_day.height:
        raise ValueError(f"{date}: legacy candidate outcomes are duplicated")
    candidates = (
        outcome_day.join(submit_meta, on=keys, how="inner", validate="1:1")
        .join(cancel_meta, on=keys, how="inner", validate="1:1")
        .with_columns(
            (
                pl.lit(base_ns, dtype=pl.Int64)
                + pl.col("_new_second").cast(pl.Int64) * ONE_SECOND_NS
            ).alias("_new_intent_time_ns"),
            (
                pl.lit(base_ns, dtype=pl.Int64)
                + pl.col("_cancel_intent_second").cast(pl.Int64)
                * ONE_SECOND_NS
            ).alias("adapter_cancel_intent_time_ns"),
        )
        .sort(["ValueCode", "generation"])
    )
    if candidates.height != submits.height:
        raise ValueError(f"{date}: actual order/outcome reconciliation failed")
    candidate_mismatch = candidates.filter(
        (pl.col("absolute_price_tick") != pl.col("_new_price_tick"))
        | (pl.col("absolute_price_tick") != pl.col("_cancel_price_tick"))
        | (
            pl.col("submit_decision_time_ns")
            != pl.col("_new_intent_time_ns")
        )
    )
    if not candidate_mismatch.is_empty():
        raise ValueError(
            f"{date}: actual-new snapshot differs from legacy makerFill snapshot"
        )

    expiry_ns = _session_second_ns(date, SESSION_END_SECOND)
    scheduler = RollingVenueScheduler("spot", SPOT_REQUEST_CAP)
    lifecycle: dict[tuple[str, str, int], dict[str, object]] = {}
    intent_events: dict[int, list[dict[str, object]]] = {}
    potential_events: dict[int, list[tuple[str, str, int]]] = {}
    request_metadata: dict[str, dict[str, object]] = {}
    timeline: list[int] = []
    queued_times: set[int] = set()
    processed_times: set[int] = set()

    def schedule_time(time_ns: int) -> None:
        if time_ns in queued_times or time_ns in processed_times:
            return
        heapq.heappush(timeline, time_ns)
        queued_times.add(time_ns)

    for row in candidates.iter_rows(named=True):
        value_code = str(row["ValueCode"])
        generation = int(row["generation"])
        key = (str(date), value_code, generation)
        new_request_id = f"{date}/{value_code}/{generation}/new"
        cancel_request_id = f"{date}/{value_code}/{generation}/cancel"
        new_time_ns = int(row["_new_intent_time_ns"])
        cancel_time_ns = int(row["adapter_cancel_intent_time_ns"])
        potential_value = row["makerfill_implied_fill_time_ns"]
        potential_time_ns = (
            None if potential_value is None else int(potential_value)
        )
        lifecycle[key] = {
            "state": "intent_pending",
            "new_request_id": new_request_id,
            "cancel_request_id": cancel_request_id,
            "actual_new_time_ns": None,
            "actual_new_row_index": None,
            "cancel_request_sent": False,
            "cancel_request_send_time_ns": None,
            "cancel_request_send_row_index": None,
            "cancel_not_needed": False,
            "cancel_pending_removed": False,
            "cancel_send_effective": False,
            "cancel_send_noop": False,
            "cancel_terminal_status": "intent_pending",
            "actual_cancel_time_ns": None,
            "actual_cancel_row_index": None,
            "active_end_time_ns": None,
            "active_end_event_sequence": None,
            "active_end_row_index": None,
            "active_end_reason": None,
            "potential_fill_time_ns": potential_time_ns,
            "potential_fill_processed": False,
            "potential_fill_accepted": False,
            "potential_fill_rejection_reason": None,
            "outcome_supported": bool(row["outcome_supported"]),
        }
        stable_id = f"{value_code}/{generation:09d}"
        new_intent = VenueRequestIntent(
            request_id=new_request_id,
            venue="spot",
            request_class="new",
            original_cursor=EventCursor(
                new_time_ns,
                REQUEST_ASSIGNMENT_PHASE,
                0,
            ),
            stable_id=stable_id,
        )
        cancel_intent = VenueRequestIntent(
            request_id=cancel_request_id,
            venue="spot",
            request_class="cancel",
            original_cursor=EventCursor(
                cancel_time_ns,
                REQUEST_ASSIGNMENT_PHASE,
                0,
            ),
            stable_id=stable_id,
            cutoff_drain=bool(row["cutoff_drain"]),
            maker_side=(MAKER_SIDE if bool(row["cutoff_drain"]) else None),
            absolute_price_tick=(
                int(row["absolute_price_tick"])
                if bool(row["cutoff_drain"])
                else None
            ),
        )
        for kind, intent, reason in (
            ("new", new_intent, str(row["_new_reason"])),
            ("cancel", cancel_intent, str(row["_cancel_reason"])),
        ):
            metadata = {
                "request_id": intent.request_id,
                "Date": str(date),
                "ValueCode": value_code,
                "generation": generation,
                "kind": kind,
                "reason": reason,
                "absolute_price_tick": int(row["absolute_price_tick"]),
                "original_time_ns": intent.original_cursor.recv_time_ns,
                "cutoff_drain": intent.cutoff_drain,
                "intent": intent,
                "key": key,
            }
            request_metadata[intent.request_id] = metadata
            intent_events.setdefault(
                intent.original_cursor.recv_time_ns, []
            ).append(metadata)
            schedule_time(intent.original_cursor.recv_time_ns)
        if potential_time_ns is not None:
            potential_events.setdefault(potential_time_ns, []).append(key)
            schedule_time(potential_time_ns)
    schedule_time(expiry_ns)

    assignment_rows: list[dict[str, object]] = []
    cancel_pending_removed_on_fill = 0

    def mark_cancel_not_needed(
        order: dict[str, object],
        reason: str,
    ) -> None:
        if bool(order["cancel_request_sent"]):
            return
        order["cancel_not_needed"] = True
        order["cancel_terminal_status"] = reason

    while timeline:
        current_ns = heapq.heappop(timeline)
        queued_times.remove(current_ns)
        processed_times.add(current_ns)

        for metadata in sorted(
            intent_events.get(current_ns, []),
            key=lambda value: str(value["request_id"]),
        ):
            order = lifecycle[metadata["key"]]  # type: ignore[index]
            if metadata["kind"] == "cancel" and order["state"] != "working":
                scheduler.cancel_pending(str(metadata["request_id"]))
                mark_cancel_not_needed(
                    order,
                    f"cancel_not_needed_{order['state']}",
                )
                continue
            scheduler.enqueue(metadata["intent"])  # type: ignore[arg-type]

        frozen = scheduler.dispatch(
            EventCursor(current_ns, REQUEST_ASSIGNMENT_PHASE, 2**62)
        )
        for assignment in frozen:
            metadata = request_metadata[assignment.request_id]
            order = lifecycle[metadata["key"]]  # type: ignore[index]
            if metadata["kind"] == "cancel":
                order["cancel_request_sent"] = True
                order["cancel_request_send_time_ns"] = current_ns
                order["cancel_request_send_row_index"] = (
                    assignment.send_sequence
                )

        for key in sorted(potential_events.get(current_ns, [])):
            order = lifecycle[key]
            order["potential_fill_processed"] = True
            if not bool(order["outcome_supported"]):
                order["potential_fill_rejection_reason"] = (
                    "outcome_unsupported"
                )
                continue
            if order["state"] != "working":
                order["potential_fill_rejection_reason"] = (
                    f"not_working_{order['state']}"
                )
                continue
            order["state"] = "filled"
            order["potential_fill_accepted"] = True
            order["active_end_time_ns"] = current_ns
            order["active_end_event_sequence"] = APPROXIMATE_FILL_PHASE
            order["active_end_row_index"] = 0
            order["active_end_reason"] = "accepted_approximate_fill"
            removed = scheduler.cancel_pending(
                str(order["cancel_request_id"])
            )
            if removed is not None:
                cancel_pending_removed_on_fill += 1
                order["cancel_pending_removed"] = True
                mark_cancel_not_needed(
                    order,
                    "cancel_not_needed_after_fill",
                )

        assignment_effects: dict[int, tuple[str, bool]] = {}
        for assignment in frozen:
            metadata = request_metadata[assignment.request_id]
            if metadata["kind"] != "cancel":
                continue
            order = lifecycle[metadata["key"]]  # type: ignore[index]
            if order["state"] == "working":
                order["state"] = "actual_cancelled"
                order["cancel_send_effective"] = True
                order["cancel_terminal_status"] = "effective_actual_cancel"
                order["actual_cancel_time_ns"] = current_ns
                order["actual_cancel_row_index"] = assignment.send_sequence
                order["active_end_time_ns"] = current_ns
                order["active_end_event_sequence"] = ACTUAL_CANCEL_PHASE
                order["active_end_row_index"] = assignment.send_sequence
                order["active_end_reason"] = "actual_cancel_send"
                assignment_effects[assignment.send_sequence] = (
                    "effective_actual_cancel",
                    True,
                )
            else:
                order["cancel_send_noop"] = True
                status = f"assigned_cancel_noop_{order['state']}"
                order["cancel_terminal_status"] = status
                assignment_effects[assignment.send_sequence] = (status, False)

        if current_ns == expiry_ns:
            for order in lifecycle.values():
                if order["state"] != "working":
                    continue
                order["state"] = "session_expired"
                order["active_end_time_ns"] = expiry_ns
                order["active_end_event_sequence"] = ACTUAL_CANCEL_PHASE
                order["active_end_row_index"] = 0
                order["active_end_reason"] = "session_expiry"
                removed = scheduler.cancel_pending(
                    str(order["cancel_request_id"])
                )
                if removed is not None:
                    mark_cancel_not_needed(
                        order,
                        "cancel_not_needed_after_expiry",
                    )

        for assignment in frozen:
            metadata = request_metadata[assignment.request_id]
            if metadata["kind"] != "new":
                continue
            order = lifecycle[metadata["key"]]  # type: ignore[index]
            expected_time = int(metadata["original_time_ns"])
            if current_ns != expected_time:
                raise ValueError(
                    f"{date}: S0 controller assumption failed; actual new "
                    "was delayed"
                )
            if order["state"] != "intent_pending":
                raise ValueError(f"{date}: new assignment has invalid state")
            if current_ns >= expiry_ns:
                raise ValueError(f"{date}: new assignment reached session expiry")
            order["state"] = "working"
            order["actual_new_time_ns"] = current_ns
            order["actual_new_row_index"] = assignment.send_sequence
            assignment_effects[assignment.send_sequence] = (
                "new_became_working",
                True,
            )

        for assignment in frozen:
            metadata = request_metadata[assignment.request_id]
            effect_status, request_effective = assignment_effects[
                assignment.send_sequence
            ]
            assignment_rows.append(
                {
                    **{
                        name: value
                        for name, value in metadata.items()
                        if name not in {"intent", "key"}
                    },
                    "actual_send_time_ns": current_ns,
                    "actual_send_event_sequence": REQUEST_ASSIGNMENT_PHASE,
                    "actual_send_row_index": assignment.send_sequence,
                    "send_sequence": assignment.send_sequence,
                    "queue_delay_ns": assignment.queue_delay_ns,
                    "effect_status": effect_status,
                    "request_effective": request_effective,
                }
            )

        next_token_ns = scheduler.next_token_time_ns()
        if next_token_ns is not None:
            if next_token_ns <= current_ns:
                raise RuntimeError("scheduler next token did not advance")
            schedule_time(next_token_ns)

    if scheduler.pending_count:
        raise RuntimeError("scheduler retained pending requests after replay")

    lifecycle_rows: list[dict[str, object]] = []
    for key, order in lifecycle.items():
        if order["actual_new_time_ns"] is None:
            raise ValueError(f"{date}: retained new request was never sent")
        if order["state"] not in {
            "filled",
            "actual_cancelled",
            "session_expired",
        }:
            raise ValueError(f"{date}: order lifecycle is not terminal")
        if order["active_end_time_ns"] is None:
            raise ValueError(f"{date}: terminal order lacks active end")
        potential_time_ns = order["potential_fill_time_ns"]
        accepted = bool(order["potential_fill_accepted"])
        new_cursor = (
            int(order["actual_new_time_ns"]),
            ACTUAL_NEW_PHASE,
            int(order["actual_new_row_index"]),
        )
        end_cursor = (
            int(order["active_end_time_ns"]),
            int(order["active_end_event_sequence"]),
            int(order["active_end_row_index"]),
        )
        if accepted:
            adapter_outcome_status = "accepted_approximate_fill"
        else:
            active_end_reason = str(order["active_end_reason"])
            if active_end_reason == "actual_cancel_send":
                terminal_status = "effective_actual_cancel"
            elif active_end_reason == "session_expiry":
                terminal_status = "session_expiry"
            else:
                raise ValueError(
                    f"{date}: non-fill terminal has unknown active end"
                )
            if potential_time_ns is None:
                adapter_outcome_status = (
                    f"{terminal_status}_no_potential_fill"
                )
            elif not bool(order["outcome_supported"]):
                adapter_outcome_status = (
                    f"{terminal_status}_with_unsupported_potential_fill"
                )
            else:
                potential_cursor = (
                    int(potential_time_ns),
                    APPROXIMATE_FILL_PHASE,
                    0,
                )
                if potential_cursor <= new_cursor:
                    adapter_outcome_status = (
                        f"{terminal_status}_after_pre_working_potential_fill"
                    )
                elif potential_cursor > end_cursor:
                    adapter_outcome_status = (
                        f"{terminal_status}_before_potential_fill"
                    )
                else:
                    raise ValueError(
                        f"{date}: supported in-interval potential was rejected"
                    )
        lifecycle_rows.append(
            {
                "Date": key[0],
                "ValueCode": key[1],
                "generation": key[2],
                **order,
                "actual_new_event_sequence": ACTUAL_NEW_PHASE,
                "cancel_request_send_event_sequence": (
                    REQUEST_ASSIGNMENT_PHASE
                    if order["cancel_request_send_time_ns"] is not None
                    else None
                ),
                "actual_cancel_event_sequence": (
                    ACTUAL_CANCEL_PHASE
                    if order["actual_cancel_time_ns"] is not None
                    else None
                ),
                "potential_fill_event_sequence": (
                    APPROXIMATE_FILL_PHASE
                    if potential_time_ns is not None
                    else None
                ),
                "potential_fill_row_index": (
                    0 if potential_time_ns is not None else None
                ),
                "approximate_fill_time_ns": (
                    potential_time_ns if accepted else None
                ),
                "approximate_fill_event_sequence": (
                    APPROXIMATE_FILL_PHASE if accepted else None
                ),
                "approximate_fill_row_index": 0 if accepted else None,
                "approximate_fill_within_actual_active_interval": accepted,
                "adapter_outcome_status": adapter_outcome_status,
            }
        )
    lifecycle_frame = pl.from_dicts(
        lifecycle_rows,
        schema_overrides={
            "actual_new_time_ns": pl.Int64,
            "actual_new_row_index": pl.Int64,
            "cancel_request_send_time_ns": pl.Int64,
            "cancel_request_send_event_sequence": pl.Int64,
            "cancel_request_send_row_index": pl.Int64,
            "actual_cancel_time_ns": pl.Int64,
            "actual_cancel_event_sequence": pl.Int64,
            "actual_cancel_row_index": pl.Int64,
            "active_end_time_ns": pl.Int64,
            "active_end_event_sequence": pl.Int64,
            "active_end_row_index": pl.Int64,
            "potential_fill_time_ns": pl.Int64,
            "potential_fill_event_sequence": pl.Int64,
            "potential_fill_row_index": pl.Int64,
            "approximate_fill_time_ns": pl.Int64,
            "approximate_fill_event_sequence": pl.Int64,
            "approximate_fill_row_index": pl.Int64,
        },
        infer_schema_length=None,
    )
    assignment_frame = pl.from_dicts(
        assignment_rows,
        schema_overrides={
            "generation": pl.Int64,
            "absolute_price_tick": pl.Int64,
            "original_time_ns": pl.Int64,
            "actual_send_time_ns": pl.Int64,
            "actual_send_event_sequence": pl.Int64,
            "actual_send_row_index": pl.Int64,
            "send_sequence": pl.Int64,
            "queue_delay_ns": pl.Int64,
        },
        infer_schema_length=None,
    ).sort(["actual_send_time_ns", "send_sequence"])
    delayed_new = assignment_frame.filter(
        (pl.col("kind") == "new") & (pl.col("queue_delay_ns") > 0)
    )
    if not delayed_new.is_empty():
        raise ValueError(
            f"{date}: S0 controller assumption failed; actual new was delayed"
        )

    joined = candidates.join(
        lifecycle_frame,
        on=keys,
        how="inner",
        validate="1:1",
    ).with_columns(
        pl.col("_new_intent_time_ns").alias("nominal_new_time_ns"),
        pl.concat_str(
            "Date",
            "ValueCode",
            pl.col("generation").cast(pl.String),
            separator="/",
        ).alias("candidate_intent_id"),
        pl.struct(
            "Date",
            "ValueCode",
            "QuoteCode",
            "absolute_price_tick",
            "actual_new_time_ns",
            "actual_new_event_sequence",
            "actual_new_row_index",
        )
        .map_elements(_raw_order_id, return_dtype=pl.String)
        .alias("raw_order_fact_id"),
        pl.lit(ROUTE_ID).alias("route"),
        pl.lit(STAGE_ID).alias("stage"),
        pl.lit(MAKER_SIDE).alias("maker_side"),
        pl.lit(True).alias("diagnostic_quote_only"),
        pl.lit(True).alias("capital_cap_infinite"),
        pl.lit(True).alias("fills_remove_live_quotes"),
        pl.lit(expiry_ns, dtype=pl.Int64).alias("session_expiry_time_ns"),
        pl.lit(ACTUAL_CANCEL_PHASE, dtype=pl.Int64).alias(
            "session_expiry_event_sequence"
        ),
        pl.lit(0, dtype=pl.Int64).alias("session_expiry_row_index"),
    )
    order_columns = (
        "raw_order_fact_id",
        "candidate_intent_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "stage",
        "maker_side",
        "generation",
        "absolute_price_tick",
        "target_price",
        "submit_point_offset",
        "exact_target_rank",
        "initial_displayed_lots",
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "boundary_source_asof_date",
        "new_request_id",
        "cancel_request_id",
        "nominal_new_time_ns",
        "nominal_stop_time_ns",
        "nominal_stop_reason",
        "adapter_cancel_intent_time_ns",
        "actual_new_time_ns",
        "actual_new_event_sequence",
        "actual_new_row_index",
        "cancel_request_sent",
        "cancel_request_send_time_ns",
        "cancel_request_send_event_sequence",
        "cancel_request_send_row_index",
        "cancel_not_needed",
        "cancel_pending_removed",
        "cancel_send_effective",
        "cancel_send_noop",
        "cancel_terminal_status",
        "actual_cancel_time_ns",
        "actual_cancel_event_sequence",
        "actual_cancel_row_index",
        "session_expiry_time_ns",
        "session_expiry_event_sequence",
        "session_expiry_row_index",
        "active_end_time_ns",
        "active_end_event_sequence",
        "active_end_row_index",
        "active_end_reason",
        "cutoff_drain",
        "outcome_supported",
        "legacy_nominal_outcome_status",
        "adapter_outcome_status",
        "makerfill_mapping_exact",
        "makerfill_fill_seconds",
        "potential_fill_time_ns",
        "potential_fill_event_sequence",
        "potential_fill_row_index",
        "potential_fill_processed",
        "potential_fill_accepted",
        "potential_fill_rejection_reason",
        "approximate_fill_time_ns",
        "approximate_fill_event_sequence",
        "approximate_fill_row_index",
        "approximate_fill_within_actual_active_interval",
        "fill_cursor_exact",
        "own_quantity_included",
        "partial_fill_included",
        "cancel_ack_observed",
        "joint_volume_allocated",
        "diagnostic_quote_only",
        "capital_cap_infinite",
        "fills_remove_live_quotes",
    )
    orders = joined.select(*order_columns).sort(
        ["Date", "ValueCode", "generation"]
    )
    audit = {
        "source_requests": selected.height,
        "source_new": selected.filter(pl.col("kind") == "submit").height,
        "adapter_requests": assignment_frame.height,
        "adapter_new": submits.height,
        "adapter_cancel_requests_sent": assignment_frame.filter(
            pl.col("kind") == "cancel"
        ).height,
        "suppressed_last_two_second_new": (
            selected.filter(
                (pl.col("kind") == "submit")
                & (
                    pl.col("second_from_open")
                    >= ENTRY_DRAIN_START_SECOND
                )
            ).height
        ),
        "actual_new_delayed": delayed_new.height,
        "cutoff_drain_cancels": cancels.filter(pl.col("cutoff_drain")).height,
        "potential_fill_events": orders.filter(
            pl.col("potential_fill_time_ns").is_not_null()
        ).height,
        "accepted_approximate_fills": orders.filter(
            pl.col("potential_fill_accepted")
        ).height,
        "rejected_potential_fills": orders.filter(
            pl.col("potential_fill_time_ns").is_not_null()
            & ~pl.col("potential_fill_accepted")
        ).height,
        "cancel_not_needed": orders.filter(
            pl.col("cancel_not_needed")
        ).height,
        "cancel_pending_removed_on_fill": cancel_pending_removed_on_fill,
        "cancel_assigned_noop": orders.filter(
            pl.col("cancel_send_noop")
        ).height,
        "effective_actual_cancels": orders.filter(
            pl.col("cancel_send_effective")
        ).height,
        "session_expiries": orders.filter(
            pl.col("active_end_reason") == "session_expiry"
        ).height,
        "max_actual_send_second_requests": int(
            assignment_frame.group_by("actual_send_time_ns")
            .len()["len"]
            .max()
            or 0
        ),
        "last_actual_cancel_time_ns": int(
            orders["actual_cancel_time_ns"].max() or 0
        ),
    }
    return orders, assignment_frame, audit


def _add_formal_state(frame: pl.DataFrame) -> pl.DataFrame:
    signal = (
        pl.when(pl.col("trial_match"))
        .then(pl.lit(0, dtype=pl.Int8))
        .when(~pl.col("trial_match") & pl.col("raw_has_book"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(None)
    )
    return frame.sort(
        ["ValueCode", "recv_time_ns", "sequence", "packet_sequence"]
    ).with_columns(
        (signal.forward_fill().over("ValueCode").fill_null(0) == 1).alias(
            "formal_after_trial"
        )
    )


def _strict_ref_band(reference: str, prices: Iterable[str]) -> pl.Expr:
    result = pl.col(reference).is_finite() & (pl.col(reference) > 0)
    for price in prices:
        result = (
            result
            & (pl.col(price) > pl.col(reference) * 0.91)
            & (pl.col(price) < pl.col(reference) * 1.08)
        )
    return result.fill_null(False)


def _raw_order_id(row: Mapping[str, object]) -> str:
    payload = "/".join(
        (
            str(row["Date"]),
            str(row["ValueCode"]),
            str(row["QuoteCode"]),
            ROUTE_ID,
            STAGE_ID,
            MAKER_SIDE,
            str(row["absolute_price_tick"]),
            str(row["actual_new_time_ns"]),
            str(row["actual_new_event_sequence"]),
            str(row["actual_new_row_index"]),
            "start_cursor_identity_v1",
        )
    )
    return "raw-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:28]


def _session_second_ns(date: str, second: int) -> int:
    parsed = datetime.strptime(str(date), "%Y%m%d").replace(
        tzinfo=ZoneInfo("Asia/Taipei")
    )
    local = parsed + timedelta(hours=9, seconds=int(second))
    return int(local.timestamp() * ONE_SECOND_NS)


def run(
    paths: AttributionPaths | None = None,
    *,
    dates: Sequence[str] | None = None,
    run_focused_tests: bool = True,
) -> Mapping[str, object]:
    """Build a fresh immutable S0 attribution bundle."""

    paths = AttributionPaths() if paths is None else paths
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
    manifest, boundary_rows = load_manifest_and_boundaries(
        paths.manifest_path,
        paths.boundary_path,
        dates=dates,
    )
    selected_dates = manifest["Date"].unique().sort().to_list()

    excursion_parts: list[pl.DataFrame] = []
    order_parts: list[pl.DataFrame] = []
    assignment_parts: list[pl.DataFrame] = []
    coverage_parts: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for index, date in enumerate(selected_dates, start=1):
        day_started = time.perf_counter()
        manifest_day = manifest.filter(pl.col("Date") == date)
        boundary_day = boundary_rows.filter(pl.col("Date") == date)
        event_path = (
            paths.message_root / f"Date={date}" / "message_events.parquet"
        )
        outcome_path = (
            paths.outcome_root
            / f"Date={date}"
            / "candidate_outcomes.parquet"
        )
        for required in (event_path, outcome_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        events = pl.read_parquet(event_path)
        outcomes = pl.read_parquet(outcome_path)
        orders, assignments, order_audit = build_quote_only_orders(
            date,
            events,
            outcomes,
        )
        order_parts.append(orders)
        assignment_parts.append(assignments)
        del events, outcomes
        gc.collect()

        codes = manifest_day["ValueCode"].sort().to_list()
        batch = load_walkforward_execution_day_batch(
            date,
            codes,
            daily_root=paths.daily_root,
            boundary_snapshot_path=paths.boundary_path,
            data_root=paths.data_root,
            quantiles=(BOUNDARY_QUANTILE,),
        )
        raw_spot_rows = batch.raw_tape.spot_states.height
        raw_future_rows = batch.raw_tape.future_states.height
        raw_states = build_raw_residual_states(batch, boundary_day)
        cutoff_ns = _session_second_ns(date, ENTRY_CUTOFF_SECOND)
        primary_states = raw_states.filter(
            pl.col("cursor_time_ns") < cutoff_ns
        )
        primary_excursions = extract_positive_excursions(
            primary_states
        ).with_columns(
            pl.concat_str(
                pl.lit("entry_primary"),
                pl.col("excursion_id"),
                separator=":",
            ).alias("excursion_id"),
            pl.lit("entry_primary").alias("analysis_window"),
        )
        session_excursions = extract_positive_excursions(
            raw_states
        ).with_columns(
            pl.concat_str(
                pl.lit("session_1320_sensitivity"),
                pl.col("excursion_id"),
                separator=":",
            ).alias("excursion_id"),
            pl.lit("session_1320_sensitivity").alias("analysis_window"),
        )
        excursion_parts.extend((primary_excursions, session_excursions))

        primary_counts = primary_states.group_by(
            "Date", "ValueCode", "QuoteCode"
        ).agg(
            pl.col("analysis_eligible_raw")
            .sum()
            .cast(pl.Int64)
            .alias("eligible_raw_state_count")
        )
        session_counts = raw_states.group_by(
            "Date", "ValueCode", "QuoteCode"
        ).agg(
            pl.col("analysis_eligible_raw")
            .sum()
            .cast(pl.Int64)
            .alias("session_eligible_raw_state_count")
        )
        coverage = (
            boundary_day.select(
                "Date",
                "ValueCode",
                "QuoteCode",
                "upper_distance_bp",
                "lower_distance_bp",
                "source_asof_date",
                "history_sessions_global",
                "history_sessions_product",
                "lookback_sessions",
                "parameter_version",
            )
            .join(
                primary_counts,
                on=["Date", "ValueCode", "QuoteCode"],
                how="left",
                validate="1:1",
            )
            .join(
                session_counts,
                on=["Date", "ValueCode", "QuoteCode"],
                how="left",
                validate="1:1",
            )
            .with_columns(
                pl.col("eligible_raw_state_count").fill_null(0),
                pl.col("session_eligible_raw_state_count").fill_null(0),
            )
        )
        coverage_parts.append(coverage)
        audit_rows.append(
            {
                "Date": date,
                "selected_products": manifest_day.height,
                "raw_spot_rows": raw_spot_rows,
                "raw_future_rows": raw_future_rows,
                "raw_residual_change_rows": raw_states.height,
                "primary_residual_change_rows": primary_states.height,
                "primary_eligible_product_days": coverage.filter(
                    pl.col("eligible_raw_state_count") > 0
                ).height,
                "session_eligible_product_days": coverage.filter(
                    pl.col("session_eligible_raw_state_count") > 0
                ).height,
                "primary_excursions": primary_excursions.height,
                "session_sensitivity_excursions": session_excursions.height,
                **order_audit,
                "elapsed_seconds": time.perf_counter() - day_started,
            }
        )
        print(
            f"[{index:02d}/{len(selected_dates):02d}] {date}: "
            f"{manifest_day.height} products, "
            f"{primary_excursions.height:,} primary excursions, "
            f"{orders.height:,} actual working orders",
            flush=True,
        )
        del (
            batch,
            raw_states,
            primary_states,
            primary_excursions,
            session_excursions,
            coverage,
            orders,
            assignments,
        )
        gc.collect()

    excursions = pl.concat(excursion_parts, how="vertical_relaxed").sort(
        ["analysis_window", "Date", "ValueCode", "excursion_sequence"]
    )
    primary_excursions = excursions.filter(
        pl.col("analysis_window") == "entry_primary"
    )
    raw_orders = pl.concat(order_parts, how="vertical_relaxed").sort(
        ["Date", "ValueCode", "generation"]
    )
    assignments = pl.concat(
        assignment_parts, how="vertical_relaxed"
    ).sort(["Date", "actual_send_time_ns", "send_sequence"])
    product_days = pl.concat(
        coverage_parts, how="vertical_relaxed"
    ).sort(["Date", "ValueCode"])
    touch_pairs = link_touch_order_pairs(
        primary_excursions,
        raw_orders,
    )
    touch_links = deduplicate_touch_pairs_to_orders(touch_pairs)
    monthly = summarize_monthly_attribution(
        product_days,
        primary_excursions,
        raw_orders,
        touch_links,
        touch_pairs=touch_pairs,
    )
    rank_summary = summarize_post_touch_by_rank(touch_links)
    decomposition = build_decomposition(monthly) if "202608" in set(
        monthly["month"].to_list()
    ) and monthly.filter(pl.col("month") < "202608").height else pl.DataFrame()
    membership = _membership_sensitivity(
        product_days,
        primary_excursions,
        raw_orders,
        touch_pairs,
        touch_links,
    )
    daily_audit = pl.from_dicts(
        audit_rows,
        infer_schema_length=None,
    ).sort("Date")

    destination = paths.output_root
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        frames = {
            "market_excursions.parquet": excursions,
            "raw_order_facts.parquet": raw_orders,
            "touch_order_attribution.parquet": touch_pairs,
            "touched_raw_order_facts.parquet": touch_links,
            "request_assignments.parquet": assignments,
            "product_day_coverage.parquet": product_days,
            "monthly_attribution.csv": monthly,
            "post_touch_fill_by_rank.csv": rank_summary,
            "membership_sensitivity.csv": membership,
            "decomposition.csv": decomposition,
            "daily_input_audit.csv": daily_audit,
        }
        for name, frame in frames.items():
            output = stage / name
            if output.suffix == ".parquet":
                frame.write_parquet(
                    output,
                    compression="zstd",
                    statistics=True,
                )
            else:
                frame.write_csv(output)
        _write_dual_panel(monthly, stage / "august_attribution_dual_panel.png")

        input_inventory = _input_inventory(paths, selected_dates)
        canonical_checks = _canonical_checks(
            paths,
            requested_dates=dates,
            selected_dates=selected_dates,
            manifest=manifest,
            git_state=git_state,
            tests=tests,
            input_inventory=input_inventory,
        )
        run_config = _run_config(
            paths,
            selected_dates,
            manifest,
            git_state,
            tests,
            input_inventory,
            canonical_checks,
        )
        _write_json(stage / "run_config.json", run_config)
        verification = _verify_domain_frames(
            manifest,
            product_days,
            excursions,
            raw_orders,
            touch_pairs,
            touch_links,
            assignments,
            monthly,
            rank_summary,
            membership,
            decomposition,
            daily_audit,
        )
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
            "nested_repo_commit": git_state["commit"],
            "dirty": git_state["dirty"],
            "canonical_eligible": all(canonical_checks.values()),
            "canonical_checks": canonical_checks,
            "diagnostic_quote_only": True,
            "capital_cap_infinite": True,
            "not_a_20m_strategy_replay": True,
            "date_count": len(selected_dates),
            "first_date": selected_dates[0],
            "last_date": selected_dates[-1],
            "entry_session_count": len(selected_dates),
            "product_day_count": manifest.height,
            "config_sha256": _canonical_sha256(run_config),
            "input_inventory_sha256": input_inventory[
                "inventory_sha256"
            ],
            "focused_tests": tests,
            "verification": verification,
            "artifacts": artifacts,
            "elapsed_seconds": time.perf_counter() - started,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        _write_json(stage / "complete.json", marker)
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return marker


def _membership_sensitivity(
    product_days: pl.DataFrame,
    excursions: pl.DataFrame,
    raw_orders: pl.DataFrame,
    touch_pairs: pl.DataFrame,
    touch_links: pl.DataFrame,
) -> pl.DataFrame:
    july = set(
        product_days.filter(pl.col("Date").str.starts_with("202607"))[
            "ValueCode"
        ].unique()
    )
    august = set(
        product_days.filter(pl.col("Date").str.starts_with("202608"))[
            "ValueCode"
        ].unique()
    )
    common = sorted(july & august)
    if not common:
        return pl.DataFrame()
    months = ["202607", "202608"]
    selected_days = product_days.filter(
        pl.col("Date").str.slice(0, 6).is_in(months)
        & pl.col("ValueCode").is_in(common)
    )
    selected_excursions = excursions.filter(
        pl.col("month").is_in(months) & pl.col("ValueCode").is_in(common)
    )
    selected_orders = raw_orders.filter(
        pl.col("Date").str.slice(0, 6).is_in(months)
        & pl.col("ValueCode").is_in(common)
    )
    selected_links = touch_links.filter(
        pl.col("month").is_in(months) & pl.col("ValueCode").is_in(common)
    )
    selected_pairs = touch_pairs.filter(
        pl.col("month").is_in(months) & pl.col("ValueCode").is_in(common)
    )
    return summarize_monthly_attribution(
        selected_days,
        selected_excursions,
        selected_orders,
        selected_links,
        touch_pairs=selected_pairs,
    ).with_columns(
        pl.lit("jul_aug_common_products").alias("sensitivity_id"),
        pl.lit(len(common), dtype=pl.Int64).alias("common_product_count"),
    ).select(
        "sensitivity_id",
        "common_product_count",
        pl.all().exclude("sensitivity_id", "common_product_count"),
    )


def _write_dual_panel(monthly: pl.DataFrame, path: Path) -> None:
    months = monthly["month"].to_list()
    labels = [f"{value[:4]}-{value[4:]}" for value in months]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    left = axes[0]
    for column, label, style in (
        ("residual_excursion_p50_bp", "Residual p50", "o-"),
        ("residual_excursion_p80_bp", "Residual p80", "s-"),
        ("residual_excursion_p95_bp", "Residual p95", "^-"),
        ("q95_boundary_p50_bp", "D-1 q95 boundary p50", "k--"),
    ):
        left.plot(labels, monthly[column].to_list(), style, label=label)
    left.set_title("A. Raw residual excursion vs D-1 q95")
    left.set_ylabel("basis points")
    left.grid(alpha=0.25)
    left.legend(fontsize=8)

    right = axes[1]
    right.plot(
        labels,
        [
            100.0 * float(value) if value is not None else None
            for value in monthly["excursion_touch_rate"]
        ],
        "o-",
        label="Excursion touch rate",
    )
    right.plot(
        labels,
        [
            100.0 * float(value) if value is not None else None
            for value in monthly["post_touch_fill_rate_pooled"]
        ],
        "s-",
        label="Post-touch fill rate (pooled)",
    )
    right.set_title("B. Market/boundary vs queue decomposition")
    right.set_ylabel("percent")
    right.grid(alpha=0.25)
    right.legend(fontsize=8)
    figure.suptitle(
        "S0 q95 attribution — conditional on monthly causal matched universe",
        fontsize=11,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _verify_domain_frames(
    manifest: pl.DataFrame,
    product_days: pl.DataFrame,
    excursions: pl.DataFrame,
    raw_orders: pl.DataFrame,
    touch_pairs: pl.DataFrame,
    touch_links: pl.DataFrame,
    assignments: pl.DataFrame,
    monthly: pl.DataFrame,
    rank_summary: pl.DataFrame,
    membership: pl.DataFrame,
    decomposition: pl.DataFrame,
    daily_audit: pl.DataFrame,
) -> dict[str, object]:
    """Recompute S0 invariants and every derived summary table."""

    keys = ["Date", "ValueCode", "QuoteCode"]
    if (
        product_days.select(*keys).n_unique() != product_days.height
        or product_days.height != manifest.height
        or not product_days.select(*keys).equals(manifest.select(*keys))
    ):
        raise ValueError("market product-day denominator drift")
    manifest_keys = manifest.select(*keys)
    for fact_name, facts in (
        ("excursions", excursions),
        ("raw orders", raw_orders),
        ("touch/order pairs", touch_pairs),
        ("touched raw orders", touch_links),
    ):
        outside = facts.select(*keys).unique().join(
            manifest_keys,
            on=keys,
            how="anti",
        )
        if not outside.is_empty():
            raise ValueError(f"{fact_name} contain facts outside manifest")
    assignment_keys = ["Date", "ValueCode"]
    outside_assignments = assignments.select(*assignment_keys).unique().join(
        manifest.select(*assignment_keys).unique(),
        on=assignment_keys,
        how="anti",
    )
    if not outside_assignments.is_empty():
        raise ValueError("request assignments contain facts outside manifest")
    if raw_orders["raw_order_fact_id"].n_unique() != raw_orders.height:
        raise ValueError("raw_order_fact_id is not unique")
    if raw_orders.select("Date", "ValueCode", "generation").n_unique() != raw_orders.height:
        raise ValueError("quote-only generation keys are duplicated")
    if excursions["excursion_id"].n_unique() != excursions.height:
        raise ValueError("excursion_id is not unique across analysis windows")
    required_order_columns = {
        "new_request_id",
        "cancel_request_id",
        "nominal_new_time_ns",
        "actual_new_time_ns",
        "actual_new_event_sequence",
        "actual_new_row_index",
        "cancel_request_sent",
        "cancel_request_send_time_ns",
        "cancel_request_send_event_sequence",
        "cancel_request_send_row_index",
        "cancel_not_needed",
        "cancel_pending_removed",
        "cancel_send_effective",
        "cancel_send_noop",
        "actual_cancel_time_ns",
        "actual_cancel_event_sequence",
        "actual_cancel_row_index",
        "active_end_time_ns",
        "active_end_event_sequence",
        "active_end_row_index",
        "active_end_reason",
        "legacy_nominal_outcome_status",
        "adapter_outcome_status",
        "potential_fill_time_ns",
        "potential_fill_event_sequence",
        "potential_fill_row_index",
        "potential_fill_processed",
        "potential_fill_accepted",
        "potential_fill_rejection_reason",
        "approximate_fill_time_ns",
        "approximate_fill_event_sequence",
        "approximate_fill_row_index",
        "approximate_fill_within_actual_active_interval",
    }
    missing_order_columns = sorted(
        required_order_columns - set(raw_orders.columns)
    )
    if missing_order_columns:
        raise ValueError(
            f"raw working-order schema missing: {missing_order_columns}"
        )

    required_assignment_columns = {
        "request_id",
        "Date",
        "ValueCode",
        "generation",
        "kind",
        "original_time_ns",
        "actual_send_time_ns",
        "actual_send_event_sequence",
        "actual_send_row_index",
        "send_sequence",
        "queue_delay_ns",
        "effect_status",
        "request_effective",
    }
    missing_assignment_columns = sorted(
        required_assignment_columns - set(assignments.columns)
    )
    if missing_assignment_columns:
        raise ValueError(
            f"request assignment schema missing: {missing_assignment_columns}"
        )
    if assignments["request_id"].n_unique() != assignments.height:
        raise ValueError("request assignments contain duplicate request IDs")
    if (
        assignments.select("Date", "send_sequence").n_unique()
        != assignments.height
    ):
        raise ValueError("request send sequence is duplicated within a day")
    invalid_assignment = assignments.filter(
        (pl.col("actual_send_event_sequence") != REQUEST_ASSIGNMENT_PHASE)
        | (pl.col("actual_send_row_index") != pl.col("send_sequence"))
        | (
            pl.col("queue_delay_ns")
            != pl.col("actual_send_time_ns") - pl.col("original_time_ns")
        )
        | (pl.col("queue_delay_ns") < 0)
        | ~pl.col("kind").is_in(["new", "cancel"])
        | ~pl.col("effect_status").is_in(
            [
                "new_became_working",
                "effective_actual_cancel",
                "assigned_cancel_noop_filled",
            ]
        )
    )
    if not invalid_assignment.is_empty():
        raise ValueError("request assignment invariants failed")

    assignment_lookup = {
        str(row["request_id"]): row
        for row in assignments.iter_rows(named=True)
    }
    expected_assignment_ids: set[str] = set()
    for order in raw_orders.iter_rows(named=True):
        new_cursor = (
            int(order["actual_new_time_ns"]),
            int(order["actual_new_event_sequence"]),
            int(order["actual_new_row_index"]),
        )
        end_cursor = (
            int(order["active_end_time_ns"]),
            int(order["active_end_event_sequence"]),
            int(order["active_end_row_index"]),
        )
        if not new_cursor < end_cursor:
            raise ValueError("raw working-order active cursor is invalid")
        if (
            int(order["actual_new_time_ns"])
            != int(order["nominal_new_time_ns"])
            or int(order["actual_new_event_sequence"]) != ACTUAL_NEW_PHASE
            or not bool(order["diagnostic_quote_only"])
            or not bool(order["capital_cap_infinite"])
            or not bool(order["fills_remove_live_quotes"])
            or bool(order["fill_cursor_exact"])
            or bool(order["own_quantity_included"])
            or bool(order["partial_fill_included"])
            or bool(order["cancel_ack_observed"])
            or bool(order["joint_volume_allocated"])
        ):
            raise ValueError("raw working-order static invariants failed")
        if str(order["raw_order_fact_id"]) != _raw_order_id(order):
            raise ValueError("raw_order_fact_id start-cursor identity drift")

        new_request_id = str(order["new_request_id"])
        expected_assignment_ids.add(new_request_id)
        new_assignment = assignment_lookup.get(new_request_id)
        if (
            new_assignment is None
            or new_assignment["kind"] != "new"
            or new_assignment["effect_status"] != "new_became_working"
            or not bool(new_assignment["request_effective"])
            or str(new_assignment["Date"]) != str(order["Date"])
            or str(new_assignment["ValueCode"])
            != str(order["ValueCode"])
            or int(new_assignment["generation"])
            != int(order["generation"])
            or int(new_assignment["original_time_ns"])
            != int(order["nominal_new_time_ns"])
            or int(new_assignment["actual_send_time_ns"])
            != int(order["actual_new_time_ns"])
            or int(new_assignment["actual_send_row_index"])
            != int(order["actual_new_row_index"])
        ):
            raise ValueError("actual new assignment does not match order fact")

        cancel_sent = bool(order["cancel_request_sent"])
        cancel_not_needed = bool(order["cancel_not_needed"])
        cancel_effective = bool(order["cancel_send_effective"])
        cancel_noop = bool(order["cancel_send_noop"])
        if sum((cancel_not_needed, cancel_effective, cancel_noop)) != 1:
            raise ValueError("cancel terminal classification is not exclusive")
        if cancel_sent != (cancel_effective or cancel_noop):
            raise ValueError("cancel sent/effect classification drift")
        if bool(order["cancel_pending_removed"]) and not cancel_not_needed:
            raise ValueError("removed pending cancel is not marked unnecessary")
        cancel_request_id = str(order["cancel_request_id"])
        cancel_assignment = assignment_lookup.get(cancel_request_id)
        if cancel_sent:
            expected_assignment_ids.add(cancel_request_id)
            if (
                cancel_assignment is None
                or cancel_assignment["kind"] != "cancel"
                or str(cancel_assignment["Date"]) != str(order["Date"])
                or str(cancel_assignment["ValueCode"])
                != str(order["ValueCode"])
                or int(cancel_assignment["generation"])
                != int(order["generation"])
                or int(cancel_assignment["original_time_ns"])
                != int(order["adapter_cancel_intent_time_ns"])
                or int(cancel_assignment["actual_send_time_ns"])
                != int(order["cancel_request_send_time_ns"])
                or int(cancel_assignment["send_sequence"])
                != int(order["cancel_request_send_row_index"])
                or int(order["cancel_request_send_event_sequence"])
                != REQUEST_ASSIGNMENT_PHASE
            ):
                raise ValueError(
                    "sent cancel assignment does not match order fact"
                )
            if bool(cancel_assignment["request_effective"]) != cancel_effective:
                raise ValueError("cancel assignment effect flag drift")
            expected_cancel_effect = (
                "effective_actual_cancel"
                if cancel_effective
                else "assigned_cancel_noop_filled"
            )
            if cancel_assignment["effect_status"] != expected_cancel_effect:
                raise ValueError("cancel assignment terminal effect drift")
            if cancel_noop and (
                order["active_end_reason"] != "accepted_approximate_fill"
                or int(order["cancel_request_send_time_ns"])
                != int(order["active_end_time_ns"])
            ):
                raise ValueError("assigned cancel no-op was not same-cursor fill")
        else:
            if cancel_assignment is not None:
                raise ValueError(
                    "cancel_not_needed request was nevertheless sent"
                )
            if any(
                order[name] is not None
                for name in (
                    "cancel_request_send_time_ns",
                    "cancel_request_send_event_sequence",
                    "cancel_request_send_row_index",
                )
            ):
                raise ValueError("unsent cancel retained an actual send cursor")

        actual_cancel_exists = order["actual_cancel_time_ns"] is not None
        if actual_cancel_exists != cancel_effective:
            raise ValueError("effective actual cancel cursor drift")
        if cancel_effective:
            actual_cancel = (
                int(order["actual_cancel_time_ns"]),
                int(order["actual_cancel_event_sequence"]),
                int(order["actual_cancel_row_index"]),
            )
            if (
                actual_cancel != end_cursor
                or order["active_end_reason"] != "actual_cancel_send"
                or int(order["actual_cancel_event_sequence"])
                != ACTUAL_CANCEL_PHASE
            ):
                raise ValueError("effective cancel is not the active end")
        elif any(
            order[name] is not None
            for name in (
                "actual_cancel_event_sequence",
                "actual_cancel_row_index",
            )
        ):
            raise ValueError("non-effective cancel retained an actual cursor")

        potential_exists = order["potential_fill_time_ns"] is not None
        potential_accepted = bool(order["potential_fill_accepted"])
        if bool(order["potential_fill_processed"]) != potential_exists:
            raise ValueError("potential fill processing flag drift")
        if potential_exists:
            potential_cursor = (
                int(order["potential_fill_time_ns"]),
                int(order["potential_fill_event_sequence"]),
                int(order["potential_fill_row_index"]),
            )
            if potential_cursor[1] != APPROXIMATE_FILL_PHASE:
                raise ValueError("potential fill phase drift")
        else:
            potential_cursor = None
            if any(
                order[name] is not None
                for name in (
                    "potential_fill_event_sequence",
                    "potential_fill_row_index",
                    "potential_fill_rejection_reason",
                )
            ):
                raise ValueError("absent potential fill retained cursor data")
        approximate_exists = order["approximate_fill_time_ns"] is not None
        if potential_accepted != approximate_exists:
            raise ValueError("accepted potential/approximate fill drift")
        expected_potential_accept = bool(
            potential_exists
            and bool(order["outcome_supported"])
            and new_cursor < potential_cursor <= end_cursor
        )
        if potential_accepted != expected_potential_accept:
            raise ValueError(
                "supported in-interval potential fill acceptance drift"
            )
        if potential_accepted:
            approximate = (
                int(order["approximate_fill_time_ns"]),
                int(order["approximate_fill_event_sequence"]),
                int(order["approximate_fill_row_index"]),
            )
            if (
                not bool(order["outcome_supported"])
                or approximate != potential_cursor
                or approximate != end_cursor
                or order["active_end_reason"]
                != "accepted_approximate_fill"
                or order["potential_fill_rejection_reason"] is not None
                or not bool(
                    order["approximate_fill_within_actual_active_interval"]
                )
            ):
                raise ValueError("accepted approximate fill invariant failed")
        else:
            if bool(order["approximate_fill_within_actual_active_interval"]):
                raise ValueError("rejected potential marked inside active interval")
            expected_rejection_reason: str | None = None
            if potential_exists:
                if not bool(order["outcome_supported"]):
                    expected_rejection_reason = "outcome_unsupported"
                elif potential_cursor <= new_cursor:
                    expected_rejection_reason = "not_working_intent_pending"
                elif potential_cursor > end_cursor:
                    if order["active_end_reason"] == "actual_cancel_send":
                        expected_rejection_reason = (
                            "not_working_actual_cancelled"
                        )
                    elif order["active_end_reason"] == "session_expiry":
                        expected_rejection_reason = (
                            "not_working_session_expired"
                        )
            if order["potential_fill_rejection_reason"] != expected_rejection_reason:
                raise ValueError("potential fill rejection reason drift")
        terminal_reason = str(order["active_end_reason"])
        if (
            (terminal_reason == "accepted_approximate_fill")
            != potential_accepted
            or (terminal_reason == "actual_cancel_send")
            != cancel_effective
            or (terminal_reason == "session_expiry")
            != (not potential_accepted and not cancel_effective)
        ):
            raise ValueError("working-order terminal reason equivalence drift")
        if order["active_end_reason"] == "session_expiry":
            session_expiry = (
                int(order["session_expiry_time_ns"]),
                int(order["session_expiry_event_sequence"]),
                int(order["session_expiry_row_index"]),
            )
            if (
                potential_accepted
                or cancel_effective
                or end_cursor != session_expiry
            ):
                raise ValueError("session expiry overlaps another terminal")
        elif order["active_end_reason"] not in {
            "accepted_approximate_fill",
            "actual_cancel_send",
        }:
            raise ValueError("unknown working-order active end reason")

        if potential_accepted:
            expected_adapter_status = "accepted_approximate_fill"
        else:
            terminal_status = (
                "effective_actual_cancel"
                if order["active_end_reason"] == "actual_cancel_send"
                else "session_expiry"
            )
            if not potential_exists:
                expected_adapter_status = (
                    f"{terminal_status}_no_potential_fill"
                )
            elif not bool(order["outcome_supported"]):
                expected_adapter_status = (
                    f"{terminal_status}_with_unsupported_potential_fill"
                )
            elif potential_cursor <= new_cursor:
                expected_adapter_status = (
                    f"{terminal_status}_after_pre_working_potential_fill"
                )
            else:
                expected_adapter_status = (
                    f"{terminal_status}_before_potential_fill"
                )
        if order["adapter_outcome_status"] != expected_adapter_status:
            raise ValueError("adapter outcome status drift")

    if set(assignment_lookup) != expected_assignment_ids:
        raise ValueError("request assignment/order fact ID set drift")

    primary = excursions.filter(
        pl.col("analysis_window") == "entry_primary"
    )
    invalid_excursion = primary.filter(
        (pl.col("start_residual_bp") <= 0)
        | (
            pl.col("primary_observable")
            & (
                pl.col("previous_residual_bp").is_null()
                | (pl.col("previous_residual_bp") > 0)
            )
        )
        | (
            pl.col("completed")
            & (
                pl.col("end_residual_bp").is_null()
                | (pl.col("end_residual_bp") > 0)
            )
        )
        | (
            pl.col("touch_time_ns").is_not_null()
            & (
                pl.col("touch_previous_residual_bp")
                >= pl.col("upper_distance_bp")
            )
        )
        | (
            pl.col("touch_time_ns").is_not_null()
            & (
                pl.col("touch_residual_bp")
                < pl.col("upper_distance_bp")
            )
        )
        | (
            pl.col("left_censored")
            & pl.col("start_already_at_or_above_upper")
            & pl.col("touch_time_ns").is_not_null()
        )
    )
    if not invalid_excursion.is_empty():
        raise ValueError("raw excursion/touch invariants failed")

    recomputed_touch_pairs = link_touch_order_pairs(
        primary,
        raw_orders,
    )
    if not _frames_equal_csv_safe(touch_pairs, recomputed_touch_pairs):
        raise ValueError("full first-touch/working-order pair mapping drift")
    recomputed_touch_links = deduplicate_touch_pairs_to_orders(
        recomputed_touch_pairs
    )
    if not _frames_equal_csv_safe(touch_links, recomputed_touch_links):
        raise ValueError("touched raw-order denominator drift")

    order_lookup = {
        str(row["raw_order_fact_id"]): row
        for row in raw_orders.iter_rows(named=True)
    }
    excursion_lookup = {
        str(row["excursion_id"]): row
        for row in primary.iter_rows(named=True)
    }
    if (
        touch_pairs.select("excursion_id", "raw_order_fact_id").n_unique()
        != touch_pairs.height
    ):
        raise ValueError("touch/order pair composite key is duplicated")
    if touch_links["raw_order_fact_id"].n_unique() != touch_links.height:
        raise ValueError("touched-order denominator is not deduplicated")
    for link in touch_pairs.iter_rows(named=True):
        order = order_lookup.get(str(link["raw_order_fact_id"]))
        excursion = excursion_lookup.get(str(link["excursion_id"]))
        if order is None or excursion is None:
            raise ValueError("touch link references an unknown fact")
        new = (
            int(order["actual_new_time_ns"]),
            int(order["actual_new_event_sequence"]),
            int(order["actual_new_row_index"]),
        )
        touch = (
            int(link["touch_time_ns"]),
            int(link["touch_event_sequence"]),
            int(link["touch_row_index"]),
        )
        end = (
            int(order["active_end_time_ns"]),
            int(order["active_end_event_sequence"]),
            int(order["active_end_row_index"]),
        )
        if not new < touch <= end:
            raise ValueError("touch/order active interval invariant failed")
        fill_time = order["approximate_fill_time_ns"]
        if fill_time is not None:
            fill = (
                int(fill_time),
                int(order["approximate_fill_event_sequence"]),
                int(order["approximate_fill_row_index"]),
            )
            if fill <= touch:
                raise ValueError("touch link retained a prior potential fill")
            expected_post_touch = bool(
                order["outcome_supported"] and fill <= end
            )
        else:
            expected_post_touch = False
        if bool(link["post_touch_fill"]) != expected_post_touch:
            raise ValueError("post-touch fill classification drift")

    max_rolling = _max_rolling_requests(assignments)
    if max_rolling > SPOT_REQUEST_CAP:
        raise ValueError("actual send stream exceeds rolling venue limit")
    if assignments.filter(
        (pl.col("kind") == "new") & (pl.col("queue_delay_ns") > 0)
    ).height:
        raise ValueError("S0 actual new cursor was delayed")

    recomputed_monthly = summarize_monthly_attribution(
        product_days,
        primary,
        raw_orders,
        touch_links,
        touch_pairs=touch_pairs,
    )
    if not _frames_equal_csv_safe(monthly, recomputed_monthly):
        raise ValueError("monthly attribution drift")
    recomputed_rank = summarize_post_touch_by_rank(touch_links)
    if not _frames_equal_csv_safe(rank_summary, recomputed_rank):
        raise ValueError("rank attribution drift")
    recomputed_membership = _membership_sensitivity(
        product_days,
        primary,
        raw_orders,
        touch_pairs,
        touch_links,
    )
    if not _frames_equal_csv_safe(membership, recomputed_membership):
        raise ValueError("membership sensitivity drift")
    recomputed_decomposition = (
        build_decomposition(monthly)
        if "202608" in set(monthly["month"].to_list())
        and monthly.filter(pl.col("month") < "202608").height
        else pl.DataFrame()
    )
    if not _frames_equal_csv_safe(decomposition, recomputed_decomposition):
        raise ValueError("decomposition drift")
    if (
        daily_audit["Date"].n_unique() != daily_audit.height
        or set(daily_audit["Date"].cast(pl.String).to_list())
        != set(manifest["Date"].cast(pl.String).to_list())
    ):
        raise ValueError("daily audit date coverage drift")
    required_audit_columns = {
        "Date",
        "adapter_requests",
        "adapter_new",
        "adapter_cancel_requests_sent",
        "actual_new_delayed",
        "cutoff_drain_cancels",
        "potential_fill_events",
        "accepted_approximate_fills",
        "rejected_potential_fills",
        "cancel_not_needed",
        "cancel_pending_removed_on_fill",
        "cancel_assigned_noop",
        "effective_actual_cancels",
        "session_expiries",
        "max_actual_send_second_requests",
        "last_actual_cancel_time_ns",
    }
    missing_audit_columns = sorted(
        required_audit_columns - set(daily_audit.columns)
    )
    if missing_audit_columns:
        raise ValueError(f"daily audit schema missing: {missing_audit_columns}")
    for audit_row in daily_audit.iter_rows(named=True):
        date = str(audit_row["Date"])
        day_orders = raw_orders.filter(pl.col("Date") == date)
        day_assignments = assignments.filter(pl.col("Date") == date)
        send_counts = day_assignments.group_by("actual_send_time_ns").len()
        expected = {
            "adapter_requests": day_assignments.height,
            "adapter_new": day_orders.height,
            "adapter_cancel_requests_sent": day_assignments.filter(
                pl.col("kind") == "cancel"
            ).height,
            "actual_new_delayed": day_assignments.filter(
                (pl.col("kind") == "new") & (pl.col("queue_delay_ns") > 0)
            ).height,
            "cutoff_drain_cancels": day_orders.filter(
                pl.col("cutoff_drain")
            ).height,
            "potential_fill_events": day_orders.filter(
                pl.col("potential_fill_time_ns").is_not_null()
            ).height,
            "accepted_approximate_fills": day_orders.filter(
                pl.col("potential_fill_accepted")
            ).height,
            "rejected_potential_fills": day_orders.filter(
                pl.col("potential_fill_time_ns").is_not_null()
                & ~pl.col("potential_fill_accepted")
            ).height,
            "cancel_not_needed": day_orders.filter(
                pl.col("cancel_not_needed")
            ).height,
            "cancel_pending_removed_on_fill": day_orders.filter(
                pl.col("cancel_pending_removed")
            ).height,
            "cancel_assigned_noop": day_orders.filter(
                pl.col("cancel_send_noop")
            ).height,
            "effective_actual_cancels": day_orders.filter(
                pl.col("cancel_send_effective")
            ).height,
            "session_expiries": day_orders.filter(
                pl.col("active_end_reason") == "session_expiry"
            ).height,
            "max_actual_send_second_requests": int(
                send_counts["len"].max() or 0
            ),
            "last_actual_cancel_time_ns": int(
                day_orders["actual_cancel_time_ns"].max() or 0
            ),
        }
        if any(
            int(audit_row[name]) != int(value)
            for name, value in expected.items()
        ):
            raise ValueError(f"{date}: daily order audit drift")
    accepted_fills = raw_orders.filter(
        pl.col("potential_fill_accepted")
    ).height
    cancel_not_needed = raw_orders.filter(
        pl.col("cancel_not_needed")
    ).height
    cancel_noops = raw_orders.filter(pl.col("cancel_send_noop")).height
    effective_cancels = raw_orders.filter(
        pl.col("cancel_send_effective")
    ).height
    return {
        "status": "pass",
        "manifest_product_days": manifest.height,
        "primary_excursions": primary.height,
        "raw_order_facts": raw_orders.height,
        "touch_order_pairs": touch_pairs.height,
        "touched_raw_order_facts": touch_links.height,
        "request_assignments": assignments.height,
        "max_rolling_spot_requests": max_rolling,
        "rolling_spot_request_cap": SPOT_REQUEST_CAP,
        "actual_new_delay_count": 0,
        "potential_fill_events": raw_orders.filter(
            pl.col("potential_fill_time_ns").is_not_null()
        ).height,
        "accepted_approximate_fills": accepted_fills,
        "cancel_not_needed": cancel_not_needed,
        "assigned_cancel_noops": cancel_noops,
        "effective_actual_cancels": effective_cancels,
        "fill_terminal_removes_working_verified": True,
        "cancel_pending_removal_verified": True,
        "same_cursor_assigned_cancel_noop_verified": True,
        "manifest_coverage_verified": True,
        "causal_q95_verified": True,
        "left_censor_contract_verified": True,
        "published_excursion_invariants_verified": True,
        "raw_tape_reconstruction_verified": False,
        "full_first_touch_working_order_mapping_recomputed": True,
        "touched_order_denominator_recomputed": True,
        "working_interval_contract_verified": True,
        "summaries_recomputed": True,
    }


def _max_rolling_requests(assignments: pl.DataFrame) -> int:
    maximum = 0
    for (_,), group in assignments.sort(
        ["Date", "actual_send_time_ns", "send_sequence"]
    ).group_by("Date", maintain_order=True):
        values = [int(value) for value in group["actual_send_time_ns"]]
        left = 0
        for right, value in enumerate(values):
            lower = value - ONE_SECOND_NS
            while left <= right and values[left] <= lower:
                left += 1
            maximum = max(maximum, right - left + 1)
    return maximum


def _frames_equal_csv_safe(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    if left.shape != right.shape or left.columns != right.columns:
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
            difference = (
                left_column - right_column
            ).abs().fill_null(0.0)
            if bool((difference > 1e-12).any()):
                return False
        elif not left_column.equals(right_column, null_equal=True):
            return False
    return True


def _run_config(
    paths: AttributionPaths,
    dates: Sequence[str],
    manifest: pl.DataFrame,
    git_state: Mapping[str, object],
    tests: Mapping[str, object],
    input_inventory: Mapping[str, object],
    canonical_checks: Mapping[str, bool],
) -> dict[str, object]:
    return {
        "runner_version": RUNNER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "diagnostic_quote_only": True,
        "capital_cap_twd": None,
        "capital_cap_infinite": True,
        "not_a_20m_strategy_replay": True,
        "policy": "q95",
        "boundary_quantile": BOUNDARY_QUANTILE,
        "route": ROUTE_ID,
        "entry_rank_admission": ["BID1", "BID2_BY_TICK"],
        "timezone": "Asia/Taipei",
        "date_contract": {
            "dates": list(dates),
            "first_date": dates[0],
            "last_date": dates[-1],
            "entry_sessions": len(dates),
            "product_days": manifest.height,
            "august_dates": [date for date in dates if date.startswith("202608")],
            "canonical_august_definition": (
                "20260803..20260813, 9 entry sessions"
            ),
        },
        "session": {
            "raw_start_second_from_0900": SESSION_START_SECOND,
            "primary_entry_opportunity_end_exclusive_second": (
                ENTRY_CUTOFF_SECOND
            ),
            "new_freeze_and_cancel_drain_start_second": (
                ENTRY_DRAIN_START_SECOND
            ),
            "nominal_entry_cutoff_second": ENTRY_CUTOFF_SECOND,
            "session_end_second": SESSION_END_SECOND,
            "session_1320_excursion_sensitivity_included": True,
        },
        "venue_scheduler": {
            "venue": "spot",
            "rolling_interval": "(t-1s,t]",
            "request_cap": SPOT_REQUEST_CAP,
            "priority": ["exposed_risk", "cancel", "new"],
            "cutoff_bid_cancel_order": "more aggressive price first",
        },
        "market_denominator": (
            "all fixed matched-manifest product-days, including zero eligible"
        ),
        "queue_denominator": (
            "q95 cap=infinity quote-only actual-working orders; "
            "outcome-supported unique raw_order_fact_id after first primary touch"
        ),
        "touch_order_artifacts": {
            "touch_order_attribution.parquet": (
                "all excursion first-touch x working-order pairs"
            ),
            "touched_raw_order_facts.parquet": (
                "one earliest qualifying touch per raw_order_fact_id; "
                "queue-rate denominator"
            ),
        },
        "excursion_contract": {
            "residual": "basis_mid_bp - anchor_ewma_120s_bp",
            "start": "eligible raw state <=0 to >0",
            "end": "next eligible raw state <=0",
            "touch": "first raw cursor with previous residual < D-1 q95 upper <= current residual",
            "left_censored_primary_excluded": True,
            "gap_left_censored": True,
            "amplitude_quantiles": "completed primary observable excursions",
        },
        "working_interval_contract": {
            "active": (
                "actual_new_send < cursor <= accepted approximate fill, "
                "effective actual cancel, or session expiry"
            ),
            "potential_fill": "legacy makerFill mixed-clock approximation",
            "potential_fill_terminal_removes_working": True,
            "unassigned_cancel_after_terminal": "cancel_not_needed",
            "same_cursor_assigned_cancel_after_fill": (
                "request consumed; cancel effect is no-op"
            ),
            "legacy_nominal_outcome_status": (
                "source makerFill label; provenance only, never authoritative"
            ),
            "adapter_outcome_status": (
                "authoritative terminal classification recomputed from the "
                "chronological working-order lifecycle"
            ),
            "fill_cursor_exact": False,
            "own_quantity_included": False,
            "partial_fill_included": False,
            "cancel_ack_observed": False,
            "joint_volume_allocated": False,
        },
        "cursor_phase": {
            "raw_future": RAW_FUTURE_PHASE,
            "raw_spot": RAW_SPOT_PHASE,
            "approximate_fill": APPROXIMATE_FILL_PHASE,
            "request_assignment_frozen": REQUEST_ASSIGNMENT_PHASE,
            "actual_cancel_effect": ACTUAL_CANCEL_PHASE,
            "actual_new_becomes_working": ACTUAL_NEW_PHASE,
        },
        "paths": {
            key: str(value.resolve())
            for key, value in asdict(paths).items()
        },
        "git": dict(git_state),
        "runtime": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "focused_tests": dict(tests),
        "canonical_checks": dict(canonical_checks),
        "input_inventory": dict(input_inventory),
        "legacy_input_provenance_caveat": (
            "72 daily partitions use migrated_nonatomic markers; file content "
            "hashes bind causal/mapping artifacts, but legacy atomic publish "
            "was not verified"
        ),
        "input_hash_contract": "full-content SHA-256 for every declared input",
    }


def _input_inventory(
    paths: AttributionPaths,
    dates: Sequence[str],
) -> dict[str, object]:
    records: list[dict[str, object]] = [
        _file_inventory(paths.manifest_path, content_hash=True),
        _file_inventory(paths.boundary_path, content_hash=True),
    ]
    for date in dates:
        daily = paths.daily_root / f"Date={date}"
        for path in (
            daily / "complete.json",
            daily / "mapping.parquet",
            daily / "causal_fair.parquet",
            paths.message_root / f"Date={date}" / "message_events.parquet",
            paths.outcome_root
            / f"Date={date}"
            / "candidate_outcomes.parquet",
        ):
            records.append(_file_inventory(path, content_hash=True))
        for path in (
            paths.data_root / "tickData" / f"{date}_StockTick.parquet",
            paths.data_root / "tickFeature" / f"{date}_tickFeature.parquet",
            futures_raw_path(date),
        ):
            records.append(_file_inventory(path, content_hash=True))
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "records": records,
        "record_count": len(records),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_checks(
    paths: AttributionPaths,
    *,
    requested_dates: Sequence[str] | None,
    selected_dates: Sequence[str],
    manifest: pl.DataFrame,
    git_state: Mapping[str, object],
    tests: Mapping[str, object],
    input_inventory: Mapping[str, object],
) -> dict[str, bool]:
    records = input_inventory.get("records")
    if not isinstance(records, list):
        raise TypeError("input inventory records must be a list")
    manifest_path = str(paths.manifest_path.resolve())
    manifest_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("path") == manifest_path
    ]
    source_paths_are_default = all(
        Path(actual).resolve() == Path(expected).resolve()
        for actual, expected in (
            (paths.manifest_path, DEFAULT_MANIFEST_PATH),
            (paths.boundary_path, DEFAULT_BOUNDARY_PATH),
            (paths.daily_root, DEFAULT_DAILY_ROOT),
            (paths.message_root, DEFAULT_MESSAGE_ROOT),
            (paths.outcome_root, DEFAULT_OUTCOME_ROOT),
            (paths.data_root, HFT_DATA_ROOT),
        )
    )
    return {
        "clean_git_worktree": not bool(git_state.get("dirty", True)),
        "focused_tests_passed": tests.get("status") == "pass",
        "full_manifest_date_run": requested_dates is None,
        "fixed_manifest_population": (
            len(selected_dates) == EXPECTED_SESSION_COUNT
            and manifest.height == EXPECTED_PRODUCT_DAY_COUNT
            and selected_dates[0] == EXPECTED_FIRST_DATE
            and selected_dates[-1] == EXPECTED_LAST_DATE
        ),
        "fixed_manifest_content_sha256": (
            len(manifest_records) == 1
            and manifest_records[0].get("sha256") == MANIFEST_SHA256
            and manifest_records[0].get("content_sha256") is True
        ),
        "default_source_paths": source_paths_are_default,
        "complete_input_inventory": (
            len(records) == EXPECTED_INPUT_RECORD_COUNT
            and len(
                {
                    str(record.get("path"))
                    for record in records
                    if isinstance(record, dict)
                }
            )
            == EXPECTED_INPUT_RECORD_COUNT
        ),
        "all_inputs_full_content_sha256": bool(records)
        and all(
            isinstance(record, dict)
            and record.get("hash_scope") == "full_content"
            and record.get("content_sha256") is True
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            for record in records
        ),
    }


def _file_inventory(path: Path, *, content_hash: bool) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    identity = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    identity_sha = _canonical_sha256(identity)
    return {
        **identity,
        "hash_scope": "full_content" if content_hash else "path_bytes_mtime",
        "sha256": _sha256_file(path) if content_hash else identity_sha,
        "content_sha256": content_hash,
        "identity_sha256": identity_sha,
    }


def _run_focused_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        *FOCUSED_TEST_MODULES,
        "-v",
    ]
    result = subprocess.run(
        command,
        cwd=MAKER_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"focused tests failed:\n{output[-8000:]}")
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


def _artifact_metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        result.update(
            {
                "rows": int(
                    pl.scan_parquet(path)
                    .select(pl.len())
                    .collect(engine="streaming")
                    .item()
                ),
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


def verify_bundle(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Mapping[str, object]:
    """Read-only validation of marker, files, hashes, and domain summaries."""

    root = Path(output_root)
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    digest = marker.get("marker_payload_sha256")
    payload = dict(marker)
    payload.pop("marker_payload_sha256", None)
    if digest != _canonical_sha256(payload):
        raise ValueError("complete marker self-hash mismatch")
    if (
        marker.get("complete") is not True
        or marker.get("runner_version") != RUNNER_VERSION
        or marker.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported or incomplete S0 bundle")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("complete marker lacks artifact inventory")
    expected_names = {
        *FRAME_ARTIFACTS,
        *NONFRAME_ARTIFACTS,
        "complete.json",
    }
    expected_artifact_names = {
        *FRAME_ARTIFACTS,
        *NONFRAME_ARTIFACTS,
    }
    if set(artifacts) != expected_artifact_names:
        raise ValueError("complete marker artifact inventory drift")
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"bundle file inventory drift: {sorted(actual_names ^ expected_names)}"
        )
    for name, expected in artifacts.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid bundle artifact: {path}")
        actual = _artifact_metadata(path)
        if actual != expected:
            raise ValueError(f"artifact metadata drift: {path}")
    run_config = json.loads((root / "run_config.json").read_text())
    if marker.get("config_sha256") != _canonical_sha256(run_config):
        raise ValueError("run_config hash mismatch")
    if (
        run_config.get("runner_version") != RUNNER_VERSION
        or run_config.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("run_config runner/schema mismatch")
    canonical_checks = run_config.get("canonical_checks")
    if (
        not isinstance(canonical_checks, dict)
        or canonical_checks != marker.get("canonical_checks")
        or any(not isinstance(value, bool) for value in canonical_checks.values())
    ):
        raise ValueError("canonical eligibility checks drift")
    canonical_eligible = bool(canonical_checks) and all(
        canonical_checks.values()
    )
    if marker.get("canonical_eligible") is not canonical_eligible:
        raise ValueError("canonical eligibility flag drift")
    inventory = run_config.get("input_inventory")
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("records"), list
    ):
        raise TypeError("run_config input inventory is malformed")
    inventory_payload = json.dumps(
        inventory["records"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()
    if (
        inventory.get("inventory_sha256") != inventory_sha256
        or inventory.get("record_count") != len(inventory["records"])
        or marker.get("input_inventory_sha256") != inventory_sha256
    ):
        raise ValueError("input inventory digest drift")

    manifest_path = Path(run_config["paths"]["manifest_path"])
    manifest_records = [
        record
        for record in inventory["records"]
        if record.get("path") == str(manifest_path.resolve())
    ]
    if (
        len(manifest_records) != 1
        or manifest_records[0].get("hash_scope") != "full_content"
        or _sha256_file(manifest_path) != manifest_records[0].get("sha256")
    ):
        raise ValueError("fixed manifest content no longer matches inventory")
    manifest = pl.read_csv(
        manifest_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
        },
    ).select("Date", "ValueCode", "QuoteCode").unique().filter(
        pl.col("Date").is_in(run_config["date_contract"]["dates"])
    ).sort(["Date", "ValueCode"])
    frames = {
        name: (
            pl.read_parquet(root / name)
            if name.endswith(".parquet")
            else pl.read_csv(root / name)
        )
        for name in FRAME_ARTIFACTS
    }
    for name in (
        "monthly_attribution.csv",
        "post_touch_fill_by_rank.csv",
        "membership_sensitivity.csv",
    ):
        if "month" in frames[name].columns:
            frames[name] = frames[name].with_columns(
                pl.col("month").cast(pl.String)
            )
    verification = _verify_domain_frames(
        manifest,
        frames["product_day_coverage.parquet"],
        frames["market_excursions.parquet"],
        frames["raw_order_facts.parquet"],
        frames["touch_order_attribution.parquet"],
        frames["touched_raw_order_facts.parquet"],
        frames["request_assignments.parquet"],
        frames["monthly_attribution.csv"],
        frames["post_touch_fill_by_rank.csv"],
        frames["membership_sensitivity.csv"],
        frames["decomposition.csv"],
        frames["daily_input_audit.csv"],
    )
    published = json.loads((root / "verification.json").read_text())
    if verification != published or verification != marker.get("verification"):
        raise ValueError("published verification drift")
    return {
        **verification,
        "bundle": str(root.resolve()),
        "complete_sha256": _sha256_file(marker_path),
        "marker_payload_sha256": digest,
    }


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
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--message-root", type=Path, default=DEFAULT_MESSAGE_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--data-root", type=Path, default=HFT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--skip-focused-tests", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_bundle(args.output), ensure_ascii=False, indent=2))
        return
    marker = run(
        AttributionPaths(
            manifest_path=args.manifest,
            boundary_path=args.boundaries,
            daily_root=args.daily_root,
            message_root=args.message_root,
            outcome_root=args.outcome_root,
            data_root=args.data_root,
            output_root=args.output,
        ),
        dates=args.dates,
        run_focused_tests=not args.skip_focused_tests,
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
