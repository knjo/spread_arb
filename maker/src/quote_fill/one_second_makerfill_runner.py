"""Fast causal-v2 AB1/2 legacy-makerFill screening runner.

This runner starts from the corrected one-second absolute-price quote lifecycle
published by :mod:`one_second_message_load_runner`.  It pairs each submitted
generation with its nominal cancel, attaches the exact as-of spot snapshot,
and uses the historical A/B1--2 ``makerFill`` table to ask whether the legacy
fill time falls inside ``(submit, nominal_stop]``.

The result is deliberately an approximation, not production execution truth:
``makerFill`` is an EOD-looking Float32/TransTime label, has no own quantity or
partial path, and its implied timestamp is anchored to snapshot ``RecvTime``.
Raw ``StockTick`` is still joined at every candidate snapshot so a target one
legal tick behind B1 is called B2 only when it actually equals displayed B2.
Unsupported book gaps and unusable mixed-clock labels fail closed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from ..fair_mid.quote_churn import tick_index_to_price

RUNNER_VERSION = "causal_q95_ab12_legacy_makerfill_screen_v1"
SCENARIO_ID = "ab12_entry_until_1300"
ONE_SECOND_NS = 1_000_000_000
PRICE_EPSILON = 1e-8

DEFAULT_EVENT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "order_message_load_causal_v2_20260822_v2"
    / "spot_message_events"
)
DEFAULT_DAILY_ROOT = MAKER_ROOT / "data" / "walkforward" / "daily"
DEFAULT_BOUNDARY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "rolling_boundaries"
    / "rolling_boundary_snapshots.parquet"
)
DEFAULT_TICK_ROOT = HFT_DATA_ROOT / "tickData"
DEFAULT_MAKERFILL_ROOT = HFT_DATA_ROOT / "makerFill"
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "one_second_makerfill_causal_v2_20260822_v1"
)


EVENT_REQUIRED_COLUMNS = {
    "scenario_id",
    "Date",
    "ValueCode",
    "second_from_open",
    "kind",
    "reason",
    "absolute_price_tick",
    "generation",
    "submit_point_offset",
    "is_cutoff",
}
STATE_REQUIRED_COLUMNS = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "seconds_from_open",
    "timestamp",
    "spot_recv_time",
    "spot_sequence",
    "anchor_ewma_120s_bp",
    "contract_size",
    "end_date",
    "fut_exec_bid",
    "spot_bid",
    "spot_ask",
}
TICK_REQUIRED_COLUMNS = {
    "ValueCode",
    "ChannelSeq",
    "RecvTime",
    "TransTime",
    "BidPrice1",
    "BidPrice2",
    "BidLots1",
    "BidLots2",
}
MAKERFILL_REQUIRED_COLUMNS = {
    "QuoteCode",
    "ChannelSeq",
    "Bid1_FillSeconds",
    "Bid2_FillSeconds",
}


@dataclass(frozen=True)
class RunnerPaths:
    """Input and output roots for one screening run."""

    event_root: Path = DEFAULT_EVENT_ROOT
    daily_root: Path = DEFAULT_DAILY_ROOT
    boundary_path: Path = DEFAULT_BOUNDARY_PATH
    tick_root: Path = DEFAULT_TICK_ROOT
    makerfill_root: Path = DEFAULT_MAKERFILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def _require_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _assert_unique(
    frame: pl.DataFrame,
    keys: Sequence[str],
    source: str,
) -> None:
    if frame.select(*keys).n_unique() != frame.height:
        raise ValueError(f"{source} keys are duplicated: {list(keys)}")


def build_candidate_windows(
    events: pl.DataFrame,
    state: pl.DataFrame,
    *,
    scenario_id: str = SCENARIO_ID,
) -> pl.DataFrame:
    """Pair quote-only submit/cancel messages and attach submit snapshots.

    The generation key is local to a product-day.  Every selected generation
    must have exactly one submit and exactly one nominal cancel.  The state
    row at ``submit_second_from_open`` supplies the 1 Hz decision timestamp
    and exact as-of spot ``ChannelSeq`` used by both raw tick and makerFill.
    """

    _require_columns(events, EVENT_REQUIRED_COLUMNS, "message events")
    _require_columns(state, STATE_REQUIRED_COLUMNS, "one-second state")
    selected = events.filter(pl.col("scenario_id") == scenario_id)
    if selected.is_empty():
        raise ValueError(f"no message events for scenario {scenario_id!r}")
    invalid_kind = selected.filter(~pl.col("kind").is_in(["submit", "cancel"]))
    if not invalid_kind.is_empty():
        raise ValueError("message events contain unsupported kinds")

    keys = ["Date", "ValueCode", "generation"]
    submits = selected.filter(pl.col("kind") == "submit")
    cancels = selected.filter(pl.col("kind") == "cancel")
    _assert_unique(submits, keys, "submit events")
    _assert_unique(cancels, keys, "cancel events")
    if submits.height != cancels.height:
        raise ValueError(
            "selected generations do not have one submit and one cancel"
        )

    submit = submits.select(
        *keys,
        pl.col("scenario_id"),
        pl.col("second_from_open").cast(pl.Int32).alias(
            "submit_second_from_open"
        ),
        pl.col("absolute_price_tick").cast(pl.Int64),
        pl.col("submit_point_offset").cast(pl.Int64),
        pl.col("reason").alias("submit_reason"),
    )
    cancel = cancels.select(
        *keys,
        pl.col("second_from_open").cast(pl.Int32).alias(
            "nominal_stop_second_from_open"
        ),
        pl.col("absolute_price_tick").cast(pl.Int64).alias(
            "_cancel_absolute_price_tick"
        ),
        pl.col("submit_point_offset").cast(pl.Int64).alias(
            "_cancel_submit_point_offset"
        ),
        pl.col("reason").alias("nominal_stop_reason"),
        pl.col("is_cutoff").alias("nominal_stop_is_cutoff"),
    )
    windows = submit.join(cancel, on=keys, how="inner", validate="1:1")
    mismatch = windows.filter(
        (pl.col("absolute_price_tick") != pl.col("_cancel_absolute_price_tick"))
        | (
            pl.col("submit_point_offset")
            != pl.col("_cancel_submit_point_offset")
        )
        | (
            pl.col("nominal_stop_second_from_open")
            <= pl.col("submit_second_from_open")
        )
    )
    if not mismatch.is_empty():
        raise ValueError("submit/cancel generation metadata are incoherent")

    state_keys = ["Date", "ValueCode", "seconds_from_open"]
    _assert_unique(state, state_keys, "one-second state")
    attached = windows.join(
        state.select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("seconds_from_open").cast(pl.Int32),
            pl.col("timestamp").alias("submit_decision_timestamp"),
            pl.col("spot_recv_time").alias("maker_snapshot_recv_timestamp"),
            pl.col("spot_sequence")
            .cast(pl.UInt64)
            .alias("maker_snapshot_channel_seq"),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("anchor_ewma_120s_bp").cast(pl.Float64),
            pl.col("contract_size").cast(pl.Float64),
            pl.col("end_date"),
            pl.col("fut_exec_bid")
            .cast(pl.Float64)
            .alias("fut_exec_bid_at_submit"),
            pl.col("spot_bid")
            .cast(pl.Float64)
            .alias("spot_bid_at_submit"),
            pl.col("spot_ask")
            .cast(pl.Float64)
            .alias("spot_ask_at_submit"),
        ),
        left_on=["Date", "ValueCode", "submit_second_from_open"],
        right_on=state_keys,
        how="left",
        validate="m:1",
    )
    missing_state = attached.filter(
        pl.col("submit_decision_timestamp").is_null()
        | pl.col("maker_snapshot_recv_timestamp").is_null()
        | pl.col("maker_snapshot_channel_seq").is_null()
    )
    if not missing_state.is_empty():
        raise ValueError("candidate submit snapshots are incomplete")

    result = (
        attached.with_columns(
            pl.col("submit_decision_timestamp")
            .dt.timestamp("ns")
            .alias("submit_decision_time_ns"),
            pl.col("maker_snapshot_recv_timestamp")
            .dt.timestamp("ns")
            .alias("maker_snapshot_recv_time_ns"),
            (
                pl.col("nominal_stop_second_from_open").cast(pl.Int64)
                - pl.col("submit_second_from_open").cast(pl.Int64)
            ).alias("nominal_lifetime_seconds"),
        )
        .with_columns(
            (
                pl.col("submit_decision_time_ns")
                + pl.col("nominal_lifetime_seconds")
                * pl.lit(ONE_SECOND_NS, dtype=pl.Int64)
            ).alias("nominal_stop_time_ns"),
            (
                pl.col("submit_decision_time_ns")
                - pl.col("maker_snapshot_recv_time_ns")
            )
            .truediv(1_000_000.0)
            .alias("maker_snapshot_age_ms"),
            pl.concat_str(
                "Date",
                "ValueCode",
                pl.col("generation").cast(pl.String),
                separator="/",
            ).alias("physical_order_id"),
        )
        .drop(
            "_cancel_absolute_price_tick",
            "_cancel_submit_point_offset",
            "submit_decision_timestamp",
            "maker_snapshot_recv_timestamp",
        )
        .sort(["Date", "ValueCode", "generation"])
    )
    negative_age = result.filter(pl.col("maker_snapshot_age_ms") < 0)
    if not negative_age.is_empty():
        raise ValueError("maker snapshot occurs after the submit decision")
    return result


def attach_q95_boundaries(
    windows: pl.DataFrame,
    boundaries: pl.DataFrame,
) -> pl.DataFrame:
    """Attach the D-1 execution-safe q95 policy used by every target."""

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require_columns(boundaries, required, "rolling boundaries")
    keys = ["Date", "ValueCode", "QuoteCode"]
    selected = (
        boundaries.filter(
            (pl.col("boundary_quantile") == 95)
            & pl.col("execution_safe_snapshot").fill_null(False)
            & ~pl.col("contains_target_day_outcome").fill_null(True)
        )
        .select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("upper_distance_bp").cast(pl.Float64),
            pl.col("lower_distance_bp").cast(pl.Float64),
            pl.col("source_asof_date").cast(pl.String).alias(
                "boundary_source_asof_date"
            ),
        )
        .unique(keys)
    )
    _assert_unique(selected, keys, "selected q95 boundaries")
    result = windows.join(
        selected,
        on=keys,
        how="left",
        validate="m:1",
    ).with_columns(
        (
            pl.col("anchor_ewma_120s_bp") + pl.col("upper_distance_bp")
        ).alias("entry_threshold_basis_bp")
    )
    missing = result.filter(
        pl.col("boundary_quantile").is_null()
        | pl.col("upper_distance_bp").is_null()
        | pl.col("lower_distance_bp").is_null()
    )
    if not missing.is_empty():
        raise ValueError("candidate q95 boundary attachment is incomplete")
    lookahead = result.filter(
        pl.col("boundary_source_asof_date") >= pl.col("Date")
    )
    if not lookahead.is_empty():
        raise ValueError("candidate q95 boundaries contain target-day lookahead")
    return result


def label_candidate_windows(
    windows: pl.DataFrame,
    tick_snapshots: pl.DataFrame,
    makerfill: pl.DataFrame,
) -> pl.DataFrame:
    """Attach exact displayed rank and approximate fill-before-stop labels."""

    _require_columns(
        windows,
        {
            "Date",
            "ValueCode",
            "generation",
            "absolute_price_tick",
            "maker_snapshot_channel_seq",
            "maker_snapshot_recv_time_ns",
            "submit_decision_time_ns",
            "nominal_stop_time_ns",
        },
        "candidate windows",
    )
    _require_columns(tick_snapshots, TICK_REQUIRED_COLUMNS, "tick snapshots")
    _require_columns(makerfill, MAKERFILL_REQUIRED_COLUMNS, "makerFill")

    tick = tick_snapshots.select(
        pl.col("ValueCode").cast(pl.String),
        pl.col("ChannelSeq").cast(pl.UInt64).alias(
            "maker_snapshot_channel_seq"
        ),
        pl.col("RecvTime").dt.timestamp("ns").alias("raw_tick_recv_time_ns"),
        pl.col("TransTime")
        .dt.timestamp("us")
        .alias("raw_tick_trans_time_us"),
        pl.col("BidPrice1").cast(pl.Float64).alias("snapshot_bid_price1"),
        pl.col("BidPrice2").cast(pl.Float64).alias("snapshot_bid_price2"),
        pl.col("BidLots1").cast(pl.Int64).alias("snapshot_bid_lots1"),
        pl.col("BidLots2").cast(pl.Int64).alias("snapshot_bid_lots2"),
    )
    maker = makerfill.select(
        pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
        pl.col("ChannelSeq").cast(pl.UInt64).alias(
            "maker_snapshot_channel_seq"
        ),
        pl.col("Bid1_FillSeconds").cast(pl.Float64),
        pl.col("Bid2_FillSeconds").cast(pl.Float64),
    )
    snapshot_keys = ["ValueCode", "maker_snapshot_channel_seq"]
    _assert_unique(tick, snapshot_keys, "tick snapshots")
    _assert_unique(maker, snapshot_keys, "makerFill snapshots")

    joined = (
        windows.join(tick, on=snapshot_keys, how="left", validate="m:1")
        .join(maker, on=snapshot_keys, how="left", validate="m:1")
        .with_columns(
            tick_index_to_price(
                pl.col("absolute_price_tick"), market="spot"
            ).alias("target_price")
        )
        .with_columns(
            pl.when(
                (
                    pl.col("target_price") - pl.col("snapshot_bid_price1")
                ).abs()
                < PRICE_EPSILON
            )
            .then(pl.lit("BID1"))
            .when(
                (
                    pl.col("target_price") - pl.col("snapshot_bid_price2")
                ).abs()
                < PRICE_EPSILON
            )
            .then(pl.lit("BID2"))
            .otherwise(None)
            .alias("exact_target_rank")
        )
        .with_columns(
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.col("Bid1_FillSeconds"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.col("Bid2_FillSeconds"))
            .otherwise(None)
            .alias("makerfill_fill_seconds"),
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.col("snapshot_bid_lots1"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.col("snapshot_bid_lots2"))
            .otherwise(None)
            .alias("initial_displayed_lots"),
            pl.when(pl.col("exact_target_rank") == "BID1")
            .then(pl.lit("Bid1_FillSeconds"))
            .when(pl.col("exact_target_rank") == "BID2")
            .then(pl.lit("Bid2_FillSeconds"))
            .otherwise(None)
            .alias("makerfill_column"),
        )
        .with_columns(
            (
                pl.col("maker_snapshot_recv_time_ns")
                + (
                    pl.col("makerfill_fill_seconds")
                    .fill_nan(None)
                    .cast(pl.Float64)
                    * ONE_SECOND_NS
                )
                .round(0)
                .cast(pl.Int64)
            ).alias("makerfill_implied_fill_time_ns"),
            (
                pl.col("raw_tick_recv_time_ns")
                == pl.col("maker_snapshot_recv_time_ns")
            )
            .fill_null(False)
            .alias("snapshot_recv_time_exact_match"),
        )
    )

    tick_missing = pl.col("raw_tick_recv_time_ns").is_null()
    rank_missing = pl.col("exact_target_rank").is_null()
    makerfill_key_missing = (
        pl.col("Bid1_FillSeconds").is_null()
        & pl.col("Bid2_FillSeconds").is_null()
    )
    fill_nan = pl.col("makerfill_fill_seconds").is_nan()
    fill_finite_nonnegative = (
        pl.col("makerfill_fill_seconds").is_finite()
        & (pl.col("makerfill_fill_seconds") >= 0)
    ).fill_null(False)
    implied_after_submit = (
        pl.col("makerfill_implied_fill_time_ns")
        > pl.col("submit_decision_time_ns")
    ).fill_null(False)
    outcome_supported = (
        ~tick_missing
        & ~rank_missing
        & ~makerfill_key_missing
        & (fill_nan | (fill_finite_nonnegative & implied_after_submit))
    ).fill_null(False)
    approximate_fill = (
        outcome_supported
        & fill_finite_nonnegative
        & (
            pl.col("makerfill_implied_fill_time_ns")
            <= pl.col("nominal_stop_time_ns")
        )
    ).fill_null(False)

    result = (
        joined.with_columns(
            (~tick_missing & ~rank_missing).fill_null(False).alias(
                "makerfill_mapping_exact"
            ),
            outcome_supported.alias("outcome_supported"),
            pl.when(outcome_supported)
            .then(fill_nan)
            .otherwise(None)
            .alias("legacy_no_fill_through_eod"),
            pl.when(outcome_supported)
            .then(approximate_fill)
            .otherwise(None)
            .alias("approximate_fill_before_nominal_stop"),
            pl.when(outcome_supported)
            .then(approximate_fill)
            .otherwise(None)
            .alias("full_fill"),
            pl.when(outcome_supported)
            .then(~approximate_fill)
            .otherwise(None)
            .alias("approximate_cancel_before_fill"),
            pl.when(approximate_fill)
            .then(
                (
                    pl.col("makerfill_implied_fill_time_ns")
                    - pl.col("submit_decision_time_ns")
                )
                / ONE_SECOND_NS
            )
            .otherwise(None)
            .alias("approximate_fill_delay_from_submit_seconds"),
            pl.when(tick_missing)
            .then(pl.lit("unsupported_missing_raw_tick_snapshot"))
            .when(rank_missing)
            .then(pl.lit("unsupported_target_not_displayed_l1_l2"))
            .when(makerfill_key_missing)
            .then(pl.lit("unsupported_missing_makerfill_key"))
            .when(fill_nan)
            .then(pl.lit("approx_no_fill_through_eod"))
            .when(~fill_finite_nonnegative)
            .then(pl.lit("unsupported_invalid_fill_seconds"))
            .when(~implied_after_submit)
            .then(pl.lit("unsupported_fill_not_after_submit"))
            .when(approximate_fill)
            .then(pl.lit("approx_fill_before_nominal_stop"))
            .otherwise(pl.lit("approx_cancel_before_later_fill"))
            .alias("outcome_status"),
            pl.lit("makerfill_l1_l2_float32_eod_mixed_clock").alias(
                "fill_backend"
            ),
            pl.lit(False).alias("fill_cursor_exact"),
            pl.lit(False).alias("fill_outcome_exact_within_model"),
            pl.lit(False).alias("own_quantity_included"),
            pl.lit(False).alias("partial_fill_included"),
            pl.lit(False).alias("cancel_ack_observed"),
            pl.lit(True).alias("independent_candidate_label"),
            pl.lit(False).alias("joint_volume_allocated"),
            pl.when(approximate_fill)
            .then(
                pl.col("makerfill_implied_fill_time_ns")
                + pl.lit(50_000_000, dtype=pl.Int64)
            )
            .otherwise(None)
            .alias("entry_hedge_decision_time_ns"),
        )
        .drop("Bid1_FillSeconds", "Bid2_FillSeconds")
        .sort(["Date", "ValueCode", "generation"])
    )
    if result.filter(~pl.col("snapshot_recv_time_exact_match")).height:
        raise ValueError("causal state and raw tick RecvTime disagree")
    return result


def aggregate_outcomes(
    outcomes: pl.LazyFrame | pl.DataFrame,
    group_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Return one denominator-explicit fill/cancel screening summary."""

    lazy = outcomes.lazy() if isinstance(outcomes, pl.DataFrame) else outcomes
    groups = list(group_columns)
    expressions = (
        pl.len().cast(pl.Int64).alias("candidate_orders"),
        pl.col("makerfill_mapping_exact")
        .sum()
        .cast(pl.Int64)
        .alias("rank_mapped_orders"),
        pl.col("outcome_supported")
        .sum()
        .cast(pl.Int64)
        .alias("outcome_supported_orders"),
        (~pl.col("outcome_supported"))
        .sum()
        .cast(pl.Int64)
        .alias("unknown_orders"),
        pl.col("approximate_fill_before_nominal_stop")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("approximate_fills"),
        pl.col("approximate_cancel_before_fill")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("approximate_cancels"),
        (~pl.col("legacy_no_fill_through_eod").fill_null(True))
        .filter(pl.col("outcome_supported"))
        .sum()
        .cast(pl.Int64)
        .alias("legacy_eod_positive_orders"),
        pl.col("nominal_lifetime_seconds")
        .median()
        .alias("nominal_lifetime_p50_seconds"),
        pl.col("nominal_lifetime_seconds")
        .quantile(0.95, interpolation="nearest")
        .alias("nominal_lifetime_p95_seconds"),
        pl.col("nominal_lifetime_seconds")
        .quantile(0.99, interpolation="nearest")
        .alias("nominal_lifetime_p99_seconds"),
        pl.col("approximate_fill_delay_from_submit_seconds")
        .median()
        .alias("filled_delay_p50_seconds"),
        pl.col("approximate_fill_delay_from_submit_seconds")
        .quantile(0.95, interpolation="nearest")
        .alias("filled_delay_p95_seconds"),
        pl.col("approximate_fill_delay_from_submit_seconds")
        .quantile(0.99, interpolation="nearest")
        .alias("filled_delay_p99_seconds"),
    )
    summary = (
        lazy.group_by(groups).agg(*expressions)
        if groups
        else lazy.select(*expressions)
    )
    return (
        summary.with_columns(
            (
                pl.col("approximate_fills")
                / pl.col("outcome_supported_orders")
            ).alias("approximate_fill_rate"),
            (
                pl.col("approximate_cancels")
                / pl.col("outcome_supported_orders")
            ).alias("approximate_cancel_rate"),
            (
                pl.col("legacy_eod_positive_orders")
                / pl.col("outcome_supported_orders")
            ).alias("legacy_eod_positive_rate"),
        )
        .sort(groups) if groups else summary.with_columns(
            (
                pl.col("approximate_fills")
                / pl.col("outcome_supported_orders")
            ).alias("approximate_fill_rate"),
            (
                pl.col("approximate_cancels")
                / pl.col("outcome_supported_orders")
            ).alias("approximate_cancel_rate"),
            (
                pl.col("legacy_eod_positive_orders")
                / pl.col("outcome_supported_orders")
            ).alias("legacy_eod_positive_rate"),
        )
    ).collect()


def _load_day_state(
    path: Path,
    value_codes: Sequence[str],
    submit_seconds: Sequence[int],
) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("ValueCode").cast(pl.String).is_in(list(value_codes))
            & pl.col("seconds_from_open").is_in(list(submit_seconds))
        )
        .select(*sorted(STATE_REQUIRED_COLUMNS))
        .collect(engine="streaming")
    )


def _load_tick_snapshots(
    path: Path,
    value_codes: Sequence[str],
    channel_sequences: Sequence[int],
) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("ValueCode").cast(pl.String).is_in(list(value_codes))
            & pl.col("ChannelSeq").is_in(list(channel_sequences))
        )
        .select(*sorted(TICK_REQUIRED_COLUMNS))
        .collect(engine="streaming")
    )


def _load_makerfill(
    path: Path,
    value_codes: Sequence[str],
    channel_sequences: Sequence[int],
) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("QuoteCode").cast(pl.String).is_in(list(value_codes))
            & pl.col("ChannelSeq").is_in(list(channel_sequences))
        )
        .select(*sorted(MAKERFILL_REQUIRED_COLUMNS))
        .collect(engine="streaming")
    )


def _event_dates(event_root: Path) -> list[str]:
    dates = sorted(
        path.parent.name.removeprefix("Date=")
        for path in event_root.glob("Date=*/message_events.parquet")
    )
    if not dates:
        raise FileNotFoundError(f"no event partitions below {event_root}")
    if any(len(date) != 8 or not date.isdigit() for date in dates):
        raise ValueError("event partition names must be Date=YYYYMMDD")
    return dates


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": int(
            pl.scan_parquet(path)
            .select(pl.len().cast(pl.Int64).alias("rows"))
            .collect(engine="streaming")
            .item()
        )
        if path.suffix == ".parquet"
        else None,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    write_marker: bool = True,
) -> Mapping[str, object]:
    """Recompute bundle invariants and every published aggregate table."""

    output_root = Path(output_root)
    complete_path = output_root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    marker = json.loads(complete_path.read_text())
    if marker.get("runner_version") != RUNNER_VERSION:
        raise ValueError("bundle runner version is unsupported")
    partition_paths = sorted(
        (output_root / "candidate_outcomes").glob(
            "Date=*/candidate_outcomes.parquet"
        )
    )
    if len(partition_paths) != int(marker.get("date_count", -1)):
        raise ValueError("candidate partition count does not match marker")
    outcomes = pl.scan_parquet(partition_paths).collect(engine="streaming")
    required = {
        "physical_order_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "generation",
        "boundary_quantile",
        "boundary_source_asof_date",
        "target_price",
        "submit_decision_time_ns",
        "nominal_stop_time_ns",
        "maker_snapshot_channel_seq",
        "maker_snapshot_recv_time_ns",
        "makerfill_fill_seconds",
        "makerfill_implied_fill_time_ns",
        "full_fill",
        "approximate_cancel_before_fill",
        "outcome_supported",
        "fill_cursor_exact",
        "fill_outcome_exact_within_model",
        "entry_hedge_decision_time_ns",
    }
    _require_columns(outcomes, required, "candidate outcome bundle")
    if outcomes.height != int(marker.get("candidate_orders", -1)):
        raise ValueError("candidate row count does not match marker")
    _assert_unique(outcomes, ["physical_order_id"], "physical candidate ids")
    _assert_unique(
        outcomes,
        ["Date", "ValueCode", "generation"],
        "physical candidate composite keys",
    )

    invalid = outcomes.filter(
        (pl.col("boundary_quantile") != 95)
        | (pl.col("boundary_source_asof_date") >= pl.col("Date"))
        | (
            pl.col("maker_snapshot_recv_time_ns")
            > pl.col("submit_decision_time_ns")
        )
        | (
            pl.col("nominal_stop_time_ns")
            <= pl.col("submit_decision_time_ns")
        )
        | pl.col("fill_cursor_exact")
        | pl.col("fill_outcome_exact_within_model")
        | (
            pl.col("outcome_supported")
            != pl.col("full_fill").is_not_null()
        )
        | (
            pl.col("outcome_supported")
            != pl.col("approximate_cancel_before_fill").is_not_null()
        )
        | (
            pl.col("outcome_supported")
            & (
                pl.col("full_fill")
                == pl.col("approximate_cancel_before_fill")
            )
        )
        | (
            pl.col("full_fill").fill_null(False)
            & (
                (
                    pl.col("makerfill_implied_fill_time_ns")
                    <= pl.col("submit_decision_time_ns")
                )
                | (
                    pl.col("makerfill_implied_fill_time_ns")
                    > pl.col("nominal_stop_time_ns")
                )
                | (
                    pl.col("entry_hedge_decision_time_ns")
                    != pl.col("makerfill_implied_fill_time_ns") + 50_000_000
                )
            )
        )
    )
    if not invalid.is_empty():
        raise ValueError("candidate outcome invariants failed")

    recomputed = {
        "overall_summary.csv": aggregate_outcomes(outcomes),
        "daily_summary.csv": aggregate_outcomes(outcomes, ["Date"]),
        "monthly_summary.csv": aggregate_outcomes(
            outcomes.with_columns(
                pl.col("Date").str.slice(0, 6).alias("month")
            ),
            ["month"],
        ),
        "rank_summary.csv": aggregate_outcomes(
            outcomes, ["exact_target_rank"]
        ),
        "stop_reason_summary.csv": aggregate_outcomes(
            outcomes, ["nominal_stop_reason"]
        ),
    }
    for filename, expected in recomputed.items():
        actual = pl.read_csv(output_root / filename).cast(expected.schema)
        if not actual.equals(expected):
            raise ValueError(f"published aggregate drift: {filename}")

    inventory = [
        {
            "path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in partition_paths
    ]
    inventory_digest = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result: dict[str, object] = {
        "status": "pass",
        "runner_version": RUNNER_VERSION,
        "candidate_orders": outcomes.height,
        "candidate_partitions": len(partition_paths),
        "physical_order_ids_unique": True,
        "causal_boundaries_verified": True,
        "mixed_clock_fail_closed_verified": True,
        "aggregate_tables_recomputed": sorted(recomputed),
        "partition_inventory_sha256": inventory_digest,
        "partition_bytes": sum(int(row["bytes"]) for row in inventory),
    }
    if write_marker:
        (output_root / "verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    return result


def run(
    paths: RunnerPaths | None = None,
    *,
    dates: Sequence[str] | None = None,
    scenario_id: str = SCENARIO_ID,
) -> Mapping[str, object]:
    """Build a fresh partitioned fill/cancel screening bundle."""

    paths = RunnerPaths() if paths is None else paths
    selected_dates = list(dates) if dates is not None else _event_dates(
        paths.event_root
    )
    if not selected_dates or len(selected_dates) != len(set(selected_dates)):
        raise ValueError("dates must be unique and non-empty")
    if paths.output_root.exists():
        raise FileExistsError(
            f"output already exists; choose a fresh path: {paths.output_root}"
        )
    paths.output_root.mkdir(parents=True)
    outcome_root = paths.output_root / "candidate_outcomes"
    outcome_root.mkdir()

    started = time.perf_counter()
    audit_rows: list[dict[str, object]] = []
    if not paths.boundary_path.is_file():
        raise FileNotFoundError(paths.boundary_path)
    boundaries = pl.read_parquet(paths.boundary_path)
    for index, date in enumerate(selected_dates, start=1):
        day_started = time.perf_counter()
        event_path = (
            paths.event_root / f"Date={date}" / "message_events.parquet"
        )
        if not event_path.is_file():
            raise FileNotFoundError(event_path)
        events = pl.read_parquet(event_path).filter(
            pl.col("scenario_id") == scenario_id
        )
        submits = events.filter(pl.col("kind") == "submit")
        value_codes = sorted(
            str(value) for value in submits.get_column("ValueCode").unique()
        )
        submit_seconds = sorted(
            int(value)
            for value in submits.get_column("second_from_open").unique()
        )
        state_path = (
            paths.daily_root / f"Date={date}" / "causal_fair.parquet"
        )
        state = _load_day_state(state_path, value_codes, submit_seconds)
        windows = build_candidate_windows(
            events, state, scenario_id=scenario_id
        )
        windows = attach_q95_boundaries(windows, boundaries)
        sequences = sorted(
            int(value)
            for value in windows.get_column(
                "maker_snapshot_channel_seq"
            ).unique()
        )
        tick_path = paths.tick_root / f"{date}_StockTick.parquet"
        makerfill_path = paths.makerfill_root / f"{date}_makerFill.parquet"
        tick = _load_tick_snapshots(tick_path, value_codes, sequences)
        makerfill = _load_makerfill(
            makerfill_path, value_codes, sequences
        )
        outcomes = label_candidate_windows(windows, tick, makerfill)

        partition = outcome_root / f"Date={date}"
        partition.mkdir()
        output_path = partition / "candidate_outcomes.parquet"
        outcomes.write_parquet(
            output_path,
            compression="zstd",
            statistics=True,
        )
        audit_rows.append(
            {
                "Date": date,
                "product_count": outcomes["ValueCode"].n_unique(),
                "candidate_orders": outcomes.height,
                "rank_mapped_orders": int(
                    outcomes["makerfill_mapping_exact"].sum()
                ),
                "outcome_supported_orders": int(
                    outcomes["outcome_supported"].sum()
                ),
                "approximate_fills": int(
                    outcomes[
                        "approximate_fill_before_nominal_stop"
                    ].fill_null(False).sum()
                ),
                "event_path": str(event_path),
                "state_path": str(state_path),
                "tick_path": str(tick_path),
                "makerfill_path": str(makerfill_path),
                "output_path": str(output_path),
                "elapsed_seconds": time.perf_counter() - day_started,
            }
        )
        print(
            f"[{index:02d}/{len(selected_dates):02d}] {date}: "
            f"{outcomes.height:,} candidates, "
            f"{int(outcomes['approximate_fill_before_nominal_stop'].fill_null(False).sum()):,} "
            "approx fills",
            flush=True,
        )
        del events, submits, state, windows, tick, makerfill, outcomes
        gc.collect()

    outcome_scan = pl.scan_parquet(
        outcome_root / "Date=*" / "candidate_outcomes.parquet"
    )
    overall = aggregate_outcomes(outcome_scan)
    daily = aggregate_outcomes(outcome_scan, ["Date"])
    monthly = aggregate_outcomes(
        outcome_scan.with_columns(
            pl.col("Date").str.slice(0, 6).alias("month")
        ),
        ["month"],
    )
    rank = aggregate_outcomes(outcome_scan, ["exact_target_rank"])
    stop = aggregate_outcomes(outcome_scan, ["nominal_stop_reason"])
    status = (
        outcome_scan.group_by("outcome_status")
        .agg(pl.len().cast(pl.Int64).alias("candidate_orders"))
        .sort("outcome_status")
        .collect(engine="streaming")
    )
    audit = pl.from_dicts(audit_rows, infer_schema_length=None).sort("Date")

    overall.write_csv(paths.output_root / "overall_summary.csv")
    daily.write_csv(paths.output_root / "daily_summary.csv")
    monthly.write_csv(paths.output_root / "monthly_summary.csv")
    rank.write_csv(paths.output_root / "rank_summary.csv")
    stop.write_csv(paths.output_root / "stop_reason_summary.csv")
    status.write_csv(paths.output_root / "outcome_status_summary.csv")
    audit.write_csv(paths.output_root / "daily_input_audit.csv")

    outcome_rows = int(overall.item(0, "candidate_orders"))
    marker: dict[str, object] = {
        "runner_version": RUNNER_VERSION,
        "scenario_id": scenario_id,
        "date_count": len(selected_dates),
        "first_date": selected_dates[0],
        "last_date": selected_dates[-1],
        "candidate_orders": outcome_rows,
        "rank_mapped_orders": int(overall.item(0, "rank_mapped_orders")),
        "outcome_supported_orders": int(
            overall.item(0, "outcome_supported_orders")
        ),
        "approximate_fills": int(overall.item(0, "approximate_fills")),
        "approximate_cancels": int(overall.item(0, "approximate_cancels")),
        "approximate_fill_rate": float(
            overall.item(0, "approximate_fill_rate")
        ),
        "event_root": str(paths.event_root),
        "daily_root": str(paths.daily_root),
        "boundary_path": str(paths.boundary_path),
        "tick_root": str(paths.tick_root),
        "makerfill_root": str(paths.makerfill_root),
        "output_root": str(paths.output_root),
        "elapsed_seconds": time.perf_counter() - started,
        "semantics": {
            "candidate_lifecycle": (
                "unique_absolute_price_1hz_forward_add_retreat_cancel"
            ),
            "active_interval": "(submit_decision, nominal_stop]",
            "snapshot_key": (
                "spot ValueCode + causal_fair spot_sequence == raw/makerFill ChannelSeq"
            ),
            "rank_mapping": "target_price must equal displayed raw BID1 or BID2",
            "fill_time": "snapshot RecvTime + legacy Float32 FillSeconds",
            "fill_backend": "legacy makerFill EOD mixed-clock approximation",
            "fill_cursor_exact": False,
            "fill_outcome_exact_within_model": False,
            "own_quantity_included": False,
            "partial_fill_included": False,
            "cancel_ack_observed": False,
            "independent_candidate_label": True,
            "joint_volume_allocated": False,
            "fills_remove_live_quotes_during_source_simulation": False,
        },
        "artifacts": {
            "candidate_partitions": len(selected_dates),
            "overall_summary.csv": str(
                paths.output_root / "overall_summary.csv"
            ),
            "daily_summary.csv": str(paths.output_root / "daily_summary.csv"),
            "monthly_summary.csv": str(
                paths.output_root / "monthly_summary.csv"
            ),
            "rank_summary.csv": str(paths.output_root / "rank_summary.csv"),
            "stop_reason_summary.csv": str(
                paths.output_root / "stop_reason_summary.csv"
            ),
            "outcome_status_summary.csv": str(
                paths.output_root / "outcome_status_summary.csv"
            ),
            "daily_input_audit.csv": str(
                paths.output_root / "daily_input_audit.csv"
            ),
        },
    }
    (paths.output_root / "complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n"
    )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument(
        "--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH
    )
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument(
        "--makerfill-root", type=Path, default=DEFAULT_MAKERFILL_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--scenario-id", default=SCENARIO_ID)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_bundle(args.output),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    marker = run(
        RunnerPaths(
            event_root=args.event_root,
            daily_root=args.daily_root,
            boundary_path=args.boundaries,
            tick_root=args.tick_root,
            makerfill_root=args.makerfill_root,
            output_root=args.output,
        ),
        dates=args.dates,
        scenario_id=args.scenario_id,
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
