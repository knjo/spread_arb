"""Causal one-second quote-message load runner.

The runner consumes the leakage-safe daily entry manifest, D-1 q95 boundary,
one-second causal fair panels, and the exact ``SpreadPairTotalCount`` attached
to each sampled spot sequence.  Spread-pair changes are counted as research
samples, but they do not themselves create an order.  Exchange requests are
emitted only by :class:`OneSecondBidQuoteController` under the current
absolute-price lifecycle.

This is a quote-intent study.  It deliberately does not terminate orders on
maker fills and therefore does not estimate futures hedge traffic.  That
second layer must come from physical fill facts and is reported separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from ..common.paths import MAKER_ROOT
from ..fair_mid.quote_churn import price_to_tick_index, tick_index_to_price
from .one_second_message_load import OneSecondBidQuoteController

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
DEFAULT_TICK_FEATURE_ROOT = Path(
    "/home/kevin/Project/HFT/data/tickFeature"
)
DEFAULT_OUTPUT_ROOT = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "order_message_load_causal_v2_20260822_v2"
)

RUNNER_VERSION = "causal_q95_unique_absolute_price_1hz_v2"
SESSION_START_SECOND = 300  # 09:05 Asia/Taipei
SESSION_END_SECOND = 15_600  # 13:20 Asia/Taipei
ENTRY_CUTOFF_SECOND = 14_400  # 13:00 Asia/Taipei
SPOT_REQUEST_CAP_PER_SECOND = 100
BOUNDARY_QUANTILE = 95


@dataclass(frozen=True)
class LoadScenario:
    scenario_id: str
    cutoff_second: int
    admission_offsets: tuple[int, ...] | None

    @property
    def ab12_only(self) -> bool:
        return self.admission_offsets is not None


SCENARIOS = (
    LoadScenario("ab12_entry_until_1300", ENTRY_CUTOFF_SECOND, (-1, 0)),
    LoadScenario("all_passive_entry_until_1300", ENTRY_CUTOFF_SECOND, None),
    LoadScenario("ab12_full_until_1320", SESSION_END_SECOND, (-1, 0)),
    LoadScenario("all_passive_full_until_1320", SESSION_END_SECOND, None),
)


EVENT_SCHEMA = {
    "scenario_id": pl.String,
    "Date": pl.String,
    "ValueCode": pl.String,
    "second_from_open": pl.Int32,
    "kind": pl.String,
    "reason": pl.String,
    "absolute_price_tick": pl.Int64,
    "generation": pl.Int64,
    "submit_point_offset": pl.Int64,
    "submit_point_bucket": pl.String,
    "is_cutoff": pl.Boolean,
}


def _normalise_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )


def _point_bucket_expr(offset: pl.Expr) -> pl.Expr:
    return (
        pl.when(offset == 0)
        .then(pl.lit("BID1"))
        .when(offset == -1)
        .then(pl.lit("BID2_BY_TICK"))
        .when(offset.is_between(-4, -2, closed="both"))
        .then(pl.lit("BID3_TO_BID5_BY_TICK"))
        .when(offset < -4)
        .then(pl.lit("DEEPER_THAN_BID5_BY_TICK"))
        .when(offset > 0)
        .then(pl.lit("INSIDE_BY_TICK"))
        .otherwise(pl.lit("UNKNOWN"))
    )


def _point_bucket(offset: int) -> str:
    if offset == 0:
        return "BID1"
    if offset == -1:
        return "BID2_BY_TICK"
    if -4 <= offset <= -2:
        return "BID3_TO_BID5_BY_TICK"
    if offset < -4:
        return "DEEPER_THAN_BID5_BY_TICK"
    return "INSIDE_BY_TICK"


def _different(current: pl.Expr, previous: pl.Expr) -> pl.Expr:
    return (
        (current.is_null() != previous.is_null())
        | (current != previous).fill_null(False)
    )


def load_inputs(
    manifest_path: Path,
    boundary_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    manifest_source = _normalise_keys(
        pl.read_csv(
            manifest_path,
            schema_overrides={
                "Date": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
                "source_month_last_date": pl.String,
                "source_asof_date": pl.String,
            },
        )
    )
    manifest_lookahead = manifest_source.filter(
        (pl.col("source_month_last_date") >= pl.col("Date"))
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if not manifest_lookahead.is_empty():
        raise ValueError("daily manifest contains non-causal source dates")
    manifest = manifest_source.select(
        "Date", "ValueCode", "QuoteCode"
    ).unique()
    boundaries = (
        _normalise_keys(pl.read_parquet(boundary_path))
        .filter(
            (pl.col("boundary_quantile") == BOUNDARY_QUANTILE)
            & pl.col("adaptive_parameter_valid").fill_null(False)
            & pl.col("execution_safe_snapshot").fill_null(False)
            & ~pl.col("contains_target_day_outcome").fill_null(True)
        )
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "upper_distance_bp",
            "source_asof_date",
        )
        .unique(["Date", "ValueCode", "QuoteCode"])
    )
    boundary_lookahead = boundaries.filter(
        pl.col("source_asof_date").cast(pl.String) >= pl.col("Date")
    )
    if not boundary_lookahead.is_empty():
        raise ValueError("q95 boundaries contain target-day lookahead")
    return manifest.sort(["Date", "ValueCode"]), boundaries


def build_day_state(
    date: str,
    products: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    daily_root: Path,
    tick_feature_root: Path,
) -> pl.DataFrame:
    """Build one row per selected product-second through 13:20."""
    codes = products["ValueCode"].to_list()
    causal_path = daily_root / f"Date={date}" / "causal_fair.parquet"
    tick_path = tick_feature_root / f"{date}_tickFeature.parquet"
    if not causal_path.exists():
        raise FileNotFoundError(causal_path)
    if not tick_path.exists():
        raise FileNotFoundError(tick_path)

    boundary_day = boundaries.filter(pl.col("Date") == date).lazy()
    causal = (
        pl.scan_parquet(causal_path)
        .filter(
            pl.col("ValueCode").is_in(codes)
            & (pl.col("seconds_from_open") >= SESSION_START_SECOND)
            & (pl.col("seconds_from_open") < SESSION_END_SECOND)
        )
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "seconds_from_open",
            "spot_sequence",
            "spot_bid",
            "spot_ask",
            "spot_ref_price",
            "fut_exec_bid",
            "analysis_eligible",
            "anchor_ewma_120s_bp",
        )
        .join(
            boundary_day,
            on=["Date", "ValueCode", "QuoteCode"],
            how="inner",
        )
    )
    spread_clock = (
        pl.scan_parquet(tick_path)
        .filter(pl.col("QuoteCode").is_in(codes))
        .select(
            pl.col("QuoteCode").alias("ValueCode"),
            pl.col("ChannelSeq").alias("spot_sequence"),
            "SpreadPairTotalCount",
        )
    )

    threshold = pl.col("anchor_ewma_120s_bp") + pl.col("upper_distance_bp")
    multiplier = 1.0 + threshold / 10_000.0
    raw_target = pl.col("fut_exec_bid") / multiplier
    target_tick = (
        price_to_tick_index(raw_target, market="spot") + 1e-10
    ).floor().cast(pl.Int64)
    spot_bid_tick = price_to_tick_index(
        pl.col("spot_bid"), market="spot"
    ).round(0).cast(pl.Int64)

    result = (
        causal.join(
            spread_clock,
            on=["ValueCode", "spot_sequence"],
            how="left",
        )
        .with_columns(
            threshold.alias("threshold_basis_bp"),
            target_tick.alias("target_tick"),
            spot_bid_tick.alias("spot_bid_tick"),
        )
        .with_columns(
            tick_index_to_price(
                pl.col("target_tick"), market="spot"
            ).alias("target_price"),
            (pl.col("target_tick") - pl.col("spot_bid_tick")).alias(
                "point_offset"
            ),
        )
        .with_columns(
            (
                pl.col("analysis_eligible").fill_null(False)
                & pl.col("target_tick").is_not_null()
                & (pl.col("target_tick") > 0)
                & pl.col("target_price").is_not_null()
                & (pl.col("target_price") < pl.col("spot_ask"))
                & (pl.col("target_price") > pl.col("spot_ref_price") * 0.91)
                & (pl.col("target_price") < pl.col("spot_ref_price") * 1.08)
            )
            .fill_null(False)
            .alias("base_gate_open")
        )
        .with_columns(
            (
                pl.col("base_gate_open")
                & pl.col("point_offset").is_in([-1, 0])
            ).alias("ab12_admission_open"),
            _point_bucket_expr(pl.col("point_offset")).alias("point_bucket"),
        )
        .select(
            "ValueCode",
            "seconds_from_open",
            "spot_sequence",
            "SpreadPairTotalCount",
            "target_tick",
            "point_offset",
            "point_bucket",
            "base_gate_open",
            "ab12_admission_open",
        )
        .sort(["ValueCode", "seconds_from_open"])
        .collect(engine="streaming")
    )
    return result


def spread_pair_second_counts(frame: pl.DataFrame, date: str) -> pl.DataFrame:
    with_previous = frame.with_columns(
        pl.col("SpreadPairTotalCount")
        .shift(1)
        .over("ValueCode")
        .alias("_previous_epoch"),
        pl.col("spot_sequence")
        .shift(1)
        .over("ValueCode")
        .alias("_previous_spot_sequence"),
        pl.col("seconds_from_open")
        .shift(1)
        .over("ValueCode")
        .alias("_previous_second"),
    )
    exact_changed = (
        pl.col("_previous_second").is_null()
        | _different(
            pl.col("SpreadPairTotalCount"), pl.col("_previous_epoch")
        )
    ) & pl.col("SpreadPairTotalCount").is_not_null()
    fallback_changed = (
        pl.col("SpreadPairTotalCount").is_null()
        & (
            pl.col("_previous_second").is_null()
            | _different(
                pl.col("spot_sequence"), pl.col("_previous_spot_sequence")
            )
        )
    )
    return (
        with_previous.with_columns(
            exact_changed.alias("exact_spread_pair_sample"),
            fallback_changed.alias("fallback_sequence_sample"),
        )
        .group_by("seconds_from_open")
        .agg(
            pl.col("exact_spread_pair_sample")
            .sum()
            .alias("exact_sample_count"),
            pl.col("fallback_sequence_sample")
            .sum()
            .alias("fallback_sequence_sample_count"),
        )
        .with_columns(
            (
                pl.col("exact_sample_count")
                + pl.col("fallback_sequence_sample_count")
            ).alias("sample_count")
        )
        .with_columns(pl.lit(date).alias("Date"))
        .select(
            "Date",
            "seconds_from_open",
            "exact_sample_count",
            "fallback_sequence_sample_count",
            "sample_count",
        )
        .sort("seconds_from_open")
    )


def thin_state_changes(frame: pl.DataFrame) -> pl.DataFrame:
    with_previous = frame.with_columns(
        pl.col("target_tick").shift(1).over("ValueCode").alias("_prev_target"),
        pl.col("base_gate_open")
        .shift(1)
        .over("ValueCode")
        .alias("_prev_gate"),
        pl.col("ab12_admission_open")
        .shift(1)
        .over("ValueCode")
        .alias("_prev_ab12"),
        pl.col("seconds_from_open")
        .shift(1)
        .over("ValueCode")
        .alias("_prev_second"),
    )
    changed = (
        pl.col("_prev_second").is_null()
        | _different(pl.col("target_tick"), pl.col("_prev_target"))
        | _different(pl.col("base_gate_open"), pl.col("_prev_gate"))
        | _different(pl.col("ab12_admission_open"), pl.col("_prev_ab12"))
    )
    return with_previous.filter(changed).select(
        "ValueCode",
        "seconds_from_open",
        "target_tick",
        "point_offset",
        "base_gate_open",
        "ab12_admission_open",
    )


def simulate_scenario(
    state_changes: pl.DataFrame,
    date: str,
    scenario: LoadScenario,
) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    eligible_offsets = (
        None
        if scenario.admission_offsets is None
        else set(scenario.admission_offsets)
    )
    before_cutoff = state_changes.filter(
        pl.col("seconds_from_open") < scenario.cutoff_second
    )
    for product in before_cutoff.partition_by(
        "ValueCode", maintain_order=True
    ):
        value_code = str(product["ValueCode"][0])
        controller = OneSecondBidQuoteController()
        for row in product.iter_rows(named=True):
            second = int(row["seconds_from_open"])
            target = row["target_tick"]
            target_tick = None if target is None else int(target)
            offset = row["point_offset"]
            point_offset = None if offset is None else int(offset)
            base_gate_open = bool(row["base_gate_open"])
            admission_open = base_gate_open and (
                eligible_offsets is None or point_offset in eligible_offsets
            )
            actions = controller.reconcile(
                second,
                target_tick,
                base_gate_open,
                admission_open,
                point_offset,
                gate_reason="input_gate_closed",
            )
            for action in actions:
                records.append(
                    _action_record(
                        action,
                        date=date,
                        value_code=value_code,
                        scenario=scenario,
                    )
                )
        for action in controller.close(
            scenario.cutoff_second,
            reason="entry_cutoff"
            if scenario.cutoff_second == ENTRY_CUTOFF_SECOND
            else "session_cutoff",
        ):
            records.append(
                _action_record(
                    action,
                    date=date,
                    value_code=value_code,
                    scenario=scenario,
                )
            )
    if not records:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    return pl.from_dicts(records, schema=EVENT_SCHEMA)


def _action_record(
    action: object,
    *,
    date: str,
    value_code: str,
    scenario: LoadScenario,
) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "Date": date,
        "ValueCode": value_code,
        "second_from_open": int(action.second),
        "kind": action.kind,
        "reason": action.reason,
        "absolute_price_tick": int(action.absolute_price_tick),
        "generation": int(action.generation),
        "submit_point_offset": int(action.submit_point_offset),
        "submit_point_bucket": _point_bucket(int(action.submit_point_offset)),
        "is_cutoff": int(action.second) == scenario.cutoff_second,
    }


def per_second_messages(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(
            schema={
                "scenario_id": pl.String,
                "Date": pl.String,
                "second_from_open": pl.Int32,
                "submits": pl.UInt32,
                "cancels": pl.UInt32,
                "requests": pl.UInt32,
                "is_cutoff": pl.Boolean,
            }
        )
    return (
        events.group_by("scenario_id", "Date", "second_from_open")
        .agg(
            (pl.col("kind") == "submit").sum().alias("submits"),
            (pl.col("kind") == "cancel").sum().alias("cancels"),
            pl.col("is_cutoff").any().alias("is_cutoff"),
        )
        .with_columns(
            (pl.col("submits") + pl.col("cancels")).alias("requests")
        )
        .sort(["scenario_id", "Date", "second_from_open"])
    )


def _quantile(series: pl.Series, quantile: float) -> float:
    if series.is_empty():
        return 0.0
    value = series.quantile(quantile, interpolation="nearest")
    return 0.0 if value is None else float(value)


def capacity_summary(
    per_second: pl.DataFrame,
    dates: list[str],
    scenarios: tuple[LoadScenario, ...] = SCENARIOS,
    *,
    cap: int = SPOT_REQUEST_CAP_PER_SECOND,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        values = per_second.filter(
            pl.col("scenario_id") == scenario.scenario_id
        )
        intraday = values.filter(~pl.col("is_cutoff"))
        cutoff = values.filter(pl.col("is_cutoff"))
        previous_bucket = intraday.select(
            "Date",
            (pl.col("second_from_open") + 1).alias("second_from_open"),
            pl.col("requests").alias("previous_bucket_requests"),
        )
        adjacent = intraday.join(
            previous_bucket,
            on=["Date", "second_from_open"],
            how="left",
        ).with_columns(
            (
                pl.col("requests")
                + pl.col("previous_bucket_requests").fill_null(0)
            ).alias("adjacent_two_bucket_requests")
        )
        total_seconds = len(dates) * (
            scenario.cutoff_second - SESSION_START_SECOND
        )
        over = intraday.filter(pl.col("requests") > cap)
        cutoff_max = int(cutoff["requests"].max() or 0)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "date_count": len(dates),
                "session_start_second": SESSION_START_SECOND,
                "cutoff_second": scenario.cutoff_second,
                "request_cap_per_second": cap,
                "total_intraday_seconds": total_seconds,
                "active_intraday_seconds": intraday.height,
                "total_requests": int(values["requests"].sum() or 0),
                "intraday_requests": int(intraday["requests"].sum() or 0),
                "mean_requests_all_intraday_seconds": (
                    float(intraday["requests"].sum() or 0) / total_seconds
                    if total_seconds
                    else 0.0
                ),
                "p50_requests_active_second": _quantile(
                    intraday["requests"], 0.50
                ),
                "p95_requests_active_second": _quantile(
                    intraday["requests"], 0.95
                ),
                "p99_requests_active_second": _quantile(
                    intraday["requests"], 0.99
                ),
                "p999_requests_active_second": _quantile(
                    intraday["requests"], 0.999
                ),
                "max_intraday_submits": int(intraday["submits"].max() or 0),
                "max_intraday_cancels": int(intraday["cancels"].max() or 0),
                "max_intraday_requests": int(intraday["requests"].max() or 0),
                "intraday_seconds_over_cap": over.height,
                "intraday_days_over_cap": over["Date"].n_unique(),
                "intraday_share_seconds_over_cap": (
                    over.height / total_seconds if total_seconds else 0.0
                ),
                "max_adjacent_two_bucket_requests": int(
                    adjacent["adjacent_two_bucket_requests"].max() or 0
                ),
                "adjacent_two_bucket_rows_over_cap": adjacent.filter(
                    pl.col("adjacent_two_bucket_requests") > cap
                ).height,
                "p50_cutoff_requests": _quantile(cutoff["requests"], 0.50),
                "p95_cutoff_requests": _quantile(cutoff["requests"], 0.95),
                "max_cutoff_requests": cutoff_max,
                "cutoff_days_over_cap": cutoff.filter(
                    pl.col("requests") > cap
                )["Date"].n_unique(),
                "minimum_seconds_to_drain_max_cutoff_at_cap": (
                    math.ceil(cutoff_max / cap) if cutoff_max else 0
                ),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def daily_summary(
    per_second: pl.DataFrame,
    selected_counts: pl.DataFrame,
) -> pl.DataFrame:
    return (
        per_second.group_by("scenario_id", "Date")
        .agg(
            pl.col("submits").sum().alias("daily_submits"),
            pl.col("cancels").sum().alias("daily_cancels"),
            pl.col("requests").sum().alias("daily_requests"),
            pl.col("requests")
            .filter(~pl.col("is_cutoff"))
            .max()
            .fill_null(0)
            .alias("max_intraday_requests"),
            pl.col("requests")
            .filter(pl.col("is_cutoff"))
            .max()
            .fill_null(0)
            .alias("cutoff_requests"),
        )
        .join(selected_counts, on="Date", how="left")
        .sort(["scenario_id", "Date"])
    )


def run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    tick_feature_root: Path = DEFAULT_TICK_FEATURE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; choose a fresh path: {output_root}"
        )
    output_root.mkdir(parents=True)
    event_root = output_root / "spot_message_events"
    event_root.mkdir()

    manifest, boundaries = load_inputs(manifest_path, boundary_path)
    dates = manifest["Date"].unique().sort().to_list()
    selected_counts = (
        manifest.group_by("Date")
        .agg(pl.len().alias("selected_products"))
        .sort("Date")
    )
    per_second_parts: list[pl.DataFrame] = []
    sample_parts: list[pl.DataFrame] = []
    point_parts: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for date_index, date in enumerate(dates, start=1):
        products = manifest.filter(pl.col("Date") == date)
        state = build_day_state(
            date,
            products,
            boundaries,
            daily_root=daily_root,
            tick_feature_root=tick_feature_root,
        )
        expected_rows = products.height * (
            SESSION_END_SECOND - SESSION_START_SECOND
        )
        if state.height != expected_rows:
            raise ValueError(
                f"{date}: expected {expected_rows} product-seconds, "
                f"found {state.height}"
            )
        if state["ValueCode"].n_unique() != products.height:
            raise ValueError(f"{date}: selected product coverage is incomplete")
        samples = spread_pair_second_counts(state, date)
        sample_parts.append(samples)
        point_parts.append(
            state.filter(pl.col("base_gate_open"))
            .group_by("point_bucket")
            .agg(pl.len().alias("product_seconds"))
            .with_columns(pl.lit(date).alias("Date"))
            .select("Date", "point_bucket", "product_seconds")
        )
        changes = thin_state_changes(state)
        event_parts = [
            simulate_scenario(changes, date, scenario)
            for scenario in SCENARIOS
        ]
        events = pl.concat(event_parts, how="vertical")
        date_dir = event_root / f"Date={date}"
        date_dir.mkdir()
        events.write_parquet(
            date_dir / "message_events.parquet",
            compression="zstd",
            statistics=True,
        )
        per_second_parts.append(per_second_messages(events))
        audit_rows.append(
            {
                "Date": date,
                "selected_products": products.height,
                "state_rows": state.height,
                "state_products": state["ValueCode"].n_unique(),
                "spread_clock_match_rate": float(
                    state["SpreadPairTotalCount"].is_not_null().mean()
                ),
                "base_gate_product_seconds": int(
                    state["base_gate_open"].sum()
                ),
                "ab12_product_seconds": int(
                    state["ab12_admission_open"].sum()
                ),
                "sparse_state_rows": changes.height,
                "spread_pair_samples": int(samples["sample_count"].sum()),
                "exact_spread_pair_samples": int(
                    samples["exact_sample_count"].sum()
                ),
                "fallback_sequence_samples": int(
                    samples["fallback_sequence_sample_count"].sum()
                ),
                "message_event_rows_all_scenarios": events.height,
            }
        )
        print(
            f"[{date_index:02d}/{len(dates):02d}] {date}: "
            f"{products.height} products, {changes.height} sparse states, "
            f"{events.height} messages",
            flush=True,
        )
        del state, changes, events, event_parts
        gc.collect()

    per_second = pl.concat(per_second_parts, how="vertical").sort(
        ["scenario_id", "Date", "second_from_open"]
    )
    samples = pl.concat(sample_parts, how="vertical").sort(
        ["Date", "seconds_from_open"]
    )
    point_observations = (
        pl.concat(point_parts, how="vertical")
        .group_by("point_bucket")
        .agg(pl.col("product_seconds").sum())
        .sort("product_seconds", descending=True)
    )
    audit = pl.from_dicts(audit_rows, infer_schema_length=None)
    capacity = capacity_summary(per_second, dates)
    daily = daily_summary(per_second, selected_counts)

    per_second.write_parquet(output_root / "spot_per_second.parquet")
    samples.write_parquet(output_root / "spread_pair_samples_per_second.parquet")
    point_observations.write_csv(output_root / "target_point_observations.csv")
    audit.write_csv(output_root / "daily_input_audit.csv")
    capacity.write_csv(output_root / "spot_capacity_summary.csv")
    daily.write_csv(output_root / "spot_daily_summary.csv")

    all_events = pl.scan_parquet(
        event_root / "Date=*" / "message_events.parquet"
    )
    point_messages = (
        all_events.group_by(
            "scenario_id", "kind", "submit_point_bucket"
        )
        .agg(pl.len().alias("messages"))
        .sort(["scenario_id", "kind", "messages"], descending=[False, False, True])
        .collect(engine="streaming")
    )
    reason_messages = (
        all_events.group_by("scenario_id", "kind", "reason")
        .agg(pl.len().alias("messages"))
        .sort(["scenario_id", "kind", "messages"], descending=[False, False, True])
        .collect(engine="streaming")
    )
    point_messages.write_csv(output_root / "spot_messages_by_submit_point.csv")
    reason_messages.write_csv(output_root / "spot_messages_by_reason.csv")

    marker = {
        "runner_version": RUNNER_VERSION,
        "boundary_quantile": BOUNDARY_QUANTILE,
        "manifest_path": str(manifest_path),
        "boundary_path": str(boundary_path),
        "daily_root": str(daily_root),
        "tick_feature_root": str(tick_feature_root),
        "date_count": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "product_day_count": manifest.height,
        "product_union_count": manifest["ValueCode"].n_unique(),
        "spot_request_cap_per_second": SPOT_REQUEST_CAP_PER_SECOND,
        "session_start_second": SESSION_START_SECOND,
        "session_end_second": SESSION_END_SECOND,
        "scenarios": [asdict(value) for value in SCENARIOS],
        "same_absolute_price_resubmitted_on_epoch_change": False,
        "spread_pair_clock_role": "sample_de_duplication_only",
        "missing_spread_pair_clock_fallback": "spot_sequence changes at the one-second sample; never an order gate",
        "fills_applied_to_quote_lifecycle": False,
        "maker_fill_and_futures_hedge_included": False,
        "ab12_definition": "target tick offset 0 or -1 versus spot BID1; L2 book gaps unavailable in one-second panel",
    }
    (output_root / "complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n"
    )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument(
        "--tick-feature-root", type=Path, default=DEFAULT_TICK_FEATURE_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    marker = run(
        manifest_path=args.manifest,
        boundary_path=args.boundaries,
        daily_root=args.daily_root,
        tick_feature_root=args.tick_feature_root,
        output_root=args.output,
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
