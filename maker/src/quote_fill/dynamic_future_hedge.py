"""Analysis-only +50 ms futures hedge screen for causal dynamic maker fills.

The entry input is the q95/AB1--2 makerFill approximation produced from the
causal daily universe manifest.  This module deliberately does not know a
fixed symbol list.  Every input product-day must be present in the supplied
manifest before raw futures data are opened.

Only futures rows carrying an L1--L5 or ``Best*`` book update are retained.
The normal raw-tape semantics are then reused: L1 and Best snapshots persist
independently, a Best price may sit inside L1, and a same-price Best/L1 queue
uses ``max(quantity)`` rather than summing.  Zero-book rows cannot change the
book and are not retained.  Consequently prices and book ages are exact as-of
the *approximate* makerFill timestamp, while TrialMatch/non-book state changes
between book updates are not represented.  The output flags this limitation
and is never labelled production/pathwise ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from ..common.paths import futures_raw_path
from .execution_runner import DEFAULT_WALKFORWARD_DAILY_ROOT
from .hedge import (
    DEFAULT_HEDGE_DELAY_NS,
    BookLevel,
    MakerFillHedgeRequest,
    OppositeBookSnapshot,
    label_delayed_taker_hedge,
)
from .layered import EventCursor
from .raw_tape import SESSION_END, SESSION_START

DYNAMIC_FUTURE_HEDGE_VERSION = "dynamic_future_hedge_v1_analysis_only"
SPOT_EVENT_PRIORITY = 2
SPOT_MAKER_LOTS_PER_HEDGE_UNIT = 2
FUTURE_HEDGE_CONTRACTS = 1
_STREAM_BATCH_SIZE = 131_072

DEFAULT_INPUT_ROOT = Path(
    "maker/data/walkforward/one_second_makerfill_causal_v2_20260822_v1"
)
DEFAULT_MANIFEST_PATH = Path(
    "maker/data/walkforward/monthly_product_selector_causal_v2_20260822/"
    "daily_entry_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/dynamic_future_hedge_causal_v1_20260822"
)

_OUTCOME_REQUIRED = {
    "physical_order_id",
    "Date",
    "ValueCode",
    "QuoteCode",
    "makerfill_implied_fill_time_ns",
    "approximate_fill_before_nominal_stop",
    "outcome_supported",
    "contract_size",
}
_MANIFEST_REQUIRED = {"Date", "ValueCode", "QuoteCode", "selector_version"}
_MAPPING_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
}

_RAW_BOOK_COLUMNS = {
    "RecvTime",
    "TransTime",
    "QuoteCode",
    "PacketSeq",
    "ChannelSeq",
    "TrialMatch",
    "DecimalLocator",
    "TotalFillLots",
    "FillPrice",
    "FillLots",
    *(f"BidPrice{level}" for level in range(1, 6)),
    *(f"BidLots{level}" for level in range(1, 6)),
    *(f"AskPrice{level}" for level in range(1, 6)),
    *(f"AskLots{level}" for level in range(1, 6)),
    "BestBidPrice",
    "BestBidLots",
    "BestAskPrice",
    "BestAskLots",
}


class _RawReceiveTimeRegression(ValueError):
    """Selected raw book rows are not safely merge-sortable as one stream."""


@dataclass(frozen=True)
class DynamicFutureHedgeConfig:
    """Frozen assumptions for the analysis-only screen."""

    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS
    spot_maker_lots: int = SPOT_MAKER_LOTS_PER_HEDGE_UNIT
    future_hedge_contracts: int = FUTURE_HEDGE_CONTRACTS
    force_global_sort_backend: bool = False

    def validate(self) -> None:
        if not isinstance(self.force_global_sort_backend, bool):
            raise TypeError("force_global_sort_backend must be boolean")
        for name, value, allow_zero in (
            ("hedge_delay_ns", self.hedge_delay_ns, True),
            ("spot_maker_lots", self.spot_maker_lots, False),
            ("future_hedge_contracts", self.future_hedge_contracts, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if allow_zero else 1)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")


DEFAULT_DYNAMIC_FUTURE_HEDGE_CONFIG = DynamicFutureHedgeConfig()


def select_manifest_validated_fills(
    outcomes: pl.DataFrame,
    manifest: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Validate dynamic-universe lineage and return supported approximate fills."""

    _require_columns(outcomes, _OUTCOME_REQUIRED, "candidate outcomes")
    _require_columns(manifest, _MANIFEST_REQUIRED, "causal daily manifest")
    normalized_outcomes = outcomes.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("physical_order_id").cast(pl.String),
        pl.col("makerfill_implied_fill_time_ns").cast(pl.Int64),
        pl.col("approximate_fill_before_nominal_stop").cast(pl.Boolean),
        pl.col("outcome_supported").cast(pl.Boolean),
        pl.col("contract_size").cast(pl.Float64),
    )
    if normalized_outcomes.filter(
        pl.col("physical_order_id").is_null()
        | (pl.col("physical_order_id").str.len_chars() == 0)
    ).height:
        raise ValueError("candidate outcomes contain an empty physical_order_id")
    duplicates = normalized_outcomes.group_by("physical_order_id").len().filter(
        pl.col("len") != 1
    )
    if duplicates.height:
        raise ValueError("physical_order_id must be unique in each input frame")

    normalized_manifest = manifest.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("selector_version").cast(pl.String),
    )
    duplicate_manifest = normalized_manifest.group_by(
        ["Date", "ValueCode"]
    ).len().filter(pl.col("len") != 1)
    if duplicate_manifest.height:
        raise ValueError("manifest must have one QuoteCode per product-day")
    keys = normalized_manifest.select("Date", "ValueCode", "QuoteCode")
    outside = normalized_outcomes.join(
        keys,
        on=["Date", "ValueCode", "QuoteCode"],
        how="anti",
    )
    if outside.height:
        example = outside.select("Date", "ValueCode", "QuoteCode").head(5)
        raise ValueError(
            "candidate outcomes are outside the causal daily manifest: "
            f"{example.to_dicts()}"
        )

    selector_versions = normalized_manifest.join(
        normalized_outcomes.select("Date", "ValueCode", "QuoteCode").unique(),
        on=["Date", "ValueCode", "QuoteCode"],
        how="semi",
    )["selector_version"].drop_nulls().unique().to_list()
    if not selector_versions or any(
        "causal" not in str(value).lower() for value in selector_versions
    ):
        raise ValueError(
            "selected product-days must come from a manifest whose selector_version "
            "is explicitly causal"
        )

    supported = pl.col("outcome_supported") == True
    filled = pl.col("approximate_fill_before_nominal_stop") == True
    missing_fill_time = normalized_outcomes.filter(
        supported & filled & pl.col("makerfill_implied_fill_time_ns").is_null()
    )
    if missing_fill_time.height:
        raise ValueError("supported approximate fills require a fill timestamp")
    selected = normalized_outcomes.filter(supported & filled).sort(
        ["Date", "ValueCode", "makerfill_implied_fill_time_ns", "physical_order_id"]
    )
    counters = {
        "candidate_outcomes": normalized_outcomes.height,
        "outcome_supported": normalized_outcomes.filter(supported).height,
        "approximate_fills": selected.height,
        "unsupported_outcomes": normalized_outcomes.filter(~supported).height,
        "supported_nonfills": normalized_outcomes.filter(supported & ~filled).height,
    }
    return selected, counters


def load_exact_daily_mapping(
    date: str,
    fills: pl.DataFrame,
    *,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
) -> pl.DataFrame:
    """Load target-day contract identity/reference rows for filled products."""

    if fills.is_empty():
        return pl.DataFrame(schema={column: pl.String for column in _MAPPING_REQUIRED})
    path = Path(daily_root) / f"Date={date}" / "mapping.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    value_codes = fills["ValueCode"].cast(pl.String).unique().to_list()
    scan = pl.scan_parquet(path)
    _require_schema_columns(scan.collect_schema().names(), _MAPPING_REQUIRED, str(path))
    mapping = (
        scan.filter(
            (pl.col("Date").cast(pl.String) == str(date))
            & pl.col("ValueCode").cast(pl.String).is_in(value_codes)
        )
        .select(sorted(_MAPPING_REQUIRED))
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("spot_ref_price").cast(pl.Float64),
            pl.col("fut_ref_price").cast(pl.Float64),
            pl.col("contract_size").cast(pl.Float64),
        )
        .collect(engine="streaming")
    )
    if mapping.height != len(value_codes):
        raise ValueError(
            f"{date}: expected {len(value_codes)} mapping rows for filled products, "
            f"found {mapping.height}"
        )
    duplicate = mapping.group_by(["ValueCode", "QuoteCode"]).len().filter(
        pl.col("len") != 1
    )
    if duplicate.height:
        raise ValueError(f"{date}: mapping is not one-to-one")
    expected = fills.select(
        "ValueCode", "QuoteCode", "contract_size"
    ).unique()
    joined = expected.join(
        mapping.select("ValueCode", "QuoteCode", "contract_size").rename(
            {"contract_size": "mapped_contract_size"}
        ),
        on=["ValueCode", "QuoteCode"],
        how="left",
    )
    mismatch = joined.filter(
        pl.col("mapped_contract_size").is_null()
        | (
            (pl.col("contract_size") - pl.col("mapped_contract_size")).abs()
            > 1e-9
        )
    )
    if mismatch.height:
        raise ValueError(f"{date}: outcome contract identity differs from daily mapping")
    return mapping.sort(["ValueCode", "QuoteCode"])


def load_relevant_future_book_hits(
    date: str,
    fills: pl.DataFrame,
    mapping: pl.DataFrame,
    *,
    config: DynamicFutureHedgeConfig = DEFAULT_DYNAMIC_FUTURE_HEDGE_CONFIG,
    future_path: Path | None = None,
) -> pl.DataFrame:
    """As-of only fill/+delay queries against independent L1 and Best clocks.

    This avoids materializing the full selected futures day.  A latest L1
    snapshot and latest Best snapshot are located independently because the
    canonical normalizer persists those two feed views independently.
    """

    config.validate()
    if mapping.is_empty() or fills.is_empty():
        return pl.DataFrame()
    path = Path(future_path) if future_path is not None else futures_raw_path(date)
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    available = parquet.schema_arrow.names
    _require_schema_columns(available, _RAW_BOOK_COLUMNS, str(path))
    queries = pl.concat(
        [
            fills.select(
                "physical_order_id",
                "Date",
                "ValueCode",
                "QuoteCode",
                pl.lit("arrival").alias("query_kind"),
                pl.col("makerfill_implied_fill_time_ns")
                .cast(pl.Int64)
                .alias("query_time_ns"),
            ),
            fills.select(
                "physical_order_id",
                "Date",
                "ValueCode",
                "QuoteCode",
                pl.lit("decision").alias("query_kind"),
                (
                    pl.col("makerfill_implied_fill_time_ns").cast(pl.Int64)
                    + config.hedge_delay_ns
                ).alias("query_time_ns"),
            ),
        ],
        how="vertical_relaxed",
    )
    if config.force_global_sort_backend:
        return _polars_sorted_future_asof_hits(
            path, queries, mapping, provenance="forced"
        )
    try:
        return _stream_future_asof_hits(parquet, queries, mapping)
    except _RawReceiveTimeRegression:
        return _polars_sorted_future_asof_hits(
            path, queries, mapping, provenance="fallback"
        )


def _stream_future_asof_hits(
    parquet: pq.ParquetFile,
    queries: pl.DataFrame,
    mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Merge sorted raw book updates into the small set of hedge queries."""

    query_rows: dict[str, list[dict[str, object]]] = {}
    for row in queries.sort(["QuoteCode", "query_time_ns"]).iter_rows(named=True):
        query_rows.setdefault(str(row["QuoteCode"]), []).append(row)
    pointers = {quote_code: 0 for quote_code in query_rows}
    states: dict[str, dict[str, dict[str, object] | None]] = {
        quote_code: {"l1": None, "best": None} for quote_code in query_rows
    }
    mapping_rows = {
        str(row["QuoteCode"]): row
        for row in mapping.select(
            "QuoteCode", "fut_ref_price", "contract_size"
        ).iter_rows(named=True)
    }
    if set(mapping_rows) != set(query_rows):
        raise ValueError("query QuoteCodes do not exactly match filled-product mapping")

    results: list[dict[str, object]] = []

    def emit_before(quote_code: str, recv_ns: int | None) -> None:
        rows = query_rows[quote_code]
        position = pointers[quote_code]
        while position < len(rows) and (
            recv_ns is None or int(rows[position]["query_time_ns"]) < recv_ns
        ):
            results.append(
                _materialize_future_hit(
                    rows[position], states[quote_code], mapping_rows[quote_code]
                )
            )
            position += 1
        pointers[quote_code] = position

    price_columns = [
        f"{side}Price{level}"
        for side in ("Bid", "Ask")
        for level in range(1, 6)
    ]
    lot_columns = [
        f"{side}Lots{level}"
        for side in ("Bid", "Ask")
        for level in range(1, 6)
    ]
    stream_columns = [
        "RecvTime",
        "TransTime",
        "QuoteCode",
        "PacketSeq",
        "ChannelSeq",
        "TrialMatch",
        "DecimalLocator",
        *price_columns,
        *lot_columns,
        "BestBidPrice",
        "BestBidLots",
        "BestAskPrice",
        "BestAskLots",
    ]
    selected_codes = list(query_rows)
    maximum_query_ns = int(queries["query_time_ns"].max())
    last_global_recv_ns: int | None = None
    last_code_cursor: dict[str, tuple[int, int, int]] = {}
    book_update = pl.any_horizontal(
        *(
            pl.col(column).fill_null(0) > 0
            for column in (*price_columns, "BestBidPrice", "BestAskPrice")
        )
    )
    for batch in parquet.iter_batches(
        batch_size=_STREAM_BATCH_SIZE,
        columns=stream_columns,
        use_threads=True,
    ):
        frame = pl.from_arrow(batch)
        recv_ns = pl.col("RecvTime").cast(pl.Datetime("ns")).cast(pl.Int64)
        batch_min = frame.select(recv_ns.min()).item()
        if batch_min is not None and int(batch_min) > maximum_query_ns:
            break
        local_time = pl.col("TransTime").dt.time()
        selected = (
            frame.filter(
                pl.col("QuoteCode").cast(pl.String).is_in(selected_codes)
                & (local_time >= pl.lit(SESSION_START))
                & (local_time < pl.lit(SESSION_END))
                & (recv_ns <= maximum_query_ns)
                & book_update
            )
            .with_columns(recv_ns.alias("_recv_time_ns"))
            .select(
                "_recv_time_ns",
                "QuoteCode",
                "ChannelSeq",
                "PacketSeq",
                "TrialMatch",
                "DecimalLocator",
                *price_columns,
                *lot_columns,
                "BestBidPrice",
                "BestBidLots",
                "BestAskPrice",
                "BestAskLots",
            )
        )
        for row in selected.iter_rows(named=True):
            event_ns = int(row["_recv_time_ns"])
            quote_code = str(row["QuoteCode"])
            sequence = int(row["ChannelSeq"])
            packet = int(row["PacketSeq"])
            cursor = (event_ns, sequence, packet)
            if last_global_recv_ns is not None and event_ns < last_global_recv_ns:
                raise _RawReceiveTimeRegression(
                    "raw futures parquet is not receive-time sorted"
                )
            if quote_code in last_code_cursor and cursor < last_code_cursor[quote_code]:
                raise _RawReceiveTimeRegression(
                    f"raw futures cursor regressed for {quote_code}: {cursor}"
                )
            last_global_recv_ns = event_ns
            last_code_cursor[quote_code] = cursor
            emit_before(quote_code, event_ns)
            decimal_locator = int(row["DecimalLocator"])
            divisor = 10.0**decimal_locator
            trial_match = bool(row["TrialMatch"] or 0)
            if any(int(row[column] or 0) > 0 for column in price_columns):
                state: dict[str, object] = {
                    "l1_recv_time_ns": event_ns,
                    "l1_sequence": sequence,
                    "l1_packet_sequence": packet,
                    "l1_trial_match": trial_match,
                }
                for side in ("Bid", "Ask"):
                    lower = side.lower()
                    for level in range(1, 6):
                        raw_price = int(row[f"{side}Price{level}"] or 0)
                        state[f"l1_{lower}_price_{level}"] = (
                            raw_price / divisor if raw_price > 0 else None
                        )
                        state[f"l1_{lower}_lots_{level}"] = (
                            int(row[f"{side}Lots{level}"] or 0)
                            if raw_price > 0
                            else None
                        )
                states[quote_code]["l1"] = state
            if int(row["BestBidPrice"] or 0) > 0 or int(
                row["BestAskPrice"] or 0
            ) > 0:
                state = {
                    "best_recv_time_ns": event_ns,
                    "best_sequence": sequence,
                    "best_packet_sequence": packet,
                    "best_trial_match": trial_match,
                }
                for side in ("Bid", "Ask"):
                    lower = side.lower()
                    raw_price = int(row[f"Best{side}Price"] or 0)
                    state[f"best_{lower}_price"] = (
                        raw_price / divisor if raw_price > 0 else None
                    )
                    state[f"best_{lower}_lots"] = (
                        int(row[f"Best{side}Lots"] or 0)
                        if raw_price > 0
                        else None
                    )
                states[quote_code]["best"] = state

    for quote_code in query_rows:
        emit_before(quote_code, None)
    if len(results) != queries.height:
        raise RuntimeError(
            f"as-of hit cardinality mismatch: {len(results)} != {queries.height}"
        )
    return pl.from_dicts(results, infer_schema_length=None).with_columns(
        pl.lit("pyarrow_sorted_stream").alias("future_asof_backend"),
        pl.lit(False).alias("raw_receive_time_regression_detected"),
        pl.lit(True).alias("regression_check_performed"),
    ).sort(
        ["physical_order_id", "query_kind"]
    )


def _polars_sorted_future_asof_hits(
    path: Path,
    queries: pl.DataFrame,
    mapping: pl.DataFrame,
    *,
    provenance: str,
) -> pl.DataFrame:
    """Fallback for sharded/unsorted raw files using an explicit global sort."""

    if provenance not in {"forced", "fallback"}:
        raise ValueError("provenance must be 'forced' or 'fallback'")

    quote_codes = mapping["QuoteCode"].cast(pl.String).unique().to_list()
    maximum_query_ns = int(queries["query_time_ns"].max())
    scan = pl.scan_parquet(path)
    local_time = pl.col("TransTime").dt.time()
    recv_ns = pl.col("RecvTime").cast(pl.Datetime("ns")).cast(pl.Int64)
    base = scan.filter(
        pl.col("QuoteCode").cast(pl.String).is_in(quote_codes)
        & (local_time >= pl.lit(SESSION_START))
        & (local_time < pl.lit(SESSION_END))
        & (recv_ns <= maximum_query_ns)
    )
    l1_update = pl.any_horizontal(
        *(
            pl.col(f"{side}Price{level}").fill_null(0) > 0
            for side in ("Bid", "Ask")
            for level in range(1, 6)
        )
    )
    best_update = pl.any_horizontal(
        pl.col("BestBidPrice").fill_null(0) > 0,
        pl.col("BestAskPrice").fill_null(0) > 0,
    )
    divisor = pl.lit(10.0).pow(pl.col("DecimalLocator").cast(pl.Float64))
    l1_columns: list[pl.Expr] = []
    for side in ("Bid", "Ask"):
        lower = side.lower()
        for level in range(1, 6):
            price = pl.col(f"{side}Price{level}").cast(pl.Float64) / divisor
            l1_columns.extend(
                [
                    pl.when(price > 0)
                    .then(price)
                    .otherwise(None)
                    .alias(f"l1_{lower}_price_{level}"),
                    pl.when(price > 0)
                    .then(pl.col(f"{side}Lots{level}").cast(pl.Int64))
                    .otherwise(None)
                    .alias(f"l1_{lower}_lots_{level}"),
                ]
            )
    l1 = (
        base.filter(l1_update)
        .select(
            pl.col("QuoteCode").cast(pl.String),
            recv_ns.alias("l1_recv_time_ns"),
            pl.col("ChannelSeq").cast(pl.Int64).alias("l1_sequence"),
            pl.col("PacketSeq").cast(pl.Int64).alias("l1_packet_sequence"),
            (pl.col("TrialMatch").fill_null(0) != 0).alias("l1_trial_match"),
            *l1_columns,
        )
        .sort(["l1_recv_time_ns", "l1_sequence", "l1_packet_sequence"])
    )
    best_columns: list[pl.Expr] = []
    for side in ("Bid", "Ask"):
        lower = side.lower()
        price = pl.col(f"Best{side}Price").cast(pl.Float64) / divisor
        best_columns.extend(
            [
                pl.when(price > 0)
                .then(price)
                .otherwise(None)
                .alias(f"best_{lower}_price"),
                pl.when(price > 0)
                .then(pl.col(f"Best{side}Lots").cast(pl.Int64))
                .otherwise(None)
                .alias(f"best_{lower}_lots"),
            ]
        )
    best = (
        base.filter(best_update)
        .select(
            pl.col("QuoteCode").cast(pl.String),
            recv_ns.alias("best_recv_time_ns"),
            pl.col("ChannelSeq").cast(pl.Int64).alias("best_sequence"),
            pl.col("PacketSeq").cast(pl.Int64).alias("best_packet_sequence"),
            (pl.col("TrialMatch").fill_null(0) != 0).alias("best_trial_match"),
            *best_columns,
        )
        .sort(["best_recv_time_ns", "best_sequence", "best_packet_sequence"])
    )
    mapping_projection = mapping.lazy().select(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("fut_ref_price").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Float64).alias("mapped_contract_size"),
    )
    result = (
        queries.lazy()
        .sort("query_time_ns")
        .join_asof(
            l1,
            left_on="query_time_ns",
            right_on="l1_recv_time_ns",
            by="QuoteCode",
            strategy="backward",
            check_sortedness=False,
        )
        .join_asof(
            best,
            left_on="query_time_ns",
            right_on="best_recv_time_ns",
            by="QuoteCode",
            strategy="backward",
            check_sortedness=False,
        )
        .join(mapping_projection, on="QuoteCode", how="left", validate="m:1")
        .with_columns(
            pl.lit(
                "polars_global_sort_forced"
                if provenance == "forced"
                else "polars_global_sort_fallback"
            ).alias("future_asof_backend"),
            pl.lit(provenance == "fallback").alias(
                "raw_receive_time_regression_detected"
            ),
            pl.lit(provenance == "fallback").alias(
                "regression_check_performed"
            ),
        )
        .collect(engine="streaming")
        .sort(["physical_order_id", "query_kind"])
    )
    if result.height != queries.height:
        raise RuntimeError(
            f"fallback as-of hit cardinality mismatch: {result.height} != "
            f"{queries.height}"
        )
    return result


def _materialize_future_hit(
    query: Mapping[str, object],
    state: Mapping[str, dict[str, object] | None],
    mapping: Mapping[str, object],
) -> dict[str, object]:
    record = dict(query)
    record.update(
        {
            "fut_ref_price": float(mapping["fut_ref_price"]),
            "mapped_contract_size": float(mapping["contract_size"]),
            "l1_recv_time_ns": None,
            "l1_sequence": None,
            "l1_packet_sequence": None,
            "l1_trial_match": None,
            "best_recv_time_ns": None,
            "best_sequence": None,
            "best_packet_sequence": None,
            "best_trial_match": None,
        }
    )
    for side in ("bid", "ask"):
        for level in range(1, 6):
            record[f"l1_{side}_price_{level}"] = None
            record[f"l1_{side}_lots_{level}"] = None
        record[f"best_{side}_price"] = None
        record[f"best_{side}_lots"] = None
    for prefix in ("l1", "best"):
        value = state[prefix]
        if value is not None:
            record.update(value)
    return record


def label_dynamic_future_hedges(
    fills: pl.DataFrame,
    future_book_hits: pl.DataFrame,
    config: DynamicFutureHedgeConfig = DEFAULT_DYNAMIC_FUTURE_HEDGE_CONFIG,
) -> pl.DataFrame:
    """Apply canonical 50 ms taker math at approximate makerFill timestamps."""

    config.validate()
    if fills.is_empty():
        return pl.DataFrame(schema=_empty_hedge_schema())
    _require_columns(fills, _OUTCOME_REQUIRED, "selected fills")
    hits = {
        (str(row["physical_order_id"]), str(row["query_kind"])): row
        for row in future_book_hits.iter_rows(named=True)
    } if not future_book_hits.is_empty() else {}

    records: list[dict[str, object]] = []
    for fill in fills.iter_rows(named=True):
        fill_ns = int(fill["makerfill_implied_fill_time_ns"])
        decision_ns = fill_ns + config.hedge_delay_ns
        identity = str(fill["physical_order_id"])
        arrival_row = hits.get((identity, "arrival"))
        decision_row = hits.get((identity, "decision"))
        asof_backends = {
            str(row["future_asof_backend"])
            for row in (arrival_row, decision_row)
            if row is not None and row.get("future_asof_backend") is not None
        }
        if len(asof_backends) > 1:
            raise ValueError(f"mixed future as-of backends for {identity}")
        asof_backend = next(iter(asof_backends), None)
        receive_regression = any(
            bool(row.get("raw_receive_time_regression_detected", False))
            for row in (arrival_row, decision_row)
            if row is not None
        )
        regression_checks = {
            bool(row["regression_check_performed"])
            for row in (arrival_row, decision_row)
            if row is not None and row.get("regression_check_performed") is not None
        }
        if len(regression_checks) > 1:
            raise ValueError(f"mixed regression-check provenance for {identity}")
        regression_check_performed = next(iter(regression_checks), None)
        snapshots_by_cursor = {}
        for row in (arrival_row, decision_row):
            if row is None:
                continue
            snapshot = _snapshot_from_hit(row)
            if snapshot is not None:
                snapshots_by_cursor[snapshot.cursor] = snapshot
        snapshots = tuple(
            snapshots_by_cursor[cursor] for cursor in sorted(snapshots_by_cursor)
        )
        request = MakerFillHedgeRequest(
            generation_id=str(fill["physical_order_id"]),
            fill_cursor=EventCursor(fill_ns, SPOT_EVENT_PRIORITY, 0),
            maker_fill_quantity=config.spot_maker_lots,
            hedge_side="sell",
            hedge_quantity=config.future_hedge_contracts,
            delay_ns=config.hedge_delay_ns,
        )
        label = label_delayed_taker_hedge(request, snapshots)
        record = asdict(label)
        for column in (
            "maker_fill_cursor",
            "arrival_snapshot_cursor",
            "decision_snapshot_cursor",
        ):
            record.pop(column, None)
        arrival_cursor = label.arrival_snapshot_cursor
        decision_cursor = label.decision_snapshot_cursor
        record.update(
            {
                key: value
                for key, value in fill.items()
                if key not in record
            }
        )
        record.update(
            {
                "hedge_version": DYNAMIC_FUTURE_HEDGE_VERSION,
                "maker_fill_recv_time_ns": fill_ns,
                "maker_fill_event_sequence": SPOT_EVENT_PRIORITY,
                "maker_fill_row_index": None,
                "arrival_snapshot_recv_time_ns": (
                    arrival_cursor.recv_time_ns if arrival_cursor else None
                ),
                "arrival_snapshot_event_sequence": (
                    arrival_cursor.event_sequence if arrival_cursor else None
                ),
                "arrival_snapshot_row_index": (
                    arrival_cursor.row_index if arrival_cursor else None
                ),
                "decision_snapshot_recv_time_ns": (
                    decision_cursor.recv_time_ns if decision_cursor else None
                ),
                "decision_snapshot_event_sequence": (
                    decision_cursor.event_sequence if decision_cursor else None
                ),
                "decision_snapshot_row_index": (
                    decision_cursor.row_index if decision_cursor else None
                ),
                "arrival_book_recv_time_ns": _optional_int(
                    _book_cursor_from_hit(arrival_row)[0]
                    if arrival_row is not None
                    else None
                ),
                "decision_book_recv_time_ns": _optional_int(
                    _book_cursor_from_hit(decision_row)[0]
                    if decision_row is not None
                    else None
                ),
                "arrival_book_age_ms": _book_age_ms(fill_ns, arrival_row),
                "decision_book_age_ms": _book_age_ms(decision_ns, decision_row),
                "universe_source": "causal_daily_manifest",
                "fixed_symbol_count_used": False,
                "entry_fill_backend": "makerfill_fast_approximation",
                "entry_fill_time_exact": False,
                "entry_fill_cursor_exact": False,
                "future_book_update_asof_exact_at_query": True,
                "future_zero_book_nonprice_events_ignored": True,
                "future_l1_best_persistence_canonical": True,
                "future_asof_rows_materialized_only": True,
                "future_asof_backend": asof_backend,
                "raw_receive_time_regression_detected": receive_regression,
                "regression_check_performed": regression_check_performed,
                "hedge_quantity_assumption": "2_spot_board_lots_to_1_future_contract",
                "hedge_joint_depth_allocated": False,
                "analysis_only": True,
                "pathwise_ev_ready": False,
            }
        )
        records.append(record)
    return pl.from_dicts(records, infer_schema_length=None).sort(
        ["Date", "ValueCode", "maker_fill_recv_time_ns", "physical_order_id"]
    )


def summarize_dynamic_future_hedges(facts: pl.DataFrame) -> pl.DataFrame:
    """Return overall and daily hedge coverage/cost summaries."""

    if facts.is_empty():
        return pl.DataFrame()
    executable = pl.col("status") == "executable"
    expressions = [
        pl.len().alias("approximate_fill_hedges"),
        executable.sum().alias("executable_hedges"),
        pl.col("physical_order_id").n_unique().alias("unique_physical_orders"),
        pl.col("ValueCode").n_unique().alias("filled_products"),
        pl.col("signed_latency_slippage_bp")
        .filter(executable)
        .median()
        .alias("latency_slippage_bp_p50"),
        pl.col("signed_latency_slippage_bp")
        .filter(executable)
        .quantile(0.95)
        .alias("latency_slippage_bp_p95"),
        pl.col("signed_depth_slippage_bp")
        .filter(executable)
        .median()
        .alias("depth_slippage_bp_p50"),
        pl.col("signed_depth_slippage_bp")
        .filter(executable)
        .quantile(0.95)
        .alias("depth_slippage_bp_p95"),
        pl.col("signed_total_slippage_bp")
        .filter(executable)
        .mean()
        .alias("total_slippage_bp_mean"),
        pl.col("signed_total_slippage_bp")
        .filter(executable)
        .median()
        .alias("total_slippage_bp_p50"),
        pl.col("signed_total_slippage_bp")
        .filter(executable)
        .quantile(0.90)
        .alias("total_slippage_bp_p90"),
        pl.col("signed_total_slippage_bp")
        .filter(executable)
        .quantile(0.95)
        .alias("total_slippage_bp_p95"),
        pl.col("decision_book_age_ms").median().alias("book_age_ms_p50"),
        pl.col("decision_book_age_ms").quantile(0.95).alias("book_age_ms_p95"),
        pl.col("decision_book_age_ms").max().alias("book_age_ms_max"),
    ]
    daily = facts.group_by("Date").agg(*expressions).with_columns(
        pl.lit("daily").alias("scope")
    )
    overall = facts.select(
        pl.lit("__all__").alias("Date"),
        *expressions,
        pl.lit("overall").alias("scope"),
    )
    return pl.concat([overall, daily], how="vertical_relaxed").with_columns(
        (
            pl.col("executable_hedges") / pl.col("approximate_fill_hedges")
        ).alias("executable_rate")
    ).sort(["scope", "Date"])


def run_dynamic_future_hedge_bundle(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    dates: Sequence[str] | None = None,
    config: DynamicFutureHedgeConfig = DEFAULT_DYNAMIC_FUTURE_HEDGE_CONFIG,
) -> Path:
    """Run/resume all requested dates and publish an analysis-only bundle."""

    config.validate()
    input_root = Path(input_root)
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if (output_root / "complete.json").is_file():
        raise FileExistsError(f"completed output already exists: {output_root}")
    manifest = pl.read_csv(manifest_path).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    _require_columns(manifest, _MANIFEST_REQUIRED, "causal daily manifest")
    requested_dates = (
        tuple(str(value) for value in dates)
        if dates is not None
        else tuple(sorted(manifest["Date"].unique().to_list()))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_facts: list[pl.DataFrame] = []
    all_audits: list[pl.DataFrame] = []
    for date in requested_dates:
        input_path = (
            input_root
            / "candidate_outcomes"
            / f"Date={date}"
            / "candidate_outcomes.parquet"
        )
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        partition = output_root / f"Date={date}"
        fact_path = partition / "future_hedge_facts.parquet"
        audit_path = partition / "audit.parquet"
        marker_path = partition / "complete.json"
        if marker_path.is_file() and fact_path.is_file() and audit_path.is_file():
            all_facts.append(pl.read_parquet(fact_path))
            all_audits.append(pl.read_parquet(audit_path))
            continue
        outcomes = pl.read_parquet(input_path)
        day_manifest = manifest.filter(pl.col("Date") == date)
        fills, counters = select_manifest_validated_fills(outcomes, day_manifest)
        if fills.is_empty():
            mapping = pl.DataFrame()
            states = pl.DataFrame()
            facts = pl.DataFrame(schema=_empty_hedge_schema())
        else:
            mapping = load_exact_daily_mapping(
                date, fills, daily_root=Path(daily_root)
            )
            states = load_relevant_future_book_hits(
                date,
                fills,
                mapping,
                config=config,
            )
            facts = label_dynamic_future_hedges(fills, states, config)
        audit = _day_audit(
            date,
            outcomes=outcomes,
            fills=fills,
            facts=facts,
            future_states=states,
            mapping=mapping,
            counters=counters,
            input_path=input_path,
        )
        _publish_day_partition(partition, facts, audit)
        all_facts.append(facts)
        all_audits.append(audit)

    facts = pl.concat(all_facts, how="diagonal_relaxed") if all_facts else pl.DataFrame()
    audits = pl.concat(all_audits, how="diagonal_relaxed") if all_audits else pl.DataFrame()
    summary = summarize_dynamic_future_hedges(facts)
    status_summary = (
        facts.group_by("status").agg(pl.len().alias("hedges")).with_columns(
            (pl.col("hedges") / facts.height).alias("rate")
        ).sort("hedges", descending=True)
        if not facts.is_empty()
        else pl.DataFrame()
    )
    facts.write_parquet(output_root / "future_hedge_facts_all.parquet")
    audits.write_parquet(output_root / "daily_audit.parquet")
    summary.write_csv(output_root / "hedge_summary.csv")
    status_summary.write_csv(output_root / "status_summary.csv")
    marker = {
        "schema_version": "dynamic_future_hedge_bundle_v1",
        "hedge_version": DYNAMIC_FUTURE_HEDGE_VERSION,
        "analysis_only": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "fixed_symbol_count_used": False,
        "universe_source": "causal_daily_manifest",
        "entry_fill_time_exact": False,
        "entry_fill_backend": "makerfill_fast_approximation",
        "future_zero_book_nonprice_events_ignored": True,
        "future_l1_best_persistence_canonical": True,
        "hedge_delay_ns": config.hedge_delay_ns,
        "spot_maker_lots": config.spot_maker_lots,
        "future_hedge_contracts": config.future_hedge_contracts,
        "requested_dates": list(requested_dates),
        "completed_dates": audits["Date"].unique().sort().to_list()
        if not audits.is_empty()
        else [],
        "candidate_outcomes": int(audits["candidate_outcomes"].sum())
        if not audits.is_empty()
        else 0,
        "approximate_fill_hedges": facts.height,
        "executable_hedges": facts.filter(pl.col("status") == "executable").height
        if not facts.is_empty()
        else 0,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "input_root": str(input_root),
        "future_asof_fact_provenance_counts": _frame_provenance_counts(facts),
        "future_asof_day_provenance_counts": _frame_provenance_counts(audits),
    }
    _atomic_write_json(output_root / "complete.json", marker)
    return output_root


def _day_audit(
    date: str,
    *,
    outcomes: pl.DataFrame,
    fills: pl.DataFrame,
    facts: pl.DataFrame,
    future_states: pl.DataFrame,
    mapping: pl.DataFrame,
    counters: Mapping[str, int],
    input_path: Path,
) -> pl.DataFrame:
    executable = (
        facts.filter(pl.col("status") == "executable").height
        if not facts.is_empty()
        else 0
    )
    return pl.from_dicts(
        [
            {
                "Date": str(date),
                **dict(counters),
                "input_product_days": outcomes.select("ValueCode").n_unique(),
                "filled_product_days": fills.select("ValueCode").n_unique()
                if not fills.is_empty()
                else 0,
                "mapped_filled_products": mapping.height,
                "relevant_future_asof_rows": future_states.height,
                "expected_future_asof_rows": fills.height * 2,
                "future_asof_backend": (
                    ",".join(
                        sorted(
                            str(value)
                            for value in future_states["future_asof_backend"]
                            .drop_nulls()
                            .unique()
                            .to_list()
                        )
                    )
                    if "future_asof_backend" in future_states.columns
                    else None
                ),
                "raw_receive_time_regression_detected": (
                    bool(
                        future_states[
                            "raw_receive_time_regression_detected"
                        ].any()
                    )
                    if "raw_receive_time_regression_detected"
                    in future_states.columns
                    else False
                ),
                "regression_check_performed": (
                    bool(future_states["regression_check_performed"].all())
                    if "regression_check_performed" in future_states.columns
                    else None
                ),
                "hedge_fact_rows": facts.height,
                "executable_hedges": executable,
                "universe_source": "causal_daily_manifest",
                "fixed_symbol_count_used": False,
                "input_path": str(input_path),
                "input_sha256": _sha256(input_path),
                "entry_fill_time_exact": False,
                "future_zero_book_nonprice_events_ignored": True,
                "future_l1_best_persistence_canonical": True,
                "analysis_only": True,
                "pathwise_ev_ready": False,
                "joint_volume_allocated": False,
            }
        ],
        infer_schema_length=None,
    )


def _publish_day_partition(
    partition: Path, facts: pl.DataFrame, audit: pl.DataFrame
) -> None:
    if partition.exists():
        if (partition / "complete.json").is_file():
            raise FileExistsError(partition)
        shutil.rmtree(partition)
    partition.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}.tmp.", dir=partition.parent)
    )
    try:
        facts.write_parquet(temporary / "future_hedge_facts.parquet")
        audit.write_parquet(temporary / "audit.parquet")
        marker = {
            "schema_version": "dynamic_future_hedge_day_v1",
            "Date": str(audit.item(0, "Date")),
            "analysis_only": True,
            "pathwise_ev_ready": False,
            "future_hedge_facts_sha256": _sha256(
                temporary / "future_hedge_facts.parquet"
            ),
            "audit_sha256": _sha256(temporary / "audit.parquet"),
        }
        for column in (
            "future_asof_backend",
            "raw_receive_time_regression_detected",
            "regression_check_performed",
        ):
            if column in audit.columns:
                marker[column] = audit.item(0, column)
        _atomic_write_json(temporary / "complete.json", marker)
        temporary.rename(partition)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _book_age_ms(query_ns: int, row: Mapping[str, object] | None) -> float | None:
    if row is None:
        return None
    book_ns = _book_cursor_from_hit(row)[0]
    return None if book_ns is None else (query_ns - book_ns) / 1_000_000.0


def _frame_provenance_counts(frame: pl.DataFrame) -> list[dict[str, object]]:
    columns = [
        column
        for column in (
            "future_asof_backend",
            "raw_receive_time_regression_detected",
            "regression_check_performed",
        )
        if column in frame.columns
    ]
    if not columns or frame.is_empty():
        return []
    return frame.group_by(columns).len().sort(columns).to_dicts()


def _book_cursor_from_hit(
    row: Mapping[str, object],
) -> tuple[int | None, int | None, int | None, bool | None]:
    candidates: list[tuple[int, int, int, bool]] = []
    for prefix in ("l1", "best"):
        recv_ns = row.get(f"{prefix}_recv_time_ns")
        sequence = row.get(f"{prefix}_sequence")
        packet = row.get(f"{prefix}_packet_sequence")
        if recv_ns is None or sequence is None:
            continue
        candidates.append(
            (
                int(recv_ns),
                int(sequence),
                int(packet) if packet is not None else 0,
                bool(row.get(f"{prefix}_trial_match", False)),
            )
        )
    if not candidates:
        return None, None, None, None
    return max(candidates, key=lambda value: value[:3])


def _snapshot_from_hit(
    row: Mapping[str, object],
) -> OppositeBookSnapshot | None:
    recv_ns, sequence, _packet, trial_match = _book_cursor_from_hit(row)
    if recv_ns is None or sequence is None:
        return None
    bids = _levels_from_hit(row, "bid")
    asks = _levels_from_hit(row, "ask")
    gate_open = True
    gate_reason: str | None = None
    if not bids or not asks or bids[0].price > asks[0].price:
        gate_open = False
        gate_reason = "invalid_executable_book"
    reference = row.get("fut_ref_price")
    if gate_open and not _positive_number(reference):
        gate_open = False
        gate_reason = "missing_ref_price"
    if gate_open:
        assert reference is not None
        lower = float(reference) * 0.91
        upper = float(reference) * 1.08
        if not (
            lower < bids[0].price < upper
            and lower < asks[0].price < upper
        ):
            gate_open = False
            gate_reason = "ref_price_band"
    return OppositeBookSnapshot(
        EventCursor(recv_ns, 1, sequence),
        bids,
        asks,
        trial_match=bool(trial_match),
        gate_open=gate_open,
        gate_reason=gate_reason,
    )


def _levels_from_hit(
    row: Mapping[str, object], side: str
) -> tuple[BookLevel, ...]:
    candidates: list[tuple[float, int]] = []
    _append_hit_level(
        candidates,
        row.get(f"best_{side}_price"),
        row.get(f"best_{side}_lots"),
    )
    for level in range(1, 6):
        _append_hit_level(
            candidates,
            row.get(f"l1_{side}_price_{level}"),
            row.get(f"l1_{side}_lots_{level}"),
        )
    by_price: dict[float, tuple[float, int]] = {}
    for price, quantity in candidates:
        key = round(price, 8)
        previous = by_price.get(key)
        if previous is None or quantity > previous[1]:
            by_price[key] = (price, quantity)
    ordered = sorted(
        by_price.values(), key=lambda value: value[0], reverse=side == "bid"
    )[:5]
    return tuple(BookLevel(price, quantity) for price, quantity in ordered)


def _append_hit_level(
    output: list[tuple[float, int]], price: object, quantity: object
) -> None:
    if not _positive_number(price) or not _positive_number(quantity):
        return
    integer_quantity = int(quantity)  # type: ignore[arg-type]
    if float(quantity) != integer_quantity:  # type: ignore[arg-type]
        raise ValueError(f"book quantity is not integral: {quantity}")
    output.append((float(price), integer_quantity))  # type: ignore[arg-type]


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _empty_hedge_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "physical_order_id": pl.String,
        "status": pl.String,
        "maker_fill_recv_time_ns": pl.Int64,
        "decision_time_ns": pl.Int64,
        "signed_latency_slippage_bp": pl.Float64,
        "signed_depth_slippage_bp": pl.Float64,
        "signed_total_slippage_bp": pl.Float64,
        "decision_book_age_ms": pl.Float64,
    }


def _require_columns(
    frame: pl.DataFrame, required: Iterable[str], source: str
) -> None:
    _require_schema_columns(frame.columns, required, source)


def _require_schema_columns(
    available: Iterable[str], required: Iterable[str], source: str
) -> None:
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_dates(values: str | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    dates = tuple(value.strip() for value in values.split(",") if value.strip())
    return dates or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_WALKFORWARD_DAILY_ROOT)
    parser.add_argument("--dates", help="optional comma-separated YYYYMMDD dates")
    parser.add_argument(
        "--force-global-sort-backend",
        action="store_true",
        help="globally sort selected futures book updates instead of streaming",
    )
    args = parser.parse_args(argv)
    output = run_dynamic_future_hedge_bundle(
        input_root=args.input_root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        daily_root=args.daily_root,
        dates=_parse_dates(args.dates),
        config=DynamicFutureHedgeConfig(
            force_global_sort_backend=args.force_global_sort_backend
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
