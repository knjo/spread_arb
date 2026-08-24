"""Memory-bounded product-day runner for maker execution fact generation.

The raw spot and futures files are too large to normalize for the complete
universe at once.  The single-product API owns exactly one product-day.  The
narrow-universe API owns one selected day at a time, reads and normalizes that
day's raw files once, then slices the shared tape into exact product inputs.
Both paths write the same atomic product-day partition schema.

This is an independent-candidate research replay.  It is suitable for
building D-1 lookup-table probabilities and costs, but it is not the final
portfolio simulator: overlapping hypothetical orders do not jointly consume
printed/displayed volume and fills do not yet update a shared inventory book.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
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
from ..quote_width.daily_facts import (
    MIGRATED_DAILY_FACT_SCHEMA_VERSION,
    validate_completion_marker,
)
from ..quote_width.rolling import (
    DEFAULT_QUANTILES,
    validate_rolling_boundary_artifact,
)
from .engine import build_layered_order_windows
from .execution_facts import (
    ExecutableExitPathConfig,
    build_executable_exit_facts,
    build_executable_taker_exit_path,
    build_execution_action_facts,
    summarize_executable_exit_daily,
    summarize_execution_daily,
)
from .hedge import DEFAULT_HEDGE_DELAY_NS
from .hedge_study import run_hedge_study
from .indexed_replay import IndexedTradeReplay
from .merged import (
    MergedTargetStudyInput,
    build_merged_target_observations_from_frames,
    observations_for_policy,
    session_cutoff_cursor,
)
from .pilot import ENTRY_ROUTES, load_spot_feature_state
from .raw_tape import RawTapeDay, load_raw_tape_day
from .study import (
    _label_policy_windows,
    _trade_events,
    summarize_raw_order_facts,
)
from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
    ROUTE_SPECS,
)


EXECUTION_RUNNER_VERSION = "product_day_execution_facts_v3_price_ladder"
DEFAULT_WALKFORWARD_DAILY_ROOT = (
    MAKER_ROOT / "data" / "walkforward" / "daily"
)
DEFAULT_ROLLING_BOUNDARY_PATH = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "rolling_boundaries"
    / "rolling_boundary_snapshots.parquet"
)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")

_SAFE_MAPPING_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
)
_SAFE_CAUSAL_FAIR_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "timestamp",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
    "anchor_ewma_120s_bp",
)
_SAFE_BOUNDARY_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "contract_size",
    "fut_ref_price",
    "spot_ref_price",
    "boundary_quantile",
    "boundary_role",
    "upper_distance_bp",
    "lower_distance_bp",
    "adaptive_parameter_valid",
    "source_asof_date",
    "parameter_version",
    "price_ladder_version",
    "future_one_dollar_tick_effective_date",
    "probability_layer",
    "execution_safe_snapshot",
    "contains_target_day_outcome",
)
_SPREAD_CLOCK_SOURCE_COLUMNS = (
    "QuoteCode",
    "ChannelSeq",
    "SpreadPairID",
    "SpreadPairSeq",
    "SpreadPairTotalCount",
    "SpreadCountAtSameCount",
)


@dataclass(frozen=True)
class ExecutionRunnerConfig:
    """Frozen action axes and raw execution timing for one run."""

    routes: tuple[str, ...] = tuple(ENTRY_ROUTES)
    boundary_quantiles: tuple[int, ...] = tuple(DEFAULT_QUANTILES)
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS
    exit_path: ExecutableExitPathConfig = ExecutableExitPathConfig()
    price_ladder_version: str = PRICE_LADDER_VERSION
    future_one_dollar_tick_effective_date: str = (
        FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    )
    runner_version: str = EXECUTION_RUNNER_VERSION

    def validate(self) -> None:
        if not self.routes or len(self.routes) != len(set(self.routes)):
            raise ValueError("routes must be nonempty and unique")
        invalid_routes = sorted(set(self.routes) - set(ENTRY_ROUTES))
        if invalid_routes:
            raise ValueError(f"unsupported entry routes: {invalid_routes}")
        if (
            not self.boundary_quantiles
            or len(self.boundary_quantiles)
            != len(set(self.boundary_quantiles))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value >= 100
                for value in self.boundary_quantiles
            )
        ):
            raise ValueError(
                "boundary_quantiles must be unique integers strictly between "
                "0 and 100"
            )
        if (
            isinstance(self.hedge_delay_ns, bool)
            or not isinstance(self.hedge_delay_ns, int)
            or self.hedge_delay_ns < 0
        ):
            raise ValueError("hedge_delay_ns must be a non-negative integer")
        if self.price_ladder_version != PRICE_LADDER_VERSION:
            raise ValueError(
                "price_ladder_version must match the implemented target and "
                "raw-trade ladder"
            )
        if (
            self.future_one_dollar_tick_effective_date
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ):
            raise ValueError(
                "future_one_dollar_tick_effective_date must match the "
                "implemented target and raw-trade ladder"
            )
        self.exit_path.validate()


@dataclass(frozen=True)
class ExecutionProductDayResult:
    """Detailed and daily facts for exactly one product-day."""

    order_aliases: pl.DataFrame
    raw_order_facts: pl.DataFrame
    hedge_facts: pl.DataFrame
    execution_action_facts: pl.DataFrame
    execution_daily_facts: pl.DataFrame
    exit_facts: pl.DataFrame
    exit_daily_facts: pl.DataFrame
    policy_audit: pl.DataFrame
    target_audit: pl.DataFrame
    raw_tape_audit: pl.DataFrame


@dataclass(frozen=True)
class WalkForwardProductDaySources:
    """Validated, execution-safe persisted inputs for one product-day.

    The mapping is projected through a strict allowlist because its source
    artifact also carries target-day realised turnover.  The causal fair
    frame similarly keeps only the anchor and contract identity needed by the
    exact raw replay.  No excursion/outcome artifact is opened here.
    """

    date: str
    value_code: str
    quote_code: str
    mapping: pl.DataFrame
    causal_fair: pl.DataFrame
    rolling_boundaries: pl.DataFrame
    daily_schema_version: str
    migrated_daily_marker: bool
    source_asof_date: str


@dataclass(frozen=True)
class WalkForwardExecutionDayBatch:
    """Validated small inputs plus one shared normalized raw day.

    Every product's mapping, fair anchor and rolling boundary rows have
    already passed the exact product-day and strictly-D-1 checks in
    :func:`load_walkforward_product_day_sources`.  ``raw_tape`` contains only
    these products, but remains combined so the large spot/futures files need
    be scanned and normalized only once.  Product slices are materialized on
    demand by :func:`merged_product_day_from_batch` and can be released after
    their atomic partition is published.
    """

    date: str
    value_codes: tuple[str, ...]
    sources: tuple[WalkForwardProductDaySources, ...]
    raw_tape: RawTapeDay
    spot_feature_state: pl.DataFrame
    boundary_quantiles: tuple[int, ...]


ProductDayLoader = Callable[[str, str], MergedTargetStudyInput]
ExecutionDayBatchLoader = Callable[
    [str, Sequence[str]], WalkForwardExecutionDayBatch
]
ExitRuleLoader = Callable[[str, str, pl.DataFrame], pl.DataFrame | None]


def load_walkforward_product_day_sources(
    date: str,
    value_code: str,
    *,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_snapshot_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    quantiles: Iterable[int] = DEFAULT_QUANTILES,
) -> WalkForwardProductDaySources:
    """Read one persisted walk-forward product-day through safe allowlists.

    The daily completion marker is validated before either daily artifact is
    trusted.  Boundary rows must be explicitly execution-safe, contain no
    target-day outcome, use the target day's exact futures contract, and have
    one strictly-prior ``source_asof_date`` shared by every requested action.
    """

    date = str(date)
    value_code = str(value_code)
    _validate_partition_key(date, value_code)
    requested_quantiles = _normalise_quantiles(quantiles)

    daily_partition = Path(daily_root) / f"Date={date}"
    marker = daily_partition / "complete.json"
    mapping_path = daily_partition / "mapping.parquet"
    fair_path = daily_partition / "causal_fair.parquet"
    boundary_path = Path(boundary_snapshot_path)
    for path in (marker, mapping_path, fair_path, boundary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Validate the small global snapshot before opening any daily or raw
    # market artifact.  A legacy/unmarked ladder therefore fails the complete
    # day batch before ``load_raw_tape_day`` can run.
    validate_rolling_boundary_artifact(
        boundary_path,
        expected_daily_root=Path(daily_root),
    )

    marker_payload = validate_completion_marker(marker)
    artifacts = marker_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"daily marker has no artifact semantics: {marker}")
    causal_semantics = artifacts.get("causal_fair.parquet")
    mapping_semantics = artifacts.get("mapping.parquet")
    if not isinstance(causal_semantics, dict) or (
        causal_semantics.get("execution_safe_causal_panel") is not True
        or causal_semantics.get("contains_target_day_outcome") is not False
    ):
        raise ValueError(f"daily causal fair is not execution-safe: {marker}")
    if not isinstance(mapping_semantics, dict) or (
        mapping_semantics.get("consumer_must_use_preopen_allowlist") is not True
    ):
        raise ValueError(f"daily mapping lacks safe-consumer semantics: {marker}")

    mapping_scan = pl.scan_parquet(mapping_path)
    _require_scan_columns(
        mapping_scan, _SAFE_MAPPING_COLUMNS, str(mapping_path)
    )
    mapping = (
        mapping_scan.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
        )
        .select(_SAFE_MAPPING_COLUMNS)
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
    if mapping.height != 1:
        raise ValueError(
            f"{date}/{value_code}: expected one exact daily mapping, "
            f"found {mapping.height}"
        )
    mapping_row = mapping.row(0, named=True)
    quote_code = str(mapping_row["QuoteCode"])
    if not quote_code or quote_code == "None":
        raise ValueError(f"{date}/{value_code}: missing exact QuoteCode")
    for column in ("spot_ref_price", "fut_ref_price", "contract_size"):
        if not _finite_positive(mapping_row[column]):
            raise ValueError(
                f"{date}/{value_code}: invalid mapping {column}"
            )

    fair_scan = pl.scan_parquet(fair_path)
    _require_scan_columns(fair_scan, _SAFE_CAUSAL_FAIR_COLUMNS, str(fair_path))
    causal_fair = (
        fair_scan.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
        )
        .select(_SAFE_CAUSAL_FAIR_COLUMNS)
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("timestamp")
            .cast(pl.Datetime("ns"))
            .alias("fair_timestamp"),
            pl.col("spot_ref_price").cast(pl.Float64),
            pl.col("fut_ref_price").cast(pl.Float64),
            pl.col("contract_size").cast(pl.Float64),
            pl.col("anchor_ewma_120s_bp").cast(pl.Float64),
        )
        .drop("timestamp")
        .collect(engine="streaming")
        .sort("fair_timestamp")
    )
    if causal_fair.is_empty():
        raise ValueError(f"{date}/{value_code}: no exact causal fair rows")
    _assert_exact_identity(
        causal_fair,
        date=date,
        value_code=value_code,
        quote_code=quote_code,
        source=str(fair_path),
    )
    if causal_fair["fair_timestamp"].null_count() or (
        causal_fair["fair_timestamp"].n_unique() != causal_fair.height
    ):
        raise ValueError(
            f"{date}/{value_code}: causal fair timestamps are null/duplicated"
        )
    # The 1-second panel can contain a few pre-first-tick nulls.  Its non-null
    # contract identity must still be exact; raw execution gates independently
    # reject those unavailable states.
    _assert_reference_identity(
        causal_fair,
        mapping_row,
        str(fair_path),
        allow_null=True,
    )

    boundary_scan = pl.scan_parquet(boundary_path)
    _require_scan_columns(
        boundary_scan, _SAFE_BOUNDARY_COLUMNS, str(boundary_path)
    )
    boundaries = (
        boundary_scan.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
            & pl.col("boundary_quantile")
            .cast(pl.Int64)
            .is_in(list(requested_quantiles))
        )
        .select(_SAFE_BOUNDARY_COLUMNS)
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("boundary_quantile").cast(pl.Int64),
            pl.col("source_asof_date").cast(pl.String),
            pl.col("price_ladder_version").cast(pl.String),
            pl.col("future_one_dollar_tick_effective_date").cast(pl.String),
            pl.col("execution_safe_snapshot").cast(pl.Boolean),
            pl.col("contains_target_day_outcome").cast(pl.Boolean),
        )
        .collect(engine="streaming")
        .sort("boundary_quantile")
    )
    actual_quantiles = boundaries["boundary_quantile"].to_list()
    if actual_quantiles != sorted(requested_quantiles):
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: rolling boundary actions "
            f"must be exactly {sorted(requested_quantiles)}, found "
            f"{actual_quantiles}"
        )
    if boundaries.select("boundary_quantile").n_unique() != boundaries.height:
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: duplicate rolling boundaries"
        )
    _assert_exact_identity(
        boundaries,
        date=date,
        value_code=value_code,
        quote_code=quote_code,
        source=str(boundary_path),
    )
    unsafe = boundaries.filter(
        (pl.col("execution_safe_snapshot") != True)  # noqa: E712
        | pl.col("execution_safe_snapshot").is_null()
        | (pl.col("contains_target_day_outcome") != False)  # noqa: E712
        | pl.col("contains_target_day_outcome").is_null()
    )
    if unsafe.height:
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: boundary snapshot is not "
            "execution-safe"
        )
    if boundaries.filter(
        pl.col("parameter_version").is_null()
        | (pl.col("parameter_version").cast(pl.String).str.len_chars() == 0)
    ).height or boundaries["parameter_version"].n_unique() != 1:
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: boundary version must be one "
            "non-null value"
        )
    if boundaries.filter(
        (pl.col("price_ladder_version") != PRICE_LADDER_VERSION)
        | pl.col("price_ladder_version").is_null()
        | (
            pl.col("future_one_dollar_tick_effective_date")
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        )
        | pl.col("future_one_dollar_tick_effective_date").is_null()
    ).height:
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: boundary price ladder "
            "lineage mismatch"
        )
    source_asof_dates = sorted(
        {
            str(value)
            for value in boundaries["source_asof_date"].to_list()
            if value is not None
        }
    )
    if (
        boundaries["source_asof_date"].null_count()
        or len(source_asof_dates) != 1
    ):
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: boundaries must share one "
            "non-null source_asof_date"
        )
    source_asof_date = source_asof_dates[0]
    _parse_yyyymmdd(source_asof_date, "source_asof_date")
    if source_asof_date >= date:
        raise ValueError(
            f"{date}/{value_code}/{quote_code}: source_asof_date must be "
            "strictly before target Date"
        )
    _assert_reference_identity(boundaries, mapping_row, str(boundary_path))

    schema_version = str(marker_payload.get("schema_version"))
    migrated = bool(marker_payload.get("migrated_legacy_marker")) or (
        schema_version == MIGRATED_DAILY_FACT_SCHEMA_VERSION
    )
    return WalkForwardProductDaySources(
        date=date,
        value_code=value_code,
        quote_code=quote_code,
        mapping=mapping,
        causal_fair=causal_fair,
        rolling_boundaries=boundaries,
        daily_schema_version=schema_version,
        migrated_daily_marker=migrated,
        source_asof_date=source_asof_date,
    )


def load_walkforward_execution_product_day(
    date: str,
    value_code: str,
    *,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_snapshot_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    data_root: Path = HFT_DATA_ROOT,
    quantiles: Iterable[int] = DEFAULT_QUANTILES,
) -> MergedTargetStudyInput:
    """Load and merge one exact raw product-day from rolling artifacts.

    This is the production-like replacement for the original eight-day pilot
    loader.  The only large frames alive together are one mapped product's raw
    tape, exact spot feature clock and roughly one day's 1-second fair anchor.
    """

    sources = load_walkforward_product_day_sources(
        date,
        value_code,
        daily_root=daily_root,
        boundary_snapshot_path=boundary_snapshot_path,
        quantiles=quantiles,
    )
    # Drop Date before joining raw feeds so no target-day realised mapping
    # fields and no duplicate date column can leak into normalized tape state.
    raw_mapping = sources.mapping.drop("Date")
    tape = load_raw_tape_day(sources.date, raw_mapping)
    spot_feature = load_spot_feature_state(
        sources.date,
        [sources.value_code],
        data_root=Path(data_root),
    )
    return _build_merged_product_day(
        sources,
        tape,
        spot_feature,
        boundary_quantiles=_normalise_quantiles(quantiles),
    )


def load_walkforward_execution_day_batch(
    date: str,
    value_codes: Sequence[str],
    *,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_snapshot_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    data_root: Path = HFT_DATA_ROOT,
    quantiles: Iterable[int] = DEFAULT_QUANTILES,
) -> WalkForwardExecutionDayBatch:
    """Load one selected day while scanning each large raw feed only once.

    All small persisted inputs are validated product-by-product *before* raw
    I/O begins.  This deliberately reuses the single-product safe loader
    rather than duplicating its allowlists and D-1 lineage checks.  Once every
    requested product is safe, their exact mappings are combined for one
    :func:`load_raw_tape_day` call and their spot feature clocks are loaded in
    one call as well.
    """

    date = str(date)
    codes = tuple(str(value) for value in value_codes)
    _validate_day_batch_keys(date, codes)
    requested_quantiles = _normalise_quantiles(quantiles)

    # Validate every small source first.  If any product has unsafe or missing
    # D-1 lineage, fail the whole batch before opening the large raw files.
    sources = tuple(
        load_walkforward_product_day_sources(
            date,
            value_code,
            daily_root=daily_root,
            boundary_snapshot_path=boundary_snapshot_path,
            quantiles=requested_quantiles,
        )
        for value_code in codes
    )
    if tuple(source.value_code for source in sources) != codes:
        raise ValueError("day-batch source loader returned different products")

    raw_mapping = pl.concat(
        [source.mapping.drop("Date") for source in sources],
        how="vertical_relaxed",
    ).sort(["ValueCode", "QuoteCode"])
    tape = load_raw_tape_day(date, raw_mapping)
    _validate_exact_raw_day_batch(tape, sources)

    spot_feature = load_spread_pair_clock_for_raw_day(
        date,
        codes,
        tape,
        data_root=Path(data_root),
    )
    _assert_exact_day_products(
        spot_feature,
        date=date,
        value_codes=codes,
        source="selected-day spot feature state",
    )
    return WalkForwardExecutionDayBatch(
        date=date,
        value_codes=codes,
        sources=sources,
        raw_tape=tape,
        spot_feature_state=spot_feature,
        boundary_quantiles=requested_quantiles,
    )


def load_spread_pair_clock_for_raw_day(
    date: str,
    value_codes: Sequence[str],
    tape: RawTapeDay,
    *,
    data_root: Path = HFT_DATA_ROOT,
) -> pl.DataFrame:
    """Join tickFeature clocks to the already-normalized spot raw keys.

    The legacy feature loader reopens ``StockTick.parquet`` to obtain its key
    set.  A day batch already owns that exact normalized raw tape, so reopening
    the large spot file would defeat batching.  This loader reads only the
    much smaller tickFeature artifact and inner-validates it against the
    selected raw ``(Date, ValueCode, ChannelSeq)`` keys.
    """

    date = str(date)
    codes = tuple(str(value) for value in value_codes)
    _validate_day_batch_keys(date, codes)
    if str(tape.date) != date:
        raise ValueError(
            f"spread clock raw tape Date mismatch: {tape.date!r} != {date!r}"
        )
    raw_keys = (
        tape.spot_states.select(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("sequence").cast(pl.UInt64).alias("spot_channel_seq"),
        )
        .sort(["ValueCode", "spot_channel_seq"])
    )
    _assert_exact_day_products(
        raw_keys,
        date=date,
        value_codes=codes,
        source="selected-day raw spot clock keys",
    )
    if raw_keys.select("ValueCode", "spot_channel_seq").n_unique() != raw_keys.height:
        raise ValueError("selected-day raw spot clock keys are duplicated")

    feature_path = Path(data_root) / "tickFeature" / f"{date}_tickFeature.parquet"
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    feature_scan = pl.scan_parquet(feature_path)
    _require_scan_columns(
        feature_scan,
        _SPREAD_CLOCK_SOURCE_COLUMNS,
        str(feature_path),
    )
    features = (
        feature_scan.filter(
            pl.col("QuoteCode").cast(pl.String).is_in(list(codes))
        )
        .select(
            pl.col("QuoteCode").cast(pl.String).alias("ValueCode"),
            pl.col("ChannelSeq").cast(pl.UInt64).alias("spot_channel_seq"),
            pl.col("SpreadPairID").cast(pl.Int64).alias("spread_pair_id"),
            pl.col("SpreadPairSeq").cast(pl.Int64).alias("spread_pair_seq"),
            pl.col("SpreadPairTotalCount")
            .cast(pl.Int64)
            .alias("spread_pair_epoch"),
            pl.col("SpreadCountAtSameCount")
            .cast(pl.Int64)
            .alias("spread_count_at_same_count"),
        )
        .collect(engine="streaming")
    )
    if (
        features.select("ValueCode", "spot_channel_seq").n_unique()
        != features.height
    ):
        raise ValueError(f"{feature_path}: spread clock keys are duplicated")
    joined = raw_keys.join(
        features,
        on=["ValueCode", "spot_channel_seq"],
        how="left",
        validate="1:1",
    )
    missing = joined.filter(pl.col("spread_pair_epoch").is_null())
    if missing.height:
        missing_sample = missing.select(
            "Date", "ValueCode", "spot_channel_seq"
        ).head(5).to_dicts()
        raise ValueError(
            "spot raw/feature clock join is incomplete: "
            f"{missing_sample}"
        )
    return joined.sort(["ValueCode", "spot_channel_seq"])


def merged_product_day_from_batch(
    batch: WalkForwardExecutionDayBatch,
    value_code: str,
) -> MergedTargetStudyInput:
    """Materialize one exact product view from a validated shared raw day."""

    value_code = str(value_code)
    if value_code not in batch.value_codes:
        raise KeyError(
            f"{batch.date}: product {value_code!r} is not in the day batch"
        )
    matches = tuple(
        source for source in batch.sources
        if source.value_code == value_code
    )
    if len(matches) != 1:
        raise ValueError(
            f"{batch.date}/{value_code}: expected one validated source, "
            f"found {len(matches)}"
        )
    sources = matches[0]
    tape = _slice_raw_tape_product(batch.raw_tape, value_code)
    spot_feature = _slice_product_frame(
        batch.spot_feature_state,
        value_code,
        source="selected-day spot feature state",
    )
    return _build_merged_product_day(
        sources,
        tape,
        spot_feature,
        boundary_quantiles=batch.boundary_quantiles,
    )


def _build_merged_product_day(
    sources: WalkForwardProductDaySources,
    tape: RawTapeDay,
    spot_feature: pl.DataFrame,
    *,
    boundary_quantiles: tuple[int, ...],
) -> MergedTargetStudyInput:
    """Build one merged input after source and raw loading are complete."""

    raw_mapping = sources.mapping.drop("Date")
    _validate_exact_raw_tape(tape, sources)
    _assert_exact_identity(
        spot_feature,
        date=sources.date,
        value_code=sources.value_code,
        quote_code=None,
        source="exact spot feature state",
    )
    observations, audit = build_merged_target_observations_from_frames(
        sources.date,
        tape.spot_states,
        tape.future_states,
        spot_feature,
        sources.causal_fair,
        sources.rolling_boundaries,
    )
    if audit.height != 1:
        raise ValueError(
            f"{sources.date}/{sources.value_code}: merged raw builder must "
            f"emit exactly one audit row, found {audit.height}"
        )
    audit = audit.with_columns(
        pl.lit(sources.quote_code).alias("QuoteCode"),
        pl.lit(sources.daily_schema_version).alias(
            "daily_source_schema_version"
        ),
        pl.lit(sources.migrated_daily_marker).alias(
            "daily_source_migrated_nonatomic"
        ),
        pl.lit(sources.source_asof_date).alias(
            "boundary_source_asof_date"
        ),
        pl.lit(",".join(map(str, boundary_quantiles))).alias(
            "boundary_quantiles_loaded"
        ),
        pl.lit(True).alias("execution_safe_source_validated"),
        pl.lit(False).alias("contains_target_day_outcome"),
    )
    return MergedTargetStudyInput(
        sources.date,
        raw_mapping,
        tape,
        observations,
        audit,
    )


def replay_execution_product_day(
    merged: MergedTargetStudyInput,
    config: ExecutionRunnerConfig = ExecutionRunnerConfig(),
    *,
    exit_rules: pl.DataFrame | None = None,
) -> ExecutionProductDayResult:
    """Replay one already-loaded product-day and emit lookup-ready facts.

    ``SpreadPairTotalCount`` semantics are inherited unchanged from
    :func:`build_layered_order_windows`: one base candidate per epoch,
    additional forward absolute prices inside the epoch, no repeated price in
    the same epoch, and a new generation when the epoch changes even if the
    absolute price is unchanged.
    """

    config.validate()
    if merged.mapping.height != 1:
        raise ValueError(
            "memory-bounded replay requires exactly one mapped product-day"
        )
    date = str(merged.date)
    value_code = str(merged.mapping.item(0, "ValueCode"))
    quote_code = str(merged.mapping.item(0, "QuoteCode"))
    cutoff = session_cutoff_cursor(date)

    trade_indexes: dict[str, IndexedTradeReplay] = {}
    for market in {ROUTE_SPECS[route].maker_market for route in config.routes}:
        tape = (
            merged.raw_tape.future_trades
            if market == "future"
            else merged.raw_tape.spot_trades
        ).filter(pl.col("ValueCode").cast(pl.String) == value_code)
        trade_indexes[market] = IndexedTradeReplay(
            _trade_events(tape, market=market)
        )

    alias_parts: list[pl.DataFrame] = []
    policy_rows: list[dict[str, object]] = []
    for route in config.routes:
        market = ROUTE_SPECS[route].maker_market
        for quantile in config.boundary_quantiles:
            action_id = f"q{quantile}"
            observations = observations_for_policy(
                merged.observations,
                date=date,
                value_code=value_code,
                route=route,
                boundary_quantile=quantile,
            )
            observations = tuple(
                observation
                for observation in observations
                if observation.cursor < cutoff
            )
            built = None
            if observations:
                policy_id = f"{date}/{value_code}/{route}/{action_id}"
                built = build_layered_order_windows(
                    observations,
                    route=route,
                    policy_id=policy_id,
                    cutoff_cursor=cutoff,
                )
                if built.windows:
                    labelled = _label_policy_windows(
                        merged.observations,
                        built.windows,
                        trade_indexes[market],
                        date=date,
                        value_code=value_code,
                        route=route,
                        boundary_quantile=quantile,
                        peak_nominal_layers=built.peak_active_layers,
                    ).with_columns(pl.lit(action_id).alias("lookup_action_id"))
                    alias_parts.append(labelled)

            transitions = built.transitions if built is not None else ()
            policy_rows.append(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "QuoteCode": quote_code,
                    "route": route,
                    "lookup_action_id": action_id,
                    "boundary_quantile": quantile,
                    "target_observations": len(observations),
                    "submitted_generations": sum(
                        transition.kind == "submit" for transition in transitions
                    ),
                    "nominal_cancel_transitions": sum(
                        transition.kind == "cancel" for transition in transitions
                    ),
                    "suppressed_seen_price_transitions": sum(
                        transition.kind == "suppress" for transition in transitions
                    ),
                    "nominal_target_retreat_transitions": sum(
                        transition.kind == "cancel"
                        and transition.reason == "target_retreat"
                        for transition in transitions
                    ),
                    "nominal_session_cutoff_transitions": sum(
                        transition.kind == "cancel"
                        and transition.reason == "session_cutoff"
                        for transition in transitions
                    ),
                    "peak_nominal_layers": (
                        built.peak_active_layers if built is not None else 0
                    ),
                    "spread_pair_clock": "SpreadPairTotalCount",
                    "same_epoch_seen_price_resubmitted": False,
                    "cross_epoch_same_price_allowed": True,
                }
            )

    aliases = _concat(alias_parts)
    raw_facts = (
        summarize_raw_order_facts(aliases)
        if not aliases.is_empty()
        else _empty_raw_facts()
    )
    if aliases.is_empty():
        hedge_facts = _empty_hedge_facts()
    else:
        hedge_facts = run_hedge_study(
            aliases,
            raw_facts,
            merged.raw_tape,
            hedge_delay_ns=config.hedge_delay_ns,
        ).hedge_facts
    action_facts = build_execution_action_facts(aliases, hedge_facts)
    policy_audit = pl.from_dicts(policy_rows, infer_schema_length=None)
    daily = summarize_execution_daily(action_facts, policy_audit)

    exit_facts = _empty_exit_facts()
    exit_daily_facts = _empty_exit_daily_facts()
    if exit_rules is not None and not exit_rules.is_empty():
        full_hedged = action_facts.filter(
            pl.col("entry_hedge_executable")
        )
        if not full_hedged.is_empty():
            start_ns = int(
                full_hedged["entry_hedge_decision_time_ns"].drop_nulls().min()
            )
            if start_ns < cutoff.recv_time_ns:
                exit_path = build_executable_taker_exit_path(
                    merged.raw_tape,
                    value_code,
                    start_time_ns=start_ns,
                    cutoff_time_ns=cutoff.recv_time_ns,
                    config=config.exit_path,
                )
                exit_facts = build_executable_exit_facts(
                    action_facts,
                    exit_rules,
                    exit_path,
                )
                exit_daily_facts = summarize_executable_exit_daily(
                    action_facts, exit_facts
                )

    return ExecutionProductDayResult(
        aliases,
        raw_facts,
        hedge_facts,
        action_facts,
        daily,
        exit_facts,
        exit_daily_facts,
        policy_audit,
        merged.audit,
        merged.raw_tape.audit,
    )


def run_partitioned_execution_replay(
    product_days: Iterable[tuple[str, str]],
    loader: ProductDayLoader,
    output_dir: Path,
    config: ExecutionRunnerConfig = ExecutionRunnerConfig(),
    *,
    exit_rule_loader: ExitRuleLoader | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Run and atomically publish one partition at a time.

    Only the small manifest rows survive between iterations.  Complete
    partitions may be resumed after their config hash and file hashes are
    verified.  Existing incomplete or mismatched partitions are never
    overwritten implicitly.
    """

    config.validate()
    keys = [(str(date), str(value_code)) for date, value_code in product_days]
    if len(keys) != len(set(keys)):
        raise ValueError("product_days contains duplicate keys")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload(config)
    config_sha256 = _canonical_sha256(config_payload)
    manifest_rows: list[dict[str, object]] = []

    for date, value_code in keys:
        _validate_partition_key(date, value_code)
        partition = root / f"Date={date}" / f"ValueCode={value_code}"
        if partition.exists():
            if not resume:
                raise FileExistsError(partition)
            manifest_rows.append(
                _verify_complete_partition(partition, config_sha256)
            )
            continue

        merged = loader(date, value_code)
        actual_codes = merged.mapping["ValueCode"].cast(pl.String).to_list()
        if str(merged.date) != date or actual_codes != [value_code]:
            raise ValueError(
                "product-day loader returned a different date or product"
            )
        result = replay_execution_product_day(
            merged,
            config,
        )
        if exit_rule_loader is not None:
            # Rules may now key exact policy-generation IDs, but their own
            # lineage validator still requires strictly D-1-safe inputs.
            rules = exit_rule_loader(
                date, value_code, result.execution_action_facts
            )
            if rules is not None and not rules.is_empty():
                result = _attach_exits(
                    result,
                    merged,
                    config,
                    rules,
                )
        payload = _publish_partition(
            partition,
            result,
            config_payload=config_payload,
            config_sha256=config_sha256,
        )
        manifest_rows.append(payload)
        # ``merged`` and every detailed frame become unreachable here before
        # the next loader call; only one compact manifest row is retained.

    return _write_execution_manifest(root, manifest_rows)


def run_day_batched_execution_replay(
    product_days: Iterable[tuple[str, str]],
    loader: ExecutionDayBatchLoader,
    output_dir: Path,
    config: ExecutionRunnerConfig = ExecutionRunnerConfig(),
    *,
    exit_rule_loader: ExitRuleLoader | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Replay product-days while loading each selected raw day only once.

    Keys are grouped by date without changing their product-day identity.
    Existing complete partitions are verified before a batch is loaded and
    omitted from that day's raw selection.  Each pending product is then
    sliced from the shared normalized day, replayed independently and
    published through the exact same atomic partition writer used by
    :func:`run_partitioned_execution_replay`.
    """

    config.validate()
    keys = [(str(date), str(value_code)) for date, value_code in product_days]
    if len(keys) != len(set(keys)):
        raise ValueError("product_days contains duplicate keys")
    by_date: dict[str, list[str]] = {}
    for date, value_code in keys:
        _validate_partition_key(date, value_code)
        by_date.setdefault(date, []).append(value_code)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload(config)
    config_sha256 = _canonical_sha256(config_payload)
    manifest_rows: list[dict[str, object]] = []

    for date, value_codes in by_date.items():
        pending: list[str] = []
        for value_code in value_codes:
            partition = root / f"Date={date}" / f"ValueCode={value_code}"
            if partition.exists():
                if not resume:
                    raise FileExistsError(partition)
                manifest_rows.append(
                    _verify_complete_partition(partition, config_sha256)
                )
            else:
                pending.append(value_code)
        if not pending:
            continue

        batch = loader(date, tuple(pending))
        _validate_loaded_day_batch(
            batch,
            date=date,
            value_codes=pending,
            boundary_quantiles=_normalise_quantiles(
                config.boundary_quantiles
            ),
        )
        for value_code in pending:
            merged = merged_product_day_from_batch(batch, value_code)
            actual_codes = merged.mapping["ValueCode"].cast(pl.String).to_list()
            if str(merged.date) != date or actual_codes != [value_code]:
                raise ValueError(
                    "day-batch product slice returned a different date or product"
                )
            result = replay_execution_product_day(merged, config)
            if exit_rule_loader is not None:
                rules = exit_rule_loader(
                    date, value_code, result.execution_action_facts
                )
                if rules is not None and not rules.is_empty():
                    result = _attach_exits(result, merged, config, rules)
            partition = root / f"Date={date}" / f"ValueCode={value_code}"
            manifest_rows.append(
                _publish_partition(
                    partition,
                    result,
                    config_payload=config_payload,
                    config_sha256=config_sha256,
                )
            )
            # The shared normalized day stays alive, but this product's
            # observations, replay indexes and output frames need not.
            del merged, result
        del batch

    return _write_execution_manifest(root, manifest_rows)


def run_narrow_universe_execution_replay(
    dates: Sequence[str],
    symbols: Sequence[str],
    output_dir: Path,
    config: ExecutionRunnerConfig = ExecutionRunnerConfig(),
    *,
    merged_builder_kwargs: Mapping[str, object] | None = None,
    exit_rule_loader: ExitRuleLoader | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Replay a narrowed universe from validated rolling walk-forward inputs.

    ``merged_builder_kwargs`` is retained as the public keyword for backward
    compatibility, but now configures
    :func:`load_walkforward_execution_day_batch` (daily root, rolling boundary
    path and raw spot-feature data root).  Quantiles always come from ``config``
    so the persisted boundary rows and replay action grid cannot silently
    disagree.
    """

    kwargs = dict(merged_builder_kwargs or {})
    if "quantiles" in kwargs:
        raise ValueError(
            "quantiles must be set through ExecutionRunnerConfig, not "
            "merged_builder_kwargs"
        )

    def loader(
        date: str, value_codes: Sequence[str]
    ) -> WalkForwardExecutionDayBatch:
        return load_walkforward_execution_day_batch(
            date,
            value_codes,
            quantiles=config.boundary_quantiles,
            **kwargs,
        )

    keys = (
        (str(date), str(symbol))
        for date in dates
        for symbol in symbols
    )
    return run_day_batched_execution_replay(
        keys,
        loader,
        output_dir,
        config,
        exit_rule_loader=exit_rule_loader,
        resume=resume,
    )


def _attach_exits(
    result: ExecutionProductDayResult,
    merged: MergedTargetStudyInput,
    config: ExecutionRunnerConfig,
    exit_rules: pl.DataFrame,
) -> ExecutionProductDayResult:
    full_hedged = result.execution_action_facts.filter(
        pl.col("entry_hedge_executable")
    )
    if full_hedged.is_empty():
        return result
    decision_times = full_hedged[
        "entry_hedge_decision_time_ns"
    ].drop_nulls()
    if decision_times.is_empty():
        return result
    start_ns = int(decision_times.min())
    cutoff_ns = session_cutoff_cursor(str(merged.date)).recv_time_ns
    if start_ns >= cutoff_ns:
        return result
    value_code = str(merged.mapping.item(0, "ValueCode"))
    exit_path = build_executable_taker_exit_path(
        merged.raw_tape,
        value_code,
        start_time_ns=start_ns,
        cutoff_time_ns=cutoff_ns,
        config=config.exit_path,
    )
    exit_facts = build_executable_exit_facts(
        result.execution_action_facts,
        exit_rules,
        exit_path,
    )
    return replace(
        result,
        exit_facts=exit_facts,
        exit_daily_facts=summarize_executable_exit_daily(
            result.execution_action_facts, exit_facts
        ),
    )


def _publish_partition(
    partition: Path,
    result: ExecutionProductDayResult,
    *,
    config_payload: dict[str, object],
    config_sha256: str,
) -> dict[str, object]:
    parent = partition.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}.tmp-", dir=parent)
    )
    frames = {
        "order_aliases.parquet": result.order_aliases,
        "raw_order_facts.parquet": result.raw_order_facts,
        "hedge_facts.parquet": result.hedge_facts,
        "execution_action_facts.parquet": result.execution_action_facts,
        "execution_daily_facts.parquet": result.execution_daily_facts,
        "exit_facts.parquet": result.exit_facts,
        "exit_daily_facts.parquet": result.exit_daily_facts,
        "policy_audit.parquet": result.policy_audit,
        "target_audit.parquet": result.target_audit,
        "raw_tape_audit.parquet": result.raw_tape_audit,
    }
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = stage / filename
            frame.write_parquet(path)
            artifacts[filename] = {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        date = str(result.policy_audit.item(0, "Date"))
        value_code = str(result.policy_audit.item(0, "ValueCode"))
        complete = {
            "complete": True,
            "Date": date,
            "ValueCode": value_code,
            "runner_version": EXECUTION_RUNNER_VERSION,
            "config": config_payload,
            "config_sha256": config_sha256,
            "artifacts": artifacts,
            "fact_semantics": {
                "spread_pair_clock": "SpreadPairTotalCount",
                "same_epoch_same_price_resubmit": False,
                "cross_epoch_same_price_add": True,
                "same_absolute_price_alias_key": "raw_order_fact_id",
                "hedge_delay_ns": config_payload["hedge_delay_ns"],
                "price_ladder_version": config_payload[
                    "price_ladder_version"
                ],
                "future_one_dollar_tick_effective_date": config_payload[
                    "future_one_dollar_tick_effective_date"
                ],
                "independent_event_label": True,
                "joint_volume_allocated": False,
                "exit_style": "optional_immediate_taker_taker",
            },
            "missing_for_net_ev": [
                "fees_and_same_day_tax_profile",
                "overnight_financing_and_next_session_realization",
                "shared_visible_volume_allocation",
                "portfolio_inventory_and_capital_replay",
                "maker_exit_fill_model",
                "spot_partial_incremental_quantity_path",
            ],
        }
        (stage / "complete.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if partition.exists():
            raise FileExistsError(partition)
        stage.replace(partition)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return _manifest_row(partition, complete)


def _verify_complete_partition(
    partition: Path, config_sha256: str
) -> dict[str, object]:
    marker = partition / "complete.json"
    if not marker.is_file():
        raise FileExistsError(
            f"existing partition is incomplete and will not be overwritten: {partition}"
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError(f"partition marker is not complete: {partition}")
    if payload.get("config_sha256") != config_sha256:
        raise ValueError(f"partition config mismatch: {partition}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"partition artifact manifest is invalid: {partition}")
    for filename, metadata in artifacts.items():
        path = partition / filename
        if not path.is_file() or not isinstance(metadata, dict):
            raise ValueError(f"partition artifact is missing: {path}")
        if _file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"partition artifact hash mismatch: {path}")
    return _manifest_row(partition, payload)


def _manifest_row(
    partition: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    row_counts = {
        filename.removesuffix(".parquet").replace("-", "_") + "_rows": int(
            metadata["rows"]
        )
        for filename, metadata in artifacts.items()
        if isinstance(metadata, dict)
    }
    return {
        "Date": str(payload["Date"]),
        "ValueCode": str(payload["ValueCode"]),
        "partition": str(partition),
        "config_sha256": str(payload["config_sha256"]),
        "complete": True,
        **row_counts,
    }


def _write_execution_manifest(
    root: Path,
    manifest_rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Atomically rebuild the compact root manifest for either runner."""

    manifest = (
        pl.from_dicts(manifest_rows, infer_schema_length=None)
        if manifest_rows
        else pl.DataFrame()
    )
    if manifest.is_empty():
        return manifest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".execution_partition_manifest.",
        suffix=".tmp.parquet",
        dir=root,
    )
    os.close(descriptor)
    try:
        Path(temporary_name).unlink()
        manifest.sort(["Date", "ValueCode"]).write_parquet(temporary_name)
        Path(temporary_name).replace(
            root / "execution_partition_manifest.parquet"
        )
    finally:
        # Concurrent disjoint workers use distinct temporary manifests; the
        # final single-worker resume rebuilds the complete index.
        Path(temporary_name).unlink(missing_ok=True)
    return manifest


def _config_payload(config: ExecutionRunnerConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["routes"] = list(config.routes)
    payload["boundary_quantiles"] = list(config.boundary_quantiles)
    return payload


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_quantiles(values: Iterable[int]) -> tuple[int, ...]:
    quantiles = tuple(values)
    if (
        not quantiles
        or len(quantiles) != len(set(quantiles))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value >= 100
            for value in quantiles
        )
    ):
        raise ValueError(
            "quantiles must be unique integers strictly between 0 and 100"
        )
    return tuple(sorted(quantiles))


def _require_scan_columns(
    frame: pl.LazyFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.collect_schema().names()))
    if missing:
        raise ValueError(f"{source} missing safe input columns: {missing}")


def _parse_yyyymmdd(value: str, label: str) -> datetime:
    try:
        if len(value) != 8 or not value.isdigit():
            raise ValueError
        return datetime.strptime(value, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _assert_exact_identity(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    quote_code: str | None,
    source: str,
) -> None:
    if frame.is_empty():
        raise ValueError(f"{source} is empty")
    expected = {"Date": str(date), "ValueCode": str(value_code)}
    if quote_code is not None:
        expected["QuoteCode"] = str(quote_code)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    for column, wanted in expected.items():
        values = frame[column].cast(pl.String)
        actual = {
            str(value) for value in values.unique().to_list() if value is not None
        }
        if values.null_count() or actual != {wanted}:
            raise ValueError(
                f"{source} {column} must be exactly {wanted!r}, found "
                f"{sorted(actual)!r}"
            )


def _assert_reference_identity(
    frame: pl.DataFrame,
    mapping_row: Mapping[str, object],
    source: str,
    *,
    allow_null: bool = False,
) -> None:
    columns = ("spot_ref_price", "fut_ref_price", "contract_size")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing contract identity: {missing}")
    for column in columns:
        expected = float(mapping_row[column])
        tolerance = max(1e-9, abs(expected) * 1e-12)
        values = pl.col(column).cast(pl.Float64)
        invalid_expression = (~values.is_finite()) | (
            (values - expected).abs() > tolerance
        )
        if allow_null:
            invalid_expression = values.is_not_null() & invalid_expression
        else:
            invalid_expression = values.is_null() | invalid_expression
        invalid = frame.filter(
            invalid_expression
        )
        if invalid.height or frame[column].drop_nulls().is_empty():
            raise ValueError(
                f"{source} {column} does not match the exact daily mapping"
            )


def _validate_exact_raw_tape(
    tape: RawTapeDay,
    sources: WalkForwardProductDaySources,
) -> None:
    if str(tape.date) != sources.date:
        raise ValueError(
            f"raw tape Date mismatch: {tape.date!r} != {sources.date!r}"
        )
    if tape.mapping.height != 1:
        raise ValueError("raw tape mapping must contain exactly one product")
    mapping = tape.mapping.with_columns(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if (
        mapping.item(0, "ValueCode") != sources.value_code
        or mapping.item(0, "QuoteCode") != sources.quote_code
    ):
        raise ValueError("raw tape mapping does not match validated contract")
    _assert_reference_identity(
        mapping,
        sources.mapping.row(0, named=True),
        "raw tape mapping",
    )
    for label, frame in (
        ("raw spot state", tape.spot_states),
        ("raw futures state", tape.future_states),
    ):
        _assert_exact_identity(
            frame,
            date=sources.date,
            value_code=sources.value_code,
            quote_code=sources.quote_code,
            source=label,
        )


def _validate_day_batch_keys(date: str, value_codes: Sequence[str]) -> None:
    _parse_yyyymmdd(date, "YYYYMMDD batch date")
    if not value_codes:
        raise ValueError("day-batch value_codes must be nonempty")
    if len(value_codes) != len(set(value_codes)):
        raise ValueError("day-batch value_codes must be unique")
    for value_code in value_codes:
        _validate_partition_key(date, value_code)


def _assert_exact_day_products(
    frame: pl.DataFrame,
    *,
    date: str,
    value_codes: Sequence[str],
    source: str,
) -> None:
    """Require exactly the selected date/product set, allowing many rows."""

    required = {"Date", "ValueCode"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    dates = frame["Date"].cast(pl.String)
    products = frame["ValueCode"].cast(pl.String)
    actual_dates = {
        str(value) for value in dates.unique().to_list() if value is not None
    }
    actual_products = {
        str(value) for value in products.unique().to_list() if value is not None
    }
    expected_products = set(map(str, value_codes))
    if dates.null_count() or actual_dates != {str(date)}:
        raise ValueError(
            f"{source} Date must be exactly {date!r}, found "
            f"{sorted(actual_dates)!r}"
        )
    if products.null_count() or actual_products != expected_products:
        raise ValueError(
            f"{source} products must be exactly "
            f"{sorted(expected_products)!r}, found {sorted(actual_products)!r}"
        )


def _slice_product_frame(
    frame: pl.DataFrame,
    value_code: str,
    *,
    source: str,
) -> pl.DataFrame:
    if "ValueCode" not in frame.columns:
        raise ValueError(f"{source} missing identity column: ValueCode")
    return frame.filter(
        pl.col("ValueCode").cast(pl.String) == str(value_code)
    )


def _slice_raw_tape_product(
    tape: RawTapeDay,
    value_code: str,
) -> RawTapeDay:
    """Create one exact product view without reopening either raw file."""

    value_code = str(value_code)
    return RawTapeDay(
        date=str(tape.date),
        mapping=_slice_product_frame(
            tape.mapping, value_code, source="selected-day raw mapping"
        ),
        spot_states=_slice_product_frame(
            tape.spot_states, value_code, source="selected-day spot state"
        ),
        future_states=_slice_product_frame(
            tape.future_states, value_code, source="selected-day future state"
        ),
        spot_trades=_slice_product_frame(
            tape.spot_trades, value_code, source="selected-day spot trades"
        ),
        future_trades=_slice_product_frame(
            tape.future_trades, value_code, source="selected-day future trades"
        ),
        audit=_slice_product_frame(
            tape.audit, value_code, source="selected-day raw audit"
        ),
    )


def _validate_exact_raw_day_batch(
    tape: RawTapeDay,
    sources: Sequence[WalkForwardProductDaySources],
) -> None:
    if not sources:
        raise ValueError("raw day batch has no validated sources")
    date = sources[0].date
    if str(tape.date) != date:
        raise ValueError(f"raw tape Date mismatch: {tape.date!r} != {date!r}")
    expected_codes = {source.value_code for source in sources}
    for label, frame, require_all in (
        ("selected-day raw mapping", tape.mapping, True),
        ("selected-day spot state", tape.spot_states, True),
        ("selected-day future state", tape.future_states, True),
        ("selected-day spot trades", tape.spot_trades, False),
        ("selected-day future trades", tape.future_trades, False),
        ("selected-day raw audit", tape.audit, True),
    ):
        _assert_selected_value_codes(
            frame,
            expected_codes,
            source=label,
            require_all=require_all,
        )
    for source in sources:
        if source.date != date:
            raise ValueError("day-batch sources contain multiple dates")
        _validate_exact_raw_tape(
            _slice_raw_tape_product(tape, source.value_code),
            source,
        )


def _assert_selected_value_codes(
    frame: pl.DataFrame,
    expected_codes: set[str],
    *,
    source: str,
    require_all: bool,
) -> None:
    if "ValueCode" not in frame.columns:
        raise ValueError(f"{source} missing identity column: ValueCode")
    values = frame["ValueCode"].cast(pl.String)
    actual = {
        str(value) for value in values.unique().to_list() if value is not None
    }
    invalid = values.null_count() or not actual.issubset(expected_codes)
    if require_all:
        invalid = invalid or actual != expected_codes
    if invalid:
        relationship = "exactly" if require_all else "a subset of"
        raise ValueError(
            f"{source} products must be {relationship} "
            f"{sorted(expected_codes)!r}, found {sorted(actual)!r}"
        )


def _validate_loaded_day_batch(
    batch: WalkForwardExecutionDayBatch,
    *,
    date: str,
    value_codes: Sequence[str],
    boundary_quantiles: tuple[int, ...],
) -> None:
    expected_codes = tuple(map(str, value_codes))
    if str(batch.date) != str(date) or batch.value_codes != expected_codes:
        raise ValueError("day-batch loader returned a different date or products")
    if batch.boundary_quantiles != boundary_quantiles:
        raise ValueError("day-batch loader returned different boundary quantiles")
    if tuple(source.value_code for source in batch.sources) != expected_codes:
        raise ValueError("day-batch sources do not match requested products")
    _validate_exact_raw_day_batch(batch.raw_tape, batch.sources)
    _assert_exact_day_products(
        batch.spot_feature_state,
        date=date,
        value_codes=expected_codes,
        source="selected-day spot feature state",
    )


def _validate_partition_key(date: str, value_code: str) -> None:
    _parse_yyyymmdd(date, "YYYYMMDD partition date")
    if not _SAFE_KEY.fullmatch(value_code):
        raise ValueError(f"unsafe ValueCode partition key: {value_code}")


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [frame for frame in frames if not frame.is_empty()]
    return (
        pl.concat(nonempty, how="diagonal_relaxed")
        if nonempty
        else _empty_aliases()
    )


def _empty_aliases() -> pl.DataFrame:
    # Complete enough for :func:`build_execution_action_facts` to return its
    # stable empty schema without asking the hedge layer to infer columns.
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "boundary_quantile": pl.Int64,
            "raw_order_fact_id": pl.String,
            "policy_generation_id": pl.String,
        }
    )


def _empty_raw_facts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "raw_order_fact_id": pl.String,
        }
    )


def _empty_hedge_facts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "raw_order_fact_id": pl.String,
            "status": pl.String,
        }
    )


def _empty_exit_facts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "raw_order_fact_id": pl.String,
            "policy_generation_id": pl.String,
            "exit_rule_id": pl.String,
            "branch_status": pl.String,
            "same_day_exit": pl.Boolean,
            "overnight_carry": pl.Boolean,
            "terminal_outcome": pl.Boolean,
            "needs_next_session_label": pl.Boolean,
            "gross_cycle_pnl_twd": pl.Float64,
            "eod_liquidation_gross_pnl_twd": pl.Float64,
            "pathwise_ev_ready": pl.Boolean,
        }
    )


def _empty_exit_daily_facts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "route": pl.String,
            "lookup_action_id": pl.String,
            "boundary_quantile": pl.Int64,
            "exit_rule_id": pl.String,
            "exit_rule_aliases": pl.Int64,
            "same_day_taker_exits": pl.Int64,
            "overnight_carry_branches": pl.Int64,
            "pathwise_ev_ready": pl.Boolean,
        }
    )
