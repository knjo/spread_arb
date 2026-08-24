"""Build expiry paired-close facts for the causal dynamic fill population.

The population is derived directly from the canonical one-second makerFill
candidate bundle.  It is never intersected with the legacy fixed-45 product
universe.  The spot leg uses the local daily ``close_price`` field.  In the
absence of a local official futures daily-close table, the futures leg uses
the final positive, non-TrialMatch trade in the day session and labels that
price as a last-trade proxy, not an official close or settlement price.
"""

from __future__ import annotations

import argparse
from datetime import date as date_type
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl


VERSION = "dynamic_expiry_paired_close_facts_v1"
DEFAULT_CANDIDATE_ROOT = Path(
    "maker/data/walkforward/one_second_makerfill_causal_v2_20260822_v1"
)
DEFAULT_SPOT_DAILY_ROOT = Path("/home/kevin/Project/HFT/data/marketData")
DEFAULT_FUTURES_RAW_ROOT = Path("/mnt/NAS/Parquet/Ticks")
DEFAULT_DB_CROSSCHECK_PATH = Path(
    "maker/data/walkforward/expiry_daily_close_facts_20260821_v1/"
    "daily_close_facts.parquet"
)
DEFAULT_OUTPUT = Path(
    "maker/data/walkforward/dynamic_expiry_paired_close_facts_20260822_v1"
)

# Expiring stock-futures contracts in the available raw files stop at 13:30.
# A broad day-session upper bound excludes the next night session without
# assuming that the final trade must occur exactly at the closing auction.
DAY_SESSION_START = time(8, 45)
DAY_SESSION_END_EXCLUSIVE = time(14, 0)

_POPULATION_REQUIRED = {
    "end_date",
    "Date",
    "ValueCode",
    "QuoteCode",
    "full_fill",
}
_SPOT_REQUIRED = {"date", "quote_code", "close_price"}
_FUTURE_REQUIRED = {
    "RecvTime",
    "TransTime",
    "QuoteCode",
    "ValueCode",
    "PacketSeq",
    "ChannelSeq",
    "TrialMatch",
    "DecimalLocator",
    "FillPrice",
    "FillLots",
}
_KEY = ["Date", "ValueCode", "QuoteCode"]


def _require_columns(
    frame: pl.DataFrame, required: set[str], label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _date_text_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    dtype = frame.schema[column]
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return pl.col(column).dt.strftime("%Y%m%d")
    return (
        pl.col(column)
        .cast(pl.String)
        .str.replace_all("-", "")
        .str.slice(0, 8)
    )


def derive_dynamic_expiry_population(candidates: pl.DataFrame) -> pl.DataFrame:
    """Return unique expiry pairs and fill counts from filled candidates."""

    _require_columns(candidates, _POPULATION_REQUIRED, "candidate outcomes")
    filled = candidates.filter(pl.col("full_fill").fill_null(False)).with_columns(
        _date_text_expr(candidates, "end_date").alias("expiry_date"),
        _date_text_expr(candidates, "Date").alias("entry_date"),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if filled.is_empty():
        raise ValueError("candidate outcomes contain no full fills")
    invalid = filled.filter(
        pl.any_horizontal(
            pl.col("expiry_date").is_null(),
            pl.col("expiry_date").str.len_chars() != 8,
            pl.col("ValueCode").is_null(),
            pl.col("QuoteCode").is_null(),
        )
    )
    if invalid.height:
        raise ValueError("filled candidates contain invalid expiry identities")
    population = (
        filled.group_by(
            pl.col("expiry_date").alias("Date"), "ValueCode", "QuoteCode"
        )
        .agg(
            pl.len().cast(pl.Int64).alias("source_fill_count"),
            pl.col("entry_date").min().alias("first_entry_date"),
            pl.col("entry_date").max().alias("last_entry_date"),
        )
        .sort(_KEY)
    )
    if population.select(_KEY).n_unique() != population.height:
        raise ValueError("dynamic expiry population is not key-unique")
    if int(population["source_fill_count"].sum()) != filled.height:
        raise ValueError("dynamic expiry population does not reconcile to fills")
    return population


def load_dynamic_expiry_population(
    candidate_root: Path,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load and validate the canonical candidate bundle before deriving pairs."""

    candidate_root = Path(candidate_root)
    marker_path = candidate_root / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("complete", True) is False:
        raise ValueError(f"candidate bundle is incomplete: {candidate_root}")
    paths = sorted(
        (candidate_root / "candidate_outcomes").glob(
            "Date=*/candidate_outcomes.parquet"
        )
    )
    if not paths:
        raise FileNotFoundError(f"candidate partitions not found: {candidate_root}")
    expected_partitions = marker.get("artifacts", {}).get("candidate_partitions")
    if expected_partitions is not None and len(paths) != int(expected_partitions):
        raise ValueError("candidate partition count changed")
    candidates = pl.scan_parquet([str(path) for path in paths]).select(
        "end_date", "Date", "ValueCode", "QuoteCode", "full_fill"
    ).collect()
    population = derive_dynamic_expiry_population(candidates)
    expected_fills = marker.get("approximate_fills")
    actual_fills = int(population["source_fill_count"].sum())
    if expected_fills is not None and actual_fills != int(expected_fills):
        raise ValueError(
            f"filled candidate count changed: expected={expected_fills}, "
            f"actual={actual_fills}"
        )
    return population, marker


def extract_spot_daily_closes(
    date: str, value_codes: Sequence[str], source_path: Path
) -> pl.DataFrame:
    """Extract one official spot daily-close field per selected product."""

    source_path = Path(source_path)
    if not source_path.is_file():
        return pl.DataFrame(
            schema={
                "ValueCode": pl.String,
                "spot_close_price": pl.Float64,
            }
        )
    schema = pl.read_parquet_schema(source_path)
    missing = sorted(_SPOT_REQUIRED - set(schema))
    if missing:
        raise ValueError(f"spot daily source lacks columns {missing}: {source_path}")
    trade_date = datetime.strptime(str(date), "%Y%m%d").date()
    frame = (
        pl.scan_parquet(source_path)
        .filter(
            (pl.col("date") == pl.lit(trade_date))
            & pl.col("quote_code").cast(pl.String).is_in(list(value_codes))
        )
        .select(
            pl.col("quote_code").cast(pl.String).alias("ValueCode"),
            pl.col("close_price").cast(pl.Float64).alias("spot_close_price"),
        )
        .collect()
    )
    if frame.select("ValueCode").n_unique() != frame.height:
        raise ValueError(f"spot daily source has duplicate products: {source_path}")
    return frame.with_columns(
        pl.when(
            pl.col("spot_close_price").is_finite()
            & (pl.col("spot_close_price") > 0)
        )
        .then(pl.col("spot_close_price"))
        .otherwise(None)
        .alias("spot_close_price")
    ).sort("ValueCode")


def extract_future_last_trades(
    date: str,
    quote_codes: Sequence[str],
    source_path: Path,
    *,
    session_start: time = DAY_SESSION_START,
    session_end_exclusive: time = DAY_SESSION_END_EXCLUSIVE,
) -> pl.DataFrame:
    """Extract the last valid non-TrialMatch futures trade in the day session."""

    source_path = Path(source_path)
    if not source_path.is_file():
        return pl.DataFrame(
            schema={
                "QuoteCode": pl.String,
                "raw_future_value_code": pl.String,
                "future_last_trade_price": pl.Float64,
                "future_last_trade_time": pl.Datetime("us"),
                "future_last_trade_recv_time_ns": pl.Int64,
                "future_last_trade_channel_seq": pl.UInt64,
                "future_last_trade_packet_seq": pl.UInt64,
                "future_last_trade_fill_lots": pl.Int64,
                "future_last_trade_decimal_locator": pl.Int16,
            }
        )
    schema = pl.read_parquet_schema(source_path)
    missing = sorted(_FUTURE_REQUIRED - set(schema))
    if missing:
        raise ValueError(f"future raw source lacks columns {missing}: {source_path}")
    trade_date = datetime.strptime(str(date), "%Y%m%d").date()
    local_time = pl.col("TransTime").dt.time()
    divisor = pl.lit(10.0).pow(pl.col("DecimalLocator").cast(pl.Float64))
    trades = (
        pl.scan_parquet(source_path)
        .filter(
            pl.col("QuoteCode").cast(pl.String).is_in(list(quote_codes))
            & (pl.col("TransTime").dt.date() == pl.lit(trade_date))
            & (local_time >= pl.lit(session_start))
            & (local_time < pl.lit(session_end_exclusive))
            & (pl.col("TrialMatch").fill_null(0) == 0)
            & (pl.col("FillLots").fill_null(0) > 0)
            & (pl.col("FillPrice").fill_null(0) > 0)
            & pl.col("DecimalLocator").is_not_null()
        )
        .select(
            pl.col("QuoteCode").cast(pl.String),
            pl.col("ValueCode").cast(pl.String).alias("raw_future_value_code"),
            (pl.col("FillPrice").cast(pl.Float64) / divisor).alias(
                "future_last_trade_price"
            ),
            pl.col("TransTime")
            .cast(pl.Datetime("us"))
            .alias("future_last_trade_time"),
            pl.col("RecvTime")
            .cast(pl.Int64)
            .alias("future_last_trade_recv_time_ns"),
            pl.col("ChannelSeq")
            .cast(pl.UInt64)
            .alias("future_last_trade_channel_seq"),
            pl.col("PacketSeq")
            .cast(pl.UInt64)
            .alias("future_last_trade_packet_seq"),
            pl.col("FillLots")
            .cast(pl.Int64)
            .alias("future_last_trade_fill_lots"),
            pl.col("DecimalLocator")
            .cast(pl.Int16)
            .alias("future_last_trade_decimal_locator"),
        )
        .collect()
    )
    if trades.is_empty():
        return trades
    result = (
        trades.sort(
            [
                "QuoteCode",
                "future_last_trade_time",
                "future_last_trade_recv_time_ns",
                "future_last_trade_channel_seq",
                "future_last_trade_packet_seq",
            ]
        )
        .group_by("QuoteCode", maintain_order=True)
        .tail(1)
        .sort("QuoteCode")
    )
    invalid = result.filter(
        pl.col("future_last_trade_price").is_null()
        | ~pl.col("future_last_trade_price").is_finite()
        | (pl.col("future_last_trade_price") <= 0)
    )
    if invalid.height:
        raise ValueError(f"invalid last futures trade in {source_path}")
    if result["QuoteCode"].n_unique() != result.height:
        raise ValueError(f"future last-trade result is not unique: {source_path}")
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _source_inventory(
    dates: Sequence[str], spot_root: Path, futures_root: Path
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for date in dates:
        spot = spot_root / f"{date}_marketData.parquet"
        future = (
            futures_root
            / date[:4]
            / date[4:6]
            / date[6:8]
            / "stock_futures.parquet"
        )
        rows.append(
            {
                "Date": date,
                "spot_source_path": str(spot.resolve()),
                "spot_source_exists": spot.is_file(),
                "spot_source_bytes": spot.stat().st_size if spot.is_file() else None,
                "spot_source_mtime_ns": (
                    spot.stat().st_mtime_ns if spot.is_file() else None
                ),
                "future_source_path": str(future.resolve()),
                "future_source_exists": future.is_file(),
                "future_source_bytes": (
                    future.stat().st_size if future.is_file() else None
                ),
                "future_source_mtime_ns": (
                    future.stat().st_mtime_ns if future.is_file() else None
                ),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("Date")


def load_db_crosscheck_facts(path: Path | None) -> pl.DataFrame:
    """Load optional pre-extracted official DB facts for overlap validation."""

    schema = {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "db_spot_close_price": pl.Float64,
        "db_future_close_price": pl.Float64,
        "db_source_identity_sha256": pl.String,
    }
    if path is None or not Path(path).is_file():
        return pl.DataFrame(schema=schema)
    source = pl.read_parquet(path)
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "spot_close_price",
        "future_close_price",
        "source_identity_sha256",
    }
    _require_columns(source, required, "DB daily-close crosscheck")
    result = source.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("spot_close_price").cast(pl.Float64).alias("db_spot_close_price"),
        pl.col("future_close_price")
        .cast(pl.Float64)
        .alias("db_future_close_price"),
        pl.col("source_identity_sha256")
        .cast(pl.String)
        .alias("db_source_identity_sha256"),
    ).sort(_KEY)
    if result.select(_KEY).n_unique() != result.height:
        raise ValueError("DB crosscheck facts are not key-unique")
    return result


def build_dynamic_expiry_close_facts(
    population: pl.DataFrame,
    *,
    spot_root: Path,
    futures_root: Path,
    db_crosscheck_facts: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve every dynamic expiry pair or retain it as explicitly unresolved."""

    _require_columns(
        population,
        set(_KEY) | {"source_fill_count", "first_entry_date", "last_entry_date"},
        "dynamic expiry population",
    )
    if population.select(_KEY).n_unique() != population.height:
        raise ValueError("dynamic expiry population is not key-unique")
    dates = sorted(str(value) for value in population["Date"].unique().to_list())
    inventory = _source_inventory(dates, Path(spot_root), Path(futures_root))
    inventory_lookup = {str(row["Date"]): row for row in inventory.iter_rows(named=True)}
    db = (
        db_crosscheck_facts
        if db_crosscheck_facts is not None
        else load_db_crosscheck_facts(None)
    )
    if not db.is_empty():
        _require_columns(
            db,
            set(_KEY)
            | {
                "db_spot_close_price",
                "db_future_close_price",
                "db_source_identity_sha256",
            },
            "DB crosscheck facts",
        )
    db_lookup = {
        tuple(str(row[column]) for column in _KEY): row
        for row in db.iter_rows(named=True)
    }

    result_rows: list[dict[str, object]] = []
    for date in dates:
        selected = population.filter(pl.col("Date") == date)
        inv = inventory_lookup[date]
        spot = extract_spot_daily_closes(
            date,
            selected["ValueCode"].to_list(),
            Path(str(inv["spot_source_path"])),
        )
        future = extract_future_last_trades(
            date,
            selected["QuoteCode"].to_list(),
            Path(str(inv["future_source_path"])),
        )
        spot_lookup = {
            str(row["ValueCode"]): row for row in spot.iter_rows(named=True)
        }
        future_lookup = {
            str(row["QuoteCode"]): row for row in future.iter_rows(named=True)
        }
        for source in selected.iter_rows(named=True):
            row = dict(source)
            key = tuple(str(row[column]) for column in _KEY)
            spot_row = spot_lookup.get(str(row["ValueCode"]))
            future_row = future_lookup.get(str(row["QuoteCode"]))
            spot_close = _optional_float(
                spot_row.get("spot_close_price") if spot_row is not None else None
            )
            future_close = _optional_float(
                future_row.get("future_last_trade_price")
                if future_row is not None
                else None
            )
            spot_available = spot_close is not None and spot_close > 0
            future_available = future_close is not None and future_close > 0
            if spot_available and future_available:
                resolution_status = "paired_local_close_available"
            elif not bool(inv["spot_source_exists"]) and not bool(
                inv["future_source_exists"]
            ):
                resolution_status = "expiry_source_files_unavailable"
            elif not spot_available and not future_available:
                resolution_status = "spot_and_future_close_unresolved"
            elif not spot_available:
                resolution_status = "spot_close_unresolved"
            else:
                resolution_status = "future_last_trade_unresolved"

            db_row = db_lookup.get(key)
            db_spot = _optional_float(
                db_row.get("db_spot_close_price") if db_row is not None else None
            )
            db_future = _optional_float(
                db_row.get("db_future_close_price") if db_row is not None else None
            )
            spot_diff = (
                spot_close - db_spot
                if spot_close is not None and db_spot is not None
                else None
            )
            future_diff = (
                future_close - db_future
                if future_close is not None and db_future is not None
                else None
            )
            future_diff_bp = (
                future_diff / db_future * 10_000.0
                if future_diff is not None and db_future not in (None, 0.0)
                else None
            )
            raw_value_code = (
                str(future_row["raw_future_value_code"])
                if future_row is not None
                and future_row.get("raw_future_value_code") is not None
                else None
            )
            identity_payload = {
                "key": key,
                "spot_close_price": spot_close,
                "future_last_trade_price": future_close,
                "future_last_trade_time": (
                    future_row.get("future_last_trade_time")
                    if future_row is not None
                    else None
                ),
                "spot_source_path": inv["spot_source_path"],
                "spot_source_mtime_ns": inv["spot_source_mtime_ns"],
                "future_source_path": inv["future_source_path"],
                "future_source_mtime_ns": inv["future_source_mtime_ns"],
                "future_semantics": "last_positive_non_trial_day_session_trade",
            }
            result_rows.append(
                {
                    **row,
                    "spot_close_price": spot_close,
                    # Compatibility name for terminal-path consumers.  Its role is
                    # explicitly described by the flags and source columns below.
                    "future_close_price": future_close,
                    "future_last_trade_price": future_close,
                    "future_last_trade_time": (
                        future_row.get("future_last_trade_time")
                        if future_row is not None
                        else None
                    ),
                    "future_last_trade_recv_time_ns": (
                        future_row.get("future_last_trade_recv_time_ns")
                        if future_row is not None
                        else None
                    ),
                    "future_last_trade_channel_seq": (
                        future_row.get("future_last_trade_channel_seq")
                        if future_row is not None
                        else None
                    ),
                    "future_last_trade_packet_seq": (
                        future_row.get("future_last_trade_packet_seq")
                        if future_row is not None
                        else None
                    ),
                    "future_last_trade_fill_lots": (
                        future_row.get("future_last_trade_fill_lots")
                        if future_row is not None
                        else None
                    ),
                    "future_last_trade_decimal_locator": (
                        future_row.get("future_last_trade_decimal_locator")
                        if future_row is not None
                        else None
                    ),
                    "raw_future_value_code": raw_value_code,
                    "raw_future_value_code_matches": (
                        raw_value_code == str(row["ValueCode"])
                        if raw_value_code is not None
                        else None
                    ),
                    "spot_close_available": spot_available,
                    "future_last_trade_available": future_available,
                    "paired_close_available": spot_available and future_available,
                    "resolution_status": resolution_status,
                    "spot_source": "local_marketData.close_price",
                    "future_source": (
                        "local_stock_futures.last_positive_non_trial_day_session_trade"
                    ),
                    "spot_close_is_official_daily_close_field": True,
                    "future_close_is_official_daily_close": False,
                    "future_close_is_official_settlement": False,
                    "future_close_is_last_trade_proxy": True,
                    "paired_mark_is_fully_official_close": False,
                    "future_settlement_price_used": False,
                    "future_day_session_start": DAY_SESSION_START.isoformat(),
                    "future_day_session_end_exclusive": (
                        DAY_SESSION_END_EXCLUSIVE.isoformat()
                    ),
                    "source_identity_sha256": _canonical_sha256(identity_payload),
                    "db_crosscheck_available": db_row is not None,
                    "db_spot_close_price": db_spot,
                    "db_future_close_price": db_future,
                    "db_spot_minus_local_spot": (
                        -spot_diff if spot_diff is not None else None
                    ),
                    "local_future_minus_db_future": future_diff,
                    "local_future_minus_db_future_bp": future_diff_bp,
                    "db_spot_exact_match": (
                        math.isclose(spot_close, db_spot, abs_tol=1e-9, rel_tol=0.0)
                        if spot_close is not None and db_spot is not None
                        else None
                    ),
                    "db_future_exact_match": (
                        math.isclose(
                            future_close, db_future, abs_tol=1e-9, rel_tol=0.0
                        )
                        if future_close is not None and db_future is not None
                        else None
                    ),
                    "db_source_identity_sha256": (
                        db_row.get("db_source_identity_sha256")
                        if db_row is not None
                        else None
                    ),
                }
            )
    facts = pl.from_dicts(result_rows, infer_schema_length=None).sort(_KEY)
    if facts.height != population.height or facts.select(_KEY).n_unique() != facts.height:
        raise ValueError("expiry close facts changed the dynamic population")
    return facts, inventory


def coverage_summary(facts: pl.DataFrame) -> pl.DataFrame:
    summary = (
        facts.group_by("Date")
        .agg(
            pl.len().cast(pl.Int64).alias("population_pairs"),
            pl.col("source_fill_count").sum().alias("source_fills"),
            pl.col("spot_close_available").sum().cast(pl.Int64).alias("spot_closes"),
            pl.col("future_last_trade_available")
            .sum()
            .cast(pl.Int64)
            .alias("future_last_trades"),
            pl.col("paired_close_available")
            .sum()
            .cast(pl.Int64)
            .alias("paired_closes"),
            (~pl.col("paired_close_available"))
            .sum()
            .cast(pl.Int64)
            .alias("unresolved_pairs"),
            pl.col("db_crosscheck_available")
            .sum()
            .cast(pl.Int64)
            .alias("db_crosscheck_pairs"),
        )
        .sort("Date")
    )
    total = facts.select(
        pl.lit("ALL").alias("Date"),
        pl.len().cast(pl.Int64).alias("population_pairs"),
        pl.col("source_fill_count").sum().alias("source_fills"),
        pl.col("spot_close_available").sum().cast(pl.Int64).alias("spot_closes"),
        pl.col("future_last_trade_available")
        .sum()
        .cast(pl.Int64)
        .alias("future_last_trades"),
        pl.col("paired_close_available")
        .sum()
        .cast(pl.Int64)
        .alias("paired_closes"),
        (~pl.col("paired_close_available"))
        .sum()
        .cast(pl.Int64)
        .alias("unresolved_pairs"),
        pl.col("db_crosscheck_available")
        .sum()
        .cast(pl.Int64)
        .alias("db_crosscheck_pairs"),
    )
    return pl.concat([summary, total], how="vertical")


def crosscheck_summary(facts: pl.DataFrame) -> pl.DataFrame:
    overlap = facts.filter(pl.col("db_crosscheck_available"))
    schema = {
        "Date": pl.String,
        "pairs": pl.Int64,
        "spot_exact_matches": pl.Int64,
        "future_exact_matches": pl.Int64,
        "future_abs_diff_bp_mean": pl.Float64,
        "future_abs_diff_bp_median": pl.Float64,
        "future_abs_diff_bp_p95": pl.Float64,
        "future_abs_diff_bp_max": pl.Float64,
    }
    if overlap.is_empty():
        return pl.DataFrame(schema=schema)
    return (
        overlap.with_columns(
            pl.col("local_future_minus_db_future_bp")
            .abs()
            .alias("future_abs_diff_bp")
        )
        .group_by("Date")
        .agg(
            pl.len().cast(pl.Int64).alias("pairs"),
            pl.col("db_spot_exact_match")
            .sum()
            .cast(pl.Int64)
            .alias("spot_exact_matches"),
            pl.col("db_future_exact_match")
            .sum()
            .cast(pl.Int64)
            .alias("future_exact_matches"),
            pl.col("future_abs_diff_bp").mean().alias("future_abs_diff_bp_mean"),
            pl.col("future_abs_diff_bp")
            .median()
            .alias("future_abs_diff_bp_median"),
            pl.col("future_abs_diff_bp")
            .quantile(0.95, interpolation="linear")
            .alias("future_abs_diff_bp_p95"),
            pl.col("future_abs_diff_bp").max().alias("future_abs_diff_bp_max"),
        )
        .sort("Date")
    )


def _artifact(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        result.update(
            {
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )
    elif path.suffix == ".csv":
        result["rows"] = pl.read_csv(path).height
    return result


def _render_readme(
    facts: pl.DataFrame,
    coverage: pl.DataFrame,
    crosscheck: pl.DataFrame,
) -> str:
    total = coverage.filter(pl.col("Date") == "ALL").row(0, named=True)
    lines = [
        "# Dynamic expiry paired-close facts",
        "",
        "This bundle derives its population directly from all canonical dynamic makerFill fills. No fixed-45 list or count cap is applied.",
        "",
        f"- Population: {total['population_pairs']} unique expiry product-contract pairs from {total['source_fills']} approximate fills.",
        f"- Paired local marks: {total['paired_closes']}; unresolved: {total['unresolved_pairs']}.",
        "- Spot price is the local `marketData.close_price` daily-close field.",
        "- Futures price is the last positive non-`TrialMatch` day-session trade in local raw ticks. It is a proxy, not an official daily close, settlement, or executable close-auction guarantee.",
        "- `future_close_price` is retained as a consumer-compatible alias of `future_last_trade_price`; its proxy flags must travel with downstream results.",
        "",
        "| Expiry | Dynamic pairs | Source fills | Paired | Unresolved | DB overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.filter(pl.col("Date") != "ALL").iter_rows(named=True):
        lines.append(
            f"| {row['Date']} | {row['population_pairs']} | {row['source_fills']} | "
            f"{row['paired_closes']} | {row['unresolved_pairs']} | "
            f"{row['db_crosscheck_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Official-DB overlap validation",
            "",
            "The older DB extract is used only as a key-by-key validation source. It never defines or filters this population.",
            "",
            "| Expiry | Pairs | Spot exact | Futures exact | Futures abs diff median / p95 / max (bp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in crosscheck.iter_rows(named=True):
        lines.append(
            f"| {row['Date']} | {row['pairs']} | {row['spot_exact_matches']} | "
            f"{row['future_exact_matches']} | "
            f"{row['future_abs_diff_bp_median']:.4f} / "
            f"{row['future_abs_diff_bp_p95']:.4f} / "
            f"{row['future_abs_diff_bp_max']:.4f} |"
        )
    unresolved = facts.filter(~pl.col("paired_close_available"))
    if unresolved.height:
        lines.extend(["", "## Unresolved", ""])
        for row in unresolved.iter_rows(named=True):
            lines.append(
                f"- {row['Date']} {row['ValueCode']}/{row['QuoteCode']}: "
                f"{row['resolution_status']} ({row['source_fill_count']} source fills)."
            )
    return "\n".join(lines) + "\n"


def publish(
    output: Path,
    *,
    candidate_root: Path,
    candidate_marker: Mapping[str, object],
    population: pl.DataFrame,
    facts: pl.DataFrame,
    inventory: pl.DataFrame,
    db_crosscheck_path: Path | None,
) -> None:
    """Atomically publish a source-bound dynamic expiry fact bundle."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        coverage = coverage_summary(facts)
        crosscheck = crosscheck_summary(facts)
        population.write_parquet(stage / "dynamic_expiry_population.parquet")
        facts.write_parquet(stage / "dynamic_expiry_close_facts.parquet")
        facts.filter(pl.col("paired_close_available")).write_parquet(
            stage / "paired_close_facts.parquet"
        )
        facts.filter(~pl.col("paired_close_available")).write_parquet(
            stage / "unresolved_close_facts.parquet"
        )
        inventory.write_parquet(stage / "source_inventory.parquet")
        coverage.write_csv(stage / "coverage_summary.csv")
        crosscheck.write_csv(stage / "db_crosscheck_summary.csv")
        (stage / "README.md").write_text(
            _render_readme(facts, coverage, crosscheck), encoding="utf-8"
        )
        artifact_inventory = {
            path.name: _artifact(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        resolved = facts.filter(pl.col("paired_close_available"))
        overlap = facts.filter(pl.col("db_crosscheck_available"))
        marker = {
            "complete": True,
            "schema_version": VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_only": True,
            "dynamic_population": True,
            "fixed45_filter_applied": False,
            "population_definition": (
                "unique(end_date, ValueCode, QuoteCode) among canonical full_fill rows"
            ),
            "population_pair_count": facts.height,
            "source_fill_count": int(facts["source_fill_count"].sum()),
            "paired_local_mark_count": resolved.height,
            "unresolved_pair_count": facts.height - resolved.height,
            "fully_official_paired_close_count": 0,
            "spot_close_is_official_daily_close_field": True,
            "future_mark_role": "last_positive_non_trial_day_session_trade_proxy",
            "future_close_is_official_daily_close": False,
            "future_close_is_official_settlement": False,
            "future_settlement_price_used": False,
            "db_crosscheck_pair_count": overlap.height,
            "db_crosscheck_does_not_define_population": True,
            "candidate_root": str(Path(candidate_root).resolve()),
            "candidate_complete_sha256": _sha256(
                Path(candidate_root) / "complete.json"
            ),
            "candidate_runner_version": candidate_marker.get("runner_version"),
            "db_crosscheck_path": (
                str(Path(db_crosscheck_path).resolve())
                if db_crosscheck_path is not None
                and Path(db_crosscheck_path).is_file()
                else None
            ),
            "db_crosscheck_sha256": (
                _sha256(Path(db_crosscheck_path))
                if db_crosscheck_path is not None
                and Path(db_crosscheck_path).is_file()
                else None
            ),
            "artifacts": artifact_inventory,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run(
    candidate_root: Path,
    spot_root: Path,
    futures_root: Path,
    db_crosscheck_path: Path | None,
    output: Path,
) -> pl.DataFrame:
    population, candidate_marker = load_dynamic_expiry_population(candidate_root)
    db = load_db_crosscheck_facts(db_crosscheck_path)
    facts, inventory = build_dynamic_expiry_close_facts(
        population,
        spot_root=spot_root,
        futures_root=futures_root,
        db_crosscheck_facts=db,
    )
    publish(
        output,
        candidate_root=candidate_root,
        candidate_marker=candidate_marker,
        population=population,
        facts=facts,
        inventory=inventory,
        db_crosscheck_path=db_crosscheck_path,
    )
    return facts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--spot-root", type=Path, default=DEFAULT_SPOT_DAILY_ROOT)
    parser.add_argument("--futures-root", type=Path, default=DEFAULT_FUTURES_RAW_ROOT)
    parser.add_argument(
        "--db-crosscheck-path", type=Path, default=DEFAULT_DB_CROSSCHECK_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    facts = run(
        args.candidate_root,
        args.spot_root,
        args.futures_root,
        args.db_crosscheck_path,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "population_pairs": facts.height,
                "source_fills": int(facts["source_fill_count"].sum()),
                "paired_local_marks": int(facts["paired_close_available"].sum()),
                "unresolved_pairs": int((~facts["paired_close_available"]).sum()),
                "fixed45_filter_applied": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
