"""Point-in-time spot and stock-futures contract mapping."""

from __future__ import annotations

import sys
from datetime import date as date_type
from pathlib import Path

import polars as pl

from .paths import DEFAULT_OUTPUT_ROOT, RESEARCH_ROOT, market_data_path, parse_date


STANDARD_CONTRACT_SIZE = 2000.0
CONTRACT_SIZE_EPS = 1e-6


def _load_futures_basic_from_existing_loader(date: str) -> pl.DataFrame:
    """Reuse the existing point-in-time product loader without copying credentials."""
    taker_dir = RESEARCH_ROOT / "taker"
    taker_path = str(taker_dir)
    if taker_path not in sys.path:
        sys.path.insert(0, taker_path)
    import arbitrage_analysis as taker_analysis

    return taker_analysis._load_futures_basic(date)


def _normalise_basic_schema(basic: pl.DataFrame) -> pl.DataFrame:
    rename = {
        "quote_code": "QuoteCode",
        "value_code": "ValueCode",
        "ref_price": "fut_ref_price",
    }
    available = {source: target for source, target in rename.items() if source in basic.columns}
    basic = basic.rename(available)
    required = {
        "QuoteCode",
        "ValueCode",
        "fut_ref_price",
        "contract_size",
        "decimal_locator",
        "end_date",
    }
    missing = sorted(required - set(basic.columns))
    if missing:
        raise ValueError(f"futures basic info missing columns: {missing}")
    return basic.select(sorted(required)).with_columns(
        pl.col("QuoteCode").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("fut_ref_price").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("decimal_locator").cast(pl.Int16),
        pl.col("end_date").cast(pl.Date),
    )


def select_near_standard_contracts(
    basic: pl.DataFrame,
    trade_date: date_type,
    value_codes: list[str] | None = None,
) -> pl.DataFrame:
    """Select one nearest unexpired standard contract for each spot symbol."""
    basic = _normalise_basic_schema(basic)
    eligible = basic.filter(
        (pl.col("QuoteCode").str.slice(2, 1) == "F")
        & ((pl.col("contract_size") - STANDARD_CONTRACT_SIZE).abs() < CONTRACT_SIZE_EPS)
        & (pl.col("end_date") >= pl.lit(trade_date))
        & (pl.col("fut_ref_price") > 0)
    )
    if value_codes is not None:
        eligible = eligible.filter(pl.col("ValueCode").is_in(value_codes))

    nearest = eligible.group_by("ValueCode").agg(
        pl.col("end_date").min().alias("end_date")
    )
    selected = eligible.join(nearest, on=["ValueCode", "end_date"], how="inner")
    duplicates = selected.group_by("ValueCode").len().filter(pl.col("len") != 1)
    if duplicates.height:
        examples = duplicates.head(10).to_dicts()
        raise ValueError(f"near-contract mapping is not one-to-one: {examples}")
    return selected.sort(["ValueCode", "QuoteCode"])


def load_spot_reference(date: str) -> pl.DataFrame:
    path = market_data_path(date)
    if not path.exists():
        raise FileNotFoundError(path)
    source = pl.scan_parquet(path)
    columns = set(source.collect_schema().names())
    turnover = (
        pl.col("trading_turnover").cast(pl.Float64)
        if "trading_turnover" in columns
        else pl.lit(None, dtype=pl.Float64).alias("trading_turnover")
    )
    spot = (
        source
        .select(
            pl.col("quote_code").cast(pl.String).alias("ValueCode"),
            pl.col("opening_ref_price").cast(pl.Float64).alias("spot_ref_price"),
            pl.col("allow_day_trade_mark").cast(pl.String).alias("day_trade_mark"),
            turnover,
            pl.col("ins_type").cast(pl.String),
        )
        .filter(
            (pl.col("day_trade_mark").str.to_uppercase() == "X")
            & pl.col("spot_ref_price").is_not_null()
            & (pl.col("spot_ref_price") > 0)
        )
        .collect()
    )
    conflicts = (
        spot.group_by("ValueCode")
        .agg(
            pl.col("spot_ref_price").n_unique().alias("ref_values"),
            pl.col("day_trade_mark").n_unique().alias("day_trade_values"),
        )
        .filter((pl.col("ref_values") != 1) | (pl.col("day_trade_values") != 1))
    )
    if conflicts.height:
        raise ValueError(
            f"{date}: conflicting spot reference rows: {conflicts.head(10).to_dicts()}"
        )
    return spot.unique(subset=["ValueCode"], keep="first").sort("ValueCode")


def load_contract_mapping(
    date: str,
    value_codes: list[str] | None = None,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Load and cache the daily one-to-one spot/futures mapping."""
    cache_dir = cache_dir or (DEFAULT_OUTPUT_ROOT / "metadata")
    cache_path = cache_dir / f"{date}_contracts.parquet"
    if cache_path.exists():
        contracts = pl.read_parquet(cache_path)
    else:
        basic = _load_futures_basic_from_existing_loader(date)
        contracts = select_near_standard_contracts(basic, parse_date(date).date())
        cache_dir.mkdir(parents=True, exist_ok=True)
        contracts.write_parquet(cache_path)

    spot = load_spot_reference(date)
    mapping = contracts.join(spot, on="ValueCode", how="inner")
    if value_codes is not None:
        mapping = mapping.filter(pl.col("ValueCode").is_in(value_codes))
    if mapping.height == 0:
        raise RuntimeError(f"{date}: no mapped day-tradable standard stock futures")
    quote_duplicates = mapping.group_by("QuoteCode").len().filter(pl.col("len") != 1)
    if quote_duplicates.height:
        raise ValueError(
            f"{date}: QuoteCode mapping is not one-to-one: "
            f"{quote_duplicates.head(10).to_dicts()}"
        )
    return mapping.sort(["ValueCode", "QuoteCode"])


def load_exact_contract_mapping(
    date: str,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Load prior-day metadata for exact spot/futures pairs chosen on another day."""
    required = {"ValueCode", "QuoteCode"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"exact contract targets missing columns: {missing}")
    pairs = targets.select("ValueCode", "QuoteCode").unique()
    pair_duplicates = pairs.group_by("ValueCode").len().filter(pl.col("len") != 1)
    if pair_duplicates.height:
        raise ValueError(
            f"exact targets are not one-to-one: {pair_duplicates.to_dicts()}"
        )

    basic = _normalise_basic_schema(_load_futures_basic_from_existing_loader(date))
    exact = basic.join(pairs, on=["ValueCode", "QuoteCode"], how="inner")
    missing_pairs = pairs.join(
        exact.select("ValueCode", "QuoteCode"),
        on=["ValueCode", "QuoteCode"],
        how="anti",
    )
    if missing_pairs.height:
        raise ValueError(
            f"{date}: exact futures contracts unavailable: {missing_pairs.to_dicts()}"
        )

    mapping = exact.join(load_spot_reference(date), on="ValueCode", how="inner")
    if mapping.height != pairs.height:
        mapped_pairs = mapping.select("ValueCode", "QuoteCode")
        unavailable = pairs.join(
            mapped_pairs,
            on=["ValueCode", "QuoteCode"],
            how="anti",
        )
        raise ValueError(
            f"{date}: exact pairs lack tradable spot metadata: {unavailable.to_dicts()}"
        )
    return mapping.sort(["ValueCode", "QuoteCode"])
