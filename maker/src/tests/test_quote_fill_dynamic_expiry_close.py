from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import polars as pl

from maker.src.quote_fill.dynamic_expiry_close import (
    build_dynamic_expiry_close_facts,
    derive_dynamic_expiry_population,
    extract_future_last_trades,
    run,
)


def _candidate_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "end_date": [
                date(2026, 5, 20),
                date(2026, 5, 20),
                date(2026, 5, 20),
                date(2026, 9, 16),
            ],
            "Date": ["20260504", "20260505", "20260506", "20260813"],
            "ValueCode": ["9999", "9999", "1111", "6443"],
            "QuoteCode": ["ZZFE6", "ZZFE6", "AAFE6", "RLFI6"],
            "full_fill": [True, True, False, True],
        }
    )


def _future_raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "RecvTime": [
                datetime(2026, 5, 20, 5, 28, tzinfo=timezone.utc),
                datetime(2026, 5, 20, 5, 29, tzinfo=timezone.utc),
                datetime(2026, 5, 20, 5, 29, 1, tzinfo=timezone.utc),
                datetime(2026, 5, 20, 7, 0, tzinfo=timezone.utc),
            ],
            "TransTime": [
                datetime(2026, 5, 20, 13, 28),
                datetime(2026, 5, 20, 13, 29),
                datetime(2026, 5, 20, 13, 29, 1),
                datetime(2026, 5, 20, 15, 0),
            ],
            "QuoteCode": ["ZZFE6"] * 4,
            "ValueCode": ["9999"] * 4,
            "PacketSeq": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "ChannelSeq": pl.Series([11, 12, 13, 14], dtype=pl.UInt64),
            "TrialMatch": [0, 0, 1, 0],
            "DecimalLocator": [2, 2, 2, 2],
            "FillPrice": [10_000, 10_100, 99_900, 10_200],
            "FillLots": [1, 2, 5, 3],
        }
    )


def _write_sources(spot_root: Path, futures_root: Path) -> None:
    spot_root.mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [date(2026, 5, 20)],
            "quote_code": ["9999"],
            "close_price": [45.0],
        }
    ).write_parquet(spot_root / "20260520_marketData.parquet")
    future_path = futures_root / "2026" / "05" / "20" / "stock_futures.parquet"
    future_path.parent.mkdir(parents=True)
    _future_raw().write_parquet(future_path)


def test_population_comes_from_all_fills_without_fixed_list() -> None:
    result = derive_dynamic_expiry_population(_candidate_rows())

    assert result.height == 2
    assert result["source_fill_count"].sum() == 3
    assert result.filter(pl.col("ValueCode") == "9999").item(
        0, "source_fill_count"
    ) == 2
    assert set(result["ValueCode"].to_list()) == {"9999", "6443"}


def test_future_proxy_uses_last_non_trial_day_trade(tmp_path: Path) -> None:
    path = tmp_path / "stock_futures.parquet"
    _future_raw().write_parquet(path)

    result = extract_future_last_trades("20260520", ["ZZFE6"], path)

    assert result.height == 1
    assert result.item(0, "future_last_trade_price") == 101.0
    assert result.item(0, "future_last_trade_fill_lots") == 2
    assert result.item(0, "future_last_trade_channel_seq") == 12


def test_builder_retains_missing_future_expiry_as_unresolved(tmp_path: Path) -> None:
    population = derive_dynamic_expiry_population(_candidate_rows())
    spot_root = tmp_path / "spot"
    futures_root = tmp_path / "future"
    _write_sources(spot_root, futures_root)
    db = pl.DataFrame(
        {
            "Date": ["20260520"],
            "ValueCode": ["9999"],
            "QuoteCode": ["ZZFE6"],
            "db_spot_close_price": [45.0],
            "db_future_close_price": [100.5],
            "db_source_identity_sha256": ["a" * 64],
        }
    )

    facts, inventory = build_dynamic_expiry_close_facts(
        population,
        spot_root=spot_root,
        futures_root=futures_root,
        db_crosscheck_facts=db,
    )

    assert facts.height == population.height
    resolved = facts.filter(pl.col("Date") == "20260520").row(0, named=True)
    assert resolved["paired_close_available"] is True
    assert resolved["future_close_price"] == 101.0
    assert resolved["future_close_is_official_daily_close"] is False
    assert resolved["db_spot_exact_match"] is True
    assert resolved["db_future_exact_match"] is False
    missing = facts.filter(pl.col("Date") == "20260916").row(0, named=True)
    assert missing["paired_close_available"] is False
    assert missing["resolution_status"] == "expiry_source_files_unavailable"
    assert inventory.height == 2


def test_runner_publishes_atomic_dynamic_bundle(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    partition = candidate_root / "candidate_outcomes" / "Date=20260504"
    partition.mkdir(parents=True)
    candidates = _candidate_rows().filter(pl.col("full_fill"))
    candidates.write_parquet(partition / "candidate_outcomes.parquet")
    (candidate_root / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "runner_version": "test",
                "approximate_fills": candidates.height,
                "artifacts": {"candidate_partitions": 1},
            }
        ),
        encoding="utf-8",
    )
    spot_root = tmp_path / "spot"
    futures_root = tmp_path / "future"
    _write_sources(spot_root, futures_root)
    db_path = tmp_path / "db.parquet"
    pl.DataFrame(
        {
            "Date": ["20260520"],
            "ValueCode": ["9999"],
            "QuoteCode": ["ZZFE6"],
            "spot_close_price": [45.0],
            "future_close_price": [101.0],
            "source_identity_sha256": ["b" * 64],
        }
    ).write_parquet(db_path)
    output = tmp_path / "output"

    result = run(candidate_root, spot_root, futures_root, db_path, output)

    assert result.height == 2
    marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    assert marker["complete"] is True
    assert marker["dynamic_population"] is True
    assert marker["fixed45_filter_applied"] is False
    assert marker["population_pair_count"] == 2
    assert marker["paired_local_mark_count"] == 1
    assert marker["unresolved_pair_count"] == 1
    assert (output / "dynamic_expiry_close_facts.parquet").is_file()
    assert (output / "unresolved_close_facts.parquet").is_file()
