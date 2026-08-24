from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from maker.src.quote_fill.august_exit_extension import (
    REQUIRED_GRID_COLUMNS,
    assemble_one_second_grid,
    compact_future_seconds,
    compact_spot_seconds,
    load_canonical_filled_population,
    load_exact_session_mapping,
    publish_extension_partition,
    refresh_stale_future_references,
    validate_extension_partition,
)

DAY = "20260814"


def _local(value: str) -> datetime:
    return datetime.fromisoformat(f"2026-08-14T{value}")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(f"2026-08-14T{value}").replace(tzinfo=UTC)


def _book_fields(
    *,
    bid: float = 0,
    ask: float = 0,
    bid_lots: int = 0,
    ask_lots: int = 0,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for level in range(1, 6):
        fields[f"BidPrice{level}"] = bid if level == 1 else 0
        fields[f"AskPrice{level}"] = ask if level == 1 else 0
        fields[f"BidLots{level}"] = bid_lots if level == 1 else 0
        fields[f"AskLots{level}"] = ask_lots if level == 1 else 0
    return fields


def _spot_raw() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    base = {
        "ValueCode": "2330",
        "QuoteCode": "2330",
        "PacketSeq": 1,
        "TrialMatch": 0,
    }
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:00:00.000000").replace(tzinfo=None),
            "TransTime": _local("09:00:00.000000"),
            "ChannelSeq": 1,
            **_book_fields(),
        }
    )
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:04:59.100000").replace(tzinfo=None),
            "TransTime": _local("09:04:59.100000"),
            "ChannelSeq": 2,
            **_book_fields(bid=100.0, ask=101.0, bid_lots=2, ask_lots=3),
        }
    )
    # A zero-book trade/event must not erase the prior L1 state.
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:04:59.500000").replace(tzinfo=None),
            "TransTime": _local("09:04:59.500000"),
            "ChannelSeq": 3,
            **_book_fields(),
        }
    )
    # Arrives after the 01:05:00 decision boundary and becomes visible at
    # 01:05:01, not one second early.
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:05:00.200000").replace(tzinfo=None),
            "TransTime": _local("09:05:00.200000"),
            "ChannelSeq": 4,
            **_book_fields(bid=99.0, ask=100.0, bid_lots=4, ask_lots=5),
        }
    )
    return pl.DataFrame(rows)


def _future_raw() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    base = {
        "ValueCode": "2330",
        "QuoteCode": "CDFH6",
        "PacketSeq": 1,
        "TrialMatch": 0,
        "DecimalLocator": 2,
        "BestBidPrice": 0,
        "BestAskPrice": 0,
        "BestBidLots": 0,
        "BestAskLots": 0,
    }
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:00:00.000000"),
            "TransTime": _local("09:00:00.000000"),
            "ChannelSeq": 1,
            **_book_fields(),
        }
    )
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:04:59.100000"),
            "TransTime": _local("09:04:59.100000"),
            "ChannelSeq": 2,
            **_book_fields(bid=10100, ask=10200, bid_lots=2, ask_lots=2),
        }
    )
    # Best is an independent component.  At the same ask price it has more
    # lots, so executable lots must be max(2, 5), not their sum.
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:04:59.300000"),
            "TransTime": _local("09:04:59.300000"),
            "ChannelSeq": 3,
            "BestBidPrice": 10100,
            "BestAskPrice": 10200,
            "BestBidLots": 3,
            "BestAskLots": 5,
            **_book_fields(),
        }
    )
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:04:59.500000"),
            "TransTime": _local("09:04:59.500000"),
            "ChannelSeq": 4,
            **_book_fields(),
        }
    )
    # L1 updates after the boundary while Best carries forward.  The later
    # executable ask remains 102 from Best, proving L1 did not clear it.
    rows.append(
        {
            **base,
            "RecvTime": _utc("01:05:00.200000"),
            "TransTime": _local("09:05:00.200000"),
            "ChannelSeq": 5,
            **_book_fields(bid=10200, ask=10400, bid_lots=4, ask_lots=4),
        }
    )
    return pl.DataFrame(rows)


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2330"],
            "QuoteCode": ["CDFH6"],
            "contract_size": [2000.0],
            "decimal_locator": [2],
            "end_date": [date(2026, 8, 19)],
            "spot_ref_price": [100.0],
            "fut_ref_price": [101.0],
            "day_trade_mark": ["X"],
            "contract_metadata_date": [DAY],
            "fut_ref_same_day_metadata": [True],
            "fut_ref_source": ["same_day_exact_contract_metadata"],
            "fut_ref_available_ns": [
                int(_utc("01:05:00").timestamp() * 1_000_000_000)
            ],
        },
        schema_overrides={"decimal_locator": pl.Int16},
    )


class AugustExitExtensionTest(unittest.TestCase):
    def test_independent_future_components_max_lots_and_causal_boundary(self) -> None:
        spot = compact_spot_seconds(_spot_raw(), ["2330"], DAY)
        future = compact_future_seconds(_future_raw(), ["CDFH6"], DAY)
        timestamps = pl.Series(
            "timestamp",
            [
                _utc("01:05:00").replace(tzinfo=None),
                _utc("01:05:01").replace(tzinfo=None),
            ],
            dtype=pl.Datetime("ns"),
        )
        result = assemble_one_second_grid(
            DAY, _mapping(), spot, future, timestamps=timestamps
        )
        first, second = result.rows(named=True)
        self.assertEqual(first["spot_bid"], 100.0)
        self.assertEqual(first["spot_bid_lots"], 2)
        self.assertEqual(first["fut_exec_ask"], 102.0)
        self.assertEqual(first["fut_exec_ask_lots"], 5)
        self.assertTrue(first["analysis_eligible"])
        self.assertAlmostEqual(first["basis_buy_taker_bp"], 200.0)
        self.assertEqual(second["spot_bid"], 99.0)
        self.assertEqual(second["fut_ask"], 104.0)
        self.assertEqual(second["fut_exec_ask"], 102.0)
        self.assertEqual(second["fut_exec_ask_lots"], 5)

    def test_population_is_filled_exact_and_expiry_filtered_without_count_cap(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition = root / "candidate_outcomes" / "Date=20260813"
            partition.mkdir(parents=True)
            pl.DataFrame(
                {
                    "Date": ["20260813", "20260813", "20260813"],
                    "ValueCode": ["1101", "2330", "2603"],
                    "QuoteCode": ["DFFH6", "CDFH6", "CZFJ6"],
                    "contract_size": [2000.0, 2000.0, 2000.0],
                    "end_date": [
                        date(2026, 8, 19),
                        date(2026, 8, 19),
                        date(2026, 10, 21),
                    ],
                    "approximate_fill_before_nominal_stop": [True, False, True],
                    "full_fill": [True, False, True],
                    "outcome_supported": [True, True, True],
                }
            ).write_parquet(partition / "candidate_outcomes.parquet")
            population = load_canonical_filled_population(DAY, entry_root=root)
            self.assertEqual(population["ValueCode"].to_list(), ["1101", "2603"])

    def test_exact_mapping_keeps_non_daytrade_carry(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            market = root / "market"
            metadata.mkdir()
            market.mkdir()
            population = pl.DataFrame(
                {
                    "ValueCode": ["2330"],
                    "QuoteCode": ["CDFH6"],
                    "contract_size": [2000.0],
                    "end_date": [date(2026, 8, 19)],
                }
            )
            pl.DataFrame(
                {
                    "ValueCode": ["2330"],
                    "QuoteCode": ["CDFH6"],
                    "contract_size": [2000.0],
                    "decimal_locator": [2],
                    "end_date": [date(2026, 8, 19)],
                    "fut_ref_price": [101.0],
                },
                schema_overrides={"decimal_locator": pl.Int16},
            ).write_parquet(metadata / f"{DAY}_contracts.parquet")
            pl.DataFrame(
                {
                    "quote_code": ["2330"],
                    "opening_ref_price": [100.0],
                    "allow_day_trade_mark": ["N"],
                }
            ).write_parquet(market / f"{DAY}_marketData.parquet")
            result = load_exact_session_mapping(
                DAY,
                population,
                metadata_root=metadata,
                market_data_root=market,
            )
            self.assertEqual(result.height, 1)
            self.assertEqual(result.item(0, "day_trade_mark"), "N")

    def test_missing_same_day_metadata_uses_causal_l1_reference_proxy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            market = root / "market"
            metadata.mkdir()
            market.mkdir()
            population = pl.DataFrame(
                {
                    "ValueCode": ["2330"],
                    "QuoteCode": ["CDFH6"],
                    "contract_size": [2000.0],
                    "end_date": [date(2026, 8, 19)],
                }
            )
            pl.DataFrame(
                {
                    "ValueCode": ["2330"],
                    "QuoteCode": ["CDFH6"],
                    "contract_size": [2000.0],
                    "decimal_locator": [2],
                    "end_date": [date(2026, 8, 19)],
                    "fut_ref_price": [999.0],
                },
                schema_overrides={"decimal_locator": pl.Int16},
            ).write_parquet(metadata / "20260813_contracts.parquet")
            pl.DataFrame(
                {
                    "quote_code": ["2330"],
                    "opening_ref_price": [100.0],
                    "allow_day_trade_mark": ["X"],
                }
            ).write_parquet(market / f"{DAY}_marketData.parquet")
            mapping = load_exact_session_mapping(
                DAY,
                population,
                metadata_root=metadata,
                market_data_root=market,
            )
            self.assertFalse(mapping.item(0, "fut_ref_same_day_metadata"))
            refreshed = refresh_stale_future_references(
                DAY,
                mapping,
                compact_future_seconds(_future_raw(), ["CDFH6"], DAY),
            )
            self.assertEqual(refreshed.item(0, "fut_ref_price"), 101.5)
            self.assertEqual(
                refreshed.item(0, "fut_ref_source"),
                "first_causal_formal_two_sided_l1_midpoint_proxy",
            )

    def test_partition_marker_is_published_and_validated(self) -> None:
        spot = compact_spot_seconds(_spot_raw(), ["2330"], DAY)
        future = compact_future_seconds(_future_raw(), ["CDFH6"], DAY)
        grid = assemble_one_second_grid(
            DAY,
            _mapping(),
            spot,
            future,
            timestamps=pl.Series(
                [_utc("01:05:00").replace(tzinfo=None)],
                dtype=pl.Datetime("ns"),
            ),
        )
        audit = pl.DataFrame({"Date": [DAY], "rows": [1]})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = publish_extension_partition(
                DAY,
                _mapping(),
                grid,
                audit,
                output_root=root,
                config=__import__(
                    "maker.src.quote_fill.august_exit_extension",
                    fromlist=["DEFAULT_CONFIG"],
                ).DEFAULT_CONFIG,
                source_paths=[],
                elapsed_seconds=0.1,
            )
            marker = root / f"Date={DAY}" / "complete.json"
            payload = validate_extension_partition(marker)
            self.assertEqual(result.rows, 1)
            self.assertFalse(payload["fixed45_universe_used"])
            self.assertEqual(
                payload["required_vertical_concat_columns"],
                list(REQUIRED_GRID_COLUMNS),
            )
            marker_payload = json.loads(marker.read_text())
            self.assertEqual(marker_payload["rows"], 1)


if __name__ == "__main__":
    unittest.main()
