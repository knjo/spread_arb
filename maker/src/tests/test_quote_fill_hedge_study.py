from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import polars as pl

from maker.src.quote_fill.hedge_study import (
    FUTURE_ASK_ROUTE,
    SPOT_BID_ROUTE,
    executable_levels_from_state,
    run_hedge_study,
)
from maker.src.quote_fill.raw_tape import RawTapeDay


MS = 1_000_000
DATE = "20260128"


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2000.0],
        }
    )


def _state(
    market: str,
    ms: int,
    sequence: int,
    *,
    bid_prices: tuple[float | None, ...],
    bid_lots: tuple[int | None, ...],
    ask_prices: tuple[float | None, ...],
    ask_lots: tuple[int | None, ...],
    best_bid: float | None = None,
    best_bid_lots: int | None = None,
    best_ask: float | None = None,
    best_ask_lots: int | None = None,
    trial_match: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": DATE,
        "market": market,
        "ValueCode": "2317",
        "QuoteCode": "DHFB6",
        "recv_time_ns": ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": trial_match,
        "ref_price": 100.0,
        "book_state_available": True,
        "book_recv_time_ns": ms * MS,
        "best_bid_price": best_bid,
        "best_bid_lots": best_bid_lots,
        "best_ask_price": best_ask,
        "best_ask_lots": best_ask_lots,
    }
    for level in range(1, 6):
        row[f"bid_price_{level}"] = bid_prices[level - 1]
        row[f"bid_lots_{level}"] = bid_lots[level - 1]
        row[f"ask_price_{level}"] = ask_prices[level - 1]
        row[f"ask_lots_{level}"] = ask_lots[level - 1]
    return row


def _raw_tape() -> RawTapeDay:
    spot = pl.from_dicts(
        [
            _state(
                "spot",
                90,
                9,
                bid_prices=(99.0, 98.0, None, None, None),
                bid_lots=(5, 5, None, None, None),
                ask_prices=(100.0, 101.0, None, None, None),
                ask_lots=(5, 5, None, None, None),
            ),
            # Same receive timestamp, but spot priority is after the futures
            # maker fill and must not become its arrival reference.
            _state(
                "spot",
                100,
                10,
                bid_prices=(89.0, 88.0, None, None, None),
                bid_lots=(5, 5, None, None, None),
                ask_prices=(90.0, 91.0, None, None, None),
                ask_lots=(5, 5, None, None, None),
            ),
            _state(
                "spot",
                150,
                15,
                bid_prices=(100.0, 99.0, None, None, None),
                bid_lots=(5, 5, None, None, None),
                ask_prices=(101.0, 102.0, None, None, None),
                ask_lots=(1, 2, None, None, None),
            ),
        ],
        infer_schema_length=None,
    )
    future = pl.from_dicts(
        [
            _state(
                "future",
                250,
                20,
                bid_prices=(99.0, 98.0, None, None, None),
                bid_lots=(2, 2, None, None, None),
                ask_prices=(101.0, 102.0, None, None, None),
                ask_lots=(2, 2, None, None, None),
                best_bid=99.0,
                best_bid_lots=3,
                best_ask=101.0,
                best_ask_lots=3,
            ),
            # Futures priority is before a spot fill at the same timestamp, so
            # the inside Best bid at 100 is observable at arrival.
            _state(
                "future",
                300,
                25,
                bid_prices=(99.0, 98.0, None, None, None),
                bid_lots=(2, 2, None, None, None),
                ask_prices=(102.0, 103.0, None, None, None),
                ask_lots=(2, 2, None, None, None),
                best_bid=100.0,
                best_bid_lots=1,
                best_ask=101.0,
                best_ask_lots=2,
            ),
            _state(
                "future",
                350,
                35,
                bid_prices=(98.0, 97.0, None, None, None),
                bid_lots=(4, 4, None, None, None),
                ask_prices=(101.0, 102.0, None, None, None),
                ask_lots=(4, 4, None, None, None),
                best_bid=99.0,
                best_bid_lots=1,
                best_ask=100.0,
                best_ask_lots=2,
            ),
        ],
        infer_schema_length=None,
    )
    empty = pl.DataFrame()
    return RawTapeDay(DATE, _mapping(), spot, future, empty, empty, empty)


def _aliases() -> pl.DataFrame:
    common = {
        "Date": DATE,
        "ValueCode": "2317",
        "QuoteCode": "DHFB6",
        "full_fill_recv_time_ns": None,
        "full_fill_event_sequence": None,
        "full_fill_row_index": None,
    }
    rows = [
        {
            **common,
            "raw_order_fact_id": "future-fill",
            "route": FUTURE_ASK_ROUTE,
            "maker_market": "future",
            "intended_quantity": 1,
            "boundary_quantile": 50,
            "full_fill": True,
            "full_fill_recv_time_ns": 100 * MS,
            "full_fill_event_sequence": 1,
            "full_fill_row_index": 10,
        },
        {
            **common,
            "raw_order_fact_id": "future-fill",
            "route": FUTURE_ASK_ROUTE,
            "maker_market": "future",
            "intended_quantity": 1,
            "boundary_quantile": 80,
            "full_fill": True,
            "full_fill_recv_time_ns": 100 * MS,
            "full_fill_event_sequence": 1,
            "full_fill_row_index": 10,
        },
        {
            **common,
            "raw_order_fact_id": "spot-fill",
            "route": SPOT_BID_ROUTE,
            "maker_market": "spot",
            "intended_quantity": 2,
            "boundary_quantile": 50,
            "full_fill": True,
            "full_fill_recv_time_ns": 300 * MS,
            "full_fill_event_sequence": 2,
            "full_fill_row_index": 30,
        },
        {
            **common,
            "raw_order_fact_id": "spot-partial",
            "route": SPOT_BID_ROUTE,
            "maker_market": "spot",
            "intended_quantity": 2,
            "boundary_quantile": 80,
            "full_fill": False,
        },
    ]
    return pl.from_dicts(rows, infer_schema_length=None)


def _raw_facts() -> pl.DataFrame:
    return pl.DataFrame(
        {"raw_order_fact_id": ["future-fill", "spot-fill", "spot-partial"]}
    )


def _quantity_paths() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "raw_order_fact_id": ["spot-partial", "spot-partial"],
            "fill_recv_time_ns": [400 * MS, 500 * MS],
            "fill_event_sequence": [2, 2],
            "fill_row_index": [40, 50],
            "fill_quantity": [1, 1],
        }
    )


class HedgeStudyTest(unittest.TestCase):
    def test_end_to_end_dedup_causal_books_and_slippage_split(self) -> None:
        result = run_hedge_study(
            _aliases(),
            _raw_facts(),
            _raw_tape(),
            quantity_paths=_quantity_paths(),
        )
        self.assertEqual(result.hedge_facts.height, 2)

        future = result.hedge_facts.filter(
            pl.col("raw_order_fact_id") == "future-fill"
        ).row(0, named=True)
        self.assertEqual(future["full_fill_policy_alias_count"], 2)
        self.assertEqual(future["arrival_snapshot_recv_time_ns"], 90 * MS)
        self.assertEqual(future["decision_snapshot_recv_time_ns"], 150 * MS)
        self.assertEqual(future["hedge_quantity_unit"], "spot_board_lot")
        self.assertEqual(future["requested_hedge_quantity"], 2)
        self.assertEqual(future["status"], "executable")
        self.assertEqual(future["executable_vwap_price"], 101.5)
        self.assertAlmostEqual(future["signed_latency_slippage_bp"], 100.0)
        self.assertAlmostEqual(
            future["signed_depth_slippage_bp"], 0.5 / 101.0 * 10_000.0
        )
        self.assertAlmostEqual(future["signed_total_slippage_bp"], 150.0)

        spot = result.hedge_facts.filter(
            pl.col("raw_order_fact_id") == "spot-fill"
        ).row(0, named=True)
        self.assertEqual(spot["arrival_snapshot_recv_time_ns"], 300 * MS)
        self.assertEqual(spot["arrival_reference_price"], 100.0)
        self.assertEqual(spot["decision_snapshot_recv_time_ns"], 350 * MS)
        self.assertEqual(spot["hedge_quantity_unit"], "future_contract")
        self.assertEqual(spot["requested_hedge_quantity"], 1)
        self.assertEqual(spot["executable_vwap_price"], 99.0)
        self.assertAlmostEqual(spot["signed_latency_slippage_bp"], 100.0)
        self.assertEqual(spot["signed_depth_slippage_bp"], 0.0)

        pooled = result.hedge_summary.filter(
            pl.col("route") == FUTURE_ASK_ROUTE
        ).row(0, named=True)
        self.assertEqual(pooled["hedge_events"], 1)
        self.assertEqual(pooled["executable_rate"], 1.0)

    def test_future_best_and_l1_are_merged_without_double_counting(self) -> None:
        row = {
            "best_bid_price": 100.0,
            "best_bid_lots": 3,
            "bid_price_1": 100.0,
            "bid_lots_1": 2,
            "bid_price_2": 99.0,
            "bid_lots_2": 4,
        }
        levels = executable_levels_from_state(row, "future", "bid")
        self.assertEqual([(level.price, level.quantity) for level in levels], [(100.0, 3), (99.0, 4)])

    def test_insufficient_l5_depth_is_an_explicit_non_execution(self) -> None:
        tape = _raw_tape()
        spot = tape.spot_states.with_columns(
            pl.when(pl.col("recv_time_ns") == 150 * MS)
            .then(None)
            .otherwise(pl.col("ask_price_2"))
            .alias("ask_price_2"),
            pl.when(pl.col("recv_time_ns") == 150 * MS)
            .then(None)
            .otherwise(pl.col("ask_lots_2"))
            .alias("ask_lots_2"),
        )
        shallow = RawTapeDay(
            tape.date,
            tape.mapping,
            spot,
            tape.future_states,
            tape.spot_trades,
            tape.future_trades,
            tape.audit,
        )
        aliases = _aliases().filter(
            pl.col("raw_order_fact_id") == "future-fill"
        )
        result = run_hedge_study(aliases, _raw_facts(), shallow)
        fact = result.hedge_facts.row(0, named=True)
        self.assertEqual(fact["status"], "insufficient_depth")
        self.assertEqual(fact["available_quantity"], 1)
        self.assertEqual(fact["executed_quantity"], 1)
        self.assertEqual(fact["depth_shortfall"], 1)
        self.assertIsNone(fact["signed_total_slippage_bp"])
        summary = result.hedge_summary.row(0, named=True)
        self.assertEqual(summary["executable_rate"], 0.0)
        self.assertEqual(summary["insufficient_depth_rate"], 1.0)

    def test_partial_horizons_need_real_path_and_reach_two_lots(self) -> None:
        with_path = run_hedge_study(
            _aliases(),
            _raw_facts(),
            _raw_tape(),
            quantity_paths=_quantity_paths(),
        ).spot_partial_horizons
        path = with_path.filter(pl.col("raw_order_fact_id") == "spot-partial")
        self.assertEqual(path["horizon_ms"].to_list(), [50, 250, 1000, 2000, 5000])
        self.assertEqual(path["spot_fill_quantity"].to_list(), [1, 2, 2, 2, 2])
        self.assertEqual(path["complete_two_lots"].to_list(), [False, True, True, True, True])
        self.assertEqual(path.item(1, "spot_fill_share_quantity"), 2000)

        without_path = run_hedge_study(
            _aliases(), _raw_facts(), _raw_tape()
        ).spot_partial_horizons
        self.assertTrue(without_path.is_empty())

    def test_exact_fill_cursor_is_required(self) -> None:
        aliases = _aliases().drop("full_fill_event_sequence")
        with self.assertRaisesRegex(ValueError, "full_fill_event_sequence"):
            run_hedge_study(aliases, _raw_facts(), _raw_tape())

    def test_writer_uses_wp03_specific_config_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_hedge_study(
                _aliases(),
                _raw_facts(),
                _raw_tape(),
                quantity_paths=_quantity_paths(),
                output_dir=output,
            )
            self.assertTrue((output / "hedge_facts.parquet").exists())
            self.assertTrue((output / "hedge_config.json").exists())
            self.assertFalse((output / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
