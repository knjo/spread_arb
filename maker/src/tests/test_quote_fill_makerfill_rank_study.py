from __future__ import annotations

import unittest
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

from maker.src.quote_fill.makerfill_rank_study import (
    LegacyMakerFillRankIndex,
    label_selected_snapshots,
)
from maker.src.quote_fill.makerfill_rank_study_runner import (
    MakerFillRankStudyConfig,
    publish_makerfill_rank_study,
    run_makerfill_rank_study,
    verify_makerfill_rank_study,
)
from maker.src.quote_fill.execution_runner import EXECUTION_RUNNER_VERSION


def _frame() -> pl.DataFrame:
    records = []
    trans = [0, 1, 2, 3, 4, 5]
    fills = [
        (0.0, 0),
        (99.0, 1),
        (100.0, 2),
        (101.0, 1),
        (102.0, 4),
        (98.0, 1),
    ]
    for index, (second, (fill_price, fill_lots)) in enumerate(
        zip(trans, fills, strict=True)
    ):
        record: dict[str, object] = {
            "TransTime": datetime(2026, 7, 31, 9, 0, second),
            "ChannelSeq": 10 + index,
            "marketOpen": True,
            "FillPrice": fill_price,
            "FillLots": fill_lots,
        }
        for level in range(1, 6):
            record[f"BidPrice{level}"] = 101.0 - level
            record[f"BidLots{level}"] = level + 1
            record[f"AskPrice{level}"] = 100.0 + level
            record[f"AskLots{level}"] = level + 1
        records.append(record)
    return pl.from_dicts(records, infer_schema_length=None)


def _brute(
    frame: pl.DataFrame, position: int, side: str, level: int
) -> tuple[float | None, str | None]:
    ordered = frame.sort("TransTime")
    prefix = "Ask" if side == "ask" else "Bid"
    target = float(ordered.item(position, f"{prefix}Price{level}"))
    queue = int(ordered.item(position, f"{prefix}Lots{level}"))
    cumulative = 0
    start = ordered.item(position, "TransTime")
    for later in range(position + 1, ordered.height):
        price = float(ordered.item(later, "FillPrice"))
        quantity = int(ordered.item(later, "FillLots"))
        if price <= 0 or quantity <= 0:
            continue
        if (side == "ask" and price > target) or (
            side == "bid" and price < target
        ):
            delta = (ordered.item(later, "TransTime") - start).total_seconds()
            return float(np.float32(delta)), "trade_through"
        if abs(price - target) < 1e-8:
            cumulative += quantity
            if cumulative >= queue:
                delta = (ordered.item(later, "TransTime") - start).total_seconds()
                return float(np.float32(delta)), "same_price_displayed_queue"
    return None, None


class LegacyMakerFillRankIndexTest(unittest.TestCase):
    def test_all_five_levels_match_brute_legacy_rule(self) -> None:
        frame = _frame()
        index = LegacyMakerFillRankIndex(frame)
        for position in range(frame.height):
            for side in ("ask", "bid"):
                for level in range(1, 6):
                    with self.subTest(position=position, side=side, level=level):
                        expected_seconds, expected_reason = _brute(
                            frame, position, side, level
                        )
                        actual = index.label(10 + position, side, level)
                        self.assertEqual(actual.fill_seconds, expected_seconds)
                        self.assertEqual(actual.fill_reason, expected_reason)
                        self.assertFalse(actual.outcome_exact)
                        self.assertFalse(actual.cancel_bounded)
                        self.assertFalse(actual.own_quantity_included)

    def test_same_price_threshold_is_legacy_greater_equal_displayed(self) -> None:
        frame = _frame().with_columns(
            pl.when(pl.col("ChannelSeq") == 11)
            .then(0.0)
            .otherwise(pl.col("FillPrice"))
            .alias("FillPrice")
        )
        label = LegacyMakerFillRankIndex(frame).label(10, "bid", 1)
        # B1=100 and displayed queue=2; the two-lot print at t=2 reaches the
        # historical >=Q threshold.  It is intentionally not the modern
        # queue+own-quantity full-fill contract.
        self.assertEqual(label.fill_seconds, 2.0)
        self.assertEqual(label.fill_reason, "same_price_displayed_queue")

    def test_trade_through_wins_before_same_price_queue(self) -> None:
        label = LegacyMakerFillRankIndex(_frame()).label(10, "bid", 2)
        # B2=99; the final 98 print crosses through before five same-price lots.
        self.assertEqual(label.fill_seconds, 5.0)
        self.assertEqual(label.fill_reason, "trade_through")

    def test_invalid_level_and_missing_channel_fail_closed(self) -> None:
        index = LegacyMakerFillRankIndex(_frame())
        with self.assertRaisesRegex(ValueError, "level"):
            index.label(10, "bid", 6)
        with self.assertRaisesRegex(KeyError, "unknown ChannelSeq"):
            index.label(999, "bid", 1)

    def test_selected_requests_preserve_request_identity(self) -> None:
        result = label_selected_snapshots(
            _frame(),
            pl.DataFrame(
                {
                    "request_id": ["a", "b"],
                    "ChannelSeq": [10, 11],
                    "side": ["bid", "ask"],
                    "level": [1, 5],
                }
            ),
        )
        self.assertEqual(result.get_column("request_id").to_list(), ["a", "b"])
        self.assertEqual(result.get_column("outcome_exact").to_list(), [False, False])

    def test_duplicate_channel_sequence_is_rejected(self) -> None:
        duplicate = pl.concat([_frame(), _frame().head(1)])
        with self.assertRaisesRegex(ValueError, "ChannelSeq"):
            LegacyMakerFillRankIndex(duplicate)

    def test_producer_epsilon_and_market_open_filter_are_exact(self) -> None:
        frame = _frame().with_columns(
            pl.when(pl.col("ChannelSeq") == 12)
            .then(100.0 + 5e-9)
            .otherwise(pl.col("FillPrice"))
            .alias("FillPrice"),
            pl.when(pl.col("ChannelSeq") == 11)
            .then(False)
            .otherwise(pl.col("marketOpen"))
            .alias("marketOpen"),
        )
        label = LegacyMakerFillRankIndex(frame).label(10, "bid", 1)
        self.assertEqual(label.fill_seconds, 2.0)
        self.assertEqual(label.fill_reason, "same_price_displayed_queue")

    def test_epsilon_boundary_and_nextafter_match_literal_producer(self) -> None:
        cases = (
            ("ask", 101.0 - 1e-8),
            ("ask", float(np.nextafter(101.0 - 1e-8, 101.0))),
            ("ask", float(np.nextafter(101.0 - 1e-8, -np.inf))),
            ("bid", 100.0 + 1e-8),
            ("bid", float(np.nextafter(100.0 + 1e-8, 100.0))),
            ("bid", float(np.nextafter(100.0 + 1e-8, np.inf))),
        )
        for side, price in cases:
            with self.subTest(side=side, price=repr(price)):
                frame = _frame().with_columns(
                    pl.when(pl.col("ChannelSeq") == 12)
                    .then(price)
                    .otherwise(pl.col("FillPrice"))
                    .alias("FillPrice")
                )
                expected = _brute(frame, 0, side, 1)
                actual = LegacyMakerFillRankIndex(frame).label(10, side, 1)
                self.assertEqual(
                    (actual.fill_seconds, actual.fill_reason), expected
                )


class MakerFillRankStudyRunnerTest(unittest.TestCase):
    def _fixture(self, root: Path) -> MakerFillRankStudyConfig:
        date = "20260731"
        execution = root / "execution"
        partition = execution / f"Date={date}" / "ValueCode=1513"
        partition.mkdir(parents=True)
        actions = pl.from_dicts(
            [
                {
                    "Date": date,
                    "ValueCode": "1513",
                    "QuoteCode": "SDFG6",
                    "route": "spot_bid_future_taker",
                    "maker_market": "spot",
                    "maker_side": "bid",
                    "boundary_quantile": 50,
                    "raw_order_fact_id": "raw-b1",
                    "policy_generation_id": "policy-b1",
                    "target_rank_at_submit": "BID1",
                    "target_price": 100.0,
                    "initial_queue_ahead": 2,
                    "intended_quantity": 2,
                    "submit_recv_time_ns": 1_000_000_000,
                    "submit_event_sequence": 2,
                    "submit_row_index": 10,
                    "nominal_stop_recv_time_ns": 4_000_000_000,
                    "nominal_stop_reason": "target_retreat",
                    "full_fill_recv_time_ns": 3_000_000_000,
                    "full_fill": True,
                    "entry_hedge_executable": True,
                    "entry_hedge_signed_total_slippage_bp": 1.25,
                },
                {
                    "Date": date,
                    "ValueCode": "1513",
                    "QuoteCode": "SDFG6",
                    "route": "spot_bid_future_taker",
                    "maker_market": "spot",
                    "maker_side": "bid",
                    "boundary_quantile": 50,
                    "raw_order_fact_id": "raw-b3",
                    "policy_generation_id": "policy-b3",
                    "target_rank_at_submit": "BID3",
                    "target_price": 98.0,
                    "initial_queue_ahead": 4,
                    "intended_quantity": 2,
                    "submit_recv_time_ns": 1_000_000_000,
                    "submit_event_sequence": 2,
                    "submit_row_index": 10,
                    "nominal_stop_recv_time_ns": 3_000_000_000,
                    "nominal_stop_reason": "target_retreat",
                    "full_fill_recv_time_ns": None,
                    "full_fill": False,
                    "entry_hedge_executable": False,
                    "entry_hedge_signed_total_slippage_bp": None,
                },
            ],
            infer_schema_length=None,
        )
        action_path = partition / "execution_action_facts.parquet"
        actions.write_parquet(action_path)
        config_payload = {"hedge_delay_ns": 50_000_000}
        config_sha = hashlib.sha256(
            json.dumps(
                config_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        (partition / "complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "Date": date,
                    "ValueCode": "1513",
                    "runner_version": EXECUTION_RUNNER_VERSION,
                    "config": config_payload,
                    "config_sha256": config_sha,
                    "fact_semantics": {
                        "hedge_delay_ns": 50_000_000,
                        "independent_event_label": True,
                        "joint_volume_allocated": False,
                    },
                    "artifacts": {
                        "execution_action_facts.parquet": {
                            "rows": actions.height,
                            "columns": actions.width,
                            "bytes": action_path.stat().st_size,
                            "sha256": hashlib.sha256(
                                action_path.read_bytes()
                            ).hexdigest(),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        pl.DataFrame(
            {
                "Date": [date],
                "ValueCode": ["1513"],
                "partition": [str(partition)],
                "config_sha256": [config_sha],
                "complete": [True],
                "execution_action_facts_rows": [actions.height],
            }
        ).write_parquet(execution / "execution_partition_manifest.parquet")

        tick_root = root / "ticks"
        makerfill_root = root / "makerfill"
        tick_root.mkdir()
        makerfill_root.mkdir()
        ticks = _frame().with_columns(pl.lit("1513").alias("ValueCode"))
        ticks.write_parquet(tick_root / f"{date}_StockTick.parquet")
        makerfill = pl.DataFrame(
            {
                "QuoteCode": ["1513"],
                "ChannelSeq": [10],
                "Bid1_FillSeconds": [1.0],
                "Bid2_FillSeconds": [5.0],
            }
        )
        makerfill.write_parquet(
            makerfill_root / f"{date}_makerFill.parquet"
        )
        return MakerFillRankStudyConfig(
            execution_root=str(execution),
            tick_root=str(tick_root),
            makerfill_root=str(makerfill_root),
            dates=(date,),
        )

    def test_real_runner_contract_and_rank_group_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixture(Path(temporary))
            result = run_makerfill_rank_study(config)
            self.assertEqual(result.raw_labels.height, 2)
            self.assertEqual(result.alias_comparison.height, 2)
            self.assertEqual(
                result.raw_labels.filter(
                    pl.col("stored_l1_l2_parity") == False  # noqa: E712
                ).height,
                0,
            )
            rollup = result.rank_summary.filter(
                (pl.col("Date") == "__all__")
                & (pl.col("population_unit") == "q_raw_order")
                & (pl.col("rank_scope") == "rank_group")
            )
            self.assertEqual(set(rollup["rank_group"]), {"L1_2", "L3_5"})

    def test_atomic_publish_verifies_and_safety_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            output = root / "output"
            publish_makerfill_rank_study(output, config)
            marker = verify_makerfill_rank_study(output)
            self.assertTrue(marker["safety"]["analysis_only"])
            marker["safety"]["pathwise_ev_ready"] = True
            (output / "complete.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "safety"):
                verify_makerfill_rank_study(output)

    def test_verify_recomputes_sources_and_rejects_coordinated_dtype_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            output = root / "output"
            publish_makerfill_rank_study(output, config)

            summary_path = output / "rank_summary.parquet"
            summary = pl.read_parquet(summary_path).with_columns(
                pl.col("n").cast(pl.Int64)
            )
            summary.write_parquet(summary_path)
            marker_path = output / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            meta = marker["artifacts"]["rank_summary.parquet"]
            meta.update(
                {
                    "bytes": summary_path.stat().st_size,
                    "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "rows": summary.height,
                    "columns": summary.width,
                    "schema": [
                        {"name": name, "dtype": str(dtype)}
                        for name, dtype in summary.schema.items()
                    ],
                }
            )
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recomputation"):
                verify_makerfill_rank_study(output)

    def test_null_execution_truth_is_not_imputed_as_no_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            action_path = next(
                (root / "execution").glob(
                    "Date=*/ValueCode=*/execution_action_facts.parquet"
                )
            )
            actions = pl.read_parquet(action_path).with_columns(
                pl.when(pl.col("policy_generation_id") == "policy-b3")
                .then(pl.lit(None, dtype=pl.Boolean))
                .otherwise(pl.col("full_fill"))
                .alias("full_fill")
            )
            actions.write_parquet(action_path)
            with self.assertRaisesRegex(ValueError, "null or incoherent"):
                run_makerfill_rank_study(config)

    def test_direct_snapshot_identity_and_float32_parity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            action_path = next(
                (root / "execution").glob(
                    "Date=*/ValueCode=*/execution_action_facts.parquet"
                )
            )
            original = pl.read_parquet(action_path)
            original.with_columns(
                pl.when(pl.col("policy_generation_id") == "policy-b3")
                .then(pl.col("target_price") + 1.0)
                .otherwise(pl.col("target_price"))
                .alias("target_price")
            ).write_parquet(action_path)
            with self.assertRaisesRegex(ValueError, "direct tick snapshot"):
                run_makerfill_rank_study(config)

            original.write_parquet(action_path)
            makerfill_path = root / "makerfill" / "20260731_makerFill.parquet"
            pl.read_parquet(makerfill_path).with_columns(
                pl.lit(1.0000005).alias("Bid1_FillSeconds")
            ).write_parquet(makerfill_path)
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                run_makerfill_rank_study(config)


if __name__ == "__main__":
    unittest.main()
