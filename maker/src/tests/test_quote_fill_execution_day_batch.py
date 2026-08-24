from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import polars as pl

from maker.src.quote_fill.execution_runner import (
    ExecutionProductDayResult,
    ExecutionRunnerConfig,
    WalkForwardExecutionDayBatch,
    WalkForwardProductDaySources,
    load_spread_pair_clock_for_raw_day,
    load_walkforward_execution_day_batch,
    merged_product_day_from_batch,
    run_day_batched_execution_replay,
)
from maker.src.quote_fill.merged import MergedTargetStudyInput
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.targets import PRICE_LADDER_VERSION


DATE = "20260102"
PRODUCTS = ("2317", "2603")
QUOTES = ("DHFA6", "DKFA6")
QUANTILES = (50, 80, 95)
MODULE = "maker.src.quote_fill.execution_runner"


def _source(value_code: str, quote_code: str) -> WalkForwardProductDaySources:
    mapping = pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
            "spot_ref_price": [100.0],
            "fut_ref_price": [101.0],
            "contract_size": [2000.0],
        }
    )
    fair = pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
        }
    )
    boundaries = pl.DataFrame(
        {
            "Date": [DATE] * 3,
            "ValueCode": [value_code] * 3,
            "QuoteCode": [quote_code] * 3,
            "boundary_quantile": list(QUANTILES),
        }
    )
    return WalkForwardProductDaySources(
        date=DATE,
        value_code=value_code,
        quote_code=quote_code,
        mapping=mapping,
        causal_fair=fair,
        rolling_boundaries=boundaries,
        daily_schema_version="test-daily-v1",
        migrated_daily_marker=False,
        source_asof_date="20260101",
    )


def _shared_tape(mapping: pl.DataFrame) -> RawTapeDay:
    identities = mapping.select("ValueCode", "QuoteCode").with_columns(
        pl.lit(DATE).alias("Date"),
        pl.int_range(1, pl.len() + 1, dtype=pl.UInt64).alias("sequence"),
    ).select("Date", "ValueCode", "QuoteCode", "sequence")
    trades = mapping.select("ValueCode").head(0)
    audit = identities.select("Date", "ValueCode")
    return RawTapeDay(
        date=DATE,
        mapping=mapping,
        spot_states=identities,
        future_states=identities,
        spot_trades=trades,
        future_trades=trades,
        audit=audit,
    )


def _batch() -> WalkForwardExecutionDayBatch:
    sources = tuple(
        _source(value_code, quote_code)
        for value_code, quote_code in zip(PRODUCTS, QUOTES, strict=True)
    )
    mapping = pl.concat(
        [source.mapping.drop("Date") for source in sources]
    )
    return WalkForwardExecutionDayBatch(
        date=DATE,
        value_codes=PRODUCTS,
        sources=sources,
        raw_tape=_shared_tape(mapping),
        spot_feature_state=pl.DataFrame(
            {"Date": [DATE, DATE], "ValueCode": list(PRODUCTS)}
        ),
        boundary_quantiles=QUANTILES,
    )


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"Date": pl.String})


def _result_for(
    merged: MergedTargetStudyInput,
    _config: ExecutionRunnerConfig,
) -> ExecutionProductDayResult:
    value_code = str(merged.mapping.item(0, "ValueCode"))
    identity = pl.DataFrame({"Date": [DATE], "ValueCode": [value_code]})
    empty = _empty_frame()
    return ExecutionProductDayResult(
        order_aliases=empty,
        raw_order_facts=empty,
        hedge_facts=empty,
        execution_action_facts=empty,
        execution_daily_facts=empty,
        exit_facts=empty,
        exit_daily_facts=empty,
        policy_audit=identity,
        target_audit=identity,
        raw_tape_audit=identity,
    )


class ExecutionDayBatchTest(unittest.TestCase):
    def test_spread_clock_reuses_raw_keys_without_reopening_stock_tick(self) -> None:
        batch = _batch()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            feature_dir = data_root / "tickFeature"
            feature_dir.mkdir()
            pl.DataFrame(
                {
                    "QuoteCode": [*PRODUCTS, "9999"],
                    "ChannelSeq": [1, 2, 999],
                    "SpreadPairID": [10, 20, 99],
                    "SpreadPairSeq": [1, 1, 1],
                    "SpreadPairTotalCount": [100, 200, 999],
                    "SpreadCountAtSameCount": [1, 1, 1],
                }
            ).write_parquet(
                feature_dir / f"{DATE}_tickFeature.parquet"
            )
            clock = load_spread_pair_clock_for_raw_day(
                DATE,
                PRODUCTS,
                batch.raw_tape,
                data_root=data_root,
            )

        self.assertEqual(clock["ValueCode"].to_list(), list(PRODUCTS))
        self.assertEqual(clock["spread_pair_epoch"].to_list(), [100, 200])
        self.assertEqual(
            clock.columns,
            [
                "Date",
                "ValueCode",
                "spot_channel_seq",
                "spread_pair_id",
                "spread_pair_seq",
                "spread_pair_epoch",
                "spread_count_at_same_count",
            ],
        )

    def test_shared_day_loads_and_normalizes_raw_once_then_slices_exactly(self) -> None:
        sources = {
            value_code: _source(value_code, quote_code)
            for value_code, quote_code in zip(PRODUCTS, QUOTES, strict=True)
        }
        source_loader = Mock(side_effect=lambda _date, code, **_kwargs: sources[code])
        raw_loader = Mock(side_effect=lambda _date, mapping: _shared_tape(mapping))
        feature_loader = Mock(
            return_value=pl.DataFrame(
                {"Date": [DATE, DATE], "ValueCode": list(PRODUCTS)}
            )
        )

        def merged_builder(
            _date: str,
            spot: pl.DataFrame,
            _future: pl.DataFrame,
            _feature: pl.DataFrame,
            _fair: pl.DataFrame,
            _boundaries: pl.DataFrame,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            value_code = str(spot.item(0, "ValueCode"))
            return (
                pl.DataFrame(),
                pl.DataFrame({"Date": [DATE], "ValueCode": [value_code]}),
            )

        with (
            patch(f"{MODULE}.load_walkforward_product_day_sources", source_loader),
            patch(f"{MODULE}.load_raw_tape_day", raw_loader),
            patch(
                f"{MODULE}.load_spread_pair_clock_for_raw_day",
                feature_loader,
            ),
            patch(
                f"{MODULE}.build_merged_target_observations_from_frames",
                side_effect=merged_builder,
            ) as builder,
        ):
            batch = load_walkforward_execution_day_batch(DATE, PRODUCTS)
            merged = tuple(
                merged_product_day_from_batch(batch, value_code)
                for value_code in PRODUCTS
            )

        self.assertEqual(source_loader.call_count, 2)
        self.assertEqual(raw_loader.call_count, 1)
        self.assertEqual(feature_loader.call_count, 1)
        self.assertEqual(builder.call_count, 2)
        loaded_mapping = raw_loader.call_args.args[1]
        self.assertEqual(
            set(loaded_mapping["ValueCode"].to_list()), set(PRODUCTS)
        )
        self.assertEqual(
            feature_loader.call_args.args[1], PRODUCTS
        )
        for value_code, product in zip(PRODUCTS, merged, strict=True):
            self.assertEqual(product.mapping["ValueCode"].to_list(), [value_code])
            self.assertEqual(
                product.raw_tape.spot_states["ValueCode"].to_list(),
                [value_code],
            )
            self.assertTrue(
                product.audit.item(0, "execution_safe_source_validated")
            )
            self.assertEqual(
                product.audit.item(0, "boundary_quantiles_loaded"),
                "50,80,95",
            )

    def test_unsafe_small_source_fails_before_any_raw_io(self) -> None:
        raw_loader = Mock()

        def source_loader(_date: str, value_code: str, **_kwargs):
            if value_code == PRODUCTS[1]:
                raise ValueError("source_asof_date must be strictly before target Date")
            return _source(PRODUCTS[0], QUOTES[0])

        with (
            patch(
                f"{MODULE}.load_walkforward_product_day_sources",
                side_effect=source_loader,
            ),
            patch(f"{MODULE}.load_raw_tape_day", raw_loader),
        ):
            with self.assertRaisesRegex(ValueError, "strictly before"):
                load_walkforward_execution_day_batch(DATE, PRODUCTS)
        raw_loader.assert_not_called()

    def test_day_runner_keeps_atomic_partition_schema_and_resumes_day(self) -> None:
        batch = _batch()
        batch_loader = Mock(return_value=batch)

        def merged_loader(
            loaded: WalkForwardExecutionDayBatch, value_code: str
        ) -> MergedTargetStudyInput:
            source = next(
                source for source in loaded.sources
                if source.value_code == value_code
            )
            tape = RawTapeDay(
                loaded.date,
                source.mapping.drop("Date"),
                loaded.raw_tape.spot_states.filter(
                    pl.col("ValueCode") == value_code
                ),
                loaded.raw_tape.future_states.filter(
                    pl.col("ValueCode") == value_code
                ),
                loaded.raw_tape.spot_trades.filter(
                    pl.col("ValueCode") == value_code
                ),
                loaded.raw_tape.future_trades.filter(
                    pl.col("ValueCode") == value_code
                ),
                loaded.raw_tape.audit.filter(
                    pl.col("ValueCode") == value_code
                ),
            )
            return MergedTargetStudyInput(
                loaded.date,
                source.mapping.drop("Date"),
                tape,
                pl.DataFrame(),
                pl.DataFrame({"Date": [DATE], "ValueCode": [value_code]}),
            )

        config = ExecutionRunnerConfig(boundary_quantiles=QUANTILES)
        keys = tuple((DATE, value_code) for value_code in PRODUCTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    f"{MODULE}.merged_product_day_from_batch",
                    side_effect=merged_loader,
                ),
                patch(
                    f"{MODULE}.replay_execution_product_day",
                    side_effect=_result_for,
                ),
            ):
                first = run_day_batched_execution_replay(
                    keys, batch_loader, root, config
                )
                resumed = run_day_batched_execution_replay(
                    keys, batch_loader, root, config
                )

            self.assertEqual(first.height, 2)
            self.assertEqual(resumed.height, 2)
            self.assertEqual(batch_loader.call_count, 1)
            batch_loader.assert_called_once_with(DATE, PRODUCTS)
            expected_files = {
                "complete.json",
                "order_aliases.parquet",
                "raw_order_facts.parquet",
                "hedge_facts.parquet",
                "execution_action_facts.parquet",
                "execution_daily_facts.parquet",
                "exit_facts.parquet",
                "exit_daily_facts.parquet",
                "policy_audit.parquet",
                "target_audit.parquet",
                "raw_tape_audit.parquet",
            }
            for value_code in PRODUCTS:
                partition = root / f"Date={DATE}" / f"ValueCode={value_code}"
                self.assertEqual(
                    {path.name for path in partition.iterdir()}, expected_files
                )
                marker = json.loads(
                    (partition / "complete.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    marker["config"]["price_ladder_version"],
                    PRICE_LADDER_VERSION,
                )
                self.assertEqual(
                    marker["fact_semantics"]["price_ladder_version"],
                    PRICE_LADDER_VERSION,
                )
            self.assertTrue(
                (root / "execution_partition_manifest.parquet").is_file()
            )

    def test_execution_config_rejects_an_unimplemented_ladder_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "price_ladder_version"):
            ExecutionRunnerConfig(
                price_ladder_version="legacy-unversioned"
            ).validate()


if __name__ == "__main__":
    unittest.main()
