"""Focused contracts for the indexed boundary batch engine."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import polars as pl

from maker.src.quote_fill.foundation_boundary_batch import (
    BoundaryBatchEngine,
    build_boundary_batch_predictions,
)
from maker.src.quote_fill.foundation_boundary_selection import (
    build_boundary_candidate_predictions,
)

ANCHOR_ID = "time_ewma_30s"
MINIMA_ONE = {50: 1, 80: 1, 95: 1}


def _date_text(value: date) -> str:
    return value.strftime("%Y%m%d")


def _episode(
    trading_date: date,
    expiry: date,
    amplitude: float,
    *,
    product: str = "1111",
    quote_code: str = "QHIST",
    side: str = "positive",
    completed: bool = True,
    left_censored: bool = False,
    anchor_model_id: str = ANCHOR_ID,
) -> dict[str, object]:
    return {
        "Date": _date_text(trading_date),
        "ValueCode": product,
        "QuoteCode": quote_code,
        "anchor_model_id": anchor_model_id,
        "side": side,
        "observed_amplitude_bp": amplitude,
        "left_censored": left_censored,
        "right_censored": not completed,
        "completed_center_return": completed,
        "start_seconds_from_open": 300,
        "end_seconds_from_open": 301 if completed else 14_399,
        "right_censor_reason": None if completed else "entry_stop",
        "tod_bucket": "0905_1000",
        "expiry_date": expiry,
        "calendar_dte": (expiry - trading_date).days,
    }


def _target(
    trading_date: date,
    expiry: date,
    *,
    product: str = "1111",
    quote_code: str = "QTARGET",
) -> dict[str, object]:
    return {
        "Date": _date_text(trading_date),
        "ValueCode": product,
        "QuoteCode": quote_code,
        "expiry_date": expiry,
        "calendar_dte": (expiry - trading_date).days,
    }


def _assert_oracle_equivalent(
    testcase: unittest.TestCase,
    batch: pl.DataFrame,
    oracle: pl.DataFrame,
) -> None:
    keys = ["tod_bucket", "boundary_quantile", "side"]
    batch_rows = {
        tuple(row[column] for column in keys): row
        for row in batch.iter_rows(named=True)
    }
    oracle_rows = {
        tuple(row[column] for column in keys): row
        for row in oracle.iter_rows(named=True)
    }
    testcase.assertEqual(set(batch_rows), set(oracle_rows))
    float_columns = {
        "completed_point_bp",
        "identified_lower_bp",
        "identified_upper_bp",
        "clipped_point_bp",
        "boundary_distance_bp",
    }
    ignored = {"anchor_model_id"}
    for key in sorted(batch_rows):
        actual = batch_rows[key]
        expected = oracle_rows[key]
        for column in oracle.columns:
            if column in ignored or column in keys:
                continue
            if column in float_columns:
                if actual[column] is None or expected[column] is None:
                    testcase.assertIs(actual[column], expected[column])
                else:
                    testcase.assertAlmostEqual(
                        float(actual[column]),
                        float(expected[column]),
                        places=10,
                    )
            else:
                testcase.assertEqual(
                    actual[column],
                    expected[column],
                    msg=f"column={column}, key={key}",
                )


class TrailingOracleEquivalenceTests(unittest.TestCase):
    def test_q0_q1_q2_match_scalar_oracle(self) -> None:
        first = date(2026, 1, 1)
        history_dates = [first + timedelta(days=offset) for offset in range(60)]
        target_date = first + timedelta(days=60)
        expiry = target_date + timedelta(days=30)
        sessions = [_date_text(value) for value in [*history_dates, target_date]]
        rows: list[dict[str, object]] = []
        for offset, trading_date in enumerate(history_dates):
            for side, side_shift in (("positive", 0.0), ("negative", 0.75)):
                rows.append(
                    _episode(
                        trading_date,
                        expiry,
                        1.0 + offset / 10.0 + side_shift,
                        side=side,
                    )
                )
                if offset % 7 == 0:
                    rows.append(
                        _episode(
                            trading_date,
                            expiry,
                            20.0 + offset + side_shift,
                            side=side,
                        )
                    )
                if offset % 10 == 0:
                    rows.append(
                        _episode(
                            trading_date,
                            expiry,
                            8.0 + side_shift,
                            side=side,
                            completed=False,
                        )
                    )
        episodes = pl.from_dicts(rows, infer_schema_length=None)
        targets = pl.from_dicts(
            [_target(target_date, expiry)], infer_schema_length=None
        )

        batch = build_boundary_batch_predictions(
            episodes,
            sessions,
            targets,
            target_dates=[_date_text(target_date)],
            candidate_ids=[
                "Q0_trail60_event_pooled",
                "Q1_trail60_date_equal",
                "Q2_trail20_date_equal",
            ],
            minimum_completed_per_side=MINIMA_ONE,
        )

        for candidate_id in (
            "Q0_trail60_event_pooled",
            "Q1_trail60_date_equal",
            "Q2_trail20_date_equal",
        ):
            oracle = build_boundary_candidate_predictions(
                episodes,
                targets,
                candidate_id=candidate_id,
                minimum_completed_per_side=MINIMA_ONE,
            )
            _assert_oracle_equivalent(
                self,
                batch.filter(pl.col("candidate_id") == candidate_id),
                oracle,
            )
        self.assertEqual(batch["anchor_model_id"].unique().to_list(), [ANCHOR_ID])

    def test_q2_uses_explicit_market_sessions_and_rejects_target_outcome(self) -> None:
        first = date(2026, 4, 1)
        dates = [first + timedelta(days=offset) for offset in range(22)]
        target_date = dates[-1]
        expiry = target_date + timedelta(days=30)
        rows: list[dict[str, object]] = []
        for index, trading_date in enumerate(dates):
            amplitude = 1_000.0 if index == 0 else 10.0
            if trading_date == target_date:
                amplitude = 999.0
            for side in ("positive", "negative"):
                rows.append(
                    _episode(
                        trading_date,
                        expiry,
                        amplitude,
                        side=side,
                    )
                )
        prediction = build_boundary_batch_predictions(
            pl.from_dicts(rows, infer_schema_length=None),
            [_date_text(value) for value in dates],
            pl.from_dicts([_target(target_date, expiry)]),
            target_dates=[_date_text(target_date)],
            candidate_ids=["Q2_trail20_date_equal"],
            minimum_completed_per_side=MINIMA_ONE,
        )
        cells = prediction.filter(
            (pl.col("tod_bucket") == "0905_1000") & (pl.col("boundary_quantile") == 50)
        )
        self.assertEqual(cells.height, 2)
        for value in cells["completed_point_bp"].to_list():
            self.assertAlmostEqual(float(value), 10.0)
        self.assertEqual(prediction["selection_sessions"].unique().to_list(), [20])
        self.assertEqual(
            prediction["history_start_date"].unique().to_list(),
            [_date_text(dates[1])],
        )
        self.assertTrue((~prediction["contains_target_day_outcome"]).all())


class PriorExpiryOracleEquivalenceTests(unittest.TestCase):
    def test_q5_q6_match_scalar_oracle(self) -> None:
        target_date = date(2026, 8, 5)
        target_expiry = date(2026, 8, 19)
        rows: list[dict[str, object]] = []
        history_dates: list[date] = []
        for expiry, quote_code in (
            (date(2026, 6, 17), "QJUN"),
            (date(2026, 7, 15), "QJUL"),
        ):
            for dte in (14, 13, 12, 11, 10):
                trading_date = expiry - timedelta(days=dte)
                history_dates.append(trading_date)
                for side, shift in (("positive", 0.0), ("negative", 0.5)):
                    rows.append(
                        _episode(
                            trading_date,
                            expiry,
                            float(dte) + shift,
                            quote_code=quote_code,
                            side=side,
                        )
                    )
        sessions = [
            _date_text(value) for value in sorted({*history_dates, target_date})
        ]
        episodes = pl.from_dicts(rows, infer_schema_length=None)
        targets = pl.from_dicts([_target(target_date, target_expiry)])
        batch = build_boundary_batch_predictions(
            episodes,
            sessions,
            targets,
            target_dates=[_date_text(target_date)],
            candidate_ids=[
                "Q5_prev1_expiry_dte5",
                "Q6_prev2_expiry_dte5",
            ],
            minimum_completed_per_side=MINIMA_ONE,
        )

        for candidate_id in (
            "Q5_prev1_expiry_dte5",
            "Q6_prev2_expiry_dte5",
        ):
            oracle = build_boundary_candidate_predictions(
                episodes,
                targets,
                candidate_id=candidate_id,
                minimum_completed_per_side=MINIMA_ONE,
            )
            _assert_oracle_equivalent(
                self,
                batch.filter(pl.col("candidate_id") == candidate_id),
                oracle,
            )


class BatchScaleContractTests(unittest.TestCase):
    def test_many_products_share_one_indexed_engine(self) -> None:
        first = date(2026, 1, 1)
        history_dates = [first + timedelta(days=offset) for offset in range(60)]
        target_date = first + timedelta(days=60)
        expiry = target_date + timedelta(days=30)
        sessions = [_date_text(value) for value in [*history_dates, target_date]]
        rows: list[dict[str, object]] = []
        targets: list[dict[str, object]] = []
        products = [f"{1000 + index}" for index in range(40)]
        for product_index, product in enumerate(products):
            targets.append(
                _target(
                    target_date,
                    expiry,
                    product=product,
                    quote_code=f"T{product}",
                )
            )
            for offset, trading_date in enumerate(history_dates):
                for side, shift in (("positive", 0.0), ("negative", 0.25)):
                    rows.append(
                        _episode(
                            trading_date,
                            expiry,
                            1.0 + product_index / 10.0 + offset / 100.0 + shift,
                            product=product,
                            quote_code=f"H{product}",
                            side=side,
                        )
                    )
                    rows.append(
                        _episode(
                            trading_date,
                            expiry,
                            2.0 + product_index / 10.0 + offset / 100.0 + shift,
                            product=product,
                            quote_code=f"H{product}",
                            side=side,
                        )
                    )
        engine = BoundaryBatchEngine(
            pl.from_dicts(rows, infer_schema_length=None), sessions
        )
        prediction = engine.predict(
            pl.from_dicts(targets, infer_schema_length=None),
            target_dates=[_date_text(target_date)],
            candidate_ids=[
                "Q0_trail60_event_pooled",
                "Q1_trail60_date_equal",
                "Q2_trail20_date_equal",
            ],
            minimum_completed_per_side=MINIMA_ONE,
        )
        expected_rows = len(products) * 3 * 4 * 2 * 3
        self.assertEqual(prediction.height, expected_rows)
        self.assertEqual(prediction["ValueCode"].n_unique(), len(products))
        self.assertTrue(prediction["native_supported"].all())

    def test_mixed_anchor_history_fails_closed(self) -> None:
        trading_date = date(2026, 5, 1)
        expiry = date(2026, 5, 20)
        rows = [
            _episode(trading_date, expiry, 1.0),
            _episode(
                trading_date,
                expiry,
                2.0,
                anchor_model_id="time_ewma_60s",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "one anchor_model_id"):
            BoundaryBatchEngine(
                pl.from_dicts(rows, infer_schema_length=None),
                [_date_text(trading_date)],
            )

    def test_episode_crossing_entry_stop_fails_closed(self) -> None:
        trading_date = date(2026, 5, 1)
        expiry = date(2026, 5, 20)
        row = _episode(trading_date, expiry, 1.0)
        row["end_seconds_from_open"] = 14_400
        with self.assertRaisesRegex(ValueError, "13:00 entry cutoff"):
            BoundaryBatchEngine(
                pl.from_dicts([row], infer_schema_length=None),
                [_date_text(trading_date)],
            )


if __name__ == "__main__":
    unittest.main()
