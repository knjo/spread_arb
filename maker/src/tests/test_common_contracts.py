"""Contract-universe tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.common.contracts import load_spot_reference


class SpotReferenceUniverseTest(unittest.TestCase):
    def test_buy_first_strategy_accepts_x_and_y_but_not_n(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "market.parquet"
            pl.DataFrame(
                {
                    "quote_code": ["x", "y", "n"],
                    "opening_ref_price": [10.0, 20.0, 30.0],
                    "allow_day_trade_mark": ["X", "Y", "N"],
                    "trading_turnover": [1.0, 1.0, 1.0],
                    "ins_type": ["stock", "stock", "stock"],
                }
            ).write_parquet(path)
            with patch(
                "maker.src.common.contracts.market_data_path",
                return_value=path,
            ):
                result = load_spot_reference("20260101")

        self.assertEqual(result["ValueCode"].to_list(), ["x", "y"])
        self.assertEqual(result["day_trade_mark"].to_list(), ["X", "Y"])


if __name__ == "__main__":
    unittest.main()
