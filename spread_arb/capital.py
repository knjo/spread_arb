"""動用本金與報酬率（實際要拿出多少現金）。

「動用本金」= 兩腳實際要拿出的現金，分開算、不混：
  現貨腳 + 期貨腳保證金 → 動用本金 = 部位×(現貨率 + 期貨率)。
報酬率 = 淨獲利金額 / 動用本金（資金效率，分母是真的要拿出的錢）。

兩種模式（MARGIN_MODES）：
  mode 1（融資融券）：買現自備4成、賣現融券9成；期貨保證金13.5%。
  mode 2（保守）：現貨全額自備(不融資券，可能借不到)；期貨保證金40%。
保證金率為常見預設，可調。
"""
from __future__ import annotations

import polars as pl

from .spread import SIDE_SELL, SIDE_BUY

# 模式 → {買現比例, 賣現比例, 期貨保證金率}
MARGIN_MODES = {
    # mode 1 融資融券：買現自備4成、賣現融券9成、期貨13.5%
    "1": {"buy_stock": 0.40, "short_stock": 0.90, "fut": 0.135},
    # mode 2 保守：現貨全額(不融資券)、期貨保證金40%
    "2": {"buy_stock": 1.00, "short_stock": 1.00, "fut": 0.40},
}
DEFAULT_MODE = "1"


def _ratios(side: str, mode: str) -> float:
    """該方向、該模式的總保證金率 = 現貨腳率 + 期貨率。"""
    m = MARGIN_MODES[mode]
    spot = m["buy_stock"] if side == SIDE_SELL else m["short_stock"]
    return spot + m["fut"]


def add_capital(df: pl.DataFrame, side: str,
                mode: str = DEFAULT_MODE,
                value_col: str = "potential_value",
                pnl_col: str = "potential_pnl") -> pl.DataFrame:
    """新增動用本金與報酬率欄。

    mode：'margin'(融資融券) 或 'conservative'(現貨全額+期貨40%)。
    capital      動用本金(元) = 部位 × (現貨率 + 期貨率)
    roi          報酬率 = 淨獲利 / 動用本金
    roi_annual   年化報酬率 = roi × 365/天數（需 days_to_settle）
    """
    capital = pl.col(value_col) * _ratios(side, mode)
    df = df.with_columns(capital.alias("capital")).with_columns(
        (pl.col(pnl_col) / pl.col("capital")).alias("roi")
    )
    if "days_to_settle" in df.columns:
        # days=0(結算日當天)視為持有1天，避免除零產生 inf(審計 V03-2)
        df = df.with_columns(
            (pl.col("roi") * 365 / pl.col("days_to_settle").clip(lower_bound=1))
            .alias("roi_annual")
        )
    return df
