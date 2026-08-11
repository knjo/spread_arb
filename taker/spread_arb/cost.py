"""費用扣除 → 淨獲利率（METHODOLOGY.md 步驟 8 + 證交稅）。

「費用」指要扣掉的交易開銷（手續費 + 證交稅），不是持倉成本/本金。
淨獲利率 = 毛獲利率(進場+出場兩刀價差) − 總費用率。總費用率拆兩腳：
  期貨腳(來回)：期交稅 0.002%/邊×2 + 定額手續費 20元/口/邊×口數×2 ÷ 部位金額
  現貨腳(來回)：手續費 0.1425%×15折=0.0214%/邊×2 + 證交稅(賣出一次)

證交稅（現股賣出課徵，兩方向各賣一次：價差賣=出場賣、價差買=進場賣）：
  當沖（converged，日內平倉）  ：0.15%（現股當沖減半）
  留倉（未收斂，抱到結算才賣）：0.30%（一般稅率）

利息：現階段「現金買賣、無融資融券」→ 不算任何利息/機會成本（include_interest
  預設 False）。融資/融券/無風險利率常數保留但不啟用，未來要恢復時打開即可。
利率/稅率為固定假設值，可調，改了用 stats.enrich 重算不必重跑回測。
"""
from __future__ import annotations

import polars as pl

from .spread import SIDE_SELL

# === 費用參數（拆期貨腳/現貨腳，可調，改了用 stats.enrich 重算不必重跑回測）===
# 期貨腳（來回兩邊）
FUT_TAX_RATE = 0.00002       # 期交稅 十萬分之2/邊（對成交金額）；股期實際稅率
FUT_FEE_PER_LOT = 20.0       # 期貨手續費 20 元/口/邊（定額，非率！按口數×金額算）
                             # L9：此 20 元為假設值、未經期貨商確認；標準/小型/ETF 期一律同值。
                             #   翻倍影響 <0.2%(有界)。開戶後回填實際費率、用 stats.enrich 重算即可。
# 現貨腳（來回兩邊）
SPOT_FEE_RATE = 0.001425 * 0.15   # 現貨手續費 = 公定0.1425% × 15折/邊
TAX_DAYTRADE = 0.0015        # 證交稅：現股當沖（收斂，賣出一次，減半）
TAX_NORMAL = 0.0030          # 證交稅：留倉（未收斂，賣出一次，一般）
# 利息（資金成本，年化；×天數/365）
MARGIN_RATE = 0.065          # 融資利率（mode1 價差賣＝買現融資用）
SHORT_RATE = 0.0             # 融券/借券成本（mode1 價差買＝賣現用）
RF_RATE = 0.015              # 無風險利率（資金機會成本，兩模式兩方向都算）


def _carry_rate(side: str, mode: str) -> float:
    """年化持有費用率 = 費用腳利率 + 無風險，依「方向 × 本金模式」。

    mode 1 融資融券：價差賣=融資6.5%+1.5%；價差買=融券0%+1.5%。
    mode 2 全額自備：不融資 → 融資利息 0，僅留無風險 1.5%（機會成本）。
    注意：現階段現金買賣不算利息（net_return 的 include_interest 預設 False），
          此函式僅在未來恢復利息時才會被呼叫。
    """
    if mode == "2":
        return RF_RATE
    leg = MARGIN_RATE if side == SIDE_SELL else SHORT_RATE
    return leg + RF_RATE


def net_return(df: pl.DataFrame, side: str, mode: str = "1",
               gross_col: str = "first_ret",
               days_col: str = "days_to_settle",
               value_col: str = "potential_value",
               lots_col: str = "potential_lots",
               include_interest: bool = False,   # 現金買賣不算利息（要恢復改 True）
               include_tax: bool = True) -> pl.DataFrame:
    """算淨獲利率與潛在獲利金額，新增欄 fee_rate / net_ret / 金額版三欄。

    新費用結構（率，皆對部位金額 potential_value）：
      期貨腳(來回)：期交稅 FUT_TAX_RATE×2
                  + 定額手續費 FUT_FEE_PER_LOT×potential_lots×2 ÷ potential_value
      現貨腳(來回)：手續費 SPOT_FEE_RATE×2
                  + 證交稅(賣一次)：當沖 TAX_DAYTRADE / 留倉 TAX_NORMAL（按 converged）
      利息：carry_rate(side,mode) × 天數/365（mode2 僅無風險 1.5%）
    證交稅按 converged 分流（需 df 含 converged 欄）。費用依 mode 不同（利息腳）。
    來回毛利率 = 進場價差(first_ret) + 出場價差(exit_cost，反邊 taker，通常為負)。
    潛在獲利金額(元) = 潛在部位金額 × 淨獲利率。
    """
    # --- 固定率費用（期交稅來回 + 現貨手續費來回）---
    fee = pl.lit(FUT_TAX_RATE * 2 + SPOT_FEE_RATE * 2)
    # --- 期貨定額手續費轉率：20元/口/邊 × 口數 × 2邊 ÷ 部位金額 ---
    #     value=0 時（理論上不會發生，potential_lots=0 已在 stats 剔除）保護除零→該項 0
    if lots_col in df.columns:
        fut_fee_rate = pl.when(pl.col(value_col) > 0) \
            .then(FUT_FEE_PER_LOT * pl.col(lots_col) * 2 / pl.col(value_col)) \
            .otherwise(0.0)
        fee = fee + fut_fee_rate
    # --- 利息（資金成本）---
    if include_interest:
        fee = fee + _carry_rate(side, mode) * (pl.col(days_col) / 365)
    # --- 證交稅（現股賣出一次，當沖/留倉分流）---
    if include_tax and "converged" in df.columns:
        fee = fee + pl.when(pl.col("converged")) \
                      .then(TAX_DAYTRADE).otherwise(TAX_NORMAL)
    # 來回實際毛利率 = 進場價差(first_ret) + 出場價差(反邊 taker，通常為負)。
    # 出場成本由事實欄推：收斂取 exit_ret(收斂那刻反邊價差)、未收斂=0(留倉到結算≈0)。
    #   優先用事實欄 exit_ret+converged 自推（純事實 CSV）；
    #   舊 CSV 已含加工後 exit_cost 欄時直接沿用（向後相容）。
    eff_gross = pl.col(gross_col)
    if "exit_ret" in df.columns and "converged" in df.columns:
        # L4 防呆：收斂列卻 exit_ret=null（出場四價缺）會被 fill_null(0) 靜默當「0 出場成本
        #   ＋當沖稅」＝雙重樂觀。賣向公式不用 exit_fut_bid 故現為零影響，開買向就會踩。
        #   行為不變（仍 fill 0、保守上界），但顯式計數告警、不再無聲。
        n_bad = df.filter(pl.col("converged") & pl.col("exit_ret").is_null()).height
        if n_bad:
            print(f"⚠️ L4：{side} 有 {n_bad} 筆 converged=true 但 exit_ret=null（出場四價缺），"
                  f"出場成本被當 0 計（雙重樂觀，偏樂觀上界）。開買向前必修。")
        exit_cost = pl.when(pl.col("converged")) \
                      .then(pl.col("exit_ret").fill_null(0.0)).otherwise(0.0)
        eff_gross = eff_gross + exit_cost
    elif "exit_cost" in df.columns:
        eff_gross = eff_gross + pl.col("exit_cost")
    return df.with_columns([fee.alias("fee_rate"), eff_gross.alias("eff_gross_ret")]) \
        .with_columns([
            (pl.col("eff_gross_ret") - pl.col("fee_rate")).alias("net_ret"),
            # 金額版：毛獲利(進+出兩刀價差)、總費用、淨獲利（元）= 部位 × 各自率
            (pl.col(value_col) * pl.col("eff_gross_ret")).alias("gross_pnl"),
            (pl.col(value_col) * pl.col("fee_rate")).alias("fee_amount"),
        ]).with_columns(
        (pl.col(value_col) * pl.col("net_ret")).alias("potential_pnl")
    )
