# Maker Basis 研究線

本目錄研究期貨／現貨 basis 的日內 maker 策略：在高 basis 以一腿 maker、另一腿固定約 50 ms 後 taker 建立多現貨／空期貨部位，
再於 basis 回落時用同樣結構平倉。

目錄固定分為：

- [`doc/`](doc/README.md)：研究規格、資料契約、決策紀錄與結果文件。
- [`src/`](src/README.md)：可重跑的資料處理、模型及回測程式。
- [`data/`](data/README.md)：基礎事實、因果 manifest 與各 run 輸出；不進 Git。

## 現況（2026-08-25）

主線是 **動態商品池因果 pipeline**：月 M 只用完整 M-1 選商品池、日 D 只看 D-1 流動性 gate，再做 1 Hz quote intent →
approximate makerFill → +50 ms 期貨 hedge → 同日／跨日／到期 terminal path → 10–50M inventory cap 回放。
入口與結論見 [`doc/quote_fill/README.md`](doc/quote_fill/README.md) 與
[`doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`](doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md)。
既有因果線顯示研究 edge 為正（uncapped net ≈ 10 bp），但這不是 production GO。S0 已完成 8 月轉弱歸因：
May～Jul pooled → August 的 q95 excursion touch rate由7.1497%降至4.2923%，碰到後 approximate fill由1.8796%降至
1.2183%，故結論是market／boundary與queue／competition兩者並列；30-session sensitivity沒有消除落差。完整口徑、
hash與限制見[`doc/quote_fill/AUGUST_ATTRIBUTION_20260824.md`](doc/quote_fill/AUGUST_ATTRIBUTION_20260824.md)。

S0.5 已完成查表基礎重驗：因果 EWMA30 的 future-center MAE 8.877 bp，優於 incumbent EWMA120 的 9.750 bp；
rolling-60 q 表在六個 q×side cell 的逐日商品 Spearman 全為正、平均 0.615～0.673，但 q95 reach 從 5 月約 8.4%
降至 8 月約 4.0%，absolute level 有 regime lag。EWMA120 incumbent 的 q-independent Spot-Bid broad reference 是
15,935 product-days，舊 q95 selector只重疊3,846。完整結果、canonical hash與限制見
[`doc/quote_fill/FOUNDATION_REVALIDATION_S05_20260825.md`](doc/quote_fill/FOUNDATION_REVALIDATION_S05_20260825.md)。

固定 45 檔時代的研究（有 universe leakage）已於 2026-08-24 清理：資料與程式刪除、文件歸檔至
[`doc/quote_fill/archive_fixed45/`](doc/quote_fill/archive_fixed45/)；程式可由本 repo commit `1348576` 撈回。

Incumbent 決策是中價用 causal EWMA120；S0.5 建議以 EWMA30 升為 development primary、EWMA120 留作 control，
但必須先用 selected-anchor residual 重建 rolling q，不能混表。上下緣 baseline 用 60-session rolling、正負側分開的 empirical quantile；取樣用
SpreadPairTotalCount epoch 與 1 Hz final-net；只掛 A/B1–2；hedge 基準 `fill RecvTime + 50 ms`。

下一步先依 [`doc/REWORK_PLAN_20260824.md`](doc/REWORK_PLAN_20260824.md) 確認 S0.5 handoff：selected anchor、唯一 level challenger與 q-independent cohort，重建 lookup 後才執行 S1 七組 policy × Spot Bid、20M／單檔50% reservation與B6 hedge event loop。S0是`cap=∞` quote-only診斷，S0.5是 known-cost pre-replay screen；兩者都不可直接當成20M績效或可部署baseline。

既有 `../taker/` 是獨立的 taker 研究線，本目錄不依賴它。
