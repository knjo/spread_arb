# Maker Basis 研究線

本目錄研究期貨／現貨 basis 的日內 maker 策略：在高 basis 以一腿 maker、另一腿固定約 50 ms 後 taker 建立多現貨／空期貨部位，
再於 basis 回落時用同樣結構平倉。

目錄固定分為：

- [`doc/`](doc/README.md)：研究規格、資料契約、決策紀錄與結果文件。
- [`src/`](src/README.md)：可重跑的資料處理、模型及回測程式。
- [`data/`](data/README.md)：基礎事實、因果 manifest 與各 run 輸出；不進 Git。

## 現況（2026-08-24）

主線是 **動態商品池因果 pipeline**：月 M 只用完整 M-1 選商品池、日 D 只看 D-1 流動性 gate，再做 1 Hz quote intent →
approximate makerFill → +50 ms 期貨 hedge → 同日／跨日／到期 terminal path → 10–50M inventory cap 回放。
入口與結論見 [`doc/quote_fill/README.md`](doc/quote_fill/README.md) 與
[`doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`](doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md)。
目前判斷：研究 edge 為正（uncapped net ≈ 10 bp），但 8 月明顯轉弱，且尚未通過 prospective holdout，不是 production GO。

固定 45 檔時代的研究（有 universe leakage）已於 2026-08-24 清理：資料與程式刪除、文件歸檔至
[`doc/quote_fill/archive_fixed45/`](doc/quote_fill/archive_fixed45/)；程式可由本 repo commit `1348576` 撈回。

已凍結的決策：中價用 causal EWMA120；上下緣用 60-session rolling、正負側分開的 empirical quantile；取樣用
SpreadPairTotalCount epoch 與 1 Hz final-net；只掛 A/B1–2；hedge 基準 `fill RecvTime + 50 ms`。

下一步依已定案的 [`doc/REWORK_PLAN_20260824.md`](doc/REWORK_PLAN_20260824.md) 執行；順序由 S0 的 8 月歸因開始，再跑七組 policy、反向 entry route、exit maker、close policy 與最終 exact 校準。

既有 `../taker/` 是獨立的 taker 研究線，本目錄不依賴它。
