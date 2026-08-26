# Maker Basis 研究線

本目錄研究期貨／現貨 basis 的日內 maker 策略：在高 basis 以一腿 maker、另一腿固定約 50 ms 後 taker 建立多現貨／空期貨部位，
再於 basis 回落時用同樣結構平倉。

目錄固定分為：

- [`doc/`](doc/README.md)：研究規格、資料契約、決策紀錄與結果文件。
- [`src/`](src/README.md)：可重跑的資料處理、模型及回測程式。
- [`data/`](data/README.md)：基礎事實、因果 manifest 與各 run 輸出；不進 Git。

## 現況（2026-08-26）

主線是 **動態商品池因果 pipeline**：月 M 只用完整 M-1 選商品池、日 D 只看 D-1 流動性 gate，再做 1 Hz quote intent →
approximate makerFill → +50 ms 期貨 hedge → 同日／跨日／到期 terminal path → inventory cap 回放。既有 10–50M
輸出是歷史 predecessor；現行 C9 primary 固定 20M global／10M 單商品。
入口與結論見 [`doc/quote_fill/README.md`](doc/quote_fill/README.md) 與
[`doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`](doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md)。
既有 predecessor 的 approximate、uncapped 回放顯示研究 edge 約為正 10 bp，但這不是現行 20M 結論，更不是
production GO。S0 已完成 8 月轉弱歸因：
May～Jul pooled → August 的 q95 excursion touch rate由7.1497%降至4.2923%，碰到後 approximate fill由1.8796%降至
1.2183%，故結論是market／boundary與queue／competition兩者並列；30-session sensitivity沒有消除落差。完整口徑、
hash與限制見[`doc/quote_fill/AUGUST_ATTRIBUTION_20260824.md`](doc/quote_fill/AUGUST_ATTRIBUTION_20260824.md)。

S0.5 已從 anchor 開始完整重作。盤中中心選 `time_ewma_15s`：預先註冊的 30～300 秒 future-center month-equal
MAE 為 8.722 bp；它只用當下以前的合法 book 動態更新，不是把盤前中價差凍住整天。D 日盤前 q lookup 選
`Q2_trail20_date_equal`，每個 prediction 嚴格只用 `<D` 歷史；它的逐日跨商品 Spearman 為 0.626，但同商品跨日
Spearman 只有 0.142，因此可作距離／排序 baseline，不能把 q95 解讀成每天固定 5% 機率。

重建後的 S1 common mother 是 15,638 product-days、71 sessions、244 商品，不用 target touch／fill／PnL 或舊 q95
selector 選樣。七組 policy 的共同 geometry也揭露一項重要修正：原 q-policy negative side是未作post-touch
conditional calibration的C1 wide control，不能直接當正常exit lower。舊C0／C2／C3 convergence又以逐秒移動的
anchor判斷，與A2「submit後鎖定絕對exit價」不一致，現已降級為sensitivity。

Frozen-at-upper-touch v2 已正式發布並通過獨立驗證。q95全mother的C0截至13:20 frozen-mid target reach proxy為92.849–100%；在C2／C3
都有lookup的共同cells上，C0／C2／C3 reach為95.437–100%／77.979–82.596%／52.817–57.534%，未作tick rounding的
nominal同日已知成本後margin p50為+0.827／+3.302／+7.466 bp。更深lower換到較大表面margin，但降低reach proxy；C2／C3 q95 cell coverage
又只有24.241%。若只按full-mother availability與reach proxy，C0是q-policy lower的completion-oriented development default proposal，仍待使用者確認；fixed15–30維持`lower=W`。這不是executable同日完成率。
完整結果、hash與限制見
[`doc/quote_fill/FOUNDATION_SELECTION_S05_REBUILD_20260826.md`](doc/quote_fill/FOUNDATION_SELECTION_S05_REBUILD_20260826.md)；
2026-08-25 的 [`FOUNDATION_REVALIDATION_S05_20260825.md`](doc/quote_fill/FOUNDATION_REVALIDATION_S05_20260825.md)
只保留為 predecessor。

固定 45 檔時代的研究（有 universe leakage）已於 2026-08-24 清理：資料與程式刪除、文件歸檔至
[`doc/quote_fill/archive_fixed45/`](doc/quote_fill/archive_fixed45/)；程式可由本 repo commit `1348576` 撈回。

現行 development baseline 是盤中 `time_ewma_15s`＋D-safe `Q2_trail20_date_equal`；30s anchor與Q2 TOD10只作
diagnostic。取樣仍用 SpreadPairTotalCount epoch 與 1 Hz final-net；entry只掛 A/B1–2；hedge基準為
`fill RecvTime + 50 ms`，當下不可執行才依 B6 往後最多5秒找第一個合法足量 book。

S1 round-trip 開跑前只剩凍結唯一normal-exit lower scheme、unsupported lookup行為與預註冊subgroup。七組仍全跑
共同mother，不加policy-specific mother gate。若採C2→C0，必須建立明示composite policy ID，不能冒稱reach80。
S0是`cap=∞` quote-only診斷；S0.5是
known-cost pre-replay foundation；兩者都不可直接當成20M績效或可部署baseline。

既有 `../taker/` 是獨立的 taker 研究線，本目錄不依賴它。
