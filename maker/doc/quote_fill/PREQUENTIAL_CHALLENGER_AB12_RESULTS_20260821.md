# AB1/2 prequential challenger：60 日帳本結果（2026-08-21）

## 結論

> **Universe leakage caveat：**本文結果只對事後凍結的 `first_wave_45` 研究 cohort 成立。這 45 檔的 membership 使用 May–Aug 2026 已實現流動性（包含 target-day 與後續窗口）挑選，因此不是 D-safe universe，也不是可部署 allowlist。下文的 prequential／D−1 GO 只證明「固定 cohort 內」的 policy lineage；實盤 D 日商品資格仍必須只用 `<D` 資訊與 runtime daily liquidity gate。

這份帳本回答的是：「若每天 `D` 只用當時已可見的歷史 labels 排名一個 diagnostic challenger，再把它套到 `D` 當天已凍結的 A/B1–2 掛單，會看到多少掛單、成交、留倉與已完成損益？」

60 個 decision sessions 中，首日沒有足夠歷史；其餘 59 日的 challenger 全是 **q95 + Lower + future-bid/spot-taker exit**。其中 entry `spot_bid_future_taker` 出現 50 日，`future_ask_spot_taker` 出現 9 日。這代表 q95 是目前最值得拿到「下一段未看資料」做 shadow/paper test 的候選，不代表 q95 已被選成可交易策略：正式欄位仍是 `best_q_selection_go=false`、`pathwise_ev_ready=false`、`production_strategy_go=false`。

正式輸出：[`prequential_challenger_ab12_60d_20260821`](../../data/walkforward/prequential_challenger_ab12_60d_20260821)

- `complete.json` SHA-256：`58b449928517e9307fe350e3f8b110ae4f97a431a01a3d840827af55e38d5b9c`
- marker payload SHA-256：`f30e4f878c72ca2c266e472045b3f1248f6fad7aa96de702379beefc2ad42215`
- ledger source SHA-256：`b043544c6dbb811da6a87b7593bb7d9938f23625c946a44e64a24e166a877966`
- 正式 build peak RSS 約 1.32 GiB；獨立 source-rebuild verifier exit 0、peak RSS 約 1.26 GiB、swap 0

## 掛單、成交與 50 ms hedge

59 個有 challenger 的 target days 共 302,582 張 route-valid、去重後的 AB1/2 physical quotes；沒有 alias excess。每張 quote 是獨立 counterfactual execution label，尚未做跨商品／跨策略共同成交量配置。

|指標|結果|
|:---|---:|
|每日 physical quotes p50 / p90 / max|4,901 / 7,272 / 10,391|
|Full fills|3,679（1.2159%）|
|Partial fills|119（0.0393%）|
|No fills|298,784（98.7448%）|
|需要撤單的 quotes|298,903（98.7841%）|
|Full-fill 後 50 ms hedge 可成交|3,672 / 3,679（99.81%）|
|Full-fill queue wait p50 / p90|6.22 s / 113.73 s|
|50 ms hedge signed slippage mean / p90|9.48 bp / 26.46 bp|
|每日 full fills p50 / p90 / max|55 / 119 / 211|

分 rank 的差異如下。`cancel %` 是 nominal strategy-stop 下需要送 cancel 的比例，不是已觀察到交易所 cancel ACK 的比例。

|Route / rank|Quotes|Full fill %|Partial %|Cancel %|Fill wait p50 / p90|50 ms slip mean / p90|
|:---|---:|---:|---:|---:|---:|---:|
|Future ASK1|11,927|2.3225|0|97.6775|5.69 / 93.61 s|9.91 / 28.09 bp|
|Future ASK2|29,871|1.0244|0|98.9756|31.73 / 377.78 s|10.55 / 26.11 bp|
|Spot BID1|69,593|2.4442|0.1293|97.5558|2.89 / 39.85 s|5.99 / 19.27 bp|
|Spot BID2|191,191|0.7296|0.0152|99.2704|11.00 / 215.52 s|13.40 / 32.82 bp|

這也支持先研究 A/B1–2：第一檔的成交與 hedge 品質明顯優於第二檔；另一本獨立五日 ASK1–5 診斷則顯示，ASK3–5 的 full-fill rate 相對 ASK1–2 在 q50/q80/q95 分別再低 36.4% / 44.1% / 50.8%。

## 建倉、平倉與已完成損益

3,679 次 full fills 中，3,672 次完成 50 ms hedge，視為 established positions。後續對接 same-day 與 cross-session frozen Lower exit path：

|Terminal 狀態|筆數|占 established|
|:---|---:|---:|
|Completed|2,411|65.659%|
|其中 same-day|2,032|55.338%|
|其中 overnight|379|10.321%|
|Censored|22|0.599%|
|Unknown / still open|1,239|33.742%|

只有 2,411 筆 completed paths 有 point cashflow。它們的 gross 合計為 **+4,894,600 TWD**，completed-only gross mean 為 **46.81 bp**、p50 為 **47.34 bp**。使用 same-day 19 bp／overnight 34 bp 的統一敏感度後，completed-only 合計為 **+2,691,736 TWD**，mean 為 **25.45 bp**。

上述數字不能稱為策略 EV 或回測淨利：1,261 筆 censored／unknown cashflow保持 null，沒有補 0；19/34 bp 不是完整且 source-bound 的 fee/tax/financing/cancel/emergency 成本；partial inventory也尚未納入。若只為看分母敏感度，把 unresolved 當 0，則 19/34 結果是每張 submitted quote **+0.203 bp**、每個 established position **+16.71 bp**；這兩欄明示為 diagnostic-only，不是 lower bound。

`daily_realized_cashflows.parquet` 有 62 個 terminal dates，completed-only 19/34 曲線終值同為 +2,691,736 TWD，期間最低 drawdown 僅 -168.56 TWD。這條曲線只把已完成路徑在 terminal date 入帳，忽略尚未定價的部位，也沒有資金與共同成交量配置，因此**不是 strategy equity curve**，不能拿該 drawdown 當實際風險。

另一本獨立的 [current-ladder D+1 benchmark](CURRENT_LADDER_D1_OVERNIGHT.md) 顯示，完整 strict-carry diagnostic cohort 中 95.23% 的 physical dependencies 可在精確 D+1 找到 fresh≤1s 的 joint taker/taker 價格，但已定價樣本 gross mean 是 -11.41 bp。它與本帳本的 q95 selected cohort／sampling unit 不同，不能拿來回填這裡的 1,239 筆 unknown、也不能與 completed-only 損益相加；它只說明強制 D+1 平倉較像庫存止損／解套 benchmark，而不是隔夜 alpha。

![AB1/2 completed-only累積損益與每日留倉；非strategy equity](assets/prequential_challenger_ab12_completed_only_20260821.png)

## 每日新倉與留倉需求

Notional 是 one-way spot-leg entry-price notional；不是兩腿 gross exposure、EOD mark、期貨保證金或實際資本需求。

|指標|p50|p90|max|
|:---|---:|---:|---:|
|每日 established positions|55|119|211|
|每日新倉 one-way notional|20.12 M|48.81 M|123.52 M TWD|
|EOD outstanding positions|24|51|150|
|EOD outstanding one-way notional|8.33 M|19.55 M|43.08 M TWD|

EOD outstanding 的 mean 是 29.76 positions／10.63 M TWD one-way entry notional。這是未加 joint-volume constraint 的上界式需求估計。119 次 partial fills 的殘餘 inventory 未進 outstanding，所以又不是完整的最壞情境資本數字。

### 與 `src/research/backTest.py` 的口徑對照

實際檔名是 case-sensitive 的 `backTest.py`。本帳本目前只與其中部分概念近似對應：

|`backTest.py` 指標|本帳本可用欄位|限制|
|:---|:---|:---|
|`CumulativeEntryNotional`|`new_entry_one_way_notional_twd`|只有現貨腿進場價 notional，不是資本／保證金／兩腿 gross|
|`unCoverPosition`|`outstanding_eod_positions`|paired positions 數，不是 signed 淨張數；未含 partial|
|`Overnight_Market_Value`|沒有等價欄；只能另列 entry-price outstanding notional|沒有 EOD mark|
|`PnL` / `CumPnL`|completed gross／19-34 bp sensitivity|只有 completed-only，不能當 daily strategy NAV|
|`Close_PnL` / `Overnight_PnL`|same-day／overnight completed 分組|不是同一部位兩種出口的完整反事實比較|
|`TotalVolume` / turnover|尚無|未算完整四腿成交額，也未共同配置市場 volume|
|MDD / Sharpe / annual return / Calmar|必須 suppress|沒有完整 daily marked NAV|
|`trade_volume_limit_multiplier`|尚無|independent labels 沒有共同消耗真實逐筆成交量|

Gross path 已使用實際 entry／50 ms hedge／exit prices，因此不能把 latency slippage再扣一次；但 `backTest.py` 式逐腿 fee/tax、financing、cancel與emergency cost仍需另建正式 cost producer。

## 部位上限敏感度

Cap sweep 依 `position_established_ns` 先進先出；unknown/censored 不假設能釋放容量，因此會一直占用 cap，是偏保守的 analysis-only 模型。以下是幾個代表設定：

|Cap|接受 established|接受率|Accepted completed|Accepted unknown/open|19/34 completed-only TWD|
|:---|---:|---:|---:|---:|---:|
|Portfolio 25 positions|43|1.17%|18|25|+11,867|
|Portfolio 50 positions|121|3.30%|71|50|+79,411|
|Portfolio 100 positions|263|7.16%|163|100|+276,810|
|Portfolio 250 positions|684|18.63%|434|250|+634,902|
|Portfolio 10 M notional|68|1.85%|41|27|+19,626|
|Portfolio 25 M notional|170|4.63%|105|65|+90,168|
|Portfolio 50 M notional|402|10.95%|249|153|+338,967|
|Portfolio 100 M notional|902|24.56%|561|341|+703,627|
|Per-product 10 M notional|2,417|65.82%|1,515|880|+1,190,337|
|Per-product 25 M notional|3,060|83.33%|1,960|1,078|+1,898,008|

所有 cap 的 `position_limit_sweep_production_ready=false`。高比例 unknown 會讓簡單 portfolio cap 很快被鎖死；這不是 cap 應該放寬的證據，而是 terminal labels／live cancel 和 actual exit 狀態必須先補齊。

## 下一段 unseen shadow test 應怎麼看

先固定一個不再重選參數的 challenger：q95、AB1/2 only、Lower exit；每日 `D` 的掛價只能使用 `D` 前已發布的 boundary/decision。Shadow logger 至少要逐 order 留下 submit、replace、cancel request、cancel ACK、late fill/cancel race、queue age、full/partial fill、fill+50 ms hedge book、實際四腿 cashflow與 EOD outstanding。

每天評估表至少分成四段：

1. Execution：unique quotes、full/partial/no-fill、cancel ACK、late fill、queue wait、50 ms hedge slippage。
2. Turnover：new positions／notional、completed same-day／overnight、EOD outstanding count／notional、holding sessions。
3. P&L：只對 point-identified terminal paths算 gross與逐項 costs；unknown維持 null並另列比例。
4. Risk／capacity：按日與按商品的 peak positions/notional、cap rejections、partial inventory、共同可成交量與 completed-only curve之外的真正 marked equity/drawdown。

至少先累積一段完全未參與 q95 選擇的資料，再與事先固定的 q50、q80 benchmark 同時 shadow；不能每晚看結果後換 q。Production gate 要等實際 cancel ACK、完整成本、partial inventory、terminal cashflow與 joint-volume allocation 都可稽核後才重開。

## Artifact inventory

|Artifact|用途|
|:---|:---|
|`selected_actions.parquet`|302,582 張 target-day AB1/2 execution labels|
|`selected_policy_paths.parquet`|3,672 個已 hedge 建倉後的 frozen exit paths|
|`daily_cohort_ledger.parquet`|60 個 decision-day cohorts|
|`daily_realized_cashflows.parquet`|62 個 terminal-day completed-only cashflows|
|`daily_outstanding.parquet`|62 日留倉 count/notional|
|`challenger_selection_counts.parquet`|兩個 q95 entry-route challenger 的出現次數|
|`entry_rank_summary.parquet`|ASK1/2、BID1/2 execution品質|
|`position_limit_sweep.parquet`|13 個 analysis-only caps|
|`overall_summary.parquet`|整體成交、部位、損益與 readiness gate|
