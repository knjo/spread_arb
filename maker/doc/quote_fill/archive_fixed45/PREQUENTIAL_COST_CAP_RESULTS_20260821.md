# AB1/2 user-cost＋combined-cap 補充結果（2026-08-21）

## 結論

這是獨立的 analysis-only 補充包，不修改既有 challenger formal root。它完成兩件事：以四個實際進出場腿價逐筆套用使用者指定 fee/tax；以及同時套用 portfolio one-way notional cap 與每個 `ValueCode` 30% cap。

> **Universe NO_GO：**來源 45 檔是以 May–Aug 2026 已實現流動性挑出的事後研究 cohort，包含 target-day 與後續窗口資訊。本文所有結果只在該固定 cohort 內成立；`universe_selection_d_safe_go=false`、`production_strategy_go=false`。

正式補充輸出：[`prequential_challenger_ab12_cost_caps_20260821_v1`](../../data/walkforward/prequential_challenger_ab12_cost_caps_20260821_v1)

- `complete.json` SHA-256：`5f139cbdc243225eebd4175fc5e9fcbb934e5d897d58362a07e202ceeee4cc3a`
- marker payload SHA-256：`3b6119bcff04d168052be189d6dbc22ac11f22d0f30196eb59bca5ccbd02ce83`
- source module SHA-256：`72eed33f90df182529ba861aebbd86d839436c92a5d63d963ab1af1c64d52f0f`
- 重新讀取 1,096 個 exact-price source artifacts 並 source-rebuild 驗證通過；build／verify 各約 6.4 秒，peak RSS 472／452 MiB，swap 0

## 交易成本

每個 physical path 是一口股票期貨對兩張現貨。依使用者提供的費率：

- 現貨買、賣手續費各 `14.25 bp × 0.12 = 1.71 bp`，各自乘實際買／賣成交金額；
- 現貨賣出稅一般 30 bp；`completed_same_day=true` 時假設符合當沖資格，使用 15 bp；
- 期貨買、賣交易稅各 0.2 bp，各自乘實際期貨腿成交金額；
- 期貨買、賣手續費各 TWD 20，因此完整 round trip 固定 TWD 40。

若四個腿價相同，變動成本參考值是當沖 18.82 bp、非當沖 33.82 bp，再加 TWD 40；正式欄位不是用這個近似值，而是以實際 `entry_spot/future_price`、`exit_spot/future_price` 逐腿精算。未建模最低手續費或逐筆貨幣取整。

|Cohort|Paths|Gross TWD|Fee/tax TWD|Net TWD|Weighted gross bp|Weighted cost bp|Weighted net bp|
|:---|---:|---:|---:|---:|---:|---:|---:|
|全部 completed|2,411|4,894,600|2,281,851.17|2,612,748.83|46.202|21.539|24.663|
|Same day|2,032|4,260,550|1,830,456.30|2,430,093.70|45.679|19.625|26.054|
|Overnight|379|634,050|451,394.86|182,655.14|50.051|35.633|14.419|
|Unresolved|1,261|null|null|null|null|null|null|

正的 completed-only net 仍不是策略淨利：1,261／3,672（34.34%）建倉 paths 沒有 terminal cashflow，保持 null，未補 0。

## 1,000／2,000／3,000 萬 combined cap

准入依 `(Date, position_established_ns, ValueCode, policy_path_id)` 排序。每筆同時檢查 portfolio cap 與單品 30% cap；已完成且 exit time 嚴格早於下一筆 entry 才釋放，timestamp 相同時先處理 entry。Accepted unknown／censored 保守地在整個 horizon 持續占額度。

|Portfolio / 單品 cap|接受 / 3,672|Completed / unresolved|Peak portfolio|Peak single product|Completed gross|Fee/tax|Completed net|
|:---|---:|---:|---:|---:|---:|---:|---:|
|10M / 3M|68（1.852%）|41 / 27|9,995,500|1,811,000|38,750|20,664.00|18,086.00|
|20M / 6M|126（3.431%）|73 / 53|19,989,000|5,446,000|119,450|59,689.91|59,760.09|
|30M / 9M|211（5.746%）|127 / 84|29,997,900|8,888,000|273,650|118,268.50|155,381.50|

拒絕原因：10M 是 portfolio-only 3,604；20M 是 portfolio-only 3,545、product-only 1；30M 是 portfolio-only 3,458、both 3。三個版本皆逐事件驗證沒有突破兩個 cap。

接受筆數並非嚴格線性。若以 10M 的 68 筆線性外推，20M 應為 136、實際 126（0.926×）；30M 應為 204、實際 211（1.034×）。差異來自離散票面、各 path 平倉時間不同、商品集中度與 unknown 永久占額；completed-only net 更不會線性，不能拿來外推 capital return。

## 名詞與目前 gate

- **Terminal cashflow complete**：每個已建倉 position 的兩腿都已有實際平倉／正式 terminal mark。現在只有 2,411／3,672 point-identified，因此仍是 NO_GO。
- **D-safe EV action**：D 日決策只可使用在 D 前已可見的 net labels，且要處理 pending／censored、成本、support與 one-sided LCB；LCB 通過才可選 action。目前沒有符合條件的 action。
- **Best q**：目前 q95 只是舊 completed-only ranking 產生的 diagnostic challenger，不是正式最佳 q。正式比較必須先修正 universe leakage、補 terminal treatment，再讓 q50／q80／q95 在相同 joint-volume、成本與盲測條件競爭。
- **Production strategy**：須固定 D-safe universe、q／route、submit／replace／cancel、ACK與 late-fill、50 ms hedge、跨日 inventory、combined caps及 forced-exit 規則。當前 `production_strategy_go=false`。

## Artifact

|檔案|內容|
|:---|:---|
|`price_source_inventory.parquet`|1,096 個實際腿價來源及 partition/artifact hashes|
|`path_transaction_costs.parquet`|3,672 paths；completed 有逐腿成本與 net，unresolved 保持 null|
|`transaction_cost_summary.parquet`|全部／same-day／overnight／unresolved 四列|
|`combined_position_limit_events.parquet`|3 scenarios × 3,672 candidates＝11,016 列逐筆准入與前後餘額|
|`combined_position_limit_sweep.parquet`|三個 combined-cap 摘要及線性診斷|

這裡的 chronological admission 有 FIFO 式先後順序，但不是共享 exit volume 下的 FIFO inventory matcher；`joint_volume_allocated=false`，因此仍不可升格為可部署回測。
