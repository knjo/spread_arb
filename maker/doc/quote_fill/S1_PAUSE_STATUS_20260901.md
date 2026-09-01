# S1 暫停與交接摘要（2026-09-01，歷史快照）

> 本文保存commit `944c0ac`第一次smoke失敗與當時清理狀態。研究其後已恢復；2354事件鏈已完成診斷並由exit risk guard承接，但新clean-source smoke仍待執行。本文不是目前的「暫停」指令。

## 現在狀態

- 當時S1執行程序已停止；沒有遺留的`s1_production_runner`、相關測試或背景replay程序。
- 成本化 S1 實作已提交於 `944c0ac`（`feat(maker): rebuild S1 as cost-aware executable screen`）。
- 尚未開始正式 71 日 × 7 policy replay，也沒有可發布的策略報酬、排名或 S2 shortlist。

## 已完成

- 七組 cost-aware policy、交易成本 gate、20M 全域／10M 單商品 cap、B6 hedge 定價、凍結 exit 目標、共同時點未平倉估值與 publication gate 已實作。
- Spot 與 makerFill 的來源可依 storage contract 轉到 SSD2；個股期貨維持 NAS 並禁止用 TXF 替代。
- 當時全套 S1 測試 359 項、路徑 contract 測試 19 項通過；Ruff、compileall 與 diff check 通過。
- nested repository 在啟動 smoke 前為乾淨狀態。

## 單 partition smoke 結果

smoke 嘗試執行 `20260505` 的第一個 partition，但在 production validation 階段 fail-closed：

```text
naked unresolved state:
position=2354
reason=exit_rollback_failed_unresolved
```

這不是缺檔或 SSD2 路徑問題。該次 manifest 已成功讀取並雜湊：

- Spot raw：`/media/kevin/SSD2/Data/tickData/20260505_StockTick.parquet`
- makerFill：`/media/kevin/SSD2/Data/makerFill/20260505_makerFill.parquet`
- 個股期貨：`/mnt/NAS/Parquet/Ticks/2026/05/05/stock_futures.parquet`

目前規格是 `fail_closed_no_cross_day_carry`，因此只要出現一筆未解決的單腿部位，整個 production run 就必須停止。該 partial run 沒有完成 partition、`results.json`、正式報告或 `complete.json`，不能據此判斷策略是否獲利。

## 已清理

- 刪除 `/tmp/s1-cost-aware-smoke-944c0ac` 的 84K partial bundle。
- 刪除本輪測試／執行生成的 6 個 `__pycache__`，約 6.4 MiB；它們都可自動重建。
- 未刪除 canonical research data、程式碼、正式輸出或外層 repository 既有的未提交修改。

## 後續診斷與修正

精確重播顯示這不是尾盤deadline壓縮。2026-05-05的2354／GCFE6在約09:52同步漲停無賣盤：Future Ask於09:52:29.067229先消失；Spot Ask約09:52:29.138670消失，同一recv timestamp有58.5元、61與46 lots兩筆實體成交。舊normal-exit target雖計算Future executable VWAP，卻未以Future完整可買depth關閉gate，且exit wake只監聽Spot；因此Future已先失去Ask，被動Spot Ask仍可成交。後續Future hedge與Spot rollback都沒有合法Ask，形成真實可達的`exit_rollback_failed_unresolved`。

承接修正有三層：

1. Passive Spot exit須同時通過當下Future buy全量可執行gate；Spot與Future book change都會喚醒重驗。Gate只決定固定價掛單是否可工作，不會重定價，也不取代fill後B6。
2. 13:19:45固定開始drain所有passive exit；到13:19:49.950最晚安全成交barrier仍有new／working／cancel未terminal時，partition直接fail closed。Actual cancel effect前或同cursor的fill仍照真實phase先成交。
3. 任何最後仍是裸腿的position繼續保留exposure與capacity並封鎖排名；不能用common-horizon mark、最後合法book或expiry basis=0補平。

這些guard修掉2354揭露的可因果避免風險窗，但不宣稱同步book消失、cancel latency或實盤reject下絕不會裸腿。下一步是用新clean source commit重跑單partition smoke；通過後才啟動完整497 partitions。
