# S1 暫停與交接摘要（2026-09-01）

## 現在狀態

- S1 已暫停；目前沒有 `s1_production_runner`、相關測試或背景 replay 程序。
- 成本化 S1 實作已提交於 `944c0ac`（`feat(maker): rebuild S1 as cost-aware executable screen`）。
- 尚未開始正式 71 日 × 7 policy replay，也沒有可發布的策略報酬、排名或 S2 shortlist。

## 已完成

- 七組 cost-aware policy、交易成本 gate、20M 全域／10M 單商品 cap、B6 hedge 定價、凍結 exit 目標、共同時點未平倉估值與 publication gate 已實作。
- Spot 與 makerFill 的來源可依 storage contract 轉到 SSD2；個股期貨維持 NAS 並禁止用 TXF 替代。
- 全套 S1 測試 359 項、路徑 contract 測試 19 項通過；Ruff、compileall 與 diff check 通過。
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

## 恢復前唯一必要處理

先針對 `position=2354` 重播事件鏈，判斷 `exit_rollback_failed_unresolved` 是狀態機錯誤，或是實際可能發生的 hedge／rollback 失敗。接著必須把「單腿失敗後如何在當日確定收斂」寫成明確規則並加 regression test。單 partition 通過後，才適合啟動完整 497 partitions replay。
