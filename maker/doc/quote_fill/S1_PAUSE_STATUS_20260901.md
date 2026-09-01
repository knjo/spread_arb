# S1 暫停與交接摘要（2026-09-01，歷史快照）

> 本文保存commit `944c0ac`第一次smoke失敗與當時清理狀態。研究從未因本文暫停；停止的是該次失敗程序。
> 後續`42b689a`／`cd2b87c`已承接2354事件鏈，`cd2b87c`的2026-05-05 control smoke成功且
> `unresolved=0`。v3正式bundle目前為1/497；第二個generic-clock partition沒有complete marker。
> v4 precommit integration、post-commit durable rerun與production進度只在[`S1_COST_AWARE_IMPLEMENTATION_20260831.md`](S1_COST_AWARE_IMPLEMENTATION_20260831.md)維護；本文不是目前的
> 「暫停」指令，也不是目前head的執行狀態。

## 當時狀態（commit `944c0ac`）

- 當時S1執行程序已停止；沒有遺留的`s1_production_runner`、相關測試或背景replay程序。
- 成本化 S1 實作已提交於 `944c0ac`（`feat(maker): rebuild S1 as cost-aware executable screen`）。
- 尚未開始正式 71 日 × 7 policy replay，也沒有可發布的策略報酬、排名或 S2 shortlist。

## 已完成

- 七組 cost-aware policy、交易成本 gate、20M 全域／10M 單商品 cap、B6 hedge 定價、凍結 exit 目標、共同時點未平倉估值與 publication gate 已實作。
- Spot 與 makerFill 的來源可依 storage contract 轉到 SSD2；個股期貨維持 NAS 並禁止用 TXF 替代。
- 後續headroom guard修正的全套S1測試365項、路徑contract測試19項通過；這是`cd2b87c`前後的後記，不是`944c0ac`當時已完成項。
- nested repository 在當次smoke前為乾淨狀態。

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

精確重播顯示這不是尾盤deadline壓縮，也不是送單額度造成。2026-05-05的2354／GCFE6，Spot exit maker在台北時間
09:44:14.660494（raw UTC 01:44）以57.2成交。Fill前最後一個causal Future book是09:44:14.656167780，Ask 57.6，
只早約4.326714 ms；Future reference是53.4，strict upper為`price < 57.672`，所以57.6是最後一個合法tick，當下雖有
足量depth，卻已沒有再承受一個向上tick的合法空間。Fill後09:44:14.661496701，Future Ask才轉為57.7並因超出band而
不合法。Cancel在fill+1 ns才送出，quota不是延後原因；既有phase因此正確保留該fill，接著+50 ms Future hedge與後續
Spot rollback都timeout，形成`exit_rollback_failed_unresolved`。

先前的初步事件歸因不正確；以上述逐cursor重播為準。這個事件也證明只要求fill前當下Future全量可執行仍不夠：
最後合法tick可以在數毫秒後越界，而同cursor／較晚cancel不能回頭刪除已發生的maker fill。

承接修正有三層：

1. Passive Spot exit除須通過當下Future buy全量可執行gate，最差swept ask上方還須保留至少一個合法Future tick；Spot與Future book change都會喚醒重驗。Gate只決定固定價掛單是否可工作，不會重定價，也不取代fill後B6。
2. 13:19:45固定開始drain所有passive exit；到13:19:49.950最晚安全成交barrier仍有new／working／cancel未terminal時，partition直接fail closed。Actual cancel effect前或同cursor的fill仍照真實phase先成交。
3. 任何最後仍是裸腿的position繼續保留exposure與capacity並封鎖排名；不能用common-horizon mark、最後合法book或expiry basis=0補平。

這些guard只修正2354揭露的可辨識上緣邊界風險，不宣稱同步book消失、cancel latency或實盤reject下絕不會裸腿；
任何unresolved仍維持fail closed。這份歷史快照當時的下一步是用新clean source commit重跑單partition smoke；
該步其後已由`cd2b87c`完成。後續v4 clock、differential、preflight與執行進度不在此歷史快照更新，統一以
[`S1_COST_AWARE_IMPLEMENTATION_20260831.md`](S1_COST_AWARE_IMPLEMENTATION_20260831.md)為準。
