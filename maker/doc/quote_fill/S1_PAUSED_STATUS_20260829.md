# S1 暫停狀態與清理紀錄（2026-08-29）

> 2026-08-31 更新：未完成的 replay bundle 已清除，停止後的最終狀態見
> [`S1_STOP_CLEANUP_20260831.md`](S1_STOP_CLEANUP_20260831.md)。以下保留為 8/29 當下的歷史紀錄。

## 現在狀態

- S1 所有由本次研究啟動的 replay、監控、smoke、測試與子任務均已停止；未處理其他專案的
  `rank-train` 程序。
- 舊七組 replay 在 **77／497 partitions** 停止，涵蓋 2026-05-05～2026-05-19 的 11 個交易日、
  七組 policy；停止時正在計算 2026-05-20／q50，該未完成 partition 沒有發布。
- 停止是人工 `SIGTERM`，service exit code 143；停止前 `NRestarts=0`，沒有 OOM 或自動重啟。
- 正式 `POLICY_COMPARISON_SPOT_BID_20260827.md` 未發布，因為完整 497 partitions 與 final verifier
  都未完成。

當時保留的唯一 partial baseline 是 `s1_spot_bid_joint_20260827_v2`（已於 2026-08-31 清除）：

- 77 個 `complete.json`、784 個檔案，約 4.6 GiB；
- source commit `828005589c3b5c61c53a8fab020869242cd24cbb`；
- run-config SHA-256 `b7707ae1a2f2aff99e87a5d837cca488d1663d1dd534c90fd9faeac85f759e01`；
- 只准稱為 **未作經濟 eligibility 的 partial baseline**，不得用來定稿 policy、挑 S2 或稱為可部署策略。

## 為什麼停止

現行七組 replay 有完整的逐腿實際成本記帳，但 entry eligibility 仍使用 q-independent common mother；
它沒有在實際 new-send 前要求：

```text
預期價差毛利
− 現貨雙邊手續費
− 現貨賣出稅（當沖／隔夜分開）
− 期貨雙邊稅與每邊 TWD 20 手續費
− 額外 adverse／latency 安全邊際
> 0
```

因此舊 run 可以量測「無經濟篩選的七個門檻會怎樣」，但不能回答研究真正要問的：在新的 causal Q 表與
使用者成本下，哪些商品／TOD／entry-lower 組合值得送入 chronological cap replay。

## 11 日方向性快照

下表是停止前由 77 個完整 partitions 做的 interim 加總。它尚未跑 full production verifier，留倉也沒有
mark-to-market，因此只作問題定位，不是正式績效。

| policy | Spot maker fills | 同日正常完成率 | 5/19 paired open | 已終結 net TWD |
|---|---:|---:|---:|---:|
| q50 | 1,081 | 69.66% | 87 | -590,822 |
| q80 | 719 | 66.76% | 63 | -511,669 |
| q95 | 445 | 66.74% | 26 | -333,200 |
| fixed15 | 439 | 42.82% | 69 | -201,828 |
| fixed20 | 310 | 37.42% | 60 | -64,959 |
| fixed25 | 243 | 32.10% | 59 | -96,913 |
| fixed30 | 197 | 28.43% | 55 | -113,581 |

方向性訊號只有兩個：q50 的成交／同日完成較高但損失最大；fixed20 的 partial net 最接近零但仍為負。
七組同日 net 均為負，足以證明後續不能只比較成交與完成率，必須同時報 gross、逐腿成本、net 與 carry
成本風險；不足以證明任何組合最終無利可圖。

## 已清理

- 刪除約 4.6 GiB 的舊 `s1_spot_bid_joint_20260827_v1`。它同樣只有 77 partitions，但綁定已淘汰的
  source commit `47cdb7956a7954ddbb83e754cfa79b05f8155e81`；若要找回只能重算。
- 刪除約 522 MiB 的 `/tmp` supplemental prototype、formal staging、smoke output、UV cache、監控腳本與
  interim 分析腳本。
- 移除尚未完成驗證的 cost-gate 程式草稿與測試，工作樹不保留半套實作。
- 清除 transient systemd unit 的 failed state；目前沒有 S1 相關背景程序。

## 重新開始前只做這四步

1. 凍結最多數個 `entry q × conditional lower × cost horizon × safety floor` policy ID；至少保留一個
   ungated control，但不得讓 control 參與 deployment shortlist。
2. 在 actual new-send cursor 用當下 Spot maker target、Future bid／ask executable VWAP、frozen exit basis
   與使用者成本，產生 same-day／overnight expected margin；unsupported lookup 明記 no-trade。
3. 每組先跑少量日期 smoke，固定報 candidate funnel、fills、同日／跨日／rollback／open、gross、commission、
   tax、net、overnight notional-days、cap displacement 與 margin bucket calibration。
4. smoke 與 verifier 都通過後才重啟 71 日 chronological replay；完整結果完成前不發布排名。
