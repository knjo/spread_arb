# S1 Cost-aware 重建：Implementation／Preflight（2026-08-31）

## 結論與狀態

舊 S1 partial 因 actual-send 前沒有經濟 eligibility，已停止且清除。新的 S1 不再只比掛單成交與同日完成；它先用
causal executable prices與使用者成本決定「這次是否值得送」，再把真正送出的候選放進共同 chronological 20M cap
replay。本文只記錄已凍結的實作契約，**不是71日研究結果，也不是 deployment GO**。

截至本文件更新，scenario／成本／absolute exit／B6／capacity／path／artifact／publication、8/13 common-horizon
open valuation、exit pre-fill headroom guard與持久化verification receipt程式均已完成，完整S1回歸365項及path contract 19項通過；clean-source smoke、497 partitions與獨立
input-content verify仍待完成。正式 report完成前不得引用
champion、Pareto或S2 shortlist。

## Frozen 七組

| scenario | upper／lower | cost admission | shortlist |
|---|---|---|---|
| `ctrl_q95_C0_ungated` | q95／C0 center | ungated control | 否 |
| `q95_C0_sd_f5` | q95／C0 | same-day modeled margin > 5 bp | 是 |
| `q95_C0_on_f0` | q95／C0 | overnight modeled margin > 0 bp | 是 |
| `q95_C2_sd_f5` | q95／C2 reach80 | same-day modeled margin > 5 bp | 是 |
| `q80_C0_sd_f5` | q80／C0 | same-day modeled margin > 5 bp | 是 |
| `q50_C3_sd_f5` | q50／C3 reach50 | same-day modeled margin > 5 bp | 是 |
| `fixed20_sym20_on_f0` | symmetric 20 bp | overnight modeled margin > 0 bp | 是 |

七組都保留共同 15,638 product-days × 4 TOD＝62,552 cells。C2／C3 lookup缺值不改用C0、不刪列，固定為
`lookup_supported=false`的no-trade。Economic gate是預先凍結的交易規則；研究本身仍沒有獲利pass/fail門檻。

## Actual-send economics

送單前以同一 causal cursor 計算：

```text
expected gross
= shares × [(frozen Spot exit target - Spot entry maker target)
            + (Future sell executable VWAP - Future buy executable VWAP)]

selected expected margin
= expected gross
  - Spot entry/exit commission
  - Spot sell tax (same-day 15 bp or overnight 30 bp)
  - Future entry/exit tax
  - Future entry/exit commission
```

已建模成本固定為 Spot commission每邊1.71 bp、Future tax每邊0.2 bp、Future commission每邊TWD 20。
Financing、borrow、futures margin opportunity cost與live reject／latency仍不可得，必須以unavailable reason揭露，不能填0。
所有 decision與actual-send estimate另外寫入可重播audit stream；control也定價，但不阻擋送單。

## 執行與出場語意

- Entry是 `Spot Bid maker → Future sell taker`；legacy makerFill只提供approximate full-fill screen。
- `actual_new_send_time`同時凍結entry Future Bid／Ask、lower provenance、absolute Spot Ask exit price與tick。
- Normal exit是 `Spot Ask maker → Future buy taker`。Absolute Spot Ask price／tick維持凍結，不因後續行情重定價；但被動單只有在Spot maker book合法、Future buy L1-L5足以完整買足該position，且最差swept ask上方仍保留至少一個嚴格位於Future合法價格band內的tick時才可工作。Spot或Future任一book變化都會喚醒重驗；Future gate關閉時撤回desired／送cancel，恢復後仍只可回原凍結價。
- 這個pre-fill hedgeability／headroom gate不是流動性預留，也不取代B6。它只修正目前可辨識的上緣邊界風險，不保證零leg risk。若Spot maker在actual cancel effect前仍真實成交，fill依舊成立，並從fill+50 ms獨立判定Future hedge、最多retry 5秒，失敗再rollback。
- S1 carry route在13:19:45固定開始撤除所有passive exit desired；13:19:49.950的最晚安全成交barrier要求所有passive lifecycle已terminal，否則partition fail closed。Actual cancel effect前或同cursor的fill仍先於cancel。這不是S4 taker+taker hard flatten，也不能保證市場同步消失時永無裸腿。
- Hedge先在trigger+50 ms判定；不可執行或venue額度不足時，往後最多5秒找第一個合法足量且可送cursor。Timeout後走共同rollback。
- 20M global／10M product reservation在entry new actual-send前成立；maker fill只把reservation轉為exposure，完整exit hedge後才釋放。
- Expiry paired residual使用使用者指定的spot-close／spot-close、basis=0 accounting convention；不是execution fill或same-day completion。
- 任何最終`entry_hedge_timeout_unresolved`／`exit_rollback_failed_unresolved`都保留裸腿與committed capacity，並封鎖economic ranking；不得用8/13 common-horizon mark或expiry basis=0洗平。

報表另以`exit_desired_withdrawal_reason_counts`揭露guard影響；上緣buffer關閉固定記為
`gate:future_upper_band_headroom_lt_1_tick`，不得併入一般無深度或`safety_cutoff`。

## 8/13共同 ranking horizon

71個development entry sessions到2026-08-13為止；8/14起仍是protected forward，不拿來替development補runoff。
完整71日facts重播後仍open的paired position，使用8/13 13:20共同 causal L1–L5 hypothetical liquidation：long Spot掃Bid、
short Future掃Ask，扣已發生actual cost及依current lot acquisition date計算的remaining exit cost。同一scenario／商品的多筆
position先聚合數量再掃一次book，不得各自重複使用同一份顯示深度；明確clear／trial後不沿用更早book。

```text
economic ranking net = terminal realized net + open net mark
```

Mark不生成request／fill、不算completion、不釋放capacity，且在report中不得改稱realized PnL。任何open position無合法足量
book時，描述統計可發布，但economic ranking／Pareto／S2 shortlist全部withhold。

## Input與重現

- Spot tick／makerFill由top-level `config/pipeline.yaml`及`src/pipeline_storage.py`解析，目前canonical在
  `/media/kevin/SSD2/Data`；required mount不存在即fail closed。
- 個股期raw只接受`/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet`；TXF資料夾明確禁止。
- 每日manifest保存實際path、bytes、mtime與SHA-256；prepared loader必須逐role等於manifest，load後再驗hash未變。
- 每個policy/date partition使用deterministic gzip JSONL＋canonical JSON、hash chain、compact capacity checkpoint與exact
  identity registry。完整run還要有`complete.json`及獨立`verification.json`。
- `verification.json`只在497 partitions、input hashes、accounting/capacity/publication與最終產物全部deep verify成功後原子寫入；
  綁定complete／run-config／results／report hash、bundle與verifier source commits、partition count與verifier version。Verifier要求
  `maker/src`為同一clean commit；deep verify只接受已存在且bytes完全相同的artifact，任何中途缺檔都失敗且不得重建／留下receipt。

## 開跑條件

1. [完成] Common-horizon valuation、report與publication ranking接線通過focused／full S1 tests。
2. Ruff、compile與`git diff --check`通過並建立clean source commit。
3. 新output root先跑一個partition smoke，檢查cost decomposition、funnel、resume與input provenance。
4. 才執行71×7＝497 partitions；完成後另跑`verify --verify-inputs`，最後發布正式Markdown與bundle hash。
