# q95 AB1/2 每秒委託訊息量：legacy coverage 診斷（2026-08-22）

> 更新：完整因果商品池的一秒 spot quote-intent 已完成，正式 spot 數字請以
> `ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md` 為準。本文件保留作 legacy exact-fill
> future hedge batching 與 makerFill coverage 的補充證據，不再用其中 scaling 當
> spot 主結論。

## 結論

這份文件只回答現有 60-session execution facts 能觀察到的訊息量，並用兩種簡單
product-rate scaling 做完整動態池的容量壓力估計。它**不是**完整動態商品池 replay，
也不替固定 45 檔 universe 背書。

- 使用者確認的實務語意是：同一絕對價格只保留舊單；target 往前的新價才新掛；
  target 後退才撤掉更積極的舊價。
- 現行 legacy runner 卻允許新 `SpreadPairTotalCount` epoch 在同價再開一個
  generation。因此直接數 generation 會把研究樣本數誤當 exchange requests。
- 把重疊 generations 折成 `Date × ValueCode × absolute price` working-window 聯集，
  再取每秒 final-net state 後，spot 掛撤需求約減半。
- 在既有資料的盤中觀測值，spot `100 requests/s` 沒有超限；但完整動態池外推的
  尾端介於 `37–101 requests/s`，只能說大致可行、尚不能認證。
- 集中在 cutoff 同一秒撤單明顯不安全：完整動態池外推 p95 約
  `156–174 requests/s`。若 13:00 停止進場，問題只會從 13:20 搬到 13:00，
  必須提早／分散清除 working entry points。
- Future 50 ms hedge 若每個 independent raw fill 各送一張，會超過
  `5 requests/s`；以 `(Date, ValueCode, hedge_decision_time_ns)` 聚合並加總 quantity
  後，既有 60 日觀測最大為 `4 requests/s`。這解決的是 request count，並未證明
  聚合後數量有足夠共同深度。

## 資料範圍與 coverage

Entry source：

```text
maker/data/walkforward/execution_narrow_60d/
  Date=*/ValueCode=*/execution_action_facts.parquet
```

篩選固定為：

```text
route = spot_bid_future_taker
boundary_quantile = 95
target_rank_at_submit in {BID1, BID2}
```

來源涵蓋 2026-05-20 至 2026-08-13 的 60 sessions、45 個 legacy products、
2,687 個 completed execution partitions。因果月選表僅用作交集診斷：

```text
maker/data/walkforward/monthly_product_selector_causal_v2_20260822/
  daily_entry_manifest.csv
```

該 manifest 共 3,886 product-days；在相同 60 個 execution dates 內有 3,418
product-days，但只有 945 個有 completed raw replay，coverage 為 `27.65%`。
其中 927 個 product-days 實際產生 q95 BID1/2 actions。因此下列「causal overlap」
不是完整動態池結果，也不能用來回頭限制商品為 45 檔。

## 兩種計數語意

### 1. Legacy generation-native

- 每筆 q95 BID1/2 action 在 `submit_recv_time_ns` 計一個 spot new request。
- 僅 `cancel_required=true` 的 action，於 `nominal_stop_recv_time_ns` 計一個 spot
  cancel request。
- Full fill 不另計 cancel。
- 這是既有 replay model 內的精確計數，但包含 cross-epoch same-price resubmit，
  不符合現在確認的 production 語意。

### 2. Point-union 1 Hz diagnostic

1. 每個 legacy generation 先形成
   `(submit_recv_time_ns, nominal_stop_recv_time_ns]` target-demand interval。
2. 依 `(Date, ValueCode, target_price_tick)` 合併所有重疊 interval。
3. 將 boundary event 取 `floor(ns / 1 second)`，同秒先做 final-net。
4. active point 由 0 變正數計一個 new-point request；由正數變 0 計一個
   delete-point request。

這個投影會正確消掉同價 cross-epoch 重掛，也會保留往前新價的另一個 point。
但它只代表 target-demand，不是新的 fill replay：它尚未處理 point 成交後何時重掛、
聚合 quantity、own-order queue、joint volume、cancel ACK 或 late fill。因此應視為
符合新語意的快速吞吐診斷，不是 execution truth。

Spot 表假設 production controller 真的在固定 1 Hz dispatch boundary 送 final-net
actions。若券商採 rolling 1-second token bucket，實作仍須用同一個 limiter，不能只在
報表中依 wall-clock second 分桶。

策略本身沒有原生 `replace`：往前是保留舊價並新增另一價，後退是刪除較積極價。
同一 product-second 同時有 new 與 delete 只能稱 mixed point change；如果券商 API
必須將真正 amend 實作為 `cancel + new`，才會消耗兩個 requests。

## Spot maker requests

| 指標 | Legacy fixed-45 | Causal-manifest overlap |
|---|---:|---:|
| source generations / new requests | 310,137 | 173,298 |
| generation-native cancel requests | 306,599 | 171,243 |
| point-union 1 Hz new points | 155,915 | 86,593 |
| point-union 1 Hz delete points | 155,915 | 86,593 |
| point projection 對 native requests 減量 | 49.44% | 49.73% |
| mixed product-seconds | 2,752 | 1,564 |
| 最多可配對的同秒 delete+new | 2,869 | 1,645 |

Point-union 的**盤中、排除 session cutoff**組合 request load：

| 指標 | Legacy fixed-45 | Causal-manifest overlap |
|---|---:|---:|
| event-second p50 / p95 / p99 | 1 / 3 / 6 | 1 / 3 / 5 |
| event-second max | 25 | 21 |
| 每日盤中 peak p50 / p95 / max | 14 / 20 / 25 | 12 / 18 / 21 |
| observed seconds `>100` | 0 | 0 |

集中 session-cutoff delete-point burst：

| 指標 | Legacy fixed-45 | Causal-manifest overlap |
|---|---:|---:|
| 每日 cutoff burst p50 / p95 / max | 53.5 / 102 / 141 | 18 / 50 / 61 |

### 完整動態池的粗估範圍

完整 selected product-day 沒有 raw replay，因此只報兩個透明、未校準的 scaling：

- `fixed-rate`：用完整 legacy 45 檔的 point load，依當日動態商品數除以 45。
- `selected-overlap-rate`：用因果 manifest 與 legacy replay 的交集 load，依當日完整
  selected products 除以 covered selected products。

| 每日組合 peak | Fixed-rate estimate | Selected-overlap-rate estimate |
|---|---:|---:|
| intraday p50 | 17.2 | 41.6 |
| intraday p95 | 28.4 | 76.0 |
| intraday max | 37.2 | 101.3 |
| cutoff p50 | 62.0 | 67.4 |
| cutoff p95 | 155.7 | 173.6 |
| cutoff max | 175.1 | 200.9 |
| projected intraday seconds `>100` / 60 sessions | 0 | 1 |
| projected cutoff seconds `>100` / 60 sessions | 13 | 21 |

以 common-date 動態池每日商品數 p50 `55` 粗估，point-union requests 約為
`6.6k–10.3k/day`（new + delete）。總量不是瓶頸；同秒 correlated burst，尤其 cutoff，
才是 spot `100 requests/s` 的風險。

## Future 50 ms taker hedge requests

這段只計 `spot_bid_future_taker` entry maker full fill 所觸發的 future taker hedge；
不包含 future exit-maker quotes。

Raw child 語意是一個 independent full-fill generation 一個 request。可部署 batching
key 固定為：

```text
(Date, ValueCode, entry_hedge_decision_time_ns)
request_quantity = sum(entry_hedge_executed_quantity)
```

不能只用整秒作 batch key，否則會把應在不同 50 ms decision cursor 送出的 hedge
延遲到同一秒尾。

| 指標 | Legacy fixed-45 | Causal-manifest overlap |
|---|---:|---:|
| raw fill children | 3,538 | 2,055 |
| exact-key batched requests | 2,304 | 1,338 |
| request reduction | 34.88% | 34.89% |
| raw event-second p95 / p99 / max | 4 / 7 / 12 | 4 / 7 / 12 |
| raw seconds `>5` | 49 | 32 |
| raw rolling-1s max / sessions曾超過5 | 15 / 29 | 15 / 19 |
| batched event-second p95 / p99 / max | 1 / 2 / 4 | 1 / 2 / 4 |
| batched seconds `>5` | 0 | 0 |
| batched rolling-1s max / sessions曾超過5 | 4 / 0 | 4 / 0 |
| maximum children / summed qty in one batch | 12 / 12 | 9 / 9 |

七個 batched keys 含至少一筆 `entry_hedge_executable=false` child；而且來源明示
`joint_volume_allocated=false`。因此 batching 後低於 5 requests/s 不等於 9–12 口
一定能在同一本 future book 成交。Production controller 應讓 fill-triggered hedge
優先於 discretionary future maker quote，並另做 aggregated-quantity depth replay。

## Future exit-maker requests 尚不能合法加總

現有：

```text
maker/data/walkforward/exit_maker_narrow_60d/
  Date=*/ValueCode=*/exit_maker_candidate_aliases.parquet
  Date=*/ValueCode=*/exit_maker_transitions.parquet
```

是逐 entry position、exit rule、route 的 independent counterfactual candidates；尚未
把同商品庫存、相同 exit price、共同 quantity、FIFO/OCO 與 position cap 合成實際
working orders。直接把其中 submit/cancel rows 加總會嚴重高估 future maker requests。
因此目前不能用這批 rows 判斷 future `5 requests/s` 是否足以同時承擔 exit maker 與
entry hedge；需要 joint inventory-to-price controller 後再數一次。這項缺口不能用
上述 hedge batching 結果掩蓋。

## Full-universe makerFill 能做什麼

`HFT/data/makerFill/{YYYYMMDD}_makerFill.parquet` 在 2026 年目前有 153 個 sessions；
目前因果 manifest 的 72 個日期皆有對應檔案。例如 `20260813_makerFill.parquet`
約 163 MiB、13,058,679 rows，schema 為：

```text
QuoteCode, ChannelSeq,
Ask1_FillSeconds, Ask2_FillSeconds,
Bid1_FillSeconds, Bid2_FillSeconds
```

其中 makerFill 的 `QuoteCode` 是現貨商品碼（例如 `1101`），不是月選表內的期貨
`QuoteCode`；須由 `ValueCode` 映射。它可以在已有 candidate 的
`(spot ValueCode, maker snapshot ChannelSeq, initial rank)` 上，
快速提供 A/B1-2 的近似 EOD-looking `FillSeconds`；再與 candidate retreat stop 比較，
可形成 approximate fill-before-cancel label。

但現有 monthly selector / dynamic full-band artifacts 沒有完整的 absolute target、spot
snapshot `ChannelSeq`、submit cursor 與 retreat cursor，不能只多做一次 parquet merge
就得到完整動態池 q95 AB1/2 fill。下一步仍須先產生修正後的 point candidate table。
而 makerFill 的 Float32 time 不是 exact fill `RecvTime`，沒有 partial/cancel ACK；
`fill + 50 ms` 的 future price仍要另查 future book。因此它適合快速 screening，不能
標成 exact hedge timestamp 或 production fill truth。

## 主要陷阱

1. 不可把 fixed 45 的 exact burst 稱作完整因果池結果。
2. 不可把 cross-epoch same-price generations 當多張 production orders。
3. 不可把一秒內 mixed new/delete 自動稱為同一張單的 replace。
4. 不可把 nominal cutoff delete 當已收到 cancel ACK。
5. 不可把 independent fill children 的 quantity/depth 當共同可成交量。
6. 不可把 future entry hedge 的 5 requests/s 結果套到 future exit maker。
7. 1 Hz final-net 會忽略秒內先掛後撤的 path；若 production 真會逐 raw event action，
   必須使用 generation-native burst 或重跑 corrected raw-event state machine。
