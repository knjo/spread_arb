# 因果商品池 q95 AB1/2：一秒掛撤與 100／5 筆上限

日期：2026-08-22  
狀態：完整因果商品池的 1 Hz quote-intent 結果；approximate entry fills 的 full-dynamic future hedge 已補完，exact joint volume 與 future exit-maker 仍待 replay

## 先講結論

目前賣價差 entry route 是現貨 Bid maker、期貨 taker hedge。依使用者確認的生命週期：

- 同一絕對價格不因 `SpreadPairTotalCount` 改變而重掛，舊單保留 queue age。
- target 往前到新的絕對價時，舊價繼續排，新價另送一張 new。
- target 後撤時，只 cancel 比新 target 更積極的舊價。
- 本策略沒有 exchange amend；訊息只分 `new` 與 `cancel`。

用不含固定 45 檔的因果 v2 商品池、q95、09:05 起每秒 final-net、只准新掛在
`BID1` 或 `BID1-1 legal tick` 後，現貨 `100 requests/s` 的**盤中容量足夠**：

| 進場時段 | active-second p50 / p95 / p99 / p99.9 | 盤中 max | 相鄰兩 bucket 保守 max | `>100` |
|---|---:|---:|---:|---:|
| 09:05–13:00 | 1 / 4 / 5 / 9 | 30 | 32 | 0 秒 |
| 09:05–13:20 | 1 / 3 / 5 / 9 | 30 | 32 | 0 秒 |

瓶頸只在 cutoff 同秒全部撤單：

| cutoff | 每日撤單 p50 / p95 / max | 超過 100 的日期 | max 最少清空時間 |
|---|---:|---:|---:|
| 13:00 | 65 / 165 / 199 | 16 / 72 | 2 秒 |
| 13:20 | 70 / 164 / 191 | 18 / 72 | 2 秒 |

所以 production 做法應是：先凍結 new，再以「最積極價優先」把 working entry
cancel 分散到兩秒；若 13:00 必須是硬 cutoff，就約 12:59:58 開始 drain。不能在
13:00.000 把全部 cancel 一次送出。

期貨 `5 requests/s` 比現貨更緊。舊 exact-fill 子樣本中，若每個 fill generation
各送一張 hedge，rolling 1 秒最高 15；但按精確
`(Date, ValueCode, hedge_decision_time_ns)` 合併同商品、同 decision cursor 的 quantity
後，rolling 1 秒最高降為 4，沒有超限。這表示 production 必須做 exact-cursor
hedge batching，且 fill hedge 優先於 future maker quote；它仍不是完整動態池的
joint-depth 認證。

## 範圍與因果性

正式輸入是：

```text
maker/data/walkforward/monthly_product_selector_causal_v2_20260822/
  daily_entry_manifest.csv
```

- 2026-05-04 至 2026-08-13，共 72 日。
- 3,886 個 product-days。
- 跨月聯集 119 檔，每日 35–74 檔。
- 月 M 名單只看完整 M-1；每日再套只看至 D-1 的 liquidity gate。
- 不再使用 retrospective fixed-45 universe。

每秒 target 使用 `causal_fair.parquet` 的 EWMA-120s anchor，加上 D-1 q95
`upper_distance_bp`，再由當下 futures executable bid 反推並向下取現貨合法 tick：

```text
threshold_bp = anchor_ewma_120s_bp + q95 upper_distance_bp
spot_bid_target = floor_to_spot_tick(
    fut_exec_bid / (1 + threshold_bp / 10,000)
)
```

`SpreadPairTotalCount` 只用於樣本去重，**不再是同價重掛理由，也不是 order gate**。
完整 09:05–13:20 共留下 959,619 個 exact spread-pair samples；只有 41 個樣本因
clock join 缺值而以一秒 `spot_sequence` change fallback。排除每日第一筆初始化後，
有變化的秒 p50 / p95 / p99 / max 為 `1 / 5 / 9 / 34` 個商品。

## 現貨 AB1/2 訊息量

### 13:00 停止進場版本

72 日合計：

| Request 類型 | 數量 | 每日平均 |
|---|---:|---:|
| new | 326,549 | 4,535 |
| 盤中 cancel | 321,211 | 4,461 |
| 13:00 cutoff cancel | 5,338 | 74 |
| 合計 | 653,098 | 9,071 |

所有 1,015,200 個盤中秒的平均為 `0.638 requests/s`；若只看有訊息的 412,176 秒，
p50 / p95 / p99 / p99.9 是 `1 / 4 / 5 / 9`。最高單秒 new 為 18、cancel 為 30、
合計為 30。固定一秒 buckets 的 100 上限沒有一次超過；再把相鄰兩 buckets 相加
作保守壓力測試，最高也只有 32。

Cancel 原因：

| 原因 | 筆數 |
|---|---:|
| target retreat | 311,428 |
| input gate close | 9,783 |
| 13:00 cutoff | 5,338 |

New 原因：

| 原因 | 筆數 |
|---|---:|
| target forward 到新絕對價 | 275,977 |
| 同價由非 AB1/2 變成可准入 | 46,143 |
| gate reopen | 3,960 |
| 每日第一個 eligible target | 469 |

因此實務上不是「每個 SpreadPair sample 都下單」；絕對價沒變就保留原單，只有
target price／gate／AB1-2 admission 的狀態改變才可能產生 request。

### 若所有 passive 點位都掛

這是壓力對照，不是建議策略：

| 指標 | AB1/2 only | 所有 passive target |
|---|---:|---:|
| 13:00 版總 requests | 653,098 | 2,118,182 |
| 每日平均 requests | 9,071 | 29,419 |
| active-second p95 / p99 | 4 / 5 | 7 / 12 |
| 盤中 max | 30 | 67 |
| 相鄰兩 bucket max | 32 | 89 |
| cutoff p95 / max | 165 / 199 | 778 / 1,040 |
| max cutoff drain | 2 秒 | 11 秒 |

就算所有 passive 點位都掛，盤中固定 bucket 仍未超過現貨 100/s；但 cutoff 幾乎
不可部署。只做 AB1/2 讓全日 request 減少 `69.2%`，也把最壞 cutoff 從 1,040 降到
199。

## q95 target 通常落在哪個點位

以下是 base gate 有效的 53,139,001 個 product-seconds。`BID2_BY_TICK` 指
`BID1-1 legal tick`；1 Hz panel 沒有完整 L2，因此若 order book 有 gap，不能把它
誤稱為逐筆 tape 的真實 BID2 rank。

| q95 target 點位 | product-seconds | 比例 |
|---|---:|---:|
| BID1 | 327,142 | 0.62% |
| BID1-1 tick | 6,588,423 | 12.40% |
| BID3–BID5 by tick | 36,586,184 | 68.85% |
| deeper than BID5 by tick | 9,623,971 | 18.11% |
| inside spread | 13,281 | 0.02% |

AB1/2 合計只占 `13.01%` 的有效 target-seconds。這能解釋為何只做 AB1/2 會大幅
降低掛撤量；但不能由這張表推論 AB3-5 的成交率或 PnL 較差，那仍須用 makerFill
fill-before-retreat label 比較。

## Future 5 requests/s

完整動態池已生成 7,730 個 approximate fill cursors，並逐筆查詢成交後 +50 ms 的 raw
futures causal book。依這批 cursor 計算：

| 指標 | 結果 |
|---|---:|
| hedge attempts | 7,730 |
| 每日 mean / p50 / p95 / max | 107.36 / 89 / 244 / 325 |
| active fixed-second p50 / p95 / max | 1 / 2 / 5 |
| fixed seconds `>5` | 0 |
| 任意 rolling 1 秒 max | 5 |
| rolling endpoints `>5` | 0 |

所以目前 approximate dynamic sample 剛好守住 futures 5 requests/s。這批 fast labels 沒有
出現相同商品、相同 exact decision timestamp 的 collision；正式 exact/joint-fill replay 仍可能
改變 clustering。Production 仍須用精確
`(Date, ValueCode, entry_hedge_decision_time_ns)` batching、rolling limiter，並讓 fill hedge
優先於 future maker quote，不能只按 wall-clock second 延遲湊單。

舊 fixed-45 exact replay 只保留作架構壓力診斷：未 batch 時 rolling-1s max 曾達 15；按同商品、
同 exact cursor 合併後降為 4。它不再作為本次動態池績效分母。

Future exit-maker 尚未計入。現有 exit transitions 是逐 position × rule 的反事實別名，
直接相加會嚴重高估；必須先把 FIFO inventory 合成商品 × 絕對價的實際 working order。
在此之前，production limiter 應保留 future 5/s 給 fill hedge，不能假設還有餘裕同時
高頻更新 future exit-maker。

## 精度邊界

1. 現貨結果是完整因果商品池的 1 Hz **quote-intent**，沒有在 maker fill 時移除 working
   order，也沒有成交後補單；因此不是 fill-aware exchange message truth。
2. 每秒只保留 final-net state，忽略秒內先掛後撤；這正對應使用者允許的秒 K controller，
   若實盤逐 tick action，必須重跑 raw-event state machine。
3. Cancel 是 request，沒有 ACK 或 cancel-race late fill。
4. AB2 是相對 BID1 的一個合法 tick，不是完整五檔 order book rank。
5. Fixed-second cap 與 rolling token bucket 不完全相同；實作仍必須有同一個 rolling limiter。
6. Spot 100/s 的結論只涵蓋 entry quote；future 5/s 已涵蓋 full dynamic approximate fills，
   但仍缺 exact/joint fills 與 joint exit-maker controller。

## 產物與速度

正式輸出：

```text
maker/data/walkforward/order_message_load_causal_v2_20260822_v2/
```

主要檔案：

| 檔案 | 用途 |
|---|---|
| `spot_capacity_summary.csv` | 四種 AB1/2／all-passive、13:00／13:20 容量摘要 |
| `spot_per_second.parquet` | 非零 product-combined second 的 new/cancel/request |
| `spot_daily_summary.csv` | 每日總量、盤中 peak、cutoff burst |
| `spread_pair_samples_per_second.parquet` | exact clock 與 fallback 樣本數 |
| `target_point_observations.csv` | q95 target 點位分布 |
| `spot_messages_by_reason.csv` | new/cancel 原因拆解 |
| `spot_message_events/Date=*/message_events.parquet` | 每張絕對價 working generation 的 sparse request |
| `daily_input_audit.csv` | 每日 coverage、clock match、稀疏化與訊息數 |
| `complete.json` | frozen runner contract |

完整 72 日重跑 wall time 約 35 秒，peak RSS 約 1.63 GB；資料逐日讀、只保留稀疏
transitions，沒有一次展開 21 GB causal panel 或 333 GB tickFeature。9 個新增 controller／
aggregation tests 與 Ruff checks 全數通過。

補充 legacy hedge 與 makerFill 可用性的逐欄證據在
`maker/doc/quote_fill/ORDER_MESSAGE_LOAD_LEGACY_DIAGNOSTIC_20260822.md`。
