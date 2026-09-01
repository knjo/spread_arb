# Work Package 03：50 ms 首次判定、B6 Retry 與成本

Entry 兩條 route 的八日 raw pilot 見
[quote_fill/PILOT_RESULTS.md](quote_fill/PILOT_RESULTS.md)。Cost-aware S1 已實作 Spot Bid entry hedge，以及第一條
exact pooled `Spot Ask maker → Future buy taker` normal-exit hedge；正式 71 日 replay、第二條 Future Bid exit route與
S3 route comparison 尚未完成。

## 時間定義

```text
t0 = hedge_trigger_cursor + 50 ms
deadline = min(t0 + 5 s, hedge venue session end)
```

`hedge_trigger_cursor` 依 route 定義：Spot Bid S1 entry 使用 legacy makerFill implied cursor，因此仍是 mixed-clock
approximation；exact pooled exit 使用累積到一個 futures-equivalent unit 的 physical fill cursor。後續仍需以 shadow／真實
委託校準。

在 `t0` 處理完所有已收到行情後，獨立檢查反向市場最後有效 L1–L5及該 venue 送單額度。若合法足量且可送，
actual request send與定價就在 `t0`；否則沿完整 raw-state change與 rolling scheduler往後找第一個同時合法足量且可送的
cursor，最長 5 秒。只有真正送出才消耗 request，價格取 actual-send cursor 的 executable VWAP；deadline inclusive
dispatch後仍失敗就原子式 timeout並走共同 emergency rollback。不得把等待期間看見的最佳價 hindsight搬回 `t0`，也不得
把超過 L5 的殘量假設成交在 A1／B1。

每筆另存 `t0` book status、actual send／book cursor、retry delay、on-time／delayed／timeout、arrival reference
coverage、slippage與rollback outcome；null reference／slippage不得補 0。

## S1 exit pre-fill hedgeability guard

第一條normal exit的Spot Ask absolute price／tick仍在entry actual-send凍結，不跟行情重定價；但被動單工作期間必須同時滿足Spot maker book合法，以及一口Future buy在當下causal L1-L5可完整執行。Spot或Future任一book變化都要喚醒reconciliation；Future gate關閉只撤回desired／送cancel，恢復時也只能回原凍結價。

這是pre-fill風險控制，不是Future liquidity reservation或成交保證，也不取代B6。Actual cancel effect前發生的Spot maker fill仍成立，並從該fill cursor +50 ms獨立執行上述hedge／retry／rollback。報表以`exit_desired_withdrawal_reason_counts`分開列出`gate:future_*`、`safety_cutoff`等撤回原因；任何最終裸腿維持fail closed。

## 四條路徑

| Maker fill | Taker hedge |
|---|---|
| Future Ask entry | Buy Spot |
| Spot Bid entry | Sell Future |
| Future Bid exit | Sell Spot |
| Spot Ask exit | Buy Future |

每條 route 分開估計 selection、latency 與 depth：

```text
selection_plus_latency
= hedge_vwap_actual_send - opposite_price_at_quote_submit

latency_plus_depth
= hedge_vwap_actual_send - opposite_price_at_hedge_trigger
```

賣出 taker 的符號反向，使正值統一代表成本。

## Spot maker partial：先量化，再選補量政策

Spot Bid／Ask maker 可能先只成交一個現貨 board lot，尚不足標準股期 `contract_size=2,000` 股。第一階段不先假設一定要立即 over-hedge，而是對每個 incremental spot fill 保存：

```text
initial_maker_fill_qty
time_to_cumulative_hedge_unit
P(cumulative maker qty >= contract_size by 50/100/250/500ms/1/2/5s)
residual_qty_at_each_horizon
```

先按商品、時段、spread ticks、queue／流動性 feature 檢查「只成交一個 board lot」是否常見，以及幾秒內補到另一個 lot 的機率。若問題集中在特定低流動性 state，讓 admission／EV 表學習該狀態，不以全商品同一假設處理。

若到候選等待期限仍不足一個股期 hedge unit，增加 `spot_taker_topup` branch：以當時 spot L1–L5 買／賣缺少的現貨數量，再建立整數股期 hedge。此 branch 會把部分成本從 future slippage 轉成 spot top-up spread／depth；「future 不會滑」只當待驗證假說，仍保存同期 future executable markout。

比較的政策至少為：

1. 等待 maker 累積到一個 hedge unit；
2. 等待 `T` 後 taker spot 補足 residual，再 hedge future；
3. 不補足，將 residual inventory 與後續 emergency cost保留給 WP04。

`T` 由樣本的 completion probability、兩市場成本與未避險風險共同選擇，不事前固定成單一秒數。

## 輸出

- Mean、p50、p90、p99 滑價。
- Hedge complete、掃過檔數、stale book、IOC partial、retry 與 emergency outcome。
- Spot initial／cumulative fill quantity、達到一個 contract-equivalent 的時間，以及 taker top-up qty／VWAP。
- 10 ms、100 ms、1 s、5 s markout。
- 30／100 ms 僅作 sensitivity；正式基準固定 50 ms。

這裡的 50 ms slippage 只量價格、深度與延遲成本，不含券商手續費或交易稅；WP04 依實際成交價格與數量統一入帳一次，避免重複扣費。
