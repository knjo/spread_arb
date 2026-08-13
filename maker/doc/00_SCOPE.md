# 策略範圍與共同假設

## 交易方向

第一版只做先賣 basis、後買回 basis：

```text
Long Spot + Short Future
```

這不是持有到期收斂策略，而是交易 basis 日內發散與回歸。

四條 maker／taker route：

| 階段 | Maker | 約 50 ms 後的 Taker |
|---|---|---|
| Entry | Future Ask | Buy Spot |
| Entry | Spot Bid | Sell Future |
| Exit | Future Bid | Sell Spot |
| Exit | Spot Ask | Buy Future |

兩條 entry 都建立同方向部位；兩條 exit 都降低同一 paired position。雙路同掛時需處理 cancel race、double fill 與部位 reservation。

## 三種 tape basis

```text
B(F, S)       = 10,000 * (F / S - 1)
B_mid         = B(FutureMid, SpotMid)
B_sell_taker  = B(FutureExecBid, SpotA1)
B_buy_taker   = B(FutureExecAsk, SpotB1)
```

`FutureExecBid/Ask` 取明掛 L1 與期貨衍生 Best quote 的可成交優價。`B_mid` 用於 fair／residual；另兩者是即時 taker／taker 基準。Maker／taker 的 `B_entry`、`B_exit` 是成交結果，不是第四種行情 basis。

## 目標報價

```text
U_t = M_t + W_open
L_t = M_t - W_close

Future Ask entry = ceil_future_tick(SpotA1 * (1 + U_t / 10,000))
Spot Bid entry   = floor_spot_tick(FutureExecBid / (1 + U_t / 10,000))
Future Bid exit  = floor_future_tick(SpotB1 * (1 + L_t / 10,000))
Spot Ask exit    = ceil_spot_tick(FutureExecAsk / (1 + L_t / 10,000))
```

基準例為 `M=50 bp, U=70 bp, L=30 bp`，只作 benchmark，不預設是最終政策。

## 第一版固定邊界

- 只使用 2026 年且期現兩腿都有 raw data 的交易日。
- 近月標準股期，基準 `contract_size=2,000`。
- Feature 主要來自現貨；期貨只作定價、maker fill 與 taker execution facts。
- Maker fill 後的正式 hedge 基準固定為 `fill RecvTime + 50 ms`。
- Taker／taker 只作即時控制組、緊急避險及 force-flat。
- 不允許使用任何當下尚未可知的全日統計或 future label。

## Width 參數的跨日規則

- D 日開盤前的商品結構參數只使用 D−1 以前資料；D 日全天平均 spread 不得回填 D 日。
- `tick BP`、`spread ticks` 與兩腿 `TTBand BP` 分開保存；`spread BP / tick BP` 只代表簿寬幾檔。
- Nominal width 轉為合法 maker tick 後，以 `rounded_target_price` 作 action identity；多個 width 落在同價時共用 raw fill fact。
- Width pilot 與限制見 [quote_width/RESULTS.md](quote_width/RESULTS.md)。
