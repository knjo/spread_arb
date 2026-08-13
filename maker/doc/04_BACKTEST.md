# Work Package 04：週轉 EV 與逐事件回測

## Cycle EV

```text
EV_cycle
= P(entry fill wins)
  * [P(hedge complete | fill)
     * {P(exit before cutoff | paired) * E(NetPnL_completed)
        + P(force close | paired) * E(NetPnL_forced)}
     + P(hedge failure | fill) * E(EmergencyPnL)]
  - E(cancel-race + residual costs)
```

往前／往後 N ticks 是不同 action；比較少賺的 spread、成交率、50 ms hedge cost、日內完成率與當沖稅務價值。

## 逐事件狀態

```text
FLAT
-> ENTRY_QUOTING
-> ENTRY_HEDGE_PENDING
-> PAIRED_POSITION
-> EXIT_QUOTING
-> EXIT_HEDGE_PENDING
-> FLAT
```

必須處理 partial fill、cancel pending、double fill、reservation、residual inventory、hedge retry、entry cutoff、exit escalation 與 force-flat。

## 帳本與報告

- 每筆成交按真實現金流、現貨手續費與交易稅、期貨費稅記帳。
- 當沖優惠只套實際同日配對完成數量。
- 報告 completed cycles/day、turnover、daily net EV、capital-time efficiency。
- OOS 結果分中性、保守、樂觀 queue 情境；只有 touch-fill 或零 latency 才獲利即 No-Go。
