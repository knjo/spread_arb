# Work Package 03：固定 50 ms Hedge 與成本

## 時間定義

```text
t_hedge = maker_fill_receive_time + 50 ms
```

`maker_fill_receive_time` 由 raw fill event `RecvTime` 重建，是 private fill notification 的公開資料代理；後續以 shadow／真實委託校準。

在 `t_hedge` 處理完所有已收到行情後，取反向市場最後有效 L1–L5，依 maker 實際成交量計算 taker VWAP。不得用 50 ms 後才收到的最佳價，也不得把超過 L5 的殘量假設成交在 A1／B1。

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
= hedge_vwap_50ms - opposite_price_at_quote_submit

latency_plus_depth
= hedge_vwap_50ms - opposite_price_at_maker_fill
```

賣出 taker 的符號反向，使正值統一代表成本。

## 輸出

- Mean、p50、p90、p99 滑價。
- Hedge complete、掃過檔數、stale book、IOC partial、retry 與 emergency outcome。
- 10 ms、100 ms、1 s、5 s markout。
- 30／100 ms 僅作 sensitivity；正式基準固定 50 ms。

這裡的 50 ms slippage 只量價格、深度與延遲成本，不含券商手續費或交易稅；WP04 依實際成交價格與數量統一入帳一次，避免重複扣費。
