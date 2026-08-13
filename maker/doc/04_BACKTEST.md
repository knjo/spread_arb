# Work Package 04：週轉 EV 與逐事件回測

## Pathwise Action EV

```text
EV(action | state)
= Σ over mutually-exclusive, exhaustive terminal outcomes o:
    P(o | action, state)
    × E(pathwise fills and cashflows
        - spot/futures fees
        - spot/futures taxes
        - financing/capital/emergency costs
        | o, action, state)
```

Outcomes 至少包含 no-fill／requote、normal same-day exit、same-day force-flat、hedge failure／partial、cancel-race／double-fill 與 residual／overnight inventory。所有 branch 必須互斥且完備，避免將邊際機率相乘或把同一 emergency cost 重複扣除。

Raw action 由 episode start、商品／合約、route、stage、rounded target、qty 與 replay version 定義；固定 BP／tick 只是 diagnostic aliases。`basis_capture_bp` 也是 latent diagnostic，PnL 必須使用四腿實際 fill price／qty、partials 與 force-flat cashflow。

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
- 現股證交稅在賣出時逐筆計算；只有同一證券商、同一帳戶、同一營業日、同一合格股票的現款買進與現券賣出相同數量，才按適用的現股當沖賣出稅率計算。資格依實際成交依法匹配，不依「正常 cycle 完成」判定；同日 force-flat／emergency fill 若符合條件也可能適用，未匹配或隔夜量按一般規則。
- 期貨每次買／賣皆依交易日與商品的期貨交易稅率計算，不套「當沖減半」。券商手續費另列，全部費率與資格使用 trade-date／product／account versioned config。
- 報告 completed cycles/day、turnover、decision-time predicted EV、realized OOS daily net PnL 與 capital-time efficiency；predicted EV 不與 realized PnL 混稱。
- OOS 結果分中性、保守、樂觀 queue 情境；只有 touch-fill 或零 latency 才獲利即 No-Go。

規則來源：[TWSE 現股當沖制度](https://www.twse.com.tw/zh/products/system/day-trading.html)、[財政部當沖證交稅措施](https://www.mof.gov.tw/singlehtml/384fb3077bb349ea973e7fc6f13b6974?cntId=4493245d64e5422887a375921e889465)、[TAIFEX 期貨交易稅說明](https://www.taifex.com.tw/cht/9/tradersQAProducts)。實作仍以交易日有效的法規與費率設定為準。
