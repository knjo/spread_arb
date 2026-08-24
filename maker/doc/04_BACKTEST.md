# Work Package 04：週轉 EV 與逐事件回測

合法tick action、canonical／decision action、terminal path schema、support gate與131-session execution rollout以[quote_fill/EV_LOOKUP.md](quote_fill/EV_LOOKUP.md)為實作契約。`ev_surface.py`與product-day execution partition介面已完成；目前八日pilot仍不是EV-ready，因尚無完整broker cost profile、overnight terminal labels與portfolio joint allocation。

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

允許隔夜，不把日終未出場一律強制平倉。接近收盤時，每個已避險部位比較的 action 至少包含：

```text
keep current maker exit
improve exit by one or more legal ticks
same-day taker exit
carry overnight
```

同日出場可因現股實際當沖配對得到較低稅負，因此可能值得犧牲部分 exit tick、掛得更積極；是否值得由「少拿的 basis edge」對「稅差、隔夜資金／跳空風險與次日預期出場價」的 joint EV 決定，不設成必然 force-flat。

Overnight branch 需延續實際 spot／future inventory與未實現現金流，加入 financing、margin、overnight gap、下一交易日出場，以及接近到期時的 roll／settlement結果。最大持有天數與 roll policy可在完整 portfolio replay 前凍結，不阻塞 WP02–03 產生事實表。

Raw action 由 episode start、商品／合約、route、stage、rounded target、qty 與 replay version 定義；固定 BP／tick 只是 diagnostic aliases。`basis_capture_bp` 也是 latent diagnostic，PnL 必須使用四腿實際 fill price／qty、partials 與 force-flat cashflow。

## 掛價與週轉不是固定目標

不預先指定「每天一定交易 1～2 次」。每個 SpreadPair epoch 先列舉當下所有合法、被動的 maker tick actions；q50／q80／q95 只提供 latent tail knots，實際 action 仍由反腿 executable quote與合法 tick ladder決定。每個 action cell至少估：

- 每 flat-hour 出現／可掛的機會率。
- Fill、retreat cancel、partial與50 ms hedge的 competing-risk分布。
- Actual hedge後 same-day target exit、aggressive exit、carry overnight與emergency branch。
- 各 branch 的四腿現金流、稅費、capital-time與tail loss。

第一版 action score 為：

```text
score(action | state, inventory)
  = pathwise EV per admitted quote
  - capital-time penalty
  - downside / emergency penalty
  - incremental inventory penalty
```

只有 score 與保守 queue 情境都通過事前門檻才掛。較極端的 threshold會得到較高單次 gross edge、較低 reach／fill與較長等待；較近的 threshold則相反。模型直接比較這個完整 trade-off，completed cycles/day只是 sequential portfolio replay的輸出，不是拿來反推或硬湊的標籤。

同一時點多個 nominal thresholds若 round 到同一絕對價，只保留一個 raw action；不同退出／隔夜政策作為該 action 的 policy aliases比較，不能複製 fill事實增加樣本數。

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

必須處理 partial fill、cancel pending、double fill、reservation、residual inventory、hedge retry、entry cutoff、exit escalation、選擇性的 same-day force-flat 與 overnight carry。

## 帳本與報告

- 每筆成交按真實現金流、現貨手續費與交易稅、期貨費稅記帳。
- 現股證交稅在賣出時逐筆計算；只有同一證券商、同一帳戶、同一營業日、同一合格股票的現款買進與現券賣出相同數量，才按適用的現股當沖賣出稅率計算。資格依實際成交依法匹配，不依「正常 cycle 完成」判定；同日 force-flat／emergency fill 若符合條件也可能適用，未匹配或隔夜量按一般規則。
- 期貨每次買／賣皆依交易日與商品的期貨交易稅率計算，不套「當沖減半」。券商手續費另列，全部費率與資格使用 trade-date／product／account versioned config。
- 報告 completed cycles/day、same-day matched quantity、overnight carry rate／days、turnover、decision-time predicted EV、realized OOS daily net PnL 與 capital-time efficiency；predicted EV 不與 realized PnL 混稱。
- OOS 結果分中性、保守、樂觀 queue 情境；只有 touch-fill 或零 latency 才獲利即 No-Go。

目前 `execution_runner.py` 產生的是independent-candidate facts：同價aliases共用raw order／hedge結果，但不同假想orders尚未共同消耗成交量、資金或部位。只有在terminal branches完備、versioned費稅／融資成本齊全，且sequential portfolio replay完成後，才能把lookup的predicted EV與realized OOS PnL列入本WP正式報告。

規則來源：[TWSE 現股當沖制度](https://www.twse.com.tw/zh/products/system/day-trading.html)、[財政部當沖證交稅措施](https://www.mof.gov.tw/singlehtml/384fb3077bb349ea973e7fc6f13b6974?cntId=4493245d64e5422887a375921e889465)、[TAIFEX 期貨交易稅說明](https://www.taifex.com.tw/cht/9/tradersQAProducts)。實作仍以交易日有效的法規與費率設定為準。
