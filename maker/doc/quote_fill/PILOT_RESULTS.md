# WP02–03 Raw Replay Pilot Results

## 結論

這版已把 D−1 商品別 adaptive upper、`SpreadPairTotalCount` 取樣、多層存續掛單、退後撤單、raw maker fill、50 ms 反腿 taker 與 fill-conditioned 日內 latent exit 串在一起。

目前最值得繼續的 route 是 `Spot Bid maker -> Future taker`；它的 fill 明顯多於 `Future Ask maker -> Spot taker`。但這仍不是可交易 EV：exit 目前只有 1 秒 basis path first-passage，尚未重播真正的 exit maker fill，也沒有把費稅、隔夜價值、共同成交量與部位上限入帳。

固定 10／15／20 BP 或 1／2 tick 沒有被升級成策略。表內 `q50`／`q80` 是每商品 D−1 non-overlap excursion quantile 所產生的 adaptive policy alias；實際 action 仍是當下反腿行情換算並 round 後的合法絕對價格。

## 範圍與取樣

- 日期：`20260128, 20260223, 20260318, 20260420, 20260609, 20260617, 20260720, 20260811`。
- 商品：`2303, 2317, 2603, 2881`。
- Raw coverage：31 pair-days；舊八日pilot的`2603/20260617`曾因上游日沖資格mapping漏掉。新版131-session point-in-time loader已確認該日`CZFF6`兩腿完整，此缺口不再延續到walk-forward replay。
- Adaptive-valid：27 pair-days；2303 只有 4 日通過 D−1 prior gate。
- Raw rows／trades：3,823,204／445,017；52,594 個 future zero-book trade rows 中，52,593 個承接先前正式 book，1 row 無 prior book。
- `SpreadPairTotalCount` epochs：2,007。
- Sparse target observations：78,692；其中 44,006 admitted。
- Policy aliases：6,489；實際 raw-order facts：4,453。2,036 個 facts 同時對應 q50／q80，不能把 aliases 合併當獨立 N。

同一 epoch、route、stage、絕對價格只建立一次；同 epoch target 往前的新價加一層並保留舊層，退後時撤掉較積極層。跨 epoch 同價允許另一個 independent research generation。所有 CI／泛化判斷都應以整日 block，而非把 orders 當 IID。

## Maker fill 與撤單

| Entry route | Boundary | Orders | Full fill | Partial | Full-fill rate | Cancel-required rate | Pair-day fill median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future Ask -> Spot taker | q50 | 1,597 | 32 | 0 | 2.004% | 97.996% | 0.806% |
| Future Ask -> Spot taker | q80 | 1,608 | 11 | 0 | 0.684% | 99.316% | 0.000% |
| Spot Bid -> Future taker | q50 | 1,621 | 142 | 2 | 8.760% | 91.240% | 3.125% |
| Spot Bid -> Future taker | q80 | 1,663 | 82 | 0 | 4.931% | 95.069% | 0.000% |

撤單率高的主因是 target retreat，不是每個 raw tick 都重掛；每個 product／route／policy-day 的撤單數中位約 38–41。Future maker order lifetime 中位約 124 秒，Spot maker 約 15.8／20.8 秒。V0 是 retreat observation 當下立即撤單；`+10/50/100/500 ms` 只保留 shadow tape，不是正式 cancel ACK 模型。

### 商品拆分

| 商品 | Valid days | Future q50 | Future q80 | Spot q50 | Spot q80 |
|---|---:|---:|---:|---:|---:|
| 2303 | 4 | 15/381 = 3.94% | 6/384 = 1.56% | 25/403 = 6.20% | 9/416 = 2.16% |
| 2317 | 8 | 6/500 = 1.20% | 1/494 = 0.20% | 27/527 = 5.12% + 1 partial | 11/553 = 1.99% |
| 2603 | 7 | 7/226 = 3.10% | 4/230 = 1.74% | 16/222 = 7.21% | 13/224 = 5.80% |
| 2881 | 8 | 4/490 = 0.82% | 0/500 | 74/469 = 15.78% + 1 partial | 49/470 = 10.43% |

Spot pooled fill 很受單日影響；`2881/20260223` 占 Spot q50 full fills 的 50%、q80 的 58.5%。因此商品表只是 pilot 描述，不能據此宣稱已校準。

### Exact queue 修正

Sparse target event 可以去重，但 submit 當刻的 queue 必須回查完整 maker raw state。修正後 2,951/6,489（45.5%）orders 的 queue 改變；any fill 由 334 降至 269，full fill 由 326 降至 267。修正版的 full fills 比舊結果少 18.1%；反過來說，舊結果比修正版多報 22.1%。本文件只引用修正版。

### Existing makerFill sanity

1,334 個可對到現貨 BID1／BID2 的 spot-origin samples 中：

- 既有「不撤、看到收盤」makerFill：1,065／1,334 = 79.84%。
- 本策略 moving target／retreat cancel raw replay：106／1,334 = 7.95%。
- 106 個策略 full fills 全部也被既有 makerFill 標到。

因此既有 makerFill 適合做方向 sanity，但不能直接作本策略 fill probability。它同時有「看到收盤、不撤單」的樂觀因素，與「初始顯示量全當 queue ahead、不扣前方撤單」的保守因素。

## Spot partial fill

Spot route 226 個 any-fill aliases 中只有 2 個最後停在單一 board lot；q50 partial/any = 1.39%，q80 為 0%。但 224 個 full fills 中有 65 個不是同一事件湊滿兩 lots，代表等待時間仍值得建模。

| Boundary | 50 ms | 250 ms | 1 s | 2 s | 5 s |
|---|---:|---:|---:|---:|---:|
| q50，P(兩 lots於期限內完成｜已 first fill) | 77.8% | 79.9% | 86.1% | 88.9% | 95.1% |
| q80，P(兩 lots於期限內完成｜已 first fill) | 79.3% | 80.5% | 89.0% | 91.5% | 96.3% |

這支持先累積兩 lots 再 hedge 的 baseline。第一 lot 後若先遇策略撤單，該 generation 在各期限內視為已知未完成；真正資料缺失才是 unknown。後續 top-up timeout 應比較 spot taker 補量成本、等待 basis risk 與 future hedge成本，而非先任意指定一秒。

## 50 ms entry hedge

p50／p80 若落在同一 raw order，50 ms hedge fact 只計一次。267 個 full-fill aliases 折疊為 202 個 raw fills：Future maker 34、Spot maker 168。

| Maker route | Raw fills | L1–L5 depth-priceable | Total slippage p50 | p80 | p95 | Decision book <=100 ms | <=1 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future Ask -> buy Spot | 34 | 34/34 | 0 bp | 0 bp | 21.37 bp | 94.1% | 100.0% |
| Spot Bid -> sell Future | 168 | 168/168 | 0 bp | 0 bp | 15.67 bp | 69.0% | 87.5% |

正 slippage 表示不利。Future maker route 有 7/34、Spot maker route 有 15/168 為正成本，其餘在 50 ms 後最佳價未變；標準一口／兩 spot lots 都未超過 L1–L5 深度。

「depth-priceable」不等於可直接下單：Spot maker 後要打 future 時，latest future book 的 p95 age 約 2.07 秒。主表尚未硬選 freshness cutoff，因此對該 route 必須同時看 `<=100 ms`、`<=1 s` sensitivity；不能用 100% depth availability 宣稱 hedge 已穩健完成。

## Raw full-fill cursor conditional latent exit

舊的 upper-crossing latent cycle 不能直接與 fill rate相乘，因 retained layer 可能在原 excursion 結束後才成交。新 label 改從每筆 raw full-fill cursor 開始，凍結 fill 當下最新 causal fair `M0`，等待 30 秒後才從 1 秒 grid 搜尋 first passage；任何 eligibility gap 都 censor，不橋接。

| Entry route | Boundary | Full fills | Frozen center hit | Frozen D−1 lower hit | Median time center | Median time lower |
|---|---:|---:|---:|---:|---:|---:|
| Future Ask -> Spot taker | q50 | 32 | 100.0% | 96.9% | 65.9 s | 114.0 s |
| Future Ask -> Spot taker | q80 | 11 | 100.0% | 100.0% | 116.8 s | 626.8 s |
| Spot Bid -> Future taker | q50 | 142 | 94.4% | 81.7% | 50.8 s | 110.5 s |
| Spot Bid -> Future taker | q80 | 82 | 81.7% | 68.3% | 48.9 s | 211.2 s |

以「order 被 full fill，且之後 basis mid 到 frozen target」計算的 independent-event latent opportunity rate為：

| Entry route | Boundary | Center / submitted order | D−1 lower / submitted order |
|---|---:|---:|---:|
| Future Ask -> Spot taker | q50 | 2.004% | 1.941% |
| Future Ask -> Spot taker | q80 | 0.684% | 0.684% |
| Spot Bid -> Future taker | q50 | 8.267% | 7.156% |
| Spot Bid -> Future taker | q80 | 4.029% | 3.367% |

這是 raw full-fill cursor conditional 的 basis-mid price-path機會，不是 completed trade probability：exit 還沒轉成合法 route、rounded price、queue 與 maker fill；也沒有使用 actual entry hedge cash price判斷真正可得 capture。

Dynamic fair 的 hit 可受 anchor 漂移促成，故決策表以 frozen fair 為較可歸因 benchmark，dynamic只保留 operational sensitivity。

## 商品 action research surface

下表把目前較有 fill 的 `Spot Bid maker -> Future taker` 依商品攤開。`Open/Lower/Band` 分別是實際合法 tick rounding 後的 entry-to-fair 距離、D−1 frozen lower 距離，以及兩者相加的 latent basis-mid band；它不是四腿現金 PnL。這三個 p50 以 submitted policy aliases／opportunities 加權，描述本 pilot 真正產生的 action surface，不是先將每個交易日等權後再取商品中位數。

| 商品 | q | Open / Lower / Band p50 | Full fill | 50ms hedge cost p95 | Hedge book <=100ms | Frozen lower hit / full fill |
|---|---:|---:|---:|---:|---:|---:|
| 2303 | 50 | 17.45 / 6.43 / 23.88 bp | 25/403 = 6.20% | 15.67 bp | 64.0% | 76.0% |
| 2303 | 80 | 22.35 / 12.10 / 34.45 bp | 9/416 = 2.16% | 15.67 bp | 55.6% | 66.7% |
| 2317 | 50 | 19.30 / 8.32 / 27.62 bp | 27/527 = 5.12% | 0 bp | 55.6% | 96.3% |
| 2317 | 80 | 25.02 / 16.20 / 41.34 bp | 11/553 = 1.99% | 0 bp | 63.6% | 90.9% |
| 2603 | 50 | 22.47 / 9.93 / 31.85 bp | 16/222 = 7.21% | 22.32 bp | 62.5% | 56.3% |
| 2603 | 80 | 24.94 / 15.82 / 40.94 bp | 13/224 = 5.80% | 22.37 bp | 69.2% | 46.2% |
| 2881 | 50 | 10.21 / 4.15 / 14.07 bp | 74/469 = 15.78% | 0 bp | 75.7% | 83.8% |
| 2881 | 80 | 16.52 / 8.17 / 24.87 bp | 49/470 = 10.43% | 10.43 bp | 71.4% | 69.4% |

這已是「機率 × tick geometry」的第一版研究表，但還不能直接把各欄相乘：50 ms cost 是 fill-conditioned 分布，lower hit 仍是 basis-mid first passage，且未包含 exit fill、未出場的 overnight value與共同資源限制。完整 16-row 兩 route 表在 `product_action_research_table.csv`，並固定標記 `ev_ready=false`。

## 目前 action table 可以與不可以回答的事

已可回答：

- D−1 adaptive q50／q80 在當下 legal tick 產生多少 quote opportunities。
- 每個 route 的 full／partial fill、retreat cancel、queue 與 shadow latency表現。
- Full fill 後 50 ms taker 的 L1–L5 price/depth cost。
- Full fill 後同日回 center／D−1 lower 的 latent first-passage。

仍不可回答：

- Exit maker order 的 fill probability與真正四腿成交現金流。
- Same-day aggressive exit 相對 overnight carry 的互斥 branch EV。
- 當沖現股實際 matched quantity、費稅、融資、隔夜 gap／roll value。
- 多 generations 共用成交量、own-order priority、共同 hedge depth與部位限制後的 portfolio PnL。

因此 `action_research_summary.csv` 是 EV surface 的骨架，不是 optimizer input；所有 row 都維持 `ev_ready=false`。

## Conservative taker-exit cash benchmark

將267個full-fill aliases折成202個實體fill，entry使用已重播的50 ms hedge價格；exit則從下一個完整秒開始，以spot bid與future executable ask同時taker平倉。若當日未達gross target，非到期合約以次一session同一`QuoteCode`的第一筆合格book作一日carry benchmark；到期合約不得偷接次月，改用到期日最後一筆合格book強制平倉。

逐筆`end_date`核對後，202個positions中有162個在合約到期日（20260223為146、20260318為9、20260617為7）。因此八日成交樣本嚴重偏向expiry regime，不能估一般日EV。

四個gross cash target中，200 TWD最接近break-even：202條filled+hedged paths的gross mean為-1.98 TWD、median為0，尚未扣現股／期貨稅費、broker commission或carry成本。若只作非常樂觀的法定稅敏感度（假設所有同日現股量皆符合1.5‰當沖賣出稅、隔日3‰、期貨每邊0.00002，且broker commission與funding均為0），平均仍約-451.17 TWD／filled path。這不是正式net EV，而是說明目前q50／q80加純taker exit的gross空間不夠；後續應比較更極端legal ticks與exit maker，而不是把八日結果外推成整個策略無效。

## 131-session loader 的五商品 smoke

新版 execution runner 已不再讀八日 fair panel；它會驗證每日 completion marker，從該日 causal fair／mapping 與嚴格 `<D` 的 60-session rolling boundary 建立 q50／q80／q95 raw replay。第一個 checkpoint 使用 `20260813 × {2303,2317,2603,2881,6005}`：

| Entry route | q | Submitted aliases | Cancel-required | Full fill | Full-fill rate |
|---|---:|---:|---:|---:|---:|
| Future Ask -> Spot taker | 50 | 310 | 302 | 8 | 2.581% |
| Future Ask -> Spot taker | 80 | 303 | 301 | 2 | 0.660% |
| Future Ask -> Spot taker | 95 | 306 | 305 | 1 | 0.327% |
| Spot Bid -> Future taker | 50 | 285 | 283 | 2 | 0.702% |
| Spot Bid -> Future taker | 80 | 291 | 290 | 1 | 0.344% |
| Spot Bid -> Future taker | 95 | 295 | 295 | 0 | 0% |

1,790 aliases折疊成1,169個實體raw-order facts；14個full-fill policy aliases只對應13個實體fill／hedge facts。13個50 ms hedge的slippage p50／p95為0／23.26 bp，但book age只有8／13在100 ms內、10／13在1秒內。探索組6005的4筆fill只有1筆book age <=1秒，其他3筆約6.94秒；因此「future maker spread很寬」不能單獨升級為候選，必須在entry-fill當下套hedge freshness／depth分支。

同一日以raw L1--L5、零額外exit latency做保守taker/taker checkpoint：

| Entry route | q | Full-fill aliases | Frozen center same-day | Frozen lower same-day |
|---|---:|---:|---:|---:|
| Future Ask -> Spot taker | 50 | 8 | 3 | 1 |
| Future Ask -> Spot taker | 80 | 2 | 0 | 0 |
| Future Ask -> Spot taker | 95 | 1 | 0 | 0 |
| Spot Bid -> Future taker | 50 | 2 | 2 | 1 |
| Spot Bid -> Future taker | 80 | 1 | 0 | 0 |
| Spot Bid -> Future taker | 95 | 0 | -- | -- |

其餘已建倉路徑都是`carry_at_eod`，不是零PnL或失敗；仍需接overnight terminal cashflow。這只是單日runner／schema checkpoint，不是機率校準。產物在`maker/data/walkforward/execution_smoke_exit/`，其中`smoke_entry_checkpoint.csv`、`smoke_hedge_checkpoint.csv`與`smoke_exit_checkpoint.csv`皆明示`pathwise_ev_ready=false`。

## 60-session 五商品 narrowed execution checkpoint

這是 60 個交易日的 research/development checkpoint，不是 production 或 OOS 結果。目標格為 60 sessions × 5 商品 = 300 product-days；實際完成並逐一核對 completion marker、artifact hash 與 row count 的 partitions 為 290。`2317/2603/2881/6005`各60日，`2303`為50日；後者另10日缺`daily_mapping`，不可暗補成零成交。290個已納入日的 rolling boundary 均通過 exact `QuoteCode`、`source_asof_date < Date`與 D−1 safety 檢查；q50／q80仍是`rolling_latent_candidate`，q95保留為`tail_diagnostic`。

全窗共有216,825個submitted policy aliases、171,678個跨q去重的physical raw-order facts、3,698個physical full fills與3,696個可定價的50 ms hedges；沒有L1--L5 depth shortfall。q內alias與raw fact是一對一，但同一實體order可同時命中多個q，因此下表各q是policy view，**跨q不可相加，也不可當成獨立樣本**。`cancel_required`表示策略在retreat時要求撤單，不是交易所cancel ACK。完整CSV同時保留pooled physical count及daily／product-day-balanced rate與support；下表用q內physical denominator做可重算的分母核對。

表格縮寫：`F`=`Future Ask maker -> Spot taker`，`S`=`Spot Bid maker -> Future taker`。下表把各route在q50／q80／q95的結果縮成range；`full/cancel`是q內physical-order rate，freshness分母是full fill，slippage是可定價hedge的p95。Spot partial欄仍按q列出，不能跨q相加。

| 商品 | F full/cancel % | S full/cancel % | S partial q50/q80/q95 | Hedge book <=1s，F/S | Slippage p95 bp，F/S |
|---|---:|---:|---:|---:|---:|
| 2303 | 1.32--5.10 / 94.9--98.7 | 1.24--7.83 / 92.2--98.8 | 10/5/3 | 100 / 98--99 | 41.0--42.9 / 27.9--35.3 |
| 2317 | 0.73--2.28 / 97.7--99.3 | 0.44--4.22 / 95.8--99.6 | 15/4/1 | 100 / 93--96 | 19.8--21.0 / 16.7--19.0 |
| 2603 | 0.44--2.24 / 97.8--99.6 | 0.76--3.66 / 96.3--99.2 | 19/5/3 | 97--100 / 85--95 | 23.3--26.3 / 22.4--23.0 |
| 2881 | 0.72--4.17 / 95.8--99.3 | 0.40--4.22 / 95.8--99.6 | 3/2/0 | 95--98 / 55--69 | 40.3--41.5 / 10.4--40.2 |
| 6005 | 1.04--2.83 / 97.2--99.0 | 0.39--2.64 / 97.4--99.6 | 14/12/3 | 66--71 / 40--48 | 7.6--12.8 / 0--69.6 |

3,698個實體fills中有3,696個hedge可定價；兩個decision books因`ref_price_band`關閉：`2603/20260520/F`與`6005/20260608/S`。`6005/S/q95`的0 bp只表示stale book沒有觀察到價格變動，不能解讀為零可執行成本。Future-maker沒有partial fill且spot hedge通常較新；Spot-maker的fill優勢依商品／q而異，且future hedge明顯較舊。

下表是product-day-balanced geometry的p50；每格為`upper/lower bp；upper/future-tick、lower/spot-tick`。Future/spot tick-BP ratio的商品p50介於0.984與0.993，因此這裡用各route真正maker leg的tick換算，而不是固定BP閾值。

| 商品 | q50 | q80 | q95 |
|---|---|---|---|
| 2303 | 7.0/7.0 bp；0.23/0.23 tick | 16.5/16.1 bp；0.54/0.53 tick | 33.1/32.4 bp；0.91/0.89 tick |
| 2317 | 8.4/8.5 bp；0.43/0.43 tick | 15.7/15.8 bp；0.80/0.81 tick | 21.7/21.8 bp；1.09/1.08 tick |
| 2603 | 8.4/8.1 bp；0.35/0.33 tick | 16.2/15.3 bp；0.66/0.63 tick | 23.6/23.4 bp；0.97/0.95 tick |
| 2881 | 8.1/6.9 bp；0.22/0.19 tick | 17.5/15.8 bp；0.46/0.41 tick | 30.6/28.6 bp；0.80/0.73 tick |
| 6005 | 7.3/7.8 bp；0.54/0.59 tick | 23.1/22.8 bp；1.62/1.66 tick | 60.2/54.1 bp；4.24/3.96 tick |

除了`6005`，四個核心商品的q50／q80與多數q95距離約一個legal tick以內，正是cross-q raw-order sharing很高的原因；q-knots不是三組獨立可成交價。`6005`雖有真正multi-tick的q80／q95 geometry，卻同時有最差的hedge freshness與很高的carry比例，不能只憑寬spread納入。

Exit只以「physical full fill且50 ms hedge可定價」為opened denominator。每格為`center same-day/carry；lower same-day/carry`；每個rule的`same-day + carry = opened`，所以carry是尚未結算的隔夜分支，不是no-fill、零PnL或失敗。

| 商品 | Route | q50 | q80 | q95 |
|---|---|---|---|---|
| 2303 | F | C 205/86；L 176/115 | C 140/47；L 102/85 | C 51/25；L 33/43 |
| 2303 | S | C 239/192；L 230/201 | C 134/81；L 101/114 | C 33/39；L 18/54 |
| 2317 | F | C 96/36；L 63/69 | C 51/27；L 30/48 | C 31/11；L 13/29 |
| 2317 | S | C 197/61；L 125/133 | C 61/24；L 26/59 | C 18/10；L 3/25 |
| 2603 | F | C 30/35；L 23/42 | C 12/13；L 6/19 | C 8/5；L 2/11 |
| 2603 | S | C 71/33；L 46/58 | C 34/20；L 16/38 | C 15/7；L 5/17 |
| 2881 | F | C 37/83；L 34/86 | C 18/34；L 9/43 | C 10/11；L 5/16 |
| 2881 | S | C 40/74；L 33/81 | C 15/21；L 10/26 | C 7/4；L 3/8 |
| 6005 | F | C 154/440；L 116/478 | C 90/371；L 33/428 | C 21/202；L 2/221 |
| 6005 | S | C 205/215；L 185/235 | C 98/110；L 86/122 | C 37/26；L 0/63 |

Runner在整個product-day都沒有可定價hedge時不建立該日exit path，所以全部entry facts的exit-fact coverage依商品約為86%--99%；缺少的exit facts不能填成no-fill或零PnL。已opened的3,696個實體positions則都有互斥且完整的center／lower same-day-or-carry標記。

這個checkpoint仍然`pathwise_ev_ready=false`：尚未加入fees/tax、overnight terminal cashflow、共同成交量／queue、portfolio inventory與exit-maker fill。尤其`product_action_lookup_checkpoint.csv`只是D−1 geometry與execution fact的join checkpoint，不是EV table；目前不能拿上述marginal rates互乘或直接選q。可重跑產物為[entry](../../data/walkforward/execution_narrow_60d/report_60_sessions/product_action_entry.csv)、[exit](../../data/walkforward/execution_narrow_60d/report_60_sessions/product_action_exit.csv)、[geometry](../../data/walkforward/execution_narrow_60d/report_60_sessions/product_action_geometry.csv)、[lookup checkpoint](../../data/walkforward/execution_narrow_60d/report_60_sessions/product_action_lookup_checkpoint.csv)與[report completion marker](../../data/walkforward/execution_narrow_60d/report_60_sessions/report_complete.json)。

## 下一步

1. 對 full-fill + 50 ms hedge 後的 achieved basis，枚舉兩條 exit maker route的合法 ticks。
2. 用同一套 layered raw replay建立 exit fill／force／carry competing outcomes。
3. EOD 同時比較 `continue maker / aggressive same-day exit / carry overnight / emergency`，overnight不是 censor。
4. 由實際四腿價格與數量套 versioned fees/taxes，建立 mutually-exclusive pathwise EV。
5. 擴完整 2026 expanding walk-forward；Jul–Aug locked holdout，整日 block bootstrap與product/peer shrinkage。

## Artifacts

主要產物位於 `maker/data/quote_fill/`：

- `target_observations.parquet`
- `order_aliases.parquet`
- `raw_order_facts.parquet`
- `fill_by_day_symbol.csv`
- `fill_summary.csv`
- `fill_by_rank.csv`
- `fill_terminal_summary.csv`
- `makerfill_sanity.parquet` / `makerfill_sanity_summary.csv`
- `spot_partial_completion.csv`
- `hedge_facts.parquet` / `hedge_summary.csv`
- `latent_exit_opportunity_labels.parquet` / `latent_exit_opportunity_summary.csv`
- `action_research_summary.csv`
- `product_action_research_table.csv`

八日 pilot 各表的機率均保留自然 base rate，沒有抽未成交樣本；60-session checkpoint則使用其290個已完成product-days的完整raw facts。
