# 全市場 A1-B1 流動性與研究商品縮減

## 結論

全市場 2026 walk-forward screen 已完成。資料範圍為 2026-01-26～2026-08-13 的 131 個共同交易日，共 31,360 個商品日、251 個曾出現商品；60-session screen 自 2026-04-02 起有 91 個可評分日。

不能把「A1-B1 spread 寬」直接等同不能做：

- Future maker → Spot taker 時，future spread 是 maker 腿。寬 spread 可能提供被動報價空間，但 fill、queue age 與 adverse selection 仍需 raw replay。
- Spot maker → Future taker 時，future spread 是 taker hedge 腿。寬 spread 會直接進入 hedge 成本，應列為 cost-test，不與前一條 route 混用。
- Spot spread 極寬則同時伴隨低 freshness、低 activity 與不足深度，較接近真正的不可用商品。

因此第一輪 raw replay 不再掃全部約240檔。2026-07-06起股票期貨500～2,500元級距改為1元tick後，v2 screen用日期與市場別重算future tick geometry；跨May／June／Jul-Aug都穩定的strict core為44檔，2330先作book-semantics quarantine，基準組保留43檔，再加入20檔pseudo-validation extension與wide-future controls。這是研究universe，不是production核准名單。

## Spread 寬度與流動性

下表以每個商品日的 09:05～13:20 一秒狀態為單位，先算該日 spread p50，再依 p50 spread 分桶。`fresh<=1s`、state-change rate 與fresh條件下的一口 hedge depth均為桶內商品日中位數。為完整涵蓋31,360個商品日，`5+`桶也保守納入457個沒有eligible spread p50的商品日。

| 市場 | 日內 spread p50 | 商品日 | Eligible | Fresh <=1s | State-change rate | 一口 hedge depth |
|---|---:|---:|---:|---:|---:|---:|
| Spot | 1 tick | 27,348 | 100.0% | 10.32% | 36.19% | 96.46% |
| Spot | 2 ticks | 2,894 | 99.92% | 6.98% | 13.88% | 80.00% |
| Spot | 3–4 ticks | 444 | 97.24% | 7.02% | 11.21% | 59.51% |
| Spot | 5+ ticks | 674 | 0.0% | 0.0% | 3.61% | 49.12% |
| Future executable | 1 tick | 4,633 | 100.0% | 27.36% | 33.08% | 100.0%* |
| Future executable | 2 ticks | 8,010 | 100.0% | 14.77% | 18.86% | 100.0%* |
| Future executable | 3–4 ticks | 9,029 | 100.0% | 8.22% | 11.69% | 100.0%* |
| Future executable | 5+ ticks | 9,688 | 99.98% | 5.75% | 10.42% | 100.0%* |

`*` Future eligible 本身已要求 executable lots > 0，因此一口深度在這個條件下接近 tautology，不能拿來證明深度穩健。真正 hedge 仍要看 50 ms 時點的 L1-L5、book age 與共同流動性。

結果支持兩件事：

1. Spread 越寬，freshness 與 book activity 明顯下降，確實是流動性警訊。
2. Future 5+ ticks 很常見，卻不等於完全沒有正式 book。直接刪除會失去 future-maker 的主要探索層；應把它與 future-taker 成本分 route 研究。

## 60-session route screen

每日 `D` 的 screen 只讀 `<D` 最近 60 個共同交易日；流動性欄用最近 20 日統計。Hard gate 包含：

- D 日 adaptive boundary 已有足夠歷史。
- Recent／long history support 足夠。
- Eligible grid 的日中位數達標。
- 兩腿 freshness、route-specific 一口 hedge depth、maker activity 與 TTBand 資料有效。

Maker／hedge spread寬只決定replay strata，不直接hard reject。歷史eligible median／q10只保存為warning／state-risk features，也不再淘汰商品；當下`eligible=false`才禁止掛單。

2026-08-13有239檔mapped商品；Future-maker route為225 pass、5 known-fail、9 insufficient-support，Spot-maker route為226／4／9：

| Route | Core | Wide maker | Wide hedge cost | Known-fail | Insufficient |
|---|---:|---:|---:|---:|---:|
| Future Ask maker → Spot taker | 78 | 144 | 3 | 5 | 9 |
| Spot Bid maker → Future taker | 78 | 0 | 148 | 4 | 9 |

修正前後pass／known-fail／insufficient funnel不變；2308、2404、8299只由core移到future-maker的wide-maker與spot-maker的wide-hedge strata。另有87檔帶`eligible_q10_stress_flag`、11檔帶`eligible_median_stress_flag`；這些都只作歷史可用率warning。它們現在照正常spread／freshness／depth／activity分類並重新進入研究池。例如2303最新兩route都為`pass/core_candidate`，但保留q10 warning，表示少數日可能因價格帶、TrialMatch或資料缺口暫停，不表示價差波動不穩。

## 商品縮減

### 可重現的 80% pseudo core

`pseudo_validation_stable_core_products.csv`要求2026-07-01～08-13至少20個product-days，且兩條route各自至少80%日數屬於`core_candidate`。日期／市場別tick ladder修正後共有68檔；相較舊版73檔移除2308、2404、6274、8046、8299。完整清單與q50／q80／q95 tick geometry直接保存在CSV。這是retrospective pseudo-validation結果，表內明確標記`production_universe_approved=false`；實盤每日仍由當日`<D` snapshot決定，不硬編商品清單。

### 第一輪 raw replay：43檔 strict core

為進一步縮小昂貴raw replay，另外要求商品在May fine-tune、June confirmation、Jul-Aug pseudo三段中，兩條route都有至少80% coverage、90% pass與90% core share。v2有44檔通過；2330先隔離檢查Best-vs-L1 book語意，基準組保留以下43檔。2308與8299因修正後future tick geometry不再符合此凍結契約：

```text
1101 1513 1605 1802 2002 2301 2313 2317 2324 2344
2353 2371 2376 2382 2408 2409 2412 2449 2474 2603 2609
2610 2615 2618 2881 2882 2891 3006 3019 3035 3045 3105
3231 3260 3374 3376 3706 3711 5347 5483 5871 8039 9958
```

Jul-Aug嚴格90% core但未跨三段穩定的20檔作第二批extension；相較舊版移除2404與6274：

```text
1312 1326 2327 2356 2377 2441 2454 3034 3036 3037 3042
3702 4904 4919 4958 6147 6239 6278 8069 8150
```

### Wide-future 對照組

不能只研究窄 spread，否則無法回答寬 future spread 究竟是 maker edge 還是不可用流動性。第一批可選 10 檔在 pseudo 31/31 都 pass 且 31/31 屬 future-wide-maker：

```text
6005 4162 6245 6547 6121 4743 1722 5876 5457 5534
```

它們只用來估 future-maker fill／cancel／50 ms spot hedge與反向 future-taker 成本，不是 EV 排名。像 5274、3552 雖然 spread 更寬，但 pass rate 為 0，不應因「看起來價差大」而優先。

## 上下界與 tick 幾何

全期 rolling valid 商品日顯示：

| Boundary | Upper 中位 | Upper / future tick | 正負兩側都 >=1 future tick |
|---|---:|---:|---:|
| q50 | 7.99 bp | 0.445 tick | 3.41% |
| q80 | 16.73 bp | 0.904 tick | 33.6% |
| q95 | 29.55 bp | 1.57 ticks | 87.3% |

這些quantiles是「每個zero-crossing excursion振幅」的分位數，不是每日最高價差分位數；每商品日可能有上百段小excursion。Jul-Aug全241檔的causal tail診斷中，q95／q97.5／q98／q98.5／q99／q99.5仍約有6.17／2.95／2.36／1.78／1.21／0.64次latent reach／日。q95本身不算每天只出現一次的極端值。

下一版action surface會加入q97.5～q99.5作tail knots，但不以quantile或每日次數直接決策。Live端仍列舉合法tick prices，模型比較各價位的reach、fill、hedge、exit／overnight joint EV；實際週轉是sequential replay輸出。

68檔v2 pseudo core中，2026-08-13有66檔具完整boundary rows（3006與3037當日缺row）；這66檔的上下完整band中位數為q50 16.54bp、q80 31.21bp、q95 48.08bp。以68檔全cohort作保守分母，q50有5/68達一個完整band future tick、1/68達兩個；q80為63/68與7/68。q95仍只是tail diagnostic。

這表示 q50／q80 只是 latent probability aliases，不能直接當兩張獨立掛單。每個 epoch 必須用當下 fair 與反腿 executable quote反算合法價格、依 tick ladder round，再以 `route + stage + spread_pair_epoch + absolute_target_price` 去重；落到同價的 q50／q80 共用一筆 raw queue fact。

## 下一步 raw replay

1. 先跑43檔strict core，取得自然base-rate的fill、partial、retreat cancel與50ms hedge成本。
2. 再跑20檔extension，檢查研究結論是否只適用最穩定商品。
3. 加入 wide-future 對照組，分別估 future maker 與 future taker route；不可合併成一個 spread 效果。
4. Eligibility warning商品與其他商品使用同一套action surface；只在當下`eligible=false`時停掛，另按warning分層報告結果。
5. Actual fill後重算achieved basis，再估same-day exit、forced／aggressive exit與overnight branch；以pathwise net cashflow建EV，不把邊際`P(fill) × P(latent hit)`直接相乘。

Pass只代表資料與route support足夠，並不代表正EV；也不設定每天固定交易次數。正式決策會列舉合法tick actions，以fill／hedge／same-day exit／overnight的joint pathwise EV、capital-time與tail risk選擇，completed cycles/day只是sequential portfolio replay的結果。

## 產物

- `maker/data/walkforward/liquidity/daily_liquidity.parquet`
- `maker/data/walkforward/liquidity/rolling_liquidity_screen.parquet`
- `maker/data/walkforward/liquidity/latest_q50_route_screen.csv`
- `maker/data/walkforward/liquidity/pseudo_validation_route_stability.csv`
- `maker/data/walkforward/liquidity/pseudo_validation_stable_core_products.csv`
- `maker/data/walkforward/liquidity/complete.json`

所有 pseudo stability 表都含 target-day outcome，僅供 retrospective universe research；唯一可供 D 日使用的是當日 `<D` rolling screen row。

### Machine universe manifest

`universe_manifest_v2`會重新由`rolling_liquidity_screen.parquet`驗算May、June與Jul-Aug三段門檻，再與凍結的stable68／strict43／extension20 membership契約互相檢查。它只接受`rolling_liquidity_screen_v6_price_ladder`與`tw_stock_spot_v1_future_20260706_v2` lineage。來源bundle內每個declared artifact都先驗SHA-256；舊tick版本、重複product-day-route、名單漂移、cohort交集、2330混入execution清單，或6005不再符合31／31 wide-control條件時均fail closed。

輸出明示`retrospective_research_selection=true`、`selection_contains_target_day_outcomes=true`、`production_universe_approved=false`與`runtime_daily_liquidity_gate_required=true`。第一波45檔只等於43檔strict core加2303 pilot及6005 control；extension與其餘wide controls不會偷混入。

```bash
uv run python -m maker.src.quote_fill.universe_manifest_cli
```

主要輸出：

- `maker/data/walkforward/liquidity/universe_manifest_v2/research_universe_manifest.parquet`
- `maker/data/walkforward/liquidity/universe_manifest_v2/strict_core_43.csv`
- `maker/data/walkforward/liquidity/universe_manifest_v2/extension_20.csv`
- `maker/data/walkforward/liquidity/universe_manifest_v2/wide_future_controls_10.csv`
- `maker/data/walkforward/liquidity/universe_manifest_v2/first_wave_45.csv`
- `maker/data/walkforward/liquidity/universe_manifest_v2/first_wave_45_symbols.txt`
- `maker/data/walkforward/liquidity/universe_manifest_v2/complete.json`
