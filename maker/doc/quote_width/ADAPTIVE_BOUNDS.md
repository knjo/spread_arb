# 商品自適應上下界機率表

更新日：2026-08-13

## 決策

正式策略不使用固定 `10／15／20 bp`，也不把固定 `1／2 tick` 當成商品的永久參數。這些格點只保留作回歸形狀、coverage 與程式正確性的 diagnostic controls。

本文件的八日 pilot 以 D−1 同商品、同一個 D 日目標合約估計 latent boundary，而且正負側分開：

```text
M_t      = causal EWMA120 fair-mid
r_t      = B_mid,t - M_t
W+_q     = D−1 正向完整 excursion 最大 r 的 q 分位數
W-_q     = D−1 負向完整 excursion 最大 -r 的 q 分位數
U_t      = M_t + W+_q
L_t      = M_t - W-_q
```

這個 D−1 版本現在只保留作 baseline。全市場 production-like 研究改用 [WALK_FORWARD.md](../WALK_FORWARD.md) 的最近 60 個交易日 pooled excursion distribution，每日只以 `<D` 資料更新；更新演算法固定，不再把單一昨日當正式商品參數。

`p50／p80` 是用來描述與初始化距離格網的 prior landmarks；`p95` 只作 tail diagnostic。它們不是策略直接選擇的掛價，也不是常態分布的信賴區間。

本文件的機率只描述 1 秒 latent price path。它還沒有 maker fill、50 ms hedge、費稅或 EV，不能直接拿來下單。

## 商品上下界

以下只納入 D−1 `eligible >= 80%`，且正、負向 completed excursions 各至少 30 段的 pair-days。數字是通過 gate 的逐日參數中位數：

| 商品 | Valid pair-days | `W+` p50 / p80 / p95 BP | `W-` p50 / p80 / p95 BP | p80 `W+ / W-` future ticks |
|---|---:|---:|---:|---:|
| 2303 | 4 / 8 | 12.02 / 22.38 / 25.87 | 13.69 / 22.82 / 27.47 | 0.80 / 0.83 |
| 2317 | 8 / 8 | 10.19 / 16.99 / 21.52 | 8.39 / 16.08 / 21.47 | 0.85 / 0.83 |
| 2603 | 7 / 7 | 7.81 / 16.09 / 22.81 | 10.34 / 15.82 / 22.90 | 0.68 / 0.69 |
| 2881 | 8 / 8 | 6.93 / 14.94 / 21.94 | 7.22 / 15.18 / 20.18 | 0.63 / 0.67 |

這正是不能用單一固定 BP 的原因：不同商品、不同方向的 typical excursion 不一樣；而且 p80 通常仍不到一個 future route tick。實際掛價必須在盤中依 route、反腿可成交價與當下 tick ladder 轉成合法價格。

2303 只有 4 個 valid pair-days，必須標為 provisional，正式版要向 microstructure peer／global prior shrink，不能只相信商品 leaf。

## 下一日碰界機率

每個 center-to-center excursion 只計一次。`P(reach)` 是 D−1 boundary 在 D 日被觸及的 pair-day median；尚未觸界便被 gate／session censor 的 excursion 留在分母，所以是保守下界。

| 商品 | p50 `P(reach U) / P(reach L)` | p80 `P(reach U) / P(reach L)` | p95 `P(reach U) / P(reach L)` |
|---|---:|---:|---:|
| 2303 | 46.6% / 41.4% | 22.6% / 23.6% | 8.5% / 7.8% |
| 2317 | 40.4% / 50.3% | 12.9% / 20.5% | 4.3% / 5.4% |
| 2603 | 60.5% / 33.3% | 16.8% / 17.5% | 7.3% / 4.8% |
| 2881 | 44.4% / 46.8% | 16.6% / 23.4% | 5.8% / 8.1% |

這張表回答「市場一天內常不常供應這個 band」，不回答掛在該處會不會成交。尤其 `P(reach U)` 不是 `P(entry maker fill)`。

## 碰上界後能否到下界

下表以 p50 商品 boundary 建立互斥 position FSM：首次穿越 `+W+` 後鎖住部位，直到碰到持續更新的 `M_s-W-` 或被 gate／收盤 censor。為降低 crossing 當下 noise，從 entry 30 秒後才開始判定；這是 diagnostic sensitivity，不是 execution latency。

| 商品 | Entries；每日 p50 | 300s 到 `L` pair/event | 600s 到 `L` pair/event | 研究 cutoff 前到 `L` 下界 | Hold p50 | Basis capture / anchor drift p50 |
|---|---:|---:|---:|---:|---:|---:|
| 2303 | 97；20.5 | 47.6% / 67.0% | 78.4% / 86.5% | 97.9% | 239 s | 26.37 / 2.75 bp |
| 2317 | 221；22.0 | 66.5% / 76.8% | 82.9% / 89.0% | 99.1% | 145 s | 21.60 / 3.25 bp |
| 2603 | 138；17.0 | 62.1% / 62.8% | 82.8% / 84.6% | 98.6% | 204 s | 25.25 / 5.78 bp |
| 2881 | 143；13.0 | 51.6% / 63.8% | 79.8% / 83.6% | 97.9% | 295 s | 14.86 / 6.40 bp |

- Horizon event 分母只包含該時間前可判定的 cycles；pair median 與 event-weighted 分開呈現。
- Cutoff 前下界為 `completed / all entries`，censored 也留在分母，因此是保守下界；它不是加入 force-flat 後的整日成交機率。
- Dynamic lower 可能因 fair 移動而提早碰到；完成 cycles 中 basis capture 非正的比例約 1.4%–7.1%，所以 actual basis capture 與 anchor drift 必須並列。
- 較嚴格的 frozen-entry fair 對「回中心」有一致 support；但 frozen full-lower 的 p50／p80 在四商品都未過目前樣本／censor gate。因此目前較有證據的是回中心，不能宣稱固定 basis 完整跨越上下界已穩定。
- p80 dynamic full-lower 目前只有 2317、2881 通過 support；2303、2603 各只有 43、48 entries，現階段不發布四商品穩定 p80 機率。

這仍只是「已碰上界的行情路徑」條件機率。可交易版本必須改成 `P(exit | entry maker fill 且 50 ms hedge 完成)`。

## 如何接到進場、平倉與 EV

機率分層，不把 touch、fill、hedge 與稅務混成一個數字：

| 層 | 條件機率／結果 | 現況 |
|---|---|---|
| Latent supply | `P(amplitude >= W+ | positive excursion started)` | 本文件已產生 prior |
| Entry execution | `P(fill before requote/gate | rounded quote episode)` | WP02 raw replay |
| Entry hedge | `P(50ms hedge complete | maker fill)` | WP03 raw book walk |
| Position | `P(target exit / force-flat / overnight | hedged fill)` | WP04 position replay |
| Economics | 各互斥 branch 的實際 cashflow、費、稅與資金時間 | WP04–05 |

完整 EV 使用互斥且完備的 pathwise terminal outcomes：`EV(action|state) = Σ_o P(o|action,state) × E(net path cashflow|o,action,state)`。現金流一次納入所有實際 fills、partials、費、稅、資金與 emergency costs，避免用彼此相關的邊際機率相乘。詳細帳本規格見 [../04_BACKTEST.md](../04_BACKTEST.md)。

Branches 至少分 `target_exit_same_day`、`force_flat_same_day`、`overnight/unresolved` 與 `hedge_failure/emergency`。不能用 latent `P(hit L)` 代替當沖稅適用機率。

現股當沖優惠應依同一證券商、同帳戶、同營業日、同一合格股票實際買賣相同數量判定；force-flat 的同日反向成交也可能符合，隔夜／未配對量則不符合。這與期貨腿是否完成同一個 cycle 是兩件事。費率與資格一律放在 trade-date versioned cost profile；規則依據見 [TWSE 現股當沖制度](https://www.twse.com.tw/zh/products/system/day-trading.html)、[財政部當沖證交稅措施](https://www.mof.gov.tw/singlehtml/384fb3077bb349ea973e7fc6f13b6974?cntId=4493245d64e5422887a375921e889465) 與 [TAIFEX 期貨交易稅說明](https://www.taifex.com.tw/cht/9/tradersQAProducts)。

## Production action 如何產生

```text
D 日開盤前 product prior snapshot（state-conditioned challenger 待實作）
-> 盤中 causal fair、spread、volatility、freshness state
-> 每條 route 枚舉當下合法 maker tick prices
-> 枚舉合法 entry tick × exit tick 組合
-> 反算每個 action pair 的 effective W+ / W- 與 gross tick capture
-> 查詢／shrink entry fill、同日 exit、overnight與各 branch cost
-> 風控後選擇最高 OOS EV 的 action
```

也就是最終表不是「商品選 p50 還是 p80」，而是一張商品／causal state 的 action surface：

```text
row key:
    product, entry route, entry maker ticks/price,
    exit route, exit maker ticks/price, state, time horizon

values:
    P(entry fill), P(exit same day | hedged fill),
    P(carry overnight), fill/hedge/top-up/carry branch cashflows,
    joint net EV and support
```

Tick 差決定每個 terminal branch 能拿到的 gross capture，機率決定各 branch 權重；實作以逐路徑 joint cashflow估 EV，避免把相依的邊際機率天真相乘。p50／p80只幫助確認格網涵蓋正常與尾端發散，不限制 optimizer。

Raw replay fact 的核心 identity 是 `episode start + product + route + stage + rounded_target_price + qty + replay version`。Fair／boundary policy 是 many-to-one alias；多個 nominal thresholds 落到同一價格時只能共用同一筆 queue／fill fact。

目前八日資料只支撐 product-all 與少量靜態分桶 diagnostic。完整 product×spread×volatility×TOD×DTE 交叉會嚴重稀疏；目前程式只做到 state → product fallback，product support 也不足時輸出 null 與 `peer_or_global_required`。Peer／global shrinkage、盤中 causal state snapshot 及 ridge／hazard challenger 都仍待實作。

## 產物與重跑

- `../../data/quote_width/adaptive/adaptive_parameter_snapshot_by_day_symbol.csv`（唯一不含 D 日 outcome 的 WP02 input candidate）
- `../../data/quote_width/adaptive/adaptive_boundary_validation_by_day_symbol.csv`（含 D 日 outcome，只供回顧驗證）
- `../../data/quote_width/adaptive/boundary_probability.csv`
- `../../data/quote_width/adaptive/boundary_probability_by_state.csv`
- `../../data/quote_width/adaptive/adaptive_latent_cycles.parquet`
- `../../data/quote_width/adaptive/adaptive_reversion_by_day_symbol.csv`
- `../../data/quote_width/adaptive/adaptive_reversion_probability.csv`
- `../../data/quote_width/adaptive/adaptive_reversion_probability_by_state.csv`
- `../../data/quote_width/adaptive/latent_policy_frontier.csv`
- `../../data/quote_width/adaptive/config.json`

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.adaptive
```

除了 `adaptive_parameter_snapshot_by_day_symbol.csv` 與 manifest `config.json`，其餘所有產物（含 validation、cycles、reversion、probability 與 frontier）都使用 D 日結果，只能作 retrospective evaluation，不能回填成 D 日當下 action。實盤／walk-forward 每個 D 日只能 materialize 使用 `<D` 資料的 versioned snapshot；本 pilot 尚未完成 full-2026 expanding calibration 與 July／August locked holdout。

另外，boundary quantile 目前由 completed excursions 估計；大型長尾 excursion 較容易被收盤／gate right-censor，可能使界線偏窄。正式版需改用 time-to-level survival／competing-risk estimator並保留 unknown risk set。
