# 掛單 Width 表：D−1 參數與 D 日驗證

更新日：2026-08-12

## 目的

本研究回答的是：不同商品的 tick、spread 與 basis 波動不同時，如何用前一交易日已知資料建立 diagnostic width grid，並驗證商品自適應 boundary prior 的可行性。

策略方向固定為先賣 basis。令：

```text
M_t = EWMA120 fair-mid basis
r_t = B_mid,t - M_t
```

正向 excursion 是 `r` 從非正值穿越到正值，直到回到 `r <= 0`。每段 excursion 只計一次最大發散，避免把同一波峰的每秒資料當成獨立樣本。

這裡的 crossing 與 path hit 都只是 **latent basis price path**：尚未重播 maker queue，不能解讀為 maker fill、成交後 50 ms hedge 或策略 EV。

## 每日使用方式

```text
D 日開盤前：用 D−1 的同一個 target QuoteCode 建立參數表
D 日盤中：參數表固定，當下 fair、spread、tick regime 與 target price 持續更新
D 日收盤後：D 日才進入下一次 expanding / prior update
```

- 不使用 D 日全天平均 spread 回填 D 日。
- 合約換月時仍取「D 日目標合約在 D−1 的資料」，不偷接 D−1 近月。
- 一個 tick 必須用 D 日 RefPrice 先估，再於盤中用實際 target price 精算；2303、2881 可能跨 100 元 tick 級距，不能永久綁商品常數。

## 商品摘要

八個 target dates、31 個 date×symbol pairs 的 D−1 表：

| 商品 | D日 Ref tickBP p50 | D−1 spot spread ticks | D−1 future spread ticks | D−1 TTBand p50 BP | 正向 excursion p50 / p80 / p95 BP | p80 換算 future ticks |
|---|---:|---:|---:|---:|---:|---:|
| 2303 | 25.60 | 1.01 | 1.10 | 33.09 | 7.37 / 13.64 / 18.26 | 0.77 |
| 2317 | 21.70 | 1.01 | 1.02 | 43.31 | 10.19 / 16.99 / 21.52 | 0.85 |
| 2603 | 25.25 | 1.03 | 1.12 | 50.06 | 7.81 / 16.09 / 22.81 | 0.68 |
| 2881 | 24.62 | 1.02 | 1.54 | 56.26 | 6.93 / 14.94 / 21.94 | 0.63 |

`spread ticks = spread BP / tick BP` 只描述 order book 寬幾檔；`tick BP` 保留一檔的絕對經濟尺度；`TTBand = B_buy_taker - B_sell_taker` 描述兩腿立即 crossing friction。三者不能合成一欄後丟掉原值。

## 隔日穩定性

D−1 excursion p80 對 D 日實現 p80：

| 指標 | 正向 excursion | 負向 excursion |
|---|---:|---:|
| 31-row diagnostic table 的 prior/current correlation | 0.839 | 0.862 |
| 隔日 p80 median absolute error | 2.15 BP | 1.80 BP |
| D / D−1 p80 median ratio | 0.944 | — |

上表有 31 rows；其中一組 prior 全為 null，所以 correlation／error 的有效 N 為 30。嚴格套用 prior gate 後剩 27 組，正向 correlation 為 `0.836`、median absolute error 為 `2.34 BP`、D／D−1 median ratio 為 `0.901`，主結論沒有改變。

所以 D−1 的正常發散幅度可作 D 日 prior candidate。這仍只是八日 pilot；商品別只有 7–8 點，2317 的正向相關為 `-0.045`，不可宣稱每個商品都已穩定。

若 prior p80 校準正確，下一日完整 excursion 超過它的比例理論上接近 20%，不是 80%。目前 pair-median 為 `16.94%`。

參數 promotion gate 暫定：D−1 `eligible rate >= 80%`，且正、負向 completed excursions 各至少 30 段。31 pairs 中 27 組通過；2303 僅 4／8 組通過，正式策略需要 pooled／hierarchical fallback。

## Diagnostic width grid

本版同時計算下列診斷格點；它們不會直接進 production optimizer：

```text
fixed:          10 / 20 / 30 / 40 BP
route tick:     1 / 2 / 3 × D日 Ref route-tick BP
prior TTBand:   0.25 / 0.50 / 0.75 × D−1 median TTBand
prior excursion D−1 positive p50 / p80 / p95
```

部分重點如下。`Potential entries/day` 是每個 pair-day 的中位 first-crossing 次數；不是 fill 次數。主診斷 path label 從 trigger 30 秒後才開始搜尋，以降低 signal 當下 `B_t` 的機械回歸影響。

| Width policy | Median width BP | Median future ticks | Potential entries/day | 60s 回中心 path hit | 300s 回中心 path hit |
|---|---:|---:|---:|---:|---:|
| Prior excursion p50 | 8.72 | 0.43 | 41 | 59.1% | 91.5% |
| Fixed 10 BP | 10.00 | 0.44 | 43 | 62.5% | 93.3% |
| Prior excursion p80 | 16.09 | 0.72 | 16 | 50.0% | 90.9% |
| Fixed 20 BP | 20.00 | 0.87 | 9 | 55.6% | 100.0% |
| Prior excursion p95 | 22.07 | 0.98 | 5 | 50.0% | 100.0% |
| Fixed 30 BP | 30.00 | 1.31 | 0 | 75.0% | 100.0% |
| Fixed 40 BP | 40.00 | 1.74 | 0 | 16.7% | 100.0% |

後兩列的 hit rate 不能解讀成比較好：30／40 BP 分別只有 15／9 個 pair-day 出現潛在 entry；能完整觀察到 300 秒的分母分別為 15 pairs／131 events 與 8 pairs／22 events。正式比較必須同時看 reach frequency、censor、fill 與持有時間。

本表的 `symmetric path hit` 仍只是會重疊的行情路徑診斷；不可拿來估完整日內週轉。後續已在 [CYCLE.md](CYCLE.md) 以獨立 position FSM 重建非重疊 latent cycles，沒有沿用這個 label。

## Nominal width 不等於實際掛價

用當下 fair 與反腿 taker 價轉成合法 tick 後，四個固定 width 的 pair-median：

| Nominal width | Future Ask effective BP / offset ticks | Spot Bid effective BP / offset ticks |
|---|---:|---:|
| 10 BP | 22.26 / 1 | 22.46 / 1 |
| 20 BP | 27.83 / 1 | 27.87 / 1 |
| 30 BP | 42.24 / 2 | 42.77 / 2 |
| 40 BP | 48.81 / 2 | 49.09 / 2 |

因此 10 與 20 BP、30 與 40 BP 經常落在相同 maker price。WP02 必須以 `rounded_target_price` 共用一筆 raw fill fact，再依 policy 判斷；不能把 nominal widths 當成四筆獨立訂單。兩條 entry route 的 rounding inequality violations 均為 0。

Geometry quantile 是所有 `analysis_eligible` 狀態的描述，另以欄位報告 strict RefPrice legal／passive rate，沒有事後只保留合法 row；壓力日寬 width 仍可能因價格帶而不可掛。

## 尚不能回答

- 潛在 crossing 是否真的在 maker quote 移動／gate 前成交。
- 成交量、queue ahead、partial fill、cancel race。
- Fill 後 50 ms taker VWAP 與 adverse selection。
- Maker fill 後 center、halfway-to-symmetric、symmetric exit 的 executable position cycles。
- 逐腿實際成交現金流、券商費用、依法匹配的現股當沖證交稅、期貨逐次交易稅、資金／庫存成本與 OOS PnL。

固定 BP／tick rows 只用來確認回歸曲線與 rounded geometry，不形成 WP02 shortlist。正式商品上下界與機率表見 [ADAPTIVE_BOUNDS.md](ADAPTIVE_BOUNDS.md)；maker fill、hedge 與 executable cycle 仍依序由 WP02、WP03、WP04 完成。

## Censor 與統計限制

- TrialMatch／RefPrice／book eligibility gap 與 session cutoff 會 censor excursion，不跨 gap 接續。
- Excursion p50／p80／p95 目前只由完整回中心的 excursions 計算；長而大的 right-censored excursion 可能被低估，因此同表保存 started／completed／censored。
- `reach_rate_all_started` 把 censored non-hit 留在分母，是保守下界；`reach_rate_completed` 有 completion selection bias。
- Outcome 是 30 秒後的 raw 1-second path touch，不是持續 5 秒的 robust convergence；簡單機率只使用能完整觀察到指定 horizon 的事件，提早 censor 的事件為 null。
- 本版尚未重建 age <= 1 秒的完整 sensitivity；所以不能宣稱 freshness-robust。
- 所有 summary 以 pair median 為主，避免高更新／高 freshness 商品完全主導 pooled rows。

## 產物與重跑

- `../../data/quote_width/daily_product_parameters.csv`
- `../../data/quote_width/product_parameter_summary.csv`
- `../../data/quote_width/next_day_parameter_validation.csv`
- `../../data/quote_width/validation_summary.csv`
- `../../data/quote_width/target_zero_crossing_excursions.parquet`
- `../../data/quote_width/potential_entry_events.parquet`
- `../../data/quote_width/width_policy_by_day_symbol.csv`
- `../../data/quote_width/width_policy_summary.csv`
- `../../data/quote_width/entry_route_geometry.csv`
- `../../data/quote_width/config.json`

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.table
```
