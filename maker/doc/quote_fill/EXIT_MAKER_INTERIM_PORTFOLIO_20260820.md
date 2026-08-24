# Exit Maker 交易量／隔夜曝險／P&L interim snapshot

> 狀態：**analysis-only、非 equity curve、非 production EV**。這份資料把仍在執行的正式 same-day replay 鎖定在 `2026-08-20T15:13:10+08:00` 的前 2,005 個 completion markers；正式 root 沒有被修改，後續新 partitions 也不會進入本 snapshot。

## 凍結範圍與可重現性

- Entry cohort：2,687 個正式 product-days；snapshot 取 entry manifest 的嚴格前綴 2,005 份，最後一份是 `20260723/3260`。
- Marker inventory SHA-256：`19d6f32f5a4f82def0868256246c5843ce0ad8c38bab5e95a38016423b63ee85`。
- 日頻表只使用完整跑完的 44 sessions（`20260520`～`20260722`）；尾端尚未完整的 `20260723` 明確排除，得到 `44 × 12 = 528` rows。
- Entry 分母只納入 `full_fill=true` 且 50 ms hedge label 已觀察、可執行的 alias-local entry。Snapshot 有 86,374 個 established entry q-aliases、64,436 個 physical entry dependencies、345,496 個 `Center/Lower × 兩條 exit route` alternative policy cells。
- 每個 position 的數量均驗證為 `2,000 spot shares = 1 futures contract × contract_size`。現貨 book 原生數量是 1,000 股一張，轉成 shares 後才輸出。

完整包在 [exit_maker_interim_portfolio_snapshot_20260820_151310](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310)：

- [input_marker_inventory.csv](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/input_marker_inventory.csv)：2,005 個 marker、entry action與position artifact hashes；
- [daily_metrics.parquet](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/daily_metrics.parquet)／[CSV](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/daily_metrics.csv)：528-row日頻資料；
- [policy_summary.csv](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/policy_summary.csv)：12格彙總；
- [entry_route_daily_metrics.parquet](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/entry_route_daily_metrics.parquet)／[CSV](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/entry_route_daily_metrics.csv)：`44 sessions × 24 entry-route-specific cells = 1,056 rows`；
- [entry_route_summary.csv](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/entry_route_summary.csv)：24格 route-split 彙總；
- [portfolio_diagnostics.png](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/portfolio_diagnostics.png)：三面板圖；
- [complete.json](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/complete.json)：逐檔 rows／columns／schema／bytes／SHA-256 與 safety flags。

## 每日建倉量

同一個 q 的 entry positions 會被四個替代 exit policies 共用，因此下表每個 q 只列一次，不能再乘四；q 之間也可能共用 raw physical entry，不能跨 q 加總成實際 portfolio volume。更重要的是，這張 12-cell 主表把兩條互相獨立的 entry routes 加總，只是 descriptive upper envelope（`entry_routes_pooled_descriptive_only=true`），不能當成單一可執行 portfolio；實際 route-split 數字請用 1,056-row 日表與 24-row 彙總。

| Entry q（兩 entry routes pooled、不可相加為 portfolio） | 日均 positions | 日均 spot shares | 日均 futures contracts | 日均累積新倉 one-way spot-leg entry-price notional | 單日最大同口徑 notional |
|---:|---:|---:|---:|---:|---:|
| 50 | 1,077.25 | 2,154,500 | 1,077.25 | 421.99m TWD | 965.20m TWD |
| 80 | 591.95 | 1,183,909 | 591.95 | 232.14m TWD | 608.86m TWD |
| 95 | 277.09 | 554,182 | 277.09 | 106.23m TWD | 318.68m TWD |

把 pooled 數字拆回各自 entry route 後，日均候選量如下；每列仍被四個 alternative exit policies 共用，q 之間仍不可相加：

| Entry q | Entry route | 日均 positions | 日均 spot shares | 日均 futures contracts | 日均 one-way spot-leg entry-price notional |
|---:|---|---:|---:|---:|---:|
| 50 | `future_ask_spot_taker` | 447.18 | 894,364 | 447.18 | 160.76m TWD |
| 50 | `spot_bid_future_taker` | 630.07 | 1,260,136 | 630.07 | 261.23m TWD |
| 80 | `future_ask_spot_taker` | 270.91 | 541,818 | 270.91 | 95.43m TWD |
| 80 | `spot_bid_future_taker` | 321.05 | 642,091 | 321.05 | 136.71m TWD |
| 95 | `future_ask_spot_taker` | 137.55 | 275,091 | 137.55 | 47.02m TWD |
| 95 | `spot_bid_future_taker` | 139.55 | 279,091 | 139.55 | 59.21m TWD |

這裡與後文所有 `notional` 都只表示「股數 × entry spot price」的 one-way spot-leg entry-price notional。EOD 欄只對該分類認定仍 open／unresolved 的 positions 加總；它不是 EOD mark、不是現貨加期貨的 two-leg gross exposure、不是 futures margin，也不是 capital requirement。新倉累積口徑最接近 `backTest.py` 的 `CumulativeEntryNotional`，但不是 intraday concurrent peak capital。現有 replay 是 independent alternatives，沒有共同資金 cap／joint volume allocation，不能宣稱這就是實盤資金需求。

## Nominal 隔夜部位與 strict 未解曝險

下表是每天 position 數的 `p50 / p90 / max`。`Future M` 是 `future_bid_spot_taker`；`Spot M` 是 `spot_ask_future_taker`。

| q | Rule | Future M nominal carry | Spot M nominal carry |
|---:|---|---:|---:|
| 50 | Center | 100.5 / 190 / 362 | 92 / 179 / 245 |
| 50 | Lower | 141 / 238 / 379 | 133 / 236 / 587 |
| 80 | Center | 50.5 / 109 / 192 | 50.5 / 98 / 136 |
| 80 | Lower | 68 / 121 / 191 | 72 / 141 / 303 |
| 95 | Center | 20.5 / 40 / 82 | 21 / 37 / 68 |
| 95 | Lower | 22.5 / 42 / 94 | 26 / 54 / 143 |

Nominal EOD carry one-way spot-leg entry-price notional的日均範圍為：q50 `33.58m～52.35m`、q80 `17.51m～28.78m`、q95 `6.74m～10.64m` TWD；各格 exact p90／max 在 `policy_summary.csv`。它是「positions classified open at EOD」的 entry-price 診斷值，不是 EOD 市值。

這些數字不能當風控上限。沒有 cancel ACK 時，strict 無法證明已 flat 的部位很多：strict unresolved 每日 p50 約為 q50 `954～963`、q80 `503～505`、q95 `221～225` positions，幾乎貼近當日建倉量。圖的第二面板因此同時畫 nominal carry 與 strict unresolved，而不是只展示較小、較樂觀的 nominal 數字。

## Completed-only 現金流敏感度

下表只把 V0 nominal 同日完成的四腿 cashflow 相加；未完成、cancel-race unknown與跨日 cashflow全部排除，沒有補零。`−19 bp` 只是對完成路徑套用的非價格成本敏感度，並非完整 fee／tax／financing／emergency-close成本。

| q | Rule | Exit route | Completed gross | Completed gross − 19 bp |
|---:|---|---|---:|---:|
| 50 | Center | Future M | +21.623m | −8.213m |
| 50 | Center | Spot M | +19.613m | −10.822m |
| 50 | Lower | Future M | +28.865m | +2.179m |
| 50 | Lower | Spot M | +26.475m | −0.595m |
| 80 | Center | Future M | +17.275m | +0.872m |
| 80 | Center | Spot M | +15.706m | −0.778m |
| 80 | Lower | Future M | +23.011m | +9.278m |
| 80 | Lower | Spot M | +21.027m | +7.574m |
| 95 | Center | Future M | +10.510m | +3.005m |
| 95 | Center | Spot M | +9.812m | +2.266m |
| 95 | Lower | Future M | +13.688m | +8.076m |
| 95 | Lower | Spot M | +12.260m | +6.929m |

![Analysis-only portfolio diagnostics](../../data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310/portfolio_diagnostics.png)

第三面板刻意寫上 `NOT EQUITY CURVE`：正斜率只說明「同日成功且已定價」子樣本的累計 cashflow；它沒有把尚未成功的 terminal loss、隔夜成本或 strict unknown 放回來。

## 與 `src/research/backTest.py` 的對照

實際檔名是 case-sensitive 的 `src/research/backTest.py`。本 snapshot 只沿用它的圖表語言，不把資料硬塞進原 engine：

| BackTest 概念 | Snapshot 對應 | 是否可當正式績效 |
|---|---|---|
| `CumulativeEntryNotional` | `cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd` | 只可作 pooled candidate throughput diagnostic；不是 capital |
| `Overnight_Market_Value` | nominal `nominal_eod_open_one_way_spot_leg_entry_price_notional_twd`；另列 strict `strict_unresolved_eod_one_way_spot_leg_entry_price_notional_twd` | 只是 open-position entry-price notional，不是 EOD mark／gross exposure／margin |
| `PnL`／`CumPnL` | `completed_gross_twd`／`completed_after19_twd` 的 completed-only cumulative line | **不可**；不是 equity curve |
| Win rate／MDD／Sharpe | `null`，`strategy_metric_status=suppressed_cross_terminal_and_full_costs_incomplete` | **刻意不算** |

`BackTest(hasOvernight=True)` 使用 `nextDayOpen` 強制出場，與目前的 frozen Center／Lower Maker 每日續掛不同；它的預設 `fee=19.3` 也不是本研究的 19 bp sensitivity。要產生正式 win rate／MDD／Sharpe，必須等 cross-session terminal labels與完整成本 profile完成，並一次只選一個 frozen policy及明確 portfolio allocation，不能把12格 counterfactual alternatives相加。

## 重跑與驗證

```bash
env MPLCONFIGDIR=/tmp/exit_maker_mpl_cache_20260820 \
  /home/kevin/Project/HFT/.venv/bin/python \
  -m maker.src.quote_fill.interim_portfolio_snapshot \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --output maker/data/walkforward/exit_maker_interim_portfolio_snapshot_20260820_151310 \
  --marker-count 2005 \
  --snapshot-time 2026-08-20T15:13:10+08:00 \
  --marker-inventory-sha256 19d6f32f5a4f82def0868256246c5843ce0ad8c38bab5e95a38016423b63ee85 \
  --last-key 20260723/3260 \
  --cost-sensitivity-bp 19
```

已存在的 bundle 用同一 module 加 `--verify-only` 可重新核對 exact artifact set、bytes、SHA-256、rows／columns／schema、CSV／Parquet logical equality、config／marker-inventory identity、pooled-vs-route rollup、PNG readability，以及每一個 safety／semantic flag。輸出 marker明列 `analysis_only=true`、`strategy_defensible=false`、`pathwise_ev_ready=false`、`cross_session_terminal_complete=false`、`full_cost_profile_complete=false`、`joint_volume_allocated=false`、`unresolved_cashflow_imputed=false`、`alternative_policy_rows_additive=false`、`entry_routes_pooled_descriptive_only=true`、`formal_roots_mutated=false`、`completed_only_curve_is_equity_curve=false`，且 notional 的四個否定旗標明示它不是 EOD mark、two-leg gross exposure、futures margin 或 capital requirement。
