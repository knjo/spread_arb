# WP01 Pilot 結果：穩定中價 Basis

更新日：2026-08-12

## 決策

第一階段結論是：**保留 120 秒 causal EWMA 作為 WP02 的 provisional 中價 anchor，但 WP01 尚未通過預先設定的完整 validation gate。**

```text
M_t = 120 秒 half-life 的 causal EWMA(B_mid)
```

程式只在合法一秒 landmark 更新 filter；invalid gap 期間不重複餵入舊 basis。因此 120 秒是完整合法網格下的名目 half-life，遇到 gap 時實際上是 120 個合法 observations。

選它的理由是掛價中心的穩定度與誤差尾端，而不是已證明 residual 具有 alpha：

- 相較當前 `B_mid`，未來 30 秒至 5 分鐘中心的 MAE 由 5.54 bp 增至 6.34 bp，但 p80／p95 由 12.61／22.55 bp 降至 11.09／20.68 bp。
- 在連續合法掛價區段內，anchor 總變動量降至 `B_mid` 的 6.57%。
- 經 tick rounding 且排除超出 RefPrice 價格帶的 target 後，fair 單獨造成的 entry 改價由約 1.41 次／分鐘降至 0.080 次／分鐘，約少 94%。
- 原始 residual 圖形顯示大致回歸，但它和 outcome 共用 noisy `B_t`；移除這個機械效果後，策略關心的正 residual 在 fresh 樣本只有弱訊號。

因此可以並行進入 WP02 建立 fill／requote／50 ms hedge labels，但 `EWMA120` 仍只是 reference candidate。固定 20 bp width、production policy、以及 residual-based aggressiveness 都尚未核准。

## Pilot 範圍

固定 panel：

```text
2303, 2317, 2603, 2881
```

日期：

```text
20260128, 20260223, 20260318, 20260420,
20260609, 20260617, 20260720, 20260811
```

- 8 個日期、31 個有效 date×symbol 組合；20260617 的 2603 未通過當日 mapping／tradability，因此沒有強行補樣本。
- 原始期現事件只在 state machine 中逐筆掃描，落地為每 pair 每秒最多一列。
- 共 483,600 個一秒 landmarks；09:05 後且 future-center label 可評估者 421,737 筆。
- `C_t` 為 `[t+30s, t+300s]` 固定網格 future median，要求至少 90% 合法 coverage。
- `B_mid` 使用直接 L1；taker basis 與 route target 使用期貨明掛 L1 和衍生 Best quote 的可成交優價。
- EWMA 遇到不合法 landmark 時不輸出 anchor，也不把舊 basis 每秒重複餵入；下次合法 observation 從原 filter state 更新。

這是刻意選的跨月份與 corner-case pilot，不是隨機 OOS，也不是完整 2026 universe。

## p80／p95 誤差的比較對象

表中的誤差不是和昨日均價、成交價或 taker basis 比，而是：

```text
error_t = anchor_t - C_t
C_t = median(B_mid at each legal second from t+30s through t+300s)
```

只有 future window 至少 90% 合法 coverage 才評估。`p80=11.09 bp` 表示 80% 評估 rows 的 `abs(EWMA120_t-C_t)` 不超過約 11.09 bp；`p95=20.68 bp` 同理涵蓋 95%。它衡量掛單中線對未來 30 秒至 5 分鐘中心的偏差，不是交易 PnL 或 maker fill 後的 executable edge。

## Anchor 比較

| Anchor | MAE bp | p80 bp | p95 bp | `TV(anchor)/TV(B_mid)` | Future Ask 改價／分 | Spot Bid 改價／分 |
|---|---:|---:|---:|---:|---:|---:|
| 當前 `B_mid` persistence | 5.54 | 12.61 | 22.55 | 100.00% | 1.414 | 1.412 |
| EWMA 30s | 5.77 | 11.19 | 21.12 | 15.87% | 0.207 | 0.204 |
| **EWMA 120s** | **6.34** | **11.09** | **20.68** | **6.57%** | **0.081** | **0.080** |
| Rolling median 300s | 6.19 | 12.89 | 23.07 | 6.07% | 0.080 | 0.080 |
| EWMA 300s | 7.20 | 12.13 | 22.27 | 3.44% | 0.040 | 0.042 |
| Opening 5m static median | 19.00 | 33.72 | 50.91 | 0.00% | 0.000 | 0.000 |

EWMA 120s 是目前的 stability／tail-error 折衷點。Rolling median 的 TV 相近但尾端較差；EWMA 30s 的 MAE稍好，但 fair-only 改價約為 EWMA 120s 的 2.6 倍；static opening anchor 無法涵蓋日內中心漂移。

Quote churn 以 `W_open=20 bp` 計算。每秒固定當下反向 taker 價，只替換當前與前一秒 anchor，換算合法 tick 後比較 target；因此不會把反向 taker 腿移動錯算成 fair churn。此表也要求 target、previous target 與 counterfactual target 均在各自 RefPrice 價格帶內。

## Residual 診斷：不能視為獨立預測力

原始定義為：

```text
residual_t = B_t - EWMA120_t
raw_reversion_t = sign(residual_t) * (B_t - B_{t+300})
```

| `abs(residual)` | N | 平均 raw reversion bp | 未來不動 | 有方向移動時符合 residual 方向 |
|---|---:|---:|---:|---:|
| < 5 bp | 264,022 | 2.13 | 47.88% | 62.21% |
| 5–10 bp | 83,934 | 5.81 | 29.23% | 73.85% |
| 10–20 bp | 57,338 | 10.17 | 23.88% | 81.48% |
| 20–40 bp | 12,533 | 18.92 | 16.29% | 90.07% |
| >= 40 bp | 318 | 32.97 | 0.31% | 89.27% |

`abs(residual) >= 20 bp` 時，八個 pilot 日的平均 raw reversion 皆為正，範圍 15.24–26.93 bp；27 個樣本數至少 30 的 date×symbol 組合也全數為正。

但這不是獨立 forecast 證據：訊號與 outcome 共用同一個 noisy `B_t`，measurement noise 或 bid-ask microstructure 本身就會產生 residual 越大、表面回歸越強的圖形。這張表只提出 WP02 要檢驗的假說，不能直接拿來決定 width、成交勝率或期望值。

## Freshness 與 noise-robust 診斷

先要求訊號端與 exact `t+300s` 端點通過相同 freshness gate：

| 雙端點條件 | N | 往 residual 指示方向 | 不動 | 有移動時符合方向 |
|---|---:|---:|---:|---:|
| 09:05 後合法且 residual 非零 | 418,867 | 41.81% | 39.83% | 69.48% |
| 兩端 age <= 1,000 ms | 72,004 | 51.19% | 26.40% | 69.55% |
| age <= 1,000 ms 且兩端 skew <= 100 ms | 14,273 | 54.27% | 21.75% | 69.36% |
| 兩端 age <= 100 ms | 3,997 | 59.79% | 13.69% | 69.28% |

這降低了 stale `t+300`／as-of artifact 的疑慮，但 future freshness 是事後診斷條件，而且仍未移除共用 `B_t` 的機械效果。

所以再使用三時點診斷：訊號仍由 `t` 的 residual 決定，但 outcome 改成 `sign(residual_t) * (B_{t+30}-B_{t+300})`，不再使用 `B_t`。這只能回答「30–300 秒的變動是否符合訊號方向」，不是實際成交損益。

| 三時點 freshness | Residual 側 | N | 平均符合方向移動 bp | 有移動時符合方向 |
|---|---|---:|---:|---:|
| age <= 1,000 ms | 全部 | 49,296 | 1.98 | 56.11% |
| age <= 1,000 ms | 負 | 23,221 | 3.32 | 60.66% |
| age <= 1,000 ms | **正** | 26,075 | **0.80** | **52.35%** |
| age <= 1,000 ms、skew <= 100 ms | 全部 | 5,259 | 2.02 | 56.14% |
| age <= 1,000 ms、skew <= 100 ms | 負 | 2,492 | 3.89 | 61.96% |
| age <= 1,000 ms、skew <= 100 ms | **正** | 2,767 | **0.33** | **51.15%** |
| age <= 100 ms | **正** | 698 | **-0.80** | **50.00%** |

結果有明顯方向不對稱。使用者策略「basis 高時先賣期貨／買現貨」對應正 residual；在嚴格 freshness 下，這一側接近 50%，尚未證明有穩健預測力。負 residual 較強，但涉及先買後賣與現貨放空／庫存條件，不能混成同一個可執行結論。

## Gate 與資料 QC

- 20260223 為春節後延結算日，四個標的均正確選到到期日為當天的 B6，而非名目月份推算的 C6。
- 20260609 的 spot／future `ref_ok=false` 分別為 3,013／19,020 rows；20260720 為 14,703／13,234 rows。這些 funnel counts 可與缺狀態或 TrialMatch 重疊，不解讀為互斥的純 RefPrice rejection。
- `RecvTime` 全程保留 nanosecond；跨市場只做 backward as-of。
- RefPrice 嚴格使用 `0.91*Ref < price < 1.08*Ref`，等號排除。
- TrialMatch 回 formal 時同時比較 `RecvTime` 與 `ChannelSeq`，只有 transition 後的新正式 book 才重新開放。
- 期貨 taker 優價納入 `BestBidPrice/BestAskPrice`。20260128 的 2303 有 20 個 landmark 與直接 L1 不同，最大改善 0.10 元，basis 約 13.3 bp。
- 早期資料的 `RecvTime-TransTime` 共同 clock lag 可達 11–83 秒；研究使用實際可收到的 `RecvTime`，並保存兩腿 age／skew sensitivity。

## 尚未通過

- EWMA 120s 的 31 個 date×symbol 中，29 組 p80 誤差低於 20 bp，16 組低於 10 bp；pair-level p80／p95 中位數為 9.96／18.72 bp。因此固定 `W_open=20 bp` 仍不能直接全市場採用。
- 預先規則要求大 residual 的日級 block-bootstrap 95% CI 下界大於零；目前尚未完成，故 WP01 只能標為 provisional。正式 gate 應改用不共用 `B_t` 的 delayed outcome，不能只 bootstrap raw reversion。
- 昨日 prior 與 level-regime pilot 已拆分完成，見 [PRIOR_DAY.md](PRIOR_DAY.md) 與 [LEVEL_STABILITY.md](LEVEL_STABILITY.md)；尚未做完整 2026 walk-forward、final holdout、時段／DTE anchor、forward-only Kalman 與 bounded predictive adjustment。
- 尚未完成 maker fill-before-move、cancel race、50 ms hedge VWAP、費稅與完整部位回測。

WP02 可先固定保存下列介面，但不得把 residual 直接當已驗證 alpha：

```text
anchor_ewma_120s_bp
basis_mid_bp
residual_bp = basis_mid_bp - anchor_ewma_120s_bp
anchor_error_regime
book_age / leg_skew / TrialMatch / RefPrice eligibility
```

研究 width 時保留 `10／20／30／40 bp` 與 uncertainty-conditioned 版本；相同 rounded target 共用 raw first-fill fact，再以各 policy 的 `tau_move` 判定 fill-before-move。

## 可重跑產物

- `../../data/fair_mid/basis_landmarks_*.parquet`
- `../../data/fair_mid/fair_anchor_panel.parquet`
- `../../data/fair_mid/metrics_by_model.csv`
- `../../data/fair_mid/metrics_by_day_symbol.csv`
- `../../data/fair_mid/metrics_by_sample.csv`
- `../../data/fair_mid/residual_bins.csv`
- `../../data/fair_mid/residual_bins_by_sample.csv`
- `../../data/fair_mid/endpoint_freshness.csv`
- `../../data/fair_mid/endpoint_residual_bins.csv`
- `../../data/fair_mid/delayed_reversion.csv`
- `../../data/fair_mid/delayed_reversion_bins.csv`
- `../../data/fair_mid/quote_churn_by_model_route.csv`
- `../../data/fair_mid/landmark_audit.csv`
- `../../data/fair_mid/coverage.csv`
