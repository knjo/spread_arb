# Work Packages 與 Assignments

| WP | 內容 | 依賴 | 交付 | 狀態 |
|---|---|---|---|---|
| 00 | Data contract 與 hard gates | 無 | schema、quality funnel、tests | WP01 所需部分已實作並測試 |
| 01 | Stable fair-mid basis | WP00 | landmarks、metrics、結果文件 | Provisional candidate；validation 未完成 |
| 01B | Cross-product diagnostic grids | WP01 | D−1 parameters、excursions、next-day validation | 八日／四商品 latent-path pilot 完成；固定 BP／tick 不進 optimizer |
| 01C | Fixed-grid latent cycles | WP01B | non-overlap FSM、center／lower sensitivity | 八日／四商品 diagnostic 完成；非 execution shortlist |
| 01D | Adaptive asymmetric boundaries | WP01B、WP01C | D−1 safe snapshot、upper/lower reach 與 conditional reversion | 八日／四商品 latent pilot 完成；非 fill／EV |
| 02 | Quote episode 與 maker fill | WP00、WP01、WP01D | SpreadPair epoch、layered rounded-target orders、competing-risk labels | 取樣／撤單契約已凍結；raw replay 未開始 |
| 03 | 固定 50 ms hedge | WP00、WP02 | route cost labels | 未開始 |
| 04 | Action EV 與完整 cycle | WP01–03 | calibrated policy frontier | 未開始 |
| 05 | Portfolio replay | WP04 | OOS turnover／PnL／risk | 未開始 |
| 06 | Shadow calibration | WP02–05 | queue／latency calibration | 未開始 |

## WP01 執行順序

1. 建立跨月份 pilot 的 1 秒 causal landmarks。
2. 驗證價格縮放、近月 mapping、時間、RefPrice、TrialMatch 與 fresh-book coverage。
3. 比較 persistence、prior anchor、EWMA、rolling median 與 forward Kalman。
4. 評估 accuracy、stability、quote churn 及 residual mean reversion。
5. 補 block-bootstrap 並擴到完整 2026 walk-forward／holdout 後，才作正式通過判定。

Pilot 決策：WP02 暫用 120 秒 causal EWMA 作 provisional reference anchor，30 秒 EWMA 與 persistence 保留為控制組；固定 20 bp width 與 residual alpha 都未通過。

每次更改定義先更新對應 work package 文件；產物只寫入 `../data/<work_package>/`。
