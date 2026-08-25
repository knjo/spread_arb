# Work Package 01：穩定中價 Basis

> 2026-08-25 canonical update：S0.5 已在 71 個 full-60 sessions 重驗。EWMA30 的 future-center MAE 8.877 bp，優於 EWMA120 的 9.750 bp，建議 EWMA30 作 development primary、EWMA120 作 incumbent control；使用 EWMA30 前必須重建 residual rolling boundaries。下列八日 pilot 與預先判定規則保留為歷史設計，正式結果與無 pass/fail 的現行研究定位以 [FOUNDATION_REVALIDATION_S05_20260825.md](quote_fill/FOUNDATION_REVALIDATION_S05_20260825.md) 及 [REWORK_PLAN_20260824.md](REWORK_PLAN_20260824.md) 為準。

## 研究問題

能否在時間 `t` 只用當時已知資訊，產生一個既接近未來可交易中心、又不會造成 maker 掛價頻繁跳動的 `M_t`？

不能只用未來 `B_mid` MAE 選模型。緊貼當前價格的追價器可能 MAE 很小，卻不適合作掛單中心。因此分成：

```text
Stable anchor A_t
= prior anchor + time/DTE seasonality + slow causal update

Predictive fair F_t^H
= A_t + bounded short-term adjustment
```

先驗證 `A_t`；完整 validation gate 通過後才把 dynamic adjustment 視為可採用模型。Execution label 可與 validation 並行，但不得把 provisional anchor 當 production fair。

## Pilot 取樣

- 固定使用 `2303／2317／2603／2881`，避免用當日成交結果挑選標的。
- 日期為 `20260128／20260223／20260318／20260420／20260609／20260617／20260720／20260811`，涵蓋一般日、結算日、延後結算與 RefPrice stress。
- 每 pair 每 1 秒最多一個 causal landmark；交易事件本身仍逐筆處理。
- 基準時段 09:05–13:20，另分開檢查開盤與尾盤。
- 主結果保留所有合法 standing book；另保存 spot／future quote age，與 100／250／500／1,000／5,000 ms 及 leg-skew sensitivity 並列。不能用 1,000 ms hard gate 把低流動股期幾乎全數刪掉。
- 遵守 [05_DATA_CONTRACT.md](05_DATA_CONTRACT.md) 的 RefPrice、TrialMatch 與 book gates。

## 候選 anchor

依序比較：

1. 當前 `B_mid` persistence：準確但不穩定的必要控制組。
2. Opening 5 分鐘 static median。
3. 30 秒、2 分鐘、5 分鐘 causal EWMA。
4. 5 分鐘 causal rolling median。
5. Expanding median。

Pilot 通過後才加入前日尾盤、train-only 時段／DTE seasonality 與 forward-only local-level Kalman；禁止 smoother。

Pilot EWMA 只以合法一秒 landmark 更新；invalid gap 不重複餵入 stale basis，所以秒數為完整合法網格下的名目 half-life。未來若要使用 wall-clock decay，需另列模型，不可靜默改動同一欄位。

後續 predictive fair 才比較 OU shrinkage、Ridge／GAM 與 quantile model；期貨不新增 feature pipeline。

## Labels

```text
C_t = future time-weighted median(B_mid, [t+30s, t+5m])
Y_t^H = future local time-weighted median around t+H
H = 30s, 1m, 3m, 5m

residual_t = B_mid_t - A_t
toward_anchor_H = sign(residual_t) * (B_mid_t - Y_t^H)
```

Label window 若遇到 TrialMatch、RefPrice gate、stale 或 session end，保存 failure／censor reason，不能事後回頭刪除當下 landmark。

## 評估面向

準確度：

- Bias、MAE、p80／p95 absolute error。
- 誤差相對候選 width 及換算成四條 route 的 maker ticks。
- 相對 persistence、prior-only 及 causal EWMA 的 paired daily uplift。

穩定度：

- `TV(anchor) / TV(B_mid)`。
- Fair 單獨造成的 rounded quote changes per minute。
- Quote lifetime 與一秒內反轉的 flicker rate。
- Fair-induced churn 與反向 taker 腿造成的 churn 分開報告。

中心與回歸性：

- Residual 分桶後，未來向 anchor 移動的幅度及機率是否單調。
- `|residual| >= width` 時的 mean reversion 及日級 block-bootstrap CI。
- Fresh-book／低 timestamp-skew 後是否仍存在；若消失，視為 stale-leg artifact。
- 另以 `sign(residual_t)*(B_{t+30}-B_{t+300})` 移除 signal／outcome 共用 `B_t` 的機械回歸，並將 residual 正負側分開。

## 預先判定規則

Anchor 至少需：

- `p80(|A-C|)` 小於採用的最小 open width；若先看 20 bp，低於 10 bp 理想，10–20 bp 灰區。
- 大 residual 時，往 anchor 移動的日級 95% CI 下界大於零。
- 回歸關係不集中於少數標的、月份或到期週。
- Fresh-book 子樣本仍成立。
- Fair-only quote churn 明顯低於直接用當前 `B_mid`。

結果解讀：

| 結果 | 後續 |
|---|---|
| Anchor 通過、dynamic 不通過 | 用慢速 anchor，width 吸收不確定性 |
| Anchor、dynamic 都通過 | 使用 bounded dynamic adjustment |
| 只有 persistence MAE 好 | 只是追價，不作 maker 中線 |
| Fresh-book 後回歸消失 | 判定為非同步報價假象，停止此方向 |

## 交付物

- `../data/fair_mid/basis_landmarks_*.parquet`
- `../data/fair_mid/metrics_by_model.csv`
- `../data/fair_mid/metrics_by_day_symbol.csv`
- `fair_mid/RESULTS.md`

## Pilot 決策

八日 pilot 已完成，詳細數據見 [fair_mid/RESULTS.md](fair_mid/RESULTS.md)。Provisional 決策：

- Reference anchor 使用 120 秒 causal EWMA。
- 30 秒 EWMA 與當前 `B_mid` persistence 保留為 accuracy／churn 控制組。
- 固定 20 bp width 未通過全市場採用；WP02 必須依 uncertainty 與 execution cost 分層。
- 預先要求的 block-bootstrap CI、完整 walk-forward 與 final holdout 尚未完成，因此 WP01 不標記正式通過。
- 此決策只允許並行 quote-fill 研究，不代表 residual alpha、production 或完整 2026 OOS 通過。
