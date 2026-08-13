# 昨日 Basis Prior 測試

更新日：2026-08-12

## 問題與結論

問題是：把昨日平均 basis 加入 EWMA120，是否能降低 anchor 對未來中心的誤差？

八日 pilot 的答案是：**只適合拿來初始化當日 EWMA，對 09:05–09:15 的 p95 有小幅改善；對全日 p80／p95 幾乎沒有實質改善。把昨日平均固定混入整天反而會增加 MAE。**

這是 leakage-safe 的 pilot 診斷，但仍不是完整 walk-forward／OOS 結論。

## Prior 定義

對每個 target date 與股票，使用前一個有完整期現資料的實際交易日：

```text
prior_mean
= mean(昨日 09:05–13:20 每秒合法 B_mid)
```

關鍵限制：昨日使用的是「今天選定要交易的同一張 QuoteCode」，即使它昨日不是近月，也不能偷換成昨日近月。昨日 RefPrice、TrialMatch 與 book eligibility 亦全部用昨日資料重建。

八個 target／prior date：

```text
20260128 <- 20260127
20260223 <- 20260211  # 春節長假，calendar gap 12 日
20260318 <- 20260317
20260420 <- 20260417
20260609 <- 20260608
20260617 <- 20260616
20260720 <- 20260717
20260811 <- 20260810
```

Prior 至少需覆蓋昨日合法一秒 grid 的 80%。31 個 target date×symbol 中有 27 組通過；4 組未通過，全部是 2303，其中 20260717 的目標合約沒有合法 coverage。

這裡的 80% 是和主模型一致的 legal standing-book coverage，不是 age <= 1 秒 coverage；部分低流動股期的 fresh coverage 很低。Fresh-only prior 必須在更大樣本另作 sensitivity，不能由本表直接推論。

## 比較方式

主要候選沒有 fitted parameter：在 09:00 前先放入一筆 `prior_mean`，再以今天每筆合法一秒 observation 更新 EWMA120。

```text
seeded_EWMA120_0 = prior_mean
seeded_EWMA120_t = EWMA120(today's legal B_mid observations)
```

同時保留三類控制組：

- 原本只使用今天資料的 EWMA120。
- 昨日 mean／median／尾盤 30 分鐘 median 全日固定不動。
- 每秒固定拉回昨日 mean 的 10%／20%／35%／50% blend；這些權重只是敏感度，沒有在 pilot 上選 production 最佳值。

所有模型只在相同的 27 組、相同 timestamp、相同 future-center label 上比較。因此本表的 current baseline 是 `10.99／20.60 bp`，不能直接和完整 31 組的 `11.09／20.68 bp` 混比。

## 結果

| 時段 | 模型 | MAE bp | p80 bp | p95 bp |
|---|---|---:|---:|---:|
| 09:05–09:15 | Current EWMA120 | 7.09 | 11.45 | 20.28 |
| 09:05–09:15 | **Seeded by prior mean** | **6.93** | **11.35** | **18.52** |
| 09:15–10:00 | Current EWMA120 | 7.92 | 12.95 | 22.70 |
| 09:15–10:00 | Seeded by prior mean | 7.92 | 12.95 | 22.70 |
| 10:00–13:20 | Current EWMA120 | 5.80 | 10.41 | 20.29 |
| 10:00–13:20 | Seeded by prior mean | 5.80 | 10.41 | 20.29 |
| 全日 | Current EWMA120 | 6.242 | 10.992 | 20.602 |
| 全日 | Seeded by prior mean | 6.235 | 10.982 | 20.535 |

Seed 的效果如預期快速消失：在每秒都合法的完整 grid 下，09:05 前吸收 300 筆今日 observations，昨日單一 seed 的名目權重約剩 17.7%；09:10 約剩 3.1%。實際遇到 eligibility gap 時，09:05 約為 17.7–31.1%，09:10 約為 3.1–5.5%。EWMA120 是 120 個合法 observations 的 half-life，不是 gap 中也持續衰減的 wall-clock 120 秒。

開盤 aggregate p95 改善 1.75 bp，但跨 pair 一致性偏弱：27 組中，MAE 改善 14 組、p95 改善 15 組；pair-level median 改善分別只有 0.15／0.26 bp。春節長假四組也只有兩組改善，不能將 aggregate 效果當成穩健規律。

把昨日平均當全日 static fair 明顯失敗：全日 MAE／p80／p95 為 `18.02／27.49／39.88 bp`。固定 10% blend 的全日 MAE／p80 為 `6.70／11.14 bp`，都比 current EWMA120 差；雖然 p95 小降至 20.52 bp，仍不是好的整體 trade-off。

## 決策

- 昨日 mean 可保留為開盤初始化候選，但不是全日中心；在 promotion 前仍需通過 prior 與 target 同時 fresh 的 sensitivity。
- 只有 prior coverage 合格才可 seed；缺值時退回 current-only EWMA120。
- 不採固定全日 blend，也不因八日 pilot 選 gamma。
- 下一次正式比較需使用完整 2026 expanding walk-forward，將 7–8 月凍結為 final holdout，並按長假、換月、DTE 與 prior freshness 分層。

## 產物

- `../../../data/fair_mid/prior_day/prior_summary.csv`
- `../../../data/fair_mid/prior_day/prior_metrics.csv`
- `../../../data/fair_mid/prior_day/prior_metrics_by_pair.csv`
- `../../../data/fair_mid/prior_day/prior_anchor_panel.parquet`

重跑：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.fair_mid.prior_day
```
