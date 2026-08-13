# EWMA Level 與 Basis 不穩定度

更新日：2026-08-12

## 問題與結論

問題是：EWMA120 的 basis 絕對水位很高或很低時，是否因其他套利者參與而更不穩定？

目前結論分成三層：

1. **直接看全市場 `abs(EWMA120)`，沒有「絕對值越大越不穩」的單調關係。**
2. 控制 date×symbol×30 分鐘時段後，局部最高與最低端呈 U 型不穩定；但這是使用整個 block 排名的 descriptive diagnostic，不可直接作即時 feature。
3. 較有希望且當下可計算的 provisional uncertainty feature 是 `abs(EWMA120-EWMA300)`；pilot 中差距越大，future-center 誤差明顯越大，但 cutoffs 與關係仍是 pooled in-sample，尚未通過 walk-forward／OOS。

價格資料本身無法識別「是哪一類套利者造成」。目前只能證明某些 regime 較不穩，不能把機制歸因給其他市場參與者。

## 全市場絕對水位

依 `abs(EWMA120)` 的 pooled quintile：

| 絕對水位區間 | p80 error bp | p95 error bp |
|---|---:|---:|
| 0–5.50 bp | 6.05 | 19.61 |
| 5.50–17.39 bp | 12.59 | 22.77 |
| 17.39–29.04 bp | 11.79 | 22.21 |
| 29.04–44.30 bp | 10.94 | 20.25 |
| >=44.30 bp | 10.25 | 20.25 |

最高水位的誤差沒有變差。這個 pooled 表混入商品 carry、DTE、日期與 tick-size 差異，因此不能用單一絕對 bp threshold 決定 width。

## 同標的、同時段的局部高低端

先在每個 date×symbol×30 分鐘 block 內將 EWMA120 排名，再比較 quintile：

| Block 內位置 | p80 error bp | p95 error bp | Median `abs(B[t+300]-anchor_t)` bp |
|---|---:|---:|---:|
| 最低 20% | 13.06 | 23.56 | 8.25 |
| 中間 20% | 9.56 | 19.59 | 4.14 |
| 最高 20% | 12.02 | 22.31 | 6.66 |

相對中間 quintile，低端 p80 較差出現在 27／31 pairs，高端較差出現在 26／31 pairs。局部兩端確實較不穩，而且低端比高端更明顯。

但 block rank 使用完整 30 分鐘分布，只是研究診斷。即時策略應改用只依賴過去資料的 prior-day／DTE／time-of-day expected basis z-score，不能在當下知道自己是整個未來 block 的最高 20%。

## Fast／slow anchor disagreement

定義完全 causal 的 state：

```text
gap_t = abs(EWMA120_t - EWMA300_t)
```

| Gap quintile | Gap 上界 bp | p80 error bp | p95 error bp | Median `abs(B[t+300]-anchor_t)` bp |
|---|---:|---:|---:|---:|
| Q1 | 0.49 | 6.44 | 18.03 | 0.46 |
| Q2 | 1.37 | 9.71 | 19.74 | 4.75 |
| Q3 | 2.48 | 10.27 | 19.81 | 5.45 |
| Q4 | 4.30 | 12.36 | 20.76 | 6.90 |
| Q5 | >4.30 | 14.40 | 25.35 | 8.93 |

這比 absolute level 更值得進一步測試為 width／size uncertainty：Q5 的 p80 比 Q1 高約 8 bp，p95 高約 7.3 bp。不過本表只要求 current row 合法；future-center label 沒有要求每一秒 age <= 1 秒，且高度重疊的 1 秒 rows 不是獨立樣本。正式 policy 前需重建 freshness-matched labels 並做 walk-forward／OOS。

## 不等於可預測套利收斂

為避免 signal 和 outcome 共用 `B_t`，用 `EWMA120-EWMA300` 在 `t` 的方向，檢查 exact `B[t+30]-B[t+300]`；並要求三時點的兩腿 age 都不超過 1 秒。

Gap 最大的 Q5：

| Gap 方向 | N | 平均符合 slow anchor 方向 bp | 有移動時符合方向 |
|---|---:|---:|---:|
| 全部 | 13,314 | 2.75 | 57.04% |
| EWMA120 < EWMA300 | 3,617 | 6.27 | 72.48% |
| **EWMA120 > EWMA300** | **9,697** | **1.43** | **52.30%** |

`EWMA120 > EWMA300` 代表 fast-over-slow 的相對 high side，不必然等於絕對 basis 很高。它雖然更不穩定，但方向命中只約 52%，所以不能把「不穩定」直接翻譯成「會收斂、應積極賣」。

## 策略使用方式

- 將 `abs(EWMA120-EWMA300)` 保留為 uncertainty model 候選；只有通過 freshness-matched walk-forward／OOS 後，才允許 gap 大時調寬 width、降低 size 或提高最低 EV gate。
- 局部 level extreme 待 prior-day／DTE／time-of-day causal baseline 完成後再轉為正式 feature。
- 不用 absolute EWMA bp 設跨商品共用 threshold。
- 若要驗證「其他套利者造成」，WP02／WP03 必須同時看到期現 signed order flow、queue depletion、maker fill、回補速度與 50 ms hedge 行為；只有價格回歸不足以識別參與者。

## 產物

- `../../../data/fair_mid/level_stability/global_abs_level.csv`
- `../../../data/fair_mid/level_stability/local_level_quintiles.csv`
- `../../../data/fair_mid/level_stability/local_level_pair_comparison.csv`
- `../../../data/fair_mid/level_stability/fast_slow_gap.csv`
- `../../../data/fair_mid/level_stability/fast_slow_gap_direction.csv`

重跑：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.fair_mid.level_stability
```
