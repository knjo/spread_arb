# S0.5：查表基礎重驗

日期：2026-08-25

狀態：**完成；canonical development bundle 已發布，但不是 execution／EV 結論**

本文件回答開始 S1 前的三個基礎問題：盤中的中心價差是否可信、D 日開盤前可取得的 q50／q80／q95 對 D 日是否有預測性、以及完整 upper＋lower band 扣除已知逐腿費稅後是否仍有研究空間。S0.5 沒有 maker fill、B6 hedge、exit、carry 或 20M sequential cap，因此不能把本結果稱為可部署策略 baseline。

## 結論

**可以重新開始做足以供實盤設計參考的 execution research，但不能直接從本表宣稱已找到可部署策略。** 理由分三層：

1. **中心不是盤前凍住一路帶到收盤。** Frozen opening-5m 中心的 product-day equal MAE 是 41.452 bp；因果 EWMA120 是 9.750 bp，EWMA30 是 8.877 bp。盤中必須隨合法期現狀態更新中心。
2. **D-1 查表有很強的商品間排序力，但 raw q level 不是固定機率。** 六個 q×side cell 的逐日 cross-sectional Spearman 在 71／71 天全部為正，平均 0.615～0.673；然而 q95 reach 由 5 月約 8.4% 降到 8 月約 4.0%，rolling-60 對下降中的 excursion amplitude 有 lag。
3. **幾何值得進 S1，不等於已賺錢。** Nominal q80 full band 中位數 33.292 bp，扣同日已知費稅後中位數仍有 11.632 bp；q95 是 38.102 bp。q50 nominal margin 則是 -6.608 bp。合法 tick 向外取整會把條件捕捉 band 顯著放大，不能把這段 rounding excess 當免費 alpha；它同時會降低 touch／fill。

因此目前 lookup foundation 的正確定位是：

- 可作 S1 前的 **causal、可重現 incumbent baseline**；
- 不可把 q95 解讀成每日固定 5% 機率；
- 不可再用舊 q95 selector 在 S0.5 的 71 日 broad cohort 先選出 3,846 個重疊 product-days，再用該樣本證明七組 policy；舊 manifest 全部 72 日本身是 3,886 筆；
- S1 前須明訂 anchor 目標、level challenger 與共同 cohort，見「S1 前 handoff」。

## 固定範圍與方法

- Source calendar：2026-01-26～2026-08-13，共 131 sessions；489,216,000 causal 1 Hz rows、31,360 product-days。
- Full-60 primary：2026-05-05～2026-08-13，共 71 sessions；5 月 19 日、6 月 21 日、7 月 22 日、8 月 9 日。
- 2026-08-14 起 locked forward 完全排除，runner 只依 `sessions.txt` 的 131 個明名日期讀檔，不使用 daily partition glob。
- Anchor accuracy：每個合法 t 重新計算同一 freshness gate 下 `[t+30s,t+300s]` 的 future center；主表是 product-day equal，whole-Date bootstrap 以 Date 為重抽樣單位。
- Delayed reversion：另報每秒 occupancy 與 exact 300 秒 non-overlap lockout，避免把重疊時間點當 iid。
- Boundary calibration：只使用 `source_asof_date < Date` 的 rolling-60 snapshot。Product-day equal 是 primary；event-pooled 只作 audit。Hit／known miss／right-censored unknown 分開，沒有 observable excursion 的 product-day 不會消失。
- Geometry：七組共同使用同一 q-independent Spot-Bid D-safe cohort；只含 opening reference、合法 tick 與已知逐腿費稅。
- 所有輸出固定 `development_only=true`、`pristine_final=false`、`actionable_execution=false`、`ev_ready=false`。

## 1. 中心價差：靜態盤前中心不成立，因果盤中中心成立

Base sample 共 17,006 product-days、71 日，主模型結果如下：

| Anchor | Product-day MAE | p80 abs error | p95 abs error | Pairwise MAE vs EWMA120 | Anchor TV／basis TV |
|---|---:|---:|---:|---:|---:|
| EWMA30 | **8.877** | **14.537** | **28.495** | -0.813 | 0.1564 |
| Persistence | 9.023 | 15.313 | 30.511 | -0.749 | 1.0000 |
| Prior-seeded EWMA120 | 9.664 | 15.343 | 29.159 | -0.081 | 0.0629 |
| EWMA120 | 9.750 | 15.411 | 29.451 | 0.000 | 0.0635 |
| Rolling median 300s | 9.800 | 16.526 | 31.870 | +0.148 | 0.0608 |
| EWMA300 | 11.654 | 17.796 | 34.332 | +1.706 | 0.0344 |
| Expanding median | 16.555 | 27.383 | 42.937 | +6.982 | 0.0106 |
| Frozen opening 5m | 41.452 | 54.461 | 67.604 | +32.068 | 0.0000 |

Whole-Date bootstrap 的 MAE 也分開：EWMA30 是 8.884 bp，95% CI `[8.614, 9.146]`；EWMA120 是 9.759 bp，`[9.455, 10.083]`。EWMA30 在 5、6、7、8 月與所有 TOD／DTE bucket 都有較低 MAE，不是單月偶然。

但「預測未來中心」不是唯一策略目標。EWMA30 比 EWMA120 活躍約 2.46 倍；positive residual 的 300 秒 non-overlap delayed-reversion，EWMA30 在有移動時同向回歸率為 53.226%、每 signal 平均一致移動 1.381 bp，EWMA120 是 54.751%／2.018 bp。較慢 anchor 創造較平滑、較有回歸幅度的 residual；較快 anchor 則更準地跟隨短期中心。EWMA30 的 delayed mean 在 71／71 日仍為正，因此不是用 forecast accuracy 換成完全沒有回歸訊號。

昨日 exact-QuoteCode prior 在 14,437／17,006 product-days 達 84.89% coverage，seeded EWMA120 只把 MAE 改善約 0.086 bp，且仍需當日 EWMA fallback；證據不支持把「昨日／盤前中心」凍住整天。

Freshness sensitivity 也有限制：要求整個 30～300 秒未來窗都維持同一 strict gate 後，1000ms 樣本僅有 2,129,510 個 evaluable seconds、占當下 eligible 5.073%；再加 100ms leg skew 只剩 2,330，100ms age 只剩 328。後兩者只能當 coverage 警告，不能拿小樣本漂亮 MAE 選 anchor。

**Anchor 判讀：** 若 anchor 的正式目標是估計未來 30～300 秒中心，建議 EWMA30 升為 development primary，EWMA120 留作 incumbent control；prior-seeded EWMA120 不升格。這不代表可把 EWMA30 塞入現有策略：必須先以 EWMA30 residual 重建 rolling boundaries，不能把 EWMA120 的 q 距離直接套過去。

## 2. q 查表：排序很強，絕對校準會漂

Calibration 母體是 16,656 個 all-boundary-supported product-days、248 商品。名目 tail reach 分別是 q50 50%、q80 20%、q95 5%。

| q | Side | Product-day LB | UB | Complete-case | Predicted distance p50 | Realized completed q p50 |
|---:|---|---:|---:|---:|---:|---:|
| 50 | Negative | 51.566% | 52.361% | 51.218% | 7.662 bp | 7.878 bp |
| 50 | Positive | 51.620% | 52.382% | 51.274% | 8.010 bp | 8.176 bp |
| 80 | Negative | 21.889% | 23.368% | 21.507% | 16.478 bp | 16.091 bp |
| 80 | Positive | 21.754% | 23.196% | 21.372% | 16.991 bp | 16.422 bp |
| 95 | Negative | 6.221% | 8.278% | 6.007% | 30.053 bp | 26.824 bp |
| 95 | Positive | 6.068% | 8.087% | 5.872% | 30.382 bp | 27.277 bp |

Whole-Date bootstrap 的 conservative LB 95% CI 下緣仍全部高於名目值：q50 negative／positive 51.017%／51.181%，q80 21.172%／21.090%，q95 5.770%／5.589%。因此 rolling-60 整體偏淺；它不是 exact probability calibration。

更重要的是月度 level drift：

| q95 | 2026-05 LB→UB | 2026-06 | 2026-07 | 2026-08 |
|---|---:|---:|---:|---:|
| Negative | 8.493%→10.747% | 6.348%→8.271% | 4.965%→7.039% | 4.256%→6.171% |
| Positive | 8.429%→10.539% | 6.225%→8.096% | 4.746%→6.769% | 4.012%→6.172% |

q95 predicted distance 中位數從 5 月約 28.4～28.8 bp 升到 8 月約 30.8 bp，realized completed q95 卻由約 29.4～29.9 bp 降到 22.9～23.8 bp。q80 LB 也由 5 月約 25.3% 降到 8 月 18.1～18.4%。這是明確的 rolling-60 level lag，不是「表完全沒資訊」。

商品間排序恰好相反地很穩：

| q | Negative Spearman mean／median | Positive mean／median | 正相關日期 |
|---:|---:|---:|---:|
| 50 | 0.635／0.638 | 0.662／0.671 | 71／71、71／71 |
| 80 | 0.621／0.625 | 0.660／0.665 | 71／71、71／71 |
| 95 | 0.615／0.619 | 0.673／0.685 | 71／71、71／71 |

Primary 71 日 episodes 中 95.978% fully observed、2.761% left-censored、1.261% right-censored。q95 unknown 使 LB～UB 寬約 2.0～2.1 個百分點，相對名目 5% 很大；後續不得只報 complete-case。

**Boundary 判讀：** rolling-60 足以當 causal distance／ranking baseline，但 S1 不應把 raw q 名稱當固定機率。應保留 rolling-60 baseline，另加只用 D-1 以前資料的 level recalibration／短窗 challenger；精確 challenger 規格須在 S1 前凍結。

## 3. 七組共同成本幾何

Broad cohort 是 15,935 product-days、244 商品、71 日。Fixed 15／20／25／30 是上下兩側各該距離，所以 full nominal band 是 30／40／50／60 bp。

| Policy | Nominal band p50 | Tick-rounded band p50 | Nominal same-day margin p50 | Nominal margin > 0 | Rounded margin p50 |
|---|---:|---:|---:|---:|---:|
| q50 | 15.532 | 35.393 | -6.608 | 3,382／15,935 = 21.224% | 12.338 |
| q80 | 33.292 | 48.076 | 11.632 | 14,669／15,935 = 92.055% | 26.162 |
| q95 | 60.172 | 81.302 | 38.102 | 15,929／15,935 = 99.962% | 58.456 |
| fixed15 | 30.000 | 47.281 | 8.818 | 15,084／15,935 = 94.660% | 25.434 |
| fixed20 | 40.000 | 57.720 | 18.818 | 15,814／15,935 = 99.241% | 35.190 |
| fixed25 | 50.000 | 68.183 | 28.800 | 15,927／15,935 = 99.950% | 45.409 |
| fixed30 | 60.000 | 76.047 | 38.800 | 15,935／15,935 = 100.000% | 53.474 |

同日已知 reference cost 中位數約 21.16～21.22 bp。q50 的 nominal band 多數不足以跨過成本；q80／fixed15 是最接近成本邊界的壓力組；q95／fixed30 是高 margin、預期低 touch／fill 的另一端。這張表沒有理由先刪任何一組，七組仍應在共同 cohort 全跑。

Tick rounding excess 中位數為 13.686～19.318 bp，最大可達 95.426 bp。向外取整後 q50 的 positive share 會從 21.224% 跳到 93.078%，這不是模型忽然變準；只有真的 touch、maker fill、hedge、exit 完成後才捕捉得到。S1 排名不能用 rounded positive share 代替 realized execution。

## 4. Cohort：舊 q95 selector 不能再當七組共同母體的唯一答案

| Funnel | Product-days | 商品 | Target outcome 用於建 cohort |
|---|---:|---:|---|
| Full-60 target mapping | 17,006 | 250 | 否 |
| q50／q80／q95 全有 D-safe boundary | 16,656 | 248 | 否 |
| Spot-Bid broad D-safe | 15,935 | 244 | 否 |
| 舊 monthly q95 selector 與 broad 的重疊 | 3,846 | 119 | 否；只作 bridge |

舊 selector 在 S0.5 的 71 個 full-60 sessions 中只覆蓋 broad 的 24.136%（3,846 筆；原 72 日 manifest 為 3,886 筆）。若先用 q95／Spot-Bid outcome proxy 選這批 product-days，再比較 q50、fixed policy 或 Future-Ask route，結論必然 conditional on q95-selected sample，無法回答原本的全體政策問題。S0.5 已建立 q-independent broad cohort；是否把它正式升為 S1 母體是下一個需明訂的 handoff，不應默默沿用舊 manifest。

## S1 前 handoff：需要明訂，不需要再猜

S0.5 建議在開始大規模 execution replay 前確認四點：

1. **Anchor objective。** 建議以「未來 30～300 秒中心準確度」為正式 anchor 目標，EWMA30 升為 development primary、EWMA120 留作 incumbent control。若接受此口徑，S1 前先用 EWMA30 重建 excursion／rolling q 與 q-independent cohort，不能混用 EWMA120 lookup；目前 15,935 是 incumbent broad reference，重建後精確筆數須重新發布。
2. **Boundary level challenger。** rolling-60 保留 baseline；另凍結一個完全 `<D` 的 level recalibration 或短窗版本，主要回答 5→8 月 amplitude regime lag。它是 challenger，不得看 S1 outcome 再改。
3. **共同 cohort。** 原則上建議 S1 primary 使用 selected-anchor 重建後的 q-independent broad product-days；若保留 EWMA120，現成母體是 15,935。舊 3,846 只留 matched sensitivity。這會增加 replay 工作量，但才真正回答七組 policy／兩種 entry route。
4. **Expiry accounting。** 依使用者口徑，真正留到 expiry 的少量 paired residual 以當日現貨收盤價同時標 spot／future、basis=0 作 terminal accounting；明標非 executable fill，不算 same-day completion。正常 lower exit 仍是一腿 maker＋另一腿 taker；13:20 taker/taker 只可作風險 hard-flatten，不是 lower 主路徑。

完成這四項凍結、且 selected-anchor lookup 重建後即可開始 S1。這不是新增「研究通過門檻」，而是避免用不同 anchor 的 q 表、循環選樣或把 accounting mark 說成成交。

## Bundle、hash 與重現

Canonical bundle：[`foundation_revalidation_s05_20260825_v1`](../../data/walkforward/foundation_revalidation_s05_20260825_v1/)

- Runner code commit：`829ae1d43b99e35a97f7b3a77cf0e539df35027a`，run 時 `dirty=false`。
- `complete.json` SHA-256：`dd89f42c4c10ef38d4b32fce492749c3646416121e5f64286034ba22b73b149b`。
- Marker payload SHA-256：`5fa22e8030b9c35d704fc165f61b6fc147316ab455004b02b6cd7dc2bbb28e71`。
- Input inventory：670 個 full-content records、21,997,293,745 bytes；inventory SHA-256 `3d36778cc2f2d0956e58d38cec2bd33e95e61453209f6bbe4c790d9a0899c35e`。
- Censor overlay：8,708,199 episodes；calibration 99,936 rows；geometry 111,545 rows。
- 19 個 canonical invariants 全部通過；`--verify-only --verify-inputs` 已再次重驗輸入內容。
- Maker test suite：282 tests、48 subtests passed；S0.5 sources 另通過 Ruff 與 `py_compile`。

Build：

```bash
UV_CACHE_DIR=/tmp/s05_uv_cache \
uv run --project /home/kevin/Project/HFT --no-sync \
python -m maker.src.quote_fill.foundation_revalidation_runner
```

完整 input 與 published artifacts 重驗：

```bash
UV_CACHE_DIR=/tmp/s05_uv_cache \
uv run --project /home/kevin/Project/HFT --no-sync \
python -m maker.src.quote_fill.foundation_revalidation_runner \
  --verify-only --verify-inputs
```

131 個 daily partitions 沿用 `daily_latent_facts_v2_migrated_nonatomic` markers。Bundle 對本次實際讀取內容做 full SHA-256 並驗證新舊 basis／analysis gate／EWMA120 完全等價，但不能倒推那些 legacy daily partitions 當初是 atomic publication；這是為何本結果明標 development／non-pristine。
