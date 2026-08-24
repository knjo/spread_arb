# 賣價差優先：M-1 選股、M 月交易的因果商品池

日期：2026-08-22  
狀態：development research；逐列選股因果，但門檻尚未經新期間 holdout，非正式 OOS／production GO

## 結論

原本固定 45 檔不能再當正式回測商品池。它是看完 May、June、Jul-Aug 三段後取交集，再回頭套到較早日期；雖然原始每日流動性欄位是 D-safe，商品名單本身仍含未來資訊。

新的表改成：

1. 只做賣價差方向：現貨掛 Bid 買進、期貨 taker 賣出，route 為 `spot_bid_future_taker`。
2. 月 M 的商品資格只使用完整 M-1 月資料。
3. M 的每個交易日再套當日可用、但只看至 D-1 的 route-specific liquidity gate。
4. 被月池移除但仍有庫存的商品只出不進，不因移除而強平。
5. 目前 8 月只有 9 個觀察日，因此絕不發布 2026-09 商品池；只有看到下一個月資料，或外部交易日曆明確證明月底完成，才把上月視為完整。

研究答案是「上月表現可用來篩下月商品」，而且訊號相當單調；但它比較像留倉風險 gate，不是獲利模型。寬市場 proxy 的 pooled 同日回落率由 `85.16%` 提到因果准入池的 `90.30%`，未同日回落事件由 `14.84%` 降到 `9.70%`。這些分母是 latent q95 upper events，不是商品數，也不是實際成交部位。真正 maker queue／50 ms hedge／frozen exit 的完成率會明顯更低，仍要 raw replay。

舊研究遺留的 `top 45` 已從 v2 正式選股規則與輸出移除。現在每月符合門檻幾檔就交易幾檔，不固定名額。

## 舊表的 leakage 在哪裡

舊 universe 先對三個已實現期間取交集，再組成 strict 43 + pilot 2303 + control 6005 = first-wave 45。原程式本身已標註：

- `retrospective_research_selection = true`
- `selection_contains_target_day_outcomes = true`
- `production_universe_approved = false`

但 execution CLI 接收一組固定 `symbols`，再建立 `sessions × symbols` 笛卡兒積。這會把事後知道穩定的 45 檔回灌到全部 60 日，正是上帝視角。

新表的主鍵不是一個固定 symbol list，而是：

```text
(effective_month, ValueCode)
        ↓ 再套 D-safe daily liquidity
(Date, ValueCode, QuoteCode)
```

因此不同月份可以是不同商品，且每列保留 `source_month_last_date`、boundary `source_asof_date` 與 liquidity `source_asof_date` 供因果 audit。

## 建議 proxy

### 1. 主 gate：上月「逐日 clustered」同日收斂下界

對每個商品日：

1. 用 D-1 q95 上下界。
2. 每個正向 q95 excursion 視為一次賣價差機會。
3. 若同日後續負向 excursion 也到達 q95 下緣，記為 dynamic full-band return。
4. 先在商品日內算這些事件的同日回落比例，再跨日計算平均與日間標準差。
5. 使用單側 80% 下界：`daily mean - 1.28155 × daily std / sqrt(signal days)`。

先按日聚合，是為了避免把同一日幾百個相關 excursion 錯當成幾百個獨立 Bernoulli 樣本。event-level Wilson 仍保留作診斷，但不再拿來准入。

目前 primary gate 固定為：

| 條件 | 門檻 |
|---|---:|
| 上月 D-safe liquidity pass days | >= 10 |
| 上月 q95 upper events | >= 20 |
| 上月有 upper signal 的日期 | >= 10 |
| 上月逐日同日收斂率單側 80% 下界 | >= 70% |
| 上月成功事件的保守等待上界 p90 | <= 3,600 秒 |

compact excursion 表沒有「首次碰到 q95」時間，因此不把 start-to-start 假裝成精確持有時間。現表保存嚴格區間：

- lower bound：`lower excursion start - upper excursion end`，最低截為 0。
- upper bound：`lower excursion end - upper excursion start`。

一小時 gate 使用較保守的 upper bound。

### 2. 結構 proxy：cycle-per-band

```text
min(positive_completed_per_session, negative_completed_per_session)
-------------------------------------------------------------------
             upper_distance_bp + lower_distance_bp
```

它衡量「相對於所需上下 band 寬度，歷史上每 session 能完成多少雙向 cycle」。在 905 個 source→target 商品月上，它對下月商品逐日同日回落率的 Spearman 相關為 `0.562`，是目前寬市場 proxy 中最高者。

不過這個選擇是看過 May-Aug 驗證後才形成，不能把它的同一段改善當成全新 OOS 證據。現行正式准入只使用預先寫死的 clustered LCB／等待時間門檻；`cycle-per-band` 應凍結為 challenger，在下一段新資料或 raw replay 比較。

### 3. 容量 proxy 要與收斂 proxy 分開

`upper events / pass day`、實際 fills/day、notional/day 適合估計可交易量，不適合單獨判斷會不會很快收斂。容量高但收斂差的商品，反而可能快速塞滿部位。

### 4. 不使用上月 PnL 選股

在既有 exact maker 子樣本中，上月 PnL 對下月的 persistence 約 `0.034`，幾乎沒有選股價值。先用收斂／留倉 proxy 控風險，再讓 exact replay 評估成本後 PnL，比直接追上月賺錢商品穩健。

## M-1 → M 驗證結果

下表的「同日率」是 latent dynamic full-band proxy，不是 maker fill-and-exit rate。2026-08 目標月目前只含 08-03 至 08-13 共 9 日，但它的名單完全來自完整 7 月。

| 目標月 | 全部可觀察商品 | 因果准入池 |
|---|---:|---:|
| 2026-05 | 86.33% | 92.59% |
| 2026-06 | 85.89% | 90.52% |
| 2026-07 | 84.42% | 89.34% |
| 2026-08* | 79.89% | 83.62% |
| pooled | 85.16% | 90.30% |

Pooled 補充：

| 指標 | 全部 | 因果准入池 |
|---|---:|---:|
| q95 upper events | 120,946 | 44,628 |
| 未同日回落 proxy | 14.84% | 9.70% |
| 商品日平均同日回落率 | 73.68% | 82.87% |
| 成功事件等待上界 p90 | 3,469 秒 | 2,602 秒 |

因果准入池相對全部樣本把未同日回落事件比例降低約 `34.7%`，但 8 月仍明顯惡化，表示商品篩選只能減少 regime risk，不能消除它。

### 上月 proxy 對下月的排序力

目標是下月每個「商品日」的平均同日回落率；相關係數使用 Spearman：

| 上月欄位 | 商品月數 | Spearman |
|---|---:|---:|
| cycle-per-band | 905 | 0.562 |
| 逐日 clustered 同日率 LCB80 | 903 | 0.499 |
| event-level Wilson LCB80 | 903 | 0.462 |
| event-level raw 同日率 | 903 | 0.454 |
| upper events / pass day | 905 | 0.420 |

排序也呈單調性：

| 上月分位 | clustered LCB：下月商品日率 | cycle-per-band：下月商品日率 |
|---|---:|---:|
| Bottom 20% | 55.15% | 54.49% |
| Q2 | 67.91% | 65.66% |
| Q3 | 69.19% | 71.26% |
| Q4 | 77.78% | 76.57% |
| Top 20% | 82.40% | 84.39% |

所以「用上個月判斷下個月某商品能不能交易」是有實證支持的；合理作法不是要求它精準預測每一筆，而是當月度准入 prior，再由每天 D-1 liquidity gate 更新。

## 與 exact maker 結果的交叉檢查

在舊固定 45 檔內，另以真正的 entry fill、50 ms hedge、frozen-lower exit 做過 M-1 因果重排。這只能驗證 outcome proxy 是否有用，不能替固定 45 的 universe leakage 洗白。

Jun→Jul 與 Jul→Aug pooled：

| 指標 | 固定 45 全部 | 用上月 exact 同日完成率篩選 |
|---|---:|---:|
| paths | 1,608 | 1,149 |
| 覆蓋率 | 100% | 71.46% |
| 同日完成 | 56.84% | 64.75% |
| 1 session 內完成 | 78.30% | 82.51% |
| 超過 2 sessions | 16.85% | 14.36% |
| 平均持有 sessions | 1.35 | 1.10 |
| completed-only net | 16.72 bp | 16.32 bp |

這支持「上月收斂行為可降低留倉」；同時也顯示它沒有提高每筆 PnL，甚至略降。因此選股 gate 的工作是控制 inventory duration，獲利仍由 q、queue fill、hedge cost 與 exit policy 決定。

## 新表內容

輸出根目錄：`maker/data/walkforward/monthly_product_selector_causal_v2_20260822/`

| 檔案 | 粒度 | 用途 |
|---|---|---|
| `monthly_product_metrics.parquet` | source_month × ValueCode | 上月 outcome、日群聚 LCB、等待界線、cycle-per-band、容量 |
| `monthly_membership.parquet` | effective_month × ValueCode | 因果准入與明確 source month |
| `daily_allowlist.parquet` | Date × ValueCode × QuoteCode | 月資格 + D-1 runtime liquidity；含 new-entry 與 exit-only flags |
| `daily_entry_manifest.csv` | 可進場 product-day | 正式因果 raw replay 輸入，共 3,886 product-days |
| `dynamic_full_band_events.parquet` | upper q95 event | latent target labels、等待上下界與兩套 source-asof provenance |
| `eligible_product_days.parquet` | eligible product-day | 每日因果 coverage audit |
| `monthly_oos_diagnostic.parquet` | target month × policy | 僅供事後評估，明標不得回灌 membership |

目前月資格與每日實際可進場檔數：

| 目標月 | 月資格 | 每日平均 active |
|---|---:|---:|
| 2026-05 | 41 | 38.25 |
| 2026-06 | 79 | 70.38 |
| 2026-07 | 59 | 55.36 |
| 2026-08 | 49 | 47.22 |

正式 manifest 覆蓋 2026-05-04 至 2026-08-13，共 72 個交易日、3,886 個 product-days、跨月聯集 119 檔。每月名單由前一個完整月獨立決定，不存在固定 45 檔。

## 尚未完成、不能誤讀的部分

1. Dynamic full-band 只證明 mid residual 先碰上 q95、後碰下 q95；沒有 queue ahead、成交、後撤取消、50 ms hedge price、交易成本或 frozen exit。
2. `90.30%` 絕不是預期實盤日內完成率；舊 exact maker 子樣本的可比數字只有約 `64.75%`。
3. 舊 generic execution CLI 仍以固定 `symbols` 建 sessions × symbols；本次新的 dynamic makerFill／hedge／portfolio runners 已改成直接讀 `daily_entry_manifest.csv` 的 `(Date, ValueCode)`，沒有再走該固定-symbol 路徑。未來若復用 generic CLI，仍必須先移除笛卡兒積介面。
4. 每日 allowlist 以 liquidity rows 為底。如果既有庫存商品當天連 liquidity row 都缺，portfolio consumer 必須把 holdings anti-join 回來並預設 `exit-only`；不能把「缺列」解釋成「沒有部位」。
5. 目前 q95/q80、70% LCB、1 小時與 10/20/10 support thresholds 都是在看過 Apr-Aug development results 後確認；所以現有改善只能叫 development diagnostic。接下來必須凍結，使用新月份 prospective holdout，或做嚴格 nested walk-forward，才可稱正式 OOS。

## 建議下一步

1. 已完成 manifest-driven 的 3,886 個因果 product-days approximate makerFill／+50 ms hedge／terminal path；下一步是把 approximate makerFill 換成 exact own-quantity／joint-volume replay。
2. 將目前規則凍結；後續新月份作 prospective holdout。若只用現有歷史，另建 nested walk-forward，不能讓 target month 參與門檻或 proxy 選擇。
3. 已完成 approximate maker 路徑的同日／跨日／到期比例、50 ms hedge slippage、成本後 PnL 與 fills/day；exact maker 版本仍待補。
4. 把移出月池但有舊庫存的商品納入 exit-only ledger。
5. 已完成同一組因果 manifest 的正常 carry 1,000／2,000／3,000／4,000／5,000 萬部位上限；13:00 積極出場版本仍待同樣重跑。

一秒掛撤容量的前置研究已改用本表全部 3,886 個 product-days 完成；不再由固定 45 檔外推。AB1/2 entry 的盤中合併峰值為 30 requests/s，低於現貨 100/s；13:00 集中撤單最高 199/s，需分兩秒 drain。完整定義與 future 5/s hedge batching 診斷見 `ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md`。

## 驗證

- 149,375 個 latent q95 upper events。
- event boundary／liquidity source-asof null：0。
- event source-asof >= target date：0。
- membership source/effective chronology violations：0。
- daily admission chronology violations：0。
- membership 中 `strong`／固定 45 欄位：0。
- 誤發布 2026-09：0。
- 15 個 selector／daily-fact／liquidity tests 全數通過。
- 完整重建 wall time 約 3.9 秒，peak RSS 約 0.85 GB；逐日讀取後不再一次展開全部 131 日 excursion 寬表。
