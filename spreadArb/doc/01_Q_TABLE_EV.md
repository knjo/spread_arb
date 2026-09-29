# Stage 1：Q 表與 EV（2026-09-17 定案版）

Q 表只回答一件事：**商品價差怎麼動**——現在在水位 e，多久會到水位 x。它只用市場資料（1 Hz 期現價差序列），
與 Stage 2 的執行事實（成交、劣化、hedge）無關。滑價、費用、hurdle 是設定值；EV 把 Q 與設定值合成「掛不掛、出場掛哪」。
Stage 2 點位表再依 EV 選出的價差 filter 出「不撤單可吃到」的點；Stage 3 回測；回測的「實現 − 預測」回頭修設定值。

maker 現行 `q_model.py` 把出場寫死在 `anchor − 5`、統計我方成交後的結局；本線不沿用它的 hazard 結構，只沿用資料契約、桶邊界與成本量測值。

## 1. 三個原則

1. **Q 是市場價差移動的機率，只用確定出現過的價差。** 狀態 e 取 tape 上真的出現的水位，到達 x 也是 tape 上真的到；
   不用假設價位、不混入我方成交或 hedge。市場不分 S1／S2，所以 Q 沒有 stream 鍵。
2. **逐日往下估。** 今天 `C0`、明天 `C1` 直接估；第 2 天起用一個 pooled 的每 session 到達率；剩餘質量到期歸 0。
3. **設定值手填 → 迭代。** `d_in`、`d_out`、費用、`ev_min`、hurdle 是 config，第一版填 maker 實測；每輪只動一個，看 Stage 3「實現 − 預測」是否縮。

## 2. 座標

```text
anchor_t   = basis_mid 的 120 s causal EWMA（決策時凍結）
scale_D−1  = 該商品 D−1 以前 trailing 20 日 (basis_mid − anchor) 的 (q95 − q50)     # 只用 < D
s_t        = (basis_mid_t − anchor_t) / scale_D−1                                   # 狀態序列（Q 的條件）
r_t        = (basis_buy_taker_t − anchor_決策時) / scale_D−1                         # 到達序列（出場觸發用的可執行 basis，anchor 凍結）
e          = 決策時的 s_t；x = 出場水位（同單位）；B_x = anchor + x · scale_D−1（bp）
```

狀態用 mid、到達用 buy-taker：出場觸發條件就是 `B(FutA1, SpotB1) ≤ B_x`（`00 §5`），Q 量的正是這件事。
座標是參數：`residual_scaled`（預設）／`residual_bp`／`absolute_quantile`；EV 面實驗直接比。
maker 量過絕對分位數同商品跨日排序相關只有 0.142，所以預設用正規化殘差讓跨商品可 pool；介面上仍顯示「該商品 q 幾／幾 bp」。

## 3. 三張表

### 3.1 到達表（`reach.py`，市場資料）

| 表 | 鍵 | 值 | 定義 |
|---|---|---|---|
| `C0` | `(e桶, x, t桶, k桶)` | P | 從時刻 t 起，**今天 13:18 前** `r` 首次 ≤ x |
| `C1` | `(e桶, x, k桶)` | P | 到**明天 13:18 前**累計（含今天） |
| `q` | `(e桶, x, k桶)` → `(x, k桶)` → `(x)` | 每 session 到達率 | 第 2 天起、尚未到達者當 session 到達的比例；實測 pooled 版對高水位樂觀 3–7 倍，故按 e 分桶 |

- `t桶` = 決策時段 `<3600 / <9000 / 其餘`；`k桶` = 距到期日曆日 `0（結算日）/ 1 / 2–3 / ≥4`（全期剖面在 4 日以外平坦、最後 3 天機制不同，見結果紀錄）；跨日用**同一口合約**的序列，到期後不接。
- 取樣點：每個商品每 30 s 一個決策點（參數），只取 `analysis_eligible` 的秒；每個取樣點是一個樣本。
- 每格另存 `n`、`n_days`、`SE = √(P(1−P)/n)`、整日 block bootstrap 的 95% 區間（同日同商品取樣點高度相關，n 不能當獨立樣本數）。
- 回退 ladder 同時要求 `n ≥ min_n`（100）與 `n_days ≥ min_days`（5）；k 桶把 20 session 窗切成約 5 天，天數才是有效樣本。

### 3.2 水位表（`qlevel.py`）

`(商品, D)` → `s` 的 `q5 / q10 / q20 / q30 / q50 / q70 / q80 / q90 / q95` 與 `scale_D−1`，只用 `< D`。用途：座標換算、顯示、到期歸 0 路徑的 `G_0`。
`scale > MAX_SCALE_BP（60）` 的商品是盤口過寬，不進 pooled 表也不進決策（`scale_raw` 保留）。

### 3.3 設定值（`config.py`；不是從資料 fit，是手填後迭代）

| 設定 | 鍵 | 定義 | 第一版（maker guard 實測 + 3 margin） |
|---|---|---|---|
| `d_in` | `(stream, hedge 腿 tick-bp 桶, t桶)` | `quote_ab − actual_ab`，quote → hedge 完成 | S1 28.7；S2 11.9；S2 另取 `max(·, 深度＋50% 一 tick)` |
| `d_out` | `(stream, route E1/E2)` | 出場實拿 basis 相對 `B_x` 的 shortfall（正 = 差） | S1／S2 皆 3（實測負值當 0） |
| 費用 | 同日／隔夜 | | 20／34 |
| `d_settle` | — | 到期日收盤賣現貨 vs 結算價的滑價（未量測） | 0 + 3 |
| `ev_min` | — | 單筆最低 EV，擋成本邊界過敏的單 | 0 |
| hurdle | — | bp／日曆日 | 20%／365 ≈ 5.5 |
| `min_anchor_bp` | — | 負 anchor（除息）不是本線的交易，直接拒（reason `anchor`）；`ab ≤ 0` 亦拒（`basis`） | 0 |

tick-bp 桶 = hedge 腿 tick ÷ 價格（S2 看現貨 tick，S1 看期貨 tick）：`<5 / 5–15 / 15–30 / ≥30 bp`。
d 表的鍵先照 maker 的量測分桶；之後 Stage 3 校準表會告訴我們哪些桶該合併或再切。

## 4. EV

```text
給定 stream、quote_ab（我方掛價鎖定的 basis）、決策時 e、t、距到期 K 個 session、出場格點 X：

對每個 x ∈ X：
  P_sd    = C0[e, x, t, k]
  P_d1    = C1[e, x, k] − P_sd
  P_dj    = (1 − C1) · (1 − q_k)^(j−2) · q_k        j = 2 … K−1
  P_never = (1 − C1) · (1 − q_k)^(K−2)               → 到期歸 0
  G_x     = quote_ab − d_in − B_x − d_out            # 在 x 出場的毛利
  G_0     = quote_ab − d_in                          # 到期結算：無出場腿，無 d_out
  EV(x)   = P_sd · (G_x − 20) + (P_d1 + Σ_j P_dj) · (G_x − 34) + P_never · (G_0 − 34)
  T(x)    = P_sd · 今日剩餘 + P_d1 · 1 + Σ_j P_dj · j + P_never · K     # 日曆日，跨週末照算
  score(x) = EV(x) / T(x)                                                # bp／日曆日

x*   = argmax_x score(x)   subject to   EV(x) ≥ ev_min
掛單 ⟺ score(x*) ≥ hurdle
```

- `x = 0（結算）` 是格點裡的一列：`P_never` 對它就是「到期一定到」，T = K。不另寫分支。
- `surplus(x) = EV(x) − hurdle · T(x)` 只作診斷輸出。
- S2 沒有進場水位可選（A1 − 1 固定），只有掛／不掛；S1 選 U 還需 `P_fill(U)`，那是執行事實（Stage 2），不在 Q。
- 出場格點第一版 `X = {+0, −0.25, −0.5, −1.0 (× scale), 結算 0}`；細不細由 §5 決定。

## 4b. 第二條路：絕對價差收斂（2026-09-18 加入）

殘差表看不到「anchor 本身很高、殘差很小」的機會（120 s EWMA 跟著 basis 走，長期站在 120 bp 的商品殘差永遠接近 0），
而絕對價差自己收斂到 0 是一條獨立的收益路。第一版用 taker 線已算好的統計當概估（`ev/abs_reach.py`）：

- 來源：`taker/convergence_days_by_settle_cycle.py` 的樣本（SSD2 `stockfuture/convergence_days/convergence_samples_enriched_*.parquet`，2026-01-26～06-25 進場）。
  每筆 = 某商品-日 taker 可執行 basis（`ret_sell = FutBid/SpotA1 − 1`）第一次跨過門檻（0.5／1／1.5／2%）；收斂 = 同合約之後第一次 `ret_buy ≥ 0`（= 我們的 `basis_buy_taker ≤ 0`，絕對水位 0）；沒收斂就抱到結算。
- 表：`(門檻, 距結算交易日桶 0 / 1–2 / 3–5 / 6–10 / 11–15 / 16+) → P(第 j 個交易日首次收斂, j = 0…15)、P(之後)、P(抱到結算)`；門檻取**離 `quote_ab − d_in` 最近者**；格子不足 30 筆退到只按門檻；低於 50 bp 沒有這條路。可 `as_of`（進場日與收斂／結算日都早於決策日）。
- EV：`Σ_j P_j (ab_eff − d_out − fee_j) + P_settle (ab_eff − d_settle − 34)`，`ab_eff = quote_ab − d_in`；T 用該合約剩餘 session 的日曆偏移；`score = EV/T`。
- 決策：殘差路線各 x、到期路線、絕對路線一起 `argmax score`，任一達 hurdle 即掛（`ExitEval.route ∈ {residual, settle, absolute}`）。

已知偏差（Stage 3 校準對象，這條路第一版是「概估」）：
1. 樣本是第一次跨門檻的事件——短暫尖峰與持續站在高位的商品都在裡面（後者每天開盤第一筆就算跨過），但兩者收斂速度差很多，pooled 表混在一起。
2. 收斂判定是「某一瞬間 taker 可執行」，不是我們在水位 0 的 maker 單成交；對成交率是上界。
3. 門檻只有四個、樣本期間到 6/25。
第二版應改用本線自己的雙向表：進場成交後、出場側第一次在絕對水位 0 成交的分佈（同一形狀的表，資料換成自己的）。

## 5. 格子多大：資料決定，只在會翻轉決策的地方切

- 起手格：`e` 5 桶（scale 單位 1 / 1.5 / 2.5 / 4；bp 座標 15 / 25 / 50 / 80）× `x` 4 點（0 / −0.25 / −0.5 / −1 scale；bp 座標 0 / −5 / −10 / −20）× `t` 3 桶 × `k` 4 桶；
  每格 `n ≥ 100` 且 `n_days ≥ 5`，不足往上 pool（去 k → 去 t → 去 e）。
- 一個月只有一次到期，近到期三個 k 格即使全期也只有 5–6 天；4 日以外的格子全期有 100+ 天。
- as-of 窗長造成的差異大於格子雜訊（5–6 月是肥月，見 `01_RESULTS.md`），它是設定值，不是統計上越長越好。
- 切細條件（兩個都要）：切開後兩半的 `|P_a − P_b| > 2 · SE_pooled`，**且**至少一半的 `score` 相對 hurdle 變號。差 5% 但兩邊都掛／都不掛，不切。
- 每格印 `n / P / SE / CI`，EV 面上直接看。

## 6. 資料來源

| 需要 | 來源 | 備註 |
|---|---|---|
| `basis_mid_bp`、`basis_buy_taker_bp`、`anchor_ewma_120s_bp`、`analysis_eligible`、`end_date` | canonical 1 Hz 格 `maker/data/walkforward/daily/Date=*/causal_fair.parquet`（2026-01-26～08-13，133 日） | 每商品每秒一列，已含合法性與 anchor |
| 8/14 以後 | 以 maker `ev_lookup/ext_grid_builder.py` 的口徑自 tick 重建同 schema | 只用 `≤` 秒界的 receive time |
| 到期、合約對應 | `mapping.parquet`、預告日曆 | K 用決策日已知的日曆 |

兩種 fit（同一份程式）：`fit(None)` 全期表——所有 session pooled，看結構、定格點、調設定值用，**非因果**；
`fit(D, window)` as-of 表——只用 D 以前最近 `window` 個 session 且 label 已可得者，回測用，每日凍結。`window` 是 Stage 3 的設定值。
商品實例的 `quote_ab` 在 Stage 2 之前只有上下界：mid basis（上界）與 `basis_sell_taker`（FutBid vs SpotA1，下界）。

## 7. 模組與測試

```text
src/common/   paths（grid 位置、時間常數）、calendar（as-of 預告日曆）、grid（1 Hz 格 → 每商品陣列）
src/ev/
├── qlevel.py   每日 residual 直方圖 → 分位數與 scale；as-of 檢查；MAX_SCALE 閘
├── reach.py    每日到達事實（x 無關）→ C0 / C1 / q 的 fit；取樣點、SE、block bootstrap；ladder（min_n、min_days）
├── config.py   d_in / d_out / d_settle / 費用 / ev_min / hurdle / min_anchor 的設定值與桶鍵
├── abs_reach.py 絕對價差收斂表（taker 樣本 → (門檻, 距結算桶) 首次收斂分佈；as_of）
├── ev.py       純函數：分支機率、EV(x)、T(x)、score、argmax、gate；無 I/O
├── build.py    CLI：建每日快取（hist + facts）
└── surface.py  CLI：某日的到達表、CI、EV 面、商品實例 → data/ev_surface/
src/tests/      test_{calendar,ev,qlevel,reach}.py
```

純函數：
- `Σ P = 1`；`quote_ab` 上升 EV 單調不減；`d_in` 上升單調不增；`x = 0` 時 `P_never` 覆蓋全部且 `T = K`。
- `C0 = 1` → `EV = G_x − 20`、`T = 今日剩餘`；`C1 = C0 = 0, q = 0` → 全部到期歸 0。
- `ev_min` 生效：`EV = 2, T = 0.05` 的格點在 `ev_min = 5` 下不得被選。

到達表：
- 桶邊界值落桶正確；回退順序；`n < 100` 不得直接用。
- as-of：`fit(D)` 讀到 `≥ D` 的列 → AssertionError；K 用預告日曆、颱風休市只在公告後。
- 跨日只接同一口合約；到期日之後的秒不進到達序列。
- 合成序列測試：已知均值回歸速度的 OU 路徑 → C0 隨 |e − x| 單調、隨 t 剩餘時間單調。

對帳（maker 既有數字，量級即可）：
- S0.5 frozen-at-upper-touch：q95 觸價後同日回到 anchor（C0 對應 x = 0）reach proxy 92.8–100%、C2 78–83%、C3 53–58%（`FOUNDATION_SELECTION_S05_REBUILD`；anchor 15 s、母體不同）。
- maker guard 組 S2 實際：當日出場 34.23%、至次 session 約 59–64%——本線 `C0`／`C1` 在 x = −5 附近應高於這些（tape 觸及 ≥ 我方成交），差距就是 Stage 3 要校準的量。

## 8. 交付：EV 面（首版見 [`01_RESULTS.md`](01_RESULTS.md)）

任選一日，每檔商品輸出 `(e, x)` 矩陣的 `EV / T / score / surplus`，每格附 `n / SE / CI`，標出 `score ≥ hurdle` 的區域。這張面是之後每次調設定值的比較基準。

## 9. 與 Stage 2／3 的介面

- Stage 2：對每個候選點，用當時的 `e, t, k` 查 Q、用 `quote_ab` 與 config 算 `EV(x*)`；留下 `score ≥ hurdle` 且 `t_fill < t_below[floor] + 50 ms` 的點與其 x* 出場列。
- Stage 3：每格輸出「Q 預測到達率 vs 點位表實際 `exit_day_offset` 分布」、「`d_in`／`d_out` 設定 vs realized」。
  已知會出現的差：tape 觸及 ≠ 我方出場單成交、成交後的逆選擇漂移。它們就是設定值迭代要吸收的量，不回頭改 Q 的定義。
