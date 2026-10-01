> 2026-09-29：本文件已由 [RD盤前交付與實作盤點](13_RD_PREMARKET_HANDOFF.md) 取代，只保留歷史設計。
> 以下「以本文為準」不再適用。到期時間、scale缺值、anchor時間與有效性、B訊號去重、公司行動及額外驗證門檻有多處與採用版不同；不可直接照此實作。
> 舊文列出的兩個盤前檔尚無正式exporter；現行交易規格見 [12_RD_TRADING_SPEC.md](12_RD_TRADING_SPEC.md)。

# spreadArb 盤前檔規格（Python，歷史）

> 文件目的：讓 RD 用 Python 獨立寫出「每個交易日開盤前產出當日交易所需資料」的程式。不依賴 maker 研究線的任何中間檔。
>
> 產出兩個檔：`{D}_spreadArb_products.parquet`（每商品一列的靜態與當日參數）與 `{D}_spreadArb_tables.json`（Q 表、絕對路線表、成本表、當日 hurdle）。
> 交易程式（[`11_TRADING.md`](11_TRADING.md)）只讀這兩個檔，不再碰歷史 tick。
>
> 所有統計只用 `< D` 的資料；任何用到 D 當天或之後資料的地方都是錯的。本文與回測程式 `src/ev/`、`src/backtest/` 的口徑一致；若不一致，以本文為準並回頭修程式。

---

## 1. 先用一句話理解盤前檔

**用前 20 個交易日的期現價差秒序列，算出每檔商品的價差水位尺（scale）、價差回落機率表（Q 表）、絕對價差收斂表，加上手填的成本表與今天的門檻，交給交易程式查。**

盤前程式分五步：

1. 建今天的商品對照：現貨 ↔ 近月個股期貨、到期日、距結算天數、漲跌停、可交易清單。
2. 把 D−1 的 tick 建成 1 Hz 秒格（basis、anchor、殘差），累加進快取。
3. 由快取的秒格算 scale、Q 表、絕對路線表（都只用 `< D`）。
4. 成本表照設定檔；B 版政策另算今天的動態 hurdle。
5. 寫出兩個檔，做驗證，失敗就告警並拒絕產出（交易程式沒有盤前檔不得開新倉）。

---

## 2. 資料來源圖例

| 標籤 | 類型 | 定義 |
|---|---|---|
| <span style="color:#c62828;font-weight:700">● [TICK]</span> | 歷史 tick | D−1 及更早的現貨 tick（SSD2 `{d}_StockTick.parquet`）與個股期貨 tick（NAS `Ticks/YYYY/MM/DD/stock_futures.parquet`）。 |
| <span style="color:#ef6c00;font-weight:700">● [REF]</span> | 商品靜態 | 現貨基本資料（`{D}_marketData.parquet`：參考價、漲跌停、當沖資格）、期貨基本資料（近月合約、到期日、合約股數 2,000）、交易日曆。 |
| <span style="color:#1565c0;font-weight:700">● [GRID]</span> | 秒格快取 | 每個交易日每商品每秒一列的 basis／anchor／合格旗標，盤前程式自己維護的快取，只增不改。 |
| <span style="color:#6a1b9a;font-weight:700">● [TABLE]</span> | 表 | 由秒格算出的 scale、Q 表、絕對路線表。 |
| <span style="color:#455a64;font-weight:700">● [PARAM]</span> | 設定 | 成本、門檻、桶邊界、視窗長度；正式值在第 9 節。 |
| <span style="color:#2e7d32;font-weight:700">● [OUT]</span> | 產出 | 兩個盤前檔。 |

---

## 3. 每日時間軸

| 時間 | 動作 | 輸入 | 產出 |
|---|---|---|---|
| D−1 收盤後（建議 15:30 起） | 建 D−1 秒格並存入快取 | [TICK] D−1 兩市場 tick；[REF] D−1 商品對照 | [GRID] `grid/Date={D−1}.parquet` |
| D−1 收盤後 | 補 D−1 的到達標籤 | [GRID] D−2、D−3 的秒格因為 D−1 已知，可以補上「次日／次次日最低」 | [GRID] `facts/Date={D−2}`、`facts/Date={D−3}` 更新 |
| D 08:00 前 | 建今天的商品對照與 target_list | [REF] D 的 marketData、期貨合約表、日曆 | products 檔的靜態欄位 |
| D 08:00 前 | 算 scale、Q 表、絕對表、hurdle | [GRID] 前 20 個交易日 | tables 檔 |
| D 08:00 前 | 驗證並寫檔 | | [OUT] 兩個檔＋ manifest |
| D 08:15 | 交易程式載入 | [OUT] | 缺檔或驗證失敗 → 交易程式今天不開新倉，只處理既有部位的出場 |

秒格與標籤是累加式的：每天只算新的一天，前面的不重算。第一次上線要回補至少 23 個交易日（20 日視窗＋ 2 日標籤＋ 1 日緩衝），絕對路線表要回補 16 個交易日以上才有格子。

---

## 4. 商品對照與靜態（products 檔）

每列一檔現貨：

| 欄位 | 來源 | 定義 |
|---|---|---|
| `ValueCode` | [REF] | 現貨代碼 |
| `QuoteCode` | [REF] | 近月標準個股期貨合約代碼。到期日當天仍用該合約；到期日之後的第一個交易日換下一個月 |
| `expiry` | [REF] | 合約最後交易日 `YYYYMMDD` |
| `k_cal` | [CALC] | 距到期日曆日（含週末），`expiry − D` |
| `k_td` | [CALC] | 距到期交易日數（用預告日曆，只算 D 已公告的休市） |
| `session_offsets` | [CALC] | D 之後到到期日的每個交易日相對 D 的日曆日偏移，list；EV 的 T 用它 |
| `settle_offset` | [CALC] | 到期日之後第一個交易日相對 D 的日曆日偏移 |
| `ref_price`、`limit_up`、`limit_down` | [REF] | 現貨當日參考價與漲跌停（TWD） |
| `fut_ref_price` | [REF] | 期貨參考價 |
| `contract_shares` | [REF] | 2,000（小型合約不做） |
| `scale_bp` | [TABLE] | 第 6 節；`null` 表示不可交易 |
| `tradable` | [CALC] | 第 4.1 節全部通過 |
| `product_cap_twd` | [PARAM] | `max(1 張現貨名目, 0.25 × cap_twd)`；1 張名目 = `ref_price × 2,000` |
| `target_list_hit` | [PARAM] | 是否在 `target_list.txt`；清單為空視為全部命中 |

### 4.1 `tradable` 的規則

全部成立才為 true；任一不成立的原因寫進 `untradable_reason`：

1. 有近月合約且 `expiry ≥ D`；合約股數 2,000。
2. `scale_bp` 存在且 `1 ≤ scale_bp ≤ 60`（第 6 節）。
3. 前 20 個交易日中至少 5 天有合格秒（第 5 節）。
4. 不在除權息、減資、停牌、處置的視窗：除權息交易日前 3 個交易日到除權息日（含）不進場；此規則用 [REF] 的公司行動表，`announce_day < D` 才可用。
5. `target_list.txt` 為空，或該商品在清單內。

不可交易的商品，交易程式**仍要訂閱行情**（既有部位要出場），只是不掛進場單。

---

## 5. 秒格（1 Hz basis grid）

每個交易日 d、每檔商品、每個 `sec ∈ [0, 15600)`（09:00:00 起算的秒序，13:20 為止），一列。

### 5.1 每秒取值

```text
對每一秒 sec：
    spot book = 該商品現貨在 open + sec + 1 秒之前（<）收到的最後一筆正式盤五檔
    fut  book = 該商品期貨合約在同一時點之前收到的最後一筆有五檔的正式盤
```

「收到」用 UTC `RecvTime`；兩市場之間只比時間，不比 `ChannelSeq`。期貨成交列常常五檔全零，要維護「最近一筆有效五檔」。

```text
SpotB1/A1 = 現貨明掛 L1 與 BestBid/BestAsk 的較優價（含量）；FutB1/A1 同理
SpotMid   = (SpotB1 + SpotA1) / 2；FutMid 同理
basis_mid_bp        = 10,000 × (FutMid / SpotMid − 1)
basis_buy_taker_bp  = 10,000 × (FutA1 / SpotB1 − 1)        # 賣現貨買期貨的可執行 basis（出場觸發序列）
basis_sell_taker_bp = 10,000 × (FutB1 / SpotA1 − 1)        # 買現貨賣期貨的可執行 basis
```

### 5.2 合格秒 `eligible`

全部成立：

1. 兩邊 book 都是正式盤（`TrialMatch == 0`），B1 > 0、A1 有量、B1 < A1（不交叉）。
2. 兩邊價格都在參考價 `0.91 ×` 到 `1.08 ×` 之間（等號排除）。
3. 兩邊 book 距該秒都不超過 60 秒（過時 book 不算）。
4. `sec ≥ 300`（09:05 以後）。

不合格的秒 basis 記 `null`，不參與 anchor 更新、殘差直方圖與到達統計。

### 5.3 anchor

```text
α = 1 − 0.5^(1/120)                          # 半衰 120 秒
anchor[sec] = anchor[sec−1] + α × (basis_mid_bp[sec] − anchor[sec−1])   # 只在合格秒更新
第一個合格秒：anchor = basis_mid_bp
不合格秒：anchor 沿用上一值（不更新）
```

anchor 每天從頭算（不跨日）。

### 5.4 快取檔

`grid/Date={d}.parquet`，欄位：`ValueCode, QuoteCode, sec, basis_mid_bp, basis_buy_taker_bp, basis_sell_taker_bp, anchor_bp, eligible`。
每天約 230 檔 × 15,600 秒。

### 5.5 到達事實 `facts`

Q 表的樣本單位是「取樣點」：每檔商品每 30 秒一個（`sec = 300, 330, …, 13,980`），只取合格且 anchor 存在的秒。每個取樣點記：

```text
resid_bp     = basis_mid_bp[t0] − anchor[t0]
anchor_bp    = anchor[t0]
min_today_bp = min over sec ∈ (t0, 15,480] 合格秒 的 basis_buy_taker_bp − anchor_bp     # 今天 13:18 前的最低可執行 basis（相對凍結 anchor）
min_d1_bp    = 次一交易日 [300, 15,480] 合格秒的最低 basis_buy_taker_bp − anchor_bp        # 同一口合約；合約到期或次日缺資料記 null
min_d2_bp    = 次次交易日同上
k_days       = 到期日 − d（日曆日）
```

`min_d1`／`min_d2` 要等次日／次次日收盤後才補得上，所以 `facts/Date={d}` 會被改寫最多兩次；改寫時只填這兩欄。

---

## 6. scale（水位尺）

```text
每個交易日 d、每檔商品：合格秒的 resid = basis_mid_bp − anchor_bp，做 1 bp 寬的直方圖（−300～300）
scale(D) = 前 20 個交易日（< D）直方圖合併後的 q95 − q50
```

`scale < 1` 或 `> 60` → `scale_bp = null`（盤口過寬或死盤），該商品今天不可交易。直方圖每天存一份（`hist/Date={d}.parquet`），合併是加總計數。

---

## 7. Q 表（價差回落機率）

### 7.1 座標

```text
e = resid_bp / scale(d)      取樣點當時的殘差，以該日的 scale 正規化（scale 用 < d 的視窗）
x ∈ {0, −0.25, −0.5, −1.0}   出場水位，scale 單位；B_x = anchor + x × scale
到達 = 相對凍結 anchor 的最低可執行 basis ≤ x × scale
```

桶：

| 桶 | 邊界 | 說明 |
|---|---|---|
| `e_b` | `(1.0, 1.5, 2.5, 4.0)` → 5 桶 | `searchsorted(side=right)`：e < 1 → 0，1 ≤ e < 1.5 → 1，… |
| `t_b` | 決策秒 `(3600, 9000)` → 3 桶 | 10:00 前／12:30 前／其餘 |
| `k_b` | 距到期日曆日 `(0, 1, 3)` → 4 桶 | 0 = 結算日、1、2–3、≥ 4；`searchsorted(side=left)` |

### 7.2 三張表

用 D 之前最近 20 個交易日的取樣點（`window = 20`），且標籤在 D 之前可觀測：

| 表 | 鍵 | 值 | 樣本條件 |
|---|---|---|---|
| `C0` | `(e_b, x, t_b, k_b)` | `P(min_today ≤ x·scale)` | 取樣日 `< D` |
| `C1` | `(e_b, x, k_b)` | `P(min_today ≤ x·scale 或 min_d1 ≤ x·scale)` | 取樣日的次一交易日 `< D` 且同一口合約 |
| `q` | `(e_b, x, k_b)` | 前兩天都沒到、第三天到的比例：`P(min_d2 ≤ x·scale ∣ 前兩天未到)` | 次次交易日 `< D` |

每格存 `n`（取樣點數）、`n_days`（天數）、`p`。格子太薄往上 pool（ladder）：

```text
C0: (e_b, x, t_b, k_b) → (e_b, x, t_b) → (e_b, x) → (x)
C1: (e_b, x, k_b) → (e_b, x) → (x)
q : (e_b, x, k_b) → (x, k_b) → (x)
可用條件：n ≥ 100 且 n_days ≥ 5；ladder 每一層都要存，交易程式查表時由細到粗取第一個可用的格
```

### 7.3 輸出格式

tables 檔的 `reach` 區塊：

```json
"reach": {
  "as_of": "20260706", "window_days": 20, "days_used": ["20260605", "..."],
  "e_edges": [1.0, 1.5, 2.5, 4.0], "x_grid": [0.0, -0.25, -0.5, -1.0], "t_edges": [3600, 9000], "k_edges": [0, 1, 3],
  "min_n": 100, "min_days": 5,
  "c0": [{"keys": ["e_b","x","t_b","k_b"], "values": [2, -0.5, 0, 3], "n": 512, "n_days": 9, "p": 0.41}, ...],
  "c1": [...], "q": [...]
}
```

---

## 8. 絕對路線表（價差收斂到 0 的機率）

第一版沿用 taker 線的收斂樣本（`stockfuture/convergence_days/convergence_samples_enriched_*.parquet`；之後改用自家秒格算同形狀的表，格式不變）。

```text
樣本：某商品-日 basis_sell_taker 第一次跨過門檻 thr ∈ {50, 100, 150, 200} bp
收斂：同一口合約之後第一次 basis_buy_taker ≤ 0 的交易日序 j（0 = 當天）；沒收斂就抱到結算
as-of D：只用進場日 ≤ D 之前第 16 個交易日的樣本（j ≤ 15 都已可觀測）；
        結果在 D 之前還不知道的樣本（既沒收斂也沒結算）算「抱到結算」
鍵：(thr, k_td 桶)，k_td 桶邊界 (0, 2, 5, 10, 15) → 0 / 1–2 / 3–5 / 6–10 / 11–15 / 16+
值：p_day[0..15]、p_later（16 日後才自然收斂）、p_settle；每格 n；n < 30 退到只按 thr
查表：thr 取離 (quote_ab − d_in) 最近者；(quote_ab − d_in) < 50 沒有這條路
```

輸出區塊 `absolute`：`{"as_of", "thresholds_bp", "k_edges_td", "min_n", "cells": [{"thr_bp":100,"k_b":3,"n":143,"p_day":[...16 個...],"p_later":0.0,"p_settle":0.03}, ...]}`。

---

## 9. 成本表與當日 hurdle

### 9.1 成本表（手填，[PARAM]）

```json
"cost": {
  "fee_same_day_bp": 20, "fee_overnight_bp": 34, "margin_bp": 3,
  "d_in_base_bp": {"S1": 25.7, "S2": 8.9},
  "d_in_override_bp": {},              # 鍵 "S2|tick_b|t_b"，tick_b 用 hedge 腿 tick/價 (5,15,30) 分 4 桶
  "d_out_base_bp": {"S1": 0.0, "S2": 0.0},
  "d_settle_base_bp": 0.0,
  "ev_min_bp": 0.0, "min_anchor_bp": 0.0
}
```

交易程式用法：`d_in = max(base, 0) + margin`；`d_out = max(base, 0) + margin`；`d_settle = max(base, 0) + margin`。
校準後（Stage 4）的值直接改這個區塊，不改程式。

### 9.2 hurdle

```text
base_bp_per_cal_day = 8.5 × 250 / 365 = 5.82           # 每筆每日曆日的最低 EV/T
政策 A：hurdle_today = base
政策 B：若 D−1 收盤時策略 committed ≥ 0.8 × cap_twd，
            hurdle_today = max(base, quantile_0.5(昨日訊號分數))；否則 = base
```

**昨日訊號分數**（盤前程式用 D−1 的秒格重算，不依賴交易程式的 log）：

```text
對 D−1 的每一秒 sec ∈ [300, 14000)、每檔可交易商品、S1 與 S2 兩個 stream：
    用 D−1 當天生效的表（as_of = D−1）與 D−1 的 anchor、scale、k 算 EV（[11 §7]）
    S2 候選價 P = FutA1 − 1 tick（需 > FutB1），quote_ab = 10,000 × (P / SpotA1 − 1)
    S1 候選價 P = SpotB1，quote_ab = 10,000 × (FutB1 / P − 1)
    預篩：quote_ab > 0；(quote_ab − anchor ≥ 25 或 quote_ab ≥ 50)
    取 score = max 路線 EV/T
訊號 = 相同 (stream, round(quote_ab), round(anchor), sec // 60, expiry, round(scale, 1)) 只算一次
只保留 score ≥ base 的訊號；取 q50
```

`committed_prev_close` 由交易程式每日收盤寫出的部位檔取得（[11 §12]）；取不到就視為未碰頂（用 base）。

輸出區塊：`"hurdle": {"policy": "B", "base_bp_per_day": 5.82, "today_bp_per_day": 13.4, "signals_prev": 2694, "signals_q50": 13.4, "committed_prev_close": 17800000}`。

---

## 10. 輸出契約與驗證

```text
{OUT_DIR}/{D}_spreadArb_products.parquet
{OUT_DIR}/{D}_spreadArb_tables.json
{OUT_DIR}/{D}_spreadArb_manifest.json      # 輸入檔清單與 hash、grid 天數、各表格數、驗證結果、產出時間
```

驗證（任一失敗 → 不寫 products／tables，只寫 manifest 標 `failed`，並告警）：

1. `reach.days_used` 全部 `< D`，且數量 = 20（上線初期允許 ≥ 5 並標 `warmup`）。
2. `absolute` 每格 `Σ p_day + p_later + p_settle = 1 ± 1e-9`。
3. 每檔 `scale_bp` 在 `[1, 60]` 或 `null`；`tradable` 商品數 ≥ 50，否則標 `warmup`。
4. 每檔 `expiry ≥ D`；`session_offsets` 嚴格遞增、最後一個 = `k_cal`。
5. hurdle 在 `[base, 60]`；超過 60 視為錯誤（棘輪或資料異常）。
6. 與昨天的 products 檔比較：合約代碼變動的商品必須是到期換月；scale 變動超過 3 倍要列在 manifest 的 `warnings`。

---

## 11. 參數總表

| 參數 | 值 | 用途 |
|---|---:|---|
| `grid_close_sec` | 15,600 | 秒格終點（13:20） |
| `quote_start_sec` / `quote_end_sec` / `withdraw_sec` | 300 / 14,000 / 15,480 | 取樣與到達視窗（09:05 / 12:53:20 / 13:18） |
| `price_band` | 0.91–1.08 × 參考價 | 合格秒 |
| `book_stale_sec` | 60 | 合格秒 |
| `anchor_half_life_sec` | 120 | anchor |
| `sample_step_sec` | 30 | Q 表取樣 |
| `window_days` | 20 | scale、Q 表 |
| `e_edges` / `x_grid` / `t_edges` / `k_edges` | (1,1.5,2.5,4) / (0,−.25,−.5,−1) / (3600,9000) / (0,1,3) | Q 表桶 |
| `min_n` / `min_days` | 100 / 5 | 格子可用 |
| `scale_min` / `scale_max` | 1 / 60 bp | 可交易 |
| `abs_thresholds` / `abs_k_edges_td` / `abs_max_j` / `abs_min_n` | (50,100,150,200) / (0,2,5,10,15) / 15 / 30 | 絕對表 |
| `abs_seasoning_td` | 16 | as-of 樣本最少年齡 |
| `hurdle_base` | 8.5 bp／交易日 = 5.82 bp／日曆日 | |
| `dyn_q` / `dyn_cap_frac` | 0.5 / 0.8 | 政策 B |
| `cap_twd` / `product_cap_frac` | 20,000,000 / 0.25 | 容量 |
| `ex_div_block_td` | 3 | 除權息前禁新倉 |
