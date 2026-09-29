# EV 查表框架交接統整（2026-09-01 ~ 2026-09-08）

> 2026-09-09 狀態更新：本文為歷史交接，以下 v17「champion」與 27.9k／日已被因果／執行重驗取代。
> 現行已完成的第一順位控制結果見 [v22](EV_LOOKUP_V22_LIQUIDITY_20260909.md)；
> 全腳成本、XC 與 Q 表的新統整見 [執行成本重驗](EV_EXECUTION_COST_REVIEW_20260909.md)。

> 撰寫：Claude（接手 Codex 的 maker 研究線）。本文是研究**統整與交接**，不是凍結 bundle；
> 所有數字皆為 scratchpad 診斷口徑（1 口/fill、approximate makerFill 進場、全實現無截尾），
> 程式留存於 [`../../src/ev_lookup/`](../../src/ev_lookup/)，結果快照於 `maker/data/ev_lookup_20260908/`（不進 git）。
> 清理紀錄見 [`EV_LOOKUP_CLEANUP_LOG_20260908.md`](EV_LOOKUP_CLEANUP_LOG_20260908.md)。

## 0. 一頁結論

| 項目 | 結論 |
|---|---|
| 策略定位 | 個股期現 basis 的**發散度 regime 資產**：6~7 月肥、8 月休眠；正價差保底 + 深水閘門後 8 月轉為正常月。 |
| 現行 champion | **雙腳疊加 + walk-forward 分流深水閘門（v17）**：20M 日均 27.9k TWD（年化 34%）、中位 17.5k、當沖率 54%、5~9 月每月為正（8 月 +23k）。 |
| 餵納 | 到 100M 近線性（50M 33%、100M 31%）；無上限日流量 ~135M、PnL 天花板 ~328k/天。 |
| 商品覆載 | 240 檔全盯（每日 83% 商品有機會、61% 有深機會）；同時掛單峰值 ~89、p95 ~66；期貨訊息 0.12/s（免費 5/s 足夠）。 |
| 最大單一發現 | 使用者提案的 **S2-inside route**：期貨 Ask maker 掛 A1−1tick（inside、隊列第一）+ 成交瞬間 taker 現貨。機會來源 = 寬 spread 期貨讓所有 taker 套利者無法收割的 25~35bp 時刻。 |
| 最重要的方法論教訓 | ① 有賺 ≠ 值得佔容量（分配鍵 = bp/容量日）；② 截尾會計與到期結算必須建對；③ 分項帳 ≠ 邊際貢獻；④ 結構知識（到期保底）勝過慢速經驗學習；⑤ 候選 dump + 秒級重放是政策迭代正解。 |

## 1. 各階段統整（含接手前）

### S0 — 8 月轉弱歸因（Codex，2026-08-24，完成）
May~Jul → Aug：q95 excursion touch rate 7.15% → 4.29%，touch 後 approximate fill 1.88% → 1.22%；市場/邊界與 queue/競爭並列。
Bundle `august_attribution_s0_20260824_v2`。**接手後的使用**：其 `raw_order_facts`（q95 BID1/2 quote-only 流）是本框架 S1 現貨 maker 進場流的真實成交來源（≤ 08-13）。

### S0.5 — 查表基礎（Codex，2026-08-26，完成）
盤中 anchor `time_ewma_15s`、D-safe q lookup `Q2_trail20_date_equal`（跨商品 Spearman 0.626、同商品跨日 0.142）、S1 mother 15,638 product-days / 71 sessions / 244 商品、frozen-at-upper-touch v2 收斂幾何（C0/C2/C3 reach 與 known-cost margin）。
**接手後的使用**：`conditional_geometry_summary_overall` + `frozen_primary_reach_summary` + S0 `post_touch_fill_by_rank` 組成第一張 EV 篩選表（§3.1）。

### S1 — 七組 policy × Spot Bid maker（Codex，v4 工程完成、無 71 日結果；接手後**未續跑**）
奈秒級 fail-closed 事件迴圈（venue quota、rollback taxonomy、三層 differential、497 partitions）。365 項回歸 + 19 項 path contract 通過，control smoke `unresolved=0`；v3 bundle 1/497 作廢，v4 precommit differential 通過但 clean-commit durable rerun / 497 partitions 未完成。
**接手判斷（使用者 2026-09-01 定調）**：實盤做不到微秒級；原意是秒K粒度 + 統計性掛單/撤單/成交估計 + 機率表 EV 最佳化。奈秒模擬器判定為過度工程，主線改為本框架；S1 的凍結規格（B5 approximate makerFill、B6 hedge、C8 到期 basis=0、C9 20M/10M cap、13:19:45 drain）作為口徑依據沿用。

### S2 — Future Ask maker route（Codex 規劃、未執行；接手後以 S2-inside 實現）
Codex 規劃為「期貨 Ask maker 排 A/B1–2 + spot taker hedge」，從未跑出結果。接手後：
- 先以「期貨 bid 穿價」保守觸發 + 耐心 join-B1 現貨 hedge 測得下限 +1.9bp/筆（chase 半邊有毒、動能 gate 無效）。
- 2026-09-04 使用者提出**掛 A1−1tick inside**：trade-print 真值（買方主動成交 ≥ 我們的價必先打到隊列第一）+ 下一秒才認成交 + 正價差 gate → uncapped 250 fills/天、+23.7bp/筆；連續全期 20M 單獨 15.9k/天（carry 書貼死 19M 是資本效率瓶頸）；疊加後成為 champion 的厚利腳。

## 2. 時間軸與結果帳（全部 20M cap、全實現口徑，除非另註）

| 日期 | 版本/分析 | 20M 日均 | 年化 | 重點 |
|---|---|---|---|---|
| 09-01 | EV 篩選（凍結表） | — | — | 九組 q-pair 五組零滑價即負；q95_C2 唯一 −10bp 下存活 |
| 09-01 | EV(U,L) 9 天格點 | ~2.4k | — | 最優 U≥20/L≤−5；量級由 fill率×口數決定；預測與 predecessor 20M cap 表 2.1k 一致 |
| 09-01 | v2 整合（Ask1 排隊出場真值） | 13.9k | 17% | 同日率 67%；50M 28.8k/14% |
| 09-01 | 掛單黏性 | — | — | 雙向不撤 stale fill 71% 毒；單邊保留上界 +124 fills/天 |
| 09-02 | hedge 腳 maker 化（S1 route） | — | — | 期貨 A1−1 hedge 淨 −8~−10bp：觸發後 2s fut bid 掉 22.5bp = 自家 alpha；B6 +50ms taker 設計正確 |
| 09-03 | v5 EV cross + 影子 λ（walk-forward） | 16.2k | 19.8% | 8 月 1.1k→4.9k（規則隨 regime 自適應） |
| 09-03 | v6 深度 EV 准入 | 17.8k | 21.8% | 薄簿 fill 轉換差（TWD 正規化 7.6→12.4bp），非穿價 |
| 09-03 | v7 出場 L* | 20.8k | 25.5% | L* = argmax[−L + (14+λ)P_sd(L)]，八成選 −10 |
| 09-03 | v8 正價差 gate | 30.7k | 37.6% | 逆價差進場整桶 −14~−16bp、未收斂 14%、佔卡死樣本 58.5%；8 月 1.6k→15.4k |
| 09-03 | v10 結構保底 EV 准入 | 39.6k | 48.5% | `est = P_sd×(effU−L−20) + (1−P_sd)×(ab−34) ≥ λ×0.6`；學習版先驗（v9）較差 |
| 09-03 | 資料延伸至 09-02 | 33.3k | 40.7% | 8/14~9/02 三週 −7~−11k/天（regime）；合成流校準 0.65× |
| 09-03 | 餵納（S1 單獨） | — | — | 未 cap 流量 ~49M/天、天花板 ~70k/天；50M 21%、100M 13% → 甜蜜點 20~50M |
| 09-04 | 多價差鏡像 | — | — | 8 月當沖 −35k（接刀）；fix50+carry(54bp) 全月為正 +3~15k；正確形態 = EV 式 + 除息 gate |
| 09-04 | v12 雙層資金（交割 20M/日內另設） | 18.8k | 23.0% | 日內額度放大無效（並發 gross 僅 ~28M、邊際單當沖辨識不足）；trim 漏損曾 −49k/天 |
| 09-04 | 到期結算修正（使用者指正） | 21.1k | 25.9% | 部位不可能跨月卡住；expiry_S ≈ 0 = 災難保險非提款機 |
| 09-04 | 早盤 slot 假設 | 22.7k | 27.8% | 30M 早盤衝失敗（留倉書無 slack，邊際失敗 −50bp） |
| 09-04 | **S2-inside route** | — | — | uncapped 250 fills/天、+23.7bp；訊息 EV 容忍帶 4.1→0.12/s |
| 09-07 | S2 單獨連續 85 天 | 15.9k | 19% | 9 樣本日 51.6k 為 carry 重置假象；carry 貼死 19M |
| 09-07 | 天真疊加 | 14.9k | 18% | S2 慢 carry 擠死 S1 週轉 |
| 09-07 | deep50 / bpday 疊加（真模擬器） | 25.5k / 49.9k | 31% / 61% | bpday 表 5-6 月估（半 in-sample） |
| **09-08** | **v17 walk-forward bpday + 稽核** | **27.9k**（中位 17.5k） | **34%** | 5~9 月 +22/+25/+41/+23/+21k；50M 68k(33%)、100M 128k(31%)、∞ 328k；稽核零違規 |

圖：`maker/data/ev_lookup_20260908/img/`（`backtest_20m_v10.png`、`backtest_ext_0902.png`、`backtest_v17_stacked.png`；同檔亦在 `script/sideProject/report/`）。

## 3. EV 計算與最佳化作法（要留存的核心）

程式：[`ev_rules.py`](../../src/ev_lookup/ev_rules.py)。成本：現貨當沖 20bp、隔夜 34bp（純稅費）、多單留倉 54bp。

### 3.1 EV 篩選（凍結表，零回放）
每筆 = 收斂率 × 同日成本後 margin + (1−收斂率) × 隔夜成本後 margin，再扣 10/20bp 執行滑價敏感度。九組 q-pair 五組不用跑即死，直接回答「Codex 為何在跑不用跑的 pair」。

### 3.2 EV(U,L) 格點（秒K機率表）
1Hz residual = basis_mid − 15s EWMA anchor；episode 首次觸及 U；P(reach L | touch U) 掃到 13:20；EV = P×(U−L−20) + (1−P)×(U−L−34)。
結論：U<12 全滅、活區 U≥20/L≤−5；TWD 面平坦 → 量級由 fill 率 × 口數決定。

### 3.3 進場准入（結構保底版，champion）
```
空: est = P_sd(進場時段) × (effU − L − 20) + (1 − P_sd) × [有隔夜slot ? ab − 34 : 強平估損]
多: est = P_sd(深度桶)   × (depth − 25)    + (1 − P_sd) × [除息gate可留倉 ? P回升×(depth − 59) − 尾損 : 強平估損]
admit ⟺ est ≥ λ × 預期佔用天
```
`ab` = 成交鎖定的絕對 basis（期貨可執行 bid ÷ 我方掛價 − 1），到期歸 0 故 carry 分支保底 = ab − 34。
效果：逆價差（保底負）自動拒；正價差但薄且尾盤（P_sd 低）也拒；不需選 0 或 34 的固定門檻。
機制驗證：逆價差進場後 30 分 anchor 中位上漂 +22.5bp（q95 觸價是「回 0 的路」非錯價）。

### 3.4 容量影子價格 λ
λ = 前 5 日被容量拒絕 fill 的估計 EV ÷ cap（bp/日，clip 15）。20M 頂到 clip（稀缺）、50M 自然變小。

### 3.5 續掛 vs cross（取代硬編碼 13:00）
```
cross ⟺ d < (14×當日 + λ) × (1 − P_fill(τ, d))
```
P_fill(τ, d) walk-forward（剩餘秒 × 距離桶）：剩 3h 且 d≤0 → 0.99；剩 <20 分 → ≈0，尾盤自動積極。

### 3.6 出場 L*
`L* = argmax_L [−L + (14+λ) × P_sd(L | 進場時段)]`（effU 消掉）。早盤 L=−10 仍 57% 同日。

### 3.7 多流容量分配：bpday 格閘門
8 格（S1/S2 × ab 桶 0-30/30-50/50-80/80+）：格內歷史 (平均 bp ÷ 平均佔用天) ≥ 8 才掛。統計在部位**解決日**入表、trailing 20 日、來源為不受容量影響的影子池、格內 <30 筆放行。
walk-forward 下實質 = **S1 只做 ab≥50、S2 只做 ab≥80**。**不是日內挑單**：表開盤前定，機會按時序來一筆查一筆。
S1 主力淺格（ab 0~30）每容量日僅 2.6bp、S2 淺格 1~3bp —— 有賺但不值得佔容量。

### 3.8 S2-inside 掛單 EV 容忍帶（訊息 4.1/s → 0.12/s）
只看「我這張單的 EV 有沒有變」：A1 抖動、被 undercut 不動；現貨 ask 上移使 held 鎖定 basis 跌破 U−5 才重掛；被穿價 = 成交。A1 後撤時留在原價當唯一最佳賣價（成交量優先於單價）。

### 3.9 到期結算
持有過到期日、次日結算：兩腿同結算價、basis 歸 0、空單收 ab − 34（= 凍結規格 C8）。實測卡到到期的單 ab 中位 30~40bp → ≈ 打平：**到期保證是災難保險，不是提款機**。

### 3.10 逐筆下單決策鏈（v18，`decide.py` — 回測與實盤共用）

候選 fill 按時序到達，每筆走四關，任一關不過即不下單並記錄原因：

```
0. 鎖定量（皆當下已知）: ab = (期貨可執行bid ÷ 我方掛價 − 1)·1e4 ; effU = ab − anchor
1. est  = P_sd(strm,時段)·(effU + 5 − 20) + (1 − P_sd)·tail
          tail = ab − 34（預期隔夜書仍有位子）| 強平估損（沒位子）        → est < λ·0.6 ⇒ 拒 'ev'
2. λ    = 前5日被容量拒絕fill的估計EV均值 ÷ cap（bp/日, clip 15）
3. cell = (strm, ab桶) 之 walk-forward 統計 (bp ÷ 佔用天) ≥ 8              → 否 ⇒ 拒 'cell'
4. gross + ntl ≤ cap                                                          → 否 ⇒ 拒 'cap'（其 est·ntl 累進明日 λ）
```

**真實 trace（2026-07-06，20M，`results/v18_trace_20260706.csv`）**：210 筆候選 → 收 53、'cap' 拒 105、'cell' 拒 26、'ev' 拒 26。

| t | vc | strm | effU | ab | P_sd | λ | est | cell | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 09:05:10 | 1326 | S1 | 32.5 | 72.9 | 0.71 | 15 | 23.8 | S1_2 | **收** |
| 09:05:29 | 3532 | S2 | 102.9 | 62.2 | 0.54 | 15 | 60.6 | S2_2 | **收** |
| 09:05:11 | 2449 | S2 | −1.0 | 55.8 | 0.54 | 15 | 1.3 | S2_2 | 拒 'ev'（深度不足） |
| 09:05:32 | 3714 | S2 | 1.5 | 15.6 | 0.54 | 15 | −15.7 | S2_0 | 拒 'ev'（保底負） |
| 09:07:04 | 4743 | S2 | 53.1 | 48.2 | 0.54 | 15 | 27.2 | S2_1 | 拒 'cell'（ab 30~50 格 bp/日 < 8） |
| 09:05:48 | 5483 | S2 | 119.0 | 224.2 | 0.54 | 15 | 143.4 | S2_3 | 拒 'cap'（書已滿） |

最後一列說明現況：容量是真正的稀缺資源（當日一半候選因 'cap' 被拒，含 est 143bp 的肥單），λ 頂在 clip；這也是「餵納到 100M 近線性」的來源。

**v18 結果**（決策鏈接入後，其餘同 v17）：20M **28.5k/天**（中位 18.6k；5~9 月 +25/+26/+41/+22/+21k）、50M 77.5k、100M 134.8k、∞ 329k；稽核零違規。與 v17（格閘門+容量）相比 20M +0.5k、50M +9k、100M +7k —— est/λ 那關在容量較鬆時貢獻較大。
出場側的 cross 規則（§3.5）在 v5~v12 驗證，v18 **未接**（出場 = 同日 maker → 後日 maker → 到期）。

## 4. 估計與回測方法

- **Fill 真值**：現貨 maker 進出場用 makerFill `Bid1/2 / Ask1_FillSeconds`（排隊真值，同 B5 approximate 等級）；期貨 A1−1 maker 用 NAS 逐筆成交（px ≥ 掛價 = 買方必先打到隊列第一）+ 下一秒起算 + 被穿價即成交。
- **兩腿定價**：entry effU 用期貨可執行 bid（空）/ ask（多）；出場觸價條件用 `basis_buy_taker`（含期貨 spread 跨越）→ hedge 滑價已內含，不另加。
- **carry**：真跨日追蹤，逐日重試 maker 出場，到期結算；容量 = gross（含 carry）≤ cap；carried 出場在實際秒釋放；到期部位持有過到期日。
- **資料延伸**：canonical 1Hz 格至 08-13；08-14~09-02 由 `ext_grid_builder.py` 自原始 tick 重建（08-28 NAS 壞檔跳過）；S1 合成流在 08-11~13 校準（fills 32/33/25 vs 實際 36/85/26，≈0.65×）。
- **稽核**（v17）：出場秒 ≤ 進場秒 0 筆、解決日 ≤ 進場日 0 筆、gross 超帽 0 筆（max_gross 精確貼帽）。
- **政策迭代**：`candidate_dump.py` 解析所有候選命運 → `policy_replay.py` 秒級比較（重放較真模擬器樂觀 ~17%）→ 勝者回真模擬器驗證。
- **對帳**：v11 33k → v12 18.8k 差距 2/3 為截尾樂觀（carry 不認損）、1/3 為工程簡化；多單邊際 +5.9k（分項帳 −1.4k 是錯覺）。

## 5. 已否決的方向（附證據，避免重走）
- 現貨 maker 進場後、期貨 hedge 腳掛 A1−1（maker+maker）：trade-print + 2s 窗仍 −12bp（觸發後 2s fut bid 掉 22.5bp；需 2s 內 fill >53%，實測 25%）。
- 雙邊搶觸發（期現同掛）：純增量 12 筆/天 × ~2bp。
- 完全不撤單 / 大容忍帶：stale fill 71% 有毒（effU −6bp）。
- 事前動能 gate 判別 chase：無效（補漲在觸發後才開始）。
- 日內額度放大（20M 交割下）：並發 gross 天然 ~28M、邊際單當沖辨識不足 → 負 EV；早盤 30M 衝同理。
- 學習版逆價差先驗（v9）：regime lag，輸給結構保底。
- 出場側現貨排 A1 賣（耐心 hedge）：−1~−3bp。

## 6. 仍存在的樂觀口徑（上線前必驗）
1. S2 側：1Hz 掛價凍結近似、我們掛單對市場的影響、trade-print fill 判定（回測無法覆蓋）。
2. 全程 1 口/fill；S1 進場為 approximate makerFill（B5 預期 exact ≈ 0.9×）。
3. 08-14 後 S1 合成流在 v17 略過（僅 S2）。
4. 已實現口徑無庫存評價：深水閘門 + 到期保底使每筆終值 ≥ 0，曲線單調；風險呈現為資金鎖定（carry 均 16.6M）、庫存浮動、期貨腿保證金，不是日損益。
5. 無融資成本、無多單借券可券性檢查；bpday 同日內表更新有小洩漏（正式版改開盤凍結）。

## 7. 建議下一步
1. 每日庫存評價線（權益曲線誠實化）+ 開盤凍結表。
2. 正式研究文件化（本文）→ 3~5 檔寬 spread 商品期貨 A1−1 各 1 口**實單試掛**，驗證 fill 模型與市場反應（回測替代不了的最後一關）。
3. 口數 P_fill(n) 模型（餵納近線性到 100M，口數是下一個槓桿）。
4. 商品預篩僅在工程必要時用「前日曾有 eb≥25 機會」（少盯 50 檔、丟 4% 深機會）。
