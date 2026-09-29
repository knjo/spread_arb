# EV 隔夜分支對齊出場規則 + 事件驅動撤掛（2026-09-09）

回應 Codex 2026-09-09 稽核兩項：(1) EV 公式把所有非當日出場按「持到到期 basis 歸零」估值，
但回放實際在隔日觸及凍結目標就出場；(2) S2 掛單只在每秒邊界檢查，現貨 ask 已變仍留單。
改動全部在 `maker/src/ev_lookup/`（v19 引擎），沒有動原始資料、canonical、S0、taker 程式；未 commit。

## 1. EV：三條出場路徑分開估值（`ev_rules.est_short`）

| 路徑 | 機率 | 估值（bp） |
|---|---|---|
| 當日於目標出場 | P_sd | eff_u − L − 20 |
| 隔夜後於目標出場 | (1−P_sd)·P_nx | eff_u − L − 34 |
| 持到到期、basis=0（C8 慣例） | (1−P_sd)(1−P_nx) | ab − 34 |

P_nx = 「非當日出場者之後仍於目標正常出場」的機率，walk-forward：只用開盤前已知、前 20 個
完成 session 內解決的影子池結果（`ResolvedOutcome.kind == "maker_exit"` 佔 carry 解決數的比例），
依 stream 分桶，n<30 用 0.8。Codex 舉的例子 ab=80 / anchor=50 / eff_u=30：舊估 46bp，
新估 P_nx=1 時 1bp、P_nx=0.75 時 12bp。`p_nx=0` 完全重現舊口徑，歷史腳本不受影響。

λ（容量影子價格）仍由被容量拒絕機會的 est 均值算出，所以 est 變誠實後 λ 自動下降；
煙霧測試裡 2 天冷啟動 λ 頂到 clip 15bp/日，全期會由表決定。

## 2. 出場選擇與 EV 對齊

- 目標出場不變：basis 觸及「掛單時凍結 anchor − 5」掛現貨 Ask，成交後 taker 買回期貨。
  這正是公式第 1、2 路徑；第 3 路徑（到期）是未觸及者的自然結局，公式與回放一致。
- 新接入 `ev_rules.should_cross`（12:00 起每秒評估）：d = taker 出場 basis（期貨 Ask / 現貨 Bid）
  距凍結目標的 bp；P_fill(τ) = 影子池「該時段仍未出場的部位，當日稍後以 maker 出場」的比例，
  4 個剩餘時間桶（<11:30、11:30–12:30、12:30–13:00、13:00–13:18），walk-forward，n<30 用 0.5。
  `d < (14·[當日進場] + λ) × (1 − P_fill)` 即撤 maker 出場單、taker 賣現貨再買期貨，
  close_kind = `taker_cross`。撤單仍有 cancel_ms 競態：撤單生效前成交的量照算。
- 早上不套 cross（保留 maker 出場賺現貨 spread 的機會）；d ≤ 0 時由原本的觸價規則掛 maker。
- `choose_L`（動態 L）仍未接：影子池只跑一個 L，沒有反事實 P_sd(L) 表，先維持 L=−5。

## 3. 事件驅動 S2 撤掛（`Portfolio.book_update` / `_requote_s2`）

- `MarketDay.spot_ask_events`：現貨最佳賣價每次變動一個事件（AskPrice1 與 BestAskPrice 取小）。
- 回放迴圈把它排在同時間戳的成交之後、任務之前；`book_update` 用事件當下的現貨 Ask 重算
  持有掛價鎖定的 basis，`basis − anchor < 20bp` 或 `basis ≤ 0` 立即送撤單（仍經 cancel_ms 延遲）。
- 撤單生效後若「新的 A1 − 1 tick 對當下現貨 Ask」仍滿足進場帶（≥ anchor + 25bp、正 basis、
  高於期貨 Bid），立刻重掛，走同一套 `submit()`（EV / bpday / 硬容量），intent id = `S2/{day}/{vc}/e{ns}`。
  不滿足就等下一秒的正常候選流程；不追價掛到 A1 之上（那會離開隊列第一且佔預留額度）。
- 每秒一次的舊檢查保留，作為事件流缺漏時的後盾。
- S1 側（S0 指令流 + 期貨 taker hedge）未動。

## 4. 煙霧測試（7/23–7/24，20M，cancel 50ms，冷啟動）

| 日 | S2 paired | hedge 後 basis ≤ 0 | 其中撤單競態（撤單已送、50ms 內成交） | 事件撤單 | 重掛 | cross |
|---|---|---|---|---|---|---|
| 7/23 | 35 | 10 | 4（成交距撤單 0.1–26ms） | 306 | 69 | 4 |
| 7/24 | 29 | 7 | 3（2.7–37ms） | 306 | 62 | 18 |

**逆選擇成交只有四成能靠事件撤單擋，而且要撤單延遲 < 幾 ms。** 其餘六成在撤單觸發前就成交：
現貨 Ask 是在期貨成交之後（hedge 50ms 內）才變，或 hedge 吃到第二檔以下。這與 Codex 的拆分一致
（−5,664/日是成交前現貨已變、−2,519/日是成交後 50ms 內的變化；後者事件撤單無能為力）。

cross 規則在 7/24 用了 7/23 單日影子表（P_fill 0–0.05）、λ 頂 clip 15，門檻放寬到 ~28bp，
18 筆 cross 平均 −20.8bp，maker 出場 9 筆 −3.4bp。這是單日表 + 舊 λ 口徑的副作用；
全期 20 session 表與誠實 λ 下的數字見第 5 節。

## 5. 全期 86 天結果（與 Codex r4 對照）

輸出 `maker/data/ev_lookup_v21_exit_ev_20260909/`（source_snapshot 為本節所用版本；
Codex 在本次跑完前已改寫工作樹，見第 7 節）。20M、hedge 50ms、cancel 50ms、85 個可交易日。

| 組合 | 已實現 TWD | /日 | 進場 | hedge 後 basis≤0 | 同日率 | 收盤種類 |
|---|---:|---:|---:|---:|---:|---|
| r4 ev_20M（舊 EV、無事件撤掛、無 cross） | +42,910 | +505 | 3,386 | 175 | 0.38 | maker 3,026 / 到期 265 |
| **v21 ev_20M（本次三項全開）** | **−67,534** | **−795** | 3,061 | 215 | 0.60 | maker 1,463 / **cross 1,327** / 到期 202 |
| r4 ev_bpday_20M | +10,504 | +124 | 245 | 15 | 0.34 | 只交易 5–6 月 |
| v21 ev_bpday_20M | +67,736 | +797 | 441 | 14 | 0.57 | 只交易 5–6 月 |

拆解 v21 ev_20M（依 close_kind，平均 bp）：

| close_kind | 筆 | TWD | bp | r4 對應 |
|---|---:|---:|---:|---|
| maker_exit 同日 | 818 | +120,243 | +11.9 | 1,274 筆 +101,854（+7.4） |
| maker_exit 隔夜 | 645 | +861 | +4.8 | 1,752 筆 −130,803（−6.0） |
| 到期 C8 | 202 | +48,429 | +11.6 | 265 筆 +78,348 |
| **taker_cross 同日** | **991** | **−160,218** | **−8.0** | — |
| taker_cross 隔夜 | 336 | −38,927 | −2.0 | — |

**cross 規則是唯一的負貢獻，而且原因單一：λ 在 86 天裡有 85 天釘在 15bp/日的 clip。**
門檻 (14+λ)(1−P_fill) 因此放寬到 ~28bp，1,327 次 cross 中 1,121 次是 λ=0 規則不會做的，
合計 −212.6k；λ=0 規則會保留的 206 次合計 +13.4k（+1.9bp）。λ 之所以釘頂，是它用被容量拒絕
機會的「估計 EV」定價：全期 admitted 部位 est 平均 30.5bp，實現平均 0.5bp，差 60 倍；
用這種 λ 去換算「早一天釋放資金值多少」必然錯。

其他兩項的效果（同一次 run，不是獨立消融）：
- 同日 maker 出場每筆 +11.9bp（r4 +7.4）、隔夜 maker 出場 +4.8bp（r4 −6.0）：EV 分支對齊後
  進場選擇有改善；但 bpday 版本仍只在 5–6 月交易，8 格閘門在誠實結果下全期關閉。
- 事件驅動撤掛（每日 266 次撤、56 次重掛）沒有降低逆選擇成交：S2 hedge 後 basis≤0 127 筆
  vs r4 131 筆。與煙霧測試一致——六成逆選擇成交在撤單觸發前就發生，四成在 50ms 競態內。
- P_fill 表（全期末）：09:00 仍持有的 carry 當日 maker 出場 11.5%，11:30 後 2.2%，12:30 後 1.4%，
  13:00 後 0.9%。**11:30 之後才想 maker 出場的部位，基本上會留倉。**

結論：(a) EV 分支拆分保留；(b) cross 規則維持關閉，直到 λ 改為以實現報酬定價
（例如 trailing 已實現 bp/日，或 est 依實現校準）；(c) 事件撤掛不是逆選擇的解，
逆選擇要靠更嚴的進場帶或 hedge 端。

## 7. 併行改動說明

本 run 啟動後，Codex 在同一目錄接續改寫：`decide.py` 改用 `exit_model.py` 的競爭風險
hazard（日風險集、含未解決 carry）取代本文的 P_nx 完成樣本比例；`Portfolio` 新增
`split_ev / event_quotes / enable_cross` 開關（`full_study.py` 預設三者關閉以重現基準）；
事件流擴到期貨盤口，重掛改為「往回掛到能恢復 basis 的價位」並由 Replay 統一廣播 intent；
`ev_event_study.py` 為 legacy / split / split+bpday × {每秒, 事件} 的消融 runner。
本文第 5 節數字來自 `source_snapshot`，不是工作樹現況；兩套測試在工作樹現況均通過。

## 6. 測試

`test_ev_exit_events.py` 新增 7 項：EV 分支數值（Codex 例子）、P_nx / P_fill 表只用完成 session、
should_cross 單調性、現貨 Ask 上移即撤 + 撤單前成交仍須 hedge、重掛只在帶內且用新 A1−tick、
午後 cross 以 taker_cross 結案、早上不 cross。原 `test_v19.py` 21 項維持通過。

執行（HFT 根目錄）：

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.test_v19
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.test_ev_exit_events
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.full_study \
  <output_dir> --configs ev_bpday_20M ev_20M
```
