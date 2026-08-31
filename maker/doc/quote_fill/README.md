# Quote Fill／Execution 研究索引

2026-08-24 起的主線是**動態商品池因果 pipeline**：M-1 選月池 → D-1 liquidity gate → 1 Hz quote intent →
approximate makerFill → +50 ms hedge → terminal path → inventory cap 回放。固定 45 檔時代的所有結果已搬到
[`archive_fixed45/`](archive_fixed45/)，資料與程式已刪除，只留文件裡的數字供對照；可由 nested repo 的
`<Snapshot>` commit `1348576` 撈回。

## 現行文件

| 文件 | 內容 | 狀態 |
|---|---|---|
| [REPLAY_SAMPLING.md](REPLAY_SAMPLING.md) | SpreadPair epoch、多層存續掛單、同價去重與撤單口徑 | 契約，已凍結 |
| [PILOT_RESULTS.md](PILOT_RESULTS.md) | 八日／四商品 raw fill、partial、50 ms hedge 與 latent exit；含舊 60-session checkpoint 段落 | 八日 pilot 有效；60-session 段落資料已刪 |
| [LIQUIDITY_SCREEN.md](LIQUIDITY_SCREEN.md) | 131 日 route liquidity、D-safe daily facts | 基礎事實，被 selector 讀 |
| [MONTHLY_PRODUCT_SELECTOR_20260822.md](MONTHLY_PRODUCT_SELECTOR_20260822.md) | M-1 → M q95／Spot-Bid proxy 商品池 | 舊 conditional bridge；canonical S0.5 overlap 3,846，不再作七組唯一母體 |
| [ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md](ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md) | 1 Hz 掛撤量、現貨 100/s 與期貨 5/s 上限、13:00 drain | 現行 |
| [ORDER_MESSAGE_LOAD_LEGACY_DIAGNOSTIC_20260822.md](ORDER_MESSAGE_LOAD_LEGACY_DIAGNOSTIC_20260822.md) | legacy hedge／makerFill 欄位可用性證據 | 現行 |
| [ONE_SECOND_MAKERFILL_CAUSAL_V2_20260822.md](ONE_SECOND_MAKERFILL_CAUSAL_V2_20260822.md) | q95 AB1/2 approximate fill／cancel screening，7,730 fills | 現行；mixed-clock approximate |
| [DYNAMIC_CAUSAL_PATH_PORTFOLIO_20260822.md](DYNAMIC_CAUSAL_PATH_PORTFOLIO_20260822.md) | +50 ms hedge、同日／跨日／到期 path、10–50M cap 回放 | 現行 |
| [DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md](DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md) | 端到端現況與精度邊界總結 | **現行結論** |
| [MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md](MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md) | 五日 BID1–2 vs BID3–5 exact fill 與 hedge slip | 「只掛 A/B1–2」決策證據 |
| [FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md](FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md) | 五日 ASK1–2 vs ASK3–5 | 同上 |
| [AUGUST_ATTRIBUTION_20260824.md](AUGUST_ATTRIBUTION_20260824.md) | S0 q95 raw-excursion touch vs post-touch queue 歸因、30-session sensitivity | **完成；兩者並列** |
| [FOUNDATION_SELECTION_S05_REBUILD_20260826.md](FOUNDATION_SELECTION_S05_REBUILD_20260826.md) | S0.5 從頭選 15s anchor／Q2 trail20、S1 mother並重算frozen lower | 完整完成；證據已由cost-aware S1 grid採用 |
| [S1_COST_AWARE_IMPLEMENTATION_20260831.md](S1_COST_AWARE_IMPLEMENTATION_20260831.md) | 新七組、成本gate、absolute exit、20M cap、path與publication preflight | **程式與359項回歸完成；尚無正式績效** |
| [S1_STOP_CLEANUP_20260831.md](S1_STOP_CLEANUP_20260831.md) | 舊 S1 最終停止狀態、保留／清除項目與重啟條件 | 歷史快照；重啟條件已由cost-aware實作承接 |
| [S1_PAUSED_STATUS_20260829.md](S1_PAUSED_STATUS_20260829.md) | 舊 S1 77／497 partial 的停止原因、11 日方向性摘要與成本感知重啟條件 | 歷史紀錄；bundle 已清除 |
| [FOUNDATION_REVALIDATION_S05_20260825.md](FOUNDATION_REVALIDATION_S05_20260825.md) | 舊 EWMA120 基礎重驗與重作理由 | 歷史 predecessor；不得作現行 S1 lookup |
| [../REWORK_PLAN_20260824.md](../REWORK_PLAN_20260824.md) | 因果線擴回原始規格的 S0–S5 checklist | **S0／S0.5完成；cost-aware S1待smoke／正式replay** |

## Archive（固定 45 檔、有 universe leakage）

[`archive_fixed45/`](archive_fixed45/) 內 20 份文件：exit maker 60d、cross-session、overnight carry、post-cross、
prequential challenger、cost/cap sweep、aggressive 13:00、compact evaluator、EV lookup 契約與 pipeline readiness。
它們證明過的東西只有兩項還在用：AB1/2 決策、以及 q95 作為 challenger 的出處。其餘數字不可再引用為現行結果。

## 資料

`maker/data/walkforward/` 的現行層包含基礎事實（`daily`、`rolling_boundaries`、`liquidity`、`sessions.txt`、
`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1`）、因果 manifest、8/22 六個 causal 輸出、
兩個 L1–L5 診斷 bundle、S0 canonical／30-session sensitivity bundle，以及現行 S0.5
`foundation_selection_s05_rebuild_20260826_v1`與`foundation_selection_s05_frozen_convergence_20260826_v2`；`foundation_revalidation_s05_20260825_v1`只保留為 predecessor。S0 舊 v1 有 explicit L1 clear
forward-fill 錯誤，僅 v2 可引用；詳細 hash 與重跑方式見上述 S0 文件。
