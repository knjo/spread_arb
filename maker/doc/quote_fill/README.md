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
| [MONTHLY_PRODUCT_SELECTOR_20260822.md](MONTHLY_PRODUCT_SELECTOR_20260822.md) | M-1 → M 因果商品池、proxy 排序力、3,886 product-day manifest | **現行 universe** |
| [ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md](ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md) | 1 Hz 掛撤量、現貨 100/s 與期貨 5/s 上限、13:00 drain | 現行 |
| [ORDER_MESSAGE_LOAD_LEGACY_DIAGNOSTIC_20260822.md](ORDER_MESSAGE_LOAD_LEGACY_DIAGNOSTIC_20260822.md) | legacy hedge／makerFill 欄位可用性證據 | 現行 |
| [ONE_SECOND_MAKERFILL_CAUSAL_V2_20260822.md](ONE_SECOND_MAKERFILL_CAUSAL_V2_20260822.md) | q95 AB1/2 approximate fill／cancel screening，7,730 fills | 現行；mixed-clock approximate |
| [DYNAMIC_CAUSAL_PATH_PORTFOLIO_20260822.md](DYNAMIC_CAUSAL_PATH_PORTFOLIO_20260822.md) | +50 ms hedge、同日／跨日／到期 path、10–50M cap 回放 | 現行 |
| [DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md](DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md) | 端到端現況與精度邊界總結 | **現行結論** |
| [MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md](MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md) | 五日 BID1–2 vs BID3–5 exact fill 與 hedge slip | 「只掛 A/B1–2」決策證據 |
| [FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md](FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md) | 五日 ASK1–2 vs ASK3–5 | 同上 |
| [../REWORK_PLAN_20260824.md](../REWORK_PLAN_20260824.md) | 因果線擴回原始規格的 S0–S5 checklist | **已定案；S0 待執行** |

## Archive（固定 45 檔、有 universe leakage）

[`archive_fixed45/`](archive_fixed45/) 內 20 份文件：exit maker 60d、cross-session、overnight carry、post-cross、
prequential challenger、cost/cap sweep、aggressive 13:00、compact evaluator、EV lookup 契約與 pipeline readiness。
它們證明過的東西只有兩項還在用：AB1/2 決策、以及 q95 作為 challenger 的出處。其餘數字不可再引用為現行結果。

## 資料

`maker/data/walkforward/` 現在只剩基礎事實（`daily`、`rolling_boundaries`、`liquidity`、`sessions.txt`、
`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1`）、因果 manifest、8/22 六個 causal 輸出，
以及兩個 L1–L5 診斷 bundle。
