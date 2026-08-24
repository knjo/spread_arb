# Quote Fill／Execution 研究索引

| 文件 | 內容 | 狀態 |
|---|---|---|
| [REPLAY_SAMPLING.md](REPLAY_SAMPLING.md) | SpreadPair epoch、多層存續掛單、同價去重與撤單口徑 | 八日pilot已實作 |
| [PILOT_RESULTS.md](PILOT_RESULTS.md) | Raw fill、partial、50 ms hedge與latent exit數據 | 八日／四商品；非EV-ready |
| [LIQUIDITY_SCREEN.md](LIQUIDITY_SCREEN.md) | 131日route liquidity與raw replay研究池 | Latent／screen完成 |
| [EV_LOOKUP.md](EV_LOOKUP.md) | 合法tick action、execution partitions、terminal path與D-safe EV lookup契約 | 介面已實作；131日execution／EV待擴樣 |
| [EXIT_MAKER_60D_RESULTS.md](EXIT_MAKER_60D_RESULTS.md) | 舊5商品、pre-Taifex-ladder-fix的matched T/T、overnight與prequential研究 | 歷史archive；canonical outputs已不存在，不可當current結果 |
| [EXIT_MAKER_45X60_PROGRESS_20260818.md](EXIT_MAKER_45X60_PROGRESS_20260818.md) | 45商品正式replay暫停點、1,416-partition interim Center／Lower結果與安全resume方式 | 歷史checkpoint；正式same-day已完成2,687／2,687 |
| [EXIT_MAKER_INTERIM_PORTFOLIO_20260820.md](EXIT_MAKER_INTERIM_PORTFOLIO_20260820.md) | 固定2,005-marker snapshot的交易量、nominal／strict隔夜曝險與completed-only P&L圖 | Analysis-only；非equity curve |
| [COMPACT_FROZEN_EVALUATOR_20260820.md](COMPACT_FROZEN_EVALUATOR_20260820.md) | 直接投影4,032,586筆凍結action facts的A/B1–2成交、撤單、等待時間與50ms hedge快速表 | Analysis-only；source-bound audit GO；核心聚合約3.5秒 |
| [COMPACT_REMAINING_TIME_STOP_REASON_20260821.md](COMPACT_REMAINING_TIME_STOP_REASON_20260821.md) | q50/80/95 × A/B1–2 × 剩餘session時間 × realized nominal-stop 的成交、撤單與等待時間表 | Source-bound analysis GO；stop reason是future label，盤中不可直接使用 |
| [MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md](MAKERFILL_L1_L5_DIAGNOSTIC_20260820.md) | 五個固定代表日的B1–2 vs B3–5 legacy makerFill、策略stop與exact indexed fill比較 | Analysis-only；source-bound audit GO |
| [FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md](FUTURE_ASK_L1_L5_INDEXED_DIAGNOSTIC_20260821.md) | 五個固定代表日的future ASK1–2 vs ASK3–5 exact indexed fill與50 ms spot hedge | Analysis-only；獨立audit GO；維持A/B1–2主研究 |
| [CURRENT_LADDER_D1_OVERNIGHT.md](CURRENT_LADDER_D1_OVERNIGHT.md) | 60日×45商品strict-carry於精確D+1、同契約、fresh≤1s第一個joint taker/taker平倉結果 | Formal verifier GO；95.23% physical可定價但gross mean -11.41 bp；非策略EV |
| [FINAL_PIPELINE_READINESS_20260820.md](FINAL_PIPELINE_READINESS_20260820.md) | Same-day→cross-session→filled-entry primary→contextual lookup的正式gate、命令與成本／unknown契約 | Same-day與cross正式roots已完成；成本／production gate仍fail-closed |
| [POST_CROSS_POSITION_EVALUATOR.md](POST_CROSS_POSITION_EVALUATOR.md) | 24格terminal cashflow、outstanding、D-safe最佳Q與部位限制報告契約 | V4低記憶體正式build與source-rebuild verifier均GO |
| [POST_CROSS_POSITION_RESULTS_20260821.md](POST_CROSS_POSITION_RESULTS_20260821.md) | 事後45商品cohort×60日跨日結果、24格策略、每日未平倉、部位cap壓測與記憶體修復紀錄 | Fixed-cohort descriptive result GO；universe／best-q／EV／production NO_GO |
| [PREQUENTIAL_CHALLENGER_AB12_RESULTS_20260821.md](PREQUENTIAL_CHALLENGER_AB12_RESULTS_20260821.md) | 固定事後cohort內，每日只用先前可見labels的AB1/2 challenger：掛單／成交、50 ms hedge、留倉、completed-only P&L與cap sweep | Within-cohort pseudo-OOS GO；universe非D-safe，q95非selected action |
| [PREQUENTIAL_COST_CAP_RESULTS_20260821.md](PREQUENTIAL_COST_CAP_RESULTS_20260821.md) | 依四腿實際成交價套用指定fee/tax，並同時測10M／20M／30M總cap與單品30% cap | Source-rebuild GO；analysis-only，universe／terminal／best-q／production NO_GO |
| [SUPPLEMENTAL_FULL_CARRY_RESULTS_20260821.md](SUPPLEMENTAL_FULL_CARRY_RESULTS_20260821.md) | unknown假設整筆跨日carry、後續normal exit replay與到期最後可用兩腿mark；含trade fallback與欄位語意 | 3,672／3,672可定價；一秒＋epoch-change近似、double-exit bias、universe／production NO_GO |
| [AGGRESSIVE_1300_ANALYSIS_BUNDLE.md](AGGRESSIVE_1300_ANALYSIS_BUNDLE.md) | 13:00後動態B1／A1 OCO平倉、normal競賽、cap再准入、成本與13:20曝險的正式結果 | V3 source-rebuild＋獨立replay GO；降低carry但aggressive子集負net；production NO_GO |
| [../WALK_FORWARD.md](../WALK_FORWARD.md) | 60-session日更、fine-tune／confirmation／holdout切分 | 驗證契約已定義 |

> **Supplemental schema warning：**v2 `supplemental_paths.parquet` 的
> `holding_session_boundaries` 仍是 formal source 欄，不能代表補充 terminal 的持倉期；
> 必須用權威 `Date`／`terminal_date` 配合 frozen 交易日曆重算。Combined-cap
> backtester 只依 terminal date／time 釋放額度，不使用這個 stale 欄位。

目前已可重播entry／exit maker fill、策略撤單需求、full-fill後50 ms taker hedge、同日matched taker/taker與一日overnight terminal benchmark，並產生D-safe prequential lookup。仍缺實際cancel ACK、完整broker cost profile與portfolio joint allocation，因此不得標成可掛單EV。
