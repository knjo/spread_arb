# 重作計畫（執行定稿）：把因果 pipeline 擴回原始規格

日期：2026-08-24

決策基線：nested repo commit `0c3e5ad`

狀態（2026-09-01 修訂）：**S0與S0.5完整完成；cost-aware S1 七組、absolute frozen exit、20M chronological cap、B6 retry、成本 ledger、SSD2／NAS input contract、8/13 common-horizon valuation、exit pre-fill risk guard、fail-closed publication gate與source-bound verification receipt均已完成程式接線，完整S1回歸363項通過。clean-source smoke、497 partitions、獨立 input verify與正式排名仍未完成，因此目前沒有新的 S1 績效或可部署結論**。完成一項就在本文件打勾並填結果與 bundle 連結。

## 研究定位（使用者定義）

這是類套利策略：估好價差、算準執行滑價與稅費，理論上不應賠錢。因此：

- **不設研究「通過／不通過」門檻**。主報表固定兩個數字：20M cap 成本後日均 net、同日完成率；每個數字必附 `entry_fill_truth=approximate/exact`，不得隱藏screen精度。
- 「不設研究獲利 pass/fail」不等於每個候選都必須送單。六個 deployment-eligible scenario 使用預先凍結的 actual-send economic admission rule；另有一個 ungated control。Publication gate 只驗完整性與跨 scenario 可比較性，不是研究獲利門檻。
- **同日完成率是最佳化目標**（當沖賣出稅 15 bp vs 隔夜 30 bp）。
- 商品池按月用完整 M-1 更新；q 界線在每個交易日 D 開盤前，用嚴格 `<D`、最多最近 60 sessions 更新。照凍結公式更新是設計，不是 leakage；看過 target-period outcome 後改公式才是。
- 所有七組 threshold 都保留完整結果，供未來實盤決定掛法；stage shortlist 只控制後續運算量，不等於淘汰研究證據。

## 已定案的十個決策

| 題 | 決定 |
|---|---|
| A1 界線 policy（2026-08-31 修訂） | cost-aware 七組全跑：`ctrl_q95_C0_ungated`、`q95_C0_sd_f5`、`q95_C0_on_f0`、`q95_C2_sd_f5`、`q80_C0_sd_f5`、`q50_C3_sd_f5`、`fixed20_sym20_on_f0`。舊 `q50/q80/q95 + fixed15/20/25/30` grid 已由停止的無經濟 admission partial 證明不適合直接續算，保留為歷史 predecessor |
| A2 出場下緣 | frozen；S1在`actual_new_send_time`以當下causal anchor鎖定絕對lower，後續maker fill沿用該target，不得在fill cursor重設。S0.5以first legal upper touch凍結作明示proxy；dynamic只當sensitivity |
| A3 8 月惡化 | 先查再跑 A1。假說：8 月價差波動縮、溢價消失（界線碰到率掉）vs 界線掛太深（碰到後成交率掉） |
| B4 期貨 Ask maker route | 加回來當對照；不做雙 entry route 同掛 |
| B5 成交標籤 | S0–S4 的 Spot Bid **entry** A/B1–2 大範圍 screening 使用 legacy `makerFill` fast adapter；明標 approximate，S5 才對凍結組合做一次同樣本 exact entry 校準。Future maker 與 pooled FIFO exit 不適用 makerFill，分別照 S2／S3 使用 indexed replay |
| B6 hedge 資料定價 | 先獨立判斷 `maker implied fill + 50 ms` 的 decision book；當下不可執行才往後最多 5 秒找第一個合法且足量 book。只有實際延後者標 delayed；不得把現有 213 筆預先全標 delayed |
| C7 exit maker | 先做「現貨 Ask maker → 買期貨 taker」；工程跑通再加「期貨 Bid maker → 賣現貨 taker」 |
| C8 13:00 後 | (a) 允許 carry，真正留到 expiry 的少量 paired residual 以 basis=0 accounting mark 結清；(b) 13:00 起積極平倉、13:20 hard flatten。比較「犧牲平倉收益換隔日容量」是否划算 |
| C9 部位 | `hard_intraday_cap_twd = TWD 20,000,000`（20M／2,000 萬），單檔 50%，即 TWD 10,000,000；其他 cap 之後再做 |
| D10 驗收 | 無 pass/fail 門檻；固定報 20M 日均 net 與同日完成率 |

## C9／B5／B6 的精確口徑

### C9：20M，不使用「2000M」

- `20M TWD = TWD 20,000,000 = 2,000 萬`；`2000M` 按 M = million 會變成 20 億，後續文件與設定一律不使用。
- 容量口徑是 outstanding **one-way spot notional**：`entry_spot_price × contract_size_shares`，不是 spot＋future 雙腿 gross notional，也不是 futures margin。
- Primary 20M／10M 是 pre-trade reservation cap，不是成交後才挑 fill 的 completed-only filter。每張 entry new request實際送出前先保留「live order最大 spot-equivalent notional」；沒有容量就不送單，所有已送單後**模型可觀測**的 fill都必須接住。S0–S4 Spot Bid的 hidden partial不在 legacy event universe，須由S5 exact揭露並重算，不能假裝已被approximate run驗證。
- Spot Bid entry 以 maker limit price × contract size保留；Future Ask entry 以 D 日 frozen causal `1.08 × opening_ref_price × contract_size` 作保守上界，spot hedge完成後用實際 spot notional reconcile並釋放差額。合法 depth 價格嚴格低於同一 frozen bound，因此實際值不得突破 reservation。
- Working reservation 在 maker fill 時轉入 `hedge_pending／paired_open`，不重複計額；未成交 leaves 只在 actual cancel send cursor（V0）或可驗證的 day-order session-expiry event 後釋放。Partial fill 同時保留 filled exposure與剩餘 leaves reservation。
- Entry hedge pending／delayed 期間仍占額；完整 exit hedge 執行後才釋放 position capacity。Entry emergency rollback 完成則釋放該 reservation；exit emergency rollback 只恢復原 paired position，不釋放原 capacity。任何 rollback failure 持續占 committed capacity 到真正 resolution 或 reporting horizon。
- 白話說法：entry maker 成交只把掛單 reservation 轉成 exposure，反腿 taker 完成後成為 paired position，容量仍被部位占用；正常 exit maker 成交後，還要等反腿 taker 真正完成才 flat 並釋放容量。不能在 maker fill 時先假設 50 ms 後一定成交而提早釋放。
- 「hard cap」只指本文件 V0 replay（cancel send 後立即生效）與該 stage可觀測 fill universe內的 hard invariant；因 S0–S4 Spot Bid看不到 partial、也沒有 exchange ACK／cancel-race truth，不宣稱已證明 production exchange-level hard cap。

### B5：接受 entry fast screen，但不把 makerFill 說成 exact

- Actual-new send 在共同 chronological event loop 內成立時，才依該 cursor snapshot 的 `(ValueCode, ChannelSeq)` 取得 A/B1–2 `Float32 FillSeconds`，以 `snapshot RecvTime + FillSeconds` 建立 mixed-clock **potential** fill event。
- Potential fill、cancel intent、現貨／期貨各自的送單額度、hedge／rollback 都在同一 event loop 競爭；`actual_cancel_send_time`／session expiry 是回放輸出，不可先拿來建 label。最終只接受位於 `(actual_new_send_time, effective_cancel_or_expiry]` 且先於其他 terminal 的 potential fill，再回填 active interval。Nominal submit／stop只留 audit；cancel 是 controller **request**，不是 makerFill 自帶撤單，也不是 exchange cancel ACK。
- 所有輸出保留 `fill_cursor_exact=false`、`own_quantity_included=false`、`partial_fill_included=false`、`cancel_ack_observed=false`、`joint_volume_allocated=false`。不得把 approximate full fill 改名成實盤成交。
- 本決策只涵蓋 Spot Bid entry 的大範圍 sweep。Future maker 沒有 legacy 欄位；pooled FIFO exit 需要 own quantity／partial／共同 printed volume，兩者使用 indexed replay，不把它們稱為 B5 的 exact calibration。
- S5 的 entry 係數定義為同一批 `raw_order_fact_id` 分母上的 `exact full-fill rate / approximate fill rate`；同報 confusion matrix。係數只作 aggregate sensitivity，不逐筆線性縮放非線性的 PnL、FIFO 或 cap admission。

### B6：213 筆是舊 label 無法定價，不是 213 次實盤失敗

本規則適用所有 entry／exit taker hedge。先定義 route-neutral `hedge_trigger_cursor`：S1 Spot Bid entry 用 legacy implied fill cursor；S2 Future Ask entry 用 indexed exact full-fill cursor；S3 pooled exit 用累積到一個 futures-equivalent hedge unit 的完成 cursor，而不是第一筆 partial；S4 forced flatten 用 initiating first-leg executable fill cursor。令 `t0 = hedge_trigger_cursor + 50 ms`：

1. Arrival book 只供 hedge trigger 當下的 slippage reference；arrival 無效不得讓程式跳過 `t0` execution-book 判定。舊結果中的 196 筆 `arrival_gate_closed` 因此不能先假設為 delayed。
2. 先用 causal as-of raw state **獨立**保存 `t0` book status，不再受 arrival status短路。若 `t0` book合法、L1–L5 足量且相應市場仍有送單額度，就在同一 scheduler 實際送出並消耗一筆 request；V0 的 `execution_time = actual_request_send_time`，價格取 send cursor 的 causal as-of executable VWAP，`hedge_delayed=false`。
3. 若 book或相應市場送單額度任一條件在 `t0` 不成立，就沿完整 raw `RecvTime`／event cursor與 scheduler 額度時間，在 `(t0, min(t0+5s, session_end)]` 找第一個同時滿足「合法足量 book＋可送 request」的 cursor，以該時點 executable VWAP 定價。內部等待／檢查不消耗 request，只有實際 marketable hedge送出時消耗一筆。
4. Hedge deadline cursor 先做 inclusive dispatch；仍未送出就把原 hedge intent 原子式標 `hedge_retry_timeout`、移出 queue，再 enqueue 主版 emergency rollback，不得日後又補送原 hedge。Rollback 用 route-neutral `initiating_first_leg_venue / initiating_execution_id`，以最高風險優先序在 `[timeout, min(timeout+5s, initiating venue session_end)]` 找第一個「opposite book 合法足量＋相應市場可送 request」cursor，經同一 scheduler消耗一筆 request，反向沖銷 initiating first-leg execution；V0 價格同樣取 actual send cursor 的 causal book。Entry rollback完成後回到 flat並釋放 reservation；exit／forced-flat rollback完成後恢復原 paired position、繼續占原 capacity。Rollback仍失敗者進 `entry_hedge_timeout_unresolved` 或 `exit_rollback_failed_unresolved`，明列 `emergency_rollback_failed` 與裸露 notional，不得消失。

合法 book 固定為：使用完整 normalized raw-state machine；非 TrialMatch，TrialMatch 後須等新的 formal book 才重開 gate；bid／ask 有正價格與正數量且 `bid <= ask`；reference price 為正，executable BBO 與實際掃到的 levels 均嚴格位於 `(0.91 × ref, 1.08 × ref)`；Best／L1 同價取最大量、不相加；L1–L5 足以完成 requested hedge quantity。Book age 記錄但不作主版 hard gate。

每筆至少保存：

- `hedge_trigger_time_ns`、`hedge_target_time_ns = t0`
- `hedge_request_send_time_ns`（套相應市場送單額度後）
- `hedge_execution_time_ns`、`hedge_book_recv_time_ns`
- `hedge_retry_delay_ms = execution_time − t0`
- `hedge_total_delay_ms = execution_time − hedge_trigger_time`
- `hedge_delayed`、`hedge_initial_status / gate_reason`、`hedge_final_status`
- `initiating_execution_id / initiating_first_leg_venue`
- `emergency_rollback_status / request_send_time / execution_time / book_recv_time / side / quantity / executable_vwap / failure_reason`
- `emergency_rollback_failure_notional_twd`
- `slippage_reference_available`；arrival reference 無效時 arrival-based latency／total slip 保持 null，實際 PnL 仍使用 execution VWAP

正值一律代表不利。On-time 與 delayed 分列筆數、pricing coverage、slippage reference coverage、delay p50／p95／max、latency／depth／total slip；不得把 null slip 當 0。

## 共同比較、選優與重現契約（S0 原凍結；S0.5 修訂）

### Development 資料與因果邊界

- S0 歷史歸因固定使用 `monthly_product_selector_causal_v2_20260822/daily_entry_manifest.csv`，SHA-256 `9f1bcddf17eff968ee51e0decdb04736a3747f0665886ce3e4fd26031cfb5891`；原 manifest 是 2026-05-04～2026-08-13 共 72 sessions、3,886 product-days，terminal／cashflow 追至 2026-08-21 共 78 reporting sessions。它是用 q95／Spot-Bid proxy 選出的 conditional matched sample，只保留成 bridge／sensitivity；其中落在 S0.5 的 71 個 full-60 sessions 且進入 broad cohort者是 3,846 product-days，兩個數字不可混用。
- S0.5 primary development panel 是 2026-05-05～2026-08-13 共 71 sessions；2026-01-26～2026-08-13 的 131 sessions只供嚴格 `<D` history，2026-08-14 起 protected forward 未讀取。盤中 anchor 已凍結為 `time_ewma_15s`，D-safe entry q 已凍結為 `Q2_trail20_date_equal`；frozen lower用first legal upper touch當S0.5 proxy。Winner身分使用71日development outcomes，故明標development-selected，不冒充untouched forward。
- S1／S2 不得依 policy／route outcome 各自重選。共同母體固定為 `foundation_selection_s05_rebuild_20260826_v1/s1_mother.parquet` 中 `s1_primary=true` 的 15,638 product-days、71 sessions、244 商品；七組共用。舊 q95 matched manifest 只列 bridge／sensitivity。
- S1 development ranking horizon 固定為 2026-08-13 13:20。仍為 paired open 的部位以該 cursor 最後合法、足量的 causal L1–L5 做共同 hypothetical liquidation mark：long Spot 用 executable Bid、short Future 用 executable Ask；扣已發生逐腿成本與依 current lot acquisition date 計算的剩餘 Spot commission／15 或 30 bp sell tax、Future tax／commission。`economic_ranking_net = terminal_realized_net + open_net_mark`；mark 不生成 request／fill、不算 completion、不釋放 cap。任一 open position 無法共同定價時 withholding economic ranking／Pareto／S2 shortlist。2026-08-14 起 protected forward 不供 development runoff 或補值。
- Entry q baseline 固定為 `Q2_trail20_date_equal`：正負側分開、最多最近 20 sessions、至少 15 日、嚴格 `<D`。`Q2_trail20_date_equal__tod10` 只作 diagnostic；rolling-60、5／10 日 level scale、prior-expiry／DTE 都已在同一 common support 比較，不再於 S1 outcome 後 fine-tune。
- S5 的 September 不重用舊 q95 manifest；它使用最終凍結的 selected-anchor、cohort／selector config，再逐日套 `<D` boundary／liquidity。

### 共用 order、position 與 accounting 狀態

- 抽出共用 `PolicySpec(policy_id, kind, upper_distance_bp, lower_distance_bp, upper_source_id, lower_source_id, upper_source_asof_date, lower_source_asof_date, combined_source_asof_date, entry_tod_bucket, fallback_reason, ...)`；message-load、makerFill adapter、S2 execution target builder、hedge、path、cap與S5 exact都讀同一份spec。D−1 lookup盤前只凍結distance；`actual_new_send_time`才以當下causal anchor±distance轉成absolute price並鎖定，後續fill不得重設。Fixed policy用constant provenance，不偽造lookup as-of。
- Policy 不放進 raw order identity。相同 `Date / ValueCode / QuoteCode / route / stage / maker_side / absolute_price_tick / start_cursor / lifecycle` 共用 `raw_order_fact_id`；這就是唯一的 physical-order lifecycle ID，不另建語意重疊的 `physical_order_id`。另用 `candidate_intent_id` 銜接 scheduler 前的同一掛單意圖、`policy_alias_id` 表示反事實 policy。跨 policy 絕不相加。
- Working-order 狀態至少為 `intent_pending → working_reserved → entry_maker_partial / filled / actual_cancelled / session_expired`；coalesced 或到 nominal stop仍未實際送出的 new不建立 `raw_order_fact_id`。Position狀態至少為 `maker_filled → hedge_pending → paired_open → exit_in_progress → flat`；另有 `entry_partial_rollback_pending / exit_partial_rollback_pending / entry_partial_unresolved / exit_partial_unresolved / entry_hedge_timeout_unresolved / exit_rollback_failed_unresolved` unpaired risk states，絕不餵給只接受 `paired_open` 的 S3 FIFO controller。
- 任一 cursor 的 `total_committed_notional = working_unfilled_reservation + entry_partial_exposure_notional + hedge_pending_notional + paired_open_notional + exit_in_progress_committed_notional`，global 不得超過 20M、單商品不得超過 10M；狀態轉移不重複計額。Entry partial 的 filled exposure＋leaves reservation不得超過原 reservation；exit partial／hedge pending把整個 futures-equivalent unit 從 `paired_open` 移到 `exit_in_progress`，仍按原 position notional計額。每個 reservation／transfer／reconcile／release timestamp 必須可由逐列 ledger 重算。
- `partial_rollback_trigger_cursor = effective_cancel_or_session_expiry`。Entry／exit sub-unit partial從 trigger起進相應 rollback-pending state，在 `[trigger, min(trigger+5s, initiating venue session_end)]` 依共同合法足量book＋相應市場送單額度、同cursor phase與actual-send定價規則反向沖銷已成交 quantity；deadline inclusive dispatch後仍未送出就原子式 expire，轉 partial-unresolved，舊intent不得日後補送。
- Spot Bid exact entry若只 partial fill：working期間照上述 split占額；trigger後賣回已成交 spot quantity，成功才釋放，失敗進 `entry_partial_unresolved`。每個有正 exact entry fill的 order都留在 exact同日率分母；partial rollback不算 intended-cycle completion。S0–S4 legacy adapter仍須明標它看不到此狀態。
- 同 timestamp 的 cap replay 採保守順序：entry reservation 先於 exit release。Opening carry 商品主版維持該交易日 exit-only，不因盤中釋放後重新開倉。
- 成本固定為：現貨手續費**每邊** 1.71 bp；現貨賣出稅當沖 15 bp、非當沖 30 bp；期貨稅**每邊** 0.2 bp、期貨手續費**每邊** TWD 20（`quote_fill/transaction_costs.py`）。主版尚未計 overnight financing、borrow 或 margin opportunity cost，因此另報 overnight notional-days，不把未建模成本說成 0。
- `transaction_costs.py` 要拆出 per-executed-leg primitives；每次 fill／hedge／rollback／forced leg 都寫 append-only ledger：`market / side / qty / price / signed_cashflow / commission / tax / execution_date / inventory_lot_id / acquisition_date`。Spot sell tax 依被解除 lot 的 acquisition date 判 15／30 bp；exit rollback 買回的 spot 是新 lot、以 rollback date 作 acquisition date。Terminal realized net 由逐腿 ledger 重建並歸 terminal Date；未解除的單腿只報 executed cashflow、inventory 與 mark／risk，不混入 realized net。
- 依使用者指定，真正留到 stock-futures expiry 的少量 **paired** residual 採 `expiry_basis_zero_accounting`：用 expiry 日現貨收盤價同時標現貨與期貨，令 terminal basis=0。它是 accounting convention，不生成交易 request／fill／slippage、不算 executable same-day completion；另報 position count、notional 與使用此 convention 的 PnL。Naked／unresolved 單腿不得用此規則洗成 paired flat。
- 13:00 停新倉；spot entry working orders 約 12:59:58 起停止 new 並分散 drain，不能假設 13:00 同秒無限量撤單。
- S1 normal exit的absolute Spot Ask target不重定價，但被動單只有在Spot maker book合法且Future buy L1-L5能完整買足該position時才可維持desired quantity；Spot或Future任一book狀態改變都喚醒重驗。Future gate關閉只撤回desired／送cancel，恢復後仍只能回到原凍結價。這是pre-fill risk guard，不是Future流動性預留，也不取代實際Spot fill後獨立執行的B6。
- 13:19:45固定開始drain所有S1 passive exit；到13:19:49.950最晚安全成交barrier時，所有exit maker new／working／cancel lifecycle必須已terminal，否則partition fail closed。Actual cancel effect前或同cursor的真實fill仍先於cancel並照B6處理。這是S1尾端風險guard，不是S4的taker+taker hard flatten，也不保證市場同步消失時永無裸腿；任何最終`exit_rollback_failed_unresolved`仍保留容量並封鎖排名，不得用common-horizon mark或expiry basis=0洗平。

本文的 venue token 只表示「該市場在該時點剩餘的一個送單 request 額度」，不是商品 token、部位或資金。實作與報表改稱**現貨送單額度**／**期貨送單額度**：以實際送出 timestamp 驗證任意 rolling interval `(t−1s, t]`，spot 最多 100 requests、future 最多 5 requests。優先序固定為 exposed-risk request（emergency rollback → entry／exit hedge → aggressive first leg）→ cancel → new；同級依原始 request cursor、再依穩定 ID FIFO，cutoff drain 的 cancel 再依最積極價格優先。Book／depth 尚不合法的 hedge不進 send-eligible queue，也不 head-of-line block其他已合法 hedge；一旦合法仍須在 dispatch cursor重驗 book，合法者才按上述 FIFO 競爭該市場送單額度。Hedge 在 deadline 前不因暫時沒有額度而 drop；deadline inclusive dispatch後依 B6 原子式 expire／rollback。Cancel intent保留到實際送出或 order terminal，若 terminal使其失效須記 `cancel_not_needed`；已指派 actual send cursor者仍計 request。尚未送出的同 order new intent只保留最新 desired state，已過 nominal stop就不再補送。容量不足的 new可留在 intent queue到 nominal stop，實際送出前才做 C9 reservation；始終未送者標 `cap_blocked`而非 fill rejection。Forced flatten另有 per-position cancel barrier：相關 passive leaves尚未 `actual_cancelled / session_expired`前，其 dependent first leg不是 send-eligible；risk-transition cancel必須先送，期間發生的 maker fills先重算 residual。若送單額度讓 hedge 晚於 `t0`，延遲納入 B6 的 5 秒總窗與 `hedge_retry_delay_ms`，不得另開一個不計價的時鐘。

同 cursor phase 固定為：先 ingest causal raw state並凍結本 cursor 的 send assignment；再讓 cursor前已 working的 order分配 printed volume／potential fill；接著執行已指派的 marketable requests，然後讓 cancel只作用於剩餘 leaves，最後才讓 new成為 working。故既有 order的同-cursor fill先於 cancel成立，已指派 cancel仍耗一筆 request；new不能吃同-cursor fill。C9 new admission以 cursor前 committed balance判斷，不得使用同-cursor exit／cancel release；V0 cancel只在上述 phase後立即生效。`actual_cancel_send_time`與active interval都是此 event loop的輸出。

### 兩個主指標

- Formal `same_day_completion_rate_20m = 20M reservation-admitted、具有任意正 **exact** entry maker fill 的 raw_order_facts 中，在 entry Date內完成 intended paired cycle並以可執行價格完整解除spot＋future風險的筆數 / 全部上述raw_order_facts`。未成交撤單不進分母；entry partial／emergency rollback、unpriced、open、censored留在分母且不進分子；expiry accounting mark不算可執行同日完成。此名稱只用於 entry truth exact 的 route或S5 exact panel；任何forward輸出也必須先滿足exact truth才可使用。
- S1–S4 Spot Bid只能報 `approx_screen_completion_rate_20m`：分母是 adapter-observable approximate **full** fills；Future Ask雖已 exact，也在stage共同screen欄並列，但必須保存 `entry_fill_truth=exact`。Shortlist以當stage的screen欄執行；不得把Spot approximate欄改名成formal exact rate。S5固定五日同表列approx-full、exact-full、exact-any-positive分母bridge；exact只驗證、不回頭重選。
- `terminal_mean_daily_net_twd_20m = 71-session共同 calendar 上、按 terminal Date 入帳的 realized after-cost net cashflow 總和 / 71`。零現金流日納入。
- S1 shortlist 使用 `economic_ranking_mean_daily_net_twd_20m = (terminal realized net + 8/13 common-horizon open net mark) / 71`；必須與 terminal-only 欄並列，另列 open priced／unpriced count、gross mark、已發生成本、remaining exit cost 與 notional，不得把 mark 改名為 realized PnL。後續 stage 若有完整共同 reporting calendar，可在凍結 spec 中改用完整 terminal cashflow，但不得混用分母。
- Spot Bid S1–S4 的 net同樣是 approximate-screen estimate，欄位必帶 `entry_fill_truth=approximate`；只有 exact entry event universe重跑後才標 exact，不因成本公式精確就把fill truth升級。
- 同日率主版以 position count 計；另列 notional-weighted sensitivity。逐月表分 `entry_cohort_month`（路徑歸 entry 月）與 `cashflow_month`（PnL 歸 terminal 月），不得混用。
- `priced_net_bp_among_20m_admitted` 只用 20M 已准入且完整可定價路徑，必須同列 priced／all admitted 分母與 coverage；不能冒充 full-population portfolio EV，也不得誤標為 uncapped。

### Stage shortlist，不是 pass/fail

不把兩個目標任意加權成一個分數。每次 shortlist 保留兩個 objective champion：

1. `completion_champion`：該 stage 可用的同日完成 screen欄最高；同率時依序比日均 net、hedge pricing coverage、`policy_id / route_id / close_policy_id` 字典序。Final formal rate須另帶 exact precision標記。
2. `net_champion`：日均 net 最高；同值（TWD 0.01 內）時依序比同日完成率、hedge pricing coverage、ID 字典序。
3. 若兩者相同，第二名取剩餘 Pareto frontier 中同日完成率最高者；仍無第二個非劣解才取上述 completion 排序的 runner-up。

比較比例時用整數分子／分母交叉相乘，不用 float `==`；同日率分母為 0 時記 null，shortlist 排在任何 defined rate之後。S1 最多留下 2 組給 S2；S2、S3 依同規則更新至多 2 個 finalist。S4 另以 full chronological replay 的 net 差回答「積極平倉是否值得」，不以較高完成率偷代替經濟答案；再對所有 `(entry, exit, close)` 組合套共同 shortlist。因研究定位已指定同日完成率為最佳化目標，S5 spec 明列 primary `completion_champion` 與 secondary `net_champion`；若兩者相同只凍結一組。Exact／September 只驗證，不回頭重選 development policy。

### 每個 stage 的 bundle

每個 S 除 Markdown 外，發布不可覆寫的 `maker/data/walkforward/<stage>_<run_id>/`。大量 append-only rows 可用 deterministic gzip JSONL，摘要可用 canonical JSON；需要 columnar downstream 時才另出 Parquet／CSV，不為格式而複製數 GB。至少包含逐列事實、summary、`run_config.json`、`complete.json`、`verification.json`。`complete.json` 固定保存：

- nested-repo commit 與 dirty flag、runner／schema version、日期與 calendar 契約
- 每個 input path＋SHA-256／inventory digest、config SHA-256
- 每個 artifact 的 rows／bytes／SHA-256，以及 focused tests／verifier 結果

任何 input／config／hash 不同就是新 run，不得續寫 completed root。Markdown 必須連到 bundle 並抄 `complete.json` SHA。
正式比較與 shortlist 只接受 `dirty=false` 的 canonical bundle；dirty worktree run 可作除錯，但不得進主表。

## 工作順序與固定 handoff

```text
S0 8 月歸因 → S0.5 anchor／selected lookup／S1 mother（已完成）
                       → frozen-at-touch convergence／geometry（已完成）
                       → cost-aware lower／unsupported=no-trade 已凍結
                                      ↓
              S1 七組 Spot Bid → S2 Future Ask → S3 exit maker
                                                     ↓
                         S5 spec／exact／Sep ← S4 close policy
```

1. S0 只診斷，不看結果改七組 primary grid；30-session 只能另列 sensitivity。
2. S0.5已選`time_ewma_15s`、`Q2_trail20_date_equal`，發布common S1 mother與frozen-at-upper-touch lower證據。它不選execution champion，也不發布PnL；若只按full-mother availability與截至13:20的reach proxy，C0是q-policy lower的completion-oriented development default proposal，仍待使用者確認。Fixed15–30維持`lower=W`。
3. S1 跑 7 組，依共同規則留最多 2 組 finalist。
4. S2 只跑這 2 組；兩 route backend／sampling 不 pooling，完成後重選最多 2 個 entry finalist。
5. S3 對 entry finalists 先跑第一條 exit route。「跑通」只指 partitions、schema、ledger invariants、verifier 與 tests 通過，與 PnL 無關；工程完成即加第二條 route，再留最多 2 個 entry×exit finalist。
6. S4 對同一 finalist stream 各自跑 carry 與 aggressive；完成後對全部 resulting combinations重新套共同 shortlist，把 primary completion champion與secondary net champion交給 S5。S5 不再搜尋新參數。

### S0　8 月惡化歸因（先做，預估一天）

- [x] 先落地 q95-only quote scheduler／working-order adapter，產生 actual new／cancel cursor；S0與既有 q95結果對帳，S1直接重用其 scheduler core並擴成含 C9／B6／terminal回饋的七組完整 event loop，不得另寫第二套 scheduler。S0這一版明標 `diagnostic_quote_only=true`，未完成前不能以 nominal clock發布 attribution，也不得冒充20M strategy replay。
- [x] 分母分開凍結：market excursion／touch panel用完整 matched-manifest product-days；queue attribution用同一 q95 intent stream的 `cap=∞` quote-only actual-working orders。S0不發布循環依賴下游terminal的20M sensitivity；待 S1 canonical q95 20M bundle完成後回填 robustness panel，且不因此改七組 grid或S1 shortlist規則。
- [x] 新增可重跑 attribution runner。`residual_excursion_bp = basis_mid_bp − anchor_ewma_120s_bp`；primary latent event 是 residual 從 `<= 0` 穿到 `> 0` 開始、到下一個 `<= 0` 結束的正向 excursion，touch cursor 是該 excursion 第一次從 `< D-1 q95 upper` 穿到 `>= upper` 的 raw event。SpreadPair episode只在 touch 後映射 live order，不能拿 order episode 或 compact excursion 的粗時間區間代替 market first-touch cursor。
- [x] 若 product-session 第一個 eligible raw state 已 `> 0`，該 excursion 標 `left_censored=true`；即使起點已在 upper 上方也不得臆造 first touch。Primary excursion 次數／幅度／touch rate排除它，另列 left-censored count、起點已達 upper count與納入 sensitivity後的結果。
- [x] 逐月分開報 residual excursion p50／p80／p95、D-1 q95 boundary p50／p80／p95、每 product-day observable excursion 數、touches／product-day、至少一次 touch 的 product-day 比例。
- [x] First-touch cursor只映射到當時仍在 working 的 `raw_order_fact_id`：`actual_new_send < touch <= active_end`，其中 `active_end` 是 actual cancel send或session expiry，且 `approximate_fill_cursor` 必須為 null或嚴格晚於 touch。`post_touch_fill_rate = approximate_fill_cursor ∈ (touch, active_end] 的 orders / outcome-supported touched raw_order_facts`；另列 BID1／BID2、pooled count 與 product-day 等權值。Nominal submit／stop只作 audit。
- [x] 用 decomposition 明確回答：touch rate 掉是市場／boundary 問題，post-touch fill 掉是 queue／競爭問題，兩者都掉就兩者並列。
- [x] 若 boundary lag 可疑，只加 30-session challenger sensitivity；不加入 S1 七組 primary shortlist，除非另改本計畫。

輸出：`doc/quote_fill/AUGUST_ATTRIBUTION_<YYYYMMDD>.md`；一張逐月表、一張雙 panel 圖、attribution bundle。

完成結果（2026-08-24）：[`AUGUST_ATTRIBUTION_20260824.md`](quote_fill/AUGUST_ATTRIBUTION_20260824.md)。May～Jul pooled → August 的 excursion touch rate為 7.1497% → 4.2923%，post-touch approximate fill為1.8796% → 1.2183%，故分類為兩者並列；30-session market-only sensitivity將August hypothetical touch提高至4.5043%，仍未消除落差，不進S1 shortlist。Canonical bundle為`maker/data/walkforward/august_attribution_s0_20260824_v2`，`complete.json` SHA-256 `5bb3addbd674fc85162630a9f2a7033b1d52bec253dfbb698deabb19cdb3f28b`；30-session bundle為`august_attribution_s0_30_session_challenger_20260824_v1`，marker SHA-256 `680d68bf6cccb9f459b19ed199e82bb7b5c6f0a7bbffb3c468e42ad30e2b15ed`。舊`august_attribution_s0_20260824_v1`有explicit L1 clear forward-fill錯誤，已由v2取代且不得引用。

### S0.5　查表基礎完整重作

- [x] 在 execution outcome 前凍結 registry：wall-clock 15／30／60／120 秒 EWMA、60／120 秒 median；boundary 比 20／60 sessions、5／10 日 level scale、prior-expiry／DTE 與 TOD scale。
- [x] 以 30～300 秒 future median、product-day／month equal 與 5-session paired whole-Date block bootstrap 選出 `time_ewma_15s`；30s 留 rank-2 diagnostic，長 horizon 另列 sensitivity。
- [x] 以 15s residual 從頭重建 131 日 censor-aware episodes，不沿用 EWMA120 distance；每日 prediction嚴格 `source_asof_date < Date`。
- [x] 在同一 common support 選出 `Q2_trail20_date_equal`；TOD10 rank-2 只作 diagnostic，60-session／level5／level10／prior-expiry 都未勝出。
- [x] 以 `frozen_anchor_at_upper_touch` proxy 重建 conditional convergence：C0 center、C1 wide control、C2 reach80、C3 reach50；20／60 日與 resolved fallback都保存。舊逐秒 moving-anchor結果只作 sensitivity，不進lower決策。
- [x] 建立 q-independent cohort；`s1_mother.parquet` 保存 17,006 mapping rows，`s1_primary=true` 為 15,638 product-days、71 日、244 商品，每筆完整 4 TOD×3q×2side。
- [x] 對七組 policy 共用 S1 mother 計 marginal two-sided control geometry；逐列確認 q-policy lower 等於 C1 independent negative-q control，不能冒充正常 conditional exit。
- [x] 以 frozen-at-touch v2 發布 C0／C2／C3 resolved supported geometry與缺值 audit；全部固定 `actionable_execution=false`、`ev_ready=false`。
- [x] 以clean source commit發布selection v1的51-artifact atomic checkpoint bundle；獨立`verify-only`通過。
- [x] 以clean source commit發布frozen v2的18-artifact atomic supplement；獨立`verify-only`重驗所有lineage／hash／row counts通過，2026-08-14起protected forward未讀取。

階段結果（2026-08-26）：[`FOUNDATION_SELECTION_S05_REBUILD_20260826.md`](quote_fill/FOUNDATION_SELECTION_S05_REBUILD_20260826.md)。短期盤中中心選 `time_ewma_15s`（30～300秒month-equal MAE 8.722 bp）；entry q選`Q2_trail20_date_equal`（cross-product Spearman 0.626、same-product temporal Spearman 0.142，屬弱temporal signal）。Selection bundle為`maker/data/walkforward/foundation_selection_s05_rebuild_20260826_v1`，`complete.json` SHA-256 `de7f6d1dddcdbe18875acfb4965d165cc9af6387ec8b02d1e8d55c5766531d60`。

Frozen supplement為`maker/data/walkforward/foundation_selection_s05_frozen_convergence_20260826_v2`，source commit `20e330c66d0632de25f22e112d66f56276b55961`，`complete.json` SHA-256 `ce98f293729588a05904ccb1e97a2902054f201e32fd25c2c1d7d61e960f45c9`，independent `verify-only`通過。q95全mother的C0截至13:20 frozen-mid target reach proxy為92.849–100%；在C2／C3都有lookup的共同cells上，C0／C2／C3 proxy為95.437–100%／77.979–82.596%／52.817–57.534%，未作tick rounding的nominal同日已知成本後margin p50為+0.827／+3.302／+7.466 bp。這不是executable同日完成率。C2／C3 q95 cell coverage只有24.241%，不能刪掉unsupported母體或把composite冒稱reach80。舊moving-anchor結果只作sensitivity；2026-08-25舊報告與bundle保留為predecessor。

S1 前 handoff：

- [x] Anchor／entry q／共同 cohort 已凍結：15s／Q2 trail20／15,638 product-days。
- [x] q-policy lower 已按 cost-aware scenario 凍結：C0 completion baseline、C2 middle sensitivity、C3 deeper-margin sensitivity；fixed20 使用 symmetric lower。C1 wide control不作正常 exit lower。
- [x] Unsupported lookup cell 保留在每組共同 62,552-cell denominator並作 explicit no-trade；不跨 lower fallback、不刪樣、不把 composite 冒稱 reach80。
- [x] 不做 q-independent mother profitability gate。六組在 actual new-send 以各自 same-day／overnight horizon、完整已知成本與 safety floor 做 economic admission；`ctrl_q95_C0_ungated`只作控制且永不進 deployment shortlist。
- [x] Expiry paired residual 採使用者指定的 spot-close／spot-close、basis=0 accounting convention；非 executable、非 same-day completion。

### S1　七組 policy × Spot Bid maker route

- [x] Selected anchor／lookup／common mother只讀canonical S0.5；S1不回接EWMA120或moving-anchor結果。
- [x] 七組 scenario spec、D-safe lookup provenance、explicit unsupported rows與 deterministic common-population hash 已實作。
- [x] 每筆 position 保存`actual_new_send_time`凍結的 absolute Spot Ask exit price／tick；後續 Future Ask不得重設target price，但若hedge side失去合法足量depth，必須撤回desired／送cancel，恢復後仍只可回原凍結價。
- [x] B6完整 raw-state retry、venue request scheduler、timeout／rollback與逐次 risk audit 已實作。
- [x] Normal exit pre-fill gate已要求當下Future buy全量可執行，Spot／Future兩venue book change都會喚醒reconciliation；actual fill仍獨立走B6。
- [x] 13:19:45 deterministic exit drain與13:19:49.950 terminal barrier已實作；同cursor fill先於cancel，任何真正naked unresolved仍fail closed。
- [x] 20M global／10M product pre-send reservation、共同 chronological loop、physical `raw_order_fact_id`／policy alias與 capacity verifier 已實作。
- [x] 2026-08-13 common-horizon open valuation及與 publication/ranking接線完成並通過測試；完整facts決定final open，book價格凍結於13:20，同scenario／商品先聚合數量再掃depth，缺合法足量book即fail closed。
- [ ] 比較表至少含：candidates、supported denominator、`entry_fill_truth`、target 位於 inside-spread／not-passive、B1、B2、B3–5、deeper／invalid 的 product-seconds、fill／actual-cancel／session-expiry／unknown、submit-to-fill p50/p95、venue request peak、hedge pricing/reference coverage、on-time／delayed／timeout／rollback outcome、delay與 slip p50/p95、同日／跨日／expiry／unresolved、20M-admitted priced net bp與coverage、reservation attempts／cap-blocked／sent／unfilled-cancelled／maker-filled、exit desired withdrawal reasons（至少`gate:future_*`／`safety_cutoff`）、cutoff／barrier sessions、日均新 spot、各 committed peak、未平與 naked notional、**20M screen日均 net、`approx_screen_completion_rate_20m`**。
- [ ] 月表同時提供 entry-cohort May／Jun／Jul／Aug（Aug 只到 08-13）與 cashflow-calendar 月份。
- [ ] 依 frozen shortlist 規則輸出兩位 champion 與完整 7 組排序／Pareto 表。
- [ ] 以 clean source commit 完成單 partition smoke。
- [ ] 完成 71×7＝497 partitions、獨立 `verify --verify-inputs`、持久化 verification receipt與正式報告。

輸出：`doc/quote_fill/POLICY_COMPARISON_SPOT_BID_<YYYYMMDD>.md` 與 S1 bundle。

### S2　Future Ask maker → Spot taker route（條件式對照）

- [ ] Future maker 沒有 legacy makerFill，用 `execution_runner` + `indexed_replay`。新增直接讀 manifest `(Date, ValueCode)` 的 CLI／adapter；現有 day-batched engine 可沿用，不重寫核心，只停用舊 `sessions × symbols` wrapper。Target builder 必須讀共同 `PolicySpec`：q 組使用 D-1 upper／lower，fixed 組直接使用常數 bp，不能把 fixed finalist 塞進現有 quantile-only config。
- [ ] Hedge 方向反過來：future maker 一口完整成交後 +50 ms 買 spot；requested shares 必須等於當日 `contract_size`（目前一般為兩個 board lots），以 spot L1–L5 VWAP 定價。Future maker 一口沒有 partial，小於一口不建立 position。
- [ ] 送單額度依市場分開：future maker new／cancel 用 futures rolling 5 requests/s；spot taker hedge 用 spot 100 requests/s 並在現貨額度內優先。B4 已決定不雙 entry route 同掛，所以 alternative routes 不在同一次 replay 共用 futures 額度。
- [ ] 沿用共同的事件式 rolling scheduler／priority／coalesce 契約；actual send cursor 必須回寫 order lifecycle與 fill replay，不能只在結果後按 fixed second 重算。Spot taker hedge完整套 B6 retry／timeout／Future-maker rollback。
- [ ] Future Ask new 在**實際送出前**以 D 日 frozen causal `1.08 × opening_ref_price × contract_size` 做 C9 reservation；future fill只把 working reservation轉入 hedge pending，spot hedge後才用實際 spot VWAP reconcile差額。Timeout／rollback期間照共同 ledger占額，任何 actual spot notional 超過 reservation即 verifier failure。
- [ ] 只跑 S1 的兩個 policy finalists。和 S1 相同欄位，加 route／backend／sampling contract；兩 route 只並列、不 pooling。
- [ ] 報告標題與結論明寫：這是 S1 凍結的 selected-anchor q-independent common cohort 上的 conditional future-route comparison，不是 future route 自己重選的商品池；舊 q95 matched sample另列 sensitivity。

輸出：`doc/quote_fill/POLICY_COMPARISON_FUTURE_ASK_<YYYYMMDD>.md` 與 S2 bundle。

### S3　第二條 Exit maker／route comparison

S1 為了可量測同日完成與成本，已把第一條 exact pooled `Spot Ask maker → Future buy taker` normal exit 工程併入同一 chronological replay；這不表示 S3 route comparison 已完成。S3 現在的主要工作是加入互斥的第二條 `Future Bid maker → Spot sell taker`，並在共同 entry stream／cap／成本口徑下比較兩條 exit route。

- [ ] 新增 product-level FIFO inventory／exit controller，不能直接加總現有逐 position、`joint_volume_allocated=false` 的 counterfactual rows。
- [ ] 第一條 route：在 frozen lower 掛 `Spot Ask maker → +50 ms Buy Future taker`。Pooled FIFO exit 直接用 indexed replay，依共同 visible volume 配 own quantity／partial；不得把 entry 的 legacy makerFill adapter延伸成 exit 真值。
- [ ] FIFO只接受 `paired_open`；entry hedge timeout／rollback pending／naked exposure不得掛 paired exit。Position allocation key固定為 `(position_established_ns, position_id)`；同商品 × route × 絕對價只有一張 active aggregate order，並以共同 `raw_order_fact_id` 識別其 lifecycle。
- [ ] Aggregate order working期間若有同價新 position加入，主版採保守 **cancel-replace whole aggregate**：舊 leaves 到 actual cancel send 前仍可成交；cancel生效後以最新 residual總量送新 order、建立新 `raw_order_fact_id`，全部 queue age歸零。不假設 exchange amend或替新增量偷留舊 priority。
- [ ] 同一 inventory 不得被多價位重複 reservation；每個 position的 quantity只能分配給一個 active working order。同-cursor fill／cancel完全沿用共同 phase。Spot只成交一 lot時把整個 unit移入 `exit_in_progress`，累積到 futures-equivalent quantity才送 futures hedge。
- [ ] Day-order actual cancel／session expiry時，若 cumulative exit fill仍不足一個 hedge unit，對已成交 quantity立即走共同 rollback scheduler，在 initiating venue反向買／賣回並計逐腿成本；成功恢復 paired inventory但新買回spot lot重設 acquisition date，失敗進 `exit_partial_unresolved`且不釋放容量，不得把單腿 partial裸露帶過夜卻仍標 carry。
- [ ] Exit maker fill 後到 hedge 完成前仍占原 cap；future exit hedge 也套 B6 retry。`cap_release_time_ns = exit_hedge_execution_time_ns`。
- [ ] S3 shortlist 前先落地共用 executable **13:20 risk hard-flatten** primitive，供 S4 aggressive 分支重用：逐 futures-equivalent unit、相關 passive-order cancel barrier、first leg 完整足量、第二腿 B6／rollback、兩腿完整才 release。Expiry 不走此 execution producer，另按已定案的 basis=0 accounting convention。
- [ ] 正常 frozen lower 只使用 `Spot Ask maker → Buy Future taker`（第二 route則 `Future Bid maker → Sell Spot taker`），保持 maker＋taker。Taker＋taker 只可作獨立風險 control／S4 13:20 hard flatten，不與 lower 主路徑 OCO，也不拿多輸的 tick 冒充 lower 收益。
- [ ] `carried_eod` 是每日狀態，不是 terminal outcome。Exit-maker 主版互斥 terminal至少含 `exit_maker_flat / expiry_basis_zero_accounting / aggressive_hard_flat / entry_emergency_rollback_flat / entry_partial_rollback_flat / entry_partial_unresolved / exit_partial_unresolved / entry_hedge_timeout_unresolved / exit_rollback_failed_unresolved / unresolved`，另存 `control_taker_taker_terminal`，不得混成一欄；rollback成功後繼續 carry者直到真正 terminal前仍只是 daily state。
- [ ] 跨日 day order 收盤 cancel、隔日重新建單，queue age 不跨日延續；商品離開 entry universe 仍可 exit-only。
- [ ] 先完成 Spot Ask route 的全部 invariant／verifier／tests，再加 `Future Bid maker → Sell Spot taker`。兩條 exit routes先作互斥 alternative scenarios，不雙掛；第二條 route 也用 indexed replay。每個 scenario 內，future maker quotes 與 future taker hedges在同一 5/s venue scheduler 中，hedge 優先。
- [ ] 每個 exit route scenario 都從共同 `candidate_intent_id` stream 做一次完整 event-driven replay：entry／exit request 共用兩個 venue scheduler，scheduler輸出的 new／cancel／hedge cursor再回饋 maker label、B6、`position_established_ns`、FIFO、reservation與terminal ledger，直到 event queue結束。不得把 S1 position／execution cursor當不可變輸入，也不得只事後重算 message-load count；alternative routes仍不相加。

每筆 position 至少有：`entry_policy / entry_route / exit_route / exit_order_id / exit_fill_qty / exit_fill_time / exit_hedge_requested_qty / exit_hedge_executed_qty / spot_residual / future_residual / exit_hedge_slip_bp / carried_eod / terminal_outcome / cap_release_time`。

輸出：`doc/quote_fill/EXIT_MAKER_CAUSAL_<YYYYMMDD>.md` 與 S3 bundle。

### S4　13:00 後 policy：carry vs aggressive

- [ ] `(a) carry`：13:00 後停新倉，Spot Ask exit maker 可掛至現貨收盤；未平者收盤 cancel、隔日重建，opening carry 商品當日維持 exit-only。該 QuoteCode 到 expiry 不再開新 entry；真正仍 paired open 的少量 residual 以 expiry 日 spot close 同時標 spot／future、basis=0 結清並釋放容量，明標 `expiry_basis_zero_accounting`。
- [ ] `(b) aggressive`：13:00 起第一條 exit route改為 Spot A1 peg；加入第二條 route後才有 Future B1 peg，不把兩 route 假裝同掛。13:20 cancel passive peg，對所有 residual 用 L1–L5 taker/taker hard flatten；這是刻意犧牲至少一個 maker tick的風險／容量 control，不是 frozen-lower route。深度不足／5 秒 retry仍失敗要列 `flatten_failure`，不能宣稱已歸零。
- [ ] 13:20 hard flatten 重用 S3 primitive，按 `position_id`／一個 futures-equivalent unit依 FIFO 執行。Spot-exit scenario先賣 Spot taker，Future-exit scenario先買 Future taker；「先賣／先買」只描述 hard-flatten 的 initiating first leg，不是正常 lower 出場。從 trigger起到 `min(trigger+5s, first-leg venue session_end)`，須先越過相關 passive-order cancel barrier，再找本腿 L1–L5足量且相應市場有送單額度的 cursor整單執行。第一腿始終不成立時原 paired unit不變、記 daily attempt `flatten_attempt_timeout`、不釋放；成功後50 ms另一腿走共同 B6。第二腿成功才記 `aggressive_hard_flat`並在其 execution cursor釋放該 unit capacity；第二腿 timeout後 rollback成功則恢復原 paired unit、記 `flatten_rollback_restored`，rollback失敗才留 naked unresolved，一律不釋放。
- [ ] `expiry_basis_zero_accounting` 是非 executable terminal convention；`aggressive_hard_flat` 是新 executable route。兩者在 ledger、完成率與 request load 中完全分開，不把 accounting mark 冒充成交。
- [ ] Carry 與 aggressive 各自從相同 candidate stream 做完整 joint-scheduler／B6／FIFO／reservation／20M ledger replay；新 exit hedge與hard-flatten request必須回饋 entry actual send、hedge execution、position time與後續 FIFO。不能只替既有 accepted cohort換 exit，因 limiter競爭及較早 release都會改隔日 admission。
- [ ] 分兩層比較：matched admitted cohort 的直接 exit 收益犧牲，以及 full sequential replay 新增 admission 後的總效果。
- [ ] `flatten_attempt_timeout / flatten_rollback_restored` 是 attempt／daily state，不是 terminal；兩者都保留 `paired_open`、`carried_eod=true`，下一 eligible session重新掛 exit並再套 close policy。Terminal taxonomy固定至少含 `aggressive_hard_flat / expiry_basis_zero_accounting / entry_partial_rollback_flat / entry_partial_unresolved / exit_partial_unresolved / entry_hedge_timeout_unresolved / exit_rollback_failed_unresolved / horizon_censored_open / unresolved`；只有實際 flat、expiry paired accounting、rollback失敗的 naked unresolved或reporting-horizon censor才寫最終 outcome。固定報：日均 net、同日完成率、隔日開盤可用容量 TWD／%、overnight notional-days、forced-cross cost、expiry accounting count／notional、flatten attempts／failures、matched-cohort sacrifice、replacement entries net。`aggressive full-sequential mean_daily_net − carry` 為正才回答「在目前已建模成本下經濟上值得」。

輸出：`doc/quote_fill/CLOSE_POLICY_CARRY_VS_AGGRESSIVE_<YYYYMMDD>.md` 與 S4 bundle。

### S5　策略 spec、同樣本 exact 校準與 September forward

- [ ] 先發布 `STRATEGY_SPEC_<YYYYMMDD>.md` 與 machine-readable `strategy_freeze.json`。至少凍結 policy、primary／challenger、entry／exit／close routes、cap、cost、B6、venue schedulers、calendar、selector config、code/config/input hashes與 tie-break。
- [ ] Exact calibration 五日固定為 `20260603 / 20260703 / 20260706 / 20260731 / 20260806`；它們是 development calibration，不是 holdout。不得看 exact 結果換日期。
- [ ] Entry-label calibration phase固定吃每個 frozen finalist完全相同的 S1 Spot-Bid entry windows、`raw_order_fact_id`、absolute prices 與 actual active intervals；不能用 generic runner另產 lifecycle，也必須支援 fixed-bp policy。它只比較 label，不改 window。
- [ ] Label-only phase對 primary／challenger 中每個 Spot Bid entry各 rank報 approximate／exact full-fill rate、`k_fill`、full-fill 2×2 confusion matrix、precision／recall、fill-time error、exact partial-order count／qty、quantity-weighted ratio與整日 block uncertainty；固定 window內不生成會反過來改 limiter的partial rollback。Future Ask entry與 S3 pooled exit本來已用 indexed replay，對應欄標 `not_applicable`；若 primary 是 Future Ask但 challenger是 Spot Bid，仍校準 challenger。
- [ ] End-to-end exact phase改從同一 `candidate_intent_id` stream重跑完整 scheduler。依 exact entry fills移除 false-positive positions、加入 false-negative／partial positions，執行 partial rollback並報 outcome，重新生成 B6、S3 exit windows／FIFO／terminal與20M ledger；不能沿用 approximate S3 windows或把 S1 execution cursor鎖死。結果和同五個 entry-day cohort的 approximate replay並列，兩者都追到 terminal／固定 reporting horizon，並列approx-full／exact-full／exact-any-positive denominator bridge；這只是 calibration panel，不冒充連續72-session portfolio estimate。
- [ ] 不把預期約 `0.9×` 當驗收值，也不把單一 ratio線性乘72日 PnL。若 calibration coefficient分母為0則報 null與原始 counts，不補值。
- [ ] `strategy_freeze.json` 必須早於第一個 September action snapshot。外部交易日曆確認 2026-08 完整後，用凍結 selector產 Sep membership；每個 Sep 日 D 開盤前發布 append-only `<D` action snapshot。若只在月底後重建，只能稱 retrospective causal check，不稱 prospective forward。
- [ ] September 使用與 development完全相同 schema／分母，列 absolute／relative delta，不設門檻、不看結果改 spec。Spot Bid若仍走legacy adapter就只報approx-screen欄並旁列五日 calibration evidence，不得線性製造exact PnL／cap或把欄位改名formal；Future Ask或另有full exact entry replay才可標exact。若 primary允許 carry，須追至所有September admitted cohort terminal；否則明列 censor／open notional，不能只因9月raw到齊就稱eventual net完整。

輸出：

- `doc/quote_fill/STRATEGY_SPEC_<YYYYMMDD>.md`
- `doc/quote_fill/EXACT_CALIBRATION_<YYYYMMDD>.md`
- `doc/quote_fill/SEPTEMBER_FORWARD_<YYYYMMDD>.md`
- 三者各自的 frozen bundle／`complete.json`

### 隨各 S 維護

- [ ] `doc/quote_fill/README.md` 隨 stage 完成更新索引與 bundle SHA。
- [ ] 舊 aggressive／exit maker 程式只在需要時由 commit `1348576` 取出參考，重寫後仍須符合 manifest／FIFO／venue scheduler 契約，不能直接把 fixed-45 結果接回主線。

## 目前 data/ 分層

| 層 | 目錄 | 大小 | 本計畫用途 |
|---|---|---:|---|
| 基礎事實 | `walkforward/daily`、`rolling_boundaries`、`liquidity`、`sessions.txt`、`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1` | 21G | S0–S5 輸入；不可覆寫 |
| 因果 manifest | `monthly_product_selector_causal_v2_20260822` | 6M | S0–S4 matched development universe |
| 8/22 q95 baseline | `order_message_load_*`、`one_second_makerfill_*`、`dynamic_future_hedge_*`、`dynamic_expiry_paired_close_*`、`dynamic_estimated_path_portfolio_*`、`august_exit_extension_*` | 230M | S0 baseline；S1 等價 q95 bundle 驗證後才可另議清理 |
| A/B1–2 決策證據 | `makerfill_rank_l1_l5_sample_20260820_v5`、`future_ask_rank_l1_l5_indexed_sample_20260821_v1` | 13M | 僅支持 A/B1–2 與五日 calibration prior |
| 八日 pilot | `fair_mid/`、`quote_fill/`、`quote_width/` | 110M | 歷史診斷，不作 S1–S5 績效分母 |
| S0.5 predecessor | `foundation_revalidation_s05_20260825_v1` | 137M | 歷史重驗；已由完整 selected-anchor rebuild 取代 |
| S0.5 selection v1 | `foundation_selection_s05_rebuild_20260826_v1` | 1,021M | 15s anchor／Q2 trail20／S1 mother／marginal geometry；moving-anchor convergence只作sensitivity |
| S0.5 frozen supplement v2 | `foundation_selection_s05_frozen_convergence_20260826_v2` | 453M | frozen-at-upper-touch C0–C3 reach、C0／C2／C3 support與known-cost geometry；非execution／EV |
