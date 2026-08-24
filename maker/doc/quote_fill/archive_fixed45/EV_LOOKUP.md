# 查表 EV、合法掛價與 Execution Replay 契約

## 目前狀態

`maker/src/quote_fill/ev_surface.py` 已實作「合法 tick action → D-safe pathwise lookup → decision-time score」的純表格介面；`execution_facts.py` 與 `execution_runner.py` 已提供單一商品日、memory-bounded 的 raw order／fill／50 ms hedge／保守同日 taker-exit facts 與原子分區契約。

這不表示策略已經 EV-ready。目前八日、四商品 pilot 仍缺完整 exit maker replay、券商／帳戶／交易日版成本設定、overnight terminal cashflow，以及多商品共同成交量與部位配置；所有 pilot action／exit 表都必須維持 `ev_ready=false` 或 `pathwise_ev_ready=false`。131 日已完成的是 latent boundary 與 liquidity layer，raw execution／terminal EV 尚待依本文件的 rollout 擴樣。

目前`execution_runner.py`仍以事前設定的q50／q80／q95 policy aliases進行raw replay；`enumerate_legal_tick_actions()`與`score_action_surface()`的全legal-tick介面尚未接到sequential raw／portfolio runner。因此「介面已實作」不等於「實盤掛單器已完成」。

## 從查表到掛單

```text
<D rolling boundary/state
    -> enumerate every legal passive tick
    -> decision_actions
    -> join lookup(asof_date=D)
    -> ev_ready && action_score >= floor
    -> one selected quote per decision/route
    -> sequential portfolio gate
```

`q50／q80／q95` 是決定搜尋範圍與研究分層的 latent knots，不是掛價 allowlist。實盤式 action surface 應由當下反腿 executable quote、交易所 tick ladder 與事前凍結的上下範圍列舉每個合法被動價；固定 10–20 bp 或固定每天交易次數都不是決策條件。

### Canonical action 與 decision action

- Canonical raw action 以 `Date × ValueCode × QuoteCode × route × stage × spread_pair_epoch × absolute_target_tick` 識別。同一 epoch 內，多個 threshold alias 或多次 decision 若 round 到相同絕對價，只建立一個 `action_id`／raw queue fact；candidate 的 lifecycle policy、queue scenario 與數量必須一致。
- `decision_id` 只作稽核，不進 canonical identity。`actions` 保存第一個 submit state；跨 epoch 即使價格相同仍可建立新 generation。
- `decision_actions` 保留同一 canonical action 在不同 causal decision clock 的出現，才是 `score_action_surface()` 的輸入。它讓 D-safe lookup 在每次決策重新評分，但不複製實體 fill 樣本。
- `aliases` 保存 q-knots、threshold 與非法／不合格原因。模型或報告可以比較 alias，physical fill／hedge support 必須以 `action_id` 或 `raw_order_fact_id` 去重。

`score_action_surface()` 只在 exact lookup cell `ev_ready=true` 且 score 達門檻時排名，同一 `Date／ValueCode／QuoteCode／route／decision_id／decision_sequence` 最多選一個價位。既有 live layers、reservation、總部位與多商品資金競爭仍由後續 sequential portfolio replay 決定。

## Execution runner 的事實表

`execution_runner.py` 一次只持有一個 product-day raw tape，重用 maker trade index後寫入：

- `order_aliases.parquet`：policy／quantile aliases與精確 submit、fill、nominal cancel cursors。
- `raw_order_facts.parquet`：以 `raw_order_fact_id` 去重的實體 maker order結果。
- `hedge_facts.parquet`：full fill後固定50 ms的反腿L1–L5 executable VWAP／depth／book age。
- `execution_action_facts.parquet`：alias層的entry、hedge與實際鎖定basis；同價 aliases仍指向同一 raw fact。
- `execution_daily_facts.parquet`：action-day的分子、分母與slippage統計。
- 可選 `exit_facts.parquet`／`exit_daily_facts.parquet`：事前凍結 exit rule下的同日 raw taker/taker hit或`carry_at_eod`。
- `policy_audit`、target／raw tape audit、artifact hashes與`complete.json`。

每個分區保留 `SpreadPairTotalCount` clock：同 epoch 同價不重掛、target向前的新價可加層、跨 epoch 同價可新增研究 generation。分區是 independent-candidate replay，不會讓重疊假想訂單共同消耗成交量或更新共享 inventory，因此不能直接加總成 portfolio PnL。

131日輸入的public API為`load_walkforward_product_day_sources()`、`load_walkforward_execution_product_day()`與`run_narrow_universe_execution_replay()`；泛用測試／自定loader保留`run_partitioned_execution_replay()`。Source loader只投影execution-safe mapping allowlist、causal fair anchor與嚴格`<D`的rolling boundary，不讀target-day excursion outcome。

### 撤單與 hedge 指標

日表至少分開保存：

- policy alias orders、unique raw orders、queue-known與same-price alias數。
- any／full／partial fill及其 observed denominators。
- `cancel_required`、target-retreat與session-cutoff cancel；這是策略需要撤單的事件，不是已觀察到交易所 cancel ACK。
- full-fill精確三段 cursor、50 ms hedge label coverage、executable／depth-shortfall。
- hedge latency、depth與total slippage；正值統一代表不利，並另外保存book freshness。

實體 fill／hedge base rate必須以 unique raw path計算。Alias率只回答某個 policy 的條件表現，不能把落在同價的 q50／q80 當兩筆獨立成交。`P(hedge executable | full fill)` 與slippage也只用明確觀察到的條件分母，不把missing當零成本。

可選raw taker-exit的`carry_at_eod`只表示收盤尚有部位，是待後overnight label，不是terminal outcome；未實現現金流不可用EOD mark偽裝成平倉。

## Terminal path 與 EV

`build_daily_ev_lookup()` 的機率單位是一張 admitted quote 的一條完整路徑。Known path必須且只能落入下列一個 terminal branch：

```text
no_fill_cancel
no_fill_expire
partial_unresolved
hedge_failure_emergency
same_day_target_exit
same_day_aggressive_exit
overnight_exit
overnight_unresolved
expiry_forced_flat
expiry_settlement
emergency_exit
```

`censored`／`unknown` 不准假裝成 no-fill、零 PnL或任一 terminal branch；它們保留在 admitted-quote denominator與unpriced mass。EV直接平均每條完整 path 的實際 fill cashflow減成本，也等於互斥 branch probability乘branch conditional cashflow的總和。不得把邊際 `P(fill) × P(hedge) × P(exit)` 相乘。

每條 known path需提供實際 fill cashflow、capital time、versioned cost profile及fee／tax／financing／overnight／cancel／emergency costs。50 ms hedge與exit slippage是診斷欄；若實際四腿fill cashflow已反映成交價，不能再扣一次slippage。

第一版 score為：

```text
expected net cashflow per admitted quote
- capital-time penalty
- expected downside penalty
- emergency-rate penalty
```

Inventory penalty與跨商品資本配置不在這張 lookup 內，須由portfolio replay另加。

## Strict 60-session EV audit（as-of 20260813）

以`execution_narrow_60d`的60個session、5個商品做strict audit；允許不完整商品日grid，實際讀到290／300個product-day。`strict_terminal_paths`共有417,870條policy path，其中as-of日20260813的3,580條只供稽核、不進lookup；lookup只使用截至20260812成熟的414,290條prior path，且`execution_safe_snapshot=true`、`contains_target_day_outcome=false`。

| outcome status | prior paths | 占比 | strict解讀 |
|---|---:|---:|---|
| known | 3,685 | 0.8895% | 僅`same_day_aggressive_exit`，有entry／hedge／exit成交事實與gross cashflow |
| censored | 5,289 | 1.2766% | `carry_at_eod`；等待overnight／expiry terminal label，不以EOD mark定價 |
| unknown | 405,316 | 97.8339% | cancel ACK／race 301,002；entry fill／queue 88,332；未配到exit rule 15,780；partial 196；hedge／emergency 6 |

49個partition沒有`exit_facts`，其中45個有admitted actions；這15,780條路徑以`__unassigned_exit_rule__`逐條保留為unknown，沒有丟棄，也沒有複製到center／lower規則。3,685條known path雖都有execution與hedge-slippage觀測，但cost-complete、exit-slippage-complete與EV-complete皆為0；fee、tax、financing、overnight、cancel、emergency六類成本及cost profile仍為null，不是零。其gross cashflow只能作「已知同日出場條件下」的診斷，不能當每張admitted quote的EV。

以`state-collapse=all`建立90個lookup cell，結果為`priced_expected_ev_cells=0`、`ev_ready_cells=0`：30格`insufficient_group_sessions`、46格`insufficient_known_paths`、14格`unpriced_censored_or_unknown`。要得到可計價EV，仍缺實際cancel ACK／cancel-race、entry queue/fill補標、overnight／expiry現金流、exit slippage reference，以及有版本的完整成本設定；strict audit沒有啟用nominal cancel model，也沒有把unknown、carry或null cost補成0。

產物：[audit_config.json](../../data/walkforward/execution_narrow_60d/ev_audit_60_sessions_asof_20260813/audit_config.json)、[strict_terminal_paths.parquet](../../data/walkforward/execution_narrow_60d/ev_audit_60_sessions_asof_20260813/strict_terminal_paths.parquet)、[asof_ev_lookup.parquet](../../data/walkforward/execution_narrow_60d/ev_audit_60_sessions_asof_20260813/asof_ev_lookup.parquet)、[terminal_path_summary.csv](../../data/walkforward/execution_narrow_60d/ev_audit_60_sessions_asof_20260813/terminal_path_summary.csv)、[ev_lookup_status_summary.csv](../../data/walkforward/execution_narrow_60d/ev_audit_60_sessions_asof_20260813/ev_lookup_status_summary.csv)。重跑命令：

```bash
uv run python -m maker.src.quote_fill.execution_ev_audit \
  --execution-root maker/data/walkforward/execution_narrow_60d \
  --sessions 60 \
  --allow-incomplete-product-grid \
  --state-collapse all
```

## Support 與 not-ready 行為

預設 lookup key為商品、route、相對BBO tick、state family／bucket、lifecycle policy version、queue scenario與entry／hedge數量。每日 D 只讀最近60個session且 `label_end_date < D` 的成熟路徑；任何 target-day outcome都不准進prior。

實作預設最低條件為40個window sessions、cell內20個training sessions與100條known paths。現行v1 gate更保守：所有 admitted paths都必須known，且known path的cost、execution status與所需slippage diagnostic必須完整；只要仍有未定價censor／unknown，就回傳not-ready而不是以known subset冒充per-admitted-quote EV。

表會保留 `n_paths`、各coverage／unknown率、branch mass、cancel／fill／hedge／exit diagnostics與明確 `ev_status`，例如：

- `insufficient_history_sessions`
- `insufficient_group_sessions`
- `insufficient_known_paths`
- `unpriced_censored_or_unknown`
- `incomplete_cost_inputs`
- `incomplete_execution_fields`
- `incomplete_slippage_diagnostics`
- `non_exhaustive_terminal_branches`
- `ready`

目前沒有hierarchical fallback，`fallback_level=none_product_state_only`。Exact cell不足就不掛；`score_action_surface()` 找不到表列時標成`missing_lookup`。正式 rollout前要凍結單一cost／policy版本；lookup雖會報告cost profile版本數，不能把不同費率語意默默混成同一production cell。

## 到期日與隔夜

到期日不能把舊 `QuoteCode` 的未平倉部位直接接到次月合約。完整 terminal path只能選：

1. 到期日前／當日以可成交價格force-flat；
2. 依明確且versioned的settlement規則形成`expiry_settlement`；
3. 明確roll，並把舊約平倉與新約建倉都記成現金流。

`taker_exit.py` 的一秒保守benchmark預設在expiry day未達target時，使用當日最後一筆合格、可成交的spot bid／future ask強制平倉；若沒有合格book則保持unresolved，不跨月偷換contract。非到期日可用exact同合約的下一session首個合格book作一日carry benchmark。這仍不是exit maker model，也不是完整overnight policy。

`ev_surface.py` 會檢查 `expiry_forced_flat ↔ expiry_status=forced_flat` 與 `expiry_settlement ↔ expiry_status=settled` 一致；expiry unresolved若尚未定價，整個cell維持not-ready。

## 131-session rollout

2026-01-26～2026-08-13的131個共同交易日已完成latent daily facts、60-session rolling boundaries與route liquidity screen。目前這批daily markers為validator可接受的`daily_latent_facts_v2_migrated_nonatomic`；它們保留`legacy_atomic_publish_verified=false`與`legacy_writer_provenance_verified=false`，可作development／pseudo-holdout研究，不能冒充pristine locked-forward資料。Execution／EV依下列順序擴樣：

1. 先用少量product-day smoke驗證分區schema、hash、resume、exact cursor與D-safe lineage。
2. 對43檔strict core逐product-day回播自然base rate，再跑20檔extension與10檔wide-future controls；先用已凍結q50／q80／q95 aliases驗證runner，再將同一raw contract擴到全legal-tick actions。一律保存零order／零fill action-day，避免幸存者偏誤。
3. 將raw entry、50 ms hedge、同日exit、overnight／expiry與完整cost profile組成互斥terminal path；未成熟label依 `label_end_date` 延後進表。
4. 每日產生60-session lookup與decision snapshot。May fine-tune、June confirmation、Jul–Aug pseudo-holdout依[WALK_FORWARD.md](../WALK_FORWARD.md)既定切分報告，不在看完結果後改gate。
5. 凍結estimator、queue／cancel、exit／expiry、cost與portfolio config hash；2026-08-14後的新完整共同資料才累積真正locked forward結果。

Production-like baseline在滿60個prior sessions後才評分；程式的40-session最低support只供causal burn-in／smoke，不把未滿60日結果冒充正式表。當前八日pilot只驗證介面與數量級，不能用來做商品排名、EV admission或穩定獲利宣稱。

完整研究狀態見[PILOT_RESULTS.md](PILOT_RESULTS.md)、[LIQUIDITY_SCREEN.md](LIQUIDITY_SCREEN.md)與[../04_BACKTEST.md](../04_BACKTEST.md)。
