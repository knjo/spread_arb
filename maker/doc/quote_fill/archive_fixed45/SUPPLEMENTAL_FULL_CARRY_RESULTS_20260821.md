# AB1/2 unresolved full-carry 補充情境（2026-08-21）

## 結論

這是獨立的 **analysis-only challenger scenario**，不修改既有 formal
cross-session root。它把原本 1,261 筆沒有 terminal cashflow 的 path 補成可定價：

- 1,239 筆 `maker_fill_state_unknown` 假設當日 exit maker 完全沒有成交，原本的
  long-spot／short-futures full pair 全數帶到下一個交易日，再重播同一套 frozen
  exit policy；
- 若直到期日仍未出場，使用到期 session 最後可用的兩腿 liquidation mark 強制終止；
- 22 筆原本 `right_censored_expiry_settlement_unpriced` 也只用同一套到期 mark
  定價，不假稱取得正式 settlement。

正式補充輸出：
[`prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback`](../../data/walkforward/prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback)

- 3,672／3,672 paths 現在都有非 null terminal cashflow；
- 2,411 筆沿用 source-completed terminal；
- 1,090 筆由後續交易日 normal exit replay 終止；
- 171 筆由 45 個商品／到期 session mark 終止；
- blocker 0，逐 path population 未改變；
- `complete.json` SHA-256：
  `f0902805f2cd555340e09d38232ba0c3a694a56df2ff11043f338cfdfff3c0f2`；
- marker payload SHA-256：
  `c7c4860f94064b438aea671182b93012dbb6cbbbd833914daf42c683ea9f0566`。

這個「全數 carry」是假設，不是從原始成交狀態辨識出的真相。原本 unknown 的
exit maker 可能實際已成交，因此 1,239 筆都明示
`model_imputed_full_carry_on_unknown=true` 與
`double_exit_bias_possible=true`。結果不可稱 exact，也不可直接當 production P&L。

## Replay 與速度口徑

Runner 從 immutable execution action facts 擷取 unresolved path 的實際 entry spot、
future 與 contract size；不重建已完成的 2,411 筆。所有同一
`(session_date, ValueCode, QuoteCode)` 的 active positions 共用一次 session replay，
避免每個 position 各載入一份約 GB 級 timeline。

正式 v2 使用
`one_second_last_state_plus_spread_epoch_approx_v1`：

- spot／future book state 保留每個 recv-time 整秒的最後一筆；
- spot 另外保留每次 `SpreadPair` epoch transition；
- raw spot／future trade prints 不抽樣；
- 最後有效 mark state 必須保留，避免抽樣漏掉 expiry liquidation BBO；
- target／cancel clock、初始 queue snapshot 與 50 ms delayed hedge snapshot 都是近似，
  對應欄位明示為 `exact=false`。

全量共 replay 841 個實際需要的 pair-sessions，約 16 分 49 秒，peak RSS 約
3.95 GB。單一代表 pair-session 的診斷是 exact raw-state 約 20.81 秒／2.07 GB，
近似版約 3.68 秒／852 MB；該樣本終止 path、日期與 spot price 相同，但 terminal
time 約差 28.8 秒。這只證明 runner 實用化，不能把全量近似結果升格成 exact。
`exact_raw_state_v1` 仍保留作小樣本 sensitivity check。

## 到期 mark 的精確命名

到期 position 是 long spot／short futures，因此可執行 liquidation side 優先使用
spot bid 與 futures ask。

|Resolution|Mark groups|Paths|來源語意|
|:---|---:|---:|:---|
|`expiry_last_valid_session_mark`|44|166|最後有效 spot bid＋futures ask；不是 official close／settlement|
|`expiry_last_observed_session_liquidation_mark`|1|5|spot BBO 全 null 時才退回最後正值 spot trade；future 仍用最後有效 ask|

唯一 trade fallback 是 `20260715 / 3374 / QLFG6`：spot 最後觀察成交價
402.0，future 最後有效 ask 402.0。另行檢查完整 raw tape 時，spot `Close` 欄與
13:30 最後成交同為 402.0、future 13:30 最後有效 BBO 為 401.5／402.0；但是 v2
實際 source binding 是 candidate-cache 的最後觀察 trade／BBO，所以欄位仍正確標成
`last_observed_session_spot_trade_not_official_close` 與
`last_valid_session_future_ask`，且 official-close／settlement flags 都是 false。

## `supplemental_paths.parquet` 欄位注意事項

Supplemental terminal 的權威欄位是：

- `filled_entry_outcome_category`、`outcome_status`、`terminal_reason`；
- `terminal_date`、`exit_decision_time_ns`；
- 四個 entry／exit 實際價、`gross_cycle_pnl_twd`、`gross_cycle_bp`；
- `terminal_cashflow_priced`、`supplemental_terminal_resolution` 與所有 supplemental
  disclosure flags。

`outstanding_interval_end_exclusive` 也已更新為 supplemental `terminal_date`。但下列
欄位仍是 formal source row 的歷史欄，overlay 沒有重算，**不得拿來做補充情境的
holding-duration 或 label-timing 統計**：

- `outcome_type`；
- `last_observed_session_date`；
- `label_availability_date`；
- `holding_session_boundaries`。

尤其 `holding_session_boundaries` 必須從 `Date` 到權威 `terminal_date`，依 frozen
交易日曆重新計算交易 session boundary 數，不能直接沿用此 parquet 的值，也不能用
calendar-day 差代替。Combined-cap backtester 的 release event 使用
`terminal_date + exit_decision_time_ns`，不讀取上述 duration／label 欄位，因此 cap
准入與釋放結果不受這個 overlay schema 限制。

## Artifact

|檔案|Rows|內容|
|:---|---:|:---|
|`unresolved_entry_prices.parquet`|1,261|immutable action facts 擷取的實際 entry 兩腿價與 shares|
|`source_inventory.parquet`|730|讀取的 action／exit partition identity 與 hash|
|`continuation_terminals.parquet`|1,090|後續交易日 normal exit terminals|
|`expiry_marks.parquet`|45|到期商品／契約 liquidation marks；含 BBO／trade fallback disclosure|
|`continuation_audit.parquet`|1,261|每個 source-unresolved path 的 replay／resolution audit|
|`candidate_session_sampling_audit.parquet`|841|每個實際 replay session 抽樣前後 row counts 與 exactness flags|
|`supplemental_paths.parquet`|3,672|保持原 population 的 terminal overlay|

## 重跑與測試

預設 CLI 是上述一秒＋epoch-change 近似；輸出目錄必須是新路徑：

```bash
uv run python -m maker.src.quote_fill.supplemental_carry_runner \
  --output /tmp/supplemental-carry-rebuild
```

先測一個 pair-session、或跑 exact sensitivity：

```bash
uv run python -m maker.src.quote_fill.supplemental_carry_runner \
  --benchmark-max-pair-sessions 1

uv run python -m maker.src.quote_fill.supplemental_carry_runner \
  --replay-mode exact_raw_state_v1 \
  --benchmark-max-pair-sessions 1
```

核心測試涵蓋：exact entry price extraction、同商品日只 replay 一次、unknown
跨多日 carry、expiry BBO／trade fallback、抽樣保留 epoch 與 last-valid state、四腿
fee arithmetic、同時 portfolio／product cap、FIFO release 與 fail-closed inputs。

這個 bundle 仍有固定 45 檔事後 cohort 的 universe leakage、state sampling 近似與
unknown full-carry counterfactual；`production_strategy_go=false`。
