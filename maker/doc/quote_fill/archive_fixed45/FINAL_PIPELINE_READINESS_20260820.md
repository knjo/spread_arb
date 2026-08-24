# 45×60 Exit Maker：最終下游管線與 EV readiness

> 更新：2026-08-20（Asia/Taipei）  
> 本文是下游執行契約，不是最終績效。正式數字只能在 same-day 2,687 個可用 product-days 與 cross-session root 都完成並通過 hash／lineage 稽核後發布。

## 狀態結論

| 階段 | 狀態 | 正式輸出語意 |
|---|---|---|
| Same-day exit-maker report | 等待 2,687 partitions 與 root manifest | 已建倉 entry 的 Center／Lower × route 同日 execution 診斷；19 bp 僅成本敏感度 |
| Cross-session frozen continuation | 有條件可跑 | 每天重建 DAY queue，沿 entry 時凍結的 Center／Lower；strict 與 nominal V0 分檔 |
| Filled-entry primary report | 程式面 GO，等待完整 cross root | 分母只含 entry full-fill 且 50 ms hedge executable；primary 只疊 nominal V0 cross outcomes |
| Contextual fixed-policy lookup | 資料 adapter NO-GO | 只能在建倉時選固定 Center／Lower × route；目前缺完整成本與有限期 terminal 契約 |
| 盤中任意切換／Bellman controller | NO-GO | 缺共同 decision clock、完整 legal actions + WAIT、next-state transitions 與 sequential replay |

## Primary 分母、strict／nominal 與成本

1. Filled-entry primary 的 profit denominator 是每個 policy cell 內，實際 `full_fill=true` 且 entry 50 ms hedge `executable` 的 physical position。未成交、partial、fill unknown、hedge failure 只進 execution diagnostics，不補 0 PnL。
2. Cross-session primary overlay 固定讀 `cross_session_nominal_policy_outcomes.parquet`。`strict` outcome 仍保留在自己的 artifact，不能混入或相加。
3. `completed` 才能是 `outcome_type=terminal`、`terminal_cashflow_priced=true` 且有 finite gross；`still_open`／`censored`／`unknown` 必須留在 `outcome_type=censored` 並保持 null cashflow。Primary gross 統計只對 completed cycles，不是 per-submitted-order EV。
4. Same-day exit report 的 `gross - 19 bp` 只是 analysis-only fee/tax/commission sensitivity。Filled-entry primary 不產 net 欄位，也沒有套完整 cost profile。
5. 四腿 gross 已使用實際 entry／hedge／exit 價格，因此 latency/depth slippage 不可再扣一次。真正 EV 仍須逐路徑加入有版本且 D-safe 的 fee、tax、commission、financing、overnight、cancel、emergency、other-risk costs。
6. q、Center／Lower 與兩條 exit route 都是替代政策。跨 q 共用 physical fill、同一 entry 的四個 exit alternatives、以及 route copies 都不能當獨立可加總部位。

## Cross-session 正式前置 gate

`maker/data/walkforward/sessions.txt` 只到 `20260813`，但多個 August contracts 到 `20260819` 才到期。若直接使用原 sessions 檔，最後幾個 entry sessions 不會讀取 `20260814／17／18／19`，而會提早變成 observation-end censor。

正式 cross run 必須先：

- 保留原本 131-session source calendar並加入 `20260814／17／18／19`，成為 135-session 遞增 candidate calendar；原 60-session entry cohort仍由精確的 2,687-row entry manifest鎖定，不能因延長 candidate calendar而改變。
- 產生並稽核 `20260814／17／18／19` 的 exact-contract metadata。這幾天的 spot tick、future tick、marketData 與 tickFeature 已存在；目前 `maker/data/walkforward/daily/metadata` 只到 `20260813`。
- 使用 entry manifest 的精確 2,687 個 Date／ValueCode 作 `--product-days`。不可用 extended calendar 的 trailing 60 sessions 重新 discover，否則會丟掉最早 entry days。
- 保留 exact `QuoteCode`，不偷換月份；資料觀察期仍早於到期的部位只能標 censored，不能補 0 或假設結算價。

前置檔已原子發布並由 `complete.json` 驗證在 versioned root `maker/data/walkforward/cross_session_prerequisites_v1_20260819`：135 個 candidate sessions、135 個逐日 metadata、2,619 個 exact-contract candidate requirements，以及 180 個 extension mapping audits。正式 CLI 強制要求 `--prerequisite-root`，而且會在建立 output／candidate cache 或讀取 raw tape 前，先用 public verifier 重驗 marker 與所有 artifact hash。`--sessions`、`--contract-calendar`、`--contract-metadata-root` 必須就是該 versioned root 內的 artifacts；`--product-days` 必須是 marker 綁定的原始 2,687-row cohort 且 content hash 完全相同。Entry／data／futures source roots 也必須吻合 marker lineage，不能再用舊 `sessions.txt`、`daily/metadata` 或內容相同但未被綁定的複本：

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache uv run --no-project \
  python -m maker.src.quote_fill.exit_maker_cross_session_cli \
  --prerequisite-root maker/data/walkforward/cross_session_prerequisites_v1_20260819 \
  --sessions maker/data/walkforward/cross_session_prerequisites_v1_20260819/candidate_sessions.txt \
  --contract-calendar maker/data/walkforward/cross_session_prerequisites_v1_20260819/exact_contract_calendar_v1.parquet \
  --product-days maker/data/walkforward/execution_narrow_60d/execution_partition_manifest.parquet \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --output maker/data/walkforward/exit_maker_cross_session_narrow_60d \
  --data-root /home/kevin/Project/HFT/data \
  --futures-root /mnt/NAS/Parquet/Ticks \
  --contract-metadata-root maker/data/walkforward/cross_session_prerequisites_v1_20260819/metadata \
  --hedge-delay-ms 50 \
  --book-age-diagnostic-ms 1000 \
  --candidate-cache-root maker/data/walkforward/exit_maker_cross_session_narrow_60d_candidate_session_cache_v8 \
  --candidate-cache-max-entries 32
```

正式 runner 版本為 `frozen_exit_maker_cross_session_runner_v8_fixed_output_schemas`。每個 cross partition 的 `complete.json` 與 root `cross_session_partition_manifest.parquet` 都會帶入 prerequisite root、marker file SHA-256、canonical marker-payload SHA-256、config SHA-256 與 source-identity SHA-256。Verifier／resume 不信任 marker 自報的 `runner_config_sha256`，而會從 embedded `config.runner` 重算 canonical SHA-256，再同時核對 marker、resume expected identity，以及 marker／partition config／runner config／runner global inputs內四份 prerequisite identity；任一份不一致即 fail closed。Runner 的 implementation identity 也包含它實際呼叫的 `exit_maker_report.py` 完整性 validators，日後 validator 語意改變會自然產生新的 runner hash，不能與舊 partition 混用。

V8 另把 filled-entry primary 的 upstream integrity 變成正式 runner 契約：entry marker、same-day exit runner與每筆 full-filled action的 hedge delay都必須是 50 ms；`full_fill／partial_fill／any_fill`、full-fill cursor、hedge label與 executable status必須一致；每個 established entry 必須恰有 `frozen_center／frozen_lower × future-maker／spot-maker` 四個 action，不能缺 Lower、加未知 route或重複；rule source必須等於 ordered session calendar中的真正前一個 session，而不只是字串上早於 Date。Same-day position的完整 identity、threshold、source、trial grid與 establishment cursor會再回連 marker/hash綁定的 entry `exit_facts.parquet` 與 entry action。V8 也明確把 unsupported／non-order action 的 nullable `full_fill` 視為未建立部位；只有 explicit `full_fill=true` 才能進 admission，而 explicit full-fill 若缺 physical hedge status仍然 fail closed。五個 cross artifacts 都由 producer 與 public verifier 共同鎖定 ordered columns／dtypes；audit 的空 candidate list固定為 `List(String)`，failure status／detail即使全空也固定為 `String`，因此 zero／nonzero／failure partitions 可安全直接跨檔 scan。

建立部位的 eligibility 固定使用 alias-local `full_fill=true AND entry_hedge_label_observed=true AND entry_hedge_executable=true`。`entry_hedge_status` 與 `entry_hedge_decision_time_ns` 是 canonical physical raw-order facts：不同 q／stop cursor 的 alias 可能共用同一 raw order，其中較短 alias 已 no-fill 停止，較長 alias 才稍後成交，所以 no-fill alias 合法保留 physical `status=executable` 與 hedge cursor；此時 alias-local hedge booleans 必須明確為 false，且該 alias 不得進 primary 分母。

在建立任何目錄前，output 與 candidate-cache 路徑都會先 canonicalize，並與 entry root、exit root、prerequisite root、`data_root` 及 `futures_raw_root` 做雙向 ancestor／descendant disjoint 檢查。即使 cache 暫時 disabled，指定的 cache root 也不能指向或落在 raw source tree內。

Book age 在這個研究仍只是診斷，不是事後篩掉成交的 freshness gate。

若程序收到 `SIGTERM`／主機中斷，原子 rename 前可能留下隱藏的 `.ValueCode=*.tmp-*` partition stage、`.cross-session-day-policy-*` spool 或 candidate-cache hidden stage。這些目錄沒有正式 `complete.json`，不會被 resume 當成完成 partition；重啟前先確認舊程序已停止，再只針對該次正式 output/cache roots 內的精確 hidden-stage 路徑盤點與清理。不可刪除已完成的 `Date=*/ValueCode=*` partitions，也不可用寬泛遞迴刪除。現階段不攔截 `SIGTERM` 強行 publish，避免把只寫到一半的 stage 升格為正式結果。

## Same-day 與 filled-entry 報表命令

Same-day 2,687 partitions 與 `exit_maker_partition_manifest.parquet` 完整後：

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache uv run --no-project \
  python -m maker.src.quote_fill.exit_maker_report \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --output-dir maker/data/walkforward/exit_maker_narrow_60d/report_60_sessions_final \
  --sessions 60 \
  --session-calendar maker/data/walkforward/sessions.txt \
  --assumed-non-price-cost-bp 19
```

Cross root 完整後，primary report 直接讀 partitioned root，不需手動 concat：

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache uv run --no-project \
  python -m maker.src.quote_fill.filled_entry_report \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --cross-session-root maker/data/walkforward/exit_maker_cross_session_narrow_60d \
  --prerequisite-root maker/data/walkforward/cross_session_prerequisites_v1_20260819 \
  --output-dir maker/data/walkforward/exit_maker_narrow_60d/filled_entry_primary_60_sessions_cross_nominal \
  --sessions 60
```

Same-day loader在正式模式強制要求 `exit_maker_partition_manifest.parquet`；缺 manifest 不再視為可接受的 partition 集合。`--cross-session-root` 必須同時提供 `--prerequisite-root`，而且 cross marker、partition config、runner、global inputs與root manifest中的 prerequisite identity都必須吻合該 root目前驗證出的精確 marker identity；`None` 或另一個雖自洽但非預期的 prerequisite都會拒絕。

Cross loader會逐 partition 驗證 marker、五個 artifact hash、root manifest、row counts、共同 runner config，並把 cross outcome 的 trial identity、entry identity、rule／route、真正 session predecessor的 D−1 rule source與 frozen threshold對回 same-day position facts。已在 same-day完成的 trial保留原本 `Date／terminal_reason／gross` terminal tuple，cross overlay只可補仍 carry／open／unknown的 trial；completed必須有 `terminal_date >= Date`，未完成則 terminal date與cashflow都必須為 null。缺一個 partition、混入 strict、policy ID多／少一筆、缺 Center／Lower action、threshold/source漂移或重寫 same-day winner都 fail closed。

Filled-entry report marker版本為 `filled_entry_primary_report_v2_alias_local_provenance_bound`，並記錄本身、same-day validator、cross verifier、prerequisite verifier與 formal CLI 的逐檔 SHA-256及其 canonical identity hash；報表程式語意變動不會沿用同一個未綁定的完成標記。

## Contextual lookup：可推導與真正缺口

現有 facts 可推導 physical entry、q、Center／Lower、route、position-establishment cursor、實際四腿 gross 與 cross terminal category；因此「建倉時從四個 frozen exit alternatives 選一個」在工程上可做。

但 `finite_horizon_policy.py` 的正式 normalized contract 目前不能由現有 roots 直接發布：

- raw entry schema 對 normalized action contract 尚缺 40 個欄名，cross outcome schema對 terminal contract 尚缺 47 個欄名；其中不少 identity/state 欄可 deterministic derive，但尚無 root-aware adapter。
- 沒有 D-safe、versioned、hashed component cost profile。Known terminal path 不得用單一 19 bp scalar 或 null 成本冒充 fully costed path。
- Cross maker policy目前跑到成交、到期或 observation censor，沒有一致的 `H`-session executable terminal policy；finite-horizon lookup 預設要求 origin + H 的明確 horizon 與 terminal rule。
- Strict cancel race 仍無 cancel ACK 可識別。Nominal V0 可以做模型敏感度／challenger ranking，不能改名為 strict production EV。
- Filled-entry population只能估「已建倉後怎麼出」。若要同時最佳化 entry 掛價，必須回到 per-submitted-entry denominator，保留 no-fill／partial／hedge-failure 與 cancel cost；不能拿 filled-only profit 表反推 entry admission EV。
- Intraday transition contract共有 63 欄；現有 18 欄 same-day transition facts只同名對上 `Date`，仍缺 62 欄所代表的共同 clock、完整 action set、唯一 WAIT、action-to-next-state tuple、inventory／OCO feasibility 與完整 sequential replay。不能把同一 terminal PnL 貼到每個盤中 observation。

所以最終可以安全發布的層級應分成：

1. actual filled+hedged conditional gross report；
2. 19 bp 與其他明示假設的 cost sensitivity；
3. nominal V0 fixed-policy challenger table（仍標 `pathwise_ev_ready=false`）；
4. strict unknown/censor audit；
5. 只有在有限期 terminal + complete component costs + D-safe support/LCB 全部通過後，才發布真正 EV lookup。
