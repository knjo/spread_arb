# Maker Research Source

2026-08-24 清理後只保留兩層：動態商品池因果 pipeline，與它之後要借用的 exact replay 引擎。
固定 45 檔分析層（56 個模組）已刪，可由 nested repo commit `1348576` 撈回。

```text
src/
├── common/        # data contract、路徑、landmarks
├── fair_mid/      # WP01 EWMA120 anchor、八日 pilot（保留供重驗）
├── quote_width/   # daily_facts（每日 causal fair／excursion）、rolling（60-session 界線）、table
├── quote_fill/
│   ├── 因果線：monthly_product_selector → one_second_message_load(_runner)
│   │           → one_second_makerfill_runner → dynamic_future_hedge(+provenance_migration)
│   │           → dynamic_expiry_close → dynamic_estimated_path_portfolio → portfolio_cap_backtester
│   │           → august_exit_extension；liquidity、walkforward、universe_manifest(_cli) 是基礎事實層
│   ├── S0.5 predecessor：foundation_anchor、foundation_boundary、foundation_geometry
│   │                     → foundation_revalidation_runner（EWMA120 基礎重驗）
│   ├── S0.5 selection v1：foundation_anchor_selection、foundation_boundary_{batch,adaptation,
│   │              orchestration,selection}、foundation_convergence_{lookup,selection}、
│   │              foundation_cohort_selection、foundation_selected_geometry、
│   │              foundation_selection_stats／registry → foundation_selection_runner
│   ├── S0.5 frozen lower v2：foundation_frozen_convergence_runner
│   │                            （已canonical發布的upper-touch frozen supplement）
│   ├── cost-aware S1：s1_scenario_spec／s1_target／s1_economic_gate／exit pre-fill risk guard
│   │                  → s1_entry_state_adapter／s1_event_loop／s1_entry_day_runner
│   │                  → s1_bundle_artifacts／s1_performance／s1_open_position_valuation
│   │                  → s1_publication_gate
│   │                  → s1_production_runner（run／resume／verify）
│   ├── 引擎：engine、replay、indexed_replay、layered、merged、raw_tape、targets、partial、
│   │         hedge、hedge_study、execution_facts、execution_runner、pilot、study、
│   │         exit_maker、exit_maker_study（期貨 route 與 exit maker 重作時借用）
│   └── transaction_costs：費稅 profile
└── tests/         # unit／integration／canonical verifier tests
```

執行環境：

```bash
cd /home/kevin/Project/HFT/src/research/futures_spot_spread
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-project --with polars \
  python -m unittest discover -s maker/src/tests -t .
```

因果線重跑順序（輸入都在 `maker/data/walkforward/`）：

```bash
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_width.daily_facts
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_width.rolling --daily-root maker/data/walkforward/daily
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.liquidity --daily-root maker/data/walkforward/daily \
  --rolling-boundaries maker/data/walkforward/rolling_boundaries/rolling_boundary_snapshots.parquet
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.monthly_product_selector
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.one_second_message_load_runner
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.one_second_makerfill_runner
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.dynamic_future_hedge
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.dynamic_expiry_close
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.dynamic_estimated_path_portfolio
uv run --project /home/kevin/Project/HFT --no-sync python -m maker.src.quote_fill.foundation_selection_runner all --execute
```

Cost-aware S1只能由clean nested source commit發布。先用獨立output root跑smoke，再跑canonical root；完整後另做
input-content verification；成功後會在bundle內原子建立hash-bound `verification.json`：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.s1_production_runner run \
  --output-root /tmp/s1-smoke --report-path /tmp/s1-smoke-report.md \
  --max-new-partitions 1

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.s1_production_runner run

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.s1_production_runner verify --verify-inputs
```

各 runner 的 `--help` 與 `--verify-only` 是正式介面；輸出根目錄由各模組常數指定，重跑前先看
`maker/doc/quote_fill/README.md` 對應文件。

S0.5 selection v1（anchor／Q2／mother；既有moving-anchor convergence只作sensitivity）是source commit
`ace2669dbc3c725a94baa035494b9bef10701304`與runner v1簽名的immutable canonical。Current HEAD的selection runner已升為v2，
不得用它resume／verify v1 work root；若要從零重建v1，須在該source commit與獨立output／work root執行。
Current HEAD由下列frozen v2 verifier pin並重驗v1 marker／boundary lineage。

Frozen-at-upper-touch convergence v2（已canonical發布；新publication仍需clean commit）：

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-project --with polars \
  python -u -m maker.src.quote_fill.foundation_frozen_convergence_runner \
  publish --execute

UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-project --with polars \
  python -u -m maker.src.quote_fill.foundation_frozen_convergence_runner \
  verify-only
```

舊因果線的已知限制（見 `doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`）仍保留作predecessor：makerFill
是mixed-clock approximate label，`execution_runner`仍是`sessions × symbols`介面。新的S1不再呼叫
`dynamic_estimated_path_portfolio`那條q95-hardcoded path，而由scenario spec與joint event loop直接產生七組結果；這不會把
Spot Bid entry fill truth升格為exact。

程式使用 Polars；禁止在 feature、取樣、label、normalization 或 validation 中使用未來資訊。
