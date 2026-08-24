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
│   ├── 引擎：engine、replay、indexed_replay、layered、merged、raw_tape、targets、partial、
│   │         hedge、hedge_study、execution_facts、execution_runner、pilot、study、
│   │         exit_maker、exit_maker_study（期貨 route 與 exit maker 重作時借用）
│   └── transaction_costs：費稅 profile
└── tests/         # 30 個測試檔
```

執行環境：

```bash
cd /home/kevin/Project/HFT/src/research/futures_spot_spread
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with polars --with pyarrow --with pytest \
  python -m pytest -q maker/src/tests
```

因果線重跑順序（輸入都在 `maker/data/walkforward/`）：

```bash
python -m maker.src.quote_width.daily_facts
python -m maker.src.quote_width.rolling --daily-root maker/data/walkforward/daily
python -m maker.src.quote_fill.liquidity --daily-root maker/data/walkforward/daily \
  --rolling-boundaries maker/data/walkforward/rolling_boundaries/rolling_boundary_snapshots.parquet
python -m maker.src.quote_fill.monthly_product_selector
python -m maker.src.quote_fill.one_second_message_load_runner
python -m maker.src.quote_fill.one_second_makerfill_runner
python -m maker.src.quote_fill.dynamic_future_hedge
python -m maker.src.quote_fill.dynamic_expiry_close
python -m maker.src.quote_fill.dynamic_estimated_path_portfolio
```

各 runner 的 `--help` 與 `--verify-only` 是正式介面；輸出根目錄由各模組常數指定，重跑前先看
`maker/doc/quote_fill/README.md` 對應文件。

已知限制（見 `doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`）：makerFill 是 mixed-clock
approximate label；`execution_runner` 仍是 `sessions × symbols` 介面，接 manifest 前必須先拆掉笛卡兒積；
`dynamic_estimated_path_portfolio` 寫死 q95。

程式使用 Polars；禁止在 feature、取樣、label、normalization 或 validation 中使用未來資訊。
