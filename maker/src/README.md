# Maker Research Source

程式依 work package 分模組：

```text
src/
├── common/       # data contract、時間、價格與合約 mapping
├── fair_mid/     # WP01 anchor、labels、metrics、tick-rounded churn
├── quote_width/  # D−1 結構參數、diagnostic grids、adaptive bounds 與 latent cycles
├── quote_fill/   # queue replay、hedge／exit facts、legal actions與D-safe EV lookup
├── hedge_50ms/   # 未來 exit hedge 與 portfolio-level depth allocation
└── backtest/     # WP04–05 policy 與 portfolio replay
```

所有執行與測試使用 HFT 專案環境，例如：

```bash
uv run --project /home/kevin/Project/HFT python -m maker.src.fair_mid.pilot --help
```

已完成 pilot 的重跑命令：

```bash
uv run --project /home/kevin/Project/HFT python -m maker.src.fair_mid.pilot \
  --dates 20260128 20260223 20260318 20260420 20260609 20260617 20260720 20260811 \
  --symbols 2303 2317 2603 2881 \
  --open-width-bp 20
```

昨日 prior 與 level stability 為獨立模組：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.fair_mid.prior_day

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.fair_mid.level_stability

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.table

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.cycle

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.adaptive
```

Raw fill／50 ms hedge 的 memory-bounded runner 以單一商品日執行，再聚合自然 base rate：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.study \
  --dates 20260128 --symbols 2317 --output-dir /tmp/wp02_20260128_2317

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.hedge_study \
  --date 20260128 --symbol 2317 \
  --quote-fill-dir maker/data/quote_fill \
  --output-dir /tmp/wp03_20260128_2317

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.post_fill

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.report
```

全市場日分區與 60-session walk-forward：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.daily_facts

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.rolling \
  --daily-root maker/data/walkforward/daily

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.liquidity \
  --daily-root maker/data/walkforward/daily \
  --rolling-boundaries maker/data/walkforward/rolling_boundaries/rolling_boundary_snapshots.parquet

# 只修改 rolling screen／報表時，可重用已核對的 daily facts：
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.liquidity \
  --daily-root maker/data/walkforward/daily \
  --rolling-boundaries maker/data/walkforward/rolling_boundaries/rolling_boundary_snapshots.parquet \
  --reuse-daily-liquidity

uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.walkforward \
  --sessions maker/data/walkforward/sessions.txt
```

新execution／EV介面位於：

- `quote_fill/execution_runner.py`：以`load_walkforward_product_day_sources`／`load_walkforward_execution_product_day`讀取D-safe輸入，單一product-day raw replay、atomic partitions與resume驗證。
- `quote_fill/execution_facts.py`：alias／raw order／50 ms hedge與保守同日taker-exit facts。
- `quote_fill/ev_surface.py`：合法被動tick列舉、canonical／decision actions、成熟terminal paths的60-session lookup與掛單排名。
- `quote_fill/taker_exit.py`：一秒taker/taker同日、expiry force與next-session benchmark；不是exit maker model。

介面與not-ready條件詳見[EV lookup契約](../doc/quote_fill/EV_LOOKUP.md)。目前八日pilot沒有完整券商成本設定或portfolio joint allocation，不得將`execution_daily_facts`直接解讀成net EV。

程式使用 Polars，資料與模型邏輯需模組化；禁止在 feature、取樣、label、normalization 或 validation 中使用未來資訊。
