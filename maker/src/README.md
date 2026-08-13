# Maker Research Source

程式依 work package 分模組：

```text
src/
├── common/       # data contract、時間、價格與合約 mapping
├── fair_mid/     # WP01 anchor、labels、metrics、tick-rounded churn
├── quote_width/  # D−1 結構參數、diagnostic grids、adaptive bounds 與 latent cycles
├── quote_fill/   # WP02 episodes 與 queue replay
├── hedge_50ms/   # WP03 book walk 與 slippage
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

程式使用 Polars，資料與模型邏輯需模組化；禁止在 feature、取樣、label、normalization 或 validation 中使用未來資訊。
