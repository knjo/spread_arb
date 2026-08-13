# Maker Research Data

此目錄只放 maker 研究衍生產物，不複製 HFT 或 NAS raw data。

建議結構：

```text
data/
├── fair_mid/
├── quote_width/
├── quote_fill/
├── hedge_50ms/
└── backtest/
```

Parquet、CSV、模型與大量報告預設不進 Git。每個資料集需保存 schema／config hash、source dates、row counts、gate funnel 與 model version，讓結果可重跑及稽核。

目前 `fair_mid/` 的主要產物：

| 產物 | 用途 |
|---|---|
| `basis_landmarks_*.parquet` | 每 pair 每秒 causal 期現狀態、三種 basis 與 eligibility |
| `fair_anchor_panel.parquet` | Anchor candidates、future labels 與 residual 研究 panel |
| `contract_mapping.csv` | 每日 spot／近月標準股期與各自 RefPrice |
| `landmark_audit.csv` | Raw rows、clock offset、TrialMatch／RefPrice／freshness funnel |
| `metrics_by_model.csv` | Accuracy、tail error、TV 與 mean reversion |
| `metrics_by_sample.csv` | 100–5,000 ms freshness 與 leg-skew sensitivity |
| `residual_bins_by_sample.csv` | Residual 大小對未來回歸的機率表 |
| `endpoint_freshness.csv` | `t` 與 exact `t+300s` 同時 fresh 的 stale/as-of 診斷 |
| `endpoint_residual_bins.csv` | 雙端點 freshness 下的 residual 分桶 |
| `delayed_reversion.csv` | 不共用 `B_t` 的 `t+30s` 至 `t+300s` 方向診斷 |
| `delayed_reversion_bins.csv` | 延遲診斷依 residual 方向與大小分桶 |
| `quote_churn_by_model_route.csv` | 依合法 tick rounding 的 fair-only／完整掛價改動率 |
| `pilot_config.json` | 日期、symbols、interval、width 與 freshness 設定 |

延伸研究分開落地：

| 目錄 | 用途 |
|---|---|
| `fair_mid/prior_day/` | 昨日同一目標合約 landmarks、prior summary、seeded anchor 與 paired metrics |
| `fair_mid/level_stability/` | Absolute level、block-relative level、fast／slow gap 與方向診斷 |
| `quote_width/` | D−1 商品參數、non-overlap excursion、width candidates、隔日驗證與 entry quote geometry |
| `quote_width/cycle/` | 固定格點的獨立 latent position FSM 與 diagnostic summaries |
| `quote_width/adaptive/` | D−1 safe parameter snapshot，以及分開標示的 retrospective reach／reversion validation |
| `quote_fill/` | SpreadPair epoch、candidate intents、physical orders、state spells、partial fills與撤單／queue replay diagnostics |
