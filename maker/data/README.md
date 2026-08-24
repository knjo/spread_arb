# Maker Research Data

此目錄只放 maker 研究衍生產物，不複製 HFT 或 NAS raw data。

建議結構：

```text
data/
├── fair_mid/
├── quote_width/
├── quote_fill/
├── walkforward/
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

`quote_fill/` 目前的八日 pilot 產物：

| 產物 | 用途 |
|---|---|
| `target_observations.parquet` | Raw 期現 state-change 上的 adaptive target／gate／queue observation |
| `order_aliases.parquet` | q50／q80 policy-specific working window、fill／cancel與shadow labels |
| `raw_order_facts.parquet` | 同 submit／route／絕對價去重後的 raw replay identity |
| `fill_by_day_symbol.csv`, `fill_summary.csv` | Natural-base-rate fill／partial／cancel tables |
| `fill_by_rank.csv`, `fill_terminal_summary.csv` | Submit rank與 competing terminal diagnostics |
| `makerfill_sanity*` | 既有現貨 makerFill 與 moving-policy raw replay對照 |
| `spot_partial_completion.csv` | 第一 lot 後於50ms–5s累積兩 lots 的可判定樣本率 |
| `hedge_facts.parquet`, `hedge_summary.csv` | Unique raw full-fill後 50 ms 反腿 L1–L5 cost |
| `latent_exit_opportunity_*` | Actual full-fill 後的 1秒／30秒 latent first-passage；非 execution |
| `action_research_summary.csv` | Fill、hedge、latent exit 串接骨架；`ev_ready=false` |
| `product_action_research_table.csv` | 商品×route×q 的 rounded band、fill、hedge與latent lower面板；`ev_ready=false` |

`walkforward/` 使用可續跑的日分區：`daily/Date=YYYYMMDD/` 保存 causal fair、excursions、point-in-time mapping 與 audit；`rolling_boundaries/` 保存只使用 `<D` 最近 60 個交易日的上下界；`liquidity/` 保存日級 A1-B1 spread／depth／freshness facts 與 route-specific raw-replay screen。Execution probability facts完成後，`quote_fill/walkforward/` 再保存 product／state 與 route-state parent 的每日 snapshot。所有 snapshot 必須帶 `train_start/end`、`label_cutoff`、版本及 `contains_target_day_outcome=false`。

`walkforward/liquidity/` 的主要產物：

| 產物 | 用途 |
|---|---|
| `daily_liquidity.parquet` | 131 日商品級 spread／freshness／depth／activity收盤後 facts |
| `rolling_liquidity_screen.parquet` | 每日只用 `<D` 歷史建立的 q50／q80／q95 route screen |
| `latest_q50_route_screen.csv` | 最新日 funnel 與 replay tier |
| `pseudo_validation_route_stability.csv` | Jul-Aug route stability；retrospective only |
| `pseudo_validation_stable_core_products.csv` | v6日期／市場別tick ladder下，兩 route 皆達 80% core 的68檔研究池；非 production |
| `complete.json` | 八個 publication artifacts 的 hash／bytes與source lineage |

## `walkforward/` 現況（2026-08-24 清理後）

固定 45 檔時代的 execution／exit maker／cross-session／post-cross／prequential／aggressive／compact 產物（約 42 GB 含 invalid 快照）已全部刪除；
資料不可復原，需依 `doc/REWORK_PLAN_20260824.md` 在因果 manifest 上重跑。目前只保留：

| 目錄 | 角色 |
|---|---|
| `daily/`、`rolling_boundaries/`、`liquidity/`、`sessions.txt`、`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1/` | 基礎事實，所有 run 的輸入 |
| `monthly_product_selector_causal_v2_20260822/` | 因果商品池 manifest（72 日、3,886 product-days） |
| `order_message_load_causal_v2_20260822_v2/`、`one_second_makerfill_causal_v2_20260822_v1/`、`dynamic_future_hedge_causal_v1_20260822/`、`dynamic_expiry_paired_close_facts_20260822_v1/`、`dynamic_estimated_path_portfolio_causal_v1_20260822/`、`august_exit_extension_causal_v1_20260822/` | 8/22 因果線 q95 輸出；互為 hard-coded 輸入 |
| `makerfill_rank_l1_l5_sample_20260820_v5/`、`future_ask_rank_l1_l5_indexed_sample_20260821_v1/` | 「只掛 A/B1–2」決策的五日證據 |
| `august_attribution_s0_20260824_v2/` | S0 canonical q95 quote-only歸因；`cap=∞`、非20M策略回放 |
| `august_attribution_s0_30_session_challenger_20260824_v1/` | S0 30-session market-only sensitivity；無scheduler／queue／makerFill且不進shortlist |

`august_attribution_s0_20260824_v1/` 有 explicit L1 clear scalar forward-fill 錯誤，已由 v2 supersede，不得引用。
