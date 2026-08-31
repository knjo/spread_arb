# 日更機率表與 Walk-forward 驗證

## 目的

把目前八日 retrospective pilot 改成接近實盤的順序：交易日 `D` 開盤前，只能使用 `D` 以前已成熟的行情與交易結果，建立當日商品／狀態／action 機率表。每日可依完全凍結的更新規則吸收新資料，但不得每天重新挑模型或交易邏輯。

第一版使用最近 60 個共同交易日，約等於三個月：

```text
latent boundary snapshot(D)
    = empirical excursions in last 60 sessions strictly before D

execution probability snapshot(D)
    = mature quote/fill/hedge/exit facts in last 60 sessions strictly before D
```

60 是第一個預先指定的 production-like baseline；40、90、expanding／decay 只能作開發期 challenger，不能看 final 後再選。

## 每日更新與固定參數的分界

每天允許更新的是：

- 商品正、負 residual excursion 分布與上下界。
- Maker fill、撤單、50ms locked edge 與出場 branch 的 sufficient statistics。
- 依既定公式產生的 posterior probability、support 與 fallback。

進入 holdout 後不可改的是：

- lookback／decay、state 欄位與 bin edges。
- product／parent shrinkage 強度與 support gate。
- 掛價、保留舊層、退後撤單與 latency 假設。
- partial／top-up、exit escalation、overnight 與 cost／tax 邏輯。
- EV floor、商品選擇方式與 portfolio 限制。

因此 holdout 中使用昨日已成熟 outcome 更新今日 table 是合法的 prequential OOS；看了昨日績效後人工改參數則不是。

## 2026 日期契約

目前 spot tick、tickFeature 與非 placeholder stock-futures raw 在 2026-01-26 至 2026-08-13 共有 131 個共同交易日。前 60 日只作 history，因此第一個完整 60-session pre-open table 是 2026-05-05。

| 階段 | 日期 | 用途 |
|---|---|---|
| history／burn-in | 2026-01-26～2026-04-30，共 59 日 | 建 daily facts；尚未滿 60 日 |
| bridge history | 2026-05-04，共 1 日 | 第 60 個 history session；只建 fact，不作完整60日table評分 |
| fine-tune | 2026-05-05～2026-05-29，共 19 日 | 每日 walk-forward 選預先列出的 estimator／掛單／EV 候選 |
| confirmation | 2026-06-01～2026-06-30，共 21 日 | 確認 May 選出的設定並凍結 champion |
| pseudo-holdout | 2026-07-01～2026-08-13，共 31 日 | preliminary outer validation；不是 pristine final |
| locked forward | 2026-08-14 後第一個完整共同資料日起累積 60 日 | 真正未看過的 final test |

07/20 與 08/11 已在八日 pilot 中被檢視並影響研究設計，所以 7–8 月只能誠實稱 pseudo-holdout。真正 final 要從 08/14 後的新資料開始，程式／設定 hash 鎖定後才逐日產生 prediction/action snapshot，累積滿 60 個 eligible sessions 再解封總結果。

Cost-aware S1 的development entry window固定停在2026-08-13，不能為了讓carry自然結束而讀8/14後資料。S1 policy
ranking先重播完整71日facts排除已terminal／expiry的position，再對final paired open固定使用8/13 13:20 market books：
同scenario／商品先聚合數量後，以當下最後合法足量L1–L5做hypothetical liquidation mark（long Spot用executable Bid、
short Future用executable Ask），扣已發生逐腿成本與lot-aware remaining exit cost。它不生成fill、不算
completion、不釋放capacity，且必須和terminal realized net分列；任何open position無共同合法價格時，withhold economic
ranking與S2 shortlist。8/14起資料只准在策略freeze後作真正forward，不得回填development mark。

## Label maturity

每列 fact 保存 `label_end_date`。`D` table 只接受：

```text
fact.Date in trailing window
and label_end_date < D
```

- Fill／cancel／50ms hedge／同日 exit：收盤後成熟，最早進入下一交易日 table。
- Next-session overnight outcome：下一交易日收盤後成熟，最早再下一交易日使用。
- 多日 carry／roll：實際 terminal session 收盤後才成熟。
- 未完成或資料中斷保留 censor／unknown，不得偷改成 loss 或 no-hit。

## 第一版表

### Rolling boundary snapshot

`maker/src/quote_width/rolling.py` 直接使用 excursion-level observations，不把每日 p50／p80 再取平均：

```text
ValueCode × target Date × q50/q80/q95
positive upper distance
negative lower distance
history sessions/dates/excursions/censors
train start/end and source_asof_date
```

正負側分開；跨換月先依 `ValueCode` pool，當日仍使用 point-in-time exact `QuoteCode`。DTE／roll conditioned challenger需另列，不能在 target key 偷換合約。這仍是 latent prior，`actionable_execution=false`、`ev_ready=false`。

### Rolling state probability

`maker/src/quote_fill/walkforward.py` 將 deterministic aliases、hedge與 post-fill facts接起，再逐日估：

- `P(full fill | state, action)`。
- `P(50ms locked edge仍達原 threshold | full fill)`。
- `P(frozen adaptive lower latent hit | full fill)`。
- 50ms adverse slippage p50／p95。

第一版 state family 分開估 `all`、maker rank、queue、time-of-day、freshness，不做稀疏的全 Cartesian cross。Product cell以固定 Beta strength向 route/state parent收縮，support不足明確 fallback；未來 ridge／tree只能作 walk-forward challenger。

`frozen lower hit` 目前仍是 basis-mid first-passage，不是 exit maker fill。真正 EV 必須等 exit raw replay、四腿 fills／qty、partial、費稅與 overnight value完成。

### Raw replay 前的流動性 screen

全商品先用 1 秒 causal panel 建 `product × day` 流動性 facts，再以同樣的 `<D` 60-session window 產生 route-specific screen：

- Spot／future A1-B1 的 spread ticks p50／p95。
- 兩腿同時合法與 age <= 1 秒的時間比例。
- Future maker 路徑使用 spot ask 深度；Spot maker 路徑使用 future bid 深度。
- 兩腿 taker-taker band，以及 adaptive upper+lower 相對該摩擦的空間。
- 每小時有 book state 更新的秒數；這是秒級 activity proxy，不是 raw message rate。

判斷必須分 route。Maker 腿 spread 寬可能增加被動報價空間，不能單獨當剔除理由。第一版 hard gate 排除 support不足，或長期合法 book、fresh-book depth、maker activity明確失敗的 route；taker spread寬度與上下界／TTBand ratio只作 replay strata，不先假設等於策略實付摩擦。`pass` 進 core／wide-maker replay，`insufficient_support` 留少量 exploration，`known_fail` 才先排除。輸出 `pre_replay_candidate` 只用來縮小昂貴 raw replay universe，`production_universe_approved=false`；閾值仍要在開發期 walk-forward 校準，不能看 pseudo/final 結果後更改。

131日結果、spread bucket、tick-ladder修正後68檔pseudo core與第一輪43檔strict core詳見[quote_fill/LIQUIDITY_SCREEN.md](quote_fill/LIQUIDITY_SCREEN.md)。歷史eligibility median／q10只作風險warning／state features，不進hard gate；當下`eligible=false`才停止掛單，避免研究discovery因少數壓力日被完全選掉。

## Fine-tune 與最終評分

May 只能使用每日當下以前的資料比較候選。建議 primary tuning objective：

```text
realized net cashflow / occupied capital-time
```

base queue 為主，tie-break依序看 whole-date block-bootstrap lower bound、emergency／overnight tail、churn。Final沿用同一 objective；co-primary為 conservative queue 的總 net PnL。Fill rate、hit rate與 Sharpe只作診斷，不可在 final 後改選目標。

Holdout必須以每日當時 table選出的 sequential actions跑完整 inventory ledger；不可在事後為每列挑 realized-best action。Predicted EV與 realized OOS PnL分開保存。

## 分區產物與重跑

全市場 latent facts以每日分區、可續跑方式產生：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.daily_facts
```

每個 `Date=YYYYMMDD/` 保存：

- `causal_fair.parquet`
- `excursions.parquet`
- `mapping.parquet`
- `audit.parquet`
- `complete.json`

Reader只接受帶 schema／row-count／size／schema fingerprint manifest 的完成分區。早期長跑若產生 legacy marker，必須在完整性稽核後顯式遷移；reader不會默默升格：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.daily_facts \
  --migrate-legacy-markers
```

遷移後 marker 會保留 `migrated_legacy_marker=true`、`legacy_atomic_publish_verified=false` 與 `legacy_writer_provenance_verified=false`；它可供目前 development／pseudo-holdout 研究，但不可冒充 pristine final。新建分區才使用 marker-last atomic publish，並綁 builder code、build config 與 raw source identity hash。

再由分區 facts 建 60-session boundary：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.rolling \
  --daily-root maker/data/walkforward/daily
```

接著產生 route-specific 流動性 screen：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.liquidity \
  --daily-root maker/data/walkforward/daily \
  --rolling-boundaries maker/data/walkforward/rolling_boundaries/rolling_boundary_snapshots.parquet
```

Execution facts 完成後建立日更 state table：

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_fill.walkforward \
  --sessions maker/data/walkforward/sessions.txt
```

八日資料可用較短 window做因果 smoke，但不能把它的機率或商品排名當成 60-session 結果。
