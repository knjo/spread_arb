# 固定格點回歸診斷：1 秒 Latent Cycle Pilot

更新日：2026-08-13

## 結論

目前資料支持「basis 高於 fair 後會回到中心」，而從上界完整走到對稱下界的速度會隨格點距離變慢。

- 固定 `10／15／20／30 bp` 全部只是 reversion-shape、coverage 與 sensitivity diagnostics，不是 production width，也不是 WP02 shortlist。
- 固定 `1／2 tick` 同樣只作 action geometry control；商品的一檔 BP 會隨價格級距改變。
- 正式界線改由 [ADAPTIVE_BOUNDS.md](ADAPTIVE_BOUNDS.md) 的 D−1 商品別、正負側 empirical quantiles 產生，再於盤中轉成合法 rounded action。
- 多個 diagnostic aliases 落在同一個 maker price 時，必須共用一筆 raw queue／fill fact。

這裡確認的是行情路徑的回歸現象，不是 maker fill、實際週轉、50 ms hedge、PnL 或 EV。

## 定義

```text
B_t = 1 秒 causal mid-basis
M_t = causal EWMA120 fair-mid
r_t = B_t - M_t
U_t = M_t + Wopen
L_t = M_t - Wclose
```

當連續合法秒由 `r < Wopen` 首次穿越到 `r >= Wopen`，建立一個 latent entry。持有期間不再接受新的上界 crossing；到達下界、資料／風控 gate 失效或收盤才結束。

出場分三層，各自跑獨立且不重疊的 position FSM：

| `Wclose / Wopen` | 下界 | 解讀 |
|---:|---|---|
| 0 | `M` | 回到中心 |
| 0.5 | `M - 0.5W` | 往對稱下界走一半；總 nominal band 為 `1.5W` |
| 1 | `M - W` | 完整 symmetric cycle；總 nominal band 為 `2W` |

另分兩種 anchor：

- `dynamic`：出場下界跟隨當下 `M_s`，接近未來實盤動態改價。
- `frozen_entry`：出場用 entry 時的 `M_t`，用來確認 basis 本身確實移動，而非只因 fair 追上行情。

主診斷表從 entry 30 秒後才允許出場，降低同秒 noise／短 bounce 的影響；1 秒 first-touch sensitivity 也完整保存。30 秒不是 execution latency，也不是正式持有規則。

這個取樣會刻意忽略提早碰下界後又彈回的短路徑；例如 dynamic symmetric W=10 有 20.6% 的 cycles 在前 30 秒已碰過下界。這是抗 microstructure bounce 的 sensitivity，不代表策略實際會等 30 秒才出場。

## 先確認：是否真的回到中心

以下使用較嚴格的 `frozen_entry` 中心：entry 後 basis 必須回到 entry 時已知的 fair。表內機率同時報 pair-day median／event-weighted；`N` 是能判定該 horizon 的事件數。

| Wopen | Entries | 300s N | 300s 回中心 | 600s N | 600s 回中心 | Completed holding p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 bp | 645 | 640 | 77.8% / 83.0% | 640 | 86.7% / 89.7% | 50 s |
| 15 bp | 396 | 391 | 77.8% / 78.8% | 391 | 86.6% / 87.0% | 78 s |
| 20 bp | 240 | 237 | 73.9% / 74.7% | 237 | 89.4% / 84.8% | 108 s |

因此「碰上界後回到中心」不是只靠 moving EWMA 造成；至少在這個八日 pilot，固定 entry fair 的對照仍有一致回歸。這支持繼續建立 execution replay infrastructure，但沒有核准任何固定 width，也不是 alpha 或獲利證明。

## 完整上界到下界

下表是 operational 對照：`dynamic M`、`+W → -W`、30 秒後才開始判定。不同 policy 各自維持部位，因此 entry 數不能跨列相加。

| W | 有 entry 的 pair-days | Latent entries；每日 p50 [q25,q75] | Censor | 300s N；完成率 pair/event | 600s N；完成率 pair/event | Holding p50 | Basis capture p50 | Anchor drift p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 bp | 31 / 31 | 592；16 [11,27] | 13 | 585；55.6% / 67.5% | 582；85.7% / 85.4% | 248 s | 23.43 bp | 5.55 bp |
| 15 bp | 30 / 31 | 303；10 [5,15] | 17 | 296；41.4% / 48.0% | 293；72.1% / 71.0% | 379 s | 32.08 bp | 9.96 bp |
| 20 bp | 28 / 31 | 146；5 [2,7] | 13 | 141；16.7% / 25.5% | 140；38.8% / 49.3% | 776 s | 39.42 bp | 11.28 bp |
| 30 bp | 18 / 31 | 50；1 [0,3] | 12 | 47；0.0% / 25.5% | 47；0.0% / 38.3% | 848 s | 51.75 bp | 17.44 bp |

解讀：

- 10 bp symmetric cycle 在日內有最多 support，600 秒內的 pair-balanced 完成率約 86%。
- 15 bp 還有研究價值，但持有時間拉長、600 秒完成率約 72%。
- 在這組 diagnostic grid 中，20／30 bp 的完整 symmetric path 較慢且 support 較少；這只描述路徑，不用來選 production exit。
- Dynamic symmetric completed events 中，basis capture 非正的比例依 W=10／15／20 為 2.1%／1.4%／0.8%。這些是 fair 移動造成的 apparent exit，已另存 `anchor_only_exit`，不可算成獲利週期。
- Frozen symmetric 更嚴格：W=10 在 300／600 秒的 event-weighted 完成率為 57.7%／64.6%；W=15 降為 24.4%／36.7%。所以 dynamic 與 frozen 必須並列，不能只拿 moving-fair 的高 hit rate 下結論。

## 狀態機與 censor

- Session／gap 後第一列已在上界外屬 left-censored；必須先看見合法 `r < W`，再發生新 crossing 才能 entry。
- TrialMatch、strict RefPrice band、book／formal gate、非連續 timestamp 或 eligibility 失效時，OPEN cycle 立即 right-censor，不跨 gap 找下界。
- Censor 不代表真實部位已平。因此一旦 OPEN 被 censor，該 pair-day×policy 進入 absorbing stopped 狀態，不會在未知庫存下重新 entry。
- Horizon 是三態 label：H 前完成為 true；連續觀察滿 H 仍未完成為 false；H 前 censor 為 null。
- Session cutoff 的未解部位只記 censor；要到 tick replay 加上 force-flat taker 成本後，才能轉成策略損益。

連續要求兩腿每秒 `age <= 1s` 會使 fresh sample 非常碎裂：dynamic symmetric、30 秒口徑的 W=10 有 38 entries，其中 30 筆被 freshness gate censor。這個 sensitivity 不能拿來選 width；WP02 需在 event-time 重建 quote validity／staleness policy，而不是要求每個牆鐘秒都有新報價。

## 下一步：商品界線轉成 raw tick action

可以繼續建立 WP02 infrastructure，但執行層會重新定義 action：

1. D 日只讀以 `<D` 資料產生的 versioned product boundary snapshot，再搭配當下 causal state。
2. 以當下 `M`、反腿 taker quote、route 與合法 tick ladder 枚舉 maker prices，反算每個 action 的 effective upper／lower；不先指定固定 BP 或固定 ticks。
3. Raw episode identity 使用 `episode_start + Date + ValueCode + QuoteCode + route + stage + rounded_target_price + qty + replay_version`；fair／boundary versions 另存在 many-to-one policy alias map。完整契約見 [../02_QUOTE_FILL.md](../02_QUOTE_FILL.md)。
4. 每個 episode 估 `fill before requote/gate`，touch 不當 fill；OPEN gate 失效需進 cancel-race／force-flat 處理。
5. Maker fill 發生後，以 `fill RecvTime + 50 ms` 重播另一腿 taker VWAP。
6. 只有完成 entry fill 與 hedge 的部位，才估 target exit、同日 force-flat、隔夜與 emergency branches，最後計算費稅後 EV。

## 研究範圍與限制

- 日期：2026-01-28、02-23、03-18、04-20、06-09、06-17、07-20、08-11。
- 商品：2303、2317、2603、2881，共 31 個可用 date×symbol pairs。
- 1 秒 row 是 time-weighted state occupancy；不同秒與不同 policy 並非獨立樣本。
- EWMA120 與 D−1 width prior 仍是 provisional；尚未做完整 2026 expanding walk-forward 與最終 July／August frozen holdout。
- Completed holding／capture 是 conditional-on-completion；censored 部位未計 force-flat，不能當成無偏 PnL 分布。
- 這些 boundary 是 residual reference bands，不是常態分布信賴區間。
- 1 秒 crossing 仍有 overshoot：dynamic symmetric 的 entry overshoot p50 在 W=10／15／20／30 約為 4.27／4.45／5.01／5.83 bp；所以上界只是觸發區域，不是精確成交 basis。

## 產物與重跑

- `../../data/quote_width/cycle/latent_full_cycles.parquet`
- `../../data/quote_width/cycle/cycle_by_day_symbol.csv`
- `../../data/quote_width/cycle/cycle_policy_summary.csv`
- `../../data/quote_width/cycle/cycle_diagnostic_frontier.csv`
- `../../data/quote_width/cycle/dynamic_symmetric_diagnostic_summary.csv`
- `../../data/quote_width/cycle/config.json`

```bash
uv run --project /home/kevin/Project/HFT --no-sync \
  python -m maker.src.quote_width.cycle
```
