# 13:00 動態 B1／A1 平倉 controller 分析包

## 目的與狀態

這個分析把 13:00 仍持有的部位，放進逐事件的 portfolio controller：期貨 B1
買回與現貨 A1 賣出都是持續 peg；較積極方向保留舊層並新增價層，後撤則取消超前層。
兩條 route 以最早 full fill 競賽，勝方在 `full_fill + 50 ms` 用 taker 對沖另一腿，
nominal 分支同時取消 sibling。它不是只在 13:00 掛一次的 static order。

本包是 analysis-only diagnostic。它沒有 cancel ACK、沒有跨 position 的 joint-volume
allocation，也沒有 D-safe 商品池，因此 `formal_ev_ready=false`、
`production_strategy_go=false`。Optimistic 分支容許同一 product-day template 被多個
position 重用，只能視為共同成交量重複計算的上界；conservative 分支每個
product-day 最多接受一個 aggressive close，是主結果。

## 2026-08-21 正式結果

正式 v3 root：

`maker/data/walkforward/aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix`

- `complete.json` SHA-256：
  `21bed1b6bbd9137e44fc3fcb43172a090498eb45ca279e07d3284191f6668330`
- marker payload SHA-256：
  `c50cd22d6c21bcec8359688fb14883f9b24a8e3179b24e7da0b556391a829553`
- 母體：3,672 paths、63 sessions；13:00 inventory 3,917 position-days／976
  product-days。
- Template coverage：961／976 product-days（seed 427＋cache replay 534）；缺15
  product-days／25 position-days，逐情境保留為未 aggressive close，沒有當成成交。
- Supplemental normal terminal 3,672／3,672 已定價；10M／20M 與同規則 normal
  control 的 path 與 daily metrics 都是零差異。

完整十情境如下；`Net` 是整個 accepted portfolio 的 after-cost realized net，不是
aggressive exit 單獨的 P&L。Entry turnover 是 accepted spot one-way notional，63 日
平均列在最後；金額單位 TWD。

| Cap | 容量假設 | Accept | Aggressive close／notional | Gross | Cost | Net／bp | Loss positions／negative days | Realized MDD | 13:20 target days／max excess | Entry turnover／日均 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | conservative | 1,209 | 0／0 | 1,749,300 | 1,035,536 | 713,764／18.44 | 244／5 | 47,709 | 63／0 | 387.053M／6.144M |
| 10M | optimistic reuse | 1,209 | 0／0 | 1,749,300 | 1,035,536 | 713,764／18.44 | 244／5 | 47,709 | 63／0 | 387.053M／6.144M |
| 20M | conservative | 1,779 | 0／0 | 3,142,800 | 1,731,677 | 1,411,123／21.03 | 343／6 | 53,880 | 63／0 | 670.853M／10.648M |
| 20M | optimistic reuse | 1,779 | 0／0 | 3,142,800 | 1,731,677 | 1,411,123／21.03 | 343／6 | 53,880 | 63／0 | 670.853M／10.648M |
| 30M | conservative | 2,043 | 81／29.890M | 3,584,450 | 2,082,273 | 1,502,177／18.30 | 448／14 | 81,990 | 63／0 | 820.740M／13.028M |
| 30M | optimistic reuse | 2,000 | 36／12.310M | 3,677,900 | 2,046,506 | 1,631,394／20.20 | 403／9 | 13,758 | 63／0 | 807.786M／12.822M |
| 40M | conservative | 2,156 | 98／35.638M | 3,923,450 | 2,275,486 | 1,647,964／18.26 | 469／12 | 53,006 | 61／1.577M | 902.375M／14.323M |
| 40M | optimistic reuse | 2,128 | 88／31.126M | 3,947,350 | 2,229,670 | 1,717,680／19.36 | 456／6 | 20,210 | 63／0 | 887.018M／14.080M |
| 50M | conservative | 2,180 | 79／29.637M | 4,106,450 | 2,317,632 | 1,788,818／19.35 | 446／10 | 15,724 | 61／2.389M | 924.403M／14.673M |
| 50M | optimistic reuse | 2,169 | 102／37.460M | 4,045,050 | 2,294,678 | 1,750,372／19.07 | 467／8 | 21,380 | 63／0 | 917.908M／14.570M |

10M／20M 的 target 等於 hard cap，因此 admission 已使 13:20 exposure 不超 target，
aggressive controller 是 identity control。主 conservative 結果在 30M／40M／50M
把平均 13:20 notional 從 normal 的 13.023M／14.016M／14.409M 降到
11.765M／11.707M／11.783M；最大值則從 25.755M／30.128M／32.843M 降到
19.934M／21.577M／22.389M。40M／50M 各有兩天因 observed template capacity
不足，仍高於20M target。

但 aggressive close 本身不是獲利來源。Conservative 30M／40M／50M 的 aggressive
子集分別是：

| Cap | Close | Aggressive-only gross | Cost | Net／bp | Wins／losses |
|---:|---:|---:|---:|---:|---:|
| 30M | 81 | -129,800 | 91,530 | -221,330／-74.05 | 2／79 |
| 40M | 98 | -115,600 | 100,131 | -215,731／-60.53 | 4／94 |
| 50M | 79 | -95,100 | 79,311 | -174,411／-58.85 | 5／74 |

相對不加 aggressive exit 的 normal control，主結果總 net 在30M／40M／50M 分別
少171,723／172,583／121,085 TWD，雖然組合總 net 仍為正。結論因此不是「更快平倉
也更賺」，而是用明確的負收益換取較低 13:20 carry exposure。Optimistic reuse 在
63／63 日達標，但共同市場容量被重用，只能當容量上界，不能用其 P&L 作正式比較。

正式 service `aggressive-1300-analysis-v3-formal-20260821` 在8G／12G／swap0限制下
exit0，wall 5分16秒、CPU 7分35.7秒、peak memory 2.5G、swap peak 0。Publisher 在
atomic rename 前完成 full source rebuild；發布後 output-only verifier與獨立
row-level replay（36,720 position-scenario、630 day-scenario）均 PASS。

先前 v2 因禁單的 post-13:00 entry 錯誤推進 normal-exit clock而判INVALID，已
recoverable 隔離於
`aggressive_1300_exit_analysis_60d_20260821_v2_invalid_post13_release_20260821`；不可引用。

## 正式資料與固定身分

- 正常 terminal：
  `prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close`
- v3 `complete.json` SHA-256：
  `402d050e6fed75c38cea540fd31dd09994f6a24ef91a5411ea4e715e14cb691d`
- v3 `supplemental_paths.parquet` SHA-256：
  `63742b773f32cdb4851fe686aea8c1dd8349d27b42be7460045dd73ec23cfda6`
- v3 rows：3,672；其中 171 筆 expiry path 使用同日 13:30 兩腿 MarketInfo
  daily close。這是 official close mark，不是可成交 BBO，也不是 settlement。
- 10M／20M identity control：
  `normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only`
- 13:00 replay seed：workspace 內 immutable 427 product-day parquet；其 SHA-256
  `68505c2ff20cbc8cc90e7f871006121372b88bade24d947341195e642a28c9da`。
- 其餘 required product-days 由 v8 candidate-session cache 重播；每個 complete
  marker 及七個 cache artifact 都重算 SHA-256。Cache 的更上游 raw 身分仍只有
  producer 所留的 stat／mtime fingerprint，所以 marker 明示
  `upstream_raw_content_integrity_bound=false`。

Loader 亦驗證 v3 對 v2 的 3,501 筆非 expiry row 完全不變、171 筆 expiry row
精確連到 daily-close facts、entry action 價格與所有 source inventories。63 個
session 的 calendar digest固定；即使某天沒有新 entry，carry 的 normal／aggressive
release 仍會重播。

## Controller 契約

正式情境為 `10/20/30/40/50M × conservative/optimistic` 共十組：
scenario ID 固定帶 `aggressive_1300_diagnostic_v2_post13_timeline_fix`，避免與已
隔離的錯誤 v1 controller 結果混接。

- hard portfolio cap 是 10／20／30／40／50M；單商品 prospective notional 不得超過
  cap 的 30%；
- 13:00 後禁止新進場；
- 若某商品在當日開始時已有前一日 carry，該商品整日 exit-only；即使當天稍後已出場
  也不重新開倉；
- admission 時同 timestamp 採 entry-before-normal-exit，與 normal control 相同；
- 13:00 後 normal 與 aggressive 以實際 decision timestamp 競賽；完全同 timestamp
  時 normal 先釋放；
- 跨商品達標後的 portfolio cancel clock 是勝方 `full fill + 50 ms` 的 decision time；
  這個 nominal controller 沒有模擬同一 50 ms 內另一商品已成交的 cancel race，因此
  target overshoot 仍是未建模風險；
- aggressive 提早出場後會立刻釋放 cap，後續 candidate 重新依事件順序 admission，
  不是固定 normal accepted set 的 overlay；
- 13:20 是主 exposure snapshot，target 為 `min(cap, 20M)`；這不是 TWSE 13:30
  收盤。另發布 13:30 expiry official-close 後的 inventory accounting，但沒有重播
  13:20–13:30 其他行情，不能解讀為完整 13:30 exposure；
- normal 未被 aggressive 搶先者沿用 v3 terminal cashflow。所有 3,672 path 已定價，
  不允許 unpriced position 靜默釋放額度。

成本逐 position 依正式 profile 重算：現貨每邊 commission 1.71 bp、同日賣出稅
15 bp／隔夜 30 bp；期貨每邊 tax 0.2 bp，來回 commission TWD 40。每日與 summary
發布 gross、cost、net、net bp、win/loss、negative day、realized-only MDD；沒有 carry
MTM。交易量分開報 entry spot one-way notional、entry paired two-leg turnover、實際
exit spot one-way turnover與 exit paired two-leg turnover，不把 alternatives 相加。

## 產物與 fail-closed verifier

正式 root：

`maker/data/walkforward/aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix`

Atomic bundle 固定 12 個 parquet 加 `complete.json`：template、daily inventory、
position overlay、missing product-days、cache inventory、entry inventory、branch summary、
coverage、controller contract、position outcomes、daily control、scenario summary。
Marker 綁 ordered schema、rows、bytes、SHA-256、config、implementation sources與安全旗標；
verify 會重算 source inventories，重新建出全部 12 表並逐 schema／value 比對。Root 已存在時
publisher fail closed，不會覆蓋。

正式執行只能在獨立 transient user service，且不可提高上限：

```bash
systemd-run --user \
  --unit=aggressive-1300-analysis-v3-formal-20260821 \
  --property=WorkingDirectory=/home/kevin/Project/HFT/src/research/futures_spot_spread \
  --property=MemoryHigh=8G \
  --property=MemoryMax=12G \
  --property=MemorySwapMax=0 \
  --property=Restart=no \
  --setenv=PYTHONPATH=. \
  --setenv=UV_CACHE_DIR=/tmp/uv-cache-aggressive-1300-v3 \
  --setenv=MPLCONFIGDIR=/tmp/mpl-aggressive-1300-v3 \
  /home/kevin/.local/bin/uv run --no-project --with polars --with pyarrow \
  python -m maker.src.quote_fill.aggressive_1300_analysis_cli
```

若 `memory.events` 的 `max/oom/oom_kill` 增加或 process 非零退出，該次 run 是 NO-GO；
不可藉由提高 cap 硬跑。CLI `--verify-only` 預設做完整 source rebuild；只有明確指定
`--no-source-rehash` 才只驗 bundle self-integrity。

## 解讀限制

45 商品是事後固定 cohort，不能當 D 日可部署 universe。Cache template 未分配不同
position 共同看到的市場成交量，nominal cancel 也沒有 ACK。Missing product-day／position
會逐 scenario 保留為未 aggressive close，而不是視作成交。即使 after-cost P&L 為正，
本包仍只能回答在上述近似下的診斷結果，不能直接變成盤中掛單 EV 或 production GO。
