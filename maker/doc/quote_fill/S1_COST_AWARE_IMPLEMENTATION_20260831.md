# S1 Cost-aware 重建：Implementation／Preflight（2026-08-31）

## 結論與狀態

舊 S1 partial 因 actual-send 前沒有經濟 eligibility，已停止且清除。新的 S1 不再只比掛單成交與同日完成；它先用
causal executable prices與使用者成本決定「這次是否值得送」，再把真正送出的候選放進共同 chronological 20M cap
replay。本文只記錄已凍結的實作契約，**不是71日研究結果，也不是 deployment GO**。

研究沒有暫停。commit `cd2b87c` 已完成scenario／成本／absolute exit／B6／capacity／path／artifact／publication、
8/13 common-horizon open valuation、exit pre-fill headroom guard與持久化verification receipt接線；該commit的完整S1回歸
365項、path contract 19項及2026-05-05 control smoke通過，`unresolved=0`。v3正式bundle只完成1/497；第二個
generic-clock partition因不必要wake放大而停止，沒有complete marker。現行v4 entry-decision-clock仍是候選語意；
precommit material differential與事件順序regression已通過，完整preflight、clean commit、post-commit durable rerun與新namespace smoke仍待完成。2026-05-05全218商品的
10／30分鐘generic／derived與full-day sparse-derived／true-full-1Hz-derived exact differential均已通過precommit integration；
clean commit後的schema v2 role-bound durable rerun仍待完成。497 partitions與獨立
input-content verify仍待完成；正式report完成前不得引用champion、Pareto或S2 shortlist。

## Entry decision clock（v4候選語意）

- Panel clock只以causal 1Hz的離散policy decision signature喚醒；只有不改變rounded entry／frozen-exit tick、reservation、support／TOD或gate status／reason的continuous anchor／margin漂移才不喚醒。
- 每次actual-send前仍用當下route-visible Spot／Future raw state重算target、economic gate與reservation；panel不是成交或book真值。
- Entry recovery使用route-visible semantic clock：Spot保留Trial／formal、canonical top-2 bid／ask BBO與source Bid1／Bid2 price／presence；Future保留sell-one／buy-one executable status／VWAP。Hedge、rollback與exit risk仍使用generic effective book clock。
- C9 denial保留product×policy-generation monitor，跨raw coalesce／pending withdrawal延續。只有沒有本商品explicit trigger，且candidate／admission inputs、policy generation與global／product committed balance均未變時，才重用incidental sibling wake前的cap-denial sleep；own raw wake、policy supersession、token與capacity retry一律重算。
- Entry與exit的pre-send candidate identity都使用商品在同`timestamp × phase`內的product-local logical ordinal，避免無關商品wake改名。Scheduler effect、actual-send、actual-cancel、execution與physical raw-order cursor仍維持全域真實因果順序。
- ContractExpiry先驗證並排入settlement；entry assignment使用pre-expiry capacity，basis-zero release到同timestamp `PHASE_SETTLEMENT`才生效，不能資助同cursor entry，釋放後最早`t+1 ns`重試。
- 這項壓縮不宣稱1Hz panel與raw reconstruction bit-exact。已通過的differential只支持各自被比較範圍內的material exact；不得把full-day B/C外推為未執行的full-day generic A/B等價，也不得續寫v3 bundle。

## Differential驗證層級與範圍

1. Event-order regression覆蓋sibling wake identity、policy supersession重綁C9 monitor、own raw／token／capacity retry重算，以及expiry release不能資助同cursor entry。
2. [完成] 2026-05-05全218商品、10分鐘entry horizon的sparse generic A與sparse derived B replay；exit／expiry生命週期仍完整執行。Sent 938、executions 130、positions 27、carry 3；八個material components全部exact。
3. [完成] 同日全218商品、30分鐘entry horizon A/B exact：sent 2,312、executions 162、positions 37、carry 6；八個material components全部exact。
4. [完成] 同日full-day sparse-derived B／true-full-1Hz-derived C exact：full policy rows 3,073,800、sparse rows 143,826；sent 4,493、executions 202、positions 50、carry 10；八個material components全部exact。
5. Full-day generic沒有執行；不得寫成full-day generic／derived等價。上述differential只驗工程語意，不是71日績效、獲利、champion或deployment GO。目前證據狀態是`precommit integration passed / post-commit durable rerun pending`。

Material component SHA-256 prefix如下；同一欄兩個對照clock的完整hash相同：

| component | 30m A＝B | full-day B＝C |
|---|---:|---:|
| accounting facts | `6bc3a087...` | `4fe8...` |
| carry in | `4f53...` | `4f53...` |
| carry out | `31ba...` | `c466...` |
| executions | `7e51...` | `311785...` |
| final capacity balances | `088780...` | `9df789...` |
| positions | `a423...` | `66c1...` |
| sent economic estimates | `6b845...` | `ee38...` |
| sent orders | `c425...` | `6d434...` |

永久重現入口為`maker.src.quote_fill.s1_clock_differential`。以下命令從nested repo root執行；clean commit後必須由該commit重跑，並把`schema=s1-clock-differential-material-v2`、`role=generic_effective / sparse_entry_route_derived / full_1hz_entry_route_derived`的checkpoints保存到durable目錄：

```bash
uv run --project ../../.. --no-sync python -m maker.src.quote_fill.s1_clock_differential \
  --date 20260505 --policy-id q95_C0_sd_f5 --horizon-minutes 10 \
  --checkpoint-dir maker/data/walkforward/s1_clock_differential_20260901_v2

uv run --project ../../.. --no-sync python -m maker.src.quote_fill.s1_clock_differential \
  --date 20260505 --policy-id q95_C0_sd_f5 --horizon-minutes 30 \
  --checkpoint-dir maker/data/walkforward/s1_clock_differential_20260901_v2

uv run --project ../../.. --no-sync python -m maker.src.quote_fill.s1_clock_differential \
  --date 20260505 --policy-id q95_C0_sd_f5 --compare-full-1hz \
  --checkpoint-dir maker/data/walkforward/s1_clock_differential_20260901_v2
```

## Run history

- `944c0ac`：第一次smoke因2354／GCFE6形成`exit_rollback_failed_unresolved`而fail closed；沒有complete partition。
- `42b689a`／`cd2b87c`：加入passive-exit風險與一tick headroom guard；`cd2b87c` control smoke成功。
- v3 root `s1_spot_bid_cost_aware_20260901_v3_exit_headroom_guard`：只有`ctrl_q95_C0_ungated × 20260505`完成（1/497）。
- v3 `q95_C0_sd_f5` generic-clock正式partition已停止且未完成；isolated diagnostic約558秒、peak RSS約20.7 GiB，只是效能診斷，不是PnL。
- v4：entry-decision-clock候選實作的precommit integration已通過10／30分鐘A/B及full-day B/C exact differential；完整preflight、clean commit、post-commit schema-v2 durable rerun及兩partition smoke尚未完成。
- v4 correctness fix：`tick_index_to_price`所有scalar price tier固定回傳float，修正500／1,000以上高價frozen-exit／carry strict JSON codec型別失敗；tick值與交易規則沒有改變。

## Frozen 七組

| scenario | upper／lower | cost admission | shortlist |
|---|---|---|---|
| `ctrl_q95_C0_ungated` | q95／C0 center | ungated control | 否 |
| `q95_C0_sd_f5` | q95／C0 | same-day modeled margin > 5 bp | 是 |
| `q95_C0_on_f0` | q95／C0 | overnight modeled margin > 0 bp | 是 |
| `q95_C2_sd_f5` | q95／C2 reach80 | same-day modeled margin > 5 bp | 是 |
| `q80_C0_sd_f5` | q80／C0 | same-day modeled margin > 5 bp | 是 |
| `q50_C3_sd_f5` | q50／C3 reach50 | same-day modeled margin > 5 bp | 是 |
| `fixed20_sym20_on_f0` | symmetric 20 bp | overnight modeled margin > 0 bp | 是 |

七組都保留共同 15,638 product-days × 4 TOD＝62,552 cells。C2／C3 lookup缺值不改用C0、不刪列，固定為
`lookup_supported=false`的no-trade。Economic gate是預先凍結的交易規則；研究本身仍沒有獲利pass/fail門檻。

## Actual-send economics

送單前以同一 causal cursor 計算：

```text
expected gross
= shares × [(frozen Spot exit target - Spot entry maker target)
            + (Future sell executable VWAP - Future buy executable VWAP)]

selected expected margin
= expected gross
  - Spot entry/exit commission
  - Spot sell tax (same-day 15 bp or overnight 30 bp)
  - Future entry/exit tax
  - Future entry/exit commission
```

已建模成本固定為 Spot commission每邊1.71 bp、Future tax每邊0.2 bp、Future commission每邊TWD 20。
Financing、borrow、futures margin opportunity cost與live reject／latency仍不可得，必須以unavailable reason揭露，不能填0。
所有 decision與actual-send estimate另外寫入可重播audit stream；control也定價，但不阻擋送單。

## 執行與出場語意

- Entry是 `Spot Bid maker → Future sell taker`；legacy makerFill只提供approximate full-fill screen。
- `actual_new_send_time`同時凍結entry Future Bid／Ask、lower provenance、absolute Spot Ask exit price與tick。
- Normal exit是 `Spot Ask maker → Future buy taker`。Absolute Spot Ask price／tick維持凍結，不因後續行情重定價；但被動單只有在Spot maker book合法、Future buy L1-L5足以完整買足該position，且最差swept ask上方仍保留至少一個嚴格位於Future合法價格band內的tick時才可工作。任一會改變normal-exit合法性、可執行VWAP、headroom或active target判定的Spot／Future狀態改變都會喚醒重驗；不影響route判定的raw lot／depth事件可略過。Future gate關閉時撤回desired／送cancel，恢復後仍只可回原凍結價。
- 這個pre-fill hedgeability／headroom gate不是流動性預留，也不取代B6。它只修正目前可辨識的上緣邊界風險，不保證零leg risk。若Spot maker在actual cancel effect前仍真實成交，fill依舊成立，並從fill+50 ms獨立判定Future hedge、最多retry 5秒，失敗再rollback。
- S1 carry route在13:19:45固定開始撤除所有passive exit desired；13:19:49.950的最晚安全成交barrier要求所有passive lifecycle已terminal，否則partition fail closed。Actual cancel effect前或同cursor的fill仍先於cancel。這不是S4 taker+taker hard flatten，也不能保證市場同步消失時永無裸腿。
- Hedge先在trigger+50 ms判定；不可執行或venue額度不足時，往後最多5秒找第一個合法足量且可送cursor。Timeout後走共同rollback。
- 20M global／10M product reservation在entry new actual-send前成立；maker fill只把reservation轉為exposure，完整exit hedge後才釋放。
- Expiry paired residual使用使用者指定的spot-close／spot-close、basis=0 accounting convention；不是execution fill或same-day completion。同cursor entry admission不得使用稍後settlement phase才釋放的expiry capacity。
- 任何最終`entry_hedge_timeout_unresolved`／`exit_rollback_failed_unresolved`都保留裸腿與committed capacity，並封鎖economic ranking；不得用8/13 common-horizon mark或expiry basis=0洗平。

報表另以`exit_desired_withdrawal_reason_counts`揭露guard影響；上緣buffer關閉固定記為
`gate:future_upper_band_headroom_lt_1_tick`，不得併入一般無深度或`safety_cutoff`。

## 8/13共同 ranking horizon

71個development entry sessions到2026-08-13為止；8/14起仍是protected forward，不拿來替development補runoff。
完整71日facts重播後仍open的paired position，使用8/13 13:20共同 causal L1–L5 hypothetical liquidation：long Spot掃Bid、
short Future掃Ask，扣已發生actual cost及依current lot acquisition date計算的remaining exit cost。同一scenario／商品的多筆
position先聚合數量再掃一次book，不得各自重複使用同一份顯示深度；明確clear／trial後不沿用更早book。

```text
economic ranking net = terminal realized net + open net mark
```

Mark不生成request／fill、不算completion、不釋放capacity，且在report中不得改稱realized PnL。任何open position無合法足量
book時，描述統計可發布，但economic ranking／Pareto／S2 shortlist全部withhold。

## Input與重現

- Spot tick／makerFill由top-level `config/pipeline.yaml`及`src/pipeline_storage.py`解析，目前canonical在
  `/media/kevin/SSD2/Data`；required mount不存在即fail closed。
- 個股期raw只接受`/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet`；TXF資料夾明確禁止。
- 每日manifest保存實際path、bytes、mtime與SHA-256；prepared loader必須逐role等於manifest，load後再驗hash未變。
- 每個policy/date partition使用deterministic gzip JSONL＋canonical JSON、hash chain、compact capacity checkpoint與exact
  identity registry。完整run還要有`complete.json`及獨立`verification.json`。
- `verification.json`只在497 partitions、input hashes、accounting/capacity/publication與最終產物全部deep verify成功後原子寫入；
  綁定complete／run-config／results／report hash、bundle與verifier source commits、partition count與verifier version。Verifier要求
  `maker/src`為同一clean commit；deep verify只接受已存在且bytes完全相同的artifact，任何中途缺檔都失敗且不得重建／留下receipt。

## 開跑條件

1. [完成，`cd2b87c`] Common-horizon valuation、report、publication ranking與exit-headroom接線通過完整preflight。
2. [完成，`cd2b87c`] 2026-05-05 control smoke通過；v3留下1/497 durable checkpoint。
3. [完成，precommit integration] 2026-05-05全218商品10／30分鐘generic A／derived B及full-day sparse-derived B／true-full-1Hz-derived C的八個material components exact。Full-day generic未執行，不在完成證據內。
4. v4完整S1 tests、path contract、Ruff、compile與`git diff --check`通過並建立clean source commit。
5. 從clean commit以永久runner重跑三組differential，保存schema v2、role-bound durable checkpoints並重驗hash。
6. v4新output root先跑control＋`q95_C0_sd_f5`兩個partition smoke，檢查cost decomposition、funnel、resume與input provenance。
7. 才執行71×7＝497 partitions；完成後另跑`verify --verify-inputs`，最後發布正式Markdown與bundle hash。
