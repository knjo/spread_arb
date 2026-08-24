# WP02 Raw Tick Replay：分層掛單與取樣契約

## 定位

本研究重播的是「同一路線可以同時保留多個不同價位的實體 maker order」，不是 target 每次變動就取消舊單的單一 requote 模型。

核心規則：

1. target 往更積極的合法價位移動時，舊的較保守掛單保留，新價位新增一層。
2. 同一個 spread epoch、route、stage、絕對掛價最多建立一次；B1／B2 只是會隨行情改變的狀態，不是訂單 ID。
3. target 往較不積極方向退回時，撤掉比新 target 更積極的 working orders；較保守的舊層繼續保留。
4. 第一版 executable replay 假設撤單立即成功；同時保存撤單需求、丟棄的 queue age，以及撤單後的 shadow tape，供後續 cancel-latency sensitivity 使用。
5. WP02 第一階段允許每個合格 order generation 依既有 maker-fill state 獨立產生研究 label，用來估條件期望值；同日／同 Pair 的事件仍共享行情，統計推論不得當成 IID。成交量、hedge depth、庫存與部位守恆延後到 portfolio replay。

本文件只定義 WP02 的取樣與 order lifecycle。50 ms hedge、費稅與 portfolio EV 分別由 WP03–05 負責。

## SpreadPair 取樣時鐘

HFT 既有 `SpreadPair` 是「現貨 spread 張開時 captured 的 A1／B1 價對」，不是期現 basis pair。WP02 要的是價對狀態真的切換才開新批次，不是每次 widening 都開：

- `SpreadPairID` 是同一組 captured `(A1, B1)` 的固定 ID；相同價對日內重現會重用 ID。
- `SpreadPairSeq` 記同一 `SpreadPairID` 第幾次重新進入。
- `SpreadPairTotalCount` 在 `SpreadPairID` 改變時單調增加，正好可作本研究的 base sampling epoch。
- `calc_spread_pair_stats()` 內部 `_pp` 會在每次有效 widening 遞增，並重設 `SpreadPairElapsed`；它對本研究太細，只保留作市場事件診斷。
- `SpreadCountAtSameCount` 可描述同一價對期間又發生幾次 spread increase，但不觸發新的 base sample。

因此 WP02 定義：

```text
spread_pair_epoch = SpreadPairTotalCount
pair identity audit = (SpreadPairID, SpreadPairSeq)
```

若上游欄位缺失，WP02 loader 才以 `SpreadPairID` 的 causal transition 重建同語義 epoch；不要改用每次 `valid_capture` 的 `_pp`。

這會刻意得到以下行為：

```text
窄化 -> 又張回完全相同 captured A1/B1
=> SpreadPairID 不變
=> SpreadPairTotalCount 不變
=> 不開新的 base sample
```

若曾切換到另一組 captured pair 後再回來，`SpreadPairTotalCount` 會增加，才視為新的 epoch；`SpreadPairID + SpreadPairSeq` 可辨認這次 re-entry。`SpreadNarrowOrderTime／Side` 是未來結果，只可作 outcome audit。

`spread_pair_epoch` 是一般新增掛單機會的 sampling clock。即使 epoch 沒更新，只要 rounded target 往前移到同 epoch 尚未使用的新絕對價格，仍可新增一層；這正是期現同方向平移、舊 B1 變 B2 而新 target 又成為 B1 的例外。若回到同 epoch 已看過的絕對價格則仍然 suppression，不會因窄化／再張開而重抽。

## 價格方向

四條 route 一律使用 maker side 的積極度，不用含糊的「上移／下移」：

```text
aggressiveness(price) = +tick_index(price), maker side = Bid
aggressiveness(price) = -tick_index(price), maker side = Ask
```

- aggressiveness 上升：target 往前；保留較保守舊層並視需要新增一層。
- aggressiveness 下降：target 退後；撤掉積極度高於新 target 的層。
- 只有 rounded legal target 真正換 tick 才觸發，不因 fair 的小數 BP 抖動製造假掛撤。

## Reconciliation 狀態機

每個 product／route／stage 維護：

```text
active_orders_by_absolute_price
seen_prices_in_current_epoch
current_rounded_target
reserved_quantity
```

每個 causal market update 先更新 fair、反腿可成交價、合法 ladder 與 target，再依下表 reconciliation：

| 事件 | 動作 |
|---|---|
| 新 `spread_pair_epoch` | 允許評估一個新的 base intent；epoch 本身不強迫取消舊單 |
| 同一 epoch，target 與已建立價格相同 | 不新增，記 `same_price_suppressed` |
| 同 epoch target 往前到未見過的新絕對價 | 新增一層，所有較保守舊層繼續 working |
| 同 epoch 重複到已見過價格 | 記 `same_price_suppressed`，不新增 order |
| target 退後 | 撤掉比新 target 更積極的 layers；保留等價與較保守 layers；同 epoch 不在退後價重開新樣本 |
| RefPrice／TrialMatch／book／risk gate 失效 | 撤掉該 gate 涵蓋的所有 layers，停止新掛 |
| cutoff／日終 | 撤掉所有 leaves，結束當日 replay |

新 epoch 代表新的研究 admission。即使某絕對價已有跨 epoch 存活 order，也建立新的 order generation／樣本；同價去重只限制在同一 `spread_pair_epoch`。因此跨 epoch 同價可以同時存在多筆研究 orders，各自使用起始 maker-fill state 估 outcome。是否真的同時送出全部數量、後單是否計入前面自己的 leaves，以及總部位上限，延後在 portfolio replay 加入，不回頭改寫 WP02 的自然樣本表。

### Spot Bid 範例

```text
t0  epoch=21，target=100（當時 B1）
    -> 建 O1@100

t1  epoch=21，期現一起上移，target=101（新 B1）
    -> O1@100 現在是 B2，保留
    -> 建 O2@101

t2  epoch=21，target 仍為 101
    -> 不建 O3

t3  epoch=21，target 退回 100
    -> 撤 O2@101
    -> O1@100 繼續 working，不重掛 100

t4  epoch=22，target=100
    -> 不論 O1 是否仍 working，epoch 更新允許建 O3@100
    -> O3 是新的研究 generation；最終 portfolio replay 才套 qty／own-order queue／position cap
```

Ask maker 完全鏡像：target 向下是往前，target 向上是退後。

## 取樣與資料表

不得把 raw tick、B1／B2 rank 變化或 order state interval各自當成新的 fill 樣本。資料分成六層：

1. `spread_pair_epoch`
   - 每次 captured `SpreadPairID` 真正切換一列；同價對內的重複 widening 不新增。
   - cohort key：`Date, ValueCode, spread_pair_epoch`。

2. `candidate_intent`
   - 一個 qualifying epoch 或同 epoch 新前方價位一列。
   - outcome：`admitted / same_price_suppressed / gated`；capacity 不在 WP02 自然樣本表刪資料。
   - 去重 key：

   ```text
   Date, ValueCode, QuoteCode, route, stage,
   spread_pair_epoch, maker_side, absolute_maker_price_tick
   ```

   `intended_qty`、fair／boundary／width policy 不在 raw intent identity；它們放在 many-to-one alias。Raw fact先保存 stop 前的 `fillable_qty_path`，再由各 qty／policy派生結果，避免同價重播多次。

3. `physical_order`
   - WP02 中是一次 independent research order generation 一列，是條件 fill 表的 order-level 分母；不表示 WP05 必然同時送出全部 generations。
   - raw replay key：

   ```text
   episode_start_recv_time, Date, ValueCode, QuoteCode,
   route, stage, maker_side, rounded_target_price,
   intended_qty, order_generation, queue_replay_version
   ```

   必帶 `independent_event_label=true`；只有 WP05 joint replay 後才能標 `executable_portfolio_order=true`。

4. `order_state_spell`
   - 同一 order 只在狀態改變時新增 interval，例如 `CURRENT -> AWAY_1 -> AWAY_2 -> CANCELLED`。
   - 每列保存當時 rank、queue ahead、target distance、fair、反腿價格與 causal state。
   - 用於 time-varying hazard，不當成獨立 physical order。

5. `fill_delta`
   - 每次新增 partial／full filled quantity 一列。WP02 的 independent-event table 各自依起始 maker-fill state 標記；另帶 cohort／date keys，避免把共享行情誤當 IID。

6. `hedge_fact`
   - 每個 incremental maker fill 對應 `fill_recv_time + 50 ms` 的 hedge child；若策略會 batch，需先合併 pending quantity再掃一次共同 book depth。

不同 fair／boundary aliases 若落到同一 raw order，可 many-to-one 共用 queue path fact；但各 policy 的 admission、gate、reservation 與 portfolio 結果仍需分開 replay。

## Order lifecycle 與撤單假設

第一版 physical order lifecycle：

```text
WORKING_CURRENT
    -> WORKING_AWAY                 target 往前，新層另開
    -> PARTIAL_WORKING / FILLED     queue depletion 或 trade-through
    -> CANCELED_TARGET_RETREAT      target 退後
    -> CANCELED_GATE / CUTOFF       safety／session

WORKING_AWAY
    -> WORKING_CURRENT              target 回到原價；不重掛
    -> PARTIAL_WORKING / FILLED
    -> CANCELED_TARGET_RETREAT
```

V0 在 retreat tick 的 causal `RecvTime` 立即讓 leaves terminal，之後的成交不計入 executable fill。仍須保存：

- `cancel_request_recv_time` 與原因；
- cancel 前 order lifetime 與被放棄的 queue age；
- raw tape 在 `+10／50／100／500 ms` 是否觸價／穿價；
- 若加入假設 cancel latency，本來會出現的 shadow cancel-race fill 與 50 ms hedge slippage。

因此 V0 的「cancel rate」是策略需要撤單的頻率，不是交易所 cancel ACK 成功率。後續有 submit／cancel latency 假設時，request 到 effective cancel 期間才正式納入 `CANCEL_PENDING` 與 cancel-race fill。

主動撤單是 competing terminal outcome，不是資料 censor；只有缺檔、raw sequence 中斷或無法判定 queue 才標 unknown／censored。

## Independent-event table 與共同資源

WP02 主表先回答「若這個 Pair／價位送出一單，條件 fill／cancel／hedge cost 表現如何」，允許跨 Pair 同價各自形成事件。這符合先用樣本估期望值、最後才決定部位控制的研究順序。

但同一路徑上的 active layers 在統計與最終實盤並非完全獨立：

- 同一筆市場成交量只能依價格時間優先分配一次。
- 同價舊 order 的未成交 leaves 必須排在較晚 generation 前面。
- 所有 entry routes 共同占用 position／unhedged／reserved quantity。
- cancel request 在真實 latency 版本中，要到 cancel effective 才釋放 reservation。
- 多筆 maker fill 的 50 ms hedge 會共享同一時點的 taker depth。

因此輸出分開標記：

- `independent_event_estimate=true`：WP02 機率／成本研究表，不套部位上限。
- `joint_portfolio_replay=true`：WP05 才依實際政策套 own-order priority、共同成交量、hedge depth、qty 與 inventory cap。

兩者不可混稱；但 WP02 不因最後尚未決定部位控制而刪掉合格樣本。

## 最低統計輸出

### 去重與掛單層數

```text
eligible_intents
physical_orders_started
same_price_suppression_rate
cross_epoch_same_price_generation_rate
new_forward_layer_rate
orders_per_spread_epoch
active_layers time-weighted p50/p95/max
same_market_path_cluster_size
```

### 撤單與 queue 浪費

```text
cancel_required_rate = cancel_required_orders / physical_orders_started
target_retreat_cancel_rate
gate_cancel_rate
cancelled_qty / submitted_qty
cancel_before_any_fill_rate
partial_then_cancel_rate
working_lifetime_before_cancel p50/p80/p95
discarded_queue_age p50/p80/p95
cancels_per_useful_fill
```

### Fill 與 adverse selection

```text
any_fill / full_fill / filled_qty_ratio
fill_reason = queue_depletion / trade_at_price / trade_through
retained_away_fill_share
fill_near_retreat_share
shadow_cancel_race_fill_rate by assumed latency
submit_to_fill edge decay
```

WP03 再按 `current layer／retained-away layer／retreat 前／cancel-pending` 分組報：

- 50 ms hedge completion；
- taker 掃過檔數；
- signed slippage p50／p90／p99；
- hedge 後 edge 與 negative-edge rate。

策略是否可行不能只看撤單率。真正的警訊是：

```text
高 cancels / useful fill
+ 大量短命訂單與 queue priority 浪費
+ fill 集中在穿價／退後前後
+ 這些 fill 的 50 ms hedge slippage 吃掉預期 edge
```

以上指標按 product、route、maker spread ticks、時段與 causal volatility state 分開報；先不事後挑任意門檻。

## 統計與驗證

- 多層共享行情，不能把 order rows 當 IID；同時報 event-weighted 結果與 equal-weight product-day／epoch 結果。
- Aggregate CI 使用 whole-date block bootstrap；單商品表可使用 product-day block。
- Train／validation 按日期 walk-forward，不能 random split 同一日的 orders。
- 若 training 抽未成交／撤單樣本，保存 inclusion probability；calibration 與 replay 保留自然 base rate。
- 先用 existing spot A1／A2／B1／B2 makerFill 作 sanity check；任意價位與 future maker fill 仍以 raw MBP bounds 重播。
- MBP 無法辨認 queue-ahead cancel 或 queue-behind cancel時，同時輸出 conservative／base／optimistic queue-depletion assumptions，不能宣稱精確 MBO queue。
