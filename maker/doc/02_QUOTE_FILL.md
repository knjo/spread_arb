# Work Package 02：動態掛價與 Maker Fill

D−1 商品上下界與 latent 機率先由
[quote_width/ADAPTIVE_BOUNDS.md](quote_width/ADAPTIVE_BOUNDS.md) 產生；
[quote_width/CYCLE.md](quote_width/CYCLE.md) 的固定 BP／tick 只作回歸診斷。
本 WP 才負責把商品 prior 與當下 causal state 轉成真正的 quote episode 與 maker fill label。

多層存續掛單、SpreadPair epoch、同價去重與撤單口徑以
[quote_fill/REPLAY_SAMPLING.md](quote_fill/REPLAY_SAMPLING.md) 為 source of truth。

## 研究問題

給定 fair、反向可 taker 價與目標掛價，每一張實體 maker order 是否能在自身被撤掉或停止前成交？target 往前產生新價位時，舊的較保守 order 繼續工作，不再把任意 target move 當成舊單 requote 終點。

```text
tau_fill
tau_opposite_move
tau_fair_move
tau_hedge_action_move
tau_target_retreat
tau_gate
tau_risk_end = min(tau_gate, hard_TTL, session_cutoff)

fill_wins = tau_fill < min(tau_target_retreat, tau_risk_end)
```

主要 competing outcomes 為 `fill_wins`、`target_retreat_cancel`、`risk_gate_cancel`；另保存 `fill_before_opposite_move` 作微結構診斷。V0 假設 retreat 時立即撤單，並另外保留撤單後 raw tape 作 latency／cancel-race shadow sensitivity。

## Quote episode 取樣

Raw physical-order replay key：

```text
episode_start_recv_time, Date, ValueCode, QuoteCode,
route, stage, maker_side, rounded_target_price,
intended_qty, order_generation, queue_replay_version
```

- 以 `SpreadPairTotalCount` 作單調 `spread_pair_epoch`：只有 captured `SpreadPairID` 真正切換才開新的 base opportunity；窄化後張回相同 A1／B1 不重抽。`SpreadPairID + SpreadPairSeq` 作 re-entry 稽核。
- 一般掛單機會由新 epoch 觸發；同 epoch 若 rounded target 往前到未見過的新絕對價，也可新增一層。
- 同 epoch、route、stage、absolute price 只建一次；同價訊號記 alias／suppressed，不複製 raw fill 樣本。
- 舊單由 B1 變成 B2／B3 只是 state spell 改變，不是新 order；新絕對 B1 才能是另一張 order。
- 往前移保留舊的較保守 layers；退後移撤掉比新 target 更積極的 layers。Fill、明確 cancel、risk gate、hard TTL 或 cutoff 才結束個別 order。
- 新 epoch 若同絕對價已有 live order，重用舊 order，不重掛同價；舊 order terminal 後才可建立新 generation。
- 同價不同 fair／boundary policies 共用 raw queue／first-fill fact；另以 many-to-one alias table 保存 `fair_model_version`、`adaptive_parameter_version`、`source_asof_date`、causal state、effective upper／lower 與 requote／gate policy。
- Episode 只存 `spot_state_id`，不複製整張 feature row。
- Validation／test 使用完整自然 base rate；train 若抽未成交樣本需保存 inclusion probability。

盤中先枚舉每條 route 的合法 maker prices，再由當下 fair 與反腿可成交價反算 effective BP／ticks。固定 `10／15／20 bp` 或 `1／2 tick` 都不作 action allowlist。

## Fill model

- Spot maker：既有 A1／A2／B1／B2 makerFill 作一致性檢查；任意掛價由 raw tick 重播。
- Future maker：用 MBP queue depletion／trade-through 重播，不建期貨預測 feature。
- Touch 不是成交，只作樂觀上界。
- V0 executable outcome 假設撤單在 retreat observation 當下生效；同時保存 `+10／50／100／500 ms` shadow path，供後續 cancel-latency／cancel-race sensitivity。
- 多個 live layers 需 joint allocation 同一筆成交量、inventory 與 50 ms hedge depth，不能各自消耗完整 tape。

## 最低輸出

- Fill／target-retreat cancel／risk-gate 的 competing probability。
- Quote lifetime、queue ahead、fill reason、partial fill與同時 active layer 數。
- 同價 suppression、每 epoch 新增 layers、reservation／capacity block。
- 依原因拆分的撤單需求率、cancelled qty、丟棄 queue age與 cancels per useful fill。
- RefPrice／TrialMatch 排除量與不同 latency 假設下的 shadow cancel-race tail。
- Route、stage、掛價 ticks、時段與現貨 state 的條件表。
