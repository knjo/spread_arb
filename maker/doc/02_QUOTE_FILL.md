# Work Package 02：動態掛價與 Maker Fill

候選 width、D−1 商品參數及 rounded entry quote geometry 已先由
[quote_width/RESULTS.md](quote_width/RESULTS.md) 產生；該表只描述 latent basis path，
本 WP 才負責把它轉為真正的 quote episode 與 maker fill label。

## 研究問題

給定 fair、反向可 taker 價與目標掛價，maker 是否能在訂單需要改價或停止前成交？

```text
tau_fill
tau_opposite_move
tau_fair_move
tau_hedge_action_move
tau_quote_move = min(tau_fair_move, tau_hedge_action_move)
tau_gate
tau_risk_end = min(tau_gate, hard_TTL, session_cutoff)

fill_wins = tau_fill < min(tau_quote_move, tau_risk_end)
```

主要 competing outcomes 為 `fill_wins`、`requote_wins`、`risk_gate_wins`；另保存 `fill_before_opposite_move` 作微結構診斷。

## Quote episode 取樣

Action signature：

```text
Date, ValueCode, QuoteCode, route, stage,
rounded_target_price, intended_qty,
eligibility_state, fair_model_version, width_policy_version
```

- Signature 不變就維持同一 episode，不因每個 tick 或 feature 更新重複造樣本。
- Fill、rounded target 改變、risk gate、hard TTL 或 cutoff 才結束。
- 同價不同 width 可共用 raw first-fill fact。
- Episode 只存 `spot_state_id`，不複製整張 feature row。
- Validation／test 使用完整自然 base rate；train 若抽未成交樣本需保存 inclusion probability。

## Fill model

- Spot maker：既有 A1／A2／B1／B2 makerFill 作一致性檢查；任意掛價由 raw tick 重播。
- Future maker：用 MBP queue depletion／trade-through 重播，不建期貨預測 feature。
- Touch 不是成交，只作樂觀上界。
- Cancel pending 期間成交必須保存為 cancel-race fill。

## 最低輸出

- Fill／requote／risk-gate calibrated probability。
- Quote lifetime、queue ahead、fill reason、partial fill。
- RefPrice／TrialMatch 排除量與 cancel-race tail。
- Route、stage、掛價 ticks、時段與現貨 state 的條件表。
