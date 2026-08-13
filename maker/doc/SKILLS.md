# Research Skills

這裡的 skill 指研究模組能力與責任邊界，不是模型 feature。

| Skill | 責任 | 不負責 |
|---|---|---|
| `data_contract` | 時間、價格、合約、RefPrice／TrialMatch gate | 模型挑選 |
| `fair_mid` | Stable anchor、predictive fair、uncertainty、churn | Maker fill 假設 |
| `quote_episode` | Action signature、取樣、competing risks | 50 ms hedge 成本 |
| `queue_replay` | Spot／future MBP fill bounds | 宣稱精確 MBO queue |
| `hedge_50ms` | L1–L5 VWAP、partial、retry、slippage | Fair 預測 |
| `portfolio_replay` | 部位、reservation、費稅、週轉 | 修改上游 labels |
| `validation` | Walk-forward、calibration、block bootstrap、leakage audit | 事後挑最佳期間 |

每個 skill 應有獨立 source module、測試與資料輸出，不在單一 script 內隱式互相修改定義。
