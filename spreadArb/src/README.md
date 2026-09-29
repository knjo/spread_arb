# spreadArb 執行入口

以 [CURRENT.md](../doc/CURRENT.md) 為規則、架構、指令與驗證狀態的唯一入口。
9/29 已選定回零 EV 篩選＋max-Q 出場，單組入口預設 B 動態 hurdle。
採用結果 `zero_score_maxq_indexed_20260924`：131 日完整驗證，B 年化 31.5347%。
見 [目前規則與期末統計](../doc/CURRENT.md)。

從 `futures_spot_spread/` 執行：

```bash
export PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=2
RUN="uv run --project /home/kevin/Project/HFT --no-sync python"
$RUN -m spreadArb.src.ev.build --start 20260126 --end 20260813
$RUN -m unittest discover -s spreadArb/src/tests -t .
# 單獨執行採用的 B；未指定 --preset 也預設 B。
$RUN -m spreadArb.src.backtest.replay --preset B --out my_B --prefetch 0
$RUN -m spreadArb.src.backtest.causal_audit --run my_B --raw-all
$RUN -m spreadArb.src.backtest.ev_validation --run my_B
$RUN -m spreadArb.src.backtest.accepted_report --run my_B
```

中斷用原本名稱、完整期間及 `--resume`。原始碼、政策及 Q 快取必須相同；否則另取新的輸出名稱。
prefetch 已移除，勿再同時啟動兩份 A／B 大回放。

清快取先跑 `$RUN -m spreadArb.src.backtest.maintenance --receipt /tmp/spreadarb_cleanup_preview.json` 看清單；加 `--apply --receipt /tmp/spreadarb_cleanup.json` 才執行。
清除後須重建 Q 快取。不要在回放／稽核期間清理。

`causal_market.py` 產生當下可見候選；`policy.py` 算 EV；`causal_replay.py` 按時序撮合、hedge 與記帳。
`valuation.py` 使用精確持有合約的官方日價格評價。`causal_audit.py` 獨立核對資金／現金／EV 訊號母體，
搭配 `fifo_audit.py` 與 `refresh_audit.py` 重建原始成交隊列、掛單壽命與當下價格。
`ev_validation.py` 重算所有送單及計時更新（含拒單）的 EV／hurdle，另列預測與實現差異；最後才產生 `accepted_report.py` 報告。

輸出保存在 `data/backtest/<run>/`。完整驗收需 `verification.json` 的 `complete=true`／`status=PASS`、全日 raw 檢查與 EV 驗證。
`legacy_replay.py`、`points/` 與舊稽核器保留作歷史研究；現行回放不讀取未來成交點位。

需要 A／B 對照時，改用 `backtest.causal_replay --out my_AB --prefetch 0`，後續檢查與報告也改用 `my_AB`；兩種回放擇一執行。
