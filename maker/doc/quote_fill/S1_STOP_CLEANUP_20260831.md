# S1 停止與清理摘要（2026-08-31）

## 結論

- 本次 S0.5／S1 啟動的 replay、測試、監控與研究代理均已停止；目前沒有 S1 背景程序。
- 舊 S1 只完成 77／497 partitions，且送單前沒有套用經濟 eligibility gate，不能續算後直接拿來排名，
  也不能視為部署 baseline。
- 未完成 bundle 與執行 cache 已清除；正式 S0、S0.5 產物、研究文件、程式碼與 Git 歷史均保留。

## 已保留的有效資訊

- S0 canonical 歸因與 30-session sensitivity。
- S0.5 frozen Q／lower lookup、校準與 geometry 結果。
- 舊 S1 的 11 日方向性摘要、source commit、run-config hash 與停止原因，保留於
  [`S1_PAUSED_STATUS_20260829.md`](S1_PAUSED_STATUS_20260829.md)。
- S1 replay engine、測試及規格文件；nested repository 在清理前是 clean worktree。

## 已清除

- `maker/data/walkforward/s1_spot_bid_joint_20260827_v2/`：4.6 GiB 的未完成 replay，包含
  4.3 GB identity registry 與 77 個 partial partitions。它沒有完整 verifier，也不符合新的成本感知 entry 問題。
- `maker/` 下的 Python `__pycache__` 與 Ruff cache；都可由下一次執行自動重建。
- 沒有刪除正式研究 bundle，也沒有停止 VS Code 語言服務、Jupyter kernel 或其他專案程序。

## 重新開始前要處理

1. 凍結少量、有經濟意義的 `Q × lower × same-day/overnight cost floor` 組合；保留 ungated control，
   但不得列入部署候選。
2. 在 actual new-send 時點，以當下可成交 Future bid／ask、Spot maker target、完整稅費與 safety margin 做 gate。
3. 把已失效的專案內 `data/` 預設路徑改為 `config/pipeline.yaml` 指向的 SSD2；個股期貨原始檔若 SSD2
   沒有等價資料，必須明示仍從 NAS 讀取，不可把 `txfTickData` 誤當個股期貨。
4. 先跑少量日期 smoke 並驗證成本分解與 publication gate，再建立全新的 71 日 output bundle。
