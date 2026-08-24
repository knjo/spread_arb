# archive_fixed45

固定 45 檔研究 cohort（2026-08-14 ～ 08-21）的結果文件。該 cohort 以 May–Aug 已實現流動性取交集後回套到全部 60 日，
含 target-day 資訊，因此全部標 `universe_selection_d_safe_go=false`。

2026-08-24 清理時：

- 對應資料（`execution_narrow_60d`、`exit_maker_narrow_60d`、cross-session cache、post-cross、prequential、aggressive、compact 等，約 23 GB）已刪除。
- 對應程式（56 個 `quote_fill` 模組、測試與 `doc/quote_fill/*.py` 畫圖腳本，約 55K 行）已刪除。
- 兩者都可由 nested repo `src/research/futures_spot_spread` 的 commit `1348576`（`<Snapshot> maker research pre-cleanup`）撈回程式；資料不可復原，需重跑。

文件內指向 `../../data/walkforward/...` 的相對連結全部失效，屬預期。
