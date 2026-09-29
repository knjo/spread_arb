# spreadArb：期現價差 A／B

現行規則、架構與執行方式統一看 [CURRENT.md](doc/CURRENT.md)；損益與年化看 [03_RESULTS.md](doc/03_RESULTS.md)。

未成交單不占部位，部分／完整成交即時計入；實際平倉後才釋放。同商品 S1／S2／E1／E2 各一張，可並存，持有部位不限筆數。
各路線掛單生效滿 60 秒送撤，撤單生效後按最新行情與條件重評。

**9/29 已採用「回零 EV 篩選＋max-Q 出場」，以 B 動態 hurdle 為主。**
單組回放預設 B；A 保留固定 hurdle 對照。採用 `zero_score_maxq_indexed_20260924` 的已驗證執行來源，
131 日完整成交／資金／EV 稽核通過，另通過 150 項回歸與 2 項索引選擇測試。

B 淨利 **3,304,834.12 元／年化 31.5347%**；A 對照為 2,967,513.18 元／28.3160%。
兩組期末各兩筆未平倉，官方淨評價各 -2,856.40 元已計入；固定資本 20M、250 日單利年化。
[完整期末統計](doc/10_EV_AND_EVALUATION_20260929.md)、[採用證據](data/backtest/selection_20260929/adoption.json)。
實際目標 EV 與舊點位引擎保留作歷史比較，不能與採用版混用。

- [指令入口](src/README.md)
- [Q 表與 EV 定義](doc/01_Q_TABLE_EV.md)
- [容量缺陷與「提前釋放」指控更正](doc/05_CAPITAL_POLICY_CLARIFICATION.md)
- [舊 nores 部分成交漏算與漏撤證據](doc/06_REVALIDATION_20260923.md)
- [每路線掛單數限制](doc/07_QUOTE_CONCURRENCY_CLARIFICATION.md)
- [整理前 README](doc/archive/README_before_refresh_20260923.md)

現行使用原始五檔／成交逐事件回放，共用 A／B 行情載入，prefetch=0。
原始資料及歷史驗證證據保留；可重建快取清理由 `backtest.maintenance` 管理。
