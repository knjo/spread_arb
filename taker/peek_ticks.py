"""進場/出場點 前後 N 秒 raw tick 抽取（單天單天跑，跟回測一樣）。

驗的東西（raw 全留、不聚合，後續自己算）：
  ① 進/出場點價格有沒有劇烈抖動  ② 哪一腳持續太短（期 vs 現各自撐多久）
  ③ 有沒有對手成交把價格吃掉（raw 含成交筆 is_fill，不過濾）
  以及後續任意指標：A1A2 spread、tick 差、滑價機率…都從這份 raw 算。

邏輯：
  1. 讀 out/events_{date}_tick.csv，取所有收斂事件(is_first_entry & converged)的 4 個錨點
     chseq：entry_fut/entry_spot/exit_fut/exit_spot。
  2. 撈當天 raw tick（load_spot/load_futures，跟回測同源；含成交筆，不過濾 fill）。
  3. 每個錨點用 ChannelSeq 定位那一筆 → 取它的 RecvTime → 框 ±窗口秒。
  4. raw 原樣保留 + 標註欄：event_id / anchor_type / leg /
     exit_stretch_secs / first_stretch_secs / first_ret / converge_time。
  5. 留重複（同段被多錨點框到各存一份，標註不同）。存 out/peek/peek_{date}.csv。

用法（代跑，連 NAS，一次一天）：
  uv run python peek_ticks.py 20260617
  uv run python peek_ticks.py 20260617 0.5     # 窗口改 ±0.5 秒
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from datetime import timedelta
import polars as pl
from spread_arb.preprocess import load_spot, load_futures, filter_near_month, TIME_COL

OUT_DIR = "out"
PEEK_DIR = "out/peek"
SEQ_COL = "ChannelSeq"   # raw tick 的序號欄；events 的 *_chseq 對應它

# 帶進每列 raw 的事件層標註欄（常用指標，免事後 join；其餘靠 event_id 回 events CSV）
# potential_lots：供「出場點共用」分析；threshold：供事後按門檻分（收斂事件含 0.5/1/1.5/2%，
#   0.5% 占絕大多數會主導全體數字，故標起來、analyze 可分門檻看）。
EV_TAGS = ["threshold", "exit_stretch_secs", "first_stretch_secs", "first_ret",
           "converge_time", "potential_lots", "contract_size"]

# 4 個錨點：(標籤, chseq 欄, 哪一腳)
ANCHORS = [
    ("entry_fut",  "entry_fut_chseq",  "fut"),
    ("entry_spot", "entry_spot_chseq", "spot"),
    ("exit_fut",   "exit_fut_chseq",   "fut"),
    ("exit_spot",  "exit_spot_chseq",  "spot"),
]


def extract(raw: pl.DataFrame, ev: pl.DataFrame, leg: str, chseq_col: str,
            anchor: str, win: float, date: int) -> pl.DataFrame:
    """對某一腳 raw，依『代碼 + event_id』鎖定每筆事件 → 用該事件 chseq 定位時間 →
    只框『該檔』±win 秒 → 加標註（含 event_id）。

    關鍵 1：每個錨點先鎖定商品（期=QuoteCode、現=ValueCode），否則框時間窗會把全市場
      那一秒灌進來（曾踩：單事件框出 1988 檔）。
    關鍵 2：事件識別用 event_id，不用 chseq——**多個事件可能在同一筆 tick(同一 chseq)收斂**
      （同一檔多部位同時平倉，0623 實測一個 chseq 對到 64 事件）。用 chseq 當鍵會把同段 raw
      複製 64 次；用 event_id 鎖事件，共用 chseq 的事件各帶自己的 event_id、各一份窗口（要的）。
      event_id 在「當天該檔」內唯一（跨日才會重複，但 peek 單天單檔故安全）。chseq 只拿來找時間。

    跨日防呆：進場錨點(entry_*)只框 date==分析日 的事件——chseq 每天重編、跨日會撞號，
      別天進場的 chseq 拿來當天 raw 找會張冠李戴。出場一定在當天(這份就是當天收斂事件)，不需此濾。
    """
    if raw.height == 0:
        return pl.DataFrame()
    code_col = "QuoteCode" if leg == "fut" else "ValueCode"   # 該腳的識別欄
    base = ev
    if anchor.startswith("entry"):
        base = ev.filter(pl.col("date").cast(pl.Int64) == date)   # 進場必須在當天才對得上 chseq
    # 事件表側：每筆事件一列 (event_id, 代碼, chseq, 標註)，chseq 非空才有
    sub = base.filter(pl.col(chseq_col).is_not_null()).select(
        pl.col("event_id"),
        pl.col(code_col).cast(pl.Utf8).str.strip_chars().alias("_code"),
        # chseq 兩邊 cast 同型 Int64：某天 events 的 chseq 欄被 read_csv 推成 str，
        # 與 raw 的 ChannelSeq(u64) join 時型別不合 → SchemaError。統一 Int64。
        pl.col(chseq_col).cast(pl.Int64, strict=False).alias("_seq"),
        *[pl.col(c) for c in EV_TAGS if c in ev.columns])
    if sub.height == 0:
        return pl.DataFrame()
    raw = raw.with_columns(pl.col(code_col).cast(pl.Utf8).str.strip_chars().alias("_code"))
    # chseq → 該事件錨點時間：raw 裡 (該檔, ChannelSeq==chseq) 的 RecvTime（chseq 只用來找時間）
    seq_time = (raw.select("_code", pl.col(SEQ_COL).cast(pl.Int64, strict=False).alias("_seq"),
                           pl.col(TIME_COL).alias("_anchor_t"))
                   .unique(subset=["_code", "_seq"]))
    anchors = sub.join(seq_time, on=["_code", "_seq"], how="inner")   # 每事件拿到自己的錨點時間
    if anchors.height == 0:
        return pl.DataFrame()
    # 每筆事件 → 只框『該檔』raw 的 ±win（共用 chseq 的事件各框一份、各帶 event_id）
    out = []
    for a in anchors.iter_rows(named=True):
        t = a["_anchor_t"]
        seg = raw.filter(
            (pl.col("_code") == a["_code"])
            & (pl.col(TIME_COL) >= t - timedelta(seconds=win))
            & (pl.col(TIME_COL) <= t + timedelta(seconds=win)))
        if seg.height == 0:
            continue
        seg = seg.with_columns([
            pl.lit(a["event_id"]).alias("event_id"),
            pl.lit(anchor).alias("anchor_type"),
            pl.lit(leg).alias("leg"),
            pl.lit(a["_code"]).alias("anchor_code"),
            pl.lit(a["_seq"]).alias("anchor_chseq"),
            # 明確指定 dtype：某天某欄整段為 None 時 pl.lit(None) 會推成 Null 型，
            # 跨 anchor 段 concat 撞 Float64 → SchemaError。時間欄 String、其餘數值 Float64。
            *[pl.lit(a[c], dtype=(pl.String if c == "converge_time" else pl.Float64)).alias(c)
              for c in EV_TAGS if c in anchors.columns],
        ])
        out.append(seg.drop("_code"))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def run_day(tw, date: int, win: float, thr: float | None = None) -> None:
    """單日：讀 events → 撈 raw → 抽各錨點 ±win → 存 peek_{date}.csv。

    thr：只取該門檻的收斂事件（None=全部）。收斂事件含 0.5/1/1.5/2%，0.5% 占多數會主導，
    要看高門檻的微結構就指定。標註欄也帶 threshold，故不指定時 analyze 仍可事後分門檻。
    """
    ev_path = os.path.join(OUT_DIR, f"events_{date}_tick.csv")
    if not os.path.exists(ev_path):
        print(f"  {date}: 找不到 {ev_path}，跳過")
        return
    ev = pl.read_csv(ev_path, infer_schema_length=20000)
    ev = ev.filter(pl.col("is_first_entry") & pl.col("converged"))
    if thr is not None:
        ev = ev.filter(pl.col("threshold") == thr)
    print(f"== peek_ticks {date} | 收斂事件 {ev.height}"
          f"{f' (門檻 {thr})' if thr is not None else ''} | 窗口 ±{win}s ==")
    if ev.height == 0:
        print("  無收斂事件，跳過"); return

    print("  撈 raw tick(NAS)...", flush=True)
    fut = filter_near_month(load_futures(tw, date), date)
    spot = load_spot(tw, date)
    print(f"  期貨 raw {fut.height:,} 列｜現貨 raw {spot.height:,} 列", flush=True)

    parts = []
    for anchor, chseq_col, leg in ANCHORS:
        raw = fut if leg == "fut" else spot
        seg = extract(raw, ev, leg, chseq_col, anchor, win, date)
        print(f"  {anchor:<11}: {seg.height:>9,} 列")
        if seg.height:
            parts.append(seg)
    if not parts:
        print("  無資料，跳過"); return
    result = pl.concat(parts, how="diagonal_relaxed")

    os.makedirs(PEEK_DIR, exist_ok=True)
    out_path = os.path.join(PEEK_DIR, f"peek_{date}.csv")
    result.write_csv(out_path)
    print(f"  → {result.height:,} 列(含重複) 已存：{out_path}")


def main():
    import argparse
    from datetime import datetime, timedelta
    p = argparse.ArgumentParser(description="進/出場點前後 N 秒 raw tick 抽取（逐日，跟 main.py 一樣）")
    p.add_argument("-s", "--start_date", type=str, required=True, help="起始日 yyyymmdd")
    p.add_argument("-e", "--end_date", type=str, help="結束日 yyyymmdd（不給=同起始日，單天）")
    p.add_argument("-w", "--window", type=float, default=1.0, help="前後窗口秒數（預設 1.0）")
    p.add_argument("-t", "--threshold", type=float, default=None,
                   help="只取該門檻收斂事件 0.005/0.01/0.015/0.02（不給=全部，標註欄仍帶 threshold）")
    args = p.parse_args()

    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start
    tw = None   # None → preprocess 直接讀 SSD2 現貨 / NAS 股期
    cal = start
    while cal <= end:
        ymd = int(cal.strftime("%Y%m%d"))
        # 沒有當天 events 檔就跳過（非交易日自然沒檔）；撈檔層另有保護
        run_day(tw, ymd, args.window, args.threshold)
        cal += timedelta(days=1)


if __name__ == "__main__":
    main()
