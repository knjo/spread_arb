"""期現貨套利 低頻分析 — 含二次進場放大版報表。

與 report_first（只算第一次進場）唯一差別：把第一次進場那筆的量放大成
  放大 lots = floor(potential_lots + carry_second_lots × 參與率)
            = 第一筆自己的量 + 後續二次進場累積量 × 10%
其餘完全仿照 report_first（成本/利潤/淨利/本金/五張報表都複用同一套函式），只是量變大。

逐日讀（for 迴圈一次一天，記憶體只進一天，避免含二次全量 405 萬列一次塞爆）：
  每天取 is_first_entry=True 的列 → 放大 lots → 累積 → 全部放大後丟給 report_first 的
  enrich + run_reports，按 threshold 分、出完整五張報表。

用法：
  python report_pool.py --ticks
  python report_pool.py --ticks -s 20260601 -e 20260608
"""
import argparse

import polars as pl

import report_first as rf

# 二次進場參與率：歷史被干預後能參與多少後續成交的估計（業內慣例先用 10%）。
PARTICIPATION = 0.1


def arg_parser():
    p = argparse.ArgumentParser(description="期現貨套利 含二次進場放大版報表")
    p.add_argument("-s", "--start_date", type=str)
    p.add_argument("-e", "--end_date", type=str)
    p.add_argument("--ticks", action="store_true",
                   help="讀 ticks 版結果(events_*_tick.csv)；圖檔名亦帶 _tick")
    return p.parse_args()


def load_amplified(start, end, ticks=True) -> pl.DataFrame:
    """全撈 + 放大：複用 report_first.load_events（含 dtype 統一、舊CSV防呆），取第一次進場列，
    把量放大 = floor(potential_lots + carry_second_lots × 參與率)。

    放大是「逐列獨立」運算（每列自己算自己，無跨日/跨列依賴）→ 不需逐日迴圈，
    一次全撈後一個 with_columns 就完成（單天版的答案逐列還原、完全等價）。
    放大只動 potential_lots；下游 enrich 用放大後的量算 部位金額/費用/淨利。
    """
    events = rf.load_events(start, end, ticks=ticks)
    if events.height == 0 or "carry_second_lots" not in events.columns:
        return pl.DataFrame()
    return events.filter(pl.col("is_first_entry")).with_columns(
        (pl.col("potential_lots")
         + pl.col("carry_second_lots").fill_null(0) * PARTICIPATION)
        .floor().cast(pl.Int64).alias("potential_lots"))


def main():
    args = arg_parser()
    ticks = args.ticks
    tag = "_tick" if ticks else ""
    # ★先跑逐日結算（含二次放大、正確口徑）：整年彙總 + 曲線圖
    rf.run_daily(amplify=True, ticks=ticks, tag=tag + "_pool",
                 start=args.start_date, end=args.end_date)

    # ── 以下為舊全撈五張報表（放大版，留作對照）──
    events = load_amplified(args.start_date, args.end_date, ticks=args.ticks)
    if events.height == 0:
        print("找不到事件 CSV（先跑 main.py --ticks 產出含二次進場的事實表）")
        return

    ver = "ticks版" if args.ticks else "分K版"
    before = events.height
    events = events.filter(pl.col("potential_lots") > 0)   # 放大後仍可能 0（錨0且加碼不足1口）
    dropped = before - events.height
    days = events["date"].unique().to_list()
    print(f"[含二次放大 參與率{PARTICIPATION:.0%}｜保守口徑｜{ver}] 第一次進場 {before} 筆，"
          f"剔除放大後仍湊不成最小單位 {dropped} 筆 → 剩 {events.height}，涵蓋 {len(days)} 天")
    # 量已放大 → enrich 用放大後 potential_lots 算 部位金額/費用/淨利（完全複用 report_first）
    events = rf.enrich(events)
    print(f"（lots 已放大＝第一筆 + 後續累積量×{PARTICIPATION:.0%}；其餘成本/淨利口徑同 report_first）")
    # 排 hold_secs==0（同 report_first）
    before_h0 = events.height
    events = events.filter(pl.col("hold_secs").is_null() | (pl.col("hold_secs") > 0))
    n_h0 = before_h0 - events.height
    if n_h0:
        print(f"（已排除 hold_secs=0 的同刻出場 {n_h0} 筆）")
    print()
    # 五張報表完全複用 report_first（量是放大版）。tag 加 _pool 區分輸出檔名。
    rf.run_reports(events, tag + "_pool")


if __name__ == "__main__":
    main()
