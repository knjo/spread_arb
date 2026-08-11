"""期現貨套利 全年分析 — 逐日獨立算、跨日疊加（各閥值獨立）。

逐日呼叫 analyze_day.analyze_one_day（同一段口徑，保證全年版可還原回單日：
  analyze_year 某天的列 == analyze_day -d 該天 的結果）。

跨日疊加口徑（★已實現可加、未實現不可加）：
  - 已實現淨利：逐日加總（每筆當天收斂只算一次、跨日不重複、不灌水）。
  - 留倉未實現帳面：是「每天收盤的帳面快照」，**跨日相加是假數字**（同一批留倉會被每天
    重複計）。故整年**不加總未實現**，只逐日呈現 + 報帳面浮動最深那天（最差快照）。

各閥值獨立、不可跨閥值相加（0.5% 的事件含後來漲到 1% 的）。

用法：
  uv run python analyze_year.py                       # 全部 out/events_*_tick.csv
  uv run python analyze_year.py -s 20260201 -e 20260228
  uv run python analyze_year.py --mode 1
"""
from __future__ import annotations

import argparse
import glob
import os

import polars as pl

from analyze_day import analyze_one_day, OUT_DIR

STATS_DIR = "out/stats"


def _dates_in_range(start: int | None, end: int | None) -> list[int]:
    files = sorted(glob.glob(os.path.join(OUT_DIR, "events_*_tick.csv")))
    dates = []
    for f in files:
        ymd = int(os.path.basename(f)[len("events_"):len("events_") + 8])
        if start and ymd < start:
            continue
        if end and ymd > end:
            continue
        dates.append(ymd)
    return dates


def run_year(start=None, end=None, mode="2") -> pl.DataFrame:
    """逐日算 → 疊成全年逐日明細（每 date×閥值 一列）。

    某天算不出來（最常見：MySQL 還沒有當日收盤/結算價，如剛收盤還沒入庫的近日）→
    跳過該天 + 印警示，不中斷整批；最後列出被跳過的天。
    """
    dates = _dates_in_range(start, end)
    if not dates:
        raise FileNotFoundError("out/ 下無 events_*_tick.csv（先跑 main.py --ticks）")
    frames, skipped = [], []
    for date in dates:
        print(f"  {date} ...", flush=True)
        try:
            frames.append(analyze_one_day(date, mode))
        except Exception as e:
            print(f"  ⚠️ {date} 跳過（{type(e).__name__}: {str(e)[:60]}）"
                  f"——多半是 MySQL 還沒有該日收盤/結算價", flush=True)
            skipped.append(date)
    if skipped:
        print(f"\n⚠️ 共跳過 {len(skipped)} 天（缺收盤/結算價，未納入彙總）: {skipped}")
    if not frames:
        raise RuntimeError("所有天都跳過了——檢查 MySQL 收盤/結算價是否有資料")
    return pl.concat(frames)


def yearly_summary(daily: pl.DataFrame) -> pl.DataFrame:
    """整年彙總（每閥值一列）。已實現加總；未實現取最深快照（不加總）。"""
    return (daily.group_by("閥值").agg([
        pl.col("總筆數").sum().alias("整年總筆數"),
        pl.col("收斂筆數").sum().alias("整年收斂筆數"),
        pl.col("已實現淨利").sum().alias("整年已實現淨利"),          # ★可加
        pl.col("留倉未實現帳面").min().alias("留倉帳面_最深快照"),     # 最負那天；不可加總
        pl.col("留倉缺價筆數").sum().alias("整年留倉缺價筆數"),
    ]).with_columns(
        (pl.col("整年收斂筆數") / pl.col("整年總筆數") * 100).round(1).alias("收斂率%")
    ).sort("閥值"))


def main():
    p = argparse.ArgumentParser(description="期現貨套利 全年分析")
    p.add_argument("-s", "--start_date", type=int, default=None)
    p.add_argument("-e", "--end_date", type=int, default=None)
    p.add_argument("--mode", type=str, default="2", choices=["1", "2"])
    args = p.parse_args()

    print(f"== 全年分析（mode {args.mode}）==")
    daily = run_year(args.start_date, args.end_date, args.mode)
    os.makedirs(STATS_DIR, exist_ok=True)
    daily_path = os.path.join(STATS_DIR, "analyze_daily.csv")
    daily.write_csv(daily_path)

    n_days = daily["date"].n_unique()
    print(f"\n涵蓋 {n_days} 天；逐日明細已存：{daily_path}\n")
    print("===== 整年彙總（各閥值獨立、勿相加）=====")
    print("【整年已實現淨利＝逐日加總(可加)】｜【留倉帳面＝每日快照、取最深那天(不可加總)】")
    print("⚠️ 已實現與留倉帳面口徑不同，不可加總、分開看。\n")
    summ = yearly_summary(daily)
    with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
        print(summ.with_columns([
            (pl.col("整年已實現淨利") / 1e8).round(3).alias("整年已實現淨利_億"),
            (pl.col("留倉帳面_最深快照") / 1e8).round(3).alias("留倉帳面最深_億"),
        ]).select("閥值", "整年總筆數", "整年收斂筆數", "收斂率%",
                  "整年已實現淨利_億", "留倉帳面最深_億", "整年留倉缺價筆數"))


if __name__ == "__main__":
    main()
