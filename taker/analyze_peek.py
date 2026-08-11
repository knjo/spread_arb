"""分析 peek_{date}.csv（純讀 peek raw，不碰 events/不撈 NAS）。逐日，跟 main.py 同款 -s/-e。

peek 每列＝一筆 raw tick + 標註(event_id/anchor_type/leg/anchor_code/anchor_chseq/指標)。
一個「錨點」＝ (event_id, anchor_type)。五種分析顆粒度：①②③④ 每錨點、⑤ 每出場點。

輸出（分資料夾）：
  主表  out/peek/anchor/analyze_{date}.csv     —— 每錨點一列：
    ① 抖動    ：窗口內 A1/B1 變動次數、報價 tick 總數（不過濾，留原始；匯總再定門檻）
    ② 該腳撐多久：錨點那格 taker 價(進場期=fut_bid/現=spot_ask；出場期=fut_ask/現=spot_bid)
                  往後到第一次該格價變動的秒數＝這腳那個價站多久（進出場都算，單腳）
    ③ 被成交  ：窗口內有無成交、第一筆成交距錨點秒數、成交量
    ④ 五檔厚度/檔差：錨點那筆(成交筆取前一筆報價)的五檔 lots + 各檔差是否>1 tick(分級)
  ⑤表  out/peek/shared_exit/shared_exit_{date}.csv —— 每出場點一列：
    共用事件數、需求口數(potential_lots 加總)、出場那刻對手掛量、覆蓋率。

用法：uv run python analyze_peek.py -s 20260623     單天
      uv run python analyze_peek.py -s 20260601 -e 20260618  區間逐日
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import polars as pl

PEEK_DIR = "out/peek"
ANCHOR_DIR = "out/peek/anchor"
SHARED_DIR = "out/peek/shared_exit"

# 錨點那格 taker 要吃的價（這腳「那個價站多久」看它）
TAKER_PX = {"entry_fut": "BidPrice1", "entry_spot": "AskPrice1",
            "exit_fut": "AskPrice1", "exit_spot": "BidPrice1"}


def tick_size(px: float) -> float:
    """台股分級跳動值（期現同表套用，價已還原）。"""
    if px < 10:    return 0.01
    if px < 50:    return 0.05
    if px < 100:   return 0.1
    if px < 500:   return 0.5
    if px < 1000:  return 1.0
    return 5.0


def per_anchor(df: pl.DataFrame) -> pl.DataFrame:
    """每錨點一列：① 抖動、③ 被成交、④ 五檔/檔差、② 該腳撐多久。"""
    keys = ["event_id", "anchor_type", "leg", "anchor_code", "anchor_chseq"]
    tags = [c for c in ("exit_stretch_secs", "first_stretch_secs", "first_ret",
                        "converge_time", "potential_lots") if c in df.columns]
    rows = []
    # 逐錨點處理（顆粒度小、數量千級可接受；要看 tick 序列故不能純 group agg）
    for (eid, at, leg, code, aseq), g in df.group_by(keys, maintain_order=True):
        g = g.sort("RecvTime")
        quotes = g.filter(pl.col("is_quote"))
        fills = g.filter(pl.col("is_fill"))
        # 錨點那筆（ChannelSeq==anchor_chseq）；若是成交筆無五檔→取它之前最後一筆報價
        at_idx = g.with_row_index().filter(pl.col("ChannelSeq") == aseq)["index"]
        anchor_t = None
        book = None
        if at_idx.len():
            i = at_idx[0]
            anchor_t = g["RecvTime"][i]
            row_i = g[i]
            if row_i["is_quote"][0]:
                book = row_i
            else:  # 成交筆 → 取 i 之前最後一筆報價
                prev = g[:i].filter(pl.col("is_quote"))
                book = prev.tail(1) if prev.height else None

        r = {"event_id": eid, "anchor_type": at, "leg": leg,
             "anchor_code": code, "anchor_chseq": aseq, "n_tick": g.height,
             "n_quote": quotes.height}
        for t in tags:
            r[t] = g[t][0]

        # ① 抖動：A1/B1 逐 tick 跟前一筆比，不同就算一次變動（不管上下、不管有沒有回到舊價；
        #    走 2400→2405→2400 算 2 次。不是 n_unique「出現幾種價」）。
        if quotes.height:
            qs = quotes.sort("RecvTime")
            r["ask1_changes"] = int((qs["AskPrice1"] != qs["AskPrice1"].shift(1)).sum()) - 1
            r["bid1_changes"] = int((qs["BidPrice1"] != qs["BidPrice1"].shift(1)).sum()) - 1
        else:
            r["ask1_changes"] = r["bid1_changes"] = None

        # ③ 被成交：分兩種——
        #   (a) 任何價成交：窗口內市場活不活躍（n_fill/fill_lots，保留）。
        #   (b) 成交在「錨點那格 taker 價」：你的出場/進場價被觸發/被吃（at_px_*）。
        #       錨點價＝該腳 taker 那格(進期=bid/進現=ask/出期=ask/出現=bid)的實際價，從 book 取。
        r["n_fill"] = fills.height
        r["fill_lots"] = int(fills["FillLots"].sum()) if fills.height else 0
        if fills.height and anchor_t is not None:
            d = (fills["RecvTime"] - anchor_t).dt.total_seconds(fractional=True)
            after = d.filter(d >= 0)
            r["first_fill_after_s"] = float(after.min()) if after.len() else None
        else:
            r["first_fill_after_s"] = None
        # (b) 成交在錨點價那格
        tpx_col = TAKER_PX.get(at)
        anchor_px = book[tpx_col][0] if (book is not None and book.height and tpx_col) else None
        if anchor_px is not None and fills.height and anchor_t is not None:
            at_fills = fills.filter(pl.col("FillPrice") == anchor_px)
            r["at_px_n_fill"] = at_fills.height
            r["at_px_lots"] = int(at_fills["FillLots"].sum()) if at_fills.height else 0
            if at_fills.height:
                d2 = (at_fills["RecvTime"] - anchor_t).dt.total_seconds(fractional=True)
                aft2 = d2.filter(d2 >= 0)
                r["at_px_first_fill_s"] = float(aft2.min()) if aft2.len() else None
            else:
                r["at_px_first_fill_s"] = None
        else:
            r["at_px_n_fill"] = None
            r["at_px_lots"] = None
            r["at_px_first_fill_s"] = None

        # ④ 五檔厚度 + 各檔差是否 >1 tick（錨點那筆 book）
        if book is not None and book.height:
            for sd, pcol, lcol in [("ask", "AskPrice", "AskLots"), ("bid", "BidPrice", "BidLots")]:
                lots = [book[f"{lcol}{k}"][0] for k in range(1, 6)]
                pxs = [book[f"{pcol}{k}"][0] for k in range(1, 6)]
                r[f"{sd}_depth5"] = int(sum(x for x in lots if x))      # 五檔總厚度
                for k in range(1, 6):
                    r[f"{sd}_lots{k}"] = lots[k - 1]
                # 相鄰檔差是否 >1 tick（價>0 才比）
                gaps_over = 0
                for k in range(4):
                    p0, p1 = pxs[k], pxs[k + 1]
                    if p0 and p1:
                        ts = tick_size(p0)
                        if abs(p1 - p0) > ts * 1.5:   # >1 tick（含浮點容差）
                            gaps_over += 1
                r[f"{sd}_gaps_over1tick"] = gaps_over
        else:
            for sd in ("ask", "bid"):
                r[f"{sd}_depth5"] = None
                r[f"{sd}_gaps_over1tick"] = None

        # ② 那格價「可成交」撐多久：錨點那格 taker 價，往後到價『往不利方向越過錨點價』為止。
        #   = 還買/賣得到一樣好或更好的價(含等於錨點價)撐多久。價往有利方向變不算消失。
        #   消失條件依該腳吃的格：吃 Bid(賣期/賣現)→ bid < 錨點價才消失；
        #                        吃 Ask(買現/買回期)→ ask > 錨點價才消失。
        #   ⚠ 用 ChannelSeq > 錨點 seq 框「往後」(同 RecvTime 多筆，用時間>=會誤判，DQFG6 踩過)。
        tpx = TAKER_PX.get(at)
        if anchor_t is not None and tpx and book is not None and book.height:
            base_px = book[tpx][0]
            # tpx 是 BidPrice1 → 吃 bid，跌破(<base)才消失；AskPrice1 → 吃 ask，漲過(>base)才消失
            gone = (pl.col(tpx) < base_px) if tpx == "BidPrice1" else (pl.col(tpx) > base_px)
            fut_q = quotes.filter((pl.col("ChannelSeq") > aseq)
                                  & (pl.col(tpx) > 0) & gone).sort("ChannelSeq")
            if fut_q.height:
                r["leg_hold_s"] = float((fut_q["RecvTime"][0] - anchor_t).total_seconds())
            else:
                r["leg_hold_s"] = None   # 整段都還可成交（沒往不利方向越過）
        else:
            r["leg_hold_s"] = None

        rows.append(r)
    return pl.DataFrame(rows)


def shared_exit(df: pl.DataFrame) -> pl.DataFrame:
    """⑤ 每出場點(anchor_code, anchor_chseq)：共用事件數、需求口數、對手掛量、覆蓋率。"""
    ex = df.filter(pl.col("anchor_type").str.starts_with("exit"))
    if ex.height == 0:
        return pl.DataFrame()
    has_lots = "potential_lots" in ex.columns
    # 先到「每事件一列」(potential_lots 去重，不被 raw 列數灌)
    ev_keys = ["anchor_code", "anchor_chseq", "anchor_type", "event_id"]
    sel = [pl.col("potential_lots").first().alias("plots")] if has_lots else []
    ev1 = ex.group_by(ev_keys).agg(sel) if sel else ex.select(ev_keys).unique()
    g_keys = ["anchor_code", "anchor_chseq", "anchor_type"]
    agg = [pl.col("event_id").n_unique().alias("n_events")]
    if has_lots:
        agg.append(pl.col("plots").sum().alias("need_lots"))
    out = ev1.group_by(g_keys).agg(agg)
    # 出場那刻對手掛量：raw 裡 ChannelSeq==anchor_chseq 該筆；exit_fut 看 AskLots1、exit_spot 看 BidLots1
    anchor_row = (df.filter(pl.col("ChannelSeq") == pl.col("anchor_chseq"))
                    .group_by(g_keys)
                    .agg(pl.col("AskLots1").first().alias("_ask1"),
                         pl.col("BidLots1").first().alias("_bid1")))
    out = out.join(anchor_row, on=g_keys, how="left").with_columns(
        pl.when(pl.col("anchor_type") == "exit_fut").then(pl.col("_ask1"))
          .otherwise(pl.col("_bid1")).alias("avail_lots"))
    if has_lots:
        out = out.with_columns((pl.col("avail_lots") / pl.col("need_lots")).alias("cover_lots"))
    return out.sort(g_keys)


def run_day(date: int) -> None:
    path = os.path.join(PEEK_DIR, f"peek_{date}.csv")
    if not os.path.exists(path):
        print(f"  {date}: 找不到 {path}，跳過"); return
    df = pl.read_csv(path, infer_schema_length=20000)
    df = df.with_columns(pl.col("RecvTime").str.to_datetime(strict=False))
    print(f"== analyze_peek {date} | peek {df.height:,} 列 ==")

    anch = per_anchor(df)
    os.makedirs(ANCHOR_DIR, exist_ok=True)
    p1 = os.path.join(ANCHOR_DIR, f"analyze_{date}.csv")
    anch.write_csv(p1)
    print(f"  主表 {anch.height} 錨點 → {p1}")

    sh = shared_exit(df)
    if sh.height:
        os.makedirs(SHARED_DIR, exist_ok=True)
        p2 = os.path.join(SHARED_DIR, f"shared_exit_{date}.csv")
        sh.write_csv(p2)
        multi = sh.filter(pl.col("n_events") > 1).height
        print(f"  ⑤表 {sh.height} 出場點(多人共用 {multi}) → {p2}")


def main():
    import argparse
    from datetime import datetime, timedelta
    p = argparse.ArgumentParser(description="分析 peek raw（逐日）")
    p.add_argument("-s", "--start_date", type=str, required=True)
    p.add_argument("-e", "--end_date", type=str)
    args = p.parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start
    cal = start
    while cal <= end:
        run_day(int(cal.strftime("%Y%m%d")))
        cal += timedelta(days=1)


if __name__ == "__main__":
    main()
