"""結帳層（每日當日彙總）— 逐日浮虧/浮盈評價，回答「實際至少要準備多少錢」。

兩步驟分析的第一步（重、迴圈、丟明細）：
  每天 call 一次 settle_day(date)：讀「那一天的 fill csv」(庫存滾動後＝當天所有在場部位)，
  用「該合約當天最後一筆 tick 四價(close_*)」把每個在場部位 mark-to-market（逐日重評價），
  算當天的浮虧/浮盈彙總（總浮虧、單筆最大浮虧、在場部位數…），**只留彙總、丟明細**
  → 記憶體不爆。整年 88 天各一列彙總 → 第二步(總分析)讀這張小表算 MAE/MFE/準備多少。

評價口徑（與 exit_ret 同口徑、taker）：
  - 收盤價是「該合約當天最後一筆 tick」的四價（每合約自己的，非全市場收盤時刻）。
  - 留倉部位若「今天 taker 平倉」：價差賣＝買回期(吃 close_fut_ask)+賣現(吃 close_spot_bid)。
    平倉那刀反邊價差 = (close_spot_bid − close_fut_ask) / close_fut_ask（通常負＝倒貼）。
  - 當日 mark 毛利率 = 進場 first_ret + 平倉反邊價差。負＝浮虧、正＝浮盈。
  - 浮虧/浮盈金額 = mark 毛利率 × 部位金額(potential_value)。

注意：
  - 收盤價只是「估今天帳上虧多少」的評價尺，**不是平倉點**（真平倉是盤中 ret_buy 收斂事件）。
    收盤打不打得到無關——這是 mark-to-market 評價，不要求成交。
  - close 任一腿無報價(null/0)→ 該筆當天評價不了，顯式計數、不靜默當 0。
  - 只看 potential_lots>0、且「今天還沒平掉(converged=false)」的在場部位。
    （converged=true 是今天盤中真平掉的，已實現損益、不算未實現浮虧。出場固定走 B 反向收斂。）

用法：
  python settle.py --ticks                    # 結帳 out/ 下所有 events_*.csv，吐 out/stats/settle_daily.csv
  python settle.py --ticks -s 20260601 -e 20260603
"""
import argparse
import glob
import os

import polars as pl

from spread_arb.spread import SIDE_SELL

OUT_DIR = "out"
STATS_DIR = "out/stats"


def settle_day(df: pl.DataFrame) -> dict | None:
    """單日結帳：df = 某天的 fill（已過濾 B / potential_lots>0）。回該日浮虧彙總 dict。

    只評價「今天還沒平掉」的在場部位（converged=false）＝未實現浮虧。
    回 None 表示當天無在場可評價部位。
    """
    # 只看第一次進場未收斂的留倉部位（最低估算；二次進場/放大不在此）。
    first_col = pl.col("is_first_entry") if "is_first_entry" in df.columns else pl.lit(True)
    inv = df.filter(~pl.col("converged") & first_col)   # 今天還抱著的（未實現）
    n_pos = inv.height
    if n_pos == 0:
        return None
    need = ["close_fut_ask", "close_spot_bid", "first_ret",
            "potential_lots", "contract_size", "entry_fut_bid"]
    if any(c not in inv.columns for c in need):
        return None
    # 部位金額（fill 不存、settle 自算，與 enrich 同口徑：口數×乘數×進場賣期價）
    inv = inv.with_columns(
        (pl.col("potential_lots") * pl.col("contract_size") * pl.col("entry_fut_bid"))
        .alias("potential_value"))
    # close 任一腿無效 → 評價不了，排除並計數（保證金看期貨腿，故只需 fut 腿；但價差浮動需兩腿）
    valid = inv.filter((pl.col("close_fut_ask") > 0) & (pl.col("close_spot_bid") > 0)
                       & (pl.col("entry_fut_bid") > 0) & (pl.col("entry_spot_ask") > 0))
    n_unpriced = n_pos - valid.height
    if valid.height == 0:
        return {"在場部位": n_pos, "可評價": 0, "評價不了": n_unpriced}
    # 期貨腿名目 = 口數×乘數×進場賣期價；保證金追繳就看這腿。
    fut_notional = pl.col("potential_lots") * pl.col("contract_size") * pl.col("entry_fut_bid")
    valid = valid.with_columns([
        # 【期貨腿浮虧】價差賣＝進場賣期(entry_fut_bid)、今平買回(close_fut_ask)。期漲→賣後虧→補保證金。
        #   浮虧率 = (賣出價 − 買回價)/賣出價；金額 = 率 × 期貨腿名目。**這是要補保證金的口徑。**
        (((pl.col("entry_fut_bid") - pl.col("close_fut_ask")) / pl.col("entry_fut_bid"))
         * fut_notional).alias("_fut_pnl"),
        # 【現貨腿浮虧】價差賣＝進場買現(entry_spot_ask)、今平賣現(close_spot_bid)。現股、不斷頭(帳面)。
        #   浮虧率 = (賣出價close_spot_bid − 買進價entry_spot_ask)/買進價；金額 × 現貨腿名目(≈同口徑用期名目近似)。
        (((pl.col("close_spot_bid") - pl.col("entry_spot_ask")) / pl.col("entry_spot_ask"))
         * fut_notional).alias("_spot_pnl"),
    ])
    valid = valid.with_columns([
        (pl.col("_fut_pnl") + pl.col("_spot_pnl")).alias("_pnl"),   # 合計價差浮動
        fut_notional.alias("_fut_notional"),
    ])
    fut, spot, tot = valid["_fut_pnl"], valid["_spot_pnl"], valid["_pnl"]
    fut_los = valid.filter(pl.col("_fut_pnl") < 0)
    tot_los = valid.filter(pl.col("_pnl") < 0)
    return {
        "在場部位": n_pos,
        "可評價": valid.height,
        "評價不了": n_unpriced,
        # ── 期貨腿（要補保證金看這個）──
        "期貨腿浮虧筆數": fut_los.height,
        "期貨腿總浮虧": float(fut_los["_fut_pnl"].sum()) if fut_los.height else 0.0,  # ★要補保證金
        "期貨腿單筆最大浮虧": float(fut.min()),
        # ── 現貨腿（現股帳面、不斷頭）──
        "現貨腿淨浮動": float(spot.sum()),
        # ── 合計價差浮動（兩腿對沖後）──
        "合計總浮虧": float(tot_los["_pnl"].sum()) if tot_los.height else 0.0,
        "合計淨浮動": float(tot.sum()),
        "合計單筆最大浮虧": float(tot.min()),
        "期貨腿名目": float(valid["_fut_notional"].sum()),
    }


def main():
    p = argparse.ArgumentParser(description="結帳層：逐日浮虧彙總")
    p.add_argument("-s", "--start_date", type=str)
    p.add_argument("-e", "--end_date", type=str)
    p.add_argument("--ticks", action="store_true")
    args = p.parse_args()

    suffix = "_tick" if args.ticks else ""
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"events_*{suffix}.csv")))
    rows = []
    for f in files:
        name = os.path.basename(f)
        if not args.ticks and name.endswith("_tick.csv"):
            continue
        ymd = name[len("events_"):len("events_") + 8]
        if args.start_date and ymd < args.start_date:
            continue
        if args.end_date and ymd > args.end_date:
            continue
        # 逐日讀（記憶體只進一天，算完丟明細）
        df = pl.read_csv(f)
        # 某天 close/exit 欄全空 → read_csv 推成 String，數值比較會炸。數值欄統一 Float64
        #   （非真字串、非時間欄；時間欄不參與此處運算，留著）。
        num_cols = [c for c in df.columns if df.schema[c] == pl.String
                    and c not in ("QuoteCode", "ValueCode", "side")
                    and not c.endswith("_time")]
        if num_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols])
        df = df.filter((pl.col("side") == SIDE_SELL)
                       & (pl.col("potential_lots") > 0))
        s = settle_day(df)
        if s is None:
            continue
        s = {"date": ymd, **s}
        rows.append(s)

    if not rows:
        print("（無可結帳資料：先跑 main.py --ticks --inventory 產含庫存的 fill）")
        return
    daily = pl.DataFrame(rows)
    os.makedirs(STATS_DIR, exist_ok=True)
    out_path = os.path.join(STATS_DIR, "settle_daily.csv")
    daily.write_csv(out_path)
    print("（★期貨腿總浮虧＝要補的保證金口徑；現貨腿是現股帳面不斷頭；合計＝兩腿對沖後價差浮動）")
    with pl.Config(tbl_cols=-1, tbl_width_chars=260, tbl_rows=-1):
        print(daily.with_columns([
            (pl.col("期貨腿總浮虧") / 1e8).round(3).alias("期貨腿浮虧_億★"),
            (pl.col("期貨腿單筆最大浮虧") / 1e4).round(1).alias("期單筆最大_萬"),
            (pl.col("現貨腿淨浮動") / 1e8).round(3).alias("現貨腿淨_億"),
            (pl.col("合計淨浮動") / 1e8).round(3).alias("合計淨_億"),
            (pl.col("期貨腿名目") / 1e8).round(1).alias("期名目_億"),
        ]).select("date", "在場部位", "可評價", "評價不了",
                  "期貨腿浮虧_億★", "期單筆最大_萬", "現貨腿淨_億", "合計淨_億", "期名目_億"))
    print(f"\n→ 每日結帳彙總已存：{out_path}")


if __name__ == "__main__":
    main()
