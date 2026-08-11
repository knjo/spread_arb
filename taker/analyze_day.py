"""期現貨套利 單日分析（乾淨重寫）— 只算第一次進場(is_first_entry)、逐閥值彙總。

取樣口徑（與舊 report_first/stats 的差異＝本次重寫的重點）：
  - 母體只取 is_first_entry=true（NON_FIRST 是二次進場明細、不分析）。
  - 不用 event_id 當鍵（每天重編、會重複，不可跨列 group）。
  - 流程順序：讀檔 → 濾 first → 濾 potential_lots>0 → enrich → 彙總。
    （舊版 enrich 在濾 first 之前跑，會被 NON_FIRST 殘缺四價污染。）

損益口徑（★已實現與未實現口徑不同，絕對不可加總、永遠分開看，故無「合計」欄）：
  - 已實現（converged=true）：當天盤中達 B 反向收斂、taker 平倉，是**扣費後淨利**。
      出場價差 exit_ret = (exit_spot_bid − exit_fut_ask)/exit_fut_ask（cost.py 推）。
      已扣稅費（期交稅 + 手續費 + 證交稅當沖 0.15%）。
  - 留倉（converged=false）：用 **A 帳面口徑**（券商慣例的浮動損益）：
      非結算日 → 未實現 = (當日收盤/結算價 − 進場價) × 部位，**純市值浮動**：
        期腿(賣) = (entry_fut_bid − fut_settle)/entry_fut_bid
        現腿(買) = (spot_close  − entry_spot_ask)/entry_spot_ask
        未實現   = (期腿率 + 現腿率) × 期貨腿名目(口×乘數×進場賣期價)
        ★A 帳面：**不扣稅費、不吃出場 bid/ask 價差**（回答「現在帳上值多少」，非「平掉實拿」）。
        收盤價單一值（不分 bid/ask）：期貨用 settlement_price、現貨用 close_price。
      結算日   → 留倉一律當作會收斂（到期強制結算），不計留倉未實現。
  - 收盤/結算價缺 → 該筆 mark 不了，顯式計數、不靜默當 0。

費用/本金的單一真相在 spread_arb/cost.py、capital.py（與產出層共用，不在此重造）。

用法：
  uv run python analyze_day.py -d 20260608
  uv run python analyze_day.py -d 20260608 --mode 1   # 樂觀本金（預設 2 保守）
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

import polars as pl

from mysql import StrategyMySQLLoader
from spread_arb.spread import SIDE_SELL
from spread_arb.cost import net_return
from spread_arb.capital import add_capital

OUT_DIR = "out"
SIDE = SIDE_SELL   # 這版只跑價差賣（價差買整年不具效益，產出層已源頭排除）


def settlement_day(date: dt.date) -> dt.date:
    """當月結算日 = 第三個星期三（不處理非交易日順延；單日驗證夠用）。"""
    weds = [dt.date(date.year, date.month, d)
            for d in range(1, 32)
            if d <= 28 + 3 and _valid(date.year, date.month, d)
            and dt.date(date.year, date.month, d).weekday() == 2]
    return weds[2]


def _valid(y: int, m: int, d: int) -> bool:
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def load_first_entries(date: int) -> pl.DataFrame:
    """讀單日事實 CSV，濾出 first entry & potential_lots>0（分析母體）。"""
    path = os.path.join(OUT_DIR, f"events_{date}_tick.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到事實檔：{path}（先跑 main.py -s {date} --ticks）")
    df = pl.read_csv(path, infer_schema_length=20000)
    # 某些欄某天全空→read_csv 推成 String，後續算術會炸：數值欄統一 Float64（時間/字串欄留著）。
    casts = [pl.col(c).cast(pl.Float64, strict=False)
             for c, t in df.schema.items()
             if t == pl.String and c not in ("QuoteCode", "ValueCode", "side")
             and not c.endswith("_time")]
    if casts:
        df = df.with_columns(casts)
    n_all = df.height
    df = df.filter(pl.col("is_first_entry") & (pl.col("side") == SIDE)
                   & (pl.col("potential_lots") > 0))
    print(f"讀入 {n_all:,} 列 → first entry & {SIDE} & potential_lots>0：{df.height:,} 筆")
    return df


def enrich(events: pl.DataFrame, mode: str) -> pl.DataFrame:
    """事實 → 衍生：exit_ret / 部位金額 / 費用 / 淨利 / 本金 / ROI（複用 cost、capital）。"""
    # 出場反邊 taker 價差（價差賣＝賣現買期）；留倉(出場四價 null)→ null。
    exit_ret = (pl.col("exit_spot_bid") - pl.col("exit_fut_ask")) / pl.col("exit_fut_ask")
    events = events.with_columns(
        pl.when(pl.col("exit_fut_ask").is_null()).then(None)
          .otherwise(exit_ret).alias("exit_ret"),
        # 部位金額 = 口數 × 逐合約乘數 × 進場賣期成交價
        (pl.col("potential_lots") * pl.col("contract_size") * pl.col("entry_fut_bid"))
            .alias("potential_value"),
    )
    events = net_return(events, SIDE, mode=mode)        # fee_rate/net_ret/potential_pnl...
    events = add_capital(events, SIDE, mode=mode)       # capital/roi/roi_annual
    return events


def mark_overnight(inv: pl.DataFrame, date: int) -> pl.DataFrame:
    """留倉(converged=false)用 MySQL 當日收盤/結算價 mark 未實現損益。

    收盤價單一值（不分 bid/ask）、已還原（不再 scale）。任一腿缺價 → 該筆 mark 不了（null）。
    """
    m = StrategyMySQLLoader()
    fut = pl.from_pandas(m.get_futures_settle_price(date)).rename(
        {"quote_code": "QuoteCode", "settlement_price": "fut_settle"})
    spot = pl.from_pandas(m.get_stock_closing_price(date)).rename(
        {"quote_code": "ValueCode", "close_price": "spot_close"})
    # 現貨表 quote_code 是 4 碼股票代號(字串) → 對 CSV 的 ValueCode(int)；統一型別再 join。
    spot = spot.with_columns(pl.col("ValueCode").cast(pl.Int64, strict=False))
    fut = fut.with_columns(pl.col("fut_settle").cast(pl.Float64, strict=False))
    spot = spot.with_columns(pl.col("spot_close").cast(pl.Float64, strict=False))

    inv = inv.join(fut, on="QuoteCode", how="left").join(spot, on="ValueCode", how="left")
    fut_notional = pl.col("potential_lots") * pl.col("contract_size") * pl.col("entry_fut_bid")
    inv = inv.with_columns([
        # 期腿(賣)：賣 entry_fut_bid、買回 fut_settle；跌賺漲賠。
        (((pl.col("entry_fut_bid") - pl.col("fut_settle")) / pl.col("entry_fut_bid"))
         * fut_notional).alias("_fut_unreal"),
        # 現腿(買)：買 entry_spot_ask、賣 spot_close；漲賺跌賠。
        (((pl.col("spot_close") - pl.col("entry_spot_ask")) / pl.col("entry_spot_ask"))
         * fut_notional).alias("_spot_unreal"),
    ]).with_columns(
        (pl.col("_fut_unreal") + pl.col("_spot_unreal")).alias("unreal_pnl"))
    return inv


def analyze_one_day(date: int, mode: str = "2") -> pl.DataFrame:
    """算單日 → 回逐閥值 DataFrame（全年版複用此函式，保證口徑一致、可還原回單日）。

    回欄：date / 閥值 / 總筆數 / 收斂筆數 / 留倉筆數 / 留倉可mark筆數 / 留倉缺價筆數 /
          已實現淨利 / 留倉未實現帳面 / is_settle。
    ⚠️ 已實現＝扣費後淨利；留倉未實現＝A 帳面浮動(不扣費)。兩者不可加總。
    """
    d = dt.datetime.strptime(str(date), "%Y%m%d").date()
    is_settle = (d == settlement_day(d))

    events = enrich(load_first_entries(date), mode)
    # 結算日：留倉全當收斂（到期強制結算）→ 視為已平掉。
    if is_settle:
        events = events.with_columns(pl.lit(True).alias("converged"))

    conv = events.filter(pl.col("converged"))
    inv = events.filter(~pl.col("converged"))
    if inv.height:
        inv = mark_overnight(inv, date)
        markable = inv.filter(pl.col("unreal_pnl").is_not_null())
    else:
        markable = inv

    rows = []
    for thr in sorted(events["threshold"].unique().to_list()):
        c = conv.filter(pl.col("threshold") == thr)
        iv = inv.filter(pl.col("threshold") == thr) if inv.height else inv
        mk = markable.filter(pl.col("threshold") == thr) if markable.height else markable
        rows.append({
            "date": date,
            "閥值": thr,
            "總筆數": events.filter(pl.col("threshold") == thr).height,
            "收斂筆數": c.height,
            "留倉筆數": iv.height if inv.height else 0,
            "留倉可mark筆數": mk.height if markable.height else 0,
            "留倉缺價筆數": (iv.height - mk.height) if inv.height else 0,
            "已實現淨利": float(c["potential_pnl"].sum()) if c.height else 0.0,
            "留倉未實現帳面": float(mk["unreal_pnl"].sum()) if (markable.height and mk.height) else 0.0,
            "is_settle": is_settle,
        })
    return pl.DataFrame(rows)


def _print_day_table(summ: pl.DataFrame) -> None:
    print("【已實現＝扣費後淨利】｜【留倉未實現＝A 帳面浮動，未扣稅費/未吃出場價差】")
    print("⚠️ 兩欄口徑不同，不可加總、請分開看。\n")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_rows=-1):
        print(summ.with_columns([
            (pl.col("收斂筆數") / pl.col("總筆數") * 100).round(1).alias("收斂率%"),
            (pl.col("已實現淨利") / 1e4).round(1).alias("已實現淨利_萬"),
            (pl.col("留倉未實現帳面") / 1e4).round(1).alias("留倉未實現帳面_萬"),
        ]).select("閥值", "總筆數", "收斂筆數", "收斂率%", "留倉可mark筆數",
                  "留倉缺價筆數", "已實現淨利_萬", "留倉未實現帳面_萬"))


def main():
    p = argparse.ArgumentParser(description="期現貨套利 單日分析")
    p.add_argument("-d", "--date", type=int, required=True, help="資料日 yyyymmdd")
    p.add_argument("--mode", type=str, default="2", choices=["1", "2"],
                   help="本金口徑：1 樂觀(融資融券) / 2 保守(全額，預設)")
    args = p.parse_args()

    d = dt.datetime.strptime(str(args.date), "%Y%m%d").date()
    is_settle = (d == settlement_day(d))
    print(f"== 單日分析 {args.date}（mode {args.mode}）==")
    print(f"結算日 {settlement_day(d).strftime('%Y%m%d')}；"
          f"今日{'＝結算日（留倉全當收斂）' if is_settle else '非結算日（留倉用收盤價 mark）'}\n")
    _print_day_table(analyze_one_day(args.date, args.mode))


if __name__ == "__main__":
    main()
