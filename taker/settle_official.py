"""官方結算價試算（STF 標的證券為股票）— 從現貨成交 ticks 推最後結算價。

跟套利管線（main.py / settle.py）完全獨立，不共用前處理（那邊 13:20 截止、用 RecvTime，
此處需要算到 13:25、且依官方須用交易所壓的 TransTime）。

口徑（第一版，使用者拍板 2026/6/16）：
  - 取樣時點：自打 5 秒網格，12:30:05 起 → 13:25:00 止（含兩端，共 660 點）。
    （官方真正用「指數揭示時點」，但我們沒有指數揭示序列，用 5 秒網格近似——
     證交所揭示頻率約 5 秒/次，故網格貼近官方取樣。）
  - 時間軸：TransTime（交易所撮合時間，已是台北 naive，不轉時區）。
    官方結算價是交易所用自己壓的成交時刻算的 → 必須用 TransTime，不可用 RecvTime。
  - 取價：每格點 t，找 TransTime <= t 的最近一筆成交價 FillPrice（as-of backward）。
    成交列 = FillPrice>0 & FillLots>0；價格 ÷10000 還原。
  - 平均：簡單算術平均（不加權量）→ 四捨五入到小數第 2 位。

用法：
  uv run python settle_official.py --code 2303 --date 20260608
  uv run python settle_official.py --code 2303 --date 20260608 --official 118.50   # 帶官價直接對帳
"""
import argparse
import sys
from datetime import datetime, timedelta

import polars as pl

from data_paths import scan_spot_ticks

SPOT_SCALE = 10000          # 現貨價 ÷10000 還原
GRID_STEP_SECS = 5          # 5 秒網格
# 取樣窗：12:30(不含) ~ 13:25(含)。網格 12:30:05 起、13:25:00 止。
GRID_START = (12, 30, 5)
GRID_END = (13, 25, 0)
ROUND_DP = 2                # 四捨五入到小數第 2 位


def _build_grid(date: datetime) -> pl.Series:
    """造 5 秒取樣網格（datetime[us]，naive 台北），含起訖兩端。"""
    start = date.replace(hour=GRID_START[0], minute=GRID_START[1], second=GRID_START[2], microsecond=0)
    end = date.replace(hour=GRID_END[0], minute=GRID_END[1], second=GRID_END[2], microsecond=0)
    pts = []
    t = start
    while t <= end:
        pts.append(t)
        t += timedelta(seconds=GRID_STEP_SECS)
    return pl.Series("grid_t", pts, dtype=pl.Datetime("us"))


def load_spot_fills(tw, date, code) -> pl.DataFrame:
    """撈現貨成交 ticks（不經套利管線 _clean）：只留正式撮合的成交列，TransTime 直接用。

    回 [TransTime, fill_price]（已 ÷10000 還原、依 TransTime 排序）。
    """
    raw = scan_spot_ticks(str(date), [code]).collect()
    if raw.height == 0:
        return pl.DataFrame(schema={"TransTime": pl.Datetime("us"), "fill_price": pl.Float64})
    df = raw
    # SSD2 現貨檔價格已是真實價（float）；只有舊 SDK 整數價才需 ÷SPOT_SCALE
    scale = 1 if df.schema["FillPrice"].is_float() else SPOT_SCALE
    # 只留正式撮合（TrialMatch==0），緩搓不成交、不能納入結算取樣
    if "TrialMatch" in df.columns:
        df = df.filter(pl.col("TrialMatch") == 0)
    # 成交列：FillPrice>0 & FillLots>0
    df = df.filter((pl.col("FillPrice") > 0) & (pl.col("FillLots") > 0))
    df = df.select(
        pl.col("TransTime"),
        (pl.col("FillPrice") / scale).alias("fill_price"),
        pl.col("FillLots").alias("fill_lots"),
    ).sort("TransTime")
    return df


def settle_price(fills: pl.DataFrame, date: datetime) -> dict:
    """用 5 秒網格 as-of backward 取每格點最近一筆成交價，簡平 → 結算價。"""
    grid = _build_grid(date).to_frame()
    if fills.height == 0:
        return {"結算價": None, "取樣點數": 0, "有對到價的點": 0, "說明": "當日無成交"}
    # 每格點 t 找 TransTime <= t 的最近一筆成交價（backward as-of）
    sampled = grid.join_asof(
        fills.select("TransTime", "fill_price"),
        left_on="grid_t", right_on="TransTime",
        strategy="backward",
    )
    n_grid = sampled.height
    valid = sampled.filter(pl.col("fill_price").is_not_null())
    n_valid = valid.height
    if n_valid == 0:
        return {"結算價": None, "取樣點數": n_grid, "有對到價的點": 0,
                "說明": "網格窗內每點往回都無成交（窗前無任何成交）"}
    avg = float(valid["fill_price"].mean())
    settle = round(avg, ROUND_DP)
    return {
        "結算價": settle,
        "取樣點數": n_grid,
        "有對到價的點": n_valid,
        "原始均價(未四捨)": avg,
        "_sampled": sampled,   # debug 用，主流程不印
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # 避免 polars 表格框線撞 cp950
    p = argparse.ArgumentParser(description="STF 官方結算價試算（從現貨成交 ticks）")
    p.add_argument("--code", required=True, help="標的股票代號，如 2303")
    p.add_argument("--date", required=True, help="結算日（最後交易日）yyyymmdd")
    p.add_argument("--official", type=float, default=None, help="官方公告結算價，帶了就直接對帳")
    args = p.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d")
    fills = load_spot_fills(None, int(args.date), args.code)
    print(f"== {args.code} {args.date} 結算價試算 ==")
    print(f"當日成交列 {fills.height:,} 筆")
    if fills.height:
        print(f"成交時間範圍 {fills['TransTime'].min()} ~ {fills['TransTime'].max()}")

    r = settle_price(fills, date)
    r.pop("_sampled", None)
    print()
    print(f"取樣窗：12:30:05 ~ 13:25:00（每 5 秒一格，共 {r['取樣點數']} 格）")
    print(f"有對到成交價的格點：{r['有對到價的點']}")
    if "原始均價(未四捨)" in r:
        print(f"原始算術均價（未四捨）：{r['原始均價(未四捨)']:.6f}")
    print(f"→ 試算結算價（四捨五入 {ROUND_DP} 位）：{r['結算價']}")
    if "說明" in r:
        print(f"   說明：{r['說明']}")

    if args.official is not None and r["結算價"] is not None:
        diff = r["結算價"] - args.official
        print()
        print(f"官方公告結算價：{args.official}")
        print(f"試算 − 官方差：{diff:+.4f}（{diff / args.official * 100:+.4f}%）")
        if abs(diff) < 0.005:
            print("   ✓ 完全吻合（差 < 0.005，落在四捨五入誤差內）")
        elif abs(diff) < 0.05:
            print("   ~ 很接近（差 < 0.05），口徑大致對，可能差在取樣邊界/揭示頻率")
        else:
            print("   ✗ 有明顯落差，需檢查口徑（取樣時點/成交 vs 揭示/邊界）")


if __name__ == "__main__":
    main()
