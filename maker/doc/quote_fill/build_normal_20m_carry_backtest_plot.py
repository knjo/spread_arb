"""Build the single-policy 20M normal-carry backtest chart and daily audit CSV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import polars as pl


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = (
    ROOT
    / "maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only"
)
DAILY_PATH = ARTIFACT_ROOT / "cap_daily.parquet"
MARKER_PATH = ARTIFACT_ROOT / "complete.json"
OUTPUT_DIR = ROOT / "maker/doc/quote_fill/assets"
OUTPUT_PNG = OUTPUT_DIR / "normal_20m_carry_backtest_20260821.png"
OUTPUT_CSV = OUTPUT_DIR / "normal_20m_carry_backtest_daily_20260821.csv"
SCENARIO_ID = "normal_cutoff_1300_hard_20000000"
CAP_TWD = 20_000_000.0
CHINESE_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_daily() -> pl.DataFrame:
    marker = json.loads(MARKER_PATH.read_text())
    if not marker.get("complete"):
        raise RuntimeError(f"Incomplete source artifact: {MARKER_PATH}")
    expected_hash = marker["artifacts"][DAILY_PATH.name]["sha256"]
    actual_hash = sha256(DAILY_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Source hash mismatch for {DAILY_PATH}: {actual_hash} != {expected_hash}"
        )

    daily = (
        pl.read_parquet(DAILY_PATH)
        .filter(pl.col("scenario_id") == SCENARIO_ID)
        .select(
            pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("opening_outstanding_one_way_notional_twd").alias(
                "opening_carry_twd"
            ),
            pl.col("accepted_entry_one_way_turnover_twd").alias(
                "new_spot_entry_twd"
            ),
            pl.col("eod_outstanding_one_way_notional_twd").alias(
                "pre_expiry_1320_carry_twd"
            ),
            pl.col("overnight_carried_out_one_way_notional_twd").alias(
                "model_carry_to_next_session_twd"
            ),
            pl.col("realized_gross_pnl_twd").alias("realized_gross_twd"),
            pl.col("realized_transaction_cost_twd").alias("transaction_cost_twd"),
            pl.col("realized_net_pnl_twd").alias("realized_net_twd"),
            pl.col("cumulative_realized_net_pnl_twd").alias(
                "cumulative_realized_net_twd"
            ),
        )
        .sort("date")
    )
    if daily.height != 63:
        raise RuntimeError(f"Expected 63 sessions, got {daily.height}")

    prior_close = daily["model_carry_to_next_session_twd"].shift(1)
    transition_error = (
        daily.with_columns(prior_close.alias("prior_close_twd"))
        .filter(
            pl.col("prior_close_twd").is_not_null()
            & (
                (pl.col("opening_carry_twd") - pl.col("prior_close_twd")).abs()
                > 1e-6
            )
        )
        .height
    )
    if transition_error:
        raise RuntimeError(
            f"D+1 opening carry does not match D close in {transition_error} sessions"
        )

    cumulative_error = daily.select(
        (
            pl.col("realized_net_twd").cum_sum()
            - pl.col("cumulative_realized_net_twd")
        )
        .abs()
        .max()
    ).item()
    if cumulative_error > 1e-6:
        raise RuntimeError(f"Cumulative P&L mismatch: {cumulative_error}")
    return daily


def twd_millions(value: float, _: int) -> str:
    return f"{value / 1_000_000:.1f}M"


def twd_thousands(value: float, _: int) -> str:
    return f"{value / 1_000:.0f}k"


def main() -> None:
    daily = load_daily()
    dates = daily["date"].to_list()
    daily_net = daily["realized_net_twd"].to_list()
    cumulative_net = daily["cumulative_realized_net_twd"].to_list()
    new_entries = daily["new_spot_entry_twd"].to_list()
    opening_carry = daily["opening_carry_twd"].to_list()
    model_carry = daily["model_carry_to_next_session_twd"].to_list()

    net_total = float(daily["realized_net_twd"].sum())
    net_daily_mean = float(daily["realized_net_twd"].mean())
    entry_daily_mean = float(daily["new_spot_entry_twd"].mean())
    carry_daily_mean = float(daily["model_carry_to_next_session_twd"].mean())
    positive_days = sum(value > 0 for value in daily_net)
    negative_days = sum(value < 0 for value in daily_net)

    if CHINESE_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
        chart_font = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
    else:
        chart_font = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chart_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "2,000 萬上限｜Normal maker 留倉回測（63 個交易日）",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.957,
        "13:00 停止新倉；不做貼價硬出；前一日模型 carry 完整接到下一交易日開盤",
        ha="center",
        fontsize=11,
        color="#334155",
    )

    pnl_ax = axes[0]
    pnl_colors = ["#16a34a" if value >= 0 else "#dc2626" for value in daily_net]
    pnl_ax.bar(dates, daily_net, color=pnl_colors, width=0.72, alpha=0.82)
    pnl_ax.axhline(0.0, color="#64748b", linewidth=0.8)
    pnl_ax.yaxis.set_major_formatter(FuncFormatter(twd_thousands))
    pnl_ax.set_ylabel("每日已實現淨利")
    pnl_ax.set_title(
        f"獲利｜總淨利 {net_total / 1_000_000:.3f}M，日均 {net_daily_mean / 1_000:.1f}k，"
        f"正／負日 {positive_days}/{negative_days}",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    cumulative_ax = pnl_ax.twinx()
    cumulative_ax.plot(
        dates,
        cumulative_net,
        color="#0f172a",
        linewidth=2.1,
        label="累積已實現淨利",
    )
    cumulative_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
    cumulative_ax.set_ylabel("累積淨利", color="#0f172a")
    cumulative_ax.spines["top"].set_visible(False)
    cumulative_ax.legend(loc="upper left", frameon=False)

    entry_ax = axes[1]
    entry_ax.bar(dates, new_entries, color="#2563eb", width=0.72, alpha=0.78)
    entry_ax.axhline(
        entry_daily_mean,
        color="#1e3a8a",
        linestyle="--",
        linewidth=1.5,
        label=f"日均 {entry_daily_mean / 1_000_000:.2f}M",
    )
    entry_ax.axhline(
        CAP_TWD,
        color="#94a3b8",
        linestyle=":",
        linewidth=1.3,
        label="20M 同時持倉上限",
    )
    entry_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
    entry_ax.set_ylabel("現貨進場名目")
    entry_ax.set_title(
        "每日新單金額｜現貨腿 one-way notional；平倉釋放額度後可在同日重用",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    entry_ax.legend(loc="upper left", frameon=False, ncols=2)

    carry_ax = axes[2]
    carry_ax.bar(
        dates,
        model_carry,
        color="#7c3aed",
        width=0.72,
        alpha=0.76,
        label="13:20 模型 carry（到期日再扣 Close）",
    )
    carry_ax.plot(
        dates,
        opening_carry,
        color="#ea580c",
        linewidth=1.4,
        marker="o",
        markersize=2.6,
        alpha=0.9,
        label="當日開盤承接庫存",
    )
    carry_ax.axhline(
        carry_daily_mean,
        color="#4c1d95",
        linestyle="--",
        linewidth=1.4,
        label=f"模型 carry 日均 {carry_daily_mean / 1_000_000:.2f}M",
    )
    carry_ax.axhline(
        CAP_TWD,
        color="#dc2626",
        linestyle=":",
        linewidth=1.5,
        label="20M hard cap",
    )
    carry_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
    carry_ax.set_ylabel("現貨腿 one-way notional")
    carry_ax.set_title(
        "模型留倉｜62/62 個相鄰日帳務相等：D+1 開盤 = D 的 13:20 carry（到期日扣 Close）",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    carry_ax.legend(loc="upper left", frameon=False, ncols=2)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].tick_params(axis="x", rotation=40)
    axes[-1].set_xlabel("交易日（2026-05-21 至 2026-08-19）")
    fig.text(
        0.5,
        0.012,
        "淨利已扣指定稅費，但只在 terminal 日認列、未做每日 carry MTM。"
        "未 replay 13:20–13:30，普通日 carry 是實際收盤留倉上界；"
        "到期日另以官方 Close 結清。固定 45 檔回溯 cohort，故仍是研究估計。",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.98, 0.94), h_pad=2.1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    daily.write_csv(OUTPUT_CSV, float_precision=4, date_format="%Y-%m-%d")

    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {OUTPUT_CSV}")
    print(
        json.dumps(
            {
                "sessions": daily.height,
                "net_total_twd": net_total,
                "net_daily_mean_twd": net_daily_mean,
                "new_spot_entry_total_twd": float(
                    daily["new_spot_entry_twd"].sum()
                ),
                "new_spot_entry_daily_mean_twd": entry_daily_mean,
                "model_carry_daily_mean_twd": carry_daily_mean,
                "model_carry_max_twd": float(
                    daily["model_carry_to_next_session_twd"].max()
                ),
                "positive_days": positive_days,
                "negative_days": negative_days,
                "carry_transition_mismatches": 0,
                "source_cap_daily_sha256": sha256(DAILY_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
