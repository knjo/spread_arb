"""Build the 10M versus 30M normal-carry backtest comparison chart."""

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
OUTPUT_PNG = OUTPUT_DIR / "normal_10m_30m_carry_comparison_20260821.png"
OUTPUT_CSV = OUTPUT_DIR / "normal_10m_30m_carry_comparison_daily_20260821.csv"
CHINESE_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SCENARIOS = (
    (10_000_000.0, "normal_cutoff_1300_hard_10000000", "#0284c7"),
    (30_000_000.0, "normal_cutoff_1300_hard_30000000", "#7c3aed"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_daily() -> dict[float, pl.DataFrame]:
    marker = json.loads(MARKER_PATH.read_text())
    if not marker.get("complete"):
        raise RuntimeError(f"Incomplete source artifact: {MARKER_PATH}")
    expected_hash = marker["artifacts"][DAILY_PATH.name]["sha256"]
    actual_hash = sha256(DAILY_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Source hash mismatch for {DAILY_PATH}: {actual_hash} != {expected_hash}"
        )

    source = pl.read_parquet(DAILY_PATH)
    result: dict[float, pl.DataFrame] = {}
    for cap_twd, scenario_id, _ in SCENARIOS:
        daily = (
            source.filter(pl.col("scenario_id") == scenario_id)
            .select(
                pl.lit(cap_twd).alias("hard_cap_twd"),
                pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("date"),
                pl.col("opening_outstanding_one_way_notional_twd").alias(
                    "opening_carry_twd"
                ),
                pl.col("accepted_entry_one_way_turnover_twd").alias(
                    "new_spot_entry_twd"
                ),
                pl.col("exit_one_way_turnover_twd").alias("spot_exit_twd"),
                pl.col("eod_outstanding_one_way_notional_twd").alias(
                    "pre_expiry_1320_carry_twd"
                ),
                pl.col("overnight_carried_out_one_way_notional_twd").alias(
                    "model_carry_to_next_session_twd"
                ),
                pl.col("realized_gross_pnl_twd").alias("realized_gross_twd"),
                pl.col("realized_transaction_cost_twd").alias(
                    "transaction_cost_twd"
                ),
                pl.col("realized_net_pnl_twd").alias("realized_net_twd"),
                pl.col("cumulative_realized_net_pnl_twd").alias(
                    "cumulative_realized_net_twd"
                ),
            )
            .sort("date")
        )
        if daily.height != 63 or daily["date"].n_unique() != 63:
            raise RuntimeError(
                f"Expected 63 unique sessions for {scenario_id}, got {daily.height}"
            )

        transition_mismatches = (
            daily.with_columns(
                pl.col("model_carry_to_next_session_twd")
                .shift(1)
                .alias("prior_model_carry_twd")
            )
            .filter(
                pl.col("prior_model_carry_twd").is_not_null()
                & (
                    (
                        pl.col("opening_carry_twd")
                        - pl.col("prior_model_carry_twd")
                    ).abs()
                    > 1e-6
                )
            )
            .height
        )
        if transition_mismatches:
            raise RuntimeError(
                f"D+1 carry mismatch in {scenario_id}: {transition_mismatches}"
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
            raise RuntimeError(
                f"Cumulative P&L mismatch in {scenario_id}: {cumulative_error}"
            )
        result[cap_twd] = daily
    return result


def twd_millions(value: float, _: int) -> str:
    return f"{value / 1_000_000:.1f}M"


def twd_thousands(value: float, _: int) -> str:
    return f"{value / 1_000:.0f}k"


def main() -> None:
    all_daily = load_all_daily()
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

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18, 13),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0.28, "wspace": 0.10},
    )
    fig.suptitle(
        "Normal maker 留倉回測｜1,000 萬 vs 3,000 萬",
        fontsize=18,
        fontweight="bold",
        y=0.987,
    )
    fig.text(
        0.5,
        0.962,
        "13:00 停止新倉｜單品上限 30%｜不做貼價硬出｜模型 carry 接續到下一交易日",
        ha="center",
        fontsize=11,
        color="#334155",
    )

    cumulative_max = max(
        float(frame["cumulative_realized_net_twd"].max())
        for frame in all_daily.values()
    )
    summaries: list[dict[str, float | int | str]] = []

    for column, (cap_twd, scenario_id, color) in enumerate(SCENARIOS):
        daily = all_daily[cap_twd]
        dates = daily["date"].to_list()
        daily_net = daily["realized_net_twd"].to_list()
        cumulative_net = daily["cumulative_realized_net_twd"].to_list()
        new_entries = daily["new_spot_entry_twd"].to_list()
        opening_carry = daily["opening_carry_twd"].to_list()
        model_carry = daily["model_carry_to_next_session_twd"].to_list()

        cap_m = cap_twd / 1_000_000
        net_total = float(daily["realized_net_twd"].sum())
        net_mean = float(daily["realized_net_twd"].mean())
        gross_total = float(daily["realized_gross_twd"].sum())
        cost_total = float(daily["transaction_cost_twd"].sum())
        entry_total = float(daily["new_spot_entry_twd"].sum())
        entry_mean = float(daily["new_spot_entry_twd"].mean())
        spot_total = float(
            (daily["new_spot_entry_twd"] + daily["spot_exit_twd"]).sum()
        )
        carry_mean = float(daily["model_carry_to_next_session_twd"].mean())
        carry_max = float(daily["model_carry_to_next_session_twd"].max())
        positive_days = sum(value > 0 for value in daily_net)
        negative_days = sum(value < 0 for value in daily_net)

        pnl_ax = axes[0, column]
        pnl_colors = ["#16a34a" if value >= 0 else "#dc2626" for value in daily_net]
        pnl_ax.bar(dates, daily_net, color=pnl_colors, width=0.72, alpha=0.82)
        pnl_ax.axhline(0.0, color="#64748b", linewidth=0.8)
        pnl_ax.yaxis.set_major_formatter(FuncFormatter(twd_thousands))
        pnl_ax.set_title(
            f"{cap_m:.0f}M｜Net {net_total / 1_000_000:.3f}M；"
            f"日均 {net_mean / 1_000:.1f}k；正/負 {positive_days}/{negative_days}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        cumulative_ax = pnl_ax.twinx()
        cumulative_ax.plot(
            dates,
            cumulative_net,
            color="#0f172a",
            linewidth=2.0,
            label="累積已實現淨利",
        )
        cumulative_ax.set_ylim(0.0, cumulative_max * 1.10)
        cumulative_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
        cumulative_ax.spines["top"].set_visible(False)
        if column == 1:
            cumulative_ax.set_ylabel("累積淨利")
        else:
            cumulative_ax.tick_params(labelright=False)
            cumulative_ax.spines["right"].set_visible(False)
        cumulative_ax.legend(loc="upper left", frameon=False)

        entry_ax = axes[1, column]
        entry_ax.bar(dates, new_entries, color=color, width=0.72, alpha=0.76)
        entry_ax.axhline(
            entry_mean,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"新單日均 {entry_mean / 1_000_000:.2f}M",
        )
        entry_ax.axhline(
            cap_twd,
            color="#64748b",
            linestyle=":",
            linewidth=1.4,
            label=f"{cap_m:.0f}M 同時持倉上限",
        )
        entry_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
        entry_ax.set_title(
            f"每日現貨新單｜63日合計 {entry_total / 1_000_000:.1f}M",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        entry_ax.legend(loc="upper left", frameon=False)

        carry_ax = axes[2, column]
        carry_ax.bar(
            dates,
            model_carry,
            color=color,
            width=0.72,
            alpha=0.66,
            label="13:20 模型 carry（到期日再扣 Close）",
        )
        carry_ax.plot(
            dates,
            opening_carry,
            color="#ea580c",
            linewidth=1.35,
            marker="o",
            markersize=2.3,
            label="當日開盤承接庫存",
        )
        carry_ax.axhline(
            carry_mean,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"模型 carry 日均 {carry_mean / 1_000_000:.2f}M",
        )
        carry_ax.axhline(
            cap_twd,
            color="#dc2626",
            linestyle=":",
            linewidth=1.5,
            label=f"{cap_m:.0f}M hard cap",
        )
        carry_ax.yaxis.set_major_formatter(FuncFormatter(twd_millions))
        carry_ax.set_title(
            "62/62 相鄰日帳務相等：D+1 opening = D model carry",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        carry_ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)

        summaries.append(
            {
                "scenario_id": scenario_id,
                "hard_cap_twd": cap_twd,
                "sessions": daily.height,
                "gross_total_twd": gross_total,
                "transaction_cost_total_twd": cost_total,
                "net_total_twd": net_total,
                "net_daily_mean_twd": net_mean,
                "new_spot_entry_total_twd": entry_total,
                "new_spot_entry_daily_mean_twd": entry_mean,
                "spot_entry_plus_exit_total_twd": spot_total,
                "spot_entry_plus_exit_daily_mean_twd": spot_total / daily.height,
                "model_carry_daily_mean_twd": carry_mean,
                "model_carry_max_twd": carry_max,
                "positive_days": positive_days,
                "negative_days": negative_days,
                "carry_transition_mismatches": 0,
            }
        )

    axes[0, 0].set_ylabel("每日已實現淨利")
    axes[1, 0].set_ylabel("現貨新單 one-way")
    axes[2, 0].set_ylabel("現貨腿 one-way notional")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[2, :]:
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.tick_params(axis="x", rotation=35)
        axis.set_xlabel("交易日（2026-05-21 至 2026-08-19）")

    fig.text(
        0.5,
        0.012,
        "Realized cashflow only，未做 carry MTM。行情 replay 止於 13:20；"
        "普通日模型 carry 是 13:30 真正留倉上界，到期日另用官方 Close 結清。"
        "固定 45 檔回溯 cohort，僅供研究。",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.98, 0.945))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    pl.concat(all_daily.values()).sort(["hard_cap_twd", "date"]).write_csv(
        OUTPUT_CSV,
        float_precision=4,
        date_format="%Y-%m-%d",
    )

    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {OUTPUT_CSV}")
    print(
        json.dumps(
            {
                "source_cap_daily_sha256": sha256(DAILY_PATH),
                "scenarios": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
