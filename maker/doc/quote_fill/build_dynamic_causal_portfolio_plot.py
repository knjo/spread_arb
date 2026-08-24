"""Plot the dynamic-universe full-population estimated portfolio replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

DEFAULT_ROOT = Path(
    "maker/data/walkforward/"
    "dynamic_estimated_path_portfolio_causal_v1_20260822"
)
DEFAULT_CAPS = (10_000_000.0, 20_000_000.0, 30_000_000.0)


def build_plot(
    daily: pl.DataFrame,
    output_path: Path,
    *,
    caps: Sequence[float] = DEFAULT_CAPS,
) -> Path:
    required = {
        "Date",
        "hard_intraday_cap_twd",
        "cumulative_realized_net_pnl_twd",
        "new_spot_notional_twd",
        "eod_spot_notional_twd",
        "unresolved_eod_spot_notional_twd",
        "unresolved_eod_positions",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"full-population cap daily panel missing columns: {missing}")
    if daily.is_empty():
        raise ValueError("full-population cap daily panel is empty")

    figure, axes = plt.subplots(
        len(caps),
        3,
        figsize=(17, 4.1 * len(caps)),
        sharex="col",
        constrained_layout=True,
    )
    if len(caps) == 1:
        axes = [axes]
    for row_index, cap in enumerate(caps):
        panel = daily.filter(pl.col("hard_intraday_cap_twd") == float(cap)).sort(
            "Date"
        )
        if panel.is_empty():
            raise ValueError(f"missing requested cap panel: {cap}")
        dates = [
            date(int(value[:4]), int(value[4:6]), int(value[6:8]))
            for value in panel["Date"]
        ]
        cap_millions = cap / 1_000_000.0

        pnl_axis, flow_axis, inventory_axis = axes[row_index]
        pnl_axis.plot(
            dates,
            panel["cumulative_realized_net_pnl_twd"].to_list(),
            color="#1769aa",
            linewidth=1.8,
        )
        pnl_axis.axhline(0.0, color="#555555", linewidth=0.7)
        pnl_axis.set_title(f"{cap_millions:g}M cap — cumulative realized net")
        pnl_axis.set_ylabel("TWD")

        flow_axis.bar(
            dates,
            (panel["new_spot_notional_twd"] / 1_000_000.0).to_list(),
            width=0.75,
            color="#4caf50",
            alpha=0.78,
        )
        flow_axis.set_title(f"{cap_millions:g}M cap — daily new spot")
        flow_axis.set_ylabel("TWD million")

        inventory_axis.plot(
            dates,
            (panel["eod_spot_notional_twd"] / 1_000_000.0).to_list(),
            color="#ef6c00",
            linewidth=1.8,
            label="all EOD inventory",
        )
        inventory_axis.plot(
            dates,
            (
                panel["unresolved_eod_spot_notional_twd"] / 1_000_000.0
            ).to_list(),
            color="#b71c1c",
            linewidth=1.3,
            linestyle="--",
            label="open/unresolved",
        )
        inventory_axis.axhline(
            cap_millions,
            color="#777777",
            linewidth=0.8,
            linestyle=":",
            label="hard cap",
        )
        final_unresolved = int(panel["unresolved_eod_positions"].tail(1).item())
        inventory_axis.set_title(
            f"{cap_millions:g}M cap — EOD spot inventory "
            f"(final unresolved={final_unresolved})"
        )
        inventory_axis.set_ylabel("TWD million")
        inventory_axis.legend(loc="upper left", fontsize=8)

        for axis in (pnl_axis, flow_axis, inventory_axis):
            axis.grid(axis="y", alpha=0.22)
            axis.tick_params(axis="x", rotation=30)

    figure.suptitle(
        "Dynamic causal universe | makerFill-estimated entries | "
        "1Hz frozen-lower taker/taker exits\n"
        "Realized PnL excludes still-open/unresolved cashflows; "
        "expiry paired marks may use a futures last-trade proxy.",
        fontsize=13,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = args.output or (root / "portfolio_10m_20m_30m.png")
    result = build_plot(
        pl.read_parquet(root / "full_population_cap_daily.parquet"),
        output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
