"""Independent cash-normalized execution-cost audit of completed replay days."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ..causal_lookup import CLOSE_SECOND, SECOND, open_ns


FIELDS = ["id", "contract", "stream", "entry_day", "quote_ns", "quote_second", "quote_ab", "anchor",
          "entry_fill_ns", "hedged_ns", "close_ns", "close_day", "close_kind", "actual_ab",
          "spot_buy_cash", "spot_buy_qty", "spot_sell_cash", "future_sell_cash", "future_sell_qty", "future_buy_cash",
          "pnl_twd", "pnl_bp"]


def filled_positions(root: Path, actor: str, days: list[str]) -> pl.DataFrame:
    frames = []
    for day in days:
        path = root / f"Date={day}" / actor / "positions.parquet"
        if not path.exists():
            continue
        schema = pl.read_parquet_schema(path)
        if "entry_fill_ns" not in schema or schema["entry_fill_ns"] == pl.Null:
            continue
        frames.append(pl.scan_parquet(path).filter(pl.col("entry_fill_ns").is_not_null())
                      .select(FIELDS).with_columns(pl.lit(day).alias("observed_day")).collect())
    if not frames:
        return pl.DataFrame()
    return (pl.concat(frames, how="diagonal_relaxed").sort("observed_day")
            .unique("id", keep="last", maintain_order=True)
            .with_columns(pl.col("entry_day").map_elements(
                lambda d: open_ns(d) + CLOSE_SECOND * SECOND, return_dtype=pl.Int64).alias("entry_end_ns")))


def enrich(frame: pl.DataFrame) -> pl.DataFrame:
    return (frame.with_columns(
        (pl.col("quote_ab") - pl.col("actual_ab")).alias("entry_decay_bp"),
        (pl.col("anchor") - 5.0).alias("target_basis_bp"),
        pl.when(pl.col("entry_day") == pl.col("close_day")).then(20.0).otherwise(34.0).alias("fee_bp"),
        (pl.col("entry_day") != pl.col("close_day")).alias("carry_exit"),
        ((pl.col("future_buy_cash") / pl.col("spot_sell_cash") - 1) * 10_000).alias("exit_basis_bp"),
        ((pl.col("future_buy_cash") - pl.col("spot_sell_cash")) / pl.col("spot_buy_cash") * 10_000)
        .alias("exit_cash_bp"))
        .with_columns((pl.col("exit_basis_bp") - pl.col("target_basis_bp")).alias("simple_exit_decay_bp"),
                      (pl.col("exit_cash_bp") - pl.col("target_basis_bp")).alias("cash_exit_decay_bp"))
        .with_columns((pl.col("cash_exit_decay_bp") - pl.col("simple_exit_decay_bp")).alias("exit_unit_error_bp"),
                      (pl.col("quote_ab") - pl.col("target_basis_bp") - pl.col("fee_bp")
                       - pl.col("entry_decay_bp") - pl.col("cash_exit_decay_bp") - pl.col("pnl_bp"))
                      .alias("cash_identity_error_bp")))


def stats(frame: pl.DataFrame, column: str, groups: list[str]) -> pl.DataFrame:
    return frame.filter(pl.col(column).is_finite()).group_by(groups).agg(
        pl.len().alias("n"), pl.col(column).mean().alias("mean_bp"),
        pl.col(column).median().alias("median_bp"), pl.col(column).quantile(.95).alias("p95_bp"),
        pl.col(column).quantile(.99).alias("p99_bp"), pl.col(column).min().alias("min_bp"),
        pl.col(column).max().alias("max_bp"),
    ).with_columns(pl.lit(column).alias("measurement"))


def audit(root: Path, output: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "checkpoint.json").read_text())
    days = state["sessions"]
    output.mkdir(parents=True, exist_ok=True)
    results, all_stats, closures = [], [], []
    decay_ids = {r["position_id"] for r in state.get("decays", []) if r["kind"] == "entry"}
    for actor in [a["name"] for a in state["actors"]]:
        frame = enrich(filled_positions(root, actor, days))
        closed = frame.filter(pl.col("close_ns").is_not_null())
        cash_error = closed.select(((pl.col("spot_sell_cash")-pl.col("spot_buy_cash")
            +pl.col("future_sell_cash")-pl.col("future_buy_cash"))/10_000
            -pl.col("spot_buy_cash")/100_000_000*pl.col("fee_bp")-pl.col("pnl_twd")).abs().max()).item()
        if cash_error is not None and cash_error > 1e-6:
            raise AssertionError("closed PnL differs from four cash legs, including forced/partial/expiry exits")
        saved = next(a for a in state["actors"] if a["name"] == actor)
        if abs(closed["pnl_twd"].sum()-sum(r["realized_twd"] for r in saved["daily"])) > 1e-5:
            raise AssertionError("independent closed cash does not reconcile to daily realized totals")
        normal = frame.filter((pl.col("close_kind") == "maker_exit") & (pl.col("spot_buy_cash") > 0))
        paired = frame.filter(pl.col("actual_ab").is_not_null())
        delayed = paired.filter(pl.col("hedged_ns") > pl.col("entry_end_ns"))
        missing = (paired.filter(~pl.col("id").is_in(decay_ids)) if actor == "shadow" and "decays" in state
                   else pl.DataFrame())
        row = dict(actor=actor, fills=frame.height, paired=paired.height, normal_exits=normal.height,
                   closed_cash_max_error_twd=cash_error,
                   delayed_entry_hedges=delayed.height, absent_shadow_entry_decay=missing.height,
                   closed_unpaired=frame.filter(pl.col("close_ns").is_not_null() & pl.col("hedged_ns").is_null()).height,
                   max_cash_identity_error_bp=normal["cash_identity_error_bp"].abs().max(),
                   simple_exit_unit_error_mean_bp=normal["exit_unit_error_bp"].mean(),
                   simple_exit_unit_error_max_abs_bp=normal["exit_unit_error_bp"].abs().max(),
                   simple_exit_unit_error_twd=normal.select(
                       (pl.col("exit_unit_error_bp") * pl.col("spot_buy_cash") / 100_000_000).sum()).item())
        results.append(row)
        for name, data, groups in [("entry_decay_bp", paired, ["stream"]),
                                   ("cash_exit_decay_bp", normal, ["stream", "carry_exit"]),
                                   ("exit_unit_error_bp", normal, ["stream", "carry_exit"])]:
            all_stats.append(stats(data, name, groups).with_columns(pl.lit(actor).alias("actor")))
        closures.append(frame.filter(pl.col("close_ns").is_not_null()).group_by("stream", "close_kind", "carry_exit")
                        .agg(pl.len().alias("n"), pl.col("pnl_twd").sum(), pl.col("pnl_bp").mean())
                        .with_columns(pl.lit(actor).alias("actor")))
        normal.select("id", "stream", "entry_day", "close_day", "simple_exit_decay_bp", "cash_exit_decay_bp",
                      "exit_unit_error_bp", "cash_identity_error_bp").write_parquet(output / f"{actor}_normal_exits.parquet")
        if missing.height:
            missing.write_parquet(output / "missing_shadow_entry_decay.parquet")
    report = dict(root=str(root.resolve()), manifest_status=manifest["status"], completed_sessions=len(days),
                  through=days[-1], actors=results,
                  interpretation="Positive cost means worse than quote/frozen exit target; bp normalized by entry spot cash.",
                  acceptance="Cash identity is an accounting check, not a claim that the forecast is calibrated.")
    pl.concat(all_stats, how="diagonal_relaxed").write_csv(output / "cost_distributions.csv")
    pl.concat(closures, how="diagonal_relaxed").write_csv(output / "close_kinds.csv")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
