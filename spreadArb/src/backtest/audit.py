"""Independent audit of saved point-replay ledgers; never changes a run or policy.

uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.audit
Use --compare A_fixed=audit_20260922_A --compare B_dyn=audit_20260922_B after replaying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ..common.paths import DATA_ROOT, grid_days, mapping_path, points_path


def load(name: str):
    root = DATA_ROOT / "backtest" / name
    positions = pl.read_parquet(root / "positions.parquet")
    daily = pl.read_csv(root / "daily.csv", schema_overrides={"day": pl.String})
    rollbacks = (pl.read_csv(root / "rollbacks.csv", schema_overrides={"day": pl.String})
                 if (root / "rollbacks.csv").exists()
                 else pl.DataFrame(schema={"day": pl.String, "cost_twd": pl.Float64}))
    config = json.loads((root / "config.json").read_text())
    return positions, daily, rollbacks, config


def metrics(values: pl.Series, cap: float) -> dict:
    a = values.to_numpy()
    equity = np.r_[0.0, np.cumsum(a)]
    return dict(days=len(a), pnl_twd=float(a.sum()), daily_twd=float(a.mean()),
                annual_simple_pct=float(a.mean() * 250 / cap * 100),
                booked_pnl_drawdown_twd=float((equity - np.maximum.accumulate(equity)).min()),
                booked_pnl_sharpe=float(a.mean() / a.std(ddof=1) * np.sqrt(250)) if a.std(ddof=1) > 0 else None)


def reconcile_daily(pos: pl.DataFrame, daily: pl.DataFrame, rb: pl.DataFrame) -> pl.DataFrame:
    closed = pos.filter(pl.col("close_kind") != "open_marked")
    realized = closed.group_by("close_day").agg(pl.col("pnl_net").sum().alias("realized_twd"))
    marked = pos.filter(pl.col("close_kind") == "open_marked").group_by("close_day").agg(
        pl.col("pnl_net").sum().alias("terminal_mark_twd"))
    rollback = rb.group_by("day").agg(pl.col("cost_twd").sum().alias("rollback_twd"))
    return (daily.select("day", pl.col("pnl_net").alias("reported_daily_twd"))
            .join(realized, left_on="day", right_on="close_day", how="left")
            .join(marked, left_on="day", right_on="close_day", how="left")
            .join(rollback, on="day", how="left")
            .fill_null(0).sort("day")
            .with_columns((pl.col("realized_twd") + pl.col("rollback_twd")
                           + pl.col("terminal_mark_twd")).alias("reconciled_booked_twd")))


def capacity_events(pos: pl.DataFrame, begin: str) -> pl.DataFrame:
    entry = pos.select("id", "vc", pl.col(begin).alias("ns"),
                       (pl.col("spot_buy_cash") / 1e4).alias("delta_twd"), pl.lit(1).alias("kind"))
    leave = pos.select("id", "vc", pl.col("exit_hedge_ns").alias("ns"),
                       (-pl.col("spot_buy_cash") / 1e4).alias("delta_twd"), pl.lit(0).alias("kind"))
    return pl.concat([entry, leave]).sort("ns", "kind", "id").with_columns(
        pl.col("delta_twd").cum_sum().alias("committed_twd"),
        pl.col("delta_twd").cum_sum().over("vc").alias("product_twd"))


def cost_sensitivity(pos: pl.DataFrame, total: float, days: int, cap: float) -> dict:
    # Project-defined fee profile, not a claim about currently applicable exchange fees.
    from maker.src.quote_fill.transaction_costs import TransactionCostProfile

    profile = TransactionCostProfile()
    exact = 0.0
    for r in pos.iter_rows(named=True):
        exact += profile.paired_cycle_cost_breakdown(
            entry_spot_price=r["spot_buy_cash"] / r["shares"] / 1e4,
            exit_spot_price=r["spot_sell_cash"] / r["shares"] / 1e4,
            entry_future_price=r["fut_sell_px"] / 1e4, exit_future_price=r["fut_buy_px"] / 1e4,
            shares=r["shares"], contracts=1.0, same_day=r["quote_day"] == r["close_day"]).total_twd
    delta = exact - float(pos["fees"].sum())
    funding = pos.select((pl.col("spot_buy_cash") / 1e4 * pl.col("holding_days") * 0.02 / 365).sum()).item()
    return dict(profile=profile.profile_id, fee_delta_twd=delta, funding_2pct_twd=funding,
                repriced_net_twd=total - delta, repriced_annual_pct=(total - delta) / days * 250 / cap * 100,
                repriced_with_funding_net_twd=total - delta - funding,
                repriced_with_funding_annual_pct=(total - delta - funding) / days * 250 / cap * 100,
                caveat="Static sensitivity: unchanged trades and rollback estimates; settlement and terminal marks use the same paired-cycle fee proxy. Not a corrected replay.")


def audit_run(name: str, out: Path) -> dict:
    pos, daily, rb, cfg = load(name)
    cap = cfg["cap_twd"]
    out.mkdir(parents=True, exist_ok=True)
    fee_bp = pl.when(pl.col("close_day") == pl.col("quote_day")).then(
        cfg["cost"]["fee_same_day_bp"]).otherwise(cfg["cost"]["fee_overnight_bp"])
    spot = (pl.col("spot_sell_cash") - pl.col("spot_buy_cash")) / 1e4
    future = (pl.col("fut_sell_px") - pl.col("fut_buy_px")) * pl.col("shares") / 1e4
    fees = pl.col("spot_buy_cash") / 1e8 * fee_bp
    errors = pos.select((spot - pl.col("pnl_spot")).abs().max().alias("spot_max_error_twd"),
                        (future - pl.col("pnl_fut")).abs().max().alias("future_max_error_twd"),
                        (fees - pl.col("fees")).abs().max().alias("fees_max_error_twd"),
                        (spot + future - fees - pl.col("pnl_net")).abs().max().alias("net_max_error_twd"))
    rec = reconcile_daily(pos, daily, rb)
    rec.write_csv(out / "reconciled_daily.csv")
    core = rec.filter(pl.col("day").is_between(pl.lit("20260401"), pl.lit("20260731")))
    late = pos.filter(pl.col("close_day") > pl.col("expiry"))
    late = late.with_columns(((pl.col("fut_sell_px") * pl.col("shares") - pl.col("spot_buy_cash"))
                             / 1e4 - pl.col("fees")).alias("basis_zero_net_twd"))
    late.write_csv(out / "after_expiry.csv")
    maps = []
    for day in daily["day"]:
        maps.append(pl.read_parquet(mapping_path(day)).select(
            pl.lit(day).alias("close_day"), pl.col("ValueCode").alias("vc"),
            pl.col("QuoteCode").alias("close_mapping_qc")))
    joined = pos.join(pl.concat(maps), on=["close_day", "vc"], how="left", validate="m:1")
    mismatches = joined.filter((pl.col("qc") != pl.col("close_mapping_qc"))
                               & (pl.col("close_kind") != "settlement"))
    mismatches.write_csv(out / "contract_mismatch.csv")
    capacities = {}
    for begin in ("quote_ns", "fill_ns", "hedge_ns"):
        events = capacity_events(pos, begin)
        over = events.filter(pl.col("committed_twd") > cap + 0.01)
        over.write_csv(out / f"capacity_over_{begin}.csv")
        capacities[begin] = dict(peak_twd=events["committed_twd"].max(), over_cap_events=over.height,
                                max_product_twd=events["product_twd"].max())
    maker = pos.filter(pl.col("close_kind") == "maker_exit")
    entry_dups = pos.group_by("quote_day", "vc", "stream", "fill_ns").len().filter(pl.col("len") > 1)
    exit_dups = maker.group_by("close_day", "vc", "exit_route", "exit_fill_ns").len().filter(pl.col("len") > 1)
    causal = pos.select((pl.col("fill_ns") <= pl.col("quote_ns")).sum().alias("fill_before_live"),
                        (pl.col("hedge_ns") < pl.col("fill_ns") + 50_000_000).sum().alias("entry_hedge_too_early"),
                        (pl.col("exit_hedge_ns") <= pl.col("hedge_ns")).sum().alias("close_before_open"),
                        (pl.col("close_day") < pl.col("quote_day")).sum().alias("close_day_before_open_day"))
    if cfg["dyn_q"] is not None:
        should_raise = ((daily["committed_end"].shift(1).fill_null(0) >= cfg["dyn_cap_frac"] * cap)
                        & (daily["signals_prev"] > 0))
        raised = daily["hurdle_used"] > cfg["cost"]["hurdle_bp_per_day"] + 0.01
        dynamic = dict(raised_days=int(raised.sum()), activation_disagreements=int((should_raise != raised).sum()),
                       previous_signal_count_disagreements=int((daily["signals_prev"] != daily["signals_today"].shift(1).fill_null(0)).sum()))
    else:
        dynamic = dict(raised_days=int((daily["hurdle_used"] > cfg["cost"]["hurdle_bp_per_day"] + 0.01).sum()))
    available = [d for d in grid_days() if points_path(d, "s1_entries").exists() and points_path(d, "s2_entries").exists()]
    gross = float(pos["pnl_net"].sum())
    total = gross + float(rb["cost_twd"].sum())
    result = dict(run=name, pairs=pos.height, days=daily.height, start=daily["day"][0], end=daily["day"][-1],
                  cap_twd=cap, complete_available_period=daily["day"].to_list() == available,
                  closed_pair_pnl_twd=float(pos.filter(pl.col("close_kind") != "open_marked")["pnl_net"].sum()),
                  terminal_mark_pairs=pos.filter(pl.col("close_kind") == "open_marked").height,
                  terminal_mark_twd=float(pos.filter(pl.col("close_kind") == "open_marked")["pnl_net"].sum()),
                  rollback_count=rb.height, rollback_twd=float(rb["cost_twd"].sum()),
                  total_twd=total, daily_twd=total / daily.height, annual_simple_pct=total / daily.height * 250 / cap * 100,
                  ledger_errors=errors.row(0, named=True), unresolved_pairs=pos["pnl_net"].null_count(),
                  daily_reconciliation_error_twd=abs(rec["reconciled_booked_twd"].sum() - total),
                  reported_full_daily=metrics(rec["reported_daily_twd"], cap),
                  reconciled_full_daily=metrics(rec["reconciled_booked_twd"], cap),
                  reported_core=metrics(core["reported_daily_twd"], cap),
                  reconciled_core=metrics(core["reconciled_booked_twd"], cap),
                  core_rollback_twd=float(core["rollback_twd"].sum()),
                  after_expiry_pairs=late.height, after_expiry_pnl_twd=float(late["pnl_net"].sum()),
                  late_basis_zero_static_delta_twd=float((late["basis_zero_net_twd"] - late["pnl_net"]).sum()),
                  mismatched_contract_pairs=mismatches.height, mismatched_contract_pnl_twd=float(mismatches["pnl_net"].sum()),
                  missing_close_mapping=joined["close_mapping_qc"].null_count(), capacities=capacities,
                  entry_duplicate_groups=entry_dups.height, exit_duplicate_groups=exit_dups.height,
                  causal_checks=causal.row(0, named=True), dynamic=dynamic,
                  entry_hedge_over_5s=pos.filter(pl.col("hedge_ns") - pl.col("fill_ns") > 5e9).height,
                  exit_hedge_over_5s=maker.filter(pl.col("exit_hedge_ns") - pl.col("exit_fill_ns") > 5e9).height,
                  dual_exit_risk_pairs=maker.filter(pl.col("exit_double_risk")).height,
                  funding_2pct_sensitivity_twd=pos.select((pl.col("spot_buy_cash") / 1e4 * pl.col("holding_days") * 0.02 / 365).sum()).item(),
                  extra_40_per_pair_sensitivity_twd=pos.height * 40,
                  project_cost_sensitivity=cost_sensitivity(pos, total, daily.height, cap),
                  caveat="Reconciled daily PnL books closes, rollbacks and the terminal mark; it is not daily mark-to-market equity. Capacity excludes unfilled orders.")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_runs(original: str, repeat: str) -> dict:
    a = load(original)
    b = load(repeat)
    result = dict(original=original, repeat=repeat, config_equal=a[3] == b[3])
    for key, x, y in (("positions", a[0].sort("id"), b[0].sort("id")),
                      ("daily", a[1].drop("load_s", "elapsed_s"), b[1].drop("load_s", "elapsed_s")),
                      ("rollbacks", a[2], b[2])):
        try:
            assert_frame_equal(x, y, check_exact=True)
            result[key + "_exact"] = True
        except AssertionError as exc:
            result[key + "_exact"] = False
            result[key + "_difference"] = str(exc)[:1000]
    return result


def audit_point_legs(names: list[str], out: Path) -> dict:
    """Match every recorded maker/hedge price to Stage 2, independently of Replay."""
    requests = []
    for name in names:
        pos = load(name)[0]
        for r in pos.iter_rows(named=True):
            requests.append(dict(run=name, id=r["id"], phase="entry", day=r["quote_day"], vc=r["vc"],
                                 source="s1" if r["stream"] == "S1" else "s2",
                                 side="buy" if r["stream"] == "S1" else "sell", qc=r["qc"],
                                 price=r["entry_price"], t_fill_ns=r["fill_ns"], hedge_ns=r["hedge_ns"],
                                 hedge_vwap=r["fut_sell_px"] if r["stream"] == "S1" else r["spot_buy_cash"] // r["shares"]))
            if r["close_kind"] == "maker_exit":
                requests.append(dict(run=name, id=r["id"], phase="exit", day=r["close_day"], vc=r["vc"],
                                     source="s1" if r["exit_route"] == "E1" else "s2",
                                     side="sell" if r["exit_route"] == "E1" else "buy", qc=r["qc"],
                                     price=r["exit_price"], t_fill_ns=r["exit_fill_ns"], hedge_ns=r["exit_hedge_ns"],
                                     hedge_vwap=r["fut_buy_px"] if r["exit_route"] == "E1" else r["spot_sell_cash"] // r["shares"]))
    frame = pl.from_dicts(requests)
    matched = []
    keys = ["vc", "side", "price", "t_fill_ns", "hedge_ns", "hedge_vwap"]
    for (day, source), group in frame.partition_by("day", "source", as_dict=True).items():
        requested_times = group["t_fill_ns"].unique().to_list()
        facts = (pl.scan_parquet(points_path(day, source + "_entries"))
                 .filter(pl.col("vc").is_in(group["vc"].unique().to_list()),
                         pl.col("t_fill_ns").is_in(requested_times)
                         | pl.col("t_fill_ns").cast(pl.Float64).cast(pl.Int64).is_in(requested_times))
                 .select(*keys, pl.col("qc").alias("point_qc")).unique().collect())
        # Nullable exit columns become float64 in Replay.load_exits(). Preserve the
        # original keys as well as those exact conversions (up to 128 ns rounding).
        variants = [facts]
        for cols in (("t_fill_ns",), ("hedge_ns",), ("t_fill_ns", "hedge_ns")):
            variants.append(facts.with_columns(*(pl.col(c).cast(pl.Float64).cast(pl.Int64) for c in cols)))
        facts = pl.concat(variants).unique()
        matched.append(group.join(facts, on=keys, how="left", validate="m:1"))
    checks = pl.concat(matched)
    issues = checks.filter(pl.col("point_qc").is_null() | (pl.col("qc") != pl.col("point_qc")))
    issues.write_csv(out / "point_leg_issues.csv")
    return {name: dict(checked_legs=g.height, missing_price_or_time_match=g["point_qc"].null_count(),
                       wrong_contract_legs=g.filter(pl.col("qc") != pl.col("point_qc")).height)
            for (name,), g in checks.partition_by("run", as_dict=True).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["A_fixed", "B_dyn"])
    parser.add_argument("--out", default="audit_20260922")
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--points", action="store_true", help="also match all maker/hedge legs to Stage 2")
    args = parser.parse_args()
    out = DATA_ROOT / "backtest" / args.out
    summaries = [audit_run(name, out / name) for name in args.runs]
    comparisons = [compare_runs(*pair.split("=", 1)) for pair in args.compare]
    points = audit_point_legs(args.runs, out) if args.points else None
    files = list(Path(__file__).parents[1].rglob("*.py"))
    source_hashes = {str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    payload = dict(summaries=summaries, comparisons=comparisons, point_legs=points, source_sha256=source_hashes)
    (out / "verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(summaries=summaries, comparisons=comparisons, point_legs=points), indent=2))


if __name__ == "__main__":
    main()
