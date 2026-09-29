"""Recompute submitted-signal EV and describe mature-cohort prediction errors.

This does not retune A/B or imply that historical fit predicts future returns.
"""
import argparse
import hashlib
import json

import numpy as np
import polars as pl

from ..common.paths import DATA_ROOT
from ..ev import abs_reach, reach, ev
from ..ev.config import CostConfig
from .policy import PolicyConfig, Decider


def pmf_arithmetic():
    from ..tests.test_ev import FakeLookup
    rng = np.random.default_rng(20260923)
    cfg = CostConfig()
    worst = dict(probability=0., ev_bp=0., days=0., score=0.)
    for _ in range(1000):
        K = int(rng.integers(0, 31))
        c0, c1, q = rng.uniform(size=3)
        cumulative = max(c0, c1)
        residual = 1-cumulative
        pmf = [c0]
        if K:
            pmf.append(cumulative-c0)
            for j in range(2, K+1):
                pmf.append(residual*q)
                residual *= 1-q
            pmf.append(residual)
        else:
            pmf.append(1-c0)
        offsets = tuple(int(v) for v in np.cumsum(rng.integers(1, 4, K)))
        h = ev.Horizon(offsets, offsets[-1] if K else 0, offsets[-1] if K else 0)
        quote = ev.Quote("S1" if rng.random() < .5 else "S2", float(rng.uniform(50, 300)),
                         30., 20., 2., int(rng.integers(300, 14000)))
        result = ev.evaluate_exit(-.5, quote, h, FakeLookup(c0, c1, q), cfg)
        remaining = (15600-quote.t_sec)/86400
        durations = [remaining]+[o+remaining for o in offsets]+[h.settle_offset+remaining]
        target = quote.anchor-.5*quote.scale
        entry_loss = max(cfg.d_in_base[quote.stream], 0)+cfg.margin_bp
        target_gross = quote.quote_ab-entry_loss-target-3
        profits = [target_gross-20]+[target_gross-34]*K+[quote.quote_ab-entry_loss-3-(20 if K == 0 else 34)]
        expected_ev, expected_t = float(np.dot(pmf, profits)), float(np.dot(pmf, durations))
        for key, error in (("probability", abs(sum(pmf)-1)), ("ev_bp", abs(result.ev_bp-expected_ev)),
                           ("days", abs(result.t_days-expected_t)), ("score", abs(result.score-expected_ev/expected_t))):
            worst[key] = max(worst[key], error)
    return dict(cases=1000, max_error=worst, passed=all(v < 1e-9 for v in worst.values()))


def decision_errors(row, decision, hurdles):
    """Check formula outputs and actual per-portfolio hurdle outcomes."""
    errors = []
    route = decision.best.route if decision.best is not None else None
    if decision.admit != row["base_admit"] or route != row["route"]:
        errors.append("signal_decision")
    for key in ("score", "ev_bp", "t_days", "p_sd"):
        expected = getattr(decision.best, key) if decision.best is not None else None
        actual = row[key]
        if expected is None:
            if actual is not None:
                errors.append("signal_"+key)
        elif actual is None or not np.isfinite(expected) or not np.isfinite(actual) or abs(expected-actual) > 1e-8:
            errors.append("signal_"+key)
    for name, hurdle in hurdles.items():
        outcome = row[name]
        if outcome == "not_requested":
            continue
        if not decision.admit:
            if outcome != decision.reason:
                errors.append(name+"_invalid_rejection")
        elif decision.best is None:
            errors.append(name+"_missing_admission_score")
        elif decision.best.score < hurdle-1e-10:
            if outcome != "hurdle":
                errors.append(name+"_below_hurdle_admission")
        elif decision.best.score > hurdle+1e-10 and outcome == "hurdle":
            errors.append(name+"_false_hurdle_rejection")
    return errors


def signals(root, manifest):
    cfg_dict = manifest["configs"]["A"]
    cfg = PolicyConfig(**{**cfg_dict, "cost": CostConfig(**cfg_dict["cost"])})
    samples = abs_reach.load_samples()
    failures, daily = [], []
    hurdles = {name: dict(pl.read_csv(root/f"{name}_daily.csv", schema_overrides={"day":pl.String})
                         .select("day", "hurdle_used").iter_rows()) for name in manifest["configs"]}
    for day in manifest["days"]:
        path = root/f"Date={day}"/"decisions.parquet"
        source = pl.scan_parquet(path)
        submitted = pl.any_horizontal([pl.col(name) == "submitted" for name in manifest["configs"]])
        refresh = pl.col("origin") == "refresh" if "origin" in source.collect_schema().names() else pl.lit(False)
        ds = source.filter(submitted | refresh).collect()
        rt = reach.fit(day, 20)
        at = abs_reach.AbsTable.fit(samples, as_of=day)
        direct = Decider(day, rt if rt.days else None, at, cfg)
        if any(d >= day for d in rt.days):
            failures.append(dict(day=day, check="future_reach_training_day"))
        for r in ds.iter_rows(named=True):
            d = direct.decide(r)
            errors = decision_errors(r, d, {name: h[day] for name, h in hurdles.items()})
            failures.extend(dict(day=day, ns=r["ns"], check=error) for error in errors)
        daily.append(dict(day=day, submitted_signals=ds.filter(submitted).height, checked_signals=ds.height,
                          refresh_checks=ds.filter(refresh).height, reach_days=rt.days, absolute_as_of=at.as_of))
        print(json.dumps(dict(stage="ev_signals", **daily[-1])), flush=True)
    source = abs_reach.source_path()
    return dict(submitted_signals=sum(r["submitted_signals"] for r in daily),
                checked_signals=sum(r["checked_signals"] for r in daily),
                refresh_checks=sum(r["refresh_checks"] for r in daily), days=daily,
                absolute_source=str(source), absolute_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), failures=failures)


def calibration(root, manifest):
    last = manifest["days"][-1]
    records, scope = [], {}
    for name in manifest["configs"]:
        ps = pl.read_parquet(root/f"{name}_positions_all.parquet")
        paired = ps.filter(pl.col("hedge_ns").is_not_null())
        mature = paired.filter(pl.col("expiry") <= last)
        unresolved = mature.filter(pl.col("state") != "closed")
        scope[name] = dict(paired=paired.height, mature=mature.height, unresolved_mature=unresolved.height,
                           excluded_unmature=paired.height-mature.height,
                           failed_entry_cycles=ps.filter(pl.col("hedge_ns").is_null()).height)
        if unresolved.height:
            raise ValueError(f"{name}: mature inventory remains unresolved")
        mature = mature.with_columns((pl.col("pnl_net")/(pl.col("entry_spot_cash")/10000)*10000).alias("actual_bp"),
            ((pl.col("close_ns")-pl.col("hedge_ns"))/1e9/86400).alias("held_calendar_days"),
            (pl.col("quote_day") == pl.col("close_day")).cast(pl.Float64).alias("same_day"))
        groups = [("all", "all", mature)]
        for column in ("stream", "route"):
            groups += [(column, key[0], f) for key, f in mature.partition_by(column, as_dict=True).items()]
        for dimension, group, f in groups:
            if not f.height:
                continue
            nominal = f["entry_spot_cash"].to_numpy()/10000
            expected = f["ev_bp"].to_numpy()
            actual = f["actual_bp"].to_numpy()
            p_sd, sd = f["p_sd_pred"].to_numpy(), f["same_day"].to_numpy()
            records.append(dict(portfolio=name, dimension=dimension, group=group, n=f.height,
                expected_ev_bp=float(expected.mean()), realized_net_bp=float(actual.mean()),
                error_bp=float((actual-expected).mean()), predicted_net_twd=float(np.dot(nominal, expected)/10000),
                realized_net_twd=float(f["pnl_net"].sum()), predicted_days=float(f["t_days_pred"].mean()),
                realized_days=float(f["held_calendar_days"].mean()), predicted_same_day=float(p_sd.mean()),
                realized_same_day=float(sd.mean()), brier_same_day=float(np.mean((p_sd-sd)**2))))
    cohort_frame = pl.from_dicts(records) if records else pl.DataFrame(
        schema={"portfolio": pl.String, "dimension": pl.String, "group": pl.String, "n": pl.Int64})
    cohort_frame.write_csv(root/"ev_mature_cohorts.csv")
    return dict(scope=scope, cohorts=records, limitation=(
        "Conditional on completed entry hedges; failed entries are included in portfolio PnL, not this conditional EV cohort. "
        "Mature means contract expiry is within the observed period, avoiding completed-only selection. "
        "Market reach probability is not maker completion probability. Holding-time prediction uses an end-of-session proxy. "
        "Fixed research costs, public-print queue assumptions and no financing remain; no out-of-sample future-return guarantee."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-signals", action="store_true")
    args = ap.parse_args()
    root = DATA_ROOT/"backtest"/args.run
    manifest = json.loads((root/"manifest.json").read_text())
    verified = json.loads((root/"verification.json").read_text())
    if not verified["complete"] or verified["status"] != "PASS":
        raise ValueError("requires a complete passing portfolio audit")
    result = dict(pmf=pmf_arithmetic(), calibration=calibration(root, manifest))
    if not args.skip_signals:
        result["signals"] = signals(root, manifest)
    result["status"] = "PASS" if result["pmf"]["passed"] and not result.get("signals", {}).get("failures") else "FAIL"
    result["complete"] = not args.skip_signals
    (root/"ev_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
