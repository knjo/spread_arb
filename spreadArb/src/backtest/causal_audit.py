"""Independent accounting and resource audit of the corrected event replay.

Reads saved cash, capacity, signal and mark facts; does not call Actor accounting.
Use --partial while a full run is still running, --raw for selected raw-tick checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..common.paths import DATA_ROOT, open_ns, CLOSE_SECOND, SECOND
from .unreserved_audit import UnreservedAudit, spot_observations
from .refresh_audit import audit_refresh, audit_quote_prices, signal_grid, audit_decision_inputs

SIGNAL_KEYS = ["vc", "qc", "stream", "quote_ab", "anchor", "quote_second", "expiry", "scale", "scale_raw",
               "e_norm", "resid_mid_bp", "tick_bp_hedge", "execution_floor_bp", "spot_a1"]
RAW_DAYS = ["20260511", "20260601", "20260706", "20260803"]


def frame(path):
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def close_enough(x, y, tolerance=1e-6):
    return x is not None and y is not None and abs(x-y) <= tolerance


def entry_submissions(events):
    # A settlement-only day legitimately has no route column. A submit event
    # must still provide a route; do not silently accept malformed submissions.
    for event in events.iter_rows(named=True):
        if event["kind"] == "submit" and event["route"] in ("S1", "S2"):
            yield event


def audit(root, partial=False, raw=False, raw_all=False):
    manifest = json.loads((root / "manifest.json").read_text())
    available = sorted(p.parent.name[5:] for p in root.glob("Date=*/complete.json"))
    expected = manifest["days"]
    official = (root/"official_valuation.json").exists()
    if official:
        for name in manifest["configs"]:
            valued = pl.read_csv(root/f"{name}_daily_official.csv",schema_overrides={"day":pl.String})["day"].to_list()
            available = [d for d in available if d in valued]
    failures = []
    metrics = {}
    if not partial and available != expected:
        failures.append(dict(check="complete_period", actual=len(available), expected=len(expected)))
    # Report exact source drift, excluding unrelated tests/report scripts added later.
    execution_files = manifest.get("execution_sources") or {
        p: sha for p, sha in manifest["sources"].items() if "/ev/" in p or "/common/" in p
        or Path(p).name in {"causal_market.py", "causal_replay.py", "policy.py", "gates.py"}}
    for p in execution_files:
        actual = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if actual != manifest["sources"][p]:
            failures.append(dict(check="source_changed", file=p))
    all_positions = {}
    for name in manifest["configs"]:
        cfg = manifest["configs"][name]
        reserved = cfg.get("reserve_on_submit", True)
        unreserved = None if reserved else UnreservedAudit(cfg)
        cap = cfg["cap_twd"]
        daily_path = root / (f"{name}_daily_official.csv" if official else f"{name}_daily.csv")
        if not daily_path.exists():
            continue
        daily = pl.read_csv(daily_path, schema_overrides={"day": pl.String}).filter(pl.col("day").is_in(available))
        rows_daily = {r["day"]: r for r in daily.iter_rows(named=True)}
        balances, reservations = defaultdict(int), {}
        cash_by_id, extra_by_id, positions = defaultdict(lambda: defaultdict(int)), defaultdict(float), {}
        realized_total, prior_equity, prior_signals, prior_committed = 0.0, 0.0, [], 0.0
        release_times, reserve_times, cash_last = {}, {}, {}
        equity_rows = []
        counts = defaultdict(int)
        max_error, max_cap, max_product, last_ledger_ns = 0.0, 0.0, 0.0, 0

        def fail(check, day, **detail):
            failures.append(dict(portfolio=name, day=day, check=check, **detail))

        for day in available:
            folder = root / f"Date={day}" / name
            dr = rows_daily[day]
            ps = frame(folder / "positions.parquet")
            for p in ps.iter_rows(named=True):
                positions[p["id"]] = p
            cash = frame(folder / "cash_legs.parquet")
            for r in cash.iter_rows(named=True):
                pid, ins, side, qty, money = r["id"], r["instrument"], r["side"], r["qty"], r["cash"]
                asset = "spot" if ins.startswith("S:") else "future"
                p = positions[pid]
                if ins != ("S:"+p["vc"] if asset == "spot" else "F:"+p["qc"]):
                    fail("wrong_contract_cash_leg", day, id=pid, instrument=ins)
                c = cash_by_id[pid]
                c[asset+"_"+side+"_qty"] += qty
                c[asset+"_"+side+"_cash"] += money
                cash_last[pid] = max(r["ns"], cash_last.get(pid, 0))
                if r["purpose"].endswith("rollback"):
                    extra_by_id[pid] += (money / 1e8 * cfg["cost"]["fee_same_day_bp"] if asset == "spot"
                                        else 40.0 + money / 1e8 * .4)
                if r["liquidity"] == "maker":
                    if r["ns"] <= r["live_ns"]:
                        fail("pre_live_fill", day, id=pid)
                    counts["maker_legs"] += 1
                elif r["liquidity"] == "taker":
                    if r["book_ns"] > r["ns"]:
                        fail("future_hedge_book", day, id=pid)
                    counts["taker_legs"] += 1
            if cash.height and "liquidity" in cash.columns:
                makers = cash.filter(pl.col("liquidity") == "maker")
                if makers.height:
                    use = makers.group_by("instrument", "ns", "sequence").agg(
                        pl.col("qty").sum().alias("used"), pl.col("printed_qty").max().alias("printed"))
                    if use.filter(pl.col("used") > pl.col("printed")).height:
                        fail("printed_volume_reused", day)
            depth = frame(folder / "depth_claims.parquet")
            if depth.height and depth.filter(pl.col("used_qty") > pl.col("shown_qty")).height:
                fail("taker_depth_reused", day)
            for r in frame(folder / "ledger.parquet").iter_rows(named=True):
                pid, ns, delta = r["id"], r["ns"], r["delta_cents"]
                vc = pid.split("/")[2]
                if ns < last_ledger_ns:
                    fail("ledger_not_chronological", day, id=pid)
                last_ledger_ns = ns
                balances[vc] += delta
                if r["kind"] == "reserve":
                    if pid in reservations:
                        fail("duplicate_reservation", day, id=pid)
                    reservations[pid] = delta
                    reserve_times[pid] = ns
                    if balances[vc] > max(delta, round(cap*cfg["product_cap_frac"]*100)):
                        fail("product_cap", day, id=pid)
                elif r["kind"] == "release":
                    old = reservations.pop(pid, None)
                    if old is None or old != -delta:
                        fail("release_amount", day, id=pid)
                    release_times[pid] = ns
                else:
                    if reserved and pid not in reservations:
                        fail("missing_reservation", day, id=pid)
                    reservations[pid] = reservations.get(pid, 0) + delta
                committed = sum(balances.values())
                if committed != r["committed_cents"] or committed < 0 or (reserved and committed > round(cap*100)):
                    fail("capital_ledger", day, id=pid, calculated=committed, reported=r["committed_cents"])
                max_cap, max_product = max(max_cap, committed/100), max(max_product, balances[vc]/100)
                counts["ledger_events"] += 1
            events = frame(folder / "events.parquet")
            refreshed, errors = audit_refresh(events.to_dicts(), cash.to_dicts(), cfg,
                                             open_ns(day)+CLOSE_SECOND*SECOND)
            for key, value in refreshed.items():
                counts[key] += value
            failures.extend(dict(portfolio=name, day=day, **e) for e in errors)
            entry_sends = {}
            if events.height:
                for event in entry_submissions(events):
                    entry_sends[event["id"]] = event["ns"]
                    if reserved and reserve_times.get(event["id"]) != event["ns"]:
                        fail("entry_not_reserved_at_send", day, id=event["id"])
                counts["entry_submissions"] += len(entry_sends)
            for ledger_row in frame(folder / "ledger.parquet").iter_rows(named=True):
                if ledger_row["kind"] == "reserve" and entry_sends.get(ledger_row["id"]) != ledger_row["ns"]:
                    fail("reservation_without_submission", day, id=ledger_row["id"])
            for leg in cash.iter_rows(named=True) if reserved else ():
                pid = leg["id"]
                if reserve_times.get(pid, leg["ns"]+1) > leg["ns"] or release_times.get(pid, leg["ns"]) < leg["ns"]:
                    fail("cash_outside_reservation", day, id=pid)
            if not close_enough(sum(balances.values())/100, dr["committed_end"]):
                fail("daily_committed", day)
            if unreserved is not None:
                ev_rows = events.to_dicts()
                codes = {e["id"].split("/")[2] for e in ev_rows if e["kind"] == "submit" and e.get("route") == "S2"}
                observations = spot_observations(day, codes)
                failures.extend(dict(portfolio=name, **f) for f in unreserved.check_day(day,
                    {p["id"]: p for p in ps.iter_rows(named=True)}, ev_rows, cash.to_dicts(),
                    frame(folder / "ledger.parquet").to_dicts(), observations))
                if unreserved.amounts != reservations:
                    fail("independent_actual_capital_balances", day)
            opened = {p["id"] for p in ps.iter_rows(named=True) if p["state"] != "closed"}
            if opened != set(reservations) or len(opened) != dr["open_end"]:
                fail("open_position_reservation_coverage", day, opened=len(opened), reserved=len(reservations))
            realized_day = 0.0
            for p in ps.iter_rows(named=True):
                c, pid = cash_by_id[p["id"]], p["id"]
                for asset in ("spot", "future"):
                    for side in ("buy", "sell"):
                        for measure in ("cash", "qty"):
                            key = f"{asset}_{side}_{measure}"
                            if c[key] != p[key]:
                                fail("position_cash_or_quantity", day, id=pid, field=key)
                sq = c["spot_buy_qty"] - c["spot_sell_qty"]
                fq = c["future_sell_qty"] - c["future_buy_qty"]
                if p["close_day"] == day:
                    if sq or fq or cash_last.get(pid, 0) > release_times.get(pid, 0):
                        fail("released_with_exposure", day, id=pid)
                    fee_bp = cfg["cost"]["fee_same_day_bp"] if p["quote_day"] == day else cfg["cost"]["fee_overnight_bp"]
                    fee = (p["entry_spot_cash"]/1e8*fee_bp if p["hedge_ns"] is not None else 0) + extra_by_id[pid]
                    net = (c["spot_sell_cash"]-c["spot_buy_cash"]+c["future_sell_cash"]-c["future_buy_cash"])/10000-fee
                    err = abs(net-p["pnl_net"])
                    max_error = max(max_error, err)
                    if err > 1e-6:
                        fail("closed_pnl", day, id=pid, error=err)
                    realized_day += net
                    if p["close_day"] > p["expiry"]:
                        fail("after_expiry_close", day, id=pid, kind=p["close_kind"])
                if p["hedge_ns"] is not None and p["hedge_ns"] < p["fill_ns"]+50_000_000:
                    fail("hedge_before_latency", day, id=pid)
            if not close_enough(realized_day, dr["realized_pnl"]):
                fail("daily_realized", day, calculated=realized_day, reported=dr["realized_pnl"])
            realized_total += realized_day
            open_value = 0.0
            marks = frame(folder / ("marks_official.parquet" if official else "marks.parquet"))
            marked = marks["id"].to_list() if marks.height else []
            if set(marked) != opened or len(marked) != len(opened):
                fail("open_position_mark_coverage", day, opened=len(opened), marked=len(marked))
            for mark in marks.iter_rows(named=True):
                pid = mark["id"]
                p, c = positions[pid], cash_by_id[pid]
                if mark["value_twd"] is None:
                    fail("missing_mark", day, id=pid)
                    continue
                sq = c["spot_buy_qty"] - c["spot_sell_qty"]
                fq = c["future_sell_qty"] - c["future_buy_qty"]
                if (sq and mark["spot_mark"] is None) or (fq and mark["future_mark"] is None):
                    fail("missing_leg_mark", day, id=pid)
                if mark["spot_qty"] != sq or mark["future_qty"] != fq:
                    fail("mark_inventory", day, id=pid)
                fee = p["entry_spot_cash"]/1e8*cfg["cost"]["fee_overnight_bp"] + extra_by_id[pid]
                gross = (c["spot_sell_cash"]-c["spot_buy_cash"]+c["future_sell_cash"]-c["future_buy_cash"])/10000
                value = gross+(sq*(mark["spot_mark"] or 0)-fq*p["shares"]*(mark["future_mark"] or 0))/10000-fee
                if not close_enough(value, mark["value_twd"]):
                    fail("open_mark", day, id=pid, calculated=value, reported=mark["value_twd"])
                open_value += value
                counts["official_future_marks" if mark["future_source"] == "official_future_daily" else "bbo_future_marks"] += 1
            equity = realized_total + open_value
            if not close_enough(equity, dr["equity_twd"]) or not close_enough(equity-prior_equity, dr["mtm_pnl"]):
                fail("daily_equity", day, calculated=equity, reported=dr["equity_twd"])
            base = cfg["cost"]["hurdle_bp_per_day"]
            hurdle = base
            if cfg["dyn_q"] is not None and prior_signals and prior_committed >= cfg["dyn_cap_frac"]*cap:
                hurdle = max(base, float(np.quantile(prior_signals, cfg["dyn_q"])))
            if not close_enough(hurdle, dr["hurdle_used"], 1e-10):
                fail("dynamic_hurdle", day, calculated=hurdle, reported=dr["hurdle_used"])
            signals = frame(root / f"Date={day}" / "base_signal_scores.parquet")["score"].to_list()
            if name == "A":
                ds = frame(root / f"Date={day}" / "decisions.parquet")
                if ds.height:
                    population = ds.filter(pl.col("origin") == "market") if "origin" in ds.columns else ds
                    rebuilt = population.unique(subset=SIGNAL_KEYS, maintain_order=True).filter(pl.col("base_admit"))["score"].to_list()
                    if len(rebuilt) != len(signals) or not np.allclose(rebuilt, signals, atol=1e-12, rtol=0):
                        fail("independent_signal_population", day, calculated=len(rebuilt), reported=len(signals))
                    if ds.filter(pl.col("base_admit") & ((pl.col("anchor") < 0) | (pl.col("scale_raw") > 60))).height:
                        fail("anchor_or_scale_admission", day)
            equity_rows.append(dict(day=day, realized_twd=realized_day, open_value_twd=open_value,
                                    equity_twd=equity, mtm_pnl=equity-prior_equity, hurdle=hurdle))
            prior_equity, prior_signals, prior_committed = equity, signals, dr["committed_end"]
        if equity_rows:
            reconciled = pl.from_dicts(equity_rows)
            reconciled.write_csv(root / f"{name}_reconciled.csv")
            eq = np.r_[0., reconciled["equity_twd"].to_numpy()]
            daily_pnl = reconciled["mtm_pnl"].to_numpy()
            total, n = float(eq[-1]), len(equity_rows)
            metrics[name] = dict(days=n, pnl_twd=total, daily_twd=total/n,
                annual_simple_pct=total/n*250/cap*100, closed_pnl_twd=realized_total,
                terminal_mark_twd=equity_rows[-1]["open_value_twd"],
                paired_entries=sum(p["hedge_ns"] is not None for p in positions.values()),
                rollback_cycles=sum(p["close_kind"] == "rollback" for p in positions.values()),
                max_cash_pnl_error_twd=max_error, capital_peak_twd=max_cap, max_product_twd=max_product,
                mtm_drawdown_twd=float(np.min(eq-np.maximum.accumulate(eq))),
                mtm_sharpe=float(daily_pnl.mean()/daily_pnl.std(ddof=1)*np.sqrt(250)) if n>1 and daily_pnl.std(ddof=1)>0 else None,
                raised_days=int((daily["hurdle_used"] > base+1e-10).sum()),
                unhedged_days=int((daily["unhedged_end"] > 0).sum()), **counts,
                **(dict(unreserved.counts) if unreserved is not None else {}))
            all_positions[name] = positions
            pl.from_dicts(list(positions.values()), infer_schema_length=None).write_parquet(root/f"{name}_positions_all.parquet")
    if raw or raw_all:
        raw_checks, raw_failures = raw_audit(root, available if raw_all else [d for d in RAW_DAYS if d in available])
        metrics["raw"] = raw_checks
        failures += raw_failures
    result = dict(complete=available == expected, days=len(available), expected_days=len(expected), official_valuation=official,
                  status="FAIL" if failures else "PASS", metrics=metrics, failures=failures)
    (root / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def raw_fingerprint(root, day):
    from ..common.books import raw_paths
    from ..common.paths import mapping_path, grid_path
    parts = []
    for p in [Path(__file__), Path(__file__).with_name("fifo_audit.py"), Path(__file__).with_name("refresh_audit.py"),
              Path(__file__).parents[1]/"common/books.py", mapping_path(day), root/"manifest.json"]:
        parts.append((str(p), hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in ("A", "B"):
        for filename in ("positions.parquet", "events.parquet", "cash_legs.parquet"):
            p = root/f"Date={day}"/name/filename
            parts.append((str(p), hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None))
    for p in raw_paths(day):
        stat = p.stat()
        parts.append((str(p), stat.st_size, stat.st_mtime_ns))
    for p in (grid_path(day), root/f"Date={day}"/"decisions.parquet"):
        if p.exists():
            stat = p.stat()
            parts.append((str(p), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def raw_audit(root, days):
    from ..common.books import load_session
    from ..common.paths import mapping_path
    from .fifo_audit import submitted_orders, audit_fifo
    manifest = json.loads((root/"manifest.json").read_text())
    accepted_policy = "execution_sources" in manifest
    refresh_policy = all("quote_refresh_ns" in cfg for cfg in manifest["configs"].values())
    failures, counts = [], defaultdict(int)
    for day in days:
        fingerprint = raw_fingerprint(root, day)
        cache_path = root/f"Date={day}"/"raw_verification.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get("fingerprint") == fingerprint:
                for k, v in cached["counts"].items():
                    counts[k] += v
                failures.extend(cached["failures"])
                print(json.dumps(dict(stage="raw_audit_cached", day=day)), flush=True)
                continue
        before_counts, first_failure = dict(counts), len(failures)
        frames = {n: frame(root/f"Date={day}"/n/"cash_legs.parquet") for n in ("A", "B")}
        event_frames = {n: frame(root/f"Date={day}"/n/"events.parquet") for n in frames}
        pos_frames = {n: {p["id"]: p for p in frame(root/f"Date={day}"/n/"positions.parquet").iter_rows(named=True)} for n in frames}
        mapping = dict(pl.read_parquet(mapping_path(day)).select("ValueCode", "QuoteCode").iter_rows())
        decisions = []
        dp = root/f"Date={day}"/"decisions.parquet"
        if refresh_policy and dp.exists():
            decisions = (pl.scan_parquet(dp).filter((pl.col("origin") == "refresh") |
                pl.any_horizontal([pl.col(n) == "submitted" for n in manifest["configs"]])).collect().to_dicts())
        ins = {v for f in frames.values() if f.height for v in f["instrument"].unique()}
        for r in decisions:
            ins.update(("S:"+r["vc"], "F:"+r["qc"]))
        if accepted_policy:
            for name in frames:
                ins.update(o["instrument"] for o in submitted_orders(event_frames[name].to_dicts(), pos_frames[name], mapping).values())
        books, prints, _, _ = load_session(day, sorted(v[2:] for v in ins if v.startswith("S:")),
                                          sorted(v[2:] for v in ins if v.startswith("F:")))
        if refresh_policy:
            checked, errors = audit_decision_inputs(decisions, books, signal_grid(day, decisions), open_ns(day))
            for k, v in checked.items():
                counts[k] += v
            failures.extend(dict(day=day, **e) for e in errors)
        for name, cash in frames.items():
            used = defaultdict(int)
            pos = pos_frames[name]
            if accepted_policy:
                checked, errors = audit_fifo(event_frames[name].to_dicts(), cash.to_dicts(), pos, mapping, books, prints)
                for k, v in checked.items():
                    counts[k] += v
                failures.extend(dict(portfolio=name, day=day, **e) for e in errors)
                checked, errors = audit_quote_prices(event_frames[name].to_dicts(),
                    submitted_orders(event_frames[name].to_dicts(), pos, mapping), books)
                for k, v in checked.items():
                    counts[k] += v
                failures.extend(dict(portfolio=name, day=day, **e) for e in errors)
            for r in cash.iter_rows(named=True):
                if r["liquidity"] == "accounting":
                    continue
                counts["checked_legs"] += 1
                instrument, ns = r["instrument"], r["ns"]
                if r["liquidity"] == "maker":
                    counts["maker_legs"] += 1
                    pr = prints.get(instrument)
                    lo, hi = np.searchsorted(pr.ns, ns, side="left"), np.searchsorted(pr.ns, ns, side="right")
                    ix = [i for i in range(lo, hi) if int(pr.seq[i]) == r["sequence"]]
                    valid = len(ix) == 1
                    if valid:
                        i = ix[0]
                        valid = (int(pr.qty[i]) == r["printed_qty"] and
                                 (pr.price[i] <= r["limit_price"] if r["side"] == "buy" else pr.price[i] >= r["limit_price"]))
                    maker_cash = r["qty"]*r["limit_price"]*(pos[r["id"]]["shares"] if instrument.startswith("F:") else 1)
                    valid = valid and maker_cash == r["cash"]
                    if not valid:
                        failures.append(dict(check="raw_maker", portfolio=name, day=day, id=r["id"], ns=ns))
                else:
                    counts["taker_legs"] += 1
                    b = books[instrument]
                    i = b.index_at(ns)
                    if int(b.ns[i]) != r["book_ns"] or int(b.seq[i]) != r["book_seq"]:
                        failures.append(dict(check="raw_hedge_snapshot", portfolio=name, day=day, id=r["id"]))
                    left, amount = r["qty"], 0
                    for price, qty in b.levels(i, "ask" if r["side"] == "buy" else "bid"):
                        key = (instrument, int(b.ns[i]), int(b.seq[i]), r["side"], price)
                        take = min(left, qty-used[key])
                        amount += take*price
                        used[key] += take
                        left -= take
                        if not left:
                            break
                    if instrument.startswith("F:"):
                        amount *= pos[r["id"]]["shares"]
                    if left or amount != r["cash"]:
                        failures.append(dict(check="raw_hedge_cash_or_depth", portfolio=name, day=day,
                                             id=r["id"], calculated=amount, reported=r["cash"]))
        counts["days"] += 1
        day_result = dict(day=day, fingerprint=fingerprint,
            counts={k: v-before_counts.get(k, 0) for k, v in counts.items()}, failures=failures[first_failure:])
        cache_path.write_text(json.dumps(day_result, indent=2)+"\n")
        print(json.dumps(dict(stage="raw_audit_complete", day=day, failures=len(day_result["failures"]))), flush=True)
        del books, prints
        from .runtime import release_memory
        release_memory()
    return dict(counts, dates=days), failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="corrected_20260922_v2")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--raw-all", action="store_true")
    args = ap.parse_args()
    result = audit(DATA_ROOT/"backtest"/args.run, args.partial, args.raw, args.raw_all)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
