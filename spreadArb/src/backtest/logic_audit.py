"""Executable document-contract probes for the frozen point replay.

This audits the current implementation without changing it. FAIL means the
documented invariant is violated; it is not an expected-failure unit test.
Run with uv run --project /home/kevin/Project/HFT --no-sync python -m
spreadArb.src.backtest.logic_audit. Exit code 1 indicates failed contracts.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl

from ..common.paths import DATA_ROOT, SECOND, open_ns
from ..ev import ev, qlevel
from ..ev.config import CostConfig
from ..points import s1, s2
from ..points.common import sweep_hedge
from ..tests.test_backtest import exits_table, position
from ..tests.test_ev import FakeLookup
from ..tests.test_points_s1 import inputs as s1_inputs
from ..tests.test_points_s2 import books, inputs as s2_inputs, prints
from . import legacy_replay as engine
from .audit import load, reconcile_daily
from .policy import Decider, PolicyConfig, decide_row, exit_target_bp

DAY = "20260706"
T0 = open_ns(DAY)


def ns(seconds: float) -> int:
    return T0 + round(seconds * SECOND)


def result(key: str, passed: bool, spec: str, **evidence) -> dict:
    return dict(check=key, status="PASS" if passed else "FAIL", spec=spec, evidence=evidence)


def row(vc="2330", quote=300.0, fill=700.0, **updates):
    r = dict(stream="S2", vc=vc, qc="CDFG6", expiry="20260715", quote_ns=ns(quote),
             quote_second=int(quote), price=1_010_000, level=0, anchor=30.0, scale=20.0,
             quote_ab=100.0, eff_u=70.0, resid_mid_bp=40.0, e_norm=2.0,
             spot_b1=1_000_000, spot_a1=1_000_000, tick_bp_hedge=10.0,
             depth_ahead=0, opp_depth_shares=20_000, opp_depth_lots=None,
             notional_twd=200_000.0, t_partial_ns=None,
             t_fill_ns=ns(fill) if fill is not None else None,
             hedge_ns=ns(fill + 0.05) if fill is not None else None,
             hedge_vwap=1_000_000 if fill is not None else None, actual_ab=100.0,
             d_in_realized=0.0, t_ab0_ns=None, t_gate_ns=ns(800.0),
             **{f"t_below_{floor}": None for floor in s2.FLOORS})
    r.update(updates)
    return r


class AlwaysAdmit:
    """Isolate portfolio execution from statistical admission in synthetic probes."""
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def decide(self, r):
        value = ev.ExitEval(None, None, 0.0, 0.0, 1.0, 50.0, 2.0, 25.0, 30.0, 0.0, 0.0, route="settle")
        return ev.Decision(True, "ok", value, (value,))

    def admitted_scores(self, base):
        return [25.0]


def run_synthetic(rows, *, cfg=None, carried=None, gates=None, marks=None):
    replay = engine.Replay(cfg or PolicyConfig(cap_twd=300_000, product_cap_frac=1.0),
                           Path("/tmp/spreadarb-logic-probe-unused"), samples=pl.DataFrame())
    for p in carried or []:
        replay.positions[p.id] = p
        replay.open_ids.add(p.id)
        replay.committed += p.notional_twd
        replay.committed_by_vc[p.vc] = replay.committed_by_vc.get(p.vc, 0.0) + p.notional_twd
    with contextlib.ExitStack() as stack:
        replacements = dict(load_entries=pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame(),
                            load_exits={}, close_marks=marks or {})
        for name, value in replacements.items():
            stack.enter_context(patch.object(engine, name, return_value=value))
        stack.enter_context(patch.object(engine.reach, "fit", return_value=SimpleNamespace(days=[])))
        stack.enter_context(patch.object(engine.abs_reach.AbsTable, "fit", return_value=None))
        stack.enter_context(patch.object(engine, "Decider", AlwaysAdmit))
        if gates is not None:
            stack.enter_context(patch.object(engine, "GateBook", return_value=gates))
        daily = replay.run_day(DAY, False)
    return replay, daily


def portfolio_probes():
    a, _ = run_synthetic([row("1111", fill=700), row("2222", quote=400, fill=500)])
    b, _ = run_synthetic([row("1111", fill=None), row("2222", quote=400, fill=500)])
    admitted_a = any(p.vc == "2222" for p in a.positions.values())
    admitted_b = any(p.vc == "2222" for p in b.positions.values())
    yield result("capacity_future_outcome_invariance", admitted_a == admitted_b,
                 "03_BACKTEST §2: reserve at submission using observable prices",
                 second_order_with_future_fill=admitted_a, second_order_without_future_fill=admitted_b,
                 identical_information_until_second_order_s=400, first_order_outcome_s=700)

    a, d = run_synthetic([row(fill=400, hedge_ns=None, hedge_vwap=None)])
    yield result("retain_unhedged_maker_fill", len(a.positions) > 0,
                 "00 §4; 03 §3: retain mandatory hedge exposure, never delete the fill",
                 maker_fill_s=400, hedge_available=False, recorded_positions=len(a.positions), recorded_fills=d["filled"])

    p = position(None, quote_day="20260703")
    p.expiry = DAY
    a, _ = run_synthetic([], carried=[p])
    yield result("expiry_missing_mark_is_explicit", p.state != "open",
                 "03_RESULTS v1: expiry-day settlement; 03 §3: explicit unresolved status",
                 state=p.state, close_kind=p.close_kind, still_open=p.id in a.open_ids)

    p = position(0.0)
    p.qc = "OLD_CONTRACT"
    table = exits_table([dict(quote_ns=700.0, price=1_010_000, level=0, quote_ab=-5.0,
                              t_fill_ns=800.0, hedge_ns=800.05, hedge_vwap=1_009_000, actual_ab=-9.9)])
    table[p.vc]["qc"] = np.array(["NEW_CONTRACT"])
    replay = engine.Replay(PolicyConfig(), Path("/tmp/unused"), samples=pl.DataFrame())
    plan = replay.plan_exit(p, table, DAY, T0, ns(600))
    yield result("exit_contract_identity", plan is None, "03 §3: same contract, explicit contract conversion",
                 held_contract=p.qc, exit_contract="NEW_CONTRACT", accepted=plan is not None)

    class Gate:
        def next_bad(self, leg, vc, t):
            return max(t, ns(300.02))
    cfg = PolicyConfig(cap_twd=300_000, product_cap_frac=1.0, gates_tag="synthetic")
    a, d = run_synthetic([row(fill=300.06)], cfg=cfg, gates=Gate())
    yield result("placement_in_flight_cancel_race", len(a.positions) > 0,
                 "00 §4: 50 ms placement/cancel, all unavoidable race fills retained; conflicts with v1 skip approximation",
                 send_s=300, gate_bad_s=300.02, live_s=300.05, fill_s=300.06, cancel_effective_s=300.07,
                 recorded_positions=len(a.positions), gate_rejects=d["gate_rejects"])

    r = row(stream="S1", price=1_000_000, fill=None, t_partial_ns=ns(400), t_gate_ns=ns(500))
    a, _ = run_synthetic([r])
    observed = a.rollbacks[0]["cost_twd"]
    # The current row has only quote-time Bid. Two later tapes with different
    # rollback-time bids are indistinguishable to the implementation.
    expected_on_lower_bid = 1000 * (99.0 - 100.0) - 100_000 * 20 / 1e4
    yield result("rollback_uses_cancel_time_cash", abs(observed - expected_on_lower_bid) < 1e-9,
                 "02 §4.2c: rollback at cancellation-time spot Bid; 00 §4: actual taker VWAP",
                 quote_bid=100.0, rollback_bid_scenario=99.0, observed_twd=observed,
                 required_scenario_twd=expected_on_lower_bid, later_bid_not_an_engine_input=True)


def point_probes():
    spot = books("S:2330", [(0, 400_000, 1000, 400_500, 20_000, True)])
    fut = books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True)])
    pr = prints("S:2330", [(300.02, 400_000, 2000), (300.10, 400_000, 1000), (300.20, 400_000, 1000)])
    rows = s1.product_rows(s1_inputs(spot, fut, pr))
    candidate = next(r for r in rows if r["side"] == "buy" and r["level"] == 0)
    live = candidate["quote_ns"] + 50_000_000
    post_live_volume = int(pr.qty[(pr.ns > live) & (pr.ns <= candidate["t_fill_ns"])].sum())
    a, _ = run_synthetic([candidate])
    yield result("s1_print_volume_after_placement", not a.positions or post_live_volume >= 2000,
                 "00 §4: prints at/before order placement cannot fill it",
                 live_s=(live - T0) / SECOND, modeled_full_fill_s=(candidate["t_fill_ns"] - T0) / SECOND,
                 shares_printed_after_live=post_live_volume, booked_shares=sum(p.shares for p in a.positions.values()))

    spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True)])
    fut = books("F:CDFG6", [(0, 401_500, 3, 403_000, 2, True)])
    pr = prints("F:CDFG6", [(400.0, 402_000, 1), (600.0, 402_000, 1)])
    p = s2_inputs(spot, fut, pr, anchor_bp=60.0)
    buys = [r for r in s2.product_rows(p) if r["side"] == "buy"]
    later = [r for r in buys if r["t_fill_ns"] == ns(600)]
    yield result("e2_same_price_later_print_available", bool(later),
                 "02 §4.2b, §4.2d: segments support requoting and each later position's exit",
                 eligible_print_seconds=[400, 600], generated_fill_seconds=[(r["t_fill_ns"] - T0) / SECOND for r in buys])

    fut_late = books("F:CDFG6", [(0, 401_500, 3, 402_000, 2, True),
                                 (14400, 401_500, 3, 403_000, 2, True)])
    late = [r for r in s2.product_rows(s2_inputs(spot, fut_late,
             prints("F:CDFG6", [(15000.0, 402_000, 1)]), anchor_bp=60.0)) if r["side"] == "buy"]
    yield result("exit_quotes_continue_until_1318", bool(late), "02 §3: exits 09:05 until 13:18 withdrawal",
                 legal_inside_exit_first_available="13:00", eligible_print="13:10", exit_rows=len(late),
                 actual_candidate_cutoff="12:53:20 for both sides")

    absolute = s2_inputs(spot, books("F:CDFG6", [(0, 403_000, 3, 404_500, 2, True)]), None, anchor_bp=85.0)
    sells = [r for r in s2.product_rows(absolute) if r["side"] == "sell"]
    basis = (404_000 / 400_500 - 1) * 1e4
    yield result("absolute_route_low_residual_supply", bool(sells),
                 "01 §4b and 03_RESULTS: absolute route admits high absolute basis with small residual",
                 quote_ab=basis, eff_u=basis - 85.0, absolute_threshold=50, entry_rows=len(sells), stage2_floor=10)

    thin_spot = books("S:2330", [(0, 400_000, 5000, 400_500, 20_000, True),
                                 (350, 400_000, 5000, 400_500, 1000, True),
                                 (401, 400_000, 5000, 400_500, 20_000, True)])
    r = next(r for r in s2.product_rows(s2_inputs(thin_spot, fut,
             prints("F:CDFG6", [(400, 402_500, 1)]), anchor_bp=20.0)) if r["side"] == "sell")
    yield result("s2_depth_deterioration_cancels", r["t_gate_ns"] is not None and r["t_gate_ns"] <= ns(350),
                 "02 §2 S2: cancel on depth failure; 00 §3: unavailable hedge closes route",
                 depth_at_submission=20_000, depth_at_350s=1000, required_shares=2000,
                 cancel_trigger_ns=r["t_gate_ns"], maker_fill_s=400, hedge_s=(r["hedge_ns"] - T0) / SECOND)

    book = books("F:CDFG6", [(0, 400_000, 1, 400_500, 1, True)])
    first = sweep_hedge(book, "bid", 1, ns(400), 50_000_000, 5 * SECOND, 1, 10**9, ns(15600))
    second = sweep_hedge(book, "bid", 1, ns(400), 50_000_000, 5 * SECOND, 1, 10**9, ns(15600))
    yield result("portfolio_shared_hedge_depth", second is None,
                 "00 §4 / 03 §1: one snapshot depth cannot be consumed twice; Stage2 independence is only an intermediate approximation",
                 available_lots=1, hedge_requests=2, filled_requests=int(first is not None) + int(second is not None))


def decision_probes():
    cfg = PolicyConfig()
    r = row(quote_ab=80.0, anchor=0.1)
    dec = Decider(DAY, FakeLookup(c0=1, c1=1), None, cfg)
    dec.decide(r)
    negative = {**r, "anchor": -0.1}
    cached, direct = dec.decide(negative), decide_row(negative, DAY, FakeLookup(c0=1, c1=1), None, cfg)
    yield result("cache_preserves_negative_anchor_gate", cached.admit == direct.admit,
                 "01 §3: reject anchor < 0",
                 cached_admit=cached.admit, direct_admit=direct.admit, direct_reason=direct.reason)

    class StateLookup(FakeLookup):
        def lookup_c0(self, e, x, t_sec, k_days):
            return SimpleNamespace(n=500, p=1.0 if e < 2 else 0.0, level=0)
        def lookup_c1(self, e, x, k_days):
            return SimpleNamespace(n=500, p=1.0 if e < 2 else 0.0, level=0)
        def lookup_q(self, x, k_days, e=None):
            return SimpleNamespace(n=500, p=0.0, level=0)
    lk = StateLookup()
    r = row(quote_ab=80.0, e_norm=1.0)
    dec = Decider(DAY, lk, None, cfg)
    dec.decide(r)
    changed = {**r, "e_norm": 5.0}
    cached, direct = dec.decide(changed), decide_row(changed, DAY, lk, None, cfg)
    yield result("cache_includes_q_state", abs(cached.best.score - direct.best.score) < 1e-9,
                 "01 §2: Q conditions on decision-time mid residual e",
                 cached_score=cached.best.score, direct_score=direct.best.score)

    dec = Decider(DAY, FakeLookup(c0=1, c1=1), None, cfg)
    dec.decide(row(vc="1111"))
    dec.decide(row(vc="2222"))
    yield result("dynamic_signal_identity_includes_product", len(dec.admitted_scores()) == 2,
                 "03_RESULTS v5: independent signals = product x minute x basis",
                 distinct_products=2, histogram_observations=len(dec.admitted_scores()))

    hists = pl.DataFrame(dict(day=["20260703", "20260703"], ValueCode=["2330", "2330"],
                             bin=[300, 380], count=[50, 50]), schema=qlevel.HIST_SCHEMA)
    wide = qlevel.table_from_hists(hists, DAY).row(0, named=True)
    r = row(scale=wide["scale"])
    d = decide_row(r, DAY, FakeLookup(c0=1, c1=1), None, cfg)
    yield result("excluded_wide_scale_is_excluded", not d.admit, "01 §3.2: scale_raw > 60 must not enter decisions",
                 source_scale_raw=wide["scale_raw"], input_scale=wide["scale"], admit=d.admit, fallback_scale=1.0)

    costly_tick = row(price=10_100, spot_a1=10_000, spot_b1=9_900, notional_twd=2000.0, tick_bp_hedge=100.0)
    d = decide_row(costly_tick, DAY, FakeLookup(c0=1, c1=1), None, cfg)
    yield result("s2_half_tick_execution_cost_floor", all(v.d_in >= 50.0 for v in d.evals),
                 "01 §3.3 / 00 §6: S2 d_in=max(empirical, depth + half stock tick)",
                 tick_bp=100.0, required_floor_bp=50.0, modeled_d_in=[v.d_in for v in d.evals])

    horizon = ev.horizon("20260715", "20260715")
    value = ev.evaluate_exit(ev.SETTLE, ev.Quote("S2", 100, 30, 20, 2, 3600), horizon, None, CostConfig())
    yield result("settlement_forecast_matches_replay_day", value.t_days < 1.0,
                 "03_RESULTS v1: settle at expiry-day close (supersedes old next-session execution rule)",
                 decision_day="20260715", expiry="20260715", forecast_days=value.t_days,
                 execution_remaining_days=(15600 - 3600) / 86400, old_spec_next_session=True)

    max_error = max(abs(ev.branches(c0, c1, q, k).p_sd + ev.branches(c0, c1, q, k).p_on
                        + ev.branches(c0, c1, q, k).p_never - 1)
                    for c0 in (0, .4, 1) for c1 in (0, .7, 1) for q in (0, .3, 1) for k in (0, 1, 5, 20))
    yield result("ev_probability_partition", max_error < 1e-12, "01 §7: all branch probabilities sum to one",
                 combinations=324, max_error=max_error)

    best = ev.ExitEval(0.0, 0.0, .5, .4, .1, 30, 1, 30, 20, 10, 3, route="absolute")
    residual = replace(best, route="residual", x=-0.5)
    target = exit_target_bp(best, 50.0, 20.0, (residual,), "max_q")
    yield result("frozen_absolute_exit_target", abs(target - 40) < 1e-9, "03_RESULTS v3/A/B: max(0, Q target)",
                 target_bp=target, expected_bp=40.0)
    yield result("absolute_ev_matches_applied_exit_target", abs(best.b_x_bp - target) < 1e-9,
                 "03 §6: changed exit policy requires matching labels/EV; A/B max_q currently changes only execution target",
                 ev_assumed_exit_basis_bp=best.b_x_bp, executed_target_bp=target,
                 score_retained_from_zero_exit=best.score)


def cli_probes():
    with patch.object(sys, "argv", ["replay", "--preset", "B", "--exit-routes", "E1",
                                     "--abs-target", "zero", "--dyn-cap-frac", "0"]), \
            patch.object(engine, "grid_days", return_value=[DAY]), patch.object(engine, "Replay") as replay:
        engine.main()
        cfg = replay.call_args.args[0]
    yield result("explicit_cli_flags_override_preset", cfg.exit_routes == ("E1",)
                 and cfg.abs_target_mode == "zero" and abs(cfg.dyn_cap_frac) < 1e-12,
                 "replay --help: explicit flags override preset fields",
                 requested=dict(exit_routes=["E1"], abs_target="zero", dyn_cap_frac=0),
                 actual=dict(exit_routes=cfg.exit_routes, abs_target=cfg.abs_target_mode, dyn_cap_frac=cfg.dyn_cap_frac))


def recorded_probes(out: Path):
    for name in ("A_fixed", "B_dyn"):
        p, daily, rb, cfg = load(name)
        negative = p.filter(pl.col("anchor") < 0)
        negative.write_csv(out / f"{name}_negative_anchor.csv")
        yield result(name + "_actual_negative_anchor", not negative.height, "01 §3: reject negative anchor",
                     count=negative.height, ids=negative["id"].to_list(), pnl_twd=negative["pnl_net"].sum())
        small = p.filter((pl.col("scale") - 1).abs() < 1e-9)
        frames = []
        for (day,), g in small.partition_by("quote_day", as_dict=True).items():
            tab = qlevel.table(day).select(pl.col("ValueCode").alias("vc"),
                                          pl.col("scale").alias("asof_scale"), "scale_raw")
            frames.append(g.join(tab, on="vc", how="left"))
        invalid = pl.concat(frames).filter(pl.col("asof_scale").is_null()) if frames else small.clear()
        invalid.write_csv(out / f"{name}_invalid_scale.csv")
        wide = invalid.filter(pl.col("scale_raw") > 60) if invalid.height else invalid
        yield result(name + "_actual_invalid_scale", not wide.height, "01 §3.2: scale_raw > 60 excluded",
                     null_fallback_count=invalid.height, fallback_pnl_twd=invalid["pnl_net"].sum(),
                     wide_over_60=wide.height, wide_pnl_twd=wide["pnl_net"].sum(), wide_ids=wide["id"].to_list(),
                     note="Missing warmup scales are separate; the explicit exclusion applies to scale_raw > 60.")
        rec = reconcile_daily(p, daily, rb)
        gap = rec["reported_daily_twd"].sum() - rec["reconciled_booked_twd"].sum()
        yield result(name + "_daily_total_conservation", abs(gap) < 1e-8, "03 §4: daily totals reconcile to net equity",
                     daily_minus_ledger_twd=gap)


def main():
    out = DATA_ROOT / "backtest" / "logic_audit_20260922"
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for probe in (portfolio_probes, point_probes, decision_probes, cli_probes):
        try:
            checks.extend(probe())
        except Exception as exc:
            checks.append(dict(check=probe.__name__, status="ERROR", error=repr(exc)))
    checks.extend(recorded_probes(out))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parents[1] / "tests"),
                                              top_level_dir=str(Path(__file__).parents[3]))
    stream = io.StringIO()
    tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / "existing_unit_tests.txt").write_text(stream.getvalue())
    root = Path(__file__).parents[2]
    paths = list((root / "src").rglob("*.py")) + list((root / "doc").glob("*.md"))
    payload = dict(checks=checks, counts={s: sum(c["status"] == s for c in checks) for s in ("PASS", "FAIL", "ERROR")},
                   existing_tests=dict(count=tested.testsRun, failures=len(tested.failures), errors=len(tested.errors)),
                   source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "verification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for item in checks:
        print(json.dumps(item, ensure_ascii=False))
    print(json.dumps(dict(counts=payload["counts"], existing_tests=payload["existing_tests"])))
    raise SystemExit(1 if payload["counts"]["FAIL"] or payload["counts"]["ERROR"] or not tested.wasSuccessful() else 0)


if __name__ == "__main__":
    main()
