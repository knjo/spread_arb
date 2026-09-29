"""Current Stage 3 CLI: raw chronological execution for one policy.

The retired point-label engine is legacy_replay.py. For a shared market pass of
both presets use `python -m spreadArb.src.backtest.causal_replay`.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..common.paths import DATA_ROOT, grid_days
from ..ev.config import CostConfig
from .policy import PolicyConfig, PRESETS
from .causal_replay import Actor, Replay
from .runtime import init_run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", default="corrected")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="B")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", type=int, choices=(0,), default=0)
    reservation = parser.add_mutually_exclusive_group()
    reservation.add_argument("--reserve-on-submit", action="store_true", default=False)
    reservation.add_argument("--no-reserve-on-submit", dest="reserve_on_submit", action="store_false")
    fields = {"--hurdle": ("hurdle_bp_per_trading_day", float), "--streams": ("streams", str),
              "--residual-min": ("residual_min_bp", float), "--exit-routes": ("exit_routes", str),
              "--gates": ("gates_tag", str), "--abs-target": ("abs_target_mode", str),
              "--dyn-q": ("dyn_q", float), "--dyn-window": ("dyn_window", int),
              "--dyn-cap-frac": ("dyn_cap_frac", float), "--dyn-source": ("dyn_source", str)}
    for flag, (dest, type_) in fields.items():
        parser.add_argument(flag, dest=dest, type=type_, default=argparse.SUPPRESS)
    for flag in ("--d-in-s1", "--d-in-s2", "--d-out"):
        parser.add_argument(flag, type=float)
    args = vars(parser.parse_args(argv))
    if args["prefetch"] < 0:
        parser.error("prefetch must be nonnegative")
    options = {k: args.pop(k) for k in ("start", "end", "out", "preset", "resume", "prefetch")}
    cost = CostConfig()
    ins, outs = dict(cost.d_in_base), dict(cost.d_out_base)
    for flag, stream in (("d_in_s1", "S1"), ("d_in_s2", "S2")):
        value = args.pop(flag)
        if value is not None:
            ins[stream] = value
    value = args.pop("d_out")
    if value is not None:
        outs = {s: value for s in ("S1", "S2")}
    for key in ("streams", "exit_routes"):
        if key in args:
            args[key] = tuple(args[key].split(","))
    if args.get("dyn_source", "admitted") != "admitted":
        parser.error("corrected dynamic policy uses independent admitted signals; filled-source is retired")
    config = PolicyConfig(**{**PRESETS[options["preset"]], **args},
                          cost=CostConfig(d_in_base=ins, d_out_base=outs))
    return options, config


def main(argv=None):
    options, cfg = parse_args(argv)
    days = [d for d in grid_days() if (options["start"] is None or d >= options["start"])
            and (options["end"] is None or d <= options["end"])]
    if not days:
        raise SystemExit("no canonical grid sessions in range")
    out = DATA_ROOT / "backtest" / options["out"]
    actors = [Actor(options["preset"], cfg)]
    init_run(out, days, actors, options["resume"])
    Replay(out, actors).run(days,options["resume"],options["prefetch"])


if __name__ == "__main__":
    main()
