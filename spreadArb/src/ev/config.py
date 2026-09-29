"""Hand-filled cost settings. These are the knobs iterated against backtests;
they are never fitted inside Stage 1.

First-version values are the maker guard-run measurements (2026-09-09) plus a
3 bp margin; negative measured exit decays are floored at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

TICK_BP_EDGES = (5.0, 15.0, 30.0)   # hedge-leg tick / price, bp
T_EDGES = (3600, 9000)


def tick_bp_bucket(tick_bp: float) -> int:
    return int(np.searchsorted(TICK_BP_EDGES, tick_bp, side="right"))


def t_bucket(t_sec: int) -> int:
    return int(np.searchsorted(T_EDGES, t_sec, side="right"))


@dataclass(frozen=True)
class CostConfig:
    fee_same_day_bp: float = 20.0
    fee_overnight_bp: float = 34.0
    ev_min_bp: float = 0.0
    hurdle_bp_per_day: float = 2000.0 / 365.0      # 20% annual, per calendar day
    margin_bp: float = 3.0
    # measured mean quote->hedge basis loss, bp (maker guard run)
    d_in_base: Mapping[str, float] = field(default_factory=lambda: {"S1": 25.7, "S2": 8.9})
    # (stream, tick_bp_bucket, t_bucket) -> measured mean; fills in as calibration arrives
    d_in_override: Mapping[tuple[str, int, int], float] = field(default_factory=dict)
    # measured exit shortfall vs target, bp; negative means better than target -> 0
    d_out_base: Mapping[str, float] = field(default_factory=lambda: {"S1": 0.0, "S2": 0.0})
    d_out_override: Mapping[tuple[str, str], float] = field(default_factory=dict)
    # closing-auction spot sale vs the settlement price when holding to expiry (unmeasured yet)
    d_settle_base: float = 0.0
    # entry gates that are not EV: dividend-driven negative anchors are a different regime
    min_anchor_bp: float = 0.0

    def d_in_bp(self, stream: str, tick_bp: float | None = None, t_sec: int | None = None,
                execution_floor_bp: float = 0.0) -> float:
        base = self.d_in_base[stream]
        if tick_bp is not None and t_sec is not None:
            base = self.d_in_override.get((stream, tick_bp_bucket(tick_bp), t_bucket(t_sec)), base)
        # empirical decay already contains hedge tick risk: take the larger, never add
        return max(max(base, 0.0) + self.margin_bp, execution_floor_bp)

    def d_out_bp(self, stream: str, route: str = "E1") -> float:
        base = self.d_out_override.get((stream, route), self.d_out_base[stream])
        return max(base, 0.0) + self.margin_bp

    def d_settle_bp(self) -> float:
        return max(self.d_settle_base, 0.0) + self.margin_bp
