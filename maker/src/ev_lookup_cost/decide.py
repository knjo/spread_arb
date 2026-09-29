"""Quote-time EV + prior-session bpday gate. Never gate an execution report.

Capacity means spot-leg entry nominal, in integer cents. Carry, intraday
positions, unhedged fills and pending orders all consume the same hard cap.
There is no prediction of which carry positions will exit later today.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .causal_lookup import CLOSE_SECOND, SECOND, DailySnapshot
from .ev_rules import cell_gate, cell_of, est_short
from .exit_model import forecast_exits, policy_ev


@dataclass(frozen=True)
class Decision:
    admit: bool
    reason: str
    est_bp: float
    lam_bp: float
    p_sd: float
    slot_free: bool
    cell: str
    p_nx: float = 0.0
    p_overnight: float = 0.0
    p_expiry: float = 0.0
    p_other: float = 0.0
    hazard_samples: int = 0
    remaining_sessions: int = 0
    execution_cost_bp: float = 0.0
    d_in: float = 0.0     # expected entry decay (quote -> hedge), bp
    d_sd: float = 0.0     # expected same-day exit shortfall vs target, bp
    d_on: float = 0.0     # expected carry exit shortfall vs target, bp


def policy_ev_xc(eff_u: float, ab: float, forecast, d_sd: float, d_on: float,
                 L: float = -5.0) -> float:
    """policy_ev with measured execution decay on every exit route.

    eff_u / ab are already net of the entry decay. Target exits pay their own
    realized shortfall table; the expiry route (C8 basis-zero) has no exit leg
    to decay, only the entry decay already removed from ab.
    """
    return (forecast.p_sd * (eff_u - L - d_sd - 20.0)
            + forecast.p_overnight * (eff_u - L - d_on - 34.0)
            + forecast.p_expiry * (ab - 34.0)
            + forecast.p_other * forecast.other_net_bp)


class EntryDecider:
    def __init__(self, cap_cents: int, snapshot: DailySnapshot, *,
                 use_ev: bool = True, use_bpday: bool = True,
                 split_ev: bool = True, calendar=None,
                 exec_cost: bool = False, decay_margin_bp: float = 3.0):
        if not isinstance(cap_cents, int) or cap_cents <= 0:
            raise ValueError("cap_cents must be a positive integer")
        if not isfinite(decay_margin_bp) or decay_margin_bp < 0:
            raise ValueError("decay margin must be finite and nonnegative")
        self.cap_cents = cap_cents
        self.snapshot = snapshot
        self.use_ev = use_ev
        self.use_bpday = use_bpday
        self.split_ev = split_ev
        self.calendar = calendar or {}
        self.exec_cost = exec_cost
        self.decay_margin_bp = decay_margin_bp
        self._forecasts = {}

    def decide(self, *, now_ns: int, stream: str, quote_second: int,
               eff_u: float, ab: float, reservation_cents: int,
               committed_cents: int, capacity_required: bool = True,
               admission_cap_cents: int | None = None, expiry: str | None = None,
               execution_cost_bp: float = 0.0, spread_bp: float = 0.0) -> Decision:
        s = self.snapshot
        if not s.cutoff_ns <= now_ns < s.cutoff_ns + CLOSE_SECOND * SECOND:
            raise ValueError("decision must use this session's frozen snapshot")
        if quote_second != (now_ns - s.cutoff_ns) // SECOND:
            raise ValueError("quote time bucket does not match the decision clock")
        if stream not in {"S1", "S2"}:
            raise ValueError("only the two short-basis maker routes are supported")
        if not all(isfinite(v) for v in (eff_u, ab)):
            raise ValueError("quote features must be finite")
        if not isfinite(execution_cost_bp) or execution_cost_bp < 0:
            raise ValueError("execution cost must be finite and nonnegative")
        if reservation_cents <= 0 or committed_cents < 0:
            raise ValueError("invalid capacity accounting")
        if self.split_ev and (expiry is None or expiry < s.day):
            raise ValueError("split EV requires the current contract's exact expiry")
        p = s.p_sd(stream, quote_second)
        limit = min(self.cap_cents, admission_cap_cents) if admission_cap_cents is not None else self.cap_cents
        free = committed_cents + reservation_cents <= limit
        # An admitted order has its entire overnight nominal reserved. P_sd
        # changes EV, never capacity. Estimate rejected opportunities on the
        # same feasible-slot basis, for the next day's lambda calculation.
        # Prior completed daily risk sets estimate normal/other exit hazards.
        # Only survival through the exact remaining contract life receives C8.
        key = (expiry, stream, p)
        forecast = self._forecasts.get(key)
        if forecast is None:
            forecast = forecast_exits(s.day, expiry or s.day, stream, p, s.exit_hazards, self.calendar)
            self._forecasts[key] = forecast
        p_nx = forecast.p_overnight / (1.0 - p) if p < 1.0 else 0.0
        # Measured execution decay (walk-forward tables + margin) is removed
        # from every branch before the EV comparison. Realized cash is untouched.
        d_in = d_sd = d_on = 0.0
        if self.exec_cost:
            d_in = s.entry_decay(stream, quote_second, spread_bp, margin_bp=self.decay_margin_bp)
            d_sd = s.exit_decay(stream, False, margin_bp=self.decay_margin_bp)
            d_on = s.exit_decay(stream, True, margin_bp=self.decay_margin_bp)
        if self.split_ev:
            est = policy_ev_xc(eff_u - d_in, ab - d_in, forecast, d_sd, d_on)
        else:
            est = est_short(eff_u - d_in, ab - d_in, p, L=-5.0, slot_free=True, p_nx=0.0) - p * d_sd
        # Empirical quote-to-hedge decay includes the hedge tick risk already.
        # Use the larger of the empirical cost and the fixed depth/tick floor,
        # never add the same entry risk twice. Other-event net is already net.
        if self.exec_cost:
            extra_entry = max(0.0, execution_cost_bp-d_in)
            mass = forecast.p_sd+forecast.p_overnight+forecast.p_expiry
            est -= mass*extra_entry
            d_in += extra_entry
        else:
            est -= execution_cost_bp
        cell = cell_of(stream, ab)
        stats = s.cells.get(cell, (0.0, 0.0, 0))
        reason = "ok"
        if ab <= 0:
            reason = "basis"
        elif self.use_ev and est < s.lam_bp * 0.6:
            reason = "ev"
        elif self.use_bpday and not cell_gate(*stats):
            reason = "cell"
        elif capacity_required and not free:
            reason = "cap"
        return Decision(reason == "ok", reason, est, s.lam_bp, p, free, cell, p_nx,
                        forecast.p_overnight, forecast.p_expiry, forecast.p_other,
                        forecast.hazard_samples, forecast.remaining_sessions, execution_cost_bp,
                        d_in, d_sd, d_on)
