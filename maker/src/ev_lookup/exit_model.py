"""Prior-session competing exit hazards; open carry remains in the risk set.

The terminal expiry mass is survival through the contract's remaining sessions,
not the complement of same-day exits or of a completed-trades-only sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping


def expiry_bucket(days: int) -> int:
    return 0 if days <= 0 else 1 if days <= 3 else 2 if days <= 10 else 3


def calendar_days(day: str, expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y%m%d") - datetime.strptime(day, "%Y%m%d")).days


def remaining_sessions(day: str, expiry: str, calendar: Mapping[str, bool]) -> list[str]:
    """Use an as-of forecast calendar, never the realized replay-day calendar.

    Production supplies every date through expiry. Empty/sparse maps are useful
    for isolated synthetic tests and use weekdays for unspecified dates.
    """
    if calendar and expiry not in calendar:
        raise ValueError("expiry outside verified forecast-calendar coverage")
    current = datetime.strptime(day, "%Y%m%d") + timedelta(days=1)
    end = datetime.strptime(expiry, "%Y%m%d")
    result = []
    while current <= end:
        key = current.strftime("%Y%m%d")
        if calendar.get(key, current.weekday() < 5):
            result.append(key)
        current += timedelta(days=1)
    return result


@dataclass(frozen=True)
class ExitForecast:
    p_sd: float
    p_overnight: float
    p_expiry: float
    p_other: float
    other_net_bp: float
    remaining_sessions: int
    hazard_samples: int


def fit_hazards(observations) -> dict[str, tuple[int, int, int, float]]:
    """n / normal exits / other exits / sum(other net bp), including non-exits."""
    rows: dict[str, list] = {}
    for r in observations:
        if not r.carry or r.start_second > 0 or not r.expiry or r.day > r.expiry:
            continue
        # C8 closes at the open after expiry, never a pre-expiry exit hazard.
        if r.close_kind == "expiry_basis_zero_accounting":
            continue
        normal = r.closed and r.close_kind == "maker_exit"
        other = r.closed and not normal
        for key in ("all", r.stream, f"{r.stream}:{expiry_bucket(calendar_days(r.day, r.expiry))}"):
            x = rows.setdefault(key, [0, 0, 0, 0.0])
            x[0] += 1
            x[1] += int(normal)
            x[2] += int(other)
            x[3] += r.net_bp if other else 0.0
    return {k: tuple(v) for k, v in rows.items()}


def forecast_exits(day: str, expiry: str, stream: str, p_sd: float,
                   hazards: Mapping[str, tuple[int, int, int, float]],
                   calendar: Mapping[str, bool]) -> ExitForecast:
    survival, normal, other, other_value = 1.0 - p_sd, 0.0, 0.0, 0.0
    sessions = remaining_sessions(day, expiry, calendar)
    samples = 0
    for future_day in sessions:
        stats = next((hazards[k] for k in
                      (f"{stream}:{expiry_bucket(calendar_days(future_day, expiry))}", stream, "all")
                      if k in hazards and hazards[k][0] >= 30), None)
        if stats is None:
            # Fixed, disclosed cold-start prior: half of surviving pairs reach
            # their target each session. No fitted claim before 30 exposures.
            q_normal, q_other, mean_other = 0.5, 0.0, 0.0
        else:
            n, n_normal, n_other, sum_other = stats
            samples = max(samples, n)
            q_normal, q_other = n_normal / n, n_other / n
            mean_other = sum_other / n_other if n_other else 0.0
        normal += survival * q_normal
        mass = survival * q_other
        other += mass
        other_value += mass * mean_other
        survival *= 1.0 - q_normal - q_other
    return ExitForecast(p_sd, normal, max(0.0, survival), other,
                        other_value / other if other > 0 else 0.0, len(sessions), samples)


def policy_ev(eff_u: float, ab: float, forecast: ExitForecast, L: float = -5.0) -> float:
    return (forecast.p_sd * (eff_u - L - 20.0)
            + forecast.p_overnight * (eff_u - L - 34.0)
            + forecast.p_expiry * (ab - 34.0)
            + forecast.p_other * forecast.other_net_bp)
