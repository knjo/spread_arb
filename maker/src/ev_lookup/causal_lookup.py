"""Only completed, observable outcomes can enter the next session's lookup."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from .ev_rules import cell_of, shadow_price
from .exit_model import fit_hazards

SECOND = 1_000_000_000
CLOSE_SECOND = 15_600


def open_ns(day: str) -> int:
    return int(datetime.strptime(day + " 01:00:00", "%Y%m%d %H:%M:%S")
               .replace(tzinfo=timezone.utc).timestamp()) * SECOND


def time_bucket(second: int) -> int:
    return 0 if second < 3600 else 1 if second < 9000 else 2


# Remaining-time buckets for the exit P_fill table: before 11:30, 11:30-12:30,
# 12:30-13:00, 13:00-13:18 (maker orders are withdrawn at 13:18 = 15,480 s).
EXIT_BUCKET_STARTS = (0, 9000, 12600, 14400)


def exit_bucket(second: int) -> int:
    return sum(second >= s for s in EXIT_BUCKET_STARTS) - 1


@dataclass(frozen=True)
class ResolvedOutcome:
    position_id: str
    stream: str
    quote_ab: float
    entry_day: str
    resolved_day: str
    available_ns: int
    pnl_bp: float
    held_sessions: float
    # How the pair ended: maker_exit / taker_cross / expiry_basis_zero_accounting /
    # partial_entry_rollback / corporate. The v21 EV uses daily carry risk sets.
    kind: str = "maker_exit"


@dataclass(frozen=True)
class ExitRiskObservation:
    """One paired position's same-session exit exposure, known only after close."""
    position_id: str
    stream: str
    day: str
    hedged_second: float
    exit_second: float | None   # same-day maker exit fill time; None if not maker-exited
    available_ns: int


@dataclass(frozen=True)
class DayEntryOutcome:
    position_id: str
    stream: str
    quote_second: int
    entry_day: str
    available_ns: int
    same_day: bool


@dataclass(frozen=True)
class CapacityObservation:
    day: str
    stream: str
    carry: bool
    start_second: float
    end_second: float
    closed: bool
    available_ns: int
    expiry: str | None = None
    close_kind: str | None = None
    net_bp: float = 0.0


@dataclass(frozen=True)
class DailySnapshot:
    day: str
    cutoff_ns: int
    cells: Mapping[str, tuple[float, float, int]]
    psd: Mapping[tuple[str, int], tuple[int, int]]
    lam_bp: float
    train_days: tuple[str, ...]
    capacity_observations: tuple[CapacityObservation, ...] = ()
    pnx: Mapping[str, tuple[int, int]] = MappingProxyType({})
    pfill: Mapping[int, tuple[int, int]] = MappingProxyType({})
    exit_hazards: Mapping[str, tuple[int, int, int, float]] = MappingProxyType({})

    def p_sd(self, stream: str, second: int) -> float:
        n, n_sd = self.psd.get((stream, time_bucket(second)), (0, 0))
        return n_sd / n if n >= 200 else 0.68

    def p_nx(self, stream: str) -> float:
        """Legacy completed-only diagnostic; never used by the v21 EV decider."""
        n, n_mx = self.pnx.get(stream, (0, 0))
        return n_mx / n if n >= 30 else 0.8

    def p_fill(self, second: int) -> float:
        """P(maker exit fills later today | still paired and open at this bucket start)."""
        n, n_f = self.pfill.get(exit_bucket(second), (0, 0))
        return n_f / n if n >= 30 else 0.5


class LookupHistory:
    """Shadow outcomes are recorded at execution time; snapshots never mutate."""

    def __init__(self) -> None:
        self.resolutions: list[ResolvedOutcome] = []
        self.entry_days: list[DayEntryOutcome] = []
        self.sessions: list[str] = []
        self._resolved_ids: set[str] = set()
        self._entry_ids: set[str] = set()
        self._exit_ids: set[str] = set()
        self.capacity_observations: list[CapacityObservation] = []
        self.exit_risks: list[ExitRiskObservation] = []

    def observe_exit_risk(self, outcome: ExitRiskObservation, now_ns: int) -> None:
        if outcome.available_ns > now_ns:
            raise ValueError("exit exposure is not observable yet")
        if outcome.available_ns < open_ns(outcome.day) + CLOSE_SECOND * SECOND:
            raise ValueError("exit exposure is finalized only after the session")
        if outcome.exit_second is not None and outcome.exit_second < outcome.hedged_second:
            raise ValueError("exit precedes hedge")
        if outcome.position_id in self._exit_ids:
            raise ValueError("duplicate exit exposure")
        self._exit_ids.add(outcome.position_id)
        self.exit_risks.append(outcome)

    def observe_resolution(self, outcome: ResolvedOutcome, now_ns: int) -> None:
        if outcome.available_ns > now_ns:
            raise ValueError("resolution is not observable yet")
        if outcome.position_id in self._resolved_ids:
            raise ValueError("duplicate resolution")
        if outcome.resolved_day < outcome.entry_day or outcome.held_sessions < 0:
            raise ValueError("resolution precedes entry")
        if outcome.available_ns < open_ns(outcome.resolved_day):
            raise ValueError("resolution timestamp precedes its session")
        self._resolved_ids.add(outcome.position_id)
        self.resolutions.append(outcome)

    def observe_entry_day(self, outcome: DayEntryOutcome, now_ns: int) -> None:
        if outcome.available_ns > now_ns:
            raise ValueError("same-day result is not observable yet")
        if outcome.available_ns < open_ns(outcome.entry_day) + CLOSE_SECOND * SECOND:
            raise ValueError("same-day status is finalized only after the session")
        if outcome.position_id in self._entry_ids:
            raise ValueError("duplicate entry-day outcome")
        self._entry_ids.add(outcome.position_id)
        self.entry_days.append(outcome)

    def freeze(self, day: str, cap_twd: float,
               rejected_by_day: Mapping[str, float]) -> DailySnapshot:
        if self.sessions and day <= self.sessions[-1]:
            raise ValueError("snapshot must follow completed sessions")
        cutoff = open_ns(day)
        prior = tuple(self.sessions[-20:])
        cells: dict[str, list] = {}
        psd: dict[tuple[str, int], list[int]] = {}
        pnx: dict[str, list[int]] = {}
        pfill: dict[int, list[int]] = {}
        for r in self.resolutions:
            if r.resolved_day not in prior or r.available_ns >= cutoff:
                continue
            c = cells.setdefault(cell_of(r.stream, r.quote_ab), [0.0, 0.0, 0])
            c[0] += r.pnl_bp
            c[1] += max(r.held_sessions, 0.15)
            c[2] += 1
            if r.resolved_day > r.entry_day:
                x = pnx.setdefault(r.stream, [0, 0])
                x[0] += 1
                x[1] += int(r.kind == "maker_exit")
        for r in self.exit_risks:
            if r.day not in prior or r.available_ns >= cutoff:
                continue
            for b, start in enumerate(EXIT_BUCKET_STARTS):
                if r.hedged_second <= start and (r.exit_second is None or r.exit_second > start):
                    f = pfill.setdefault(b, [0, 0])
                    f[0] += 1
                    f[1] += int(r.exit_second is not None)
        for r in self.entry_days:
            if r.entry_day >= day or r.available_ns >= cutoff:
                continue
            c = psd.setdefault((r.stream, time_bucket(r.quote_second)), [0, 0])
            c[0] += 1
            c[1] += int(r.same_day)
        lam = shadow_price([rejected_by_day.get(d, 0.0) for d in self.sessions[-5:]], cap_twd)
        return DailySnapshot(day, cutoff,
                             MappingProxyType({k: tuple(v) for k, v in cells.items()}),
                             MappingProxyType({k: tuple(v) for k, v in psd.items()}),
                             lam, prior,
                             tuple(r for r in self.capacity_observations
                                   if r.day in prior and r.available_ns < cutoff),
                             MappingProxyType({k: tuple(v) for k, v in pnx.items()}),
                             MappingProxyType({k: tuple(v) for k, v in pfill.items()}),
                             MappingProxyType(fit_hazards(r for r in self.capacity_observations
                                 if r.day in prior and r.available_ns < cutoff)))

    def finish_session(self, day: str) -> None:
        if self.sessions and day <= self.sessions[-1]:
            raise ValueError("sessions must be completed in increasing order")
        self.sessions.append(day)
