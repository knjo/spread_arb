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
    # Q-table features (2026-09-11): quote-time depth / futures spread / time, and the
    # capital-days the pair actually occupied (same-day = fraction of the session).
    eff_u: float = 0.0
    spread_bp: float = 0.0
    quote_second: int = 0
    cap_days: float = 0.0


@dataclass(frozen=True)
class ExitRiskObservation:
    """One paired position's same-session exit exposure, known only after close."""
    position_id: str
    stream: str
    day: str
    hedged_second: float
    exit_second: float | None   # same-day maker exit fill time; None if not maker-exited
    available_ns: int


SPREAD_BUCKETS = (30.0, 80.0)


def spread_bucket(spread_bp: float) -> int:
    """Futures top-of-book spread at quote time: <30 / 30-80 / >=80 bp."""
    return sum(spread_bp >= s for s in SPREAD_BUCKETS)


U_BUCKETS = (40.0, 60.0, 100.0)


def u_bucket(eff_u: float) -> int:
    """Quote-time depth above anchor: <40 / 40-60 / 60-100 / >=100 bp."""
    return sum(eff_u >= u for u in U_BUCKETS)


def q_key(stream: str, eff_u: float, spread_bp: float, second: int) -> tuple:
    return (stream, u_bucket(eff_u), spread_bucket(spread_bp), time_bucket(second))


@dataclass(frozen=True)
class DecayObservation:
    """Execution decay of one shadow position, known only after the session.

    kind="entry": bp = quote_ab - actual_ab (basis lost between quote and hedge;
                  pre-fill drift + post-fill hedge execution). bucket = time
                  bucket of the quote second, spread = futures spread bucket.
    kind="exit":  bp = realized exit basis - (frozen anchor - 5); bucket = 0 for
                  same-day exits, 1 for carry exits. Positive = worse than target.
    """
    position_id: str
    stream: str
    kind: str
    bucket: int
    spread: int
    bp: float
    day: str
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
    # (kind, stream, bucket, spread) -> (n, sum_bp, sumsq_bp); -1 = pooled, "*" = all streams
    decay: Mapping[tuple, tuple[int, float, float]] = MappingProxyType({})
    # Q-table: (stream, u, spread, time) -> (sum_bp, sum_cap_days, n); -1 = pooled over time / spread
    qcells: Mapping[tuple, tuple[float, float, int]] = MappingProxyType({})

    def q_bpday(self, stream: str, eff_u: float, spread_bp: float, second: int,
                min_n: int = 30) -> float | None:
        """Realized bp per capital-day of this quote's cell (prior 20 sessions),
        falling back to time-pooled then spread-pooled cells; None while unknown."""
        s, u, sp, t = q_key(stream, eff_u, spread_bp, second)
        for key in ((s, u, sp, t), (s, u, sp, -1), (s, u, -1, -1)):
            bp, days, n = self.qcells.get(key, (0.0, 0.0, 0))
            if n >= min_n:
                return (bp / n) / max(days / n, 0.05)
        return None

    def _decay(self, keys, min_n: int) -> tuple[float, int] | None:
        for key in keys:
            n, s, _ = self.decay.get(key, (0, 0.0, 0.0))
            if n >= min_n:
                return s / n, n
        return None

    def entry_decay(self, stream: str, second: int, spread_bp: float, *,
                    margin_bp: float = 3.0, min_n: int = 30, prior_bp: float = 15.0) -> float:
        """Expected basis lost between quote and hedge (bp, >= 0), plus a margin.

        Walk-forward: shadow positions hedged in the prior 20 completed
        sessions. Falls back from (stream, time, spread) to pooled cells, then
        to a disclosed prior (the 2026-09 measured mean of ~15 bp)."""
        sb = spread_bucket(spread_bp)
        found = self._decay((("entry", stream, time_bucket(second), sb),
                             ("entry", stream, -1, sb),
                             ("entry", stream, -1, -1),
                             ("entry", "*", -1, -1)), min_n)
        mean = prior_bp if found is None else found[0]
        return max(0.0, mean) + margin_bp

    def exit_decay(self, stream: str, carry: bool, *, margin_bp: float = 3.0,
                   min_n: int = 30, prior_bp: float = 5.0) -> float:
        """Expected shortfall of the realized exit basis vs the frozen target (bp, >= 0)."""
        found = self._decay((("exit", stream, int(carry), -1),
                             ("exit", "*", int(carry), -1)), min_n)
        mean = prior_bp if found is None else found[0]
        return max(0.0, mean) + margin_bp

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
        self.decays: list[DecayObservation] = []
        self._decay_ids: set[str] = set()

    def observe_decay(self, outcome: DecayObservation, now_ns: int) -> None:
        if outcome.available_ns > now_ns:
            raise ValueError("decay is not observable yet")
        if outcome.available_ns < open_ns(outcome.day) + CLOSE_SECOND * SECOND:
            raise ValueError("decay is finalized only after the session")
        if outcome.kind not in {"entry", "exit"}:
            raise ValueError("unknown decay kind")
        key = f"{outcome.kind}:{outcome.position_id}"
        if key in self._decay_ids:
            raise ValueError("duplicate decay observation")
        self._decay_ids.add(key)
        self.decays.append(outcome)

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
        qcells: dict[tuple, list] = {}
        for r in self.resolutions:
            if r.resolved_day not in prior or r.available_ns >= cutoff or r.cap_days <= 0:
                continue
            s, u, sp, t = q_key(r.stream, r.eff_u, r.spread_bp, r.quote_second)
            for key in ((s, u, sp, t), (s, u, sp, -1), (s, u, -1, -1)):
                q = qcells.setdefault(key, [0.0, 0.0, 0])
                q[0] += r.pnl_bp
                q[1] += r.cap_days
                q[2] += 1
        decay: dict[tuple, list] = {}
        for r in self.decays:
            if r.day not in prior or r.available_ns >= cutoff:
                continue
            if r.kind == "entry":
                keys = ((r.kind, r.stream, r.bucket, r.spread), (r.kind, r.stream, -1, r.spread),
                        (r.kind, r.stream, -1, -1), (r.kind, "*", -1, -1))
            else:
                keys = ((r.kind, r.stream, r.bucket, -1), (r.kind, "*", r.bucket, -1))
            for key in keys:
                d = decay.setdefault(key, [0, 0.0, 0.0])
                d[0] += 1
                d[1] += r.bp
                d[2] += r.bp * r.bp
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
                                 if r.day in prior and r.available_ns < cutoff)),
                             MappingProxyType({k: tuple(v) for k, v in decay.items()}),
                             MappingProxyType({k: tuple(v) for k, v in qcells.items()}))

    def finish_session(self, day: str) -> None:
        if self.sessions and day <= self.sessions[-1]:
            raise ValueError("sessions must be completed in increasing order")
        self.sessions.append(day)
