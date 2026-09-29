"""Actual exposure ledger for quotes that do not reserve capital.

Admission is checked by the actor. Exchange fills cannot be rejected after the
fact: cancellation races and hedge repricing may temporarily exceed the limit.
"""
from __future__ import annotations


class PositionCapital:
    def __init__(self, cap_cents: int):
        self.cap_cents = cap_cents
        self.amounts: dict[str, int] = {}
        self.committed_cents = self.peak_cents = self.last_ns = 0
        self.events: list[dict] = []

    def exposure(self, key: str, cents: int, now_ns: int, kind: str, **evidence):
        if cents <= 0 or now_ns < self.last_ns:
            raise ValueError("invalid exposure or nonchronological capital event")
        old = self.amounts.get(key, 0)
        self.amounts[key] = cents
        self._record(key, now_ns, kind, cents-old, **evidence)

    def _record(self, key, now_ns, kind, delta, **evidence):
        if now_ns < self.last_ns:
            raise ValueError("nonchronological capital event")
        self.last_ns = now_ns
        self.committed_cents += delta
        if self.committed_cents < 0:
            raise AssertionError("negative actual exposure")
        self.peak_cents = max(self.peak_cents, self.committed_cents)
        self.events.append(dict(id=key, ns=now_ns, kind=kind, delta_cents=delta,
                                committed_cents=self.committed_cents,
                                excess_cents=max(0, self.committed_cents-self.cap_cents), **evidence))

    def release(self, key, now_ns, *, terminal):
        if not terminal:
            raise ValueError("cannot release unresolved exposure")
        old = self.amounts.pop(key)
        self._record(key, now_ns, "release", -old)
