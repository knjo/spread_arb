"""Planned session calendar known as of a decision day.

Weekday closures observed in the 2026 canonical grid. Planned holidays are
treated as published before the study (annual TAIFEX calendar); the July 10
typhoon closure is usable only from its July 9 announcement. Realized closures
after a decision day must never be used for that day's forecast.
"""
from __future__ import annotations

from datetime import date, timedelta

# closure day -> first day it may be used in a forecast
KNOWN_CLOSURES: dict[str, str] = {
    "20260212": "20260101", "20260213": "20260101", "20260216": "20260101",
    "20260217": "20260101", "20260218": "20260101", "20260219": "20260101",
    "20260220": "20260101", "20260227": "20260101", "20260403": "20260101",
    "20260406": "20260101", "20260501": "20260101", "20260619": "20260101",
    "20260710": "20260709",
}


def parse(day: str) -> date:
    return date(int(day[:4]), int(day[4:6]), int(day[6:8]))


def fmt(d: date) -> str:
    return d.strftime("%Y%m%d")


def calendar_days(start: str, end: str) -> int:
    return (parse(end) - parse(start)).days


def is_planned_session(day: str, as_of: str) -> bool:
    if parse(day).weekday() >= 5:
        return False
    published = KNOWN_CLOSURES.get(day)
    return published is None or published > as_of


def sessions_after(day: str, through: str, as_of: str | None = None) -> list[str]:
    """Planned sessions strictly after `day` up to and including `through`."""
    as_of = as_of or day
    current, end, result = parse(day) + timedelta(days=1), parse(through), []
    while current <= end:
        text = fmt(current)
        if is_planned_session(text, as_of):
            result.append(text)
        current += timedelta(days=1)
    return result


def next_session(day: str, as_of: str | None = None) -> str:
    as_of = as_of or day
    current = parse(day) + timedelta(days=1)
    for _ in range(30):
        if is_planned_session(fmt(current), as_of):
            return fmt(current)
        current += timedelta(days=1)
    raise ValueError("no planned session within 30 days")
