"""Cross-check S1 accounting terminals against capacity lifecycle facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .capacity_ledger import CapacityTransition
from .layered import EventCursor
from .s1_accounting import (
    AccountingFact,
    ExecutedLeg,
    ExpiryAccountingMark,
    PositionEstablishedFact,
    TerminalRealizedAccounting,
)


class S1CrossLedgerError(ValueError):
    """Accounting and capacity facts do not describe one lifecycle."""


@dataclass(frozen=True, slots=True)
class S1CrossLedgerReport:
    establishments: int
    executable_terminals: int
    expiry_marks: int
    linked_capacity_transitions: int
    linked_event_counts: tuple[tuple[str, int], ...]


_TERMINAL_CAPACITY_EVENT = {
    "exit_maker_flat": "exit_hedge_complete",
    "entry_emergency_rollback_flat": "entry_rollback_complete",
    "entry_partial_rollback_flat": "entry_rollback_complete",
}
_LINKABLE_CAPACITY_EVENTS = frozenset(
    {
        "entry_hedge_complete",
        "exit_hedge_complete",
        "entry_rollback_complete",
        "expiry_basis_zero_release",
    }
)


def verify_s1_accounting_capacity_links(
    accounting_facts: Sequence[AccountingFact],
    capacity_transitions: Sequence[CapacityTransition],
) -> S1CrossLedgerReport:
    """Require every S1 establishment/release to join exactly once.

    Each link is checked on transition identity, event type, capacity account,
    product, and causal cursor.  Linkable capacity mutations must not remain
    orphaned, so two individually replayable ledgers cannot silently describe
    different portfolios.
    """

    facts = _typed_sequence(
        accounting_facts,
        "accounting_facts",
        expected_type=(
            ExecutedLeg,
            PositionEstablishedFact,
            TerminalRealizedAccounting,
            ExpiryAccountingMark,
        ),
    )
    transitions = _typed_sequence(
        capacity_transitions,
        "capacity_transitions",
        expected_type=CapacityTransition,
    )
    by_id: dict[str, CapacityTransition] = {}
    for transition in transitions:
        if transition.transition_id in by_id:
            raise S1CrossLedgerError("capacity transition_id is duplicated")
        by_id[transition.transition_id] = transition

    used: set[str] = set()
    counts: Counter[str] = Counter()
    establishment_count = 0
    terminal_count = 0
    expiry_count = 0
    for fact in facts:
        if isinstance(fact, PositionEstablishedFact):
            establishment_count += 1
            transition_id = fact.capacity_transition_id
            expected_event = "entry_hedge_complete"
        elif isinstance(fact, TerminalRealizedAccounting):
            terminal_count += 1
            transition_id = fact.capacity_release_transition_id
            try:
                expected_event = _TERMINAL_CAPACITY_EVENT[fact.terminal_outcome]
            except KeyError as error:
                raise S1CrossLedgerError(
                    "terminal outcome lacks a registered capacity release contract"
                ) from error
        elif isinstance(fact, ExpiryAccountingMark):
            expiry_count += 1
            transition_id = fact.capacity_release_transition_id
            expected_event = "expiry_basis_zero_release"
        else:
            continue
        if transition_id in used:
            raise S1CrossLedgerError(
                "one capacity transition is linked by multiple accounting facts"
            )
        try:
            transition = by_id[transition_id]
        except KeyError as error:
            raise S1CrossLedgerError(
                "accounting fact links an unknown capacity transition"
            ) from error
        if transition.event_type != expected_event:
            raise S1CrossLedgerError(
                "accounting fact links the wrong capacity event type"
            )
        if transition.capacity_id != fact.capacity_id:
            raise S1CrossLedgerError(
                "accounting/capacity capacity_id values disagree"
            )
        if transition.product_id != fact.value_code:
            raise S1CrossLedgerError(
                "accounting value_code differs from capacity product_id"
            )
        if _transition_cursor(transition) >= fact.cursor:
            raise S1CrossLedgerError(
                "capacity transition must causally precede accounting settlement"
            )
        used.add(transition_id)
        counts[expected_event] += 1

    expected_used = {
        transition.transition_id
        for transition in transitions
        if transition.event_type in _LINKABLE_CAPACITY_EVENTS
    }
    if used != expected_used:
        missing = sorted(expected_used - used)
        extra = sorted(used - expected_used)
        raise S1CrossLedgerError(
            f"linkable capacity transitions are not one-to-one; "
            f"orphaned={missing}, unexpected={extra}"
        )
    return S1CrossLedgerReport(
        establishments=establishment_count,
        executable_terminals=terminal_count,
        expiry_marks=expiry_count,
        linked_capacity_transitions=len(used),
        linked_event_counts=tuple(sorted(counts.items())),
    )


def _transition_cursor(transition: CapacityTransition) -> EventCursor:
    return EventCursor(
        transition.timestamp_ns,
        transition.event_sequence,
        transition.row_index,
    )


def _typed_sequence[Row](
    values: object,
    name: str,
    *,
    expected_type: type[Row] | tuple[type[object], ...] | None = None,
) -> tuple[Row, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    rows = tuple(values)
    if expected_type is not None and any(
        not isinstance(value, expected_type) for value in rows
    ):
        raise TypeError(f"{name} contains an invalid value")
    return rows  # type: ignore[return-value]


__all__ = [
    "S1CrossLedgerError",
    "S1CrossLedgerReport",
    "verify_s1_accounting_capacity_links",
]
