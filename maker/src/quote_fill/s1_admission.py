"""Shadow C9 admission planning for one frozen request-assignment phase.

The venue scheduler must choose among currently send-eligible ``new``
requests without mutating the append-only capacity ledger inside its callback.
This planner snapshots committed totals at the beginning of one timestamp,
adds successful reservations to shadow balances in deterministic evaluation
order, and later commits every attempt to :class:`CapacityLedger` before any
same-timestamp cancel or exit release is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .capacity_ledger import CapacityLedger, CapacityTransition


@dataclass(frozen=True, slots=True)
class PlannedAdmission:
    evaluation_sequence: int
    request_id: str
    capacity_id: str
    product_id: str
    requested_notional_twd: int
    admitted: bool
    status: str
    assignment_send_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class CommittedAdmission:
    plan: PlannedAdmission
    transition: CapacityTransition


class CapacityAdmissionPlanner:
    """One-shot shadow planner bound to a ledger's cursor-start balances."""

    def __init__(self, ledger: CapacityLedger) -> None:
        if not isinstance(ledger, CapacityLedger):
            raise TypeError("ledger must be a CapacityLedger")
        self._ledger = ledger
        self._base_global_twd = ledger.global_balances.total_committed_notional_twd
        self._base_product_twd: dict[str, int] = {}
        self._shadow_global_twd = 0
        self._shadow_product_twd: dict[str, int] = {}
        self._plans: list[PlannedAdmission] = []
        self._by_request: dict[str, PlannedAdmission] = {}
        self._committed = False

    @property
    def plans(self) -> tuple[PlannedAdmission, ...]:
        return tuple(self._plans)

    @property
    def cursor_start_global_twd(self) -> int:
        return self._base_global_twd

    def evaluate(
        self,
        *,
        request_id: str,
        capacity_id: str,
        product_id: str,
        requested_notional_twd: int,
    ) -> PlannedAdmission:
        """Evaluate once and reserve admitted notional only in shadow state."""

        if self._committed:
            raise RuntimeError("planner was already committed")
        request_id = _identifier(request_id, "request_id")
        capacity_id = _identifier(capacity_id, "capacity_id")
        product_id = _identifier(product_id, "product_id")
        requested = _positive_integer(requested_notional_twd)
        existing = self._by_request.get(request_id)
        if existing is not None:
            if (
                existing.capacity_id != capacity_id
                or existing.product_id != product_id
                or existing.requested_notional_twd != requested
            ):
                raise ValueError("repeated request_id changed admission inputs")
            return existing
        if any(plan.capacity_id == capacity_id for plan in self._plans):
            raise ValueError("capacity_id cannot be shared by two requests")

        base_product = self._base_product_twd.setdefault(
            product_id,
            self._ledger.product_balances(product_id).total_committed_notional_twd,
        )
        projected_global = self._base_global_twd + self._shadow_global_twd + requested
        projected_product = (
            base_product + self._shadow_product_twd.get(product_id, 0) + requested
        )
        global_blocked = projected_global > self._ledger.global_cap_twd
        product_blocked = projected_product > self._ledger.product_cap_twd
        admitted = not global_blocked and not product_blocked
        if admitted:
            status = "shadow_admitted"
            self._shadow_global_twd += requested
            self._shadow_product_twd[product_id] = (
                self._shadow_product_twd.get(product_id, 0) + requested
            )
        elif global_blocked and product_blocked:
            status = "shadow_blocked_both_caps"
        elif global_blocked:
            status = "shadow_blocked_global_cap"
        else:
            status = "shadow_blocked_product_cap"
        plan = PlannedAdmission(
            evaluation_sequence=len(self._plans) + 1,
            request_id=request_id,
            capacity_id=capacity_id,
            product_id=product_id,
            requested_notional_twd=requested,
            admitted=admitted,
            status=status,
        )
        self._plans.append(plan)
        self._by_request[request_id] = plan
        return plan

    def bind_assignment(
        self,
        request_id: str,
        *,
        send_sequence: int,
    ) -> PlannedAdmission:
        """Bind a venue assignment to an admitted shadow reservation."""

        if self._committed:
            raise RuntimeError("planner was already committed")
        request_id = _identifier(request_id, "request_id")
        send_sequence = _positive_integer(send_sequence, "send_sequence")
        try:
            plan = self._by_request[request_id]
        except KeyError as error:
            raise ValueError("assignment has no admission evaluation") from error
        if not plan.admitted:
            raise ValueError("blocked admission cannot receive an assignment")
        if plan.assignment_send_sequence is not None:
            raise ValueError("admission assignment is already bound")
        if any(
            value.assignment_send_sequence == send_sequence for value in self._plans
        ):
            raise ValueError("send_sequence is already bound")
        bound = replace(plan, assignment_send_sequence=send_sequence)
        index = plan.evaluation_sequence - 1
        self._plans[index] = bound
        self._by_request[request_id] = bound
        return bound

    def commit(
        self,
        *,
        timestamp_ns: int,
        event_sequence: int,
        first_row_index: int = 1,
        transition_prefix: str,
    ) -> tuple[CommittedAdmission, ...]:
        """Append all evaluations before any same-timestamp release effect."""

        if self._committed:
            raise RuntimeError("planner was already committed")
        if self._ledger.global_balances.total_committed_notional_twd != (
            self._base_global_twd
        ):
            raise RuntimeError("ledger committed total changed before admission commit")
        drifted_products = [
            product_id
            for product_id, base_total in self._base_product_twd.items()
            if self._ledger.product_balances(product_id).total_committed_notional_twd
            != base_total
        ]
        if drifted_products:
            raise RuntimeError(
                "ledger product committed total changed before admission commit: "
                f"{sorted(drifted_products)}"
            )
        first_row_index = _positive_integer(first_row_index, "first_row_index")
        transition_prefix = _identifier(transition_prefix, "transition_prefix")
        missing_assignment = [
            plan.request_id
            for plan in self._plans
            if plan.admitted and plan.assignment_send_sequence is None
        ]
        if missing_assignment:
            raise RuntimeError(
                f"admitted plans lack venue assignments: {missing_assignment}"
            )
        committed: list[CommittedAdmission] = []
        for offset, plan in enumerate(self._plans):
            decision = self._ledger.attempt_new_reservation(
                transition_id=(
                    f"{transition_prefix}/{plan.evaluation_sequence:06d}/"
                    f"{plan.request_id}"
                ),
                timestamp_ns=timestamp_ns,
                event_sequence=event_sequence,
                row_index=first_row_index + offset,
                capacity_id=plan.capacity_id,
                product_id=plan.product_id,
                requested_notional_twd=plan.requested_notional_twd,
            )
            if decision.admitted != plan.admitted:
                raise RuntimeError("shadow and append-only admission decisions differ")
            committed.append(CommittedAdmission(plan, decision.transition))
        self._committed = True
        return tuple(committed)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: int, name: str = "requested_notional_twd") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "CapacityAdmissionPlanner",
    "CommittedAdmission",
    "PlannedAdmission",
]
