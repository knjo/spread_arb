"""Production bridge from S1 actual-new facts to legacy makerFill labels.

The chronological loop needs only a potential event to schedule, but research
denominators also need every supported, no-fill, and unsupported assessment.
This adapter therefore records exactly one immutable makerFill event per sent
raw order while returning a scheduled fill only for a supported positive
legacy label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType

from .makerfill_adapter import (
    ADAPTER_VERSION,
    MakerFillLabelIndex,
    MakerFillObservedSnapshot,
    MakerFillScalarEvent,
    MakerFillScalarOrder,
)
from .s1_event_loop import (
    ActualSendMakerSnapshot,
    PotentialEntryFill,
    SentEntryOrder,
)


class S1MakerFillBridge:
    """Stateful one-call-per-order makerFill adapter with complete audit."""

    def __init__(self, index: MakerFillLabelIndex) -> None:
        if not isinstance(index, MakerFillLabelIndex):
            raise TypeError("index must be a MakerFillLabelIndex")
        self._index = index
        self._events_by_raw_id: dict[str, MakerFillScalarEvent] = {}

    @property
    def assessments(self) -> tuple[MakerFillScalarEvent, ...]:
        """Return all assessments in deterministic raw-order identity order."""

        return tuple(
            self._events_by_raw_id[raw_id]
            for raw_id in sorted(self._events_by_raw_id)
        )

    @property
    def assessments_by_raw_id(self) -> Mapping[str, MakerFillScalarEvent]:
        return MappingProxyType(dict(self._events_by_raw_id))

    def assessment_rows(self) -> tuple[dict[str, object], ...]:
        """Return serialization-ready rows without changing scalar semantics."""

        return tuple(asdict(event) for event in self.assessments)

    def potential_fill(
        self,
        order: SentEntryOrder,
        snapshot: ActualSendMakerSnapshot,
    ) -> PotentialEntryFill | None:
        """Assess one actual-new order and optionally schedule its legacy fill."""

        if not isinstance(order, SentEntryOrder):
            raise TypeError("order must be a SentEntryOrder")
        if not isinstance(snapshot, ActualSendMakerSnapshot):
            raise TypeError("snapshot must be an ActualSendMakerSnapshot")
        if order.raw_order_fact_id in self._events_by_raw_id:
            raise ValueError("raw order already has a makerFill assessment")
        if snapshot != order.maker_snapshot:
            raise ValueError("fill snapshot differs from the frozen sent-order snapshot")

        scalar_order = MakerFillScalarOrder(
            raw_order_fact_id=order.raw_order_fact_id,
            date=order.date,
            value_code=order.value_code,
            quote_code=order.quote_code,
            absolute_price_tick=order.absolute_price_tick,
            maker_snapshot_channel_seq=snapshot.channel_seq,
            maker_snapshot_recv_time_ns=snapshot.recv_time_ns,
            actual_new_send_time_ns=order.actual_start_cursor.recv_time_ns,
        )
        observed = MakerFillObservedSnapshot(
            value_code=snapshot.value_code,
            channel_seq=snapshot.channel_seq,
            recv_time_ns=snapshot.recv_time_ns,
            bid_price1=snapshot.bid_price1,
            bid_price2=snapshot.bid_price2,
            bid_lots1=snapshot.bid_lots1,
            bid_lots2=snapshot.bid_lots2,
        )
        event = self._index.derive_potential_fill_event(scalar_order, observed)
        self._events_by_raw_id[order.raw_order_fact_id] = event
        if not event.outcome_supported or event.makerfill_potential_fill is not True:
            return None
        fill_time_ns = event.makerfill_potential_fill_time_ns
        if fill_time_ns is None:
            raise RuntimeError("supported makerFill event lacks a potential fill time")
        rank = event.exact_target_rank or "unknown"
        return PotentialEntryFill(
            fill_time_ns=fill_time_ns,
            source_id=(
                f"{ADAPTER_VERSION}/{order.date}/{order.value_code}/"
                f"{snapshot.channel_seq}/{rank}"
            ),
            execution_truth="approximate",
            fill_cursor_exact=False,
        )


__all__ = ["S1MakerFillBridge"]
