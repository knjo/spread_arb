"""Raw-book bridge for S1 entry decisions and legacy makerFill snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .layered import EventCursor
from .s1_event_loop import ActualSendMakerSnapshot
from .s1_hedge import CausalBookState
from .s1_raw_book_adapter import (
    RawBookDayIndex,
    RawBookKey,
    RawBookRiskAdapter,
)

Venue = Literal["spot", "future"]


@dataclass(frozen=True, slots=True)
class S1RawEntryBookProvider:
    """Expose one compact day index through the entry-state protocol.

    The current formal spot source row is frozen as the actual-send makerFill
    snapshot.  TrialMatch is deliberately fail-closed: its row is a causal
    book barrier, not a quote snapshot that can support a BID1/BID2 label.
    """

    index: RawBookDayIndex
    _risk: RawBookRiskAdapter = field(init=False, repr=False)
    _spot_keys: Mapping[str, RawBookKey] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.index, RawBookDayIndex):
            raise TypeError("index must be a RawBookDayIndex")
        spot_keys = {
            key.value_code: key for key in self.index.keys if key.market == "spot"
        }
        if len(spot_keys) * 2 != len(self.index.keys):
            raise ValueError("raw-book index requires one spot/future pair per product")
        object.__setattr__(self, "_risk", self.index.as_risk_book_adapter())
        object.__setattr__(self, "_spot_keys", MappingProxyType(spot_keys))

    def state_as_of(
        self,
        venue: Venue,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        return self._risk.state_as_of(venue, product_id, cursor)

    def maker_snapshot_as_of(
        self,
        product_id: str,
        cursor: EventCursor,
    ) -> ActualSendMakerSnapshot | None:
        if not isinstance(cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        key = self._spot_key(product_id)
        indexed = self.index.indexed_event_as_of(key, cursor)
        if (
            indexed is None
            or indexed.event.trial_match
            or not indexed.event.formal_book
        ):
            return None
        source = indexed.source_cursor
        snapshot = self.index.spot_source_snapshot(
            key,
            source.channel_sequence,
            recv_time_ns=source.recv_time_ns,
        )
        if snapshot is None:
            raise RuntimeError("formal spot source has no unique scalar snapshot")
        return ActualSendMakerSnapshot(
            value_code=product_id,
            channel_seq=source.channel_sequence,
            recv_time_ns=source.recv_time_ns,
            bid_price1=_positive_or_none(snapshot.bid_price1),
            bid_price2=_positive_or_none(snapshot.bid_price2),
            bid_lots1=snapshot.bid_lots1,
            bid_lots2=snapshot.bid_lots2,
        )

    def _spot_key(self, product_id: str) -> RawBookKey:
        if not isinstance(product_id, str) or not product_id:
            raise ValueError("product_id must be a non-empty string")
        try:
            return self._spot_keys[product_id]
        except KeyError as error:
            raise ValueError(
                "product_id is not in the exact raw-book mapping"
            ) from error


def _positive_or_none(value: float) -> float | None:
    return None if value <= 0 else value


__all__ = ["S1RawEntryBookProvider"]
