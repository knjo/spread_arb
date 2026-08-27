"""Deterministic identities for candidate intents, physical orders, and aliases.

The physical order identity deliberately starts at the actual new-request send
cursor.  Policy, generation, epoch, and nominal timing are metadata and never
participate in that identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final, Literal

from .layered import EventCursor

MakerSide = Literal["bid", "ask"]

CANDIDATE_INTENT_ID_FIELDS: Final[tuple[str, ...]] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "stage",
    "maker_side",
    "absolute_price_tick",
    "intent_recv_time_ns",
    "intent_event_sequence",
    "intent_row_index",
)
RAW_ORDER_FACT_ID_FIELDS: Final[tuple[str, ...]] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "stage",
    "maker_side",
    "absolute_price_tick",
    "actual_start_recv_time_ns",
    "actual_start_event_sequence",
    "actual_start_row_index",
)
POLICY_ALIAS_ID_FIELDS: Final[tuple[str, ...]] = (
    "raw_order_fact_id",
    "policy_id",
)

_FORBIDDEN_RAW_ID_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "policy_id",
        "policy_epoch",
        "epoch",
        "generation",
        "generation_id",
        "nominal_recv_time_ns",
        "nominal_event_sequence",
        "nominal_row_index",
        "nominal_start_time_ns",
        "candidate_intent_id",
    }
)


def candidate_intent_id(
    *,
    Date: str,
    ValueCode: str,
    QuoteCode: str,
    route: str,
    stage: str,
    maker_side: MakerSide,
    absolute_price_tick: int,
    intent_cursor: EventCursor,
) -> str:
    """Identify one pre-send physical order intent, independent of policy."""

    common = _physical_fields(
        Date, ValueCode, QuoteCode, route, stage, maker_side, absolute_price_tick
    )
    cursor = _cursor_fields(intent_cursor, "intent")
    return _identity("candidate_intent", {**common, **cursor})


def raw_order_fact_id(
    *,
    Date: str,
    ValueCode: str,
    QuoteCode: str,
    route: str,
    stage: str,
    maker_side: MakerSide,
    absolute_price_tick: int,
    actual_start_cursor: EventCursor | None,
) -> str:
    """Identify a sent physical order from its complete actual start cursor.

    An unsent intent has no physical order fact and is rejected rather than
    receiving an identity based on nominal timing.
    """

    if actual_start_cursor is None:
        raise ValueError("raw_order_fact_id requires an actual send cursor")
    common = _physical_fields(
        Date, ValueCode, QuoteCode, route, stage, maker_side, absolute_price_tick
    )
    cursor = _cursor_fields(actual_start_cursor, "actual_start")
    return _identity("raw_order_fact", {**common, **cursor})


def policy_alias_id(*, raw_order_fact_id: str, policy_id: str) -> str:
    """Identify one policy's alias onto an already-sent physical order."""

    payload = {
        "raw_order_fact_id": _text("raw_order_fact_id", raw_order_fact_id),
        "policy_id": _text("policy_id", policy_id),
    }
    return _identity("policy_alias", payload)


def verify_candidate_intent_record(record: Mapping[str, object]) -> None:
    """Verify a materialized candidate record against its canonical ID."""

    _require_fields(record, ("candidate_intent_id", *CANDIDATE_INTENT_ID_FIELDS))
    expected = candidate_intent_id(
        **_candidate_arguments(record)  # type: ignore[arg-type]
    )
    _verify_value(record, "candidate_intent_id", expected)


def verify_raw_order_fact_record(record: Mapping[str, object]) -> None:
    """Verify that a physical-order record uses only actual-send identity."""

    _require_fields(record, ("raw_order_fact_id", *RAW_ORDER_FACT_ID_FIELDS))
    expected = raw_order_fact_id(**_raw_arguments(record))  # type: ignore[arg-type]
    _verify_value(record, "raw_order_fact_id", expected)


def verify_policy_alias_record(record: Mapping[str, object]) -> None:
    """Verify a policy-to-physical-order alias record."""

    _require_fields(record, ("policy_alias_id", *POLICY_ALIAS_ID_FIELDS))
    expected = policy_alias_id(
        raw_order_fact_id=record["raw_order_fact_id"],  # type: ignore[arg-type]
        policy_id=record["policy_id"],  # type: ignore[arg-type]
    )
    _verify_value(record, "policy_alias_id", expected)


def forbidden_raw_order_identity_fields() -> frozenset[str]:
    """Return metadata fields that must never enter raw-order ID payloads."""

    return _FORBIDDEN_RAW_ID_FIELDS


def _physical_fields(
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
    stage: str,
    maker_side: MakerSide,
    absolute_price_tick: int,
) -> dict[str, object]:
    if maker_side not in ("bid", "ask"):
        raise ValueError("maker_side must be 'bid' or 'ask'")
    if (
        isinstance(absolute_price_tick, bool)
        or not isinstance(absolute_price_tick, int)
        or absolute_price_tick <= 0
    ):
        raise ValueError("absolute_price_tick must be a positive integer")
    return {
        "Date": _text("Date", date),
        "ValueCode": _text("ValueCode", value_code),
        "QuoteCode": _text("QuoteCode", quote_code),
        "route": _text("route", route),
        "stage": _text("stage", stage),
        "maker_side": maker_side,
        "absolute_price_tick": absolute_price_tick,
    }


def _cursor_fields(cursor: EventCursor, prefix: str) -> dict[str, int]:
    if not isinstance(cursor, EventCursor):
        raise TypeError(f"{prefix}_cursor must be an EventCursor")
    return {
        f"{prefix}_recv_time_ns": cursor.recv_time_ns,
        f"{prefix}_event_sequence": cursor.event_sequence,
        f"{prefix}_row_index": cursor.row_index,
    }


def _identity(kind: str, fields: Mapping[str, object]) -> str:
    envelope = {"identity_kind": kind, "schema_version": 1, "fields": fields}
    encoded = json.dumps(
        envelope,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}_v1_{hashlib.sha256(encoded).hexdigest()}"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_fields(record: Mapping[str, object], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"identity record is missing fields: {missing}")


def _cursor_from_record(record: Mapping[str, object], prefix: str) -> EventCursor:
    values = tuple(
        record[f"{prefix}_{suffix}"]
        for suffix in ("recv_time_ns", "event_sequence", "row_index")
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{prefix} cursor fields must be integers")
    return EventCursor(*values)  # type: ignore[arg-type]


def _common_arguments(record: Mapping[str, object]) -> dict[str, object]:
    return {name: record[name] for name in CANDIDATE_INTENT_ID_FIELDS[:7]}


def _candidate_arguments(record: Mapping[str, object]) -> dict[str, object]:
    return {
        **_common_arguments(record),
        "intent_cursor": _cursor_from_record(record, "intent"),
    }


def _raw_arguments(record: Mapping[str, object]) -> dict[str, object]:
    return {
        **_common_arguments(record),
        "actual_start_cursor": _cursor_from_record(record, "actual_start"),
    }


def _verify_value(record: Mapping[str, object], field: str, expected: str) -> None:
    actual = record[field]
    if actual != expected:
        raise ValueError(f"{field} does not match canonical physical fields")


__all__ = [
    "CANDIDATE_INTENT_ID_FIELDS",
    "POLICY_ALIAS_ID_FIELDS",
    "RAW_ORDER_FACT_ID_FIELDS",
    "candidate_intent_id",
    "forbidden_raw_order_identity_fields",
    "policy_alias_id",
    "raw_order_fact_id",
    "verify_candidate_intent_record",
    "verify_policy_alias_record",
    "verify_raw_order_fact_record",
]
