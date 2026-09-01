"""Deterministic pre-trade capacity ledger for the S1 chronological loop.

The ledger models one-way spot-equivalent committed notional.  It reserves
capacity before an entry ``new`` request is sent and keeps the same notional
committed while risk moves through five mutually exclusive buckets:

``working_unfilled`` -> ``entry_partial`` -> ``hedge_pending`` ->
``paired_open`` -> ``exit_in_progress``.

Transfers never create or release capacity.  Only an admitted reservation
creates capacity; an actual cancel/session expiry releases unfilled leaves;
and a completed exit hedge releases an exiting paired unit.  All amounts are
integer TWD.  ``Decimal`` inputs are accepted only when they represent an
exact integer, and floats are rejected.

This module deliberately has no runner, clock scheduler, Polars, pricing, or
execution dependencies.  Its immutable transition rows are suitable for an
append-only audit artifact, and :func:`replay_capacity_transitions` verifies
every recorded before/after snapshot and both hard caps from scratch.

The compact checkpoint API intentionally omits closed accounts and historical
identity sets.  It is safe only when every partition is committed together
with an external exact ``UNIQUE`` identity registry covering both transition
IDs and historically admitted capacity IDs.  The registry receipt stored here
binds counts and the registry digest into the checkpoint; it does not, by
itself, attest that a historical identity was never reused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal
from typing import Literal

WORKING_UNFILLED = "working_unfilled"
ENTRY_PARTIAL = "entry_partial"
HEDGE_PENDING = "hedge_pending"
PAIRED_OPEN = "paired_open"
EXIT_IN_PROGRESS = "exit_in_progress"

BucketName = Literal[
    "working_unfilled",
    "entry_partial",
    "hedge_pending",
    "paired_open",
    "exit_in_progress",
]

BUCKET_NAMES: tuple[BucketName, ...] = (
    WORKING_UNFILLED,
    ENTRY_PARTIAL,
    HEDGE_PENDING,
    PAIRED_OPEN,
    EXIT_IN_PROGRESS,
)

DEFAULT_GLOBAL_CAP_TWD = 20_000_000
DEFAULT_PRODUCT_CAP_TWD = 10_000_000
CAPACITY_LEDGER_SEED_SCHEMA_VERSION = 1
CAPACITY_LEDGER_COMPACT_CHECKPOINT_SCHEMA_VERSION = 2
CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION = 1
CAPACITY_TRANSITION_RECORD_SCHEMA_VERSION = 1
_TRANSITION_CHAIN_DOMAIN = "capacity-ledger-transition-chain-v1"
_CHECKPOINT_DOMAIN = "capacity-ledger-checkpoint-v1"
_COMPACT_CHECKPOINT_DOMAIN = "capacity-ledger-compact-checkpoint-v2"
_INITIAL_TRANSITION_CHAIN_SHA256 = hashlib.sha256(
    _TRANSITION_CHAIN_DOMAIN.encode("utf-8")
).hexdigest()
_CAPACITY_TRANSITION_RECORD_TYPE = "capacity_transition"
_CAPACITY_IDENTITY_REGISTRY_RECEIPT_RECORD_TYPE = "capacity_identity_registry_receipt"
_CAPACITY_LEDGER_COMPACT_CHECKPOINT_RECORD_TYPE = "capacity_ledger_compact_checkpoint"


@dataclass(frozen=True)
class BucketBalances:
    """Committed integer TWD split across the five lifecycle buckets."""

    working_unfilled: int = 0
    entry_partial: int = 0
    hedge_pending: int = 0
    paired_open: int = 0
    exit_in_progress: int = 0

    @property
    def total_committed_notional_twd(self) -> int:
        return sum(getattr(self, name) for name in BUCKET_NAMES)

    def amount(self, bucket: BucketName) -> int:
        _validate_bucket(bucket)
        return int(getattr(self, bucket))

    def add(self, delta: BucketBalances) -> BucketBalances:
        values = {
            name: getattr(self, name) + getattr(delta, name) for name in BUCKET_NAMES
        }
        if any(value < 0 for value in values.values()):
            raise CapacityTransitionError("capacity bucket cannot become negative")
        return BucketBalances(**values)

    def as_dict(self, *, prefix: str = "") -> dict[str, int]:
        result = {f"{prefix}{name}_twd": getattr(self, name) for name in BUCKET_NAMES}
        result[f"{prefix}total_committed_notional_twd"] = (
            self.total_committed_notional_twd
        )
        return result


ZERO_BALANCES = BucketBalances()


@dataclass(frozen=True)
class CapacityAccountSeed:
    """One admitted capacity identity persisted across session boundaries."""

    capacity_id: str
    product_id: str
    balances: BucketBalances


@dataclass(frozen=True)
class CapacityIdentityRegistryReceipt:
    """Receipt from an external exact ``UNIQUE`` identity registry.

    The external registry must contain every transition ID and every admitted
    capacity ID through this receipt.  ``registry_sha256`` is the canonical
    digest produced by that registry.  This value is integrity-bound into a
    compact checkpoint, but it is not a replacement for the exact registry and
    cannot independently prove historical duplicate freedom.
    """

    schema_version: int
    transition_count: int
    admitted_count: int
    registry_sha256: str


@dataclass(frozen=True)
class CapacityLedgerCompactCheckpoint:
    """Integrity-checked v2 restart state containing only live accounts.

    Closed accounts and historical transition IDs are deliberately omitted.
    Consequently this checkpoint may be used only with the exact external
    ``UNIQUE`` identity registry represented by ``identity_registry_receipt``.
    Local replay checks IDs inside the resumed partition; the registry must
    reject collisions with all earlier partitions before the new receipt is
    accepted.  Neither this checkpoint nor its receipt alone attests historical
    duplicate freedom.
    """

    schema_version: int
    through_date: str
    global_cap_twd: int
    product_cap_twd: int
    live_accounts: tuple[CapacityAccountSeed, ...]
    transition_sequence_offset: int
    last_timestamp_ns: int | None
    last_event_sequence: int | None
    last_row_index: int | None
    transition_chain_sha256: str
    identity_registry_receipt: CapacityIdentityRegistryReceipt
    checkpoint_sha256: str

    @property
    def last_cursor(self) -> tuple[int, int, int] | None:
        if self.last_timestamp_ns is None:
            return None
        assert self.last_event_sequence is not None
        assert self.last_row_index is not None
        return (
            self.last_timestamp_ns,
            self.last_event_sequence,
            self.last_row_index,
        )

    @property
    def account_balances(self) -> dict[str, BucketBalances]:
        return {account.capacity_id: account.balances for account in self.live_accounts}

    @property
    def account_products(self) -> dict[str, str]:
        return {
            account.capacity_id: account.product_id for account in self.live_accounts
        }

    @property
    def product_balances(self) -> dict[str, BucketBalances]:
        products, _ = _derive_totals_from_account_seeds(self.live_accounts)
        return products

    @property
    def global_balances(self) -> BucketBalances:
        _, global_balances = _derive_totals_from_account_seeds(self.live_accounts)
        return global_balances


@dataclass(frozen=True)
class CapacityLedgerSeed:
    """Immutable, integrity-checked restart state for one capacity ledger.

    Product and global balances are deliberately omitted from the serialized
    fields: both are derived from ``accounts`` during validation and restore.
    ``seen_transition_ids`` is retained in full because a hash alone cannot
    prove that a caller-supplied transition ID has not appeared before.
    """

    schema_version: int
    through_date: str
    global_cap_twd: int
    product_cap_twd: int
    accounts: tuple[CapacityAccountSeed, ...]
    transition_sequence_offset: int
    last_timestamp_ns: int | None
    last_event_sequence: int | None
    last_row_index: int | None
    seen_transition_ids: tuple[str, ...]
    transition_chain_sha256: str
    checkpoint_sha256: str

    @property
    def last_cursor(self) -> tuple[int, int, int] | None:
        if self.last_timestamp_ns is None:
            return None
        assert self.last_event_sequence is not None
        assert self.last_row_index is not None
        return (
            self.last_timestamp_ns,
            self.last_event_sequence,
            self.last_row_index,
        )

    @property
    def account_balances(self) -> dict[str, BucketBalances]:
        return {account.capacity_id: account.balances for account in self.accounts}

    @property
    def account_products(self) -> dict[str, str]:
        return {account.capacity_id: account.product_id for account in self.accounts}

    @property
    def product_balances(self) -> dict[str, BucketBalances]:
        products, _ = _derive_totals_from_account_seeds(self.accounts)
        return products

    @property
    def global_balances(self) -> BucketBalances:
        _, global_balances = _derive_totals_from_account_seeds(self.accounts)
        return global_balances


@dataclass(frozen=True)
class CapacityTransition:
    """One immutable, replayable mutation or blocked reservation attempt."""

    sequence: int
    transition_id: str
    timestamp_ns: int
    event_sequence: int
    row_index: int
    event_type: str
    capacity_id: str
    product_id: str
    status: str
    admitted: bool | None
    requested_notional_twd: int | None
    moved_notional_twd: int
    from_bucket: BucketName | None
    to_bucket: BucketName | None
    delta: BucketBalances
    account_before: BucketBalances
    account_after: BucketBalances
    product_before: BucketBalances
    product_after: BucketBalances
    global_before: BucketBalances
    global_after: BucketBalances
    global_cap_twd: int
    product_cap_twd: int

    def as_dict(self) -> dict[str, object]:
        """Return a flat representation suitable for a later table writer."""

        result: dict[str, object] = {
            "sequence": self.sequence,
            "transition_id": self.transition_id,
            "timestamp_ns": self.timestamp_ns,
            "event_sequence": self.event_sequence,
            "row_index": self.row_index,
            "event_type": self.event_type,
            "capacity_id": self.capacity_id,
            "product_id": self.product_id,
            "status": self.status,
            "admitted": self.admitted,
            "requested_notional_twd": self.requested_notional_twd,
            "moved_notional_twd": self.moved_notional_twd,
            "from_bucket": self.from_bucket,
            "to_bucket": self.to_bucket,
            "global_cap_twd": self.global_cap_twd,
            "product_cap_twd": self.product_cap_twd,
        }
        result.update(self.delta.as_dict(prefix="delta_"))
        result.update(self.account_before.as_dict(prefix="account_before_"))
        result.update(self.account_after.as_dict(prefix="account_after_"))
        result.update(self.product_before.as_dict(prefix="product_before_"))
        result.update(self.product_after.as_dict(prefix="product_after_"))
        result.update(self.global_before.as_dict(prefix="global_before_"))
        result.update(self.global_after.as_dict(prefix="global_after_"))
        return result


@dataclass(frozen=True)
class ReservationDecision:
    admitted: bool
    status: str
    transition: CapacityTransition


@dataclass(frozen=True)
class CapacityReplayResult:
    """Final state reconstructed solely from transition rows."""

    account_balances: dict[str, BucketBalances]
    account_products: dict[str, str]
    product_balances: dict[str, BucketBalances]
    global_balances: BucketBalances
    last_timestamp_ns: int | None
    last_event_sequence: int | None
    last_row_index: int | None
    last_sequence: int
    seen_transition_ids: tuple[str, ...]
    transition_chain_sha256: str


@dataclass(frozen=True)
class _CapacityVerificationKey:
    """Exact live-ledger state covered by one successful verification."""

    mutation_revision: int
    global_cap_twd: int
    product_cap_twd: int
    accounts: tuple[tuple[str, str, BucketBalances], ...]
    product_balances: tuple[tuple[str, BucketBalances], ...]
    global_balances: BucketBalances
    transitions: tuple[CapacityTransition, ...]
    transition_ids: frozenset[str]
    last_cursor: tuple[int, int, int] | None
    sequence_offset: int
    transition_chain_sha256: str
    replay_seed: CapacityLedgerSeed | None
    replay_compact_checkpoint: CapacityLedgerCompactCheckpoint | None
    resume_boundary_pending: bool


@dataclass(frozen=True)
class _CapacityVerificationCache:
    key: _CapacityVerificationKey
    replay: CapacityReplayResult


@dataclass
class _Account:
    product_id: str
    balances: BucketBalances


class CapacityLedgerError(ValueError):
    """Base class for invalid inputs or lifecycle operations."""


class CapacityTransitionError(CapacityLedgerError):
    """Raised when a transition violates the lifecycle state."""


class CapacityReplayError(CapacityLedgerError):
    """Raised when append-only rows cannot be deterministically replayed."""


class CapacityCodecError(CapacityReplayError):
    """Raised when a persisted capacity record is not canonical JSON data."""


def encode_capacity_transition(
    transition: CapacityTransition,
) -> dict[str, object]:
    """Encode one transition as a canonical, nested JSON-safe object record."""

    if not isinstance(transition, CapacityTransition):
        raise TypeError("transition must be a CapacityTransition")
    _validate_transition_for_codec(transition)
    return {
        "record_type": _CAPACITY_TRANSITION_RECORD_TYPE,
        "schema_version": CAPACITY_TRANSITION_RECORD_SCHEMA_VERSION,
        "sequence": transition.sequence,
        "transition_id": transition.transition_id,
        "timestamp_ns": transition.timestamp_ns,
        "event_sequence": transition.event_sequence,
        "row_index": transition.row_index,
        "event_type": transition.event_type,
        "capacity_id": transition.capacity_id,
        "product_id": transition.product_id,
        "status": transition.status,
        "admitted": transition.admitted,
        "requested_notional_twd": transition.requested_notional_twd,
        "moved_notional_twd": transition.moved_notional_twd,
        "from_bucket": transition.from_bucket,
        "to_bucket": transition.to_bucket,
        "delta": _encode_bucket_balances(transition.delta),
        "account_before": _encode_bucket_balances(transition.account_before),
        "account_after": _encode_bucket_balances(transition.account_after),
        "product_before": _encode_bucket_balances(transition.product_before),
        "product_after": _encode_bucket_balances(transition.product_after),
        "global_before": _encode_bucket_balances(transition.global_before),
        "global_after": _encode_bucket_balances(transition.global_after),
        "global_cap_twd": transition.global_cap_twd,
        "product_cap_twd": transition.product_cap_twd,
    }


def decode_capacity_transition(record: Mapping[str, object]) -> CapacityTransition:
    """Strictly reconstruct one immutable transition from a JSON object."""

    if not isinstance(record, Mapping):
        raise TypeError("capacity transition record must be a mapping")
    expected_keys = {
        "record_type",
        "schema_version",
        "sequence",
        "transition_id",
        "timestamp_ns",
        "event_sequence",
        "row_index",
        "event_type",
        "capacity_id",
        "product_id",
        "status",
        "admitted",
        "requested_notional_twd",
        "moved_notional_twd",
        "from_bucket",
        "to_bucket",
        "delta",
        "account_before",
        "account_after",
        "product_before",
        "product_after",
        "global_before",
        "global_after",
        "global_cap_twd",
        "product_cap_twd",
    }
    _require_exact_record_keys(record, expected_keys, "capacity transition")
    _require_record_tag(
        record,
        expected_type=_CAPACITY_TRANSITION_RECORD_TYPE,
        expected_schema=CAPACITY_TRANSITION_RECORD_SCHEMA_VERSION,
        path="capacity transition",
    )

    admitted = record["admitted"]
    if admitted is not None and type(admitted) is not bool:
        raise CapacityCodecError("capacity transition admitted must be boolean or null")
    requested_notional = _decode_optional_record_integer(
        record["requested_notional_twd"],
        "capacity transition requested_notional_twd",
    )
    transition = CapacityTransition(
        sequence=_decode_record_integer(
            record["sequence"], "capacity transition sequence"
        ),
        transition_id=_decode_record_string(
            record["transition_id"], "capacity transition transition_id"
        ),
        timestamp_ns=_decode_record_integer(
            record["timestamp_ns"], "capacity transition timestamp_ns"
        ),
        event_sequence=_decode_record_integer(
            record["event_sequence"], "capacity transition event_sequence"
        ),
        row_index=_decode_record_integer(
            record["row_index"], "capacity transition row_index"
        ),
        event_type=_decode_record_string(
            record["event_type"], "capacity transition event_type"
        ),
        capacity_id=_decode_record_string(
            record["capacity_id"], "capacity transition capacity_id"
        ),
        product_id=_decode_record_string(
            record["product_id"], "capacity transition product_id"
        ),
        status=_decode_record_string(record["status"], "capacity transition status"),
        admitted=admitted,
        requested_notional_twd=requested_notional,
        moved_notional_twd=_decode_record_integer(
            record["moved_notional_twd"],
            "capacity transition moved_notional_twd",
        ),
        from_bucket=_decode_optional_bucket(
            record["from_bucket"], "capacity transition from_bucket"
        ),
        to_bucket=_decode_optional_bucket(
            record["to_bucket"], "capacity transition to_bucket"
        ),
        delta=_decode_bucket_balances(record["delta"], "capacity transition delta"),
        account_before=_decode_bucket_balances(
            record["account_before"], "capacity transition account_before"
        ),
        account_after=_decode_bucket_balances(
            record["account_after"], "capacity transition account_after"
        ),
        product_before=_decode_bucket_balances(
            record["product_before"], "capacity transition product_before"
        ),
        product_after=_decode_bucket_balances(
            record["product_after"], "capacity transition product_after"
        ),
        global_before=_decode_bucket_balances(
            record["global_before"], "capacity transition global_before"
        ),
        global_after=_decode_bucket_balances(
            record["global_after"], "capacity transition global_after"
        ),
        global_cap_twd=_decode_record_integer(
            record["global_cap_twd"], "capacity transition global_cap_twd"
        ),
        product_cap_twd=_decode_record_integer(
            record["product_cap_twd"], "capacity transition product_cap_twd"
        ),
    )
    _validate_transition_for_codec(transition)
    return transition


def encode_capacity_transitions(
    transitions: Sequence[CapacityTransition],
) -> list[dict[str, object]]:
    """Encode an ordered transition stream as a JSON-safe array."""

    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise TypeError("capacity transitions must be a sequence")
    return [encode_capacity_transition(transition) for transition in transitions]


def decode_capacity_transitions(records: object) -> tuple[CapacityTransition, ...]:
    """Decode a JSON array without accepting an in-memory tuple as JSON."""

    if type(records) is not list:
        raise CapacityCodecError("capacity transition records must be a JSON array")
    return tuple(decode_capacity_transition(record) for record in records)


def encode_capacity_identity_registry_receipt(
    receipt: CapacityIdentityRegistryReceipt,
) -> dict[str, object]:
    """Encode an exact-registry receipt as a canonical JSON-safe object."""

    if not isinstance(receipt, CapacityIdentityRegistryReceipt):
        raise TypeError("receipt must be a CapacityIdentityRegistryReceipt")
    try:
        _validate_identity_registry_receipt(receipt)
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error
    return {
        "record_type": _CAPACITY_IDENTITY_REGISTRY_RECEIPT_RECORD_TYPE,
        "schema_version": receipt.schema_version,
        "transition_count": receipt.transition_count,
        "admitted_count": receipt.admitted_count,
        "registry_sha256": receipt.registry_sha256,
    }


def decode_capacity_identity_registry_receipt(
    record: Mapping[str, object],
) -> CapacityIdentityRegistryReceipt:
    """Strictly reconstruct an exact-registry receipt from a JSON object."""

    if not isinstance(record, Mapping):
        raise TypeError("capacity identity registry receipt record must be a mapping")
    expected_keys = {
        "record_type",
        "schema_version",
        "transition_count",
        "admitted_count",
        "registry_sha256",
    }
    _require_exact_record_keys(record, expected_keys, "identity registry receipt")
    _require_record_tag(
        record,
        expected_type=_CAPACITY_IDENTITY_REGISTRY_RECEIPT_RECORD_TYPE,
        expected_schema=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
        path="identity registry receipt",
    )
    receipt = CapacityIdentityRegistryReceipt(
        schema_version=_decode_record_integer(
            record["schema_version"], "identity registry receipt schema_version"
        ),
        transition_count=_decode_record_integer(
            record["transition_count"], "identity registry receipt transition_count"
        ),
        admitted_count=_decode_record_integer(
            record["admitted_count"], "identity registry receipt admitted_count"
        ),
        registry_sha256=_decode_record_string(
            record["registry_sha256"], "identity registry receipt registry_sha256"
        ),
    )
    try:
        _validate_identity_registry_receipt(receipt)
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error
    return receipt


def encode_capacity_ledger_compact_checkpoint(
    checkpoint: CapacityLedgerCompactCheckpoint,
) -> dict[str, object]:
    """Encode a verified compact checkpoint as canonical JSON-safe data."""

    if not isinstance(checkpoint, CapacityLedgerCompactCheckpoint):
        raise TypeError("checkpoint must be a CapacityLedgerCompactCheckpoint")
    try:
        _validate_capacity_ledger_compact_checkpoint(checkpoint)
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error
    last_cursor = checkpoint.last_cursor
    encoded_cursor: dict[str, int] | None = None
    if last_cursor is not None:
        encoded_cursor = {
            "timestamp_ns": last_cursor[0],
            "event_sequence": last_cursor[1],
            "row_index": last_cursor[2],
        }
    return {
        "record_type": _CAPACITY_LEDGER_COMPACT_CHECKPOINT_RECORD_TYPE,
        "schema_version": checkpoint.schema_version,
        "through_date": checkpoint.through_date,
        "global_cap_twd": checkpoint.global_cap_twd,
        "product_cap_twd": checkpoint.product_cap_twd,
        "live_accounts": [
            {
                "capacity_id": account.capacity_id,
                "product_id": account.product_id,
                "balances": _encode_bucket_balances(account.balances),
            }
            for account in checkpoint.live_accounts
        ],
        "transition_sequence_offset": checkpoint.transition_sequence_offset,
        "last_cursor": encoded_cursor,
        "transition_chain_sha256": checkpoint.transition_chain_sha256,
        "identity_registry_receipt": encode_capacity_identity_registry_receipt(
            checkpoint.identity_registry_receipt
        ),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }


def decode_capacity_ledger_compact_checkpoint(
    record: Mapping[str, object],
) -> CapacityLedgerCompactCheckpoint:
    """Strictly reconstruct and integrity-check a compact checkpoint."""

    if not isinstance(record, Mapping):
        raise TypeError("capacity compact checkpoint record must be a mapping")
    expected_keys = {
        "record_type",
        "schema_version",
        "through_date",
        "global_cap_twd",
        "product_cap_twd",
        "live_accounts",
        "transition_sequence_offset",
        "last_cursor",
        "transition_chain_sha256",
        "identity_registry_receipt",
        "checkpoint_sha256",
    }
    _require_exact_record_keys(record, expected_keys, "capacity compact checkpoint")
    _require_record_tag(
        record,
        expected_type=_CAPACITY_LEDGER_COMPACT_CHECKPOINT_RECORD_TYPE,
        expected_schema=CAPACITY_LEDGER_COMPACT_CHECKPOINT_SCHEMA_VERSION,
        path="capacity compact checkpoint",
    )
    live_account_records = record["live_accounts"]
    if type(live_account_records) is not list:
        raise CapacityCodecError(
            "capacity compact checkpoint live_accounts must be a JSON array"
        )
    live_accounts: list[CapacityAccountSeed] = []
    for index, account_record in enumerate(live_account_records):
        path = f"capacity compact checkpoint live_accounts[{index}]"
        if not isinstance(account_record, Mapping):
            raise CapacityCodecError(f"{path} must be a JSON object")
        _require_exact_record_keys(
            account_record,
            {"capacity_id", "product_id", "balances"},
            path,
        )
        live_accounts.append(
            CapacityAccountSeed(
                capacity_id=_decode_record_string(
                    account_record["capacity_id"], f"{path} capacity_id"
                ),
                product_id=_decode_record_string(
                    account_record["product_id"], f"{path} product_id"
                ),
                balances=_decode_bucket_balances(
                    account_record["balances"], f"{path} balances"
                ),
            )
        )
    last_timestamp_ns, last_event_sequence, last_row_index = _decode_checkpoint_cursor(
        record["last_cursor"]
    )
    receipt_record = record["identity_registry_receipt"]
    if not isinstance(receipt_record, Mapping):
        raise CapacityCodecError(
            "capacity compact checkpoint identity_registry_receipt must be a JSON object"
        )
    checkpoint = CapacityLedgerCompactCheckpoint(
        schema_version=_decode_record_integer(
            record["schema_version"], "capacity compact checkpoint schema_version"
        ),
        through_date=_decode_record_string(
            record["through_date"], "capacity compact checkpoint through_date"
        ),
        global_cap_twd=_decode_record_integer(
            record["global_cap_twd"], "capacity compact checkpoint global_cap_twd"
        ),
        product_cap_twd=_decode_record_integer(
            record["product_cap_twd"], "capacity compact checkpoint product_cap_twd"
        ),
        live_accounts=tuple(live_accounts),
        transition_sequence_offset=_decode_record_integer(
            record["transition_sequence_offset"],
            "capacity compact checkpoint transition_sequence_offset",
        ),
        last_timestamp_ns=last_timestamp_ns,
        last_event_sequence=last_event_sequence,
        last_row_index=last_row_index,
        transition_chain_sha256=_decode_record_string(
            record["transition_chain_sha256"],
            "capacity compact checkpoint transition_chain_sha256",
        ),
        identity_registry_receipt=decode_capacity_identity_registry_receipt(
            receipt_record
        ),
        checkpoint_sha256=_decode_record_string(
            record["checkpoint_sha256"],
            "capacity compact checkpoint checkpoint_sha256",
        ),
    )
    try:
        _validate_capacity_ledger_compact_checkpoint(checkpoint)
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error
    return checkpoint


class CapacityLedger:
    """Pure integer-TWD capacity state machine.

    A ``capacity_id`` identifies one entry reservation throughout its complete
    lifecycle.  Blocked attempts do not create an account and may be retried
    under the same ``capacity_id`` with a new ``transition_id``.  Once an
    attempt is admitted, that identity can never be reserved a second time,
    even after its balances reach zero.
    """

    def __init__(
        self,
        *,
        global_cap_twd: int | Decimal = DEFAULT_GLOBAL_CAP_TWD,
        product_cap_twd: int | Decimal = DEFAULT_PRODUCT_CAP_TWD,
    ) -> None:
        self.global_cap_twd = _integer_twd(global_cap_twd, "global_cap_twd")
        self.product_cap_twd = _integer_twd(product_cap_twd, "product_cap_twd")
        if self.product_cap_twd > self.global_cap_twd:
            raise CapacityLedgerError("product cap cannot exceed global cap")
        self._accounts: dict[str, _Account] = {}
        self._product_balances: dict[str, BucketBalances] = {}
        self._global_balances = ZERO_BALANCES
        self._transitions: list[CapacityTransition] = []
        self._transition_ids: set[str] = set()
        self._last_cursor: tuple[int, int, int] | None = None
        self._sequence_offset = 0
        self._transition_chain_sha256 = _INITIAL_TRANSITION_CHAIN_SHA256
        self._replay_seed: CapacityLedgerSeed | None = None
        self._replay_compact_checkpoint: CapacityLedgerCompactCheckpoint | None = None
        self._resume_boundary_pending = False
        self._seed_through_date: str | None = None
        self._mutation_revision = 0
        self._verification_cache: _CapacityVerificationCache | None = None

    @classmethod
    def from_seed(cls, seed: CapacityLedgerSeed) -> CapacityLedger:
        """Restore a ledger while keeping the next day's rows as a daily delta."""

        product_balances, global_balances = _validate_capacity_ledger_seed(seed)
        ledger = cls(
            global_cap_twd=seed.global_cap_twd,
            product_cap_twd=seed.product_cap_twd,
        )
        ledger._accounts = {
            account.capacity_id: _Account(account.product_id, account.balances)
            for account in seed.accounts
        }
        ledger._product_balances = product_balances
        ledger._global_balances = global_balances
        ledger._transition_ids = set(seed.seen_transition_ids)
        ledger._last_cursor = seed.last_cursor
        ledger._sequence_offset = seed.transition_sequence_offset
        ledger._transition_chain_sha256 = seed.transition_chain_sha256
        ledger._replay_seed = seed
        ledger._resume_boundary_pending = seed.last_cursor is not None
        ledger._seed_through_date = seed.through_date
        ledger.verify()
        return ledger

    @classmethod
    def from_compact_checkpoint(
        cls,
        checkpoint: CapacityLedgerCompactCheckpoint,
    ) -> CapacityLedger:
        """Restore live state while keeping only resumed-partition identities.

        This API is safe only when ``checkpoint.identity_registry_receipt`` was
        produced by an external exact ``UNIQUE`` registry covering every prior
        transition ID and admitted capacity ID.  The restored ledger rejects
        duplicates within its new local partition, but cannot attest collisions
        against identities omitted from the compact checkpoint.
        """

        product_balances, global_balances = (
            _validate_capacity_ledger_compact_checkpoint(checkpoint)
        )
        ledger = cls(
            global_cap_twd=checkpoint.global_cap_twd,
            product_cap_twd=checkpoint.product_cap_twd,
        )
        ledger._accounts = {
            account.capacity_id: _Account(account.product_id, account.balances)
            for account in checkpoint.live_accounts
        }
        ledger._product_balances = product_balances
        ledger._global_balances = global_balances
        ledger._transition_ids = set()
        ledger._last_cursor = checkpoint.last_cursor
        ledger._sequence_offset = checkpoint.transition_sequence_offset
        ledger._transition_chain_sha256 = checkpoint.transition_chain_sha256
        ledger._replay_compact_checkpoint = checkpoint
        ledger._resume_boundary_pending = checkpoint.last_cursor is not None
        ledger._seed_through_date = checkpoint.through_date
        ledger.verify()
        return ledger

    @property
    def transitions(self) -> tuple[CapacityTransition, ...]:
        return tuple(self._transitions)

    @property
    def historical_transition_identities_omitted(self) -> bool:
        """Whether this ledger resumed from an externally attested compact state."""

        return self._replay_compact_checkpoint is not None

    @property
    def global_balances(self) -> BucketBalances:
        return self._global_balances

    def product_balances(self, product_id: str) -> BucketBalances:
        return self._product_balances.get(
            _identifier(product_id, "product_id"), ZERO_BALANCES
        )

    def account_balances(self, capacity_id: str) -> BucketBalances:
        identifier = _identifier(capacity_id, "capacity_id")
        account = self._accounts.get(identifier)
        return account.balances if account is not None else ZERO_BALANCES

    def to_seed(self, through_date: str) -> CapacityLedgerSeed:
        """Return a canonical checkpoint after verifying all local delta rows."""

        if self._replay_compact_checkpoint is not None:
            raise CapacityLedgerError(
                "legacy cumulative seed cannot be produced from compact state"
            )
        through_date = _through_date(through_date)
        if (
            self._seed_through_date is not None
            and through_date < self._seed_through_date
        ):
            raise CapacityLedgerError("checkpoint through_date cannot regress")
        self.verify()
        accounts = tuple(
            CapacityAccountSeed(
                capacity_id=capacity_id,
                product_id=account.product_id,
                balances=account.balances,
            )
            for capacity_id, account in sorted(self._accounts.items())
        )
        values: dict[str, object] = {
            "schema_version": CAPACITY_LEDGER_SEED_SCHEMA_VERSION,
            "through_date": through_date,
            "global_cap_twd": self.global_cap_twd,
            "product_cap_twd": self.product_cap_twd,
            "accounts": accounts,
            "transition_sequence_offset": self._sequence_offset
            + len(self._transitions),
            "last_timestamp_ns": (
                self._last_cursor[0] if self._last_cursor is not None else None
            ),
            "last_event_sequence": (
                self._last_cursor[1] if self._last_cursor is not None else None
            ),
            "last_row_index": (
                self._last_cursor[2] if self._last_cursor is not None else None
            ),
            "seen_transition_ids": tuple(sorted(self._transition_ids)),
            "transition_chain_sha256": self._transition_chain_sha256,
        }
        seed = CapacityLedgerSeed(
            **values,
            checkpoint_sha256=_checkpoint_sha256(values),
        )
        _validate_capacity_ledger_seed(seed)
        return seed

    def to_compact_checkpoint(
        self,
        through_date: str,
        *,
        identity_registry_receipt: CapacityIdentityRegistryReceipt,
    ) -> CapacityLedgerCompactCheckpoint:
        """Return v2 live state bound to an exact external-registry receipt.

        The caller must first commit this partition's transition IDs and newly
        admitted capacity IDs to an external exact ``UNIQUE`` registry, then
        provide the resulting receipt.  This method checks receipt counts and
        binds its SHA-256 into the checkpoint.  It cannot inspect the registry,
        so the resulting checkpoint alone does not attest historical duplicate
        freedom.
        """

        through_date = _through_date(through_date)
        if (
            self._seed_through_date is not None
            and through_date < self._seed_through_date
        ):
            raise CapacityLedgerError("checkpoint through_date cannot regress")
        self.verify()
        _validate_identity_registry_receipt(identity_registry_receipt)

        transition_count = self._sequence_offset + len(self._transitions)
        if identity_registry_receipt.transition_count != transition_count:
            raise CapacityReplayError(
                "identity registry transition count does not match sequence offset"
            )
        if self._replay_compact_checkpoint is None:
            expected_admitted_count = len(self._accounts)
        else:
            expected_admitted_count = (
                self._replay_compact_checkpoint.identity_registry_receipt.admitted_count
                + sum(
                    transition.event_type == "new_reservation_attempt"
                    and transition.admitted is True
                    for transition in self._transitions
                )
            )
        if identity_registry_receipt.admitted_count != expected_admitted_count:
            raise CapacityReplayError(
                "identity registry admitted count does not match ledger history"
            )

        live_accounts = tuple(
            CapacityAccountSeed(
                capacity_id=capacity_id,
                product_id=account.product_id,
                balances=account.balances,
            )
            for capacity_id, account in sorted(self._accounts.items())
            if account.balances.total_committed_notional_twd != 0
        )
        values: dict[str, object] = {
            "schema_version": CAPACITY_LEDGER_COMPACT_CHECKPOINT_SCHEMA_VERSION,
            "through_date": through_date,
            "global_cap_twd": self.global_cap_twd,
            "product_cap_twd": self.product_cap_twd,
            "live_accounts": live_accounts,
            "transition_sequence_offset": transition_count,
            "last_timestamp_ns": (
                self._last_cursor[0] if self._last_cursor is not None else None
            ),
            "last_event_sequence": (
                self._last_cursor[1] if self._last_cursor is not None else None
            ),
            "last_row_index": (
                self._last_cursor[2] if self._last_cursor is not None else None
            ),
            "transition_chain_sha256": self._transition_chain_sha256,
            "identity_registry_receipt": identity_registry_receipt,
        }
        checkpoint = CapacityLedgerCompactCheckpoint(
            **values,
            checkpoint_sha256=_compact_checkpoint_sha256(values),
        )
        _validate_capacity_ledger_compact_checkpoint(checkpoint)
        return checkpoint

    def attempt_new_reservation(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        product_id: str,
        requested_notional_twd: int | Decimal,
    ) -> ReservationDecision:
        """Attempt a pre-send reservation without overbooking either cap."""

        transition_id, cursor = self._validate_common(
            transition_id, timestamp_ns, event_sequence, row_index
        )
        timestamp_ns, event_sequence, row_index = cursor
        capacity_id = _identifier(capacity_id, "capacity_id")
        product_id = _identifier(product_id, "product_id")
        requested = _integer_twd(requested_notional_twd, "requested_notional_twd")
        if capacity_id in self._accounts:
            raise CapacityTransitionError(
                f"capacity_id already admitted: {capacity_id}"
            )

        account_before = ZERO_BALANCES
        product_before = self._product_balances.get(product_id, ZERO_BALANCES)
        global_before = self._global_balances
        global_blocked = (
            global_before.total_committed_notional_twd + requested > self.global_cap_twd
        )
        product_blocked = (
            product_before.total_committed_notional_twd + requested
            > self.product_cap_twd
        )
        admitted = not global_blocked and not product_blocked
        if admitted:
            status = "admitted"
            delta = _single_bucket_delta(WORKING_UNFILLED, requested)
            account_after = account_before.add(delta)
            product_after = product_before.add(delta)
            global_after = global_before.add(delta)
            self._accounts[capacity_id] = _Account(product_id, account_after)
            self._product_balances[product_id] = product_after
            self._global_balances = global_after
        else:
            if global_blocked and product_blocked:
                status = "blocked_both_caps"
            elif global_blocked:
                status = "blocked_global_cap"
            else:
                status = "blocked_product_cap"
            delta = ZERO_BALANCES
            account_after = account_before
            product_after = product_before
            global_after = global_before

        transition = self._append(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            event_type="new_reservation_attempt",
            capacity_id=capacity_id,
            product_id=product_id,
            status=status,
            admitted=admitted,
            requested_notional_twd=requested,
            moved_notional_twd=requested if admitted else 0,
            from_bucket=None,
            to_bucket=WORKING_UNFILLED,
            delta=delta,
            account_before=account_before,
            account_after=account_after,
            product_before=product_before,
            product_after=product_after,
            global_before=global_before,
            global_after=global_after,
        )
        return ReservationDecision(admitted, status, transition)

    def record_entry_partial_fill(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        filled_notional_twd: int | Decimal,
    ) -> CapacityTransition:
        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=filled_notional_twd,
            from_bucket=WORKING_UNFILLED,
            to_bucket=ENTRY_PARTIAL,
            event_type="entry_partial_fill",
            status="working_to_entry_partial",
        )

    def record_entry_full_fill(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
    ) -> CapacityTransition:
        """Move all currently unfilled reservation directly to hedge pending."""

        account = self._required_account(capacity_id)
        amount = account.balances.working_unfilled
        if amount <= 0:
            raise CapacityTransitionError("no working leaves remain to fill")
        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=amount,
            from_bucket=WORKING_UNFILLED,
            to_bucket=HEDGE_PENDING,
            event_type="entry_full_fill",
            status="working_to_hedge_pending",
        )

    def move_entry_partial_to_hedge_pending(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=ENTRY_PARTIAL,
            to_bucket=HEDGE_PENDING,
            event_type="entry_partial_hedge_trigger",
            status="entry_partial_to_hedge_pending",
        )

    def complete_entry_hedge(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=HEDGE_PENDING,
            to_bucket=PAIRED_OPEN,
            event_type="entry_hedge_complete",
            status="hedge_pending_to_paired_open",
        )

    def begin_exit(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=PAIRED_OPEN,
            to_bucket=EXIT_IN_PROGRESS,
            event_type="exit_started",
            status="paired_open_to_exit_in_progress",
        )

    def release_working_leaves(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        reason: Literal["actual_cancel", "session_expiry"],
    ) -> CapacityTransition:
        """Release all unfilled leaves after an observable terminal event."""

        if reason not in ("actual_cancel", "session_expiry"):
            raise CapacityTransitionError(
                "working leaves release requires actual_cancel or session_expiry"
            )
        account = self._required_account(capacity_id)
        amount = account.balances.working_unfilled
        if amount <= 0:
            raise CapacityTransitionError("no working leaves remain to release")
        return self._release(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=amount,
            from_bucket=WORKING_UNFILLED,
            event_type="working_leaves_release",
            status=reason,
        )

    def complete_exit_hedge(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        """Release capacity only after the full exit hedge has executed."""

        return self._release(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=EXIT_IN_PROGRESS,
            event_type="exit_hedge_complete",
            status="exit_in_progress_released",
        )

    def complete_expiry_basis_zero(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
    ) -> CapacityTransition:
        """Release one exactly paired residual at contractual expiry.

        This is a non-executable accounting terminal, not an order or fill.
        It is legal only when the account contains paired-open capacity and no
        working, naked, or exit-in-progress bucket.  The amount is derived
        from the account rather than supplied by a caller.
        """

        account = self._required_account(capacity_id)
        amount = account.balances.paired_open
        if amount <= 0:
            raise CapacityTransitionError(
                "expiry basis-zero release requires paired-open capacity"
            )
        if account.balances.total_committed_notional_twd != amount:
            raise CapacityTransitionError(
                "expiry basis-zero release requires an exactly paired residual"
            )
        return self._release(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=amount,
            from_bucket=PAIRED_OPEN,
            event_type="expiry_basis_zero_release",
            status="paired_open_expiry_released",
        )

    def complete_entry_rollback(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        """Release entry exposure only after its rollback actually completes."""

        return self._release(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=HEDGE_PENDING,
            event_type="entry_rollback_complete",
            status="hedge_pending_rollback_released",
        )

    def complete_exit_rollback(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        """Restore an unsuccessfully exiting unit to paired-open capacity."""

        return self._transfer(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            from_bucket=EXIT_IN_PROGRESS,
            to_bucket=PAIRED_OPEN,
            event_type="exit_rollback_complete",
            status="exit_in_progress_rollback_to_paired_open",
        )

    def record_entry_rollback_failure(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        """Record failure while leaving hedge-pending capacity untouched."""

        return self._record_rollback_failure(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            bucket=HEDGE_PENDING,
            event_type="entry_rollback_failed",
            status="hedge_pending_unresolved",
        )

    def record_exit_rollback_failure(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int = 0,
        row_index: int = 0,
        capacity_id: str,
        notional_twd: int | Decimal,
    ) -> CapacityTransition:
        """Record failure while leaving exit-in-progress capacity untouched."""

        return self._record_rollback_failure(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            amount_twd=notional_twd,
            bucket=EXIT_IN_PROGRESS,
            event_type="exit_rollback_failed",
            status="exit_in_progress_unresolved",
        )

    def verify(self) -> CapacityReplayResult:
        """Replay and validate rows, caching only an unchanged successful state."""

        if (
            self._verification_cache is not None
            and self._verification_cache.key == self._verification_key()
        ):
            return _copy_capacity_replay_result(self._verification_cache.replay)

        replay = replay_capacity_transitions(
            self._transitions,
            seed=self._replay_seed,
            compact_checkpoint=self._replay_compact_checkpoint,
        )
        actual_accounts = {
            identifier: account.balances
            for identifier, account in self._accounts.items()
        }
        actual_products = {
            identifier: account.product_id
            for identifier, account in self._accounts.items()
        }
        if replay.account_balances != actual_accounts:
            raise CapacityReplayError(
                "replayed account balances differ from live state"
            )
        if replay.account_products != actual_products:
            raise CapacityReplayError(
                "replayed account products differ from live state"
            )
        if replay.product_balances != self._product_balances:
            raise CapacityReplayError(
                "replayed product balances differ from live state"
            )
        if replay.global_balances != self._global_balances:
            raise CapacityReplayError("replayed global balances differ from live state")
        replay_cursor = (
            (
                replay.last_timestamp_ns,
                replay.last_event_sequence,
                replay.last_row_index,
            )
            if replay.last_timestamp_ns is not None
            else None
        )
        if replay_cursor != self._last_cursor:
            raise CapacityReplayError("replayed last cursor differs from live state")
        if replay.last_sequence != self._sequence_offset + len(self._transitions):
            raise CapacityReplayError(
                "replayed transition sequence differs from live state"
            )
        if set(replay.seen_transition_ids) != self._transition_ids:
            raise CapacityReplayError(
                "replayed transition identities differ from live state"
            )
        if replay.transition_chain_sha256 != self._transition_chain_sha256:
            raise CapacityReplayError(
                "replayed transition chain differs from live state"
            )
        cached_replay = _copy_capacity_replay_result(replay)
        self._verification_cache = _CapacityVerificationCache(
            key=self._verification_key(),
            replay=cached_replay,
        )
        return _copy_capacity_replay_result(cached_replay)

    def _verification_key(self) -> _CapacityVerificationKey:
        return _CapacityVerificationKey(
            mutation_revision=self._mutation_revision,
            global_cap_twd=self.global_cap_twd,
            product_cap_twd=self.product_cap_twd,
            accounts=tuple(
                (capacity_id, account.product_id, account.balances)
                for capacity_id, account in sorted(self._accounts.items())
            ),
            product_balances=tuple(sorted(self._product_balances.items())),
            global_balances=self._global_balances,
            transitions=tuple(self._transitions),
            transition_ids=frozenset(self._transition_ids),
            last_cursor=self._last_cursor,
            sequence_offset=self._sequence_offset,
            transition_chain_sha256=self._transition_chain_sha256,
            replay_seed=self._replay_seed,
            replay_compact_checkpoint=self._replay_compact_checkpoint,
            resume_boundary_pending=self._resume_boundary_pending,
        )

    def _transfer(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int,
        row_index: int,
        capacity_id: str,
        amount_twd: int | Decimal,
        from_bucket: BucketName,
        to_bucket: BucketName,
        event_type: str,
        status: str,
    ) -> CapacityTransition:
        if from_bucket == to_bucket:
            raise CapacityTransitionError("transfer buckets must differ")
        amount = _integer_twd(amount_twd, "notional_twd")
        delta = _transfer_delta(from_bucket, to_bucket, amount)
        return self._mutate_existing(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            event_type=event_type,
            status=status,
            amount=amount,
            from_bucket=from_bucket,
            to_bucket=to_bucket,
            delta=delta,
        )

    def _release(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int,
        row_index: int,
        capacity_id: str,
        amount_twd: int | Decimal,
        from_bucket: BucketName,
        event_type: str,
        status: str,
    ) -> CapacityTransition:
        amount = _integer_twd(amount_twd, "notional_twd")
        return self._mutate_existing(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            capacity_id=capacity_id,
            event_type=event_type,
            status=status,
            amount=amount,
            from_bucket=from_bucket,
            to_bucket=None,
            delta=_single_bucket_delta(from_bucket, -amount),
        )

    def _record_rollback_failure(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int,
        row_index: int,
        capacity_id: str,
        amount_twd: int | Decimal,
        bucket: BucketName,
        event_type: str,
        status: str,
    ) -> CapacityTransition:
        amount = _integer_twd(amount_twd, "notional_twd")
        transition_id, cursor = self._validate_common(
            transition_id, timestamp_ns, event_sequence, row_index
        )
        timestamp_ns, event_sequence, row_index = cursor
        capacity_id = _identifier(capacity_id, "capacity_id")
        account = self._required_account(capacity_id)
        if account.balances.amount(bucket) < amount:
            raise CapacityTransitionError(
                f"insufficient {bucket} capacity for {capacity_id}"
            )
        product_before = self._product_balances[account.product_id]
        return self._append(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            event_type=event_type,
            capacity_id=capacity_id,
            product_id=account.product_id,
            status=status,
            admitted=None,
            requested_notional_twd=None,
            moved_notional_twd=amount,
            from_bucket=bucket,
            to_bucket=bucket,
            delta=ZERO_BALANCES,
            account_before=account.balances,
            account_after=account.balances,
            product_before=product_before,
            product_after=product_before,
            global_before=self._global_balances,
            global_after=self._global_balances,
        )

    def _mutate_existing(
        self,
        *,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int,
        row_index: int,
        capacity_id: str,
        event_type: str,
        status: str,
        amount: int,
        from_bucket: BucketName,
        to_bucket: BucketName | None,
        delta: BucketBalances,
    ) -> CapacityTransition:
        transition_id, cursor = self._validate_common(
            transition_id, timestamp_ns, event_sequence, row_index
        )
        timestamp_ns, event_sequence, row_index = cursor
        capacity_id = _identifier(capacity_id, "capacity_id")
        account = self._required_account(capacity_id)
        product_id = account.product_id
        account_before = account.balances
        if account_before.amount(from_bucket) < amount:
            raise CapacityTransitionError(
                f"insufficient {from_bucket} capacity for {capacity_id}"
            )
        product_before = self._product_balances[product_id]
        global_before = self._global_balances
        account_after = account_before.add(delta)
        product_after = product_before.add(delta)
        global_after = global_before.add(delta)
        if to_bucket is not None and (
            account_after.total_committed_notional_twd
            != account_before.total_committed_notional_twd
        ):
            raise CapacityTransitionError("bucket transfer changed committed total")
        account.balances = account_after
        self._product_balances[product_id] = product_after
        self._global_balances = global_after
        return self._append(
            transition_id=transition_id,
            timestamp_ns=timestamp_ns,
            event_sequence=event_sequence,
            row_index=row_index,
            event_type=event_type,
            capacity_id=capacity_id,
            product_id=product_id,
            status=status,
            admitted=None,
            requested_notional_twd=None,
            moved_notional_twd=amount,
            from_bucket=from_bucket,
            to_bucket=to_bucket,
            delta=delta,
            account_before=account_before,
            account_after=account_after,
            product_before=product_before,
            product_after=product_after,
            global_before=global_before,
            global_after=global_after,
        )

    def _validate_common(
        self,
        transition_id: str,
        timestamp_ns: int,
        event_sequence: int,
        row_index: int,
    ) -> tuple[str, tuple[int, int, int]]:
        identifier = _identifier(transition_id, "transition_id")
        cursor = _cursor(timestamp_ns, event_sequence, row_index)
        if identifier in self._transition_ids:
            raise CapacityTransitionError(f"duplicate transition_id: {identifier}")
        if (
            self._resume_boundary_pending
            and self._last_cursor is not None
            and cursor <= self._last_cursor
        ):
            raise CapacityTransitionError(
                "first transition after checkpoint must strictly advance cursor"
            )
        if self._last_cursor is not None and cursor < self._last_cursor:
            raise CapacityTransitionError(
                "transition cursors must be tuple-nondecreasing"
            )
        return identifier, cursor

    def _required_account(self, capacity_id: str) -> _Account:
        identifier = _identifier(capacity_id, "capacity_id")
        account = self._accounts.get(identifier)
        if account is None:
            raise CapacityTransitionError(f"unknown capacity_id: {identifier}")
        return account

    def _append(self, **values: object) -> CapacityTransition:
        self._verification_cache = None
        self._mutation_revision += 1
        transition = CapacityTransition(
            sequence=self._sequence_offset + len(self._transitions) + 1,
            global_cap_twd=self.global_cap_twd,
            product_cap_twd=self.product_cap_twd,
            **values,
        )
        _verify_transition_caps(transition)
        self._transitions.append(transition)
        self._transition_ids.add(transition.transition_id)
        self._transition_chain_sha256 = _extend_transition_chain(
            self._transition_chain_sha256,
            transition,
        )
        self._last_cursor = (
            transition.timestamp_ns,
            transition.event_sequence,
            transition.row_index,
        )
        self._resume_boundary_pending = False
        return transition


def _copy_capacity_replay_result(
    replay: CapacityReplayResult,
) -> CapacityReplayResult:
    """Copy mutable replay maps so callers cannot modify cached verification."""

    return CapacityReplayResult(
        account_balances=dict(replay.account_balances),
        account_products=dict(replay.account_products),
        product_balances=dict(replay.product_balances),
        global_balances=replay.global_balances,
        last_timestamp_ns=replay.last_timestamp_ns,
        last_event_sequence=replay.last_event_sequence,
        last_row_index=replay.last_row_index,
        last_sequence=replay.last_sequence,
        seen_transition_ids=replay.seen_transition_ids,
        transition_chain_sha256=replay.transition_chain_sha256,
    )


def replay_capacity_transitions(
    transitions: Iterable[CapacityTransition],
    *,
    seed: CapacityLedgerSeed | None = None,
    compact_checkpoint: CapacityLedgerCompactCheckpoint | None = None,
) -> CapacityReplayResult:
    """Rebuild state from zero, a cumulative seed, or a compact checkpoint.

    With ``compact_checkpoint``, ``seen_transition_ids`` contains only IDs from
    ``transitions``.  Historical uniqueness is intentionally delegated to the
    exact external ``UNIQUE`` registry bound by the checkpoint receipt.
    """

    if seed is not None and compact_checkpoint is not None:
        raise CapacityReplayError("seed and compact checkpoint are mutually exclusive")

    if compact_checkpoint is not None:
        products, global_balances = _validate_capacity_ledger_compact_checkpoint(
            compact_checkpoint
        )
        accounts = compact_checkpoint.account_balances
        account_products = compact_checkpoint.account_products
        transition_ids = set()
        last_cursor = compact_checkpoint.last_cursor
        expected_sequence = compact_checkpoint.transition_sequence_offset + 1
        caps = (
            compact_checkpoint.global_cap_twd,
            compact_checkpoint.product_cap_twd,
        )
        transition_chain_sha256 = compact_checkpoint.transition_chain_sha256
        resume_boundary_pending = last_cursor is not None
    elif seed is None:
        accounts: dict[str, BucketBalances] = {}
        account_products: dict[str, str] = {}
        products: dict[str, BucketBalances] = {}
        global_balances = ZERO_BALANCES
        transition_ids: set[str] = set()
        last_cursor: tuple[int, int, int] | None = None
        expected_sequence = 1
        caps: tuple[int, int] | None = None
        transition_chain_sha256 = _INITIAL_TRANSITION_CHAIN_SHA256
        resume_boundary_pending = False
    else:
        products, global_balances = _validate_capacity_ledger_seed(seed)
        accounts = seed.account_balances
        account_products = seed.account_products
        transition_ids = set(seed.seen_transition_ids)
        last_cursor = seed.last_cursor
        expected_sequence = seed.transition_sequence_offset + 1
        caps = (seed.global_cap_twd, seed.product_cap_twd)
        transition_chain_sha256 = seed.transition_chain_sha256
        resume_boundary_pending = last_cursor is not None

    for transition in transitions:
        if not isinstance(transition, CapacityTransition):
            raise CapacityReplayError("all rows must be CapacityTransition instances")
        if transition.sequence != expected_sequence:
            raise CapacityReplayError("transition sequence is not contiguous")
        expected_sequence += 1
        if transition.transition_id in transition_ids:
            raise CapacityReplayError("transition_id values must be unique")
        transition_ids.add(transition.transition_id)
        try:
            cursor = _cursor(
                transition.timestamp_ns,
                transition.event_sequence,
                transition.row_index,
            )
        except CapacityLedgerError as error:
            raise CapacityReplayError("transition cursor is invalid") from error
        if (
            resume_boundary_pending
            and last_cursor is not None
            and cursor <= last_cursor
        ):
            raise CapacityReplayError(
                "first transition after checkpoint did not strictly advance cursor"
            )
        if last_cursor is not None and cursor < last_cursor:
            raise CapacityReplayError("transition cursors are not tuple-nondecreasing")
        last_cursor = cursor
        resume_boundary_pending = False
        row_caps = (transition.global_cap_twd, transition.product_cap_twd)
        if caps is None:
            caps = row_caps
        elif caps != row_caps:
            raise CapacityReplayError("capacity limits changed within one ledger")
        _verify_transition_caps(transition)

        account_before = accounts.get(transition.capacity_id, ZERO_BALANCES)
        product_before = products.get(transition.product_id, ZERO_BALANCES)
        if account_before != transition.account_before:
            raise CapacityReplayError("account_before does not match replay state")
        if product_before != transition.product_before:
            raise CapacityReplayError("product_before does not match replay state")
        if global_balances != transition.global_before:
            raise CapacityReplayError("global_before does not match replay state")

        known_product = account_products.get(transition.capacity_id)
        if known_product is not None and known_product != transition.product_id:
            raise CapacityReplayError("capacity_id changed product")
        if transition.event_type == "new_reservation_attempt":
            if known_product is not None:
                raise CapacityReplayError("admitted capacity_id was reserved twice")
            if transition.admitted:
                account_products[transition.capacity_id] = transition.product_id
        elif known_product is None:
            raise CapacityReplayError("mutation references an unknown capacity_id")

        expected_account_after = account_before.add(transition.delta)
        expected_product_after = product_before.add(transition.delta)
        expected_global_after = global_balances.add(transition.delta)
        if expected_account_after != transition.account_after:
            raise CapacityReplayError("account_after does not match row delta")
        if expected_product_after != transition.product_after:
            raise CapacityReplayError("product_after does not match row delta")
        if expected_global_after != transition.global_after:
            raise CapacityReplayError("global_after does not match row delta")

        _verify_transition_semantics(transition)
        state_mutated = (
            transition.admitted or transition.event_type != "new_reservation_attempt"
        )
        if state_mutated:
            accounts[transition.capacity_id] = expected_account_after
            products[transition.product_id] = expected_product_after
        global_balances = expected_global_after
        transition_chain_sha256 = _extend_transition_chain(
            transition_chain_sha256,
            transition,
        )

    return CapacityReplayResult(
        account_balances=accounts,
        account_products=account_products,
        product_balances=products,
        global_balances=global_balances,
        last_timestamp_ns=last_cursor[0] if last_cursor is not None else None,
        last_event_sequence=last_cursor[1] if last_cursor is not None else None,
        last_row_index=last_cursor[2] if last_cursor is not None else None,
        last_sequence=expected_sequence - 1,
        seen_transition_ids=tuple(sorted(transition_ids)),
        transition_chain_sha256=transition_chain_sha256,
    )


def _validate_capacity_ledger_seed(
    seed: CapacityLedgerSeed,
) -> tuple[dict[str, BucketBalances], BucketBalances]:
    if not isinstance(seed, CapacityLedgerSeed):
        raise CapacityReplayError("seed must be a CapacityLedgerSeed")
    if seed.schema_version != CAPACITY_LEDGER_SEED_SCHEMA_VERSION:
        raise CapacityReplayError("unsupported capacity ledger seed schema")
    try:
        _through_date(seed.through_date)
    except CapacityLedgerError as error:
        raise CapacityReplayError("seed through_date is invalid") from error

    for label, value in (
        ("global_cap_twd", seed.global_cap_twd),
        ("product_cap_twd", seed.product_cap_twd),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CapacityReplayError(f"seed {label} must be a positive integer")
    if seed.product_cap_twd > seed.global_cap_twd:
        raise CapacityReplayError("seed product cap exceeds global cap")
    if not isinstance(seed.accounts, tuple):
        raise CapacityReplayError("seed accounts must be a tuple")

    account_ids: list[str] = []
    for account in seed.accounts:
        if not isinstance(account, CapacityAccountSeed):
            raise CapacityReplayError("seed account has an invalid type")
        try:
            capacity_id = _identifier(account.capacity_id, "capacity_id")
            _identifier(account.product_id, "product_id")
        except CapacityLedgerError as error:
            raise CapacityReplayError("seed account identity is invalid") from error
        account_ids.append(capacity_id)
        _validate_seed_balances(account.balances)
    if tuple(account_ids) != tuple(sorted(set(account_ids))):
        raise CapacityReplayError(
            "seed accounts must have unique, canonical capacity_id order"
        )

    offset = seed.transition_sequence_offset
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise CapacityReplayError("seed transition sequence offset is invalid")
    if not isinstance(seed.seen_transition_ids, tuple):
        raise CapacityReplayError("seed seen transition IDs must be a tuple")
    try:
        seen_ids = tuple(
            _identifier(identifier, "transition_id")
            for identifier in seed.seen_transition_ids
        )
    except CapacityLedgerError as error:
        raise CapacityReplayError("seed transition identity is invalid") from error
    if seen_ids != tuple(sorted(set(seen_ids))):
        raise CapacityReplayError(
            "seed transition IDs must be unique and in canonical order"
        )
    if len(seen_ids) != offset:
        raise CapacityReplayError(
            "seed transition sequence offset does not match seen IDs"
        )
    if len(seed.accounts) > offset:
        raise CapacityReplayError("seed has more admitted accounts than transitions")

    cursor_values = (
        seed.last_timestamp_ns,
        seed.last_event_sequence,
        seed.last_row_index,
    )
    if all(value is None for value in cursor_values):
        last_cursor = None
    elif any(value is None for value in cursor_values):
        raise CapacityReplayError("seed last cursor must be entirely present or null")
    else:
        try:
            last_cursor = _cursor(
                seed.last_timestamp_ns,  # type: ignore[arg-type]
                seed.last_event_sequence,  # type: ignore[arg-type]
                seed.last_row_index,  # type: ignore[arg-type]
            )
        except CapacityLedgerError as error:
            raise CapacityReplayError("seed last cursor is invalid") from error
    if (offset == 0) != (last_cursor is None):
        raise CapacityReplayError("seed sequence offset and last cursor disagree")

    _validate_sha256(seed.transition_chain_sha256, "transition chain")
    _validate_sha256(seed.checkpoint_sha256, "checkpoint")
    if offset == 0:
        if seed.transition_chain_sha256 != _INITIAL_TRANSITION_CHAIN_SHA256:
            raise CapacityReplayError("empty seed has a non-empty transition chain")
    elif seed.transition_chain_sha256 == _INITIAL_TRANSITION_CHAIN_SHA256:
        raise CapacityReplayError("non-empty seed has an empty transition chain")

    products, global_balances = _derive_totals_from_account_seeds(seed.accounts)
    if global_balances.total_committed_notional_twd > seed.global_cap_twd:
        raise CapacityReplayError("seed global hard cap exceeded")
    if any(
        balances.total_committed_notional_twd > seed.product_cap_twd
        for balances in products.values()
    ):
        raise CapacityReplayError("seed product hard cap exceeded")

    expected_checkpoint = _checkpoint_sha256(_seed_values(seed))
    if seed.checkpoint_sha256 != expected_checkpoint:
        raise CapacityReplayError("capacity ledger checkpoint digest mismatch")
    return products, global_balances


def _validate_identity_registry_receipt(
    receipt: CapacityIdentityRegistryReceipt,
) -> None:
    if not isinstance(receipt, CapacityIdentityRegistryReceipt):
        raise CapacityReplayError(
            "identity registry receipt must be CapacityIdentityRegistryReceipt"
        )
    if (
        isinstance(receipt.schema_version, bool)
        or not isinstance(receipt.schema_version, int)
        or receipt.schema_version != CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION
    ):
        raise CapacityReplayError("unsupported identity registry receipt schema")
    for label, value in (
        ("transition_count", receipt.transition_count),
        ("admitted_count", receipt.admitted_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapacityReplayError(
                f"identity registry receipt {label} must be a non-negative integer"
            )
    if receipt.admitted_count > receipt.transition_count:
        raise CapacityReplayError(
            "identity registry admitted count exceeds transition count"
        )
    if (
        not isinstance(receipt.registry_sha256, str)
        or len(receipt.registry_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in receipt.registry_sha256
        )
    ):
        raise CapacityReplayError("identity registry receipt SHA-256 is invalid")


def _validate_capacity_ledger_compact_checkpoint(
    checkpoint: CapacityLedgerCompactCheckpoint,
) -> tuple[dict[str, BucketBalances], BucketBalances]:
    if not isinstance(checkpoint, CapacityLedgerCompactCheckpoint):
        raise CapacityReplayError(
            "compact checkpoint must be a CapacityLedgerCompactCheckpoint"
        )
    if (
        isinstance(checkpoint.schema_version, bool)
        or not isinstance(checkpoint.schema_version, int)
        or checkpoint.schema_version
        != CAPACITY_LEDGER_COMPACT_CHECKPOINT_SCHEMA_VERSION
    ):
        raise CapacityReplayError("unsupported capacity compact checkpoint schema")
    try:
        _through_date(checkpoint.through_date)
    except CapacityLedgerError as error:
        raise CapacityReplayError(
            "compact checkpoint through_date is invalid"
        ) from error

    for label, value in (
        ("global_cap_twd", checkpoint.global_cap_twd),
        ("product_cap_twd", checkpoint.product_cap_twd),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CapacityReplayError(
                f"compact checkpoint {label} must be a positive integer"
            )
    if checkpoint.product_cap_twd > checkpoint.global_cap_twd:
        raise CapacityReplayError("compact checkpoint product cap exceeds global cap")
    if not isinstance(checkpoint.live_accounts, tuple):
        raise CapacityReplayError("compact checkpoint live accounts must be a tuple")

    account_ids: list[str] = []
    for account in checkpoint.live_accounts:
        if not isinstance(account, CapacityAccountSeed):
            raise CapacityReplayError(
                "compact checkpoint live account has an invalid type"
            )
        try:
            capacity_id = _identifier(account.capacity_id, "capacity_id")
            _identifier(account.product_id, "product_id")
        except CapacityLedgerError as error:
            raise CapacityReplayError(
                "compact checkpoint live account identity is invalid"
            ) from error
        account_ids.append(capacity_id)
        _validate_seed_balances(account.balances)
        if account.balances.total_committed_notional_twd == 0:
            raise CapacityReplayError(
                "compact checkpoint must omit closed zero-balance accounts"
            )
    if tuple(account_ids) != tuple(sorted(set(account_ids))):
        raise CapacityReplayError(
            "compact checkpoint live accounts must have unique, canonical order"
        )

    offset = checkpoint.transition_sequence_offset
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise CapacityReplayError(
            "compact checkpoint transition sequence offset is invalid"
        )
    _validate_identity_registry_receipt(checkpoint.identity_registry_receipt)
    receipt = checkpoint.identity_registry_receipt
    if receipt.transition_count != offset:
        raise CapacityReplayError(
            "identity registry transition count does not match sequence offset"
        )
    if len(checkpoint.live_accounts) > receipt.admitted_count:
        raise CapacityReplayError(
            "compact checkpoint has more live accounts than admitted identities"
        )

    cursor_values = (
        checkpoint.last_timestamp_ns,
        checkpoint.last_event_sequence,
        checkpoint.last_row_index,
    )
    if all(value is None for value in cursor_values):
        last_cursor = None
    elif any(value is None for value in cursor_values):
        raise CapacityReplayError(
            "compact checkpoint last cursor must be entirely present or null"
        )
    else:
        try:
            last_cursor = _cursor(
                checkpoint.last_timestamp_ns,  # type: ignore[arg-type]
                checkpoint.last_event_sequence,  # type: ignore[arg-type]
                checkpoint.last_row_index,  # type: ignore[arg-type]
            )
        except CapacityLedgerError as error:
            raise CapacityReplayError(
                "compact checkpoint last cursor is invalid"
            ) from error
    if (offset == 0) != (last_cursor is None):
        raise CapacityReplayError(
            "compact checkpoint sequence offset and last cursor disagree"
        )

    _validate_sha256(checkpoint.transition_chain_sha256, "transition chain")
    _validate_sha256(checkpoint.checkpoint_sha256, "compact checkpoint")
    if offset == 0:
        if checkpoint.transition_chain_sha256 != _INITIAL_TRANSITION_CHAIN_SHA256:
            raise CapacityReplayError(
                "empty compact checkpoint has a non-empty transition chain"
            )
    elif checkpoint.transition_chain_sha256 == _INITIAL_TRANSITION_CHAIN_SHA256:
        raise CapacityReplayError(
            "non-empty compact checkpoint has an empty transition chain"
        )

    products, global_balances = _derive_totals_from_account_seeds(
        checkpoint.live_accounts
    )
    if global_balances.total_committed_notional_twd > checkpoint.global_cap_twd:
        raise CapacityReplayError("compact checkpoint global hard cap exceeded")
    if any(
        balances.total_committed_notional_twd > checkpoint.product_cap_twd
        for balances in products.values()
    ):
        raise CapacityReplayError("compact checkpoint product hard cap exceeded")

    expected_checkpoint = _compact_checkpoint_sha256(
        _compact_checkpoint_values(checkpoint)
    )
    if checkpoint.checkpoint_sha256 != expected_checkpoint:
        raise CapacityReplayError("capacity compact checkpoint digest mismatch")
    return products, global_balances


def _validate_seed_balances(balances: BucketBalances) -> None:
    if not isinstance(balances, BucketBalances):
        raise CapacityReplayError("seed balances must be BucketBalances")
    values = tuple(getattr(balances, name) for name in BUCKET_NAMES)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise CapacityReplayError("seed balance buckets must be integer TWD")
    if any(value < 0 for value in values):
        raise CapacityReplayError("seed balance bucket cannot be negative")


def _derive_totals_from_account_seeds(
    accounts: tuple[CapacityAccountSeed, ...],
) -> tuple[dict[str, BucketBalances], BucketBalances]:
    products: dict[str, BucketBalances] = {}
    global_balances = ZERO_BALANCES
    for account in accounts:
        products[account.product_id] = products.get(
            account.product_id, ZERO_BALANCES
        ).add(account.balances)
        global_balances = global_balances.add(account.balances)
    return products, global_balances


def _seed_values(seed: CapacityLedgerSeed) -> dict[str, object]:
    return {
        "schema_version": seed.schema_version,
        "through_date": seed.through_date,
        "global_cap_twd": seed.global_cap_twd,
        "product_cap_twd": seed.product_cap_twd,
        "accounts": seed.accounts,
        "transition_sequence_offset": seed.transition_sequence_offset,
        "last_timestamp_ns": seed.last_timestamp_ns,
        "last_event_sequence": seed.last_event_sequence,
        "last_row_index": seed.last_row_index,
        "seen_transition_ids": seed.seen_transition_ids,
        "transition_chain_sha256": seed.transition_chain_sha256,
    }


def _compact_checkpoint_values(
    checkpoint: CapacityLedgerCompactCheckpoint,
) -> dict[str, object]:
    return {
        "schema_version": checkpoint.schema_version,
        "through_date": checkpoint.through_date,
        "global_cap_twd": checkpoint.global_cap_twd,
        "product_cap_twd": checkpoint.product_cap_twd,
        "live_accounts": checkpoint.live_accounts,
        "transition_sequence_offset": checkpoint.transition_sequence_offset,
        "last_timestamp_ns": checkpoint.last_timestamp_ns,
        "last_event_sequence": checkpoint.last_event_sequence,
        "last_row_index": checkpoint.last_row_index,
        "transition_chain_sha256": checkpoint.transition_chain_sha256,
        "identity_registry_receipt": checkpoint.identity_registry_receipt,
    }


def _checkpoint_sha256(values: dict[str, object]) -> str:
    accounts = values["accounts"]
    assert isinstance(accounts, tuple)
    payload = {
        "domain": _CHECKPOINT_DOMAIN,
        "schema_version": values["schema_version"],
        "through_date": values["through_date"],
        "global_cap_twd": values["global_cap_twd"],
        "product_cap_twd": values["product_cap_twd"],
        "accounts": [
            {
                "capacity_id": account.capacity_id,
                "product_id": account.product_id,
                "balances": {
                    name: getattr(account.balances, name) for name in BUCKET_NAMES
                },
            }
            for account in accounts
        ],
        "transition_sequence_offset": values["transition_sequence_offset"],
        "last_cursor": [
            values["last_timestamp_ns"],
            values["last_event_sequence"],
            values["last_row_index"],
        ],
        "seen_transition_ids": list(values["seen_transition_ids"]),
        "transition_chain_sha256": values["transition_chain_sha256"],
    }
    return _canonical_sha256(payload)


def _compact_checkpoint_sha256(values: dict[str, object]) -> str:
    accounts = values["live_accounts"]
    receipt = values["identity_registry_receipt"]
    assert isinstance(accounts, tuple)
    assert isinstance(receipt, CapacityIdentityRegistryReceipt)
    payload = {
        "domain": _COMPACT_CHECKPOINT_DOMAIN,
        "schema_version": values["schema_version"],
        "through_date": values["through_date"],
        "global_cap_twd": values["global_cap_twd"],
        "product_cap_twd": values["product_cap_twd"],
        "live_accounts": [
            {
                "capacity_id": account.capacity_id,
                "product_id": account.product_id,
                "balances": {
                    name: getattr(account.balances, name) for name in BUCKET_NAMES
                },
            }
            for account in accounts
        ],
        "transition_sequence_offset": values["transition_sequence_offset"],
        "last_cursor": [
            values["last_timestamp_ns"],
            values["last_event_sequence"],
            values["last_row_index"],
        ],
        "transition_chain_sha256": values["transition_chain_sha256"],
        "identity_registry_receipt": {
            "schema_version": receipt.schema_version,
            "transition_count": receipt.transition_count,
            "admitted_count": receipt.admitted_count,
            "registry_sha256": receipt.registry_sha256,
        },
    }
    return _canonical_sha256(payload)


def _extend_transition_chain(
    previous_sha256: str,
    transition: CapacityTransition,
) -> str:
    return _canonical_sha256(
        {
            "domain": _TRANSITION_CHAIN_DOMAIN,
            "previous_sha256": previous_sha256,
            "transition": transition.as_dict(),
        }
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapacityReplayError(f"seed {label} digest is invalid")


def _verify_transition_semantics(transition: CapacityTransition) -> None:
    delta_total = transition.delta.total_committed_notional_twd
    amount = transition.moved_notional_twd
    if transition.event_type == "new_reservation_attempt":
        requested = transition.requested_notional_twd
        if requested is None or requested <= 0:
            raise CapacityReplayError("reservation row lacks positive request")
        if (
            transition.from_bucket is not None
            or transition.to_bucket != WORKING_UNFILLED
        ):
            raise CapacityReplayError("reservation bucket identity is invalid")
        global_blocked = (
            transition.global_before.total_committed_notional_twd + requested
            > transition.global_cap_twd
        )
        product_blocked = (
            transition.product_before.total_committed_notional_twd + requested
            > transition.product_cap_twd
        )
        expected_admitted = not global_blocked and not product_blocked
        if transition.admitted is not expected_admitted:
            raise CapacityReplayError("reservation decision does not match hard caps")
        if transition.admitted:
            expected = _single_bucket_delta(WORKING_UNFILLED, requested)
            if transition.status != "admitted" or transition.delta != expected:
                raise CapacityReplayError("admitted reservation delta is invalid")
            if amount != requested:
                raise CapacityReplayError("admitted reservation moved amount differs")
        else:
            expected_status = (
                "blocked_both_caps"
                if global_blocked and product_blocked
                else "blocked_global_cap"
                if global_blocked
                else "blocked_product_cap"
            )
            if transition.status != expected_status:
                raise CapacityReplayError("blocked reservation reason is invalid")
            if transition.delta != ZERO_BALANCES or amount != 0:
                raise CapacityReplayError("blocked reservation mutated capacity")
        return

    if transition.admitted is not None or transition.requested_notional_twd is not None:
        raise CapacityReplayError("non-reservation row has reservation fields")
    if amount <= 0:
        raise CapacityReplayError("mutation moved amount must be positive")
    if transition.from_bucket is None:
        raise CapacityReplayError("mutation lacks from_bucket")
    expected_identity = {
        "entry_partial_fill": (
            WORKING_UNFILLED,
            ENTRY_PARTIAL,
            "working_to_entry_partial",
        ),
        "entry_full_fill": (
            WORKING_UNFILLED,
            HEDGE_PENDING,
            "working_to_hedge_pending",
        ),
        "entry_partial_hedge_trigger": (
            ENTRY_PARTIAL,
            HEDGE_PENDING,
            "entry_partial_to_hedge_pending",
        ),
        "entry_hedge_complete": (
            HEDGE_PENDING,
            PAIRED_OPEN,
            "hedge_pending_to_paired_open",
        ),
        "exit_started": (
            PAIRED_OPEN,
            EXIT_IN_PROGRESS,
            "paired_open_to_exit_in_progress",
        ),
        "exit_hedge_complete": (
            EXIT_IN_PROGRESS,
            None,
            "exit_in_progress_released",
        ),
        "expiry_basis_zero_release": (
            PAIRED_OPEN,
            None,
            "paired_open_expiry_released",
        ),
        "entry_rollback_complete": (
            HEDGE_PENDING,
            None,
            "hedge_pending_rollback_released",
        ),
        "exit_rollback_complete": (
            EXIT_IN_PROGRESS,
            PAIRED_OPEN,
            "exit_in_progress_rollback_to_paired_open",
        ),
    }.get(transition.event_type)
    failure_identity = {
        "entry_rollback_failed": (
            HEDGE_PENDING,
            "hedge_pending_unresolved",
        ),
        "exit_rollback_failed": (
            EXIT_IN_PROGRESS,
            "exit_in_progress_unresolved",
        ),
    }.get(transition.event_type)
    if failure_identity is not None:
        bucket, status = failure_identity
        if (
            transition.from_bucket != bucket
            or transition.to_bucket != bucket
            or transition.status != status
            or transition.delta != ZERO_BALANCES
            or transition.account_before != transition.account_after
            or transition.product_before != transition.product_after
            or transition.global_before != transition.global_after
            or transition.account_before.amount(bucket) < amount
        ):
            raise CapacityReplayError("rollback failure changed committed capacity")
        return
    if transition.event_type == "working_leaves_release":
        if (
            transition.from_bucket != WORKING_UNFILLED
            or transition.to_bucket is not None
            or transition.status not in {"actual_cancel", "session_expiry"}
        ):
            raise CapacityReplayError("working leaves release identity is invalid")
    elif expected_identity is None:
        raise CapacityReplayError("unknown capacity event_type")
    elif (
        transition.from_bucket,
        transition.to_bucket,
        transition.status,
    ) != expected_identity:
        raise CapacityReplayError("capacity event bucket identity is invalid")
    if transition.to_bucket is None:
        expected = _single_bucket_delta(transition.from_bucket, -amount)
        if delta_total != -amount:
            raise CapacityReplayError("release did not reduce committed total")
    else:
        expected = _transfer_delta(transition.from_bucket, transition.to_bucket, amount)
        if delta_total != 0:
            raise CapacityReplayError("transfer changed committed total")
    if transition.delta != expected:
        raise CapacityReplayError("mutation delta does not match bucket movement")


def _verify_transition_caps(transition: CapacityTransition) -> None:
    for name in (
        "sequence",
        "timestamp_ns",
        "event_sequence",
        "row_index",
        "moved_notional_twd",
        "global_cap_twd",
        "product_cap_twd",
    ):
        value = getattr(transition, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise CapacityReplayError(f"{name} must be an integer")
    if transition.requested_notional_twd is not None and (
        isinstance(transition.requested_notional_twd, bool)
        or not isinstance(transition.requested_notional_twd, int)
    ):
        raise CapacityReplayError("requested_notional_twd must be integer or null")
    if transition.global_cap_twd <= 0 or transition.product_cap_twd <= 0:
        raise CapacityReplayError("hard caps must be positive")
    if transition.product_cap_twd > transition.global_cap_twd:
        raise CapacityReplayError("product cap exceeds global cap")
    for name, balances in (
        ("delta", transition.delta),
        ("account_before", transition.account_before),
        ("account_after", transition.account_after),
        ("product_before", transition.product_before),
        ("product_after", transition.product_after),
        ("global_before", transition.global_before),
        ("global_after", transition.global_after),
    ):
        if not isinstance(balances, BucketBalances):
            raise CapacityReplayError(f"{name} must be BucketBalances")
        values = tuple(
            getattr(balances, field.name) for field in fields(BucketBalances)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise CapacityReplayError(f"{name} buckets must be integer TWD")
        if name != "delta" and any(value < 0 for value in values):
            raise CapacityReplayError(f"{name} contains a negative bucket")
    if transition.global_after.total_committed_notional_twd > transition.global_cap_twd:
        raise CapacityReplayError("global hard cap exceeded")
    if (
        transition.product_after.total_committed_notional_twd
        > transition.product_cap_twd
    ):
        raise CapacityReplayError("product hard cap exceeded")


def _validate_transition_for_codec(transition: CapacityTransition) -> None:
    if type(transition.sequence) is not int or transition.sequence <= 0:
        raise CapacityCodecError("capacity transition sequence must be positive")
    try:
        _cursor(
            transition.timestamp_ns,
            transition.event_sequence,
            transition.row_index,
        )
        _identifier(transition.transition_id, "transition_id")
        _identifier(transition.event_type, "event_type")
        _identifier(transition.capacity_id, "capacity_id")
        _identifier(transition.product_id, "product_id")
        _identifier(transition.status, "status")
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error
    if transition.admitted is not None and type(transition.admitted) is not bool:
        raise CapacityCodecError("capacity transition admitted must be boolean or null")
    if (
        transition.requested_notional_twd is not None
        and type(transition.requested_notional_twd) is not int
    ):
        raise CapacityCodecError(
            "capacity transition requested_notional_twd must be integer or null"
        )
    for path, bucket in (
        ("from_bucket", transition.from_bucket),
        ("to_bucket", transition.to_bucket),
    ):
        if bucket is not None and (
            type(bucket) is not str or bucket not in BUCKET_NAMES
        ):
            raise CapacityCodecError(
                f"capacity transition {path} must be a capacity bucket or null"
            )
    try:
        _verify_transition_caps(transition)
        _verify_transition_semantics(transition)
        for path, before, after in (
            (
                "account",
                transition.account_before,
                transition.account_after,
            ),
            (
                "product",
                transition.product_before,
                transition.product_after,
            ),
            (
                "global",
                transition.global_before,
                transition.global_after,
            ),
        ):
            if before.add(transition.delta) != after:
                raise CapacityCodecError(
                    f"capacity transition {path}_after disagrees with delta"
                )
    except CapacityCodecError:
        raise
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error


def _encode_bucket_balances(balances: BucketBalances) -> dict[str, int]:
    if not isinstance(balances, BucketBalances):
        raise CapacityCodecError("capacity balances must be BucketBalances")
    result: dict[str, int] = {}
    for name in BUCKET_NAMES:
        value = getattr(balances, name)
        if type(value) is not int:
            raise CapacityCodecError("capacity balance buckets must be integers")
        result[name] = value
    return result


def _decode_bucket_balances(value: object, path: str) -> BucketBalances:
    if not isinstance(value, Mapping):
        raise CapacityCodecError(f"{path} must be a JSON object")
    _require_exact_record_keys(value, set(BUCKET_NAMES), path)
    values = {
        name: _decode_record_integer(value[name], f"{path}.{name}")
        for name in BUCKET_NAMES
    }
    return BucketBalances(**values)


def _require_exact_record_keys(
    record: Mapping[object, object],
    expected_keys: set[str],
    path: str,
) -> None:
    actual_keys = set(record)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys.difference(actual_keys))
    unknown = sorted(repr(key) for key in actual_keys.difference(expected_keys))
    raise CapacityCodecError(
        f"{path} schema mismatch: missing={missing}, unknown={unknown}"
    )


def _require_record_tag(
    record: Mapping[str, object],
    *,
    expected_type: str,
    expected_schema: int,
    path: str,
) -> None:
    record_type = record["record_type"]
    if type(record_type) is not str or record_type != expected_type:
        raise CapacityCodecError(f"{path} record_type is invalid")
    schema_version = record["schema_version"]
    if type(schema_version) is not int or schema_version != expected_schema:
        raise CapacityCodecError(f"{path} schema_version is unsupported")


def _decode_record_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise CapacityCodecError(f"{path} must be an integer")
    return value


def _decode_optional_record_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _decode_record_integer(value, path)


def _decode_record_string(value: object, path: str) -> str:
    if type(value) is not str:
        raise CapacityCodecError(f"{path} must be a string")
    return value


def _decode_optional_bucket(value: object, path: str) -> BucketName | None:
    if value is None:
        return None
    if type(value) is not str or value not in BUCKET_NAMES:
        raise CapacityCodecError(f"{path} must be a capacity bucket or null")
    return value  # type: ignore[return-value]


def _decode_checkpoint_cursor(
    value: object,
) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, Mapping):
        raise CapacityCodecError(
            "capacity compact checkpoint last_cursor must be a JSON object or null"
        )
    _require_exact_record_keys(
        value,
        {"timestamp_ns", "event_sequence", "row_index"},
        "capacity compact checkpoint last_cursor",
    )
    cursor = (
        _decode_record_integer(
            value["timestamp_ns"],
            "capacity compact checkpoint last_cursor.timestamp_ns",
        ),
        _decode_record_integer(
            value["event_sequence"],
            "capacity compact checkpoint last_cursor.event_sequence",
        ),
        _decode_record_integer(
            value["row_index"],
            "capacity compact checkpoint last_cursor.row_index",
        ),
    )
    try:
        return _cursor(*cursor)
    except CapacityLedgerError as error:
        raise CapacityCodecError(str(error)) from error


def _single_bucket_delta(bucket: BucketName, amount: int) -> BucketBalances:
    _validate_bucket(bucket)
    return BucketBalances(**{bucket: amount})


def _transfer_delta(
    from_bucket: BucketName, to_bucket: BucketName, amount: int
) -> BucketBalances:
    _validate_bucket(from_bucket)
    _validate_bucket(to_bucket)
    if from_bucket == to_bucket:
        raise CapacityTransitionError("transfer buckets must differ")
    return BucketBalances(**{from_bucket: -amount, to_bucket: amount})


def _validate_bucket(bucket: str) -> None:
    if bucket not in BUCKET_NAMES:
        raise CapacityTransitionError(f"unknown capacity bucket: {bucket}")


def _integer_twd(value: int | Decimal, label: str) -> int:
    if isinstance(value, (bool, float)):
        raise CapacityLedgerError(f"{label} must be integer TWD, not float")
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise CapacityLedgerError(f"{label} must be exact integer TWD")
        result = int(value)
    elif isinstance(value, int):
        result = value
    else:
        raise CapacityLedgerError(f"{label} must be int or Decimal")
    if result <= 0:
        raise CapacityLedgerError(f"{label} must be positive")
    return result


def _cursor(
    timestamp_ns: int, event_sequence: int, row_index: int
) -> tuple[int, int, int]:
    values = (timestamp_ns, event_sequence, row_index)
    labels = ("timestamp_ns", "event_sequence", "row_index")
    for value, label in zip(values, labels, strict=True):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapacityLedgerError(f"{label} must be a non-negative integer")
    return values


def _through_date(value: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isascii():
        raise CapacityLedgerError("through_date must be YYYYMMDD")
    try:
        parsed = date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise CapacityLedgerError(
            "through_date must be a valid YYYYMMDD date"
        ) from error
    if parsed.strftime("%Y%m%d") != value:
        raise CapacityLedgerError("through_date must be canonical YYYYMMDD")
    return value


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapacityLedgerError(f"{label} must be a non-empty string")
    return value
