"""Exact SQLite identity registry for partitioned S1 capacity replay.

The compact capacity checkpoint deliberately omits historical identities.  This
module is the exact external ``UNIQUE`` registry required by that checkpoint:

* transition IDs are unique by ``(policy_id, transition_id)``;
* admitted capacity IDs are unique by ``(policy_id, capacity_id)``;
* one receipt is committed by ``(policy_id, partition_date)``.

``commit_partition`` uses one ``BEGIN IMMEDIATE`` transaction, so the receipt,
both identity sets, and the latest-policy state either all commit or all roll
back.  The class owns one SQLite connection and is intentionally a single-writer
component; callers must not share one instance concurrently across threads.

Only admitted ``new_reservation_attempt`` rows register a capacity ID.  Blocked
attempts remain transition identities but do not consume the capacity-ID
namespace, so a later retry may still be admitted.

Each partition root is the canonical JSON SHA-256 of the previous registry
root, policy/date, global sequence range, ordered transition-ID digest, and
ordered admitted-ID digest.  The two ordered digests and the exact retry-input
fingerprint use domain-separated streaming hash chains, avoiding a second
full-size JSON copy of a large partition.

These hashes make accidental or unilateral row drift detectable; they are not
digital signatures and do not authenticate a database plus receipts that an
attacker can rewrite together.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Self

from .capacity_ledger import (
    CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
    CapacityIdentityRegistryReceipt,
    CapacityTransition,
)

S1_CAPACITY_IDENTITY_REGISTRY_SCHEMA_VERSION = 2

_REGISTRY_CHAIN_DOMAIN = "s1-capacity-identity-registry-chain-v1"
_TRANSITION_IDS_DOMAIN = "s1-capacity-transition-ids-v1"
_ADMITTED_IDS_DOMAIN = "s1-capacity-admitted-ids-v1"
_PARTITION_INPUT_DOMAIN = "s1-capacity-registry-partition-input-v1"
_PARTITION_RECORD_DOMAIN = "s1-capacity-registry-partition-record-v1"
_GENESIS_REGISTRY_SHA256 = hashlib.sha256(
    _REGISTRY_CHAIN_DOMAIN.encode("utf-8")
).hexdigest()


class CapacityIdentityRegistryError(ValueError):
    """Base class for invalid registry input or state."""


class CapacityIdentityRegistrySchemaError(CapacityIdentityRegistryError):
    """Raised when the SQLite schema or immutable metadata has drifted."""


class CapacityIdentityRegistryIntegrityError(CapacityIdentityRegistryError):
    """Raised when persisted rows fail deterministic reconstruction."""


class CapacityIdentityRegistryConflictError(CapacityIdentityRegistryError):
    """Raised when partition lineage or idempotent content does not match."""


class CapacityIdentityRegistryDuplicateError(CapacityIdentityRegistryError):
    """Raised before commit when an exact identity is already registered."""


@dataclass(frozen=True)
class _IncomingPartition:
    policy_id: str
    partition_date: str
    transition_identities: tuple[tuple[int, str], ...]
    admitted_identities: tuple[tuple[int, str, str], ...]
    ordered_transition_ids_sha256: str
    admitted_ids_sha256: str
    partition_input_sha256: str


@dataclass(frozen=True)
class CapacityIdentityRegistryTip:
    """Fully verified latest partition identity and its immediate predecessor."""

    policy_id: str
    partition_date: str
    previous_partition_date: str | None
    receipt: CapacityIdentityRegistryReceipt
    previous_receipt: CapacityIdentityRegistryReceipt | None


_TABLE_DDLS: dict[str, str] = {
    "registry_meta": """
        CREATE TABLE registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT
    """,
    "policy_state": """
        CREATE TABLE policy_state (
            policy_id TEXT PRIMARY KEY,
            latest_partition_date TEXT NOT NULL,
            transition_count INTEGER NOT NULL CHECK (transition_count >= 0),
            admitted_count INTEGER NOT NULL CHECK (admitted_count >= 0),
            registry_sha256 TEXT NOT NULL
        ) STRICT
    """,
    "partition_receipts": """
        CREATE TABLE partition_receipts (
            policy_id TEXT NOT NULL,
            partition_date TEXT NOT NULL,
            previous_partition_date TEXT,
            sequence_start INTEGER NOT NULL CHECK (sequence_start >= 1),
            sequence_end INTEGER NOT NULL CHECK (sequence_end >= 0),
            partition_transition_count INTEGER NOT NULL
                CHECK (partition_transition_count >= 0),
            partition_admitted_count INTEGER NOT NULL
                CHECK (partition_admitted_count >= 0),
            previous_transition_count INTEGER NOT NULL
                CHECK (previous_transition_count >= 0),
            previous_admitted_count INTEGER NOT NULL
                CHECK (previous_admitted_count >= 0),
            cumulative_transition_count INTEGER NOT NULL
                CHECK (cumulative_transition_count >= 0),
            cumulative_admitted_count INTEGER NOT NULL
                CHECK (cumulative_admitted_count >= 0),
            previous_registry_sha256 TEXT NOT NULL,
            ordered_transition_ids_sha256 TEXT NOT NULL,
            admitted_ids_sha256 TEXT NOT NULL,
            partition_input_sha256 TEXT NOT NULL,
            registry_sha256 TEXT NOT NULL,
            partition_record_sha256 TEXT NOT NULL,
            PRIMARY KEY (policy_id, partition_date)
        ) STRICT
    """,
    "transition_identities": """
        CREATE TABLE transition_identities (
            policy_id TEXT NOT NULL,
            transition_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK (sequence >= 1),
            PRIMARY KEY (policy_id, transition_id),
            UNIQUE (policy_id, sequence)
        ) WITHOUT ROWID, STRICT
    """,
    "admitted_capacity_identities": """
        CREATE TABLE admitted_capacity_identities (
            policy_id TEXT NOT NULL,
            capacity_id TEXT NOT NULL,
            partition_date TEXT NOT NULL,
            transition_id TEXT NOT NULL,
            transition_sequence INTEGER NOT NULL CHECK (transition_sequence >= 1),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (policy_id, capacity_id),
            UNIQUE (policy_id, transition_id),
            UNIQUE (policy_id, partition_date, ordinal),
            FOREIGN KEY (policy_id, partition_date)
                REFERENCES partition_receipts (policy_id, partition_date),
            FOREIGN KEY (policy_id, transition_id)
                REFERENCES transition_identities (policy_id, transition_id)
        ) STRICT
    """,
}

_EXPECTED_META = {
    "schema_version": str(S1_CAPACITY_IDENTITY_REGISTRY_SCHEMA_VERSION),
    "receipt_schema_version": str(CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION),
    "registry_chain_domain": _REGISTRY_CHAIN_DOMAIN,
    "transition_ids_domain": _TRANSITION_IDS_DOMAIN,
    "admitted_ids_domain": _ADMITTED_IDS_DOMAIN,
    "partition_input_domain": _PARTITION_INPUT_DOMAIN,
    "partition_record_domain": _PARTITION_RECORD_DOMAIN,
    "genesis_registry_sha256": _GENESIS_REGISTRY_SHA256,
}


class S1CapacityIdentityRegistry:
    """Single-writer exact identity registry backed by SQLite."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        timeout_seconds: float = 30.0,
        readonly: bool = False,
    ):
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise CapacityIdentityRegistryError("timeout_seconds must be positive")
        if not isinstance(readonly, bool):
            raise CapacityIdentityRegistryError("readonly must be boolean")
        path = Path(database_path)
        self.database_path = str(path)
        self.readonly = readonly
        if readonly and not path.is_file():
            raise CapacityIdentityRegistryError(
                "readonly registry database does not exist"
            )
        try:
            self._connection = sqlite3.connect(
                (
                    f"{path.resolve().as_uri()}?mode=ro"
                    if readonly
                    else self.database_path
                ),
                isolation_level=None,
                timeout=float(timeout_seconds),
                uri=readonly,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            if readonly:
                with self._transaction("DEFERRED") as connection:
                    self._validate_schema_and_meta(connection)
            else:
                self._connection.execute("PRAGMA synchronous = FULL")
                self._initialize_or_validate()
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def commit_partition(
        self,
        policy_id: str,
        partition_date: str,
        transitions: Iterable[CapacityTransition],
        expected_previous_receipt: CapacityIdentityRegistryReceipt | None,
        *,
        verify_full_history_before_commit: bool = True,
    ) -> CapacityIdentityRegistryReceipt:
        """Atomically register one strictly ordered policy/date partition.

        A repeated commit of the same ``(policy_id, partition_date)`` and exact
        transition content is idempotent and returns the original receipt.  A
        drifted retry fails.  ``expected_previous_receipt`` is the receipt of
        the partition immediately before this one, or ``None`` for the first
        partition, including idempotent recovery calls.  Before any append, the
        existing policy chain is fully recomputed inside the same transaction by
        default.  Callers processing many partitions may explicitly set
        ``verify_full_history_before_commit=False`` to validate only the cached
        tail plus the exact local delta, then run :meth:`verify` before declaring
        the registry complete.
        """

        if self.readonly:
            raise CapacityIdentityRegistryError(
                "cannot commit through a readonly identity registry"
            )
        if not isinstance(verify_full_history_before_commit, bool):
            raise CapacityIdentityRegistryError(
                "verify_full_history_before_commit must be boolean"
            )
        incoming = _incoming_partition(policy_id, partition_date, transitions)
        _validate_optional_receipt(expected_previous_receipt)
        with self._transaction("IMMEDIATE") as connection:
            self._validate_schema_and_meta(connection)
            verified_latest_receipt = (
                self._verify_policy_chain(connection, incoming.policy_id)
                if verify_full_history_before_commit
                else None
            )
            existing = connection.execute(
                """
                SELECT *
                FROM partition_receipts
                WHERE policy_id = ? AND partition_date = ?
                """,
                (incoming.policy_id, incoming.partition_date),
            ).fetchone()
            if existing is not None:
                return self._idempotent_existing_receipt(
                    connection,
                    incoming,
                    existing,
                    expected_previous_receipt,
                )

            previous_date, previous_receipt = self._latest_state_for_commit(
                connection,
                incoming.policy_id,
            )
            if (
                verify_full_history_before_commit
                and previous_receipt != verified_latest_receipt
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "cached policy state differs from fully verified registry"
                )
            _require_expected_previous_receipt(
                expected_previous_receipt,
                previous_receipt,
            )
            if previous_date is not None and incoming.partition_date <= previous_date:
                raise CapacityIdentityRegistryConflictError(
                    "partition_date must strictly advance for a new partition"
                )

            previous_transition_count = (
                previous_receipt.transition_count if previous_receipt is not None else 0
            )
            previous_admitted_count = (
                previous_receipt.admitted_count if previous_receipt is not None else 0
            )
            previous_registry_sha256 = (
                previous_receipt.registry_sha256
                if previous_receipt is not None
                else _GENESIS_REGISTRY_SHA256
            )
            _validate_contiguous_sequences(
                incoming.transition_identities,
                previous_transition_count,
            )
            self._precheck_historical_duplicates(connection, incoming)

            partition_transition_count = len(incoming.transition_identities)
            partition_admitted_count = len(incoming.admitted_identities)
            cumulative_transition_count = (
                previous_transition_count + partition_transition_count
            )
            cumulative_admitted_count = (
                previous_admitted_count + partition_admitted_count
            )
            sequence_start = previous_transition_count + 1
            sequence_end = cumulative_transition_count
            registry_sha256 = _extend_registry_chain(
                previous_registry_sha256=previous_registry_sha256,
                policy_id=incoming.policy_id,
                partition_date=incoming.partition_date,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                ordered_transition_ids_sha256=(incoming.ordered_transition_ids_sha256),
                admitted_ids_sha256=incoming.admitted_ids_sha256,
            )
            partition_record_sha256 = _partition_record_sha256(
                policy_id=incoming.policy_id,
                partition_date=incoming.partition_date,
                previous_partition_date=previous_date,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                partition_transition_count=partition_transition_count,
                partition_admitted_count=partition_admitted_count,
                previous_transition_count=previous_transition_count,
                previous_admitted_count=previous_admitted_count,
                cumulative_transition_count=cumulative_transition_count,
                cumulative_admitted_count=cumulative_admitted_count,
                previous_registry_sha256=previous_registry_sha256,
                ordered_transition_ids_sha256=(incoming.ordered_transition_ids_sha256),
                admitted_ids_sha256=incoming.admitted_ids_sha256,
                partition_input_sha256=incoming.partition_input_sha256,
                registry_sha256=registry_sha256,
            )

            try:
                connection.execute(
                    """
                    INSERT INTO partition_receipts (
                        policy_id,
                        partition_date,
                        previous_partition_date,
                        sequence_start,
                        sequence_end,
                        partition_transition_count,
                        partition_admitted_count,
                        previous_transition_count,
                        previous_admitted_count,
                        cumulative_transition_count,
                        cumulative_admitted_count,
                        previous_registry_sha256,
                        ordered_transition_ids_sha256,
                        admitted_ids_sha256,
                        partition_input_sha256,
                        registry_sha256,
                        partition_record_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        incoming.policy_id,
                        incoming.partition_date,
                        previous_date,
                        sequence_start,
                        sequence_end,
                        partition_transition_count,
                        partition_admitted_count,
                        previous_transition_count,
                        previous_admitted_count,
                        cumulative_transition_count,
                        cumulative_admitted_count,
                        previous_registry_sha256,
                        incoming.ordered_transition_ids_sha256,
                        incoming.admitted_ids_sha256,
                        incoming.partition_input_sha256,
                        registry_sha256,
                        partition_record_sha256,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO transition_identities (
                        policy_id,
                        transition_id,
                        sequence
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        (
                            incoming.policy_id,
                            transition_id,
                            sequence,
                        )
                        for sequence, transition_id in incoming.transition_identities
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO admitted_capacity_identities (
                        policy_id,
                        capacity_id,
                        partition_date,
                        transition_id,
                        transition_sequence,
                        ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            incoming.policy_id,
                            capacity_id,
                            incoming.partition_date,
                            transition_id,
                            sequence,
                            ordinal,
                        )
                        for ordinal, (
                            sequence,
                            transition_id,
                            capacity_id,
                        ) in enumerate(incoming.admitted_identities)
                    ),
                )
                if previous_receipt is None:
                    connection.execute(
                        """
                        INSERT INTO policy_state (
                            policy_id,
                            latest_partition_date,
                            transition_count,
                            admitted_count,
                            registry_sha256
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            incoming.policy_id,
                            incoming.partition_date,
                            cumulative_transition_count,
                            cumulative_admitted_count,
                            registry_sha256,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE policy_state
                        SET latest_partition_date = ?,
                            transition_count = ?,
                            admitted_count = ?,
                            registry_sha256 = ?
                        WHERE policy_id = ?
                          AND latest_partition_date = ?
                          AND transition_count = ?
                          AND admitted_count = ?
                          AND registry_sha256 = ?
                        """,
                        (
                            incoming.partition_date,
                            cumulative_transition_count,
                            cumulative_admitted_count,
                            registry_sha256,
                            incoming.policy_id,
                            previous_date,
                            previous_receipt.transition_count,
                            previous_receipt.admitted_count,
                            previous_receipt.registry_sha256,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise CapacityIdentityRegistryIntegrityError(
                            "latest policy state changed during partition commit"
                        )
            except sqlite3.IntegrityError as error:
                raise CapacityIdentityRegistryDuplicateError(
                    "SQLite uniqueness rejected partition identities"
                ) from error

            return CapacityIdentityRegistryReceipt(
                schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
                transition_count=cumulative_transition_count,
                admitted_count=cumulative_admitted_count,
                registry_sha256=registry_sha256,
            )

    def latest(self, policy_id: str) -> CapacityIdentityRegistryReceipt | None:
        """Return the latest receipt after full chain/count reconstruction."""

        result = self.verify(policy_id)
        assert not isinstance(result, dict)
        return result

    def latest_receipt(
        self,
        policy_id: str,
    ) -> CapacityIdentityRegistryReceipt | None:
        """Alias for :meth:`latest`."""

        return self.latest(policy_id)

    def latest_tip(self, policy_id: str) -> CapacityIdentityRegistryTip | None:
        """Return the fully verified latest partition and predecessor receipts.

        Unlike :meth:`latest`, this preserves partition dates.  That distinction
        lets crash recovery prove that a registry-only partition is exactly the
        canonical next partition even when its local transition delta is empty.
        """

        policy_id = _identifier(policy_id, "policy_id")
        with self._transaction("DEFERRED") as connection:
            self._validate_schema_and_meta(connection)
            receipt = self._verify_policy_chain(connection, policy_id)
            if receipt is None:
                return None
            row = connection.execute(
                """
                SELECT partition_date, previous_partition_date,
                       previous_transition_count, previous_admitted_count,
                       previous_registry_sha256
                FROM partition_receipts
                WHERE policy_id = ?
                ORDER BY partition_date DESC
                LIMIT 1
                """,
                (policy_id,),
            ).fetchone()
            if row is None:
                raise CapacityIdentityRegistryIntegrityError(
                    "verified policy lacks a latest partition receipt"
                )
            previous_partition_date = row["previous_partition_date"]
            previous_receipt = (
                None
                if previous_partition_date is None
                else CapacityIdentityRegistryReceipt(
                    schema_version=(CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION),
                    transition_count=int(row["previous_transition_count"]),
                    admitted_count=int(row["previous_admitted_count"]),
                    registry_sha256=str(row["previous_registry_sha256"]),
                )
            )
            return CapacityIdentityRegistryTip(
                policy_id=policy_id,
                partition_date=str(row["partition_date"]),
                previous_partition_date=(
                    None
                    if previous_partition_date is None
                    else str(previous_partition_date)
                ),
                receipt=receipt,
                previous_receipt=previous_receipt,
            )

    def verify(
        self,
        policy_id: str | None = None,
    ) -> (
        CapacityIdentityRegistryReceipt
        | None
        | dict[str, CapacityIdentityRegistryReceipt]
    ):
        """Recompute full chains and counts for one policy or the whole DB."""

        if policy_id is not None:
            policy_id = _identifier(policy_id, "policy_id")
        with self._transaction("DEFERRED") as connection:
            self._validate_schema_and_meta(connection)
            if policy_id is not None:
                return self._verify_policy_chain(connection, policy_id)
            integrity_rows = tuple(
                str(row[0]) for row in connection.execute("PRAGMA integrity_check")
            )
            if integrity_rows != ("ok",):
                raise CapacityIdentityRegistryIntegrityError(
                    "SQLite integrity_check failed"
                )
            policies = sorted(
                {
                    _identifier(str(row["policy_id"]), "policy_id")
                    for row in connection.execute(
                        """
                        SELECT policy_id FROM policy_state
                        UNION
                        SELECT policy_id FROM partition_receipts
                        UNION
                        SELECT policy_id FROM transition_identities
                        UNION
                        SELECT policy_id FROM admitted_capacity_identities
                        """
                    )
                }
            )
            results: dict[str, CapacityIdentityRegistryReceipt] = {}
            for current_policy_id in policies:
                receipt = self._verify_policy_chain(connection, current_policy_id)
                if receipt is None:
                    raise CapacityIdentityRegistryIntegrityError(
                        "registry table contains an orphan policy identity"
                    )
                results[current_policy_id] = receipt
            return results

    def _initialize_or_validate(self) -> None:
        with self._transaction("IMMEDIATE") as connection:
            tables = _user_table_names(connection)
            user_version = _user_version(connection)
            if not tables and user_version == 0:
                for ddl in _TABLE_DDLS.values():
                    connection.execute(ddl)
                connection.executemany(
                    "INSERT INTO registry_meta (key, value) VALUES (?, ?)",
                    sorted(_EXPECTED_META.items()),
                )
                connection.execute(
                    f"PRAGMA user_version = {S1_CAPACITY_IDENTITY_REGISTRY_SCHEMA_VERSION}"
                )
            self._validate_schema_and_meta(connection)

    @contextmanager
    def _transaction(self, mode: str) -> Iterator[sqlite3.Connection]:
        if self._connection.in_transaction:
            raise CapacityIdentityRegistryError("nested SQLite transaction is invalid")
        self._connection.execute(f"BEGIN {mode}")
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _validate_schema_and_meta(self, connection: sqlite3.Connection) -> None:
        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise CapacityIdentityRegistrySchemaError(
                "SQLite foreign key enforcement is disabled"
            )
        if _user_version(connection) != S1_CAPACITY_IDENTITY_REGISTRY_SCHEMA_VERSION:
            raise CapacityIdentityRegistrySchemaError(
                "SQLite user_version does not match registry schema"
            )
        table_names = _user_table_names(connection)
        if table_names != set(_TABLE_DDLS):
            raise CapacityIdentityRegistrySchemaError(
                "SQLite registry table set does not match schema"
            )
        unexpected_objects = tuple(
            (str(row["type"]), str(row["name"]))
            for row in connection.execute(
                """
                SELECT type, name
                FROM sqlite_master
                WHERE type IN ('view', 'trigger', 'index')
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )
        if unexpected_objects:
            raise CapacityIdentityRegistrySchemaError(
                "SQLite registry contains unexpected schema objects"
            )
        for table_name, expected_ddl in _TABLE_DDLS.items():
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            if row is None or _normalized_ddl(str(row["sql"])) != _normalized_ddl(
                expected_ddl
            ):
                raise CapacityIdentityRegistrySchemaError(
                    f"SQLite registry table schema drifted: {table_name}"
                )
        meta = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key, value FROM registry_meta ORDER BY key"
            )
        }
        if meta != _EXPECTED_META:
            raise CapacityIdentityRegistrySchemaError(
                "SQLite registry immutable metadata does not match schema"
            )

    def _latest_state_for_commit(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> tuple[str | None, CapacityIdentityRegistryReceipt | None]:
        state = connection.execute(
            "SELECT * FROM policy_state WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        latest_partition = connection.execute(
            """
            SELECT *
            FROM partition_receipts
            WHERE policy_id = ?
            ORDER BY partition_date DESC
            LIMIT 1
            """,
            (policy_id,),
        ).fetchone()
        if state is None:
            if latest_partition is not None:
                raise CapacityIdentityRegistryIntegrityError(
                    "partition rows exist without policy state"
                )
            orphan_transition = connection.execute(
                "SELECT 1 FROM transition_identities WHERE policy_id = ? LIMIT 1",
                (policy_id,),
            ).fetchone()
            orphan_admitted = connection.execute(
                """
                SELECT 1
                FROM admitted_capacity_identities
                WHERE policy_id = ?
                LIMIT 1
                """,
                (policy_id,),
            ).fetchone()
            if orphan_transition is not None or orphan_admitted is not None:
                raise CapacityIdentityRegistryIntegrityError(
                    "identity rows exist without policy state"
                )
            return None, None
        if latest_partition is None:
            raise CapacityIdentityRegistryIntegrityError(
                "policy state exists without partition receipt"
            )
        receipt = _validate_cached_partition_row(latest_partition, policy_id)
        if (
            str(state["latest_partition_date"])
            != str(latest_partition["partition_date"])
            or int(state["transition_count"]) != receipt.transition_count
            or int(state["admitted_count"]) != receipt.admitted_count
            or str(state["registry_sha256"]) != receipt.registry_sha256
        ):
            raise CapacityIdentityRegistryIntegrityError(
                "latest policy state disagrees with partition receipt"
            )
        return str(state["latest_partition_date"]), receipt

    def _idempotent_existing_receipt(
        self,
        connection: sqlite3.Connection,
        incoming: _IncomingPartition,
        existing: sqlite3.Row,
        expected_previous_receipt: CapacityIdentityRegistryReceipt | None,
    ) -> CapacityIdentityRegistryReceipt:
        state = connection.execute(
            "SELECT * FROM policy_state WHERE policy_id = ?",
            (incoming.policy_id,),
        ).fetchone()
        existing_receipt = _validate_cached_partition_row(
            existing,
            incoming.policy_id,
        )
        if (
            state is None
            or str(state["latest_partition_date"]) != incoming.partition_date
            or int(state["transition_count"]) != existing_receipt.transition_count
            or int(state["admitted_count"]) != existing_receipt.admitted_count
            or str(state["registry_sha256"]) != existing_receipt.registry_sha256
        ):
            raise CapacityIdentityRegistryConflictError(
                "only the latest partition may be retried idempotently"
            )
        previous_date = existing["previous_partition_date"]
        if previous_date is None:
            stored_previous_receipt = None
        else:
            previous_row = connection.execute(
                """
                SELECT *
                FROM partition_receipts
                WHERE policy_id = ? AND partition_date = ?
                """,
                (incoming.policy_id, str(previous_date)),
            ).fetchone()
            if previous_row is None:
                raise CapacityIdentityRegistryIntegrityError(
                    "partition receipt references a missing predecessor"
                )
            stored_previous_receipt = _receipt_from_partition_row(previous_row)
            if (
                stored_previous_receipt.transition_count
                != int(existing["previous_transition_count"])
                or stored_previous_receipt.admitted_count
                != int(existing["previous_admitted_count"])
                or stored_previous_receipt.registry_sha256
                != str(existing["previous_registry_sha256"])
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "partition predecessor receipt fields disagree"
                )
        _require_expected_previous_receipt(
            expected_previous_receipt,
            stored_previous_receipt,
        )

        stored_transitions = tuple(
            (int(row["sequence"]), str(row["transition_id"]))
            for row in connection.execute(
                """
                SELECT sequence, transition_id
                FROM transition_identities
                WHERE policy_id = ? AND sequence BETWEEN ? AND ?
                ORDER BY sequence
                """,
                (
                    incoming.policy_id,
                    int(existing["sequence_start"]),
                    int(existing["sequence_end"]),
                ),
            )
        )
        stored_admitted = tuple(
            (
                int(row["transition_sequence"]),
                str(row["transition_id"]),
                str(row["capacity_id"]),
            )
            for row in connection.execute(
                """
                SELECT transition_sequence, transition_id, capacity_id
                FROM admitted_capacity_identities
                WHERE policy_id = ? AND partition_date = ?
                ORDER BY ordinal
                """,
                (incoming.policy_id, incoming.partition_date),
            )
        )
        if (
            stored_transitions != incoming.transition_identities
            or stored_admitted != incoming.admitted_identities
            or str(existing["partition_input_sha256"])
            != incoming.partition_input_sha256
            or str(existing["ordered_transition_ids_sha256"])
            != incoming.ordered_transition_ids_sha256
            or str(existing["admitted_ids_sha256"]) != incoming.admitted_ids_sha256
        ):
            raise CapacityIdentityRegistryConflictError(
                "existing partition content differs from idempotent retry"
            )

        previous_transition_count = int(existing["previous_transition_count"])
        previous_admitted_count = int(existing["previous_admitted_count"])
        _validate_contiguous_sequences(
            incoming.transition_identities,
            previous_transition_count,
        )
        if int(existing["partition_transition_count"]) != len(stored_transitions):
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition transition count drifted"
            )
        if int(existing["partition_admitted_count"]) != len(stored_admitted):
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition admitted count drifted"
            )
        if int(existing["sequence_start"]) != previous_transition_count + 1:
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition sequence start drifted"
            )
        sequence_end = previous_transition_count + len(stored_transitions)
        if int(existing["sequence_end"]) != sequence_end:
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition sequence end drifted"
            )
        if int(existing["cumulative_transition_count"]) != sequence_end:
            raise CapacityIdentityRegistryIntegrityError(
                "existing cumulative transition count drifted"
            )
        cumulative_admitted_count = previous_admitted_count + len(stored_admitted)
        if int(existing["cumulative_admitted_count"]) != cumulative_admitted_count:
            raise CapacityIdentityRegistryIntegrityError(
                "existing cumulative admitted count drifted"
            )
        expected_registry_sha256 = _extend_registry_chain(
            previous_registry_sha256=str(existing["previous_registry_sha256"]),
            policy_id=incoming.policy_id,
            partition_date=incoming.partition_date,
            sequence_start=previous_transition_count + 1,
            sequence_end=sequence_end,
            ordered_transition_ids_sha256=incoming.ordered_transition_ids_sha256,
            admitted_ids_sha256=incoming.admitted_ids_sha256,
        )
        if str(existing["registry_sha256"]) != expected_registry_sha256:
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition registry chain drifted"
            )
        expected_record_sha256 = _partition_record_sha256(
            policy_id=incoming.policy_id,
            partition_date=incoming.partition_date,
            previous_partition_date=(
                str(previous_date) if previous_date is not None else None
            ),
            sequence_start=previous_transition_count + 1,
            sequence_end=sequence_end,
            partition_transition_count=len(stored_transitions),
            partition_admitted_count=len(stored_admitted),
            previous_transition_count=previous_transition_count,
            previous_admitted_count=previous_admitted_count,
            cumulative_transition_count=sequence_end,
            cumulative_admitted_count=cumulative_admitted_count,
            previous_registry_sha256=str(existing["previous_registry_sha256"]),
            ordered_transition_ids_sha256=incoming.ordered_transition_ids_sha256,
            admitted_ids_sha256=incoming.admitted_ids_sha256,
            partition_input_sha256=incoming.partition_input_sha256,
            registry_sha256=expected_registry_sha256,
        )
        if str(existing["partition_record_sha256"]) != expected_record_sha256:
            raise CapacityIdentityRegistryIntegrityError(
                "existing partition record digest drifted"
            )
        return CapacityIdentityRegistryReceipt(
            schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
            transition_count=sequence_end,
            admitted_count=cumulative_admitted_count,
            registry_sha256=expected_registry_sha256,
        )

    def _precheck_historical_duplicates(
        self,
        connection: sqlite3.Connection,
        incoming: _IncomingPartition,
    ) -> None:
        connection.execute("DROP TABLE IF EXISTS temp.incoming_transition_ids")
        connection.execute("DROP TABLE IF EXISTS temp.incoming_admitted_ids")
        connection.execute(
            "CREATE TEMP TABLE incoming_transition_ids (identity TEXT PRIMARY KEY) STRICT"
        )
        connection.execute(
            "CREATE TEMP TABLE incoming_admitted_ids (identity TEXT PRIMARY KEY) STRICT"
        )
        try:
            connection.executemany(
                "INSERT INTO incoming_transition_ids (identity) VALUES (?)",
                ((identity,) for _, identity in incoming.transition_identities),
            )
            connection.executemany(
                "INSERT INTO incoming_admitted_ids (identity) VALUES (?)",
                ((capacity_id,) for _, _, capacity_id in incoming.admitted_identities),
            )
            duplicate_transition = connection.execute(
                """
                SELECT existing.transition_id
                FROM transition_identities AS existing
                JOIN incoming_transition_ids AS incoming
                  ON incoming.identity = existing.transition_id
                WHERE existing.policy_id = ?
                LIMIT 1
                """,
                (incoming.policy_id,),
            ).fetchone()
            if duplicate_transition is not None:
                raise CapacityIdentityRegistryDuplicateError(
                    "transition_id was already registered for this policy: "
                    f"{duplicate_transition['transition_id']}"
                )
            duplicate_capacity = connection.execute(
                """
                SELECT existing.capacity_id
                FROM admitted_capacity_identities AS existing
                JOIN incoming_admitted_ids AS incoming
                  ON incoming.identity = existing.capacity_id
                WHERE existing.policy_id = ?
                LIMIT 1
                """,
                (incoming.policy_id,),
            ).fetchone()
            if duplicate_capacity is not None:
                raise CapacityIdentityRegistryDuplicateError(
                    "admitted capacity_id was already registered for this policy: "
                    f"{duplicate_capacity['capacity_id']}"
                )
        finally:
            connection.execute("DROP TABLE IF EXISTS temp.incoming_transition_ids")
            connection.execute("DROP TABLE IF EXISTS temp.incoming_admitted_ids")

    def _verify_policy_chain(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> CapacityIdentityRegistryReceipt | None:
        partitions = tuple(
            connection.execute(
                """
                SELECT *
                FROM partition_receipts
                WHERE policy_id = ?
                ORDER BY partition_date
                """,
                (policy_id,),
            )
        )
        state = connection.execute(
            "SELECT * FROM policy_state WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        if not partitions:
            transition_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM transition_identities
                    WHERE policy_id = ?
                    """,
                    (policy_id,),
                ).fetchone()["count"]
            )
            admitted_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM admitted_capacity_identities
                    WHERE policy_id = ?
                    """,
                    (policy_id,),
                ).fetchone()["count"]
            )
            if state is not None or transition_count != 0 or admitted_count != 0:
                raise CapacityIdentityRegistryIntegrityError(
                    "policy state or identities exist without partition receipts"
                )
            return None

        previous_date: str | None = None
        transition_count = 0
        admitted_count = 0
        registry_sha256 = _GENESIS_REGISTRY_SHA256

        for partition in partitions:
            current_date = _through_date(str(partition["partition_date"]))
            partition_previous_transition_count = transition_count
            partition_previous_admitted_count = admitted_count
            partition_previous_registry_sha256 = registry_sha256
            if previous_date is not None and current_date <= previous_date:
                raise CapacityIdentityRegistryIntegrityError(
                    "persisted partition dates are not strictly increasing"
                )
            stored_previous_date = partition["previous_partition_date"]
            if stored_previous_date != previous_date:
                raise CapacityIdentityRegistryIntegrityError(
                    "partition predecessor date does not match chain"
                )
            if (
                int(partition["previous_transition_count"]) != transition_count
                or int(partition["previous_admitted_count"]) != admitted_count
                or str(partition["previous_registry_sha256"]) != registry_sha256
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "partition predecessor receipt does not match chain"
                )

            transition_rows = tuple(
                connection.execute(
                    """
                    SELECT transition_id, sequence
                    FROM transition_identities
                    WHERE policy_id = ? AND sequence BETWEEN ? AND ?
                    ORDER BY sequence
                    """,
                    (
                        policy_id,
                        int(partition["sequence_start"]),
                        int(partition["sequence_end"]),
                    ),
                )
            )
            transition_identities: list[tuple[int, str]] = []
            for row in transition_rows:
                sequence = int(row["sequence"])
                transition_id = _identifier(
                    str(row["transition_id"]),
                    "transition_id",
                )
                transition_identities.append((sequence, transition_id))
            _validate_contiguous_sequences(
                tuple(transition_identities),
                transition_count,
                error_type=CapacityIdentityRegistryIntegrityError,
            )

            transition_by_sequence = {
                sequence: transition_id
                for sequence, transition_id in transition_identities
            }
            admitted_rows = tuple(
                connection.execute(
                    """
                    SELECT
                        capacity_id,
                        transition_id,
                        transition_sequence,
                        ordinal
                    FROM admitted_capacity_identities
                    WHERE policy_id = ? AND partition_date = ?
                    ORDER BY ordinal
                    """,
                    (policy_id, current_date),
                )
            )
            admitted_identities: list[tuple[int, str, str]] = []
            for ordinal, row in enumerate(admitted_rows):
                if int(row["ordinal"]) != ordinal:
                    raise CapacityIdentityRegistryIntegrityError(
                        "admitted identity ordinals are not contiguous"
                    )
                sequence = int(row["transition_sequence"])
                transition_id = _identifier(
                    str(row["transition_id"]),
                    "transition_id",
                )
                capacity_id = _identifier(str(row["capacity_id"]), "capacity_id")
                if transition_by_sequence.get(sequence) != transition_id:
                    raise CapacityIdentityRegistryIntegrityError(
                        "admitted identity does not reference its partition transition"
                    )
                admitted_identities.append((sequence, transition_id, capacity_id))

            partition_transition_count = len(transition_identities)
            partition_admitted_count = len(admitted_identities)
            sequence_start = transition_count + 1
            sequence_end = transition_count + partition_transition_count
            if (
                int(partition["sequence_start"]) != sequence_start
                or int(partition["sequence_end"]) != sequence_end
                or int(partition["partition_transition_count"])
                != partition_transition_count
                or int(partition["partition_admitted_count"])
                != partition_admitted_count
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "partition sequence range or local counts drifted"
                )
            ordered_digest = _ordered_transition_ids_sha256(
                tuple(transition_identities)
            )
            admitted_digest = _admitted_ids_sha256(tuple(admitted_identities))
            if (
                str(partition["ordered_transition_ids_sha256"]) != ordered_digest
                or str(partition["admitted_ids_sha256"]) != admitted_digest
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "partition identity digest drifted"
                )
            _validate_sha256(
                str(partition["partition_input_sha256"]),
                "partition input",
            )

            transition_count = sequence_end
            admitted_count += partition_admitted_count
            if (
                int(partition["cumulative_transition_count"]) != transition_count
                or int(partition["cumulative_admitted_count"]) != admitted_count
            ):
                raise CapacityIdentityRegistryIntegrityError(
                    "partition cumulative counts drifted"
                )
            registry_sha256 = _extend_registry_chain(
                previous_registry_sha256=registry_sha256,
                policy_id=policy_id,
                partition_date=current_date,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                ordered_transition_ids_sha256=ordered_digest,
                admitted_ids_sha256=admitted_digest,
            )
            if str(partition["registry_sha256"]) != registry_sha256:
                raise CapacityIdentityRegistryIntegrityError(
                    "partition cumulative registry chain drifted"
                )
            partition_record_sha256 = _partition_record_sha256(
                policy_id=policy_id,
                partition_date=current_date,
                previous_partition_date=previous_date,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                partition_transition_count=partition_transition_count,
                partition_admitted_count=partition_admitted_count,
                previous_transition_count=partition_previous_transition_count,
                previous_admitted_count=partition_previous_admitted_count,
                cumulative_transition_count=transition_count,
                cumulative_admitted_count=admitted_count,
                previous_registry_sha256=partition_previous_registry_sha256,
                ordered_transition_ids_sha256=ordered_digest,
                admitted_ids_sha256=admitted_digest,
                partition_input_sha256=str(partition["partition_input_sha256"]),
                registry_sha256=registry_sha256,
            )
            if str(partition["partition_record_sha256"]) != partition_record_sha256:
                raise CapacityIdentityRegistryIntegrityError(
                    "partition record digest drifted"
                )
            previous_date = current_date

        exact_transition_count = int(
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM transition_identities
                WHERE policy_id = ?
                """,
                (policy_id,),
            ).fetchone()["count"]
        )
        exact_admitted_count = int(
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM admitted_capacity_identities
                WHERE policy_id = ?
                """,
                (policy_id,),
            ).fetchone()["count"]
        )
        if (
            exact_transition_count != transition_count
            or exact_admitted_count != admitted_count
        ):
            raise CapacityIdentityRegistryIntegrityError(
                "exact registry row counts disagree with partition chain"
            )
        if state is None:
            raise CapacityIdentityRegistryIntegrityError(
                "partition receipts exist without policy state"
            )
        if (
            str(state["latest_partition_date"]) != previous_date
            or int(state["transition_count"]) != transition_count
            or int(state["admitted_count"]) != admitted_count
            or str(state["registry_sha256"]) != registry_sha256
        ):
            raise CapacityIdentityRegistryIntegrityError(
                "latest policy state does not match reconstructed chain"
            )
        return CapacityIdentityRegistryReceipt(
            schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
            transition_count=transition_count,
            admitted_count=admitted_count,
            registry_sha256=registry_sha256,
        )


def _incoming_partition(
    policy_id: str,
    partition_date: str,
    transitions: Iterable[CapacityTransition],
) -> _IncomingPartition:
    policy_id = _identifier(policy_id, "policy_id")
    partition_date = _through_date(partition_date)
    materialized = tuple(transitions)
    transition_identities: list[tuple[int, str]] = []
    admitted_identities: list[tuple[int, str, str]] = []
    seen_transition_ids: set[str] = set()
    seen_admitted_ids: set[str] = set()
    for transition in materialized:
        if not isinstance(transition, CapacityTransition):
            raise CapacityIdentityRegistryError(
                "partition rows must be CapacityTransition instances"
            )
        sequence = transition.sequence
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise CapacityIdentityRegistryError(
                "transition sequence must be a positive integer"
            )
        transition_id = _identifier(transition.transition_id, "transition_id")
        _identifier(transition.capacity_id, "capacity_id")
        if transition_id in seen_transition_ids:
            raise CapacityIdentityRegistryDuplicateError(
                f"partition contains duplicate transition_id: {transition_id}"
            )
        seen_transition_ids.add(transition_id)
        transition_identities.append((sequence, transition_id))
        if transition.event_type == "new_reservation_attempt":
            if not isinstance(transition.admitted, bool):
                raise CapacityIdentityRegistryError(
                    "reservation transition admitted flag must be boolean"
                )
            if transition.admitted:
                capacity_id = _identifier(transition.capacity_id, "capacity_id")
                if capacity_id in seen_admitted_ids:
                    raise CapacityIdentityRegistryDuplicateError(
                        "partition contains duplicate admitted capacity_id: "
                        f"{capacity_id}"
                    )
                seen_admitted_ids.add(capacity_id)
                admitted_identities.append((sequence, transition_id, capacity_id))
        elif transition.admitted is not None:
            raise CapacityIdentityRegistryError(
                "non-reservation transition admitted flag must be null"
            )
    identities = tuple(transition_identities)
    admitted = tuple(admitted_identities)
    return _IncomingPartition(
        policy_id=policy_id,
        partition_date=partition_date,
        transition_identities=identities,
        admitted_identities=admitted,
        ordered_transition_ids_sha256=_ordered_transition_ids_sha256(identities),
        admitted_ids_sha256=_admitted_ids_sha256(admitted),
        partition_input_sha256=_partition_input_sha256(
            policy_id,
            partition_date,
            materialized,
        ),
    )


def _validate_contiguous_sequences(
    identities: tuple[tuple[int, str], ...],
    previous_transition_count: int,
    *,
    error_type: type[CapacityIdentityRegistryError] = (
        CapacityIdentityRegistryConflictError
    ),
) -> None:
    expected = previous_transition_count + 1
    for sequence, _ in identities:
        if sequence != expected:
            raise error_type("partition transition sequence is not globally contiguous")
        expected += 1


def _ordered_transition_ids_sha256(
    identities: tuple[tuple[int, str], ...],
) -> str:
    rows_chain_sha256 = hashlib.sha256(
        f"{_TRANSITION_IDS_DOMAIN}:rows".encode()
    ).hexdigest()
    for ordinal, (sequence, transition_id) in enumerate(identities):
        rows_chain_sha256 = _canonical_sha256(
            {
                "domain": _TRANSITION_IDS_DOMAIN,
                "previous_sha256": rows_chain_sha256,
                "ordinal": ordinal,
                "sequence": sequence,
                "transition_id": transition_id,
            }
        )
    return _canonical_sha256(
        {
            "domain": _TRANSITION_IDS_DOMAIN,
            "identity_count": len(identities),
            "rows_chain_sha256": rows_chain_sha256,
        }
    )


def _admitted_ids_sha256(
    identities: tuple[tuple[int, str, str], ...],
) -> str:
    rows_chain_sha256 = hashlib.sha256(
        f"{_ADMITTED_IDS_DOMAIN}:rows".encode()
    ).hexdigest()
    for ordinal, (sequence, transition_id, capacity_id) in enumerate(identities):
        rows_chain_sha256 = _canonical_sha256(
            {
                "domain": _ADMITTED_IDS_DOMAIN,
                "previous_sha256": rows_chain_sha256,
                "ordinal": ordinal,
                "sequence": sequence,
                "transition_id": transition_id,
                "capacity_id": capacity_id,
            }
        )
    return _canonical_sha256(
        {
            "domain": _ADMITTED_IDS_DOMAIN,
            "identity_count": len(identities),
            "rows_chain_sha256": rows_chain_sha256,
        }
    )


def _partition_input_sha256(
    policy_id: str,
    partition_date: str,
    transitions: tuple[CapacityTransition, ...],
) -> str:
    rows_chain_sha256 = hashlib.sha256(
        f"{_PARTITION_INPUT_DOMAIN}:rows".encode()
    ).hexdigest()
    for ordinal, transition in enumerate(transitions):
        rows_chain_sha256 = _canonical_sha256(
            {
                "domain": _PARTITION_INPUT_DOMAIN,
                "previous_sha256": rows_chain_sha256,
                "ordinal": ordinal,
                "transition": transition.as_dict(),
            }
        )
    return _canonical_sha256(
        {
            "domain": _PARTITION_INPUT_DOMAIN,
            "policy_id": policy_id,
            "partition_date": partition_date,
            "transition_count": len(transitions),
            "rows_chain_sha256": rows_chain_sha256,
        }
    )


def _extend_registry_chain(
    *,
    previous_registry_sha256: str,
    policy_id: str,
    partition_date: str,
    sequence_start: int,
    sequence_end: int,
    ordered_transition_ids_sha256: str,
    admitted_ids_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "domain": _REGISTRY_CHAIN_DOMAIN,
            "previous_registry_sha256": previous_registry_sha256,
            "policy_id": policy_id,
            "partition_date": partition_date,
            "sequence_start": sequence_start,
            "sequence_end": sequence_end,
            "ordered_transition_ids_sha256": ordered_transition_ids_sha256,
            "admitted_ids_sha256": admitted_ids_sha256,
        }
    )


def _partition_record_sha256(
    *,
    policy_id: str,
    partition_date: str,
    previous_partition_date: str | None,
    sequence_start: int,
    sequence_end: int,
    partition_transition_count: int,
    partition_admitted_count: int,
    previous_transition_count: int,
    previous_admitted_count: int,
    cumulative_transition_count: int,
    cumulative_admitted_count: int,
    previous_registry_sha256: str,
    ordered_transition_ids_sha256: str,
    admitted_ids_sha256: str,
    partition_input_sha256: str,
    registry_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "domain": _PARTITION_RECORD_DOMAIN,
            "policy_id": policy_id,
            "partition_date": partition_date,
            "previous_partition_date": previous_partition_date,
            "sequence_start": sequence_start,
            "sequence_end": sequence_end,
            "partition_transition_count": partition_transition_count,
            "partition_admitted_count": partition_admitted_count,
            "previous_transition_count": previous_transition_count,
            "previous_admitted_count": previous_admitted_count,
            "cumulative_transition_count": cumulative_transition_count,
            "cumulative_admitted_count": cumulative_admitted_count,
            "previous_registry_sha256": previous_registry_sha256,
            "ordered_transition_ids_sha256": ordered_transition_ids_sha256,
            "admitted_ids_sha256": admitted_ids_sha256,
            "partition_input_sha256": partition_input_sha256,
            "registry_sha256": registry_sha256,
        }
    )


def _validate_cached_partition_row(
    row: sqlite3.Row,
    expected_policy_id: str,
) -> CapacityIdentityRegistryReceipt:
    try:
        policy_id = _identifier(str(row["policy_id"]), "policy_id")
        partition_date = _through_date(str(row["partition_date"]))
        previous_partition_date = (
            _through_date(str(row["previous_partition_date"]))
            if row["previous_partition_date"] is not None
            else None
        )
    except CapacityIdentityRegistryError as error:
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition identity or date is invalid"
        ) from error
    if policy_id != expected_policy_id:
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition policy identity drifted"
        )
    if (
        previous_partition_date is not None
        and previous_partition_date >= partition_date
    ):
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition predecessor date is not earlier"
        )

    numeric_fields = {
        name: int(row[name])
        for name in (
            "sequence_start",
            "sequence_end",
            "partition_transition_count",
            "partition_admitted_count",
            "previous_transition_count",
            "previous_admitted_count",
            "cumulative_transition_count",
            "cumulative_admitted_count",
        )
    }
    if (
        numeric_fields["sequence_start"] < 1
        or numeric_fields["sequence_end"] < 0
        or any(
            numeric_fields[name] < 0
            for name in (
                "partition_transition_count",
                "partition_admitted_count",
                "previous_transition_count",
                "previous_admitted_count",
                "cumulative_transition_count",
                "cumulative_admitted_count",
            )
        )
        or numeric_fields["partition_admitted_count"]
        > numeric_fields["partition_transition_count"]
        or numeric_fields["previous_admitted_count"]
        > numeric_fields["previous_transition_count"]
        or numeric_fields["cumulative_admitted_count"]
        > numeric_fields["cumulative_transition_count"]
    ):
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition counts are invalid"
        )
    if (
        numeric_fields["sequence_start"]
        != numeric_fields["previous_transition_count"] + 1
        or numeric_fields["sequence_end"]
        != numeric_fields["previous_transition_count"]
        + numeric_fields["partition_transition_count"]
        or numeric_fields["cumulative_transition_count"]
        != numeric_fields["sequence_end"]
        or numeric_fields["cumulative_admitted_count"]
        != numeric_fields["previous_admitted_count"]
        + numeric_fields["partition_admitted_count"]
    ):
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition sequence range or cumulative counts drifted"
        )

    sha_fields = {
        name: str(row[name])
        for name in (
            "previous_registry_sha256",
            "ordered_transition_ids_sha256",
            "admitted_ids_sha256",
            "partition_input_sha256",
            "registry_sha256",
            "partition_record_sha256",
        )
    }
    for name, value in sha_fields.items():
        _validate_sha256(value, f"cached partition {name}")
    expected_registry_sha256 = _extend_registry_chain(
        previous_registry_sha256=sha_fields["previous_registry_sha256"],
        policy_id=policy_id,
        partition_date=partition_date,
        sequence_start=numeric_fields["sequence_start"],
        sequence_end=numeric_fields["sequence_end"],
        ordered_transition_ids_sha256=sha_fields["ordered_transition_ids_sha256"],
        admitted_ids_sha256=sha_fields["admitted_ids_sha256"],
    )
    if sha_fields["registry_sha256"] != expected_registry_sha256:
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition registry chain drifted"
        )
    expected_record_sha256 = _partition_record_sha256(
        policy_id=policy_id,
        partition_date=partition_date,
        previous_partition_date=previous_partition_date,
        sequence_start=numeric_fields["sequence_start"],
        sequence_end=numeric_fields["sequence_end"],
        partition_transition_count=numeric_fields["partition_transition_count"],
        partition_admitted_count=numeric_fields["partition_admitted_count"],
        previous_transition_count=numeric_fields["previous_transition_count"],
        previous_admitted_count=numeric_fields["previous_admitted_count"],
        cumulative_transition_count=numeric_fields["cumulative_transition_count"],
        cumulative_admitted_count=numeric_fields["cumulative_admitted_count"],
        previous_registry_sha256=sha_fields["previous_registry_sha256"],
        ordered_transition_ids_sha256=sha_fields["ordered_transition_ids_sha256"],
        admitted_ids_sha256=sha_fields["admitted_ids_sha256"],
        partition_input_sha256=sha_fields["partition_input_sha256"],
        registry_sha256=sha_fields["registry_sha256"],
    )
    if sha_fields["partition_record_sha256"] != expected_record_sha256:
        raise CapacityIdentityRegistryIntegrityError(
            "cached partition record digest drifted"
        )
    return _receipt_from_partition_row(row)


def _receipt_from_partition_row(row: sqlite3.Row) -> CapacityIdentityRegistryReceipt:
    receipt = CapacityIdentityRegistryReceipt(
        schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
        transition_count=int(row["cumulative_transition_count"]),
        admitted_count=int(row["cumulative_admitted_count"]),
        registry_sha256=str(row["registry_sha256"]),
    )
    _validate_optional_receipt(receipt)
    return receipt


def _require_expected_previous_receipt(
    supplied: CapacityIdentityRegistryReceipt | None,
    actual: CapacityIdentityRegistryReceipt | None,
) -> None:
    if supplied != actual:
        raise CapacityIdentityRegistryConflictError(
            "expected_previous_receipt does not match registry lineage"
        )


def _validate_optional_receipt(
    receipt: CapacityIdentityRegistryReceipt | None,
) -> None:
    if receipt is None:
        return
    if not isinstance(receipt, CapacityIdentityRegistryReceipt):
        raise CapacityIdentityRegistryError(
            "expected_previous_receipt has an invalid type"
        )
    if (
        isinstance(receipt.schema_version, bool)
        or not isinstance(receipt.schema_version, int)
        or receipt.schema_version != CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION
    ):
        raise CapacityIdentityRegistryError(
            "expected_previous_receipt schema is unsupported"
        )
    for label, value in (
        ("transition_count", receipt.transition_count),
        ("admitted_count", receipt.admitted_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapacityIdentityRegistryError(
                f"expected_previous_receipt {label} is invalid"
            )
    if receipt.admitted_count > receipt.transition_count:
        raise CapacityIdentityRegistryError(
            "expected_previous_receipt admitted count exceeds transition count"
        )
    _validate_sha256(receipt.registry_sha256, "expected previous registry")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapacityIdentityRegistryError(f"{label} must be a non-empty string")
    return value


def _through_date(value: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isascii():
        raise CapacityIdentityRegistryError("partition_date must be YYYYMMDD")
    try:
        parsed = date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise CapacityIdentityRegistryError(
            "partition_date must be a valid YYYYMMDD date"
        ) from error
    if parsed.strftime("%Y%m%d") != value:
        raise CapacityIdentityRegistryError("partition_date must be canonical YYYYMMDD")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapacityIdentityRegistryIntegrityError(f"{label} SHA-256 is invalid")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _user_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _user_table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        )
    }


def _normalized_ddl(value: str) -> str:
    return " ".join(value.strip().rstrip(";").split())
