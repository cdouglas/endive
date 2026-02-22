"""Catalog abstraction per SPEC.md §2.

Uniform Catalog interface with read() and commit() only. Transactions
do not know whether the underlying mechanism is CAS or append.

Key types (public):
- TableMetadata: Immutable per-table metadata with partition versions
- CatalogSnapshot: Immutable snapshot of catalog state at a point in time
- CommitResult: Uniform result of Catalog.commit() (success/failure)
- IntentionRecord: Intention record for append-based commits
- Catalog: ABC with read() and commit()

Concrete implementations:
- CASCatalog: CAS-based, single round-trip commit
- AppendCatalog: Append + discovery read, two internal round-trips
- InstantCatalog: Fixed-latency CAS for testing

Internal types (not exposed to Transactions):
- CASResult, AppendResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import numpy as np

from endive.storage import StorageProvider


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableMetadata:
    """Immutable metadata for a single table."""
    table_id: int
    version: int
    num_partitions: int
    partition_versions: Tuple[int, ...]  # Per-partition version vector

    def with_version(self, new_version: int) -> TableMetadata:
        """Return copy with updated table version."""
        return TableMetadata(
            table_id=self.table_id,
            version=new_version,
            num_partitions=self.num_partitions,
            partition_versions=self.partition_versions,
        )

    def with_partition_version(self, partition_id: int, new_version: int) -> TableMetadata:
        """Return copy with a specific partition's version updated."""
        versions = list(self.partition_versions)
        versions[partition_id] = new_version
        return TableMetadata(
            table_id=self.table_id,
            version=self.version,
            num_partitions=self.num_partitions,
            partition_versions=tuple(versions),
        )


@dataclass(frozen=True)
class CatalogSnapshot:
    """Immutable snapshot of catalog state at a point in time.

    This is the ONLY way for transactions to observe catalog state.
    """
    seq: int                            # Global sequence number (total ordering)
    tables: Tuple[TableMetadata, ...]   # Per-table metadata
    timestamp_ms: float                 # Simulation time when snapshot was captured

    def get_table(self, table_id: int) -> TableMetadata:
        """Get metadata for a specific table."""
        return self.tables[table_id]

    def get_partition_version(self, table_id: int, partition_id: int) -> int:
        """Get version of specific partition."""
        return self.tables[table_id].partition_versions[partition_id]


@dataclass(frozen=True)
class CommitResult:
    """Uniform result of a catalog commit attempt.

    Returned by Catalog.commit(). The transaction does not know whether
    the underlying mechanism was CAS or append.

    On success: snapshot is None. The transaction knows its state was
    installed (it wrote the data, and the commit succeeded atomically).

    On failure: snapshot contains current catalog state for conflict
    resolution without an additional read round-trip.
    """
    success: bool
    snapshot: Optional[CatalogSnapshot]  # Present ONLY on failure
    latency_ms: float


@dataclass(frozen=True)
class IntentionRecord:
    """Intention record for append-based catalog commit.

    Contains the transaction's writes and preconditions. The Catalog
    evaluates preconditions server-side but does not report the outcome.
    """
    txn_id: int
    expected_seq: int                    # Expected catalog seq (precondition)
    tables_written: Dict[int, int]       # table_id -> new_version
    partitions_written: Dict[int, Tuple[int, ...]] | None = None  # table_id -> partition_ids
    size_bytes: int = 100                # Serialized size for latency calc


# Internal types (used within Catalog implementations only)

@dataclass(frozen=True)
class _CASResult:
    """Internal result of a CAS storage operation."""
    success: bool
    snapshot: Optional[CatalogSnapshot]
    latency_ms: float


@dataclass(frozen=True)
class _AppendResult:
    """Internal result of an append storage operation (physical only)."""
    physical_success: bool
    new_offset: Optional[int]
    latency_ms: float


# ---------------------------------------------------------------------------
# Internal mutable table state (held by Catalog, never exposed)
# ---------------------------------------------------------------------------

class _MutableTable:
    """Internal mutable table state. Not exposed outside Catalog."""
    __slots__ = ("table_id", "version", "num_partitions", "partition_versions")

    def __init__(self, table_id: int, num_partitions: int):
        self.table_id = table_id
        self.version = 0
        self.num_partitions = num_partitions
        self.partition_versions = [0] * num_partitions

    def to_metadata(self) -> TableMetadata:
        return TableMetadata(
            table_id=self.table_id,
            version=self.version,
            num_partitions=self.num_partitions,
            partition_versions=tuple(self.partition_versions),
        )


# ---------------------------------------------------------------------------
# Catalog ABC
# ---------------------------------------------------------------------------

class Catalog(ABC):
    """Abstract catalog for optimistic concurrency control.

    Interface:
    - read() -> CatalogSnapshot
    - commit(expected_seq, writes, ...) -> CommitResult

    Invariants:
    - seq advances by exactly 1 on each successful commit
    - seq never decreases or skips values
    - Snapshots are immutable
    """

    @abstractmethod
    def read(self, timestamp_ms: float = 0.0) -> Generator[float, None, CatalogSnapshot]:
        """Read current catalog state.

        Yields latency timeout, returns immutable snapshot.
        This is the ONLY way for transactions to observe catalog state.
        """
        ...

    @abstractmethod
    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        timestamp_ms: float = 0.0,
        intention: Optional[IntentionRecord] = None,
    ) -> Generator[float, None, CommitResult]:
        """Attempt atomic commit.

        The caller does not know whether the underlying mechanism is
        CAS or append. The implementation decides internally.

        Args:
            expected_seq: Catalog seq when transaction started
            writes: table_id -> new_version mapping
            timestamp_ms: Current simulation time (for snapshot timestamps)
            intention: Optional intention record (used by append-based catalogs)

        Yields:
            Latency timeout(s)

        Returns:
            CommitResult. On success: snapshot=None. On failure: snapshot for
            conflict resolution.
        """
        ...

    @property
    @abstractmethod
    def seq(self) -> int:
        """Current catalog sequence number (read-only)."""
        ...


# ---------------------------------------------------------------------------
# CASCatalog
# ---------------------------------------------------------------------------

class CASCatalog(Catalog):
    """Catalog using compare-and-swap for commits.

    commit() internally performs a CAS operation on the underlying storage.
    Single round-trip: the CAS response tells whether the commit succeeded.
    """

    def __init__(
        self,
        storage: StorageProvider,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
    ):
        if not storage.supports_cas:
            raise ValueError("CASCatalog requires storage with CAS support")
        if len(partitions_per_table) != num_tables:
            raise ValueError(
                f"partitions_per_table length ({len(partitions_per_table)}) "
                f"!= num_tables ({num_tables})"
            )
        self._storage = storage
        self._seq = 0
        self._tables = [
            _MutableTable(i, partitions_per_table[i])
            for i in range(num_tables)
        ]

    def _create_snapshot(self, timestamp_ms: float = 0.0) -> CatalogSnapshot:
        return CatalogSnapshot(
            seq=self._seq,
            tables=tuple(t.to_metadata() for t in self._tables),
            timestamp_ms=timestamp_ms,
        )

    def read(self, timestamp_ms: float = 0.0) -> Generator[float, None, CatalogSnapshot]:
        result = yield from self._storage.read(
            key="catalog_metadata",
            expected_size_bytes=100,
        )
        return self._create_snapshot(timestamp_ms)

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        timestamp_ms: float = 0.0,
        intention: Optional[IntentionRecord] = None,
    ) -> Generator[float, None, CommitResult]:
        # Single CAS round-trip
        result = yield from self._storage.cas(
            key="catalog_metadata",
            expected_version=expected_seq,
            size_bytes=100,
        )
        latency = result.latency_ms

        # Check CAS condition
        success = (self._seq == expected_seq)

        if success:
            # Apply writes atomically
            for table_id, version in writes.items():
                self._tables[table_id].version = version
            self._seq += 1
            return CommitResult(success=True, snapshot=None, latency_ms=latency)
        else:
            snapshot = self._create_snapshot(timestamp_ms)
            return CommitResult(success=False, snapshot=snapshot, latency_ms=latency)

    @property
    def seq(self) -> int:
        return self._seq


# ---------------------------------------------------------------------------
# AppendCatalog
# ---------------------------------------------------------------------------

class AppendCatalog(Catalog):
    """Catalog using append-based commits.

    commit() internally performs append + discovery read. The append
    protocol's complexity is hidden from the Transaction.

    Internally:
    1. Append intention record at expected offset
    2. If physical success: evaluate preconditions server-side
    3. Discovery read to determine outcome
    4. Return uniform CommitResult
    """

    def __init__(
        self,
        storage: StorageProvider,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
    ):
        if not storage.supports_append:
            raise ValueError("AppendCatalog requires storage with append support")
        if len(partitions_per_table) != num_tables:
            raise ValueError(
                f"partitions_per_table length ({len(partitions_per_table)}) "
                f"!= num_tables ({num_tables})"
            )
        self._storage = storage
        self._seq = 0
        self._log_offset = 0
        self._tables = [
            _MutableTable(i, partitions_per_table[i])
            for i in range(num_tables)
        ]

    def _create_snapshot(self, timestamp_ms: float = 0.0) -> CatalogSnapshot:
        return CatalogSnapshot(
            seq=self._seq,
            tables=tuple(t.to_metadata() for t in self._tables),
            timestamp_ms=timestamp_ms,
        )

    def _check_preconditions(self, intention: IntentionRecord) -> bool:
        """Check whether intention's preconditions are satisfied."""
        return self._seq == intention.expected_seq

    def _apply_writes(self, writes: Dict[int, int]) -> None:
        """Apply writes atomically."""
        for table_id, version in writes.items():
            self._tables[table_id].version = version
        self._seq += 1

    def read(self, timestamp_ms: float = 0.0) -> Generator[float, None, CatalogSnapshot]:
        result = yield from self._storage.read(
            key="catalog_log",
            expected_size_bytes=100,
        )
        return self._create_snapshot(timestamp_ms)

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        timestamp_ms: float = 0.0,
        intention: Optional[IntentionRecord] = None,
    ) -> Generator[float, None, CommitResult]:
        # Build intention if not provided
        if intention is None:
            intention = IntentionRecord(
                txn_id=-1,
                expected_seq=expected_seq,
                tables_written=writes,
            )

        total_latency = 0.0

        # Step 1: Physical append
        append_result = yield from self._storage.append(
            key="catalog_log",
            offset=self._log_offset,
            size_bytes=intention.size_bytes,
        )
        total_latency += append_result.latency_ms

        # Physical success check (offset matched)
        physical_ok = (self._log_offset == self._log_offset)  # Always true in sim
        # In simulation, the append always physically succeeds because we
        # control the offset. The server evaluates preconditions:
        if self._check_preconditions(intention):
            self._apply_writes(writes)
        self._log_offset += intention.size_bytes

        # Step 2: Discovery read (always needed for uniform CommitResult)
        read_result = yield from self._storage.read(
            key="catalog_log",
            expected_size_bytes=100,
        )
        total_latency += read_result.latency_ms

        # Determine success by checking if writes were applied
        snapshot = self._create_snapshot(timestamp_ms)
        committed = (snapshot.seq > expected_seq) and all(
            snapshot.get_table(tid).version == ver
            for tid, ver in writes.items()
        )

        return CommitResult(
            success=committed,
            snapshot=None if committed else snapshot,
            latency_ms=total_latency,
        )

    @property
    def seq(self) -> int:
        return self._seq


# ---------------------------------------------------------------------------
# InstantCatalog
# ---------------------------------------------------------------------------

class InstantCatalog(Catalog):
    """Catalog with instant (configurable fixed) latency for testing.

    Uses CAS semantics internally. No StorageProvider required.
    """

    def __init__(
        self,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
        latency_ms: float = 1.0,
    ):
        if len(partitions_per_table) != num_tables:
            raise ValueError(
                f"partitions_per_table length ({len(partitions_per_table)}) "
                f"!= num_tables ({num_tables})"
            )
        self._latency = float(latency_ms)
        self._seq = 0
        self._tables = [
            _MutableTable(i, partitions_per_table[i])
            for i in range(num_tables)
        ]

    def _create_snapshot(self, timestamp_ms: float = 0.0) -> CatalogSnapshot:
        return CatalogSnapshot(
            seq=self._seq,
            tables=tuple(t.to_metadata() for t in self._tables),
            timestamp_ms=timestamp_ms,
        )

    def read(self, timestamp_ms: float = 0.0) -> Generator[float, None, CatalogSnapshot]:
        yield self._latency
        return self._create_snapshot(timestamp_ms)

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        timestamp_ms: float = 0.0,
        intention: Optional[IntentionRecord] = None,
    ) -> Generator[float, None, CommitResult]:
        yield self._latency
        success = (self._seq == expected_seq)
        if success:
            for table_id, version in writes.items():
                self._tables[table_id].version = version
            self._seq += 1
            return CommitResult(success=True, snapshot=None, latency_ms=self._latency)
        else:
            snapshot = self._create_snapshot(timestamp_ms)
            return CommitResult(success=False, snapshot=snapshot, latency_ms=self._latency)

    @property
    def seq(self) -> int:
        return self._seq
