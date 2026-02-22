# Endive Simulator Refactoring Specification

**Version**: 1.0
**Status**: Draft
**Date**: 2026-02-21

## Executive Summary

This specification describes a refactored architecture for the Endive simulator. The current implementation uses global variables updated via code interleaving, making it brittle and difficult to modify. The new architecture defines independent modules with clear API boundaries and encapsulated state.

### Design Principles

1. **Message-Passing Only**: Information between Transaction, Catalog, and Storage flows only through messages that incur latency drawn from distributions
2. **Encapsulation**: Internal state is private; modules interact only through defined APIs
3. **Immutability**: Snapshots and results are immutable data structures
4. **Opaque Distributions**: Latency sampling is always delegated to distribution objects, never conditionally computed inline
5. **Fixed Topology**: Tables and partitions are fixed at simulation start

---

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Simulation                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                              SimPy Environment                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│         ┌───────────────────────────┼───────────────────────────┐           │
│         │                           │                           │           │
│         ▼                           ▼                           ▼           │
│  ┌─────────────┐           ┌─────────────────┐          ┌─────────────────┐│
│  │  Workload   │──────────▶│   Transaction   │─────────▶│    Catalog      ││
│  │  Generator  │           │   (active)      │          │ (CAS/Append/    ││
│  └─────────────┘           └─────────────────┘          │  Instant)       ││
│         │                          │                    └────────┬────────┘│
│         │                          │                             │          │
│         │                          │                             ▼          │
│         │                          │                    ┌─────────────────┐│
│         │                          │                    │    Storage      ││
│         │                          └───────────────────▶│    Provider     ││
│         │                                               └─────────────────┘│
│         │                                                        │          │
│         ▼                                                        ▼          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           Statistics Collector                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Storage Provider

The `Storage` provider abstracts cloud object storage with latency-bearing operations. Latencies are **always** drawn from opaque distribution objects provided at construction.

### 1.1 Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

@dataclass(frozen=True)
class StorageResult:
    """Immutable result of a storage operation."""
    success: bool
    latency_ms: float
    data_size_bytes: int

class StorageProvider(ABC):
    """Abstract storage provider with latency-bearing operations.

    All operations are generators that yield SimPy timeouts.
    Latencies are drawn from distribution objects provided at construction.
    """

    @abstractmethod
    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        """Read object from storage.

        Args:
            key: Object key/path
            expected_size_bytes: Expected size for latency calculation

        Yields:
            Latency timeout in milliseconds

        Returns:
            StorageResult with success status and actual latency
        """
        pass

    @abstractmethod
    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Write object to storage (unconditional PUT).

        Args:
            key: Object key/path
            size_bytes: Size of object being written

        Yields:
            Latency timeout in milliseconds

        Returns:
            StorageResult with success status
        """
        pass

    @abstractmethod
    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Compare-and-swap operation.

        Args:
            key: Object key/path
            expected_version: Version to compare against
            size_bytes: Size of new value

        Yields:
            Latency timeout in milliseconds

        Returns:
            StorageResult with success=True if CAS succeeded

        Raises:
            UnsupportedOperationError: If provider doesn't support CAS
        """
        pass

    @abstractmethod
    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Conditional append at specific offset.

        Args:
            key: Object key/path
            offset: Expected current offset (conditional)
            size_bytes: Size of data to append

        Yields:
            Latency timeout in milliseconds

        Returns:
            StorageResult with success=True if append at offset succeeded

        Raises:
            UnsupportedOperationError: If provider doesn't support append
        """
        pass

    @abstractmethod
    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Unconditional append to end of object (FIFO queue semantic).

        Args:
            key: Object key/path
            size_bytes: Size of data to append

        Yields:
            Latency timeout in milliseconds

        Returns:
            StorageResult with success status

        Raises:
            UnsupportedOperationError: If provider doesn't support tail_append
        """
        pass

    @property
    @abstractmethod
    def supports_cas(self) -> bool:
        """Whether this provider supports CAS operations."""
        pass

    @property
    @abstractmethod
    def supports_append(self) -> bool:
        """Whether this provider supports conditional append."""
        pass

    @property
    @abstractmethod
    def supports_tail_append(self) -> bool:
        """Whether this provider supports unconditional tail append."""
        pass


class UnsupportedOperationError(Exception):
    """Raised when storage operation is not supported by provider."""
    pass
```

### 1.2 Concrete Implementations

```python
@dataclass
class LatencyDistribution:
    """Opaque latency distribution that samples values."""

    def sample(self) -> float:
        """Draw a latency sample in milliseconds."""
        raise NotImplementedError


@dataclass
class LognormalLatency(LatencyDistribution):
    """Lognormal distribution with minimum floor."""
    mu: float
    sigma: float
    min_latency_ms: float = 1.0

    def sample(self) -> float:
        raw = np.random.lognormal(mean=self.mu, sigma=self.sigma)
        return max(raw, self.min_latency_ms)


@dataclass
class SizeBasedLatency(LatencyDistribution):
    """Size-dependent latency model: base + rate * size_mib + noise."""
    base_latency_ms: float
    latency_per_mib_ms: float
    sigma: float
    min_latency_ms: float = 1.0
    _size_bytes: int = 0  # Set before sampling

    def with_size(self, size_bytes: int) -> 'SizeBasedLatency':
        """Return copy with size set."""
        return SizeBasedLatency(
            self.base_latency_ms, self.latency_per_mib_ms,
            self.sigma, self.min_latency_ms, size_bytes
        )

    def sample(self) -> float:
        size_mib = self._size_bytes / (1024 * 1024)
        deterministic = self.base_latency_ms + self.latency_per_mib_ms * size_mib
        if self.sigma > 0:
            noisy = np.random.lognormal(mean=np.log(deterministic), sigma=self.sigma)
        else:
            noisy = deterministic
        return max(noisy, self.min_latency_ms)


class S3StorageProvider(StorageProvider):
    """AWS S3 Standard storage provider.

    Supports: read, write, cas
    Does not support: append, tail_append
    """

    def __init__(
        self,
        read_latency: LatencyDistribution,
        write_latency: SizeBasedLatency,
        cas_success_latency: LatencyDistribution,
        cas_failure_latency: LatencyDistribution,
    ):
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._cas_success = cas_success_latency
        self._cas_failure = cas_failure_latency

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._read_latency.sample()
        yield latency
        return StorageResult(success=True, latency_ms=latency, data_size_bytes=expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._write_latency.with_size(size_bytes).sample()
        yield latency
        return StorageResult(success=True, latency_ms=latency, data_size_bytes=size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        # Latency drawn BEFORE knowing success/failure
        # Caller determines success based on catalog state
        latency = self._cas_success.sample()  # Optimistic
        yield latency
        # Success determined by caller; this just models latency
        return StorageResult(success=True, latency_ms=latency, data_size_bytes=size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("S3 Standard does not support conditional append")

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("S3 Standard does not support tail append")

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return False

    @property
    def supports_tail_append(self) -> bool:
        return False


class S3ExpressStorageProvider(StorageProvider):
    """AWS S3 Express One Zone storage provider.

    Supports: read, write, cas, append
    Does not support: tail_append
    """
    # Similar to S3, but supports_append = True
    pass


class AzureBlobStorageProvider(StorageProvider):
    """Azure Blob Storage provider (Standard or Premium).

    Supports: read, write, cas, append
    Does not support: tail_append
    """
    pass


class GCPStorageProvider(StorageProvider):
    """Google Cloud Storage provider.

    Supports: read, write, cas
    Does not support: append, tail_append

    Note: GCP has extremely heavy-tailed CAS failure latencies.
    """
    pass


class InstantStorageProvider(StorageProvider):
    """Instant storage for testing (1ms latency).

    Supports: read, write, cas, append, tail_append
    """
    pass
```

### 1.3 Provider Capabilities Summary

| Provider | `read` | `write` | `cas` | `append` | `tail_append` |
|----------|--------|---------|-------|----------|---------------|
| S3 Standard | ✓ | ✓ | ✓ | ✗ | ✗ |
| S3 Express | ✓ | ✓ | ✓ | ✓ | ✗ |
| Azure Standard | ✓ | ✓ | ✓ | ✓ | ✗ |
| Azure Premium | ✓ | ✓ | ✓ | ✓ | ✗ |
| GCP | ✓ | ✓ | ✓ | ✗ | ✗ |
| Instant | ✓ | ✓ | ✓ | ✓ | ✓ |
| FIFO Queue* | ✓ | ✗ | ✗ | ✗ | ✓ |

*FIFO Queue is a hypothetical provider for research purposes.

---

## 2. Catalog

The `Catalog` manages optimistic concurrency control for table metadata. It wraps a `Storage` provider and exposes snapshot-based operations.

### 2.1 Core Abstractions

```python
from dataclasses import dataclass
from typing import FrozenSet, Dict, Tuple, Optional

@dataclass(frozen=True)
class TableMetadata:
    """Immutable metadata for a single table."""
    table_id: int
    version: int
    manifest_list_offset: int
    num_partitions: int
    partition_versions: Tuple[int, ...]  # Per-partition versions (or empty if not partitioned)


@dataclass(frozen=True)
class CatalogSnapshot:
    """Immutable snapshot of catalog state at a point in time.

    This is the ONLY way for transactions to observe catalog state.
    All fields are frozen to enforce immutability.
    """
    seq: int  # Global sequence number (total ordering)
    tables: Tuple[TableMetadata, ...]  # Per-table metadata
    timestamp_ms: float  # Simulation time when snapshot was captured

    def get_table(self, table_id: int) -> TableMetadata:
        """Get metadata for a specific table."""
        return self.tables[table_id]

    def get_partition_version(self, table_id: int, partition_id: int) -> int:
        """Get version of specific partition."""
        return self.tables[table_id].partition_versions[partition_id]


@dataclass(frozen=True)
class CASResult:
    """Result of a CAS operation.

    CAS is a single atomic request-response: the server evaluates
    the condition and returns success/failure.

    On success: no snapshot is returned. The transaction knows its
    state was installed because CAS guarantees the state it read was
    unmodified. The transaction already has all the information it
    needs (it wrote the state, so it knows what was committed).

    On failure: the current snapshot is returned so the transaction
    can resolve conflicts without an additional read round-trip.
    """
    success: bool
    snapshot: Optional[CatalogSnapshot]  # Present ONLY on failure (for conflict resolution)
    latency_ms: float


@dataclass(frozen=True)
class IntentionRecord:
    """Intention record for append-based catalog commit.

    Contains the transaction's writes and the preconditions that
    must hold for the intention to be applied. The Catalog evaluates
    these server-side but does not report the outcome to the caller.
    """
    txn_id: int
    tables_written: Dict[int, int]  # table_id -> new_version
    tables_read: Dict[int, int]     # table_id -> expected_version (preconditions)
    size_bytes: int = 100           # Serialized size for latency calculation


@dataclass(frozen=True)
class AppendResult:
    """Result of an append operation (physical outcome only).

    The append protocol is asymmetric with CAS: a successful physical
    append does NOT tell the transaction whether its preconditions were
    satisfied. The transaction must discover the logical outcome by
    subsequently reading the catalog.

    This models the real Iceberg append protocol:
    1. Transaction appends intention record at expected offset
    2. Storage confirms physical success (offset matched)
    3. Transaction reads catalog to discover whether its intention
       was applied (logical success) or whether a conflict occurred

    The Catalog internally evaluates logical preconditions when the
    append arrives, but this information is NOT returned to the caller.
    """
    physical_success: bool          # Offset matched (the ONLY thing the transaction learns)
    new_offset: Optional[int]       # Current offset (may not be returned on failure)
    latency_ms: float
    # NOTE: No logical_success field. No snapshot field.
    # The transaction must call catalog.read() to discover the outcome.


@dataclass(frozen=True)
class CommitResult:
    """Uniform result of a catalog commit attempt.

    This is the public-facing result type returned by Catalog.commit().
    The transaction does not know whether the underlying mechanism was
    CAS or append — it receives the same CommitResult either way.

    On success: snapshot is None. The transaction knows its state was
    installed (it wrote the data, and the commit succeeded atomically).

    On failure: snapshot contains the current catalog state, enabling
    conflict resolution without an additional read round-trip. For
    append-based catalogs, this requires an internal discovery read
    (the cost is included in latency_ms).
    """
    success: bool
    snapshot: Optional[CatalogSnapshot]  # Present ONLY on failure
    latency_ms: float
```

Note: `CASResult` and `AppendResult` are internal types used within
Catalog implementations. `CommitResult` is the only type exposed to
Transactions.

### 2.2 Catalog Interface

```python
from abc import ABC, abstractmethod

class Catalog(ABC):
    """Abstract catalog for optimistic concurrency control.

    Invariants:
    - seq advances by exactly 1 on each successful commit
    - seq never decreases or skips values
    - Snapshots are immutable and capture state at specific point in time
    """

    @abstractmethod
    def read(self) -> Generator[float, None, CatalogSnapshot]:
        """Read current catalog state.

        Yields latency timeout, returns immutable snapshot.
        This is the ONLY way for transactions to observe catalog state.
        """
        pass

    @abstractmethod
    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],  # table_id -> new_version
        intention: Optional['IntentionRecord'] = None,
    ) -> Generator[float, None, 'CommitResult']:
        """Attempt atomic commit.

        The caller does not know whether the underlying mechanism is
        CAS or append. The Catalog implementation decides internally.

        Args:
            expected_seq: Catalog seq when transaction started
            writes: Table versions to write
            intention: Optional intention record (used by append-based catalogs)

        Yields:
            Latency timeout(s)

        Returns:
            CommitResult with success status. On failure, includes
            current snapshot for conflict resolution.
        """
        pass


class CASCatalog(Catalog):
    """Catalog using compare-and-swap for commits.

    Traditional Iceberg commit protocol. commit() internally
    performs a CAS operation on the underlying storage.

    CAS is a single round-trip: the response tells the Catalog
    whether the commit succeeded. On success, the transaction
    already knows its state was installed (CAS guarantee). On
    failure, the Catalog captures a snapshot for conflict resolution.
    """

    def __init__(
        self,
        storage: StorageProvider,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
    ):
        if not storage.supports_cas:
            raise ValueError("CASCatalog requires storage with CAS support")
        self._storage = storage
        self._num_tables = num_tables
        self._partitions = partitions_per_table
        # Internal mutable state (not exposed)
        self._seq = 0
        self._tables = [...]  # Internal table state

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        intention: Optional['IntentionRecord'] = None,
    ) -> Generator[float, None, CommitResult]:
        """CAS-based commit. Single round-trip."""
        # Model CAS as request/response
        latency_half = ...  # Draw from distribution
        yield latency_half  # Request

        # Check condition
        success = (self._seq == expected_seq)

        if success:
            # Apply writes atomically
            for table_id, version in writes.items():
                self._tables[table_id].version = version
            self._seq += 1
            yield latency_half  # Response
            return CommitResult(
                success=True,
                snapshot=None,  # Success: no snapshot needed
                latency_ms=latency_half * 2,
            )
        else:
            # Capture snapshot for conflict resolution
            snapshot = self._create_snapshot()
            yield latency_half  # Response
            return CommitResult(
                success=False,
                snapshot=snapshot,  # Failure: snapshot for conflict resolution
                latency_ms=latency_half * 2,
            )


class AppendCatalog(Catalog):
    """Catalog using append-based commits.

    commit() internally performs an append + discovery read. The
    append protocol differs from CAS in its information flow, but
    this complexity is hidden from the Transaction.

    Internally:
    1. Append intention record at expected offset (physical operation)
    2. If physical failure: retry append at new offset (internal loop)
    3. If physical success: read catalog to discover logical outcome
    4. Return CommitResult (same interface as CAS)

    The extra discovery read is included in the commit latency. The
    Transaction sees the same CommitResult interface regardless of
    whether the underlying mechanism is CAS or append.

    ML+ interaction:
    - In ML+ mode, the Transaction appends to the manifest list
      BEFORE calling commit() (this is Transaction-level logic)
    - If commit succeeds, the ML entry is valid
    - If commit fails, the ML entry is orphaned (harmless)
    - On retry after conflict, the Transaction must read an ML that
      contains all committed transactions before retrying
    - The ML logical_success is undefined from Storage's perspective;
      it is determined only by the subsequent catalog state
    """

    def __init__(
        self,
        storage: StorageProvider,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
    ):
        if not storage.supports_append:
            raise ValueError("AppendCatalog requires storage with append support")
        self._storage = storage
        self._num_tables = num_tables
        self._partitions = partitions_per_table
        # Internal mutable state
        self._seq = 0
        self._log_offset = 0
        self._tables = [...]

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        intention: Optional['IntentionRecord'] = None,
    ) -> Generator[float, None, CommitResult]:
        """Append-based commit. Two round-trips (append + discovery read).

        The append and discovery read happen internally; the Transaction
        sees only the final CommitResult.
        """
        assert intention is not None, "AppendCatalog requires an IntentionRecord"

        # Step 1: Attempt physical append (may retry internally)
        append_result = yield from self._storage.append(
            key="catalog_log",
            offset=self._log_offset,  # Internal offset tracking
            size_bytes=intention.size_bytes,
        )
        total_latency = append_result.latency_ms

        physical_success = (self._log_offset == intention.expected_offset)

        if physical_success:
            # Server-side: evaluate preconditions (opaque to caller)
            logical_success = self._check_preconditions(intention)
            if logical_success:
                self._apply_intention(intention)
            self._log_offset += intention.size_bytes

        # Step 2: Discovery read (always needed after physical success;
        # on physical failure, read to get current state for caller)
        snapshot = yield from self.read()
        total_latency += ...  # Read latency included

        # Determine success: the Transaction's writes are reflected
        committed = self._writes_reflected(writes, snapshot)

        return CommitResult(
            success=committed,
            snapshot=None if committed else snapshot,
            latency_ms=total_latency,
        )


class InstantCatalog(Catalog):
    """Catalog with instant (1ms) operations for testing.

    Simulates ideal catalog with negligible latency.
    Uses CAS semantics internally.
    Useful for isolating storage latency effects.
    """

    def __init__(
        self,
        num_tables: int,
        partitions_per_table: Tuple[int, ...],
        latency_ms: float = 1.0,
    ):
        self._latency = latency_ms
        self._seq = 0
        self._tables = [...]

    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],
        intention: Optional['IntentionRecord'] = None,
    ) -> Generator[float, None, CommitResult]:
        yield self._latency
        success = (self._seq == expected_seq)
        if success:
            for table_id, version in writes.items():
                self._tables[table_id].version = version
            self._seq += 1
            return CommitResult(success=True, snapshot=None, latency_ms=self._latency)
        else:
            snapshot = self._create_snapshot()
            return CommitResult(success=False, snapshot=snapshot, latency_ms=self._latency)
```

---

## 3. Transaction Types

Transactions encapsulate the commit protocol and conflict handling. Each transaction type has different conflict resolution behavior.

### 3.1 Core Transaction Abstraction

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Set, Dict, FrozenSet

class TransactionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass
class TransactionResult:
    """Immutable result of transaction execution."""
    status: TransactionStatus
    commit_time_ms: float  # -1 if not committed
    abort_time_ms: float   # -1 if not aborted
    abort_reason: Optional[str]
    total_retries: int
    commit_latency_ms: float  # Time spent in commit protocol
    total_latency_ms: float   # End-to-end time

    # Detailed I/O tracking
    manifest_list_reads: int
    manifest_list_writes: int
    manifest_file_reads: int
    manifest_file_writes: int


class Transaction(ABC):
    """Abstract transaction with operation-specific conflict handling.

    Subclasses implement different Iceberg operation semantics:
    - FastAppend: Additive, no conflicts possible
    - MergeAppend: Must re-merge manifests on conflict
    - ValidatedOverwrite: Full validation, real conflicts abort
    """

    def __init__(
        self,
        txn_id: int,
        submit_time_ms: float,
        runtime_ms: float,
        tables_read: FrozenSet[int],
        tables_written: FrozenSet[int],
        partitions_read: Optional[Dict[int, FrozenSet[int]]] = None,
        partitions_written: Optional[Dict[int, FrozenSet[int]]] = None,
    ):
        self.id = txn_id
        self.submit_time = submit_time_ms
        self.runtime = runtime_ms
        self.tables_read = tables_read
        self.tables_written = tables_written
        self.partitions_read = partitions_read or {}
        self.partitions_written = partitions_written or {}

        # Mutable state (internal only)
        self._status = TransactionStatus.PENDING
        self._retries = 0
        self._snapshot: Optional[CatalogSnapshot] = None

    @abstractmethod
    def can_have_real_conflict(self) -> bool:
        """Whether this operation type can encounter real conflicts."""
        pass

    @abstractmethod
    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> 'ConflictCost':
        """Calculate I/O cost for resolving conflict."""
        pass

    @abstractmethod
    def should_abort_on_real_conflict(self) -> bool:
        """Whether to abort (vs retry) on real conflict."""
        pass

    def execute(
        self,
        sim: simpy.Environment,
        catalog: Catalog,
        storage: StorageProvider,
        conflict_detector: 'ConflictDetector',
        max_retries: int,
    ) -> Generator[float, None, TransactionResult]:
        """Execute transaction through commit protocol.

        This is the main entry point. The transaction:
        1. Reads catalog snapshot
        2. Executes for runtime duration
        3. Attempts commit (with retries on conflict)
        4. Returns result

        All I/O operations yield latency timeouts.
        """
        # Phase 1: Read catalog snapshot
        self._snapshot = yield from catalog.read()
        self._status = TransactionStatus.EXECUTING

        # Phase 2: Execute transaction work
        yield self.runtime

        # Phase 3: Commit protocol
        self._status = TransactionStatus.COMMITTING
        result = yield from self._commit_loop(
            sim, catalog, storage, conflict_detector, max_retries
        )

        return result

    @abstractmethod
    def _commit_loop(
        self,
        sim: simpy.Environment,
        catalog: Catalog,
        storage: StorageProvider,
        conflict_detector: 'ConflictDetector',
        max_retries: int,
    ) -> Generator[float, None, TransactionResult]:
        """Execute commit loop with retries."""
        pass


@dataclass(frozen=True)
class ConflictCost:
    """I/O operations required to resolve a conflict."""
    metadata_reads: int = 0
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    historical_ml_reads: int = 0  # For validation history
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0
```

### 3.2 Concrete Transaction Types

```python
class FastAppendTransaction(Transaction):
    """Append-only operation that cannot conflict.

    Semantics:
    - Appends new data files to table
    - No validation against existing data
    - Conflicts are always "false" (merge manifest pointers)
    - Never aborts on conflict; always retries

    Conflict Cost:
    - 1 manifest list read (current state)
    - 1 manifest list write (merged pointers)
    - In ML+ mode: 0 manifest list writes (entry already valid)
    """

    def can_have_real_conflict(self) -> bool:
        return False

    def should_abort_on_real_conflict(self) -> bool:
        return False  # Never aborts

    def get_conflict_cost(self, n_behind: int, ml_append_mode: bool) -> ConflictCost:
        return ConflictCost(
            metadata_reads=1,
            manifest_list_reads=1,
            manifest_list_writes=0 if ml_append_mode else 1,
        )


class MergeAppendTransaction(Transaction):
    """Merge operation that must re-merge manifests on conflict.

    Semantics:
    - Merges data from multiple manifest files
    - No validation against existing data
    - On conflict: must re-merge with concurrent commits
    - Never aborts; always retries

    Conflict Cost:
    - N manifest file reads (one per concurrent commit)
    - N manifest file writes (re-merged files)
    - 1 manifest list read + write
    """

    def __init__(self, *args, manifests_per_concurrent_commit: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._manifests_per_commit = manifests_per_concurrent_commit

    def can_have_real_conflict(self) -> bool:
        return False

    def should_abort_on_real_conflict(self) -> bool:
        return False

    def get_conflict_cost(self, n_behind: int, ml_append_mode: bool) -> ConflictCost:
        n_manifests = int(n_behind * self._manifests_per_commit)
        return ConflictCost(
            metadata_reads=1,
            manifest_list_reads=1,
            manifest_list_writes=0 if ml_append_mode else 1,
            manifest_file_reads=n_manifests,
            manifest_file_writes=n_manifests,
        )


class ValidatedOverwriteTransaction(Transaction):
    """Overwrite operation with full validation.

    Semantics:
    - Reads existing data and overwrites
    - Full validation via validationHistory()
    - Can have real conflicts (data overlap)
    - Aborts with ValidationException on real conflict

    Conflict Cost (I/O Convoy):
    - N historical manifest list reads (one per missed snapshot)
    - 1 current manifest list read + write
    - Possible abort before any writes
    """

    def __init__(self, *args, real_conflict_detector: 'RealConflictDetector', **kwargs):
        super().__init__(*args, **kwargs)
        self._conflict_detector = real_conflict_detector

    def can_have_real_conflict(self) -> bool:
        return True

    def should_abort_on_real_conflict(self) -> bool:
        return True  # ValidationException

    def get_conflict_cost(self, n_behind: int, ml_append_mode: bool) -> ConflictCost:
        return ConflictCost(
            metadata_reads=1,
            historical_ml_reads=n_behind,  # I/O convoy
            manifest_list_reads=1,
            manifest_list_writes=0 if ml_append_mode else 1,
        )
```

### 3.3 Commit Protocol

From the Transaction's perspective, the commit protocol is **uniform**:
call `catalog.commit()` and receive a `CommitResult`. The Transaction
does not know whether the underlying mechanism is CAS or append.

```
Transaction                              Catalog
    │                                       │
    ├──── commit(seq, writes) ─────────────▶│
    │                                       │  [implementation-specific:
    │                                       │   CAS, append+read, etc.]
    │                                       │
    │◀──── CommitResult ───────────────────│
    │                                       │
    │  success=True:  snapshot=None         │
    │    Transaction knows its state was    │
    │    installed (it wrote the data).     │
    │                                       │
    │  success=False: snapshot=<current>    │
    │    Conflict resolution using snapshot.│
```

#### Internal: CAS-Based Commit

```
Catalog (CAS impl)                                  Storage
    │                                                  │
    ├──── cas(key, expected_ver, data) ───────────────▶│
    │◀──── StorageResult ─────────────────────────────│
    │                                                  │
    │  [single round-trip; success/failure known       │
    │   immediately from CAS response]                 │
    │                                                  │
    │  On success: return CommitResult(True, None)     │
    │  On failure: capture snapshot,                   │
    │    return CommitResult(False, snapshot)           │
```

#### Internal: Append-Based Commit

```
Catalog (Append impl)                               Storage
    │                                                  │
    │  1. Physical append                              │
    ├──── append(key, offset, data) ──────────────────▶│
    │◀──── StorageResult ─────────────────────────────│
    │                                                  │
    │  [physical success: offset matched               │
    │   physical failure: offset moved]                │
    │                                                  │
    │  If physical success:                            │
    │    Server evaluates preconditions internally     │
    │    (logical outcome not returned to caller)      │
    │                                                  │
    │  2. Discovery read (internal to Catalog)         │
    ├──── read(key, size) ────────────────────────────▶│
    │◀──── StorageResult ─────────────────────────────│
    │                                                  │
    │  Catalog compares snapshot vs writes:            │
    │  - Writes reflected → CommitResult(True, None)   │
    │  - Not reflected → CommitResult(False, snapshot)  │
```

The append protocol requires **two internal round-trips** (append +
discovery read), but this is hidden from the Transaction. The
Transaction always sees one `commit()` call returning one `CommitResult`.

#### ML+ Manifest List Protocol

In ML+ (manifest list append) mode, the manifest list is updated
**before** the catalog commit. This is Transaction-level logic,
not Catalog-level:

```
Transaction                           Storage                      Catalog
    │                                    │                            │
    │  1. Append ML entry (tentative)    │                            │
    ├──── append(ml_key, offset, sz) ───▶│                            │
    │◀─── StorageResult ────────────────│                            │
    │                                    │                            │
    │  2. Attempt catalog commit (uniform interface)                  │
    ├──────────────────────────────────────── commit(seq, writes) ───▶│
    │◀──────────────────────────────────────── CommitResult ─────────│
    │                                    │                            │
    │  If committed:                     │                            │
    │    ML entry is now valid           │                            │
    │                                    │                            │
    │  If conflict:                      │                            │
    │    ML entry is orphaned (harmless) │                            │
    │    Must read ML with ALL committed │                            │
    │    transactions before retry       │                            │
    │                                    │                            │
    ├──── read(ml_key, size) ───────────▶│                            │
    │◀─── StorageResult ────────────────│                            │
    │                                    │                            │
    │  Verify ML contains all committed  │                            │
    │  entries, then retry commit        │                            │
```

Key invariant: the Transaction must not attempt to commit until it
has read a manifest list that reflects all committed transactions
up to the current catalog sequence number. Otherwise it would commit
with a stale view of the manifest list.

This means the ML append's "logical success" (whether the entry
represents a committed transaction) is **undefined at append time**.
It is determined only by the subsequent catalog commit outcome.

---

## 4. Workload Generator

The `Workload` class generates transactions. Rate and parameters are encapsulated and not visible outside.

### 4.1 Interface

```python
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class WorkloadConfig:
    """Immutable workload configuration."""
    # Inter-arrival distribution (opaque)
    inter_arrival: LatencyDistribution

    # Runtime distribution (opaque)
    runtime: LatencyDistribution

    # Operation type weights
    fast_append_weight: float = 0.7
    merge_append_weight: float = 0.2
    validated_overwrite_weight: float = 0.1

    # Topology (fixed at simulation start, known to Workload)
    num_tables: int
    partitions_per_table: Tuple[int, ...]  # Per-table partition counts

    # Table selection
    tables_per_txn: LatencyDistribution
    table_selection: 'TableSelector'

    # Partition selection (optional)
    partitions_per_txn: Optional[LatencyDistribution] = None
    partition_selection: Optional['PartitionSelector'] = None

    # Conflict parameters
    real_conflict_probability: float = 0.0
    data_overlap_probability: float = 0.0


class Workload:
    """Transaction generator with encapsulated rate and parameters.

    The Workload knows the table/partition topology and configures
    Transactions accordingly. Tables and partitions are fixed at
    simulation start. Some tables/partitions may be hotter than
    others (Zipf), but the selections are drawn from distributions.

    The rate at which transactions are produced and their parameters
    are not visible outside this class.
    """

    def __init__(
        self,
        config: WorkloadConfig,
        seed: Optional[int] = None,
    ):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._txn_counter = 0

    def generate(self, sim: simpy.Environment) -> Generator[Transaction, None, None]:
        """Generate transactions according to configured distribution.

        Yields Transaction objects at inter-arrival times.
        This is the ONLY interface for obtaining transactions.
        """
        while True:
            # Wait for next arrival
            inter_arrival = self._config.inter_arrival.sample()
            yield sim.timeout(inter_arrival)

            # Generate transaction parameters (encapsulated)
            txn = self._create_transaction(sim.now)
            yield txn

    def _create_transaction(self, submit_time: float) -> Transaction:
        """Create transaction with sampled parameters.

        This method is private; parameters are not exposed.
        """
        self._txn_counter += 1

        # Sample runtime
        runtime = self._config.runtime.sample()

        # Sample tables (topology known to Workload, not Catalog)
        n_tables = int(self._config.tables_per_txn.sample())
        tables_read, tables_written = self._config.table_selection.select(
            n_tables, self._config.num_tables
        )

        # Sample partitions (if enabled)
        partitions_read = {}
        partitions_written = {}
        if self._config.partitions_per_txn:
            for table_id in tables_read | tables_written:
                n_parts = self._config.partitions_per_table[table_id]
                pr, pw = self._config.partition_selection.select(
                    int(self._config.partitions_per_txn.sample()),
                    n_parts
                )
                partitions_read[table_id] = pr
                if table_id in tables_written:
                    partitions_written[table_id] = pw

        # Sample operation type
        op_type = self._sample_operation_type()

        # Create appropriate transaction type
        if op_type == 'fast_append':
            return FastAppendTransaction(
                txn_id=self._txn_counter,
                submit_time_ms=submit_time,
                runtime_ms=runtime,
                tables_read=frozenset(tables_read),
                tables_written=frozenset(tables_written),
                partitions_read=partitions_read,
                partitions_written=partitions_written,
            )
        elif op_type == 'merge_append':
            return MergeAppendTransaction(...)
        else:
            return ValidatedOverwriteTransaction(...)

    def _sample_operation_type(self) -> str:
        """Sample operation type from configured weights."""
        weights = [
            self._config.fast_append_weight,
            self._config.merge_append_weight,
            self._config.validated_overwrite_weight,
        ]
        return self._rng.choice(
            ['fast_append', 'merge_append', 'validated_overwrite'],
            p=weights / np.sum(weights)
        )
```

### 4.2 Table and Partition Selectors

```python
class TableSelector(ABC):
    """Selects which tables a transaction touches."""

    @abstractmethod
    def select(self, n_tables: int, total_tables: int) -> Tuple[Set[int], Set[int]]:
        """Select read and write table sets.

        Returns:
            (tables_read, tables_written) where tables_written ⊆ tables_read
        """
        pass


class ZipfTableSelector(TableSelector):
    """Zipf-distributed table selection (hot tables)."""

    def __init__(self, alpha: float = 1.5, write_fraction: float = 0.5):
        self._alpha = alpha
        self._write_fraction = write_fraction

    def select(self, n_tables: int, total_tables: int) -> Tuple[Set[int], Set[int]]:
        # Zipf sampling for read set
        pmf = truncated_zipf_pmf(total_tables, self._alpha)
        tables_read = set(np.random.choice(total_tables, size=n_tables, p=pmf, replace=False))

        # Write subset
        n_write = max(1, int(len(tables_read) * self._write_fraction))
        tables_written = set(np.random.choice(list(tables_read), size=n_write, replace=False))

        return tables_read, tables_written


class PartitionSelector(ABC):
    """Selects which partitions within a table are touched."""

    @abstractmethod
    def select(self, n_partitions: int, total_partitions: int) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """Select read and write partition sets."""
        pass


class ZipfPartitionSelector(PartitionSelector):
    """Zipf-distributed partition selection (hot partitions)."""
    pass


class UniformPartitionSelector(PartitionSelector):
    """Uniform random partition selection."""
    pass
```

---

## 5. Conflict Detection

Conflict detection is encapsulated and opaque. The detector determines if conflicts are "real" (data overlap) or "false" (no overlap).

```python
class ConflictDetector(ABC):
    """Detects whether a conflict is real or false.

    Real conflicts indicate actual data overlap that requires abort
    or complex merge. False conflicts are concurrent commits to
    unrelated data that can be merged.
    """

    @abstractmethod
    def is_real_conflict(
        self,
        txn: Transaction,
        current_snapshot: CatalogSnapshot,
        start_snapshot: CatalogSnapshot,
    ) -> bool:
        """Determine if conflict is real (data overlap) or false.

        Called when CAS fails. Returns True if the conflict involves
        actual data overlap (same partitions modified).
        """
        pass


class ProbabilisticConflictDetector(ConflictDetector):
    """Probabilistic conflict detection.

    Returns real conflict with configured probability.
    Simple model for parameter sweeps.
    """

    def __init__(self, real_conflict_probability: float):
        self._probability = real_conflict_probability

    def is_real_conflict(self, txn, current, start) -> bool:
        if not txn.can_have_real_conflict():
            return False
        return np.random.random() < self._probability


class PartitionOverlapConflictDetector(ConflictDetector):
    """Partition-based conflict detection.

    Returns real conflict if same (table, partition) pair was
    modified by concurrent transaction.
    """

    def is_real_conflict(self, txn, current, start) -> bool:
        if not txn.can_have_real_conflict():
            return False

        # Check each written partition
        for table_id, partitions in txn.partitions_written.items():
            for partition_id in partitions:
                start_ver = start.get_partition_version(table_id, partition_id)
                current_ver = current.get_partition_version(table_id, partition_id)
                if current_ver != start_ver:
                    return True  # Partition was modified

        return False
```

---

## 6. Simulation Runner

The simulation orchestrates all components.

```python
@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    duration_ms: float
    seed: Optional[int]

    # Components
    storage_provider: StorageProvider
    catalog: Catalog
    workload: Workload
    conflict_detector: ConflictDetector

    # Transaction limits
    max_retries: int = 10

    # Backoff configuration
    backoff_enabled: bool = False
    backoff_base_ms: float = 10.0
    backoff_multiplier: float = 2.0
    backoff_max_ms: float = 5000.0
    backoff_jitter: float = 0.1


class Simulation:
    """Main simulation runner.

    Coordinates workload generation, transaction execution,
    and statistics collection.
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        self._stats = Statistics()

    def run(self) -> Statistics:
        """Run simulation and return collected statistics."""
        if self._config.seed is not None:
            np.random.seed(self._config.seed)

        env = simpy.Environment()

        # Start workload generator
        env.process(self._run_workload(env))

        # Run until duration
        env.run(until=self._config.duration_ms)

        return self._stats

    def _run_workload(self, env: simpy.Environment) -> Generator:
        """Generate and execute transactions."""
        for txn in self._config.workload.generate(env):
            # Each transaction runs as separate process
            env.process(self._execute_transaction(env, txn))
            yield  # Allow other processes to run

    def _execute_transaction(
        self,
        env: simpy.Environment,
        txn: Transaction
    ) -> Generator:
        """Execute single transaction and record result."""
        result = yield from txn.execute(
            env,
            self._config.catalog,
            self._config.storage_provider,
            self._config.conflict_detector,
            self._config.max_retries,
        )

        self._stats.record_transaction(txn, result)


@dataclass
class Statistics:
    """Collected simulation statistics."""
    transactions: List[TransactionResult] = field(default_factory=list)

    # Aggregate counters
    committed: int = 0
    aborted: int = 0
    false_conflicts: int = 0
    real_conflicts: int = 0
    validation_exceptions: int = 0

    # I/O counters
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0

    def record_transaction(self, txn: Transaction, result: TransactionResult):
        """Record completed transaction."""
        self.transactions.append(result)
        if result.status == TransactionStatus.COMMITTED:
            self.committed += 1
        else:
            self.aborted += 1
            if result.abort_reason == 'validation_exception':
                self.validation_exceptions += 1

        self.manifest_list_reads += result.manifest_list_reads
        self.manifest_list_writes += result.manifest_list_writes
        # ...

    def to_dataframe(self) -> pd.DataFrame:
        """Export transactions to DataFrame for analysis."""
        pass

    def export_parquet(self, path: str):
        """Export to parquet file."""
        pass
```

---

## 7. Configuration Loading

Configuration is loaded from TOML and validated.

```python
def load_simulation_config(config_path: str) -> SimulationConfig:
    """Load simulation configuration from TOML file.

    This is the ONLY entry point for configuration.
    All parameters are validated before returning.
    """
    with open(config_path) as f:
        raw = toml.load(f)

    # Validate configuration
    errors, warnings = validate_config(raw)
    if errors:
        raise ConfigurationError(errors)
    for warning in warnings:
        logger.warning(warning)

    # Extract topology (shared between Catalog and Workload)
    num_tables = raw['catalog']['num_tables']
    partitions_per_table = _build_partition_counts(raw.get('catalog', {}))

    # Build storage provider
    storage = _build_storage_provider(raw.get('storage', {}))

    # Build catalog (topology passed at construction)
    catalog = _build_catalog(raw.get('catalog', {}), storage,
                             num_tables, partitions_per_table)

    # Build workload (topology passed via config, NOT from catalog)
    workload = _build_workload(raw.get('transaction', {}),
                               num_tables, partitions_per_table)

    # Build conflict detector
    conflict_detector = _build_conflict_detector(raw.get('conflict', {}))

    return SimulationConfig(
        duration_ms=raw['simulation']['duration_ms'],
        seed=raw['simulation'].get('seed'),
        storage_provider=storage,
        catalog=catalog,
        workload=workload,
        conflict_detector=conflict_detector,
        max_retries=raw['transaction'].get('retry', 10),
        # ...
    )
```

---

## 8. Invariants to Preserve

These invariants from the current implementation MUST be preserved:

### 8.1 Version Monotonicity
- `Catalog.seq` advances by exactly 1 on each successful commit
- Never decreases, never skips values

### 8.2 Snapshot Isolation
- Transactions see consistent snapshot from creation time
- State changes only visible after commit

### 8.3 Manifest List Exactness
- When N snapshots behind, read exactly N manifest lists
- Not N-1, not N+1

### 8.4 Conflict Type Distinction
- False conflicts: Different partitions, merge and retry
- Real conflicts: Same partition, may abort (operation-dependent)

### 8.5 Determinism
- Same seed produces identical results
- All randomness uses seeded RNG

### 8.6 Minimum Latency
- All operations have minimum latency floor (default 1ms)
- Prevents unrealistic zero-latency scenarios

### 8.7 Uniform Catalog Interface
- Transactions call only `read()` and `commit()` on the Catalog
- Transactions do NOT know whether the underlying mechanism is CAS or append
- `commit()` returns `CommitResult` with the same semantics for all implementations
- On success: `snapshot=None` (transaction knows its state was installed)
- On failure: `snapshot=<current>` (for conflict resolution, no extra read needed)
- Implementation details (CAS vs append, discovery reads) are internal

### 8.8 CAS Success Returns No State
- A successful CAS does not return a catalog snapshot
- The transaction knows its state was installed because CAS guarantees
  the state it read was unmodified
- Only failed CAS includes a snapshot (for conflict resolution)

### 8.9 Information Asymmetry in Append Protocol
- `Storage.append()` returns ONLY physical success (offset matched)
- Logical outcome (preconditions satisfied) is NEVER returned to caller
- The Catalog internally performs a discovery read to determine the outcome
- This complexity is hidden from the Transaction by the uniform `commit()` interface

### 8.10 ML+ Deferred Validity
- In ML+ mode, manifest list appends are tentative until catalog commit
- The ML entry's validity is determined by the catalog commit outcome
- Transaction must read an ML containing all committed transactions
  before attempting retry after conflict
- ML `append()` "logical success" is undefined; it is not a property
  of the storage operation

### 8.11 Topology Ownership
- Table and partition counts are fixed at simulation start
- The Workload knows the topology and configures Transactions accordingly
- The Catalog does NOT expose topology queries (`get_table_count()`, etc.)
- Some tables/partitions may be hotter than others (Zipf distribution)

---

## 9. Test Reusability Analysis

Based on analysis of existing tests:

### 9.1 Directly Reusable (100%)

These tests validate fundamental invariants and transfer directly:

- `test_exponential_backoff.py` (6 tests) - Algorithm correctness
- `test_numerical_accuracy.py` (10+ tests) - Mathematical invariants
- `test_statistical_rigor.py` (10+ tests) - Distribution properties
- `test_edge_cases.py` (8 tests) - Boundary conditions

### 9.2 Adaptable with New API

These tests validate correct behavior but need API adaptation:

- `test_simulator.py` - Update configuration builder
- `test_conflict_resolution.py` - Use new conflict detector interface
- `test_snapshot_versioning.py` - Use new catalog interface
- `test_operation_types.py` - Use new transaction types

### 9.3 Test Organization for New API

```
tests/
  core/
    test_determinism.py          # Seed-based reproducibility
    test_version_tracking.py     # Catalog invariants
    test_conflict_semantics.py   # Conflict detection

  storage/
    test_storage_provider.py     # Storage interface
    test_latency_distributions.py # Distribution sampling

  transactions/
    test_fast_append.py          # FastAppend behavior
    test_merge_append.py         # MergeAppend behavior
    test_validated_overwrite.py  # ValidatedOverwrite behavior

  workload/
    test_workload_generation.py  # Transaction generation
    test_table_selection.py      # Table/partition selection

  integration/
    test_full_simulation.py      # End-to-end tests
    test_edge_cases.py           # Boundary conditions
```

---

## 10. Migration Strategy

### Phase 1: Extract Storage Provider
1. Create `storage.py` with interface and implementations
2. Migrate latency generation functions
3. Verify determinism with existing tests

### Phase 2: Extract Catalog
1. Create `catalog.py` with interface and implementations
2. Migrate CAS/Append logic
3. Verify version invariants

### Phase 3: Extract Transaction Types
1. Create `transaction.py` with types
2. Migrate conflict resolution
3. Verify operation-specific behavior

### Phase 4: Extract Workload
1. Create `workload.py` with generator
2. Migrate table/partition selection
3. Verify distribution properties

### Phase 5: Integration
1. Create new `simulation.py` runner
2. Migrate statistics collection
3. Full test suite validation

---

## Appendix A: TOML Configuration Schema

```toml
[simulation]
duration_ms = 3600000
seed = 42
output_path = "results.parquet"

[experiment]
label = "exp_baseline"

[storage]
provider = "s3x"  # s3, s3x, azure, azurex, gcp, instant

[catalog]
type = "cas"  # cas, append, instant
num_tables = 1

[catalog.partitions]
enabled = true
num_partitions = 100

[transaction]
retry = 10
runtime.mean = 180000
runtime.sigma = 1.5

inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0

real_conflict_probability = 0.0

[transaction.operation_types]
fast_append = 0.7
merge_append = 0.2
validated_overwrite = 0.1

[transaction.retry_backoff]
enabled = true
base_ms = 10.0
multiplier = 2.0
max_ms = 5000.0
jitter = 0.1
```

---

## Appendix B: Glossary

- **CAS**: Compare-and-swap; atomic conditional update
- **ML+**: Manifest list append mode; avoids rewrite on false conflict
- **False Conflict**: Concurrent commit to unrelated data
- **Real Conflict**: Concurrent commit to overlapping data
- **I/O Convoy**: Reading N manifest lists for N missed snapshots
- **Snapshot Isolation**: Transaction sees consistent point-in-time state
- **Validation Exception**: Abort due to data overlap detection
