"""Transaction types per SPEC.md §3.

Transactions encapsulate the commit protocol and conflict handling.
Each type has different conflict resolution behavior:

- FastAppendTransaction: No validation, cheap retry (~160ms)
- MergeAppendTransaction: Re-merge manifests on retry
- ValidatedOverwriteTransaction: Full validation, aborts on real conflict

All transactions use catalog.commit() (uniform interface). They never
know whether the underlying mechanism is CAS or append.

Key types (public):
- TransactionStatus: Lifecycle enum
- ConflictCost: I/O operations for conflict resolution (frozen)
- TransactionResult: Immutable execution result
- ConflictDetector: ABC for real/false conflict detection
- Transaction: ABC with execute() entry point
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, Generator, Optional

from endive.catalog import Catalog, CatalogSnapshot, CommitResult
from endive.storage import StorageProvider


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------

class TransactionStatus(Enum):
    """Transaction lifecycle states."""
    PENDING = auto()
    EXECUTING = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class ConflictCost:
    """I/O operations required to resolve a conflict.

    All counts are number of I/O operations. Actual latency is
    determined by the StorageProvider's latency distributions.
    """
    metadata_reads: int = 0
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    historical_ml_reads: int = 0   # For validation history (I/O convoy)
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0


@dataclass(frozen=True)
class TransactionResult:
    """Immutable result of transaction execution."""
    status: TransactionStatus
    txn_id: int
    commit_time_ms: float          # Absolute sim time, -1 if not committed
    abort_time_ms: float           # Absolute sim time, -1 if not aborted
    abort_reason: Optional[str]
    total_retries: int
    commit_latency_ms: float       # Time in commit protocol only
    total_latency_ms: float        # End-to-end time

    # Detailed I/O tracking
    manifest_list_reads: int
    manifest_list_writes: int
    manifest_file_reads: int
    manifest_file_writes: int


# ---------------------------------------------------------------------------
# Conflict detection interface (implementations in endive-f34)
# ---------------------------------------------------------------------------

class ConflictDetector(ABC):
    """Abstract conflict detector per SPEC.md §5.

    Determines whether a conflict is "real" (data overlap requiring
    abort) or "false" (concurrent commits to unrelated data).
    """

    @abstractmethod
    def is_real_conflict(
        self,
        txn: Transaction,
        current_snapshot: CatalogSnapshot,
        start_snapshot: CatalogSnapshot,
    ) -> bool:
        """Determine if conflict is real (data overlap) or false.

        Called when catalog.commit() fails. Returns True if the
        conflict involves actual data overlap (same partitions).
        """
        ...


# ---------------------------------------------------------------------------
# Transaction ABC
# ---------------------------------------------------------------------------

class Transaction(ABC):
    """Abstract transaction with operation-specific conflict handling.

    Subclasses implement different Iceberg operation semantics:
    - FastAppendTransaction: Additive, no conflicts possible
    - MergeAppendTransaction: Must re-merge manifests on conflict
    - ValidatedOverwriteTransaction: Full validation, real conflicts abort
    """

    def __init__(
        self,
        txn_id: int,
        submit_time_ms: float,
        runtime_ms: float,
        tables_written: FrozenSet[int],
        partitions_written: Optional[Dict[int, FrozenSet[int]]] = None,
    ):
        self.id = txn_id
        self.submit_time = submit_time_ms
        self.runtime = runtime_ms
        self.tables_written = tables_written
        self.partitions_written = partitions_written or {}

        # Internal mutable state
        self._status = TransactionStatus.PENDING
        self._retries = 0
        self._start_snapshot: Optional[CatalogSnapshot] = None
        self._elapsed = 0.0
        self._commit_start_elapsed = 0.0

        # I/O counters
        self._ml_reads = 0
        self._ml_writes = 0
        self._mf_reads = 0
        self._mf_writes = 0

    @property
    def status(self) -> TransactionStatus:
        return self._status

    @abstractmethod
    def can_have_real_conflict(self) -> bool:
        """Whether this operation type can encounter real conflicts."""
        ...

    @abstractmethod
    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        """Calculate I/O cost for resolving a conflict."""
        ...

    @abstractmethod
    def should_abort_on_real_conflict(self) -> bool:
        """Whether to abort (vs retry) on real conflict."""
        ...

    # ------------------------------------------------------------------
    # Generator helpers
    # ------------------------------------------------------------------

    def _yield_from(self, gen: Generator) -> Generator:
        """Yield from sub-generator while tracking elapsed time.

        Each yielded float is accumulated into self._elapsed before
        being re-yielded to the outer caller.
        """
        try:
            latency = next(gen)
        except StopIteration as e:
            return e.value

        while True:
            self._elapsed += latency
            sent = yield latency
            try:
                latency = gen.send(sent)
            except StopIteration as e:
                return e.value

    # ------------------------------------------------------------------
    # Execute (main entry point)
    # ------------------------------------------------------------------

    def execute(
        self,
        catalog: Catalog,
        storage: StorageProvider,
        conflict_detector: ConflictDetector,
        max_retries: int = 10,
        ml_append_mode: bool = False,
    ) -> Generator[float, None, TransactionResult]:
        """Execute transaction through commit protocol.

        This is the main entry point. The transaction:
        1. Reads catalog snapshot
        2. Executes for runtime duration
        3. Attempts commit (with retries on conflict)
        4. Returns TransactionResult

        All I/O operations yield latency timeouts (floats).
        """
        # Phase 1: Read catalog snapshot
        self._start_snapshot = yield from self._yield_from(
            catalog.read(self.submit_time)
        )
        self._status = TransactionStatus.EXECUTING

        # Phase 2: Execute transaction work
        self._elapsed += self.runtime
        yield self.runtime

        # Phase 3: Commit protocol
        self._status = TransactionStatus.COMMITTING
        self._commit_start_elapsed = self._elapsed

        result = yield from self._commit_loop(
            catalog, storage, conflict_detector, max_retries, ml_append_mode
        )
        return result

    # ------------------------------------------------------------------
    # Commit loop
    # ------------------------------------------------------------------

    def _compute_writes(self, snapshot: CatalogSnapshot) -> Dict[int, int]:
        """Compute write versions based on current snapshot."""
        return {
            t: snapshot.get_table(t).version + 1
            for t in self.tables_written
        }

    def _make_result(
        self,
        status: TransactionStatus,
        abort_reason: Optional[str] = None,
    ) -> TransactionResult:
        """Create TransactionResult from current state."""
        return TransactionResult(
            status=status,
            txn_id=self.id,
            commit_time_ms=(
                self.submit_time + self._elapsed
                if status == TransactionStatus.COMMITTED
                else -1.0
            ),
            abort_time_ms=(
                self.submit_time + self._elapsed
                if status == TransactionStatus.ABORTED
                else -1.0
            ),
            abort_reason=abort_reason,
            total_retries=self._retries,
            commit_latency_ms=self._elapsed - self._commit_start_elapsed,
            total_latency_ms=self._elapsed,
            manifest_list_reads=self._ml_reads,
            manifest_list_writes=self._ml_writes,
            manifest_file_reads=self._mf_reads,
            manifest_file_writes=self._mf_writes,
        )

    def _commit_loop(
        self,
        catalog: Catalog,
        storage: StorageProvider,
        conflict_detector: ConflictDetector,
        max_retries: int,
        ml_append_mode: bool,
    ) -> Generator[float, None, TransactionResult]:
        """Execute commit loop with retries.

        On each attempt:
        1. Compute writes from current snapshot
        2. Call catalog.commit() (uniform interface)
        3. On success: return COMMITTED result
        4. On failure: check real conflict, pay I/O cost, retry
        """
        last_snapshot = self._start_snapshot

        for attempt in range(max_retries + 1):
            writes = self._compute_writes(last_snapshot)

            commit_result = yield from self._yield_from(
                catalog.commit(
                    expected_seq=last_snapshot.seq,
                    writes=writes,
                    timestamp_ms=self.submit_time + self._elapsed,
                )
            )

            if commit_result.success:
                self._status = TransactionStatus.COMMITTED
                return self._make_result(TransactionStatus.COMMITTED)

            # Commit failed — conflict resolution
            current_snapshot = commit_result.snapshot

            # Check for real conflict (only matters for validated ops)
            if self.can_have_real_conflict():
                is_real = conflict_detector.is_real_conflict(
                    self, current_snapshot, self._start_snapshot
                )
                if is_real and self.should_abort_on_real_conflict():
                    self._status = TransactionStatus.ABORTED
                    return self._make_result(
                        TransactionStatus.ABORTED,
                        abort_reason="validation_exception",
                    )

            # No more retries left — abort
            if attempt >= max_retries:
                break

            # Pay conflict resolution I/O cost
            n_behind = current_snapshot.seq - last_snapshot.seq
            cost = self.get_conflict_cost(n_behind, ml_append_mode)
            yield from self._pay_conflict_cost(cost, storage)

            # Update state for retry
            last_snapshot = current_snapshot
            self._retries += 1

        # All attempts exhausted
        self._status = TransactionStatus.ABORTED
        return self._make_result(
            TransactionStatus.ABORTED,
            abort_reason="max_retries_exceeded",
        )

    def _pay_conflict_cost(
        self,
        cost: ConflictCost,
        storage: StorageProvider,
    ) -> Generator[float, None, None]:
        """Pay I/O cost for conflict resolution via storage operations."""
        # Metadata reads
        for _ in range(cost.metadata_reads):
            yield from self._yield_from(
                storage.read(key="metadata", expected_size_bytes=1024)
            )

        # Historical ML reads (I/O convoy for validated operations)
        for _ in range(cost.historical_ml_reads):
            yield from self._yield_from(
                storage.read(key="manifest_list", expected_size_bytes=10240)
            )
            self._ml_reads += 1

        # Current ML reads
        for _ in range(cost.manifest_list_reads):
            yield from self._yield_from(
                storage.read(key="manifest_list", expected_size_bytes=10240)
            )
            self._ml_reads += 1

        # Manifest file reads
        for _ in range(cost.manifest_file_reads):
            yield from self._yield_from(
                storage.read(key="manifest_file", expected_size_bytes=102400)
            )
            self._mf_reads += 1

        # Manifest file writes
        for _ in range(cost.manifest_file_writes):
            yield from self._yield_from(
                storage.write(key="manifest_file", size_bytes=102400)
            )
            self._mf_writes += 1

        # ML writes
        for _ in range(cost.manifest_list_writes):
            yield from self._yield_from(
                storage.write(key="manifest_list", size_bytes=10240)
            )
            self._ml_writes += 1


# ---------------------------------------------------------------------------
# Concrete transaction types
# ---------------------------------------------------------------------------

class FastAppendTransaction(Transaction):
    """Append-only operation that cannot conflict.

    Semantics:
    - Appends new data files to table
    - No validation against existing data
    - Conflicts are always "false" (merge manifest pointers)
    - Never aborts on conflict; always retries

    Conflict Cost:
    - 1 metadata read
    - 1 manifest list read (current state)
    - 1 manifest list write (rewrite mode) or 0 (ML+ mode)
    """

    def can_have_real_conflict(self) -> bool:
        return False

    def should_abort_on_real_conflict(self) -> bool:
        return False

    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
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
    - 1 metadata read + 1 ML read + 0-1 ML write
    - N manifest file reads + N manifest file writes (re-merge)
    - N = n_behind * manifests_per_concurrent_commit
    """

    def __init__(
        self,
        *args,
        manifests_per_concurrent_commit: float = 1.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._manifests_per_commit = manifests_per_concurrent_commit

    def can_have_real_conflict(self) -> bool:
        return False

    def should_abort_on_real_conflict(self) -> bool:
        return False

    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        n_manifests = int(n_snapshots_behind * self._manifests_per_commit)
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
    - 1 metadata read + 1 current ML read + 0-1 ML write
    - Real conflicts abort BEFORE paying ML write cost
    """

    def can_have_real_conflict(self) -> bool:
        return True

    def should_abort_on_real_conflict(self) -> bool:
        return True

    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        return ConflictCost(
            metadata_reads=1,
            historical_ml_reads=n_snapshots_behind,  # I/O convoy
            manifest_list_reads=1,
            manifest_list_writes=0 if ml_append_mode else 1,
        )


# ---------------------------------------------------------------------------
# Legacy types (used by endive/main.py during migration, remove in cleanup)
# ---------------------------------------------------------------------------

from collections import defaultdict
from dataclasses import field

if not hasattr(ConflictCost, '_legacy_marker'):
    from typing import TYPE_CHECKING as _TC
    if _TC:
        from endive.snapshot import CatalogSnapshot as _LegacyCatalogSnapshot
        from endive.operation import OperationType as _LegacyOperationType
        from endive.operation import OperationBehavior as _LegacyOperationBehavior


@dataclass
class Txn:
    """Legacy transaction state (used by old main.py during migration)."""
    id: int
    t_submit: int
    t_runtime: int
    v_catalog_seq: int
    v_tblr: dict[int, int]
    v_tblw: dict[int, int]
    n_retries: int = 0
    t_commit: int = field(default=-1)
    t_abort: int = field(default=-1)
    v_dirty: dict[int, int] = field(default_factory=lambda: defaultdict(dict))
    v_log_offset: int = 0
    v_ml_offset: dict[int, int] = field(default_factory=dict)
    partitions_read: dict[int, set[int]] = field(default_factory=dict)
    partitions_written: dict[int, set[int]] = field(default_factory=dict)
    v_partition_seq: dict[int, dict[int, int]] = field(default_factory=dict)
    v_partition_ml_offset: dict[int, dict[int, int]] = field(default_factory=dict)
    start_snapshot: 'CatalogSnapshot | None' = None
    current_snapshot: 'CatalogSnapshot | None' = None
    operation_type: 'OperationType | None' = None
    abort_reason: str | None = None

    def get_behavior(self, manifests_per_commit: float = 1.0) -> 'OperationBehavior':
        from endive.operation import OperationType, get_behavior, FAST_APPEND_BEHAVIOR
        if self.operation_type is None:
            return FAST_APPEND_BEHAVIOR
        return get_behavior(self.operation_type, manifests_per_commit)


@dataclass
class LogEntry:
    """Legacy log entry (used by old main.py during migration)."""
    txn_id: int
    tables_written: dict[int, int]
    tables_read: dict[int, int]
    sealed: bool = False
