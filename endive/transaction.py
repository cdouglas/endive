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
    operation_type: str            # "fast_append", "merge_append", "validated_overwrite"
    runtime_ms: float              # Transaction runtime (execution phase)

    # Detailed I/O tracking
    manifest_list_reads: int
    manifest_list_writes: int
    manifest_file_reads: int
    manifest_file_writes: int

    # Timing decomposition (ms) — audit telemetry
    catalog_read_ms: float = 0.0       # Time to read initial catalog snapshot
    per_attempt_io_ms: float = 0.0     # Total time in per-attempt storage I/O
    conflict_io_ms: float = 0.0        # Total time in retry-specific I/O
    catalog_commit_ms: float = 0.0     # Total time in catalog.commit() calls

    # DES engine profiling
    event_count: int = 0               # Number of SimPy events processed by this txn


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
        self.partitions_written: Dict[int, FrozenSet[int]] = partitions_written if partitions_written is not None else {}

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

        # Timing decomposition accumulators (ms)
        self._catalog_read_ms = 0.0
        self._per_attempt_io_ms = 0.0
        self._conflict_io_ms = 0.0
        self._catalog_commit_ms = 0.0

    @property
    def status(self) -> TransactionStatus:
        return self._status

    @abstractmethod
    def can_have_real_conflict(self) -> bool:
        """Whether this operation type can encounter real conflicts."""
        ...

    def get_per_attempt_cost(self, ml_append_mode: bool) -> ConflictCost:
        """I/O cost paid on every commit attempt (before CAS).

        Every attempt must:
        - Read the current manifest list (1 ML read)
        - Write a new manifest file (1 MF write)
        - Write a new manifest list (1 ML write) unless in ML+ mode
        """
        return ConflictCost(
            manifest_list_reads=1,
            manifest_file_writes=1,
            manifest_list_writes=0 if ml_append_mode else 1,
        )

    @abstractmethod
    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        """Calculate additional I/O cost for conflict resolution on retry.

        This returns only the retry-specific cost (e.g., re-merge, I/O convoy).
        Per-attempt cost (ML read, MF write, ML write) is handled separately
        by get_per_attempt_cost().
        """
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
    # Write overlap detection
    # ------------------------------------------------------------------

    def has_write_overlap(
        self,
        old_snapshot: CatalogSnapshot,
        new_snapshot: CatalogSnapshot,
    ) -> bool:
        """Check if any table+partition this transaction wrote was also
        modified between old_snapshot and new_snapshot.

        Returns False for cross-table CAS failures and disjoint-partition
        conflicts, enabling the commit loop to skip per-attempt I/O on retry.
        """
        for table_id in self.tables_written:
            old_table = old_snapshot.get_table(table_id)
            new_table = new_snapshot.get_table(table_id)
            if old_table.version == new_table.version:
                continue
            # Table was modified — check partition-level overlap
            for pid in self.partitions_written.get(table_id, ()):
                if old_table.partition_versions[pid] != new_table.partition_versions[pid]:
                    return True
        return False

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
        before = self._elapsed
        self._start_snapshot = yield from self._yield_from(
            catalog.read(self.submit_time)
        )
        self._catalog_read_ms = self._elapsed - before
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

    @property
    def operation_type(self) -> str:
        """Return operation type string for this transaction class."""
        raise NotImplementedError

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
            operation_type=self.operation_type,
            runtime_ms=self.runtime,
            manifest_list_reads=self._ml_reads,
            manifest_list_writes=self._ml_writes,
            manifest_file_reads=self._mf_reads,
            manifest_file_writes=self._mf_writes,
            catalog_read_ms=self._catalog_read_ms,
            per_attempt_io_ms=self._per_attempt_io_ms,
            conflict_io_ms=self._conflict_io_ms,
            catalog_commit_ms=self._catalog_commit_ms,
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
        1. Pay per-attempt I/O (skipped if prior CAS failure was cross-table)
        2. Compute writes from current snapshot
        3. Call catalog.commit() (uniform interface)
        4. On success: return COMMITTED result
        5. On failure: read catalog, check write overlap, pay conflict I/O
        """
        last_snapshot = self._start_snapshot
        skip_per_attempt_io = False

        for attempt in range(max_retries + 1):
            writes = self._compute_writes(last_snapshot)

            # Per-attempt I/O: ML read + MF write + ML write
            # Skipped when the prior CAS failure was cross-table/disjoint
            if not skip_per_attempt_io:
                before = self._elapsed
                per_attempt = self.get_per_attempt_cost(ml_append_mode)
                yield from self._pay_conflict_cost(per_attempt, storage)
                self._per_attempt_io_ms += self._elapsed - before

            # CAS
            before = self._elapsed
            commit_result = yield from self._yield_from(
                catalog.commit(
                    expected_seq=last_snapshot.seq,
                    writes=writes,
                    timestamp_ms=self.submit_time + self._elapsed,
                    partitions_written=self.partitions_written,
                )
            )
            self._catalog_commit_ms += self._elapsed - before

            if commit_result.success:
                self._status = TransactionStatus.COMMITTED
                return self._make_result(TransactionStatus.COMMITTED)

            # CAS failed — read catalog to get current state
            before = self._elapsed
            current_snapshot = yield from self._yield_from(
                catalog.read(self.submit_time + self._elapsed)
            )
            self._catalog_read_ms += self._elapsed - before

            # Check whether the intervening commits overlap our writes
            overlap = self.has_write_overlap(last_snapshot, current_snapshot)

            if overlap:
                # Same-table/partition conflict: pay validation I/O
                before = self._elapsed
                n_behind = current_snapshot.seq - last_snapshot.seq
                cost = self.get_conflict_cost(n_behind, ml_append_mode)
                yield from self._pay_conflict_cost(cost, storage)
                self._conflict_io_ms += self._elapsed - before

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

            # Update state for retry
            last_snapshot = current_snapshot
            self._retries += 1
            skip_per_attempt_io = not overlap

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

    Per-attempt cost (paid every attempt):
    - 1 ML read + 1 MF write + 1 ML write (0 in ML+ mode)

    Additional retry cost: none
    """

    @property
    def operation_type(self) -> str:
        return "fast_append"

    def can_have_real_conflict(self) -> bool:
        return False

    def should_abort_on_real_conflict(self) -> bool:
        return False

    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        # No additional retry cost — per-attempt cost covers everything
        return ConflictCost()


class MergeAppendTransaction(Transaction):
    """Merge operation that must re-merge manifests on conflict.

    Semantics:
    - Merges data from multiple manifest files
    - No validation against existing data
    - On conflict: must re-merge with concurrent commits
    - Never aborts; always retries

    Per-attempt cost (paid every attempt):
    - 1 ML read + 1 MF write + 1 ML write (0 in ML+ mode)

    Additional retry cost:
    - N manifest file reads + N manifest file writes (re-merge)
    - N = n_behind * manifests_per_concurrent_commit
    """

    @property
    def operation_type(self) -> str:
        return "merge_append"

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
        # Only re-merge cost — per-attempt cost (ML read/MF write/ML write) is separate
        n_manifests = int(n_snapshots_behind * self._manifests_per_commit)
        return ConflictCost(
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

    Per-attempt cost (paid every attempt):
    - 1 ML read + 1 MF write + 1 ML write (0 in ML+ mode)

    Additional retry cost (I/O Convoy):
    - N historical manifest list reads (one per missed snapshot)
    """

    @property
    def operation_type(self) -> str:
        return "validated_overwrite"

    def can_have_real_conflict(self) -> bool:
        return True

    def should_abort_on_real_conflict(self) -> bool:
        return True

    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
    ) -> ConflictCost:
        # Only I/O convoy cost — per-attempt cost (ML read/MF write/ML write) is separate
        return ConflictCost(
            historical_ml_reads=n_snapshots_behind,
        )
