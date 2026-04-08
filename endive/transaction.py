"""Transaction types per SPEC.md §3.

Transactions encapsulate the commit protocol and conflict handling.
Each type has different conflict resolution behavior:

- FastAppendTransaction: No validation, cheap retry (~160ms)
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
class WriteOverlap:
    """Details of write overlap between transaction and intervening commits.

    Replaces the boolean has_write_overlap() with structured information
    about which tables/partitions overlap, enabling I/O cost scaling.
    """
    overlapping: Dict[int, FrozenSet[int]]  # table_id → overlapping partition IDs

    @property
    def has_overlap(self) -> bool:
        return bool(self.overlapping)

    @property
    def n_tables(self) -> int:
        return len(self.overlapping)

    @property
    def n_partitions(self) -> int:
        return sum(len(pids) for pids in self.overlapping.values())


NO_OVERLAP = WriteOverlap(overlapping={})


@dataclass(frozen=True)
class ConflictCost:
    """I/O operations required to resolve a conflict.

    All counts are number of I/O operations. Actual latency is
    determined by the StorageProvider's latency distributions.
    """
    table_metadata_reads: int = 0  # Table metadata file reads (non-inlined only)
    table_metadata_writes: int = 0 # Table metadata file writes (non-inlined only)
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    historical_ml_reads: int = 0   # For validation history (I/O convoy)


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
    operation_type: str            # "fast_append", "validated_overwrite"
    runtime_ms: float              # Transaction runtime (execution phase)

    # Detailed I/O tracking
    table_metadata_reads: int
    table_metadata_writes: int
    manifest_list_reads: int
    manifest_list_writes: int

    # Timing decomposition (ms) — audit telemetry
    catalog_read_ms: float = 0.0       # Time to read initial catalog snapshot
    per_attempt_io_ms: float = 0.0     # Total time in per-attempt storage I/O
    conflict_io_ms: float = 0.0        # Total time in retry-specific I/O
    catalog_commit_ms: float = 0.0     # Total time in catalog.commit() calls

    # DES engine profiling
    event_count: int = 0               # Number of SimPy events processed by this txn

    # Table/partition targeting (parallel pair lists)
    write_table_ids: tuple = ()        # Flattened table IDs
    write_partition_ids: tuple = ()    # Flattened partition IDs (parallel with table IDs)

    # Retry characterization
    catalog_conflicts: int = 0         # CAS failures with no write overlap
    tblptn_conflicts: int = 0          # CAS failures with write overlap
    max_snapshots_behind: int = 0      # Max (new_seq - old_seq) across retries


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
        self._tm_reads = 0
        self._tm_writes = 0
        self._ml_reads = 0
        self._ml_writes = 0

        # Timing decomposition accumulators (ms)
        self._catalog_read_ms = 0.0
        self._per_attempt_io_ms = 0.0
        self._conflict_io_ms = 0.0
        self._catalog_commit_ms = 0.0

        # Retry characterization accumulators
        self._catalog_conflicts = 0
        self._tblptn_conflicts = 0
        self._max_snapshots_behind = 0

    @property
    def status(self) -> TransactionStatus:
        return self._status

    @abstractmethod
    def can_have_real_conflict(self) -> bool:
        """Whether this operation type can encounter real conflicts."""
        ...

    def get_per_attempt_cost(
        self,
        ml_append_mode: bool,
        n_partitions: int = 1,
        metadata_inlined: bool = False,
    ) -> ConflictCost:
        """I/O cost paid on every commit attempt (before CAS).

        Every attempt must, per partition:
        - Read the table metadata file (1 TM read, non-inlined only)
        - Read the current manifest list (1 ML read)
        - Write a new manifest list (1 ML write) unless in ML+ mode
        - Write the table metadata file (1 TM write, non-inlined only)

        When metadata_inlined=True, table metadata is stored in the catalog
        CAS object. The TM file doesn't exist, so TM reads/writes are
        eliminated. ML reads/writes remain (the ML is not inlined).

        Args:
            ml_append_mode: If True, skip manifest list writes.
            n_partitions: Number of partitions needing manifest work.
            metadata_inlined: If True, skip table metadata file I/O.
        """
        return ConflictCost(
            table_metadata_reads=0 if metadata_inlined else 1,
            table_metadata_writes=0 if metadata_inlined else 1,
            manifest_list_reads=n_partitions,
            manifest_list_writes=0 if ml_append_mode else n_partitions,
        )

    @abstractmethod
    def get_conflict_cost(
        self,
        n_snapshots_behind: int,
        ml_append_mode: bool,
        n_partitions: int = 1,
    ) -> ConflictCost:
        """Calculate additional I/O cost for conflict resolution on retry.

        This returns only the retry-specific cost (e.g., re-merge, I/O convoy).
        Per-attempt cost (ML read, ML write) is handled separately
        by get_per_attempt_cost().

        Args:
            n_snapshots_behind: Number of catalog versions behind.
            ml_append_mode: If True, ML+ mode is active.
            n_partitions: Number of overlapping partitions (scales I/O cost).
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

    def compute_write_overlap(
        self,
        old_snapshot: CatalogSnapshot,
        new_snapshot: CatalogSnapshot,
    ) -> WriteOverlap:
        """Compute structured write overlap between this transaction and
        intervening commits between old_snapshot and new_snapshot.

        Returns a WriteOverlap with the set of overlapping partitions per
        table, enabling I/O cost scaling by overlap size.
        """
        overlapping: Dict[int, FrozenSet[int]] = {}
        for table_id in self.tables_written:
            old_table = old_snapshot.get_table(table_id)
            new_table = new_snapshot.get_table(table_id)
            if old_table.version == new_table.version:
                continue
            # Table was modified — check partition-level overlap
            overlap_pids = frozenset(
                pid for pid in self.partitions_written.get(table_id, ())
                if old_table.partition_versions[pid] != new_table.partition_versions[pid]
            )
            if overlap_pids:
                overlapping[table_id] = overlap_pids
        if not overlapping:
            return NO_OVERLAP
        return WriteOverlap(overlapping=overlapping)

    def has_write_overlap(
        self,
        old_snapshot: CatalogSnapshot,
        new_snapshot: CatalogSnapshot,
    ) -> bool:
        """Check if any table+partition this transaction wrote was also
        modified between old_snapshot and new_snapshot.

        Backward-compatible wrapper around compute_write_overlap().
        """
        return self.compute_write_overlap(old_snapshot, new_snapshot).has_overlap

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
        metadata_inlined: bool = False,
    ) -> Generator[float, None, TransactionResult]:
        """Execute transaction through commit protocol.

        This is the main entry point. The transaction:
        1. Reads catalog snapshot
        2. Executes for runtime duration
        3. Attempts commit (with retries on conflict)
        4. Returns TransactionResult

        All I/O operations yield latency timeouts (floats).

        Args:
            metadata_inlined: If True, table metadata is inlined in the
                catalog CAS object. Eliminates table metadata file I/O
                from per-attempt cost.
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
            catalog, storage, conflict_detector, max_retries,
            ml_append_mode, metadata_inlined,
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
        # Flatten partitions_written dict into parallel pair lists
        _tids, _pids = [], []
        for tid in sorted(self.partitions_written):
            for pid in sorted(self.partitions_written[tid]):
                _tids.append(tid)
                _pids.append(pid)

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
            table_metadata_reads=self._tm_reads,
            table_metadata_writes=self._tm_writes,
            manifest_list_reads=self._ml_reads,
            manifest_list_writes=self._ml_writes,
            catalog_read_ms=self._catalog_read_ms,
            per_attempt_io_ms=self._per_attempt_io_ms,
            conflict_io_ms=self._conflict_io_ms,
            catalog_commit_ms=self._catalog_commit_ms,
            write_table_ids=tuple(_tids),
            write_partition_ids=tuple(_pids),
            catalog_conflicts=self._catalog_conflicts,
            tblptn_conflicts=self._tblptn_conflicts,
            max_snapshots_behind=self._max_snapshots_behind,
        )

    def _commit_loop(
        self,
        catalog: Catalog,
        storage: StorageProvider,
        conflict_detector: ConflictDetector,
        max_retries: int,
        ml_append_mode: bool,
        metadata_inlined: bool = False,
    ) -> Generator[float, None, TransactionResult]:
        """Execute commit loop with retries.

        On each attempt:
        1. Pay per-attempt I/O scaled by partition count
        2. Compute writes from current snapshot
        3. Call catalog.commit() (uniform interface)
        4. On success: return COMMITTED result
        5. On failure: read catalog, compute overlap, pay scaled conflict I/O
        """
        last_snapshot = self._start_snapshot
        # First attempt: all written partitions need manifest work
        n_written = sum(len(pids) for pids in self.partitions_written.values())
        per_attempt_n = n_written

        for attempt in range(max_retries + 1):
            writes = self._compute_writes(last_snapshot)

            # Per-attempt I/O scaled by partitions needing manifest work.
            # When per_attempt_n == 0 (disjoint retry), the retry is free:
            # no partitions overlap, so no ML or TM work is needed.
            # Each partition has its own entry in the catalog, so disjoint
            # writes don't interact.
            if per_attempt_n > 0:
                before = self._elapsed
                per_attempt = self.get_per_attempt_cost(
                    ml_append_mode, n_partitions=per_attempt_n,
                    metadata_inlined=metadata_inlined,
                )
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
            # Non-inlined: reading the catalog only gets a pointer to the
            # table metadata file. A separate TM read is needed to get
            # partition versions for overlap detection.
            if not metadata_inlined:
                yield from self._yield_from(
                    storage.read(key="table_metadata",
                                 expected_size_bytes=10240)
                )
                self._tm_reads += 1
            self._catalog_read_ms += self._elapsed - before

            # Compute structured overlap with intervening commits
            overlap = self.compute_write_overlap(last_snapshot, current_snapshot)
            n_behind = current_snapshot.seq - last_snapshot.seq
            self._max_snapshots_behind = max(self._max_snapshots_behind, n_behind)

            if overlap.has_overlap:
                # Same-table/partition conflict: pay scaled validation I/O
                # Use per-table version deltas (not catalog seq delta) to
                # determine I/O cost.  Only commits to overlapping tables
                # require manifest list / manifest file work.
                n_table_versions_behind = sum(
                    current_snapshot.get_table(tid).version
                    - last_snapshot.get_table(tid).version
                    for tid in overlap.overlapping
                )
                before = self._elapsed
                cost = self.get_conflict_cost(
                    n_table_versions_behind, ml_append_mode,
                    n_partitions=overlap.n_partitions,
                )
                yield from self._pay_conflict_cost(cost, storage)
                self._conflict_io_ms += self._elapsed - before

                # Check for real conflict (only matters for validated ops)
                if self.can_have_real_conflict():
                    is_real = conflict_detector.is_real_conflict(
                        self, current_snapshot, self._start_snapshot
                    )
                    if is_real and self.should_abort_on_real_conflict():
                        self._tblptn_conflicts += 1
                        self._status = TransactionStatus.ABORTED
                        return self._make_result(
                            TransactionStatus.ABORTED,
                            abort_reason="validation_exception",
                        )

            # No more retries left — abort
            if attempt >= max_retries:
                break

            # Update state for retry — conflict type tracked per retry
            if overlap.has_overlap:
                self._tblptn_conflicts += 1
            else:
                self._catalog_conflicts += 1
            last_snapshot = current_snapshot
            self._retries += 1
            per_attempt_n = overlap.n_partitions  # 0 = skip on next attempt

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
        # Table metadata reads (non-inlined only)
        for _ in range(cost.table_metadata_reads):
            yield from self._yield_from(
                storage.read(key="table_metadata", expected_size_bytes=10240)
            )
            self._tm_reads += 1

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

        # ML writes
        for _ in range(cost.manifest_list_writes):
            yield from self._yield_from(
                storage.write(key="manifest_list", size_bytes=10240)
            )
            self._ml_writes += 1

        # Table metadata writes (non-inlined only)
        for _ in range(cost.table_metadata_writes):
            yield from self._yield_from(
                storage.write(key="table_metadata", size_bytes=10240)
            )
            self._tm_writes += 1


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
        n_partitions: int = 1,
    ) -> ConflictCost:
        # No additional retry cost — per-attempt cost covers everything
        return ConflictCost()



class ValidatedOverwriteTransaction(Transaction):
    """Overwrite operation with full validation.

    Semantics:
    - Reads existing data and overwrites
    - Full validation via validationHistory()
    - Can have real conflicts (data overlap)
    - Aborts with ValidationException on real conflict

    Per-attempt cost (paid every attempt):
    - 1 ML read + 1 ML write (0 in ML+ mode)

    Additional retry cost (I/O Convoy):
    - N-1 historical manifest list reads (one per missed snapshot)
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
        n_partitions: int = 1,
    ) -> ConflictCost:
        # I/O convoy: read historical MLs at versions K+1..K+N to validate.
        # The per-attempt cost already reads the ML at version K+N, so only
        # K+1..K+N-1 need additional reads (N-1, not N).
        #
        # Inlining the table manifest does NOT eliminate the convoy. The
        # current ML state is free (from the catalog read), but historical
        # ML states must still be read to validate the read set.
        n_historical = max(0, n_snapshots_behind - 1) * n_partitions
        return ConflictCost(
            historical_ml_reads=n_historical,
        )
