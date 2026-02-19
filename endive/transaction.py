"""Transaction state and log entry types.

This module defines the client-side state for transactions in the simulation.
Transactions hold CatalogSnapshots obtained via catalog.read() or catalog.try_cas().
All catalog state access uses these immutable snapshots, not direct references.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from endive.snapshot import CatalogSnapshot
    from endive.operation import OperationType, OperationBehavior


@dataclass
class Txn:
    """Transaction state for the simulation.

    Represents a single transaction from creation through commit/abort.
    Holds immutable snapshots from catalog reads to ensure proper
    message-passing semantics.

    Attributes:
        id: Unique transaction identifier
        t_submit: Simulation time when transaction was submitted (ms)
        t_runtime: Transaction execution duration (ms)
        v_catalog_seq: Catalog sequence number when snapshot was taken
        v_tblr: Table versions read {table_id: version}
        v_tblw: Table versions to write {table_id: version}
        n_retries: Number of retry attempts after CAS failures
        t_commit: Simulation time of successful commit (-1 if not committed)
        t_abort: Simulation time of abort (-1 if not aborted)
        v_dirty: Versions being validated (union of read/write)
        v_log_offset: Log offset for append mode
        v_ml_offset: Per-table manifest list offsets
        partitions_read: Partitions read per table (partition mode)
        partitions_written: Partitions written per table (partition mode)
        v_partition_seq: Per-partition versions snapshot
        v_partition_ml_offset: Per-partition ML offsets snapshot
        start_snapshot: Initial catalog snapshot from read()
        current_snapshot: Latest snapshot (updated on CAS failure)
    """
    id: int
    t_submit: int  # ms submitted since start
    t_runtime: int  # ms between submission and commit
    v_catalog_seq: int  # version of catalog read (CAS in storage)
    v_tblr: dict[int, int]  # versions of tables read
    v_tblw: dict[int, int]  # versions of tables written
    n_retries: int = 0  # number of retries
    t_commit: int = field(default=-1)
    t_abort: int = field(default=-1)
    v_dirty: dict[int, int] = field(default_factory=lambda: defaultdict(dict))  # versions validated
    # Append mode fields
    v_log_offset: int = 0  # Log offset when snapshot was taken (for append mode)
    v_ml_offset: dict[int, int] = field(default_factory=dict)  # Per-table manifest list offsets
    # Partition mode fields (when PARTITION_ENABLED)
    # Indexed by table_id -> set of partition_ids or table_id -> partition_id -> value
    partitions_read: dict[int, set[int]] = field(default_factory=dict)  # table_id -> partition_ids read
    partitions_written: dict[int, set[int]] = field(default_factory=dict)  # table_id -> partition_ids written
    v_partition_seq: dict[int, dict[int, int]] = field(default_factory=dict)  # table_id -> partition_id -> version
    v_partition_ml_offset: dict[int, dict[int, int]] = field(default_factory=dict)  # table_id -> partition_id -> offset
    # Snapshot-based fields (for message-passing semantics)
    start_snapshot: 'CatalogSnapshot | None' = None  # Initial catalog state from read()
    current_snapshot: 'CatalogSnapshot | None' = None  # Latest snapshot (updated on CAS failure)
    # Operation type (affects conflict resolution cost)
    operation_type: 'OperationType | None' = None  # FastAppend, MergeAppend, or ValidatedOverwrite
    # Abort tracking (for ValidationException on real conflicts)
    abort_reason: str | None = None  # None if not aborted, or reason string (e.g., "validation_exception")

    def get_behavior(self, manifests_per_commit: float = 1.0) -> 'OperationBehavior':
        """Get the conflict resolution behavior for this transaction's operation type.

        Args:
            manifests_per_commit: For MergeAppend, manifests per concurrent commit

        Returns:
            OperationBehavior implementation for this transaction's operation type.
            Defaults to FastAppend if operation_type is not set (backward compatibility).
        """
        from endive.operation import OperationType, get_behavior, FAST_APPEND_BEHAVIOR

        if self.operation_type is None:
            # Default to FastAppend for backward compatibility
            return FAST_APPEND_BEHAVIOR
        return get_behavior(self.operation_type, manifests_per_commit)


@dataclass
class LogEntry:
    """Log entry for append-based catalog operations.

    Each entry represents a committed transaction's effect on the catalog.
    Used for conflict detection and compaction in append mode.

    Attributes:
        txn_id: Transaction ID (used for deduplication)
        tables_written: Table versions after this transaction {table_id: new_version}
        tables_read: Table versions read by this transaction {table_id: version}
        sealed: True if this entry triggers compaction
    """
    txn_id: int  # Transaction ID (used for deduplication)
    tables_written: dict[int, int]  # table_id -> new_version after this txn
    tables_read: dict[int, int]  # table_id -> version_read by this txn
    sealed: bool = False  # True if this entry triggers compaction
