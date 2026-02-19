"""Conflict resolution for Iceberg transactions.

This module implements accurate conflict resolution based on operation type:

1. FastAppend: No validation, cannot have conflicts. Cheapest retry (~160ms).
2. MergeAppend: No validation, but must re-merge manifests. No I/O convoy.
3. ValidatedOverwrite: Full validation via validationHistory(). Real conflicts
   ABORT with ValidationException - they are NOT retried.

Key corrections from SIMULATOR_REVIEW.md:
- Real conflicts abort, they don't merge-and-retry
- I/O convoy (O(N) ML reads) only affects validated operations
- FastAppend retries are ~160ms, not 28 seconds

References:
- SIMULATOR_REVIEW.md Section 2.1-2.3: Operation costs and abort behavior
- Iceberg source: SnapshotProducer.java, MergingSnapshotProducer.java
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Generator
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from endive.snapshot import CatalogSnapshot
    from endive.transaction import Txn
    from endive.operation import ConflictCost, OperationBehavior

logger = logging.getLogger(__name__)


@dataclass
class ConflictResult:
    """Result of conflict resolution.

    Attributes:
        should_retry: True if transaction should retry, False if it should abort
        abort_reason: Reason string if aborting (e.g., "validation_exception"), None otherwise
    """
    should_retry: bool
    abort_reason: str | None = None


def resolve_conflict(
    sim,
    txn: 'Txn',
    snapshot: 'CatalogSnapshot',
    real_conflict_probability: float,
    manifest_list_mode: str = "rewrite",
    manifests_per_commit: float = 1.0,
    stats=None,
) -> Generator:
    """Resolve a CAS conflict based on operation type.

    This is the main entry point for conflict resolution. It:
    1. Gets the transaction's operation behavior
    2. For validated operations, checks if there's a real conflict
    3. If real conflict on validated op: ABORT (ValidationException)
    4. Otherwise: pay false conflict cost and retry

    Args:
        sim: SimPy environment
        txn: Transaction that failed CAS
        snapshot: Catalog state at CAS failure time (from CASResult)
        real_conflict_probability: Probability of real conflict on validated ops
        manifest_list_mode: "rewrite" or "append" (ML+ mode)
        manifests_per_commit: For MergeAppend, manifests per concurrent commit
        stats: Statistics collector (STATS global)

    Yields:
        SimPy timeout events for I/O operations

    Returns:
        ConflictResult with should_retry and abort_reason
    """
    # Import here to avoid circular dependencies
    from endive.main import (
        get_metadata_root_latency,
        get_manifest_list_latency,
        get_manifest_file_latency,
        get_put_latency,
        get_manifest_list_write_latency,
        MAX_PARALLEL,
        T_PUT,
        TABLE_METADATA_INLINED,
        get_table_metadata_latency,
        STATS,
    )

    if stats is None:
        stats = STATS

    behavior = txn.get_behavior(manifests_per_commit)
    n_behind = snapshot.seq - txn.v_catalog_seq
    ml_append_mode = manifest_list_mode == "append"

    logger.debug(f"{sim.now} TXN {txn.id} Conflict resolution: {behavior.get_name()}, "
                f"n_behind={n_behind}, ml_append_mode={ml_append_mode}")

    # 1. For validated operations, check for real conflicts
    if behavior.can_have_real_conflict():
        # Determine if this is a real conflict (partition overlap)
        # In a real system, this would be detected during validation
        is_real_conflict = np.random.random() < real_conflict_probability

        if is_real_conflict:
            # Real conflict on validated operation: ABORT with ValidationException
            # Iceberg does NOT retry real conflicts - it throws and the caller must handle
            logger.debug(f"{sim.now} TXN {txn.id} Real conflict detected - ValidationException (abort)")
            stats.real_conflicts += 1
            stats.validation_exceptions += 1
            return ConflictResult(should_retry=False, abort_reason="validation_exception")

    # 2. No real conflict (or operation doesn't validate) - pay false conflict cost
    cost = behavior.get_false_conflict_cost(n_behind, ml_append_mode)

    logger.debug(f"{sim.now} TXN {txn.id} False conflict cost: metadata_reads={cost.metadata_reads}, "
                f"ml_reads={cost.ml_reads}, historical_ml_reads={cost.historical_ml_reads}, "
                f"ml_writes={cost.ml_writes}, mf_reads={cost.manifest_file_reads}, "
                f"mf_writes={cost.manifest_file_writes}")

    # Pay metadata read cost
    for _ in range(cost.metadata_reads):
        yield sim.timeout(get_metadata_root_latency('read'))

    # Table metadata read if not inlined
    if not TABLE_METADATA_INLINED:
        yield sim.timeout(get_table_metadata_latency('read'))

    # Pay historical ML read cost (the I/O convoy for validated operations)
    if cost.historical_ml_reads > 0:
        yield from _read_manifest_lists_batched(
            sim, cost.historical_ml_reads, txn.id, stats, MAX_PARALLEL,
            get_manifest_list_latency, get_put_latency, T_PUT
        )

    # Pay current ML read cost
    if cost.ml_reads > 0:
        for _ in range(cost.ml_reads):
            yield sim.timeout(get_manifest_list_latency('read'))
            stats.manifest_list_reads += 1

    # Pay manifest file read/write cost (for MergeAppend)
    if cost.manifest_file_reads > 0:
        yield from _read_manifest_files_batched(
            sim, cost.manifest_file_reads, txn.id, stats, MAX_PARALLEL,
            get_manifest_file_latency
        )

    if cost.manifest_file_writes > 0:
        yield from _write_manifest_files_batched(
            sim, cost.manifest_file_writes, txn.id, stats, MAX_PARALLEL,
            get_manifest_file_latency
        )

    # Pay ML write cost
    if cost.ml_writes > 0:
        for _ in range(cost.ml_writes):
            yield sim.timeout(get_manifest_list_latency('write'))
            stats.manifest_list_writes += 1

    # Table metadata write if not inlined
    if not TABLE_METADATA_INLINED:
        yield sim.timeout(get_table_metadata_latency('write'))

    # Track false conflict
    stats.false_conflicts += 1

    # Update transaction state for retry
    txn.v_catalog_seq = snapshot.seq
    _update_txn_dirty_versions(txn, snapshot)

    return ConflictResult(should_retry=True, abort_reason=None)


def _read_manifest_lists_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_ml_latency, get_put_latency, t_put
) -> Generator:
    """Read manifest lists in batches respecting MAX_PARALLEL."""
    if count <= 0:
        return

    logger.debug(f"{sim.now} TXN {txn_id} Reading {count} historical manifest lists")

    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_ml_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    stats.manifest_list_reads += count


def _read_manifest_files_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_mf_latency
) -> Generator:
    """Read manifest files in batches respecting MAX_PARALLEL."""
    if count <= 0:
        return

    logger.debug(f"{sim.now} TXN {txn_id} Reading {count} manifest files")

    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_mf_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    stats.manifest_files_read += count


def _write_manifest_files_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_mf_latency
) -> Generator:
    """Write manifest files in batches respecting MAX_PARALLEL."""
    if count <= 0:
        return

    logger.debug(f"{sim.now} TXN {txn_id} Writing {count} manifest files")

    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_mf_latency('write') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    stats.manifest_files_written += count


def _update_txn_dirty_versions(txn: 'Txn', snapshot: 'CatalogSnapshot') -> None:
    """Update transaction's dirty versions from snapshot for retry."""
    for t in txn.v_dirty.keys():
        txn.v_dirty[t] = snapshot.tbl[t]


# Legacy compatibility: ConflictResolver class wrapper
class ConflictResolverV2:
    """New conflict resolver using operation-type-aware conflict handling.

    This replaces the old ConflictResolver when operation types are configured.
    The main difference is that real conflicts on ValidatedOverwrite operations
    cause immediate abort (ValidationException) rather than merge-and-retry.
    """

    @staticmethod
    def resolve(
        sim,
        txn: 'Txn',
        snapshot: 'CatalogSnapshot',
        real_conflict_probability: float,
        manifest_list_mode: str = "rewrite",
        manifests_per_commit: float = 1.0,
        stats=None,
    ) -> Generator:
        """Resolve conflict and return result.

        Yields I/O events, then returns ConflictResult.
        """
        result = yield from resolve_conflict(
            sim, txn, snapshot, real_conflict_probability,
            manifest_list_mode, manifests_per_commit, stats
        )
        return result

    @staticmethod
    def should_use_operation_aware_resolution(txn: 'Txn') -> bool:
        """Check if transaction has operation type set (use new resolver)."""
        return txn.operation_type is not None
