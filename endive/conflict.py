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

Partition-Aware Conflict Resolution:
- Partition overlap determines which partitions need ML read/write work
- Data overlap within a partition determines abort vs merge (for validated ops)
- Real conflicts discovered AFTER paying ML read cost for that partition

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


def resolve_partition_conflict(
    sim,
    txn: 'Txn',
    snapshot: 'CatalogSnapshot',
    partition_seq_snapshot: dict,
    data_overlap_probability: float,
    manifest_list_mode: str = "rewrite",
    manifests_per_commit: float = 1.0,
    stats=None,
) -> Generator:
    """Resolve CAS conflict with partition-aware costs.

    This implements accurate conflict resolution where:
    1. Partition overlap determines which partitions need ML read/write work
    2. For ValidatedOverwrite: check data overlap per overlapping partition
       - Real conflict (data overlap): pay ML read cost THEN abort
       - False conflict (no data overlap): pay ML read + write cost (merge)
    3. For FastAppend/MergeAppend: no validation, so all conflicts are "false"
       (partition overlap requires merge, but no abort possible)

    Key insight from user clarification:
    - Real conflicts are discovered AFTER reading the ML for that partition
    - We must pay the ML read cost before we can abort

    Args:
        sim: SimPy environment
        txn: Transaction that failed CAS
        snapshot: Catalog state at CAS failure time (from CASResult)
        partition_seq_snapshot: Partition versions at server time {table_id: {part_id: version}}
        data_overlap_probability: Probability of data overlap (real conflict) per partition
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
        sample_conflicting_manifests,
        STATS,
    )

    if stats is None:
        stats = STATS

    behavior = txn.get_behavior(manifests_per_commit)
    ml_append_mode = manifest_list_mode == "append"
    can_have_real_conflict = behavior.can_have_real_conflict()

    # 1. Identify overlapping partitions (partition version changed since txn read)
    overlapping_partitions = _compute_overlapping_partitions(txn, partition_seq_snapshot)

    if not overlapping_partitions:
        # No partition conflicts - just update state and retry
        txn.v_catalog_seq = snapshot.seq
        _update_txn_dirty_versions(txn, snapshot)
        return ConflictResult(should_retry=True, abort_reason=None)

    total_overlapping = sum(len(parts) for parts in overlapping_partitions.values())
    logger.debug(f"{sim.now} TXN {txn.id} Partition conflict resolution: "
                f"{total_overlapping} overlapping partitions, behavior={behavior.get_name()}")

    # 2. Process each overlapping partition
    for tbl_id, partitions in overlapping_partitions.items():
        for part_id in partitions:
            # Calculate how many snapshots behind for this partition
            snapshot_v = partition_seq_snapshot.get(tbl_id, {}).get(part_id, 0)
            txn_v = txn.v_partition_seq.get(tbl_id, {}).get(part_id, 0)
            n_behind = snapshot_v - txn_v

            # Pay ML read cost for this partition (discovering the conflict)
            # This must happen BEFORE we know if it's a real conflict
            if n_behind > 0:
                yield from _read_partition_manifest_lists(
                    sim, n_behind, txn.id, tbl_id, part_id, stats, MAX_PARALLEL,
                    get_manifest_list_latency, get_put_latency, T_PUT
                )

            # Now we've read the ML - determine if this is a real conflict (data overlap)
            if can_have_real_conflict:
                is_real_conflict = np.random.random() < data_overlap_probability

                if is_real_conflict:
                    # Real conflict on validated operation - abort with ValidationException
                    # We've already paid the ML read cost (correct per user clarification)
                    logger.debug(f"{sim.now} TXN {txn.id} Real conflict at table {tbl_id} "
                                f"partition {part_id} - ValidationException (abort)")
                    stats.real_conflicts += 1
                    stats.validation_exceptions += 1
                    return ConflictResult(should_retry=False, abort_reason="validation_exception")

            # False conflict (no data overlap, or operation doesn't validate)
            # Pay merge costs: ML write (rewrite mode) or manifest file ops (merge_append)
            yield from _resolve_partition_false_conflict(
                sim, txn, tbl_id, part_id, behavior, ml_append_mode,
                manifests_per_commit, n_behind, stats, MAX_PARALLEL,
                get_manifest_list_latency, get_manifest_file_latency,
                get_put_latency, T_PUT
            )
            stats.false_conflicts += 1

    # 3. Update transaction state for retry
    txn.v_catalog_seq = snapshot.seq
    _update_txn_dirty_versions(txn, snapshot)
    _update_txn_partition_versions(txn, snapshot, overlapping_partitions)

    return ConflictResult(should_retry=True, abort_reason=None)


def _compute_overlapping_partitions(
    txn: 'Txn',
    partition_seq_snapshot: dict,
) -> dict[int, set[int]]:
    """Compute partitions where version changed since transaction read.

    An overlapping partition is one where:
    - Transaction read or wrote to it
    - Partition version in snapshot differs from transaction's recorded version

    Args:
        txn: Transaction object
        partition_seq_snapshot: Partition versions at server time {table_id: {part_id: version}}

    Returns:
        Dict of {table_id: set of partition_ids} with version conflicts
    """
    overlapping = {}

    # Check partitions read
    for tbl_id, partitions in txn.partitions_read.items():
        for p in partitions:
            snapshot_v = partition_seq_snapshot.get(tbl_id, {}).get(p, 0)
            txn_v = txn.v_partition_seq.get(tbl_id, {}).get(p, 0)
            if snapshot_v != txn_v:
                if tbl_id not in overlapping:
                    overlapping[tbl_id] = set()
                overlapping[tbl_id].add(p)

    # Check partitions written
    for tbl_id, partitions in txn.partitions_written.items():
        for p in partitions:
            snapshot_v = partition_seq_snapshot.get(tbl_id, {}).get(p, 0)
            txn_v = txn.v_partition_seq.get(tbl_id, {}).get(p, 0)
            if snapshot_v != txn_v:
                if tbl_id not in overlapping:
                    overlapping[tbl_id] = set()
                overlapping[tbl_id].add(p)

    return overlapping


def _read_partition_manifest_lists(
    sim, count: int, txn_id: int, table_id: int, partition_id: int,
    stats, max_parallel: int, get_ml_latency, get_put_latency, t_put
) -> Generator:
    """Read manifest lists for a partition's history.

    This is the I/O cost paid when discovering conflicts - happens BEFORE
    we know if it's a real or false conflict.
    """
    if count <= 0:
        return

    logger.debug(f"{sim.now} TXN {txn_id} Reading {count} MLs for table {table_id} "
                f"partition {partition_id}")

    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_ml_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    stats.manifest_list_reads += count


def _resolve_partition_false_conflict(
    sim, txn: 'Txn', table_id: int, partition_id: int,
    behavior: 'OperationBehavior', ml_append_mode: bool,
    manifests_per_commit: float, n_behind: int,
    stats, max_parallel: int,
    get_ml_latency, get_mf_latency, get_put_latency, t_put
) -> Generator:
    """Resolve false conflict for a partition (version changed, no data overlap).

    Cost depends on operation type:
    - FastAppend: Just ML write (rewrite mode) or nothing (ML+ mode)
    - MergeAppend: ML write + manifest file re-merge
    - ValidatedOverwrite: ML write (already paid historical ML reads)
    """
    from endive.main import sample_conflicting_manifests

    logger.debug(f"{sim.now} TXN {txn.id} False conflict table {table_id} "
                f"partition {partition_id}, behavior={behavior.get_name()}")

    if ml_append_mode:
        # ML+ mode: Tentative entry still valid, no ML update needed
        logger.debug(f"{sim.now} TXN {txn.id} ML+ mode: partition entry still valid")
    else:
        # Rewrite mode: Write new ML for this partition
        yield sim.timeout(get_ml_latency('write'))
        stats.manifest_list_writes += 1

    # MergeAppend requires manifest file re-merge
    if behavior.get_name() == "merge_append" and n_behind > 0:
        k_manifests = int(n_behind * manifests_per_commit)
        if k_manifests > 0:
            # Read manifest files
            for batch_start in range(0, k_manifests, max_parallel):
                batch_size = min(max_parallel, k_manifests - batch_start)
                batch_latencies = [get_mf_latency('read') for _ in range(batch_size)]
                yield sim.timeout(max(batch_latencies))
            stats.manifest_files_read += k_manifests

            # Write merged manifest files
            for batch_start in range(0, k_manifests, max_parallel):
                batch_size = min(max_parallel, k_manifests - batch_start)
                batch_latencies = [get_mf_latency('write') for _ in range(batch_size)]
                yield sim.timeout(max(batch_latencies))
            stats.manifest_files_written += k_manifests


def _update_txn_partition_versions(
    txn: 'Txn',
    snapshot: 'CatalogSnapshot',
    overlapping_partitions: dict[int, set[int]]
) -> None:
    """Update transaction's partition versions from snapshot for retry."""
    if snapshot.partition_seq is None:
        return

    for tbl_id, partitions in overlapping_partitions.items():
        if tbl_id not in txn.v_partition_seq:
            txn.v_partition_seq[tbl_id] = {}
        for p in partitions:
            txn.v_partition_seq[tbl_id][p] = snapshot.partition_seq[tbl_id][p]

        # Also update ML offsets if available
        if snapshot.partition_ml_offset is not None:
            if tbl_id not in txn.v_partition_ml_offset:
                txn.v_partition_ml_offset[tbl_id] = {}
            for p in partitions:
                txn.v_partition_ml_offset[tbl_id][p] = snapshot.partition_ml_offset[tbl_id][p]


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
