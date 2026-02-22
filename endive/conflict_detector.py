"""Conflict detection per SPEC.md §5.

Conflict detection is encapsulated and opaque. The detector determines
if conflicts are "real" (data overlap requiring abort) or "false"
(concurrent commits to unrelated data that can be merged).

Implementations:
- ProbabilisticConflictDetector: Returns real conflict with configured
  probability. Simple model for parameter sweeps.
- PartitionOverlapConflictDetector: Checks per-(table, partition) version
  changes between start and current snapshots. Real conflict if a written
  partition was modified by a concurrent transaction.

Both respect txn.can_have_real_conflict(): FastAppend and MergeAppend
always return False (they cannot detect conflicts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from endive.catalog import CatalogSnapshot
from endive.transaction import ConflictDetector

if TYPE_CHECKING:
    from endive.transaction import Transaction


class ProbabilisticConflictDetector(ConflictDetector):
    """Probabilistic conflict detection.

    Returns real conflict with configured probability for transactions
    that can_have_real_conflict(). FastAppend and MergeAppend always
    return False regardless of probability.

    Uses a seeded RNG for deterministic simulation results.
    """

    def __init__(
        self,
        real_conflict_probability: float,
        rng: np.random.RandomState | None = None,
    ):
        if not 0.0 <= real_conflict_probability <= 1.0:
            raise ValueError(
                f"real_conflict_probability must be in [0, 1], "
                f"got {real_conflict_probability}"
            )
        self._probability = real_conflict_probability
        self._rng = rng if rng is not None else np.random.RandomState()

    @property
    def probability(self) -> float:
        return self._probability

    def is_real_conflict(
        self,
        txn: Transaction,
        current_snapshot: CatalogSnapshot,
        start_snapshot: CatalogSnapshot,
    ) -> bool:
        if not txn.can_have_real_conflict():
            return False
        return float(self._rng.random()) < self._probability


class PartitionOverlapConflictDetector(ConflictDetector):
    """Partition-based conflict detection.

    Returns real conflict if any (table, partition) pair written by the
    transaction was also modified by a concurrent transaction (detected
    by comparing partition versions between start and current snapshots).

    Only returns True for transactions where can_have_real_conflict() is
    True (i.e., ValidatedOverwriteTransaction).
    """

    def is_real_conflict(
        self,
        txn: Transaction,
        current_snapshot: CatalogSnapshot,
        start_snapshot: CatalogSnapshot,
    ) -> bool:
        if not txn.can_have_real_conflict():
            return False

        for table_id, partitions in txn.partitions_written.items():
            for partition_id in partitions:
                start_ver = start_snapshot.get_partition_version(
                    table_id, partition_id
                )
                current_ver = current_snapshot.get_partition_version(
                    table_id, partition_id
                )
                if current_ver != start_ver:
                    return True  # Partition was modified by concurrent txn

        return False
