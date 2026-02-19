"""Operation types and conflict cost models for Iceberg transactions.

This module models the three distinct operation types in Iceberg, each with
different conflict resolution costs:

1. FastAppend: No validation, cheapest retry (~160ms). Appends are additive
   and cannot conflict with each other.

2. MergeAppend: No validation, but must re-merge manifests against new base.
   More expensive than FastAppend due to manifest file I/O.

3. ValidatedOverwrite: Full validation via validationHistory(). Reads O(N)
   manifest lists for N missed snapshots. Real conflicts abort with
   ValidationException (not retried by Iceberg).

References:
- SIMULATOR_REVIEW.md Section 2.1: Operation type costs
- SIMULATOR_REVIEW.md Section 2.2: Real conflicts abort
- Iceberg source: SnapshotProducer.java, MergingSnapshotProducer.java
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class OperationType(Enum):
    """Iceberg operation types with distinct conflict costs.

    FAST_APPEND: No validation. Appends are additive - cannot conflict.
                 Retry cost: ~160ms (1 metadata read + 1 ML read + 1 ML write)

    MERGE_APPEND: No validation, but must re-merge manifests against new base.
                  Retry cost: ~160ms + manifest file I/O for merging

    VALIDATED_OVERWRITE: Full validation via validationHistory().
                         Retry cost: O(N) ML reads for N missed snapshots.
                         Real conflicts abort with ValidationException.
    """
    FAST_APPEND = "fast_append"
    MERGE_APPEND = "merge_append"
    VALIDATED_OVERWRITE = "validated_overwrite"


@dataclass
class ConflictCost:
    """Cost model for a conflict resolution attempt.

    All counts are number of I/O operations required.
    Actual latency is computed by the simulator based on configured distributions.

    Attributes:
        metadata_reads: Number of metadata file reads (typically 1)
        ml_reads: Manifest list reads for current snapshot (typically 1)
        ml_writes: Manifest list writes (1 for rewrite mode, 0 for ML+)
        historical_ml_reads: ML reads for validation history (O(N) for validated ops)
        manifest_file_reads: Manifest file reads for merge/validation
        manifest_file_writes: Manifest file writes for merge
    """
    metadata_reads: int = 1
    ml_reads: int = 1
    ml_writes: int = 1
    historical_ml_reads: int = 0
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0

    def total_ml_reads(self) -> int:
        """Total manifest list reads including history."""
        return self.ml_reads + self.historical_ml_reads


class OperationBehavior(ABC):
    """Strategy pattern for operation-specific conflict handling.

    Each operation type has a different cost model for conflict resolution.
    FastAppend is cheapest, ValidatedOverwrite is most expensive.
    """

    @abstractmethod
    def get_false_conflict_cost(
        self,
        n_behind: int,
        ml_append_mode: bool = False
    ) -> ConflictCost:
        """Cost when CAS fails but validation passes (or no validation needed).

        Args:
            n_behind: Number of snapshots behind catalog (for historical ML reads)
            ml_append_mode: True if manifest_list_mode="append" (ML+ mode)

        Returns:
            ConflictCost with I/O operation counts
        """
        pass

    @abstractmethod
    def can_have_real_conflict(self) -> bool:
        """Whether this operation type can detect real conflicts.

        Only validated operations (ValidatedOverwrite) can detect real conflicts.
        FastAppend and MergeAppend have no validation, so they don't detect
        conflicts at all - their changes are purely additive.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable name for logging."""
        pass


class FastAppendBehavior(OperationBehavior):
    """No validation. Appends are additive - cannot conflict.

    FastAppend is the cheapest operation type. It simply appends new data
    files without validation. Since appends don't modify or delete existing
    data, they cannot conflict with concurrent operations.

    Retry cost: ~160ms (metadata read + ML read + ML write)
    The I/O convoy (O(N) historical ML reads) does NOT apply.
    """

    def get_false_conflict_cost(
        self,
        n_behind: int,
        ml_append_mode: bool = False
    ) -> ConflictCost:
        """FastAppend retry cost is constant, regardless of n_behind.

        No historical ML reads needed - appends don't validate against history.
        """
        return ConflictCost(
            metadata_reads=1,
            ml_reads=1,  # Just current snapshot's ML
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=0,  # NO validation history
            manifest_file_reads=0,
            manifest_file_writes=0,
        )

    def can_have_real_conflict(self) -> bool:
        """Appends never conflict - they're purely additive."""
        return False

    def get_name(self) -> str:
        return "FastAppend"


class MergeAppendBehavior(OperationBehavior):
    """No validation, but must re-merge manifests against new base.

    MergeAppend creates compacted manifests by merging small manifest files.
    On retry, it must re-merge against the new snapshot's manifest structure.
    Still no validation, so no real conflicts detected.

    Retry cost: ~160ms + manifest file I/O for re-merging
    The I/O convoy does NOT apply (no validation).
    """

    def __init__(self, manifests_per_commit: float = 1.0):
        """Initialize with expected manifests to merge per concurrent commit.

        Args:
            manifests_per_commit: Average manifests added per commit (for merge sizing)
        """
        self.manifests_per_commit = manifests_per_commit

    def get_false_conflict_cost(
        self,
        n_behind: int,
        ml_append_mode: bool = False
    ) -> ConflictCost:
        """MergeAppend must re-merge manifests for commits that occurred.

        The number of manifest files to read/write scales with n_behind.
        """
        # Estimate manifests to merge: ~1 per concurrent commit
        k_manifests = max(1, int(n_behind * self.manifests_per_commit))

        return ConflictCost(
            metadata_reads=1,
            ml_reads=1,
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=0,  # NO validation
            manifest_file_reads=k_manifests,
            manifest_file_writes=k_manifests,
        )

    def can_have_real_conflict(self) -> bool:
        """No validation means no detected conflicts."""
        return False

    def get_name(self) -> str:
        return "MergeAppend"


class ValidatedOverwriteBehavior(OperationBehavior):
    """Full validation via validationHistory(). Real conflicts abort.

    ValidatedOverwrite is used for operations that modify or delete data:
    - Overwrite (REPLACE)
    - Delete
    - RowDelta (merge-on-read deletes)

    These operations MUST validate that the data they're modifying hasn't
    changed since the transaction started. This requires reading O(N)
    manifest lists for N missed snapshots (the I/O convoy).

    If validation finds overlapping changes (real conflict), Iceberg throws
    ValidationException and the operation aborts - it is NOT retried.

    Retry cost: O(N) ML reads for validation history
    Real conflicts: ABORT (ValidationException), not merge-and-retry
    """

    def get_false_conflict_cost(
        self,
        n_behind: int,
        ml_append_mode: bool = False
    ) -> ConflictCost:
        """ValidatedOverwrite must read N manifest lists for validation.

        This is the I/O convoy: O(N) reads for N missed snapshots.
        """
        return ConflictCost(
            metadata_reads=1,
            ml_reads=1,
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=n_behind,  # THE I/O CONVOY
            manifest_file_reads=0,  # Only if real conflict (but those abort)
            manifest_file_writes=0,
        )

    def can_have_real_conflict(self) -> bool:
        """Validated operations can detect partition overlap conflicts."""
        return True

    def get_name(self) -> str:
        return "ValidatedOverwrite"


# Singleton instances for common use
FAST_APPEND_BEHAVIOR = FastAppendBehavior()
VALIDATED_OVERWRITE_BEHAVIOR = ValidatedOverwriteBehavior()


def get_behavior(
    op_type: OperationType,
    manifests_per_commit: float = 1.0
) -> OperationBehavior:
    """Get the behavior implementation for an operation type.

    Args:
        op_type: The operation type
        manifests_per_commit: For MergeAppend, manifests per concurrent commit

    Returns:
        OperationBehavior implementation
    """
    if op_type == OperationType.FAST_APPEND:
        return FAST_APPEND_BEHAVIOR
    elif op_type == OperationType.MERGE_APPEND:
        return MergeAppendBehavior(manifests_per_commit)
    elif op_type == OperationType.VALIDATED_OVERWRITE:
        return VALIDATED_OVERWRITE_BEHAVIOR
    else:
        raise ValueError(f"Unknown operation type: {op_type}")
