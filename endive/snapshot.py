"""Immutable snapshot types for catalog state.

This module defines the core data types for message-passing between
the catalog (server) and transactions (client). All types are frozen
dataclasses to ensure immutability.

Design principle: Transactions NEVER access Catalog state directly.
All state is obtained via catalog.read() or catalog.try_cas() which
return immutable snapshots.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogSnapshot:
    """Immutable snapshot of catalog state at a specific time.

    Transactions receive snapshots via message-passing (catalog.read() or CASResult).
    All catalog state access goes through snapshots, ensuring proper distributed semantics:
    - State is captured at a specific point in time
    - Cannot accidentally read stale or future state
    - All reads have explicit latency costs

    Attributes:
        seq: Global sequence number (increments on each commit)
        tbl: Frozen tuple of per-table version numbers
        partition_seq: Frozen 2D tuple of per-partition versions (or None if partitions disabled)
        ml_offset: Frozen tuple of manifest list byte offsets per table
        partition_ml_offset: Frozen 2D tuple of partition ML offsets (or None if partitions disabled)
        timestamp: Simulation time when snapshot was taken
    """
    seq: int
    tbl: tuple[int, ...]                              # Frozen table versions
    partition_seq: tuple[tuple[int, ...], ...] | None  # Frozen partition versions (or None if disabled)
    ml_offset: tuple[int, ...]                        # Frozen ML offsets per table
    partition_ml_offset: tuple[tuple[int, ...], ...] | None  # Frozen partition ML offsets (or None)
    timestamp: int  # Simulation time when snapshot was taken


@dataclass(frozen=True)
class CASResult:
    """Result of a CAS operation.

    Contains the success/failure status and a snapshot of catalog state
    at server-processing time. On failure, this snapshot is used for
    conflict resolution without additional round-trips.

    Attributes:
        success: True if CAS succeeded, False if conflict detected
        snapshot: Catalog state at server processing time (before mutation on success)
    """
    success: bool
    snapshot: CatalogSnapshot  # State at server processing time
