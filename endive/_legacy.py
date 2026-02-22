"""Legacy types used by endive/main.py during migration.

This module consolidates old types from the pre-rewrite codebase:
- CatalogSnapshot, CASResult (from old snapshot.py)
- StorageProvider, ObjectStorageProvider, StorageConfig, STORAGE_CONFIGS (from old storage_provider.py)
- CatalogProvider, CatalogConfig, CATALOG_CONFIGS, InstantCatalog, ObjectStorageCatalog (from old catalog_provider.py)
- OperationType, ConflictCost, OperationBehavior, behaviors (from old operation.py)
- ConflictResult, resolve_conflict, ConflictResolverV2 (from old conflict.py)
- Txn, LogEntry (from old transaction.py legacy section)

All of this will be deleted when main.py is rewritten to use the new
Transaction/Catalog/StorageProvider interfaces.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Generator

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# snapshot.py → CatalogSnapshot, CASResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogSnapshot:
    """Immutable snapshot of catalog state at a specific time."""
    seq: int
    tbl: tuple[int, ...]
    partition_seq: tuple[tuple[int, ...], ...] | None
    ml_offset: tuple[int, ...]
    partition_ml_offset: tuple[tuple[int, ...], ...] | None
    timestamp: int


@dataclass(frozen=True)
class CASResult:
    """Result of a CAS operation."""
    success: bool
    snapshot: CatalogSnapshot


# ---------------------------------------------------------------------------
# storage_provider.py → StorageConfig, STORAGE_CONFIGS, StorageProvider, etc.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    """Configuration for a storage provider. All latencies in milliseconds."""
    ml_read_median: float
    ml_read_sigma: float
    ml_write_median: float
    ml_write_sigma: float
    mf_read_median: float
    mf_read_sigma: float
    mf_write_median: float
    mf_write_sigma: float
    put_base_latency: float
    put_per_mib: float
    put_sigma: float
    min_latency: float


STORAGE_CONFIGS = {
    "instant": StorageConfig(
        ml_read_median=1.0, ml_read_sigma=0.1,
        ml_write_median=1.0, ml_write_sigma=0.1,
        mf_read_median=1.0, mf_read_sigma=0.1,
        mf_write_median=1.0, mf_write_sigma=0.1,
        put_base_latency=0.5, put_per_mib=0.1, put_sigma=0.1,
        min_latency=1.0,
    ),
    "s3x": StorageConfig(
        ml_read_median=22.0, ml_read_sigma=0.5,
        ml_write_median=21.0, ml_write_sigma=0.5,
        mf_read_median=22.0, mf_read_sigma=0.5,
        mf_write_median=21.0, mf_write_sigma=0.5,
        put_base_latency=10.0, put_per_mib=10.0, put_sigma=0.3,
        min_latency=10.0,
    ),
    "s3": StorageConfig(
        ml_read_median=61.0, ml_read_sigma=0.5,
        ml_write_median=63.0, ml_write_sigma=0.5,
        mf_read_median=61.0, mf_read_sigma=0.5,
        mf_write_median=63.0, mf_write_sigma=0.5,
        put_base_latency=30.0, put_per_mib=20.0, put_sigma=0.3,
        min_latency=43.0,
    ),
    "azurex": StorageConfig(
        ml_read_median=64.0, ml_read_sigma=0.5,
        ml_write_median=70.0, ml_write_sigma=0.5,
        mf_read_median=64.0, mf_read_sigma=0.5,
        mf_write_median=70.0, mf_write_sigma=0.5,
        put_base_latency=30.0, put_per_mib=15.0, put_sigma=0.3,
        min_latency=40.0,
    ),
    "azure": StorageConfig(
        ml_read_median=87.0, ml_read_sigma=0.5,
        ml_write_median=95.0, ml_write_sigma=0.5,
        mf_read_median=87.0, mf_read_sigma=0.5,
        mf_write_median=95.0, mf_write_sigma=0.5,
        put_base_latency=50.0, put_per_mib=25.0, put_sigma=0.3,
        min_latency=51.0,
    ),
    "gcp": StorageConfig(
        ml_read_median=170.0, ml_read_sigma=0.5,
        ml_write_median=180.0, ml_write_sigma=0.5,
        mf_read_median=170.0, mf_read_sigma=0.5,
        mf_write_median=180.0, mf_write_sigma=0.5,
        put_base_latency=40.0, put_per_mib=17.0, put_sigma=0.3,
        min_latency=118.0,
    ),
}
STORAGE_CONFIGS["aws"] = STORAGE_CONFIGS["s3x"]


class StorageProvider(ABC):
    """Abstract interface for storage operations."""

    @abstractmethod
    def get_manifest_list_read_latency(self, size_kib: float = 10.0) -> float: ...

    @abstractmethod
    def get_manifest_list_write_latency(self, size_kib: float = 10.0) -> float: ...

    @abstractmethod
    def get_manifest_file_read_latency(self, size_kib: float = 100.0) -> float: ...

    @abstractmethod
    def get_manifest_file_write_latency(self, size_kib: float = 100.0) -> float: ...

    @abstractmethod
    def get_put_latency(self, size_mib: float = 1.0) -> float: ...

    @abstractmethod
    def get_min_latency(self) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class ObjectStorageProvider(StorageProvider):
    """Storage provider backed by object storage (S3, Azure, GCS)."""

    def __init__(self, provider: str):
        if provider not in STORAGE_CONFIGS:
            raise ValueError(f"Unknown storage provider: {provider}. "
                           f"Valid options: {list(STORAGE_CONFIGS.keys())}")
        self._provider = provider
        self._config = STORAGE_CONFIGS[provider]

    def _sample_lognormal(self, median: float, sigma: float) -> float:
        latency = np.random.lognormal(mean=np.log(median), sigma=sigma)
        return max(self._config.min_latency, latency)

    def _sample_with_size(self, median: float, sigma: float,
                          size_kib: float, base_size_kib: float = 10.0) -> float:
        return self._sample_lognormal(median, sigma)

    def get_manifest_list_read_latency(self, size_kib: float = 10.0) -> float:
        return self._sample_with_size(self._config.ml_read_median, self._config.ml_read_sigma, size_kib)

    def get_manifest_list_write_latency(self, size_kib: float = 10.0) -> float:
        return self._sample_with_size(self._config.ml_write_median, self._config.ml_write_sigma, size_kib)

    def get_manifest_file_read_latency(self, size_kib: float = 100.0) -> float:
        return self._sample_with_size(self._config.mf_read_median, self._config.mf_read_sigma, size_kib)

    def get_manifest_file_write_latency(self, size_kib: float = 100.0) -> float:
        return self._sample_with_size(self._config.mf_write_median, self._config.mf_write_sigma, size_kib)

    def get_put_latency(self, size_mib: float = 1.0) -> float:
        base = self._config.put_base_latency
        size_component = size_mib * self._config.put_per_mib
        median = base + size_component
        latency = np.random.lognormal(mean=np.log(median), sigma=self._config.put_sigma)
        return max(self._config.min_latency, latency)

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return self._provider


def create_storage_provider(provider: str) -> StorageProvider:
    return ObjectStorageProvider(provider)


# ---------------------------------------------------------------------------
# catalog_provider.py → CatalogConfig, CATALOG_CONFIGS, CatalogProvider, etc.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogConfig:
    """Configuration for a catalog provider."""
    cas_median: float
    cas_sigma: float
    min_latency: float
    supports_append: bool = False
    append_median: float | None = None
    append_sigma: float | None = None


CATALOG_CONFIGS = {
    "instant": CatalogConfig(
        cas_median=1.0, cas_sigma=0.1, min_latency=1.0,
        supports_append=True, append_median=1.0, append_sigma=0.1,
    ),
    "s3x": CatalogConfig(
        cas_median=22.0, cas_sigma=0.5, min_latency=10.0,
        supports_append=True, append_median=21.0, append_sigma=0.5,
    ),
    "s3": CatalogConfig(
        cas_median=61.0, cas_sigma=0.5, min_latency=43.0,
        supports_append=False,
    ),
    "azurex": CatalogConfig(
        cas_median=64.0, cas_sigma=0.5, min_latency=40.0,
        supports_append=True, append_median=70.0, append_sigma=0.5,
    ),
    "azure": CatalogConfig(
        cas_median=93.0, cas_sigma=0.5, min_latency=51.0,
        supports_append=True, append_median=87.0, append_sigma=0.5,
    ),
    "gcp": CatalogConfig(
        cas_median=170.0, cas_sigma=0.5, min_latency=118.0,
        supports_append=False,
    ),
}
CATALOG_CONFIGS["aws"] = CATALOG_CONFIGS["s3x"]


class CatalogProvider(ABC):
    """Abstract interface for catalog operations."""

    @abstractmethod
    def get_cas_latency(self) -> float: ...

    @abstractmethod
    def get_append_latency(self) -> float: ...

    @abstractmethod
    def supports_append(self) -> bool: ...

    @abstractmethod
    def get_min_latency(self) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class InstantCatalog(CatalogProvider):
    """Instant catalog with ~1ms latency."""

    def __init__(self, config: CatalogConfig | None = None):
        self._config = config or CATALOG_CONFIGS["instant"]

    def get_cas_latency(self) -> float:
        latency = np.random.lognormal(mean=np.log(self._config.cas_median), sigma=self._config.cas_sigma)
        return max(self._config.min_latency, latency)

    def get_append_latency(self) -> float:
        if not self._config.supports_append:
            raise NotImplementedError("Instant catalog doesn't support append")
        latency = np.random.lognormal(mean=np.log(self._config.append_median), sigma=self._config.append_sigma)
        return max(self._config.min_latency, latency)

    def supports_append(self) -> bool:
        return self._config.supports_append

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return "instant"


class ObjectStorageCatalog(CatalogProvider):
    """Catalog backed by object storage conditional writes."""

    def __init__(self, provider: str):
        if provider not in CATALOG_CONFIGS:
            raise ValueError(f"Unknown catalog provider: {provider}. "
                           f"Valid options: {list(CATALOG_CONFIGS.keys())}")
        self._provider = provider
        self._config = CATALOG_CONFIGS[provider]

    def get_cas_latency(self) -> float:
        latency = np.random.lognormal(mean=np.log(self._config.cas_median), sigma=self._config.cas_sigma)
        return max(self._config.min_latency, latency)

    def get_append_latency(self) -> float:
        if not self._config.supports_append:
            raise NotImplementedError(f"{self._provider} catalog doesn't support append operations")
        latency = np.random.lognormal(mean=np.log(self._config.append_median), sigma=self._config.append_sigma)
        return max(self._config.min_latency, latency)

    def supports_append(self) -> bool:
        return self._config.supports_append

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return self._provider


def create_catalog_provider(catalog_type: str, storage_provider: str | None = None) -> CatalogProvider:
    if catalog_type == "instant":
        return InstantCatalog()
    elif catalog_type == "object_storage":
        if storage_provider is None:
            raise ValueError("storage_provider required for object_storage catalog")
        return ObjectStorageCatalog(storage_provider)
    else:
        raise ValueError(f"Unknown catalog type: {catalog_type}. "
                        f"Valid options: 'instant', 'object_storage'")


# ---------------------------------------------------------------------------
# operation.py → OperationType, ConflictCost (legacy), OperationBehavior, etc.
# ---------------------------------------------------------------------------

class OperationType(Enum):
    """Iceberg operation types with distinct conflict costs."""
    FAST_APPEND = "fast_append"
    MERGE_APPEND = "merge_append"
    VALIDATED_OVERWRITE = "validated_overwrite"


@dataclass
class LegacyConflictCost:
    """Cost model for a conflict resolution attempt (legacy version)."""
    metadata_reads: int = 1
    ml_reads: int = 1
    ml_writes: int = 1
    historical_ml_reads: int = 0
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0

    def total_ml_reads(self) -> int:
        return self.ml_reads + self.historical_ml_reads


# Keep the old name for backward compatibility
ConflictCost = LegacyConflictCost


class OperationBehavior(ABC):
    """Strategy pattern for operation-specific conflict handling."""

    @abstractmethod
    def get_false_conflict_cost(self, n_behind: int, ml_append_mode: bool = False) -> ConflictCost: ...

    @abstractmethod
    def can_have_real_conflict(self) -> bool: ...

    @abstractmethod
    def get_name(self) -> str: ...


class FastAppendBehavior(OperationBehavior):
    def get_false_conflict_cost(self, n_behind: int, ml_append_mode: bool = False) -> ConflictCost:
        return ConflictCost(
            metadata_reads=1, ml_reads=1,
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=0,
            manifest_file_reads=0, manifest_file_writes=0,
        )

    def can_have_real_conflict(self) -> bool:
        return False

    def get_name(self) -> str:
        return "FastAppend"


class MergeAppendBehavior(OperationBehavior):
    def __init__(self, manifests_per_commit: float = 1.0):
        self.manifests_per_commit = manifests_per_commit

    def get_false_conflict_cost(self, n_behind: int, ml_append_mode: bool = False) -> ConflictCost:
        k_manifests = max(1, int(n_behind * self.manifests_per_commit))
        return ConflictCost(
            metadata_reads=1, ml_reads=1,
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=0,
            manifest_file_reads=k_manifests, manifest_file_writes=k_manifests,
        )

    def can_have_real_conflict(self) -> bool:
        return False

    def get_name(self) -> str:
        return "MergeAppend"


class ValidatedOverwriteBehavior(OperationBehavior):
    def get_false_conflict_cost(self, n_behind: int, ml_append_mode: bool = False) -> ConflictCost:
        return ConflictCost(
            metadata_reads=1, ml_reads=1,
            ml_writes=0 if ml_append_mode else 1,
            historical_ml_reads=n_behind,
            manifest_file_reads=0, manifest_file_writes=0,
        )

    def can_have_real_conflict(self) -> bool:
        return True

    def get_name(self) -> str:
        return "ValidatedOverwrite"


FAST_APPEND_BEHAVIOR = FastAppendBehavior()
VALIDATED_OVERWRITE_BEHAVIOR = ValidatedOverwriteBehavior()


def get_behavior(op_type: OperationType, manifests_per_commit: float = 1.0) -> OperationBehavior:
    if op_type == OperationType.FAST_APPEND:
        return FAST_APPEND_BEHAVIOR
    elif op_type == OperationType.MERGE_APPEND:
        return MergeAppendBehavior(manifests_per_commit)
    elif op_type == OperationType.VALIDATED_OVERWRITE:
        return VALIDATED_OVERWRITE_BEHAVIOR
    else:
        raise ValueError(f"Unknown operation type: {op_type}")


# ---------------------------------------------------------------------------
# conflict.py → ConflictResult, resolve_conflict, ConflictResolverV2, etc.
# ---------------------------------------------------------------------------

@dataclass
class ConflictResult:
    """Result of conflict resolution."""
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
    """Resolve a CAS conflict based on operation type."""
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
        is_real_conflict = np.random.random() < real_conflict_probability

        if is_real_conflict:
            logger.debug(f"{sim.now} TXN {txn.id} Real conflict detected - ValidationException (abort)")
            stats.real_conflicts += 1
            stats.validation_exceptions += 1
            return ConflictResult(should_retry=False, abort_reason="validation_exception")

    # 2. No real conflict - pay false conflict cost
    cost = behavior.get_false_conflict_cost(n_behind, ml_append_mode)

    logger.debug(f"{sim.now} TXN {txn.id} False conflict cost: metadata_reads={cost.metadata_reads}, "
                f"ml_reads={cost.ml_reads}, historical_ml_reads={cost.historical_ml_reads}, "
                f"ml_writes={cost.ml_writes}, mf_reads={cost.manifest_file_reads}, "
                f"mf_writes={cost.manifest_file_writes}")

    for _ in range(cost.metadata_reads):
        yield sim.timeout(get_metadata_root_latency('read'))

    if not TABLE_METADATA_INLINED:
        yield sim.timeout(get_table_metadata_latency('read'))

    if cost.historical_ml_reads > 0:
        yield from _read_manifest_lists_batched(
            sim, cost.historical_ml_reads, txn.id, stats, MAX_PARALLEL,
            get_manifest_list_latency, get_put_latency, T_PUT
        )

    if cost.ml_reads > 0:
        for _ in range(cost.ml_reads):
            yield sim.timeout(get_manifest_list_latency('read'))
            stats.manifest_list_reads += 1

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

    if cost.ml_writes > 0:
        for _ in range(cost.ml_writes):
            yield sim.timeout(get_manifest_list_latency('write'))
            stats.manifest_list_writes += 1

    if not TABLE_METADATA_INLINED:
        yield sim.timeout(get_table_metadata_latency('write'))

    stats.false_conflicts += 1

    txn.v_catalog_seq = snapshot.seq
    _update_txn_dirty_versions(txn, snapshot)

    return ConflictResult(should_retry=True, abort_reason=None)


def _read_manifest_lists_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_ml_latency, get_put_latency, t_put
) -> Generator:
    if count <= 0:
        return
    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_ml_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))
    stats.manifest_list_reads += count


def _read_manifest_files_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_mf_latency
) -> Generator:
    if count <= 0:
        return
    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_mf_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))
    stats.manifest_files_read += count


def _write_manifest_files_batched(
    sim, count: int, txn_id: int, stats, max_parallel: int,
    get_mf_latency
) -> Generator:
    if count <= 0:
        return
    for batch_start in range(0, count, max_parallel):
        batch_size = min(max_parallel, count - batch_start)
        batch_latencies = [get_mf_latency('write') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))
    stats.manifest_files_written += count


def _update_txn_dirty_versions(txn: 'Txn', snapshot: 'CatalogSnapshot') -> None:
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
    """Resolve CAS conflict with partition-aware costs."""
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

    overlapping_partitions = _compute_overlapping_partitions(txn, partition_seq_snapshot)

    if not overlapping_partitions:
        txn.v_catalog_seq = snapshot.seq
        _update_txn_dirty_versions(txn, snapshot)
        return ConflictResult(should_retry=True, abort_reason=None)

    total_overlapping = sum(len(parts) for parts in overlapping_partitions.values())
    logger.debug(f"{sim.now} TXN {txn.id} Partition conflict resolution: "
                f"{total_overlapping} overlapping partitions, behavior={behavior.get_name()}")

    for tbl_id, partitions in overlapping_partitions.items():
        for part_id in partitions:
            snapshot_v = partition_seq_snapshot.get(tbl_id, {}).get(part_id, 0)
            txn_v = txn.v_partition_seq.get(tbl_id, {}).get(part_id, 0)
            n_behind = snapshot_v - txn_v

            if n_behind > 0:
                yield from _read_partition_manifest_lists(
                    sim, n_behind, txn.id, tbl_id, part_id, stats, MAX_PARALLEL,
                    get_manifest_list_latency, get_put_latency, T_PUT
                )

            if can_have_real_conflict:
                is_real_conflict = np.random.random() < data_overlap_probability

                if is_real_conflict:
                    logger.debug(f"{sim.now} TXN {txn.id} Real conflict at table {tbl_id} "
                                f"partition {part_id} - ValidationException (abort)")
                    stats.real_conflicts += 1
                    stats.validation_exceptions += 1
                    return ConflictResult(should_retry=False, abort_reason="validation_exception")

            yield from _resolve_partition_false_conflict(
                sim, txn, tbl_id, part_id, behavior, ml_append_mode,
                manifests_per_commit, n_behind, stats, MAX_PARALLEL,
                get_manifest_list_latency, get_manifest_file_latency,
                get_put_latency, T_PUT
            )
            stats.false_conflicts += 1

    txn.v_catalog_seq = snapshot.seq
    _update_txn_dirty_versions(txn, snapshot)
    _update_txn_partition_versions(txn, snapshot, overlapping_partitions)

    return ConflictResult(should_retry=True, abort_reason=None)


def _compute_overlapping_partitions(txn: 'Txn', partition_seq_snapshot: dict) -> dict[int, set[int]]:
    overlapping = {}
    for tbl_id, partitions in txn.partitions_read.items():
        for p in partitions:
            snapshot_v = partition_seq_snapshot.get(tbl_id, {}).get(p, 0)
            txn_v = txn.v_partition_seq.get(tbl_id, {}).get(p, 0)
            if snapshot_v != txn_v:
                if tbl_id not in overlapping:
                    overlapping[tbl_id] = set()
                overlapping[tbl_id].add(p)
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
    if count <= 0:
        return
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
    from endive.main import sample_conflicting_manifests

    if ml_append_mode:
        pass
    else:
        yield sim.timeout(get_ml_latency('write'))
        stats.manifest_list_writes += 1

    if behavior.get_name() == "merge_append" and n_behind > 0:
        k_manifests = int(n_behind * manifests_per_commit)
        if k_manifests > 0:
            for batch_start in range(0, k_manifests, max_parallel):
                batch_size = min(max_parallel, k_manifests - batch_start)
                batch_latencies = [get_mf_latency('read') for _ in range(batch_size)]
                yield sim.timeout(max(batch_latencies))
            stats.manifest_files_read += k_manifests

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
    if snapshot.partition_seq is None:
        return
    for tbl_id, partitions in overlapping_partitions.items():
        if tbl_id not in txn.v_partition_seq:
            txn.v_partition_seq[tbl_id] = {}
        for p in partitions:
            txn.v_partition_seq[tbl_id][p] = snapshot.partition_seq[tbl_id][p]
        if snapshot.partition_ml_offset is not None:
            if tbl_id not in txn.v_partition_ml_offset:
                txn.v_partition_ml_offset[tbl_id] = {}
            for p in partitions:
                txn.v_partition_ml_offset[tbl_id][p] = snapshot.partition_ml_offset[tbl_id][p]


class ConflictResolverV2:
    """Legacy conflict resolver wrapper."""

    @staticmethod
    def resolve(sim, txn, snapshot, real_conflict_probability, manifest_list_mode="rewrite",
                manifests_per_commit=1.0, stats=None) -> Generator:
        result = yield from resolve_conflict(
            sim, txn, snapshot, real_conflict_probability,
            manifest_list_mode, manifests_per_commit, stats
        )
        return result

    @staticmethod
    def should_use_operation_aware_resolution(txn: 'Txn') -> bool:
        return txn.operation_type is not None


# ---------------------------------------------------------------------------
# transaction.py legacy types → Txn, LogEntry
# ---------------------------------------------------------------------------

@dataclass
class Txn:
    """Legacy transaction state (used by old main.py)."""
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
    start_snapshot: CatalogSnapshot | None = None
    current_snapshot: CatalogSnapshot | None = None
    operation_type: OperationType | None = None
    abort_reason: str | None = None

    def get_behavior(self, manifests_per_commit: float = 1.0) -> OperationBehavior:
        if self.operation_type is None:
            return FAST_APPEND_BEHAVIOR
        return get_behavior(self.operation_type, manifests_per_commit)


@dataclass
class LogEntry:
    """Legacy log entry (used by old main.py)."""
    txn_id: int
    tables_written: dict[int, int]
    tables_read: dict[int, int]
    sealed: bool = False
