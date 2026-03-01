# Simulation Model Simplifications

This document describes simplifications made in the Endive discrete-event simulator compared to real Apache Iceberg implementations.

## Catalog Model

### Transaction Isolation
- **Simplification**: Transactions contend on a single global catalog sequence number (`seq`). All commits — regardless of target table — must increment the same `seq`, modeling a single-file `FileIOCatalog`.
- **Reality**: Iceberg supports per-table versioning in REST catalogs backed by databases, which eliminates cross-table CAS failures entirely.
- **Mitigation**: Cross-table CAS failures are modeled as cheap retries (catalog read + re-CAS, no manifest I/O). The write overlap check (`has_write_overlap()`) detects this and skips conflict resolution.

### Compare-and-Swap (CAS)
- **Simplification**: CAS is atomic and instantaneous at the logical level; latency is added separately via `LatencyDistribution` objects.
- **Reality**: CAS implementations vary by catalog backend (Hive Metastore, Nessie, REST, etc.) with different consistency guarantees.

### Conflict Detection
- **Simplification**: `ProbabilisticConflictDetector` determines real vs false conflicts by configured probability. `PartitionOverlapConflictDetector` uses per-partition version tracking.
- **Reality**: Conflict type depends on actual data file overlap; partition pruning, schema evolution, and delete files affect conflict detection.

### Partition-Level Modeling
- **Simplification**: Partitions are abstract; each has a version counter in `TableMetadata.partition_versions`. Conflicts detected by comparing partition version vectors between snapshots.
- **Reality**: Partitions map to physical directory structures; conflict detection involves comparing manifest file lists and data file paths.

## Storage Model

### Latency Distributions
- **Simplification**: All latencies drawn from `LognormalLatency` distributions with provider-specific parameters loaded from `endive/providers/*.toml`.
- **Reality**: Real cloud storage has multimodal latency distributions, regional variation, and time-of-day effects.

### Size-Based Latency
- **Simplification**: PUT latency = `base + size_mib * rate + noise` via `SizeBasedLatency` (Durner et al. VLDB 2023 model).
- **Reality**: Object storage has step-function pricing tiers, multipart upload thresholds, and caching effects.

### Manifest Operations
- **Simplification**: Manifest file reads/writes use fixed latency distributions; size not tracked per-file.
- **Reality**: Manifest file sizes vary significantly based on partition count and metadata complexity.

## Transaction Model

### Runtime Distribution
- **Simplification**: Transaction runtime drawn from lognormal distribution via `WorkloadConfig.runtime`.
- **Reality**: Query execution time depends on data volume, parallelism, resource contention, and query complexity.

### Operation Types
- **Simplification**: Three types (`FastAppend`, `MergeAppend`, `ValidatedOverwrite`) with probabilistic mix via `WorkloadConfig` weights. Each type has fixed conflict cost formulas.
- **Reality**: Operation type is determined by the query, and conflict cost depends on actual manifest and data file sizes.

### Retry Behavior
- **Simplification**: Failed transactions retry with optional exponential backoff. Per-attempt and conflict I/O costs modeled via `ConflictCost` dataclass.
- **Reality**: Real systems may have circuit breakers, queue management, and priority scheduling.

## Append Operations (ML+ Mode)

### Conditional Append
- **Simplification**: Append operations check offset only; no conditional checksums or ETags.
- **Reality**: Cloud providers use ETags (AWS), leases (Azure), or preconditions for conditional operations.

### Tentative Entries
- **Simplification**: Tentative ML entries are not stored; filtering is implicit in the transaction's commit protocol.
- **Reality**: Readers must filter uncommitted entries or use snapshot isolation.

## What's NOT Modeled

1. **Network partitions and failures**: No modeling of transient network errors
2. **Storage throttling**: No rate limiting or quota enforcement
3. **Multi-region replication**: Single region only
4. **Schema evolution**: Static schema throughout simulation
5. **Delete files and position deletes**: Write-only workload
6. **Bloom filters and column statistics**: No read optimization modeling
7. **Query planning and execution**: Only commit path modeled
8. **External catalog services**: REST API latency not modeled separately
9. **Concurrent readers**: Only writers modeled

## Validation Status

The model has been validated against:
- **Latency distributions**: K-S tests against measured cloud latencies
- **Throughput bounds**: Little's Law predictions for single-server queue
- **Retry behavior**: Deterministic replay with fixed seeds

See `tests/test_statistical_rigor.py` and `tests/test_numerical_accuracy.py` for validation tests.
