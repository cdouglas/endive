# Simulation Model Simplifications

This document describes simplifications made in the Endive discrete-event simulator compared to real Apache Iceberg implementations.

## Catalog Model

### Transaction Isolation
- **Simplification**: Transactions are isolated by table groups (`num_groups`). When `num_groups = 1`, all writes conflict at catalog level.
- **Reality**: Iceberg uses table-level isolation; cross-table transactions with ACID guarantees require external coordination.

### Compare-and-Swap (CAS)
- **Simplification**: CAS is atomic and instantaneous at the logical level; latency is added separately.
- **Reality**: CAS implementations vary by catalog backend (Hive Metastore, Nessie, REST, etc.) with different consistency guarantees.

### Conflict Resolution
- **Simplification**: False vs real conflicts determined by `REAL_CONFLICT_PROBABILITY` parameter.
- **Reality**: Conflict type depends on actual data file overlap; partition pruning, schema evolution, and delete files affect conflict detection.

## Storage Model

### Latency Distributions
- **Simplification**: All latencies drawn from lognormal distributions with provider-specific parameters.
- **Reality**: Real cloud storage has multimodal latency distributions, regional variation, and time-of-day effects.

### Size-Based Latency
- **Simplification**: PUT latency = `base + size_mib * rate + noise` (Durner et al. VLDB 2023 model).
- **Reality**: Object storage has step-function pricing tiers, multipart upload thresholds, and caching effects.

### Manifest Operations
- **Simplification**: Manifest file reads/writes use fixed latency distributions; size not tracked per-file.
- **Reality**: Manifest file sizes vary significantly based on partition count and metadata complexity.

### Table Metadata
- **Simplification**: Table metadata size is a static configuration (`TABLE_METADATA_SIZE_BYTES`, default 10KB).
- **Reality**: Table metadata grows with schema evolution, property changes, and snapshot retention.

## Transaction Model

### Runtime Distribution
- **Simplification**: Transaction runtime drawn from truncated lognormal distribution.
- **Reality**: Query execution time depends on data volume, parallelism, resource contention, and query complexity.

### Retry Behavior
- **Simplification**: Failed transactions retry with optional exponential backoff.
- **Reality**: Real systems may have circuit breakers, queue management, and priority scheduling.

### Manifest List Updates
- **Simplification**: Manifest list size tracked as running sum of entry sizes.
- **Reality**: Manifest lists are Avro files with compression; actual size depends on encoding efficiency.

## Append Operations (ML+ Mode)

### Conditional Append
- **Simplification**: Append operations check offset only; no conditional checksums or ETags.
- **Reality**: Cloud providers use ETags (AWS), leases (Azure), or preconditions for conditional operations.

### Sealing and Compaction
- **Simplification**: Manifest lists seal at byte threshold; rewrite resets offset to `MANIFEST_LIST_ENTRY_SIZE`.
- **Reality**: Sealing may involve versioning, tombstones, or reader coordination.

### Tentative Entries
- **Simplification**: Tentative ML entries are not stored; filtering is implicit in conflict resolution.
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

---

*For detailed technical debt and deferred tasks, see [errata.md](errata.md).*
