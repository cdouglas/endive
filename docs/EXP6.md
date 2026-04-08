# Experiment 6: Inlined Table Manifest

## 1. Summary

### Goal

Evaluate the tradeoff of inlining the table manifest in the catalog CAS
object. With inlining, the ML content for each partition is stored in the
CAS payload. This eliminates all separate ML I/O (reads and writes are
absorbed into the CAS operation), but the CAS payload grows with each
successful commit, increasing CAS latency over time.

The experiment repeats exp5's sweep with `table_metadata_inlined = true`,
enabling direct comparison. Early in the simulation, inlined commits are
faster (no ML round-trips). Late in the simulation, the growing CAS payload
degrades performance. The crossover point depends on load, partition count
per transaction, and partition selection distribution.

### Variants

| Config | Op mix | Partition selection | Purpose |
|--------|--------|-------------------|---------|
| `exp6a_inlined_fa.toml` | 100% FA | Uniform | Inlining effect on pure-append |
| `exp6a_zipf_inlined_fa.toml` | 100% FA | Zipf (alpha=1.5) | Inlining + skew |
| `exp6b_inlined_mix.toml` | 90/10 FA/VO | Uniform | VO convoy eliminated by inlining |
| `exp6b_zipf_inlined_mix.toml` | 90/10 FA/VO | Zipf (alpha=1.5) | Inlining + skew + VO |

### Swept Parameters (2-dimensional per config)

| Parameter | Values | Config key |
|-----------|--------|------------|
| `arrival_rate` | LOAD_SWEEP (20..5000) | `inter_arrival.scale` |
| `partitions_per_txn` | 1, 2, 4, 8, 16 | `partition.partitions_per_txn` |

Same sweep as exp5, enabling direct comparison.

### Fixed Parameters

Same as exp5, except:
- Catalog: S3 CAS, **inlined** (`table_metadata_inlined = true`)
- `initial_partition_size_bytes`: 2048 (32 partitions x 2 KiB = 64 KiB initial)
- `commit_growth_bytes`: 100 (CAS payload grows 100 bytes per commit)

### Catalog Size Growth Model

```
initial_size = 32 * 2048 = 65,536 bytes (64 KiB)
size(N)      = 65,536 + N * 100
```

After 1000 commits: 164 KiB. After 10,000 commits: 1040 KiB.

CAS latency grows with size through the S3 storage provider's
`SizeBasedLatency` model. The CAS read and write operations pass
`catalog_size_bytes` to the storage provider, which scales latency
proportionally to payload size.

### Expected Patterns

- **Throughput degrades over time.** `commit_rate_over_time` shows a
  downward curve (vs exp5's flat line).
- **Initially faster than exp5.** No ML round-trips means lower per-attempt
  cost. At low cumulative commits, inlined wins.
- **Eventually slower than exp5.** CAS growth overtakes ML savings. The
  crossover depends on throughput (more commits = faster growth).
- **VO benefits from inlining per-attempt savings (exp6b vs exp5b).** The
  I/O convoy (historical ML reads) is NOT eliminated — VO must still read
  historical MLs to validate the read set. But each retry attempt is
  cheaper (j MF writes vs 3j ops), which reduces total retry latency.
- **Zipf variants degrade faster.** Hot partitions drive higher commit rates
  early, which accelerates CAS growth.

---

## 2. Analysis

### Correctness Checks

1. **Monotonic CAS latency increase.** Binning commits into time windows
   should show increasing mean CAS latency.
2. **Operation type invariant (6a):** All rows must be `fast_append`.
3. **VO convoy persists (6b):** `conflict_io_ms` for VO retries with
   partition overlap should be nonzero (historical ML reads remain).
   On disjoint retries, `conflict_io_ms` should be 0 (same as exp5).
4. **Reduced per-attempt I/O:** `per_attempt_io_ms` should be lower than
   exp5 at matching parameters early in the simulation.
5. **Catalog size growth:** `catalog.catalog_size_bytes` at end of
   simulation should match `initial_size + committed * 100`.

### Graphs Produced

Same structure as exp5. Key comparison: overlay exp5 and exp6
`commit_rate_over_time` to show flat vs degrading throughput.

---

## 3. Implementation

### What Changes vs Exp5

The only config difference is `table_metadata_inlined = true` plus
`initial_partition_size_bytes` and `commit_growth_bytes`. The code paths:

#### Per-Attempt Cost Elimination (`transaction.py:232-235`)

When `metadata_inlined = True`:
```python
return ConflictCost(manifest_file_writes=n_partitions)
```

ML reads and ML writes eliminated. Only MF writes remain (per-partition
storage work that exists regardless of inlining).

#### VO Conflict Cost Unchanged (`transaction.py:741-750`)

The I/O convoy is NOT eliminated by inlining. Historical ML reads are still
required to validate the read set. Inlining the current table manifest gives
the client the current partition state (from the catalog read), but not the
history of intermediate changes. The convoy cost remains:
```
historical_ml_reads = max(0, n_snapshots_behind - 1) * n_partitions
```

#### CAS Size Growth (`catalog.py:318-319, 557-558`)

On each successful commit:
```python
if self._metadata_inlined:
    self._catalog_size_bytes += self._commit_growth_bytes
```

`CASCatalog` passes `_catalog_size_bytes` to S3 storage's `read()` and
`cas()` methods, which use `SizeBasedLatency` to scale latency with size.
`InstantCatalog` uses `_effective_latency()` with `latency_per_kib_ms`.

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| `_effective_latency()` formula | `TestInlinedMetadataLatency` (7 tests) | Formula, edge cases, commit latency |
| Catalog size growth | `TestCatalogSizeGrowth` (5 tests) | Growth on success, no growth on failure, N-commit accumulation |
| Inlined cost elimination | `TestInlinedMetadataCosts` (11 tests) | ML elimination, MF retention, partition scaling, VO convoy |
| Config loading | `TestComponentBuilding` (3 tests) | Inlined params, latency_per_kib_ms flow |
| Full simulation | `TestEndToEnd::test_inlined_metadata_simulation` | Latency growth over time assertion |
| Partition version tracking | `TestPartitionVersionTracking` (3 tests) | All catalog types |
| Partition overlap detection | `TestPartitionOverlapConflictDetector` (10 tests) | Same/different partition, multi-table, FA/MA immunity |
| Write overlap computation | `TestComputeWriteOverlap` | Multi-table/partition overlap scenarios |
| Config loading (partitions) | `TestComponentBuilding::test_partition_overlap_detector` | partition.enabled → detector creation |

### Known Approximation

The VO conflict cost uses the table-level version delta (`n_table_versions_behind`)
rather than per-partition version deltas. With one table, all commits to any
partition increment the table version. If 10 commits landed but only 2
touched the overlapping partition, the model charges for 10 historical ML
reads when 2 would suffice. This is conservative (overstates VO cost) and
only affects exp5b/exp6b at 10% VO. Noted for future refinement.
