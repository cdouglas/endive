# Experiment 6: Inlined Table Manifest

## 1. Summary

### Goal

Evaluate the tradeoff of inlining table metadata in the catalog CAS object.
With inlining, the table metadata file no longer exists as a separate storage
object — its state is read and written as part of the catalog CAS. This
eliminates 2 storage operations per commit attempt (TM read + TM write), but
the CAS payload grows with each successful commit, increasing CAS latency
over time.

The manifest list is NOT inlined — it remains a separate storage object.
Per-attempt cost is still ML read(N) + ML write(N), but without the TM
overhead.

The experiment repeats exp5's sweep with `table_metadata_inlined = true`,
enabling direct comparison. Early in the simulation, inlined commits are
faster (2 fewer ops per attempt). Late in the simulation, the growing CAS
payload degrades performance. The crossover point depends on load, partition
count per transaction, and partition selection distribution.

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
- `initial_partition_size_bytes`: 16000 (32 partitions × ~16 KiB ≈ 500 KiB)
- `commit_growth_bytes`: 0 (fixed size — trimming/compression at equilibrium)

### Catalog Size Model

```
catalog_size = 32 * 16,000 = 512,000 bytes (~500 KiB)
```

The CAS payload is fixed at ~500 KiB, modeling a table in steady state where
snapshot expiration and metadata compression balance new commit growth. This
avoids transient warmup effects in the 1-hour simulation (first and last 15
minutes are discarded as warmup/cooldown).

CAS latency scales with payload size through the S3 storage provider's
`SizeBasedLatency` model. At 500 KiB, the S3 PUT model adds
`~0.5 MiB × 20 ms/MiB ≈ 10ms` to the base CAS latency.

### Expected Patterns

- **Flat throughput over time** (same as exp5). Fixed CAS size means no
  degradation.
- **Faster than exp5.** 2 fewer storage ops per attempt (no TM read/write).
  The ~500 KiB CAS payload adds ~10ms latency, but saving 2 × ~43ms ops
  is a net win of ~76ms per attempt on S3.
- **VO benefits from inlining per-attempt savings (exp6b vs exp5b).** The
  I/O convoy (historical ML reads) is NOT eliminated — VO must still read
  historical MLs to validate the read set. But each retry attempt saves
  2 ops (no TM read/write), reducing total retry latency.
- **Zipf variants** show same relative benefit as uniform (fixed CAS size).

---

## 2. Analysis

### Correctness Checks

1. **Flat CAS latency.** With fixed-size catalog, CAS latency should be
   constant across time windows (no growth).
2. **Operation type invariant (6a):** All rows must be `fast_append`.
3. **VO convoy persists (6b):** `conflict_io_ms` for VO retries with
   partition overlap should be nonzero (historical ML reads remain).
   On disjoint retries, `conflict_io_ms` should be 0 (same as exp5).
4. **Reduced per-attempt I/O:** `per_attempt_io_ms` should be lower than
   exp5 at matching parameters (2 fewer ops: no TM read/write).
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

#### Per-Attempt Cost Reduction (`transaction.py:get_per_attempt_cost`)

When `metadata_inlined=True`, the TM file does not exist. Per-attempt cost
eliminates TM read and TM write:
```python
# Non-inlined: TM_r(1) + ML_r(N) + ML_w(N) + TM_w(1)  = 2 + 2N ops
# Inlined:     ML_r(N) + ML_w(N)                        = 2N ops
```

#### Failure-Path Cost Reduction

After CAS failure, the non-inlined path reads the catalog + a separate TM
read to get partition versions. Inlined absorbs this into the catalog read
(the TM state is in the CAS object):
```python
# Non-inlined failure: catalog_read + TM_read → overlap check
# Inlined failure:     catalog_read → overlap check
```

#### VO Conflict Cost Unchanged

The I/O convoy is NOT eliminated by inlining. Historical ML reads are still
required to validate the read set. Inlining gives the client the current
partition state (from the catalog read), but not the history of intermediate
changes. The convoy cost remains:
```
historical_ml_reads = max(0, n_table_versions_behind - 1) * n_partitions
```

#### CAS Payload Size

With `commit_growth_bytes=0`, the CAS payload is fixed at
`32 × initial_partition_size_bytes = 512,000 bytes` (~500 KiB).
`CASCatalog` passes `_catalog_size_bytes` to S3 storage's `read()` and
`cas()` methods, which use `SizeBasedLatency` to scale latency with size.

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| `_effective_latency()` formula | `TestInlinedMetadataLatency` (7 tests) | Formula, edge cases, commit latency |
| Catalog size growth | `TestCatalogSizeGrowth` (5 tests) | Growth on success, no growth on failure, N-commit accumulation |
| Inlined cost elimination | `TestYieldCount`, `TestRetryCostDecomposition` | TM elimination, ML retention, partition scaling, VO convoy |
| Config loading | `TestComponentBuilding` (3 tests) | Inlined params, latency_per_kib_ms flow |
| Full simulation | `TestEndToEnd::test_inlined_metadata_simulation` | Latency growth over time assertion |
| Partition version tracking | `TestPartitionVersionTracking` (3 tests) | All catalog types |
| Partition overlap detection | `TestPartitionOverlapConflictDetector` (10 tests) | Same/different partition, multi-table, FA/MA immunity |
| Write overlap computation | `TestComputeWriteOverlap` | Multi-table/partition overlap scenarios |
| Config loading (partitions) | `TestComponentBuilding::test_partition_overlap_detector` | partition.enabled → detector creation |

### Known Approximation

The VO I/O convoy uses per-table version deltas. With one table, all commits
to any partition increment the table version. If 10 commits landed but only 2
touched the overlapping partition, the model charges for 9 historical ML reads
(10-1) when 1 would suffice. This is conservative (overstates VO cost) and
only affects exp5b/exp6b at 10% VO weight.

For multi-table VO, the convoy is correctly computed per-table (not summed
across tables). See SPEC.md §3.8.
