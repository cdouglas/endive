# Experiment 6: Inlined Table Metadata

## 1. Summary

### Goal

Evaluate the tradeoff of inlining table manifests in the catalog CAS object.
In standard Iceberg, each commit attempt reads a manifest list (ML) from
object storage, writes a new manifest file (MF), and writes a new ML. With
inlining, the manifest is stored directly in the CAS payload, eliminating ML
reads and writes. The tradeoff: the CAS object grows with every successful
commit, increasing CAS latency over time.

This models a design where the catalog service (e.g. REST catalog, Nessie)
stores enough table state in its own persistence layer that clients can skip
the ML round-trips. The experiment quantifies when the growing payload cost
exceeds the ML I/O savings.

### Variants

| Config | Operation mix | Purpose |
|--------|--------------|---------|
| `exp6a_inlined_fa.toml` | 100 % FastAppend | Isolate inlining effect on pure-append workload |
| `exp6b_inlined_mix.toml` | 90 % FA / 10 % VO | Show inlining's larger benefit for VO (eliminates I/O convoy) |

### Swept Parameters (5-dimensional)

| Parameter | Values | Config key |
|-----------|--------|------------|
| `num_tables` | 10, 50 | `catalog.num_tables` |
| `partitions_per_table` | 1, 10, 100 | `catalog.partition.num_partitions` |
| `tables_per_txn` | 1, 3 | `transaction.tables_per_txn` |
| `catalog_latency` (base) | 1, 50 ms | `catalog.service.latency_ms` |
| `arrival_rate` | LOAD_SWEEP (20..5000) | `inter_arrival.scale` |

Same 5D sweep as exp5, enabling direct comparison.

### Fixed Parameters

- Duration: 1 hour simulated
- Catalog: single-file (`num_groups = 1`), **inlined** (`table_metadata_inlined = true`)
- `initial_partition_size_bytes`: 2048 (2 KiB per partition in the CAS object)
- `commit_growth_bytes`: 100 (each successful commit adds 100 bytes to CAS)
- `latency_per_kib_ms`: 0.5 (CAS latency scales at 0.5 ms per KiB of payload)
- Catalog backend: `service` / `instant`
- Storage: S3, `max_parallel = 4`
- Partition: enabled, `partitions_per_txn = 1`, uniform random
- Retry limit: 10
- Manifest list mode: `rewrite`

### Expected Patterns

- **Throughput degrades over time** as the catalog CAS payload grows. The
  `commit_rate_over_time` plot should show a downward curve, unlike exp5
  where throughput is flat.
- **More partitions -> larger initial catalog -> earlier degradation onset.**
  With 10 tables x 100 partitions x 2 KiB = 2000 KiB initial, the
  size-based latency starts at 1000 ms (at 0.5 ms/KiB), likely dominating
  from the start. With 10 tables x 1 partition x 2 KiB = 20 KiB, it starts
  at 10 ms.
- **Compared to exp5 (non-inlined):** Initially faster (no ML I/O per
  attempt), but eventually slower (CAS growth). The crossover point depends
  on partition count and base latency.
- **Exp6b: VO benefits more from inlining than FA.** FastAppend has zero
  conflict cost regardless. ValidatedOverwrite's I/O convoy
  (`n_snapshots_behind * n_partitions` historical ML reads) is completely
  eliminated by inlining. This should be visible as a large VO success rate
  improvement in exp6b vs exp5b.

### Catalog Size Growth Model

```
initial_size = num_tables * partitions_per_table * initial_partition_size_bytes
size(N)      = initial_size + N * commit_growth_bytes
latency(N)   = base_latency + (size(N) / 1024) * latency_per_kib_ms
```

Example at `num_tables = 10, partitions_per_table = 10`:
- Initial: 200 KiB -> latency = base + 100 ms
- After 1000 commits: 300 KiB -> latency = base + 150 ms
- After 10000 commits: 1200 KiB -> latency = base + 600 ms

---

## 2. Analysis

### Correctness Checks

1. **Monotonic latency increase:** CAS latency in the results should
   increase over simulation time. Binning commits into time windows and
   computing mean CAS latency per window should show a monotonically
   increasing trend.
2. **Operation type invariant (exp6a):** All rows must have
   `operation_type == "fast_append"`.
3. **No I/O convoy for VO (exp6b):** With inlining, ValidatedOverwrite
   conflict cost should be zero (no historical ML reads). The
   `conflict_io_ms` column for VO transactions should be 0.0 on retries
   where there is no partition overlap.
4. **Reduced per-attempt I/O:** With inlining, per-attempt cost eliminates
   ML reads and ML writes. Only manifest file writes remain. The
   `per_attempt_io_ms` column should be lower than exp5 at matching
   parameters (S3 ML read/write eliminated = ~86 ms savings per attempt).
5. **Cross-validation:** Same checks as exp5 for partition-related
   correctness (conflict sum invariant, partition ID ranges).
6. **Size growth sanity:** If the simulator exposes catalog size (via
   `catalog.catalog_size_bytes`), verify it matches the formula above.

### Graphs Produced

**Exp6a (100 % FA, inlined):**

| Graph type | Grouping / axes | Filters | File suffix |
|------------|----------------|---------|-------------|
| `commit_rate_over_time` | default | — | — |
| `heatmap` (x4) | X = arrival_rate, Y = partitions_per_table; metrics: success, throughput, mean/p99 latency | CAS={1,50}ms x tpt={1,3} x T=10 | `cas_Xms_tptY_t10` |
| `latency_vs_throughput` (x3) | group_by partitions_per_table or tables_per_txn | CAS={1,50}ms, various combos | various |

**Exp6b (90/10 mix, inlined):**

| Graph type | Grouping / axes | Filters | File suffix |
|------------|----------------|---------|-------------|
| `commit_rate_over_time` | default | — | — |
| `heatmap` (x4) | X = arrival_rate, Y = partitions_per_table; per_type_metrics: success_rate, mean_latency | CAS={1,50}ms x tpt={1,3} x T=10 | `cas_Xms_tptY_t10` |
| `latency_vs_throughput` (x2) | group_by partitions_per_table | CAS={1,50}ms, tpt=1, T=10 | `cas_Xms_tpt1_t10` |
| `operation_types` | group_by catalog_service_latency_ms | — | — |

### Key Metrics to Watch

- **`commit_rate_over_time`:** The signature plot for exp6. Should show clear
  degradation as the catalog grows. Compare directly against exp5's flat
  curve.
- **Mean latency vs. partitions_per_table:** At high partition counts (100),
  the initial catalog is large (2000 KiB = 1000 ms latency overhead).
  Throughput may be low from the start, masking the growth effect.
- **VO success rate (exp6b vs exp5b):** Inlining eliminates the I/O convoy.
  At high load, VO success rate should be substantially higher in exp6b than
  exp5b.
- **P99 latency growth over time:** Tail latency should be more sensitive to
  catalog growth than mean latency due to contention amplification.

---

## 3. Implementation

### New Functionality Compared to Exp1-4

Exp6 builds on exp5's partition machinery (Section 3 of EXP5.md) and adds
inlined metadata modeling. The key new components:

#### 3a. Catalog Size Tracking (`endive/catalog.py`)

Both `CASCatalog` and `InstantCatalog` track `_catalog_size_bytes`:

- **Initialization:** When `metadata_inlined = True`:
  ```
  total_partitions = sum(partitions_per_table)
  _catalog_size_bytes = total_partitions * initial_partition_size_bytes
  ```
  When `metadata_inlined = False`: fixed at 100 bytes (CASCatalog) or 0
  (InstantCatalog).

- **Growth:** On each successful commit (inside `if success:` block):
  ```python
  if self._metadata_inlined:
      self._catalog_size_bytes += self._commit_growth_bytes
  ```
  Growth only occurs on success, never on failed CAS attempts.

- **Impact on CAS I/O:** `CASCatalog` passes `_catalog_size_bytes` to its
  backing `CASStorage.compare_and_swap()` as `size_bytes`, which scales the
  storage latency model.

#### 3b. Size-Dependent Latency (`endive/catalog.py:515-520`)

`InstantCatalog._effective_latency()` computes:

```python
if not self._metadata_inlined or self._latency_per_kib_ms == 0.0:
    return self._base_latency
size_kib = self._catalog_size_bytes / 1024.0
return self._base_latency + self._latency_per_kib_ms * size_kib
```

This is used for both `read()` and `commit()` operations. The `commit()`
method splits the effective latency into two halves (request + response),
with the CAS check happening at the midpoint.

**Note:** Only `InstantCatalog` implements `_effective_latency`. The
`CASCatalog` achieves the same effect by passing growing `size_bytes` to the
underlying `CASStorage`, which has its own latency model. Both exp5 and exp6
use `provider = "instant"`, so `InstantCatalog` is the relevant
implementation.

#### 3c. Per-Attempt Cost Elimination (`endive/transaction.py:232-235`)

When `metadata_inlined = True`, the per-attempt cost drops ML I/O:

| Component | `metadata_inlined = False` | `metadata_inlined = True` |
|-----------|---------------------------|--------------------------|
| ML reads | n_partitions | 0 |
| MF writes | n_partitions | n_partitions |
| ML writes | n_partitions (0 in ML+ mode) | 0 |

This saves 2 storage round-trips per attempt (or 1 in ML+ mode).

#### 3d. Conflict Cost Elimination (`endive/transaction.py:741-743`)

For ValidatedOverwrite with `metadata_inlined = True`:

```python
if metadata_inlined:
    return ConflictCost()  # Zero — historical MLs inlined in catalog
```

Without inlining, VO pays `max(0, n_snapshots_behind - 1) * n_partitions`
historical ML reads. This "I/O convoy" is the dominant retry cost for VO at
high contention. Inlining eliminates it entirely.

FastAppend and MergeAppend conflict costs are unaffected (FA has zero cost
regardless; MA's re-merge cost stays because it operates on manifest files,
not manifest lists).

#### 3e. Config and Sweep Integration

The sweep generator in `run_all_experiments.py:540-568` handles exp6 configs
identically to exp5 — same 5D parameter space, same override keys. The
config difference is entirely in the TOML: `table_metadata_inlined = true`
plus `initial_partition_size_bytes`, `commit_growth_bytes`, and
`latency_per_kib_ms`.

The analysis pipeline (`extract_key_parameters`) extracts
`table_metadata_inlined` from `cfg.toml` (`saturation_analysis.py:619`),
enabling filtering and comparison across exp5 (non-inlined) and exp6
(inlined).

### Readiness Assessment

**Ready to run: YES.**

The partition machinery is well tested (see EXP5.md). The inlining-specific
code now has comprehensive test coverage:

| Component | Test coverage | Tests |
|-----------|--------------|-------|
| `_effective_latency()` formula | **YES** | `TestInlinedMetadataLatency` (7 tests): formula, edge cases, commit latency |
| `commit_growth_bytes` accumulation | **YES** | `TestCatalogSizeGrowth` (5 tests): growth on success, no growth on failure, N-commit accumulation |
| `initial_partition_size_bytes` | **YES** | `TestInlinedMetadataLatency::test_initial_size_from_partitions`, `test_initial_size_default` |
| `latency_per_kib_ms` config loading | **YES** | `TestComponentBuilding::test_inlined_latency_per_kib_flows_to_catalog` |
| Per-attempt cost with `metadata_inlined=True` | **YES** | `TestInlinedMetadataCosts` (6 tests): ML elimination, MF retention, partition scaling |
| VO conflict cost with `metadata_inlined=True` | **YES** | `TestInlinedMetadataCosts::test_vo_conflict_cost_zero_when_inlined`, `test_vo_conflict_cost_inlined_ignores_partitions` |
| Full simulation with inlined config | **YES** | `TestEndToEnd::test_inlined_metadata_simulation`: runs simulation, verifies latency growth |
| Catalog size growth over simulation time | **YES** | `TestCatalogSizeGrowth::test_latency_increases_with_growth`: monotonic latency check |
| Config loading (metadata_inlined, sizes) | **YES** | `TestComponentBuilding::test_inlined_metadata_config`, `test_inlined_metadata_default_false` |
| FA/MA cost unaffected by inlining | **YES** | `TestInlinedMetadataCosts::test_fa_conflict_cost_unaffected_by_inlining`, `test_ma_conflict_cost_unaffected_by_inlining` |

### Test Gaps and Recommendations

All high-priority gaps have been addressed (26 new tests added). Remaining
medium-priority items that could further increase confidence:

1. **Exp5 vs exp6 comparison test.** At identical parameters, run both
   configs and assert that exp6's per-attempt I/O time is lower (confirming
   ML elimination) but total commit latency at the end is higher (confirming
   CAS growth penalty).

2. **Large initial catalog test.** At `partitions_per_table=100,
   num_tables=50` (the extreme corner), the initial catalog is 10 MB.
   `_effective_latency()` returns `base + 5000 ms`. Verify this doesn't
   cause simulation timeouts or numerical issues. This is a realistic
   concern: at 5-second CAS latency and 10 retries, a single transaction
   could take 50+ seconds of simulated time.

3. **Analysis pipeline: `table_metadata_inlined` extraction.** Verify that
   `extract_key_parameters()` correctly reads `table_metadata_inlined` from
   `cfg.toml` and that filtering on this parameter works in the heatmap
   pipeline.
