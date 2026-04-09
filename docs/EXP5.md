# Experiment 5: Partition-Aware Single Table

## 1. Summary

### Goal

Show that partitioning a table reduces commit conflict costs. Since the
catalog is a file, atomic multi-partition commits are free — partitioning a
table is isomorphic to splitting it into sub-tables. Each partition has its
own manifest list within the catalog, so concurrent writers to disjoint
partitions retry for free (catalog re-read + CAS only, no ML I/O).

The experiment quantifies this by sweeping the number of partitions written
per transaction (`partitions_per_txn`) against arrival rate. At low
`partitions_per_txn`, overlap between concurrent writers is rare and
throughput approaches the catalog-limited ceiling (as seen when distributing
load across tables in exp4). At high `partitions_per_txn`, overlap is near
certain and the system converges to single-partition behavior (exp1).

A secondary question is how the partition selection distribution affects
this: uniform selection spreads writes evenly; Zipf concentrates writes on
hot partitions, reducing the effective parallelism.

### Variants

| Config | Op mix | Partition selection | Purpose |
|--------|--------|-------------------|---------|
| `exp5a_partition_fa.toml` | 100% FA | Uniform | Isolate partition effect on pure-append |
| `exp5a_zipf_partition_fa.toml` | 100% FA | Zipf (alpha=1.5) | Show skew collapses to single-partition |
| `exp5b_partition_mix.toml` | 90/10 FA/VO | Uniform | VO I/O convoy cost reduced by partitions |
| `exp5b_zipf_partition_mix.toml` | 90/10 FA/VO | Zipf (alpha=1.5) | Skew amplifies VO retry cost |

### Swept Parameters (2-dimensional per config)

| Parameter | Values | Config key |
|-----------|--------|------------|
| `arrival_rate` | LOAD_SWEEP (20..5000) | `inter_arrival.scale` |
| `partitions_per_txn` | 1, 2, 4, 8, 16 | `partition.partitions_per_txn` |

Total per variant: 10 load levels x 5 partition counts = 50 points x 3+ seeds.

### Fixed Parameters

- Duration: 1 hour simulated
- `num_tables`: 1
- `partitions_per_table`: 32
- Catalog: S3 CAS (non-inlined, `table_metadata_inlined = false`)
- Storage: S3, `max_parallel = 4`
- Retry limit: 10
- `real_conflict_probability`: 0.0
- Manifest list mode: `rewrite`

### Partition Design

The catalog is a single file containing per-partition manifest lists. This is
**Design 1** from the design discussion: single table manifest with multiple
manifest lists. Each commit atomically updates all partition state via CAS.

Per-attempt I/O cost (non-inlined): TM read(1) + ML read(N) + ML write(N) +
TM write(1), where N = partitions written. On disjoint retry: free (catalog
re-read + CAS). On overlapping retry: same cost scaled to M overlapping
partitions. See SPEC.md §3.4 for the full cost table.

### Overlap Probability

For two concurrent uniform-random writers each selecting k of 32 partitions:

| k | P(overlap) |
|---|-----------|
| 1 | 3% |
| 2 | 12% |
| 4 | 34% |
| 8 | 73% |
| 16 | 99% |

### Expected Patterns

- **partitions_per_txn=1, uniform:** Throughput approaches catalog-limited
  ceiling (similar to 32 tables in exp4). Free retries dominate.
- **partitions_per_txn=16, uniform:** Converges to single-partition behavior
  (exp1). Nearly every CAS failure has partition overlap.
- **Zipf at any k:** Hot partition concentrates conflicts. Even at k=1,
  ~45% of writes hit the hottest partition, reducing effective parallelism.
- **VO (exp5b):** Benefits more from partitions than FA. FA has zero conflict
  cost regardless; VO pays an I/O convoy that scales with overlapping
  partitions. Fewer overlapping partitions = cheaper convoy.

---

## 2. Analysis

### Correctness Checks

1. **Operation type invariant (5a):** All rows must be `fast_append`.
2. **Conflict classification:** `catalog_conflicts` should dominate at low
   `partitions_per_txn` (disjoint = catalog-level conflict only);
   `tblptn_conflicts` should dominate at high `partitions_per_txn`.
3. **Conflict sum invariant:** `catalog_conflicts + tblptn_conflicts ==
   total_retries` for every transaction.
4. **Partition IDs in range:** `write_partition_ids` values in [0, 32).
5. **Cross-validation:** `partitions_per_txn=16` should approximate exp1
   throughput at matching arrival rate and catalog latency.

### Graphs Produced

Each config produces:

| Graph | Purpose |
|-------|---------|
| `commit_rate_over_time` | Verify steady-state (should be flat for non-inlined) |
| `latency_vs_throughput` grouped by `partitions_per_txn` | Main result: saturation curves shift right with fewer partitions per txn |
| `heatmap` (arrival_rate x partitions_per_txn) | Success rate, throughput, latency overview |

Mix variants (5b) also produce:

| Graph | Purpose |
|-------|---------|
| `operation_types` grouped by `partitions_per_txn` | FA vs VO per-type metrics |

---

## 3. Implementation

### Partition Conflict Model

The simulator uses **Design 1** from the design discussion:

1. **Single CAS target** — the catalog file. All writers contend on one CAS.
2. **Per-partition manifest lists** — each partition has its own ML within
   the catalog. Per-attempt I/O scales with the number of partitions written.
3. **Partition version vectors** — `CatalogSnapshot` stores per-partition
   versions. On CAS failure, `compute_write_overlap()` compares the txn's
   written partitions against intervening version changes.
4. **Disjoint = free retry** — if no written partition was modified by a
   concurrent commit, the retry pays zero ML I/O (just catalog re-read + CAS).
5. **Overlapping = scaled I/O** — only the overlapping partitions pay ML
   read + ML write on retry.

### Key Code Paths

- **Partition version tracking:** `catalog.py` — `_MutableTable.partition_versions`
  incremented on successful commit.
- **Overlap detection:** `transaction.py:compute_write_overlap()` — returns
  `WriteOverlap` with per-table partition sets.
- **Per-attempt cost scaling:** `transaction.py:_commit_loop()` — sets
  `per_attempt_n = overlap.n_partitions` (0 for disjoint = skip).
- **Conflict detection:** `conflict_detector.py:PartitionOverlapConflictDetector`
  — checks per-partition version changes for real conflicts (VO only).
- **Partition selection:** `workload.py:UniformPartitionSelector` /
  `ZipfPartitionSelector` — selects which partitions each txn writes.

### Test Coverage

See EXP6.md for full test coverage table. All partition machinery is well
covered by existing tests.
