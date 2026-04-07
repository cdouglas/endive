# Experiment 5: Partition-Aware Multi-Table Transactions

## 1. Summary

### Goal

Show that table-level partitioning reduces false-positive conflicts when
multiple tables share a single-file catalog. In Iceberg's OCC model, every
commit contends on a single catalog sequence number. Without partition
awareness, any concurrent commit to the same table looks like a conflict even
if the writers touched disjoint partitions. Experiment 5 adds per-partition
version tracking so the simulator can distinguish "same table, different
partition" (free retry) from "same table, same partition" (real I/O work).

A secondary goal is to quantify how multi-table transactions (`tables_per_txn
> 1`) interact with partition granularity: more tables per transaction
increases the probability of overlapping with a concurrent writer, but
partitions mitigate the cost of that overlap.

### Variants

| Config | Operation mix | Purpose |
|--------|--------------|---------|
| `exp5a_partition_fa.toml` | 100 % FastAppend | Isolate partition effect on pure-append workload |
| `exp5b_partition_mix.toml` | 90 % FA / 10 % VO | Show partition effect on ValidatedOverwrite retry cost (I/O convoy) |

### Swept Parameters (5-dimensional)

| Parameter | Values | Config key |
|-----------|--------|------------|
| `num_tables` | 10, 50 | `catalog.num_tables` |
| `partitions_per_table` | 1, 10, 100 | `catalog.partition.num_partitions` |
| `tables_per_txn` | 1, 3 | `transaction.tables_per_txn` |
| `catalog_latency` | 1, 50 ms | `catalog.service.latency_ms` |
| `arrival_rate` | LOAD_SWEEP (20..5000) | `inter_arrival.scale` |

Total configurations per variant (full sweep): 2 x 3 x 2 x 2 x 10 = 240
points, each run at 3+ seeds.

### Fixed Parameters

- Duration: 1 hour simulated
- Catalog: single-file (`num_groups = 1`), non-inlined (`table_metadata_inlined = false`)
- Catalog backend: `service` / `instant` (fixed-latency CAS)
- Storage: S3, `max_parallel = 4`
- Partition selection: 1 partition per table per txn (`partitions_per_txn = 1`), uniform random
- Retry limit: 10
- `real_conflict_probability`: 0.0 (exp5a), 0.0 (exp5b — conflict determined by partition overlap)
- Manifest list mode: `rewrite`

### Expected Patterns

- **More partitions -> fewer partition-level conflicts -> higher throughput.**
  With 100 partitions per table and 1 partition per txn, the probability of
  two concurrent writers hitting the same partition is ~1 %, versus 100 % at
  `partitions_per_table = 1`.
- **`tables_per_txn = 3` increases overlap probability** but partitions
  mitigate it: three tables x one partition each still rarely overlaps
  another writer's three tables x one partition.
- **Exp5b: VO benefits more from partitions than FA.** FastAppend conflict
  cost is always zero (no historical ML reads). ValidatedOverwrite pays an
  I/O convoy cost proportional to `n_snapshots_behind * n_partitions`. Fewer
  overlapping partitions means smaller convoy.
- **Cross-validation:** `partitions_per_table = 1, tables_per_txn = 1` should
  reproduce exp4a (FA) / exp4b (mix) results at matching `num_tables` and
  `catalog_latency`.

---

## 2. Analysis

### Correctness Checks

1. **Operation type invariant (exp5a):** Every row in `results.parquet` must
   have `operation_type == "fast_append"`. No `validation_exception` aborts.
2. **Conflict classification:** With `PartitionOverlapConflictDetector`
   active, the `tblptn_conflicts` column (partition-level real conflicts)
   should decrease as `partitions_per_table` increases.
3. **Conflict sum invariant:** `catalog_conflicts + tblptn_conflicts ==
   total_retries` for every transaction (already covered by existing test
   `test_conflict_sum_invariant_in_simulation`).
4. **Cross-validation:** At `partitions_per_table = 1`, the partition-aware
   detector degenerates to the same behavior as the probabilistic detector
   with `real_conflict_probability = 0.0` (everything is a false conflict).
   Throughput and latency should match exp4a/4b within seed variance.
5. **Partition column output:** `write_partition_ids` and `write_table_ids`
   columns in the parquet output should contain non-empty lists. Partition IDs
   should be in range `[0, partitions_per_table)`.

### Graphs Produced

**Exp5a (100 % FA):**

| Graph type | Grouping / axes | Filters | File suffix |
|------------|----------------|---------|-------------|
| `commit_rate_over_time` | default | — | — |
| `heatmap` (x4) | X = arrival_rate, Y = partitions_per_table; metrics: success, throughput, mean/p99 latency | CAS={1,50}ms x tpt={1,3} x T=10 | `cas_Xms_tptY_t10` |
| `latency_vs_throughput` (x4) | group_by partitions_per_table or tables_per_txn | CAS={1,50}ms, various tpt/ppt combos | various |

**Exp5b (90/10 mix):**

| Graph type | Grouping / axes | Filters | File suffix |
|------------|----------------|---------|-------------|
| `commit_rate_over_time` | default | — | — |
| `heatmap` (x4) | X = arrival_rate, Y = partitions_per_table; per_type_metrics: success_rate, mean_latency | CAS={1,50}ms x tpt={1,3} x T=10 | `cas_Xms_tptY_t10` |
| `latency_vs_throughput` (x2) | group_by partitions_per_table | CAS={1,50}ms, tpt=1, T=10 | `cas_Xms_tpt1_t10` |
| `operation_types` | group_by catalog_service_latency_ms | — | — |

### Key Metrics to Watch

- **Success rate vs. partitions_per_table:** Should approach 100 % as
  partitions grow (fewer real conflicts, more free retries).
- **P99 latency vs. partitions_per_table:** Tail latency should drop as
  partition granularity increases.
- **VO-specific success rate (exp5b):** Should show steeper improvement with
  partitions than FA, since VO's retry cost is dominated by I/O convoy.

---

## 3. Implementation

### New Functionality Compared to Exp1-4

Experiments 1-4 model each table as a single opaque unit. The conflict
detector is probabilistic (`ProbabilisticConflictDetector`): on CAS failure,
a coin flip at `real_conflict_probability` determines whether the conflict is
real or false. Exp5 replaces this with deterministic, structural detection.

#### 3a. Partition Version Tracking (`endive/catalog.py`)

Each table now stores a tuple of per-partition versions:

```
TableMetadata.partition_versions: Tuple[int, ...]
```

On successful commit, `catalog.commit()` increments
`partition_versions[pid] += 1` for each `(table_id, pid)` in the
`partitions_written` dict. `CatalogSnapshot.get_partition_version(table_id,
partition_id)` exposes this for conflict detection.

**Invariant:** Partition versions only advance on successful commit. They are
never skipped or reset.

#### 3b. Partition Overlap Conflict Detection (`endive/conflict_detector.py`)

`PartitionOverlapConflictDetector` replaces `ProbabilisticConflictDetector`
when `[partition].enabled = true` in config. On CAS failure it:

1. Iterates through `txn.partitions_written` (a `Dict[int, FrozenSet[int]]`
   mapping table_id to partition IDs).
2. For each (table_id, partition_id), compares the partition version at the
   txn's start snapshot against the current snapshot.
3. If any written partition was concurrently modified, AND the transaction
   type supports real conflicts (`can_have_real_conflict()` — only
   ValidatedOverwrite), returns `is_real_conflict = True`.
4. Otherwise returns `False` (false conflict — free retry).

FastAppend and MergeAppend always return `False` regardless of partition
overlap, matching Iceberg semantics.

#### 3c. Write Overlap and I/O Cost Scaling (`endive/transaction.py`)

`compute_write_overlap()` returns a structured `WriteOverlap` with the set of
overlapping partitions per table. This feeds into:

- **Per-attempt cost:** On first attempt, `n_partitions` = total partitions
  written. On retry, `n_partitions = overlap.n_partitions` (only overlapping
  partitions need manifest work). If no overlap at all, the retry is free
  (catalog read + re-CAS only).
- **Conflict cost:** `get_conflict_cost(n_partitions=overlap.n_partitions)`
  scales the I/O convoy (for VO) by the number of overlapping partitions,
  not total partitions.

#### 3d. Workload Generation (`endive/workload.py`)

When `[partition].enabled = true`:
- `UniformPartitionSelector` picks `partitions_per_txn` partitions per table
  uniformly at random.
- The resulting `partitions_written: Dict[int, FrozenSet[int]]` is passed to
  the `Transaction` constructor.

#### 3e. Config Integration (`endive/config.py`)

- `[partition].enabled = true` triggers `PartitionOverlapConflictDetector`
  creation (tested in `test_config_loading.py::test_partition_overlap_detector`).
- `catalog.partition.num_partitions` is extracted by
  `extract_key_parameters()` as `partitions_per_table` for the analysis
  pipeline.

### Readiness Assessment

**Ready to run: YES, with caveats.**

The core partition machinery (version tracking, overlap detection, cost
scaling, workload generation) is implemented and has good unit test coverage:

| Component | Test file | Coverage |
|-----------|----------|----------|
| Partition version tracking | `test_catalog.py::TestPartitionVersionTracking` | 3 tests across all catalog types |
| PartitionOverlapConflictDetector | `test_conflict_detector.py::TestPartitionOverlapConflictDetector` | 10 tests: same/different partition, multi-table, FA/MA immunity |
| Write overlap computation | `test_transaction_types.py::TestComputeWriteOverlap` | Multi-table/partition overlap scenarios |
| Scaled per-attempt cost | `test_transaction_types.py::TestScaledPerAttemptCost` | n_partitions scaling, ML+ mode |
| Scaled conflict cost | `test_transaction_types.py::TestScaledConflictCost` | VO I/O convoy, MA re-merge scaling |
| Config loading | `test_config_loading.py::test_partition_overlap_detector` | partition.enabled → detector creation |
| Parquet output columns | `test_simulation.py::TestNewSchemaColumns` | write_table_ids, write_partition_ids, conflict sums |
| Transaction commit lifecycle | `test_transaction_commit.py::TestOverlapScaling` | Multi-partition first attempt, partial overlap retry |

### Test Gaps and Recommendations

1. **No full-simulation integration test with partition config.** Unit tests
   exercise each component in isolation. An integration test that runs a
   short simulation (e.g. 60 s) with `partition.enabled = true`,
   `partitions_per_table = 10`, and verifies that `tblptn_conflicts` <
   `catalog_conflicts` would catch wiring bugs.

2. **Cross-validation test against exp4.** A regression test confirming that
   `partitions_per_table = 1` produces statistically indistinguishable
   results from exp4a (same seed, same parameters) would validate that the
   partition machinery doesn't introduce bias when it should be inert.

3. **Multi-table + multi-partition transaction lifecycle.** The existing
   `TestOverlapScaling` tests exercise 2-3 table scenarios, but do not test
   the full lifecycle with `tables_per_txn = 3` and `partitions_per_table =
   100` (the extreme of the sweep). A targeted test at this corner would
   increase confidence.

4. **Partition ID range validation.** No test asserts that generated
   partition IDs stay in `[0, partitions_per_table)`. A workload generation
   test should verify this.

5. **Analysis pipeline: `partitions_per_table` extraction.** The
   `extract_key_parameters()` function extracts this field
   (`saturation_analysis.py:665`), and it is used as a heatmap Y-axis and
   filter target. No unit test exercises the extraction path for this
   parameter. Incorrect extraction would produce empty or misaligned
   heatmaps.
