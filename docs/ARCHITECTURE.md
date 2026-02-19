# Iceberg Catalog Simulator - Architecture

## Overview

**Purpose**: Discrete-event simulation of Apache Iceberg-style optimistic concurrency control to explore commit latency tradeoffs and throughput limits in shared-storage catalog formats.

**Framework**: SimPy (discrete-event simulation)

**Core Components**:
- **Catalog**: Versioned state with monotonically increasing sequence number
- **Transactions**: Capture snapshot version at start, attempt CAS at commit
- **Conflict Resolution**: Distinguish false vs real conflicts, handle retries

## Module Structure

The simulator is organized into focused modules:

```
endive/
├── snapshot.py       # CatalogSnapshot, CASResult (immutable data types)
├── transaction.py    # Txn, LogEntry (transaction state)
├── main.py           # Catalog, ConflictResolver, simulation orchestration
├── config.py         # Configuration loading and validation
├── capstats.py       # Statistics collection
└── utils.py          # Utilities (git sha, table partitioning)
```

### Message-Passing Design

The key architectural principle is **message-passing semantics** between catalog (server) and transactions (client):

1. **CatalogSnapshot**: Immutable frozen dataclass capturing catalog state at a specific time
2. **CASResult**: Contains success/failure status plus server-time snapshot
3. **Catalog.read()**: Returns `CatalogSnapshot` with full RTT latency
4. **Catalog.try_cas()**: Returns `CASResult` with server-time state capture

Transactions **never access Catalog state directly**. All state is obtained via `read()` or `try_cas()` which return immutable snapshots. This ensures:
- State is captured at a specific point in time
- Cannot accidentally read stale or future state
- All reads have explicit latency costs

## Key Invariants

### Version Tracking (CRITICAL)

1. **Snapshot capture**: Transaction captures `catalog.seq` as `v_catalog_seq` at creation (S_i)
2. **CAS attempt**: On commit, checks against current `catalog.seq` (S_{i+n})
3. **On CAS failure** with `n = catalog.seq - v_catalog_seq`:
   - Read EXACTLY n manifest lists (one per intermediate snapshot)
   - Determine conflict type (false vs real)
   - Resolve conflicts appropriately
   - Update `v_catalog_seq` to current `catalog.seq`
   - Attempt to install `v_catalog_seq + 1`
4. **Monotonicity**: `catalog.seq` advances by exactly 1 on each successful commit
5. **Table versions**: Never decrease

**Implementation**: `endive/main.py:527` (txn_gen), `359-393` (try_CAS), `437` (version update)

### Table Grouping (CRITICAL)

1. **Partitioning**: Tables divided into G groups (1 ≤ G ≤ T)
2. **Transaction boundaries**: Transactions NEVER span group boundaries
3. **Conflict modes**:
   - `G=1`: Catalog-level (any concurrent writes conflict)
   - `G=T`: Table-level (only same-table writes conflict)
   - `1<G<T`: Group-level (multi-tenant isolation)
4. **Determinism**: Group assignment controlled by seed

**Implementation**: `endive/main.py:52-131` (partition), `460-497` (table selection), `369-384` (conflict detection)

### Partition-Level Modeling (Optional)

When `partition.enabled = true`, conflicts are detected at partition granularity:

1. **Per-table partitions**: Each table has N_PARTITIONS partitions
2. **Per-partition state**: `partition_seq[table_id][partition_id]` tracks versions
3. **Multi-partition transactions**: Transactions CAN span multiple partitions (unlike table groups)
4. **Vector clock CAS**: Only checks versions of touched partitions
5. **Partition selection**: Zipf (hot partitions) or uniform distribution

**Key difference from table groups**:
- Table groups: Transactions confined to ONE group
- Partitions: Transactions can span MULTIPLE partitions

**Configuration**:
```toml
[partition]
enabled = true
num_partitions = 100
partitions_per_txn_mean = 3.0
selection.distribution = "zipf"  # or "uniform"
selection.zipf_alpha = 1.5
```

**Implementation**: `endive/main.py` - `select_partitions()`, `_try_CAS_partition()`, `get_conflicting_partitions()`

### Conflict Resolution (NEW)

**False Conflict** - Version changed, no data overlap:
- **Cost**: Read metadata root only (~1ms with fast catalog)
- **No manifest file operations** required
- **Reason**: Changes didn't touch same data files

**Real Conflict** - Overlapping data changes:
- **Cost**: Read manifest list + read/write N manifest files (~400ms+)
- **Manifest count**: Sampled from configured distribution (exponential, uniform, or fixed)
- **Parallelism**: Batch operations respect `max_parallel` limit
- **Reason**: Must merge conflicting data file changes

**Probability**: Controlled by `real_conflict_probability` (0.0 to 1.0)

**Implementation**: `endive/main.py:627-719` (conflict resolution split)

### Exponential Backoff (NEW)

**Purpose**: Reduce wasted CAS attempts under high contention

**Mechanism**: After first CAS failure, transaction waits before retrying
- **Formula**: `delay = min(base_ms × multiplier^(retry-1), max_ms) × (1 ± jitter)`
- **Example**: 10ms → 20ms → 40ms → 80ms → 160ms → 320ms → capped at 5s
- **Jitter**: ±10% randomization prevents thundering herd

**Configuration** (in `[transaction.retry_backoff]` section):
- `enabled`: Boolean (default: false)
- `base_ms`: Initial delay in milliseconds (default: 10.0)
- `multiplier`: Growth factor per retry (default: 2.0)
- `max_ms`: Maximum delay cap (default: 5000.0)
- `jitter`: Randomization factor 0.0-1.0 (default: 0.1)

**Trade-offs**:
- **Benefit**: Fewer wasted CAS attempts, reduced catalog load
- **Cost**: Longer time to eventual success, increased latency
- **Hypothesis**: More beneficial with expensive real conflicts

**Implementation**: `endive/main.py:547-570` (calculate_backoff_time), `939-944` (apply in retry loop)

## Storage Model

### Latency Characteristics

All operations use normal distributions with mean ± stddev:

| Operation | Baseline (fast catalog) | Realistic S3 | Description |
|-----------|------------------------|--------------|-------------|
| T_CAS | 1ms | 1ms | Catalog pointer swap (atomic) |
| T_METADATA_ROOT (read/write) | 1ms | 50ms | Metadata file access |
| T_MANIFEST_LIST (read/write) | 50ms | 50ms | Manifest list file access |
| T_MANIFEST_FILE (read/write) | 50ms | 50ms | Manifest file access |

**MIN_LATENCY**: Enforced minimum (default: 1ms) to prevent unrealistic zeros

### Parallelism Model

- **MAX_PARALLEL**: Limits concurrent manifest operations (default: 4)
- **Batch processing**: Read MAX_PARALLEL manifest files at a time
- **Latency model**: `max(batch_latencies)` simulates parallel S3 reads
- **Example**: Reading 10 manifest files with MAX_PARALLEL=4:
  - Batch 1: max(4 latencies) → e.g., 58ms
  - Batch 2: max(4 latencies) → e.g., 52ms
  - Batch 3: max(2 latencies) → e.g., 49ms
  - Total: 159ms (vs 500ms+ sequential)

**Implementation**: `endive/main.py:443-452` (manifest list reading), `695-713` (manifest file reading/writing)

## Simulation Fidelity

### Accurately Modeled

✅ Optimistic concurrency control with CAS
✅ Snapshot isolation semantics
✅ Manifest list reading on conflict (n lists for n snapshots behind)
✅ **False vs real conflict distinction**
✅ **Variable manifest file merge cost**
✅ Stochastic storage latencies
✅ Parallel I/O with configurable limits
✅ Retry logic with exponential backoff
✅ Multi-table transactions with table-level isolation
✅ **Partition-level conflicts** (per-partition manifest lists, Zipf/uniform access)

### Simplified

⚠️ Actual data merging (simulated with latency)
⚠️ Manifest file structure (not explicitly modeled)
⚠️ Network failures/retries (latencies are deterministic from distributions)
⚠️ Garbage collection/compaction
⚠️ Read queries (only write transactions simulated)
⚠️ Deletion vectors (cost lumped into manifest operations)

### Parameterized

🔧 Storage latencies (configurable distributions)
🔧 Workload patterns (Zipf distributions for table access)
🔧 Transaction runtime (lognormal distribution)
🔧 Client arrival patterns (exponential, uniform, normal, fixed)
🔧 Conflict types (false vs real probability)
🔧 Manifest count distribution (for real conflicts)

## Critical Code Locations

### Transaction Lifecycle

```python
# 1. Transaction generation
# endive/main.py:527-580 (txn_gen)
txn = Txn(...)
txn.v_catalog_seq = catalog.seq  # Capture snapshot version
yield env.process(txn.execute())  # Execute transaction work
yield env.process(txn.commit())   # Attempt commit

# 2. CAS attempt
# endive/main.py:585-625 (Txn.commit)
success, v_current = yield catalog.try_CAS(txn)
while not success and txn.retries < MAX_RETRIES:
    yield ConflictResolver.resolve_conflicts(...)
    success, v_current = yield catalog.try_CAS(txn)

# 3. Conflict resolution
# endive/main.py:627-719 (ConflictResolver)
def merge_table_conflicts(sim, txn, v_catalog):
    for t, v in txn.v_dirty.items():
        if v_catalog[t] != v:
            is_real = random.random() < REAL_CONFLICT_PROBABILITY
            if is_real:
                yield from resolve_real_conflict(...)
            else:
                yield from resolve_false_conflict(...)
```

### Conflict Resolution Detail

**False Conflict** (`resolve_false_conflict` @ line 649):
```python
# Read metadata root to understand new snapshot
yield sim.timeout(get_metadata_root_latency('read'))

# Update validation version (no file operations needed)
txn.v_dirty[table_id] = v_catalog[table_id]
```

**Real Conflict** (`resolve_real_conflict` @ line 671):
```python
# Sample number of conflicting manifest files
n_conflicting = sample_conflicting_manifests()

# Read metadata root
yield sim.timeout(get_metadata_root_latency('read'))

# Read manifest list
yield sim.timeout(get_manifest_list_latency('read'))

# Read conflicting manifest files (with parallelism)
for batch in batches(n_conflicting, MAX_PARALLEL):
    batch_latencies = [get_manifest_file_latency('read') for _ in batch]
    yield sim.timeout(max(batch_latencies))

# Write merged manifest files (with parallelism)
for batch in batches(n_conflicting, MAX_PARALLEL):
    batch_latencies = [get_manifest_file_latency('write') for _ in batch]
    yield sim.timeout(max(batch_latencies))

# Write updated manifest list
yield sim.timeout(get_manifest_list_latency('write'))

# Update validation version
txn.v_dirty[table_id] = v_catalog[table_id]
```

### Configuration Loading

**Location**: `endive/main.py:134-240` (configure_from_toml)

**Key parameters**:
```python
# Conflict type configuration
REAL_CONFLICT_PROBABILITY = cfg['transaction']['real_conflict_probability']

# Manifest distribution configuration
CONFLICTING_MANIFESTS_DIST = cfg['transaction']['conflicting_manifests']['distribution']
CONFLICTING_MANIFESTS_PARAMS = {
    'mean': cfg['transaction']['conflicting_manifests']['mean'],
    'min': cfg['transaction']['conflicting_manifests']['min'],
    'max': cfg['transaction']['conflicting_manifests']['max']
}
```

## Statistics Collection

**Location**: `endive/capstats.py`

**Per-transaction metrics**:
- `t_submit`: Arrival time
- `t_start`: Start of transaction work
- `t_commit_start`: Start of commit attempt
- `t_commit_end`: End of commit (success or abort)
- `status`: 'committed' or 'aborted'
- `n_retries`: Number of retry attempts
- `commit_latency`: Time spent in commit protocol
- `total_latency`: `t_commit_end - t_submit`
- `t_runtime`: Transaction execution time

**Aggregate statistics**:
- `STATS.false_conflicts`: Count of false conflicts resolved
- `STATS.real_conflicts`: Count of real conflicts resolved
- `STATS.manifest_files_read`: Total manifest files read
- `STATS.manifest_files_written`: Total manifest files written

**Analysis-computed metrics**:
- `overhead_pct = (commit_latency / total_latency) × 100`
- Success rate, throughput, latency percentiles

## Testing Strategy

**Total tests**: 63

### Core Simulator (endive/main.py)
- Determinism (same seed → identical results)
- Parameter effects (more load → more retries)
- Conflict type distribution
- Manifest count sampling

### Conflict Resolution (endive/main.py:627-719)
- False conflict handling
- Real conflict handling
- Parallel manifest operations
- Stochastic behavior validation

### Table Grouping (endive/main.py:52-131)
- Partitioning correctness
- Transaction boundary enforcement
- Deterministic assignment

### Snapshot Versioning (endive/main.py:359-452)
- Version capture at start
- CAS correctness
- Manifest list reading (exactly n lists)
- Multi-retry version progression

### Analysis Pipeline (endive/saturation_analysis.py)
- Parameter extraction from configs
- Multi-seed aggregation
- Warmup period filtering
- Overhead computation

### Distribution Conformance (tests/test_distribution_conformance.py)
- Transaction runtime matches lognormal(mean, sigma) + min
- Inter-arrival matches exponential(scale)
- Commit latency behavior validation

## Common Pitfalls to Avoid

1. **Breaking determinism**: All randomness must use `np.random` with seed control
2. **Violating group boundaries**: Never let transactions span groups
3. **Version monotonicity**: `catalog.seq` and table versions never decrease
4. **Manifest count**: Must read EXACTLY n lists for n snapshots behind
5. **MIN_LATENCY**: Always enforce minimum to prevent unrealistic zeros
6. **Parallel I/O**: Batch operations must respect MAX_PARALLEL limit
7. **Conflict type**: Don't confuse false (cheap) vs real (expensive) conflict handling

## Configuration Guidelines

### Infinitely Fast Catalog (Baseline)

```toml
[storage]
T_CAS.mean = 1
T_CAS.stddev = 0.1
T_METADATA_ROOT.read.mean = 1
T_METADATA_ROOT.read.stddev = 0.1
T_METADATA_ROOT.write.mean = 1
T_METADATA_ROOT.write.stddev = 0.1
```

### Realistic S3 Storage

```toml
[storage]
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6
T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
```

### Realistic Transaction Runtime

```toml
[transaction]
# Typical Iceberg transactions: 30s - 3min
runtime.min = 30000    # 30 seconds
runtime.mean = 180000  # 3 minutes
runtime.sigma = 1.5    # Lognormal shape parameter
```

### False Conflicts Only (Baseline)

```toml
[transaction]
real_conflict_probability = 0.0
```

### Real Conflicts

```toml
[transaction]
real_conflict_probability = 0.5  # 50% of conflicts are real

# Distribution of conflicting manifests
conflicting_manifests.distribution = "exponential"
conflicting_manifests.mean = 3.0
conflicting_manifests.min = 1
conflicting_manifests.max = 10
```

## Future Enhancement Opportunities

1. **Configuration validation** before simulation runs (check parameter ranges)
2. **Event logging** for detailed post-simulation analysis (CSV trace file)
3. **Storage backend abstractions** for different cloud providers
4. **Read-only transactions** (no commit, only read snapshots)
5. **Batch commit optimization** (group multiple transactions)
6. **Deletion vector modeling** (separate from manifest operations)
7. **Compaction simulation** (background maintenance operations)

## Performance Characteristics

- **Simulation speed**: ~1000× real-time (1 hour in ~3.6 seconds wall-clock)
- **Memory usage**: ~100MB for 1-hour simulation with 20,000 transactions
- **Parallelization**: Linear speedup with multiple cores for independent experiments
- **Determinism**: Exact reproducibility with same seed and code version

## References

- **SimPy Documentation**: https://simpy.readthedocs.io/
- **Apache Iceberg Spec**: https://iceberg.apache.org/spec/
- **Simulator Internals**: [`SIMULATOR.md`](SIMULATOR.md) - Detailed pseudocode and latency models
- **Snapshot Versioning**: [`SNAPSHOT_VERSIONING.md`](SNAPSHOT_VERSIONING.md)
- **Analysis Plan**: [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)
- **Baseline Results**: [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md)
