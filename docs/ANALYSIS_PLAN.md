# Analysis Plan: Iceberg Transaction Throughput Limits

## Current Status (Updated 2025-01)

### ✅ Completed

**Phase 1 - Simulator Implementation:**
- ✅ Real vs false conflict distinction implemented
- ✅ Variable conflicting manifest distribution
- ✅ Statistics tracking (false_conflicts, real_conflicts, manifest I/O)
- ✅ Configuration parameters working

**Phase 2 - Baseline Experiments (False Conflicts):**
- ✅ Exp 2.1: Single-table saturation (9 loads × 5 seeds = 45 runs)
- ✅ Exp 2.2: Multi-table scaling (54 configs × 5 seeds = 270 runs... but only 9+54 unique configs run)
- ✅ Analysis: Saturation curves, overhead measurement, markdown tables
- ✅ Docker support for containerized execution

### 📊 Key Findings

**Question 1a (Single-table, false conflicts)**: ✅ **ANSWERED**
- Peak throughput: ~60 commits/sec
- Saturation: 55-60 c/s (50% success rate)
- Overhead at saturation: 26% (commit protocol takes 1/4 of time)
- **Bottleneck**: Contention, not catalog speed

**Question 2a (Multi-table, false conflicts)**: ✅ **ANSWERED**
- Throughput scales sub-linearly: 50× tables → 2.3× throughput
- Sweet spot: 10-20 tables
- **Latency paradox**: More tables = higher throughput but WORSE tail latency
- Overhead at 50 tables: 59% (commit takes MORE time than transaction!)

### 🔬 Ready to Run

**Phase 3 - Real Conflict Experiments:**
- ⏳ Exp 3.1: Single-table real conflicts (Question 1b)
- ⏳ Exp 3.2: Manifest count distribution variance
- ⏳ Exp 3.3: Multi-table real conflicts (Question 2b)

**Phase 4 - Exponential Backoff Experiments (NEW):**
- ⏳ Exp 4.1: Single-table false conflicts with backoff (Question 4a)
- ⏳ Exp 3.4: Multi-table real conflicts with backoff (Question 4b)

**Outstanding questions:**
- How do real conflicts shift saturation point?
- Cost difference: false (~1ms) vs real (~400ms)?
- How do real conflicts compound in multi-table transactions?
- Does exponential backoff improve performance under contention?
- Is backoff more beneficial with expensive real conflicts?

---

## Research Goal
Characterize workloads that could (not) be supported by a storage-only catalog and understand the fundamental per-table throughput limits of Iceberg-style optimistic concurrency control.

## Questions to Answer

### 1. Single Table Saturation (Infinitely Fast Catalog)

**1a. False Conflicts Only**
- **Setup**: Single table, all conflicts are "false" (version changed but no overlapping data)
- **False conflict cost**: Read manifest lists only, no manifest file rewriting
- **Goal**: Find saturation point (abort rate > 50%) as offered load increases
- **Output**: Latency vs throughput graph, each point annotated with % successful

**1b. Real Conflicts with Variable Cost**
- **Setup**: Same as 1a, but add probability of "real" conflicts
- **Real conflict cost**: Must read/rewrite manifest files, update deletion vectors
- **Variable**: Number of conflicting manifests drawn from distribution
- **Goal**: Show how real conflicts shift saturation point
- **Output**: Compare false vs real conflict saturation curves

### 2. Multi-Table Saturation (Infinitely Fast Catalog)

**2a. Multi-Table False Conflicts**
- **Setup**: N tables, multi-table transactions, all false conflicts
- **Cost**: Manifest lists from multiple tables, respecting parallelism limit
- **Goal**: Understand how table count affects throughput
- **Output**: Saturation curves for different N

**2b. Multi-Table Real Conflicts**
- **Setup**: Same as 2a, but with real conflict probability
- **Effect**: P(≥1 real conflict) = 1 - (1 - p)^n_tables → compounds
- **Goal**: Show interaction between table count and conflict type
- **Output**: Heatmap of throughput vs (num_tables, real_conflict_prob)

---

## Critical Assessment of Current Simulator

### ✅ What the Simulator Does Well

1. **Version tracking**: Transactions capture S_i, check against S_{i+n}, read n manifest lists
2. **Retry logic**: Full retry mechanism with correct version progression
3. **Parallelism**: MAX_PARALLEL correctly limits concurrent manifest operations
4. **Multi-table**: Transactions can access multiple tables with proper conflict detection
5. **Configurable latencies**: All storage operations have realistic latency distributions

### ✅ Previously Identified Gaps (NOW IMPLEMENTED)

#### Gap 1: Real vs False Conflict Distinction → ✅ IMPLEMENTED

**Implementation** (`icecap/main.py:627-719`):
```python
def merge_table_conflicts(sim, txn, v_catalog):
    for t, v in txn.v_dirty.items():
        if v_catalog[t] != v:
            # Determine if this is a real conflict
            is_real_conflict = np.random.random() < REAL_CONFLICT_PROBABILITY

            if is_real_conflict:
                yield from ConflictResolver.resolve_real_conflict(sim, txn, t, v_catalog)
                STATS.real_conflicts += 1
            else:
                yield from ConflictResolver.resolve_false_conflict(sim, txn, t, v_catalog)
                STATS.false_conflicts += 1
```

**False conflict** (line 649):
- Reads metadata root only
- No manifest file operations
- Cost: ~1ms (with infinitely fast catalog)

**Real conflict** (line 671):
- Samples conflicting manifest count from distribution
- Reads manifest list
- Reads/writes N manifest files (respects MAX_PARALLEL)
- Cost: ~400ms+ depending on N

**Now can answer**:
- ✅ Question 1a: False conflicts only (real_conflict_probability = 0.0)
- ✅ Question 1b: Variable real conflict probability (0.0 to 1.0)
- ✅ Cost measurement: Statistics track false vs real conflicts separately

#### Gap 2: Variable Conflict Resolution Cost → ✅ IMPLEMENTED

**Implementation** (`icecap/main.py:235-250`):
```python
def sample_conflicting_manifests() -> int:
    dist = CONFLICTING_MANIFESTS_DIST
    params = CONFLICTING_MANIFESTS_PARAMS

    if dist == "fixed":
        return int(params['value'])
    elif dist == "exponential":
        value = np.random.exponential(params['mean'])
        return max(params['min'], min(params['max'], int(value)))
    elif dist == "uniform":
        return np.random.randint(params['min'], params['max'] + 1)
```

**Configuration**:
```toml
conflicting_manifests.distribution = "exponential"  # or "uniform", "fixed"
conflicting_manifests.mean = 3.0
conflicting_manifests.min = 1
conflicting_manifests.max = 10
```

**Now can model**: Realistic variance in manifest merge cost

#### Gap 3: Manifest List Already Read → ✅ RESOLVED

**Current behavior**:
- `read_manifest_lists()` reads n lists for n snapshots behind (line 443-452)
- For **false conflicts**: Only read metadata root (line 665)
- For **real conflicts**: Read manifest list for file pointers (line 693)

**This is correct**: False conflicts skip redundant manifest list reads

### ✅ What Works for Baseline Experiments

Good news: Can immediately run some experiments with current simulator:

1. ✅ **"Infinitely fast catalog"**: Set `T_CAS.mean = 1` (essentially instant CAS)
2. ✅ **Single table**: Set `num_tables = 1`
3. ✅ **Multi-table**: Already supported with `num_tables = N`
4. ✅ **Load sweep**: Vary `inter_arrival.scale` to increase offered load
5. ✅ **Measurement**: Already capture latency, throughput, success rate, retries

**Can answer TODAY**: Questions 1a and 2a if we interpret them as "current conflict model" (which is between false and real)

---

## Implementation Plan

### Phase 0: Clarify Definitions (Discussion)

Before implementing, need to clarify:

1. **"Infinitely fast catalog"** - Interpretation:
   - ✅ Catalog metadata operations (CAS) near-instant: `T_CAS = 1ms`
   - ✅ Storage operations (manifest lists/files) still have realistic latency
   - ❓ Metadata root operations? Suggest same as CAS (near-instant)

2. **"False conflict"** - Iceberg definition:
   - Version changed (catalog moved from S_i to S_{i+n})
   - But no overlapping file modifications
   - Result: Read manifest lists to understand changes, merge metadata, retry
   - NO manifest file rewriting needed

3. **"Real conflict"** - Iceberg definition:
   - Overlapping files were added/deleted/modified
   - Result: Must read manifest files, merge/rewrite them, update deletion vectors
   - Much more expensive

4. **Saturation definition** - Suggest:
   - Abort rate > 50% (as stated), OR
   - Throughput plateau (marginal increase < 1%), OR
   - Latency explosion (p99 > 10x p50)

### Phase 1: Extend Simulator - Conflict Type Modeling (HIGH PRIORITY)

#### 1.1 Add Configuration Parameters

```toml
[transaction]
# Conflict resolution configuration
real_conflict_probability = 0.0  # 0.0 = all false, 1.0 = all real

# For real conflicts: number of manifest files that need merging
conflicting_manifests.distribution = "exponential"  # "exponential", "uniform", "fixed"
conflicting_manifests.mean = 3.0
conflicting_manifests.min = 1
conflicting_manifests.max = 10
```

#### 1.2 Refactor ConflictResolver

**Split `merge_table_conflicts()` into two methods:**

```python
@staticmethod
def merge_table_conflicts(sim, txn: 'Txn', v_catalog: dict):
    """Merge conflicts for tables that have changed."""
    for t, v in txn.v_dirty.items():
        if v_catalog[t] != v:
            # Determine if this is a real conflict
            is_real_conflict = np.random.random() < REAL_CONFLICT_PROBABILITY

            if is_real_conflict:
                yield from ConflictResolver.resolve_real_conflict(sim, txn, t, v_catalog)
            else:
                yield from ConflictResolver.resolve_false_conflict(sim, txn, t, v_catalog)

@staticmethod
def resolve_false_conflict(sim, txn, table_id, v_catalog):
    """Resolve false conflict (version changed, no data overlap).

    Manifest lists were already read in read_manifest_lists().
    Just need to update metadata and validation version.
    """
    # Read metadata root to understand new snapshot
    yield sim.timeout(get_metadata_root_latency('read'))

    # Update validation version (no file operations needed)
    txn.v_dirty[table_id] = v_catalog[table_id]

    logger.debug(f"{sim.now} TXN {txn.id} Resolved false conflict for table {table_id}")

@staticmethod
def resolve_real_conflict(sim, txn, table_id, v_catalog):
    """Resolve real conflict (overlapping data changes).

    Must read and rewrite manifest files.
    """
    logger.debug(f"{sim.now} TXN {txn.id} Resolving real conflict for table {table_id}")

    # Determine number of conflicting manifest files
    n_conflicting = sample_conflicting_manifests()

    # Read metadata root
    yield sim.timeout(get_metadata_root_latency('read'))

    # Read manifest list (to get pointers to manifest files)
    yield sim.timeout(get_manifest_list_latency('read'))

    # Read conflicting manifest files (respects MAX_PARALLEL)
    for batch_start in range(0, n_conflicting, MAX_PARALLEL):
        batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
        batch_latencies = [get_manifest_file_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    # Write merged manifest files
    for batch_start in range(0, n_conflicting, MAX_PARALLEL):
        batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
        batch_latencies = [get_manifest_file_latency('write') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    # Write updated manifest list
    yield sim.timeout(get_manifest_list_latency('write'))

    # Update validation version
    txn.v_dirty[table_id] = v_catalog[table_id]
```

#### 1.3 Add Helper Function

```python
def sample_conflicting_manifests() -> int:
    """Sample number of conflicting manifest files from configured distribution."""
    dist = CONFLICTING_MANIFESTS_DIST
    params = CONFLICTING_MANIFESTS_PARAMS

    if dist == "fixed":
        return int(params['value'])
    elif dist == "exponential":
        value = np.random.exponential(params['mean'])
        return max(params['min'], min(params['max'], int(value)))
    elif dist == "uniform":
        return np.random.randint(params['min'], params['max'] + 1)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
```

#### 1.4 Update Statistics

Track conflict types in `Stats` class:

```python
class Stats:
    def __init__(self):
        # ... existing fields ...
        self.false_conflicts = 0
        self.real_conflicts = 0
        self.manifest_files_read = 0
        self.manifest_files_written = 0
```

### Phase 2: Baseline Experiments - False Conflicts Only

#### Experiment 2.1: Single Table Saturation (Question 1a)

**Configuration:**
```toml
[simulation]
duration_ms = 100000  # 100 seconds

[catalog]
num_tables = 1
num_groups = 1  # Catalog-level conflicts

[transaction]
retry = 10
real_conflict_probability = 0.0  # FALSE CONFLICTS ONLY

# Simple workload: single table access
ntable.zipf = 10.0  # Ensure exactly 1 table per txn

[storage]
max_parallel = 4
min_latency = 1

# Infinitely fast catalog
T_CAS.mean = 1
T_CAS.stddev = 0.1

T_METADATA_ROOT.read.mean = 1
T_METADATA_ROOT.read.stddev = 0.1
T_METADATA_ROOT.write.mean = 1
T_METADATA_ROOT.write.stddev = 0.1

# Realistic storage latencies
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6

T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
```

**Load sweep:**
```python
inter_arrival_times = [
    10,    # ~100 txn/sec offered
    20,    # ~50 txn/sec
    50,    # ~20 txn/sec
    100,   # ~10 txn/sec
    200,   # ~5 txn/sec
    500,   # ~2 txn/sec
    1000,  # ~1 txn/sec
    2000,  # ~0.5 txn/sec
    5000,  # ~0.2 txn/sec
]

# For each value, run simulation and measure:
# - Achieved throughput (commits/sec)
# - Latency distribution (mean, p50, p95, p99)
# - Success rate (% committed)
# - Mean retries per transaction
```

**Expected results:**
- **Low load** (5000ms): ~99% success, low latency (~150ms), linear throughput
- **Medium load** (200-500ms): Success drops to 90-95%, latency increases, throughput plateaus
- **High load** (10-50ms): Success < 50%, latency explodes, throughput may decrease

**Saturation point**: Where throughput stops increasing despite higher offered load

#### Experiment 2.2: Multi-Table Saturation (Question 2a)

**Vary number of tables:**
```python
num_tables_configs = [1, 2, 5, 10, 20, 50]

# For each num_tables:
# - Set num_groups = num_tables (table-level conflicts)
# - Configure transactions to access 2-3 tables on average (ntable.zipf = 1.5)
# - Sweep offered load
# - Compare saturation points
```

**Hypothesis:**
- More tables → less contention → higher saturation point
- But multi-table transactions read more manifest lists
- Parallelism limit (MAX_PARALLEL = 4) becomes bottleneck

### Phase 3: Real Conflict Experiments

#### Experiment 3.1: Real Conflict Impact (Question 1b)

**Sweep real_conflict_probability:**
```python
real_conflict_probs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# For each probability:
# - Single table setup
# - Sweep offered load
# - Measure saturation point

# Compare how saturation shifts with real conflict probability
```

**Expected results:**
- Higher prob → earlier saturation
- Much higher retry latency (manifest file operations)
- Lower maximum achievable throughput

#### Experiment 3.2: Conflicting Manifest Count (Question 1b continued)

**Vary distribution of conflicting manifests:**
```python
configs = [
    {"distribution": "fixed", "value": 1},
    {"distribution": "fixed", "value": 5},
    {"distribution": "fixed", "value": 10},
    {"distribution": "exponential", "mean": 3, "min": 1, "max": 20},
]

# For each config:
# - Set real_conflict_probability = 0.5
# - Sweep offered load
# - Show how variance affects saturation
```

#### Experiment 3.3: Multi-Table Real Conflicts (Question 2b)

**Combine table count and real conflicts:**
```python
num_tables_sweep = [1, 2, 5, 10, 20]
real_conflict_probs = [0.0, 0.1, 0.3, 0.5]

# 20 experiments: for each (num_tables, prob) pair:
# - Run load sweep
# - Measure saturation point

# Generate heatmap showing interaction effects
```

**Key insight**: P(≥1 real conflict in multi-table txn) = 1 - (1 - p)^n_tables

### Phase 4: Exponential Backoff Experiments

#### Experiment 4.1: Backoff Impact on False Conflicts (Question 4a)

**Research Question**: Does exponential backoff improve performance under contention with cheap false conflicts?

**Comparison with Exp 2.1**:
- Same setup: single table, false conflicts only, fast catalog
- NEW: Exponential backoff enabled (10ms base, 2x multiplier, 5s max, 10% jitter)

**Configuration**:
```toml
[transaction.retry_backoff]
enabled = true
base_ms = 10.0        # Start with 10ms delay
multiplier = 2.0      # Double each retry: 10, 20, 40, 80, 160, ...
max_ms = 5000.0       # Cap at 5 seconds
jitter = 0.1          # ±10% randomization
```

**Hypothesis**:
- Backoff reduces wasted CAS attempts during high contention
- Trade-off: Lower retry overhead vs longer time to success
- Success rate curves may shift right (higher saturation point)
- P95/P99 latency may improve near saturation

**Expected results**:
- Modest improvement (false conflicts are cheap ~1ms)
- Benefit mainly at high load (near saturation)
- Backoff delay < conflict resolution time

#### Experiment 4.2 (3.4): Backoff with Real Conflicts (Question 4b)

**Research Question**: Does backoff provide MORE benefit with expensive real conflicts?

**Comparison with Exp 3.3**:
- Same setup: multi-table, real conflicts, table-level isolation
- NEW: Exponential backoff enabled (same parameters as 4.1)

**Sweep parameters**:
```python
num_tables = [1, 2, 5, 10, 20]
real_conflict_probs = [0.0, 0.1, 0.3, 0.5]
inter_arrival_scale = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
```

**Hypothesis**:
- Backoff MORE beneficial with real conflicts (~400ms+ cost)
- At high contention + real conflicts: backoff delay < conflict resolution
- Success rate improvements larger than exp4.1
- Key trade-off: 5s max backoff vs 400ms+ conflict resolution

**Expected results**:
- Larger improvements than exp4.1 (expensive retries)
- Benefit scales with real_conflict_probability
- Most impactful at high table count (compound conflicts)

**Key metrics to compare**:
- Success rate at each load level (with/without backoff)
- P95 latency improvement
- Mean retries per transaction
- Saturation throughput shift

### Phase 5: Visualization and Analysis

#### 4.1 Primary Result: Latency vs Throughput Curves

**For each experiment:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Plot p50, p95, p99 latency vs achieved throughput
for percentile in [50, 95, 99]:
    ax.plot(throughputs, latencies[percentile],
            label=f'P{percentile}', marker='o')

    # Annotate each point with success rate
    for i, (tp, lat) in enumerate(zip(throughputs, latencies[percentile])):
        ax.annotate(f"{success_rates[i]:.0f}%",
                   (tp, lat), textcoords="offset points",
                   xytext=(0,5), ha='center', fontsize=8)

ax.set_xlabel('Achieved Throughput (commits/sec)')
ax.set_ylabel('Commit Latency (ms)')
ax.set_title('Single Table Saturation - False Conflicts Only')
ax.legend()
ax.grid(True, alpha=0.3)

# Mark saturation point (50% success rate)
saturation_idx = np.argmin(np.abs(success_rates - 50))
ax.axvline(throughputs[saturation_idx], color='red',
           linestyle='--', label='50% Success Rate')

plt.savefig('latency_vs_throughput_1a.png', dpi=300)
```

#### 4.2 Comparative Analysis: False vs Real Conflicts

**Overlay curves for different real_conflict_probability:**
```python
fig, ax = plt.subplots(figsize=(12, 7))

for prob in [0.0, 0.1, 0.3, 0.5, 1.0]:
    data = results[prob]
    ax.plot(data['throughput'], data['p95_latency'],
            label=f'Real conflict prob={prob}',
            marker='o', linewidth=2)

ax.set_xlabel('Achieved Throughput (commits/sec)')
ax.set_ylabel('P95 Commit Latency (ms)')
ax.set_title('Impact of Real Conflicts on Saturation')
ax.legend()
ax.grid(True, alpha=0.3)

# Highlight how saturation point shifts left
for prob in [0.0, 0.5, 1.0]:
    sat_throughput = find_saturation_point(results[prob])
    ax.axvline(sat_throughput, linestyle=':', alpha=0.5)
    ax.text(sat_throughput, ax.get_ylim()[1] * 0.9,
            f'p={prob}', rotation=90)

plt.savefig('false_vs_real_conflicts.png', dpi=300)
```

#### 4.3 Multi-Table Scaling Analysis

**Heatmap: Saturation throughput vs (num_tables, real_conflict_prob):**
```python
import seaborn as sns

# Build matrix of saturation throughputs
matrix = np.zeros((len(num_tables_configs), len(real_conflict_probs)))
for i, n_tables in enumerate(num_tables_configs):
    for j, prob in enumerate(real_conflict_probs):
        matrix[i, j] = compute_saturation_throughput(results[n_tables][prob])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt='.1f',
            xticklabels=[f'{p:.1f}' for p in real_conflict_probs],
            yticklabels=num_tables_configs,
            cmap='YlOrRd', ax=ax)

ax.set_xlabel('Real Conflict Probability')
ax.set_ylabel('Number of Tables')
ax.set_title('Saturation Throughput (commits/sec)\nTable-Level Conflicts')

plt.savefig('multi_table_saturation_heatmap.png', dpi=300)
```

#### 4.4 Summary Statistics Table

**Generate CSV with key metrics:**
```python
import pandas as pd

summary = []
for experiment_name, result in results.items():
    summary.append({
        'experiment': experiment_name,
        'max_throughput': result['throughput'].max(),
        'saturation_throughput': find_saturation_point(result),
        'p50_latency_at_saturation': result['p50_latency'][saturation_idx],
        'p99_latency_at_saturation': result['p99_latency'][saturation_idx],
        'success_rate_at_saturation': result['success_rate'][saturation_idx],
        'mean_retries_at_saturation': result['mean_retries'][saturation_idx],
    })

df = pd.DataFrame(summary)
df.to_csv('saturation_summary.csv', index=False)
print(df.to_string())
```

---

## Additional Analyses (Beyond Core Questions)

### 5. Retry Cascade Dynamics

**Question**: How do retries create cascading conflicts?

**Scenario**:
- Transaction T1 starts at S_0
- While T1 prepares, catalog moves to S_5
- T1 reads 5 manifest lists, updates to S_5, attempts CAS
- But catalog moved to S_8 (other txns committed during T1's retry)
- T1 must read 3 MORE manifest lists → cascading retry

**Experiment**:
```python
# Track retry depth distribution
# Measure: fraction of transactions requiring 1, 2, 3, 4+ retry attempts
# Analyze: under what load do cascades become common?

# Plot: Retry depth histogram at different offered loads
```

**Insight**: Cascades create "convoy effect" - transactions get stuck in retry loops

### 6. Parallelism Limit Impact

**Question**: How does MAX_PARALLEL affect saturation throughput?

**Experiment**:
```python
max_parallel_configs = [1, 2, 4, 8, 16, 32]

# For each config:
# - Single table, false conflicts only
# - Sweep offered load
# - Measure saturation point

# Expected: Returns diminish after some point (storage throughput limit)
```

### 7. Hotspot Analysis

**Question**: What happens with skewed workload (hot tables)?

**Experiment**:
```python
# Configuration:
num_tables = 20
num_groups = 20  # Table-level conflicts

# Vary table selection skew
seltbl_zipf_configs = [1.0, 1.4, 2.0, 2.5, 3.0]

# 1.0 = uniform (all tables equally likely)
# 3.0 = extreme hotspot (80% of txns hit same table)

# Measure saturation with different skew levels
```

**Hypothesis**: Higher skew → earlier saturation (hot table becomes bottleneck)

### 8. Transaction Runtime Impact

**Question**: Do longer-running transactions cause more conflicts?

**Experiment**:
```python
# Vary transaction runtime
runtime_means = [100, 500, 1000, 5000, 10000]  # ms

# Hypothesis: Longer runtime → transaction more likely to be behind
# → more retries → lower throughput
```

### 9. Storage Latency Sensitivity

**Question**: How sensitive is saturation to storage performance?

**Experiment**:
```python
storage_configs = [
    {"name": "S3 Express",     "mean": 10,  "stddev": 2},
    {"name": "S3 Standard",    "mean": 50,  "stddev": 10},
    {"name": "S3 Cross-region","mean": 200, "stddev": 40},
]

# For each storage system:
# - Run saturation experiment
# - Compare maximum achievable throughput
```

**Expected**: Faster storage → higher saturation point, but fundamental concurrency limit remains

### 10. Group Size Optimization

**Question**: Is there an optimal group size between catalog-level (G=1) and table-level (G=N)?

**Experiment**:
```python
# Configuration:
num_tables = 20
num_groups_configs = [1, 2, 5, 10, 20]  # 1=catalog, 20=table-level

# For each group size:
# - Measure saturation throughput
# - Analyze false conflict rate
```

**Insight**: Group size is a knob for tuning false conflict vs coordination overhead

---

## Implementation Priority

### HIGH PRIORITY (Required for stated questions)

1. ✅ **Infinitely fast catalog config** - Set CAS/metadata to 1ms (can do now)
2. ⚠️ **Real vs false conflict distinction** - Must implement (Phase 1)
3. ⚠️ **Variable conflicting manifests** - Must implement (Phase 1)
4. ✅ **Load sweep experiments** - Can do now with existing framework
5. ⚠️ **Visualization scripts** - Extend analysis.py (Phase 4)

### MEDIUM PRIORITY (Illuminates broader questions)

6. Retry cascade analysis
7. Parallelism sensitivity
8. Hotspot analysis (can do now with zipf)
9. Storage latency sensitivity (can do now)

### LOW PRIORITY (Advanced topics)

10. Group size optimization
11. Transaction runtime impact
12. Read-only transaction modeling
13. Batch commit optimization

---

## Validation Strategy

### Sanity Checks

1. **Zero load**: Success rate should be ~100%, latency = runtime + CAS
2. **Infinite load**: Success rate → 0%, latency → infinity (all retries exhausted)
3. **False conflicts**: Should be cheaper than real conflicts
4. **More tables**: Should reduce contention (table-level) or no effect (catalog-level)

### Comparison Points

**Theoretical lower bound (false conflicts only, single table):**
```
Max throughput ≈ 1 / (T_cas + T_manifest_list_write)
              ≈ 1 / (1ms + 60ms)
              ≈ 16 commits/sec
```

**But**: With retries, actual saturation will be much lower

### Cross-Validation

Run same experiment with different seeds → results should be statistically similar

---

## Deliverables

### Code Deliverables

1. ✅ Extended `icecap/main.py` with false/real conflict modeling
2. ✅ New experiment script: `icecap/saturation_analysis.py`
3. ✅ Enhanced `icecap/analysis.py` with saturation curve plotting
4. ✅ Configuration templates for each experiment

### Analysis Deliverables

1. **Latency vs Throughput plots** (Questions 1a, 1b, 2a, 2b)
2. **Saturation point comparison table**
3. **Cost breakdown**: time in CAS vs manifest lists vs manifest files
4. **Sensitivity analysis**: impact of storage latency, parallelism, real conflict prob
5. **Summary report** with recommendations

### Documentation Deliverables

1. **SATURATION_ANALYSIS.md** - Methodology and results
2. **Updated QUICKSTART.md** - How to run saturation experiments
3. **Configuration guide** - Settings for different scenarios

---

## Timeline Estimate

- **Phase 0** (Clarification): 1 day
- **Phase 1** (Implementation): 2-3 days
- **Phase 2** (Baseline experiments): 1 day (runtime ~30 min per experiment × 20 configs)
- **Phase 3** (Real conflict experiments): 1 day (runtime ~30 min × 40 configs)
- **Phase 4** (Visualization & analysis): 1-2 days
- **Additional analyses**: 1-2 days
- **Documentation**: 1 day

**Total**: ~7-10 days for complete analysis

---

## Open Questions for Discussion

1. **Conflict definition**: Confirm interpretation of false vs real conflicts
2. **Saturation definition**: 50% abort rate, or also consider throughput plateau?
3. **Manifest operations**: Should false conflicts skip manifest list reads entirely (already read in read_manifest_lists())?
4. **Deletion vectors**: Should we model these separately, or lump into manifest file operations?
5. **Read-only transactions**: Should we add support for non-committing readers?
6. **Batching**: Out of scope for now, or worth exploring?

---

## Conclusion

### Can We Answer the Questions?

**Question 1a (Single table, false conflicts)**:
- ⚠️ Needs Phase 1 implementation to properly distinguish false conflicts
- Current simulator does too much work (manifest file operations)

**Question 1b (Single table, real conflicts)**:
- ⚠️ Needs Phase 1 implementation for variable conflict probability and cost

**Question 2a (Multi-table, false conflicts)**:
- ⚠️ Same as 1a - needs false conflict modeling

**Question 2b (Multi-table, real conflicts)**:
- ⚠️ Same as 1b - needs real conflict modeling

### Critical Path

1. Implement real vs false conflict distinction (Phase 1)
2. Run baseline experiments (Phase 2)
3. Run real conflict experiments (Phase 3)
4. Analyze and visualize results (Phase 4)

### Broader Value

Beyond answering stated questions, this analysis will:
- Establish fundamental throughput limits for Iceberg
- Identify optimal operating regions (load vs latency tradeoff)
- Guide catalog optimization priorities (what's worth optimizing?)
- Inform capacity planning for production deployments
- Suggest architectural improvements (grouping, batching, etc.)
