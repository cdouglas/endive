# Baseline Experiment Results

Summary of experiments characterizing single-table and multi-table saturation with an infinitely fast catalog.

## Experiment Configuration

**Common parameters:**
- Catalog latency: 1ms (infinitely fast baseline)
- Transaction runtime: Lognormal(μ=180s, σ=1.5, min=30s)
- Conflict type: False conflicts only (no manifest file merging)
- Retry limit: 10 attempts
- Simulation duration: 1 hour
- Analysis window: 30 minutes (15-45 min, excluding warmup/cooldown)
- Seeds per configuration: 5

## Experiment 2.1: Single-Table Saturation

**Goal**: Establish baseline throughput limits for a single table.

**Configuration**: 1 table, 9 load levels (10ms to 5000ms inter-arrival)

### Key Findings

#### Peak Throughput: ~62 commits/sec

At most aggressive load (10ms inter-arrival):
- Throughput: 62.0 commits/sec
- Success rate: 18.1% (system thrashing)
- P50 latency: 26s
- P95 latency: 144s
- P99 latency: 269s
- Mean retries: 8.7 (near 10-retry limit)

#### Efficient Operating Point: ~50 commits/sec @ 76% success

At 50ms inter-arrival:
- Throughput: 50.3 commits/sec
- Success rate: 76.2%
- P50 latency: 21s
- P95 latency: 131s
- P99 latency: 255s
- Mean retries: 7.9

#### Saturation Threshold: 50-60 commits/sec

50% success rate crossed between:
- 58.5 commits/sec (35% success at 20ms inter-arrival)
- 50.3 commits/sec (76% success at 50ms inter-arrival)

### Latency Scaling

**Hockey stick pattern** approaching saturation:

| Load (ms) | Throughput | P50 | P95 | P99 | Retries |
|-----------|------------|-----|-----|-----|---------|
| 5000 (light) | 0.6 c/s | 0.3s | 1.8s | 3.2s | 2.1 |
| 1000 | 3.3 c/s | 1.3s | 8.3s | 15.9s | 2.8 |
| 500 | 6.6 c/s | 2.5s | 16.5s | 32.9s | 3.2 |
| 200 | 16.5 c/s | 6.5s | 44.0s | 86.5s | 4.3 |
| 100 | 32.3 c/s | 13.5s | 89.6s | 166s | 5.9 |
| 50 | 50.3 c/s | 21.4s | 131s | 255s | 7.9 |
| 20 | 58.5 c/s | 24.2s | 142s | 274s | 8.5 |
| 10 | 62.0 c/s | 25.7s | 144s | 269s | 8.7 |

**Key observation**: P99 latency grows 84× from light load to saturation (3.2s → 269s)

### Commit Overhead Analysis

**Overhead percentage** = (commit_latency / total_latency) × 100

This represents the fraction of total transaction time spent in the commit protocol (retries, exponential backoff, manifest I/O) vs actual transaction execution.

| Load (ms) | Throughput | Success | Mean Overhead | P95 Overhead | P99 Overhead |
|-----------|------------|---------|---------------|--------------|--------------|
| 5000 (light) | 0.6 c/s | 100% | 0.4% | 0.6% | 0.7% |
| 1000 | 3.3 c/s | 100% | 1.5% | 1.8% | 2.0% |
| 500 | 6.6 c/s | 100% | 2.9% | 3.3% | 3.5% |
| 200 | 16.5 c/s | 100% | 7.0% | 7.6% | 7.9% |
| 100 | 32.3 c/s | 99% | 13.6% | 14.3% | 14.8% |
| 50 | 50.3 c/s | 76% | 21.1% | 21.6% | 21.8% |
| 20 | 58.5 c/s | 35% | 24.5% | 24.9% | 25.1% |
| 10 | 62.0 c/s | 18% | 25.9% | 26.3% | 26.4% |

**Key insights:**
- Light load (< 10 c/s): Commit overhead < 3% - transactions execute cleanly with minimal conflicts
- Moderate load (10-30 c/s): Overhead grows to 7-14% - conflicts increase but system still efficient
- Saturation (50+ c/s): Overhead exceeds 20% - **1/4 of transaction time spent retrying commits**

**At saturation, the system spends more time fighting contention than doing useful work.**

### Interpretation

**Bottleneck is contention, not catalog latency:**
- With 1ms catalog CAS, single table still saturates at ~60 c/s
- Every commit invalidates all in-flight transactions
- False conflicts (manifest list reads) still expensive due to retry overhead
- At saturation, transactions exhaust retry budget (8.7 mean retries)

**Practical implications:**
- Target <30 commits/sec per table for good latency (P95 < 90s)
- 30-50 commits/sec: acceptable with degrading latency
- 50-60 commits/sec: saturation zone (high retries, exponential latency)
- Beyond 60 commits/sec: thrashing (>80% abort rate)

## Experiment 2.2: Multi-Table Scaling

**Goal**: Determine if table-level conflict isolation improves throughput.

**Configuration**: 6 table counts (1, 2, 5, 10, 20, 50 tables), 9 load levels each

### Key Findings

#### Throughput Scaling is Sub-Linear

At 10ms inter-arrival (most aggressive):

| Tables | Throughput | Success | Improvement | Scaling Efficiency |
|--------|------------|---------|-------------|-------------------|
| 1 | 62.0 c/s | 18% | baseline | - |
| 2 | 94.3 c/s | 28% | +52% | 76% |
| 5 | 111.0 c/s | 33% | +79% | 32% |
| 10 | 121.2 c/s | 36% | +95% | 19% |
| 20 | 131.0 c/s | 39% | +111% | 11% |
| 50 | 144.0 c/s | 44% | +132% | 5% |

**Key insight**: 50× more tables → only 2.3× throughput improvement

#### The Latency Paradox: More Tables = Higher Latency

**Surprising finding**: At the same throughput level, more tables have WORSE tail latency.

At ~100 commits/sec:
- 2 tables: P99 = 200k ms (3.3 minutes)
- 5 tables: P99 = 180k ms
- 10 tables: P99 = 170k ms
- 20 tables: P99 = 200k ms
- 50 tables: P99 = 260k ms (4.3 minutes)

**Why?** Multi-table transactions have coordination overhead:
- Must read manifest lists for ALL touched tables (sequential with max_parallel limit)
- Must write manifest lists for ALL touched tables
- More I/O operations = more opportunities for delays
- "Parallelism benefit" (reduced conflicts) offset by "coordination cost"

#### Commit Overhead Confirms Coordination Cost

**Mean overhead percentage** at saturation (10ms inter-arrival):

| Tables | Throughput | Success | Mean Overhead | P95 Overhead |
|--------|------------|---------|---------------|--------------|
| 1 | 62.0 c/s | 18% | 25.9% | 26.3% |
| 2 | 94.3 c/s | 28% | 39.3% | 39.6% |
| 5 | 111.0 c/s | 33% | 46.1% | 46.4% |
| 10 | 121.2 c/s | 36% | 50.3% | 50.5% |
| 20 | 131.0 c/s | 39% | 54.2% | 54.5% |
| 50 | 144.0 c/s | 44% | **59.4%** | **59.7%** |

**Stunning finding**: With 50 tables at saturation, commit protocol consumes **MORE time than transaction execution** (60% overhead)!

**Why overhead scales with table count:**
- Each additional table adds manifest I/O operations
- Multi-table transactions must coordinate across all touched tables
- More coordination = more retry attempts when ANY table conflicts
- Exponential backoff time grows with retry count

**This explains the latency paradox**:
- More tables reduces per-table contention ✓
- But multi-table coordination overhead dominates ✗
- Net result: Higher throughput but WORSE tail latency

**At light load** (100ms inter-arrival, ~33 c/s for all configs):
- All table counts show similar overhead (13-14%)
- Coordination cost is present but not dominant
- System under-saturated, minimal conflicts

#### Saturation Points Shift Rightward

50% success threshold:

| Tables | Saturation Throughput | Improvement |
|--------|-----------------------|-------------|
| 1 | ~55 c/s | baseline |
| 2 | ~85 c/s | +55% |
| 5 | ~105 c/s | +91% |
| 10 | ~115 c/s | +109% |
| 20 | ~125 c/s | +127% |
| 50 | ~140 c/s | +155% |

#### Diminishing Returns Beyond 10 Tables

| Jump | Throughput Gain | Tables Added | Gain per Table |
|------|-----------------|--------------|----------------|
| 1→2 | +32 c/s | +1 table | 32 c/s/table |
| 2→5 | +17 c/s | +3 tables | 5.7 c/s/table |
| 5→10 | +10 c/s | +5 tables | 2.0 c/s/table |
| 10→20 | +10 c/s | +10 tables | 1.0 c/s/table |
| 20→50 | +13 c/s | +30 tables | 0.4 c/s/table |

**Sweet spot: 10-20 tables** for best throughput/complexity tradeoff.

### Load-Dependent Behavior

**At moderate load (100-200ms)**: Table count doesn't matter
- All configurations: ~33 c/s, 100% success
- Similar latencies across table counts
- System under-saturated

**At high load (10-20ms)**: Multi-table helps throughput but hurts latency
- Throughput: 62 → 144 c/s (+132%)
- Success: 18% → 44% (+144%)
- P99 latency: 270s → 418s (+55% worse!)

## Cross-Experiment Comparison

### Throughput Limits

| Configuration | Peak Throughput | @ Success Rate | P99 Latency |
|---------------|----------------|----------------|-------------|
| 1 table | 62 c/s | 18% | 269s |
| 2 tables | 94 c/s | 28% | 340s |
| 10 tables | 121 c/s | 36% | 374s |
| 50 tables | 144 c/s | 44% | 418s |

**Observation**: Each doubling of tables adds ~20-30 c/s, but latency increases proportionally.

### Retry Behavior

Mean retries at peak throughput:

| Tables | Mean Retries | Note |
|--------|--------------|------|
| 1 | 8.7 | Near 10-retry limit |
| 2 | ~8.5 | Slight improvement |
| 10 | ~8.0 | Moderate improvement |
| 50 | ~7.5 | Best, but still high |

**Interpretation**: Table isolation reduces conflicts but doesn't eliminate retry explosion at saturation.

### Fundamental Limit at ~65 commits/sec

At moderate load (50-100ms inter-arrival):
- All configurations plateau at ~65 commits/sec
- 100% success rate
- Similar latencies

**Suggests**: Bottleneck shifts from contention to transaction duration.
- Transaction runtime: mean 3 minutes
- Pipeline depth limited by total simulation time
- Maximum sustainable throughput ≈ 1 / mean_runtime

## Key Insights

### 1. Catalog Latency is Not the Bottleneck

With 1ms (infinitely fast) catalog:
- Single table: 62 c/s
- 50 tables: 144 c/s

**Real-world catalogs** (10-100ms CAS latency) would perform WORSE.

### 2. Contention Dominates at High Load

Even with table-level conflict isolation:
- Success rates collapse above ~100-130 c/s
- Retry explosion (8+ retries mean)
- Exponential tail latency growth

### 3. Multi-Table Parallelism Has Limited Benefit

**Expected**: 50 tables → 50× throughput (linear scaling)
**Actual**: 50 tables → 2.3× throughput (sub-linear)

**Reasons**:
- Multi-table transactions have coordination overhead
- Transactions still conflict within table boundaries
- Retry amplification effects still present

**Overhead data confirms this**:
- Single table @ saturation: 26% overhead
- 50 tables @ saturation: **59% overhead** (commit protocol takes MORE time than transaction work!)
- Coordination cost scales with table count, negating parallelism benefits

### 4. Tail Latency is the Real Problem

At saturation:
- P50 latency: 26-71s (manageable)
- P99 latency: 269-418s (4.5-7 minutes!)

**For user-facing workloads**: P99 > 1 minute is unacceptable.

**Operating point**: Stay well below saturation (30-50 c/s per table) to maintain reasonable tail latency.

### 5. Sweet Spot: 10-20 Tables

Best tradeoff between throughput and complexity:
- 10 tables: 2× throughput vs single table
- Beyond 20 tables: diminishing returns (< 1 c/s per table)
- Coordination overhead increases with table count

## Recommendations for Production

### Throughput Targets

For P95 < 90s latency:

| Isolation | Target Throughput | Safety Margin |
|-----------|------------------|----------------|
| Single table | < 30 c/s | 50% of peak |
| 10 tables | < 60 c/s | 50% of peak |
| 50 tables | < 70 c/s | 50% of peak |

### Design Implications

1. **Partition large tables** into 10-20 logical tables for optimal throughput
2. **Avoid multi-table transactions** when possible (coordination overhead)
3. **Monitor tail latency** (P95/P99) as key health metric
4. **Scale horizontally** beyond ~100 c/s (multiple catalog instances)

### When to Worry

Warning signs of approaching saturation:
- Success rate drops below 90%
- Mean retries exceed 5
- P95 latency > 90s
- P99 latency > 3 minutes

## Limitations and Future Work

### Current Limitations

1. **False conflicts only**: Real conflicts (manifest file merging) would be more expensive
2. **Infinitely fast catalog**: Real catalogs (S3, JDBC) have 10-100ms latency
3. **Uniform workload**: Real workloads have skewed access patterns
4. **No query latency**: Only measuring commit latency, not total query time
5. **Limited overhead breakdown**: Cannot separate manifest I/O from exponential backoff (see below)

### Overhead Breakdown Limitations

**What we measured:**
- `commit_latency`: Total time in commit protocol
- `total_latency`: Transaction runtime + commit latency
- `n_retries`: Number of retry attempts
- **Overhead %**: (commit_latency / total_latency) × 100

**What we CANNOT measure with current instrumentation:**

The simulator does not track operation-level timing. To break down commit overhead into components, we would need to instrument `main.py` to record:

1. **Manifest list read time**: Time spent reading manifest lists for all tables
   - Separate: Initial read vs retry reads
   - Separate: Per-table breakdown for multi-table transactions

2. **Manifest file write/repair time**: Time spent writing manifest files
   - Only relevant when `real_conflict_probability > 0`
   - Includes file merge operations

3. **Exponential backoff wait time**: Time spent in `asyncio.sleep()` between retries
   - Grows exponentially: `min(BASE_DELAY_MS * 2^attempt, MAX_DELAY_MS)`
   - Dominates overhead at high retry counts

4. **CAS operation time**: Time in `compare_and_swap()` calls
   - Currently hardcoded to 1ms (catalog latency)
   - Would scale with real catalog (S3, JDBC)

**Current data structure** (results.parquet):
```python
{
    'txn_id': int,
    't_submit': float,      # Transaction submission time
    't_runtime': float,     # Actual execution time (query + compute)
    't_commit': float,      # Completion timestamp
    'commit_latency': float,  # t_commit - t_runtime - t_submit (TOTAL commit overhead)
    'total_latency': float,   # t_commit - t_submit
    'n_retries': int,         # Number of retry attempts
    'status': str             # 'committed' or 'aborted'
}
```

**To add detailed breakdown**, we would need:
```python
{
    # ... existing fields ...
    'time_manifest_reads': float,      # Total time reading manifest lists
    'time_manifest_writes': float,     # Total time writing manifest files
    'time_backoff_waits': float,       # Total time in exponential backoff
    'time_cas_operations': float,      # Total time in CAS calls
    'per_retry_timings': List[Dict]    # Breakdown per retry attempt
}
```

**Why this matters:**
- At saturation, 50-table transactions spend 60% of time in commit protocol
- We know retry count (mean 7-8 retries) but not how time is spent
- Is it mostly backoff waits? Manifest I/O? Both?
- This would guide optimization priorities (faster catalog? smarter backoff? reduced I/O?)

**Workaround for current experiments:**
- Estimate backoff time from retry count: `sum(min(BASE_DELAY * 2^i, MAX_DELAY) for i in range(n_retries))`
- Estimate manifest I/O from table count: `n_tables_read + n_tables_written` operations
- Remainder is CAS and coordination overhead

**Recommendation**: Add instrumentation in future experiments to track operation-level timing.

### Future Experiments

**Experiment 3**: Real conflict costs
- Vary `real_conflict_probability` from 0.0 to 1.0
- Measure manifest file merge overhead
- Quantify impact on throughput and latency

**Experiment 4**: Catalog latency sensitivity
- Vary CAS latency from 1ms to 100ms
- Determine catalog performance requirements

**Experiment 5**: Skewed access patterns
- Zipfian table access distribution
- Hot partition contention

**Experiment 6**: Group-level isolation
- Between table-level and catalog-level
- Multi-tenant modeling

## Conclusion

With an **infinitely fast catalog**, Iceberg-style optimistic concurrency:
- **Single table**: Saturates at ~60 commits/sec
- **50 tables**: Reaches ~144 commits/sec (2.3× improvement)
- **Tail latency**: Explodes at saturation (P99 > 4 minutes)

**Key takeaway**: Contention, not catalog latency, is the primary bottleneck. Multi-table parallelism helps but has diminishing returns due to coordination overhead.

**For production**: Stay at 50% of saturation throughput to maintain acceptable tail latency (P95 < 90s).
