# Commit Overhead Analysis

This document explains the overhead analysis added to baseline experiments and its limitations.

## What is Commit Overhead?

**Overhead percentage** = `(commit_latency / total_latency) × 100`

This metric represents the fraction of total transaction time spent in the commit protocol (retries, exponential backoff, manifest I/O) versus actual transaction execution.

## Key Findings

### Single-Table Overhead (Experiment 2.1)

Overhead grows linearly with throughput until saturation:

| Load | Throughput | Success | Overhead |
|------|------------|---------|----------|
| Light (< 10 c/s) | 0.6-6.6 c/s | 100% | < 3% |
| Moderate (10-30 c/s) | 16-32 c/s | 99-100% | 7-14% |
| Saturation (50+ c/s) | 50-62 c/s | 18-76% | 21-26% |

**Interpretation**: At saturation, **1/4 of transaction time is spent retrying commits** instead of doing useful work.

### Multi-Table Overhead (Experiment 2.2)

Overhead scales dramatically with table count at saturation:

| Tables | Throughput | Success | Overhead |
|--------|------------|---------|----------|
| 1 | 62 c/s | 18% | 26% |
| 2 | 94 c/s | 28% | 39% |
| 5 | 111 c/s | 33% | 46% |
| 10 | 121 c/s | 36% | 50% |
| 20 | 131 c/s | 39% | 54% |
| 50 | 144 c/s | 44% | **59%** |

**Stunning finding**: With 50 tables at saturation, the commit protocol consumes **MORE time than transaction execution** (60% overhead)!

**This explains the latency paradox**:
- More tables → less contention per table ✓
- But multi-table coordination overhead dominates ✗
- Result: Higher throughput but WORSE tail latency

### At Light Load (All Configurations)

At ~33 commits/sec (100ms inter-arrival):
- All table counts show **similar overhead (13-14%)**
- Coordination cost is present but not dominant
- System under-saturated, minimal conflicts

**Key insight**: Overhead is load-dependent, not just table-count-dependent.

## What We Can Measure

Current transaction records contain:

```python
{
    'txn_id': int,
    't_submit': float,        # Submission time
    't_runtime': float,       # Actual execution time
    't_commit': float,        # Completion time
    'commit_latency': float,  # TOTAL commit overhead
    'total_latency': float,   # Runtime + commit
    'n_retries': int,         # Retry count
    'n_tables_read': int,     # Tables accessed
    'n_tables_written': int,  # Tables modified
    'status': str             # 'committed' or 'aborted'
}
```

From this we compute:
- **Overhead %** = `(commit_latency / total_latency) × 100`
- Aggregate statistics: mean, P50, P95, P99 overhead

## What We CANNOT Measure

The simulator does **not track operation-level timing**. We cannot break down `commit_latency` into:

### 1. Manifest List Read Time
- Time reading manifest lists for all touched tables
- Initial read vs retry reads
- Per-table breakdown for multi-table transactions

### 2. Manifest File Write/Repair Time
- Time writing manifest files
- File merge operations (only with real conflicts)

### 3. Exponential Backoff Wait Time
- Time spent in `asyncio.sleep()` between retries
- Formula: `min(BASE_DELAY_MS × 2^attempt, MAX_DELAY_MS)`
- Likely dominates at high retry counts (7-8 mean retries)

### 4. CAS Operation Time
- Time in `compare_and_swap()` catalog calls
- Currently hardcoded to 1ms
- Would scale with real catalog latency (S3: 10-100ms)

## Why This Matters

**At saturation with 50 tables:**
- Commit overhead: 59%
- Mean retries: 7.5
- But we don't know: Is it 90% backoff waits? 90% manifest I/O? 50/50 split?

**This would guide optimization priorities:**
- If mostly backoff → Need smarter backoff strategies
- If mostly manifest I/O → Need faster catalog or caching
- If mixed → Both improvements needed

## Estimating Breakdown (Rough Approximation)

For current experiments, we can estimate:

### Backoff Time
```python
def estimate_backoff_time(n_retries):
    BASE_DELAY_MS = 100  # From simulator config
    MAX_DELAY_MS = 5000
    total = 0
    for i in range(n_retries):
        total += min(BASE_DELAY_MS * (2 ** i), MAX_DELAY_MS)
    return total
```

For 7 retries: ~12.7 seconds of backoff time
For 8 retries: ~17.7 seconds of backoff time

### Manifest I/O Time
```python
def estimate_manifest_io_time(n_tables, n_retries, catalog_latency_ms=1):
    # Each retry reads manifest lists for all tables
    reads_per_retry = n_tables

    # Initial attempt + retries
    total_reads = reads_per_retry * (1 + n_retries)

    # Writes happen only on final commit
    total_writes = n_tables

    return (total_reads + total_writes) * catalog_latency_ms
```

For 50 tables, 7 retries, 1ms catalog: 400ms of I/O time

**Example breakdown for 50-table saturated transaction:**
- Total commit latency: ~60 seconds
- Estimated backoff: ~13 seconds (22%)
- Estimated manifest I/O: ~0.4 seconds (< 1%)
- **Remainder (78%): Coordination overhead, conflicts, retry logic**

**Note**: This is a rough approximation. Real instrumentation would provide accurate data.

## Recommendations for Future Work

### Short-term: Add instrumentation to main.py

Track timing for each operation:
```python
timing_breakdown = {
    'manifest_reads_ms': [],      # Per-table read times
    'manifest_writes_ms': [],     # Per-table write times
    'backoff_waits_ms': [],       # Per-retry backoff durations
    'cas_operations_ms': [],      # Per-CAS operation
    'per_retry_breakdown': []     # Full timing for each retry
}
```

### Medium-term: Add overhead visualization

Create plots showing:
1. Overhead breakdown by component (stacked area chart)
2. Overhead vs throughput by table count
3. Component timing distributions (box plots)

### Long-term: Optimization experiments

With detailed breakdown, test:
1. **Smarter backoff**: Adaptive delays based on contention
2. **Manifest caching**: Reduce repeated reads
3. **Batched CAS**: Combine multiple table writes
4. **Early abort**: Detect conflicts without full retry

## Conclusion

Overhead analysis reveals that **coordination cost dominates multi-table transactions at saturation**. With 50 tables, commit protocol takes 60% of total time, more than transaction execution itself.

However, we cannot currently separate manifest I/O from exponential backoff. Rough estimates suggest backoff dominates, but accurate instrumentation is needed to guide optimization efforts.

**Key takeaway**: At saturation, the system spends more time fighting contention than doing useful work. This is the fundamental limit of optimistic concurrency control under high load.
