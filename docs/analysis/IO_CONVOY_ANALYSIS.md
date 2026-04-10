# I/O Convoy in Partition-Level OCC

## Summary

Under high throughput with partition-level conflict detection, the Endive simulator exhibits an **I/O convoy** pattern where commit latency is dominated by historical manifest list reading, not CAS contention. Transactions self-synchronize into waves that retry together, producing a stable mean retry count of ~2.0 regardless of load.

## Phenomenon

In the `instant_partition_vs_tables` experiments (instant catalog, S3 storage, 2 partitions, 50 commits/sec):

| Metric | Observed |
|--------|----------|
| P50 commit latency | 37 seconds |
| Mean retries | 2.0 |
| Success rate | 100% |

The 37-second latency is surprising given:
- CAS takes 1ms (instant catalog)
- False conflict resolution takes ~60ms
- No backoff is configured

## Root Cause

Iceberg's conflict resolution requires reading manifest lists from **all intermediate snapshots** between the transaction's read and the current state (`validationHistory`). This creates an I/O cost proportional to how far behind the transaction is.

```
n_behind = (current_partition_seq) - (txn_snapshot_partition_seq)
         ≈ partition_commit_rate × transaction_runtime
         ≈ 25 commits/sec × 150 sec
         = 3,750 manifest lists
```

Reading 3,750 MLs in batches of 4 at ~30ms per batch:
```
ML read time = ceil(3750 / 4) × 30ms = 28 seconds
```

## The Convoy Pattern

**Phase 1: Execution** (0-150s)
- Transactions execute independently
- Catalog advances ~3,750 commits per partition

**Phase 2: Discovery** (150s)
- Transactions attempt CAS, all fail
- Each discovers it's ~3,750 snapshots behind
- Each begins reading historical MLs

**Phase 3: I/O Convoy** (150-178s)
- All transactions spend ~28s reading MLs
- No CAS contention—everyone is doing parallel I/O
- During this window, ~700 new commits occur

**Phase 4: Synchronized Retry** (178s)
- All transactions retry simultaneously
- Now only ~700 snapshots behind
- ML read takes ~5s, then CAS succeeds

**Phase 5: Success** (183s)
- Most transactions commit on second attempt
- A few need a third attempt
- Mean retries stabilizes at ~2.0

## Evidence

The stable mean_retries ≈ 2.0 across all partition counts is the signature:

| Partitions | Throughput | P50 Latency | Mean Retries |
|------------|------------|-------------|--------------|
| 2 | 50 c/s | 37.5s | 2.0 |
| 5 | 50 c/s | 23.0s | 2.0 |
| 10 | 50 c/s | 12.2s | 2.0 |
| 20 | 50 c/s | 6.3s | 2.0 |

If this were CAS contention, we'd expect retry counts to vary with contention level. Instead, the pattern is:
1. First attempt: Always fails (thousands behind)
2. Second attempt: Usually succeeds (hundreds behind)
3. Stable equilibrium regardless of load

## Why More Partitions Help

With N partitions and uniform distribution:
- Per-partition commit rate = total_rate / N
- n_behind = (total_rate / N) × runtime

Doubling partitions halves n_behind, halving ML read time:

| Partitions | Per-partition rate | n_behind (150s) | ML read time |
|------------|-------------------|-----------------|--------------|
| 2 | 25 c/s | 3,750 | 28s |
| 10 | 5 c/s | 750 | 5.6s |
| 20 | 2.5 c/s | 375 | 2.8s |

## Implications

**1. Backoff is ineffective**
Transactions aren't competing for CAS—they're independently reading MLs. Adding backoff would only delay the inevitable without reducing I/O.

**2. ML+ mode eliminates the convoy**
In append mode, false conflicts don't require reading historical MLs. The tentative entry remains valid, bypassing the I/O convoy entirely.

**3. This is not OCC contention**
Classic OCC contention causes cascading retries with increasing retry counts under load. This pattern shows stable retry counts because the bottleneck is I/O, not CAS conflicts.

**4. Latency is proportional to runtime × commit rate / partitions**
```
P50_commit_latency ≈ 2 × (runtime × rate / partitions / 4) × ML_latency
```

## Comparison with Table-Level Isolation

The `instant_ntbl_trivial` experiments use `num_groups=1`, meaning all tables share catalog-level conflicts. This shows **no scaling benefit** from multiple tables because every commit increments the global sequence number.

| num_tables | P50 @ 20 c/s |
|------------|--------------|
| 1 | 18.1s |
| 20 | 18.4s |

Partition mode provides true isolation because partitions have independent version counters.

## Recommendations

1. **Use partition-level isolation** for workloads with independent partition access patterns
2. **Consider ML+ mode** to eliminate historical ML reading on false conflicts
3. **Size partition count** based on expected concurrent transactions and acceptable latency
4. **Don't rely on backoff** to solve this pattern—it's I/O-bound, not contention-bound
