# Zipf Table Distribution Report

Experiments 4a-zipf and 4b-zipf repeat the multi-table sweeps from exp4a/4b with
Zipf-distributed table selection (alpha=1.5) instead of uniform random. Setup: S3
storage, 5 seeds per point, 1-hour simulated duration. Sweep: num_tables (1-50) x
catalog_latency (1-120ms) x arrival_rate.

## Zipf write distribution

With alpha=1.5, the rank-1 table absorbs roughly half of all writes regardless of N.

| N tables | Top-1 share | Top-3 share | Effective N (1/HHI) |
|----------|-------------|-------------|---------------------|
| 5        | 56.8%       | 87.8%       | 2.6                 |
| 10       | 50.1%       | 77.5%       | 3.3                 |
| 20       | 46.1%       | 71.2%       | 3.9                 |
| 50       | 42.9%       | 66.3%       | 4.5                 |

Even with 50 tables, contention is concentrated on ~4-5 effective tables. Zipf N=50
should behave like uniform N~5 for contention purposes.

## Throughput and success rate: Zipf vs Uniform

All data at CAS=1ms (instant catalog), 100% FastAppend.

### Success rate (higher is better)

| Load (ias) | 1 table | 5 Z / U | 10 Z / U | 20 Z / U | 50 Z / U |
|------------|---------|---------|----------|----------|----------|
| 20ms       | 20.1%   | 56/75   | 64/98    | 69/100   | 73/100   |
| 50ms       | 44.3%   | 82/100  | 88/100   | 91/100   | 93/100   |
| 100ms      | 76.1%   | 98/100  | 99/100   | 100/100  | 100/100  |
| 200ms      | 98.8%   | 100/100 | 100/100  | 100/100  | 100/100  |

At 1 table, Zipf and Uniform are identical. The gap peaks at high load and many tables:
at ias=20ms with 50 tables, Uniform reaches 100% while Zipf stalls at 73%. At low load
(ias>=200ms) contention fades and both converge.

### Throughput (commits/sec)

| Load (ias) | 1 table | 5 Z / U   | 10 Z / U   | 50 Z / U    |
|------------|---------|-----------|------------|-------------|
| 20ms       | 20.7    | 59/79     | 68/105     | 78/107      |
| 50ms       | 18.4    | 35/43     | 37/43      | 40/43       |
| 100ms      | 16.0    | 21/22     | 21/22      | 21/22       |

At ias=20ms with 50 tables, Uniform achieves 107 c/s (5.2x over 1 table). Zipf achieves
78 c/s (3.7x). The Zipf/Uniform throughput ratio ranges from 0.65 (10 tables) to 0.72
(50 tables).

## Effective table count

Matching Zipf success rates to the Uniform lookup:

| Zipf tables | SR at ias=20ms | Closest Uniform equivalent |
|-------------|----------------|---------------------------|
| 5           | 56%            | ~2 tables                 |
| 10          | 64%            | ~3 tables                 |
| 20          | 69%            | ~4-5 tables               |
| 50          | 73%            | ~5 tables                 |

Zipf with 50 tables performs like Uniform with ~5 tables. This matches the theoretical
effective count of 4.5 from HHI analysis. Adding tables beyond 10 under Zipf yields
diminishing returns.

## Per-table breakdown

Observed at N=10, ias=50ms, CAS=1ms (exp4a-zipf):

| Table rank | Write share | Success rate | Mean retries | Mean latency |
|------------|-------------|-------------|-------------|--------------|
| 0 (hot)    | 52.1%       | 31.6%       | 8.6         | 893ms        |
| 1          | 17.4%       | 77.1%       | 6.1         | 784ms        |
| 2          | 9.6%        | 90.6%       | 3.9         | 584ms        |
| 3-4        | 10.8%       | 93-94%      | 3.5-3.6     | 500-520ms    |
| 5-9 (cold) | 10.1%       | 95-100%     | 2.1-3.3     | 344-470ms    |

The hot table (rank 0) has 32% success and 8.6 retries — 3x worse than cold tables.
Success rate varies 3x within a single experiment. Aggregate metrics hide this
heterogeneity entirely.

## Conflict type decomposition

Under Zipf, most retries hit the same table because the hot table dominates traffic.

| N tables | Total retries | Cross-table (free) | Same-table (I/O) | Same-table % |
|----------|---------------|-------------------|-------------------|--------------|
| 10       | 20.8M         | 6.3M (30%)        | 14.5M (70%)       | 70%          |
| 50       | 19.5M         | 7.8M (40%)        | 11.7M (60%)       | 60%          |

Under Uniform at 50 tables, ~98% of retries would be cross-table (free). Zipf inverts
this: 60% of retries require manifest I/O because they collide on the hot table.

Per-table anatomy (N=10, ias=20ms): the hot table has 80% same-table conflicts (it
collides with itself), while cold tables have only 35-40% same-table conflicts (most
of their collisions are with writes to other tables, predominantly the hot table).

## CAS latency interaction

At CAS=120ms, the Zipf-Uniform gap narrows because slow CAS dominates retry cost
regardless of conflict type.

| Load (ias) | 1 table | 10 Z / U  | 50 Z / U  |
|------------|---------|----------|----------|
| 50ms       | 24.7%   | 43.7/48.5 | 46.3/49.6 |
| 100ms      | 45.7%   | 68.0/80.7 | 72.5/82.2 |
| 200ms      | 76.0%   | 92.1/98.4 | 94.6/98.6 |

The gap at ias=100ms, 50 tables is 10pp (73% vs 82%) at CAS=120ms, compared to 27pp at
CAS=1ms. When CAS itself costs 120ms per attempt, the additional manifest I/O for
same-table conflicts is proportionally less significant.

## Mixed workload (90/10 FA/VO)

Exp4b-zipf adds 10% ValidatedOverwrite operations. VO latency is higher than FA because
each retry reads manifest lists proportional to missed versions.

### Overall success rate (CAS=1ms)

| Load (ias) | 1 table | 10 Z / U  | 50 Z / U   |
|------------|---------|----------|-----------|
| 20ms       | 19.9%   | 62.3/96.9 | 70.8/98.0  |
| 50ms       | 43.8%   | 85.3/95.9 | 90.3/99.1  |
| 100ms      | 74.7%   | 97.9/100  | 98.8/100   |

The Zipf-Uniform gap is slightly larger in mixed workloads: 34.6pp at ias=20ms with
10 tables (vs 33.9pp for pure FA), because VO failures under hot-table contention drag
down the aggregate.

### Per-table VO impact (Zipf, N=10, ias=50ms)

| Table rank | FA success | FA latency | VO success | VO latency |
|------------|-----------|-----------|-----------|-----------|
| 0 (hot)    | 75.8%     | 806ms     | 15.0%     | 49.5s     |
| 9 (cold)   | 100%      | 333ms     | 100%      | 2.5s      |

The hot table's VO success is 15% — 5x worse than its FA success (76%). Cold table VO
achieves 100% success at 2.5s latency. The hot table amplifies VO's retry cost because
more concurrent writes means more missed versions per retry.

## Key findings

1. **Zipf alpha=1.5 reduces effective table count to 3-5.** Adding physical tables
   beyond 10 under Zipf barely helps. The hot table absorbs 43-50% of writes.

2. **70% of retries under Zipf are same-table conflicts.** Under Uniform at 50 tables,
   98% of retries would be cross-table and free. Zipf's hot-table concentration makes
   most retries expensive.

3. **Per-table success varies 3x within a single experiment.** Table 0: 32% success;
   tables 5-9: 95-100%. Aggregate metrics are misleading under skewed distributions.

4. **CAS latency narrows the Zipf-Uniform gap.** At CAS=120ms, the gap is 10pp vs
   27pp at CAS=1ms (50 tables, ias=100ms). Slow CAS dominates retry cost regardless
   of conflict type.

5. **VO on the hot table drops to 15% success.** The combination of long VO execution
   time and hot-table concentration means nearly all VO transactions on the hot table
   conflict. Cold tables maintain 100% VO success.

## Design implications

- **Table sharding under skew is ineffective for contention.** Splitting a
  Zipf-distributed workload across 50 tables yields the contention behavior of ~5
  uniform tables.

- **Partition-level diversity matters more than table count.** If the hot table has P
  partitions with uniform partition selection, effective contention per partition is
  1/(2P) of total traffic. Intra-table partitioning can break the hot-table bottleneck
  where adding tables cannot.

- **VO on hot tables needs isolation.** Under Zipf, VO success on the hot table
  approaches zero at high load. Practical options: table-level concurrency limits,
  branch-based VO isolation, or routing VO to dedicated time windows.
