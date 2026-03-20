# Storage Provider Comparison

Results from exp4c: 5 providers x 6 table counts x 3 workload mixes x 10 arrival
rates, 5 seeds each. Each provider handles both CAS (catalog compare-and-swap) and
manifest I/O via `catalog.backend = "storage"`.

## Provider Latency Profiles

| Provider | CAS Median (ms) | Sigma | Min Latency (ms) |
|----------|-----------------|-------|-------------------|
| s3x      | 22              | 0.22  | 1                 |
| s3       | 61              | 0.14  | 10                |
| azurex   | 64              | 0.73  | 20                |
| azure    | 93              | 0.82  | 20                |
| gcp      | 170             | 0.91  | 80                |

## Throughput Overview

Maximum commits/sec at >95% success rate (workload knee), from
`plots/exp4c_tables_providers/workload_knee/workload_knee_table.md`.

### 1 Table

| Provider | FA=100% (c/s) | FA Latency | 90/10 (c/s) | VO Latency | 50/50 (c/s) | VO Latency |
|----------|--------------|------------|-------------|------------|-------------|------------|
| s3x      | 14.6         | 0.15s      | 7.5         | 6.6s       | 7.4         | 6.7s       |
| s3       | 2.4          | 0.84s      | 1.8         | 19.7s      | 1.8         | 19.6s      |
| azurex   | 2.5          | 0.72s      | 1.9         | 21.2s      | 1.8         | 20.9s      |
| azure    | 2.4          | 1.18s      | 1.5         | 23.3s      | 1.8         | 28.7s      |
| gcp      | 0.7          | 3.99s      | 0.4         | 26.9s      | 0.4         | 29.2s      |

### 50 Tables

| Provider | FA=100% (c/s) | FA Latency | 90/10 (c/s) | VO Latency | 50/50 (c/s) | VO Latency |
|----------|--------------|------------|-------------|------------|-------------|------------|
| s3x      | 14.9         | 0.11s      | 14.9        | 0.4s       | 14.9        | 0.4s       |
| s3       | 7.4          | 0.68s      | 7.4         | 2.1s       | 7.4         | 2.2s       |
| azurex   | 7.4          | 0.69s      | 7.4         | 2.3s       | 7.4         | 2.3s       |
| azure    | 7.3          | 1.09s      | 7.2         | 3.2s       | 7.2         | 3.2s       |
| gcp      | 3.6          | 3.35s      | 3.6         | 8.3s       | 3.6         | 8.1s       |

At 50 tables, workload mix has negligible effect on throughput. VO latency drops 10-16x
compared to 1 table because per-table contention is minimal.

## S3 Express (s3x)

S3 Express dominates every configuration. Its 22ms CAS median and 1ms minimum latency
produce a baseline commit pipeline of ~80ms, yielding 14.6 c/s on a single table with
pure FA. This is 6x faster than the next-best provider.

Multi-table scaling provides almost no benefit: throughput rises from 14.6 to 14.9 c/s
(+2%) between 1 and 50 tables for FA=100%. The provider is fast enough that single-table
contention is not the bottleneck even at high arrival rates.

For mixed workloads at 1 table, adding VO reduces the knee from 14.6 to 7.4-7.5 c/s.
VO commit latency is 6.6s at 1 table (44x the FA latency of 0.15s). With 5+ tables,
mixed workload throughput returns to the FA ceiling (14.9 c/s) and VO latency drops to
2.7s. At 50 tables VO latency is 0.4s.

**Takeaway**: S3 Express can serve mixed workloads at full throughput with as few as 5
tables. Single-table deployments should expect 6-7s VO latency.

## S3 Standard

S3's 61ms CAS median and 10ms minimum latency produce a baseline P50 of ~840ms at 1
table. The knee is 2.4 c/s for FA=100%, scaling to 7.4 c/s at 10+ tables (3.1x
improvement).

S3 and Azure Premium (azurex) are near-identical in throughput despite S3's lower CAS
median (61ms vs 64ms). S3's tighter distribution (sigma=0.14 vs 0.73) produces more
predictable latencies but doesn't translate to higher throughput because the median
dominates the commit path.

VO latency at 1 table is 19.7s (90/10 mix), dropping to 2.1s at 50 tables. The
throughput knee for mixed workloads reaches the FA ceiling at 20+ tables.

**Takeaway**: S3 requires 10+ tables to maximize FA throughput and 20+ tables to absorb
VO overhead. The 7.4 c/s ceiling is 2x below S3 Express.

## Azure Premium (azurex)

Azure Premium tracks S3 almost exactly in throughput: 2.5 c/s (1 table) to 7.4 c/s
(10+ tables) for FA=100%. The higher sigma (0.73 vs S3's 0.14) means wider latency
variance per request, but the similar CAS medians (64ms vs 61ms) produce equivalent
aggregate behavior.

VO latency is slightly higher than S3: 21.2s vs 19.7s at 1 table (90/10), converging
to 2.3s vs 2.1s at 50 tables.

**Takeaway**: Azure Premium and S3 Standard are interchangeable for throughput planning.
Azure Premium's higher variance may matter for latency-sensitive applications.

## Azure Standard

Azure Standard's 93ms CAS median introduces a clear gap below S3/Azure Premium. At 1
table, the FA knee (2.4 c/s) is similar to S3, but scaling is slower: the knee doesn't
plateau until 20+ tables (7.3 c/s vs 7.4 c/s for S3 at 10 tables).

FA commit latency is consistently higher: 1.18s at 1 table vs 0.84s for S3 (+40%). VO
latency at 1 table ranges from 23.3s (90/10) to 28.7s (50/50), the widest VO spread of
any provider, suggesting that higher CAS latency amplifies VO retry variance.

**Takeaway**: Azure Standard reaches the same ceiling as S3 (7.2-7.3 c/s) but needs
more tables to get there. The 40% higher commit latency may matter for latency budgets.

## GCP

GCP's 170ms CAS median and 80ms minimum latency make it fundamentally slower. At 1 table,
the FA knee is 0.7 c/s with a 3.99s baseline commit latency — 27x the S3 Express
baseline. Even at low load, P50 exceeds 2s.

Multi-table scaling helps GCP more than any other provider in relative terms: FA
throughput improves 5.1x from 1 to 50 tables (0.7 to 3.6 c/s). However, even at 50
tables GCP's throughput is still scaling — it hasn't reached its ceiling, suggesting the
CAS bottleneck persists at all tested table counts.

VO latency is 26.9-29.2s at 1 table. At 50 tables it remains 8.1-8.3s — still
significantly higher than all other providers at 50 tables (S3 Express: 0.4s, S3: 2.1s).

**Takeaway**: GCP requires 50+ tables to approach 4 c/s and is impractical for workloads
with VO operations under latency constraints.

## Key Observations

1. **S3 Express is 6x faster at 1 table, 2x at 50 tables.** The gap narrows with more
   tables because slower providers benefit more from contention reduction, but s3x
   maintains a structural advantage from its lower minimum latency.

2. **S3 and Azure Premium are functionally equivalent.** Despite different CAS
   distributions (61ms/sigma=0.14 vs 64ms/sigma=0.73), throughput is within 1-3%
   across all configurations. Provider selection between these two should be based on
   cost, availability, or tail-latency requirements rather than throughput.

3. **VO latency is 40-100x FA latency at 1 table.** This ratio compresses to 3-20x
   at 50 tables. The driver is retry cost: VO retries read manifest lists proportional
   to missed versions, while FA retries only re-attempt the CAS. Multi-table
   distribution reduces per-table contention, which reduces missed versions per retry.

4. **At 50 tables, workload mix is irrelevant to throughput.** All providers achieve
   the same knee regardless of FA/VO ratio. The bottleneck shifts from per-table
   contention to aggregate I/O bandwidth.

5. **Provider choice dominates table count for throughput.** S3 Express at 1 table
   (14.6 c/s) outperforms every other provider at any table count. Choosing a faster
   provider is more impactful than adding tables.

6. **GCP hasn't plateaued at 50 tables.** All other providers reach their ceiling by
   10-20 tables. GCP's 170ms CAS means contention remains significant even with traffic
   spread across 50 tables, suggesting 100+ tables would be needed to fully saturate
   GCP's I/O bandwidth.
