# Experiment 4: Multi-Table Catalog Contention

Summary of experiments 4a through 4c, covering multi-table scaling under uniform and
Zipf table distributions across five storage providers. All experiments use a single-file
catalog (all tables share one sequence number), 5 seeds per configuration, 1-hour
simulated duration.

## Experiment Matrix

| Experiment | Tables | CAS Latency | Workload | Distribution | Configs |
|------------|--------|-------------|----------|-------------|---------|
| 4a         | 1-50   | 1ms, 120ms  | FA=100%  | Uniform     | 240     |
| 4b         | 1-50   | 1ms, 120ms  | FA=90%   | Uniform     | 240     |
| 4a-zipf    | 1-50   | 1ms, 120ms  | FA=100%  | Zipf α=1.5  | 240     |
| 4b-zipf    | 1-50   | 1ms, 120ms  | FA=90%   | Zipf α=1.5  | 240     |
| 4c         | 1-50   | per-provider | All mixes| Uniform     | 900     |

## Exp4a: Uniform Tables, FastAppend

Distributing writes uniformly across N tables reduces per-table contention linearly.

**CAS = 1ms (instant catalog)**

| Tables | Max Throughput | Success @ Max | ~98% Success Throughput |
|--------|---------------|---------------|------------------------|
| 1      | 7.8 c/s       | 19%           | 2.8 c/s                |
| 5      | 29.8 c/s      | 73%           | 16.3 c/s               |
| 10     | 40.2 c/s      | 98%           | 40.2 c/s               |
| 50     | 41.0 c/s      | 100%          | 41.0 c/s               |

Near-linear scaling from 1 to 5 tables. At 10+ tables the I/O bandwidth ceiling (~41
c/s) is reached. The bottleneck shifts from catalog contention to aggregate S3 I/O.

**CAS = 120ms**: Ceiling drops to ~8 c/s at 50 tables — 5x lower than with 1ms CAS.
Catalog round-trip overhead on retries is the bottleneck.

## Exp4b: Uniform Tables, Mixed Workload (90/10)

Adding 10% VO to the workload mix reduces the ceiling modestly (37 vs 41 c/s at 50
tables, CAS=1ms) but introduces extreme tail latency: P99 exceeds 400s at moderate
loads due to VO retry cascading.

With 120ms CAS, the ceiling is ~8 c/s — identical to exp4a — confirming that catalog
latency dominates over operation type at high CAS.

## Exp4a-Zipf: Skewed Tables, FastAppend

Zipf alpha=1.5 concentrates ~50% of writes on the rank-1 table, reducing the effective
table count to 3-5 regardless of physical N.

| Metric (ias=20ms, CAS=1ms) | Zipf 50 tables | Uniform 50 tables |
|-----------------------------|----------------|-------------------|
| Success rate                | 73%            | 100%              |
| Throughput                  | 78 c/s         | 107 c/s           |
| Zipf effective equivalent   | ~5 uniform     | —                 |

70% of retries under Zipf are same-table conflicts (requiring manifest I/O), vs <2%
under Uniform. The hot table has 32% success while cold tables reach 95-100%.

See [ZIPF_REPORT.md](ZIPF_REPORT.md) for per-table breakdowns and conflict analysis.

## Exp4b-Zipf: Skewed Tables, Mixed Workload

VO on the Zipf hot table is particularly fragile: 15% success (vs 76% FA) at moderate
load because VO's longer execution time means more versions accumulate during each
attempt. The Zipf-Uniform gap widens slightly with VO (34.6pp vs 33.9pp at 10 tables,
ias=20ms).

## Exp4c: Provider Comparison

Five storage providers at 1-50 tables across three workload mixes. The provider ranking
is consistent across all configurations:

| Provider | CAS (ms) | 1-table FA knee | 50-table FA knee | Scaling factor |
|----------|----------|-----------------|------------------|---------------|
| s3x      | 22       | 14.6 c/s        | 14.9 c/s         | 1.02x         |
| s3       | 61       | 2.4 c/s         | 7.4 c/s          | 3.1x          |
| azurex   | 64       | 2.5 c/s         | 7.4 c/s          | 3.0x          |
| azure    | 93       | 2.4 c/s         | 7.3 c/s          | 3.0x          |
| gcp      | 170      | 0.7 c/s         | 3.6 c/s          | 5.1x          |

S3 Express at 1 table outperforms every other provider at any table count. GCP benefits
most from table distribution (5.1x) but still hasn't plateaued at 50 tables.

At 50 tables, workload mix has no effect on throughput — all providers achieve the same
knee regardless of FA/VO ratio. VO latency drops 10-16x from 1 to 50 tables.

See [PROVIDER_SUMMARY.md](PROVIDER_SUMMARY.md) for per-provider analysis.

## Takeaways

### 1. Provider choice > table count > workload mix

S3 Express at 1 table (14.6 c/s) outperforms S3 at 50 tables with 120ms CAS (8 c/s).
Choosing a faster provider yields more throughput improvement than any amount of table
sharding.

### 2. The single-file catalog has a hard CAS ceiling

At 120ms CAS, 50 tables can only sustain ~8 c/s regardless of workload. This is set by
catalog round-trip time, not I/O bandwidth. A catalog service with sub-10ms CAS would
remove this constraint.

### 3. 10 tables is the sweet spot for uniform distribution

Under uniform table selection, 10 tables captures most of the contention benefit (98%
of the ceiling for FA at CAS=1ms). Beyond 10 tables, returns are negligible unless CAS
latency is very high.

### 4. Zipf distribution negates table sharding

With alpha=1.5, 50 Zipf tables perform like ~5 uniform tables. The hot table absorbs
50% of writes and 70% of expensive same-table retries. Table sharding is not a solution
for skewed workloads.

### 5. VO is the tail-latency driver, not the throughput driver

At 50 tables, VO has zero effect on the throughput knee. Its impact is on latency: VO
commit latency is 3-100x FA latency depending on provider and table count. Systems that
tolerate high VO latency can run mixed workloads at the FA throughput ceiling.

### 6. Multi-table VO latency is manageable

VO latency drops from 6.6-29s (1 table) to 0.4-8s (50 tables) across providers. The
reduction is proportional to contention reduction — fewer missed versions per retry means
shorter manifest list reads. At 20+ tables, VO latency is within 10x of FA latency for
all providers except GCP.

## Actionable Recommendations

- **For high-throughput FA workloads**: Use S3 Express. A single table sustains 14.6 c/s
  with no sharding overhead.

- **For mixed FA/VO workloads**: Use 10-20 tables to reduce VO latency to acceptable
  levels. At 20 tables, S3 Express VO latency is 0.8s; S3/Azure Premium is 4.5s.

- **For Zipf-skewed workloads**: Table sharding is insufficient. Consider partition-level
  diversity within hot tables, or isolate VO operations on hot tables to dedicated time
  windows.

- **For cost-sensitive deployments**: S3 Standard and Azure Premium are interchangeable
  at 7.4 c/s ceiling (10+ tables). Choose based on cost/availability, not throughput.

- **For GCP**: Budget for 50+ tables and accept 3.6 c/s ceiling, or consider a separate
  low-latency catalog service to decouple CAS from storage I/O.
