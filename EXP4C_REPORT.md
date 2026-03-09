# Exp4c: Provider Comparison Report

Results from the Endive OCC simulator, run with 5 seeds per configuration. All
experiments use 1-hour simulated duration with 15-minute warmup/cooldown periods.
Unlike exp1-4b, which fix storage at S3 and sweep catalog latency independently,
exp4c uses `backend = "storage"` so each provider determines both CAS and manifest
I/O latency end-to-end.

**Terminology**: Throughout this report and EXP_REPORT.md, "knee" refers to the
highest measured throughput at which the system still sustains >95% success rate.
Below the knee, adding load increases throughput roughly linearly. Above it,
success rate degrades rapidly and effective throughput plateaus or declines. The
knee is the practical operating limit for a workload.

## Provider Profiles

Each provider has a lognormal latency distribution for CAS, READ, and WRITE
operations, parameterized by a median and sigma. The `min_latency_ms` floor
applies to all operations.

| Provider | CAS Median | CAS Sigma | WRITE Base | READ Base | Min Latency |
|----------|-----------|-----------|------------|-----------|-------------|
| s3x      | 22 ms     | 0.22      | 6.5 ms     | 2.5 ms    | 1 ms        |
| s3       | 61 ms     | 0.14      | 60 ms      | 27 ms     | 10 ms       |
| azurex   | 64 ms     | 0.73      | 41 ms      | 35 ms     | 20 ms       |
| azure    | 93 ms     | 0.82      | 45 ms      | 38 ms     | 20 ms       |
| gcp      | 170 ms    | 0.91      | 200 ms     | 200 ms    | 80 ms       |

CAS sigma controls tail variance. S3 has the tightest CAS distribution (sigma=0.14),
meaning predictable latency but no chance of a fast outlier. GCP has the widest
(sigma=0.91), producing high variance on top of an already-high median.

The sweep is 4-dimensional: provider x num_tables (1-50) x FA ratio (100%, 90%, 50%)
x inter-arrival scale.

## FA=100%, 1 Table

This is the simplest case: pure FastAppend on a single table, where the provider
determines the full commit pipeline latency (CAS + manifest write + manifest read
on retry).

| Provider | P50 @ Low Load | Knee (c/s) | Success @ Knee | Max Throughput | Success @ Max |
|----------|----------------|------------|----------------|----------------|---------------|
| s3x      | 0.08s          | 16.1       | 98.0%          | 23.8           | 58.0%         |
| s3       | 0.47s          | 2.7        | 98.6%          | 5.3            | 13.0%         |
| azurex   | 0.42s          | 2.7        | 99.3%          | 6.4            | 15.7%         |
| azure    | 0.58s          | 2.7        | 97.4%          | 6.2            | 15.2%         |
| gcp      | 1.94s          | 0.8        | 96.3%          | 1.7            | 4.1%          |

S3 Express sustains 6x the throughput of S3 at the knee (16.1 vs 2.7 c/s). The low-load
P50 of 80ms reflects the tight end-to-end pipeline: 22ms CAS + manifest I/O from
a 1ms floor. S3's 470ms P50 is dominated by the 61ms CAS plus 60ms WRITE base, with
the 10ms floor preventing fast manifest completions.

Azurex and S3 have nearly identical knees (2.7 c/s) despite azurex having a slightly
higher CAS median (64 vs 61ms). Azurex compensates with a lower WRITE base (41 vs
60ms). The two providers diverge at saturation: azurex reaches 6.4 c/s max vs
S3's 5.3 c/s, a 20% advantage attributable to faster manifest writes reducing
per-retry cost.

Azure (standard) matches S3 and azurex at the knee (2.7 c/s) but achieves this with
a higher P50 (0.93s vs 0.70s for S3), meaning it's closer to its capacity limit at
the same throughput.

GCP saturates before reaching 1 c/s at the knee. At low load, the 1.94s P50 is
already 24x worse than s3x. Even a single CAS round-trip (170ms median) plus a
single manifest write (200ms base) pushes the minimum commit pipeline close to 0.5s,
and any retry pushes well past 1s.

### Comparison to exp1 baseline

The exp1 baseline (1ms instant catalog + S3 storage) achieves 4.0 c/s at the knee
and 7.8 c/s max. S3 as a unified provider (exp4c) achieves 2.7 c/s at the knee
and 5.3 c/s max. The 32% throughput reduction is the cost of replacing a 1ms catalog
with S3's 61ms CAS. S3 Express as a unified provider (16.1 c/s knee) exceeds the
exp1 baseline by 4x, because s3x's manifest I/O is 10x faster than S3's, more than
compensating for its 22ms CAS.

## FA=90%, 1 Table

A 90/10 FastAppend/ValidatedOverwrite mix is the most realistic operating point in
the sweep: production Iceberg workloads are predominantly appends with occasional
schema-validating overwrites. The question is whether 10% VO is enough to
meaningfully change the story told by pure FA.

| Provider | Knee (c/s) | P50 @ Knee | P95 @ Knee | P99 @ Knee |
|----------|------------|------------|------------|------------|
| s3x      | 16.0       | 0.13s      | 6.9s       | 36.5s      |
| s3       | 2.7        | 0.73s      | 12.4s      | 66.0s      |
| azurex   | 2.7        | 0.62s      | 15.6s      | 75.8s      |
| azure    | 2.6        | 1.01s      | 15.8s      | 92.9s      |
| gcp      | 0.4        | 2.30s      | 16.2s      | 64.3s      |

The knee throughputs are nearly identical to FA=100% — adding 10% VO does not
measurably reduce the throughput ceiling. S3 Express still sustains 16.0 c/s, S3
and azurex still reach 2.7 c/s. The system handles the same offered load before
success starts falling.

The tail latency tells a different story. At the knee, P99 jumps from sub-second
(FA=100%) to tens of seconds (FA=90%) across all providers:

| Provider | P99 @ Knee (FA=100%) | P99 @ Knee (FA=90%) | Ratio |
|----------|---------------------|---------------------|-------|
| s3x      | 0.45s               | 36.5s               | 81x   |
| s3       | 2.62s               | 66.0s               | 25x   |
| azurex   | 2.33s               | 75.8s               | 33x   |
| azure    | 3.76s               | 92.9s               | 25x   |
| gcp      | 12.1s               | 64.3s               | 5x    |

S3 Express shows the most dramatic P99 explosion (81x) because its FA=100% P99 is
so low (0.45s). The absolute P99 at FA=90% (36.5s) is still the best of any
provider, but it's no longer in a different league from the rest. A single VO
retry cascade erases most of the tail latency advantage that fast I/O provides.

The P95 tells the same story at lower magnitude: s3x goes from 0.35s to 6.9s (20x),
while S3 goes from 1.93s to 12.4s (6x). The 10% VO fraction is enough to dominate
the tail, even though P50 barely changes.

GCP's knee drops from 0.8 c/s (FA=100%) to 0.4 c/s (FA=90%). Even at very low load
(IA=1000ms, 0.8 c/s), GCP only achieves 94.6% success — below the knee threshold.
The 170ms CAS plus 200ms manifest writes make each VO retry so expensive that even
a small fraction of VO pushes the sustainable throughput below 1 c/s.

At low load (IA=5000ms), the P50 is indistinguishable from FA=100% because 90% of
transactions are fast appends. But P99 is already 4-7x higher than FA=100% even
at this minimal load, showing that the tail impact of 10% VO is not a saturation
phenomenon — it's present at any load level.

## FA=50%, 1 Table

Adding 50% ValidatedOverwrite operations amplifies provider differences because VO
retries are more expensive: each retry reads manifest lists proportional to missed
snapshot versions, then writes updated manifests. Faster I/O reduces per-retry cost,
giving low-latency providers a compounding advantage.

| Provider | Knee (c/s) | P50 @ Knee | P95 @ Knee | P99 @ Knee |
|----------|------------|------------|------------|------------|
| s3x      | 15.7       | 0.34s      | 37.1s      | 69.0s      |
| s3       | 2.6        | 1.74s      | 72.3s      | 131.5s     |
| azurex   | 2.7        | 1.82s      | 77.4s      | 141.7s     |
| azure    | 2.0        | 2.70s      | 78.5s      | 147.3s     |
| gcp      | 0.4        | 4.77s      | 75.0s      | 132.2s     |

The knee throughputs drop slightly from FA=90%: azure falls from 2.6 to 2.0 c/s,
s3x from 16.0 to 15.7. But the bigger change is in the tails. P99 roughly doubles
from FA=90% to FA=50% across providers (36s to 69s for s3x, 66s to 131s for S3).

GCP's knee remains at 0.4 c/s — the same as FA=90%. At this load (IA=2000ms),
GCP achieves 99.9% success, but P99 is already 132s. The 50% VO workload is
unviable on GCP above 0.4 c/s.

Within the S3/azurex/azure cluster, the knee throughputs compress to 2.0-2.7 c/s,
but P99 spreads from 131s (S3) to 147s (azure). Azure's higher CAS sigma (0.82
vs 0.14) produces more extreme tails on retry sequences.

### The FA=100% → 90% → 50% progression

Comparing the three workload mixes reveals where VO impact is linear versus
nonlinear:

| Provider | Knee (FA=100%) | Knee (FA=90%) | Knee (FA=50%) |
|----------|---------------|---------------|---------------|
| s3x      | 16.1          | 16.0          | 15.7          |
| s3       | 2.7           | 2.7           | 2.6           |
| azurex   | 2.7           | 2.7           | 2.7           |
| azure    | 2.7           | 2.6           | 2.0           |
| gcp      | 0.8           | 0.4           | 0.4           |

For s3x, S3, and azurex, throughput at the knee is essentially unchanged across
all three mixes. The throughput ceiling is set by the commit pipeline (CAS + I/O),
not by the VO fraction. What changes is the tail: P99 goes from <1s to ~36-76s
to ~69-142s. VO doesn't reduce how many transactions you can push through — it
changes how long the unlucky ones take.

Azure and GCP are the exceptions. Azure's knee drops from 2.7 to 2.0 c/s at
FA=50% because its high CAS sigma means VO retry cascades occasionally produce
outliers extreme enough to consume retry budgets. GCP drops from 0.8 to 0.4 c/s
already at FA=90% because its base latencies are so high that even a single VO
retry doubles the commit time.

## Multi-Table Scaling

With 20 tables, per-table contention drops to ~5% of traffic per table under uniform
selection. The aggregate throughput increases and provider differences narrow, but
the ranking is preserved.

### FA=100%, 20 Tables

| Provider | Knee @ 1T (c/s) | Knee @ 20T (c/s) | Scaling Factor |
|----------|-----------------|-------------------|----------------|
| s3x      | 16.1            | 16.4              | 1.0x           |
| s3       | 2.7             | 8.1               | 3.0x           |
| azurex   | 2.7             | 8.1               | 3.0x           |
| azure    | 2.7             | 7.9               | 2.9x           |
| gcp      | 0.8             | 4.0               | 5.0x           |

S3 Express shows almost no multi-table scaling (16.1 to 16.4) because its 1-table
knee is already limited by aggregate I/O throughput, not per-table contention. The
system is I/O-bound even with 1 table.

The slower providers show 3-5x scaling. GCP benefits most (5x) because at 1 table
its high CAS latency creates severe contention — spreading load across tables
directly reduces retry rates. S3 and azurex scale identically (3x), confirming
that their similar CAS latencies produce similar contention dynamics.

This contrasts with exp4a/4b (instant catalog at 1ms), where 20 tables achieved
near-linear scaling up to the I/O ceiling (~41 c/s). In exp4c, the catalog CAS
scales with provider latency, preventing the same degree of multi-table benefit.

### FA=90%, 20 Tables

| Provider | Knee @ 1T (c/s) | Knee @ 20T (c/s) | P99 @ Knee (20T) |
|----------|-----------------|-------------------|-----------------|
| s3x      | 16.0            | 16.4              | 39.4s           |
| s3       | 2.7             | 8.0               | 228.9s          |
| azurex   | 2.7             | 8.1               | 245.4s          |
| azure    | 2.6             | 7.7               | 297.3s          |
| gcp      | 0.4             | 2.6               | 409.2s          |

Multi-table scaling for FA=90% is similar to FA=100% — throughput scales by the
same factors. But the P99 tails at 20 tables are *worse* than at 1 table despite
higher throughput. At 20 tables and 8 c/s (S3's knee), the system is processing
more total transactions, and the VO transactions that do conflict on the same table
face a longer queue of concurrent commits to wade through. S3's P99 at the 20-table
knee (228.9s, nearly 4 minutes) is 3.5x worse than at the 1-table knee (66.0s).

GCP's knee improves from 0.4 to 2.6 c/s with 20 tables — a 6.5x scaling factor,
the largest of any provider. But its P99 of 409s (nearly 7 minutes) at the knee
makes this throughput level impractical for anything latency-sensitive.

### FA=50%, 20 Tables

| Provider | Knee @ 1T (c/s) | Knee @ 20T (c/s) | P99 @ Knee (20T) |
|----------|-----------------|-------------------|-----------------|
| s3x      | 15.7            | 16.3              | 74.0s           |
| s3       | 2.6             | 7.8               | 410.6s          |
| azurex   | 2.7             | 7.8               | 439.4s          |
| azure    | 2.0             | 4.0               | 315.3s          |
| gcp      | 0.4             | 2.0               | 681.8s          |

GCP at 20 tables, FA=50%: P99 = 681.8s (over 11 minutes). This is the worst tail
latency in the entire experiment suite.

## Heatmap Observations

The per-provider heatmaps (num_tables x inter-arrival, filtered by FA ratio) reveal
the contention surface for each provider.

**S3 Express (FA=100%)**: Success remains >95% across a wide operating region.
Even at 1 table and IA=100ms, success is ~76%. At 10+ tables, success is near
100% for all load levels above IA=50ms. The heatmap is predominantly green,
confirming that s3x has headroom for most realistic workloads.

**S3 / Azurex (FA=100%)**: Similar heatmap shapes. The transition from green
(>95% success) to red (<50% success) occurs roughly between IA=100-200ms for
1 table. At 5+ tables, the green region extends to IA=50ms. The two providers
are difficult to distinguish in the heatmap view.

**GCP (FA=100%)**: The heatmap shows red (low success) starting at IA=200ms
for 1 table — a load level where all other providers maintain >95% success.
Even at 50 tables, GCP shows yellow (70-90% success) at IA=100ms. The
entire heatmap is shifted toward lower loads compared to other providers.

**Mixed workload heatmaps (FA=50%)**: Adding VO compresses the green region for
all providers. For GCP, the viable operating region shrinks to IA >= 500ms even
with 50 tables. For s3x, the green region remains wide, though P99 values
(not visible in the heatmap) are extreme in the transition zone.

## Cross-Cutting Findings

1. **Provider choice is a multiplier, not an offset.** S3 Express sustains 6x
   the throughput of S3 at the knee, and 20x the throughput of GCP. These are
   not constant differences — they widen under load because faster retry cycles
   compound.

2. **10% VO preserves throughput but destroys tail latency.** The FA=90% knee is
   within 5% of FA=100% for all providers. But P99 at the knee jumps 25-81x.
   A workload that is "90% appends" has the same throughput ceiling as pure
   appends, but the P99 experience is closer to 50/50 than to pure FA.

3. **CAS latency is necessary but not sufficient.** Azurex has a slightly higher
   CAS median than S3 (64 vs 61ms) but outperforms it at saturation because of
   faster manifest writes. The full commit pipeline — CAS + WRITE + (conditional
   READ on retry) — determines throughput, not CAS alone.

4. **Provider sigma shapes the tail.** S3's tight CAS distribution (sigma=0.14)
   produces predictable latency. Azure's wide distribution (sigma=0.82) produces
   worse P99 despite similar P50. Under VO retry cascades, high sigma compounds
   across retries, making tail behavior increasingly unpredictable.

5. **Multi-table scaling helps throughput but worsens tails.** Adding tables
   increases the knee by 3-5x for slower providers, but P99 at the new knee is
   often worse than P99 at the old knee. More tables mean more aggregate
   transactions, so the VO transactions that do collide face deeper contention.

6. **GCP is unsuitable for contention-sensitive workloads.** With 170ms CAS
   median and 80ms minimum latency, GCP's single-table knee is 0.8 c/s for
   pure FA and drops to 0.4 c/s with just 10% VO. Any workload expecting more
   than ~1 commit per second on a single table will require either a faster
   provider or a catalog service (REST/JDBC) that decouples CAS from storage I/O.
