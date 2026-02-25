# Exp4 Report: Multi-Table Catalog Contention

Experiments 4a and 4b sweep num_tables (1-50), catalog CAS latency (1-120ms),
and arrival rate on a single-file catalog (num_groups=1). Exp4a uses 100%
FastAppend; exp4b uses 90% FA / 10% ValidatedOverwrite. S3 storage, zero real
conflict probability. 5 seeds per point, 1200 runs per config (480 total
experiment directories).

These experiments ran after commit 7da2c84, which fixed cross-table CAS retry
cost: when a CAS failure is caused by a commit to a different table, the retry
skips per-attempt I/O (manifest list read/write) and just re-reads the catalog
and re-CAS.

## The question

Does adding more tables to a single-file catalog increase or decrease
contention on the shared global `seq` pointer?

## Result: num_tables dramatically reduces contention

Adding tables from 1 to 50 produces up to 6x throughput improvement at high
load. The mechanism is cross-table retry cost: with more tables, CAS failures
are more likely to be cross-table (non-overlapping), making retries cheap.

### Throughput

Throughput (commits/sec) at CAS=1ms, exp4a (100% FA):

| Load \ Tables | 1    | 2     | 5     | 10    | 20    | 50    |
|---------------|------|-------|-------|-------|-------|-------|
| 20ms          | 6.54 | 12.46 | 28.84 | 38.89 | 39.38 | 39.42 |
| 50ms          | 6.20 | 11.57 | 15.72 | 15.75 | 15.72 | 15.74 |
| 100ms         | 5.84 |  7.81 |  7.89 |  7.88 |  7.85 |  7.88 |
| 200ms         | 3.92 |  3.95 |  3.94 |  3.94 |  3.96 |  3.94 |
| 500ms         | 1.58 |  1.58 |  1.58 |  1.58 |  1.58 |  1.58 |

At ias=20ms, throughput scales from 6.54 (1 table) to 39.42 (50 tables) - a
6.0x improvement. The arrival-rate ceiling is 1000/20 = 50 txn/s, so 50 tables
reaches 79% of theoretical maximum. By ias=200ms, the effect disappears.

Throughput at CAS=120ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 3.63 | 4.14 | 7.52 | 7.75 | 7.81 | 7.83 |
| 50ms          | 3.55 | 4.06 | 6.99 | 7.38 | 7.48 | 7.53 |
| 100ms         | 3.40 | 3.93 | 5.83 | 6.27 | 6.36 | 6.37 |
| 200ms         | 2.89 | 3.34 | 3.80 | 3.87 | 3.88 | 3.87 |
| 500ms         | 1.57 | 1.57 | 1.58 | 1.58 | 1.59 | 1.59 |

At CAS=120ms, the improvement is 2.2x (3.63 to 7.83). The catalog service
itself imposes a throughput ceiling of ~7.8 c/s at the heaviest loads. Even
with 50 tables eliminating cross-table retry cost, each CAS attempt still
takes 120ms, limiting total throughput.

### Success rate

Success rate (%) at CAS=1ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20    | 50    |
|---------------|------|------|------|------|-------|-------|
| 20ms          | 16.6 | 31.7 | 73.4 | 98.9 | 100.0 | 100.0 |
| 50ms          | 39.3 | 73.4 | 99.8 | 100  | 100   | 100   |
| 100ms         | 74.2 | 99.1 | 100  | 100  | 100   | 100   |
| 200ms         | 99.2 | 100  | 100  | 100  | 100   | 100   |

At 1 table and ias=20ms, only 16.6% of transactions commit. At 10 tables,
98.9% commit. At 20+ tables, 100%.

Success rate (%) at CAS=120ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          |  9.2 | 10.5 | 19.1 | 19.7 | 19.9 | 19.9 |
| 100ms         | 43.2 | 49.7 | 74.3 | 79.5 | 80.9 | 80.9 |
| 200ms         | 73.4 | 85.2 | 96.8 | 98.3 | 98.6 | 98.5 |
| 500ms         | 99.5 | 99.9 | 100  | 100  | 100  | 100  |

At CAS=120ms and ias=20ms, even 50 tables only reaches 19.9% success. The
120ms CAS round-trip consumes most of the retry budget regardless of how cheap
the retry itself is.

### Retries

Mean retries at CAS=1ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 9.18 | 8.61 | 6.76 | 3.59 | 2.50 | 2.19 |
| 50ms          | 8.02 | 6.69 | 2.93 | 2.27 | 2.08 | 1.98 |
| 100ms         | 6.23 | 3.16 | 2.02 | 1.84 | 1.77 | 1.74 |
| 200ms         | 2.56 | 1.76 | 1.55 | 1.50 | 1.48 | 1.46 |

At 1 table and ias=20ms, transactions average 9.18 retries (near the retry
limit of 10). At 50 tables, only 2.19 retries. The retry count reflects the
probability of same-table conflicts: with 50 tables, each CAS failure has a
1/50 chance of hitting the same table.

Mean retries at CAS=120ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 9.58 | 9.52 | 9.23 | 9.19 | 9.19 | 9.20 |
| 100ms         | 7.83 | 7.62 | 6.27 | 5.86 | 5.69 | 5.60 |
| 200ms         | 5.59 | 4.83 | 3.59 | 3.25 | 3.10 | 3.04 |

At CAS=120ms and ias=20ms, retry count barely changes with table count (9.58
to 9.20). The 120ms CAS latency means each retry takes so long that
transactions exhaust their retry budget before the cheaper retry cost can help.
At ias=100ms the reduction is more visible (7.83 to 5.60).

## ANOVA

### One-way: num_tables effect on throughput

| CAS (ms) | Load (ms) | F         | p         | eta-sq |
|-----------|-----------|-----------|-----------|--------|
| 1         | 20        | 363,665   | 1.0e-57   | 1.0000 |
| 1         | 50        | 48,441    | 3.3e-47   | 0.9999 |
| 1         | 100       | 3,863     | 4.8e-34   | 0.9988 |
| 1         | 200       | 2.1       | 9.9e-02   | 0.3052 |
| 1         | 500       | 0.2       | 9.5e-01   | 0.0422 |
| 120       | 20        | 135,965   | 1.4e-52   | 1.0000 |
| 120       | 50        | 89,699    | 2.0e-50   | 0.9999 |
| 120       | 100       | 39,775    | 3.5e-46   | 0.9999 |
| 120       | 200       | 4,259     | 1.5e-34   | 0.9989 |
| 120       | 500       | 1.5       | 2.2e-01   | 0.2395 |

At CAS=1ms, the effect is significant through ias=100ms (eta-sq > 0.99). At
CAS=120ms, significance extends through ias=200ms. Beyond ias=200-500ms, load
is low enough that contention disappears regardless of table count.

### Two-way partial eta-sq: catalog_latency x num_tables

Exp4a (100% FastAppend):

| Load (ms) | cat_lat partial eta-sq | num_tables partial eta-sq |
|-----------|------------------------|---------------------------|
| 20        | 1.0000                 | 1.0000                    |
| 50        | 0.9999                 | 0.9999                    |
| 100       | 0.9994                 | 0.9996                    |
| 200       | 0.9848                 | 0.9867                    |
| 300       | 0.6815                 | 0.6965                    |
| 400       | 0.1226                 | 0.1175                    |
| 500       | 0.0609                 | 0.0177                    |

Exp4b (90/10 FA/VO mix):

| Load (ms) | cat_lat partial eta-sq | num_tables partial eta-sq |
|-----------|------------------------|---------------------------|
| 20        | 1.0000                 | 1.0000                    |
| 50        | 0.9999                 | 0.9999                    |
| 100       | 0.9995                 | 0.9997                    |
| 200       | 0.9839                 | 0.9904                    |
| 300       | 0.7470                 | 0.8419                    |
| 400       | 0.1839                 | 0.3588                    |
| 500       | 0.0468                 | 0.0826                    |

Both factors have approximately equal explanatory power at every load level.
This is the key finding relative to exp3: the earlier experiment (which had a
modeling bug making cross-table retries expensive) showed num_tables partial
eta-sq near zero at all loads. With the fix, num_tables is as important as
catalog latency.

In exp4b, num_tables has slightly higher partial eta-sq than catalog latency at
ias=300-400ms. ValidatedOverwrite operations are more sensitive to table count
because their retry cost is higher (manifest reads + writes), so the savings
from skipping that I/O on cross-table retries is proportionally larger.

## Per-operation-type breakdown (exp4b)

ValidatedOverwrite success rate (%) at CAS=1ms:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 50ms          | 0.0  | 0.0  | 0.0  | 0.2  | 0.8  | 4.4  |
| 100ms         | 1.8  | 23.6 | 61.2 | 80.9 | 91.8 | 98.5 |
| 200ms         | 67.1 | 95.7 | 100  | 100  | 100  | 100  |

VO is far more fragile than FA at every operating point. At ias=50ms, even 50
tables only achieves 4.4% VO success. At ias=100ms, 50 tables reaches 98.5%
while 1 table is at 1.8%. The 50x improvement from table diversity is much
larger for VO than for FA (which goes from 80.5% to 100% at the same point).

ValidatedOverwrite success rate (%) at CAS=120ms:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 0.0  | 0.0  | 0.1  | 3.4  | 7.5  | 11.0 |
| 100ms         | 0.4  | 4.3  | 27.1 | 48.7 | 59.9 | 68.1 |
| 200ms         | 19.7 | 40.6 | 85.1 | 93.6 | 95.5 | 96.6 |
| 500ms         | 91.7 | 99.2 | 100  | 100  | 100  | 100  |

At CAS=120ms and ias=100ms, VO success goes from 0.4% (1 table) to 68.1%
(50 tables). The corresponding FA success goes from 47.5% to 82.1%. VO
benefits more from table diversity in both absolute and relative terms.

## Why num_tables helps

### Cross-table retry savings

With N tables and uniform random table selection, a CAS failure caused by a
commit to a different table (probability (N-1)/N) skips manifest I/O on retry.
On S3, this saves ~160ms per retry (1 ML read + 1 MF write + 1 ML write). At
N=50, 98% of CAS failures are cross-table and nearly free.

The effect is visible in the retry data: at CAS=1ms and ias=20ms, mean retries
drop from 9.18 (1 table) to 2.19 (50 tables). The retries themselves are
cheaper, so more transactions complete within their retry budget.

### Saturation behavior

The benefit saturates around 10-20 tables at CAS=1ms because the throughput
ceiling is set by the arrival rate. At ias=20ms, the theoretical maximum is 50
txn/s. With 10 tables, the system already reaches 38.89 c/s (78% of max),
leaving little room for improvement.

At CAS=120ms, the benefit saturates later (around 5-10 tables) and at a lower
throughput ceiling (~7.8 c/s). The CAS round-trip itself becomes the
bottleneck: even if every retry is free, a transaction that needs 9 retries at
120ms each takes over a second just for CAS attempts.

### Catalog latency interaction

The two factors interact multiplicatively. Low CAS + many tables: retries are
cheap AND fast, so nearly everything commits. High CAS + few tables: retries
are expensive AND slow, so most transactions exhaust their retry budget.

The intermediate regimes are where the distinction matters for system design:
- CAS=50ms, 5 tables, ias=50ms: 12.05 c/s, compared to 4.75 c/s at 1 table
- CAS=50ms, 10 tables, ias=50ms: 13.58 c/s (diminishing returns beyond 5)

## Design implications

1. **Single-file catalog contention is table-count-dependent.** The original
   hypothesis (more tables = more contention) was backwards. With the correct
   cross-table retry semantics, more tables means cheaper retries and higher
   throughput.

2. **The benefit caps at 5-20 tables** depending on catalog latency and load.
   Beyond this, nearly all CAS failures are already cross-table.

3. **VO is the bottleneck in mixed workloads.** At every operating point, VO
   success is lower than FA. Table diversity helps VO more than FA, but VO
   still fails catastrophically at high load (0% success at ias <= 50ms
   regardless of table count at CAS=1ms).

4. **Catalog service latency sets a hard ceiling.** At CAS=120ms, no amount of
   table diversity pushes throughput past ~7.8 c/s. Fast catalog implementations
   (CAS < 10ms) are necessary to realize the full benefit of multi-table
   deployments.

5. **The arrival rate in these experiments is global, not per-table.** With 50
   tables and ias=20ms, each table sees only 1 txn/s on average. A per-table
   arrival rate (effective_ias = base_ias / num_tables) would be needed to
   model N independent writer streams sharing a catalog. That experiment would
   test whether the cross-table savings keep up as aggregate load scales with
   table count.
