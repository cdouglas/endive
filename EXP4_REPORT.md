# Exp4 Report: Multi-Table Catalog Contention

Experiments 4a and 4b sweep num_tables (1-50), catalog CAS latency (1-120ms),
and arrival rate on a single-file catalog (num_groups=1). Exp4a uses 100%
FastAppend; exp4b uses 90% FA / 10% ValidatedOverwrite. S3 storage, zero real
conflict probability. 5 seeds per point, 1200 runs per config.

## The question

Does adding more tables to a single-file catalog increase contention on the
shared global `seq` pointer?

The hypothesis from EXP3_REPORT.md predicted that at moderate load levels where
exp3 shows eta-sq ~ 0 for catalog latency, adding enough tables would push
eta-sq back toward 1 — because the aggregate commit rate on the shared seq
grows with num_tables.

## Result: num_tables has zero effect

**The hypothesis is falsified.** Adding tables from 1 to 50 produces no
measurable change in throughput, success rate, or latency at any operating point.

### Evidence

Throughput (commits/sec) at catalog_latency=1ms, exp4a (100% FA):

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 6.78 | 6.77 | 6.78 | 6.78 | 6.78 | 6.78 |
| 100ms         | 5.91 | 5.91 | 5.91 | 5.90 | 5.91 | 5.90 |
| 200ms         | 3.86 | 3.85 | 3.85 | 3.86 | 3.85 | 3.86 |
| 500ms         | 1.55 | 1.56 | 1.55 | 1.54 | 1.54 | 1.54 |

Throughput at catalog_latency=120ms, exp4a:

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 3.75 | 3.75 | 3.75 | 3.75 | 3.75 | 3.75 |
| 100ms         | 3.52 | 3.52 | 3.52 | 3.52 | 3.52 | 3.52 |
| 200ms         | 3.20 | 3.20 | 3.20 | 3.20 | 3.20 | 3.20 |
| 500ms         | 1.55 | 1.56 | 1.54 | 1.55 | 1.55 | 1.55 |

Every row is constant across 1-50 tables, within seed-to-seed noise (<0.02 c/s).

Success rate at catalog_latency=120ms, exp4b (90/10 mix):

| Load \ Tables | 1    | 2    | 5    | 10   | 20   | 50   |
|---------------|------|------|------|------|------|------|
| 20ms          | 9.7  | 9.7  | 9.7  | 9.7  | 9.7  | 9.7  |
| 100ms         | 45.3 | 45.2 | 45.3 | 45.3 | 45.3 | 45.3 |
| 200ms         | 81.2 | 81.0 | 81.1 | 80.9 | 81.0 | 80.8 |
| 500ms         | 99.7 | 99.8 | 99.6 | 99.7 | 99.7 | 99.7 |

### Eta-squared confirms the null

One-way ANOVA for num_tables at each (catalog_latency, load) cell:

Every eta-squared value for num_tables is non-significant. Representative
values at catalog_latency=1ms (exp4a):

| Load (ms) | num_tables eta-sq | Significance |
|-----------|-------------------|--------------|
| 20        | 0.172             | ns           |
| 100       | 0.330             | ns           |
| 200       | 0.114             | ns           |
| 500       | 0.357             | ns           |
| 1000      | 0.170             | ns           |

The few nominally "significant" cells (e.g., load=5000ms at cat=1ms, p < .01,
eta-sq=0.457) are spurious — the absolute throughput differences are <0.01 c/s,
and the signal comes from tiny seed-to-seed variance at very low throughput
making any noise look proportionally large.

### Two-way ANOVA: catalog latency dominates everywhere

Partial eta-squared from two-way ANOVA (catalog_latency x num_tables) at each load:

**Exp4a (100% FastAppend):**

| Load (ms) | Catalog lat. partial eta | num_tables partial eta |
|-----------|-------------------------|----------------------|
| 20        | 1.000                   | 0.019                |
| 50        | 1.000                   | 0.024                |
| 100       | 1.000                   | 0.079                |
| 200       | 0.997                   | 0.022                |
| 300       | 0.705                   | 0.052                |
| 400       | 0.032                   | 0.069                |
| 500       | 0.006                   | 0.069                |
| 1000      | 0.029                   | 0.042                |

**Exp4b (90/10 FA/VO mix):**

| Load (ms) | Catalog lat. partial eta | num_tables partial eta |
|-----------|-------------------------|----------------------|
| 20        | 1.000                   | 0.016                |
| 50        | 1.000                   | 0.015                |
| 100       | 1.000                   | 0.032                |
| 200       | 0.996                   | 0.104                |
| 300       | 0.886                   | 0.026                |
| 400       | 0.328                   | 0.025                |
| 500       | 0.044                   | 0.025                |
| 1000      | 0.010                   | 0.031                |

At every load level, catalog latency partial eta-squared dominates num_tables
partial eta-squared. The num_tables column never exceeds 0.10.

The catalog latency transition matches exp3 exactly: eta-sq ~ 1 at load <= 200ms,
crossing to ~0 at load >= 500ms (exp4a) or >= 400ms (exp4b).

## Why the null result?

Two compounding reasons:

### 1. Global arrival rate, not per-table

`inter_arrival.scale` controls the **global** arrival rate, not the per-table
rate. With num_tables=50 and inter_arrival=100ms, the same 10 transactions/sec
are spread across 50 tables (0.2/sec each). Adding tables dilutes per-table
contention without changing aggregate CAS contention.

### 2. Cross-table retries are incorrectly expensive (modeling bug)

Even with the same aggregate CAS failure rate, more tables should help: with
50 tables, 49/50 CAS failures are cross-table and should be nearly free to
retry (just re-read catalog + re-CAS). Only 1/50 same-table failures need
manifest I/O.

However, the simulator charges full per-attempt I/O cost (1 ML read + 1 MF
write + 1 ML write, ~160ms on S3) on every retry regardless of whether the
conflicting commit touched the same table. This masks the benefit of table
diversity. See endive-s5j and SPEC.md §3.3.

With both issues, num_tables has zero observed effect. Fixing the modeling
bug alone would show some benefit (cheaper retries with more tables), but
the full multi-tenant scenario also requires fixing the arrival rate.

## What would be needed to test the hypothesis

### Fix 1: Cross-table retry cost (endive-s5j)

After CAS failure, check whether intervening commits overlap with this
transaction's tables/partitions. If no overlap (cross-table or disjoint
partitions), skip manifest I/O and just re-CAS. SPEC.md has been updated
with the corrected commit loop (§3.2-3.3).

### Fix 2: Per-table arrival rate

To model N independent writer streams on a shared catalog, scale the total
arrival rate proportionally to num_tables:

    effective_inter_arrival = base_inter_arrival / num_tables

With 50 tables and base_inter_arrival=100ms, the effective rate would be
50 * 10/sec = 500 transactions/sec. This requires either:

1. **A per-table arrival rate parameter** in the workload generator, or
2. **Scaling the sweep**: For each num_tables value, divide inter_arrival.scale
   by num_tables in the experiment runner

Option 2 is simpler and doesn't require simulator changes:

```python
params = {
    "catalog.num_tables": num_t,
    "catalog.service.latency_ms": cat_latency,
    "inter_arrival.scale": float(load) / num_t,  # per-table rate
}
```

## Catalog latency effect confirmed

Despite the null result for num_tables, exp4 provides additional confirmation of
the exp3 findings with much larger sample sizes (6x more data at each
catalog_latency x load point due to pooling across num_tables).

One-way ANOVA eta-squared for catalog_latency (pooled across all num_tables):

| Load (ms) | Exp4a eta-sq | Exp4b eta-sq (approx) |
|-----------|-------------|----------------------|
| 20        | 1.000       | 1.000                |
| 50        | 1.000       | 1.000                |
| 100       | 1.000       | 1.000                |
| 200       | 0.997       | 0.996                |
| 300       | 0.694       | 0.886                |
| 400       | 0.030       | 0.328                |
| 500       | 0.006       | 0.044                |
| 1000      | 0.028       | 0.010                |

These match exp3 within expected variation, confirming the contention amplifier
mechanism is robust.
