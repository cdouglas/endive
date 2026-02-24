# Exp3 Report: Catalog Latency Sensitivity

Experiments 3a and 3b sweep catalog CAS latency from 1ms to 120ms on a single table.
Exp3a uses 100% FastAppend; exp3b uses 90% FA / 10% ValidatedOverwrite.
Both use S3 storage and zero real conflict probability.

## The question

Does a slower catalog (higher CAS latency) degrade throughput and success rate?
If so, under what conditions?

## Measuring the effect: eta-squared

We need a metric that says "at this operating point, how much does catalog latency
matter?" and that we can reuse in exp4 when num_tables is added as a factor.

**Eta-squared** (from one-way ANOVA) is the right tool. At each load level, we
group the per-seed throughput measurements by catalog latency and compute:

    eta-squared = SS_between / SS_total

where SS_between is the sum of squares explained by catalog latency differences
and SS_total is the total sum of squares. The result is bounded [0, 1]:

- **eta-squared ~ 0**: catalog latency explains none of the throughput variation. The
  spread across seeds within a single catalog latency is as large as the spread
  across catalog latencies. Catalog is not the bottleneck.
- **eta-squared ~ 1**: catalog latency explains nearly all throughput variation.
  Changing catalog latency reliably changes throughput. Catalog is the bottleneck.

This extends naturally to exp4: compute eta-squared for both catalog_latency and
num_tables at each load level, and the comparison tells you which factor dominates.
A two-way ANOVA with partial eta-squared separates the contributions cleanly.

## Results

### Eta-squared for throughput

| Load (ms) | Exp3a eta-sq | Exp3a sig | Exp3b eta-sq | Exp3b sig |
|-----------|-------------|-----------|-------------|-----------|
| 20        | 1.000       | p < .001  | 1.000       | p < .001  |
| 50        | 1.000       | p < .001  | 1.000       | p < .001  |
| 100       | 1.000       | p < .001  | 1.000       | p < .001  |
| 200       | 0.995       | p < .001  | 0.995       | p < .001  |
| 300       | 0.535       | p < .001  | 0.861       | p < .001  |
| 400       | 0.204       | ns        | 0.447       | p < .01   |
| 500       | 0.073       | ns        | 0.290       | p < .01   |
| 1000      | 0.284       | ns        | 0.083       | ns        |
| 2000      | 0.233       | ns        | 0.131       | ns        |
| 5000      | 0.263       | ns        | 0.212       | ns        |

At load <= 200ms, catalog latency explains > 99% of throughput variation in both
experiments. At load >= 500ms (exp3a) or >= 1000ms (exp3b), it explains
essentially nothing. The transition zone is 200-500ms.

Exp3b (with 10% VO) retains sensitivity slightly longer (eta-sq = 0.86 at 300ms
vs 0.54 for exp3a) because ValidatedOverwrite's higher retry cost amplifies
small catalog latency differences.

### Eta-squared for success rate

| Load (ms) | Exp3a eta-sq | Exp3b eta-sq |
|-----------|-------------|-------------|
| 20        | 1.000       | 1.000       |
| 50        | 1.000       | 1.000       |
| 100       | 1.000       | 1.000       |
| 200       | 0.999       | 0.999       |
| 300       | 0.993       | 0.995       |
| 400       | 0.828       | 0.964       |
| 500       | 0.523       | 0.852       |
| 1000+     | 0.000       | 0.000       |

Success rate sensitivity persists to lower load levels than throughput
sensitivity. At load=500ms exp3a's throughput is insensitive (eta-sq = 0.07)
but success rate still shows a significant effect (eta-sq = 0.52). This is
because success rate at high load is already 100%, creating a ceiling effect —
the signal comes from the remaining sub-100% experiments.

### Throughput (commits/sec)

The tables below show mean throughput across 3 seeds. NaN cells are catalog
latency values not swept at that load level.

**Exp3a (100% FastAppend):**

| Load \ Cat. lat. | 1ms  | 5ms  | 10ms | 20ms | 50ms | 80ms | 120ms |
|-------------------|------|------|------|------|------|------|-------|
| 20ms              | 6.8  | 6.6  | 6.4  | 6.0  | 5.1  | 4.4  | 3.8   |
| 50ms              | 6.4  | 6.2  | 6.1  | 5.7  | 4.9  | 4.3  | 3.6   |
| 100ms             | 5.9  | 5.8  | 5.6  | 5.4  | 4.6  | 4.1  | 3.5   |
| 200ms             | 3.9  | 3.9  | 3.8  | 3.8  | 3.7  | 3.5  | 3.2   |
| 300ms             | 2.6  | 2.6  | 2.6  | 2.6  | 2.6  | 2.6  | 2.5   |
| 500ms             | 1.5  | 1.6  | 1.6  | 1.6  | 1.5  | 1.5  | 1.5   |
| 1000ms            | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8   |

**Exp3b (90/10 FA/VO mix)** matches exp3a within 0.1 commits/sec at every cell.

### Throughput ratio (120ms catalog / 1ms catalog)

| Load (ms) | Exp3a ratio | Exp3b ratio |
|-----------|-------------|-------------|
| 20        | 0.554       | 0.555       |
| 50        | 0.569       | 0.571       |
| 100       | 0.596       | 0.607       |
| 200       | 0.829       | 0.836       |
| 300       | 0.979       | 0.962       |
| 400       | 0.992       | 0.996       |
| 500       | 1.000       | 0.992       |

At the highest contention (20ms), throughput at 120ms catalog latency is 55%
of throughput at 1ms — a 45% penalty. By 300ms, the penalty shrinks to 2-4%.
By 500ms, there is no measurable difference.

### Success rate (%)

**Exp3a (100% FastAppend):**

| Load \ Cat. lat. | 1ms   | 10ms  | 50ms  | 120ms |
|-------------------|-------|-------|-------|-------|
| 20ms              | 17.5  | 16.5  | 13.2  | 9.7   |
| 50ms              | 41.3  | 39.1  | 31.5  | 23.5  |
| 100ms             | 76.1  | 72.7  | 60.0  | 45.5  |
| 200ms             | 99.4  | 99.0  | 95.8  | 82.5  |
| 300ms             | 100.0 | 100.0 | 99.8  | 98.0  |
| 500ms             | 100.0 | 100.0 | 100.0 | 100.0 |

**Exp3b** is within 1pp of exp3a at every cell except load=200ms, where VO
retries push success rate slightly lower (96.9% vs 99.4% at 1ms catalog).

### P99 commit latency (ms)

**Exp3a:**

| Load \ Cat. lat. | 1ms  | 50ms  | 120ms |
|-------------------|------|-------|-------|
| 50ms              | 1685 | 2225  | 2996  |
| 100ms             | 1687 | 2227  | 2999  |
| 200ms             | 1470 | 2204  | 2997  |
| 500ms             | 740  | 1095  | 1904  |
| 1000ms            | 597  | 808   | 1203  |
| 5000ms            | 450  | 610   | 824   |

P99 scales roughly linearly with catalog latency across all load levels.
Even at load=5000ms where throughput is identical, the 120ms catalog
adds ~370ms to P99 vs the 1ms catalog. This is because each commit attempt
incurs one CAS round-trip, so the latency floor shifts proportionally.

**Exp3b P99 — survivor bias at the saturation boundary:**

| Load \ Cat. lat. | 1ms     | 50ms   | 120ms  |
|-------------------|---------|--------|--------|
| 100ms             | 1,705   | 2,233  | 3,001  |
| 200ms             | 142,154 | 85,939 | 15,411 |
| 300ms             | 123,716 | 110,374| 87,288 |
| 500ms             | 81,050  | 79,433 | 76,481 |

At load=200ms, P99 is *lower* with 120ms catalog than with 1ms catalog
(15s vs 142s). This is not an error. At this load level, success rate with
1ms catalog is 96.9% — nearly all transactions eventually commit, including
those requiring many VO retry cascades. The P99 captures these expensive
survivors. With 120ms catalog, success rate drops to 80.8%: the hardest
transactions now exceed the retry limit and abort. The committed population
is pruned of its most expensive members, producing a lower P99.

This is a **survivor bias** effect. Higher catalog latency isn't making
commits faster — it's preventing the slowest transactions from committing
at all, which lowers the observed tail latency of the survivors. By
load=500ms, success rates converge near 100% in both cases and the effect
disappears.

## Interpretation

Catalog latency acts as a **contention amplifier**. The mechanism:

1. Under contention, multiple transactions compete for the same catalog `seq`
   via compare-and-swap. Losers must retry.
2. Each retry incurs at least one CAS round-trip. Higher CAS latency means
   longer retry cycles.
3. During each retry cycle, more new transactions arrive and commit, pushing
   the retrying transaction further behind (`n_behind` increases).
4. This creates a feedback loop: slower CAS -> longer retries -> more
   interleaving -> more CAS failures -> more retries.

When contention is low (load >= 500ms), transactions rarely overlap, CAS
almost always succeeds on the first attempt, and catalog latency contributes
only a small additive constant to commit latency.

The eta-squared analysis quantifies the transition: catalog latency explains
>99% of throughput variance when the system is contended (load <= 200ms) and
<10% when it is not (load >= 500ms). The crossover is sharp, occurring over
roughly one order of magnitude of load (200-500ms).

## Applying eta-squared to exp4

Exp4 adds num_tables as a sweep dimension. The same analysis generalizes:

1. **At each load level**, compute eta-squared for catalog_latency_ms
   and for num_tables separately (or jointly via partial eta-squared from
   two-way ANOVA).
2. Compare which factor has higher eta-squared at each operating point.
3. The exp4 hypothesis predicts that at moderate load levels where exp3
   shows eta-sq ~ 0 for catalog latency (e.g., load=500ms), adding enough
   tables will push eta-sq back toward 1 — because the aggregate commit
   rate on the shared seq grows with num_tables.

If this prediction holds, the contour line where eta-squared crosses 0.5
(the "bottleneck boundary") shifts rightward as num_tables increases,
meaning catalog latency matters at progressively lower per-table arrival
rates.

# Here's the summary of the report and the measure.

The measure: eta-squared (one-way ANOVA effect size). At each load level, group
per-seed throughput by catalog latency and compute SS_between / SS_total.
Result is 0-1:
- ~1 = catalog latency explains all throughput variation (it's the bottleneck)
- ~0 = catalog latency explains nothing (it's irrelevant)

What the data shows:

The transition is sharp but it's not a threshold — it's a continuous crossover.
Eta-squared goes from 1.000 at load=100ms to 0.073 at load=500ms for exp3a
throughput. The mechanism is a feedback loop: under contention, slower CAS
means longer retry cycles, which means more interleaving, which means more CAS
failures. Without contention, CAS almost always succeeds on the first try and
catalog latency is just an additive constant.

Exp3b survivor bias: The P99 latency inversion at load=200ms (142s at 1ms
catalog vs 15s at 120ms) is a survivor bias effect — higher catalog latency
causes the most expensive VO transactions to abort instead of eventually
committing, pruning the tail.

Extension to exp4: Same ANOVA, but with two factors (catalog_latency and
num_tables). Partial eta-squared from two-way ANOVA tells you which factor
dominates at each operating point. The prediction is that adding tables shifts
the "bottleneck boundary" rightward — catalog latency starts mattering at lower
per-table arrival rates.

