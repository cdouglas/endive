# DES Engine Profiling Report

**Date**: 2026-03-01
**Data**: 233 profiled runs across exp1–exp3b, S3 provider, 1 table, scale 20–5000ms

## Executive Summary

Wall-clock time is almost entirely explained by DES event count (R² = 0.987). The DES engine processes events at a roughly constant rate of ~75–107K events/sec, with moderate degradation (~30%) under extreme queue depth. The dominant cost driver is not the arrival rate itself, but the number of **retries and conflict resolution I/O operations** each transaction generates — a mixed workload with ValidatedOverwrite at scale=20ms produces 270M events (1 hour wall-clock), while a FastAppend-only workload at the same arrival rate produces 8M events (2 minutes wall-clock).

## Key Findings

### 1. Cost per event is constant

```
wall_clock_s = 13.5 µs × event_count − 12.5
R² = 0.987
```

Event count alone explains 98.7% of wall-clock variance. The per-event cost of ~13.5 µs reflects SimPy's heap operations (heappush/heappop), generator frame switching, and Python interpreter overhead.

### 2. DES rate is stable across load levels

| Arrival scale (ms) | DES rate (events/sec) | Queue depth | Active processes |
|--------------------:|----------------------:|------------:|-----------------:|
| 5000 | 76,600 | 66 | 90 |
| 1000 | 94,600 | 325 | 431 |
| 500 | 97,700 | 626 | 821 |
| 200 | 106,900 | 1,637 | 2,126 |
| 100 | 99,000 | 3,169 | 4,140 |
| 50 | 93,900 | 6,635 | 8,593 |
| 20 | 74,500 | 16,701 | 21,606 |

The rate peaks around scale=200ms (~107K/s) and drops ~30% at scale=20ms (~75K/s) where 20K+ SimPy processes are active. This degradation is from heap operations on a larger priority queue (O(log n) per event with n=20K+).

### 3. Events per transaction vary 100× by workload

| Workload | Events/txn (scale=20ms) | Events/txn (scale=5000ms) |
|----------|------------------------:|--------------------------:|
| exp1 FA baseline | 43 | 12 |
| exp3a catalog FA | 43 | 11 |
| exp3b catalog mix (90/10) | 212 | 16 |
| exp2 heatmap (varies) | 856 | 29 |

At low contention (scale=5000ms), all workloads converge to ~11–29 events/txn (the base cost: arrival timeout + catalog read + runtime yield + commit + a few I/O operations). At high contention (scale=20ms), retries multiply event count — each retry adds catalog read + CAS + conflict resolution I/O yields.

### 4. The exp2 heatmap dominates wall-clock time

At scale=20ms, the top 6 slowest runs are ALL exp2_mix_heatmap configs:

| FA weight | Wall-clock (s) | Events | DES rate |
|----------:|---------------:|-------:|---------:|
| 0.0 (all VO) | 3,624 | 278M | 78K |
| 0.1 | 3,716 | 270M | 72K |
| 0.3 | 3,341 | 240M | 72K |
| 0.5 | 2,765 | 186M | 68K |
| 0.7 | 1,951 | 120M | 63K |
| 0.8 | 1,377 | 86M | 65K |
| 0.9 | 728 | 46M | 65K |
| 1.0 (all FA) | 144 | 8M | 56K |

The 26× difference (3,716s vs 144s) between all-VO and all-FA is entirely event volume: ValidatedOverwrite transactions abort on real conflicts and retry with expensive I/O convoy reads, generating ~856 events/txn vs ~43 for FastAppend.

### 5. Simulation speed stays above real-time (mostly)

Only 118 of 14,000+ sampling intervals across all 233 runs dipped below real-time (sim_speed < 1.0), and 118 of those are in scale=20ms runs. Even the worst run (3,716s wall-clock) maintained sim_speed ≥ 0.6 — the simulator never falls catastrophically behind.

### 6. Queue depth tracks active processes almost exactly

Correlation between `queue_depth_mean` and `peak_processes` is 1.000. Each active transaction process has approximately one pending event in the queue at any time, consistent with the generator-based I/O model where processes are suspended waiting on a single `env.timeout()`.

## Cost Model

**To estimate wall-clock time for a configuration:**

```
events_per_txn ≈ base_events + retries × events_per_retry

where:
  base_events ≈ 11 (arrival + read + runtime + commit + I/O)
  retries depend on contention (arrival_rate × catalog_latency)
  events_per_retry ≈ 5 (FA) to 30+ (VO with I/O convoy)

total_events = (sim_duration_ms / inter_arrival_scale) × events_per_txn
wall_clock_s ≈ total_events × 13.5 µs
```

**Example predictions:**
- FA-only, scale=20ms: 180K txns × 43 events = 7.7M events → 104s (actual: 141s)
- Mixed 90/10, scale=20ms: 180K × 212 = 38M events → 516s (actual: 447s)
- All-VO, scale=20ms: 180K × 856 = 154M events → 2,080s (actual: 2,206s mean)

The model is approximate because events_per_txn itself depends on contention level, which is load-dependent.

## Implications for Experiment Runs

1. **Full experiment suite timing**: The 8,050-run suite at `--parallel 8` is dominated by ~80 exp2_mix_heatmap runs at scale ≤ 50ms. The ~20 runs at scale=20ms with VO weight ≥ 0.5 each take 30–60 minutes wall-clock. Total: ~24–36 hours with 8 cores.

2. **No optimization opportunity in the simulator itself**: The DES engine operates at near-maximum CPython throughput (75–107K events/sec). The ~30% degradation at high queue depth is inherent to heap-based priority queues. PyPy could deliver 3–5× improvement but risks SimPy/NumPy compatibility issues.

3. **The expensive runs are expensive because the workload is expensive**: High-VO, high-arrival-rate configs generate massive retry cascades. This is the correct simulation behavior — those configs model genuine pathological scenarios. Reducing event count would mean changing the model.

## Data Location

- Per-run profiles: `experiments/<experiment>/<seed>/.profile.json`
- Aggregated CSV: `experiments/.profile.csv` (233 rows)
- Generated by: `python scripts/run_all_experiments.py --profile`
