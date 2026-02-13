---
layout: page
title: "Appendix: Simulator Details"
permalink: /appendix/simulator
---

# Appendix: Simulator Implementation Details

This appendix provides technical details about the discrete-event simulator used to study Apache Iceberg's optimistic concurrency control. For complete implementation, see the [simulator code](https://github.com/cdouglas/endive) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## Probability Distributions

The simulator models stochastic workload characteristics using several probability distributions:

**Transaction Runtime:** Lognormal distribution with minimum threshold. Default: $$T_{\text{min}} = 30{,}000$$ ms, mean = 180,000 ms, $$\sigma = 1.5$$ captures right-skewed execution times.

**Inter-Arrival Time:** Exponential distribution (Poisson arrivals). The `scale` parameter controls offered load---lower values = higher contention.

**Storage Latencies:** Normal distributions truncated at $$T_{\text{min}} = 1$$ ms to prevent unrealistic zero-latency operations. Models cloud storage variability (e.g., S3).

**Conflicting Manifests:** Number of manifest files requiring merge during real conflicts. Three distributions: fixed (deterministic), exponential (default: mean=3, range=[1,10]), or uniform.

**Table Selection:** Truncated Zipf distribution. Parameter `ntable.zipf = 10.0` enforces single-table transactions; `ntable.zipf = 1.5` yields multi-table workload averaging 2-3 tables per transaction.

## Experiment Parameters

All experiments simulate 1 hour (3,600,000 ms) and sweep `inter_arrival.scale` across [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms with 3-5 random seeds per configuration.

| Parameter | Exp 2.1<br/>Single Table<br/>False Conflicts | Exp 2.2<br/>Multi-Table<br/>False Conflicts | Exp 3.1<br/>Single Table<br/>Real Conflicts | Exp 3.2<br/>Manifest<br/>Distribution | Exp 3.3<br/>Multi-Table<br/>Real Conflicts | Exp 4.1<br/>Backoff<br/>Comparison |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **Tables ($$T$$)** | 1 | **[1, 2, 5, 10, 20, 50]** | 1 | 1 | **[1, 2, 5, 10, 20]** | 1 |
| **Groups ($$G$$)** | 1 | $$G = T$$ | 1 | 1 | $$G = T$$ | 1 |
| **Real Conflict Prob ($$p$$)** | 0.0 | 0.0 | **[0.0, 0.1, 0.2,<br/>0.3, 0.5, 0.7, 1.0]** | 0.5 | **[0.0, 0.1,<br/>0.3, 0.5]** | 0.0 |
| **Conflicting Manifests** | exp(3)<br/>[1,10] | exp(3)<br/>[1,10] | exp(3)<br/>[1,10] | **fixed: [1, 5, 10]<br/>exp: [1, 3, 5, 10]<br/>uniform: [1,10]** | exp(3)<br/>[1,10] | exp(3)<br/>[1,10] |
| **Txn Runtime** | LN(180k, 1.5)<br/>min=30k ms | LN(180k, 1.5)<br/>min=30k ms | LN(180k, 1.5)<br/>min=30k ms | LN(180k, 1.5)<br/>min=30k ms | LN(180k, 1.5)<br/>min=30k ms | LN(180k, 1.5)<br/>min=30k ms |
| **Inter-Arrival** | **Exp([10--5000])** | **Exp([10--5000])** | **Exp([10--5000])** | **Exp([10--5000])** | **Exp([10--5000])** | **Exp([10--5000])** |
| **$$T_{\text{CAS}}$$** | $$1 \pm 0.1$$ ms | $$1 \pm 0.1$$ ms | $$1 \pm 0.1$$ ms | $$1 \pm 0.1$$ ms | $$1 \pm 0.1$$ ms | $$1 \pm 0.1$$ ms |
| **$$T_{\text{manifest-list}}$$** | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms |
| **$$T_{\text{manifest-file}}$$** | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms | $$50 \pm 5$$ ms |
| **Max Parallel Reads** | 4 | 4 | 4 | 4 | 4 | 4 |
| **Retry Backoff** | None | None | None | None | None | **None (baseline)<br/>Conservative<br/>Moderate<br/>Aggressive** |

**Notes:**
- **Bold** parameters are swept/varied in that experiment
- exp($$\mu$$) = exponential distribution with mean $$\mu$$
- LN($$\mu$$, $$\sigma$$) = lognormal with mean $$\mu$$ and shape $$\sigma$$
- Storage latencies use normal distributions: mean $$\pm$$ stddev
- $$G = T$$ means table-level isolation (each table is its own conflict group)

### Backoff Configurations (Exp 4.1)

Backoff time after $$r$$ retries: $$T_{\text{backoff}}(r) = \min(\text{base} \times \text{multiplier}^r, \text{max}) \times (1 + \mathcal{U}(-0.1, 0.1))$$

| Strategy | Base (ms) | Multiplier | Max (ms) |
|----------|-----------|------------|----------|
| None | --- | --- | --- |
| Conservative | 100 | $$1.5 \times$$ | 5000 |
| Moderate | 50 | $$2.0 \times$$ | 2000 |
| Aggressive | 10 | $$2.0 \times$$ | 1000 |

All strategies include $$\pm 10\%$$ jitter to prevent thundering herd.

## Key Cost Formulas

**False conflict cost** (same table modified, no data overlap):

A false conflict occurs when another transaction committed changes to the same table, but to different partitions (no overlapping data). The transaction must still create a new snapshot that combines both sets of manifest file pointers.

*Traditional (rewrite) mode:*
$$
E[T_{\text{false}}] = T_{\text{CAS}} + T_{\text{metadata-root-read}} + T_{\text{ML-read}} + T_{\text{ML-write}} + T_{\text{table-metadata}}
$$

With defaults: $$E[T_{\text{false}}] \approx 1 + 1 + 50 + 50 + 10 = 112$$ ms

*ML+ (append) mode:*
$$
E[T_{\text{false}}] = T_{\text{CAS}} + T_{\text{metadata-root-read}} + T_{\text{table-metadata}}
$$

With defaults: $$E[T_{\text{false}}] \approx 1 + 1 + 10 = 12$$ ms

The ML+ mode saves ~100ms per false conflict because the tentative manifest list entry (appended before the CAS attempt) is still valid—readers filter it by committed transaction list until the CAS succeeds.

**Real conflict cost** ($$n$$ conflicting manifests, parallelism=4):

$$
E[T_{\text{real}}] = T_{\text{CAS}} + T_{\text{metadata-root}} + T_{\text{ML-read}} + \left\lceil \frac{n}{4} \right\rceil \times 2 \times T_{\text{manifest-file}} + T_{\text{ML-write}}
$$

With defaults ($$n=3$$): $$E[T_{\text{real}}] \approx 1 + 1 + 50 + \lceil 3/4 \rceil \times 2 \times 50 + 50 = 202$$ ms

Real conflicts are $$\approx$$**2$$\times$$ more expensive** than false conflicts in rewrite mode (due to manifest file I/O), and $$\approx$$**17$$\times$$ more expensive** than false conflicts in ML+ mode.

**Multi-table real conflict probability** (Exp 3.3): Transaction touching $$k$$ tables with per-table conflict probability $$p$$:

$$
P(\text{≥1 real conflict}) = 1 - (1-p)^k
$$

Example: $$p=0.3$$, $$k=3$$ $$\rightarrow$$ $$P \approx 0.657$$ (conflicts compound across tables).

## Warmup and Cooldown Periods

All experiments exclude warmup and cooldown periods to measure steady-state performance. The simulator calculates warmup duration as:

$$
T_{\text{warmup}} = \min\left(900 \text{ s}, \frac{T_{\text{sim}}}{4}\right)
$$

with symmetric cooldown period at the end. For 1-hour simulations: 900s warmup + 1800s active measurement + 900s cooldown. Transactions that start outside the active window or span boundaries are excluded from statistics. This filtering removes transient startup effects and ensures commit queues drain naturally.

See `saturation_analysis.py:compute_warmup_duration()` for implementation.

## Statistical Methodology

- **Replication:** 3-5 independent seeds per configuration
- **Error bands:** Standard deviation across seeds
- **Saturation threshold:** Success rate < 95% or P95 latency > $$2 \times$$ P50
- **Numerical precision:** IEEE 754 double ($$\approx$$16 decimal digits)
- **Validation:** See `tests/test_numerical_accuracy.py` and `tests/test_statistical_rigor.py`

## Implementation

- **Simulator speed:** $$\approx 1000 \times$$ real-time (1 hour in 3.6 seconds)
- **Core engine:** [`endive/main.py`](../endive/main.py) (~1100 lines)
- **Analysis pipeline:** [`endive/saturation_analysis.py`](../endive/saturation_analysis.py) (~1700 lines)
- **Experiment configs:** [`experiment_configs/`](../experiment_configs/)
- **Architecture docs:** [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
