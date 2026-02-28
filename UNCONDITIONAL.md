# Unconditional GET and PUT Latency Distributions

All latencies in milliseconds. Values below each provider's `min_latency_ms` floor are clamped.

## Sources

| Provider | GET source | PUT source |
|----------|-----------|------------|
| S3 Express | YCSB benchmark (June 2025) | YCSB benchmark (June 2025) |
| S3 Standard | In-region microbenchmark (Dec 2025), p50/p90/p99 measured | In-region benchmark (Mar 2025), ~500 KiB objects, base derived |
| Azure Premium | In-region field test (May 2023), min/avg/max from n=100 | Estimated from 1 MB upload avg/max + overhead model |
| Azure Standard | In-region field test (May 2023), min/avg/max from n=100 | Estimated from 1 MB upload avg/max + overhead model |
| GCP | Coarse estimate bounded by observed 100-400ms range | Coarse estimate; no vendor percentile table available |

## Distribution Parameters

### GET (unconditional read)

Modeled as `Lognormal(mu=ln(median), sigma)`, floored at `min_latency_ms`.

| Provider | median (ms) | sigma | min_latency (ms) |
|----------|----------:|------:|------------------:|
| S3 Express | 22 | 0.22 | 10 |
| S3 Standard | 27 | 0.62 | 10 |
| Azure Premium | 35 | 0.08 | 20 |
| Azure Standard | 38 | 0.66 | 20 |
| GCP | 200 | 0.30 | 80 |

### PUT (unconditional write)

Modeled as `Lognormal(mu=ln(base + rate * size_MiB), sigma)`, floored at `min_latency_ms`.

| Provider | base (ms) | rate (ms/MiB) | sigma | min_latency (ms) |
|----------|----------:|--------------:|------:|------------------:|
| S3 Express | 10 | 10 | 0.25 | 10 |
| S3 Standard | 60 | 20 | 0.29 | 10 |
| Azure Premium | 41 | 15 | 0.10 | 20 |
| Azure Standard | 45 | 25 | 0.50 | 20 |
| GCP | 200 | 17 | 0.30 | 80 |

## GET Percentiles

| Provider | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|----------|---:|----:|----:|----:|----:|----:|----:|----:|
| S3 Express | 15 | 17 | 19 | 22 | 26 | 29 | 32 | 37 |
| S3 Standard | 10 | 12 | 18 | 27 | 41 | 60 | 75 | 114 |
| Azure Premium | 31 | 32 | 33 | 35 | 37 | 39 | 40 | 42 |
| Azure Standard | 20 | 20 | 24 | 38 | 59 | 89 | 113 | 176 |
| GCP | 122 | 136 | 163 | 200 | 245 | 294 | 328 | 402 |

## PUT Percentiles at 10 KiB (manifest list size)

| Provider | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|----------|---:|----:|----:|----:|----:|----:|----:|----:|
| S3 Express | 10 | 10 | 10 | 10 | 12 | 14 | 15 | 18 |
| S3 Standard | 37 | 42 | 50 | 60 | 73 | 87 | 97 | 118 |
| Azure Premium | 35 | 36 | 39 | 41 | 44 | 47 | 49 | 52 |
| Azure Standard | 20 | 24 | 32 | 45 | 63 | 86 | 103 | 145 |
| GCP | 122 | 136 | 164 | 200 | 245 | 294 | 328 | 402 |

## PUT Percentiles at 100 KiB (manifest file size)

| Provider | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|----------|---:|----:|----:|----:|----:|----:|----:|----:|
| S3 Express | 10 | 10 | 10 | 11 | 13 | 15 | 17 | 20 |
| S3 Standard | 38 | 43 | 51 | 62 | 75 | 90 | 100 | 122 |
| Azure Premium | 36 | 37 | 40 | 43 | 45 | 48 | 50 | 54 |
| Azure Standard | 21 | 25 | 34 | 47 | 67 | 90 | 108 | 152 |
| GCP | 123 | 137 | 165 | 202 | 247 | 296 | 330 | 405 |

## CAS Percentiles (conditional write, for reference)

| Provider | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|----------|---:|----:|----:|----:|----:|----:|----:|----:|
| S3 Express | 15 | 17 | 19 | 22 | 26 | 29 | 32 | 37 |
| S3 Standard | 49 | 51 | 56 | 61 | 67 | 73 | 77 | 85 |
| Azure Premium | 20 | 25 | 39 | 64 | 105 | 163 | 213 | 350 |
| Azure Standard | 24 | 33 | 54 | 93 | 162 | 266 | 358 | 626 |
| GCP | 80 | 80 | 92 | 170 | 314 | 546 | 760 | 1412 |

## Notes

- **S3 Express**: GET and PUT parameters are from YCSB benchmarks, not validated against the independent benchmarks used for S3 Standard. PUT base=10ms is suspiciously low relative to the GET median=22ms; this may need revisiting.
- **S3 Standard**: Strongest data. GET from Dec 2025 microbenchmark (4 KiB reads, in-region). PUT base derived from Mar 2025 benchmark at ~500 KiB: 69.8ms - 20 * 0.488 MiB = 60ms.
- **Azure Premium**: Very tight distributions (sigma=0.08-0.10) are consistent with Premium's low-variability positioning, but derived from min/avg/max of n=100 downloads — not directly measured percentiles.
- **Azure Standard**: High sigma on GET (0.66) reflects heavy tails seen in the field test (max=176ms on n=100). PUT estimates are model-derived and less reliable.
- **GCP**: All estimates are coarse. The 100-400ms range from one environment (JuiceFS benchmark note, Nov 2024) is the only public latency data found. No vendor percentile table exists.
- **Object sizes in simulation**: Manifest lists are 10 KiB, manifest files are 100 KiB, table metadata is 1 KiB.
- **Size sensitivity**: For objects under 100 KiB, per-operation overhead dominates; latency differences between 10 KiB and 100 KiB are small (typically <5ms).
