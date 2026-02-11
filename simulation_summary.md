# Conditional Write Simulation Summary

Statistical summary of YCSB benchmark measurements (June 2025) across
AWS, Azure, and GCP.  Data are separated by operation type: **CAS**
(compare-and-swap) and **conditional append**.  All latencies were
measured at the client and include network round-trip time.

## Data Sources

| Cloud | VM Type | CAS Data | Append Data | Auth |
|-------|---------|----------|-------------|------|
| AWS   | m5.4xlarge | Yes (base + extended) | Yes (extended) | IAM (metal) |
| Azure | Standard_D16s_v3 | Yes (base + extended) | Yes (base + extended) | SAS (canonical) |
| GCP   | n2-standard-16 | Yes (base only) | No | Service account (metal) |

Each benchmark ran 5 trials per thread count (1-16 threads), with
concurrent-window trimming applied so that only the time interval where
all JVM instances overlapped is included.

---

## AWS

### CAS

**Aggregate** (12,098,603 operations, 5 runs/thread, threads 1-16)

- Successful: 2,172,913  |  Failed: 9,925,690  |  Overall failure rate: 82.0%
- Latency (ms): mean=29.2, median=22.9, p95=65.4, p99=78.1, p99.9=125

**Distribution fit (all threads combined):**

**Lognormal**: mu=10.1619, sigma=0.4509 (KS stat=0.2674, p=0.0000)
**Gamma**: shape=4.3779, scale=6660.21 (KS stat=0.2976, p=0.0000)

**Per-thread latency:**

| Threads | N (success) | Failure Rate | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | P99.9 (ms) |
|--------:|------------:|-------------:|----------:|------------:|---------:|---------:|-----------:|
|       1 |     124,414 |         0.0% |      24.0 |        14.6 |     72.0 |     88.9 |        121 |
|       2 |     128,319 |        41.3% |      25.3 |        15.4 |     69.5 |     82.9 |        106 |
|       3 |     135,369 |        57.0% |      26.7 |        21.4 |     66.2 |     78.5 |       95.1 |
|       4 |     136,635 |        66.3% |      27.3 |        21.7 |     65.3 |     76.1 |        103 |
|       5 |     138,333 |        72.1% |      27.7 |        21.9 |     65.5 |     76.6 |        104 |
|       6 |     138,527 |        76.1% |      28.3 |        22.3 |     63.9 |     74.2 |       99.6 |
|       7 |     137,174 |        79.0% |      29.1 |        23.0 |     61.9 |     71.3 |       95.5 |
|       8 |     140,548 |        81.3% |      28.9 |        22.8 |     61.4 |     70.5 |       99.2 |
|       9 |     137,684 |        83.1% |      29.8 |        23.4 |     62.8 |     73.2 |        115 |
|      10 |     138,667 |        84.4% |      30.2 |        23.5 |     61.3 |     69.9 |        120 |
|      11 |     138,008 |        85.8% |      30.5 |        23.4 |     65.1 |     76.6 |        120 |
|      12 |     136,917 |        86.9% |      30.9 |        23.4 |     68.5 |     82.7 |        132 |
|      13 |     138,431 |        87.6% |      30.9 |        23.5 |     63.3 |     74.6 |        129 |
|      14 |     136,469 |        88.5% |      31.5 |        23.7 |     65.1 |     77.0 |        134 |
|      15 |     134,335 |        89.1% |      32.1 |        23.8 |     67.2 |     80.5 |        138 |
|      16 |     133,083 |        89.7% |      32.6 |        23.9 |     68.6 |     81.9 |        137 |

**Failure rate by thread count:**

| Threads | Success | Fail | Total | Failure Rate |
|--------:|--------:|-----:|------:|-------------:|
|       1 | 124,414 |    0 | 124,414 |         0.0% |
|       2 | 128,319 | 90,139 | 218,458 |        41.3% |
|       3 | 135,369 | 179,123 | 314,492 |        57.0% |
|       4 | 136,635 | 268,524 | 405,159 |        66.3% |
|       5 | 138,333 | 358,349 | 496,682 |        72.1% |
|       6 | 138,527 | 441,416 | 579,943 |        76.1% |
|       7 | 137,174 | 517,244 | 654,418 |        79.0% |
|       8 | 140,548 | 609,975 | 750,523 |        81.3% |
|       9 | 137,684 | 676,884 | 814,568 |        83.1% |
|      10 | 138,667 | 752,467 | 891,134 |        84.4% |
|      11 | 138,008 | 830,528 | 968,536 |        85.8% |
|      12 | 136,917 | 904,598 | 1,041,515 |        86.9% |
|      13 | 138,431 | 981,518 | 1,119,949 |        87.6% |
|      14 | 136,469 | 1,047,154 | 1,183,623 |        88.5% |
|      15 | 134,335 | 1,103,112 | 1,237,447 |        89.1% |
|      16 | 133,083 | 1,164,659 | 1,297,742 |        89.7% |

**Throughput (successful ops/sec):**

| Threads | Mean Throughput (ops/s) | Std Dev |
|--------:|-----------------------:|--------:|
|       1 |                   28.8 |     0.2 |
|       2 |                   29.8 |     0.1 |
|       3 |                   31.4 |     0.1 |
|       4 |                   31.7 |     0.2 |
|       5 |                   32.2 |     0.1 |
|       6 |                   32.3 |     0.1 |
|       7 |                   32.0 |     0.2 |
|       8 |                   32.9 |     0.4 |
|       9 |                   32.3 |     0.1 |
|      10 |                   32.6 |     0.2 |
|      11 |                   32.5 |     0.2 |
|      12 |                   32.3 |     0.3 |
|      13 |                   32.7 |     0.2 |
|      14 |                   32.3 |     0.1 |
|      15 |                   31.9 |     0.1 |
|      16 |                   31.7 |     0.1 |

**Distribution fits by thread count:**

- **1 thread(s):** Lognormal(mu=9.854, sigma=0.589) | Gamma(shape=2.313, scale=10374.1)
- **4 thread(s):** Lognormal(mu=10.086, sigma=0.464) | Gamma(shape=4.045, scale=6748.2)
- **8 thread(s):** Lognormal(mu=10.182, sigma=0.393) | Gamma(shape=5.648, scale=5122.9)
- **16 thread(s):** Lognormal(mu=10.290, sigma=0.413) | Gamma(shape=5.062, scale=6442.1)

### Conditional Append

**Aggregate** (9,006,823 operations, 5 runs/thread, threads 1-16)

- Successful: 2,106,992  |  Failed: 6,899,831  |  Overall failure rate: 76.6%
- Latency (ms): mean=20.7, median=20.5, p95=27.8, p99=48.2, p99.9=88.5

**Distribution fit (all threads combined):**

**Lognormal**: mu=9.9033, sigma=0.2471 (KS stat=0.1537, p=0.0000)
**Gamma**: shape=14.7091, scale=1407.05 (KS stat=0.1641, p=0.0000)

**Per-thread latency:**

| Threads | N (success) | Failure Rate | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | P99.9 (ms) |
|--------:|------------:|-------------:|----------:|------------:|---------:|---------:|-----------:|
|       1 |     106,086 |         0.0% |      13.1 |        13.0 |     15.0 |     16.5 |       23.7 |
|       2 |     122,898 |        30.7% |      17.0 |        15.8 |     22.8 |     24.4 |       30.1 |
|       3 |     127,762 |        48.5% |      19.1 |        20.0 |     23.2 |     25.1 |       32.8 |
|       4 |     130,629 |        58.4% |      19.6 |        20.2 |     23.4 |     26.4 |       37.2 |
|       5 |     132,314 |        65.2% |      19.9 |        20.2 |     23.5 |     27.8 |       41.5 |
|       6 |     132,266 |        69.9% |      20.3 |        20.4 |     24.1 |     31.0 |       47.3 |
|       7 |     130,710 |        73.0% |      20.9 |        21.0 |     25.2 |     34.9 |       59.3 |
|       8 |     134,039 |        76.2% |      20.6 |        20.4 |     24.9 |     36.0 |       62.5 |
|       9 |     134,763 |        78.3% |      20.9 |        20.5 |     26.3 |     40.8 |       70.6 |
|      10 |     134,970 |        80.0% |      21.2 |        20.6 |     27.3 |     43.6 |       77.8 |
|      11 |     134,347 |        81.2% |      21.9 |        21.0 |     31.4 |     49.2 |       92.6 |
|      12 |     136,328 |        82.5% |      21.9 |        20.7 |     32.9 |     51.0 |       94.6 |
|      13 |     137,393 |        83.4% |      22.2 |        20.9 |     35.6 |     54.6 |        103 |
|      14 |     137,205 |        84.2% |      22.7 |        21.1 |     37.8 |     57.5 |        110 |
|      15 |     137,166 |        84.7% |      23.4 |        21.4 |     42.0 |     63.2 |        117 |
|      16 |     138,116 |        85.0% |      24.2 |        21.7 |     45.9 |     67.8 |        128 |

**Failure rate by thread count:**

| Threads | Success | Fail | Total | Failure Rate |
|--------:|--------:|-----:|------:|-------------:|
|       1 | 106,086 |    0 | 106,086 |         0.0% |
|       2 | 122,898 | 54,499 | 177,397 |        30.7% |
|       3 | 127,762 | 120,098 | 247,860 |        48.5% |
|       4 | 130,629 | 183,563 | 314,192 |        58.4% |
|       5 | 132,314 | 248,221 | 380,535 |        65.2% |
|       6 | 132,266 | 307,037 | 439,303 |        69.9% |
|       7 | 130,710 | 353,383 | 484,093 |        73.0% |
|       8 | 134,039 | 429,192 | 563,231 |        76.2% |
|       9 | 134,763 | 485,562 | 620,325 |        78.3% |
|      10 | 134,970 | 539,934 | 674,904 |        80.0% |
|      11 | 134,347 | 578,432 | 712,779 |        81.2% |
|      12 | 136,328 | 642,836 | 779,164 |        82.5% |
|      13 | 137,393 | 688,835 | 826,228 |        83.4% |
|      14 | 137,205 | 729,727 | 866,932 |        84.2% |
|      15 | 137,166 | 758,562 | 895,728 |        84.7% |
|      16 | 138,116 | 779,950 | 918,066 |        85.0% |

**Throughput (successful ops/sec):**

| Threads | Mean Throughput (ops/s) | Std Dev |
|--------:|-----------------------:|--------:|
|       1 |                   71.0 |     0.8 |
|       2 |                   82.3 |     0.7 |
|       3 |                   85.7 |     0.2 |
|       4 |                   87.7 |     0.7 |
|       5 |                   88.9 |     0.3 |
|       6 |                   88.9 |     0.9 |
|       7 |                   88.0 |     0.3 |
|       8 |                   90.3 |     1.0 |
|       9 |                   90.9 |     0.4 |
|      10 |                   91.1 |     0.9 |
|      11 |                   90.9 |     0.3 |
|      12 |                   92.3 |     1.0 |
|      13 |                   93.1 |     0.6 |
|      14 |                   93.1 |     1.1 |
|      15 |                   93.2 |     1.5 |
|      16 |                   94.1 |     0.5 |

**Distribution fits by thread count:**

- **1 thread(s):** Lognormal(mu=9.476, sigma=0.091) | Gamma(shape=117.359, scale=111.7)
- **4 thread(s):** Lognormal(mu=9.870, sigma=0.174) | Gamma(shape=34.240, scale=573.5)
- **8 thread(s):** Lognormal(mu=9.915, sigma=0.175) | Gamma(shape=28.947, scale=711.1)
- **16 thread(s):** Lognormal(mu=10.034, sigma=0.316) | Gamma(shape=8.360, scale=2898.5)

## Azure

### CAS

**Aggregate** (1,451,797 operations, 5 runs/thread, threads 1-16)

- Successful: 593,881  |  Failed: 857,916  |  Overall failure rate: 59.1%
- Latency (ms): mean=268, median=74.7, p95=586, p99=4294, p99.9=44343

**Distribution fit (all threads combined):**

**Lognormal**: mu=11.4418, sigma=0.7991 (KS stat=0.2459, p=0.0000)
**Gamma**: shape=0.5865, scale=456939.57 (KS stat=0.3940, p=0.0000)

**Per-thread latency:**

| Threads | N (success) | Failure Rate | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | P99.9 (ms) |
|--------:|------------:|-------------:|----------:|------------:|---------:|---------:|-----------:|
|       1 |      38,496 |         0.0% |      77.3 |        62.8 |      134 |      186 |        302 |
|       2 |      38,669 |        14.2% |       106 |        67.8 |      132 |      571 |       5107 |
|       3 |      38,438 |        25.9% |       132 |        69.3 |      140 |      613 |      16253 |
|       4 |      37,010 |        37.3% |       154 |        70.1 |      188 |     1246 |      16812 |
|       5 |      38,067 |        37.9% |       193 |        71.4 |      185 |     1607 |      18114 |
|       6 |      36,838 |        46.9% |       221 |        73.5 |      568 |     4077 |      18216 |
|       7 |      37,510 |        49.6% |       249 |        76.3 |      575 |     4226 |      44210 |
|       8 |      36,083 |        58.1% |       279 |        76.4 |      589 |     4313 |      44518 |
|       9 |      36,999 |        60.9% |       286 |        76.1 |      590 |     4345 |      44265 |
|      10 |      36,824 |        63.7% |       312 |        75.8 |      595 |     4582 |      45153 |
|      11 |      36,610 |        66.7% |       322 |        76.7 |      600 |     4616 |      44638 |
|      12 |      37,292 |        68.4% |       342 |        76.4 |      603 |     4775 |      44624 |
|      13 |      36,391 |        70.8% |       377 |        77.9 |      614 |     4865 |      44983 |
|      14 |      36,246 |        73.0% |       390 |        78.6 |      622 |     4844 |      44971 |
|      15 |      35,453 |        74.7% |       437 |        78.6 |      665 |     5627 |      45012 |
|      16 |      36,955 |        74.3% |       446 |        77.4 |      689 |     5639 |      45190 |

**Failure rate by thread count:**

| Threads | Success | Fail | Total | Failure Rate |
|--------:|--------:|-----:|------:|-------------:|
|       1 |  38,496 |    0 | 38,496 |         0.0% |
|       2 |  38,669 | 6,375 | 45,044 |        14.2% |
|       3 |  38,438 | 13,421 | 51,859 |        25.9% |
|       4 |  37,010 | 22,052 | 59,062 |        37.3% |
|       5 |  38,067 | 23,236 | 61,303 |        37.9% |
|       6 |  36,838 | 32,492 | 69,330 |        46.9% |
|       7 |  37,510 | 36,850 | 74,360 |        49.6% |
|       8 |  36,083 | 50,033 | 86,116 |        58.1% |
|       9 |  36,999 | 57,562 | 94,561 |        60.9% |
|      10 |  36,824 | 64,552 | 101,376 |        63.7% |
|      11 |  36,610 | 73,305 | 109,915 |        66.7% |
|      12 |  37,292 | 80,550 | 117,842 |        68.4% |
|      13 |  36,391 | 88,087 | 124,478 |        70.8% |
|      14 |  36,246 | 97,855 | 134,101 |        73.0% |
|      15 |  35,453 | 104,563 | 140,016 |        74.7% |
|      16 |  36,955 | 106,983 | 143,938 |        74.3% |

**Throughput (successful ops/sec):**

| Threads | Mean Throughput (ops/s) | Std Dev |
|--------:|-----------------------:|--------:|
|       1 |                   24.1 |     0.9 |
|       2 |                   24.1 |     0.6 |
|       3 |                   23.8 |     0.8 |
|       4 |                   23.9 |     0.2 |
|       5 |                   25.0 |     0.8 |
|       6 |                   24.1 |     0.4 |
|       7 |                   23.2 |     0.4 |
|       8 |                   23.6 |     0.7 |
|       9 |                   24.3 |     0.5 |
|      10 |                   24.4 |     0.8 |
|      11 |                   24.2 |     0.6 |
|      12 |                   23.2 |     0.6 |
|      13 |                   23.5 |     1.0 |
|      14 |                   23.4 |     0.8 |
|      15 |                   22.9 |     0.5 |
|      16 |                   22.2 |     0.7 |

**Distribution fits by thread count:**

- **1 thread(s):** Lognormal(mu=11.198, sigma=0.317) | Gamma(shape=8.824, scale=8759.7)
- **4 thread(s):** Lognormal(mu=11.334, sigma=0.594) | Gamma(shape=0.955, scale=160864.8)
- **8 thread(s):** Lognormal(mu=11.474, sigma=0.802) | Gamma(shape=0.583, scale=477669.9)
- **16 thread(s):** Lognormal(mu=11.616, sigma=1.035) | Gamma(shape=0.462, scale=966805.8)

### Conditional Append

**Aggregate** (722,353 operations, 5 runs/thread, threads 1-16)

- Successful: 597,150  |  Failed: 125,203  |  Overall failure rate: 17.3%
- Latency (ms): mean=81.3, median=76.9, p95=111, p99=158, p99.9=616

**Distribution fit (all threads combined):**

**Lognormal**: mu=11.2457, sigma=0.2770 (KS stat=0.0613, p=0.0000)
**Gamma**: shape=8.4412, scale=9633.37 (KS stat=0.1256, p=0.0000)

**Per-thread latency:**

| Threads | N (success) | Failure Rate | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | P99.9 (ms) |
|--------:|------------:|-------------:|----------:|------------:|---------:|---------:|-----------:|
|       1 |      37,425 |         0.0% |      79.5 |        77.0 |      113 |      160 |        277 |
|       2 |      37,719 |         2.6% |      79.3 |        77.0 |      111 |      150 |        372 |
|       3 |      37,445 |         5.4% |      79.8 |        76.9 |      112 |      163 |        410 |
|       4 |      37,642 |         7.6% |      80.0 |        77.2 |      111 |      153 |        561 |
|       5 |      37,660 |        10.2% |      79.8 |        76.9 |      110 |      145 |        594 |
|       6 |      37,363 |        12.1% |      80.5 |        77.1 |      112 |      158 |        598 |
|       7 |      37,759 |        14.5% |      79.7 |        76.6 |      109 |      146 |        596 |
|       8 |      37,108 |        16.6% |      81.4 |        77.5 |      113 |      167 |        615 |
|       9 |      37,271 |        18.4% |      82.3 |        77.3 |      112 |      159 |        626 |
|      10 |      36,896 |        20.1% |      82.5 |        77.5 |      113 |      165 |        619 |
|      11 |      37,793 |        22.1% |      80.4 |        75.7 |      109 |      149 |        633 |
|      12 |      37,085 |        23.3% |      82.5 |        76.6 |      112 |      153 |       1581 |
|      13 |      36,608 |        25.3% |      84.0 |        77.6 |      113 |      176 |       1577 |
|      14 |      37,702 |        26.5% |      81.5 |        76.1 |      106 |      147 |       1595 |
|      15 |      36,725 |        28.2% |      84.7 |        76.6 |      111 |      193 |       1599 |
|      16 |      36,949 |        29.7% |      83.5 |        76.7 |      108 |      166 |       1599 |

**Failure rate by thread count:**

| Threads | Success | Fail | Total | Failure Rate |
|--------:|--------:|-----:|------:|-------------:|
|       1 |  37,425 |    0 | 37,425 |         0.0% |
|       2 |  37,719 | 1,025 | 38,744 |         2.6% |
|       3 |  37,445 | 2,146 | 39,591 |         5.4% |
|       4 |  37,642 | 3,100 | 40,742 |         7.6% |
|       5 |  37,660 | 4,299 | 41,959 |        10.2% |
|       6 |  37,363 | 5,140 | 42,503 |        12.1% |
|       7 |  37,759 | 6,412 | 44,171 |        14.5% |
|       8 |  37,108 | 7,411 | 44,519 |        16.6% |
|       9 |  37,271 | 8,389 | 45,660 |        18.4% |
|      10 |  36,896 | 9,301 | 46,197 |        20.1% |
|      11 |  37,793 | 10,699 | 48,492 |        22.1% |
|      12 |  37,085 | 11,255 | 48,340 |        23.3% |
|      13 |  36,608 | 12,389 | 48,997 |        25.3% |
|      14 |  37,702 | 13,622 | 51,324 |        26.5% |
|      15 |  36,725 | 14,391 | 51,116 |        28.2% |
|      16 |  36,949 | 15,624 | 52,573 |        29.7% |

**Throughput (successful ops/sec):**

| Threads | Mean Throughput (ops/s) | Std Dev |
|--------:|-----------------------:|--------:|
|       1 |                   23.4 |     0.3 |
|       2 |                   23.5 |     0.3 |
|       3 |                   22.8 |     0.8 |
|       4 |                   22.5 |     0.2 |
|       5 |                   22.2 |     0.1 |
|       6 |                   22.1 |     0.3 |
|       7 |                   22.4 |     0.2 |
|       8 |                   22.1 |     0.2 |
|       9 |                   22.1 |     0.2 |
|      10 |                   21.7 |     0.4 |
|      11 |                   22.0 |     0.3 |
|      12 |                   21.4 |     0.1 |
|      13 |                   21.2 |     0.4 |
|      14 |                   21.9 |     0.2 |
|      15 |                   21.4 |     0.4 |
|      16 |                   21.3 |     0.3 |

**Distribution fits by thread count:**

- **1 thread(s):** Lognormal(mu=11.248, sigma=0.253) | Gamma(shape=14.153, scale=5617.9)
- **4 thread(s):** Lognormal(mu=11.244, sigma=0.260) | Gamma(shape=11.205, scale=7139.2)
- **8 thread(s):** Lognormal(mu=11.248, sigma=0.282) | Gamma(shape=8.547, scale=9526.2)
- **16 thread(s):** Lognormal(mu=11.251, sigma=0.293) | Gamma(shape=6.226, scale=13415.7)

## GCP

### CAS

**Aggregate** (48,792 operations, 5 runs/thread, threads 1-16)

- Successful: 23,635  |  Failed: 25,157  |  Overall failure rate: 51.6%
- Latency (ms): mean=530, median=170, p95=2437, p99=6546, p99.9=10717

**Distribution fit (all threads combined):**

**Lognormal**: mu=12.4443, sigma=0.9113 (KS stat=0.3593, p=0.0000)
**Gamma**: shape=0.8057, scale=657585.25 (KS stat=0.4004, p=0.0000)

**Per-thread latency:**

| Threads | N (success) | Failure Rate | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | P99.9 (ms) |
|--------:|------------:|-------------:|----------:|------------:|---------:|---------:|-----------:|
|       1 |       2,135 |         0.1% |       701 |         173 |     3421 |     9350 |      20478 |
|       2 |       1,998 |        18.3% |       577 |         173 |     2613 |     6184 |      11945 |
|       3 |       1,867 |        28.8% |       546 |         172 |     2476 |     6763 |      11299 |
|       4 |       1,728 |        36.5% |       508 |         172 |     2223 |     6781 |       9264 |
|       5 |       1,699 |        40.6% |       501 |         168 |     2246 |     5873 |       9174 |
|       6 |       1,613 |        45.3% |       507 |         168 |     2386 |     5795 |       9195 |
|       7 |       1,561 |        48.4% |       482 |         170 |     2169 |     5757 |       9091 |
|       8 |       1,460 |        52.5% |       496 |         169 |     2257 |     5778 |       9071 |
|       9 |       1,325 |        57.6% |       488 |         169 |     1945 |     6169 |       8545 |
|      10 |       1,269 |        59.4% |       511 |         169 |     2363 |     5885 |       8474 |
|      11 |       1,278 |        60.4% |       475 |         169 |     2257 |     5970 |       8433 |
|      12 |       1,224 |        63.5% |       532 |         167 |     2542 |     6937 |       9038 |
|      13 |       1,181 |        65.4% |       476 |         169 |     2019 |     5400 |       8236 |
|      14 |       1,155 |        67.6% |       515 |         165 |     2498 |     5858 |       8941 |
|      15 |       1,070 |        69.8% |       554 |         166 |     2617 |     7058 |       9272 |
|      16 |       1,072 |        70.2% |       493 |         167 |     2263 |     6727 |       8188 |

**Failure rate by thread count:**

| Threads | Success | Fail | Total | Failure Rate |
|--------:|--------:|-----:|------:|-------------:|
|       1 |   2,135 |    2 | 2,137 |         0.1% |
|       2 |   1,998 |  449 | 2,447 |        18.3% |
|       3 |   1,867 |  754 | 2,621 |        28.8% |
|       4 |   1,728 |  993 | 2,721 |        36.5% |
|       5 |   1,699 | 1,162 | 2,861 |        40.6% |
|       6 |   1,613 | 1,337 | 2,950 |        45.3% |
|       7 |   1,561 | 1,462 | 3,023 |        48.4% |
|       8 |   1,460 | 1,616 | 3,076 |        52.5% |
|       9 |   1,325 | 1,803 | 3,128 |        57.6% |
|      10 |   1,269 | 1,857 | 3,126 |        59.4% |
|      11 |   1,278 | 1,947 | 3,225 |        60.4% |
|      12 |   1,224 | 2,132 | 3,356 |        63.5% |
|      13 |   1,181 | 2,236 | 3,417 |        65.4% |
|      14 |   1,155 | 2,405 | 3,560 |        67.6% |
|      15 |   1,070 | 2,472 | 3,542 |        69.8% |
|      16 |   1,072 | 2,530 | 3,602 |        70.2% |

**Throughput (successful ops/sec):**

| Threads | Mean Throughput (ops/s) | Std Dev |
|--------:|-----------------------:|--------:|
|       1 |                    1.4 |     0.1 |
|       2 |                    1.3 |     0.0 |
|       3 |                    1.3 |     0.0 |
|       4 |                    1.2 |     0.0 |
|       5 |                    1.1 |     0.0 |
|       6 |                    1.1 |     0.0 |
|       7 |                    1.1 |     0.0 |
|       8 |                    1.0 |     0.0 |
|       9 |                    0.9 |     0.1 |
|      10 |                    0.9 |     0.0 |
|      11 |                    0.9 |     0.1 |
|      12 |                    0.9 |     0.0 |
|      13 |                    0.9 |     0.0 |
|      14 |                    0.8 |     0.0 |
|      15 |                    0.8 |     0.0 |
|      16 |                    0.8 |     0.0 |

**Distribution fits by thread count:**

- **1 thread(s):** Lognormal(mu=12.520, sigma=1.015) | Gamma(shape=0.650, scale=1078442.3)
- **4 thread(s):** Lognormal(mu=12.441, sigma=0.884) | Gamma(shape=0.844, scale=602230.9)
- **8 thread(s):** Lognormal(mu=12.436, sigma=0.888) | Gamma(shape=0.865, scale=573566.4)
- **16 thread(s):** Lognormal(mu=12.394, sigma=0.885) | Gamma(shape=0.827, scale=596643.9)

### Conditional Append

_No data available._

---

## Azure Authentication Method Comparison

Azure was benchmarked with three authentication methods. SAS is used
as the canonical dataset above; this section compares all three.

### CAS

| Auth Method | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Failure Rate |
|-------------|----------:|------------:|---------:|---------:|-------------:|
| metal       |       337 |         112 |      938 |     2563 |        55.4% |
| sas         |       268 |        74.7 |      586 |     4294 |        59.1% |
| user        |       267 |        74.2 |      584 |     4326 |        57.3% |

### Conditional Append

| Auth Method | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Failure Rate |
|-------------|----------:|------------:|---------:|---------:|-------------:|
| metal       |       275 |        96.9 |      911 |     2523 |        31.4% |
| sas         |      81.3 |        76.9 |      111 |      158 |        17.3% |
| user        |      82.5 |        78.1 |      114 |      162 |        17.6% |

---

## Observations for Simulation

### 1. Latency distributions are approximately lognormal

Across all clouds and operation types, latency data fit a lognormal distribution.  The lognormal is the recommended distribution family for simulation.  The K-S p-values below indicate fit quality (higher is better; low values are expected for large sample sizes even when the shape is correct).

- **AWS CAS**: Lognormal(mu=10.162, sigma=0.451), KS p-value=0.0000
- **AWS append**: Lognormal(mu=9.903, sigma=0.247), KS p-value=0.0000
- **Azure CAS**: Lognormal(mu=11.442, sigma=0.799), KS p-value=0.0000
- **Azure append**: Lognormal(mu=11.246, sigma=0.277), KS p-value=0.0000
- **GCP CAS**: Lognormal(mu=12.444, sigma=0.911), KS p-value=0.0000


### 2. Failure rate scales with contention (thread count)

Failure rates increase with thread count because more concurrent writers
contend on the same object.  Under CAS, only one writer can succeed per
generation; under conditional append, the append condition (ETag/version)
similarly serializes writers.

- **AWS CAS**: 0.0% at 1 thread -> 89.7% at 16 threads
- **AWS append**: 0.0% at 1 thread -> 85.0% at 16 threads
- **Azure CAS**: 0.0% at 1 thread -> 74.3% at 16 threads
- **Azure append**: 0.0% at 1 thread -> 29.7% at 16 threads
- **GCP CAS**: 0.1% at 1 thread -> 70.2% at 16 threads

The theoretical failure rate under pure CAS with N contending writers is
`1 - 1/N`.  Comparing measured vs theoretical at selected thread counts:

| Cloud | Op | Threads | Measured | Theoretical (1-1/N) | Delta |
|-------|-----|--------:|---------:|--------------------:|------:|
| AWS | CAS | 2 | 41.3% | 50.0% | -8.7pp |
| AWS | CAS | 4 | 66.3% | 75.0% | -8.7pp |
| AWS | CAS | 8 | 81.3% | 87.5% | -6.2pp |
| AWS | CAS | 16 | 89.7% | 93.8% | -4.0pp |
| AWS | append | 2 | 30.7% | 50.0% | -19.3pp |
| AWS | append | 4 | 58.4% | 75.0% | -16.6pp |
| AWS | append | 8 | 76.2% | 87.5% | -11.3pp |
| AWS | append | 16 | 85.0% | 93.8% | -8.8pp |
| Azure | CAS | 2 | 14.2% | 50.0% | -35.8pp |
| Azure | CAS | 4 | 37.3% | 75.0% | -37.7pp |
| Azure | CAS | 8 | 58.1% | 87.5% | -29.4pp |
| Azure | CAS | 16 | 74.3% | 93.8% | -19.4pp |
| Azure | append | 2 | 2.6% | 50.0% | -47.4pp |
| Azure | append | 4 | 7.6% | 75.0% | -67.4pp |
| Azure | append | 8 | 16.6% | 87.5% | -70.9pp |
| Azure | append | 16 | 29.7% | 93.8% | -64.0pp |
| GCP | CAS | 2 | 18.3% | 50.0% | -31.7pp |
| GCP | CAS | 4 | 36.5% | 75.0% | -38.5pp |
| GCP | CAS | 8 | 52.5% | 87.5% | -35.0pp |
| GCP | CAS | 16 | 70.2% | 93.8% | -23.5pp |

The AWS failure rates closely track `1-1/N`, indicating near-ideal CAS
contention behavior.  Azure CAS undershoots the model (lower failure rates),
suggesting its storage layer may batch or serialize requests.  Azure append
has much lower failure rates than `1-1/N`, likely because the append
condition is less restrictive than full CAS.


### 3. CAS and conditional append have different failure profiles

CAS operations and conditional appends exhibit different failure rates even under comparable contention levels.  This may reflect differences in the storage backend's conflict resolution mechanism.  A simulator should model these separately.

- **AWS**: CAS failure rate 82.0% vs append failure rate 76.6%
- **Azure**: CAS failure rate 59.1% vs append failure rate 17.3%


### 4. Successful-operation latency increases with contention

Mean latency for successful operations rises with concurrency.  A simulator should scale base latency by a contention factor.

- **AWS CAS**: 24.0 ms at 1 thread -> 32.6 ms at 16 threads (1.4x increase)
- **AWS append**: 13.1 ms at 1 thread -> 24.2 ms at 16 threads (1.8x increase)
- **Azure CAS**: 77.3 ms at 1 thread -> 446 ms at 16 threads (5.8x increase)
- **Azure append**: 79.5 ms at 1 thread -> 83.5 ms at 16 threads (1.1x increase)
- **GCP CAS**: 701 ms at 1 thread -> 493 ms at 16 threads (0.7x increase)


### 5. Tail latency amplification

The ratio of tail latency (p99, p99.9) to median indicates how heavy-tailed the distribution is.  Higher ratios mean the simulator must account for occasional very slow operations.

- **AWS CAS**: p99/p50 = 3.4x, p99.9/p50 = 5.4x
- **AWS append**: p99/p50 = 2.3x, p99.9/p50 = 4.3x
- **Azure CAS**: p99/p50 = 57.5x, p99.9/p50 = 593.9x
- **Azure append**: p99/p50 = 2.1x, p99.9/p50 = 8.0x
- **GCP CAS**: p99/p50 = 38.6x, p99.9/p50 = 63.2x


### 6. Cross-cloud latency comparison

Median latency varies across providers.  A simulator that targets a specific cloud should use that cloud's fitted distribution; a cloud-agnostic simulator can use the range to bound expected behavior.

CAS median latency ranking: AWS (22.9 ms) < Azure (74.7 ms) < GCP (170 ms)

Append median latency ranking: AWS (20.5 ms) < Azure (76.9 ms)


### 7. Failed operations have distinct latency characteristics

Failed (conflicting) operations may complete faster or slower than successful ones depending on the storage backend.  A simulator should draw from a separate distribution for failed attempts.

- **AWS CAS**: success mean=29.2 ms, fail mean=34.1 ms (1.17x)
- **AWS append**: success mean=20.7 ms, fail mean=22.6 ms (1.09x)
- **Azure CAS**: success mean=268 ms, fail mean=274 ms (1.02x)
- **Azure append**: success mean=81.3 ms, fail mean=2787 ms (34.28x)
- **GCP CAS**: success mean=530 ms, fail mean=7111 ms (13.42x)


### 8. Data quality notes and caveats

- **GCP** has the smallest dataset (23,635 successful CAS operations total; 2,135 at 1 thread, 1,072 at 16 threads).  GCP's extremely heavy tail (sigma > 0.9 in the lognormal fit) means the sample mean is unstable — the apparent *decrease* in mean latency from 1 thread (701 ms) to 16 threads (493 ms) is an artifact of outlier sensitivity, not a real speedup.  The median is stable at ~170 ms across all thread counts.  Use the median or the lognormal parameters, not the mean, for GCP simulation.

- **Azure conditional append** failures are 34x slower than successes (mean 2787 ms vs 81.3 ms).  This suggests failed appends may involve server-side retry or conflict-resolution delays.  AWS failures, by contrast, are only slightly slower than successes.



### 9. Recommended simulation approach

Based on the data:

1. **Model latency as lognormal** with per-cloud, per-operation-type parameters.  Use the per-thread fits when simulating specific contention levels, or the aggregate fit for a general model.

2. **Model failure probability** as a function of thread count.  The failure rate is roughly `1 - 1/N` for N concurrent writers under CAS, but the actual rates show provider-specific deviations from this ideal.

3. **Use separate latency distributions for success and failure** paths, as their characteristics differ.

4. **Scale latency with contention**: multiply the base (1-thread) latency by a contention factor derived from the per-thread data.

5. **Account for tail latency**: the p99/p50 ratios indicate that a small fraction of operations will take significantly longer than the median.  The lognormal distribution naturally captures this.

6. **For Azure, SAS and user-delegation auth have similar performance**; metal (bearer token) auth adds overhead that increases latency.  Use SAS parameters as representative.


