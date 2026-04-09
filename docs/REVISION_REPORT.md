# Experiment Result Revision Report

**Date:** 2026-04-09

## Summary of I/O Cost Model Changes

Three bugs were fixed in the simulator's I/O cost model since the blog posts were written:

1. **Manifest file write removed** (9d1d8e1, 2026-04-08): Every commit attempt was incorrectly charging a manifest file write (~43ms S3 latency per partition). Manifest file writes are part of the transaction runtime, not the commit protocol. **Net: -1 op per attempt.**

2. **Table metadata I/O added** (5c8aa30, 2026-04-08): Every commit attempt must read and write the table metadata file. This was missing from the model entirely. **Net: +2 ops per attempt (non-inlined).**

3. **Failure-path table metadata read** (ec383ff, 2026-04-08): After a CAS failure, the non-inlined path needs a separate table metadata read to get partition versions for overlap detection. **Net: +1 op per failure.**

4. **CAS size scaling** (7cfec8b, 2026-04-09): CAS latency was using a fixed distribution that ignored payload size. Now uses `max(base_cas, write_latency(size))`. **Net: larger payloads (exp6 inlined) have higher CAS latency.**

5. **MergeAppend removed** (d88290f, 2026-04-08): MergeAppendTransaction was unused in all experiments and removed.

### Net effect on per-attempt cost

| Model | Operations | Est. S3 median |
|-------|-----------|---------------|
| Old (buggy) | ML_read(10K) + MF_write(100K) + ML_write(10K) | ~156ms |
| New (correct) | TM_read(10K) + ML_read(10K) + ML_write(10K) + TM_write(10K) | ~188ms |

The new model is ~20% more expensive per attempt. This shifts all saturation curves left (lower throughput) and up (higher latency).

## Exp1 FA Baseline Comparison

| Metric | Old | New | Change |
|--------|-----|-----|--------|
| Max throughput (20ms IA) | 7.7 c/s | 13.5 c/s | +75% |
| Success rate at 100ms IA | 73.7% | 96.7% | +23pp |
| P50 latency at 5s IA | 321ms | 190ms | -41% |
| P95 latency at 5s IA | ~340ms | 267ms | -21% |
| 100% success threshold | 500ms IA | 200ms IA | 2.5x better |

**Note:** The new results show HIGHER throughput and LOWER latency despite the new model being more expensive per-attempt. This is because the new results were generated with the post-rewrite codebase which includes architectural changes beyond just the I/O cost model (corrected CAS half-RTT timing, partition version tracking, etc.).

## Exp4a Multi-Table Comparison

| Config | Old throughput | New throughput | Old success | New success | Old P50 | New P50 |
|--------|---------------|---------------|-------------|-------------|---------|---------|
| 1 table, 100ms IA | 4.6 c/s | 7.7 c/s | 55.8% | 93.7% | 1296ms | 387ms |
| 10 tables, 100ms IA | 8.1 c/s | 8.2 c/s | 99.2% | 100.0% | 647ms | 194ms |

Multi-table with 10+ tables: throughput is similar (catalog CAS is the bottleneck, not per-attempt I/O), but latency is dramatically lower in new results. Single-table shows the same pattern as exp1 — large improvement from the architectural rewrite.

## Claims Requiring Revision in 2026-03-09-catalog.md

### 1. Absolute throughput numbers (HIGH IMPACT)

The blog states:
- "100% success up to 2.7 c/s" -- now 100% up to ~4.1 c/s (200ms IA)
- "2.1 c/s sustained with 100% success (P50: 0.35s)" -- now ~2.7 c/s at P50: 0.20s
- "4.0 c/s: 98.7% success" -- now ~7.9 c/s at 96.7% (or ~4.1 at 100%)
- "Sustained commit rates above 1-2 commits/sec are unattainable" -- now sustainable at 2.7+ c/s

**Action:** All throughput/latency numbers in the exp1 results section need updating.

### 2. Per-attempt cost description (MEDIUM IMPACT)

The blog states: "Each FA retry costs ~300ms (reading, merging, writing manifest list + latest manifest list)"

The correct model is: TM_read + ML_read(N) + ML_write(N) + TM_write per attempt, ~188ms on S3 for 1 partition. The ~300ms figure was from the old 3-op model under contention (multiple retries inflating the average).

**Action:** Revise the I/O cost breakdown description.

### 3. VO convoy cost model (MEDIUM IMPACT)

The blog states VO reads "ALL manifest lists committed since the read snapshot." This is correct in principle but the convoy formula changed: now `(V-1) * M` historical ML reads where V is per-table version delta and M is overlapping partitions, computed per-table for multi-table transactions.

**Action:** Clarify that the convoy reads N-1 (not N) historical MLs, and the count is per-table.

### 4. Heatmap plots (HIGH IMPACT)

All exp1/exp2 heatmaps need regeneration. The saturation boundaries, color scales, and absolute values will shift.

**Action:** Replace all heatmap PNGs with new results.

### 5. Manifest file size assumption (LOW IMPACT)

The blog states "manifest file sizes: 10 KiB each (fixed in simulations)." This is still correct for ML operations but the old manifest FILE write was at 100 KiB. Since MF writes are removed from the model, this claim is now irrelevant.

**Action:** Remove mention of manifest file sizes from the I/O cost model description.

## Claims Requiring Revision in 2026-03-23-providercatalog.md

### 1. Single-table provider throughput table (HIGH IMPACT)

The blog provides a detailed table of per-provider throughput at >95% VO success:
- S3 Express: 14.6 c/s FA, 7.5 c/s mixed
- S3 Standard: 2.4 c/s FA, 1.8 c/s mixed
- Azure Premium: 2.5 c/s FA, 1.9 c/s mixed
- etc.

These numbers all derive from the old I/O model. Every entry needs recalculation.

**Action:** Regenerate the provider comparison table with new results.

### 2. Multi-table scaling claims (HIGH IMPACT)

The blog states: "S3 Express sustains 14.6 c/s on a single table -- more per-table throughput than spreading across 50 tables on S3 Standard"

The relative ordering may hold but the absolute numbers will change.

**Action:** Regenerate multi-table comparison with new results.

### 3. VO tail latency improvements with table count (MEDIUM IMPACT)

"P99 latencies drop dramatically: S3 from 69.5s to 11.5s with 20 tables"

These specific numbers depend on the convoy cost model which changed (per-table decomposition).

**Action:** Regenerate VO P99 latency data.

### 4. Provider latency distribution parameters (NO CHANGE)

The YCSB benchmark-derived parameters (GET/PUT medians, sigmas, min latencies) are correct. These are empirical measurements, not model outputs.

**Action:** None.

### 5. Exp4c provider comparison methodology (NO CHANGE)

The experimental methodology (sweeping provider x tables x FA ratio) is correct. Only the numerical results change.

**Action:** Regenerate results, keep methodology description.

### 6. CAS latency characterization (LOW IMPACT)

The blog treats CAS as a fixed-latency operation. With the size-scaling fix, CAS latency now depends on payload size. For small catalogs (exp1-4 at ~100 bytes), the change is negligible. For exp6 (500 KiB inlined), CAS adds ~10ms.

**Action:** Note that CAS latency is now size-scaled in the model description.

## Key Qualitative Claims That Remain Valid

These conclusions are directionally correct and likely survive re-running:

1. **Storage I/O is the primary bottleneck** -- still true, now even more so with 4 ops per attempt
2. **VO forms I/O convoys** -- still true, convoy model is more precise now
3. **Multi-table partitioning shifts contention to catalog CAS** -- still true
4. **Zipf skew collapses effective table count** -- still true (structural, not latency-dependent)
5. **Provider choice is a larger lever than table count** -- likely still true
6. **GCS is not viable for high commit rates** -- still true (GCS latency is empirical)
7. **S3 Express provides ~5x improvement over S3 Standard** -- ratio may change slightly

## Plots to Regenerate

All plots from exp1 through exp4 need regeneration with the new results. The old plots are in `plots-bak/` for comparison.

New experiments (exp5/exp6) produce entirely new plots not in either blog post.
