# Blog Update Notes — exp1–4 Post-Fix Results

Written 2026-04-13 after regenerating `plots/exp{1,2,3a,3b,4a*,4b*}/` from fresh
experiments/ directories. Supersedes the (deleted) `docs/REVISION_REPORT.md`.

> **STATUS: INVALID — RE-RUN REQUIRED.** The re-runs this document analyses
> were executed with `table_metadata_inlined = true` for exp1–4 (see §0).
> The blog posts describe the **non-inlined** protocol. Numbers below should
> not be used for the correction. Flag has been flipped to `false` in the
> configs; all of exp1–4 must be re-run before any blog edit is issued.

## 0. Silent config drift (discovered 2026-04-13)

All exp1–4 configs had `table_metadata_inlined = true` set on 2026-02-23 by the
"Unified experiment config" refactor (`be348a3`). The flag was **inert**
until 2026-03-26 (`f1ad9ef` — "Add partition-aware conflicts (exp5) and
inlined metadata (exp6)"), which wired it through `Catalog` and
`Transaction`. From that point forward, every re-run of exp1–4 quietly
eliminated TM I/O from the per-attempt cost — a mode the blog posts do not
describe.

Consequences:

- All `plots/exp{1,2,3a,3b,4a*,4b*,4c}/` output from before 2026-04-13 is
  against the inlined workload and should not be published.
- The "physics check" in §4.2 below mistakenly compared observed throughput
  against the 1/(3L) inlined bound. Under the 1/(5L) non-inlined bound that
  the blog implies, the observed 13.4 c/s exceeds the S3-median ceiling by
  roughly 3×.
- The flag has been flipped to `false` in exp1–4a/4b/4c (commit pending).
  Exp5 was already `false`; exp6 stays `true` as the explicit inlined
  experiment.

Other config audit results (2026-04-13):

| Dimension | Status across exp1–4 |
|---|---|
| `duration_ms = 3600000` | consistent |
| `num_tables = 1` default (swept 1/2/5/10/20/50 in exp4) | consistent |
| `num_groups = 1` | consistent |
| `retry = 10` | consistent |
| `runtime.mean = 180000` (180 s) | consistent |
| `inter_arrival.distribution = "exponential"` | consistent |
| `operation_types.fast_append` (1.0 / 0.9 / mix-swept) | per-experiment as intended |
| `real_conflict_probability = 0.0` | consistent |
| `manifest_list_mode = "rewrite"` | consistent |
| `[storage] provider = "s3"` | consistent (exp4c swept separately) |
| `max_parallel = 4` | **Removed 2026-04-13** from all configs, SPEC.md, and `tests/test_latency_separation.py`. The simulator issues I/O serially; the config flag was dead. See §2.7 for the required blog-post edit. |
| `backend = "service"` / `"storage"` | `"service"` for exp1–4b, `"storage"` for exp4c (expected) |

`max_parallel` is a second, quieter form of drift: the config claimed a
parallel-I/O limit but the simulator issues all I/O serially. This directly
affects the physics check, because the per-attempt cycle that feeds the
throughput bound is `read + read + write + write + CAS` **serialized**, not
parallelised. Resolution: `max_parallel` has been removed from every config,
from `SPEC.md`, and from `tests/test_latency_separation.py`
(`tests/test_throughput_bounds.py` remains green after the removal).

## 1. What changed in the simulator

Commits after the blog posts were written (newest first):

| SHA | Fix | Net effect |
|-----|-----|------------|
| `a6e98e0` | Split-yield `catalog.read()` (server-side evaluation at half-RTT) | Eliminates an information leak; lowers observed throughput slightly |
| `7cfec8b` | CAS latency scales with payload size for inlined metadata | Inlined-metadata commits now pay write-size latency for CAS |
| `ec383ff` | Failure-path TM read; free disjoint retry; CAS size cap | Cross-table retries are cheap; same-table/overlap retries pay the full per-attempt cost |
| `5c8aa30` | Added table-metadata read/write to per-attempt cost model | +2 ops per attempt (non-inlined) |
| `9d1d8e1` | Removed manifest-*file* write from per-attempt cost | −1 op per attempt |
| `fa51753` | Per-table VO convoy decomposition | Convoy cost is `(V-1)·M` per table, per attempt (not global `V·M`) |

Also added: `tests/test_throughput_bounds.py` asserts observed throughput
against the closed-form `1/(5L)` / `1/(4L)` / `1/(3L)` upper bounds for
non-inlined / ML-append / inlined modes, at constant storage latency. All six
cases pass in the current tree.

## 2. Required edits to `2026-03-09-catalog.md` (exp1–3)

### 2.1 `tl;dr` bullets and conclusions (HIGH IMPACT)

| Blog claim | New value | Status |
|------------|-----------|--------|
| "1–2 commits/sec is the ceiling" | 100 % success up to **4 c/s**; >95 % at **~8 c/s** (FA-only, 1 ms cat.) | **Revise wording** |
| "3–4 c/s" FA ceiling | ~8 c/s at ≥95 % success; peak 13.4 c/s at 33 % success | **Revise wording** |
| "adding 10 % VO → ~2 c/s ceiling" | FA=0.9 holds 100 % FA success up to **3.3 c/s**, 99.95 % at **~5 c/s** (VO tail still minutes) | **Revise wording** |

### 2.2 `exp1_fa_baseline` table (§ 1a — HIGH IMPACT)

Full replacement, 1 ms catalog, 5 seeds:

| Throughput (c/s) | Success (%) | P50 (s) | P95 (s) | P99 (s) | Mean retries |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 100.0 | 0.19 | 0.27 | 0.33 | 1.0 |
| 0.4 | 100.0 | 0.19 | 0.28 | 0.34 | 1.0 |
| 0.8 | 100.0 | 0.19 | 0.30 | 0.38 | 1.1 |
| 1.6 | 100.0 | 0.20 | 0.34 | 0.44 | 1.2 |
| 2.0 | 100.0 | 0.20 | 0.36 | 0.48 | 1.2 |
| 2.7 | 100.0 | 0.20 | 0.40 | 0.53 | 1.4 |
| **4.1** | **100.0** | **0.22** | **0.49** | **0.70** | **1.6** |
| 7.9 | 96.8 | 0.31 | 0.87 | 1.10 | 3.0 |
| 11.2 | 68.4 | 0.48 | 1.04 | 1.17 | 4.6 |
| 13.4 | 32.8 | 0.56 | 1.05 | 1.16 | 5.2 |

Alt-text around the annotated latency plot needs the same substitutions
(100 % success up to 4 c/s; P50 floor 190 ms, not 320 ms; P95/P99 converge
near ~1100 ms, not ~1750 ms).

### 2.3 Per-attempt cost description (MEDIUM)

Current text: "Each FA retry costs ~300 ms reading, merging, writing manifest
list + latest ML."

Replace with the current model: per-attempt cost is
`TM_read + ML_read + ML_write + TM_write + CAS` ≈ 188 ms S3 median (or
`ML_read + ML_write + CAS` ≈ 88 ms when `table_metadata_inlined=true` — which
**is the setting used in exp1/exp2**). The old 300 ms figure conflated
per-attempt cost with steady-state latency under contention.

### 2.4 § 1b heatmaps (exp2 — HIGH IMPACT)

Replace every `heatmap_{fa,vo}_success_rate.png` and `xheatmap_*_{mean,p95,p99}_latency.png`.
Representative new values (1 ms catalog, 1 table, inlined):

- FA=1.0 row: 20 ms IA → 34.5 % success (was 20 %); 50 ms → 70.5 % (was ~40 %).
- FA=0.9 row: 200 ms IA → 99.95 % FA success, 100 % at 300 ms. Old blog cell
  was ~20 % at 100 ms IA, which is now ~96 %.
- VO-only (FA=0.0): 500 ms IA → 100 % VO success but P99 = 93 s;
  200 ms → 99.89 % with P99 = 232 s. Convoy story survives but the boundary
  shifted down by roughly a factor of two in arrival rate.

### 2.5 § 2a/2b catalog-CAS heatmaps (exp3 — HIGH IMPACT)

Replace both heatmap sets. The qualitative story is unchanged, but the knees
move:

- Exp3a at 200 ms IA: 1 ms CAS → 100 %, 120 ms CAS → 87 % (was 99 % → 76 %).
  Catalog latency bites later than the blog implied.
- Exp3b at 200 ms IA, 90/10 mix, VO at 120 ms CAS → ~30 % (was 41 %);
  VO at 500 ms IA, 120 ms CAS → ~99 % (matches old).

### 2.7 Parallel-I/O claim (MEDIUM)

The post contains the footnote:

> "Up to 4 I/O operations can run in parallel."
> `[^parallelio]: The default in Iceberg is the number of logical processors.`

The simulator **does not** parallelise I/O within a transaction. The
`max_parallel` config flag was never consumed by `endive/` and has been
deleted (2026-04-13). The 5L / 4L / 3L throughput bounds in
`tests/test_throughput_bounds.py` assume strict serialization and pass —
this is the model the re-runs will produce. Remove the footnote and any
surrounding text that implies parallel I/O, or re-introduce the parallelism
in the simulator before citing it.

### 2.6 Sentences that survive unchanged

The three qualitative takeaways remain correct and can be kept verbatim:

- Storage I/O is the dominant bottleneck for single-table workloads.
- VO retries form IO convoys; P99 VO latency reaches minutes even at moderate arrival rates.
- Catalog CAS latency up to 120 ms has modest effect on FA-only single-table throughput.

## 3. Required edits to `2026-03-23-providercatalog.md` (exp4)

### 3.1 Scope limitation

**Exp4c (real-provider sweep) has not yet been re-run.** Everything in this
post that quotes per-provider absolute throughput (the Azure/Azure Premium/S3
Express tables) is still using the old I/O model. Flag: all Exp4c numbers
and plots in the post must wait for the Exp4c re-run. Do **not** publish a
partial correction that updates Exp4a/4b but leaves Exp4c stale.

### 3.2 Exp4a (FA-only, uniform — HIGH IMPACT)

Representative cells at CAS=10 ms:

| Cell | Old success | New success | Old mean (ms) | New mean (ms) |
|---|---:|---:|---:|---:|
| 1 tbl, 20 ms IA | 18.7 % | **30.6 %** | 1087 | **676** |
| 1 tbl, 100 ms IA | 71.0 % | **93.4 %** | 958 | **482** |
| 10 tbl, 20 ms IA | 94.1 % | **98.4 %** | 698 | **362** |
| 50 tbl, 20 ms IA | 99.5 % | **99.5 %** | 422 | **272** |
| 50 tbl, 100 ms IA | 100 % | 100 % | 373 | **236** |

Pattern: latencies drop ~30–50 %; success rates at low table counts improve
materially. The "more tables → less per-table contention" story is intact
but the single-table baseline is no longer as catastrophic. Alt-text that
quotes "1 table at 20ms is 18.7 %" or "at 5000 ms inter-arrival baseline
converges to ~350 ms" must be recomputed cell-by-cell from the new
`plots/exp4a_tables_fa/cas_*ms/heatmap_data.csv`.

### 3.3 Exp4a Zipf (HIGH IMPACT)

CAS=50 ms cells:

| Cell | Old success | New success | Old p99 (ms) | New p99 (ms) |
|---|---:|---:|---:|---:|
| 1 tbl, 20 ms | 14.6 % | **21.0 %** | 2906 | **2188** |
| 50 tbl, 20 ms | 43.4 % | **44.3 %** | 2293 | **1741** |
| 50 tbl, 100 ms | 92.5 % | **96.5 %** | 2389 | **1727** |

The "adding tables beyond 10 barely helps" claim survives; the absolute
numbers shift.

### 3.4 Exp4b (90/10 mix — HIGH IMPACT)

Same pattern as 4a for FA success rates. VO success/latency heatmap
comparisons require re-running with VO-only breakdown cached — see
`xheatmap_vo_*_latency.png` in `plots/exp4b_tables_mix/`.

## 4. Critical evaluation: do the new numbers make sense?

### 4.1 Throughput upper bound from first principles

For single-partition, single-table, non-inlined FA with constant-latency
storage `L`, the minimum interval between two successful CAS's is

```
catalog-read half-RTT + TM_read + ML_read + ML_write + TM_write + CAS half-RTT
= L/2 + L + L + L + L + L/2 = 5L     →  max rate = 1/(5L)
```

Inlined mode eliminates the TM hops: `1/(3L)`. ML-append mode eliminates the
ML-write: `1/(4L)`. `tests/test_throughput_bounds.py` locks these in at ±3 %
using `InstantStorageProvider(latency_ms=L)`.

### 4.2 Applying the bound to S3 lognormal (1 ms catalog)

Exp1/Exp2 use `table_metadata_inlined=true`, so the relevant bound is
`1/(3L)`. With S3 medians (GET 27 ms, PUT 60 ms, CAS 1 ms) and inlined-mode
steps `catalog-read_halfRTT + ML_read + ML_write + CAS_halfRTT`:

```
0.5 + 27 + 60 + 0.5 = 88 ms    →  ~11.4 c/s at median latencies
```

Observed: 13.4 c/s at 20 ms IA / 32.8 % success. **17 % above the median
bound.** Plausible given lognormal variance (σ_GET=0.62, σ_PUT=0.29 widen the
effective distribution of prep cycles). Fast-tail samples (p10 GET ≈ 12 ms,
p10 PUT ≈ 42 ms) give a lower-bound prep of ~55 ms ≈ 18 c/s, which bounds the
observation from above. So empirically:

| Metric | Value | Interpretation |
|---|---|---|
| Median bound | 11.4 c/s | If every op is at its median |
| Observed | 13.4 c/s | +17 % — consistent with lognormal skew |
| Fast-tail bound (p10) | ≈18 c/s | Hard ceiling given σ's; observation is below |

The numbers are **at the edge** of physical plausibility but not impossible.
A stricter check (below) is required before publishing.

### 4.3 Multi-table sanity

Exp4a at 50 tables, 20 ms IA (50 c/s offered total): 99.5 % success,
steady-state commit rate ≈ 50 c/s × 0.995 ≈ 49.8 c/s. The heatmap's
`throughput` column is in **commits per hour** (confirmed at
`saturation_analysis.py:2393`); 106 885 c/h = 29.7 c/s, which is the
middle-80 %-windowed slice of a full simulation — consistent with the offered
rate. No free lunches.

### 4.4 Direction of the change relative to the old run

The deleted `REVISION_REPORT.md` noted that new per-attempt costs are ~20 %
higher than the old buggy model (188 ms vs 156 ms on S3 median), yet new
results show **higher throughput and lower latency**. This is not a
contradiction:

1. The old model double-counted a manifest **file** write (100 KiB, ~43 ms)
   per attempt. Removing it dominates the net cost change.
2. The split-yield catalog (`a6e98e0`) correctly refuses to tell a client the
   "next version" until half-RTT has elapsed. This slightly **lowers**
   throughput vs. the old leaky model — not raises it. So the throughput
   bump comes from (1), not (3).
3. The `fa51753` VO convoy decomposition per table reduces charged I/O on
   multi-table VO retries — explains most of the Exp4b VO-latency
   compression.

The direction is consistent: fewer spurious ops per attempt → shorter per-txn
latency → higher steady-state commit rate.

### 4.5 Residual risks (things I could not verify without more work)

1. **Inlined-mode CAS size scaling** (`7cfec8b`). Exp1/Exp2 use inlined
   metadata, so CAS latency should scale with inlined metadata size. The blog
   post's implicit "CAS is free" assumption is no longer tight. If the
   inlined table metadata size in Exp1 is ~1 KiB, CAS ≈ `max(1 ms, PUT(1 KiB))`
   ≈ 60 ms — which would drop the observed 13.4 c/s to near the inlined
   bound. **Action needed: verify the inlined-metadata size used in Exp1.**
   If CAS ≠ 1 ms in the new runs, the bound analysis above changes.

2. **13.4 c/s exceeds the median bound by 17 %.** This is plausible under
   lognormal but not *proven* plausible. A targeted sanity run with S3
   latencies and inlined mode at low success rate, seeded 20× and compared
   against a Monte-Carlo estimate of `E[1/(ML_read + ML_write)]`, would
   confirm or falsify.

3. **Exp4c is still from the old run.** Any blog edit that cites per-provider
   numbers (S3 Express 14.6 c/s FA, etc.) is premature.

## 5. Suggested additional validation before we publish the correction

Priority-ordered. Each one is cheap relative to the correction risk.

1. **Extend `test_throughput_bounds.py` with lognormal cases.** Fit a
   Monte-Carlo `E[1/prep_cycle]` for S3, S3 Express, Azure medians; assert
   the simulator's single-table FA throughput is within ±5 % of that MC
   estimate at saturation. Rationale: the constant-L tests don't cover the
   actual distributions we run in exp1–4.

2. **Sanity run: single-table FA, 20 ms IA, 20 seeds, 2 hours.** Fit the
   empirical distribution of inter-commit intervals and confirm its mean
   matches `E[prep_cycle]`. Any systematic undershoot indicates a remaining
   leak.

3. **Inlined-metadata size audit for Exp1/Exp2.** Print `size_bytes` passed
   to `catalog.commit()` for an inlined run; confirm CAS latency is drawn
   from `max(base_cas, PUT(size))`, not from the fixed 1 ms. If CAS is
   genuinely 1 ms, document why in SPEC.md.

4. **Re-run Exp4c before publishing any correction.** Reasoning above. If
   the provider table in the blog cannot be regenerated yet, pull that
   section entirely from the corrected post rather than publish half-stale
   data.

5. **Numerical comparison against experiments-bak**, not just the old blog
   alt-text. The blog's front-matter alt-text is written from memory of
   plot images; `experiments-bak/` contains the underlying parquet. Running
   `regenerate_plots.py` against `experiments-bak/` (with a `--base-dir`
   override or equivalent) gives a clean before/after delta — more
   trustworthy than cell-by-cell transcription.

6. **Single-correction policy.** Batch all of the above into one correction
   post or one revision note. Issuing a partial correction, then a second
   correction for Exp4c a week later, would create the same confusion as
   the original overstatement.

## 6. Files touched

- Generated: `plots/exp{1_fa_baseline,2_mix_heatmap,3a_catalog_fa,3b_catalog_mix,4a_tables_fa,4a_zipf_tables_fa,4b_tables_mix,4b_zipf_tables_mix}/*`
- Not modified: either blog post, `docs/`, or any source in `endive/`.
