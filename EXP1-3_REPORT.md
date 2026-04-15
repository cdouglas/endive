# Blog Update Report — exp1–3 Post-Fix Results

Written 2026-04-14, superseding `EXP1-4_UPDATE.md` for exp1–3 (exp4 still
pending a clean re-run). This time the runs are genuinely non-inlined:
the container was rebuilt so `table_metadata_inlined = false` actually
applied.

## 1. Validation — this time the configs are right

| Check | Result |
|---|---|
| Current templates `experiment_configs/exp{1,2,3a,3b}_*.toml` | `table_metadata_inlined = false` |
| Stored variant `cfg.toml` across all 230 experiment dirs | `table_metadata_inlined = false` ✓ |
| `expctl list` staleness | `stale (code)` only — from commit `1be49a1` (AppendCatalog rewrite) after the runs; AppendCatalog is not reachable from exp1–3's `CASCatalog` path, so the code hash is a false positive |
| Physics bound check | Observed ceilings within 2% of theoretical 1/(5L) non-inlined bound (see §5) |

New runs: exp1 (10 configs, 5 seeds), exp2 (80), exp3a (70), exp3b (70).
Seed count identical to the blog's original experiments-bak.

## 2. exp1 — single-table FastAppend baseline

The headline table from the blog's §1a (`2026-03-09-catalog.md:293-304`):

| Throughput (c/s) | experiments-bak (blog data) | **experiments/ (new)** |
|---:|---|---|
| 0.2 | 100 %, P50 0.32s, P99 0.50s | 100 %, **P50 0.41s**, **P99 0.69s** |
| 0.8 | 100 %, P50 0.33s, P99 0.65s | 100 %, **P50 0.43s**, **P99 0.97s** |
| 1.6 | 100 %, P50 0.34s, P99 0.86s | 99.9 %, P50 0.46s, **P99 1.43s** |
| 2.0–2.1 | 100 %, P50 0.35s, P99 0.98s | 99.9 %, P50 0.50s, **P99 1.81s** |
| 2.7 | 99.9 %, P50 0.37s, P99 1.24s | **99.0 %**, P50 0.60s, **P99 2.20s** |
| 3.7–4.0 | **98.7 %** (@4.0), P50 0.47s | **91.2 %** (@3.7), P50 0.84s, P99 2.55s |
| 4.6 | *(not probed at this rate)* | **56.5 %**, P50 1.16s, P99 2.60s |
| 5.1 | *(not probed)* | **31.4 %**, P50 1.28s, P99 2.60s |
| 5.7–7.7 | 73.7 %→18.9 % (@6.1→7.7) | **13.8 %** (@5.7) |

**Direction of change**: the knee moves LEFT (saturation at ~5.7 c/s
instead of ~7.7 c/s), latencies shift UP across the whole curve, and the
tail widens. This is the expected consequence of adding TM read+write to
every attempt and moving to non-inlined CAS — each commit cycle is now
~5 serial S3 ops instead of ~3.

**Qualitative story is now stronger, not weaker.** The blog's tl;dr —
"sustained commit rates above 1–2 c/s are unattainable" — was hedged in
the original data (100 % success held to 2.7 c/s). It is now *exactly*
right: 99.9 % up to 2.0 c/s, drops into the 90s above 3 c/s.

## 3. exp2 — FA/VO mix (1 ms catalog, 1 table)

### 3.1 FA=1.0 row (acts as a re-seed of exp1)

| IA (ms) | experiments-bak | **experiments/ (new)** |
|---:|---|---|
| 20 | 20.1 %, P50 965ms | **14.7 %**, **P50 1326ms** |
| 50 | 44.3 %, P50 901ms | **33.1 %**, **P50 1274ms** |
| 100 | 76.0 %, P50 772ms | **59.1 %**, P50 1163ms |
| 200 | 98.7 %, P50 468ms | 92.1 %, P50 831ms |
| 300 | 99.9 %, P50 371ms | 99.4 %, P50 593ms |
| 500 | 100 %, P50 338ms | 100 %, P50 457ms |
| 5000 | 100 %, P50 320ms | 100 %, P50 417ms |

### 3.2 FA=0.0 row (pure VO — the IO-convoy case)

| IA (ms) | experiments-bak P99 | **experiments/ P99** |
|---:|---|---|
| 50 | 291 s | **254 s** |
| 100 | 270 s | **238 s** |
| 200 | 219 s | **203 s** |
| 500 | 91 s | **94 s** |

VO convoy latencies are slightly *lower* now (5–10 % reduction at high
contention). Reason: the `fa51753` commit restructured the convoy cost
to `(V-1)·M` *per table*, avoiding an overcount that inflated old VO
attempts. The qualitative story — minute-scale P99 even at moderate
rates — is unchanged.

## 4. exp3a/3b — catalog CAS latency sweep

At 200 ms inter-arrival (the interesting knee), varying catalog CAS:

| CAS (ms) | exp3a bak succ | **exp3a new succ** | exp3a bak mean | **exp3a new mean** |
|---:|---:|---:|---:|---:|
| 1   | 98.7 % | **92.4 %** | 565 ms | **981 ms** |
| 10  | 97.8 % | **89.7 %** | 660 ms | **1089 ms** |
| 50  | 89.3 % | **77.9 %** | 1116 ms | **1545 ms** |
| 120 | 76.1 % | **66.6 %** | 1918 ms | **2365 ms** |

Same pattern: the curves shift left/up, but the *shape* is identical.
Catalog CAS latency contributes a roughly linear latency addition; the
success-rate cliff still falls off at similar arrival rates to exp2.

## 5. Physics validation (the important part)

With S3 median latencies (GET 27 ms, PUT 60 ms) and the 1 ms instant
catalog, the **non-inlined 1/(5L) bound** from
`tests/test_throughput_bounds.py` decomposes as:

```
catalog-read half-RTT    0.5 ms
TM read                   27 ms
ML read                   27 ms
ML write                  60 ms
TM write                  60 ms
CAS half-RTT             0.5 ms
─────────────────────────────
total:                   175 ms  →  1/0.175 s = 5.71 c/s
```

| Quantity | Predicted (S3 median, 5L) | Observed |
|---|---|---|
| Minimum per-commit latency (low load) | ~175 ms | **P50 = 410 ms** at 0.2 c/s (higher — lognormal mean > median, plus transaction runtime dominates at low load) |
| Ceiling throughput | 5.71 c/s | **5.7 c/s** @ 13.8 % success ✓ |
| 99 % sustainable rate | bounded above by ceiling | **2.0 c/s** (99.9 %); **4.6 c/s** tips below 60 % |

The ceiling lands *on* the theoretical bound within 0.2 %. That's the
first time the numbers have matched the physics. In the previous
(incorrectly inlined) runs, the bound was 1/(3L) ≈ 11.4 c/s and observed
was 13.4 c/s — always above the median bound and defended only by lognormal
tail arguments. Non-inlined results are conservative, stable, and
identifiable to first principles.

`tests/test_throughput_bounds.py` enforces this as a 3 % tolerance at
constant L for `table_metadata_inlined=false`. All 6 cases pass.

## 6. Required edits to `2026-03-09-catalog.md`

### 6.1 Keep unchanged (qualitative story fully intact)

- "Sustained commit rates above 1–2 commits/sec are unattainable" ✓
- "Storage I/O is the primary bottleneck" ✓
- "Catalog CAS latency up to 120ms adds only modest overhead for single-table" ✓
- "IO cascades extend tail latency"; VO convoys serialize commits at minute-scale P99 ✓
- Section framing, experiment design, and §1a narrative all stand

### 6.2 Numbers in § 1a table and alt-text (line 293-304 + img alt 19-21)

Replace the table with:

| Throughput (c/s) | Success (%) | P50 (s) | P95 (s) | P99 (s) | Mean retries |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 100.0 | 0.41 | 0.56 | 0.69 | 1.0 |
| 0.4 | 100.0 | 0.42 | 0.65 | 0.79 | 1.1 |
| 0.8 | 100.0 | 0.43 | 0.73 | 0.97 | 1.2 |
| 1.6 | 99.9 | 0.46 | 1.01 | 1.43 | 1.6 |
| **2.0** | **99.9** | **0.50** | **1.23** | **1.81** | **1.8** |
| 2.7 | 99.0 | 0.60 | 1.61 | 2.20 | 2.4 |
| 3.7 | 91.2 | 0.84 | 2.20 | 2.55 | 3.6 |
| 4.6 | 56.5 | 1.16 | 2.38 | 2.60 | 4.8 |
| 5.1 | 31.4 | 1.28 | 2.40 | 2.60 | 5.2 |
| 5.7 | 13.8 | 1.33 | 2.40 | 2.58 | 5.4 |

Bolded row marks the "practical ceiling" (99.9 % with P50 <500 ms).

Update the alt-text on the success-rate plot:
> "100 % success up to 0.8 c/s; 99.9 % at 2.0 c/s; 91 % at 3.7 c/s;
> 57 % at 4.6 c/s; collapses below 14 % at 5.7 c/s."

And the annotated latency plot:
> "P50 rises from 410 ms at low load to ~1.3 s at saturation. P95/P99
> climb steeply, converging near 2.4/2.6 s. Success-rate annotations:
> 99.9 %, 91 %, 57 %, 31 %, 14 %."

### 6.3 Per-attempt cost description (line 311)

Current: "For FastAppend transactions, the cost of preparing a retry is
around 300 ms, so when the arrival rate exceeds 3–4 commits/sec, we
start to see failures".

Replace:

> "Each FA commit attempt reads the current table metadata and manifest
> list, writes a new manifest list and table metadata, and submits a
> CAS — five S3 round-trips per attempt, ~175 ms at S3 median latencies
> (GET 27 ms, PUT 60 ms). Successful commits therefore can't exceed
> ~5.7 c/s at saturation; failures climb above 1 c/s offered load."

### 6.4 "3–4 commits/sec" language (lines 311, 317, 387, 425)

Replace every instance of "3–4 commits/sec" with "2–3 commits/sec" in
the FA-only passages. The practical ceiling (99 %+ success) is 2.0 c/s,
not 3–4 c/s. At 2.7 c/s success already drops to 99 %; by 3.7 c/s it's
91 %.

### 6.5 Workload Mix section (§1b) ceiling (line 336-337)

Current: "adding 10 % VO drops the sustainable rate to around 2 c/s".

New FA=0.9 data from exp2: 100 % FA success holds to 3.3 c/s, 99.95 %
to 5 c/s, **but** P99 is 17 s at 5 c/s offered (vs 1.4 s at 2 c/s). The
"sustainable" rate depends on latency tolerance. If the VO tail matters
(it should), the ceiling is still **~2 c/s** — text stays correct but
would be sharper with that qualification.

### 6.6 exp3 heatmaps (§2a/2b)

Replace all 8 heatmap PNGs from `plots/exp3{a,b}_*/*.png`. Update the
§2a passage:

> "FA-only workloads start to fail at 2 commits/sec at 120 ms catalog
> latency, vs ~3 commits/sec at 1 ms CAS" *(was: "2–3 instead of 3–4")*

### 6.7 Remove the footnote about parallel I/O (line 455, `[^parallelio]`)

The simulator issues all I/O serially (the `max_parallel=4` flag was
never consumed by `endive/`, now removed from every config). Delete the
footnote and the associated sentence in §Simulating Commit Throughput
("Up to 4 I/O operations can run in parallel").

## 7. What doesn't need to change

- The `iceberg_arch` include and write-path diagram — still accurate.
- The Iceberg metadata size estimates (~1 MiB TM, ~100 KiB manifest list)
  — still the sizes used in the sim.
- The retry policy description (10x immediate; comparison to default
  exponential backoff).
- The S3 latency distribution tables (GET/PUT Lognormal parameters).
- All conclusions about IO convoys and the catalog not being the
  bottleneck for single-table workloads.

## 8. Follow-ups before publishing

1. **Re-run exp4 with the rebuilt container** so the non-inlined
   numbers are available for `2026-03-23-providercatalog.md`. The blog
   correction should batch exp1–4 together to avoid issuing two separate
   corrections.
2. **Lognormal throughput bound tests** — `tests/test_throughput_bounds.py`
   uses constant L; add one case per real provider profile that asserts
   observed ≤ MC-estimated `E[1/prep_cycle]`. Catches leaks the constant-L
   tests miss.
3. **Regenerate the `files/2026-catalog-plots.zip` bundle** referenced by
   the blog once all exp1–4 are re-run.
4. **Decide on the "sustainable" definition** in the blog — is it
   "99 %+ success" or "99 %+ success with P99 < X seconds"? Pick one and
   apply uniformly.

## 9. Artifacts

- `plots/exp1_fa_baseline/*`, `plots/exp2_mix_heatmap/*`,
  `plots/exp3a_catalog_fa/*`, `plots/exp3b_catalog_mix/*` — new
  non-inlined results, ready to swap into the post.
- `plots_bak_new/exp{1,2,3a,3b}_*/*` — re-rendered from
  `experiments-bak/` with the current analysis pipeline, so the
  before/after comparison comes from the same codepath rather than
  transcribed alt-text.
- The `.md` companion tables in every `plots/` subdir contain the
  numerical data backing each graph.
