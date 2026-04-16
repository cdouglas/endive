# Blog Update Report — exp4 Post-Fix Results

Written 2026-04-16, companion to `EXP1-3_REPORT.md`. Covers exp4a (FA-only
multi-table), exp4a_zipf, exp4b (90/10 mix multi-table), exp4b_zipf. **Exp4c
(real-provider sweep) is still pending a re-run** — see §9.

## 1. Validation

| Check | Result |
|---|---|
| Templates `experiment_configs/exp4{a,b}_*.toml` | `table_metadata_inlined = false` ✓ |
| Stored `cfg.toml` across all 960 exp4a/b dirs | `table_metadata_inlined = false` ✓ |
| Sweep coverage | 4×240 configs, 4×1200 seeds — all present, none crashed |
| `expctl list` | `stale (code)` — from `1be49a1` (AppendCatalog rewrite), not reachable from CAS catalog path used by exp4 |

This report contrasts `experiments/` (2026-04-15/16 runs, post-fix) against
`experiments-bak/` (pre-publication runs the 2026-03-23 blog was built on).
Both sets were rendered through the current `saturation_analysis.py`
pipeline so the comparison is apples-to-apples.

## 2. Errata — what changed since publication (summary)

Full detail in `EXP1-3_REPORT.md §2`. Same errata apply to exp4a/4b,
with one additional multi-table-specific fix worth calling out:

- **`ec383ff`** "free disjoint retry": for multi-table transactions, a
  CAS failure caused by a commit on a *different* table no longer
  charges manifest I/O. Cross-table contention at the catalog now costs
  a catalog-read + re-CAS only. This is the biggest single source of
  difference in exp4a/4b vs. the old numbers — it **lowers** latency
  and success-rate loss at the right edge of the heatmaps (high table
  count, moderate load), and has no effect on single-table cells.
- **`fa51753`** "per-table VO convoy decomposition": convoy cost now
  `Σ_table (V_table − 1) · M_table` instead of a global `V_global · M`.
  Big effect on exp4b VO tails at mid table counts.
- **`edbe6da` / `a6e908e`** split-yield catalog: eliminates information
  leaks that previously let clients see the catalog's future. Tightens
  physics for every experiment.

## 3. exp4a — FA-only, uniform tables

### 3.1 Representative cells (10 ms CAS)

| Cell | BAK success | **NEW success** | BAK mean | **NEW mean** | BAK P99 | **NEW P99** |
|---|---:|---:|---:|---:|---:|---:|
| 1 tbl, 20 ms IA  | 18.7 % | **13.9 %** | 1087 ms | **1443 ms** | 2073 ms | **2767 ms** |
| 1 tbl, 100 ms IA | 71.0 % | **55.9 %** | 958 ms  | **1337 ms** | 2079 ms | **2792 ms** |
| 5 tbl, 50 ms IA  | 98.8 % | **87.0 %** | 605 ms  | **1017 ms** | 1646 ms | **2273 ms** |
| 10 tbl, 20 ms IA | 94.1 % | **63.5 %** | 698 ms  | **981 ms**  | 1728 ms | **2025 ms** |
| 50 tbl, 20 ms IA | 99.5 % | **69.5 %** | 422 ms  | **701 ms**  | 807 ms  | **1279 ms** |
| 50 tbl, 100 ms IA| 100 %  | 100 %      | 373 ms  | **528 ms**  | 589 ms  | **909 ms**  |

Pattern:

- At **low load** (500 ms+ IA), curves shift up on latency (roughly the
  TM read+write pair, ~60–100 ms) but success rates are unchanged at 100 %.
- At **high load × low table count**, every cell gets worse — the knee
  where success rates collapse moves left in IA and up in table count.
- **50 tables at 20 ms IA no longer fully mitigates contention** (99.5 %
  → 69.5 %). The old numbers oversold the "just add tables" story.

### 3.2 What this means for the "tables × arrival" message

Blog claim (alt-text and body): "50 tables achieve 99.5 % even at 20 ms
inter-arrival vs 18.7 % with 1 table." New numbers:

> 50 tables hit **69.5 %** at 20 ms IA; 1 table is **13.9 %**.

Gap narrows (~3.5×) but the qualitative "tables fix single-table
saturation" finding survives at moderate loads. At 50 ms+ IA, 50 tables
remains ≥99 % at all CAS latencies we swept.

## 4. exp4a_zipf — FA-only, Zipf table selection

Zipf concentration (α=1.5) means the hot table absorbs most traffic,
limiting the benefit of adding more tables.

| Cell (CAS=50 ms) | BAK succ | **NEW succ** | BAK P99 | **NEW P99** |
|---|---:|---:|---:|---:|
| 10 tbl, 100 ms IA | 88.4 % | **78.9 %** | 2523 ms | **3035 ms** |
| 50 tbl, 100 ms IA | 92.5 % | **84.2 %** | 2389 ms | **2893 ms** |

| Cell (CAS=120 ms) | BAK succ | **NEW succ** | BAK P99 | **NEW P99** |
|---|---:|---:|---:|---:|
| 10 tbl, 100 ms IA | 67.8 % | **61.1 %** | 3799 ms | **4303 ms** |
| 50 tbl, 100 ms IA | 72.5 % | **64.8 %** | 3681 ms | **4183 ms** |

The "zipf 50 tables ≈ uniform 5 tables" analogy from the blog **still
holds**, but the absolute numbers shift down ~5–10 pp success. The
heatmap still shows the same "adding tables beyond 10 barely helps"
shape.

## 5. exp4b — 90/10 FA/VO mix, uniform

FA side tracks exp4a closely. The interesting changes are on VO:

| Cell (CAS=10 ms) | BAK VO succ | **NEW VO succ** | BAK VO P99 | **NEW VO P99** |
|---|---:|---:|---:|---:|
| 1 tbl, 100 ms IA | 69.7 %  | **55.5 %** | 31.7 s | **62.8 s** ⚠ |
| 1 tbl, 500 ms IA | 100 %   | 100 %      | 38.6 s | **40.4 s** |
| 10 tbl, 100 ms IA| 100 %   | 99.9 %     | 19.6 s | **19.5 s** |
| 50 tbl, 100 ms IA| 100 %   | 100 %      | 4.22 s | **4.39 s** |

| Cell (CAS=120 ms) | BAK VO succ | **NEW VO succ** | BAK VO P99 | **NEW VO P99** |
|---|---:|---:|---:|---:|
| 1 tbl, 100 ms IA | 45.4 % | **38.5 %** | 9.8 s  | **29.4 s** ⚠ |
| 10 tbl, 100 ms IA| 80.6 % | **70.8 %** | 16.0 s | **14.6 s** |
| 50 tbl, 100 ms IA| 81.9 % | **72.8 %** | 4.85 s | **4.71 s** |

Note the sign flip: **single-table VO tail roughly doubles** at the
contention boundary (100 ms IA, 1 tbl, CAS≥10 ms), but **multi-table VO
tails are essentially unchanged**. The per-table convoy decomposition
in `fa51753` correctly reduced multi-table overcounting; single-table
gets worse because the added TM read+write on every convoy retry now
compound at every table-version delta.

This is actually *more correct* — single-table VO under contention was
always the pathological case and now reflects that more honestly.

## 6. exp4b_zipf — 90/10 mix, Zipf selection

| Cell (CAS=50 ms, 100 ms IA) | BAK VO succ | **NEW VO succ** | BAK P99 | **NEW P99** |
|---|---:|---:|---:|---:|
| 10 tables | 87.9 % | **78.7 %** | 28.1 s | **29.3 s** |
| 50 tables | 92.2 % | **83.9 %** | 25.7 s | **25.3 s** |

Same pattern as 4b: success rates down, tails roughly unchanged. The
"hot-table dominates" narrative is unchanged.

## 7. Physics validation — multi-table ceiling

Per-table throughput bound is still 1/(5L) at S3 medians = **5.71 c/s
per table**. Aggregate bound for N uniform tables = `N × 5.71`. At 50
tables, 20 ms IA (offered rate ≈ 50 c/s over middle 80 % window), the
observed rate lands below the per-table ceiling × table count, as
expected — no super-physical throughput anywhere in the grid.

Zipf concentration: the hot table sees ~50 % of offered load regardless
of N. Its per-table bound (5.71 c/s) caps aggregate throughput at
~11.4 c/s once offered load exceeds that on the hot table, which
matches the observed ceiling in the Zipf heatmaps.

## 8. Required edits to `2026-03-23-providercatalog.md`

### 8.1 Qualitative story (KEEP)

- "More tables improve throughput" — **still true** for uniform, still
  tables up to ~10 for Zipf.
- "VO success degrades faster than FA as load grows" — still true.
- "Zipf concentrates contention on hot tables; adding tables helps less"
  — still true.
- "Catalog CAS latency up to 120 ms has modest effect at moderate
  loads" — still true, though the absolute latencies are now higher.

### 8.2 exp4a alt-text + heatmaps (HIGH IMPACT)

Every embedded heatmap needs replacement from `plots/exp4a_tables_fa/`
and `plots/exp4a_zipf_tables_fa/`. Notable alt-text substitutions:

| Blog text | Current value | New value |
|---|---|---|
| "50 tables at 20 ms is 99.5 % (vs 18.7 % with 1 table)" | 99.5 %/18.7 % | **69.5 %/13.9 %** (at CAS=10 ms) |
| "50 tables at 20 ms is 422 ms" (mean) | 422 ms | **701 ms** |
| "vs 1089 ms for 1 table" (mean)       | 1087 ms | **1443 ms** |
| "1 table at 20 ms reaches 2072 ms" (p99) | 2073 ms | **2767 ms** |
| "50 tables at 20 ms is 807 ms" (p99)  | 807 ms  | **1279 ms** |

(Full new tables are in `plots/exp4a_tables_fa/cas_*ms/heatmap_data.csv`;
the `experiments-bak` numbers are in `plots_bak_new/` for cross-check.)

### 8.3 exp4b alt-text + heatmaps (HIGH IMPACT)

The VO P99 numbers embedded in alt-text need careful review — many will
go up (single-table) or barely change (multi-table). Representative:

| Blog text | Current | New |
|---|---|---|
| "50 tables at 20 ms is 99.5 %" (FA mix) | 99.5 % | **87.6 %** (at CAS=1 ms), **69.6 %** (CAS=10 ms) |
| "1 table at 20 ms is 0.2 %" (VO mix)    | 0.2 %  | ~0 % still, but the surrounding narrative should note that VO at single table is catastrophic |
| "VO converges to FA success rates with enough tables" | still true, but threshold now ~20 tables (was ~10) |

### 8.4 Single-table retry curves (MEDIUM)

The blog discusses "per-table retry cost" as ~300 ms. With the corrected
per-attempt model this is now ~188 ms S3 median (non-inlined:
`TM_r + ML_r + ML_w + TM_w + CAS ≈ 27+27+60+60+1`). Update any mention
of "~300 ms retry" — the prior figure was inflated by the double-counted
manifest-file write.

### 8.5 Workload knee / "tables required for N c/s" claims

Any specific "N tables buys X c/s" language should be regenerated from
the new `latency_vs_throughput.md` tables in `plots/exp4{a,b}_*/` per
CAS latency. The shape of the curve survives; the exact thresholds shift
~1–2 table-counts higher for a given target success rate.

### 8.6 Parallel I/O footnote

Same issue as the 2026-03-09 post: remove any mention of "up to 4 I/O
operations in parallel." The simulator is serial. `max_parallel=4` has
been deleted from every config.

## 9. Exp4c is still pending

The 2026-03-23 post's **per-provider comparison tables** and the
`exp4c_*` heatmaps (Azure / Azure Premium / S3 / S3 Express / GCP) are
the core empirical result of the blog. They are still driven by the
pre-fix inlined-mode simulator and **must be regenerated before any
correction is published**.

The re-run command once the other fixes are committed:

```bash
.venv/bin/python scripts/run_all_experiments.py \
    --groups providers --seeds 5 \
    > experiment_logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimated runtime: ~5 hours at 72-parallel based on the 4a/4b timing.

**Do not publish a partial correction** that updates exp4a/4b but
leaves exp4c stale — it would create exactly the kind of inconsistent
picture (per-provider numbers from the old model vs. multi-table
numbers from the new) that the single-correction rule (`EXP1-4_UPDATE.md §5`)
was meant to prevent.

## 10. Directional summary for the editor

- **Nothing in the blog's argument needs to be rewritten.** Every
  conclusion (tables help, Zipf limits help, VO is expensive, catalog
  modest impact) survives.
- **Absolute numbers all move in the same direction**: lower success
  rates at contention boundaries, higher latencies across the board,
  wider tails for single-table VO under contention.
- **The "magic number" updates are mechanical**: replace cell values
  from the new heatmap CSVs and alt-text quantities pointed to in §8.
- **Exp4c is load-bearing** for the post's per-provider comparison and
  must be redone.

## 11. Artifacts

- `plots/exp4a_tables_fa/cas_*ms/*`, `plots/exp4a_zipf_tables_fa/cas_*ms/*` — new non-inlined heatmaps, ready to swap in.
- `plots/exp4b_tables_mix/cas_*ms/*`, `plots/exp4b_zipf_tables_mix/cas_*ms/*` — same for the mix.
- `plots_bak_new/exp4{a,b}*/` — `experiments-bak/` rendered through the current pipeline, so the blog's old numbers can be cross-checked without relying on transcribed alt-text.
- Companion `.md` tables in each plot subdir contain the numerical data.

Human-reviewable before/after delta is easiest via
`diff plots_bak_new/exp4{a,b}_*/cas_*ms/latency_vs_throughput.md
      plots/exp4{a,b}_*/cas_*ms/latency_vs_throughput.md`
— side-by-side they compress to a small set of per-cell substitutions.
