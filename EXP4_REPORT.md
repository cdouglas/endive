# Blog Update Report — exp4 Post-Fix Results

Written 2026-04-16 (exp4a/b), updated 2026-04-17 (exp4c). Companion to
`EXP1-3_REPORT.md`. Covers exp4a (FA-only multi-table), exp4a_zipf,
exp4b (90/10 mix multi-table), exp4b_zipf, and exp4c (real-provider
sweep across S3, S3 Express, Azure, Azure Premium, GCS).

## 1. Validation

| Check | Result |
|---|---|
| Templates `experiment_configs/exp4{a,b,c}_*.toml` | `table_metadata_inlined = false` ✓ |
| Stored `cfg.toml` across all 1860 exp4a/b/c dirs | `table_metadata_inlined = false` ✓ |
| Sweep coverage — exp4a/b | 4×240 configs, 4×1200 seeds — all present |
| Sweep coverage — exp4c | 900 configs, 4499/4500 seeds — 1 seed crashed at 96.7 % (S3x, 1 tbl, 20 ms IA, 50/50 mix — the heaviest possible config; ran 7.5 h). 5-seed aggregation unaffected. |
| `expctl list` | `stale (code)` — from `1be49a1` (AppendCatalog rewrite), not reachable from CAS catalog path |

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

## 9. Exp4c — real-provider sweep (the centerpiece)

Exp4c sweeps 5 providers × 6 table counts × 3 FA/VO mixes × 10 arrival
rates = 900 configs. All on `backend="storage"` (CAS through each
provider's own conditional write). 4499/4500 seeds complete.

### 9.1 Workload knee table (>95 % success threshold)

This table is the blog post's main empirical result. Comparing
`plots/exp4c_tables_providers/workload_knee/workload_knee_table.md`
(new) vs. `plots_bak_new/exp4c_tables_providers/workload_knee/` (bak):

#### Single table (the headline row)

| Provider | BAK FA-only | **NEW FA-only** | BAK 90/10 | **NEW 90/10** |
|---|---:|---:|---:|---:|
| S3 Express  | 14.6 c/s | **14.6 c/s** | 7.5 c/s | **7.4 c/s** |
| S3 Standard | 2.4 c/s  | **1.8 c/s**  | 1.8 c/s | **1.8 c/s** |
| Azure Prem  | 2.5 c/s  | **2.4 c/s**  | 1.9 c/s | **1.8 c/s** |
| Azure Std   | 2.4 c/s  | **1.8 c/s**  | 1.5 c/s | **1.5 c/s** |
| GCP         | 0.7 c/s  | **0.4 c/s**  | 0.4 c/s | **0.4 c/s** |

S3 Express is unchanged — its per-op latency is low enough (~10 ms)
that the additional TM pair barely dents the per-table bound. For S3
Standard, the per-attempt cost roughly doubles (3 → 5 ops) and the
FA-only knee drops proportionally from 2.4 → 1.8 c/s. GCP is hit
hardest in relative terms (0.7 → 0.4 c/s FA-only).

The **90/10 mix** knees are largely unchanged across all providers
because VO was already the binding constraint and VO cost is dominated
by the convoy, not the per-attempt cost.

#### Multi-table scaling (the structural change)

| Provider | Tables | BAK FA-only | **NEW FA-only** | Change |
|---|---:|---:|---:|---|
| S3 Standard | 5  | 7.2 c/s | **3.7 c/s** | **−49 %** |
| S3 Standard | 10 | 7.4 c/s | **3.7 c/s** | −50 % |
| S3 Standard | 20 | 7.4 c/s | **3.7 c/s** | −50 % |
| S3 Standard | 50 | 7.4 c/s | **7.2 c/s** | −3 % (catalog-bound) |
| Azure Prem  | 5  | 7.2 c/s | **3.7 c/s** | −49 % |
| Azure Prem  | 10 | 7.4 c/s | **3.7 c/s** | −50 % |
| Azure Prem  | 50 | 7.4 c/s | **3.7 c/s** | −50 % |
| GCP         | 5  | 1.8 c/s | **0.7 c/s** | −61 % |
| GCP         | 10 | 2.4 c/s | **0.7 c/s** | −71 % |
| GCP         | 50 | 3.6 c/s | **0.7 c/s** | −81 % |
| S3 Express  | 50 | 14.9 c/s| **36.0 c/s** | **+142 %** |

The **biggest structural change**: for S3 Standard at 5–20 tables,
the FA-only knee was 7.2–7.4 c/s (catalog-CAS-bound); now it's
**3.7 c/s** (per-table-bound). The per-table ceiling dropped from
1/(3L) ≈ 11.4 c/s (old inlined model) to 1/(5L) ≈ 5.7 c/s
(non-inlined). When the per-table bound tightens, it binds before
the catalog CAS does, flattening the multi-table scaling curve.

Only at **50 tables** does the catalog CAS again become the bottleneck
(7.2 c/s, nearly matching the old 7.4 c/s) — because per-table
offered load drops below the per-table bound.

S3 Express at 50 tables **jumps to 36 c/s** (was 14.9). The old model's
per-table inlined bound was already above the catalog-CAS rate, so
the old 14.9 was catalog-limited. The non-inlined model has a higher
per-attempt cost but S3 Express's CAS is so fast (~5 ms) that the
catalog remains the bottleneck only up to ~20 tables. At 50 tables,
per-table contention drops enough that the free-retry fix (`ec383ff`:
cross-table CAS failures skip manifest I/O) lets more commits through.

GCP is hit hardest because its per-op latency is already high
(~118 ms). The 5-op non-inlined cost of ~590 ms limits each table to
~1.7 c/s, so even at 50 tables the per-table bound is still binding
(0.7 c/s × 50 = 35 c/s theoretical, but GCP's CAS at ~118 ms caps
the catalog at ~8.5 c/s). The combination gives 0.7 c/s aggregate
at the knee — not enough per-table traffic to exceed the per-table
bound *and* not enough tables to overcome the CAS latency.

### 9.2 Per-provider heatmaps

All `plots/exp4c_tables_providers/{provider}_{mix}/` heatmaps need
replacement. The same "success drops, latency rises" pattern from
exp4a/4b applies, but the *magnitude* varies by provider. Representative:

| Provider | Cell (1 tbl, 100 ms IA, FA=90%) | BAK FA succ | **NEW FA succ** |
|---|---|---:|---:|
| S3 Express | | 76.2 % | **58.3 %** |
| S3 Standard | | 60.0 % | **48.4 %** |
| Azure Premium | | 64.0 % | **51.2 %** |
| Azure Standard | | 55.7 % | **45.8 %** |
| GCP | | 23.3 % | **16.4 %** |

(Approximate values read from heatmap CSV. Full precision in the
per-provider `heatmap_data.csv` files.)

### 9.3 VO latencies in the provider comparison

VO P99 latencies at the single-table knee are now:

| Provider | BAK VO P99 (1 tbl, 90/10, at knee) | **NEW VO P99** |
|---|---:|---:|
| S3 Express | 6.6 s | **6.7 s** |
| S3 Standard | 19.7 s | **20.2 s** |
| Azure Premium | 21.2 s | **21.7 s** |
| Azure Standard | 23.3 s | **22.6 s** |
| GCP | 26.9 s | **32.9 s** |

VO tails are within ±10 % of the old numbers across all providers.
The convoy cost model changes (§2 errata) largely cancel out for VO
because the per-table decomposition correction roughly offsets the
added TM pair. GCP's VO tail grows more because its higher per-op
latency amplifies the TM pair.

## 10. Required edits to `2026-03-23-providercatalog.md` — exp4c additions

(Extending §8 from the exp4a/b report.)

### 10.1 Workload knee / per-provider throughput table (HIGH IMPACT)

The blog's summary table of "sustainable commit rates by provider" is
the post's main claim. **Replace entirely** from the new
`plots/exp4c_tables_providers/workload_knee/workload_knee_table.md`.

Key headline substitutions:

| Blog claim | New value |
|---|---|
| "S3 Standard: 2.4 c/s FA, 1.8 c/s mixed" (1 table) | **1.8 c/s FA, 1.8 c/s mixed** |
| "S3 Express sustains 14.6 c/s on a single table" | **still 14.6 c/s** ✓ |
| "GCP: 0.7 c/s FA" (1 table) | **0.4 c/s** |
| "S3 Standard at 10+ tables: 7.4 c/s FA" | **3.7 c/s** ⚠ (most impactful change) |
| "S3 Express at 50 tables: 14.9 c/s FA" | **36.0 c/s** (up — free retry helps) |

### 10.2 "More tables = more throughput" narrative (MEDIUM)

The blog's scaling narrative claims that adding tables shifts the
bottleneck from per-table metadata I/O to catalog CAS. This is still
true but the transition now happens at **~50 tables** instead of ~5
for S3/Azure. The passage should explain that non-inlined metadata adds
enough per-table cost that the per-table bound binds through 20 tables
for S3 Standard.

### 10.3 All exp4c heatmaps (HIGH IMPACT)

Every `exp4c_*` PNG in the blog must be replaced from
`plots/exp4c_tables_providers/`. Alt-text that quotes specific cell
values must be updated cell-by-cell from the new heatmap CSVs.

### 10.4 GCP scaling claim

If the blog claims GCP benefits substantially from multi-table scaling,
that needs softening. Old: 0.7 c/s (1 tbl) → 3.6 c/s (50 tbl) FA.
New: 0.4 c/s (1 tbl) → 0.7 c/s (50 tbl) FA. The per-table bound is
now so tight for GCP that adding tables barely helps.

## 11. Directional summary for the editor

- **Nothing in the blog's argument needs to be rewritten.** Every
  conclusion (tables help, Zipf limits help, VO is expensive, catalog
  modest impact) survives — but the "tables help" story needs
  quantitative hedging: the per-table bound is now tighter, so the
  scaling flattens sooner.
- **Absolute numbers all move in the same direction**: lower success
  rates at contention boundaries, higher latencies across the board,
  wider tails for single-table VO under contention.
- **The biggest structural change**: multi-table FA-only scaling for
  S3/Azure at 5–20 tables drops from ~7.4 to ~3.7 c/s. The blog's
  "spreading across tables" argument is less impactful than presented.
  Only at 50 tables does catalog CAS again become the bottleneck.
- **S3 Express is the exception**: its fast ops mean the per-table bound
  rarely binds. At 50 tables it jumps to 36 c/s (free-retry benefit).
- **GCP scaling is weaker than presented**: 50 tables at GCP only
  reaches 0.7 c/s FA (was 3.6 c/s in the old model).

## 12. Artifacts

- `plots/exp4a_tables_fa/cas_*ms/*`, `plots/exp4a_zipf_tables_fa/cas_*ms/*` — new non-inlined heatmaps, ready to swap in.
- `plots/exp4b_tables_mix/cas_*ms/*`, `plots/exp4b_zipf_tables_mix/cas_*ms/*` — same for the mix.
- `plots/exp4c_tables_providers/` — new per-provider heatmaps, workload knee tables, conflict-type breakdowns. 48 graph sets generated.
- `plots_bak_new/exp4{a,b,c}*/` — `experiments-bak/` rendered through the current pipeline for apples-to-apples comparison.
- Companion `.md` tables in each plot subdir contain the numerical data.

Human-reviewable before/after delta for the workload knee:
```bash
diff plots_bak_new/exp4c_tables_providers/workload_knee/workload_knee_table.md \
     plots/exp4c_tables_providers/workload_knee/workload_knee_table.md
```
