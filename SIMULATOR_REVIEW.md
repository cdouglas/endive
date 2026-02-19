# Simulator Fidelity Report: Endive vs. Apache Iceberg

## Executive Summary

The Endive simulator captures the high-level shape of Iceberg's optimistic concurrency control correctly, but contains **several significant inaccuracies in its conflict resolution model** that affect the validity of throughput arguments. The most important issue is that Iceberg's commit cost structure varies dramatically by operation type, and the simulator's uniform treatment obscures where the real throughput bottlenecks lie. The I/O convoy phenomenon described in `IO_CONVOY_ANALYSIS.md` is unrealistic for pure append workloads, but points toward a real and compelling problem: **validated overwrite operations (compaction, MERGE INTO) competing with high-throughput appends become practically un-committable** as append rates increase.

---

## 1. Accurately Modeled Aspects

### 1.1 Optimistic Concurrency Control Structure

The simulator correctly models the fundamental OCC pattern: read snapshot → do work → attempt CAS → retry on failure. This matches `SnapshotProducer.commit()` (core `SnapshotProducer.java:424-467`), which uses `Tasks.foreach().retry().run()` with the same logical structure.

### 1.2 Exponential Backoff Parameters

The simulator's backoff model (`base_ms × multiplier^(retry-1)` with jitter) closely matches Iceberg's implementation in `Tasks.runTaskWithRetry()` (`Tasks.java:403-469`):

- **Iceberg**: `minSleepTimeMs × scaleFactor^(attempt-1)`, capped at `maxSleepTimeMs`, with ±10% jitter
- **Simulator**: `base_ms × multiplier^(retry-1)`, capped at `max_ms`, with configurable jitter

The default parameters differ (Iceberg: 100ms base, 60s max, 4 retries; simulator: configurable), but the shape is correct.

### 1.3 Monotonic Version Advancement

The invariant that `catalog.seq` advances by exactly 1 on each successful commit accurately reflects Iceberg's snapshot sequencing, where each commit creates exactly one new snapshot with a monotonically increasing sequence number.

### 1.4 Manifest List as Unit of Storage Metadata

The simulator correctly identifies manifest lists as a key IO artifact. In Iceberg, `BaseSnapshot.cacheManifests()` (`BaseSnapshot.java:169-198`) lazy-loads the manifest list file when `dataManifests(io)` or `deleteManifests(io)` is first called, and this is indeed one of the primary storage IO operations during commit.

### 1.5 Parallel IO with Batch Limits

The simulator's batch parallelism model (process MAX_PARALLEL items, take max latency per batch) reasonably approximates Iceberg's use of `Tasks.range().executeWith(workerPool())` for parallel manifest operations in `ManifestFilterManager.filterManifests()` and `SnapshotProducer.apply()`.

---

## 2. Inaccurate Aspects

### 2.1 CRITICAL: False Conflict Cost Depends on Operation Type

**SIMULATOR.md claims** (lines 99-101):

> **False Conflict** - Version changed, no data overlap:
> - **Cost**: Read metadata root only (~1ms with fast catalog)
> - **No manifest file operations** required

**Actual Iceberg behavior** varies dramatically by operation type.

#### FastAppend — The Cheapest Path

`FastAppend` extends `SnapshotProducer` directly (`FastAppend.java:36`) and has **no `validate()` override** — it inherits the empty default at `SnapshotProducer.java:241`. On retry, `SnapshotProducer.apply()` calls `refresh()` then the subclass `apply(base, snapshot)` (`FastAppend.java:146-170`), which does:

1. `refresh()` → reads metadata file from storage
2. `writeNewManifests()` → cached from first attempt, no IO
3. `snapshot.allManifests(ops().io())` → reads ONE manifest list (the current snapshot's)
4. Writes a new manifest list via `SnapshotProducer.apply()` (`SnapshotProducer.java:263-289`)

**Cost: 1 metadata read + 1 manifest list read + 1 manifest list write. NO historical manifest lists. NO manifest file reads.** Appends can't conflict by definition — new data files are added alongside existing ones.

#### MergeAppend — The Typical Append Path

`MergeAppend` extends `MergingSnapshotProducer` (`MergeAppend.java:24`) but also does NOT override `validate()` — it inherits the same empty default. However, `MergingSnapshotProducer.apply()` (`MergingSnapshotProducer.java:922-972`) does additional work:

1. `filterManager.filterManifests()` — iterates every manifest in the current snapshot, but for pure appends (no delete expression, no dropped files) the `canContainDeletedFiles()` check (`ManifestFilterManager.java:380-392`) returns `false` via in-memory metadata checks. **No manifest file reads.** Results are cached per-manifest.

2. `mergeManager.mergeManifests()` — bin-packs small manifests and merges them by reading and rewriting manifest files (`ManifestMergeManager.createManifest()` at lines 163-207). Results are cached by input bin (line 167), **BUT on retry the base snapshot is different** (new manifests from concurrent commits), so bin composition changes → cache miss → manifest files re-read and re-merged.

If N concurrent commits each added a manifest during the transaction's execution, the current snapshot has ~N more manifests. The merge manager may need to re-read and re-merge these. **This IS O(N) storage reads proportional to commit rate** — similar in magnitude to the simulator's N manifest list reads, but located in manifest FILES (during merging), not manifest LISTS (during validation). The aggregate IO volume is comparable; the simulator attributes it to the wrong artifact.

#### Overwrites with Validation — The Expensive Path

`BaseOverwriteFiles`, `BaseReplacePartitions`, and `BaseRowDelta` extend `MergingSnapshotProducer` and override `validate()` to call `validationHistory()` (`MergingSnapshotProducer.java:868-907`). This:

1. Iterates ALL intermediate snapshots via `SnapshotUtil.ancestorsBetween()`
2. For EACH, calls `currentSnapshot.dataManifests(ops().io())` → reads that snapshot's manifest list file (`BaseSnapshot.java:183-185`)
3. Filters manifests by snapshot ID (in-memory)
4. Passes collected manifests to `ManifestGroup.entries()` which reads actual manifest files (`ManifestGroup.java:302-356`)

**Cost: 1 metadata read + N manifest list reads (one per intermediate snapshot) + M manifest file reads + manifest list rebuild.** This is the path that matches the simulator's cost model most closely.

**Important caveat**: Validation is **opt-in** even for overwrite operations. `BaseReplacePartitions` initializes `validateConflictingData = false` and `validateConflictingDeletes = false` (`BaseReplacePartitions.java:31-32`). The default mode is described in the API as "idempotent" — the overwrite commits regardless of concurrent changes. Validation must be explicitly requested by the caller.

#### Summary of Retry Costs by Operation Type

| Step | FastAppend | MergeAppend | Overwrite (validated) |
|------|-----------|-------------|----------------------|
| Refresh metadata | 1 read | 1 read | 1 read |
| Validate (historical MLs) | — | — | N ML reads |
| Validate (manifest files) | — | — | M manifest reads |
| Read current ML | 1 read | 1 read | 1 read |
| Filter manifests | — | in-memory | in-memory |
| Merge manifests | — | ~K file reads + writes | ~K file reads + writes |
| Write new ML | 1 write | 1 write | 1 write |

### 2.2 CRITICAL: Real Conflicts Abort — They Are Not Merged

**The simulator models** real conflicts as expensive-but-recoverable: read manifest list + read N manifest files + write N merged manifest files + retry CAS.

**In actual Iceberg**, when validation finds a real conflict, it throws `ValidationException` — for example in `validateAddedDataFiles()` (`MergingSnapshotProducer.java:360-367`):

```java
if (conflicts.hasNext()) {
    throw new ValidationException(
        "Found conflicting files that can contain records matching partitions %s: %s", ...);
}
```

`ValidationException` is **NOT** `CommitFailedException`, so the retry loop at `SnapshotProducer.commit()` (line 436: `.onlyRetryOn(CommitFailedException.class)`) does **not catch it**. The operation **aborts permanently** at the Iceberg level.

There is no "merging conflicting manifest files" step in Iceberg's commit protocol. The retry loop only handles CAS failures (metadata pointer stale), not data-level conflicts.

**What the simulator could model instead**: After a `ValidationException`, the *application* (not Iceberg) may choose to retry. This means starting an entirely new Iceberg operation from scratch: re-read the table, re-compute changes (possibly re-scanning source data), and attempt a new commit. The cost is essentially the full transaction runtime again, not just manifest IO. Alternatively, the simulator could model real conflicts as permanent aborts, which is what Iceberg itself does.

### 2.3 The Retry Loop Structure is Inverted

**The simulator models**: CAS attempt → failure → conflict resolution (IO) → retry CAS

**Iceberg actually does** (visible in `SnapshotProducer.commit()` at lines 438-466):

```java
.run(taskOps -> {
    Snapshot newSnapshot = apply();              // ← ALL expensive IO happens here
    // ... in-memory metadata building ...
    taskOps.commit(base, updated.withUUID());    // ← CAS (cheap pointer swap)
});
```

Where `apply()` (`SnapshotProducer.java:253-261`) is:

```java
public Snapshot apply() {
    refresh();                                              // 1. Read metadata from storage
    Snapshot parentSnapshot = ...;                          // 2. In-memory lookup
    validate(base, parentSnapshot);                         // 3. Validation IO (operation-dependent)
    List<ManifestFile> manifests = apply(base, parentSnapshot); // 4. Build manifests (IO)
    // ... write manifest list to storage ...                // 5. Write new ML
    return new BaseSnapshot(...);
}
```

The expensive IO happens **before** the CAS attempt, not after. Each retry calls `apply()` at line 440, which re-does all five steps before `taskOps.commit()` is even called.

This means:
- A failed CAS means ALL the preparation work (validation, manifest building) was wasted
- The "conflict resolution" cost is really the cost of the NEXT attempt's preparation, not a recovery step after the failed attempt
- For the simulator, this distinction doesn't change the aggregate IO per retry, but it changes the timing: the expensive IO occurs during the period *between* CAS attempts, not after the failure

### 2.4 Manifest Caching on Retry is Partially Effective

Iceberg has caching optimizations across retries, but their effectiveness varies:

**Effective caches (survive retry):**
- **`MergingSnapshotProducer.cachedNewDataManifests`** — manifest files for *this commit's new data* are reused (`MergingSnapshotProducer.java:1050-1081`)
- **`FastAppend.newManifests`** — `writeNewManifests()` skips rewriting if already done
- **`ManifestFilterManager.filteredManifests`** — per-manifest filter results cached; manifests unchanged from the previous attempt get cache hits

**Partially effective caches (cache miss on changed inputs):**
- **`ManifestMergeManager.mergedManifests`** — cached by input bin (`ManifestMergeManager.java:167`), but on retry the base snapshot has new manifests from concurrent commits → different bins → cache miss for affected bins → manifest files re-read and re-merged

The net effect: the first attempt is most expensive (writes new data manifests + merges). Subsequent retries reuse the new data manifests but must re-merge with the changed base.

### 2.5 The Two-Level Commit Architecture Mismatch

In standard Iceberg, each table has its **own independent commit retry loop**. Table A's commits don't interfere with Table B's commits at all — there is no shared sequence number or global CAS.

The simulator models a **catalog-level CAS** where all tables share a global sequence number. With G=1, ANY concurrent write causes a conflict regardless of which table it touches. This is the FileIOCatalog model, not standard Iceberg.

The simulator acknowledges this with the G=T mode (per-table CAS checks), but even in G=T mode, the conflict resolution logic (reading N manifest lists for N snapshots behind at the CATALOG level) doesn't match Iceberg's per-table retry, which only cares about the specific table being committed.

---

## 3. Assessment of the I/O Convoy Analysis

The I/O convoy described in `IO_CONVOY_ANALYSIS.md` follows logically from the simulator's assumptions but is unrealistic for its stated scenario (uniform workload at 50 commits/sec). However, it points toward a **real and important problem** in mixed workloads.

### 3.1 Pure Append Workloads: No Convoy

The convoy depends on transactions reading N manifest lists for N intermediate snapshots. In Iceberg, both `FastAppend` and `MergeAppend` have **no validation** and never read historical manifest lists, regardless of how far behind they are.

For MergeAppend, the retry cost is O(N) manifest FILE reads for re-merging (not manifest LIST reads for validation), where N is the number of new manifests in the current snapshot. This is significant but bounded differently than the convoy model predicts — it depends on the *current state* of the table, not the full *history* between snapshots.

### 3.2 Pure Overwrite Workloads: Real Conflicts Abort

If many concurrent commits touch the same partition, validation finds actual conflicting data files and throws `ValidationException`. The operation aborts immediately — there is no multi-second manifest-reading phase followed by a successful retry.

### 3.3 The Real Convoy: Mixed Workloads (Appends + Overwrites)

The I/O convoy IS realistic for **validated overwrite operations competing with high-throughput appends**. Consider:

- **Streaming ingestion**: 50 `FastAppend` commits/sec (cheap, no validation)
- **Periodic compaction**: `BaseOverwriteFiles` with validation, runs for 3 minutes

The compaction operation starts, captures a snapshot, runs for 3 minutes. During those 3 minutes, 9,000 append commits land on the table. When compaction tries to commit:

1. CAS fails (`CommitFailedException` — metadata pointer stale)
2. Retry calls `apply()` → `refresh()` → `validate()`:
   - `validationHistory()` iterates 9,000 intermediate snapshots
   - For EACH: reads the snapshot's manifest list (`snapshot.dataManifests(ops().io())`)
   - Checks manifest metadata for partition overlap with the compaction
3. If no overlap (appends touch different partitions) → validation passes → rebuild ML → retry CAS
4. But during the ~minutes spent reading 9,000 manifest lists, thousands more appends landed...
5. Next retry reads even more manifest lists → validation time grows → **eventually exceeds the 30-minute retry budget**

**This makes maintenance operations practically un-committable at high append rates**, even when the maintenance touches completely different partitions from the appends. The validation IO is O(append_rate × overwrite_runtime) per retry, and each retry faces a moving target.

This is the strongest honest version of the throughput argument: **the ceiling on a table's effective throughput is set not by the append rate alone, but by the operational cost of maintenance operations that must validate against a growing commit history.**

### 3.4 Corrected Convoy Cost by Scenario

| Scenario | Retry Cost | Convoy? |
|----------|-----------|---------|
| Pure FastAppend | ~160ms (1 metadata + 1 ML + 1 ML write) | No |
| Pure MergeAppend | ~160ms + O(N) manifest merge IO (cached partially) | Mild |
| Pure Overwrite (no validation) | ~160ms + manifest merge IO | No |
| Pure Overwrite (validated) | O(N) ML reads where N = intermediate snapshots | Yes |
| **Appends + validated overwrite** | **O(append_rate × runtime) ML reads per retry** | **Yes — this is the real problem** |

---

## 4. Recommendations

### 4.1 Model Two Operation Classes

Rather than a single uniform operation type, the simulator should model a mixed workload with two classes:

| | Append (majority) | Overwrite/Maintenance (minority) |
|---|---|---|
| Validate | No | Yes (reads N historical MLs) |
| Retry cost (false conflict) | O(1) ML reads | O(N) ML reads, N = snapshots behind |
| Real conflict | Cannot happen (appends are additive) | Abort (ValidationException) |
| False conflict | Cheap rebase: metadata + 1 ML read + 1 ML write | Expensive validation proportional to commit rate |

The interesting throughput question becomes: **at what append rate do occasional maintenance operations become un-committable?** This is parameterized by:
- `append_rate`: commits/sec for append operations
- `overwrite_runtime`: execution time for maintenance operations
- `overwrite_interval`: how often maintenance operations are attempted
- `validation_overlap`: probability that appends and overwrites touch the same partitions

### 4.2 Correct the False Conflict Cost Per Operation Type

**For append operations**, false conflict cost should be:

- Read metadata file (`T_METADATA_ROOT`)
- Read current manifest list (`T_MANIFEST_LIST` read) — just ONE, not N
- For MergeAppend: re-merge manifests — O(K) manifest file reads/writes where K depends on bin-packing against the new snapshot's manifest set
- Write new manifest list (`T_MANIFEST_LIST` write)
- **No historical manifest list reads**

**For overwrite operations with validation**, false conflict cost should be:

- All of the above, PLUS
- Read N manifest lists for N intermediate snapshots (as currently modeled — this is correct)
- Possibly read manifest files if partition metadata in the manifest list isn't sufficient to rule out conflicts (often it is, via `ManifestFile.partitions()` bounds)

### 4.3 Model Real Conflicts Accurately

Two options, depending on the desired fidelity:

**Option A (simple, accurate):** Real conflicts cause permanent abort. The transaction fails. The application may start a completely new operation from scratch (cost = full transaction runtime again). This matches Iceberg's actual `ValidationException` behavior.

**Option B (nuanced):** Distinguish between CAS failure and validation failure:
- **CAS failure** (another commit landed): Iceberg retries automatically. The retry pays the "false conflict" cost from §4.2 above. If validation passes on retry, the commit proceeds.
- **Validation failure** (actual data overlap): Iceberg aborts. Application-level retry means full restart.

Option B is more realistic. The key insight: what the simulator currently calls a "false conflict" is really Iceberg's **CAS failure with successful re-validation**, and what it calls a "real conflict" is really **CAS failure followed by validation failure → abort**.

### 4.4 Move Expensive IO Before CAS, Not After

Restructure the retry flow to match Iceberg's actual architecture:

```
retry:
  refresh metadata                          (IO)
  validate against new base                 (IO — manifest list/file reads, operation-dependent)
  if validation fails: abort                (no retry for data conflicts)
  build new manifest list                   (IO — manifest merge + ML write)
  attempt CAS                               (cheap)
  if CAS fails: goto retry
```

Currently the simulator does: `CAS → fail → resolve conflict (IO) → retry CAS`. The correct ordering places the IO BEFORE the CAS. This doesn't change aggregate IO per retry but changes timing: the expensive work occurs between CAS attempts, not as post-failure recovery.

### 4.5 Model the Mixed-Workload Throughput Ceiling

The strongest argument the simulator can make is about **mixed workloads**. The recommended experiment:

1. Fix append rate at various levels (10, 50, 100, 500 commits/sec)
2. Inject periodic validated overwrite operations (e.g., one every 5 minutes, runtime 3 minutes)
3. Measure: can the overwrite operations commit within the retry budget?
4. Find the append rate at which overwrites become un-committable

The expected result: there exists an append rate threshold above which maintenance operations cannot complete validation before the retry budget expires. This threshold depends on ML read latency, parallelism, and the overwrite's execution time.

This is a more compelling and honest argument than "all operations hit the I/O convoy" because:
- It reflects a real operational problem (compaction competing with ingestion)
- The cost model is accurate (validation DOES read N historical MLs)
- The failure mode is realistic (retry budget exceeded, not infinite retries)

### 4.6 Separate the Catalog-Level vs Table-Level CAS

If the simulator is meant to model FileIOCatalog specifically (catalog-level CAS), document this clearly and note that the conflict resolution costs are different from standard Iceberg table-level commits. If it's meant to model general Iceberg, the G=T mode should use the table-level cost model described above.

---

## 5. Summary Table

| Aspect | Simulator | Actual Iceberg | Verdict |
|--------|-----------|---------------|---------|
| OCC with CAS | Global or per-table seq | Per-table metadata pointer | **Correct** for G=T |
| Exponential backoff | Configurable | 100ms base, 2.0×, 60s max, 4 retries | **Correct** shape |
| False conflict (FastAppend) | N ML reads + metadata | 1 metadata + 1 ML read + 1 ML write | **Overstated ~175×** |
| False conflict (MergeAppend) | N ML reads | ~K manifest file reads for re-merge | **Wrong artifact** (files not lists), **comparable magnitude** |
| False conflict (validated overwrite) | N ML reads + metadata | N ML reads + M manifest reads + rebuild | **Understated** (missing manifest file reads) |
| Real conflict | Merge manifests + retry | `ValidationException` → **abort** | **Wrong** (abort, not merge) |
| Retry IO placement | After CAS failure | Before CAS attempt | **Inverted** |
| Manifest caching across retries | Not modeled | Effective for new data; partial for merges | **Missing** |
| Operation type distinction | Uniform | FastAppend / MergeAppend / Validated Overwrite | **Missing** |
| I/O convoy (pure appends) | 28s retry cost | ~160ms retry cost | **Unrealistic** |
| I/O convoy (mixed workload) | Not specifically modeled | Validated overwrites become un-committable | **Real problem, not yet captured** |

---

## 6. Key Source Locations Referenced

| File | Lines | Relevant Content |
|------|-------|-----------------|
| `core/.../SnapshotProducer.java` | 241 | Empty default `validate()` |
| `core/.../SnapshotProducer.java` | 253-327 | `apply()` — refresh, validate, build, write ML |
| `core/.../SnapshotProducer.java` | 424-467 | `commit()` — retry loop with `Tasks.foreach()` |
| `core/.../FastAppend.java` | 36, 146-170 | No validation; `apply()` reads ONE manifest list |
| `core/.../MergeAppend.java` | 24 | Extends `MergingSnapshotProducer`, no `validate()` override |
| `core/.../MergingSnapshotProducer.java` | 355-404 | `validateAddedDataFiles()` — throws `ValidationException` |
| `core/.../MergingSnapshotProducer.java` | 868-907 | `validationHistory()` — reads N manifest lists |
| `core/.../MergingSnapshotProducer.java` | 922-972 | `apply()` — filters and merges manifests |
| `core/.../BaseSnapshot.java` | 169-198 | `cacheManifests()` — lazy manifest list loading |
| `core/.../BaseReplacePartitions.java` | 31-32, 89-110 | Validation flags (opt-in, default off) |
| `core/.../ManifestGroup.java` | 302-356 | `entries()` — reads manifest files from storage |
| `core/.../ManifestFilterManager.java` | 208-231, 350-392 | `filterManifests()` — parallel filter with caching |
| `core/.../ManifestMergeManager.java` | 47, 72-207 | `mergeManifests()` — bin-pack and merge with cache |
| `core/.../Tasks.java` | 403-469 | Retry loop with exponential backoff |
| `core/.../TableProperties.java` | 86-93 | Retry defaults: 4 retries, 100ms min, 60s max |
