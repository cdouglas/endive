# Append Simulation Design

This document describes how the Endive simulator models append-based operations for Apache Iceberg catalogs, specifically the `LogCatalogFormat` approach where transactions append to a catalog log rather than performing compare-and-swap (CAS) on the entire catalog state.

## Overview

The append model introduces several orthogonal optimizations:

1. **Catalog Log Append**: Instead of CAS on catalog state, transactions append intention records. Conflicts are detected at the table level, not catalog level.

2. **Manifest List Append (ML+)**: Instead of rewriting manifest lists, transactions append tentative entries tagged with transaction ID. Readers filter entries based on committed transactions. Entry validity is determined by catalog commit outcome, not ML append.

3. **Table Metadata Inlining**: Table metadata can be inlined in the catalog/intention record, eliminating separate I/O operations.

All can be enabled independently via configuration. In particular, ML+ works with both CAS and append catalog modes.

## Catalog Log Append Model

### Real-World Behavior (LogCatalogFormat)

In Iceberg's `LogCatalogFormat`:
- The catalog is stored as a checkpoint plus a log of transactions
- Each transaction appends a `TRANSACTION` record containing its actions (table updates, reads, creates)
- On append, the transaction is verified against the current catalog state
- If verification fails (table version mismatch), the intention record is in the log but not applied
- When the log exceeds a threshold (16MB), the next commit seals the log and triggers compaction via CAS

### Simulation Model

```
┌─────────────────────────────────────────────────────────────┐
│                      AppendCatalog                          │
├─────────────────────────────────────────────────────────────┤
│  seq: int                    # Number of committed txns     │
│  tbl: list[int]              # Per-table versions           │
│  log_offset: int             # Current log byte position    │
│  checkpoint_offset: int      # Offset at last compaction    │
│  entries_since_checkpoint: int                              │
│  sealed: bool                # True if compaction needed    │
│  committed_txn: set          # Successful transaction IDs   │
│  ml_offset: list[int]        # Per-table manifest list offset│
│  ml_sealed: list[bool]       # Per-table manifest list sealed│
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The simulation does NOT store log entries. The catalog "knows" whether intention records will be applied because validation happens at append time. We track:
- `log_offset` for physical conflict detection
- `tbl` for table-level validation (like CAS, but per-table)
- `committed_txn` for deduplication
- `ml_offset` / `ml_sealed` for manifest list append state

**Key simplifications:**
- Log entries are fixed-size (`LOG_ENTRY_SIZE`, default 100 bytes) rather than variable
- Transaction IDs are integers rather than UUIDs

### Commit Protocol

The key insight is that **physical conflict rate bounds logical conflict rate**. Physical failures are cheap (just retry at new offset); logical failures require reading manifest lists to repair.

**Important:** The transaction only knows whether the physical append succeeded. After physical success, it must re-read the catalog to discover the logical outcome. The **simulator** computes the outcome at append time, but models the I/O cost of the transaction discovering it.

```
Transaction starts:
  1. Capture catalog.log_offset as v_log_offset
  2. Capture table versions in v_tblr, v_tblw
  3. Capture per-table manifest list offsets (if manifest_list_mode = append)
  4. If table_metadata_inlined = false: Read table metadata (I/O cost)

Transaction executes:
  5. Run transaction logic (sim.timeout for runtime)
  6. If table_metadata_inlined = false: Write table metadata (I/O cost)
  7. Write manifest lists (rewrite or append mode)

Transaction commits (txn_commit_append):
  8. If catalog.sealed → perform compaction CAS first
  9. Create intention record with tables_written, tables_read
 10. Attempt physical append at expected offset:

     Physical Failure (offset moved):
       - New offset returned by failed append
       - Retry at new offset (no I/O needed)

     Physical Success:
       - Intention record appended to log
       - If log exceeds threshold → seal for compaction

 11. Re-read catalog to discover logical outcome (I/O cost modeled)
     - Simulator already computed outcome, but transaction must "discover" it

 12. Based on logical outcome:

     Logical Success:
       - Intention record was applied (table versions matched)
       - Transaction committed

     Logical Failure:
       - Table versions conflicted, intention record not applied
       - Must repair: read manifest list, resolve conflicts
       - If table_metadata_inlined: catalog state already available
       - If tables commute (no overlapping data), no manifest re-append needed
```

### Validation at Append Time

Validation happens inside `try_APPEND`, same as CAS but at table-level:

```python
def try_APPEND(self, sim, txn, entry) -> tuple[bool, bool]:
    # Physical check: Does offset match?
    if self.log_offset != txn.v_log_offset:
        return (False, None)  # Physical failure - retry at new offset

    # Logical check: Do table versions match? (like CAS, but per-table)
    for tbl_id, new_ver in entry.tables_written.items():
        expected_ver = new_ver - 1
        if self.tbl[tbl_id] != expected_ver:
            self.log_offset += LOG_ENTRY_SIZE  # Entry in log, not applied
            return (True, False)  # Logical failure

    # Check read-set for serializable isolation
    for tbl_id, read_ver in entry.tables_read.items():
        if tbl_id not in entry.tables_written:
            if self.tbl[tbl_id] != read_ver:
                self.log_offset += LOG_ENTRY_SIZE
                return (True, False)

    # Success - update table state
    for tbl_id, new_ver in entry.tables_written.items():
        self.tbl[tbl_id] = new_ver
    self.committed_txn.add(entry.txn_id)
    self.log_offset += LOG_ENTRY_SIZE
    return (True, True)  # Committed
```

This allows concurrent transactions writing to **different tables** to both succeed, unlike CAS mode where any concurrent commit causes failure.

### Compaction

Compaction is triggered when **either** condition is met (minimum of the two):
1. **Size threshold**: `log_offset - checkpoint_offset > COMPACTION_THRESHOLD`
2. **Entry count threshold**: `entries_since_checkpoint >= COMPACTION_MAX_ENTRIES` (if > 0)

When triggered:
1. The committing transaction seals the catalog
2. Next commit must perform CAS to compact:
   - Write new checkpoint containing current state
   - Update checkpoint_offset to current log_offset
   - Reset entries_since_checkpoint to 0

## Table Metadata Inlining

Configurable via `TABLE_METADATA_INLINED` (default: `true`):

### Inlined (default)
- Table metadata is part of catalog/intention record
- No separate read/write needed for table metadata
- Catalog merge yields table state directly
- Lower latency, fewer I/O operations

### Not Inlined
- Table metadata stored separately from catalog
- Must read table metadata at transaction start (per-table I/O cost)
- Must write table metadata before commit (per-table I/O cost)
- Affects both CAS and append modes equally

```
Transaction with non-inlined metadata:
  1. Read catalog state
  2. Read table metadata for each accessed table  ← Additional I/O
  3. Execute transaction
  4. Write table metadata for each written table  ← Additional I/O
  5. Write manifest lists
  6. Commit (CAS or append)
```

## Manifest List Append Model (ML+)

### Real-World Behavior

In traditional Iceberg:
- Each table has a manifest list pointing to manifest files
- Commits rewrite the entire manifest list
- Concurrent commits to the same table conflict

With manifest list append (ML+):
- Entries are appended tentatively, tagged with transaction ID
- **Readers filter entries** based on whether the associated transaction committed
- Entry validity determined by **catalog commit outcome**, not ML append
- This is orthogonal to catalog mode (works with both CAS and append)
- When threshold exceeded, manifest list is sealed and must be rewritten

### Key Insight: Tentative Entries

The critical insight is that ML entries are **tentative** until the catalog transaction commits:

1. Transaction appends entry to ML (tagged with txn_id) → entry is tentative
2. Transaction attempts catalog commit
3. **On catalog success**: ML entry becomes permanent (readers include it)
4. **On catalog failure**:
   - Entry remains in ML but readers filter it out (txn not committed)
   - Conflict resolution determines if entry needs update

### Conflict Resolution by Mode

On catalog conflict, the ML mode determines what operations are required:

**Traditional (rewrite) mode:**

*False Conflict* (different partition, no data overlap):
- Must read committed snapshot's manifest list
- Must write NEW manifest list combining both transactions' manifest file pointers
- Update table metadata with new ML pointer
- Retry catalog commit

*Real Conflict* (same partition, overlapping data):
- Read manifest list
- Read/merge conflicting manifest files
- Write merged manifest files
- Write NEW manifest list
- Update table metadata
- Retry catalog commit

**ML+ (append) mode:**

*False Conflict* (different partition, no data overlap):
- ML entry is still valid (same data files, different partition)
- **No ML update needed** - readers filter tentative entries by committed txn list
- Just retry catalog commit with updated intention record
- ML entry becomes permanent when catalog commit succeeds

*Real Conflict* (same partition, overlapping data):
- Original ML entry is invalid (data files need merging)
- Read manifest list
- Read/merge conflicting manifest files
- Write merged manifest files
- Append NEW entry with merged data
- Original entry stays in ML but filtered by readers (txn not committed)
- Retry catalog commit

**Key advantage of ML+:** Under high contention with false conflicts (common in
multi-tenant workloads where transactions touch different partitions), ML+ avoids
the manifest list read/write per retry, saving ~100ms of I/O per conflict.

### Simulation Model

```
Per-table manifest list state (in Catalog/AppendCatalog):
  - ml_offset[tbl_id]: Current byte offset
  - ml_sealed[tbl_id]: Whether sealed (needs rewrite)

Per-transaction state:
  - v_ml_offset[tbl_id]: Expected offset when transaction started

Manifest list append (before catalog commit):
  1. For each written table:

     a. Check if sealed → must rewrite entire manifest list
        - Read current manifest list
        - Write new manifest list object
        - Unseal and reset offset

     b. Attempt physical append at expected offset:

        Physical Failure (offset moved OR sealed):
          - Update expected offset to current
          - Retry at new offset (or rewrite if sealed)

        Physical Success:
          - Entry written (tentative, tagged with txn_id)

  2. Proceed to catalog commit

After catalog logical failure (conflict resolution):
  - For false conflicts: ML entry still valid, no action
  - For real conflicts: Re-append with merged data
```

### Key Differences from Catalog Append

| Aspect | Catalog Append | Manifest List Append |
|--------|---------------|---------------------|
| Scope | Single catalog | Per-table |
| Entry validity | Immediate (via table version check) | Deferred (via catalog commit) |
| Cross-entity conflicts | Different tables don't conflict | N/A (single table) |
| Seal handling | CAS compaction | Rewrite to new object |
| Threshold config | `compaction_threshold` | `manifest_list_seal_threshold` |
| Reader filtering | By committed_txn set | By committed transactions |

Writers who append to a sealed manifest list discover this on physical failure and know their entry wasn't included - they must rewrite.

## Integration with Existing Simulation

### Configuration

```toml
[catalog]
mode = "append"                    # "cas" (default) or "append"
compaction_threshold = 16000000    # 16MB - seal after this many bytes
compaction_max_entries = 0         # 0 = disabled; >0 = seal after N entries
log_entry_size = 100               # Bytes per catalog log entry
table_metadata_inlined = true      # Whether table metadata is in catalog

[transaction]
manifest_list_mode = "append"      # "rewrite" (default) or "append"
manifest_list_seal_threshold = 0   # 0 = disabled; >0 = seal after N bytes
manifest_list_entry_size = 50      # Bytes per manifest list entry

[storage]
# Catalog append latencies
T_APPEND.mean = 50                 # Append operation latency
T_APPEND.stddev = 5
T_LOG_ENTRY_READ.mean = 5          # Per-entry log read (unused in current model)
T_LOG_ENTRY_READ.stddev = 1
T_COMPACTION.mean = 200            # Compaction CAS (larger payload)
T_COMPACTION.stddev = 20

# Table metadata latencies (when not inlined)
T_TABLE_METADATA.read.mean = 20
T_TABLE_METADATA.read.stddev = 5
T_TABLE_METADATA.write.mean = 30
T_TABLE_METADATA.write.stddev = 5
```

### Transaction Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          txn_gen()                                    │
├──────────────────────────────────────────────────────────────────────┤
│  1. Select tables (rand_tbl)                                         │
│  2. Capture catalog state:                                           │
│     - CAS mode: catalog.seq                                          │
│     - Append mode: catalog.log_offset                                │
│  3. Capture manifest list offsets (if manifest_list_mode = append)   │
│  4. Read table metadata (if table_metadata_inlined = false)          │
│  5. Execute transaction (sim.timeout for runtime)                    │
│  6. Write table metadata (if table_metadata_inlined = false)         │
│  7. Write manifest lists:                                            │
│     - rewrite mode: txn_ml_w()                                       │
│     - append mode: txn_ml_append()                                   │
│  8. Commit loop:                                                     │
│     - CAS mode: txn_commit()                                         │
│     - Append mode: txn_commit_append()                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Catalog Selection

```python
def setup(sim):
    if CATALOG_MODE == "append":
        catalog = AppendCatalog(sim)
    else:
        catalog = Catalog(sim)
```

Both `Catalog` and `AppendCatalog` track manifest list state (`ml_offset`, `ml_sealed`) for manifest list append mode.

### Statistics Tracking

New counters in `Stats` class:

```python
# Catalog append statistics
append_physical_success      # Append landed at expected offset
append_logical_success       # + table versions matched → commit
append_logical_conflict      # Append landed but table conflict
append_physical_failure      # Offset moved, retry needed
append_compactions_triggered # Sealed due to threshold
append_compactions_completed # Successful compaction CAS

# Manifest list append statistics (ML+ mode)
# Note: ML entries are tentative until catalog commits.
# Entry validity is determined by catalog commit outcome.
manifest_append_physical_success   # Physical append succeeded (entry tentative)
manifest_append_physical_failure   # Offset moved, retry needed
manifest_append_sealed_rewrite     # Manifest list was sealed, had to rewrite

# Existing conflict statistics track ML+ outcomes:
# - false_conflicts: ML entry still valid, no ML update needed
# - real_conflicts: ML entry needs update, re-appended after merge
```

## Example: Concurrent Non-Conflicting Transactions

**Setup:** Catalog with tables A, B (both at version 0)

**T1:** Update table A (v0 → v1)
**T2:** Update table B (v0 → v1)

### CAS Mode (rewrite manifest list)
```
T1: CAS succeeds → catalog.seq = 1
T2: CAS fails (seq changed)
T2: Resolve false conflict:
    - Read manifest list (to get T1's manifest file pointers)
    - Write NEW manifest list (combining T1's and T2's pointers)
    - Update table metadata
T2: CAS succeeds → catalog.seq = 2
Result: 1 CAS failure, 1 retry, 1 ML read + 1 ML write
```

### Append Mode
```
T1: Append at offset 0
    - Physical success (offset matched)
    - Re-read catalog
    - Logical success (table A at v0) → commit
    - offset now 100

T2: Append at offset 0 (stale)
    - Physical failure (offset moved to 100)
    - Retry at offset 100
    - Physical success
    - Re-read catalog
    - Logical success (table B still at v0) → commit
    - offset now 200

Result: 1 physical failure (cheap), 0 logical conflicts
Both commit without expensive conflict resolution
```

## Example: ML+ with Catalog Conflict

**Setup:** Table A at version 0, ML+ enabled

**T1:** Update partition P1 of table A
**T2:** Update partition P2 of table A (different partition)

```
T1: Append to ML (tentative, offset 0)
    - Physical success
T1: Append to catalog
    - Physical success
    - Logical success → commit
    - Table A now at v1

T2: Append to ML (tentative, offset 50)
    - Physical success (T1's entry doesn't block T2)
T2: Append to catalog
    - Physical success
    - Logical FAILURE (table A at v1, expected v0)
    - Re-read catalog to understand conflict

Conflict resolution:
    - Conflict is FALSE (different partitions, no data overlap)
    - T2's ML entry is still valid (same data files)
    - No ML re-append needed!
    - Just retry catalog with updated intention record

T2: Append to catalog (retry)
    - Physical success
    - Logical success (writing v2 based on v1) → commit

Result: 1 catalog logical failure, 0 ML re-appends
T2's original ML entry becomes permanent when T2 commits
```

**Contrast with real conflict:**

If T1 and T2 updated the same partition:
- Conflict is REAL (overlapping data)
- T2's ML entry is invalid (data files need merging)
- Must read manifest files, merge, re-append to ML
- Then retry catalog commit

## Limitations and Future Work

1. **Simplified log sizing**: Fixed entry size rather than actual serialization
2. **No partial log reads**: Model assumes full catalog re-read after physical success
3. **Compaction always succeeds**: No modeling of compaction contention
4. **No manifest content merging**: Version check only, no actual merge logic
5. **Manifest list seal always at write**: Real implementations might seal on read

See `docs/append_errata.md` for detailed technical debt documentation.
