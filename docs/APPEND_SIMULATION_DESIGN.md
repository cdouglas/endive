# Append Simulation Design

This document describes how the Endive simulator models append-based operations for Apache Iceberg catalogs, specifically the `LogCatalogFormat` approach where transactions append to a catalog log rather than performing compare-and-swap (CAS) on the entire catalog state.

## Overview

The append model introduces two orthogonal optimizations:

1. **Catalog Log Append**: Instead of CAS on catalog state, transactions append log entries containing their table mutations. Conflicts are detected at the table level, not catalog level.

2. **Manifest List Append**: Instead of rewriting manifest lists, transactions append entries at an expected table version.

Both can be enabled independently via configuration.

## Catalog Log Append Model

### Real-World Behavior (LogCatalogFormat)

In Iceberg's `LogCatalogFormat`:
- The catalog is stored as a checkpoint plus a log of transactions
- Each transaction appends a `TRANSACTION` record containing its actions (table updates, reads, creates)
- On append, the transaction is verified against the current catalog state
- If verification fails (table version mismatch), the transaction fails
- When the log exceeds a threshold (16MB), the next commit seals the log and triggers compaction via CAS

### Simulation Model

```
┌─────────────────────────────────────────────────────────────┐
│                      AppendCatalog                          │
├─────────────────────────────────────────────────────────────┤
│  seq: int              # Compatibility with Txn class       │
│  tbl: list[int]        # Per-table versions                 │
│  log_offset: int       # Current log byte position          │
│  log_entries: list     # In-memory log (simulation only)    │
│  checkpoint_offset: int                                     │
│  sealed: bool          # True if compaction needed          │
│  committed_txn: set    # For deduplication                  │
└─────────────────────────────────────────────────────────────┘
```

**Key simplifications:**
- Log entries are fixed-size (`LOG_ENTRY_SIZE`, default 100 bytes) rather than variable
- Log is stored in-memory as a list rather than serialized bytes
- Transaction IDs are integers rather than UUIDs

### Commit Protocol

The key insight is that **physical append success ≠ transaction success**. After a successful
physical append, the transaction must re-read the entire catalog and verify its entry was
accepted.

```
Transaction starts:
  1. Capture catalog.log_offset as v_log_offset
  2. Capture table versions in v_tblr, v_tblw

Transaction commits (txn_commit_append):
  1. If catalog.sealed → perform compaction CAS first
  2. Create LogEntry with tables_written, tables_read
  3. Attempt physical append at expected offset:

     Physical Failure (offset moved):
       - Re-read log entries since our snapshot
       - Update v_log_offset to current
       - Retry physical append at new offset

     Physical Success (offset matches):
       - Entry appended to log
       - If log exceeds threshold → seal for compaction
       - Proceed to verification (step 4)

  4. Re-read entire catalog (checkpoint + log)
  5. Verify all log entries in order:
     - For each entry, check table versions match expected
     - Verified entries update table state and join committed set
     - Conflicting entries are rejected (not in committed set)

  6. Check if our transaction ID is in committed set:

     Logical Success (in committed set):
       - Transaction committed successfully
       - Table state already updated during verification

     Logical Failure (not in committed set):
       - Another entry conflicted with ours
       - Must do full retry (re-read table metadata, update manifest)
```

### Verification Logic

During verification, each log entry is checked against the evolving catalog state:

```python
def _verify_entry(entry, current_tbl):
    # Check all written tables have expected versions
    for tbl_id, new_ver in entry.tables_written.items():
        expected_ver = new_ver - 1  # Writing v+1 expects current to be v
        if current_tbl[tbl_id] != expected_ver:
            return False  # Conflict!

    # Check all read tables haven't changed
    for tbl_id, read_ver in entry.tables_read.items():
        if tbl_id not in entry.tables_written:
            if current_tbl[tbl_id] != read_ver:
                return False  # Conflict!

    return True  # Entry verified - will be applied
```

This allows concurrent transactions writing to **different tables** to both physically append
and both pass verification, unlike CAS mode where any concurrent commit causes failure.

This allows concurrent transactions writing to **different tables** to both succeed, unlike CAS mode where any concurrent commit causes failure.

### Compaction

Compaction is triggered when **either** condition is met (minimum of the two):
1. **Size threshold**: `log_offset - checkpoint_offset > COMPACTION_THRESHOLD`
2. **Entry count threshold**: `entries_since_checkpoint >= COMPACTION_MAX_ENTRIES` (if > 0)

When triggered:
1. The committing transaction is marked as `sealed`
2. Next commit must perform CAS to compact:
   - Write new checkpoint containing current state
   - Clear log entries
   - Reset checkpoint_offset and entries_since_checkpoint

## Manifest List Append Model

### Real-World Behavior

In traditional Iceberg:
- Each table has a manifest list pointing to manifest files
- Commits rewrite the entire manifest list
- Concurrent commits to the same table conflict

With manifest list append:
- New manifest entries are appended at an expected table version
- If version matches, append succeeds
- If version changed, must read new state and potentially merge

### Simulation Model

```python
def txn_ml_append(sim, txn, catalog):
    for tbl_id in txn.v_tblw:
        expected_ver = txn.v_dirty[tbl_id]
        current_ver = catalog.tbl[tbl_id]

        if current_ver == expected_ver:
            # Version matches - append succeeds
            yield sim.timeout(manifest_list_write_latency)
        else:
            # Version mismatch - read and retry
            yield sim.timeout(manifest_list_read_latency)
            txn.v_dirty[tbl_id] = current_ver
            yield sim.timeout(manifest_list_write_latency)
```

**Simplifications:**
- No actual manifest content is modeled
- Version check is instantaneous (no I/O for version lookup)
- Retry always succeeds after reading new state

## Integration with Existing Simulation

### Configuration

```toml
[catalog]
mode = "append"           # "cas" (default) or "append"
compaction_threshold = 16000000  # 16MB (size-based)
compaction_max_entries = 0       # 0 = disabled; >0 = compact after N entries
log_entry_size = 100

[transaction]
manifest_list_mode = "append"  # "rewrite" (default) or "append"

[storage]
T_APPEND.mean = 50        # Append operation latency
T_APPEND.stddev = 5
T_LOG_ENTRY_READ.mean = 5 # Per-entry log read
T_LOG_ENTRY_READ.stddev = 1
T_COMPACTION.mean = 200   # Compaction CAS (larger payload)
T_COMPACTION.stddev = 20
```

### Transaction Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      txn_gen()                               │
├──────────────────────────────────────────────────────────────┤
│  1. Select tables (rand_tbl)                                 │
│  2. Capture catalog state:                                   │
│     - CAS mode: catalog.seq                                  │
│     - Append mode: catalog.log_offset                        │
│  3. Execute transaction (sim.timeout)                        │
│  4. Write manifest lists:                                    │
│     - rewrite mode: txn_ml_w()                              │
│     - append mode: txn_ml_append()                          │
│  5. Commit loop:                                             │
│     - CAS mode: txn_commit()                                │
│     - Append mode: txn_commit_append()                      │
└──────────────────────────────────────────────────────────────┘
```

### Catalog Selection

```python
def setup(sim):
    if CATALOG_MODE == "append":
        catalog = AppendCatalog(sim)
    else:
        catalog = Catalog(sim)
```

### Statistics Tracking

New counters in `Stats` class:

```python
# Catalog append statistics
append_physical_success    # Append landed at expected offset
append_logical_success     # + verification passed
append_logical_conflict    # Append landed but conflict detected
append_physical_failure    # Offset moved
append_compactions_triggered
append_compactions_completed

# Manifest list append statistics
manifest_append_success    # Append at correct version
manifest_append_retry      # Version mismatch, had to retry
```

## Example: Concurrent Non-Conflicting Transactions

**Setup:** Catalog with tables A, B (both at version 0)

**T1:** Update table A (v0 → v1)
**T2:** Update table B (v0 → v1)

### CAS Mode (existing)
```
T1: CAS succeeds → catalog.seq = 1
T2: CAS fails (seq changed)
T2: Read manifest list, resolve false conflict
T2: CAS succeeds → catalog.seq = 2
Result: 1 retry
```

### Append Mode (new)
```
T1: Append at offset 0 → log = [T1], offset = 100
T2: Append at offset 100 → log = [T1, T2], offset = 200
    (T2 reads T1's entry, checks: T1 wrote A, T2 writes B → no conflict)
Result: 0 retries, both commit on first attempt
```

## Limitations and Future Work

1. **Simplified log sizing**: Fixed entry size rather than actual serialization
2. **No partial log reads**: Re-reads all entries since snapshot
3. **Compaction always succeeds**: No modeling of compaction contention
4. **No manifest content merging**: Version check only, no actual merge logic

See `docs/append_errata.md` for detailed technical debt documentation.
