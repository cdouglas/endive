# Append Operations Implementation - Errata and Technical Debt

This document tracks shortcuts, deferred tasks, and technical debt incurred during the implementation of append-based operations for the Endive simulator.

## Phase 1: Core Append Model (Catalog)

### Shortcuts Taken

1. **Simplified Log Entry Size**: Log entry size is modeled as a fixed constant (`LOG_ENTRY_SIZE`) rather than calculating actual byte sizes based on transaction content. This is acceptable because:
   - The simulator doesn't serialize actual data
   - Compaction threshold logic only needs relative sizes
   - Real implementations would have variable entry sizes based on table count and action types

2. **In-Memory Log Storage**: The `log_entries` list grows unbounded during simulation (cleared only on compaction). For long simulations with high throughput, this could consume significant memory.
   - **Mitigation**: Compaction clears entries when threshold is reached
   - **Future work**: Could implement sliding window or summary statistics instead of full log

3. **UUID Generation**: Using simple incrementing integers for transaction UUIDs instead of actual UUIDv7. This is sufficient for deduplication logic testing.

### Deferred Tasks

1. **Transaction UUID in Parquet Export**: The `txn_uuid` field is not yet exported to parquet files. Will need to add this column when implementing deduplication analysis.

2. **Log Entry Reads Per Batch**: Currently reads all new log entries in a single I/O operation. Real implementations may batch reads based on entry count or byte size.

3. **Compaction Contention Modeling**: When multiple transactions trigger compaction simultaneously, only one succeeds and others retry. The retry logic is simplified - real implementations may have more sophisticated leader election.

### Known Limitations

1. **No Partial Log Reads**: After append failure, we re-read all entries since our snapshot. Real implementations might read only the new entries incrementally.

2. **Compaction Always Succeeds Eventually**: The model assumes compaction eventually succeeds. Real implementations may have compaction failures that require external intervention.

---

## Phase 2: Manifest List Append

### Shortcuts Taken

1. **Simplified Retry Logic**: When manifest append fails due to version mismatch, we immediately read and retry. Real implementations might queue the retry or apply backoff.

2. **No Manifest Content Modeling**: The manifest list append doesn't model actual manifest content - it only checks table versions. Real implementations would need to merge manifest entries.

3. **Version Mismatch Handling**: On version mismatch, we update the dirty version and retry. This doesn't model the complexity of detecting whether the version change is compatible (e.g., non-overlapping data files).

### Deferred Tasks

1. **Manifest Content Merge**: When manifest append fails, real implementations may need to merge content from the new version. This is deferred as it requires more complex manifest modeling.

2. **Per-Table Append Statistics**: Currently we track global manifest_append_success/retry counts. Per-table breakdowns could be useful for analysis.

---

## Phase 3: Combined Model

Phase 3 was largely completed during Phases 1 and 2 implementation:
- Catalog append and manifest list append modes work together
- Transaction flow handles both modes via configuration
- Tested in `test_manifest_list_append_with_catalog_append`

### Notes

- The combined model enables maximum benefit: both catalog-level and table-level append operations
- Statistics are tracked separately for catalog append vs manifest list append
- No new conflict categories were added (tracked via existing stats)

---

## Phase 4: Statistics and Analysis

### Experiment Configurations Created

- `exp6_1_append_vs_cas_baseline.toml`: Compare append vs CAS modes
- `exp6_2_multi_table_append.toml`: Multi-table scaling with append mode
- `exp6_3_compaction_threshold.toml`: Compaction threshold sensitivity
- `exp6_4_manifest_list_append.toml`: Manifest list append impact

### Deferred Tasks

1. **Analysis Pipeline Updates**: The saturation_analysis.py may need updates to extract append-specific parameters (mode, compaction_threshold, manifest_list_mode) for grouping and filtering.

2. **Composite Plots for Append Experiments**: Similar to existing exp3.x composite plots, new groupings may be needed for:
   - `--group-by catalog_mode` for append vs CAS comparison
   - `--group-by compaction_threshold` for threshold sensitivity

3. **Append Statistics Export**: The append-specific statistics (append_physical_success, etc.) are not currently exported to parquet files. This would require extending the transaction record or adding a separate stats export.

---

## General Notes

### Code Quality Trade-offs

- **Global State Pattern**: Continues using the existing global configuration pattern from main.py for consistency, even though a config object pattern would be cleaner.

- **Dual Catalog Classes**: `Catalog` and `AppendCatalog` share no common base class. This is intentional to keep the changes minimal and avoid modifying working code.

### Testing Gaps

- Integration tests comparing CAS vs Append mode under identical workloads are deferred to Phase 4.
- Edge cases around compaction during high-contention scenarios need more testing.

### Performance Considerations

- No performance profiling has been done yet on the append code path
- The log entry list operations (append, slice) should be O(1) amortized but may have memory allocation overhead

---

*Last updated: Phase 1 implementation*
