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

## Phase 2: Manifest List Append (ML+)

### Protocol Clarification

The ML+ protocol was revised to correctly model tentative entries:

1. **Tentative Entries**: ML entries are written before catalog commit, tagged with txn_id
2. **Reader Filtering**: Readers filter entries based on committed transactions
3. **Deferred Validation**: Entry validity determined by catalog commit, not ML append
4. **Conflict-Aware Updates**: On catalog conflict:
   - False conflict (different partition): ML entry still valid, no re-append
   - Real conflict (same partition): ML entry needs update, re-append after merge

### Shortcuts Taken

1. **No Per-Entry Transaction Tracking**: The simulation doesn't actually store ML entries or track which txn_id wrote each entry. The filtering behavior is implicit in the protocol.

2. **No Manifest Content Modeling**: The simulation uses `REAL_CONFLICT_PROBABILITY` to determine if a conflict requires ML re-append, rather than modeling actual manifest content.

3. **Simplified Sealing**: Sealing is triggered purely by offset threshold, not by entry count or other factors.

### Deferred Tasks

1. **Per-Table Append Statistics**: Currently we track global manifest_append_physical_success/failure counts. Per-table breakdowns could be useful for analysis.

2. **Concurrent ML Append Contention**: Multiple transactions appending to the same table's ML may have contention patterns worth modeling.

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

## Protocol Simplification: Validation at Append Time

### Model Refinement

The implementation was simplified based on the insight that the simulation does not need
to store log entries. The catalog "knows" whether intention records will be applied because
validation happens at append time (same as CAS, but at table-level).

### Key Insights

1. **Physical conflict rate bounds logical conflict rate**: Physical failures (offset moved)
   are cheap - just retry at new offset (returned by failed append). Logical failures
   (table version mismatch) require reading manifest lists to repair.

2. **No log storage needed**: The simulation tracks:
   - `log_offset` for physical conflict detection
   - `tbl` for table-level validation
   - `committed_txn` for deduplication

3. **Inlined table metadata**: The intention record contains table metadata, so catalog
   merge yields table state directly without separate storage trip.

4. **Commuting transactions**: If tables commute (no overlapping data files), no need
   to re-append to manifest list after logical failure.

### Changes Made

- `try_APPEND()` now returns `(physical_success, logical_success)` tuple
- Validation happens inside `try_APPEND()`, same as CAS but per-table
- Removed `log_entries` list, `read_and_verify()`, `_verify_entry()` methods
- Physical failure: Just retry at new offset (no I/O)
- Logical failure: Read manifest list, resolve conflicts (same as CAS conflict resolution)

---

## ML+ Protocol Refinement: Tentative Entries

### Key Insight

ML entries are **tentative** until the associated catalog transaction commits. This is
critical for understanding conflict resolution with ML+.

### Protocol Flow

1. Transaction appends to ML (entry is tentative, tagged with txn_id)
2. Transaction attempts catalog commit
3. On catalog success: ML entry becomes permanent
4. On catalog failure:
   - False conflict (different partition): ML entry still valid, just retry catalog
   - Real conflict (same partition): ML entry needs update, re-append after merge

### Changes Made

- `try_ML_APPEND()` now does physical validation only (returns `bool`)
- Removed logical validation from ML append (no table version check)
- ML entry validity determined by catalog commit outcome
- `resolve_real_conflict()` now re-appends to ML in ML+ mode
- `resolve_false_conflict()` does NOT re-append (entry still valid)
- Conflict type (false vs real) determined by `REAL_CONFLICT_PROBABILITY`

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

## Realistic Latency Modeling (Phase 1)

### Implementation Notes

1. **Lognormal distribution added**: New `generate_latency_lognormal()` function supports measured cloud latency distributions.

2. **Dual format support**: Configuration parsing handles both:
   - Legacy: `mean/stddev` (normal distribution)
   - New: `median/sigma` (lognormal distribution)

3. **Provider profiles defined**: AWS, Azure, GCP, and "instant" profiles with measured parameters.

### Technical Debt

1. **Provider profiles not yet applied automatically**: The `PROVIDER_PROFILES` dict is defined but not used in config loading. Phase 2 will add `storage.provider` config to apply profiles.

2. **No failure latency multipliers yet**: The `failure_multiplier` field is parsed but not used in latency generation. Phase 4 will implement this.

3. **No contention scaling yet**: The `contention_scaling` field is defined in profiles but not implemented. Phase 4 will add `ContentionTracker`.

4. **Lognormal conversion approximation**: The `convert_mean_stddev_to_lognormal()` uses an approximation that may not perfectly preserve the mean. This is acceptable for backward compatibility but noted as a known limitation.

### Deferred to Later Phases

- Phase 2: Storage/catalog separation, provider config application
- Phase 3: Configuration precedence rules
- Phase 4: Failure multipliers, contention tracking
- Phase 5: Full validation test suite

---

## Storage/Catalog Separation (Phase 2)

### Implementation Notes

1. **Storage provider config**: `storage.provider = "aws"` applies AWS profile to all storage operations (manifests).

2. **Catalog backend config**: `catalog.backend` determines where catalog ops run:
   - `"storage"` (default): CAS/append use storage provider latencies
   - `"service"`: CAS/append use `[catalog.service]` config
   - `"fifo_queue"`: Append uses queue config, checkpoints use storage

3. **Precedence implemented**: Provider defaults < explicit T_* config (Phase 3 scope reduced)

### Technical Debt

1. **Contention scaling not applied yet**: Provider profiles define `contention_scaling` but it's not used. Phase 4 will implement this.

2. **Failure multiplier not applied yet**: Provider profiles define `failure_multiplier` but latency functions don't use it. Phase 4 will implement this.

3. **No provider validation**: Invalid provider names silently fall back to defaults. Could add warning/error.

4. **Checkpoint latency in FIFO mode**: Uses T_CAS for compaction but doesn't distinguish checkpoint vs regular CAS.

### Deferred to Later Phases

- Phase 4: Failure multipliers, contention tracking
- Phase 5: Full validation test suite with throughput bounds

---

## Failure Latency and Contention Scaling (Phase 4)

### Implementation Notes

1. **Failure latency multiplier**: `get_cas_latency(success=False)` applies `failure_multiplier` from config. Based on measurements:
   - AWS CAS: 1.17x (failures slightly slower)
   - Azure append: 34.3x (failures dramatically slower!)

2. **ContentionTracker class**: Tracks concurrent CAS/append operations. Applies linear scaling from 1.0 at 1 concurrent to provider's `contention_scaling` at 16 concurrent.

3. **Auto-enable with provider**: Contention scaling auto-enables when `storage.provider` is set. Can be explicitly disabled with `contention_scaling_enabled = false`.

### Technical Debt

1. **Contention tracking not integrated with simulation**: The `CONTENTION_TRACKER.enter_cas()` / `exit_cas()` calls are not yet added to the commit paths. The tracking infrastructure is in place but the actual counting of concurrent operations requires integration with the simulation loop.

2. **Failure multiplier not applied in simulation**: The `success` parameter to `get_cas_latency()` / `get_append_latency()` is available but the commit paths don't yet pass the correct value based on CAS outcome.

3. **No per-thread scaling data**: Contention scaling uses aggregate 16-thread vs 1-thread ratio. The per-thread measurements in simulation_summary.md could enable finer-grained modeling.

### Deferred

- Integration of contention tracking with commit_txn() and append commit paths
- Passing success/failure state to latency functions based on CAS outcome

---

*Last updated: Phase 4 - Failure latency and contention scaling*
