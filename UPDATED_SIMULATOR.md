# Simulator Updates: Operation Types and Accurate Conflict Resolution

This document summarizes the changes made to the Endive simulator to model Iceberg's operation types and accurate conflict resolution behavior.

## Problem Statement

The previous simulator had several inaccuracies identified in SIMULATOR_REVIEW.md:

1. **Uniform operation treatment**: All transactions were treated identically, but Iceberg has distinct operation types with different conflict costs
2. **Real conflicts incorrectly merged**: The simulator merged manifest files on real conflicts; Iceberg throws `ValidationException` and **aborts**
3. **False conflict costs overstated**: FastAppend false conflicts cost ~160ms, not O(N) ML reads
4. **I/O convoy unrealistic for appends**: The O(N) historical ML reads only affect validated operations, not appends

## Solution: Operation Types

### Three Operation Types

| Type | Validation | I/O Convoy | Real Conflicts |
|------|------------|------------|----------------|
| `FAST_APPEND` | None | No | Cannot occur |
| `MERGE_APPEND` | None | No | Cannot occur |
| `VALIDATED_OVERWRITE` | Full | Yes (O(N) ML reads) | Abort with ValidationException |

### Conflict Cost Models

**FastAppend** (~160ms retry):
- 1 metadata read
- 1 ML read (current snapshot only)
- 1 ML write (rewrite mode) or 0 (ML+ mode)
- 0 historical ML reads (NO I/O convoy)
- 0 manifest file operations

**MergeAppend** (~160ms + manifest I/O):
- Same as FastAppend, plus:
- K manifest file reads/writes (K = n_behind × manifests_per_commit)

**ValidatedOverwrite** (O(N) cost):
- 1 metadata read
- 1 ML read
- N historical ML reads (THE I/O CONVOY)
- 1 ML write
- If real conflict detected: **ABORT** (not merge-and-retry)

## Configuration

```toml
[transaction]
# Operation type distribution (must sum to 1.0)
operation_types.fast_append = 0.7
operation_types.merge_append = 0.2
operation_types.validated_overwrite = 0.1

# Probability of real conflict on validated operations
real_conflict_probability = 0.3

# For MergeAppend: manifests to re-merge per concurrent commit
manifests_per_concurrent_commit = 1.0
```

## New Files

| File | Purpose |
|------|---------|
| `endive/operation.py` | OperationType enum, ConflictCost, OperationBehavior classes |
| `endive/conflict.py` | resolve_conflict(), resolve_partition_conflict(), ConflictResolverV2 |
| `tests/test_operation_types.py` | 35 tests (operation types + partition-aware conflicts) |

## Modified Files

| File | Changes |
|------|---------|
| `endive/transaction.py` | Added `operation_type`, `abort_reason` fields; `get_behavior()` method |
| `endive/main.py` | Config loading, `_sample_operation_type()`, operation-aware commit |
| `endive/capstats.py` | `validation_exceptions` counter, `operation_type`/`abort_reason` in records |
| `endive/config.py` | Validation for operation_types config section |

## Backward Compatibility

**Default behavior unchanged**: When no `operation_types` section is configured (or 100% fast_append), the simulator uses legacy conflict resolution. This ensures:

- Existing configs produce identical results
- All existing tests pass (438 tests, 0 failures)
- No changes required to existing experiment configurations

## Key Behavioral Changes

### Before (Legacy)
- All conflicts paid O(N) ML read cost
- Real conflicts merged manifest files and retried
- `real_conflict_probability` applied to all transactions

### After (With Operation Types)
- FastAppend/MergeAppend: No O(N) cost, no real conflicts possible
- ValidatedOverwrite: O(N) cost, real conflicts **abort** with ValidationException
- `real_conflict_probability` only applies to ValidatedOverwrite transactions

## Partition-Aware Conflict Resolution

When partition mode is enabled (`partition.enabled = true`) with operation types, conflicts are resolved at the partition level:

### Key Insight

**Real conflicts are discovered AFTER reading the ML for that partition**. The ML read cost is paid before we know if it's a real conflict.

### Conflict Resolution Flow

1. **Identify overlapping partitions**: Partitions where version changed since transaction read
2. **For each overlapping partition**:
   - Pay ML read cost (discovering the conflict state)
   - For ValidatedOverwrite: Roll dice for data overlap per partition
     - If data overlap (real conflict): **ABORT** (ML read cost already paid)
     - If no data overlap (false conflict): Pay ML write cost (merge)
   - For FastAppend/MergeAppend: Always pay merge cost (no validation)

### Configuration

```toml
[transaction]
# Probability of data overlap within an overlapping partition
# Defaults to real_conflict_probability if not specified
data_overlap_probability = 0.3

[partition]
enabled = true
num_partitions = 10
```

### Cost Model

For a transaction with P overlapping partitions:

| Operation Type | ML Reads | ML Writes | Abort Possible |
|----------------|----------|-----------|----------------|
| FastAppend | P | P (rewrite) or 0 (ML+) | No |
| MergeAppend | P | P (rewrite) or 0 (ML+) | No |
| ValidatedOverwrite | P (before abort) | P (if no abort) | Yes |

### New Functions

- `resolve_partition_conflict()` in `conflict.py`: Main entry point
- `_compute_overlapping_partitions()`: Identifies partitions with version changes
- `_read_partition_manifest_lists()`: Pays ML read cost per partition
- `_resolve_partition_false_conflict()`: Handles merge for non-conflicting partitions

## Statistics Tracking

New statistics added:
- `validation_exceptions`: Count of aborts due to ValidationException
- Per-transaction `operation_type`: Recorded in parquet output
- Per-transaction `abort_reason`: "validation_exception" or "max_retries"

## Example Output

With mixed operation types under contention:
```
Operation type distribution:
  fast_append            45%
  merge_append           34%
  validated_overwrite    21%

Abort reasons:
  max_retries             73%
  validation_exception    27%

Stats:
  False conflicts: 501
  Real conflicts: 14
  Validation exceptions: 14
```

## References

- SIMULATOR_REVIEW.md: Analysis of simulator inaccuracies
- Iceberg source: SnapshotProducer.java, MergingSnapshotProducer.java
