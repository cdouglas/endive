# Consolidated Experiment Results Format

## Overview

Starting with version 2.0, experiment results are stored in a single consolidated Parquet file (`experiments/consolidated.parquet`) in addition to the original per-seed files. This format provides:

- **Faster analysis**: Predicate pushdown filters data at storage layer
- **Lower memory usage**: Only loads relevant experiments, not entire dataset
- **Simpler access**: Single file instead of 1800+ individual files
- **Schema consistency**: Normalized data types across all experiments

## File Structure

```
experiments/
├── consolidated.parquet          # Single consolidated file (1.39 GB)
│                                 # Contains ALL 371 experiments, 114M rows
├── exp2_1_single_table_false-hash/
│   ├── cfg.toml                  # Configuration (preserved)
│   └── seed/
│       └── results.parquet       # Original per-seed results (preserved)
└── exp2_2_multi_table_false-hash/
    ├── cfg.toml
    └── seed/
        └── results.parquet
```

**Important**: Original files are **preserved** for safety and backward compatibility.

## Schema

The consolidated format adds metadata columns for efficient filtering:

| Column | Type | Description |
|--------|------|-------------|
| `exp_name` | string | Experiment label (e.g., "exp2_1_single_table_false") |
| `exp_hash` | string | Unique hash for parameter combination |
| `seed` | int64 | Random seed for this run |
| `config` | map<string, string> | Flattened experiment configuration |
| `txn_id` | int64 | Transaction ID |
| `t_submit` | int64 | Submission timestamp (ms) |
| `t_commit` | int64 | Commit timestamp (ms) |
| `commit_latency` | int64 | Commit latency (ms) |
| `total_latency` | int64 | Total latency (ms) |
| `status` | string | "committed" or "aborted" |
| `n_retries` | int8 | Number of retry attempts |
| `n_tables_read` | int8 | Tables read |
| `n_tables_written` | int8 | Tables written |
| `t_runtime` | int64 | Transaction runtime (ms) |

### Schema Changes from Original

1. **Time columns**: float64 → int64 (milliseconds, lossless rounding)
2. **Count columns**: int64 → int8 (sufficient range for retry/table counts)
3. **Seed column**: string → int64 (more efficient, enables numeric filtering)
4. **Added columns**: exp_name, exp_hash, config (for filtering and context)

## Sorting and Row Groups

**Sort order**: `(exp_name, exp_hash, seed, t_submit)`

This ordering enables:
- Efficient filtering by experiment
- Sequential access within experiments
- Natural time-series order within each seed

**Row group structure**: One row group per seed (~10 MB each)
- Enables efficient predicate pushdown
- Skips irrelevant experiments at storage layer
- Maintains seed-level isolation for parallel processing

## Compression

**Format**: Parquet with ZSTD compression level 3

**Compression ratio**: ~64% reduction
- Original: ~3.5 GB across all results.parquet files
- Consolidated: 1.39 GB
- Overhead: Metadata columns add <5%

## Usage

### Python API

```python
import pandas as pd
from endive import saturation_analysis

# Automatic: Uses consolidated file by default
saturation_analysis.CONFIG = saturation_analysis.get_default_config()
df = saturation_analysis.load_and_aggregate_results_consolidated(exp_info)

# Manual: Direct filtering
df = pd.read_parquet(
    'experiments/consolidated.parquet',
    filters=[
        ('exp_name', '==', 'exp2_1_single_table_false'),
        ('exp_hash', '==', '7fe68106'),
        ('t_submit', '>=', 900000),    # After 15-min warmup
        ('t_submit', '<', 2700000)     # Before cooldown
    ]
)
```

### Configuration

Control consolidated file usage in `analysis.toml`:

```toml
[paths]
consolidated_file = "experiments/consolidated.parquet"

[analysis]
use_consolidated = true  # Default: true
```

Override via CLI:

```bash
# Force using consolidated file
python -m endive.saturation_analysis --use-consolidated

# Force using original files (fallback)
python -m endive.saturation_analysis --no-use-consolidated
```

## Migration and Compatibility

### Automatic Fallback

Analysis code automatically falls back to original files if:
- Consolidated file doesn't exist
- Consolidated file is corrupted
- `use_consolidated = false` in config

### Verification

Verify consolidated file matches original data:

```bash
# Test 20 random experiments
python scripts/test_consolidated_analysis.py --sample 20

# Test specific experiment pattern
python scripts/test_consolidated_analysis.py --pattern "exp2_1*"

# Verify file integrity (efficient predicate pushdown)
python scripts/verify_consolidation_efficient.py --sample 20
```

### Regeneration

Recreate consolidated file from scratch:

```bash
# Incremental writing (low memory, ~22 minutes)
python scripts/consolidate_all_experiments_incremental.py

# Output: experiments/consolidated.parquet (1.39 GB)
# Memory: <2 GB peak
```

## Performance Benefits

### Analysis Speed

**Before** (individual files):
```python
# Loads 1851 separate files, ~3.5 GB total
df = load_and_aggregate_results(exp_info)
# Time: ~45 seconds for one experiment
# Memory: ~8 GB peak (loads everything)
```

**After** (consolidated with predicate pushdown):
```python
# Loads only relevant rows using filters
df = load_and_aggregate_results_consolidated(exp_info)
# Memory: ~200 MB peak (only loads filtered data via predicate pushdown)
```

### Warmup/Cooldown Filtering

Predicate pushdown on `t_submit` avoids loading transient data:

```python
# Before: Load all 114M rows, filter in memory (OOM risk)
df = pd.read_parquet('experiments/consolidated.parquet')
df = df[(df['t_submit'] >= warmup) & (df['t_submit'] < cooldown)]

# After: Filter at storage layer (efficient)
df = pd.read_parquet(
    'experiments/consolidated.parquet',
    filters=[
        ('exp_name', '==', exp_name),
        ('exp_hash', '==', exp_hash),
        ('t_submit', '>=', warmup),      # Skip warmup at storage
        ('t_submit', '<', cooldown)      # Skip cooldown at storage
    ]
)
```

## Size and Statistics

**File**: `experiments/consolidated.parquet`
- **Size**: 1.39 GB (ZSTD level 3)
- **Rows**: 114,142,130 transactions
- **Experiments**: 371 unique configurations
- **Seeds**: 1,851 total runs (3-5 seeds per experiment)
- **Time range**: 1 hour per seed (3600 seconds)

**Per-experiment statistics**:
- Average: ~308,000 rows per experiment
- Range: 1,730 to 2.6M rows (varies with load)
- Typical size: ~400 KB per seed after filtering

## File Format Details

**Parquet version**: 2.6
**Writer**: PyArrow 15.0.0
**Schema evolution**: Forward compatible (new columns can be added)

**Metadata**:
```python
import pyarrow.parquet as pq
meta = pq.read_metadata('experiments/consolidated.parquet')

print(f"Num row groups: {meta.num_row_groups}")    # 1851 (one per seed)
print(f"Num rows: {meta.num_rows}")                # 114,142,130
print(f"Serialized size: {meta.serialized_size}")  # ~1.39 GB
```

## Best Practices

1. **Always use predicate filters**: Never load entire file into memory
2. **Filter early**: Apply exp_name, exp_hash, t_submit filters at read time
3. **Preserve originals**: Don't delete original results.parquet files
4. **Verify after regeneration**: Run verification scripts before using new consolidated file
5. **Monitor memory**: Even with filters, large experiments can use significant memory

## Troubleshooting

### Out of Memory

```bash
# Reduce sample size in analysis
python -m endive.saturation_analysis --pattern "exp2_1*" --min-seeds 3

# Or use original files (slower but lower memory)
python -m endive.saturation_analysis --no-use-consolidated
```

### Corrupted Consolidated File

```bash
# Verify integrity
python scripts/verify_consolidation_efficient.py --sample 10

# If corrupted, regenerate
python scripts/consolidate_all_experiments_incremental.py
```

### Mismatched Results

```bash
# Compare both methods on same experiment
python scripts/test_consolidated_analysis.py --pattern "exp2_1_*" --sample 5

# Check for differences in commit rates or latencies
```

## Future Enhancements

Potential improvements to consolidated format:

1. **Partitioning**: Partition by exp_name for even faster filtering
2. **Columnar stats**: Pre-compute statistics per row group
3. **Compression tuning**: Test ZSTD levels 1-9 for size/speed tradeoff
4. **Delta updates**: Incrementally append new experiments
5. **Deduplication**: Remove any accidental duplicate rows

## References

- **Implementation**: `scripts/consolidate_all_experiments_incremental.py`
- **Analysis code**: `endive/saturation_analysis.py:load_and_aggregate_results_consolidated()`
- **Verification**: `scripts/verify_consolidation_efficient.py`
- **Testing**: `scripts/test_consolidated_analysis.py`
- **Parquet docs**: https://parquet.apache.org/docs/
- **PyArrow filtering**: https://arrow.apache.org/docs/python/parquet.html#filtering
