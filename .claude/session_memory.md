# Session Memory: Analysis Configuration and Visualization

This file contains key insights and decisions from development sessions to help future sessions be more efficient.

## Key Design Decisions

### Standard Deviation Implementation

**Decision**: Compute per-seed statistics first, then aggregate with mean/stddev
**Rationale**: Allows both individual seed analysis and multi-seed aggregation
**Implementation**:
- `compute_per_seed_statistics()`: Returns DataFrame with one row per seed
- `compute_aggregate_statistics()`: Computes mean/stddev across seed DataFrame
- Backward compatible: Falls back to direct aggregation if no seed column

**Key Code Location**: `endive/saturation_analysis.py:372-520`

### Mean Retries Calculation

**Decision**: Compute mean_retries over committed transactions only
**Rationale**: Aborted transactions hit retry limit; including them would skew metric
**Code**: `committed['n_retries'].mean()` at line 488, 513
**Implication**: Metric answers "How many retries did successful transactions need?"

### Table-Level vs Catalog-Level Conflicts

**Decision**: When `num_groups == num_tables`, use table-level conflict detection
**Code**: `endive/main.py:789-804`
**Behavior**:
- Table-level: Only check versions of tables transaction touched
- Catalog-level: Check global catalog sequence number
- **Implication**: Multi-table experiments get true isolation between different tables

### Conflict Resolution Cost

**Decision**: Only resolve conflicts for tables with version mismatches
**Code**: `endive/main.py:676-686`
**Behavior**:
- Iterates through `txn.v_dirty` (all tables touched)
- Only processes tables where `v_catalog[t] != v`
- **Implication**: 10-table transaction with 2 conflicts only pays for resolving 2 tables

### Minimum Seed Filtering

**Decision**: Skip experiments with < min_seeds during index building
**Default**: 3 seeds minimum
**Rationale**: Ensures reliable standard deviation estimates
**Code**: `endive/saturation_analysis.py:557-562`

## Configuration Structure

### Three-Level Hierarchy
1. **Defaults** in `get_default_config()` (saturation_analysis.py:28)
2. **Config file** `analysis.toml` (optional, auto-discovered)
3. **CLI arguments** (highest priority)

### Deep Merge Pattern
Partial configs merge with defaults recursively:
```python
# User provides only: [plots.saturation] enabled = false
# System merges with all other defaults
```

### Key Config Sections
- `[paths]`: I/O locations and patterns
- `[analysis]`: Statistics and filtering (min_seeds, warmup, group_by)
- `[plots]`: Visual styling (figsize, fonts, colors, markers)
- `[plots.saturation]`: Saturation annotation control
- `[plots.stddev]`: Standard deviation display
- `[output.files]`: Output filenames

## Data Flow

### Analysis Pipeline
```
1. scan_experiment_directories() → Find exp dirs
2. load_and_aggregate_results() → Load parquet, add 'seed' column
3. compute_transient_period_duration() → Calculate warmup
4. Filter warmup/cooldown periods
5. compute_per_seed_statistics() → Per-seed DataFrame
6. compute_aggregate_statistics() → Mean + stddev
7. build_experiment_index() → Full DataFrame
8. plot_*() functions → Generate visualizations
9. generate_*_table() functions → Markdown tables
```

### Transaction Data Structure
```
Parquet columns:
- t_submit: Time (ms)
- commit_latency: Commit protocol time (ms)
- total_latency: End-to-end time (ms)
- n_retries: Retry count
- status: 'committed' or 'aborted'
- seed: Added during load_and_aggregate_results()
```

## Common Pitfalls

### 1. Editing Without Reading
**Problem**: File modified since last read
**Solution**: Always `Read` before `Edit`, even if recently read

### 2. String Matching in Edit
**Problem**: Line numbers in Read output don't match file content
**Solution**: Copy exact text excluding line number prefix

### 3. Global CONFIG State
**Problem**: CONFIG may not be initialized in tests
**Solution**: Use `CONFIG.get('key', default)` pattern everywhere

### 4. Backward Compatibility
**Problem**: New features break existing tests
**Solution**: Check for optional columns/features before using:
```python
has_stddev = 'p50_commit_latency_std' in df.columns
if has_stddev:
    # Use stddev
else:
    # Fall back
```

## Token Optimization Strategies

### 1. Use Line-Based Read
```python
Read(file_path, offset=X, limit=Y)  # Read specific section
```
Instead of reading entire 1400-line file

### 2. Grep Before Read
```python
Grep(pattern="def function_name", output_mode="content", -n=True)
```
Get line number, then Read specific range

### 3. Reference CODE_NAVIGATION.md
Look up function line numbers instead of searching

### 4. Copy-Paste-Modify Pattern
For similar features (e.g., plots), copy existing implementation

### 5. Incremental Testing
Test each change immediately instead of batch changes

## File Sizes (Reference)

- `endive/saturation_analysis.py`: ~1400 lines, ~45KB
- `endive/main.py`: ~950 lines, ~35KB
- `tests/test_saturation_analysis.py`: ~600 lines, ~20KB
- `analysis.toml`: ~130 lines, ~4KB

## Quick Reference: Where Things Are

**Add plot**: Line 900+ in saturation_analysis.py
**Add config**: Line 28-115 in saturation_analysis.py + analysis.toml
**Modify stats**: Line 372-520 in saturation_analysis.py
**Modify tables**: Line 1009-1273 in saturation_analysis.py
**CLI entry**: Line 1299 in saturation_analysis.py
**Conflict logic**: Line 667-813 in main.py
**Table selection**: Line 884-920 in main.py

## Experiment Patterns

### Naming Convention
```
{experiment_label}-{config_hash}/{seed}/results.parquet
```
Example: `exp2_1_single_table_false-517705b9/1234567890/results.parquet`

### Standard Seeds
All experiments use 5 seeds (configurable with min_seeds=3 filter)

### Experiment Groups
- `exp2_1_*`: Single-table, false conflicts (9 configs)
- `exp2_2_*`: Multi-table, false conflicts (54 configs)
- `exp3_1_*`: Single-table, real conflicts (58 configs)
- `exp3_2_*`: Manifest distribution variance (32 configs)

## Scripts

### Plot Regeneration
```bash
./scripts/regenerate_all_plots.sh [--parallel N]
```
Regenerates all 8 output files for each experiment group

### Analysis Commands
```bash
# Single group
python -m endive.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1

# With grouping
python -m endive.saturation_analysis -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables

# With custom config
python -m endive.saturation_analysis --config analysis.toml
```

## Testing Strategy

### Run Specific Test Class
```bash
pytest tests/test_saturation_analysis.py::TestConfigurationSystem -v
```

### Run Single Test
```bash
pytest tests/test_saturation_analysis.py::TestConfigurationSystem::test_default_config_structure -v
```

### Quick Validation
Test configuration loading:
```bash
pytest tests/test_saturation_analysis.py -k "config" -v
```

## Session Efficiency Metrics

Target for future sessions:
- **Token budget**: 200k
- **Efficient usage**: < 150k (75%)
- **Critical threshold**: 180k (90%)

Strategies to stay under budget:
1. Start with CODE_NAVIGATION.md for line numbers
2. Use targeted Grep instead of full file reads
3. Read only changed sections after edits
4. Use session_memory.md for context instead of re-reading code
5. Batch related changes to minimize context switching
