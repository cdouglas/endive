# Code Navigation Guide

Quick reference for efficiently navigating and modifying the codebase.

## Key Files and Their Purpose

### Analysis Pipeline (`endive/saturation_analysis.py`)

**Size**: ~1400 lines
**Token-saving tip**: Use function-level grep instead of reading full file

**Key Functions by Line Number**:
- `get_default_config()` (line 28): Configuration defaults
- `load_config()` (line 115): TOML config loading with deep merge
- `scan_experiment_directories()` (line 163): Find experiment dirs
- `compute_transient_period_duration()` (line 222): Warmup calculation
- `load_and_aggregate_results()` (line 286): Load parquet files across seeds
- `compute_per_seed_statistics()` (line 372): Per-seed stats for stddev
- `compute_aggregate_statistics()` (line 422): Aggregate stats with stddev support
- `build_experiment_index()` (line 523): Build DataFrame with all experiments
- `plot_latency_vs_throughput()` (line 583): Main latency plot with error bands
- `plot_success_rate_vs_load()` (line 691): Success vs inter-arrival
- `plot_success_rate_vs_throughput()` (line 768): Success vs throughput
- `plot_overhead_vs_throughput()` (line 827): Overhead plot
- `plot_commit_rate_over_time()` (line 900): Time-series commit rate
- `format_value_with_stddev()` (line 997): Format "value ± stddev"
- `generate_latency_vs_throughput_table()` (line 1009): Markdown table with stddev
- `generate_overhead_table()` (line 1156): Markdown table with stddev
- `cli()` (line 1299): Main entry point

**Common Modifications**:
- Add new plot: Add function after line 900, wire into cli() after line 1400
- Add config option: Update get_default_config() and analysis.toml
- Modify statistics: Edit compute_aggregate_statistics() or compute_per_seed_statistics()
- Change table format: Edit generate_*_table() functions

### Simulator Core (`endive/main.py`)

**Key Sections**:
- Lines 65-141: Table grouping logic (partition_tables_into_groups)
- Lines 200-320: Configuration loading
- Lines 526-543: Conflicting manifest sampling
- Lines 667-759: Conflict resolution (ConflictResolver class)
- Lines 771-813: Catalog CAS logic (table-level vs catalog-level conflicts)
- Lines 828-882: Transaction commit with conflict resolution
- Lines 884-920: Table selection (Zipf distributions)

**Key Insight**: When `num_groups == num_tables`, conflicts are table-level (line 789-804)

### Configuration (`analysis.toml`)

**Sections**:
- `[paths]`: Input/output directories and patterns
- `[analysis]`: min_seeds, k_min_cycles, warmup settings, group_by
- `[plots]`: DPI, figsize, fonts, colors, markers
- `[plots.saturation]`: Saturation annotation settings
- `[plots.stddev]`: Standard deviation display settings
- `[output.files]`: Output filenames

## Data Structures

### Experiment Index DataFrame

**Columns**:
- Identity: `label`, `hash`, `num_seeds`, `exp_dir`
- Parameters: `inter_arrival_scale`, `num_tables`, `num_groups`, `real_conflict_probability`
- Aggregate stats: `total_txns`, `committed`, `success_rate`, `throughput`
- Latency: `p50_commit_latency`, `p95_commit_latency`, `p99_commit_latency`
- Overhead: `mean_overhead_pct`, `p50_overhead_pct`, `p95_overhead_pct`, `p99_overhead_pct`
- Stddev columns: `*_std` (e.g., `throughput_std`, `p50_commit_latency_std`)

### Raw Transaction Data (Parquet)

**Columns**:
- `t_submit`: Submission time (ms)
- `commit_latency`: Time spent in commit protocol (ms)
- `total_latency`: Total transaction latency (ms)
- `n_retries`: Number of retry attempts
- `status`: 'committed' or 'aborted'
- `seed`: Random seed identifier

## Common Patterns

### Adding a New Plot

1. **Add function** after line 900 in saturation_analysis.py
2. **Add filename** to CONFIG['output']['files'] in get_default_config()
3. **Wire into CLI** in cli() function after line 1400
4. **Test** with: `python -m endive.saturation_analysis -i experiments -p "exp2_1_*" -o plots/test`

### Adding a Configuration Option

1. **Add to get_default_config()** with nested structure
2. **Add to analysis.toml** with comments
3. **Use in code**: `CONFIG.get('section', {}).get('key', default)`
4. **Document** in docs/QUICKSTART.md

### Computing Statistics with Stddev

```python
# Check if 'seed' column exists
has_stddev = 'seed' in df.columns and df['seed'].nunique() > 1

if has_stddev:
    per_seed_df = compute_per_seed_statistics(df)
    mean = per_seed_df['metric'].mean()
    std = per_seed_df['metric'].std()
else:
    mean = df['metric'].mean()
    std = None
```

### Filtering Data

```python
# Apply warmup/cooldown filter
warmup_ms = compute_transient_period_duration(config)
cooldown_ms = warmup_ms
duration_ms = config['simulation']['duration_ms']

df_filtered = df[
    (df['t_submit'] >= warmup_ms) &
    (df['t_submit'] <= duration_ms - cooldown_ms)
]
```

## Token-Saving Strategies

### When Debugging
1. Use **Grep with line numbers** to find functions: `grep -n "def function_name"`
2. Read **specific line ranges**: `Read(offset=X, limit=Y)`
3. Search for **specific patterns** before reading large files

### When Adding Features
1. **Check existing similar features** first (e.g., other plot functions)
2. **Copy-paste-modify** pattern for consistency
3. **Test incrementally** rather than reading full files multiple times

### When Understanding Code
1. **Start with function signatures** (grep "def ")
2. **Read docstrings only** first (first 10 lines of functions)
3. **Use CODE_NAVIGATION.md** (this file) for line number references

## Quick Commands

```bash
# Find function definition with line number
grep -n "def function_name" endive/saturation_analysis.py

# Count functions in a file
grep -c "^def " endive/saturation_analysis.py

# Find all references to a variable
grep -n "CONFIG\['analysis'\]" endive/saturation_analysis.py

# Find where stats are computed
grep -n "mean_retries\|p50_commit_latency" endive/saturation_analysis.py

# Test a specific function
pytest tests/test_saturation_analysis.py::TestClass::test_function -v
```

## Architecture Principles

1. **Configuration over hardcoding**: All parameters in analysis.toml
2. **Backward compatibility**: Handle both with/without stddev cases
3. **Per-seed then aggregate**: Compute per seed first, then mean/stddev
4. **Warmup/cooldown filtering**: Always filter transient periods
5. **Consistent formatting**: Use format_value_with_stddev() for tables
6. **Minimum seed filtering**: Skip experiments with < min_seeds (default: 3)

## Recent Changes (Session Context)

This session added:
1. **Standard deviation visualization**: Error bands in plots, ± in tables
2. **Commit rate over time plots**: Time-series view of throughput
3. **Minimum seed filtering**: Skip incomplete experiments
4. **Plot regeneration script**: `scripts/regenerate_all_plots.sh`

Key files modified:
- `endive/saturation_analysis.py`: +200 lines (stddev, commit rate plot)
- `analysis.toml`: +min_seeds, +commit_rate_over_time_plot
- `scripts/regenerate_all_plots.sh`: New automation script

## Future Token Optimization

Potential improvements:
1. **Extract plot_base() function**: Common plotting setup code
2. **Create table_base() function**: Common table generation logic
3. **Split saturation_analysis.py**: Separate plotting, statistics, and CLI
4. **Add module docstrings**: High-level file purpose at top
5. **Create ARCHITECTURE.md**: Data flow diagrams and component relationships
