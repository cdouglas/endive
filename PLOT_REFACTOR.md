# Plot Infrastructure Refactor

## Problem

The plotting infrastructure grew organically:
- 3 separate scripts: `saturation_analysis.py` CLI, `plot_heatmap.py`, `analyze_operation_types.py`
- A 500-line shell script (`regenerate_all_plots.sh`) with hardcoded experiment patterns
- `analysis.toml` with global defaults disconnected from per-experiment needs
- Config filenames that didn't match experiment labels

Adding a new experiment required editing 3+ files to wire up its plots.

## Solution

**Single source of truth**: each experiment config declares exactly which graphs to generate.

### Architecture

```
plotting.toml               <- Global defaults, one section per function
experiment_configs/*.toml    <- Each has [plots] with positive list of graphs
scripts/regenerate_plots.py  <- Reads configs, dispatches to functions
endive/saturation_analysis.py <- All plotting functions (including heatmap, op-type)
```

### How It Works

1. **`plotting.toml`** defines per-function defaults:
   ```toml
   [latency_vs_throughput]
   title = "Latency vs Throughput"
   figsize = [12, 8]
   percentiles = [50, 95, 99]
   ```

2. **Experiment configs** declare a positive list of graphs:
   ```toml
   [plots]
   output_dir = "plots/exp1_fa_baseline"

   [[plots.graphs]]
   type = "latency_vs_throughput"
   group_by = "catalog_service_latency_ms"  # Per-experiment override
   ```

3. **`regenerate_plots.py`** scans configs, merges defaults with overrides, dispatches:
   ```
   plotting.toml[latency_vs_throughput]  <-  [[plots.graphs]] overrides
                    |
                    v
          plot_latency_vs_throughput(index_df, output_path, **merged)
   ```

### Graph Registry

Each `type` maps to exactly one Python function:

| type | function | data pipeline |
|------|----------|---------------|
| `latency_vs_throughput` | `plot_latency_vs_throughput()` | index_df |
| `latency_vs_throughput_table` | `generate_latency_vs_throughput_table()` | index_df |
| `success_rate_vs_load` | `plot_success_rate_vs_load()` | index_df |
| `success_rate_vs_throughput` | `plot_success_rate_vs_throughput()` | index_df |
| `overhead_vs_throughput` | `plot_overhead_vs_throughput()` | index_df |
| `commit_rate_over_time` | `plot_commit_rate_over_time()` | time_series |
| `sustainable_throughput` | `plot_sustainable_throughput()` | index_df |
| `heatmap` | `generate_heatmap_plots()` | raw parquet |
| `operation_types` | `generate_operation_type_plots()` | raw parquet |

### Design Decisions

**Positive list, not skip list**: Experiments declare what to generate, not what to exclude.

**Function-mapped sections**: Each `plotting.toml` section name = graph type = function suffix. Adding a function means adding a section.

**Two data pipelines**: Most graphs consume `index_df` from `build_experiment_index()`. Heatmaps and operation-type plots do their own data loading (they need per-seed, per-operation-type granularity). The registry knows which pipeline each type needs.

**One entry can produce multiple files**: `heatmap` with 4 metrics generates 4+ PNGs. The mapping is 1:function, not 1:file.

**Config in experiment file, not separate**: Keeping `[plots]` in the experiment config (rather than a separate `plots.toml`) avoids doubling config files and keeps tightly coupled information together.

**Hash stability**: `compute_experiment_hash()` excludes `[plots]` so adding plot config doesn't invalidate experiment directories.

### Usage

```bash
# Regenerate all plots
python scripts/regenerate_plots.py

# Dry run
python scripts/regenerate_plots.py --dry-run

# Single config
python scripts/regenerate_plots.py --config experiment_configs/exp1_fa_baseline.toml

# Pattern filter
python scripts/regenerate_plots.py --pattern "exp3*"
```

## Files Changed

| File | Change |
|------|--------|
| `plotting.toml` | New: function-mapped global defaults |
| `scripts/regenerate_plots.py` | New: unified plotting dispatch |
| `endive/config.py` | Exclude `[plots]` from hash |
| `endive/saturation_analysis.py` | Added heatmap/op-type functions, removed `--skip-plots` |
| `experiment_configs/exp1_fa_baseline.toml` | Renamed + `[plots]` section |
| `experiment_configs/exp2_mix_heatmap.toml` | Added `[plots]` section |
| `experiment_configs/exp3a_catalog_fa.toml` | Renamed + `[plots]` section |
| `experiment_configs/exp3b_catalog_mix.toml` | Renamed + `[plots]` section |

**Deleted**: `scripts/regenerate_all_plots.sh`, `scripts/plot_heatmap.py`, `scripts/analyze_operation_types.py`, `analysis.toml`
