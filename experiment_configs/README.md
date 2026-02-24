# Experiment Configurations

## Active Experiments

| Config | Group | Sweep | Description |
|--------|-------|-------|-------------|
| `exp1_fa_baseline.toml` | `baseline` | load | 100% FastAppend, instant catalog, S3 storage |
| `exp2_mix_heatmap.toml` | `heatmap` | load × fa_ratio | FA/VO operation mix 2D sweep |
| `exp3a_catalog_fa.toml` | `catalog` | catalog_latency × load | Catalog latency impact (100% FA) |
| `exp3b_catalog_mix.toml` | `catalog` | catalog_latency × load | Catalog latency impact (90/10 FA/VO mix) |
| `exp4a_tables_fa.toml` | `tables` | num_tables × catalog_latency × load | Multi-table catalog contention (100% FA) |
| `exp4b_tables_mix.toml` | `tables` | num_tables × catalog_latency × load | Multi-table catalog contention (90/10 FA/VO mix) |

Each config has a `[plots]` section declaring which graphs to generate.
See `plotting.toml` for default plot parameters.

## Running

```bash
# All experiments
python scripts/run_all_experiments.py --seeds 5 --parallel 8

# Single group
python scripts/run_all_experiments.py --groups heatmap --seeds 3

# Quick test (1 min duration, fewer params)
python scripts/run_all_experiments.py --quick --seeds 1

# Dry run
python scripts/run_all_experiments.py --dry-run
```

## Plotting

```bash
# Regenerate all plots
python scripts/regenerate_plots.py

# Dry run
python scripts/regenerate_plots.py --dry-run

# Single config
python scripts/regenerate_plots.py --config experiment_configs/exp2_mix_heatmap.toml
```

## Operation Types

| Type | Validation | Retry Cost | Can Abort |
|------|------------|------------|-----------|
| `fast_append` | None | ~160ms (1 ML read/write) | No |
| `validated_overwrite` | O(N) ML reads | ~N×100ms | Yes (ValidationException) |
