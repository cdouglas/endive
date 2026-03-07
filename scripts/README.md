# Scripts

Utilities for running experiments, analyzing results, and managing experiment data.

## run_all_experiments.py

Run parameter-sweep experiments with progress tracking and resume support.

```bash
python scripts/run_all_experiments.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--parallel`, `-p` | 4 | Number of parallel workers |
| `--seeds`, `-s` | 3 | Seeds per configuration |
| `--groups`, `-g` | all | Comma-separated groups: `baseline`, `heatmap`, `catalog`, `tables`, `providers` |
| `--quick`, `-q` | | 1-minute duration, fewer params |
| `--dry-run`, `-n` | | Show runs without executing |
| `--status` | | Show runner progress |
| `--resume`, `-r` | | Skip previously completed runs |
| `--force`, `-f` | | Re-run existing experiments |
| `--profile` | | Write `.profile.json` per run |

**Examples:**

```bash
# Full suite with 8 cores
python scripts/run_all_experiments.py --parallel 8 --seeds 5

# Specific groups
python scripts/run_all_experiments.py --groups baseline,heatmap --seeds 3

# Quick smoke test
python scripts/run_all_experiments.py --quick --seeds 1

# Check progress of a running batch
python scripts/run_all_experiments.py --status

# Resume after interruption
python scripts/run_all_experiments.py --resume --parallel 8

# Background with logging
nohup python scripts/run_all_experiments.py --parallel 8 > experiments.log 2>&1 &
```

## check_progress.py

Monitor running experiments by reading `.runner_state.json` and per-seed `.progress.json` files.

```bash
python scripts/check_progress.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--watch`, `-w` | | Continuously refresh display |
| `--interval`, `-i` | 10 | Refresh interval in seconds |

**Examples:**

```bash
# One-shot status
python scripts/check_progress.py

# Live dashboard
python scripts/check_progress.py --watch --interval 5
```

## consolidate.py

Merge per-seed parquet files into a single consolidated file for efficient analysis.

```bash
python scripts/consolidate.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base-dir` | `experiments` | Experiments directory |
| `--output` | `experiments/consolidated.parquet` | Output path (single-file mode) |
| `--batch-size` | 50 | Experiments per write batch |
| `--compression` | `zstd` | Compression codec (`zstd` or `snappy`) |
| `--compression-level` | 3 | Compression level |
| `--verify` | | Verify after writing |
| `--verify-sample` | 20 | Number of files to sample-verify |
| `--verify-only` | | Verify existing file without consolidating |
| `--full` | | Verify every row group (with `--verify-only`) |
| `--destructive` | | Delete source directories after writing |
| `--partition` | | One file per label instead of single file |
| `--max-workers` | CPU count | Workers for `--partition` mode |

**Examples:**

```bash
# Default consolidation
python scripts/consolidate.py

# Consolidate and verify
python scripts/consolidate.py --verify

# Per-label files for selective loading
python scripts/consolidate.py --partition

# Verify existing consolidated file
python scripts/consolidate.py --verify-only

# Full row-level verification
python scripts/consolidate.py --verify-only --full

# Consolidate and reclaim disk space
python scripts/consolidate.py --destructive --verify
```

## regenerate_plots.py

Generate plots from `[plots]` sections in experiment configs. Merges per-graph overrides with `plotting.toml` defaults.

```bash
python scripts/regenerate_plots.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--parallel`, `-p` | 4 | Concurrent plot workers |
| `--config`, `-c` | | Process a single config file |
| `--pattern` | | Only configs matching glob (e.g., `exp3*`) |
| `--dry-run`, `-n` | | Show what would be generated |
| `--input-dir`, `-i` | `experiments` | Experiments base directory |

**Examples:**

```bash
# Regenerate all plots
python scripts/regenerate_plots.py

# Preview what would be generated
python scripts/regenerate_plots.py --dry-run

# Single experiment
python scripts/regenerate_plots.py --config experiment_configs/exp1_fa_baseline.toml

# Pattern match
python scripts/regenerate_plots.py --pattern "exp4*"
```

## dump_results.py

Inspect parquet result files with filtering, formatting, and statistics.

```bash
python scripts/dump_results.py FILE [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--schema` | | Show schema only |
| `--summary` | | Show summary only |
| `--stats` | | Show detailed statistics |
| `--all` | | Print all rows |
| `--head N` | | First N rows |
| `--tail N` | | Last N rows |
| `--exp-name` | | Filter by experiment name (consolidated) |
| `--exp-hash` | | Filter by experiment hash (consolidated) |
| `--seed` | | Filter by seed (consolidated) |
| `--status` | | Filter by `committed` or `aborted` |
| `--time-range START END` | | Filter by submission time (ms) |
| `--retries-min N` | | Transactions with >= N retries |
| `--format` | `table` | Output format: `table`, `csv`, `json` |
| `--columns COL ...` | | Show only named columns |
| `--sort COLUMN` | | Sort by column |
| `--no-schema` | | Skip schema display |
| `--no-summary` | | Skip summary display |

**Examples:**

```bash
# Schema and summary
python scripts/dump_results.py experiments/exp1_fa_baseline-11132500/42/results.parquet

# First 20 rows
python scripts/dump_results.py results.parquet --head 20

# Committed transactions as CSV
python scripts/dump_results.py results.parquet --status committed --format csv

# High-retry transactions
python scripts/dump_results.py results.parquet --retries-min 5 --columns txn_id n_retries commit_latency

# Query consolidated file
python scripts/dump_results.py experiments/consolidated.parquet \
    --exp-name exp2_mix_heatmap --exp-hash 078af7bc --seed 42 --stats
```

## validate_experiments.py

Run end-to-end pipeline validation: parquet schema, operation distribution, conflict resolution, per-operation metrics.

```bash
python scripts/validate_experiments.py [options]
```

| Flag | Description |
|------|-------------|
| `--run` | Run validation experiments |
| `--check-only` | Only check existing results |

**Examples:**

```bash
# Run experiments and validate
python scripts/validate_experiments.py --run

# Check existing results
python scripts/validate_experiments.py --check-only
```

## expctl.py

Experiment management CLI (in progress). Provides `ExperimentStore` for scanning and filtering experiments across disk and consolidated parquet.

```bash
python scripts/expctl.py {list,compact,gc,complete}
```

Subcommands are under development.

## plot_distributions.py

Visualize configured distributions (runtime, inter-arrival) from experiment configs.

```bash
python scripts/plot_distributions.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input-dir` | `experiments` | Experiments directory |
| `-o`, `--output-dir` | `plots/distributions` | Output directory |
| `-p`, `--pattern` | `exp2_*` | Pattern for experiment directories |

**Examples:**

```bash
# Default (exp2 distributions)
python scripts/plot_distributions.py

# All experiments
python scripts/plot_distributions.py --pattern "exp*"
```

## plot_optimization_comparison.py

Compare optimization variants (baseline, ml_append, metadata, combined) per storage provider. No arguments.

```bash
python scripts/plot_optimization_comparison.py
```

## plot_parameter_sensitivity.py

Generate parameter sensitivity heatmaps at fixed load levels. No arguments.

```bash
python scripts/plot_parameter_sensitivity.py
```
