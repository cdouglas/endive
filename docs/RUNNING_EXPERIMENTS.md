# Running Experiments

Quick reference guide for running optimization experiments.

## Quick Start

### Run All Experiments

```bash
# Full experiment suite with 3 seeds per configuration
./scripts/run_all_experiments.sh --parallel 4 --seeds 3

# Quick test mode (1 minute simulations, 3 load levels)
./scripts/run_all_experiments.sh --quick --parallel 4

# Dry run to see what would execute
./scripts/run_all_experiments.sh --dry-run --seeds 3
```

### Run Specific Groups

```bash
# Only baseline experiments (no optimizations)
./scripts/run_all_experiments.sh --groups baseline --seeds 3

# Only optimization experiments
./scripts/run_all_experiments.sh --groups metadata,ml_append,combined --seeds 3

# Compare standard vs premium tiers
./scripts/run_all_experiments.sh --groups baseline,metadata --seeds 3
```

## Experiment Groups

| Group | Configs | Description |
|-------|:-------:|-------------|
| `trivial` | 2 | Single table trivial conflicts |
| `mixed` | 1 | Single table mixed conflicts |
| `multi_table` | 2 | Multi-table experiments |
| `baseline` | 4 | No optimizations (S3, S3x, Azure, Azurex) |
| `metadata` | 4 | Metadata inlining only |
| `ml_append` | 3 | ML+ append only (S3x, Azure, Azurex) |
| `combined` | 3 | Both optimizations |
| `partition` | 3 | Partition-level modeling (Zipf, uniform, scaling) |

## Factorial Design

The optimization experiments use a 2×2 factorial design:

| Config | table_metadata_inlined | manifest_list_mode | Effect |
|--------|:----------------------:|:------------------:|--------|
| baseline | false | rewrite | Control (no optimizations) |
| metadata | true | rewrite | Inlining effect only |
| ml_append | false | append | ML+ effect only |
| combined | true | append | Interaction effect |

**Provider coverage:**

| Config | S3 | S3 Express | Azure | Azure Premium |
|--------|:--:|:----------:|:-----:|:-------------:|
| baseline | ✓ | ✓ | ✓ | ✓ |
| metadata | ✓ | ✓ | ✓ | ✓ |
| ml_append | - | ✓ | ✓ | ✓ |
| combined | - | ✓ | ✓ | ✓ |

*S3 Standard doesn't support conditional append.*

## Background Execution

### Run in Background with Logging

```bash
# Run in background with logging
nohup ./scripts/run_all_experiments.sh --parallel 8 --seeds 3 2>&1 | \
    tee experiment_logs/run_$(date +%Y%m%d_%H%M%S).log &

# Check progress
./scripts/run_all_experiments.sh --status

# Or tail the log
tail -f experiment_logs/run_*.log
```

### Docker Execution

```bash
# Build image
docker build -t cdouglas/endive-sim:latest .

# Run experiments in container
docker run -d \
    -e DOCKER_CONTAINER=1 \
    -e OMP_NUM_THREADS=1 \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/experiment_logs:/app/experiment_logs \
    cdouglas/endive-sim:latest \
    bash -c "scripts/run_all_experiments.sh --parallel 8 --seeds 3 2>&1 | \
        tee experiment_logs/run_\$(date +%Y%m%d_%H%M%S).log"

# Check container logs
docker logs -f <container_id>
```

## Output Structure

```
experiments/
├── baseline_s3-HASH/
│   ├── cfg.toml              # Configuration snapshot
│   ├── SEED1/results.parquet # First seed
│   ├── SEED2/results.parquet # Second seed
│   └── SEED3/results.parquet # Third seed
├── baseline_s3x-HASH/
├── metadata_s3-HASH/
├── ml_append_s3x-HASH/
├── combined_optimizations_s3x-HASH/
└── ...

experiment_logs/
└── run_YYYYMMDD_HHMMSS.log
```

## Time Estimates

| Configuration | Experiments | Est. Time (4 workers) |
|--------------|:-----------:|:---------------------:|
| Quick test (1 seed) | ~59 | ~15 seconds |
| Optimization groups (3 seeds) | ~177 | ~45 seconds |
| Full suite (3 seeds) | ~450 | ~2 minutes |

*Actual times depend on load levels and conflict rates.*

## Analysis After Completion

```bash
# Generate plots for specific experiment
python -m endive.saturation_analysis \
    -i experiments \
    -p "baseline_s3x-*" \
    -o plots/baseline_s3x

# Compare optimizations on same provider
python -m endive.saturation_analysis \
    -i experiments \
    -p "*_s3x-*" \
    -o plots/s3x_comparison \
    --group-by label

# Compare providers for same optimization
python -m endive.saturation_analysis \
    -i experiments \
    -p "baseline_*" \
    -o plots/baseline_providers \
    --group-by label
```

## Resume Capability

The runner supports resume after interruption:

```bash
# Check status of interrupted run
./scripts/run_all_experiments.sh --status

# Resume automatically picks up where it left off
./scripts/run_all_experiments.sh --parallel 4 --seeds 3
```

State is saved in `experiments/.runner_state.json`.

## Command Reference

```bash
# Basic usage
./scripts/run_all_experiments.sh [OPTIONS]

# Options
--groups GROUP1,GROUP2    # Run specific groups (default: all)
--seeds N                 # Number of seeds per config (default: 3)
--parallel N              # Parallel workers (default: 4)
--quick                   # Quick mode (1 min sims, 3 load levels)
--dry-run                 # Show what would run without executing
--status                  # Show progress of current/last run

# Examples
./scripts/run_all_experiments.sh --quick --parallel 4
./scripts/run_all_experiments.sh --groups baseline,metadata --seeds 5
./scripts/run_all_experiments.sh --dry-run --seeds 3
```

## Tips

1. **Test first**: Always run `--quick` before full experiments
2. **Parallelism**: Use `--parallel` equal to CPU cores for best throughput
3. **Seeds**: 3 seeds for exploration, 5+ seeds for publication-quality results
4. **Disk space**: Check available space (~100MB per experiment group)
5. **Background**: Use `nohup` and `tee` for long runs
