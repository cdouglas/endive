# Docker Setup for Iceberg Catalog Simulator

This document explains how to run the simulator experiments using Docker.

## Quick Start

### Build the Image

```bash
docker build -t cdouglas/endive-sim:latest .
```

### Run Experiments

```bash
# Run specific experiments with custom settings
docker run -d \
    -e EXP_ARGS="--exp4.1 --seeds 5 --parallel 112" \
    -e DOCKER_CONTAINER=1 \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/experiment_logs:/app/experiment_logs \
    cdouglas/endive-sim:latest \
    bash -c "scripts/run_baseline_experiments.sh \${EXP_ARGS} 2>&1 | tee experiment_logs/run_\$(date +%Y%m%d_%H%M%S).log"
```

Results will be available in:
- `./experiments/` - Raw experiment results (.parquet files)
- `./experiment_logs/` - Execution logs with timestamps

## Controlling Experiment Execution

### EXP_ARGS Environment Variable

Use the `EXP_ARGS` environment variable to control which experiments run and how:

```bash
# Run single experiment suite with 5 seeds
-e EXP_ARGS="--exp2.1 --seeds 5"

# Run multiple experiments in parallel with 96 cores
-e EXP_ARGS="--exp3.1 --exp3.2 --seeds 5 --parallel 96"

# Run with specific seed count and parallelism
-e EXP_ARGS="--exp4.1 --seeds 3 --parallel 112"

# Run all phase 2 experiments
-e EXP_ARGS="--exp2.1 --exp2.2 --seeds 5"

# Run all phase 3 experiments
-e EXP_ARGS="--exp3.1 --exp3.2 --exp3.3 --exp3.4 --seeds 5 --parallel 96"

# Run phase 4 experiments (backoff strategies)
-e EXP_ARGS="--exp4.1 --seeds 5"
```

### Available Experiment Flags

| Flag | Description | Configs | Seeds | Total Runs |
|------|-------------|---------|-------|------------|
| `--exp2.1` | Single-table false conflicts | 9 loads | 5 | 45 |
| `--exp2.2` | Multi-table false conflicts | 9 loads × 6 tables | 5 | 270 |
| `--exp3.1` | Single-table real conflicts | 9 loads × 7 p_real | 5 | 315 |
| `--exp3.2` | Manifest distributions | 9 loads × 4 dists | 5 | 180 |
| `--exp3.3` | Multi-table real conflicts | 9 loads × 5 tables × 4 p_real | 5 | 900 |
| `--exp3.4` | Exponential backoff | 9 loads | 5 | 45 |
| `--exp4.1` | Backoff with real conflicts | 9 loads × 6 backoff | 5 | 270 |

### Thread Control Environment Variables

Always set these to prevent thread oversubscription:

```bash
-e DOCKER_CONTAINER=1
-e OMP_NUM_THREADS=1
-e OPENBLAS_NUM_THREADS=1
-e MKL_NUM_THREADS=1
-e NUMEXPR_NUM_THREADS=1
```

These ensure numpy/scipy operations use single threads, allowing process-level parallelism via `--parallel`.

## Volume Mounts

The Docker setup uses three volume mounts:

| Host Path | Container Path | Purpose | Access |
|-----------|----------------|---------|--------|
| `./experiments` | `/app/experiments` | Experiment results (.parquet, cfg.toml) | Read/Write |
| `./plots` | `/app/plots` | Analysis plots and tables | Read/Write |
| `./experiment_logs` | `/app/experiment_logs` | Execution logs | Read/Write |
| `./experiment_configs` | `/app/experiment_configs` | Experiment configurations | Read-Only |

Ensure directories exist before running:
```bash
mkdir -p experiments plots experiment_logs
```

## Analysis

After experiments complete, generate plots:

```bash
docker run --rm \
    -v $(pwd)/experiments:/app/experiments:ro \
    -v $(pwd)/plots:/app/plots \
    cdouglas/endive-sim:latest \
    bash -c "scripts/regenerate_all_plots.sh"
```

Output will be in `./plots/exp{2,3,4,5}_*/`.

## Interactive Shell

For manual experimentation:

```bash
docker run -it --rm \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/plots:/app/plots \
    cdouglas/endive-sim:latest \
    bash
```

Inside the container:
```bash
# Run a single experiment
python -m endive.main experiment_configs/exp2_1_single_table_false_conflicts.toml --yes

# Run analysis for specific experiments
python -m endive.saturation_analysis -i experiments -o plots/custom -p "exp2_1_*"

# Run tests
pytest tests/ -v
```

## Monitoring Progress

### View Logs in Real-Time

```bash
# Get container ID
docker ps

# Follow logs
docker logs -f <container-id>

# Or check log files directly
tail -f experiment_logs/run_*.log
```

### Check Progress

```bash
# Count completed experiments
find experiments -name "results.parquet" | wc -l

# Show most recent experiments
ls -ltr experiments/*/results.parquet | tail -20
```

### Monitor Resource Usage

```bash
docker stats <container-id>
```

## Resource Requirements

### Minimum Requirements
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Disk:** 10 GB free space

### Recommended for Full Baseline
- **CPU:** 96+ cores (high parallelism)
- **RAM:** 64 GB (for consolidation operations)
- **Disk:** 100 GB (experiment results ~50GB, consolidated format ~5GB)

### Parallelism Guidelines

Match `--parallel` to available cores:
```bash
# 8-core machine
-e EXP_ARGS="--exp2.1 --seeds 5 --parallel 8"

# 96-core machine
-e EXP_ARGS="--exp3.3 --seeds 5 --parallel 96"

# 112-core machine
-e EXP_ARGS="--exp4.1 --seeds 5 --parallel 112"
```

## Advanced Usage

### Running Single Experiment

```bash
docker run --rm \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/experiment_configs:/app/experiment_configs:ro \
    cdouglas/endive-sim:latest \
    bash -c "python -m endive.main experiment_configs/exp2_1_single_table_false_conflicts.toml --yes"
```

### Custom Configuration

1. Create your config file in `experiment_configs/my_experiment.toml`
2. Run with:
```bash
docker run --rm \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/experiment_configs:/app/experiment_configs:ro \
    cdouglas/endive-sim:latest \
    bash -c "python -m endive.main experiment_configs/my_experiment.toml --yes"
```

### Consolidate Results

Create single consolidated.parquet file for efficient storage:

```bash
docker run --rm \
    -v $(pwd)/experiments:/app/experiments \
    cdouglas/endive-sim:latest \
    python scripts/consolidate.py
```

## Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker logs <container-id>
```

### Out of Memory

Reduce parallelism or increase Docker memory limit:
```bash
# Reduce parallel experiments
-e EXP_ARGS="--exp3.1 --seeds 5 --parallel 48"

# Docker Desktop: Settings → Resources → Memory (recommend 16GB+)
```

### Permission Errors on Volumes

Ensure directories are writable:
```bash
mkdir -p experiments plots experiment_logs
chmod 755 experiments plots experiment_logs
```

### Experiments Taking Too Long

For testing, run subset of experiments:
```bash
# Just one experiment suite with fewer seeds
-e EXP_ARGS="--exp2.1 --seeds 3 --parallel 8"
```

## Cleaning Up

### Remove Experiment Results

```bash
rm -rf experiments/* plots/* experiment_logs/*
```

### Remove Docker Images

```bash
docker rmi cdouglas/endive-sim:latest
```

## Multi-Architecture Support

Build for different platforms:

```bash
# Build for Linux AMD64 (most common)
docker build --platform linux/amd64 -t cdouglas/endive-sim:amd64 .

# Build for Linux ARM64 (Mac M1/M2)
docker build --platform linux/arm64 -t cdouglas/endive-sim:arm64 .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 \
    -t cdouglas/endive-sim:latest .
```

## Performance Tips

1. **Use SSD storage** for experiment results (high I/O)
2. **Match --parallel to CPU cores** for optimal throughput
3. **Monitor disk space** - experiments generate ~50GB of data
4. **Use consolidated format** to reduce storage by 90%
5. **Run analysis separately** after experiments complete
6. **Use `--no-cache`** when rebuilding after code changes:
   ```bash
   docker build --no-cache -t cdouglas/endive-sim:latest .
   ```
