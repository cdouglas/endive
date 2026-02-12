# Experiment Configurations

## Blog Post Experiments

These experiments support the blog post on catalog and table conflicts.

### Question 1: Single-Table Saturation

| Config | Question | Sweep Parameter |
|--------|----------|-----------------|
| `single_table_trivial.toml` | 1a: Single table saturation | `inter_arrival.scale` |
| `single_table_trivial_backoff.toml` | 1a variant: With exponential backoff | `inter_arrival.scale` |
| `single_table_mixed.toml` | 1b: Non-trivial conflict impact | `real_conflict_probability` |

### Question 2: Multi-Table Scaling

| Config | Question | Sweep Parameter |
|--------|----------|-----------------|
| `multi_table_trivial.toml` | 2a: Multi-table scaling | `num_tables` |
| `multi_table_mixed.toml` | 2b: Multi-table with conflicts | `num_tables` x `real_conflict_probability` |

### Question 3: Optimization Impact

Compare baseline, metadata inlining, and manifest list append across providers.

| Config | Optimization | Provider |
|--------|--------------|----------|
| `baseline_s3.toml` | Baseline (inlined, rewrite) | S3 |
| `baseline_s3x.toml` | Baseline | S3 Express |
| `baseline_azure.toml` | Baseline | Azure Blob |
| `metadata_not_inlined_s3.toml` | Separate table metadata | S3 |
| `metadata_not_inlined_s3x.toml` | Separate table metadata | S3 Express |
| `metadata_not_inlined_azure.toml` | Separate table metadata | Azure Blob |
| `ml_append_s3.toml` | ML+ append mode | S3 |
| `ml_append_s3x.toml` | ML+ append mode | S3 Express |
| `ml_append_azure.toml` | ML+ append mode | Azure Blob |
| `combined_optimizations_s3.toml` | Both optimizations | S3 |
| `combined_optimizations_s3x.toml` | Both optimizations | S3 Express |
| `combined_optimizations_azure.toml` | Both optimizations | Azure Blob |

## Key Configuration Options

### Realistic Latencies

All configs use provider-calibrated latencies:

```toml
[storage]
provider = "s3"  # or "s3x", "azure", "azurex", "gcp"
```

### Table Metadata Inlining

```toml
[catalog]
table_metadata_inlined = false  # Model table metadata I/O separately
```

When `table_metadata_inlined = false`, each commit requires:
- Read table metadata JSON (~10KB)
- Write updated table metadata JSON

### Manifest List Mode

```toml
[transaction]
manifest_list_mode = "append"  # ML+ mode (vs "rewrite")
```

In ML+ (append) mode:
- Manifest list entries are appended instead of rewritten
- Entries are tagged with txn_id and filtered by committed transactions
- False conflicts resolved faster (no ML rewrite needed)

## Storage Providers

| Provider | CAS Median | PUT Base | PUT Rate | Source |
|----------|------------|----------|----------|--------|
| `s3` | 23ms | 30ms | 20 ms/MiB | YCSB/Durner |
| `s3x` | 22ms | 10ms | 10 ms/MiB | YCSB/estimate |
| `azure` | 87ms | 50ms | 25 ms/MiB | YCSB/estimate |
| `azurex` | 64ms | 30ms | 15 ms/MiB | YCSB/estimate |
| `gcp` | 170ms | 40ms | 17 ms/MiB | YCSB/Durner |

## Running Experiments

### Quick Start (local)

```bash
# Run all experiments (may take several hours)
./scripts/run_all_experiments.sh --parallel 4 --seeds 3

# Run specific groups
./scripts/run_all_experiments.sh --groups baseline,metadata --seeds 3

# Quick test mode (1 minute duration)
./scripts/run_all_experiments.sh --quick --parallel 4

# Check progress
./scripts/run_all_experiments.sh --status
```

### Background Execution (detachable)

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
    -e EXP_ARGS="--groups baseline,metadata --seeds 3 --parallel 8" \
    -e DOCKER_CONTAINER=1 \
    -e OMP_NUM_THREADS=1 \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/experiment_logs:/app/experiment_logs \
    cdouglas/endive-sim:latest \
    bash -c "scripts/run_all_experiments.sh \${EXP_ARGS} 2>&1 | \
        tee experiment_logs/run_\$(date +%Y%m%d_%H%M%S).log"

# Check container logs
docker logs -f <container_id>

# Check progress
docker exec <container_id> ./scripts/run_all_experiments.sh --status
```

### Available Experiment Groups

| Group | Description | Configs |
|-------|-------------|---------|
| `trivial` | Single table trivial conflicts | 2 configs |
| `mixed` | Single table mixed conflicts | 1 config |
| `multi_table` | Multi-table experiments | 2 configs |
| `baseline` | Baseline (3 providers) | 3 configs |
| `metadata` | Metadata not inlined | 3 configs |
| `ml_append` | Manifest list append | 3 configs |
| `combined` | Combined optimizations | 3 configs |

## Analysis

```bash
# Generate plots comparing optimizations
python -m endive.saturation_analysis \
    -i experiments \
    -p "baseline_s3-*" \
    -o plots/baseline_s3

# Compare across providers
python -m endive.saturation_analysis \
    -i experiments \
    -p "*_s3-*" \
    -o plots/s3_comparison \
    --group-by label
```

## Archive

Old experiment configs (expX_Y naming) are in `archive/`.
