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

Compare baseline (no optimizations), metadata inlining, and manifest list append across providers.

**Factorial Design:**

| Config | table_metadata_inlined | manifest_list_mode | Effect |
|--------|:----------------------:|:------------------:|--------|
| baseline | false | rewrite | No optimizations |
| metadata | true | rewrite | Inlining only |
| ml_append | false | append | ML+ only |
| combined | true | append | Both optimizations |

**Provider Coverage:**

| Config | S3 | S3 Express | Azure | Azure Premium |
|--------|:--:|:----------:|:-----:|:-------------:|
| baseline | ✓ | ✓ | ✓ | ✓ |
| metadata | ✓ | ✓ | ✓ | ✓ |
| ml_append | - | ✓ | ✓ | ✓ |
| combined | - | ✓ | ✓ | ✓ |

*Note: S3 Standard doesn't support conditional append, so ml_append and combined only run on S3 Express, Azure, and Azure Premium.*

## Key Configuration Options

### Baseline (no optimizations)

```toml
[catalog]
table_metadata_inlined = false  # Separate I/O for table metadata

[transaction]
manifest_list_mode = "rewrite"  # Traditional Iceberg
```

### Metadata Inlining Optimization

```toml
[catalog]
table_metadata_inlined = true  # No separate table metadata I/O
```

When `table_metadata_inlined = true`, table metadata is inlined in the
catalog pointer, eliminating separate read/write I/O per commit.

### Manifest List Append Optimization (ML+)

```toml
[transaction]
manifest_list_mode = "append"  # ML+ mode
```

In ML+ (append) mode:
- Manifest list entries are appended instead of rewritten
- Entries are tagged with txn_id and filtered by committed transactions
- False conflicts resolved faster (no ML rewrite needed)

## Storage Providers

| Provider | CAS Median | Append | PUT Base | PUT Rate | Source |
|----------|------------|--------|----------|----------|--------|
| `s3` | 61ms | N/A | 30ms | 20 ms/MiB | YCSB June 2025 |
| `s3x` | 22ms | 21ms | 10ms | 10 ms/MiB | YCSB June 2025 |
| `azure` | 93ms | 87ms | 50ms | 25 ms/MiB | YCSB June 2025 |
| `azurex` | 64ms | 70ms | 30ms | 15 ms/MiB | YCSB estimate |
| `gcp` | 170ms | N/A | 40ms | 17 ms/MiB | YCSB/Durner |

## Running Experiments

### Quick Start (local)

```bash
# Run all experiments (may take several hours)
python scripts/run_all_experiments.py --parallel 4 --seeds 3

# Run specific groups
python scripts/run_all_experiments.py --groups baseline,metadata --seeds 3

# Quick test mode (1 minute duration)
python scripts/run_all_experiments.py --quick --parallel 4

# Check progress
python scripts/run_all_experiments.py --status
```

### Background Execution (detachable)

```bash
# Run in background with logging
nohup python scripts/run_all_experiments.py --parallel 8 --seeds 3 2>&1 | \
    tee experiment_logs/run_$(date +%Y%m%d_%H%M%S).log &

# Check progress
python scripts/run_all_experiments.py --status

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
    bash -c "python scripts/run_all_experiments.py \${EXP_ARGS} 2>&1 | \
        tee experiment_logs/run_\$(date +%Y%m%d_%H%M%S).log"

# Check container logs
docker logs -f <container_id>

# Check progress
docker exec <container_id> python scripts/run_all_experiments.py --status
```

### Available Experiment Groups

| Group | Description | Configs |
|-------|-------------|---------|
| `trivial` | Single table trivial conflicts | 2 configs |
| `mixed` | Single table mixed conflicts | 1 config |
| `multi_table` | Multi-table experiments | 2 configs |
| `baseline` | Baseline - no optimizations | 3 configs |
| `metadata` | Metadata inlined (optimization) | 3 configs |
| `ml_append` | Manifest list append (optimization) | 3 configs |
| `combined` | Both optimizations | 3 configs |
| `partition` | Partition-level modeling | 3 configs |

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

## Partition-Level Modeling

Partition experiments model tables with per-partition manifest lists, where transactions can span multiple partitions and conflicts are detected per-partition.

### Partition Configs

| Config | Distribution | Description |
|--------|--------------|-------------|
| `partition_zipf.toml` | Zipf | Hot partition scenario (partition 0 is hottest) |
| `partition_uniform.toml` | Uniform | Baseline with no hot partitions |
| `partition_scaling.toml` | Zipf | Scaling analysis: vary num_partitions |

### Configuration Options

```toml
[partition]
enabled = true
num_partitions = 100          # Partitions per table
partitions_per_txn_mean = 3.0 # Mean partitions per transaction
partitions_per_txn_max = 10   # Maximum partitions per transaction

[partition.selection]
distribution = "zipf"   # "zipf" or "uniform"
zipf_alpha = 1.5        # Higher = more skewed (hotter hot partitions)
```

### Key Differences from Table Groups

| Feature | Table Groups | Partitions |
|---------|--------------|------------|
| Spans multiple | No | Yes |
| Conflict boundary | Table | Partition |
| ML distribution | Per-table | Per-partition |
| Use case | Multi-tenant | Partitioned tables |

With partitions enabled:
- Each partition has its own manifest list (distributed ML load)
- Transactions can span multiple partitions (unlike table groups)
- Conflicts only occur when transactions touch the same partition
- Higher partition count reduces conflict probability

## Operation Type Experiments

These experiments study the impact of Iceberg operation types (FastAppend, ValidatedOverwrite) on commit throughput.

### Experiment Groups

| Group | Config | Sweep | Description |
|-------|--------|-------|-------------|
| `op_type_baseline` | `exp1_fastappend_baseline.toml` | load | 100% FastAppend baseline |
| `op_type_heatmap` | `exp2_mix_heatmap.toml` | load × fa_ratio | 2D heatmap: operation mix impact |
| `op_type_catalog` | `exp3a_catalog_latency_fa.toml` | provider × load | Catalog latency (100% FA) |
| `op_type_catalog` | `exp3b_catalog_latency_mix.toml` | provider × load | Catalog latency (90/10 mix) |

### Operation Types

| Type | Validation | Retry Cost | Can Abort |
|------|------------|------------|-----------|
| `fast_append` | None | ~160ms (1 ML read/write) | No |
| `validated_overwrite` | O(N) ML reads | ~N×100ms | Yes (ValidationException) |

### Running

```bash
# Quick test (1 min, subset of params)
python scripts/run_all_experiments.py --groups op_type_baseline --quick --seeds 1

# Full experiments
python scripts/run_all_experiments.py --groups op_type_baseline,op_type_heatmap --seeds 5
```

### Validation

```bash
python scripts/validate_experiments.py --run    # Run V1-V5 validation experiments
python scripts/validate_experiments.py --check-only  # Validate existing results
```

## Archive

Old experiment configs (expX_Y naming) are in `archive/`.
