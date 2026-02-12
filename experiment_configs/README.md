# Experiment Configurations

## Blog Post Experiments

These experiments support the blog post on catalog and table conflicts.

| Config | Question | Sweep Parameter |
|--------|----------|-----------------|
| `single_table_trivial.toml` | 1a: Single table saturation | `inter_arrival.scale` |
| `single_table_trivial_backoff.toml` | 1a variant: With exponential backoff | `inter_arrival.scale` |
| `single_table_mixed.toml` | 1b: Non-trivial conflict impact | `real_conflict_probability` |
| `multi_table_trivial.toml` | 2a: Multi-table scaling | `num_tables` |
| `multi_table_mixed.toml` | 2b: Multi-table with conflicts | `num_tables` x `real_conflict_probability` |

## Key Configuration Options

### Realistic Latencies

All configs use:
```toml
[storage]
provider = "s3"  # Size-based latency from Durner et al. VLDB 2023

[catalog]
table_metadata_inlined = false  # Model table metadata I/O separately
```

### What's Modeled

With `table_metadata_inlined = false`, the simulator models:

**Initial transaction:**
- Table metadata read (per table accessed)
- Table metadata write (per table written)
- Manifest list write (per table written)

**Conflict resolution (trivial):**
- Table metadata read
- Table metadata write
- Manifest list already read via `read_manifest_lists()`

**Conflict resolution (non-trivial):**
- Table metadata read
- Manifest list read
- Manifest files read/write (1-10 files, parallel up to 4)
- Manifest list write
- Table metadata write

### Sweep Parameters

Configs document which parameters to sweep in comments. Common sweeps:

**Inter-arrival time** (offered load):
```
inter_arrival.scale = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
# Lower = higher arrival rate = more contention
# 100ms = ~10 TPS offered
```

**Conflict probability**:
```
real_conflict_probability = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
# 0.0 = all trivial (table metadata + manifest list only)
# 1.0 = all real (also rewrite manifest files)
```

**Table count**:
```
num_tables = [1, 2, 5, 10, 20, 50]
# More tables = lower per-table conflict rate
```

## Running Experiments

```bash
# Single simulation
python -m endive.main experiment_configs/single_table_trivial.toml --yes

# Parameter sweep example
for scale in 10 20 50 100 200 500 1000 2000 5000; do
    sed "s/inter_arrival.scale = .*/inter_arrival.scale = $scale/" \
        experiment_configs/single_table_trivial.toml > /tmp/config.toml
    python -m endive.main /tmp/config.toml --yes
done
```

## Storage Providers

Available providers (set via `storage.provider`):

| Provider | CAS Median | PUT Base | PUT Rate | Source |
|----------|------------|----------|----------|--------|
| `s3` | 23ms | 30ms | 20 ms/MiB | YCSB/Durner |
| `s3x` | 22ms | 10ms | 10 ms/MiB | YCSB/estimate |
| `azure` | 87ms | 50ms | 25 ms/MiB | YCSB/estimate |
| `azurex` | 64ms | 30ms | 15 ms/MiB | YCSB/estimate |
| `gcp` | 170ms | 40ms | 17 ms/MiB | YCSB/Durner |

## Analysis

```bash
# Generate plots for an experiment
python -m endive.saturation_analysis \
    -i experiments \
    -p "single_table_trivial*" \
    -o plots/single_table_trivial

# Compare configurations
python -m endive.saturation_analysis \
    -i experiments \
    -p "single_table_*" \
    -o plots/single_table \
    --group-by label
```

## Archive

Old experiment configs (expX_Y naming) are in `archive/`.
