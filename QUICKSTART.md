# Quick Start Guide

Get up and running with the Iceberg Catalog Simulator in minutes.

## Prerequisites

```bash
# Activate virtual environment
source bin/activate

# Verify installation
python -m icecap.main --help
pytest tests/ -v  # Optional: verify all 63 tests pass
```

## Understanding the Simulator

Icecap models Apache Iceberg's optimistic concurrency control:

- **Compare-and-swap (CAS)** for atomic catalog updates
- **Conflict detection** when transactions fall behind
- **Retry logic** with manifest list reading and merging
- **Stochastic latencies** using normal distributions
- **Configurable parallelism** for manifest operations

## Run Your First Simulation

### Single Simulation

```bash
# Run with default configuration
python -m icecap.main cfg.toml

# The simulator will:
# 1. Display configuration summary
# 2. Ask for confirmation (press Enter or 'Y')
# 3. Show progress bar
# 4. Export results to results.parquet
# 5. Display summary statistics
```

**Example output:**
```
Simulation Summary:
  Total transactions: 20083
  Committed: 19856 (98.9%)
  Aborted: 227 (1.1%)
  Commit latency (ms):
    Mean: 234.56
    Median: 210.00
    P95: 485.20
    P99: 892.45
  Retries per transaction:
    Mean: 0.87
    Max: 8
```

**Note:** Results are written to `.running.parquet` during execution, then renamed to `results.parquet` on completion. Interrupted runs leave `.running.parquet` for easy cleanup.

## Run Baseline Experiments

The baseline experiments characterize system behavior under varying load:

### Quick Test (2 minutes)

```bash
# Test with short simulations and reduced configurations
./scripts/run_baseline_experiments.sh --quick --seeds 1
```

### Full Baseline (24 hours with 8 cores)

```bash
# Run all baseline experiments with parallel execution
./scripts/run_baseline_experiments.sh --seeds 3

# Or run in background with logging
nohup ./scripts/run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Monitor progress (refreshes every 5 seconds)
./scripts/monitor_experiments.sh --watch 5
```

**What runs:**
- **Experiment 2.1**: Single table saturation (9 load levels × 3 seeds = 27 runs)
- **Experiment 2.2**: Multi-table scaling (6 table counts × 9 load levels × 3 seeds = 162 runs)
- **Total**: 189 simulations (~1 hour each)

**Parallel execution benefits:**
- Default: Uses all CPU cores (e.g., 8 cores = 8 parallel jobs)
- Custom: `--parallel 4` limits to 4 concurrent experiments
- Speedup: ~8x faster with 8 cores (24 hours vs 8 days sequential)

## Analyze Results

### Saturation Analysis

Generate latency vs throughput curves:

```bash
# Analyze single-table experiments
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Analyze multi-table experiments (grouped by table count)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables
```

**Outputs:**
- `experiment_index.csv` - Summary statistics for all experiments
- `latency_vs_throughput.png` - P50/P95/P99 latency curves with saturation point
- `success_rate_vs_throughput.png` - Transaction success rate degradation
- `success_rate_vs_load.png` - Success rate vs offered load (inter-arrival time)

**Analysis features:**
- **Warmup filtering**: Automatically excludes first 9 minutes (transient period)
- **Multi-seed aggregation**: Combines results across random seeds
- **Saturation detection**: Marks 50% success rate threshold
- **Steady-state metrics**: P50/P95/P99 latency, throughput, success rate

### Warmup Validation

Visualize metric convergence to validate warmup period:

```bash
# Validate warmup period for an experiment
python -m icecap.warmup_validation \
    experiments/exp2_1_single_table_false-abc123/12345

# Custom window size for time-series analysis
python -m icecap.warmup_validation \
    experiments/exp2_1_*/12345 \
    --window-size 30
```

**Outputs:**
- Throughput over time (shows stabilization after warmup)
- Success rate over time (shows convergence)
- Mean and P95 latency trends
- Statistical stability analysis (coefficient of variation)

## Verify and Cleanup

```bash
# Count completed experiments
find experiments/ -name 'results.parquet' | wc -l

# Check for incomplete runs (interrupted simulations)
find experiments/ -name '.running.parquet'

# Clean up incomplete runs
find experiments/ -name '.running.parquet' -delete
```

## Configuration

### Key Parameters

Edit experiment configs in `experiment_configs/`:

```toml
[simulation]
duration_ms = 3600000        # 1 hour (stable statistics)
output_path = "results.parquet"

[experiment]
label = "exp2_1_single_table_false"  # Organizes output

[catalog]
num_tables = 1               # Number of tables
num_groups = 1               # Conflict granularity

[transaction]
retry = 10                   # Maximum retries

# Realistic Iceberg transaction durations
runtime.min = 30000          # 30 seconds minimum
runtime.mean = 180000        # 3 minutes mean
runtime.sigma = 1.5

# Inter-arrival time (offered load)
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0  # Mean inter-arrival (ms)

# Conflict types
real_conflict_probability = 0.0  # 0.0 = all false conflicts

[storage]
max_parallel = 4             # Parallel manifest operations

# Infinitely fast catalog (baseline)
T_CAS.mean = 1
T_CAS.stddev = 0.1

# Realistic S3 latencies
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
```

### Conflict Detection Granularity

Control how transactions conflict:

- **`num_groups = 1`**: Catalog-level conflicts (all transactions conflict)
- **`num_groups = T`**: Table-level conflicts (only same-table transactions conflict)
- **`1 < num_groups < T`**: Group-level isolation (multi-tenant modeling)

### False vs Real Conflicts

- **False conflicts** (`real_conflict_probability = 0.0`):
  - Version changed, no data overlap
  - Only metadata read required
  - Fast retry (~100ms)

- **Real conflicts** (`real_conflict_probability = 1.0`):
  - Overlapping data changes
  - Requires manifest file merge
  - Expensive retry (~500ms+)

## Interpret Results

### Success Rate Guidelines

```
>95% success  → Under-loaded, room to increase throughput
90-95% success → Healthy load with some contention
80-90% success → High contention, consider scaling
<80% success  → Overloaded, excessive retries
```

### Latency Scaling

```
P50 = 150ms   → Typical fast-path commit
P95 = 450ms   → Some retries (2-3 snapshots behind)
P99 = 1200ms  → Heavy conflicts (5+ snapshots behind)
```

### Saturation Point

The **50% success rate threshold** marks system saturation:
- Below this point: Increased load → increased throughput
- At this point: Peak throughput achieved
- Beyond this point: Increased load → decreased throughput (retry overhead)

## Warmup Period Methodology

The simulator automatically applies a **warmup period filter** to exclude initial transient behavior:

```python
warmup_duration = max(5min, min(3 × mean_runtime, 15min))
```

**For baseline experiments** (runtime.mean = 180s):
- **Warmup: 9 minutes** (15% of simulation excluded)
- **Active window: 51 minutes** (85% analyzed for steady-state)

**Why warmup matters:**
- System starts empty (no transactions in flight)
- Early transactions see artificially low contention
- Queue depths need time to stabilize
- Metrics converge after multiple transaction cycles

See [`docs/WARMUP_PERIOD.md`](docs/WARMUP_PERIOD.md) for detailed methodology.

## Common Tasks

### Run Single Experiment with Custom Parameters

```bash
# Copy base config
cp experiment_configs/exp2_1_single_table_false_conflicts.toml my_experiment.toml

# Edit parameters (e.g., change inter_arrival.scale = 100)
# Run with auto-confirm
echo "Y" | python -m icecap.main my_experiment.toml
```

### Compare Different Configurations

```bash
# Catalog-level conflicts (baseline)
python -m icecap.main config_catalog_level.toml

# Table-level conflicts (reduced contention)
python -m icecap.main config_table_level.toml

# Analyze programmatically
python -c "
import pandas as pd

catalog = pd.read_parquet('experiments/catalog-*/*/results.parquet')
table = pd.read_parquet('experiments/table-*/*/results.parquet')

print('Catalog-level:')
print(f'  Success: {(catalog.status==\"committed\").mean()*100:.1f}%')
print(f'  Mean retries: {catalog[catalog.status==\"committed\"].n_retries.mean():.2f}')

print('\\nTable-level:')
print(f'  Success: {(table.status==\"committed\").mean()*100:.1f}%')
print(f'  Mean retries: {table[table.status==\"committed\"].n_retries.mean():.2f}')
"
```

### Analyze Custom Results

```python
import pandas as pd

# Load results with warmup filtering
df = pd.read_parquet('experiments/my_exp-abc123/12345/results.parquet')
warmup_ms = 540000  # 9 minutes
df = df[df['t_submit'] >= warmup_ms]

# Compute key metrics
committed = df[df['status'] == 'committed']
success_rate = len(committed) / len(df) * 100
throughput = len(committed) / ((df['t_submit'].max() - warmup_ms) / 1000)

print(f"Success rate: {success_rate:.1f}%")
print(f"Throughput: {throughput:.2f} commits/sec")
print(f"P50 latency: {committed['commit_latency'].quantile(0.5):.0f}ms")
print(f"P95 latency: {committed['commit_latency'].quantile(0.95):.0f}ms")
print(f"P99 latency: {committed['commit_latency'].quantile(0.99):.0f}ms")
```

## Troubleshooting

**High abort rate:**
- Increase `retry` limit (default: 10)
- Decrease load (increase `inter_arrival.scale`)
- Use table-level conflicts (`num_groups = num_tables`)

**Simulation runs too long:**
- Reduce `duration_ms` for testing (e.g., 60000 for 1 minute)
- Use `--quick` mode for experiments
- Increase `inter_arrival.scale` (fewer transactions)

**Incomplete results (.running.parquet files):**
- Normal after interruption (Ctrl+C)
- Clean up: `find experiments/ -name '.running.parquet' -delete`
- Resume by re-running experiment script

**Unexpected metrics:**
- Check warmup period is appropriate (9 min for baseline)
- Verify multiple seeds are averaged (use `--seeds 3`)
- Ensure simulation duration is sufficient (1 hour for baseline)
- Validate with `python -m icecap.warmup_validation`

## Next Steps

1. **Run baseline experiments**: `./scripts/run_baseline_experiments.sh --quick --seeds 1`
2. **Analyze results**: `python -m icecap.saturation_analysis -i experiments -p "exp2_*" -o plots`
3. **Validate warmup**: `python -m icecap.warmup_validation experiments/exp2_1_*/12345`
4. **Read documentation**:
   - [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) - Common commands
   - [`docs/WARMUP_PERIOD.md`](docs/WARMUP_PERIOD.md) - Steady-state methodology
   - [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md) - Detailed analysis workflow
   - [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) - Research methodology

## Additional Resources

**Testing:**
```bash
pytest tests/ -v  # Run all 63 tests
pytest tests/test_saturation_analysis.py -v  # Test analysis pipeline
```

**Documentation:**
- [`README.md`](README.md) - Complete feature overview
- [`docs/README.md`](docs/README.md) - Documentation index
- [`docs/RUNNING_EXPERIMENTS.md`](docs/RUNNING_EXPERIMENTS.md) - Detailed experiment guide

**Configuration Examples:**
- `experiment_configs/exp2_1_single_table_false_conflicts.toml` - Single table baseline
- `experiment_configs/exp2_2_multi_table_false_conflicts.toml` - Multi-table baseline
- `cfg.toml` - Default configuration with all parameters documented
