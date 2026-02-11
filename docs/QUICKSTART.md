# Quick Start Guide

Get started with the Iceberg Catalog Simulator in minutes.

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Verify installation
python -m endive.main --help
pytest tests/ -v  # Optional: run 63 tests
```

## Understanding the Simulator

Endive models Apache Iceberg's optimistic concurrency control with:

- **Compare-and-swap (CAS)** for atomic catalog updates
- **Conflict detection** when catalog advances during transaction execution
- **Retry logic** with manifest list reading and merging
- **False conflicts**: Version changed, no data overlap (~1ms to resolve)
- **Real conflicts**: Overlapping data changes (~400ms+ to resolve manifest files)
- **Configurable storage latencies** using realistic distributions
- **Parallel manifest operations** with configurable limits

## Your First Simulation

### Run Single Experiment

```bash
# Use pre-configured experiment
echo "Y" | python -m endive.main experiment_configs/exp2_1_single_table_false_conflicts.toml

# The simulator will:
# 1. Display configuration summary
# 2. Auto-confirm with "Y"
# 3. Show progress bar
# 4. Export to experiments/exp2_1_single_table_false-<hash>/<seed>/results.parquet
# 5. Display summary statistics
```

**Example output:**
```
Simulation Summary:
  Total transactions: 18,247
  Committed: 18,021 (98.8%)
  Aborted: 226 (1.2%)

  Commit latency (ms):
    Mean: 1,245
    P50: 892
    P95: 3,104
    P99: 5,892

  Retries per transaction:
    Mean: 0.9
    Max: 8
```

**Note**: Results written to `.running.parquet` during execution, renamed to `results.parquet` on completion.

## Quick Test (2 minutes)

Test the full pipeline with shortened simulations:

```bash
./scripts/run_baseline_experiments.sh --quick --seeds 1

# This runs:
# - 9 single-table configs (varying load)
# - 54 multi-table configs (6 table counts × 9 load levels)
# - Duration: 10 seconds per simulation (vs 1 hour for full)
# - Total: ~2 minutes
```

## Run Baseline Experiments (24 hours)

Full baseline characterizes saturation across all load levels:

```bash
# Run all baseline experiments with parallel execution
./scripts/run_baseline_experiments.sh --seeds 3

# Or run in background with logging
nohup ./scripts/run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Monitor progress (refreshes every 5 seconds)
./scripts/monitor_experiments.sh --watch 5
```

**What runs:**
- **Experiment 2.1**: Single table, false conflicts (9 load levels × 3 seeds = 27 runs)
- **Experiment 2.2**: Multi-table scaling (6 table counts × 9 loads × 3 seeds = 162 runs)
- **Total**: 189 simulations @ 1 hour each

**Parallel execution:**
- Default: Uses all CPU cores (e.g., 8 cores = 8 parallel jobs)
- Custom: `--parallel 4` limits to 4 concurrent experiments
- Speedup: ~8× with 8 cores (24 hours vs 8 days sequential)

## Analyze Results

### Generate Saturation Curves

```bash
# Single-table experiments
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Multi-table experiments (grouped by table count)
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables

# With custom configuration
python -m endive.saturation_analysis \
    --config analysis.toml \
    -i experiments \
    -p "exp2_*" \
    -o plots

# View results
open plots/exp2_1/latency_vs_throughput.png
open plots/exp2_1/success_rate_vs_throughput.png
cat plots/exp2_1/experiment_index.csv
```

### Analysis Configuration

Customize analysis via `analysis.toml`:

```toml
# Control saturation annotations
[plots.saturation]
enabled = true      # Show/hide saturation markers
threshold = 50.0    # Success rate threshold (%)
tolerance = 5.0     # Detection tolerance (%)

# Customize plot appearance
[plots]
dpi = 300
[plots.figsize]
latency_vs_throughput = [12, 8]

# Configure warmup/cooldown
[analysis]
k_min_cycles = 5
min_warmup_ms = 300000
max_warmup_ms = 900000
```

See `analysis.toml` and `example_saturation_config.toml` for complete options.

**Generated files:**
- `experiment_index.csv` - Summary statistics (markdown table also generated)
- `latency_vs_throughput.{png,md}` - P50/P95/P99 curves with saturation point
- `success_rate_vs_throughput.{png,md}` - Transaction success rate degradation
- `success_rate_vs_load.{png,md}` - Success vs offered load (inter-arrival time)
- `overhead_vs_throughput.{png,md}` - Commit protocol overhead percentage

**Analysis features:**
- **Warmup filtering**: Excludes first 9 minutes (transient period)
- **Multi-seed aggregation**: Combines results across random seeds
- **Saturation detection**: Marks 50% success rate threshold
- **Overhead computation**: `(commit_latency / total_latency) × 100`

### Validate Warmup Period

Visualize metric convergence to verify steady-state:

```bash
python -m endive.warmup_validation \
    experiments/exp2_1_single_table_false-abc123/12345

# Custom window size
python -m endive.warmup_validation \
    experiments/exp2_1_*/12345 \
    --window-size 30
```

**Outputs:**
- Throughput over time (stabilization after warmup)
- Success rate convergence
- Mean and P95 latency trends
- Statistical stability (coefficient of variation)

## Configuration

### Key Parameters

Located in `experiment_configs/*.toml`:

```toml
[simulation]
duration_ms = 3600000        # 1 hour (stable statistics)

[experiment]
label = "exp2_1_single_table_false"  # Output organization

[catalog]
num_tables = 1               # Number of tables
num_groups = 1               # Conflict granularity

[transaction]
retry = 10                   # Maximum retries

# Transaction runtime (lognormal distribution)
runtime.min = 30000          # 30 seconds minimum
runtime.mean = 180000        # 3 minutes mean
runtime.sigma = 1.5

# Offered load
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0  # Mean inter-arrival (ms)

# Conflict types
real_conflict_probability = 0.0  # 0.0=false, 1.0=real

# For real conflicts
conflicting_manifests.distribution = "exponential"
conflicting_manifests.mean = 3.0

[storage]
max_parallel = 4             # Parallel manifest operations

# Infinitely fast catalog (baseline)
T_CAS.mean = 1
T_METADATA_ROOT.read.mean = 1

# Realistic S3 latencies
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_FILE.read.mean = 50
```

### Conflict Granularity

Control how transactions detect conflicts:

- **`num_groups = 1`**: Catalog-level (all transactions conflict)
- **`num_groups = T`**: Table-level (only same-table transactions conflict)
- **`1 < num_groups < T`**: Group-level (multi-tenant scenarios)

### False vs Real Conflicts

- **False conflicts** (`real_conflict_probability = 0.0`):
  - Catalog version changed, no data overlap
  - Only metadata read required
  - Fast retry (~1ms with infinitely fast catalog)
  - **Cost**: Read metadata root

- **Real conflicts** (`real_conflict_probability = 1.0`):
  - Overlapping data file changes
  - Must read and rewrite manifest files
  - Expensive retry (~400ms+)
  - **Cost**: Read manifest list + read/write N manifest files (N sampled from distribution)

## Interpret Results

### Success Rate Guidelines

| Success Rate | Status | Action |
|-------------|--------|---------|
| >95% | Under-loaded | Room to increase throughput |
| 90-95% | Healthy | Some contention, good balance |
| 80-90% | High contention | Consider scaling out |
| <80% | Overloaded | Excessive retries, reduce load |

### Latency Indicators

| P99 Latency | Interpretation |
|------------|----------------|
| ~1s | Fast-path commits, minimal conflicts |
| ~10s | Moderate conflicts (2-3 snapshots behind) |
| ~60s | Heavy conflicts (5+ snapshots behind) |
| ~300s+ | Saturation zone, cascading retries |

### Saturation Point

**50% success rate threshold** marks system saturation:
- **Below**: Increased load → increased throughput
- **At**: Peak throughput achieved
- **Beyond**: Increased load → decreased throughput (retry overhead dominates)

## Warmup Period Methodology

The simulator applies a **warmup period filter** to exclude initial transient behavior:

```python
warmup_duration = max(5min, min(3 × mean_runtime, 15min))
```

**For baseline experiments** (runtime.mean = 180s = 3 minutes):
- **Warmup**: 9 minutes (3 × 3 min)
- **Active window**: 51 minutes (85% of 1-hour simulation)

**Why warmup matters:**
- System starts empty (no transactions in flight)
- Early transactions see artificially low contention
- Queue depths stabilize after ~3 transaction cycles
- Metrics converge to steady-state

See [`WARMUP_PERIOD.md`](WARMUP_PERIOD.md) for detailed methodology.

## Verify and Cleanup

```bash
# Count completed experiments
find experiments/ -name 'results.parquet' | wc -l

# Check for incomplete runs
find experiments/ -name '.running.parquet'

# Clean up incomplete runs
find experiments/ -name '.running.parquet' -delete

# Check experiment completion
ls experiments/exp2_1_*/*/results.parquet | wc -l  # Should be 45 (9 configs × 5 seeds)
```

## Common Tasks

### Custom Single Experiment

```bash
# Copy base config
cp experiment_configs/exp2_1_single_table_false_conflicts.toml my_test.toml

# Edit parameters (e.g., inter_arrival.scale = 100)
# Run
echo "Y" | python -m endive.main my_test.toml
```

### Sweep Parameters

```bash
# Sweep inter-arrival time
for scale in 10 20 50 100 200 500 1000; do
    sed "s/inter_arrival.scale = .*/inter_arrival.scale = $scale/" \
        experiment_configs/exp2_1_single_table_false_conflicts.toml > temp.toml
    echo "Y" | python -m endive.main temp.toml
done
rm temp.toml
```

### Analyze Custom Results

```python
import pandas as pd

# Load with warmup filtering
df = pd.read_parquet('experiments/my_exp-abc123/12345/results.parquet')
warmup_ms = 540000  # 9 minutes
df = df[df['t_submit'] >= warmup_ms]

# Compute key metrics
committed = df[df['status'] == 'committed']
success_rate = len(committed) / len(df) * 100
duration_sec = (df['t_submit'].max() - warmup_ms) / 1000
throughput = len(committed) / duration_sec

print(f"Success rate: {success_rate:.1f}%")
print(f"Throughput: {throughput:.2f} commits/sec")
print(f"P50 latency: {committed['commit_latency'].quantile(0.5):.0f}ms")
print(f"P95 latency: {committed['commit_latency'].quantile(0.95):.0f}ms")
print(f"P99 latency: {committed['commit_latency'].quantile(0.99):.0f}ms")
print(f"Mean retries: {committed['n_retries'].mean():.2f}")
```

## Troubleshooting

### High Abort Rate

**Symptoms**: Success rate < 80%, mean retries > 8

**Solutions:**
- Increase `retry` limit (e.g., 15 or 20)
- Decrease offered load (increase `inter_arrival.scale`)
- Use table-level conflicts (`num_groups = num_tables`)
- Check if approaching saturation threshold

### Simulation Takes Too Long

**Solutions:**
- Use `--quick` mode for testing (10s vs 1 hour per simulation)
- Reduce `duration_ms` temporarily (e.g., 60000 for 1 minute)
- Increase `inter_arrival.scale` (fewer transactions generated)

### Incomplete Results

**Symptoms**: `.running.parquet` files remain after simulation

**Cause**: Normal after interruption (Ctrl+C)

**Solutions:**
```bash
# Clean up and re-run
find experiments/ -name '.running.parquet' -delete
./scripts/run_baseline_experiments.sh --seeds 3
```

### Unexpected Metrics

**Check:**
- Warmup period appropriate (9 min for baseline, longer for longer transactions)
- Multiple seeds averaged (`--seeds 3` minimum)
- Simulation duration sufficient (1 hour for stable statistics)
- Validate with: `python -m endive.warmup_validation experiments/exp2_1_*/12345`

### Analysis Errors

**Common issues:**
- No matching experiments: Check pattern matches directory names
- Missing cfg.toml: Experiments directory corrupted
- Empty plots: No committed transactions (100% abort rate, reduce load)

## Quick Reference

### Common Commands

```bash
# Run baseline experiments
./scripts/run_baseline_experiments.sh --seeds 3

# Quick test (2 min)
./scripts/run_baseline_experiments.sh --quick --seeds 1

# Analyze single-table
python -m endive.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1

# Analyze multi-table
python -m endive.saturation_analysis -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables

# Validate warmup
python -m endive.warmup_validation experiments/exp2_1_*/12345

# Run tests
pytest tests/ -v
```

### File Locations

| What | Where |
|------|-------|
| Documentation | `docs/` |
| Experiment configs | `experiment_configs/` |
| Automation scripts | `scripts/` |
| Results | `experiments/` |
| Plots & analysis | `plots/` |
| Tests | `tests/` |
| Core simulator | `endive/main.py` |

### Key Metrics

| Metric | Command/Location |
|--------|------------------|
| Success rate | From `experiment_index.csv` or parquet analysis |
| Throughput | Commits/sec in `experiment_index.csv` |
| Latency percentiles | P50/P95/P99 in `latency_vs_throughput.png` |
| Overhead % | `overhead_vs_throughput.md` table |
| Retries | Mean retries in `experiment_index.csv` |

## Next Steps

1. **Run baseline**: `./scripts/run_baseline_experiments.sh --quick --seeds 1`
2. **Analyze**: `python -m endive.saturation_analysis -i experiments -p "exp2_*" -o plots`
3. **Review results**: `open plots/exp2_1/latency_vs_throughput.png`
4. **Read detailed findings**: [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md)
5. **Understand methodology**: [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)
6. **Run real conflict experiments**: Edit `exp3_1` configs and run Phase 3

## Additional Resources

### Documentation
- [`README.md`](../README.md) - Project overview with key findings
- [`docs/README.md`](README.md) - Complete documentation index
- [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md) - Detailed experimental results
- [`OVERHEAD_ANALYSIS.md`](OVERHEAD_ANALYSIS.md) - Commit overhead breakdown
- [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) - Research methodology
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - Simulator design
- [`DOCKER.md`](DOCKER.md) - Container-based execution

### Testing
```bash
pytest tests/ -v                              # All 63 tests
pytest tests/ --cov=endive --cov-report=html  # With coverage
pytest tests/test_saturation_analysis.py -v   # Analysis tests only
```

### Example Configs
- `experiment_configs/exp2_1_single_table_false_conflicts.toml` - Single table baseline
- `experiment_configs/exp2_2_multi_table_false_conflicts.toml` - Multi-table baseline
- `experiment_configs/exp3_1_single_table_real_conflicts.toml` - Real conflicts (ready to run)
- `cfg.toml` - Default configuration with all parameters
