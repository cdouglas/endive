# Icecap: Iceberg Catalog Simulator

A discrete-event simulator for exploring commit latency tradeoffs in shared-storage catalog formats like Apache Iceberg. The simulator models optimistic concurrency control (OCC) with compare-and-swap (CAS) operations, conflict resolution, and retry logic.

## 📚 Documentation

Complete documentation is available in the [`docs/`](docs/) directory:

- **[Quick Start Guide](QUICKSTART.md)** - Get started in minutes
- **[Quick Reference](QUICK_REFERENCE.md)** - Common commands and file locations
- **[Running Experiments](docs/RUNNING_EXPERIMENTS.md)** - Execute baseline experiments with parallel execution
- **[Analysis Guide](docs/ANALYSIS_GUIDE.md)** - Analyze results and generate plots
- **[Warmup Period Analysis](docs/WARMUP_PERIOD.md)** - Steady-state measurement methodology
- **[Documentation Index](docs/README.md)** - Complete documentation reference
- **[Research Plan](ANALYSIS_PLAN.md)** - Research methodology and experiment design

## Overview

When multiple writers attempt to commit changes to an Iceberg table simultaneously, conflicts can occur. A failed pointer swap at the catalog requires additional round trips to:

1. Read the manifest file (JSON)
2. Merge the old snapshot and write a new manifest file
3. Read the manifest list
4. Merge the updated manifest list with changes in this transaction
5. For all conflicts in manifest files: merge and rewrite manifest files
6. Retry the pointer swap at the catalog

This simulator helps explore how different parameters (number of concurrent clients, catalog latency, table distribution) affect end-to-end commit latency and success rates.

## Key Features

✅ **Parallel Execution** - Run experiments concurrently (8x faster with 8 cores)
✅ **Warmup Period Analysis** - Steady-state measurement with transient filtering
✅ **Saturation Analysis** - Identify throughput limits and latency scaling
✅ **Experiment Organization** - Deterministic hashing with multi-seed support
✅ **Comprehensive Testing** - 63 tests covering all major functionality
✅ **Interruptible Runs** - Safe Ctrl+C with partial result detection

## Installation

```bash
# Create virtual environment
python3 -m venv .

# Activate virtual environment
source bin/activate  # Linux/Mac
# or
.\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package (for CLI tools)
pip install -e .
```

## Quick Start

### 1. Run a Single Simulation

```bash
# Use default configuration (cfg.toml)
python -m icecap.main cfg.toml

# Use custom configuration
python -m icecap.main my_config.toml

# Skip confirmation prompt (for automation)
echo "Y" | python -m icecap.main my_config.toml
```

**Output:**
- Results written to `.running.parquet` during execution
- Renamed to `results.parquet` on successful completion
- Interrupted runs leave `.running.parquet` for easy cleanup

### 2. Run Baseline Experiments (Parallel)

```bash
# Run all baseline experiments with default parallelism (# of CPU cores)
./scripts/run_baseline_experiments.sh --seeds 3

# Limit to 4 parallel jobs
./scripts/run_baseline_experiments.sh --parallel 4 --seeds 3

# Run in background
nohup ./scripts/run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Quick test (shorter simulations, fewer configs)
./scripts/run_baseline_experiments.sh --quick --seeds 1
```

**Time Estimates:**
- Full baseline (3 seeds): ~24 hours with 8 cores (~8 days sequential)
- Quick test: ~2 minutes

### 3. Analyze Results

```bash
# Generate saturation analysis for single-table experiments
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Generate saturation analysis for multi-table experiments (grouped)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables

# Validate warmup period for an experiment
python -m icecap.warmup_validation \
    experiments/exp2_1_single_table_false-abc123/12345
```

**Outputs:**
- `experiment_index.csv` - Summary statistics for all experiments
- `latency_vs_throughput.png` - P50/P95/P99 latency curves
- `success_rate_vs_throughput.png` - Success rate degradation
- `success_rate_vs_load.png` - Success rate vs offered load

### 4. Verify Results

```bash
# Count completed experiments
find experiments/ -name 'results.parquet' | wc -l

# Check for incomplete runs (interrupted simulations)
find experiments/ -name '.running.parquet'

# Clean up incomplete runs
find experiments/ -name '.running.parquet' -delete
```

## Configuration

The simulator uses TOML configuration files. Key parameters:

### Simulation Parameters

```toml
[simulation]
duration_ms = 3600000        # 1 hour (for stable statistics)
output_path = "results.parquet"
seed = null                  # Random seed (null = random)
```

### Experiment Organization

```toml
[experiment]
# Optional experiment label for organized output
# Output goes to: experiments/$label-$hash/$seed/results.parquet
label = "exp2_1_single_table_false"
```

**Benefits:**
- **Reproducibility**: Same config + code → same hash
- **Multi-seed support**: Multiple runs organized under same experiment
- **Code versioning**: Hash changes when simulator changes

### Catalog Configuration

```toml
[catalog]
num_tables = 10              # Number of tables in catalog
num_groups = 1               # Conflict detection granularity
group_size_distribution = "uniform"  # or "longtail"
```

**Conflict Granularity:**
- `num_groups = 1`: Catalog-level conflicts (all transactions conflict)
- `num_groups = T`: Table-level conflicts (only same-table transactions conflict)
- `1 < num_groups < T`: Group-level isolation (multi-tenant modeling)

### Transaction Parameters

```toml
[transaction]
retry = 10                   # Maximum retries

# Transaction runtime (lognormal distribution)
# Realistic Iceberg transactions: 30s - 3min
runtime.min = 30000          # 30 seconds minimum
runtime.mean = 180000        # 3 minutes mean
runtime.sigma = 1.5

# Inter-arrival time (offered load)
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0  # Mean inter-arrival (ms)

# Conflict types
real_conflict_probability = 0.0  # 0.0 = all false conflicts
```

**False vs Real Conflicts:**
- **False conflicts**: Version changed, no data overlap (~100ms to retry)
- **Real conflicts**: Overlapping data changes, requires manifest merge (~500ms+)

### Storage Latencies

```toml
[storage]
max_parallel = 4             # Parallel manifest operations

# Infinitely fast catalog (baseline)
T_CAS.mean = 1
T_CAS.stddev = 0.1

# Realistic S3 latencies
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
```

## Warmup Period Methodology

The simulator automatically applies a **warmup period filter** during analysis to exclude initial transient behavior:

```python
warmup_duration = max(5min, min(3 × mean_runtime, 15min))
```

**For baseline experiments** (runtime.mean = 180s):
- Warmup: **9 minutes** (15% excluded)
- Active window: **51 minutes** (85% analyzed)

This ensures steady-state measurements by allowing:
- Queue depths to stabilize
- Contention to reach equilibrium
- Latency distributions to converge

**Validation:**
```bash
python -m icecap.warmup_validation experiments/exp2_1_*/12345
```

See [`docs/WARMUP_PERIOD.md`](docs/WARMUP_PERIOD.md) for detailed methodology.

## Analysis Capabilities

### Saturation Analysis

Generates latency vs throughput curves to identify system limits:

```bash
python -m icecap.saturation_analysis -i experiments -p "exp2_*" -o plots
```

**Metrics computed:**
- Achieved throughput (commits/sec)
- Success rate (% committed)
- Commit latency (P50, P95, P99)
- Mean retries per transaction
- Saturation point (50% success threshold)

### Warmup Validation

Visualizes metric convergence over time:

```bash
python -m icecap.warmup_validation experiments/exp2_1_*/12345 --window-size 60
```

**Outputs:**
- Throughput over time
- Success rate over time
- Latency convergence
- Statistical stability analysis

## Project Structure

```
.
├── icecap/                  # Core simulator code
│   ├── main.py              # Simulation engine
│   ├── capstats.py          # Statistics collection
│   ├── saturation_analysis.py  # Saturation analysis
│   └── warmup_validation.py    # Warmup validation
├── scripts/                 # Automation scripts
│   ├── run_baseline_experiments.sh  # Parallel experiment runner
│   └── monitor_experiments.sh       # Progress monitoring
├── experiment_configs/      # Experiment templates
│   ├── exp2_1_single_table_false_conflicts.toml
│   └── exp2_2_multi_table_false_conflicts.toml
├── tests/                   # Test suite (63 tests)
├── docs/                    # Documentation
├── experiments/             # Results (created at runtime)
└── plots/                   # Analysis outputs

```

## Testing

```bash
# Run all tests (63 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=icecap --cov-report=html

# Run specific test module
pytest tests/test_saturation_analysis.py -v
```

**Test coverage:**
- Core simulation (determinism, conflict types)
- Conflict resolution (CAS failures, retries)
- Experiment organization (hashing, directory structure)
- Analysis pipeline (parameter extraction, aggregation)
- Warmup period calculation

## Architecture

### Simulation Engine (SimPy)

- **Discrete-event simulation** using SimPy framework
- **Transaction lifecycle**: Submit → Execute → Commit (with retries)
- **Conflict detection**: CAS failures trigger retry logic
- **Resource modeling**: Catalog operations with configurable latencies

### Experiment Organization

- **Deterministic hashing**: Config + code → 8-char hash
- **Multi-seed support**: Multiple runs under same experiment
- **Atomic writes**: `.running.parquet` → `results.parquet` on completion

### Analysis Pipeline

1. **Scan experiments**: Find all matching experiment directories
2. **Extract parameters**: Parse `cfg.toml` from each experiment
3. **Load results**: Aggregate across seeds with warmup filtering
4. **Compute statistics**: Throughput, latency percentiles, success rates
5. **Generate visualizations**: Latency curves, saturation points

## Research Applications

The simulator supports investigating:

1. **Saturation analysis**: When does throughput plateau?
2. **Latency scaling**: How does P99 latency grow with load?
3. **Multi-table scaling**: Does parallelism reduce contention?
4. **Conflict resolution costs**: Impact of false vs real conflicts
5. **Catalog performance**: Sensitivity to catalog latencies

See [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) for full research methodology.

## Experimental Results

**Baseline experiments** (Phase 2) characterize:
- Single table saturation with false conflicts
- Multi-table scaling (1, 2, 5, 10, 20, 50 tables)
- Load sweep (10ms to 5000ms inter-arrival)
- 3 seeds per configuration for statistical confidence

**Key findings** (example):
- Single table saturates at ~75 commits/sec (50% success)
- Multi-table provides ~linear scaling up to contention threshold
- P99 latency increases exponentially approaching saturation

## Performance

**Simulation speed:**
- ~100,000 transactions in 100 seconds simulated time
- Real-time factor: ~1000x (100s simulation in 0.1s wall-clock)

**Parallel execution:**
- 8 cores: 189 experiments in ~24 hours
- Sequential: Same experiments in ~8 days
- Speedup: ~8x (linear scaling)

## Contributing

Run tests before committing:
```bash
pytest tests/ -v
```

Update documentation for:
- New configuration parameters
- Analysis methods
- Experiment templates

## License

(Add license information here)

## Citation

(Add citation information here)

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **SimPy**: https://simpy.readthedocs.io/
- **Optimistic Concurrency Control**: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
