# Icecap: Iceberg Catalog Simulator

A discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC) that characterizes commit throughput and latency under varying workloads.

## What This Simulates

Apache Iceberg uses optimistic concurrency control with compare-and-swap (CAS) operations. When multiple writers commit simultaneously:

1. Failed CAS triggers conflict resolution
2. **False conflicts** (version changed, no data overlap): read metadata only (~1ms)
3. **Real conflicts** (overlapping changes): read and rewrite manifest files (~400ms+)
4. Transaction retries with optional exponential backoff

The simulator explores: When does throughput saturate? What causes latency to explode?

## Installation

```bash
python3 -m venv .
source bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Run a Single Simulation

```bash
# Run experiment 2.1 (single table, false conflicts)
python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml --yes

# Results saved to: experiments/exp2_1_single_table_false-<hash>/
```

This simulates 1 hour of workload in ~3.6 seconds.

### Command Line Options

```bash
# Run with specific seed
python -m icecap.main my_config.toml --seed 42 --yes

# Run batch experiments (multiple seeds)
./scripts/run_baseline_experiments.sh --seeds 3

# Quick test mode (2 minutes simulation)
./scripts/run_baseline_experiments.sh --quick --seeds 1
```

## Analysis

### Generate Plots

```bash
# Analyze experiment 2.1 results
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Analyze with grouping (e.g., experiment 2.2 by table count)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables
```

This generates:
- `latency_vs_throughput.png` - Latency curves with error bands
- `success_vs_throughput.png` - Success rate degradation
- `overhead_vs_throughput.png` - Commit protocol overhead
- `experiment_index.csv` - Summary statistics

### Regenerate All Plots

```bash
./scripts/regenerate_all_plots.sh
```

## Configuration Example

```toml
[simulation]
duration_ms = 3600000      # 1 hour
output_path = "results.parquet"

[experiment]
label = "exp2_1_single_table_false"

[catalog]
num_tables = 1
num_groups = 1

[transaction]
retry = 10
runtime.mean = 180000      # 3 minutes
runtime.sigma = 1.5
inter_arrival.scale = 500.0  # ~2 txn/sec offered load

# Conflict configuration
real_conflict_probability = 0.0  # 0=false only, 1=real only

[storage]
T_CAS.mean = 1             # Fast catalog
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_FILE.read.mean = 50
```

See [`experiment_configs/`](experiment_configs/) for complete examples.

## Running Tests

```bash
# All tests (~3 minutes)
pytest tests/ -v

# Specific test suites
pytest tests/test_simulator.py -v                    # Core simulator
pytest tests/test_numerical_accuracy.py -v           # Numerical validation
pytest tests/test_statistical_rigor.py -v            # Distribution conformance

# With coverage
pytest tests/ --cov=icecap --cov-report=html

# Fast subset
pytest tests/test_simulator.py tests/test_conflict_resolution.py -v
```

## Documentation

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Detailed getting started guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Simulator design and invariants
- **[docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md)** - Plot generation and interpretation
- **[experiment_configs/README.md](experiment_configs/README.md)** - Experiment descriptions
- **[docs/](docs/)** - Complete documentation index

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **SimPy Framework**: https://simpy.readthedocs.io/
