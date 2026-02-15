# Endive: Iceberg Catalog Simulator

Discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC). Models catalog contention, conflict resolution, and commit latency under varying workloads to answer: **When does commit throughput saturate? What causes latency to explode?**

## What Endive Simulates

Iceberg installs new versions of tables using a compare-and-set (CAS) in its Catalog. When CAS fails but the transaction commutes with the new state, it merges its metadata with concurrent transactions before retrying.

The simulator models:
- **CAS-based commits**: Traditional catalog pointer updates
- **Append-based commits**: Conditional append to catalog log (ML+)
- **Conflict resolution**: False conflicts (no data overlap) vs real conflicts (manifest file I/O)
- **Partition-level conflicts**: Per-partition manifest lists with Zipf/uniform access patterns
- **Cloud storage latencies**: Provider-specific distributions from YCSB measurements

Latency parameters are calibrated from:
- **CAS/Append operations**: June 2025 YCSB benchmarks (AWS, Azure, GCP)
- **PUT/GET operations**: Durner et al. VLDB 2023 measurements

## Quick Start

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run single simulation (1 hour simulated, ~4 seconds wall-clock)
python -m endive.main experiment_configs/baseline_s3x.toml --yes

# Run tests
pytest tests/ -v
```

## Running Experiments

```bash
# Run all experiments (parallel, with progress bar)
./scripts/run_all_experiments.sh --parallel 4 --seeds 3

# Quick test mode (1 minute simulations)
./scripts/run_all_experiments.sh --quick --parallel 4

# Run specific groups
./scripts/run_all_experiments.sh --groups baseline,metadata --seeds 3
```

## Analysis

```bash
# Generate saturation curves
python -m endive.saturation_analysis \
    -i experiments \
    -p "baseline_s3x-*" \
    -o plots/baseline_s3x

# Compare configurations across providers
python -m endive.saturation_analysis \
    -i experiments \
    -p "*_s3x-*" \
    -o plots/s3x_comparison \
    --group-by label
```

Outputs:
- `latency_vs_throughput.png` - Latency curves with error bands
- `success_vs_throughput.png` - Success rate degradation
- `experiment_index.csv` - Summary statistics

## Configuration

```toml
[simulation]
duration_ms = 3600000      # 1 hour

[experiment]
label = "baseline_s3x"

[catalog]
num_tables = 1
table_metadata_inlined = false  # true = inlining optimization

[transaction]
retry = 10
runtime.mean = 180000      # 3 minutes
inter_arrival.scale = 100.0
manifest_list_mode = "rewrite"  # or "append" for ML+

[storage]
provider = "s3x"           # s3, s3x, azure, azurex, gcp
max_parallel = 4

[partition]                # Optional: partition-level modeling
enabled = true
num_partitions = 100
partitions_per_txn_mean = 3.0
selection.distribution = "zipf"  # or "uniform"
selection.zipf_alpha = 1.5
```

See `experiment_configs/` for complete examples.

## Experiment Design

**Factorial design for optimization experiments:**

| Config | table_metadata_inlined | manifest_list_mode | Effect |
|--------|:----------------------:|:------------------:|--------|
| baseline | false | rewrite | No optimizations (control) |
| metadata | true | rewrite | Inlining effect only |
| ml_append | false | append | ML+ effect only |
| combined | true | append | Both optimizations |

**Provider coverage:**

| Config | S3 | S3 Express | Azure | Azure Premium |
|--------|:--:|:----------:|:-----:|:-------------:|
| baseline | ✓ | ✓ | ✓ | ✓ |
| metadata | ✓ | ✓ | ✓ | ✓ |
| ml_append | - | ✓ | ✓ | ✓ |
| combined | - | ✓ | ✓ | ✓ |

*S3 Standard excluded from append experiments (no conditional append support).*

## Storage Provider Latencies

| Provider | CAS Median | Append | PUT Base | PUT Rate |
|----------|------------|--------|----------|----------|
| S3 Standard | 61ms | N/A | 30ms | 20 ms/MiB |
| S3 Express | 22ms | 21ms | 10ms | 10 ms/MiB |
| Azure Std | 93ms | 87ms | 50ms | 25 ms/MiB |
| Azure Premium | 64ms | 70ms | 30ms | 15 ms/MiB |

*Sources: YCSB June 2025, Durner et al. VLDB 2023*

## Tests

```bash
pytest tests/ -v                              # All tests (~3 min)
pytest tests/test_simulator.py -v             # Core simulator
pytest tests/test_experiment_runner.py -v     # Experiment runner
pytest tests/test_storage_provider_config.py  # Provider configs
```

## Documentation

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Getting started
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Design and invariants
- **[docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md)** - Running experiments
- **[experiment_configs/README.md](experiment_configs/README.md)** - Experiment descriptions

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **Durner et al. VLDB 2023**: "Exploiting Cloud Object Storage for High-Performance Analytics"
- **SimPy Framework**: https://simpy.readthedocs.io/
