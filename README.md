# Endive: Iceberg Catalog Simulator

Discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC). Models catalog contention, conflict resolution, and commit latency under varying workloads to answer: **When does commit throughput saturate? What causes latency to explode?**

## What Endive Simulates

Iceberg installs new versions of tables using a compare-and-set (CAS) in its Catalog. When CAS fails but the transaction commutes with the new state, it merges its metadata with concurrent transactions before retrying.

The simulator models:
- **CAS-based commits**: Traditional catalog pointer updates
- **Append-based commits**: Conditional append to catalog log (ML+)
- **Conflict resolution**: False conflicts (no data overlap) vs real conflicts (manifest file I/O)
- **Cloud storage latencies**: Provider-specific distributions from YCSB measurements

Latency parameters are calibrated from:
- **CAS/Append operations**: June 2025 YCSB benchmarks (AWS, Azure, GCP)
- **PUT/GET operations**: Durner et al. VLDB 2023 measurements

## Quick Start

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run single simulation (1 hour simulated, ~4 seconds wall-clock)
python -m endive.main experiment_configs/exp8_0_baseline_s3x.toml --yes

# Run tests
pytest tests/ -v
```

## Analysis

```bash
# Generate saturation curves
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp8_*_s3x*" \
    -o plots/exp8_s3x

# With grouping (compare configurations)
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp8_*" \
    -o plots/exp8 \
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
label = "exp8_0_baseline"

[catalog]
num_tables = 10
num_groups = 1             # Catalog-level contention
table_metadata_inlined = true

[transaction]
retry = 10
runtime.mean = 180000      # 3 minutes
inter_arrival.scale = 100.0
manifest_list_mode = "rewrite"  # or "append" for ML+

[storage]
provider = "s3x"           # s3, s3x, azure, azurex, gcp
max_parallel = 4
```

See `experiment_configs/` for complete examples.

## Experiment Configurations

| Label | Description |
|-------|-------------|
| exp8_0_baseline | CAS catalog, full metadata write |
| exp8_1_metadata_inlining | Table metadata inlined in CAS |
| exp8_3_ml_append | Manifest list append mode |
| exp8_5_combined | Both optimizations |

Providers: `s3x` (S3 Express), `azure` (Azure Standard), `azurex` (Azure Premium)

## Storage Provider Latencies

| Provider | CAS Median | PUT Base | PUT Rate |
|----------|------------|----------|----------|
| S3 Express | 22ms | 10ms | 10 ms/MiB |
| Azure Std | 87ms | 50ms | 25 ms/MiB |
| Azure Premium | 64ms | 30ms | 15 ms/MiB |
| S3 Standard | 23ms | 30ms | 20 ms/MiB |

*Sources: YCSB June 2025, Durner et al. VLDB 2023*

## Tests

```bash
pytest tests/ -v                            # All tests (~3 min)
pytest tests/test_simulator.py -v           # Core simulator
pytest tests/test_append_catalog.py -v      # Append mode
pytest tests/test_statistical_rigor.py -v   # Distribution tests
```

## Documentation

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Getting started
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Design and invariants
- **[docs/model.md](docs/model.md)** - Model simplifications
- **[docs/errata.md](docs/errata.md)** - Technical debt and gaps
- **[experiment_configs/README.md](experiment_configs/README.md)** - Experiment descriptions

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **Durner et al. VLDB 2023**: "Exploiting Cloud Object Storage for High-Performance Analytics"
- **SimPy Framework**: https://simpy.readthedocs.io/
