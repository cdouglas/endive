# Endive: Iceberg Catalog Simulator

One-off, deterministic discrete-event simulator for Apache Iceberg workloads.
Endive simulates workloads of spurious conflicts, measuring commit throughput
and latency under varying conditions. Object store latency, etc. are drawn from
distributions.

## What Endive Simulates

Iceberg installs new versions of tables using a compare-and-set (CAS) in its
Catalog. If the CAS fails but the transaction commutes with the new state, it
will merge its metadata with the concurrent transaction(s) before retrying.

A lot of attention is paid to the root of each table (i.e., the
[Catalog](https://iceberg.apache.org/rest-catalog-spec/)) and formats at the
leaves (e.g., Apaches [ORC](https://orc.apache.org) and
[Parquet](https://parquet.apache.org), [Vortex](https://vortex.dev/),
[F3](https://dl.acm.org/doi/abs/10.1145/3749163))),
[AnyBlox](https://dl.acm.org/doi/10.14778/3749646.3749672), but the metadata
near the root of the table can also be a bottleneck even when there is no real
conflict and the catalog is unreasonably fast.

There are simpler ways to model these conflicts, but it was honestly fun to
develop using Claude Code and parallelize simulation runs.

## Quick Start

### Run a Single Simulation

```bash
# Run experiment 2.1 (single table, false conflicts)
python -m endive.main experiment_configs/exp2_1_single_table_false_conflicts.toml --yes

# Results saved to: experiments/exp2_1_single_table_false-<hash>/
```

## Analysis

### Generate Plots

```bash
# Analyze experiment 2.1 results
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Analyze with grouping (e.g., experiment 2.2 by table count)
python -m endive.saturation_analysis \
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
pytest tests/ --cov=endive --cov-report=html

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
